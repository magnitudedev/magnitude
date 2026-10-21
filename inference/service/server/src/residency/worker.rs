//! The engine worker process (integration spec §8.1, §10): the service executable's
//! `inference-worker` role runs the engine's worker over its standard streams. The service owns
//! the spawn, the framed transport, crash handling and the proof that the process has retired.

use std::collections::VecDeque;
use std::io::Read;
use std::process::ExitStatus;
use std::sync::{Arc, Mutex, OnceLock};

use magnitude_engine::error::{LoadError, UnloadCause};
use magnitude_engine::options::ExecutionManifest;
use magnitude_engine::worker::protocol::LoadProgress;
use magnitude_engine::worker::transport::{FramedTransport, stdio_worker_transport};
use magnitude_engine::LoadAttempt;
use magnitude_engine::worker::{
    EngineClient, WorkerConnection, WorkerExit, WorkerProcess, WorkerReport, connect_worker,
    serve_worker,
};
use magnitude_executor::UnpreparedUnits;

use super::ResidencyWorker;
use super::controller::GRACEFUL_EXIT;
use crate::device_lock::DeviceHold;
use crate::worker_process::{BlockingChild, BlockingProcess, WorkerLauncher, WorkerRole};

/// The most recent worker diagnostics kept for failure reports.
const DIAGNOSTIC_TAIL_BYTES: usize = 16 * 1024;

/// How the `inference-worker` process reports its end to the service. Crash handling reads
/// only whether the worker is gone; the code is diagnostic.
pub fn worker_exit_code(exit: &WorkerExit) -> i32 {
    match exit {
        WorkerExit::Shutdown
        | WorkerExit::Prepared
        | WorkerExit::Measured
        | WorkerExit::Unprepared
        | WorkerExit::HostLost { .. } => 0,
        WorkerExit::Unloaded(UnloadCause::MemoryPressure) => 10,
        WorkerExit::Unloaded(_) => 11,
        WorkerExit::LoadFailed(_) => 12,
        WorkerExit::BuildMismatch { .. } => 13,
        WorkerExit::ProtocolViolation(_) => 14,
        WorkerExit::TransportFailed(_) => 15,
    }
}

/// The `inference-worker` process entry: serve one load over standard streams until shutdown,
/// host loss or unload.
pub fn run_inference_worker() -> i32 {
    let exit = serve_worker(stdio_worker_transport());
    let code = worker_exit_code(&exit);
    if code != 0 {
        eprintln!("inference worker ended: {exit:?}");
    }
    code
}

/// How long a load failure waits for an ended worker's exit status.
const EXIT_OBSERVATION: std::time::Duration = std::time::Duration::from_secs(1);

/// Record a worker's exit once, when its retirement is first observed: a crash (on Windows an
/// NTSTATUS such as 0xc0000005) is otherwise visible nowhere.
fn log_exit(pid: u32, status: ExitStatus) {
    if status.success() {
        tracing::info!(worker.pid = pid, %status, "inference worker exited");
    } else {
        tracing::warn!(worker.pid = pid, %status, "inference worker exited");
    }
}

/// A worker process, and the device hold of the load or optimization that started it: the
/// device is held until the process is proven retired.
enum ProcessState {
    Owned {
        process: BlockingProcess,
        /// Held, never read: dropping it with the owner releases the device.
        _device: DeviceHold,
    },
    Retired(ExitStatus),
}

/// The bounded tail of a worker's standard error.
#[derive(Default)]
struct DiagnosticTail(VecDeque<u8>);

impl DiagnosticTail {
    fn append(&mut self, bytes: &[u8]) {
        self.0.extend(bytes);
        let excess = self.0.len().saturating_sub(DIAGNOSTIC_TAIL_BYTES);
        self.0.drain(..excess);
    }

    fn text(&self) -> String {
        let (front, back) = self.0.as_slices();
        let mut bytes = Vec::with_capacity(self.0.len());
        bytes.extend_from_slice(front);
        bytes.extend_from_slice(back);
        String::from_utf8_lossy(&bytes).trim().to_owned()
    }
}

/// One engine worker process. It owns the process until retirement is proven, and the engine
/// client once the worker is ready.
pub struct EngineWorker {
    process: Mutex<ProcessState>,
    client: OnceLock<EngineClient>,
    diagnostics: Arc<Mutex<DiagnosticTail>>,
}

/// The host end of an inference worker process's transport.
pub type HostEnd = FramedTransport<
    std::io::BufReader<crate::worker_process::BlockingStdout>,
    crate::worker_process::BlockingStdin,
    magnitude_engine::worker::protocol::WorkerMessage,
    magnitude_engine::worker::protocol::HostMessage,
>;

/// A spawned worker and the host end of its transport, before the load handshake.
pub struct SpawnedWorker {
    pub worker: Arc<EngineWorker>,
    transport: HostEnd,
}

impl EngineWorker {
    /// Spawn an `inference-worker` process for the load or optimization holding `device`.
    pub fn spawn(
        launcher: &WorkerLauncher,
        device: DeviceHold,
    ) -> anyhow::Result<SpawnedWorker> {
        let command = launcher.command(WorkerRole::Inference)?;
        let BlockingChild {
            process,
            stdin,
            stdout,
            stderr,
        } = launcher.spawn_blocking_io(command)?;
        let pid = process_id(&process);
        let worker = Arc::new(Self {
            process: Mutex::new(ProcessState::Owned {
                process,
                _device: device,
            }),
            client: OnceLock::new(),
            diagnostics: Arc::new(Mutex::new(DiagnosticTail::default())),
        });
        drain_diagnostics(stderr, Arc::clone(&worker.diagnostics), pid)?;
        Ok(SpawnedWorker {
            worker,
            transport: FramedTransport::new(std::io::BufReader::new(stdout), stdin),
        })
    }

    /// The worker's recent standard error, for failure reports.
    pub fn diagnostics(&self) -> String {
        self.diagnostics
            .lock()
            .expect("diagnostic tail lock")
            .text()
    }

    /// Prove retired a worker that reported its outcome and is exiting on its own
    /// ([`reported_exit`]), stopping it only if it has not exited in time.
    fn retire_reported(&self) -> std::io::Result<ExitStatus> {
        if let Some(status) = reported_exit(|| self.try_wait())? {
            return Ok(status);
        }
        tracing::warn!(
            worker.pid = self.pid(),
            "inference worker reported its outcome and did not exit; stopping it"
        );
        self.retire()
    }

    /// The exit status of a worker whose protocol stream ended, observed for up to
    /// [`EXIT_OBSERVATION`]: its stream closes just before the process is reaped. `None` while it
    /// is still running or cannot be observed.
    pub fn ended_status(&self) -> Option<ExitStatus> {
        let started = std::time::Instant::now();
        loop {
            match self.try_wait() {
                Ok(Some(status)) => return Some(status),
                Ok(None) if started.elapsed() < EXIT_OBSERVATION => {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
                Ok(None) | Err(_) => return None,
            }
        }
    }

    fn retire(&self) -> std::io::Result<ExitStatus> {
        let mut state = self.process.lock().expect("worker process lock");
        match &mut *state {
            ProcessState::Retired(status) => Ok(*status),
            ProcessState::Owned { process, .. } => {
                // An error leaves the exact owner in place. Only proof of retirement replaces it.
                let pid = process_id(process);
                let status = stop(process)?;
                log_exit(pid, status);
                *state = ProcessState::Retired(status);
                Ok(status)
            }
        }
    }
}

impl SpawnedWorker {
    /// One attempt to load: send `Hello` and `Load{manifest, unprepared}` and wait for
    /// readiness, reporting load progress. Blocking: the worker reports its progress until it
    /// is ready or fails. A worker asked to report tuning units without a stored record that
    /// met one placed nothing; it is proven retired before `Unprepared` is returned.
    pub fn connect(
        self,
        manifest: ExecutionManifest,
        unprepared: UnpreparedUnits,
        progress: impl FnMut(LoadProgress),
    ) -> Result<LoadAttempt<(Arc<EngineWorker>, WorkerConnection)>, (Arc<EngineWorker>, LoadError)>
    {
        match connect_worker(self.transport, Some((manifest, unprepared)), progress) {
            Ok(LoadAttempt::Loaded(connection)) => {
                self.worker
                    .client
                    .set(connection.client.clone())
                    .unwrap_or_else(|_| unreachable!("a worker connects once"));
                Ok(LoadAttempt::Loaded((self.worker, connection)))
            }
            Ok(LoadAttempt::Unprepared) => match self.worker.retire_reported() {
                Ok(_) => Ok(LoadAttempt::Unprepared),
                Err(error) => Err((
                    self.worker,
                    LoadError::WorkerLost {
                        reason: format!(
                            "the unprepared Load worker could not be shown to have ended: {error}"
                        ),
                    },
                )),
            },
            Err(error) => Err((self.worker, error)),
        }
    }

    /// This worker as the process of a model's Prepare (`magnitude_engine::worker::prepare`,
    /// `load_prepared`): its transport, and the proof of its retirement. A Prepare that
    /// reported its outcome is left to exit; one that reported nothing is stopped.
    pub fn into_process(
        self,
    ) -> WorkerProcess<HostEnd, impl FnOnce(WorkerReport) -> Result<(), String>> {
        let Self { worker, transport } = self;
        WorkerProcess {
            transport,
            retire: move |report| {
                match report {
                    WorkerReport::Reported => worker.retire_reported(),
                    WorkerReport::Unreported => worker.retire(),
                }
                .map(drop)
                .map_err(|error| error.to_string())
            },
        }
    }
}

/// An inference worker process run for one job outside residency (the measurement job): it
/// holds no device share, and its owner retires it.
pub struct JobWorker {
    process: BlockingProcess,
    diagnostics: Arc<Mutex<DiagnosticTail>>,
}

impl JobWorker {
    /// Spawn an `inference-worker` process, with the host end of its transport.
    pub fn spawn(launcher: &WorkerLauncher) -> anyhow::Result<(Self, HostEnd)> {
        let command = launcher.command(WorkerRole::Inference)?;
        let BlockingChild {
            process,
            stdin,
            stdout,
            stderr,
        } = launcher.spawn_blocking_io(command)?;
        let pid = process_id(&process);
        let worker = Self {
            process,
            diagnostics: Arc::new(Mutex::new(DiagnosticTail::default())),
        };
        if let Err(error) = drain_diagnostics(stderr, Arc::clone(&worker.diagnostics), pid) {
            worker.retire(false)?;
            return Err(error.into());
        }
        let transport = FramedTransport::new(std::io::BufReader::new(stdout), stdin);
        Ok((worker, transport))
    }

    pub fn pid(&self) -> u32 {
        process_id(&self.process)
    }

    /// The worker's recent standard error, for failure reports.
    pub fn diagnostics(&self) -> String {
        self.diagnostics
            .lock()
            .expect("diagnostic tail lock")
            .text()
    }

    /// Return once the process is proven gone. One that `reported` its outcome is left to exit
    /// on its own ([`reported_exit`]); any other, or one that has not exited by then, is
    /// stopped. Blocking.
    pub fn retire(mut self, reported: bool) -> std::io::Result<ExitStatus> {
        let pid = process_id(&self.process);
        let exited = if reported {
            reported_exit(|| observe_exit(&mut self.process))?
        } else {
            None
        };
        let status = match exited {
            Some(status) => status,
            None => stop(&mut self.process)?,
        };
        log_exit(pid, status);
        Ok(status)
    }
}

impl ResidencyWorker for EngineWorker {
    fn pid(&self) -> Option<u32> {
        match &*self.process.lock().expect("worker process lock") {
            ProcessState::Owned { process, .. } => Some(process_id(process)),
            ProcessState::Retired(_) => None,
        }
    }

    fn try_wait(&self) -> std::io::Result<Option<ExitStatus>> {
        let mut state = self.process.lock().expect("worker process lock");
        match &mut *state {
            ProcessState::Retired(status) => Ok(Some(*status)),
            ProcessState::Owned { process, .. } => {
                let pid = process_id(process);
                let status = observe_exit(process)?;
                if let Some(status) = status {
                    log_exit(pid, status);
                    *state = ProcessState::Retired(status);
                }
                Ok(status)
            }
        }
    }

    fn terminate(&self, code: &str, reason: &str) {
        tracing::info!(worker.pid = self.pid(), code, reason, "terminating inference worker");
        if let Err(error) = self.retire() {
            tracing::error!(%error, "inference worker retirement failed; retaining its owner");
        }
    }

    /// Ask a ready worker to unload and exit. A worker that never became ready has no protocol
    /// session to ask, so it is terminated.
    fn shutdown(&self) {
        match self.client.get() {
            Some(client) => {
                if let Err(error) = client.shutdown() {
                    tracing::warn!(%error, "inference worker did not accept shutdown");
                }
            }
            None => self.terminate("worker_shutdown", "the worker was not ready"),
        }
    }
}

impl Drop for EngineWorker {
    fn drop(&mut self) {
        if let Err(error) = self.retire() {
            tracing::error!(%error, "inference worker cleanup could not prove retirement");
        }
    }
}

/// Log a worker's standard error line by line as it arrives, keeping its tail in `tail`.
fn drain_diagnostics(
    mut stderr: crate::worker_process::BlockingStderr,
    tail: Arc<Mutex<DiagnosticTail>>,
    pid: u32,
) -> std::io::Result<()> {
    std::thread::Builder::new()
        .name(format!("inference-worker-stderr-{pid}"))
        .spawn(move || {
            let mut buffer = [0_u8; 4096];
            let mut lines = DiagnosticLines::default();
            let log = |line: &[u8]| {
                let text = String::from_utf8_lossy(line);
                let text = text.trim_end();
                if !text.is_empty() {
                    tracing::warn!(worker.pid = pid, message = %text, "inference worker");
                }
            };
            loop {
                match stderr.read(&mut buffer) {
                    Ok(0) | Err(_) => break,
                    Ok(length) => {
                        let bytes = &buffer[..length];
                        lines.push(bytes, log);
                        tail.lock().expect("diagnostic tail lock").append(bytes);
                    }
                }
            }
            lines.finish(log);
        })
        .map(drop)
}

/// The longest diagnostic line logged as one record; longer output is logged in pieces.
const DIAGNOSTIC_LINE_BYTES: usize = 4096;

/// Splits worker stderr into whole lines, so a line written in many small pieces is one record.
#[derive(Default)]
struct DiagnosticLines(Vec<u8>);

impl DiagnosticLines {
    fn push(&mut self, bytes: &[u8], mut emit: impl FnMut(&[u8])) {
        for &byte in bytes {
            if byte == b'\n' {
                emit(&self.0);
                self.0.clear();
            } else {
                self.0.push(byte);
                if self.0.len() >= DIAGNOSTIC_LINE_BYTES {
                    emit(&self.0);
                    self.0.clear();
                }
            }
        }
    }

    fn finish(self, mut emit: impl FnMut(&[u8])) {
        if !self.0.is_empty() {
            emit(&self.0);
        }
    }
}

fn process_id(process: &BlockingProcess) -> u32 {
    process.id()
}

/// Wait for a worker that reported its outcome to exit on its own, as `exited` observes it: it
/// is left to exit, as a released worker is after its shutdown, for the same bound
/// ([`GRACEFUL_EXIT`]). Killing it at once would skip everything its normal exit completes (the
/// system's compiled-kernel cache a Prepare's search wrote among it). `None` if it has not
/// exited by then, and must be stopped.
fn reported_exit(
    mut exited: impl FnMut() -> std::io::Result<Option<ExitStatus>>,
) -> std::io::Result<Option<ExitStatus>> {
    let deadline = std::time::Instant::now() + GRACEFUL_EXIT;
    while std::time::Instant::now() < deadline {
        if let Some(status) = exited()? {
            return Ok(Some(status));
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    Ok(None)
}

/// How long a stopped worker is waited for once killed: on Windows, the bound [`stop`] gives
/// its job object's processes to be proven gone. On Unix a killed process is reaped once the
/// kernel releases it, which a driver can delay without bound, so a caller that must not wait
/// on that holds it to the same bound.
pub(crate) const KILL_REAP: std::time::Duration = std::time::Duration::from_secs(2);

#[cfg(unix)]
fn observe_exit(process: &mut BlockingProcess) -> std::io::Result<Option<ExitStatus>> {
    process.try_wait()
}

#[cfg(windows)]
fn observe_exit(process: &mut BlockingProcess) -> std::io::Result<Option<ExitStatus>> {
    process.try_retirement()
}

#[cfg(unix)]
fn stop(process: &mut BlockingProcess) -> std::io::Result<ExitStatus> {
    if process.try_wait()?.is_none() {
        process.kill()?;
    }
    process.wait()
}

#[cfg(windows)]
fn stop(process: &mut BlockingProcess) -> std::io::Result<ExitStatus> {
    process.retire(KILL_REAP)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_tail_keeps_the_most_recent_bytes() {
        let mut tail = DiagnosticTail::default();
        tail.append(&vec![b'a'; DIAGNOSTIC_TAIL_BYTES]);
        tail.append(b"last line");
        let text = tail.text();
        assert_eq!(text.len(), DIAGNOSTIC_TAIL_BYTES);
        assert!(text.ends_with("last line"));
    }

    #[test]
    fn diagnostic_lines_join_pieces_into_whole_lines() {
        let mut lines = DiagnosticLines::default();
        let mut seen = Vec::new();
        for piece in [&b"load"[..], b"ing ", b"model\nsecond", b" line\n", b"partial"] {
            lines.push(piece, |line| seen.push(line.to_vec()));
        }
        lines.push(&vec![b'x'; DIAGNOSTIC_LINE_BYTES], |line| seen.push(line.to_vec()));
        lines.finish(|line| seen.push(line.to_vec()));
        assert_eq!(seen[0], b"loading model");
        assert_eq!(seen[1], b"second line");
        assert_eq!(seen[2].len(), DIAGNOSTIC_LINE_BYTES);
        assert!(seen[2].starts_with(b"partial"));
        assert_eq!(seen[3], vec![b'x'; 7]);
        assert_eq!(seen.len(), 4);
    }

    #[test]
    fn memory_pressure_unload_has_its_own_exit_code() {
        assert_eq!(worker_exit_code(&WorkerExit::Shutdown), 0);
        assert_ne!(
            worker_exit_code(&WorkerExit::Unloaded(UnloadCause::MemoryPressure)),
            worker_exit_code(&WorkerExit::Unloaded(UnloadCause::Shutdown))
        );
    }

    #[cfg(unix)]
    #[test]
    fn retirement_is_proven_once_and_clears_the_live_pid() {
        let process = std::process::Command::new("/bin/sleep")
            .arg("60")
            .spawn()
            .unwrap();
        let pid = process.id();
        let device = crate::device_lock::DeviceLock::default();
        let worker = EngineWorker {
            process: Mutex::new(ProcessState::Owned {
                process,
                _device: DeviceHold::Share(Arc::new(device.try_share().unwrap())),
            }),
            client: OnceLock::new(),
            diagnostics: Arc::new(Mutex::new(DiagnosticTail::default())),
        };
        assert_eq!(worker.pid(), Some(pid));
        assert!(worker.try_wait().unwrap().is_none());
        // The device is shared while the worker runs and released once it is proven retired.
        assert!(device.try_job().is_none());
        worker.shutdown();
        let status = worker
            .try_wait()
            .unwrap()
            .expect("unready shutdown terminates");
        assert_eq!(worker.pid(), None);
        assert!(device.try_job().is_some());
        assert_eq!(worker.retire().unwrap(), status);
    }
}
