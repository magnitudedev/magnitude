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
    EngineClient, WorkerConnection, WorkerEnd, WorkerExit, WorkerProcess, WorkerReport, connect_worker,
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
/// host loss or unload, then end the process, which releases what the load holds.
pub fn run_inference_worker() -> ! {
    let served = serve_worker(stdio_worker_transport());
    let code = worker_exit_code(&served.exit);
    if code != 0 {
        eprintln!("inference worker ended: {:?}", served.exit);
    }
    served.end_process(code)
}

/// How long a load failure waits for an ended worker's exit status.
const EXIT_OBSERVATION: std::time::Duration = std::time::Duration::from_secs(1);

/// Record a worker's start as it is spawned: with its exit ([`log_exit`]), the service's record
/// of which workers ran when.
fn log_start(pid: u32) {
    tracing::info!(worker.pid = pid, "inference worker started");
}

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
        /// Dropping it with the owner releases the device; a stop the system refuses marks it
        /// unretired ([`DeviceHold::unretired`]).
        device: DeviceHold,
    },
    /// Its one retirer ([`EngineWorker::retire`]) holds the process and its device while it
    /// stops it, outside this state's lock, so nothing that reads or ends the worker waits on a
    /// reap. The retirer sends the kill: a terminate meanwhile has none to send.
    Retiring { pid: u32 },
    Retired(ExitStatus),
}

/// The worker's process state stands from its spawn until its drop takes it.
const STANDS: &str = "a worker's process state stands until its drop";

/// A worker is retired by the one owner of its [`SpawnedWorker`], which holds it until then.
const ONE_RETIRER: &str = "a worker has one retirer, which holds it while it retires it";

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
    /// Taken only by the worker's drop, which hands an unretired process to its reaper.
    process: Mutex<Option<ProcessState>>,
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
        log_start(pid);
        let worker = Arc::new(Self {
            process: Mutex::new(Some(ProcessState::Owned {
                process,
                device,
            })),
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

    /// Stop the worker and return once it is reaped, the process lock held only to take the
    /// process and to put back what became of it: a reap that hangs in the driver's teardown
    /// blocks this retirer alone. An error puts the exact owner back, its device marked
    /// unretired. Only proof of retirement replaces it.
    fn retire(&self) -> std::io::Result<ExitStatus> {
        let (pid, mut process, device) = {
            let mut state = self.process.lock().expect("worker process lock");
            let state = state.as_mut().expect(STANDS);
            let pid = match &*state {
                ProcessState::Retired(status) => return Ok(*status),
                ProcessState::Retiring { .. } => unreachable!("{ONE_RETIRER}"),
                ProcessState::Owned { process, .. } => process_id(process),
            };
            let ProcessState::Owned { process, device } = std::mem::replace(state, ProcessState::Retiring { pid }) else {
                unreachable!("the state was read as owned under the same lock")
            };
            (pid, process, device)
        };
        device.dying(pid);
        let stopped = stop(&mut process);
        let mut state = self.process.lock().expect("worker process lock");
        let state = state.as_mut().expect(STANDS);
        match stopped {
            Ok(status) => {
                log_exit(pid, status);
                *state = ProcessState::Retired(status);
                // The device is released once the retirement is shown.
                drop(device);
                Ok(status)
            }
            Err(error) => {
                device.unretired();
                *state = ProcessState::Owned { process, device };
                Err(error)
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
    /// `load_prepared`): its transport, and the proof of its retirement with how it ended. A
    /// Prepare that reported its outcome is left to exit, as is one whose stream ended, for as
    /// long as its exit is observed ([`EngineWorker::ended_status`]); one that reported
    /// nothing is stopped unless it had already ended.
    pub fn into_process(
        self,
    ) -> WorkerProcess<HostEnd, impl FnOnce(WorkerReport) -> Result<WorkerEnd, String>> {
        let Self { worker, transport } = self;
        WorkerProcess {
            transport,
            retire: move |report| {
                let error = |error: std::io::Error| error.to_string();
                let exited = match report {
                    WorkerReport::Reported => {
                        return worker.retire_reported().map(WorkerEnd::Exited).map_err(error);
                    }
                    WorkerReport::Ended => worker.ended_status(),
                    WorkerReport::Unreported => worker.try_wait().map_err(error)?,
                };
                match exited {
                    Some(status) => Ok(WorkerEnd::Exited(status)),
                    None => worker.retire().map(|_| WorkerEnd::Stopped).map_err(error),
                }
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
        log_start(pid);
        let worker = Self {
            process,
            diagnostics: Arc::new(Mutex::new(DiagnosticTail::default())),
        };
        if let Err(error) = drain_diagnostics(stderr, Arc::clone(&worker.diagnostics), pid) {
            worker.retire(false, |_| {}).map_err(|(_, error)| error)?;
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
    /// stopped ([`stop`]), `stopping` told its process just before. Blocking. One whose end the
    /// system will not show comes back with why, still holding all it holds
    /// ([`RETIREMENT_RETRY`]).
    pub fn retire(mut self, reported: bool, stopping: impl FnOnce(u32)) -> Result<ExitStatus, (Self, std::io::Error)> {
        let pid = process_id(&self.process);
        let exited = match reported {
            true => reported_exit(|| observe_exit(&mut self.process)),
            false => Ok(None),
        };
        let stopped = |process: &mut BlockingProcess| {
            stopping(pid);
            stop(process)
        };
        let status = match exited.and_then(|exited| exited.map_or_else(|| stopped(&mut self.process), Ok)) {
            Ok(status) => status,
            Err(error) => return Err((self, error)),
        };
        log_exit(pid, status);
        Ok(status)
    }
}

impl ResidencyWorker for EngineWorker {
    fn pid(&self) -> Option<u32> {
        match self.process.lock().expect("worker process lock").as_ref().expect(STANDS) {
            ProcessState::Owned { process, .. } => Some(process_id(process)),
            ProcessState::Retiring { pid } => Some(*pid),
            ProcessState::Retired(_) => None,
        }
    }

    fn try_wait(&self) -> std::io::Result<Option<ExitStatus>> {
        let mut state = self.process.lock().expect("worker process lock");
        let state = state.as_mut().expect(STANDS);
        match &mut *state {
            ProcessState::Retired(status) => Ok(Some(*status)),
            ProcessState::Retiring { .. } => Ok(None),
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

    fn terminate(&self, code: &str, reason: &str) -> std::io::Result<()> {
        tracing::info!(worker.pid = self.pid(), code, reason, "terminating inference worker");
        let mut state = self.process.lock().expect("worker process lock");
        let state = state.as_mut().expect(STANDS);
        match &mut *state {
            ProcessState::Retired(_) | ProcessState::Retiring { .. } => Ok(()),
            ProcessState::Owned { process, .. } => {
                let pid = process_id(process);
                match observe_exit(process)? {
                    Some(status) => {
                        log_exit(pid, status);
                        *state = ProcessState::Retired(status);
                        Ok(())
                    }
                    None => start_kill(process),
                }
            }
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
            None => self.end("worker_shutdown", "the worker was not ready"),
        }
    }

    /// The worker's `Unloaded` answer arrived. A connection that merely closed is no answer: a
    /// process still running behind it is killed at the deadline. A worker that never became
    /// ready has no session to answer, and is killed by its shutdown.
    fn answered_shutdown(&self) -> bool {
        self.client.get().is_none_or(|client| client.unloaded())
    }
}

impl Drop for EngineWorker {
    /// A worker dropped before its retirement was shown has no owner left to poll it: it is
    /// stopped on a thread of its own, which holds its device until the kernel has reaped it, so
    /// no drop waits on the reap and no device is released before it. A stop the system refuses
    /// is tried again ([`RETIREMENT_RETRY`]).
    fn drop(&mut self) {
        // The dropped worker's state is read no more: what it held moves to its reaper.
        let state = self.process.get_mut().expect("worker process lock").take().expect(STANDS);
        let (mut process, device) = match state {
            ProcessState::Owned { process, device } => (process, device),
            ProcessState::Retired(_) => return,
            ProcessState::Retiring { .. } => unreachable!("{ONE_RETIRER}"),
        };
        let pid = process_id(&process);
        std::thread::Builder::new()
            .name(format!("inference-worker-reap-{pid}"))
            .spawn(move || {
                device.dying(pid);
                let status = loop {
                    match stop(&mut process) {
                        Ok(status) => break status,
                        Err(error) => {
                            tracing::error!(worker.pid = pid, %error, "an ownerless inference worker could not be shown to have ended; trying again");
                            device.unretired();
                            std::thread::sleep(RETIREMENT_RETRY);
                        }
                    }
                };
                log_exit(pid, status);
                drop(device);
            })
            .expect("a thread to reap an ownerless inference worker");
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

/// How long a worker whose end cannot be shown waits before it is tried again: the one policy
/// for a kill the system refused to send, or an exit it refused to report. Such a worker keeps
/// everything it holds, its device among it, is reported, and is tried again until its end is
/// shown, by its owner: a stop retries a resident's ([`super::ReleaseOperation`]'s retirement
/// failure), the measurement job's owner retries its job's before taking its end, and a dropped
/// worker's own thread retries it.
pub(crate) const RETIREMENT_RETRY: std::time::Duration = std::time::Duration::from_secs(1);

#[cfg(unix)]
fn observe_exit(process: &mut BlockingProcess) -> std::io::Result<Option<ExitStatus>> {
    process.try_wait()
}

#[cfg(windows)]
fn observe_exit(process: &mut BlockingProcess) -> std::io::Result<Option<ExitStatus>> {
    process.try_retirement()
}

/// Send `process` the kill, waiting for nothing.
#[cfg(unix)]
fn start_kill(process: &mut BlockingProcess) -> std::io::Result<()> {
    process.kill()
}

#[cfg(windows)]
fn start_kill(process: &mut BlockingProcess) -> std::io::Result<()> {
    process.start_kill()
}

/// Stop `process` and return once it is reaped, however long that takes: its owner gives way
/// meanwhile as the system reports its death (`DeviceJob::dying`). An error, a kill or a reap the
/// system refused, is a process the system will not show gone, which its owner marks unretired
/// (`DeviceJob::unretired`).
#[cfg(unix)]
fn stop(process: &mut BlockingProcess) -> std::io::Result<ExitStatus> {
    if let Some(status) = process.try_wait()? {
        return Ok(status);
    }
    process.kill()?;
    process.wait()
}

#[cfg(windows)]
fn stop(process: &mut BlockingProcess) -> std::io::Result<ExitStatus> {
    process.retire()
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
            process: Mutex::new(Some(ProcessState::Owned {
                process,
                device: DeviceHold::Share(Arc::new(device.try_share().unwrap())),
            })),
            client: OnceLock::new(),
            diagnostics: Arc::new(Mutex::new(DiagnosticTail::default())),
        };
        assert_eq!(worker.pid(), Some(pid));
        assert!(worker.try_wait().unwrap().is_none());
        // The device is shared while the worker runs and released once it is proven retired.
        assert!(device.try_job().is_none());
        // An unready worker's shutdown sends the kill; its retirement is shown once reaped.
        worker.shutdown();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let status = loop {
            if let Some(status) = worker.try_wait().unwrap() {
                break status;
            }
            assert!(std::time::Instant::now() < deadline, "the killed worker was never reaped");
            std::thread::sleep(std::time::Duration::from_millis(1));
        };
        assert_eq!(worker.pid(), None);
        assert!(device.try_job().is_some());
        assert_eq!(worker.retire().unwrap(), status);
    }
}
