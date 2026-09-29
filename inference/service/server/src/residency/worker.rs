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
use magnitude_engine::worker::{
    EngineClient, WorkerConnection, WorkerExit, connect_worker, prepare_worker, serve_worker,
};

use super::ResidencyWorker;
use crate::worker_process::{BlockingChild, BlockingProcess, WorkerLauncher, WorkerRole};

/// The most recent worker diagnostics kept for failure reports.
const DIAGNOSTIC_TAIL_BYTES: usize = 16 * 1024;

/// How the `inference-worker` process reports its end to the service. Crash handling reads
/// only whether the worker is gone; the code is diagnostic.
pub fn worker_exit_code(exit: &WorkerExit) -> i32 {
    match exit {
        WorkerExit::Shutdown | WorkerExit::Prepared | WorkerExit::HostLost { .. } => 0,
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

enum ProcessState {
    Owned(BlockingProcess),
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
    /// Keeps the measurement job off the device for as long as the worker may hold it.
    _device: tokio::sync::OwnedRwLockReadGuard<()>,
}

/// A spawned worker and the host end of its transport, before the load handshake.
pub struct SpawnedWorker {
    pub worker: Arc<EngineWorker>,
    transport: FramedTransport<
        std::io::BufReader<crate::worker_process::BlockingStdout>,
        crate::worker_process::BlockingStdin,
        magnitude_engine::worker::protocol::WorkerMessage,
        magnitude_engine::worker::protocol::HostMessage,
    >,
}

impl EngineWorker {
    /// Spawn an `inference-worker` process holding `device` (the residency side of the device
    /// exclusion) for its lifetime.
    pub fn spawn(
        launcher: &WorkerLauncher,
        device: tokio::sync::OwnedRwLockReadGuard<()>,
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
            process: Mutex::new(ProcessState::Owned(process)),
            client: OnceLock::new(),
            diagnostics: Arc::new(Mutex::new(DiagnosticTail::default())),
            _device: device,
        });
        drain_diagnostics(stderr, Arc::clone(&worker.diagnostics), pid)?;
        Ok(SpawnedWorker {
            worker,
            transport: FramedTransport::new(std::io::BufReader::new(stdout), stdin),
        })
    }

    /// The worker's recent standard error, for failure reports.
    pub fn diagnostics(&self) -> String {
        self.diagnostics.lock().expect("diagnostic tail lock").text()
    }

    fn retire(&self) -> std::io::Result<ExitStatus> {
        let mut state = self.process.lock().expect("worker process lock");
        match &mut *state {
            ProcessState::Retired(status) => Ok(*status),
            ProcessState::Owned(process) => {
                // An error leaves the exact owner in place. Only proof of retirement replaces it.
                let status = stop(process)?;
                *state = ProcessState::Retired(status);
                Ok(status)
            }
        }
    }
}

impl SpawnedWorker {
    /// Send `Hello` and `Load{manifest}` and wait for readiness, reporting load progress.
    /// Blocking: the worker reports its progress until it is ready or fails.
    pub fn connect(
        self,
        manifest: ExecutionManifest,
        progress: impl FnMut(LoadProgress),
    ) -> Result<(Arc<EngineWorker>, WorkerConnection), (Arc<EngineWorker>, LoadError)> {
        match connect_worker(self.transport, Some(manifest), progress) {
            Ok(connection) => {
                self.worker
                    .client
                    .set(connection.client.clone())
                    .unwrap_or_else(|_| unreachable!("a worker connects once"));
                Ok((self.worker, connection))
            }
            Err(error) => Err((self.worker, error)),
        }
    }
}

impl SpawnedWorker {
    /// Send `Hello` and `Prepare{manifest}` and wait until the worker has prepared the model's
    /// programs, reporting its progress. Blocking; the worker exits once prepared.
    pub fn prepare(
        self,
        manifest: ExecutionManifest,
        progress: impl FnMut(LoadProgress),
    ) -> Result<(), LoadError> {
        prepare_worker(self.transport, manifest, progress)
    }
}

impl ResidencyWorker for EngineWorker {
    fn pid(&self) -> Option<u32> {
        match &*self.process.lock().expect("worker process lock") {
            ProcessState::Owned(process) => Some(process_id(process)),
            ProcessState::Retired(_) => None,
        }
    }

    fn try_wait(&self) -> std::io::Result<Option<ExitStatus>> {
        let mut state = self.process.lock().expect("worker process lock");
        match &mut *state {
            ProcessState::Retired(status) => Ok(Some(*status)),
            ProcessState::Owned(process) => {
                let status = observe_exit(process)?;
                if let Some(status) = status {
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

fn drain_diagnostics(
    mut stderr: crate::worker_process::BlockingStderr,
    tail: Arc<Mutex<DiagnosticTail>>,
    pid: u32,
) -> std::io::Result<()> {
    std::thread::Builder::new()
        .name(format!("inference-worker-stderr-{pid}"))
        .spawn(move || {
            let mut buffer = [0_u8; 4096];
            loop {
                match stderr.read(&mut buffer) {
                    Ok(0) | Err(_) => break,
                    Ok(length) => {
                        let bytes = &buffer[..length];
                        tracing::warn!(
                            worker.pid = pid,
                            message = %String::from_utf8_lossy(bytes).trim_end(),
                            "inference worker"
                        );
                        tail.lock().expect("diagnostic tail lock").append(bytes);
                    }
                }
            }
        })
        .map(drop)
}

fn process_id(process: &BlockingProcess) -> u32 {
    process.id()
}

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
    process.retire(std::time::Duration::from_secs(2))
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
        let lock = Arc::new(tokio::sync::RwLock::new(()));
        let guard = lock.clone().try_read_owned().unwrap();
        let worker = EngineWorker {
            process: Mutex::new(ProcessState::Owned(process)),
            client: OnceLock::new(),
            diagnostics: Arc::new(Mutex::new(DiagnosticTail::default())),
            _device: guard,
        };
        assert_eq!(worker.pid(), Some(pid));
        assert!(worker.try_wait().unwrap().is_none());
        worker.shutdown();
        let status = worker.try_wait().unwrap().expect("unready shutdown terminates");
        assert_eq!(worker.pid(), None);
        assert_eq!(worker.retire().unwrap(), status);
        assert!(lock.try_write().is_err(), "the device stays excluded while the worker exists");
        drop(worker);
        assert!(lock.try_write().is_ok());
    }
}
