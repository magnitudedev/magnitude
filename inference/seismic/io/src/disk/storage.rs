//! A volume's read line as the measurement job takes it: `λ + b/D`, a read
//! of `b` bytes, from two kinds of read of a probe file the job keeps on the
//! volume ([`StorageProbe`]), past the host's file cache:
//!
//! - **lone reads** ([`StorageProbe::lone`]): one [`LONE_READ`] at a time,
//!   at random aligned offsets: `λ + LONE_READ/D`, a demand request's;
//! - **full reads** ([`StorageProbe::full`]): [`FULL_DEPTH`] readers each
//!   reading [`FULL_READ`] at random aligned offsets, the volume's widest
//!   probed window in flight ([`WIDEST_WINDOW`]): `D`, its full rate.
//!
//! Both are of the volume, not of any model: the file is the job's own, so
//! no model's file is read or evicted to time them, and a volume that holds
//! no model yet is measured as one that does.

use super::file::SourceFile;
use super::platform::{evict, uncached};
use super::probe::{LEAST_READ, WIDEST_WINDOW};
use super::ALIGNMENT;
use crate::{Buffer, Durability, Error, Privacy, Store};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// The bytes of the probe file: 256 MiB, far past what a lone or a full
/// read touches, so random offsets rarely meet, and written once.
pub const STORAGE_PROBE_BYTES: u64 = 256 << 20;

/// A lone read's bytes: the least read size a volume is read in.
pub const LONE_READ: u64 = LEAST_READ;

/// A full read's bytes and its readers: the volume's widest probed window
/// in flight, where every measured device reaches its rate.
pub const FULL_READ: u64 = 1 << 20;
pub const FULL_DEPTH: u64 = WIDEST_WINDOW / FULL_READ;

/// Fault every 4 KiB page of `buffer` into the process page table before timed
/// I/O so page-fault overhead never contaminates storage throughput measurements.
fn fault_buffer(buffer: &mut Buffer) {
    for page in buffer.bytes_mut().chunks_mut(4096) {
        page[0] = 1;
    }
}

/// Shared synchronization state for [`FullWorkers`].
struct WorkerShared {
    state: Mutex<WorkerState>,
    start: Condvar,
    done: Condvar,
}

struct WorkerState {
    ready: usize,
    epoch: u64,
    offsets: Vec<Vec<u64>>,
    completed: usize,
    error: Option<Error>,
    shutdown: bool,
}

/// Persistent pool of [`FULL_DEPTH`] reader threads, each owning a pre-allocated
/// and pre-faulted [`FULL_READ`] aligned buffer, spawned and warmed outside any
/// timed [`StorageProbe::full`] span.
struct FullWorkers {
    run_lock: Mutex<()>,
    shared: Arc<WorkerShared>,
    handles: Vec<JoinHandle<()>>,
}

impl FullWorkers {
    fn new(file: Arc<SourceFile>) -> Result<Self, Error> {
        let depth = FULL_DEPTH as usize;
        let shared = Arc::new(WorkerShared {
            state: Mutex::new(WorkerState {
                ready: 0,
                epoch: 0,
                offsets: Vec::new(),
                completed: 0,
                error: None,
                shutdown: false,
            }),
            start: Condvar::new(),
            done: Condvar::new(),
        });
        let mut handles: Vec<JoinHandle<()>> = Vec::with_capacity(depth);
        for worker_idx in 0..depth {
            let file = Arc::clone(&file);
            let worker_shared = Arc::clone(&shared);
            let handle = seismic_threads::spawn(format!("seismic-storage-{worker_idx}"), move || {
                let mut buffer = Buffer::new(FULL_READ as usize);
                fault_buffer(&mut buffer);
                {
                    let mut state = worker_shared.state.lock().expect("storage probe worker state");
                    state.ready += 1;
                    if state.ready == depth {
                        worker_shared.done.notify_one();
                    }
                }
                let mut seen_epoch = 0u64;
                loop {
                    let offsets = {
                        let mut state = worker_shared.state.lock().expect("storage probe worker state");
                        while state.epoch == seen_epoch && !state.shutdown {
                            state = worker_shared.start.wait(state).expect("storage probe worker wait");
                        }
                        if state.shutdown {
                            return;
                        }
                        seen_epoch = state.epoch;
                        std::mem::take(&mut state.offsets[worker_idx])
                    };
                    let result = offsets.iter().try_for_each(|&at| file.read_block(at, buffer.bytes_mut()));
                    let mut state = worker_shared.state.lock().expect("storage probe worker state");
                    if let Err(err) = result {
                        if state.error.is_none() {
                            state.error = Some(err);
                        }
                    }
                    state.completed += 1;
                    if state.completed == depth {
                        worker_shared.done.notify_one();
                    }
                }
            })
            .map_err(|cause| {
                {
                    let mut state = shared.state.lock().expect("storage probe worker state");
                    state.shutdown = true;
                    shared.start.notify_all();
                }
                for handle in handles.drain(..) {
                    let _ = handle.join();
                }
                Error::ThreadStart(cause)
            })?;
            handles.push(handle);
        }
        // Wait until every worker thread has spawned, allocated, and faulted its buffer.
        let mut state = shared.state.lock().expect("storage probe worker state");
        while state.ready < depth {
            state = shared.done.wait(state).expect("storage probe worker ready");
        }
        drop(state);
        Ok(Self { run_lock: Mutex::new(()), shared, handles })
    }

    fn run(&self, offsets: Vec<Vec<u64>>) -> Result<(), Error> {
        let depth = FULL_DEPTH as usize;
        let _run_guard = self.run_lock.lock().expect("storage probe run lock");
        let mut state = self.shared.state.lock().expect("storage probe worker state");
        state.offsets = offsets;
        state.completed = 0;
        state.error = None;
        state.epoch += 1;
        self.shared.start.notify_all();
        while state.completed < depth {
            state = self.shared.done.wait(state).expect("storage probe worker done");
        }
        match state.error.take() {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

impl Drop for FullWorkers {
    fn drop(&mut self) {
        {
            let mut state = self.shared.state.lock().expect("storage probe worker state");
            state.shutdown = true;
            self.shared.start.notify_all();
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

/// The probe file the measurement job reads a volume through.
pub struct StorageProbe {
    file: Arc<SourceFile>,
    state: Mutex<u64>,
    lone_buffer: Mutex<Buffer>,
    workers: FullWorkers,
}

#[derive(Debug)]
struct StorageProbePrepareError {
    path: PathBuf,
    source: std::io::Error,
}

impl std::fmt::Display for StorageProbePrepareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "cannot prepare storage calibration probe at {}: {}; calibration needs a probe on the model-store filesystem; check that the store is writable",
            self.path.display(),
            self.source,
        )
    }
}

impl std::error::Error for StorageProbePrepareError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

impl StorageProbe {
    /// The probe file in `directory`, written there first where it is not
    /// whole, its pages then dropped from the host's file cache.
    pub fn open(directory: &Path) -> Result<Self, Error> {
        let path = Self::path(directory);
        if std::fs::metadata(&path).map_or(true, |metadata| metadata.len() != STORAGE_PROBE_BYTES) {
            Self::write(directory, &path).map_err(|source| {
                let kind = source.kind();
                let error = StorageProbePrepareError { path: path.clone(), source };
                tracing::warn!(error = %error, "storage calibration probe could not be prepared");
                std::io::Error::new(kind, error)
            })?;
        }
        let file = std::fs::File::open(&path)?;
        evict(&file, STORAGE_PROBE_BYTES)?;
        let file = Arc::new(SourceFile { path: path.as_path().into(), size: STORAGE_PROBE_BYTES, through: uncached(&file, &path)? });
        let mut lone_buffer = Buffer::new(LONE_READ as usize);
        fault_buffer(&mut lone_buffer);
        let workers = FullWorkers::new(Arc::clone(&file))?;
        Ok(Self { file, state: Mutex::new(0x9e37_79b9_7f4a_7c15), lone_buffer: Mutex::new(lone_buffer), workers })
    }

    /// Where the probe file of `directory` lies.
    pub fn path(directory: &Path) -> PathBuf {
        directory.join(".storage-probe")
    }

    /// Write the probe file: bytes no volume stores as a hole or compresses
    /// (a xorshift stream), published through the directory's store
    /// ([`crate::Shared::publish`]), so a job killed while writing, or two
    /// writing at once, leave no short probe file, and the temporary file of
    /// one killed is the store's to sweep.
    fn write(directory: &Path, path: &Path) -> std::io::Result<()> {
        let store = Store::open(directory.to_owned(), Privacy::Shared).map_err(std::io::Error::other)?;
        store
            .shared()
            .and_then(|shared| {
                shared.publish(Path::new(path.file_name().expect("the probe file has a name")), Durability::Durable, |file| {
                    let mut state = 0x2545_f491_4f6c_dd1du64;
                    let mut chunk = vec![0u8; 8 << 20];
                    for _ in 0..STORAGE_PROBE_BYTES / chunk.len() as u64 {
                        for word in chunk.chunks_exact_mut(8) {
                            state = next(state);
                            word.copy_from_slice(&state.to_le_bytes());
                        }
                        file.write_all(&chunk)?;
                    }
                    Ok(())
                })
            })
            .map_err(std::io::Error::other)?;
        Ok(())
    }

    /// `passes` lone reads, one after another, into the pre-allocated,
    /// pre-faulted lone buffer.
    pub fn lone(&self, passes: usize) -> Result<(), Error> {
        let mut buffer = self.lone_buffer.lock().expect("storage probe lone buffer");
        for _ in 0..passes {
            let at = self.offset(LONE_READ);
            self.file.read_block(at, buffer.bytes_mut())?;
        }
        Ok(())
    }

    /// `passes` full reads: [`FULL_DEPTH`] persistent pre-spawned readers,
    /// each reading `passes` reads of [`FULL_READ`] into its pre-allocated,
    /// pre-faulted buffer.
    pub fn full(&self, passes: usize) -> Result<(), Error> {
        let offsets = (0..FULL_DEPTH).map(|_| (0..passes).map(|_| self.offset(FULL_READ)).collect::<Vec<_>>()).collect::<Vec<_>>();
        self.workers.run(offsets)
    }

    /// A random aligned offset a read of `length` bytes lies within the
    /// file from.
    fn offset(&self, length: u64) -> u64 {
        let mut state = self.state.lock().expect("the probe's offsets");
        *state = next(*state);
        *state % ((STORAGE_PROBE_BYTES - length) / ALIGNMENT + 1) * ALIGNMENT
    }
}

/// The xorshift step after `state`.
fn next(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_probe_prepare_error_preserves_store_failure() {
        // A file cannot be used as a directory, even when tests run as root.
        let store = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let path = StorageProbe::path(&store);
        let Err(error) = StorageProbe::open(&store) else {
            panic!("a regular file must not be accepted as a model store");
        };
        let message = error.to_string();
        assert!(message.contains(&path.display().to_string()), "{message}");
        assert!(message.contains("model-store filesystem"), "{message}");
        let Error::Io(error) = error else {
            panic!("expected a probe I/O error");
        };
        let context = error
            .get_ref()
            .and_then(|error| error.downcast_ref::<StorageProbePrepareError>())
            .expect("expected typed probe context");
        let source = std::error::Error::source(context)
            .and_then(|error| error.downcast_ref::<std::io::Error>())
            .expect("expected the original I/O source");
        assert_eq!(error.kind(), source.kind());
        let store_error = source.get_ref().expect("expected the original store error");
        assert!(message.contains(&store_error.to_string()), "{message}");
    }

    #[test]
    fn persistent_workers_run_repeated_lone_and_full_passes() {
        let dir = std::env::temp_dir().join(format!("seismic-storage-probe-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let probe = StorageProbe::open(&dir).expect("open storage probe");
        probe.lone(2).expect("lone pass");
        probe.full(1).expect("first full pass");
        probe.full(2).expect("second full pass");
        drop(probe);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
