//! The threads a load's storage work runs on off the submitting thread.

use crate::CompletionWake;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, LockResult};
use std::time::Instant;

/// A step's outcome once it ends on a [`StorageThread`], and when.
pub(crate) struct Done<T>(Arc<(Mutex<Option<(T, Instant)>>, Condvar)>);

impl<T> Clone for Done<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> Done<T> {
    pub(crate) fn new() -> Self {
        Self(Arc::new((Mutex::new(None), Condvar::new())))
    }

    pub(crate) fn lock(&self) -> LockResult<MutexGuard<'_, Option<(T, Instant)>>> {
        self.0.0.lock()
    }

    /// Block until the storage thread records the outcome and take it.
    pub(crate) fn wait(&self) -> (T, Instant) {
        let mut guard = self.0.0.lock().expect("storage work outcome");
        loop {
            if let Some(done) = guard.take() {
                return done;
            }
            guard = self.0.1.wait(guard).expect("storage work outcome");
        }
    }

    fn complete(&self, outcome: T) {
        *self.0.0.lock().expect("storage work outcome") = Some((outcome, Instant::now()));
        self.0.1.notify_all();
    }
}

/// A thread storage work runs on, started with its owner and joined when it
/// drops: the expert bank's records' resizes and its host tier's (each
/// store's own, so neither queues behind the other), and the load's
/// releases of device storage (`Releases`), each a store of work its
/// owner only hands it. The kernel creates a thread from memory it may wait
/// for, and memory is short exactly when storage is released: a 72-slot
/// shrink of Qwen3.6-35B-A3B under 8 GB of pressure held the submitting
/// thread 483.9 ms, all but its 3.5 ms plan and moves in the spawn of its
/// prepare's thread (the prepare of a shrink backs, zeroes and forms
/// nothing; it was done 483.4 ms after the decision), M4 Pro, 2026-10-27.
pub(crate) struct StorageThread {
    work: Option<std::sync::mpsc::Sender<Work>>,
    /// Its one thread, busy while its owner's work runs, beside decode for
    /// the host tier's backing (`experts::RESIZE_THREADS`).
    thread: Option<std::thread::JoinHandle<()>>,
}

/// A step of storage work, run on the thread: its panic ends the process
/// (`seismic::threads`).
type Work = Box<dyn FnOnce() + Send>;

impl StorageThread {
    /// Its thread, named `name`, placed on every CPU the process may run on
    /// (`seismic::pin`, which also keeps it out of the serving executor's
    /// placement, `HostCompute::serving`): a host tier's backing first
    /// touches its pages on the threads it starts, which inherit that
    /// placement, so the tier is spread over every memory node. On the
    /// runtime's CPUs alone (nodes 2 and 3 of an EPYC 7313) the tier lay on
    /// half the memory channels and decode measured 45% slower than unplaced
    /// (one alternated pair); spread, 7.6%; on the workers' own nodes, 21.9
    /// ms against 11.9.
    pub(crate) fn start(name: &str) -> Result<Self, String> {
        let every = seismic::discover_processors().allowed().iter().copied().collect::<Vec<_>>();
        let (work, received) = std::sync::mpsc::channel::<Work>();
        let (placed, placement) = std::sync::mpsc::sync_channel(1);
        let thread = seismic::threads::spawn(name, move || {
            let pinned = seismic::pin(&every);
            let serves = pinned.is_ok();
            placed.send(pinned).expect("a storage thread's start waits for its placement");
            if serves {
                received.iter().for_each(|work| work());
            }
        })
        .map_err(|error| format!("the {name} thread: {error}"))?;
        placement.recv().map_err(|error| format!("the {name} thread: {error}"))?.map_err(|error| format!("the {name} thread's placement: {error}"))?;
        Ok(Self {
            work: Some(work),
            thread: Some(thread),
        })
    }

    /// Run `work` on the thread: its outcome lands in the returned slot,
    /// then `wake` completes. It cannot fail: the thread takes work until
    /// its owner drops it, so the work, and everything it owns, always
    /// reaches it.
    pub(crate) fn run<T: Send + 'static>(&self, wake: CompletionWake, work: impl FnOnce() -> T + Send + 'static) -> Done<T> {
        let done = Done::new();
        let slot = done.clone();
        self.work
            .as_ref()
            .expect("a storage thread takes work until its owner drops it")
            .send(Box::new(move || {
                let outcome = work();
                slot.complete(outcome);
                wake.complete();
            }))
            .expect("a storage thread takes work until its owner drops it, since its panic ends the process");
        done
    }
}

impl Drop for StorageThread {
    fn drop(&mut self) {
        drop(self.work.take());
        if let Some(thread) = self.thread.take() {
            thread.join().expect("a storage thread's panic ends the process");
        }
    }
}

/// The load's one thread its device storage is released on, each release
/// handed to it by its owner: the expert bank's shrinks and the graph slots
/// the pools release (`AllocatedResources::release_idle_slots`). A release of
/// device storage waits for the device (CUDA's `cuMemFree` synchronises it),
/// so on the submitting thread it holds the next step: releasing 49 idle
/// graph slots of LFM2.5 8B on a GB10 held it 31.2 ms against an 8.5 ms
/// step. Releases queue only behind each other, never behind a growth's
/// backing, which runs on its store's resize thread. Clones share it.
#[derive(Clone)]
pub struct Releases(Arc<StorageThread>);

impl Releases {
    pub(crate) fn start() -> Result<Self, String> {
        StorageThread::start("device-release").map(|thread| Self(Arc::new(thread)))
    }

    /// Release what `work` drops on the thread ([`StorageThread::run`]).
    pub(crate) fn run<T: Send + 'static>(&self, wake: CompletionWake, work: impl FnOnce() -> T + Send + 'static) -> Done<T> {
        self.0.run(wake, work)
    }
}
