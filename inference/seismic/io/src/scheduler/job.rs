use super::engine::Engine;
use super::{Ended, Panic, Priority};
use super::Range;
use crate::Error;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Instant;

/// A submitted job: promoted, withdrawn or waited for. Dropping it withdraws
/// it and waits for its pieces in flight, so no destination is written
/// after its job's handle is gone; the job's state holds every destination
/// until its last piece ends, whatever becomes of the handle.
pub struct Job {
    pub(crate) id: u64,
    pub(crate) state: Arc<JobState>,
    pub(crate) engine: &'static Engine,
}

impl Job {
    /// Its submission: the id its timeline events carry ([`crate::timeline`]).
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Withdraw the job's pieces no worker has taken: they read nothing, the
    /// ranges they serve fail [`Error::Withdrawn`], and it ends with its
    /// pieces in flight (its hook runs, its waiters wake).
    pub fn withdraw(&self) {
        let Some(queued) = self.engine.queue.lock().expect("read queue").withdraw(self.id) else {
            return;
        };
        let withdrawn = Arc::new(Error::Withdrawn);
        let ranges = queued.pieces.iter().flat_map(|piece| piece.parts.iter().map(|part| part.range));
        let failed = ranges.collect::<std::collections::BTreeSet<_>>().into_iter().map(|range| (range, withdrawn.clone()));
        self.state.pieces_ended(queued.pieces.len(), failed, None);
    }

    /// When a worker takes its first piece, set then: unset while every
    /// piece waits (or where the job had none to read). Shared, so a hook
    /// on its end reads it without the job.
    pub fn started(&self) -> Arc<OnceLock<Instant>> {
        self.state.started.clone()
    }

    /// Read the job's pieces no less urgently than `priority`.
    pub fn promote(&self, priority: Priority) {
        let woken = self.engine.queue.lock().expect("read queue").promote(self.id, priority);
        for _ in 0..woken {
            self.engine.work.notify_one();
        }
    }

    /// Run `hook` with how the job ended once it has: at once where it has,
    /// else on the thread that ends it (the worker that read its last piece,
    /// or the submitter of a job of no piece), after its end is set, so the
    /// hook, and whatever it wakes, finds the job ended ([`JobState::finish`]).
    /// One hook a job; it must not wait for a read.
    pub fn on_end(&self, hook: impl FnOnce(&Ended) + Send + 'static) {
        let mut progress = self.state.progress.lock().expect("job progress");
        match progress.ended.clone() {
            Some(ended) => {
                drop(progress);
                hook(&ended);
            }
            None => {
                assert!(progress.hook.is_none(), "one hook a job");
                progress.hook = Some(Box::new(hook));
            }
        }
    }

    /// How the job ended, waiting for it.
    pub fn wait(&self) -> Ended {
        let mut progress = self.state.progress.lock().expect("job progress");
        loop {
            if let Some(ended) = &progress.ended {
                return ended.clone();
            }
            progress = self.state.changed.wait(progress).expect("job progress");
        }
    }

    /// How the job ended, waiting for it until `deadline`; `None` while it
    /// runs.
    pub fn wait_until(&self, deadline: Instant) -> Option<Ended> {
        let mut progress = self.state.progress.lock().expect("job progress");
        loop {
            if let Some(ended) = &progress.ended {
                return Some(ended.clone());
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return None;
            }
            progress = self
                .state
                .changed
                .wait_timeout(progress, left)
                .expect("job progress")
                .0;
        }
    }
}

impl Drop for Job {
    /// Withdraw what no worker has taken, and wait for what one has.
    fn drop(&mut self) {
        self.withdraw();
        self.wait();
    }
}

pub(crate) struct JobState {
    pub(crate) progress: Mutex<Progress>,
    changed: Condvar,
    /// Its reader's jobs not yet ended, of which it is one until it ends.
    outstanding: Arc<std::sync::atomic::AtomicUsize>,
    /// When a worker took its first piece ([`Job::started`]).
    pub(crate) started: Arc<OnceLock<Instant>>,
    /// Its ranges, which hold their destinations while any piece may write
    /// them.
    _ranges: Vec<Range>,
}

pub(crate) struct Progress {
    /// Pieces not yet ended.
    left: usize,
    pub(crate) failed: Vec<(usize, Arc<Error>)>,
    /// The first panic of a worker reading one of its pieces.
    panic: Option<Panic>,
    pub(crate) ended: Option<Ended>,
    /// What runs once it ends ([`Job::on_end`]).
    hook: Option<Box<dyn FnOnce(&Ended) + Send>>,
}

impl JobState {
    pub(crate) fn new(pieces: usize, failed: Vec<(usize, Arc<Error>)>, outstanding: Arc<std::sync::atomic::AtomicUsize>, ranges: Vec<Range>) -> Arc<Self> {
        Arc::new(Self {
            progress: Mutex::new(Progress {
                left: pieces,
                failed,
                panic: None,
                ended: None,
                hook: None,
            }),
            changed: Condvar::new(),
            outstanding,
            started: Arc::default(),
            _ranges: ranges,
        })
    }

    /// End the job now: it has no piece.
    pub(crate) fn end(&self, ended: Ended) {
        let progress = self.progress.lock().expect("job progress");
        self.finish(progress, ended);
    }

    /// The job's end, one transition: its outcome is set first, under its
    /// lock, then every observer is announced: its reader's count of jobs
    /// outstanding, its waiters, and its hook. The only way a job ends and
    /// the only place a hook registered before the end runs, so no observer
    /// is announced before the state it would read: a hook's wake (an
    /// event another thread waits on) finds the job ended, as does any
    /// `running` it calls.
    fn finish(&self, mut progress: std::sync::MutexGuard<'_, Progress>, ended: Ended) {
        if progress.ended.is_some() {
            return;
        }
        progress.ended = Some(ended.clone());
        let hook = progress.hook.take();
        drop(progress);
        self.outstanding.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        self.changed.notify_all();
        if let Some(hook) = hook {
            hook(&ended);
        }
    }

    /// `count` of the job's pieces ended, failing `failed`, or by a panic;
    /// the job ends with its last piece, so no destination is written once it has.
    pub(crate) fn pieces_ended(&self, count: usize, failed: impl IntoIterator<Item = (usize, Arc<Error>)>, panic: Option<Panic>) {
        let mut progress = self.progress.lock().expect("job progress");
        progress.failed.extend(failed);
        if progress.panic.is_none() {
            progress.panic = panic;
        }
        progress.left -= count;
        if progress.left == 0 {
            let ended = match progress.panic.take() {
                Some(panic) => Ended::Panicked(panic),
                None => Ended::Read {
                    failed: first_failures(std::mem::take(&mut progress.failed)),
                },
            };
            self.finish(progress, ended);
        }
    }
}

/// Each failed range once, with its first error, in the job's order.
pub(crate) fn first_failures(mut failed: Vec<(usize, Arc<Error>)>) -> Vec<(usize, Arc<Error>)> {
    failed.sort_by_key(|(index, _)| *index);
    failed.dedup_by_key(|(index, _)| *index);
    failed
}
