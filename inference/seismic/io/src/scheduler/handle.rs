use super::engine::Engine;
use super::{Ended, Panic, Want};
use super::Range;
use crate::Error;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Instant;

/// A request made of a link: it owns the request's pieces no worker has
/// taken. Dropping it cancels them: they read nothing, the ranges they
/// serve fail [`Error::Cancelled`], and the request ends with its pieces in
/// flight (its hook runs, its waiters wake). Those finish into the
/// destinations the request's state holds until its last piece ends,
/// whatever becomes of the handle, so a read into freed memory has no
/// expression; the caller learns that end through its [`Ending`], which
/// the handle dereferences to and which outlives it ([`Handle::ending`]),
/// before it reuses the bytes ([`crate::Reader::request`]).
pub struct Handle {
    pub(crate) id: u64,
    pub(crate) ending: Ending,
    pub(crate) engine: &'static Engine,
}

/// A request's end as its observers see it: shared, and holding nothing of
/// the request but its state, so it may outlive the request's handle.
#[derive(Clone)]
pub struct Ending(pub(crate) Arc<RequestState>);

impl Handle {
    /// Its submission: the id its timeline events carry ([`crate::timeline`]).
    pub fn id(&self) -> u64 {
        self.id
    }

    /// An observer of its end that outlives it.
    pub fn ending(&self) -> Ending {
        self.ending.clone()
    }

    /// Order its pieces not yet taken by `want` from now: earlier or later,
    /// due or not (a request a consumer now waits on is repriced to
    /// [`Want::now`]).
    pub fn reprice(&self, want: Want) {
        let woken = self.engine.queue().reprice(self.id, want);
        self.engine.wake(woken);
    }
}

impl std::ops::Deref for Handle {
    type Target = Ending;

    fn deref(&self) -> &Ending {
        &self.ending
    }
}

impl Drop for Handle {
    /// Cancel what no worker has taken; what one has reads on.
    fn drop(&mut self) {
        let Some(queued) = self.engine.queue().cancel(self.id) else {
            return;
        };
        let cancelled = Arc::new(Error::Cancelled);
        let ranges = queued.pieces.iter().flat_map(|piece| piece.parts.iter().map(|part| part.range));
        let failed = ranges.collect::<std::collections::BTreeSet<_>>().into_iter().map(|range| (range, cancelled.clone()));
        self.ending.0.pieces_ended(queued.pieces.len(), failed, None);
    }
}

impl Ending {
    /// When a worker takes its request's first piece, set then: unset while
    /// every piece waits (or where the request had none to read). Shared,
    /// so a hook on its end reads it.
    pub fn started(&self) -> Arc<OnceLock<Instant>> {
        self.0.started.clone()
    }

    /// Run `hook` with how the request ended once it has: at once where it
    /// has, else on the thread that ends it (the worker that read its last
    /// piece, the dropper that cancelled its last, or the submitter of a
    /// request of no piece), after its end is set, so the hook, and whatever
    /// it wakes, finds the request ended ([`RequestState::finish`]). One hook
    /// a request; it must not wait for a read.
    pub fn on_end(&self, hook: impl FnOnce(&Ended) + Send + 'static) {
        let mut progress = self.0.progress.lock().expect("request progress");
        match progress.ended.clone() {
            Some(ended) => {
                drop(progress);
                hook(&ended);
            }
            None => {
                assert!(progress.hook.is_none(), "one hook a request");
                progress.hook = Some(Box::new(hook));
            }
        }
    }

    /// How the request ended, waiting for it.
    pub fn wait(&self) -> Ended {
        let mut progress = self.0.progress.lock().expect("request progress");
        loop {
            if let Some(ended) = &progress.ended {
                return ended.clone();
            }
            progress = self.0.changed.wait(progress).expect("request progress");
        }
    }

    /// How the request ended, waiting for it until `deadline`; `None` while
    /// it runs.
    pub fn wait_until(&self, deadline: Instant) -> Option<Ended> {
        let mut progress = self.0.progress.lock().expect("request progress");
        loop {
            if let Some(ended) = &progress.ended {
                return Some(ended.clone());
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return None;
            }
            progress = self
                .0
                .changed
                .wait_timeout(progress, left)
                .expect("request progress")
                .0;
        }
    }
}

pub(crate) struct RequestState {
    pub(crate) progress: Mutex<Progress>,
    changed: Condvar,
    /// Its reader's requests not yet ended, of which it is one until it
    /// ends.
    outstanding: Arc<std::sync::atomic::AtomicUsize>,
    /// When a worker took its first piece ([`Handle::started`]).
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
    /// What runs once it ends ([`Handle::on_end`]).
    hook: Option<Box<dyn FnOnce(&Ended) + Send>>,
}

impl RequestState {
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

    /// End the request now: it has no piece.
    pub(crate) fn end(&self, ended: Ended) {
        let progress = self.progress.lock().expect("request progress");
        self.finish(progress, ended);
    }

    /// The request's end, one transition: its outcome is set first, under
    /// its lock, then every observer is announced: its reader's count of
    /// requests outstanding, its waiters, and its hook. The only way a
    /// request ends and the only place a hook registered before the end
    /// runs, so no observer is announced before the state it would read: a
    /// hook's wake (an event another thread waits on) finds the request
    /// ended, as does any `running` it calls.
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

    /// `count` of the request's pieces ended, failing `failed`, or by a
    /// panic; the request ends with its last piece, so no destination is
    /// written once it has.
    pub(crate) fn pieces_ended(&self, count: usize, failed: impl IntoIterator<Item = (usize, Arc<Error>)>, panic: Option<Panic>) {
        let mut progress = self.progress.lock().expect("request progress");
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

/// Each failed range once, with its first error, in the request's order.
pub(crate) fn first_failures(mut failed: Vec<(usize, Arc<Error>)>) -> Vec<(usize, Arc<Error>)> {
    failed.sort_by_key(|(index, _)| *index);
    failed.dedup_by_key(|(index, _)| *index);
    failed
}
