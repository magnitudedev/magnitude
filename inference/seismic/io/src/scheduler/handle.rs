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
        let Some((queued, wake)) = self.engine.queue().cancel(self.id) else {
            return;
        };
        if wake > 0 {
            self.engine.wake(wake);
        }
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
    /// Its submitter, as whose a panic reading its pieces ends.
    pub(crate) submitter: seismic_threads::Submitter,
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
    /// A request submitted on the calling thread, for which its pieces are
    /// read ([`Self::submitter`]).
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
            submitter: seismic_threads::Submitter::here(),
        })
    }

    /// End the request now: it has no piece.
    pub(crate) fn end(&self, ended: Ended) {
        let progress = self.progress.lock().expect("request progress");
        self.finish(progress, ended);
    }

    /// The request's end, one transition: its outcome is set and its
    /// reader's count of requests outstanding is decremented under its lock,
    /// before releasing the lock and announcing its waiters and its hook.
    /// The only way a request ends and the only place a hook registered
    /// before the end runs, so no observer (via `wait`, `wait_until`, or
    /// `on_end`) can observe the request ended while `outstanding` still
    /// counts it.
    fn finish(&self, mut progress: std::sync::MutexGuard<'_, Progress>, ended: Ended) {
        if progress.ended.is_some() {
            return;
        }
        progress.ended = Some(ended.clone());
        self.outstanding.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        let hook = progress.hook.take();
        drop(progress);
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

#[cfg(test)]
mod tests {
    use super::{Ending, RequestState};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::time::{Duration, Instant};

    /// Once any observer (`wait`, `wait_until`, or `on_end` before or after
    /// finish) observes a request as ended, `outstanding` has already been
    /// decremented.
    #[test]
    fn finish_decrements_outstanding_before_observers_see_ended() {
        // Hook registered before finish sees outstanding == 0.
        let outstanding = Arc::new(AtomicUsize::new(1));
        let state = RequestState::new(1, Vec::new(), outstanding.clone(), Vec::new());
        let ending = Ending(state.clone());
        let ran = Arc::new(AtomicBool::new(false));
        let hook_outstanding = outstanding.clone();
        let hook_ran = ran.clone();
        ending.on_end(move |_| {
            assert_eq!(hook_outstanding.load(Ordering::Acquire), 0);
            hook_ran.store(true, Ordering::Release);
        });
        state.pieces_ended(1, [], None);
        assert!(ran.load(Ordering::Acquire));

        // Concurrent wait, wait_until, and post-finish on_end never see ended
        // while outstanding still counts the finished request.
        for _ in 0..2_000 {
            let outstanding = Arc::new(AtomicUsize::new(1));
            let state = RequestState::new(1, Vec::new(), outstanding.clone(), Vec::new());
            let ending = Ending(state.clone());
            let barrier = Arc::new(Barrier::new(3));

            let b1 = barrier.clone();
            let e1 = ending.clone();
            let o1 = outstanding.clone();
            let waiter = std::thread::spawn(move || {
                b1.wait();
                let _ = e1.wait();
                assert_eq!(o1.load(Ordering::Acquire), 0, "wait observed ended before outstanding decremented");
            });

            let b2 = barrier.clone();
            let e2 = ending.clone();
            let o2 = outstanding.clone();
            let timed = std::thread::spawn(move || {
                b2.wait();
                let _ = e2.wait_until(Instant::now() + Duration::from_secs(5));
                assert_eq!(o2.load(Ordering::Acquire), 0, "wait_until observed ended before outstanding decremented");
            });

            barrier.wait();
            state.pieces_ended(1, [], None);
            ending.on_end(move |_| {
                assert_eq!(outstanding.load(Ordering::Acquire), 0, "on_end observed ended before outstanding decremented");
            });
            waiter.join().unwrap();
            timed.join().unwrap();
        }
    }
}
