//! A lock its background holder gives way on: the one shape of a value one
//! thread works on at length in the background (the ranking's feed,
//! residency's keeper) while others hold it for what a device waits on (a
//! parked request's claims, the request service's passes).
//!
//! A plain mutex lets a background holder that releases it and takes it
//! again at once take it before a waiter the release woke can, so the
//! waiter waits out the next hold too, and the one after. Here a holder
//! the device waits on ([`Yielding::hold`]) is counted while it waits; the
//! background takes the lock only once none is counted
//! ([`Yielding::background`]), and between its jobs gives it to any that
//! came meanwhile ([`Yielding::give_way`]): a waiter waits out at most the
//! one background job it found running, never a run of them.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, TryLockResult};

/// `T` behind a lock its background holder gives way on (module
/// documentation).
pub(crate) struct Yielding<T> {
    value: Mutex<T>,
    /// The holders waiting for it, besides the background.
    waiting: AtomicUsize,
}

impl<T> Yielding<T> {
    pub(crate) fn new(value: T) -> Self {
        Self { value: Mutex::new(value), waiting: AtomicUsize::new(0) }
    }

    /// The value, for a holder the background gives way to. `poisoned`
    /// says why a panic can never have left it.
    pub(crate) fn hold(&self, poisoned: &str) -> MutexGuard<'_, T> {
        self.waiting.fetch_add(1, Ordering::AcqRel);
        let held = self.value.lock().expect(poisoned);
        self.waiting.fetch_sub(1, Ordering::AcqRel);
        held
    }

    /// The value, for the background: once no other holder waits for it.
    pub(crate) fn background(&self, poisoned: &str) -> MutexGuard<'_, T> {
        while self.waiting.load(Ordering::Acquire) > 0 {
            std::thread::yield_now();
        }
        self.value.lock().expect(poisoned)
    }

    /// The background's `held` (its guard, or what wraps it, which
    /// `again` wraps it in once more), given first to every holder that
    /// came to wait for it since: the value is theirs between two of the
    /// background's jobs. Released by dropping `held`, so a wrapper's
    /// release runs as at any other.
    pub(crate) fn give_way<'a, H>(&'a self, held: H, poisoned: &str, again: impl FnOnce(MutexGuard<'a, T>) -> H) -> H {
        if self.waiting.load(Ordering::Acquire) == 0 {
            return held;
        }
        drop(held);
        again(self.background(poisoned))
    }

    /// The holders waiting for it now, besides the background: a test's
    /// sign that one waits.
    #[cfg(test)]
    pub(crate) fn waiting(&self) -> usize {
        self.waiting.load(Ordering::Acquire)
    }

    /// The value if no one holds it now, for a report that must not wait.
    pub(crate) fn try_hold(&self) -> TryLockResult<MutexGuard<'_, T>> {
        self.value.try_lock()
    }
}
