//! The load's events: what every host job of expert streaming waits on
//! instead of a time. Each event that may let a job progress counts one and
//! wakes every waiter: a step begun or ended (`Steps`), a landing's end
//! (`landing::Ended`), a read's end (`source::Ticket::on_end`), a batch of
//! the transfer queue queued or complete (`copies`), the device reaching a
//! park where parks are submission boundaries (`watcher::parks`), a growth
//! joined, a resize asked, serving ended and the load stopping. A waiter
//! takes the count ([`Events::seen`]) before it looks at what it waits for,
//! so no event after it looks is missed; it then re-reads its own state, so
//! an event that changed nothing of it costs one look.

use std::sync::{Condvar, Mutex};
use std::time::Duration;

/// The count of the load's events and their waiters.
#[derive(Default)]
pub(in crate::experts) struct Events {
    count: Mutex<u64>,
    changed: Condvar,
}

impl Events {
    fn count(&self) -> std::sync::MutexGuard<'_, u64> {
        self.count.lock().expect("the load's events")
    }

    /// The events so far, what a wait ([`Self::wait`]) waits past: taken
    /// before the waiter looks at its state.
    pub fn seen(&self) -> u64 {
        *self.count()
    }

    /// An event: every waiter wakes.
    pub fn notify(&self) {
        *self.count() += 1;
        self.changed.notify_all();
    }

    /// Wait until an event after `seen` of them.
    pub fn wait(&self, seen: u64) {
        drop(self.changed.wait_while(self.count(), |count| *count == seen).expect("the load's events"));
    }

    /// [`Self::wait`], at most `bound`: for the one waiter that must also
    /// look at what no event announces (the device's words, `watcher`'s
    /// request service).
    pub fn wait_at_most(&self, seen: u64, bound: Duration) {
        drop(self.changed.wait_timeout_while(self.count(), bound, |count| *count == seen).expect("the load's events"));
    }
}
