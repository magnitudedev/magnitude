//! The lent bank records' reservation: beside a host tier a prompt group's
//! scratch lies in its lender's bottom bank records (`planning::ScratchPlan`),
//! which decode's promotions also fill between groups. A group that uses
//! scratch reserves them before any of its steps begins, and holds them
//! until the device is past it: from then no promotion is planned into one,
//! and a promotion yet to copy into one copies nothing and is dropped as
//! the group's service starts (`BankRecords::lend`), so the lend never
//! waits on a promotion or a copy. A group begins only once
//! every copy into them has completed: that wait is the host's, before the
//! group's steps exist, so nothing the device runs waits on a copy. The
//! wait counts as a reservation already, so no copy into a lent record
//! queues behind the batch it waits on; it ends with serving, whose end or
//! stop wakes it, as no copy is carried to its completion once serving
//! ends.
//!
//! The order is the steps' lock, then the reservation's: promotions and
//! releases are asked, and each release's expert unmapped from the first
//! step not yet begun, under the steps' lock, and a group reserves under it
//! before its steps begin (a wait counted as reserved), so every release
//! of a lent record unmapped its expert from a step at or before the
//! group's first, or saw the reservation and asked around it; and no
//! promotion is asked into a free lent record while the group holds it.

use super::liveness::Life;
use crate::watched::Wait;
use std::sync::{Condvar, Mutex, MutexGuard};

/// Who holds the lender's lent records (which they are is the bank
/// records' own, `experts::records`, or on the one store its host
/// records'): beside a host tier no promotion copies into one while a
/// prompt group holds them, and on either placer the reserve unmaps none
/// then (`HostRecords::refill`).
pub(in crate::experts) struct LentRecords {
    state: Mutex<Lending>,
    copied: Condvar,
}

/// The reservation's state.
#[derive(Default)]
pub(in crate::experts) struct Lending {
    /// The prompt groups begun whose device work has not passed.
    reserved: usize,
    /// The prompt groups waiting to begin until no copy into a lent record
    /// runs ([`LentRecords::await_copies`]).
    pending: usize,
    /// The promotions' copies into lent records queued and not completed.
    copies: usize,
    /// Whether the reservation ended since the promotions last asked
    /// ([`Self::take_freed`]): they may take the lent records again.
    freed: bool,
}

impl Lending {
    /// Whether a prompt group holds the lent records, or waits to.
    pub(in crate::experts) fn reserved(&self) -> bool {
        self.reserved > 0 || self.pending > 0
    }

    /// Whether the reservation ended since this was last asked: the
    /// promotions, which pass over the lent records while it holds, may
    /// take them again.
    pub(in crate::experts) fn take_freed(&mut self) -> bool {
        std::mem::take(&mut self.freed)
    }

    /// Whether the reservation ended since [`Self::take_freed`] was last
    /// asked, which it leaves for the promotions to take.
    pub(in crate::experts) fn freed(&self) -> bool {
        self.freed
    }

    /// Leave the reservation (a group holding or awaiting it): marked freed
    /// once no group holds or awaits it.
    fn leave(&mut self, reserved: bool) {
        match reserved {
            true => self.reserved = self.reserved.checked_sub(1).expect("the lent records released unreserved"),
            false => self.pending -= 1,
        }
        self.freed |= !self.reserved();
    }

    /// A promotion's copy into a lent record is queued
    /// (`BankRecords::copy`): never while a prompt group holds them.
    pub(in crate::experts) fn copying(&mut self) {
        assert!(!self.reserved(), "a copy into a lent record while a prompt group holds them");
        self.copies += 1;
    }
}

impl LentRecords {
    pub(in crate::experts) fn new() -> Self {
        Self { state: Mutex::new(Lending::default()), copied: Condvar::new() }
    }

    /// The reservation's state, under its lock.
    pub(in crate::experts) fn lock(&self) -> MutexGuard<'_, Lending> {
        self.state.lock().expect("the lent records' reservation is never left by a panic")
    }

    /// Reserve the lent records for a prompt group about to begin, under the
    /// steps' lock: whether it holds them now, no copy into them running.
    /// Else it waits to hold them ([`Self::await_copies`]), outside that
    /// lock, which a copy's completion needs.
    pub(in crate::experts) fn reserve(&self) -> bool {
        let mut state = self.lock();
        let free = state.copies == 0;
        match free {
            true => state.reserved += 1,
            false => state.pending += 1,
        }
        free
    }

    /// Wait, for a prompt group [`Self::reserve`] found a copy into the lent
    /// records running, until every one has completed, then hold them; or
    /// until serving ends or the load stops, after which none completes:
    /// why.
    pub(super) fn await_copies(&self, life: &Life) -> Result<(), String> {
        let wait = Wait::idle("a prompt group's begin", "the promotions' copies into the lent bank records");
        let mut state = wait.on(self.lock(), &self.copied, |state| state.copies == 0 || life.ended().is_some() || life.stopping());
        if state.copies > 0 {
            state.leave(false);
            return Err(life.ended().unwrap_or_else(|| "the load stops".to_owned()));
        }
        state.pending -= 1;
        state.reserved += 1;
        Ok(())
    }

    /// Serving ended or the load stops: a begin waiting for a copy returns.
    pub(super) fn wake(&self) {
        let _state = self.lock();
        self.copied.notify_all();
    }

    /// A prompt group that reserved them did not begin, or the device is
    /// past it.
    pub(in crate::experts) fn release(&self) {
        self.lock().leave(true);
    }

    /// A copy into a lent record ended, completed or failed
    /// (`BankRecords::promoted`, `BankRecords::unpromoted`).
    pub(in crate::experts) fn copied(&self) {
        let mut state = self.lock();
        state.copies = state.copies.checked_sub(1).expect("more copies into lent records ended than were queued");
        if state.copies == 0 {
            self.copied.notify_all();
        }
    }
}
