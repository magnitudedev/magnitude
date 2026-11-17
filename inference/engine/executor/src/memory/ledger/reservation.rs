//! A request's whole state, reserved as it opens: every row it may write is
//! held before its first, so no row of it ever waits on memory.

use super::admission::Short;
use super::claims::Claim;
use super::view::Ledger;
use super::vocabulary::{CommittedKind, Holding, Need, Value, size};
use crate::operation::RequestId;

/// A request's whole need, reserved at open ([`Ledger::reserve`]). Its rows
/// draw within it and cannot fail. Dropped, its bytes are free, or go back
/// to the request slot it drew them from.
#[must_use]
pub struct Reservation {
    claim: Claim,
    rows: u64,
    through: u64,
}

impl Ledger {
    /// Reserve `sequence`'s whole state, `rows` rows of it
    /// ([`Need::Request`]): from `slot`, the load's request slot, where no
    /// request draws on it and it holds them, so the first request takes
    /// the room the load's fit counted for it; otherwise from free room
    /// beside the guarantee, within the boundary, the cache worth nothing
    /// yielding to it (`State::take_beside`). The bank and retained
    /// prefixes give it room only through a relief plan. Short, nothing is
    /// held, and the open waits for the next reading.
    ///
    /// # Panics
    /// Where `slot` is not this store's request slot.
    pub fn reserve(&self, slot: &Claim, sequence: RequestId, rows: u64) -> Result<Reservation, Short> {
        assert!(
            slot.kind() == CommittedKind::RequestSlot && std::sync::Arc::ptr_eq(&slot.ledger().store, &self.store),
            "invariant violated: a reservation draws on a {:?} claim of another store",
            slot.kind()
        );
        let need = Need::Request { sequence, rows };
        let (kind, bytes) = (need.kind(), size(&self.shape, &need));
        let mut state = self.state();
        let free = !state.lent.values().any(|lent| *lent == slot.id());
        let held = state.entry(slot.id()).bytes;
        let id = match free && held >= bytes {
            true => {
                state.resize(slot.id(), held - bytes);
                let id = state.insert(Holding::Committed(kind), bytes);
                state.lent.insert(id, slot.id());
                id
            }
            false => {
                let at = state.reading(&self.store.identity, None);
                match state.take_beside(at, kind, bytes) {
                    Ok(id) => id,
                    Err(over) => {
                        state.waiting = true;
                        return Err(over.of(&self.store.identity, bytes));
                    }
                }
            }
        };
        drop(state);
        Ok(Reservation { claim: Claim::held(self.clone(), id, kind), rows, through: 0 })
    }
}

impl Reservation {
    /// The rows it holds.
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// The rows its request has written through.
    pub fn through(&self) -> u64 {
        self.through
    }

    pub fn bytes(&self) -> u64 {
        self.claim.bytes()
    }

    /// Its request writes rows through `through`, all within the rows it
    /// holds. A row past them is an invariant violation: its open reserved
    /// every row the request may write.
    pub fn grow(&mut self, through: u64) {
        assert!(
            through <= self.rows,
            "invariant violated: a request writes through row {through}, past the {} rows it reserved",
            self.rows
        );
        self.through = self.through.max(through);
    }

    /// What the request's state is worth per byte, as its holder measures
    /// it (the seconds its replay costs per byte held): a shortage's relief
    /// preempts the lowest first.
    pub fn value(&self, value: Value) {
        let mut state = self.claim.ledger().state();
        state.entry(self.claim.id()).value = value;
        state.holdings += 1;
    }
}
