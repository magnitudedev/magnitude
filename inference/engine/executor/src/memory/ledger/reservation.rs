//! A request's state as it is written: reserved as it opens for the rows it
//! is about to write, and extended ahead of its steps at their boundaries,
//! so no row of it ever waits on memory mid-launch.

use super::admission::Short;
use super::claims::Claim;
use super::state::{EntryId, State};
use super::view::Ledger;
use super::vocabulary::{CommittedKind, Holding, Need, Value, size};
use crate::operation::RequestId;

/// A request's state, `rows` rows of it ([`Need::Request`]): reserved as it
/// opens ([`Ledger::reserve`]) and extended ahead of its steps
/// ([`Ledger::extend`]). Its rows draw within it and cannot fail. Dropped,
/// what it drew of the load's request slot goes back to the slot and the
/// rest is free.
#[must_use]
pub struct Reservation {
    claim: Claim,
    rows: u64,
    through: u64,
}

impl Ledger {
    /// Reserve `rows` rows of `sequence`'s state as it opens: what the slot
    /// holds first, then free room beside the guarantee, within the
    /// boundary, the cache worth nothing yielding to it (`State::draw`).
    /// The bank and retained prefixes give it room only through a relief
    /// plan. Short, nothing is held, and the open waits for the next
    /// reading.
    ///
    /// # Panics
    /// Where `slot` is not this store's request slot.
    pub fn reserve(&self, slot: &Claim, sequence: RequestId, rows: u64) -> Result<Reservation, Short> {
        self.check_slot(slot);
        let kind = CommittedKind::Request(sequence);
        let bytes = size(&self.shape, &Need::Request { sequence, rows });
        let id = self.state().draw(&self.store.identity, slot.id(), None, kind, bytes)?;
        Ok(Reservation { claim: Claim::held(self.clone(), id, kind), rows, through: 0 })
    }

    /// Extend `reservation` to `rows` rows, ahead of the rows its next step
    /// writes, as [`Self::reserve`] takes room. Short, it holds what it
    /// held, and the step is relieved or waits.
    ///
    /// # Panics
    /// Where `slot` is not this store's request slot, or `rows` holds no
    /// more than the reservation does.
    pub fn extend(&self, slot: &Claim, reservation: &mut Reservation, rows: u64) -> Result<(), Short> {
        self.check_slot(slot);
        assert!(rows > reservation.rows, "invariant violated: a reservation of {} rows extends to {rows}", reservation.rows);
        let CommittedKind::Request(sequence) = reservation.claim.kind() else {
            unreachable!("a reservation holds a request's state")
        };
        let bytes = size(&self.shape, &Need::Request { sequence, rows });
        let more = bytes.checked_sub(reservation.bytes()).expect("invariant violated: a request's state shrinks as its rows grow");
        self.state().draw(&self.store.identity, slot.id(), Some(reservation.claim.id()), reservation.claim.kind(), more)?;
        reservation.rows = rows;
        Ok(())
    }

    fn check_slot(&self, slot: &Claim) {
        assert!(
            slot.kind() == CommittedKind::RequestSlot && std::sync::Arc::ptr_eq(&slot.ledger().store, &self.store),
            "invariant violated: a reservation draws on a {:?} claim of another store",
            slot.kind()
        );
    }
}

impl State {
    /// Hold `bytes` more of `kind`, a request's state, in `into` or a new
    /// entry: what `slot` holds first, the rest beside it
    /// (`State::take_beside`). The slot's part is lent: it goes back to the
    /// slot as the entry is released. Short, nothing moves.
    fn draw(&mut self, store: &seismic::StoreIdentity, slot: EntryId, into: Option<EntryId>, kind: CommittedKind, bytes: u64) -> Result<EntryId, Short> {
        let held = self.entry(slot).bytes;
        let drawn = held.min(bytes);
        let beside = match bytes - drawn {
            0 => None,
            beside => {
                let at = self.reading(store, None);
                match self.take_beside(at, kind, beside) {
                    Ok(id) => Some(id),
                    Err(over) => {
                        self.waiting = true;
                        return Err(over.of(store, bytes));
                    }
                }
            }
        };
        self.resize(slot, held - drawn);
        let id = match into {
            Some(into) => {
                let has = self.entry(into).bytes;
                self.resize(into, has + drawn);
                into
            }
            None => self.insert(Holding::Committed(kind), drawn),
        };
        if let Some(beside) = beside {
            self.merge(beside, id);
        }
        if drawn > 0 {
            let lent = self.lent.entry(id).or_insert((slot, 0));
            lent.1 += drawn;
        }
        Ok(id)
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
    /// holds. A row past them is an invariant violation: each step's rows
    /// are reserved before it is admitted ([`Ledger::extend`]).
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
