//! A prompt group's scratch: the records a prompt converts each streamed
//! layer's cold experts through, in sub-passes of as many experts as it
//! holds. The bank's floor already holds a decode step's set; what a group
//! holds beyond it is an optional committed claim on the lender's store,
//! taken as the group opens from what yields at once (free room, spare
//! granules, bank slots above the floor), never preempting a sequence and
//! never refused, and held, fixed, until the group closes.

use super::ledger::{Claim, CommittedKind, Ledger};
use std::num::NonZeroU64;

/// What a group's scratch is sized from: the bank's header-derived counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchShape {
    /// The decode step's set, which the floor holds (`WorkingSet::lent`).
    pub decode: u64,
    /// The widest streamed layer's cold need: the records a prompt converts
    /// a layer through in one pass.
    pub cold: u64,
    /// One record of the lender's, in its store.
    pub record_bytes: NonZeroU64,
}

/// A prompt group's scratch, held from its open to its close: `S`, the
/// records its sub-passes convert through, and the committed claim for
/// those beyond the decode set.
#[must_use]
pub struct PromptScratch {
    claim: Claim,
    records: u64,
}

impl PromptScratch {
    /// The scratch of a group opening now in `ledger` (the lender's store):
    /// `S = min(cold, decode + the most records that yield at once)`, at
    /// least the decode set, which the floor holds; fixed for the group.
    pub fn open(ledger: &Ledger, shape: ScratchShape) -> Self {
        let wanted = shape.cold.saturating_sub(shape.decode);
        let (claim, more) = ledger.claim_units(CommittedKind::Scratch, shape.record_bytes, wanted);
        Self { claim, records: shape.cold.min(shape.decode + more) }
    }

    /// `S`: the records the group's sub-passes convert each layer through.
    pub fn records(&self) -> u64 {
        self.records
    }

    /// The bytes it commits beyond the floor's decode set.
    pub fn bytes(&self) -> u64 {
        self.claim.bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::ledger::{Available, CacheKind, Ledgers, LoadShape, Need};
    use std::sync::Arc;
    use std::time::Instant;

    const RECORD: u64 = 1 << 20;

    /// Host RAM's ledger in a scope of its own, read with `total` free.
    fn ledger(total: u64) -> (Ledgers, Ledger) {
        let scope = Ledgers::new();
        let ledger = scope.ledger(seismic::StoreIdentity::HostRam, Arc::new(LoadShape { stores: Vec::new() }));
        ledger.read(|| Ok(Available::new(total, total, None, Instant::now()).expect("a reading")));
        (scope, ledger)
    }

    fn shape(decode: u64, cold: u64) -> ScratchShape {
        ScratchShape { decode, cold, record_bytes: NonZeroU64::new(RECORD).unwrap() }
    }

    /// A group takes its cold need whole where free room holds it, and
    /// gives it back as it closes.
    #[test]
    fn a_group_takes_its_cold_need_where_room_holds_it() {
        let (_scope, ledger) = ledger(64 * RECORD);
        let scratch = PromptScratch::open(&ledger, shape(16, 40));
        assert_eq!((scratch.records(), scratch.bytes()), (40, 24 * RECORD));
        drop(scratch);
        assert_eq!(ledger.snapshot().committed(), 0, "the group's close frees its scratch");
    }

    /// Short of room, the scratch is the decode set and the whole records
    /// free room and the bank's cache yield: never a sequence's state, and
    /// never a refusal.
    #[test]
    fn a_group_takes_only_what_yields_at_once() {
        let (_scope, ledger) = ledger(64 * RECORD);
        let committed = ledger.claim(Need::Measurement { bytes: 50 * RECORD }).expect("room");
        let bank = ledger.cache(CacheKind::Bank, 10 * RECORD + RECORD / 2).expect("room");
        // Free: 3.5 records; the bank's cache: 10.5. Whole records: 14.
        let scratch = PromptScratch::open(&ledger, shape(16, 40));
        assert_eq!(scratch.records(), 16 + 14);
        assert_eq!(committed.bytes(), 50 * RECORD, "no committed claim yielded");
        assert!(bank.bytes() < 10 * RECORD + RECORD / 2, "the bank's cache yielded");
        // With nothing left to yield, the scratch is the decode set.
        let none = PromptScratch::open(&ledger, shape(16, 40));
        assert_eq!((none.records(), none.bytes()), (16, 0));
    }

    /// A layer whose cold need is within the decode set takes nothing.
    #[test]
    fn a_cold_need_within_the_decode_set_takes_nothing() {
        let (_scope, ledger) = ledger(64 * RECORD);
        let scratch = PromptScratch::open(&ledger, shape(16, 12));
        assert_eq!((scratch.records(), scratch.bytes()), (12, 0));
    }
}
