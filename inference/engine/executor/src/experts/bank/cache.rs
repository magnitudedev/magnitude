//! A bank's cache on its stores' ledgers: what its holdings hold above
//! their floors, in the pieces its growths took, the running growth's apart
//! as a claim allocated as its slots back it, until an advance reports what
//! it grew. The one bookkeeping every bank keeps of it, the served bank's
//! and a test's alike.

use crate::domain::{BankHolding, BankResized};
use crate::memory::ledger::{CacheKind, Cached, Claim};
use std::sync::Mutex;

#[derive(Default)]
pub(crate) struct BankCache {
    held: Mutex<Vec<(BankHolding, Cached)>>,
    growing: Mutex<Option<Growing>>,
}

/// A running growth: its holding, the holding's charge as it began, and
/// its claim (`Ledger::backing`).
struct Growing {
    holding: BankHolding,
    from: u64,
    claim: Claim,
}

impl BankCache {
    /// `cache` held for `holding` from now: what it holds above its floor.
    pub(crate) fn hold(&self, holding: BankHolding, cache: Cached) {
        self.held.lock().expect("bank cache").push((holding, cache));
    }

    /// `claim` (`Ledger::backing`) taken for a growth of `holding` that
    /// began at a charge of `from`: allocated as its slots back it
    /// ([`Self::backed`]), until an advance reports what it grew
    /// ([`Self::settle`]).
    pub(crate) fn grow(&self, holding: BankHolding, from: u64, claim: Claim) {
        let previous = self.growing.lock().expect("bank growth").replace(Growing { holding, from, claim });
        assert!(previous.is_none(), "invariant violated: a growth began beside another");
    }

    /// The holding a growth of runs, where one does.
    pub(crate) fn growing(&self) -> Option<BankHolding> {
        self.growing.lock().expect("bank growth").as_ref().map(|growing| growing.holding)
    }

    /// The growing holding's slots backed so far are charged `charged`:
    /// its claim is allocated what they hold, never less than it was (the
    /// resize thread records each slot as it backs, the owner the slots
    /// backed before it held the claim). None runs, nothing is recorded.
    pub(crate) fn backed(&self, charged: u64) {
        if let Some(growing) = self.growing.lock().expect("bank growth").as_mut() {
            let backed = charged.saturating_sub(growing.from);
            if backed > growing.claim.allocated_bytes() {
                growing.claim.backed(backed);
            }
        }
    }

    /// The cache of the resize an advance reports finished: a growth's
    /// claim turns cache of what it grew and frees the rest, one stopped or
    /// refused frees it all, a shrink frees what it released
    /// ([`Self::release`]).
    pub(crate) fn settle(&self, resized: &BankResized) {
        match *resized {
            BankResized::Shrunk { holding, released, .. } => self.release(holding, released),
            BankResized::Grown { holding, grown, .. } => {
                let growing = self.growing.lock().expect("bank growth").take().expect("invariant violated: a growth without its claim");
                assert_eq!(growing.holding, holding, "invariant violated: a growth of another holding's claim");
                self.hold(holding, growing.claim.into_backed_cache(grown, CacheKind::Bank));
            }
            BankResized::Stopped { .. } | BankResized::GrowthRefused { .. } => {
                drop(self.growing.lock().expect("bank growth").take());
            }
        }
    }

    /// A shrink of `holding` freed `bytes`: first what claims took of the
    /// bank's cache in its store and the bank had not freed yet, the
    /// holding's own pieces' before the others' there; then the holding's
    /// latest pieces.
    fn release(&self, holding: BankHolding, mut bytes: u64) {
        let mut held = self.held.lock().expect("bank cache");
        let mut payers: Vec<usize> = (0..held.len()).filter(|at| held[*at].0 == holding).collect();
        if let Some(store) = payers.first().map(|at| held[*at].1.store()) {
            payers.extend((0..held.len()).filter(|at| held[*at].0 != holding && held[*at].1.store() == store));
        }
        for at in payers {
            let paid = held[at].1.releasing().min(bytes);
            held[at].1.released(paid);
            bytes -= paid;
        }
        while bytes > 0 {
            let Some(at) = held.iter().rposition(|(of, _)| *of == holding) else {
                return;
            };
            let has = held[at].1.bytes();
            match has <= bytes {
                true => {
                    held.remove(at);
                    bytes -= has;
                }
                false => {
                    drop(held[at].1.split(bytes, CacheKind::Bank));
                    bytes = 0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::ledger::{Available, Ledger, Ledgers, LoadShape, Need};

    const GB: u64 = 1 << 30;

    fn ledger() -> (Ledgers, Ledger) {
        let scope = Ledgers::new();
        let ledger = scope.ledger(seismic::StoreIdentity::HostRam, std::sync::Arc::new(LoadShape { stores: Vec::new() }));
        let at = std::time::Instant::now();
        ledger.read(|| Ok(Available::new(64 * GB, 64 * GB, None, at).expect("a reading")));
        (scope, ledger)
    }

    fn times() -> crate::domain::ResizeTimes {
        let now = std::time::Instant::now();
        let zero = std::time::Duration::ZERO;
        crate::domain::ResizeTimes {
            decided: now,
            queued: zero,
            planned: zero,
            trips: Default::default(),
            moves: zero,
            prepared: now,
            installed: now,
            released: now,
            unpinned: zero,
        }
    }

    /// A shrink frees first what claims took of the bank in process: the
    /// claim allocates it then, and the bank's pieces keep what they hold.
    #[test]
    fn a_shrink_frees_what_claims_took_first() {
        let (_scope, ledger) = ledger();
        let cache = BankCache::default();
        let holding = BankHolding::Experts;
        cache.hold(holding, ledger.cache(CacheKind::Bank, 8 * GB).expect("room"));
        cache.hold(holding, ledger.cache(CacheKind::Bank, 8 * GB).expect("room"));
        let _taken = ledger.claim(Need::Measurement { bytes: 52 * GB }).expect("the bank yields 4 GB");
        assert_eq!((ledger.snapshot().releasing(), ledger.snapshot().allocatable()), (4 * GB, 48 * GB));
        cache.settle(&BankResized::Shrunk { holding, from: 16, to: 12, released: 4 * GB, times: times() });
        let snapshot = ledger.snapshot();
        assert_eq!((snapshot.releasing(), snapshot.cache_of(CacheKind::Bank), snapshot.allocatable()), (0, 12 * GB, 52 * GB));
    }

    /// A growth's claim is allocated as its slots back it; ended, it holds
    /// what it grew as cache and frees the rest.
    #[test]
    fn a_growth_is_allocated_as_it_backs() {
        let (_scope, ledger) = ledger();
        let cache = BankCache::default();
        let holding = BankHolding::Experts;
        cache.grow(holding, 10 * GB, ledger.backing(8 * GB).expect("room"));
        assert_eq!(cache.growing(), Some(holding));
        assert_eq!(ledger.snapshot().allocated(), 0, "nothing backs it yet");
        cache.backed(13 * GB);
        assert_eq!(ledger.snapshot().allocated(), 3 * GB, "its joined slots");
        cache.settle(&BankResized::Grown { holding, from: 10, to: 15, grown: 5 * GB, refused: None, times: times() });
        let snapshot = ledger.snapshot();
        assert_eq!((snapshot.allocated(), snapshot.cache_of(CacheKind::Bank), snapshot.committed()), (5 * GB, 5 * GB, 0));
        assert_eq!(cache.growing(), None);
    }
}
