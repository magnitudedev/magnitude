//! A bank's cache on its stores' ledgers: what its holdings hold above
//! their floors, in the pieces its growths took, the running growth's apart
//! as a claim allocated as its slots back it, until an advance reports what
//! it grew. The one bookkeeping every bank keeps of it, the served bank's
//! and a test's alike.

use crate::domain::{BankHolding, BankResized};
use crate::memory::ledger::{CacheKind, Cached, Changes, Changing, Claim, DeviceLedgers};
use std::sync::{Arc, Mutex};

pub(crate) struct BankCache {
    held: Mutex<Vec<(BankHolding, Cached)>>,
    growing: Mutex<Option<Growing>>,
    /// The changes of the ledgers' scope it holds its cache in, from the
    /// bank's open: every memory change of the bank, the load's own
    /// backing too, is one change with its record ([`Changes::change`]).
    changes: Arc<Changes>,
}

/// Its last holder may be a resize's or release's work on a thread of its
/// own, which outlives the bank (a domain torn down with a shrink's release
/// in flight): what it holds is let go as a change of its own there, joined
/// to the one its thread has under way, the owner's teardown or the step
/// that ends the release.
impl Drop for BankCache {
    fn drop(&mut self) {
        let held = std::mem::take(self.held.get_mut().unwrap_or_else(std::sync::PoisonError::into_inner));
        let growing = self.growing.get_mut().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        self.changes.clone().back(|_| drop((held, growing)));
    }
}

/// A running growth: its holding, the holding's charge as it began, and
/// its claim (`Ledger::backing`).
struct Growing {
    holding: BankHolding,
    from: u64,
    claim: Claim,
}

impl BankCache {
    /// A bank's cache on the ledgers of `changes`' scope, holding nothing.
    pub(crate) fn new(changes: Arc<Changes>) -> Self {
        Self { held: Mutex::default(), growing: Mutex::default(), changes }
    }

    /// A bank's cache as the bank opens on the ledgers of `changes`' scope:
    /// a host tier of `host_slots` holds its backing's claim (`host`: the
    /// tier's charge at none, and the claim) as a growth from the open, each
    /// slot allocated in it as it backs.
    ///
    /// # Panics
    /// Where a host tier has no backing's claim, or a claim no host tier.
    pub(crate) fn opened(changes: Arc<Changes>, host_slots: u64, host: Option<(u64, Claim)>) -> Self {
        let cache = Self::new(changes);
        match (host_slots > 0, host) {
            (true, Some((from, claim))) => cache.grow(BankHolding::Host, from, claim),
            (false, None) => {}
            (true, None) => panic!("invariant violated: a host tier opened without its backing's claim"),
            (false, Some(_)) => panic!("invariant violated: a host tier's backing claimed for a bank without one"),
        }
        cache
    }

    /// The cache as a served bank takes it on `ledgers`: of the scope it
    /// opened on, a host tier of `host_slots` still holding its backing's
    /// claim from the open, which only an advance settles.
    ///
    /// # Panics
    /// Where either does not hold.
    pub(crate) fn served_on(&self, ledgers: &DeviceLedgers, host_slots: u64) {
        assert!(Arc::ptr_eq(&self.changes, ledgers.changes()), "invariant violated: a bank served on ledgers of another scope than it opened on");
        if host_slots > 0 {
            assert_eq!(self.growing(), Some(BankHolding::Host), "invariant violated: a host tier served without its backing's claim");
        }
    }

    /// The changes of the scope it holds its cache in.
    pub(crate) fn changes(&self) -> &Arc<Changes> {
        &self.changes
    }

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

    /// The growth held for did not begin: its claim is let go, nothing of
    /// it backed.
    pub(crate) fn abandon(&self) {
        let growing = self.growing.lock().expect("bank growth").take();
        self.changes.change(|_| drop(growing));
    }

    /// The holding a growth of runs, where one does.
    pub(crate) fn growing(&self) -> Option<BankHolding> {
        self.growing.lock().expect("bank growth").as_ref().map(|growing| growing.holding)
    }

    /// The growing holding's slots backed so far, within `changing`, are
    /// charged `charged`: its claim is allocated what they hold, never less
    /// than it was (the resize thread records each slot as it backs, the
    /// owner the slots backed before it held the claim). None runs, nothing
    /// is recorded.
    pub(crate) fn backed(&self, charged: u64, changing: &Changing) {
        if let Some(growing) = self.growing.lock().expect("bank growth").as_mut() {
            let backed = charged.saturating_sub(growing.from);
            if backed > growing.claim.allocated_bytes() {
                growing.claim.backed(backed, changing);
            }
        }
    }

    /// The cache of the resize an advance reports finished: a growth's
    /// claim turns cache of what it grew and frees the rest, one stopped or
    /// refused frees it all. A shrink's release was recorded where its
    /// memory was freed ([`Self::released`]).
    pub(crate) fn settle(&self, resized: &BankResized) {
        match *resized {
            BankResized::Shrunk { .. } => {}
            BankResized::Grown { holding, grown, .. } => {
                let growing = self.growing.lock().expect("bank growth").take().expect("invariant violated: a growth without its claim");
                assert_eq!(growing.holding, holding, "invariant violated: a growth of another holding's claim");
                let cache = self.changes.change(|changing| growing.claim.into_backed_cache(grown, CacheKind::Bank, changing));
                self.hold(holding, cache);
            }
            // What its slots backed is freed with them.
            BankResized::Stopped { .. } | BankResized::GrowthRefused { .. } => {
                let growing = self.growing.lock().expect("bank growth").take();
                self.changes.change(|_| drop(growing));
            }
        }
    }

    /// A shrink of `holding` freed `bytes` within `changing`, so no reading
    /// sees the memory free and still counted the bank's: first what claims
    /// took of the bank's cache in its store and the bank had not freed
    /// yet, the holding's own pieces' before the others' there; then the
    /// holding's latest pieces.
    pub(crate) fn released(&self, holding: BankHolding, mut bytes: u64, changing: &Changing) {
        let mut held = self.held.lock().expect("bank cache");
        let mut payers: Vec<usize> = (0..held.len()).filter(|at| held[*at].0 == holding).collect();
        if let Some(store) = payers.first().map(|at| held[*at].1.store()) {
            payers.extend((0..held.len()).filter(|at| held[*at].0 != holding && held[*at].1.store() == store));
        }
        for at in payers {
            let paid = held[at].1.releasing().min(bytes);
            held[at].1.released(paid, changing);
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
    use crate::memory::ledger::{Available, DeviceLedgers, Ledger, Ledgers, LoadShape, Need};

    const GB: u64 = 1 << 30;

    fn ledger() -> (Ledgers, Ledger) {
        let scope = Ledgers::new();
        let ledger = scope.ledger(seismic::StoreIdentity::HostRam, std::sync::Arc::new(LoadShape { stores: Vec::new() }));
        let at = std::time::Instant::now();
        ledger.read(ledger.sampled(at), Ok(Available::new(64 * GB, 64 * GB, None).expect("a reading")));
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
        let (scope, ledger) = ledger();
        let cache = BankCache::new(scope.changes().clone());
        let holding = BankHolding::Experts;
        for _ in 0..2 {
            cache.hold(holding, ledger.cached(CacheKind::Bank, 8 * GB).expect("room"));
        }
        let _taken = ledger.claim(Need::Measurement { bytes: 52 * GB }).expect("the bank yields 4 GB");
        assert_eq!((ledger.snapshot().releasing(), ledger.snapshot().allocatable()), (4 * GB, 48 * GB));
        ledger.change(|changing| cache.released(holding, 4 * GB, changing));
        let snapshot = ledger.snapshot();
        assert_eq!((snapshot.releasing(), snapshot.cache_of(CacheKind::Bank), snapshot.allocatable()), (0, 12 * GB, 52 * GB));
    }

    /// A bank's cache whose last holder is a release's work on its own
    /// thread, after the owner let go of the bank: what it holds is let go
    /// there as a change, under the process's rule, and the ledger records
    /// it.
    #[test]
    fn a_cache_last_held_off_the_owners_thread_lets_go_in_a_change() {
        let scope = Ledgers::strict();
        let _owner = scope.changes().own();
        let ledger = scope.ledger(seismic::StoreIdentity::HostRam, std::sync::Arc::new(LoadShape { stores: Vec::new() }));
        ledger.read(ledger.sampled(std::time::Instant::now()), Ok(Available::new(64 * GB, 64 * GB, None).expect("a reading")));
        let cache = Arc::new(BankCache::new(scope.changes().clone()));
        let piece = ledger.change(|changing| ledger.cache(CacheKind::Bank, 8 * GB, changing)).expect("room");
        cache.hold(BankHolding::Experts, piece);
        let release = crate::storage_thread::Releases::start().expect("a release thread");
        let (go, wait) = std::sync::mpsc::channel::<()>();
        let held = cache.clone();
        let done = release.run(crate::CompletionWake::new(|| {}), move || {
            wait.recv().expect("the owner lets go first");
            drop(held);
        });
        scope.changes().change(|_| drop(cache));
        go.send(()).expect("the release waits");
        drop(release);
        assert!(done.lock().expect("the release's outcome").as_ref().is_some_and(|(outcome, _)| outcome.is_ok()), "the release let go of the cache without a panic");
        assert_eq!(ledger.snapshot().allocated(), 0, "and the ledger recorded it");
    }

    /// A change of the owner's begun on a storage thread, which owns no
    /// memory, is the invariant's violation: its changes are steps.
    #[test]
    fn an_owners_change_on_a_storage_thread_panics() {
        let scope = Ledgers::strict();
        let changes = scope.changes().clone();
        let _owner = changes.own();
        let release = crate::storage_thread::Releases::start().expect("a release thread");
        let done = release.run(crate::CompletionWake::new(|| {}), move || changes.change(|_| ()));
        drop(release);
        let outcome = done.lock().expect("the release's outcome").take().expect("the work ran").0;
        assert!(outcome.is_err_and(|error| error.contains("an owner's change begun off its owner's thread")));
    }

    /// The ledgers of `device`, the CPU's, in `scope`, read with all their
    /// room free.
    fn device_ledgers(scope: &Ledgers) -> DeviceLedgers {
        let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
        let device = crate::test_support::open_backend(&catalog, seismic::BackendName::Cpu).expect("the CPU backend opens");
        let ledgers = DeviceLedgers::open_in(scope, &device, std::sync::Arc::new(LoadShape { stores: Vec::new() }));
        let at = std::time::Instant::now();
        for ledger in [ledgers.device(), ledgers.host()] {
            ledger.read(ledger.sampled(at), Ok(Available::new(64 * GB, 64 * GB, None).expect("a reading")));
        }
        ledgers
    }

    /// A bank opens, then is served: its host tier's backing claim, held
    /// from the open, is still the growth the served bank finds, on the
    /// ledgers of the scope it opened on.
    #[test]
    fn a_bank_served_after_its_open_finds_its_host_backing() {
        let scope = Ledgers::new();
        let ledgers = device_ledgers(&scope);
        let claim = ledgers.host().backing(8 * GB).expect("room");
        let cache = BankCache::opened(scope.changes().clone(), 4, Some((0, claim)));
        cache.served_on(&ledgers, 4);
        assert_eq!(cache.growing(), Some(BankHolding::Host));
    }

    #[test]
    #[should_panic(expected = "a bank served on ledgers of another scope")]
    fn a_bank_served_on_another_scopes_ledgers_panics() {
        let (opened, served) = (Ledgers::new(), Ledgers::new());
        let cache = BankCache::opened(opened.changes().clone(), 0, None);
        cache.served_on(&device_ledgers(&served), 0);
    }

    #[test]
    #[should_panic(expected = "a host tier opened without its backing's claim")]
    fn host_slots_without_a_claim_panic() {
        let scope = Ledgers::new();
        drop(BankCache::opened(scope.changes().clone(), 4, None));
    }

    /// A growth's claim is allocated as its slots back it; ended, it holds
    /// what it grew as cache and frees the rest.
    #[test]
    fn a_growth_is_allocated_as_it_backs() {
        let (scope, ledger) = ledger();
        let cache = BankCache::new(scope.changes().clone());
        let holding = BankHolding::Experts;
        cache.grow(holding, 10 * GB, ledger.backing(8 * GB).expect("room"));
        assert_eq!(cache.growing(), Some(holding));
        assert_eq!(ledger.snapshot().allocated(), 0, "nothing backs it yet");
        ledger.change(|changing| cache.backed(13 * GB, changing));
        assert_eq!(ledger.snapshot().allocated(), 3 * GB, "its joined slots");
        cache.settle(&BankResized::Grown { holding, from: 10, to: 15, grown: 5 * GB, refused: None, times: times() });
        let snapshot = ledger.snapshot();
        assert_eq!((snapshot.allocated(), snapshot.cache_of(CacheKind::Bank), snapshot.committed()), (5 * GB, 5 * GB, 0));
        assert_eq!(cache.growing(), None);
    }
}
