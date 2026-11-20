//! A bank's cache on its stores' ledgers: what its holdings hold above
//! their floors, in the pieces its growths took, each running growth's apart
//! as a claim of its holding, until an advance reports what it grew (an
//! expert growth runs beside the host tier's backing, so two can run). Its slots lie in
//! granules of the store's pool, whose entry the backend charges each block
//! once (`memory::ledger::Ledger::pool_block`): the pieces hold granules and
//! allocate none, a yield lowers what they hold at once, and a shrink cuts
//! them to the slots left ([`BankCache::fit`]). The one bookkeeping every
//! bank keeps of it, the served bank's and a test's alike.

use crate::domain::{BankHolding, BankResized};
use crate::memory::ledger::{CacheKind, CacheReclaimer, Cached, Changes, Claim, DeviceLedgers, Reclaim, Value};
use crate::planning::BankTier;
use std::sync::{Arc, Mutex};

pub(crate) struct BankCache {
    held: Mutex<Vec<(BankHolding, Cached)>>,
    /// What its slots are worth per byte, as the domain last valued them
    /// ([`Self::value`]): every piece holds it, a piece held later too.
    value: Mutex<Value>,
    /// The running growths, at most one per holding.
    growing: Mutex<Vec<Growing>>,
    /// What each holding owes claims that took its cache ([`Self::owed`]).
    owed: Mutex<Vec<(BankHolding, u64)>>,
    /// Synchronous reclaimers for each tier's bank cache entries, owned here
    /// (each holding only a weak reference to `ExpertBank`, so no cycle).
    reclaimers: Mutex<[Option<Arc<dyn CacheReclaimer>>; 2]>,
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
        let growing = std::mem::take(self.growing.get_mut().unwrap_or_else(std::sync::PoisonError::into_inner));
        self.changes.clone().back(|_| drop((held, growing)));
    }
}

/// A running growth: its holding and its claim (`Ledger::backing`), which
/// holds its slots' granules as the pool backs them.
struct Growing {
    holding: BankHolding,
    claim: Claim,
}

impl BankCache {
    fn tier_of(holding: BankHolding) -> Option<BankTier> {
        match holding {
            BankHolding::Experts => Some(BankTier::Bank),
            BankHolding::Host => Some(BankTier::Host),
            BankHolding::Table { .. } => None,
        }
    }

    /// A bank's cache on the ledgers of `changes`' scope, holding nothing.
    pub(crate) fn new(changes: Arc<Changes>) -> Self {
        Self {
            held: Mutex::default(),
            value: Mutex::new(Value::NONE),
            growing: Mutex::default(),
            owed: Mutex::default(),
            reclaimers: Mutex::new([None, None]),
            changes,
        }
    }

    /// Install `reclaimer` for `tier` before any cache entry of `tier` is
    /// created, so every `Bank` cache entry carries its [`Reclaim::Backed`]
    /// reclaimer from creation.
    pub(crate) fn set_reclaimer(&self, tier: BankTier, reclaimer: Arc<dyn CacheReclaimer>) {
        self.reclaimers.lock().expect("bank reclaimers")[tier.index()] = Some(reclaimer);
    }

    /// The [`Reclaim`] policy for a cache entry of `holding`: `Reclaim::Backed`
    /// with the tier's installed reclaimer for `Experts`/`Host`, `Reclaim::Deferred`
    /// for stored tables.
    pub(crate) fn reclaim_for(&self, holding: BankHolding) -> Reclaim {
        match Self::tier_of(holding) {
            Some(tier) => {
                let guard = self.reclaimers.lock().expect("bank reclaimers");
                let reclaimer = guard[tier.index()]
                    .as_ref()
                    .unwrap_or_else(|| panic!("invariant violated: {tier:?} bank cache created before its reclaimer was installed"));
                Reclaim::Backed(Arc::downgrade(reclaimer))
            }
            None => Reclaim::Deferred,
        }
    }

    /// Turn a host tier's open backing claim (`host`) into [`CacheKind::Bank`]
    /// with its installed host reclaimer immediately so host admissions can
    /// reclaim it even while `back_host` is still running.
    ///
    /// # Panics
    /// Where a host tier has no backing's claim, or a claim no host tier.
    pub(crate) fn open_host(&self, host_slots: u64, host: Option<Claim>) {
        match (host_slots > 0, host) {
            (true, Some(claim)) => {
                let reclaim = self.reclaim_for(BankHolding::Host);
                let cached = self.changes.back(|_| claim.into_backed_cache(CacheKind::Bank, reclaim));
                self.hold(BankHolding::Host, cached);
            }
            (false, None) => {}
            (true, None) => panic!("invariant violated: a host tier opened without its backing's claim"),
            (false, Some(_)) => panic!("invariant violated: a host tier's backing claimed for a bank without one"),
        }
    }

    /// The cache as a served bank takes it on `ledgers`: of the scope it
    /// opened on, a host tier of `host_slots` holding its cache entry from
    /// the open.
    ///
    /// # Panics
    /// Where either does not hold.
    pub(crate) fn served_on(&self, ledgers: &DeviceLedgers, host_slots: u64) {
        assert!(Arc::ptr_eq(&self.changes, ledgers.changes()), "invariant violated: a bank served on ledgers of another scope than it opened on");
        if host_slots > 0 {
            assert!(
                self.held.lock().expect("bank cache").iter().any(|(holding, _)| *holding == BankHolding::Host),
                "invariant violated: a host tier served without its backing's cache entry"
            );
        }
    }

    /// The changes of the scope it holds its cache in.
    pub(crate) fn changes(&self) -> &Arc<Changes> {
        &self.changes
    }

    /// `cache` held for `holding` from now: what it holds above its floor.
    pub(crate) fn hold(&self, holding: BankHolding, cache: Cached) {
        cache.value(*self.value.lock().expect("bank cache value"));
        self.held.lock().expect("bank cache").push((holding, cache));
    }

    /// Value its slots at `value` per byte (`memory::ledger::Cached::value`):
    /// the order the ledger takes cache in.
    pub(crate) fn value(&self, value: Value) {
        *self.value.lock().expect("bank cache value") = value;
        self.held.lock().expect("bank cache").iter().for_each(|(_, cached)| cached.value(value));
    }

    /// `claim` (`Ledger::backing`) taken for a growth of `holding`: it
    /// holds the granules the pool backs for its slots, until an advance
    /// reports what it grew ([`Self::settle`]).
    pub(crate) fn grow(&self, holding: BankHolding, claim: Claim) {
        let mut growing = self.growing.lock().expect("bank growth");
        assert!(growing.iter().all(|running| running.holding != holding), "invariant violated: a growth of {holding:?} began beside another of it");
        growing.push(Growing { holding, claim });
    }

    /// `holding`'s growth, where one runs, taken: its claim.
    fn take(&self, holding: BankHolding) -> Option<Growing> {
        let mut growing = self.growing.lock().expect("bank growth");
        growing.iter().position(|running| running.holding == holding).map(|at| growing.swap_remove(at))
    }

    /// The growth of `holding` held for did not begin: its claim is let go,
    /// nothing of it backed.
    pub(crate) fn abandon(&self, holding: BankHolding) {
        let growing = self.take(holding);
        self.changes.change(|_| drop(growing));
    }

    /// Whether a growth of `holding` runs.
    pub(crate) fn growing(&self, holding: BankHolding) -> bool {
        self.growing.lock().expect("bank growth").iter().any(|running| running.holding == holding)
    }

    /// What `holding`, holding `above` bytes of slots above its floor,
    /// owes claims of the process: what they took of its pieces in process
    /// (`Cached::take_claimed`), which they allocate only once its slots
    /// free it, at most what its slots hold past its cache (its pieces and
    /// its running growth's claim), so none once a shrink paid it. Its
    /// target sheds it whatever the room's threshold says
    /// (`experts::derived::held_target`); what a shortage took of it is no
    /// part of it.
    pub(crate) fn owed(&self, holding: BankHolding, above: u64) -> u64 {
        let held = self.held.lock().expect("bank cache");
        self.settle_owed(&held, holding, above)
    }

    /// [`Self::owed`], with `held` the pieces: what claims took of them
    /// since read joins what `holding` owes, which its slots past its cache
    /// bound.
    fn settle_owed(&self, held: &[(BankHolding, Cached)], holding: BankHolding, above: u64) -> u64 {
        let pieces = held.iter().filter(|(of, _)| *of == holding);
        let claimed = pieces.clone().map(|(_, cached)| cached.take_claimed()).sum::<u64>();
        let cached = pieces.map(|(_, cached)| cached.bytes()).sum::<u64>()
            + self.growing.lock().expect("bank growth").iter().filter(|running| running.holding == holding).map(|running| running.claim.bytes()).sum::<u64>();
        let mut owed = self.owed.lock().expect("bank owed");
        let at = match owed.iter().position(|(of, _)| *of == holding) {
            Some(at) => at,
            None => {
                owed.push((holding, 0));
                owed.len() - 1
            }
        };
        owed[at].1 = (owed[at].1 + claimed).min(above.saturating_sub(cached));
        owed[at].1
    }

    /// A shrink of `holding` left it `above` bytes of slots above its floor:
    /// its pieces hold no more, the latest cut first. What yields took of
    /// them they hold no more already; what claims took is read before any
    /// piece is cut, and what the holding still owes is what its slots left
    /// hold past its cache ([`Self::owed`]).
    pub(crate) fn fit(&self, holding: BankHolding, above: u64) {
        let mut held = self.held.lock().expect("bank cache");
        let claimed = held.iter().filter(|(of, _)| *of == holding).map(|(_, cached)| cached.take_claimed()).sum::<u64>();
        if claimed > 0 {
            let mut owed = self.owed.lock().expect("bank owed");
            match owed.iter_mut().find(|(of, _)| *of == holding) {
                Some((_, owes)) => *owes += claimed,
                None => owed.push((holding, claimed)),
            }
        }
        let mut holds = held.iter().filter(|(of, _)| *of == holding).map(|(_, cached)| cached.bytes()).sum::<u64>();
        let mut cut = Vec::new();
        while holds > above {
            let at = held.iter().rposition(|(of, _)| *of == holding).expect("a holding holding bytes has a piece");
            let has = held[at].1.bytes();
            let less = (holds - above).min(has);
            match less == has {
                true => cut.push(held.remove(at).1),
                false => cut.push(held[at].1.split(less, CacheKind::Bank)),
            }
            holds -= less;
        }
        self.settle_owed(&held, holding, above);
        drop(held);
        self.changes.back(|_| drop(cut));
    }

    /// The cache of the resize an advance reports finished: a growth's
    /// claim turns cache of the granules it holds, one stopped or refused
    /// frees it all. A shrink cut the pieces as it freed its slots
    /// ([`Self::fit`]). A host tier's open backing (`back_host`) already
    /// holds its cache entry in `held`, so settling it without a `Growing`
    /// claim leaves that entry in place.
    pub(crate) fn settle(&self, resized: &BankResized) {
        match *resized {
            BankResized::Shrunk { .. } => {}
            BankResized::Grown { holding, .. } => {
                if let Some(growing) = self.take(holding) {
                    let reclaim = self.reclaim_for(holding);
                    let cache = self.changes.back(|_| growing.claim.into_backed_cache(CacheKind::Bank, reclaim));
                    self.hold(holding, cache);
                }
            }
            // What its slots backed is freed with them.
            BankResized::Stopped { holding, .. } | BankResized::GrowthRefused { holding, .. } => {
                if self.growing(holding) {
                    self.abandon(holding);
                }
            }
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::fixtures::NO_LAG;
    use crate::memory::ledger::{DeviceLedgers, Ledger, Ledgers, LoadShape};
    use crate::memory::supply::Reading;

    const GB: u64 = 1 << 30;

    /// Host RAM's ledger in a scope of its own, read with 64 GB free, its
    /// pool open with blocks of 1 GB.
    fn ledger() -> (Ledgers, Ledger, std::sync::Arc<dyn seismic::memory::Entry>) {
        let scope = Ledgers::new();
        let ledger = scope.ledger(seismic::StoreIdentity::HostRam, std::sync::Arc::new(LoadShape { stores: Vec::new() }));
        let at = std::time::Instant::now();
        ledger.read(ledger.sampled(at), Reading::device(64 * GB, 64 * GB, NO_LAG));
        let pool = ledger.pool_block(crate::memory::ledger::PoolClass::Records, std::num::NonZeroU64::new(GB).expect("a block")).blocks;
        (scope, ledger, pool)
    }

    fn times() -> crate::domain::ResizeTimes {
        let now = std::time::Instant::now();
        let zero = std::time::Duration::ZERO;
        crate::domain::ResizeTimes {
            decided: now,
            queued: zero,
            planned: zero,
            trips: Default::default(),
            prepared: now,
            installed: now,
            released: now,
        }
    }

    /// A bank's cache whose last holder is a release's work on its own
    /// thread, after the owner let go of the bank: what it holds is let go
    /// there as a change, under the process's rule, and the ledger records
    /// it: its granules are the pool's spare, allocated still until the
    /// pool returns its blocks.
    #[test]
    fn a_cache_last_held_off_the_owners_thread_lets_go_in_a_change() {
        let scope = Ledgers::strict();
        let _owner = scope.changes().own();
        let ledger = scope.ledger(seismic::StoreIdentity::HostRam, std::sync::Arc::new(LoadShape { stores: Vec::new() }));
        ledger.read(ledger.sampled(std::time::Instant::now()), Reading::device(64 * GB, 64 * GB, NO_LAG));
        let cache = Arc::new(BankCache::new(scope.changes().clone()));
        let pool = ledger.pool_block(crate::memory::ledger::PoolClass::Records, std::num::NonZeroU64::new(GB).expect("a block")).blocks;
        let piece = ledger.cached(CacheKind::Bank, 8 * GB).expect("room");
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
        assert!(done.lock().expect("the release's outcome").is_some(), "the release let go of the cache");
        let estimate = ledger.estimate();
        assert_eq!(estimate.cache_of(CacheKind::Bank), 0, "and the ledger recorded it");
        assert_eq!((estimate.cache_of(CacheKind::Spare), estimate.allocated()), (8 * GB, 8 * GB), "its granules spare in the pool");
        // The pool returns its blocks as it closes: a free of its own, as
        // the process makes it.
        scope.changes().change(|_| drop(pool));
        assert_eq!(ledger.estimate().allocated(), 0, "the pool's close freed its granules");
    }

    /// The ledgers of `device`, the CPU's, in `scope`, read with all their
    /// room free.
    fn device_ledgers(scope: &Ledgers) -> DeviceLedgers {
        let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
        let device = crate::test_support::open_backend(&catalog, seismic::BackendName::Cpu).expect("the CPU backend opens");
        DeviceLedgers::open_in(scope, &device, std::sync::Arc::new(LoadShape { stores: Vec::new() })).read_scripted(Reading::device(64 * GB, 64 * GB, NO_LAG)).0
    }

    struct NoopReclaimer;
    impl CacheReclaimer for NoopReclaimer {
        fn reclaim(&self, _need: u64) -> Result<u64, crate::memory::ledger::ReclaimError> {
            Err(crate::memory::ledger::ReclaimError::AtFloor)
        }
    }

    fn cache_with_reclaimers(changes: Arc<Changes>) -> BankCache {
        let cache = BankCache::new(changes);
        cache.set_reclaimer(BankTier::Bank, Arc::new(NoopReclaimer));
        cache.set_reclaimer(BankTier::Host, Arc::new(NoopReclaimer));
        cache
    }

    /// A bank opens, then is served: its host tier's backing claim, held
    /// from the open as `CacheKind::Bank`, is found on the ledgers of the
    /// scope it opened on.
    #[test]
    fn a_bank_served_after_its_open_finds_its_host_backing() {
        let scope = Ledgers::new();
        let ledgers = device_ledgers(&scope);
        let claim = ledgers.host().change(|keeping| ledgers.host().backing(&keeping.claims(), crate::memory::ledger::PoolClass::Records, 8 * GB)).expect("room");
        let cache = cache_with_reclaimers(scope.changes().clone());
        cache.open_host(4, Some(claim));
        cache.served_on(&ledgers, 4);
        assert_eq!(ledgers.host().estimate().cache_of(CacheKind::Bank), 8 * GB);
    }

    /// An expert growth begins beside the host tier's backing, held as
    /// `CacheKind::Bank` from the open, and settles independently.
    #[test]
    fn an_expert_growth_runs_beside_the_host_backing() {
        let scope = Ledgers::new();
        let ledgers = device_ledgers(&scope);
        let backing = ledgers.host().change(|keeping| ledgers.host().backing(&keeping.claims(), crate::memory::ledger::PoolClass::Records, 8 * GB)).expect("room");
        let cache = cache_with_reclaimers(scope.changes().clone());
        cache.open_host(4, Some(backing));
        let claim = ledgers.of(crate::planning::BankTier::Bank).change(|keeping| ledgers.of(crate::planning::BankTier::Bank).backing(&keeping.claims(), crate::memory::ledger::PoolClass::Records, GB)).expect("room");
        cache.grow(BankHolding::Experts, claim);
        assert!(cache.growing(BankHolding::Experts));
        cache.settle(&BankResized::Stopped { holding: BankHolding::Experts, at: 10 });
        assert!(!cache.growing(BankHolding::Experts));
        assert_eq!(ledgers.host().estimate().cache_of(CacheKind::Bank), 8 * GB);
    }

    #[test]
    #[should_panic(expected = "a bank served on ledgers of another scope")]
    fn a_bank_served_on_another_scopes_ledgers_panics() {
        let (opened, served) = (Ledgers::new(), Ledgers::new());
        let cache = cache_with_reclaimers(opened.changes().clone());
        cache.open_host(0, None);
        cache.served_on(&device_ledgers(&served), 0);
    }

    #[test]
    #[should_panic(expected = "a host tier opened without its backing's claim")]
    fn host_slots_without_a_claim_panic() {
        let scope = Ledgers::new();
        let cache = cache_with_reclaimers(scope.changes().clone());
        cache.open_host(4, None);
    }

    /// A growth's claim holds its slots' granules as the pool backs them,
    /// charging the pool's entry; ended, it holds them as cache.
    #[test]
    fn a_growth_holds_what_its_slots_backed() {
        let (scope, ledger, pool) = ledger();
        let cache = cache_with_reclaimers(scope.changes().clone());
        let holding = BankHolding::Experts;
        let claim = ledger.change(|keeping| ledger.backing(&keeping.claims(), crate::memory::ledger::PoolClass::Records, 3 * GB)).expect("room");
        assert_eq!(ledger.estimate().allocated(), 0, "nothing backs it yet");
        ledger.change(|_| { let entry = &pool; entry.take(&entry.change(), 3 * GB, seismic::memory::Wiring::Pageable) }).expect("the pool backs its slots");
        cache.grow(holding, claim);
        assert!(cache.growing(holding));
        assert_eq!(ledger.estimate().allocated(), 3 * GB, "its backed slots, the pool's allocation");
        cache.settle(&BankResized::Grown { holding, from: 10, to: 13, grown: 3 * GB, refused: None, times: times() });
        let estimate = ledger.estimate();
        assert_eq!((estimate.allocated(), estimate.cache_of(CacheKind::Bank), estimate.committed()), (3 * GB, 3 * GB, 0));
        assert!(!cache.growing(holding));
    }
}
