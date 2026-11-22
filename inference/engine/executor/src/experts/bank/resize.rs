//! The bank's resizes over its stores' granule pools. Each store's records
//! (the bank's own, and a linked host tier's) are one slab store per group
//! in that store's pool, a record a slab: one granule holds an expert's
//! record of every routed tensor. A growth claims granules (a free one in
//! process first, else the pool grows by a block its ledger entry admits),
//! slot by slot on the store's resize thread, each slot joining the watcher
//! as soon as every group holds it. A shrink evicts the records the watcher
//! chooses, beside running steps or not, and moves nothing: their slabs
//! return to the pool once every pass that may read them is submitted, each
//! granule free again once the last flight that read it completes; then the
//! pool returns its empty blocks.

use super::ExpertBank;
use crate::domain::{BankHolding, BankResized, BankStep, DomainError, ResizeTimes, RunningBegun, RunningPressure, Trip};
use crate::experts::source::Records;
use crate::experts::watcher::{Candidates, Evicted, Evicting, Join, ScratchGroup};
use crate::memory::ledger::{CacheReclaimer, Changes, Ledger, ReclaimError};
use crate::planning::{BankTier, Placer};
use crate::storage_thread::{Done, StorageThread};
use crate::CompletionWake;
use seismic::{Pool, SlabTensor, Tensor, TensorError};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant};

/// Waiter-priority gate paired with the bank's `rises: Mutex<()>` so
/// domain-thread observations ([`ExpertBank::hold_rises`]) are never starved
/// behind consecutive background growth slots ([`grow_by_slots`]) under an
/// unfair OS mutex.
///
/// Lock ordering: `priority_waiters` is never held while acquiring `rises`,
/// and `WaiterGuard` always decrements `priority_waiters` and wakes waiters on
/// drop (including unwind), so deadlock is impossible.
pub(super) struct RiseGate {
    priority_waiters: Mutex<usize>,
    cleared: Condvar,
}

impl RiseGate {
    pub(super) fn new() -> Self {
        Self { priority_waiters: Mutex::new(0), cleared: Condvar::new() }
    }

    /// Acquire `rises` with priority over background callers: while this
    /// caller waits for `rises`, background callers yield instead of
    /// re-acquiring `rises` for another slot.
    pub(super) fn hold_priority<'a>(&self, rises: &'a Mutex<()>) -> MutexGuard<'a, ()> {
        struct WaiterGuard<'g>(&'g RiseGate);
        impl Drop for WaiterGuard<'_> {
            fn drop(&mut self) {
                let mut waiters = self.0.priority_waiters.lock().expect("rise gate priority waiters");
                *waiters -= 1;
                if *waiters == 0 {
                    self.0.cleared.notify_all();
                }
            }
        }
        {
            let mut waiters = self.priority_waiters.lock().expect("rise gate priority waiters");
            *waiters += 1;
        }
        let _waiter = WaiterGuard(self);
        rises.lock().expect("the bank's charge rises")
    }

    /// Acquire `rises` as a background caller, yielding whenever any priority
    /// caller is waiting so the priority caller acquires `rises` on the very
    /// next release.
    pub(super) fn hold_background<'a>(&self, rises: &'a Mutex<()>) -> MutexGuard<'a, ()> {
        {
            let mut waiters = self.priority_waiters.lock().expect("rise gate priority waiters");
            while *waiters > 0 {
                waiters = self.cleared.wait(waiters).expect("rise gate priority waiters");
            }
        }
        rises.lock().expect("the bank's charge rises")
    }

    #[cfg(test)]
    fn priority_waiters(&self) -> usize {
        *self.priority_waiters.lock().expect("rise gate priority waiters")
    }
}

/// One store's records: each group's slab store in the store's pool, a
/// record a slab, its leading `lent` records over the lender's lent
/// allocation for the load's life, each region one routed tensor's record
/// of `record_bytes`. `SlabTensor` provides interior mutability so no outer
/// per-group mutex is ever held across `add_slab` / `allocate_block`.
pub(super) struct Stores {
    pool: Pool,
    block: seismic::BlockSize,
    groups: Vec<SlabTensor>,
    lent: Vec<u64>,
    record_bytes: Vec<Vec<u64>>,
    rise_gate: Arc<RiseGate>,
}

impl Stores {
    /// `groups`, each a store with its leading `lent` records and its
    /// regions' record bytes, in `pool` whose blocks are sized by `block`.
    pub(super) fn new(pool: Pool, block: seismic::BlockSize, groups: Vec<(SlabTensor, u64, Vec<u64>)>) -> Self {
        let (mut stores, mut lent, mut record_bytes) = (Vec::new(), Vec::new(), Vec::new());
        for (store, leading, bytes) in groups {
            stores.push(store);
            lent.push(leading);
            record_bytes.push(bytes);
        }
        Self { pool, block, groups: stores, lent, record_bytes, rise_gate: Arc::new(RiseGate::new()) }
    }

    /// Waiter-priority gate paired with the bank's `rises` mutex.
    pub(super) fn rise_gate(&self) -> &Arc<RiseGate> {
        &self.rise_gate
    }

    fn group(&self, group: usize) -> &SlabTensor {
        &self.groups[group]
    }

    /// The records group `group` holds: its slabs, lent ones included.
    pub(super) fn present(&self, group: usize) -> u64 {
        self.group(group).backed_count() as u64
    }

    /// What the pool charges the records: whole blocks (`BlockSize::charge`)
    /// holding a granule for each record beyond the lent ones (the same
    /// formula as `ExpertBankPlan::resident_charge` and `host_charge`).
    pub(super) fn charged(&self) -> u64 {
        let granules: u64 = (0..self.groups.len()).map(|group| self.present(group) - self.lent[group]).sum();
        self.block
            .charge(self.pool.granule(), granules)
            .expect("invariant violated: bank record charge overflows")
    }

    /// The tensor kernels bind as group `group`'s region `region`: its
    /// store's view, its record table, which every record held now or
    /// later is reached through.
    pub(super) fn region(&self, group: usize, region: usize) -> Result<Tensor, TensorError> {
        self.group(group).logical_region(region)
    }

    /// Claim records of group `group` until it holds `records`, each a
    /// granule of the pool (`SlabTensor::add_slabs`: free ones in process,
    /// else the pool grows by blocks its ledger entry admits), unwritten:
    /// claimed records join the watcher free (`Directory::grow` /
    /// `Directory::grow_host`) and are never published to the device until a
    /// promotion copy or file read writes every byte and completes. The
    /// records claimed, the lowest free first; on a refusal, none (those
    /// claimed before it are returned cleanly, never having been published).
    pub(super) fn claim_to(&self, group: usize, records: u64) -> Result<Vec<usize>, TensorError> {
        seismic::memory::assert_blocking_allocation_allowed("Stores::claim_to");
        let store = self.group(group);
        let have = store.backed_count() as u64;
        let need = usize::try_from(records.saturating_sub(have))
            .map_err(|_| TensorError::SlabLayout("requested records exceed host domain".into()))?;
        store.add_slabs(need, seismic::SlabFill::Unwritten)
    }

    /// Free `claimed` of group `group`, which nothing read: a refusal
    /// breaks the pool's invariant.
    fn unclaim(&self, group: usize, claimed: &[usize]) {
        Self::unclaim_in(self.group(group), group, claimed);
    }

    fn unclaim_in(store: &SlabTensor, group: usize, claimed: &[usize]) {
        for &record in claimed {
            if let Err(error) = store.free_slab(record) {
                if matches!(error, TensorError::Execution(seismic::ExecutionError::DeviceLost(_))) {
                    tracing::warn!("bank rollback stopped freeing claimed slabs of group {group} after device loss: {error}");
                    return;
                }
                panic!("invariant violated: the bank's pool refused to free record {record} of group {group}, claimed and never read: {error}");
            }
        }
    }

    /// Return `records` of group `group` to the pool: each granule free
    /// once the last flight that read it completes. Never waits.
    pub(super) fn free(&self, group: usize, records: &[u32]) -> Result<(), TensorError> {
        seismic::memory::assert_blocking_allocation_allowed("Stores::free");
        let store = self.group(group);
        records.iter().try_for_each(|&record| store.free_slab(record as usize).map(drop))
    }

    /// Return `freed` bytes of the pool to the OS, what a shrink freed:
    /// every empty block, then the emptiest emptied while those fall short
    /// (`seismic::Pool::return_freed`). The bytes returned.
    fn return_freed(&self, freed: u64) -> Result<u64, TensorError> {
        seismic::memory::assert_blocking_allocation_allowed("Stores::return_freed");
        self.pool.return_freed(freed)
    }

    /// Record `record` of group `group`'s region `region`, while it is
    /// held. Safe inside a no-blocking-allocation scope: `Stores::groups`
    /// holds no outer mutex and `SlabTensor::slab_region` acquires only the
    /// `Slots::slabs` leaf lock (never held across allocation or submission).
    pub(super) fn view(&self, group: usize, region: usize, record: usize) -> Option<Tensor> {
        self.group(group).slab_region(record, region).expect("a record and region lie in their group's store")
    }

    /// Each group's routed tensors' records the host addresses, resolved on
    /// demand from the live slab store so idle slabs hold no view and the
    /// pool can vacate their blocks on a shrink.
    pub(super) fn records(self: &Arc<Self>) -> Result<Vec<Vec<Records>>, String> {
        Ok((0..self.groups.len())
            .map(|group| {
                let regions = self.group(group).region_count();
                (0..regions)
                    .map(|region| {
                        Records::pooled(
                            Arc::new(StoreRegion { stores: self.clone(), group, region }),
                            self.record_bytes[group][region],
                        )
                    })
                    .collect()
            })
            .collect())
    }
}

/// On-demand record resolver for one `(group, region)` of [`Stores`].
struct StoreRegion {
    stores: Arc<Stores>,
    group: usize,
    region: usize,
}

impl crate::experts::source::RecordStore for StoreRegion {
    fn view(&self, record: u64) -> Option<Tensor> {
        let record = usize::try_from(record).ok()?;
        self.stores.view(self.group, self.region, record)
    }

    fn count(&self) -> u64 {
        self.stores.group(self.group).backed().last().map_or(0, |record| record + 1) as u64
    }
}

/// What a store commits: the slots its records serve, its resize lifecycle
/// ([`Resize`]), and any running shrink beginning owed to the domain.
pub(super) struct Commitment {
    /// Written by whoever resizes: a shrink on the submitting thread, a
    /// growth's slots on the resize thread as each joins.
    pub(super) slots: Arc<AtomicU64>,
    pub(super) resize: Resize,
    /// A shrink begun beside a running chain the domain has not yet taken
    /// ([`ExpertBank::take_running_begun`]).
    pub(super) running: Option<RunningBegun>,
    /// Serializes concurrent [`ExpertBank::reclaim`] calls on this tier
    /// outside the commitment lock. Never acquired by domain resize methods.
    reclaim_lock: Arc<Mutex<()>>,
    /// True while [`ExpertBank::reclaim`] is executing on this tier, keeping
    /// [`Self::resizing`] true across its lock-free blocking waits so no
    /// concurrent domain `grow`/`shrink`/`release_running` can race with it.
    reclaiming: bool,
}

impl Commitment {
    pub(super) fn new(slots: Arc<AtomicU64>) -> Self {
        Self { slots, resize: Resize::Idle, running: None, reclaim_lock: Arc::new(Mutex::new(())), reclaiming: false }
    }

    /// Whether a resize of the store runs until an advance finishes it: a
    /// synchronous reclaim, a growth, a shrink's release, or an outcome not
    /// yet reported.
    fn resizing(&self) -> bool {
        self.reclaiming || !matches!(self.resize, Resize::Idle | Resize::Broken(_))
    }
}

/// A linked host tier: its records' stores, its host slots committed as
/// [`Commitment`] commits the bank's, the most it holds (every expert,
/// `ExpertBankPlan::host_ceiling`), and the thread its growth runs on: host
/// RAM is a store of its own, so its growth never queues a resize of the
/// device's records behind it, nor waits on one.
pub(super) struct HostTier {
    pub(super) stores: Arc<Stores>,
    pub(super) commitment: Commitment,
    pub(super) ceiling: u64,
    pub(super) thread: StorageThread,
    /// Its host slots' last finished resize
    /// ([`crate::domain::ElasticBank::record_resize`]).
    pub(super) last_resize: Option<crate::domain::BankResize>,
}

/// A store's resize lifecycle: idle, a growth on its resize thread (with
/// `backing` true for a host tier's open backing to its planned slots,
/// which no domain resize began), a shrink's evicted records waiting for
/// the passes that may read them, a finished outcome not yet reported to
/// the domain, or broken (a resize step panicked or broke an invariant:
/// none runs again, and each asked fails with why).
pub(super) enum Resize {
    Idle,
    Growing { growing: Growing, backing: bool },
    Releasing(Releasing),
    Reported(BankResized),
    Broken(String),
}

impl Resize {
    fn take_growing(&mut self) -> Option<(Growing, bool)> {
        match std::mem::replace(self, Self::Idle) {
            Self::Growing { growing, backing } => Some((growing, backing)),
            other => {
                *self = other;
                None
            }
        }
    }

    fn take_releasing(&mut self) -> Option<Releasing> {
        match std::mem::replace(self, Self::Idle) {
            Self::Releasing(releasing) => Some(releasing),
            other => {
                *self = other;
                None
            }
        }
    }

    fn take_reported(&mut self) -> Option<BankResized> {
        match std::mem::replace(self, Self::Idle) {
            Self::Reported(resized) => Some(resized),
            other => {
                *self = other;
                None
            }
        }
    }
}

/// A shrink's evicted records, each group's, whose storage returns once
/// every pass that may read them is submitted: beside the running chain
/// whose last step is `last`, every pass of block `after` (at once where
/// none: its steps had decided no hit of them), and at any boundary between
/// flights. From `from` slots to `to`, the records charged `charged` before.
pub(super) struct Releasing {
    from: u64,
    to: u64,
    charged: u64,
    evicted: Vec<Vec<u32>>,
    last: u64,
    after: Option<usize>,
    begun: Begun,
}

/// A growth of a store's records from `from` slots, the records charged
/// `charged` before it, decided at `decided`: on the resize thread, slot by
/// slot toward its target, each joined to the watcher at once
/// ([`grow_by_slots`]), so its first slots serve while later ones are still
/// claimed, and nothing waits for a gap between flights. `stop` ends it at
/// its next slot.
pub(super) struct Growing {
    pub(super) from: u64,
    pub(super) charged: u64,
    pub(super) decided: Instant,
    pub(super) stop: Arc<AtomicBool>,
    pub(super) done: Done<Grown>,
}

/// How a growth ended on the resize thread: the slots reached, when its
/// first and its last slot joined (none where none did), what cut it short
/// of its target, if anything did, and where its time went.
pub(super) struct Grown {
    pub(super) at: u64,
    pub(super) joined: Option<(Instant, Instant)>,
    pub(super) cut: Cut,
    pub(super) spent: Spent,
}

/// A growth's time on the resize thread by phase, over its slots: each
/// slot's claims (a pool block's allocation among them, with the wait for
/// the bank's charge lock) and its join posted.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Spent {
    slots: u64,
    claim: Duration,
    join: Duration,
}

impl std::fmt::Display for Spent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mean = |phase: Duration| phase.as_secs_f64() * 1e3 / self.slots.max(1) as f64;
        write!(formatter, "a slot's mean over {}: claim {:.3} ms, join {:.3} ms", self.slots, mean(self.claim), mean(self.join))
    }
}

/// What ended a growth short of its target.
pub(super) enum Cut {
    /// Nothing: it reached its target, or was stopped (`Growing::stop`).
    None,
    /// The pool refused its next slot's storage (its ledger entry admitted
    /// no block), which joined nothing: what its groups claimed of it was
    /// freed at once.
    Refused(TensorError),
    /// Its next slot's join could not be formed: a broken invariant.
    Unjoined(String),
}

/// What a growth's slots join as each is claimed (`watcher::Join`): the
/// bank's records, which the device places into, or records the host lands
/// into, addressed by their views.
#[derive(Clone, Copy)]
pub(super) enum Joining {
    Bank,
    Landed,
}

/// A store's records of group `group` at `slots` slots: the bank's
/// (`BankCounts::records_at`) or a host tier's (`BankCounts::host_records`).
pub(super) type RecordsAt = Arc<dyn Fn(usize, u64) -> u64 + Send + Sync>;

/// Grow `stores` from `from` to `to` slots, one slot at a time unless
/// `stop` is set: each group claims the records it lacks at the slot
/// (`records(group, slot)`), holding `rises` while it does (the records'
/// charge rises, [`ExpertBank::hold_rises`]), its granules one change of
/// `changes`' scope (the pool's blocks charged to its ledger entry,
/// [`Changes::back`]); then the slot joins the
/// watcher (`post`), as `joining` says, and `slots` holds it. A refusal
/// ends it at the slots joined before. Each slot begins by `begin`, held
/// until it is joined, none where it gives none (the load's traffic is
/// held, `watcher::Shared::backing_slot`).
#[allow(clippy::too_many_arguments)]
pub(super) fn grow_by_slots<S>(
    stores: &Arc<Stores>,
    (from, to): (u64, u64),
    records: &RecordsAt,
    joining: Joining,
    changes: &Arc<Changes>,
    post: impl Fn(Join),
    (slots, rises): (&AtomicU64, &Mutex<()>),
    stop: &AtomicBool,
    begin: impl Fn() -> Option<S>,
) -> Grown {
    grow_by_slots_gated(stores, (from, to), records, joining, changes, post, (slots, rises, None), stop, begin, None)
}

#[allow(clippy::too_many_arguments)]
fn grow_by_slots_gated<S>(
    stores: &Arc<Stores>,
    (from, to): (u64, u64),
    records: &RecordsAt,
    joining: Joining,
    changes: &Arc<Changes>,
    post: impl Fn(Join),
    (slots, rises, rise_gate): (&AtomicU64, &Mutex<()>, Option<&RiseGate>),
    stop: &AtomicBool,
    begin: impl Fn() -> Option<S>,
    can_allocate: Option<&dyn Fn(u64) -> bool>,
) -> Grown {
    let groups = stores.groups.len();
    let mut reached = from;
    let mut joined: Option<(Instant, Instant)> = None;
    let mut spent = Spent::default();
    // Pooled records resolve newly claimed slabs on demand, holding no
    // persistent views.
    let views = match joining {
        Joining::Bank => None,
        Joining::Landed => match stores.records() {
            Ok(views) => Some(views),
            Err(reason) => return Grown { at: reached, joined, cut: Cut::Unjoined(reason), spent },
        },
    };
    for at in from + 1..=to {
        let Some(_slot) = begin().filter(|_| !stop.load(Ordering::Acquire)) else {
            break;
        };
        if let Some(check) = can_allocate {
            if !check(at) {
                break;
            }
        }
        spent.slots += 1;
        let claiming = Instant::now();
        // Acquire `rises` before `changes.back` (matching `domain::observe`
        // order), yielding through `rise_gate` whenever a priority caller
        // (`ExpertBank::hold_rises`) is waiting so background growth never
        // starves domain observations. `SlabTensor::add_slab` grows the pool
        // outside slab/pool locks via single-flight `GrowGate` when no granule
        // is free.
        let claimed = {
            let _rises = match rise_gate {
                Some(gate) => gate.hold_background(rises),
                None => rises.lock().expect("the bank's charge rises"),
            };
            changes.back(|_| {
                let mut claimed: Vec<Vec<usize>> = Vec::with_capacity(groups);
                for group in 0..groups {
                    match stores.claim_to(group, records(group, at)) {
                        Ok(records) => claimed.push(records),
                        Err(refusal) => {
                            // The slot joins nothing: what its groups claimed
                            // before returns at once, never read.
                            for (group, records) in claimed.iter().enumerate() {
                                stores.unclaim(group, records);
                            }
                            return Err(refusal);
                        }
                    }
                }
                Ok(claimed)
            })
        };
        let claimed = match claimed {
            Ok(claimed) => claimed,
            Err(refusal) => return Grown { at: reached, joined, cut: Cut::Refused(refusal), spent },
        };
        let joining_at = Instant::now();
        spent.claim += joining_at - claiming;
        let join = match &views {
            None => Join::Bank { records: claimed },
            Some(views) => {
                let records = claimed
                    .iter()
                    .map(|records| {
                        records.iter().map(|&record| u32::try_from(record).map_err(|_| "landed records exceed u32".to_owned())).collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<Vec<_>, _>>();
                match records {
                    Ok(records) => Join::Landed { records, views: views.clone() },
                    Err(reason) => return Grown { at: reached, joined, cut: Cut::Unjoined(reason), spent },
                }
            }
        };
        post(join);
        spent.join += joining_at.elapsed();
        slots.store(at, Ordering::Release);
        reached = at;
        let now = Instant::now();
        joined = Some((joined.map_or(now, |(first, _)| first), now));
    }
    Grown { at: reached, joined, cut: Cut::None, spent }
}

impl Growing {
    /// Its end, once the resize thread finished it: what it came to for
    /// `holding`, its records' charge read by `charged` after the outcome is
    /// acquired, and where its time went; or why the store is broken (a
    /// slot claimed could not join).
    pub(super) fn ended(&self, holding: BankHolding, charged: impl FnOnce() -> u64) -> Option<Result<(BankResized, Spent), String>> {
        let done = self.done.lock().expect("bank growth outcome").take()?;
        Some(self.finish(holding, done, charged))
    }

    pub(super) fn finish(&self, holding: BankHolding, (Grown { at, joined, cut, spent }, released): (Grown, Instant), charged: impl FnOnce() -> u64) -> Result<(BankResized, Spent), String> {
        let refused = match cut {
            Cut::None => None,
            Cut::Refused(refusal) => Some(refusal),
            Cut::Unjoined(reason) => return Err(reason),
        };
        // No slot joined: refused at its first, or stopped before it.
        let Some((first, last)) = joined else {
            return Ok(match refused {
                Some(refusal) => (BankResized::GrowthRefused { holding, at, refusal }, spent),
                None => (BankResized::Stopped { holding, at }, spent),
            });
        };
        let Some(grown) = charged().checked_sub(self.charged) else {
            return Err("a growth lowered its records' charge".to_owned());
        };
        let zero = Duration::ZERO;
        let begun = Begun { decided: self.decided, queued: zero, planned: zero, trip: Trip::default() };
        let times = begun.times(first, last, released);
        Ok((BankResized::Grown { holding, from: self.from, to: at, grown, refused, times }, spent))
    }
}

/// How a resize began on the submitting thread (`ResizeTimes`).
#[derive(Clone, Copy)]
struct Begun {
    decided: Instant,
    queued: Duration,
    planned: Duration,
    trip: Trip,
}

impl Begun {
    fn times(self, prepared: Instant, installed: Instant, released: Instant) -> ResizeTimes {
        let Self { decided, queued, planned, trip } = self;
        ResizeTimes { decided, queued, planned, trips: [trip, Trip::default()], prepared, installed, released }
    }
}

impl ExpertBank {
    pub(super) fn commitment(&self) -> MutexGuard<'_, Commitment> {
        self.commitment.lock().expect("bank commitment")
    }

    fn host(&self) -> Option<MutexGuard<'_, HostTier>> {
        self.host.as_ref().map(|host| host.lock().expect("host tier"))
    }

    /// The records' charge: the pool's granules the bank's stores hold.
    pub(super) fn charged(&self) -> u64 {
        self.stores.charged()
    }

    /// The host tier's records' charge: its pool's granules they hold.
    pub(super) fn host_charged(&self) -> u64 {
        self.host().map_or(0, |tier| tier.stores.charged())
    }

    /// What `tier`'s records commit by slots.
    fn tier_counts(&self, tier: BankTier) -> RecordsAt {
        let counts = self.counts.clone();
        match tier {
            BankTier::Bank => Arc::new(move |group, slots| counts.records_at(group, slots)),
            BankTier::Host => Arc::new(move |group, slots| counts.host_records(group, slots)),
        }
    }

    /// What `tier`'s records are charged at their slots.
    fn tier_charge(&self, tier: BankTier) -> fn(&crate::planning::BankCounts, u64) -> u64 {
        match tier {
            BankTier::Bank => crate::planning::BankCounts::resident_charge,
            BankTier::Host => crate::planning::BankCounts::host_charge,
        }
    }

    /// What `tier`'s records are charged above their floor at `slots`: what
    /// the bank's cache holds of them at most (`BankCache::fit`). The
    /// bank's floor is its least slots, a host tier's none.
    fn above_floor(&self, tier: BankTier, slots: u64) -> u64 {
        let charge = self.tier_charge(tier);
        charge(&self.counts, slots).saturating_sub(charge(&self.counts, self.floor(tier)))
    }

    /// The least slots `tier`'s records shrink to: the bank's floor, a host
    /// tier's none.
    pub(super) fn floor(&self, tier: BankTier) -> u64 {
        match tier {
            BankTier::Bank => self.counts.floor(),
            BankTier::Host => 0,
        }
    }

    /// The candidate shrinks of `tier`'s records from `shallowest` slots
    /// down to `deepest` ([`Candidates`]): each group's records at a
    /// candidate's slots.
    fn candidates(&self, tier: BankTier, (shallowest, deepest): (u64, u64)) -> Result<Candidates, DomainError> {
        assert!(deepest <= shallowest, "a shrink's deepest cut {deepest} lies below its shallowest {shallowest}");
        let slots = |slots: u64| usize::try_from(slots).map_err(|_| DomainError::invariant("bank slots exceed the host"));
        let (records, groups) = (self.tier_counts(tier), self.plan.groups().len());
        Ok(Candidates {
            shallowest: slots(shallowest)?,
            deepest: slots(deepest)?,
            records: Box::new(move |slots| (0..groups).map(|group| records(group, slots as u64) as usize).collect()),
        })
    }

    /// Which records the watcher evicts of `tier`'s: the bank's where the
    /// device places, else those the host lands into.
    fn evicting(&self, tier: BankTier) -> Evicting {
        match (tier, self.plan.placer()) {
            (BankTier::Bank, Placer::Tier) => Evicting::Bank,
            (BankTier::Bank, Placer::Bank) | (BankTier::Host, _) => Evicting::Landed,
        }
    }

    /// Begin growing the host tier to `to` host slots on its thread, each
    /// host slot joining the tier as soon as every group holds it; while a
    /// resize of it runs, none begins.
    pub(super) fn grow_host(&self, to: u64, host_ledger: Ledger, wake: CompletionWake) -> Result<BankStep, DomainError> {
        Ok(match self.host_growth(to, host_ledger, wake, false)? {
            true => BankStep::Begun(BankTier::Host),
            false => BankStep::Running,
        })
    }

    /// Back the host tier from the floor it opens at up to `to` host slots
    /// (at most `self.plan.host_slots`, clamped to live post-domain `HostRam`
    /// pinned room by `ServedBank::begin_host_backing`), on its thread: each
    /// host slot joins the tier as soon as every group holds it, so the load
    /// waits for none of it (backing tens of gigabytes of pinned host storage
    /// before the bank opened held a 6.89 GB load of the 3090 18-26 s, its
    /// import 25-33 s against 5-10 s at the floor, 2026-11-01). It is the
    /// open's, no resize: its end reports none.
    pub(super) fn back_host_to(&self, to: u64, host_ledger: Ledger) -> Result<(), DomainError> {
        let clamped = to.min(self.plan.host_slots);
        if clamped == 0 {
            return Ok(());
        }
        self.host_growth(clamped, host_ledger, CompletionWake::new(|| {}), true).map(|_| ())
    }

    /// Begin growing the host tier to `to` host slots, the open's backing
    /// where `backing`; false while a resize of it runs, which begins none.
    fn host_growth(&self, to: u64, host_ledger: Ledger, wake: CompletionWake, backing: bool) -> Result<bool, DomainError> {
        let mut tier = self.host().ok_or_else(|| DomainError::invariant("a host tier growth without one"))?;
        let tier = &mut *tier;
        if let Resize::Broken(reason) = &tier.commitment.resize {
            return Err(DomainError::invariant(reason.clone()));
        }
        if tier.commitment.resizing() {
            return Ok(false);
        }
        let growing = self.growing(&tier.stores, (&tier.commitment.slots, to), BankTier::Host, Joining::Landed, Some(host_ledger), &tier.thread, wake);
        tier.commitment.resize = Resize::Growing { growing, backing };
        Ok(true)
    }

    /// Begin growing `stores`, `tier`'s, from their `slots` now to `to` on
    /// the store's resize `thread` ([`grow_by_slots`]), each slot joining
    /// the watcher as `joining` says as soon as every group holds it, its
    /// granules held by the growth's claim (`BankCache::grow`); `wake`
    /// completed once it ends.
    #[allow(clippy::too_many_arguments)]
    fn growing(
        &self,
        stores: &Arc<Stores>,
        (slots, to): (&Arc<AtomicU64>, u64),
        tier: BankTier,
        joining: Joining,
        host_ledger: Option<Ledger>,
        thread: &StorageThread,
        wake: CompletionWake,
    ) -> Growing {
        let (decided, from, charged) = (Instant::now(), slots.load(Ordering::Acquire), stores.charged());
        let (shared, slots, stop) = (self.watcher.shared().clone(), slots.clone(), Arc::new(AtomicBool::new(false)));
        let (stopping, rises, rise_gate, stores, records) =
            (stop.clone(), self.rises.clone(), self.stores.rise_gate().clone(), stores.clone(), self.tier_counts(tier));
        let counts = self.counts.clone();
        let changes = self.cache.changes().clone();
        let done = thread.run(wake, move || {
            let slot = || shared.backing_slot(&stopping);
            // For `BankTier::Host`, check live pinned HostRam headroom through
            // the single `Estimate::live_pinned_headroom` owner before each
            // slot that requires a new OS block so pinned host allocations
            // never outrun live `HostRam` headroom, OS/container reserve,
            // `guarantee`, or standing `cap` while the owner thread is busy.
            let host_guard = |at: u64| {
                let delta = counts.host_charge(at).saturating_sub(stores.charged());
                if delta == 0 {
                    return true;
                }
                let sample = seismic::host_sample();
                let estimate = host_ledger
                    .as_ref()
                    .expect("invariant violated: host tier growth without HostRam ledger")
                    .estimate();
                let headroom = estimate.live_pinned_headroom(&sample);
                let fits = headroom >= delta;
                if !fits {
                    tracing::warn!(
                        "host tier growth stopped before slot {at} of {to}: live pinned HostRam headroom \
                         ({headroom} bytes; free_beyond_target={} bytes, clean_cache={} bytes, \
                         pinned_reserve={} bytes, guarantee={} bytes) cannot back {delta} more bytes",
                        sample.supply.free_beyond_target,
                        sample.supply.clean_cache,
                        estimate.pinned_reserve(),
                        estimate.guarantee(),
                    );
                }
                fits
            };
            let guard: Option<&dyn Fn(u64) -> bool> = match tier {
                BankTier::Host => Some(&host_guard),
                BankTier::Bank => None,
            };
            grow_by_slots_gated(
                &stores,
                (from, to),
                &records,
                joining,
                &changes,
                |join| shared.join(join),
                (&slots, &rises, Some(&rise_gate)),
                &stopping,
                slot,
                guard,
            )
        });
        Growing { from, charged, decided, stop, done }
    }

    /// End a running growth of the holdings in `tier` storage at its next
    /// slot ([`Growing::stop`]): a shrink of that store waits for it no
    /// longer than that slot's claims. The host tier's backing is its
    /// growth.
    pub(super) fn stop_growth(&self, tier: BankTier) {
        let stop = |resize: &Resize| {
            if let Resize::Growing { growing, .. } = resize {
                growing.stop.store(true, Ordering::Release);
            }
        };
        match tier {
            BankTier::Bank => stop(&self.commitment().resize),
            BankTier::Host => {
                if let Some(tier) = self.host() {
                    stop(&tier.commitment.resize);
                }
            }
        }
        // A growth waiting between its slots on a hold looks again.
        self.watcher.shared().events().notify();
    }

    /// Drain settled blocks owed to the OS/driver from `stores`'s pool at a
    /// flight boundary: fenced granules released by an earlier async shrink
    /// settle once their flight completes, and vacating here returns them on
    /// the very next boundary instead of leaving them trapped in fragmented
    /// blocks.
    fn drain_owed(&self, stores: &Stores, label: &str) -> Result<u64, DomainError> {
        if stores.pool.owed() == 0 {
            return Ok(0);
        }
        self.cache
            .changes()
            .clone()
            .change(|_| stores.return_freed(0))
            .map_err(|error| DomainError::invariant(format!("the {label}'s pool could not return settled blocks: {error}")))
    }

    /// Finish a host tier's resize: its growth once its thread ended it, or
    /// its shrink. Its outcome, and whether a domain resize began it (the
    /// open's backing to its planned slots did not, so its end is reported
    /// to none).
    pub(super) fn advance_host(&self) -> Result<Option<(BankResized, bool)>, DomainError> {
        let Some(mut tier) = self.host() else {
            return Ok(None);
        };
        let tier = &mut *tier;
        if let Some(mut resized) = tier.commitment.resize.take_reported() {
            let drained = self.drain_owed(&tier.stores, "host tier")?;
            if let BankResized::Shrunk { returned, .. } = &mut resized {
                *returned += drained;
            }
            return Ok(Some((resized, true)));
        }
        self.drain_owed(&tier.stores, "host tier")?;
        let Resize::Growing { growing, backing } = &tier.commitment.resize else {
            return Ok(None);
        };
        let backing = *backing;
        let Some(ended) = growing.ended(BankHolding::Host, || tier.stores.charged()) else {
            return Ok(None);
        };
        match ended {
            Ok((resized, spent)) => {
                let slots = tier.commitment.slots.load(Ordering::Acquire);
                let kind = if backing { "backing" } else { "growth" };
                let (planned, ceiling) = (self.plan.host_slots, tier.ceiling);
                tracing::info!("the host tier's {kind} ended at {slots} of {planned} planned host slots ({ceiling} every expert), {spent}: {resized:?}");
                tier.commitment.resize = Resize::Idle;
                Ok(Some((resized, !backing)))
            }
            Err(reason) => {
                tier.commitment.resize = Resize::Broken(reason.clone());
                Err(DomainError::invariant(reason))
            }
        }
    }

    /// Whether a host tier's growth runs, or its shrink is not yet
    /// reported, until its advance finishes it.
    pub(super) fn host_growing(&self) -> bool {
        self.host()
            .is_some_and(|tier| tier.commitment.resizing() && !matches!(tier.commitment.resize, Resize::Growing { backing: true, .. }))
    }

    /// Whether the host tier's backing to its planned slots runs
    /// ([`Self::back_host`]), until its advance finishes it.
    pub(super) fn host_backing(&self) -> bool {
        self.host().is_some_and(|tier| matches!(tier.commitment.resize, Resize::Growing { backing: true, .. }))
    }

    /// [`ElasticBank::shrink`] of `tier`'s records now, beside running steps
    /// or not, to the deepest of `cuts` (its shallowest and deepest slots)
    /// whose every group holds enough records nothing writes or reads in
    /// place, decided at `decided` and queued `queued`: the watcher evicts
    /// them, moving nothing ([`watcher::Shared::shrink`]), and their slabs
    /// return to the pool at once (every step begun is submitted on this
    /// thread before the domain resizes, so the last flight that read them
    /// is known), each granule free once that flight completes, and the
    /// pool returns its empty blocks. What it came to, which the next
    /// advance reports; none where the watcher evicted nothing (a whole bank
    /// beside running steps, or every cut keeping fewer records than those
    /// in flight), which the next boundary decides again. Not while a
    /// resize of the store runs.
    pub(super) fn shrink(&self, tier: BankTier, cuts: (u64, u64), decided: Instant, queued: Duration) -> Result<BankStep, DomainError> {
        let candidates = self.candidates(tier, cuts)?;
        let evicting = self.evicting(tier);
        let shrink = |commitment: &mut Commitment, stores: &Stores, holding: BankHolding| -> Result<BankStep, DomainError> {
            if let Resize::Broken(reason) = &commitment.resize {
                return Err(DomainError::invariant(reason.clone()));
            }
            if commitment.resizing() {
                return Ok(BankStep::Running);
            }
            let (from, charged) = (commitment.slots.load(Ordering::Acquire), stores.charged());
            let Some(Evicted { slots, records, trip, .. }) = self.watcher.shared().shrink(evicting, candidates, None).map_err(DomainError::invariant)? else {
                tracing::info!(?holding, from, "a shrink waits a boundary: the watcher evicts nothing now");
                return Ok(BankStep::Deferred);
            };
            let to = slots as u64;
            commitment.slots.store(to, Ordering::Release);
            let begun = Begun { decided, queued, planned: decided.elapsed().saturating_sub(queued), trip };
            commitment.resize = Resize::Reported(self.release(stores, tier, (from, to), charged, &records, begun)?);
            Ok(BankStep::Begun(tier))
        };
        match tier {
            BankTier::Bank => shrink(&mut self.commitment(), &self.stores, BankHolding::Experts),
            BankTier::Host => {
                let mut host = self.host().ok_or_else(|| DomainError::invariant("a host tier shrink without one"))?;
                let host = &mut *host;
                shrink(&mut host.commitment, &host.stores, BankHolding::Host)
            }
        }
    }

    /// Return a shrink's `evicted` records of `stores` to their pool, cut
    /// `BankCache` to `above_floor(tier, to)`, and return freed blocks to the
    /// OS (`return_freed_sync` when `sync_need` is `Some` for synchronous
    /// admission reclaim, `return_freed` otherwise for `between_flights`
    /// external-pressure shrinks). The one shrink-release path both reach.
    fn release_evicted(
        &self,
        stores: &Stores,
        tier: BankTier,
        to: u64,
        charged: u64,
        evicted: &[Vec<u32>],
        sync_need: Option<u64>,
    ) -> Result<(u64, u64), DomainError> {
        let holding = match tier {
            BankTier::Bank => BankHolding::Experts,
            BankTier::Host => BankHolding::Host,
        };
        self.cache.changes().clone().back(|_| {
            for (group, records) in evicted.iter().enumerate() {
                stores.free(group, records).map_err(|refusal| DomainError::invariant(format!("the expert bank's pool refused a shrink's release: {refusal}")))?;
            }
            let released = charged.checked_sub(stores.charged()).ok_or_else(|| DomainError::invariant("a shrink raised its records' charge"))?;
            self.cache.fit(holding, self.above_floor(tier, to));
            let returned = match sync_need {
                Some(need) => stores.pool.return_freed_sync(released.max(need)),
                None => stores.return_freed(released),
            }
            .map_err(|error| DomainError::invariant(format!("the expert bank's pool could not return a shrink's blocks: {error}")))?;
            tracing::debug!(?holding, released, returned, sync = sync_need.is_some(), "a shrink's granules return to the pool, its empty blocks to the OS");
            Ok((released, returned))
        })
    }

    fn release(&self, stores: &Stores, tier: BankTier, (from, to): (u64, u64), charged: u64, evicted: &[Vec<u32>], begun: Begun) -> Result<BankResized, DomainError> {
        let holding = match tier {
            BankTier::Bank => BankHolding::Experts,
            BankTier::Host => BankHolding::Host,
        };
        let releasing = Instant::now();
        let (released, returned) = self.release_evicted(stores, tier, to, charged, evicted, None)?;
        let times = begun.times(releasing, releasing, Instant::now());
        Ok(BankResized::Shrunk { holding, from, to, released, returned, times })
    }

    /// [`ElasticBank::grow`] to `to` slots, its growth target, on the resize
    /// thread, each slot joining the bank as soon as every group holds it
    /// ([`Self::growing`]): the device's records where it places, the one
    /// store's landed records where the host does.
    pub(super) fn grow(&self, to: u64, wake: CompletionWake) -> Result<BankStep, DomainError> {
        let mut commitment = self.commitment();
        if let Resize::Broken(reason) = &commitment.resize {
            return Err(DomainError::invariant(reason.clone()));
        }
        if commitment.resizing() {
            return Ok(BankStep::Running);
        }
        let joining = match self.plan.placer() {
            Placer::Tier => Joining::Bank,
            Placer::Bank => Joining::Landed,
        };
        let growing = self.growing(&self.stores, (&commitment.slots, to), BankTier::Bank, joining, None, &self.resize_thread, wake);
        commitment.resize = Resize::Growing { growing, backing: false };
        Ok(BankStep::Begun(BankTier::Bank))
    }

    /// [`ElasticBank::advance`]: at a boundary between flights every pass
    /// is submitted, so a shrink's evicted records return to the pool here;
    /// a shrink done is reported, and a growth finishes once its resize
    /// thread ended it.
    pub(super) fn advance(&self) -> Result<Option<BankResized>, DomainError> {
        let mut commitment = self.commitment();
        if let Some(releasing) = commitment.resize.take_releasing() {
            match self.released(releasing) {
                Ok(shrunk) => commitment.resize = Resize::Reported(shrunk),
                Err(error) => {
                    commitment.resize = Resize::Broken(error.to_string());
                    return Err(error);
                }
            }
        }
        if let Some(mut resized) = commitment.resize.take_reported() {
            let drained = self.drain_owed(&self.stores, "expert bank")?;
            if let BankResized::Shrunk { returned, .. } = &mut resized {
                *returned += drained;
            }
            return Ok(Some(resized));
        }
        self.drain_owed(&self.stores, "expert bank")?;
        let Resize::Growing { growing, .. } = &commitment.resize else {
            return Ok(None);
        };
        match growing.ended(BankHolding::Experts, || self.charged()) {
            None => Ok(None),
            Some(Ok((grown, spent))) => {
                let slots = commitment.slots.load(Ordering::Acquire);
                tracing::info!("the expert bank's growth ended at {slots} slots, {spent}: {grown:?}");
                commitment.resize = Resize::Idle;
                Ok(Some(grown))
            }
            Some(Err(reason)) => {
                commitment.resize = Resize::Broken(reason.clone());
                Err(DomainError::invariant(reason))
            }
        }
    }

    /// `releasing`'s records returned to the pool: its shrink, done.
    fn released(&self, releasing: Releasing) -> Result<BankResized, DomainError> {
        let Releasing { from, to, charged, evicted, begun, .. } = releasing;
        self.release(&self.stores, BankTier::Bank, (from, to), charged, &evicted, begun)
    }

    /// Whether a resize of the bank's records runs, until its advance
    /// finishes it (broken records run none: a resize asked of them fails
    /// with why).
    pub(super) fn resizing(&self) -> bool {
        self.commitment().resizing()
    }

    /// At a submission boundary of the scratch-served chain, prompt group
    /// `chain`'s steps, every pass of its blocks through
    /// `submitted` submitted (none where no block's is), the bank's release
    /// beside it, on every placement:
    /// - no resize running, where `pressure` finds the bank's storage short,
    ///   the watcher evicts records beside the chain, from the nearest
    ///   slots that release storage toward those releasing what it is short
    ///   of (`watcher::Shared::shrink`), moving nothing: those the chain's
    ///   steps have yet to decide leave the map from its first step, the
    ///   rest from the first step not yet begun;
    /// - its evicted records return to the pool once every pass of the
    ///   highest block whose hits the chain had decided is submitted (at
    ///   once where none, or at a later chain's boundary): the next advance
    ///   reports it as any shrink.
    ///
    /// The block whose passes' submission the release awaits, where one
    /// does.
    pub(crate) fn release_running(&self, chain: ScratchGroup, submitted: Option<usize>, pressure: &dyn Fn() -> Option<RunningPressure>) -> Option<usize> {
        let started = Instant::now();
        // Memory is read outside the commitment's lock.
        let short = {
            let commitment = self.commitment();
            matches!(commitment.resize, Resize::Idle) && !commitment.resizing() && commitment.slots.load(Ordering::Acquire) > self.counts.floor()
        };
        let short = short.then(pressure).flatten().filter(|pressure| pressure.bytes > 0);
        let mut commitment = self.commitment();
        if !matches!(commitment.resize, Resize::Releasing(_)) {
            let short = short.filter(|_| matches!(commitment.resize, Resize::Idle) && !commitment.resizing())?;
            if let Err(reason) = self.evict_beside(&mut commitment, chain, short, started) {
                commitment.resize = Resize::Broken(reason);
                return None;
            }
        }
        let Resize::Releasing(releasing) = &commitment.resize else {
            return None;
        };
        // Due once every pass of the decided block is submitted, or at once
        // beside a later chain, whose steps resolve none of them.
        let awaits = (releasing.last == chain.last()).then_some(releasing.after).flatten().filter(|&block| submitted.is_none_or(|through| through < block));
        if awaits.is_some() {
            return awaits;
        }
        let releasing = commitment.resize.take_releasing().expect("a release is due");
        match self.released(releasing) {
            Ok(shrunk) => commitment.resize = Resize::Reported(shrunk),
            Err(error) => commitment.resize = Resize::Broken(error.to_string()),
        }
        None
    }

    /// Ask the watcher to evict the bank's records beside the running chain,
    /// prompt group `chain`'s steps, its storage `short` as
    /// read at `started`: from the nearest slots that release storage
    /// toward those releasing what it is short of. Where it evicted, the
    /// commitment's slots follow at once (no step from then on resolves an
    /// evicted record), the domain is owed its beginning
    /// ([`Self::take_running_begun`]), and the records wait for their
    /// release; where none releases storage or the watcher evicted nothing,
    /// nothing changes. Why the records broke, where they did.
    fn evict_beside(&self, commitment: &mut Commitment, chain: ScratchGroup, short: RunningPressure, started: Instant) -> Result<(), String> {
        let RunningPressure { bytes } = short;
        let (from, floor) = (commitment.slots.load(Ordering::Acquire), self.counts.floor());
        let decided = Instant::now();
        let charge = |at| Ok::<_, std::convert::Infallible>(self.counts.resident_charge(at));
        let releasing = |bytes| crate::planning::releasing(from, floor, bytes, charge).unwrap_or_else(|never| match never {});
        let (Some(shallowest), Some(deepest)) = (releasing(0), releasing(bytes)) else {
            return Ok(());
        };
        let candidates = self.candidates(BankTier::Bank, (shallowest, deepest)).map_err(|error| format!("a shrink beside a running chain: {error}"))?;
        let charged = self.charged();
        let Some(Evicted { slots, records, decided: layer, trip }) = self.watcher.shared().shrink(self.evicting(BankTier::Bank), candidates, Some(chain))? else {
            return Ok(());
        };
        tracing::info!(from, shallowest, deepest, short = bytes, "the expert bank is shrunk beside a running chain");
        let to = slots as u64;
        commitment.slots.store(to, Ordering::Release);
        assert!(commitment.running.replace(RunningBegun { decided, held: started.elapsed() }).is_none(), "a running shrink's beginning is taken before the next begins");
        let after = layer.map(|layer| self.plan.layers()[layer].sublayer().block as usize);
        let begun = Begun { decided, queued: Duration::ZERO, planned: trip.picked + trip.served, trip };
        commitment.resize = Resize::Releasing(Releasing { from, to, charged, evicted: records, last: chain.last(), after, begun });
        Ok(())
    }

    /// Return `freed` bytes of `tier`'s pool to the OS, what a shrink of a
    /// holding in it freed (`seismic::Pool::return_freed`): the bytes
    /// returned now.
    pub(super) fn return_freed(&self, tier: BankTier, freed: u64) -> Result<u64, DomainError> {
        let pool = match tier {
            BankTier::Bank => self.stores.pool.clone(),
            BankTier::Host => self.host().ok_or_else(|| DomainError::invariant("a host tier's pool without one"))?.stores.pool.clone(),
        };
        pool.return_freed(freed).map_err(|error| DomainError::invariant(format!("the {tier:?} pool could not return a shrink's blocks: {error}")))
    }

    /// Synchronously return at least `freed` bytes of `tier`'s pool to the OS
    /// (`seismic::Pool::return_freed_sync`), for synchronous admission reclaim.
    pub(super) fn return_freed_sync(&self, tier: BankTier, freed: u64) -> Result<u64, DomainError> {
        let pool = match tier {
            BankTier::Bank => self.stores.pool.clone(),
            BankTier::Host => self.host().ok_or_else(|| DomainError::invariant("a host tier's pool without one"))?.stores.pool.clone(),
        };
        pool.return_freed_sync(freed).map_err(|error| DomainError::invariant(format!("the {tier:?} pool could not return a shrink's blocks: {error}")))
    }

    /// [`ElasticBank::running_begun`](crate::domain::ElasticBank::running_begun).
    pub(super) fn take_running_begun(&self) -> Option<RunningBegun> {
        self.commitment().running.take()
    }

    /// Synchronously reclaim at least `need` bytes of `tier`'s bank cache back
    /// to the OS (down to `tier`'s floor): stop and await any running growth,
    /// compute exact target slots releasing `>= need` from exact slot & granule
    /// sizes (no 1-block jitter tolerance), evict from the watcher (waiting on
    /// in-flight bank traffic if needed), free slabs, trim `BankCache`, and
    /// synchronously return freed blocks to the OS (`return_freed_sync`).
    pub(super) fn reclaim(&self, tier: BankTier, need: u64) -> Result<u64, ReclaimError> {
        if need == 0 {
            return Ok(0);
        }
        let holding = match tier {
            BankTier::Bank => BankHolding::Experts,
            BankTier::Host => BankHolding::Host,
        };
        let (stores, reclaim_lock): (Arc<Stores>, Arc<Mutex<()>>) = match tier {
            BankTier::Bank => (self.stores.clone(), self.commitment().reclaim_lock.clone()),
            BankTier::Host => match self.host() {
                Some(host) => (host.stores.clone(), host.commitment.reclaim_lock.clone()),
                None => return Err(ReclaimError::AtFloor),
            },
        };

        // Serialize concurrent `reclaim` calls on `tier` outside the
        // commitment lock, then mark `commitment.reclaiming = true` so no
        // domain `grow`/`shrink`/`release_running` can start on `tier` while
        // `reclaim` runs across blocking waits.
        let _reclaim_lock = reclaim_lock.lock().expect("bank reclaim");

        struct ReclaimGuard<'a> {
            bank: &'a ExpertBank,
            tier: BankTier,
        }
        impl Drop for ReclaimGuard<'_> {
            fn drop(&mut self) {
                match self.tier {
                    BankTier::Bank => self.bank.commitment().reclaiming = false,
                    BankTier::Host => {
                        if let Some(mut host) = self.bank.host() {
                            host.commitment.reclaiming = false;
                        }
                    }
                }
            }
        }

        // 1. Mark `reclaiming = true` and stop any running growth on `tier`,
        //    then await its current slot outside the commitment lock so no
        //    deadlock is possible.
        let begin_reclaim = |commitment: &mut Commitment, label: &str| -> Result<Option<(Growing, bool)>, ReclaimError> {
            if let Resize::Broken(reason) = &commitment.resize {
                return Err(ReclaimError::Invariant(reason.clone()));
            }
            assert!(!commitment.reclaiming, "invariant violated: concurrent {label} reclaim despite reclaim_lock");
            commitment.reclaiming = true;
            Ok(commitment.resize.take_growing().inspect(|(growing, _)| growing.stop.store(true, Ordering::Release)))
        };
        let stopped = match tier {
            BankTier::Bank => begin_reclaim(&mut self.commitment(), "bank")?,
            BankTier::Host => {
                let Some(mut host) = self.host() else {
                    return Err(ReclaimError::AtFloor);
                };
                begin_reclaim(&mut host.commitment, "host")?
            }
        };
        let _reclaiming = ReclaimGuard { bank: self, tier };
        if let Some((growing, was_backing)) = stopped {
            self.watcher.shared().events().notify();
            let outcome = growing.done.wait();
            match growing.finish(holding, outcome, || stores.charged()) {
                Ok((resized, _spent)) => {
                    self.cache.settle(&resized);
                    let slots = match tier {
                        BankTier::Bank => self.commitment().slots.load(Ordering::Acquire),
                        BankTier::Host => self.host().map_or(0, |h| h.commitment.slots.load(Ordering::Acquire)),
                    };
                    self.cache.fit(holding, self.above_floor(tier, slots));
                    if !was_backing {
                        match tier {
                            BankTier::Bank => self.commitment().resize = Resize::Reported(resized),
                            BankTier::Host => {
                                if let Some(mut host) = self.host() {
                                    host.commitment.resize = Resize::Reported(resized);
                                }
                            }
                        }
                    }
                }
                Err(reason) => {
                    match tier {
                        BankTier::Bank => self.commitment().resize = Resize::Broken(reason.clone()),
                        BankTier::Host => {
                            if let Some(mut host) = self.host() {
                                host.commitment.resize = Resize::Broken(reason.clone());
                            }
                        }
                    }
                    return Err(ReclaimError::Invariant(reason));
                }
            }
        }

        // 2. Flush any pending `Resize::Releasing` from an earlier running
        //    shrink on Bank, storing its `BankResized` in `Resize::Reported`
        //    so the next `advance()` pairs with `bank_began[Bank]`.
        if tier == BankTier::Bank {
            let releasing = self.commitment().resize.take_releasing();
            if let Some(releasing) = releasing {
                match self.released(releasing) {
                    Ok(shrunk) => self.commitment().resize = Resize::Reported(shrunk),
                    Err(error) => {
                        let reason = error.to_string();
                        self.commitment().resize = Resize::Broken(reason.clone());
                        return Err(ReclaimError::Invariant(reason));
                    }
                }
            }
        }

        // 3. Loop across eviction passes until `total_returned >= need`, the
        //    tier reaches its floor, or nothing further can be evicted.
        //    A single eviction pass can free granules scattered across
        //    partially-occupied pool blocks without emptying a whole block
        //    yet; retrying with at least `block_bytes.max(deficit)` guarantees
        //    forward progress until whole blocks are returned or floor is hit.
        let mut total_returned = 0u64;
        let mut total_released = 0u64;
        let mut evicted_any = false;
        let mut _traffic = None;
        let initial_from = match tier {
            BankTier::Bank => self.commitment().slots.load(Ordering::Acquire),
            BankTier::Host => self.host().map_or(0, |h| h.commitment.slots.load(Ordering::Acquire)),
        };
        let mut current_from = initial_from;
        let block_bytes = stores.pool.block_bytes();
        let charge_fn = self.tier_charge(tier);
        let charge = |at| Ok::<_, std::convert::Infallible>(charge_fn(&self.counts, at));

        while total_returned < need {
            let deficit = need - total_returned;

            // Flush already-free or fenced pool granules first before evicting
            // more live bank slots.
            let mut flushed = self
                .cache
                .changes()
                .clone()
                .back(|_| stores.pool.return_freed_sync(deficit))
                .map_err(|error| ReclaimError::Invariant(error.to_string()))?;
            // If at least one full block's worth of granules is already free
            // in the pool yet nothing returned, an in-flight promotion or
            // landing may momentarily hold a temporary view on the candidate
            // block. Drain in-flight traffic so those views drop, then retry
            // returning before evicting additional live slots.
            if flushed == 0 && stores.pool.free_granules() as u64 * stores.pool.granule() >= block_bytes {
                if _traffic.is_none() {
                    _traffic = Some(self.watcher.shared().hold_traffic());
                    self.watcher.shared().wait_traffic_quiet().map_err(ReclaimError::Invariant)?;
                }
                flushed = self
                    .cache
                    .changes()
                    .clone()
                    .back(|_| stores.pool.return_freed_sync(deficit))
                    .map_err(|error| ReclaimError::Invariant(error.to_string()))?;
            }
            total_returned = total_returned.saturating_add(flushed);
            if total_returned >= need {
                break;
            }
            let deficit = need - total_returned;

            let (from, floor) = match tier {
                BankTier::Bank => (self.commitment().slots.load(Ordering::Acquire), self.floor(tier)),
                BankTier::Host => (self.host().map_or(0, |h| h.commitment.slots.load(Ordering::Acquire)), self.floor(tier)),
            };
            current_from = from;
            if from <= floor {
                break;
            }

            let evict_bytes = if !evicted_any {
                let already_free = stores.pool.free_granules() as u64 * stores.pool.granule();
                let target_blocks = deficit.div_ceil(block_bytes).saturating_mul(block_bytes);
                match target_blocks.saturating_sub(already_free) {
                    0 => deficit,
                    bytes => bytes,
                }
            } else {
                // Previous eviction freed granules without vacating enough
                // whole blocks: evict at least one full pool block so
                // fragmentation cannot stall progress.
                block_bytes.max(deficit)
            };

            let shallowest = crate::planning::releasing(from, floor, 0, charge).unwrap_or_else(|never| match never {});
            let deepest = crate::planning::releasing(from, floor, evict_bytes, charge).unwrap_or_else(|never| match never {});
            let (Some(shallowest), Some(deepest)) = (shallowest, deepest) else {
                break;
            };

            // Hold background promotion/landing traffic across eviction and
            // `release_evicted` so transient views neither block vacation nor
            // force extra live-slot evictions during synchronous reclaim.
            if _traffic.is_none() {
                _traffic = Some(self.watcher.shared().hold_traffic());
                self.watcher.shared().wait_traffic_quiet().map_err(ReclaimError::Invariant)?;
            }
            let evicting = self.evicting(tier);
            let charged = stores.charged();
            let candidates = self.candidates(tier, (shallowest, deepest)).map_err(|error| ReclaimError::Invariant(error.to_string()))?;
            let evicted = self.watcher.shared().shrink(evicting, candidates, None).map_err(ReclaimError::Invariant)?;

            let Some(Evicted { slots, records, .. }) = evicted else {
                break;
            };

            let to = slots as u64;
            if to >= from {
                break;
            }
            match tier {
                BankTier::Bank => self.commitment().slots.store(to, Ordering::Release),
                BankTier::Host => {
                    if let Some(host) = self.host() {
                        host.commitment.slots.store(to, Ordering::Release);
                    }
                }
            }
            let (released, returned) = self
                .release_evicted(&stores, tier, to, charged, &records, Some(deficit))
                .map_err(|error| ReclaimError::Invariant(error.to_string()))?;
            evicted_any = true;
            current_from = to;
            total_released = total_released.saturating_add(released);
            total_returned = total_returned.saturating_add(returned);
        }

        if evicted_any {
            tracing::info!(
                ?holding,
                from = initial_from,
                to = current_from,
                need,
                released = total_released,
                returned = total_returned,
                "expert bank reclaimed synchronously for admission"
            );
        }
        match total_returned {
            0 => Err(ReclaimError::AtFloor),
            n => Ok(n),
        }
    }
}

/// Synchronous [`CacheReclaimer`] for one tier of an [`ExpertBank`], holding
/// only a weak reference so it never extends the bank's lifetime.
pub(super) struct BankReclaimer {
    bank: Weak<ExpertBank>,
    tier: BankTier,
}

impl BankReclaimer {
    pub(super) fn new(bank: &Arc<ExpertBank>, tier: BankTier) -> Arc<Self> {
        Arc::new(Self { bank: Arc::downgrade(bank), tier })
    }
}

impl CacheReclaimer for BankReclaimer {
    fn reclaim(&self, need: u64) -> Result<u64, ReclaimError> {
        let Some(bank) = self.bank.upgrade() else {
            return Err(ReclaimError::AtFloor);
        };
        bank.reclaim(self.tier, need)
    }
}

#[cfg(test)]
mod rise_gate_tests {
    use super::RiseGate;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    /// Under a tight background `hold_background` loop (modelling consecutive
    /// `grow_by_slots` iterations), a `hold_priority` caller (`hold_rises`)
    /// acquires `rises` on the very next release and is never starved.
    #[test]
    fn priority_caller_acquires_on_very_next_release() {
        const ROUNDS: usize = 16;
        let gate = Arc::new(RiseGate::new());
        let rises = Arc::new(Mutex::new(()));
        let stop = Arc::new(AtomicBool::new(false));
        let entered_round = Arc::new(AtomicUsize::new(0));
        let completed_slots = Arc::new(AtomicUsize::new(0));
        let hold_until_waiter = Arc::new(AtomicBool::new(true));

        let bg = {
            let (gate, rises, stop, entered_round, completed_slots, hold_until_waiter) = (
                gate.clone(),
                rises.clone(),
                stop.clone(),
                entered_round.clone(),
                completed_slots.clone(),
                hold_until_waiter.clone(),
            );
            std::thread::spawn(move || {
                while !stop.load(Ordering::Acquire) {
                    let _guard = gate.hold_background(&rises);
                    if stop.load(Ordering::Acquire) {
                        break;
                    }
                    let slot = completed_slots.fetch_add(1, Ordering::AcqRel) + 1;
                    if hold_until_waiter.swap(false, Ordering::AcqRel) {
                        entered_round.store(slot, Ordering::Release);
                        // Hold `rises` until the priority caller has registered
                        // itself as a waiter, then drop and immediately loop to
                        // try to re-acquire `rises` without yielding.
                        while gate.priority_waiters() == 0 && !stop.load(Ordering::Acquire) {
                            std::thread::yield_now();
                        }
                    }
                }
            })
        };

        for expected_slot in 1..=ROUNDS {
            while entered_round.load(Ordering::Acquire) < expected_slot {
                std::thread::yield_now();
            }
            let _priority = gate.hold_priority(&rises);
            let seen = completed_slots.load(Ordering::Acquire);
            assert_eq!(
                seen, expected_slot,
                "priority caller was starved: expected acquisition right after slot {expected_slot}, saw {seen}"
            );
            if expected_slot < ROUNDS {
                hold_until_waiter.store(true, Ordering::Release);
            } else {
                stop.store(true, Ordering::Release);
            }
        }

        bg.join().expect("background thread finished cleanly");
        assert_eq!(gate.priority_waiters(), 0);
    }

    /// Reproduces `79bbf8` on a 64 GiB Linux container with `cuMemHostAlloc`
    /// growth approaching the `clean_cache` / `guarantee` floor:
    /// `Estimate::live_pinned_headroom` (the single owner replacing ad-hoc
    /// `host_supply_covers`) never counts swap-backed `cold_reclaimable`,
    /// excludes `clean_cache` on unmonitored shared cgroups, always preserves
    /// the machine's OS/container clean-page-cache reserve (`pinned_reserve` =
    /// 2 GiB on 64 GiB) plus `guarantee` (others' rise), and stops per-slot
    /// growth cleanly before clean page cache can be driven to 0 bytes.
    #[test]
    fn live_pinned_headroom_stops_growth_before_draining_clean_cache_or_guarantee() {
        use crate::memory::ledger::{Ledgers, LoadShape, PoolClass};
        use crate::memory::supply::Reading;
        use std::num::NonZeroU64;
        use std::sync::Arc;
        use std::time::{Duration, Instant};

        const MIB: u64 = 1 << 20;
        const GIB: u64 = 1 << 30;
        const TOTAL: u64 = 64 * GIB;
        const SLOT: u64 = 384 * MIB;
        const RESERVE: u64 = 2 * GIB; // scarcity_cushion(64 GiB, 384 MiB)

        let t0 = Instant::now();
        let sample_with = |at: Instant, free_beyond_target: i64, clean_cache: u64, cold_reclaimable: u64, monitored: bool| {
            seismic::HostSample {
                at,
                page_bytes: 4096,
                total: TOTAL,
                supply: seismic::Supply { free_beyond_target, clean_cache, cold_reclaimable },
                counted: seismic::Counted { at, lag: Duration::ZERO },
                cold: seismic::ColdPages { machine: cold_reclaimable, own: None },
                own_reclaimed: None,
                wired: None,
                harm: monitored.then_some(seismic::Harm { machine: 0, own: 0 }),
                pressure: monitored.then_some(seismic::PressureLevel::Normal),
                lag: seismic::Lag { release: Duration::ZERO, growth: Duration::ZERO },
            }
        };

        let scope = Ledgers::new();
        let ledger = scope.ledger(seismic::StoreIdentity::HostRam, Arc::new(LoadShape { stores: Vec::new() }));
        let _pool = ledger.pool_block(PoolClass::Records, NonZeroU64::new(SLOT).unwrap()).blocks;

        // Initial 64 GiB Linux container reading: 10 GiB free, 4 GiB clean cache,
        // 16 GiB swap-backed cold_reclaimable (must never be counted for pinned).
        let s0 = sample_with(t0, (10 * GIB) as i64, 4 * GIB, 16 * GIB, true);
        ledger.read(ledger.sampled(t0), Reading::host(&s0));
        let est = ledger.estimate();
        assert_eq!(est.pinned_reserve(), RESERVE);
        assert_eq!(est.guarantee(), 0);
        // Immediate supply = 10 + 4 = 14 GiB; minus 2 GiB reserve = 12 GiB.
        assert_eq!(est.pinned_room(), 12 * GIB);
        assert_eq!(est.pinned_import_room(), 12 * GIB);
        assert_eq!(est.live_pinned_headroom(&s0), 12 * GIB, "live_pinned_headroom agrees with pinned_import_room");

        // Simulate cuMemHostAlloc growing slot-by-slot until only 2 GiB clean
        // page cache remains (free_beyond_target = 0, clean_cache = 2 GiB,
        // cold_reclaimable = 16 GiB). Growth MUST stop with the 2 GiB clean
        // page cache completely intact (never driven to 0 bytes).
        let at_floor = sample_with(t0 + Duration::from_secs(1), 0, 2 * GIB, 16 * GIB, true);
        assert_eq!(
            est.live_pinned_headroom(&at_floor),
            0,
            "growth must stop at the 2 GiB clean-cache reserve even with 16 GiB swap"
        );

        // One slot above the 2 GiB reserve fits; one byte below does not.
        let one_slot_above = sample_with(t0 + Duration::from_secs(1), 0, RESERVE + SLOT, 0, true);
        assert_eq!(est.live_pinned_headroom(&one_slot_above), SLOT);
        let one_byte_short = sample_with(t0 + Duration::from_secs(1), 0, RESERVE + SLOT - 1, 16 * GIB, true);
        assert!(est.live_pinned_headroom(&one_byte_short) < SLOT);

        // Unmonitored shared cgroup (harm: None, pressure: None) excludes clean_cache.
        let unmonitored = sample_with(t0 + Duration::from_secs(1), (RESERVE + SLOT) as i64, 8 * GIB, 0, false);
        assert_eq!(est.live_pinned_headroom(&unmonitored), SLOT);
        let unmonitored_cache_only = sample_with(t0 + Duration::from_secs(1), 0, 8 * GIB, 0, false);
        assert_eq!(est.live_pinned_headroom(&unmonitored_cache_only), 0);

        // Negative free_beyond_target (cgroup/OS dipped below watermark) offsets clean_cache.
        let dipped = sample_with(t0 + Duration::from_secs(1), -(512 * MIB as i64), RESERVE + 512 * MIB, 0, true);
        assert_eq!(est.live_pinned_headroom(&dipped), 0);

        // Others' rise sets `guarantee` > 0 on the HostRam ledger: live_pinned_headroom
        // must preserve BOTH the 2 GiB OS reserve AND `guarantee`.
        let t1 = t0 + Duration::from_secs(2);
        let s1 = sample_with(t1, (9 * GIB) as i64, 4 * GIB, 16 * GIB, true); // others rose by 1 GiB
        ledger.read(ledger.sampled(t1), Reading::host(&s1));
        let est1 = ledger.estimate();
        assert_eq!(est1.guarantee(), GIB);
        // Immediate supply = 9 + 4 = 13 GiB; minus 2 GiB reserve minus 1 GiB guarantee = 10 GiB.
        assert_eq!(est1.pinned_import_room(), 10 * GIB);
        assert_eq!(est1.live_pinned_headroom(&s1), 10 * GIB);
        let with_guarantee_tight = sample_with(t1, 0, RESERVE + GIB + SLOT - 1, 0, true);
        assert!(
            est1.live_pinned_headroom(&with_guarantee_tight) < SLOT,
            "guarantee ({} B) + reserve ({RESERVE} B) must both be preserved",
            est1.guarantee()
        );
    }
}
