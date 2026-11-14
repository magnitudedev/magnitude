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
use crate::experts::watcher::{BankRecordViews, Candidates, Evicted, Evicting, Join};
use crate::memory::ledger::Changes;
use crate::planning::{BankTier, Placer};
use crate::storage_thread::{Done, StorageThread};
use crate::CompletionWake;
use seismic::{Pool, SlabTensor, Tensor, TensorError};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// One store's records: each group's slab store in the store's pool, a
/// record a slab, its leading `lent` records over the lender's lent
/// allocation for the load's life, each region one routed tensor's record
/// of `record_bytes`.
pub(super) struct Stores {
    pool: Pool,
    groups: Vec<Mutex<SlabTensor>>,
    lent: Vec<u64>,
    record_bytes: Vec<Vec<u64>>,
}

impl Stores {
    /// `groups`, each a store with its leading `lent` records and its
    /// regions' record bytes, in `pool`.
    pub(super) fn new(pool: Pool, groups: Vec<(SlabTensor, u64, Vec<u64>)>) -> Self {
        let (mut stores, mut lent, mut record_bytes) = (Vec::new(), Vec::new(), Vec::new());
        for (store, leading, bytes) in groups {
            stores.push(Mutex::new(store));
            lent.push(leading);
            record_bytes.push(bytes);
        }
        Self { pool, groups: stores, lent, record_bytes }
    }

    fn group(&self, group: usize) -> MutexGuard<'_, SlabTensor> {
        self.groups[group].lock().expect("a bank group's store")
    }

    /// The records group `group` holds: its slabs, lent ones included.
    pub(super) fn present(&self, group: usize) -> u64 {
        self.group(group).backed().len() as u64
    }

    /// What the pool charges the records: a granule each beyond the lent
    /// ones, which the lender's lent allocation holds.
    pub(super) fn charged(&self) -> u64 {
        (0..self.groups.len()).map(|group| self.present(group) - self.lent[group]).sum::<u64>() * self.pool.granule()
    }

    /// The tensor kernels bind as group `group`'s region `region`: its
    /// store's view, its record table, which every record held now or
    /// later is reached through.
    pub(super) fn region(&self, group: usize, region: usize) -> Result<Tensor, TensorError> {
        self.group(group).logical_region(region)
    }

    /// Claim records of group `group` until it holds `records`, each a
    /// granule of the pool (`SlabTensor::add_slab`: a free one in process,
    /// else the pool grows by a block its ledger entry admits), unwritten:
    /// the watcher writes a record before any step reads it. The records
    /// claimed, the lowest free first; on a refusal, none (those claimed
    /// before it are freed, never having been read).
    pub(super) fn claim_to(&self, group: usize, records: u64) -> Result<Vec<usize>, TensorError> {
        let mut store = self.group(group);
        let mut claimed = Vec::new();
        while (store.backed().len() as u64) < records {
            match store.add_slab(seismic::SlabFill::Unwritten) {
                Ok(record) => claimed.push(record),
                Err(refusal) => {
                    Self::unclaim_in(&mut store, group, &claimed);
                    return Err(refusal);
                }
            }
        }
        Ok(claimed)
    }

    /// Free `claimed` of group `group`, which nothing read: a refusal
    /// breaks the pool's invariant.
    fn unclaim(&self, group: usize, claimed: &[usize]) {
        Self::unclaim_in(&mut self.group(group), group, claimed);
    }

    fn unclaim_in(store: &mut SlabTensor, group: usize, claimed: &[usize]) {
        for &record in claimed {
            if let Err(error) = store.free_slab(record) {
                panic!("invariant violated: the bank's pool refused to free record {record} of group {group}, claimed and never read: {error}");
            }
        }
    }

    /// Return `records` of group `group` to the pool: each granule free
    /// once the last flight that read it completes. Never waits.
    pub(super) fn free(&self, group: usize, records: &[u32]) -> Result<(), TensorError> {
        let store = self.group(group);
        records.iter().try_for_each(|&record| store.free_slab(record as usize).map(drop))
    }

    /// Return `freed` bytes of the pool to the OS, what a shrink freed:
    /// every empty block, then the emptiest emptied while those fall short
    /// (`seismic::Pool::return_freed`). The bytes returned.
    fn return_freed(&self, freed: u64) -> Result<u64, TensorError> {
        self.pool.return_freed(freed)
    }

    /// Record `record` of group `group`'s region `region`, while it is
    /// held.
    pub(super) fn view(&self, group: usize, region: usize, record: usize) -> Option<Tensor> {
        self.group(group).slab_region(record, region).expect("a record and region lie in their group's store")
    }

    /// Each group's routed tensors' records the host addresses, as held
    /// now, through the last held: holes where none is.
    pub(super) fn records(&self) -> Result<Vec<Vec<Records>>, String> {
        (0..self.groups.len())
            .map(|group| {
                let store = self.group(group);
                let held = store.backed().last().map_or(0, |record| record + 1);
                (0..store.region_count())
                    .map(|region| {
                        let views = (0..held)
                            .map(|record| store.slab_region(record, region).expect("a record and region lie in their group's store"))
                            .collect();
                        Records::new(views, self.record_bytes[group][region])
                    })
                    .collect()
            })
            .collect()
    }
}

impl BankRecordViews for Stores {
    fn view(&self, group: usize, tensor: usize, record: usize) -> Tensor {
        Stores::view(self, group, tensor, record)
            .unwrap_or_else(|| panic!("invariant violated: a promotion copies into record {record} of group {group}, which the bank holds no storage for"))
    }
}

/// What a store commits: the slots its records serve, its growth running,
/// a shrink's evicted records waiting for the passes that may read them,
/// and a shrink done until an advance reports it.
pub(super) struct Commitment {
    /// Written by whoever resizes: a shrink on the submitting thread, a
    /// growth's slots on the resize thread as each joins.
    pub(super) slots: Arc<AtomicU64>,
    pub(super) resize: Resize,
    pub(super) releasing: Option<Releasing>,
    pub(super) shrunk: Option<BankResized>,
    /// A shrink begun beside a running chain the domain has not yet taken
    /// ([`ExpertBank::take_running_begun`]).
    pub(super) running: Option<RunningBegun>,
}

impl Commitment {
    pub(super) fn new(slots: Arc<AtomicU64>) -> Self {
        Self { slots, resize: Resize::Idle, releasing: None, shrunk: None, running: None }
    }

    /// Whether a resize of the store runs until an advance finishes it: a
    /// growth, a shrink's release, or a shrink not yet reported.
    fn resizing(&self) -> bool {
        matches!(self.resize, Resize::Growing(_)) || self.releasing.is_some() || self.shrunk.is_some()
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
    /// The growth running is its backing to the plan's host slots as the
    /// bank opens ([`ExpertBank::back_host`]), which no domain resize began.
    pub(super) backing: bool,
    /// Its host slots' last finished resize
    /// ([`crate::domain::ElasticBank::record_resize`]).
    pub(super) last_resize: Option<crate::domain::BankResize>,
}

/// A store's resize: none, a growth on its resize thread, or broken (a
/// resize step panicked or broke an invariant: none runs again, and each
/// asked fails with why).
pub(super) enum Resize {
    Idle,
    Growing(Growing),
    Broken(String),
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
    stores: &Stores,
    (from, to): (u64, u64),
    records: &RecordsAt,
    joining: Joining,
    changes: &Arc<Changes>,
    post: impl Fn(Join),
    (slots, rises): (&AtomicU64, &Mutex<()>),
    stop: &AtomicBool,
    begin: impl Fn() -> Option<S>,
) -> Grown {
    let groups = stores.groups.len();
    let mut reached = from;
    let mut joined: Option<(Instant, Instant)> = None;
    let mut spent = Spent::default();
    // The views of the records the host lands into, which each slot's join
    // carries as they are then.
    let mut views = match joining {
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
        spent.slots += 1;
        let claiming = Instant::now();
        let claimed = changes.back(|_| {
            let _rises = rises.lock().expect("the bank's charge rises");
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
        });
        let claimed = match claimed {
            Ok(claimed) => claimed,
            Err(refusal) => return Grown { at: reached, joined, cut: Cut::Refused(refusal), spent },
        };
        let joining_at = Instant::now();
        spent.claim += joining_at - claiming;
        let join = match &mut views {
            None => Join::Bank { records: claimed },
            Some(views) => {
                let formed = claimed.iter().enumerate().try_for_each(|(group, records)| {
                    for &record in records {
                        for (region, view) in views[group].iter_mut().enumerate() {
                            *view = view.with(record as u64, stores.view(group, region, record))?;
                        }
                    }
                    Ok::<_, String>(())
                });
                let records = claimed
                    .iter()
                    .map(|records| {
                        records.iter().map(|&record| u32::try_from(record).map_err(|_| "landed records exceed u32".to_owned())).collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<Vec<_>, _>>();
                match formed.and(records) {
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
    /// acquired, and where its time went; or why the store is broken (its
    /// step panicked, or a slot claimed could not join).
    pub(super) fn ended(&self, holding: BankHolding, charged: impl FnOnce() -> u64) -> Option<Result<(BankResized, Spent), String>> {
        let (outcome, released) = self.done.lock().expect("bank growth outcome").take()?;
        let Grown { at, joined, cut, spent } = match outcome {
            Ok(grown) => grown,
            Err(panicked) => return Some(Err(panicked)),
        };
        let refused = match cut {
            Cut::None => None,
            Cut::Refused(refusal) => Some(refusal),
            Cut::Unjoined(reason) => return Some(Err(reason)),
        };
        // No slot joined: refused at its first, or stopped before it.
        let Some((first, last)) = joined else {
            return Some(Ok(match refused {
                Some(refusal) => (BankResized::GrowthRefused { holding, at, refusal }, spent),
                None => (BankResized::Stopped { holding, at }, spent),
            }));
        };
        let Some(grown) = charged().checked_sub(self.charged) else {
            return Some(Err("a growth lowered its records' charge".to_owned()));
        };
        let zero = Duration::ZERO;
        let begun = Begun { decided: self.decided, queued: zero, planned: zero, trip: Trip::default() };
        let times = begun.times(first, last, released);
        Some(Ok((BankResized::Grown { holding, from: self.from, to: at, grown, refused, times }, spent)))
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
    pub(super) fn grow_host(&self, to: u64, wake: CompletionWake) -> Result<BankStep, DomainError> {
        Ok(match self.host_growth(to, wake, false)? {
            true => BankStep::Begun(BankTier::Host),
            false => BankStep::Running,
        })
    }

    /// Back the host tier from the floor it opens at to the plan's host
    /// slots, its load-time size (`ExpertBankPlan::host_slots`), on its
    /// thread: each host slot joins the tier as soon as every group holds
    /// it, so the load waits for none of it (backing tens of gigabytes of
    /// pinned host storage before the bank opened held a 6.89 GB load of
    /// the 3090 18-26 s, its import 25-33 s against 5-10 s at the floor,
    /// 2026-11-01). It is the open's, no resize: its end reports none.
    pub(super) fn back_host(&self) -> Result<(), DomainError> {
        self.host_growth(self.plan.host_slots, CompletionWake::new(|| {}), true).map(|_| ())
    }

    /// Begin growing the host tier to `to` host slots, the open's backing
    /// where `backing`; false while a resize of it runs, which begins none.
    fn host_growth(&self, to: u64, wake: CompletionWake, backing: bool) -> Result<bool, DomainError> {
        let mut tier = self.host().ok_or_else(|| DomainError::invariant("a host tier growth without one"))?;
        let tier = &mut *tier;
        if let Resize::Broken(reason) = &tier.commitment.resize {
            return Err(DomainError::invariant(reason.clone()));
        }
        if tier.commitment.resizing() {
            return Ok(false);
        }
        let growing = self.growing(&tier.stores, (&tier.commitment.slots, to), BankTier::Host, Joining::Landed, &tier.thread, wake);
        tier.commitment.resize = Resize::Growing(growing);
        tier.backing = backing;
        Ok(true)
    }

    /// Begin growing `stores`, `tier`'s, from their `slots` now to `to` on
    /// the store's resize `thread` ([`grow_by_slots`]), each slot joining
    /// the watcher as `joining` says as soon as every group holds it, its
    /// granules held by the growth's claim (`BankCache::grow`); `wake`
    /// completed once it ends.
    fn growing(
        &self,
        stores: &Arc<Stores>,
        (slots, to): (&Arc<AtomicU64>, u64),
        tier: BankTier,
        joining: Joining,
        thread: &StorageThread,
        wake: CompletionWake,
    ) -> Growing {
        let (decided, from, charged) = (Instant::now(), slots.load(Ordering::Acquire), stores.charged());
        let (shared, slots, stop) = (self.watcher.shared().clone(), slots.clone(), Arc::new(AtomicBool::new(false)));
        let (stopping, rises, stores, records) = (stop.clone(), self.rises.clone(), stores.clone(), self.tier_counts(tier));
        let changes = self.cache.changes().clone();
        let done = thread.run(wake, move || {
            let slot = || shared.backing_slot(&stopping);
            grow_by_slots(&stores, (from, to), &records, joining, &changes, |join| shared.join(join), (&slots, &rises), &stopping, slot)
        });
        Growing { from, charged, decided, stop, done }
    }

    /// End a running growth of the holdings in `tier` storage at its next
    /// slot ([`Growing::stop`]): a shrink of that store waits for it no
    /// longer than that slot's claims. The host tier's backing is its
    /// growth.
    pub(super) fn stop_growth(&self, tier: BankTier) {
        let stop = |resize: &Resize| {
            if let Resize::Growing(growing) = resize {
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

    /// Finish a host tier's resize: its growth once its thread ended it, or
    /// its shrink. Its outcome, and whether a domain resize began it (the
    /// open's backing to its planned slots did not, so its end is reported
    /// to none).
    pub(super) fn advance_host(&self) -> Result<Option<(BankResized, bool)>, DomainError> {
        let Some(mut tier) = self.host() else {
            return Ok(None);
        };
        let tier = &mut *tier;
        if let Some(shrunk) = tier.commitment.shrunk.take() {
            return Ok(Some((shrunk, true)));
        }
        let Resize::Growing(growing) = &tier.commitment.resize else {
            return Ok(None);
        };
        let Some(ended) = growing.ended(BankHolding::Host, || tier.stores.charged()) else {
            return Ok(None);
        };
        match ended {
            Ok((resized, spent)) => {
                let slots = tier.commitment.slots.load(Ordering::Acquire);
                let kind = if tier.backing { "backing" } else { "growth" };
                let (planned, ceiling) = (self.plan.host_slots, tier.ceiling);
                tracing::info!("the host tier's {kind} ended at {slots} of {planned} planned host slots ({ceiling} every expert), {spent}: {resized:?}");
                tier.commitment.resize = Resize::Idle;
                Ok(Some((resized, !std::mem::take(&mut tier.backing))))
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
        self.host().is_some_and(|tier| !tier.backing && tier.commitment.resizing())
    }

    /// Whether the host tier's backing to its planned slots runs
    /// ([`Self::back_host`]), until its advance finishes it.
    pub(super) fn host_backing(&self) -> bool {
        self.host().is_some_and(|tier| tier.backing && matches!(tier.commitment.resize, Resize::Growing(_)))
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
            commitment.shrunk = Some(self.release(stores, tier, (from, to), charged, &records, begun)?);
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

    /// Return a shrink's `evicted` records of `stores` to their pool, the
    /// records charged `charged` before it, then what they freed to the OS:
    /// the pool's empty blocks, and where those return less, its emptiest
    /// blocks emptied for it, their copies enqueued and never waited for
    /// (`seismic::Pool::return_freed`), the bank's
    /// cache of `tier`'s holding cut to what stays above its floor
    /// ([`Self::above_floor`], `BankCache::fit`): one change. The shrink,
    /// done.
    fn release(&self, stores: &Stores, tier: BankTier, (from, to): (u64, u64), charged: u64, evicted: &[Vec<u32>], begun: Begun) -> Result<BankResized, DomainError> {
        let holding = match tier {
            BankTier::Bank => BankHolding::Experts,
            BankTier::Host => BankHolding::Host,
        };
        let releasing = Instant::now();
        let released = self.cache.changes().clone().change(|_| {
            for (group, records) in evicted.iter().enumerate() {
                stores.free(group, records).map_err(|refusal| DomainError::invariant(format!("the expert bank's pool refused a shrink's release: {refusal}")))?;
            }
            let released = charged.checked_sub(stores.charged()).ok_or_else(|| DomainError::invariant("a shrink raised its records' charge"))?;
            self.cache.fit(holding, self.above_floor(tier, to));
            let returned = stores
                .return_freed(released)
                .map_err(|error| DomainError::invariant(format!("the expert bank's pool could not return a shrink's blocks: {error}")))?;
            tracing::debug!(?holding, released, returned, "a shrink's granules return to the pool, its empty blocks to the OS");
            Ok::<_, DomainError>((released, returned))
        })?;
        let (released, returned) = released;
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
        let growing = self.growing(&self.stores, (&commitment.slots, to), BankTier::Bank, joining, &self.resize_thread, wake);
        commitment.resize = Resize::Growing(growing);
        Ok(BankStep::Begun(BankTier::Bank))
    }

    /// [`ElasticBank::advance`]: at a boundary between flights every pass
    /// is submitted, so a shrink's evicted records return to the pool here;
    /// a shrink done is reported, and a growth finishes once its resize
    /// thread ended it.
    pub(super) fn advance(&self) -> Result<Option<BankResized>, DomainError> {
        let mut commitment = self.commitment();
        if let Some(releasing) = commitment.releasing.take() {
            match self.released(releasing) {
                Ok(shrunk) => commitment.shrunk = Some(shrunk),
                Err(error) => {
                    commitment.resize = Resize::Broken(error.to_string());
                    return Err(error);
                }
            }
        }
        if let Some(shrunk) = commitment.shrunk.take() {
            return Ok(Some(shrunk));
        }
        let Resize::Growing(growing) = &commitment.resize else {
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

    /// At a submission boundary of the scratch-served chain whose steps run
    /// from `first` through `last`, every pass of its blocks through
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
    pub(crate) fn release_running(&self, (first, last): (u64, u64), submitted: Option<usize>, pressure: &dyn Fn() -> Option<RunningPressure>) -> Option<usize> {
        let started = Instant::now();
        // Memory is read outside the commitment's lock.
        let short = {
            let commitment = self.commitment();
            matches!(commitment.resize, Resize::Idle) && !commitment.resizing() && commitment.slots.load(Ordering::Acquire) > self.counts.floor()
        };
        let short = short.then(pressure).flatten().filter(|pressure| pressure.bytes > 0);
        let mut commitment = self.commitment();
        if commitment.releasing.is_none() {
            let short = short.filter(|_| !commitment.resizing())?;
            if let Err(reason) = self.evict_beside(&mut commitment, (first, last), short, started) {
                commitment.resize = Resize::Broken(reason);
                return None;
            }
        }
        let releasing = commitment.releasing.as_ref()?;
        // Due once every pass of the decided block is submitted, or at once
        // beside a later chain, whose steps resolve none of them.
        let awaits = (releasing.last == last).then_some(releasing.after).flatten().filter(|&block| submitted.is_none_or(|through| through < block));
        if awaits.is_some() {
            return awaits;
        }
        let releasing = commitment.releasing.take().expect("a release is due");
        match self.released(releasing) {
            Ok(shrunk) => commitment.shrunk = Some(shrunk),
            Err(error) => commitment.resize = Resize::Broken(error.to_string()),
        }
        None
    }

    /// Ask the watcher to evict the bank's records beside the running chain
    /// whose steps run from `first` through `last`, its storage `short` as
    /// read at `started`: from the nearest slots that release storage
    /// toward those releasing what it is short of. Where it evicted, the
    /// commitment's slots follow at once (no step from then on resolves an
    /// evicted record), the domain is owed its beginning
    /// ([`Self::take_running_begun`]), and the records wait for their
    /// release; where none releases storage or the watcher evicted nothing,
    /// nothing changes. Why the records broke, where they did.
    fn evict_beside(&self, commitment: &mut Commitment, (first, last): (u64, u64), short: RunningPressure, started: Instant) -> Result<(), String> {
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
        let Some(Evicted { slots, records, decided: layer, trip }) = self.watcher.shared().shrink(self.evicting(BankTier::Bank), candidates, Some((first, last)))? else {
            return Ok(());
        };
        tracing::info!(from, shallowest, deepest, short = bytes, "the expert bank is shrunk beside a running chain");
        let to = slots as u64;
        commitment.slots.store(to, Ordering::Release);
        assert!(commitment.running.replace(RunningBegun { decided, held: started.elapsed() }).is_none(), "a running shrink's beginning is taken before the next begins");
        let after = layer.map(|layer| self.plan.layers()[layer].sublayer().block as usize);
        let begun = Begun { decided, queued: Duration::ZERO, planned: trip.picked + trip.served, trip };
        commitment.releasing = Some(Releasing { from, to, charged, evicted: records, last, after, begun });
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

    /// [`ElasticBank::running_begun`](crate::domain::ElasticBank::running_begun).
    pub(super) fn take_running_begun(&self) -> Option<RunningBegun> {
        self.commitment().running.take()
    }
}
