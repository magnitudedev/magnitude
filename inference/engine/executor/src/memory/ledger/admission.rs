//! Admission: how a store's state takes and yields. Nothing is preempted
//! here: a claim short of memory, a request's growth included, is relieved
//! only by a plan ([`super::relief`]).

use std::collections::BTreeSet;
use std::num::NonZeroU64;
use std::sync::{Arc, Weak};

use crate::operation::RequestId;

use super::estimate::Read;
use super::state::{EntryId, State};
use super::view::Ledger;
use super::vocabulary::{CacheKind, CommittedKind, Holding, PoolClass, Value};

/// Failure while synchronously reclaiming physical memory from a cache entry
/// or granule pool.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReclaimError {
    /// The holding reached its floor or has no further reclaimable blocks:
    /// admission treats this as a real shortfall once all eligible reclaimers
    /// are exhausted.
    AtFloor,
    /// An internal or device invariant broke while freeing memory: admission
    /// panics rather than warning and continuing.
    Invariant(String),
}

/// Reclaims physical memory backing a cache entry or granule pool in a store,
/// synchronously freeing at least `need` bytes back to the OS (down to floor),
/// waiting on any bounded in-flight fence or growth-stop if needed.
///
/// Attached at cache creation via [`Reclaim::Backed`] and invoked OUTSIDE the
/// ledger mutex on host admission paths only.
pub trait CacheReclaimer: Send + Sync {
    fn reclaim(&self, need: u64) -> Result<u64, ReclaimError>;
}

/// How a cache entry frees its physical memory when admission needs its room.
///
/// Every cache entry carries a `Reclaim` at creation — there is no
/// "unregistered physical cache" state.
#[derive(Clone)]
pub enum Reclaim {
    /// Physical memory freed synchronously by [`CacheReclaimer`] outside the
    /// ledger mutex before any claim is handed out.
    Backed(Weak<dyn CacheReclaimer>),
    /// Holds no physical memory of its own (`allocated == 0`, enforced at
    /// creation and in `entry_take`): logical cache accounting or granules
    /// freed by their own pool's reclaimer.
    Unbacked,
    /// Physical memory freed asynchronously via a [`ReliefPlan`] rather than
    /// synchronously on admission. Admission never hands out a [`Claim`]
    /// against `Deferred` bytes while `allocatable < alloc_needed`.
    Deferred,
}

/// A claim refused outright, of any kind: a load's allocation
/// ([`DeviceLedgers::claim`]) or a need ([`Ledger::claim`]): `store` is
/// `over` bytes short of the `required` the allocation adds there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Short {
    pub store: seismic::StoreIdentity,
    pub required: u64,
    pub over: u64,
}

/// A need refused: committed memory it may not preempt, with the need,
/// exceeds the budget by `bytes`. Cache never refuses a claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Over {
    pub bytes: u64,
}

impl Over {
    /// The refusal of `required` bytes in `store` this many short.
    pub(super) fn of(self, store: &seismic::StoreIdentity, required: u64) -> Short {
        Short { store: store.clone(), required, over: self.bytes }
    }
}

/// What a relief frees room for, which decides what a preempted request
/// frees for it ([`State::preemptible_by_value`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Frees {
    /// A request's state, which draws on the load's request slot first.
    RequestState,
    /// Anything else: room beside the slot.
    Room,
}

/// Admission by the budget's figures alone: the check inside every take,
/// and the arena's ([`Ledger::admit_arena`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admit {
    /// The need fits beside every committed claim (cache yields to it).
    Admit,
    Refuse(Over),
}

impl State {
    /// The requests a relief may preempt, never the earliest admitted (it
    /// always progresses), each with the bytes its preemption frees for
    /// `frees`: all it holds for a request's state, which draws on the slot
    /// first; for anything else, what it holds beyond what it drew of a
    /// request slot, which returns to the slot. None that frees nothing.
    /// Lowest value first ([`Reservation::value`]), of equal value the
    /// latest admitted first.
    pub(super) fn preemptible_by_value(&self, frees: Frees) -> Vec<(RequestId, u64)> {
        let drawn = |sequence: RequestId| {
            self.lent
                .iter()
                .filter(|(id, _)| self.held[*id].holding == Holding::Committed(CommittedKind::Request(sequence)))
                .map(|(_, (_, drawn))| *drawn)
                .sum::<u64>()
        };
        let mut sequences: Vec<(Value, RequestId, u64)> = self
            .state_of
            .iter()
            .skip(1)
            .map(|(sequence, bytes)| {
                let frees = match frees {
                    Frees::RequestState => *bytes,
                    Frees::Room => bytes - drawn(*sequence),
                };
                (self.value_of(*sequence), *sequence, frees)
            })
            .filter(|(_, _, frees)| *frees > 0)
            .collect();
        sequences.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
        sequences.into_iter().map(|(_, sequence, bytes)| (sequence, bytes)).collect()
    }

    /// Whether free room at `at` and the cache cover `bytes` more committed
    /// memory.
    pub(super) fn covers(&self, at: Read, bytes: u64) -> Admit {
        if bytes == 0 {
            return Admit::Admit;
        }
        let covered = self.estimate_at(at).free() + self.cache;
        match bytes.checked_sub(covered) {
            Some(over) if over > 0 => Admit::Refuse(Over { bytes: over }),
            _ => Admit::Admit,
        }
    }

    /// Free room at `at`; none asked of none.
    pub(super) fn free(&self, at: Read, bytes: u64) -> u64 {
        match bytes {
            0 => 0,
            _ => self.estimate_at(at).free(),
        }
    }

    /// Free room at `at` `holding` takes `bytes` of before cache yields to
    /// it: what lies beside the guarantee (others' rise), whose own room a
    /// claim takes only once cache is gone ([`Self::take`]); all of it for
    /// a pool's block slack, its own spare granules. So no change of the
    /// process's own spends the room the bank's target leaves for others'
    /// rise: the bank owes what such a change took of it, and sheds it
    /// whatever the room's threshold says (`experts::derived::held_target`).
    pub(super) fn free_for(&self, at: Read, holding: Holding, bytes: u64) -> u64 {
        let free = self.free(at, bytes);
        match holding {
            Holding::Pool(_) => free,
            _ => free.saturating_sub(self.estimate_at(at).guarantee()),
        }
    }

    /// Where `took`, bytes a driver took for `holding` (never refused),
    /// left them in the guarantee's room, the cache of the kinds `of`
    /// admits yields what they took of it to the holding, worth least
    /// first, as it yields to a claim ([`Self::give`]): the bank owes it.
    /// What others' movement took of the room is no part of it.
    pub(super) fn keep_guarantee(&mut self, holding: Holding, took: u64, of: impl Fn(CacheKind) -> bool) {
        let Some(at) = self.read else {
            return;
        };
        // A pool's entry grows by block slack, its own spare granules: no
        // cache of another holding gives way to it.
        if matches!(holding, Holding::Pool(_)) {
            return;
        }
        let lacks = self.estimate_at(at).over_guarantee().min(took).min(self.cache);
        if lacks > 0 {
            self.give(lacks, of, Some(holding));
        }
    }

    /// Hold `bytes` more committed memory of `kind` in a new committed
    /// entry, in the one order the holding kinds give: free room beside the
    /// guarantee, then cache in rank order, then (only once cache is gone)
    /// the guarantee's own room. Refused past the budget. Admitted at `at`
    /// ([`Self::reading`]).
    pub(super) fn take(&mut self, at: Read, kind: CommittedKind, bytes: u64) -> Result<EntryId, Over> {
        if let Admit::Refuse(over) = self.covers(at, bytes) {
            return Err(over);
        }
        let free = self.free(at, bytes);
        let beside = self.free_for(at, Holding::Committed(kind), bytes);
        let holding = Holding::Committed(kind);
        let same_pool = |class: PoolClass| holding.pooled() && holding.pool_class() == Some(class);
        let yieldable: u64 = self
            .held
            .values()
            .filter_map(|entry| match entry.holding {
                Holding::Cache(_) if entry.bytes > 0 => Some(entry.bytes),
                Holding::Pool(class) if entry.bytes > 0 && same_pool(class) => Some(entry.bytes),
                _ => None,
            })
            .sum();
        if free.saturating_add(yieldable) < bytes {
            return Err(Over { bytes: bytes - free.saturating_add(yieldable) });
        }
        let owed = self.give(bytes.saturating_sub(beside), |_| true, Some(holding)).saturating_sub(free - beside);
        debug_assert_eq!(owed, 0, "yieldable cache check guarantees give covers the shortfall");
        Ok(self.insert(holding, bytes))
    }

    /// Yield up to `bytes` of the cache of the kinds `of` admits, in the one
    /// order cache yields in: worth least per byte first ([`Value`]); of
    /// equal worth, in [`CacheKind`]'s order, then the oldest first. What is
    /// left of `bytes` once that cache is gone.
    /// `to` is the holding taking them, none for a shortage: a pooled one
    /// takes pooled cache's granules in place. Unreturned [`Holding::Pool`]
    /// bytes never yield to a non-pooled or different-pool consumer here:
    /// they can only be reclaimed synchronously via their [`CacheReclaimer`]
    /// (`Pool::return_freed_sync`) so blocks physically drop to the driver
    /// before any claim is minted.
    pub(super) fn give(&mut self, mut bytes: u64, of: impl Fn(CacheKind) -> bool, to: Option<Holding>) -> u64 {
        let same_pool = |class: PoolClass| to.is_some_and(|to| to.pooled() && to.pool_class() == Some(class));
        let mut order: Vec<(Value, CacheKind, EntryId)> = self
            .held
            .iter()
            .filter_map(|(id, entry)| match entry.holding {
                Holding::Cache(kind) if of(kind) && entry.bytes > 0 => Some((entry.value, kind, *id)),
                Holding::Pool(class) if of(CacheKind::Spare) && entry.bytes > 0 && same_pool(class) => {
                    Some((entry.value, CacheKind::Spare, *id))
                }
                _ => None,
            })
            .collect();
        order.sort();
        for (_, _, id) in order {
            if bytes == 0 {
                break;
            }
            let yields = self.held[&id].bytes.min(bytes);
            // A pooled holding takes the granules of its own class's pool
            // in place.
            let class = |holding: Holding| holding.pool_class();
            let within = to.is_some_and(|to| to.pooled() && class(to).is_some() && class(to) == class(self.held[&id].holding));
            self.yields(id, yields, within, to.is_some());
            bytes -= yields;
        }
        bytes
    }
}

/// How an admission-reclaimable candidate can give up its bytes.
enum CandidateReclaim {
    /// Physical memory freed synchronously by a live [`CacheReclaimer`].
    Backed(Arc<dyn CacheReclaimer>),
    /// Logical cache holding zero physical bytes (`Reclaim::Unbacked`).
    Unbacked,
}

/// One eligible cache or pool entry considered during admission reclamation,
/// ordered by `(value, kind, id)` (worth least per byte first, then `CacheKind`
/// order, then oldest first).
struct Candidate {
    value: Value,
    kind: CacheKind,
    id: EntryId,
    bytes: u64,
    releasing: u64,
    reclaim: CandidateReclaim,
}

/// One step of [`Ledger::reclaim_until`].
enum ReclaimStep<T> {
    Done(T),
    Reclaim {
        id: EntryId,
        reclaimer: Arc<dyn CacheReclaimer>,
        need: u64,
    },
}

/// How [`Ledger::admit_or_reclaim`] judges and reclaims cache for a claim.
pub(super) struct AdmitPolicy<F> {
    pub(super) at: Option<Read>,
    pub(super) kind: CommittedKind,
    pub(super) bytes: u64,
    /// Which cache kinds may yield to this claim.
    pub(super) of: F,
    /// Whether admission is bounded by the store's boundary cap B.
    pub(super) within_boundary: bool,
    /// Whether this claim may spend the guarantee's room once cache is gone.
    pub(super) allow_guarantee: bool,
}

impl State {
    /// Collect eligible cache and pool candidates in rank order `(value, kind, id)`.
    fn reclaim_candidates(
        &self,
        own_pool: Option<PoolClass>,
        of: &impl Fn(CacheKind) -> bool,
        need_own_pool_flush: bool,
        exhausted: &BTreeSet<EntryId>,
    ) -> Vec<Candidate> {
        let mut candidates: Vec<Candidate> = self
            .held
            .iter()
            .filter_map(|(id, entry)| {
                let (kind, bytes, releasing) = match entry.holding {
                    Holding::Cache(ck) if of(ck) && entry.bytes > 0 => (ck, entry.bytes, 0),
                    Holding::Pool(class) if Some(class) != own_pool && of(CacheKind::Spare) && (entry.bytes > 0 || entry.releasing > 0) => {
                        (CacheKind::Spare, entry.bytes, entry.releasing)
                    }
                    Holding::Pool(class) if Some(class) == own_pool && entry.releasing > 0 && need_own_pool_flush => {
                        (CacheKind::Spare, 0, entry.releasing)
                    }
                    _ => return None,
                };
                let reclaim = match self.reclaim.get(id) {
                    Some(Reclaim::Backed(weak)) => {
                        let live = (!exhausted.contains(id)).then(|| weak.upgrade()).flatten()?;
                        CandidateReclaim::Backed(live)
                    }
                    Some(Reclaim::Unbacked) => {
                        assert!(
                            !matches!(entry.holding, Holding::Pool(_)),
                            "invariant violated: Holding::Pool entry {id} cannot use Reclaim::Unbacked"
                        );
                        CandidateReclaim::Unbacked
                    }
                    Some(Reclaim::Deferred) | None => return None,
                };
                Some(Candidate { value: entry.value, kind, id: *id, bytes, releasing, reclaim })
            })
            .collect();
        candidates.sort_by_key(|c| (c.value, c.kind, c.id));
        candidates
    }

    /// Yield up to `bytes` of [`Reclaim::Unbacked`] cache entries matching `of`
    /// (entries that hold no physical memory of their own). Returns any
    /// remaining bytes not covered by unbacked cache.
    fn give_unbacked(&mut self, mut bytes: u64, of: &impl Fn(CacheKind) -> bool, to: Holding) -> u64 {
        let mut order: Vec<(Value, CacheKind, EntryId)> = self
            .held
            .iter()
            .filter_map(|(id, entry)| match (entry.holding, self.reclaim.get(id)) {
                (Holding::Cache(kind), Some(Reclaim::Unbacked)) if of(kind) && entry.bytes > 0 => Some((entry.value, kind, *id)),
                _ => None,
            })
            .collect();
        order.sort();
        for (_, _, id) in order {
            if bytes == 0 {
                break;
            }
            let yields = self.held[&id].bytes.min(bytes);
            let within = to.pooled() && to.pool_class().is_some() && to.pool_class() == self.held[&id].holding.pool_class();
            self.yields(id, yields, within, true);
            bytes -= yields;
        }
        bytes
    }

    /// One lock-held admission attempt for `policy`: either admits `policy.bytes`
    /// (when physical `allocatable >= alloc_needed` and free/unbacked room cover
    /// it), requests a synchronous reclaim outside the lock, or refuses with
    /// [`Short`].
    fn try_admit<F: Fn(CacheKind) -> bool>(
        &mut self,
        store: &seismic::StoreIdentity,
        policy: &AdmitPolicy<F>,
        exhausted: &BTreeSet<EntryId>,
        reclaimed: u64,
    ) -> ReclaimStep<Result<EntryId, Short>> {
        let read = self.reading(store, policy.at);
        let estimate = self.estimate_at(read);
        let holding = Holding::Committed(policy.kind);
        let own_pool = holding.pool_class().filter(|_| holding.pooled());
        // `Backing(PoolClass::State)` is only requested when the state store
        // must allocate a fresh slab block (no free granules exist in the pool):
        // any non-zero `spare_of(State)` is uncounted live slabs, not reusable
        // free granules, so it must not mask `external` demand from reclaim.
        let own_spare = match holding {
            Holding::Committed(CommittedKind::Backing(PoolClass::State)) => 0,
            _ => own_pool.map_or(0, |class| self.spare_of(class)),
        };
        let external = policy.bytes.saturating_sub(own_spare);
        if external == 0 {
            return ReclaimStep::Done(Ok(self.insert(holding, policy.bytes)));
        }
        let alloc_needed = match holding {
            Holding::Committed(CommittedKind::Backing(class)) => match self.pools[class.index()].as_ref() {
                Some(pool) => external.div_ceil(pool.block.get()).saturating_mul(pool.block.get()),
                None => external,
            },
            _ => external,
        };
        let budget = estimate.budget();
        let held_now = estimate.held_now();
        let free = estimate.free();
        let beside = {
            let raw = self.free_for(read, holding, external).saturating_add(reclaimed).min(free);
            if policy.within_boundary {
                let within = estimate.cap().map_or(u64::MAX, |cap| cap.saturating_sub(held_now));
                raw.min(within)
            } else {
                raw
            }
        };
        // Compute deficits without saturating away any over-budget overhang
        // (`held_now > budget` or `allocated + credited > budget`) so an
        // over-budget admission requests the full deficit in one reclaim call
        // (or fails fast with `Short`) instead of looping one slot at a time.
        let need_alloc = estimate.allocated().saturating_add(estimate.credited()).saturating_add(alloc_needed).saturating_sub(budget);
        let need_budget_ext = held_now.saturating_add(external).saturating_sub(budget);
        let need_budget_alloc = held_now.saturating_add(alloc_needed).saturating_sub(budget);
        let need_cap = if policy.within_boundary {
            estimate.cap().map_or(0, |cap| held_now.saturating_add(external).saturating_sub(cap))
        } else {
            0
        };
        let candidates = self.reclaim_candidates(own_pool, &policy.of, need_alloc > 0, exhausted);
        let reclaimable_room: u64 = candidates.iter().map(|c| c.bytes).sum();
        let reclaimable_shed: u64 = candidates.iter().map(|c| c.releasing).sum::<u64>().min(estimate.shed());
        let max_reclaimable_held = reclaimable_room.saturating_add(reclaimable_shed);
        let min_need_held = if policy.allow_guarantee {
            need_budget_ext
        } else {
            external.saturating_sub(beside).max(need_budget_ext)
        }
        .max(need_cap);
        if min_need_held > max_reclaimable_held {
            self.waiting = true;
            return ReclaimStep::Done(Err(Over { bytes: min_need_held - max_reclaimable_held }.of(store, policy.bytes)));
        }

        let need_room = external.saturating_sub(beside).max(need_budget_alloc).max(need_cap);
        let need_reclaim = need_room.max(need_alloc);

        if need_reclaim == 0 {
            return ReclaimStep::Done(Ok(self.insert(holding, policy.bytes)));
        }

        if let Some((id, reclaimer)) = candidates.into_iter().find_map(|c| match c.reclaim {
            CandidateReclaim::Backed(reclaimer) => Some((c.id, reclaimer)),
            CandidateReclaim::Unbacked => None,
        }) {
            return ReclaimStep::Reclaim { id, reclaimer, need: need_reclaim };
        }

        // No live reclaimer remains: only `Reclaim::Unbacked` cache (which
        // holds 0 physical bytes) and guarantee room (when allowed) may be
        // taken, and ONLY when physical `allocatable >= alloc_needed` already
        // holds. `Reclaim::Deferred` cache is never taken here without being
        // physically freed first.
        if need_alloc > 0 {
            self.waiting = true;
            return ReclaimStep::Done(Err(Over { bytes: need_alloc }.of(store, policy.bytes)));
        }
        let owed = self.give_unbacked(need_room, &policy.of, holding);
        let short = if policy.allow_guarantee { owed.saturating_sub(free - beside) } else { owed };
        if short > 0 {
            self.waiting = true;
            return ReclaimStep::Done(Err(Over { bytes: short }.of(store, policy.bytes)));
        }
        ReclaimStep::Done(Ok(self.insert(holding, policy.bytes)))
    }
}

impl Ledger {
    /// Register `reclaimer` for the store's granule pool of `class`, owned for
    /// the pool's lifetime and stored as [`Reclaim::Backed`].
    pub(super) fn register_pool_reclaimer(&self, class: PoolClass, reclaimer: Arc<dyn CacheReclaimer>) {
        let mut state = self.state();
        if let Some(pool) = state.pools[class.index()].as_mut() {
            let entry = pool.entry;
            let weak = Arc::downgrade(&reclaimer);
            pool.reclaimer = Some(reclaimer);
            state.reclaim.insert(entry, Reclaim::Backed(weak));
        }
    }

    /// The single reclaim loop shared by [`Self::admit_or_reclaim`],
    /// [`Self::admit_units`], and [`Self::reclaim_and_draw`].
    fn reclaim_until<T>(&self, mut step: impl FnMut(&mut State, &BTreeSet<EntryId>, u64) -> ReclaimStep<T>) -> T {
        let mut exhausted: BTreeSet<EntryId> = BTreeSet::new();
        let mut reclaimed: u64 = 0;
        loop {
            let next = {
                let mut state = self.state();
                step(&mut state, &exhausted, reclaimed)
            };
            match next {
                ReclaimStep::Done(done) => return done,
                ReclaimStep::Reclaim { id, reclaimer, need } => match reclaimer.reclaim(need) {
                    Ok(0) | Err(ReclaimError::AtFloor) => {
                        exhausted.insert(id);
                    }
                    Ok(freed) => {
                        reclaimed = reclaimed.saturating_add(freed);
                    }
                    Err(ReclaimError::Invariant(reason)) => {
                        panic!("invariant violated: cache reclaim failed for entry {id} (need {need} bytes): {reason}");
                    }
                },
            }
        }
    }

    /// The one allocation-admission owner: synchronously reclaims cache
    /// entries with [`Reclaim::Backed`] reclaimers OUTSIDE the ledger mutex
    /// until physical `allocatable` and free room cover `policy.bytes` (or
    /// returns [`Short`] for a real shortfall below floor), then inserts the
    /// committed entry under the lock.
    pub(super) fn admit_or_reclaim<F: Fn(CacheKind) -> bool>(&self, policy: AdmitPolicy<F>) -> Result<EntryId, Short> {
        self.reclaim_until(|state, exhausted, reclaimed| state.try_admit(&self.store.identity, &policy, exhausted, reclaimed))
    }

    /// Hold up to `most` whole `unit`s of committed memory of `kind`,
    /// synchronously reclaiming eligible [`Reclaim::Backed`] cache OUTSIDE the
    /// ledger mutex via [`Self::reclaim_until`] before inserting the claim.
    pub(super) fn admit_units(&self, kind: CommittedKind, unit: NonZeroU64, most: u64) -> (EntryId, u64) {
        assert!(!matches!(kind, CommittedKind::Request(_)), "a request's state is reserved by its own path, never in units");
        let yielding = |ck: CacheKind| matches!(ck, CacheKind::Spare | CacheKind::GraphSlots | CacheKind::Bank);
        self.reclaim_until(|state, exhausted, reclaimed| {
            let read = state.reading(&self.store.identity, None);
            let estimate = state.estimate_at(read);
            let holding = Holding::Committed(kind);
            let free = state.free_for(read, holding, most).saturating_add(reclaimed).min(state.free(read, most));
            let candidates = state.reclaim_candidates(None, &yielding, false, exhausted);
            let reclaimable_room: u64 = candidates.iter().map(|c| c.bytes).sum();
            let reclaimable_shed: u64 = candidates.iter().map(|c| c.releasing).sum::<u64>().min(estimate.shed());
            let held_overhang = estimate.held_now().saturating_sub(estimate.budget());
            let cover = free
                .saturating_add(reclaimable_room)
                .saturating_add(reclaimable_shed)
                .saturating_sub(held_overhang);
            let unbacked_room: u64 = candidates
                .iter()
                .filter(|c| matches!(c.reclaim, CandidateReclaim::Unbacked))
                .map(|c| c.bytes)
                .sum();
            let units = most.min(cover / unit.get());
            let bytes = units * unit.get();
            if bytes == 0 {
                return ReclaimStep::Done((state.insert(holding, 0), 0));
            }
            let budget = estimate.budget();
            let need_held = estimate.held_now().saturating_add(bytes).saturating_sub(budget);
            let need_alloc = estimate.allocated().saturating_add(estimate.credited()).saturating_add(bytes).saturating_sub(budget);
            let need_reclaim = bytes.saturating_sub(free).max(need_held).max(need_alloc);
            if need_reclaim == 0 {
                return ReclaimStep::Done((state.insert(holding, bytes), units));
            }
            if let Some((id, reclaimer)) = candidates.into_iter().find_map(|c| match c.reclaim {
                CandidateReclaim::Backed(reclaimer) => Some((c.id, reclaimer)),
                CandidateReclaim::Unbacked => None,
            }) {
                return ReclaimStep::Reclaim { id, reclaimer, need: need_reclaim };
            }
            // No live reclaimer remains: admit only as many units as free +
            // unbacked room and physical `allocatable` actually cover.
            let physical_units = most.min(free.saturating_add(unbacked_room).min(estimate.allocatable()) / unit.get());
            let physical_bytes = physical_units * unit.get();
            let owed = state.give_unbacked(physical_bytes.saturating_sub(free), &yielding, holding);
            assert_eq!(owed, 0, "free and unbacked cache cover the units counted from them");
            ReclaimStep::Done((state.insert(holding, physical_bytes), physical_units))
        })
    }

    /// Draw `bytes` for a request's state (`reserve` / `extend`): recomputes
    /// `drawn` from `slot` and `rem` atomically under the ledger lock on every
    /// [`Self::reclaim_until`] iteration, so `slot` is never read across a
    /// lock drop.
    pub(super) fn reclaim_and_draw(&self, slot: EntryId, into: Option<EntryId>, kind: CommittedKind, bytes: u64) -> Result<EntryId, Short> {
        self.reclaim_until(|state, exhausted, reclaimed| {
            let held = state.entry(slot).bytes;
            let drawn = held.min(bytes);
            let rem = bytes - drawn;
            let beside = if rem == 0 {
                None
            } else {
                let policy = AdmitPolicy {
                    at: None,
                    kind,
                    bytes: rem,
                    of: |ck: CacheKind| !ck.valued(),
                    within_boundary: true,
                    allow_guarantee: false,
                };
                match state.try_admit(&self.store.identity, &policy, exhausted, reclaimed) {
                    ReclaimStep::Reclaim { id, reclaimer, need } => {
                        return ReclaimStep::Reclaim { id, reclaimer, need };
                    }
                    ReclaimStep::Done(Err(short)) => {
                        return ReclaimStep::Done(Err(Short { required: bytes, ..short }));
                    }
                    ReclaimStep::Done(Ok(id)) => Some(id),
                }
            };
            state.resize(slot, held - drawn);
            let id = match into {
                Some(into) => {
                    let has = state.entry(into).bytes;
                    state.resize(into, has + drawn);
                    into
                }
                None => state.insert(Holding::Committed(kind), drawn),
            };
            if let Some(beside) = beside {
                state.merge(beside, id);
            }
            if drawn > 0 {
                let lent = state.lent.entry(id).or_insert((slot, 0));
                lent.1 += drawn;
            }
            ReclaimStep::Done(Ok(id))
        })
    }
}
