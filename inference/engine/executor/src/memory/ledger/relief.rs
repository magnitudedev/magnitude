//! Relief known to cover a need, built whole or not at all: a store's cache
//! sheds, or a request is preempted, only by a plan that covers what it is
//! for, so no partial shed ever drops a prefix or a bank slot for nothing.

use std::collections::BTreeSet;

use crate::operation::RequestId;

use super::admission::{Frees, Reclaim, ReclaimError};
use super::changes::Claiming;
use super::state::{EntryId, State};
use super::view::Ledger;
use super::vocabulary::{CacheKind, Holding, Value};

/// What a relief may take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Relief<'a> {
    /// Cache only, in rank order: a pressure event's.
    CacheOnly,
    /// Cache, then the reservations of those requests its holder can evict
    /// now (the lowest value first, never the earliest admitted): a plan
    /// names no other, so its holder carries out every plan it is given.
    CacheAndRequests(&'a BTreeSet<RequestId>),
}

/// Relief known to cover `need`: cache entries in rank order, then
/// preemptible requests, just enough. No partial plan exists
/// ([`Ledger::plan_relief`]); its one use is [`Ledger::execute`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use]
pub struct ReliefPlan {
    store: seismic::StoreIdentity,
    yields: Vec<(EntryId, u64)>,
    preempt: Vec<RequestId>,
    covers: u64,
}

/// What executing a plan leaves its holder to do: the requests to preempt,
/// the cache bytes it shed (which their holders free), the cache bytes
/// already freed synchronously by [`Reclaim::Backed`] reclaimers, and any
/// [`Reclaim::Backed`] entries whose reclaimer reported `AtFloor` / `0` bytes
/// so callers can exclude them on the next relief pass.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Owed {
    pub yielded: u64,
    pub reclaimed: u64,
    pub exhausted: Vec<EntryId>,
    pub preempt: Vec<RequestId>,
}

impl ReliefPlan {
    /// The requests it preempts, the lowest value first.
    pub fn preempt(&self) -> &[RequestId] {
        &self.preempt
    }

    /// The bytes it covers.
    pub fn covers(&self) -> u64 {
        self.covers
    }

    /// The store it relieves.
    pub fn store(&self) -> seismic::StoreIdentity {
        self.store.clone()
    }
}

impl State {
    /// The relief that covers `need` in `store` for what `frees` names,
    /// taking what `allow` admits; none where all of it together does not.
    pub(super) fn plan_relief(&self, store: &seismic::StoreIdentity, need: u64, allow: Relief, frees: Frees) -> Option<ReliefPlan> {
        self.plan_relief_excluding(store, need, allow, frees, &BTreeSet::new())
    }

    /// Like [`Self::plan_relief`], skipping entries in `exhausted` whose
    /// reclaimers already reported `AtFloor` or `0` freed bytes.
    pub(super) fn plan_relief_excluding(
        &self,
        store: &seismic::StoreIdentity,
        need: u64,
        allow: Relief,
        frees: Frees,
        exhausted: &BTreeSet<EntryId>,
    ) -> Option<ReliefPlan> {
        let mut order: Vec<(Value, CacheKind, EntryId, u64)> = self
            .held
            .iter()
            .filter(|(id, _)| !exhausted.contains(id))
            .filter_map(|(id, entry)| match entry.holding {
                Holding::Cache(kind) if entry.bytes > 0 => Some((entry.value, kind, *id, entry.bytes)),
                Holding::Pool(_) if entry.bytes > 0 => Some((entry.value, CacheKind::Spare, *id, entry.bytes)),
                _ => None,
            })
            .collect();
        order.sort();
        let mut plan = ReliefPlan { store: store.clone(), yields: Vec::new(), preempt: Vec::new(), covers: 0 };
        for (_, _, id, bytes) in order {
            if plan.covers >= need {
                break;
            }
            let yields = bytes.min(need - plan.covers);
            plan.yields.push((id, yields));
            plan.covers += yields;
        }
        if let Relief::CacheAndRequests(evictable) = allow {
            for (sequence, bytes) in self.preemptible_by_value(frees).into_iter().filter(|(sequence, _)| evictable.contains(sequence)) {
                if plan.covers >= need {
                    break;
                }
                plan.preempt.push(sequence);
                plan.covers = plan.covers.saturating_add(bytes);
            }
        }
        (plan.covers >= need).then_some(plan)
    }
}

impl Ledger {
    /// The relief that covers `need` bytes in the store for what `frees`
    /// names, taking what `allow` admits; none where nothing the store holds
    /// covers it, and nothing is taken then.
    pub fn plan_relief(&self, need: u64, allow: Relief, frees: Frees) -> Option<ReliefPlan> {
        self.state().plan_relief(&self.store.identity, need, allow, frees)
    }

    /// Like [`Self::plan_relief`], skipping entries in `exhausted`.
    pub fn plan_relief_excluding(&self, need: u64, allow: Relief, frees: Frees, exhausted: &BTreeSet<EntryId>) -> Option<ReliefPlan> {
        self.state().plan_relief_excluding(&self.store.identity, need, allow, frees, exhausted)
    }

    /// Carry out `plan` whole, in `claiming`, a change's claim phase: its
    /// cache yields now, each holder freeing
    /// what was taken of it by its own rule; its requests are the holder's
    /// to preempt. The only way cache yields to anything but a claim. An
    /// entry its holder let go since the plan was made freed its bytes
    /// already, so the plan still covers its need.
    ///
    /// # Panics
    /// For a plan of another store.
    pub fn execute(&self, claiming: &Claiming, plan: ReliefPlan) -> Owed {
        claiming.of_scope(&self.store.changes);
        assert_eq!(plan.store, self.store.identity, "invariant violated: a relief plan executes in its own store");
        let mut yielded = 0u64;
        let mut reclaimed = 0u64;
        let mut exhausted = Vec::new();
        for (id, bytes) in plan.yields {
            let action = {
                let mut state = self.state();
                let Some(entry) = state.held.get(&id).copied() else { continue };
                let need = bytes.min(entry.bytes);
                if need == 0 || !matches!(entry.holding, Holding::Cache(_) | Holding::Pool(_)) {
                    continue;
                }
                match state.reclaim.get(&id) {
                    Some(Reclaim::Backed(weak)) => match weak.upgrade() {
                        Some(reclaimer) => Some((reclaimer, need)),
                        None => {
                            exhausted.push(id);
                            None
                        }
                    },
                    Some(Reclaim::Deferred | Reclaim::Unbacked) | None => {
                        state.yields(id, need, false, false);
                        yielded += need;
                        None
                    }
                }
            };
            if let Some((reclaimer, need)) = action {
                match reclaimer.reclaim(need) {
                    Ok(0) | Err(ReclaimError::AtFloor) => {
                        exhausted.push(id);
                    }
                    Ok(freed) => {
                        reclaimed += freed;
                        yielded += freed;
                    }
                    Err(ReclaimError::Invariant(reason)) => {
                        panic!("invariant violated: relief cache reclaim failed for entry {id} (need {need} bytes): {reason}");
                    }
                }
            }
        }
        Owed { yielded, reclaimed, exhausted, preempt: plan.preempt }
    }
}
