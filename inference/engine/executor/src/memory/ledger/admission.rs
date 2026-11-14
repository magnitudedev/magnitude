//! Admission: what a take came to, and how a store's state takes, yields and
//! preempts.

use std::num::NonZeroU64;

use crate::memory::shortage;
use crate::operation::RequestId;

use super::estimate::Read;
use super::state::{EntryId, State};
use super::vocabulary::{CacheKind, CommittedKind, Holding, PoolClass, Value};

/// A claim refused outright, of any kind: a load's allocation
/// ([`DeviceLedgers::claim`]) or a need ([`Ledger::claim`]): `store` is
/// `over` bytes short of the `required` the allocation adds there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Short {
    pub store: seismic::StoreIdentity,
    pub required: u64,
    pub over: u64,
}

/// What a take came to.
pub(super) enum Taken {
    /// The entry holds the whole need.
    Held(EntryId),
    /// The entry holds what free room and cache gave; these sequences'
    /// eviction gives the rest.
    Preempting { entry: EntryId, sequences: Vec<RequestId> },
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

/// Admission by the budget's figures alone: the check inside every take,
/// and the arena's ([`Ledger::admit_arena`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admit {
    /// The need fits beside every committed claim (cache yields to it).
    Admit,
    Refuse(Over),
}

impl State {
    /// [`Self::take`] in `store` at `at` ([`Self::reading`]), refused as
    /// [`Short`].
    pub(super) fn take_in(&mut self, store: &seismic::StoreIdentity, at: Option<Read>, kind: CommittedKind, bytes: u64) -> Result<Taken, Short> {
        let at = self.reading(store, at);
        self.take(at, kind, bytes).map_err(|over| {
            self.waiting = true;
            over.of(store, bytes)
        })
    }

    /// The sequences a shortage may preempt with their state, never the
    /// earliest admitted (it always progresses): lowest value first
    /// ([`Claim::value`]), of equal value the latest admitted first.
    pub(super) fn preemptible_by_value(&self) -> Vec<(RequestId, u64)> {
        let mut sequences: Vec<(Value, RequestId, u64)> =
            self.state_of.iter().skip(1).map(|(sequence, bytes)| (self.value_of(*sequence), *sequence, *bytes)).collect();
        sequences.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
        sequences.into_iter().map(|(_, sequence, bytes)| (sequence, bytes)).collect()
    }

    /// The state entries a claim of `kind` may take: a sequence's growth
    /// preempts only sequences admitted after it, the lowest value first
    /// ([`Claim::value`]), of equal value the latest admitted first; nothing
    /// else preempts.
    fn preemptible(&self, kind: CommittedKind) -> Vec<EntryId> {
        let CommittedKind::State(claimant) = kind else {
            return Vec::new();
        };
        let mut victims: Vec<(Value, RequestId, EntryId)> = self
            .held
            .iter()
            .filter_map(|(id, entry)| match entry.holding {
                Holding::Committed(CommittedKind::State(sequence))
                    if sequence > claimant
                        && entry.bytes > 0
                        && !self.reserved.contains_key(id)
                        && !self.pending.contains(id) =>
                {
                    Some((self.value_of(sequence), sequence, *id))
                }
                _ => None,
            })
            .collect();
        victims.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)).then(b.2.cmp(&a.2)));
        victims.into_iter().map(|(_, _, id)| id).collect()
    }

    /// Whether `bytes` more committed memory of `kind` fits the budget
    /// beside every committed holding it may not preempt: whether free
    /// room, cache and preemptible state cover it.
    ///
    /// The first sequence admitted makes the in-process term live
    /// ([`Self::in_process_left`]): it is admitted only where the room
    /// covers that term beside it, and while a real shortage stands
    /// (`memory::shortage`) only where cache meets the shortage with bytes
    /// left above its floor (§4.5): an idle load at its floor never unloads,
    /// so new work is refused instead.
    pub(super) fn admit(&self, at: Read, kind: CommittedKind, bytes: u64) -> Admit {
        let takable: u64 = self.preemptible(kind).iter().map(|id| self.held[id].bytes).sum();
        let first = matches!(kind, CommittedKind::State(_)) && self.state_of.is_empty();
        let needed = self.raise(kind, bytes).saturating_add(if first { self.in_process } else { 0 });
        if first && needed > 0 {
            let estimate = self.estimate_at(at);
            if let Some(cause) = shortage::cause(&estimate, self.boundary.harmed(), self.boundary.pressed()) {
                let owed = match cause {
                    shortage::Cause::Harm { bytes } => bytes.saturating_sub(self.shed_at_reading),
                    _ => shortage::lacks(&estimate, cause),
                };
                let Some(cache) = self.cache.checked_sub(owed).filter(|cache| *cache > 0) else {
                    return Admit::Refuse(Over { bytes: needed });
                };
                let covered = estimate.free() + cache + takable;
                return match needed.checked_sub(covered) {
                    Some(over) if over > 0 => Admit::Refuse(Over { bytes: over }),
                    _ => Admit::Admit,
                };
            }
        }
        self.covers(at, needed, takable)
    }

    /// What `bytes` more committed memory of `kind` raise what is held: a
    /// sequence's state is held first by the state floor, which is already
    /// held, so only what it adds beyond the floor's unused bytes raises it.
    pub(super) fn raise(&self, kind: CommittedKind, bytes: u64) -> u64 {
        match kind {
            CommittedKind::State(_) => bytes - bytes.min(self.floor.saturating_sub(self.state_total)),
            _ => bytes,
        }
    }

    /// Whether free room at `at`, the cache and `takable` committed bytes
    /// cover `bytes`.
    pub(super) fn covers(&self, at: Read, bytes: u64, takable: u64) -> Admit {
        if bytes == 0 {
            return Admit::Admit;
        }
        let covered = self.estimate_at(at).free() + self.cache + takable;
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

    /// Whether `holding` draws on the guarantee's room: the state's, its
    /// pool's holdings (a sequence's rows, which the in-process term holds
    /// that room for and falls by as they are claimed, `in_process_left`,
    /// the slabs they grow in and those slabs' cache). No other holding of
    /// the process does.
    pub(super) fn draws_guarantee(holding: Holding) -> bool {
        holding.pool_class() == Some(PoolClass::State)
    }

    /// Free room at `at` `holding` takes `bytes` of before cache yields to
    /// it: all of it for one that draws on the guarantee
    /// ([`Self::draws_guarantee`]), else what lies beside the guarantee,
    /// whose own room such a claim takes only once cache is gone
    /// ([`Self::take`]). So no change of the process's own spends the room the bank's
    /// target leaves for the state's growth in process: the bank owes what
    /// such a change took of it, and sheds it whatever the room's
    /// threshold says (`experts::derived::held_target`).
    pub(super) fn free_for(&self, at: Read, holding: Holding, bytes: u64) -> u64 {
        let free = self.free(at, bytes);
        match Self::draws_guarantee(holding) {
            true => free,
            false => free.saturating_sub(self.estimate_at(at).guarantee()),
        }
    }

    /// Where `took`, bytes a driver took for `holding` (never refused),
    /// left them in the guarantee's room, cache yields what they took of
    /// it to the holding, worth least first, as it yields to a claim
    /// ([`Self::give`]): the bank owes it. What others' movement took of
    /// the room is no part of it.
    pub(super) fn keep_guarantee(&mut self, holding: Holding, took: u64) {
        let Some(at) = self.read else {
            return;
        };
        // A pool's entry grows by block slack, its own spare granules: no
        // cache of another holding gives way to it.
        if Self::draws_guarantee(holding) || matches!(holding, Holding::Pool(_)) {
            return;
        }
        let lacks = self.estimate_at(at).over_guarantee().min(took).min(self.cache);
        if lacks > 0 {
            self.give(lacks, |_| true, Some(holding));
        }
    }

    /// Hold `bytes` more committed memory of `kind` in a new committed
    /// entry, in the one order the holding kinds give: free room, then
    /// cache in rank order, then the state of sequences admitted after
    /// the claimant's, the lowest value first and just enough. Where it needs
    /// state, the entry holds what free room and cache gave and the
    /// sequences' entries are reserved to it: their bytes join it as they
    /// are evicted, and no other claim can take them. Admitted at `at`
    /// ([`Self::reading`]).
    pub(super) fn take(&mut self, at: Read, kind: CommittedKind, bytes: u64) -> Result<Taken, Over> {
        if let Admit::Refuse(over) = self.admit(at, kind, bytes) {
            return Err(over);
        }
        let raise = self.raise(kind, bytes);
        // Free room beside the guarantee first, then cache, then (for a
        // holding that does not draw on it) the guarantee's own room, as a
        // last resort: admission refuses only past the budget, and the
        // bank owes what cache gave ([`Self::free_for`]).
        let free = self.free(at, raise);
        let beside = self.free_for(at, Holding::Committed(kind), raise);
        let mut owed = raise.saturating_sub(beside);
        let mut short = raise.saturating_sub(free).saturating_sub(self.cache);
        let mut victims = Vec::new();
        for id in self.preemptible(kind) {
            if short == 0 {
                break;
            }
            short = short.saturating_sub(self.held[&id].bytes);
            victims.push(id);
        }
        assert_eq!(short, 0, "admission counts preemptible state only where it covers the rest");
        owed = self.give(owed, |_| true, Some(Holding::Committed(kind))).saturating_sub(free - beside);
        let entry = self.insert(Holding::Committed(kind), bytes - owed);
        if victims.is_empty() {
            return Ok(Taken::Held(entry));
        }
        let sequences = victims
            .iter()
            .map(|id| match self.held[id].holding {
                Holding::Committed(CommittedKind::State(sequence)) => sequence,
                _ => unreachable!("only state is preemptible"),
            })
            .collect();
        for victim in victims {
            self.reserved.insert(victim, entry);
        }
        self.pending.insert(entry);
        Ok(Taken::Preempting { entry, sequences })
    }

    /// Yield up to `bytes` of the cache of the kinds `of` admits, in the one
    /// order cache yields in: worth least per byte first ([`Value`]); of
    /// equal worth, in [`CacheKind`]'s order, then the oldest first. What is
    /// left of `bytes` once that cache is gone.
    /// `to` is the holding taking them, none for a shortage: a pooled one
    /// takes pooled cache's granules in place.
    pub(super) fn give(&mut self, mut bytes: u64, of: impl Fn(CacheKind) -> bool, to: Option<Holding>) -> u64 {
        let mut order: Vec<(Value, CacheKind, EntryId)> = self
            .held
            .iter()
            .filter_map(|(id, entry)| match entry.holding {
                Holding::Cache(kind) if of(kind) && entry.bytes > 0 => Some((entry.value, kind, *id)),
                Holding::Pool(_) if of(CacheKind::Spare) && entry.bytes > 0 => Some((entry.value, CacheKind::Spare, *id)),
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

    /// The preemption whose entry is `entry` ends with `bytes` held for its
    /// need: every sequence it reserved was evicted. What their eviction
    /// freed beyond the need turns spare.
    pub(super) fn completed(&mut self, entry: EntryId, bytes: u64) {
        assert!(
            !self.reserved.values().any(|to| *to == entry),
            "a preemption completes only once every sequence it named is evicted"
        );
        self.pending.remove(&entry);
        let has = self.held[&entry].bytes;
        let spare = has.checked_sub(bytes).expect("an evicted sequence frees what it held");
        self.resize(entry, bytes);
        if spare > 0 {
            // The evicted sequences' granules stay the pool's allocation:
            // they turn spare, held, as their entries did.
            self.insert(Holding::Cache(CacheKind::Spare), spare);
        }
    }

    /// The preemption whose entry is `entry` was abandoned: its sequences
    /// stand, and what it held is let go.
    pub(super) fn abandoned(&mut self, entry: EntryId) {
        self.reserved.retain(|_, to| *to != entry);
        self.pending.remove(&entry);
        self.release(entry);
    }

    /// Hold up to `most` whole `unit`s of committed memory of `kind`: as
    /// many as free room, then spare granules, idle graph slots and bank
    /// slots above the floor cover now, that cache yielding in rank order
    /// ([`Self::give`]). Never preempts and never refuses. The entry and its
    /// units.
    pub(super) fn take_units(&mut self, store: &seismic::StoreIdentity, kind: CommittedKind, unit: NonZeroU64, most: u64) -> (EntryId, u64) {
        assert!(!matches!(kind, CommittedKind::State(_)), "a sequence's state is claimed by its rows, never in units");
        let yielding = |kind: CacheKind| matches!(kind, CacheKind::Spare | CacheKind::GraphSlots | CacheKind::Bank);
        let free = self.free_for(self.reading(store, None), Holding::Committed(kind), most);
        let cover = CacheKind::ALL.into_iter().filter(|kind| yielding(*kind)).fold(free, |cover, kind| cover.saturating_add(self.cache_of[kind.index()]));
        let units = most.min(cover / unit.get());
        let bytes = units * unit.get();
        let owed = self.give(bytes.saturating_sub(free), yielding, Some(Holding::Committed(kind)));
        assert_eq!(owed, 0, "free room, spare and bank cache cover the units counted from them");
        (self.insert(Holding::Committed(kind), bytes), units)
    }
}
