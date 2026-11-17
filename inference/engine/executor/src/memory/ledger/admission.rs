//! Admission: how a store's state takes and yields. Nothing is preempted
//! here: a claim short of memory, a request's growth included, is relieved
//! only by a plan ([`super::relief`]).

use std::num::NonZeroU64;

use crate::operation::RequestId;

use super::estimate::Read;
use super::state::{EntryId, State};
use super::vocabulary::{CacheKind, CommittedKind, Holding, Value};

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
    /// [`Self::take`] in `store` at `at` ([`Self::reading`]), refused as
    /// [`Short`].
    pub(super) fn take_in(&mut self, store: &seismic::StoreIdentity, at: Option<Read>, kind: CommittedKind, bytes: u64) -> Result<EntryId, Short> {
        let at = self.reading(store, at);
        self.take(at, kind, bytes).map_err(|over| {
            self.waiting = true;
            over.of(store, bytes)
        })
    }

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
        let owed = self.give(bytes.saturating_sub(beside), |_| true, Some(Holding::Committed(kind))).saturating_sub(free - beside);
        Ok(self.insert(Holding::Committed(kind), bytes - owed))
    }

    /// Hold `bytes` of committed memory of `kind` for new work beside the
    /// load (a request's reservation, a grant): free room beside the
    /// guarantee, within the boundary B where it stands, then the cache
    /// worth nothing (spare granules, idle graph slots) yielding in rank
    /// order. The bank and the retained prefixes give it room only through
    /// a relief plan ([`super::relief`]): refused here past what these
    /// cover, nothing taken.
    pub(super) fn take_beside(&mut self, at: Read, kind: CommittedKind, bytes: u64) -> Result<EntryId, Over> {
        let worthless = |kind: CacheKind| !kind.valued();
        let estimate = self.estimate_at(at);
        let within = estimate.cap().map_or(u64::MAX, |cap| cap.saturating_sub(estimate.held_now()));
        let beside = self.free_for(at, Holding::Committed(kind), bytes).min(within);
        let cache = CacheKind::ALL.into_iter().filter(|kind| worthless(*kind)).map(|kind| self.cache_of[kind.index()]).sum::<u64>();
        if let Some(over) = bytes.checked_sub(beside.saturating_add(cache)).filter(|over| *over > 0) {
            return Err(Over { bytes: over });
        }
        let owed = self.give(bytes.saturating_sub(beside), worthless, Some(Holding::Committed(kind)));
        assert_eq!(owed, 0, "invariant violated: the cache worth nothing covers what it was counted to");
        Ok(self.insert(Holding::Committed(kind), bytes))
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

    /// Hold up to `most` whole `unit`s of committed memory of `kind`: as
    /// many as free room, then spare granules, idle graph slots and bank
    /// slots above the floor cover now, that cache yielding in rank order
    /// ([`Self::give`]). Never preempts and never refuses. The entry and its
    /// units.
    pub(super) fn take_units(&mut self, store: &seismic::StoreIdentity, kind: CommittedKind, unit: NonZeroU64, most: u64) -> (EntryId, u64) {
        assert!(!matches!(kind, CommittedKind::Request(_)), "a request's state is reserved by its own path, never in units");
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
