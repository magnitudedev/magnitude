//! The holders of entries ([`Claim`], [`Cached`], [`ArenaClaim`]) and the
//! way a view takes them.

use std::num::NonZeroU64;
use std::sync::Arc;

use super::admission::{AdmitPolicy, Over, Reclaim, Short};
use super::changes::{Claiming, Keeping};
use super::charges::Holder;
use super::estimate::Estimate;
use super::state::EntryId;
use super::view::Ledger;
use super::vocabulary::{CacheKind, CommittedKind, Holding, Need, PoolClass, Value, size};

/// A committed claim: the right to allocate its bytes, held until dropped.
/// Dropping it frees its bytes once no allocation Seismic charged to it
/// remains ([`Self::entry`]); its holder drops it once no GPU fence reads
/// them. Not `Clone`: one holder.
#[must_use]
pub struct Claim {
    record: Holder,
    kind: CommittedKind,
}

/// The arena's claim, carved once at load from what the load recorded
/// allocating and living for the load: its graph scratch, the held pass's
/// buffers and the floor's group of one place, the bytes of the load's own
/// graph placement. A group's class beyond one place is never the arena's:
/// it is claimed as the group opens and freed as it closes
/// ([`CommittedKind::GroupClass`]).
#[must_use]
pub struct ArenaClaim {
    claims: Vec<Claim>,
}

impl ArenaClaim {
    /// The arena's claims, each charged the allocation it was made for.
    pub(super) fn of(claims: Vec<Claim>) -> Self {
        Self { claims }
    }

    pub fn bytes(&self) -> u64 {
        self.claims.iter().map(Claim::bytes).sum()
    }

    /// What the arena's allocations charged its claims.
    #[cfg(test)]
    pub(crate) fn allocated(&self) -> u64 {
        self.claims.iter().map(Claim::allocated_bytes).sum()
    }
}

/// A cache holding: the right to allocate its bytes, held until dropped.
/// Dropping it frees its bytes, as a claim's, and what it yielded. A claim
/// may take its bytes in process ([`Ledger::claim`]): its holder reads what
/// it still holds from [`Self::bytes`] and evicts the rest; Seismic frees
/// that memory through the entry, and the claim allocates it only then.
/// Its holder takes it up again only in a change's keep phase
/// ([`Keeping::keep`]), before that change claims anything.
#[must_use]
pub struct Cached {
    record: Holder,
    kind: CacheKind,
    #[cfg(test)]
    pub(super) _reclaimer: Option<Arc<dyn super::admission::CacheReclaimer>>,
}

impl Ledger {
    /// Hold `bytes` of committed memory of `kind`, an allocation about to
    /// be made, admitted at `held`, a reading of the store a load holds,
    /// or else at the latest (a domain's claim as it serves): the one way
    /// an allocation's claim is taken in a store
    /// ([`DeviceLedgers::claim`], [`DeviceLedgers::claim_at`]), in
    /// `claiming`, a change's claim phase. Cache yields to it in rank
    /// order; nothing is preempted. Refused, nothing is held.
    ///
    /// # Panics
    /// For a request's state, which is reserved by its own path
    /// ([`Self::reserve`]), or a grant ([`Self::grant`]).
    pub fn claim_at(&self, claiming: &Claiming, held: Option<&Estimate>, kind: CommittedKind, bytes: u64) -> Result<Claim, Short> {
        claiming.of_scope(&self.store.changes);
        assert!(!matches!(kind, CommittedKind::Request(_) | CommittedKind::Grant(_)), "a {kind:?} claim is taken by its own path");
        let id = self.admit_or_reclaim(AdmitPolicy {
            at: held.map(|held| held.read),
            kind,
            bytes,
            of: |_| true,
            within_boundary: false,
            allow_guarantee: true,
        })?;
        Ok(Claim::held(self.clone(), id, kind))
    }

    /// Committed memory for `need`, sized by the formulas, synchronously,
    /// in the one order the holding kinds give: free room, then cache in
    /// rank order (worth least per byte first, [`Value`]). Each cache
    /// holder reads what it still holds from its handle's `bytes`. Refused
    /// where committed memory with the need exceeds the budget.
    ///
    /// # Panics
    /// For a request's state or a grant, as [`Self::claim_at`].
    pub fn claim(&self, claiming: &Claiming, need: Need) -> Result<Claim, Short> {
        let bytes = size(&self.shape, &need);
        self.claim_at(claiming, None, need.kind(), bytes)
    }

    /// Committed memory of `kind` in whole `unit`s, at most `most` of them:
    /// as many as free room, then spare granules, then idle graph slots,
    /// then bank slots above the
    /// floor cover now, yielding that cache as [`Self::claim`] does. It
    /// never preempts a sequence and is never refused: it may hold none
    /// (0 bytes). Its holder reads the units it got from the second value.
    ///
    /// # Panics
    /// For a request's state, which is reserved by its own path.
    pub fn claim_units(&self, claiming: &Claiming, kind: CommittedKind, unit: NonZeroU64, most: u64) -> (Claim, u64) {
        claiming.of_scope(&self.store.changes);
        let (id, units) = self.admit_units(kind, unit, most);
        (Claim::held(self.clone(), id, kind), units)
    }

    /// Hold `bytes` of free room as cache of `kind`, none of it allocated:
    /// the allocation Seismic makes charging it allocates it, so it is
    /// counted once: the pool's entry, for pooled cache (bank slots,
    /// prefixes, spare granules), which holds spare granules in place
    /// first; its own entry ([`Cached::entry`]) otherwise. Refused where
    /// free room is short of them. Free room is the budget less everything
    /// held; the bank's target subtracts the guarantee before it asks.
    pub fn cache(&self, kind: CacheKind, bytes: u64, reclaim: Reclaim) -> Result<Cached, Short> {
        let mut state = self.state();
        let room = state.room_for(&self.store.identity, Holding::Cache(kind), bytes);
        if room < bytes {
            return Err(Over { bytes: bytes - room }.of(&self.store.identity, bytes));
        }
        let id = state.insert(Holding::Cache(kind), bytes);
        state.reclaim.insert(id, reclaim);
        drop(state);
        Ok(Cached {
            record: Holder::new(self.clone(), id),
            kind,
            #[cfg(test)]
            _reclaimer: None,
        })
    }

    /// Hold `bytes` as a [`CommittedKind::Backing`] claim in `class`'s pool:
    /// a bank's growth or a state store's, whose slots the pool backs as it
    /// goes, charging its blocks to the pool's entry, and cache once it ends
    /// ([`Claim::into_backed_cache`]). State slab backing (`PoolClass::State`)
    /// reclaims cache (including bank slots above floor) synchronously so
    /// physical memory is freed before the pool allocates; bank growth
    /// (`PoolClass::Records`) draws only on free room, its own spare granules,
    /// and worthless cache without cannibalizing bank cache or the guarantee.
    pub fn backing(&self, claiming: &Claiming, class: PoolClass, bytes: u64) -> Result<Claim, Short> {
        claiming.of_scope(&self.store.changes);
        let kind = CommittedKind::Backing(class);
        let id = match class {
            PoolClass::State => self.admit_or_reclaim(AdmitPolicy {
                at: None,
                kind,
                bytes,
                of: |_| true,
                within_boundary: false,
                allow_guarantee: true,
            })?,
            PoolClass::Records => self.admit_or_reclaim(AdmitPolicy {
                at: None,
                kind,
                bytes,
                of: |ck: CacheKind| !ck.valued(),
                within_boundary: false,
                allow_guarantee: false,
            })?,
        };
        Ok(Claim::held(self.clone(), id, kind))
    }
}

impl Claim {
    /// The claim on entry `id` of `ledger`, which it holds from now.
    pub(super) fn held(ledger: Ledger, id: EntryId, kind: CommittedKind) -> Self {
        Self { record: Holder::new(ledger, id), kind }
    }

    pub(super) fn ledger(&self) -> &Ledger {
        &self.record.ledger
    }

    pub(super) fn id(&self) -> EntryId {
        self.record.id
    }

    pub fn kind(&self) -> CommittedKind {
        self.kind
    }

    pub fn bytes(&self) -> u64 {
        self.ledger().state().entry(self.id()).bytes
    }

    /// What Seismic allocated of it through its entry ([`Self::entry`]);
    /// none for a pooled claim, whose granules the pool's entry allocates.
    pub fn allocated_bytes(&self) -> u64 {
        self.ledger().state().entry(self.id()).allocated
    }

    /// Release what its allocations did not take, once they are made: a
    /// claim taken at a priced figure (the plan's arena, a group's class)
    /// then holds exactly what Seismic charged its entry. The figure differs
    /// from the charges by the backend's own accounting (CPU buffers carved
    /// from whole host page blocks; CUDA's graphs charged to the device's
    /// own entry, `GraphPool`), and what they took beyond it was admitted
    /// as they took it ([`seismic::memory::Entry::take`]).
    ///
    /// # Panics
    /// For a pooled claim: the pool's entry allocates its granules.
    pub fn fit_allocated(&mut self) {
        let mut state = self.ledger().state();
        let entry = *state.entry(self.id());
        assert!(!entry.holding.pooled(), "invariant violated: a {:?} claim is fitted; the pool's entry allocates its granules", entry.holding);
        if entry.bytes > entry.allocated {
            state.resize(self.id(), entry.allocated);
        }
    }

    /// Its entry, which Seismic charges what it allocates for the claim
    /// (arena, weights, driver objects): shared with Seismic, the claim
    /// itself staying the one holder.
    ///
    /// # Panics
    /// For a pooled claim: the pool's entry allocates its granules
    /// ([`Ledger::pool_block`]).
    pub fn entry(&self) -> Arc<dyn seismic::memory::Entry> {
        assert!(
            !Holding::Committed(self.kind).pooled(),
            "invariant violated: {:?} is charged; the pool's entry allocates its granules",
            self.kind
        );
        self.record.entry()
    }

    /// The backing ended: the claim's bytes, the granules the pool backed
    /// for it, are cache of `kind` with reclamation policy `reclaim`
    /// ([`Self::into_cache`]).
    pub fn into_backed_cache(self, kind: CacheKind, reclaim: Reclaim) -> Cached {
        assert!(matches!(self.kind, CommittedKind::Backing(_)), "invariant violated: a {:?} claim ends as a backing", self.kind);
        self.into_cache(kind, reclaim)
    }

    /// Split off `bytes` of it, none allocated, as a records
    /// [`CommittedKind::Backing`] claim of their own: memory it covers that
    /// backs after the rest is made (a load's host tier, after its import).
    ///
    /// # Panics
    /// Past what its holder has not allocated.
    pub(super) fn split_backing(&mut self, bytes: u64) -> Claim {
        let mut state = self.ledger().state();
        let entry = *state.entry(self.id());
        let unallocated = entry.bytes - entry.allocated;
        assert!(bytes <= unallocated, "invariant violated: a claim splits off {bytes} bytes of its {unallocated} unallocated");
        state.resize(self.id(), entry.bytes - bytes);
        let kind = CommittedKind::Backing(PoolClass::Records);
        let id = state.insert(Holding::Committed(kind), bytes);
        drop(state);
        Claim::held(self.ledger().clone(), id, kind)
    }

    /// Turn the claim into cache of `kind` in place, in its own entry, its
    /// bytes, their allocation and its last value kept throughout (a closed
    /// group's class, an active drafter going idle): claims in process take
    /// it as any cache. Its entry stays the one Seismic charges.
    ///
    /// # Panics
    /// Where it is not allocated whole: cache is memory the process holds.
    /// A pooled claim's granules are allocated by the pool.
    pub fn into_cache(self, kind: CacheKind, reclaim: Reclaim) -> Cached {
        let Claim { record, .. } = self;
        let mut state = record.ledger.state();
        let entry = *state.entry(record.id);
        assert!(
            entry.holding.pooled() || entry.allocated == entry.bytes,
            "invariant violated: a {:?} claim turns cache only once allocated whole: {} of its {} bytes allocated",
            entry.holding,
            entry.allocated,
            entry.bytes
        );
        if matches!(reclaim, Reclaim::Unbacked) {
            assert_eq!(entry.allocated, 0, "invariant violated: Reclaim::Unbacked cache holds {} allocated bytes", entry.allocated);
        }
        state.turn(record.id, Holding::Cache(kind));
        state.reclaim.insert(record.id, reclaim);
        drop(state);
        Cached {
            record,
            kind,
            #[cfg(test)]
            _reclaimer: None,
        }
    }
}

impl Keeping<'_> {
    /// Take `holding`, cache this change takes up again (an idle component
    /// made active again, a closed group's idle slots for a class that
    /// opens again), committed of `kind` in place, whole: each entry turned
    /// as [`Claim::into_cache`] turned it, its bytes, their allocation and
    /// its value kept. The one way cache is committed again: in the keep
    /// phase, so no claim of this change can have taken any of it. Returned
    /// as cache, every entry as it was, where a claim of an earlier change
    /// took some of it in process: it is no longer whole.
    ///
    /// # Panics
    /// Where the change claims already ([`Keeping::may_keep`]).
    pub fn keep(&self, holding: Vec<Cached>, kind: CommittedKind) -> Result<Vec<Claim>, Vec<Cached>> {
        let mut kept: Vec<(Claim, CacheKind, Reclaim)> = Vec::with_capacity(holding.len());
        let mut holding = holding.into_iter();
        while let Some(cached) = holding.next() {
            self.may_keep(&cached.ledger().store.changes);
            let was = cached.kind;
            match cached.commit(kind) {
                Ok((claim, reclaim)) => kept.push((claim, was, reclaim)),
                Err(taken) => {
                    let cached = kept.into_iter().map(|(claim, was, reclaim)| claim.into_cache(was, reclaim)).chain([taken]).chain(holding).collect();
                    return Err(cached);
                }
            }
        }
        Ok(kept.into_iter().map(|(claim, _, _)| claim).collect())
    }
}

impl Cached {
    fn ledger(&self) -> &Ledger {
        &self.record.ledger
    }

    pub(super) fn id(&self) -> EntryId {
        self.record.id
    }

    pub fn kind(&self) -> CacheKind {
        self.kind
    }

    /// Its entry, which Seismic charges what it allocates for the holding,
    /// as [`Claim::entry`].
    ///
    /// # Panics
    /// For pooled cache, as [`Claim::entry`].
    pub fn entry(&self) -> Arc<dyn seismic::memory::Entry> {
        assert!(
            !Holding::Cache(self.kind).pooled(),
            "invariant violated: {:?} cache is charged; the pool's entry allocates its granules",
            self.kind
        );
        self.record.entry()
    }

    /// The holding committed of `kind` in place ([`Keeping::keep`]), or
    /// returned as it is where claims took some of it in process.
    fn commit(self, kind: CommittedKind) -> Result<(Claim, Reclaim), Cached> {
        let mut state = self.ledger().state();
        let entry = *state.entry(self.id());
        if entry.releasing > 0 || (!entry.holding.pooled() && entry.allocated != entry.bytes) {
            drop(state);
            return Err(self);
        }
        state.turn(self.id(), Holding::Committed(kind));
        let reclaim = state.reclaim.remove(&self.id()).unwrap_or(Reclaim::Deferred);
        drop(state);
        Ok((Claim { record: self.record, kind }, reclaim))
    }

    /// What it is worth per byte now, as its holder measures it: it yields
    /// by it ([`Value`]).
    ///
    /// # Panics
    /// For spare granules and idle graph slots, which are worth nothing.
    pub fn value(&self, value: Value) {
        assert!(self.kind.valued(), "invariant violated: {:?} cache is valued", self.kind);
        let mut state = self.ledger().state();
        state.entry(self.id()).value = value;
        state.holdings += 1;
    }

    /// What it still holds: claims may have taken some in process.
    pub fn bytes(&self) -> u64 {
        self.ledger().state().entry(self.id()).bytes
    }

    /// What claims took of it in process that Seismic has not freed yet;
    /// none for pooled cache, whose granules the pool frees.
    pub fn releasing(&self) -> u64 {
        self.ledger().state().entry(self.id()).releasing
    }

    /// What claims of the process took of it in process since its holder
    /// last asked, read once: pooled cache's granules stay its holder's
    /// until it frees them, and those claims allocate the bytes only then,
    /// so its holder owes them. None of what a shortage took
    /// (`Ledger::execute`), which the holder frees by its own rule.
    pub fn take_claimed(&self) -> u64 {
        let mut state = self.ledger().state();
        std::mem::take(&mut state.entry(self.id()).claimed)
    }

    /// The store it is held in.
    pub fn store(&self) -> seismic::StoreIdentity {
        self.ledger().store()
    }

    /// Split off `bytes` of pooled cache as a pooled holding of its own, of
    /// `kind`, in place (a bank's cut slots, a prefix's slabs cut back):
    /// what is held moves, and nothing allocated or wired, which is the
    /// pool's entry's ([`Holding::Pool`]). Only pooled cache splits: a
    /// holding Seismic charges allocates its bytes in allocations the
    /// ledger does not divide, wired or not, so it is let go whole.
    ///
    /// # Panics
    /// Where the holding or `kind` is not pooled.
    pub fn split(&mut self, bytes: u64, kind: CacheKind) -> Cached {
        let mut state = self.ledger().state();
        let entry = *state.entry(self.id());
        assert!(
            entry.holding.pooled() && Holding::Cache(kind).pooled() && entry.holding.pool_class() == Holding::Cache(kind).pool_class(),
            "invariant violated: {:?} splits off {kind:?}; only pooled cache splits, within its pool",
            entry.holding
        );
        let rest = entry.bytes.checked_sub(bytes).expect("a split takes no more than the holding has");
        state.resize(self.id(), rest);
        let id = state.insert(Holding::Cache(kind), bytes);
        if let Some(reclaim) = state.reclaim.get(&self.id()).cloned() {
            state.reclaim.insert(id, reclaim);
        }
        drop(state);
        Cached {
            record: Holder::new(self.ledger().clone(), id),
            kind,
            #[cfg(test)]
            _reclaimer: None,
        }
    }
}
