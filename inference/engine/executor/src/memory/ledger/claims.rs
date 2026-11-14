//! The holders of entries ([`Claim`], [`Cached`], [`ArenaClaim`],
//! [`Preemption`], [`InProcess`]) and the way a view takes them.

use std::num::NonZeroU64;
use std::sync::Arc;

use crate::operation::RequestId;

use super::admission::{Over, Short, Taken};
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
#[must_use]
pub struct Cached {
    record: Holder,
    kind: CacheKind,
}

/// Why a claim was not taken whole.
#[derive(Debug)]
pub enum Refused {
    /// Committed memory it may not preempt, with the need, exceeds the
    /// budget. Nothing was taken.
    Short(Short),
    /// Free room and cache cannot cover it: the preemption holds what they
    /// gave, and the bytes of the sequences it names are reserved to it.
    Preempt(Preemption),
}

/// A claim waiting on preemption: what free room and cache gave its need,
/// and the sequences, the lowest value first and just enough, whose
/// state is reserved to it. Their holders evict them (fenced) and drop
/// their claims; each one's bytes join the preemption as it drops, and no
/// other claim can take them. Then [`Ledger::complete`] (or
/// [`Claim::complete`] for a growth) holds the need, which cannot fail.
/// Dropped before that, it lets go of what it holds and the sequences
/// stand.
#[must_use]
pub struct Preemption {
    ledger: Ledger,
    entry: EntryId,
    kind: CommittedKind,
    bytes: u64,
    sequences: Vec<RequestId>,
}

impl Preemption {
    /// The sequences to evict, the lowest value first.
    pub fn sequences(&self) -> &[RequestId] {
        &self.sequences
    }

    /// The refusal it comes to where its holder cannot evict the sequences
    /// it names: its need, over by what their state would give.
    pub fn short(&self) -> Short {
        let held = self.ledger.state().held[&self.entry].bytes;
        Over { bytes: self.bytes - held }.of(&self.ledger.store.identity, self.bytes)
    }

    /// Its parts, without letting go of what it holds.
    fn into_parts(self) -> (Ledger, EntryId, CommittedKind, u64) {
        let this = std::mem::ManuallyDrop::new(self);
        // SAFETY: `this` is never dropped, so its ledger moves out once.
        let ledger = unsafe { std::ptr::read(&this.ledger) };
        // SAFETY: as the ledger; the sequences' allocation is let go here.
        drop(unsafe { std::ptr::read(&this.sequences) });
        (ledger, this.entry, this.kind, this.bytes)
    }
}

impl std::fmt::Debug for Preemption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Preemption").field("kind", &self.kind).field("bytes", &self.bytes).field("sequences", &self.sequences).finish()
    }
}

impl Drop for Preemption {
    /// Recorded as every release is ([`Ledger::unpoisoned`]).
    fn drop(&mut self) {
        if let Some(mut state) = self.ledger.unpoisoned() {
            state.abandoned(self.entry);
        }
    }
}

/// A load's in-process figure in its store (§3.2): the state growth of the
/// largest launch its arena admits, held from load to unload. Dropped, the
/// store's guarantee no longer counts it.
#[must_use]
pub struct InProcess {
    ledger: Ledger,
    bytes: u64,
}

impl Drop for InProcess {
    /// Recorded as every release is ([`Ledger::unpoisoned`]).
    fn drop(&mut self) {
        if let Some(mut state) = self.ledger.unpoisoned() {
            state.in_process -= self.bytes;
            state.holdings += 1;
        }
    }
}

impl Ledger {
    /// Hold `bytes`, a load's in-process figure (§3.2), in the store's
    /// guarantee until the returned figure drops. Set once at load, on the
    /// store its state lies in.
    pub fn in_process(&self, bytes: u64) -> InProcess {
        let mut state = self.state();
        state.in_process = state.in_process.checked_add(bytes).expect("a store's in-process bytes fit u64");
        state.holdings += 1;
        InProcess { ledger: self.clone(), bytes }
    }

    /// Hold `bytes` of committed memory of `kind`, an allocation about to
    /// be made, admitted at `held`, a reading of the store a load holds,
    /// or else at the latest (a domain's claim as it serves): the one way
    /// an allocation's claim is taken in a store
    /// ([`DeviceLedgers::claim`], [`DeviceLedgers::claim_at`]). Cache
    /// yields to it in rank order; nothing is preempted. Refused, nothing
    /// is held.
    ///
    /// # Panics
    /// For a sequence's state, which is claimed by its rows alone.
    pub fn claim_at(&self, held: Option<&Estimate>, kind: CommittedKind, bytes: u64) -> Result<Claim, Short> {
        assert!(!matches!(kind, CommittedKind::State(_)), "a sequence's state is claimed by its rows");
        let taken = self.state().take_in(&self.store.identity, held.map(|held| held.read), kind, bytes)?;
        match taken {
            Taken::Held(id) => Ok(Claim::held(self.clone(), id, kind)),
            Taken::Preempting { .. } => unreachable!("only a sequence's growth preempts"),
        }
    }

    /// The one way committed memory is taken: `need` sized by the
    /// formulas, synchronously, in the one order the holding kinds give:
    /// free room, then cache in rank order (worth least per byte first,
    /// [`Value`]). Each cache holder reads what it still holds from its
    /// handle's `bytes`. Where these cannot cover it, the state of
    /// sequences admitted after the claimant's covers the rest
    /// ([`Refused::Preempt`]); refused outright only where committed memory
    /// it may not preempt, with the need, exceeds the budget.
    pub fn claim(&self, need: Need) -> Result<Claim, Refused> {
        let bytes = size(&self.shape, &need);
        let kind = need.kind();
        let taken = self.state().take_in(&self.store.identity, None, kind, bytes).map_err(Refused::Short)?;
        match taken {
            Taken::Held(id) => Ok(Claim::held(self.clone(), id, kind)),
            Taken::Preempting { entry, sequences } => {
                Err(Refused::Preempt(Preemption { ledger: self.clone(), entry, kind, bytes, sequences }))
            }
        }
    }

    /// Committed memory of `kind` in whole `unit`s, at most `most` of them:
    /// as many as free room, then spare granules, then idle graph slots,
    /// then bank slots above the
    /// floor cover now, yielding that cache as [`Self::claim`] does. It
    /// never preempts a sequence and is never refused: it may hold none
    /// (0 bytes). Its holder reads the units it got from the second value.
    ///
    /// # Panics
    /// For a sequence's state, which is claimed by its rows alone.
    pub fn claim_units(&self, kind: CommittedKind, unit: NonZeroU64, most: u64) -> (Claim, u64) {
        let (id, units) = self.state().take_units(&self.store.identity, kind, unit, most);
        (Claim::held(self.clone(), id, kind), units)
    }

    /// The claim `preemption` waited for, every sequence it named evicted.
    /// A sequence not yet evicted, or a preemption for a growth, is a
    /// broken invariant.
    pub fn complete(&self, preemption: Preemption) -> Claim {
        let (ledger, entry, kind, bytes) = preemption.into_parts();
        assert!(Arc::ptr_eq(&ledger.store, &self.store), "a preemption completes in its own ledger");
        ledger.state().completed(entry, bytes);
        Claim::held(ledger, entry, kind)
    }

    /// Hold `bytes` of free room as cache of `kind`, none of it allocated:
    /// the allocation Seismic makes charging it allocates it, so it is
    /// counted once: the pool's entry, for pooled cache (bank slots,
    /// prefixes, spare granules), which holds spare granules in place
    /// first; its own entry ([`Cached::entry`]) otherwise. Refused where
    /// free room is short of them. Free room is the budget less everything
    /// held; the bank's target subtracts the guarantee before it asks.
    pub fn cache(&self, kind: CacheKind, bytes: u64) -> Result<Cached, Short> {
        let mut state = self.state();
        let room = state.room_for(&self.store.identity, Holding::Cache(kind), bytes);
        if room < bytes {
            return Err(Over { bytes: bytes - room }.of(&self.store.identity, bytes));
        }
        let id = state.insert(Holding::Cache(kind), bytes);
        drop(state);
        Ok(Cached { record: Holder::new(self.clone(), id), kind })
    }

    /// Hold `bytes` of free room as a [`CommittedKind::Backing`] claim in
    /// `class`'s pool: a bank's growth or a state store's, whose slots the
    /// pool backs as it goes, charging its blocks to the pool's entry, and
    /// cache once it ends ([`Claim::into_backed_cache`]). Refused as
    /// [`Self::cache`] refuses.
    pub fn backing(&self, class: PoolClass, bytes: u64) -> Result<Claim, Short> {
        let kind = CommittedKind::Backing(class);
        let mut state = self.state();
        let room = state.room_for(&self.store.identity, Holding::Committed(kind), bytes);
        if room < bytes {
            return Err(Over { bytes: bytes - room }.of(&self.store.identity, bytes));
        }
        let id = state.insert(Holding::Committed(kind), bytes);
        drop(state);
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

    /// Whether this claim's granules are allocated by a pool entry rather than
    /// charged directly through [`Self::entry`].
    pub(crate) fn pooled(&self) -> bool {
        Holding::Committed(self.kind).pooled()
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

    /// What the sequence's state is worth per byte, as its holder measures
    /// it (the seconds its replay costs per byte held): a shortage preempts
    /// the lowest first ([`Shortage`]), as does a growth among the
    /// sequences admitted after its own.
    ///
    /// # Panics
    /// For a claim that is not a sequence's state.
    pub fn value(&self, value: Value) {
        assert!(matches!(self.kind, CommittedKind::State(_)), "invariant violated: a {:?} claim is valued; only a sequence's state is", self.kind);
        let mut state = self.ledger().state();
        state.entry(self.id()).value = value;
        state.holdings += 1;
    }

    /// The backing ended: the claim's bytes, the granules the pool backed
    /// for it, are cache of `kind` ([`Self::into_cache`]).
    pub fn into_backed_cache(self, kind: CacheKind) -> Cached {
        assert!(matches!(self.kind, CommittedKind::Backing(_)), "invariant violated: a {:?} claim ends as a backing", self.kind);
        self.into_cache(kind)
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

    /// Grow by `need`, of this claim's kind (a sequence's rows), as
    /// [`Ledger::claim`] takes.
    pub fn grow(&mut self, need: Need) -> Result<(), Refused> {
        assert_eq!(need.kind(), self.kind, "a claim grows by a need of its own kind");
        let ledger = self.ledger();
        let bytes = size(&ledger.shape, &need);
        let mut state = ledger.state();
        match state.take_in(&ledger.store.identity, None, self.kind, bytes).map_err(Refused::Short)? {
            Taken::Held(entry) => {
                state.merge(entry, self.id());
                Ok(())
            }
            Taken::Preempting { entry, sequences } => {
                drop(state);
                Err(Refused::Preempt(Preemption { ledger: ledger.clone(), entry, kind: self.kind, bytes, sequences }))
            }
        }
    }

    /// Grow by the need `preemption` waited for, every sequence it named
    /// evicted ([`Ledger::complete`]).
    pub fn complete(&mut self, preemption: Preemption) {
        let (ledger, entry, kind, bytes) = preemption.into_parts();
        assert!(Arc::ptr_eq(&ledger.store, &self.ledger().store) && kind == self.kind, "a growth completes into its own claim");
        let mut state = ledger.state();
        state.completed(entry, bytes);
        state.merge(entry, self.id());
    }

    /// Turn the claim into cache of `kind` in place, in its own entry, its
    /// bytes and their allocation kept throughout (a closed group's class,
    /// an active drafter going idle, a sequence's slabs retained as a
    /// prefix): claims in process take it as any cache. Its entry stays the
    /// one Seismic charges.
    ///
    /// # Panics
    /// Where it is not allocated whole: cache is memory the process holds.
    /// A pooled claim's granules are allocated by the pool.
    pub fn into_cache(self, kind: CacheKind) -> Cached {
        let Claim { record, .. } = self;
        let mut state = record.ledger.state();
        assert!(!state.reserved.contains_key(&record.id), "a preempted sequence's state is dropped, its bytes the preemption's");
        let entry = *state.entry(record.id);
        assert!(
            entry.holding.pooled() || entry.allocated == entry.bytes,
            "invariant violated: a {:?} claim turns cache only once allocated whole: {} of its {} bytes allocated",
            entry.holding,
            entry.allocated,
            entry.bytes
        );
        state.turn(record.id, Holding::Cache(kind));
        drop(state);
        Cached { record, kind }
    }
}

impl Cached {
    fn ledger(&self) -> &Ledger {
        &self.record.ledger
    }

    fn id(&self) -> EntryId {
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

    /// Turn the holding committed of `kind` in place, in its own entry, its
    /// bytes and their allocation kept (an idle component made active
    /// again): a claim as [`Claim::into_cache`] turned it. Returned as it is
    /// where claims took some of it in process: it is no longer whole.
    pub fn into_claim(self, kind: CommittedKind) -> Result<Claim, Cached> {
        let mut state = self.ledger().state();
        let entry = *state.entry(self.id());
        if entry.releasing > 0 || (!entry.holding.pooled() && entry.allocated != entry.bytes) {
            drop(state);
            return Err(self);
        }
        state.turn(self.id(), Holding::Committed(kind));
        state.entry(self.id()).value = Value::NONE;
        drop(state);
        Ok(Claim { record: self.record, kind })
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
    /// (`Ledger::yield_cache`), which the holder frees by its own rule.
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
        drop(state);
        Cached { record: Holder::new(self.ledger().clone(), id), kind }
    }
}
