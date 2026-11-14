//! What an entry is: its kinds, value and holding, and the needs a claim is
//! sized from.

use std::num::NonZeroU64;

use crate::operation::RequestId;
use crate::planning::BankTier;

/// What committed memory is held for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CommittedKind {
    /// A sequence's state (KV), as it grows. The only committed kind a
    /// claim may take ([`Ledger::claim`]): an earlier sequence's growth
    /// preempts a later one's.
    State(RequestId),
    /// A streamed tier's landing rows.
    Landing,
    /// An optional component's memory (a draft head, a vision encoder):
    /// its weights, imported after the load started, and what binding it
    /// and its pools hold, for as long as it is resident.
    Component,
    /// A program set's qualification fixtures, held while its preparation
    /// runs.
    Qualification,
    /// An import's upload window, staged through host RAM where that is a
    /// store apart: held while the import runs.
    Upload,
    /// Retained launch outputs' slots, held while their pool keeps them.
    Outputs,
    /// The arena at load: workspace, pass buffers, graph slots, group
    /// classes.
    Arena,
    /// The expert bank's floor: one step's routed experts per layer.
    BankFloor,
    /// A measurement job's room, in another process.
    Measurement,
    /// A prompt group's scratch beyond the decode set the floor holds, from
    /// its open to its close: optional, taken only from what yields at once
    /// ([`Ledger::claim_units`]).
    Scratch,
    /// A prompt group's class beyond one place, from its open to its
    /// close: optional, as [`Self::Scratch`].
    GroupClass,
    /// What a load allocated as it started ([`Recorded`]), until the
    /// holdings that name it carve it out.
    Load,
    /// An allocation backed after it begins in its class's pool, allocated
    /// as its memory is backed and cache once it ends ([`Ledger::backing`]):
    /// a bank holding's growth or a host tier's backing to its load-time
    /// slots (records), a state store's slabs' growth (state).
    Backing(PoolClass),
    /// A granule pool's absent granule, of its class: zero, held by no slab,
    /// the one every free slot of the pool's stores names. Allocated as the
    /// pool opens, outside its blocks, and held for the pool's life
    /// ([`super::Ledger::pool_block`]): never a spare granule, as no claim
    /// can take it.
    Absent(PoolClass),
    /// One request's minimum state: what a load needs to serve beside its
    /// weights and its bank's floor. Held for the domain's life; each
    /// sequence's state is credited against it ([`State::raise`]), so a
    /// request's first rows draw from room the load already holds and no
    /// window opens in which it is lost.
    StateFloor,
    /// The state stores' address tables, which the pool's granules are
    /// bound through: allocated as the stores are made, outside the pool,
    /// and held for the domain's life.
    StateTables,
    /// What a device holds of its own in the store: staging, contexts,
    /// streams ([`seismic::memory::Store::own`]).
    Device,
}

/// What cache memory is held for. Cache fills all room no committed claim
/// holds, and yields to one, and to a real shortage
/// ([`Ledger::yield_cache`]), the entry worth least per byte first
/// ([`Value`]); of equal worth, in this order, then the oldest first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CacheKind {
    /// Free granules of blocks the granule pool holds.
    Spare,
    /// Idle graph slots beyond the floor's, a closed prompt group's class
    /// until its pools release them.
    GraphSlots,
    /// Expert bank slots above the floor.
    Bank,
    /// Retained prefixes.
    Prefix,
    /// An idle drafter.
    Drafter,
    /// An idle vision encoder.
    Vision,
}

/// What a cache entry is worth per byte: the seconds of serving it saves
/// for each byte held. A prefix, the drafter or vision: the recompute
/// seconds the entry saves, times its uses since it was retained; the bank:
/// the read seconds its miss curve says its slots saved. No time decay.
/// Spare granules and idle graph slots are worth nothing. Finite and never
/// negative by construction, so it orders totally.
#[derive(Clone, Copy, Debug, Default)]
pub struct Value(f64);

impl Value {
    /// Nothing is lost when it goes.
    pub const NONE: Self = Self(0.0);

    /// `seconds` saved by holding `bytes`.
    ///
    /// # Panics
    /// Where `seconds` is negative or not finite.
    pub fn per_byte(seconds: f64, bytes: NonZeroU64) -> Self {
        assert!(seconds.is_finite() && seconds >= 0.0, "invariant violated: a cache holding saves {seconds} seconds");
        Self(seconds / bytes.get() as f64)
    }

    /// An entry of `bytes` whose recompute takes `seconds`, used `uses`
    /// times since it was retained.
    pub fn recompute(seconds: f64, uses: u64, bytes: NonZeroU64) -> Self {
        Self::per_byte(seconds * uses as f64, bytes)
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for Value {}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// An entry's state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Holding {
    Committed(CommittedKind),
    Cache(CacheKind),
    /// One of the store's granule pools (§4.7), of its class: every block
    /// the backend allocates for it is charged here, once, and only here.
    /// Its `bytes` are the granules no holding of its class holds, which
    /// are spare cache ([`CacheKind::Spare`]); bank slots and KV slabs are
    /// holdings over the granules of their class's pool
    /// ([`Self::pool_class`]), allocated by it and never by their own
    /// entries.
    Pool(PoolClass),
}

/// The class of a store's granule pool, each pooled at its own granule
/// (§4.7): sequences' state slabs at the state's page slab, and the expert
/// bank's records at its record slab. A store holds at most one pool of
/// each class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PoolClass {
    State,
    Records,
}

impl PoolClass {
    pub(super) fn index(self) -> usize {
        match self {
            Self::State => 0,
            Self::Records => 1,
        }
    }
}

impl Holding {
    /// The pool whose granules the holding lies in: KV slabs (a sequence's
    /// state, the state floor, retained prefixes, spare state slabs) in the
    /// state's, bank slots (the floor, landing rows, a growth's backing,
    /// slots above the floor) in the records'; a pool's own entry in
    /// itself. The pool's entry allocates them; moving one between
    /// holdings of one class moves only what is held.
    pub(super) fn pool_class(self) -> Option<PoolClass> {
        match self {
            Self::Committed(CommittedKind::State(_) | CommittedKind::StateFloor) => Some(PoolClass::State),
            Self::Committed(CommittedKind::BankFloor | CommittedKind::Landing) => Some(PoolClass::Records),
            Self::Committed(CommittedKind::Backing(class)) => Some(class),
            Self::Committed(_) => None,
            Self::Cache(CacheKind::Spare | CacheKind::Prefix) => Some(PoolClass::State),
            Self::Cache(CacheKind::Bank) => Some(PoolClass::Records),
            Self::Cache(_) => None,
            Self::Pool(class) => Some(class),
        }
    }

    /// Whether the holding lies in granules of one of its store's pools,
    /// other than a pool's own entry ([`Self::pool_class`]).
    pub(super) fn pooled(self) -> bool {
        !matches!(self, Self::Pool(_)) && self.pool_class().is_some()
    }
}

/// One holding of the process in one store: `bytes` held, `allocated` of
/// them allocated by Seismic through its entry ([`seismic::memory::Entry`]).
/// `allocated ≤ bytes` always, but for the pool's entry, which allocates
/// the granules every pooled holding holds; a pooled holding allocates
/// none of its own ([`Holding::Pool`] of its class). `releasing` is memory a cache
/// holding yielded that Seismic has not freed yet: allocated still, held
/// no more (for pooled cache, the pool's). A cache entry's `value` ranks
/// it ([`Value`]); a sequence's state is valued by its holder
/// ([`Claim::value`]); any other committed entry's is none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    pub holding: Holding,
    pub bytes: u64,
    pub allocated: u64,
    /// Of `allocated` and `releasing`, what the OS wires while the process
    /// holds it (`seismic::memory::Wiring::Wired`).
    pub wired: u64,
    pub releasing: u64,
    /// Of what pooled cache yielded, what claims of the process took in
    /// process (`State::yields`): bytes they
    /// allocate only once the holder's granules free them, not yet read by
    /// the holder (`Cached::take_claimed`). A shortage's yield is none of it.
    pub claimed: u64,
    pub value: Value,
}

// ---------------------------------------------------------------------------
// Needs, and the formulas that size them.
// ---------------------------------------------------------------------------

/// What committed memory is needed for, with the inputs its formula sizes
/// it from, or its bytes where its caller holds them. The one input to
/// [`Ledger::claim`] and [`Ledger::admit`]: there are no kind-specific
/// claims. The arena is carved from what the load recorded allocating
/// ([`Recorded::claim_arena`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Need {
    /// The state `sequence`'s history takes as it grows from `from` rows
    /// to `to`: the pages `to` rows occupy beyond those `from` rows occupy
    /// ([`LoadShape::state_bytes`]), so its claim holds what its stores
    /// allocate for it.
    StateRows { sequence: RequestId, from: u64, to: u64 },
    /// A streamed tier's landing rows, at load, of `slot_bytes` each.
    LandingRows { tier: BankTier, rows: u64, slot_bytes: u64 },
    /// A bank holding's floor, at load: the bank plan's charge at it, which
    /// plan and assessment size the bank by too.
    BankFloor { bytes: u64 },
    /// A measurement job's room, as the job declares it.
    Measurement { bytes: u64 },
    /// One request's minimum state, at load: the state `rows` rows take, held
    /// for the domain's life ([`CommittedKind::StateFloor`]).
    StateFloor { rows: u64 },
}

impl Need {
    /// The kind a claim for this need holds.
    pub fn kind(&self) -> CommittedKind {
        match self {
            Self::StateRows { sequence, .. } => CommittedKind::State(*sequence),
            Self::LandingRows { .. } => CommittedKind::Landing,
            Self::BankFloor { .. } => CommittedKind::BankFloor,
            Self::Measurement { .. } => CommittedKind::Measurement,
            Self::StateFloor { .. } => CommittedKind::StateFloor,
        }
    }
}

/// The header-derived inputs a load's needs are sized from: each of its
/// state stores' shape. Never planner objects.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadShape {
    pub stores: Vec<StoreShape>,
}

/// One state store's history domains, and the context and largest advance
/// the store lays their pages out at
/// ([`magnitude_state::HistoryDomainLayout::geometry`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoreShape {
    pub domains: Vec<magnitude_state::HistoryDomainLayout>,
    pub context: usize,
    pub max_advance: usize,
}

impl StoreShape {
    /// The shape of the state store `state` plans. The one way a store's
    /// shape is built.
    pub fn of(state: &crate::StateStorePlan) -> Self {
        Self {
            domains: state.history.iter().map(|domain| domain.layout.clone()).collect(),
            context: state.context_rows,
            max_advance: state.max_advance,
        }
    }

    /// The state `sequences` histories of `depth` rows take in this store,
    /// in `unit` ([`magnitude_state::state_bytes`]).
    pub fn state_bytes(
        &self,
        unit: magnitude_state::StateUnit,
        sequences: u64,
        depth: u64,
    ) -> Result<u64, magnitude_state::Error> {
        magnitude_state::state_bytes(&self.domains, self.context, self.max_advance, unit, sequences, depth)
    }
}

impl LoadShape {
    /// The shape of the load `plan` places ([`Self::of_states`]). The one
    /// way a load's shape is built.
    pub fn of(plan: &crate::planning::ExecutionPlan) -> Self {
        Self::of_states(plan.resources().target_state(), plan.resources().head_state())
    }

    /// One request's minimum state in the load `plan` places: what its first
    /// step claims, its one row and the most speculative rows its drafter adds
    /// past it ([`crate::domain::claimed_through`]), so a load that holds it
    /// serves a request whatever the method.
    pub fn minimum_state(plan: &crate::planning::ExecutionPlan) -> u64 {
        Self::of(plan).state_bytes(crate::domain::claimed_through(0, 1, plan.policy().method()))
    }

    /// The history domains of every state store a load holds: the
    /// target's, and its drafter's own where it keeps one (an MTP or draft
    /// head of its own history). A sequence's rows advance both, so its
    /// state claim holds both; state no claim holds is memory the ledger
    /// cannot see.
    pub fn of_states(target: &crate::StateStorePlan, head: Option<&crate::StateStorePlan>) -> Self {
        Self { stores: std::iter::once(target).chain(head).map(StoreShape::of).collect() }
    }

    /// The state one history of `rows` rows occupies across the load's
    /// stores, by the page rule ([`magnitude_state::state_bytes`]): the
    /// target's, and its drafter's own where it keeps one.
    ///
    /// # Panics
    /// Where a store's history domains do not size it: a load's domains
    /// size every history it admits.
    pub fn state_bytes(&self, rows: u64) -> u64 {
        self.stores
            .iter()
            .try_fold(0u64, |total, store| {
                let bytes = store
                    .state_bytes(magnitude_state::StateUnit::Pages, 1, rows)
                    .expect("a load's history domains size every history it admits");
                total.checked_add(bytes)
            })
            .expect("a load's state bytes fit u64")
    }
}

/// The bytes `need` takes: the one place any need's bytes come from. State
/// rows by the pages they occupy ([`LoadShape::state_bytes`]); the rest as
/// their callers hold them.
pub(super) fn size(shape: &LoadShape, need: &Need) -> u64 {
    match need {
        Need::StateRows { from, to, .. } => {
            assert!(from <= to, "a sequence's state need grows its history, never shrinks it");
            shape.state_bytes(*to) - shape.state_bytes(*from)
        }
        Need::LandingRows { rows, slot_bytes, .. } => rows.checked_mul(*slot_bytes).expect("landing rows' bytes fit u64"),
        Need::Measurement { bytes } | Need::BankFloor { bytes } => *bytes,
        Need::StateFloor { rows } => shape.state_bytes(*rows),
    }
}

impl CacheKind {
    /// In declaration order, which [`Self::index`] counts.
    pub(super) const ALL: [Self; 6] = [Self::Spare, Self::GraphSlots, Self::Bank, Self::Prefix, Self::Drafter, Self::Vision];

    /// Whether its holder measures what it is worth ([`Value`]): spare
    /// granules and idle graph slots are worth nothing.
    pub(super) fn valued(self) -> bool {
        !matches!(self, Self::Spare | Self::GraphSlots)
    }

    pub(super) const fn index(self) -> usize {
        self as usize
    }
}
