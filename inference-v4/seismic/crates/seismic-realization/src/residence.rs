//! Residence graphs, lifetimes, placement domains, and routed strategies
//! (package D1).
//!
//! The residence graph, not a backend walker, owns physical allocation
//! choices. Workgroup and participant residences are legal only for a
//! kernel-local lifetime; ABI residences only for root boundary leaves;
//! device-arena residences cover cross-step values and state. Multiple
//! legal placements become solver choices.

use crate::ids::{
    BlockId, CanonicalLeafId, CanonicalStorageId, CanonicalValueId, ChoiceVarId, KernelAxisId,
    KernelInputId, KernelLocalId, KernelOutputId, ObligationRef, OwnedNodeRef, ResidenceId, StepId,
};
use crate::routes::{RouteTable, TensorRoute, ValueRoute};
use crate::strategy::ClosedStrategyShape;
use seismic_lang::logical::boundary::BoundaryLeaf;
use seismic_lang::logical::IdVec;
use seismic_lang::sym::Sym;
use seismic_lang::types::{NonEmpty, TensorType};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StorageScope {
    /// Public root ABI buffer; root boundary leaves only.
    Abi,
    DeviceArena,
    Workgroup,
    Participant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Replication {
    Once,
    PerSubgroup,
    PerWorkgroup,
    PerParticipant,
}

/// What a residence holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidenceSource {
    RootInput(BoundaryLeaf),
    RootResult(BoundaryLeaf),
    LogicalStorage(CanonicalStorageId),
    ComputedSpill(CanonicalValueId),
    KernelLocal { block: BlockId, value: CanonicalValueId },
    /// The 4-byte device pull counter of one `DynamicPull` block: zeroed by
    /// the block's `PullCounterReset` step, claimed from by its launch. Not
    /// a semantic value: it has no leaf and no route; the block names it
    /// through `RoutedBlock::pull_counter`.
    PullCounter { block: BlockId },
}

/// The placement domain of one residence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidenceChoice {
    Fixed(StorageScope),
    /// The solver selects one scope; the decision is exported as `var`.
    SolverChoice { options: NonEmpty<StorageScope>, var: ChoiceVarId },
}

/// The exact lifetime of one residence in the strategy's structured schedule.
///
/// Step identities are the pre-order numbering of the shape's structured
/// schedule (`RoutedStrategy::steps`): a control step precedes every step of
/// its bodies, and a body's steps are a contiguous range. `Steps` is the
/// closed interval from the first to the last step that touches the
/// residence, widened to the end of any `Repeat` the interval enters from
/// outside (a value produced before a loop and consumed inside it must
/// survive every visit).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Lifetime {
    /// Live for the whole strategy (root boundary residences).
    Whole,
    /// Live over the closed interval of schedule steps, inclusive.
    Steps { first: StepId, last: StepId },
    /// Live only inside one kernel block.
    KernelLocal(BlockId),
}

/// One structured schedule step of the shape, anchored to the shape object
/// it executes (dense `StepId`, pre-order). `Lifetime::Steps` intervals are
/// over these ids.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StepAnchor {
    Launch(BlockId),
    Guard(ObligationRef),
    /// Zero the pull counter residence of `block`; always the step
    /// immediately before `Launch(block)`.
    PullCounterReset(BlockId),
    Call(OwnedNodeRef),
    If(OwnedNodeRef),
    Repeat(OwnedNodeRef),
}

/// One step with the span of steps it encloses: `[id, last]` is the step
/// itself plus every step of its bodies (`last == id` for a launch, guard,
/// or call).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduleStep {
    pub anchor: StepAnchor,
    pub last: StepId,
}

/// One representation plane of a residence.
///
/// `bytes` is the plane's byte count for one replica at capacity (runtime
/// extents at their checked capacity; static extents exact). A residence
/// with `Replication::PerParticipant` occupies `bytes` per participant of
/// its owning block, `PerWorkgroup` per workgroup, `PerSubgroup` per
/// subgroup; M1 multiplies by the block's participant expression.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanePlan {
    pub plane: StoragePlane,
    /// Bytes at capacity, as a size expression over tuning parameters.
    pub bytes: Sym,
    pub alignment: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StoragePlane {
    Dense,
    Representation { name: String, ordinal: u32 },
}

/// One residence.
///
/// `shape` is the tensor type the residence holds in its own coordinates:
/// the leaf type for root boundary residences, the logical storage's shape
/// for a storage chain (also when a root result leaf is a view of a local
/// storage and that storage's residence is the ABI result), and the value
/// type for spills and kernel locals. Every route naming the residence
/// carries its transform from these coordinates to the value's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Residence {
    pub source: ResidenceSource,
    pub shape: TensorType,
    pub planes: NonEmpty<PlanePlan>,
    pub lifetime: Lifetime,
    pub choice: ResidenceChoice,
    pub replication: Replication,
}

/// The residence graph of one strategy. Residence ids are dense
/// (`ResidenceId(0..len)`), and so are the solver choice variables of every
/// `ResidenceChoice::SolverChoice` (`ChoiceVarId(0..choice_vars)`), in
/// residence order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidenceGraph {
    residences: IdVec<ResidenceId, Residence>,
    choice_vars: u32,
}

impl ResidenceGraph {
    pub(crate) fn seal(residences: IdVec<ResidenceId, Residence>, choice_vars: u32) -> ResidenceGraph {
        ResidenceGraph {
            residences,
            choice_vars,
        }
    }

    pub fn residence(&self, id: ResidenceId) -> &Residence {
        &self.residences[id]
    }

    pub fn len(&self) -> usize {
        self.residences.len()
    }

    pub fn is_empty(&self) -> bool {
        self.residences.is_empty()
    }

    pub fn ids(&self) -> impl Iterator<Item = ResidenceId> + '_ {
        self.residences.ids()
    }

    pub fn iter(&self) -> impl Iterator<Item = (ResidenceId, &Residence)> + '_ {
        self.residences.entries()
    }

    /// The number of solver scope decisions this strategy exports; every
    /// `SolverChoice.var` is below it and each is used exactly once.
    pub fn choice_vars(&self) -> u32 {
        self.choice_vars
    }
}

impl seismic_lang::logical::IdIndex for ResidenceId {
    fn from_index(index: usize) -> Self {
        ResidenceId(index as u32)
    }
    fn index(self) -> usize {
        self.0 as usize
    }
}

impl seismic_lang::logical::IdIndex for StepId {
    fn from_index(index: usize) -> Self {
        StepId(index as u32)
    }
    fn index(self) -> usize {
        self.0 as usize
    }
}

/// One external input of a closed kernel interface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelInputDecl {
    pub id: KernelInputId,
    pub leaf: CanonicalLeafId,
    pub route: ValueRoute,
}

/// One external destination of a closed kernel interface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelOutputDecl {
    pub id: KernelOutputId,
    pub leaf: CanonicalLeafId,
    pub route: ValueRoute,
}

/// One kernel-local addressable residence of a closed kernel interface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelLocalDecl {
    pub id: KernelLocalId,
    pub value: CanonicalValueId,
    pub route: TensorRoute,
}

/// One iteration axis of a closed kernel interface. Axis ids are the
/// ordinals of the block's `independent_axes`; every axis D1 declares binds
/// that axis's loop binder (`binder` is always `Some`; the field keeps its
/// frozen shape).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelAxisDecl {
    pub id: KernelAxisId,
    pub binder: Option<CanonicalValueId>,
}

/// The complete typed interface of one kernel block, frozen before any
/// operation is lowered. Every reference K1 emits names one of these dense
/// declarations.
///
/// - `inputs`: every leaf of the block cut's `inputs`, plus every dynamic
///   slice endpoint leaf of a tensor input's composed transform that the
///   block does not itself produce; in canonical leaf order.
/// - `outputs`: every leaf of the cut's `outputs`, plus every such endpoint
///   leaf this block produces for another block; in canonical leaf order.
/// - `locals`: one declaration per block-internal tensor value addressable
///   through a kernel-local residence (each view of a kernel-local logical
///   storage produced by a node of the block, and each computed value the
///   proposal required a local residence for); residence order, then value.
/// - `axes`: one per independent axis in ordinal order.
/// - `iteration`: `LinearIterationMap::from_axes` over those axes.
///
/// Serial binders and ordered carries of the participant map are block-local
/// SSA (K1), never inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosedKernelInterface {
    pub inputs: IdVec<KernelInputId, KernelInputDecl>,
    pub outputs: IdVec<KernelOutputId, KernelOutputDecl>,
    pub locals: IdVec<KernelLocalId, KernelLocalDecl>,
    pub axes: IdVec<KernelAxisId, KernelAxisDecl>,
    pub iteration: crate::dispatch::LinearIterationMap,
}

impl seismic_lang::logical::IdIndex for KernelInputId {
    fn from_index(index: usize) -> Self {
        KernelInputId(index as u32)
    }
    fn index(self) -> usize {
        self.0 as usize
    }
}
impl seismic_lang::logical::IdIndex for KernelOutputId {
    fn from_index(index: usize) -> Self {
        KernelOutputId(index as u32)
    }
    fn index(self) -> usize {
        self.0 as usize
    }
}
impl seismic_lang::logical::IdIndex for KernelLocalId {
    fn from_index(index: usize) -> Self {
        KernelLocalId(index as u32)
    }
    fn index(self) -> usize {
        self.0 as usize
    }
}
impl seismic_lang::logical::IdIndex for KernelAxisId {
    fn from_index(index: usize) -> Self {
        KernelAxisId(index as u32)
    }
    fn index(self) -> usize {
        self.0 as usize
    }
}

/// The pull counter of one block: `Counter` exactly when the block's
/// participant policy is `DynamicPull` (the counter residence its launch
/// claims coordinates from and its `PullCounterReset` step zeroes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PullCounter {
    None,
    Counter(ResidenceId),
}

/// One block with its closed interface (K1 input).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutedBlock {
    pub id: BlockId,
    pub interface: ClosedKernelInterface,
    pub pull_counter: PullCounter,
}

/// A strategy with total routes and residences. No public constructor;
/// sealed by `DataflowFormer::form` only.
///
/// Dense per-strategy allocations: executor scalar slots are
/// `ExecutorScalarSlotId(0..scalar_slots)`, one per scalar leaf routed as
/// `ScalarRoute::ExecutorSlot`; residence ids and choice variables are dense
/// in `ResidenceGraph`; step ids are dense in `steps`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutedStrategy {
    shape: ClosedStrategyShape,
    routes: RouteTable,
    residences: ResidenceGraph,
    blocks: IdVec<BlockId, RoutedBlock>,
    steps: IdVec<StepId, ScheduleStep>,
    scalar_slots: u32,
}

impl RoutedStrategy {
    pub(crate) fn seal(
        shape: ClosedStrategyShape,
        routes: RouteTable,
        residences: ResidenceGraph,
        blocks: IdVec<BlockId, RoutedBlock>,
        steps: IdVec<StepId, ScheduleStep>,
        scalar_slots: u32,
    ) -> RoutedStrategy {
        RoutedStrategy {
            shape,
            routes,
            residences,
            blocks,
            steps,
            scalar_slots,
        }
    }

    pub fn shape(&self) -> &ClosedStrategyShape {
        &self.shape
    }
    pub fn routes(&self) -> &RouteTable {
        &self.routes
    }
    pub fn residences(&self) -> &ResidenceGraph {
        &self.residences
    }
    pub fn blocks(&self) -> &IdVec<BlockId, RoutedBlock> {
        &self.blocks
    }
    /// The pre-order step table `Lifetime::Steps` intervals refer to.
    pub fn steps(&self) -> &IdVec<StepId, ScheduleStep> {
        &self.steps
    }
    /// The number of executor scalar slots this strategy allocates.
    pub fn scalar_slots(&self) -> u32 {
        self.scalar_slots
    }
}

/// The sole constructor of `RoutedStrategy` (package D1).
pub struct DataflowFormer;

impl DataflowFormer {
    /// Total over a closed strategy shape and the occurrence facts it was
    /// formed from.
    pub fn form(
        facts: &crate::occurrence::OccurrenceFacts<'_>,
        profile: &crate::target::EffectiveTargetProfile,
        shape: ClosedStrategyShape,
    ) -> Result<RoutedStrategy, crate::failure::CompilerDefect> {
        crate::formation::dataflow::form(facts, profile, shape)
    }
}
