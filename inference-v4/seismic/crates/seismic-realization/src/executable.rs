//! The plan family, one structured schedule authority, nested boundary
//! environments, physical storage/global placement, and the infallible
//! resolver.
//!
//! A physical alternative is constructed by consuming every node, region
//! result, call, state edge, safety obligation, input and output of one
//! logical task-graph alternative. The public consuming transitions are
//! `map_primitive`, `map_reduction`, `fuse`, `fuse_call`, `split`,
//! `schedule_if`, `schedule_loop`, `invoke`, `discharge`, `complete_result`,
//! and `finish_alternative` (which succeeds only when the pending sets are
//! empty). Raw schedule construction is not public.
//!
//! Declaration companions allocate identities or facts without consuming a
//! logical obligation: `declare_executor_scalar_slot`, `plan_parameter`,
//! `solver_participants`, `stage_storage`, `barrier`, `status_field`,
//! `note_numerical`.
//!
//! Phases and predecessor edges do not exist: the structured schedule tree
//! (`ScheduleTemplate`/`ResolvedSchedule`) is the sole order authority.

use crate::numerics::NumericalTransfer;
use magnitude_solver::FeasibleAssignment;
use seismic_lang::{
    abi::{ScalarLayout, ScalarParameter},
    intrinsics::{IntrinsicId, PrimitiveId},
    logical::{
        Access, BoundaryInputKind, BoundaryResultKind, CallNode, ChoiceId, FunctionInterface,
        GraphValue, GraphValueId, IdIndex, IdVec, JoinSlot, LogicalIdentity, LogicalNode,
        LogicalNodeKind, LogicalProgram, LogicalRange, LogicalStorageId, LogicalViewId, NodeId,
        PrimitiveOp, RegionParameter, RegionResult, RuntimeExtent, RuntimeScalarExpr,
        SafetyObligation, StateJoin, StateToken, StateTokenId, TaskGraph, ViewTransform,
    },
    precision::NumericalAssessment,
    repr,
    sir::{LoopKind, ParamOwnership},
    sym::Sym,
    types::{canonical_leaves, DType, Leaf},
    types::{ExtentExpr, NonEmpty, RuntimeExtentId, TensorType, ValuePath, ValueType},
};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    fmt::Debug,
    rc::Rc,
};

/// Symbolic size/cost expression over planning parameters.
pub type SizeExpr = Sym;
pub type CostExpr = Sym;
/// Builder/reason errors that describe malformed construction input, never
/// post-solve rejection.
pub type BuilderError = String;

// ---------------------------------------------------------------------------
// Ids
// ---------------------------------------------------------------------------

macro_rules! ids {
    ($($(#[$meta:meta])* $name:ident),+ $(,)?) => {$(
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub u32);
    )+};
}

macro_rules! resolved_ids {
    ($($(#[$meta:meta])* $name:ident),+ $(,)?) => {$(
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub u64);
    )+};
}

ids!(
    /// One conditional physical storage in the family's global storage table.
    PhysicalStorageTemplateId,
    /// One declared tuning/placement parameter.
    PlanParamId,
    /// One planned executor-scalar slot (device-produced control/status scalar).
    ExecutorScalarSlotId,
    /// One kernel-local SSA value inside a single launch.
    KernelValueTemplateId,
);

resolved_ids!(
    ResolvedStorageId,
    ResolvedKernelValueId,
    ResolvedExecutorScalarId,
    ResolvedLaunchId,
    /// One public buffer binding of the root ABI.
    BufferBindingId,
    /// One field of the compiler-owned result scalar block.
    ResultScalarFieldId,
    /// One field of the root status binding.
    StatusFieldId,
);

/// Position of one region inside its task graph. Node ids are region-local,
/// so a region path plus a node id names exactly one logical node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RegionStep {
    IfThen(NodeId),
    IfElse(NodeId),
    LoopBody(NodeId),
    /// The inlined root region of one cross-call fused callee; imported
    /// callee regions live under this prefix.
    FusedCall(NodeId),
}

pub type RegionPath = Vec<RegionStep>;

impl RegionStep {
    /// The node holding this region.
    pub fn node(self) -> NodeId {
        match self {
            RegionStep::IfThen(node)
            | RegionStep::IfElse(node)
            | RegionStep::LoopBody(node)
            | RegionStep::FusedCall(node) => node,
        }
    }
}

/// One logical node, region-qualified. The builder consumes exact ids.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeRef {
    pub region: RegionPath,
    pub node: NodeId,
}

/// Identity and id translation of one child graph imported by cross-call
/// fusion. Backend facts derived from the child must use the same translation
/// as the consuming builder.
#[derive(Clone, Debug)]
pub struct FusedImport {
    pub root: RegionPath,
    pub value_offset: u32,
    pub state_offset: u32,
    pub storage_offset: u32,
}

/// One safety obligation: the `index`-th obligation of one node.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObligationRef {
    pub node: NodeRef,
    pub index: usize,
}

macro_rules! id_index {
    ($($name:ident),+ $(,)?) => {$(
        impl IdIndex for $name {
            fn from_index(index: usize) -> Self {
                $name(index as u32)
            }
            fn index(self) -> usize {
                self.0 as usize
            }
        }
    )+};
}

id_index!(
    PhysicalStorageTemplateId,
    PlanParamId,
    KernelValueTemplateId,
    ExecutorScalarSlotId
);

macro_rules! id_index_u64 {
    ($($name:ident),+ $(,)?) => {$(
        impl IdIndex for $name {
            fn from_index(index: usize) -> Self {
                $name(index as u64)
            }
            fn index(self) -> usize {
                self.0 as usize
            }
        }
    )+};
}

id_index_u64!(ResolvedStorageId);

// ---------------------------------------------------------------------------
// Effective target profile
// ---------------------------------------------------------------------------

/// Hard resource limits plus the exact effective capability signatures of the
/// compilation target. Legality is decided against this profile during
/// alternative construction, never after selection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectiveTargetProfile {
    pub backend: String,
    pub capability_fingerprint: String,
    pub toolchain_fingerprint: String,
    pub effective_signatures: BTreeSet<IntrinsicId>,
    pub limits: TargetLimits,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetLimits {
    pub max_participants: i64,
    pub max_workgroups_axis: [i64; 3],
    pub max_workgroup_bytes: i64,
    pub max_explicit_private_bytes: i64,
    pub max_direct_bindings: i64,
    pub max_argument_table_bytes: i64,
    pub max_device_bytes: i64,
}

// ---------------------------------------------------------------------------
// Dialect legalization
// ---------------------------------------------------------------------------

pub mod sealed {
    pub trait Sealed {}
}

/// One exact physical primitive offered to the dialect for legalization. It
/// carries the registry operation and the canonical operand/result types; the
/// dialect returns sealed opcodes plus exact consequences, or `Inapplicable`
/// for an optional capability strategy. Universal portable `Inapplicable` is
/// a compiler bug detected by the family builder.
#[derive(Clone, Debug, PartialEq)]
pub struct PhysicalPrimitive {
    pub op: PrimitiveOp,
    pub inputs: Vec<ValueType>,
    pub results: Vec<ValueType>,
}

/// Legalization result: nonempty sealed opcodes, or inapplicability of an
/// optional optimized form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Legalized<Op> {
    Ops(NonEmpty<Op>),
    Inapplicable { reason: String },
}

impl<Op> Legalized<Op> {
    pub fn ops(&self) -> Option<&NonEmpty<Op>> {
        match self {
            Legalized::Ops(ops) => Some(ops),
            Legalized::Inapplicable { .. } => None,
        }
    }
}

/// Everything the compiler controls exactly, plus the honest contract for
/// what the opaque native compiler may assign.
#[derive(Clone, Debug, PartialEq)]
pub struct PhysicalConsequences {
    pub hard: HardResources,
    pub native_contract: NativeResourceContract,
    pub cost: CostEstimate,
    pub numerical: NumericalTransfer,
    /// The exact capability signature this opcode requires, if any.
    pub capability: Option<IntrinsicId>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HardResources {
    pub explicit_private_bytes: u64,
    pub explicit_workgroup_bytes: u64,
    pub explicit_device_bytes: u64,
    pub direct_bindings: u32,
    pub argument_table_bytes: u64,
    pub argument_table_population_ops: u32,
    pub required_subgroup_width: Option<u32>,
    pub geometry_upper_bound_participants: Option<u64>,
    pub static_code_units: u64,
}

/// The facts the native compiler may reflect after encoding, with their
/// conservative admissible domains. A reflected value outside its domain is
/// `CompilerBug`. The selected geometry expression is legal for every fact in
/// the domain (e.g. `min(preferred_width, native_max_width)` guarantees at
/// least one resident participant).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NativeResourceContract {
    pub max_resident_participants: (u64, u64),
    pub native_subgroup_width: Option<(u64, u64)>,
}

/// Relative cost units; ranking only, never legality.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CostEstimate(pub u64);

/// The sealed backend dialect contract.
pub trait ExecutableDialect: sealed::Sealed + Clone + Debug + PartialEq + Eq + 'static {
    /// Closed opcode enum; emitted exhaustively, never with a wildcard.
    type Op: Clone + Debug + PartialEq + Eq;
    type LayoutTemplate: Clone + Debug + PartialEq + Eq;
    type ResolvedLayout: Clone + Debug + PartialEq + Eq;

    fn legalize(p: &PhysicalPrimitive, t: &EffectiveTargetProfile) -> Legalized<Self::Op>;
    fn consequences(op: &Self::Op) -> PhysicalConsequences;
    /// Layout template for a public (root ABI) storage of this shape.
    fn public_layout(tensor: &TensorType) -> Self::LayoutTemplate;
    /// Layout template for an internal (arena/workgroup/participant) storage.
    fn internal_layout(tensor: &TensorType) -> Self::LayoutTemplate;
    /// Layout template for a staged (raw-byte) workgroup/participant
    /// allocation declared by `stage_storage`. The default encodes the
    /// allocation as a `u32`-element tensor with one symbolic extent; a
    /// dialect may override for a richer staged layout.
    fn staged_layout(bytes: &SizeExpr, _alignment: u64) -> Self::LayoutTemplate {
        Self::internal_layout(&TensorType::new(
            vec![ExtentExpr::Sym(bytes.clone())],
            seismic_lang::types::Elem::Dtype(DType::U32),
        ))
    }
    /// Mechanical layout substitution under the solved plan values. It makes
    /// no legality decision.
    fn resolve_layout(
        layout: &Self::LayoutTemplate,
        values: &PlanValues,
    ) -> Result<Self::ResolvedLayout, InvariantReport>;
    /// Substitute every template storage/slot identity embedded in an opcode.
    /// Dialects whose opcodes contain no physical identities use the default.
    fn resolve_op(
        op: &Self::Op,
        _identities: &mut dyn PhysicalIdentityResolver,
    ) -> Result<Self::Op, InvariantReport> {
        Ok(op.clone())
    }
}

/// The only authority for turning family-local physical identities into the
/// resolved identities retained by the executable plan.
pub trait PhysicalIdentityResolver {
    fn storage(
        &mut self,
        template: PhysicalStorageTemplateId,
    ) -> Result<ResolvedStorageId, InvariantReport>;
    fn slot(
        &mut self,
        template: ExecutorScalarSlotId,
    ) -> Result<ResolvedExecutorScalarId, InvariantReport>;
    fn result_scalar(
        &mut self,
        path: &ValuePath,
        endpoint: Option<seismic_lang::abi::RangeEndpoint>,
        dtype: DType,
    ) -> Result<ResultScalarFieldId, InvariantReport>;
}

/// Solved planning values used for mechanical substitution.
#[derive(Clone, Debug, Default)]
pub struct PlanValues {
    pub symbols: BTreeMap<String, i64>,
}

impl PlanValues {
    pub fn eval(&self, expr: &Sym) -> Result<u64, InvariantReport> {
        let value = expr
            .eval(&|name| self.symbols.get(name).copied())
            .ok_or_else(|| InvariantReport(format!("plan expression `{expr}` is unresolved")))?;
        u64::try_from(value)
            .map_err(|_| InvariantReport(format!("plan expression `{expr}` is negative")))
    }
}

/// A broken compiler invariant. Ordinary resolution is infallible; this is
/// reported as `CompileFailure::CompilerBug` and never retried.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvariantReport(pub String);

impl From<&str> for InvariantReport {
    fn from(value: &str) -> Self {
        InvariantReport(value.to_string())
    }
}

impl std::fmt::Display for InvariantReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ---------------------------------------------------------------------------
// Boundary environments and transports
// ---------------------------------------------------------------------------

/// One canonical boundary leaf, qualified by its interface parameter so
/// that same-shaped parameters never collide: boundary leaf paths are
/// per-parameter paths, and several parameters can share `ValuePath([])`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BoundaryLeaf {
    Input { param: u32, leaf: ValuePath },
    Result { leaf: ValuePath },
}

impl Default for BoundaryLeaf {
    fn default() -> Self {
        BoundaryLeaf::Result {
            leaf: ValuePath::default(),
        }
    }
}

/// Conditional transport of one value across launch, call, and control
/// boundaries. `Boundary(leaf)` is the child-side placeholder: it names one
/// canonical boundary leaf and resolves directly to the caller's transport;
/// it never creates a child allocation.
#[derive(Clone, Debug, PartialEq)]
pub enum TransportTemplate {
    Void,
    Kernel(KernelValueTemplateId),
    ExecutorScalar(ExecutorScalarTemplate),
    /// Ordered representation planes of one tensor leaf.
    Storage(NonEmpty<StorageViewTemplate>),
    Tuple(NonEmpty<TransportTemplate>),
    /// Boundary placeholder (non-root alternatives only).
    Boundary(BoundaryLeaf),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutorScalarTemplate {
    pub source: ExecutorScalarSource,
    pub dtype: DType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutorScalarSource {
    /// A scalar already present in the root scalar ABI (or, for a result
    /// leaf, the compiler-owned result scalar block); used directly.
    Abi {
        leaf: BoundaryLeaf,
        endpoint: Option<seismic_lang::abi::RangeEndpoint>,
    },
    /// A planned executor-scalar slot, written by a preceding planned launch.
    Slot(ExecutorScalarSlotId),
    /// A computed executor scalar: checked symbolic arithmetic over ABI
    /// scalars, planned slots, runtime-extent values, solved plan
    /// parameters, and constants — e.g. `ceil(extent / window)` for
    /// runtime-axis streaming repeat ranges.
    Computed(ExecutorComputedScalar),
}

/// One computed control-scalar expression. Planning parameters evaluate to
/// their solved values; runtime extents retain their runtime values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutorComputedScalar {
    Abi {
        path: ValuePath,
        dtype: DType,
    },
    Slot(ExecutorScalarSlotId),
    Extent(RuntimeExtentId),
    /// A solved plan parameter, by name.
    Param(String),
    Const(i64),
    CeilDiv(Box<ExecutorComputedScalar>, Box<ExecutorComputedScalar>),
    Add(Box<ExecutorComputedScalar>, Box<ExecutorComputedScalar>),
    Sub(Box<ExecutorComputedScalar>, Box<ExecutorComputedScalar>),
    Mul(Box<ExecutorComputedScalar>, Box<ExecutorComputedScalar>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct StorageViewTemplate {
    pub storage: PhysicalStorageTemplateId,
    pub access: Access,
    /// Semantic view transform; backends address in storage coordinates.
    pub transform: ViewTransform,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StateTransportTemplate {
    /// A storage of this alternative (or an ancestor's, reached through
    /// activation-qualified template ids).
    Storage(PhysicalStorageTemplateId),
    /// Boundary placeholder resolved to the caller's state transport.
    Boundary(BoundaryLeaf),
}

/// The resolved environment of one call: parent transports keyed by canonical
/// boundary path. Children are instantiated against this; boundary storage
/// resolves directly to caller ids/views, and only child internals receive
/// fresh resolved ids.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BoundaryEnvironment {
    pub inputs: BTreeMap<BoundaryLeaf, ResolvedTransport>,
    pub results: BTreeMap<BoundaryLeaf, ResolvedTransport>,
    pub states: BTreeMap<BoundaryLeaf, ResolvedStateTransport>,
}

/// Caller-supplied transports for one `invoke`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BoundaryTemplates {
    pub inputs: BTreeMap<BoundaryLeaf, TransportTemplate>,
    pub results: BTreeMap<BoundaryLeaf, TransportTemplate>,
    pub states: BTreeMap<BoundaryLeaf, StateTransportTemplate>,
}

// ---------------------------------------------------------------------------
// Physical storage and global placement
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StorageScope {
    /// Public root ABI buffer. The root environment alone has ABI allocations.
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

/// A storage is active iff every listed (choice, alternative) selection
/// holds. An empty literal is unconditionally active (root ABI).
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ActivationLiteral(pub BTreeSet<ActivationTerm>);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ActivationTerm {
    pub choice: ChoiceId,
    pub logical_alternative: u32,
    pub physical_alternative: u32,
}

impl ActivationLiteral {
    pub fn always() -> Self {
        ActivationLiteral(BTreeSet::new())
    }

    /// Whether this literal holds under the given complete selection.
    pub fn holds(&self, selections: &BTreeMap<ChoiceId, (u32, u32)>) -> bool {
        self.0.iter().all(|term| {
            selections
                .get(&term.choice)
                .is_some_and(|(logical, physical)| {
                    *logical == term.logical_alternative && *physical == term.physical_alternative
                })
        })
    }
}

/// Lifetime of one conditional storage as step spans within the owning
/// alternative's schedule order. Nested child internals span the calling
/// step's interval; sequential siblings may reuse space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StorageLifetime {
    /// Live for the whole alternative (values live across calls).
    Always,
    Spans(Vec<(u32, u32)>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConditionalStorageTemplate<D: ExecutableDialect> {
    pub active_if: ActivationLiteral,
    pub scope: StorageScope,
    pub bytes: SizeExpr,
    pub alignment: u64,
    pub replication: Replication,
    pub layout: D::LayoutTemplate,
    pub lifetime: StorageLifetime,
}

/// One planned executor-scalar slot: bytes and alignment are hard resources.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutorScalarSlotDecl {
    pub dtype: DType,
    pub name: String,
}

impl ExecutorScalarSlotDecl {
    pub fn bytes(&self) -> u64 {
        u64::from(self.dtype.bytes())
    }
}

// ---------------------------------------------------------------------------
// One structured schedule authority
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct ScheduleTemplate<D: ExecutableDialect> {
    pub steps: NonEmpty<ScheduleStepTemplate<D>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ScheduleStepTemplate<D: ExecutableDialect> {
    Launch(LaunchTemplate<D>),
    Call(CallTemplate),
    If(ScheduleIfTemplate<D>),
    Repeat(ScheduleRepeatTemplate<D>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScheduleIfTemplate<D: ExecutableDialect> {
    pub condition: ExecutorPredicateTemplate,
    pub then_schedule: ScheduleTemplate<D>,
    pub else_schedule: ScheduleTemplate<D>,
    pub joins: Vec<PhysicalJoinTemplate>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScheduleRepeatTemplate<D: ExecutableDialect> {
    pub logical_kind: LoopKind,
    pub range: ExecutorRangeTemplate,
    /// The scalar binder, rebound to each ascending coordinate.
    pub binder: ExecutorScalarSlot,
    pub body: ScheduleTemplate<D>,
    pub carried: Vec<PhysicalCarryTemplate>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExecutorPredicateTemplate {
    /// The retained bool predicate: a scalar already present in the root
    /// scalar ABI (used directly), a value materialized by a preceding
    /// planned launch into a planned slot, or a boundary scalar.
    pub value: TransportTemplate,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExecutorRangeTemplate {
    pub start: TransportTemplate,
    pub end: TransportTemplate,
    pub bound: ExtentExpr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutorScalarSlot {
    pub slot: ExecutorScalarSlotId,
    pub dtype: DType,
}

/// A carried transport, rebound by the body each visit. It is storage or an
/// executor-scalar slot: something that survives across visits.
#[derive(Clone, Debug, PartialEq)]
pub struct PhysicalCarryTemplate {
    pub transport: TransportTemplate,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PhysicalJoinTemplate {
    Value {
        then: TransportTemplate,
        else_branch: TransportTemplate,
        joined: TransportTemplate,
    },
    State {
        storage: PhysicalStorageTemplateId,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct LaunchTemplate<D: ExecutableDialect> {
    pub geometry: DispatchTemplate,
    /// Retained zero-work launch condition; the runtime skips the launch when
    /// it evaluates to zero. Zero native grids are never submitted.
    pub condition: LaunchCondition,
    pub bindings: Vec<BindingGroupTemplate>,
    pub kernel: KernelTemplate<D>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DispatchTemplate {
    pub workgroups: [SizeExpr; 3],
    pub participants_per_workgroup: [SizeExpr; 3],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchCondition {
    pub work_items: SizeExpr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BindingGroupTemplate {
    pub kind: BindingGroupKind,
    pub slot: u32,
    pub members: NonEmpty<BindingMember>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingGroupKind {
    Direct,
    /// Planned descriptor/argument table: its bytes, alignment, population
    /// operation, binding slot and access mode are hard resources.
    ArgumentTable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindingMember {
    pub storage: PhysicalStorageTemplateId,
    pub access: AccessMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
    Atomic,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelTemplate<D: ExecutableDialect> {
    /// Workgroup-scope storage templates used by this launch.
    pub workgroup_storage: Vec<PhysicalStorageTemplateId>,
    /// Participant-scope storage templates used by this launch.
    pub participant_storage: Vec<PhysicalStorageTemplateId>,
    pub steps: NonEmpty<KernelStepTemplate<D>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum KernelStepTemplate<D: ExecutableDialect> {
    Mapped {
        iteration: crate::dispatch::LinearIterationMap,
        /// Operand transports of the mapped logical values.
        bindings: Vec<(GraphValueId, TransportTemplate)>,
        ops: NonEmpty<D::Op>,
    },
    Barrier {
        scope: BarrierScope,
    },
    /// Publish a tensor plane into retained device storage.
    Publish {
        storage: PhysicalStorageTemplateId,
    },
    /// Publish a device-produced scalar into a planned executor-scalar slot.
    PublishScalar {
        slot: ExecutorScalarSlotId,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BarrierScope {
    Subgroup,
    Workgroup,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CallTemplate {
    pub call: NodeRef,
    pub choice: ChoiceId,
    pub boundary: BoundaryTemplates,
}

// ---------------------------------------------------------------------------
// Strategy inputs of the builder transitions
// ---------------------------------------------------------------------------

/// Exact reduction strategy: topology, iteration, opcodes, and the one
/// publication of the reduced result.
#[derive(Clone, Debug, PartialEq)]
pub struct ReductionStrategyTemplate<D: ExecutableDialect> {
    pub topology: ReductionTopology,
    pub iteration: crate::dispatch::LinearIterationMap,
    pub ops: Legalized<D::Op>,
    pub result: TransportTemplate,
}

/// Exact reduction topology (hosted in `crate::numerics`; re-exported for
/// strategy construction).
pub use crate::numerics::ReductionTopology;

/// A fused strategy covers a connected node set in one launch.
#[derive(Clone, Debug, PartialEq)]
pub struct FusedStrategyTemplate<D: ExecutableDialect> {
    pub iteration: crate::dispatch::LinearIterationMap,
    pub ops: Legalized<D::Op>,
}

/// How one safety obligation is consumed.
#[derive(Clone, Debug, PartialEq)]
pub enum ObligationDisposition<D: ExecutableDialect> {
    StaticallyProved {
        reason: String,
    },
    RuntimeChecked {
        /// The planned predicate (already legalized opcodes).
        predicate: Legalized<D::Op>,
        inactive: InactiveBehavior,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InactiveBehavior {
    /// Skip the guarded operation.
    Skip,
    /// Produce the operation's neutral value.
    ProduceNeutral,
}

/// Receipt of one `discharge`: the status field a runtime check writes, or
/// `None` when statically proved.
pub type DispositionReceipt = Option<StatusFieldId>;

// ---------------------------------------------------------------------------
// Plan family
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct PhysicalAlternative<D: ExecutableDialect> {
    pub logical_alternative: u32,
    pub physical_alternative: u32,
    pub schedule: ScheduleTemplate<D>,
    pub obligations: ConsumedObligations<D>,
    pub cost: CostExpr,
    pub numerical: NumericalTransfer,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConsumedObligations<D: ExecutableDialect> {
    pub discharged: Vec<DischargedObligation<D>>,
    /// Strategy-declared status fields not tied to one obligation.
    pub declared_status_fields: Vec<StatusFieldId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DischargedObligation<D: ExecutableDialect> {
    pub node: NodeRef,
    pub index: usize,
    /// `Some(field)` for a runtime check writing the root status binding.
    pub runtime_checked: Option<StatusFieldId>,
    /// The planned predicate opcodes of a runtime check.
    pub predicate: Option<NonEmpty<D::Op>>,
}

#[derive(Clone, Debug)]
pub struct PhysicalChoice<D: ExecutableDialect> {
    /// The occurrence's typed call interface (ABI source of the root choice).
    pub interface: FunctionInterface,
    pub alternatives: NonEmpty<PhysicalAlternative<D>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanParameter {
    pub name: String,
    pub lower: i64,
    pub upper: i64,
}

#[derive(Clone, Debug)]
pub struct PlanFamily<D: ExecutableDialect> {
    pub logical: LogicalIdentity,
    pub target: seismic_lang::logical::EffectiveTargetIdentity,
    pub entry: ChoiceId,
    pub choices: IdVec<ChoiceId, PhysicalChoice<D>>,
    pub parameters: IdVec<PlanParamId, PlanParameter>,
    /// The global conditional storage table. Boundary storage resolves to
    /// caller ids; only child internals received fresh template ids.
    pub storages: IdVec<PhysicalStorageTemplateId, ConditionalStorageTemplate<D>>,
    pub executor_scalar_slots: IdVec<ExecutorScalarSlotId, ExecutorScalarSlotDecl>,
    /// Retained runtime extents: value expressions (never capacities).
    pub runtime_extents: IdVec<RuntimeExtentId, RuntimeExtent>,
    /// ABI storage templates by canonical root boundary leaf.
    pub abi_storage_paths: BTreeMap<BoundaryLeaf, PhysicalStorageTemplateId>,
    /// Solver-facing hard facts from the exact same objects the resolver
    /// consumes, plus lifetime interference pairs.
    pub model: PlanningModelExport,
}

/// Solver-facing hard facts for the one global planning model.
#[derive(Clone, Debug, Default)]
pub struct PlanningModelExport {
    pub alternatives: Vec<AlternativeFacts>,
    /// Device-arena storage activations (global table).
    pub storage: Vec<StorageActivation>,
    /// Device-arena template pairs whose lifetimes overlap under some
    /// alternative; each receives a solver ordering disjunction.
    pub interference: Vec<StorageInterference>,
}

#[derive(Clone, Debug)]
pub struct AlternativeFacts {
    pub choice: ChoiceId,
    pub logical_alternative: u32,
    pub physical_alternative: u32,
    pub launches: Vec<LaunchFacts>,
    pub cost: CostExpr,
    pub numerical: NumericalTransfer,
    pub required_capabilities: BTreeSet<IntrinsicId>,
}

#[derive(Clone, Debug)]
pub struct LaunchFacts {
    pub preferred_participants: SizeExpr,
    pub workgroups: [SizeExpr; 3],
    pub workgroup_bytes: SizeExpr,
    pub private_bytes_per_participant: SizeExpr,
    pub direct_bindings: u32,
    pub argument_table_bytes: u64,
    pub launch_condition_work_items: SizeExpr,
    /// Planned barriers inside the launch.
    pub barriers: u32,
    pub capabilities: BTreeSet<IntrinsicId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageActivation {
    pub storage: PhysicalStorageTemplateId,
    pub active_if: ActivationLiteral,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct StorageInterference {
    pub left: PhysicalStorageTemplateId,
    pub right: PhysicalStorageTemplateId,
}

// ---------------------------------------------------------------------------
// Resolved plan and ABI
// ---------------------------------------------------------------------------

/// Retained execution expression. It is not a planning symbol: it names
/// solved constants, runtime-extent values, ABI scalar fields, and planned
/// executor scalars.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionExpr {
    Const(u64),
    /// The retained runtime value of one runtime extent (never its capacity).
    Extent(RuntimeExtentId),
    AbiScalar {
        path: ValuePath,
        dtype: DType,
    },
    ExecutorScalar(ResolvedExecutorScalarId),
    Add(Box<ExecutionExpr>, Box<ExecutionExpr>),
    Sub(Box<ExecutionExpr>, Box<ExecutionExpr>),
    Mul(Box<ExecutionExpr>, Box<ExecutionExpr>),
    /// `ceil(left / right)` with `right > 0`.
    CeilDiv(Box<ExecutionExpr>, Box<ExecutionExpr>),
    /// `left / right` with a retained runtime denominator.
    Div(Box<ExecutionExpr>, Box<ExecutionExpr>),
    /// `left % right` with a retained runtime denominator.
    Rem(Box<ExecutionExpr>, Box<ExecutionExpr>),
    Min(Box<ExecutionExpr>, Box<ExecutionExpr>),
}

#[derive(Clone, Debug)]
pub struct ResolvedPlan<D: ExecutableDialect> {
    pub identity: ResolutionIdentity,
    /// The root ABI, created once from the entry interface. The internal
    /// arena and nested boundaries never appear in it.
    pub abi: ResolvedAbi<D>,
    pub internal_arena: ResolvedArena,
    /// The global resolved storage table (the root alone owns it).
    pub storage: IdVec<ResolvedStorageId, ResolvedStorage<D>>,
    pub entry: ResolvedPlanBody<D>,
    /// Retained runtime extent values (execution expressions, not capacities).
    pub runtime_extents: IdVec<RuntimeExtentId, ExecutionExpr>,
    pub resources: ResolvedProgramResources,
    pub numerical: NumericalAssessment,
    pub estimated_cost: u64,
    pub optimal: bool,
}

#[derive(Clone, Debug)]
pub struct ResolvedPlanBody<D: ExecutableDialect> {
    pub boundary: ResolvedBoundary,
    pub schedule: ResolvedSchedule<D>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResolvedBoundary {
    pub inputs: BTreeMap<BoundaryLeaf, ResolvedTransport>,
    pub results: BTreeMap<BoundaryLeaf, ResolvedTransport>,
    pub states: BTreeMap<BoundaryLeaf, ResolvedStateTransport>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedStateTransport {
    pub storage: ResolvedStorageId,
}

#[derive(Clone, Debug)]
pub struct ResolvedSchedule<D: ExecutableDialect> {
    pub steps: NonEmpty<ResolvedStep<D>>,
}

#[derive(Clone, Debug)]
pub enum ResolvedStep<D: ExecutableDialect> {
    Launch(ResolvedLaunch<D>),
    Call(ResolvedCall<D>),
    If(ResolvedScheduleIf<D>),
    Repeat(ResolvedScheduleRepeat<D>),
}

#[derive(Clone, Debug)]
pub struct ResolvedLaunch<D: ExecutableDialect> {
    pub id: ResolvedLaunchId,
    pub geometry: ResolvedDispatchGeometry,
    /// The retained zero-work launch condition.
    pub work_items: ExecutionExpr,
    pub bindings: Vec<ResolvedBindingGroup>,
    /// Kernel-local SSA identity: template id to resolved id, so emitters
    /// can name fused values.
    pub value_map: BTreeMap<KernelValueTemplateId, ResolvedKernelValueId>,
    pub kernel: ResolvedKernel<D>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedDispatchGeometry {
    pub workgroups: [ExecutionExpr; 3],
    pub participants_per_workgroup: [ExecutionExpr; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedBindingGroup {
    pub kind: BindingGroupKind,
    pub slot: u32,
    pub members: NonEmpty<ResolvedBindingMember>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedBindingMember {
    pub storage: ResolvedStorageId,
    pub access: AccessMode,
}

#[derive(Clone, Debug)]
pub struct ResolvedKernel<D: ExecutableDialect> {
    pub workgroup_storage: Vec<ResolvedStorageId>,
    pub participant_storage: Vec<ResolvedStorageId>,
    pub steps: NonEmpty<ResolvedKernelStep<D>>,
    pub resources: ResolvedKernelResources,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResolvedKernelResources {
    pub workgroup_bytes: u64,
    pub private_bytes_per_participant: u64,
    pub capabilities: BTreeSet<IntrinsicId>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResolvedKernelStep<D: ExecutableDialect> {
    Mapped {
        iteration: crate::dispatch::LinearIterationMap,
        bindings: Vec<(GraphValueId, ResolvedTransport)>,
        ops: NonEmpty<D::Op>,
    },
    Barrier {
        scope: BarrierScope,
    },
    Publish {
        storage: ResolvedStorageId,
    },
    PublishScalar {
        slot: ResolvedExecutorScalarId,
    },
}

#[derive(Clone, Debug)]
pub struct ResolvedCall<D: ExecutableDialect> {
    pub call: NodeRef,
    pub choice: ChoiceId,
    /// The instantiated child environment: transports are ids in the same
    /// global storage table.
    pub boundary: ResolvedBoundary,
    /// The nested child body. It owns no second ABI or arena.
    pub body: Box<ResolvedPlanBody<D>>,
}

#[derive(Clone, Debug)]
pub struct ResolvedScheduleIf<D: ExecutableDialect> {
    pub condition: ResolvedExecutorScalar,
    pub then_schedule: ResolvedSchedule<D>,
    pub else_schedule: ResolvedSchedule<D>,
    pub joins: Vec<ResolvedJoin>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResolvedJoin {
    Value {
        then: ResolvedTransport,
        else_branch: ResolvedTransport,
        joined: ResolvedTransport,
    },
    State {
        storage: ResolvedStorageId,
    },
}

#[derive(Clone, Debug)]
pub struct ResolvedScheduleRepeat<D: ExecutableDialect> {
    pub logical_kind: LoopKind,
    pub range: ResolvedExecutorRange,
    pub binder: ResolvedExecutorScalarId,
    pub body: ResolvedSchedule<D>,
    pub carried: Vec<ResolvedTransport>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedExecutorRange {
    pub start: ResolvedExecutorScalar,
    pub end: ResolvedExecutorScalar,
    /// The retained bound of the half-open ascending range.
    pub bound: ExecutionExpr,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResolvedTransport {
    Void,
    Kernel(ResolvedKernelValueId),
    ExecutorScalar(ResolvedExecutorScalar),
    Storage(NonEmpty<ResolvedStorageView>),
    Tuple(NonEmpty<ResolvedTransport>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedExecutorScalar {
    /// A scalar in the invocation ABI, with its final byte offset.
    Abi {
        path: ValuePath,
        offset: u64,
        dtype: DType,
    },
    /// A scalar in the compiler-owned result block.
    Result {
        path: ValuePath,
        field: ResultScalarFieldId,
        dtype: DType,
    },
    Slot {
        slot: ResolvedExecutorScalarId,
        dtype: DType,
    },
    /// A computed control scalar: a retained execution expression.
    Computed { expr: ExecutionExpr, dtype: DType },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedStorageView {
    pub storage: ResolvedStorageId,
    pub access: Access,
    /// Semantic view transform; backends address in storage coordinates.
    pub transform: ViewTransform,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedStorage<D: ExecutableDialect> {
    pub id: ResolvedStorageId,
    pub placement: ResolvedStoragePlacement,
    pub replication: Replication,
    pub bytes: u64,
    pub alignment: u64,
    pub layout: D::ResolvedLayout,
}

/// A resolved storage has exactly the location information required by its
/// memory space. Missing arena offsets and ABI bindings are unrepresentable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolvedStoragePlacement {
    Abi { binding: BufferBindingId },
    Arena { offset: u64 },
    Workgroup,
    Participant,
}

impl ResolvedStoragePlacement {
    pub fn scope(self) -> StorageScope {
        match self {
            Self::Abi { .. } => StorageScope::Abi,
            Self::Arena { .. } => StorageScope::DeviceArena,
            Self::Workgroup => StorageScope::Workgroup,
            Self::Participant => StorageScope::Participant,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedArena {
    pub bytes: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResolvedProgramResources {
    pub max_workgroup_bytes: u64,
    pub max_private_bytes_per_participant: u64,
    pub direct_bindings: u32,
    pub argument_table_bytes: u64,
    pub arena_bytes: u64,
    pub capabilities: BTreeSet<IntrinsicId>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolutionIdentity {
    pub logical: LogicalIdentity,
    pub entry: String,
    pub target: seismic_lang::logical::EffectiveTargetIdentity,
    pub toolchain_fingerprint: String,
    pub selections: BTreeMap<ChoiceId, (u32, u32)>,
}

// --- ABI --------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ResolvedAbi<D: ExecutableDialect> {
    pub buffers: Vec<BufferBinding>,
    pub scalars: ScalarLayout,
    pub results: Vec<ResultBinding>,
    pub alias_rules: Vec<AliasRule>,
    pub status: Option<StatusBinding<D>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferBinding {
    pub path: ValuePath,
    /// Representation plane name; `"dense"` for dense tensors.
    pub plane: String,
    pub binding: BufferBindingId,
    pub bytes: u64,
    pub alignment: u64,
    pub role: AbiRole,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbiRole {
    Parameter { ordinal: u32 },
    Result,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResultBinding {
    Buffer {
        path: ValuePath,
        plane: String,
        binding: BufferBindingId,
    },
    Scalar {
        path: ValuePath,
        field: ResultScalarFieldId,
        dtype: DType,
    },
    Range {
        path: ValuePath,
        start: ResultScalarFieldId,
        end: ResultScalarFieldId,
        bound: u64,
    },
}

/// Root ABI alias contract: shared-read ranges may overlap; exclusive/owned
/// ranges are disjoint from other live parameter ranges; results are
/// distinct; distinct representation planes are disjoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AliasRule {
    MayOverlap {
        left: BufferBindingId,
        right: BufferBindingId,
    },
    MustDisjoint {
        left: BufferBindingId,
        right: BufferBindingId,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StatusBinding<D: ExecutableDialect> {
    pub fields: Vec<StatusField<D>>,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StatusField<D: ExecutableDialect> {
    pub id: StatusFieldId,
    /// Provenance of the first error written to this field.
    pub node: NodeRef,
    pub index: usize,
    /// The planned predicate opcodes, when this field is a runtime check.
    pub predicate: Option<NonEmpty<D::Op>>,
}

// ---------------------------------------------------------------------------
// Feasible plan assignment
// ---------------------------------------------------------------------------

/// Compiler-side decoded plan values that accompany a solver-validated
/// feasible solution.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DecodedPlanValues {
    pub selections: BTreeMap<ChoiceId, (u32, u32)>,
    /// Plan parameter values by name (the substitution symbol environment).
    pub symbols: BTreeMap<String, i64>,
    pub offsets: BTreeMap<PhysicalStorageTemplateId, u64>,
}

#[derive(Clone, Debug)]
pub struct FeasiblePlanAssignment {
    assignment: FeasibleAssignment,
    decoded: DecodedPlanValues,
}

impl FeasiblePlanAssignment {
    /// The only constructor: requires the solver-validated feasible
    /// assignment of the exact model and the decoded values of that same
    /// solution. Resolution accepts nothing else.
    pub fn from_feasible(
        assignment: FeasibleAssignment,
        decoded: DecodedPlanValues,
    ) -> Result<Self, InvariantReport> {
        if decoded.symbols.values().any(|v| *v < 0) {
            return Err(InvariantReport(
                "decoded plan values contain a negative entry".into(),
            ));
        }
        Ok(Self {
            assignment,
            decoded,
        })
    }

    pub fn solver_assignment(&self) -> &FeasibleAssignment {
        &self.assignment
    }

    pub fn selections(&self) -> &BTreeMap<ChoiceId, (u32, u32)> {
        &self.decoded.selections
    }

    pub fn symbols(&self) -> &BTreeMap<String, i64> {
        &self.decoded.symbols
    }

    pub fn offsets(&self) -> &BTreeMap<PhysicalStorageTemplateId, u64> {
        &self.decoded.offsets
    }
}

// ---------------------------------------------------------------------------
// Cross-call fusion import
// ---------------------------------------------------------------------------

/// Whether a transport is kernel-local and therefore cannot cross a launch
/// or call boundary.
fn crosses_launch_boundary(transport: &TransportTemplate) -> bool {
    match transport {
        TransportTemplate::Kernel(_) => true,
        TransportTemplate::Tuple(items) => items.iter().any(crosses_launch_boundary),
        _ => false,
    }
}

/// Fresh id offsets for one cross-call import: child ids are shifted past
/// every id the importing alternative already uses.
struct ChildRemap {
    value: u32,
    state: u32,
    storage: u32,
    view: u32,
}

impl ChildRemap {
    fn value_id(&self, id: GraphValueId) -> GraphValueId {
        GraphValueId(id.0 + self.value)
    }
    fn state_id(&self, id: StateTokenId) -> StateTokenId {
        StateTokenId(id.0 + self.state)
    }
    fn storage_id(&self, id: LogicalStorageId) -> LogicalStorageId {
        LogicalStorageId(id.0 + self.storage)
    }
    fn view(&self, id: LogicalViewId) -> LogicalViewId {
        LogicalViewId(id.0 + self.view)
    }

    fn parameter(&self, parameter: &RegionParameter) -> RegionParameter {
        match parameter {
            RegionParameter::Value { id, ty } => RegionParameter::Value {
                id: self.value_id(*id),
                ty: ty.clone(),
            },
            RegionParameter::State { id, storage } => RegionParameter::State {
                id: self.state_id(*id),
                storage: self.storage_id(*storage),
            },
        }
    }

    fn result(&self, result: &RegionResult) -> RegionResult {
        match result {
            RegionResult::Value { id, ty } => RegionResult::Value {
                id: self.value_id(*id),
                ty: ty.clone(),
            },
            RegionResult::State { id, storage, join } => RegionResult::State {
                id: self.state_id(*id),
                storage: self.storage_id(*storage),
                join: join.clone(),
            },
        }
    }

    fn graph_value(&self, value: &GraphValue) -> GraphValue {
        GraphValue {
            id: self.value_id(value.id),
            ty: value.ty.clone(),
            view: value.view.map(|view| self.view(view)),
        }
    }

    fn state_token(&self, token: &StateToken) -> StateToken {
        StateToken {
            id: self.state_id(token.id),
            storage: self.storage_id(token.storage),
            join: token.join.as_ref().map(|join| self.state_join(join)),
        }
    }

    fn state_join(&self, join: &StateJoin) -> StateJoin {
        match join {
            StateJoin::DisjointWrite { binder, range } => StateJoin::DisjointWrite {
                binder: self.value_id(*binder),
                range: self.range(range),
            },
            StateJoin::Atomic { operations } => StateJoin::Atomic {
                operations: operations.clone(),
            },
        }
    }

    fn range(&self, range: &LogicalRange) -> LogicalRange {
        LogicalRange {
            start: self.value_id(range.start),
            end: self.value_id(range.end),
            bound: range.bound.clone(),
        }
    }

    fn obligation(&self, obligation: &SafetyObligation) -> SafetyObligation {
        match obligation {
            SafetyObligation::IndexInBounds { index, extent } => SafetyObligation::IndexInBounds {
                index: self.value_id(*index),
                extent: extent.clone(),
            },
            SafetyObligation::RangeInBounds { start, end, extent } => {
                SafetyObligation::RangeInBounds {
                    start: self.value_id(*start),
                    end: self.value_id(*end),
                    extent: extent.clone(),
                }
            }
            SafetyObligation::DivisorNonZero { value } => SafetyObligation::DivisorNonZero {
                value: self.value_id(*value),
            },
            SafetyObligation::SignedDivisionNoOverflow { lhs, rhs } => {
                SafetyObligation::SignedDivisionNoOverflow {
                    lhs: self.value_id(*lhs),
                    rhs: self.value_id(*rhs),
                }
            }
            SafetyObligation::ShiftInRange { value } => SafetyObligation::ShiftInRange {
                value: self.value_id(*value),
            },
            SafetyObligation::ShapeProductFits { factors, bits } => {
                SafetyObligation::ShapeProductFits {
                    factors: factors.clone(),
                    bits: *bits,
                }
            }
        }
    }

    fn join(&self, join: &JoinSlot) -> JoinSlot {
        match join {
            JoinSlot::Value {
                then_result,
                else_result,
                joined,
                ty,
            } => JoinSlot::Value {
                then_result: *then_result,
                else_result: *else_result,
                joined: self.value_id(*joined),
                ty: ty.clone(),
            },
            JoinSlot::State {
                then_result,
                else_result,
                joined,
                storage,
            } => JoinSlot::State {
                then_result: *then_result,
                else_result: *else_result,
                joined: self.state_id(*joined),
                storage: self.storage_id(*storage),
            },
        }
    }

    fn node(&self, node: &LogicalNode) -> LogicalNode {
        LogicalNode {
            inputs: node.inputs.iter().map(|id| self.value_id(*id)).collect(),
            state_inputs: node
                .state_inputs
                .iter()
                .map(|id| self.state_id(*id))
                .collect(),
            kind: self.kind(&node.kind),
            outputs: node.outputs.iter().map(|v| self.graph_value(v)).collect(),
            state_outputs: node
                .state_outputs
                .iter()
                .map(|t| self.state_token(t))
                .collect(),
            safety: node.safety.iter().map(|o| self.obligation(o)).collect(),
            span: node.span.clone(),
        }
    }

    fn kind(&self, kind: &LogicalNodeKind) -> LogicalNodeKind {
        match kind {
            LogicalNodeKind::Primitive(application) => {
                LogicalNodeKind::Primitive(application.clone())
            }
            LogicalNodeKind::If(if_node) => LogicalNodeKind::If(seismic_lang::logical::IfNode {
                condition: self.value_id(if_node.condition),
                then_region: self.rewrite_region(&if_node.then_region),
                else_region: self.rewrite_region(&if_node.else_region),
                joins: if_node.joins.iter().map(|j| self.join(j)).collect(),
                captured: if_node
                    .captured
                    .iter()
                    .map(|capture| seismic_lang::logical::IfCapture {
                        parameter: self.value_id(capture.parameter),
                        outer: self.value_id(capture.outer),
                    })
                    .collect(),
            }),
            LogicalNodeKind::Loop(loop_node) => {
                LogicalNodeKind::Loop(seismic_lang::logical::LoopNode {
                    kind: loop_node.kind,
                    range: self.range(&loop_node.range),
                    binder: self.value_id(loop_node.binder),
                    invariant_values: loop_node
                        .invariant_values
                        .iter()
                        .map(|id| self.value_id(*id))
                        .collect(),
                    initial_values: loop_node
                        .initial_values
                        .iter()
                        .map(|id| self.value_id(*id))
                        .collect(),
                    initial_states: loop_node
                        .initial_states
                        .iter()
                        .map(|id| self.state_id(*id))
                        .collect(),
                    body: self.rewrite_region(&loop_node.body),
                    carried: loop_node
                        .carried
                        .iter()
                        .map(|slot| seismic_lang::logical::CarriedSlot {
                            initial: match slot.initial {
                                seismic_lang::logical::RegionInput::Value(id) => {
                                    seismic_lang::logical::RegionInput::Value(self.value_id(id))
                                }
                                seismic_lang::logical::RegionInput::State(id) => {
                                    seismic_lang::logical::RegionInput::State(self.state_id(id))
                                }
                            },
                            body_parameter: slot.body_parameter,
                            body_result: slot.body_result,
                            loop_result: slot.loop_result,
                        })
                        .collect(),
                })
            }
            LogicalNodeKind::Reduction(reduction) => {
                LogicalNodeKind::Reduction(seismic_lang::logical::ReductionNode {
                    operand: self.value_id(reduction.operand),
                    axis: reduction.axis,
                    op: reduction.op,
                    order: reduction.order,
                    accumulator: reduction.accumulator,
                    result: reduction.result.clone(),
                })
            }
            LogicalNodeKind::Call(call_node) => LogicalNodeKind::Call(self.call(call_node)),
        }
    }

    fn call(&self, call_node: &CallNode) -> CallNode {
        CallNode {
            choice: call_node.choice,
            boundary_inputs: call_node
                .boundary_inputs
                .iter()
                .map(|input| seismic_lang::logical::BoundaryInput {
                    path: input.path.clone(),
                    param: input.param,
                    kind: match input.kind {
                        BoundaryInputKind::Value(value) => {
                            BoundaryInputKind::Value(self.value_id(value))
                        }
                        BoundaryInputKind::Shared { value, state } => BoundaryInputKind::Shared {
                            value: self.value_id(value),
                            state: self.state_id(state),
                        },
                        BoundaryInputKind::Exclusive { value, state } => {
                            BoundaryInputKind::Exclusive {
                                value: self.value_id(value),
                                state: self.state_id(state),
                            }
                        }
                        BoundaryInputKind::Move { value, state } => BoundaryInputKind::Move {
                            value: self.value_id(value),
                            state: self.state_id(state),
                        },
                    },
                })
                .collect(),
            boundary_results: call_node
                .boundary_results
                .iter()
                .map(|result| seismic_lang::logical::BoundaryResult {
                    path: result.path.clone(),
                    kind: match &result.kind {
                        BoundaryResultKind::Value(value) => {
                            BoundaryResultKind::Value(self.value_id(*value))
                        }
                        BoundaryResultKind::Storage { storage, ty, token } => {
                            BoundaryResultKind::Storage {
                                storage: self.storage_id(*storage),
                                ty: ty.clone(),
                                token: self.state_id(*token),
                            }
                        }
                        BoundaryResultKind::State(token) => {
                            BoundaryResultKind::State(self.state_id(*token))
                        }
                    },
                })
                .collect(),
        }
    }

    fn rewrite_region(
        &self,
        region: &seismic_lang::logical::GraphRegion,
    ) -> seismic_lang::logical::GraphRegion {
        seismic_lang::logical::GraphRegion {
            parameters: region
                .parameters
                .iter()
                .map(|p| self.parameter(p))
                .collect(),
            nodes: IdVec::from_iter(
                region
                    .nodes
                    .ids()
                    .zip(region.nodes.iter())
                    .map(|(id, node)| (id, self.node(node))),
            ),
            results: region.results.iter().map(|r| self.result(r)).collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared allocation state between the family and alternative builders
// ---------------------------------------------------------------------------

struct FamilyShared<D: ExecutableDialect> {
    next_storage: u32,
    storages: BTreeMap<u32, ConditionalStorageTemplate<D>>,
    next_slot: u32,
    slots: BTreeMap<u32, ExecutorScalarSlotDecl>,
    next_status_field: u32,
    next_kernel_value: u32,
    next_param: u32,
    params: BTreeMap<u32, PlanParameter>,
    /// ABI storage templates by boundary leaf, recorded by the entry
    /// alternative when it creates them.
    abi_leaves: BTreeMap<BoundaryLeaf, PhysicalStorageTemplateId>,
}

impl<D: ExecutableDialect> FamilyShared<D> {
    fn new() -> Self {
        Self {
            next_storage: 0,
            storages: BTreeMap::new(),
            next_slot: 0,
            slots: BTreeMap::new(),
            next_status_field: 0,
            next_kernel_value: 0,
            next_param: 0,
            params: BTreeMap::new(),
            abi_leaves: BTreeMap::new(),
        }
    }

    /// Register one plan parameter. Names are unique across the whole family
    /// (an alternative's symbol can never capture another launch's).
    fn insert_parameter(
        &mut self,
        name: String,
        lower: i64,
        upper: i64,
    ) -> Result<PlanParamId, BuilderError> {
        if lower < 1 || upper < lower {
            return Err(format!(
                "plan parameter domain {lower}..={upper} for `{name}` is invalid"
            ));
        }
        if self.params.values().any(|parameter| parameter.name == name) {
            return Err(format!("plan parameter `{name}` is declared twice"));
        }
        let id = PlanParamId(self.next_param);
        self.next_param += 1;
        self.params
            .insert(id.0, PlanParameter { name, lower, upper });
        Ok(id)
    }

    /// A parameter name under `prefix` that no other parameter of this family
    /// uses.
    fn unique_parameter_name(&mut self, prefix: &str) -> String {
        loop {
            let candidate = format!("{prefix}-{}", self.next_param);
            if !self
                .params
                .values()
                .any(|parameter| parameter.name == candidate)
            {
                return candidate;
            }
            self.next_param += 1;
        }
    }

    fn insert_storage(
        &mut self,
        template: ConditionalStorageTemplate<D>,
    ) -> PhysicalStorageTemplateId {
        let id = PhysicalStorageTemplateId(self.next_storage);
        self.next_storage += 1;
        self.storages.insert(id.0, template);
        id
    }

    fn fresh_slot(&mut self, dtype: DType, name: String) -> ExecutorScalarSlotId {
        let id = ExecutorScalarSlotId(self.next_slot);
        self.next_slot += 1;
        self.slots
            .insert(id.0, ExecutorScalarSlotDecl { dtype, name });
        id
    }

    fn fresh_kernel_value(&mut self) -> KernelValueTemplateId {
        let id = KernelValueTemplateId(self.next_kernel_value);
        self.next_kernel_value += 1;
        id
    }
}

// ---------------------------------------------------------------------------
// The consuming alternative builder
// ---------------------------------------------------------------------------

/// One snapshot region of the logical alternative.
#[derive(Clone, Debug)]
struct RegionSnapshot {
    path: RegionPath,
    parameters: Vec<RegionParameter>,
    nodes: BTreeMap<NodeId, LogicalNode>,
    results: Vec<RegionResult>,
}

/// One pending state edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum StateEdge {
    NodeInput {
        region: usize,
        node: NodeId,
        token: StateTokenId,
    },
    NodeOutput {
        region: usize,
        node: NodeId,
        token: StateTokenId,
    },
    LoopInitial {
        region: usize,
        node: NodeId,
        token: StateTokenId,
    },
    JoinState {
        region: usize,
        node: NodeId,
        token: StateTokenId,
    },
    CallBoundary {
        region: usize,
        node: NodeId,
        token: StateTokenId,
    },
    RegionState {
        region: usize,
        ordinal: usize,
        token: StateTokenId,
    },
}

/// One consumer of a value: a mapped node, or a structural use that forces an
/// external (non-kernel) transport.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Consumer {
    Node(NodeRef),
    Structural,
}

/// The sole constructor of one physical alternative. Every node, region
/// result, call, state edge, safety obligation, input and output of the
/// logical alternative starts pending; the consuming transitions remove exact
/// ids, and `finish_alternative` succeeds only when the logical and physical
/// pending sets are empty.
pub struct AlternativeBuilder<D: ExecutableDialect> {
    shared: Rc<RefCell<FamilyShared<D>>>,
    graph: TaskGraph,
    /// The occurrence's call interface; parameter ownership decides the
    /// access mode of entry tensor transports.
    interface: FunctionInterface,
    is_entry: bool,
    choice: ChoiceId,
    logical_alternative: u32,
    physical_alternative: u32,
    regions: Vec<RegionSnapshot>,
    consumers: BTreeMap<GraphValueId, Vec<Consumer>>,
    transports: BTreeMap<GraphValueId, TransportTemplate>,
    storage_templates: BTreeMap<LogicalStorageId, PhysicalStorageTemplateId>,
    abi_storages: BTreeMap<BoundaryLeaf, PhysicalStorageTemplateId>,
    pending_nodes: BTreeMap<NodeRef, LogicalNode>,
    pending_results: BTreeSet<(usize, usize)>,
    pending_state_edges: BTreeSet<StateEdge>,
    pending_obligations: BTreeSet<ObligationRef>,
    pending_inputs: BTreeSet<(usize, usize)>,
    pending_values: BTreeSet<GraphValueId>,
    parameter_values: BTreeMap<GraphValueId, (usize, usize)>,
    /// State-token region parameters, consumed when the token is consumed.
    parameter_states: BTreeMap<StateTokenId, (usize, usize)>,
    /// The paired view value of each tensor State parameter.
    parameter_state_values: BTreeMap<StateTokenId, GraphValueId>,
    step_storage_refs: BTreeMap<u32, BTreeSet<PhysicalStorageTemplateId>>,
    /// Storage templates referenced by completed boundary results.
    bound_result_templates: BTreeSet<PhysicalStorageTemplateId>,
    /// Arena-scoped templates created by this builder (activation is
    /// reference-based; unreferenced ones are dropped at finish).
    created_storage_templates: Vec<PhysicalStorageTemplateId>,
    /// Staged workgroup/participant storage attached to every launch.
    staged_workgroup: Vec<PhysicalStorageTemplateId>,
    staged_participant: Vec<PhysicalStorageTemplateId>,
    /// A planned barrier keeps the next mapped work in the same launch.
    extend_launch: bool,
    next_step: u32,
    scopes: Vec<Scope<D>>,
    discharges: Vec<DischargedObligation<D>>,
    /// Strategy-declared status fields (preconditions).
    declared_status_fields: Vec<StatusFieldId>,
    capabilities: BTreeSet<IntrinsicId>,
    cost: CostExpr,
    numerical: NumericalTransfer,
}

#[derive(Debug)]
struct Scope<D: ExecutableDialect> {
    steps: Vec<ScheduleStepTemplate<D>>,
}

impl<D: ExecutableDialect> AlternativeBuilder<D> {
    fn new(
        shared: Rc<RefCell<FamilyShared<D>>>,
        graph: TaskGraph,
        interface: FunctionInterface,
        is_entry: bool,
        choice: ChoiceId,
        logical_alternative: u32,
        physical_alternative: u32,
    ) -> Result<Self, BuilderError> {
        let mut builder = Self {
            shared,
            graph,
            interface,
            is_entry,
            choice,
            logical_alternative,
            physical_alternative,
            regions: Vec::new(),
            consumers: BTreeMap::new(),
            transports: BTreeMap::new(),
            storage_templates: BTreeMap::new(),
            abi_storages: BTreeMap::new(),
            pending_nodes: BTreeMap::new(),
            pending_results: BTreeSet::new(),
            pending_state_edges: BTreeSet::new(),
            pending_obligations: BTreeSet::new(),
            pending_inputs: BTreeSet::new(),
            pending_values: BTreeSet::new(),
            parameter_values: BTreeMap::new(),
            parameter_states: BTreeMap::new(),
            parameter_state_values: BTreeMap::new(),
            step_storage_refs: BTreeMap::new(),
            bound_result_templates: BTreeSet::new(),
            created_storage_templates: Vec::new(),
            staged_workgroup: Vec::new(),
            staged_participant: Vec::new(),
            extend_launch: false,
            next_step: 0,
            scopes: vec![Scope { steps: Vec::new() }],
            discharges: Vec::new(),
            declared_status_fields: Vec::new(),
            capabilities: BTreeSet::new(),
            cost: Sym::constant(0),
            numerical: NumericalTransfer::Exact,
        };
        builder.snapshot_regions(Vec::new())?;
        builder.collect_pending()?;
        builder.bind_boundary()?;
        Ok(builder)
    }

    // -- setup --------------------------------------------------------------

    /// Depth-first snapshot of every region, registering each in `regions`.
    fn snapshot_regions(&mut self, path: RegionPath) -> Result<(), BuilderError> {
        let region = self.region_at(&path)?;
        for (node_id, node) in region.nodes.ids().zip(region.nodes.iter()) {
            match &node.kind {
                LogicalNodeKind::If(_) => {
                    let mut then_path = path.clone();
                    then_path.push(RegionStep::IfThen(node_id));
                    self.snapshot_regions(then_path)?;
                    let mut else_path = path.clone();
                    else_path.push(RegionStep::IfElse(node_id));
                    self.snapshot_regions(else_path)?;
                }
                LogicalNodeKind::Loop(_) => {
                    let mut body_path = path.clone();
                    body_path.push(RegionStep::LoopBody(node_id));
                    self.snapshot_regions(body_path)?;
                }
                _ => {}
            }
        }
        let mut nodes = BTreeMap::new();
        for (node_id, node) in region.nodes.ids().zip(region.nodes.iter()) {
            nodes.insert(node_id, node.clone());
        }
        self.regions.push(RegionSnapshot {
            path,
            parameters: region.parameters.clone(),
            nodes,
            results: region.results.clone(),
        });
        Ok(())
    }

    fn region_at(
        &self,
        path: &RegionPath,
    ) -> Result<seismic_lang::logical::GraphRegion, BuilderError> {
        // Imported (cross-call fused) regions exist only as snapshots.
        if let Some(found) = self.regions.iter().find(|region| &region.path == path) {
            let mut nodes = Vec::new();
            for (node_id, node) in &found.nodes {
                nodes.push((*node_id, node.clone()));
            }
            return Ok(seismic_lang::logical::GraphRegion {
                parameters: found.parameters.clone(),
                nodes: IdVec::from_iter(nodes),
                results: found.results.clone(),
            });
        }
        let mut region = self.graph.root.clone();
        for step in path {
            let node = region
                .nodes
                .get(step.node())
                .ok_or_else(|| format!("region path names absent node#{}", step.node().0))?;
            match (&node.kind, step) {
                (LogicalNodeKind::If(if_node), RegionStep::IfThen(_)) => {
                    region = if_node.then_region.clone()
                }
                (LogicalNodeKind::If(if_node), RegionStep::IfElse(_)) => {
                    region = if_node.else_region.clone()
                }
                (LogicalNodeKind::Loop(loop_node), RegionStep::LoopBody(_)) => {
                    region = loop_node.body.clone()
                }
                _ => return Err("region path disagrees with graph structure".into()),
            }
        }
        Ok(region)
    }

    fn region_index(&self, path: &RegionPath) -> Result<usize, BuilderError> {
        self.regions
            .iter()
            .position(|region| &region.path == path)
            .ok_or_else(|| "region path is absent".to_string())
    }

    /// Register every pending obligation of the logical alternative.
    fn collect_pending(&mut self) -> Result<(), BuilderError> {
        for index in 0..self.regions.len() {
            self.register_region(index)?;
        }
        // Root outputs are the graph boundary results; their values are
        // always materialized externally.
        for result in self.graph.results.iter() {
            if let RegionResult::Value { id, .. } = result {
                self.consumers
                    .entry(*id)
                    .or_default()
                    .push(Consumer::Structural);
            }
        }
        Ok(())
    }

    /// Register every pending obligation of one (already snapshotted)
    /// region. Used at builder open and for cross-call fused imports.
    fn register_region(&mut self, index: usize) -> Result<(), BuilderError> {
        let region = self.regions[index].clone();
        let index = index;
        {
            for (param_index, parameter) in region.parameters.iter().enumerate() {
                self.pending_inputs.insert((index, param_index));
                if let RegionParameter::Value { id, .. } = parameter {
                    self.pending_values.insert(*id);
                    self.parameter_values.insert(*id, (index, param_index));
                }
                if let RegionParameter::State { id, .. } = parameter {
                    self.parameter_states.insert(*id, (index, param_index));
                    // A tensor interface parameter is one Value/State pair in
                    // the region parameter list: consuming the state also
                    // consumes the paired view value.
                    if param_index > 0 {
                        if let Some(RegionParameter::Value {
                            id: value,
                            ty: ValueType::Tensor(_),
                        }) = region.parameters.get(param_index - 1)
                        {
                            self.parameter_state_values.insert(*id, *value);
                        }
                    }
                }
            }
            for (ordinal, result) in region.results.iter().enumerate() {
                self.pending_results.insert((index, ordinal));
                match result {
                    RegionResult::Value { id, .. } => {
                        self.pending_values.insert(*id);
                        if !region.path.is_empty() {
                            self.consumers
                                .entry(*id)
                                .or_default()
                                .push(Consumer::Structural);
                        }
                    }
                    RegionResult::State { id, .. } => {
                        self.pending_state_edges.insert(StateEdge::RegionState {
                            region: index,
                            ordinal,
                            token: *id,
                        });
                    }
                }
            }
            for (node_id, node) in &region.nodes {
                let node_ref = NodeRef {
                    region: region.path.clone(),
                    node: *node_id,
                };
                self.pending_nodes.insert(node_ref.clone(), node.clone());
                for (obligation_index, _) in node.safety.iter().enumerate() {
                    self.pending_obligations.insert(ObligationRef {
                        node: node_ref.clone(),
                        index: obligation_index,
                    });
                }
                for token in node.state_inputs.iter() {
                    self.pending_state_edges.insert(StateEdge::NodeInput {
                        region: index,
                        node: *node_id,
                        token: *token,
                    });
                }
                for token in node.state_outputs.iter() {
                    self.pending_state_edges.insert(StateEdge::NodeOutput {
                        region: index,
                        node: *node_id,
                        token: token.id,
                    });
                }
                match &node.kind {
                    LogicalNodeKind::Primitive(_) => {
                        for input in node.inputs.iter() {
                            self.consumers
                                .entry(*input)
                                .or_default()
                                .push(Consumer::Node(node_ref.clone()));
                        }
                    }
                    LogicalNodeKind::If(if_node) => {
                        self.consumers
                            .entry(if_node.condition)
                            .or_default()
                            .push(Consumer::Structural);
                        for join in &if_node.joins {
                            if let seismic_lang::logical::JoinSlot::State { joined, .. } = join {
                                self.pending_state_edges.insert(StateEdge::JoinState {
                                    region: index,
                                    node: *node_id,
                                    token: *joined,
                                });
                            }
                        }
                    }
                    LogicalNodeKind::Loop(loop_node) => {
                        self.consumers
                            .entry(loop_node.range.start)
                            .or_default()
                            .push(Consumer::Structural);
                        self.consumers
                            .entry(loop_node.range.end)
                            .or_default()
                            .push(Consumer::Structural);
                        for value in loop_node
                            .invariant_values
                            .iter()
                            .chain(&loop_node.initial_values)
                        {
                            self.consumers
                                .entry(*value)
                                .or_default()
                                .push(Consumer::Structural);
                        }
                        for token in loop_node.initial_states.iter() {
                            self.pending_state_edges.insert(StateEdge::LoopInitial {
                                region: index,
                                node: *node_id,
                                token: *token,
                            });
                        }
                    }
                    LogicalNodeKind::Reduction(reduction) => {
                        self.consumers
                            .entry(reduction.operand)
                            .or_default()
                            .push(Consumer::Node(node_ref.clone()));
                    }
                    LogicalNodeKind::Call(call_node) => {
                        for input in &call_node.boundary_inputs {
                            match input.kind {
                                BoundaryInputKind::Value(value) => {
                                    self.consumers
                                        .entry(value)
                                        .or_default()
                                        .push(Consumer::Structural);
                                }
                                BoundaryInputKind::Shared {
                                    value,
                                    state: token,
                                }
                                | BoundaryInputKind::Exclusive {
                                    value,
                                    state: token,
                                }
                                | BoundaryInputKind::Move {
                                    value,
                                    state: token,
                                } => {
                                    self.consumers
                                        .entry(value)
                                        .or_default()
                                        .push(Consumer::Structural);
                                    self.pending_state_edges.insert(StateEdge::CallBoundary {
                                        region: index,
                                        node: *node_id,
                                        token,
                                    });
                                }
                            }
                        }
                    }
                }
                for output in node.outputs.iter() {
                    self.pending_values.insert(output.id);
                }
            }
        }
        Ok(())
    }

    /// Bind the boundary of this alternative: the entry binds root parameter
    /// and result leaves to the root ABI; every other choice binds its
    /// boundary leaves to placeholders resolved by the caller. Produced
    /// values receive default transports (scalar slots / storage views).
    fn canonical_abi_storage(
        &self,
        leaf: BoundaryLeaf,
        bytes: SizeExpr,
        alignment: u64,
        layout: D::LayoutTemplate,
    ) -> PhysicalStorageTemplateId {
        if let Some(existing) = self.shared.borrow().abi_leaves.get(&leaf).copied() {
            return existing;
        }
        let id = self
            .shared
            .borrow_mut()
            .insert_storage(ConditionalStorageTemplate {
                active_if: ActivationLiteral::always(),
                scope: StorageScope::Abi,
                bytes,
                alignment,
                replication: Replication::Once,
                layout,
                lifetime: StorageLifetime::Always,
            });
        self.shared.borrow_mut().abi_leaves.insert(leaf, id);
        id
    }

    fn bind_boundary(&mut self) -> Result<(), BuilderError> {
        for (storage_id, storage) in self.graph.storages.ids().zip(self.graph.storages.iter()) {
            let bytes = storage_bytes(&storage.shape)?;
            let alignment = alignment_of(&storage.shape);
            let origin = storage.origin.clone();
            let template = match (&origin, self.is_entry) {
                (seismic_lang::logical::StorageOrigin::Parameter { ordinal, path, .. }, true) => {
                    let leaf = BoundaryLeaf::Input {
                        param: *ordinal,
                        leaf: path.clone(),
                    };
                    let id = self.canonical_abi_storage(
                        leaf.clone(),
                        bytes.clone(),
                        alignment,
                        D::public_layout(&storage.shape),
                    );
                    self.abi_storages.insert(leaf, id);
                    Some(id)
                }
                (seismic_lang::logical::StorageOrigin::Result { owner: None, path }, true) => {
                    let leaf = BoundaryLeaf::Result { leaf: path.clone() };
                    let id = self.canonical_abi_storage(
                        leaf.clone(),
                        bytes.clone(),
                        alignment,
                        D::public_layout(&storage.shape),
                    );
                    self.abi_storages.insert(leaf, id);
                    Some(id)
                }
                // Owned storage inside this graph: a fresh device-arena
                // template conditional on this alternative.
                (seismic_lang::logical::StorageOrigin::Owned, _) => {
                    let mut active = ActivationLiteral::default();
                    active.0.insert(ActivationTerm {
                        choice: self.choice,
                        logical_alternative: self.logical_alternative,
                        physical_alternative: self.physical_alternative,
                    });
                    Some(
                        self.shared
                            .borrow_mut()
                            .insert_storage(ConditionalStorageTemplate {
                                active_if: active,
                                scope: StorageScope::DeviceArena,
                                bytes,
                                alignment,
                                replication: Replication::Once,
                                layout: D::internal_layout(&storage.shape),
                                lifetime: StorageLifetime::Always,
                            }),
                    )
                }
                // An occurrence-owned call-result storage inside the entry
                // graph is an internal arena allocation (the public entry
                // results are promoted to ABI below); inside a child graph
                // the caller provides it.
                (seismic_lang::logical::StorageOrigin::Result { owner: Some(_), .. }, true) => {
                    let mut active = ActivationLiteral::default();
                    active.0.insert(ActivationTerm {
                        choice: self.choice,
                        logical_alternative: self.logical_alternative,
                        physical_alternative: self.physical_alternative,
                    });
                    Some(
                        self.shared
                            .borrow_mut()
                            .insert_storage(ConditionalStorageTemplate {
                                active_if: active,
                                scope: StorageScope::DeviceArena,
                                bytes,
                                alignment,
                                replication: Replication::Once,
                                layout: D::internal_layout(&storage.shape),
                                lifetime: StorageLifetime::Always,
                            }),
                    )
                }
                // Non-entry parameter/result storage: the caller provides it.
                _ => None,
            };
            if let Some(id) = template {
                self.storage_templates.insert(storage_id, id);
                self.created_storage_templates.push(id);
            }
        }
        // Transports for root-region parameter leaves.  Region parameters
        // are flattened canonical leaves, while the public interface retains
        // aggregate parameter ordinals; walk both structures together so a
        // tuple's second leaf does not masquerade as the next parameter.
        let root = self
            .regions
            .iter()
            .find(|region| region.path.is_empty())
            .expect("the root region exists")
            .clone();
        let mut param_index = 0usize;
        for (ordinal, parameter) in self.interface.params.iter().enumerate() {
            let paths = canonical_leaves(&parameter.ty)
                .map(|leaves| leaves.into_iter().map(|(path, _)| path).collect::<Vec<_>>())
                .unwrap_or_else(|_| vec![ValuePath::default()]);
            for path in paths {
                let Some(RegionParameter::Value { id, ty }) = root.parameters.get(param_index)
                else {
                    return Err(format!(
                        "root parameter leaf {ordinal}{path} has no value parameter"
                    ));
                };
                let is_tensor = matches!(ty, ValueType::Tensor(_));
                let transport = self.boundary_transport(
                    ty,
                    BoundaryLeaf::Input {
                        param: ordinal as u32,
                        leaf: path,
                    },
                    parameter.ownership,
                )?;
                self.transports.insert(*id, transport);
                param_index += 1;
                if is_tensor {
                    match root.parameters.get(param_index) {
                        Some(RegionParameter::State { .. }) => param_index += 1,
                        _ => {
                            return Err(
                                "a tensor root parameter has no paired state parameter".into()
                            );
                        }
                    }
                }
            }
        }
        if param_index != root.parameters.len() {
            return Err("root parameters exceed the flattened function interface".into());
        }
        // Entry result storages are public ABI buffers regardless of the
        // occurrence that wrote them; identify them through the value ->
        // storage map of node outputs and region state results.
        let mut value_storage: BTreeMap<GraphValueId, LogicalStorageId> = BTreeMap::new();
        for region in &self.regions {
            for node in region.nodes.values() {
                for output in node.outputs.iter() {
                    if let Some(view) = output.view {
                        if let Some(logical_view) = self.graph.views.get(view) {
                            value_storage.insert(output.id, logical_view.storage);
                        }
                    }
                }
            }
        }
        let result_leaves = canonical_leaves(&self.interface.result)?;
        let mut entry_result_storages: BTreeMap<LogicalStorageId, ValuePath> = BTreeMap::new();
        let mut result_paths: BTreeMap<GraphValueId, ValuePath> = BTreeMap::new();
        let mut value_ordinal = 0usize;
        for result in self.graph.results.iter() {
            match result {
                RegionResult::Value { id, ty } => {
                    let path = result_leaves
                        .get(value_ordinal)
                        .map(|(path, _)| path.clone())
                        .ok_or_else(|| {
                            "a graph value result has no canonical interface leaf".to_string()
                        })?;
                    value_ordinal += 1;
                    result_paths.insert(*id, path.clone());
                    if let Some(storage) = value_storage.get(id) {
                        entry_result_storages
                            .entry(*storage)
                            .or_insert_with(|| path.clone());
                    }
                    let _ = ty;
                }
                // State results close dataflow for parameter and owned
                // storages; they are not additional function-result leaves.
                // Public result storage is identified by the tensor value
                // result that views it.
                RegionResult::State { .. } => {}
            }
        }
        if value_ordinal != result_leaves.len() {
            return Err("graph value results do not match the flattened interface results".into());
        }
        for (storage_id, leaf) in entry_result_storages {
            // Skip only when the storage already has a public ABI template;
            // an owned storage backing an entry result is promoted from its
            // arena template to the public ABI here (the orphan arena
            // template is dropped by reference-based activation).
            if let Some(existing) = self.storage_templates.get(&storage_id) {
                let is_abi = matches!(
                    self.shared
                        .borrow()
                        .storages
                        .get(&existing.0)
                        .map(|t| t.scope),
                    Some(StorageScope::Abi)
                );
                if is_abi {
                    continue;
                }
            }
            let storage = self.graph.storages.get(storage_id).cloned();
            let Some(storage) = storage else { continue };
            if self.is_entry {
                let bytes = storage_bytes(&storage.shape)?;
                let alignment = alignment_of(&storage.shape);
                let leaf = BoundaryLeaf::Result { leaf };
                let id = self.canonical_abi_storage(
                    leaf.clone(),
                    bytes,
                    alignment,
                    D::public_layout(&storage.shape),
                );
                self.abi_storages.insert(leaf, id);
                self.storage_templates.insert(storage_id, id);
            }
        }
        // Every entry result tensor leaf gets a public ABI result buffer,
        // including a result that is a moved-in parameter returned (the
        // parameter binding stays for the body; the result value binds the
        // result buffer).
        if self.is_entry {
            for result in self.graph.results.iter() {
                if let RegionResult::Value { id, ty } = result {
                    let ValueType::Tensor(shape) = ty else {
                        continue;
                    };
                    let leaf = result_paths.get(id).cloned().ok_or_else(|| {
                        "a graph result has no canonical interface leaf".to_string()
                    })?;
                    let key = BoundaryLeaf::Result { leaf };
                    if self.abi_storages.contains_key(&key) {
                        continue;
                    }
                    let bytes = storage_bytes(shape)?;
                    let alignment = alignment_of(shape);
                    let id = self.canonical_abi_storage(
                        key.clone(),
                        bytes,
                        alignment,
                        D::public_layout(shape),
                    );
                    self.abi_storages.insert(key, id);
                }
            }
        }
        // Transports for graph boundary result values (entry: ABI result
        // leaves; child: boundary placeholders).
        for result in self.graph.results.iter() {
            if let RegionResult::Value { id, ty } = result {
                // A placeholder leaf; the invoking caller's boundary result
                // transport replaces it (or the entry completes it).
                let leaf = result_paths
                    .get(id)
                    .cloned()
                    .ok_or_else(|| "a graph result has no canonical interface leaf".to_string())?;
                let transport = self.boundary_transport(
                    ty,
                    BoundaryLeaf::Result { leaf },
                    ParamOwnership::Owned,
                )?;
                self.transports.insert(*id, transport);
            }
        }
        // Transports for node output values (produced inside this graph).
        for region in &self.regions {
            for node in region.nodes.values() {
                for output in node.outputs.iter() {
                    let transport = self.produced_transport(output, node)?;
                    self.transports.insert(output.id, transport);
                }
            }
        }
        Ok(())
    }

    /// Transport of one boundary leaf: entry leaves are ABI-backed,
    /// non-entry leaves are boundary placeholders.
    fn boundary_transport(
        &self,
        ty: &ValueType,
        leaf: BoundaryLeaf,
        ownership: ParamOwnership,
    ) -> Result<TransportTemplate, BuilderError> {
        match ty {
            ValueType::Void => Ok(TransportTemplate::Void),
            ValueType::Scalar(dtype) => {
                if self.is_entry {
                    Ok(TransportTemplate::ExecutorScalar(ExecutorScalarTemplate {
                        source: ExecutorScalarSource::Abi {
                            leaf,
                            endpoint: None,
                        },
                        dtype: *dtype,
                    }))
                } else {
                    Ok(TransportTemplate::Boundary(leaf))
                }
            }
            ValueType::Index { .. } => {
                self.boundary_transport(&ValueType::Scalar(DType::I32), leaf, ownership)
            }
            ValueType::Range { .. } => {
                let scalar = |endpoint| {
                    if self.is_entry {
                        TransportTemplate::ExecutorScalar(ExecutorScalarTemplate {
                            source: ExecutorScalarSource::Abi {
                                leaf: leaf.clone(),
                                endpoint: Some(endpoint),
                            },
                            dtype: DType::I32,
                        })
                    } else {
                        TransportTemplate::Boundary(leaf.clone())
                    }
                };
                let start = scalar(seismic_lang::abi::RangeEndpoint::Start);
                let end = scalar(seismic_lang::abi::RangeEndpoint::End);
                Ok(TransportTemplate::Tuple(
                    NonEmpty::new(vec![start, end]).expect("a range has two scalar fields"),
                ))
            }
            ValueType::Tensor(_) => {
                if self.is_entry {
                    let template = self
                        .abi_storages
                        .get(&leaf)
                        .copied()
                        .ok_or_else(|| format!("no ABI storage for boundary leaf {leaf:?}"))?;
                    // Writability follows the parameter ownership: a
                    // shared borrow reads; an owned or exclusively borrowed
                    // tensor is a write path.
                    let access = match ownership {
                        ParamOwnership::Shared | ParamOwnership::Value => Access::Shared,
                        ParamOwnership::Owned | ParamOwnership::Exclusive => Access::Exclusive,
                    };
                    Ok(TransportTemplate::Storage(
                        NonEmpty::new(vec![StorageViewTemplate {
                            storage: template,
                            access,
                            transform: ViewTransform::Identity,
                        }])
                        .expect("one plane"),
                    ))
                } else {
                    Ok(TransportTemplate::Boundary(leaf))
                }
            }
            ValueType::Tuple(items) => Ok(TransportTemplate::Tuple(
                NonEmpty::new(
                    items
                        .iter()
                        .enumerate()
                        .map(|(index, item)| {
                            let component = match &leaf {
                                BoundaryLeaf::Input { param, leaf } => BoundaryLeaf::Input {
                                    param: *param,
                                    leaf: leaf.extend(index as u32),
                                },
                                BoundaryLeaf::Result { leaf } => BoundaryLeaf::Result {
                                    leaf: leaf.extend(index as u32),
                                },
                            };
                            self.boundary_transport(item, component, ownership)
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )
                .expect("a tuple is nonempty"),
            )),
            ValueType::CapabilityValue(_) => {
                Err("a capability value cannot cross a portable boundary".into())
            }
        }
    }

    /// Default transport of one produced (node output) value: tensors live in
    /// their backing storage; scalars cross boundaries through fresh planned
    /// executor-scalar slots written by the producing launch.
    fn produced_transport(
        &self,
        output: &GraphValue,
        node: &LogicalNode,
    ) -> Result<TransportTemplate, BuilderError> {
        if output.view.is_some() {
            let access = output
                .view
                .and_then(|view| self.graph.views.get(view))
                .map(|view| view.access)
                .unwrap_or(Access::Exclusive);
            return self.tensor_transport(output, access);
        }
        if let LogicalNodeKind::Primitive(application) = &node.kind {
            match &application.op {
                PrimitiveOp::Primitive(PrimitiveId::TuplePack) => {
                    let items = node
                        .inputs
                        .iter()
                        .map(|value| self.transport_of(*value))
                        .collect::<Result<Vec<_>, _>>()?;
                    return Ok(TransportTemplate::Tuple(
                        NonEmpty::new(items).ok_or("tuple.pack has no components")?,
                    ));
                }
                PrimitiveOp::Primitive(PrimitiveId::TupleGet(index)) => {
                    let tuple = node
                        .inputs
                        .first()
                        .copied()
                        .ok_or("tuple.get has no tuple operand")?;
                    let TransportTemplate::Tuple(items) = self.transport_of(tuple)? else {
                        return Err("tuple.get operand does not transport as a tuple".into());
                    };
                    return items
                        .iter()
                        .nth(*index)
                        .cloned()
                        .ok_or_else(|| format!("tuple.get index {index} is out of bounds"));
                }
                _ => {}
            }
        }
        match &output.ty {
            ValueType::Void => Ok(TransportTemplate::Void),
            ValueType::Scalar(_) | ValueType::Index { .. } => {
                let dtype = output.ty.scalar_dtype().unwrap_or(DType::I32);
                let name = format!("slot-{}", self.shared.borrow().next_slot);
                let slot = self.shared.borrow_mut().fresh_slot(dtype, name);
                Ok(TransportTemplate::ExecutorScalar(ExecutorScalarTemplate {
                    source: ExecutorScalarSource::Slot(slot),
                    dtype,
                }))
            }
            ValueType::Range { .. } | ValueType::Tuple(_) => {
                let _ = node;
                let leaves = canonical_leaves(&output.ty).map_err(|reason| {
                    format!("produced value has no canonical leaves: {reason}")
                })?;
                let leaf_transports: Vec<TransportTemplate> = leaves
                    .iter()
                    .map(|(_, leaf)| match leaf {
                        Leaf::Scalar(dtype) => {
                            Ok(TransportTemplate::ExecutorScalar(ExecutorScalarTemplate {
                                source: ExecutorScalarSource::Slot(
                                    self.shared.borrow_mut().fresh_slot(*dtype, "leaf".into()),
                                ),
                                dtype: *dtype,
                            }))
                        }
                        Leaf::Index(_) => {
                            Ok(TransportTemplate::ExecutorScalar(ExecutorScalarTemplate {
                                source: ExecutorScalarSource::Slot(
                                    self.shared
                                        .borrow_mut()
                                        .fresh_slot(DType::I32, "leaf".into()),
                                ),
                                dtype: DType::I32,
                            }))
                        }
                        Leaf::Range(_) => Ok(TransportTemplate::Tuple(
                            NonEmpty::new(vec![
                                TransportTemplate::ExecutorScalar(ExecutorScalarTemplate {
                                    source: ExecutorScalarSource::Slot(
                                        self.shared
                                            .borrow_mut()
                                            .fresh_slot(DType::I32, "range".into()),
                                    ),
                                    dtype: DType::I32,
                                }),
                                TransportTemplate::ExecutorScalar(ExecutorScalarTemplate {
                                    source: ExecutorScalarSource::Slot(
                                        self.shared
                                            .borrow_mut()
                                            .fresh_slot(DType::I32, "range".into()),
                                    ),
                                    dtype: DType::I32,
                                }),
                            ])
                            .expect("a range has two scalar fields"),
                        )),
                        Leaf::Tensor(_) => {
                            Err("a tensor tuple leaf requires structural tuple transport".into())
                        }
                    })
                    .collect::<Result<Vec<_>, BuilderError>>()?;
                Ok(TransportTemplate::Tuple(
                    NonEmpty::new(leaf_transports).expect("a tuple is nonempty"),
                ))
            }
            ValueType::Tensor(_) => {
                Err("a computed tensor output has no explicit logical storage view".into())
            }
            ValueType::CapabilityValue(_) => Err(
                "a capability value cannot cross a launch boundary unfused; fuse its producer"
                    .into(),
            ),
        }
    }

    /// The transport of one tensor value through its backing logical
    /// storage's physical template. Non-entry boundary storages without a
    /// template transport by boundary path.
    fn tensor_transport(
        &self,
        value: &GraphValue,
        access: Access,
    ) -> Result<TransportTemplate, BuilderError> {
        let view = value
            .view
            .ok_or_else(|| format!("tensor value#{} has no backing view", value.id.0))?;
        let storage_id = self
            .graph
            .views
            .get(view)
            .ok_or_else(|| format!("view#{} is absent", view.0))?
            .storage;
        let transform = self
            .graph
            .views
            .get(view)
            .map(|view| view.transform.clone())
            .unwrap_or(ViewTransform::Identity);
        if let Some(template) = self.storage_templates.get(&storage_id) {
            Ok(TransportTemplate::Storage(
                NonEmpty::new(vec![StorageViewTemplate {
                    storage: *template,
                    access,
                    transform,
                }])
                .expect("one plane"),
            ))
        } else {
            let leaf = match self
                .graph
                .storages
                .get(storage_id)
                .map(|s| s.origin.clone())
            {
                Some(seismic_lang::logical::StorageOrigin::Parameter { ordinal, path, .. }) => {
                    BoundaryLeaf::Input {
                        param: ordinal,
                        leaf: path,
                    }
                }
                Some(seismic_lang::logical::StorageOrigin::Result { path, .. }) => {
                    BoundaryLeaf::Result { leaf: path }
                }
                _ => return Err(format!("storage#{} has no physical home", storage_id.0)),
            };
            Ok(TransportTemplate::Boundary(leaf))
        }
    }

    // -- queries (non-consuming) -------------------------------------------

    /// Recorded transport of one value (the strategy's view of a producer).
    pub fn transport_of(&self, value: GraphValueId) -> Result<TransportTemplate, BuilderError> {
        self.transports.get(&value).cloned().ok_or_else(|| {
            format!(
                "value#{} has no bound transport (producer unmapped)",
                value.0
            )
        })
    }

    /// Publish one produced value through a chosen transport (a strategy
    /// defining where its result physically lives, like a reduction result
    /// or a matrix accumulation into an operand storage).
    pub fn bind_value_transport(
        &mut self,
        value: GraphValueId,
        transport: TransportTemplate,
    ) -> Result<(), BuilderError> {
        if self.pending_values.remove(&value) {
            self.transports.insert(value, transport);
            Ok(())
        } else {
            Err(format!(
                "value#{} is not a pending produced value (already consumed or absent)",
                value.0
            )
            .into())
        }
    }

    /// The ABI storage template backing one root boundary leaf (entry only).
    pub fn abi_storage_of(&self, leaf: &BoundaryLeaf) -> Option<PhysicalStorageTemplateId> {
        self.abi_storages.get(leaf).copied()
    }

    /// The physical template backing one logical storage, if this graph owns it.
    pub fn storage_of(&self, storage: LogicalStorageId) -> Option<PhysicalStorageTemplateId> {
        self.storage_templates.get(&storage).copied()
    }

    /// Physical transport of a logical storage at a graph boundary. Owned
    /// storages must have a physical template; only parameter/result origins
    /// may remain boundary placeholders in a non-entry graph.
    pub fn transport_of_storage(
        &self,
        storage: LogicalStorageId,
        access: Access,
    ) -> Result<TransportTemplate, BuilderError> {
        self.logical_storage_transport(storage, access)
    }

    /// Construct the one canonical transport boundary for a call occurrence.
    ///
    /// Logical normalization records both halves of every tensor argument:
    /// the exact view value and the storage state/version.  Realization owns
    /// the conversion of those logical facts into physical transports; a
    /// backend must not reconstruct or approximate this boundary itself.
    /// The canonical boundary templates of one call (used by invoke and by
    /// cross-call fusion).
    pub fn canonical_call_boundary(
        &self,
        call: &CallNode,
    ) -> Result<BoundaryTemplates, BuilderError> {
        let mut boundary = BoundaryTemplates::default();
        for input in &call.boundary_inputs {
            let leaf = BoundaryLeaf::Input {
                param: input.param,
                leaf: input.path.clone(),
            };
            match input.kind {
                BoundaryInputKind::Value(value) => {
                    let transport = self.transport_of(value)?;
                    if crosses_launch_boundary(&transport) {
                        return Err(format!(
                            "kernel-local value#{} cannot cross call boundary {leaf:?}",
                            value.0
                        ));
                    }
                    boundary.inputs.insert(leaf, transport);
                }
                BoundaryInputKind::Shared { value, state }
                | BoundaryInputKind::Exclusive { value, state }
                | BoundaryInputKind::Move { value, state } => {
                    let transport = self.transport_of(value)?;
                    if crosses_launch_boundary(&transport) {
                        return Err(format!(
                            "kernel-local tensor view value#{} cannot cross call boundary {leaf:?}",
                            value.0
                        ));
                    }
                    boundary.inputs.insert(leaf.clone(), transport);
                    let storage = self.logical_storage_of_token(state).ok_or_else(|| {
                        format!("call boundary state#{} has no logical storage", state.0)
                    })?;
                    boundary.states.insert(leaf, self.state_transport(storage)?);
                }
            }
        }
        for result in &call.boundary_results {
            let leaf = BoundaryLeaf::Result {
                leaf: result.path.clone(),
            };
            match &result.kind {
                BoundaryResultKind::Value(value) => {
                    let transport = self.transport_of(*value)?;
                    if crosses_launch_boundary(&transport) {
                        return Err(format!(
                            "kernel-local result value#{} cannot cross call boundary {leaf:?}",
                            value.0
                        ));
                    }
                    boundary.results.insert(leaf, transport);
                }
                BoundaryResultKind::Storage { storage, .. } => {
                    boundary.results.insert(
                        leaf,
                        self.logical_storage_transport(*storage, Access::Exclusive)?,
                    );
                }
                BoundaryResultKind::State(token) => {
                    let storage = self.logical_storage_of_token(*token).ok_or_else(|| {
                        format!("call result state#{} has no logical storage", token.0)
                    })?;
                    boundary.states.insert(leaf, self.state_transport(storage)?);
                }
            }
        }
        Ok(boundary)
    }

    fn logical_storage_transport(
        &self,
        storage: LogicalStorageId,
        access: Access,
    ) -> Result<TransportTemplate, BuilderError> {
        if let Some(template) = self.storage_of(storage) {
            return Ok(TransportTemplate::Storage(
                NonEmpty::new(vec![StorageViewTemplate {
                    storage: template,
                    access,
                    transform: ViewTransform::Identity,
                }])
                .expect("one storage view"),
            ));
        }
        Ok(TransportTemplate::Boundary(
            self.storage_boundary_leaf(storage)?,
        ))
    }

    fn state_transport(
        &self,
        storage: LogicalStorageId,
    ) -> Result<StateTransportTemplate, BuilderError> {
        if let Some(template) = self.storage_of(storage) {
            return Ok(StateTransportTemplate::Storage(template));
        }
        Ok(StateTransportTemplate::Boundary(
            self.storage_boundary_leaf(storage)?,
        ))
    }

    fn storage_boundary_leaf(
        &self,
        storage_id: LogicalStorageId,
    ) -> Result<BoundaryLeaf, BuilderError> {
        let storage = self
            .graph
            .storages
            .get(storage_id)
            .ok_or_else(|| format!("logical storage#{} is absent", storage_id.0))?;
        match &storage.origin {
            seismic_lang::logical::StorageOrigin::Parameter { ordinal, path, .. } => {
                Ok(BoundaryLeaf::Input {
                    param: *ordinal,
                    leaf: path.clone(),
                })
            }
            seismic_lang::logical::StorageOrigin::Result { path, .. } => {
                Ok(BoundaryLeaf::Result { leaf: path.clone() })
            }
            seismic_lang::logical::StorageOrigin::Owned => Err(format!(
                "owned logical storage#{} has no physical template",
                storage_id.0
            )),
        }
    }

    /// Declare a planned executor-scalar slot for carries/control materialization.
    pub fn declare_executor_scalar_slot(
        &mut self,
        dtype: DType,
        name: impl Into<String>,
    ) -> ExecutorScalarSlotId {
        self.shared.borrow_mut().fresh_slot(dtype, name.into())
    }

    /// Declare one solver-tunable plan parameter from inside an alternative
    /// (e.g. the participant width of a grid-stride `LinearIterationMap`).
    /// The name is unique across the whole family, so the returned symbol can
    /// never capture another alternative's or launch's; the parameter is
    /// registered in the family's `parameters` and the one planning model
    /// constrains it like every other tuning parameter.
    pub fn plan_parameter(
        &mut self,
        name: impl Into<String>,
        lower: i64,
        upper: i64,
    ) -> Result<PlanParamId, BuilderError> {
        self.shared
            .borrow_mut()
            .insert_parameter(name.into(), lower, upper)
    }

    /// A unique solver-tunable symbol for the physical linear participant
    /// count of a grid-stride iteration map: allocates a family-unique
    /// `participants-<n>` plan parameter over `lower..=upper` and returns its
    /// symbol. Use as `map.with_participants(builder.solver_participants(1, 256)?)`.
    pub fn solver_participants(&mut self, lower: i64, upper: i64) -> Result<Sym, BuilderError> {
        let name = self
            .shared
            .borrow_mut()
            .unique_parameter_name("participants");
        self.plan_parameter(name.clone(), lower, upper)?;
        Ok(Sym::param(&name))
    }

    /// Declare one staged workgroup/participant storage allocation: exact
    /// aligned bytes and replication are hard resources of every launch of
    /// this alternative.
    pub fn stage_storage(
        &mut self,
        scope: StorageScope,
        bytes: SizeExpr,
        alignment: u64,
        replication: Replication,
        lifetime: StorageLifetime,
    ) -> Result<PhysicalStorageTemplateId, BuilderError> {
        if !matches!(scope, StorageScope::Workgroup | StorageScope::Participant) {
            return Err("staged storage is workgroup- or participant-scoped".into());
        }
        if alignment == 0 {
            return Err("staged storage alignment must be positive".into());
        }
        let mut active = ActivationLiteral::default();
        active.0.insert(ActivationTerm {
            choice: self.choice,
            logical_alternative: self.logical_alternative,
            physical_alternative: self.physical_alternative,
        });
        let template = ConditionalStorageTemplate {
            active_if: active,
            scope,
            bytes: bytes.clone(),
            alignment,
            replication,
            layout: D::staged_layout(&bytes, alignment),
            lifetime,
        };
        let id = self.shared.borrow_mut().insert_storage(template);
        self.created_storage_templates.push(id);
        match scope {
            StorageScope::Workgroup => self.staged_workgroup.push(id),
            StorageScope::Participant => self.staged_participant.push(id),
            _ => unreachable!("scope checked above"),
        }
        Ok(id)
    }

    /// Plan one barrier between staged phases of the open launch of the
    /// current scope. Emitters add no synchronization of their own. The next
    /// `map_primitive`/`map_reduction`/`fuse` continues in that same launch
    /// after the barrier.
    pub fn barrier(&mut self, scope: BarrierScope) -> Result<(), BuilderError> {
        let launch_open = {
            let scope_steps = self.scopes.last_mut().expect("a schedule scope is open");
            let last = scope_steps
                .steps
                .last_mut()
                .ok_or_else(|| "a barrier needs an open launch".to_string())?;
            let launch = match last {
                ScheduleStepTemplate::Launch(launch) => launch,
                _ => return Err("a barrier needs the last step of the scope to be a launch".into()),
            };
            let mut steps = launch.kernel.steps.clone().into_vec();
            steps.push(KernelStepTemplate::Barrier { scope });
            launch.kernel.steps = NonEmpty::new(steps).expect("steps is nonempty");
            true
        };
        if !launch_open {
            return Err("a barrier needs an open launch".into());
        }
        // A barrier is a planned fact: it carries cost and is retained.
        self.cost = self.cost.add(&Sym::constant(1));
        self.extend_launch = true;
        Ok(())
    }

    /// Allocate one status field for a strategy-level precondition (e.g. a
    /// reduction nonempty check). The field is a planned status write of
    /// this alternative, sized with the discharged fields.
    pub fn status_field(&mut self) -> StatusFieldId {
        let field = StatusFieldId(u64::from(self.shared.borrow().next_status_field));
        self.shared.borrow_mut().next_status_field += 1;
        self.declared_status_fields.push(field);
        field
    }

    /// Contribute one numerical transfer of this alternative; composed
    /// immediately via `numerics::compose` into the whole-candidate
    /// transfer.
    pub fn note_numerical(&mut self, transfer: NumericalTransfer) {
        self.numerical = crate::numerics::compose(&transfer, &self.numerical);
    }

    /// The cross-call form of `fuse`: fuse one caller/callee interval
    /// through boundary substitution and consumption of the child graph
    /// obligations. The child alternative's regions are imported (under a
    /// `RegionStep::FusedCall` path prefix) with fresh value/state/storage
    /// ids; their pending obligations become this alternative's own, to be
    /// consumed by the ordinary transitions. `TransportTemplate::Kernel` is
    /// illegal across the fused boundary; the child's boundary transports
    /// resolve directly to caller storage/slots/placeholders.
    pub fn fuse_call(
        &mut self,
        call: NodeRef,
        child_graph: &TaskGraph,
        boundary: BoundaryTemplates,
    ) -> Result<FusedImport, BuilderError> {
        self.check_pending(&call)?;
        let logical = self
            .pending_nodes
            .get(&call)
            .cloned()
            .ok_or_else(internal_pending)?;
        let call_node = match &logical.kind {
            LogicalNodeKind::Call(call_node) => call_node.clone(),
            _ => return Err("fuse_call requires a call node".into()),
        };
        let call_region = self.region_index(&call.region)?;
        // Kernel-local transports never cross a call boundary.
        for (leaf, transport) in boundary.inputs.iter().chain(boundary.results.iter()) {
            if crosses_launch_boundary(transport) {
                return Err(format!(
                    "a kernel-local transport cannot cross the fused call boundary at {leaf:?}"
                ));
            }
        }
        // Consume the call occurrence (as `invoke` does), but emit no
        // `CallTemplate` step: the child interval is inlined here.
        for input in &call_node.boundary_inputs {
            let input_leaf = BoundaryLeaf::Input {
                param: input.param,
                leaf: input.path.clone(),
            };
            match input.kind {
                BoundaryInputKind::Value(value) => {
                    let transport =
                        boundary.inputs.get(&input_leaf).cloned().ok_or_else(|| {
                            format!("fuse_call omits boundary input {input_leaf:?}")
                        })?;
                    self.transports.insert(value, transport);
                    self.consume_value(value)?;
                }
                BoundaryInputKind::Shared {
                    value,
                    state: token,
                }
                | BoundaryInputKind::Exclusive {
                    value,
                    state: token,
                }
                | BoundaryInputKind::Move {
                    value,
                    state: token,
                } => {
                    let transport =
                        boundary.inputs.get(&input_leaf).cloned().ok_or_else(|| {
                            format!("fuse_call omits boundary input {input_leaf:?}")
                        })?;
                    self.transports.insert(value, transport);
                    self.consume_value(value)?;
                    if !self.pending_state_edges.remove(&StateEdge::CallBoundary {
                        region: call_region,
                        node: call.node,
                        token,
                    }) {
                        return Err("a call boundary state edge is absent".into());
                    }
                    self.consume_parameter_state(token);
                }
            }
        }
        for result in &call_node.boundary_results {
            let result_leaf = BoundaryLeaf::Result {
                leaf: result.path.clone(),
            };
            match &result.kind {
                BoundaryResultKind::Value(value) => {
                    let transport =
                        boundary.results.get(&result_leaf).cloned().ok_or_else(|| {
                            format!("fuse_call omits boundary result {result_leaf:?}")
                        })?;
                    self.transports.insert(*value, transport);
                    self.consume_value(*value)?;
                }
                BoundaryResultKind::State(token) => {
                    if !boundary.states.contains_key(&result_leaf) {
                        return Err(format!("fuse_call omits boundary state {result_leaf:?}"));
                    }
                    // Like ordinary invocation, the next state is a call-node
                    // output and is consumed exactly once by `consume_node_edges`.
                    let _ = token;
                }
                BoundaryResultKind::Storage { storage, .. } => {
                    // The occurrence-owned result storage of an inlined call
                    // is not allocated by the fused alternative: the boundary
                    // result transport names where the value actually lives.
                    // Override the call node's paired output value so later
                    // uses bind the caller's storage.
                    let transport =
                        boundary.results.get(&result_leaf).cloned().ok_or_else(|| {
                            format!("fuse_call omits boundary result {result_leaf:?}")
                        })?;
                    for output in logical.outputs.iter() {
                        if let Some(view) = output.view {
                            if let Some(logical_view) = self.graph.views.get(view) {
                                if logical_view.storage == *storage {
                                    self.transports.insert(output.id, transport.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
        self.consume_node_edges(&call)?;
        self.pending_nodes.remove(&call);

        // Import the child graph beneath the exact caller occurrence. Node ids
        // are only region-local, so a bare `FusedCall(call.node)` would alias
        // calls with the same node id in different caller regions.
        let mut prefix = call.region.clone();
        prefix.push(RegionStep::FusedCall(call.node));
        let remap = self.child_remap(child_graph);
        let mut imported = Vec::new();
        self.import_region(
            child_graph,
            &child_graph.root,
            Vec::new(),
            &prefix,
            &remap,
            &mut imported,
        )?;
        // Merge remapped storages/views into this graph's lookup tables and
        // allocate physical homes for imported owned storages.
        self.merge_child_tables(child_graph, &remap)?;
        // Register the imported regions and their pending obligations.
        // The child's root region is the exact region-qualified fused-call
        // prefix (nested regions are snapshotted before their parent).
        let first_imported = self.regions.len();
        for snapshot in imported {
            self.regions.push(snapshot);
        }
        let root_index = self
            .regions
            .iter()
            .skip(first_imported)
            .position(|region| region.path == prefix)
            .map(|index| index + first_imported)
            .ok_or_else(|| "the fused child root region is absent".to_string())?;
        for index in first_imported..self.regions.len() {
            self.register_region(index)?;
        }
        // Boundary substitution first: child root parameters adopt the
        // caller transports; child root results adopt the caller result
        // transports and are consumed (the fused alternative owns them now).
        self.substitute_child_boundary(root_index, &call_node, &boundary)?;
        // Default transports for imported node outputs (a substituted
        // boundary value keeps its caller transport).
        for index in first_imported..self.regions.len() {
            let region = self.regions[index].clone();
            for node in region.nodes.values() {
                for output in node.outputs.iter() {
                    if self.transports.contains_key(&output.id) {
                        continue;
                    }
                    let transport = self.produced_transport(output, node)?;
                    self.transports.insert(output.id, transport);
                }
            }
        }
        Ok(FusedImport {
            root: prefix,
            value_offset: remap.value,
            state_offset: remap.state,
            storage_offset: remap.storage,
        })
    }

    /// Fresh id offsets for a cross-call import: child ids are shifted past
    /// every id this alternative already uses.
    fn child_remap(&self, child_graph: &TaskGraph) -> ChildRemap {
        let mut max_value = 0u32;
        let mut max_state = 0u32;
        for region in &self.regions {
            for parameter in &region.parameters {
                match parameter {
                    RegionParameter::Value { id, .. } => max_value = max_value.max(id.0 + 1),
                    RegionParameter::State { id, .. } => max_state = max_state.max(id.0 + 1),
                }
            }
            for result in &region.results {
                match result {
                    RegionResult::Value { id, .. } => max_value = max_value.max(id.0 + 1),
                    RegionResult::State { id, .. } => max_state = max_state.max(id.0 + 1),
                }
            }
            for node in region.nodes.values() {
                for input in node.inputs.iter() {
                    max_value = max_value.max(input.0 + 1);
                }
                for token in node.state_inputs.iter() {
                    max_state = max_state.max(token.0 + 1);
                }
                for output in node.outputs.iter() {
                    max_value = max_value.max(output.id.0 + 1);
                }
                for token in node.state_outputs.iter() {
                    max_state = max_state.max(token.id.0 + 1);
                }
                match &node.kind {
                    LogicalNodeKind::If(if_node) => {
                        max_value = max_value.max(if_node.condition.0 + 1);
                        for join in &if_node.joins {
                            if let seismic_lang::logical::JoinSlot::Value { joined, .. } = join {
                                max_value = max_value.max(joined.0 + 1);
                            }
                        }
                    }
                    LogicalNodeKind::Loop(loop_node) => {
                        max_value = max_value
                            .max(loop_node.range.start.0 + 1)
                            .max(loop_node.range.end.0 + 1)
                            .max(loop_node.binder.0 + 1);
                        for value in loop_node
                            .invariant_values
                            .iter()
                            .chain(&loop_node.initial_values)
                        {
                            max_value = max_value.max(value.0 + 1);
                        }
                        for token in loop_node.initial_states.iter() {
                            max_state = max_state.max(token.0 + 1);
                        }
                    }
                    LogicalNodeKind::Reduction(reduction) => {
                        max_value = max_value.max(reduction.operand.0 + 1);
                    }
                    LogicalNodeKind::Call(call_node) => {
                        for input in &call_node.boundary_inputs {
                            match input.kind {
                                BoundaryInputKind::Value(value) => {
                                    max_value = max_value.max(value.0 + 1)
                                }
                                BoundaryInputKind::Shared {
                                    value,
                                    state: token,
                                }
                                | BoundaryInputKind::Exclusive {
                                    value,
                                    state: token,
                                }
                                | BoundaryInputKind::Move {
                                    value,
                                    state: token,
                                } => {
                                    max_value = max_value.max(value.0 + 1);
                                    max_state = max_state.max(token.0 + 1)
                                }
                            }
                        }
                        for result in &call_node.boundary_results {
                            match &result.kind {
                                BoundaryResultKind::Value(value) => {
                                    max_value = max_value.max(value.0 + 1)
                                }
                                BoundaryResultKind::Storage { token, .. }
                                | BoundaryResultKind::State(token) => {
                                    max_state = max_state.max(token.0 + 1)
                                }
                            }
                        }
                    }
                    LogicalNodeKind::Primitive(_) => {}
                }
                for obligation in node.safety.iter() {
                    for value in obligation.values() {
                        max_value = max_value.max(value.0 + 1);
                    }
                }
            }
        }
        let max_storage = self
            .graph
            .storages
            .ids()
            .map(|id| id.0 + 1)
            .max()
            .unwrap_or(0);
        let max_view = self.graph.views.ids().map(|id| id.0 + 1).max().unwrap_or(0);
        let _ = child_graph;
        ChildRemap {
            value: max_value,
            state: max_state,
            storage: max_storage,
            view: max_view,
        }
    }

    /// Rewrite one child region (and its nested regions) under `prefix`,
    /// collecting snapshots with fresh value/state ids; node ids stay
    /// region-local under the fresh region paths.
    fn import_region(
        &self,
        child_graph: &TaskGraph,
        region: &seismic_lang::logical::GraphRegion,
        child_path: RegionPath,
        prefix: &RegionPath,
        remap: &ChildRemap,
        imported: &mut Vec<RegionSnapshot>,
    ) -> Result<(), BuilderError> {
        let _ = child_graph;
        // Nested regions first, so their snapshots exist regardless of order.
        for (node_id, node) in region.nodes.ids().zip(region.nodes.iter()) {
            match &node.kind {
                LogicalNodeKind::If(if_node) => {
                    let mut then_path = child_path.clone();
                    then_path.push(RegionStep::IfThen(node_id));
                    self.import_region(
                        child_graph,
                        &if_node.then_region,
                        then_path,
                        prefix,
                        remap,
                        imported,
                    )?;
                    let mut else_path = child_path.clone();
                    else_path.push(RegionStep::IfElse(node_id));
                    self.import_region(
                        child_graph,
                        &if_node.else_region,
                        else_path,
                        prefix,
                        remap,
                        imported,
                    )?;
                }
                LogicalNodeKind::Loop(loop_node) => {
                    let mut body_path = child_path.clone();
                    body_path.push(RegionStep::LoopBody(node_id));
                    self.import_region(
                        child_graph,
                        &loop_node.body,
                        body_path,
                        prefix,
                        remap,
                        imported,
                    )?;
                }
                _ => {}
            }
        }
        let rewritten = self.rewrite_region(region, remap);
        let mut new_path = prefix.clone();
        new_path.extend(child_path.iter().copied());
        let mut nodes = BTreeMap::new();
        for (node_id, node) in rewritten.nodes.ids().zip(rewritten.nodes.iter()) {
            nodes.insert(node_id, node.clone());
        }
        imported.push(RegionSnapshot {
            path: new_path,
            parameters: rewritten.parameters.clone(),
            nodes,
            results: rewritten.results.clone(),
        });
        Ok(())
    }

    /// Deep-rewrite one child region with fresh ids (nested regions included).
    fn rewrite_region(
        &self,
        region: &seismic_lang::logical::GraphRegion,
        remap: &ChildRemap,
    ) -> seismic_lang::logical::GraphRegion {
        seismic_lang::logical::GraphRegion {
            parameters: region
                .parameters
                .iter()
                .map(|p| remap.parameter(p))
                .collect(),
            nodes: IdVec::from_iter(
                region
                    .nodes
                    .ids()
                    .zip(region.nodes.iter())
                    .map(|(id, node)| (id, remap.node(node))),
            ),
            results: region.results.iter().map(|r| remap.result(r)).collect(),
        }
    }

    /// Merge the remapped child storages/views into this graph's lookup
    /// tables and allocate physical homes for imported owned storages.
    fn merge_child_tables(
        &mut self,
        child_graph: &TaskGraph,
        remap: &ChildRemap,
    ) -> Result<(), BuilderError> {
        let mut storages: Vec<(LogicalStorageId, seismic_lang::logical::LogicalStorage)> = self
            .graph
            .storages
            .ids()
            .zip(self.graph.storages.iter())
            .map(|(id, s)| (id, s.clone()))
            .collect();
        let mut views: Vec<(
            seismic_lang::logical::LogicalViewId,
            seismic_lang::logical::LogicalView,
        )> = self
            .graph
            .views
            .ids()
            .zip(self.graph.views.iter())
            .map(|(id, v)| (id, v.clone()))
            .collect();
        for (id, storage) in child_graph.storages.ids().zip(child_graph.storages.iter()) {
            let new_id = LogicalStorageId(id.0 + remap.storage);
            let mut storage = storage.clone();
            if let seismic_lang::logical::StorageOrigin::Parameter {
                ordinal,
                path,
                name,
            } = &mut storage.origin
            {
                let _ = (ordinal, name);
                let _ = path;
            }
            storages.push((new_id, storage.clone()));
            if matches!(storage.origin, seismic_lang::logical::StorageOrigin::Owned) {
                // An owned storage of the inlined interval becomes this
                // alternative's own arena storage.
                let bytes = storage_bytes(&storage.shape)?;
                let alignment = alignment_of(&storage.shape);
                let mut active = ActivationLiteral::default();
                active.0.insert(ActivationTerm {
                    choice: self.choice,
                    logical_alternative: self.logical_alternative,
                    physical_alternative: self.physical_alternative,
                });
                let template =
                    self.shared
                        .borrow_mut()
                        .insert_storage(ConditionalStorageTemplate {
                            active_if: active,
                            scope: StorageScope::DeviceArena,
                            bytes,
                            alignment,
                            replication: Replication::Once,
                            layout: D::internal_layout(&storage.shape),
                            lifetime: StorageLifetime::Always,
                        });
                self.created_storage_templates.push(template);
                self.storage_templates.insert(new_id, template);
            }
        }
        for (id, view) in child_graph.views.ids().zip(child_graph.views.iter()) {
            let new_id = seismic_lang::logical::LogicalViewId(id.0 + remap.view);
            let mut view = view.clone();
            view.storage = LogicalStorageId(view.storage.0 + remap.storage);
            views.push((new_id, view));
        }
        self.graph.storages = IdVec::from_iter(storages);
        self.graph.views = IdVec::from_iter(views);
        Ok(())
    }

    /// Boundary substitution: the imported child root parameters adopt the
    /// caller transports; imported root results adopt the caller result
    /// transports and are consumed by this alternative.
    fn substitute_child_boundary(
        &mut self,
        root_index: usize,
        call_node: &CallNode,
        boundary: &BoundaryTemplates,
    ) -> Result<(), BuilderError> {
        let root = self.regions[root_index].clone();
        // Interface order: one Value param for a value boundary input; a
        // Value+State pair for a tensor boundary input (shared/exclusive/
        // moved tensors are one semantic leaf and one state).
        let mut param_cursor = 0usize;
        let mut inputs = call_node.boundary_inputs.clone();
        inputs.sort_by_key(|input| input.param);
        for input in &inputs {
            let input_leaf = BoundaryLeaf::Input {
                param: input.param,
                leaf: input.path.clone(),
            };
            let parameters = &root.parameters;
            match input.kind {
                BoundaryInputKind::Value(_) => {
                    let Some(RegionParameter::Value { id, ty }) = parameters.get(param_cursor)
                    else {
                        return Err(format!(
                            "the fused child parameter#{param_cursor} disagrees with boundary input {input_leaf:?}"
                        ));
                    };
                    let transport =
                        boundary.inputs.get(&input_leaf).cloned().ok_or_else(|| {
                            format!("fuse_call omits boundary input {input_leaf:?}")
                        })?;
                    if matches!(ty, ValueType::Tensor(_))
                        && !matches!(
                            transport,
                            TransportTemplate::Storage(_) | TransportTemplate::Boundary(_)
                        )
                    {
                        return Err(format!(
                            "a tensor boundary input at {input_leaf:?} must transport through storage"
                        ));
                    }
                    self.transports.insert(*id, transport);
                    param_cursor += 1;
                }
                BoundaryInputKind::Shared { .. }
                | BoundaryInputKind::Exclusive { .. }
                | BoundaryInputKind::Move { .. } => {
                    let Some(RegionParameter::Value { id, ty }) = parameters.get(param_cursor)
                    else {
                        return Err(format!(
                            "the fused child parameter#{param_cursor} disagrees with boundary input {input_leaf:?}"
                        ));
                    };
                    if !matches!(ty, ValueType::Tensor(_)) {
                        return Err(format!(
                            "the tensor boundary input at {input_leaf:?} names a non-tensor child parameter"
                        ));
                    }
                    let transport =
                        boundary.inputs.get(&input_leaf).cloned().ok_or_else(|| {
                            format!("fuse_call omits boundary input {input_leaf:?}")
                        })?;
                    self.transports.insert(*id, transport);
                    param_cursor += 1;
                    let Some(RegionParameter::State { id: token, storage }) =
                        parameters.get(param_cursor)
                    else {
                        return Err(format!(
                            "the tensor boundary input at {input_leaf:?} has no paired child state parameter"
                        ));
                    };
                    match boundary.states.get(&input_leaf) {
                        Some(StateTransportTemplate::Storage(template)) => {
                            self.storage_templates.insert(*storage, *template);
                        }
                        Some(StateTransportTemplate::Boundary(_)) => {}
                        None => {
                            return Err(format!("fuse_call omits boundary state {input_leaf:?}"));
                        }
                    }
                    let _ = token;
                    param_cursor += 1;
                }
            }
        }
        if param_cursor != root.parameters.len() {
            return Err(format!(
                "the fused child has {} root parameters but the boundary substituted {param_cursor}",
                root.parameters.len()
            ));
        }
        // Results: ordinal-wise against the call's boundary results.
        let results = root.results.clone();
        for (ordinal, result) in results.iter().enumerate() {
            let leaf = call_node
                .boundary_results
                .get(ordinal)
                .map(|result| BoundaryLeaf::Result {
                    leaf: result.path.clone(),
                })
                .ok_or_else(|| {
                    format!("the fused child has more results than the call boundary ({ordinal})")
                })?;
            match result {
                RegionResult::Value { id, .. } => {
                    let transport = boundary
                        .results
                        .get(&leaf)
                        .cloned()
                        .ok_or_else(|| format!("fuse_call omits boundary result {leaf:?}"))?;
                    self.transports.insert(*id, transport);
                }
                RegionResult::State { storage, .. } => match boundary.states.get(&leaf) {
                    Some(StateTransportTemplate::Storage(template)) => {
                        self.storage_templates.insert(*storage, *template);
                    }
                    Some(StateTransportTemplate::Boundary(_)) => {}
                    None => return Err(format!("fuse_call omits boundary state {leaf:?}")),
                },
            }
            self.consume_region_result(root_index, ordinal)?;
        }
        Ok(())
    }

    /// The graph of this alternative (for strategy inspection).
    pub fn graph(&self) -> &TaskGraph {
        &self.graph
    }

    /// The logical storage backing one state token (its producer or region
    /// parameter origin), for boundary/carry construction.
    pub fn logical_storage_of_token(&self, token: StateTokenId) -> Option<LogicalStorageId> {
        for region in &self.regions {
            for parameter in &region.parameters {
                if let RegionParameter::State { id, storage } = parameter {
                    if *id == token {
                        return Some(*storage);
                    }
                }
            }
            for node in region.nodes.values() {
                for state in node.state_outputs.iter() {
                    if state.id == token {
                        return Some(state.storage);
                    }
                }
            }
        }
        None
    }

    /// The nodes of one region of this alternative (imported cross-call
    /// regions included), in node order. Strategies use this to map fused
    /// child intervals.
    pub fn region_nodes(
        &self,
        region: &RegionPath,
    ) -> Result<Vec<(NodeId, LogicalNode)>, BuilderError> {
        let found = self
            .regions
            .iter()
            .find(|candidate| &candidate.path == region)
            .ok_or_else(|| format!("region {region:?} is absent"))?;
        Ok(found
            .nodes
            .iter()
            .map(|(id, node)| (*id, node.clone()))
            .collect())
    }

    /// Remaining pending obligations, for strategies that must discharge them.
    pub fn pending_obligations(&self) -> Vec<ObligationRef> {
        self.pending_obligations.iter().cloned().collect()
    }

    /// Remaining pending nodes.
    pub fn pending_nodes(&self) -> Vec<NodeRef> {
        self.pending_nodes.keys().cloned().collect()
    }

    // -- consuming transitions ----------------------------------------------

    /// Map one primitive node to a launch with an exact iteration map and
    /// legalized opcodes. Consumes the node, its edges, and its input values.
    pub fn map_primitive(
        &mut self,
        node: NodeRef,
        iteration: crate::dispatch::LinearIterationMap,
        legalized: Legalized<D::Op>,
    ) -> Result<(), BuilderError> {
        let ops = legalized.ops().cloned().ok_or_else(|| {
            "a universal primitive mapping is inapplicable (compiler bug)".to_string()
        })?;
        self.check_pending_primitive(&node)?;
        self.launch(vec![node], iteration, ops.into_vec())
    }

    /// Map one reduction node with an exact strategy. The result publication
    /// transport is part of the strategy.
    pub fn map_reduction(
        &mut self,
        node: NodeRef,
        strategy: ReductionStrategyTemplate<D>,
    ) -> Result<(), BuilderError> {
        let ops = strategy
            .ops
            .ops()
            .cloned()
            .ok_or_else(|| "a reduction strategy is inapplicable (compiler bug)".to_string())?;
        self.check_pending(&node)?;
        let logical = self
            .pending_nodes
            .get(&node)
            .cloned()
            .ok_or_else(|| internal_pending())?;
        let reduction = match &logical.kind {
            LogicalNodeKind::Reduction(reduction) => reduction.clone(),
            _ => return Err("map_reduction requires a reduction node".into()),
        };
        // A reassociating topology of an ordered reduction carries the exact
        // `Reassociate` transfer (data-dependent; evidence-gated). A source
        // `unordered` reduction permits reassociation; parallel outer
        // coordinates with serial inner folds preserve the reference order.
        if !strategy.topology.is_reference_order()
            && reduction.order == seismic_lang::logical::ReductionOrder::Ascending
        {
            let transfer = NumericalTransfer::Reassociate {
                op: reduction.op,
                topology: strategy.topology.clone(),
            };
            self.numerical = crate::numerics::compose(&transfer, &self.numerical);
        }
        // The reduced result is published through the strategy transport.
        let result_value = logical
            .outputs
            .first()
            .map(|output| output.id)
            .ok_or_else(|| "a reduction has a result value".to_string())?;
        self.transports.insert(result_value, strategy.result);
        let operand_transport = self.transport_of(reduction.operand)?;
        let result_transport = self.transport_of(result_value)?;
        let bindings = vec![
            (reduction.operand, operand_transport),
            (result_value, result_transport),
        ];
        self.consume_value(reduction.operand)?;
        self.consume_node_edges(&node)?;
        self.pending_nodes.remove(&node);
        self.commit_launch(strategy.iteration, bindings, ops.into_vec())
    }

    /// Fuse a connected node set into one launch. Internal edges among the
    /// fused nodes become kernel-local SSA transports.
    pub fn fuse(
        &mut self,
        nodes: Vec<NodeRef>,
        strategy: FusedStrategyTemplate<D>,
    ) -> Result<(), BuilderError> {
        let ops = strategy
            .ops
            .ops()
            .cloned()
            .ok_or_else(|| "a fused strategy is inapplicable".to_string())?;
        if nodes.is_empty() {
            return Err("fuse requires a nonempty node set".into());
        }
        for node in &nodes {
            self.check_pending(node)?;
            match &self.pending_nodes.get(node).expect("checked").kind {
                LogicalNodeKind::Call(_) => {
                    return Err(
                        "a fused interval cannot contain a call; use invoke or fuse_call".into(),
                    );
                }
                LogicalNodeKind::Reduction(_) => {
                    return Err(
                        "a fused interval cannot contain a reduction; use map_reduction".into(),
                    );
                }
                _ => {}
            }
        }
        // Loops and ifs pull their nested nodes in; a nested call or
        // reduction is retained and must use the structured transitions.
        let mut flat: Vec<NodeRef> = nodes.clone();
        for node in &nodes {
            let logical = self
                .pending_nodes
                .get(node)
                .cloned()
                .ok_or_else(internal_pending)?;
            match &logical.kind {
                LogicalNodeKind::If(_) | LogicalNodeKind::Loop(_) => {
                    self.flatten_structural(node, &logical, &mut flat)?;
                }
                _ => {}
            }
        }
        // Consume the structural pending of every loop/if in the set.
        for node in flat.iter().collect::<Vec<_>>() {
            let logical = self
                .pending_nodes
                .get(node)
                .cloned()
                .ok_or_else(internal_pending)?;
            match &logical.kind {
                LogicalNodeKind::If(if_node) => {
                    self.consume_if_structure(node, if_node)?;
                }
                LogicalNodeKind::Loop(loop_node) => {
                    self.consume_loop_structure(node, loop_node)?;
                }
                _ => {}
            }
        }
        // Rewrite internal edges to kernel transports.
        let node_ids: BTreeSet<NodeRef> = flat.iter().cloned().collect();
        for node in &flat {
            let logical = self
                .pending_nodes
                .get(node)
                .cloned()
                .ok_or_else(internal_pending)?;
            for output in logical.outputs.iter() {
                let internal = self
                    .consumers
                    .get(&output.id)
                    .map(|consumers| {
                        consumers.iter().all(|consumer| match consumer {
                            Consumer::Node(node) => node_ids.contains(node),
                            Consumer::Structural => false,
                        })
                    })
                    .unwrap_or(true);
                if internal && output.view.is_none() {
                    let kernel_value = self.shared.borrow_mut().fresh_kernel_value();
                    self.transports
                        .insert(output.id, TransportTemplate::Kernel(kernel_value));
                }
            }
        }
        self.launch(flat, strategy.iteration, ops.into_vec())
    }

    /// Pull the nested nodes of one loop/if into a fused set.
    fn flatten_structural(
        &self,
        node: &NodeRef,
        logical: &LogicalNode,
        flat: &mut Vec<NodeRef>,
    ) -> Result<(), BuilderError> {
        let nested: Vec<(RegionPath, seismic_lang::logical::GraphRegion)> = match &logical.kind {
            LogicalNodeKind::If(if_node) => vec![
                (
                    {
                        let mut path = node.region.clone();
                        path.push(RegionStep::IfThen(node.node));
                        path
                    },
                    if_node.then_region.clone(),
                ),
                (
                    {
                        let mut path = node.region.clone();
                        path.push(RegionStep::IfElse(node.node));
                        path
                    },
                    if_node.else_region.clone(),
                ),
            ],
            LogicalNodeKind::Loop(loop_node) => vec![(
                {
                    let mut path = node.region.clone();
                    path.push(RegionStep::LoopBody(node.node));
                    path
                },
                loop_node.body.clone(),
            )],
            _ => return Ok(()),
        };
        for (path, region) in nested {
            for (nested_id, nested_node) in region.nodes.ids().zip(region.nodes.iter()) {
                let nested_ref = NodeRef {
                    region: path.clone(),
                    node: nested_id,
                };
                match &nested_node.kind {
                    LogicalNodeKind::Call(_) => {
                        return Err(
                            "a fused loop/if body retains a call; use schedule_loop or schedule_if"
                                .into(),
                        );
                    }
                    LogicalNodeKind::Reduction(_) => {
                        return Err(
                            "a fused loop/if body retains a reduction; use map_reduction".into(),
                        );
                    }
                    LogicalNodeKind::If(_) | LogicalNodeKind::Loop(_) => {
                        self.flatten_structural(&nested_ref, nested_node, flat)?;
                    }
                    _ => {}
                }
                if !flat.contains(&nested_ref) {
                    flat.push(nested_ref);
                }
            }
        }
        Ok(())
    }

    /// Consume the structural pending of one fused if node.
    fn consume_if_structure(
        &mut self,
        node: &NodeRef,
        if_node: &seismic_lang::logical::IfNode,
    ) -> Result<(), BuilderError> {
        self.consume_value(if_node.condition)?;
        let then_index = self.region_index(&{
            let mut path = node.region.clone();
            path.push(RegionStep::IfThen(node.node));
            path
        })?;
        let else_index = self.region_index(&{
            let mut path = node.region.clone();
            path.push(RegionStep::IfElse(node.node));
            path
        })?;
        // Branch parameters adopt their captured outer values' transports.
        for capture in &if_node.captured {
            let transport = self.transport_of(capture.outer)?;
            self.transports.insert(capture.parameter, transport);
            self.pending_values.remove(&capture.outer);
        }
        self.consume_region_inputs(then_index)?;
        self.consume_region_inputs(else_index)?;
        for join in &if_node.joins {
            match join {
                seismic_lang::logical::JoinSlot::Value {
                    then_result,
                    else_result,
                    joined,
                    ..
                } => {
                    let then_value = match self.regions[then_index].results.get(then_result.index())
                    {
                        Some(RegionResult::Value { id, .. }) => *id,
                        _ => return Err("a fused join names a non-value then result".into()),
                    };
                    self.consume_region_result(then_index, then_result.index())?;
                    self.consume_region_result(else_index, else_result.index())?;
                    let joined_transport = self.transport_of(then_value)?;
                    self.transports.insert(*joined, joined_transport);
                    self.pending_values.remove(joined);
                }
                seismic_lang::logical::JoinSlot::State {
                    then_result,
                    else_result,
                    joined,
                    ..
                } => {
                    self.consume_region_result(then_index, then_result.index())?;
                    self.consume_region_result(else_index, else_result.index())?;
                    if !self.pending_state_edges.remove(&StateEdge::JoinState {
                        region: self.region_index(&node.region)?,
                        node: node.node,
                        token: *joined,
                    }) {
                        return Err("a fused state join names an absent state edge".into());
                    }
                    self.consume_parameter_state(*joined);
                }
            }
        }
        for (index, ordinal) in [(then_index, 0), (else_index, 0)] {
            let _ = ordinal;
            while let Some((_, ordinal)) = self
                .pending_results
                .iter()
                .find(|(region, _)| *region == index)
                .copied()
            {
                self.consume_region_result(index, ordinal)?;
            }
        }
        Ok(())
    }

    /// Consume the structural pending of one fused loop node.
    fn consume_loop_structure(
        &mut self,
        node: &NodeRef,
        loop_node: &seismic_lang::logical::LoopNode,
    ) -> Result<(), BuilderError> {
        self.consume_value(loop_node.range.start)?;
        self.consume_value(loop_node.range.end)?;
        for token in loop_node.initial_states.iter() {
            if !self.pending_state_edges.remove(&StateEdge::LoopInitial {
                region: self.region_index(&node.region)?,
                node: node.node,
                token: *token,
            }) {
                return Err("a fused loop initial state edge is absent".into());
            }
            self.consume_parameter_state(*token);
        }
        let body_index = self.region_index(&{
            let mut path = node.region.clone();
            path.push(RegionStep::LoopBody(node.node));
            path
        })?;
        if self.regions[body_index].results.len() != loop_node.body.results.len() {
            return Err(format!(
                "loop node {:?} resolved body {:?} with {} snapshotted results but owns {} logical results",
                node,
                self.regions[body_index].path,
                self.regions[body_index].results.len(),
                loop_node.body.results.len()
            ));
        }
        self.consume_region_inputs(body_index)?;
        // The binder is kernel-local control state.
        let binder_value = self.shared.borrow_mut().fresh_kernel_value();
        self.transports
            .insert(loop_node.binder, TransportTemplate::Kernel(binder_value));
        // Carried body parameters adopt their initial transports; invariant
        // copies adopt the outer invariant transports.
        let carried_indices: BTreeSet<usize> = loop_node
            .carried
            .iter()
            .map(|slot| slot.body_parameter.index())
            .collect();
        let mut invariant_ordinal = 0usize;
        for (param_index, parameter) in self.regions[body_index].parameters.iter().enumerate() {
            if let RegionParameter::Value { id, .. } = parameter {
                if *id == loop_node.binder || carried_indices.contains(&param_index) {
                    continue;
                }
                let outer = loop_node
                    .invariant_values
                    .get(invariant_ordinal)
                    .copied()
                    .ok_or_else(|| "a fused body parameter has no invariant value".to_string())?;
                let transport = self.transport_of(outer)?;
                self.transports.insert(*id, transport);
                invariant_ordinal += 1;
            }
        }
        for invariant in loop_node.invariant_values.iter() {
            self.pending_values.remove(invariant);
            if let Some(parameter) = self.parameter_values.get(invariant).copied() {
                self.pending_inputs.remove(&parameter);
            }
        }
        for slot in loop_node.carried.iter() {
            match slot.initial {
                seismic_lang::logical::RegionInput::Value(initial) => {
                    let initial_transport = self.transport_of(initial)?;
                    self.pending_values.remove(&initial);
                    if let Some(parameter) = self.regions[body_index]
                        .parameters
                        .get(slot.body_parameter.index())
                    {
                        if let RegionParameter::Value { id, .. } = parameter {
                            self.transports.insert(*id, initial_transport);
                        }
                    }
                }
                seismic_lang::logical::RegionInput::State(token) => {
                    self.consume_parameter_state(token);
                }
            }
        }
        while let Some((_, ordinal)) = self
            .pending_results
            .iter()
            .find(|(region, _)| *region == body_index)
            .copied()
        {
            self.consume_region_result(body_index, ordinal)?;
        }
        Ok(())
    }

    /// Replace logical edges with explicit publication: each cut value's
    /// transport is set to the supplied transport (retained storage or a
    /// planned executor-scalar slot), with retained lifetime and consumer
    /// binding through the schedule order.
    pub fn split(
        &mut self,
        region: RegionPath,
        cuts: Vec<GraphValueId>,
        transports: Vec<TransportTemplate>,
    ) -> Result<(), BuilderError> {
        if cuts.len() != transports.len() {
            return Err("split needs one transport per cut edge".into());
        }
        let _ = self.region_index(&region)?;
        for (cut, transport) in cuts.into_iter().zip(transports) {
            if !self.pending_values.contains(&cut) {
                return Err(format!(
                    "split cut value#{} is absent or already consumed",
                    cut.0
                ));
            }
            match &transport {
                TransportTemplate::Storage(_)
                | TransportTemplate::ExecutorScalar(_)
                | TransportTemplate::Void => {}
                _ => {
                    return Err(
                        "a split publication must be retained storage, an executor scalar slot, or void"
                            .into(),
                    )
                }
            }
            self.transports.insert(cut, transport);
        }
        Ok(())
    }

    /// Consume one `if` node as a structured executor `If` step: exactly one
    /// retained predicate, exactly one branch per visit, explicit joins. The
    /// branch schedules are built with this same builder inside the closures.
    pub fn schedule_if(
        &mut self,
        node: NodeRef,
        predicate: ExecutorPredicateTemplate,
        then_schedule: impl FnOnce(&mut Self) -> Result<(), BuilderError>,
        else_schedule: impl FnOnce(&mut Self) -> Result<(), BuilderError>,
        joins: Vec<PhysicalJoinTemplate>,
    ) -> Result<(), BuilderError> {
        self.check_pending(&node)?;
        let logical = self
            .pending_nodes
            .get(&node)
            .cloned()
            .ok_or_else(internal_pending)?;
        let if_node = match &logical.kind {
            LogicalNodeKind::If(if_node) => if_node.clone(),
            _ => return Err("schedule_if requires an if node".into()),
        };
        self.consume_value(if_node.condition)?;
        self.consume_node_edges(&node)?;
        self.pending_nodes.remove(&node);

        let then_index = self.region_index(&{
            let mut path = node.region.clone();
            path.push(RegionStep::IfThen(node.node));
            path
        })?;
        let else_index = self.region_index(&{
            let mut path = node.region.clone();
            path.push(RegionStep::IfElse(node.node));
            path
        })?;
        // Branch parameters adopt their captured outer values' transports.
        for capture in &if_node.captured {
            let transport = self.transport_of(capture.outer)?;
            self.transports.insert(capture.parameter, transport);
            self.pending_values.remove(&capture.outer);
        }
        self.consume_region_inputs(then_index)?;
        self.consume_region_inputs(else_index)?;

        self.scopes.push(Scope { steps: Vec::new() });
        then_schedule(self)?;
        let then_steps = self.pop_scope("the then branch has no step")?;
        self.scopes.push(Scope { steps: Vec::new() });
        else_schedule(self)?;
        let else_steps = self.pop_scope("the else branch has no step")?;

        if joins.len() != if_node.joins.len() {
            return Err("schedule_if joins must match the logical join slots".into());
        }
        for (slot, join) in if_node.joins.iter().zip(&joins) {
            match (slot, join) {
                (
                    seismic_lang::logical::JoinSlot::Value {
                        then_result,
                        else_result,
                        joined,
                        ..
                    },
                    PhysicalJoinTemplate::Value {
                        joined: joined_transport,
                        ..
                    },
                ) => {
                    self.consume_region_result(then_index, then_result.index())?;
                    self.consume_region_result(else_index, else_result.index())?;
                    self.transports.insert(*joined, joined_transport.clone());
                    self.consume_value(*joined)?;
                }
                (
                    seismic_lang::logical::JoinSlot::State {
                        then_result,
                        else_result,
                        joined,
                        storage,
                        ..
                    },
                    PhysicalJoinTemplate::State {
                        storage: join_storage,
                    },
                ) => {
                    self.consume_region_result(then_index, then_result.index())?;
                    self.consume_region_result(else_index, else_result.index())?;
                    if self.storage_templates.get(storage) != Some(join_storage) {
                        return Err("a state join must name the joined storage's template".into());
                    }
                    if !self.pending_state_edges.remove(&StateEdge::JoinState {
                        region: self.region_index(&node.region)?,
                        node: node.node,
                        token: *joined,
                    }) {
                        return Err("a state join names an absent state edge".into());
                    }
                    self.consume_parameter_state(*joined);
                }
                _ => {
                    return Err(
                        "schedule_if join kinds disagree with the logical join slots".into(),
                    );
                }
            }
        }

        let step = ScheduleStepTemplate::If(ScheduleIfTemplate {
            condition: predicate,
            then_schedule: ScheduleTemplate { steps: then_steps },
            else_schedule: ScheduleTemplate { steps: else_steps },
            joins,
        });
        self.append_step(step)
    }

    /// Consume one loop node as a structured executor `Repeat` step: a
    /// retained half-open ascending range, a rebound scalar binder, and
    /// carried transports rebound each visit. The body schedule is built
    /// with this same builder inside the closure.
    pub fn schedule_loop(
        &mut self,
        node: NodeRef,
        executor_range: ExecutorRangeTemplate,
        carried: Vec<PhysicalCarryTemplate>,
        body: impl FnOnce(&mut Self) -> Result<(), BuilderError>,
    ) -> Result<(), BuilderError> {
        self.check_pending(&node)?;
        let logical = self
            .pending_nodes
            .get(&node)
            .cloned()
            .ok_or_else(internal_pending)?;
        let loop_node = match &logical.kind {
            LogicalNodeKind::Loop(loop_node) => loop_node.clone(),
            _ => return Err("schedule_loop requires a loop node".into()),
        };
        self.consume_value(loop_node.range.start)?;
        self.consume_value(loop_node.range.end)?;
        self.consume_node_edges(&node)?;
        for token in loop_node.initial_states.iter() {
            if !self.pending_state_edges.remove(&StateEdge::LoopInitial {
                region: self.region_index(&node.region)?,
                node: node.node,
                token: *token,
            }) {
                return Err("a loop initial state edge is absent".into());
            }
            self.consume_parameter_state(*token);
        }
        self.pending_nodes.remove(&node);

        if carried.len() != loop_node.carried.len() {
            return Err("schedule_loop carries must match the logical carried slots".into());
        }
        let body_index = self.region_index(&{
            let mut path = node.region.clone();
            path.push(RegionStep::LoopBody(node.node));
            path
        })?;
        self.consume_region_inputs(body_index)?;

        let binder_slot = {
            let name = format!("binder-{}", self.shared.borrow().next_slot);
            self.shared.borrow_mut().fresh_slot(DType::I32, name)
        };
        // Bind every body region parameter: the binder (rebound per visit),
        // the invariant copies (transports of the outer values), and the
        // carried slots (rebound transports).
        let binder_param_index = self.regions[body_index]
            .parameters
            .iter()
            .position(|parameter| {
                matches!(parameter, RegionParameter::Value { id, .. } if *id == loop_node.binder)
            })
            .ok_or_else(|| "the loop binder is not a body parameter".to_string())?;
        let carried_param_indices: BTreeSet<usize> = loop_node
            .carried
            .iter()
            .map(|slot| slot.body_parameter.index())
            .collect();
        self.transports.insert(
            loop_node.binder,
            TransportTemplate::ExecutorScalar(ExecutorScalarTemplate {
                source: ExecutorScalarSource::Slot(binder_slot),
                dtype: DType::I32,
            }),
        );
        let mut invariant_ordinal = 0usize;
        for (param_index, parameter) in self.regions[body_index].parameters.iter().enumerate() {
            if param_index == binder_param_index || carried_param_indices.contains(&param_index) {
                continue;
            }
            if let RegionParameter::Value { id, ty } = parameter {
                // One copied invariant per remaining Value parameter, in the
                // loop's invariant order.
                let outer = loop_node
                    .invariant_values
                    .get(invariant_ordinal)
                    .copied()
                    .ok_or_else(|| {
                        format!("body parameter#{param_index} ({ty}) has no invariant value")
                    })?;
                let transport = self.transport_of(outer)?;
                self.transports.insert(*id, transport);
                invariant_ordinal += 1;
            }
        }
        // The outer invariant values are consumed by the loop. A parameter
        // view value may already have been consumed together with its state
        // token (Value/State parameter pair); removal is idempotent.
        for invariant in loop_node.invariant_values.iter() {
            self.pending_values.remove(invariant);
            if let Some(parameter) = self.parameter_values.get(invariant).copied() {
                self.pending_inputs.remove(&parameter);
            }
        }
        let mut value_carry_ordinal = 0usize;
        for (slot, carry) in loop_node.carried.iter().zip(&carried) {
            if slot.body_result.index() >= self.regions[body_index].results.len() {
                return Err(format!(
                    "loop node {:?} carry names body result#{} but body {:?} has {} results ({} carries)",
                    node,
                    slot.body_result.0,
                    self.regions[body_index].path,
                    self.regions[body_index].results.len(),
                    loop_node.carried.len()
                ));
            }
            match &carry.transport {
                TransportTemplate::ExecutorScalar(_)
                | TransportTemplate::Storage(_)
                | TransportTemplate::Void
                | TransportTemplate::Boundary(_) => {}
                _ => return Err("a carried transport must survive across visits".into()),
            }
            match slot.initial {
                seismic_lang::logical::RegionInput::Value(initial) => {
                    self.consume_value(initial)?;
                    let parameter = self.regions[body_index]
                        .parameters
                        .get(slot.body_parameter.index())
                        .cloned()
                        .ok_or_else(|| "a carried body parameter is absent".to_string())?;
                    if let RegionParameter::Value { id, .. } = parameter {
                        self.transports.insert(id, carry.transport.clone());
                    }
                    // The loop's exit value transports as the final carry.
                    if let Some(exit) = logical.outputs.get(value_carry_ordinal) {
                        self.transports.insert(exit.id, carry.transport.clone());
                    }
                    value_carry_ordinal += 1;
                }
                seismic_lang::logical::RegionInput::State(_) => {
                    // The carried storage's state leaves the body per visit.
                }
            }
            self.consume_region_result(body_index, slot.body_result.index())?;
        }
        // Remaining body region results are the loop's per-visit states
        // (disjoint-write or atomic joins, pass-throughs): consumed by the
        // loop as a whole.
        for ordinal in 0..self.regions[body_index].results.len() {
            if self.pending_results.contains(&(body_index, ordinal)) {
                self.consume_region_result(body_index, ordinal)?;
            }
        }

        self.scopes.push(Scope { steps: Vec::new() });
        body(self)?;
        let body_steps = self.pop_scope("a loop body has no step")?;

        let step = ScheduleStepTemplate::Repeat(ScheduleRepeatTemplate {
            logical_kind: loop_node.kind,
            range: executor_range,
            binder: ExecutorScalarSlot {
                slot: binder_slot,
                dtype: DType::I32,
            },
            body: ScheduleTemplate { steps: body_steps },
            carried,
        });
        self.append_step(step)
    }

    /// Consume one call occurrence as a nested plan invocation. The boundary
    /// templates transport each canonical leaf directly to caller storage,
    /// slots, or placeholders; the child allocates nothing for the boundary.
    pub fn invoke_canonical(&mut self, call: NodeRef) -> Result<(), BuilderError> {
        self.check_pending(&call)?;
        let logical = self.pending_nodes.get(&call).ok_or_else(internal_pending)?;
        let call_node = match &logical.kind {
            LogicalNodeKind::Call(call_node) => call_node.clone(),
            _ => return Err("invoke_canonical requires a call node".into()),
        };
        let boundary = self.canonical_call_boundary(&call_node)?;
        self.invoke(call, boundary)
    }

    /// Consume one call occurrence with an explicitly substituted boundary.
    /// This is used by cross-call fusion; ordinary backend call lowering must
    /// use [`Self::invoke_canonical`] so boundary semantics have one owner.
    pub fn invoke(
        &mut self,
        call: NodeRef,
        boundary: BoundaryTemplates,
    ) -> Result<(), BuilderError> {
        self.check_pending(&call)?;
        let logical = self
            .pending_nodes
            .get(&call)
            .cloned()
            .ok_or_else(internal_pending)?;
        let call_node = match &logical.kind {
            LogicalNodeKind::Call(call_node) => call_node.clone(),
            _ => return Err("invoke requires a call node".into()),
        };
        let call_region = self.region_index(&call.region)?;
        let mut inputs = BTreeMap::new();
        for input in &call_node.boundary_inputs {
            let input_leaf = BoundaryLeaf::Input {
                param: input.param,
                leaf: input.path.clone(),
            };
            match input.kind {
                BoundaryInputKind::Value(value) => {
                    let transport = boundary
                        .inputs
                        .get(&input_leaf)
                        .cloned()
                        .ok_or_else(|| format!("invoke omits boundary input {input_leaf:?}"))?;
                    self.transports.insert(value, transport.clone());
                    self.consume_value(value)?;
                    inputs.insert(input_leaf, transport);
                }
                BoundaryInputKind::Shared {
                    value,
                    state: token,
                }
                | BoundaryInputKind::Exclusive {
                    value,
                    state: token,
                }
                | BoundaryInputKind::Move {
                    value,
                    state: token,
                } => {
                    if !boundary.states.contains_key(&input_leaf) {
                        return Err(format!("invoke omits boundary state {input_leaf:?}"));
                    }
                    // A tensor boundary parameter also transports its view
                    // value to the child (the child's Value parameter
                    // resolves through the environment); the caller-supplied
                    // transport is recorded in the call boundary.
                    let view_transport = boundary
                        .inputs
                        .get(&input_leaf)
                        .cloned()
                        .ok_or_else(|| {
                            format!(
                                "invoke omits boundary input {input_leaf:?} for the child's view value"
                            )
                        })?;
                    inputs.insert(input_leaf, view_transport);
                    self.consume_value(value)?;
                    if !self.pending_state_edges.remove(&StateEdge::CallBoundary {
                        region: call_region,
                        node: call.node,
                        token,
                    }) {
                        return Err("a call boundary state edge is absent".into());
                    }
                    self.consume_parameter_state(token);
                }
            }
        }
        let mut results = BTreeMap::new();
        for result in &call_node.boundary_results {
            let result_leaf = BoundaryLeaf::Result {
                leaf: result.path.clone(),
            };
            match &result.kind {
                BoundaryResultKind::Value(value) => {
                    let transport =
                        boundary.results.get(&result_leaf).cloned().ok_or_else(|| {
                            format!("invoke omits boundary result {result_leaf:?}")
                        })?;
                    self.transports.insert(*value, transport.clone());
                    self.consume_value(*value)?;
                    results.insert(result_leaf, transport);
                }
                BoundaryResultKind::Storage { storage, token, .. } => {
                    let transport =
                        boundary.results.get(&result_leaf).cloned().ok_or_else(|| {
                            format!("invoke omits boundary result {result_leaf:?}")
                        })?;
                    match &transport {
                        TransportTemplate::Storage(views) => {
                            let view_storage = views.first().storage;
                            if let Some(template) = self.storage_templates.get(storage) {
                                if *template != view_storage && self.is_entry {
                                    // The entry materializes the occurrence-owned result; a
                                    // child may route it to caller storage directly.
                                    return Err(
                                        "a call result storage must bind its occurrence-owned template"
                                            .into(),
                                    );
                                }
                            }
                        }
                        TransportTemplate::Boundary(_) => {}
                        _ => {
                            return Err(
                                "a tensor call result must transport through storage".into()
                            );
                        }
                    }
                    // The result storage's state is one of the call node's
                    // own outputs, consumed with the node's edges below.
                    let _ = token;
                    results.insert(result_leaf, transport);
                }
                BoundaryResultKind::State(token) => {
                    if !boundary.states.contains_key(&result_leaf) {
                        return Err(format!("invoke omits boundary state {result_leaf:?}"));
                    }
                    // The next state of an inout storage is a call node
                    // output, consumed with the node's edges below.
                    let _ = token;
                }
            }
        }
        self.consume_node_edges(&call)?;
        self.pending_nodes.remove(&call);
        let step = ScheduleStepTemplate::Call(CallTemplate {
            call,
            choice: call_node.choice,
            boundary: BoundaryTemplates {
                inputs,
                results,
                states: boundary.states,
            },
        });
        self.append_step(step)
    }

    /// Consume one safety obligation: statically proved, or runtime checked
    /// with a planned predicate, inactive behavior, and a status field.
    pub fn discharge(
        &mut self,
        obligation: ObligationRef,
        disposition: ObligationDisposition<D>,
    ) -> Result<DispositionReceipt, BuilderError> {
        if !self.pending_obligations.remove(&obligation) {
            return Err(format!(
                "obligation node#{} index#{} is absent or already discharged",
                obligation.node.node.0, obligation.index
            ));
        }
        let (runtime_checked, predicate) = match disposition {
            ObligationDisposition::StaticallyProved { .. } => (None, None),
            ObligationDisposition::RuntimeChecked { predicate, .. } => {
                let field = StatusFieldId(u64::from(self.shared.borrow().next_status_field));
                self.shared.borrow_mut().next_status_field += 1;
                // An inapplicable predicate means the strategy plans the
                // check itself (its own opcode in the owning launch); real
                // predicate opcodes are retained for consumers.
                let ops = predicate.ops().cloned();
                (Some(field), ops)
            }
        };
        self.discharges.push(DischargedObligation {
            node: obligation.node,
            index: obligation.index,
            runtime_checked,
            predicate,
        });
        Ok(runtime_checked)
    }

    /// Complete one boundary output of the alternative with its transport.
    pub fn complete_result(
        &mut self,
        result: u32,
        transport: TransportTemplate,
    ) -> Result<(), BuilderError> {
        let root_index = self
            .region_index(&Vec::new())
            .expect("the root region exists");
        let boundary = self
            .graph
            .results
            .get(result as usize)
            .cloned()
            .ok_or_else(|| format!("boundary result#{result} is absent"))?;
        match &boundary {
            RegionResult::Value { id, ty } => {
                let compatible = match (&transport, ty) {
                    (TransportTemplate::Void, ValueType::Void) => true,
                    (TransportTemplate::ExecutorScalar(_), ValueType::Scalar(_))
                    | (TransportTemplate::ExecutorScalar(_), ValueType::Index { .. }) => true,
                    (TransportTemplate::Tuple(_), ValueType::Range { .. }) => true,
                    (TransportTemplate::Tuple(_), ValueType::Tuple(_)) => true,
                    (TransportTemplate::Storage(_), ValueType::Tensor(_)) => true,
                    (TransportTemplate::Boundary(_), ValueType::Tensor(_)) => true,
                    (TransportTemplate::Boundary(_), ValueType::Scalar(_)) => true,
                    (TransportTemplate::Boundary(_), ValueType::Index { .. }) => true,
                    (TransportTemplate::Boundary(_), ValueType::Range { .. }) => true,
                    (TransportTemplate::Boundary(_), ValueType::Tuple(_)) => true,
                    _ => false,
                };
                if !compatible {
                    return Err(format!(
                        "boundary result#{result} transport disagrees with its type {ty}"
                    ));
                }
                if let TransportTemplate::Storage(views) = &transport {
                    for view in views.iter() {
                        self.bound_result_templates.insert(view.storage);
                    }
                }
                self.transports.insert(*id, transport);
                // A boundary result value may already have been consumed
                // structurally (e.g. as a loop invariant): completing the
                // boundary is its terminal use, not a second use.
                self.pending_values.remove(id);
                if let Some(parameter) = self.parameter_values.get(id).copied() {
                    self.pending_inputs.remove(&parameter);
                }
            }
            RegionResult::State { id, .. } => {
                match &transport {
                    TransportTemplate::Storage(views) => {
                        let storage = views.first().storage;
                        let template_ok = self.storage_templates.values().any(|t| *t == storage);
                        if !template_ok {
                            return Err(
                                "a state boundary result must name a storage of this alternative"
                                    .into(),
                            );
                        }
                        self.bound_result_templates.insert(storage);
                    }
                    TransportTemplate::Boundary(_) => {}
                    _ => return Err("a state boundary result transports through storage".into()),
                }
                if !self.pending_state_edges.remove(&StateEdge::RegionState {
                    region: root_index,
                    ordinal: result as usize,
                    token: *id,
                }) {
                    return Err(format!("boundary state result#{result} is absent"));
                }
            }
        }
        if !self.pending_results.remove(&(root_index, result as usize)) {
            return Err(format!("boundary result#{result} is already completed"));
        }
        Ok(())
    }

    /// Finish this alternative. Available only when the logical and physical
    /// pending sets are empty; otherwise the transition is refused.
    pub fn finish_alternative(mut self) -> Result<PhysicalAlternative<D>, BuilderError> {
        if !self.pending_nodes.is_empty()
            || !self.pending_results.is_empty()
            || !self.pending_state_edges.is_empty()
            || !self.pending_obligations.is_empty()
            || !self.pending_inputs.is_empty()
            || !self.pending_values.is_empty()
        {
            return Err(format!(
                "alternative choice#{} logical#{} physical#{} has unconsumed obligations \
                 (nodes {}, results {}, state edges {}, obligations {}, inputs {}, values {})",
                self.choice.0,
                self.logical_alternative,
                self.physical_alternative,
                self.pending_nodes.len(),
                self.pending_results.len(),
                self.pending_state_edges.len(),
                self.pending_obligations.len(),
                self.pending_inputs.len(),
                self.pending_values.len(),
            ));
        }
        if self.scopes.len() != 1 {
            return Err("an alternative finishes with an open schedule scope".into());
        }
        let steps = NonEmpty::new(self.scopes.remove(0).steps)
            .ok_or_else(|| "an alternative has no schedule step".to_string())?;
        // Reference-based activation: an arena-scoped template
        // this alternative created but never references — in its schedule
        // (launches, calls, joins, carries) or completed results — activates
        // no storage and is dropped from the family table.
        let mut referenced = self.bound_result_templates.clone();
        for step in steps.iter() {
            referenced.extend(self.step_refs_of(step));
        }
        let dropped: Vec<PhysicalStorageTemplateId> = self
            .created_storage_templates
            .iter()
            .copied()
            .filter(|id| {
                !referenced.contains(id)
                    && matches!(
                        self.shared.borrow().storages.get(&id.0).map(|t| t.scope),
                        Some(StorageScope::DeviceArena)
                    )
            })
            .collect();
        {
            let mut shared = self.shared.borrow_mut();
            for id in &dropped {
                shared.storages.remove(&id.0);
            }
        }
        Ok(PhysicalAlternative {
            logical_alternative: self.logical_alternative,
            physical_alternative: self.physical_alternative,
            schedule: ScheduleTemplate { steps },
            obligations: ConsumedObligations {
                discharged: self.discharges,
                declared_status_fields: self.declared_status_fields.clone(),
            },
            cost: self.cost,
            numerical: self.numerical,
        })
    }

    // -- internals -----------------------------------------------------------

    fn check_pending(&self, node: &NodeRef) -> Result<(), BuilderError> {
        if self.region_index(&node.region).is_err() {
            return Err(format!(
                "node#{} region is not a region of this alternative",
                node.node.0
            ));
        }
        if !self.pending_nodes.contains_key(node) {
            return Err(format!(
                "node#{} is absent or already consumed",
                node.node.0
            ));
        }
        Ok(())
    }

    fn check_pending_primitive(&self, node: &NodeRef) -> Result<(), BuilderError> {
        self.check_pending(node)?;
        match &self.pending_nodes.get(node).expect("checked").kind {
            LogicalNodeKind::Primitive(_) => Ok(()),
            other => Err(format!(
                "node#{} is {:?}, not a primitive node",
                node.node.0,
                std::mem::discriminant(other)
            )),
        }
    }

    fn consume_parameter_state(&mut self, token: StateTokenId) {
        if let Some(parameter) = self.parameter_states.get(&token).copied() {
            self.pending_inputs.remove(&parameter);
        }
        if let Some(value) = self.parameter_state_values.get(&token).copied() {
            self.pending_values.remove(&value);
            if let Some(parameter) = self.parameter_values.get(&value).copied() {
                self.pending_inputs.remove(&parameter);
            }
        }
    }

    fn consume_value(&mut self, value: GraphValueId) -> Result<(), BuilderError> {
        // Values may have several consumers; the pending set tracks the
        // first use. Unused values are caught by `finish_alternative`.
        self.pending_values.remove(&value);
        if let Some(parameter) = self.parameter_values.get(&value).copied() {
            self.pending_inputs.remove(&parameter);
        }
        Ok(())
    }

    fn consume_node_edges(&mut self, node: &NodeRef) -> Result<(), BuilderError> {
        let region = self.region_index(&node.region)?;
        let logical = self
            .pending_nodes
            .get(node)
            .cloned()
            .ok_or_else(internal_pending)?;
        for token in logical.state_inputs.iter() {
            if self.pending_state_edges.remove(&StateEdge::NodeInput {
                region,
                node: node.node,
                token: *token,
            }) {
                self.consume_parameter_state(*token);
            }
        }
        for token in logical.state_outputs.iter() {
            self.pending_state_edges.remove(&StateEdge::NodeOutput {
                region,
                node: node.node,
                token: token.id,
            });
        }
        Ok(())
    }

    fn consume_region_inputs(&mut self, region: usize) -> Result<(), BuilderError> {
        for parameter_index in 0..self.regions[region].parameters.len() {
            self.pending_inputs.remove(&(region, parameter_index));
        }
        Ok(())
    }

    #[track_caller]
    fn consume_region_result(&mut self, region: usize, ordinal: usize) -> Result<(), BuilderError> {
        let caller = std::panic::Location::caller();
        let result = self
            .regions
            .get(region)
            .and_then(|r| r.results.get(ordinal))
            .cloned()
            .ok_or_else(|| {
                let detail = self
                    .regions
                    .get(region)
                    .map(|r| format!("path {:?} has {} results", r.path, r.results.len()))
                    .unwrap_or_else(|| format!("region index {region} is absent"));
                format!(
                    "region result#{ordinal} is absent: {detail}; requested at {}",
                    caller
                )
            })?;
        if !self.pending_results.remove(&(region, ordinal)) {
            return Err(format!("region result#{ordinal} is already consumed"));
        }
        if let RegionResult::Value { id, .. } = result {
            self.consume_value(id)?;
        }
        if let RegionResult::State { id, .. } = result {
            if self.pending_state_edges.remove(&StateEdge::RegionState {
                region,
                ordinal,
                token: id,
            }) {
                self.consume_parameter_state(id);
            }
        }
        Ok(())
    }

    fn pop_scope(
        &mut self,
        message: &str,
    ) -> Result<NonEmpty<ScheduleStepTemplate<D>>, BuilderError> {
        let scope = self
            .scopes
            .pop()
            .ok_or_else(|| "a schedule scope is absent".to_string())?;
        NonEmpty::new(scope.steps).ok_or_else(|| message.to_string())
    }

    fn append_step(&mut self, step: ScheduleStepTemplate<D>) -> Result<(), BuilderError> {
        let ordinal = self.next_step;
        self.next_step += 1;
        let refs = self.step_refs_of(&step);
        self.step_storage_refs.insert(ordinal, refs);
        let scope = self.scopes.last_mut().expect("a scope is open");
        scope.steps.push(step);
        Ok(())
    }

    /// Storage templates referenced by one step (including child internals of
    /// a call, which span only the call step).
    fn step_refs_of(&self, step: &ScheduleStepTemplate<D>) -> BTreeSet<PhysicalStorageTemplateId> {
        let mut refs = BTreeSet::new();
        fn walk<D: ExecutableDialect>(
            refs: &mut BTreeSet<PhysicalStorageTemplateId>,
            step: &ScheduleStepTemplate<D>,
        ) {
            match step {
                ScheduleStepTemplate::Launch(launch) => {
                    for storage in launch
                        .kernel
                        .workgroup_storage
                        .iter()
                        .chain(&launch.kernel.participant_storage)
                    {
                        refs.insert(*storage);
                    }
                    for step in launch.kernel.steps.iter() {
                        if let KernelStepTemplate::Mapped { bindings, .. } = step {
                            for (_, transport) in bindings {
                                collect_transport_storages(refs, std::slice::from_ref(transport));
                            }
                        }
                    }
                }
                ScheduleStepTemplate::Call(call) => {
                    for transport in call
                        .boundary
                        .inputs
                        .values()
                        .chain(call.boundary.results.values())
                    {
                        collect_transport_storages(refs, std::slice::from_ref(transport));
                    }
                    for state in call.boundary.states.values() {
                        if let StateTransportTemplate::Storage(template) = state {
                            refs.insert(*template);
                        }
                    }
                }
                ScheduleStepTemplate::If(if_step) => {
                    for step in if_step.then_schedule.steps.iter() {
                        walk(refs, step);
                    }
                    for step in if_step.else_schedule.steps.iter() {
                        walk(refs, step);
                    }
                    for join in &if_step.joins {
                        match join {
                            PhysicalJoinTemplate::Value {
                                then,
                                else_branch,
                                joined,
                            } => {
                                collect_transport_storages(
                                    refs,
                                    &[then.clone(), else_branch.clone(), joined.clone()],
                                );
                            }
                            PhysicalJoinTemplate::State { storage } => {
                                refs.insert(*storage);
                            }
                        }
                    }
                }
                ScheduleStepTemplate::Repeat(repeat) => {
                    for step in repeat.body.steps.iter() {
                        walk(refs, step);
                    }
                    for carry in &repeat.carried {
                        collect_transport_storages(refs, std::slice::from_ref(&carry.transport));
                    }
                }
            }
        }
        fn collect_transport_storages(
            refs: &mut BTreeSet<PhysicalStorageTemplateId>,
            transports: &[TransportTemplate],
        ) {
            for transport in transports {
                match transport {
                    TransportTemplate::Storage(views) => {
                        for view in views.iter() {
                            refs.insert(view.storage);
                        }
                    }
                    TransportTemplate::Tuple(items) => {
                        collect_transport_storages(refs, items.as_slice())
                    }
                    _ => {}
                }
            }
        }
        walk(&mut refs, step);
        refs
    }

    fn append_launch(
        &mut self,
        iteration: crate::dispatch::LinearIterationMap,
        bindings: Vec<(GraphValueId, TransportTemplate)>,
        ops: Vec<D::Op>,
    ) -> Result<(), BuilderError> {
        let ops = NonEmpty::new(ops).ok_or_else(|| "a launch has no opcode".to_string())?;
        // Op consequences: exact hard resources, cost, capabilities, numerics.
        // Cost is per work item: the block runs once per visit of the
        // iteration domain, priced at the workload's expected total for
        // runtime domains (geometry and resources below keep the bound).
        let work_priced = iteration.cost_symbol();
        for op in ops.iter() {
            let consequences = D::consequences(op);
            self.cost = self.cost.add(
                &Sym::constant(
                    i64::try_from(consequences.cost.0).map_err(|_| "cost estimate exceeds i64")?,
                )
                .mul(&work_priced),
            );
            if let Some(capability) = &consequences.capability {
                self.capabilities.insert(capability.clone());
            }
            self.numerical = crate::numerics::compose(&consequences.numerical, &self.numerical);
        }
        // Geometry from the iteration map: work items and participants are
        // planning expressions; runtime extents contribute their capacity as
        // the conservative resource bound.
        let work_items = iteration.total_symbol();
        let participants = iteration.participants.clone();
        let one = Sym::constant(1);
        let workgroups = [
            work_items
                .clone()
                .add(&participants.sub(&one))
                .quot(&participants),
            one.clone(),
            one.clone(),
        ];
        // Bindings: one direct group per distinct referenced storage.
        let mut members: BTreeMap<PhysicalStorageTemplateId, AccessMode> = BTreeMap::new();
        for (_, transport) in &bindings {
            collect_access(&mut members, transport);
        }
        fn collect_access(
            members: &mut BTreeMap<PhysicalStorageTemplateId, AccessMode>,
            transport: &TransportTemplate,
        ) {
            match transport {
                TransportTemplate::Storage(views) => {
                    for view in views.iter() {
                        let access = match view.access {
                            Access::Shared => AccessMode::Read,
                            Access::Exclusive => AccessMode::Write,
                        };
                        members.insert(view.storage, access);
                    }
                }
                TransportTemplate::Tuple(items) => {
                    for item in items.as_slice() {
                        collect_access(members, item);
                    }
                }
                _ => {}
            }
        }
        let mut binding_groups = Vec::new();
        if !members.is_empty() {
            let mut list: Vec<(PhysicalStorageTemplateId, AccessMode)> =
                members.into_iter().collect();
            list.sort_by_key(|(storage, _)| storage.0);
            let group_members = NonEmpty::new(
                list.into_iter()
                    .map(|(storage, access)| BindingMember { storage, access })
                    .collect(),
            )
            .expect("members is nonempty");
            binding_groups.push(BindingGroupTemplate {
                kind: BindingGroupKind::Direct,
                slot: 0,
                members: group_members,
            });
        }
        // Kernel steps: the mapped op block, then scalar publications (every
        // produced value bound to a slot in this launch is written by it).
        let mut kernel_steps = vec![KernelStepTemplate::Mapped {
            iteration,
            bindings: bindings.clone(),
            ops,
        }];
        for (_, transport) in &bindings {
            if let TransportTemplate::ExecutorScalar(ExecutorScalarTemplate {
                source: ExecutorScalarSource::Slot(slot),
                ..
            }) = transport
            {
                kernel_steps.push(KernelStepTemplate::PublishScalar { slot: *slot });
            }
        }
        let step = ScheduleStepTemplate::Launch(LaunchTemplate {
            geometry: DispatchTemplate {
                workgroups,
                participants_per_workgroup: [participants, one.clone(), one.clone()],
            },
            condition: LaunchCondition {
                work_items: work_items.clone(),
            },
            bindings: binding_groups,
            kernel: KernelTemplate {
                workgroup_storage: self.staged_workgroup.clone(),
                participant_storage: self.staged_participant.clone(),
                steps: NonEmpty::new(kernel_steps).expect("steps is nonempty"),
            },
        });
        self.append_step(step)
    }

    /// Commit one mapped block: a new launch, or — after a planned
    /// `barrier` — an additional phase of the already-open launch of the
    /// current scope (inside a launch, order is program order or planned
    /// barrier; emitters add no synchronization of their own).
    fn commit_launch(
        &mut self,
        iteration: crate::dispatch::LinearIterationMap,
        bindings: Vec<(GraphValueId, TransportTemplate)>,
        ops: Vec<D::Op>,
    ) -> Result<(), BuilderError> {
        if self.extend_launch {
            self.append_mapped_phase(iteration, bindings, ops)
        } else {
            self.append_launch(iteration, bindings, ops)
        }
    }

    /// Append one mapped phase (plus its scalar publications) to the open
    /// launch of the current scope.
    fn append_mapped_phase(
        &mut self,
        iteration: crate::dispatch::LinearIterationMap,
        bindings: Vec<(GraphValueId, TransportTemplate)>,
        ops: Vec<D::Op>,
    ) -> Result<(), BuilderError> {
        let ops = NonEmpty::new(ops).ok_or_else(|| "a mapped phase has no opcode".to_string())?;
        for op in ops.iter() {
            let consequences = D::consequences(op);
            self.cost = self.cost.add(&Sym::constant(
                i64::try_from(consequences.cost.0).map_err(|_| "cost estimate exceeds i64")?,
            ));
            if let Some(capability) = &consequences.capability {
                self.capabilities.insert(capability.clone());
            }
            self.numerical = crate::numerics::compose(&consequences.numerical, &self.numerical);
        }
        let scope = self.scopes.last_mut().expect("a schedule scope is open");
        let last = scope
            .steps
            .last_mut()
            .ok_or_else(|| "a barrier continuation needs an open launch".to_string())?;
        let launch = match last {
            ScheduleStepTemplate::Launch(launch) => launch,
            _ => return Err("a barrier continuation needs an open launch".into()),
        };
        let mut steps = launch.kernel.steps.clone().into_vec();
        let mut phase = vec![KernelStepTemplate::Mapped {
            iteration,
            bindings: bindings.clone(),
            ops,
        }];
        for (_, transport) in &bindings {
            if let TransportTemplate::ExecutorScalar(ExecutorScalarTemplate {
                source: ExecutorScalarSource::Slot(slot),
                ..
            }) = transport
            {
                phase.push(KernelStepTemplate::PublishScalar { slot: *slot });
            }
        }
        // Merge the phase's storage accesses into the launch's binding
        // groups: a later phase of the same launch may reference storages
        // the first commit never bound.
        merge_storage_members(&mut launch.bindings, &bindings);
        steps.extend(phase);
        launch.kernel.steps = NonEmpty::new(steps).expect("steps is nonempty");
        self.extend_launch = false;
        Ok(())
    }

    /// Shared launch construction for map_primitive / fuse.
    fn launch(
        &mut self,
        nodes: Vec<NodeRef>,
        iteration: crate::dispatch::LinearIterationMap,
        ops: Vec<D::Op>,
    ) -> Result<(), BuilderError> {
        let mut bindings = Vec::new();
        for node in &nodes {
            let logical = self
                .pending_nodes
                .get(node)
                .cloned()
                .ok_or_else(internal_pending)?;
            for input in logical.inputs.iter() {
                let transport = self.transport_of(*input)?;
                bindings.push((*input, transport));
                self.consume_value(*input)?;
            }
            self.consume_node_edges(node)?;
        }
        // Bind node outputs after inputs; emitters rely on this order.
        for node in &nodes {
            let logical = self
                .pending_nodes
                .get(node)
                .cloned()
                .ok_or_else(internal_pending)?;
            for output in logical.outputs.iter() {
                let transport = self.transport_of(output.id)?;
                bindings.push((output.id, transport));
            }
        }
        for node in &nodes {
            self.pending_nodes.remove(node);
        }
        self.commit_launch(iteration, bindings, ops)
    }
}

fn internal_pending() -> BuilderError {
    "internal: pending node disappeared".to_string()
}

/// Collect the resolved storages one resolved transport touches.
fn collect_resolved_storage(
    members: &mut BTreeMap<ResolvedStorageId, AccessMode>,
    transport: &ResolvedTransport,
) {
    match transport {
        ResolvedTransport::Storage(views) => {
            for view in views.iter() {
                let access = match view.access {
                    Access::Shared => AccessMode::Read,
                    Access::Exclusive => AccessMode::Write,
                };
                members.insert(
                    view.storage,
                    match members.get(&view.storage) {
                        Some(existing) if *existing != access => AccessMode::ReadWrite,
                        _ => access,
                    },
                );
            }
        }
        ResolvedTransport::Tuple(items) => {
            for item in items.as_slice() {
                collect_resolved_storage(members, item);
            }
        }
        _ => {}
    }
}

/// Union one phase's storage accesses into a launch's direct binding group.
fn merge_storage_members(
    groups: &mut Vec<BindingGroupTemplate>,
    bindings: &[(GraphValueId, TransportTemplate)],
) {
    fn access_of(access: Access) -> AccessMode {
        match access {
            Access::Shared => AccessMode::Read,
            Access::Exclusive => AccessMode::Write,
        }
    }
    fn collect(
        members: &mut BTreeMap<PhysicalStorageTemplateId, AccessMode>,
        transport: &TransportTemplate,
    ) {
        match transport {
            TransportTemplate::Storage(views) => {
                for view in views.iter() {
                    let access = access_of(view.access);
                    members.insert(
                        view.storage,
                        match members.get(&view.storage) {
                            Some(existing) if *existing != access => AccessMode::ReadWrite,
                            _ => access,
                        },
                    );
                }
            }
            TransportTemplate::Tuple(items) => {
                for item in items.as_slice() {
                    collect(members, item);
                }
            }
            _ => {}
        }
    }
    let mut members: BTreeMap<PhysicalStorageTemplateId, AccessMode> = BTreeMap::new();
    for (_, transport) in bindings {
        collect(&mut members, transport);
    }
    if members.is_empty() {
        return;
    }
    for group in groups.iter_mut() {
        if !matches!(group.kind, BindingGroupKind::Direct) {
            continue;
        }
        let mut merged: BTreeMap<PhysicalStorageTemplateId, AccessMode> = group
            .members
            .as_slice()
            .iter()
            .map(|member| (member.storage, member.access))
            .collect();
        for (storage, access) in &members {
            merged.insert(
                *storage,
                match merged.get(storage) {
                    Some(existing) if *existing != *access => AccessMode::ReadWrite,
                    _ => *access,
                },
            );
        }
        let mut list: Vec<(PhysicalStorageTemplateId, AccessMode)> = merged.into_iter().collect();
        list.sort_by_key(|(storage, _)| storage.0);
        group.members = NonEmpty::new(
            list.into_iter()
                .map(|(storage, access)| BindingMember { storage, access })
                .collect(),
        )
        .expect("members is nonempty");
        return;
    }
    let mut list: Vec<(PhysicalStorageTemplateId, AccessMode)> = members.into_iter().collect();
    list.sort_by_key(|(storage, _)| storage.0);
    groups.push(BindingGroupTemplate {
        kind: BindingGroupKind::Direct,
        slot: 0,
        members: NonEmpty::new(
            list.into_iter()
                .map(|(storage, access)| BindingMember { storage, access })
                .collect(),
        )
        .expect("members is nonempty"),
    });
}

// ---------------------------------------------------------------------------
// Byte size of logical storage
// ---------------------------------------------------------------------------

/// Byte size of one logical tensor storage. Runtime extents use their
/// capacity (the conservative resource bound; semantics always use `value`).
pub fn storage_bytes(shape: &TensorType) -> Result<SizeExpr, BuilderError> {
    let mut elements = Sym::constant(1);
    for axis in &shape.axes {
        let extent = extent_symbol(axis)?;
        elements = elements.mul(&extent);
    }
    let bytes = match &shape.elem {
        seismic_lang::types::Elem::Dtype(dtype) => elements.scale(i64::from(dtype.bytes())),
        seismic_lang::types::Elem::Repr(name) => {
            // Outer rows times the per-row plane extents over the packed axis.
            let packed_axis = shape
                .packed_axis
                .unwrap_or(shape.axes.len().saturating_sub(1));
            let packed_extent = extent_symbol(
                shape
                    .axes
                    .get(packed_axis)
                    .unwrap_or(&ExtentExpr::Static(0)),
            )?;
            let mut rows = Sym::constant(1);
            for (axis, extent) in shape.axes.iter().enumerate() {
                if axis != packed_axis {
                    rows = rows.mul(&extent_symbol(extent)?);
                }
            }
            let representation = repr::lookup(name)
                .ok_or_else(|| format!("unknown packed representation `{name}`"))?;
            let mut per_row = Sym::constant(0);
            for plane in representation.planes() {
                let plane_elements = plane.extent(&packed_extent);
                let plane_bytes =
                    i64::try_from(plane.dtype().bytes()).map_err(|_| "plane dtype bytes")?;
                per_row = per_row.add(&plane_elements.scale(plane_bytes));
            }
            rows.mul(&per_row)
        }
        seismic_lang::types::Elem::Param(_) => {
            return Err("unsubstituted element parameter in storage bytes".into());
        }
    };
    Ok(bytes)
}

/// The planning (resource-bound) symbol of one extent: static values stay
/// exact; runtime extents contribute their capacity.
pub fn extent_symbol(extent: &ExtentExpr) -> Result<SizeExpr, BuilderError> {
    match extent {
        ExtentExpr::Static(n) => Ok(Sym::constant(
            i64::try_from(*n).map_err(|_| "extent exceeds i64")?,
        )),
        ExtentExpr::Sym(sym) => Ok(sym.clone()),
        ExtentExpr::Runtime(_) => Err("runtime extents require their capacity bound".into()),
    }
}

fn alignment_of(shape: &TensorType) -> u64 {
    match &shape.elem {
        seismic_lang::types::Elem::Dtype(dtype) => u64::from(dtype.bytes()),
        // Packed planes are u32 word rows.
        seismic_lang::types::Elem::Repr(_) => 4,
        seismic_lang::types::Elem::Param(_) => 1,
    }
}

// ---------------------------------------------------------------------------
// The plan family builder
// ---------------------------------------------------------------------------

/// Constructs one `PlanFamily` from a logical program. Each physical
/// alternative is produced by an `AlternativeBuilder` obtained from
/// `alternative`; every applicable portable alternative must receive at
/// least one universal physical alternative (compiler bug otherwise).
pub struct PlanFamilyBuilder<D: ExecutableDialect> {
    logical_identity: LogicalIdentity,
    target: seismic_lang::logical::EffectiveTargetIdentity,
    entry: ChoiceId,
    interfaces: BTreeMap<ChoiceId, FunctionInterface>,
    logical_alternatives: BTreeMap<ChoiceId, u32>,
    graphs: BTreeMap<(ChoiceId, u32), TaskGraph>,
    runtime_extents: Vec<RuntimeExtent>,
    shared: Rc<RefCell<FamilyShared<D>>>,
    alternatives: BTreeMap<ChoiceId, Vec<PhysicalAlternative<D>>>,
    /// ABI storage templates by canonical boundary leaf (root only).
    abi_storage_paths: BTreeMap<BoundaryLeaf, PhysicalStorageTemplateId>,
}

impl<D: ExecutableDialect> PlanFamilyBuilder<D> {
    pub fn from_logical(logical: &LogicalProgram) -> Result<Self, BuilderError> {
        let mut interfaces = BTreeMap::new();
        let mut logical_alternatives = BTreeMap::new();
        let mut graphs = BTreeMap::new();
        for (choice_id, choice) in logical.choices.ids().zip(logical.choices.iter()) {
            interfaces.insert(choice_id, choice.interface.clone());
            logical_alternatives.insert(choice_id, choice.alternatives.len() as u32);
            for (ordinal, alternative) in choice.alternatives.iter().enumerate() {
                graphs.insert(
                    (choice_id, ordinal as u32),
                    logical.graph(alternative.graph).clone(),
                );
            }
        }
        Ok(Self {
            logical_identity: logical.identity,
            target: logical.target.clone(),
            entry: logical.entry_choice,
            interfaces,
            logical_alternatives,
            graphs,
            runtime_extents: logical.runtime_extents.iter().cloned().collect(),
            shared: Rc::new(RefCell::new(FamilyShared::new())),
            alternatives: BTreeMap::new(),
            abi_storage_paths: BTreeMap::new(),
        })
    }

    /// One declared tuning/placement parameter (participant width,
    /// workgroups, vector width, fan-in, unroll, tiles, window width…).
    /// Names are unique across the family; the parameter is visible to every
    /// alternative of the family and constrained by the one planning model.
    pub fn tuning_parameter(
        &mut self,
        name: impl Into<String>,
        lower: i64,
        upper: i64,
    ) -> Result<PlanParamId, BuilderError> {
        self.shared
            .borrow_mut()
            .insert_parameter(name.into(), lower, upper)
    }

    /// A family-level planned executor-scalar slot for carries/control.
    pub fn executor_scalar_slot(
        &mut self,
        dtype: DType,
        name: impl Into<String>,
    ) -> ExecutorScalarSlotId {
        self.shared.borrow_mut().fresh_slot(dtype, name.into())
    }

    /// Read-only listing of every declared plan parameter (for strategies
    /// and tests to verify registration without finishing the family).
    pub fn declared_parameters(&self) -> Vec<(PlanParamId, PlanParameter)> {
        self.shared
            .borrow()
            .params
            .iter()
            .map(|(id, parameter)| (PlanParamId(*id), parameter.clone()))
            .collect()
    }

    /// Read-only access to one choice alternative's logical task graph (the
    /// graph a cross-call `fuse_call` imports).
    pub fn graph_of(&self, choice: ChoiceId, logical_alternative: u32) -> Option<&TaskGraph> {
        self.graphs.get(&(choice, logical_alternative))
    }

    /// Open the builder of one physical alternative of (choice, logical
    /// alternative). The physical ordinal is the next alternative slot.
    pub fn alternative(
        &self,
        choice: ChoiceId,
        logical_alternative: u32,
    ) -> Result<AlternativeBuilder<D>, BuilderError> {
        let interface = self
            .interfaces
            .get(&choice)
            .cloned()
            .ok_or_else(|| format!("choice#{} is absent", choice.0))?;
        if logical_alternative
            >= *self
                .logical_alternatives
                .get(&choice)
                .ok_or_else(|| format!("choice#{} is absent", choice.0))?
        {
            return Err(format!(
                "choice#{} has no logical alternative#{logical_alternative}",
                choice.0
            ));
        }
        let graph = self
            .graphs
            .get(&(choice, logical_alternative))
            .cloned()
            .ok_or_else(|| "the logical alternative's graph is absent".to_string())?;
        let physical_alternative = self
            .alternatives
            .get(&choice)
            .map(|list| list.len() as u32)
            .unwrap_or(0);
        let builder = AlternativeBuilder::new(
            Rc::clone(&self.shared),
            graph,
            interface,
            choice == self.entry,
            choice,
            logical_alternative,
            physical_alternative,
        )?;
        Ok(builder)
    }

    /// Commit one finished alternative into the family. The owning choice is
    /// named explicitly: family assembly is not an alternative-builder
    /// transition.
    pub fn add_alternative(
        &mut self,
        choice: ChoiceId,
        alternative: PhysicalAlternative<D>,
    ) -> Result<(), BuilderError> {
        if !self.interfaces.contains_key(&choice) {
            return Err(format!("choice#{} is absent", choice.0));
        }
        let expected = self
            .alternatives
            .get(&choice)
            .map(|list| list.len() as u32)
            .unwrap_or(0);
        if alternative.physical_alternative != expected {
            return Err(format!(
                "alternative ordinal {} disagrees with the next slot {expected}",
                alternative.physical_alternative
            ));
        }
        self.alternatives
            .entry(choice)
            .or_default()
            .push(alternative);
        Ok(())
    }

    /// Finish the family: coverage, activation, interference, and hard facts.
    pub fn finish(self) -> Result<PlanFamily<D>, BuilderError> {
        // Every applicable logical alternative of every *activated* choice
        // has a universal physical one. A choice no alternative invokes (its
        // only caller fused the interval) activates nothing and needs no
        // standalone alternative.
        let mut reachable: BTreeSet<ChoiceId> = BTreeSet::from([self.entry]);
        let mut queue = vec![self.entry];
        while let Some(choice) = queue.pop() {
            for ordinal in 0..self.logical_alternatives.get(&choice).copied().unwrap_or(0) {
                let Some(alternatives) = self.alternatives.get(&choice) else {
                    continue;
                };
                for alternative in alternatives.iter() {
                    if alternative.logical_alternative != ordinal {
                        continue;
                    }
                    for step in flat_steps_public(&alternative.schedule) {
                        if let ScheduleStepTemplate::Call(call) = step {
                            if reachable.insert(call.choice) {
                                queue.push(call.choice);
                            }
                        }
                    }
                }
            }
        }
        for (choice, count) in &self.logical_alternatives {
            if !reachable.contains(choice) {
                continue;
            }
            for ordinal in 0..*count {
                let covered = self
                    .alternatives
                    .get(choice)
                    .is_some_and(|list| list.iter().any(|alt| alt.logical_alternative == ordinal));
                if !covered {
                    return Err(format!(
                        "applicable choice#{} logical alternative#{ordinal} has no universal \
                         physical alternative (compiler bug)",
                        choice.0
                    ));
                }
            }
        }
        // Assemble the choice table. A choice no alternative invokes (its
        // only caller fused the interval) is unselectable and left out.
        let mut choices = Vec::new();
        for (choice, interface) in &self.interfaces {
            if !reachable.contains(choice) {
                continue;
            }
            let mut list = self
                .alternatives
                .get(choice)
                .cloned()
                .unwrap_or_default()
                .into_iter();
            let first = list
                .next()
                .ok_or_else(|| format!("choice#{} has no physical alternative", choice.0))?;
            let mut alternative_list = vec![first];
            alternative_list.extend(list);
            let alternatives = NonEmpty::new(alternative_list).expect("the list is nonempty");
            choices.push((
                *choice,
                PhysicalChoice {
                    interface: interface.clone(),
                    alternatives,
                },
            ));
        }
        let choices = IdVec::from_iter(choices);
        // Global tables from the shared allocation state.
        let shared = self.shared.borrow();
        let storages = IdVec::from_iter(
            shared
                .storages
                .iter()
                .map(|(id, template)| (PhysicalStorageTemplateId(*id), template.clone())),
        );
        let executor_scalar_slots = IdVec::from_iter(
            shared
                .slots
                .iter()
                .map(|(id, decl)| (ExecutorScalarSlotId(*id), decl.clone())),
        );
        let parameters = IdVec::from_iter(
            shared
                .params
                .iter()
                .map(|(id, parameter)| (PlanParamId(*id), parameter.clone())),
        );
        drop(shared);
        let runtime_extents = IdVec::new(self.runtime_extents.clone());
        let abi_storage_paths = self.collect_abi_paths();
        let _ = &self.abi_storage_paths;
        let mut family = PlanFamily {
            logical: self.logical_identity.clone(),
            target: self.target.clone(),
            entry: self.entry,
            choices,
            parameters,
            storages,
            executor_scalar_slots,
            runtime_extents,
            abi_storage_paths: abi_storage_paths.clone(),
            model: PlanningModelExport::default(),
        };
        family.model = export_model(&family)?;
        Ok(family)
    }

    /// ABI storage templates by canonical boundary leaf (root only),
    /// recorded when the entry alternative created them.
    fn collect_abi_paths(&self) -> BTreeMap<BoundaryLeaf, PhysicalStorageTemplateId> {
        self.shared.borrow().abi_leaves.clone()
    }
}

/// Every schedule step of one alternative, in execution-tree order.
pub fn flat_steps_public<'a, D: ExecutableDialect>(
    schedule: &'a ScheduleTemplate<D>,
) -> Vec<&'a ScheduleStepTemplate<D>> {
    fn walk<'a, D: ExecutableDialect>(
        schedule: &'a ScheduleTemplate<D>,
        steps: &mut Vec<&'a ScheduleStepTemplate<D>>,
    ) {
        for step in schedule.steps.iter() {
            steps.push(step);
            match step {
                ScheduleStepTemplate::If(if_step) => {
                    walk(&if_step.then_schedule, steps);
                    walk(&if_step.else_schedule, steps);
                }
                ScheduleStepTemplate::Repeat(repeat) => {
                    walk(&repeat.body, steps);
                }
                _ => {}
            }
        }
    }
    let mut steps = Vec::new();
    walk(schedule, &mut steps);
    steps
}

/// Per-alternative schedule walk used by facts and interference.
fn walk_steps<D: ExecutableDialect>(
    schedule: &ScheduleTemplate<D>,
    visit: &mut dyn FnMut(&ScheduleStepTemplate<D>),
) {
    for step in schedule.steps.iter() {
        visit_step(step, visit);
    }
}

fn visit_step<D: ExecutableDialect>(
    step: &ScheduleStepTemplate<D>,
    visit: &mut dyn FnMut(&ScheduleStepTemplate<D>),
) {
    visit(step);
    match step {
        ScheduleStepTemplate::If(if_step) => {
            walk_steps(&if_step.then_schedule, visit);
            walk_steps(&if_step.else_schedule, visit);
        }
        ScheduleStepTemplate::Repeat(repeat) => {
            walk_steps(&repeat.body, visit);
        }
        _ => {}
    }
}

/// Hard facts and interference pairs for the global planning model.
fn export_model<D: ExecutableDialect>(
    family: &PlanFamily<D>,
) -> Result<PlanningModelExport, BuilderError> {
    // Hard facts per alternative.
    let mut alternative_facts = Vec::new();
    for choice_id in family.choices.ids() {
        let physical = &family.choices[choice_id];
        for alternative in physical.alternatives.iter() {
            let mut launches = Vec::new();
            let mut capabilities: BTreeSet<IntrinsicId> = BTreeSet::new();
            walk_steps(&alternative.schedule, &mut |step| {
                if let ScheduleStepTemplate::Launch(launch) = step {
                    let mut launch_capabilities = BTreeSet::new();
                    let mut barriers = 0u32;
                    for kernel_step in launch.kernel.steps.iter() {
                        if let KernelStepTemplate::Barrier { .. } = kernel_step {
                            barriers += 1;
                        }
                        if let KernelStepTemplate::Mapped { ops, .. } = kernel_step {
                            for op in ops.iter() {
                                let consequences = D::consequences(op);
                                if let Some(capability) = consequences.capability {
                                    launch_capabilities.insert(capability.clone());
                                    capabilities.insert(capability);
                                }
                            }
                        }
                    }
                    launches.push(LaunchFacts {
                        preferred_participants: launch.geometry.participants_per_workgroup[0]
                            .clone(),
                        workgroups: launch.geometry.workgroups.clone(),
                        workgroup_bytes: launch
                            .kernel
                            .workgroup_storage
                            .iter()
                            .filter_map(|storage| family.storages.get(*storage))
                            .fold(Sym::constant(0), |total, template| {
                                total.add(&template.bytes)
                            }),
                        private_bytes_per_participant: launch
                            .kernel
                            .participant_storage
                            .iter()
                            .filter_map(|storage| family.storages.get(*storage))
                            .fold(Sym::constant(0), |total, template| {
                                total.add(&template.bytes)
                            }),
                        direct_bindings: launch
                            .bindings
                            .iter()
                            .filter(|group| group.kind == BindingGroupKind::Direct)
                            .map(|group| group.members.len() as u32)
                            .sum(),
                        argument_table_bytes: 0,
                        launch_condition_work_items: launch.condition.work_items.clone(),
                        barriers,
                        capabilities: launch_capabilities,
                    });
                }
            });
            alternative_facts.push(AlternativeFacts {
                choice: choice_id,
                logical_alternative: alternative.logical_alternative,
                physical_alternative: alternative.physical_alternative,
                launches,
                cost: alternative.cost.clone(),
                numerical: alternative.numerical.clone(),
                required_capabilities: capabilities,
            });
        }
    }
    // Device-arena activations.
    let storage: Vec<StorageActivation> = family
        .storages
        .ids()
        .zip(family.storages.iter())
        .filter(|(_, template)| template.scope == StorageScope::DeviceArena)
        .map(|(storage_id, template)| StorageActivation {
            storage: storage_id,
            active_if: template.active_if.clone(),
        })
        .collect();
    // Lifetime interference: spans per alternative (child internals span the
    // calling step; sequential siblings may reuse space).
    let mut interference: BTreeSet<StorageInterference> = BTreeSet::new();
    let mut span_cache: BTreeMap<
        (ChoiceId, u32, u32),
        BTreeMap<PhysicalStorageTemplateId, (u32, u32)>,
    > = BTreeMap::new();
    for choice_id in family.choices.ids() {
        for alternative in family.choices[choice_id].alternatives.iter() {
            let spans = alternative_spans(
                family,
                choice_id,
                alternative.logical_alternative,
                alternative.physical_alternative,
                &mut span_cache,
            );
            let mut ids: Vec<PhysicalStorageTemplateId> = spans
                .keys()
                .copied()
                .filter(|id| {
                    family
                        .storages
                        .get(*id)
                        .is_some_and(|template| template.scope == StorageScope::DeviceArena)
                })
                .collect();
            ids.sort_by_key(|id| id.0);
            for index in 0..ids.len() {
                for other in index + 1..ids.len() {
                    let (left, right) = (ids[index], ids[other]);
                    let (a1, a2) = spans[&left];
                    let (b1, b2) = spans[&right];
                    if a1 <= b2 && b1 <= a2 {
                        interference.insert(StorageInterference { left, right });
                    }
                }
            }
        }
    }
    Ok(PlanningModelExport {
        alternatives: alternative_facts,
        storage,
        interference: interference.into_iter().collect(),
    })
}

/// Lifetime spans (first..last step ordinal) of every storage referenced by
/// one alternative. Child internals span the calling step's ordinal.
fn alternative_spans<D: ExecutableDialect>(
    family: &PlanFamily<D>,
    choice: ChoiceId,
    logical: u32,
    physical: u32,
    cache: &mut BTreeMap<(ChoiceId, u32, u32), BTreeMap<PhysicalStorageTemplateId, (u32, u32)>>,
) -> BTreeMap<PhysicalStorageTemplateId, (u32, u32)> {
    let key = (choice, logical, physical);
    if let Some(cached) = cache.get(&key) {
        return cached.clone();
    }
    let alternative = family
        .choices
        .get(choice)
        .and_then(|physical_choice| {
            physical_choice.alternatives.iter().find(|alternative| {
                alternative.logical_alternative == logical
                    && alternative.physical_alternative == physical
            })
        })
        .expect("the alternative exists");
    let mut spans: BTreeMap<PhysicalStorageTemplateId, (u32, u32)> = BTreeMap::new();
    let mut next_ordinal: u32 = 0;
    fn record(
        spans: &mut BTreeMap<PhysicalStorageTemplateId, (u32, u32)>,
        storage: PhysicalStorageTemplateId,
        ordinal: u32,
    ) {
        spans
            .entry(storage)
            .and_modify(|(first, last)| {
                if ordinal < *first {
                    *first = ordinal;
                }
                if ordinal > *last {
                    *last = ordinal;
                }
            })
            .or_insert((ordinal, ordinal));
    }
    fn walk<D: ExecutableDialect>(
        family: &PlanFamily<D>,
        schedule: &ScheduleTemplate<D>,
        next_ordinal: &mut u32,
        spans: &mut BTreeMap<PhysicalStorageTemplateId, (u32, u32)>,
        cache: &mut BTreeMap<(ChoiceId, u32, u32), BTreeMap<PhysicalStorageTemplateId, (u32, u32)>>,
    ) {
        for step in schedule.steps.iter() {
            match step {
                ScheduleStepTemplate::Launch(launch) => {
                    let ordinal = *next_ordinal;
                    *next_ordinal += 1;
                    for storage in launch
                        .kernel
                        .workgroup_storage
                        .iter()
                        .chain(&launch.kernel.participant_storage)
                    {
                        record(spans, *storage, ordinal);
                    }
                    for kernel_step in launch.kernel.steps.iter() {
                        if let KernelStepTemplate::Mapped { bindings, .. } = kernel_step {
                            for (_, transport) in bindings {
                                record_transport(spans, transport, ordinal);
                            }
                        }
                    }
                }
                ScheduleStepTemplate::Call(call) => {
                    let ordinal = *next_ordinal;
                    *next_ordinal += 1;
                    // Child internals span only this call step; consider
                    // every physical alternative of the child choice.
                    if let Some(child_choice) = family.choices.get(call.choice) {
                        for child in child_choice.alternatives.iter() {
                            let child_spans = alternative_spans(
                                family,
                                call.choice,
                                child.logical_alternative,
                                child.physical_alternative,
                                cache,
                            );
                            for storage in child_spans.keys() {
                                if matches!(
                                    family.storages.get(*storage).map(|t| t.scope),
                                    Some(StorageScope::DeviceArena)
                                ) {
                                    record(spans, *storage, ordinal);
                                }
                            }
                        }
                    }
                }
                ScheduleStepTemplate::If(if_step) => {
                    walk(family, &if_step.then_schedule, next_ordinal, spans, cache);
                    walk(family, &if_step.else_schedule, next_ordinal, spans, cache);
                }
                ScheduleStepTemplate::Repeat(repeat) => {
                    walk(family, &repeat.body, next_ordinal, spans, cache);
                }
            }
        }
    }
    fn record_transport(
        spans: &mut BTreeMap<PhysicalStorageTemplateId, (u32, u32)>,
        transport: &TransportTemplate,
        ordinal: u32,
    ) {
        match transport {
            TransportTemplate::Storage(views) => {
                for view in views.iter() {
                    record(spans, view.storage, ordinal);
                }
            }
            TransportTemplate::Tuple(items) => {
                for item in items.as_slice() {
                    record_transport(spans, item, ordinal);
                }
            }
            _ => {}
        }
    }
    walk(
        family,
        &alternative.schedule,
        &mut next_ordinal,
        &mut spans,
        cache,
    );
    // Always-live storages (live across calls) span the whole alternative.
    for (storage_id, template) in family.storages.ids().zip(family.storages.iter()) {
        if template.scope == StorageScope::DeviceArena
            && template.lifetime == StorageLifetime::Always
            && template.active_if.holds(&{
                // Active under this alternative's own selection alone when
                // its literal is exactly this alternative's term (or empty).
                let mut selections = BTreeMap::new();
                selections.insert(choice, (logical, physical));
                selections
            })
            && spans.contains_key(&storage_id)
        {
            spans.insert(storage_id, (0, next_ordinal.saturating_sub(1)));
        }
    }
    cache.insert(key, spans.clone());
    spans
}

// ---------------------------------------------------------------------------
// The resolver
// ---------------------------------------------------------------------------

struct Resolver<'a, D: ExecutableDialect> {
    family: &'a PlanFamily<D>,
    values: PlanValues,
    selections: BTreeMap<ChoiceId, (u32, u32)>,
    offsets: BTreeMap<PhysicalStorageTemplateId, u64>,
    storage_ids: BTreeMap<PhysicalStorageTemplateId, ResolvedStorageId>,
    resolved_storages: Vec<(ResolvedStorageId, ResolvedStorage<D>)>,
    /// ABI binding per storage template (root ABI storages only).
    abi_bindings: BTreeMap<PhysicalStorageTemplateId, BufferBindingId>,
    abi_scalars: Option<ScalarLayout>,
    abi_results: Vec<ResultBinding>,
    slot_ids: BTreeMap<ExecutorScalarSlotId, ResolvedExecutorScalarId>,
    next_storage: u64,
    next_kernel_value: u64,
    next_launch: u64,
    next_slot_resolved: u64,
    /// Scalar graph values resolved to execution expressions (for runtime
    /// extent value resolution).
    scalar_exprs: BTreeMap<GraphValueId, ExecutionExpr>,
    status_fields: Vec<StatusField<D>>,
    max_workgroup_bytes: u64,
    max_private_bytes: u64,
    direct_bindings: u32,
    capabilities: BTreeSet<IntrinsicId>,
    cost: u64,
}

impl<D: ExecutableDialect> PhysicalIdentityResolver for Resolver<'_, D> {
    fn storage(
        &mut self,
        template: PhysicalStorageTemplateId,
    ) -> Result<ResolvedStorageId, InvariantReport> {
        self.resolved_storage(template)
    }

    fn slot(
        &mut self,
        template: ExecutorScalarSlotId,
    ) -> Result<ResolvedExecutorScalarId, InvariantReport> {
        self.resolved_slot(template)
    }

    fn result_scalar(
        &mut self,
        path: &ValuePath,
        endpoint: Option<seismic_lang::abi::RangeEndpoint>,
        dtype: DType,
    ) -> Result<ResultScalarFieldId, InvariantReport> {
        match self.resolve_abi_scalar(
            &BoundaryLeaf::Result { leaf: path.clone() },
            endpoint,
            dtype,
        )? {
            ResolvedExecutorScalar::Result { field, .. } => Ok(field),
            _ => Err(InvariantReport(
                "result scalar resolved outside the result block".into(),
            )),
        }
    }
}

impl<D: ExecutableDialect> PlanFamily<D> {
    /// Resolve the selected plan. Ordinary resolution is infallible: it only
    /// evaluates solved expressions, allocates ids, instantiates boundaries,
    /// substitutes offsets/geometry/opcodes, and recurses. It rechecks no
    /// legality. Overflow or a missing id is `CompilerBug`.
    pub fn resolve(
        &self,
        assignment: &FeasiblePlanAssignment,
        numerical: NumericalAssessment,
        optimal: bool,
    ) -> Result<ResolvedPlan<D>, InvariantReport> {
        let values = PlanValues {
            symbols: assignment.symbols().clone(),
        };
        let mut resolver = Resolver {
            family: self,
            values,
            selections: assignment.selections().clone(),
            offsets: assignment.offsets().clone(),
            storage_ids: BTreeMap::new(),
            resolved_storages: Vec::new(),
            abi_bindings: BTreeMap::new(),
            abi_scalars: None,
            abi_results: Vec::new(),
            slot_ids: BTreeMap::new(),
            next_storage: 0,
            next_kernel_value: 0,
            next_launch: 0,
            next_slot_resolved: 0,
            scalar_exprs: BTreeMap::new(),
            status_fields: Vec::new(),
            max_workgroup_bytes: 0,
            max_private_bytes: 0,
            direct_bindings: 0,
            capabilities: BTreeSet::new(),
            cost: 0,
        };
        let abi = resolver.build_abi()?;
        resolver.abi_scalars = Some(abi.scalars.clone());
        resolver.abi_results = abi.results.clone();
        let entry = resolver.resolve_body(self.entry, None)?;
        // Retained runtime extent values (never capacities).
        let mut resolved_extents = Vec::new();
        for (id, extent) in self.runtime_extents.ids().zip(self.runtime_extents.iter()) {
            let value = resolver.resolve_scalar_expr(&extent.value.clone())?;
            resolved_extents.push((id, value));
        }
        let runtime_extents = IdVec::from_iter(resolved_extents);
        let arena_bytes = resolver.arena_bytes()?;
        let storage = IdVec::from_iter(std::mem::take(&mut resolver.resolved_storages));
        let estimated_cost = resolver.cost;
        let resources = ResolvedProgramResources {
            max_workgroup_bytes: resolver.max_workgroup_bytes,
            max_private_bytes_per_participant: resolver.max_private_bytes,
            direct_bindings: resolver.direct_bindings,
            argument_table_bytes: 0,
            arena_bytes,
            capabilities: resolver.capabilities.clone(),
        };
        let entry_interface = &self.choices[self.entry].interface;
        let identity = ResolutionIdentity {
            logical: self.logical,
            entry: entry_interface.name.clone(),
            target: self.target.clone(),
            toolchain_fingerprint: String::new(),
            selections: assignment.selections().clone(),
        };
        Ok(ResolvedPlan {
            identity,
            abi: ResolvedAbi {
                buffers: abi.buffers,
                scalars: abi.scalars,
                results: abi.results,
                alias_rules: abi.alias_rules,
                status: (!resolver.status_fields.is_empty()).then(|| StatusBinding {
                    bytes: (resolver.status_fields.len() as u64).saturating_mul(4),
                    fields: resolver.status_fields.clone(),
                }),
            },
            internal_arena: ResolvedArena { bytes: arena_bytes },
            storage,
            entry,
            runtime_extents,
            resources,
            numerical,
            estimated_cost,
            optimal,
        })
    }
}

impl<'a, D: ExecutableDialect> Resolver<'a, D> {
    fn arena_bytes(&self) -> Result<u64, InvariantReport> {
        let mut arena_bytes = 0u64;
        for activation in &self.family.model.storage {
            if !activation.active_if.holds(&self.selections) {
                continue;
            }
            let template = self
                .family
                .storages
                .get(activation.storage)
                .ok_or_else(|| {
                    InvariantReport(format!("storage#{:?} is absent", activation.storage))
                })?;
            let bytes = self.values.eval(&template.bytes)?;
            if bytes == 0 {
                continue;
            }
            let offset = self
                .offsets
                .get(&activation.storage)
                .copied()
                .ok_or_else(|| {
                    InvariantReport(format!(
                        "active arena storage#{:?} has no solved offset",
                        activation.storage
                    ))
                })?;
            let end = offset.checked_add(bytes).ok_or_else(|| {
                InvariantReport(format!(
                    "arena extent overflows for storage#{:?}",
                    activation.storage
                ))
            })?;
            arena_bytes = arena_bytes.max(end);
        }
        Ok(arena_bytes)
    }

    fn selected_alternative(
        &self,
        choice: ChoiceId,
    ) -> Result<&PhysicalAlternative<D>, InvariantReport> {
        let (logical, physical) = self
            .selections
            .get(&choice)
            .copied()
            .ok_or_else(|| InvariantReport(format!("choice#{} has no selection", choice.0)))?;
        let physical_choice = self.family.choices.get(choice).ok_or_else(|| {
            InvariantReport(format!("choice#{} is absent from the family", choice.0))
        })?;
        physical_choice
            .alternatives
            .iter()
            .find(|alternative| {
                alternative.logical_alternative == logical
                    && alternative.physical_alternative == physical
            })
            .ok_or_else(|| {
                InvariantReport(format!(
                    "choice#{} has no alternative logical#{logical} physical#{physical}",
                    choice.0
                ))
            })
    }

    // -- root ABI --------------------------------------------------------------

    fn build_abi(&mut self) -> Result<AbiSkeleton, InvariantReport> {
        let interface = self.family.choices[self.family.entry].interface.clone();
        let mut scalar_parameters: Vec<ScalarParameter> = Vec::new();
        let mut buffers: Vec<BufferBinding> = Vec::new();
        let mut parameter_bindings: Vec<(BoundaryLeaf, ParamOwnership, BufferBindingId)> =
            Vec::new();
        let mut next_binding: u64 = 0;
        for (ordinal, parameter) in interface.params.iter().enumerate() {
            let leaves =
                canonical_leaves(&parameter.ty).map_err(|reason| InvariantReport(reason))?;
            for (path, leaf) in leaves {
                match leaf {
                    Leaf::Scalar(dtype) => scalar_parameters.push(ScalarParameter {
                        name: abi_scalar_name(&parameter.name, &path),
                        dtype,
                        index_bound: None,
                        range: None,
                    }),
                    Leaf::Index(bound) => scalar_parameters.push(ScalarParameter {
                        name: abi_scalar_name(&parameter.name, &path),
                        dtype: DType::I32,
                        index_bound: Some(extent_bound(bound)?),
                        range: None,
                    }),
                    Leaf::Range(bound) => {
                        let bound = extent_bound(bound)?;
                        let name = abi_scalar_name(&parameter.name, &path);
                        scalar_parameters.push(ScalarParameter {
                            name: name.clone(),
                            dtype: DType::I32,
                            index_bound: None,
                            range: Some(seismic_lang::abi::RangeScalar {
                                parameter: name.clone(),
                                endpoint: seismic_lang::abi::RangeEndpoint::Start,
                                bound,
                            }),
                        });
                        scalar_parameters.push(ScalarParameter {
                            name: name.clone(),
                            dtype: DType::I32,
                            index_bound: None,
                            range: Some(seismic_lang::abi::RangeScalar {
                                parameter: name,
                                endpoint: seismic_lang::abi::RangeEndpoint::End,
                                bound,
                            }),
                        });
                    }
                    Leaf::Tensor(_) => {
                        let key = BoundaryLeaf::Input {
                            param: ordinal as u32,
                            leaf: path.clone(),
                        };
                        let template = self
                            .family
                            .abi_storage_paths
                            .get(&key)
                            .copied()
                            .ok_or_else(|| {
                                InvariantReport(format!(
                                    "the root ABI has no storage for parameter leaf {key:?}"
                                ))
                            })?;
                        let bytes = self.storage_bytes(template)?;
                        let binding = BufferBindingId(next_binding);
                        next_binding += 1;
                        buffers.push(BufferBinding {
                            path: path.clone(),
                            plane: "dense".into(),
                            binding,
                            bytes,
                            alignment: self
                                .family
                                .storages
                                .get(template)
                                .map(|t| t.alignment)
                                .unwrap_or(4),
                            role: AbiRole::Parameter {
                                ordinal: ordinal as u32,
                            },
                        });
                        parameter_bindings.push((key, parameter.ownership, binding));
                        self.abi_bindings.insert(template, binding);
                    }
                }
            }
        }
        // Result bindings: tensors are runtime-allocated by path/plane;
        // scalar/index/range results use the compiler-owned result scalar block.
        let mut results: Vec<ResultBinding> = Vec::new();
        let mut next_result_field: u64 = 0;
        let result_leaves =
            canonical_leaves(&interface.result).map_err(|reason| InvariantReport(reason))?;
        for (path, leaf) in result_leaves {
            match leaf {
                Leaf::Tensor(_) => {
                    let key = BoundaryLeaf::Result { leaf: path.clone() };
                    let template = self
                        .family
                        .abi_storage_paths
                        .get(&key)
                        .copied()
                        .ok_or_else(|| {
                            InvariantReport(format!(
                                "the root ABI has no storage for result leaf {key:?}"
                            ))
                        })?;
                    let bytes = self.storage_bytes(template)?;
                    let binding = BufferBindingId(next_binding);
                    next_binding += 1;
                    buffers.push(BufferBinding {
                        path: path.clone(),
                        plane: "dense".into(),
                        binding,
                        bytes,
                        alignment: self
                            .family
                            .storages
                            .get(template)
                            .map(|t| t.alignment)
                            .unwrap_or(4),
                        role: AbiRole::Result,
                    });
                    self.abi_bindings.insert(template, binding);
                    results.push(ResultBinding::Buffer {
                        path,
                        plane: "dense".into(),
                        binding,
                    });
                }
                Leaf::Scalar(dtype) => {
                    let field = ResultScalarFieldId(next_result_field);
                    next_result_field += 1;
                    results.push(ResultBinding::Scalar { path, field, dtype });
                }
                Leaf::Index(bound) => {
                    let field = ResultScalarFieldId(next_result_field);
                    next_result_field += 1;
                    results.push(ResultBinding::Scalar {
                        path,
                        field,
                        dtype: DType::I32,
                    });
                    let _ = bound;
                }
                Leaf::Range(bound) => {
                    let bound = extent_bound(bound)?;
                    let start = ResultScalarFieldId(next_result_field);
                    next_result_field += 1;
                    let end = ResultScalarFieldId(next_result_field);
                    next_result_field += 1;
                    results.push(ResultBinding::Range {
                        path,
                        start,
                        end,
                        bound,
                    });
                }
            }
        }
        let scalars =
            ScalarLayout::words(&scalar_parameters).map_err(|reason| InvariantReport(reason))?;
        // Alias rules: shared-read parameter ranges may overlap;
        // exclusive/owned parameter ranges are disjoint from other live
        // parameter ranges; results are distinct from parameters and each
        // other; distinct representation planes are disjoint.
        let mut alias_rules = Vec::new();
        for index in 0..parameter_bindings.len() {
            for other in index + 1..parameter_bindings.len() {
                let (_, left_access, left_binding) = &parameter_bindings[index];
                let (_, right_access, right_binding) = &parameter_bindings[other];
                let both_shared = matches!(
                    (left_access, right_access),
                    (ParamOwnership::Shared, ParamOwnership::Shared)
                );
                alias_rules.push(if both_shared {
                    AliasRule::MayOverlap {
                        left: *left_binding,
                        right: *right_binding,
                    }
                } else {
                    AliasRule::MustDisjoint {
                        left: *left_binding,
                        right: *right_binding,
                    }
                });
            }
        }
        for (_, _, parameter_binding) in &parameter_bindings {
            for buffer in &buffers {
                if buffer.binding != *parameter_binding {
                    alias_rules.push(AliasRule::MustDisjoint {
                        left: *parameter_binding,
                        right: buffer.binding,
                    });
                }
            }
        }
        Ok(AbiSkeleton {
            buffers,
            scalars,
            results,
            alias_rules,
        })
    }

    fn storage_bytes(&self, template: PhysicalStorageTemplateId) -> Result<u64, InvariantReport> {
        let storage = self
            .family
            .storages
            .get(template)
            .ok_or_else(|| InvariantReport(format!("storage#{template:?} is absent")))?;
        self.values.eval(&storage.bytes)
    }

    fn resolved_storage(
        &mut self,
        template: PhysicalStorageTemplateId,
    ) -> Result<ResolvedStorageId, InvariantReport> {
        if let Some(id) = self.storage_ids.get(&template) {
            return Ok(*id);
        }
        let storage_template = self
            .family
            .storages
            .get(template)
            .ok_or_else(|| InvariantReport(format!("storage#{template:?} is absent")))?;
        let bytes = self.values.eval(&storage_template.bytes)?;
        let id = ResolvedStorageId(self.next_storage);
        self.next_storage += 1;
        let placement = match storage_template.scope {
            StorageScope::Abi => ResolvedStoragePlacement::Abi {
                binding: self.abi_bindings.get(&template).copied().ok_or_else(|| {
                    let leaves = self
                        .family
                        .abi_storage_paths
                        .iter()
                        .filter_map(|(leaf, candidate)| (*candidate == template).then_some(leaf))
                        .collect::<Vec<_>>();
                    InvariantReport(format!(
                        "ABI storage#{template:?} has no binding; registered leaves: {leaves:?}"
                    ))
                })?,
            },
            StorageScope::DeviceArena => ResolvedStoragePlacement::Arena {
                offset: if bytes == 0 {
                    0
                } else {
                    self.offsets.get(&template).copied().ok_or_else(|| {
                        InvariantReport(format!("storage#{template:?} has no solved arena offset"))
                    })?
                },
            },
            StorageScope::Workgroup => ResolvedStoragePlacement::Workgroup,
            StorageScope::Participant => ResolvedStoragePlacement::Participant,
        };
        let layout = D::resolve_layout(&storage_template.layout, &self.values)?;
        let resolved = ResolvedStorage {
            id,
            placement,
            replication: storage_template.replication,
            bytes,
            alignment: storage_template.alignment,
            layout,
        };
        self.storage_ids.insert(template, id);
        self.resolved_storages.push((id, resolved));
        Ok(id)
    }

    fn resolved_slot(
        &mut self,
        slot: ExecutorScalarSlotId,
    ) -> Result<ResolvedExecutorScalarId, InvariantReport> {
        if let Some(id) = self.slot_ids.get(&slot) {
            return Ok(*id);
        }
        let id = ResolvedExecutorScalarId(self.next_slot_resolved);
        self.next_slot_resolved += 1;
        self.slot_ids.insert(slot, id);
        Ok(id)
    }

    // -- boundary environments --------------------------------------------------

    fn resolve_body(
        &mut self,
        choice: ChoiceId,
        environment: Option<BoundaryEnvironment>,
    ) -> Result<ResolvedPlanBody<D>, InvariantReport> {
        let (schedule_template, discharged, cost, numerical, declared_status_fields) = {
            let alternative = self.selected_alternative(choice)?;
            (
                alternative.schedule.clone(),
                alternative.obligations.discharged.clone(),
                alternative.cost.clone(),
                alternative.numerical.clone(),
                alternative.obligations.declared_status_fields.clone(),
            )
        };
        let _ = numerical;
        // Status fields: runtime-checked obligations and strategy-declared
        // precondition fields.
        for obligation in discharged.iter() {
            if let Some(field) = obligation.runtime_checked {
                self.status_fields.push(StatusField {
                    id: field,
                    node: obligation.node.clone(),
                    index: obligation.index,
                    predicate: obligation.predicate.clone(),
                });
            }
        }
        for field in declared_status_fields.iter() {
            self.status_fields.push(StatusField {
                id: *field,
                node: NodeRef {
                    region: Vec::new(),
                    node: NodeId(u32::MAX),
                },
                index: usize::MAX,
                predicate: None,
            });
        }
        self.cost = self
            .cost
            .checked_add(self.values.eval(&cost)?)
            .ok_or_else(|| InvariantReport("estimated cost total overflows u64".into()))?;
        let environment = match environment {
            Some(environment) => environment,
            None => self.root_environment()?,
        };
        let schedule = self.resolve_schedule(&schedule_template, &environment)?;
        Ok(ResolvedPlanBody {
            boundary: ResolvedBoundary {
                inputs: environment.inputs,
                results: environment.results,
                states: environment.states,
            },
            schedule,
        })
    }

    /// The root boundary: ABI-backed transports for entry interface leaves,
    /// keyed by param-qualified boundary leaf.
    fn root_environment(&mut self) -> Result<BoundaryEnvironment, InvariantReport> {
        let interface = self.family.choices[self.family.entry].interface.clone();
        let mut environment = BoundaryEnvironment::default();
        for (ordinal, parameter) in interface.params.iter().enumerate() {
            let leaves =
                canonical_leaves(&parameter.ty).map_err(|reason| InvariantReport(reason))?;
            for (path, leaf) in leaves {
                let key = BoundaryLeaf::Input {
                    param: ordinal as u32,
                    leaf: path.clone(),
                };
                let transport = match leaf {
                    Leaf::Scalar(dtype) => ResolvedTransport::ExecutorScalar(
                        self.resolve_abi_scalar(&key, None, dtype)?,
                    ),
                    Leaf::Index(_) => ResolvedTransport::ExecutorScalar(self.resolve_abi_scalar(
                        &key,
                        None,
                        DType::I32,
                    )?),
                    Leaf::Range(_) => ResolvedTransport::Tuple(
                        NonEmpty::new(vec![
                            ResolvedTransport::ExecutorScalar(self.resolve_abi_scalar(
                                &key,
                                Some(seismic_lang::abi::RangeEndpoint::Start),
                                DType::I32,
                            )?),
                            ResolvedTransport::ExecutorScalar(self.resolve_abi_scalar(
                                &key,
                                Some(seismic_lang::abi::RangeEndpoint::End),
                                DType::I32,
                            )?),
                        ])
                        .expect("a range has two scalar fields"),
                    ),
                    Leaf::Tensor(_) => {
                        let template = self
                            .family
                            .abi_storage_paths
                            .get(&key)
                            .copied()
                            .ok_or_else(|| {
                                InvariantReport(format!(
                                    "the root ABI has no storage for leaf {key:?}"
                                ))
                            })?;
                        let storage = self.resolved_storage(template)?;
                        // Writability follows the parameter ownership: a
                        // shared borrow reads; an owned or exclusively
                        // borrowed tensor is a write path.
                        let access = match parameter.ownership {
                            ParamOwnership::Shared | ParamOwnership::Value => Access::Shared,
                            ParamOwnership::Owned | ParamOwnership::Exclusive => Access::Exclusive,
                        };
                        ResolvedTransport::Storage(
                            NonEmpty::new(vec![ResolvedStorageView {
                                storage,
                                access,
                                transform: ViewTransform::Identity,
                            }])
                            .expect("one plane"),
                        )
                    }
                };
                environment.inputs.insert(key, transport);
            }
        }
        // Result leaves: tensor results in ABI result buffers, scalar results
        // in the compiler-owned result scalar block.
        let result_leaves = canonical_leaves(&interface.result).map_err(InvariantReport)?;
        for (path, leaf) in result_leaves {
            let key = BoundaryLeaf::Result { leaf: path.clone() };
            let transport = match leaf {
                Leaf::Tensor(_) => {
                    let template = self
                        .family
                        .abi_storage_paths
                        .get(&key)
                        .copied()
                        .ok_or_else(|| {
                            InvariantReport(format!(
                                "the root ABI has no result storage for leaf {key:?}"
                            ))
                        })?;
                    let storage = self.resolved_storage(template)?;
                    ResolvedTransport::Storage(
                        NonEmpty::new(vec![ResolvedStorageView {
                            storage,
                            access: Access::Exclusive,
                            transform: ViewTransform::Identity,
                        }])
                        .expect("one plane"),
                    )
                }
                Leaf::Scalar(dtype) => {
                    ResolvedTransport::ExecutorScalar(self.resolve_abi_scalar(&key, None, dtype)?)
                }
                Leaf::Index(_) => ResolvedTransport::ExecutorScalar(self.resolve_abi_scalar(
                    &key,
                    None,
                    DType::I32,
                )?),
                Leaf::Range(_) => ResolvedTransport::Tuple(
                    NonEmpty::new(vec![
                        ResolvedTransport::ExecutorScalar(self.resolve_abi_scalar(
                            &key,
                            Some(seismic_lang::abi::RangeEndpoint::Start),
                            DType::I32,
                        )?),
                        ResolvedTransport::ExecutorScalar(self.resolve_abi_scalar(
                            &key,
                            Some(seismic_lang::abi::RangeEndpoint::End),
                            DType::I32,
                        )?),
                    ])
                    .expect("a range has two scalar fields"),
                ),
            };
            environment.results.insert(key, transport);
        }
        Ok(environment)
    }

    fn resolve_transport(
        &mut self,
        transport: &TransportTemplate,
        environment: &BoundaryEnvironment,
    ) -> Result<ResolvedTransport, InvariantReport> {
        Ok(match transport {
            TransportTemplate::Void => ResolvedTransport::Void,
            TransportTemplate::Kernel(_template) => {
                let id = ResolvedKernelValueId(self.next_kernel_value);
                self.next_kernel_value += 1;
                ResolvedTransport::Kernel(id)
            }
            TransportTemplate::ExecutorScalar(scalar) => {
                ResolvedTransport::ExecutorScalar(self.resolve_executor_scalar(scalar)?)
            }
            TransportTemplate::Storage(views) => ResolvedTransport::Storage(
                NonEmpty::new(
                    views
                        .iter()
                        .map(|view| {
                            Ok::<_, InvariantReport>(ResolvedStorageView {
                                storage: self.resolved_storage(view.storage)?,
                                access: view.access,
                                transform: view.transform.clone(),
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )
                .expect("one or more planes"),
            ),
            TransportTemplate::Tuple(items) => ResolvedTransport::Tuple(
                NonEmpty::new(
                    items
                        .iter()
                        .map(|item| self.resolve_transport(item, environment))
                        .collect::<Result<Vec<_>, _>>()?,
                )
                .expect("a tuple is nonempty"),
            ),
            TransportTemplate::Boundary(leaf) => {
                let found = match leaf {
                    BoundaryLeaf::Input { .. } => environment.inputs.get(leaf),
                    BoundaryLeaf::Result { .. } => environment.results.get(leaf),
                };
                found.cloned().ok_or_else(|| {
                    InvariantReport(format!(
                        "boundary placeholder leaf {leaf:?} has no caller transport"
                    ))
                })?
            }
        })
    }

    fn resolve_executor_scalar(
        &mut self,
        scalar: &ExecutorScalarTemplate,
    ) -> Result<ResolvedExecutorScalar, InvariantReport> {
        Ok(match &scalar.source {
            ExecutorScalarSource::Abi { leaf, endpoint } => {
                self.resolve_abi_scalar(leaf, *endpoint, scalar.dtype)?
            }
            ExecutorScalarSource::Slot(slot) => ResolvedExecutorScalar::Slot {
                slot: self.resolved_slot(*slot)?,
                dtype: scalar.dtype,
            },
            ExecutorScalarSource::Computed(expression) => ResolvedExecutorScalar::Computed {
                expr: self.resolve_computed_scalar(expression)?,
                dtype: scalar.dtype,
            },
        })
    }

    fn resolve_abi_scalar(
        &self,
        leaf: &BoundaryLeaf,
        endpoint: Option<seismic_lang::abi::RangeEndpoint>,
        dtype: DType,
    ) -> Result<ResolvedExecutorScalar, InvariantReport> {
        match leaf {
            BoundaryLeaf::Input { param, leaf: path } => {
                let parameter = self.family.choices[self.family.entry]
                    .interface
                    .params
                    .get(*param as usize)
                    .ok_or_else(|| InvariantReport(format!("ABI parameter#{param} is absent")))?;
                let name = abi_scalar_name(&parameter.name, path);
                let layout = self.abi_scalars.as_ref().ok_or_else(|| {
                    InvariantReport("the scalar ABI was not built before resolution".into())
                })?;
                let field = layout
                    .fields
                    .iter()
                    .find(|field| {
                        field.parameter.name == name
                            && match (endpoint, field.parameter.range.as_ref()) {
                                (None, None) => true,
                                (Some(wanted), Some(range)) => range.endpoint == wanted,
                                _ => false,
                            }
                    })
                    .ok_or_else(|| {
                        InvariantReport(format!(
                    "ABI scalar parameter#{param} leaf {path} endpoint {endpoint:?} has no field"
                ))
                    })?;
                Ok(ResolvedExecutorScalar::Abi {
                    path: path.clone(),
                    offset: field.offset as u64,
                    dtype,
                })
            }
            BoundaryLeaf::Result { leaf: path } => {
                let field = self
                    .abi_results
                    .iter()
                    .find_map(|binding| match (binding, endpoint) {
                        (
                            ResultBinding::Scalar {
                                path: candidate,
                                field,
                                dtype: actual,
                            },
                            None,
                        ) if candidate == path && *actual == dtype => Some(*field),
                        (
                            ResultBinding::Range {
                                path: candidate,
                                start,
                                ..
                            },
                            Some(seismic_lang::abi::RangeEndpoint::Start),
                        ) if candidate == path => Some(*start),
                        (
                            ResultBinding::Range {
                                path: candidate,
                                end,
                                ..
                            },
                            Some(seismic_lang::abi::RangeEndpoint::End),
                        ) if candidate == path => Some(*end),
                        _ => None,
                    })
                    .ok_or_else(|| {
                        InvariantReport(format!(
                    "ABI result scalar leaf {path} endpoint {endpoint:?} has no result field"
                ))
                    })?;
                Ok(ResolvedExecutorScalar::Result {
                    path: path.clone(),
                    field,
                    dtype,
                })
            }
        }
    }

    /// Resolve one computed control-scalar expression to a retained
    /// execution expression. Plan parameters evaluate to their solved
    /// values; runtime extents stay runtime.
    fn resolve_computed_scalar(
        &mut self,
        expression: &ExecutorComputedScalar,
    ) -> Result<ExecutionExpr, InvariantReport> {
        Ok(match expression {
            ExecutorComputedScalar::Abi { path, dtype } => ExecutionExpr::AbiScalar {
                path: path.clone(),
                dtype: *dtype,
            },
            ExecutorComputedScalar::Slot(slot) => {
                ExecutionExpr::ExecutorScalar(self.resolved_slot(*slot)?)
            }
            ExecutorComputedScalar::Extent(id) => ExecutionExpr::Extent(*id),
            ExecutorComputedScalar::Param(name) => {
                let value = self.values.symbols.get(name).copied().ok_or_else(|| {
                    InvariantReport(format!(
                        "the computed control scalar references unsolved parameter `{name}`"
                    ))
                })?;
                ExecutionExpr::Const(u64::try_from(value).map_err(|_| {
                    InvariantReport(format!("the solved parameter `{name}` is negative"))
                })?)
            }
            ExecutorComputedScalar::Const(value) => {
                ExecutionExpr::Const(u64::try_from(*value).map_err(|_| {
                    InvariantReport("a computed control-scalar constant is negative".into())
                })?)
            }
            ExecutorComputedScalar::CeilDiv(left, right) => ExecutionExpr::CeilDiv(
                Box::new(self.resolve_computed_scalar(left)?),
                Box::new(self.resolve_computed_scalar(right)?),
            ),
            ExecutorComputedScalar::Add(left, right) => ExecutionExpr::Add(
                Box::new(self.resolve_computed_scalar(left)?),
                Box::new(self.resolve_computed_scalar(right)?),
            ),
            ExecutorComputedScalar::Sub(left, right) => ExecutionExpr::Sub(
                Box::new(self.resolve_computed_scalar(left)?),
                Box::new(self.resolve_computed_scalar(right)?),
            ),
            ExecutorComputedScalar::Mul(left, right) => ExecutionExpr::Mul(
                Box::new(self.resolve_computed_scalar(left)?),
                Box::new(self.resolve_computed_scalar(right)?),
            ),
        })
    }

    /// A control scalar (predicate/range endpoint): an executor-scalar
    /// transport, possibly reached through a boundary placeholder.
    fn resolve_control_scalar(
        &mut self,
        transport: &TransportTemplate,
        environment: &BoundaryEnvironment,
    ) -> Result<ResolvedExecutorScalar, InvariantReport> {
        match self.resolve_transport(transport, environment)? {
            ResolvedTransport::ExecutorScalar(scalar) => Ok(scalar),
            _ => Err(InvariantReport(
                "a control value must transport through an executor scalar".into(),
            )),
        }
    }

    fn resolve_state_transport(
        &mut self,
        transport: &StateTransportTemplate,
        environment: &BoundaryEnvironment,
    ) -> Result<ResolvedStateTransport, InvariantReport> {
        Ok(match transport {
            StateTransportTemplate::Storage(template) => ResolvedStateTransport {
                storage: self.resolved_storage(*template)?,
            },
            StateTransportTemplate::Boundary(leaf) => {
                environment.states.get(leaf).cloned().ok_or_else(|| {
                    InvariantReport(format!(
                        "boundary state leaf {leaf:?} has no caller transport"
                    ))
                })?
            }
        })
    }

    // -- one structured schedule authority --------------------------------------

    fn resolve_schedule(
        &mut self,
        schedule: &ScheduleTemplate<D>,
        environment: &BoundaryEnvironment,
    ) -> Result<ResolvedSchedule<D>, InvariantReport> {
        let mut steps = Vec::new();
        for step in schedule.steps.iter() {
            steps.push(self.resolve_step(step, environment)?);
        }
        Ok(ResolvedSchedule {
            steps: NonEmpty::new(steps)
                .ok_or_else(|| InvariantReport("a resolved schedule has no step".into()))?,
        })
    }

    fn resolve_step(
        &mut self,
        step: &ScheduleStepTemplate<D>,
        environment: &BoundaryEnvironment,
    ) -> Result<ResolvedStep<D>, InvariantReport> {
        Ok(match step {
            ScheduleStepTemplate::Launch(launch) => {
                ResolvedStep::Launch(self.resolve_launch(launch, environment)?)
            }
            ScheduleStepTemplate::Call(call) => {
                ResolvedStep::Call(self.resolve_call(call, environment)?)
            }
            ScheduleStepTemplate::If(if_step) => {
                let condition =
                    self.resolve_control_scalar(&if_step.condition.value, environment)?;
                let then_schedule = self.resolve_schedule(&if_step.then_schedule, environment)?;
                let else_schedule = self.resolve_schedule(&if_step.else_schedule, environment)?;
                let mut joins = Vec::new();
                for join in &if_step.joins {
                    joins.push(match join {
                        PhysicalJoinTemplate::Value {
                            then,
                            else_branch,
                            joined,
                        } => ResolvedJoin::Value {
                            then: self.resolve_transport(then, environment)?,
                            else_branch: self.resolve_transport(else_branch, environment)?,
                            joined: self.resolve_transport(joined, environment)?,
                        },
                        PhysicalJoinTemplate::State { storage } => ResolvedJoin::State {
                            storage: self.resolved_storage(*storage)?,
                        },
                    });
                }
                ResolvedStep::If(ResolvedScheduleIf {
                    condition,
                    then_schedule,
                    else_schedule,
                    joins,
                })
            }
            ScheduleStepTemplate::Repeat(repeat) => {
                let start = self.resolve_control_scalar(&repeat.range.start, environment)?;
                let end = self.resolve_control_scalar(&repeat.range.end, environment)?;
                let bound = self.resolve_bound(&repeat.range.bound)?;
                let binder = self.resolved_slot(repeat.binder.slot)?;
                let body = self.resolve_schedule(&repeat.body, environment)?;
                let carried = repeat
                    .carried
                    .iter()
                    .map(|carry| self.resolve_transport(&carry.transport, environment))
                    .collect::<Result<Vec<_>, _>>()?;
                ResolvedStep::Repeat(ResolvedScheduleRepeat {
                    logical_kind: repeat.logical_kind,
                    range: ResolvedExecutorRange { start, end, bound },
                    binder,
                    body,
                    carried,
                })
            }
        })
    }

    fn resolve_bound(&mut self, bound: &ExtentExpr) -> Result<ExecutionExpr, InvariantReport> {
        Ok(match bound {
            ExtentExpr::Static(n) => ExecutionExpr::Const(*n),
            ExtentExpr::Sym(sym) => ExecutionExpr::Const(self.values.eval(sym)?),
            ExtentExpr::Runtime(id) => ExecutionExpr::Extent(*id),
        })
    }

    fn resolve_call(
        &mut self,
        call: &CallTemplate,
        environment: &BoundaryEnvironment,
    ) -> Result<ResolvedCall<D>, InvariantReport> {
        // The child environment from parent transports.
        let mut child = BoundaryEnvironment::default();
        for (path, transport) in &call.boundary.inputs {
            child.inputs.insert(
                path.clone(),
                self.resolve_transport(transport, environment)?,
            );
        }
        for (path, transport) in &call.boundary.results {
            child.results.insert(
                path.clone(),
                self.resolve_transport(transport, environment)?,
            );
        }
        for (path, transport) in &call.boundary.states {
            child.states.insert(
                path.clone(),
                self.resolve_state_transport(transport, environment)?,
            );
        }
        let body = self.resolve_body(call.choice, Some(child.clone()))?;
        Ok(ResolvedCall {
            call: call.call.clone(),
            choice: call.choice,
            boundary: ResolvedBoundary {
                inputs: child.inputs,
                results: child.results,
                states: child.states,
            },
            body: Box::new(body),
        })
    }

    fn resolve_launch(
        &mut self,
        launch: &LaunchTemplate<D>,
        environment: &BoundaryEnvironment,
    ) -> Result<ResolvedLaunch<D>, InvariantReport> {
        let id = ResolvedLaunchId(self.next_launch);
        self.next_launch += 1;
        // Geometry: work items are the retained product of the iteration
        // extents (runtime extents stay runtime, never capacities); axis 0
        // covers ceil(work items / participants), axes 1-2 are one.
        let (work_items, participants_expr) = launch_geometry(launch, &self.values)?;
        let workgroups = [
            ExecutionExpr::CeilDiv(
                Box::new(work_items.clone()),
                Box::new(participants_expr.clone()),
            ),
            ExecutionExpr::Const(1),
            ExecutionExpr::Const(1),
        ];
        let participants_per_workgroup = [
            participants_expr,
            ExecutionExpr::Const(1),
            ExecutionExpr::Const(1),
        ];
        let mut bindings = Vec::new();
        let mut direct_bindings = 0u32;
        for group in &launch.bindings {
            let members = NonEmpty::new(
                group
                    .members
                    .iter()
                    .map(|member| {
                        Ok::<_, InvariantReport>(ResolvedBindingMember {
                            storage: self.resolved_storage(member.storage)?,
                            access: member.access,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
            .expect("a group is nonempty");
            if group.kind == BindingGroupKind::Direct {
                direct_bindings += members.len() as u32;
            }
            bindings.push(ResolvedBindingGroup {
                kind: group.kind,
                slot: group.slot,
                members,
            });
        }
        self.direct_bindings = self.direct_bindings.max(direct_bindings);
        let mut kernel_steps = Vec::new();
        let mut value_map: BTreeMap<KernelValueTemplateId, ResolvedKernelValueId> = BTreeMap::new();
        let mut workgroup_storage = Vec::new();
        let mut participant_storage = Vec::new();
        for template in launch.kernel.workgroup_storage.iter() {
            workgroup_storage.push(self.resolved_storage(*template)?);
        }
        for template in launch.kernel.participant_storage.iter() {
            participant_storage.push(self.resolved_storage(*template)?);
        }
        let mut workgroup_bytes = 0u64;
        for storage in &workgroup_storage {
            let bytes = self
                .resolved_storages
                .iter()
                .find(|(id, _)| id == storage)
                .map(|(_, resolved)| resolved.bytes)
                .ok_or_else(|| {
                    InvariantReport(format!(
                        "resolved workgroup storage#{:?} is absent from the storage table",
                        storage
                    ))
                })?;
            workgroup_bytes = workgroup_bytes
                .checked_add(bytes)
                .ok_or_else(|| InvariantReport("workgroup bytes overflow u64".into()))?;
        }
        let mut private_bytes = 0u64;
        for storage in &participant_storage {
            let bytes = self
                .resolved_storages
                .iter()
                .find(|(id, _)| id == storage)
                .map(|(_, resolved)| resolved.bytes)
                .ok_or_else(|| {
                    InvariantReport(format!(
                        "resolved participant storage#{:?} is absent from the storage table",
                        storage
                    ))
                })?;
            private_bytes = private_bytes
                .checked_add(bytes)
                .ok_or_else(|| InvariantReport("private bytes overflow u64".into()))?;
        }
        self.max_workgroup_bytes = self.max_workgroup_bytes.max(workgroup_bytes);
        self.max_private_bytes = self.max_private_bytes.max(private_bytes);
        for kernel_step in launch.kernel.steps.iter() {
            kernel_steps.push(match kernel_step {
                KernelStepTemplate::Mapped {
                    iteration,
                    bindings,
                    ops,
                } => {
                    let mut resolved_bindings = Vec::new();
                    for (value, transport) in bindings {
                        let resolved = match transport {
                            TransportTemplate::Kernel(template) => {
                                let next = &mut self.next_kernel_value;
                                let resolved = *value_map.entry(*template).or_insert_with(|| {
                                    let id = ResolvedKernelValueId(*next);
                                    *next += 1;
                                    id
                                });
                                ResolvedTransport::Kernel(resolved)
                            }
                            other => self.resolve_transport(other, environment)?,
                        };
                        // Record scalar sources for runtime extent resolution.
                        if let ResolvedTransport::ExecutorScalar(scalar) = &resolved {
                            self.scalar_exprs.insert(
                                *value,
                                match scalar {
                                    ResolvedExecutorScalar::Abi { path, dtype, .. } => {
                                        ExecutionExpr::AbiScalar {
                                            path: path.clone(),
                                            dtype: *dtype,
                                        }
                                    }
                                    ResolvedExecutorScalar::Result { .. } => {
                                        return Err(InvariantReport(
                                            "a result scalar cannot define a runtime extent".into(),
                                        ));
                                    }
                                    ResolvedExecutorScalar::Slot { slot, .. } => {
                                        ExecutionExpr::ExecutorScalar(*slot)
                                    }
                                    ResolvedExecutorScalar::Computed { expr, .. } => expr.clone(),
                                },
                            );
                        }
                        resolved_bindings.push((*value, resolved));
                    }
                    let resolved_ops = NonEmpty::new(
                        ops.iter()
                            .map(|op| D::resolve_op(op, self))
                            .collect::<Result<Vec<_>, _>>()?,
                    )
                    .ok_or_else(|| InvariantReport("a mapped step has no opcodes".into()))?;
                    ResolvedKernelStep::Mapped {
                        iteration: iteration.clone(),
                        bindings: resolved_bindings,
                        ops: resolved_ops,
                    }
                }
                KernelStepTemplate::Barrier { scope } => {
                    ResolvedKernelStep::Barrier { scope: *scope }
                }
                KernelStepTemplate::Publish { storage } => ResolvedKernelStep::Publish {
                    storage: self.resolved_storage(*storage)?,
                },
                KernelStepTemplate::PublishScalar { slot } => ResolvedKernelStep::PublishScalar {
                    slot: self.resolved_slot(*slot)?,
                },
            });
        }
        // Complete the direct binding groups from the resolved mapped
        // bindings: boundary placeholders that resolved to caller storages
        // were invisible to the template-time group collection, but the
        // resolved launch must bind every storage its kernels touch.
        {
            let mut members: BTreeMap<ResolvedStorageId, AccessMode> = BTreeMap::new();
            for step in kernel_steps.iter() {
                if let ResolvedKernelStep::Mapped { bindings, .. } = step {
                    for (_, transport) in bindings {
                        collect_resolved_storage(&mut members, transport);
                    }
                }
            }
            if !members.is_empty() {
                let mut merged: BTreeMap<ResolvedStorageId, AccessMode> = BTreeMap::new();
                let mut direct: Option<usize> = None;
                for (index, group) in bindings.iter_mut().enumerate() {
                    if !matches!(group.kind, BindingGroupKind::Direct) {
                        continue;
                    }
                    direct = Some(index);
                    for member in group.members.as_slice() {
                        merged.insert(member.storage, member.access);
                    }
                    break;
                }
                for (storage, access) in members {
                    merged.insert(
                        storage,
                        match merged.get(&storage) {
                            Some(existing) if *existing != access => AccessMode::ReadWrite,
                            _ => access,
                        },
                    );
                }
                let mut list: Vec<(ResolvedStorageId, AccessMode)> = merged.into_iter().collect();
                list.sort_by_key(|(storage, _)| storage.0);
                let group_members = NonEmpty::new(
                    list.into_iter()
                        .map(|(storage, access)| ResolvedBindingMember { storage, access })
                        .collect(),
                )
                .expect("members is nonempty");
                match direct {
                    Some(index) => bindings[index].members = group_members,
                    None => bindings.push(ResolvedBindingGroup {
                        kind: BindingGroupKind::Direct,
                        slot: 0,
                        members: group_members,
                    }),
                }
            }
        }
        Ok(ResolvedLaunch {
            id,
            geometry: ResolvedDispatchGeometry {
                workgroups,
                participants_per_workgroup,
            },
            work_items,
            bindings,
            value_map,
            kernel: ResolvedKernel {
                workgroup_storage,
                participant_storage,
                steps: NonEmpty::new(kernel_steps)
                    .ok_or_else(|| InvariantReport("a resolved kernel has no step".into()))?,
                resources: ResolvedKernelResources {
                    workgroup_bytes,
                    private_bytes_per_participant: private_bytes,
                    capabilities: BTreeSet::new(),
                },
            },
        })
    }

    /// Resolve a retained runtime scalar expression (a runtime extent value).
    fn resolve_scalar_expr(
        &self,
        expr: &RuntimeScalarExpr,
    ) -> Result<ExecutionExpr, InvariantReport> {
        Ok(match expr {
            RuntimeScalarExpr::Const(value) => ExecutionExpr::Const(
                u64::try_from(*value)
                    .map_err(|_| InvariantReport("runtime scalar constant is negative".into()))?,
            ),
            RuntimeScalarExpr::Value(value) => {
                self.scalar_exprs.get(value).cloned().ok_or_else(|| {
                    InvariantReport(format!(
                        "runtime scalar value#{} has no resolved source",
                        value.0
                    ))
                })?
            }
            RuntimeScalarExpr::Extent(id) => ExecutionExpr::Extent(*id),
            RuntimeScalarExpr::Add(left, right) => ExecutionExpr::Add(
                Box::new(self.resolve_scalar_expr(left)?),
                Box::new(self.resolve_scalar_expr(right)?),
            ),
            RuntimeScalarExpr::Sub(left, right) => ExecutionExpr::Sub(
                Box::new(self.resolve_scalar_expr(left)?),
                Box::new(self.resolve_scalar_expr(right)?),
            ),
            RuntimeScalarExpr::Mul(left, right) => ExecutionExpr::Mul(
                Box::new(self.resolve_scalar_expr(left)?),
                Box::new(self.resolve_scalar_expr(right)?),
            ),
            RuntimeScalarExpr::Div(left, right) => ExecutionExpr::Div(
                Box::new(self.resolve_scalar_expr(left)?),
                Box::new(self.resolve_scalar_expr(right)?),
            ),
            RuntimeScalarExpr::Rem(left, right) => ExecutionExpr::Rem(
                Box::new(self.resolve_scalar_expr(left)?),
                Box::new(self.resolve_scalar_expr(right)?),
            ),
        })
    }
}

struct AbiSkeleton {
    buffers: Vec<BufferBinding>,
    scalars: ScalarLayout,
    results: Vec<ResultBinding>,
    alias_rules: Vec<AliasRule>,
}

fn abi_scalar_name(parameter: &str, path: &ValuePath) -> String {
    if path.0.is_empty() {
        parameter.to_string()
    } else {
        format!("{parameter}{}", path)
    }
}

fn extent_bound(extent: &ExtentExpr) -> Result<u64, InvariantReport> {
    match extent {
        ExtentExpr::Static(n) => Ok(*n),
        ExtentExpr::Sym(_) => Err(InvariantReport(
            "an unspecialized symbolic extent reached the ABI".into(),
        )),
        ExtentExpr::Runtime(_) => Err(InvariantReport(
            "a runtime extent has no static ABI bound".into(),
        )),
    }
}

/// Convert a retained runtime scalar expression of a linear total into an
/// execution expression. `Value` refs cannot occur (totals are products of
/// constants and runtime extents); reaching one is a compiler bug.
fn runtime_total_expr(expr: &RuntimeScalarExpr) -> Result<ExecutionExpr, InvariantReport> {
    Ok(match expr {
        RuntimeScalarExpr::Const(value) => ExecutionExpr::Const(
            u64::try_from(*value)
                .map_err(|_| InvariantReport("a runtime total constant is negative".into()))?,
        ),
        RuntimeScalarExpr::Extent(id) => ExecutionExpr::Extent(*id),
        RuntimeScalarExpr::Value(value) => {
            return Err(InvariantReport(format!(
                "runtime total references graph value#{}",
                value.0
            )));
        }
        RuntimeScalarExpr::Add(left, right) => ExecutionExpr::Add(
            Box::new(runtime_total_expr(left)?),
            Box::new(runtime_total_expr(right)?),
        ),
        RuntimeScalarExpr::Sub(left, right) => ExecutionExpr::Sub(
            Box::new(runtime_total_expr(left)?),
            Box::new(runtime_total_expr(right)?),
        ),
        RuntimeScalarExpr::Mul(left, right) => ExecutionExpr::Mul(
            Box::new(runtime_total_expr(left)?),
            Box::new(runtime_total_expr(right)?),
        ),
        RuntimeScalarExpr::Div(left, right) => ExecutionExpr::Div(
            Box::new(runtime_total_expr(left)?),
            Box::new(runtime_total_expr(right)?),
        ),
        RuntimeScalarExpr::Rem(left, right) => ExecutionExpr::Rem(
            Box::new(runtime_total_expr(left)?),
            Box::new(runtime_total_expr(right)?),
        ),
    })
}

/// Retained launch geometry: the overflow-checked row-major total of the
/// mapped iteration (runtime domains retain their exact runtime product,
/// never the capacity bound) and the solved participant count.
fn launch_geometry<D: ExecutableDialect>(
    launch: &LaunchTemplate<D>,
    values: &PlanValues,
) -> Result<(ExecutionExpr, ExecutionExpr), InvariantReport> {
    let participants = values.eval(&launch.geometry.participants_per_workgroup[0])?;
    let mut work_items: Option<ExecutionExpr> = None;
    for kernel_step in launch.kernel.steps.iter() {
        if let KernelStepTemplate::Mapped { iteration, .. } = kernel_step {
            let product = match &iteration.total {
                crate::dispatch::LinearTotal::Static(total) => ExecutionExpr::Const(*total),
                crate::dispatch::LinearTotal::Runtime { product, .. } => {
                    runtime_total_expr(product)?
                }
            };
            work_items = Some(match work_items {
                None => product,
                Some(existing) => ExecutionExpr::Add(Box::new(existing), Box::new(product)),
            });
            break;
        }
    }
    let work_items =
        work_items.ok_or_else(|| InvariantReport("a launch has no mapped iteration".into()))?;
    Ok((work_items, ExecutionExpr::Const(participants)))
}
