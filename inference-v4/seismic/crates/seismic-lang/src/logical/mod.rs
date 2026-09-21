//! The logical program.
//!
//! `LogicalProgram` is occurrence-qualified implementation choices plus
//! hierarchical SSA task graphs. It is built by one forward pass over the
//! checked body (`normalize`): constructors create outputs and
//! dependencies immediately, so nothing is reconstructed by recursive scanning
//! afterwards. Validity is enforced during construction by the private
//! `GraphBuilder` (`builder`), whose consuming `seal` is the only constructor
//! of a `TaskGraph`; a `LogicalProgram` exists only after every graph sealed.
//! `verify` re-checks cached data against the same invariants.
//!
//! Logical storage and views carry semantic shape, owner, initialization,
//! access and transforms only — no address space, byte stride, pointer,
//! allocation or physical tile. The ownership/effect dependency is the
//! state-token chain itself: every graph value and state token has exactly one
//! region-parameter or node origin, reads consume the current state, and
//! writes/atomics/moves/exclusive calls consume one state and produce the next.
//!
//! Every value has one exhaustive kind (`value::GraphValueKind`); a tensor is
//! computed or a view whose base is one logical storage or one computed
//! tensor value (`value::TensorSource`, `value::ViewBase`). Every graph owns
//! one explicit boundary contract (`boundary::LogicalBoundary`) keyed by the
//! canonical leaves of its checked interface; a call instantiates the callee's
//! contract with the caller's identities (`boundary::CallBoundary`).

use crate::intrinsics::{CapabilityId, IntrinsicId, PrimitiveId, ReduceOp};
use crate::sir::{DefId, IntrinsicUse, LoopKind, Mode, ParamOwnership, Program};
use crate::span::Span;
use crate::types::{DType, ExtentExpr, NonEmpty, RuntimeExtentId, TensorType, ValueType};
use std::collections::{BTreeMap, BTreeSet};

mod builder;
mod normalize;
mod verify;

pub mod boundary;
pub mod specialization;
pub mod value;

pub use boundary::{
    BoundaryLeaf, CallBoundary, CallInput, LogicalBoundary, LogicalBoundaryInput,
    LogicalBoundaryResult,
};
pub use normalize::{ApplicabilityReport, LogicalConstructionError, OccurrenceRejection};
pub use specialization::{ShapeField, ShapeFieldId, SpecializationDomain};
pub use value::{GraphValue, GraphValueKind, LogicalStorageOwner, TensorSource, ViewBase};

/// One definition of the checked program, as an implementation alternative.
pub type DefinitionId = DefId;

// ---------------------------------------------------------------------------
// Identity and target
// ---------------------------------------------------------------------------

/// Identity of the logical program: changing semantics (source, registry
/// revision, entry, target, or the specialization domain) changes this key,
/// and with it every downstream plan, cache and evidence identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LogicalIdentity(pub [u8; 32]);

/// The effective target the program is compiled for: backend identity plus the
/// exact effective capability environment fingerprint.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EffectiveTargetIdentity {
    pub backend: String,
    pub capability_fingerprint: String,
}

/// The typed call interface of one occurrence: canonical parameter types
/// (substituted to the occurrence's concrete shapes/elements) and result.
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionInterface {
    pub name: String,
    pub params: Vec<InterfaceParam>,
    pub result: ValueType,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InterfaceParam {
    pub name: String,
    pub mode: Mode,
    pub ownership: ParamOwnership,
    pub ty: ValueType,
}

// ---------------------------------------------------------------------------
// Ids and indexed vectors
// ---------------------------------------------------------------------------

macro_rules! identity {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub u32);

        impl $name {
            /// Positional index into the owning list.
            pub fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

identity!(GraphValueId);
identity!(StateTokenId);
identity!(LogicalStorageId);
identity!(LogicalViewId);
identity!(NodeId);
identity!(GraphId);
identity!(ChoiceId);
identity! {
    /// Position of one parameter in a region's parameter list.
    RegionParameterId
}

/// One nested logical region within a task graph. Region paths are logical
/// identity; physical planning may reference them but may not extend them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RegionStep {
    IfThen(NodeId),
    IfElse(NodeId),
    LoopBody(NodeId),
}

impl RegionStep {
    pub fn node(self) -> NodeId {
        match self {
            RegionStep::IfThen(node) | RegionStep::IfElse(node) | RegionStep::LoopBody(node) => {
                node
            }
        }
    }
}

pub type RegionPath = Vec<RegionStep>;

/// One node within one task graph. Node ids are region-local.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeRef {
    pub region: RegionPath,
    pub node: NodeId,
}

/// Stable identity of one logical implementation graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GraphKey {
    pub choice: ChoiceId,
    pub logical_alternative: u32,
}

/// One program-wide logical node identity.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GraphNodeRef {
    pub graph: GraphKey,
    pub node: NodeRef,
}

/// One program-wide logical value identity. Graph value ids are unique across
/// the whole program (one allocator serves every graph).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GraphValueRef {
    pub graph: GraphKey,
    pub value: GraphValueId,
}

/// One physical kernel leaf of a program-wide logical value. `path` names the
/// canonical semantic leaf. A range remains one semantic ABI leaf but expands
/// to two independently addressable kernel scalars, distinguished explicitly
/// by `endpoint`; all other leaf kinds use `None`. Tensors remain one kernel
/// leaf even when their representation has multiple storage planes.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GraphValueLeafRef {
    pub value: GraphValueRef,
    pub path: crate::types::ValuePath,
    pub endpoint: Option<crate::abi::RangeEndpoint>,
}
identity! {
    /// Position of one result in a region's result list (also used for the
    /// result ordinals a loop node exposes for its carried slots).
    RegionResultId
}

/// An id-indexed collection. Ids are unique within the owning artifact and
/// may be globally allocated (gaps are legal); iteration is in id order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IdVec<I, T> {
    items: std::collections::BTreeMap<u32, T>,
    marker: std::marker::PhantomData<I>,
}

impl<I: IdIndex, T> IdVec<I, T> {
    /// An indexed vector whose ids start at zero.
    pub fn new(items: Vec<T>) -> IdVec<I, T> {
        let mut out = IdVec {
            items: BTreeMap::new(),
            marker: std::marker::PhantomData,
        };
        for (index, item) in items.into_iter().enumerate() {
            out.items.insert(index as u32, item);
        }
        out
    }

    /// An indexed vector from `(id, item)` pairs (globally allocated ids,
    /// gaps legal).
    pub fn from_iter(entries: impl IntoIterator<Item = (I, T)>) -> IdVec<I, T> {
        IdVec {
            items: entries
                .into_iter()
                .map(|(id, item)| (id.index() as u32, item))
                .collect(),
            marker: std::marker::PhantomData,
        }
    }

    pub fn get(&self, id: I) -> Option<&T> {
        self.items.get(&(id.index() as u32))
    }

    pub fn get_mut(&mut self, id: I) -> Option<&mut T> {
        self.items.get_mut(&(id.index() as u32))
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        self.items.values()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> + '_ {
        self.items.values_mut()
    }

    pub fn ids(&self) -> impl Iterator<Item = I> + '_ {
        self.items.keys().map(|id| I::from_index(*id as usize))
    }

    /// `(id, item)` pairs in id order.
    pub fn entries(&self) -> impl Iterator<Item = (I, &T)> + '_ {
        self.items
            .iter()
            .map(|(id, item)| (I::from_index(*id as usize), item))
    }

    pub fn into_vec(self) -> Vec<T> {
        self.items.into_values().collect()
    }
}

impl<I: IdIndex, T> std::ops::Index<I> for IdVec<I, T> {
    type Output = T;
    fn index(&self, id: I) -> &T {
        &self.items[&(id.index() as u32)]
    }
}

pub trait IdIndex {
    fn from_index(index: usize) -> Self;
    fn index(self) -> usize;
}

macro_rules! id_index {
    ($name:ident) => {
        impl IdIndex for $name {
            fn from_index(index: usize) -> Self {
                $name(index as u32)
            }
            fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

id_index!(GraphValueId);
id_index!(StateTokenId);
id_index!(LogicalStorageId);
id_index!(LogicalViewId);
id_index!(GraphId);
id_index!(ChoiceId);
id_index!(RuntimeExtentId);
id_index!(ShapeFieldId);

impl IdIndex for NodeId {
    fn from_index(index: usize) -> Self {
        NodeId(index as u32)
    }
    fn index(self) -> usize {
        self.0 as usize
    }
}

// ---------------------------------------------------------------------------
// The logical program
// ---------------------------------------------------------------------------

/// The logical program of one entry under one specialization domain on one
/// effective target. Constructed only by `construct`, after every task graph
/// sealed; every table is read through the typed accessors.
#[derive(Clone, Debug)]
pub struct LogicalProgram {
    pub identity: LogicalIdentity,
    pub target: EffectiveTargetIdentity,
    /// The complete, validated binding of every entry shape and element
    /// parameter. The entry name is `domain.entry()`.
    pub domain: SpecializationDomain,
    /// One retained invocation shape field per bounded entry shape
    /// parameter, in interface order. Exact parameters have no field.
    pub shape_fields: IdVec<ShapeFieldId, ShapeField>,
    pub entry_choice: ChoiceId,
    choices: IdVec<ChoiceId, ImplementationChoice>,
    graphs: IdVec<GraphId, TaskGraph>,
    runtime_extents: IdVec<RuntimeExtentId, RuntimeExtent>,
}

impl LogicalProgram {
    pub fn entry(&self) -> &str {
        self.domain.entry()
    }

    pub fn choice(&self, id: ChoiceId) -> &ImplementationChoice {
        &self.choices[id]
    }

    pub fn choices(&self) -> impl Iterator<Item = (ChoiceId, &ImplementationChoice)> + '_ {
        self.choices.entries()
    }

    pub fn graph(&self, id: GraphId) -> &TaskGraph {
        &self.graphs[id]
    }

    pub fn graphs(&self) -> impl Iterator<Item = (GraphId, &TaskGraph)> + '_ {
        self.graphs.entries()
    }

    pub fn runtime_extent(&self, id: RuntimeExtentId) -> &RuntimeExtent {
        &self.runtime_extents[id]
    }

    pub fn runtime_extents(&self) -> impl Iterator<Item = &RuntimeExtent> + '_ {
        self.runtime_extents.iter()
    }

    pub fn shape_field(&self, id: ShapeFieldId) -> &ShapeField {
        &self.shape_fields[id]
    }

    /// Defensive re-validation of cached data. Normal semantic discovery
    /// happens during construction; this only confirms that a stored program
    /// still satisfies every structural invariant.
    pub fn verify(&self) -> Result<(), Vec<String>> {
        verify::verify(self)
    }
}

/// The implementation decision at one call occurrence (the entry included).
/// Every occurrence has its own `ChoiceId`; interned body templates never
/// share occurrence identity, applicability or cost.
#[derive(Clone, Debug)]
pub struct ImplementationChoice {
    pub interface: FunctionInterface,
    pub alternatives: NonEmpty<LogicalAlternative>,
}

/// One applicable semantic implementation of the occurrence's contract.
#[derive(Clone, Debug)]
pub struct LogicalAlternative {
    pub definition: DefinitionId,
    pub kind: ImplementationKind,
    /// The task graph of this definition under the occurrence's
    /// specialization (its local storages are graph-internal; its results
    /// are the values named by the graph's boundary).
    pub graph: GraphId,
    pub required_capabilities: BTreeSet<CapabilityId>,
    /// Authored numerical effects of this alternative. The portable reference
    /// body contributes the registry's reference numerics (capability uses
    /// carry their transfer); a non-reference authored implementation starts
    /// with `Unknown` whole-candidate equivalence. Composition is downstream.
    pub authored_numerical_effects: Vec<NumericalTransfer>,
}

/// Which authored implementation shape an alternative comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImplementationKind {
    /// The portable reference `fn` body (peer semantic alternative on every
    /// target).
    PortableBody,
    /// A backend-specific `fn` body, reachable only from the same backend.
    BackendBody,
    /// A `lower … for target` body, applicable only on that target.
    Lowering,
}

/// Registry-level numerical transfer contributed by one alternative.
pub use crate::intrinsics::NumericalTransfer;

// ---------------------------------------------------------------------------
// Task graphs and regions
// ---------------------------------------------------------------------------

/// One sealed implementation graph. The root region's parameters are the
/// origins of the boundary's input values and entry states, in canonical
/// interface-leaf order; the root region has no positional results — the
/// boundary is the only statement of what the graph returns and which
/// parameter states it leaves behind. Every value, storage, view and state
/// token of the graph is defined exactly once in its table.
#[derive(Clone, Debug)]
pub struct TaskGraph {
    pub choice: ChoiceId,
    /// Ordinal of the alternative within the choice.
    pub alternative: u32,
    pub boundary: LogicalBoundary,
    root: GraphRegion,
    values: IdVec<GraphValueId, GraphValue>,
    storages: IdVec<LogicalStorageId, LogicalStorage>,
    views: IdVec<LogicalViewId, LogicalView>,
    /// The storage every state token of the graph versions.
    states: IdVec<StateTokenId, LogicalStorageId>,
}

impl TaskGraph {
    pub fn root(&self) -> &GraphRegion {
        &self.root
    }

    /// The value with this id. Total: every value id of the graph is defined.
    pub fn value(&self, id: GraphValueId) -> &GraphValue {
        &self.values[id]
    }

    pub fn values(&self) -> impl Iterator<Item = &GraphValue> + '_ {
        self.values.iter()
    }

    pub fn storage(&self, id: LogicalStorageId) -> &LogicalStorage {
        &self.storages[id]
    }

    pub fn storages(&self) -> impl Iterator<Item = (LogicalStorageId, &LogicalStorage)> + '_ {
        self.storages.entries()
    }

    pub fn view(&self, id: LogicalViewId) -> &LogicalView {
        &self.views[id]
    }

    pub fn views(&self) -> impl Iterator<Item = (LogicalViewId, &LogicalView)> + '_ {
        self.views.entries()
    }

    /// The storage a state token versions. Total: every state token of the
    /// graph is defined.
    pub fn state_storage(&self, token: StateTokenId) -> LogicalStorageId {
        self.states[token]
    }

    pub fn states(&self) -> impl Iterator<Item = (StateTokenId, LogicalStorageId)> + '_ {
        self.states.entries().map(|(token, storage)| (token, *storage))
    }
}

#[derive(Clone, Debug)]
pub struct GraphRegion {
    pub parameters: Vec<RegionParameter>,
    pub nodes: IdVec<NodeId, LogicalNode>,
    /// Results of a nested region (branch joins, loop carries and visit
    /// states). The root region's list is empty: its results are the graph
    /// boundary.
    pub results: Vec<RegionResult>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RegionParameter {
    Value {
        id: GraphValueId,
        ty: ValueType,
    },
    State {
        id: StateTokenId,
        storage: LogicalStorageId,
    },
}

#[derive(Clone, Debug)]
pub enum RegionResult {
    Value {
        id: GraphValueId,
        ty: ValueType,
    },
    /// The state of `storage` this region leaves behind. `join` is set only
    /// where an independent loop admits cross-visit state joins
    /// (`DisjointWrite` with the checker proof, or `Atomic` with the admitted
    /// operation); branch and ordered-loop results carry no join.
    State {
        id: StateTokenId,
        storage: LogicalStorageId,
        join: Option<StateJoin>,
    },
}

/// How the states produced by the independent visits of one loop are joined.
/// These are the only admitted cross-visit state joins.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StateJoin {
    /// Every captured write went through an index the checker proved injective
    /// in the loop binder (affine with a nonzero coefficient, a mixed radix over
    /// nested binders, or a slice no longer than its stride; recorded through
    /// `LoopMutationSummary::disjoint_writes`): the visits write proved-disjoint
    /// places of the same storage.
    DisjointWrite {
        binder: GraphValueId,
        range: LogicalRange,
    },
    /// The storage was updated by an admitted atomic operation in every visit.
    Atomic { operations: Vec<AtomicOperation> },
}

/// One admitted atomic update of a storage: its combining operation, the
/// element dtype, and the number of point indices of the place.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtomicOperation {
    pub op: crate::intrinsics::AtomicOp,
    pub dtype: DType,
    pub arity: usize,
}

// ---------------------------------------------------------------------------
// Storage, state and views
// ---------------------------------------------------------------------------

/// One version of one logical storage. A fresh token is produced by the
/// region parameter or node that writes/initializes the storage; reads consume
/// the current token without versioning it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateToken {
    pub id: StateTokenId,
    pub storage: LogicalStorageId,
    /// Set when this token is the join of independent loop visits.
    pub join: Option<StateJoin>,
}

/// Semantic storage: shape, owner, initialization. No address space, byte
/// stride, pointer, allocation or physical tile. Returning a value is a
/// boundary fact, never a storage provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct LogicalStorage {
    pub shape: TensorType,
    pub owner: LogicalStorageOwner,
    pub initialization: Initialization,
}

/// Initialization state of one storage: independent disjoint
/// coverage composes structurally.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Initialization {
    Uninitialized,
    /// Some elements are proved written; the coverage proof records which axes
    /// are structurally covered (a whole write, or a binder-covering loop with
    /// the disjoint-write proof).
    PartiallyInitialized(Coverage),
    FullyInitialized,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Coverage {
    /// One entry per axis: whether construction proved the axis covered.
    pub axes: Vec<bool>,
}

impl Coverage {
    pub fn full(axes: usize) -> Coverage {
        Coverage {
            axes: vec![true; axes],
        }
    }

    pub fn none(axes: usize) -> Coverage {
        Coverage {
            axes: vec![false; axes],
        }
    }

    pub fn is_full(&self) -> bool {
        self.axes.iter().all(|a| *a)
    }

    /// Structural composition: an axis stays covered only when a proof covers it.
    pub fn union(&self, other: &Coverage) -> Coverage {
        Coverage {
            axes: self
                .axes
                .iter()
                .zip(&other.axes)
                .map(|(a, b)| *a || *b)
                .collect(),
        }
    }
}

/// A semantic view over one base: logical storage, or a computed tensor
/// value. Access is carried here, transforms are shape-only structure;
/// dynamic selection endpoints are graph values. A view of a computed value
/// is read-only (`Access::Shared`).
#[derive(Clone, Debug, PartialEq)]
pub struct LogicalView {
    pub base: ViewBase,
    pub shape: TensorType,
    pub access: Access,
    pub transform: ViewTransform,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Access {
    Shared,
    Exclusive,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ViewTransform {
    Identity,
    Reshape { source_shape: Vec<ExtentExpr> },
    Transpose { permutation: Vec<u32> },
    Slice { axes: Vec<SliceAxis> },
}

#[derive(Clone, Debug, PartialEq)]
pub enum SliceAxis {
    Full,
    Point(GraphValueId),
    Range {
        start: Option<GraphValueId>,
        end: Option<GraphValueId>,
    },
}

// ---------------------------------------------------------------------------
// Nodes
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LogicalNode {
    pub inputs: Vec<GraphValueId>,
    pub state_inputs: Vec<StateTokenId>,
    pub kind: LogicalNodeKind,
    /// The values this node originates, with their exhaustive kinds (the same
    /// records the graph's value table holds).
    pub outputs: Vec<GraphValue>,
    pub state_outputs: Vec<StateToken>,
    pub safety: Vec<SafetyObligation>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum LogicalNodeKind {
    Primitive(PrimitiveApplication),
    If(IfNode),
    Loop(LoopNode),
    Reduction(ReductionNode),
    Call(CallNode),
}

/// One primitive application: a registry primitive, a typed constant, or a
/// typed capability intrinsic (`If`/`Loop`/`Call` are graph structure, never
/// primitives).
#[derive(Clone, Debug, PartialEq)]
pub struct PrimitiveApplication {
    pub op: PrimitiveOp,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PrimitiveOp {
    Constant(crate::sir::Literal),
    /// The value of one runtime extent, as an i32 scalar.
    RuntimeExtent(RuntimeExtentId),
    Primitive(PrimitiveId),
    Capability(IntrinsicId),
}

#[derive(Clone, Debug)]
pub struct IfNode {
    pub condition: GraphValueId,
    pub then_region: GraphRegion,
    pub else_region: GraphRegion,
    pub joins: Vec<JoinSlot>,
    /// Branch value parameters with the outer values they snapshot. Both
    /// branches share one parameter schema, so one list covers both.
    pub captured: Vec<IfCapture>,
}

/// One branch parameter and the outer value it captures.
#[derive(Clone, Debug, PartialEq)]
pub struct IfCapture {
    pub parameter: GraphValueId,
    pub outer: GraphValueId,
}

/// The explicit join of one differing value or same-storage state across an
/// `if`. Branch regions share parameter schemas and both produce a result for
/// every join slot (a pass-through where the branch did not change it).
#[derive(Clone, Debug)]
pub enum JoinSlot {
    Value {
        then_result: RegionResultId,
        else_result: RegionResultId,
        joined: GraphValueId,
        ty: ValueType,
    },
    State {
        then_result: RegionResultId,
        else_result: RegionResultId,
        joined: StateTokenId,
        storage: LogicalStorageId,
    },
}

#[derive(Clone, Debug)]
pub struct LoopNode {
    pub kind: LoopKind,
    pub range: LogicalRange,
    /// The per-visit binder value; a `Value` region parameter of the body.
    pub binder: GraphValueId,
    /// Captured values read by the body and never written (region parameters,
    /// not carries).
    pub invariant_values: Vec<GraphValueId>,
    /// Initial values of the carried slots, in carried order (ordered loops).
    pub initial_values: Vec<GraphValueId>,
    /// Entry state tokens of the carried or joined storages.
    pub initial_states: Vec<StateTokenId>,
    pub body: GraphRegion,
    /// Ordered loops carry every changed captured value/state. Independent
    /// loops have no data carries (`carried` is empty).
    pub carried: Vec<CarriedSlot>,
}

#[derive(Clone, Debug)]
pub struct CarriedSlot {
    pub initial: RegionInput,
    pub body_parameter: RegionParameterId,
    pub body_result: RegionResultId,
    /// Ordinal of this carry's loop-level result (a result of the loop node).
    pub loop_result: RegionResultId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionInput {
    Value(GraphValueId),
    State(StateTokenId),
}

/// A half-open iteration range in ascending coordinate order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalRange {
    pub start: GraphValueId,
    pub end: GraphValueId,
    pub bound: ExtentExpr,
}

#[derive(Clone, Debug)]
pub struct ReductionNode {
    pub operand: GraphValueId,
    pub axis: usize,
    pub op: ReduceOp,
    pub order: ReductionOrder,
    pub accumulator: DType,
    pub result: ValueType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReductionOrder {
    /// Visits ascending coordinates; the reference order.
    Ascending,
    /// The source marked the reduction reassociable.
    Unordered,
}

/// One call: the occurrence-specific choice it selects from and the callee
/// contract instantiated with the caller's identities. The node's inputs are
/// the boundary input values (key order), its state inputs the consumed
/// tensor states, its outputs the result values, and its state outputs the
/// final states of exclusively borrowed caller storages. A void call has no
/// outputs and retains its completion as a node of the region.
#[derive(Clone, Debug)]
pub struct CallNode {
    pub choice: ChoiceId,
    pub boundary: CallBoundary,
}

// ---------------------------------------------------------------------------
// Runtime extents and safety
// ---------------------------------------------------------------------------

/// A runtime-determined extent: `value` is the retained runtime expression
/// used for all semantics; `capacity` is a resource/proof bound only. Logical
/// construction derives it from the expression's leaves (shape-field domains,
/// earlier extents' capacities, the representation ranges of its integer and
/// index values) with checked interval arithmetic; an expression without a
/// derivable finite bound is a construction error, never a substituted
/// capacity. Planning may lower it.
#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeExtent {
    pub id: RuntimeExtentId,
    pub value: RuntimeScalarExpr,
    pub capacity: u64,
    /// The caller's expected runtime value. Sourced only from a bounded entry
    /// shape binding: `Some` exactly when `value` is that binding's
    /// `ShapeField`. Every other runtime extent (a value-derived slice
    /// length, a shape expression) has no stated expectation and is priced at
    /// its capacity. Semantics and resource bounds never read it.
    pub expected: Option<u64>,
}

/// A retained runtime scalar expression over graph values, invocation shape
/// fields and other runtime extents. Representation only; evaluation belongs
/// to planning/resolution.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RuntimeScalarExpr {
    Const(i64),
    Value(GraphValueId),
    Extent(RuntimeExtentId),
    /// The actual value of one bounded entry shape parameter, supplied by the
    /// invocation and validated against its finite domain before submission.
    ShapeField(ShapeFieldId),
    Add(Box<RuntimeScalarExpr>, Box<RuntimeScalarExpr>),
    Sub(Box<RuntimeScalarExpr>, Box<RuntimeScalarExpr>),
    Mul(Box<RuntimeScalarExpr>, Box<RuntimeScalarExpr>),
    Div(Box<RuntimeScalarExpr>, Box<RuntimeScalarExpr>),
    Rem(Box<RuntimeScalarExpr>, Box<RuntimeScalarExpr>),
}

/// Runtime obligations created by logical construction; each physical
/// alternative consumes every one of them later as `StaticallyProved` or
/// `RuntimeChecked`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SafetyObligation {
    /// An operation whose semantics start from the first element requires a
    /// nonempty logical extent. This belongs to the semantic program rather
    /// than to any backend reduction strategy.
    ExtentPositive {
        extent: ExtentExpr,
    },
    IndexInBounds {
        index: GraphValueId,
        extent: ExtentExpr,
    },
    RangeInBounds {
        start: GraphValueId,
        end: GraphValueId,
        extent: ExtentExpr,
    },
    DivisorNonZero {
        value: GraphValueId,
    },
    SignedDivisionNoOverflow {
        lhs: GraphValueId,
        rhs: GraphValueId,
    },
    ShiftInRange {
        value: GraphValueId,
    },
    ShapeProductFits {
        factors: Vec<ExtentExpr>,
        bits: u8,
    },
}

impl SafetyObligation {
    /// The values this obligation reads at runtime.
    pub fn values(&self) -> Vec<GraphValueId> {
        match self {
            SafetyObligation::ExtentPositive { .. } => Vec::new(),
            SafetyObligation::IndexInBounds { index, .. } => vec![*index],
            SafetyObligation::RangeInBounds { start, end, .. } => vec![*start, *end],
            SafetyObligation::DivisorNonZero { value } => vec![*value],
            SafetyObligation::SignedDivisionNoOverflow { lhs, rhs } => vec![*lhs, *rhs],
            SafetyObligation::ShiftInRange { value } => vec![*value],
            SafetyObligation::ShapeProductFits { .. } => Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Construction entry point
// ---------------------------------------------------------------------------

/// Build the logical program of the domain's entry for one effective target
/// under one specialization domain. Every call occurrence receives its own
/// choice; applicability is evaluated per occurrence against the effective
/// target and the domain (portable bodies and same-backend lowerings are
/// peer semantic alternatives; an implementation predicate admits an
/// alternative only when it holds on the whole domain).
pub fn construct(
    program: &Program,
    target: &EffectiveTargetIdentity,
    supports_intrinsic: &dyn Fn(&IntrinsicUse) -> Result<(), String>,
    domain: &SpecializationDomain,
) -> Result<LogicalProgram, LogicalConstructionError> {
    normalize::construct(program, target, supports_intrinsic, domain).map_err(|error| match error {
        normalize::BuildError::NoImplementation(report) => {
            LogicalConstructionError::NoApplicableImplementation(report)
        }
        normalize::BuildError::Invalid(reason) => LogicalConstructionError::InvalidProgram(reason),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::{compile, SourceFile};
    use crate::types::DType;
    use specialization::ShapeBinding;
    use std::collections::BTreeMap;

    fn check(sources: &[(&str, &str)]) -> Result<crate::sir::Program, String> {
        let files: Vec<SourceFile> = sources
            .iter()
            .map(|(path, text)| SourceFile {
                path: path.to_string(),
                text: text.to_string(),
            })
            .collect();
        compile(&files).map_err(|d| d.iter().map(|d| d.render()).collect::<Vec<_>>().join("\n"))
    }

    fn supports_all(_: &crate::sir::IntrinsicUse) -> Result<(), String> {
        Ok(())
    }

    fn target(backend: &str) -> EffectiveTargetIdentity {
        EffectiveTargetIdentity {
            backend: backend.to_string(),
            capability_fingerprint: "test-fingerprint".to_string(),
        }
    }

    fn exact(program: &Program, entry: &str, entries: &[(&str, u64)]) -> SpecializationDomain {
        let shapes = entries
            .iter()
            .map(|(name, value)| (name.to_string(), ShapeBinding::Exact(*value)))
            .collect();
        SpecializationDomain::new(program, entry, shapes, BTreeMap::new())
            .expect("the exact domain binds every entry parameter")
    }

    fn entry_graph(logical: &LogicalProgram) -> &TaskGraph {
        logical.graph(
            logical
                .choice(logical.entry_choice)
                .alternatives
                .iter()
                .next()
                .expect("the entry has an alternative")
                .graph,
        )
    }

    /// One representative kernel: allocation, borrowed views, a reduction,
    /// ordered and independent loops, a call, and an atomic join.
    const KERNEL: &str = "fn add[M, N](x: &tensor[M, N] f32, y: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32:\n    let mut output = result\n    parallel for row in 0..M:\n        for col in 0..N:\n            output[row, col] = x[row, col] + y[row, col]\n    return output\n\nfn linear[M, N](x: &tensor[M, N] f32, y: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32:\n    return add(x, y, result)\n";

    #[test]
    fn representative_kernel_constructs_and_verifies() {
        let program = check(&[("kernel.seismic", KERNEL)]).expect("the kernel checks");
        let domain = exact(&program, "linear", &[("M", 4), ("N", 8)]);
        let logical = construct(&program, &target("cpu"), &supports_all, &domain)
            .expect("construction succeeds");
        logical.verify().expect("the built program verifies");

        // The entry choice and one occurrence-specific call choice.
        assert_eq!(logical.entry_choice, ChoiceId(0));
        assert_eq!(logical.choices().count(), 2);
        let entry = logical.choice(logical.entry_choice);
        assert_eq!(entry.interface.name, "linear");
        assert_eq!(entry.alternatives.iter().count(), 1);

        // The entry graph is a single call whose boundary passes two shared
        // borrows and one move, and whose result leaf is a computed value of
        // the caller (no fabricated result storage).
        let entry_graph = entry_graph(&logical);
        let call = entry_graph
            .root()
            .nodes
            .iter()
            .find_map(|node| match &node.kind {
                LogicalNodeKind::Call(call) => Some(call),
                _ => None,
            })
            .expect("the entry body is one call");
        let ownerships: Vec<ParamOwnership> = call
            .boundary
            .inputs
            .values()
            .map(|input| match input {
                CallInput::Tensor { ownership, .. } => *ownership,
                CallInput::Computed { ownership, .. } => *ownership,
                CallInput::Value(_) => ParamOwnership::Value,
            })
            .collect();
        assert_eq!(
            ownerships,
            vec![
                ParamOwnership::Shared,
                ParamOwnership::Shared,
                ParamOwnership::Owned
            ]
        );
        assert!(call.boundary.final_states.is_empty());
        assert_eq!(call.boundary.results.len(), 1);
        let (leaf, result) = call.boundary.results.iter().next().unwrap();
        assert_eq!(
            *leaf,
            BoundaryLeaf::Result {
                leaf: crate::types::ValuePath::default()
            }
        );
        assert_eq!(
            entry_graph.value(*result).tensor_source(),
            Some(TensorSource::Computed)
        );
        // The entry boundary returns that computed value directly.
        let returned = entry_graph
            .boundary
            .results()
            .values()
            .next()
            .expect("the entry returns one leaf");
        assert_eq!(returned.value, *result);

        // The callee graph contains the independent outer loop with a
        // disjoint-write join and the ordered inner loop with carries.
        let add_choice = logical
            .choices()
            .map(|(_, choice)| choice)
            .find(|choice| choice.interface.name == "add")
            .expect("the call created its own choice");
        let add_graph = logical.graph(add_choice.alternatives.iter().next().unwrap().graph);
        let mut independent_joins = 0;
        let mut ordered_carries = 0;
        fn scan(region: &GraphRegion, joins: &mut usize, carries: &mut usize) {
            for node in region.nodes.iter() {
                match &node.kind {
                    LogicalNodeKind::Loop(loop_node) => {
                        match loop_node.kind {
                            LoopKind::Independent => {
                                *joins += node
                                    .state_outputs
                                    .iter()
                                    .filter(|token| token.join.is_some())
                                    .count();
                                assert!(loop_node.carried.is_empty());
                            }
                            LoopKind::Ordered => {
                                *carries += loop_node.carried.len();
                                assert!(node
                                    .state_outputs
                                    .iter()
                                    .all(|token| token.join.is_none()));
                            }
                        }
                        scan(&loop_node.body, joins, carries);
                    }
                    LogicalNodeKind::If(if_node) => {
                        scan(&if_node.then_region, joins, carries);
                        scan(&if_node.else_region, joins, carries);
                    }
                    LogicalNodeKind::Primitive(_)
                    | LogicalNodeKind::Reduction(_)
                    | LogicalNodeKind::Call(_) => {}
                }
            }
        }
        scan(
            add_graph.root(),
            &mut independent_joins,
            &mut ordered_carries,
        );
        assert!(independent_joins > 0, "the parallel loop joins its visits");
        assert!(ordered_carries > 0, "the ordered loop carries its state");

        // The returned parameter storage is fully initialized: the nested
        // loops cover both axes structurally.
        for (_, storage) in add_graph.storages() {
            if matches!(storage.owner, LogicalStorageOwner::Parameter(_)) {
                assert_eq!(storage.initialization, Initialization::FullyInitialized);
            }
        }
    }

    #[test]
    fn reductions_are_nodes_with_registry_accumulators() {
        let program = check(&[(
            "reduce.seismic",
            "fn sum[N](x: &tensor[N] f32) -> f32:\n    return reduce(f32(x), 0, sum)\n\nfn arg[N](x: &tensor[N] f16) -> i32:\n    return reduce(f16(x), 0, argmax)\n",
        )])
        .expect("the reductions check");
        let logical = construct(
            &program,
            &target("cpu"),
            &supports_all,
            &exact(&program, "sum", &[("N", 16)]),
        )
        .expect("construction succeeds");
        logical.verify().expect("verification succeeds");
        let graph = entry_graph(&logical);
        let reduction = graph
            .root()
            .nodes
            .iter()
            .find_map(|node| match &node.kind {
                LogicalNodeKind::Reduction(reduction) => Some(reduction),
                _ => None,
            })
            .expect("the sum is one reduction node");
        assert_eq!(reduction.op, crate::intrinsics::ReduceOp::Sum);
        assert_eq!(reduction.accumulator, DType::F32);
        assert_eq!(reduction.order, ReductionOrder::Ascending);
        // The cast operand is a computed tensor reduced without storage.
        assert_eq!(
            graph.value(reduction.operand).tensor_source(),
            Some(TensorSource::Computed)
        );

        let logical = construct(
            &program,
            &target("cpu"),
            &supports_all,
            &exact(&program, "arg", &[("N", 16)]),
        )
        .expect("construction succeeds");
        let graph = entry_graph(&logical);
        let reduction = graph
            .root()
            .nodes
            .iter()
            .find_map(|node| match &node.kind {
                LogicalNodeKind::Reduction(reduction) => Some(reduction),
                _ => None,
            })
            .expect("the argmax is one reduction node");
        assert_eq!(reduction.op, crate::intrinsics::ReduceOp::Argmax);
        assert_eq!(reduction.accumulator, DType::I32);
    }

    #[test]
    fn computed_tensors_are_returned_without_storage() {
        let program = check(&[(
            "computed.seismic",
            "fn f[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return f32(x)\n",
        )])
        .expect("the source checks");
        let logical = construct(
            &program,
            &target("cpu"),
            &supports_all,
            &exact(&program, "f", &[("N", 4)]),
        )
        .expect("construction succeeds");
        logical.verify().expect("verification succeeds");
        let graph = entry_graph(&logical);
        let returned = graph
            .boundary
            .results()
            .values()
            .next()
            .expect("one result leaf");
        assert_eq!(
            graph.value(returned.value).tensor_source(),
            Some(TensorSource::Computed)
        );
        // Only the parameter owns storage.
        assert!(graph
            .storages()
            .all(|(_, storage)| matches!(storage.owner, LogicalStorageOwner::Parameter(_))));
    }

    #[test]
    fn safety_obligations_are_created_not_discharged() {
        let program = check(&[(
            "safety.seismic",
            "fn f[N](x: &tensor[N] f32, i: index[N]) -> f32:\n    return x[i] + 1.0 / x[0]\n",
        )])
        .expect("the source checks");
        let logical = construct(
            &program,
            &target("cpu"),
            &supports_all,
            &exact(&program, "f", &[("N", 4)]),
        )
        .expect("construction succeeds");
        let graph = entry_graph(&logical);
        let obligations: Vec<&SafetyObligation> = graph
            .root()
            .nodes
            .iter()
            .flat_map(|node| node.safety.iter())
            .collect();
        assert!(
            obligations
                .iter()
                .any(|o| matches!(o, SafetyObligation::IndexInBounds { .. })),
            "the element read creates an index obligation"
        );
        assert!(
            obligations
                .iter()
                .any(|o| matches!(o, SafetyObligation::DivisorNonZero { .. })),
            "the division creates a divisor obligation"
        );
    }

    #[test]
    fn slice_views_create_range_obligations_and_new_views() {
        let program = check(&[(
            "views.seismic",
            "fn f[N](x: &tensor[N, N] f32) -> tensor[N] f32:\n    let v = x[0:N, 0]\n    return to_owned(v)\n",
        )])
        .expect("the source checks");
        let logical = construct(
            &program,
            &target("cpu"),
            &supports_all,
            &exact(&program, "f", &[("N", 4)]),
        )
        .expect("construction succeeds");
        logical.verify().expect("verification succeeds");
        let graph = entry_graph(&logical);
        // The slice view shares the parameter storage and carries its transform.
        let parameter_storage_id = graph
            .storages()
            .find(|(_, storage)| matches!(storage.owner, LogicalStorageOwner::Parameter(_)))
            .map(|(id, _)| id)
            .expect("the parameter has storage");
        let sliced = graph
            .views()
            .map(|(_, view)| view)
            .find(|view| matches!(view.transform, ViewTransform::Slice { .. }))
            .expect("the slice is a view transform");
        match sliced.base {
            ViewBase::Storage(storage) => assert_eq!(storage, parameter_storage_id),
            ViewBase::Value(_) => panic!("the slice views the parameter's storage"),
        }
        assert!(graph
            .root()
            .nodes
            .iter()
            .flat_map(|node| node.safety.iter())
            .any(|o| matches!(o, SafetyObligation::RangeInBounds { .. })));
        // `to_owned` names local storage; the returned value is its view.
        let returned = graph
            .boundary
            .results()
            .values()
            .next()
            .expect("one result leaf");
        let Some(TensorSource::View(view)) = graph.value(returned.value).tensor_source() else {
            panic!("to_owned produces a view of local storage");
        };
        match graph.view(view).base {
            ViewBase::Storage(storage) => assert_eq!(
                graph.storage(storage).owner,
                LogicalStorageOwner::Local
            ),
            ViewBase::Value(_) => panic!("to_owned produces a view of local storage"),
        }
    }

    #[test]
    fn views_of_computed_tensors_name_the_value_as_base() {
        // A view over a computed tensor is a view of that value: no storage
        // is declared for the operand and none is fabricated for the result.
        let program = check(&[(
            "computed-view.seismic",
            "fn f[N](x: &tensor[N, N] f32) -> tensor[N, N] f32:\n    return reshape(f32(x), (N, N))\n",
        )])
        .expect("the source checks");
        let logical = construct(
            &program,
            &target("cpu"),
            &supports_all,
            &exact(&program, "f", &[("N", 4)]),
        )
        .expect("construction succeeds");
        logical.verify().expect("verification succeeds");
        let graph = entry_graph(&logical);
        let returned = graph
            .boundary
            .results()
            .values()
            .next()
            .expect("one result leaf");
        let Some(TensorSource::View(view)) = graph.value(returned.value).tensor_source() else {
            panic!("the reshape of a computed tensor is a view");
        };
        // The view's base is the computed cast value, not any storage.
        let ViewBase::Value(base) = graph.view(view).base else {
            panic!("the reshape views the computed value");
        };
        assert_eq!(
            graph.value(base).tensor_source(),
            Some(TensorSource::Computed)
        );
        assert_eq!(graph.view(view).access, Access::Shared);
        assert!(matches!(
            graph.view(view).transform,
            ViewTransform::Reshape { .. }
        ));
        // Only the parameter owns storage; the reshape reads no storage.
        assert!(graph
            .storages()
            .all(|(_, storage)| matches!(storage.owner, LogicalStorageOwner::Parameter(_))));
        let reshape_node = graph
            .root()
            .nodes
            .iter()
            .find(|node| {
                matches!(
                    &node.kind,
                    LogicalNodeKind::Primitive(PrimitiveApplication {
                        op: PrimitiveOp::Primitive(PrimitiveId::Reshape),
                    })
                )
            })
            .expect("the reshape is one node");
        assert!(reshape_node.state_inputs.is_empty());
    }

    #[test]
    fn computed_tensors_are_read_as_values() {
        let program = check(&[(
            "computed-read.seismic",
            "fn f[N](x: &tensor[N] f32) -> f32:\n    return f32(x)[0]\n",
        )])
        .expect("the source checks");
        let logical = construct(
            &program,
            &target("cpu"),
            &supports_all,
            &exact(&program, "f", &[("N", 4)]),
        )
        .expect("construction succeeds");
        logical.verify().expect("verification succeeds");
        let graph = entry_graph(&logical);
        let read = graph
            .root()
            .nodes
            .iter()
            .find(|node| {
                matches!(
                    &node.kind,
                    LogicalNodeKind::Primitive(PrimitiveApplication {
                        op: PrimitiveOp::Primitive(PrimitiveId::ElementRead { .. }),
                    })
                )
            })
            .expect("the element read is one node");
        // The computed operand is read as a value: no storage dependency.
        assert!(read.state_inputs.is_empty());
        assert!(graph.storages().all(|(_, storage)| {
            matches!(storage.owner, LogicalStorageOwner::Parameter(_))
        }));
    }

    #[test]
    fn computed_tensors_are_call_arguments_without_storage() {
        // A call result and a cast are computed values at the logical level;
        // the owned move and the shared borrow of them pass the values
        // themselves, with no caller storage fabricated for either.
        let program = check(&[(
            "computed-arg.seismic",
            "fn dup[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return f32(x)\n\nfn add[N](a: tensor[N] f32, b: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(b) + a\n\nfn f[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return add(dup(x), f32(x))\n",
        )])
        .expect("the source checks");
        let logical = construct(
            &program,
            &target("cpu"),
            &supports_all,
            &exact(&program, "f", &[("N", 4)]),
        )
        .expect("construction succeeds");
        logical.verify().expect("verification succeeds");
        let graph = entry_graph(&logical);
        let call = graph
            .root()
            .nodes
            .iter()
            .find_map(|node| match &node.kind {
                LogicalNodeKind::Call(call)
                    if logical.choice(call.choice).interface.name == "add" =>
                {
                    Some(call)
                }
                _ => None,
            })
            .expect("the entry body ends in the `add` call");
        let mut moved = 0;
        let mut shared = 0;
        for input in call.boundary.inputs.values() {
            match input {
                // The owned move of the computed call result: the value.
                CallInput::Computed { value, ownership } => {
                    assert_eq!(
                        graph.value(*value).tensor_source(),
                        Some(TensorSource::Computed)
                    );
                    match ownership {
                        ParamOwnership::Owned => moved += 1,
                        ParamOwnership::Shared => shared += 1,
                        ParamOwnership::Exclusive | ParamOwnership::Value => {
                            panic!("a computed argument is shared or moved")
                        }
                    }
                }
                CallInput::Tensor { .. } => {
                    panic!("every argument is a computed value")
                }
                CallInput::Value(_) => panic!("every argument is a tensor leaf"),
            }
        }
        assert_eq!(moved, 1);
        assert_eq!(shared, 1);
        // No caller storage was fabricated for either computed argument.
        assert!(graph.storages().all(|(_, storage)| {
            matches!(storage.owner, LogicalStorageOwner::Parameter(_))
        }));
    }

    #[test]
    fn in_place_writes_to_computed_locals_realize_state() {
        // A `let mut` local written in place names mutable state: its
        // realization into local storage is the write's own state, the one
        // materialization construction performs.
        let program = check(&[(
            "write-computed.seismic",
            "fn f[N](x: &tensor[N] f32) -> tensor[N] f32:\n    let mut v = f32(x)\n    v[0] = 1.0\n    return v\n",
        )])
        .expect("the source checks");
        let logical = construct(
            &program,
            &target("cpu"),
            &supports_all,
            &exact(&program, "f", &[("N", 4)]),
        )
        .expect("construction succeeds");
        logical.verify().expect("verification succeeds");
        let graph = entry_graph(&logical);
        let returned = graph
            .boundary
            .results()
            .values()
            .next()
            .expect("one result leaf");
        let Some(TensorSource::View(view)) = graph.value(returned.value).tensor_source() else {
            panic!("the written local is a view of its realized storage");
        };
        let ViewBase::Storage(storage) = graph.view(view).base else {
            panic!("the written local's storage is logical storage");
        };
        assert_eq!(graph.storage(storage).owner, LogicalStorageOwner::Local);
        assert_eq!(
            graph.storage(storage).initialization,
            Initialization::FullyInitialized
        );
    }

    #[test]
    fn atomic_joins_are_admitted_operations() {
        let program = check(&[(
            "atomic.seismic",
            "fn hist[N](x: &tensor[N] f32, out: tensor[64] f32) -> f32 for metal:\n    let mut acc = out\n    parallel for i in 0..N:\n        atomic(add, acc[0], x[i])\n    return reduce(f32(acc), 0, sum)\n",
        )])
        .expect("the source checks");
        let logical = construct(
            &program,
            &target("metal"),
            &supports_all,
            &exact(&program, "hist", &[("N", 128)]),
        )
        .expect("construction succeeds on the same backend");
        logical.verify().expect("verification succeeds");
        let graph = entry_graph(&logical);
        let loop_node = graph
            .root()
            .nodes
            .iter()
            .find_map(|node| match &node.kind {
                LogicalNodeKind::Loop(loop_node) => Some(loop_node),
                _ => None,
            })
            .expect("the parallel loop is a loop node");
        assert_eq!(loop_node.kind, LoopKind::Independent);
        let join = graph
            .root()
            .nodes
            .iter()
            .flat_map(|node| node.state_outputs.iter())
            .find_map(|token| token.join.as_ref())
            .expect("the atomic update joins across visits");
        assert!(
            matches!(join, StateJoin::Atomic { .. }),
            "the join records the admitted atomic operation"
        );
    }

    #[test]
    fn missing_implementation_on_the_wrong_backend_is_reported() {
        let program = check(&[(
            "metal-only.seismic",
            "fn f[M](x: &tensor[M] f32) -> f32 for metal:\n    return reduce(f32(x), 0, sum)\n",
        )])
        .expect("the source checks");
        let error = construct(
            &program,
            &target("cpu"),
            &supports_all,
            &exact(&program, "f", &[("M", 4)]),
        )
        .expect_err("the metal-only family is inapplicable on cpu");
        match error {
            LogicalConstructionError::NoApplicableImplementation(report) => {
                assert_eq!(report.entry, "f");
                assert_eq!(report.occurrences.len(), 1);
                assert!(report.occurrences[0]
                    .rejections
                    .iter()
                    .any(|(_, reason)| reason.contains("metal")));
            }
            other => panic!("expected an applicability report, got {other:?}"),
        }
    }

    #[test]
    fn identity_tracks_the_domain() {
        let program = check(&[("kernel.seismic", KERNEL)]).expect("the kernel checks");
        let a = construct(
            &program,
            &target("cpu"),
            &supports_all,
            &exact(&program, "linear", &[("M", 4), ("N", 8)]),
        )
        .expect("construction succeeds");
        let b = construct(
            &program,
            &target("cpu"),
            &supports_all,
            &exact(&program, "linear", &[("M", 4), ("N", 8)]),
        )
        .expect("construction succeeds");
        let c = construct(
            &program,
            &target("cpu"),
            &supports_all,
            &exact(&program, "linear", &[("M", 5), ("N", 8)]),
        )
        .expect("construction succeeds");
        assert_eq!(a.identity, b.identity);
        assert_ne!(a.identity, c.identity);
    }

    #[test]
    fn bounded_shapes_become_one_runtime_extent_each() {
        let program = check(&[("kernel.seismic", KERNEL)]).expect("the kernel checks");
        let shapes = [
            (
                "M".to_string(),
                ShapeBinding::Bounded {
                    min: 1,
                    max: 64,
                    expected: 16,
                },
            ),
            ("N".to_string(), ShapeBinding::Exact(8)),
        ]
        .into_iter()
        .collect();
        let domain = SpecializationDomain::new(&program, "linear", shapes, BTreeMap::new())
            .expect("the bounded domain binds every entry parameter");
        let logical = construct(&program, &target("cpu"), &supports_all, &domain)
            .expect("construction succeeds");
        logical.verify().expect("verification succeeds");
        assert_eq!(logical.shape_fields.len(), 1);
        let field = logical.shape_field(ShapeFieldId(0));
        assert_eq!(field.name, "M");
        assert_eq!(field.expected, 16);
        let sourced: Vec<&RuntimeExtent> = logical
            .runtime_extents()
            .filter(|extent| extent.value == RuntimeScalarExpr::ShapeField(ShapeFieldId(0)))
            .collect();
        assert_eq!(sourced.len(), 1);
        assert_eq!(sourced[0].capacity, 64);
        assert_eq!(sourced[0].expected, Some(16));
        // The entry interface retains the runtime extent for `M` and the
        // static extent for `N`.
        let entry = logical.choice(logical.entry_choice);
        let ValueType::Tensor(shape) = &entry.interface.params[0].ty else {
            panic!("x is a tensor");
        };
        assert_eq!(shape.axes[0], ExtentExpr::Runtime(sourced[0].id));
        assert_eq!(shape.axes[1], ExtentExpr::Static(8));
    }

    #[test]
    fn coverage_composes_structurally() {
        // Independent and ordered binder-covering loops compose axis by axis:
        // the returned storage is fully initialized.
        let program = check(&[(
            "coverage.seismic",
            "fn fill[M, N](out: tensor[M, N] f32) -> tensor[M, N] f32:\n    let mut acc = out\n    parallel for i in 0..M:\n        for j in 0..N:\n            acc[i, j] = f32(1.0)\n    return acc\n",
        )])
        .expect("the sources check");
        let logical = construct(
            &program,
            &target("cpu"),
            &supports_all,
            &exact(&program, "fill", &[("M", 4), ("N", 4)]),
        )
        .expect("construction succeeds");
        logical.verify().expect("verification succeeds");
        let graph = entry_graph(&logical);
        let (_, out_storage) = graph
            .storages()
            .find(|(_, storage)| matches!(storage.owner, LogicalStorageOwner::Parameter(_)))
            .expect("the output parameter has storage");
        assert_eq!(out_storage.initialization, Initialization::FullyInitialized);
    }

    #[test]
    fn lowerings_are_peer_alternatives_only_on_their_backend() {
        let program = check(&[(
            "lower.seismic",
            "fn f[M](x: &tensor[M] f32) -> tensor[M] f32:\n    return to_owned(x)\n\nlower f[M](x: &tensor[M] f32) -> tensor[M] f32 for cpu:\n    return to_owned(x)\n",
        )])
        .expect("the sources check");
        let cpu = construct(
            &program,
            &target("cpu"),
            &supports_all,
            &exact(&program, "f", &[("M", 4)]),
        )
        .expect("construction succeeds on cpu");
        let entry = cpu.choice(cpu.entry_choice);
        let kinds: Vec<ImplementationKind> = entry.alternatives.iter().map(|a| a.kind).collect();
        assert!(kinds.contains(&ImplementationKind::PortableBody));
        assert!(kinds.contains(&ImplementationKind::Lowering));
        // The lowering starts with unknown whole-candidate equivalence.
        for alternative in entry.alternatives.iter() {
            if alternative.kind == ImplementationKind::Lowering {
                assert_eq!(alternative.authored_numerical_effects.len(), 1);
            }
        }
        cpu.verify().expect("verification succeeds");

        let metal = construct(
            &program,
            &target("metal"),
            &supports_all,
            &exact(&program, "f", &[("M", 4)]),
        )
        .expect("the portable body still applies on metal");
        let entry = metal.choice(metal.entry_choice);
        assert_eq!(entry.alternatives.iter().count(), 1);
        assert_eq!(
            entry.alternatives.iter().next().unwrap().kind,
            ImplementationKind::PortableBody
        );
    }

    #[test]
    fn capability_absence_removes_the_alternative() {
        let program = check(&[(
            "cap.seismic",
            "fn f[M](x: &tensor[M] f32) -> f32 for metal requires metal.subgroup:\n    let s = metal.subgroup.simd_sum(x[0])\n    return s\n\nfn g[M](x: &tensor[M] f32) -> f32 for metal requires metal.subgroup:\n    return f(x)\n",
        )])
        .expect("the sources check");
        // Supporting the capability: both alternatives apply.
        let logical = construct(
            &program,
            &target("metal"),
            &|_| Ok(()),
            &exact(&program, "g", &[("M", 4)]),
        )
        .expect("construction succeeds");
        logical.verify().expect("verification succeeds");
        // Rejecting it: the inner occurrence has no implementation.
        let error = construct(
            &program,
            &target("metal"),
            &|_| Err("no subgroup hardware".to_string()),
            &exact(&program, "g", &[("M", 4)]),
        )
        .expect_err("the capability is absent");
        match error {
            LogicalConstructionError::NoApplicableImplementation(report) => {
                assert!(report.occurrences.iter().any(|rejection| rejection
                    .rejections
                    .iter()
                    .any(|(_, reason)| reason.contains("no subgroup hardware"))));
            }
            other => panic!("expected an applicability report, got {other:?}"),
        }
    }
}
