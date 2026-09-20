//! The logical program.
//!
//! `LogicalProgram` is occurrence-qualified implementation choices plus
//! hierarchical SSA task graphs. It is built by one forward pass over the
//! checked body (`normalize`): constructors create outputs and
//! dependencies immediately, so nothing is reconstructed by recursive scanning
//! afterwards. Validity is enforced during construction by the type-state
//! `GraphBuilder` (`builder`) and re-checked defensively for cached data by
//! `verify`.
//!
//! Logical storage and views carry semantic shape, origin, initialization,
//! access and transforms only — no address space, byte stride, pointer,
//! allocation or physical tile. The ownership/effect dependency is the
//! state-token chain itself: every graph value and state token has exactly one
//! region-parameter or node origin, reads consume the current state, and
//! writes/atomics/moves/exclusive calls consume one state and produce the next.

use crate::intrinsics::{CapabilityId, IntrinsicId, PrimitiveId, ReduceOp};
use crate::sir::{DefId, IntrinsicUse, LoopKind, Mode, ParamOwnership, Program};
use crate::span::Span;
use crate::types::{DType, Elem, ExtentExpr, NonEmpty, RuntimeExtentId, TensorType, ValueType};
use std::collections::{BTreeMap, BTreeSet};

mod builder;
mod normalize;
mod verify;

pub use builder::{
    Building, Complete, GraphBuilder, Ids, LoopOutcome, LoopSpec, Output, PrimitiveOutcome,
    PrimitiveSpec, WriteEffect,
};
pub use normalize::{ApplicabilityReport, LogicalConstructionError, OccurrenceRejection};

/// One definition of the checked program, as an implementation alternative.
pub type DefinitionId = DefId;

// ---------------------------------------------------------------------------
// Identity and target
// ---------------------------------------------------------------------------

/// Identity of the logical program: changing semantics (source, registry
/// revision, entry, or the concrete specialization) changes this key, and with
/// it every downstream plan, cache and evidence identity.
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
identity! {
    /// Position of one result in a region's result list (also used for the
    /// result ordinals a loop node exposes for its carried slots).
    RegionResultId
}

/// An id-indexed collection. Ids are unique within the owning artifact and
/// may be globally allocated (gaps are legal); iteration is in id order.
#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
pub struct LogicalProgram {
    pub identity: LogicalIdentity,
    pub entry: String,
    pub target: EffectiveTargetIdentity,
    /// Concrete shape parameters of the entry specialization.
    pub shapes: BTreeMap<String, i64>,
    /// Concrete element parameters of the entry specialization.
    pub elements: BTreeMap<String, Elem>,
    pub entry_choice: ChoiceId,
    pub choices: IdVec<ChoiceId, ImplementationChoice>,
    pub graphs: IdVec<GraphId, TaskGraph>,
    pub runtime_extents: IdVec<RuntimeExtentId, RuntimeExtent>,
}

impl LogicalProgram {
    pub fn choice(&self, id: ChoiceId) -> &ImplementationChoice {
        &self.choices[id]
    }

    pub fn graph(&self, id: GraphId) -> &TaskGraph {
        &self.graphs[id]
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
/// share occurrence identity, applicability, cost or result storage.
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
    /// The interned task graph of this definition under the occurrence's
    /// specialization (internal storages are template-internal; result storage
    /// is occurrence-owned and allocated by the caller).
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

#[derive(Clone, Debug)]
pub struct TaskGraph {
    pub choice: ChoiceId,
    /// Ordinal of the alternative within the choice.
    pub alternative: u32,
    /// The root region's value/state parameters, in interface order.
    pub parameters: Vec<RegionParameter>,
    pub storages: IdVec<LogicalStorageId, LogicalStorage>,
    pub views: IdVec<LogicalViewId, LogicalView>,
    pub root: GraphRegion,
    /// The boundary results of this implementation, in canonical path order:
    /// the function result leaves, then the final states of `inout` tensor
    /// parameters.
    pub results: Vec<RegionResult>,
}

#[derive(Clone, Debug)]
pub struct GraphRegion {
    pub parameters: Vec<RegionParameter>,
    pub nodes: IdVec<NodeId, LogicalNode>,
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
// Values, storage, views
// ---------------------------------------------------------------------------

/// One SSA graph value. Tensor values are backed by exactly one logical view
/// (`view`); scalar, index, range, tuple and capability values are pure data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphValue {
    pub id: GraphValueId,
    pub ty: ValueType,
    pub view: Option<LogicalViewId>,
}

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

/// Semantic storage: shape, origin, initialization. No address space, byte
/// stride, pointer, allocation or physical tile.
#[derive(Clone, Debug, PartialEq)]
pub struct LogicalStorage {
    pub shape: TensorType,
    pub origin: StorageOrigin,
    pub initialization: Initialization,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StorageOrigin {
    Parameter {
        ordinal: u32,
        path: crate::types::ValuePath,
        name: String,
    },
    /// A boundary result storage. `owner: None` is compiler-owned (an entry
    /// result); `owner: Some(choice)` is occurrence-owned (one call result).
    Result {
        owner: Option<ChoiceId>,
        path: crate::types::ValuePath,
    },
    /// Storage allocated inside the graph that owns it.
    Owned,
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

/// A semantic view of one storage. Access is carried here, transforms are
/// shape-only structure; dynamic selection endpoints are graph values.
#[derive(Clone, Debug, PartialEq)]
pub struct LogicalView {
    pub storage: LogicalStorageId,
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

#[derive(Clone, Debug)]
pub struct CallNode {
    /// The occurrence-specific choice this call selects from.
    pub choice: ChoiceId,
    pub boundary_inputs: Vec<BoundaryInput>,
    pub boundary_results: Vec<BoundaryResult>,
}

/// One interface parameter of a call, at its canonical path. Value parameters
/// pass graph values; borrowed and owned tensors pass explicit states.
#[derive(Clone, Debug)]
pub struct BoundaryInput {
    pub path: crate::types::ValuePath,
    /// Interface parameter ordinal.
    pub param: u32,
    pub kind: BoundaryInputKind,
}

#[derive(Clone, Debug)]
pub enum BoundaryInputKind {
    /// A plain value (scalar, index, range, tuple, capability value).
    Value(GraphValueId),
    /// A shared borrow: consumes the current state without versioning it.
    Shared {
        value: GraphValueId,
        state: StateTokenId,
    },
    /// An exclusive mutable borrow: consumes the current state; the call
    /// produces the next state.
    Exclusive {
        value: GraphValueId,
        state: StateTokenId,
    },
    /// An owned tensor moved into the callee: consumes the current state; the
    /// caller's storage is moved (no later use).
    Move {
        value: GraphValueId,
        state: StateTokenId,
    },
}

/// One result leaf of a call, at its canonical path. Tensor results are
/// occurrence-owned storages allocated by the caller and written by the call;
/// scalar/index/range/tuple leaves are result values; `inout` tensor
/// parameters yield the next state of the caller's storage. A void call has no
/// boundary results and retains its completion as a node of the region.
#[derive(Clone, Debug)]
pub struct BoundaryResult {
    pub path: crate::types::ValuePath,
    pub kind: BoundaryResultKind,
}

#[derive(Clone, Debug)]
pub enum BoundaryResultKind {
    /// A data result value (scalar, index, range, or a tuple of these).
    Value(GraphValueId),
    /// Occurrence-owned (or, at the entry, compiler-owned) result storage,
    /// fully initialized by the call.
    Storage {
        storage: LogicalStorageId,
        ty: TensorType,
        token: StateTokenId,
    },
    /// The next state of an `inout`/exclusively borrowed caller storage.
    State(StateTokenId),
}

// ---------------------------------------------------------------------------
// Runtime extents and safety
// ---------------------------------------------------------------------------

/// A runtime-determined extent: `value` is the retained runtime expression
/// used for all semantics; `capacity` is a resource/tuning bound only (logical
/// construction installs the best proven static upper bound, or the maximal
/// domain when none is proven; planning may lower it).
#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeExtent {
    pub id: RuntimeExtentId,
    pub value: RuntimeScalarExpr,
    pub capacity: u64,
    /// The caller's expected runtime value, when the workload states one for
    /// every entry parameter `value` closes over. Planning prices runtime
    /// domains with it; semantics and resource bounds never read it.
    pub expected: Option<u64>,
}

impl LogicalProgram {
    /// Record the expected runtime value of every runtime extent whose
    /// retained expression closes over entry `index` parameters named in
    /// `expected` (keyed by interface parameter name). Extents over other
    /// values, or over parameters the workload leaves unstated, keep `None`
    /// and are priced at their capacity.
    pub fn install_expected_extents(&mut self, expected: &BTreeMap<String, i64>) {
        if expected.is_empty() {
            return;
        }
        let choice = &self.choices[self.entry_choice];
        let mut values: BTreeMap<GraphValueId, i64> = BTreeMap::new();
        for alternative in choice.alternatives.as_slice() {
            let graph = &self.graphs[alternative.graph];
            for (param, parameter) in choice.interface.params.iter().zip(&graph.parameters) {
                if let (Some(value), RegionParameter::Value { id, .. }) =
                    (expected.get(&param.name), parameter)
                {
                    values.insert(*id, *value);
                }
            }
        }
        // Extents may read earlier extents; ids are allocated in definition order.
        let mut installed: Vec<Option<u64>> = Vec::with_capacity(self.runtime_extents.len());
        for extent in self.runtime_extents.iter_mut() {
            let evaluated = expected_scalar(&extent.value, &values, &installed)
                .and_then(|value| u64::try_from(value).ok());
            extent.expected = evaluated;
            installed.push(evaluated);
        }
    }
}

/// Evaluate a retained runtime scalar under expected entry values; `None`
/// when any leaf is unstated or the arithmetic is undefined.
fn expected_scalar(
    expr: &RuntimeScalarExpr,
    values: &BTreeMap<GraphValueId, i64>,
    extents: &[Option<u64>],
) -> Option<i64> {
    use RuntimeScalarExpr::*;
    let binary = |a: &RuntimeScalarExpr, b: &RuntimeScalarExpr| {
        Some((
            expected_scalar(a, values, extents)?,
            expected_scalar(b, values, extents)?,
        ))
    };
    match expr {
        Const(c) => Some(*c),
        Value(id) => values.get(id).copied(),
        Extent(id) => extents
            .get(id.0 as usize)
            .copied()
            .flatten()
            .and_then(|v| i64::try_from(v).ok()),
        Add(a, b) => binary(a, b).and_then(|(a, b)| a.checked_add(b)),
        Sub(a, b) => binary(a, b).and_then(|(a, b)| a.checked_sub(b)),
        Mul(a, b) => binary(a, b).and_then(|(a, b)| a.checked_mul(b)),
        Div(a, b) => binary(a, b).and_then(|(a, b)| a.checked_div(b)),
        Rem(a, b) => binary(a, b).and_then(|(a, b)| a.checked_rem(b)),
    }
}

/// A retained runtime scalar expression over graph values and other runtime
/// extents. Representation only; evaluation belongs to planning/resolution.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RuntimeScalarExpr {
    Const(i64),
    Value(GraphValueId),
    Extent(RuntimeExtentId),
    Add(Box<RuntimeScalarExpr>, Box<RuntimeScalarExpr>),
    Sub(Box<RuntimeScalarExpr>, Box<RuntimeScalarExpr>),
    Mul(Box<RuntimeScalarExpr>, Box<RuntimeScalarExpr>),
    Div(Box<RuntimeScalarExpr>, Box<RuntimeScalarExpr>),
    Rem(Box<RuntimeScalarExpr>, Box<RuntimeScalarExpr>),
}

/// Runtime obligations created by logical construction; each physical
/// alternative consumes every one of them later as `StaticallyProved` or
/// `RuntimeChecked`.
#[derive(Clone, Debug, PartialEq)]
pub enum SafetyObligation {
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

/// Concrete semantic specialization of the entry: shape and element
/// parameters. Part of the logical identity.
pub struct Specialization {
    pub shapes: BTreeMap<String, i64>,
    pub elems: BTreeMap<String, Elem>,
}

/// Build the logical program of one entry for one effective target under one
/// concrete specialization. Every call occurrence receives its own choice;
/// applicability is evaluated per occurrence against the effective target
/// (portable bodies and same-backend lowerings are peer semantic
/// alternatives).
pub fn construct(
    program: &Program,
    entry: &str,
    target: &EffectiveTargetIdentity,
    supports_intrinsic: &dyn Fn(&IntrinsicUse) -> Result<(), String>,
    shapes: BTreeMap<String, i64>,
    elems: BTreeMap<String, Elem>,
) -> Result<LogicalProgram, LogicalConstructionError> {
    normalize::construct(
        program,
        entry,
        target,
        supports_intrinsic,
        Specialization { shapes, elems },
    )
    .map_err(|error| match error {
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

    fn shapes(entries: &[(&str, i64)]) -> BTreeMap<String, i64> {
        entries.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    /// One representative kernel: allocation, borrowed views, a reduction,
    /// ordered and independent loops, a call, and an atomic join.
    const KERNEL: &str = "fn add[M, N](x: &tensor[M, N] f32, y: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32:\n    let mut output = result\n    parallel for row in 0..M:\n        for col in 0..N:\n            output[row, col] = x[row, col] + y[row, col]\n    return output\n\nfn linear[M, N](x: &tensor[M, N] f32, y: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32:\n    return add(x, y, result)\n";

    #[test]
    fn representative_kernel_constructs_and_verifies() {
        let program = check(&[("kernel.seismic", KERNEL)]).expect("the kernel checks");
        let logical = construct(
            &program,
            "linear",
            &target("cpu"),
            &supports_all,
            shapes(&[("M", 4), ("N", 8)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        logical.verify().expect("the built program verifies");

        // The entry choice and one occurrence-specific call choice.
        assert_eq!(logical.entry_choice, ChoiceId(0));
        assert_eq!(logical.choices.len(), 2);
        let entry = logical.choice(logical.entry_choice);
        assert_eq!(entry.interface.name, "linear");
        assert_eq!(entry.alternatives.iter().count(), 1);

        // The entry graph is a single call whose boundary passes two shared
        // borrows and one move, and receives occurrence-owned result storage.
        let entry_graph = logical.graph(entry.alternatives.iter().next().unwrap().graph);
        let call = entry_graph
            .root
            .nodes
            .iter()
            .find_map(|node| match &node.kind {
                LogicalNodeKind::Call(call) => Some(call),
                _ => None,
            })
            .expect("the entry body is one call");
        let shared = call
            .boundary_inputs
            .iter()
            .filter(|input| matches!(input.kind, BoundaryInputKind::Shared { .. }))
            .count();
        let moved = call
            .boundary_inputs
            .iter()
            .filter(|input| matches!(input.kind, BoundaryInputKind::Move { .. }))
            .count();
        assert_eq!((shared, moved), (2, 1));
        let storage_results = call
            .boundary_results
            .iter()
            .filter(|result| matches!(result.kind, BoundaryResultKind::Storage { .. }))
            .count();
        assert_eq!(storage_results, 1);

        // The callee graph contains the independent outer loop with a
        // disjoint-write join and the ordered inner loop with carries.
        let add_choice = logical
            .choices
            .iter()
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
                    _ => {}
                }
            }
        }
        scan(
            &add_graph.root,
            &mut independent_joins,
            &mut ordered_carries,
        );
        assert!(independent_joins > 0, "the parallel loop joins its visits");
        assert!(ordered_carries > 0, "the ordered loop carries its state");

        // The returned storage is fully initialized: the nested loops cover
        // both axes structurally.
        for storage in add_graph.storages.iter() {
            if matches!(storage.origin, StorageOrigin::Parameter { .. }) {
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
            "sum",
            &target("cpu"),
            &supports_all,
            shapes(&[("N", 16)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        logical.verify().expect("verification succeeds");
        let graph = logical.graph(
            logical
                .choice(logical.entry_choice)
                .alternatives
                .iter()
                .next()
                .unwrap()
                .graph,
        );
        let reduction = graph
            .root
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

        let logical = construct(
            &program,
            "arg",
            &target("cpu"),
            &supports_all,
            shapes(&[("N", 16)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        let graph = logical.graph(
            logical
                .choice(logical.entry_choice)
                .alternatives
                .iter()
                .next()
                .unwrap()
                .graph,
        );
        let reduction = graph
            .root
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
    fn safety_obligations_are_created_not_discharged() {
        let program = check(&[(
            "safety.seismic",
            "fn f[N](x: &tensor[N] f32, i: index[N]) -> f32:\n    return x[i] + 1.0 / x[0]\n",
        )])
        .expect("the source checks");
        let logical = construct(
            &program,
            "f",
            &target("cpu"),
            &supports_all,
            shapes(&[("N", 4)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        let graph = logical.graph(
            logical
                .choice(logical.entry_choice)
                .alternatives
                .iter()
                .next()
                .unwrap()
                .graph,
        );
        let obligations: Vec<&SafetyObligation> = graph
            .root
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
            "f",
            &target("cpu"),
            &supports_all,
            shapes(&[("N", 4)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        logical.verify().expect("verification succeeds");
        let graph = logical.graph(
            logical
                .choice(logical.entry_choice)
                .alternatives
                .iter()
                .next()
                .unwrap()
                .graph,
        );
        // The slice view shares the parameter storage and carries its transform.
        let parameter_storage_id = graph
            .storages
            .ids()
            .zip(graph.storages.iter())
            .find(|(_, storage)| matches!(storage.origin, StorageOrigin::Parameter { .. }))
            .map(|(id, _)| id)
            .expect("the parameter has storage");
        let sliced = graph
            .views
            .iter()
            .find(|view| matches!(view.transform, ViewTransform::Slice { .. }))
            .expect("the slice is a view transform");
        assert_eq!(sliced.storage, parameter_storage_id);
        assert!(graph
            .root
            .nodes
            .iter()
            .flat_map(|node| node.safety.iter())
            .any(|o| matches!(o, SafetyObligation::RangeInBounds { .. })));
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
            "hist",
            &target("metal"),
            &supports_all,
            shapes(&[("N", 128)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds on the same backend");
        logical.verify().expect("verification succeeds");
        let graph = logical.graph(
            logical
                .choice(logical.entry_choice)
                .alternatives
                .iter()
                .next()
                .unwrap()
                .graph,
        );
        let loop_node = graph
            .root
            .nodes
            .iter()
            .find_map(|node| match &node.kind {
                LogicalNodeKind::Loop(loop_node) => Some(loop_node),
                _ => None,
            })
            .expect("the parallel loop is a loop node");
        assert_eq!(loop_node.kind, LoopKind::Independent);
        let join = graph
            .root
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
            "f",
            &target("cpu"),
            &supports_all,
            shapes(&[("M", 4)]),
            BTreeMap::new(),
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
    fn identity_tracks_the_specialization() {
        let program = check(&[("kernel.seismic", KERNEL)]).expect("the kernel checks");
        let a = construct(
            &program,
            "linear",
            &target("cpu"),
            &supports_all,
            shapes(&[("M", 4), ("N", 8)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        let b = construct(
            &program,
            "linear",
            &target("cpu"),
            &supports_all,
            shapes(&[("M", 4), ("N", 8)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        let c = construct(
            &program,
            "linear",
            &target("cpu"),
            &supports_all,
            shapes(&[("M", 5), ("N", 8)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        assert_eq!(a.identity, b.identity);
        assert_ne!(a.identity, c.identity);
    }

    #[test]
    fn builder_rejects_unfinished_and_invalid_graphs() {
        use super::builder::{GraphBuilder, Ids, PrimitiveSpec, WriteEffect};

        // Sealing with an open region fails.
        let mut builder = GraphBuilder::new(ChoiceId(0), 0, Ids::default());
        builder
            .begin_root(Vec::new())
            .expect("the root region opens");
        assert!(builder.seal().is_err(), "a region is still open");

        // A sealable void graph finishes.
        let mut builder = GraphBuilder::new(ChoiceId(0), 0, Ids::default());
        builder.begin_root(Vec::new()).expect("root opens");
        builder
            .end_region(Vec::new())
            .expect("a void graph has no results");
        let (graph, _) = builder.seal().expect("seal succeeds").finish();
        assert!(graph.results.is_empty());

        // Reading a value that does not dominate the use fails.
        let mut builder = GraphBuilder::new(ChoiceId(0), 0, Ids::default());
        builder.begin_root(Vec::new()).expect("root opens");
        let spec = PrimitiveSpec {
            op: PrimitiveOp::Constant(crate::sir::Literal::Int(1)),
            inputs: vec![GraphValueId(99)],
            reads: Vec::new(),
            write: None,
            outputs: vec![Output::Value(ValueType::Scalar(DType::I32))],
            safety: Vec::new(),
            span: Span::default(),
        };
        assert!(builder.add_primitive(spec).is_err());
        let _ = builder.end_region(Vec::new());

        // Reading uninitialized storage fails; writes initialize it.
        let mut builder = GraphBuilder::new(ChoiceId(0), 0, Ids::default());
        builder.begin_root(Vec::new()).expect("root opens");
        let storage = builder.declare_storage(
            TensorType::new(
                vec![ExtentExpr::Static(4)],
                crate::types::Elem::Dtype(DType::F32),
            ),
            StorageOrigin::Owned,
            Initialization::Uninitialized,
        );
        let token = builder.fresh_state();
        builder.bind_state(token, storage);
        builder.set_current_state(storage, token);
        let view = builder.declare_view(
            storage,
            TensorType::new(
                vec![ExtentExpr::Static(4)],
                crate::types::Elem::Dtype(DType::F32),
            ),
            Access::Exclusive,
            ViewTransform::Identity,
        );
        let value = builder
            .fresh_value(
                ValueType::Tensor(TensorType::new(
                    vec![ExtentExpr::Static(4)],
                    crate::types::Elem::Dtype(DType::F32),
                )),
                Some(view),
            )
            .expect("the value is backed by the view");
        let read = PrimitiveSpec {
            op: PrimitiveOp::Primitive(PrimitiveId::Materialize),
            inputs: vec![value],
            reads: vec![storage],
            write: None,
            outputs: Vec::new(),
            safety: Vec::new(),
            span: Span::default(),
        };
        assert!(
            builder.add_primitive(read).is_err(),
            "uninitialized storage cannot be read"
        );
        // A whole write initializes the storage; a partial write does not.
        let write = PrimitiveSpec {
            op: PrimitiveOp::Primitive(PrimitiveId::Fill {
                value: 0.0,
                dtype: DType::F32,
            }),
            inputs: Vec::new(),
            reads: Vec::new(),
            write: Some(WriteEffect {
                storage,
                coverage: Coverage::full(1),
                atomic: false,
                initializing: true,
            }),
            outputs: Vec::new(),
            safety: Vec::new(),
            span: Span::default(),
        };
        builder.add_primitive(write).expect("the write initializes");
        assert_eq!(
            builder.initialization(storage),
            &Initialization::FullyInitialized
        );

        // Branch regions must share parameter schemas.
        let mut builder = GraphBuilder::new(ChoiceId(0), 0, Ids::default());
        builder.begin_root(Vec::new()).expect("root opens");
        let condition = builder
            .fresh_value(ValueType::Scalar(DType::Bool), None)
            .expect("the condition allocates");
        builder.begin_region(Vec::new()).expect("then opens");
        let then_region = builder.end_region(Vec::new()).expect("then closes");
        builder
            .begin_region(vec![RegionParameter::Value {
                id: condition,
                ty: ValueType::Scalar(DType::Bool),
            }])
            .expect("else opens");
        let else_region = builder.end_region(Vec::new()).expect("else closes");
        assert!(builder
            .add_if(
                condition,
                then_region,
                else_region,
                Vec::new(),
                Vec::new(),
                Span::default()
            )
            .is_err());
        let _ = builder.end_region(Vec::new());
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
            "fill",
            &target("cpu"),
            &supports_all,
            shapes(&[("M", 4), ("N", 4)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        logical.verify().expect("verification succeeds");
        let graph = logical.graph(
            logical
                .choice(logical.entry_choice)
                .alternatives
                .iter()
                .next()
                .unwrap()
                .graph,
        );
        let out_storage = graph
            .storages
            .iter()
            .find(|storage| matches!(storage.origin, StorageOrigin::Parameter { .. }))
            .expect("the output parameter has storage");
        assert_eq!(out_storage.initialization, Initialization::FullyInitialized);

        // The builder-level lattice: a partial write leaves an axis uncovered
        // and two disjoint partial writes compose to full coverage.
        use super::builder::{GraphBuilder, Ids, PrimitiveSpec, WriteEffect};
        let mut builder = GraphBuilder::new(ChoiceId(0), 0, Ids::default());
        builder.begin_root(Vec::new()).expect("root opens");
        let storage = builder.declare_storage(
            TensorType::new(
                vec![ExtentExpr::Static(2), ExtentExpr::Static(2)],
                crate::types::Elem::Dtype(DType::F32),
            ),
            StorageOrigin::Owned,
            Initialization::Uninitialized,
        );
        let write = |builder: &mut GraphBuilder<Building>, coverage: Coverage| {
            let spec = PrimitiveSpec {
                op: PrimitiveOp::Primitive(PrimitiveId::Fill {
                    value: 0.0,
                    dtype: DType::F32,
                }),
                inputs: Vec::new(),
                reads: Vec::new(),
                write: Some(WriteEffect {
                    storage,
                    coverage,
                    atomic: false,
                    initializing: false,
                }),
                outputs: Vec::new(),
                safety: Vec::new(),
                span: Span::default(),
            };
            builder.add_primitive(spec).expect("the write applies");
        };
        let _ = builder.fresh_state();
        builder.set_current_state(storage, StateTokenId(0));
        write(
            &mut builder,
            Coverage {
                axes: vec![true, false],
            },
        );
        assert_eq!(
            builder.initialization(storage),
            &Initialization::PartiallyInitialized(Coverage {
                axes: vec![true, false]
            })
        );
        write(
            &mut builder,
            Coverage {
                axes: vec![false, true],
            },
        );
        assert_eq!(
            builder.initialization(storage),
            &Initialization::FullyInitialized
        );
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
            "f",
            &target("cpu"),
            &supports_all,
            shapes(&[("M", 4)]),
            BTreeMap::new(),
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
            "f",
            &target("metal"),
            &supports_all,
            shapes(&[("M", 4)]),
            BTreeMap::new(),
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
            "g",
            &target("metal"),
            &|_| Ok(()),
            shapes(&[("M", 4)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        logical.verify().expect("verification succeeds");
        // Rejecting it: the inner occurrence has no implementation.
        let error = construct(
            &program,
            "g",
            &target("metal"),
            &|_| Err("no subgroup hardware".to_string()),
            shapes(&[("M", 4)]),
            BTreeMap::new(),
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
