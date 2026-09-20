//! Specialized logical execution contract.
//!
//! This module is the semantic boundary consumed by backend planning.  It names
//! logical values and owned storage, but deliberately has no memory spaces,
//! launches, participants, physical tiles, allocations, or native identifiers.

use crate::{
    family::{self, TargetEnvironment, Workload},
    intrinsics::Operation,
    precision::NumericalEffect,
    sir::{self, Mode, Program},
    span::Span,
    sym::{Atom, Sym},
    types::{DType, Elem, Extent, Shaped, Ty},
};
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LogicalCompilationIdentity {
    pub semantic_program: [u8; 32],
    pub workload: [u8; 32],
}

struct StableSha256(sha2::Sha256);

impl Default for StableSha256 {
    fn default() -> Self {
        use sha2::Digest;
        Self(sha2::Sha256::new())
    }
}

impl Hasher for StableSha256 {
    fn finish(&self) -> u64 {
        use sha2::Digest;
        let bytes = self.0.clone().finalize();
        u64::from_le_bytes(bytes[..8].try_into().expect("sha256 prefix"))
    }

    fn write(&mut self, bytes: &[u8]) {
        use sha2::Digest;
        self.0.update((bytes.len() as u64).to_le_bytes());
        self.0.update(bytes);
    }

    fn write_u8(&mut self, value: u8) {
        self.write(&value.to_le_bytes());
    }
    fn write_u16(&mut self, value: u16) {
        self.write(&value.to_le_bytes());
    }
    fn write_u32(&mut self, value: u32) {
        self.write(&value.to_le_bytes());
    }
    fn write_u64(&mut self, value: u64) {
        self.write(&value.to_le_bytes());
    }
    fn write_u128(&mut self, value: u128) {
        self.write(&value.to_le_bytes());
    }
    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }
    fn write_i8(&mut self, value: i8) {
        self.write(&value.to_le_bytes());
    }
    fn write_i16(&mut self, value: i16) {
        self.write(&value.to_le_bytes());
    }
    fn write_i32(&mut self, value: i32) {
        self.write(&value.to_le_bytes());
    }
    fn write_i64(&mut self, value: i64) {
        self.write(&value.to_le_bytes());
    }
    fn write_i128(&mut self, value: i128) {
        self.write(&value.to_le_bytes());
    }
    fn write_isize(&mut self, value: isize) {
        self.write_i64(value as i64);
    }
}

fn stable_identity(value: &impl Hash) -> [u8; 32] {
    use sha2::Digest;
    let mut hasher = StableSha256::default();
    value.hash(&mut hasher);
    hasher.0.finalize().into()
}

mod body_specialize;

macro_rules! identity {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub u32);

        impl $name {
            fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

identity!(ValueId);
identity!(StorageId);
identity!(ViewId);
identity!(ChoiceId);
identity!(ResultSlotId);
identity!(TaskGraphId);
identity!(TaskId);
identity!(CallBoundaryId);
identity!(OperandId);
identity!(DependencyId);
identity!(LocalValueId);
identity!(LocalStorageId);
identity!(LocalViewId);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorType {
    pub shape: Vec<Sym>,
    pub elem: Elem,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Scalar(DType),
    Index {
        bound: Sym,
    },
    Range {
        bound: Sym,
    },
    Tensor(TensorType),
    Tuple(Vec<Type>),
    CapabilityValue {
        target: String,
        name: String,
        shape: Vec<Sym>,
        elem: Option<Elem>,
    },
    Void,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Access {
    Shared,
    Exclusive,
}

/// Logical geometry over one storage identity.  A reshape changes `shape` but
/// retains the same storage.  Backend layouts and byte strides do not belong here.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct View {
    pub storage: StorageId,
    pub shape: Vec<Sym>,
    pub elem: Elem,
    pub access: Access,
    pub transform: ViewTransform,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ViewTransform {
    Identity,
    Reshape {
        source_shape: Vec<Sym>,
    },
    Transpose {
        permutation: Vec<u32>,
    },
    /// A bounded logical selection.  Dynamic endpoints are values in the same
    /// logical program; their range refinements are checked before construction.
    Slice {
        axes: Vec<SliceAxis>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SliceAxis {
    Full,
    Point(ValueId),
    Range {
        start: Option<ValueId>,
        end: Option<ValueId>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum StorageOrigin {
    Parameter {
        ordinal: u32,
        path: Vec<u32>,
        name: String,
    },
    /// Stable logical result path.  Physical ABI flattening consumes this path;
    /// source and ABI spelling are never native identifiers.
    Result {
        path: Vec<u32>,
    },
    Owned,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Storage {
    pub ty: TensorType,
    pub origin: StorageOrigin,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Value {
    pub ty: Type,
    pub kind: ValueKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ValueKind {
    ScalarParameter {
        ordinal: u32,
        name: String,
    },
    RangeParameter {
        ordinal: u32,
        name: String,
    },
    Scalar,
    Tensor(ViewId),
    Tuple(Vec<ValueId>),
    /// Value supplied at the common result port of one choice.
    ChoiceResult {
        choice: ChoiceId,
        result: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResultSlot {
    pub id: ResultSlotId,
    pub path: Vec<u32>,
    pub storage: StorageId,
    pub ty: TensorType,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Effect {
    Read(StorageId),
    Write(StorageId),
    Move(StorageId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PortEffect {
    Read { port: u32, path: Vec<u32> },
    Write { port: u32, path: Vec<u32> },
    MoveResult { port: u32, path: Vec<u32> },
}

/// A typed port is the only identity visible both outside and inside an
/// implementation branch. All other identities in a fragment are branch-local.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Port {
    pub path: Vec<u32>,
    pub ty: Type,
    pub access: Option<Access>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LogicalExtent {
    Static(i64),
    Dynamic(ValueId),
    /// A runtime-bound semantic extent retained as an expression (for example
    /// a callee dimension bound to the length of a caller range).
    Symbolic(Sym),
    /// An authored structural partition whose concrete width is selected by
    /// physical planning. This is logical geometry, not a thread/block width.
    Structural {
        choice: ChoiceId,
        alternative: u32,
        site: u32,
        upper_bound: i64,
    },
}

/// One authored structural extent, identified independently of its numeric value.
/// The choice/alternative pair is part of the identity because a site only exists
/// while that implementation branch is active.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StructuralSiteRef {
    pub choice: ChoiceId,
    pub alternative: u32,
    pub site: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LogicalAlternativeRef {
    pub choice: ChoiceId,
    pub alternative: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum StructuralConstraintKind {
    Multiple(i64),
    AtLeast(i64),
    AtMost(i64),
    Equal(i64),
    Divides(i64),
}

/// A numeric condition contributed by an implementation alternative. `active_if`
/// is deliberately distinct from `site`: a nested candidate may constrain a site
/// owned by an ancestor candidate.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StructuralConstraint {
    pub active_if: LogicalAlternativeRef,
    pub site: StructuralSiteRef,
    pub kind: StructuralConstraintKind,
}

/// The value of `refinement` divides the value of `refined`. Both endpoints carry
/// their owning branch, so activation is the conjunction of those branches.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StructuralRefinement {
    pub refinement: StructuralSiteRef,
    pub refined: StructuralSiteRef,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LocalStorageOrigin {
    Constructed,
    Snapshot {
        source: StorageRef,
    },
    Clone {
        source: LocalStorageId,
    },
    /// Destination storage of one occurrence-specific call result. This is
    /// created during logical specialization, before physical planning.
    CallResult {
        choice: ChoiceId,
        path: Vec<u32>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LocalStorage {
    pub ty: TensorType,
    pub origin: LocalStorageOrigin,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LocalView {
    pub storage: StorageRef,
    pub shape: Vec<Sym>,
    pub elem: Elem,
    pub access: Access,
    pub transform: LocalViewTransform,
}

/// Task-local views contain only shape-only transforms. Dynamic slicing is a
/// `LogicalIndex`, whose endpoint expressions are occurrence-local values; a
/// local view cannot retain program-wide `ValueId`s.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LocalViewTransform {
    Identity,
    Reshape { source_shape: Vec<Sym> },
    Transpose { permutation: Vec<u32> },
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ValueRef {
    Input(u32),
    Local(LocalValueId),
    Result(u32),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StorageRef {
    Input { port: u32, path: Vec<u32> },
    Result { port: u32, path: Vec<u32> },
    Local(LocalStorageId),
}

#[derive(Clone, Debug, PartialEq)]
pub struct LogicalExpr {
    pub ty: Type,
    pub kind: LogicalExprKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LogicalIndex {
    Point(Box<LogicalExpr>),
    Coordinate(LocalValueId),
    Slice(u32),
    Range {
        start: Option<Box<LogicalExpr>>,
        end: Option<Box<LogicalExpr>>,
    },
}

/// Target-neutral checked expression vocabulary. Unlike execution IR this has
/// no tile, memory-space, participant, launch or native-value category.
#[derive(Clone, Debug, PartialEq)]
pub enum LogicalExprKind {
    Int(i64),
    Float(u64),
    Bool(bool),
    Value(ValueRef),
    Shape(Sym),
    Tuple(Vec<LogicalExpr>),
    Range(Box<LogicalExpr>, Box<LogicalExpr>),
    Field(Box<LogicalExpr>, usize),
    Construct {
        storage: LocalStorageId,
    },
    Filled {
        storage: LocalStorageId,
        like: Box<LogicalExpr>,
        value: u64,
    },
    View {
        view: LocalViewId,
        base: Box<LogicalExpr>,
    },
    Index {
        base: Box<LogicalExpr>,
        indices: Vec<LogicalIndex>,
    },
    Snapshot {
        storage: LocalStorageId,
        source: Box<LogicalExpr>,
    },
    Decode {
        storage: LocalStorageId,
        source: Box<LogicalExpr>,
    },
    /// An owned logical producer with an explicit destination identity.
    Materialize {
        storage: LocalStorageId,
        value: Box<LogicalExpr>,
    },
    Cast {
        dtype: DType,
        expr: Box<LogicalExpr>,
    },
    Unary {
        op: crate::syntax::ast::UnaryOp,
        expr: Box<LogicalExpr>,
    },
    Binary {
        op: crate::syntax::ast::BinaryOp,
        lhs: Box<LogicalExpr>,
        rhs: Box<LogicalExpr>,
    },
    Math {
        op: sir::Math,
        args: Vec<LogicalExpr>,
    },
    Select {
        cond: Box<LogicalExpr>,
        then: Box<LogicalExpr>,
        els: Box<LogicalExpr>,
    },
    Reduce {
        value: Box<LogicalExpr>,
        axis: usize,
        op: sir::ReduceOp,
        unordered: bool,
    },
    Coordinate(LocalValueId),
    Extent {
        base: Box<LogicalExpr>,
        axis: usize,
    },
    Call {
        choice: ChoiceId,
        args: Vec<LogicalExpr>,
        results: Vec<CallResultStorage>,
    },
    Region(Box<LogicalRegion>),
    Intrinsic {
        operation: Operation,
        args: Vec<LogicalExpr>,
    },
    Accessor {
        base: Box<LogicalExpr>,
        name: String,
    },
    Geometry {
        base: Box<LogicalExpr>,
        axis: usize,
        valid: bool,
    },
    Atomic {
        op: crate::syntax::ast::BinaryOp,
        place: Box<LogicalExpr>,
        value: Box<LogicalExpr>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallResultStorage {
    pub path: Vec<u32>,
    pub storage: LocalStorageId,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResultWrite {
    pub port: u32,
    pub path: Vec<u32>,
    pub value: LogicalExpr,
    pub transfer: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LogicalOperation {
    pub kind: LogicalOperationKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LogicalOperationKind {
    Bind {
        pattern: LogicalPattern,
        value: LogicalExpr,
    },
    Assign {
        target: LogicalExpr,
        op: crate::syntax::ast::AssignOp,
        value: LogicalExpr,
    },
    Region(LogicalRegion),
    Stages(Vec<LogicalStage>),
    For {
        independent: bool,
        binder: LocalValueId,
        lo: LogicalExpr,
        hi: LogicalExpr,
        source: Option<LogicalExpr>,
        body: LogicalBlock,
    },
    Coordinates {
        binders: Vec<LocalValueId>,
        value: LogicalExpr,
        axes: Vec<usize>,
        body: LogicalBlock,
    },
    Members {
        binder: LocalValueId,
        slice: u32,
        body: LogicalBlock,
    },
    If {
        condition: LogicalExpr,
        then: LogicalBlock,
        els: LogicalBlock,
    },
    Publish {
        value: LogicalExpr,
        destination: LogicalExpr,
    },
    Yield(Vec<LogicalExpr>),
    /// An explicit logical reduction. Reduction is a task operation rather than
    /// an expression-site mapping decision.
    Reduction {
        binder: LocalValueId,
        value: LogicalExpr,
        axis: usize,
        op: sir::ReduceOp,
        unordered: bool,
    },
    /// Phi-like merge of mutually exclusive conditional task results.
    ConditionalMerge {
        binder: LocalValueId,
        cases: Vec<LogicalConditionalCase>,
    },
    /// Destination-passing completion of this choice's output ports.
    Return(Vec<ResultWrite>),
    Expr(LogicalExpr),
}

pub type LogicalBlock = Vec<LogicalOperation>;

/// Choice-free scalar work belonging to one coherent logical domain.
/// Scheduling constructs and call/reduction expressions are rejected by
/// verification; specialization removes them while constructing the task graph.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarBody {
    pub operations: LogicalBlock,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LogicalPattern {
    Value(LocalValueId),
    Tuple(Vec<LogicalPattern>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct LogicalRegion {
    pub id: u32,
    pub mode: sir::RegionMode,
    pub binders: Vec<LocalValueId>,
    pub source: Option<Box<LogicalExpr>>,
    pub body: LogicalBlock,
    pub merge: Option<LogicalMerge>,
    pub result: Option<Type>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LogicalMerge {
    pub left: LogicalPattern,
    pub right: LogicalPattern,
    pub identity: LogicalExpr,
    pub body: LogicalBlock,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LogicalStage {
    pub name: String,
    pub ports: Vec<LocalValueId>,
    pub body: LogicalBlock,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LogicalOperand {
    pub id: OperandId,
    pub ty: Type,
    pub value: ValueRef,
    pub storage: Option<StorageRef>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AxisOrder {
    Independent,
    Ordered,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LogicalAxisSource {
    Range {
        lo: LogicalExpr,
        hi: LogicalExpr,
        runtime: Option<LogicalExpr>,
    },
    Coordinates {
        value: LogicalExpr,
        axes: Vec<usize>,
    },
    Members {
        slice: u32,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct LogicalAxis {
    pub binders: Vec<LocalValueId>,
    pub order: AxisOrder,
    pub source: LogicalAxisSource,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct LogicalDomain {
    pub axes: Vec<LogicalAxis>,
    pub regions: Vec<LogicalRegionFrame>,
    pub predicates: Vec<LogicalPredicate>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LogicalPredicate {
    pub condition: LogicalExpr,
    pub when_true: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LogicalConditionalCase {
    pub predicates: Vec<LogicalPredicate>,
    pub value: OperandId,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LogicalRegionFrame {
    pub id: u32,
    pub mode: sir::RegionMode,
    pub binders: Vec<LocalValueId>,
    pub source: Option<LogicalExpr>,
    pub merge: Option<LogicalMergeSignature>,
    pub result: Option<Type>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LogicalMergeSignature {
    pub left: LogicalPattern,
    pub right: LogicalPattern,
    pub identity: LogicalExpr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LogicalAccessKind {
    Read,
    Write,
    Move,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LogicalAccess {
    pub storage: StorageRef,
    pub view: Option<LocalViewId>,
    pub kind: LogicalAccessKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct EffectFootprint {
    pub accesses: Vec<LogicalAccess>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LogicalTask {
    pub id: TaskId,
    pub domain: LogicalDomain,
    pub body: ScalarBody,
    pub inputs: Vec<OperandId>,
    pub outputs: Vec<OperandId>,
    pub effects: EffectFootprint,
    pub capabilities: BTreeSet<String>,
    pub numerical_semantics: Vec<NumericalEffect>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LogicalCallBoundary {
    pub id: CallBoundaryId,
    pub choice: ChoiceId,
    pub inputs: Vec<OperandId>,
    /// Whole expression result used by scalar consumers. `outputs` are the
    /// flattened typed choice ports used for graph composition.
    pub result: OperandId,
    pub outputs: Vec<OperandId>,
    pub domain: LogicalDomain,
    pub results: Vec<CallResultStorage>,
    pub effects: EffectFootprint,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogicalEndpoint {
    Input(u32),
    Task(TaskId),
    Call(CallBoundaryId),
    Output(u32),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogicalDependencyKind {
    Control,
    Value(OperandId),
    Effect(StorageRef),
    Ownership(StorageRef),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LogicalDependency {
    pub id: DependencyId,
    pub from: LogicalEndpoint,
    pub to: LogicalEndpoint,
    pub kind: LogicalDependencyKind,
}

/// Scheduling-normal work graph for one specialized implementation
/// alternative. Backend planning consumes tasks and dependency obligations;
/// it never receives an opaque whole-function body.
#[derive(Clone, Debug, PartialEq)]
pub struct LogicalTaskGraph {
    pub id: TaskGraphId,
    pub choice: ChoiceId,
    pub alternative: u32,
    pub inputs: Vec<Port>,
    pub results: Vec<Port>,
    /// Candidate parameter ordinal -> common input port ordinal.
    pub input_remap: Vec<u32>,
    /// Candidate result leaf ordinal -> common result port ordinal.
    pub result_remap: Vec<u32>,
    pub values: Vec<Type>,
    pub storage: Vec<LocalStorage>,
    pub views: Vec<LocalView>,
    /// Occurrence-qualified runtime extent values, expressed in this
    /// fragment's scalar environment. Capacity bounds live on LogicalProgram.
    pub runtime_extents: BTreeMap<String, LogicalExpr>,
    pub operands: Vec<LogicalOperand>,
    pub tasks: Vec<LogicalTask>,
    pub calls: Vec<LogicalCallBoundary>,
    pub dependencies: Vec<LogicalDependency>,
}

impl LogicalTaskGraph {
    pub fn task(&self, id: TaskId) -> Option<&LogicalTask> {
        self.tasks.get(id.index())
    }

    pub fn call(&self, id: CallBoundaryId) -> Option<&LogicalCallBoundary> {
        self.calls.get(id.index())
    }

    pub fn operand(&self, id: OperandId) -> Option<&LogicalOperand> {
        self.operands.get(id.index())
    }

    pub fn dependency(&self, id: DependencyId) -> Option<&LogicalDependency> {
        self.dependencies.get(id.index())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Interface {
    pub inputs: Vec<Type>,
    pub results: Vec<Type>,
    pub effects: Vec<Effect>,
    pub port_effects: Vec<PortEffect>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Alternative {
    /// Stable semantic definition identity.  This is provenance, not a native
    /// function symbol and not an implementation priority.
    pub definition: u32,
    pub interface: Interface,
    pub capabilities: BTreeSet<String>,
    pub numerical_effects: Vec<NumericalEffect>,
    pub task_graph: TaskGraphId,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Choice {
    /// Temporary construction provenance. Its numeric value is not a solver
    /// assignment and has no meaning after specialization.
    pub occurrence: u32,
    pub interface: Interface,
    pub inputs: Vec<Port>,
    pub results: Vec<Port>,
    pub alternatives: Vec<Alternative>,
}

/// One entry specialized for one workload and effective target.  It remains
/// choice-bearing: physical planning selects semantic and physical alternatives
/// jointly.
#[derive(Clone, Debug, PartialEq)]
pub struct LogicalProgram {
    pub identity: LogicalCompilationIdentity,
    pub entry: String,
    pub target: String,
    pub capability_fingerprint: String,
    pub shapes: BTreeMap<String, i64>,
    pub elems: BTreeMap<String, Elem>,
    pub storage: Vec<Storage>,
    pub views: Vec<View>,
    pub values: Vec<Value>,
    pub result_slots: Vec<ResultSlot>,
    pub choices: Vec<Choice>,
    pub entry_choice: ChoiceId,
    pub extents: Vec<LogicalExtent>,
    pub structural_constraints: Vec<StructuralConstraint>,
    pub structural_refinements: Vec<StructuralRefinement>,
    /// Compile-time capacity bounds for occurrence-qualified runtime extent
    /// symbols. Bounds are resource facts; they never replace runtime values.
    pub extent_bounds: BTreeMap<String, Sym>,
    pub task_graphs: Vec<LogicalTaskGraph>,
}

impl LogicalProgram {
    pub fn task_graph(&self, id: TaskGraphId) -> Option<&LogicalTaskGraph> {
        self.task_graphs.get(id.index())
    }

    pub fn extent_capacity(&self, extent: &Sym) -> Option<i64> {
        fn evaluate(
            program: &LogicalProgram,
            extent: &Sym,
            visiting: &std::cell::RefCell<BTreeSet<String>>,
        ) -> Option<i64> {
            extent.eval(&|name| {
                if !visiting.borrow_mut().insert(name.to_string()) {
                    return None;
                }
                let value = program
                    .extent_bounds
                    .get(name)
                    .and_then(|bound| evaluate(program, bound, visiting))
                    .or_else(|| {
                        name.strip_prefix("@site").and_then(|site| {
                            let site = site.parse::<u32>().ok()?;
                            program.extents.iter().find_map(|extent| match extent {
                                LogicalExtent::Structural {
                                    site: candidate,
                                    upper_bound,
                                    ..
                                } if *candidate == site => Some(*upper_bound),
                                _ => None,
                            })
                        })
                    });
                visiting.borrow_mut().remove(name);
                value
            })
        }
        evaluate(self, extent, &std::cell::RefCell::new(BTreeSet::new()))
    }

    pub fn verify(&self) -> Result<(), String> {
        self.verify_storage_origins()?;
        self.verify_views()?;
        self.verify_values()?;
        self.verify_results()?;
        self.verify_choices()?;
        self.verify_task_graphs()?;
        self.verify_extents()
    }

    fn storage(&self, id: StorageId) -> Result<&Storage, String> {
        self.storage
            .get(id.index())
            .ok_or_else(|| format!("storage#{} is outside the logical program", id.0))
    }

    fn value(&self, id: ValueId) -> Result<&Value, String> {
        self.values
            .get(id.index())
            .ok_or_else(|| format!("value#{} is outside the logical program", id.0))
    }

    fn verify_storage_origins(&self) -> Result<(), String> {
        let mut parameters = BTreeSet::new();
        let mut result_paths = BTreeSet::new();
        for storage in &self.storage {
            match &storage.origin {
                StorageOrigin::Parameter { ordinal, path, .. }
                    if !parameters.insert((*ordinal, path.clone())) =>
                {
                    return Err(format!(
                        "tensor parameter ordinal {ordinal} path {path:?} owns two storage identities"
                    ));
                }
                StorageOrigin::Result { path } if !result_paths.insert(path.clone()) => {
                    return Err(format!("result path {path:?} owns two storage identities"));
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn verify_views(&self) -> Result<(), String> {
        for (ordinal, view) in self.views.iter().enumerate() {
            let storage = self.storage(view.storage)?;
            if view.elem != storage.ty.elem {
                return Err(format!(
                    "view#{ordinal} changes the element type of storage#{}",
                    view.storage.0
                ));
            }
            match &view.transform {
                ViewTransform::Identity if view.shape != storage.ty.shape => {
                    return Err(format!("identity view#{ordinal} changes storage geometry"));
                }
                ViewTransform::Reshape { source_shape } if source_shape != &storage.ty.shape => {
                    return Err(format!(
                        "reshape view#{ordinal} records the wrong source geometry"
                    ));
                }
                ViewTransform::Transpose { permutation } => {
                    let rank = storage.ty.shape.len();
                    if permutation.len() != rank {
                        return Err(format!(
                            "transpose view#{ordinal} has the wrong permutation rank"
                        ));
                    }
                    let axes = permutation.iter().copied().collect::<BTreeSet<_>>();
                    if axes.len() != rank || axes.iter().copied().ne(0..rank as u32) {
                        return Err(format!("transpose view#{ordinal} is not a permutation"));
                    }
                }
                ViewTransform::Slice { axes } => {
                    if axes.len() != storage.ty.shape.len() {
                        return Err(format!("slice view#{ordinal} has the wrong source rank"));
                    }
                    for axis in axes {
                        match axis {
                            SliceAxis::Point(value) => {
                                self.value(*value)?;
                            }
                            SliceAxis::Range { start, end } => {
                                for value in start.iter().chain(end.iter()) {
                                    self.value(*value)?;
                                }
                            }
                            SliceAxis::Full => {}
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn verify_values(&self) -> Result<(), String> {
        for (ordinal, value) in self.values.iter().enumerate() {
            match &value.kind {
                ValueKind::Tensor(view) => {
                    let view = self
                        .views
                        .get(view.index())
                        .ok_or_else(|| format!("value#{ordinal} names an absent view"))?;
                    let expected = Type::Tensor(TensorType {
                        shape: view.shape.clone(),
                        elem: view.elem.clone(),
                    });
                    if value.ty != expected {
                        return Err(format!(
                            "value#{ordinal} tensor type disagrees with its view"
                        ));
                    }
                }
                ValueKind::Tuple(items) => {
                    let members = items
                        .iter()
                        .map(|id| self.value(*id).map(|value| value.ty.clone()))
                        .collect::<Result<Vec<_>, _>>()?;
                    if value.ty != Type::Tuple(members) {
                        return Err(format!(
                            "value#{ordinal} tuple type disagrees with its members"
                        ));
                    }
                }
                ValueKind::ChoiceResult { choice, result } => {
                    let choice = self
                        .choices
                        .get(choice.index())
                        .ok_or_else(|| format!("value#{ordinal} names an absent choice"))?;
                    let expected = choice
                        .interface
                        .results
                        .get(*result as usize)
                        .ok_or_else(|| format!("value#{ordinal} names an absent choice result"))?;
                    if &value.ty != expected {
                        return Err(format!(
                            "value#{ordinal} disagrees with its choice result type"
                        ));
                    }
                }
                ValueKind::ScalarParameter { .. } | ValueKind::Scalar => {
                    if !matches!(value.ty, Type::Scalar(_) | Type::Index { .. }) {
                        return Err(format!(
                            "value#{ordinal} is scalar but has a non-scalar type"
                        ));
                    }
                }
                ValueKind::RangeParameter { .. } => {
                    if !matches!(value.ty, Type::Range { .. }) {
                        return Err(format!(
                            "value#{ordinal} is a range parameter but has another type"
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn verify_results(&self) -> Result<(), String> {
        let mut paths = BTreeSet::new();
        let mut storage = BTreeSet::new();
        for (ordinal, slot) in self.result_slots.iter().enumerate() {
            if slot.id.index() != ordinal {
                return Err(format!(
                    "result slot#{} is stored at position {ordinal}",
                    slot.id.0
                ));
            }
            if !paths.insert(slot.path.clone()) {
                return Err(format!("result path {:?} occurs twice", slot.path));
            }
            if !storage.insert(slot.storage) {
                return Err(format!(
                    "result storage#{} backs two result slots",
                    slot.storage.0
                ));
            }
            let backing = self.storage(slot.storage)?;
            if backing.ty != slot.ty {
                return Err(format!(
                    "result slot#{ordinal} type disagrees with its storage"
                ));
            }
            if backing.origin
                != (StorageOrigin::Result {
                    path: slot.path.clone(),
                })
            {
                return Err(format!(
                    "result slot#{ordinal} storage has the wrong origin"
                ));
            }
        }
        let declared = self
            .storage
            .iter()
            .filter_map(|storage| match &storage.origin {
                StorageOrigin::Result { path } => Some(path),
                _ => None,
            })
            .count();
        if declared != self.result_slots.len() {
            return Err("a result storage has no result slot".into());
        }
        Ok(())
    }

    fn verify_choices(&self) -> Result<(), String> {
        self.choices
            .get(self.entry_choice.index())
            .ok_or_else(|| "entry choice is outside the logical program".to_string())?;
        for (ordinal, choice) in self.choices.iter().enumerate() {
            if choice.occurrence as usize != ordinal {
                return Err(format!(
                    "choice#{} has unstable occurrence {}",
                    ordinal, choice.occurrence
                ));
            }
            if choice.alternatives.is_empty() {
                return Err(format!("choice#{ordinal} has no applicable implementation"));
            }
            for alternative in &choice.alternatives {
                if alternative.interface != choice.interface {
                    return Err(format!(
                        "choice#{ordinal} alternative changes its logical interface"
                    ));
                }
                let graph = self
                    .task_graphs
                    .get(alternative.task_graph.index())
                    .ok_or_else(|| {
                        format!("choice#{ordinal} alternative names an absent task graph")
                    })?;
                if graph.choice != ChoiceId(ordinal as u32) {
                    return Err(format!(
                        "choice#{ordinal} alternative task graph belongs to another choice"
                    ));
                }
                for effect in &alternative.interface.effects {
                    let storage = match effect {
                        Effect::Read(storage) | Effect::Write(storage) | Effect::Move(storage) => {
                            *storage
                        }
                    };
                    self.storage(storage)?;
                }
            }
        }
        Ok(())
    }

    fn verify_task_graphs(&self) -> Result<(), String> {
        let mut linked = BTreeSet::new();
        for choice in &self.choices {
            for alternative in &choice.alternatives {
                if !linked.insert(alternative.task_graph) {
                    return Err(format!(
                        "task graph#{} is shared by alternatives",
                        alternative.task_graph.0
                    ));
                }
            }
        }
        if linked.len() != self.task_graphs.len() {
            return Err("a logical task graph is not owned by an alternative".into());
        }
        for (ordinal, graph) in self.task_graphs.iter().enumerate() {
            if graph.id.index() != ordinal {
                return Err(format!(
                    "task graph#{} is stored at position {ordinal}",
                    graph.id.0
                ));
            }
            let choice = self
                .choices
                .get(graph.choice.index())
                .ok_or_else(|| format!("task graph#{ordinal} names an absent choice"))?;
            if graph.alternative as usize >= choice.alternatives.len() {
                return Err(format!(
                    "task graph#{ordinal} has an absent alternative ordinal"
                ));
            }
            if graph.inputs != choice.inputs || graph.results != choice.results {
                return Err(format!("task graph#{ordinal} changes its choice ports"));
            }
            if graph.input_remap.len() != graph.inputs.len() {
                return Err(format!("task graph#{ordinal} does not remap every input"));
            }
            for (storage_ordinal, storage) in graph.storage.iter().enumerate() {
                match &storage.origin {
                    LocalStorageOrigin::Snapshot { source } => verify_storage_ref(graph, source)
                        .map_err(|reason| {
                            format!("task graph#{ordinal} storage#{storage_ordinal}: {reason}")
                        })?,
                    LocalStorageOrigin::Clone { source }
                        if source.index() >= graph.storage.len() =>
                    {
                        return Err(format!(
                            "task graph#{ordinal} storage#{storage_ordinal} clones absent storage#{}",
                            source.0
                        ));
                    }
                    LocalStorageOrigin::CallResult { choice, path } => {
                        let called = self.choices.get(choice.index()).ok_or_else(|| {
                            format!("task graph#{ordinal} storage#{storage_ordinal} names absent call choice#{}", choice.0)
                        })?;
                        let port = called.results.iter().find(|port| &port.path == path).ok_or_else(|| {
                            format!("task graph#{ordinal} storage#{storage_ordinal} names absent call result path {path:?}")
                        })?;
                        if port.ty != Type::Tensor(storage.ty.clone()) {
                            return Err(format!(
                                "task graph#{ordinal} storage#{storage_ordinal} changes call choice#{} result {path:?} type: destination {:?}, port {:?}",
                                choice.0,
                                Type::Tensor(storage.ty.clone()),
                                port.ty,
                            ));
                        }
                    }
                    _ => {}
                }
            }
            let remapped = graph.input_remap.iter().copied().collect::<BTreeSet<_>>();
            if remapped.len() != graph.inputs.len()
                || remapped
                    .iter()
                    .any(|port| *port as usize >= graph.inputs.len())
            {
                return Err(format!(
                    "task graph#{ordinal} input remapping is not a permutation"
                ));
            }
            let result_remap = graph.result_remap.iter().copied().collect::<BTreeSet<_>>();
            if graph.result_remap.len() != graph.results.len()
                || result_remap.len() != graph.results.len()
                || result_remap
                    .iter()
                    .any(|port| *port as usize >= graph.results.len())
            {
                return Err(format!(
                    "task graph#{ordinal} result remapping is not a permutation"
                ));
            }
            for (view_ordinal, view) in graph.views.iter().enumerate() {
                match &view.storage {
                    StorageRef::Local(storage) => {
                        let storage = graph.storage.get(storage.index()).ok_or_else(|| {
                            format!("task graph#{ordinal} view#{view_ordinal} names absent storage")
                        })?;
                        if storage.ty.elem != view.elem {
                            return Err(format!(
                                "task graph#{ordinal} view#{view_ordinal} changes element type"
                            ));
                        }
                    }
                    StorageRef::Input { port, .. } if *port as usize >= graph.inputs.len() => {
                        return Err(format!(
                            "task graph#{ordinal} view#{view_ordinal} names absent input storage"
                        ));
                    }
                    StorageRef::Result { port, .. } if *port as usize >= graph.results.len() => {
                        return Err(format!(
                            "task graph#{ordinal} view#{view_ordinal} names absent result storage"
                        ));
                    }
                    _ => {}
                }
            }
            for (symbol, value) in &graph.runtime_extents {
                if !symbol.starts_with("@runtime.") {
                    return Err(format!(
                        "task graph#{ordinal} runtime extent binding `{symbol}` is not occurrence-qualified"
                    ));
                }
                verify_scalar_expr(self, graph, value)?;
            }
            verify_graph(self, graph)?;
        }
        Ok(())
    }

    fn verify_extents(&self) -> Result<(), String> {
        let mut sites = BTreeSet::new();
        for extent in &self.extents {
            match extent {
                LogicalExtent::Static(value) if *value < 0 => {
                    return Err("logical extent is negative".into());
                }
                LogicalExtent::Dynamic(value) => {
                    self.value(*value)?;
                }
                LogicalExtent::Symbolic(sym) if sym.as_constant().is_some() => {
                    return Err("constant logical extent is misclassified as symbolic".into());
                }
                LogicalExtent::Structural {
                    choice,
                    alternative,
                    site,
                    upper_bound,
                } => {
                    let owner = self
                        .choices
                        .get(choice.index())
                        .ok_or_else(|| format!("extent site#{site} names absent choice"))?;
                    if *alternative as usize >= owner.alternatives.len() {
                        return Err(format!("extent site#{site} names absent alternative"));
                    }
                    if *upper_bound < 0 {
                        return Err(format!("extent site#{site} has negative upper bound"));
                    }
                    if !sites.insert((*choice, *alternative, *site)) {
                        return Err(format!("extent site#{site} occurs twice"));
                    }
                }
                _ => {}
            }
        }
        let verify_alternative = |reference: LogicalAlternativeRef| -> Result<(), String> {
            let choice = self.choices.get(reference.choice.index()).ok_or_else(|| {
                format!(
                    "structural constraint names absent choice#{}",
                    reference.choice.0
                )
            })?;
            if reference.alternative as usize >= choice.alternatives.len() {
                return Err(format!(
                    "structural constraint names absent choice#{} alternative#{}",
                    reference.choice.0, reference.alternative
                ));
            }
            Ok(())
        };
        let verify_site = |reference: StructuralSiteRef| -> Result<(), String> {
            verify_alternative(LogicalAlternativeRef {
                choice: reference.choice,
                alternative: reference.alternative,
            })?;
            if !sites.contains(&(reference.choice, reference.alternative, reference.site)) {
                return Err(format!(
                    "structural relation names absent site#{} on choice#{} alternative#{}",
                    reference.site, reference.choice.0, reference.alternative
                ));
            }
            Ok(())
        };
        for constraint in &self.structural_constraints {
            verify_alternative(constraint.active_if)?;
            verify_site(constraint.site)?;
            match constraint.kind {
                StructuralConstraintKind::Multiple(unit) if unit <= 0 => {
                    return Err("structural multiple must be positive".into());
                }
                StructuralConstraintKind::Divides(extent) if extent < 0 => {
                    return Err("structural dividend must be non-negative".into());
                }
                _ => {}
            }
        }
        for refinement in &self.structural_refinements {
            verify_site(refinement.refinement)?;
            verify_site(refinement.refined)?;
            if refinement.refinement == refinement.refined {
                return Err("structural site cannot refine itself".into());
            }
        }
        Ok(())
    }
}

fn verify_graph(program: &LogicalProgram, graph: &LogicalTaskGraph) -> Result<(), String> {
    let mut producers = BTreeMap::<OperandId, LogicalEndpoint>::new();
    for (ordinal, operand) in graph.operands.iter().enumerate() {
        if operand.id.index() != ordinal {
            return Err(format!(
                "task graph#{} operand#{} is stored at position {ordinal}",
                graph.id.0, operand.id.0
            ));
        }
        match operand.value {
            ValueRef::Input(port) => {
                let declared = graph
                    .inputs
                    .get(port as usize)
                    .ok_or_else(|| format!("operand#{} names absent input#{port}", operand.id.0))?;
                if declared.ty != operand.ty {
                    return Err(format!(
                        "operand#{} changes input#{port} type",
                        operand.id.0
                    ));
                }
            }
            ValueRef::Result(port) => {
                let declared = graph.results.get(port as usize).ok_or_else(|| {
                    format!("operand#{} names absent result#{port}", operand.id.0)
                })?;
                if declared.ty != operand.ty {
                    return Err(format!(
                        "operand#{} changes result#{port} type",
                        operand.id.0
                    ));
                }
            }
            ValueRef::Local(value) => verify_local_value(graph, value)?,
        }
        if let Some(storage) = &operand.storage {
            verify_storage_ref(graph, storage)?;
        }
    }
    for (ordinal, task) in graph.tasks.iter().enumerate() {
        if task.id.index() != ordinal {
            return Err(format!(
                "task graph#{} task#{} is stored at position {ordinal}",
                graph.id.0, task.id.0
            ));
        }
        for operand in task.inputs.iter().chain(&task.outputs) {
            graph
                .operand(*operand)
                .ok_or_else(|| format!("task#{} names absent operand#{}", task.id.0, operand.0))?;
        }
        for output in &task.outputs {
            if producers
                .insert(*output, LogicalEndpoint::Task(task.id))
                .is_some()
            {
                return Err(format!("operand#{} has more than one producer", output.0));
            }
        }
        verify_domain(program, graph, &task.domain)?;
        for access in &task.effects.accesses {
            verify_storage_ref(graph, &access.storage)?;
            if let Some(view) = access.view {
                graph
                    .views
                    .get(view.index())
                    .ok_or_else(|| format!("task#{} names absent view#{}", task.id.0, view.0))?;
            }
        }
        verify_scalar_block(program, graph, &task.body.operations)?;
    }
    for (ordinal, call) in graph.calls.iter().enumerate() {
        if call.id.index() != ordinal {
            return Err(format!(
                "task graph#{} call#{} is stored at position {ordinal}",
                graph.id.0, call.id.0
            ));
        }
        let called = program
            .choices
            .get(call.choice.index())
            .ok_or_else(|| format!("call#{} names absent choice#{}", call.id.0, call.choice.0))?;
        if call.inputs.len() != called.inputs.len() || call.outputs.len() != called.results.len() {
            return Err(format!(
                "call#{} does not match choice#{} port counts",
                call.id.0, call.choice.0
            ));
        }
        for (operand, port) in call.inputs.iter().zip(&called.inputs) {
            if graph.operand(*operand).map(|operand| &operand.ty) != Some(&port.ty) {
                return Err(format!(
                    "call#{} input operand#{} has the wrong type",
                    call.id.0, operand.0
                ));
            }
        }
        for (operand, port) in call.outputs.iter().zip(&called.results) {
            if graph.operand(*operand).map(|operand| &operand.ty) != Some(&port.ty) {
                return Err(format!(
                    "call#{} output operand#{} has the wrong type",
                    call.id.0, operand.0
                ));
            }
        }
        let expected_tensor_results = called
            .results
            .iter()
            .filter(|port| matches!(port.ty, Type::Tensor(_)))
            .map(|port| port.path.clone())
            .collect::<BTreeSet<_>>();
        let actual_tensor_results = call
            .results
            .iter()
            .map(|result| result.path.clone())
            .collect::<BTreeSet<_>>();
        if expected_tensor_results != actual_tensor_results
            || actual_tensor_results.len() != call.results.len()
        {
            return Err(format!(
                "call#{} does not provide every tensor result destination exactly once",
                call.id.0
            ));
        }
        for result in &call.results {
            let storage = graph
                .storage
                .get(result.storage.index())
                .ok_or_else(|| format!("call#{} names absent result storage", call.id.0))?;
            if storage.origin
                != (LocalStorageOrigin::CallResult {
                    choice: call.choice,
                    path: result.path.clone(),
                })
            {
                return Err(format!(
                    "call#{} result storage has the wrong origin",
                    call.id.0
                ));
            }
        }
        let result = graph.operand(call.result).ok_or_else(|| {
            format!(
                "call#{} names absent whole-result operand#{}",
                call.id.0, call.result.0
            )
        })?;
        let expected_result = called
            .interface
            .results
            .first()
            .cloned()
            .unwrap_or(Type::Void);
        if result.ty != expected_result {
            return Err(format!(
                "call#{} whole-result operand has the wrong type",
                call.id.0
            ));
        }
        verify_domain(program, graph, &call.domain)?;
        for operand in call
            .inputs
            .iter()
            .chain(&call.outputs)
            .chain(std::iter::once(&call.result))
        {
            graph
                .operand(*operand)
                .ok_or_else(|| format!("call#{} names absent operand#{}", call.id.0, operand.0))?;
        }
        for output in call.outputs.iter().chain(std::iter::once(&call.result)) {
            if let Some(previous) = producers.insert(*output, LogicalEndpoint::Call(call.id)) {
                if previous != LogicalEndpoint::Call(call.id) {
                    return Err(format!("operand#{} has more than one producer", output.0));
                }
            }
        }
        for access in &call.effects.accesses {
            verify_storage_ref(graph, &access.storage)?;
        }
    }
    let mut unique = BTreeSet::new();
    for (ordinal, dependency) in graph.dependencies.iter().enumerate() {
        if dependency.id.index() != ordinal {
            return Err(format!(
                "task graph#{} dependency#{} is stored at position {ordinal}",
                graph.id.0, dependency.id.0
            ));
        }
        verify_endpoint(graph, dependency.from)?;
        verify_endpoint(graph, dependency.to)?;
        if !unique.insert((dependency.from, dependency.to, dependency.kind.clone())) {
            return Err(format!("task graph#{} repeats a dependency", graph.id.0));
        }
        match &dependency.kind {
            LogicalDependencyKind::Value(operand) => {
                graph.operand(*operand).ok_or_else(|| {
                    format!(
                        "dependency#{} names absent operand#{}",
                        dependency.id.0, operand.0
                    )
                })?;
            }
            LogicalDependencyKind::Effect(storage) | LogicalDependencyKind::Ownership(storage) => {
                verify_storage_ref(graph, storage)?
            }
            LogicalDependencyKind::Control => {}
        }
    }
    verify_dependency_dag(graph)?;
    for port in 0..graph.results.len() as u32 {
        let producers = graph
            .dependencies
            .iter()
            .filter(|edge| {
                edge.to == LogicalEndpoint::Output(port)
                    && matches!(
                        edge.from,
                        LogicalEndpoint::Task(_) | LogicalEndpoint::Call(_)
                    )
                    && matches!(edge.kind, LogicalDependencyKind::Value(_))
            })
            .count();
        if producers != 1 {
            return Err(format!(
                "task graph#{} output#{port} has {producers} producers",
                graph.id.0
            ));
        }
    }
    for task in &graph.tasks {
        for input in &task.inputs {
            let producer = producers.get(input).ok_or_else(|| {
                format!(
                    "task#{} input operand#{} has no producer",
                    task.id.0, input.0
                )
            })?;
            if !graph.dependencies.iter().any(|edge| {
                edge.from == *producer
                    && edge.to == LogicalEndpoint::Task(task.id)
                    && edge.kind == LogicalDependencyKind::Value(*input)
            }) {
                return Err(format!(
                    "task#{} input operand#{} has no value dependency",
                    task.id.0, input.0
                ));
            }
        }
    }
    for call in &graph.calls {
        for input in &call.inputs {
            let producer = producers.get(input).ok_or_else(|| {
                format!(
                    "call#{} input operand#{} has no producer",
                    call.id.0, input.0
                )
            })?;
            if !graph.dependencies.iter().any(|edge| {
                edge.from == *producer
                    && edge.to == LogicalEndpoint::Call(call.id)
                    && edge.kind == LogicalDependencyKind::Value(*input)
            }) {
                return Err(format!(
                    "call#{} input operand#{} has no value dependency",
                    call.id.0, input.0
                ));
            }
        }
    }
    Ok(())
}

fn verify_domain(
    program: &LogicalProgram,
    graph: &LogicalTaskGraph,
    domain: &LogicalDomain,
) -> Result<(), String> {
    for axis in &domain.axes {
        if axis.binders.is_empty() {
            return Err("logical domain has an axis without a binder".into());
        }
        for binder in &axis.binders {
            verify_local_value(graph, *binder)?;
        }
        match &axis.source {
            LogicalAxisSource::Range { lo, hi, runtime } => {
                verify_scalar_expr(program, graph, lo)?;
                verify_scalar_expr(program, graph, hi)?;
                if let Some(runtime) = runtime {
                    verify_scalar_expr(program, graph, runtime)?;
                }
            }
            LogicalAxisSource::Coordinates { value, .. } => {
                verify_scalar_expr(program, graph, value)?
            }
            LogicalAxisSource::Members { .. } => {}
        }
    }
    for region in &domain.regions {
        for binder in &region.binders {
            verify_local_value(graph, *binder)?;
        }
        if let Some(source) = &region.source {
            verify_scalar_expr(program, graph, source)?;
        }
        if let Some(merge) = &region.merge {
            verify_pattern(graph, &merge.left)?;
            verify_pattern(graph, &merge.right)?;
            verify_scalar_expr(program, graph, &merge.identity)?;
        }
    }
    for predicate in &domain.predicates {
        verify_scalar_expr(program, graph, &predicate.condition)?;
    }
    Ok(())
}

fn verify_dependency_dag(graph: &LogicalTaskGraph) -> Result<(), String> {
    let mut indegree = BTreeMap::new();
    let mut successors = BTreeMap::<LogicalEndpoint, Vec<LogicalEndpoint>>::new();
    for task in &graph.tasks {
        indegree.insert(LogicalEndpoint::Task(task.id), 0usize);
    }
    for call in &graph.calls {
        indegree.insert(LogicalEndpoint::Call(call.id), 0usize);
    }
    for edge in &graph.dependencies {
        if indegree.contains_key(&edge.from) && indegree.contains_key(&edge.to) {
            *indegree
                .get_mut(&edge.to)
                .expect("internal endpoint was checked") += 1;
            successors.entry(edge.from).or_default().push(edge.to);
        }
    }
    let mut ready = indegree
        .iter()
        .filter_map(|(node, degree)| (*degree == 0).then_some(*node))
        .collect::<Vec<_>>();
    let mut visited = 0usize;
    while let Some(node) = ready.pop() {
        visited += 1;
        for successor in successors.get(&node).into_iter().flatten() {
            let degree = indegree.get_mut(successor).expect("successor is internal");
            *degree -= 1;
            if *degree == 0 {
                ready.push(*successor);
            }
        }
    }
    if visited != indegree.len() {
        return Err(format!(
            "task graph#{} dependency graph is cyclic",
            graph.id.0
        ));
    }
    Ok(())
}

fn verify_endpoint(graph: &LogicalTaskGraph, endpoint: LogicalEndpoint) -> Result<(), String> {
    match endpoint {
        LogicalEndpoint::Input(port) if port as usize >= graph.inputs.len() => {
            Err(format!("absent graph input#{port}"))
        }
        LogicalEndpoint::Output(port) if port as usize >= graph.results.len() => {
            Err(format!("absent graph output#{port}"))
        }
        LogicalEndpoint::Task(id) if graph.task(id).is_none() => {
            Err(format!("absent task#{}", id.0))
        }
        LogicalEndpoint::Call(id) if graph.call(id).is_none() => {
            Err(format!("absent call#{}", id.0))
        }
        _ => Ok(()),
    }
}

fn verify_scalar_block(
    program: &LogicalProgram,
    fragment: &LogicalTaskGraph,
    block: &LogicalBlock,
) -> Result<(), String> {
    for operation in block {
        if matches!(
            operation.kind,
            LogicalOperationKind::Region(_)
                | LogicalOperationKind::Stages(_)
                | LogicalOperationKind::For { .. }
                | LogicalOperationKind::Coordinates { .. }
                | LogicalOperationKind::Members { .. }
        ) {
            return Err(format!(
                "task graph#{} scalar body contains a scheduling construct",
                fragment.id.0
            ));
        }
        match &operation.kind {
            LogicalOperationKind::Bind { pattern, value } => {
                verify_pattern(fragment, pattern)?;
                verify_scalar_expr(program, fragment, value)?;
            }
            LogicalOperationKind::Assign { target, value, .. } => {
                verify_scalar_expr(program, fragment, target)?;
                verify_scalar_expr(program, fragment, value)?;
            }
            LogicalOperationKind::Region(region) => verify_region(program, fragment, region)?,
            LogicalOperationKind::Stages(stages) => {
                for stage in stages {
                    for port in &stage.ports {
                        verify_local_value(fragment, *port)?;
                    }
                    verify_scalar_block(program, fragment, &stage.body)?;
                }
            }
            LogicalOperationKind::For {
                binder,
                lo,
                hi,
                source,
                body,
                ..
            } => {
                verify_local_value(fragment, *binder)?;
                verify_scalar_expr(program, fragment, lo)?;
                verify_scalar_expr(program, fragment, hi)?;
                if let Some(source) = source {
                    verify_scalar_expr(program, fragment, source)?;
                }
                verify_scalar_block(program, fragment, body)?;
            }
            LogicalOperationKind::Coordinates {
                binders,
                value,
                body,
                ..
            } => {
                for binder in binders {
                    verify_local_value(fragment, *binder)?;
                }
                verify_scalar_expr(program, fragment, value)?;
                verify_scalar_block(program, fragment, body)?;
            }
            LogicalOperationKind::Members { binder, body, .. } => {
                verify_local_value(fragment, *binder)?;
                verify_scalar_block(program, fragment, body)?;
            }
            LogicalOperationKind::If {
                condition,
                then,
                els,
            } => {
                verify_scalar_expr(program, fragment, condition)?;
                verify_scalar_block(program, fragment, then)?;
                verify_scalar_block(program, fragment, els)?;
            }
            LogicalOperationKind::Publish { value, destination } => {
                verify_scalar_expr(program, fragment, value)?;
                verify_scalar_expr(program, fragment, destination)?;
            }
            LogicalOperationKind::Yield(values) => {
                for value in values {
                    verify_scalar_expr(program, fragment, value)?;
                }
            }
            LogicalOperationKind::Return(writes) => {
                let ports = writes
                    .iter()
                    .map(|write| write.port)
                    .collect::<BTreeSet<_>>();
                if ports.len() != writes.len() || writes.len() != fragment.results.len() {
                    return Err(format!(
                        "fragment#{} return does not initialize every result port exactly once",
                        fragment.id.0
                    ));
                }
                for write in writes {
                    let port = fragment.results.get(write.port as usize).ok_or_else(|| {
                        format!("fragment#{} return names absent result port", fragment.id.0)
                    })?;
                    if port.path != write.path || port.ty != write.value.ty {
                        return Err(format!(
                            "fragment#{} return disagrees with result port#{}",
                            fragment.id.0, write.port
                        ));
                    }
                    if write.transfer != matches!(port.ty, Type::Tensor(_)) {
                        return Err(format!(
                            "fragment#{} return has incorrect ownership transfer",
                            fragment.id.0
                        ));
                    }
                    verify_scalar_expr(program, fragment, &write.value)?;
                }
            }
            LogicalOperationKind::Reduction { binder, value, .. } => {
                verify_local_value(fragment, *binder)?;
                verify_scalar_expr(program, fragment, value)?;
            }
            LogicalOperationKind::ConditionalMerge { binder, cases } => {
                verify_local_value(fragment, *binder)?;
                if cases.is_empty() {
                    return Err("conditional merge has no cases".into());
                }
                for case in cases {
                    fragment.operand(case.value).ok_or_else(|| {
                        format!("conditional merge names absent operand#{}", case.value.0)
                    })?;
                    for predicate in &case.predicates {
                        verify_scalar_expr(program, fragment, &predicate.condition)?;
                    }
                }
            }
            LogicalOperationKind::Expr(expr) => verify_scalar_expr(program, fragment, expr)?,
        }
    }
    Ok(())
}

fn verify_region(
    program: &LogicalProgram,
    fragment: &LogicalTaskGraph,
    region: &LogicalRegion,
) -> Result<(), String> {
    for binder in &region.binders {
        verify_local_value(fragment, *binder)?;
    }
    if let Some(source) = &region.source {
        verify_scalar_expr(program, fragment, source)?;
    }
    verify_scalar_block(program, fragment, &region.body)?;
    if let Some(merge) = &region.merge {
        verify_pattern(fragment, &merge.left)?;
        verify_pattern(fragment, &merge.right)?;
        verify_scalar_expr(program, fragment, &merge.identity)?;
        verify_scalar_block(program, fragment, &merge.body)?;
    }
    Ok(())
}
fn verify_pattern(fragment: &LogicalTaskGraph, pattern: &LogicalPattern) -> Result<(), String> {
    match pattern {
        LogicalPattern::Value(id) => verify_local_value(fragment, *id),
        LogicalPattern::Tuple(items) => {
            for item in items {
                verify_pattern(fragment, item)?;
            }
            Ok(())
        }
    }
}
fn verify_local_value(fragment: &LogicalTaskGraph, id: LocalValueId) -> Result<(), String> {
    fragment.values.get(id.index()).map(|_| ()).ok_or_else(|| {
        format!(
            "fragment#{} names absent local value#{}",
            fragment.id.0, id.0
        )
    })
}

fn verify_storage_ref(fragment: &LogicalTaskGraph, storage: &StorageRef) -> Result<(), String> {
    match storage {
        StorageRef::Input { port, path } => verify_storage_path(
            fragment.inputs.get(*port as usize),
            path,
            &format!("input storage port#{port}"),
        ),
        StorageRef::Result { port, path } => verify_storage_path(
            fragment.results.get(*port as usize),
            path,
            &format!("result storage port#{port}"),
        ),
        StorageRef::Local(id) if id.index() >= fragment.storage.len() => {
            Err(format!("absent local storage#{}", id.0))
        }
        _ => Ok(()),
    }
}

fn verify_storage_path(port: Option<&Port>, path: &[u32], label: &str) -> Result<(), String> {
    let mut ty = &port.ok_or_else(|| format!("absent {label}"))?.ty;
    for index in path {
        let Type::Tuple(items) = ty else {
            return Err(format!("{label} path descends through a non-tuple"));
        };
        ty = items
            .get(*index as usize)
            .ok_or_else(|| format!("{label} path names an absent tuple field"))?;
    }
    if !matches!(ty, Type::Tensor(_)) {
        return Err(format!("{label} does not identify tensor storage"));
    }
    Ok(())
}

fn verify_scalar_expr(
    program: &LogicalProgram,
    fragment: &LogicalTaskGraph,
    expr: &LogicalExpr,
) -> Result<(), String> {
    if matches!(
        expr.kind,
        LogicalExprKind::Call { .. } | LogicalExprKind::Reduce { .. } | LogicalExprKind::Region(_)
    ) {
        return Err(format!(
            "task graph#{} scalar expression contains a call, reduction, or region",
            fragment.id.0
        ));
    }
    let nested: Vec<&LogicalExpr> = match &expr.kind {
        LogicalExprKind::Value(ValueRef::Input(port)) => {
            if *port as usize >= fragment.inputs.len() {
                return Err(format!(
                    "fragment#{} names absent input port#{}",
                    fragment.id.0, port
                ));
            }
            vec![]
        }
        LogicalExprKind::Value(ValueRef::Local(value)) => {
            verify_local_value(fragment, *value)?;
            vec![]
        }
        LogicalExprKind::Value(ValueRef::Result(port)) => {
            if *port as usize >= fragment.results.len() {
                return Err(format!(
                    "fragment#{} names absent result port#{}",
                    fragment.id.0, port
                ));
            }
            vec![]
        }
        LogicalExprKind::Tuple(items)
        | LogicalExprKind::Math { args: items, .. }
        | LogicalExprKind::Intrinsic { args: items, .. } => items.iter().collect(),
        LogicalExprKind::Range(lo, hi)
        | LogicalExprKind::Binary {
            lhs: lo, rhs: hi, ..
        } => vec![lo, hi],
        LogicalExprKind::Field(base, _)
        | LogicalExprKind::Cast { expr: base, .. }
        | LogicalExprKind::Unary { expr: base, .. }
        | LogicalExprKind::Reduce { value: base, .. }
        | LogicalExprKind::Extent { base, .. }
        | LogicalExprKind::Accessor { base, .. }
        | LogicalExprKind::Geometry { base, .. } => vec![base],
        LogicalExprKind::View { view, base } => {
            fragment
                .views
                .get(view.index())
                .ok_or_else(|| format!("fragment#{} names absent local view", fragment.id.0))?;
            vec![base]
        }
        LogicalExprKind::Filled { storage, like, .. } => {
            fragment
                .storage
                .get(storage.index())
                .ok_or_else(|| format!("fragment#{} names absent local storage", fragment.id.0))?;
            vec![like]
        }
        LogicalExprKind::Construct { storage } => {
            fragment
                .storage
                .get(storage.index())
                .ok_or_else(|| format!("fragment#{} names absent local storage", fragment.id.0))?;
            vec![]
        }
        LogicalExprKind::Index { base, indices } => {
            for index in indices {
                match index {
                    LogicalIndex::Point(value) => verify_scalar_expr(program, fragment, value)?,
                    LogicalIndex::Coordinate(value) => verify_local_value(fragment, *value)?,
                    LogicalIndex::Range { start, end } => {
                        for value in start.iter().chain(end.iter()) {
                            verify_scalar_expr(program, fragment, value)?;
                        }
                    }
                    LogicalIndex::Slice(_) => {}
                }
            }
            vec![base]
        }
        LogicalExprKind::Snapshot { storage, source }
        | LogicalExprKind::Decode { storage, source }
        | LogicalExprKind::Materialize {
            storage,
            value: source,
        } => {
            fragment
                .storage
                .get(storage.index())
                .ok_or_else(|| format!("fragment#{} names absent local storage", fragment.id.0))?;
            vec![source]
        }
        LogicalExprKind::Select { cond, then, els } => vec![cond, then, els],
        LogicalExprKind::Coordinate(value) => {
            verify_local_value(fragment, *value)?;
            vec![]
        }
        LogicalExprKind::Call {
            choice,
            args,
            results,
        } => {
            let called = program.choices.get(choice.index()).ok_or_else(|| {
                format!(
                    "fragment#{} calls absent choice#{}",
                    fragment.id.0, choice.0
                )
            })?;
            let expected = called
                .results
                .iter()
                .filter(|port| matches!(port.ty, Type::Tensor(_)))
                .map(|port| port.path.clone())
                .collect::<BTreeSet<_>>();
            let actual = results
                .iter()
                .map(|result| result.path.clone())
                .collect::<BTreeSet<_>>();
            if expected != actual || actual.len() != results.len() {
                return Err(format!(
                    "fragment#{} call choice#{} does not name every tensor result destination exactly once",
                    fragment.id.0, choice.0
                ));
            }
            for result in results {
                let storage = fragment
                    .storage
                    .get(result.storage.index())
                    .ok_or_else(|| {
                        format!(
                            "fragment#{} call result names absent local storage",
                            fragment.id.0
                        )
                    })?;
                if storage.origin
                    != (LocalStorageOrigin::CallResult {
                        choice: *choice,
                        path: result.path.clone(),
                    })
                {
                    return Err(format!(
                        "fragment#{} call result destination origin disagrees",
                        fragment.id.0
                    ));
                }
            }
            args.iter().collect()
        }
        LogicalExprKind::Region(region) => {
            verify_region(program, fragment, region)?;
            vec![]
        }
        LogicalExprKind::Atomic { place, value, .. } => vec![place, value],
        LogicalExprKind::Shape(value) => {
            for atom in value.atoms() {
                if let Atom::Param(symbol) = atom {
                    if symbol.starts_with("@runtime.")
                        && !fragment.runtime_extents.contains_key(&symbol)
                    {
                        return Err(format!(
                            "fragment#{} (choice#{}, alternative#{}) runtime Shape atom `{symbol}` has no producing expression; available runtime extents: {:?}; inputs: {:?}",
                            fragment.id.0,
                            fragment.choice.0,
                            fragment.alternative,
                            fragment.runtime_extents.keys().collect::<Vec<_>>(),
                            fragment
                                .inputs
                                .iter()
                                .map(|port| &port.ty)
                                .collect::<Vec<_>>()
                        ));
                    }
                }
            }
            vec![]
        }
        LogicalExprKind::Int(_) | LogicalExprKind::Float(_) | LogicalExprKind::Bool(_) => vec![],
    };
    for child in nested {
        verify_scalar_expr(program, fragment, child)?;
    }
    Ok(())
}

/// Specialize one checked entry into the complete choice-bearing logical
/// program consumed by physical planning. No witness is accepted or created.
pub fn specialize_entry_contract(
    program: &Program,
    entry: &str,
    environment: &TargetEnvironment<'_>,
    workload: &Workload,
) -> Result<LogicalProgram, String> {
    let family = family::construct(program, entry, environment, workload)?;
    specialize_entry_contract_from_family(program, &family)
}

/// Build the complete logical program using `Family` only as temporary
/// applicability and occurrence-topology input. The result contains no witness
/// and retains no reference to the family.
pub fn specialize_entry_contract_from_family(
    program: &Program,
    family: &family::Family,
) -> Result<LogicalProgram, String> {
    let entry = family.entry.as_str();
    let root = family
        .occurrences
        .first()
        .ok_or_else(|| format!("family of `{entry}` has no entry occurrence"))?;
    if root.candidates.is_empty() {
        return Err(format!(
            "family of `{entry}` has no applicable implementation"
        ));
    }
    let contract = program.resolve_family(entry)?;
    let contract_definition = program.definition(contract.contract);

    let mut logical = LogicalProgram {
        identity: LogicalCompilationIdentity {
            semantic_program: program.identity(),
            workload: stable_identity(&family.workload),
        },
        entry: entry.to_owned(),
        target: family.target.clone(),
        capability_fingerprint: family.capability_fingerprint.clone(),
        shapes: family.workload.shapes.clone(),
        elems: family.workload.elems.clone(),
        storage: Vec::new(),
        views: Vec::new(),
        values: Vec::new(),
        result_slots: Vec::new(),
        choices: Vec::new(),
        entry_choice: ChoiceId(0),
        extents: Vec::new(),
        structural_constraints: Vec::new(),
        structural_refinements: Vec::new(),
        extent_bounds: BTreeMap::new(),
        task_graphs: Vec::new(),
    };

    let mut effects = Vec::new();
    for (ordinal, param) in contract_definition.params.iter().enumerate() {
        let ty = specialize_type(&param.ty, &family.workload.shapes, &family.workload.elems)?;
        add_parameter_value(
            &ty,
            ordinal as u32,
            &param.name,
            param.mode,
            &mut Vec::new(),
            &mut logical,
            &mut effects,
        )?;
    }

    let result = specialize_result(
        &contract_definition.result,
        &family.workload.shapes,
        &family.workload.elems,
    )?;
    add_result_slots(&result, &mut Vec::new(), &mut logical, &mut effects)?;
    body_specialize::populate(program, family, &mut logical)?;
    logical.verify()?;
    Ok(logical)
}

fn add_parameter_value(
    ty: &Type,
    ordinal: u32,
    name: &str,
    mode: Mode,
    path: &mut Vec<u32>,
    logical: &mut LogicalProgram,
    effects: &mut Vec<Effect>,
) -> Result<ValueId, String> {
    let kind = match ty {
        Type::Tensor(tensor) => {
            let storage = StorageId(logical.storage.len() as u32);
            logical.storage.push(Storage {
                ty: tensor.clone(),
                origin: StorageOrigin::Parameter {
                    ordinal,
                    path: path.clone(),
                    name: name.to_owned(),
                },
            });
            let view = ViewId(logical.views.len() as u32);
            logical.views.push(View {
                storage,
                shape: tensor.shape.clone(),
                elem: tensor.elem.clone(),
                access: if mode == Mode::In {
                    Access::Shared
                } else {
                    Access::Exclusive
                },
                transform: ViewTransform::Identity,
            });
            match mode {
                Mode::In => effects.push(Effect::Read(storage)),
                Mode::Out => effects.push(Effect::Write(storage)),
                Mode::Inout => {
                    effects.push(Effect::Read(storage));
                    effects.push(Effect::Write(storage));
                }
            }
            ValueKind::Tensor(view)
        }
        Type::Scalar(_) | Type::Index { .. } => ValueKind::ScalarParameter {
            ordinal,
            name: name.to_owned(),
        },
        Type::Range { .. } => ValueKind::RangeParameter {
            ordinal,
            name: name.to_owned(),
        },
        Type::Tuple(items) => {
            let mut members = Vec::with_capacity(items.len());
            for (member, item) in items.iter().enumerate() {
                path.push(member as u32);
                members.push(add_parameter_value(
                    item, ordinal, name, mode, path, logical, effects,
                )?);
                path.pop();
            }
            ValueKind::Tuple(members)
        }
        Type::CapabilityValue { .. } | Type::Void => {
            return Err(format!(
                "entry parameter `{name}` is not invocation ABI data"
            ));
        }
    };
    let id = ValueId(logical.values.len() as u32);
    logical.values.push(Value {
        ty: ty.clone(),
        kind,
    });
    Ok(id)
}

fn specialize_sym(sym: &Sym, shapes: &BTreeMap<String, i64>) -> Result<Sym, String> {
    sym.eval(&|name| shapes.get(name).copied())
        .map(Sym::constant)
        .ok_or_else(|| format!("logical specialization left an unbound extent `{sym}`"))
}

fn specialize_tensor(
    shaped: &Shaped,
    shapes: &BTreeMap<String, i64>,
    elems: &BTreeMap<String, Elem>,
) -> Result<TensorType, String> {
    let shape = shaped
        .axes
        .iter()
        .map(|axis| match axis {
            Extent::Semantic(sym) => specialize_sym(sym, shapes),
            Extent::Structural(slice) => Err(format!(
                "entry contract contains unresolved structural extent slice#{}",
                slice.0
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let elem = match &shaped.elem {
        Elem::Param(name) => elems
            .get(name)
            .cloned()
            .ok_or_else(|| format!("logical specialization left element `{name}` unbound"))?,
        elem => elem.clone(),
    };
    Ok(TensorType { shape, elem })
}

fn specialize_type(
    ty: &Ty,
    shapes: &BTreeMap<String, i64>,
    elems: &BTreeMap<String, Elem>,
) -> Result<Type, String> {
    Ok(match ty {
        Ty::Scalar(dtype) => Type::Scalar(*dtype),
        Ty::Index(bound) => Type::Index {
            bound: specialize_sym(bound, shapes)?,
        },
        Ty::Range(bound) => Type::Range {
            bound: specialize_sym(bound, shapes)?,
        },
        Ty::Tensor(shaped) | Ty::View(shaped) | Ty::Tile(shaped) => {
            Type::Tensor(specialize_tensor(shaped, shapes, elems)?)
        }
        Ty::Coord(slice) | Ty::Slice(slice) => Type::Index {
            bound: Sym::param(&format!("@slice{}", slice.0)),
        },
        Ty::Native(native) => Type::CapabilityValue {
            target: native.target.clone(),
            name: native.name.clone(),
            shape: native.shape.clone(),
            elem: native.elem.clone(),
        },
        Ty::Tuple(items) => Type::Tuple(
            items
                .iter()
                .map(|item| specialize_type(item, shapes, elems))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Ty::Void => Type::Void,
        other => {
            return Err(format!(
                "entry contract type `{other}` is not invocation ABI data"
            ));
        }
    })
}

fn specialize_result(
    ty: &Ty,
    shapes: &BTreeMap<String, i64>,
    elems: &BTreeMap<String, Elem>,
) -> Result<Type, String> {
    match ty {
        Ty::Scalar(_) | Ty::Index(_) | Ty::Tensor(_) | Ty::Tuple(_) | Ty::Void => {
            specialize_type(ty, shapes, elems)
        }
        other => Err(format!(
            "entry result `{other}` is not a scalar, owned tensor, or tuple of those"
        )),
    }
}

fn add_result_slots(
    ty: &Type,
    path: &mut Vec<u32>,
    logical: &mut LogicalProgram,
    effects: &mut Vec<Effect>,
) -> Result<(), String> {
    match ty {
        Type::Tensor(tensor) => {
            let storage = StorageId(logical.storage.len() as u32);
            logical.storage.push(Storage {
                ty: tensor.clone(),
                origin: StorageOrigin::Result { path: path.clone() },
            });
            logical.result_slots.push(ResultSlot {
                id: ResultSlotId(logical.result_slots.len() as u32),
                path: path.clone(),
                storage,
                ty: tensor.clone(),
            });
            effects.push(Effect::Write(storage));
            effects.push(Effect::Move(storage));
        }
        Type::Tuple(items) => {
            for (ordinal, item) in items.iter().enumerate() {
                path.push(ordinal as u32);
                add_result_slots(item, path, logical, effects)?;
                path.pop();
            }
        }
        Type::Scalar(_) | Type::Index { .. } | Type::Void => {}
        Type::Range { .. } | Type::CapabilityValue { .. } => {
            return Err("a range or capability value cannot be returned".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor() -> TensorType {
        TensorType {
            shape: vec![Sym::constant(32), Sym::constant(128), Sym::constant(128)],
            elem: Elem::Dtype(DType::F32),
        }
    }

    fn result_program() -> LogicalProgram {
        let ty = tensor();
        let result_type = Type::Tensor(ty.clone());
        let port = Port {
            path: vec![],
            ty: result_type.clone(),
            access: Some(Access::Exclusive),
        };
        let interface = Interface {
            inputs: vec![],
            results: vec![result_type.clone()],
            effects: vec![Effect::Write(StorageId(0)), Effect::Move(StorageId(0))],
            port_effects: vec![
                PortEffect::Write {
                    port: 0,
                    path: vec![],
                },
                PortEffect::MoveResult {
                    port: 0,
                    path: vec![],
                },
            ],
        };
        let fragment = Fragment {
            id: FragmentId(0),
            choice: ChoiceId(0),
            alternative: 0,
            inputs: vec![],
            results: vec![port.clone()],
            input_remap: vec![],
            result_remap: vec![0],
            values: vec![],
            storage: vec![],
            views: vec![],
            runtime_extents: BTreeMap::new(),
            body: vec![LogicalOperation {
                kind: LogicalOperationKind::Return(vec![ResultWrite {
                    port: 0,
                    path: vec![],
                    value: LogicalExpr {
                        ty: result_type,
                        kind: LogicalExprKind::Value(ValueRef::Result(0)),
                        span: Span::default(),
                    },
                    transfer: true,
                }]),
                span: Span::default(),
            }],
        };
        LogicalProgram {
            identity: LogicalCompilationIdentity {
                semantic_program: [0; 32],
                workload: [0; 32],
            },
            entry: "recurrent".into(),
            target: "metal".into(),
            capability_fingerprint: "test".into(),
            shapes: BTreeMap::new(),
            elems: BTreeMap::new(),
            storage: vec![Storage {
                ty: ty.clone(),
                origin: StorageOrigin::Result { path: vec![] },
            }],
            views: vec![View {
                storage: StorageId(0),
                shape: ty.shape.clone(),
                elem: ty.elem.clone(),
                access: Access::Exclusive,
                transform: ViewTransform::Identity,
            }],
            values: vec![Value {
                ty: Type::Tensor(ty.clone()),
                kind: ValueKind::Tensor(ViewId(0)),
            }],
            result_slots: vec![ResultSlot {
                id: ResultSlotId(0),
                path: vec![],
                storage: StorageId(0),
                ty,
            }],
            choices: vec![Choice {
                occurrence: 0,
                interface: interface.clone(),
                inputs: vec![],
                results: vec![port],
                alternatives: vec![Alternative {
                    definition: 0,
                    interface,
                    capabilities: BTreeSet::new(),
                    numerical_effects: vec![],
                    fragment: FragmentId(0),
                }],
            }],
            entry_choice: ChoiceId(0),
            extents: vec![],
            structural_constraints: vec![],
            structural_refinements: vec![],
            extent_bounds: BTreeMap::new(),
            fragments: vec![fragment],
        }
    }

    #[test]
    fn owned_result_is_one_logical_storage_identity() {
        result_program().verify().unwrap();
    }

    #[test]
    fn extent_capacity_evaluates_affine_runtime_bounds() {
        let mut program = result_program();
        program
            .extent_bounds
            .insert("row#8".into(), Sym::constant(16_383));
        assert_eq!(
            program.extent_capacity(&Sym::constant(1).add(&Sym::param("row#8"))),
            Some(16_384)
        );
    }

    #[test]
    fn result_slot_cannot_point_at_a_late_local_copy() {
        let mut program = result_program();
        program.storage[0].origin = StorageOrigin::Owned;
        assert!(program.verify().unwrap_err().contains("wrong origin"));
    }

    #[test]
    fn reshape_preserves_storage_identity() {
        let mut program = result_program();
        program.views.push(View {
            storage: StorageId(0),
            shape: vec![Sym::constant(32 * 128), Sym::constant(128)],
            elem: Elem::Dtype(DType::F32),
            access: Access::Exclusive,
            transform: ViewTransform::Reshape {
                source_shape: tensor().shape,
            },
        });
        program.verify().unwrap();
        assert_eq!(program.views[0].storage, program.views[1].storage);
    }

    #[test]
    fn choice_alternatives_cannot_change_semantics() {
        let mut program = result_program();
        let interface = Interface {
            inputs: Vec::new(),
            results: vec![Type::Tensor(tensor())],
            effects: vec![Effect::Write(StorageId(0)), Effect::Move(StorageId(0))],
            port_effects: Vec::new(),
        };
        program.choices.push(Choice {
            occurrence: 1,
            interface: interface.clone(),
            inputs: Vec::new(),
            results: Vec::new(),
            alternatives: vec![Alternative {
                definition: 7,
                interface: Interface {
                    results: vec![Type::Void],
                    ..interface
                },
                capabilities: BTreeSet::new(),
                numerical_effects: Vec::new(),
                fragment: FragmentId(0),
            }],
        });
        assert!(program
            .verify()
            .unwrap_err()
            .contains("changes its logical interface"));
    }
}
