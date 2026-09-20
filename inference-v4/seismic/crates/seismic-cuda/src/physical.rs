//! CUDA executable dialect and universal strategy construction over the one
//! common family builder.
//!
//! * Sealed exhaustive `CudaOp` legalizes the complete portable matrix; a
//!   universal `Inapplicable` is a compiler bug detected by the family
//!   builder. PTX emission accepts every opcode.
//! * Arbitrary-rank independent domains use the common `LinearIterationMap`
//!   (rank > 3 delinearizes; logical rank is never native grid rank).
//! * Exact hard resources — explicit shared/local bytes, bindings, and
//!   geometry bounds — are declared in consequences *before* the solve;
//!   native register/spill/occupancy facts only instantiate the selected
//!   bounded `NativeResourceContract` geometry after encoding.
//! * Two physical alternatives per logical alternative: the universal
//!   column (#0: serialized exact atomics, the versioned `seismic_math`
//!   software sequence, parallel-outer reductions) and the optimized column
//!   (#1: the planned CAS narrow-floating atomic add wherever the layout
//!   admits it, and the PTX approximate `ex2.approx` sequence for
//!   `exp_fast` carrying an `Approximate` transfer bound/evidence-gated by
//!   the numerical policy).
//! * Kernel-local control: an independent loop whose body (transitively)
//!   contains only primitives, reductions, and ordered loops is consumed by
//!   `schedule_loop` with a single-visit executor range (endpoint slots
//!   named `absorbed:{node}:start` / `absorbed:{node}:end`, which the
//!   encoder resolves to the constants 0 and 1), its independent axes
//!   becoming the launch's linear iteration map and its ordered axes
//!   becoming in-kernel `SerialFor` loops. The retained executor `Repeat`
//!   honestly describes the executor's visits; the loop's logical range is
//!   carried by the iteration map and the `SerialFor` extents.

use seismic_compiler::terminal::{
    discharge_with_builder, form_primitive, universal_node, CheckPredicate, FormationError,
    GraphFacts, ObligationDischarge,
};
use seismic_lang::{
    intrinsics::{primitive, ErrorBound, IntrinsicId, MathOp, PlaneField, PrimitiveId, ReduceOp},
    logical::{
        self, Access, GraphRegion, GraphValueId, JoinSlot, LogicalNode, LogicalNodeKind,
        LogicalProgram, LogicalStorageId, LoopNode, NodeId, RegionParameter, RegionResult,
        RuntimeExtent, TaskGraph, ViewTransform,
    },
    repr,
    sir::{Literal, LoopKind},
    sym::Sym,
    types::{DType, Elem, ExtentExpr, NonEmpty, RuntimeExtentId, TensorType, ValueType},
};
use seismic_realization::{
    dispatch::LinearIterationMap,
    executable::{
        self, BuilderError, Legalized, PhysicalStorageTemplateId, StatusFieldId, TransportTemplate,
    },
    numerics::NumericalTransfer,
};
use std::collections::{BTreeMap, BTreeSet};

impl seismic_realization::executable::sealed::Sealed for CudaDialect {}

// ---------------------------------------------------------------------------
// Sealed opcode namespace
// ---------------------------------------------------------------------------

/// One kernel-local SSA register; unique within its launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CudaSsa(pub u32);

/// Reserved placeholder base for the positional input references produced by
/// `CudaDialect::legalize`. Real graph value ids start at zero, so anything
/// at or above this base is unambiguously a placeholder the strategy
/// rewrites to the node's real input value (or an absorbed binder's SSA
/// register).
const PLACEHOLDER: u32 = 0x8000_0000;

fn in_(index: usize) -> CudaOperand {
    CudaOperand::Value(GraphValueId(PLACEHOLDER + index as u32))
}

fn placeholder_index(value: GraphValueId) -> Option<usize> {
    (value.0 >= PLACEHOLDER).then(|| (value.0 - PLACEHOLDER) as usize)
}

/// One opcode operand: a bound graph value (materialized from its resolved
/// transport by the emitter) or a kernel-local SSA register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaOperand {
    Value(GraphValueId),
    Ssa(CudaSsa),
}

/// Where one op's scalar result is published.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaScalarDest {
    /// Executor scalar slot (device-produced or runtime-rebound).
    Slot(executable::ExecutorScalarSlotId),
    ResolvedSlot(executable::ResolvedExecutorScalarId),
    /// One root-ABI scalar result field.
    Abi {
        path: seismic_lang::types::ValuePath,
        endpoint: Option<seismic_lang::abi::RangeEndpoint>,
        dtype: DType,
    },
    ResolvedResult(executable::ResultScalarFieldId),
    /// Strategy-time placeholder for the node's i-th output value.
    Output(u32),
}

/// A storage reference inside one opcode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaStorageRef {
    Template(PhysicalStorageTemplateId),
    Resolved(executable::ResolvedStorageId),
    /// Placeholder for the node's i-th input value's storage.
    Input(u32),
    /// Placeholder for the node's i-th output value's storage.
    Output(u32),
    /// A boundary storage: transports through the value's launch binding
    /// and resolves to the caller's storage at emission.
    Binding(GraphValueId),
}

/// How one atomic add is realized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtomicMode {
    /// Serialized exact load/add/round/store (the universal column).
    Serialized,
    /// Planned compare/exchange loop over the containing 32-bit word,
    /// which is proved to lie inside the tensor's own allocation — the loop
    /// never reads beyond an ABI allocation.
    CasWord,
}

/// Whether the planned CAS narrow-floating atomic add is admissible: the
/// last element's containing 32-bit word must lie inside the tensor's own
/// dense allocation — 32-bit elements always qualify; narrow floats need
/// an even static element count (an odd final half straddles the end).
pub fn cas_admissible(view_shape: &[ExtentExpr], dtype: DType) -> bool {
    match dtype {
        DType::F32 | DType::I32 | DType::U32 => true,
        DType::F16 | DType::BF16 => {
            let mut elements = 1u64;
            for extent in view_shape {
                match extent.as_static() {
                    Some(n) => elements = elements.saturating_mul(n),
                    // A runtime count cannot prove word containment.
                    None => return false,
                }
            }
            elements % 2 == 0
        }
        DType::Bool => false,
    }
}

/// How one transcendental is realized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathMode {
    /// The versioned `seismic_math` software sequence: the exact reference.
    Software,
    /// PTX approximate instruction carrying an `Approximate` transfer;
    /// admitted only with a bound or accepted evidence under the policy.
    FastApprox,
}

/// Eq-encoded typed constant carried by opcodes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaConst {
    Int(i64),
    FloatBits(u32),
    Bool(bool),
    ShapeParam(String),
}

impl CudaConst {
    fn of(value: &Literal) -> CudaConst {
        match value {
            Literal::Int(bits) => CudaConst::Int(*bits),
            Literal::Float(value) => CudaConst::FloatBits((*value as f32).to_bits()),
            Literal::Bool(flag) => CudaConst::Bool(*flag),
            Literal::ShapeParam(name) => CudaConst::ShapeParam(name.clone()),
        }
    }
}

/// Eq-encoded view transform carried by opcodes (the emitter addresses in
/// storage coordinates exactly as `ViewTransform` describes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaView {
    Identity,
    Reshape { source_shape: Vec<ExtentExpr> },
    Transpose { permutation: Vec<u32> },
    Slice { axes: Vec<CudaSliceAxis> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaSliceAxis {
    Full,
    Point(GraphValueId),
    Range {
        start: Option<GraphValueId>,
        end: Option<GraphValueId>,
    },
}

impl CudaView {
    fn of(view: &ViewTransform) -> CudaView {
        match view {
            ViewTransform::Identity => CudaView::Identity,
            ViewTransform::Reshape { source_shape } => CudaView::Reshape {
                source_shape: source_shape.clone(),
            },
            ViewTransform::Transpose { permutation } => CudaView::Transpose {
                permutation: permutation.clone(),
            },
            ViewTransform::Slice { axes } => CudaView::Slice {
                axes: axes
                    .iter()
                    .map(|axis| match axis {
                        logical::SliceAxis::Full => CudaSliceAxis::Full,
                        logical::SliceAxis::Point(value) => CudaSliceAxis::Point(*value),
                        logical::SliceAxis::Range { start, end } => CudaSliceAxis::Range {
                            start: *start,
                            end: *end,
                        },
                    })
                    .collect(),
            },
        }
    }
}

/// Length of one in-kernel serial loop.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SerialLength {
    Static(u64),
    Runtime(RuntimeExtentId),
}

impl SerialLength {
    fn of(extent: &ExtentExpr) -> Result<SerialLength, String> {
        match extent {
            ExtentExpr::Static(n) => Ok(SerialLength::Static(*n)),
            // An unresolved symbol at specialization is a compiler bug; a
            // wrong loop length would silently skip work.
            ExtentExpr::Sym(sym) => sym
                .as_constant()
                .and_then(|c| u64::try_from(c).ok())
                .map(SerialLength::Static)
                .ok_or_else(|| {
                    format!(
                        "an unresolved planning symbol `{sym}` survived into a serial loop length"
                    )
                }),
            ExtentExpr::Runtime(id) => Ok(SerialLength::Runtime(*id)),
        }
    }
}

/// Kind of one planned runtime check (mirrors the safety taxonomy).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckKind {
    IndexInBounds,
    RangeInBounds,
    DivisionByZero,
    DivisionOverflow,
    ShiftOutOfRange,
    ShapeOverflow,
    EmptyReductionInput,
}

/// The sealed CUDA opcode set. PTX emission matches this enum exhaustively;
/// no selected opcode is rejected by the emitter, and no check is added or
/// omitted: every guard comes from a discharge decided during alternative
/// construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaOp {
    /// Consumed marker for plan-declared objects that emit nothing
    /// (`tensor` allocation, view-only transforms).
    NoOp,
    /// Typed constant.
    Const {
        dest: CudaSsa,
        value: CudaConst,
        dtype: DType,
    },
    /// The retained runtime value of one runtime extent (a kernel param).
    ExtentValue {
        dest: CudaSsa,
        extent: RuntimeExtentId,
    },
    /// The participant's coordinate along one axis of this launch's linear
    /// iteration map (an absorbed independent-loop binder).
    AxisCoordinate { axis: u32, dest: CudaSsa },
    /// An in-kernel serial ascending loop; `binder` is rebound each visit.
    SerialFor {
        binder: CudaSsa,
        length: SerialLength,
        body: Vec<CudaOp>,
    },

    Unary {
        op: seismic_lang::syntax::ast::UnaryOp,
        source: CudaOperand,
        dest: CudaSsa,
    },
    Binary {
        op: seismic_lang::syntax::ast::BinaryOp,
        lhs: CudaOperand,
        rhs: CudaOperand,
        dest: CudaSsa,
        dtype: DType,
        guard: Option<CudaSsa>,
    },
    Fma {
        a: CudaOperand,
        b: CudaOperand,
        c: CudaOperand,
        dest: CudaSsa,
        dtype: DType,
    },
    Cast {
        source: CudaOperand,
        dest: CudaSsa,
        source_dtype: DType,
        target_dtype: DType,
    },
    Math {
        op: MathOp,
        arguments: Vec<CudaOperand>,
        dest: CudaSsa,
        mode: MathMode,
    },
    Select {
        condition: CudaOperand,
        then_value: CudaOperand,
        else_value: CudaOperand,
        dest: CudaSsa,
        dtype: DType,
    },
    TuplePack {
        parts: Vec<CudaOperand>,
        dest: CudaSsa,
    },
    TupleGet {
        source: CudaOperand,
        index: u32,
        dest: CudaSsa,
    },
    RangeMake {
        start: CudaOperand,
        end: CudaOperand,
        dest: CudaSsa,
    },
    /// Start endpoint of a range value.
    RangeStart { source: CudaOperand, dest: CudaSsa },
    /// End endpoint of a range value.
    RangeEnd { source: CudaOperand, dest: CudaSsa },
    /// `extent` / `valid extent` of one bound tensor value.
    ExtentOf {
        base: CudaOperand,
        axis: u32,
        dest: CudaSsa,
        valid: bool,
    },

    /// Checked element load; `base` is a storage-backed tensor value whose
    /// resolved transport supplies the pointer, `view` its transform.
    ElementRead {
        base: CudaOperand,
        view: CudaView,
        view_shape: Vec<ExtentExpr>,
        indices: Vec<CudaOperand>,
        dest: CudaSsa,
        dtype: DType,
        guard: Option<CudaSsa>,
    },
    /// Checked element store.
    ElementWrite {
        base: CudaOperand,
        view: CudaView,
        view_shape: Vec<ExtentExpr>,
        indices: Vec<CudaOperand>,
        value: CudaOperand,
        dtype: DType,
        guard: Option<CudaSsa>,
    },
    /// One atomic update of an element. `mode` is `Serialized` for every
    /// operation; the planned CAS word loop exists for `add` only.
    Atomic {
        op: seismic_lang::intrinsics::AtomicOp,
        base: CudaOperand,
        view: CudaView,
        view_shape: Vec<ExtentExpr>,
        indices: Vec<CudaOperand>,
        value: CudaOperand,
        dtype: DType,
        mode: AtomicMode,
        guard: Option<CudaSsa>,
    },
    /// Arbitrary-rank linear fill of one destination storage.
    Fill {
        dest: CudaStorageRef,
        view: CudaView,
        view_shape: Vec<ExtentExpr>,
        dtype: DType,
        value: CudaOperand,
    },
    /// Arbitrary-rank linear element copy between two storages
    /// (`copy.into`/`materialize`/`clone`/`load` plane traffic).
    CopyElements {
        source: CudaStorageRef,
        source_view: CudaView,
        dest: CudaStorageRef,
        dest_view: CudaView,
        view_shape: Vec<ExtentExpr>,
        dtype: DType,
    },
    /// Packed decode: dense f32 elements decoded from representation
    /// planes (`decode`, and `load` of a packed view).
    Decode {
        source: CudaStorageRef,
        source_view: CudaView,
        dest: CudaStorageRef,
        dest_view: CudaView,
        view_shape: Vec<ExtentExpr>,
        repr: String,
    },
    /// Read one representation plane of a packed value (read-only).
    PackedPlaneRead {
        source: CudaStorageRef,
        view: CudaView,
        plane: PlaneField,
        view_shape: Vec<ExtentExpr>,
        repr: String,
        dest: CudaSsa,
    },
    /// One decoded element of a packed view, addressed through the
    /// representation planes.
    PackedElementRead {
        source: CudaStorageRef,
        view: CudaView,
        view_shape: Vec<ExtentExpr>,
        indices: Vec<CudaOperand>,
        repr: String,
        dest: CudaSsa,
        guard: Option<CudaSsa>,
    },
    /// Publish one scalar into its executor destination.
    StoreScalar {
        dest: CudaScalarDest,
        source: CudaOperand,
        dtype: DType,
    },
    /// One planned runtime predicate of the node identified by
    /// `(node, index)`: writes the first error to that discharge's status
    /// field and defines the guard register (false when violated).
    Check {
        node: executable::NodeRef,
        index: usize,
        kind: CheckKind,
        values: Vec<GraphValueId>,
        extents: Vec<ExtentExpr>,
        status: StatusFieldId,
        guard: CudaSsa,
    },
    /// Parallel outer coordinates with an in-kernel serial ascending fold
    /// of the reduced axis (the universal reduction strategy). `nonempty`
    /// names the declared status field written when an identity-less fold
    /// finds an empty reduced axis.
    SerialFold {
        operand: CudaStorageRef,
        view: CudaView,
        view_shape: Vec<ExtentExpr>,
        axis: usize,
        length: ExtentExpr,
        op: ReduceOp,
        accumulator: DType,
        dest: CudaScalarDest,
        nonempty: Option<StatusFieldId>,
    },
    /// `cuda.subgroup.lane_index`.
    LaneIndex { dest: CudaSsa },
    /// `cuda.subgroup.shuffle`.
    Shuffle {
        value: CudaOperand,
        index: CudaOperand,
        dest: CudaSsa,
        dtype: DType,
    },
    /// `cuda.subgroup.simd_{sum,max,min}` — an optional capability strategy.
    SubgroupReduce {
        op: ReduceOp,
        value: CudaOperand,
        dest: CudaSsa,
        dtype: DType,
    },
}

fn nonempty_vec<T>(items: Vec<T>) -> NonEmpty<T> {
    NonEmpty::new(items).expect("the input is nonempty")
}

fn ops_of(items: Vec<CudaOp>) -> Legalized<CudaOp> {
    debug_assert!(!items.is_empty(), "a universal legalization is nonempty");
    Legalized::Ops(nonempty_vec(items))
}

// ---------------------------------------------------------------------------
// Layouts
// ---------------------------------------------------------------------------

/// Dense storage layout: element dtype plus a symbolic element count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaLayoutTemplate {
    pub dtype: Option<DType>,
    pub elements: Sym,
}

/// Resolved dense layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaResolvedLayout {
    pub dtype: Option<DType>,
    pub elements: u64,
}

fn layout_of(tensor: &TensorType) -> CudaLayoutTemplate {
    let (dtype, per_element) = match &tensor.elem {
        Elem::Dtype(dtype) => (Some(*dtype), u64::from(dtype.bytes())),
        Elem::Param(_) => (Some(DType::F32), 4),
        // Packed storage: outer rows times the per-row plane extents over
        // the packed axis (the same sizing as the logical storage bytes).
        Elem::Repr(name) => {
            let representation = repr::lookup(name);
            let packed_axis = tensor
                .packed_axis
                .unwrap_or(tensor.axes.len().saturating_sub(1));
            let packed_extent = tensor
                .axes
                .get(packed_axis)
                .cloned()
                .unwrap_or(ExtentExpr::Static(0));
            let packed_symbol = match &packed_extent {
                ExtentExpr::Static(n) => Sym::constant(i64::try_from(*n).unwrap_or(i64::MAX)),
                ExtentExpr::Sym(sym) => sym.clone(),
                ExtentExpr::Runtime(id) => Sym::param(&format!("@runtime{}", id.0)),
            };
            let mut rows = Sym::constant(1);
            for (axis, extent) in tensor.axes.iter().enumerate() {
                if axis != packed_axis {
                    let factor = match extent {
                        ExtentExpr::Static(n) => {
                            Sym::constant(i64::try_from(*n).unwrap_or(i64::MAX))
                        }
                        ExtentExpr::Sym(sym) => sym.clone(),
                        ExtentExpr::Runtime(id) => Sym::param(&format!("@runtime{}", id.0)),
                    };
                    rows = rows.mul(&factor);
                }
            }
            let mut bytes = Sym::constant(0);
            if let Some(representation) = representation {
                for plane in representation.planes() {
                    let plane_elements = plane.extent(&packed_symbol);
                    let plane_bytes = i64::try_from(plane.dtype().bytes()).unwrap_or(i64::MAX);
                    bytes = bytes.add(&plane_elements.scale(plane_bytes));
                }
            }
            return CudaLayoutTemplate {
                dtype: Some(DType::F32),
                elements: rows.mul(&bytes),
            };
        }
    };
    let mut elements = Sym::constant(1);
    for axis in &tensor.axes {
        let factor = match axis {
            ExtentExpr::Static(n) => Sym::constant(i64::try_from(*n).unwrap_or(i64::MAX)),
            ExtentExpr::Sym(sym) => sym.clone(),
            // Runtime extents contribute their capacity as the resource
            // bound; semantics always use the retained runtime value.
            ExtentExpr::Runtime(id) => Sym::param(&format!("@runtime{}", id.0)),
        };
        elements = elements.mul(&factor);
    }
    CudaLayoutTemplate {
        dtype,
        elements: elements.scale(per_element as i64),
    }
}

// ---------------------------------------------------------------------------
// The dialect
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CudaDialect;

fn subgroup_intrinsic(name: &str) -> IntrinsicId {
    IntrinsicId {
        capability: seismic_lang::intrinsics::CapabilityId::new("cuda", "subgroup"),
        name: name.into(),
    }
}

impl executable::ExecutableDialect for CudaDialect {
    type Op = CudaOp;
    type LayoutTemplate = CudaLayoutTemplate;
    type ResolvedLayout = CudaResolvedLayout;

    /// Legalize one physical primitive against the effective target. The
    /// portable matrix is exhaustive; `Inapplicable` names only a capability
    /// application without an effective exact signature — an optional
    /// optimized route, since the portable reference body remains a peer
    /// semantic alternative.
    fn legalize(
        p: &executable::PhysicalPrimitive,
        t: &executable::EffectiveTargetProfile,
    ) -> Legalized<CudaOp> {
        use seismic_lang::logical::PrimitiveOp;
        let inputs = &p.inputs;
        let dtype_of = |index: usize| -> Option<DType> {
            inputs.get(index).and_then(|ty| match ty {
                ValueType::Scalar(d) => Some(*d),
                ValueType::Index { .. } => Some(DType::I32),
                ValueType::Tensor(s) => match &s.elem {
                    Elem::Dtype(d) => Some(*d),
                    Elem::Param(_) => Some(DType::F32),
                    Elem::Repr(_) => None,
                },
                _ => None,
            })
        };
        let arithmetic_dtype = || dtype_of(0).or_else(|| dtype_of(1)).unwrap_or(DType::F32);
        let shape_of = |tys: &[ValueType], index: usize| -> Vec<ExtentExpr> {
            tys.get(index)
                .and_then(|ty| ty.shaped().map(|s| s.axes.clone()))
                .unwrap_or_default()
        };
        match &p.op {
            PrimitiveOp::Constant(value) => ops_of(vec![CudaOp::Const {
                dest: CudaSsa(0),
                value: CudaConst::of(value),
                dtype: dtype_of(0).unwrap_or(DType::I32),
            }]),
            PrimitiveOp::RuntimeExtent(id) => ops_of(vec![CudaOp::ExtentValue {
                dest: CudaSsa(0),
                extent: *id,
            }]),
            PrimitiveOp::Capability(intrinsic) => {
                if !t.effective_signatures.contains(intrinsic) {
                    return Legalized::Inapplicable {
                        reason: format!(
                            "capability `{}` has no effective signature on this target",
                            intrinsic.path()
                        ),
                    };
                }
                // `cuda.subgroup` only: `cuda.matrix` stays absent until its
                // typing, reference, legalization, resources, numerics and
                // emission are complete.
                let op = match (intrinsic.capability.name.as_str(), intrinsic.name.as_str()) {
                    ("subgroup", "lane_index") => CudaOp::LaneIndex { dest: CudaSsa(0) },
                    ("subgroup", "shuffle") => CudaOp::Shuffle {
                        value: in_(0),
                        index: in_(1),
                        dest: CudaSsa(2),
                        dtype: dtype_of(0).unwrap_or(DType::F32),
                    },
                    ("subgroup", "simd_sum") => CudaOp::SubgroupReduce {
                        op: ReduceOp::Sum,
                        value: in_(0),
                        dest: CudaSsa(1),
                        dtype: dtype_of(0).unwrap_or(DType::F32),
                    },
                    ("subgroup", "simd_max") => CudaOp::SubgroupReduce {
                        op: ReduceOp::Max,
                        value: in_(0),
                        dest: CudaSsa(1),
                        dtype: dtype_of(0).unwrap_or(DType::F32),
                    },
                    ("subgroup", "simd_min") => CudaOp::SubgroupReduce {
                        op: ReduceOp::Min,
                        value: in_(0),
                        dest: CudaSsa(1),
                        dtype: dtype_of(0).unwrap_or(DType::F32),
                    },
                    _ => {
                        return Legalized::Inapplicable {
                            reason: format!(
                                "CUDA capability `{}` has no complete realization; \
                                 cuda.matrix stays absent until complete",
                                intrinsic.path()
                            ),
                        };
                    }
                };
                ops_of(vec![op])
            }
            PrimitiveOp::Primitive(id) => match id {
                PrimitiveId::TuplePack => ops_of(vec![CudaOp::TuplePack {
                    parts: (0..inputs.len()).map(in_).collect(),
                    dest: CudaSsa(0),
                }]),
                PrimitiveId::TupleGet(index) => ops_of(vec![CudaOp::TupleGet {
                    source: in_(0),
                    index: *index as u32,
                    dest: CudaSsa(0),
                }]),
                PrimitiveId::RangeMake => ops_of(vec![CudaOp::RangeMake {
                    start: in_(0),
                    end: in_(1),
                    dest: CudaSsa(0),
                }]),
                PrimitiveId::RangeStart => ops_of(vec![CudaOp::RangeStart {
                    source: in_(0),
                    dest: CudaSsa(0),
                }]),
                PrimitiveId::RangeEnd => ops_of(vec![CudaOp::RangeEnd {
                    source: in_(0),
                    dest: CudaSsa(0),
                }]),
                PrimitiveId::Select => ops_of(vec![CudaOp::Select {
                    condition: in_(0),
                    then_value: in_(1),
                    else_value: in_(2),
                    dest: CudaSsa(0),
                    dtype: dtype_of(1).unwrap_or(DType::F32),
                }]),
                PrimitiveId::Extent { axis } => ops_of(vec![CudaOp::ExtentOf {
                    base: in_(0),
                    axis: *axis as u32,
                    dest: CudaSsa(0),
                    valid: false,
                }]),
                PrimitiveId::ValidExtent { axis } => ops_of(vec![CudaOp::ExtentOf {
                    base: in_(0),
                    axis: *axis as u32,
                    dest: CudaSsa(0),
                    valid: true,
                }]),
                PrimitiveId::Unary(op) => ops_of(vec![CudaOp::Unary {
                    op: *op,
                    source: in_(0),
                    dest: CudaSsa(0),
                }]),
                PrimitiveId::Binary(op) => ops_of(vec![CudaOp::Binary {
                    op: *op,
                    lhs: in_(0),
                    rhs: in_(1),
                    dest: CudaSsa(0),
                    dtype: arithmetic_dtype(),
                    guard: None,
                }]),
                PrimitiveId::Cast(target) => ops_of(vec![CudaOp::Cast {
                    source: in_(0),
                    dest: CudaSsa(0),
                    source_dtype: dtype_of(0).unwrap_or(DType::F32),
                    target_dtype: *target,
                }]),
                PrimitiveId::Math(op) => ops_of(vec![CudaOp::Math {
                    op: *op,
                    arguments: (0..op.arity()).map(in_).collect(),
                    dest: CudaSsa(0),
                    mode: MathMode::Software,
                }]),
                // Storage is declared by the plan and the transform is
                // carried by the accessing op: both emit nothing here.
                PrimitiveId::TensorAlloc { .. }
                | PrimitiveId::SliceView { .. }
                | PrimitiveId::Transpose
                | PrimitiveId::Reshape => ops_of(vec![CudaOp::NoOp]),
                PrimitiveId::Fill { dtype, .. } => ops_of(vec![CudaOp::Fill {
                    dest: CudaStorageRef::Output(0),
                    view: CudaView::Identity,
                    view_shape: shape_of(&p.results, 0),
                    dtype: *dtype,
                    value: in_(0),
                }]),
                PrimitiveId::Materialize | PrimitiveId::Clone | PrimitiveId::Load => {
                    ops_of(vec![CudaOp::CopyElements {
                        source: CudaStorageRef::Input(0),
                        source_view: CudaView::Identity,
                        dest: CudaStorageRef::Output(0),
                        dest_view: CudaView::Identity,
                        view_shape: shape_of(&p.results, 0),
                        dtype: dtype_of(0).unwrap_or(DType::F32),
                    }])
                }
                PrimitiveId::Decode => ops_of(vec![CudaOp::Decode {
                    source: CudaStorageRef::Input(0),
                    source_view: CudaView::Identity,
                    dest: CudaStorageRef::Output(0),
                    dest_view: CudaView::Identity,
                    view_shape: shape_of(&p.results, 0),
                    repr: packed_repr_name(inputs),
                }]),
                PrimitiveId::PackedRead(plane) => ops_of(vec![CudaOp::PackedPlaneRead {
                    source: CudaStorageRef::Input(0),
                    view: CudaView::Identity,
                    plane: *plane,
                    view_shape: shape_of(&p.results, 0),
                    repr: packed_repr_name(inputs),
                    dest: CudaSsa(0),
                }]),
                PrimitiveId::ElementRead { arity } => {
                    let repr = inputs.first().and_then(|ty| match ty {
                        ValueType::Tensor(tensor) => match &tensor.elem {
                            Elem::Repr(name) => Some(name.clone()),
                            _ => None,
                        },
                        _ => None,
                    });
                    match repr {
                        Some(repr) => ops_of(vec![CudaOp::PackedElementRead {
                            source: CudaStorageRef::Input(0),
                            view: CudaView::Identity,
                            view_shape: shape_of(inputs, 0),
                            indices: (1..1 + *arity).map(in_).collect(),
                            repr,
                            dest: CudaSsa(0),
                            guard: None,
                        }]),
                        None => ops_of(vec![CudaOp::ElementRead {
                            base: in_(0),
                            view: CudaView::Identity,
                            view_shape: shape_of(inputs, 0),
                            indices: (1..1 + *arity).map(in_).collect(),
                            dest: CudaSsa(0),
                            dtype: dtype_of(1 + *arity).unwrap_or(DType::F32),
                            guard: None,
                        }]),
                    }
                }
                PrimitiveId::ElementWrite { arity } => ops_of(vec![CudaOp::ElementWrite {
                    base: in_(0),
                    view: CudaView::Identity,
                    view_shape: shape_of(inputs, 0),
                    indices: (1..1 + *arity).map(in_).collect(),
                    value: in_(1 + *arity),
                    dtype: dtype_of(1 + *arity).unwrap_or(DType::F32),
                    guard: None,
                }]),
                PrimitiveId::CopyInto => ops_of(vec![CudaOp::CopyElements {
                    source: CudaStorageRef::Input(1),
                    source_view: CudaView::Identity,
                    dest: CudaStorageRef::Input(0),
                    dest_view: CudaView::Identity,
                    view_shape: shape_of(inputs, 1),
                    dtype: dtype_of(1).unwrap_or(DType::F32),
                }]),
                PrimitiveId::Atomic { op, arity } => ops_of(vec![CudaOp::Atomic {
                    op: *op,
                    base: in_(0),
                    view: CudaView::Identity,
                    view_shape: shape_of(inputs, 0),
                    indices: (1..1 + *arity).map(in_).collect(),
                    value: in_(1 + *arity),
                    dtype: dtype_of(1 + *arity).unwrap_or(DType::F32),
                    mode: AtomicMode::Serialized,
                    guard: None,
                }]),
                PrimitiveId::Reduce { op, axis, .. } => {
                    // Reductions are `ReductionNode`s consumed by reduction
                    // strategies; reaching primitive legalization is a bug.
                    let _ = (op, axis);
                    return Legalized::Inapplicable {
                        reason: format!(
                            "the `{}` reduction reached primitive legalization; \
                             reductions are consumed by map_reduction",
                            op.name()
                        ),
                    };
                }
            },
        }
    }

    /// Exact compiler-controlled hard resources, the native
    /// contract, ranking-only cost, capability, and numerics. Costs carry
    /// the `cuda-estimate-unqualified-v0` identity:
    /// uncalibrated coefficients affect ranking only, never legality.
    fn consequences(op: &CudaOp) -> executable::PhysicalConsequences {
        use executable::{
            CostEstimate, HardResources, NativeResourceContract, PhysicalConsequences,
        };
        let (hard, cost, numerical, capability) = match op {
            CudaOp::SubgroupReduce { .. } => (
                HardResources {
                    required_subgroup_width: Some(32),
                    ..HardResources::default()
                },
                2,
                NumericalTransfer::Exact,
                Some(subgroup_intrinsic("simd_sum")),
            ),
            CudaOp::Shuffle { .. } => (
                HardResources {
                    required_subgroup_width: Some(32),
                    ..HardResources::default()
                },
                1,
                NumericalTransfer::Exact,
                Some(subgroup_intrinsic("shuffle")),
            ),
            CudaOp::LaneIndex { .. } => (
                HardResources::default(),
                1,
                NumericalTransfer::Exact,
                Some(subgroup_intrinsic("lane_index")),
            ),
            CudaOp::Math {
                op: math,
                mode: MathMode::FastApprox,
                ..
            } => (
                HardResources::default(),
                1,
                NumericalTransfer::Approximate {
                    operation: primitive(PrimitiveId::Math(*math)).id.clone(),
                    bound: ErrorBound {
                        relative: 2e-6,
                        absolute: 2e-6,
                    },
                },
                None,
            ),
            CudaOp::Atomic {
                mode: AtomicMode::CasWord,
                ..
            } => (
                HardResources::default(),
                // A CAS loop preserves the registry load/add/round/store
                // exactly; only its cost differs.
                4,
                NumericalTransfer::Exact,
                None,
            ),
            CudaOp::Fill { .. } | CudaOp::CopyElements { .. } | CudaOp::Decode { .. } => (
                HardResources {
                    static_code_units: 16,
                    ..HardResources::default()
                },
                8,
                NumericalTransfer::Exact,
                None,
            ),
            CudaOp::SerialFold { .. } => (
                HardResources {
                    static_code_units: 24,
                    ..HardResources::default()
                },
                8,
                NumericalTransfer::Exact,
                None,
            ),
            CudaOp::SerialFor { body, .. } => {
                let mut units = 8u64;
                for nested in body {
                    units = units.saturating_add(u64::from(
                        <CudaDialect as executable::ExecutableDialect>::consequences(nested)
                            .cost
                            .0,
                    ));
                }
                (
                    HardResources {
                        static_code_units: units,
                        ..HardResources::default()
                    },
                    1,
                    NumericalTransfer::Exact,
                    None,
                )
            }
            _ => (HardResources::default(), 1, NumericalTransfer::Exact, None),
        };
        PhysicalConsequences {
            hard,
            native_contract: NativeResourceContract {
                // The universal column bounds live SSA/code shape so the
                // native compiler is guaranteed at least one resident
                // participant; a reflected fact outside this domain is
                // CompilerBug, never a retry.
                max_resident_participants: (1, u64::MAX),
                native_subgroup_width: None,
            },
            cost: CostEstimate(cost),
            numerical,
            capability,
        }
    }

    fn public_layout(tensor: &TensorType) -> CudaLayoutTemplate {
        layout_of(tensor)
    }

    fn internal_layout(tensor: &TensorType) -> CudaLayoutTemplate {
        layout_of(tensor)
    }

    fn resolve_layout(
        layout: &CudaLayoutTemplate,
        values: &executable::PlanValues,
    ) -> Result<CudaResolvedLayout, executable::InvariantReport> {
        let elements = layout
            .elements
            .eval(&|name| values.symbols.get(name).copied())
            .ok_or_else(|| {
                executable::InvariantReport(format!(
                    "CUDA layout element count `{}` is unresolved",
                    layout.elements
                ))
            })?;
        let elements = u64::try_from(elements).map_err(|_| {
            executable::InvariantReport("CUDA layout element count is negative".into())
        })?;
        Ok(CudaResolvedLayout {
            dtype: layout.dtype,
            elements,
        })
    }

    fn resolve_op(
        op: &Self::Op,
        identities: &mut dyn executable::PhysicalIdentityResolver,
    ) -> Result<Self::Op, executable::InvariantReport> {
        resolve_op_identities(op, identities)
    }
}

fn resolve_op_identities(
    op: &CudaOp,
    identities: &mut dyn executable::PhysicalIdentityResolver,
) -> Result<CudaOp, executable::InvariantReport> {
    fn storage(
        reference: &mut CudaStorageRef,
        identities: &mut dyn executable::PhysicalIdentityResolver,
    ) -> Result<(), executable::InvariantReport> {
        if let CudaStorageRef::Template(template) = reference {
            *reference = CudaStorageRef::Resolved(identities.storage(*template)?);
        }
        Ok(())
    }
    fn scalar(
        destination: &mut CudaScalarDest,
        identities: &mut dyn executable::PhysicalIdentityResolver,
    ) -> Result<(), executable::InvariantReport> {
        match destination {
            CudaScalarDest::Slot(template) => {
                *destination = CudaScalarDest::ResolvedSlot(identities.slot(*template)?);
            }
            CudaScalarDest::Abi {
                path,
                endpoint,
                dtype,
            } => {
                *destination = CudaScalarDest::ResolvedResult(
                    identities.result_scalar(path, *endpoint, *dtype)?,
                );
            }
            _ => {}
        }
        Ok(())
    }

    let mut resolved = op.clone();
    match &mut resolved {
        CudaOp::SerialFor { body, .. } => {
            *body = body
                .iter()
                .map(|op| resolve_op_identities(op, identities))
                .collect::<Result<Vec<_>, _>>()?;
        }
        CudaOp::Fill { dest, .. } => storage(dest, identities)?,
        CudaOp::CopyElements { source, dest, .. } | CudaOp::Decode { source, dest, .. } => {
            storage(source, identities)?;
            storage(dest, identities)?;
        }
        CudaOp::PackedPlaneRead { source, .. } | CudaOp::PackedElementRead { source, .. } => {
            storage(source, identities)?
        }
        CudaOp::StoreScalar { dest, .. } => scalar(dest, identities)?,
        CudaOp::SerialFold { operand, dest, .. } => {
            storage(operand, identities)?;
            scalar(dest, identities)?;
        }
        CudaOp::NoOp
        | CudaOp::Const { .. }
        | CudaOp::ExtentValue { .. }
        | CudaOp::AxisCoordinate { .. }
        | CudaOp::Unary { .. }
        | CudaOp::Binary { .. }
        | CudaOp::Fma { .. }
        | CudaOp::Cast { .. }
        | CudaOp::Math { .. }
        | CudaOp::Select { .. }
        | CudaOp::TuplePack { .. }
        | CudaOp::TupleGet { .. }
        | CudaOp::RangeMake { .. }
        | CudaOp::RangeStart { .. }
        | CudaOp::RangeEnd { .. }
        | CudaOp::ExtentOf { .. }
        | CudaOp::ElementRead { .. }
        | CudaOp::ElementWrite { .. }
        | CudaOp::Atomic { .. }
        | CudaOp::Check { .. }
        | CudaOp::LaneIndex { .. }
        | CudaOp::Shuffle { .. }
        | CudaOp::SubgroupReduce { .. } => {}
    }
    Ok(resolved)
}

// ---------------------------------------------------------------------------
// Effective target profile
// ---------------------------------------------------------------------------

/// The effective CUDA target profile: exact `cuda.subgroup` signatures and
/// hard limits. `cuda.matrix` stays absent until complete and never prevents
/// portable matmul.
pub fn cuda_target_profile(
    limits: &crate::mapping::Limits,
    target: &crate::target::TargetProfile,
) -> executable::EffectiveTargetProfile {
    let mut effective = BTreeSet::new();
    for name in ["lane_index", "shuffle", "simd_sum", "simd_max", "simd_min"] {
        effective.insert(subgroup_intrinsic(name));
    }
    executable::EffectiveTargetProfile {
        backend: crate::mapping::TARGET.into(),
        capability_fingerprint: capability_fingerprint(limits, target),
        toolchain_fingerprint: target.fingerprint().to_string(),
        effective_signatures: effective,
        limits: executable::TargetLimits {
            max_participants: i64::from(limits.max_threads_per_block),
            max_workgroups_axis: [i64::from(limits.max_grid_x), 65_535, 65_535],
            max_workgroup_bytes: 48 * 1024,
            max_explicit_private_bytes: i64::try_from(limits.max_scratch_bytes).unwrap_or(i64::MAX),
            max_direct_bindings: (crate::native::MAX_KERNEL_PARAMETER_BYTES / 8) as i64,
            max_argument_table_bytes: 0,
            max_device_bytes: i64::try_from(limits.max_scratch_bytes).unwrap_or(i64::MAX),
        },
    }
}

/// Fingerprint of every availability and identity input.
pub fn capability_fingerprint(
    limits: &crate::mapping::Limits,
    target: &crate::target::TargetProfile,
) -> String {
    format!(
        "seismic-cuda-physical-v3:{}:threads={}:grid={}:warp={}:scratch={}",
        target.fingerprint(),
        limits.max_threads_per_block,
        limits.max_grid_x,
        limits.warp_size,
        limits.max_scratch_bytes
    )
}

// ---------------------------------------------------------------------------
// Strategy construction
// ---------------------------------------------------------------------------

/// Fixed universal grid-stride participant width (one CUDA block). A
/// constant width keeps the emitted geometry mechanical; calibrated width
/// tuning is strategy-library work and never changes legality.
const UNIVERSAL_PARTICIPANTS: i64 = 256;

/// Which optional forms this alternative uses.
#[derive(Clone, Copy, Debug)]
pub struct StrategyFlags {
    /// Use the planned CAS narrow-floating atomic add wherever the layout
    /// admits it.
    pub cas_atomic: bool,
    /// Use the PTX approximate sequence for `exp_fast`, carrying an
    /// `Approximate` transfer (bound/evidence-gated by the policy).
    pub approx_exp_fast: bool,
}

impl StrategyFlags {
    /// The universal column: serialized atomics, exact software math.
    pub const UNIVERSAL: StrategyFlags = StrategyFlags {
        cas_atomic: false,
        approx_exp_fast: false,
    };
    /// The optimized column: CAS atomics and approximate `exp_fast`.
    pub const OPTIMIZED: StrategyFlags = StrategyFlags {
        cas_atomic: true,
        approx_exp_fast: true,
    };
}

/// Construct the complete plan family: for every choice and every logical
/// alternative, the universal physical alternative (#0) plus the optimized
/// alternative (#1). Every applicable portable alternative is covered.
pub fn elaborate(
    logical: &LogicalProgram,
    target: &executable::EffectiveTargetProfile,
) -> Result<executable::PlanFamily<CudaDialect>, BuilderError> {
    let mut family = executable::PlanFamilyBuilder::from_logical(logical)?;
    for choice in logical.choices.ids() {
        let count = logical.choice(choice).alternatives.iter().count() as u32;
        for logical_alternative in 0..count {
            for flags in [StrategyFlags::UNIVERSAL, StrategyFlags::OPTIMIZED] {
                let mut builder = family.alternative(choice, logical_alternative)?;
                let graph = builder.graph().clone();
                let facts = GraphFacts::collect(&graph, &logical.runtime_extents);
                let ctx = GraphContext::collect(&graph).with_extents(&facts.runtime_extents);
                map_region(&mut builder, Vec::new(), &ctx, target, flags)?;
                complete_boundary(&mut builder, &graph)?;
                let alternative = builder.finish_alternative()?;
                family.add_alternative(choice, alternative)?;
            }
        }
    }
    family.finish()
}

/// Precomputed per-graph context: the view transform and backing storage
/// of every tensor value, runtime extents, and (per mapping level) absorbed
/// axes, in-kernel serial frames, and binder SSA bindings.
#[derive(Clone, Default)]
struct GraphContext {
    #[allow(clippy::type_complexity)]
    value_views: BTreeMap<GraphValueId, ViewTransform>,
    value_storages: BTreeMap<GraphValueId, LogicalStorageId>,
    runtime_extents: BTreeMap<RuntimeExtentId, RuntimeExtent>,
    /// Independent axes absorbed into the current launch's iteration map,
    /// outermost first.
    axes: Vec<ExtentExpr>,
    /// In-kernel serial frames, outermost first.
    serial: Vec<SerialFrame>,
    /// The SSA register and binding kind of each in-scope loop binder.
    binder_kinds: BTreeMap<GraphValueId, (CudaSsa, BinderKind)>,
}

#[derive(Clone, Debug)]
struct SerialFrame {
    binder: CudaSsa,
    length: SerialLength,
}

impl GraphContext {
    fn collect(graph: &TaskGraph) -> GraphContext {
        let mut ctx = GraphContext::default();
        ctx.walk_region(graph, &graph.root.clone());
        ctx
    }

    fn with_extents(mut self, extents: &BTreeMap<RuntimeExtentId, RuntimeExtent>) -> GraphContext {
        self.runtime_extents = extents.clone();
        self
    }

    fn walk_region(&mut self, graph: &TaskGraph, region: &GraphRegion) {
        // A tensor `Value` parameter is paired with the `State` parameter of
        // its storage (the logical builder's convention); its view is the
        // identity view of that storage.
        for (index, parameter) in region.parameters.iter().enumerate() {
            if let RegionParameter::State { storage, .. } = parameter {
                if index > 0 {
                    if let Some(RegionParameter::Value { id, ty }) =
                        region.parameters.get(index - 1)
                    {
                        if matches!(ty, ValueType::Tensor(_)) {
                            self.value_storages.insert(*id, *storage);
                            self.value_views.insert(*id, ViewTransform::Identity);
                        }
                    }
                }
            }
        }
        for node in region.nodes.iter() {
            for output in &node.outputs {
                if let Some(view) = output.view {
                    if let Some(logical) = graph.views.get(view) {
                        self.value_storages.insert(output.id, logical.storage);
                        self.value_views
                            .insert(output.id, logical.transform.clone());
                    }
                }
            }
            match &node.kind {
                LogicalNodeKind::If(if_node) => {
                    self.walk_region(graph, &if_node.then_region);
                    self.walk_region(graph, &if_node.else_region);
                }
                LogicalNodeKind::Loop(loop_node) => {
                    self.walk_region(graph, &loop_node.body);
                }
                _ => {}
            }
        }
    }

    /// The linear iteration map over the absorbed independent axes with the
    /// universal grid-stride participant width (the serial map when no axis
    /// is absorbed).
    fn launch_map(&self) -> Result<LinearIterationMap, BuilderError> {
        if self.axes.is_empty() {
            Ok(LinearIterationMap::serial())
        } else {
            LinearIterationMap::linear(&self.axes, &self.runtime_extents)
                .map(|map| map.with_participants(Sym::constant(UNIVERSAL_PARTICIPANTS)))
                .map_err(|error| format!("infeasible CUDA launch geometry: {error}").into())
        }
    }
}

/// Map every node of one region with the universal strategy.
fn map_region(
    builder: &mut executable::AlternativeBuilder<CudaDialect>,
    region: executable::RegionPath,
    ctx: &GraphContext,
    target: &executable::EffectiveTargetProfile,
    flags: StrategyFlags,
) -> Result<(), BuilderError> {
    let graph = builder.graph().clone();
    let snapshot = region_snapshot(&graph, &region)?;
    let mut node_ids: Vec<NodeId> = snapshot.nodes.keys().copied().collect();
    node_ids.sort_by_key(|id| id.0);
    for node_id in node_ids {
        let node = snapshot
            .nodes
            .get(&node_id)
            .cloned()
            .expect("the node exists");
        let node_ref = executable::NodeRef {
            region: region.clone(),
            node: node_id,
        };
        match &node.kind {
            LogicalNodeKind::Primitive(_) => {
                lower_primitive(builder, &node_ref, ctx, target, flags)?
            }
            LogicalNodeKind::Reduction(_) => lower_reduction(builder, &node_ref, ctx)?,
            LogicalNodeKind::Loop(loop_node) => {
                lower_loop(builder, &node_ref, loop_node, ctx, target, flags)?
            }
            LogicalNodeKind::If(_) => lower_if(builder, &node_ref, &node, ctx, target, flags)?,
            LogicalNodeKind::Call(_) => lower_call(builder, &node_ref, &node, ctx)?,
        }
    }
    Ok(())
}

fn facts_of(
    builder: &executable::AlternativeBuilder<CudaDialect>,
    ctx: &GraphContext,
) -> GraphFacts {
    let extents = seismic_lang::logical::IdVec::from_iter(
        ctx.runtime_extents
            .iter()
            .map(|(id, extent)| (*id, extent.clone())),
    );
    GraphFacts::collect(builder.graph(), &extents)
}

fn region_snapshot(
    graph: &TaskGraph,
    path: &executable::RegionPath,
) -> Result<RegionSnapshot, BuilderError> {
    let mut current = graph.root.clone();
    for step in path {
        let node = current
            .nodes
            .get(step.node())
            .ok_or_else(|| format!("region path names absent node#{}", step.node().0))?;
        match (&node.kind, step) {
            (LogicalNodeKind::If(if_node), executable::RegionStep::IfThen(_)) => {
                current = if_node.then_region.clone()
            }
            (LogicalNodeKind::If(if_node), executable::RegionStep::IfElse(_)) => {
                current = if_node.else_region.clone()
            }
            (LogicalNodeKind::Loop(loop_node), executable::RegionStep::LoopBody(_)) => {
                current = loop_node.body.clone()
            }
            _ => return Err("region path disagrees with graph structure".into()),
        }
    }
    let mut nodes = BTreeMap::new();
    for (node_id, node) in current.nodes.ids().zip(current.nodes.iter()) {
        nodes.insert(node_id, node.clone());
    }
    Ok(RegionSnapshot {
        nodes,
        results: current.results.clone(),
    })
}

struct RegionSnapshot {
    nodes: BTreeMap<NodeId, LogicalNode>,
    results: Vec<RegionResult>,
}

// --- absorbed binder registers --------------------------------------------

/// Deterministic kernel-local register of one loop binder: derived from the
/// loop node id, so absorbed binders never collide with node SSA registers
/// (strategy-local, allocated from zero) or placeholder values.
fn binder_register(node: NodeId) -> CudaSsa {
    CudaSsa(0x4000_0000 + node.0)
}

/// How one in-scope binder is bound in the current absorption context.
#[derive(Clone, Copy, Debug)]
enum BinderKind {
    /// The participant's coordinate along `axis` of this launch's iteration
    /// map (an absorbed independent loop).
    Axis(u32),
    /// An in-kernel serial loop counter (an absorbed ordered loop); the
    /// `SerialFor` opcode rebinds it each iteration.
    Serial,
}

// --- loops ---------------------------------------------------------------

/// Whether every node of this region (transitively, through nested loops)
/// is a primitive, a reduction, or an ordered loop — the precondition for
/// absorbing an independent loop's domain into its body's launches.
fn absorbable(region: &GraphRegion) -> bool {
    for node in region.nodes.iter() {
        match &node.kind {
            LogicalNodeKind::Primitive(_) | LogicalNodeKind::Reduction(_) => {}
            LogicalNodeKind::Loop(loop_node) if loop_node.kind == LoopKind::Ordered => {
                if !absorbable(&loop_node.body) {
                    return false;
                }
            }
            // An `if`, a call, or a nested independent loop forces executor
            // structure around this region.
            _ => return false,
        }
    }
    true
}

fn lower_loop(
    builder: &mut executable::AlternativeBuilder<CudaDialect>,
    node: &executable::NodeRef,
    loop_node: &LoopNode,
    ctx: &GraphContext,
    target: &executable::EffectiveTargetProfile,
    flags: StrategyFlags,
) -> Result<(), BuilderError> {
    let body_path = {
        let mut path = node.region.clone();
        path.push(executable::RegionStep::LoopBody(node.node));
        path
    };
    let absorbed = ctx.binder_kinds.contains_key(&loop_node.binder);
    match loop_node.kind {
        LoopKind::Independent if absorbed || absorbable(&loop_node.body) => {
            // Kernel-local control: the independent axis joins this launch's
            // linear iteration map and the executor repeats exactly once.
            let mut next = ctx.clone();
            let axis = next.axes.len() as u32;
            next.axes.push(loop_node.range.bound.clone());
            next.binder_kinds.insert(
                loop_node.binder,
                (binder_register(node.node), BinderKind::Axis(axis)),
            );
            absorb_inner(&mut next, &loop_node.body)?;
            let range = single_visit_range(builder, node.node);
            builder.schedule_loop(node.clone(), range, Vec::new(), |body| {
                map_region(body, body_path, &next, target, flags)
            })
        }
        _ => {
            // A plain executor `Repeat` with the loop's real retained range;
            // the binder and carries are executor values rebound per visit
            // and resolved by the emitter through the launch's bindings.
            let start = builder.transport_of(loop_node.range.start)?;
            let end = builder.transport_of(loop_node.range.end)?;
            let carried = loop_carries(builder, loop_node)?;
            builder.schedule_loop(
                node.clone(),
                executable::ExecutorRangeTemplate {
                    start,
                    end,
                    bound: loop_node.range.bound.clone(),
                },
                carried,
                |body| map_region(body, body_path, ctx, target, flags),
            )
        }
    }
}

/// The retained executor range of an absorbed loop: one visit over the
/// constants 0 and 1, carried by computed constant transports.
fn single_visit_range(
    _builder: &mut executable::AlternativeBuilder<CudaDialect>,
    _node: NodeId,
) -> executable::ExecutorRangeTemplate {
    let endpoint = |constant: i64| {
        executable::TransportTemplate::ExecutorScalar(executable::ExecutorScalarTemplate {
            source: executable::ExecutorScalarSource::Computed(
                executable::ExecutorComputedScalar::Const(constant),
            ),
            dtype: DType::I32,
        })
    };
    executable::ExecutorRangeTemplate {
        start: endpoint(0),
        end: endpoint(1),
        bound: ExtentExpr::Static(1),
    }
}

/// Record the ordered loops nested inside an absorbed body as in-kernel
/// serial frames, outermost first.
fn absorb_inner(ctx: &mut GraphContext, region: &GraphRegion) -> Result<(), BuilderError> {
    for inner in region.nodes.iter() {
        if let LogicalNodeKind::Loop(inner_loop) = &inner.kind {
            if inner_loop.kind == LoopKind::Ordered {
                // The frame is pending until its region mapping begins; the
                // register is derived from the loop node id.
                let node_id = region
                    .nodes
                    .ids()
                    .zip(region.nodes.iter())
                    .find(|(_, candidate)| same_loop(candidate, inner_loop))
                    .map(|(id, _)| id);
                if let Some(node_id) = node_id {
                    ctx.serial.push(SerialFrame {
                        binder: binder_register(node_id),
                        length: SerialLength::of(&inner_loop.range.bound)?,
                    });
                    ctx.binder_kinds.insert(
                        inner_loop.binder,
                        (binder_register(node_id), BinderKind::Serial),
                    );
                }
                absorb_inner(ctx, &inner_loop.body)?;
            }
        }
    }
    Ok(())
}

/// Whether two loop nodes are the same occurrence (their binder and range
/// identify the node within its region).
fn same_loop(candidate: &LogicalNode, loop_node: &LoopNode) -> bool {
    if let LogicalNodeKind::Loop(other) = &candidate.kind {
        other.binder == loop_node.binder && other.range.bound == loop_node.range.bound
    } else {
        false
    }
}

/// The carried transports of one executor loop, in carried order.
fn loop_carries(
    builder: &mut executable::AlternativeBuilder<CudaDialect>,
    loop_node: &LoopNode,
) -> Result<Vec<executable::PhysicalCarryTemplate>, BuilderError> {
    let graph = builder.graph().clone();
    let mut carries = Vec::new();
    for slot in &loop_node.carried {
        let transport = match slot.initial {
            seismic_lang::logical::RegionInput::Value(initial) => {
                let initial_transport = builder.transport_of(initial)?;
                match initial_transport {
                    executable::TransportTemplate::Storage(_)
                    | executable::TransportTemplate::ExecutorScalar(_) => initial_transport,
                    _ => {
                        let dtype = carried_dtype(&graph, initial);
                        executable::TransportTemplate::ExecutorScalar(
                            executable::ExecutorScalarTemplate {
                                source: executable::ExecutorScalarSource::Slot(
                                    builder.declare_executor_scalar_slot(dtype, "carry"),
                                ),
                                dtype,
                            },
                        )
                    }
                }
            }
            seismic_lang::logical::RegionInput::State(token) => {
                let storage = builder
                    .logical_storage_of_token(token)
                    .ok_or("a carried state token has no storage")?;
                state_transport(builder, &graph, storage)?
            }
        };
        carries.push(executable::PhysicalCarryTemplate { transport });
    }
    Ok(carries)
}

fn carried_dtype(graph: &TaskGraph, value: GraphValueId) -> DType {
    fn scan(region: &GraphRegion, value: GraphValueId) -> Option<DType> {
        for node in region.nodes.iter() {
            if let Some(output) = node.outputs.iter().find(|output| output.id == value) {
                return output.ty.scalar_dtype();
            }
            match &node.kind {
                LogicalNodeKind::If(if_node) => {
                    if let Some(found) = scan(&if_node.then_region, value) {
                        return Some(found);
                    }
                    if let Some(found) = scan(&if_node.else_region, value) {
                        return Some(found);
                    }
                }
                LogicalNodeKind::Loop(loop_node) => {
                    if let Some(found) = scan(&loop_node.body, value) {
                        return Some(found);
                    }
                }
                _ => {}
            }
        }
        None
    }
    scan(&graph.root, value).unwrap_or(DType::I32)
}

/// The canonical boundary leaf of one storage (its parameter/result origin).
fn boundary_leaf(graph: &TaskGraph, storage: LogicalStorageId) -> executable::BoundaryLeaf {
    graph
        .storages
        .get(storage)
        .map(|s| match &s.origin {
            logical::StorageOrigin::Parameter { ordinal, path, .. } => {
                executable::BoundaryLeaf::Input {
                    param: *ordinal,
                    leaf: path.clone(),
                }
            }
            logical::StorageOrigin::Result { path, .. } => {
                executable::BoundaryLeaf::Result { leaf: path.clone() }
            }
            logical::StorageOrigin::Owned => executable::BoundaryLeaf::default(),
        })
        .unwrap_or_default()
}

/// The tensor transport of one logical storage: its own template, or the
/// caller's boundary placeholder.
fn state_transport(
    builder: &executable::AlternativeBuilder<CudaDialect>,
    graph: &TaskGraph,
    storage: LogicalStorageId,
) -> Result<executable::TransportTemplate, BuilderError> {
    if let Some(template) = builder.storage_of(storage) {
        Ok(executable::TransportTemplate::Storage(one_view(
            template,
            Access::Exclusive,
        )))
    } else {
        Ok(executable::TransportTemplate::Boundary(boundary_leaf(
            graph, storage,
        )))
    }
}

fn one_view(
    template: PhysicalStorageTemplateId,
    access: Access,
) -> NonEmpty<executable::StorageViewTemplate> {
    nonempty_vec(vec![executable::StorageViewTemplate {
        storage: template,
        access,
        transform: ViewTransform::Identity,
    }])
}

// --- conditionals ---------------------------------------------------------

fn lower_if(
    builder: &mut executable::AlternativeBuilder<CudaDialect>,
    node: &executable::NodeRef,
    logical: &LogicalNode,
    ctx: &GraphContext,
    target: &executable::EffectiveTargetProfile,
    flags: StrategyFlags,
) -> Result<(), BuilderError> {
    let LogicalNodeKind::If(if_node) = &logical.kind else {
        return Err("lower_if requires an if node".into());
    };
    let condition = builder.transport_of(if_node.condition)?;
    let then_region = child_path(node, executable::RegionStep::IfThen(node.node));
    let else_region = child_path(node, executable::RegionStep::IfElse(node.node));
    let graph = builder.graph().clone();
    let joins = if_node
        .joins
        .iter()
        .map(|slot| match slot {
            JoinSlot::Value {
                then_result,
                else_result,
                joined,
                ..
            } => {
                let then_value = region_result_value(&graph, &then_region, then_result);
                let else_value = region_result_value(&graph, &else_region, else_result);
                Ok(executable::PhysicalJoinTemplate::Value {
                    then: builder.transport_of(then_value)?,
                    else_branch: builder.transport_of(else_value)?,
                    joined: builder.transport_of(*joined)?,
                })
            }
            JoinSlot::State { storage, .. } => Ok(executable::PhysicalJoinTemplate::State {
                storage: builder
                    .storage_of(*storage)
                    .ok_or("a state join must name storage of this alternative")?,
            }),
        })
        .collect::<Result<Vec<_>, BuilderError>>()?;
    builder.schedule_if(
        node.clone(),
        executable::ExecutorPredicateTemplate { value: condition },
        |then_builder| map_region(then_builder, then_region, ctx, target, flags),
        |else_builder| map_region(else_builder, else_region, ctx, target, flags),
        joins,
    )
}

fn child_path(node: &executable::NodeRef, step: executable::RegionStep) -> executable::RegionPath {
    let mut path = node.region.clone();
    path.push(step);
    path
}

fn region_result_value(
    graph: &TaskGraph,
    region: &executable::RegionPath,
    result: &seismic_lang::logical::RegionResultId,
) -> GraphValueId {
    let snapshot = region_snapshot(graph, region).expect("the branch region exists");
    match snapshot
        .results
        .get(result.index())
        .expect("the join names a real result")
    {
        RegionResult::Value { id, .. } => *id,
        RegionResult::State { .. } => unreachable!("a value join never names a state result"),
    }
}

// --- calls ---------------------------------------------------------------

/// Lower one call node: the boundary carries, per leaf, the caller's
/// transports keyed by canonical boundary leaf. A tensor boundary input
/// supplies both its state entry and the child's view value entry.
fn lower_call(
    builder: &mut executable::AlternativeBuilder<CudaDialect>,
    node: &executable::NodeRef,
    logical: &LogicalNode,
    _ctx: &GraphContext,
) -> Result<(), BuilderError> {
    let LogicalNodeKind::Call(_) = &logical.kind else {
        return Err("lower_call requires a call node".into());
    };
    builder.invoke_canonical(node.clone())
}

// --- boundary completion ---------------------------------------------------

/// Complete every boundary output with the transport its producer recorded.
fn complete_boundary(
    builder: &mut executable::AlternativeBuilder<CudaDialect>,
    graph: &TaskGraph,
) -> Result<(), BuilderError> {
    for ordinal in 0..graph.results.len() as u32 {
        match &graph.results[ordinal as usize] {
            RegionResult::Value { id, .. } => {
                let transport = builder.transport_of(*id)?;
                builder.complete_result(ordinal, transport)?;
            }
            RegionResult::State { storage, .. } => {
                let transport = builder.transport_of_storage(*storage, Access::Exclusive)?;
                builder.complete_result(ordinal, transport)?;
            }
        }
    }
    Ok(())
}

// --- primitives -----------------------------------------------------------

/// SSA id allocator for one node's opcode list.
struct SsaAllocator {
    next: u32,
}

impl SsaAllocator {
    fn fresh(&mut self) -> CudaSsa {
        let id = CudaSsa(self.next);
        self.next += 1;
        id
    }
}

/// Lower one primitive node: form it against the registry, legalize it
/// against the dialect, rewrite the positional placeholders into the node's
/// real values, destinations, storages, and view transforms, wire every
/// discharged obligation as a guard with its status field, and consume it
/// with `map_primitive` under the context's launch iteration.
fn lower_primitive(
    builder: &mut executable::AlternativeBuilder<CudaDialect>,
    node: &executable::NodeRef,
    ctx: &GraphContext,
    target: &executable::EffectiveTargetProfile,
    flags: StrategyFlags,
) -> Result<(), BuilderError> {
    let logical = node_node(builder, node);
    let formed = form_primitive(node.node, &logical, &facts_of(builder, ctx))
        .map_err(|error| format!("CUDA formation of node#{}: {error}", node.node.0))?;
    let blueprint = <CudaDialect as executable::ExecutableDialect>::legalize(
        &formed.physical_primitive(),
        target,
    );
    let Some(blueprint_ops) = blueprint.ops() else {
        return Err(match &formed.op {
            seismic_lang::logical::PrimitiveOp::Capability(intrinsic) => {
                FormationError::RequiresCapability {
                    intrinsic: intrinsic.clone(),
                }
                .to_string()
            }
            _ => FormationError::Bug(
                "a universal primitive mapping legalized as inapplicable".into(),
            )
            .to_string(),
        });
    };
    let mut ssa = SsaAllocator { next: 0 };
    let mut remap: BTreeMap<u32, CudaSsa> = BTreeMap::new();
    let mut ops = Vec::new();
    // Every absorbed independent-loop binder of this context is defined by
    // the launch's iteration map: its axis coordinate. Emission is
    // idempotent, so repeating the definition per node is harmless.
    for (register, kind) in ctx.binder_kinds.values() {
        if let BinderKind::Axis(axis) = kind {
            ops.push(CudaOp::AxisCoordinate {
                axis: *axis,
                dest: *register,
            });
        }
    }
    // Runtime-checked obligations become guards on this node's operations;
    // the discharge receipt names the status field of the first error. The
    // checks precede the guarded operations so the guard register is
    // defined before its first use.
    let mut guard = None;
    for (index, (_, discharge)) in formed.obligations.iter().enumerate() {
        let receipt = discharge_with_builder(
            builder,
            executable::ObligationRef {
                node: node.clone(),
                index,
            },
            discharge,
            // The planned predicate is carried by the `Check` opcode
            // appended below; the builder retains only the discharge.
            |_| ops_of(vec![CudaOp::NoOp]),
        )
        .map_err(|error| format!("CUDA discharge at node#{}: {error}", node.node.0))?;
        if let ObligationDischarge::RuntimeChecked(check) = discharge {
            if let Some(status) = receipt {
                let (kind, values, extents) = decompose_predicate(&check.predicate);
                let guard_register = ssa.fresh();
                ops.push(CudaOp::Check {
                    node: node.clone(),
                    index,
                    kind,
                    values,
                    extents,
                    status,
                    guard: guard_register,
                });
                guard = Some(guard_register);
            }
        }
    }
    for blueprint in blueprint_ops.iter() {
        ops.extend(rebind_op(
            blueprint, &logical, builder, ctx, flags, &mut ssa, &mut remap,
        )?);
    }
    if guard.is_some() {
        set_guard(&mut ops, guard);
    }
    let iteration = match &formed.iteration {
        Some(own) => Ok(own
            .clone()
            .with_participants(Sym::constant(UNIVERSAL_PARTICIPANTS))),
        None => ctx.launch_map(),
    }?;
    builder
        .map_primitive(node.clone(), iteration, ops_of(ops))
        .map_err(|error| format!("CUDA mapping of node#{}: {error}", node.node.0))
}

/// The logical node behind one node reference.
fn node_node(
    builder: &executable::AlternativeBuilder<CudaDialect>,
    node: &executable::NodeRef,
) -> LogicalNode {
    let graph = builder.graph();
    let mut region = graph.root.clone();
    for step in &node.region {
        let candidate = region
            .nodes
            .get(step.node())
            .expect("the region path names a real node");
        region = match (&candidate.kind, step) {
            (LogicalNodeKind::If(if_node), executable::RegionStep::IfThen(_)) => {
                if_node.then_region.clone()
            }
            (LogicalNodeKind::If(if_node), executable::RegionStep::IfElse(_)) => {
                if_node.else_region.clone()
            }
            (LogicalNodeKind::Loop(loop_node), executable::RegionStep::LoopBody(_)) => {
                loop_node.body.clone()
            }
            _ => panic!("the region path disagrees with the graph structure"),
        };
    }
    region
        .nodes
        .get(node.node)
        .cloned()
        .expect("the node reference names a real node")
}

/// Decompose one planned predicate into its check kind, watched values,
/// and extents.
fn decompose_predicate(
    predicate: &CheckPredicate,
) -> (CheckKind, Vec<GraphValueId>, Vec<ExtentExpr>) {
    match predicate {
        CheckPredicate::IndexInBounds { index, extent } => {
            (CheckKind::IndexInBounds, vec![*index], vec![extent.clone()])
        }
        CheckPredicate::RangeInBounds { start, end, extent } => (
            CheckKind::RangeInBounds,
            vec![*start, *end],
            vec![extent.clone()],
        ),
        CheckPredicate::DivisorNonZero { value } => {
            (CheckKind::DivisionByZero, vec![*value], Vec::new())
        }
        CheckPredicate::DivisionSafe { lhs, rhs } => {
            (CheckKind::DivisionOverflow, vec![*lhs, *rhs], Vec::new())
        }
        CheckPredicate::ShiftInRange { value } => {
            (CheckKind::ShiftOutOfRange, vec![*value], Vec::new())
        }
        CheckPredicate::ProductFits { factors, bits } => {
            let _ = bits;
            (CheckKind::ShapeOverflow, Vec::new(), factors.clone())
        }
        CheckPredicate::ExtentPositive { extent } => (
            CheckKind::EmptyReductionInput,
            Vec::new(),
            vec![extent.clone()],
        ),
    }
}

/// Attach one guard to every guardable operation of the node (division,
/// shift, checked access, atomic); pure computation is skipped along with
/// its guarded effect when a check fails, and the first error is recorded.
fn set_guard(ops: &mut [CudaOp], guard: Option<CudaSsa>) {
    for op in ops.iter_mut() {
        match op {
            CudaOp::Binary { guard: slot, .. }
            | CudaOp::ElementRead { guard: slot, .. }
            | CudaOp::PackedElementRead { guard: slot, .. }
            | CudaOp::ElementWrite { guard: slot, .. }
            | CudaOp::Atomic { guard: slot, .. } => {
                if slot.is_none() {
                    *slot = guard;
                }
            }
            _ => {}
        }
    }
}

/// One fresh register per placeholder id, stable within the opcode list.
fn fresh(id: &CudaSsa, ssa: &mut SsaAllocator, remap: &mut BTreeMap<u32, CudaSsa>) -> CudaSsa {
    *remap.entry(id.0).or_insert_with(|| ssa.fresh())
}

/// The representation name of a packed operand (empty when dense).
fn packed_repr_name(inputs: &[ValueType]) -> String {
    inputs
        .first()
        .and_then(|ty| match ty {
            ValueType::Tensor(tensor) => match &tensor.elem {
                Elem::Repr(name) => Some(name.clone()),
                _ => None,
            },
            _ => None,
        })
        .unwrap_or_default()
}

/// Rewrite one legalized opcode: positional input placeholders become the
/// node's real input values (or an absorbed binder's register), positional
/// output placeholders become concrete storages and scalar destinations,
/// and view transforms are restored from the graph's view table.
#[allow(clippy::too_many_arguments)]
fn rebind_op(
    op: &CudaOp,
    node: &LogicalNode,
    builder: &executable::AlternativeBuilder<CudaDialect>,
    ctx: &GraphContext,
    flags: StrategyFlags,
    ssa: &mut SsaAllocator,
    remap: &mut BTreeMap<u32, CudaSsa>,
) -> Result<Vec<CudaOp>, BuilderError> {
    use crate::physical::CudaOperand;
    // Resolve one operand placeholder to a real value reference or an
    // absorbed binder's register.
    fn resolve_operand(
        operand: &CudaOperand,
        node: &LogicalNode,
        ctx: &GraphContext,
        ssa: &mut SsaAllocator,
        remap: &mut BTreeMap<u32, CudaSsa>,
    ) -> Result<CudaOperand, BuilderError> {
        match operand {
            CudaOperand::Ssa(id) => Ok(CudaOperand::Ssa(
                *remap.entry(id.0).or_insert_with(|| ssa.fresh()),
            )),
            CudaOperand::Value(value) => {
                if let Some(index) = placeholder_index(*value) {
                    let real = node.inputs.get(index).copied().ok_or_else(|| {
                        BuilderError::from(format!("legalization references absent input #{index}"))
                    })?;
                    Ok(bind_value(real, ctx, ssa, remap))
                } else {
                    Ok(bind_value(*value, ctx, ssa, remap))
                }
            }
        }
    }
    // Bind one real value: an absorbed binder uses its register; otherwise
    // the value stays (the emitter resolves it through its transport).
    fn bind_value(
        value: GraphValueId,
        ctx: &GraphContext,
        ssa: &mut SsaAllocator,
        remap: &mut BTreeMap<u32, CudaSsa>,
    ) -> CudaOperand {
        if let Some((register, _)) = ctx.binder_kinds.get(&value) {
            let _ = (ssa, remap);
            CudaOperand::Ssa(*register)
        } else {
            CudaOperand::Value(value)
        }
    }
    // The concrete storage template behind one value.
    fn storage_ref_of(
        value: GraphValueId,
        _ctx: &GraphContext,
        builder: &executable::AlternativeBuilder<CudaDialect>,
    ) -> Result<CudaStorageRef, BuilderError> {
        match builder.transport_of(value)? {
            TransportTemplate::Storage(_) | TransportTemplate::Boundary(_) => {
                Ok(CudaStorageRef::Binding(value))
            }
            _ => Err(format!(
                "tensor value#{} does not have a storage transport",
                value.0
            )),
        }
    }
    // The view transform of one tensor value.
    let view_of = |value: GraphValueId| -> ViewTransform {
        ctx.value_views
            .get(&value)
            .cloned()
            .unwrap_or(ViewTransform::Identity)
    };
    let input = |index: usize| -> GraphValueId {
        node.inputs
            .get(index)
            .copied()
            .unwrap_or_else(|| panic!("legalization references absent input #{index}"))
    };
    let output_value = |index: u32| -> GraphValueId {
        node.outputs
            .get(index as usize)
            .map(|output| output.id)
            .unwrap_or_else(|| panic!("legalization references absent output #{index}"))
    };
    Ok(match op {
        CudaOp::NoOp => vec![CudaOp::NoOp],
        CudaOp::Const { dest, value, dtype } => vec![CudaOp::Const {
            value: value.clone(),
            dtype: *dtype,
            dest: fresh(dest, ssa, remap),
        }],
        CudaOp::ExtentValue { dest, extent } => vec![CudaOp::ExtentValue {
            extent: *extent,
            dest: fresh(dest, ssa, remap),
        }],
        CudaOp::AxisCoordinate { axis, dest } => vec![CudaOp::AxisCoordinate {
            axis: *axis,
            dest: fresh(dest, ssa, remap),
        }],
        CudaOp::SerialFor {
            binder,
            length,
            body,
        } => {
            let binder_register = fresh(binder, ssa, remap);
            let mut rewritten = Vec::new();
            for nested in body {
                rewritten.extend(rebind_op(nested, node, builder, ctx, flags, ssa, remap)?);
            }
            vec![CudaOp::SerialFor {
                binder: binder_register,
                length: length.clone(),
                body: rewritten,
            }]
        }
        CudaOp::Unary {
            op: kind,
            source,
            dest,
        } => vec![CudaOp::Unary {
            op: *kind,
            source: resolve_operand(source, node, ctx, ssa, remap)?,
            dest: fresh(dest, ssa, remap),
        }],
        CudaOp::Binary {
            op: kind,
            lhs,
            rhs,
            dest,
            dtype,
            ..
        } => vec![CudaOp::Binary {
            op: *kind,
            lhs: resolve_operand(lhs, node, ctx, ssa, remap)?,
            rhs: resolve_operand(rhs, node, ctx, ssa, remap)?,
            dest: fresh(dest, ssa, remap),
            dtype: *dtype,
            guard: None,
        }],
        CudaOp::Fma {
            a,
            b,
            c,
            dest,
            dtype,
        } => vec![CudaOp::Fma {
            a: resolve_operand(a, node, ctx, ssa, remap)?,
            b: resolve_operand(b, node, ctx, ssa, remap)?,
            c: resolve_operand(c, node, ctx, ssa, remap)?,
            dest: fresh(dest, ssa, remap),
            dtype: *dtype,
        }],
        CudaOp::Cast {
            source,
            dest,
            source_dtype,
            target_dtype,
        } => vec![CudaOp::Cast {
            source: resolve_operand(source, node, ctx, ssa, remap)?,
            dest: fresh(dest, ssa, remap),
            source_dtype: *source_dtype,
            target_dtype: *target_dtype,
        }],
        CudaOp::Math {
            op: kind,
            arguments,
            dest,
            mode,
        } => vec![CudaOp::Math {
            op: *kind,
            arguments: arguments
                .iter()
                .map(|argument| resolve_operand(argument, node, ctx, ssa, remap))
                .collect::<Result<Vec<_>, _>>()?,
            dest: fresh(dest, ssa, remap),
            mode: if flags.approx_exp_fast && *kind == MathOp::ExpFast {
                MathMode::FastApprox
            } else {
                *mode
            },
        }],
        CudaOp::Select {
            condition,
            then_value,
            else_value,
            dest,
            dtype,
        } => {
            vec![CudaOp::Select {
                condition: resolve_operand(condition, node, ctx, ssa, remap)?,
                then_value: resolve_operand(then_value, node, ctx, ssa, remap)?,
                else_value: resolve_operand(else_value, node, ctx, ssa, remap)?,
                dest: fresh(dest, ssa, remap),
                dtype: *dtype,
            }]
        }
        CudaOp::TuplePack { parts, dest } => vec![CudaOp::TuplePack {
            parts: parts
                .iter()
                .map(|part| resolve_operand(part, node, ctx, ssa, remap))
                .collect::<Result<Vec<_>, _>>()?,
            dest: fresh(dest, ssa, remap),
        }],
        CudaOp::TupleGet {
            source,
            index,
            dest,
        } => vec![CudaOp::TupleGet {
            source: resolve_operand(source, node, ctx, ssa, remap)?,
            index: *index,
            dest: fresh(dest, ssa, remap),
        }],
        CudaOp::RangeMake { start, end, dest } => vec![CudaOp::RangeMake {
            start: resolve_operand(start, node, ctx, ssa, remap)?,
            end: resolve_operand(end, node, ctx, ssa, remap)?,
            dest: fresh(dest, ssa, remap),
        }],
        CudaOp::RangeStart { source, dest } => vec![CudaOp::RangeStart {
            source: resolve_operand(source, node, ctx, ssa, remap)?,
            dest: fresh(dest, ssa, remap),
        }],
        CudaOp::RangeEnd { source, dest } => vec![CudaOp::RangeEnd {
            source: resolve_operand(source, node, ctx, ssa, remap)?,
            dest: fresh(dest, ssa, remap),
        }],
        CudaOp::ExtentOf {
            base,
            axis,
            dest,
            valid,
        } => vec![CudaOp::ExtentOf {
            base: resolve_operand(base, node, ctx, ssa, remap)?,
            axis: *axis,
            dest: fresh(dest, ssa, remap),
            valid: *valid,
        }],
        CudaOp::ElementRead {
            base,
            view_shape,
            indices,
            dest,
            dtype,
            ..
        } => {
            let base = resolve_operand(base, node, ctx, ssa, remap)?;
            let base_value = match base {
                CudaOperand::Value(value) => value,
                _ => {
                    return Err(BuilderError::from(
                        "an element read base must be a bound value".to_string(),
                    ));
                }
            };
            vec![CudaOp::ElementRead {
                base,
                view: CudaView::of(&view_of(base_value)),
                view_shape: view_shape.clone(),
                indices: indices
                    .iter()
                    .map(|index| resolve_operand(index, node, ctx, ssa, remap))
                    .collect::<Result<Vec<_>, _>>()?,
                dest: fresh(dest, ssa, remap),
                dtype: *dtype,
                guard: None,
            }]
        }
        CudaOp::PackedElementRead {
            source,
            view_shape,
            indices,
            repr,
            dest,
            ..
        } => {
            let source_value = match source {
                CudaStorageRef::Input(index) => input(*index as usize),
                CudaStorageRef::Output(index) => output_value(*index),
                CudaStorageRef::Template(_)
                | CudaStorageRef::Resolved(_)
                | CudaStorageRef::Binding(_) => {
                    return Err(BuilderError::from(
                        "a concrete packed read source appeared".to_string(),
                    ));
                }
            };
            vec![CudaOp::PackedElementRead {
                source: storage_ref_of(source_value, ctx, builder)?,
                view: CudaView::of(&view_of(source_value)),
                view_shape: view_shape.clone(),
                indices: indices
                    .iter()
                    .map(|index| resolve_operand(index, node, ctx, ssa, remap))
                    .collect::<Result<Vec<_>, _>>()?,
                repr: repr.clone(),
                dest: fresh(dest, ssa, remap),
                guard: None,
            }]
        }
        CudaOp::ElementWrite {
            base,
            view_shape,
            indices,
            value,
            dtype,
            ..
        } => {
            let base = resolve_operand(base, node, ctx, ssa, remap)?;
            let base_value = match base {
                CudaOperand::Value(value) => value,
                _ => {
                    return Err(BuilderError::from(
                        "an element write base must be a bound value".to_string(),
                    ));
                }
            };
            vec![CudaOp::ElementWrite {
                base,
                view: CudaView::of(&view_of(base_value)),
                view_shape: view_shape.clone(),
                indices: indices
                    .iter()
                    .map(|index| resolve_operand(index, node, ctx, ssa, remap))
                    .collect::<Result<Vec<_>, _>>()?,
                value: resolve_operand(value, node, ctx, ssa, remap)?,
                dtype: *dtype,
                guard: None,
            }]
        }
        CudaOp::Atomic {
            op,
            base,
            view_shape,
            indices,
            value,
            dtype,
            mode,
            ..
        } => {
            let base = resolve_operand(base, node, ctx, ssa, remap)?;
            let base_value = match base {
                CudaOperand::Value(value) => value,
                _ => {
                    return Err(BuilderError::from(
                        "an atomic base must be a bound value".to_string(),
                    ));
                }
            };
            // The CAS word loop is the `add` strategy only; `max`/`min` stay
            // serialized.
            let mode = if flags.cas_atomic
                && *op == seismic_lang::intrinsics::AtomicOp::Add
                && cas_admissible(view_shape, *dtype)
            {
                AtomicMode::CasWord
            } else {
                *mode
            };
            vec![CudaOp::Atomic {
                op: *op,
                base,
                view: CudaView::of(&view_of(base_value)),
                view_shape: view_shape.clone(),
                indices: indices
                    .iter()
                    .map(|index| resolve_operand(index, node, ctx, ssa, remap))
                    .collect::<Result<Vec<_>, _>>()?,
                value: resolve_operand(value, node, ctx, ssa, remap)?,
                dtype: *dtype,
                mode,
                guard: None,
            }]
        }
        CudaOp::Fill {
            dest: _,
            view: _,
            view_shape,
            dtype,
            value,
        } => {
            let dest_value = output_value(0);
            vec![CudaOp::Fill {
                dest: storage_ref_of(dest_value, ctx, builder)?,
                view: CudaView::of(&view_of(dest_value)),
                view_shape: view_shape.clone(),
                dtype: *dtype,
                value: resolve_operand(value, node, ctx, ssa, remap)?,
            }]
        }
        CudaOp::CopyElements {
            source,
            source_view: _,
            dest,
            dest_view: _,
            view_shape,
            dtype,
        } => {
            let source_value = match source {
                CudaStorageRef::Input(index) => input(*index as usize),
                CudaStorageRef::Output(index) => output_value(*index),
                CudaStorageRef::Template(_)
                | CudaStorageRef::Resolved(_)
                | CudaStorageRef::Binding(_) => {
                    return Err(BuilderError::from(
                        "a concrete copy source appeared".to_string(),
                    ));
                }
            };
            let dest_value = match dest {
                CudaStorageRef::Input(index) => input(*index as usize),
                CudaStorageRef::Output(index) => output_value(*index),
                CudaStorageRef::Template(_)
                | CudaStorageRef::Resolved(_)
                | CudaStorageRef::Binding(_) => {
                    return Err(BuilderError::from(
                        "a concrete copy destination appeared".to_string(),
                    ));
                }
            };
            vec![CudaOp::CopyElements {
                source: storage_ref_of(source_value, ctx, builder)?,
                source_view: CudaView::of(&view_of(source_value)),
                dest: storage_ref_of(dest_value, ctx, builder)?,
                dest_view: CudaView::of(&view_of(dest_value)),
                view_shape: view_shape.clone(),
                dtype: *dtype,
            }]
        }
        CudaOp::Decode {
            source,
            source_view: _,
            dest,
            dest_view: _,
            view_shape,
            repr,
        } => {
            let source_value = match source {
                CudaStorageRef::Input(index) => input(*index as usize),
                CudaStorageRef::Output(index) => output_value(*index),
                CudaStorageRef::Template(_)
                | CudaStorageRef::Resolved(_)
                | CudaStorageRef::Binding(_) => {
                    return Err(BuilderError::from(
                        "a concrete decode source appeared".to_string(),
                    ));
                }
            };
            let dest_value = match dest {
                CudaStorageRef::Output(index) => output_value(*index),
                CudaStorageRef::Input(index) => input(*index as usize),
                CudaStorageRef::Template(_)
                | CudaStorageRef::Resolved(_)
                | CudaStorageRef::Binding(_) => {
                    return Err(BuilderError::from(
                        "a concrete decode destination appeared".to_string(),
                    ));
                }
            };
            vec![CudaOp::Decode {
                source: storage_ref_of(source_value, ctx, builder)?,
                source_view: CudaView::of(&view_of(source_value)),
                dest: storage_ref_of(dest_value, ctx, builder)?,
                dest_view: CudaView::of(&view_of(dest_value)),
                view_shape: view_shape.clone(),
                repr: repr.clone(),
            }]
        }
        CudaOp::PackedPlaneRead {
            source,
            view: _,
            plane,
            view_shape,
            repr,
            dest,
        } => {
            let source_value = match source {
                CudaStorageRef::Input(index) => input(*index as usize),
                CudaStorageRef::Template(_)
                | CudaStorageRef::Resolved(_)
                | CudaStorageRef::Binding(_) => {
                    return Err(BuilderError::from(
                        "a concrete plane source appeared".to_string(),
                    ));
                }
                CudaStorageRef::Output(_) => output_value(0),
            };
            vec![CudaOp::PackedPlaneRead {
                source: storage_ref_of(source_value, ctx, builder)?,
                view: CudaView::of(&view_of(source_value)),
                plane: *plane,
                view_shape: view_shape.clone(),
                repr: repr.clone(),
                dest: fresh(dest, ssa, remap),
            }]
        }
        CudaOp::StoreScalar {
            dest,
            source,
            dtype,
        } => {
            let output = node
                .outputs
                .get(0)
                .ok_or_else(|| BuilderError::from("a store has no output".to_string()))?;
            let transport = builder.transport_of(output.id)?;
            let stores = flatten_scalar_dest(&transport, output.ty.clone(), dest.clone());
            let source = resolve_operand(source, node, ctx, ssa, remap)?;
            stores
                .into_iter()
                .map(|dest| -> Result<CudaOp, BuilderError> {
                    Ok(CudaOp::StoreScalar {
                        dest,
                        source: source.clone(),
                        dtype: *dtype,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        }
        CudaOp::Check { .. } => vec![op.clone()],
        CudaOp::SerialFold { .. } => vec![op.clone()],
        CudaOp::LaneIndex { dest } => vec![CudaOp::LaneIndex {
            dest: fresh(dest, ssa, remap),
        }],
        CudaOp::Shuffle {
            value,
            index,
            dest,
            dtype,
        } => vec![CudaOp::Shuffle {
            value: resolve_operand(value, node, ctx, ssa, remap)?,
            index: resolve_operand(index, node, ctx, ssa, remap)?,
            dest: fresh(dest, ssa, remap),
            dtype: *dtype,
        }],
        CudaOp::SubgroupReduce {
            op: reduce,
            value,
            dest,
            dtype,
        } => vec![CudaOp::SubgroupReduce {
            op: *reduce,
            value: resolve_operand(value, node, ctx, ssa, remap)?,
            dest: fresh(dest, ssa, remap),
            dtype: *dtype,
        }],
    })
}

/// Flatten one scalar output transport into per-leaf destinations.
fn flatten_scalar_dest(
    transport: &TransportTemplate,
    ty: ValueType,
    placeholder: CudaScalarDest,
) -> Vec<CudaScalarDest> {
    match transport {
        TransportTemplate::ExecutorScalar(executable::ExecutorScalarTemplate { source, dtype }) => {
            match source {
                executable::ExecutorScalarSource::Slot(slot) => {
                    vec![CudaScalarDest::Slot(*slot)]
                }
                executable::ExecutorScalarSource::Abi { leaf, endpoint } => {
                    vec![CudaScalarDest::Abi {
                        path: match leaf {
                            executable::BoundaryLeaf::Input { leaf, .. }
                            | executable::BoundaryLeaf::Result { leaf } => leaf.clone(),
                        },
                        endpoint: *endpoint,
                        dtype: *dtype,
                    }]
                }
                // Computed control scalars are never store destinations.
                executable::ExecutorScalarSource::Computed(_) => Vec::new(),
            }
        }
        TransportTemplate::Tuple(items) => {
            let fields = match ty {
                ValueType::Tuple(items) => items.into_vec(),
                ValueType::Range { .. } => vec![ValueType::Scalar(DType::I32); 2],
                _ => Vec::new(),
            };
            items
                .iter()
                .zip(fields)
                .flat_map(|(item, field)| flatten_scalar_dest(item, field, placeholder.clone()))
                .collect()
        }
        _ => Vec::new(),
    }
}

// --- reductions -----------------------------------------------------------

/// Lower one reduction node with the universal parallel-outer strategy:
/// parallel outer coordinates, one logical participant per output, an
/// in-kernel serial ascending fold with the registry accumulator,
/// identity, and tie semantics, and exactly one publication.
fn lower_reduction(
    builder: &mut executable::AlternativeBuilder<CudaDialect>,
    node: &executable::NodeRef,
    ctx: &GraphContext,
) -> Result<(), BuilderError> {
    let logical = node_node(builder, node);
    let formed = universal_node(node.node, &logical, &facts_of(builder, ctx))
        .map_err(|error| format!("CUDA reduction formation of node#{}: {error}", node.node.0))?;
    let seismic_compiler::terminal::UniversalNode::Reduction(universal) = formed else {
        return Err(BuilderError::from(
            "lower_reduction requires a reduction node".to_string(),
        ));
    };
    let reduction = match &logical.kind {
        LogicalNodeKind::Reduction(reduction) => reduction.clone(),
        _ => {
            return Err(BuilderError::from(
                "lower_reduction requires a reduction node".to_string(),
            ));
        }
    };
    let operand_value = reduction.operand;
    let operand_storage = ctx
        .value_storages
        .get(&operand_value)
        .copied()
        .ok_or_else(|| {
            BuilderError::from(format!(
                "the reduction operand value#{} has no backing storage",
                operand_value.0
            ))
        })?;
    let operand_template = builder.storage_of(operand_storage).ok_or_else(|| {
        BuilderError::from(format!(
            "the reduction storage#{} has no physical template",
            operand_storage.0
        ))
    })?;
    let view = ctx
        .value_views
        .get(&operand_value)
        .cloned()
        .unwrap_or(ViewTransform::Identity);
    let view_shape = facts_of(builder, ctx)
        .types
        .get(&operand_value)
        .and_then(|ty| ty.shaped().map(|shape| shape.axes.clone()))
        .unwrap_or_default();
    let result_value = logical
        .outputs
        .first()
        .map(|output| output.id)
        .ok_or_else(|| BuilderError::from("a reduction has no result".to_string()))?;
    let result_transport = builder.transport_of(result_value)?;
    let dest = match &result_transport {
        TransportTemplate::ExecutorScalar(executable::ExecutorScalarTemplate { source, dtype }) => {
            match source {
                executable::ExecutorScalarSource::Slot(slot) => CudaScalarDest::Slot(*slot),
                executable::ExecutorScalarSource::Abi { leaf, endpoint } => CudaScalarDest::Abi {
                    path: match leaf {
                        executable::BoundaryLeaf::Input { leaf, .. }
                        | executable::BoundaryLeaf::Result { leaf } => leaf.clone(),
                    },
                    endpoint: *endpoint,
                    dtype: *dtype,
                },
                executable::ExecutorScalarSource::Computed(_) => {
                    return Err(BuilderError::from(
                        "a reduction result cannot transport through a computed scalar".to_string(),
                    ));
                }
            }
        }
        _ => {
            return Err(BuilderError::from(
                "a reduction result must transport through an executor scalar".to_string(),
            ));
        }
    };
    // The outer coordinates are the launch's traversal over the operand's
    // axes minus the reduced axis.
    let mut outer_axes = view_shape.clone();
    let length = outer_axes.remove(reduction.axis);
    let iteration = if outer_axes.is_empty() {
        LinearIterationMap::serial()
    } else {
        LinearIterationMap::linear(&outer_axes, &ctx.runtime_extents)
            .map(|map| map.with_participants(Sym::constant(UNIVERSAL_PARTICIPANTS)))
            .map_err(|error| {
                BuilderError::from(format!("infeasible CUDA reduction geometry: {error}"))
            })?
    };
    // Identity-less folds declare one status field for the empty-axis
    // precondition; the fold op writes it when the axis is empty.
    let nonempty = matches!(
        universal.strategy.identity,
        seismic_compiler::terminal::ReductionIdentity::FirstElement
            | seismic_compiler::terminal::ReductionIdentity::FirstElementNonEmpty
    )
    .then(|| builder.status_field());
    let fold = CudaOp::SerialFold {
        operand: CudaStorageRef::Template(operand_template),
        view: CudaView::of(&view),
        view_shape,
        axis: reduction.axis,
        length,
        op: reduction.op,
        accumulator: reduction.accumulator,
        dest,
        nonempty,
    };
    // Statically impossible preconditions abort the alternative here.
    for (_, discharge) in &universal.preconditions {
        if let ObligationDischarge::StaticallyImpossible(reason) = discharge {
            return Err(BuilderError::from(format!(
                "the reduction at node#{} is statically impossible: {reason}",
                node.node.0
            )));
        }
    }
    builder
        .map_reduction(
            node.clone(),
            executable::ReductionStrategyTemplate {
                topology: universal.strategy.topology.clone(),
                iteration,
                ops: ops_of(vec![fold]),
                result: result_transport,
            },
        )
        .map_err(|error| format!("CUDA reduction mapping of node#{}: {error}", node.node.0))
}
