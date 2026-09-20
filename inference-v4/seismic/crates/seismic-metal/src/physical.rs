//! Metal executable planning: the sealed `MetalOp` opcode enum, the
//! `MetalDialect` legalization/consequence contract over the portable
//! primitive matrix, and the universal and capability strategies that consume
//! logical alternatives through the common `PlanFamilyBuilder`.
//!
//! Logical task graphs are consumed exactly once, here: every primitive,
//! reduction, loop, conditional, and call occurrence is routed through its
//! family-builder transition. The dialect offers subgroup capability opcodes
//! only behind exact effective profile signatures; `metal.matrix` fragments
//! are physical opcodes that stay unregistered until emission is complete.
//!
//! OpCodes carry launch-internal references: SSA results of earlier opcodes
//! in the same mapped block, operand positions into the launch's value
//! bindings, or the produced (output) value's own graph id, resolved through
//! the launch's value bindings. The MSL printer accepts `MetalOp`
//! exhaustively and adds, omits, and decides nothing.

use seismic_compiler::strategies::{self, CostModelId};
use seismic_compiler::terminal::{
    self, discharge_with_builder, form_primitive, reduction, FormedPrimitive, GraphFacts,
    LinearIterationMap, LinearLoopOp, ObligationDischarge, SeismicMathReference,
    UniversalLegalization, SEISMIC_MATH,
};
use seismic_lang::{
    intrinsics::{
        atomic_dtype, AtomicOp, CapabilityId, IntrinsicId, MathOp, PlaneField, PrimitiveId,
        ReduceOp,
    },
    logical::{
        self, Access, ChoiceId, GraphRegion, GraphValueId, IdVec, IfNode, JoinSlot, LogicalNode,
        LogicalNodeKind, LogicalProgram, LogicalStorageId, LoopNode, NodeId, PrimitiveOp,
        ReductionNode, ReductionOrder, RegionResult, RuntimeExtent, SafetyObligation, StateTokenId,
        TaskGraph,
    },
    sir::{Literal, LoopKind},
    sym::Sym,
    types::{DType, Elem, ExtentExpr, NonEmpty, RuntimeExtentId, TensorType, ValueType},
};
use seismic_realization::executable::{
    self as realization, AlternativeBuilder, BuilderError, EffectiveTargetProfile,
    ExecutorRangeTemplate, FusedStrategyTemplate, HardResources, Legalized, NativeResourceContract,
    NodeRef, ObligationRef, PhysicalConsequences, PhysicalJoinTemplate, PlanFamily,
    PlanFamilyBuilder, ReductionStrategyTemplate, RegionPath, RegionStep, StorageViewTemplate,
    TransportTemplate,
};
use std::collections::BTreeMap;

/// Sealed marker: the dialect is closed over this crate.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MetalDialect;

impl seismic_realization::executable::sealed::Sealed for MetalDialect {}

// ---------------------------------------------------------------------------
// Layouts
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetalLayout {
    Dense {
        dtype: DType,
        elements: Sym,
    },
    PackedPlane {
        representation: String,
        plane: String,
        dtype: DType,
        elements: Sym,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedMetalLayout {
    Dense {
        dtype: DType,
        elements: u64,
    },
    PackedPlane {
        representation: String,
        plane: String,
        dtype: DType,
        elements: u64,
    },
}

// ---------------------------------------------------------------------------
// Sealed opcodes
// ---------------------------------------------------------------------------

/// One runtime-evaluable address/extent factor: a constant, a runtime extent
/// value, or a product.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AddrExpr {
    Const(u64),
    Extent(RuntimeExtentId),
    Mul(Box<AddrExpr>, Box<AddrExpr>),
}

/// Reference to one scalar inside a launch: an SSA result of an earlier
/// opcode in the same mapped block, or a position in the launch's input
/// bindings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueRef {
    Ssa(u32),
    Operand(usize),
    /// An iteration axis coordinate of the launch (an absorbed independent
    /// loop's binder).
    Axis(usize),
}

/// One write destination: an input binding (a place view), or a produced
/// (output) value, resolved through the launch's value bindings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dst {
    Operand(usize),
    Produced(GraphValueId),
}

/// One index term of an element access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexRef {
    Value(ValueRef),
    /// The coordinate of logical axis `n` of the current iteration.
    Axis(usize),
    /// The reduced-axis variable of a serial fold.
    ReducedAxis,
}

/// Row-major address layout of one tensor view access.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccessLayout {
    pub strides: Vec<AddrExpr>,
    /// Point-slice offset terms (stride × value).
    pub offset: Vec<(AddrExpr, ValueRef)>,
}

/// One scalar operand of an opcode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Src {
    Scalar(ValueRef),
    Element {
        operand: usize,
        layout: AccessLayout,
        indices: Vec<IndexRef>,
    },
}

/// Scalar constants carried as exact bits (`Float` holds IEEE-754 bit
/// patterns) so opcodes stay `Eq`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstValue {
    Int(i64),
    Float(u64),
    Bool(bool),
}

/// One planned runtime guard: the conjunctive predicate and the status field
/// the first failing check writes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Guard {
    pub predicates: Vec<GuardPredicate>,
    pub status: u64,
}

/// Strategy-level guard predicates: value references are launch-local.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GuardPredicate {
    IndexInBounds {
        index: ValueRef,
        extent: AddrExpr,
    },
    RangeInBounds {
        start: ValueRef,
        end: ValueRef,
        extent: AddrExpr,
    },
    DivisorNonZero {
        value: ValueRef,
    },
    DivisionSafe {
        lhs: ValueRef,
        rhs: ValueRef,
    },
    ShiftInRange {
        value: ValueRef,
    },
    ProductFits {
        factors: Vec<AddrExpr>,
        bits: u8,
    },
    ExtentPositive {
        extent: AddrExpr,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoolOp {
    And,
    Or,
    Not,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntOp {
    Add,
    Sub,
    Mul,
    BitAnd,
    BitOr,
    BitXor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DivRemOp {
    Div,
    Rem,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShiftOp {
    Shl,
    Shr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    BitNot,
    Not,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatArith {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

/// Where a reduction publishes its result: a produced scalar value (its
/// executor slot), or one element of a produced tensor per output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReduceResult {
    Scalar {
        value: GraphValueId,
    },
    Tensor {
        dst: Dst,
        layout: AccessLayout,
        indices: Vec<IndexRef>,
        dtype: DType,
    },
}

/// The closed Metal opcode enum: the universal primitive forms plus the
/// metal.subgroup capability opcodes. Emitted exhaustively; never rejected
/// once selected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetalOp {
    Const {
        into: ValueRef,
        value: ConstValue,
        dtype: DType,
    },
    /// The retained runtime value of one extent (a runtime extent id, or a
    /// computed shape-axis extent).
    ExtentOf {
        value: AddrExpr,
        into: ValueRef,
    },
    Select {
        condition: ValueRef,
        then: Src,
        otherwise: Src,
        into: ValueRef,
        dtype: DType,
    },
    Compare {
        op: RelOp,
        lhs: Src,
        rhs: Src,
        into: ValueRef,
        dtype: DType,
    },
    BoolLogic {
        op: BoolOp,
        operands: Vec<ValueRef>,
        into: ValueRef,
    },
    IntOp {
        op: IntOp,
        lhs: Src,
        rhs: Src,
        into: ValueRef,
        dtype: DType,
    },
    IntDivRem {
        op: DivRemOp,
        lhs: Src,
        rhs: Src,
        into: ValueRef,
        dtype: DType,
        guard: Option<Guard>,
    },
    Shift {
        op: ShiftOp,
        value: Src,
        amount: Src,
        into: ValueRef,
        dtype: DType,
        guard: Option<Guard>,
    },
    Unary {
        op: UnaryOp,
        operand: Src,
        into: ValueRef,
        dtype: DType,
    },
    FloatOp {
        op: FloatArith,
        lhs: Src,
        rhs: Src,
        into: ValueRef,
        dtype: DType,
    },
    Fma {
        a: Src,
        b: Src,
        c: Src,
        into: ValueRef,
        dtype: DType,
    },
    Math {
        op: MathOp,
        args: Vec<Src>,
        into: ValueRef,
        dtype: DType,
        reference: SeismicMathReference,
    },
    Cast {
        from: DType,
        to: DType,
        operand: Src,
        into: ValueRef,
    },
    RangeEndpoint {
        start: bool,
        operand: usize,
        into: ValueRef,
    },
    TupleGet {
        operand: usize,
        index: usize,
        into: ValueRef,
    },
    /// LayoutAddress: pure view metadata (the address layouts of accesses
    /// through the view are carried by those accesses). Emits nothing.
    View,
    /// Uninitialized dense storage: declared by the plan, never emitted.
    Alloc,
    ReadElement {
        operand: usize,
        layout: AccessLayout,
        indices: Vec<IndexRef>,
        dtype: DType,
        into: ValueRef,
        guard: Option<Guard>,
    },
    WriteElement {
        operand: usize,
        layout: AccessLayout,
        indices: Vec<IndexRef>,
        value: Src,
        dtype: DType,
        guard: Option<Guard>,
    },
    /// Store one computed scalar into a produced tensor element.
    StoreResult {
        value: ValueRef,
        dst: Dst,
        layout: AccessLayout,
        indices: Vec<IndexRef>,
        dtype: DType,
    },
    LinearLoop {
        op: LinearLoopOp,
        source: Option<(usize, AccessLayout)>,
        dst: Dst,
        dst_layout: AccessLayout,
        dtype: DType,
        fill: Option<ConstValue>,
    },
    PackedPlaneRead {
        operand: usize,
        plane: usize,
        dtype: DType,
        into: ValueRef,
    },
    /// One decoded element of a packed view, addressed through the
    /// representation planes.
    PackedElementRead {
        operand: usize,
        layout: AccessLayout,
        indices: Vec<IndexRef>,
        repr: String,
        into: ValueRef,
        guard: Option<Guard>,
    },
    /// In-kernel serial loop (an ordered loop absorbed by fusion).
    SerialFor {
        binder: ValueRef,
        length: AddrExpr,
        body: Vec<MetalOp>,
    },
    /// In-kernel conditional (an `if` absorbed by fusion).
    Branch {
        condition: ValueRef,
        then_ops: Vec<MetalOp>,
        else_ops: Vec<MetalOp>,
    },
    /// Logical matrix multiplication through the `metal.matrix` capability:
    /// rank-two operands, launch-mapped over the output elements, one
    /// ascending-k fma chain per participant. The fragment/staging strategy
    /// is later optimization work; this is the exact fma chain the
    /// reference body defines.
    MatrixMatmul {
        left: (usize, AccessLayout),
        right: (usize, AccessLayout),
        into: (usize, AccessLayout),
        accumulate: bool,
        k: AddrExpr,
        dtype: DType,
        signature: IntrinsicId,
        signature_arguments: Vec<ValueType>,
    },
    /// Packed decode: emission not implemented; reaching it is a CompilerBug.
    PackedDecode {
        operand: usize,
    },
    /// Serialized exact atomic update (load / combine / round / store) of one
    /// element; the containing independent domain runs on one participant.
    AtomicSerial {
        op: AtomicOp,
        operand: usize,
        layout: AccessLayout,
        indices: Vec<IndexRef>,
        value: Src,
        dtype: DType,
        guard: Option<Guard>,
    },
    /// Concurrent atomic update of one 32-bit element from the ordinary
    /// participant domain: native `atomic_fetch_{add,max,min}` for integers
    /// and float `max`/`min` via a compare/exchange on the bits, and a
    /// compare/exchange loop for float `add` (which reassociates).
    AtomicDevice {
        op: AtomicOp,
        operand: usize,
        layout: AccessLayout,
        indices: Vec<IndexRef>,
        value: Src,
        dtype: DType,
        guard: Option<Guard>,
    },
    ReduceSerial {
        op: ReduceOp,
        operand: usize,
        layout: AccessLayout,
        axis: usize,
        axis_length: AddrExpr,
        input_dtype: DType,
        accumulator: DType,
        result: ReduceResult,
        guard: Option<Guard>,
    },
    /// The blocked-cover fold: each of `lanes` participants folds its
    /// strided share of the reduced axis ascending, publishes its partial
    /// into planned workgroup storage, and lane zero combines the partials
    /// in lane order (the interleaved tree cover).
    BlockedReduce {
        op: ReduceOp,
        operand: usize,
        layout: AccessLayout,
        axis: usize,
        axis_length: AddrExpr,
        lanes: u32,
        input_dtype: DType,
        accumulator: DType,
        result: ReduceResult,
        nonempty: Option<Guard>,
    },
    SubgroupLaneIndex {
        into: ValueRef,
    },
    SubgroupShuffle {
        value: Src,
        index: Src,
        into: ValueRef,
        dtype: DType,
    },
    SubgroupReduce {
        op: ReduceOp,
        operand: usize,
        layout: AccessLayout,
        axis: usize,
        axis_length: AddrExpr,
        input_dtype: DType,
        accumulator: DType,
        result: ReduceResult,
        guard: Option<Guard>,
    },
}

// ---------------------------------------------------------------------------
// Consequences
// ---------------------------------------------------------------------------

/// Measured identity of the Metal cost model: the probe-calibrated estimate
/// model carried with provenance. Uncalibrated values affect ranking only,
/// never legality.
pub fn cost_model_identity() -> &'static str {
    crate::target::COST_MODEL_IDENTITY
}

const LANE_OP_COST: u64 = 2;
const ACCESS_COST: u64 = 4;
const COLLECTIVE_COST: u64 = 25;

fn subgroup_intrinsic(name: &str) -> IntrinsicId {
    IntrinsicId {
        capability: CapabilityId::new("metal", "subgroup"),
        name: name.into(),
    }
}

fn matrix_intrinsic(name: &str) -> IntrinsicId {
    IntrinsicId {
        capability: CapabilityId::new("metal", "matrix"),
        name: name.into(),
    }
}

/// Conservative cost units of one address expression (static products
/// evaluated; runtime extents count as one axis of work).
fn addr_units(expr: &AddrExpr) -> u64 {
    match expr {
        AddrExpr::Const(n) => (*n).max(1),
        _ => 8,
    }
}

fn universal_consequences(units: u64, static_code_units: u64) -> PhysicalConsequences {
    PhysicalConsequences {
        hard: HardResources {
            static_code_units,
            ..HardResources::default()
        },
        native_contract: NativeResourceContract {
            max_resident_participants: (1, u64::MAX),
            native_subgroup_width: None,
        },
        cost: seismic_realization::executable::CostEstimate(units),
        numerical: seismic_realization::numerics::NumericalTransfer::Exact,
        capability: None,
    }
}

fn subgroup_consequences(
    intrinsic: IntrinsicId,
    numerical: seismic_realization::numerics::NumericalTransfer,
    units: u64,
) -> PhysicalConsequences {
    PhysicalConsequences {
        hard: HardResources {
            required_subgroup_width: Some(32),
            static_code_units: 8,
            ..HardResources::default()
        },
        native_contract: NativeResourceContract {
            max_resident_participants: (1, u64::MAX),
            // The MSL simd width on Apple GPUs is 32; the conservative
            // admissible domain admits any reflected width in 1..=64.
            native_subgroup_width: Some((1, 64)),
        },
        cost: seismic_realization::executable::CostEstimate(units),
        numerical,
        capability: Some(intrinsic),
    }
}

fn reassociate_transfer(op: ReduceOp) -> seismic_realization::numerics::NumericalTransfer {
    seismic_realization::numerics::NumericalTransfer::Reassociate {
        op,
        topology: seismic_realization::numerics::ReductionTopology::Subgroup {
            width: 32,
            inner: Box::new(
                seismic_realization::numerics::ReductionTopology::SerialAxis {
                    axis: 0,
                    length: ExtentExpr::Static(0),
                },
            ),
        },
    }
}

impl seismic_realization::executable::ExecutableDialect for MetalDialect {
    type Op = MetalOp;
    type LayoutTemplate = MetalLayout;
    type ResolvedLayout = ResolvedMetalLayout;

    fn legalize(
        p: &realization::PhysicalPrimitive,
        t: &EffectiveTargetProfile,
    ) -> Legalized<MetalOp> {
        match &p.op {
            PrimitiveOp::Capability(intrinsic) => {
                // Capability applications are not universal: their physical
                // alternatives live behind the exact effective signature and
                // are constructed by the capability strategy.
                let reason = if t.effective_signatures.contains(intrinsic) {
                    format!(
                        "capability `{}` is consumed by the capability strategy, not the \
                         universal primitive mapping",
                        intrinsic.path()
                    )
                } else {
                    format!(
                        "exact capability signature `{}` is absent from the effective Metal \
                         target profile",
                        intrinsic.path()
                    )
                };
                Legalized::Inapplicable { reason }
            }
            op => {
                // The universal column is total over the closed registry; the
                // returned form-classifying opcode is replaced by the
                // strategy's value-carrying opcodes (see `fuse_segment`).
                match terminal::universal_form(op, &p.inputs) {
                    Ok(UniversalLegalization::Form(_)) => Legalized::Ops(
                        NonEmpty::new(vec![MetalOp::View]).expect("one classifying opcode"),
                    ),
                    Ok(UniversalLegalization::RequiresCapability { intrinsic }) => {
                        Legalized::Inapplicable {
                            reason: format!(
                                "capability `{}` has no portable physical opcode",
                                intrinsic.path()
                            ),
                        }
                    }
                    Err(bug) => Legalized::Inapplicable {
                        reason: format!("compiler bug in universal legalization: {bug}"),
                    },
                }
            }
        }
    }

    fn consequences(op: &MetalOp) -> PhysicalConsequences {
        use MetalOp::*;
        match op {
            Const { .. }
            | ExtentOf { .. }
            | View
            | Alloc
            | TupleGet { .. }
            | RangeEndpoint { .. } => universal_consequences(1, 2),
            Select { .. } | Compare { .. } | BoolLogic { .. } | Unary { .. } | Cast { .. } => {
                universal_consequences(LANE_OP_COST, 4)
            }
            IntOp { .. } | FloatOp { .. } | Fma { .. } => universal_consequences(LANE_OP_COST, 4),
            IntDivRem { .. } | Shift { .. } => universal_consequences(LANE_OP_COST * 4, 8),
            Math { op, .. } => {
                let exact = matches!(
                    op,
                    MathOp::Sqrt | MathOp::Fma | MathOp::Abs | MathOp::Max | MathOp::Min
                );
                let mut consequences = universal_consequences(LANE_OP_COST * 8, 16);
                if !exact {
                    // Native MSL transcendental: not proved bit-equivalent to
                    // the seismic_math reference (Apple GPUs have no FP64, so
                    // the software sequence cannot be hosted). Evidence-gated
                    // `Unknown` transfer; never implicit.
                    consequences.numerical =
                        seismic_realization::numerics::NumericalTransfer::Unknown {
                            reason: format!(
                                "MSL native `{}` is not proved bit-equivalent to the {}-v{} \
                                 reference sequence",
                                op.name(),
                                SEISMIC_MATH.identity,
                                SEISMIC_MATH.version
                            ),
                        };
                }
                consequences
            }
            ReadElement { .. } | PackedPlaneRead { .. } => universal_consequences(ACCESS_COST, 8),
            PackedElementRead { .. } => universal_consequences(ACCESS_COST, 8),
            SerialFor { .. } | Branch { .. } => universal_consequences(ACCESS_COST, 8),
            WriteElement { .. } | StoreResult { .. } => universal_consequences(ACCESS_COST, 8),
            LinearLoop { .. } => universal_consequences(ACCESS_COST * 2, 16),
            PackedDecode { .. } => universal_consequences(ACCESS_COST * 4, 32),
            AtomicSerial { .. } => universal_consequences(ACCESS_COST * 2, 12),
            AtomicDevice { op, dtype, .. } => {
                // Contention is not modelled; the cost is the atomic's own
                // round trip. Float `add` combines in a data-dependent order.
                let mut consequences = universal_consequences(ACCESS_COST * 4, 16);
                if *op == AtomicOp::Add && matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
                    consequences.numerical = reassociate_transfer(ReduceOp::Sum);
                }
                consequences
            }
            ReduceSerial { .. } => universal_consequences(ACCESS_COST * 2, 16),
            BlockedReduce { .. } => {
                let mut consequences = universal_consequences(ACCESS_COST * 2, 16);
                // The partials occupy one accumulator word per lane in
                // planned workgroup storage.
                consequences.hard.explicit_workgroup_bytes = 128;
                consequences
            }
            SubgroupLaneIndex { .. } => subgroup_consequences(
                subgroup_intrinsic("lane_index"),
                seismic_realization::numerics::NumericalTransfer::Exact,
                1,
            ),
            SubgroupShuffle { dtype, .. } => subgroup_consequences(
                subgroup_intrinsic("shuffle"),
                seismic_realization::numerics::NumericalTransfer::Round {
                    dtype: *dtype,
                    count: seismic_realization::numerics::CountExpr::one(),
                },
                COLLECTIVE_COST / 4,
            ),
            SubgroupReduce { op, .. } => subgroup_consequences(
                subgroup_intrinsic(match op {
                    ReduceOp::Sum => "simd_sum",
                    ReduceOp::Max => "simd_max",
                    ReduceOp::Min => "simd_min",
                    ReduceOp::Argmax => unreachable!("argmax never reassociates"),
                }),
                reassociate_transfer(*op),
                COLLECTIVE_COST,
            ),
            MatrixMatmul {
                k,
                signature,
                signature_arguments,
                ..
            } => PhysicalConsequences {
                hard: HardResources::default(),
                native_contract: NativeResourceContract {
                    max_resident_participants: (1, u64::MAX),
                    native_subgroup_width: None,
                },
                cost: seismic_realization::executable::CostEstimate(LANE_OP_COST * addr_units(k)),
                numerical: seismic_realization::numerics::NumericalTransfer::Capability {
                    signature: seismic_realization::numerics::CapabilitySignatureId::new(
                        signature.clone(),
                        signature_arguments.clone(),
                    ),
                    bound: None,
                },
                capability: Some(signature.clone()),
            },
        }
    }

    fn public_layout(tensor: &TensorType) -> MetalLayout {
        layout_of(tensor)
    }

    fn internal_layout(tensor: &TensorType) -> MetalLayout {
        layout_of(tensor)
    }

    fn resolve_layout(
        layout: &MetalLayout,
        values: &realization::PlanValues,
    ) -> Result<ResolvedMetalLayout, realization::InvariantReport> {
        let elements = values.eval(&layout_elements(layout))?;
        Ok(match layout {
            MetalLayout::Dense { dtype, .. } => ResolvedMetalLayout::Dense {
                dtype: *dtype,
                elements,
            },
            MetalLayout::PackedPlane {
                representation,
                plane,
                dtype,
                ..
            } => ResolvedMetalLayout::PackedPlane {
                representation: representation.clone(),
                plane: plane.clone(),
                dtype: *dtype,
                elements,
            },
        })
    }
}

fn layout_elements(layout: &MetalLayout) -> Sym {
    match layout {
        MetalLayout::Dense { elements, .. } | MetalLayout::PackedPlane { elements, .. } => {
            elements.clone()
        }
    }
}

fn layout_of(tensor: &TensorType) -> MetalLayout {
    let mut elements = Sym::constant(1);
    for axis in &tensor.axes {
        let factor = match axis {
            ExtentExpr::Static(n) => Sym::constant(i64::try_from(*n).unwrap_or(i64::MAX)),
            ExtentExpr::Sym(sym) => sym.clone(),
            // Runtime extents contribute their capacity bound to the
            // resource-size expression; the retained runtime value is carried
            // by the plan's runtime extents.
            ExtentExpr::Runtime(_) => Sym::constant(i64::MAX),
        };
        elements = elements.mul(&factor);
    }
    match &tensor.elem {
        Elem::Dtype(dtype) => MetalLayout::Dense {
            dtype: *dtype,
            elements,
        },
        Elem::Param(_) => MetalLayout::Dense {
            dtype: DType::F32,
            elements,
        },
        Elem::Repr(_) => MetalLayout::Dense {
            dtype: DType::U32,
            elements,
        },
    }
}

// ---------------------------------------------------------------------------
// Elaboration
// ---------------------------------------------------------------------------

/// Hard limits the strategies need.
#[derive(Clone, Copy, Debug)]
pub struct StrategyLimits {
    pub max_participants: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReductionMode {
    Universal,
    Subgroup,
    /// The interleaved blocked cover: lane-strided partial folds plus a
    /// workgroup tree combine (fixed 32 lanes).
    Blocked,
}

/// How a launch containing `atomic` updates is realized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtomicMapping {
    /// The universal form: the launch domain runs on one participant and
    /// every update is the exact load/combine/round/store in visit order.
    Serialized,
    /// The concurrent form: the ordinary participant domain, every update a
    /// device atomic (native for integers and `max`/`min`; a compare/exchange
    /// loop for `f32 add`). Float `add` reassociates.
    Device,
}

/// Construct the complete Metal plan family of one logical program: the
/// universal physical alternative of every applicable logical alternative,
/// plus the subgroup reduction alternative where exact profile facts admit
/// it.
pub fn elaborate(
    logical: &LogicalProgram,
    profile: &EffectiveTargetProfile,
    limits: StrategyLimits,
) -> Result<PlanFamily<MetalDialect>, BuilderError> {
    let mut builder = PlanFamilyBuilder::<MetalDialect>::from_logical(logical)?;
    for choice_id in logical.choices.ids() {
        let choice = logical
            .choices
            .get(choice_id)
            .ok_or_else(|| format!("choice#{} is absent", choice_id.0))?;
        for ordinal in 0..choice.alternatives.len() {
            let graph = logical
                .graph(choice.alternatives.iter().nth(ordinal).unwrap().graph)
                .clone();
            let facts = GraphFacts::collect(&graph, &logical.runtime_extents);
            // Universal alternative.
            {
                let mut alternative = builder.alternative(choice_id, ordinal as u32)?;
                build_alternative(
                    &mut alternative,
                    logical,
                    &graph,
                    &facts,
                    profile,
                    limits,
                    ReductionMode::Universal,
                    false,
                    false,
                    AtomicMapping::Serialized,
                )?;
                builder.add_alternative(choice_id, alternative.finish_alternative()?)?;
            }
            // Device-atomic alternative: launches containing `atomic` updates
            // keep their participant domain and combine through device
            // atomics (32-bit elements only; float `add` reassociates).
            if atomics_admitted(&graph, &facts) {
                let mut alternative = builder.alternative(choice_id, ordinal as u32)?;
                build_alternative(
                    &mut alternative,
                    logical,
                    &graph,
                    &facts,
                    profile,
                    limits,
                    ReductionMode::Universal,
                    false,
                    false,
                    AtomicMapping::Device,
                )?;
                builder.add_alternative(choice_id, alternative.finish_alternative()?)?;
            }
            // Blocked-cover alternative: interleaved lane folds with a
            // workgroup tree combine, admitted for source-unordered
            // reductions (no collective signature needed).
            if blocked_admitted(&graph) {
                let mut alternative = builder.alternative(choice_id, ordinal as u32)?;
                build_alternative(
                    &mut alternative,
                    logical,
                    &graph,
                    &facts,
                    profile,
                    limits,
                    ReductionMode::Blocked,
                    false,
                    false,
                    AtomicMapping::Serialized,
                )?;
                builder.add_alternative(choice_id, alternative.finish_alternative()?)?;
            }
            // Runtime-axis streaming alternative: ordered loops whose
            // carried state matches the structural scan rule window over a
            // solver-tunable width instead of materializing the axis.
            if streaming_admitted(&graph, &facts) {
                let mut alternative = builder.alternative(choice_id, ordinal as u32)?;
                build_alternative(
                    &mut alternative,
                    logical,
                    &graph,
                    &facts,
                    profile,
                    limits,
                    ReductionMode::Universal,
                    false,
                    true,
                    AtomicMapping::Serialized,
                )?;
                builder.add_alternative(choice_id, alternative.finish_alternative()?)?;
            }
            // Cross-call fusion alternative: every call whose child graph
            // holds only primitives fuses inline instead of invoking.
            if cross_call_admitted(&graph, logical) {
                let mut alternative = builder.alternative(choice_id, ordinal as u32)?;
                build_alternative(
                    &mut alternative,
                    logical,
                    &graph,
                    &facts,
                    profile,
                    limits,
                    ReductionMode::Universal,
                    true,
                    false,
                    AtomicMapping::Serialized,
                )?;
                builder.add_alternative(choice_id, alternative.finish_alternative()?)?;
            }
            // Subgroup reduction alternative, only with exact profile facts.
            if subgroup_admitted(&graph, profile) {
                let mut alternative = builder.alternative(choice_id, ordinal as u32)?;
                build_alternative(
                    &mut alternative,
                    logical,
                    &graph,
                    &facts,
                    profile,
                    limits,
                    ReductionMode::Subgroup,
                    false,
                    false,
                    AtomicMapping::Serialized,
                )?;
                builder.add_alternative(choice_id, alternative.finish_alternative()?)?;
            }
        }
    }
    builder.finish()
}

/// Map one ordered loop through the structural streaming rule: the axis
/// windows over a solver-tunable width, the derived lanes become explicit
/// physical carries, and the body maps once per window.
fn strategies_cost_model() -> CostModelId {
    CostModelId::measured(
        "metal",
        crate::target::COST_MODEL_IDENTITY,
        "probe-calibrated device timing",
    )
}

fn stream_loop_node(
    builder: &mut AlternativeBuilder<MetalDialect>,
    cell: &std::cell::RefCell<StrategyContext<'_>>,
    path: RegionPath,
    node_id: NodeId,
    node: &LogicalNode,
) -> Result<(), BuilderError> {
    use strategies::streaming::{derive_scan_state, stream_scan, ScanStateMatch, StreamingWindow};
    let node_ref = NodeRef {
        region: path.clone(),
        node: node_id,
    };
    discharge_node_obligations(builder, cell, &node_ref, node)?;
    let (state, capacity) = {
        let context = cell.borrow();
        let loop_node = match &node.kind {
            LogicalNodeKind::Loop(loop_node) => loop_node,
            _ => unreachable!("the caller matched a loop"),
        };
        let capacity = match &loop_node.range.bound {
            ExtentExpr::Static(n) => *n,
            ExtentExpr::Runtime(id) => context
                .facts
                .runtime_extents
                .get(id)
                .map(|extent| extent.capacity)
                .unwrap_or(u64::MAX),
            ExtentExpr::Sym(sym) => {
                return Err(format!(
                    "an unresolved symbol `{sym}` survived into a streaming window"
                )
                .into());
            }
        };
        match derive_scan_state(context.graph, &node_ref, &context.facts) {
            ScanStateMatch::Streaming(state) => (state, capacity),
            ScanStateMatch::NoStreaming { reason } => {
                return Err(format!("the streaming rule stopped matching: {reason}").into());
            }
        }
    };
    let window = StreamingWindow::declare(builder, &node_ref, capacity)?;
    let body_path = {
        let mut p = path.clone();
        p.push(RegionStep::LoopBody(node_id));
        p
    };
    let model = strategies_cost_model();
    stream_scan(
        builder,
        node_ref,
        &state,
        &window,
        |builder| build_region(builder, cell, body_path.clone()),
        model,
    )?;
    Ok(())
}

/// Whether the streaming alternative is offered: some ordered loop in the
/// graph matches the structural scan rule.
fn streaming_admitted(graph: &TaskGraph, facts: &GraphFacts) -> bool {
    fn scan(
        region: &GraphRegion,
        graph: &TaskGraph,
        facts: &GraphFacts,
        path: &mut RegionPath,
    ) -> bool {
        for (node_id, node) in region.nodes.ids().zip(region.nodes.iter()) {
            if let LogicalNodeKind::Loop(loop_node) = &node.kind {
                if loop_node.kind == LoopKind::Ordered {
                    path.push(RegionStep::LoopBody(node_id));
                    let occurrence = NodeRef {
                        region: path.clone(),
                        node: node_id,
                    };
                    if matches!(
                        strategies::streaming::derive_scan_state(graph, &occurrence, facts),
                        strategies::streaming::ScanStateMatch::Streaming(_)
                    ) {
                        return true;
                    }
                    path.pop();
                    if scan(&loop_node.body, graph, facts, path) {
                        return true;
                    }
                }
            }
        }
        false
    }
    let mut path = Vec::new();
    scan(&graph.root, graph, facts, &mut path)
}

/// Whether the blocked alternative is offered: every reduction in the
/// graph is `unordered` (source-admitted reassociation) and not argmax.
fn blocked_admitted(graph: &TaskGraph) -> bool {
    fn scan(region: &GraphRegion, verdict: &mut bool, any: &mut bool) {
        for node in region.nodes.iter() {
            if let LogicalNodeKind::Reduction(reduction) = &node.kind {
                *any = true;
                if reduction.op == ReduceOp::Argmax || reduction.order != ReductionOrder::Unordered
                {
                    *verdict = false;
                }
            }
            match &node.kind {
                LogicalNodeKind::Loop(inner) => scan(&inner.body, verdict, any),
                LogicalNodeKind::If(inner) => {
                    scan(&inner.then_region, verdict, any);
                    scan(&inner.else_region, verdict, any);
                }
                _ => {}
            }
        }
    }
    let mut verdict = true;
    let mut any = false;
    scan(&graph.root, &mut verdict, &mut any);
    verdict && any
}

/// Whether the cross-call fusion alternative is offered: some call in the
/// graph targets a child graph whose body holds only primitives.
fn cross_call_admitted(graph: &TaskGraph, logical: &LogicalProgram) -> bool {
    fn scan(region: &GraphRegion, logical: &LogicalProgram) -> bool {
        for node in region.nodes.iter() {
            if let LogicalNodeKind::Call(call) = &node.kind {
                if let Some(choice) = logical.choices.get(call.choice) {
                    let first = choice.alternatives.first();
                    if let Some(child) = logical.graphs.get(first.graph) {
                        if region_absorbable(&child.root) {
                            return true;
                        }
                    }
                }
            }
            match &node.kind {
                LogicalNodeKind::Loop(inner) => {
                    if scan(&inner.body, logical) {
                        return true;
                    }
                }
                LogicalNodeKind::If(inner) => {
                    if scan(&inner.then_region, logical) || scan(&inner.else_region, logical) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }
    scan(&graph.root, logical)
}

/// Whether the blocked alternative is offered:
/// Whether the subgroup alternative is offered: every reduction in the graph
/// is `unordered` (source-admitted reassociation), not argmax, and its
/// collective signature is exactly effective.
/// Elements the device-atomic mapping combines: one 32-bit word each, so a
/// native atomic or a compare/exchange on the word is exact per update.
fn device_atomic_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::I32 | DType::U32)
}

/// Whether the graph updates storage atomically anywhere, every such update
/// on a 32-bit element (the device-atomic alternative is then constructed).
fn atomics_admitted(graph: &TaskGraph, facts: &GraphFacts) -> bool {
    fn walk(region: &GraphRegion, facts: &GraphFacts, found: &mut bool, admitted: &mut bool) {
        for node in region.nodes.iter() {
            match &node.kind {
                LogicalNodeKind::Primitive(application) => {
                    if let PrimitiveOp::Primitive(PrimitiveId::Atomic { .. }) = &application.op {
                        *found = true;
                        let value = node.inputs.last().copied();
                        let dtype = value
                            .and_then(|value| facts.types.get(&value))
                            .and_then(|ty| ty.scalar_dtype());
                        if !dtype.is_some_and(device_atomic_dtype) {
                            *admitted = false;
                        }
                    }
                }
                LogicalNodeKind::Loop(inner) => walk(&inner.body, facts, found, admitted),
                LogicalNodeKind::If(inner) => {
                    walk(&inner.then_region, facts, found, admitted);
                    walk(&inner.else_region, facts, found, admitted);
                }
                _ => {}
            }
        }
    }
    let mut found = false;
    let mut admitted = true;
    walk(&graph.root, facts, &mut found, &mut admitted);
    found && admitted
}

/// Whether any entry of a fused launch (at any nesting) is an atomic update.
fn entries_update_atomically(entries: &[MetalEntry]) -> bool {
    entries.iter().any(|entry| match &entry.kind {
        MEntryKind::Primitive => matches!(
            &entry.logical.kind,
            LogicalNodeKind::Primitive(application)
                if matches!(application.op, PrimitiveOp::Primitive(PrimitiveId::Atomic { .. }))
        ),
        MEntryKind::Axis { .. } => false,
        MEntryKind::SerialLoop { body, .. } => entries_update_atomically(body),
        MEntryKind::Branch {
            then_body,
            else_body,
            ..
        } => entries_update_atomically(then_body) || entries_update_atomically(else_body),
    })
}

fn subgroup_admitted(graph: &TaskGraph, profile: &EffectiveTargetProfile) -> bool {
    let mut verdict = true;
    let mut any = false;
    fn scan(
        region: &GraphRegion,
        profile: &EffectiveTargetProfile,
        verdict: &mut bool,
        any: &mut bool,
    ) {
        for node in region.nodes.iter() {
            if let LogicalNodeKind::Reduction(reduction) = &node.kind {
                *any = true;
                let admitted = reduction.order == logical::ReductionOrder::Unordered
                    && reduction.op == ReduceOp::Sum
                    && collective_signature_effective(reduction, profile);
                if !admitted {
                    *verdict = false;
                }
            }
            match &node.kind {
                LogicalNodeKind::Loop(loop_node) => scan(&loop_node.body, profile, verdict, any),
                LogicalNodeKind::If(if_node) => {
                    scan(&if_node.then_region, profile, verdict, any);
                    scan(&if_node.else_region, profile, verdict, any);
                }
                _ => {}
            }
        }
    }
    scan(&graph.root, profile, &mut verdict, &mut any);
    any && verdict
}

fn collective_signature_effective(
    reduction: &ReductionNode,
    profile: &EffectiveTargetProfile,
) -> bool {
    // Only simd_sum is offered: simd_max/simd_min emission is not implemented.
    match reduction.op {
        ReduceOp::Sum => profile
            .effective_signatures
            .contains(&subgroup_intrinsic("simd_sum")),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// One alternative: the consuming walk
// ---------------------------------------------------------------------------

struct StrategyContext<'a> {
    graph: &'a TaskGraph,
    facts: GraphFacts,
    profile: &'a EffectiveTargetProfile,
    participants: Sym,
    reduction_mode: ReductionMode,
    /// Joined independent-loop extents of the region being absorbed into
    /// one launch (empty for plain domain-keyed segments).
    axes: Vec<ExtentExpr>,
    /// Absorbed independent-loop binders and their iteration axis.
    binder_axis: BTreeMap<GraphValueId, usize>,
    /// Whether call occurrences fuse their child graphs inline (the
    /// cross-call fusion alternative).
    fuse_calls: bool,
    /// Whether ordered loops whose carried state matches the structural
    /// streaming rule become windowed scan launches.
    streaming: bool,
    /// The first-alternative graph of every callee choice of this graph.
    child_graphs: BTreeMap<ChoiceId, TaskGraph>,
    next_ssa: u32,
    /// Values produced as SSA inside the current mapped block (for point-slice
    /// offsets referencing them).
    ssa_values: BTreeMap<GraphValueId, u32>,
    /// Launch operand index of each value bound by the node being mapped
    /// (slice point and binder values resolve through these).
    operand_values: BTreeMap<GraphValueId, ValueRef>,
    /// How launches containing `atomic` updates are realized.
    atomics: AtomicMapping,
}

/// Build one physical alternative; returns the boundary-result transports.
fn build_alternative(
    builder: &mut AlternativeBuilder<MetalDialect>,
    logical: &LogicalProgram,
    graph: &TaskGraph,
    facts: &GraphFacts,
    profile: &EffectiveTargetProfile,
    limits: StrategyLimits,
    reduction_mode: ReductionMode,
    fuse_calls: bool,
    streaming: bool,
    atomics: AtomicMapping,
) -> Result<(), BuilderError> {
    // The first-alternative graph of every callee choice of this graph.
    let mut child_graphs: BTreeMap<ChoiceId, TaskGraph> = BTreeMap::new();
    fn scan_calls(
        region: &GraphRegion,
        logical: &LogicalProgram,
        children: &mut BTreeMap<ChoiceId, TaskGraph>,
    ) {
        for node in region.nodes.iter() {
            if let LogicalNodeKind::Call(call) = &node.kind {
                if !children.contains_key(&call.choice) {
                    if let Some(choice) = logical.choices.get(call.choice) {
                        let first = choice.alternatives.first();
                        if let Some(graph) = logical.graphs.get(first.graph) {
                            children.insert(call.choice, graph.clone());
                        }
                    }
                }
            }
            match &node.kind {
                LogicalNodeKind::Loop(inner) => scan_calls(&inner.body, logical, children),
                LogicalNodeKind::If(inner) => {
                    scan_calls(&inner.then_region, logical, children);
                    scan_calls(&inner.else_region, logical, children);
                }
                _ => {}
            }
        }
    }
    scan_calls(&graph.root, logical, &mut child_graphs);
    // One solver-tunable participant width per alternative (grid-stride
    // shaped-domain launches).
    let participants = builder.solver_participants(1, limits.max_participants.max(1))?;
    let context = StrategyContext {
        graph,
        facts: facts.clone(),
        profile,
        participants,
        reduction_mode,
        axes: Vec::new(),
        binder_axis: BTreeMap::new(),
        fuse_calls,
        streaming,
        child_graphs,
        next_ssa: 0,
        ssa_values: BTreeMap::new(),
        operand_values: BTreeMap::new(),
        atomics,
    };
    let cell = std::cell::RefCell::new(context);
    build_region(builder, &cell, Vec::new())?;
    let results = graph.results.clone();
    for (ordinal, result) in results.iter().enumerate() {
        let transport = match result {
            RegionResult::Value { id, .. } => builder.transport_of(*id)?,
            RegionResult::State { storage, .. } => {
                builder.transport_of_storage(*storage, Access::Exclusive)?
            }
        };
        builder.complete_result(ordinal as u32, transport)?;
    }
    Ok(())
}

/// How one collected entry contributes to a fused Metal launch.
enum MEntryKind {
    Primitive,
    /// An absorbed independent loop: its axis joins the launch domain.
    Axis {
        binder: GraphValueId,
    },
    /// An absorbed ordered loop: an in-kernel serial frame.
    SerialLoop {
        binder: GraphValueId,
        length: ExtentExpr,
        body: Vec<MetalEntry>,
    },
    /// An absorbed conditional: an in-kernel branch.
    Branch {
        condition: GraphValueId,
        then_body: Vec<MetalEntry>,
        else_body: Vec<MetalEntry>,
    },
}

struct MetalEntry {
    node: NodeRef,
    logical: LogicalNode,
    formed: Option<FormedPrimitive>,
    kind: MEntryKind,
}

/// One collected region segment: a fused launch, or a structural node on
/// the executor path.
enum MSegment {
    Component(Vec<MetalEntry>),
    Executor(NodeId, LogicalNode),
}

fn build_region(
    builder: &mut AlternativeBuilder<MetalDialect>,
    cell: &std::cell::RefCell<StrategyContext<'_>>,
    path: RegionPath,
) -> Result<(), BuilderError> {
    // Each region collects and commits under its own joined axes; nested
    // executor bodies start fresh.
    {
        let mut context = cell.borrow_mut();
        context.axes.clear();
        context.binder_axis.clear();
    }
    let segments = collect_region(builder, cell, &path, false)?;
    let result = commit_segments(builder, cell, &path, segments);
    {
        let mut context = cell.borrow_mut();
        context.axes.clear();
        context.binder_axis.clear();
    }
    result
}

/// Collect the region's segments in order. Independent loops with
/// primitive-only bodies are absorbed into one component; ordered loops and
/// conditionals inside an absorbed region become serial frames and
/// branches; anything retaining a call or reduction (or a top-level ordered
/// loop) is an executor segment.
fn collect_region(
    builder: &mut AlternativeBuilder<MetalDialect>,
    cell: &std::cell::RefCell<StrategyContext<'_>>,
    path: &RegionPath,
    absorbing: bool,
) -> Result<Vec<MSegment>, BuilderError> {
    let nodes = builder.region_nodes(path)?;
    let mut segments: Vec<MSegment> = Vec::new();
    let mut run: Vec<MetalEntry> = Vec::new();
    for (node_id, node) in nodes {
        let node_ref = NodeRef {
            region: path.clone(),
            node: node_id,
        };
        match &node.kind {
            LogicalNodeKind::Primitive(application) => {
                if let PrimitiveOp::Capability(intrinsic) = &application.op {
                    if intrinsic.capability.path() == "metal.matrix"
                        && matches!(intrinsic.name.as_str(), "matmul" | "matmul_add")
                    {
                        // Matrix intrinsics map through their dedicated
                        // path; they are never fused entries.
                        segments.push(MSegment::Executor(node_id, node.clone()));
                        continue;
                    }
                }
                let formed = {
                    let context = cell.borrow();
                    form_primitive(node_id, &node, &context.facts).map_err(|e| e.to_string())?
                };
                run.push(MetalEntry {
                    node: node_ref,
                    logical: node.clone(),
                    formed: Some(formed),
                    kind: MEntryKind::Primitive,
                });
            }
            LogicalNodeKind::Loop(loop_node) => {
                let absorbable = region_absorbable(&loop_node.body);
                match (loop_node.kind, absorbable, absorbing) {
                    (LoopKind::Independent, true, _) => {
                        // The axis joins the current launch domain; record
                        // the binder's axis and absorb the body inline.
                        let _axis = {
                            let mut context = cell.borrow_mut();
                            let axis = context.axes.len();
                            context.axes.push(loop_node.range.bound.clone());
                            context.binder_axis.insert(loop_node.binder, axis);
                            axis
                        };
                        run.push(MetalEntry {
                            node: node_ref,
                            logical: node.clone(),
                            formed: None,
                            kind: MEntryKind::Axis {
                                binder: loop_node.binder,
                            },
                        });
                        let body_path = {
                            let mut p = path.clone();
                            p.push(RegionStep::LoopBody(node_id));
                            p
                        };
                        let body_segments = collect_region(builder, cell, &body_path, true)?;
                        for segment in body_segments {
                            match segment {
                                MSegment::Component(entries) => run.extend(entries),
                                MSegment::Executor(..) => {
                                    // absorbable() guarantees this is
                                    // unreachable for the body.
                                    return Err(BuilderError::from(
                                        "an absorbable body retained an executor node".to_string(),
                                    ));
                                }
                            }
                        }
                    }
                    (LoopKind::Ordered, true, true) => {
                        let body_path = {
                            let mut p = path.clone();
                            p.push(RegionStep::LoopBody(node_id));
                            p
                        };
                        let body_segments = collect_region(builder, cell, &body_path, true)?;
                        let mut body = Vec::new();
                        for segment in body_segments {
                            match segment {
                                MSegment::Component(entries) => body.extend(entries),
                                MSegment::Executor(..) => {
                                    return Err(BuilderError::from(
                                        "an absorbable body retained an executor node".to_string(),
                                    ));
                                }
                            }
                        }
                        run.push(MetalEntry {
                            node: node_ref,
                            logical: node.clone(),
                            formed: None,
                            kind: MEntryKind::SerialLoop {
                                binder: loop_node.binder,
                                length: loop_node.range.bound.clone(),
                                body,
                            },
                        });
                    }
                    _ => {
                        if !run.is_empty() {
                            segments.push(MSegment::Component(std::mem::take(&mut run)));
                        }
                        segments.push(MSegment::Executor(node_id, node.clone()));
                    }
                }
            }
            LogicalNodeKind::If(if_node) if absorbing => {
                let then_path = {
                    let mut p = path.clone();
                    p.push(RegionStep::IfThen(node_id));
                    p
                };
                let else_path = {
                    let mut p = path.clone();
                    p.push(RegionStep::IfElse(node_id));
                    p
                };
                let flatten = |segments: Vec<MSegment>| -> Result<Vec<MetalEntry>, BuilderError> {
                    let mut flat = Vec::new();
                    for segment in segments {
                        match segment {
                            MSegment::Component(entries) => flat.extend(entries),
                            MSegment::Executor(..) => {
                                return Err(BuilderError::from(
                                    "an absorbable body retained an executor node".to_string(),
                                ));
                            }
                        }
                    }
                    Ok(flat)
                };
                let then_body = flatten(collect_region(builder, cell, &then_path, true)?)?;
                let else_body = flatten(collect_region(builder, cell, &else_path, true)?)?;
                run.push(MetalEntry {
                    node: node_ref,
                    logical: node.clone(),
                    formed: None,
                    kind: MEntryKind::Branch {
                        condition: if_node.condition,
                        then_body,
                        else_body,
                    },
                });
            }
            LogicalNodeKind::If(_) | LogicalNodeKind::Reduction(_) | LogicalNodeKind::Call(_) => {
                if !run.is_empty() {
                    segments.push(MSegment::Component(std::mem::take(&mut run)));
                }
                segments.push(MSegment::Executor(node_id, node.clone()));
            }
        }
    }
    if !run.is_empty() {
        segments.push(MSegment::Component(run));
    }
    Ok(segments)
}

/// Commit each collected segment in order.
fn commit_segments(
    builder: &mut AlternativeBuilder<MetalDialect>,
    cell: &std::cell::RefCell<StrategyContext<'_>>,
    path: &RegionPath,
    segments: Vec<MSegment>,
) -> Result<(), BuilderError> {
    for segment in segments {
        match segment {
            MSegment::Component(entries) => {
                if !entries.is_empty() {
                    fuse_segment(builder, cell, path.clone(), entries)?;
                }
            }
            MSegment::Executor(node_id, node) => {
                match &node.kind {
                    LogicalNodeKind::Reduction(reduction) => {
                        map_reduction_node(builder, cell, path.clone(), node_id, &node, reduction)?;
                    }
                    LogicalNodeKind::Loop(loop_node) => {
                        let streamable = {
                            let context = cell.borrow();
                            context.streaming
                                && loop_node.kind == LoopKind::Ordered
                                && matches!(
                                    strategies::streaming::derive_scan_state(
                                        context.graph,
                                        &NodeRef {
                                            region: path.clone(),
                                            node: node_id,
                                        },
                                        &context.facts,
                                    ),
                                    strategies::streaming::ScanStateMatch::Streaming(_)
                                )
                        };
                        if streamable {
                            stream_loop_node(builder, cell, path.clone(), node_id, &node)?;
                        } else {
                            schedule_loop_node(
                                builder,
                                cell,
                                path.clone(),
                                node_id,
                                &node,
                                loop_node,
                            )?;
                        }
                    }
                    LogicalNodeKind::If(if_node) => {
                        schedule_if_node(builder, cell, path.clone(), node_id, &node, if_node)?;
                    }
                    LogicalNodeKind::Call(call_node) => {
                        let fused = {
                            let context = cell.borrow();
                            context.fuse_calls
                                && context
                                    .child_graphs
                                    .get(&call_node.choice)
                                    .is_some_and(|child| region_absorbable(&child.root))
                        };
                        if fused {
                            // Cross-call fusion: substitute the boundary,
                            // import the child graph, and walk it inline.
                            let node_ref = NodeRef {
                                region: path.clone(),
                                node: node_id,
                            };
                            discharge_node_obligations(builder, cell, &node_ref, &node)?;
                            let child_graph = {
                                let context = cell.borrow();
                                context
                                    .child_graphs
                                    .get(&call_node.choice)
                                    .cloned()
                                    .expect("checked above")
                            };
                            let boundary = builder.canonical_call_boundary(call_node)?;
                            let import = builder.fuse_call(node_ref, &child_graph, boundary)?;
                            // Runtime extents are program-global and already
                            // present in the parent facts; only the child's
                            // remapped value/state facts are merged here.
                            let no_extents =
                                IdVec::<RuntimeExtentId, RuntimeExtent>::new(Vec::new());
                            let child_facts = GraphFacts::collect(&child_graph, &no_extents);
                            {
                                let mut context = cell.borrow_mut();
                                context
                                    .facts
                                    .types
                                    .extend(child_facts.types.into_iter().map(|(id, ty)| {
                                        (GraphValueId(id.0 + import.value_offset), ty)
                                    }));
                                context.facts.constants.extend(
                                    child_facts.constants.into_iter().map(|(id, value)| {
                                        (GraphValueId(id.0 + import.value_offset), value)
                                    }),
                                );
                                context
                                    .facts
                                    .states
                                    .extend(child_facts.states.into_iter().map(|(id, storage)| {
                                        (
                                            StateTokenId(id.0 + import.state_offset),
                                            LogicalStorageId(storage.0 + import.storage_offset),
                                        )
                                    }));
                            }
                            let child_path = import.root;
                            build_region(builder, cell, child_path)?;
                        } else {
                            invoke_node(builder, cell, path.clone(), node_id, &node, call_node)?;
                        }
                    }
                    LogicalNodeKind::Primitive(application) => {
                        let PrimitiveOp::Capability(intrinsic) = &application.op else {
                            unreachable!("a primitive node is never an executor segment")
                        };
                        map_matrix_node(builder, cell, path.clone(), node_id, &node, intrinsic)?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Whether a loop body holds only primitives and nested control (no call
/// or reduction): such a body can fuse into one kernel-local launch.
fn region_absorbable(region: &GraphRegion) -> bool {
    fn walk(region: &GraphRegion) -> bool {
        for node in region.nodes.iter() {
            match &node.kind {
                LogicalNodeKind::Primitive(_) => {}
                LogicalNodeKind::Loop(inner) => {
                    if !walk(&inner.body) {
                        return false;
                    }
                }
                LogicalNodeKind::If(inner) => {
                    if !walk(&inner.then_region) || !walk(&inner.else_region) {
                        return false;
                    }
                }
                LogicalNodeKind::Reduction(_) | LogicalNodeKind::Call(_) => {
                    return false;
                }
            }
        }
        true
    }
    walk(region)
}

// -- guards -------------------------------------------------------------------

/// One node's discharged obligations as emission-ready guards.
struct NodeGuards {
    /// Conjunctive index-bounds guard (all index obligations).
    index: Option<Guard>,
    /// Division guard (divisor nonzero and no signed overflow).
    division: Option<Guard>,
    shift: Option<Guard>,
}

/// Map one logical matrix intrinsic (`metal.matrix.matmul[_add]`): the
/// launch covers the output elements, one ascending-k fma chain per
/// participant, accumulating into the `into` operand. The result value
/// publishes through the into storage (the registry semantics: the result
/// is the updated accumulator).
fn map_matrix_node(
    builder: &mut AlternativeBuilder<MetalDialect>,
    cell: &std::cell::RefCell<StrategyContext<'_>>,
    path: RegionPath,
    node_id: NodeId,
    node: &LogicalNode,
    intrinsic: &IntrinsicId,
) -> Result<(), BuilderError> {
    let node_ref = NodeRef {
        region: path,
        node: node_id,
    };
    discharge_node_obligations(builder, cell, &node_ref, node)?;
    let context = cell.borrow();
    if !context.profile.effective_signatures.contains(intrinsic) {
        return Err(format!(
            "compiler bug: the effective profile lacks `{}` but its lowering reached mapping",
            intrinsic.path()
        )
        .into());
    }
    let tensor_of = |value: GraphValueId| -> Result<TensorType, BuilderError> {
        match context.facts.types.get(&value) {
            Some(ValueType::Tensor(tensor)) => Ok(tensor.clone()),
            _ => Err(format!("matrix operand value#{value:?} is not a rank-two tensor").into()),
        }
    };
    let left = tensor_of(node.inputs[0])?;
    let right = tensor_of(node.inputs[1])?;
    let result = tensor_of(node.outputs[0].id)?;
    if left.axes.len() != 2 || right.axes.len() != 2 || result.axes.len() != 2 {
        return Err("logical matrix operands must be rank two".into());
    }
    let dtype = match &result.elem {
        Elem::Dtype(dtype) => *dtype,
        _ => return Err("a matrix result is a dense dtype".into()),
    };
    let left_layout = access_layout_of(&context, node.inputs[0])?;
    let right_layout = access_layout_of(&context, node.inputs[1])?;
    let into_layout = access_layout_of(&context, node.inputs[2])?;
    let k = extent_addr(&left.axes[1])?;
    let op = MetalOp::MatrixMatmul {
        left: (0, left_layout),
        right: (1, right_layout),
        into: (2, into_layout),
        accumulate: intrinsic.name == "matmul_add",
        k,
        dtype,
        signature: intrinsic.clone(),
        signature_arguments: vec![
            ValueType::Tensor(left),
            ValueType::Tensor(right),
            ValueType::Tensor(result.clone()),
        ],
    };
    let iteration = LinearIterationMap::linear(
        &[result.axes[0].clone(), result.axes[1].clone()],
        &context.facts.runtime_extents,
    )
    .map_err(|error| format!("matrix geometry: {error}"))?
    .with_participants(context.participants.clone());
    drop(context);
    // The result value lives in the into operand's storage.
    let into_transport = builder.transport_of(node.inputs[2])?;
    builder.bind_value_transport(node.outputs[0].id, into_transport)?;
    builder.map_primitive(
        node_ref,
        iteration,
        Legalized::Ops(NonEmpty::new(vec![op]).expect("one opcode")),
    )
}

fn discharge_node_obligations(
    builder: &mut AlternativeBuilder<MetalDialect>,
    cell: &std::cell::RefCell<StrategyContext<'_>>,
    node_ref: &NodeRef,
    node: &LogicalNode,
) -> Result<NodeGuards, BuilderError> {
    let context = cell.borrow();
    let mut index_predicates = Vec::new();
    let mut index_status: Option<u64> = None;
    let mut division: Option<Guard> = None;
    let mut shift: Option<Guard> = None;
    let input_position = |value: GraphValueId| -> Option<usize> {
        node.inputs.iter().position(|input| *input == value)
    };
    let vref = |value: GraphValueId| -> Result<ValueRef, BuilderError> {
        input_position(value)
            .map(ValueRef::Operand)
            .or_else(|| context.ssa_values.get(&value).map(|n| ValueRef::Ssa(*n)))
            .ok_or_else(|| {
                format!("guard value#{value:?} is neither a launch operand nor kernel-local SSA")
            })
    };
    for (index, obligation) in node.safety.iter().enumerate() {
        let classification = terminal::discharge(obligation, &context.facts, node.span);
        let receipt = discharge_with_builder(
            builder,
            ObligationRef {
                node: node_ref.clone(),
                index,
            },
            &classification,
            |_| Legalized::Ops(NonEmpty::new(vec![MetalOp::View]).expect("inert predicate marker")),
        )?;
        let status = receipt.map(|field| field.0);
        match obligation {
            SafetyObligation::IndexInBounds { index, extent } => {
                if let Some(status) = status {
                    index_predicates.push(GuardPredicate::IndexInBounds {
                        index: vref(*index)?,
                        extent: extent_addr(extent)?,
                    });
                    index_status = Some(status);
                }
            }
            SafetyObligation::RangeInBounds { start, end, extent } => {
                if let Some(status) = status {
                    index_predicates.push(GuardPredicate::RangeInBounds {
                        start: vref(*start)?,
                        end: vref(*end)?,
                        extent: extent_addr(extent)?,
                    });
                    index_status = Some(status);
                }
            }
            SafetyObligation::DivisorNonZero { value } => {
                if let Some(status) = status {
                    division = Some(Guard {
                        predicates: vec![GuardPredicate::DivisorNonZero {
                            value: vref(*value)?,
                        }],
                        status,
                    });
                }
            }
            SafetyObligation::SignedDivisionNoOverflow { lhs, rhs } => {
                if let Some(status) = status {
                    division = Some(Guard {
                        predicates: vec![GuardPredicate::DivisionSafe {
                            lhs: vref(*lhs)?,
                            rhs: vref(*rhs)?,
                        }],
                        status,
                    });
                }
            }
            SafetyObligation::ShiftInRange { value } => {
                if let Some(status) = status {
                    shift = Some(Guard {
                        predicates: vec![GuardPredicate::ShiftInRange {
                            value: vref(*value)?,
                        }],
                        status,
                    });
                }
            }
            SafetyObligation::ShapeProductFits { factors, bits } => {
                // Shape products are checked at strategy time; a runtime
                // variant records its guard through the extent slot.
                let _ = (factors, bits);
            }
        };
        if matches!(classification, ObligationDischarge::StaticallyImpossible(_)) {
            // The alternative is inapplicable; `discharge_with_builder`
            // already returned the reason as an error.
        }
    }
    let index = index_status.map(|status| Guard {
        predicates: index_predicates,
        status,
    });
    // A statically-impossible classification surfaces as a BuilderError from
    // `discharge_with_builder`; nothing more is needed here.
    Ok(NodeGuards {
        index,
        division,
        shift,
    })
}

// -- fused primitive segments -------------------------------------------------

/// Fuse one maximal run of primitive nodes into a single launch: scalar SSA
/// stays kernel-local, tensor values stay storage-backed. The iteration map
/// is the first shaped domain among the run (grid-stride over the
/// participants parameter), else the serial single-visit map.
fn fuse_segment(
    builder: &mut AlternativeBuilder<MetalDialect>,
    cell: &std::cell::RefCell<StrategyContext<'_>>,
    path: RegionPath,
    entries: Vec<MetalEntry>,
) -> Result<(), BuilderError> {
    // The launch domain: the joined independent axes of the absorbed
    // region, or the first shaped domain among the primitives.
    let iteration = {
        let context = cell.borrow();
        if !context.axes.is_empty() {
            LinearIterationMap::linear(&context.axes, &context.facts.runtime_extents)
                .map_err(|e| e.to_string())?
                .with_participants(context.participants.clone())
        } else {
            let mut iteration: Option<LinearIterationMap> = None;
            for entry in &entries {
                if let (LogicalNodeKind::Primitive(_), Some(formed)) =
                    (&entry.logical.kind, entry.formed.as_ref())
                {
                    if let Some(domain) = &formed.iteration {
                        if iteration.is_none() {
                            let map = LinearIterationMap::linear(
                                &domain.extents,
                                &context.facts.runtime_extents,
                            )
                            .map_err(|e| e.to_string())?;
                            iteration = Some(map.with_participants(context.participants.clone()));
                        }
                    }
                }
            }
            iteration.unwrap_or_else(LinearIterationMap::serial)
        }
    };
    // A launch that updates storage atomically runs on one participant under
    // the serialized mapping; the device-atomic mapping keeps the domain.
    let iteration = if entries_update_atomically(&entries)
        && cell.borrow().atomics == AtomicMapping::Serialized
    {
        LinearIterationMap::serialized(&iteration)
    } else {
        iteration
    };
    // Discharge every obligation first; the receipts name the status fields
    // the guarded opcodes write.
    // Keyed by region-qualified node: absorbed bodies are separate regions
    // whose node ids restart, so a bare `NodeId` collides across nesting.
    let mut guards: BTreeMap<NodeRef, NodeGuards> = BTreeMap::new();
    fn collect_obligations(
        builder: &mut AlternativeBuilder<MetalDialect>,
        cell: &std::cell::RefCell<StrategyContext<'_>>,
        path: &RegionPath,
        entries: &[MetalEntry],
        guards: &mut BTreeMap<NodeRef, NodeGuards>,
    ) -> Result<(), BuilderError> {
        for entry in entries {
            guards.insert(
                entry.node.clone(),
                discharge_node_obligations(builder, cell, &entry.node, &entry.logical)?,
            );
            match &entry.kind {
                MEntryKind::Primitive | MEntryKind::Axis { .. } => {}
                MEntryKind::SerialLoop { body, .. } => {
                    collect_obligations(builder, cell, path, body, guards)?;
                }
                MEntryKind::Branch {
                    then_body,
                    else_body,
                    ..
                } => {
                    collect_obligations(builder, cell, path, then_body, guards)?;
                    collect_obligations(builder, cell, path, else_body, guards)?;
                }
            }
        }
        Ok(())
    }
    collect_obligations(builder, cell, &path, &entries, &mut guards)?;
    // Build the opcode block: global operand indices, sequential SSA numbers.
    {
        let mut context = cell.borrow_mut();
        context.next_ssa = 0;
        context.ssa_values.clear();
    }
    let mut operand_base = 0usize;
    let ops = entries_ops(
        builder,
        cell,
        &entries,
        &iteration,
        &mut guards,
        &mut operand_base,
    )?;
    let ops = NonEmpty::new(ops).ok_or("a fused segment has no opcode")?;
    let mut node_refs = Vec::new();
    fn collect_refs(entries: &[MetalEntry], refs: &mut Vec<NodeRef>) {
        for entry in entries {
            refs.push(entry.node.clone());
            match &entry.kind {
                MEntryKind::Primitive | MEntryKind::Axis { .. } => {}
                MEntryKind::SerialLoop { body, .. } => collect_refs(body, refs),
                MEntryKind::Branch {
                    then_body,
                    else_body,
                    ..
                } => {
                    collect_refs(then_body, refs);
                    collect_refs(else_body, refs);
                }
            }
        }
    }
    collect_refs(&entries, &mut node_refs);
    builder.fuse(
        node_refs,
        FusedStrategyTemplate {
            iteration,
            ops: Legalized::Ops(ops),
        },
    )
}

/// The op stream of one entry tree: guards and the opcode per primitive,
/// serial frames and branches around nested bodies. `operand_base` tracks
/// the launch-global operand index across the in-order traversal.
fn entries_ops(
    _builder: &mut AlternativeBuilder<MetalDialect>,
    cell: &std::cell::RefCell<StrategyContext<'_>>,
    entries: &[MetalEntry],
    iteration: &LinearIterationMap,
    guards: &mut BTreeMap<NodeRef, NodeGuards>,
    operand_base: &mut usize,
) -> Result<Vec<MetalOp>, BuilderError> {
    let mut ops = Vec::new();
    for entry in entries {
        match &entry.kind {
            MEntryKind::Primitive => {
                let formed = entry.formed.clone().ok_or_else(|| {
                    format!(
                        "node#{} reached op building without formation",
                        entry.node.node.0
                    )
                })?;
                let node_guard = guards.remove(&entry.node).ok_or_else(|| {
                    format!(
                        "region-qualified node {:?} has no discharged safety guards",
                        entry.node
                    )
                })?;
                let produced = {
                    let mut context = cell.borrow_mut();
                    ops_of_primitive(
                        &mut context,
                        &formed,
                        &entry.logical,
                        *operand_base,
                        node_guard,
                        iteration,
                    )?
                };
                ops.extend(produced);
                // The node's scalar output (primitives produce one SSA value
                // for their single scalar/index output) is kernel-local for
                // later nodes.
                {
                    let mut context = cell.borrow_mut();
                    if let Some(output) = entry.logical.outputs.first() {
                        if matches!(output.ty, ValueType::Scalar(_) | ValueType::Index { .. }) {
                            let ssa = context.next_ssa - 1;
                            context.ssa_values.insert(output.id, ssa);
                        }
                    }
                }
                *operand_base += entry.logical.inputs.len();
            }
            MEntryKind::Axis { .. } => {
                // The binder's axis coordinate was recorded at collection;
                // operand resolution reads it from the context.
                *operand_base += entry.logical.inputs.len();
            }
            MEntryKind::SerialLoop {
                binder,
                length,
                body,
            } => {
                let binder_ref = {
                    let mut context = cell.borrow_mut();
                    let ssa = context.next_ssa;
                    context.next_ssa += 1;
                    context.ssa_values.insert(*binder, ssa);
                    ValueRef::Ssa(ssa)
                };
                let frame_length = extent_addr(length)?;
                let body_ops = entries_ops(_builder, cell, body, iteration, guards, operand_base)?;
                ops.push(MetalOp::SerialFor {
                    binder: binder_ref,
                    length: frame_length,
                    body: body_ops,
                });
                *operand_base += entry.logical.inputs.len();
            }
            MEntryKind::Branch {
                condition,
                then_body,
                else_body,
            } => {
                let condition_ref = {
                    let context = cell.borrow();
                    context
                        .ssa_values
                        .get(condition)
                        .map(|ssa| ValueRef::Ssa(*ssa))
                        .or_else(|| {
                            context
                                .binder_axis
                                .get(condition)
                                .map(|axis| ValueRef::Axis(*axis))
                        })
                        .or_else(|| context.operand_values.get(condition).cloned())
                        .ok_or_else(|| {
                            format!("branch condition value#{condition:?} is not kernel-local")
                        })?
                };
                let then_ops =
                    entries_ops(_builder, cell, then_body, iteration, guards, operand_base)?;
                let else_ops =
                    entries_ops(_builder, cell, else_body, iteration, guards, operand_base)?;
                ops.push(MetalOp::Branch {
                    condition: condition_ref,
                    then_ops,
                    else_ops,
                });
                *operand_base += entry.logical.inputs.len();
            }
        }
    }
    Ok(ops)
}

/// The scalar source of one input: tensor inputs are element loads at the
/// iteration coordinates; scalar inputs are binding references.
fn src_of_input(
    context: &StrategyContext<'_>,
    node: &LogicalNode,
    local: usize,
    domain_rank: usize,
) -> Result<Src, BuilderError> {
    let value = node.inputs[local];
    // Values produced earlier in the same fused segment are kernel-local SSA;
    // absorbed independent-loop binders are axis coordinates.
    if let Some(axis) = context.binder_axis.get(&value) {
        return Ok(Src::Scalar(ValueRef::Axis(*axis)));
    }
    if let Some(ssa) = context.ssa_values.get(&value) {
        return Ok(Src::Scalar(ValueRef::Ssa(*ssa)));
    }
    let ty = context
        .facts
        .types
        .get(&value)
        .ok_or_else(|| format!("input value#{value:?} has no type"))?;
    if let ValueType::Tensor(_) = ty {
        let layout = access_layout_of(context, value)?;
        Ok(Src::Element {
            operand: local,
            layout,
            indices: (0..domain_rank).map(IndexRef::Axis).collect(),
        })
    } else {
        Ok(Src::Scalar(ValueRef::Operand(local)))
    }
}

fn dtype_of_value(context: &StrategyContext<'_>, value: GraphValueId) -> DType {
    context
        .facts
        .types
        .get(&value)
        .and_then(|ty| ty.scalar_dtype())
        .unwrap_or(DType::F32)
}

/// The opcodes of one formed primitive inside a fused segment.
fn ops_of_primitive(
    context: &mut StrategyContext<'_>,
    formed: &FormedPrimitive,
    node: &LogicalNode,
    operand_base: usize,
    guards: NodeGuards,
    iteration: &LinearIterationMap,
) -> Result<Vec<MetalOp>, BuilderError> {
    let inputs = &node.inputs;
    let operand = |local: usize| ValueRef::Operand(operand_base + local);
    for (local, value) in inputs.iter().enumerate() {
        // Absorbed binders resolve to their axis coordinate or serial
        // variable, not a launch operand.
        let reference = if let Some(axis) = context.binder_axis.get(value) {
            ValueRef::Axis(*axis)
        } else if let Some(ssa) = context.ssa_values.get(value) {
            ValueRef::Ssa(*ssa)
        } else {
            ValueRef::Operand(operand_base + local)
        };
        context.operand_values.insert(*value, reference);
    }
    let domain_rank = iteration.extents.len();
    let src = |context: &StrategyContext<'_>, local: usize| -> Result<Src, BuilderError> {
        src_of_input(context, node, local, domain_rank)
    };
    let next = |context: &mut StrategyContext<'_>| -> ValueRef {
        let n = context.next_ssa;
        context.next_ssa += 1;
        ValueRef::Ssa(n)
    };
    // If the node's result is a tensor, append a store of the computed scalar
    // into the produced tensor element.
    let store_op = |context: &mut StrategyContext<'_>,
                    value: ValueRef|
     -> Result<Option<MetalOp>, BuilderError> {
        let output = node
            .outputs
            .first()
            .ok_or_else(|| "an elementwise node has an output".to_string())?;
        if let ValueType::Tensor(_) = &output.ty {
            let layout = access_layout_of(context, output.id)?;
            let rank = layout.strides.len();
            Ok(Some(MetalOp::StoreResult {
                value,
                dst: Dst::Produced(output.id),
                layout,
                indices: (0..rank).map(IndexRef::Axis).collect(),
                dtype: dtype_of_value(context, output.id),
            }))
        } else {
            Ok(None)
        }
    };

    let mut ops = Vec::new();
    match &formed.op {
        PrimitiveOp::Constant(literal) => {
            let (value, dtype) = match literal {
                Literal::Int(v) => (ConstValue::Int(*v), DType::I32),
                Literal::Float(v) => (ConstValue::Float(v.to_bits()), DType::F32),
                Literal::Bool(v) => (ConstValue::Bool(*v), DType::Bool),
                Literal::ShapeParam(name) => {
                    // Specialized shape parameters resolve to their bound
                    // value through the graph's constants; the interpreter's
                    // value is retained by the logical layer.
                    return Err(format!(
                        "compiler bug: shape parameter `{name}` reached Metal strategy \
                         formation unresolved"
                    ));
                }
            };
            ops.push(MetalOp::Const {
                into: next(context),
                value,
                dtype,
            });
        }
        PrimitiveOp::RuntimeExtent(id) => {
            ops.push(MetalOp::ExtentOf {
                value: AddrExpr::Extent(*id),
                into: next(context),
            });
        }
        PrimitiveOp::Capability(intrinsic) => {
            return Err(capability_route_message(intrinsic, context.profile));
        }
        PrimitiveOp::Primitive(id) => match id {
            PrimitiveId::TuplePack | PrimitiveId::RangeMake => {
                // Aggregates are leaf-lowered; their leaf transports are the
                // produced slots (pending the resolver's produced-value
                // binding — see crate docs). The pack itself emits nothing.
                ops.push(MetalOp::View);
            }
            PrimitiveId::TupleGet(index) => {
                ops.push(MetalOp::TupleGet {
                    operand: operand_base,
                    index: *index,
                    into: next(context),
                });
            }
            PrimitiveId::RangeStart | PrimitiveId::RangeEnd => {
                ops.push(MetalOp::RangeEndpoint {
                    start: matches!(id, PrimitiveId::RangeStart),
                    operand: operand_base,
                    into: next(context),
                });
            }
            PrimitiveId::Select => {
                let dtype = dtype_of_value(context, node.outputs[0].id);
                let into = next(context);
                ops.push(MetalOp::Select {
                    condition: operand(0),
                    then: src(context, 1)?,
                    otherwise: src(context, 2)?,
                    into,
                    dtype,
                });
                if let Some(store) = store_op(context, into)? {
                    ops.push(store);
                }
            }
            PrimitiveId::Extent { axis } | PrimitiveId::ValidExtent { axis } => {
                let ty = context
                    .facts
                    .types
                    .get(&inputs[0])
                    .ok_or("an extent operand has a type")?;
                let value = match ty {
                    ValueType::Tensor(shape) => shape
                        .axes
                        .get(*axis)
                        .cloned()
                        .map(|extent| extent_addr(&extent))
                        .transpose()?
                        .unwrap_or(AddrExpr::Const(0)),
                    ValueType::Range { bound } => extent_addr(bound)?,
                    _ => AddrExpr::Const(0),
                };
                ops.push(MetalOp::ExtentOf {
                    value,
                    into: next(context),
                });
            }
            PrimitiveId::Unary(unary) => {
                use seismic_lang::syntax::ast::UnaryOp as U;
                let dtype = dtype_of_value(context, inputs[0]);
                let (op, into) = match unary {
                    U::Not => (UnaryOp::Not, next(context)),
                    U::BitNot => (UnaryOp::BitNot, next(context)),
                    U::Neg => (UnaryOp::Neg, next(context)),
                };
                ops.push(MetalOp::Unary {
                    op,
                    operand: src(context, 0)?,
                    into,
                    dtype,
                });
                if let Some(store) = store_op(context, into)? {
                    ops.push(store);
                }
            }
            PrimitiveId::Binary(binary) => {
                use seismic_lang::syntax::ast::BinaryOp as B;
                let dtype = dtype_of_value(context, inputs[0]);
                let lhs = src(context, 0)?;
                let rhs = src(context, 1)?;
                let into = next(context);
                let op = match binary {
                    B::Or => MetalOp::BoolLogic {
                        op: BoolOp::Or,
                        operands: vec![operand(0), operand(1)],
                        into,
                    },
                    B::And => MetalOp::BoolLogic {
                        op: BoolOp::And,
                        operands: vec![operand(0), operand(1)],
                        into,
                    },
                    B::Eq => MetalOp::Compare {
                        op: RelOp::Eq,
                        lhs,
                        rhs,
                        into,
                        dtype,
                    },
                    B::Ne => MetalOp::Compare {
                        op: RelOp::Ne,
                        lhs,
                        rhs,
                        into,
                        dtype,
                    },
                    B::Lt => MetalOp::Compare {
                        op: RelOp::Lt,
                        lhs,
                        rhs,
                        into,
                        dtype,
                    },
                    B::Le => MetalOp::Compare {
                        op: RelOp::Le,
                        lhs,
                        rhs,
                        into,
                        dtype,
                    },
                    B::Gt => MetalOp::Compare {
                        op: RelOp::Gt,
                        lhs,
                        rhs,
                        into,
                        dtype,
                    },
                    B::Ge => MetalOp::Compare {
                        op: RelOp::Ge,
                        lhs,
                        rhs,
                        into,
                        dtype,
                    },
                    B::BitOr => MetalOp::IntOp {
                        op: IntOp::BitOr,
                        lhs,
                        rhs,
                        into,
                        dtype,
                    },
                    B::BitXor => MetalOp::IntOp {
                        op: IntOp::BitXor,
                        lhs,
                        rhs,
                        into,
                        dtype,
                    },
                    B::BitAnd => MetalOp::IntOp {
                        op: IntOp::BitAnd,
                        lhs,
                        rhs,
                        into,
                        dtype,
                    },
                    B::Shl => MetalOp::Shift {
                        op: ShiftOp::Shl,
                        value: lhs,
                        amount: rhs,
                        into,
                        dtype,
                        guard: guards.shift,
                    },
                    B::Shr => MetalOp::Shift {
                        op: ShiftOp::Shr,
                        value: lhs,
                        amount: rhs,
                        into,
                        dtype,
                        guard: guards.shift,
                    },
                    B::Add | B::Sub | B::Mul => {
                        let (int_op, float_op) = match binary {
                            B::Add => (IntOp::Add, FloatArith::Add),
                            B::Sub => (IntOp::Sub, FloatArith::Sub),
                            _ => (IntOp::Mul, FloatArith::Mul),
                        };
                        if dtype.is_float() {
                            MetalOp::FloatOp {
                                op: float_op,
                                lhs,
                                rhs,
                                into,
                                dtype,
                            }
                        } else {
                            MetalOp::IntOp {
                                op: int_op,
                                lhs,
                                rhs,
                                into,
                                dtype,
                            }
                        }
                    }
                    B::Div | B::Rem => {
                        if dtype.is_float() {
                            MetalOp::FloatOp {
                                op: if matches!(binary, B::Div) {
                                    FloatArith::Div
                                } else {
                                    FloatArith::Rem
                                },
                                lhs,
                                rhs,
                                into,
                                dtype,
                            }
                        } else {
                            MetalOp::IntDivRem {
                                op: if matches!(binary, B::Div) {
                                    DivRemOp::Div
                                } else {
                                    DivRemOp::Rem
                                },
                                lhs,
                                rhs,
                                into,
                                dtype,
                                guard: guards.division,
                            }
                        }
                    }
                };
                ops.push(op);
                if let Some(store) = store_op(context, into)? {
                    ops.push(store);
                }
            }
            PrimitiveId::Cast(target) => {
                let from = dtype_of_value(context, inputs[0]);
                let into = next(context);
                ops.push(MetalOp::Cast {
                    from,
                    to: *target,
                    operand: src(context, 0)?,
                    into,
                });
                if let Some(store) = store_op(context, into)? {
                    ops.push(store);
                }
            }
            PrimitiveId::Math(math) => {
                let dtype = dtype_of_value(context, inputs[0]);
                let mut args = Vec::new();
                for local in 0..inputs.len() {
                    args.push(src(context, local)?);
                }
                let into = next(context);
                ops.push(MetalOp::Math {
                    op: *math,
                    args,
                    into,
                    dtype,
                    reference: SEISMIC_MATH,
                });
                if let Some(store) = store_op(context, into)? {
                    ops.push(store);
                }
            }
            PrimitiveId::TensorAlloc { .. } => {
                ops.push(MetalOp::Alloc);
            }
            PrimitiveId::Fill { value, dtype } => {
                let output = node.outputs.first().ok_or("a fill node has an output")?;
                let layout = access_layout_of(context, output.id)?;
                ops.push(MetalOp::LinearLoop {
                    op: LinearLoopOp::Fill,
                    source: None,
                    dst: Dst::Produced(output.id),
                    dst_layout: layout.clone(),
                    dtype: *dtype,
                    fill: Some(ConstValue::Float(value.to_bits())),
                });
            }
            PrimitiveId::Materialize | PrimitiveId::Clone | PrimitiveId::Load => {
                let loop_op = match id {
                    PrimitiveId::Materialize => LinearLoopOp::Materialize,
                    PrimitiveId::Clone => LinearLoopOp::Clone,
                    _ => LinearLoopOp::Load,
                };
                let output = node
                    .outputs
                    .first()
                    .ok_or("a snapshot node has an output")?;
                let dst_layout = access_layout_of(context, output.id)?;
                let source_layout = access_layout_of(context, inputs[0])?;
                ops.push(MetalOp::LinearLoop {
                    op: loop_op,
                    source: Some((0, source_layout)),
                    dst: Dst::Produced(output.id),
                    dst_layout,
                    dtype: dtype_of_value(context, output.id),
                    fill: None,
                });
            }
            PrimitiveId::CopyInto => {
                let source_layout = access_layout_of(context, inputs[1])?;
                let dst_layout = access_layout_of(context, inputs[0])?;
                ops.push(MetalOp::LinearLoop {
                    op: LinearLoopOp::Copy,
                    source: Some((1, source_layout)),
                    dst: Dst::Operand(0),
                    dst_layout,
                    dtype: dtype_of_value(context, inputs[1]),
                    fill: None,
                });
            }
            PrimitiveId::Decode => {
                return Err(
                    "compiler bug: packed decode emission is not implemented; the universal \
                     alternative cannot reach it"
                        .into(),
                );
            }
            PrimitiveId::PackedRead(plane) => {
                ops.push(MetalOp::PackedPlaneRead {
                    operand: operand_base,
                    plane: plane_ordinal(plane),
                    dtype: plane_dtype(plane),
                    into: next(context),
                });
            }
            PrimitiveId::Transpose | PrimitiveId::Reshape | PrimitiveId::SliceView { .. } => {
                ops.push(MetalOp::View);
            }
            PrimitiveId::ElementRead { arity } => {
                let layout = access_layout_of(context, inputs[0])?;
                let indices = (0..*arity)
                    .map(|k| IndexRef::Value(operand(1 + k)))
                    .collect();
                let repr = context.facts.types.get(&inputs[0]).and_then(|ty| match ty {
                    ValueType::Tensor(tensor) => match &tensor.elem {
                        Elem::Repr(name) => Some(name.clone()),
                        _ => None,
                    },
                    _ => None,
                });
                let op = match repr {
                    Some(repr) => MetalOp::PackedElementRead {
                        operand: operand_base,
                        layout,
                        indices,
                        repr,
                        into: next(context),
                        guard: guards.index,
                    },
                    None => MetalOp::ReadElement {
                        operand: operand_base,
                        layout,
                        indices,
                        dtype: dtype_of_value(context, node.outputs[0].id),
                        into: next(context),
                        guard: guards.index,
                    },
                };
                ops.push(op);
            }
            PrimitiveId::ElementWrite { arity } => {
                let layout = access_layout_of(context, inputs[0])?;
                let indices = (0..*arity)
                    .map(|k| IndexRef::Value(operand(1 + k)))
                    .collect();
                let value_local = inputs.len() - 1;
                ops.push(MetalOp::WriteElement {
                    operand: operand_base,
                    layout,
                    indices,
                    value: src(context, value_local)?,
                    dtype: dtype_of_value(context, inputs[value_local]),
                    guard: guards.index,
                });
            }
            PrimitiveId::Atomic { op, arity } => {
                let layout = access_layout_of(context, inputs[0])?;
                let indices = (0..*arity)
                    .map(|k| IndexRef::Value(operand(1 + k)))
                    .collect();
                let value_local = inputs.len() - 1;
                let dtype = dtype_of_value(context, inputs[value_local]);
                if !atomic_dtype(dtype) {
                    return Err(format!(
                        "compiler bug: atomic {} is undefined for {}",
                        op.name(),
                        dtype.name()
                    ));
                }
                let value = src(context, value_local)?;
                match context.atomics {
                    AtomicMapping::Serialized => ops.push(MetalOp::AtomicSerial {
                        op: *op,
                        operand: operand_base,
                        layout,
                        indices,
                        value,
                        dtype,
                        guard: guards.index,
                    }),
                    AtomicMapping::Device => {
                        if !device_atomic_dtype(dtype) {
                            return Err(format!(
                                "compiler bug: the device-atomic mapping admits 32-bit elements only, not {}",
                                dtype.name()
                            ));
                        }
                        ops.push(MetalOp::AtomicDevice {
                            op: *op,
                            operand: operand_base,
                            layout,
                            indices,
                            value,
                            dtype,
                            guard: guards.index,
                        })
                    }
                }
            }
            PrimitiveId::Reduce { .. } => {
                return Err(
                    "compiler bug: a reduction reached scalar legalization; reductions are \
                     consumed by map_reduction"
                        .into(),
                );
            }
        },
    }
    Ok(ops)
}

fn capability_route_message(intrinsic: &IntrinsicId, profile: &EffectiveTargetProfile) -> String {
    if profile.effective_signatures.contains(intrinsic) {
        format!(
            "compiler bug: capability `{}` is effective but has no Metal capability strategy \
             for this occurrence",
            intrinsic.path()
        )
    } else {
        format!(
            "no applicable implementation: exact capability signature `{}` is absent from the \
             effective Metal target profile (authored lowering removed before planning; the \
             portable body remains)",
            intrinsic.path()
        )
    }
}

fn plane_ordinal(plane: &PlaneField) -> usize {
    match plane {
        PlaneField::Words => 0,
        PlaneField::Scale => 1,
        PlaneField::Bias => 2,
        PlaneField::Coefficients => 3,
        PlaneField::ScaleFactor => 4,
        PlaneField::BiasFactor => 5,
    }
}

fn plane_dtype(plane: &PlaneField) -> DType {
    match plane {
        PlaneField::Words => DType::U32,
        PlaneField::Scale | PlaneField::Bias => DType::F16,
        PlaneField::Coefficients | PlaneField::ScaleFactor | PlaneField::BiasFactor => DType::F32,
    }
}

// -- views and addresses -------------------------------------------------------

/// The row-major access layout of one tensor value through its logical view
/// transform.
fn access_layout_of(
    context: &StrategyContext<'_>,
    value: GraphValueId,
) -> Result<AccessLayout, BuilderError> {
    let ty = context
        .facts
        .types
        .get(&value)
        .cloned()
        .ok_or_else(|| format!("value#{value:?} has no type"))?;
    let ValueType::Tensor(_) = &ty else {
        return Err(format!("value#{value:?} is not a tensor access"));
    };
    let ty = context
        .facts
        .types
        .get(&value)
        .cloned()
        .ok_or_else(|| format!("value#{value:?} has no type"))?;
    let ValueType::Tensor(shape) = &ty else {
        return Err(format!("value#{value:?} is not a tensor access"));
    };
    // Region parameters (and loop-body invariant copies) are identity views
    // of their own type shape; node outputs carry their view transforms.
    let strides = match view_of_value(context.graph, value) {
        Some(view) => row_major_strides(&view.shape.axes)?,
        None => row_major_strides(&shape.axes)?,
    };
    let mut offset = Vec::new();
    if let Some(view) = view_of_value(context.graph, value) {
        if let logical::ViewTransform::Slice { axes } = &view.transform {
            for (axis, slice) in axes.iter().enumerate() {
                if let logical::SliceAxis::Point(point) = slice {
                    let stride = strides.get(axis).cloned().unwrap_or(AddrExpr::Const(0));
                    let reference = context
                        .ssa_values
                        .get(point)
                        .map(|n| ValueRef::Ssa(*n))
                        .or_else(|| context.operand_values.get(point).copied())
                        .ok_or_else(|| {
                            format!(
                                "point-slice offset value#{point:?} is not kernel-local; \
                                 slice the value through a kernel-local computation"
                            )
                        })?;
                    offset.push((stride, reference));
                }
            }
        }
    }
    Ok(AccessLayout { strides, offset })
}

/// Locate the logical view backing one graph value (any region).
fn view_of_value(graph: &TaskGraph, value: GraphValueId) -> Option<logical::LogicalView> {
    fn scan(
        graph: &TaskGraph,
        region: &GraphRegion,
        value: GraphValueId,
    ) -> Option<logical::LogicalView> {
        for node in region.nodes.iter() {
            for output in node.outputs.iter() {
                if output.id == value {
                    if let Some(view) = output.view {
                        return graph.views.get(view).cloned();
                    }
                    return None;
                }
            }
            match &node.kind {
                LogicalNodeKind::Loop(loop_node) => {
                    if let Some(found) = scan(graph, &loop_node.body, value) {
                        return Some(found);
                    }
                }
                LogicalNodeKind::If(if_node) => {
                    if let Some(found) = scan(graph, &if_node.then_region, value) {
                        return Some(found);
                    }
                    if let Some(found) = scan(graph, &if_node.else_region, value) {
                        return Some(found);
                    }
                }
                _ => {}
            }
        }
        None
    }
    scan(graph, &graph.root, value)
}

fn row_major_strides(axes: &[ExtentExpr]) -> Result<Vec<AddrExpr>, BuilderError> {
    let mut strides = Vec::with_capacity(axes.len());
    let mut stride = AddrExpr::Const(1);
    for axis in axes.iter().rev() {
        strides.push(stride.clone());
        stride = AddrExpr::Mul(Box::new(stride), Box::new(extent_addr(axis)?));
    }
    strides.reverse();
    Ok(strides)
}

fn extent_addr(extent: &ExtentExpr) -> Result<AddrExpr, BuilderError> {
    match extent {
        ExtentExpr::Static(n) => Ok(AddrExpr::Const(*n)),
        ExtentExpr::Runtime(id) => Ok(AddrExpr::Extent(*id)),
        ExtentExpr::Sym(sym) => sym
            .as_constant()
            .and_then(|c| u64::try_from(c).ok())
            .map(AddrExpr::Const)
            .ok_or_else(|| "an unresolved symbolic extent survived specialization".into()),
    }
}

// -- reductions -----------------------------------------------------------------

fn map_reduction_node(
    builder: &mut AlternativeBuilder<MetalDialect>,
    cell: &std::cell::RefCell<StrategyContext<'_>>,
    path: RegionPath,
    node_id: NodeId,
    node: &LogicalNode,
    reduction: &ReductionNode,
) -> Result<(), BuilderError> {
    let node_ref = NodeRef {
        region: path,
        node: node_id,
    };
    discharge_node_obligations(builder, cell, &node_ref, node)?;
    let context = cell.borrow();
    let shaped = context
        .facts
        .types
        .get(&reduction.operand)
        .and_then(|ty| ty.shaped().cloned())
        .ok_or("a reduction operand is a tensor")?;
    let strategy = match context.reduction_mode {
        ReductionMode::Universal => reduction::universal(reduction, &shaped)
            .map_err(|reason| format!("compiler bug: {reason}"))?,
        ReductionMode::Subgroup => {
            let inner = seismic_realization::numerics::ReductionTopology::SerialAxis {
                axis: reduction.axis,
                length: shaped.axes[reduction.axis].clone(),
            };
            reduction::reassociable(
                reduction,
                &shaped,
                seismic_realization::numerics::ReductionTopology::Subgroup {
                    width: 32,
                    inner: Box::new(inner),
                },
                reduction::ReassociationAdmission::SourceUnordered,
            )
            .map_err(|error| format!("subgroup reduction alternative: {error:?}"))?
        }
        ReductionMode::Blocked => {
            // The interleaved cover: lane-strided ascending partial folds,
            // one-level tree combine over the 32 workgroup partials.
            let inner = seismic_realization::numerics::ReductionTopology::SerialAxis {
                axis: reduction.axis,
                length: shaped.axes[reduction.axis].clone(),
            };
            reduction::reassociable(
                reduction,
                &shaped,
                seismic_realization::numerics::ReductionTopology::Tree {
                    fan_in: 32,
                    depth: 1,
                    inner: Box::new(inner),
                },
                reduction::ReassociationAdmission::SourceUnordered,
            )
            .map_err(|error| format!("blocked reduction alternative: {error:?}"))?
        }
    };
    // Iteration over the outer coordinates, grid-stride over participants.
    let mut outer_axes = shaped.axes.clone();
    let axis_length = outer_axes.remove(reduction.axis);
    // The blocked cover's domain appends one lane axis; its participants are
    // the fixed lane count (one workgroup per output element's lanes).
    let (iteration_axes, participants) = match context.reduction_mode {
        ReductionMode::Blocked => {
            let mut axes = outer_axes.clone();
            axes.push(ExtentExpr::Static(32));
            (axes, Sym::constant(32))
        }
        _ => (outer_axes.clone(), context.participants.clone()),
    };
    let iteration = LinearIterationMap::linear(&iteration_axes, &context.facts.runtime_extents)
        .map_err(|e| e.to_string())?
        .with_participants(participants);
    let operand_layout = access_layout_of(&context, reduction.operand)?;
    // Operand coordinates: outer coordinates plus the reduced-axis variable.
    let mut operand_indices = Vec::new();
    let mut outer_position = 0;
    for position in 0..shaped.rank() {
        if position == reduction.axis {
            operand_indices.push(IndexRef::ReducedAxis);
        } else {
            operand_indices.push(IndexRef::Axis(outer_position));
            outer_position += 1;
        }
    }
    let result_value = node
        .outputs
        .first()
        .map(|output| output.id)
        .ok_or("a reduction has a result value")?;
    let result_transport = builder.transport_of(result_value)?;
    // A runtime-checked nonempty precondition gets a planned status field.
    let mut nonempty = None;
    for precondition in &strategy.preconditions {
        let reduction::ReductionPrecondition::NonEmpty { length, .. } = precondition;
        let classification =
            terminal::discharge_precondition(precondition, &context.facts, node.span);
        if let ObligationDischarge::RuntimeChecked(_) = classification {
            let field = builder.status_field();
            nonempty = Some(Guard {
                predicates: vec![GuardPredicate::ExtentPositive {
                    extent: extent_addr(length)?,
                }],
                status: field.0,
            });
        }
    }
    let result = match &reduction.result {
        ValueType::Scalar(_) | ValueType::Index { .. } => ReduceResult::Scalar {
            value: result_value,
        },
        ValueType::Tensor(result_shape) => {
            let result_layout = access_layout_of(&context, result_value)?;
            ReduceResult::Tensor {
                dst: Dst::Produced(result_value),
                layout: result_layout,
                indices: (0..result_shape.rank()).map(IndexRef::Axis).collect(),
                dtype: match &result_shape.elem {
                    Elem::Dtype(d) => *d,
                    _ => DType::F32,
                },
            }
        }
        _ => return Err("a reduction result is a scalar or tensor".into()),
    };
    let op = match context.reduction_mode {
        ReductionMode::Universal => MetalOp::ReduceSerial {
            op: reduction.op,
            operand: 0,
            layout: operand_layout,
            axis: reduction.axis,
            axis_length: extent_addr(&axis_length)?,
            input_dtype: input_dtype_of(&shaped)?,
            accumulator: strategy.accumulator,
            result,
            guard: nonempty,
        },
        ReductionMode::Subgroup => MetalOp::SubgroupReduce {
            op: reduction.op,
            operand: 0,
            layout: operand_layout,
            axis: reduction.axis,
            axis_length: extent_addr(&axis_length)?,
            input_dtype: input_dtype_of(&shaped)?,
            accumulator: strategy.accumulator,
            result,
            guard: nonempty,
        },
        ReductionMode::Blocked => MetalOp::BlockedReduce {
            op: reduction.op,
            operand: 0,
            layout: operand_layout,
            axis: reduction.axis,
            axis_length: extent_addr(&axis_length)?,
            lanes: 32,
            input_dtype: input_dtype_of(&shaped)?,
            accumulator: strategy.accumulator,
            result,
            nonempty,
        },
    };
    builder.map_reduction(
        node_ref,
        ReductionStrategyTemplate {
            topology: strategy.topology,
            iteration,
            ops: Legalized::Ops(NonEmpty::new(vec![op]).expect("one reduction opcode")),
            result: result_transport,
        },
    )
}

fn input_dtype_of(shaped: &TensorType) -> Result<DType, BuilderError> {
    match &shaped.elem {
        Elem::Dtype(d) => Ok(*d),
        Elem::Param(_) => Ok(DType::F32),
        Elem::Repr(r) => Err(format!("a packed representation `{r}` cannot be reduced")),
    }
}

/// The executor transport of one control scalar: a known constant becomes a
/// computed executor scalar (resolved to a retained constant expression by
/// the resolver); anything else keeps its recorded transport.
fn executor_transport(
    builder: &AlternativeBuilder<MetalDialect>,
    cell: &std::cell::RefCell<StrategyContext<'_>>,
    value: GraphValueId,
) -> Result<TransportTemplate, BuilderError> {
    let context = cell.borrow();
    if let Some(constant) = context.facts.constants.get(&value) {
        return Ok(TransportTemplate::ExecutorScalar(
            realization::ExecutorScalarTemplate {
                source: realization::ExecutorScalarSource::Computed(
                    realization::ExecutorComputedScalar::Const(*constant),
                ),
                dtype: DType::I32,
            },
        ));
    }
    builder.transport_of(value)
}

// -- loops -----------------------------------------------------------------------

fn schedule_loop_node(
    builder: &mut AlternativeBuilder<MetalDialect>,
    cell: &std::cell::RefCell<StrategyContext<'_>>,
    path: RegionPath,
    node_id: NodeId,
    node: &LogicalNode,
    loop_node: &LoopNode,
) -> Result<(), BuilderError> {
    let node_ref = NodeRef {
        region: path.clone(),
        node: node_id,
    };
    discharge_node_obligations(builder, cell, &node_ref, node)?;
    let start = executor_transport(builder, cell, loop_node.range.start)?;
    let end = executor_transport(builder, cell, loop_node.range.end)?;
    // Carried transports: the channel that survives across visits. State
    // carries transport through their storage; value carries through the
    // body result value's produced transport.
    let mut carried = Vec::new();
    for slot in &loop_node.carried {
        let transport = match slot.initial {
            seismic_lang::logical::RegionInput::State(token) => {
                let storage = builder
                    .logical_storage_of_token(token)
                    .ok_or_else(|| format!("carried token#{token:?} has no storage"))?;
                let template = builder
                    .storage_of(storage)
                    .ok_or_else(|| format!("carried storage#{storage:?} has no template"))?;
                TransportTemplate::Storage(
                    NonEmpty::new(vec![StorageViewTemplate {
                        storage: template,
                        access: Access::Exclusive,
                        transform: logical::ViewTransform::Identity,
                    }])
                    .expect("one plane"),
                )
            }
            seismic_lang::logical::RegionInput::Value(_) => {
                let result = loop_node
                    .body
                    .results
                    .get(slot.body_result.index())
                    .cloned()
                    .ok_or("a carried body result is absent")?;
                match result {
                    RegionResult::Value { id, .. } => builder.transport_of(id)?,
                    RegionResult::State { .. } => {
                        return Err("a value carry names a value body result".into());
                    }
                }
            }
        };
        carried.push(realization::PhysicalCarryTemplate { transport });
    }
    let body_path = {
        let mut p = path.clone();
        p.push(RegionStep::LoopBody(node_id));
        p
    };
    builder.schedule_loop(
        node_ref,
        ExecutorRangeTemplate {
            start,
            end,
            bound: loop_node.range.bound.clone(),
        },
        carried,
        |builder| build_region(builder, cell, body_path.clone()),
    )
}

// -- conditionals ------------------------------------------------------------------

fn schedule_if_node(
    builder: &mut AlternativeBuilder<MetalDialect>,
    cell: &std::cell::RefCell<StrategyContext<'_>>,
    path: RegionPath,
    node_id: NodeId,
    node: &LogicalNode,
    if_node: &IfNode,
) -> Result<(), BuilderError> {
    let node_ref = NodeRef {
        region: path.clone(),
        node: node_id,
    };
    discharge_node_obligations(builder, cell, &node_ref, node)?;
    let condition = executor_transport(builder, cell, if_node.condition)?;
    let mut joins = Vec::new();
    for slot in &if_node.joins {
        joins.push(match slot {
            JoinSlot::Value {
                then_result,
                else_result,
                joined,
                ..
            } => {
                let then_value = region_result_value(&if_node.then_region, then_result.index())?;
                let else_value = region_result_value(&if_node.else_region, else_result.index())?;
                PhysicalJoinTemplate::Value {
                    then: builder.transport_of(then_value)?,
                    else_branch: builder.transport_of(else_value)?,
                    joined: builder.transport_of(*joined)?,
                }
            }
            JoinSlot::State { storage, .. } => {
                let template = builder
                    .storage_of(*storage)
                    .ok_or_else(|| format!("joined storage#{storage:?} has no template"))?;
                PhysicalJoinTemplate::State { storage: template }
            }
        });
    }
    let then_path = {
        let mut p = path.clone();
        p.push(RegionStep::IfThen(node_id));
        p
    };
    let else_path = {
        let mut p = path.clone();
        p.push(RegionStep::IfElse(node_id));
        p
    };
    builder.schedule_if(
        node_ref,
        realization::ExecutorPredicateTemplate { value: condition },
        |builder| build_region(builder, cell, then_path.clone()),
        |builder| build_region(builder, cell, else_path.clone()),
        joins,
    )
}

fn region_result_value(region: &GraphRegion, ordinal: usize) -> Result<GraphValueId, BuilderError> {
    match region.results.get(ordinal) {
        Some(RegionResult::Value { id, .. }) => Ok(*id),
        _ => Err(format!("region result#{ordinal} is not a value")),
    }
}

// -- calls --------------------------------------------------------------------------

fn invoke_node(
    builder: &mut AlternativeBuilder<MetalDialect>,
    cell: &std::cell::RefCell<StrategyContext<'_>>,
    path: RegionPath,
    node_id: NodeId,
    node: &LogicalNode,
    _call_node: &logical::CallNode,
) -> Result<(), BuilderError> {
    let node_ref = NodeRef {
        region: path,
        node: node_id,
    };
    discharge_node_obligations(builder, cell, &node_ref, node)?;
    builder.invoke_canonical(node_ref)
}
