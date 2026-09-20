//! CPU physical planning: the sealed `CpuOp` opcode vocabulary, the
//! `CpuDialect` legalization of the full portable matrix, and
//! the universal CPU strategies over the common family builder.
//!
//! Universal portable `Inapplicable` is a compiler bug and never occurs: every
//! registry primitive legalizes to exactly one opcode. Capability applications
//! have no CPU opcode (this backend offers no backend intrinsics), so they
//! route to the capability path and are inapplicable on this target.
//!
//! Emission convention of `CpuOp` operand/result positions: a mapped launch
//! binds the consumed nodes' input values (node order, then input order —
//! including kernel-internal edges of a fused set) followed by their output
//! values. `operands`/`results` index that combined binding list.

use crate::mapping::{self, Limits, PARTICIPANTS_PARAMETER};
use seismic_compiler::{
    pipeline::{Backend, EncodedPlan},
    terminal::{
        self, discharge as classify_obligation, form_primitive, reduction_identity,
        universal_consequences, universal_node, CheckPredicate, GraphFacts, LayoutTransform,
        LinearIterationMap, LinearLoopOp, ObligationDischarge, ReductionIdentity,
        ReductionStrategy, SeismicMathReference, UniversalNode,
    },
};
use seismic_lang::syntax::ast::{BinaryOp, UnaryOp};
use seismic_lang::{
    intrinsics::{MathOp, PlaneField, PrimitiveId, ReduceOp},
    logical::{
        self, Access, GraphRegion, GraphValueId, JoinSlot, LogicalNode, LogicalNodeKind,
        LogicalProgram, LogicalStorageId, NodeId, PrimitiveOp, ReductionNode, RegionInput,
        RegionResult, RuntimeExtent, TaskGraph,
    },
    repr,
    sir::{Literal, LoopKind},
    sym::Sym,
    types::{DType, Elem, ExtentExpr, NonEmpty, RuntimeExtentId, TensorType, ValueType},
};
use seismic_realization::executable::{
    self as exec, AlternativeBuilder, BuilderError, EffectiveTargetProfile, ExecutableDialect,
    InvariantReport, Legalized, ObligationRef, PhysicalConsequences, PhysicalPrimitive, PlanFamily,
    PlanFamilyBuilder, PlanValues, RegionPath, ResolvedLaunch,
};
use seismic_realization::executable::{
    BoundaryLeaf, ExecutorPredicateTemplate, ExecutorRangeTemplate, FusedStrategyTemplate, NodeRef,
    PhysicalCarryTemplate, PhysicalJoinTemplate, ReductionStrategyTemplate, RegionStep,
    StorageViewTemplate, TransportTemplate,
};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// The sealed CPU opcode vocabulary
// ---------------------------------------------------------------------------

/// A typed constant in the S-value model (value bits + result dtype).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstValue {
    Int(i64),
    /// `f64` bits.
    FloatBits(u64),
    Bool(bool),
}

/// One planned runtime-check predicate. Operand positions index the launch
/// binding list; extent expressions are retained (static values or runtime
/// extents).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckKind {
    IndexInBounds {
        index: u16,
        extent: ExtentExpr,
    },
    RangeInBounds {
        start: u16,
        end: u16,
        extent: ExtentExpr,
    },
    DivisorNonZero {
        value: u16,
    },
    DivisionSafe {
        lhs: u16,
        rhs: u16,
    },
    ShiftInRange {
        value: u16,
    },
    ProductFits {
        factors: Vec<ExtentExpr>,
        bits: u8,
    },
    ExtentPositive {
        extent: ExtentExpr,
    },
}

/// The sealed CPU opcode enum: exhaustive over everything the CPU backend
/// emits. Cranelift codegen matches it without a wildcard.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CpuOpKind {
    /// A typed constant (S-value).
    Const {
        value: ConstValue,
        dtype: DType,
    },
    /// The retained runtime value of one runtime extent as an i32 S-value.
    RuntimeExtent {
        extent: RuntimeExtentId,
    },
    /// Write every input leaf into the tuple output binding.
    TuplePack,
    /// Read one leaf of a tuple binding.
    TupleGet {
        index: usize,
    },
    /// Write the two index leaves into the range output binding.
    RangeMake,
    /// Read the start leaf of a range binding.
    RangeStart,
    /// Read the end leaf of a range binding.
    RangeEnd,
    /// Scalar select over S-values of one dtype.
    Select {
        dtype: DType,
    },
    /// The retained extent of one axis (or its valid extent) as an i32
    /// S-value; the extent expression is embedded.
    ExtentOf {
        extent: ExtentExpr,
    },
    ValidExtentOf {
        extent: ExtentExpr,
    },
    /// Boolean not.
    Not,
    /// A comparison producing a bool S-value.
    Compare {
        op: BinaryOp,
        left: DType,
        right: DType,
    },
    /// Boolean and/or.
    BoolBinary {
        op: BinaryOp,
    },
    /// 32-bit wrapping integer unary op, reinterpreted at `dtype`.
    IntUnary {
        op: UnaryOp,
        dtype: DType,
    },
    /// 32-bit wrapping / Euclidean integer binary op.
    IntBinary {
        op: BinaryOp,
        dtype: DType,
    },
    /// Float arithmetic computed in binary64 and rounded once at `dtype`
    /// (the registry reference model).
    FloatUnary {
        op: UnaryOp,
        dtype: DType,
    },
    FloatBinary {
        op: BinaryOp,
        dtype: DType,
    },
    /// The versioned `seismic_math` software sequence: binary64 host std
    /// evaluation rounded once — the same code path the reference
    /// interpreter runs, so the bits agree by construction.
    SeismicMath {
        op: MathOp,
        dtype: DType,
        reference: SeismicMathReference,
    },
    /// Registry cast: integer↔integer preserves bits; everything else
    /// converts by value with the destination rounding.
    Cast {
        source: DType,
        target: DType,
    },
    /// A representation-layout view producer: the output binding aliases the
    /// operand storage; no runtime computation.
    LayoutAddress {
        transform: LayoutTransform,
    },
    /// Checked typed element load at explicit indices over `shape`.
    LoadElement {
        dtype: DType,
        shape: Vec<ExtentExpr>,
    },
    /// Checked typed element store at explicit indices over `shape`.
    StoreElement {
        dtype: DType,
        shape: Vec<ExtentExpr>,
    },
    /// Uninitialized storage declaration: planned, never emitted.
    StorageAllocation,
    /// An arbitrary-rank linear loop over elements (or planes). For `Fill`
    /// the constant is embedded as f64 bits.
    LinearElementLoop {
        op: LinearLoopOp,
        dtype: DType,
        shape: Vec<ExtentExpr>,
        fill_bits: u64,
    },
    /// Packed decode: dense f32 elements decoded from representation planes
    /// through the intrinsic registry.
    PackedDecode {
        repr: String,
        shape: Vec<ExtentExpr>,
    },
    /// Raw readable representation-plane element.
    PackedPlaneRead {
        plane: PlaneField,
        repr: String,
        shape: Vec<ExtentExpr>,
    },
    /// One decoded element of a packed view, addressed through its
    /// representation planes.
    PackedElementRead {
        repr: String,
        shape: Vec<ExtentExpr>,
    },
    /// Serialized exact atomic update: load / combine / round / store.
    SerializedAtomic {
        op: seismic_lang::intrinsics::AtomicOp,
        dtype: DType,
        shape: Vec<ExtentExpr>,
    },
    /// Concurrent atomic update from the worker domain: a compare/exchange
    /// loop on the element's 32-bit word (float `add` reassociates).
    AtomicDevice {
        op: seismic_lang::intrinsics::AtomicOp,
        dtype: DType,
        shape: Vec<ExtentExpr>,
    },
    /// A planned safety predicate: on failure it writes the first error into
    /// its status field and skips the guarded opcode (the next one).
    Check {
        kind: CheckKind,
        status: exec::StatusFieldId,
    },
    /// In-kernel serial loop (an ordered loop absorbed by fusion). The
    /// binder is the launch operand position of the loop variable.
    SerialFor {
        binder: u16,
        length: ExtentExpr,
        body: Vec<CpuOp>,
    },
    /// Binds one launch operand position to an iteration axis coordinate
    /// (an absorbed independent loop's binder).
    AxisBinder {
        position: u16,
        axis: usize,
    },
    /// In-kernel conditional (an `if` absorbed by fusion). The condition is
    /// the launch operand position of the predicate value.
    Branch {
        condition: u16,
        then_ops: Vec<CpuOp>,
        else_ops: Vec<CpuOp>,
    },
    /// The ordered universal reduction: ascending serial fold with registry
    /// accumulator/identity/tie semantics; parallel outer coordinates are
    /// the launch domain, one logical participant per output, one
    /// publication. The `argmax` nonempty precondition is retained: a zero
    /// length skips the fold and reports the first error.
    ReduceFold {
        op: ReduceOp,
        input: DType,
        accumulator: DType,
        /// The dtype of the reduced result (registry rule).
        result_dtype: DType,
        axis: usize,
        outer: Vec<ExtentExpr>,
        length: ExtentExpr,
        nonempty_precondition: bool,
    },
}

/// One CPU opcode with its operand and result binding positions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuOp {
    pub kind: CpuOpKind,
    /// Binding positions of the consumed node inputs, in node input order.
    pub operands: Vec<u16>,
    /// Binding positions the op writes (scalar slots, kernel values, or
    /// output storage elements).
    pub results: Vec<u16>,
}

impl CpuOp {
    fn new(kind: CpuOpKind, operands: Vec<u16>, results: Vec<u16>) -> Self {
        CpuOp {
            kind,
            operands,
            results,
        }
    }

    /// The opcode with re-anchored positions (used when a single-node
    /// legalization is placed inside a fused launch).
    fn with_positions(mut self, operands: Vec<u16>, results: Vec<u16>) -> Self {
        self.operands = operands;
        self.results = results;
        self
    }
}

// ---------------------------------------------------------------------------
// Layouts
// ---------------------------------------------------------------------------

/// A dense row-major plane, or the representation planes of one packed
/// tensor. The template retains only semantic shape; byte strides are the
/// canonical dense layout of each plane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CpuLayoutTemplate {
    Dense { dtype: DType, axes: Vec<ExtentExpr> },
    Packed { repr: String, axes: Vec<ExtentExpr> },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuResolvedPlane {
    pub name: String,
    pub dtype: DType,
    pub elements: u64,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CpuResolvedLayout {
    Dense {
        dtype: DType,
        shape: Vec<u64>,
        bytes: u64,
    },
    Packed {
        repr: String,
        values: u64,
        planes: Vec<CpuResolvedPlane>,
    },
}

impl CpuResolvedLayout {
    pub fn dense_dtype(&self) -> Option<DType> {
        match self {
            CpuResolvedLayout::Dense { dtype, .. } => Some(*dtype),
            CpuResolvedLayout::Packed { .. } => None,
        }
    }
}

// ---------------------------------------------------------------------------
// The dialect
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CpuDialect;

impl seismic_realization::executable::sealed::Sealed for CpuDialect {}

impl ExecutableDialect for CpuDialect {
    type Op = CpuOp;
    type LayoutTemplate = CpuLayoutTemplate;
    type ResolvedLayout = CpuResolvedLayout;

    fn legalize(p: &PhysicalPrimitive, t: &EffectiveTargetProfile) -> Legalized<CpuOp> {
        let _ = t;
        // Positional convention of the single-node mapping: bindings are the
        // node's inputs followed by its outputs.
        let input_len = u16::try_from(p.inputs.len()).unwrap_or(u16::MAX);
        let operands = (0..input_len).collect::<Vec<_>>();
        let results = (input_len..input_len + u16::try_from(p.results.len()).unwrap_or(0))
            .collect::<Vec<_>>();
        let scalar_of = |position: usize| p.inputs.get(position).and_then(scalar_dtype_of);
        let kind = match &p.op {
            logical::PrimitiveOp::Constant(literal) => {
                let dtype = p
                    .results
                    .first()
                    .and_then(scalar_dtype_of)
                    .unwrap_or(DType::I32);
                let value = match literal {
                    Literal::Int(v) => ConstValue::Int(*v),
                    Literal::Float(v) => ConstValue::FloatBits(v.to_bits()),
                    Literal::Bool(v) => ConstValue::Bool(*v),
                    Literal::ShapeParam(name) => {
                        return Legalized::Inapplicable {
                            reason: format!(
                                "the shape parameter `{name}` survived specialization \
                                 (compiler bug)"
                            ),
                        };
                    }
                };
                CpuOpKind::Const { value, dtype }
            }
            logical::PrimitiveOp::RuntimeExtent(id) => CpuOpKind::RuntimeExtent { extent: *id },
            logical::PrimitiveOp::Capability(intrinsic) => {
                return Legalized::Inapplicable {
                    reason: format!(
                        "the CPU backend implements no backend intrinsic `{}`; the portable \
                         reference body or the exact effective signature on another target \
                         must supply this alternative",
                        intrinsic.path()
                    ),
                };
            }
            logical::PrimitiveOp::Primitive(id) => match id {
                PrimitiveId::TuplePack => CpuOpKind::TuplePack,
                PrimitiveId::TupleGet(index) => CpuOpKind::TupleGet { index: *index },
                PrimitiveId::RangeMake => CpuOpKind::RangeMake,
                PrimitiveId::RangeStart => CpuOpKind::RangeStart,
                PrimitiveId::RangeEnd => CpuOpKind::RangeEnd,
                PrimitiveId::Select => CpuOpKind::Select {
                    dtype: p
                        .results
                        .first()
                        .and_then(scalar_dtype_of)
                        .or_else(|| scalar_of(1))
                        .unwrap_or(DType::F32),
                },
                PrimitiveId::Extent { axis } => CpuOpKind::ExtentOf {
                    extent: tensor_axis(&p.inputs, 0, *axis),
                },
                PrimitiveId::ValidExtent { axis } => CpuOpKind::ValidExtentOf {
                    extent: tensor_axis(&p.inputs, 0, *axis),
                },
                PrimitiveId::Unary(op) => match op {
                    UnaryOp::Not => CpuOpKind::Not,
                    UnaryOp::BitNot => CpuOpKind::IntUnary {
                        op: *op,
                        dtype: scalar_of(0).unwrap_or(DType::I32),
                    },
                    UnaryOp::Neg => {
                        let dtype = scalar_of(0).unwrap_or(DType::I32);
                        if dtype.is_float() {
                            CpuOpKind::FloatUnary { op: *op, dtype }
                        } else {
                            CpuOpKind::IntUnary { op: *op, dtype }
                        }
                    }
                },
                PrimitiveId::Binary(op) => match op {
                    BinaryOp::And | BinaryOp::Or => CpuOpKind::BoolBinary { op: *op },
                    BinaryOp::Eq
                    | BinaryOp::Ne
                    | BinaryOp::Lt
                    | BinaryOp::Le
                    | BinaryOp::Gt
                    | BinaryOp::Ge => CpuOpKind::Compare {
                        op: *op,
                        left: scalar_of(0).unwrap_or(DType::I32),
                        right: scalar_of(1).unwrap_or(DType::I32),
                    },
                    BinaryOp::BitOr
                    | BinaryOp::BitXor
                    | BinaryOp::BitAnd
                    | BinaryOp::Shl
                    | BinaryOp::Shr => CpuOpKind::IntBinary {
                        op: *op,
                        dtype: scalar_of(0).unwrap_or(DType::I32),
                    },
                    BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::Div
                    | BinaryOp::Rem => {
                        let left = scalar_of(0).unwrap_or(DType::I32);
                        let right = scalar_of(1).unwrap_or(DType::I32);
                        if left.is_float() || right.is_float() {
                            CpuOpKind::FloatBinary {
                                op: *op,
                                dtype: p
                                    .results
                                    .first()
                                    .and_then(scalar_dtype_of)
                                    .unwrap_or(DType::promote(left, right).unwrap_or(left)),
                            }
                        } else {
                            CpuOpKind::IntBinary {
                                op: *op,
                                dtype: left,
                            }
                        }
                    }
                },
                PrimitiveId::Cast(target) => CpuOpKind::Cast {
                    source: scalar_of(0).unwrap_or(DType::I32),
                    target: *target,
                },
                PrimitiveId::Math(op) => CpuOpKind::SeismicMath {
                    op: *op,
                    dtype: p
                        .results
                        .first()
                        .and_then(scalar_dtype_of)
                        .or_else(|| scalar_of(0))
                        .unwrap_or(DType::F32),
                    reference: SeismicMathReference::current(),
                },
                PrimitiveId::TensorAlloc { .. } => CpuOpKind::StorageAllocation,
                PrimitiveId::Fill { value, dtype } => CpuOpKind::LinearElementLoop {
                    op: LinearLoopOp::Fill,
                    dtype: *dtype,
                    shape: tensor_axes(&p.inputs, 0),
                    fill_bits: value.to_bits(),
                },
                PrimitiveId::Materialize => CpuOpKind::LinearElementLoop {
                    op: LinearLoopOp::Materialize,
                    dtype: dense_dtype(&p.inputs, 0),
                    shape: tensor_axes(&p.inputs, 0),
                    fill_bits: 0,
                },
                PrimitiveId::Clone => CpuOpKind::LinearElementLoop {
                    op: LinearLoopOp::Clone,
                    dtype: dense_dtype(&p.inputs, 0),
                    shape: tensor_axes(&p.inputs, 0),
                    fill_bits: 0,
                },
                PrimitiveId::Load => CpuOpKind::LinearElementLoop {
                    op: LinearLoopOp::Load,
                    dtype: dense_dtype(&p.inputs, 0),
                    shape: tensor_axes(&p.inputs, 0),
                    fill_bits: 0,
                },
                PrimitiveId::Decode => CpuOpKind::PackedDecode {
                    repr: packed_repr(&p.inputs, 0),
                    shape: tensor_axes(&p.inputs, 0),
                },
                PrimitiveId::PackedRead(plane) => CpuOpKind::PackedPlaneRead {
                    plane: *plane,
                    repr: packed_repr(&p.inputs, 0),
                    shape: tensor_axes(&p.inputs, 0),
                },
                PrimitiveId::Transpose => CpuOpKind::LayoutAddress {
                    transform: LayoutTransform::Transpose,
                },
                PrimitiveId::Reshape => CpuOpKind::LayoutAddress {
                    transform: LayoutTransform::Reshape,
                },
                PrimitiveId::SliceView { .. } => CpuOpKind::LayoutAddress {
                    transform: LayoutTransform::Slice,
                },
                PrimitiveId::ElementRead { .. } => {
                    if p.inputs.first().is_some_and(|ty| {
                        matches!(
                            ty,
                            ValueType::Tensor(TensorType {
                                elem: Elem::Repr(_),
                                ..
                            })
                        )
                    }) {
                        CpuOpKind::PackedElementRead {
                            repr: packed_repr(&p.inputs, 0),
                            shape: tensor_axes(&p.inputs, 0),
                        }
                    } else {
                        CpuOpKind::LoadElement {
                            dtype: dense_dtype(&p.inputs, 0),
                            shape: tensor_axes(&p.inputs, 0),
                        }
                    }
                }
                PrimitiveId::ElementWrite { .. } => CpuOpKind::StoreElement {
                    dtype: dense_dtype(&p.inputs, 0),
                    shape: tensor_axes(&p.inputs, 0),
                },
                PrimitiveId::CopyInto => CpuOpKind::LinearElementLoop {
                    op: LinearLoopOp::Copy,
                    dtype: dense_dtype(&p.inputs, 0),
                    shape: tensor_axes(&p.inputs, 0),
                    fill_bits: 0,
                },
                PrimitiveId::Atomic { op, .. } => CpuOpKind::SerializedAtomic {
                    op: *op,
                    dtype: scalar_value_dtype(&p.inputs),
                    shape: tensor_axes(&p.inputs, 0),
                },
                PrimitiveId::Reduce { op, .. } => {
                    return Legalized::Inapplicable {
                        reason: format!(
                            "the `{}` reduction reached scalar legalization (compiler bug); \
                             reductions are ReductionNodes consumed by map_reduction",
                            op.name()
                        ),
                    };
                }
            },
        };
        Legalized::Ops(
            NonEmpty::new(vec![CpuOp::new(kind, operands, results)]).expect("one opcode"),
        )
    }

    fn consequences(op: &CpuOp) -> PhysicalConsequences {
        // Exact hard resources of the universal mapping: no private or
        // workgroup bytes (large aggregates are planned arena/worker storage,
        // never implicit stack arrays), bounded code shape, no capability,
        // exact `direct_bindings`. The native contract is honest about
        // Cranelift: register allocation is opaque, so the admissible domain
        // admits any reflected maximum resident participant count of at
        // least one, and the resolved geometry `min(preferred, native_max)`
        // always keeps a resident participant.
        let mut consequences = universal_consequences(1, 0);
        // Uncalibrated CPU cost: ranking only, never legality.
        consequences.cost = exec::CostEstimate(match &op.kind {
            CpuOpKind::SeismicMath { .. } => 8,
            CpuOpKind::StorageAllocation | CpuOpKind::LayoutAddress { .. } => 0,
            // A compare/exchange round trip; contention is not modelled.
            CpuOpKind::AtomicDevice { .. } => 4,
            _ => 1,
        });
        // A concurrent float `add` combines in a data-dependent order.
        if let CpuOpKind::AtomicDevice {
            op: seismic_lang::intrinsics::AtomicOp::Add,
            dtype: DType::F32,
            ..
        } = &op.kind
        {
            consequences.numerical =
                seismic_realization::numerics::NumericalTransfer::Reassociate {
                    op: ReduceOp::Sum,
                    topology: seismic_realization::numerics::ReductionTopology::SerialAxis {
                        axis: 0,
                        length: ExtentExpr::Static(0),
                    },
                };
        }
        if let CpuOpKind::SeismicMath { reference, .. } = &op.kind {
            debug_assert_eq!(
                (reference.identity, reference.version),
                (
                    terminal::SEISMIC_MATH.identity,
                    terminal::SEISMIC_MATH.version
                ),
                "a math opcode must reference the current seismic_math version"
            );
        }
        consequences
    }

    fn public_layout(tensor: &TensorType) -> CpuLayoutTemplate {
        layout_of(tensor)
    }

    fn internal_layout(tensor: &TensorType) -> CpuLayoutTemplate {
        layout_of(tensor)
    }

    /// Layout template for a staged raw-byte workgroup/participant
    /// allocation: a word-aligned byte blob. The universal CPU strategy
    /// declares none; optimized CPU strategies may stage worker scratch
    /// through it.
    fn staged_layout(bytes: &seismic_lang::sym::Sym, alignment: u64) -> CpuLayoutTemplate {
        let words = bytes
            .clone()
            .add(&Sym::constant(
                i64::try_from(alignment.max(1) * 4 - 1).unwrap_or(3),
            ))
            .quot(&Sym::constant(4));
        CpuLayoutTemplate::Dense {
            dtype: DType::U32,
            axes: vec![ExtentExpr::Sym(words)],
        }
    }

    fn resolve_layout(
        layout: &CpuLayoutTemplate,
        values: &PlanValues,
    ) -> Result<CpuResolvedLayout, InvariantReport> {
        let extent = |expr: &ExtentExpr| -> Result<u64, InvariantReport> {
            match expr {
                ExtentExpr::Static(n) => Ok(*n),
                ExtentExpr::Sym(sym) => values.eval(sym),
                ExtentExpr::Runtime(_) => Err(InvariantReport(
                    "a runtime extent survived into a resolved layout".into(),
                )),
            }
        };
        Ok(match layout {
            CpuLayoutTemplate::Dense { dtype, axes } => {
                let mut shape = Vec::with_capacity(axes.len());
                let mut elements = 1u64;
                for axis in axes {
                    let n = extent(axis)?;
                    shape.push(n);
                    elements = elements.checked_mul(n).ok_or_else(|| {
                        InvariantReport("dense layout element total overflows".into())
                    })?;
                }
                let bytes = elements
                    .checked_mul(u64::from(dtype.bytes()))
                    .ok_or_else(|| InvariantReport("dense layout bytes overflow".into()))?;
                CpuResolvedLayout::Dense {
                    dtype: *dtype,
                    shape,
                    bytes,
                }
            }
            CpuLayoutTemplate::Packed { repr: name, axes } => {
                let representation = repr::lookup(name)
                    .ok_or_else(|| InvariantReport(format!("unknown representation `{name}`")))?;
                let mut count = 1u64;
                for axis in axes {
                    count = count.checked_mul(extent(axis)?).ok_or_else(|| {
                        InvariantReport("packed layout element total overflows".into())
                    })?;
                }
                let mut planes = Vec::new();
                for plane in representation.planes() {
                    let elements = plane.storage_elements(count).ok_or_else(|| {
                        InvariantReport(format!(
                            "plane `{}` element total overflows for `{name}`",
                            plane.name
                        ))
                    })?;
                    let bytes = plane.bytes(count).ok_or_else(|| {
                        InvariantReport(format!(
                            "plane `{}` byte total overflows for `{name}`",
                            plane.name
                        ))
                    })?;
                    planes.push(CpuResolvedPlane {
                        name: plane.name.to_string(),
                        dtype: plane.dtype(),
                        elements,
                        bytes,
                    });
                }
                CpuResolvedLayout::Packed {
                    repr: name.clone(),
                    values: count,
                    planes,
                }
            }
        })
    }
}

fn layout_of(tensor: &TensorType) -> CpuLayoutTemplate {
    match &tensor.elem {
        Elem::Dtype(dtype) => CpuLayoutTemplate::Dense {
            dtype: *dtype,
            axes: tensor.axes.clone(),
        },
        Elem::Repr(name) => CpuLayoutTemplate::Packed {
            repr: name.clone(),
            axes: tensor.axes.clone(),
        },
        // An unresolved element parameter reads as f32 (the registry's
        // portable-scope rule); specialization resolves it before layouts.
        Elem::Param(_) => CpuLayoutTemplate::Dense {
            dtype: DType::F32,
            axes: tensor.axes.clone(),
        },
    }
}

fn scalar_dtype_of(ty: &ValueType) -> Option<DType> {
    match ty {
        ValueType::Scalar(d) => Some(*d),
        ValueType::Index { .. } => Some(DType::I32),
        _ => None,
    }
}

fn tensor_of(inputs: &[ValueType], position: usize) -> Option<&TensorType> {
    inputs.get(position).and_then(|ty| match ty {
        ValueType::Tensor(tensor) => Some(tensor),
        _ => None,
    })
}

fn tensor_axes(inputs: &[ValueType], position: usize) -> Vec<ExtentExpr> {
    tensor_of(inputs, position)
        .map(|t| t.axes.clone())
        .unwrap_or_default()
}

fn tensor_axis(inputs: &[ValueType], position: usize, axis: usize) -> ExtentExpr {
    tensor_of(inputs, position)
        .and_then(|t| t.axes.get(axis).cloned())
        .unwrap_or(ExtentExpr::Static(0))
}

fn dense_dtype(inputs: &[ValueType], position: usize) -> DType {
    tensor_of(inputs, position)
        .map(|t| match &t.elem {
            Elem::Dtype(d) => *d,
            _ => DType::F32,
        })
        .unwrap_or(DType::F32)
}

fn packed_repr(inputs: &[ValueType], position: usize) -> String {
    tensor_of(inputs, position)
        .and_then(|t| match &t.elem {
            Elem::Repr(name) => Some(name.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

/// The dtype of the added value of an atomic: the scalar (non-index) input.
fn scalar_value_dtype(inputs: &[ValueType]) -> DType {
    inputs
        .iter()
        .find_map(|ty| match ty {
            ValueType::Scalar(d) => Some(*d),
            _ => None,
        })
        .unwrap_or(DType::F32)
}

// ---------------------------------------------------------------------------
// Universal CPU strategies over the common builder
// ---------------------------------------------------------------------------

/// How one component entry contributes to the fused launch.
enum EntryKind {
    /// A formed primitive node.
    Primitive,
    /// An absorbed independent loop: its axis joins the launch domain, its
    /// node is consumed by the fusion, and it contributes no opcode.
    Axis { binder: GraphValueId },
    /// An absorbed ordered loop: an in-kernel serial frame around its body.
    SerialLoop {
        binder: GraphValueId,
        length: ExtentExpr,
        body: Vec<ComponentEntry>,
    },
    /// An absorbed conditional: an in-kernel branch around its arm bodies.
    Branch {
        condition: GraphValueId,
        then_body: Vec<ComponentEntry>,
        else_body: Vec<ComponentEntry>,
    },
}

struct ComponentEntry {
    node: NodeRef,
    logical: LogicalNode,
    formed: Option<terminal::FormedPrimitive>,
    kind: EntryKind,
}

/// How a launch containing `atomic` updates is realized on the CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtomicMapping {
    /// The universal form: one worker traverses the launch domain and every
    /// update is the exact load/combine/round/store in visit order.
    Serialized,
    /// The concurrent form: the worker pool traverses the domain and every
    /// update is a compare/exchange loop on the element's 32-bit word.
    /// Float `add` reassociates.
    Device,
}

struct Strategy {
    facts: GraphFacts,
    runtime_extents: BTreeMap<RuntimeExtentId, RuntimeExtent>,
    participants: Sym,
    atomics: AtomicMapping,
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
                        let dtype = node
                            .inputs
                            .last()
                            .and_then(|value| facts.types.get(value))
                            .and_then(|ty| ty.scalar_dtype());
                        if !matches!(dtype, Some(DType::F32 | DType::I32 | DType::U32)) {
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
fn entries_update_atomically(entries: &[ComponentEntry]) -> bool {
    entries.iter().any(|entry| match &entry.kind {
        EntryKind::Primitive => matches!(
            &entry.logical.kind,
            LogicalNodeKind::Primitive(application)
                if matches!(application.op, PrimitiveOp::Primitive(PrimitiveId::Atomic { .. }))
        ),
        EntryKind::Axis { .. } => false,
        EntryKind::SerialLoop { body, .. } => entries_update_atomically(body),
        EntryKind::Branch {
            then_body,
            else_body,
            ..
        } => entries_update_atomically(then_body) || entries_update_atomically(else_body),
    })
}

/// Construct the complete CPU plan family: every applicable logical
/// alternative of every choice receives one universal physical alternative.
pub fn elaborate(
    logical: &LogicalProgram,
    limits: &Limits,
) -> Result<PlanFamily<CpuDialect>, String> {
    let mut family = PlanFamilyBuilder::<CpuDialect>::from_logical(logical)?;
    family.tuning_parameter(
        PARTICIPANTS_PARAMETER,
        1,
        i64::try_from(limits.workers).map_err(|_| "CPU worker count exceeds i64")?,
    )?;
    let participants = Sym::param(PARTICIPANTS_PARAMETER);
    for choice_id in logical.choices.ids() {
        let alternatives = logical.choice(choice_id).alternatives.len() as u32;
        for ordinal in 0..alternatives {
            let mut builder = family.alternative(choice_id, ordinal)?;
            build_alternative(
                &mut builder,
                logical,
                participants.clone(),
                AtomicMapping::Serialized,
            )?;
            let alternative = builder
                .finish_alternative()
                .map_err(|error| format!("CPU universal alternative: {error}"))?;
            family
                .add_alternative(choice_id, alternative)
                .map_err(|error| format!("CPU family assembly: {error}"))?;
            // Device-atomic alternative: launches with `atomic` updates keep
            // the worker domain and combine through compare/exchange.
            let graph = logical
                .graph(logical.choice(choice_id).alternatives.as_slice()[ordinal as usize].graph);
            let facts = GraphFacts::collect(graph, &logical.runtime_extents);
            if atomics_admitted(graph, &facts) {
                let mut builder = family.alternative(choice_id, ordinal)?;
                build_alternative(
                    &mut builder,
                    logical,
                    participants.clone(),
                    AtomicMapping::Device,
                )?;
                let alternative = builder
                    .finish_alternative()
                    .map_err(|error| format!("CPU device-atomic alternative: {error}"))?;
                family
                    .add_alternative(choice_id, alternative)
                    .map_err(|error| format!("CPU family assembly: {error}"))?;
            }
        }
    }
    family
        .finish()
        .map_err(|error| format!("CPU family finish: {error}"))
}

fn build_alternative(
    builder: &mut AlternativeBuilder<CpuDialect>,
    logical: &LogicalProgram,
    participants: Sym,
    atomics: AtomicMapping,
) -> Result<(), BuilderError> {
    let graph = builder.graph().clone();
    let facts = GraphFacts::collect(&graph, &logical.runtime_extents);
    let runtime_extents = logical
        .runtime_extents
        .ids()
        .zip(logical.runtime_extents.iter())
        .map(|(id, extent)| (id, extent.clone()))
        .collect::<BTreeMap<_, _>>();
    let strategy = Strategy {
        facts,
        runtime_extents,
        participants,
        atomics,
    };
    strategy.region(builder, &graph, &Vec::new())?;
    // Complete the graph boundary results in canonical order.
    for (ordinal, result) in graph.results.iter().enumerate() {
        match result {
            RegionResult::Value { id, .. } => {
                let transport = builder.transport_of(*id)?;
                builder.complete_result(ordinal as u32, transport)?;
            }
            RegionResult::State { storage, .. } => {
                let transport = builder.transport_of_storage(*storage, Access::Shared)?;
                builder.complete_result(ordinal as u32, transport)?;
            }
        }
    }
    Ok(())
}

/// The canonical boundary leaf of one storage (its parameter/result origin).
fn boundary_leaf(graph: &TaskGraph, storage: LogicalStorageId) -> BoundaryLeaf {
    graph
        .storages
        .get(storage)
        .map(|s| match &s.origin {
            logical::StorageOrigin::Parameter { ordinal, path, .. } => BoundaryLeaf::Input {
                param: *ordinal,
                leaf: path.clone(),
            },
            logical::StorageOrigin::Result { path, .. } => {
                BoundaryLeaf::Result { leaf: path.clone() }
            }
            logical::StorageOrigin::Owned => BoundaryLeaf::default(),
        })
        .unwrap_or_default()
}

/// Whether a loop body holds only primitives and nested control (no call
/// or reduction): such a body can fuse into one kernel-local launch.
fn absorbable(region: &GraphRegion) -> bool {
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

/// Navigate to one region of a task graph by region path.
fn region_at<'g>(
    graph: &'g TaskGraph,
    path: &RegionPath,
) -> Result<&'g logical::GraphRegion, String> {
    let mut region = &graph.root;
    for step in path {
        let node = region
            .nodes
            .get(step.node())
            .ok_or_else(|| format!("region path names absent node#{}", step.node().0))?;
        match (&node.kind, step) {
            (LogicalNodeKind::If(if_node), RegionStep::IfThen(_)) => region = &if_node.then_region,
            (LogicalNodeKind::If(if_node), RegionStep::IfElse(_)) => region = &if_node.else_region,
            (LogicalNodeKind::Loop(loop_node), RegionStep::LoopBody(_)) => region = &loop_node.body,
            _ => return Err("region path disagrees with graph structure".into()),
        }
    }
    Ok(region)
}

/// One collected region segment: a fused component, or a structural node
/// that requires the executor transitions.
enum Segment {
    /// One fused launch over the collected entries; `axes` are the joined
    /// independent-loop extents of the absorbed region (empty for a plain
    /// domain-keyed component).
    Component {
        axes: Vec<ExtentExpr>,
        entries: Vec<ComponentEntry>,
    },
    Executor(NodeId, LogicalNode),
}

/// Emit each collected group as one component segment, in group order.
fn flush_groups(
    groups: BTreeMap<Option<String>, Vec<ComponentEntry>>,
    axes: Vec<ExtentExpr>,
) -> Vec<Segment> {
    groups
        .into_iter()
        .filter(|(_, entries)| !entries.is_empty())
        .map(|(_, entries)| Segment::Component {
            axes: axes.clone(),
            entries,
        })
        .collect()
}

impl Strategy {
    /// Consume every node of one region: collect its segments (absorbing
    /// loops and conditionals whose bodies hold only primitives), then
    /// commit each in order.
    fn region(
        &self,
        builder: &mut AlternativeBuilder<CpuDialect>,
        graph: &TaskGraph,
        path: &RegionPath,
    ) -> Result<(), BuilderError> {
        let segments = self.collect(graph, path, &mut Vec::new(), false)?;
        self.commit(builder, graph, path, segments)
    }

    /// Collect the component groups of one region. Independent loops with
    /// primitive-only bodies are absorbed: their axis joins the launch
    /// domain and the loop node is consumed by the fusion. Ordered loops
    /// and conditionals inside an absorbed region become in-kernel serial
    /// frames and branches; anything retaining a call or reduction uses the
    /// structured executor transitions.
    fn collect(
        &self,
        graph: &TaskGraph,
        path: &RegionPath,
        axes: &mut Vec<ExtentExpr>,
        absorbing: bool,
    ) -> Result<Vec<Segment>, BuilderError> {
        let region = region_at(graph, path).map_err(BuilderError::from)?;
        let mut segments: Vec<Segment> = Vec::new();
        let mut groups: BTreeMap<Option<String>, Vec<ComponentEntry>> = BTreeMap::new();
        let absorbing_key = absorbing.then(|| "absorbed-region".to_string());
        for (node_id, node) in region.nodes.ids().zip(region.nodes.iter()) {
            let node_ref = NodeRef {
                region: path.clone(),
                node: node_id,
            };
            match &node.kind {
                LogicalNodeKind::Primitive(_) => {
                    let formed = form_primitive(node_id, node, &self.facts)
                        .map_err(|error| format!("CPU formation: {error}"))?;
                    let key = if absorbing {
                        // One component per absorbed region: the launch
                        // domain is the joined axes, not per-node domains.
                        absorbing_key.clone()
                    } else {
                        formed
                            .iteration
                            .as_ref()
                            .map(|map| format!("{:?}", map.extents))
                    };
                    groups.entry(key).or_default().push(ComponentEntry {
                        node: node_ref,
                        logical: node.clone(),
                        formed: Some(formed),
                        kind: EntryKind::Primitive,
                    });
                }
                LogicalNodeKind::Loop(loop_node) => {
                    let absorbable = absorbable(&loop_node.body);
                    match (loop_node.kind, absorbable, absorbing) {
                        (LoopKind::Independent, true, _) => {
                            // The axis joins the launch domain; the body's
                            // nodes join the same absorbed component.
                            axes.push(loop_node.range.bound.clone());
                            groups
                                .entry(absorbing_key.clone())
                                .or_default()
                                .push(ComponentEntry {
                                    node: node_ref,
                                    logical: node.clone(),
                                    formed: None,
                                    kind: EntryKind::Axis {
                                        binder: loop_node.binder,
                                    },
                                });
                            let body_path = {
                                let mut p = path.clone();
                                p.push(RegionStep::LoopBody(node_id));
                                p
                            };
                            let body_segments = self.collect(graph, &body_path, axes, true)?;
                            for segment in body_segments {
                                match segment {
                                    Segment::Component { entries, .. } => {
                                        groups
                                            .entry(absorbing_key.clone())
                                            .or_default()
                                            .extend(entries);
                                    }
                                    Segment::Executor(..) => {
                                        return Err(BuilderError::from(
                                            "an absorbable body retained an executor node"
                                                .to_string(),
                                        ));
                                    }
                                }
                            }
                            // Commit while the absorbed axis is still in scope;
                            // after the pop it is no longer part of the parent
                            // region's launch domain.
                            segments
                                .extend(flush_groups(std::mem::take(&mut groups), axes.clone()));
                            axes.pop();
                        }
                        (LoopKind::Ordered, true, true) => {
                            // An in-kernel serial frame around the body.
                            let body_path = {
                                let mut p = path.clone();
                                p.push(RegionStep::LoopBody(node_id));
                                p
                            };
                            let body_segments = self.collect(graph, &body_path, axes, true)?;
                            let mut body = Vec::new();
                            for segment in body_segments {
                                match segment {
                                    Segment::Component { entries, .. } => body.extend(entries),
                                    Segment::Executor(..) => {
                                        return Err(BuilderError::from(
                                            "an absorbable body retained an executor node"
                                                .to_string(),
                                        ));
                                    }
                                }
                            }
                            groups
                                .entry(absorbing_key.clone())
                                .or_default()
                                .push(ComponentEntry {
                                    node: node_ref,
                                    logical: node.clone(),
                                    formed: None,
                                    kind: EntryKind::SerialLoop {
                                        binder: loop_node.binder,
                                        length: loop_node.range.bound.clone(),
                                        body,
                                    },
                                });
                        }
                        _ => {
                            // Executor `Repeat` with the retained range; the
                            // body is walked as its own region.
                            segments
                                .extend(flush_groups(std::mem::take(&mut groups), axes.clone()));
                            segments.push(Segment::Executor(node_id, node.clone()));
                        }
                    }
                }
                LogicalNodeKind::If(if_node) if absorbing => {
                    // An in-kernel branch around the arm bodies.
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
                    let flatten = |segments: Vec<Segment>| {
                        let mut flat = Vec::new();
                        for segment in segments {
                            match segment {
                                Segment::Component { entries, .. } => flat.extend(entries),
                                Segment::Executor(..) => {
                                    return Err(BuilderError::from(
                                        "an absorbable body retained an executor node".to_string(),
                                    ));
                                }
                            }
                        }
                        Ok(flat)
                    };
                    let then_body = flatten(self.collect(graph, &then_path, axes, true)?)?;
                    let else_body = flatten(self.collect(graph, &else_path, axes, true)?)?;
                    groups
                        .entry(absorbing_key.clone())
                        .or_default()
                        .push(ComponentEntry {
                            node: node_ref,
                            logical: node.clone(),
                            formed: None,
                            kind: EntryKind::Branch {
                                condition: if_node.condition,
                                then_body,
                                else_body,
                            },
                        });
                }
                LogicalNodeKind::If(_)
                | LogicalNodeKind::Reduction(_)
                | LogicalNodeKind::Call(_) => {
                    segments.extend(flush_groups(std::mem::take(&mut groups), axes.clone()));
                    segments.push(Segment::Executor(node_id, node.clone()));
                }
            }
        }
        segments.extend(flush_groups(groups, axes.clone()));
        Ok(segments)
    }

    /// Commit each collected segment in order.
    fn commit(
        &self,
        builder: &mut AlternativeBuilder<CpuDialect>,
        graph: &TaskGraph,
        path: &RegionPath,
        segments: Vec<Segment>,
    ) -> Result<(), BuilderError> {
        for segment in segments {
            match segment {
                Segment::Component { axes, entries } => {
                    if !entries.is_empty() {
                        self.component(builder, axes, entries)?;
                    }
                }
                Segment::Executor(node_id, node) => match &node.kind {
                    LogicalNodeKind::Reduction(reduction) => {
                        self.reduction(builder, path, node_id, &node, reduction)?;
                    }
                    LogicalNodeKind::Loop(_) => {
                        self.loop_node(builder, graph, path, node_id, &node)?;
                    }
                    LogicalNodeKind::If(_) => {
                        self.if_node(builder, graph, path, node_id, &node)?;
                    }
                    LogicalNodeKind::Call(_) => {
                        self.call(builder, graph, path, node_id, &node)?;
                    }
                    LogicalNodeKind::Primitive(_) => {
                        unreachable!("a primitive node is never an executor segment")
                    }
                },
            }
        }
        Ok(())
    }

    /// One fused universal launch over one component. Absorbed regions
    /// launch over their joined independent axes; plain components over
    /// their shared elementwise domain (or a single-visit serial map).
    fn component(
        &self,
        builder: &mut AlternativeBuilder<CpuDialect>,
        axes: Vec<ExtentExpr>,
        entries: Vec<ComponentEntry>,
    ) -> Result<(), BuilderError> {
        // Predicted binding layout (the builder's contract): every node's
        // inputs in order, then every node's outputs in order, over the
        // whole entry tree.
        let mut input_values: Vec<GraphValueId> = Vec::new();
        let mut output_values: Vec<GraphValueId> = Vec::new();
        fn walk_values(
            entries: &[ComponentEntry],
            inputs: &mut Vec<GraphValueId>,
            outputs: &mut Vec<GraphValueId>,
        ) {
            for entry in entries {
                inputs.extend(entry.logical.inputs.iter().copied());
                outputs.extend(entry.logical.outputs.iter().map(|output| output.id));
                match &entry.kind {
                    EntryKind::Primitive | EntryKind::Axis { .. } => {}
                    EntryKind::SerialLoop { body, .. } => walk_values(body, inputs, outputs),
                    EntryKind::Branch {
                        then_body,
                        else_body,
                        ..
                    } => {
                        walk_values(then_body, inputs, outputs);
                        walk_values(else_body, inputs, outputs);
                    }
                }
            }
        }
        walk_values(&entries, &mut input_values, &mut output_values);
        let input_count = input_values.len() as u16;
        let position_of = |value: GraphValueId| -> Option<u16> {
            input_values
                .iter()
                .position(|candidate| *candidate == value)
                .map(|index| index as u16)
                .or_else(|| {
                    output_values
                        .iter()
                        .position(|candidate| *candidate == value)
                        .map(|index| input_count + index as u16)
                })
        };
        let iteration = if !axes.is_empty() {
            LinearIterationMap::linear(&axes, &self.runtime_extents)
                .map_err(|error| format!("CPU geometry: {error}"))?
                .with_participants(self.participants.clone())
        } else {
            let domain = entries.iter().find_map(|entry| {
                entry
                    .formed
                    .as_ref()
                    .and_then(|formed| formed.iteration.as_ref().map(|map| map.extents.clone()))
            });
            match domain {
                Some(extents) => LinearIterationMap::linear(&extents, &self.runtime_extents)
                    .map_err(|error| format!("CPU geometry: {error}"))?
                    .with_participants(self.participants.clone()),
                None => LinearIterationMap::serial(),
            }
        };
        // A launch that updates storage atomically runs on one worker under
        // the serialized mapping; the device-atomic mapping keeps the domain.
        let iteration =
            if self.atomics == AtomicMapping::Serialized && entries_update_atomically(&entries) {
                LinearIterationMap::serialized(&iteration)
            } else {
                iteration
            };
        let mut node_refs = Vec::new();
        fn collect_refs(entries: &[ComponentEntry], refs: &mut Vec<NodeRef>) {
            for entry in entries {
                refs.push(entry.node.clone());
                match &entry.kind {
                    EntryKind::Primitive | EntryKind::Axis { .. } => {}
                    EntryKind::SerialLoop { body, .. } => collect_refs(body, refs),
                    EntryKind::Branch {
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
        let ops = self.entries_ops(builder, &entries, &position_of)?;
        let ops = NonEmpty::new(ops).ok_or_else(|| "a CPU launch has no opcode".to_string())?;
        builder.fuse(
            node_refs,
            FusedStrategyTemplate {
                iteration,
                ops: Legalized::Ops(ops),
            },
        )
    }

    /// The op stream of one entry tree: planned checks and the opcode for
    /// each primitive, serial frames and branches around nested bodies.
    fn entries_ops(
        &self,
        builder: &mut AlternativeBuilder<CpuDialect>,
        entries: &[ComponentEntry],
        position_of: &impl Fn(GraphValueId) -> Option<u16>,
    ) -> Result<Vec<CpuOp>, BuilderError> {
        let mut next_axis = 0usize;
        self.entries_ops_with_axes(builder, entries, position_of, &mut next_axis)
    }

    fn entries_ops_with_axes(
        &self,
        builder: &mut AlternativeBuilder<CpuDialect>,
        entries: &[ComponentEntry],
        position_of: &impl Fn(GraphValueId) -> Option<u16>,
        next_axis: &mut usize,
    ) -> Result<Vec<CpuOp>, BuilderError> {
        let mut ops = Vec::new();
        for entry in entries {
            // The node's own safety obligations, discharged in order.
            for (index, obligation) in entry.logical.safety.iter().enumerate() {
                let classified = classify_obligation(obligation, &self.facts, entry.logical.span);
                let receipt = terminal::discharge_with_builder(
                    builder,
                    ObligationRef {
                        node: entry.node.clone(),
                        index,
                    },
                    &classified,
                    |_| Legalized::Inapplicable {
                        reason: "the predicate is planned into the owning launch".into(),
                    },
                )?;
                if let (ObligationDischarge::RuntimeChecked(check), Some(status)) =
                    (classified, receipt)
                {
                    let kind = self.check_kind(&check.predicate, position_of)?;
                    ops.push(CpuOp::new(
                        CpuOpKind::Check { kind, status },
                        Vec::new(),
                        Vec::new(),
                    ));
                }
            }
            match &entry.kind {
                EntryKind::Primitive => {
                    let op = self.primitive_op(entry, position_of)?;
                    ops.push(op);
                }
                EntryKind::Axis { binder } => {
                    let position = position_of(*binder).ok_or_else(|| {
                        format!("axis binder value#{} has no binding position", binder.0)
                    })?;
                    ops.push(CpuOp::new(
                        CpuOpKind::AxisBinder {
                            position,
                            axis: *next_axis,
                        },
                        Vec::new(),
                        Vec::new(),
                    ));
                    *next_axis += 1;
                }
                EntryKind::SerialLoop {
                    binder,
                    length,
                    body,
                } => {
                    let binder_position = position_of(*binder).ok_or_else(|| {
                        format!("serial binder value#{} has no binding position", binder.0)
                    })?;
                    let body_ops =
                        self.entries_ops_with_axes(builder, body, position_of, next_axis)?;
                    ops.push(CpuOp::new(
                        CpuOpKind::SerialFor {
                            binder: binder_position,
                            length: length.clone(),
                            body: body_ops,
                        },
                        Vec::new(),
                        Vec::new(),
                    ));
                }
                EntryKind::Branch {
                    condition,
                    then_body,
                    else_body,
                } => {
                    let condition_position = position_of(*condition).ok_or_else(|| {
                        format!(
                            "branch condition value#{} has no binding position",
                            condition.0
                        )
                    })?;
                    let then_ops =
                        self.entries_ops_with_axes(builder, then_body, position_of, next_axis)?;
                    let else_ops =
                        self.entries_ops_with_axes(builder, else_body, position_of, next_axis)?;
                    ops.push(CpuOp::new(
                        CpuOpKind::Branch {
                            condition: condition_position,
                            then_ops,
                            else_ops,
                        },
                        Vec::new(),
                        Vec::new(),
                    ));
                }
            }
        }
        Ok(ops)
    }

    /// The opcode of one formed primitive, re-anchored onto the fused layout.
    fn primitive_op(
        &self,
        entry: &ComponentEntry,
        position_of: &impl Fn(GraphValueId) -> Option<u16>,
    ) -> Result<CpuOp, BuilderError> {
        let formed = entry.formed.as_ref().ok_or_else(|| {
            format!(
                "CPU node#{} reached primitive mapping without formation",
                entry.node.node.0
            )
        })?;
        let physical = formed.physical_primitive();
        let profile = mapping::target_profile(&Limits {
            workers: 1,
            max_scratch_bytes: mapping::SCRATCH_BYTES,
        });
        let op = CpuDialect::legalize(&physical, &profile)
            .ops()
            .and_then(|ops| ops.iter().next().cloned())
            .ok_or_else(|| {
                format!(
                    "a universal CPU primitive mapping is inapplicable (compiler bug): {:?}",
                    formed.op
                )
            })?;
        let operands = entry
            .logical
            .inputs
            .iter()
            .map(|value| {
                position_of(*value)
                    .ok_or_else(|| format!("CPU operand value#{} has no binding position", value.0))
            })
            .collect::<Result<Vec<_>, BuilderError>>()?;
        let results = entry
            .logical
            .outputs
            .iter()
            .map(|output| {
                position_of(output.id).ok_or_else(|| {
                    format!("CPU result value#{} has no binding position", output.id.0)
                })
            })
            .collect::<Result<Vec<_>, BuilderError>>()?;
        let CpuOp {
            kind,
            operands: _,
            results: _,
        } = op;
        let kind = match (self.atomics, kind) {
            // The device-atomic mapping keeps the worker domain, so every
            // update is a compare/exchange on the element's word.
            (AtomicMapping::Device, CpuOpKind::SerializedAtomic { op, dtype, shape }) => {
                CpuOpKind::AtomicDevice { op, dtype, shape }
            }
            (_, kind) => kind,
        };
        Ok(CpuOp::new(kind, Vec::new(), Vec::new()).with_positions(operands, results))
    }

    /// One planned runtime-check predicate with positional operands.
    fn check_kind(
        &self,
        predicate: &CheckPredicate,
        position_of: &impl Fn(GraphValueId) -> Option<u16>,
    ) -> Result<CheckKind, BuilderError> {
        let position = |value: GraphValueId| -> Result<u16, BuilderError> {
            position_of(value)
                .ok_or_else(|| format!("check operand value#{} has no binding position", value.0))
        };
        Ok(match predicate {
            CheckPredicate::IndexInBounds { index, extent } => CheckKind::IndexInBounds {
                index: position(*index)?,
                extent: extent.clone(),
            },
            CheckPredicate::RangeInBounds { start, end, extent } => CheckKind::RangeInBounds {
                start: position(*start)?,
                end: position(*end)?,
                extent: extent.clone(),
            },
            CheckPredicate::DivisorNonZero { value } => CheckKind::DivisorNonZero {
                value: position(*value)?,
            },
            CheckPredicate::DivisionSafe { lhs, rhs } => CheckKind::DivisionSafe {
                lhs: position(*lhs)?,
                rhs: position(*rhs)?,
            },
            CheckPredicate::ShiftInRange { value } => CheckKind::ShiftInRange {
                value: position(*value)?,
            },
            CheckPredicate::ProductFits { factors, bits } => CheckKind::ProductFits {
                factors: factors.clone(),
                bits: *bits,
            },
            CheckPredicate::ExtentPositive { extent } => CheckKind::ExtentPositive {
                extent: extent.clone(),
            },
        })
    }

    // -- structured nodes ---------------------------------------------------

    fn reduction(
        &self,
        builder: &mut AlternativeBuilder<CpuDialect>,
        path: &RegionPath,
        node_id: NodeId,
        node: &LogicalNode,
        reduction: &ReductionNode,
    ) -> Result<(), BuilderError> {
        let node_ref = NodeRef {
            region: path.clone(),
            node: node_id,
        };
        let formed = universal_node(node_id, node, &self.facts)
            .map_err(|error| format!("CPU reduction formation: {error}"))?;
        let UniversalNode::Reduction(universal) = &formed else {
            return Err("CPU reduction formation returned a non-reduction".into());
        };
        let strategy: ReductionStrategy = universal.strategy.clone();
        // The reduction preconditions (`argmax` nonempty) are retained by
        // the fold opcode itself: a zero length reports the first error.
        let _ = &universal.preconditions;
        let operand_type = self
            .facts
            .types
            .get(&reduction.operand)
            .cloned()
            .ok_or("the reduction operand has no type")?;
        let tensor = operand_type
            .shaped()
            .cloned()
            .ok_or("the reduction operand is not a tensor")?;
        let input_dtype = match &tensor.elem {
            Elem::Dtype(d) => *d,
            _ => DType::F32,
        };
        let mut outer = tensor.axes.clone();
        let length = outer.remove(reduction.axis);
        let result_dtype = match &reduction.result {
            ValueType::Scalar(d) => *d,
            _ => input_dtype,
        };
        let nonempty_precondition =
            reduction_identity(reduction.op) == ReductionIdentity::FirstElementNonEmpty;
        let iteration = LinearIterationMap::linear(&outer, &self.runtime_extents)
            .map_err(|error| format!("CPU reduction geometry: {error}"))?
            .with_participants(self.participants.clone());
        let result_value = node
            .outputs
            .first()
            .map(|output| output.id)
            .ok_or("a reduction has no result value")?;
        let result_transport = builder.transport_of(result_value)?;
        // Predicted layout: [operand, result].
        let op = CpuOp::new(
            CpuOpKind::ReduceFold {
                op: reduction.op,
                input: input_dtype,
                accumulator: strategy.accumulator,
                result_dtype,
                axis: reduction.axis,
                outer,
                length,
                nonempty_precondition,
            },
            vec![0],
            vec![1],
        );
        builder.map_reduction(
            node_ref,
            ReductionStrategyTemplate {
                topology: strategy.topology.clone(),
                iteration,
                ops: Legalized::Ops(NonEmpty::new(vec![op]).expect("one opcode")),
                result: result_transport,
            },
        )
    }

    fn loop_node(
        &self,
        builder: &mut AlternativeBuilder<CpuDialect>,
        graph: &TaskGraph,
        path: &RegionPath,
        node_id: NodeId,
        node: &LogicalNode,
    ) -> Result<(), BuilderError> {
        let node_ref = NodeRef {
            region: path.clone(),
            node: node_id,
        };
        let formed = universal_node(node_id, node, &self.facts)
            .map_err(|error| format!("CPU loop formation: {error}"))?;
        let UniversalNode::Loop(universal) = &formed else {
            return Err("CPU loop formation returned a non-loop".into());
        };
        // The loop node's own safety obligations (range bounds) are
        // classified and discharged; the runtime evaluates the retained
        // range predicate when it interprets the `Repeat` step.
        for (index, obligation) in node.safety.iter().enumerate() {
            let classified = classify_obligation(obligation, &self.facts, node.span);
            terminal::discharge_with_builder(
                builder,
                ObligationRef {
                    node: node_ref.clone(),
                    index,
                },
                &classified,
                |_| Legalized::Inapplicable {
                    reason: "the range predicate is evaluated by the executor".into(),
                },
            )?;
        }
        let carried = universal
            .carries
            .iter()
            .map(|slot| match slot.initial {
                RegionInput::Value(value) => Ok(PhysicalCarryTemplate {
                    transport: builder.transport_of(value)?,
                }),
                RegionInput::State(token) => {
                    let storage = builder
                        .logical_storage_of_token(token)
                        .ok_or("a carried state token has no storage")?;
                    Ok(PhysicalCarryTemplate {
                        transport: match builder.storage_of(storage) {
                            Some(template) => TransportTemplate::Storage(
                                NonEmpty::new(vec![StorageViewTemplate {
                                    storage: template,
                                    access: Access::Exclusive,
                                    transform: seismic_lang::logical::ViewTransform::Identity,
                                }])
                                .expect("one plane"),
                            ),
                            None => TransportTemplate::Boundary(boundary_leaf(graph, storage)),
                        },
                    })
                }
            })
            .collect::<Result<Vec<_>, BuilderError>>()?;
        let range = ExecutorRangeTemplate {
            start: builder.transport_of(universal.range.start)?,
            end: builder.transport_of(universal.range.end)?,
            bound: universal.range.bound.clone(),
        };
        let body_path = {
            let mut p = path.clone();
            p.push(RegionStep::LoopBody(node_id));
            p
        };
        builder.schedule_loop(node_ref, range, carried, |builder| {
            self.region(builder, graph, &body_path)
        })
    }

    fn if_node(
        &self,
        builder: &mut AlternativeBuilder<CpuDialect>,
        graph: &TaskGraph,
        path: &RegionPath,
        node_id: NodeId,
        node: &LogicalNode,
    ) -> Result<(), BuilderError> {
        let _ = graph;
        let node_ref = NodeRef {
            region: path.clone(),
            node: node_id,
        };
        let formed = universal_node(node_id, node, &self.facts)
            .map_err(|error| format!("CPU conditional formation: {error}"))?;
        let UniversalNode::If(universal) = &formed else {
            return Err("CPU conditional formation returned a non-conditional".into());
        };
        let predicate = ExecutorPredicateTemplate {
            value: builder.transport_of(universal.condition)?,
        };
        // Aggregate joins are leaf-lowered: each value join copies the taken
        // branch's transport into the joined transport (leaf slots/storage
        // views already agree); state joins name the joined storage.
        let joins = universal
            .joins
            .iter()
            .map(|slot| match slot {
                JoinSlot::Value { joined, .. } => {
                    let transport = builder.transport_of(*joined)?;
                    Ok(PhysicalJoinTemplate::Value {
                        then: transport.clone(),
                        else_branch: transport.clone(),
                        joined: transport,
                    })
                }
                JoinSlot::State { storage, .. } => {
                    let template = builder
                        .storage_of(*storage)
                        .ok_or("a joined state storage has no template")?;
                    Ok(PhysicalJoinTemplate::State { storage: template })
                }
            })
            .collect::<Result<Vec<_>, BuilderError>>()?;
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
            predicate,
            |builder| self.region(builder, graph, &then_path),
            |builder| self.region(builder, graph, &else_path),
            joins,
        )
    }

    fn call(
        &self,
        builder: &mut AlternativeBuilder<CpuDialect>,
        _graph: &TaskGraph,
        path: &RegionPath,
        node_id: NodeId,
        node: &LogicalNode,
    ) -> Result<(), BuilderError> {
        let node_ref = NodeRef {
            region: path.clone(),
            node: node_id,
        };
        let formed = universal_node(node_id, node, &self.facts)
            .map_err(|error| format!("CPU call formation: {error}"))?;
        let UniversalNode::Call(universal) = &formed else {
            return Err("CPU call formation returned a non-call".into());
        };
        let _ = universal;
        builder.invoke_canonical(node_ref)
    }
}

// ---------------------------------------------------------------------------
// The pipeline backend
// ---------------------------------------------------------------------------

/// Mechanically retained resolved launch (encoding decisions happen at
/// assembly, where the whole resolved plan is available).
pub struct EncodedLaunch {
    pub launch: ResolvedLaunch<CpuDialect>,
}

/// The CPU native artifact: executable memory plus the runtime glue's
/// launch table.
pub struct NativeArtifact {
    pub kernel: crate::native::Kernel,
}

impl Backend for mapping::Cpu {
    type Dialect = CpuDialect;
    type EncodedLaunch = EncodedLaunch;
    type NativeArtifact = NativeArtifact;

    fn target(&self) -> &'static str {
        mapping::TARGET
    }

    fn capability_fingerprint(&self) -> String {
        mapping::capability_fingerprint(self.limits())
    }

    fn supports_intrinsic(
        &self,
        intrinsic: &seismic_lang::sir::IntrinsicUse,
    ) -> Result<(), String> {
        Err(format!(
            "the CPU backend implements no backend intrinsic `{}`",
            intrinsic.id.path()
        ))
    }

    fn target_profile(&self) -> &EffectiveTargetProfile {
        self.profile()
    }

    fn elaborate(&self, logical: &LogicalProgram) -> Result<PlanFamily<CpuDialect>, String> {
        elaborate(logical, self.limits())
    }

    fn encode_launch(
        &self,
        launch: &ResolvedLaunch<CpuDialect>,
    ) -> Result<Self::EncodedLaunch, String> {
        Ok(EncodedLaunch {
            launch: launch.clone(),
        })
    }

    fn assemble(
        &self,
        encoded: EncodedPlan<CpuDialect, EncodedLaunch>,
    ) -> Result<Self::NativeArtifact, String> {
        let kernel = crate::native::assemble(encoded)?;
        Ok(NativeArtifact { kernel })
    }
}
