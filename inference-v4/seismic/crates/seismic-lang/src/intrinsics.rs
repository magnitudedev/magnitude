//! The intrinsic registry.
//!
//! Typing, reference execution, logical construction, backend legalization,
//! and numerical analysis all consume this single closed table. Low-level
//! emission fragments (native matrix load/store blocks, participant-width
//! declarations) belong to backend dialects and are deliberately absent.

use crate::sym::Sym;
use crate::syntax::ast::{BinaryOp, UnaryOp};
use crate::types::{DType, Elem, ExtentExpr, TensorType, ValueType};

/// Revision of the registry contract. Capability fingerprints and every
/// downstream identity retain this value so cached decisions cannot survive a
/// semantic change.
pub const REGISTRY_REVISION: &str = "seismic-registry-v4";

// ---------------------------------------------------------------------------
// Identities
// ---------------------------------------------------------------------------

/// Stable source-level identity of one backend capability namespace.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CapabilityId {
    pub backend: String,
    pub name: String,
}

impl CapabilityId {
    pub fn new(backend: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            backend: backend.into(),
            name: name.into(),
        }
    }

    pub fn path(&self) -> String {
        format!("{}.{}", self.backend, self.name)
    }
}

/// Stable source-level identity of an intrinsic within a capability namespace:
/// `backend.namespace.member`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IntrinsicId {
    pub capability: CapabilityId,
    pub name: String,
}

impl IntrinsicId {
    pub fn path(&self) -> String {
        format!("{}.{}", self.capability.path(), self.name)
    }
}

impl std::fmt::Display for IntrinsicId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.path())
    }
}

/// A concrete (parameter-free) type as used by capability signatures.
pub type ConcreteType = ValueType;

// ---------------------------------------------------------------------------
// Operation vocabularies
// ---------------------------------------------------------------------------

/// Mathematical operations admitted by the source `max`, `min`, `fma`, … calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MathOp {
    Fma,
    Exp,
    ExpFast,
    Rsqrt,
    Sqrt,
    Log,
    Sin,
    Cos,
    Abs,
    Max,
    Min,
}

impl MathOp {
    pub fn name(self) -> &'static str {
        match self {
            MathOp::Fma => "fma",
            MathOp::Exp => "exp",
            MathOp::ExpFast => "exp_fast",
            MathOp::Rsqrt => "rsqrt",
            MathOp::Sqrt => "sqrt",
            MathOp::Log => "log",
            MathOp::Sin => "sin",
            MathOp::Cos => "cos",
            MathOp::Abs => "abs",
            MathOp::Max => "max",
            MathOp::Min => "min",
        }
    }

    pub fn arity(self) -> usize {
        match self {
            MathOp::Fma => 3,
            MathOp::Max | MathOp::Min => 2,
            _ => 1,
        }
    }

    /// `true` when the operation is defined on numeric (not only float) operands.
    pub fn numeric_operands(self) -> bool {
        matches!(self, MathOp::Max | MathOp::Min | MathOp::Abs)
    }
}

/// The four reductions. `argmax` returns `i32` and chooses the smaller
/// coordinate on ties; it never accepts reassociation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
    Argmax,
}

impl ReduceOp {
    pub fn name(self) -> &'static str {
        match self {
            ReduceOp::Sum => "sum",
            ReduceOp::Max => "max",
            ReduceOp::Min => "min",
            ReduceOp::Argmax => "argmax",
        }
    }
}

/// Structure of one index slot of an indexing primitive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IndexSlot {
    Point,
    /// `lo:hi` with present bounds; omitted bounds are the axis ends.
    Range {
        start: bool,
        end: bool,
    },
}

/// One readable physical plane of a packed representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlaneField {
    Words,
    Scale,
    Bias,
    Coefficients,
    ScaleFactor,
    BiasFactor,
}

impl PlaneField {
    pub fn from_name(name: &str) -> Option<PlaneField> {
        Some(match name {
            "words" => PlaneField::Words,
            "scale" => PlaneField::Scale,
            "bias" => PlaneField::Bias,
            "coefficients" => PlaneField::Coefficients,
            "scale_factor" => PlaneField::ScaleFactor,
            "bias_factor" => PlaneField::BiasFactor,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            PlaneField::Words => "words",
            PlaneField::Scale => "scale",
            PlaneField::Bias => "bias",
            PlaneField::Coefficients => "coefficients",
            PlaneField::ScaleFactor => "scale_factor",
            PlaneField::BiasFactor => "bias_factor",
        }
    }
}

/// The combining operation of an `atomic` update. `add` is the registry
/// load/add/round/store; `max` and `min` are exact and order-independent
/// (NaN operands are ignored, as in the reference reductions).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AtomicOp {
    Add,
    Max,
    Min,
}

impl AtomicOp {
    pub fn name(self) -> &'static str {
        match self {
            AtomicOp::Add => "add",
            AtomicOp::Max => "max",
            AtomicOp::Min => "min",
        }
    }

    pub fn parse(name: &str) -> Option<AtomicOp> {
        match name {
            "add" => Some(AtomicOp::Add),
            "max" => Some(AtomicOp::Max),
            "min" => Some(AtomicOp::Min),
            _ => None,
        }
    }
}

/// The closed portable primitive vocabulary. Every checked expression is a
/// registry primitive (or a static function-family call); payloads carry only
/// structure the operands cannot express (constant axes, index slot shapes,
/// allocation elements, reduction axes).
#[derive(Clone, Debug, PartialEq)]
pub enum PrimitiveId {
    /// Tuple construction from two or more components.
    TuplePack,
    /// Ordinal projection of a tuple.
    TupleGet(usize),
    /// `lo..hi` range construction.
    RangeMake,
    /// Start endpoint of a range value.
    RangeStart,
    /// End endpoint of a range value.
    RangeEnd,
    Unary(UnaryOp),
    Binary(BinaryOp),
    /// Scalar or elementwise cast; also the dense decode of a packed value.
    Cast(DType),
    Math(MathOp),
    Select,
    /// `tensor[shape] elem`: uninitialized owned storage.
    TensorAlloc {
        elem: Elem,
    },
    /// `zeros_like` / `ones_like`: shape of the operand, constant fill.
    Fill {
        value: f64,
        dtype: DType,
    },
    /// `to_owned`: new owned storage from a borrowed or computed value.
    Materialize,
    /// `clone`: duplicate an owned tensor.
    Clone,
    /// `load`: snapshot in the operand's own representation.
    Load,
    /// `decode`: dense `f32` value of a packed view.
    Decode,
    /// Packed plane read (`.words`, `.scale`, …): readable, never writable.
    PackedRead(PlaneField),
    Transpose,
    Reshape,
    /// View selection `t[i, j:k, …]`.
    SliceView {
        indices: Vec<IndexSlot>,
    },
    /// Point read `t[i, j]`.
    ElementRead {
        arity: usize,
    },
    /// Point write (carried by checked assignments; consumed by logical
    /// construction and backend legalization).
    ElementWrite {
        arity: usize,
    },
    /// Slice write of a shaped value into a place.
    CopyInto,
    /// `extent(v, axis)`.
    Extent {
        axis: usize,
    },
    /// `valid(v, axis)`: the extent in effect at runtime.
    ValidExtent {
        axis: usize,
    },
    /// `atomic(add|max|min, place, value)`. Defined for f32, f16, bf16, i32,
    /// u32; bool is rejected because bool arithmetic is undefined.
    Atomic {
        op: AtomicOp,
        arity: usize,
    },
    Reduce {
        op: ReduceOp,
        axis: usize,
        unordered: bool,
    },
}

impl PrimitiveId {
    pub fn name(&self) -> String {
        match self {
            PrimitiveId::TuplePack => "tuple.pack".into(),
            PrimitiveId::TupleGet(i) => format!("tuple.get.{i}"),
            PrimitiveId::RangeMake => "range.make".into(),
            PrimitiveId::RangeStart => "range.start".into(),
            PrimitiveId::RangeEnd => "range.end".into(),
            PrimitiveId::Unary(op) => format!("unary.{}", op.text().trim()),
            PrimitiveId::Binary(op) => format!("binary.{}", op.text()),
            PrimitiveId::Cast(d) => format!("cast.{}", d.name()),
            PrimitiveId::Math(op) => format!("math.{}", op.name()),
            PrimitiveId::Select => "select".into(),
            PrimitiveId::TensorAlloc { .. } => "tensor.alloc".into(),
            PrimitiveId::Fill { .. } => "tensor.fill".into(),
            PrimitiveId::Materialize => "tensor.materialize".into(),
            PrimitiveId::Clone => "tensor.clone".into(),
            PrimitiveId::Load => "tensor.load".into(),
            PrimitiveId::Decode => "tensor.decode".into(),
            PrimitiveId::PackedRead(f) => format!("packed.read.{}", f.name()),
            PrimitiveId::Transpose => "tensor.transpose".into(),
            PrimitiveId::Reshape => "tensor.reshape".into(),
            PrimitiveId::SliceView { .. } => "tensor.slice".into(),
            PrimitiveId::ElementRead { .. } => "tensor.read".into(),
            PrimitiveId::ElementWrite { .. } => "tensor.write".into(),
            PrimitiveId::CopyInto => "tensor.copy".into(),
            PrimitiveId::Extent { .. } => "tensor.extent".into(),
            PrimitiveId::ValidExtent { .. } => "tensor.valid_extent".into(),
            PrimitiveId::Atomic { op, .. } => format!("atomic.{}", op.name()),
            PrimitiveId::Reduce { op, .. } => format!("reduce.{}", op.name()),
        }
    }
}

impl std::fmt::Display for PrimitiveId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name())
    }
}

// ---------------------------------------------------------------------------
// Parameter and result type functions
// ---------------------------------------------------------------------------

/// Classes of scalar dtypes admitted by a parameter pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DTypeClass {
    Any,
    Float,
    Int,
    Numeric,
    Bool,
}

impl DTypeClass {
    pub fn matches(self, d: DType) -> bool {
        match self {
            DTypeClass::Any => true,
            DTypeClass::Float => d.is_float(),
            DTypeClass::Int => d.is_int(),
            DTypeClass::Numeric => d.is_numeric(),
            DTypeClass::Bool => d == DType::Bool,
        }
    }
}

/// Element classes admitted by a tensor parameter pattern. An element
/// parameter of the enclosing declaration is admitted as a float wherever a
/// dense float class is required (it is bound to a concrete element at
/// specialization).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ElemClass {
    Any,
    Dense(DTypeClass),
    Packed,
}

impl ElemClass {
    /// Whether a scalar operand satisfies this element class.
    pub fn matches_dtype(self, d: DType) -> bool {
        match self {
            ElemClass::Any => true,
            ElemClass::Packed => false,
            ElemClass::Dense(class) => class.matches(d),
        }
    }

    pub fn matches(self, elem: &Elem) -> bool {
        match (self, elem) {
            (ElemClass::Any, _) => true,
            (ElemClass::Packed, Elem::Repr(_)) => true,
            (ElemClass::Packed, _) => false,
            (ElemClass::Dense(_), Elem::Repr(_)) => false,
            (ElemClass::Dense(class), Elem::Dtype(d)) => class.matches(*d),
            // An element parameter is admitted where a dense float is; it is
            // resolved to a concrete element at specialization.
            (ElemClass::Dense(DTypeClass::Float), Elem::Param(_)) => true,
            (ElemClass::Dense(DTypeClass::Any), Elem::Param(_)) => true,
            (ElemClass::Dense(..), Elem::Param(_)) => false,
        }
    }
}

/// Pattern for one parameter of a primitive.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TypePattern {
    Exact(ValueType),
    /// A scalar (or index) value of a dtype class.
    ScalarOf(DTypeClass),
    /// A shaped value whose element lies in a class.
    TensorOf(ElemClass),
    /// A scalar or dense elementwise operand; tiles broadcast over scalars.
    Elementwise(ElemClass),
    /// The same type as an earlier parameter.
    SameAs(usize),
    /// A tuple with the given component patterns.
    TupleOf(Vec<TypePattern>),
    /// A range value with any bound.
    Range,
    Any,
}

impl TypePattern {
    pub fn matches(&self, ty: &ValueType) -> bool {
        match self {
            TypePattern::Exact(expected) => expected == ty,
            TypePattern::ScalarOf(class) => ty.scalar_dtype().is_some_and(|d| class.matches(d)),
            TypePattern::TensorOf(class) => ty.shaped().is_some_and(|s| class.matches(&s.elem)),
            TypePattern::Elementwise(class) => match ty {
                ValueType::Scalar(d) => class.matches_dtype(*d),
                ValueType::Index { .. } => true,
                ValueType::Tensor(s) => class.matches(&s.elem),
                _ => false,
            },
            // `SameAs` is a documentation pattern; relation checking happens in
            // the result function, so the pattern itself admits any operand.
            TypePattern::SameAs(_) => true,
            TypePattern::TupleOf(items) => match ty {
                ValueType::Tuple(parts) => {
                    parts.len() == items.len() && parts.iter().zip(items).all(|(t, p)| p.matches(t))
                }
                _ => false,
            },
            TypePattern::Range => matches!(ty, ValueType::Range { .. }),
            TypePattern::Any => true,
        }
    }
}

/// How the dtype of an elementwise result is chosen.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ResultDType {
    SameAs(usize),
    Promoted(Vec<usize>),
    Dtype(DType),
}

/// The result type of a primitive given its operand types. `None` marks
/// primitives whose result the checker derives from source structure the
/// operands alone do not carry (allocation shapes, packed-plane geometry,
/// range bounds); the checker supplies and pattern-validates that type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TypeFunction {
    Fixed(ValueType),
    /// Same type as operand `i`.
    SameAs(usize),
    /// Elementwise over the shape of operand `shape_of` with a derived dtype.
    Elementwise {
        shape_of: usize,
        dtype: ResultDType,
    },
    TupleOfOperands,
    TupleComponent {
        tuple: usize,
        index: usize,
    },
    /// Ordered reduction of operand 0 along the primitive's axis.
    Reduction {
        operand: usize,
        op: ReduceOp,
    },
    /// Checker-supplied (allocation, packed plane, range construction).
    Structural,
}

fn elementwise(source: &ValueType, dtype: DType) -> Option<ValueType> {
    match source {
        ValueType::Tensor(s) => Some(ValueType::Tensor(TensorType::new(
            s.axes.clone(),
            Elem::Dtype(dtype),
        ))),
        ValueType::Scalar(_) | ValueType::Index { .. } => Some(ValueType::Scalar(dtype)),
        _ => None,
    }
}

fn scalar_of(operands: &[ValueType], i: usize) -> Option<DType> {
    match operands.get(i)? {
        ValueType::Scalar(d) => Some(*d),
        ValueType::Index { .. } => Some(DType::I32),
        ValueType::Tensor(s) => match &s.elem {
            Elem::Dtype(d) => Some(*d),
            // An unresolved element parameter reads as f32 at portable scope.
            Elem::Param(_) => Some(DType::F32),
            Elem::Repr(_) => None,
        },
        _ => None,
    }
}

fn promoted(operands: &[ValueType], at: &[usize]) -> Option<DType> {
    let mut dtype = None;
    for i in at {
        let d = scalar_of(operands, *i)?;
        dtype = Some(match dtype {
            None => d,
            Some(p) => DType::promote(p, d)?,
        });
    }
    dtype
}

/// The accumulator and result dtype of a reduction (registry decision):
/// floating ordered/unordered `sum` of f16/bf16/f32 accumulates and results in
/// f32; integer `sum` retains the input dtype and wraps; `max`/`min` retain the
/// input dtype; `argmax` results in i32.
pub fn accumulator_dtype(op: ReduceOp, input: DType) -> DType {
    match op {
        ReduceOp::Sum if input.is_float() => DType::F32,
        ReduceOp::Argmax => DType::I32,
        _ => input,
    }
}

/// The element the fold starts from (registry semantics).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReduceIdentity {
    /// `sum`: the additive identity of the accumulator dtype.
    Zero,
    /// `max`/`min`: no identity; the fold starts from the first (ascending)
    /// element, so the reduced axis must be nonempty.
    FirstElement,
    /// `argmax`: no identity, smaller-index ties, nonempty input required.
    FirstElementNonEmpty,
}

/// How ties are resolved. The registry admits exactly one rule; strategies
/// may not weaken it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TieRule {
    /// The smaller coordinate index wins.
    SmallerCoordinateIndex,
}

/// The algebraic combination law of a reduction operator over its
/// accumulator: what reorderings of the fold are meaning-preserving in
/// exact arithmetic. The numerical transfer of any reassociation under
/// finite precision is realization numerics' authority, not the registry's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CombineLaw {
    /// Regrouping preserves meaning; operand order does not.
    Associative,
    /// Regrouping and reordering both preserve meaning.
    AssociativeCommutative,
    /// Only the ascending reference fold defines the result.
    OrderedOnly,
}

/// The complete reduction schema of one operator over one input dtype: the
/// single owner of accumulator/result dtypes, identity, tie rule, and
/// combination law consumed by kernel formation, strategy formation, and
/// backends.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReduceSchema {
    /// Dtype of the running combined value: the `accumulator_dtype` for
    /// `sum`/`max`/`min`; the input dtype (the running extremum) for
    /// `argmax`, whose running coordinate is an implicit `i32`.
    pub accumulator: DType,
    /// Dtype of the published result (`accumulator_dtype`).
    pub result: DType,
    pub identity: ReduceIdentity,
    pub ties: TieRule,
    pub combine: CombineLaw,
}

/// The reduction schema of `op` over inputs of dtype `input`. Exhaustive
/// over `ReduceOp`.
pub fn reduce_schema(op: ReduceOp, input: DType) -> ReduceSchema {
    let result = accumulator_dtype(op, input);
    match op {
        ReduceOp::Sum => ReduceSchema {
            accumulator: result,
            result,
            identity: ReduceIdentity::Zero,
            ties: TieRule::SmallerCoordinateIndex,
            combine: CombineLaw::AssociativeCommutative,
        },
        ReduceOp::Max | ReduceOp::Min => ReduceSchema {
            accumulator: result,
            result,
            identity: ReduceIdentity::FirstElement,
            ties: TieRule::SmallerCoordinateIndex,
            combine: CombineLaw::AssociativeCommutative,
        },
        ReduceOp::Argmax => ReduceSchema {
            accumulator: input,
            result,
            identity: ReduceIdentity::FirstElementNonEmpty,
            ties: TieRule::SmallerCoordinateIndex,
            combine: CombineLaw::OrderedOnly,
        },
    }
}

/// The result type of reducing `operand` along `axis` with `op`.
pub fn reduction_result(operand: &ValueType, op: ReduceOp, axis: usize) -> Option<ValueType> {
    let shaped = operand.shaped()?;
    let input = match &shaped.elem {
        Elem::Dtype(d) => *d,
        // An unresolved element parameter reads as f32 at portable scope.
        Elem::Param(_) => DType::F32,
        Elem::Repr(_) => return None,
    };
    let dtype = accumulator_dtype(op, input);
    let mut axes = shaped.axes.clone();
    if axis >= axes.len() {
        return None;
    }
    axes.remove(axis);
    Some(if axes.is_empty() {
        ValueType::Scalar(dtype)
    } else {
        ValueType::Tensor(TensorType::new(axes, Elem::Dtype(dtype)))
    })
}

impl TypeFunction {
    pub fn apply(&self, operands: &[ValueType]) -> Option<ValueType> {
        match self {
            TypeFunction::Fixed(t) => Some(t.clone()),
            TypeFunction::SameAs(i) => operands.get(*i).cloned(),
            TypeFunction::Elementwise { shape_of, dtype } => {
                let d = match dtype {
                    ResultDType::SameAs(i) => scalar_of(operands, *i)?,
                    ResultDType::Promoted(at) => promoted(operands, at)?,
                    ResultDType::Dtype(d) => *d,
                };
                // Broadcast: a scalar shape source takes the axes of the first
                // tensor operand, if any.
                let source = match operands.get(*shape_of) {
                    Some(ValueType::Scalar(_) | ValueType::Index { .. }) => operands
                        .iter()
                        .find(|t| matches!(t, ValueType::Tensor(_)))
                        .unwrap_or(operands.get(*shape_of)?),
                    other => other?,
                };
                elementwise(source, d)
            }
            TypeFunction::TupleOfOperands => {
                let items = operands.to_vec();
                Some(ValueType::Tuple(crate::types::NonEmpty::new(items)?))
            }
            TypeFunction::TupleComponent { tuple, index } => match operands.get(*tuple)? {
                ValueType::Tuple(parts) => parts.as_slice().get(*index).cloned(),
                _ => None,
            },
            TypeFunction::Reduction { operand, op } => {
                let shaped = operands.get(*operand)?.shaped()?;
                let input = match &shaped.elem {
                    Elem::Dtype(d) => *d,
                    Elem::Param(_) => DType::F32,
                    Elem::Repr(_) => return None,
                };
                let _ = accumulator_dtype(*op, input);
                None // axis is a payload of the primitive id; see PrimitiveSignature::result_type
            }
            TypeFunction::Structural => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Effects, safety, reference semantics, numerics
// ---------------------------------------------------------------------------

/// What a primitive does to storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EffectFunction {
    Pure,
    /// Reads existing storage.
    Reads,
    /// Reads operand storage and allocates fresh owned storage.
    Allocates,
    /// Writes storage; `whole` marks a whole-object write.
    Writes {
        whole: bool,
    },
    /// Atomic read-modify-write of one element.
    Atomic,
}

/// Runtime obligations a primitive carries (the checked-level form of the
/// logical `SafetyObligation` kinds; static proofs discharge them).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SafetyFunction {
    None,
    IndexInBounds,
    RangeInBounds,
    DivisorNonZero,
    SignedDivisionNoOverflow,
    ShiftInRange,
}

/// The reference execution meaning of a primitive.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReferenceSemantics {
    ScalarOp,
    ElementwiseOp,
    TupleOp,
    RangeOp,
    AllocateUninitialized,
    FillConstant,
    MaterializeSnapshot,
    CloneOwned,
    LoadSnapshot,
    DecodePacked,
    ReadPackedPlane,
    ViewTransform,
    ReadElement,
    ReadExtent,
    /// Read the place, combine with the value by the primitive's `AtomicOp`,
    /// write back; the reference applies visits in loop order.
    Atomic,
    Reduce {
        /// Accumulator dtype per `accumulator_dtype`.
        accumulator: DType,
        /// Visits ascending coordinates.
        ascending: bool,
        /// Ties choose the smaller coordinate.
        smaller_coordinate_ties: bool,
    },
}

/// The reference numerical contract of a primitive.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReferenceNumerics {
    Exact,
    /// Arithmetic rounds once at the result dtype.
    RoundsOnce {
        dtype: DType,
    },
    /// Integer add/subtract/multiply wrap at 32 bits.
    Wraps {
        dtype: DType,
    },
    /// Division/remainder are Euclidean with nonzero divisor and no signed overflow.
    EuclideanDivision,
    /// Shift counts must lie in `0..32`.
    Shifts,
    /// Integer-to-integer casts preserve the low 32 bits.
    LowBitsCast,
    /// Other casts convert by value with defined rounding/saturation.
    ConvertingCast {
        dtype: DType,
    },
    /// One versioned portable software sequence defines the reference bits.
    SoftwareMath {
        algorithm: &'static str,
    },
    /// Ordered sum visits ascending coordinates and rounds each step.
    Accumulates {
        accumulator: DType,
    },
}

/// The core kernel-operation family that lowers a primitive category. This
/// is the registry's lowering schema: kernel formation matches it
/// exhaustively and maps each family onto the closed core kernel algebra
/// (`seismic-realization::kernel::CoreKernelOp`) or onto a structural
/// consequence with no operation. Backends never see a primitive id; they
/// see the core operation the family names.
///
/// The families cover the complete primitive category of the semantic
/// coverage basis: typed constants and runtime extents (which are logical
/// primitive operations without a `PrimitiveId`) and every registry
/// primitive. `lowering(id)` maps every `PrimitiveId` onto exactly one
/// family; `Constant` is the family of the typed-literal category only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CoreLoweringFamily {
    /// Scalar operator arithmetic: `Unary`, `Binary`, and `Compare` core ops
    /// applied per element under the participant map.
    ScalarArithmetic,
    /// Registry math sequences: the `Math` core op applied per element.
    ElementwiseMap,
    /// Dtype conversion: the `Cast` core op applied per element.
    Cast,
    /// Conditional selection: the `Select` core op applied per element.
    Select,
    /// Tuple and range construction/projection: pure value structure, no
    /// kernel operation.
    Structural,
    /// Owned uninitialized storage: a residence only, no kernel operation.
    TensorAlloc,
    /// Constant fill of fresh storage: a `Const` and a `Store` per element.
    Fill,
    /// Slice write of a shaped value into a place: a `Load` and a `Store`
    /// per element of the written range.
    CopyInto,
    /// Whole-object copy into fresh owned storage (`to_owned`, `clone`,
    /// `load`): a `Load` and a `Store` per storage element of every plane.
    BulkCopy,
    /// Dense decode of a packed element: the `PackedDecode` core op.
    PackedDecode,
    /// Raw plane field read of a packed element: the `PackedPlaneRead` core
    /// op.
    PackedPlaneRead,
    /// View selection (`transpose`, `reshape`, slicing): a route only, no
    /// kernel operation.
    ViewTransform,
    /// Point read of one element: the `Load` core op.
    ElementRead,
    /// Point write of one element: the `Store` core op.
    ElementWrite,
    /// The checked (capacity) extent of an axis: a `Const` for a static
    /// extent, the retained `RuntimeExtent` for a runtime extent.
    ExtentRead,
    /// Atomic read-modify-write of one element: the `Atomic` core op.
    Atomic,
    /// Reduction along one axis: the `Fold` core op (universal form) or an
    /// optimized strategy alternative preserving the `ReduceSchema`.
    Reduce,
    /// A typed literal constant: the `Const` core op. Not a registry
    /// primitive; the family of `logical::PrimitiveOp::Constant`.
    Constant,
    /// The extent in effect at runtime (`valid`): the `RuntimeExtent` core
    /// op. Also the family of `logical::PrimitiveOp::RuntimeExtent`.
    RuntimeExtent,
}

/// The core lowering family of one primitive identity. Exhaustive: adding a
/// `PrimitiveId` variant fails compilation here.
pub fn lowering(id: &PrimitiveId) -> CoreLoweringFamily {
    match id {
        PrimitiveId::TuplePack
        | PrimitiveId::TupleGet(_)
        | PrimitiveId::RangeMake
        | PrimitiveId::RangeStart
        | PrimitiveId::RangeEnd => CoreLoweringFamily::Structural,
        PrimitiveId::Unary(_) | PrimitiveId::Binary(_) => CoreLoweringFamily::ScalarArithmetic,
        PrimitiveId::Cast(_) => CoreLoweringFamily::Cast,
        PrimitiveId::Math(_) => CoreLoweringFamily::ElementwiseMap,
        PrimitiveId::Select => CoreLoweringFamily::Select,
        PrimitiveId::TensorAlloc { .. } => CoreLoweringFamily::TensorAlloc,
        PrimitiveId::Fill { .. } => CoreLoweringFamily::Fill,
        PrimitiveId::Materialize | PrimitiveId::Clone | PrimitiveId::Load => {
            CoreLoweringFamily::BulkCopy
        }
        PrimitiveId::Decode => CoreLoweringFamily::PackedDecode,
        PrimitiveId::PackedRead(_) => CoreLoweringFamily::PackedPlaneRead,
        PrimitiveId::Transpose | PrimitiveId::Reshape | PrimitiveId::SliceView { .. } => {
            CoreLoweringFamily::ViewTransform
        }
        PrimitiveId::ElementRead { .. } => CoreLoweringFamily::ElementRead,
        PrimitiveId::ElementWrite { .. } => CoreLoweringFamily::ElementWrite,
        PrimitiveId::CopyInto => CoreLoweringFamily::CopyInto,
        PrimitiveId::Extent { .. } => CoreLoweringFamily::ExtentRead,
        PrimitiveId::ValidExtent { .. } => CoreLoweringFamily::RuntimeExtent,
        PrimitiveId::Atomic { .. } => CoreLoweringFamily::Atomic,
        PrimitiveId::Reduce { .. } => CoreLoweringFamily::Reduce,
    }
}

/// One primitive's complete signature.
#[derive(Clone, Debug, PartialEq)]
pub struct PrimitiveSignature {
    pub id: PrimitiveId,
    pub parameters: Vec<TypePattern>,
    pub result: TypeFunction,
    pub effects: EffectFunction,
    pub safety: SafetyFunction,
    pub reference: ReferenceSemantics,
    pub numerical: ReferenceNumerics,
    /// The core kernel-operation family that lowers this primitive.
    pub lowering: CoreLoweringFamily,
}

impl PrimitiveSignature {
    /// The result type under concrete operand types, when the operands alone
    /// determine it. Reductions and structural primitives require their id
    /// payload; use `PrimitiveSignature::of` and `reduction_result`.
    pub fn result_type(&self, operands: &[ValueType]) -> Option<ValueType> {
        if let PrimitiveId::Reduce { op, axis, .. } = &self.id {
            return reduction_result(operands.first()?, *op, *axis);
        }
        self.result.apply(operands)
    }

    /// Whether the operand types match the declared parameter patterns.
    pub fn accepts(&self, operands: &[ValueType]) -> bool {
        operands.len() == self.parameters.len()
            && operands
                .iter()
                .zip(&self.parameters)
                .all(|(t, p)| p.matches(t))
    }
}

/// Numerical transfer of a capability (the registry-level form of the
/// whole-program transfer taxonomy; strategy-level transfers extend it in the
/// realization layer).
#[derive(Clone, Debug, PartialEq)]
pub enum NumericalTransfer {
    Exact,
    Round {
        dtype: DType,
    },
    Approximate {
        operation: String,
        bound: Option<ErrorBound>,
    },
    Capability {
        signature: IntrinsicId,
        bound: Option<ErrorBound>,
    },
    Unknown {
        reason: String,
    },
}

/// A relative/absolute error bound qualifying an approximate alternative.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErrorBound {
    pub relative: f64,
    pub absolute: f64,
}

/// Reference meaning of a capability intrinsic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CapabilitySemantics {
    /// Index of the participant within its group.
    ParticipantIndex,
    /// Exchange a value with another participant of the group.
    Exchange,
    /// Reduction over the participants of a group.
    SubgroupReduction(ReduceOp),
    /// Logical matrix multiplication with an explicit accumulation dtype.
    MatrixMatmul,
    /// Logical matrix multiplication added to an accumulator.
    MatrixMatmulAdd,
}

/// The capability family of an intrinsic: which backend feature namespace
/// realizes it. Derived from `CapabilitySemantics`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CapabilityFamily {
    /// Participant-group (subgroup/warp/SIMD-group) operations.
    Subgroup,
    /// Cooperative matrix (simdgroup matrix / tensor core) operations.
    Matrix,
}

/// The capability family of one reference meaning. Exhaustive.
pub fn capability_family(semantics: CapabilitySemantics) -> CapabilityFamily {
    match semantics {
        CapabilitySemantics::ParticipantIndex
        | CapabilitySemantics::Exchange
        | CapabilitySemantics::SubgroupReduction(_) => CapabilityFamily::Subgroup,
        CapabilitySemantics::MatrixMatmul | CapabilitySemantics::MatrixMatmulAdd => {
            CapabilityFamily::Matrix
        }
    }
}

/// The typed lowering of one capability intrinsic: every operand role,
/// result role, shape, and dtype a backend `IntrinsicCatalog::lower` needs,
/// so that lowering is a mechanical table over this enum and never a
/// reconstruction from names or operand inspection.
///
/// Operand ordinals are the positions in `CapabilitySignature::arguments`;
/// the roles below state them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntrinsicLowering {
    /// No operands; result `i32`: the index of the participant in its group.
    ParticipantIndex,
    /// Operand 0: the value (`dtype`); operand 1: the source participant
    /// index (`i32`); result: the exchanged value (`dtype`).
    Exchange { dtype: DType },
    /// Operand 0: the participant's value (`dtype`); result: the reduction
    /// of every participant's value (`dtype`).
    SubgroupReduce { op: ReduceOp, dtype: DType },
    /// Operand 0: left `[rows, inner]` of `elem`; operand 1: right
    /// `[inner, columns]` of `elem`; result: `[rows, columns]` of
    /// `accumulator`.
    MatrixMatmul {
        rows: ExtentExpr,
        columns: ExtentExpr,
        inner: ExtentExpr,
        elem: DType,
        accumulator: DType,
    },
    /// Operand 0: left `[rows, inner]` of `elem`; operand 1: right
    /// `[inner, columns]` of `elem`; operand 2: accumulator
    /// `[rows, columns]` of `accumulator`; result: `[rows, columns]` of
    /// `accumulator`.
    MatrixMatmulAdd {
        rows: ExtentExpr,
        columns: ExtentExpr,
        inner: ExtentExpr,
        elem: DType,
        accumulator: DType,
    },
}

/// One capability intrinsic's complete signature.
#[derive(Clone, Debug, PartialEq)]
pub struct CapabilitySignature {
    pub id: IntrinsicId,
    pub arguments: Vec<ConcreteType>,
    pub result: ConcreteType,
    pub semantics: CapabilitySemantics,
    pub numerical: NumericalTransfer,
    /// The backend feature namespace realizing the intrinsic.
    pub family: CapabilityFamily,
    /// The only legal target of this signature (`id.capability.backend`).
    pub backend: String,
    /// The typed operand/result roles a backend lowers mechanically.
    pub lowering: IntrinsicLowering,
}

/// The concrete dtype of a rank-two matrix operand of a capability
/// signature. A non-concrete or packed element is a defect: capability
/// lowering happens on specialized programs, and the registry admits dense
/// dtype elements only.
fn matrix_operand(id: &IntrinsicId, role: &str, ty: &ValueType) -> (DType, [ExtentExpr; 2]) {
    let ValueType::Tensor(tensor) = ty else {
        panic!("`{id}` {role} operand must be a rank-two tensor, found `{ty}`");
    };
    let [first, second] = tensor.axes.as_slice() else {
        panic!("`{id}` {role} operand must be rank two, found `{ty}`");
    };
    let Elem::Dtype(dtype) = &tensor.elem else {
        panic!("`{id}` {role} operand element must be a concrete dtype, found `{ty}`");
    };
    (*dtype, [first.clone(), second.clone()])
}

/// The scalar dtype of a scalar operand of a capability signature.
fn scalar_operand(id: &IntrinsicId, role: &str, ty: &ValueType) -> DType {
    let ValueType::Scalar(dtype) = ty else {
        panic!("`{id}` {role} operand must be a scalar, found `{ty}`");
    };
    *dtype
}

/// The typed lowering of one capability signature from its reference
/// meaning and exact operand/result types. Total over registry entries and
/// checked uses; a structural mismatch is a defect named by `id`.
fn intrinsic_lowering(
    id: &IntrinsicId,
    semantics: CapabilitySemantics,
    arguments: &[ValueType],
    result: &ValueType,
) -> IntrinsicLowering {
    match semantics {
        CapabilitySemantics::ParticipantIndex => {
            if !arguments.is_empty() {
                panic!("`{id}` takes no operands, found {}", arguments.len());
            }
            IntrinsicLowering::ParticipantIndex
        }
        CapabilitySemantics::Exchange => {
            let [value, lane] = arguments else {
                panic!("`{id}` takes a value and a participant index, found {}", arguments.len());
            };
            let dtype = scalar_operand(id, "value", value);
            if scalar_operand(id, "participant index", lane) != DType::I32 {
                panic!("`{id}` participant index must be `i32`, found `{lane}`");
            }
            if scalar_operand(id, "result", result) != dtype {
                panic!("`{id}` result must be `{}`, found `{result}`", dtype.name());
            }
            IntrinsicLowering::Exchange { dtype }
        }
        CapabilitySemantics::SubgroupReduction(op) => {
            let [value] = arguments else {
                panic!("`{id}` takes one value operand, found {}", arguments.len());
            };
            let dtype = scalar_operand(id, "value", value);
            if scalar_operand(id, "result", result) != dtype {
                panic!("`{id}` result must be `{}`, found `{result}`", dtype.name());
            }
            IntrinsicLowering::SubgroupReduce { op, dtype }
        }
        // The checker proved the inner axes equal (by the prover, not
        // structurally) and built the result axes from the operand axes; the
        // lowering carries the left operand's `rows`/`inner`, the right
        // operand's `columns`, and the result's element. Operand element
        // equality is guaranteed by the registry entry (`entry`) or by
        // `signature_admits` (a use).
        CapabilitySemantics::MatrixMatmul => {
            let [left, right] = arguments else {
                panic!("`{id}` takes two matrix operands, found {}", arguments.len());
            };
            let (elem, [rows, inner]) = matrix_operand(id, "left", left);
            let (_, [_, columns]) = matrix_operand(id, "right", right);
            let (accumulator, _) = matrix_operand(id, "result", result);
            IntrinsicLowering::MatrixMatmul {
                rows,
                columns,
                inner,
                elem,
                accumulator,
            }
        }
        CapabilitySemantics::MatrixMatmulAdd => {
            let [left, right, addend] = arguments else {
                panic!(
                    "`{id}` takes two matrix operands and an accumulator, found {}",
                    arguments.len()
                );
            };
            let (elem, [rows, inner]) = matrix_operand(id, "left", left);
            let (_, [_, columns]) = matrix_operand(id, "right", right);
            let (addend_elem, _) = matrix_operand(id, "accumulator", addend);
            let (accumulator, _) = matrix_operand(id, "result", result);
            if addend_elem != accumulator {
                panic!(
                    "`{id}` accumulator `{addend}` and result `{result}` elements differ"
                );
            }
            IntrinsicLowering::MatrixMatmulAdd {
                rows,
                columns,
                inner,
                elem,
                accumulator,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The registry
// ---------------------------------------------------------------------------

fn rows2(elem: Elem) -> ValueType {
    ValueType::Tensor(TensorType::new(
        vec![
            ExtentExpr::Sym(Sym::param("rows")),
            ExtentExpr::Sym(Sym::param("columns")),
        ],
        elem,
    ))
}

fn inner2(elem: Elem) -> ValueType {
    ValueType::Tensor(TensorType::new(
        vec![
            ExtentExpr::Sym(Sym::param("rows")),
            ExtentExpr::Sym(Sym::param("inner")),
        ],
        elem,
    ))
}

fn columns2(elem: Elem) -> ValueType {
    ValueType::Tensor(TensorType::new(
        vec![
            ExtentExpr::Sym(Sym::param("inner")),
            ExtentExpr::Sym(Sym::param("columns")),
        ],
        elem,
    ))
}

/// The signature of one primitive: the single closed table consulted by
/// checking, reference execution, logical construction, legalization, and
/// numerical analysis.
pub fn primitive(id: PrimitiveId) -> PrimitiveSignature {
    let sig = |parameters, result, effects, safety, reference, numerical| PrimitiveSignature {
        id: id.clone(),
        parameters,
        result,
        effects,
        safety,
        reference,
        numerical,
        lowering: lowering(&id),
    };
    let bulk_copy = |reference: ReferenceSemantics| {
        sig(
            vec![TypePattern::TensorOf(ElemClass::Any)],
            TypeFunction::SameAs(0),
            EffectFunction::Allocates,
            SafetyFunction::None,
            reference,
            ReferenceNumerics::Exact,
        )
    };
    let ew = |class| TypePattern::Elementwise(ElemClass::Dense(class));
    match &id {
        PrimitiveId::TuplePack => sig(
            vec![TypePattern::Any; 2],
            TypeFunction::TupleOfOperands,
            EffectFunction::Pure,
            SafetyFunction::None,
            ReferenceSemantics::TupleOp,
            ReferenceNumerics::Exact,
        ),
        PrimitiveId::TupleGet(_) => sig(
            vec![TypePattern::Any],
            TypeFunction::Structural,
            EffectFunction::Pure,
            SafetyFunction::None,
            ReferenceSemantics::TupleOp,
            ReferenceNumerics::Exact,
        ),
        PrimitiveId::RangeMake => sig(
            vec![TypePattern::ScalarOf(DTypeClass::Int); 2],
            TypeFunction::Structural,
            EffectFunction::Pure,
            SafetyFunction::None,
            ReferenceSemantics::RangeOp,
            ReferenceNumerics::Exact,
        ),
        PrimitiveId::RangeStart | PrimitiveId::RangeEnd => sig(
            vec![TypePattern::Range],
            TypeFunction::Fixed(ValueType::Scalar(DType::I32)),
            EffectFunction::Pure,
            SafetyFunction::None,
            ReferenceSemantics::RangeOp,
            ReferenceNumerics::Exact,
        ),
        PrimitiveId::Unary(op) => {
            let class = match op {
                UnaryOp::Neg => DTypeClass::Numeric,
                UnaryOp::Not => DTypeClass::Bool,
                UnaryOp::BitNot => DTypeClass::Int,
            };
            sig(
                vec![ew(class)],
                TypeFunction::Elementwise {
                    shape_of: 0,
                    dtype: ResultDType::SameAs(0),
                },
                EffectFunction::Pure,
                SafetyFunction::None,
                ReferenceSemantics::ElementwiseOp,
                ReferenceNumerics::RoundsOnce { dtype: DType::F32 },
            )
        }
        PrimitiveId::Binary(op) => {
            let comparison = matches!(
                op,
                BinaryOp::Eq
                    | BinaryOp::Ne
                    | BinaryOp::Lt
                    | BinaryOp::Le
                    | BinaryOp::Gt
                    | BinaryOp::Ge
            );
            let logic = matches!(op, BinaryOp::And | BinaryOp::Or);
            let shift = matches!(op, BinaryOp::Shl | BinaryOp::Shr);
            let bit = matches!(op, BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor);
            let class = if logic {
                DTypeClass::Bool
            } else if shift || bit {
                DTypeClass::Int
            } else {
                DTypeClass::Numeric
            };
            let result = if comparison || logic {
                TypeFunction::Elementwise {
                    shape_of: 0,
                    dtype: ResultDType::Dtype(DType::Bool),
                }
            } else if shift {
                TypeFunction::Elementwise {
                    shape_of: 0,
                    dtype: ResultDType::SameAs(0),
                }
            } else {
                TypeFunction::Elementwise {
                    shape_of: 0,
                    dtype: ResultDType::Promoted(vec![0, 1]),
                }
            };
            let (safety, numerical) = match op {
                BinaryOp::Div | BinaryOp::Rem => (
                    SafetyFunction::DivisorNonZero,
                    ReferenceNumerics::EuclideanDivision,
                ),
                BinaryOp::Shl | BinaryOp::Shr => {
                    (SafetyFunction::ShiftInRange, ReferenceNumerics::Shifts)
                }
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul => (
                    SafetyFunction::None,
                    ReferenceNumerics::Wraps { dtype: DType::I32 },
                ),
                _ => (
                    SafetyFunction::None,
                    ReferenceNumerics::RoundsOnce { dtype: DType::F32 },
                ),
            };
            sig(
                vec![ew(class), ew(if shift { DTypeClass::Int } else { class })],
                result,
                EffectFunction::Pure,
                safety,
                ReferenceSemantics::ElementwiseOp,
                numerical,
            )
        }
        PrimitiveId::Cast(dtype) => sig(
            vec![TypePattern::Elementwise(ElemClass::Any)],
            TypeFunction::Elementwise {
                shape_of: 0,
                dtype: ResultDType::Dtype(*dtype),
            },
            EffectFunction::Pure,
            SafetyFunction::None,
            ReferenceSemantics::ElementwiseOp,
            if dtype.is_int() {
                ReferenceNumerics::LowBitsCast
            } else {
                ReferenceNumerics::ConvertingCast { dtype: *dtype }
            },
        ),
        PrimitiveId::Math(op) => {
            let class = if op.numeric_operands() {
                DTypeClass::Numeric
            } else {
                DTypeClass::Float
            };
            let arity = op.arity();
            sig(
                vec![ew(class); arity],
                TypeFunction::Elementwise {
                    shape_of: 0,
                    dtype: ResultDType::Promoted((0..arity).collect()),
                },
                EffectFunction::Pure,
                SafetyFunction::None,
                ReferenceSemantics::ElementwiseOp,
                ReferenceNumerics::SoftwareMath {
                    algorithm: "seismic_math-v1",
                },
            )
        }
        PrimitiveId::Select => sig(
            vec![
                ew(DTypeClass::Bool),
                TypePattern::Elementwise(ElemClass::Dense(DTypeClass::Any)),
                TypePattern::Elementwise(ElemClass::Dense(DTypeClass::Any)),
            ],
            TypeFunction::Elementwise {
                shape_of: 1,
                dtype: ResultDType::Promoted(vec![1, 2]),
            },
            EffectFunction::Pure,
            SafetyFunction::None,
            ReferenceSemantics::ElementwiseOp,
            ReferenceNumerics::RoundsOnce { dtype: DType::F32 },
        ),
        PrimitiveId::TensorAlloc { .. } => sig(
            Vec::new(),
            TypeFunction::Structural,
            EffectFunction::Allocates,
            SafetyFunction::None,
            ReferenceSemantics::AllocateUninitialized,
            ReferenceNumerics::Exact,
        ),
        PrimitiveId::Fill { dtype, .. } => sig(
            vec![TypePattern::Any],
            TypeFunction::Structural,
            EffectFunction::Allocates,
            SafetyFunction::None,
            ReferenceSemantics::FillConstant,
            ReferenceNumerics::RoundsOnce { dtype: *dtype },
        ),
        PrimitiveId::Materialize => bulk_copy(ReferenceSemantics::MaterializeSnapshot),
        PrimitiveId::Clone => bulk_copy(ReferenceSemantics::CloneOwned),
        PrimitiveId::Load => bulk_copy(ReferenceSemantics::LoadSnapshot),
        PrimitiveId::Decode => sig(
            vec![TypePattern::TensorOf(ElemClass::Packed)],
            TypeFunction::Elementwise {
                shape_of: 0,
                dtype: ResultDType::Dtype(DType::F32),
            },
            EffectFunction::Allocates,
            SafetyFunction::None,
            ReferenceSemantics::DecodePacked,
            ReferenceNumerics::RoundsOnce { dtype: DType::F32 },
        ),
        PrimitiveId::PackedRead(_) => sig(
            vec![TypePattern::TensorOf(ElemClass::Packed)],
            TypeFunction::Structural,
            EffectFunction::Reads,
            SafetyFunction::None,
            ReferenceSemantics::ReadPackedPlane,
            ReferenceNumerics::Exact,
        ),
        PrimitiveId::Transpose => sig(
            vec![TypePattern::TensorOf(ElemClass::Any)],
            TypeFunction::Structural,
            EffectFunction::Pure,
            SafetyFunction::None,
            ReferenceSemantics::ViewTransform,
            ReferenceNumerics::Exact,
        ),
        PrimitiveId::Reshape => sig(
            vec![TypePattern::TensorOf(ElemClass::Any)],
            TypeFunction::Structural,
            EffectFunction::Pure,
            SafetyFunction::None,
            ReferenceSemantics::ViewTransform,
            ReferenceNumerics::Exact,
        ),
        PrimitiveId::SliceView { .. } => sig(
            vec![TypePattern::TensorOf(ElemClass::Any)],
            TypeFunction::Structural,
            EffectFunction::Reads,
            SafetyFunction::RangeInBounds,
            ReferenceSemantics::ViewTransform,
            ReferenceNumerics::Exact,
        ),
        PrimitiveId::ElementRead { .. } => sig(
            vec![TypePattern::TensorOf(ElemClass::Any)],
            TypeFunction::Structural,
            EffectFunction::Reads,
            SafetyFunction::IndexInBounds,
            ReferenceSemantics::ReadElement,
            ReferenceNumerics::Exact,
        ),
        PrimitiveId::ElementWrite { .. } => sig(
            vec![TypePattern::TensorOf(ElemClass::Any)],
            TypeFunction::Fixed(ValueType::Void),
            EffectFunction::Writes { whole: false },
            SafetyFunction::IndexInBounds,
            ReferenceSemantics::ElementwiseOp,
            ReferenceNumerics::RoundsOnce { dtype: DType::F32 },
        ),
        PrimitiveId::CopyInto => sig(
            vec![
                TypePattern::TensorOf(ElemClass::Any),
                TypePattern::TensorOf(ElemClass::Any),
            ],
            TypeFunction::Fixed(ValueType::Void),
            EffectFunction::Writes { whole: false },
            SafetyFunction::RangeInBounds,
            ReferenceSemantics::ElementwiseOp,
            ReferenceNumerics::RoundsOnce { dtype: DType::F32 },
        ),
        PrimitiveId::Extent { .. } | PrimitiveId::ValidExtent { .. } => sig(
            vec![TypePattern::TensorOf(ElemClass::Any)],
            TypeFunction::Fixed(ValueType::Scalar(DType::I32)),
            EffectFunction::Reads,
            SafetyFunction::None,
            ReferenceSemantics::ReadExtent,
            ReferenceNumerics::Exact,
        ),
        PrimitiveId::Atomic { op, .. } => sig(
            Vec::new(),
            TypeFunction::Fixed(ValueType::Void),
            EffectFunction::Atomic,
            SafetyFunction::IndexInBounds,
            ReferenceSemantics::Atomic,
            match op {
                // `add` rounds the sum once at the element dtype.
                AtomicOp::Add => ReferenceNumerics::RoundsOnce { dtype: DType::F32 },
                // `max`/`min` select one of two representable values.
                AtomicOp::Max | AtomicOp::Min => ReferenceNumerics::Exact,
            },
        ),
        PrimitiveId::Reduce { op, .. } => {
            // The accumulator dtype is fixed by the registry rule from the
            // operand's dtype; `result_type` applies it to concrete operands.
            let accumulator = accumulator_dtype(*op, DType::F32);
            sig(
                vec![TypePattern::TensorOf(ElemClass::Dense(DTypeClass::Any))],
                TypeFunction::Reduction {
                    operand: 0,
                    op: *op,
                },
                EffectFunction::Reads,
                SafetyFunction::None,
                ReferenceSemantics::Reduce {
                    accumulator,
                    ascending: true,
                    smaller_coordinate_ties: true,
                },
                ReferenceNumerics::Accumulates { accumulator },
            )
        }
    }
}

/// Every admitted unary operator.
pub fn unary_ops() -> [UnaryOp; 3] {
    [UnaryOp::Neg, UnaryOp::Not, UnaryOp::BitNot]
}

/// Every admitted binary operator.
pub fn binary_ops() -> [BinaryOp; 18] {
    use BinaryOp::*;
    [
        Or, And, Eq, Ne, Lt, Le, Gt, Ge, BitOr, BitXor, BitAnd, Shl, Shr, Add, Sub, Mul, Div, Rem,
    ]
}

/// Every admitted math operation.
pub fn math_ops() -> [MathOp; 11] {
    use MathOp::*;
    [Fma, Exp, ExpFast, Rsqrt, Sqrt, Log, Sin, Cos, Abs, Max, Min]
}

/// Every admitted dtype cast.
pub fn cast_dtypes() -> [DType; 6] {
    use DType::*;
    [F32, BF16, F16, I32, U32, Bool]
}

/// Enumerate the full portable registry: one signature per concrete primitive
/// identity (operator payloads included).
pub fn primitives() -> Vec<PrimitiveSignature> {
    let mut out = vec![
        primitive(PrimitiveId::TuplePack),
        primitive(PrimitiveId::TupleGet(0)),
        primitive(PrimitiveId::RangeMake),
        primitive(PrimitiveId::RangeStart),
        primitive(PrimitiveId::RangeEnd),
        primitive(PrimitiveId::Select),
        primitive(PrimitiveId::Materialize),
        primitive(PrimitiveId::Clone),
        primitive(PrimitiveId::Load),
        primitive(PrimitiveId::Decode),
        primitive(PrimitiveId::Transpose),
        primitive(PrimitiveId::Reshape),
        primitive(PrimitiveId::SliceView { indices: vec![] }),
        primitive(PrimitiveId::ElementRead { arity: 1 }),
        primitive(PrimitiveId::ElementWrite { arity: 1 }),
        primitive(PrimitiveId::CopyInto),
        primitive(PrimitiveId::Extent { axis: 0 }),
        primitive(PrimitiveId::ValidExtent { axis: 0 }),
        primitive(PrimitiveId::Atomic {
            op: AtomicOp::Add,
            arity: 1,
        }),
    ];
    for op in unary_ops() {
        out.push(primitive(PrimitiveId::Unary(op)));
    }
    for op in binary_ops() {
        out.push(primitive(PrimitiveId::Binary(op)));
    }
    for op in math_ops() {
        out.push(primitive(PrimitiveId::Math(op)));
    }
    for dtype in cast_dtypes() {
        out.push(primitive(PrimitiveId::Cast(dtype)));
    }
    for op in [
        ReduceOp::Sum,
        ReduceOp::Max,
        ReduceOp::Min,
        ReduceOp::Argmax,
    ] {
        for unordered in [false, true] {
            if op == ReduceOp::Argmax && unordered {
                continue;
            }
            out.push(primitive(PrimitiveId::Reduce {
                op,
                axis: 0,
                unordered,
            }));
        }
    }
    for field in [
        PlaneField::Words,
        PlaneField::Scale,
        PlaneField::Bias,
        PlaneField::Coefficients,
        PlaneField::ScaleFactor,
        PlaneField::BiasFactor,
    ] {
        out.push(primitive(PrimitiveId::PackedRead(field)));
    }
    out.push(primitive(PrimitiveId::TensorAlloc {
        elem: Elem::Dtype(DType::F32),
    }));
    out.push(primitive(PrimitiveId::Fill {
        value: 0.0,
        dtype: DType::F32,
    }));
    out
}

/// Whether a dtype admits `atomic`: f32, f16, bf16, i32, u32. Bool is
/// rejected because bool addition is undefined.
pub fn atomic_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::F32 | DType::F16 | DType::BF16 | DType::I32 | DType::U32
    )
}

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------

fn entry(
    backend: &str,
    capability: &str,
    name: &str,
    arguments: Vec<ConcreteType>,
    result: ConcreteType,
    semantics: CapabilitySemantics,
    numerical: NumericalTransfer,
) -> CapabilitySignature {
    let id = IntrinsicId {
        capability: CapabilityId::new(backend, capability),
        name: name.into(),
    };
    let lowering = intrinsic_lowering(&id, semantics, &arguments, &result);
    CapabilitySignature {
        id,
        arguments,
        result,
        semantics,
        numerical,
        family: capability_family(semantics),
        backend: backend.into(),
        lowering,
    }
}

/// Whether one registry signature admits a checked use: the same identity,
/// the same arity, scalars exactly, tensors by rank and element. This is the
/// checker's admission relation sharpened to the element (the registry
/// declares one signature per matrix element dtype).
fn signature_admits(signature: &CapabilitySignature, intrinsic: &crate::sir::IntrinsicUse) -> bool {
    signature.id == intrinsic.id
        && signature.arguments.len() == intrinsic.arguments.len()
        && signature
            .arguments
            .iter()
            .zip(&intrinsic.arguments)
            .all(|(parameter, argument)| match (parameter, argument) {
                (ValueType::Scalar(p), ValueType::Scalar(a)) => p == a,
                (ValueType::Tensor(p), ValueType::Tensor(a)) => {
                    p.rank() == a.rank() && p.elem == a.elem
                }
                (ValueType::Index { .. }, argument) => {
                    argument.scalar_dtype() == Some(DType::I32)
                }
                (parameter, argument) => parameter == argument,
            })
}

/// The complete signature of one checked capability use, instantiated at
/// the use's exact operand and result types (the registry entry's symbolic
/// matrix shapes become the use's shapes; the lowering carries them). Total
/// over checked uses: checking admitted the use against exactly one
/// registry signature, so no admitting signature or more than one is a
/// defect, reported with the use.
pub fn capability_signature(intrinsic: &crate::sir::IntrinsicUse) -> CapabilitySignature {
    let admitting: Vec<CapabilitySignature> = capabilities()
        .into_iter()
        .filter(|signature| signature_admits(signature, intrinsic))
        .collect();
    let describe = || {
        let arguments: Vec<String> = intrinsic.arguments.iter().map(|t| t.to_string()).collect();
        format!(
            "`{}`({}) -> {}",
            intrinsic.id,
            arguments.join(", "),
            intrinsic.result
        )
    };
    let signature = match admitting.as_slice() {
        [signature] => signature,
        [] => panic!("checked capability use {} has no registry signature", describe()),
        [_, _, ..] => panic!(
            "checked capability use {} is admitted by {} registry signatures",
            describe(),
            admitting.len()
        ),
    };
    let lowering = intrinsic_lowering(
        &intrinsic.id,
        signature.semantics,
        &intrinsic.arguments,
        &intrinsic.result,
    );
    CapabilitySignature {
        id: intrinsic.id.clone(),
        arguments: intrinsic.arguments.clone(),
        result: intrinsic.result.clone(),
        semantics: signature.semantics,
        numerical: signature.numerical.clone(),
        family: signature.family,
        backend: signature.backend.clone(),
        lowering,
    }
}

/// The complete author-visible capability registry. Availability on an
/// effective target is a later intersection with planner, emitter, toolchain,
/// and hardware facts.
pub fn capabilities() -> Vec<CapabilitySignature> {
    let mut out = Vec::new();
    for backend in ["metal", "cuda"] {
        out.push(entry(
            backend,
            "subgroup",
            "lane_index",
            vec![],
            ValueType::Scalar(DType::I32),
            CapabilitySemantics::ParticipantIndex,
            NumericalTransfer::Exact,
        ));
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            out.push(entry(
                backend,
                "subgroup",
                "shuffle",
                vec![ValueType::Scalar(dtype), ValueType::Scalar(DType::I32)],
                ValueType::Scalar(dtype),
                CapabilitySemantics::Exchange,
                NumericalTransfer::Round { dtype },
            ));
            for (name, op) in [
                ("simd_sum", ReduceOp::Sum),
                ("simd_max", ReduceOp::Max),
                ("simd_min", ReduceOp::Min),
            ] {
                out.push(entry(
                    backend,
                    "subgroup",
                    name,
                    vec![ValueType::Scalar(dtype)],
                    ValueType::Scalar(dtype),
                    CapabilitySemantics::SubgroupReduction(op),
                    NumericalTransfer::Capability {
                        signature: IntrinsicId {
                            capability: CapabilityId::new(backend, "subgroup"),
                            name: name.into(),
                        },
                        bound: None,
                    },
                ));
            }
        }
        for elem in [Elem::Dtype(DType::F16), Elem::Dtype(DType::F32)] {
            out.push(entry(
                backend,
                "matrix",
                "matmul",
                vec![inner2(elem.clone()), columns2(elem.clone())],
                rows2(Elem::Dtype(DType::F32)),
                CapabilitySemantics::MatrixMatmul,
                NumericalTransfer::Capability {
                    signature: IntrinsicId {
                        capability: CapabilityId::new(backend, "matrix"),
                        name: "matmul".into(),
                    },
                    bound: None,
                },
            ));
            out.push(entry(
                backend,
                "matrix",
                "matmul_add",
                vec![
                    inner2(elem.clone()),
                    columns2(elem.clone()),
                    rows2(elem.clone()),
                ],
                rows2(elem),
                CapabilitySemantics::MatrixMatmulAdd,
                NumericalTransfer::Capability {
                    signature: IntrinsicId {
                        capability: CapabilityId::new(backend, "matrix"),
                        name: "matmul_add".into(),
                    },
                    bound: None,
                },
            ));
        }
    }
    out
}

pub fn known_backend(backend: &str) -> bool {
    matches!(backend, "metal" | "cuda" | "cpu" | "vulkan")
}

/// The namespace identity, when the backend exposes that capability namespace.
pub fn capability(backend: &str, name: &str) -> Option<CapabilityId> {
    capabilities()
        .into_iter()
        .any(|entry| entry.id.capability.backend == backend && entry.id.capability.name == name)
        .then(|| CapabilityId::new(backend, name))
}

/// Every signature of one capability intrinsic.
pub fn lookup(backend: &str, capability: &str, name: &str) -> Vec<CapabilitySignature> {
    capabilities()
        .into_iter()
        .filter(|entry| {
            entry.id.capability.backend == backend
                && entry.id.capability.name == capability
                && entry.id.name == name
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floating_ordered_reduction_uses_f32_accumulator() {
        for input in [DType::F16, DType::BF16, DType::F32] {
            assert_eq!(accumulator_dtype(ReduceOp::Sum, input), DType::F32);
        }
        for input in [DType::I32, DType::U32, DType::Bool] {
            assert_eq!(accumulator_dtype(ReduceOp::Sum, input), input);
            assert_eq!(accumulator_dtype(ReduceOp::Max, input), input);
        }
        assert_eq!(accumulator_dtype(ReduceOp::Argmax, DType::F16), DType::I32);
    }

    #[test]
    fn reduction_result_agrees_with_accumulator() {
        let operand = ValueType::Tensor(TensorType::new(
            vec![
                ExtentExpr::Sym(Sym::param("M")),
                ExtentExpr::Sym(Sym::param("K")),
            ],
            Elem::Dtype(DType::F16),
        ));
        let result = reduction_result(&operand, ReduceOp::Sum, 1).unwrap();
        assert_eq!(
            result,
            ValueType::Tensor(TensorType::new(
                vec![ExtentExpr::Sym(Sym::param("M"))],
                Elem::Dtype(DType::F32)
            ))
        );
        let argmax = reduction_result(&operand, ReduceOp::Argmax, 1).unwrap();
        assert_eq!(
            argmax,
            ValueType::Tensor(TensorType::new(
                vec![ExtentExpr::Sym(Sym::param("M"))],
                Elem::Dtype(DType::I32)
            ))
        );
    }

    #[test]
    fn every_primitive_carries_its_lowering_family() {
        for signature in primitives() {
            assert_eq!(signature.lowering, lowering(&signature.id), "{}", signature.id);
            let expected_by_reference = match &signature.reference {
                ReferenceSemantics::TupleOp | ReferenceSemantics::RangeOp => {
                    Some(CoreLoweringFamily::Structural)
                }
                ReferenceSemantics::AllocateUninitialized => Some(CoreLoweringFamily::TensorAlloc),
                ReferenceSemantics::FillConstant => Some(CoreLoweringFamily::Fill),
                ReferenceSemantics::MaterializeSnapshot
                | ReferenceSemantics::CloneOwned
                | ReferenceSemantics::LoadSnapshot => Some(CoreLoweringFamily::BulkCopy),
                ReferenceSemantics::DecodePacked => Some(CoreLoweringFamily::PackedDecode),
                ReferenceSemantics::ReadPackedPlane => Some(CoreLoweringFamily::PackedPlaneRead),
                ReferenceSemantics::ViewTransform => Some(CoreLoweringFamily::ViewTransform),
                ReferenceSemantics::ReadElement => Some(CoreLoweringFamily::ElementRead),
                ReferenceSemantics::Atomic => Some(CoreLoweringFamily::Atomic),
                ReferenceSemantics::Reduce { .. } => Some(CoreLoweringFamily::Reduce),
                // Several families share these reference meanings; the id
                // decides (checked below).
                ReferenceSemantics::ScalarOp
                | ReferenceSemantics::ElementwiseOp
                | ReferenceSemantics::ReadExtent => None,
            };
            if let Some(expected) = expected_by_reference {
                assert_eq!(signature.lowering, expected, "{}", signature.id);
            }
        }
        assert_eq!(lowering(&PrimitiveId::Binary(BinaryOp::Lt)), CoreLoweringFamily::ScalarArithmetic);
        assert_eq!(lowering(&PrimitiveId::Math(MathOp::Fma)), CoreLoweringFamily::ElementwiseMap);
        assert_eq!(lowering(&PrimitiveId::Cast(DType::BF16)), CoreLoweringFamily::Cast);
        assert_eq!(lowering(&PrimitiveId::Select), CoreLoweringFamily::Select);
        assert_eq!(lowering(&PrimitiveId::CopyInto), CoreLoweringFamily::CopyInto);
        assert_eq!(
            lowering(&PrimitiveId::ElementWrite { arity: 2 }),
            CoreLoweringFamily::ElementWrite
        );
        assert_eq!(lowering(&PrimitiveId::Extent { axis: 0 }), CoreLoweringFamily::ExtentRead);
        assert_eq!(
            lowering(&PrimitiveId::ValidExtent { axis: 0 }),
            CoreLoweringFamily::RuntimeExtent
        );
    }

    #[test]
    fn capability_signatures_carry_family_backend_and_typed_lowering() {
        for signature in capabilities() {
            assert_eq!(signature.backend, signature.id.capability.backend);
            assert_eq!(signature.family, capability_family(signature.semantics));
            match (&signature.semantics, &signature.lowering) {
                (CapabilitySemantics::ParticipantIndex, IntrinsicLowering::ParticipantIndex) => {}
                (CapabilitySemantics::Exchange, IntrinsicLowering::Exchange { dtype }) => {
                    assert_eq!(signature.result, ValueType::Scalar(*dtype));
                }
                (
                    CapabilitySemantics::SubgroupReduction(op),
                    IntrinsicLowering::SubgroupReduce { op: lowered, dtype },
                ) => {
                    assert_eq!(op, lowered);
                    assert_eq!(signature.result, ValueType::Scalar(*dtype));
                }
                (
                    CapabilitySemantics::MatrixMatmul,
                    IntrinsicLowering::MatrixMatmul {
                        rows,
                        columns,
                        inner,
                        elem,
                        accumulator,
                    },
                ) => {
                    assert_eq!(*rows, ExtentExpr::Sym(Sym::param("rows")));
                    assert_eq!(*columns, ExtentExpr::Sym(Sym::param("columns")));
                    assert_eq!(*inner, ExtentExpr::Sym(Sym::param("inner")));
                    assert_eq!(signature.arguments[0], inner2(Elem::Dtype(*elem)));
                    assert_eq!(*accumulator, DType::F32);
                }
                (
                    CapabilitySemantics::MatrixMatmulAdd,
                    IntrinsicLowering::MatrixMatmulAdd {
                        elem, accumulator, ..
                    },
                ) => {
                    assert_eq!(elem, accumulator);
                    assert_eq!(signature.arguments[2], rows2(Elem::Dtype(*accumulator)));
                }
                (semantics, lowering) => {
                    panic!("`{}`: {semantics:?} lowered as {lowering:?}", signature.id)
                }
            }
        }
    }

    #[test]
    fn capability_signature_of_a_checked_use_instantiates_the_use_types() {
        let id = IntrinsicId {
            capability: CapabilityId::new("metal", "matrix"),
            name: "matmul".into(),
        };
        let matrix = |rows: u64, columns: u64, dtype: DType| {
            ValueType::Tensor(TensorType::new(
                vec![ExtentExpr::Static(rows), ExtentExpr::Static(columns)],
                Elem::Dtype(dtype),
            ))
        };
        let used = crate::sir::IntrinsicUse {
            id: id.clone(),
            arguments: vec![matrix(64, 32, DType::F16), matrix(32, 16, DType::F16)],
            result: matrix(64, 16, DType::F32),
        };
        let signature = capability_signature(&used);
        assert_eq!(signature.id, id);
        assert_eq!(signature.backend, "metal");
        assert_eq!(signature.family, CapabilityFamily::Matrix);
        assert_eq!(signature.arguments, used.arguments);
        assert_eq!(signature.result, used.result);
        assert_eq!(
            signature.lowering,
            IntrinsicLowering::MatrixMatmul {
                rows: ExtentExpr::Static(64),
                columns: ExtentExpr::Static(16),
                inner: ExtentExpr::Static(32),
                elem: DType::F16,
                accumulator: DType::F32,
            }
        );

        let shuffle = crate::sir::IntrinsicUse {
            id: IntrinsicId {
                capability: CapabilityId::new("cuda", "subgroup"),
                name: "shuffle".into(),
            },
            arguments: vec![ValueType::Scalar(DType::BF16), ValueType::Scalar(DType::I32)],
            result: ValueType::Scalar(DType::BF16),
        };
        assert_eq!(
            capability_signature(&shuffle).lowering,
            IntrinsicLowering::Exchange { dtype: DType::BF16 }
        );
    }

    #[test]
    #[should_panic(expected = "has no registry signature")]
    fn capability_signature_rejects_an_unregistered_use_as_a_defect() {
        let unregistered = crate::sir::IntrinsicUse {
            id: IntrinsicId {
                capability: CapabilityId::new("metal", "subgroup"),
                name: "shuffle".into(),
            },
            arguments: vec![ValueType::Scalar(DType::I32), ValueType::Scalar(DType::I32)],
            result: ValueType::Scalar(DType::I32),
        };
        capability_signature(&unregistered);
    }

    #[test]
    fn reduce_schema_agrees_with_accumulator_dtype_and_laws() {
        for input in [DType::F16, DType::BF16, DType::F32, DType::I32, DType::U32, DType::Bool] {
            for op in [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min] {
                let schema = reduce_schema(op, input);
                assert_eq!(schema.result, accumulator_dtype(op, input));
                assert_eq!(schema.accumulator, schema.result);
                assert_eq!(schema.combine, CombineLaw::AssociativeCommutative);
                assert_eq!(schema.ties, TieRule::SmallerCoordinateIndex);
            }
            let argmax = reduce_schema(ReduceOp::Argmax, input);
            assert_eq!(argmax.result, DType::I32);
            assert_eq!(argmax.accumulator, input);
            assert_eq!(argmax.identity, ReduceIdentity::FirstElementNonEmpty);
            assert_eq!(argmax.combine, CombineLaw::OrderedOnly);
        }
        assert_eq!(reduce_schema(ReduceOp::Sum, DType::F16).identity, ReduceIdentity::Zero);
        assert_eq!(reduce_schema(ReduceOp::Max, DType::U32).identity, ReduceIdentity::FirstElement);
    }

    #[test]
    fn registry_covers_every_operator_and_atomic_dtypes() {
        assert!(primitives().len() > 60);
        assert!(atomic_dtype(DType::F32));
        assert!(atomic_dtype(DType::U32));
        assert!(!atomic_dtype(DType::Bool));
        assert!(lookup("metal", "subgroup", "simd_sum").len() == 3);
        assert!(capability("cuda", "matrix").is_some());
    }
}
