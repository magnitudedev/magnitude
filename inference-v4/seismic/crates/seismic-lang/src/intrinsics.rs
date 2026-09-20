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
pub const REGISTRY_REVISION: &str = "seismic-registry-v3";

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

/// One capability intrinsic's complete signature.
#[derive(Clone, Debug, PartialEq)]
pub struct CapabilitySignature {
    pub id: IntrinsicId,
    pub arguments: Vec<ConcreteType>,
    pub result: ConcreteType,
    pub semantics: CapabilitySemantics,
    pub numerical: NumericalTransfer,
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
        PrimitiveId::Materialize | PrimitiveId::Clone | PrimitiveId::Load => sig(
            vec![TypePattern::TensorOf(ElemClass::Any)],
            TypeFunction::SameAs(0),
            EffectFunction::Allocates,
            SafetyFunction::None,
            match id {
                PrimitiveId::Materialize => ReferenceSemantics::MaterializeSnapshot,
                PrimitiveId::Clone => ReferenceSemantics::CloneOwned,
                _ => ReferenceSemantics::LoadSnapshot,
            },
            ReferenceNumerics::Exact,
        ),
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
    CapabilitySignature {
        id: IntrinsicId {
            capability: CapabilityId::new(backend, capability),
            name: name.into(),
        },
        arguments,
        result,
        semantics,
        numerical,
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
    fn registry_covers_every_operator_and_atomic_dtypes() {
        assert!(primitives().len() > 60);
        assert!(atomic_dtype(DType::F32));
        assert!(atomic_dtype(DType::U32));
        assert!(!atomic_dtype(DType::Bool));
        assert!(lookup("metal", "subgroup", "simd_sum").len() == 3);
        assert!(capability("cuda", "matrix").is_some());
    }
}
