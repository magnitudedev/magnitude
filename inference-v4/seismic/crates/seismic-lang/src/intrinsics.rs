//! The closed primitive vocabulary: operator enums shared by the checker, the
//! semantic program (`entry::NodeKind`), the reference interpreter and every
//! consumer of a `LogicalEntry`.
//!
//! Public items are closed enums and pure functions over them. Signature
//! tables (parameter patterns, result functions) and the capability
//! intrinsic table are crate-private: the checker consumes them by structure
//! and `registry` interns the capability table behind typed ids. No string
//! lookup exists here.

use crate::expr::IntExpr;
use crate::ids::RepresentationId;
use crate::syntax::ast::{BinaryOp, UnaryOp};
use crate::types::{DType, Elem, NonEmpty, TensorType, ValueType};

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
    pub const ALL: [MathOp; 11] = [
        MathOp::Fma,
        MathOp::Exp,
        MathOp::ExpFast,
        MathOp::Rsqrt,
        MathOp::Sqrt,
        MathOp::Log,
        MathOp::Sin,
        MathOp::Cos,
        MathOp::Abs,
        MathOp::Max,
        MathOp::Min,
    ];

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

    pub fn parse(name: &str) -> Option<MathOp> {
        MathOp::ALL.into_iter().find(|op| op.name() == name)
    }

    pub fn arity(self) -> usize {
        match self {
            MathOp::Fma => 3,
            MathOp::Max | MathOp::Min => 2,
            _ => 1,
        }
    }

    /// `true` when the operation is defined on numeric (not only float)
    /// operands.
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
    pub const ALL: [ReduceOp; 4] = [
        ReduceOp::Sum,
        ReduceOp::Max,
        ReduceOp::Min,
        ReduceOp::Argmax,
    ];

    pub fn name(self) -> &'static str {
        match self {
            ReduceOp::Sum => "sum",
            ReduceOp::Max => "max",
            ReduceOp::Min => "min",
            ReduceOp::Argmax => "argmax",
        }
    }

    pub fn parse(name: &str) -> Option<ReduceOp> {
        ReduceOp::ALL.into_iter().find(|op| op.name() == name)
    }
}

/// Structure of one index slot of a view selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IndexSlot {
    Point,
    /// `lo:hi` with present bounds; omitted bounds are the axis ends.
    Range {
        start: bool,
        end: bool,
    },
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
    pub const ALL: [AtomicOp; 3] = [AtomicOp::Add, AtomicOp::Max, AtomicOp::Min];

    pub fn name(self) -> &'static str {
        match self {
            AtomicOp::Add => "add",
            AtomicOp::Max => "max",
            AtomicOp::Min => "min",
        }
    }

    pub fn parse(name: &str) -> Option<AtomicOp> {
        AtomicOp::ALL.into_iter().find(|op| op.name() == name)
    }
}

/// A typed scalar literal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Constant {
    Int(i64),
    Float(f64),
    Bool(bool),
}

/// The constant of a `zeros_like` / `ones_like` fill.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FillConstant {
    Zero,
    One,
}

impl FillConstant {
    pub fn value(self) -> f64 {
        match self {
            FillConstant::Zero => 0.0,
            FillConstant::One => 1.0,
        }
    }
}

/// The closed portable primitive vocabulary. Every checked expression is a
/// registry primitive, a capability intrinsic, or a static function-family
/// call; payloads carry only structure the operands cannot express.
#[derive(Clone, Debug, PartialEq)]
pub enum PrimitiveId {
    /// A typed literal; the node's output type supplies the dtype.
    Constant(Constant),
    /// The `i32` value of a symbolic integer expression over the entry's
    /// dimensions, scalar parameters and loop binders, in the program's arena.
    Symbolic(IntExpr),
    /// Tuple construction from two or more components.
    TuplePack,
    /// Ordinal projection of a tuple.
    TupleGet(u32),
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
    /// `tensor[shape] elem`: uninitialized owned storage; the node's output
    /// type carries shape and element.
    TensorAlloc,
    /// `zeros_like` / `ones_like`: shape of the operand, constant fill.
    Fill(FillConstant),
    /// `to_owned`: new owned storage from a borrowed or computed value.
    Materialize,
    /// `clone`: duplicate an owned tensor.
    Clone,
    /// `load`: snapshot in the operand's own representation.
    Load,
    /// Exact registry-declared conversion between two storage
    /// representations. The result is a completely initialized owned value.
    RepresentationConvert(RepresentationTarget),
    /// `decode`: dense `f32` value of a packed view.
    Decode,
    Transpose,
    Reshape,
    /// View selection `t[i, j:k, …]`.
    SliceView {
        indices: Vec<IndexSlot>,
    },
    /// Point read `t[i, j]`.
    ElementRead {
        arity: u32,
    },
    /// `extent(v, axis)`.
    Extent {
        axis: u32,
    },
    /// `atomic(add|max|min, place, value)`. Defined for f32, f16, bf16, i32,
    /// u32; bool is rejected because bool arithmetic is undefined.
    Atomic {
        op: AtomicOp,
        arity: u32,
    },
    Reduce {
        op: ReduceOp,
        axis: u32,
        unordered: bool,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RepresentationTarget {
    Concrete(RepresentationId),
    Parameter(String),
}

impl PrimitiveId {
    pub fn name(&self) -> String {
        match self {
            PrimitiveId::Constant(c) => match c {
                Constant::Int(v) => format!("const.{v}"),
                Constant::Float(v) => format!("const.{v}"),
                Constant::Bool(v) => format!("const.{v}"),
            },
            PrimitiveId::Symbolic(e) => format!("symbolic.{e:?}"),
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
            PrimitiveId::TensorAlloc => "tensor.alloc".into(),
            PrimitiveId::Fill(_) => "tensor.fill".into(),
            PrimitiveId::Materialize => "tensor.materialize".into(),
            PrimitiveId::Clone => "tensor.clone".into(),
            PrimitiveId::Load => "tensor.load".into(),
            PrimitiveId::RepresentationConvert(id) => format!("representation.convert.{id:?}"),
            PrimitiveId::Decode => "tensor.decode".into(),
            PrimitiveId::Transpose => "tensor.transpose".into(),
            PrimitiveId::Reshape => "tensor.reshape".into(),
            PrimitiveId::SliceView { .. } => "tensor.slice".into(),
            PrimitiveId::ElementRead { .. } => "tensor.read".into(),
            PrimitiveId::Extent { .. } => "tensor.extent".into(),
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
// Reduction schema
// ---------------------------------------------------------------------------

/// The accumulator and result dtype of a reduction (registry decision):
/// floating `sum` of f16/bf16/f32 accumulates and results in f32; integer
/// `sum` retains the input dtype and wraps; `max`/`min` retain the input
/// dtype; `argmax` results in i32.
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

/// How ties are resolved. The registry admits exactly one rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TieRule {
    /// The smaller coordinate index wins.
    SmallerCoordinateIndex,
}

/// The algebraic combination law of a reduction operator over its
/// accumulator: what reorderings of the fold are meaning-preserving in exact
/// arithmetic. The numerical transfer of any reassociation under finite
/// precision is the compiler's numerical analysis, not the registry's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CombineLaw {
    /// Regrouping preserves meaning; operand order does not.
    Associative,
    /// Regrouping and reordering both preserve meaning.
    AssociativeCommutative,
    /// Only the ascending reference fold defines the result.
    OrderedOnly,
}

/// The complete reduction schema of one operator over one input dtype.
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

/// Whether a dtype admits `atomic`: f32, f16, bf16, i32, u32. Bool is
/// rejected because bool addition is undefined.
pub fn atomic_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::F32 | DType::F16 | DType::BF16 | DType::I32 | DType::U32
    )
}

// ---------------------------------------------------------------------------
// Primitive signatures (crate-private: consumed by the checker)
// ---------------------------------------------------------------------------

/// Classes of scalar dtypes admitted by a parameter pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum DTypeClass {
    Any,
    Float,
    Int,
    Numeric,
    Bool,
}

impl DTypeClass {
    pub(crate) fn matches(self, d: DType) -> bool {
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
/// monomorphization).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ElemClass {
    Any,
    Dense(DTypeClass),
    Packed,
}

impl ElemClass {
    pub(crate) fn matches_dtype(self, d: DType) -> bool {
        match self {
            ElemClass::Any => true,
            ElemClass::Packed => false,
            ElemClass::Dense(class) => class.matches(d),
        }
    }

    pub(crate) fn matches(self, elem: &Elem) -> bool {
        match (self, elem) {
            (ElemClass::Any, _) => true,
            (ElemClass::Packed, Elem::Repr(_)) => true,
            (ElemClass::Packed, _) => false,
            (ElemClass::Dense(_), Elem::Repr(_)) => false,
            (ElemClass::Dense(class), Elem::Dtype(d)) => class.matches(*d),
            (ElemClass::Dense(DTypeClass::Float | DTypeClass::Any), Elem::Param(_)) => true,
            (ElemClass::Dense(_), Elem::Param(_)) => false,
        }
    }
}

/// Pattern for one parameter of a primitive.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum TypePattern {
    /// A scalar (or index) value of a dtype class.
    ScalarOf(DTypeClass),
    /// A shaped value whose element lies in a class.
    TensorOf(ElemClass),
    /// A scalar or dense elementwise operand; tiles broadcast over scalars.
    Elementwise(ElemClass),
    /// A range value with any bound.
    Range,
    Any,
}

impl TypePattern {
    pub(crate) fn matches(&self, ty: &ValueType) -> bool {
        match self {
            TypePattern::ScalarOf(class) => ty.scalar_dtype().is_some_and(|d| class.matches(d)),
            TypePattern::TensorOf(class) => ty.shaped().is_some_and(|s| class.matches(&s.elem)),
            TypePattern::Elementwise(class) => match ty {
                ValueType::Scalar(d) => class.matches_dtype(*d),
                ValueType::Index { .. } => true,
                ValueType::Tensor(s) => class.matches(&s.elem),
                _ => false,
            },
            TypePattern::Range => matches!(ty, ValueType::Range { .. }),
            TypePattern::Any => true,
        }
    }
}

/// How the dtype of an elementwise result is chosen.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ResultDType {
    SameAs(usize),
    Promoted(Vec<usize>),
    Dtype(DType),
}

/// The result type of a primitive given its operand types. `Structural`
/// marks primitives whose result the checker derives from source structure
/// the operands alone do not carry.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum TypeFunction {
    Fixed(ValueType),
    /// Same type as operand `i`.
    SameAs(usize),
    /// Elementwise over the shape of operand `shape_of` with a derived dtype.
    Elementwise {
        shape_of: usize,
        dtype: ResultDType,
    },
    TupleOfOperands,
    /// Reduction of operand 0 along the primitive's axis.
    Reduction {
        op: ReduceOp,
        axis: u32,
    },
    /// Checker-supplied (allocation, range construction, views, reads).
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
        ValueType::Tensor(s) => s.elem.dense_dtype(),
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

/// The result type of reducing `operand` along `axis` with `op`.
pub(crate) fn reduction_result(operand: &ValueType, op: ReduceOp, axis: u32) -> Option<ValueType> {
    let shaped = operand.shaped()?;
    let input = shaped.elem.dense_dtype()?;
    let dtype = accumulator_dtype(op, input);
    let mut axes = shaped.axes.clone();
    let axis = axis as usize;
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
    pub(crate) fn apply(&self, operands: &[ValueType]) -> Option<ValueType> {
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
                Some(ValueType::Tuple(NonEmpty::new(operands.to_vec())?))
            }
            TypeFunction::Reduction { op, axis } => reduction_result(operands.first()?, *op, *axis),
            TypeFunction::Structural => None,
        }
    }
}

/// One primitive's typing signature.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PrimitiveSignature {
    pub parameters: Vec<TypePattern>,
    pub result: TypeFunction,
}

impl PrimitiveSignature {
    pub(crate) fn result_type(&self, operands: &[ValueType]) -> Option<ValueType> {
        self.result.apply(operands)
    }

    /// Whether the operand types match the declared parameter patterns.
    pub(crate) fn accepts(&self, operands: &[ValueType]) -> bool {
        operands.len() == self.parameters.len()
            && operands
                .iter()
                .zip(&self.parameters)
                .all(|(t, p)| p.matches(t))
    }
}

/// The typing signature of one primitive: the single table consulted by
/// checking.
pub(crate) fn primitive(id: &PrimitiveId) -> PrimitiveSignature {
    let sig = |parameters, result| PrimitiveSignature { parameters, result };
    let bulk_copy = || {
        sig(
            vec![TypePattern::TensorOf(ElemClass::Any)],
            TypeFunction::SameAs(0),
        )
    };
    let ew = |class| TypePattern::Elementwise(ElemClass::Dense(class));
    match id {
        PrimitiveId::Constant(_) | PrimitiveId::Symbolic(_) => {
            sig(Vec::new(), TypeFunction::Structural)
        }
        PrimitiveId::TuplePack => sig(vec![TypePattern::Any; 2], TypeFunction::TupleOfOperands),
        PrimitiveId::TupleGet(_) => sig(vec![TypePattern::Any], TypeFunction::Structural),
        PrimitiveId::RangeMake => sig(
            vec![TypePattern::ScalarOf(DTypeClass::Int); 2],
            TypeFunction::Structural,
        ),
        PrimitiveId::RangeStart | PrimitiveId::RangeEnd => sig(
            vec![TypePattern::Range],
            TypeFunction::Fixed(ValueType::Scalar(DType::I32)),
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
            sig(
                vec![ew(class), ew(if shift { DTypeClass::Int } else { class })],
                result,
            )
        }
        PrimitiveId::Cast(dtype) => sig(
            vec![TypePattern::Elementwise(ElemClass::Any)],
            TypeFunction::Elementwise {
                shape_of: 0,
                dtype: ResultDType::Dtype(*dtype),
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
        ),
        PrimitiveId::TensorAlloc => sig(Vec::new(), TypeFunction::Structural),
        PrimitiveId::Fill(_) => sig(vec![TypePattern::Any], TypeFunction::Structural),
        PrimitiveId::Materialize | PrimitiveId::Clone | PrimitiveId::Load => bulk_copy(),
        PrimitiveId::RepresentationConvert(_) => sig(
            vec![TypePattern::TensorOf(ElemClass::Packed)],
            TypeFunction::Structural,
        ),
        PrimitiveId::Decode => sig(
            vec![TypePattern::TensorOf(ElemClass::Packed)],
            TypeFunction::Elementwise {
                shape_of: 0,
                dtype: ResultDType::Dtype(DType::F32),
            },
        ),
        PrimitiveId::Transpose | PrimitiveId::Reshape => sig(
            vec![TypePattern::TensorOf(ElemClass::Any)],
            TypeFunction::Structural,
        ),
        PrimitiveId::SliceView { .. } | PrimitiveId::ElementRead { .. } => sig(
            vec![TypePattern::TensorOf(ElemClass::Any)],
            TypeFunction::Structural,
        ),
        PrimitiveId::Extent { .. } => sig(
            vec![TypePattern::TensorOf(ElemClass::Any)],
            TypeFunction::Fixed(ValueType::Scalar(DType::I32)),
        ),
        PrimitiveId::Atomic { .. } => sig(Vec::new(), TypeFunction::Fixed(ValueType::Void)),
        PrimitiveId::Reduce { op, axis, .. } => sig(
            vec![TypePattern::TensorOf(ElemClass::Dense(DTypeClass::Any))],
            TypeFunction::Reduction {
                op: *op,
                axis: *axis,
            },
        ),
    }
}

// ---------------------------------------------------------------------------
// Capability intrinsic table (crate-private: interned by `registry`)
// ---------------------------------------------------------------------------

/// Reference meaning of a capability intrinsic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CapabilitySemantics {
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

/// One row of the static capability table, before interning. Representations
/// are dense dtypes here; `registry` maps them to `RepresentationId`.
pub(crate) struct CapabilityRow {
    pub backend: crate::registry::BackendName,
    pub capability: &'static str,
    pub name: &'static str,
    pub arguments: Vec<(&'static str, RowOperand)>,
    pub result: RowResult,
    pub execution: crate::registry::IntrinsicExecution,
    pub participation: crate::registry::IntrinsicParticipation,
    pub result_uniformity: crate::registry::IntrinsicUniformity,
    pub semantics: CapabilitySemantics,
    pub numerics: crate::registry::IntrinsicNumerics,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowOperand {
    Scalar(DType),
    /// A readable dense tensor of the dtype and rank.
    Readable(DType, u32),
    /// A readable tensor in one named packed representation.
    ReadableRepresentation(&'static str, u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowResult {
    Scalar(DType),
    Owned(DType, u32),
}

/// The complete author-visible capability table, in interning order: every
/// backend that exposes capabilities, its namespaces in name order, and each
/// namespace's signatures in declaration order.
pub(crate) fn capability_rows() -> Vec<CapabilityRow> {
    use crate::registry::{
        BackendName, IntrinsicExecution, IntrinsicNumerics, IntrinsicParticipation,
        IntrinsicUniformity,
    };
    let mut out = Vec::new();
    for backend in [BackendName::Metal, BackendName::Cuda] {
        // `matrix` sorts before `subgroup`.
        let matrix_elements: &[DType] = match backend {
            BackendName::Metal => &[DType::F16, DType::F32],
            BackendName::Cuda => &[DType::F16, DType::BF16],
            BackendName::Cpu => &[],
        };
        for &elem in matrix_elements {
            out.push(CapabilityRow {
                backend,
                capability: "matrix",
                name: "matmul",
                arguments: vec![
                    ("left", RowOperand::Readable(elem, 2)),
                    ("right", RowOperand::Readable(elem, 2)),
                ],
                result: RowResult::Owned(DType::F32, 2),
                execution: IntrinsicExecution::WholeTensor { result: 0 },
                participation: IntrinsicParticipation::FullWorkgroup,
                result_uniformity: IntrinsicUniformity::Varying,
                semantics: CapabilitySemantics::MatrixMatmul,
                numerics: IntrinsicNumerics::Reassociated {
                    accumulator: DType::F32,
                },
            });
            out.push(CapabilityRow {
                backend,
                capability: "matrix",
                name: "matmul_add",
                arguments: vec![
                    ("left", RowOperand::Readable(elem, 2)),
                    ("right", RowOperand::Readable(elem, 2)),
                    ("accumulator", RowOperand::Readable(DType::F32, 2)),
                ],
                result: RowResult::Owned(DType::F32, 2),
                execution: IntrinsicExecution::WholeTensor { result: 0 },
                participation: IntrinsicParticipation::FullWorkgroup,
                result_uniformity: IntrinsicUniformity::Varying,
                semantics: CapabilitySemantics::MatrixMatmulAdd,
                numerics: IntrinsicNumerics::Reassociated {
                    accumulator: DType::F32,
                },
            });
        }
        // Ordinary packed matrix rows are generated from the representation
        // table itself. Both native matrix paths stage the registry decode
        // into dense tiles, so every resident f32-decoded integer-code format
        // has the same semantic operation; block-float NVFP4 remains its
        // explicitly scaled CUDA family below.
        for representation in crate::repr::REPRS
            .iter()
            .filter(|representation| representation.float_code.is_none())
        {
            out.push(CapabilityRow {
                backend,
                capability: "matrix",
                name: "matmul",
                arguments: vec![
                    ("left", RowOperand::Readable(DType::F32, 2)),
                    (
                        "right",
                        RowOperand::ReadableRepresentation(representation.name, 2),
                    ),
                ],
                result: RowResult::Owned(DType::F32, 2),
                execution: IntrinsicExecution::WholeTensor { result: 0 },
                participation: IntrinsicParticipation::FullWorkgroup,
                result_uniformity: IntrinsicUniformity::Varying,
                semantics: CapabilitySemantics::MatrixMatmul,
                numerics: IntrinsicNumerics::Reassociated {
                    accumulator: DType::F32,
                },
            });
            out.push(CapabilityRow {
                backend,
                capability: "matrix",
                name: "matmul_add",
                arguments: vec![
                    ("left", RowOperand::Readable(DType::F32, 2)),
                    (
                        "right",
                        RowOperand::ReadableRepresentation(representation.name, 2),
                    ),
                    ("accumulator", RowOperand::Readable(DType::F32, 2)),
                ],
                result: RowResult::Owned(DType::F32, 2),
                execution: IntrinsicExecution::WholeTensor { result: 0 },
                participation: IntrinsicParticipation::FullWorkgroup,
                result_uniformity: IntrinsicUniformity::Varying,
                semantics: CapabilitySemantics::MatrixMatmulAdd,
                numerics: IntrinsicNumerics::Reassociated {
                    accumulator: DType::F32,
                },
            });
        }
        if backend == BackendName::Cuda {
            out.push(CapabilityRow {
                backend,
                capability: "matrix",
                name: "nvfp4_matmul",
                arguments: vec![
                    (
                        "left",
                        RowOperand::ReadableRepresentation("nvfp4_e2m1_block16", 2),
                    ),
                    (
                        "right",
                        RowOperand::ReadableRepresentation("nvfp4_e2m1_block16", 2),
                    ),
                    ("left_global_scale", RowOperand::Scalar(DType::F32)),
                    ("right_global_scale", RowOperand::Scalar(DType::F32)),
                ],
                result: RowResult::Owned(DType::F32, 2),
                execution: IntrinsicExecution::WholeTensor { result: 0 },
                participation: IntrinsicParticipation::FullWorkgroup,
                result_uniformity: IntrinsicUniformity::Varying,
                semantics: CapabilitySemantics::MatrixMatmul,
                numerics: IntrinsicNumerics::Reassociated {
                    accumulator: DType::F32,
                },
            });
            out.push(CapabilityRow {
                backend,
                capability: "matrix",
                name: "nvfp4_matmul_add",
                arguments: vec![
                    (
                        "left",
                        RowOperand::ReadableRepresentation("nvfp4_e2m1_block16", 2),
                    ),
                    (
                        "right",
                        RowOperand::ReadableRepresentation("nvfp4_e2m1_block16", 2),
                    ),
                    ("left_global_scale", RowOperand::Scalar(DType::F32)),
                    ("right_global_scale", RowOperand::Scalar(DType::F32)),
                    ("accumulator", RowOperand::Readable(DType::F32, 2)),
                ],
                result: RowResult::Owned(DType::F32, 2),
                execution: IntrinsicExecution::WholeTensor { result: 0 },
                participation: IntrinsicParticipation::FullWorkgroup,
                result_uniformity: IntrinsicUniformity::Varying,
                semantics: CapabilitySemantics::MatrixMatmulAdd,
                numerics: IntrinsicNumerics::Reassociated {
                    accumulator: DType::F32,
                },
            });
        }
        out.push(CapabilityRow {
            backend,
            capability: "subgroup",
            name: "lane_index",
            arguments: vec![],
            result: RowResult::Scalar(DType::I32),
            execution: IntrinsicExecution::WithinEnclosingParallel,
            participation: IntrinsicParticipation::Independent,
            result_uniformity: IntrinsicUniformity::Varying,
            semantics: CapabilitySemantics::ParticipantIndex,
            numerics: IntrinsicNumerics::Exact,
        });
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            out.push(CapabilityRow {
                backend,
                capability: "subgroup",
                name: "shuffle",
                arguments: vec![
                    ("value", RowOperand::Scalar(dtype)),
                    ("participant", RowOperand::Scalar(DType::I32)),
                ],
                result: RowResult::Scalar(dtype),
                execution: IntrinsicExecution::WithinEnclosingParallel,
                participation: IntrinsicParticipation::FullSubgroup,
                result_uniformity: IntrinsicUniformity::Varying,
                semantics: CapabilitySemantics::Exchange,
                numerics: IntrinsicNumerics::Exact,
            });
            for (name, op) in [
                ("simd_sum", ReduceOp::Sum),
                ("simd_max", ReduceOp::Max),
                ("simd_min", ReduceOp::Min),
            ] {
                out.push(CapabilityRow {
                    backend,
                    capability: "subgroup",
                    name,
                    arguments: vec![("value", RowOperand::Scalar(dtype))],
                    result: RowResult::Scalar(dtype),
                    execution: IntrinsicExecution::WithinEnclosingParallel,
                    participation: IntrinsicParticipation::FullSubgroup,
                    result_uniformity: IntrinsicUniformity::Subgroup,
                    semantics: CapabilitySemantics::SubgroupReduction(op),
                    numerics: match op {
                        ReduceOp::Sum => IntrinsicNumerics::Reassociated { accumulator: dtype },
                        _ => IntrinsicNumerics::Exact,
                    },
                });
            }
        }
    }
    out
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
    fn reduce_schema_agrees_with_accumulator_dtype_and_laws() {
        for input in DType::ALL {
            for op in [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min] {
                let schema = reduce_schema(op, input);
                assert_eq!(schema.result, accumulator_dtype(op, input));
                assert_eq!(schema.accumulator, schema.result);
                assert_eq!(schema.combine, CombineLaw::AssociativeCommutative);
            }
            let argmax = reduce_schema(ReduceOp::Argmax, input);
            assert_eq!(argmax.result, DType::I32);
            assert_eq!(argmax.accumulator, input);
            assert_eq!(argmax.combine, CombineLaw::OrderedOnly);
        }
    }

    #[test]
    fn capability_rows_are_grouped_by_backend_and_namespace() {
        let rows = capability_rows();
        let keys: Vec<(crate::registry::BackendName, &str)> =
            rows.iter().map(|r| (r.backend, r.capability)).collect();
        let mut sorted = keys.clone();
        sorted.dedup();
        assert_eq!(
            keys.iter().collect::<std::collections::BTreeSet<_>>().len(),
            sorted.len()
        );
        assert!(atomic_dtype(DType::F32));
        assert!(!atomic_dtype(DType::Bool));
    }
}
