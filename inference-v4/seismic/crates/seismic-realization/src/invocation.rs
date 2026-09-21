//! The invocation expression algebra and invocation contract (package A1).
//!
//! Two expression families are separated by type:
//!
//! - `CheckedInvocationExpr`: invocation-known arithmetic (shape fields, ABI
//!   scalars, constants, reflected native facts folded after assembly). Every
//!   node retains its proven `Bound` and every partial node its
//!   `DomainProof`; `InvocationContract::evaluate` (called once by
//!   `CompiledPlan::prepare`, package R1) evaluates the whole contract into
//!   dense validated words.
//! - `GuardedExecutionExpr`: arithmetic over executor-produced scalars. Every
//!   partial operation names the sealed guard (`GuardIx`) that structurally
//!   dominates its use; guard failure is `SafetyViolation`.
//!
//! No expression variant is unchecked. Invocation expressions are built only
//! by the crate-private `ExprBuilder`, which computes every bound by interval
//! arithmetic and either proves a partial operation total (`DomainProof::
//! Interval`) or records the relation that `evaluate` checks before the node
//! is computed (`DomainProof::Predicate`). A proof it cannot establish under
//! the envelope is an `ExprDefect` (a compiler defect of the seal), never a
//! runtime category. Native facts enter only through `ExprBuilder::native_fact`
//! with the declared domain (`NativeFactDeclaration`) as their interval; the
//! native seal (N1) reflects the constant, and `evaluate` folds it through the
//! `native` accessor of the sealed artifact.
//!
//! Soundness of the retained bounds rests on every leaf being inside its
//! interval when `evaluate` runs: constants are exact, shape fields and ABI
//! scalars are validated against their domains here (invocation failures),
//! and reflected native facts are validated against their declared domain by
//! the native seal (a defect there never reaches execution). Under that
//! premise evaluation is total by construction: every proven partial
//! operation is computed through its discharged proof (the `ProvenNonZero`
//! and `ProvenSubtrahend` witnesses), so an evaluated value is never an
//! `Option` and no evaluation path panics. Where a partial operation
//! legitimately has no value — a retained relation that does not hold, or a
//! dominating guard that failed — that outcome is the retained
//! `InvalidInvocation` or `SafetyViolation` return.

use crate::failure::{CompilerDefect, InvalidInvocation, Package, SafetyKind, SafetyViolation};
use crate::ids::{
    BufferSlot, DenseIndex, DenseMap, GuardIx, InvocationValueId, NativeFactIx, ResultFieldIx,
    ScalarSlot, ScalarSlotIx,
};
use seismic_lang::abi::RangeEndpoint;
use seismic_lang::logical::specialization::{ShapeDomain, ShapeFieldId};
use seismic_lang::sir::ParamOwnership;
use seismic_lang::types::{DType, ValuePath};
use std::fmt;

/// A proven finite interval of one expression under the workload envelope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Bound {
    pub min: u64,
    pub max: u64,
}

impl Bound {
    pub fn exact(value: u64) -> Bound {
        Bound {
            min: value,
            max: value,
        }
    }

    pub fn contains(self, value: u64) -> bool {
        self.min <= value && value <= self.max
    }
}

/// Why a partial operation is defined on its whole input domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DomainProof {
    /// The operand intervals prove the operation total (no zero divisor,
    /// no negative difference).
    Interval,
    /// The retained relation is discharged against the evaluated operand
    /// values immediately before the node is computed. The relation
    /// references only operands of the node, so its operands are already
    /// evaluated when the node is reached in topological order.
    Predicate { predicate: InvocationPredicate },
}

/// The integer ABI representations an invocation expression can read. A
/// floating-point or boolean scalar has no arithmetic value in the algebra;
/// the seal decides this exhaustively through `AbiIntegerType::of`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AbiIntegerType {
    I32,
    U32,
}

impl AbiIntegerType {
    pub fn of(dtype: DType) -> Result<AbiIntegerType, ExprDefect> {
        match dtype {
            DType::I32 => Ok(AbiIntegerType::I32),
            DType::U32 => Ok(AbiIntegerType::U32),
            DType::F32 | DType::BF16 | DType::F16 | DType::Bool => {
                Err(ExprDefect::NonIntegerAbiScalar { dtype })
            }
        }
    }

    pub fn dtype(self) -> DType {
        match self {
            AbiIntegerType::I32 => DType::I32,
            AbiIntegerType::U32 => DType::U32,
        }
    }

    /// The non-negative values the representation can hold.
    pub fn representable(self) -> Bound {
        match self {
            AbiIntegerType::I32 => Bound {
                min: 0,
                max: i32::MAX as u64,
            },
            AbiIntegerType::U32 => Bound {
                min: 0,
                max: u64::from(u32::MAX),
            },
        }
    }

    /// The unsigned arithmetic value of one ABI word in this representation.
    /// A negative `I32` word has none.
    fn decode(self, word: ScalarWord) -> Result<u64, NegativeScalar> {
        let low = (word.bits & u64::from(u32::MAX)) as u32;
        match self {
            AbiIntegerType::U32 => Ok(u64::from(low)),
            AbiIntegerType::I32 => {
                let signed = low as i32;
                u64::try_from(signed).map_err(|_| NegativeScalar { value: signed })
            }
        }
    }
}

/// A signed ABI word used where an unsigned invocation quantity is required.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NegativeScalar {
    value: i32,
}

/// The invocation-known expression algebra. Variants are constructed only by
/// the private builder; every node retains its proven `Bound` (leaves retain
/// the domain the bound is read from) and every partial node its
/// `DomainProof`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckedInvocationExpr {
    Const(u64),
    ShapeField {
        field: ShapeFieldId,
        domain: ShapeDomain,
    },
    AbiScalar {
        slot: ScalarSlot,
        repr: AbiIntegerType,
        endpoint: Option<RangeEndpoint>,
        /// The retained static interval of the scalar (at most the
        /// representation range); the invocation is rejected outside it.
        bound: Bound,
    },
    /// A native fact reflected during assembly and folded to a constant; the
    /// index names the launch fact table entry it came from and `domain` is
    /// the declared interval the reflected constant was validated against.
    NativeFact {
        index: NativeFactIx,
        domain: Bound,
    },
    Add {
        left: InvocationValueId,
        right: InvocationValueId,
        bound: Bound,
    },
    Sub {
        left: InvocationValueId,
        right: InvocationValueId,
        bound: Bound,
        proof: DomainProof,
    },
    Mul {
        left: InvocationValueId,
        right: InvocationValueId,
        bound: Bound,
    },
    Div {
        left: InvocationValueId,
        right: InvocationValueId,
        bound: Bound,
        proof: DomainProof,
    },
    Rem {
        left: InvocationValueId,
        right: InvocationValueId,
        bound: Bound,
        proof: DomainProof,
    },
    CeilDiv {
        left: InvocationValueId,
        right: InvocationValueId,
        bound: Bound,
        proof: DomainProof,
    },
    Min {
        left: InvocationValueId,
        right: InvocationValueId,
        bound: Bound,
    },
    Max {
        left: InvocationValueId,
        right: InvocationValueId,
        bound: Bound,
    },
}

impl CheckedInvocationExpr {
    /// The proven finite interval of this expression under capacity. Every
    /// node retains it, so no table walk is needed.
    pub fn bound(&self) -> Bound {
        match self {
            Self::Const(value) => Bound::exact(*value),
            Self::ShapeField { domain, .. } => shape_bound(*domain),
            Self::AbiScalar { bound, .. } => *bound,
            Self::NativeFact { domain, .. } => *domain,
            Self::Add { bound, .. }
            | Self::Sub { bound, .. }
            | Self::Mul { bound, .. }
            | Self::Div { bound, .. }
            | Self::Rem { bound, .. }
            | Self::CeilDiv { bound, .. }
            | Self::Min { bound, .. }
            | Self::Max { bound, .. } => *bound,
        }
    }
}

fn shape_bound(domain: ShapeDomain) -> Bound {
    Bound {
        min: domain.min(),
        max: domain.max(),
    }
}

/// A relational fact the invocation must satisfy before submission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InvocationPredicate {
    NonZero(InvocationValueId),
    Le(InvocationValueId, InvocationValueId),
    Lt(InvocationValueId, InvocationValueId),
    /// `start <= end <= bound` for one range parameter.
    RangeOrdered {
        start: InvocationValueId,
        end: InvocationValueId,
        bound: InvocationValueId,
    },
    /// The product of the factors fits in `bits`.
    ProductFits {
        factors: Vec<InvocationValueId>,
        bits: u8,
    },
}

impl InvocationPredicate {
    /// Whether the predicate holds over already-evaluated derived values.
    fn holds(&self, values: &[u64]) -> bool {
        match self {
            Self::NonZero(value) => values[value.index()] != 0,
            Self::Le(left, right) => values[left.index()] <= values[right.index()],
            Self::Lt(left, right) => values[left.index()] < values[right.index()],
            Self::RangeOrdered { start, end, bound } => {
                let (start, end, bound) = (
                    values[start.index()],
                    values[end.index()],
                    values[bound.index()],
                );
                start <= end && end <= bound
            }
            Self::ProductFits { factors, bits } => {
                match checked_product(factors.iter().map(|factor| values[factor.index()])) {
                    Some(product) => fits_bits(product, *bits),
                    None => false,
                }
            }
        }
    }

    /// The predicate with its evaluated operands, for failure reports.
    fn describe(&self, values: &[u64]) -> String {
        match self {
            Self::NonZero(value) => format!("value {} = 0 must be nonzero", value.0),
            Self::Le(left, right) => format!(
                "value {} = {} must be <= value {} = {}",
                left.0,
                values[left.index()],
                right.0,
                values[right.index()]
            ),
            Self::Lt(left, right) => format!(
                "value {} = {} must be < value {} = {}",
                left.0,
                values[left.index()],
                right.0,
                values[right.index()]
            ),
            Self::RangeOrdered { start, end, bound } => format!(
                "range start {} <= end {} <= bound {} must hold",
                values[start.index()],
                values[end.index()],
                values[bound.index()]
            ),
            Self::ProductFits { factors, bits } => {
                let factors: Vec<u64> = factors.iter().map(|factor| values[factor.index()]).collect();
                format!("product of {factors:?} must fit {bits} bits")
            }
        }
    }
}

fn checked_product(factors: impl Iterator<Item = u64>) -> Option<u64> {
    let mut product: u64 = 1;
    for factor in factors {
        product = product.checked_mul(factor)?;
    }
    Some(product)
}

/// Whether `product` fits unsigned `bits` bits. A product that overflows
/// `u64` never fits (its caller reports `None` as not fitting).
fn fits_bits(product: u64, bits: u8) -> bool {
    bits >= 64 || product < (1u64 << bits)
}

/// A divisor proven nonzero: by its operand interval at seal time
/// (`DomainProof::Interval`), by a discharged retained relation
/// (`DomainProof::Predicate`), or by a dominating guard the executor just
/// reported established. Constructed only on those paths; division,
/// remainder, and ceiling division by it are total (`u64` division never
/// overflows).
#[derive(Clone, Copy, Debug)]
struct ProvenNonZero(u64);

impl ProvenNonZero {
    fn quotient(self, left: u64) -> u64 {
        left / self.0
    }

    fn remainder(self, left: u64) -> u64 {
        left % self.0
    }

    fn ceil_quotient(self, left: u64) -> u64 {
        left.div_ceil(self.0)
    }
}

/// A subtrahend proven not to exceed the minuend it is subtracted from: by
/// the operand intervals at seal time, by a discharged retained relation, or
/// by a dominating guard the executor just reported established. Constructed
/// only on those paths; the wrapped difference of an ordered pair is the
/// exact difference.
#[derive(Clone, Copy, Debug)]
struct ProvenSubtrahend(u64);

impl ProvenSubtrahend {
    fn difference(self, left: u64) -> u64 {
        left.wrapping_sub(self.0)
    }
}

/// The sum of two operands whose retained bounds the seal proved additive
/// (`ExprBuilder::add` rejects an overflowing bound); the wrapped sum of
/// in-bound operands is exact.
fn sealed_sum(left: u64, right: u64) -> u64 {
    left.wrapping_add(right)
}

/// The product of two operands whose retained bounds the seal proved finite
/// (`ExprBuilder::mul` rejects an overflowing bound); the wrapped product of
/// in-bound operands is exact.
fn sealed_product(left: u64, right: u64) -> u64 {
    left.wrapping_mul(right)
}

/// One public buffer of the root ABI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferContract {
    pub slot: BufferSlot,
    pub path: ValuePath,
    /// Representation plane name; `"dense"` for dense tensors.
    pub plane: String,
    pub role: BufferRole,
    pub dtype: DType,
    /// Byte requirement as an expression of actual validated extents.
    pub bytes: InvocationValueId,
    pub alignment: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferRole {
    Parameter {
        ordinal: u32,
        ownership: ParamOwnership,
    },
    Result,
}

/// One by-value scalar of the root ABI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalarContract {
    pub slot: ScalarSlot,
    pub name: String,
    pub dtype: DType,
    pub domain: ScalarDomain,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScalarDomain {
    Any,
    /// `0 <= value < bound`, where `value` is the derived node reading this
    /// scalar (`CheckedInvocationExpr::AbiScalar`) and `bound` an
    /// invocation-known derived value.
    Index {
        value: InvocationValueId,
        bound: InvocationValueId,
    },
    /// One endpoint of a range parameter; the pair is validated by the
    /// `RangeOrdered` predicate.
    RangeEndpoint { endpoint: RangeEndpoint },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AliasRule {
    MayOverlap,
    MustDisjoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AliasContract {
    pub left: BufferSlot,
    pub right: BufferSlot,
    pub rule: AliasRule,
}

/// The complete invocation contract emitted by the physical seal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvocationContract {
    shape_fields: DenseMap<ShapeFieldId, ShapeFieldContract>,
    buffers: Vec<BufferContract>,
    /// Slot order: `scalars[i].slot.index() == i`.
    scalars: Vec<ScalarContract>,
    aliases: Vec<AliasContract>,
    relations: Vec<InvocationPredicate>,
    /// Topologically ordered: every operand precedes its user.
    derived: DenseMap<InvocationValueId, CheckedInvocationExpr>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapeFieldContract {
    pub name: String,
    pub domain: ShapeDomain,
}

impl DenseIndex for ShapeFieldId {
    fn from_index(index: usize) -> Self {
        ShapeFieldId(u32::try_from(index).expect("shape field index fits u32"))
    }
    fn index(self) -> usize {
        self.0 as usize
    }
}

impl DenseIndex for InvocationValueId {
    fn from_index(index: usize) -> Self {
        InvocationValueId(u32::try_from(index).expect("invocation value index fits u32"))
    }
    fn index(self) -> usize {
        self.0 as usize
    }
}

impl InvocationContract {
    /// The only constructor; private to the physical seal (P1), which
    /// establishes: every buffer/scalar slot is dense, unique, and in slot
    /// order; every `InvocationValueId` is defined before use; `derived` and
    /// `relations` come from one `ExprBuilder::finish` so every partial
    /// operation carries its proof; every shape field has one domain equal to
    /// the domain retained by its `ShapeField` node.
    pub(crate) fn new(
        shape_fields: DenseMap<ShapeFieldId, ShapeFieldContract>,
        buffers: Vec<BufferContract>,
        scalars: Vec<ScalarContract>,
        aliases: Vec<AliasContract>,
        relations: Vec<InvocationPredicate>,
        derived: DenseMap<InvocationValueId, CheckedInvocationExpr>,
    ) -> InvocationContract {
        InvocationContract {
            shape_fields,
            buffers,
            scalars,
            aliases,
            relations,
            derived,
        }
    }

    pub fn shape_fields(&self) -> &DenseMap<ShapeFieldId, ShapeFieldContract> {
        &self.shape_fields
    }
    pub fn buffers(&self) -> &[BufferContract] {
        &self.buffers
    }
    pub fn scalars(&self) -> &[ScalarContract] {
        &self.scalars
    }
    pub fn aliases(&self) -> &[AliasContract] {
        &self.aliases
    }
    pub fn relations(&self) -> &[InvocationPredicate] {
        &self.relations
    }
    pub fn derived(&self) -> &DenseMap<InvocationValueId, CheckedInvocationExpr> {
        &self.derived
    }

    /// Evaluate the contract once for one invocation. `CompiledPlan::prepare`
    /// (R1) is the caller; it supplies each bound scalar word and shape value
    /// by contract entry (reporting `MissingScalar`/`MissingShape`/
    /// `ScalarRepresentation` itself) and the reflected native facts of the
    /// sealed artifact.
    ///
    /// Order: every shape field against its domain; every derived value in
    /// topological order, validating each ABI scalar leaf against its
    /// retained interval and discharging each `DomainProof::Predicate`
    /// before its node (`ArithmeticDomain`); every `ScalarDomain::Index`
    /// (`IndexOutOfBounds`); every relation (`Predicate`). Buffer and alias
    /// validation is R1's, using the returned `derived` byte values.
    pub fn evaluate(
        &self,
        scalar: &mut dyn FnMut(&ScalarContract) -> Result<ScalarWord, InvalidInvocation>,
        shape: &mut dyn FnMut(ShapeFieldId, &ShapeFieldContract) -> Result<u64, InvalidInvocation>,
        native: &dyn Fn(NativeFactIx) -> u64,
    ) -> Result<InvocationValues, InvalidInvocation> {
        let mut shapes: Vec<u64> = Vec::with_capacity(self.shape_fields.len());
        for (field, contract) in self.shape_fields.iter() {
            let value = shape(field, contract)?;
            if !contract.domain.contains(value) {
                return Err(InvalidInvocation::ShapeOutsideDomain {
                    field,
                    value,
                    domain: contract.domain,
                });
            }
            shapes.push(value);
        }

        let mut words: Vec<ScalarWord> = Vec::with_capacity(self.scalars.len());
        for contract in &self.scalars {
            words.push(scalar(contract)?);
        }

        let mut values: Vec<u64> = Vec::with_capacity(self.derived.len());
        for (id, expr) in self.derived.iter() {
            let value = match expr {
                CheckedInvocationExpr::Const(value) => *value,
                CheckedInvocationExpr::ShapeField { field, domain } => {
                    let value = shapes[field.index()];
                    if !domain.contains(value) {
                        return Err(InvalidInvocation::ShapeOutsideDomain {
                            field: *field,
                            value,
                            domain: *domain,
                        });
                    }
                    value
                }
                CheckedInvocationExpr::AbiScalar {
                    slot, repr, bound, ..
                } => {
                    let value = repr.decode(words[slot.index()]).map_err(|negative| {
                        InvalidInvocation::ScalarRepresentation {
                            slot: *slot,
                            reason: format!(
                                "{} is negative where an unsigned quantity is required",
                                negative.value
                            ),
                        }
                    })?;
                    if !bound.contains(value) {
                        return Err(InvalidInvocation::ScalarOutsideDomain {
                            slot: *slot,
                            value,
                            min: bound.min,
                            max: bound.max,
                        });
                    }
                    value
                }
                CheckedInvocationExpr::NativeFact { index, .. } => native(*index),
                CheckedInvocationExpr::Add { left, right, .. } => {
                    sealed_sum(values[left.index()], values[right.index()])
                }
                CheckedInvocationExpr::Sub {
                    left,
                    right,
                    proof,
                    ..
                } => {
                    let (minuend, subtrahend) = (values[left.index()], values[right.index()]);
                    self.ordered_subtrahend(proof, id, subtrahend, &values)?.difference(minuend)
                }
                CheckedInvocationExpr::Mul { left, right, .. } => {
                    sealed_product(values[left.index()], values[right.index()])
                }
                CheckedInvocationExpr::Div {
                    left,
                    right,
                    proof,
                    ..
                } => {
                    let (dividend, divisor) = (values[left.index()], values[right.index()]);
                    self.nonzero_divisor(proof, id, divisor, &values)?.quotient(dividend)
                }
                CheckedInvocationExpr::Rem {
                    left,
                    right,
                    proof,
                    ..
                } => {
                    let (dividend, divisor) = (values[left.index()], values[right.index()]);
                    self.nonzero_divisor(proof, id, divisor, &values)?.remainder(dividend)
                }
                CheckedInvocationExpr::CeilDiv {
                    left,
                    right,
                    proof,
                    ..
                } => {
                    let (dividend, divisor) = (values[left.index()], values[right.index()]);
                    self.nonzero_divisor(proof, id, divisor, &values)?.ceil_quotient(dividend)
                }
                CheckedInvocationExpr::Min { left, right, .. } => {
                    values[left.index()].min(values[right.index()])
                }
                CheckedInvocationExpr::Max { left, right, .. } => {
                    values[left.index()].max(values[right.index()])
                }
            };
            values.push(value);
        }

        for contract in &self.scalars {
            match &contract.domain {
                ScalarDomain::Any | ScalarDomain::RangeEndpoint { .. } => {}
                ScalarDomain::Index { value, bound } => {
                    let (value, bound) = (values[value.index()], values[bound.index()]);
                    if value >= bound {
                        return Err(InvalidInvocation::IndexOutOfBounds {
                            slot: contract.slot,
                            value,
                            bound,
                        });
                    }
                }
            }
        }

        for relation in &self.relations {
            if !relation.holds(&values) {
                return Err(InvalidInvocation::Predicate {
                    predicate: relation.clone(),
                    description: relation.describe(&values),
                });
            }
        }

        Ok(InvocationValues {
            scalars: DenseMap::from_vec(words),
            shapes: DenseMap::from_vec(shapes),
            derived: DenseMap::from_vec(values),
        })
    }

    /// Discharge the proof of one subtraction, yielding the subtrahend as a
    /// value proven not to exceed the minuend. An `Interval` proof was
    /// established at seal time and needs nothing at evaluation; a
    /// `Predicate` proof is the relation the invocation must satisfy for the
    /// difference to be defined, checked here against the values evaluated
    /// so far.
    fn ordered_subtrahend(
        &self,
        proof: &DomainProof,
        node: InvocationValueId,
        subtrahend: u64,
        values: &[u64],
    ) -> Result<ProvenSubtrahend, InvalidInvocation> {
        match proof {
            DomainProof::Interval => Ok(ProvenSubtrahend(subtrahend)),
            DomainProof::Predicate { predicate } => {
                if predicate.holds(values) {
                    Ok(ProvenSubtrahend(subtrahend))
                } else {
                    Err(InvalidInvocation::ArithmeticDomain {
                        value: node,
                        description: predicate.describe(values),
                    })
                }
            }
        }
    }

    /// Discharge the proof of one division, remainder, or ceiling division,
    /// yielding the divisor as a value proven nonzero. An `Interval` proof
    /// was established at seal time; a `Predicate` proof is the relation the
    /// invocation must satisfy for the quotient to be defined, checked here
    /// against the values evaluated so far.
    fn nonzero_divisor(
        &self,
        proof: &DomainProof,
        node: InvocationValueId,
        divisor: u64,
        values: &[u64],
    ) -> Result<ProvenNonZero, InvalidInvocation> {
        match proof {
            DomainProof::Interval => Ok(ProvenNonZero(divisor)),
            DomainProof::Predicate { predicate } => {
                if predicate.holds(values) {
                    Ok(ProvenNonZero(divisor))
                } else {
                    Err(InvalidInvocation::ArithmeticDomain {
                        value: node,
                        description: predicate.describe(values),
                    })
                }
            }
        }
    }
}

/// The validated invocation values: the output of `CompiledPlan::prepare`
/// (package R1) before it is paired with the native artifact. Every derived
/// value has been evaluated exactly once with every predicate checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvocationValues {
    pub scalars: DenseMap<ScalarSlot, ScalarWord>,
    pub shapes: DenseMap<ShapeFieldId, u64>,
    pub derived: DenseMap<InvocationValueId, u64>,
}

/// One validated by-value scalar, encoded in its ABI representation: `bits`
/// holds the dtype's bit pattern zero-extended to 64 bits (`U32` value,
/// `I32` two's complement, IEEE bits for the floats, `0`/`1` for `Bool`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScalarWord {
    pub dtype: DType,
    pub bits: u64,
}

impl ScalarWord {
    pub fn unsigned(value: u32) -> ScalarWord {
        ScalarWord {
            dtype: DType::U32,
            bits: u64::from(value),
        }
    }

    pub fn signed(value: i32) -> ScalarWord {
        ScalarWord {
            dtype: DType::I32,
            bits: u64::from(value as u32),
        }
    }
}

/// Arithmetic over executor-produced scalars and result-block fields. Every
/// partial operation names the sealed guard that dominates every use of its
/// result; `evaluate` consults that guard before computing the operation.
/// `Add`/`Mul` retain the interval the seal proved from the operand bounds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GuardedExecutionExpr {
    Invocation(InvocationValueId),
    /// One executor scalar slot with the interval the seal established for
    /// it: the slot's 32-bit representation range, or the tighter domain a
    /// guard that structurally dominates this expression proves.
    Executor { slot: ScalarSlotIx, bound: Bound },
    /// One field of the compiler-owned result block, read as an operand. The
    /// seal established `bound` for the field and sealed the guard that
    /// structurally dominates this read; `evaluate` consults that guard
    /// before the value is used.
    ResultField {
        field: ResultFieldIx,
        bound: Bound,
        guard: GuardIx,
    },
    Add(Box<GuardedExecutionExpr>, Box<GuardedExecutionExpr>, Bound),
    Sub(Box<GuardedExecutionExpr>, Box<GuardedExecutionExpr>, GuardIx),
    Mul(Box<GuardedExecutionExpr>, Box<GuardedExecutionExpr>, Bound),
    Div(Box<GuardedExecutionExpr>, Box<GuardedExecutionExpr>, GuardIx),
    Rem(Box<GuardedExecutionExpr>, Box<GuardedExecutionExpr>, GuardIx),
    CeilDiv(Box<GuardedExecutionExpr>, Box<GuardedExecutionExpr>, GuardIx),
    Min(Box<GuardedExecutionExpr>, Box<GuardedExecutionExpr>),
}

impl GuardedExecutionExpr {
    /// The finite interval of this expression: retained on `Add`/`Mul` and
    /// the leaves, derived by interval arithmetic over the guarded domain
    /// elsewhere.
    pub fn bound(&self, contract: &InvocationContract) -> Bound {
        match self {
            Self::Invocation(id) => contract.derived()[*id].bound(),
            Self::Executor { bound, .. } => *bound,
            Self::ResultField { bound, .. } => *bound,
            Self::Add(_, _, bound) | Self::Mul(_, _, bound) => *bound,
            Self::Sub(left, right, _) => difference_bounds(left.bound(contract), right.bound(contract)),
            Self::Div(left, right, _) => quotient_bounds(left.bound(contract), right.bound(contract)),
            Self::Rem(left, right, _) => remainder_bounds(left.bound(contract), right.bound(contract)),
            Self::CeilDiv(left, right, _) => {
                ceil_quotient_bounds(left.bound(contract), right.bound(contract))
            }
            Self::Min(left, right) => min_bounds(left.bound(contract), right.bound(contract)),
        }
    }

    /// Evaluate against validated invocation values, executor scalars, and
    /// result-block fields. Every partial operation first asks `guards`
    /// whether its dominating guard is established (the executor answers
    /// from the guard's status); a failed guard is the `SafetyViolation`
    /// returned, and an established guard yields the operand as a proven
    /// value through which the operation is total. A result-field read
    /// consults its dominating guard the same way before `results` yields
    /// the field's word. `Add`/`Mul` retain the interval the seal proved
    /// from the operand bounds, under which the operation is total.
    pub fn evaluate(
        &self,
        invocation: &InvocationValues,
        executor: &dyn Fn(ScalarSlotIx) -> u64,
        results: &dyn Fn(ResultFieldIx) -> u64,
        guards: &mut dyn FnMut(GuardIx, SafetyKind) -> Result<(), SafetyViolation>,
    ) -> Result<u64, SafetyViolation> {
        Ok(match self {
            Self::Invocation(id) => invocation.derived[*id],
            Self::Executor { slot, .. } => executor(*slot),
            Self::ResultField { field, guard, .. } => {
                guards(*guard, SafetyKind::ResultFieldUsable)?;
                results(*field)
            }
            Self::Add(left, right, _) => {
                let left = left.evaluate(invocation, executor, results, guards)?;
                let right = right.evaluate(invocation, executor, results, guards)?;
                sealed_sum(left, right)
            }
            Self::Sub(left, right, guard) => {
                let minuend = left.evaluate(invocation, executor, results, guards)?;
                let subtrahend = right.evaluate(invocation, executor, results, guards)?;
                guards(*guard, SafetyKind::RangeInBounds)?;
                ProvenSubtrahend(subtrahend).difference(minuend)
            }
            Self::Mul(left, right, _) => {
                let left = left.evaluate(invocation, executor, results, guards)?;
                let right = right.evaluate(invocation, executor, results, guards)?;
                sealed_product(left, right)
            }
            Self::Div(left, right, guard) => {
                let dividend = left.evaluate(invocation, executor, results, guards)?;
                let divisor = right.evaluate(invocation, executor, results, guards)?;
                guards(*guard, SafetyKind::DivisorNonZero)?;
                ProvenNonZero(divisor).quotient(dividend)
            }
            Self::Rem(left, right, guard) => {
                let dividend = left.evaluate(invocation, executor, results, guards)?;
                let divisor = right.evaluate(invocation, executor, results, guards)?;
                guards(*guard, SafetyKind::DivisorNonZero)?;
                ProvenNonZero(divisor).remainder(dividend)
            }
            Self::CeilDiv(left, right, guard) => {
                let dividend = left.evaluate(invocation, executor, results, guards)?;
                let divisor = right.evaluate(invocation, executor, results, guards)?;
                guards(*guard, SafetyKind::DivisorNonZero)?;
                ProvenNonZero(divisor).ceil_quotient(dividend)
            }
            Self::Min(left, right) => {
                let left = left.evaluate(invocation, executor, results, guards)?;
                let right = right.evaluate(invocation, executor, results, guards)?;
                left.min(right)
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Interval arithmetic
// ---------------------------------------------------------------------------

fn add_bounds(left: Bound, right: Bound) -> Option<Bound> {
    Some(Bound {
        min: left.min.checked_add(right.min)?,
        max: left.max.checked_add(right.max)?,
    })
}

fn mul_bounds(left: Bound, right: Bound) -> Option<Bound> {
    Some(Bound {
        min: left.min.checked_mul(right.min)?,
        max: left.max.checked_mul(right.max)?,
    })
}

/// Interval of `left - right` over the domain `left >= right`. An empty
/// domain (`left.max < right.min`) yields `[0, 0]`; the builder rejects that
/// case before calling.
fn difference_bounds(left: Bound, right: Bound) -> Bound {
    Bound {
        min: if left.min >= right.max {
            left.min - right.max
        } else {
            0
        },
        max: if left.max >= right.min {
            left.max - right.min
        } else {
            0
        },
    }
}

/// The divisor interval over the domain `right >= 1`, or `None` when the
/// domain is empty (`right.max == 0`).
fn divisor_interval(right: Bound) -> Option<Bound> {
    if right.max == 0 {
        None
    } else {
        Some(Bound {
            min: right.min.max(1),
            max: right.max,
        })
    }
}

fn quotient_bounds(left: Bound, right: Bound) -> Bound {
    match divisor_interval(right) {
        Some(divisor) => Bound {
            min: left.min / divisor.max,
            max: left.max / divisor.min,
        },
        None => Bound { min: 0, max: 0 },
    }
}

fn remainder_bounds(left: Bound, right: Bound) -> Bound {
    match divisor_interval(right) {
        Some(divisor) => Bound {
            min: 0,
            max: left.max.min(divisor.max - 1),
        },
        None => Bound { min: 0, max: 0 },
    }
}

fn ceil_quotient_bounds(left: Bound, right: Bound) -> Bound {
    match divisor_interval(right) {
        Some(divisor) => Bound {
            min: left.min.div_ceil(divisor.max),
            max: left.max.div_ceil(divisor.min),
        },
        None => Bound { min: 0, max: 0 },
    }
}

fn min_bounds(left: Bound, right: Bound) -> Bound {
    Bound {
        min: left.min.min(right.min),
        max: left.max.min(right.max),
    }
}

fn max_bounds(left: Bound, right: Bound) -> Bound {
    Bound {
        min: left.min.max(right.min),
        max: left.max.max(right.max),
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// A construction-time diagnostic of the expression builder: the seal could
/// not prove an expression finite under capacity, or proved a partial
/// operation undefined on the whole envelope. Reported as a compiler defect
/// by the caller; never a runtime category.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExprDefect {
    /// `left + right` exceeds `u64::MAX` at capacity.
    AddOverflow {
        left: InvocationValueId,
        right: InvocationValueId,
    },
    /// `left * right` exceeds `u64::MAX` at capacity.
    MulOverflow {
        left: InvocationValueId,
        right: InvocationValueId,
    },
    /// `left >= right` holds nowhere in the envelope.
    EmptyDifference {
        left: InvocationValueId,
        right: InvocationValueId,
    },
    /// The divisor is zero on the whole envelope.
    ZeroDivisor { right: InvocationValueId },
    /// `start <= end <= bound` holds nowhere in the envelope.
    EmptyRange {
        start: InvocationValueId,
        end: InvocationValueId,
        bound: InvocationValueId,
    },
    /// The product does not fit `bits` even at the envelope minimum.
    ProductNeverFits {
        factors: Vec<InvocationValueId>,
        bits: u8,
    },
    /// A leaf interval with `min > max`.
    EmptyBound { min: u64, max: u64 },
    /// An ABI scalar interval wider than its representation.
    BoundExceedsRepresentation {
        slot: ScalarSlot,
        bound: Bound,
        repr: AbiIntegerType,
    },
    /// An ABI scalar of a non-integer dtype used in invocation arithmetic.
    NonIntegerAbiScalar { dtype: DType },
    /// The derived-value table of one contract exceeds `u32` identity
    /// capacity.
    ValueIdExhausted { count: usize },
    /// The relation table of one contract exceeds `u32` positions.
    RelationIndexExhausted { count: usize },
}

impl fmt::Display for ExprDefect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AddOverflow { left, right } => write!(
                f,
                "invocation value {} + value {} exceeds u64 at capacity",
                left.0, right.0
            ),
            Self::MulOverflow { left, right } => write!(
                f,
                "invocation value {} * value {} exceeds u64 at capacity",
                left.0, right.0
            ),
            Self::EmptyDifference { left, right } => write!(
                f,
                "invocation value {} >= value {} holds nowhere in the envelope",
                left.0, right.0
            ),
            Self::ZeroDivisor { right } => {
                write!(f, "invocation value {} is zero on the whole envelope", right.0)
            }
            Self::EmptyRange { start, end, bound } => write!(
                f,
                "range start value {} <= end value {} <= bound value {} holds nowhere in the envelope",
                start.0, end.0, bound.0
            ),
            Self::ProductNeverFits { factors, bits } => {
                let factors: Vec<u32> = factors.iter().map(|factor| factor.0).collect();
                write!(
                    f,
                    "product of invocation values {factors:?} does not fit {bits} bits at the envelope minimum"
                )
            }
            Self::EmptyBound { min, max } => write!(f, "leaf interval [{min}, {max}] is empty"),
            Self::BoundExceedsRepresentation { slot, bound, repr } => write!(
                f,
                "scalar slot {} interval [{}, {}] exceeds its {:?} representation",
                slot.index(),
                bound.min,
                bound.max,
                repr
            ),
            Self::NonIntegerAbiScalar { dtype } => write!(
                f,
                "an ABI scalar of dtype {} has no value in invocation arithmetic",
                dtype.name()
            ),
            Self::ValueIdExhausted { count } => write!(
                f,
                "the invocation contract defines {count} derived values, exceeding u32 identity capacity"
            ),
            Self::RelationIndexExhausted { count } => write!(
                f,
                "the invocation contract defines {count} relations, exceeding u32 position capacity"
            ),
        }
    }
}

impl From<ExprDefect> for CompilerDefect {
    fn from(defect: ExprDefect) -> Self {
        CompilerDefect::new(Package::A1, defect.to_string())
    }
}

/// The private expression builder (A1): the only constructor of
/// `CheckedInvocationExpr` nodes and of proof relations. Every method returns
/// a node whose bound is computed from its operands by interval arithmetic,
/// with its proof, or an `ExprDefect`.
pub(crate) struct ExprBuilder {
    derived: Vec<CheckedInvocationExpr>,
    relations: Vec<InvocationPredicate>,
}

impl ExprBuilder {
    pub(crate) fn new() -> ExprBuilder {
        ExprBuilder {
            derived: Vec::new(),
            relations: Vec::new(),
        }
    }

    fn push(&mut self, expr: CheckedInvocationExpr) -> Result<InvocationValueId, ExprDefect> {
        let count = self.derived.len();
        let id = InvocationValueId(
            u32::try_from(count).map_err(|_| ExprDefect::ValueIdExhausted { count })?,
        );
        self.derived.push(expr);
        Ok(id)
    }

    fn record(&mut self, predicate: InvocationPredicate) -> Result<u32, ExprDefect> {
        let count = self.relations.len();
        let index =
            u32::try_from(count).map_err(|_| ExprDefect::RelationIndexExhausted { count })?;
        self.relations.push(predicate);
        Ok(index)
    }

    pub(crate) fn constant(&mut self, value: u64) -> Result<InvocationValueId, ExprDefect> {
        self.push(CheckedInvocationExpr::Const(value))
    }

    pub(crate) fn shape_field(
        &mut self,
        field: ShapeFieldId,
        domain: ShapeDomain,
    ) -> Result<InvocationValueId, ExprDefect> {
        self.push(CheckedInvocationExpr::ShapeField { field, domain })
    }

    /// An ABI scalar with its retained static interval, at most the
    /// representation range of `repr`.
    pub(crate) fn abi_scalar(
        &mut self,
        slot: ScalarSlot,
        repr: AbiIntegerType,
        endpoint: Option<RangeEndpoint>,
        bound: Bound,
    ) -> Result<InvocationValueId, ExprDefect> {
        if bound.min > bound.max {
            return Err(ExprDefect::EmptyBound {
                min: bound.min,
                max: bound.max,
            });
        }
        if bound.max > repr.representable().max {
            return Err(ExprDefect::BoundExceedsRepresentation { slot, bound, repr });
        }
        self.push(CheckedInvocationExpr::AbiScalar {
            slot,
            repr,
            endpoint,
            bound,
        })
    }

    /// A native fact with its declared domain. Before assembly the fact is
    /// this interval; the native seal reflects the constant inside it.
    pub(crate) fn native_fact(
        &mut self,
        index: NativeFactIx,
        domain: Bound,
    ) -> Result<InvocationValueId, ExprDefect> {
        if domain.min > domain.max {
            return Err(ExprDefect::EmptyBound {
                min: domain.min,
                max: domain.max,
            });
        }
        self.push(CheckedInvocationExpr::NativeFact { index, domain })
    }

    pub(crate) fn add(
        &mut self,
        left: InvocationValueId,
        right: InvocationValueId,
    ) -> Result<InvocationValueId, ExprDefect> {
        let bound = add_bounds(self.bound(left), self.bound(right))
            .ok_or(ExprDefect::AddOverflow { left, right })?;
        self.push(CheckedInvocationExpr::Add { left, right, bound })
    }

    /// `left - right`: total when `left.min >= right.max`; otherwise proven
    /// by the retained relation `right <= left`, discharged immediately
    /// before the node is computed.
    pub(crate) fn sub(
        &mut self,
        left: InvocationValueId,
        right: InvocationValueId,
    ) -> Result<InvocationValueId, ExprDefect> {
        let (lb, rb) = (self.bound(left), self.bound(right));
        if lb.max < rb.min {
            return Err(ExprDefect::EmptyDifference { left, right });
        }
        let proof = if lb.min >= rb.max {
            DomainProof::Interval
        } else {
            DomainProof::Predicate {
                predicate: InvocationPredicate::Le(right, left),
            }
        };
        let bound = difference_bounds(lb, rb);
        self.push(CheckedInvocationExpr::Sub {
            left,
            right,
            bound,
            proof,
        })
    }

    pub(crate) fn mul(
        &mut self,
        left: InvocationValueId,
        right: InvocationValueId,
    ) -> Result<InvocationValueId, ExprDefect> {
        let bound = mul_bounds(self.bound(left), self.bound(right))
            .ok_or(ExprDefect::MulOverflow { left, right })?;
        self.push(CheckedInvocationExpr::Mul { left, right, bound })
    }

    /// The proof that `right >= 1`: total when its interval excludes zero;
    /// otherwise the retained relation `NonZero(right)`, discharged
    /// immediately before the node is computed; a divisor that is zero
    /// everywhere is a defect.
    fn divisor_proof(&mut self, right: InvocationValueId) -> Result<DomainProof, ExprDefect> {
        let rb = self.bound(right);
        if rb.max == 0 {
            return Err(ExprDefect::ZeroDivisor { right });
        }
        Ok(if rb.min >= 1 {
            DomainProof::Interval
        } else {
            DomainProof::Predicate {
                predicate: InvocationPredicate::NonZero(right),
            }
        })
    }

    pub(crate) fn div(
        &mut self,
        left: InvocationValueId,
        right: InvocationValueId,
    ) -> Result<InvocationValueId, ExprDefect> {
        let proof = self.divisor_proof(right)?;
        let bound = quotient_bounds(self.bound(left), self.bound(right));
        self.push(CheckedInvocationExpr::Div {
            left,
            right,
            bound,
            proof,
        })
    }

    pub(crate) fn rem(
        &mut self,
        left: InvocationValueId,
        right: InvocationValueId,
    ) -> Result<InvocationValueId, ExprDefect> {
        let proof = self.divisor_proof(right)?;
        let bound = remainder_bounds(self.bound(left), self.bound(right));
        self.push(CheckedInvocationExpr::Rem {
            left,
            right,
            bound,
            proof,
        })
    }

    /// `ceil(left / right)`; `u64::div_ceil` cannot overflow, so only the
    /// divisor domain is partial.
    pub(crate) fn ceil_div(
        &mut self,
        left: InvocationValueId,
        right: InvocationValueId,
    ) -> Result<InvocationValueId, ExprDefect> {
        let proof = self.divisor_proof(right)?;
        let bound = ceil_quotient_bounds(self.bound(left), self.bound(right));
        self.push(CheckedInvocationExpr::CeilDiv {
            left,
            right,
            bound,
            proof,
        })
    }

    pub(crate) fn min(
        &mut self,
        left: InvocationValueId,
        right: InvocationValueId,
    ) -> Result<InvocationValueId, ExprDefect> {
        let bound = min_bounds(self.bound(left), self.bound(right));
        self.push(CheckedInvocationExpr::Min { left, right, bound })
    }

    pub(crate) fn max(
        &mut self,
        left: InvocationValueId,
        right: InvocationValueId,
    ) -> Result<InvocationValueId, ExprDefect> {
        let bound = max_bounds(self.bound(left), self.bound(right));
        self.push(CheckedInvocationExpr::Max { left, right, bound })
    }

    /// Record one relation the invocation must satisfy; returns its index.
    pub(crate) fn relation(&mut self, predicate: InvocationPredicate) -> Result<u32, ExprDefect> {
        self.record(predicate)
    }

    /// Record `start <= end <= bound` for one range parameter; a range that
    /// can never be ordered under the envelope is a defect.
    pub(crate) fn range_ordered(
        &mut self,
        start: InvocationValueId,
        end: InvocationValueId,
        bound: InvocationValueId,
    ) -> Result<u32, ExprDefect> {
        let (sb, eb, bb) = (self.bound(start), self.bound(end), self.bound(bound));
        if eb.max < sb.min || bb.max < eb.min {
            return Err(ExprDefect::EmptyRange { start, end, bound });
        }
        self.record(InvocationPredicate::RangeOrdered { start, end, bound })
    }

    /// Record that the product of `factors` fits `bits`; a product that does
    /// not fit even at the envelope minimum is a defect.
    pub(crate) fn product_fits(
        &mut self,
        factors: Vec<InvocationValueId>,
        bits: u8,
    ) -> Result<u32, ExprDefect> {
        let minimum = checked_product(factors.iter().map(|factor| self.bound(*factor).min));
        let fits = match minimum {
            Some(product) => fits_bits(product, bits),
            None => false,
        };
        if !fits {
            return Err(ExprDefect::ProductNeverFits { factors, bits });
        }
        self.record(InvocationPredicate::ProductFits { factors, bits })
    }

    pub(crate) fn bound(&self, value: InvocationValueId) -> Bound {
        self.derived[value.index()].bound()
    }

    pub(crate) fn finish(
        self,
    ) -> (
        DenseMap<InvocationValueId, CheckedInvocationExpr>,
        Vec<InvocationPredicate>,
    ) {
        (DenseMap::from_vec(self.derived), self.relations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::failure::SafetyViolationSource;
    use crate::ids::{ObligationRef, OccurrenceId, OwnedGraphKey, OwnedNodeRef};
    use seismic_lang::logical::{NodeId, NodeRef};

    fn bounded(min: u64, max: u64) -> ShapeDomain {
        ShapeDomain::Bounded { min, max }
    }

    #[test]
    fn subtraction_is_total_by_interval_or_proven_by_retained_relation() {
        let mut builder = ExprBuilder::new();
        let n = builder.shape_field(ShapeFieldId(0), bounded(8, 16)).unwrap();
        let k = builder.shape_field(ShapeFieldId(1), bounded(0, 4)).unwrap();
        let m = builder.shape_field(ShapeFieldId(2), bounded(4, 12)).unwrap();
        let total = builder.sub(n, k).unwrap();
        let relational = builder.sub(m, n).unwrap();
        let (derived, relations) = builder.finish();

        assert!(matches!(
            derived[total],
            CheckedInvocationExpr::Sub {
                proof: DomainProof::Interval,
                bound: Bound { min: 4, max: 16 },
                ..
            }
        ));
        assert!(matches!(
            derived[relational],
            CheckedInvocationExpr::Sub {
                proof: DomainProof::Predicate {
                    predicate: InvocationPredicate::Le(left, right)
                },
                bound: Bound { min: 0, max: 4 },
                ..
            } if left == n && right == m
        ));
        assert!(relations.is_empty());
    }

    #[test]
    fn subtraction_with_an_empty_domain_is_a_defect() {
        let mut builder = ExprBuilder::new();
        let small = builder.constant(3).unwrap();
        let large = builder.shape_field(ShapeFieldId(0), bounded(4, 8)).unwrap();
        assert_eq!(
            builder.sub(small, large),
            Err(ExprDefect::EmptyDifference {
                left: small,
                right: large
            })
        );
    }

    #[test]
    fn division_needs_a_nonzero_relation_unless_the_interval_excludes_zero() {
        let mut builder = ExprBuilder::new();
        let n = builder.shape_field(ShapeFieldId(0), bounded(0, 64)).unwrap();
        let maybe_zero = builder.shape_field(ShapeFieldId(1), bounded(0, 8)).unwrap();
        let positive = builder.shape_field(ShapeFieldId(2), bounded(1, 8)).unwrap();
        let relational = builder.div(n, maybe_zero).unwrap();
        let total = builder.ceil_div(n, positive).unwrap();
        let remainder = builder.rem(n, positive).unwrap();
        let zero = builder.constant(0).unwrap();
        assert_eq!(
            builder.div(n, zero),
            Err(ExprDefect::ZeroDivisor { right: zero })
        );
        let (derived, relations) = builder.finish();

        assert!(matches!(
            derived[relational],
            CheckedInvocationExpr::Div {
                proof: DomainProof::Predicate {
                    predicate: InvocationPredicate::NonZero(divisor)
                },
                bound: Bound { min: 0, max: 64 },
                ..
            } if divisor == maybe_zero
        ));
        assert!(matches!(
            derived[total],
            CheckedInvocationExpr::CeilDiv {
                proof: DomainProof::Interval,
                bound: Bound { min: 0, max: 64 },
                ..
            }
        ));
        assert!(matches!(
            derived[remainder],
            CheckedInvocationExpr::Rem {
                proof: DomainProof::Interval,
                bound: Bound { min: 0, max: 7 },
                ..
            }
        ));
        assert!(relations.is_empty());
    }

    #[test]
    fn overflow_at_capacity_is_a_defect_not_a_runtime_category() {
        let mut builder = ExprBuilder::new();
        let huge = builder.constant(u64::MAX).unwrap();
        let two = builder.constant(2).unwrap();
        assert_eq!(
            builder.add(huge, two),
            Err(ExprDefect::AddOverflow {
                left: huge,
                right: two
            })
        );
        assert_eq!(
            builder.mul(huge, two),
            Err(ExprDefect::MulOverflow {
                left: huge,
                right: two
            })
        );
        let defect: CompilerDefect = ExprDefect::MulOverflow {
            left: huge,
            right: two,
        }
        .into();
        assert_eq!(defect.package, Package::A1);
    }

    /// A contract with shape `n` in `[0, 16]`, index scalar `k < n`, derived
    /// `n - k` (retained relation), `n / 2`, a native fact, and `n * n < 256`
    /// (contract relation 0).
    fn contract() -> (InvocationContract, InvocationValueId, InvocationValueId) {
        let mut builder = ExprBuilder::new();
        let n = builder.shape_field(ShapeFieldId(0), bounded(0, 16)).unwrap();
        let k = builder
            .abi_scalar(
                ScalarSlot::from_index(0),
                AbiIntegerType::I32,
                None,
                Bound { min: 0, max: 16 },
            )
            .unwrap();
        let difference = builder.sub(n, k).unwrap();
        let two = builder.constant(2).unwrap();
        let half = builder.div(n, two).unwrap();
        let width = builder
            .native_fact(NativeFactIx::from_index(0), Bound { min: 32, max: 1024 })
            .unwrap();
        let scaled = builder.mul(half, width).unwrap();
        builder.product_fits(vec![n, n], 8).unwrap();
        let (derived, relations) = builder.finish();

        let contract = InvocationContract::new(
            DenseMap::from_vec(vec![ShapeFieldContract {
                name: "n".to_string(),
                domain: bounded(0, 16),
            }]),
            Vec::new(),
            vec![ScalarContract {
                slot: ScalarSlot::from_index(0),
                name: "k".to_string(),
                dtype: DType::I32,
                domain: ScalarDomain::Index { value: k, bound: n },
            }],
            Vec::new(),
            relations,
            derived,
        );
        (contract, difference, scaled)
    }

    fn evaluate(contract: &InvocationContract, n: u64, k: i32) -> Result<InvocationValues, InvalidInvocation> {
        contract.evaluate(
            &mut |_| Ok(ScalarWord::signed(k)),
            &mut |_, _| Ok(n),
            &|_| 64,
        )
    }

    #[test]
    fn evaluate_accepts_a_valid_bounded_invocation_and_folds_native_facts() {
        let (contract, difference, scaled) = contract();
        let values = evaluate(&contract, 4, 1).unwrap();
        assert_eq!(values.derived[difference], 3);
        assert_eq!(values.derived[scaled], 128);
        assert_eq!(values.shapes[ShapeFieldId(0)], 4);
        assert_eq!(values.scalars[ScalarSlot::from_index(0)], ScalarWord::signed(1));
    }

    #[test]
    fn evaluate_rejects_each_invalid_invocation_with_its_exact_variant() {
        let (contract, difference, _) = contract();

        assert_eq!(
            evaluate(&contract, 17, 0),
            Err(InvalidInvocation::ShapeOutsideDomain {
                field: ShapeFieldId(0),
                value: 17,
                domain: bounded(0, 16),
            })
        );
        assert!(matches!(
            evaluate(&contract, 2, 4),
            Err(InvalidInvocation::ArithmeticDomain { value, .. }) if value == difference
        ));
        assert_eq!(
            evaluate(&contract, 4, 4),
            Err(InvalidInvocation::IndexOutOfBounds {
                slot: ScalarSlot::from_index(0),
                value: 4,
                bound: 4,
            })
        );
        assert!(matches!(
            evaluate(&contract, 16, 0),
            Err(InvalidInvocation::Predicate {
                predicate: InvocationPredicate::ProductFits { .. },
                ..
            })
        ));
        assert!(matches!(
            evaluate(&contract, 4, -1),
            Err(InvalidInvocation::ScalarRepresentation { .. })
        ));
    }

    fn guard_violation(guard: GuardIx, kind: SafetyKind) -> SafetyViolation {
        SafetyViolation {
            source: SafetyViolationSource::Guard(guard),
            obligation: ObligationRef {
                node: OwnedNodeRef {
                    graph: OwnedGraphKey {
                        occurrence: OccurrenceId(0),
                        logical_alternative: 0,
                    },
                    node: NodeRef {
                        region: Vec::new(),
                        node: NodeId(0),
                    },
                },
                index: 0,
            },
            kind,
        }
    }

    #[test]
    fn guarded_evaluation_consults_the_guard_before_dividing() {
        let (contract, _, _) = contract();
        let values = evaluate(&contract, 4, 1).unwrap();
        let representation = Bound {
            min: 0,
            max: u64::from(u32::MAX),
        };
        let guard = GuardIx::from_index(3);
        let quotient = GuardedExecutionExpr::Div(
            Box::new(GuardedExecutionExpr::Executor {
                slot: ScalarSlotIx::from_index(0),
                bound: representation,
            }),
            Box::new(GuardedExecutionExpr::Executor {
                slot: ScalarSlotIx::from_index(1),
                bound: representation,
            }),
            guard,
        );
        let executor = |slot: ScalarSlotIx| -> u64 { if slot.index() == 0 { 10 } else { 0 } };
        let results = |_: ResultFieldIx| -> u64 { 0 };

        let mut consulted = Vec::new();
        let failed = quotient.evaluate(&values, &executor, &results, &mut |guard, kind| {
            consulted.push((guard, kind));
            Err(guard_violation(guard, kind))
        });
        assert_eq!(failed, Err(guard_violation(guard, SafetyKind::DivisorNonZero)));
        assert_eq!(consulted, vec![(guard, SafetyKind::DivisorNonZero)]);

        let executor = |slot: ScalarSlotIx| -> u64 { if slot.index() == 0 { 10 } else { 5 } };
        let passed = quotient.evaluate(&values, &executor, &results, &mut |_, _| Ok(()));
        assert_eq!(passed, Ok(2));
        assert_eq!(quotient.bound(&contract), Bound { min: 0, max: u64::from(u32::MAX) });
    }

    #[test]
    fn a_result_field_read_consults_its_dominating_guard_before_yielding_the_word() {
        let (contract, _, _) = contract();
        let values = evaluate(&contract, 4, 1).unwrap();
        let guard = GuardIx::from_index(1);
        let field = ResultFieldIx::from_index(2);
        let read = GuardedExecutionExpr::ResultField {
            field,
            bound: Bound { min: 0, max: 64 },
            guard,
        };
        let executor = |_: ScalarSlotIx| -> u64 { 0 };
        let results = |requested: ResultFieldIx| -> u64 {
            if requested == field { 48 } else { u64::MAX }
        };

        let mut consulted = Vec::new();
        let failed = read.evaluate(&values, &executor, &results, &mut |guard, kind| {
            consulted.push((guard, kind));
            Err(guard_violation(guard, kind))
        });
        assert_eq!(
            failed,
            Err(guard_violation(guard, SafetyKind::ResultFieldUsable))
        );
        assert_eq!(consulted, vec![(guard, SafetyKind::ResultFieldUsable)]);

        let passed = read.evaluate(&values, &executor, &results, &mut |_, _| Ok(()));
        assert_eq!(passed, Ok(48));
        assert_eq!(read.bound(&contract), Bound { min: 0, max: 64 });
    }
}
