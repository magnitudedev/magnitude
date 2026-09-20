//! The portable legalization matrix:
//! the universal physical form of every registry primitive family.
//!
//! The universal column is exact and portable: it legalizes on every backend
//! with `Exact` numerical transfer relative to the registry reference, so a
//! strict caller policy always retains a legal physical alternative. Optional
//! optimized forms (vector SSA/predication, contraction with a numerical
//! transfer, native math with a bound or evidence, coalesced layouts, parallel
//! atomics, reassociating reduction topologies) are backend strategies
//! declared in their own alternatives; they are not part of this matrix.
//!
//! Host `libm`, MSL math, or PTX approximate instructions are not exact
//! unless proved bit-equivalent for the full input domain: the reference
//! transcendental is the one versioned `seismic_math` software sequence below,
//! shared by the interpreter, CPU, Metal, and CUDA exact strategies.

use seismic_lang::{
    intrinsics::{
        primitive, AtomicOp, IntrinsicId, MathOp, PlaneField, PrimitiveId, ReferenceNumerics,
    },
    logical::PrimitiveOp,
    syntax::ast::{BinaryOp, UnaryOp},
    types::{DType, Elem, TensorType, ValueType},
};

// ---------------------------------------------------------------------------
// The versioned transcendental reference
// ---------------------------------------------------------------------------

/// Identity of the one versioned portable transcendental algorithm family.
pub const SEISMIC_MATH_IDENTITY: &str = "seismic_math";

/// Revision of the portable software math sequence. Changing the algorithm
/// changes this version, the registry's `SoftwareMath` algorithm string, and
/// with them every downstream plan, cache, and evidence identity.
pub const SEISMIC_MATH_VERSION: u32 = 1;

/// The versioned reference: interpreter, CPU, Metal, and CUDA exact
/// strategies all execute this sequence, which defines the reference bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SeismicMathReference {
    pub identity: &'static str,
    pub version: u32,
}

impl SeismicMathReference {
    pub const fn current() -> Self {
        SeismicMathReference {
            identity: SEISMIC_MATH_IDENTITY,
            version: SEISMIC_MATH_VERSION,
        }
    }

    /// The algorithm string the registry carries in
    /// `ReferenceNumerics::SoftwareMath`.
    pub fn algorithm(&self) -> String {
        format!("{}-v{}", self.identity, self.version)
    }
}

/// The current versioned `seismic_math` reference.
pub const SEISMIC_MATH: SeismicMathReference = SeismicMathReference {
    identity: SEISMIC_MATH_IDENTITY,
    version: SEISMIC_MATH_VERSION,
};

// ---------------------------------------------------------------------------
// Universal physical forms
// ---------------------------------------------------------------------------

/// The universal physical form of one primitive family (the universal column
/// of the 16.1 matrix). Each variant names the exact portable emission the
/// backend dialects legalize to; none is ever `Inapplicable`.
#[derive(Clone, Debug, PartialEq)]
pub enum UniversalForm {
    /// Constants, tuple leaves, and scalar select as typed SSA; tuples are
    /// recursively leaf-lowered through the canonical traversal.
    TypedSsa,
    /// Integer arithmetic/bitwise/shift as 32-bit wrapping operations, with
    /// planned division/shift checks discharged from the node's obligations.
    IntegerArithmetic,
    /// Float arithmetic/fma at the registry dtype and rounding point.
    FloatArithmetic { dtype: DType },
    /// Transcendental math as the versioned `seismic_math` software sequence
    /// that defines the reference bits.
    SoftwareMath {
        op: MathOp,
        reference: SeismicMathReference,
    },
    /// Casts per the registry conversion: integer-to-integer preserves the
    /// low 32 bits; others convert by value with defined rounding/saturation.
    Cast { source: DType, target: DType },
    /// Reshape/transpose/slice as representation-layout address expressions.
    LayoutAddress { transform: LayoutTransform },
    /// Read/write as a checked address plus typed load/store.
    CheckedAccess { access: DataAccess },
    /// Uninitialized dense storage: declared by the plan, never emitted.
    StorageAllocation,
    /// Fill/copy/materialize/clone as an arbitrary-rank linear loop over
    /// elements (or representation planes).
    LinearElementLoop { op: LinearLoopOp },
    /// Packed decode: dense `f32` elements decoded from representation-plane
    /// addresses of the packed view.
    PackedDecode,
    /// Packed accessor: a readable representation plane (never writable).
    PackedPlaneRead { plane: PlaneField },
    /// Packed element read: one decoded element addressed through the
    /// representation planes of the packed view.
    PackedElementRead,
    /// Atomic update as a serialized exact load/combine/round/store. The
    /// containing independent domain is serialized by the universal strategy;
    /// a parallel form exists only as a safe native/planned CAS alternative.
    SerializedAtomic { op: AtomicOp, dtype: DType },
}

/// Layout transform kinds legalized as address expressions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LayoutTransform {
    Reshape,
    Transpose,
    Slice,
}

/// Element access kind of a checked load/store.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DataAccess {
    Load,
    Store,
}

/// Aggregate linear-loop operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LinearLoopOp {
    Fill,
    Copy,
    Materialize,
    Clone,
    /// `load`: a snapshot in the operand's own representation (plane copy).
    Load,
}

// ---------------------------------------------------------------------------
// Legalization
// ---------------------------------------------------------------------------

/// Result of universal legalization: a form, or the capability route.
#[derive(Clone, Debug, PartialEq)]
pub enum UniversalLegalization {
    Form(UniversalForm),
    /// A capability application has no portable physical opcode: the
    /// alternative containing it is legal only through the exact effective
    /// capability signature on this target, or via the portable reference
    /// body. Universal portable `Inapplicable` for a primitive is a compiler
    /// bug and never occurs here.
    RequiresCapability {
        intrinsic: IntrinsicId,
    },
}

/// A universal legalization failure. Only reducible to a compiler bug: a
/// reduction primitive reaching scalar legalization, or an operand shape the
/// closed registry cannot classify.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LegalizationBug(pub String);

impl std::fmt::Display for LegalizationBug {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "compiler bug in universal legalization: {}", self.0)
    }
}
impl std::error::Error for LegalizationBug {}

/// The elementwise (scalar) dtype one operand contributes, per the registry's
/// portable-scope rule (an unresolved element parameter reads as f32).
fn operand_dtype(ty: &ValueType) -> Option<DType> {
    match ty {
        ValueType::Scalar(d) => Some(*d),
        ValueType::Index { .. } => Some(DType::I32),
        ValueType::Tensor(s) => match &s.elem {
            Elem::Dtype(d) => Some(*d),
            Elem::Param(_) => Some(DType::F32),
            Elem::Repr(_) => None,
        },
        _ => None,
    }
}

fn arithmetic_form(operands: &[ValueType]) -> Result<UniversalForm, LegalizationBug> {
    let dtype = operand_dtype(operands.first().ok_or_else(|| {
        LegalizationBug("an arithmetic primitive has no operand to classify".into())
    })?)
    .ok_or_else(|| LegalizationBug("arithmetic on a packed representation".into()))?;
    Ok(if dtype.is_float() {
        UniversalForm::FloatArithmetic { dtype }
    } else {
        UniversalForm::IntegerArithmetic
    })
}

/// The universal physical form of one primitive application. The match is
/// exhaustive over the closed registry vocabulary; there is no wildcard and
/// no optional-`Inapplicable` path for primitives.
pub fn universal_form(
    op: &PrimitiveOp,
    operands: &[ValueType],
) -> Result<UniversalLegalization, LegalizationBug> {
    match op {
        PrimitiveOp::Constant(_) | PrimitiveOp::RuntimeExtent(_) => {
            Ok(UniversalLegalization::Form(UniversalForm::TypedSsa))
        }
        PrimitiveOp::Capability(intrinsic) => Ok(UniversalLegalization::RequiresCapability {
            intrinsic: intrinsic.clone(),
        }),
        PrimitiveOp::Primitive(id) => match id {
            PrimitiveId::TuplePack
            | PrimitiveId::TupleGet(_)
            | PrimitiveId::RangeMake
            | PrimitiveId::RangeStart
            | PrimitiveId::RangeEnd
            | PrimitiveId::Select
            | PrimitiveId::Extent { .. }
            | PrimitiveId::ValidExtent { .. } => {
                Ok(UniversalLegalization::Form(UniversalForm::TypedSsa))
            }
            PrimitiveId::Unary(unary) => match unary {
                UnaryOp::Not => Ok(UniversalLegalization::Form(UniversalForm::TypedSsa)),
                UnaryOp::BitNot => Ok(UniversalLegalization::Form(
                    UniversalForm::IntegerArithmetic,
                )),
                UnaryOp::Neg => Ok(UniversalLegalization::Form(arithmetic_form(operands)?)),
            },
            PrimitiveId::Binary(binary) => match binary {
                BinaryOp::Or
                | BinaryOp::And
                | BinaryOp::Eq
                | BinaryOp::Ne
                | BinaryOp::Lt
                | BinaryOp::Le
                | BinaryOp::Gt
                | BinaryOp::Ge => Ok(UniversalLegalization::Form(UniversalForm::TypedSsa)),
                BinaryOp::BitOr
                | BinaryOp::BitXor
                | BinaryOp::BitAnd
                | BinaryOp::Shl
                | BinaryOp::Shr => Ok(UniversalLegalization::Form(
                    UniversalForm::IntegerArithmetic,
                )),
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem => {
                    Ok(UniversalLegalization::Form(arithmetic_form(operands)?))
                }
            },
            PrimitiveId::Cast(target) => {
                let source =
                    operand_dtype(operands.first().ok_or_else(|| {
                        LegalizationBug("a cast primitive has no operand".into())
                    })?)
                    .ok_or_else(|| LegalizationBug("cast of a packed representation".into()))?;
                Ok(UniversalLegalization::Form(UniversalForm::Cast {
                    source,
                    target: *target,
                }))
            }
            PrimitiveId::Math(math) => {
                Ok(UniversalLegalization::Form(UniversalForm::SoftwareMath {
                    op: *math,
                    reference: SeismicMathReference::current(),
                }))
            }
            PrimitiveId::TensorAlloc { .. } => Ok(UniversalLegalization::Form(
                UniversalForm::StorageAllocation,
            )),
            PrimitiveId::Fill { .. } => Ok(UniversalLegalization::Form(
                UniversalForm::LinearElementLoop {
                    op: LinearLoopOp::Fill,
                },
            )),
            PrimitiveId::Materialize => Ok(UniversalLegalization::Form(
                UniversalForm::LinearElementLoop {
                    op: LinearLoopOp::Materialize,
                },
            )),
            PrimitiveId::Clone => Ok(UniversalLegalization::Form(
                UniversalForm::LinearElementLoop {
                    op: LinearLoopOp::Clone,
                },
            )),
            PrimitiveId::Load => Ok(UniversalLegalization::Form(
                UniversalForm::LinearElementLoop {
                    op: LinearLoopOp::Load,
                },
            )),
            PrimitiveId::Decode => Ok(UniversalLegalization::Form(UniversalForm::PackedDecode)),
            PrimitiveId::PackedRead(plane) => Ok(UniversalLegalization::Form(
                UniversalForm::PackedPlaneRead { plane: *plane },
            )),
            PrimitiveId::Transpose => {
                Ok(UniversalLegalization::Form(UniversalForm::LayoutAddress {
                    transform: LayoutTransform::Transpose,
                }))
            }
            PrimitiveId::Reshape => Ok(UniversalLegalization::Form(UniversalForm::LayoutAddress {
                transform: LayoutTransform::Reshape,
            })),
            PrimitiveId::SliceView { .. } => {
                Ok(UniversalLegalization::Form(UniversalForm::LayoutAddress {
                    transform: LayoutTransform::Slice,
                }))
            }
            PrimitiveId::ElementRead { .. } => {
                // A point read of a packed view decodes through its
                // representation planes instead of a dense typed load.
                let packed = operands.first().is_some_and(|ty| {
                    matches!(
                        ty,
                        ValueType::Tensor(TensorType {
                            elem: Elem::Repr(_),
                            ..
                        })
                    )
                });
                Ok(UniversalLegalization::Form(if packed {
                    UniversalForm::PackedElementRead
                } else {
                    UniversalForm::CheckedAccess {
                        access: DataAccess::Load,
                    }
                }))
            }
            PrimitiveId::ElementWrite { .. } => {
                Ok(UniversalLegalization::Form(UniversalForm::CheckedAccess {
                    access: DataAccess::Store,
                }))
            }
            PrimitiveId::CopyInto => Ok(UniversalLegalization::Form(
                UniversalForm::LinearElementLoop {
                    op: LinearLoopOp::Copy,
                },
            )),
            PrimitiveId::Atomic { op, .. } => {
                // The combined value is the operand after the place.
                let dtype = operand_dtype(
                    operands
                        .get(1)
                        .ok_or_else(|| LegalizationBug("an atomic update has no value".into()))?,
                )
                .ok_or_else(|| {
                    LegalizationBug("atomic update of a packed representation".into())
                })?;
                Ok(UniversalLegalization::Form(
                    UniversalForm::SerializedAtomic { op: *op, dtype },
                ))
            }
            PrimitiveId::Reduce { op, .. } => Err(LegalizationBug(format!(
                "the `{}` reduction reached scalar legalization; reductions are ReductionNodes \
                 consumed by reduction strategies",
                op.name()
            ))),
        },
    }
}

/// The universal column is exact relative to the registry reference: this
/// function exists so consumers and tests can rely on it structurally.
pub fn universal_numerical(_form: &UniversalForm) -> crate::terminal::numerics::NumericalTransfer {
    crate::terminal::numerics::NumericalTransfer::Exact
}

/// Confirm the registry's math signatures reference the current `seismic_math`
/// version. Called by construction and by tests; a mismatch is a compiler bug.
pub fn registry_math_is_versioned() -> Result<(), LegalizationBug> {
    let current = SeismicMathReference::current().algorithm();
    for op in seismic_lang::intrinsics::math_ops() {
        let signature = primitive(PrimitiveId::Math(op));
        match &signature.numerical {
            ReferenceNumerics::SoftwareMath { algorithm } => {
                if *algorithm != current {
                    return Err(LegalizationBug(format!(
                        "registry math `{}` references algorithm `{algorithm}`, expected `{current}`",
                        op.name()
                    )));
                }
            }
            other => {
                return Err(LegalizationBug(format!(
                    "registry math `{}` is not software math ({other:?})",
                    op.name()
                )));
            }
        }
    }
    Ok(())
}
