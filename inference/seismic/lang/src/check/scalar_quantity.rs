//! Exact source scalar meaning. Construction never performs bound search.
//! Opaque producers are occurrence-owned by the caller; supported operations
//! retain their shared recipe even when no arithmetic view can be derived.
use crate::expr::{ErasedScalarExpr, ExprArena, IntExpr, ScalarOperand};
use crate::intrinsics::{MathOp, PrimitiveId};
use crate::reference_math::{self, ReferenceScalar};
use crate::syntax::ast::{BinaryOp, UnaryOp};
use crate::types::{DType, ValueType};

pub(super) struct Operand<'a> {
    pub ty: &'a ValueType,
    pub value: Option<IntExpr>,
    pub scalar: Option<ErasedScalarExpr>,
    pub literal: Option<ReferenceScalar>,
}

fn quantity(ty: &ValueType) -> bool {
    matches!(ty, ValueType::Integer | ValueType::Index { .. })
}

/// Exact meaning over reached operands, independent of the current constraints.
/// Recipe failure remains in the node and in the source operation's check site.
pub(super) fn exact(
    arena: &mut ExprArena,
    id: &PrimitiveId,
    result: &ValueType,
    operands: &[Operand<'_>],
) -> Option<IntExpr> {
    if let PrimitiveId::Constant(value) = id {
        return match value {
            ReferenceScalar::I32(value) => Some(arena.int(i64::from(*value))),
            ReferenceScalar::U32(value) => Some(arena.int(i64::from(*value))),
            _ => None,
        };
    }
    if let ValueType::Scalar(dtype @ (DType::I32 | DType::U32)) = result {
        if matches!(id, PrimitiveId::Copy) && operands.len() == 1 && operands[0].ty == result {
            return operands[0].value;
        }
        if matches!(id, PrimitiveId::Cast(_)) && operands.len() == 1 && quantity(operands[0].ty) {
            // A mathematical quantity is not an already-converted word operand.
            // Expose its conversion explicitly, without an overflow-prone shift.
            return Some(arena.int_to_word(operands[0].value?, *dtype));
        }
        let operation = reference_math::scalar_operation(id)?;
        let inputs = operands
            .iter()
            .map(|operand| match operand.ty {
                ValueType::Scalar(dtype @ (DType::I32 | DType::U32)) => Some(ScalarOperand::Word {
                    dtype: *dtype,
                    value: operand.value?,
                }),
                ValueType::Scalar(_) => {
                    let scalar = match operand.scalar {
                        Some(scalar) => scalar,
                        None => arena.scalar_const_erased(operand.literal?),
                    };
                    Some(ScalarOperand::Scalar(scalar))
                }
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        return Some(arena.scalar_integer(operation, &inputs));
    }
    if !quantity(result) {
        return None;
    }
    let values = operands
        .iter()
        .map(|operand| operand.value)
        .collect::<Option<Vec<_>>>()?;
    match (id, values.as_slice()) {
        (PrimitiveId::Math(MathOp::Min), &[a, b]) => Some(arena.int_min(a, b)),
        (PrimitiveId::Math(MathOp::Max), &[a, b]) => Some(arena.int_max(a, b)),
        (PrimitiveId::Binary(BinaryOp::Add), &[a, b]) => Some(arena.int_add(a, b)),
        (PrimitiveId::Binary(BinaryOp::Sub), &[a, b]) => Some(arena.int_sub(a, b)),
        (PrimitiveId::Binary(BinaryOp::Mul), &[a, b]) => Some(arena.int_mul(a, b)),
        (PrimitiveId::Binary(BinaryOp::Div), &[a, b]) => Some(arena.int_div(a, b)),
        (PrimitiveId::Binary(BinaryOp::Rem), &[a, b]) => Some(arena.int_rem(a, b)),
        (PrimitiveId::Unary(UnaryOp::Neg), &[a]) => {
            let zero = arena.int(0);
            Some(arena.int_sub(zero, a))
        }
        (PrimitiveId::Copy, &[a]) if operands[0].ty == result => Some(a),
        _ => None,
    }
}
