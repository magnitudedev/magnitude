//! Interpreter consumption of language-owned scalar recipes. Values retain their
//! typed payload bits; this module only performs the checked source promotions.
use crate::intrinsics::MathOp;
use crate::reference_math::{
    ReferenceScalar, ScalarFailure, ScalarOp, evaluate as evaluate_recipe, scalar_recipe,
};
use crate::syntax::ast::{BinaryOp, UnaryOp};
use crate::types::DType;

fn evaluate(op: ScalarOp, args: &[ReferenceScalar]) -> Result<ReferenceScalar, ScalarFailure> {
    if let Some(value) = host_f32(op, args) {
        return Ok(value);
    }
    let types: Vec<_> = args.iter().map(|v| v.dtype()).collect();
    evaluate_recipe(&scalar_recipe(op, &types), args)
}

/// `op` of `args` as the host's IEEE binary32 arithmetic gives it, where that
/// is the recipe's result to the bit: correctly rounded F32 addition,
/// subtraction, multiplication, division and fused multiply-add, negation as
/// the sign bit, and conversions to F32 that are exact or round once to
/// nearest even, each of a result that is not a NaN (the recipe's NaN is its
/// canonical one); `None` otherwise.
fn host_f32(op: ScalarOp, args: &[ReferenceScalar]) -> Option<ReferenceScalar> {
    use crate::reference_math::conversion::exact_f64;
    let float = |value: &ReferenceScalar| match value {
        ReferenceScalar::F32(bits) => Some(f32::from_bits(*bits)),
        _ => None,
    };
    let value = match (op, args) {
        (ScalarOp::Unary(UnaryOp::Neg), [ReferenceScalar::F32(bits)]) => {
            return Some(ReferenceScalar::F32(bits ^ 0x8000_0000));
        }
        (ScalarOp::Binary(op), [a, b]) => {
            let (a, b) = (float(a)?, float(b)?);
            match op {
                BinaryOp::Add => a + b,
                BinaryOp::Sub => a - b,
                BinaryOp::Mul => a * b,
                BinaryOp::Div => a / b,
                _ => return None,
            }
        }
        (ScalarOp::Math(MathOp::Fma), [a, b, c]) => float(a)?.mul_add(float(b)?, float(c)?),
        (ScalarOp::Cast(DType::F32), [value]) => match *value {
            ReferenceScalar::F32(_) => return Some(*value),
            ReferenceScalar::F16(bits) => exact_f64(DType::F16, u32::from(bits)) as f32,
            ReferenceScalar::BF16(bits) => exact_f64(DType::BF16, u32::from(bits)) as f32,
            ReferenceScalar::I32(code) => code as f32,
            ReferenceScalar::U32(code) => code as f32,
            ReferenceScalar::U64(_) | ReferenceScalar::Bool(_) => return None,
        },
        _ => return None,
    };
    (!value.is_nan()).then(|| ReferenceScalar::F32(value.to_bits()))
}
pub(super) fn binary(
    op: BinaryOp,
    a: ReferenceScalar,
    b: ReferenceScalar,
    hint: Option<DType>,
) -> Result<ReferenceScalar, ScalarFailure> {
    let dtype = DType::promote(a.dtype(), b.dtype())
        .or_else(|| hint.filter(|d| d.is_numeric()))
        .unwrap_or(a.dtype());
    let a = cast(dtype, a);
    let b = if matches!(op, BinaryOp::Shl | BinaryOp::Shr) {
        b
    } else {
        cast(dtype, b)
    };
    evaluate(ScalarOp::Binary(op), &[a, b])
}
pub(super) fn unary(op: UnaryOp, a: ReferenceScalar) -> Result<ReferenceScalar, ScalarFailure> {
    evaluate(ScalarOp::Unary(op), &[a])
}
pub(super) fn cast(to: DType, a: ReferenceScalar) -> ReferenceScalar {
    evaluate(ScalarOp::Cast(to), &[a]).expect("numeric conversion has no source failure")
}
pub(super) fn math(op: MathOp, args: &[ReferenceScalar]) -> Result<ReferenceScalar, ScalarFailure> {
    assert_eq!(
        args.len(),
        op.arity(),
        "checked scalar recipe arity differs"
    );
    let dtype = args[1..].iter().fold(args[0].dtype(), |d, a| {
        DType::promote(d, a.dtype()).unwrap_or(d)
    });
    let args: Vec<_> = args.iter().map(|&a| cast(dtype, a)).collect();
    evaluate(ScalarOp::Math(op), &args)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The recipe's result of `op`, without the host's.
    fn recipe(op: ScalarOp, args: &[ReferenceScalar]) -> ReferenceScalar {
        let types: Vec<_> = args.iter().map(|v| v.dtype()).collect();
        evaluate_recipe(&scalar_recipe(op, &types), args).unwrap()
    }

    /// The host's F32 arithmetic is the recipes' to the bit wherever it
    /// answers: random words of every class (subnormal, normal, infinite,
    /// zero, NaN), every F16 and BF16 payload and integers about F32's
    /// precision.
    #[test]
    fn host_f32_arithmetic_is_the_recipes() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut word = || {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            let bits = (state >> 32) as u32;
            // A third subnormal or zero, a third about one, a third anything.
            match state % 3 {
                0 => bits & 0x807f_ffff,
                1 => (bits & 0x83ff_ffff) | 0x3c00_0000,
                _ => bits,
            }
        };
        let check = |op: ScalarOp, args: &[ReferenceScalar]| {
            if let Some(host) = host_f32(op, args) {
                assert_eq!(host, recipe(op, args), "{op:?} {args:?}");
            }
        };
        for _ in 0..3000 {
            let [a, b, c] = [word(), word(), word()].map(ReferenceScalar::F32);
            for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
                check(ScalarOp::Binary(op), &[a, b]);
            }
            check(ScalarOp::Math(MathOp::Fma), &[a, b, c]);
            check(ScalarOp::Unary(UnaryOp::Neg), &[a]);
            let integer = word() >> (word() % 32);
            check(ScalarOp::Cast(DType::F32), &[ReferenceScalar::U32(integer)]);
            check(ScalarOp::Cast(DType::F32), &[ReferenceScalar::I32(integer as i32)]);
        }
        for bits in 0..=u16::MAX {
            check(ScalarOp::Cast(DType::F32), &[ReferenceScalar::F16(bits)]);
            check(ScalarOp::Cast(DType::F32), &[ReferenceScalar::BF16(bits)]);
        }
    }
}
