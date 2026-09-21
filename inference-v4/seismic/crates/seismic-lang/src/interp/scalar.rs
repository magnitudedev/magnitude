//! Scalar operation semantics. Ordinary primitives execute at their dtype;
//! unary reference math evaluates the language-owned ordered primitive recipe.
use super::round_to;
use super::value::Scalar;
use crate::intrinsics::MathOp;
use crate::syntax::ast::{BinaryOp, UnaryOp};
use crate::types::DType;

fn boolean(b: bool) -> Scalar {
    (DType::Bool, u8::from(b) as f64)
}

fn bits(v: f64) -> u32 {
    v as i64 as u32
}

fn integer_value(dtype: DType, bits: u32) -> i64 {
    match dtype {
        DType::I32 => i64::from(bits as i32),
        DType::U32 => i64::from(bits),
        _ => unreachable!("checked integer operation has a non-integer dtype"),
    }
}

pub(super) fn binary(
    op: BinaryOp,
    a: Scalar,
    b: Scalar,
    hint: Option<DType>,
) -> Result<Scalar, String> {
    use BinaryOp::*;
    let comparison = matches!(op, Eq | Ne | Lt | Le | Gt | Ge);
    match (a.0, b.0) {
        (DType::Bool, DType::Bool) => {
            let (x, y) = (a.1 != 0.0, b.1 != 0.0);
            Ok(boolean(match op {
                And | BitAnd => x && y,
                Or | BitOr => x || y,
                Eq => x == y,
                Ne | BitXor => x != y,
                _ => return Err("arithmetic on bools".into()),
            }))
        }
        (x, y) if x.is_int() && y.is_int() => {
            let (p, q) = (a.1 as i64, b.1 as i64);
            let d = if x == y || matches!(op, Shl | Shr) {
                x
            } else {
                hint.filter(|h| h.is_int()).unwrap_or(x)
            };
            let value = |bits: u32| Ok((d, integer_value(d, bits) as f64));
            match op {
                Eq => Ok(boolean(p == q)),
                Ne => Ok(boolean(p != q)),
                Lt => Ok(boolean(p < q)),
                Le => Ok(boolean(p <= q)),
                Gt => Ok(boolean(p > q)),
                Ge => Ok(boolean(p >= q)),
                Add => value((p as u32).wrapping_add(q as u32)),
                Sub => value((p as u32).wrapping_sub(q as u32)),
                Mul => value((p as u32).wrapping_mul(q as u32)),
                Div | Rem => {
                    if q == 0 || (d == DType::I32 && p == i64::from(i32::MIN) && q == -1) {
                        return Err("integer division by zero or signed overflow".into());
                    }
                    let r = if op == Rem {
                        p.checked_rem_euclid(q)
                    } else {
                        p.checked_div_euclid(q)
                    };
                    r.map(|r| (d, r as f64))
                        .ok_or_else(|| "integer division overflow".to_string())
                }
                Shl | Shr => {
                    if !(0..32).contains(&q) {
                        return Err("integer shift count must be in 0..32".into());
                    }
                    value(if op == Shl {
                        (p as u32).wrapping_shl(q as u32)
                    } else if d == DType::I32 {
                        ((p as i32) >> q) as u32
                    } else {
                        (p as u32) >> q
                    })
                }
                BitAnd => value((p as u32) & (q as u32)),
                BitOr => value((p as u32) | (q as u32)),
                BitXor => value((p as u32) ^ (q as u32)),
                And | Or => Err("logic on integers".into()),
            }
        }
        (x, y) if x.is_numeric() && y.is_numeric() => {
            if comparison {
                return Ok(boolean(match op {
                    Eq => a.1 == b.1,
                    Ne => a.1 != b.1,
                    Lt => a.1 < b.1,
                    Le => a.1 <= b.1,
                    Gt => a.1 > b.1,
                    _ => a.1 >= b.1,
                }));
            }
            let d = DType::promote(x, y)
                .or_else(|| hint.filter(|h| h.is_float()))
                .unwrap_or(if x.is_float() { x } else { y });
            let r = match op {
                Add => a.1 + b.1,
                Sub => a.1 - b.1,
                Mul => a.1 * b.1,
                Div => a.1 / b.1,
                Rem => a.1 % b.1,
                other => return Err(format!("`{}` on floats", other.text())),
            };
            Ok((d, round_to(d, r)))
        }
        (x, y) => Err(format!("`{}` on {} and {}", op.text(), x.name(), y.name())),
    }
}

pub(super) fn unary(op: UnaryOp, a: Scalar) -> Result<Scalar, String> {
    match (op, a.0) {
        (UnaryOp::Neg, d) if d.is_float() => Ok((d, -a.1)),
        (UnaryOp::Neg, d) if d.is_int() => {
            Ok((d, integer_value(d, bits(a.1).wrapping_neg()) as f64))
        }
        (UnaryOp::Not, DType::Bool) => Ok(boolean(a.1 == 0.0)),
        (UnaryOp::BitNot, d) if d.is_int() => Ok((d, integer_value(d, !bits(a.1)) as f64)),
        (op, d) => Err(format!("unary {op:?} on {}", d.name())),
    }
}

/// Integer-to-integer casts preserve bits; everything else converts by value with the
/// destination's rounding (saturating for float-to-integer).
pub(super) fn cast(to: DType, a: Scalar) -> Scalar {
    if a.0.is_int() && to.is_int() {
        (to, integer_value(to, bits(a.1)) as f64)
    } else {
        (to, round_to(to, a.1))
    }
}

pub(super) fn math(op: MathOp, args: &[Scalar]) -> Result<Scalar, String> {
    let arity = if op == MathOp::Fma {
        3
    } else if matches!(op, MathOp::Max | MathOp::Min) {
        2
    } else {
        1
    };
    if args.len() != arity {
        return Err(format!("{op:?} takes {arity} arguments"));
    }
    let d = args[1..]
        .iter()
        .fold(args[0].0, |d, a| DType::promote(d, a.0).unwrap_or(d));
    let a = args[0].1;
    Ok(match op {
        MathOp::Fma => {
            let (b, c) = (args[1].1, args[2].1);
            // Narrow-float operands are exact in f32, so this is the single-rounded f32 result.
            if d == DType::F32 {
                (d, (a as f32).mul_add(b as f32, c as f32) as f64)
            } else {
                (d, round_to(d, a.mul_add(b, c)))
            }
        }
        MathOp::Exp | MathOp::Rsqrt | MathOp::Sqrt | MathOp::Log | MathOp::Sin | MathOp::Cos => {
            reference_math(op, d, a)
        }
        // `exp_fast` is explicitly approximate and is never admitted as a
        // reference recipe operation.
        MathOp::ExpFast => (d, round_to(d, a.exp())),
        MathOp::Abs => {
            if d == DType::I32 {
                (d, f64::from((a as i32).wrapping_abs()))
            } else if d.is_int() {
                (d, a)
            } else {
                reference_math(op, d, a)
            }
        }
        MathOp::Max => (d, a.max(args[1].1)),
        MathOp::Min => (d, a.min(args[1].1)),
    })
}

fn reference_math(op: MathOp, dtype: DType, value: f64) -> Scalar {
    use super::tensor::{bf16_round, f16_bits, f16_to_f32};
    use crate::reference_math::{evaluate, recipe, ReferenceMathOp, ReferenceScalar};
    let input = match dtype {
        DType::F16 => ReferenceScalar::F16(f16_bits(value as f32)),
        DType::BF16 => ReferenceScalar::BF16((bf16_round(value as f32).to_bits() >> 16) as u16),
        DType::F32 => ReferenceScalar::F32((value as f32).to_bits()),
        _ => unreachable!("reference transcendental input is not floating"),
    };
    let op = ReferenceMathOp::try_from(op)
        .unwrap_or_else(|()| unreachable!("non-reference operation reached reference evaluator"));
    let output = evaluate(&recipe(op, dtype), input);
    let value = match output {
        ReferenceScalar::F16(bits) => f16_to_f32(bits) as f64,
        ReferenceScalar::BF16(bits) => f32::from_bits(u32::from(bits) << 16) as f64,
        ReferenceScalar::F32(bits) => f32::from_bits(bits) as f64,
        _ => unreachable!("reference transcendental output is not floating"),
    };
    (dtype, value)
}
