//! Scalar operation semantics: every operation is performed at its dtype and rounded once.
use super::value::S;
use super::round_to;
use crate::numeric::{integer_division_is_defined, integer_shift_is_defined, integer_value};
use crate::sir::Math;
use crate::syntax::ast::{AssignOp, BinaryOp, UnaryOp};
use crate::types::DType;

fn boolean(b: bool) -> S {
    (DType::Bool, u8::from(b) as f64)
}

fn bits(v: f64) -> u32 {
    v as i64 as u32
}

pub(super) fn binary(op: BinaryOp, a: S, b: S, hint: Option<DType>) -> Result<S, String> {
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
            let d = if x == y || matches!(op, Shl | Shr) { x } else { hint.filter(|h| h.is_int()).unwrap_or(x) };
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
                    if !integer_division_is_defined(d, Some(p), Some(q)) {
                        return Err("integer division by zero or signed overflow".into());
                    }
                    let r = if op == Rem { p.checked_rem_euclid(q) } else { p.checked_div_euclid(q) };
                    r.map(|r| (d, r as f64)).ok_or_else(|| "integer division overflow".to_string())
                }
                Shl | Shr => {
                    if !integer_shift_is_defined(Some(q)) {
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

pub(super) fn unary(op: UnaryOp, a: S) -> Result<S, String> {
    match (op, a.0) {
        (UnaryOp::Neg, d) if d.is_float() => Ok((d, -a.1)),
        (UnaryOp::Neg, d) if d.is_int() => Ok((d, integer_value(d, bits(a.1).wrapping_neg()) as f64)),
        (UnaryOp::Not, DType::Bool) => Ok(boolean(a.1 == 0.0)),
        (UnaryOp::BitNot, d) if d.is_int() => Ok((d, integer_value(d, !bits(a.1)) as f64)),
        (op, d) => Err(format!("unary {op:?} on {}", d.name())),
    }
}

/// Integer-to-integer casts preserve bits; everything else converts by value with the
/// destination's rounding (saturating for float-to-integer).
pub(super) fn cast(to: DType, a: S) -> S {
    if a.0.is_int() && to.is_int() {
        (to, integer_value(to, bits(a.1)) as f64)
    } else {
        (to, round_to(to, a.1))
    }
}

pub(super) fn assign(op: AssignOp, current: S, value: S) -> Result<S, String> {
    let op = match op {
        AssignOp::Assign => return Ok(cast(current.0, value)),
        AssignOp::Add => BinaryOp::Add,
        AssignOp::Sub => BinaryOp::Sub,
        AssignOp::Mul => BinaryOp::Mul,
    };
    let (d, x) = (current.0, current.1);
    if d.is_int() {
        return Ok(cast(d, binary(op, (d, x), cast(d, value), Some(d))?));
    }
    let r = match op {
        BinaryOp::Add => x + value.1,
        BinaryOp::Sub => x - value.1,
        _ => x * value.1,
    };
    Ok((d, round_to(d, r)))
}

pub(super) fn math(op: Math, args: &[S]) -> Result<S, String> {
    let arity = if op == Math::Fma { 3 } else if matches!(op, Math::Max | Math::Min) { 2 } else { 1 };
    if args.len() != arity {
        return Err(format!("{op:?} takes {arity} arguments"));
    }
    let d = args[1..].iter().fold(args[0].0, |d, a| DType::promote(d, a.0).unwrap_or(d));
    let a = args[0].1;
    Ok(match op {
        Math::Fma => {
            let (b, c) = (args[1].1, args[2].1);
            // Narrow-float operands are exact in f32, so this is the single-rounded f32 result.
            if d == DType::F32 { (d, (a as f32).mul_add(b as f32, c as f32) as f64) } else { (d, round_to(d, a.mul_add(b, c))) }
        }
        Math::Exp | Math::ExpFast => (d, round_to(d, a.exp())),
        Math::Rsqrt => (d, round_to(d, 1.0 / a.sqrt())),
        Math::Sqrt => (d, round_to(d, a.sqrt())),
        Math::Log => (d, round_to(d, a.ln())),
        Math::Sin => (d, round_to(d, a.sin())),
        Math::Cos => (d, round_to(d, a.cos())),
        Math::Abs => {
            if d == DType::I32 {
                (d, f64::from((a as i32).wrapping_abs()))
            } else if d.is_int() {
                (d, a)
            } else {
                (d, round_to(d, a.abs()))
            }
        }
        Math::Max => (d, a.max(args[1].1)),
        Math::Min => (d, a.min(args[1].1)),
    })
}
