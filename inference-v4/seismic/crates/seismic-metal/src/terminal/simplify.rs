//! Local realization of typed scalar operations. These rewrites happen before
//! both rendering and resource derivation; no native compiler facts are used.
use super::{Expression as E, Statement as S, Type as T};
use crate::support::Helper;
use std::collections::{HashMap, HashSet};
pub(super) type Facts = HashMap<String, (i128, i128)>;
use seismic_lang::ast::{BinaryOp as B, UnaryOp as U};

/// An integer enclosure is also evidence that evaluating the expression cannot
/// fail or access memory. Unknown integer variables are pure, but helpers,
/// reads, floating conversions and potentially overflowing arithmetic are not
/// admitted merely because their declared result type is bounded.
pub(super) fn bounds_in(e: &E, facts: &Facts) -> Option<(i128, i128)> {
    let bounds = |e: &E| bounds_in(e, facts);
    let range = match e {
        E::Integer(n, ty) => {
            let n = match ty {
                T::U32 => i128::from(*n as u32),
                T::U64 => i128::from(*n as u64),
                T::Bool => i128::from(*n != 0),
                _ => i128::from(*n),
            };
            (n, n)
        }
        E::Variable(name, ty) | E::Parameter { name, ty } => {
            facts.get(name).copied().or_else(|| limits(*ty))?
        }
        E::Cast(ty, value) => {
            let (lo, hi) = bounds(value)?;
            let (min, max) = limits(*ty)?;
            if lo < min || hi > max {
                return None;
            }
            (lo, hi)
        }
        E::Bitcast(ty, value) if ty.bytes() == value.ty().bytes() => {
            let (lo, hi) = bounds(value)?;
            let (min, max) = limits(*ty)?;
            if lo < min || hi > max {
                return None;
            }
            (lo, hi)
        }
        E::Select(condition, yes, no) => {
            let (lo, hi) = bounds(condition)?;
            if hi == 0 { bounds(no)? }
            else if lo > 0 { bounds(yes)? }
            else {
                // Within one pure scalar expression the predicate and arm
                // observe the same values, even if a surrounding loop later
                // mutates a scalar. Each arm must independently be total.
                let mut yes_facts = facts.clone();
                let mut no_facts = facts.clone();
                assume(condition, true, &mut yes_facts, &HashSet::new());
                assume(condition, false, &mut no_facts, &HashSet::new());
                let (yl, yh) = bounds_in(yes, &yes_facts)?;
                let (nl, nh) = bounds_in(no, &no_facts)?;
                (yl.min(nl), yh.max(nh))
            }
        }
        E::ShortCircuit { or, left, right } => {
            let (lo, hi) = bounds(left)?;
            if *or && lo > 0 { (1, 1) }
            else if !*or && hi == 0 { (0, 0) }
            else {
                let mut right_facts = facts.clone();
                assume(left, !*or, &mut right_facts, &HashSet::new());
                let (rl, rh) = bounds_in(right, &right_facts)?;
                if (*or && hi == 0) || (!*or && lo > 0) { (rl, rh) }
                else if *or { (rl, 1) } else { (0, rh) }
            }
        }
        E::Unary(U::Neg, value, _) => {
            let (lo, hi) = bounds(value)?;
            (hi.checked_neg()?, lo.checked_neg()?)
        }
        E::Unary(U::Not, value, T::Bool) => {
            let (lo, hi) = bounds(value)?;
            (i128::from(hi == 0), i128::from(lo == 0))
        }
        E::Binary(op, left, right, _) => {
            let (al, ah) = bounds(left)?;
            let (bl, bh) = bounds(right)?;
            match op {
                B::Add => (al.checked_add(bl)?, ah.checked_add(bh)?),
                B::Sub => (al.checked_sub(bh)?, ah.checked_sub(bl)?),
                B::Mul => {
                    let values = [
                        al.checked_mul(bl)?,
                        al.checked_mul(bh)?,
                        ah.checked_mul(bl)?,
                        ah.checked_mul(bh)?,
                    ];
                    (*values.iter().min()?, *values.iter().max()?)
                }
                B::Div if al >= 0 && bl > 0 => (al / bh, ah / bl),
                B::Rem if al >= 0 && bl > 0 => {
                    if al == ah && bl == bh {
                        (al % bl, al % bl)
                    } else {
                        let step = if bl == bh { divisor_step(left, bh, facts) } else { 1 };
                        (0, ah.min(bh - step))
                    }
                }
                B::BitAnd if al >= 0 && bl >= 0 => {
                    if al == ah && bl == bh {
                        (al & bl, al & bl)
                    } else {
                        let hi = ah.min(bh);
                        let step = if bl == bh && u128::try_from(bh + 1).is_ok_and(|n| n.is_power_of_two()) {
                            divisor_step(left, bh + 1, facts)
                        } else { 1 };
                        (0, hi / step * step)
                    }
                }
                B::Shr if al >= 0 && bl >= 0 && bh < i128::from(left.ty().bytes() * 8) => {
                    (al >> bh, ah >> bl)
                }
                B::Shl if al >= 0 && bl >= 0 && bh < i128::from(left.ty().bytes() * 8) => {
                    (al.checked_shl(bl as u32)?, ah.checked_shl(bh as u32)?)
                }
                B::Lt => (i128::from(ah < bl), i128::from(al < bh)),
                B::Le => (i128::from(ah <= bl), i128::from(al <= bh)),
                B::Gt => (i128::from(al > bh), i128::from(ah > bl)),
                B::Ge => (i128::from(al >= bh), i128::from(ah >= bl)),
                B::Eq => (
                    i128::from(al == ah && bl == bh && al == bl),
                    i128::from(al <= bh && bl <= ah),
                ),
                B::Ne => (
                    i128::from(ah < bl || bh < al),
                    i128::from(al != ah || bl != bh || al != bl),
                ),
                B::And if left.ty() == T::Bool && right.ty() == T::Bool => (al & bl, ah & bh),
                B::Or if left.ty() == T::Bool && right.ty() == T::Bool => (al | bl, ah | bh),
                _ => return None,
            }
        }
        _ => return None,
    };
    let (lo, hi) = limits(e.ty())?;
    (range.0 >= lo && range.1 <= hi).then_some(range)
}

fn divisor_step(value: &E, modulus: i128, facts: &Facts) -> i128 {
    let Some(polynomial) = symbolic(value, facts) else { return 1; };
    let (mut a, mut b) = (i128::from(polynomial.coefficient_divisor()), modulus);
    while b != 0 { (a, b) = (b, a % b); }
    a.max(1)
}

/// Only constrain immutable scalar identities. A branch comparison is not a
/// license to invert wrapping arithmetic or assume an effectful helper pure.
fn assume(condition: &E, truth: bool, facts: &mut Facts, writes: &HashSet<String>) {
    fn identity<'a>(e: &'a E, facts: &Facts) -> Option<(&'a String, T)> {
        match e {
            E::Variable(name, ty) | E::Parameter { name, ty } => Some((name, *ty)),
            E::Cast(ty, value) => {
                let (lo, hi) = bounds_in(value, facts)?;
                let (min, max) = limits(*ty)?;
                if lo < min || hi > max { return None; }
                identity(value, facts)
            }
            _ => None,
        }
    }
    match condition {
        E::Unary(U::Not, value, T::Bool) => assume(value, !truth, facts, writes),
        E::Binary(B::And, a, b, T::Bool) if truth => {
            assume(a, true, facts, writes);
            assume(b, true, facts, writes);
        }
        E::Binary(B::Or, a, b, T::Bool) if !truth => {
            assume(a, false, facts, writes);
            assume(b, false, facts, writes);
        }
        E::Binary(op, a, b, T::Bool) => {
            let (name, ty, other, op) = match (identity(a, facts), identity(b, facts)) {
                (Some((name, ty)), _) => {
                    (name, ty, b.as_ref(), *op)
                }
                (_, Some((name, ty))) => {
                    let op = match op {
                        B::Lt => B::Gt,
                        B::Le => B::Ge,
                        B::Gt => B::Lt,
                        B::Ge => B::Le,
                        B::Eq => B::Eq,
                        B::Ne => B::Ne,
                        _ => return,
                    };
                    (name, ty, a.as_ref(), op)
                }
                _ => return,
            };
            if writes.contains(name) {
                return;
            }
            let Some((low, high)) = bounds_in(other, facts) else {
                return;
            };
            let Some((mut lo, mut hi)) = facts.get(name).copied().or_else(|| limits(ty)) else {
                return;
            };
            match (op, truth) {
                (B::Lt, true) | (B::Ge, false) => hi = hi.min(high - 1),
                (B::Le, true) | (B::Gt, false) => hi = hi.min(high),
                (B::Gt, true) | (B::Le, false) => lo = lo.max(low + 1),
                (B::Ge, true) | (B::Lt, false) => lo = lo.max(low),
                (B::Eq, true) | (B::Ne, false) => {
                    lo = lo.max(low);
                    hi = hi.min(high);
                }
                _ => return,
            }
            // Inconsistent conditions describe dead code, but this pass does
            // not remove control or manufacture facts for an empty interval.
            if lo <= hi {
                facts.insert(name.clone(), (lo, hi));
            }
        }
        _ => {}
    }
}
fn limits(t: T) -> Option<(i128, i128)> {
    Some(match t {
        T::Bool => (0, 1),
        T::I32 => (i128::from(i32::MIN), i128::from(i32::MAX)),
        T::U32 => (0, i128::from(u32::MAX)),
        T::I64 => (i128::from(i64::MIN), i128::from(i64::MAX)),
        T::U64 => (0, i128::from(u64::MAX)),
        _ => return None,
    })
}

/// Compare correlated affine addresses only after proving that each typed
/// subexpression fits its integer type. Polynomial cancellation by itself
/// cannot justify discarding modular arithmetic or failure behavior.
fn symbolic(e: &E, facts: &Facts) -> Option<seismic_lang::sym::Sym> {
    symbolic_with(e, facts, &|_| None)
}

pub(super) fn symbolic_with(e: &E, facts: &Facts, value: &dyn Fn(&str) -> Option<seismic_lang::sym::Sym>) -> Option<seismic_lang::sym::Sym> {
    use seismic_lang::sym::Sym;
    let symbolic = |e: &E, facts: &Facts| symbolic_with(e, facts, value);
    bounds_in(e, facts)?;
    Some(match e {
        E::Integer(_, _) => Sym::constant(i64::try_from(bounds_in(e, facts)?.0).ok()?),
        E::Variable(name, _) | E::Parameter { name, .. } => value(name).unwrap_or_else(|| Sym::param(name)),
        E::Cast(_, x) | E::Bitcast(_, x) => symbolic(x, facts)?,
        E::Binary(B::Shr, a, b, _) if bounds_in(a, facts)?.0 >= 0 => {
            let (lo, hi) = bounds_in(b, facts)?;
            if lo != hi || !(0..63).contains(&lo) { return None; }
            symbolic(a, facts)?.quot(&Sym::constant(1i64 << lo as u32))
        }
        E::Binary(B::BitAnd, a, b, _) if bounds_in(a, facts)?.0 >= 0 => {
            let (lo, hi) = bounds_in(b, facts)?;
            let modulus = i64::try_from(hi.checked_add(1)?).ok()?;
            if lo != hi || !u64::try_from(modulus).ok()?.is_power_of_two() { return None; }
            symbolic(a, facts)?.rem(&Sym::constant(modulus))
        }
        E::Binary(op, a, b, _) => {
            let a = symbolic(a, facts)?;
            let b = symbolic(b, facts)?;
            match op {
                B::Add => a.add(&b),
                B::Sub => a.sub(&b),
                B::Mul => a.mul(&b),
                B::Div => a.quot(&b),
                B::Rem => a.rem(&b),
                _ => return None,
            }
        }
        _ => return None,
    })
}

fn difference(a: &E, b: &E, facts: &Facts) -> Option<(i64, i64)> {

    symbolic(a, facts)?
        .sub(&symbolic(b, facts)?)
        .eval_interval(&|name| {
            let &(lo, hi) = facts.get(name)?;
            Some((i64::try_from(lo).ok()?, i64::try_from(hi).ok()?))
        })
}

#[cfg(test)]
fn expression(e: E) -> E {
    expression_with(e, &Facts::new())
}
fn expression_with(e: E, facts: &Facts) -> E {
    let expression = |e| expression_with(e, facts);
    let bounds = |e: &E| bounds_in(e, facts);
    let e = match e {
        E::Binary(op, a, b, ty) => {
            let a = expression(*a);
            let b = expression(*b);
            if ty == T::BF16 {
                // MSL bfloat operators compute float results. Make widening
                // and the semantic result rounding explicit in the same IR
                // that accounting sees, including nested arithmetic.
                E::binary(op, a.cast(T::F32), b.cast(T::F32), T::F32).cast(T::BF16)
            } else {
                E::Binary(op, Box::new(a), Box::new(b), ty)
            }
        }
        E::Unary(op, x, ty) => {
            let x = expression(*x);
            if ty == T::BF16 {
                E::Unary(op, Box::new(x.cast(T::F32)), T::F32).cast(T::BF16)
            } else {
                E::Unary(op, Box::new(x), ty)
            }
        }
        E::Builtin(name, args, ty) => {
            let widen = ty == T::BF16
                && matches!(
                    name.as_str(),
                    "fma"
                        | "exp"
                        | "fast::exp"
                        | "rsqrt"
                        | "sqrt"
                        | "log"
                        | "sin"
                        | "cos"
                        | "abs"
                        | "max"
                        | "min"
                );
            let args = args
                .into_iter()
                .map(expression)
                .map(|a| {
                    if widen && a.ty() == T::BF16 {
                        a.cast(T::F32)
                    } else {
                        a
                    }
                })
                .collect();
            if widen {
                E::Builtin(name, args, T::F32).cast(T::BF16)
            } else {
                E::Builtin(name, args, ty)
            }
        }
        E::Cast(t, x) => expression(*x).cast(t),
        E::Bitcast(t, x) => E::Bitcast(t, Box::new(expression(*x))),
        E::Read {
            name,
            index,
            space,
            ty,
        } => E::Read {
            name,
            index: Box::new(expression(*index)),
            space,
            ty,
        },
        E::Select(c, a, b) => E::Select(
            Box::new(expression(*c)),
            Box::new(expression(*a)),
            Box::new(expression(*b)),
        ),
        E::ShortCircuit { or, left, right } => E::ShortCircuit {
            or,
            left: Box::new(expression(*left)),
            right: Box::new(expression(*right)),
        },
        E::Helper(helper, args, ty) => {
            let args: Vec<_> = args.into_iter().map(expression).collect();
            match (helper, args.as_slice()) {
                (Helper::Read, [E::Variable(name, T::U64), index, count])
                    if bounds(index)
                        .zip(bounds(count))
                        .is_some_and(|((lo, hi), (count, _))| lo >= 0 && hi < count) =>
                {
                    E::Read {
                        name: name.clone(),
                        index: Box::new(index.clone()),
                        space: super::Space::Device,
                        ty,
                    }
                }
                (Helper::SliceValid, [start, end, extent])
                    if bounds(start)
                        .zip(bounds(end))
                        .zip(bounds(extent))
                        .is_some_and(|(((sl, sh), (el, eh)), (n, _))| {
                            sl >= 0
                                && (sh <= el
                                    || difference(end, start, facts).is_some_and(|(lo, _)| lo >= 0))
                                && eh <= n
                        }) =>
                {
                    E::Integer(1, T::Bool)
                }
                (Helper::SliceStart, [E::Integer(1, T::Bool), start]) => start.clone().cast(ty),
                (Helper::SliceExtent, [E::Integer(1, T::Bool), start, end]) => {
                    E::binary(B::Sub, end.clone(), start.clone(), T::I64).cast(ty)
                }
                (Helper::Index, [index, extent])
                    if bounds(index)
                        .zip(bounds(extent))
                        .is_some_and(|((lo, hi), (extent, _))| lo >= 0 && hi < extent) =>
                {
                    index.clone().cast(ty)
                }
                (
                    Helper::ShiftSigned | Helper::ShiftUnsigned,
                    [value, shift, E::Integer(left, T::Bool)],
                ) if bounds(shift).is_some_and(|(lo, hi)| lo >= 0 && hi < 32) => {
                    if *left != 0 {
                        let bits = if helper == Helper::ShiftSigned {
                            E::Bitcast(T::U32, Box::new(value.clone()))
                        } else {
                            value.clone()
                        };
                        let shifted = E::binary(B::Shl, bits, shift.clone().cast(T::U32), T::U32);
                        if helper == Helper::ShiftSigned {
                            E::Bitcast(T::I32, Box::new(shifted))
                        } else {
                            shifted
                        }
                    } else {
                        E::binary(B::Shr, value.clone(), shift.clone().cast(T::U32), ty)
                    }
                }
                (Helper::DivideUnsigned, [a, b, E::Integer(rem, T::Bool)])
                    if bounds(b).is_some_and(|(lo, _)| lo > 0) =>
                {
                    E::binary(
                        if *rem != 0 { B::Rem } else { B::Div },
                        a.clone(),
                        b.clone(),
                        ty,
                    )
                }
                (Helper::DivideSigned, [a, b, E::Integer(rem, T::Bool)])
                    if bounds(a).is_some_and(|(lo, _)| lo >= 0)
                        && bounds(b).is_some_and(|(lo, _)| lo > 0) =>
                {
                    E::binary(
                        if *rem != 0 { B::Rem } else { B::Div },
                        a.clone(),
                        b.clone(),
                        ty,
                    )
                }
                _ if args.iter().all(|arg| bounds(arg).is_some()) => {
                    if let Some(inlined) = super::helper::single_expression(helper, &args, ty) {
                        expression(inlined)
                    } else { E::Helper(helper, args, ty) }
                }
                _ => E::Helper(helper, args, ty),
            }
        }
        other => other,
    };
    let e = integer_identity(e, facts);
    if let Some((lo, hi)) = bounds(&e) {
        if lo == hi {
            if let Ok(value) = i64::try_from(lo) {
                return E::Integer(value, e.ty());
            }
        }
    }
    e
}

/// Identities remove only integer operations. An annihilating identity may
/// discard an operand only when the existing bounds analysis establishes that
/// its evaluation is pure and cannot fail. Floating arithmetic is untouched.
fn integer_identity(e: E, facts: &Facts) -> E {
    if let E::Bitcast(ty, value) = &e {
        if value.ty() == *ty {
            return (**value).clone();
        }
        if let E::Bitcast(_, inner) = value.as_ref() {
            if inner.ty() == *ty && inner.ty().bytes() == value.ty().bytes() {
                return (**inner).clone();
            }
        }
    }
    let E::Binary(op, a, b, ty) = &e else {
        return e;
    };
    if !matches!(ty, T::I32 | T::U32 | T::I64 | T::U64)
        || a.ty() != *ty
        || (!matches!(op, B::Shl | B::Shr) && b.ty() != *ty)
    {
        return e;
    }
    let constant = |e: &E| match e {
        E::Integer(_, _) => bounds_in(e, facts).map(|(lo, _)| lo),
        _ => None,
    };
    let (ac, bc) = (constant(a), constant(b));
    let value = |e: &E| e.clone().cast(*ty);
    match op {
        B::Add | B::BitOr | B::BitXor if bc == Some(0) => return value(a),
        B::Add | B::BitOr | B::BitXor if ac == Some(0) => return value(b),
        B::Sub | B::Shl | B::Shr if bc == Some(0) => return value(a),
        B::Mul | B::Div if bc == Some(1) => return value(a),
        B::Mul if ac == Some(1) => return value(b),
        B::Mul | B::BitAnd if bc == Some(0) && bounds_in(a, facts).is_some() => {
            return E::Integer(0, *ty);
        }
        B::Mul | B::BitAnd if ac == Some(0) && bounds_in(b, facts).is_some() => {
            return E::Integer(0, *ty);
        }
        B::Sub | B::BitXor if a == b && bounds_in(a, facts).is_some() => return E::Integer(0, *ty),
        B::Rem if bc == Some(1) && bounds_in(a, facts).is_some() => return E::Integer(0, *ty),
        B::BitAnd if bc.is_some_and(|mask| mask >= 0 && u128::try_from(mask + 1).is_ok_and(|n| n.is_power_of_two()))
            && bounds_in(a, facts).is_some_and(|(lo, hi)| lo >= 0 && hi <= bc.unwrap()) => return value(a),
        B::Div | B::Rem => {
            if bc.is_some_and(|divisor| divisor > 0)
                && bounds_in(a, facts).is_some_and(|(lo, hi)| lo >= 0 && hi < bc.unwrap()) {
                return if *op == B::Rem { value(a) } else { E::Integer(0, *ty) };
            }
            // Unsigned/nonnegative division by 2^k is a shift; remainder is
            // its low-bit mask. The divisor is a literal, so no evaluation is
            // lost. Signed negative division keeps its truncation semantics.
            if let Some(divisor) = bc
                .and_then(|n| u64::try_from(n).ok())
                .filter(|n| n.is_power_of_two())
            {
                if matches!(ty, T::U32 | T::U64)
                    || bounds_in(a, facts).is_some_and(|(lo, _)| lo >= 0)
                {
                    return if *op == B::Div {
                        E::Binary(
                            B::Shr,
                            Box::new(value(a)),
                            Box::new(E::Integer(i64::from(divisor.trailing_zeros()), T::U32)),
                            *ty,
                        )
                    } else {
                        E::Binary(
                            B::BitAnd,
                            Box::new(value(a)),
                            Box::new(E::Integer((divisor - 1) as i64, *ty)),
                            *ty,
                        )
                    };
                }
            }
        }
        _ => {}
    }
    e
}

pub(super) fn statement(s: S) -> S {
    statement_with(s, &Facts::new())
}
fn statement_with(s: S, facts: &Facts) -> S {
    let expression = |e| expression_with(e, facts);
    match s {
        S::Let { name, ty, value } => S::Let {
            name,
            ty,
            value: expression(value).cast(ty),
        },
        S::Assign { name, value } => S::Assign {
            name,
            value: expression(value),
        },
        S::Pointer {
            name,
            base,
            index,
            space,
            ty,
        } => S::Pointer {
            name,
            base,
            index: expression(index),
            space,
            ty,
        },
        S::VectorRead { name, base, index, ty, components } => S::VectorRead { name, base, index: expression(index), ty, components },
        S::Write {
            name,
            index,
            space,
            ty,
            value,
        } => S::Write {
            name,
            index: expression(index),
            space,
            ty,
            value: expression(value).cast(ty),
        },
        S::Evaluate(e) => {
            let e = expression(e);
            match &e {
                E::Helper(Helper::Write, args, _) => match args.as_slice() {
                    [E::Variable(name, T::U64), index, count, value]
                        if bounds_in(index, facts)
                            .zip(bounds_in(count, facts))
                            .is_some_and(|((lo, hi), (count, _))| lo >= 0 && hi < count) =>
                    {
                        S::Write {
                            name: name.clone(),
                            index: index.clone(),
                            space: super::Space::Device,
                            ty: value.ty(),
                            value: value.clone(),
                        }
                    }
                    _ => S::Evaluate(e),
                },
                _ => S::Evaluate(e),
            }
        }
        S::For {
            name,
            start,
            end,
            step,
        } => S::For {
            name,
            start: expression(start),
            end: expression(end),
            step,
        },
        // Preserve the selected control predicate. Facts discharge checked
        // scalar operations inside its region; they do not specialize away
        // the region's execution condition. Predicate specialization needs a
        // separately qualified native mapping for private packed storage.
        S::If(e) => S::If(expression_with(e, &Facts::new())),
        S::ReturnIf(e) => S::ReturnIf(expression(e)),
        S::Return(e) => S::Return(e.map(expression)),
        S::MatrixLoad {
            fragment,
            layout,
            base,
            offset,
            leading,
            space,
            transpose,
        } => S::MatrixLoad {
            fragment,
            layout,
            base,
            offset: expression(offset),
            leading: expression(leading),
            space,
            transpose,
        },
        S::MatrixStore {
            fragment,
            layout,
            base,
            offset,
            leading,
            space,
        } => S::MatrixStore {
            fragment,
            layout,
            base,
            offset: expression(offset),
            leading: expression(leading),
            space,
        },
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_identities_preserve_failed_evaluations_and_signed_division() {
        let x = E::variable("x", T::I32);
        for op in [B::Add, B::Sub, B::BitOr, B::BitXor, B::Shl, B::Shr] {
            assert_eq!(
                expression(E::binary(op, x.clone(), E::integer(0), T::I32)),
                x
            );
        }
        let u = E::variable("u", T::U32);
        for shift in [1, 3, 5, 31] {
            let divisor = E::Integer(1i64 << shift, T::U32);
            assert!(matches!(
                expression(E::binary(B::Div, u.clone(), divisor.clone(), T::U32)),
                E::Binary(B::Shr, ..)
            ));
            assert!(matches!(
                expression(E::binary(B::Rem, u.clone(), divisor, T::U32)),
                E::Binary(B::BitAnd, ..)
            ));
        }
        let signed = E::binary(B::Div, x.clone(), E::integer(8), T::I32);
        assert_eq!(
            expression(signed.clone()),
            signed,
            "negative values cannot use arithmetic shift for truncating division"
        );
        let positive = E::binary(B::BitAnd, x, E::integer(127), T::I32);
        let facts = Facts::from([("x".into(), (0, 127))]);
        assert!(matches!(
            expression_with(E::binary(B::Div, positive, E::integer(8), T::I32), &facts),
            E::Binary(B::Shr, ..)
        ));
        let failure = E::Helper(
            Helper::Index,
            vec![E::Integer(-1, T::I64), E::Integer(4, T::I64)],
            T::I64,
        );
        assert!(matches!(
            expression(E::binary(
                B::Mul,
                failure.clone(),
                E::Integer(0, T::I64),
                T::I64
            )),
            E::Binary(B::Mul, ..)
        ));
        assert!(matches!(
            expression(E::binary(B::Rem, failure, E::Integer(1, T::I64), T::I64)),
            E::Binary(B::Rem, ..)
        ));
        let floating = E::binary(
            B::Mul,
            E::variable("f", T::F32),
            E::Float(1.0f64.to_bits(), T::F32),
            T::F32,
        );
        assert_eq!(expression(floating.clone()), floating);
        let wrapped = E::Bitcast(
            T::I32,
            Box::new(E::Bitcast(T::U32, Box::new(E::variable("bits", T::I32)))),
        );
        assert_eq!(expression(wrapped), E::variable("bits", T::I32));
    }

    #[test]
    fn guarded_segments_discharge_only_dominated_checks() {
        let logical = E::variable("logical", T::I32);
        let guard = E::binary(
            B::And,
            E::binary(B::Gt, logical.clone(), E::integer(0), T::Bool),
            E::binary(B::Le, logical.clone(), E::integer(128), T::Bool),
            T::Bool,
        );
        let check = || {
            S::Evaluate(E::Helper(
                Helper::Index,
                vec![
                    E::variable("segment", T::I32).cast(T::I64),
                    E::Integer(128, T::I64),
                ],
                T::I64,
            ))
        };
        let definition = || S::Let {
            name: "segment".into(),
            ty: T::I32,
            value: E::binary(B::Sub, logical.clone(), E::integer(1), T::I32),
        };
        let statements = vec![
            S::If(guard),
            definition(),
            check(),
            S::Else,
            definition(),
            check(),
            S::End,
            definition(),
            check(),
        ];
        let mut sites: Vec<_> = statements
            .into_iter()
            .map(|statement| super::super::Site {
                operation: None,
                statement,
            })
            .collect();
        launch(&mut sites);
        assert!(!matches!(sites[2].statement, S::Evaluate(E::Helper(..))));
        assert!(matches!(sites[5].statement, S::Evaluate(E::Helper(..))));
        assert!(matches!(sites[8].statement, S::Evaluate(E::Helper(..))));
        let mut facts = Facts::new();
        assume(
            &E::binary(B::Gt, logical.clone(), E::integer(0), T::Bool),
            true,
            &mut facts,
            &HashSet::from(["logical".into()]),
        );
        assert!(
            !facts.contains_key("logical"),
            "loop-carried values cannot acquire branch-wide facts"
        );
    }

    #[test]
    fn packet_word_bounds_and_entry_guards_keep_their_integer_meaning() {
        let offset = E::binary(B::Mul, E::integer(16), E::variable("segment", T::I32), T::I32);
        let packet = E::binary(B::Shr, offset.clone(), E::Integer(6, T::U32), T::I32);
        let within = E::binary(B::BitAnd, offset, E::integer(63), T::I32);
        let word = E::binary(B::Add, E::binary(B::Mul, E::integer(8), packet, T::I32),
            E::binary(B::Shr, E::binary(B::Mul, E::integer(4), within, T::I32), E::Integer(5, T::U32), T::I32), T::I32);
        let facts = Facts::from([("segment".into(), (0, 255))]);
        assert_eq!(bounds_in(&word, &facts), Some((0, 510)));
        let second = E::binary(B::Add, word, E::integer(1), T::I32).cast(T::I64);
        assert!(!matches!(expression_with(E::Helper(Helper::Index, vec![second, E::Integer(512, T::I64)], T::I64), &facts), E::Helper(..)));

        let sites = vec![
            S::Let { name: "admitted".into(), ty: T::Bool, value: E::Helper(Helper::WorkItemLive, vec![E::variable("item", T::U32), E::Integer(17, T::U32)], T::Bool) },
            S::ReturnIf(E::Unary(U::Not, Box::new(E::variable("admitted", T::Bool)), T::Bool)),
            S::Evaluate(E::Helper(Helper::Index, vec![E::variable("item", T::U32).cast(T::I64), E::Integer(17, T::I64)], T::I64)),
        ];
        let mut body: Vec<_> = sites.into_iter().map(|statement| super::super::Site { operation: None, statement }).collect();
        launch(&mut body);
        assert!(matches!(&body[2].statement, S::Evaluate(E::Cast(T::I64, _))));
        // A mutable admission predicate cannot establish that the original
        // item comparison was true when this later early return executes.
        body.insert(1, super::super::Site { operation: None, statement: S::Assign { name: "admitted".into(), value: E::Integer(1, T::Bool) } });
        body[3].statement = S::Evaluate(E::Helper(Helper::Index, vec![E::variable("item", T::U32).cast(T::I64), E::Integer(17, T::I64)], T::I64));
        launch(&mut body);
        assert!(matches!(&body[3].statement, S::Evaluate(E::Helper(Helper::Index, ..))));
    }

    #[test]
    fn lexical_bounds_exclude_loop_carried_values_and_expired_scopes() {
        let index = |name: &str| {
            S::Evaluate(E::Helper(
                Helper::Index,
                vec![
                    E::variable(name, T::I32).cast(T::I64),
                    E::Integer(4, T::I64),
                ],
                T::I64,
            ))
        };
        let statements = vec![
            S::Let {
                name: "j".into(),
                ty: T::I32,
                value: E::integer(0),
            },
            S::For {
                name: "i".into(),
                start: E::integer(0),
                end: E::integer(4),
                step: 1,
            },
            index("i"),
            index("j"),
            S::Assign {
                name: "j".into(),
                value: E::binary(B::Add, E::variable("j", T::I32), E::integer(2), T::I32),
            },
            S::End,
            S::For {
                name: "i".into(),
                start: E::integer(0),
                end: E::integer(8),
                step: 1,
            },
            index("i"),
            S::End,
        ];
        let mut sites: Vec<_> = statements
            .into_iter()
            .map(|statement| super::super::Site {
                operation: None,
                statement,
            })
            .collect();
        launch(&mut sites);
        assert!(!matches!(sites[2].statement, S::Evaluate(E::Helper(..))));
        assert!(matches!(
            sites[3].statement,
            S::Evaluate(E::Helper(Helper::Index, _, _))
        ));
        assert!(matches!(
            sites[7].statement,
            S::Evaluate(E::Helper(Helper::Index, _, _))
        ));
    }

    #[test]
    fn bounded_integer_helpers_preserve_failures_and_wrapping() {
        let index = |i, n| E::Helper(Helper::Index, vec![i, E::Integer(n, T::I64)], T::I64);
        assert_eq!(expression(index(E::integer(3), 4)), E::Integer(3, T::I64));
        assert!(matches!(
            expression(index(E::integer(4), 4)),
            E::Helper(Helper::Index, _, _)
        ));
        let unknown = E::variable("code", T::U32);
        let masked = E::binary(B::BitAnd, unknown, E::Integer(15, T::U32), T::U32);
        assert!(!matches!(expression(index(masked, 16)), E::Helper(..)));
        let failed = index(E::integer(-1), 4);
        // An annihilating arithmetic result cannot erase a failed evaluation.
        assert!(matches!(
            expression(E::binary(B::Mul, failed, E::Integer(0, T::I64), T::I64)),
            E::Binary(..)
        ));
        let shift = |amount| {
            E::Helper(
                Helper::ShiftSigned,
                vec![
                    E::variable("x", T::I32),
                    E::Integer(amount, T::I64),
                    E::Integer(1, T::Bool),
                ],
                T::I32,
            )
        };
        assert!(matches!(expression(shift(31)), E::Bitcast(T::I32, _)));
        assert!(matches!(
            expression(shift(32)),
            E::Helper(Helper::ShiftSigned, _, _)
        ));
        let division = E::Helper(
            Helper::DivideSigned,
            vec![
                E::Integer(i64::from(i32::MIN), T::I32),
                E::integer(-1),
                E::Integer(0, T::Bool),
            ],
            T::I32,
        );
        assert!(matches!(
            expression(division),
            E::Helper(Helper::DivideSigned, _, _)
        ));
        let negative = E::Helper(
            Helper::DivideSigned,
            vec![E::integer(-7), E::integer(3), E::Integer(0, T::Bool)],
            T::I32,
        );
        assert!(
            matches!(expression(negative), E::Helper(Helper::DivideSigned, _, _)),
            "Euclidean division must not become truncating native division"
        );
        let wrapped = E::Bitcast(
            T::I32,
            Box::new(E::binary(
                B::Add,
                E::Integer(u32::MAX.into(), T::U32),
                E::Integer(1, T::U32),
                T::U32,
            )),
        );
        assert!(matches!(expression(wrapped), E::Bitcast(..)));

        let read = |index| {
            E::Helper(
                Helper::Read,
                vec![
                    E::variable("input", T::U64),
                    E::Integer(index, T::I64),
                    E::Integer(4, T::U64),
                ],
                T::F32,
            )
        };
        assert!(matches!(expression(read(3)), E::Read { .. }));
        assert!(matches!(expression(read(4)), E::Helper(Helper::Read, _, _)));
        assert!(matches!(
            expression(read(-1)),
            E::Helper(Helper::Read, _, _)
        ));
        // Even a proven-valid read remains a read when surrounding arithmetic
        // has a constant result: memory and floating semantics are retained.
        assert!(matches!(
            expression(E::binary(B::Mul, read(3), E::Float(0, T::F32), T::F32)),
            E::Binary(..)
        ));
    }

    #[test]
    fn bf16_rounding_is_visible_between_nested_operations() {
        let a = E::variable("a", T::BF16);
        let value = E::binary(
            B::Mul,
            E::binary(B::Add, a.clone(), a.clone(), T::BF16),
            a,
            T::BF16,
        );
        let realized = expression(value);
        let E::Cast(T::BF16, product) = &realized else {
            panic!("missing result rounding")
        };
        let E::Binary(B::Mul, first, _, T::F32) = product.as_ref() else {
            panic!("missing wide arithmetic")
        };
        let E::Cast(T::F32, rounded) = first.as_ref() else {
            panic!("missing operand widening")
        };
        assert!(matches!(rounded.as_ref(), E::Cast(T::BF16, _)));
        assert_eq!(expression(realized.clone()), realized);
    }
}

type Conditions = HashMap<String, std::sync::Arc<E>>;
fn assume_condition(condition: &E, truth: bool, facts: &mut Facts, writes: &HashSet<String>, conditions: &Conditions) {
    fn visit(condition: &E, truth: bool, facts: &mut Facts, writes: &HashSet<String>, conditions: &Conditions, seen: &mut HashSet<(String, bool)>) {
        match condition {
            E::Variable(name, T::Bool) if !writes.contains(name) => {
                if let Some(value) = conditions.get(name) {
                    if seen.insert((name.clone(), truth)) { visit(value, truth, facts, writes, conditions, seen); }
                }
            }
            E::Unary(U::Not, value, T::Bool) => visit(value, !truth, facts, writes, conditions, seen),
            E::Binary(B::And, a, b, T::Bool) | E::ShortCircuit { or: false, left: a, right: b } if truth => {
                visit(a, true, facts, writes, conditions, seen); visit(b, true, facts, writes, conditions, seen);
            }
            E::Binary(B::Or, a, b, T::Bool) | E::ShortCircuit { or: true, left: a, right: b } if !truth => {
                visit(a, false, facts, writes, conditions, seen); visit(b, false, facts, writes, conditions, seen);
            }
            _ => assume(condition, truth, facts, writes),
        }
    }
    visit(condition, truth, facts, writes, conditions, &mut HashSet::new());
}

/// Bounds are lexical facts about this completed terminal body. Scalar writes
/// are excluded before visiting any loop, so a fact from iteration zero can
/// never be used after a loop-carried update. Unknown textual implementations
/// disable this pass; their effects and scope cannot be inferred from strings.
pub(super) fn launch(sites: &mut [super::Site]) {
    walk_facts(sites, |_, site, facts| { site.statement = statement_with(site.statement.clone(), facts); });
}

pub(super) fn scope_facts(sites: &[super::Site], retained: &HashSet<usize>) -> HashMap<usize, Facts> {
    let mut result = HashMap::new();
    walk_facts(&mut sites.to_vec(), |index, _, facts| { if retained.contains(&index) { result.insert(index, facts.clone()); } });
    result
}

pub(super) fn walk_facts(sites: &mut [super::Site], mut visit: impl FnMut(usize, &mut super::Site, &Facts)) {
    if sites.iter().any(|s| matches!(s.statement, S::Unmapped(_))) {
        return;
    }
    let mut writes = HashSet::new();
    for site in sites.iter() {
        match &site.statement {
            S::Assign { name, .. } => {
                writes.insert(name.clone());
            }
            _ => {}
        }
    }
    let mut facts = Facts::new();
    // This name is the typed Metal kernel ABI's thread_index_in_simdgroup.
    // It is not a user scalar or a device-name-dependent performance policy.
    if !writes.contains("lane") {
        facts.insert(
            "lane".into(),
            (0, i128::from(crate::execution::SUBGROUP - 1)),
        );
    }
    let mut scopes = Vec::new();
    let mut conditions = Conditions::new();
    for (index, site) in sites.iter_mut().enumerate() {
        visit(index, site, &facts);
        match &site.statement {
            S::Let { name, value, ty } => {
                let bound = bounds_in(&value.clone().cast(*ty), &facts);
                conditions.remove(name);
                if *ty == T::Bool && !writes.contains(name) && bound.is_some() {
                    conditions.insert(name.clone(), std::sync::Arc::new(value.clone()));
                }
                facts.remove(name);
                if !writes.contains(name) {
                    if let Some(bound) = bound {
                        facts.insert(name.clone(), bound);
                    }
                }
            }
            S::For {
                name,
                start,
                end,
                step,
            } => {
                scopes.push((facts.clone(), None, conditions.clone()));
                let bounds = bounds_in(start, &facts).zip(bounds_in(end, &facts));
                facts.remove(name);
                if !writes.contains(name) && *step > 0 {
                    if let Some(((lo, _), (_, hi))) = bounds {
                        // The emitted induction variable is I32. Also prove
                        // its final increment cannot overflow that type.
                        if lo >= i128::from(i32::MIN)
                            && lo < hi
                            && hi - 1 + i128::from(*step) <= i128::from(i32::MAX)
                        {
                            facts.insert(name.clone(), (lo, hi - 1));
                        }
                    }
                }
            }
            S::ReturnIf(condition) => assume_condition(condition, false, &mut facts, &writes, &conditions),
            S::Scope => scopes.push((facts.clone(), None, conditions.clone())),
            S::If(condition) => {
                scopes.push((facts.clone(), Some(condition.clone()), conditions.clone()));
                assume_condition(condition, true, &mut facts, &writes, &conditions);
            }
            S::Else => {
                if let Some((parent, condition, parent_conditions)) = scopes.last() {
                    facts = parent.clone();
                    conditions = parent_conditions.clone();
                    if let Some(condition) = condition {
                        assume_condition(condition, false, &mut facts, &writes, &conditions);
                    }
                }
            }
            S::End => {
                if let Some((parent, _, parent_conditions)) = scopes.pop() {
                    facts = parent;
                    conditions = parent_conditions;
                }
            }
            _ => {}
        }
    }
}
