//! Conservative per-lane integer intervals for dispatch-invariant control.
//! Unsupported arithmetic and any possible wrap lose the interval. These facts
//! supplement the same terminal walk; they do not generate operations or costs.
use super::*;
pub(super) type Ranges = [Option<(i128, i128)>; 32];
fn bounds(ty: Type) -> Option<(i128, i128)> {
    Some(match ty {
        Type::Bool => (0, 1), Type::I32 => (i32::MIN.into(), i32::MAX.into()),
        Type::U32 => (0, u32::MAX.into()), Type::I64 => (i64::MIN.into(), i64::MAX.into()),
        Type::U64 => (0, u64::MAX.into()), _ => return None,
    })
}
pub(super) fn fit(range: (i128, i128), ty: Type) -> Option<(i128, i128)> {
    let (lo, hi) = bounds(ty)?;
    (lo <= range.0 && range.0 <= range.1 && range.1 <= hi).then_some(range)
}
pub(super) fn exact(value: u64, ty: Type) -> Option<(i128, i128)> {
    let value = match ty {
        Type::I32 => i128::from(value as i32), Type::U32 => i128::from(value as u32),
        Type::I64 => i128::from(value as i64), Type::U64 => i128::from(value),
        Type::Bool => i128::from(value != 0), _ => return None,
    };
    Some((value, value))
}
fn integer_width(ty: Type) -> Option<u32> {
    match ty { Type::I32 | Type::U32 => Some(32), Type::I64 | Type::U64 => Some(64), _ => None }
}
/// On either side of the signed boundary an integer bitcast is an exact
/// translation. A range crossing that boundary has no single affine image.
pub(super) fn bitcast_offset(range: (i128, i128), from: Type, to: Type) -> Option<i128> {
    let bits = integer_width(from)?;
    if integer_width(to)? != bits { return None; }
    fit(range, from)?;
    if from == to { return Some(0); }
    if matches!(from, Type::I32 | Type::I64) {
        if range.0 >= 0 { Some(0) }
        else if range.1 < 0 { Some(1i128 << bits) }
        else { None }
    } else {
        let sign = 1i128 << (bits - 1);
        if range.1 < sign { Some(0) }
        else if range.0 >= sign { Some(-(1i128 << bits)) }
        else { None }
    }
}
fn bitcast(range: (i128, i128), from: Type, to: Type) -> Option<(i128, i128)> {
    if integer_width(from)? != integer_width(to)? { return None; }
    fit(range, from)?;
    match bitcast_offset(range, from, to) {
        Some(offset) => fit((range.0.checked_add(offset)?, range.1.checked_add(offset)?), to),
        None => bounds(to),
    }
}
impl Derivation<'_> {
    pub(super) fn expression_ranges(&self, e: &Expression) -> Ranges {
        if matches!(e.ty(), Type::F16 | Type::BF16 | Type::F32) { return [None; 32]; }
        std::array::from_fn(|lane| self.interval(e, lane))
    }
    pub(super) fn interval(&self, e: &Expression, lane: usize) -> Option<(i128, i128)> {
        let observed = self.facts.get(e).and_then(|facts| facts.ranges[lane]);
        let assumed = self.assumptions.get(e).and_then(|facts| facts.ranges[lane]);
        match (observed, assumed) {
            (Some((a, b)), Some((c, d))) => return fit((a.max(c), b.min(d)), e.ty()),
            (Some(range), None) | (None, Some(range)) => return fit(range, e.ty()),
            (None, None) => {},
        }
        use Expression as E;
        let range = match e {
            E::Integer(n, ty) => exact(*n as u64, *ty)?,
            E::Variable(name, ty) | E::Parameter { name, ty } => self.ranges.get(name)
                .and_then(|r| r[lane]).or_else(|| self.env.get(name).and_then(|v| v[lane]).and_then(|v| exact(v, *ty)))?,
            E::Cast(_, value) => self.interval(value, lane)?,
            E::Bitcast(ty, value) => bitcast(self.interval(value, lane)?, value.ty(), *ty)?,
            E::VectorElement { name, component, ty } => self.env.get(&format!("{name}[{component}]"))
                .and_then(|v| v[lane]).and_then(|v| exact(v, *ty))?,
            E::Unary(UnaryOp::Neg, value, _) => {
                let (lo, hi) = self.interval(value, lane)?;
                (hi.checked_neg()?, lo.checked_neg()?)
            }
            E::Unary(UnaryOp::Not, value, _) => {
                let (lo, hi) = self.interval(value, lane)?;
                if lo == 0 && hi == 0 { (1, 1) } else if lo > 0 || hi < 0 { (0, 0) } else { return None; }
            }
            E::Binary(op, left, right, _) => {
                if left.ty() != right.ty() { return None; }
                if matches!(op, BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge | BinaryOp::Eq | BinaryOp::Ne) {
                    if let Some((lo, hi)) = self.affine_value(left, lane).as_ref().and_then(|a|
                        self.affine_value(right, lane).as_ref().and_then(|b| a.add(b.scale(-1)?))).and_then(|v| v.bounds()) {
                        let (yes, no) = match op {
                            BinaryOp::Lt => (hi < 0, lo >= 0), BinaryOp::Le => (hi <= 0, lo > 0),
                            BinaryOp::Gt => (lo > 0, hi <= 0), BinaryOp::Ge => (lo >= 0, hi < 0),
                            BinaryOp::Eq => (lo == 0 && hi == 0, hi < 0 || lo > 0),
                            BinaryOp::Ne => (hi < 0 || lo > 0, lo == 0 && hi == 0), _ => unreachable!(),
                        };
                        if yes { return Some((1, 1)); }
                        if no { return Some((0, 0)); }
                    }
                }
                let (a, b) = self.interval(left, lane)?;
                let (c, d) = self.interval(right, lane)?;
                let boolean = |yes: bool, no: bool| if yes { Some((1, 1)) } else if no { Some((0, 0)) } else { None };
                match op {
                    BinaryOp::Add => (a.checked_add(c)?, b.checked_add(d)?),
                    BinaryOp::Sub => (a.checked_sub(d)?, b.checked_sub(c)?),
                    BinaryOp::Mul => {
                        let values = [a.checked_mul(c)?, a.checked_mul(d)?, b.checked_mul(c)?, b.checked_mul(d)?];
                        (*values.iter().min()?, *values.iter().max()?)
                    }
                    BinaryOp::Div if c > 0 && a >= 0 => (a / d, b / c),
                    BinaryOp::Rem if c > 0 && a >= 0 => {
                        let first = a / d;
                        let last = b / c;
                        if first == last { (a.checked_sub(first.checked_mul(d)?)?, b.checked_sub(first.checked_mul(c)?)?) }
                        else { (0, b.min(d - 1)) }
                    }
                    BinaryOp::And => boolean((a > 0 || b < 0) && (c > 0 || d < 0), (a == 0 && b == 0) || (c == 0 && d == 0))?,
                    BinaryOp::Or => boolean((a > 0 || b < 0) || (c > 0 || d < 0), a == 0 && b == 0 && c == 0 && d == 0)?,
                    BinaryOp::Shl if c == d && c >= 0 && c < if matches!(left.ty(), Type::I64 | Type::U64) { 64 } else { 32 } && a >= 0 => (a.checked_mul(1i128 << c)?, b.checked_mul(1i128 << c)?),
                    BinaryOp::Shr if c == d && c >= 0 && c < if matches!(left.ty(), Type::I64 | Type::U64) { 64 } else { 32 } && a >= 0 => (a >> c, b >> c),
                    BinaryOp::BitAnd if c == d && c >= 0 && a >= 0 && ((c + 1) as u128).is_power_of_two() => {
                        let modulus = c + 1;
                        if a / modulus == b / modulus { (a % modulus, b % modulus) } else { (0, c) }
                    }
                    BinaryOp::Lt => boolean(b < c, a >= d)?,
                    BinaryOp::Le => boolean(b <= c, a > d)?,
                    BinaryOp::Gt => boolean(a > d, b <= c)?,
                    BinaryOp::Ge => boolean(a >= d, b < c)?,
                    BinaryOp::Eq => boolean(a == b && c == d && a == c, b < c || d < a)?,
                    BinaryOp::Ne => boolean(b < c || d < a, a == b && c == d && a == c)?,
                    _ => return None,
                }
            }
            _ => return None,
        };
        fit(range, e.ty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn state() -> Derivation<'static> {
        Derivation::new(Sink::Count { account: InvocationAccount { operations: vec![], unmapped: vec![], exhausted: None, visits: 0 }, indices: Default::default() },
            DerivationLimits { instructions: 100, operations: 100 })
    }
    #[test]
    fn integer_ranges_cover_concrete_arithmetic_and_predicates() {
        let mut state = state();
        for (lo, hi) in [(-4, -2), (-2, 3), (0, 7)] {
            state.ranges.insert("x".into(), [Some((lo, hi)); 32]);
            for constant in -5..=8 {
                for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Lt, BinaryOp::Le,
                    BinaryOp::Gt, BinaryOp::Ge, BinaryOp::Eq, BinaryOp::Ne] {
                    let ty = if matches!(op, BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul) { Type::I32 } else { Type::Bool };
                    let expression = Expression::binary(op, Expression::variable("x", Type::I32), Expression::integer(constant), ty);
                    if let Some((lower, upper)) = state.expression_ranges(&expression)[0] {
                        for value in lo..=hi {
                            let bits = integer_binary(op, value as u64, constant as u64, Type::I32).unwrap();
                            let (actual, _) = exact(bits, ty).unwrap();
                            assert!(lower <= actual && actual <= upper);
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn overflowing_and_ambiguous_ranges_do_not_establish_constants() {
        let mut state = state();
        state.ranges.insert("x".into(), [Some((i128::from(u32::MAX) - 1, i128::from(u32::MAX))); 32]);
        let x = Expression::variable("x", Type::U32);
        let overflow = Expression::binary(BinaryOp::Add, x.clone(), Expression::Integer(1, Type::U32), Type::U32);
        assert_eq!(state.expression_ranges(&overflow), [None; 32]);
        let cast = Expression::Cast(Type::I32, Box::new(x.clone()));
        assert_eq!(state.expression_ranges(&cast), [None; 32]);
        let boundary = Expression::binary(BinaryOp::Lt, x, Expression::Integer(i64::from(u32::MAX), Type::U32), Type::Bool);
        assert_eq!(state.expression_ranges(&boundary), [None; 32]);
        state.active = 1;
        state.assign("x", [None; 32]);
        assert!(!state.ranges.contains_key("x"), "assignments must invalidate old interval facts");
    }
}
