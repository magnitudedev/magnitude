//! Integer facts follow value selection and subgroup exchange without assigning
//! host floating-point semantics to device operations.
use super::*;

impl Derivation<'_> {
    pub(super) fn expression_facts(&self, expression: &Expression) -> Facts {
        Facts { ranges: self.expression_ranges(expression), affine: self.expression_affine(expression) }
    }
}

fn extremum(a: (i128, i128), b: (i128, i128), maximum: bool) -> (i128, i128) {
    if maximum { (a.0.max(b.0), a.1.max(b.1)) } else { (a.0.min(b.0), a.1.min(b.1)) }
}

pub(super) fn eager_selection(condition: Values, yes: &Facts, no: &Facts, active: u32) -> Facts {
    let mut result = Facts::default();
    for lane in 0..32 {
        if active & (1 << lane) == 0 { continue; }
        let selected = match condition[lane] { Some(0) => Some(no), Some(_) => Some(yes), None => None };
        if let Some(selected) = selected {
            result.ranges[lane] = selected.ranges[lane];
            result.affine[lane] = selected.affine[lane].clone();
        } else {
            result.ranges[lane] = yes.ranges[lane].zip(no.ranges[lane])
                .map(|((yl, yh), (nl, nh))| (yl.min(nl), yh.max(nh)));
            if yes.affine[lane] == no.affine[lane] { result.affine[lane] = yes.affine[lane].clone(); }
        }
    }
    result
}

fn extremum_affine(a: &affine::Value, b: &affine::Value, maximum: bool) -> Option<affine::Value> {
    let (lo, hi) = a.add(b.scale(-1)?)?.bounds()?;
    if hi <= 0 { Some(if maximum { b } else { a }.clone()) }
    else if lo >= 0 { Some(if maximum { a } else { b }.clone()) }
    else { None }
}

pub(super) fn builtin(name: &str, args: &[Expression], facts: &[Facts], ty: Type, active: u32) -> Facts {
    let mut result = Facts::default();
    if matches!(ty, Type::F16 | Type::BF16 | Type::F32) { return result; }
    if name == "simd_shuffle" && args.len() == 2 && args[0].ty() == ty {
        for lane in 0..32 {
            if active & (1 << lane) == 0 { continue; }
            let Some((lo, hi)) = facts[1].ranges[lane] else { continue; };
            if lo != hi || !(0..32).contains(&lo) || active & (1 << lo) == 0 { continue; }
            result.ranges[lane] = facts[0].ranges[lo as usize];
            result.affine[lane] = facts[0].affine[lo as usize].clone();
        }
    } else if matches!(name, "min" | "max") && args.len() == 2 && args.iter().all(|arg| arg.ty() == ty) {
        for lane in 0..32 {
            if active & (1 << lane) == 0 { continue; }
            result.ranges[lane] = facts[0].ranges[lane].zip(facts[1].ranges[lane])
                .map(|(a, b)| extremum(a, b, name == "max"));
            result.affine[lane] = facts[0].affine[lane].as_ref().zip(facts[1].affine[lane].as_ref())
                .and_then(|(a, b)| extremum_affine(a, b, name == "max"));
        }
    } else if matches!(name, "simd_min" | "simd_max") && args.len() == 1 && args[0].ty() == ty && active != 0 {
        let first = active.trailing_zeros() as usize;
        let mut range = facts[0].ranges[first];
        let mut affine = facts[0].affine[first].clone();
        for lane in first + 1..32 {
            if active & (1 << lane) == 0 { continue; }
            range = range.zip(facts[0].ranges[lane]).map(|(a, b)| extremum(a, b, name == "simd_max"));
            affine = affine.as_ref().zip(facts[0].affine[lane].as_ref())
                .and_then(|(a, b)| extremum_affine(a, b, name == "simd_max"));
        }
        for lane in 0..32 {
            if active & (1 << lane) != 0 {
                result.ranges[lane] = range;
                result.affine[lane] = affine.clone();
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    fn state() -> Derivation<'static> {
        let mut state = Derivation::new(Sink::Count {
            account: InvocationAccount { operations: vec![], unmapped: vec![], exhausted: None, visits: 0 },
            indices: Default::default(),
        }, DerivationLimits { instructions: 1000, operations: 1000 });
        state.env.insert("x".into(), [None; 32]);
        state.affine.insert("x".into(), std::array::from_fn(|lane|
            affine::Value::coordinate(0, 0, 100).add(affine::Value::constant(lane as i128))));
        state.ranges.insert("x".into(), std::array::from_fn(|lane| Some((lane as i128, lane as i128 + 100))));
        state
    }
    #[test]
    fn symbolic_selection_and_subgroup_exchange_preserve_correlated_addresses() {
        let mut state = state();
        let x = Expression::variable("x", Type::I32);
        let plus = Expression::binary(BinaryOp::Add, x.clone(), Expression::integer(3), Type::I32);
        let maximum = Expression::Builtin("max".into(), vec![x.clone(), plus], Type::I32);
        let select = Expression::Select(Box::new(Expression::Integer(1, Type::Bool)), Box::new(maximum), Box::new(x));
        let shuffle = Expression::Builtin("simd_shuffle".into(), vec![select, Expression::Integer(7, Type::U32)], Type::I32);
        assert_eq!(state.expr(&shuffle).unwrap().unwrap(), [None; 32]);
        let facts = state.expression_facts(&shuffle);
        assert_eq!(facts.ranges, [Some((10, 110)); 32]);
        assert_eq!(facts.affine[0], state.affine["x"][7].as_ref().and_then(|v| v.add(affine::Value::constant(3))));
        let reduced = Expression::Builtin("simd_min".into(), vec![Expression::variable("x", Type::I32)], Type::I32);
        state.expr(&reduced).unwrap().unwrap();
        assert_eq!(state.expression_affine(&reduced)[0], state.affine["x"][0]);
        state.active = 1;
        state.expr(&shuffle).unwrap().unwrap();
        assert!(state.expression_ranges(&shuffle)[0].is_none(), "inactive shuffle sources supply no value");
    }

    #[test]
    fn helper_arguments_use_the_declared_signed_width() {
        let mut state = state();
        // I32 -1 widens to I64 -1, then the checked index helper takes its
        // failure branch. Zero-extension would incorrectly admit this index.
        let helper = Expression::Helper(crate::support::Helper::Index,
            vec![Expression::Integer(-1, Type::I32), Expression::Integer(1i64 << 33, Type::I64), Expression::Integer(0, Type::U64)], Type::I64);
        assert!(state.expr(&helper).unwrap().unwrap_err().contains("validity failure"));
    }
}
