//! Exact lane addresses over the abstract dispatch coordinate. A common affine
//! translation preserves the union of lane intervals at every transaction
//! granularity dividing that translation. Intervals separately exclude wrap.
use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Value {
    pub base: i128,
    /// Independent coordinates with their exact coefficient and finite domain.
    pub terms: BTreeMap<u64, (i128, (i128, i128))>,
}
pub(crate) type Values = [Option<Value>; 32];
impl Value {
    pub fn domain(id: u64, domain: &seismic_accounting::workload::IntegerDomain) -> Result<Self, String> {
        Self::coordinate(id, 0, (domain.range.max - domain.range.min) / i128::from(domain.range.stride))
            .scale(i128::from(domain.range.stride)).and_then(|value| value.add(Self::constant(domain.range.min)))
            .ok_or_else(|| "symbolic input domain overflow".into())
    }
    pub fn constant(base: i128) -> Self { Self { base, terms: BTreeMap::new() } }
    pub fn coordinate(id: u64, lo: i128, hi: i128) -> Self {
        Self { base: 0, terms: BTreeMap::from([(id, (1, (lo, hi)))]) }
    }
    pub fn add(&self, other: Self) -> Option<Self> {
        let mut result = self.clone();
        result.base = result.base.checked_add(other.base)?;
        for (id, (coefficient, domain)) in other.terms {
            let entry = result.terms.entry(id).or_insert((0, domain));
            if entry.1 != domain { return None; }
            entry.0 = entry.0.checked_add(coefficient)?;
        }
        result.terms.retain(|_, (coefficient, _)| *coefficient != 0);
        Some(result)
    }
    pub fn scale(&self, factor: i128) -> Option<Self> {
        let mut result = Self::constant(self.base.checked_mul(factor)?);
        if factor != 0 {
            for (&id, &(coefficient, domain)) in &self.terms {
                result.terms.insert(id, (coefficient.checked_mul(factor)?, domain));
            }
        }
        Some(result)
    }
    pub fn bounds(&self) -> Option<(i128, i128)> {
        let (mut lo, mut hi) = (self.base, self.base);
        for &(coefficient, (first, last)) in self.terms.values() {
            let a = coefficient.checked_mul(first)?;
            let b = coefficient.checked_mul(last)?;
            lo = lo.checked_add(a.min(b))?;
            hi = hi.checked_add(a.max(b))?;
        }
        Some((lo, hi))
    }
    pub fn origin(&self) -> Option<i128> {
        self.terms.values().try_fold(self.base, |base, (coefficient, (lo, _))| base.checked_add(coefficient.checked_mul(*lo)?))
    }
    pub fn alignment(&self, allocation: u64) -> u64 {
        self.terms.values().fold(allocation, |alignment, (coefficient, _)|
            if *coefficient == 0 { alignment } else { alignment.min(1u64 << coefficient.unsigned_abs().trailing_zeros().min(63)) })
    }
}
impl Derivation<'_> {
    pub(super) fn expression_affine(&self, expression: &Expression) -> Values {
        if matches!(expression.ty(), Type::F16 | Type::BF16 | Type::F32) { return std::array::from_fn(|_| None); }
        if let Some(facts) = self.facts.get(expression) { return facts.affine.clone(); }
        std::array::from_fn(|lane| self.affine_value(expression, lane))
    }
    pub(super) fn affine_value(&self, e: &Expression, lane: usize) -> Option<Value> {
        if let Some(facts) = self.facts.get(e) { return facts.affine[lane].clone(); }
        use Expression as E;
        let result = match e {
            E::Variable(name, _) | E::Parameter { name, .. } if self.affine.get(name).and_then(|v| v[lane].as_ref()).is_some() => self.affine[name][lane].clone()?,
            E::Cast(_, value) => self.affine_value(value, lane)?,
            E::Unary(UnaryOp::Neg, value, _) => self.affine_value(value, lane)?.scale(-1)?,
            E::Binary(op, left, right, _) if left.ty() == right.ty() => {
                let a = self.affine_value(left, lane)?;
                let b = self.affine_value(right, lane)?;
                match op {
                    BinaryOp::Add => a.add(b)?,
                    BinaryOp::Sub => a.add(b.scale(-1)?)?,
                    BinaryOp::Mul if a.terms.is_empty() => b.scale(a.base)?,
                    BinaryOp::Mul if b.terms.is_empty() => a.scale(b.base)?,
                    BinaryOp::Shl if b.terms.is_empty() && b.base >= 0 && b.base < if matches!(left.ty(), Type::I64 | Type::U64) { 64 } else { 32 } && a.bounds()?.0 >= 0 => a.scale(1i128 << b.base)?,
                    BinaryOp::Div | BinaryOp::Rem if b.terms.is_empty() && b.base > 0 && a.terms.values().all(|(coefficient, _)| coefficient % b.base == 0) => {
                        // Truncating signed division only distributes over the
                        // translation when every numerator is nonnegative.
                        if a.bounds()?.0 < 0 { return None; }
                        if *op == BinaryOp::Div { Value { base: a.base.div_euclid(b.base), terms: a.terms.into_iter().map(|(id, (coefficient, domain))| (id, (coefficient / b.base, domain))).collect() } }
                        else { Value::constant(a.base.rem_euclid(b.base)) }
                    }
                    _ => return self.constant_affine(e, lane),
                }
            }
            _ => return self.constant_affine(e, lane),
        };
        let (lo, hi) = result.bounds()?;
        ranges::fit((lo, hi), e.ty())?;
        Some(result)
    }
    fn constant_affine(&self, e: &Expression, lane: usize) -> Option<Value> {
        let (lo, hi) = self.interval(e, lane)?;
        (lo == hi).then_some(Value::constant(lo))
    }
    pub(super) fn access(&self, name: &str, index: &Expression, indices: super::Values, element_bytes: u64, width: u64) -> Option<AccessPattern> {
        self.memory.vector_access(name, indices, index.ty(), element_bytes, width, self.active)
            .or_else(|| self.memory.symbolic_access(name, self.expression_affine(index), element_bytes, width, self.active))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_coordinates_preserve_cancellation_domains_and_alignment() {
        let group = Value::coordinate(0, 0, 1000).scale(256).unwrap();
        let inner = Value::coordinate(1, 0, 1023).scale(4).unwrap();
        let address = group.add(inner.clone()).unwrap().add(Value::constant(12)).unwrap();
        assert_eq!(address.bounds(), Some((12, 256000 + 4092 + 12)));
        assert_eq!(address.alignment(256), 4);
        assert_eq!(address.add(inner.scale(-1).unwrap()).unwrap().bounds(), Some((12, 256012)));
        assert!(Value::coordinate(0, 0, 4).add(Value::coordinate(0, 0, 5)).is_none());
    }
}
