//! Exact affine integer expressions over independent checked invocation inputs.
//! Coordinates are normalized to 0..=steps. Arithmetic never samples inputs;
//! failure to represent wrapping or nonlinear results leaves analysis unresolved.
use crate::workload::{IntegerInput, IntegerRange};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Affine {
    constant: i128,
    terms: BTreeMap<IntegerInput, (i128, u64)>,
}
impl Affine {
    pub fn constant(value: i128) -> Self {
        Self {
            constant: value,
            terms: BTreeMap::new(),
        }
    }
    pub fn domain(symbol: IntegerInput, range: IntegerRange) -> Option<Self> {
        if range.stride == 0 || range.min > range.max || !range.contains(range.max) {
            return None;
        }
        let steps =
            u64::try_from(range.max.checked_sub(range.min)? / i128::from(range.stride)).ok()?;
        let terms = if steps == 0 {
            BTreeMap::new()
        } else {
            BTreeMap::from([(symbol, (i128::from(range.stride), steps))])
        };
        Some(Self {
            constant: range.min,
            terms,
        })
    }
    pub fn exact(&self) -> Option<i128> {
        self.terms.is_empty().then_some(self.constant)
    }
    pub fn bounds(&self) -> Option<(i128, i128)> {
        self.terms.values().try_fold(
            (self.constant, self.constant),
            |(lo, hi), &(coefficient, steps)| {
                let delta = coefficient.checked_mul(i128::from(steps))?;
                Some((lo.checked_add(delta.min(0))?, hi.checked_add(delta.max(0))?))
            },
        )
    }
    pub fn add(&self, other: &Self) -> Option<Self> {
        let mut result = Self {
            constant: self.constant.checked_add(other.constant)?,
            terms: self.terms.clone(),
        };
        for (&symbol, &(coefficient, steps)) in &other.terms {
            let current = result.terms.entry(symbol).or_insert((0, steps));
            if current.1 != steps {
                return None;
            }
            current.0 = current.0.checked_add(coefficient)?;
        }
        result.terms.retain(|_, term| term.0 != 0);
        result.bounds()?;
        Some(result)
    }
    pub fn sub(&self, other: &Self) -> Option<Self> {
        self.add(&other.scale(-1)?)
    }
    pub fn scale(&self, coefficient: i128) -> Option<Self> {
        let mut result = Self {
            constant: self.constant.checked_mul(coefficient)?,
            terms: self
                .terms
                .iter()
                .map(|(&symbol, &(c, steps))| Some((symbol, (c.checked_mul(coefficient)?, steps))))
                .collect::<Option<_>>()?,
        };
        result.terms.retain(|_, term| term.0 != 0);
        result.bounds()?;
        Some(result)
    }
    /// Integer interpretation at the specified width. A single wrap/sign
    /// region is affine; a crossing is unavailable until separately partitioned.
    pub fn interpreted(&self, bits: u32, signed: bool) -> Option<Self> {
        let period = 1i128.checked_shl(bits)?;
        let origin = if signed { -(period / 2) } else { 0 };
        let (lo, hi) = self.bounds()?;
        let cycle = lo.checked_sub(origin)?.div_euclid(period);
        if hi.checked_sub(origin)?.div_euclid(period) != cycle {
            return None;
        }
        self.add(&Self::constant(cycle.checked_mul(period)?.checked_neg()?))
    }
    pub fn aligned(&self, bytes: u64) -> bool {
        bytes != 0
            && self.constant.rem_euclid(i128::from(bytes)) == 0
            && self
                .terms
                .values()
                .all(|&(c, _)| c.rem_euclid(i128::from(bytes)) == 0)
    }
    /// Exact floor quotient when coordinate contributions are whole divisor
    /// multiples, or the complete expression occupies one quotient bucket.
    pub fn quotient(&self, divisor: u64) -> Option<Self> {
        if divisor == 0 {
            return None;
        }
        let divisor = i128::from(divisor);
        let (lo, hi) = self.bounds()?;
        if lo.div_euclid(divisor) == hi.div_euclid(divisor) {
            return Some(Self::constant(lo.div_euclid(divisor)));
        }
        if self.terms.values().any(|&(c, _)| c % divisor != 0) {
            return None;
        }
        Some(Self {
            constant: self.constant.div_euclid(divisor),
            terms: self
                .terms
                .iter()
                .map(|(&symbol, &(c, steps))| (symbol, (c / divisor, steps)))
                .collect(),
        })
    }
    pub fn remainder(&self, divisor: u64) -> Option<Self> {
        self.sub(&self.quotient(divisor)?.scale(i128::from(divisor))?)
    }
    /// A fixed residue follows from every independently varying coefficient.
    pub fn residue(&self, divisor: u64) -> Option<u64> {
        u64::try_from(self.remainder(divisor)?.exact()?).ok()
    }
    pub fn disjoint(&self, bytes: u64, other: &Self, other_bytes: u64) -> Option<bool> {
        let (lo, hi) = self.sub(other)?.bounds()?;
        if lo >= i128::from(other_bytes) || hi <= -i128::from(bytes) {
            Some(true)
        } else if lo > -i128::from(bytes) && hi < i128::from(other_bytes) {
            Some(false)
        } else {
            None
        }
    }
}
impl From<u64> for Affine {
    fn from(value: u64) -> Self {
        Self::constant(i128::from(value))
    }
}
