//! Backwards interval propagation from retained participation guards. These
//! enclosures are scoped interpretation facts; solver parameter domains remain
//! the original family domains. Unsupported arithmetic supplies no deduction.
use seismic_accounting::algebra::Value;
use seismic_lang::sym::{Atom, Sym};
use std::collections::BTreeMap;

#[derive(Clone, Default)]
pub(super) struct Enclosure {
    pub parameters: BTreeMap<String, (i64, i64)>,
    pub inconsistent: bool,
}
impl Enclosure {
    pub fn derive(parameters: &BTreeMap<String, Value>, intervals: &[(Sym, i64, i64)]) -> Self {
        let mut result = Self {
            parameters: parameters.iter().filter_map(|(name, value)| {
                Some((name.clone(), (i64::try_from(value.bounds().0).ok()?, i64::try_from(value.bounds().1).ok()?)))
            }).collect(),
            inconsistent: false,
        };
        // Bound propagation by the input graph, never by numeric domain size.
        // An unfinished fixed point is a weaker enclosure, not infeasibility.
        for _ in 0..=parameters.len().saturating_add(intervals.len()) {
            let mut changed = false;
            for (expression, low, high) in intervals {
                changed |= result.refine(expression, (i128::from(*low), i128::from(*high)), 0);
                if result.inconsistent { return result; }
            }
            if !changed { break; }
        }
        result
    }
    fn bounds(&self, expression: &Sym) -> Option<(i128, i128)> {
        expression.eval_interval(&|name| self.parameters.get(name).copied())
            .map(|(low, high)| (i128::from(low), i128::from(high)))
    }
    fn refine(&mut self, expression: &Sym, mut wanted: (i128, i128), depth: usize) -> bool {
        if self.inconsistent || depth >= 64 { return false; }
        if let Some((low, high)) = self.bounds(expression) {
            wanted = (wanted.0.max(low), wanted.1.min(high));
        }
        if wanted.0 > wanted.1 { self.inconsistent = true; return false; }
        let mut changed = false;
        for (monomial, _) in expression.monomials() {
            if monomial.len() != 1 { continue; }
            let Some((atom, &1)) = monomial.iter().next() else { continue; };
            let Some((coefficient, rest)) = expression.linear_in(atom) else { continue; };
            let Some((low, high)) = self.bounds(&rest) else { continue; };
            let (Some(mut lower), Some(mut upper)) = (wanted.0.checked_sub(high), wanted.1.checked_sub(low)) else { continue; };
            let mut scale = i128::from(coefficient);
            if scale < 0 {
                let (Some(a), Some(b)) = (upper.checked_neg(), lower.checked_neg()) else { continue; };
                lower = a; upper = b; scale = -scale;
            }
            let first = lower.div_euclid(scale) + i128::from(lower.rem_euclid(scale) != 0);
            let last = upper.div_euclid(scale);
            changed |= self.refine_atom(atom, (first, last), depth + 1);
            if self.inconsistent { break; }
        }
        changed
    }
    fn refine_atom(&mut self, atom: &Atom, mut wanted: (i128, i128), depth: usize) -> bool {
        if wanted.0 > wanted.1 { self.inconsistent = true; return false; }
        match atom {
            Atom::Param(name) => {
                let Some(range) = self.parameters.get_mut(name) else { return false; };
                wanted = (wanted.0.max(i128::from(range.0)), wanted.1.min(i128::from(range.1)));
                if wanted.0 > wanted.1 { self.inconsistent = true; return false; }
                let (Ok(low), Ok(high)) = (i64::try_from(wanted.0), i64::try_from(wanted.1)) else { return false; };
                let changed = *range != (low, high);
                *range = (low, high);
                changed
            },
            Atom::Quot(numerator, divisor) => {
                let (Some((nl, nh)), Some((dl, dh))) = (self.bounds(numerator), self.bounds(divisor)) else { return false; };
                if nl < 0 || dl <= 0 { return false; }
                let (ql, qh) = (wanted.0.max(nl / dh), wanted.1.min(nh / dl));
                if ql > qh { self.inconsistent = true; return false; }
                let Some(next) = qh.checked_add(1) else { return false; };
                // q*d <= n < (q+1)*d. Invert both operands using enclosing
                // bounds; correlations may tighten the result later.
                let divisor_low = nl / next + 1;
                let divisor_high = if ql > 0 { dh.min(nh / ql) } else { dh };
                let mut changed = self.refine(divisor, (dl.max(divisor_low), divisor_high), depth);
                if let (Some(low), Some(high)) = (ql.checked_mul(dl), next.checked_mul(dh).and_then(|value| value.checked_sub(1))) {
                    changed |= self.refine(numerator, (nl.max(low), nh.min(high)), depth);
                }
                changed
            },
            Atom::Rem(numerator, divisor) => {
                let (Some((nl, nh)), Some((dl, dh))) = (self.bounds(numerator), self.bounds(divisor)) else { return false; };
                if nl < 0 || dl <= 0 { return false; }
                // Euclidean remainder satisfies 0 <= r < d and r <= n. These
                // implications hold even when the quotient is still unknown.
                wanted = (wanted.0.max(0), wanted.1.min(nh).min(dh - 1));
                if wanted.0 > wanted.1 { self.inconsistent = true; return false; }
                let Some(minimum_divisor) = wanted.0.checked_add(1) else { return false; };
                let mut changed = self.refine(divisor, (dl.max(minimum_divisor), dh), depth);
                changed |= self.refine(numerator, (nl.max(wanted.0), nh), depth);
                if self.inconsistent { return changed; }

                // Narrowing an operand may turn the remainder into a constant.
                // Only a successfully evaluated, disjoint enclosure is a proof
                // of contradiction; overflow or unsupported evaluation is not.
                let remainder = Sym::atom(atom.clone());
                if let Some((low, high)) = self.bounds(&remainder) {
                    wanted = (wanted.0.max(low), wanted.1.min(high));
                    if wanted.0 > wanted.1 { self.inconsistent = true; return changed; }
                }
                let (Some((nl, nh)), Some((dl, dh))) = (self.bounds(numerator), self.bounds(divisor)) else { return changed; };
                if nl < 0 || dl <= 0 { return changed; }
                let quotient = nl / dh;
                if quotient == nh / dl {
                    // With a fixed quotient the exact relation n = q*d + r
                    // propagates both remainder bounds back to the operands.
                    if let (Some(low), Some(high)) = (
                        quotient.checked_mul(dl).and_then(|value| value.checked_add(wanted.0)),
                        quotient.checked_mul(dh).and_then(|value| value.checked_add(wanted.1)),
                    ) {
                        changed |= self.refine(numerator, (nl.max(low), nh.min(high)), depth);
                    }
                    if quotient > 0 {
                        if let (Some(low), Some(high)) = (nl.checked_sub(wanted.1), nh.checked_sub(wanted.0)) {
                            let low = low.div_euclid(quotient) + i128::from(low.rem_euclid(quotient) != 0);
                            let high = high.div_euclid(quotient);
                            changed |= self.refine(divisor, (dl.max(low), dh.min(high)), depth);
                        }
                    }
                    if let Some((low, high)) = self.bounds(&remainder) {
                        if high < wanted.0 || low > wanted.1 { self.inconsistent = true; }
                    }
                }
                changed
            },
        }
    }
}
