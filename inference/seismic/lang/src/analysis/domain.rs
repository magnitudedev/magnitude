//! Conditions decided by an entry's domain.
//!
//! An entry's `where` clauses and dimension types relate its dimensions
//! (`4 * KC == 2 * P + S`). A branch of the portable body that tests such a
//! relation is taken or not for every call of the entry, and the trace follows
//! only the arm that runs. A condition is decided by refutation: its
//! negation, added to the domain's linear constraints, has no rational
//! solution (Fourier-Motzkin elimination over exact integer coefficients).
//! Nonlinear subterms are opaque unknowns. The procedure is sound and
//! incomplete: an undecided condition stays a guard.

use super::term::{Binary, Node, Term, Unary};

/// `Σ coefficient * atom + constant >= 0`, or `> 0` when `strict`.
#[derive(Clone, Debug, PartialEq)]
struct Constraint {
    /// `(atom, coefficient)`, ascending by atom, no zero coefficient.
    terms: Vec<(usize, i128)>,
    constant: i128,
    strict: bool,
}

impl Constraint {
    fn negated_sum(&self) -> Constraint {
        Constraint {
            terms: self.terms.iter().map(|(atom, coefficient)| (*atom, -coefficient)).collect(),
            constant: -self.constant,
            strict: self.strict,
        }
    }

    fn with_strict(mut self, strict: bool) -> Constraint {
        self.strict = strict;
        self
    }

    fn coefficient(&self, atom: usize) -> i128 {
        self.terms
            .iter()
            .find(|(found, _)| *found == atom)
            .map_or(0, |(_, coefficient)| *coefficient)
    }

    /// `left * self + right * other`, for positive multipliers. `None` when
    /// a coefficient leaves the exact range.
    fn combined(&self, left: i128, other: &Constraint, right: i128) -> Option<Constraint> {
        let mut terms: Vec<(usize, i128)> = Vec::new();
        for (source, factor) in [(self, left), (other, right)] {
            for (atom, coefficient) in &source.terms {
                let scaled = coefficient.checked_mul(factor)?;
                match terms.iter_mut().find(|(found, _)| found == atom) {
                    Some((_, existing)) => *existing = existing.checked_add(scaled)?,
                    None => terms.push((*atom, scaled)),
                }
            }
        }
        terms.retain(|(_, coefficient)| *coefficient != 0);
        terms.sort_by_key(|(atom, _)| *atom);
        let constant = self
            .constant
            .checked_mul(left)?
            .checked_add(other.constant.checked_mul(right)?)?;
        let divisor = terms
            .iter()
            .map(|(_, coefficient)| coefficient.unsigned_abs())
            .chain([constant.unsigned_abs()])
            .fold(0, gcd);
        let divisor = i128::try_from(divisor.max(1)).ok()?;
        Some(Constraint {
            terms: terms
                .into_iter()
                .map(|(atom, coefficient)| (atom, coefficient / divisor))
                .collect(),
            constant: constant / divisor,
            strict: self.strict || other.strict,
        })
    }

    /// Whether this constraint, having no unknown left, cannot hold.
    fn contradicts(&self) -> bool {
        self.terms.is_empty() && (self.constant < 0 || (self.strict && self.constant == 0))
    }
}

fn gcd(a: u128, b: u128) -> u128 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// The linear constraints an entry's domain places on its dimensions.
pub(crate) struct Domain {
    atoms: Vec<Term>,
    constraints: Vec<Constraint>,
    /// Comparisons over the domain's atoms already decided.
    decided: Vec<(Term, Option<bool>)>,
}

/// `value` as an exact integer coefficient.
fn exact(value: f64) -> Option<i128> {
    (value.fract() == 0.0 && value.abs() < 9_007_199_254_740_992.0).then_some(value as i128)
}

impl Domain {
    /// The domain stating `predicate` and that every one of `naturals` is
    /// nonnegative.
    pub(crate) fn new(predicate: &Term, naturals: impl IntoIterator<Item = Term>) -> Domain {
        let mut domain = Domain {
            atoms: Vec::new(),
            constraints: Vec::new(),
            decided: Vec::new(),
        };
        for natural in naturals {
            domain.assume(&Term::binary(Binary::Ge, natural, Term::constant(0.0)));
        }
        domain.assume(predicate);
        domain
    }

    fn assume(&mut self, fact: &Term) {
        match fact.node() {
            Node::Binary(Binary::And, a, b) => {
                self.assume(a);
                self.assume(b);
            }
            Node::Binary(op @ (Binary::Eq | Binary::Ge | Binary::Gt | Binary::Le | Binary::Lt), a, b) => {
                let Some(form) = a.clone().sub(b.clone()).affine() else {
                    return;
                };
                for (atom, _) in &form.terms {
                    if !self.atoms.contains(atom) {
                        self.atoms.push(atom.clone());
                    }
                }
                let Some(difference) = self.linear(a, b) else {
                    return;
                };
                self.constraints.extend(match op {
                    Binary::Eq => vec![difference.clone(), difference.negated_sum()],
                    Binary::Ge => vec![difference],
                    Binary::Gt => vec![difference.with_strict(true)],
                    Binary::Le => vec![difference.negated_sum()],
                    _ => vec![difference.negated_sum().with_strict(true)],
                });
            }
            _ => {}
        }
    }

    /// `a - b >= 0` over this domain's atoms. `None` when the difference is
    /// not an integer-linear form over them: an unknown the domain says
    /// nothing about leaves the difference free.
    fn linear(&self, a: &Term, b: &Term) -> Option<Constraint> {
        let difference = a.clone().sub(b.clone()).affine()?;
        let mut terms = Vec::new();
        for (atom, coefficient) in &difference.terms {
            let position = self.atoms.iter().position(|known| known == atom)?;
            terms.push((position, exact(*coefficient)?));
        }
        terms.sort_by_key(|(atom, _)| *atom);
        Some(Constraint {
            terms,
            constant: exact(difference.constant)?,
            strict: false,
        })
    }

    /// Whether `condition` holds for every call in the domain (`Some(true)`),
    /// for none (`Some(false)`), or is not decided by it.
    pub(crate) fn decide(&mut self, condition: &Term) -> Option<bool> {
        match condition.node() {
            Node::Constant(value) => Some(*value != 0.0),
            Node::Unary(Unary::Not, inner) => self.decide(inner).map(|value| !value),
            Node::Binary(op @ (Binary::And | Binary::Or), a, b) => {
                let (a, b) = (self.decide(a), self.decide(b));
                let absorbing = *op == Binary::Or;
                if a == Some(absorbing) || b == Some(absorbing) {
                    Some(absorbing)
                } else if a == Some(!absorbing) && b == Some(!absorbing) {
                    Some(!absorbing)
                } else {
                    None
                }
            }
            Node::Binary(op, a, b) => {
                let difference = self.linear(a, b)?;
                if let Some((_, known)) = self.decided.iter().find(|(term, _)| term == condition) {
                    return *known;
                }
                let decision = self.compare(*op, difference);
                self.decided.push((condition.clone(), decision));
                decision
            }
            _ => None,
        }
    }

    /// The truth of `difference op 0`, where `difference >= 0` is the form
    /// given.
    fn compare(&self, op: Binary, difference: Constraint) -> Option<bool> {
        let positive = difference.clone().with_strict(true);
        let negative = difference.negated_sum().with_strict(true);
        let nonpositive = difference.negated_sum();
        let nonnegative = difference;
        // A condition holds when its negation has no solution, and fails
        // when it has none itself.
        let decided = |holds: &[Constraint], fails: &[Constraint]| {
            if self.refuted(fails) {
                Some(true)
            } else if self.refuted(holds) {
                Some(false)
            } else {
                None
            }
        };
        let equal = || {
            if self.refuted(std::slice::from_ref(&positive))
                && self.refuted(std::slice::from_ref(&negative))
            {
                Some(true)
            } else if self.refuted(&[nonnegative.clone(), nonpositive.clone()]) {
                Some(false)
            } else {
                None
            }
        };
        match op {
            Binary::Eq => equal(),
            Binary::Ne => equal().map(|value| !value),
            Binary::Lt => decided(std::slice::from_ref(&negative), std::slice::from_ref(&nonnegative)),
            Binary::Le => decided(std::slice::from_ref(&nonpositive), std::slice::from_ref(&positive)),
            Binary::Gt => decided(std::slice::from_ref(&positive), std::slice::from_ref(&nonpositive)),
            Binary::Ge => decided(std::slice::from_ref(&nonnegative), std::slice::from_ref(&negative)),
            _ => None,
        }
    }

    /// Whether the domain together with `assumed` has no solution.
    fn refuted(&self, assumed: &[Constraint]) -> bool {
        let mut constraints = self.constraints.clone();
        constraints.extend_from_slice(assumed);
        loop {
            crate::cancellation::check();
            if constraints.iter().any(Constraint::contradicts) {
                return true;
            }
            let Some(atom) = constraints
                .iter()
                .flat_map(|constraint| constraint.terms.iter().map(|(atom, _)| *atom))
                .min()
            else {
                return false;
            };
            let (bounded, rest): (Vec<_>, Vec<_>) = constraints
                .into_iter()
                .partition(|constraint| constraint.coefficient(atom) != 0);
            let mut next = rest;
            for lower in bounded.iter().filter(|constraint| constraint.coefficient(atom) > 0) {
                for upper in bounded.iter().filter(|constraint| constraint.coefficient(atom) < 0) {
                    crate::cancellation::check();
                    // A combination outside the exact range is dropped:
                    // fewer constraints refute less, never more.
                    let combined = lower.combined(
                        -upper.coefficient(atom),
                        upper,
                        lower.coefficient(atom),
                    );
                    if let Some(combined) = combined {
                        if !next.contains(&combined) {
                            next.push(combined);
                        }
                    }
                }
            }
            constraints = next;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::term::Variable;
    use super::*;

    fn dimension(name: &str) -> Term {
        Term::variable(Variable::Dimension(name.into()))
    }

    fn times(factor: f64, term: Term) -> Term {
        Term::constant(factor).mul(term)
    }

    fn compare(op: Binary, a: Term, b: Term) -> Term {
        Term::binary(op, a, b)
    }

    #[test]
    fn dimension_relations_decide_the_conditions_they_imply() {
        let width = times(2.0, dimension("P")).add(dimension("S"));
        let predicate = [
            compare(Binary::Eq, times(4.0, dimension("KC")), width.clone()),
            compare(Binary::Eq, times(8.0, dimension("VC")), width.clone()),
            compare(Binary::Ge, width.clone(), Term::constant(32.0)),
        ]
        .into_iter()
        .reduce(|a, b| Term::binary(Binary::And, a, b))
        .unwrap();
        let naturals = ["P", "S", "KC", "VC", "R"].map(dimension);
        let mut domain = Domain::new(&predicate, naturals);

        let keys = compare(Binary::Eq, times(4.0, dimension("KC")), width.clone());
        let values = compare(Binary::Eq, times(4.0, dimension("VC")), width.clone());
        assert_eq!(domain.decide(&keys), Some(true));
        assert_eq!(domain.decide(&values), Some(false));
        assert_eq!(
            domain.decide(&compare(Binary::Ne, times(4.0, dimension("VC")), width.clone())),
            Some(true)
        );
        assert_eq!(domain.decide(&Term::unary(Unary::Not, values.clone())), Some(true));
        assert_eq!(domain.decide(&Term::binary(Binary::Or, values.clone(), keys.clone())), Some(true));
        assert_eq!(domain.decide(&Term::binary(Binary::And, values, keys)), Some(false));
        assert_eq!(
            domain.decide(&compare(Binary::Gt, dimension("VC"), Term::constant(3.0))),
            Some(true)
        );
        assert_eq!(
            domain.decide(&compare(Binary::Lt, dimension("KC"), dimension("VC"))),
            Some(false)
        );
        // Stated by nothing: a free dimension and a value of the call.
        assert_eq!(domain.decide(&compare(Binary::Gt, dimension("R"), Term::constant(0.0))), None);
        assert_eq!(domain.decide(&compare(Binary::Ge, dimension("R"), Term::constant(0.0))), Some(true));
        let data = Term::variable(Variable::Element {
            parameter: "visible".into(),
            index: Vec::new(),
        });
        assert_eq!(domain.decide(&compare(Binary::Ge, data, Term::constant(0.0))), None);
    }
}
