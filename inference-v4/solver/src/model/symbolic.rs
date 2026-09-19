//! Bounded algebra directly required by unresolved work/partition geometry.
//! Relations preserve their finite solution sets without tables of combinations.
use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Arithmetic {
    Product {
        left: VarId,
        right: VarId,
        product: VarId,
    },
    CeilDiv {
        numerator: VarId,
        denominator: VarId,
        quotient: VarId,
    },
    DivRem {
        numerator: VarId,
        denominator: VarId,
        quotient: VarId,
        remainder: VarId,
    },
    Minimum {
        left: VarId,
        right: VarId,
        result: VarId,
    },
    Maximum {
        left: VarId,
        right: VarId,
        result: VarId,
    },
}
impl Arithmetic {
    pub fn scope(&self) -> Vec<VarId> {
        unique_scope(match self {
            Self::Product {
                left,
                right,
                product,
            } => vec![*left, *right, *product],
            Self::CeilDiv {
                numerator,
                denominator,
                quotient,
            } => vec![*numerator, *denominator, *quotient],
            Self::DivRem {
                numerator,
                denominator,
                quotient,
                remainder,
            } => vec![*numerator, *denominator, *quotient, *remainder],
            Self::Minimum {
                left,
                right,
                result,
            }
            | Self::Maximum {
                left,
                right,
                result,
            } => vec![*left, *right, *result],
        })
    }
    pub fn validate(&self, domains: &[Domain]) -> Result<()> {
        for var in self.scope() {
            let domain = get_domain(domains, var)?;
            if domain.min().is_some_and(|n| n < 0) {
                return Err(Error::InvalidModel(
                    "symbolic size/count relations require nonnegative domains".into(),
                ));
            }
        }
        let denominator = match self {
            Self::CeilDiv { denominator, .. } | Self::DivRem { denominator, .. } => {
                Some(*denominator)
            }
            _ => None,
        };
        if denominator.is_some_and(|d| domains[d.0].contains(0)) {
            return Err(Error::InvalidModel(
                "symbolic division requires a positive denominator domain".into(),
            ));
        }
        Ok(())
    }
    pub fn assess(&self, domains: &[Domain]) -> Result<Assessment> {
        self.validate(domains)?;
        if self.scope().iter().any(|v| domains[v.0].is_empty()) {
            return Ok(Assessment::infeasible());
        }
        let scope = self.scope();
        let mut narrowed: Vec<_> = scope.iter().map(|v| domains[v.0].clone()).collect();
        let local = |v: &VarId| VarId(scope.binary_search(v).expect("scope contains operand"));
        let relation = match self {
            Self::Product {
                left,
                right,
                product,
            } => Self::Product {
                left: local(left),
                right: local(right),
                product: local(product),
            },
            Self::CeilDiv {
                numerator,
                denominator,
                quotient,
            } => Self::CeilDiv {
                numerator: local(numerator),
                denominator: local(denominator),
                quotient: local(quotient),
            },
            Self::DivRem {
                numerator,
                denominator,
                quotient,
                remainder,
            } => Self::DivRem {
                numerator: local(numerator),
                denominator: local(denominator),
                quotient: local(quotient),
                remainder: local(remainder),
            },
            Self::Minimum {
                left,
                right,
                result,
            } => Self::Minimum {
                left: local(left),
                right: local(right),
                result: local(result),
            },
            Self::Maximum {
                left,
                right,
                result,
            } => Self::Maximum {
                left: local(left),
                right: local(right),
                result: local(result),
            },
        };
        if relation.propagate(&mut narrowed)?.infeasible {
            return Ok(Assessment::infeasible());
        }
        let fixed = self.scope().iter().all(|v| domains[v.0].is_singleton());
        let zero_case = match self {
            Self::Product {
                left,
                right,
                product,
            } => {
                (domains[left.0].singleton_value() == Some(0)
                    || domains[right.0].singleton_value() == Some(0))
                    && domains[product.0].singleton_value() == Some(0)
            }
            Self::CeilDiv {
                numerator,
                quotient,
                ..
            } => {
                domains[numerator.0].singleton_value() == Some(0)
                    && domains[quotient.0].singleton_value() == Some(0)
            }
            Self::DivRem {
                numerator,
                quotient,
                remainder,
                ..
            } => {
                domains[numerator.0].singleton_value() == Some(0)
                    && domains[quotient.0].singleton_value() == Some(0)
                    && domains[remainder.0].singleton_value() == Some(0)
            }
            Self::Minimum {
                left,
                right,
                result,
            } => minimum_fixed(domains, *left, *right, *result),
            Self::Maximum {
                left,
                right,
                result,
            } => maximum_fixed(domains, *left, *right, *result),
        };
        Ok(if fixed || zero_case {
            Assessment::exact(0)
        } else {
            Assessment::bounded(0)
        })
    }
    pub fn propagate(&self, domains: &mut [Domain]) -> Result<Propagation> {
        self.validate(domains)?;
        if self.scope().iter().any(|v| domains[v.0].is_empty()) {
            return Ok(Propagation {
                changed: false,
                infeasible: true,
            });
        }
        let mut state = State {
            domains,
            outcome: Propagation::default(),
        };
        match self {
            Self::Product {
                left,
                right,
                product,
            } => {
                let (a0, a1) = state.range(*left);
                let (b0, b1) = state.range(*right);
                state.restrict(*product, a0 * b0, a1 * b1)?;
                if state.outcome.infeasible {
                    return Ok(state.outcome);
                }
                let (p0, p1) = state.range(*product);
                if b1 > 0 {
                    state.restrict(
                        *left,
                        ceil(p0, b1),
                        if b0 > 0 { p1 / b0 } else { i64::MAX as i128 },
                    )?;
                }
                if a1 > 0 {
                    state.restrict(
                        *right,
                        ceil(p0, a1),
                        if a0 > 0 { p1 / a0 } else { i64::MAX as i128 },
                    )?;
                }
            }
            Self::CeilDiv {
                numerator,
                denominator,
                quotient,
            } => {
                let (n0, n1) = state.range(*numerator);
                let (d0, d1) = state.range(*denominator);
                state.restrict(*quotient, ceil(n0, d1), ceil(n1, d0))?;
                if state.outcome.infeasible {
                    return Ok(state.outcome);
                }
                let (q0, q1) = state.range(*quotient);
                state.restrict(
                    *numerator,
                    if q0 == 0 { 0 } else { (q0 - 1) * d0 + 1 },
                    q1 * d1,
                )?;
                if q1 > 0 {
                    state.restrict(
                        *denominator,
                        ceil(n0, q1).max(1),
                        if q0 > 1 {
                            (n1 - 1) / (q0 - 1)
                        } else {
                            i64::MAX as i128
                        },
                    )?;
                }
            }
            Self::DivRem {
                numerator,
                denominator,
                quotient,
                remainder,
            } => {
                let (_, d1) = state.range(*denominator);
                state.restrict(*remainder, 0, d1 - 1)?;
                if state.outcome.infeasible {
                    return Ok(state.outcome);
                }
                let (r0, _) = state.range(*remainder);
                state.restrict(*denominator, r0 + 1, i64::MAX as i128)?;
                if state.outcome.infeasible {
                    return Ok(state.outcome);
                }
                let (d0, d1) = state.range(*denominator);
                let (q0, q1) = state.range(*quotient);
                let (r0, r1) = state.range(*remainder);
                state.restrict(*numerator, d0 * q0 + r0, d1 * q1 + r1)?;
                if state.outcome.infeasible {
                    return Ok(state.outcome);
                }
                let (n0, n1) = state.range(*numerator);
                state.restrict(*quotient, ceil((n0 - r1).max(0), d1), (n1 - r0) / d0)?;
                state.restrict(*remainder, (n0 - d1 * q1).max(0), n1 - d0 * q0)?;
                if q1 > 0 {
                    state.restrict(
                        *denominator,
                        ceil((n0 - r1).max(0), q1).max(1),
                        if q0 > 0 {
                            (n1 - r0) / q0
                        } else {
                            i64::MAX as i128
                        },
                    )?;
                }
            }
            Self::Minimum {
                left,
                right,
                result,
            } => {
                let (a0, a1) = state.range(*left);
                let (b0, b1) = state.range(*right);
                state.restrict(*result, a0.min(b0), a1.min(b1))?;
                if state.outcome.infeasible {
                    return Ok(state.outcome);
                }
                let (r0, r1) = state.range(*result);
                state.restrict(*left, r0, i64::MAX as i128)?;
                state.restrict(*right, r0, i64::MAX as i128)?;
                if a0 > r1 {
                    state.restrict(*right, 0, r1)?;
                }
                if b0 > r1 {
                    state.restrict(*left, 0, r1)?;
                }
            }
            Self::Maximum {
                left,
                right,
                result,
            } => {
                let (a0, a1) = state.range(*left);
                let (b0, b1) = state.range(*right);
                state.restrict(*result, a0.max(b0), a1.max(b1))?;
                if state.outcome.infeasible {
                    return Ok(state.outcome);
                }
                let (r0, r1) = state.range(*result);
                state.restrict(*left, 0, r1)?;
                state.restrict(*right, 0, r1)?;
                if a1 < r0 {
                    state.restrict(*right, r0, i64::MAX as i128)?;
                }
                if b1 < r0 {
                    state.restrict(*left, r0, i64::MAX as i128)?;
                }
            }
        }
        Ok(state.outcome)
    }
    pub(crate) fn remap(&self, map: &[VarId]) -> Self {
        match self {
            Self::Product {
                left,
                right,
                product,
            } => Self::Product {
                left: map[left.0],
                right: map[right.0],
                product: map[product.0],
            },
            Self::CeilDiv {
                numerator,
                denominator,
                quotient,
            } => Self::CeilDiv {
                numerator: map[numerator.0],
                denominator: map[denominator.0],
                quotient: map[quotient.0],
            },
            Self::DivRem {
                numerator,
                denominator,
                quotient,
                remainder,
            } => Self::DivRem {
                numerator: map[numerator.0],
                denominator: map[denominator.0],
                quotient: map[quotient.0],
                remainder: map[remainder.0],
            },
            Self::Minimum {
                left,
                right,
                result,
            } => Self::Minimum {
                left: map[left.0],
                right: map[right.0],
                result: map[result.0],
            },
            Self::Maximum {
                left,
                right,
                result,
            } => Self::Maximum {
                left: map[left.0],
                right: map[right.0],
                result: map[result.0],
            },
        }
    }
}
fn ceil(n: i128, d: i128) -> i128 {
    n / d + i128::from(n % d != 0)
}
fn minimum_fixed(domains: &[Domain], a: VarId, b: VarId, r: VarId) -> bool {
    domains[r.0].singleton_value().is_some_and(|value| {
        (domains[a.0].singleton_value() == Some(value)
            && domains[b.0].min().is_some_and(|v| v >= value))
            || (domains[b.0].singleton_value() == Some(value)
                && domains[a.0].min().is_some_and(|v| v >= value))
    })
}
fn maximum_fixed(domains: &[Domain], a: VarId, b: VarId, r: VarId) -> bool {
    domains[r.0].singleton_value().is_some_and(|value| {
        (domains[a.0].singleton_value() == Some(value)
            && domains[b.0].max().is_some_and(|v| v <= value))
            || (domains[b.0].singleton_value() == Some(value)
                && domains[a.0].max().is_some_and(|v| v <= value))
    })
}
struct State<'a> {
    domains: &'a mut [Domain],
    outcome: Propagation,
}
impl State<'_> {
    fn range(&self, var: VarId) -> (i128, i128) {
        (
            self.domains[var.0].min().expect("checked nonempty") as i128,
            self.domains[var.0].max().expect("checked nonempty") as i128,
        )
    }
    fn restrict(&mut self, var: VarId, min: i128, max: i128) -> Result<()> {
        if self.outcome.infeasible {
            return Ok(());
        }
        let next = if min > max || min > i64::MAX as i128 || max < 0 {
            Domain::empty()
        } else {
            self.domains[var.0].restrict(min.max(0) as i64, max.min(i64::MAX as i128) as i64)?
        };
        self.outcome.changed |= next != self.domains[var.0];
        self.outcome.infeasible |= next.is_empty();
        self.domains[var.0] = next;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn relation_holds(relation: &Arithmetic, values: &[i64]) -> bool {
        let v = |id: &VarId| values[id.0] as i128;
        match relation {
            Arithmetic::Product {
                left,
                right,
                product,
            } => v(left) * v(right) == v(product),
            Arithmetic::CeilDiv {
                numerator,
                denominator,
                quotient,
            } => {
                let n = v(numerator);
                let d = v(denominator);
                (n + d - 1) / d == v(quotient)
            }
            Arithmetic::DivRem {
                numerator,
                denominator,
                quotient,
                remainder,
            } => {
                v(numerator) / v(denominator) == v(quotient)
                    && v(numerator) % v(denominator) == v(remainder)
            }
            Arithmetic::Minimum {
                left,
                right,
                result,
            } => v(left).min(v(right)) == v(result),
            Arithmetic::Maximum {
                left,
                right,
                result,
            } => v(left).max(v(right)) == v(result),
        }
    }
    fn relations() -> Vec<Arithmetic> {
        vec![
            Arithmetic::Product {
                left: VarId(0),
                right: VarId(1),
                product: VarId(2),
            },
            Arithmetic::CeilDiv {
                numerator: VarId(0),
                denominator: VarId(1),
                quotient: VarId(2),
            },
            Arithmetic::DivRem {
                numerator: VarId(0),
                denominator: VarId(1),
                quotient: VarId(2),
                remainder: VarId(3),
            },
            Arithmetic::Minimum {
                left: VarId(0),
                right: VarId(1),
                result: VarId(2),
            },
            Arithmetic::Maximum {
                left: VarId(0),
                right: VarId(1),
                result: VarId(2),
            },
        ]
    }
    #[test]
    fn exact_semantics_matches_independent_arithmetic() {
        for relation in relations() {
            for a in 0..=6 {
                for b in 1..=4 {
                    for c in 0..=8 {
                        for d in 0..=4 {
                            let values = [a, b, c, d];
                            let domains = values.map(Domain::singleton);
                            let assessment = relation.assess(&domains).unwrap();
                            assert_eq!(
                                !assessment.infeasible,
                                relation_holds(&relation, &values),
                                "{relation:?}, {values:?}"
                            );
                            if !assessment.infeasible {
                                assert_eq!(assessment.exact_cost, Some(0));
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn interval_and_holey_propagation_never_removes_a_feasible_tuple() {
        for relation in relations() {
            for seed in 0..40 {
                let mut domains = vec![
                    Domain::interval(0, 6).unwrap(),
                    Domain::interval(1, 4).unwrap(),
                    Domain::interval(0, 8).unwrap(),
                    Domain::interval(0, 4).unwrap(),
                ];
                for (index, domain) in domains.iter_mut().enumerate() {
                    if seed % 2 == 0 {
                        *domain = domain.without((seed / 2 + index) as i64 % 5);
                    }
                    if seed % 3 == 0 {
                        *domain = domain.restrict((seed / 3 + index) as i64 % 3, 8).unwrap();
                    }
                }
                let before = domains.clone();
                let outcome = relation.propagate(&mut domains).unwrap();
                for a in before[0].values() {
                    for b in before[1].values() {
                        for c in before[2].values() {
                            for d in before[3].values() {
                                let values = [a, b, c, d];
                                if relation_holds(&relation, &values) {
                                    assert!(
                                        !outcome.infeasible,
                                        "{relation:?} {before:?} {values:?}"
                                    );
                                    for (domain, value) in domains.iter().zip(values) {
                                        assert!(
                                            domain.contains(value),
                                            "{relation:?} {before:?} {values:?}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn extreme_products_are_not_truncated_and_empty_work_has_no_tiles() {
        let product = Arithmetic::Product {
            left: VarId(0),
            right: VarId(1),
            product: VarId(2),
        };
        assert!(
            product
                .assess(&[
                    Domain::singleton(i64::MAX),
                    Domain::singleton(i64::MAX),
                    Domain::interval(0, i64::MAX).unwrap()
                ])
                .unwrap()
                .infeasible
        );
        let ceil = Arithmetic::CeilDiv {
            numerator: VarId(0),
            denominator: VarId(1),
            quotient: VarId(2),
        };
        assert_eq!(
            ceil.assess(&[
                Domain::singleton(0),
                Domain::interval(1, i64::MAX).unwrap(),
                Domain::singleton(0)
            ])
            .unwrap()
            .exact_cost,
            Some(0)
        );
        assert!(ceil
            .validate(&[
                Domain::singleton(1),
                Domain::interval(0, 2).unwrap(),
                Domain::singleton(1)
            ])
            .is_err());
    }
}
