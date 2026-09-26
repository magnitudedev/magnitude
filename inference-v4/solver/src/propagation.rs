//! One inspectable factor propagation step. Search owns the fixed-point queue.
use crate::model::{
    self, Constraint, Domain, Error, Factor, FactorKind, Literal, Propagation, Result, VarId,
};
use std::collections::BTreeMap;

pub fn propagate_factor(factor: &Factor, domains: &mut [Domain]) -> Result<Propagation> {
    factor.validate(domains)?;
    propagate_factor_validated(factor, domains)
}
pub(crate) fn propagate_factor_validated(
    factor: &Factor,
    domains: &mut [Domain],
) -> Result<Propagation> {
    let mut unknown = BTreeMap::new();
    for guard in &factor.guards {
        if let Some(old) = unknown.get(&guard.variable) {
            if *old != guard.value {
                return Ok(Propagation::default());
            }
        }
        match guard.state(domains)? {
            Some(false) => return Ok(Propagation::default()),
            None => {
                unknown.insert(guard.variable, guard.value);
            }
            Some(true) => (),
        }
    }
    if factor.scope().iter().any(|v| domains[v.0].is_empty()) {
        return Ok(Propagation {
            changed: false,
            infeasible: true,
        });
    }
    let FactorKind::Constraint(constraint) = &factor.kind else {
        return Ok(Propagation::default());
    };
    if !unknown.is_empty() {
        // Reification: if the body is impossible assuming its guards, reject the
        // last possible activating literal; never constrain an inactive body.
        if unknown.len() == 1 {
            let mut active = domains.to_vec();
            for guard in &factor.guards {
                active[guard.variable.0] = Domain::singleton(guard.value);
            }
            if constraint.assess(&active)?.infeasible {
                let (variable, value) = unknown.into_iter().next().expect("one unknown");
                let next = domains[variable.0].without(value);
                return replace(domains, variable, next);
            }
        }
        return Ok(Propagation::default());
    }
    propagate_constraint(constraint, domains)
}

pub fn propagate_constraint(
    constraint: &Constraint,
    domains: &mut [Domain],
) -> Result<Propagation> {
    constraint.validate(domains)?;
    let assessment = constraint.assess(domains)?;
    if assessment.infeasible {
        return Ok(Propagation {
            changed: false,
            infeasible: true,
        });
    }
    if assessment.exact_cost.is_some() {
        return Ok(Propagation::default());
    }
    let mut result = Propagation::default();
    match constraint {
        Constraint::Equal { left, right } => {
            let both = domains[left.0].intersect(&domains[right.0])?;
            merge(&mut result, replace(domains, *left, both.clone())?);
            merge(&mut result, replace(domains, *right, both)?);
        }
        Constraint::NotEqual { left, right } => {
            if let Some(value) = domains[left.0].singleton_value() {
                let next = domains[right.0].without(value);
                merge(&mut result, replace(domains, *right, next)?);
            }
            if let Some(value) = domains[right.0].singleton_value() {
                let next = domains[left.0].without(value);
                merge(&mut result, replace(domains, *left, next)?);
            }
        }
        Constraint::InDomain { variable, domain } => {
            let next = domains[variable.0].intersect(domain)?;
            merge(&mut result, replace(domains, *variable, next)?);
        }
        Constraint::Table { variables, tuples } => {
            let mut supported = vec![Vec::new(); variables.len()];
            for tuple in tuples {
                if model::matching_tuple(variables, tuple, domains) {
                    for (i, value) in tuple.iter().enumerate() {
                        supported[i].push(*value);
                    }
                }
            }
            for (variable, values) in variables.iter().zip(supported) {
                merge(
                    &mut result,
                    replace(domains, *variable, Domain::set(values))?,
                );
            }
        }
        Constraint::ForbiddenTuple { variables, values } => {
            let mut unknown = None;
            for (variable, value) in variables.iter().zip(values) {
                if domains[variable.0].singleton_value() == Some(*value) {
                    continue;
                }
                if unknown.is_some() {
                    return Ok(result);
                }
                unknown = Some((*variable, *value));
            }
            if let Some((variable, value)) = unknown {
                let next = domains[variable.0].without(value);
                merge(&mut result, replace(domains, variable, next)?);
            }
        }
        Constraint::LinearLe { terms, rhs } => {
            merge(&mut result, propagate_linear_le(terms, *rhs, domains)?);
        }
        Constraint::ExactlyOne { variables } => {
            let fixed = variables
                .iter()
                .find(|v| domains[v.0].singleton_value() == Some(1))
                .copied();
            if let Some(selected) = fixed {
                for variable in variables {
                    if *variable != selected {
                        merge(&mut result, assign(domains, *variable, 0)?);
                    }
                }
            } else {
                let possible: Vec<_> = variables
                    .iter()
                    .filter(|v| domains[v.0].contains(1))
                    .copied()
                    .collect();
                if possible.len() == 1 {
                    merge(&mut result, assign(domains, possible[0], 1)?);
                }
            }
        }
        Constraint::Clause { literals } => {
            let unresolved = literals
                .iter()
                .filter_map(|literal| match literal.state(domains) {
                    Ok(None) => Some(Ok(*literal)),
                    Ok(Some(_)) => None,
                    Err(error) => Some(Err(error)),
                })
                .collect::<Result<Vec<_>>>()?;
            if let [literal] = unresolved.as_slice() {
                merge(
                    &mut result,
                    assign(domains, literal.variable, literal.value)?,
                );
            }
        }
        Constraint::BoolAnd { output, inputs } => {
            if inputs
                .iter()
                .any(|v| domains[v.0].singleton_value() == Some(0))
            {
                merge(&mut result, assign(domains, *output, 0)?);
            } else if inputs
                .iter()
                .all(|v| domains[v.0].singleton_value() == Some(1))
            {
                merge(&mut result, assign(domains, *output, 1)?);
            }
            if domains[output.0].singleton_value() == Some(1) {
                for input in inputs {
                    merge(&mut result, assign(domains, *input, 1)?);
                }
            } else if domains[output.0].singleton_value() == Some(0) {
                let possible_false: Vec<_> = model::unique_scope(inputs.iter().copied())
                    .into_iter()
                    .filter(|v| domains[v.0].contains(0))
                    .collect();
                if possible_false.len() == 1 {
                    merge(&mut result, assign(domains, possible_false[0], 0)?);
                }
            }
        }
        Constraint::BoolOr { output, inputs } => {
            if inputs
                .iter()
                .any(|v| domains[v.0].singleton_value() == Some(1))
            {
                merge(&mut result, assign(domains, *output, 1)?);
            } else if inputs
                .iter()
                .all(|v| domains[v.0].singleton_value() == Some(0))
            {
                merge(&mut result, assign(domains, *output, 0)?);
            }
            if domains[output.0].singleton_value() == Some(0) {
                for input in inputs {
                    merge(&mut result, assign(domains, *input, 0)?);
                }
            } else if domains[output.0].singleton_value() == Some(1) {
                let possible_true: Vec<_> = model::unique_scope(inputs.iter().copied())
                    .into_iter()
                    .filter(|v| domains[v.0].contains(1))
                    .collect();
                if possible_true.len() == 1 {
                    merge(&mut result, assign(domains, possible_true[0], 1)?);
                }
            }
        }
        Constraint::BoolNot { output, input } => {
            if let Some(value) = domains[input.0].singleton_value() {
                merge(&mut result, assign(domains, *output, 1 - value)?);
            }
            if let Some(value) = domains[output.0].singleton_value() {
                merge(&mut result, assign(domains, *input, 1 - value)?);
            }
        }
        Constraint::ReifiedLinearLe {
            indicator,
            terms,
            rhs,
        } => {
            let (lower, upper) = model::linear_bounds(terms, domains)?;
            if upper <= *rhs {
                merge(&mut result, assign(domains, *indicator, 1)?);
            } else if lower > *rhs {
                merge(&mut result, assign(domains, *indicator, 0)?);
            }
            match domains[indicator.0].singleton_value() {
                Some(1) => merge(&mut result, propagate_linear_le(terms, *rhs, domains)?),
                Some(0) => {
                    let lower = rhs
                        .checked_add(1)
                        .ok_or_else(|| Error::Overflow("reified affine complement".into()))?;
                    merge(&mut result, propagate_linear_ge(terms, lower, domains)?);
                }
                _ => (),
            }
        }
        Constraint::Implies {
            premise,
            consequence,
        } => {
            if premise.variable == consequence.variable && premise.value != consequence.value {
                let next = domains[premise.variable.0].without(premise.value);
                merge(&mut result, replace(domains, premise.variable, next)?);
            } else {
                if premise.state(domains)? == Some(true) {
                    merge(
                        &mut result,
                        assign(domains, consequence.variable, consequence.value)?,
                    );
                }
                if consequence.state(domains)? == Some(false) {
                    let next = domains[premise.variable.0].without(premise.value);
                    merge(&mut result, replace(domains, premise.variable, next)?);
                }
            }
        }
        Constraint::InactiveValue {
            active,
            variable,
            inactive,
        } => {
            if active.state(domains)? == Some(false) {
                merge(&mut result, assign(domains, *variable, *inactive)?);
            }
            if Literal::new(*variable, *inactive).state(domains)? == Some(false) {
                merge(&mut result, assign(domains, active.variable, active.value)?);
            }
        }
        Constraint::Schedule(schedule) => merge(&mut result, schedule.propagate(domains)?),
        Constraint::Arithmetic(arithmetic) => merge(&mut result, arithmetic.propagate(domains)?),
    }
    Ok(result)
}

fn propagate_linear_le(
    terms: &[crate::model::LinearTerm],
    rhs: i128,
    domains: &mut [Domain],
) -> Result<Propagation> {
    let mut result = Propagation::default();
    let merged = model::merged_terms(terms)?;
    let (lower, _) = model::linear_bounds(terms, domains)?;
    for (variable, coefficient) in merged {
        let domain = &domains[variable.0];
        let extreme = if coefficient > 0 {
            domain.min()
        } else {
            domain.max()
        };
        let Some(extreme) = extreme else {
            result.infeasible = true;
            break;
        };
        let own_min = coefficient
            .checked_mul(extreme as i128)
            .ok_or_else(|| Error::Overflow("propagation affine product".into()))?;
        let others = lower
            .checked_sub(own_min)
            .ok_or_else(|| Error::Overflow("propagation affine residual".into()))?;
        let allowance = rhs
            .checked_sub(others)
            .ok_or_else(|| Error::Overflow("propagation affine allowance".into()))?;
        let next = if coefficient > 0 {
            restrict_wide(domain, i64::MIN as i128, floor_div(allowance, coefficient)?)?
        } else {
            restrict_wide(domain, ceil_div(allowance, coefficient)?, i64::MAX as i128)?
        };
        merge(&mut result, replace(domains, variable, next)?);
    }
    Ok(result)
}

fn propagate_linear_ge(
    terms: &[crate::model::LinearTerm],
    rhs: i128,
    domains: &mut [Domain],
) -> Result<Propagation> {
    let mut result = Propagation::default();
    let merged = model::merged_terms(terms)?;
    let (_, upper) = model::linear_bounds(terms, domains)?;
    for (variable, coefficient) in merged {
        let domain = &domains[variable.0];
        let extreme = if coefficient > 0 {
            domain.max()
        } else {
            domain.min()
        };
        let Some(extreme) = extreme else {
            result.infeasible = true;
            break;
        };
        let own_max = coefficient
            .checked_mul(extreme as i128)
            .ok_or_else(|| Error::Overflow("propagation affine product".into()))?;
        let others = upper
            .checked_sub(own_max)
            .ok_or_else(|| Error::Overflow("propagation affine residual".into()))?;
        let requirement = rhs
            .checked_sub(others)
            .ok_or_else(|| Error::Overflow("propagation affine requirement".into()))?;
        let next = if coefficient > 0 {
            restrict_wide(
                domain,
                ceil_div(requirement, coefficient)?,
                i64::MAX as i128,
            )?
        } else {
            restrict_wide(
                domain,
                i64::MIN as i128,
                floor_div(requirement, coefficient)?,
            )?
        };
        merge(&mut result, replace(domains, variable, next)?);
    }
    Ok(result)
}

fn replace(domains: &mut [Domain], variable: VarId, next: Domain) -> Result<Propagation> {
    let current = domains
        .get_mut(variable.0)
        .ok_or_else(|| Error::InvalidModel("propagation variable missing".into()))?;
    // Every caller derives a subset. Equal cardinality therefore means equal
    // values, even if a table and an intersection choose different run layouts.
    // Keeping the original layout prevents representation-only fixed-point loops.
    let changed = current.cardinality() != next.cardinality();
    let infeasible = next.is_empty();
    if changed {
        *current = next;
    }
    Ok(Propagation {
        changed,
        infeasible,
    })
}
fn assign(domains: &mut [Domain], variable: VarId, value: i64) -> Result<Propagation> {
    let next = domains[variable.0].intersect(&Domain::singleton(value))?;
    replace(domains, variable, next)
}
fn merge(result: &mut Propagation, other: Propagation) {
    result.changed |= other.changed;
    result.infeasible |= other.infeasible;
}
fn restrict_wide(domain: &Domain, min: i128, max: i128) -> Result<Domain> {
    if min > i64::MAX as i128 || max < i64::MIN as i128 || min > max {
        return Ok(Domain::empty());
    }
    domain.restrict(
        min.max(i64::MIN as i128) as i64,
        max.min(i64::MAX as i128) as i64,
    )
}
fn floor_div(a: i128, b: i128) -> Result<i128> {
    let q = a
        .checked_div(b)
        .ok_or_else(|| Error::Overflow("affine division".into()))?;
    let r = a
        .checked_rem(b)
        .ok_or_else(|| Error::Overflow("affine remainder".into()))?;
    Ok(if r != 0 && (r > 0) != (b > 0) {
        q - 1
    } else {
        q
    })
}
fn ceil_div(a: i128, b: i128) -> Result<i128> {
    let q = a
        .checked_div(b)
        .ok_or_else(|| Error::Overflow("affine division".into()))?;
    let r = a
        .checked_rem(b)
        .ok_or_else(|| Error::Overflow("affine remainder".into()))?;
    Ok(if r != 0 && (r > 0) == (b > 0) {
        q + 1
    } else {
        q
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::LinearTerm;
    #[test]
    fn affine_propagation_preserves_all_feasible_tuples() {
        for a in -3..=3 {
            for b in -3..=3 {
                for rhs in -4..=4 {
                    let c = Constraint::LinearLe {
                        terms: vec![LinearTerm::new(VarId(0), a), LinearTerm::new(VarId(1), b)],
                        rhs,
                    };
                    let mut domains = vec![Domain::interval(-3, 3).unwrap(); 2];
                    let status = propagate_constraint(&c, &mut domains).unwrap();
                    for x in -3..=3 {
                        for y in -3..=3 {
                            if a as i128 * x as i128 + b as i128 * y as i128 <= rhs {
                                assert!(!status.infeasible);
                                assert!(domains[0].contains(x));
                                assert!(domains[1].contains(y));
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn false_guard_does_not_constrain_body() {
        let factor = Factor {
            guards: vec![Literal::new(VarId(0), 1)],
            kind: FactorKind::Constraint(Constraint::InDomain {
                variable: VarId(1),
                domain: Domain::singleton(9),
            }),
        };
        let mut domains = vec![Domain::singleton(0), Domain::interval(0, 100).unwrap()];
        assert!(!propagate_factor(&factor, &mut domains).unwrap().changed);
        assert_eq!(domains[1].cardinality(), 101);
    }
    #[test]
    fn impossible_body_disables_last_guard() {
        let factor = Factor {
            guards: vec![Literal::new(VarId(0), 1)],
            kind: FactorKind::Constraint(Constraint::InDomain {
                variable: VarId(1),
                domain: Domain::singleton(9),
            }),
        };
        let mut domains = vec![Domain::boolean(), Domain::interval(0, 2).unwrap()];
        assert!(propagate_factor(&factor, &mut domains).unwrap().changed);
        assert_eq!(domains[0], Domain::singleton(0));
    }
    #[test]
    fn nonconvex_domains_retain_all_supported_values() {
        let c = Constraint::Table {
            variables: vec![VarId(0), VarId(1)],
            tuples: vec![vec![1, 7], vec![4, 2]],
        };
        let mut domains = vec![Domain::interval(0, 10).unwrap(); 2];
        assert!(propagate_constraint(&c, &mut domains).unwrap().changed);
        assert_eq!(domains[0].values().collect::<Vec<_>>(), vec![1, 4]);
        assert_eq!(domains[1].values().collect::<Vec<_>>(), vec![2, 7]);
    }
    #[test]
    fn equivalent_run_layouts_do_not_oscillate_between_constraints() {
        let a = Domain::set([0, 2, 4, 5, 6]);
        let b = Domain::interval(0, 6).unwrap().without(1).without(3);
        assert_ne!(a, b);
        assert_eq!(
            a.values().collect::<Vec<_>>(),
            b.values().collect::<Vec<_>>()
        );
        let mut domains = vec![a, b];
        let equal = Constraint::Equal {
            left: VarId(0),
            right: VarId(1),
        };
        let table = Constraint::Table {
            variables: vec![VarId(0)],
            tuples: [0, 2, 4, 5, 6].into_iter().map(|v| vec![v]).collect(),
        };
        for _ in 0..3 {
            assert!(!propagate_constraint(&equal, &mut domains).unwrap().changed);
            assert!(!propagate_constraint(&table, &mut domains).unwrap().changed);
        }
    }
}
