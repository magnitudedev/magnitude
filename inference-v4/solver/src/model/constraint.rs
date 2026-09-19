use super::*;

/// Closed constraint vocabulary. Every family owns exact evaluation, scope,
/// propagation and a sound relaxation; there is no trusted numeric callback.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Constraint {
    Equal {
        left: VarId,
        right: VarId,
    },
    NotEqual {
        left: VarId,
        right: VarId,
    },
    InDomain {
        variable: VarId,
        domain: Domain,
    },
    Table {
        variables: Vec<VarId>,
        tuples: Vec<Vec<i64>>,
    },
    LinearLe {
        terms: Vec<LinearTerm>,
        rhs: i128,
    },
    ExactlyOne {
        variables: Vec<VarId>,
    },
    BoolAnd {
        output: VarId,
        inputs: Vec<VarId>,
    },
    Implies {
        premise: Literal,
        consequence: Literal,
    },
    /// `active` false forces `variable` to its canonical inactive value.
    InactiveValue {
        active: Literal,
        variable: VarId,
        inactive: i64,
    },
    Arithmetic(Arithmetic),
    Schedule(crate::scheduling::SchedulingConstraint),
}
impl Constraint {
    pub(crate) fn heap_bytes(&self) -> usize {
        match self {
            Self::InDomain { domain, .. } => {
                domain.retained_bytes() - std::mem::size_of::<Domain>()
            }
            Self::Table { variables, tuples } => {
                variables.capacity() * std::mem::size_of::<VarId>()
                    + tuples.capacity() * std::mem::size_of::<Vec<i64>>()
                    + tuples
                        .iter()
                        .map(|t| t.capacity() * std::mem::size_of::<i64>())
                        .sum::<usize>()
            }
            Self::LinearLe { terms, .. } => terms.capacity() * std::mem::size_of::<LinearTerm>(),
            Self::ExactlyOne { variables } => variables.capacity() * std::mem::size_of::<VarId>(),
            Self::BoolAnd { inputs, .. } => inputs.capacity() * std::mem::size_of::<VarId>(),
            Self::Schedule(c) => match c {
                crate::scheduling::SchedulingConstraint::NoOverlap { intervals } => {
                    intervals.capacity() * std::mem::size_of::<crate::scheduling::Interval>()
                        + intervals
                            .iter()
                            .map(|i| i.presence.capacity() * std::mem::size_of::<VarId>())
                            .sum::<usize>()
                }
                crate::scheduling::SchedulingConstraint::Cumulative { reservations, .. } => {
                    reservations.capacity() * std::mem::size_of::<crate::scheduling::Reservation>()
                        + reservations
                            .iter()
                            .map(|r| r.interval.presence.capacity() * std::mem::size_of::<VarId>())
                            .sum::<usize>()
                }
                crate::scheduling::SchedulingConstraint::ActivityCumulative {
                    activities, ..
                } => {
                    activities.capacity()
                        * std::mem::size_of::<crate::scheduling::ActivityReservation>()
                }
                _ => 0,
            },
            _ => 0,
        }
    }
    pub fn scope(&self) -> Vec<VarId> {
        match self {
            Self::Equal { left, right } | Self::NotEqual { left, right } => {
                unique_scope([*left, *right])
            }
            Self::InDomain { variable, .. } => vec![*variable],
            Self::Table { variables, .. } | Self::ExactlyOne { variables } => {
                unique_scope(variables.iter().copied())
            }
            Self::LinearLe { terms, .. } => unique_scope(terms.iter().map(|t| t.variable)),
            Self::BoolAnd { output, inputs } => {
                unique_scope(inputs.iter().copied().chain([*output]))
            }
            Self::Implies {
                premise,
                consequence,
            } => unique_scope([premise.variable, consequence.variable]),
            Self::InactiveValue {
                active, variable, ..
            } => unique_scope([active.variable, *variable]),
            Self::Arithmetic(c) => c.scope(),
            Self::Schedule(c) => c.scope(),
        }
    }
    pub fn validate(&self, domains: &[Domain]) -> Result<()> {
        for variable in self.scope() {
            get_domain(domains, variable)?;
        }
        match self {
            Self::Table { variables, tuples } => {
                validate_table(variables, tuples.iter().cloned(), domains)
            }
            Self::LinearLe { terms, .. } => {
                if self.scope().iter().any(|v| domains[v.0].is_empty()) {
                    return Ok(());
                }
                linear_bounds(terms, domains).map(|_| ())
            }
            Self::InDomain { domain, .. } => domain.validate(),
            Self::ExactlyOne { variables } => {
                if unique_scope(variables.iter().copied()).len() != variables.len() {
                    return Err(Error::InvalidModel(
                        "exactly-one variables must be distinct".into(),
                    ));
                }
                validate_booleans(variables, domains)
            }
            Self::BoolAnd { output, inputs } => validate_booleans(
                &inputs.iter().copied().chain([*output]).collect::<Vec<_>>(),
                domains,
            ),
            Self::Schedule(c) => c.validate(domains),
            Self::Arithmetic(c) => c.validate(domains),
            _ => Ok(()),
        }
    }
    pub(crate) fn assess(&self, domains: &[Domain]) -> Result<Assessment> {
        for variable in self.scope() {
            if get_domain(domains, variable)?.is_empty() {
                return Ok(Assessment::infeasible());
            }
        }
        let status = match self {
            Self::Equal { left, right } => {
                if left == right {
                    Some(true)
                } else if domains[left.0].intersect(&domains[right.0])?.is_empty() {
                    Some(false)
                } else if domains[left.0].is_singleton() && domains[right.0].is_singleton() {
                    Some(true)
                } else {
                    None
                }
            }
            Self::NotEqual { left, right } => {
                if left == right {
                    Some(false)
                } else if domains[left.0].intersect(&domains[right.0])?.is_empty() {
                    Some(true)
                } else if domains[left.0].is_singleton() && domains[right.0].is_singleton() {
                    Some(false)
                } else {
                    None
                }
            }
            Self::InDomain { variable, domain } => {
                let intersection = domains[variable.0].intersect(domain)?;
                if intersection.is_empty() {
                    Some(false)
                } else if intersection.cardinality() == domains[variable.0].cardinality() {
                    Some(true)
                } else {
                    None
                }
            }
            Self::Table { variables, tuples } => {
                let matching = tuples
                    .iter()
                    .filter(|t| matching_tuple(variables, t, domains))
                    .count() as u128;
                if matching == 0 {
                    Some(false)
                } else {
                    let combinations = variables
                        .iter()
                        .try_fold(1_u128, |n, v| n.checked_mul(domains[v.0].cardinality()));
                    if combinations == Some(matching) {
                        Some(true)
                    } else {
                        None
                    }
                }
            }
            Self::LinearLe { terms, rhs } => {
                let (lower, upper) = linear_bounds(terms, domains)?;
                if lower > *rhs {
                    Some(false)
                } else if upper <= *rhs {
                    Some(true)
                } else {
                    None
                }
            }
            Self::ExactlyOne { variables } => {
                let fixed = variables
                    .iter()
                    .filter(|v| domains[v.0].singleton_value() == Some(1))
                    .count();
                let possible = variables
                    .iter()
                    .filter(|v| domains[v.0].contains(1))
                    .count();
                if fixed > 1 || possible == 0 {
                    Some(false)
                } else if fixed == 1 && possible == 1 {
                    Some(true)
                } else {
                    None
                }
            }
            Self::BoolAnd { output, inputs } => {
                let any_false = inputs
                    .iter()
                    .any(|v| domains[v.0].singleton_value() == Some(0));
                let all_true = inputs
                    .iter()
                    .all(|v| domains[v.0].singleton_value() == Some(1));
                let out = &domains[output.0];
                if any_false {
                    if !out.contains(0) {
                        Some(false)
                    } else if out.is_singleton() {
                        Some(true)
                    } else {
                        None
                    }
                } else if all_true {
                    if !out.contains(1) {
                        Some(false)
                    } else if out.is_singleton() {
                        Some(true)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            Self::Implies {
                premise,
                consequence,
            } => {
                if premise == consequence {
                    Some(true)
                } else {
                    match (premise.state(domains)?, consequence.state(domains)?) {
                        (Some(false), _) | (_, Some(true)) => Some(true),
                        (Some(true), Some(false)) => Some(false),
                        _ => None,
                    }
                }
            }
            Self::InactiveValue {
                active,
                variable,
                inactive,
            } => {
                let value = Literal::new(*variable, *inactive).state(domains)?;
                match (active.state(domains)?, value) {
                    (Some(true), _) | (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                }
            }
            Self::Schedule(c) => return c.assess(domains),
            Self::Arithmetic(c) => return c.assess(domains),
        };
        Ok(match status {
            Some(true) => Assessment::exact(0),
            Some(false) => Assessment::infeasible(),
            None => Assessment::bounded(0),
        })
    }
    pub(crate) fn remap(&self, map: &[VarId]) -> Self {
        let literal = |l: &Literal| Literal::new(map[l.variable.0], l.value);
        match self {
            Self::Equal { left, right } => Self::Equal {
                left: map[left.0],
                right: map[right.0],
            },
            Self::NotEqual { left, right } => Self::NotEqual {
                left: map[left.0],
                right: map[right.0],
            },
            Self::InDomain { variable, domain } => Self::InDomain {
                variable: map[variable.0],
                domain: domain.clone(),
            },
            Self::Table { variables, tuples } => Self::Table {
                variables: variables.iter().map(|v| map[v.0]).collect(),
                tuples: tuples.clone(),
            },
            Self::LinearLe { terms, rhs } => Self::LinearLe {
                terms: terms
                    .iter()
                    .map(|t| LinearTerm::new(map[t.variable.0], t.coefficient))
                    .collect(),
                rhs: *rhs,
            },
            Self::ExactlyOne { variables } => Self::ExactlyOne {
                variables: variables.iter().map(|v| map[v.0]).collect(),
            },
            Self::BoolAnd { output, inputs } => Self::BoolAnd {
                output: map[output.0],
                inputs: inputs.iter().map(|v| map[v.0]).collect(),
            },
            Self::Implies {
                premise,
                consequence,
            } => Self::Implies {
                premise: literal(premise),
                consequence: literal(consequence),
            },
            Self::InactiveValue {
                active,
                variable,
                inactive,
            } => Self::InactiveValue {
                active: literal(active),
                variable: map[variable.0],
                inactive: *inactive,
            },
            Self::Schedule(c) => Self::Schedule(c.remap(map)),
            Self::Arithmetic(c) => Self::Arithmetic(c.remap(map)),
        }
    }
}
fn validate_booleans(variables: &[VarId], domains: &[Domain]) -> Result<()> {
    for variable in variables {
        let domain = get_domain(domains, *variable)?;
        if domain.min().is_some_and(|v| v < 0) || domain.max().is_some_and(|v| v > 1) {
            return Err(Error::InvalidModel(format!(
                "variable {} requires a Boolean domain",
                variable.0
            )));
        }
    }
    Ok(())
}
