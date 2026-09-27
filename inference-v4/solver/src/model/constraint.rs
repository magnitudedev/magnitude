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
    /// Forbids exactly one tuple while admitting every other combination.
    ForbiddenTuple {
        variables: Vec<VarId>,
        values: Vec<i64>,
    },
    LinearLe {
        terms: Vec<LinearTerm>,
        rhs: i128,
    },
    ExactlyOne {
        variables: Vec<VarId>,
    },
    /// A disjunction of signed Boolean literals. An empty clause is false.
    Clause {
        literals: Vec<Literal>,
    },
    BoolAnd {
        output: VarId,
        inputs: Vec<VarId>,
    },
    BoolOr {
        output: VarId,
        inputs: Vec<VarId>,
    },
    BoolNot {
        output: VarId,
        input: VarId,
    },
    /// `indicator == 1` exactly when the affine sum is at most `rhs`.
    ReifiedLinearLe {
        indicator: VarId,
        terms: Vec<LinearTerm>,
        rhs: i128,
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
            Self::ForbiddenTuple { variables, values } => {
                variables.capacity() * std::mem::size_of::<VarId>()
                    + values.capacity() * std::mem::size_of::<i64>()
            }
            Self::LinearLe { terms, .. } | Self::ReifiedLinearLe { terms, .. } => {
                terms.capacity() * std::mem::size_of::<LinearTerm>()
            }
            Self::ExactlyOne { variables } => variables.capacity() * std::mem::size_of::<VarId>(),
            Self::Clause { literals } => literals.capacity() * std::mem::size_of::<Literal>(),
            Self::BoolAnd { inputs, .. } | Self::BoolOr { inputs, .. } => {
                inputs.capacity() * std::mem::size_of::<VarId>()
            }
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
            Self::Table { variables, .. }
            | Self::ForbiddenTuple { variables, .. }
            | Self::ExactlyOne { variables } => unique_scope(variables.iter().copied()),
            Self::LinearLe { terms, .. } => unique_scope(terms.iter().map(|t| t.variable)),
            Self::Clause { literals } => unique_scope(literals.iter().map(|l| l.variable)),
            Self::BoolAnd { output, inputs } | Self::BoolOr { output, inputs } => {
                unique_scope(inputs.iter().copied().chain([*output]))
            }
            Self::BoolNot { output, input } => unique_scope([*output, *input]),
            Self::ReifiedLinearLe {
                indicator, terms, ..
            } => unique_scope(terms.iter().map(|t| t.variable).chain([*indicator])),
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
            Self::ForbiddenTuple { variables, values } => {
                if variables.len() != values.len() {
                    return Err(Error::InvalidModel(
                        "forbidden tuple has wrong arity".into(),
                    ));
                }
                if unique_scope(variables.iter().copied()).len() != variables.len() {
                    return Err(Error::InvalidModel(
                        "forbidden-tuple variables must be distinct".into(),
                    ));
                }
                Ok(())
            }
            Self::LinearLe { terms, .. } | Self::ReifiedLinearLe { terms, .. } => {
                if self.scope().iter().any(|v| domains[v.0].is_empty()) {
                    return Ok(());
                }
                linear_bounds(terms, domains)?;
                if let Self::ReifiedLinearLe { indicator, .. } = self {
                    validate_booleans(&[*indicator], domains)?;
                }
                Ok(())
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
            Self::Clause { literals } => {
                if literals
                    .iter()
                    .any(|literal| !matches!(literal.value, 0 | 1))
                {
                    return Err(Error::InvalidModel(
                        "clause literals must be signed Boolean literals".into(),
                    ));
                }
                validate_booleans(
                    &literals
                        .iter()
                        .map(|literal| literal.variable)
                        .collect::<Vec<_>>(),
                    domains,
                )
            }
            Self::BoolAnd { output, inputs } | Self::BoolOr { output, inputs } => {
                validate_booleans(
                    &inputs.iter().copied().chain([*output]).collect::<Vec<_>>(),
                    domains,
                )
            }
            Self::BoolNot { output, input } => validate_booleans(&[*output, *input], domains),
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
        let status =
            match self {
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
                Self::ForbiddenTuple { variables, values } => {
                    if variables
                        .iter()
                        .zip(values)
                        .any(|(variable, value)| !domains[variable.0].contains(*value))
                    {
                        Some(true)
                    } else if variables.iter().zip(values).all(|(variable, value)| {
                        domains[variable.0].singleton_value() == Some(*value)
                    }) {
                        Some(false)
                    } else {
                        None
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
                Self::Clause { literals } => {
                    let states = literals
                        .iter()
                        .map(|literal| literal.state(domains))
                        .collect::<Result<Vec<_>>>()?;
                    if states.iter().any(|state| *state == Some(true)) {
                        Some(true)
                    } else if states.iter().all(|state| *state == Some(false)) {
                        Some(false)
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
                Self::BoolOr { output, inputs } => {
                    let any_true = inputs
                        .iter()
                        .any(|v| domains[v.0].singleton_value() == Some(1));
                    let all_false = inputs
                        .iter()
                        .all(|v| domains[v.0].singleton_value() == Some(0));
                    let out = &domains[output.0];
                    if any_true {
                        if !out.contains(1) {
                            Some(false)
                        } else if out.is_singleton() {
                            Some(true)
                        } else {
                            None
                        }
                    } else if all_false {
                        if !out.contains(0) {
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
                Self::BoolNot { output, input } => {
                    let out = &domains[output.0];
                    let input = &domains[input.0];
                    match (out.singleton_value(), input.singleton_value()) {
                        (Some(a), Some(b)) => Some(a == 1 - b),
                        (Some(a), None) if !input.contains(1 - a) => Some(false),
                        (None, Some(b)) if !out.contains(1 - b) => Some(false),
                        _ => None,
                    }
                }
                Self::ReifiedLinearLe {
                    indicator,
                    terms,
                    rhs,
                } => {
                    let (lower, upper) = linear_bounds(terms, domains)?;
                    let out = &domains[indicator.0];
                    if upper <= *rhs {
                        if !out.contains(1) {
                            Some(false)
                        } else if out.is_singleton() {
                            Some(true)
                        } else {
                            None
                        }
                    } else if lower > *rhs {
                        if !out.contains(0) {
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
            Self::ForbiddenTuple { variables, values } => Self::ForbiddenTuple {
                variables: variables.iter().map(|v| map[v.0]).collect(),
                values: values.clone(),
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
            Self::Clause { literals } => Self::Clause {
                literals: literals
                    .iter()
                    .map(|literal| Literal::new(map[literal.variable.0], literal.value))
                    .collect(),
            },
            Self::BoolAnd { output, inputs } => Self::BoolAnd {
                output: map[output.0],
                inputs: inputs.iter().map(|v| map[v.0]).collect(),
            },
            Self::BoolOr { output, inputs } => Self::BoolOr {
                output: map[output.0],
                inputs: inputs.iter().map(|v| map[v.0]).collect(),
            },
            Self::BoolNot { output, input } => Self::BoolNot {
                output: map[output.0],
                input: map[input.0],
            },
            Self::ReifiedLinearLe {
                indicator,
                terms,
                rhs,
            } => Self::ReifiedLinearLe {
                indicator: map[indicator.0],
                terms: terms
                    .iter()
                    .map(|term| LinearTerm::new(map[term.variable.0], term.coefficient))
                    .collect(),
                rhs: *rhs,
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
