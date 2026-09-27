use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Cost {
    Constant(u64),
    Linear {
        constant: i128,
        terms: Vec<LinearTerm>,
    },
    /// A total function on the active Cartesian domain, checked during validation.
    Table {
        variables: Vec<VarId>,
        entries: Vec<(Vec<i64>, u64)>,
    },
}
impl Cost {
    pub(crate) fn heap_bytes(&self) -> usize {
        match self {
            Self::Constant(_) => 0,
            Self::Linear { terms, .. } => terms.capacity() * std::mem::size_of::<LinearTerm>(),
            Self::Table { variables, entries } => {
                variables.capacity() * std::mem::size_of::<VarId>()
                    + entries.capacity() * std::mem::size_of::<(Vec<i64>, u64)>()
                    + entries
                        .iter()
                        .map(|(row, _)| row.capacity() * std::mem::size_of::<i64>())
                        .sum::<usize>()
            }
        }
    }
    pub fn scope(&self) -> Vec<VarId> {
        match self {
            Self::Constant(_) => Vec::new(),
            Self::Linear { terms, .. } => unique_scope(terms.iter().map(|t| t.variable)),
            Self::Table { variables, .. } => unique_scope(variables.iter().copied()),
        }
    }
    pub fn validate(&self, domains: &[Domain], guards: &[Literal]) -> Result<()> {
        for variable in self.scope() {
            get_domain(domains, variable)?;
        }
        if let Self::Table { variables, entries } = self {
            validate_table(
                variables,
                entries.iter().map(|(tuple, _)| tuple.clone()),
                domains,
            )?;
        }
        let mut active_domains = domains.to_vec();
        for guard in guards {
            let original = get_domain(&active_domains, guard.variable)?;
            active_domains[guard.variable.0] =
                original.intersect(&Domain::singleton(guard.value))?;
        }
        // An impossible conjunction never evaluates the body.
        if guards
            .iter()
            .any(|g| active_domains[g.variable.0].is_empty())
        {
            return Ok(());
        }
        if self.scope().iter().any(|v| active_domains[v.0].is_empty()) {
            return Ok(());
        }
        match self {
            Self::Constant(_) => Ok(()),
            Self::Linear { constant, terms } => {
                let (min, max) = linear_bounds(terms, &active_domains)?;
                let min = min
                    .checked_add(*constant)
                    .ok_or_else(|| Error::Overflow("affine objective lower".into()))?;
                let max = max
                    .checked_add(*constant)
                    .ok_or_else(|| Error::Overflow("affine objective upper".into()))?;
                if min < 0 {
                    return Err(Error::InvalidModel(
                        "objective terms must be nonnegative over their active domains".into(),
                    ));
                }
                u64::try_from(max)
                    .map_err(|_| Error::Overflow("objective term exceeds u64".into()))?;
                Ok(())
            }
            Self::Table { variables, entries } => {
                let matching = entries
                    .iter()
                    .filter(|(tuple, _)| matching_tuple(variables, tuple, &active_domains))
                    .count() as u128;
                let needed = variables.iter().try_fold(1_u128, |n, v| {
                    n.checked_mul(active_domains[v.0].cardinality())
                });
                if needed != Some(matching) {
                    return Err(Error::InvalidModel(
                        "cost table does not cover its complete active domain".into(),
                    ));
                }
                Ok(())
            }
        }
    }
    pub(crate) fn assess(&self, domains: &[Domain]) -> Result<Assessment> {
        match self {
            Self::Constant(value) => Ok(Assessment::exact(*value)),
            Self::Linear { constant, terms } => {
                let (min, max) = linear_bounds(terms, domains)?;
                let min = min
                    .checked_add(*constant)
                    .ok_or_else(|| Error::Overflow("affine objective lower".into()))?;
                let max = max
                    .checked_add(*constant)
                    .ok_or_else(|| Error::Overflow("affine objective upper".into()))?;
                let min = u64::try_from(min)
                    .map_err(|_| Error::Overflow("affine cost not in u64".into()))?;
                let max = u64::try_from(max)
                    .map_err(|_| Error::Overflow("affine cost not in u64".into()))?;
                Ok(if min == max {
                    Assessment::exact(min)
                } else {
                    Assessment::bounded(min)
                })
            }
            Self::Table { variables, entries } => {
                let mut min = None::<u64>;
                let mut max = 0;
                for (tuple, cost) in entries {
                    if matching_tuple(variables, tuple, domains) {
                        min = Some(min.map_or(*cost, |value| value.min(*cost)));
                        max = max.max(*cost);
                    }
                }
                let min = min.ok_or_else(|| {
                    Error::InvalidAssignment("active cost table has no matching row".into())
                })?;
                Ok(if min == max {
                    Assessment::exact(min)
                } else {
                    Assessment::bounded(min)
                })
            }
        }
    }
    pub(crate) fn remap(&self, map: &[VarId]) -> Self {
        match self {
            Self::Constant(cost) => Self::Constant(*cost),
            Self::Linear { constant, terms } => Self::Linear {
                constant: *constant,
                terms: terms
                    .iter()
                    .map(|t| LinearTerm::new(map[t.variable.0], t.coefficient))
                    .collect(),
            },
            Self::Table { variables, entries } => Self::Table {
                variables: variables.iter().map(|v| map[v.0]).collect(),
                entries: entries.clone(),
            },
        }
    }
}
