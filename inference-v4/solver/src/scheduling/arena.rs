use crate::model::{Assessment, Domain, Error, Result, VarId};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ArenaLiteral {
    pub variable: VarId,
    pub value: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ArenaExpression {
    Constant(i64),
    Variable(VarId),
    Sum(Vec<(i64, ArenaExpression)>),
    Product(Vec<ArenaExpression>),
    Quotient(Box<ArenaExpression>, Box<ArenaExpression>),
    Remainder(Box<ArenaExpression>, Box<ArenaExpression>),
}

impl ArenaExpression {
    pub fn scope(&self, scope: &mut Vec<VarId>) {
        match self {
            Self::Constant(_) => {}
            Self::Variable(variable) => scope.push(*variable),
            Self::Sum(terms) => {
                for (_, expression) in terms {
                    expression.scope(scope);
                }
            }
            Self::Product(factors) => {
                for factor in factors {
                    factor.scope(scope);
                }
            }
            Self::Quotient(left, right) | Self::Remainder(left, right) => {
                left.scope(scope);
                right.scope(scope);
            }
        }
    }

    fn evaluate(&self, values: &[i64]) -> Result<i64> {
        match self {
            Self::Constant(value) => Ok(*value),
            Self::Variable(variable) => values.get(variable.0).copied().ok_or_else(|| {
                Error::InvalidModel(format!(
                    "arena expression names variable {} absent from the assignment",
                    variable.0
                ))
            }),
            Self::Sum(terms) => terms
                .iter()
                .try_fold(0i64, |sum, (coefficient, expression)| {
                    let term = expression
                        .evaluate(values)?
                        .checked_mul(*coefficient)
                        .ok_or_else(|| Error::Overflow("arena expression product".into()))?;
                    sum.checked_add(term)
                        .ok_or_else(|| Error::Overflow("arena expression sum".into()))
                }),
            Self::Product(factors) => factors.iter().try_fold(1i64, |product, factor| {
                product
                    .checked_mul(factor.evaluate(values)?)
                    .ok_or_else(|| Error::Overflow("arena expression product".into()))
            }),
            Self::Quotient(numerator, denominator) => {
                let denominator = denominator.evaluate(values)?;
                if denominator <= 0 {
                    return Err(Error::InvalidModel(
                        "arena expression has a nonpositive divisor".into(),
                    ));
                }
                Ok(numerator.evaluate(values)?.div_euclid(denominator))
            }
            Self::Remainder(numerator, denominator) => {
                let denominator = denominator.evaluate(values)?;
                if denominator <= 0 {
                    return Err(Error::InvalidModel(
                        "arena expression has a nonpositive divisor".into(),
                    ));
                }
                Ok(numerator.evaluate(values)?.rem_euclid(denominator))
            }
        }
    }

    fn minimum(&self, domains: &[Domain]) -> Result<i64> {
        match self {
            Self::Constant(value) => Ok(*value),
            Self::Variable(variable) => {
                domains
                    .get(variable.0)
                    .and_then(Domain::min)
                    .ok_or_else(|| {
                        Error::InvalidModel("arena expression has an empty variable domain".into())
                    })
            }
            Self::Sum(terms) => terms
                .iter()
                .try_fold(0i64, |sum, (coefficient, expression)| {
                    if *coefficient < 0 {
                        return Ok(0);
                    }
                    let term = expression
                        .minimum(domains)?
                        .checked_mul(*coefficient)
                        .ok_or_else(|| Error::Overflow("arena lower-bound product".into()))?;
                    sum.checked_add(term)
                        .ok_or_else(|| Error::Overflow("arena lower-bound sum".into()))
                }),
            Self::Product(factors) => factors.iter().try_fold(1i64, |product, factor| {
                product
                    .checked_mul(factor.minimum(domains)?)
                    .ok_or_else(|| Error::Overflow("arena lower-bound product".into()))
            }),
            Self::Quotient(_, _) | Self::Remainder(_, _) => Ok(0),
        }
    }

    fn remap(&self, variables: &[VarId]) -> Self {
        match self {
            Self::Constant(value) => Self::Constant(*value),
            Self::Variable(variable) => Self::Variable(variables[variable.0]),
            Self::Sum(terms) => Self::Sum(
                terms
                    .iter()
                    .map(|(coefficient, expression)| (*coefficient, expression.remap(variables)))
                    .collect(),
            ),
            Self::Product(factors) => Self::Product(
                factors
                    .iter()
                    .map(|factor| factor.remap(variables))
                    .collect(),
            ),
            Self::Quotient(left, right) => Self::Quotient(
                Box::new(left.remap(variables)),
                Box::new(right.remap(variables)),
            ),
            Self::Remainder(left, right) => Self::Remainder(
                Box::new(left.remap(variables)),
                Box::new(right.remap(variables)),
            ),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ArenaItem {
    pub ordinal: u32,
    /// Disjunctive normal form. The item is active when any clause holds.
    pub activation: Vec<Vec<ArenaLiteral>>,
    pub bytes: ArenaExpression,
    pub alignment: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ArenaPacking {
    pub capacity: u64,
    pub items: Vec<ArenaItem>,
    pub interference: Vec<(u32, u32)>,
}

impl ArenaPacking {
    pub fn scope(&self) -> Vec<VarId> {
        let mut scope = Vec::new();
        for item in &self.items {
            for clause in &item.activation {
                scope.extend(clause.iter().map(|literal| literal.variable));
            }
            item.bytes.scope(&mut scope);
        }
        scope.sort_unstable();
        scope.dedup();
        scope
    }

    pub fn remap(&self, variables: &[VarId]) -> Self {
        Self {
            capacity: self.capacity,
            items: self
                .items
                .iter()
                .map(|item| ArenaItem {
                    ordinal: item.ordinal,
                    activation: item
                        .activation
                        .iter()
                        .map(|clause| {
                            clause
                                .iter()
                                .map(|literal| ArenaLiteral {
                                    variable: variables[literal.variable.0],
                                    value: literal.value,
                                })
                                .collect()
                        })
                        .collect(),
                    bytes: item.bytes.remap(variables),
                    alignment: item.alignment,
                })
                .collect(),
            interference: self.interference.clone(),
        }
    }

    pub fn validate(&self, domains: &[Domain]) -> Result<()> {
        let ordinals = self
            .items
            .iter()
            .map(|item| item.ordinal)
            .collect::<BTreeSet<_>>();
        if ordinals.len() != self.items.len() {
            return Err(Error::InvalidModel(
                "arena item ordinals must be unique".into(),
            ));
        }
        for item in &self.items {
            if item.alignment == 0 {
                return Err(Error::InvalidModel(
                    "arena item alignment must be nonzero".into(),
                ));
            }
            if item.activation.is_empty() {
                return Err(Error::InvalidModel(
                    "arena item activation DNF must contain a clause".into(),
                ));
            }
            for clause in &item.activation {
                for literal in clause {
                    let domain = domains.get(literal.variable.0).ok_or_else(|| {
                        Error::InvalidModel(format!(
                            "arena activation names unknown variable {}",
                            literal.variable.0
                        ))
                    })?;
                    let _ = domain;
                }
            }
        }
        let mut edges = BTreeSet::new();
        for &(left, right) in &self.interference {
            if left == right || !ordinals.contains(&left) || !ordinals.contains(&right) {
                return Err(Error::InvalidModel(
                    "arena interference edge is invalid".into(),
                ));
            }
            if !edges.insert((left.min(right), left.max(right))) {
                return Err(Error::InvalidModel(
                    "arena interference edge is duplicated".into(),
                ));
            }
        }
        Ok(())
    }

    pub fn assess(&self, domains: &[Domain]) -> Result<Assessment> {
        self.validate(domains)?;
        if self
            .scope()
            .iter()
            .all(|variable| domains[variable.0].is_singleton())
        {
            let values = domains
                .iter()
                .map(|domain| domain.singleton_value().unwrap_or(0))
                .collect::<Vec<_>>();
            return Ok(match arena_offsets(self, &values)? {
                Some(_) => Assessment::exact(0),
                None => Assessment::infeasible(),
            });
        }
        if self.clique_lower_bound_exceeds_capacity(domains)? {
            Ok(Assessment::infeasible())
        } else {
            Ok(Assessment::bounded(0))
        }
    }

    fn clique_lower_bound_exceeds_capacity(&self, domains: &[Domain]) -> Result<bool> {
        let active = self
            .items
            .iter()
            .filter(|item| {
                item.activation.iter().any(|clause| {
                    clause.iter().all(|literal| {
                        domains[literal.variable.0].singleton_value() == Some(literal.value)
                    })
                })
            })
            .collect::<Vec<_>>();
        let edges = self
            .interference
            .iter()
            .map(|&(left, right)| (left.min(right), left.max(right)))
            .collect::<BTreeSet<_>>();
        for (index, item) in active.iter().enumerate() {
            let mut clique = vec![*item];
            for candidate in active.iter().skip(index + 1) {
                if clique.iter().all(|member| {
                    edges.contains(&(
                        member.ordinal.min(candidate.ordinal),
                        member.ordinal.max(candidate.ordinal),
                    ))
                }) {
                    clique.push(*candidate);
                }
            }
            let bytes = clique.iter().try_fold(0u64, |sum, member| {
                let minimum = member.bytes.minimum(domains)?;
                let minimum = u64::try_from(minimum).map_err(|_| {
                    Error::InvalidModel("arena item has a negative minimum size".into())
                })?;
                sum.checked_add(minimum)
                    .ok_or_else(|| Error::Overflow("arena clique lower bound".into()))
            })?;
            if bytes > self.capacity {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

#[derive(Clone, Copy)]
struct ActiveItem {
    ordinal: u32,
    bytes: u64,
    alignment: u64,
}

fn align_up(value: u64, alignment: u64) -> Option<u64> {
    value
        .checked_add(alignment - 1)
        .and_then(|value| (value / alignment).checked_mul(alignment))
}

/// Deterministic exact certificate constructor shared by solver evaluation and
/// plan decoding. `None` is a proof that no placement fits the capacity.
pub fn arena_offsets(packing: &ArenaPacking, values: &[i64]) -> Result<Option<BTreeMap<u32, u64>>> {
    let active = |item: &ArenaItem| {
        item.activation.iter().any(|clause| {
            clause
                .iter()
                .all(|literal| values.get(literal.variable.0).copied() == Some(literal.value))
        })
    };
    let edges = packing
        .interference
        .iter()
        .map(|&(left, right)| (left.min(right), left.max(right)))
        .collect::<BTreeSet<_>>();
    let mut items = Vec::new();
    for item in &packing.items {
        if active(item) {
            let bytes = u64::try_from(item.bytes.evaluate(values)?)
                .map_err(|_| Error::InvalidModel("arena item size is negative".into()))?;
            if bytes > packing.capacity {
                return Ok(None);
            }
            items.push(ActiveItem {
                ordinal: item.ordinal,
                bytes,
                alignment: item.alignment,
            });
        }
    }
    items.sort_by_key(|item| {
        let degree = edges
            .iter()
            .filter(|(left, right)| *left == item.ordinal || *right == item.ordinal)
            .count();
        (
            std::cmp::Reverse(degree),
            std::cmp::Reverse(item.bytes),
            item.ordinal,
        )
    });

    fn candidates(
        item: ActiveItem,
        placed: &BTreeMap<u32, (u64, u64)>,
        edges: &BTreeSet<(u32, u32)>,
        capacity: u64,
    ) -> Vec<u64> {
        let mut candidates = vec![0];
        for (&ordinal, &(offset, bytes)) in placed {
            if edges.contains(&(ordinal.min(item.ordinal), ordinal.max(item.ordinal))) {
                if let Some(end) = offset.checked_add(bytes) {
                    candidates.push(end);
                }
            }
        }
        let mut candidates = candidates
            .into_iter()
            .filter_map(|offset| align_up(offset, item.alignment))
            .filter(|offset| {
                offset
                    .checked_add(item.bytes)
                    .is_some_and(|end| end <= capacity)
            })
            .collect::<Vec<_>>();
        candidates.sort_unstable();
        candidates.dedup();
        candidates.retain(|offset| {
            let end = *offset + item.bytes;
            placed
                .iter()
                .all(|(&ordinal, &(other_offset, other_bytes))| {
                    !edges.contains(&(ordinal.min(item.ordinal), ordinal.max(item.ordinal)))
                        || end <= other_offset
                        || other_offset + other_bytes <= *offset
                })
        });
        candidates
    }

    fn search(
        items: &[ActiveItem],
        index: usize,
        placed: &mut BTreeMap<u32, (u64, u64)>,
        edges: &BTreeSet<(u32, u32)>,
        capacity: u64,
        first_only: bool,
    ) -> bool {
        if index == items.len() {
            return true;
        }
        let item = items[index];
        let candidates = candidates(item, placed, edges, capacity);
        for offset in candidates {
            placed.insert(item.ordinal, (offset, item.bytes));
            if search(items, index + 1, placed, edges, capacity, first_only) {
                return true;
            }
            placed.remove(&item.ordinal);
            if first_only {
                break;
            }
        }
        false
    }

    let mut placed = BTreeMap::new();
    if !search(&items, 0, &mut placed, &edges, packing.capacity, true) {
        placed.clear();
        if !search(&items, 0, &mut placed, &edges, packing.capacity, false) {
            return Ok(None);
        }
    }
    let mut offsets = packing
        .items
        .iter()
        .map(|item| (item.ordinal, 0))
        .collect::<BTreeMap<_, _>>();
    for (ordinal, (offset, _)) in placed {
        offsets.insert(ordinal, offset);
    }
    Ok(Some(offsets))
}
