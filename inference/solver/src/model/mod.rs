//! Domain-independent, inspectable finite optimization models.
mod constraint;
mod domain;
mod fragment;
mod objective;
mod symbolic;

pub use constraint::Constraint;
pub use domain::{Domain, DomainValues};
pub use fragment::{Fragment, FragmentInstance, FragmentInterface};
pub use objective::Cost;
pub use symbolic::Arithmetic;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidModel(String),
    Overflow(String),
    InvalidAssignment(String),
    Invariant(String),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidModel(s) => write!(f, "invalid model: {s}"),
            Self::Overflow(s) => write!(f, "arithmetic overflow: {s}"),
            Self::InvalidAssignment(s) => write!(f, "invalid assignment: {s}"),
            Self::Invariant(s) => write!(f, "solver invariant: {s}"),
        }
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VarId(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FactorId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Literal {
    pub variable: VarId,
    pub value: i64,
}
impl Literal {
    pub const fn new(variable: VarId, value: i64) -> Self {
        Self { variable, value }
    }
    pub(crate) fn state(&self, domains: &[Domain]) -> Result<Option<bool>> {
        let d = get_domain(domains, self.variable)?;
        Ok(if !d.contains(self.value) {
            Some(false)
        } else if d.is_singleton() {
            Some(true)
        } else {
            None
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LinearTerm {
    pub variable: VarId,
    pub coefficient: i64,
}
impl LinearTerm {
    pub const fn new(variable: VarId, coefficient: i64) -> Self {
        Self {
            variable,
            coefficient,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Variable {
    pub name: String,
    pub domain: Domain,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ObligationKind {
    /// Supply the missing finite mathematical construction.
    Construction,
    /// Supply a missing typed analysis/model relation; more search alone cannot.
    Analysis,
    /// Extend the represented refinement repertoire; larger budgets alone cannot.
    Refinement,
}
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Obligation {
    pub kind: ObligationKind,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FactorKind {
    Constraint(Constraint),
    Cost(Cost),
    /// Complete immutable factor definition. Search may materialize its children
    /// incrementally, but its entire scope exists before any decomposition.
    Fragment {
        factors: Vec<Factor>,
    },
    /// A declared gap in construction or analysis; it is never a feasible witness.
    Unresolved {
        kind: ObligationKind,
        reason: String,
    },
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Factor {
    pub guards: Vec<Literal>,
    pub kind: FactorKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Assessment {
    pub infeasible: bool,
    pub lower_bound: u64,
    /// Present only when the factor is completely discharged on these domains.
    pub exact_cost: Option<u64>,
    pub unresolved: Option<Obligation>,
}
impl Assessment {
    pub fn infeasible() -> Self {
        Self {
            infeasible: true,
            lower_bound: 0,
            exact_cost: None,
            unresolved: None,
        }
    }
    pub fn exact(cost: u64) -> Self {
        Self {
            infeasible: false,
            lower_bound: cost,
            exact_cost: Some(cost),
            unresolved: None,
        }
    }
    pub fn bounded(lower_bound: u64) -> Self {
        Self {
            infeasible: false,
            lower_bound,
            exact_cost: None,
            unresolved: None,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Propagation {
    pub changed: bool,
    pub infeasible: bool,
}

impl Factor {
    /// Retained owned bytes, including immutable nested definitions. Allocator
    /// bookkeeping and shared process/runtime allocations are not included.
    pub fn retained_bytes(&self) -> usize {
        let mut bytes = std::mem::size_of::<Self>();
        let mut pending = vec![self];
        while let Some(factor) = pending.pop() {
            bytes += factor.guards.capacity() * std::mem::size_of::<Literal>();
            match &factor.kind {
                FactorKind::Fragment { factors } => {
                    bytes += factors.capacity() * std::mem::size_of::<Factor>();
                    pending.extend(factors);
                }
                FactorKind::Constraint(c) => bytes += c.heap_bytes(),
                FactorKind::Cost(c) => bytes += c.heap_bytes(),
                FactorKind::Unresolved { reason, .. } => bytes += reason.capacity(),
            }
        }
        bytes
    }
    pub fn scope(&self) -> Vec<VarId> {
        let mut scope = Vec::new();
        let mut pending = vec![self];
        while let Some(factor) = pending.pop() {
            scope.extend(factor.guards.iter().map(|g| g.variable));
            match &factor.kind {
                FactorKind::Constraint(c) => scope.extend(c.scope()),
                FactorKind::Cost(c) => scope.extend(c.scope()),
                FactorKind::Fragment { factors } => pending.extend(factors),
                FactorKind::Unresolved { .. } => (),
            }
        }
        scope.sort_unstable();
        scope.dedup();
        scope
    }
    pub fn assess(&self, domains: &[Domain]) -> Result<Assessment> {
        self.validate(domains)?;
        self.assess_validated(domains)
    }
    pub(crate) fn assess_validated(&self, domains: &[Domain]) -> Result<Assessment> {
        for var in self.scope() {
            if get_domain(domains, var)?.is_empty() {
                return Ok(Assessment::infeasible());
            }
        }
        let mut pending = vec![self];
        let mut lower_bound = 0_u64;
        let mut exact = true;
        let mut unresolved = None;
        while let Some(factor) = pending.pop() {
            match factor.guard_state(domains)? {
                Some(false) => continue,
                None => {
                    exact = false;
                    continue;
                }
                Some(true) => (),
            }
            let assessment = match &factor.kind {
                FactorKind::Constraint(c) => c.assess(domains)?,
                FactorKind::Cost(c) => c.assess(domains)?,
                FactorKind::Fragment { factors } => {
                    pending.extend(factors);
                    continue;
                }
                FactorKind::Unresolved { kind, reason } => Assessment {
                    unresolved: Some(Obligation {
                        kind: *kind,
                        reason: reason.clone(),
                    }),
                    ..Assessment::bounded(0)
                },
            };
            if assessment.infeasible {
                return Ok(assessment);
            }
            lower_bound = lower_bound
                .checked_add(assessment.lower_bound)
                .ok_or_else(|| Error::Overflow("fragment objective sum".into()))?;
            exact &= assessment.exact_cost.is_some();
            if assessment.unresolved.is_some() {
                unresolved = assessment.unresolved;
            }
        }
        Ok(Assessment {
            infeasible: false,
            lower_bound,
            exact_cost: exact.then_some(lower_bound),
            unresolved,
        })
    }
    pub fn guard_state(&self, domains: &[Domain]) -> Result<Option<bool>> {
        let mut active = true;
        let mut guard_assignments = BTreeMap::new();
        for guard in &self.guards {
            if guard_assignments
                .insert(guard.variable, guard.value)
                .is_some_and(|value| value != guard.value)
            {
                return Ok(Some(false));
            }
            match guard.state(domains)? {
                Some(false) => return Ok(Some(false)),
                None => active = false,
                Some(true) => (),
            }
        }
        Ok(active.then_some(true))
    }
    pub(crate) fn validate(&self, domains: &[Domain]) -> Result<()> {
        let mut pending = vec![(self, Vec::<Literal>::new(), 0_usize)];
        while let Some((factor, mut inherited, depth)) = pending.pop() {
            if depth > 128 {
                return Err(Error::InvalidModel(
                    "fragment nesting exceeds 128 levels".into(),
                ));
            }
            for guard in &factor.guards {
                get_domain(domains, guard.variable)?;
            }
            inherited.extend_from_slice(&factor.guards);
            match &factor.kind {
                FactorKind::Constraint(c) => c.validate(domains)?,
                FactorKind::Cost(c) => c.validate(domains, &inherited)?,
                FactorKind::Fragment { factors } => {
                    pending.extend(
                        factors
                            .iter()
                            .map(|child| (child, inherited.clone(), depth + 1)),
                    );
                }
                FactorKind::Unresolved { reason, .. } if reason.is_empty() => {
                    return Err(Error::InvalidModel(
                        "unresolved coverage requires a reason".into(),
                    ))
                }
                FactorKind::Unresolved { .. } => (),
            }
        }
        Ok(())
    }
    pub(crate) fn remap(&self, map: &[VarId]) -> Self {
        let guards = self
            .guards
            .iter()
            .map(|g| Literal::new(map[g.variable.0], g.value))
            .collect();
        let kind = match &self.kind {
            FactorKind::Constraint(c) => FactorKind::Constraint(c.remap(map)),
            FactorKind::Cost(c) => FactorKind::Cost(c.remap(map)),
            FactorKind::Fragment { factors } => FactorKind::Fragment {
                factors: factors.iter().map(|f| f.remap(map)).collect(),
            },
            FactorKind::Unresolved { kind, reason } => FactorKind::Unresolved {
                kind: *kind,
                reason: reason.clone(),
            },
        };
        Self { guards, kind }
    }
}

/// Immutable model. Exact equality, not a hash alone, determines model identity.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Model {
    variables: Vec<Variable>,
    factors: Vec<Factor>,
    units: String,
}
#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Model {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Representation {
            variables: Vec<Variable>,
            factors: Vec<Factor>,
            units: String,
        }
        let value = Representation::deserialize(deserializer)?;
        let model = Model {
            variables: value.variables,
            factors: value.factors,
            units: value.units,
        };
        model.validate().map_err(serde::de::Error::custom)?;
        Ok(model)
    }
}
impl Model {
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.units.capacity()
            + self.variables.capacity() * std::mem::size_of::<Variable>()
            + self
                .variables
                .iter()
                .map(|v| {
                    v.name.capacity() + v.domain.retained_bytes() - std::mem::size_of::<Domain>()
                })
                .sum::<usize>()
            + self.factors.capacity() * std::mem::size_of::<Factor>()
            + self
                .factors
                .iter()
                .map(|f| f.retained_bytes() - std::mem::size_of::<Factor>())
                .sum::<usize>()
    }
    pub fn variables(&self) -> &[Variable] {
        &self.variables
    }
    pub fn factors(&self) -> &[Factor] {
        &self.factors
    }
    pub fn units(&self) -> &str {
        &self.units
    }
    pub fn domains(&self) -> Vec<Domain> {
        self.variables.iter().map(|v| v.domain.clone()).collect()
    }
    pub fn validate(&self) -> Result<()> {
        if self.units.is_empty() {
            return Err(Error::InvalidModel(
                "objective units cannot be empty".into(),
            ));
        }
        for variable in &self.variables {
            variable.domain.validate()?;
        }
        let domains = self.domains();
        for factor in &self.factors {
            factor.validate(&domains)?;
        }
        Ok(())
    }
    /// Checks a full assignment against original domains, every guard and every factor.
    /// Coverage gaps return `unresolved`; they never validate an upper witness.
    pub fn validate_assignment(&self, values: &[i64]) -> Result<Assessment> {
        if values.len() != self.variables.len() {
            return Err(Error::InvalidAssignment(format!(
                "expected {} values, received {}",
                self.variables.len(),
                values.len()
            )));
        }
        for (var, value) in self.variables.iter().zip(values) {
            if !var.domain.contains(*value) {
                return Err(Error::InvalidAssignment(format!(
                    "{} does not contain {value}",
                    var.name
                )));
            }
        }
        let domains: Vec<_> = values.iter().map(|v| Domain::singleton(*v)).collect();
        let mut total = 0_u64;
        let mut unresolved = None;
        for factor in &self.factors {
            let result = factor.assess_validated(&domains)?;
            if result.infeasible {
                return Ok(Assessment::infeasible());
            }
            total = total
                .checked_add(result.lower_bound)
                .ok_or_else(|| Error::Overflow("objective sum".into()))?;
            if let Some(reason) = result.unresolved {
                unresolved = Some(reason);
            } else if result.exact_cost.is_none() {
                return Err(Error::Invariant(
                    "fixed supported factor has no exact assessment".into(),
                ));
            }
        }
        Ok(Assessment {
            infeasible: false,
            lower_bound: total,
            exact_cost: unresolved.is_none().then_some(total),
            unresolved,
        })
    }
}

#[derive(Clone, Debug)]
pub struct ModelBuilder {
    pub(crate) variables: Vec<Variable>,
    pub(crate) factors: Vec<Factor>,
    pub(crate) guards: Vec<Literal>,
    units: String,
}
impl Default for ModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}
impl ModelBuilder {
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            factors: Vec::new(),
            guards: Vec::new(),
            units: "cost".into(),
        }
    }
    pub fn units(&mut self, units: impl Into<String>) -> &mut Self {
        self.units = units.into();
        self
    }
    pub fn variable(&mut self, name: impl Into<String>, domain: Domain) -> VarId {
        let id = VarId(self.variables.len());
        self.variables.push(Variable {
            name: name.into(),
            domain,
        });
        id
    }
    pub fn constraint(&mut self, constraint: Constraint) -> FactorId {
        self.guarded_constraint(Vec::new(), constraint)
    }
    pub fn cost(&mut self, cost: Cost) -> FactorId {
        self.guarded_cost(Vec::new(), cost)
    }
    pub fn guarded_constraint(&mut self, guards: Vec<Literal>, constraint: Constraint) -> FactorId {
        self.factor(guards, FactorKind::Constraint(constraint))
    }
    pub fn guarded_cost(&mut self, guards: Vec<Literal>, cost: Cost) -> FactorId {
        self.factor(guards, FactorKind::Cost(cost))
    }
    pub fn unresolved(&mut self, guards: Vec<Literal>, reason: impl Into<String>) -> FactorId {
        self.obligation(guards, ObligationKind::Construction, reason)
    }
    pub fn obligation(
        &mut self,
        guards: Vec<Literal>,
        kind: ObligationKind,
        reason: impl Into<String>,
    ) -> FactorId {
        self.factor(
            guards,
            FactorKind::Unresolved {
                kind,
                reason: reason.into(),
            },
        )
    }
    pub fn factor(&mut self, mut guards: Vec<Literal>, kind: FactorKind) -> FactorId {
        guards.extend_from_slice(&self.guards);
        guards.sort_by_key(|g| (g.variable, g.value));
        guards.dedup();
        let id = FactorId(self.factors.len());
        self.factors.push(Factor { guards, kind });
        id
    }
    /// Lexically guard all factors added by `build`. Inactive private variables should
    /// be introduced with `local_variable`, which gives them one canonical value.
    pub fn when<T>(&mut self, guard: Literal, build: impl FnOnce(&mut Self) -> T) -> T {
        self.guards.push(guard);
        let result = build(self);
        self.guards.pop();
        result
    }
    pub fn local_variable(&mut self, name: impl Into<String>, domain: Domain) -> Result<VarId> {
        let inactive = domain
            .min()
            .ok_or_else(|| Error::InvalidModel("fragment local has empty domain".into()))?;
        let id = self.variable(name, domain);
        for guard in self.guards.clone() {
            // !guard => local == inactive, encoded as local!=inactive => guard.
            // Each possible non-inactive local value is not enumerated: a guarded
            // table on guard/local would be huge, so use a dedicated implication.
            let guards = self.guards.clone();
            self.guards.clear();
            self.factor(
                Vec::new(),
                FactorKind::Constraint(Constraint::InactiveValue {
                    active: guard,
                    variable: id,
                    inactive,
                }),
            );
            self.guards = guards;
        }
        Ok(id)
    }
    pub fn build(self) -> Result<Model> {
        let model = Model {
            variables: self.variables,
            factors: self.factors,
            units: self.units,
        };
        model.validate()?;
        Ok(model)
    }
}

pub(crate) fn get_domain(domains: &[Domain], var: VarId) -> Result<&Domain> {
    domains
        .get(var.0)
        .ok_or_else(|| Error::InvalidModel(format!("variable {} is not in the model", var.0)))
}
pub(crate) fn unique_scope(vars: impl IntoIterator<Item = VarId>) -> Vec<VarId> {
    vars.into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}
pub(crate) fn merged_terms(terms: &[LinearTerm]) -> Result<Vec<(VarId, i128)>> {
    let mut result = BTreeMap::<VarId, i128>::new();
    for term in terms {
        let current = result.entry(term.variable).or_default();
        *current = current
            .checked_add(term.coefficient as i128)
            .ok_or_else(|| Error::Overflow("affine coefficient sum".into()))?;
    }
    Ok(result
        .into_iter()
        .filter(|(_, coefficient)| *coefficient != 0)
        .collect())
}
pub(crate) fn linear_bounds(terms: &[LinearTerm], domains: &[Domain]) -> Result<(i128, i128)> {
    let mut lower = 0_i128;
    let mut upper = 0_i128;
    for (var, coefficient) in merged_terms(terms)? {
        let domain = get_domain(domains, var)?;
        let min = domain
            .min()
            .ok_or_else(|| Error::InvalidAssignment("empty affine domain".into()))?
            as i128;
        let max = domain.max().expect("nonempty") as i128;
        let a = coefficient
            .checked_mul(min)
            .ok_or_else(|| Error::Overflow("affine multiplication".into()))?;
        let b = coefficient
            .checked_mul(max)
            .ok_or_else(|| Error::Overflow("affine multiplication".into()))?;
        lower = lower
            .checked_add(a.min(b))
            .ok_or_else(|| Error::Overflow("affine lower sum".into()))?;
        upper = upper
            .checked_add(a.max(b))
            .ok_or_else(|| Error::Overflow("affine upper sum".into()))?;
    }
    Ok((lower, upper))
}
pub(crate) fn validate_table(
    variables: &[VarId],
    tuples: impl Iterator<Item = Vec<i64>>,
    domains: &[Domain],
) -> Result<()> {
    if unique_scope(variables.iter().copied()).len() != variables.len() {
        return Err(Error::InvalidModel(
            "table variables must be distinct".into(),
        ));
    }
    for var in variables {
        get_domain(domains, *var)?;
    }
    let mut seen = BTreeSet::new();
    for tuple in tuples {
        if tuple.len() != variables.len() {
            return Err(Error::InvalidModel("table tuple has wrong arity".into()));
        }
        if !seen.insert(tuple) {
            return Err(Error::InvalidModel("duplicate table tuple".into()));
        }
    }
    Ok(())
}
pub(crate) fn matching_tuple(variables: &[VarId], tuple: &[i64], domains: &[Domain]) -> bool {
    variables
        .iter()
        .zip(tuple)
        .all(|(v, value)| domains[v.0].contains(*value))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_partial_cost_tables() {
        let mut b = ModelBuilder::new();
        let x = b.variable("x", Domain::boolean());
        b.cost(Cost::Table {
            variables: vec![x],
            entries: vec![(vec![0], 3)],
        });
        assert!(matches!(b.build(), Err(Error::InvalidModel(_))));
    }
    #[test]
    fn guarded_costs_need_only_cover_active_assignments() {
        let mut b = ModelBuilder::new();
        let x = b.variable("x", Domain::boolean());
        b.guarded_cost(
            vec![Literal::new(x, 1)],
            Cost::Table {
                variables: vec![x],
                entries: vec![(vec![1], 7)],
            },
        );
        let m = b.build().unwrap();
        assert_eq!(m.validate_assignment(&[0]).unwrap().exact_cost, Some(0));
        assert_eq!(m.validate_assignment(&[1]).unwrap().exact_cost, Some(7));
    }
    #[test]
    fn repeated_affine_terms_are_combined_before_bounding() {
        let mut b = ModelBuilder::new();
        let x = b.variable("x", Domain::interval(i64::MIN, i64::MAX).unwrap());
        b.cost(Cost::Linear {
            constant: 3,
            terms: vec![LinearTerm::new(x, i64::MAX), LinearTerm::new(x, -i64::MAX)],
        });
        let m = b.build().unwrap();
        assert_eq!(
            m.factors()[0].assess(&m.domains()).unwrap().exact_cost,
            Some(3)
        );
    }
    #[test]
    fn objective_overflow_is_an_error_not_a_bound() {
        let mut b = ModelBuilder::new();
        b.cost(Cost::Constant(u64::MAX));
        b.cost(Cost::Constant(1));
        let m = b.build().unwrap();
        assert!(matches!(
            m.validate_assignment(&[]),
            Err(Error::Overflow(_))
        ));
    }
    #[test]
    fn unsupported_coverage_cannot_validate_witness() {
        let mut b = ModelBuilder::new();
        b.unresolved(vec![], "missing typed construction");
        let m = b.build().unwrap();
        let a = m.validate_assignment(&[]).unwrap();
        assert!(!a.infeasible);
        assert_eq!(a.exact_cost, None);
        assert!(a.unresolved.is_some());
    }
    #[test]
    fn domains_and_constraint_scopes_are_validated() {
        let mut b = ModelBuilder::new();
        b.constraint(Constraint::Equal {
            left: VarId(0),
            right: VarId(0),
        });
        assert!(matches!(b.build(), Err(Error::InvalidModel(_))));
    }
    #[test]
    fn direct_assessment_rejects_malformed_public_factor_values() {
        let factor = Factor {
            guards: vec![],
            kind: FactorKind::Constraint(Constraint::Table {
                variables: vec![VarId(0)],
                tuples: vec![vec![]],
            }),
        };
        assert!(matches!(
            factor.assess(&[Domain::boolean()]),
            Err(Error::InvalidModel(_))
        ));
        let factor = Factor {
            guards: vec![],
            kind: FactorKind::Cost(Cost::Table {
                variables: vec![VarId(2)],
                entries: vec![(vec![0], 1)],
            }),
        };
        assert!(matches!(factor.assess(&[]), Err(Error::InvalidModel(_))));
    }
}
