//! One global solver model and one budgeted solve (package P1).
//!
//! `plan` exports the sealed plan space's decisions and exact consequence
//! facts into one `magnitude_solver` model, decides feasibility first
//! (independently of the optimization budget), and resolves exactly one
//! complete assignment through `SolverModelView::validate` — the only
//! `CompleteAssignment` constructor. There is no greedy selector, no retry,
//! and no fallback search strategy; a budget may stop the proof of
//! optimality but can never produce a partial assignment.
//!
//! # Resource-expression identity
//!
//! Every `Sym` a strategy exports is materialized exactly once:
//! - tuning atoms are qualified by their `(occurrence, strategy)` identity
//!   (`{occurrence}.{strategy}.{name}`) and become model variables over
//!   their declared domains;
//! - `@runtime<N>` atoms fold to constants — the checked **capacity** for
//!   feasibility and resource constraints, the workload's **expected**
//!   extent for cost (never semantics);
//! - `@leaf<L>` atoms become per-strategy variables over the leaf's proved
//!   interval (an index/range bound, else the unsigned 32-bit range).
//!
//! Resolution evaluates the identical expressions through the identical
//! qualification (`seismic_realization::formation::seal::qualify`; the
//! qualification function is duplicated here because it crosses a crate
//! boundary — both copies are one identity): one expression, one value.

use magnitude_solver::{
    model::{Arithmetic, Constraint, Cost, Domain, LinearTerm, Literal},
    scheduling::{ArenaExpression, ArenaItem, ArenaLiteral, ArenaPacking, SchedulingConstraint},
    Algorithm, Budget as SolverBudget, Budgeted, FeasibleSolution, Limits as SolverLimits,
    Model, ModelBuilder, Options, Search, VarId,
};
use seismic_lang::{
    logical::{EffectiveTargetIdentity, IdIndex},
    sym::{Atom, Sym},
    types::{DType, ExtentExpr, RuntimeExtentId},
};
use seismic_realization::{
    consequences::SolverConstraint,
    failure::{CompilerDefect, Package},
    ids::{CanonicalLeafId, ChoiceVarId, OccurrenceId, ResidenceId, StrategyId},
    kernel::ExecutableDialect,
    numerics::{self, PolicyDecision},
    occurrence::OccurrenceFacts,
    plan_space::{
        CompleteAssignment, Decision, DecisionVariables, NumericalContext, PlanSpace,
        SolverModelView,
    },
    residence::{Lifetime, ResidenceChoice, StorageScope},
    strategy::{ParticipantPolicy, ShapeSchedule, ShapeStep},
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

/// The sentinel strategy value of an inactivated occurrence.
const INACTIVE: i64 = -1;

/// The qualification of one strategy's tuning atom: strategy-local
/// declaration names are qualified by their `(occurrence, strategy)`
/// identity so the one global model never conflates two strategies'
/// declarations of the same name. The identical function qualifies the atoms
/// on the resolution side (`seismic_realization::formation::seal`); the
/// reserved `@`-prefixed atom families are global and never qualified.
fn tuning_symbol(occurrence: OccurrenceId, strategy: StrategyId, name: &str) -> String {
    format!("{}.{}.{}", occurrence.0, strategy.0, name)
}

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------

/// Optimization budget: it limits optimization only. Feasibility is decided
/// first regardless of work and time, so a budget can never cause a
/// no-incumbent production failure.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub work: u64,
    pub time: Option<Duration>,
}
impl Default for Budget {
    fn default() -> Self {
        Self {
            work: 200_000,
            time: Some(Duration::from_secs(2)),
        }
    }
}

#[derive(Debug)]
pub enum PlanningFailure {
    /// Every applicable physical plan is proved infeasible against the
    /// target's hard resources, absent capabilities, or the caller's
    /// numerical policy.
    Infeasible,
    /// A broken compiler invariant; never retried.
    CompilerBug(CompilerDefect),
    Solver(String),
}
impl std::fmt::Display for PlanningFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Infeasible => write!(f, "planning is infeasible"),
            Self::CompilerBug(defect) => write!(f, "{defect}"),
            Self::Solver(reason) => write!(f, "solver defect: {reason}"),
        }
    }
}
impl std::error::Error for PlanningFailure {}

/// One solve of the whole space: export the global model, obtain a feasible
/// assignment, spend the optimization budget, and validate exactly one
/// complete assignment. Resolution performs no legality check.
pub fn plan<D: ExecutableDialect>(
    space: &PlanSpace<'_, D>,
    numerics: &NumericalContext<'_>,
    budget: Budget,
) -> Result<CompleteAssignment, PlanningFailure> {
    let view = space.solver_model(numerics);
    let export = build_model(space, &view, numerics)?;
    let (solution, optimal) = solve(&export, budget)?;
    let variables = DecisionVariables {
        variables: export.decision_vars,
        optimal,
        arena: export.arena,
        arena_items: export.arena_items,
        leaves: export.leaf_vars,
    };
    view.validate(&solution, &variables)
        .map_err(PlanningFailure::CompilerBug)
}

// ---------------------------------------------------------------------------
// Model construction
// ---------------------------------------------------------------------------

struct Export {
    model: Arc<Model>,
    decision_vars: Vec<Vec<VarId>>,
    arena: ArenaPacking,
    arena_items: Vec<(OccurrenceId, StrategyId, ResidenceId)>,
    /// The leaf-bound variable of every `(occurrence, strategy, leaf)` the
    /// model materialized, in materialization order (the candidate's
    /// `@leaf<L>` bindings are decoded from these).
    leaf_vars: Vec<(
        OccurrenceId,
        StrategyId,
        CanonicalLeafId,
        VarId,
    )>,
}

/// How `@runtime<N>` atoms bind for one materialization: capacity for
/// feasibility and resources, expected for cost.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RuntimeBinding {
    Capacity,
    Expected,
}

#[derive(Clone, Copy, Debug)]
enum ScalarValue {
    Const(i64),
    Var(VarId),
}

#[derive(Clone, Copy, Debug)]
struct Materialized {
    value: ScalarValue,
    low: i64,
    high: i64,
}

/// One model variable over a declared interval (a tuning parameter or a
/// leaf bound).
#[derive(Clone, Copy)]
struct ParameterVar {
    variable: VarId,
    lower: i64,
    upper: i64,
}

#[derive(Default)]
struct ExpressionCache {
    expressions: BTreeMap<Sym, Materialized>,
    fixed_variables: BTreeMap<i64, VarId>,
    products: BTreeMap<(VarId, VarId), Materialized>,
    div_rem: BTreeMap<(Sym, Sym), (Materialized, Materialized)>,
}

impl ExpressionCache {
    fn variable(&mut self, builder: &mut ModelBuilder, value: ScalarValue) -> VarId {
        match value {
            ScalarValue::Var(variable) => variable,
            ScalarValue::Const(constant) => *self
                .fixed_variables
                .entry(constant)
                .or_insert_with(|| {
                    builder.variable("canonical fixed scalar", Domain::singleton(constant))
                }),
        }
    }
}

/// The mutable model-construction state shared by every materialization.
struct ModelAssembler {
    builder: ModelBuilder,
    cache: ExpressionCache,
    /// `(occurrence, strategy, name)` -> the tuning parameter's variable.
    tuning: BTreeMap<(OccurrenceId, StrategyId, String), ParameterVar>,
    /// `(occurrence, strategy, leaf)` -> the leaf-bound variable.
    leaves: BTreeMap<(OccurrenceId, StrategyId, CanonicalLeafId), ParameterVar>,
    /// Proved leaf intervals, computed once per leaf.
    leaf_intervals: BTreeMap<CanonicalLeafId, (i64, i64)>,
    /// `@runtime<N>` constants under the current binding.
    runtime: BTreeMap<RuntimeExtentId, i64>,
    capacities: BTreeMap<RuntimeExtentId, i64>,
    expected: BTreeMap<RuntimeExtentId, i64>,
    /// `(occurrence, strategy, choice var)` -> the scope decision variable.
    scopes: BTreeMap<(OccurrenceId, StrategyId, ChoiceVarId), VarId>,
    /// The arena packing items, one per Device-arena placement.
    packing_items: Vec<ArenaItem>,
}

impl ModelAssembler {
    fn materialize(
        &mut self,
        expression: &Sym,
        occurrence: OccurrenceId,
        strategy: StrategyId,
        facts: &OccurrenceFacts<'_>,
    ) -> Result<Materialized, PlanningFailure> {
        if let Some(value) = expression.as_constant() {
            return Ok(Materialized {
                value: ScalarValue::Const(value),
                low: value,
                high: value,
            });
        }
        if let Some(value) = self.cache.expressions.get(expression) {
            return Ok(*value);
        }
        let mut terms: Vec<(ScalarValue, i64)> = Vec::new();
        let mut low = 0i128;
        let mut high = 0i128;
        for (monomial, coefficient) in expression.monomials() {
            let mut value = Materialized {
                value: ScalarValue::Const(1),
                low: 1,
                high: 1,
            };
            for (factor, degree) in monomial {
                for _ in 0..*degree {
                    let right = self.atom(factor, occurrence, strategy, facts)?;
                    value = self.multiply(value, right)?;
                }
            }
            terms.push((value.value, coefficient));
            if coefficient >= 0 {
                low += i128::from(coefficient) * i128::from(value.low);
                high += i128::from(coefficient) * i128::from(value.high);
            } else {
                low += i128::from(coefficient) * i128::from(value.high);
                high += i128::from(coefficient) * i128::from(value.low);
            }
        }
        let low = i64::try_from(low).map_err(|_| bug("symbolic value exceeds i64"))?;
        let high = i64::try_from(high).map_err(|_| bug("symbolic value exceeds i64"))?;
        let mut constant = 0i128;
        let mut variable_terms = Vec::new();
        for (value, coefficient) in terms {
            match value {
                ScalarValue::Const(value) => {
                    constant += i128::from(value) * i128::from(coefficient)
                }
                ScalarValue::Var(variable) => {
                    variable_terms.push(LinearTerm::new(variable, coefficient))
                }
            }
        }
        let value = if variable_terms.is_empty() {
            ScalarValue::Const(
                i64::try_from(constant).map_err(|_| bug("symbolic constant exceeds i64"))?,
            )
        } else if constant == 0
            && variable_terms.len() == 1
            && variable_terms[0].coefficient == 1
        {
            ScalarValue::Var(variable_terms[0].variable)
        } else {
            let result = self.builder.variable(
                "symbolic expression",
                Domain::interval(low, high)
                    .map_err(|error| PlanningFailure::Solver(error.to_string()))?,
            );
            variable_terms.push(LinearTerm::new(result, -1));
            equal_zero(&mut self.builder, variable_terms, -constant);
            ScalarValue::Var(result)
        };
        let materialized = Materialized { value, low, high };
        self.cache.expressions.insert(expression.clone(), materialized);
        Ok(materialized)
    }

    fn atom(
        &mut self,
        atom: &Atom,
        occurrence: OccurrenceId,
        strategy: StrategyId,
        facts: &OccurrenceFacts<'_>,
    ) -> Result<Materialized, PlanningFailure> {
        let Atom::Param(name) = atom else {
            return self.arithmetic_atom(atom, occurrence, strategy, facts);
        };
        if let Some(runtime) = name.strip_prefix("@runtime") {
            let id = RuntimeExtentId(
                runtime
                    .parse::<u32>()
                    .map_err(|_| bug(format!("malformed runtime atom `{name}`")))?,
            );
            let value = *self
                .runtime
                .get(&id)
                .ok_or_else(|| bug(format!("runtime extent {} has no binding", id.0)))?;
            return Ok(Materialized {
                value: ScalarValue::Const(value),
                low: value,
                high: value,
            });
        }
        if let Some(leaf) = name.strip_prefix("@leaf") {
            let id = CanonicalLeafId(
                leaf.parse::<u32>()
                    .map_err(|_| bug(format!("malformed leaf atom `{name}`")))?,
            );
            let key = (occurrence, strategy, id);
            if let Some(parameter) = self.leaves.get(&key) {
                return Ok(Materialized {
                    value: ScalarValue::Var(parameter.variable),
                    low: parameter.lower,
                    high: parameter.upper,
                });
            }
            let (low, high) = match self.leaf_intervals.get(&id) {
                Some(interval) => *interval,
                None => {
                    let interval = leaf_interval(facts, id)?;
                    self.leaf_intervals.insert(id, interval);
                    interval
                }
            };
            let variable = self.builder.variable(
                format!("leaf {} bound", id.0),
                Domain::interval(low, high)
                    .map_err(|error| PlanningFailure::Solver(error.to_string()))?,
            );
            let parameter = ParameterVar {
                variable,
                lower: low,
                upper: high,
            };
            self.leaves.insert(key, parameter);
            return Ok(Materialized {
                value: ScalarValue::Var(variable),
                low,
                high,
            });
        }
        let parameter = self
            .tuning
            .get(&(occurrence, strategy, name.clone()))
            .copied()
            .ok_or_else(|| {
                bug(format!(
                    "the plan expression references undeclared symbol `{name}`"
                ))
            })?;
        Ok(Materialized {
            value: ScalarValue::Var(parameter.variable),
            low: parameter.lower,
            high: parameter.upper,
        })
    }

    fn arithmetic_atom(
        &mut self,
        atom: &Atom,
        occurrence: OccurrenceId,
        strategy: StrategyId,
        facts: &OccurrenceFacts<'_>,
    ) -> Result<Materialized, PlanningFailure> {
        let (numerator, denominator, quotient_atom) = match atom {
            Atom::Quot(numerator, denominator) => (numerator, denominator, true),
            Atom::Rem(numerator, denominator) => (numerator, denominator, false),
            Atom::Param(_) => unreachable!("handled by the caller"),
        };
        let key = ((**numerator).clone(), (**denominator).clone());
        if let Some((quotient, remainder)) = self.cache.div_rem.get(&key) {
            return Ok(if quotient_atom { *quotient } else { *remainder });
        }
        let n = self.materialize(numerator, occurrence, strategy, facts)?;
        let d = self.materialize(denominator, occurrence, strategy, facts)?;
        if n.low < 0 || d.low <= 0 {
            return Err(bug("invalid symbolic division domain"));
        }
        if let (ScalarValue::Const(n), ScalarValue::Const(d)) = (n.value, d.value) {
            let quotient = Materialized {
                value: ScalarValue::Const(n / d),
                low: n / d,
                high: n / d,
            };
            let remainder = Materialized {
                value: ScalarValue::Const(n % d),
                low: n % d,
                high: n % d,
            };
            self.cache.div_rem.insert(key, (quotient, remainder));
            return Ok(if quotient_atom { quotient } else { remainder });
        }
        let qh = n.high / d.low;
        let rh = n.high.min(d.high.saturating_sub(1));
        let quotient_var = self.builder.variable(
            "symbolic quotient",
            Domain::interval(0, qh)
                .map_err(|error| PlanningFailure::Solver(error.to_string()))?,
        );
        let remainder_var = self.builder.variable(
            "symbolic remainder",
            Domain::interval(0, rh)
                .map_err(|error| PlanningFailure::Solver(error.to_string()))?,
        );
        let numerator_var = self.cache.variable(&mut self.builder, n.value);
        let denominator_var = self.cache.variable(&mut self.builder, d.value);
        self.builder
            .constraint(Constraint::Arithmetic(Arithmetic::DivRem {
                numerator: numerator_var,
                denominator: denominator_var,
                quotient: quotient_var,
                remainder: remainder_var,
            }));
        let quotient = Materialized {
            value: ScalarValue::Var(quotient_var),
            low: 0,
            high: qh,
        };
        let remainder = Materialized {
            value: ScalarValue::Var(remainder_var),
            low: 0,
            high: rh,
        };
        self.cache.div_rem.insert(key, (quotient, remainder));
        Ok(if quotient_atom { quotient } else { remainder })
    }

    fn multiply(
        &mut self,
        left: Materialized,
        right: Materialized,
    ) -> Result<Materialized, PlanningFailure> {
        match (left.value, right.value) {
            (ScalarValue::Const(0), _) | (_, ScalarValue::Const(0)) => Ok(Materialized {
                value: ScalarValue::Const(0),
                low: 0,
                high: 0,
            }),
            (ScalarValue::Const(1), _) => Ok(right),
            (_, ScalarValue::Const(1)) => Ok(left),
            (ScalarValue::Const(a), ScalarValue::Const(b)) => {
                let value = a
                    .checked_mul(b)
                    .ok_or_else(|| bug("symbolic product overflow"))?;
                Ok(Materialized {
                    value: ScalarValue::Const(value),
                    low: value,
                    high: value,
                })
            }
            _ => {
                let left_var = self.cache.variable(&mut self.builder, left.value);
                let right_var = self.cache.variable(&mut self.builder, right.value);
                let key = if left_var <= right_var {
                    (left_var, right_var)
                } else {
                    (right_var, left_var)
                };
                if let Some(product) = self.cache.products.get(&key) {
                    return Ok(*product);
                }
                let low = i64::try_from(i128::from(left.low) * i128::from(right.low))
                    .map_err(|_| bug("symbolic product underflow"))?;
                let high = i64::try_from(i128::from(left.high) * i128::from(right.high))
                    .map_err(|_| bug("symbolic product overflow"))?;
                let variable = self.builder.variable(
                    "symbolic product",
                    Domain::interval(low, high)
                        .map_err(|error| PlanningFailure::Solver(error.to_string()))?,
                );
                self.builder
                    .constraint(Constraint::Arithmetic(Arithmetic::Product {
                        left: left_var,
                        right: right_var,
                        product: variable,
                    }));
                let product = Materialized {
                    value: ScalarValue::Var(variable),
                    low,
                    high,
                };
                self.cache.products.insert(key, product);
                Ok(product)
            }
        }
    }

    /// One `Sym` as an arena expression: the identical atom resolution as
    /// `materialize`, over the arena's expression algebra.
    fn arena_expression(
        &mut self,
        expression: &Sym,
        occurrence: OccurrenceId,
        strategy: StrategyId,
        facts: &OccurrenceFacts<'_>,
    ) -> Result<ArenaExpression, PlanningFailure> {
        if let Some(value) = expression.as_constant() {
            return Ok(ArenaExpression::Constant(value));
        }
        let mut terms = Vec::new();
        for (monomial, coefficient) in expression.monomials() {
            let mut factors = Vec::new();
            for (factor, degree) in monomial {
                for _ in 0..*degree {
                    let materialized = self.atom(factor, occurrence, strategy, facts)?;
                    factors.push(match materialized.value {
                        ScalarValue::Const(value) => ArenaExpression::Constant(value),
                        ScalarValue::Var(variable) => ArenaExpression::Variable(variable),
                    });
                }
            }
            let value = match factors.as_slice() {
                [] => ArenaExpression::Constant(1),
                [factor] => factor.clone(),
                _ => ArenaExpression::Product(factors),
            };
            terms.push((coefficient, value));
        }
        Ok(ArenaExpression::Sum(terms))
    }

    /// Set the `@runtime` constants for one materialization context.
    fn bind_runtime(&mut self, binding: RuntimeBinding) {
        self.runtime = match binding {
            RuntimeBinding::Capacity => self.capacities.clone(),
            RuntimeBinding::Expected => self.expected.clone(),
        };
    }
}

/// The proved interval of one scalar leaf's atom, from its value kind: an
/// index or range bound, else the unsigned 32-bit range.
fn leaf_interval(
    facts: &OccurrenceFacts<'_>,
    leaf: CanonicalLeafId,
) -> Result<(i64, i64), PlanningFailure> {
    use seismic_lang::logical::value::GraphValueKind;
    let value = facts.leaf(leaf).value;
    let high = match facts.value_kind(value) {
        GraphValueKind::Index { bound } | GraphValueKind::Range { bound } => match bound {
            ExtentExpr::Static(n) => *n,
            ExtentExpr::Runtime(id) => facts.runtime_extent(*id).capacity,
            ExtentExpr::Sym(_) => {
                return Err(bug("a symbolic extent survived specialization"))
            }
        },
        GraphValueKind::Scalar(DType::I32 | DType::U32) => u64::from(u32::MAX),
        _ => {
            return Err(bug(format!(
                "leaf {} has no unsigned arithmetic domain",
                leaf.0
            )))
        }
    };
    let high = i64::try_from(high)
        .map_err(|_| bug("a leaf interval exceeds the solver numeric range"))?;
    Ok((0, high))
}

fn bug(invariant: impl Into<String>) -> PlanningFailure {
    PlanningFailure::CompilerBug(CompilerDefect::new(Package::P1, invariant))
}

fn equal_zero(builder: &mut ModelBuilder, terms: Vec<LinearTerm>, rhs: i128) {
    builder.constraint(Constraint::LinearLe {
        terms: terms.clone(),
        rhs,
    });
    builder.constraint(Constraint::LinearLe {
        terms: terms
            .into_iter()
            .map(|term| LinearTerm::new(term.variable, -term.coefficient))
            .collect(),
        rhs: -rhs,
    });
}

/// The occurrences a strategy retains as calls, in schedule order (child
/// activity follows the parent's selection).
fn collect_calls(schedule: &ShapeSchedule, out: &mut Vec<OccurrenceId>) {
    for step in &schedule.steps {
        match step {
            ShapeStep::Call { occurrence, .. } => out.push(*occurrence),
            ShapeStep::If {
                then_schedule,
                else_schedule,
                ..
            } => {
                collect_calls(then_schedule, out);
                collect_calls(else_schedule, out);
            }
            ShapeStep::Repeat { body, .. } => collect_calls(body, out),
            ShapeStep::Launch(_) | ShapeStep::Guard { .. } | ShapeStep::PullCounterReset { .. } => {
            }
        }
    }
}

fn build_model<D: ExecutableDialect>(
    space: &PlanSpace<'_, D>,
    view: &SolverModelView<'_>,
    numerics: &NumericalContext<'_>,
) -> Result<Export, PlanningFailure> {
    let facts = space.forest().facts();
    let entry = facts.entry();
    let profile = space.profile();
    let mut assembler = ModelAssembler {
        builder: ModelBuilder::new(),
        cache: ExpressionCache::default(),
        tuning: BTreeMap::new(),
        leaves: BTreeMap::new(),
        leaf_intervals: BTreeMap::new(),
        runtime: BTreeMap::new(),
        capacities: BTreeMap::new(),
        expected: BTreeMap::new(),
        scopes: BTreeMap::new(),
        packing_items: Vec::new(),
    };
    assembler.builder.units("estimated executable cost");

    // Runtime extent bindings: capacity for feasibility/resources, expected
    // for cost.
    for extent in space.logical().runtime_extents() {
        let capacity = i64::try_from(extent.capacity)
            .map_err(|_| bug("a runtime capacity exceeds the solver numeric range"))?;
        let expected = match extent.expected {
            Some(expected) => i64::try_from(expected)
                .map_err(|_| bug("a runtime expectation exceeds the solver numeric range"))?,
            None => capacity,
        };
        assembler.capacities.insert(extent.id, capacity);
        assembler.expected.insert(extent.id, expected);
    }
    assembler.bind_runtime(RuntimeBinding::Capacity);

    // --- decisions ------------------------------------------------------------
    let mut decision_vars: Vec<Vec<VarId>> = Vec::new();
    let mut occurrence_vars: BTreeMap<OccurrenceId, VarId> = BTreeMap::new();
    let mut occurrence_domains: BTreeMap<OccurrenceId, Vec<i64>> = BTreeMap::new();
    for decision in view.decisions() {
        match decision {
            Decision::Strategy {
                occurrence,
                options,
                inactive_allowed,
            } => {
                let mut domain: Vec<i64> = (0..options.len() as i64).collect();
                if *inactive_allowed {
                    domain.push(INACTIVE);
                }
                let variable = assembler.builder.variable(
                    format!("occurrence {}", occurrence.0),
                    Domain::set(domain.clone()),
                );
                occurrence_vars.insert(*occurrence, variable);
                occurrence_domains.insert(*occurrence, domain);
                decision_vars.push(vec![variable]);
            }
            Decision::Tuning {
                occurrence,
                strategy,
                parameter,
                lower,
                upper,
            } => {
                let declaration = space
                    .strategies(*occurrence)[*strategy]
                    .consequences()
                    .tuning
                    .get(*parameter)
                    .ok_or_else(|| {
                        bug("a tuning decision names a parameter absent from its strategy")
                    })?;
                let symbol = tuning_symbol(*occurrence, *strategy, &declaration.name);
                let (lower, upper) = (
                    i64::try_from(*lower)
                        .map_err(|_| bug("a tuning lower bound exceeds i64"))?,
                    i64::try_from(*upper)
                        .map_err(|_| bug("a tuning upper bound exceeds i64"))?,
                );
                let variable = assembler.builder.variable(
                    format!("parameter {symbol}"),
                    Domain::interval(lower, upper)
                        .map_err(|error| PlanningFailure::Solver(error.to_string()))?,
                );
                assembler.tuning.insert(
                    (*occurrence, *strategy, declaration.name.clone()),
                    ParameterVar {
                        variable,
                        lower,
                        upper,
                    },
                );
                decision_vars.push(vec![variable]);
            }
            Decision::Scope {
                occurrence,
                strategy,
                residence,
                var,
                options,
            } => {
                // The residence names the placement the choice var belongs
                // to; the model variable is keyed by the identity triple.
                let _ = residence;
                let domain: Vec<i64> = (0..options.len() as i64).collect();
                let variable = assembler.builder.variable(
                    format!("scope {}.{}", strategy.0, var.0),
                    Domain::set(domain),
                );
                assembler
                    .scopes
                    .insert((*occurrence, *strategy, *var), variable);
                decision_vars.push(vec![variable]);
            }
            Decision::ArenaOffset { .. } => {
                // Offsets are decoded from the arena packing certificate;
                // the decision records the domain only.
                decision_vars.push(Vec::new());
            }
        }
    }

    // --- child activity follows the parent's selection ----------------------
    for occurrence in occurrence_vars.keys().copied().collect::<Vec<_>>() {
        if occurrence == entry {
            continue;
        }
        let parent = facts
            .occurrence(occurrence)
            .parent
            .as_ref()
            .map(|parent| parent.occurrence)
            .ok_or_else(|| bug("a non-entry occurrence has no parent call"))?;
        let child_domain = occurrence_domains[&occurrence].clone();
        let parent_domain = occurrence_domains[&parent].clone();
        let mut tuples = Vec::new();
        for parent_value in &parent_domain {
            if *parent_value == INACTIVE {
                tuples.push(vec![INACTIVE, INACTIVE]);
                continue;
            }
            let parent_strategy = StrategyId::from_index(*parent_value as usize);
            let mut retains = Vec::new();
            collect_calls(
                space.strategies(parent)[parent_strategy]
                    .routed()
                    .shape()
                    .schedule(),
                &mut retains,
            );
            if retains.contains(&occurrence) {
                for child_value in &child_domain {
                    tuples.push(vec![*parent_value, *child_value]);
                }
            } else {
                tuples.push(vec![*parent_value, INACTIVE]);
            }
        }
        assembler.builder.constraint(Constraint::Table {
            variables: vec![occurrence_vars[&parent], occurrence_vars[&occurrence]],
            tuples,
        });
    }

    // --- identity-matching evidence records -------------------------------
    // The non-assignment `EvidenceKey` identity fields select which records
    // may qualify anything; the assignment and policy components are decided
    // by `numerics::evidence_qualifies` (record level here, assignment level
    // in `SolverModelView::validate`).
    let identity = space.logical().identity;
    let workload = numerics::workload_fingerprint(&space.logical().domain);
    let target = EffectiveTargetIdentity {
        backend: profile.backend.clone(),
        capability_fingerprint: profile.capability_fingerprint.clone(),
    };
    let matching_evidence: Vec<&numerics::NumericalEvidence> = numerics
        .evidence
        .iter()
        .filter(|record| {
            record.key.logical == identity
                && record.key.workload == workload
                && record.key.target == target
                && record.key.toolchain.0 == profile.toolchain_fingerprint
        })
        .collect();

    // --- per-strategy constraints, cost, and arena items --------------------
    let mut arena_items: Vec<(OccurrenceId, StrategyId, ResidenceId)> = Vec::new();
    let mut arena_records: Vec<(OccurrenceId, StrategyId, Lifetime, u64)> = Vec::new();
    for (occurrence, _) in facts.occurrences() {
        let strategies = space.strategies(occurrence);
        for (strategy, closed) in strategies.entries() {
            let variable = occurrence_vars[&occurrence];
            let ordinal = strategy.0 as i64;
            let selected = Literal::new(variable, ordinal);
            let domain_without: Vec<i64> = occurrence_domains[&occurrence]
                .iter()
                .copied()
                .filter(|value| *value != ordinal)
                .collect();
            let consequences = closed.consequences();
            let routed = closed.routed();

            // Capability coverage: an exact signature absent from the
            // effective target removes the strategy before solving.
            if !consequences
                .required_capabilities
                .is_subset(&profile.effective_signatures)
            {
                assembler.builder.constraint(Constraint::InDomain {
                    variable,
                    domain: Domain::set(domain_without.clone()),
                });
                continue;
            }

            // Replay M1's per-residence scoped byte expressions (residence
            // order, option order; ABI options contribute none).
            let mut scoped: BTreeMap<(ResidenceId, StorageScope), Sym> = BTreeMap::new();
            let mut cursor = consequences
                .constraints
                .iter()
                .filter(|constraint| {
                    matches!(constraint, SolverConstraint::ScopedBytes { .. })
                });
            for (residence, record) in routed.residences().iter() {
                let options: Vec<StorageScope> = match &record.choice {
                    ResidenceChoice::Fixed(scope) => vec![*scope],
                    ResidenceChoice::SolverChoice { options, .. } => {
                        options.as_slice().to_vec()
                    }
                };
                for scope in options {
                    if scope == StorageScope::Abi {
                        continue;
                    }
                    let Some(SolverConstraint::ScopedBytes { bytes, .. }) = cursor.next() else {
                        return Err(bug(
                            "the consequence constraints disagree with the residence graph",
                        ));
                    };
                    scoped.insert((residence, scope), bytes.clone());
                }
            }
            if cursor.next().is_some() {
                return Err(bug(
                    "the consequence constraints disagree with the residence graph",
                ));
            }

            // M1 constraints, guarded by this strategy's selection.
            assembler.bind_runtime(RuntimeBinding::Capacity);
            for constraint in &consequences.constraints {
                match constraint {
                    SolverConstraint::Range {
                        expr,
                        lower,
                        upper,
                    } => {
                        let value =
                            assembler.materialize(expr, occurrence, strategy, facts)?;
                        let (lower, upper) = (
                            i64::try_from(*lower)
                                .map_err(|_| bug("a resource lower bound exceeds i64"))?,
                            i64::try_from(*upper)
                                .map_err(|_| bug("a resource upper bound exceeds i64"))?,
                        );
                        match value.value {
                            ScalarValue::Const(constant) => {
                                if !(constant >= lower && constant <= upper) {
                                    assembler
                                        .builder
                                        .constraint(Constraint::InDomain {
                                            variable,
                                            domain: Domain::set(domain_without.clone()),
                                        });
                                }
                            }
                            ScalarValue::Var(value_var) => {
                                assembler.builder.guarded_constraint(
                                    vec![selected],
                                    Constraint::InDomain {
                                        variable: value_var,
                                        domain: Domain::interval(lower, upper).map_err(
                                            |error| PlanningFailure::Solver(error.to_string()),
                                        )?,
                                    },
                                );
                            }
                        }
                    }
                    SolverConstraint::ScopeOptions { .. } => {
                        // The scope decision's domain is built from the same
                        // options; no further constraint.
                    }
                    SolverConstraint::ScopedBytes { .. } => {
                        // Device-arena placements enter the packing below;
                        // the aggregate limits are M1's Range constraints.
                    }
                    SolverConstraint::Numerical(transfer) => {
                        // Numerical admission: satisfied transfers are free,
                        // violated transfers remove the strategy. A transfer
                        // that requires evidence is admitted only when some
                        // identity-matching record could qualify an
                        // assignment containing this strategy — its retained
                        // fingerprint matches its own witnessed selection, its
                        // witnessed selection selects this strategy, and its
                        // assessment satisfies the policy. The exact
                        // assignment-level predicate
                        // (`numerics::evidence_qualifies` over the solved
                        // selection and symbols) is enforced in
                        // `SolverModelView::validate`. The rounding-count
                        // bound is evaluated at capacity: admission must hold
                        // for every invocation in the envelope.
                        let runtime = |id: RuntimeExtentId| {
                            Some(space.logical().runtime_extent(id).capacity)
                        };
                        match numerics::satisfies_policy(
                            transfer,
                            numerics.precision,
                            &runtime,
                        ) {
                            PolicyDecision::Satisfied => {}
                            PolicyDecision::Violated { .. } => {
                                assembler
                                    .builder
                                    .constraint(Constraint::InDomain {
                                        variable,
                                        domain: Domain::set(domain_without.clone()),
                                    });
                            }
                            PolicyDecision::RequiresEvidence { .. } => {
                                let admitted = numerics::accepts_evidence(numerics.precision)
                                    && matching_evidence.iter().any(|record| {
                                        record.key.assignment
                                            == numerics::assignment_fingerprint(
                                                &record.selections,
                                            )
                                            && record.selections.get(&occurrence)
                                                == Some(&Some(strategy))
                                            && record.assessment.satisfies(numerics.precision)
                                    });
                                if !admitted {
                                    assembler
                                        .builder
                                        .constraint(Constraint::InDomain {
                                            variable,
                                            domain: Domain::set(domain_without.clone()),
                                        });
                                }
                            }
                        }
                    }
                }
            }

            // Per-axis workgroup geometry: the capacity total must fit in
            // `max_workgroups_axis[0]` workgroups of the solved participant
            // count (`ceil(T / P) <= M` iff `T <= M * P`, exact for `P >= 1`,
            // which M1's participant constraint guarantees for any launch
            // with structurally nonzero work; a launch with structurally
            // zero work is skipped by the seal). A serialized traversal
            // seals a single-workgroup grid and is exempt: its native grid
            // is 1x1x1 regardless of the total.
            let max_axis = i64::try_from(profile.limits.max_workgroups_axis[0])
                .map_err(|_| bug("max_workgroups_axis exceeds the solver numeric range"))?;
            if max_axis < 1
                || profile.limits.max_workgroups_axis[1] < 1
                || profile.limits.max_workgroups_axis[2] < 1
            {
                return Err(bug("max_workgroups_axis admits no workgroup"));
            }
            for (block, routed_block) in routed.blocks().entries() {
                if matches!(
                    routed.shape().blocks()[block].participants.policy,
                    ParticipantPolicy::Serial
                ) {
                    continue;
                }
                let iteration = &routed_block.interface.iteration;
                let total =
                    assembler.materialize(&iteration.total_symbol(), occurrence, strategy, facts)?;
                let participants = assembler.materialize(
                    &iteration.participants,
                    occurrence,
                    strategy,
                    facts,
                )?;
                // T <= max_axis * P: -max_axis * P + T <= 0.
                let terms = vec![
                    LinearTerm::new(
                        assembler
                            .cache
                            .variable(&mut assembler.builder, participants.value),
                        -max_axis,
                    ),
                    LinearTerm::new(
                        assembler.cache.variable(&mut assembler.builder, total.value),
                        1,
                    ),
                ];
                assembler.builder.guarded_constraint(
                    vec![selected],
                    Constraint::LinearLe { terms, rhs: 0 },
                );
            }

            // Cost: the exact same expression the resolver prices.
            assembler.bind_runtime(RuntimeBinding::Expected);
            let cost = assembler.materialize(&consequences.cost, occurrence, strategy, facts)?;
            if cost.low < 0 {
                return Err(bug("an alternative cost may be negative"));
            }
            match cost.value {
                ScalarValue::Const(constant) => {
                    let constant = u64::try_from(constant)
                        .map_err(|_| bug("an alternative cost is negative"))?;
                    assembler
                        .builder
                        .guarded_cost(vec![selected], Cost::Constant(constant));
                }
                ScalarValue::Var(cost_var) => {
                    assembler.builder.guarded_cost(
                        vec![selected],
                        Cost::Linear {
                            constant: 0,
                            terms: vec![LinearTerm::new(cost_var, 1)],
                        },
                    );
                }
            }
            assembler.bind_runtime(RuntimeBinding::Capacity);

            // Arena items: every Device-arena placement of this strategy.
            for (residence, record) in routed.residences().iter() {
                let scope_choice = match &record.choice {
                    ResidenceChoice::Fixed(StorageScope::DeviceArena) => Some(None),
                    ResidenceChoice::SolverChoice { options, var } => options
                        .as_slice()
                        .iter()
                        .position(|scope| *scope == StorageScope::DeviceArena)
                        .map(|index| Some((index, *var))),
                    _ => None,
                };
                let Some(scope_choice) = scope_choice else {
                    continue;
                };
                let bytes = scoped
                    .get(&(residence, StorageScope::DeviceArena))
                    .ok_or_else(|| {
                        bug("a Device-arena residence has no scoped byte expression")
                    })?;
                let alignment = record
                    .planes
                    .iter()
                    .map(|plane| plane.alignment)
                    .max()
                    .unwrap_or(1)
                    .max(1);
                let mut activation = vec![ArenaLiteral {
                    variable,
                    value: ordinal,
                }];
                if let Some((index, var)) = scope_choice {
                    let scope_var = assembler.scopes[&(occurrence, strategy, var)];
                    activation.push(ArenaLiteral {
                        variable: scope_var,
                        value: index as i64,
                    });
                }
                let ordinal_id = assembler.packing_items.len() as u32;
                let bytes =
                    assembler.arena_expression(bytes, occurrence, strategy, facts)?;
                assembler.packing_items.push(ArenaItem {
                    ordinal: ordinal_id,
                    activation: vec![activation],
                    bytes,
                    alignment,
                });
                arena_items.push((occurrence, strategy, residence));
                arena_records.push((
                    occurrence,
                    strategy,
                    record.lifetime.clone(),
                    alignment,
                ));
            }
        }
    }

    // Arena interference: simultaneously active items with overlapping
    // lifetimes must not share arena bytes. Strategies of one occurrence
    // are mutually exclusive (no edge); items of one strategy edge iff their
    // lifetimes overlap in its step space; items of different occurrences
    // can be simultaneously active and their lifetimes live in different
    // step spaces, so they interfere conservatively.
    let mut interference: BTreeSet<(u32, u32)> = BTreeSet::new();
    for (left, (left_occ, left_strategy, left_lifetime, _)) in
        arena_records.iter().enumerate()
    {
        for (right, (right_occ, right_strategy, right_lifetime, _)) in
            arena_records.iter().enumerate().skip(left + 1)
        {
            if left_occ == right_occ {
                if left_strategy != right_strategy {
                    // Mutually exclusive strategies of one occurrence.
                    continue;
                }
                let step_count = space.strategies(*left_occ)[*left_strategy]
                    .routed()
                    .steps()
                    .len() as u32;
                if !lifetimes_overlap(step_count, left_lifetime, right_lifetime) {
                    continue;
                }
            }
            interference.insert((left as u32, right as u32));
        }
    }

    let arena = ArenaPacking {
        capacity: view.arena_capacity(),
        items: std::mem::take(&mut assembler.packing_items),
        interference: interference.into_iter().collect(),
    };
    assembler
        .builder
        .constraint(Constraint::Schedule(SchedulingConstraint::ArenaPacking(
            arena.clone(),
        )));

    let model = Arc::new(
        assembler
            .builder
            .build()
            .map_err(|error| PlanningFailure::Solver(error.to_string()))?,
    );
    let leaf_vars = assembler
        .leaves
        .iter()
        .map(|((occurrence, strategy, leaf), parameter)| {
            (*occurrence, *strategy, *leaf, parameter.variable)
        })
        .collect();
    Ok(Export {
        model,
        decision_vars,
        arena,
        arena_items,
        leaf_vars,
    })
}

/// Whether two lifetimes of one strategy overlap in its step space.
fn lifetimes_overlap(step_count: u32, left: &Lifetime, right: &Lifetime) -> bool {
    let interval = |lifetime: &Lifetime| -> (u32, u32) {
        match lifetime {
            Lifetime::Whole => (0, step_count.saturating_sub(1)),
            Lifetime::Steps { first, last } => (first.0, last.0),
            Lifetime::KernelLocal(_) => (0, step_count.saturating_sub(1)),
        }
    };
    let (left_first, left_last) = interval(left);
    let (right_first, right_last) = interval(right);
    left_first <= right_last && right_first <= left_last
}

// ---------------------------------------------------------------------------
// Solve (budget ordering)
// ---------------------------------------------------------------------------

fn solve(export: &Export, budget: Budget) -> Result<(FeasibleSolution, bool), PlanningFailure> {
    let mut search = Search::new(
        Arc::clone(&export.model),
        Options {
            algorithm: Algorithm::Exact,
            ..Options::default()
        },
    )
    .map_err(|error| PlanningFailure::Solver(error.to_string()))?;
    // Feasibility is decided first; the budget limits optimization only.
    let outcome = search
        .advance_budgeted(SolverBudget {
            optimization: SolverLimits {
                work: budget.work.max(1),
                time: budget.time,
                memory_bytes: None,
            },
        })
        .map_err(|error| PlanningFailure::Solver(error.to_string()))?;
    Ok(match outcome {
        Budgeted::Optimal(solution) => (solution.feasible().clone(), true),
        Budgeted::Incumbent { solution, .. } => (solution, false),
        Budgeted::Infeasible => return Err(PlanningFailure::Infeasible),
        Budgeted::Suspended(progress) => match progress.incumbent {
            // Production has no incomplete/no-incumbent result: a suspension
            // with an incumbent is a budget stop, without one it is a bug.
            Some(solution) => (solution, false),
            None => {
                return Err(bug("the search suspended before feasibility was decided"));
            }
        },
    })
}
