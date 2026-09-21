//! Exact finite planning over `magnitude-solver`.
//!
//! Only implementation selection, finite decisions, and decision-only
//! subconstraints enter the CP model. Mixed invocation/decision expressions
//! remain exact arena expressions and become frozen residual variant guards.
//! A monotonic cursor adds one exact no-good after every witnessed assignment,
//! so it is deterministic, finite, and never repeats an assignment.

use crate::expression::PlanningExpr;
use crate::implementation::{Implementation as ClosedImplementation, ImplementationIdentity};
use crate::target::Backend;
use seismic_lang::expr::{
    BoolExpr, DecisionId, ExprArena, PartialAssignment, SymbolId, SymbolKind, SymbolValue,
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SolverAllowance {
    pub work: u64,
    pub memory_bytes: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct RawAssignment {
    implementation: u32,
    decisions: Vec<(DecisionId, i64)>,
}

impl RawAssignment {
    fn new(implementation: u32, decisions: Vec<(DecisionId, i64)>) -> Self {
        Self {
            implementation,
            decisions,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct FeasibleAssignment(RawAssignment);

impl FeasibleAssignment {
    pub(crate) fn universal() -> Self {
        Self(RawAssignment::new(0, Vec::new()))
    }
    pub(crate) fn implementation(&self) -> u32 {
        self.0.implementation
    }
    pub(crate) fn decisions(&self) -> &[(DecisionId, i64)] {
        &self.0.decisions
    }
    pub(crate) fn value(&self, decision: DecisionId) -> Option<i64> {
        self.0.value(decision)
    }
}

impl RawAssignment {
    fn decisions(&self) -> &[(DecisionId, i64)] {
        &self.decisions
    }
    fn value(&self, decision: DecisionId) -> Option<i64> {
        self.decisions
            .iter()
            .find(|(id, _)| *id == decision)
            .map(|(_, value)| *value)
    }
}

#[derive(Debug)]
pub(crate) enum AssignmentStep {
    Assignment(FeasibleAssignment),
    Complete,
    /// Optional alternative search stopped at its declared optimization
    /// budget. Mandatory coverage is already supplied directly by the
    /// universal portable assignment and never depends on this cursor.
    BudgetExhausted(CursorBudgetReport),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CursorBudgetReport {
    pub work_units: u64,
    pub elapsed_ms: u64,
    pub memory_bytes: u64,
}

#[derive(Debug)]
pub struct SolverModel {
    inner: Option<internals::Model>,
}

pub struct SolverModelBuilder<'a> {
    inner: internals::ModelBuilder<'a>,
}

impl<'a> SolverModelBuilder<'a> {
    pub fn new(arena: &'a mut ExprArena) -> Self {
        Self {
            inner: internals::ModelBuilder::new(arena),
        }
    }

    pub fn bind_target(&mut self, symbol: SymbolId, value: SymbolValue) {
        self.inner.bind_target(symbol, value)
    }

    /// Duration stays on the implementation as an interval-valued runtime
    /// evaluator. It is deliberately not flattened into an unrelated proxy.
    pub(crate) fn implementation<B: Backend>(
        &mut self,
        index: u32,
        candidate: &ClosedImplementation<B>,
        planning: PlanningExpr<BoolExpr>,
    ) {
        let decisions = candidate
            .decisions()
            .iter()
            .map(|(decision, _)| *decision)
            .collect::<Vec<_>>();
        self.inner
            .implementation(index, candidate.identity().clone(), &decisions, planning)
    }

    pub fn build(self) -> SolverModel {
        SolverModel {
            inner: self.inner.build(),
        }
    }
}

impl SolverModel {
    pub(crate) fn assignments(&self, budget: SolverAllowance) -> AssignmentCursor<'_> {
        AssignmentCursor {
            inner: self.inner.as_ref().map(|model| model.cursor(budget)),
        }
    }
}

pub(crate) struct AssignmentCursor<'a> {
    inner: Option<internals::Cursor<'a>>,
}

impl AssignmentCursor<'_> {
    pub(crate) fn next(&mut self) -> AssignmentStep {
        self.inner
            .as_mut()
            .map_or(AssignmentStep::Complete, internals::Cursor::next)
    }
}

mod internals {
    use super::*;
    use magnitude_solver::model::{
        Constraint, Domain, Fragment, LinearTerm, Literal, ModelBuilder as GenericBuilder, VarId,
    };
    use magnitude_solver::{Algorithm, Limits, Options, Outcome, Policy, Search};
    use seismic_lang::expr::{
        compiled::CompiledDecisionPredicate, AnyExpr, BinaryOp, CmpOp, EvalError, NaryOp, NodeView,
        SymbolSort, UnaryOp,
    };
    use std::collections::{BTreeMap, HashMap, HashSet};
    use std::sync::Arc;
    use std::time::Instant;

    struct PendingImplementation {
        index: u32,
        identity: ImplementationIdentity,
        decisions: Vec<DecisionId>,
        planning: PlanningExpr<BoolExpr>,
    }

    struct Implementation {
        index: u32,
        identity: ImplementationIdentity,
        selected: VarId,
        decisions: Vec<(DecisionId, SymbolId, VarId)>,
        exact_filter: Option<CompiledDecisionPredicate>,
    }

    impl std::fmt::Debug for Implementation {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Implementation")
                .field("index", &self.index)
                .field("identity", &self.identity)
                .field("selected", &self.selected)
                .field("decisions", &self.decisions)
                .field("has_exact_filter", &self.exact_filter.is_some())
                .finish()
        }
    }

    #[derive(Debug)]
    pub(super) struct Model {
        base: Fragment,
        variables: Vec<(String, Domain)>,
        implementations: Vec<Implementation>,
        assignment_variables: Vec<VarId>,
        retained_bytes: u64,
    }

    pub(super) struct ModelBuilder<'a> {
        arena: &'a mut ExprArena,
        targets: PartialAssignment,
        bound_targets: HashSet<SymbolId>,
        implementations: Vec<PendingImplementation>,
    }

    impl<'a> ModelBuilder<'a> {
        pub(super) fn new(arena: &'a mut ExprArena) -> Self {
            Self {
                arena,
                targets: PartialAssignment::new(),
                bound_targets: HashSet::new(),
                implementations: Vec::new(),
            }
        }

        pub(super) fn bind_target(&mut self, symbol: SymbolId, value: SymbolValue) {
            if !matches!(
                self.arena.symbol_kind(symbol),
                SymbolKind::TargetConstant(_)
            ) {
                panic!("SolverModelBuilder::bind_target received a non-target symbol");
            }
            assert_value_sort(self.arena.symbol_sort(symbol), value);
            if !self.bound_targets.insert(symbol) {
                panic!("SolverModelBuilder target constant was bound twice");
            }
            self.targets.bind(symbol, value);
        }

        pub(super) fn implementation(
            &mut self,
            index: u32,
            identity: ImplementationIdentity,
            decisions: &[DecisionId],
            planning: PlanningExpr<BoolExpr>,
        ) {
            if self
                .implementations
                .iter()
                .any(|implementation| implementation.index == index)
            {
                panic!("SolverModelBuilder implementation index was registered twice");
            }
            if self
                .implementations
                .iter()
                .any(|implementation| implementation.identity == identity)
            {
                panic!("SolverModelBuilder implementation identity was registered twice");
            }
            let mut unique = HashSet::new();
            for decision in decisions {
                if !unique.insert(*decision) {
                    panic!("SolverModelBuilder implementation repeats a decision");
                }
                self.arena.decision_domain(*decision);
            }
            let declared_symbols = unique
                .iter()
                .map(|decision| self.arena.decision_symbol(*decision))
                .collect::<HashSet<_>>();
            for expression in [AnyExpr::Bool(planning.node())] {
                for symbol in self.arena.free_symbols(expression) {
                    match self.arena.symbol_kind(symbol) {
                        SymbolKind::TargetConstant(_) => {}
                        SymbolKind::Decision(_) if declared_symbols.contains(&symbol) => {}
                        SymbolKind::Decision(_) => panic!(
                            "implementation expression references a decision it did not declare"
                        ),
                        SymbolKind::CallDimension(_) | SymbolKind::CallScalar(_) => {
                            panic!("planning expression contains an invocation symbol")
                        }
                        SymbolKind::RuntimeValue(_)
                        | SymbolKind::TemplateDimension(_)
                        | SymbolKind::LoopBinder(_)
                        | SymbolKind::ScheduleSlot(_) => {
                            panic!("planning expression contains a non-planning symbol")
                        }
                    }
                }
            }
            self.implementations.push(PendingImplementation {
                index,
                identity,
                decisions: decisions.to_vec(),
                planning,
            });
        }

        pub(super) fn build(self) -> Option<Model> {
            if self.implementations.is_empty() {
                return None;
            }
            for symbol in self.arena.symbols() {
                if matches!(
                    self.arena.symbol_kind(symbol),
                    SymbolKind::TargetConstant(_)
                ) && !self.bound_targets.contains(&symbol)
                {
                    panic!("every target constant must be bound before solver export");
                }
            }

            let mut builder = GenericBuilder::new();
            builder.units("feasibility");
            let mut decision_vars = HashMap::<DecisionId, VarId>::new();
            let mut decision_symbols = HashMap::<SymbolId, VarId>::new();
            let mut decision_bounds = HashMap::<VarId, (i64, i64)>::new();
            let mut ordered_decision_vars = Vec::new();
            let mut implementations = Vec::new();
            let selection_vars = self
                .implementations
                .iter()
                .map(|implementation| {
                    builder.variable(
                        format!("implementation.{}.selected", implementation.index),
                        Domain::boolean(),
                    )
                })
                .collect::<Vec<_>>();
            builder.constraint(Constraint::ExactlyOne {
                variables: selection_vars.clone(),
            });

            for (pending, selected) in self
                .implementations
                .into_iter()
                .zip(selection_vars.iter().copied())
            {
                let mut decisions = Vec::new();
                for decision in pending.decisions {
                    if decision_vars.contains_key(&decision) {
                        panic!("a finite decision belongs to more than one implementation");
                    }
                    let symbol = self.arena.decision_symbol(decision);
                    let domain = self.arena.decision_domain(decision);
                    let variable = builder.variable(
                        format!("decision.{decision:?}"),
                        Domain::set(domain.values().iter().copied()),
                    );
                    builder.constraint(Constraint::InactiveValue {
                        active: Literal::new(selected, 1),
                        variable,
                        inactive: domain.values()[0],
                    });
                    decision_vars.insert(decision, variable);
                    decision_symbols.insert(symbol, variable);
                    decision_bounds.insert(
                        variable,
                        (
                            *domain.values().first().expect("finite domain is non-empty"),
                            *domain.values().last().expect("finite domain is non-empty"),
                        ),
                    );
                    ordered_decision_vars.push(variable);
                    decisions.push((decision, symbol, variable));
                }

                let planning = guarded_total(self.arena, pending.planning.node());
                let planning = self.arena.partial(planning, &self.targets);
                let mut conjuncts = Vec::new();
                collect_decision_conjuncts(self.arena, planning, &mut conjuncts);

                let exact_filter = if conjuncts.is_empty() {
                    None
                } else {
                    let conjunction = self.arena.all(&conjuncts);
                    Some(self.arena.compile_decision_bool(conjunction))
                };
                let mut exporter = PruningExporter::new(
                    self.arena,
                    &mut builder,
                    &decision_symbols,
                    &decision_bounds,
                );
                for conjunct in conjuncts {
                    if let Some(predicate) = exporter.boolean(conjunct) {
                        exporter.builder.constraint(Constraint::Clause {
                            literals: vec![Literal::new(selected, 0), Literal::new(predicate, 1)],
                        });
                    }
                }
                implementations.push(Implementation {
                    index: pending.index,
                    identity: pending.identity,
                    selected,
                    decisions,
                    exact_filter,
                });
            }

            let assignment_variables = selection_vars
                .into_iter()
                .chain(ordered_decision_vars)
                .collect::<Vec<_>>();
            drop(decision_vars);
            let generic = builder
                .build()
                .unwrap_or_else(|error| panic!("private Seismic solver model is invalid: {error}"));
            let variables = generic
                .variables()
                .iter()
                .map(|variable| (variable.name.clone(), variable.domain.clone()))
                .collect::<Vec<_>>();
            let retained_bytes = u64::try_from(generic.retained_bytes()).unwrap_or(u64::MAX);
            let interface = (0..variables.len()).map(VarId).collect();
            let base = Fragment::new(generic, interface).unwrap_or_else(|error| {
                panic!("private Seismic solver fragment is invalid: {error}")
            });
            Some(Model {
                base,
                variables,
                implementations,
                assignment_variables,
                retained_bytes,
            })
        }
    }

    impl Model {
        pub(super) fn cursor(&self, budget: SolverAllowance) -> Cursor<'_> {
            Cursor {
                model: self,
                budget,
                started: Instant::now(),
                work: 0,
                exclusions: Vec::new(),
                complete: false,
                exhausted: false,
            }
        }

        fn instantiate(
            &self,
            exclusions: &[Vec<i64>],
        ) -> (Arc<magnitude_solver::Model>, Vec<VarId>) {
            let mut builder = GenericBuilder::new();
            builder.units("feasibility");
            let bindings = self
                .variables
                .iter()
                .map(|(name, domain)| builder.variable(name.clone(), domain.clone()))
                .collect::<Vec<_>>();
            let instance = builder
                .instantiate("seismic.base", &self.base, &bindings, Vec::new())
                .unwrap_or_else(|error| {
                    panic!("private Seismic model instantiation failed: {error}")
                });
            let assignment_variables = self
                .assignment_variables
                .iter()
                .map(|variable| {
                    instance.variable(*variable).unwrap_or_else(|| {
                        panic!("solver fragment omitted an assignment interface variable")
                    })
                })
                .collect::<Vec<_>>();
            for values in exclusions {
                builder.constraint(Constraint::ForbiddenTuple {
                    variables: assignment_variables.clone(),
                    values: values.clone(),
                });
            }
            let model = Arc::new(builder.build().unwrap_or_else(|error| {
                panic!("private Seismic cursor model is invalid: {error}")
            }));
            let mapping = (0..self.variables.len())
                .map(VarId)
                .map(|variable| {
                    instance.variable(variable).unwrap_or_else(|| {
                        panic!("solver fragment omitted a base interface variable")
                    })
                })
                .collect();
            (model, mapping)
        }

        fn decode(&self, values: &[i64]) -> (&Implementation, RawAssignment) {
            let implementation = self
                .implementations
                .iter()
                .find(|implementation| values[implementation.selected.0] == 1)
                .unwrap_or_else(|| panic!("validated solver witness selected no implementation"));
            let decisions = implementation
                .decisions
                .iter()
                .map(|(decision, _, variable)| (*decision, values[variable.0]))
                .collect();
            (
                implementation,
                RawAssignment::new(implementation.index, decisions),
            )
        }

        fn exact(&self, implementation: &Implementation, assignment: &RawAssignment) -> bool {
            let mut values = PartialAssignment::new();
            for ((_, symbol, _), (_, value)) in
                implementation.decisions.iter().zip(assignment.decisions())
            {
                values.bind(*symbol, SymbolValue::Int(*value));
            }
            implementation.exact_filter.as_ref().is_none_or(|filter| {
                match filter.evaluate(&values) {
                    Ok(value) => value,
                    Err(EvalError::Unbound(symbol)) => panic!(
                        "decision-only predicate retained an unbound non-decision symbol: {symbol:?}"
                    ),
                    Err(
                        EvalError::DivisionByZero
                        | EvalError::NegativeNat
                        | EvalError::Unrepresentable,
                    ) => false,
                }
            })
        }
    }

    pub(super) struct Cursor<'a> {
        model: &'a Model,
        budget: SolverAllowance,
        started: Instant,
        work: u64,
        exclusions: Vec<Vec<i64>>,
        complete: bool,
        exhausted: bool,
    }

    impl Cursor<'_> {
        pub(super) fn next(&mut self) -> AssignmentStep {
            if self.complete {
                return AssignmentStep::Complete;
            }
            if self.exhausted {
                return AssignmentStep::BudgetExhausted(self.report());
            }
            loop {
                if self.over_budget() {
                    self.exhausted = true;
                    return AssignmentStep::BudgetExhausted(self.report());
                }
                let (model, mapping) = self.model.instantiate(&self.exclusions);
                let mut search = Search::new(
                    model,
                    Options {
                        algorithm: Algorithm::Exact,
                        policy: Policy::DepthFirst,
                        memoize: true,
                        decompose: true,
                        bounds: true,
                    },
                )
                .unwrap_or_else(|error| panic!("private Seismic search is invalid: {error}"));
                let remaining_work = self.budget.work.saturating_sub(self.work);
                let outcome = search
                    .advance(Limits {
                        work: remaining_work,
                        time: None,
                        memory_bytes: self.budget.memory_bytes.map(|bytes| {
                            let retained = self.retained_bytes();
                            usize::try_from(bytes.saturating_sub(retained)).unwrap_or(usize::MAX)
                        }),
                    })
                    .unwrap_or_else(|error| panic!("private Seismic search failed: {error}"));
                self.work = self.work.saturating_add(search.stats().work);
                let witness = match outcome {
                    Outcome::Optimal(solution) => Some(solution.values().to_vec()),
                    Outcome::Infeasible => {
                        self.complete = true;
                        return AssignmentStep::Complete;
                    }
                    Outcome::Incomplete(progress) => {
                        if let Some(solution) = progress.incumbent {
                            Some(solution.values().to_vec())
                        } else {
                            self.exhausted = true;
                            return AssignmentStep::BudgetExhausted(self.report());
                        }
                    }
                };
                let witness = witness.expect("solver outcome supplies a witness");
                let witness = mapping
                    .iter()
                    .map(|variable| witness[variable.0])
                    .collect::<Vec<_>>();
                let excluded = self
                    .model
                    .assignment_variables
                    .iter()
                    .map(|variable| witness[variable.0])
                    .collect::<Vec<_>>();
                self.exclusions.push(excluded);
                let (implementation, assignment) = self.model.decode(&witness);
                if self.model.exact(implementation, &assignment) {
                    return AssignmentStep::Assignment(FeasibleAssignment(assignment));
                }
            }
        }

        fn over_budget(&self) -> bool {
            self.work >= self.budget.work
                || self
                    .budget
                    .memory_bytes
                    .is_some_and(|limit| self.retained_bytes() > limit)
        }

        fn retained_bytes(&self) -> u64 {
            let exclusions = self
                .exclusions
                .iter()
                .map(|values| {
                    (std::mem::size_of::<Constraint>()
                        + values.capacity() * std::mem::size_of::<i64>()
                        + self.model.assignment_variables.capacity() * std::mem::size_of::<VarId>())
                        as u64
                })
                .sum::<u64>();
            self.model.retained_bytes.saturating_add(exclusions)
        }

        fn report(&self) -> CursorBudgetReport {
            CursorBudgetReport {
                work_units: self.work,
                elapsed_ms: u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX),
                memory_bytes: self.retained_bytes(),
            }
        }
    }

    fn guarded_total(arena: &mut ExprArena, predicate: BoolExpr) -> BoolExpr {
        let side_conditions = arena.side_conditions(AnyExpr::Bool(predicate));
        arena.and(side_conditions, predicate)
    }

    fn collect_decision_conjuncts(
        arena: &ExprArena,
        predicate: BoolExpr,
        output: &mut Vec<BoolExpr>,
    ) {
        if let NodeView::Binary {
            op: BinaryOp::And,
            lhs: AnyExpr::Bool(left),
            rhs: AnyExpr::Bool(right),
        } = arena.view(AnyExpr::Bool(predicate))
        {
            collect_decision_conjuncts(arena, left, output);
            collect_decision_conjuncts(arena, right, output);
            return;
        }
        if let NodeView::Nary {
            op: NaryOp::All,
            operands,
        } = arena.view(AnyExpr::Bool(predicate))
        {
            for operand in operands {
                let AnyExpr::Bool(term) = *operand else {
                    panic!("Bool All node contains a non-Bool operand");
                };
                collect_decision_conjuncts(arena, term, output);
            }
            return;
        }
        let free = arena.free_symbols(AnyExpr::Bool(predicate));
        if free
            .iter()
            .all(|symbol| matches!(arena.symbol_kind(*symbol), SymbolKind::Decision(_)))
        {
            output.push(predicate);
        }
    }

    struct PruningExporter<'a, 'b> {
        arena: &'a ExprArena,
        builder: &'b mut GenericBuilder,
        symbols: &'a HashMap<SymbolId, VarId>,
        bounds: &'a HashMap<VarId, (i64, i64)>,
        booleans: HashMap<BoolExpr, VarId>,
        boolean_support: HashMap<BoolExpr, bool>,
        affines: HashMap<AnyExpr, Option<Affine>>,
        indicators: HashSet<VarId>,
    }

    impl<'a, 'b> PruningExporter<'a, 'b> {
        fn new(
            arena: &'a ExprArena,
            builder: &'b mut GenericBuilder,
            symbols: &'a HashMap<SymbolId, VarId>,
            bounds: &'a HashMap<VarId, (i64, i64)>,
        ) -> Self {
            Self {
                arena,
                builder,
                symbols,
                bounds,
                booleans: HashMap::new(),
                boolean_support: HashMap::new(),
                affines: HashMap::new(),
                indicators: HashSet::new(),
            }
        }

        fn boolean(&mut self, expression: BoolExpr) -> Option<VarId> {
            if let Some(variable) = self.booleans.get(&expression) {
                return Some(*variable);
            }
            if !self.supports_boolean(expression) {
                return None;
            }
            let variable = self
                .boolean_uncached(expression)
                .unwrap_or_else(|| panic!("Boolean export diverged from its exact preflight"));
            self.booleans.insert(expression, variable);
            Some(variable)
        }

        fn supports_boolean(&mut self, expression: BoolExpr) -> bool {
            if let Some(supported) = self.boolean_support.get(&expression) {
                return *supported;
            }
            let supported = match self.arena.view(AnyExpr::Bool(expression)) {
                NodeView::BoolConst(_) => true,
                NodeView::Unary {
                    op: UnaryOp::Not,
                    operand: AnyExpr::Bool(input),
                } => self.supports_boolean(input),
                NodeView::Binary {
                    op: BinaryOp::And | BinaryOp::Or | BinaryOp::Implies | BinaryOp::Iff,
                    lhs: AnyExpr::Bool(left),
                    rhs: AnyExpr::Bool(right),
                } => self.supports_boolean(left) && self.supports_boolean(right),
                NodeView::Nary { op, operands } if matches!(op, NaryOp::All | NaryOp::Any) => {
                    let operands = operands.to_vec();
                    operands.iter().all(|operand| match operand {
                        AnyExpr::Bool(value) => self.supports_boolean(*value),
                        _ => false,
                    })
                }
                NodeView::Cmp { op, lhs, rhs } => match (self.affine(lhs), self.affine(rhs)) {
                    (Some(left), Some(right)) => self.comparison_supported(op, left, right),
                    _ => false,
                },
                NodeView::In { operand, values } => {
                    let values = values.to_vec();
                    match self.affine(operand) {
                        Some(operand) => values.iter().all(|value| {
                            self.comparison_supported(
                                CmpOp::Eq,
                                operand.clone(),
                                Affine::constant(i128::from(*value)),
                            )
                        }),
                        None => false,
                    }
                }
                _ => false,
            };
            self.boolean_support.insert(expression, supported);
            supported
        }

        fn boolean_uncached(&mut self, expression: BoolExpr) -> Option<VarId> {
            let variable = match self.arena.view(AnyExpr::Bool(expression)) {
                NodeView::BoolConst(value) => {
                    self.indicator("bool.constant", Domain::singleton(i64::from(value)))
                }
                NodeView::Unary {
                    op: UnaryOp::Not,
                    operand: AnyExpr::Bool(input),
                } => {
                    let input = self.boolean(input)?;
                    let output = self.indicator("bool.not", Domain::boolean());
                    self.builder
                        .constraint(Constraint::BoolNot { output, input });
                    output
                }
                NodeView::Binary {
                    op,
                    lhs: AnyExpr::Bool(left),
                    rhs: AnyExpr::Bool(right),
                } => {
                    let left = self.boolean(left)?;
                    let right = self.boolean(right)?;
                    match op {
                        BinaryOp::And => self.bool_relation(true, vec![left, right]),
                        BinaryOp::Or => self.bool_relation(false, vec![left, right]),
                        BinaryOp::Implies => {
                            let not_left = self.indicator("implies.not", Domain::boolean());
                            self.builder.constraint(Constraint::BoolNot {
                                output: not_left,
                                input: left,
                            });
                            self.bool_relation(false, vec![not_left, right])
                        }
                        BinaryOp::Iff => {
                            let equal = self.indicator("bool.iff", Domain::boolean());
                            let le = self.reified_le(left, right, 0)?;
                            let ge = self.reified_le(right, left, 0)?;
                            self.builder.constraint(Constraint::BoolAnd {
                                output: equal,
                                inputs: vec![le, ge],
                            });
                            equal
                        }
                        _ => return None,
                    }
                }
                NodeView::Nary { op, operands } if matches!(op, NaryOp::All | NaryOp::Any) => {
                    let operands = operands.to_vec();
                    let inputs = operands
                        .iter()
                        .map(|operand| match operand {
                            AnyExpr::Bool(value) => self.boolean(*value),
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>()?;
                    self.bool_relation(op == NaryOp::All, inputs)
                }
                NodeView::Cmp { op, lhs, rhs } => {
                    let left = self.affine(lhs)?;
                    let right = self.affine(rhs)?;
                    self.comparison(op, left, right)?
                }
                NodeView::In { operand, values } => {
                    let values = values.to_vec();
                    let operand = self.affine(operand)?;
                    let equals = values
                        .iter()
                        .map(|value| {
                            self.comparison(
                                CmpOp::Eq,
                                operand.clone(),
                                Affine::constant(*value as i128),
                            )
                        })
                        .collect::<Option<Vec<_>>>()?;
                    self.bool_relation(false, equals)
                }
                _ => return None,
            };
            Some(variable)
        }

        fn bool_relation(&mut self, and: bool, inputs: Vec<VarId>) -> VarId {
            let output =
                self.indicator(if and { "bool.and" } else { "bool.or" }, Domain::boolean());
            self.builder.constraint(if and {
                Constraint::BoolAnd { output, inputs }
            } else {
                Constraint::BoolOr { output, inputs }
            });
            output
        }

        fn indicator(&mut self, name: &'static str, domain: Domain) -> VarId {
            let variable = self.builder.variable(name, domain);
            self.indicators.insert(variable);
            variable
        }

        fn affine(&mut self, expression: AnyExpr) -> Option<Affine> {
            if let Some(value) = self.affines.get(&expression) {
                return value.clone();
            }
            let value = self.affine_uncached(expression);
            self.affines.insert(expression, value.clone());
            value
        }

        fn affine_uncached(&mut self, expression: AnyExpr) -> Option<Affine> {
            match self.arena.view(expression) {
                NodeView::NatConst(value) => Some(Affine::constant(value as i128)),
                NodeView::IntConst(value) => Some(Affine::constant(value as i128)),
                NodeView::Symbol(symbol) => {
                    self.symbols.get(&symbol).copied().map(Affine::variable)
                }
                NodeView::Unary {
                    op: UnaryOp::NatFromInt | UnaryOp::IntFromNat,
                    operand,
                } => self.affine(operand),
                NodeView::Binary {
                    op: op @ (BinaryOp::Add | BinaryOp::Sub),
                    lhs,
                    rhs,
                } => {
                    let mut left = self.affine(lhs)?;
                    left.add(self.affine(rhs)?, if op == BinaryOp::Add { 1 } else { -1 })?;
                    Some(left)
                }
                NodeView::Binary {
                    op: BinaryOp::Mul,
                    lhs,
                    rhs,
                } => match (self.affine(lhs)?, self.affine(rhs)?) {
                    (mut value, Affine { constant, terms }) if terms.is_empty() => {
                        value.scale(constant)?;
                        Some(value)
                    }
                    (Affine { constant, terms }, mut value) if terms.is_empty() => {
                        value.scale(constant)?;
                        Some(value)
                    }
                    _ => None,
                },
                _ => None,
            }
        }

        fn comparison(&mut self, op: CmpOp, left: Affine, right: Affine) -> Option<VarId> {
            match op {
                CmpOp::Le => self.affine_le(left, right, 0),
                CmpOp::Lt => self.affine_le(left, right, -1),
                CmpOp::Ge => self.affine_le(right, left, 0),
                CmpOp::Gt => self.affine_le(right, left, -1),
                CmpOp::Eq => {
                    let le = self.affine_le(left.clone(), right.clone(), 0)?;
                    let ge = self.affine_le(right, left, 0)?;
                    Some(self.bool_relation(true, vec![le, ge]))
                }
                CmpOp::Ne => {
                    let equal = self.comparison(CmpOp::Eq, left, right)?;
                    let output = self.indicator("cmp.ne", Domain::boolean());
                    self.builder.constraint(Constraint::BoolNot {
                        output,
                        input: equal,
                    });
                    Some(output)
                }
            }
        }

        fn comparison_supported(&self, op: CmpOp, left: Affine, right: Affine) -> bool {
            match op {
                CmpOp::Le | CmpOp::Lt => self
                    .normalized_le(left, right, if op == CmpOp::Lt { -1 } else { 0 })
                    .is_some(),
                CmpOp::Ge | CmpOp::Gt => self
                    .normalized_le(right, left, if op == CmpOp::Gt { -1 } else { 0 })
                    .is_some(),
                CmpOp::Eq | CmpOp::Ne => {
                    self.normalized_le(left.clone(), right.clone(), 0).is_some()
                        && self.normalized_le(right, left, 0).is_some()
                }
            }
        }

        fn affine_le(&mut self, left: Affine, right: Affine, offset: i128) -> Option<VarId> {
            let (terms, rhs) = self.normalized_le(left, right, offset)?;
            let output = self.indicator("cmp.le", Domain::boolean());
            self.builder.constraint(Constraint::ReifiedLinearLe {
                indicator: output,
                terms,
                rhs,
            });
            Some(output)
        }

        fn normalized_le(
            &self,
            mut left: Affine,
            right: Affine,
            offset: i128,
        ) -> Option<(Vec<LinearTerm>, i128)> {
            left.add(right, -1)?;
            let rhs = offset.checked_sub(left.constant)?;
            let terms = left
                .terms
                .into_iter()
                .map(|(variable, coefficient)| {
                    i64::try_from(coefficient)
                        .ok()
                        .map(|coefficient| LinearTerm::new(variable, coefficient))
                })
                .collect::<Option<Vec<_>>>()?;
            if !self.linear_bounds_fit(&terms) {
                return None;
            }
            Some((terms, rhs))
        }

        fn linear_bounds_fit(&self, terms: &[LinearTerm]) -> bool {
            let mut lower = 0_i128;
            let mut upper = 0_i128;
            for term in terms {
                let (min, max) = match self.bounds.get(&term.variable).copied() {
                    Some(bounds) => bounds,
                    None if self.indicators.contains(&term.variable) => (0, 1),
                    None => return false,
                };
                let coefficient = i128::from(term.coefficient);
                let a = coefficient.checked_mul(i128::from(min));
                let b = coefficient.checked_mul(i128::from(max));
                let (Some(a), Some(b)) = (a, b) else {
                    return false;
                };
                let Some(next_lower) = lower.checked_add(a.min(b)) else {
                    return false;
                };
                let Some(next_upper) = upper.checked_add(a.max(b)) else {
                    return false;
                };
                lower = next_lower;
                upper = next_upper;
            }
            true
        }

        fn reified_le(&mut self, left: VarId, right: VarId, rhs: i128) -> Option<VarId> {
            self.affine_le(Affine::variable(left), Affine::variable(right), rhs)
        }
    }

    #[derive(Clone)]
    struct Affine {
        constant: i128,
        terms: BTreeMap<VarId, i128>,
    }

    impl Affine {
        fn constant(constant: i128) -> Self {
            Self {
                constant,
                terms: BTreeMap::new(),
            }
        }
        fn variable(variable: VarId) -> Self {
            Self {
                constant: 0,
                terms: BTreeMap::from([(variable, 1)]),
            }
        }
        fn add(&mut self, other: Self, scale: i128) -> Option<()> {
            self.constant = self
                .constant
                .checked_add(other.constant.checked_mul(scale)?)?;
            for (variable, coefficient) in other.terms {
                let next = self
                    .terms
                    .get(&variable)
                    .copied()
                    .unwrap_or(0)
                    .checked_add(coefficient.checked_mul(scale)?)?;
                if next == 0 {
                    self.terms.remove(&variable);
                } else {
                    self.terms.insert(variable, next);
                }
            }
            Some(())
        }
        fn scale(&mut self, scale: i128) -> Option<()> {
            self.constant = self.constant.checked_mul(scale)?;
            for coefficient in self.terms.values_mut() {
                *coefficient = coefficient.checked_mul(scale)?;
            }
            Some(())
        }
    }

    fn assert_value_sort(sort: SymbolSort, value: SymbolValue) {
        let valid = matches!(
            (sort, value),
            (SymbolSort::Nat, SymbolValue::Nat(_))
                | (SymbolSort::Int, SymbolValue::Int(_))
                | (
                    SymbolSort::Scalar(seismic_lang::types::DType::F32),
                    SymbolValue::F32(_)
                )
                | (
                    SymbolSort::Scalar(seismic_lang::types::DType::F16),
                    SymbolValue::F16(_)
                )
                | (
                    SymbolSort::Scalar(seismic_lang::types::DType::BF16),
                    SymbolValue::BF16(_)
                )
                | (
                    SymbolSort::Scalar(seismic_lang::types::DType::Bool),
                    SymbolValue::Bool(_)
                )
                | (
                    SymbolSort::Scalar(seismic_lang::types::DType::I32),
                    SymbolValue::I32(_)
                )
                | (
                    SymbolSort::Scalar(seismic_lang::types::DType::U32),
                    SymbolValue::U32(_)
                )
        );
        if !valid {
            panic!("target constant value does not match its symbol sort");
        }
    }
}
