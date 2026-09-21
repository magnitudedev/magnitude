//! Sealed plan spaces and complete solver assignments (package P1).
//!
//! `form_plan_space` is the only way to build a `PlanSpace`. It expands
//! occurrences (O1), applies the universal and catalog rules (S1), routes
//! (D1), forms kernels (K1), derives consequences (M1), and admits only
//! strategies carrying every seal. `PlanSpace::solver_model` exports the one
//! global solver model; `PlanSpace::resolve` consumes a `CompleteAssignment`
//! (constructible only by solver-result validation) and substitutes.

use crate::consequences::ClosedStrategy;
use crate::failure::{CompilerDefect, Package};
use crate::ids::{ChoiceVarId, OccurrenceId, PlanParamId, ResidenceId, StrategyId};
use crate::kernel::ExecutableDialect;
use crate::numerics::{self, NumericalEvidence};
use crate::occurrence::OccurrenceForest;
use crate::physical::PhysicalPlan;
use crate::residence::{ResidenceChoice, StorageScope};
use crate::strategy::{MappingCatalog, NativeFactDomain};
use crate::target::EffectiveTargetProfile;
use magnitude_solver::scheduling::{arena_offsets, ArenaPacking};
use seismic_lang::logical::{
    EffectiveTargetIdentity, IdVec, LogicalIdentity, LogicalProgram,
};
use seismic_lang::precision::{NumericalAssessment, PrecisionPolicy};
use seismic_lang::sym::Sym;
use seismic_lang::types::RuntimeExtentId;
use std::collections::BTreeMap;

/// The complete plan space of one logical program on one target.
pub struct PlanSpace<'l, D: ExecutableDialect> {
    logical: &'l LogicalProgram,
    forest: OccurrenceForest<'l>,
    profile: EffectiveTargetProfile,
    strategies: IdVec<OccurrenceId, IdVec<StrategyId, ClosedStrategy<D>>>,
    /// Native resource facts the backend catalog declared (validated by N1
    /// against the domains retained here).
    native_fact_domains: Vec<NativeFactDomain>,
}

impl seismic_lang::logical::IdIndex for StrategyId {
    fn from_index(index: usize) -> Self {
        StrategyId(index as u32)
    }
    fn index(self) -> usize {
        self.0 as usize
    }
}

/// Build the complete plan space. Total over sealed inputs: infeasibility
/// belongs to the solver, toolchain/system failure to assembly; a defect is
/// a contradicted construction invariant of the named package.
pub fn form_plan_space<'l, D: ExecutableDialect>(
    logical: &'l LogicalProgram,
    profile: &EffectiveTargetProfile,
    catalog: &impl MappingCatalog<D>,
) -> Result<PlanSpace<'l, D>, crate::failure::CompilerDefect> {
    crate::formation::seal::form_plan_space(logical, profile, catalog)
}

impl<'l, D: ExecutableDialect> PlanSpace<'l, D> {
    pub(crate) fn seal(
        logical: &'l LogicalProgram,
        forest: OccurrenceForest<'l>,
        profile: EffectiveTargetProfile,
        strategies: IdVec<OccurrenceId, IdVec<StrategyId, ClosedStrategy<D>>>,
        native_fact_domains: Vec<NativeFactDomain>,
    ) -> PlanSpace<'l, D> {
        PlanSpace {
            logical,
            forest,
            profile,
            strategies,
            native_fact_domains,
        }
    }

    pub fn logical(&self) -> &'l LogicalProgram {
        self.logical
    }
    pub fn identity(&self) -> LogicalIdentity {
        self.logical.identity
    }
    pub fn profile(&self) -> &EffectiveTargetProfile {
        &self.profile
    }
    pub fn forest(&self) -> &OccurrenceForest<'l> {
        &self.forest
    }
    pub fn strategies(&self, occurrence: OccurrenceId) -> &IdVec<StrategyId, ClosedStrategy<D>> {
        &self.strategies[occurrence]
    }
    /// The whole strategy table, keyed by occurrence (formation and the
    /// solver export read it; resolution consumes it).
    pub(crate) fn strategies_table(
        &self,
    ) -> &IdVec<OccurrenceId, IdVec<StrategyId, ClosedStrategy<D>>> {
        &self.strategies
    }
    /// Consume the space into its parts (the physical seal's substitution
    /// inputs).
    pub(crate) fn into_parts(
        self,
    ) -> (
        &'l LogicalProgram,
        OccurrenceForest<'l>,
        EffectiveTargetProfile,
        IdVec<OccurrenceId, IdVec<StrategyId, ClosedStrategy<D>>>,
        Vec<NativeFactDomain>,
    ) {
        (
            self.logical,
            self.forest,
            self.profile,
            self.strategies,
            self.native_fact_domains,
        )
    }

    /// The one global solver model view: every decision, constraint,
    /// residence choice, and cost, from the exact objects resolution
    /// substitutes.
    pub fn solver_model(&self, numerics: &NumericalContext<'_>) -> SolverModelView<'_> {
        crate::formation::seal::solver_model(self, numerics)
    }

    /// Resolve by substitution only. Consumes the space; infallible over a
    /// validated complete assignment.
    pub fn resolve(self, assignment: CompleteAssignment) -> PhysicalPlan<D> {
        crate::formation::seal::resolve(self, assignment)
    }
}

/// The caller's numerical policy and accepted evidence.
pub struct NumericalContext<'a> {
    pub precision: &'a PrecisionPolicy,
    pub evidence: &'a [crate::numerics::NumericalEvidence],
}

/// One decision of the global model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    /// Which strategy an occurrence selects; `None` when absorbed.
    Strategy { occurrence: OccurrenceId, options: Vec<StrategyId>, inactive_allowed: bool },
    /// A tuning parameter of one strategy.
    Tuning { occurrence: OccurrenceId, strategy: StrategyId, parameter: PlanParamId, lower: u64, upper: u64 },
    /// A residence scope choice.
    Scope { occurrence: OccurrenceId, strategy: StrategyId, residence: ResidenceId, var: ChoiceVarId, options: Vec<crate::residence::StorageScope> },
    /// A device-arena offset of one residence plane when active.
    ArenaOffset { occurrence: OccurrenceId, strategy: StrategyId, residence: ResidenceId, plane: u32 },
}

/// The solver-facing view of a plan space: decisions and the constraint and
/// cost facts of the exact sealed strategies. P1 exports this into the
/// `magnitude_solver` model.
pub struct SolverModelView<'s> {
    pub(crate) decisions: Vec<Decision>,
    pub(crate) facts: Vec<StrategyFactsRef<'s>>,
    pub(crate) arena_capacity: u64,
    /// The numerical context the view was exported under (cloned: `validate`
    /// composes the whole-plan assessment from it).
    pub(crate) precision: PrecisionPolicy,
    pub(crate) evidence: Vec<NumericalEvidence>,
    /// Runtime extent capacities, keyed by `RuntimeExtentId`: the value the
    /// `@runtime<N>` atom binds to for every resource and layout expression.
    pub(crate) runtime_capacities: BTreeMap<RuntimeExtentId, u64>,
    /// The identity components of the evidence key.
    pub(crate) logical: LogicalIdentity,
    pub(crate) target: EffectiveTargetIdentity,
    pub(crate) toolchain: String,
    pub(crate) workload: numerics::WorkloadFingerprint,
    pub(crate) entry: OccurrenceId,
}

/// One strategy's sealed facts referenced from the model view.
pub struct StrategyFactsRef<'s> {
    pub occurrence: OccurrenceId,
    pub strategy: StrategyId,
    pub consequences: &'s crate::consequences::StrategyConsequences,
    pub residences: &'s crate::residence::ResidenceGraph,
    pub routed: &'s crate::residence::RoutedStrategy,
    pub absorbed: &'s BTreeMap<OccurrenceId, u32>,
}

/// The solver variables P1 allocated for each decision (in decision order),
/// the optimality of the solve, and the arena certificate inputs: the
/// packing the offsets are decoded from, the residence each arena ordinal
/// names, and the leaf-bound variable each `(occurrence, strategy, leaf)`
/// materialized (the `@leaf<L>` atoms an evidence record may witness).
/// Constructed by `seismic_compiler::planning::plan`; consumed by
/// `validate`.
pub struct DecisionVariables {
    pub variables: Vec<Vec<magnitude_solver::VarId>>,
    /// Whether the solve proved optimality over the whole model.
    pub optimal: bool,
    /// The arena packing exactly as supplied to the solver's scheduling
    /// constraint; `arena_offsets` over the solved values yields the
    /// certificate.
    pub arena: ArenaPacking,
    /// Arena ordinal -> the residence it packs (one item per residence; its
    /// planes are laid out contiguously from the item's offset).
    pub arena_items: Vec<(OccurrenceId, StrategyId, ResidenceId)>,
    /// The leaf-bound variable of every `(occurrence, strategy, leaf)` the
    /// model materialized, in materialization order.
    pub leaves: Vec<(
        OccurrenceId,
        StrategyId,
        crate::ids::CanonicalLeafId,
        magnitude_solver::VarId,
    )>,
}

impl<'s> SolverModelView<'s> {
    pub fn decisions(&self) -> &[Decision] {
        &self.decisions
    }
    pub fn facts(&self) -> &[StrategyFactsRef<'s>] {
        &self.facts
    }
    pub fn arena_capacity(&self) -> u64 {
        self.arena_capacity
    }

    /// The only constructor of `CompleteAssignment`: validates that the
    /// solver's feasible solution assigns exactly one value to every
    /// decision of this view and names no foreign or duplicate decision.
    pub fn validate(
        &self,
        solution: &magnitude_solver::FeasibleSolution,
        variables: &DecisionVariables,
    ) -> Result<CompleteAssignment, CompilerDefect> {
        let values = solution.values();
        if variables.variables.len() != self.decisions.len() {
            return Err(CompilerDefect::new(
                Package::P1,
                "the decision variables do not cover the model's decisions exactly",
            ));
        }
        let mut seen: BTreeMap<magnitude_solver::VarId, ()> = BTreeMap::new();
        for vars in &variables.variables {
            for &var in vars {
                if var.0 >= values.len() || seen.insert(var, ()).is_some() {
                    return Err(CompilerDefect::new(
                        Package::P1,
                        "the decision variables name a foreign or duplicated solver variable",
                    ));
                }
            }
        }
        let read = |var: magnitude_solver::VarId| -> Result<i64, CompilerDefect> {
            values
                .get(var.0)
                .copied()
                .ok_or_else(|| CompilerDefect::new(Package::P1, "the solution omits a decision"))
        };

        let mut selections: BTreeMap<OccurrenceId, Option<StrategyId>> = BTreeMap::new();
        let mut symbols: BTreeMap<String, u64> = BTreeMap::new();
        let mut scopes: BTreeMap<(OccurrenceId, StrategyId, ChoiceVarId), crate::residence::StorageScope> =
            BTreeMap::new();
        let mut next = 0usize;
        for decision in &self.decisions {
            let vars = &variables.variables[next];
            next += 1;
            match decision {
                Decision::Strategy {
                    occurrence,
                    options,
                    inactive_allowed,
                } => {
                    let ordinal = read(vars[0])?;
                    let selected = if ordinal == INACTIVE {
                        if !*inactive_allowed {
                            return Err(CompilerDefect::new(
                                Package::P1,
                                "the solution inactivated an occurrence that must stay active",
                            ));
                        }
                        None
                    } else {
                        let index = usize::try_from(ordinal).map_err(|_| {
                            CompilerDefect::new(
                                Package::P1,
                                "the solution selects a strategy ordinal outside the domain",
                            )
                        })?;
                        Some(*options.get(index).ok_or_else(|| {
                            CompilerDefect::new(
                                Package::P1,
                                "the solution selects a strategy ordinal outside the domain",
                            )
                        })?)
                    };
                    if selections.insert(*occurrence, selected).is_some() {
                        return Err(CompilerDefect::new(
                            Package::P1,
                            "the model exports one strategy decision per occurrence",
                        ));
                    }
                }
                Decision::Tuning {
                    occurrence,
                    strategy,
                    parameter,
                    lower,
                    upper,
                } => {
                    let value = read(vars[0])?;
                    let value = u64::try_from(value).map_err(|_| {
                        CompilerDefect::new(
                            Package::P1,
                            "a solved tuning value is negative",
                        )
                    })?;
                    if value < *lower || value > *upper {
                        return Err(CompilerDefect::new(
                            Package::P1,
                            "a solved tuning value lies outside its declared domain",
                        ));
                    }
                    let name = self
                        .facts
                        .iter()
                        .find(|facts| facts.occurrence == *occurrence && facts.strategy == *strategy)
                        .and_then(|facts| facts.consequences.tuning.get(*parameter))
                        .map(|declaration| declaration.name.clone())
                        .ok_or_else(|| {
                            CompilerDefect::new(
                                Package::P1,
                                "a tuning decision names a parameter absent from its strategy",
                            )
                        })?;
                    symbols.insert(tuning_symbol(*occurrence, *strategy, &name), value);
                }
                Decision::Scope {
                    occurrence,
                    strategy,
                    residence,
                    var,
                    options,
                } => {
                    // The residence names the placement the choice var
                    // belongs to; the decode key is the identity triple.
                    let _ = residence;
                    let index = usize::try_from(read(vars[0])?).map_err(|_| {
                        CompilerDefect::new(Package::P1, "a solved scope choice is negative")
                    })?;
                    let scope = *options.get(index).ok_or_else(|| {
                        CompilerDefect::new(
                            Package::P1,
                            "a solved scope choice lies outside its options",
                        )
                    })?;
                    scopes.insert((*occurrence, *strategy, *var), scope);
                }
                Decision::ArenaOffset {
                    occurrence,
                    strategy,
                    residence,
                    plane,
                } => {
                    // Offsets are decoded from the arena certificate below;
                    // the decision records the domain only.
                    let _ = (occurrence, strategy, residence, plane);
                }
            }
        }

        // The `@runtime<N>` atoms bind to their checked capacities for every
        // expression resolution evaluates (resources, geometry, layouts).
        for (id, capacity) in &self.runtime_capacities {
            symbols.insert(format!("@runtime{}", id.0), *capacity);
        }

        // The `@leaf<L>` atoms: bound by the candidate iff every selected
        // strategy that materialized the leaf's bound variable solves it to
        // the same value (a disagreement leaves the atom unbound, so a
        // record witnessing it cannot qualify).
        let mut leaf_bindings: BTreeMap<crate::ids::CanonicalLeafId, (bool, Option<u64>)> =
            BTreeMap::new();
        for (occurrence, strategy, leaf, var) in &variables.leaves {
            if selections.get(occurrence) != Some(&Some(*strategy)) {
                continue;
            }
            let value = read(*var)?;
            let value = u64::try_from(value).map_err(|_| {
                CompilerDefect::new(Package::P1, "a solved leaf-bound value is negative")
            })?;
            let (seen, agreed) = leaf_bindings.entry(*leaf).or_insert((false, None));
            if !*seen {
                *seen = true;
                *agreed = Some(value);
            } else if *agreed != Some(value) {
                *agreed = None;
            }
        }
        for (leaf, (seen, agreed)) in leaf_bindings {
            if let (true, Some(value)) = (seen, agreed) {
                symbols.insert(format!("@leaf{}", leaf.0), value);
            }
        }

        // Arena certificate: every active item receives an offset; each
        // plane of its residence is laid out contiguously, aligned per plane.
        let solved = SolvedValues {
            symbols: symbols.clone(),
        };
        let certified = arena_offsets(&variables.arena, values)
            .map_err(|error| CompilerDefect::new(Package::P1, error.to_string()))?
            .ok_or_else(|| {
                CompilerDefect::new(
                    Package::P1,
                    "the feasible assignment has no arena packing certificate",
                )
            })?;
        let mut offsets: BTreeMap<(OccurrenceId, StrategyId, ResidenceId, u32), u64> =
            BTreeMap::new();
        for (ordinal, (occurrence, strategy, residence)) in
            variables.arena_items.iter().enumerate()
        {
            let Some(facts) = self
                .facts
                .iter()
                .find(|facts| facts.occurrence == *occurrence && facts.strategy == *strategy)
            else {
                return Err(CompilerDefect::new(
                    Package::P1,
                    "an arena item names a strategy absent from the model",
                ));
            };
            let record = facts.residences.residence(*residence);
            let scope_active = match &record.choice {
                ResidenceChoice::Fixed(_) => true,
                ResidenceChoice::SolverChoice { var, .. } => scopes
                    .get(&(*occurrence, *strategy, *var))
                    .is_some_and(|scope| *scope == StorageScope::DeviceArena),
            };
            if selections.get(occurrence) != Some(&Some(*strategy)) || !scope_active {
                continue;
            }
            let base = certified.get(&(ordinal as u32)).copied().ok_or_else(|| {
                CompilerDefect::new(
                    Package::P1,
                    "the arena certificate omits an active arena residence",
                )
            })?;
            let mut prefix = base;
            for (plane, plan) in record.planes.iter().enumerate() {
                let plane = u32::try_from(plane)
                    .map_err(|_| CompilerDefect::new(Package::P1, "a plane index exceeds u32"))?;
                offsets.insert((*occurrence, *strategy, *residence, plane), prefix);
                let bytes = solved.eval(&plan.bytes);
                let alignment = plan.alignment.max(1);
                let aligned = bytes
                    .checked_add(alignment - 1)
                    .and_then(|value| value.checked_div(alignment).map(|v| v * alignment))
                    .ok_or_else(|| {
                        CompilerDefect::new(
                            Package::P1,
                            "a residence plane's aligned bytes exceed u64 under capacity",
                        )
                    })?;
                prefix = prefix.checked_add(aligned).ok_or_else(|| {
                    CompilerDefect::new(
                        Package::P1,
                        "a residence's arena extent overflows u64",
                    )
                })?;
            }
        }

        // Proof-carrying resolution facts, built once here and consumed
        // positionally by the seal: the descent spine, the dense scope
        // table, and the dense arena offset table.
        let mut spine: Vec<(OccurrenceId, StrategyId)> = Vec::new();
        self.walk_spine(self.entry, &selections, &mut spine)?;
        for facts in &self.facts {
            if selections.get(&facts.occurrence) == Some(&Some(facts.strategy)) {
                for child in facts.absorbed.keys() {
                    if selections.get(child) != None {
                        return Err(CompilerDefect::new(
                            Package::P1,
                            "an absorbed child occurrence was left active",
                        ));
                    }
                }
            }
        }
        let scope_table = self.scope_table(&scopes);
        let offset_table = self.offset_table(&offsets);
        let resolved = ResolvedFacts {
            spine,
            scopes: scope_table,
            offsets: offset_table,
        };

        let numerical = self.composed_assessment(&selections, &symbols)?;
        Ok(CompleteAssignment::new(
            selections,
            solved,
            scopes,
            offsets,
            resolved,
            variables.optimal,
            numerical,
            solution.cost(),
        ))
    }

    /// The resolution's descent spine: the entry first, then every retained
    /// child of a selected strategy in schedule order, recursively, with
    /// statically empty repeats pruned exactly as the seal prunes them. The
    /// seal consumes the spine positionally, so every entry proves the
    /// selection of an occurrence the resolution reaches.
    fn walk_spine(
        &self,
        occurrence: OccurrenceId,
        selections: &BTreeMap<OccurrenceId, Option<StrategyId>>,
        spine: &mut Vec<(OccurrenceId, StrategyId)>,
    ) -> Result<(), CompilerDefect> {
        let strategy = selections
            .get(&occurrence)
            .copied()
            .flatten()
            .ok_or_else(|| {
                CompilerDefect::new(
                    Package::P1,
                    format!(
                        "occurrence {} is reached by the resolution but has no selected strategy",
                        occurrence.0
                    ),
                )
            })?;
        spine.push((occurrence, strategy));
        let facts = self
            .facts
            .iter()
            .find(|facts| facts.occurrence == occurrence && facts.strategy == strategy)
            .ok_or_else(|| {
                CompilerDefect::new(
                    Package::P1,
                    "the selection names a strategy absent from the model",
                )
            })?;
        self.walk_spine_schedule(facts.routed.shape().schedule(), selections, spine)
    }

    fn walk_spine_schedule(
        &self,
        schedule: &crate::strategy::ShapeSchedule,
        selections: &BTreeMap<OccurrenceId, Option<StrategyId>>,
        spine: &mut Vec<(OccurrenceId, StrategyId)>,
    ) -> Result<(), CompilerDefect> {
        use crate::strategy::ShapeStep;
        for step in &schedule.steps {
            match step {
                ShapeStep::Call { occurrence, .. } => {
                    self.walk_spine(*occurrence, selections, spine)?
                }
                ShapeStep::If {
                    then_schedule,
                    else_schedule,
                    ..
                } => {
                    self.walk_spine_schedule(then_schedule, selections, spine)?;
                    self.walk_spine_schedule(else_schedule, selections, spine)?;
                }
                ShapeStep::Repeat { bound, body, .. } => {
                    if bound.as_static() != Some(0) {
                        self.walk_spine_schedule(body, selections, spine)?;
                    }
                }
                ShapeStep::Launch(_) | ShapeStep::Guard { .. } | ShapeStep::PullCounterReset { .. } => {
                }
            }
        }
        Ok(())
    }

    /// The dense scope table `[occurrence][strategy][choice var]`, every
    /// cell filled from its decoded scope decision in ChoiceVarId order
    /// (every SolverChoice residence exports one decision, and each choice
    /// var is used exactly once, so the cells cover the table completely).
    fn scope_table(
        &self,
        scopes: &BTreeMap<(OccurrenceId, StrategyId, ChoiceVarId), crate::residence::StorageScope>,
    ) -> Vec<Vec<Vec<crate::residence::StorageScope>>> {
        let occurrence_count = self
            .decisions
            .iter()
            .filter(|decision| matches!(decision, Decision::Strategy { .. }))
            .count();
        let mut table: Vec<Vec<Vec<crate::residence::StorageScope>>> = vec![Vec::new(); occurrence_count];
        for facts in &self.facts {
            let occurrence = facts.occurrence.0 as usize;
            while table[occurrence].len() <= facts.strategy.0 as usize {
                table[occurrence].push(Vec::new());
            }
            let mut choices = Vec::with_capacity(facts.routed.residences().choice_vars() as usize);
            for (var, _) in facts.routed.residences().iter().filter_map(|(residence, record)| {
                match &record.choice {
                    ResidenceChoice::SolverChoice { var, .. } => Some((*var, residence)),
                    ResidenceChoice::Fixed(_) => None,
                }
            }) {
                choices.push(scopes[&(facts.occurrence, facts.strategy, var)]);
            }
            table[occurrence][facts.strategy.0 as usize] = choices;
        }
        table
    }

    /// The dense arena offset table
    /// `[occurrence][strategy][residence][plane]`: a certified offset for
    /// every plane of a placement the packing certified, and an unread zero
    /// for every cell the seal never reads (only an Arena-placed storage
    /// reads its cell, and Arena placement proves the certification).
    fn offset_table(
        &self,
        offsets: &BTreeMap<(OccurrenceId, StrategyId, ResidenceId, u32), u64>,
    ) -> Vec<Vec<Vec<Vec<u64>>>> {
        let occurrence_count = self
            .decisions
            .iter()
            .filter(|decision| matches!(decision, Decision::Strategy { .. }))
            .count();
        let mut table: Vec<Vec<Vec<Vec<u64>>>> = vec![Vec::new(); occurrence_count];
        for facts in &self.facts {
            let occurrence = facts.occurrence.0 as usize;
            while table[occurrence].len() <= facts.strategy.0 as usize {
                table[occurrence].push(Vec::new());
            }
            let mut residences = Vec::with_capacity(facts.routed.residences().len());
            for (residence, record) in facts.routed.residences().iter() {
                let mut planes = vec![0u64; record.planes.len()];
                for plane in 0..record.planes.len() as u32 {
                    if let Some(offset) =
                        offsets.get(&(facts.occurrence, facts.strategy, residence, plane))
                    {
                        planes[plane as usize] = *offset;
                    }
                }
                residences.push(planes);
            }
            table[occurrence][facts.strategy.0 as usize] = residences;
        }
        table
    }

    /// The whole-plan numerical assessment of one complete selection, and
    /// the assignment-level qualification backstop: when a selected
    /// strategy's transfer requires evidence, some identity-matching record
    /// must answer `numerics::evidence_qualifies` for exactly this
    /// selection and its solved symbols.
    fn composed_assessment(
        &self,
        selections: &BTreeMap<OccurrenceId, Option<StrategyId>>,
        symbols: &BTreeMap<String, u64>,
    ) -> Result<NumericalAssessment, CompilerDefect> {
        let values = |name: &str| symbols.get(name).copied();
        if let Some(record) = self.evidence.iter().find(|record| {
            self.identity_matches(record)
                && numerics::evidence_qualifies(record, &self.precision, selections, &values)
        }) {
            return Ok(record.assessment.clone());
        }
        let runtime = |id: RuntimeExtentId| self.runtime_capacities.get(&id).copied();
        let mut transfers = Vec::new();
        let mut requires_evidence = false;
        for facts in &self.facts {
            if selections.get(&facts.occurrence) == Some(&Some(facts.strategy)) {
                if matches!(
                    numerics::satisfies_policy(
                        &facts.consequences.numerical,
                        &self.precision,
                        &runtime
                    ),
                    numerics::PolicyDecision::RequiresEvidence { .. }
                ) {
                    requires_evidence = true;
                }
                transfers.push(facts.consequences.numerical.clone());
            }
        }
        if requires_evidence {
            return Err(CompilerDefect::new(
                Package::P1,
                "the solved assignment selects a numerical transfer requiring evidence that no \
                 identity-matching record qualifies for this selection and its solved symbols",
            ));
        }
        if transfers
            .iter()
            .all(|transfer| matches!(transfer, numerics::NumericalTransfer::Exact))
        {
            return Ok(NumericalAssessment::exact());
        }
        let composed = numerics::compose_all(&transfers);
        Ok(match &self.precision {
            PrecisionPolicy::Unconstrained => NumericalAssessment::unknown(format!(
                "unconstrained compilation selected numerical transfer {composed:?}"
            )),
            PrecisionPolicy::Exact => NumericalAssessment::unknown(
                "a non-exact transfer survived exact numerical planning",
            ),
            PrecisionPolicy::Bounded { .. } => {
                let mut relative = 0.0;
                let mut absolute = 0.0;
                for selected in &transfers {
                    let Some(bound) = numerics::analytical_bound(selected, &runtime) else {
                        return Ok(NumericalAssessment::unknown(format!(
                            "selected transfer requires qualification but no matching assessment was retained: {selected:?}"
                        )));
                    };
                    relative += bound.relative;
                    absolute += bound.absolute;
                }
                NumericalAssessment::proven(
                    self.precision.clone(),
                    format!(
                        "whole-plan analytical envelope proved: relative {relative:e}, absolute {absolute:e}"
                    ),
                )
                .unwrap_or_else(NumericalAssessment::unknown)
            }
        })
    }

    /// Whether one evidence record carries the non-assignment identity of
    /// this compilation (logical program, workload, target, toolchain); the
    /// assignment and policy components are decided by
    /// `numerics::evidence_qualifies`.
    fn identity_matches(&self, record: &NumericalEvidence) -> bool {
        record.key.logical == self.logical
            && record.key.workload == self.workload
            && record.key.target == self.target
            && record.key.toolchain.0 == self.toolchain
    }
}

/// The sentinel strategy value of an inactivated occurrence.
pub(crate) const INACTIVE: i64 = -1;

/// The solver symbol of one strategy's tuning parameter: strategy-local
/// declaration names are qualified by their `(occurrence, strategy)`
/// identity so the one global model never conflates two strategies'
/// declarations of the same name. The identical function qualifies the
/// atom on the resolution side (`formation::seal::qualify`); the reserved
/// `@`-prefixed atom families are global and never qualified.
pub(crate) fn tuning_symbol(occurrence: OccurrenceId, strategy: StrategyId, name: &str) -> String {
    format!("{}.{}.{}", occurrence.0, strategy.0, name)
}

/// The solved values a resolution substitutes: qualified tuning parameter
/// names and `@runtime<N>` capacities. Every atom an expression names was
/// declared and every declared atom is solved.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SolvedValues {
    pub symbols: BTreeMap<String, u64>,
}

impl SolvedValues {
    /// Evaluate one size/cost expression. Infallible: every symbol the
    /// expression names was declared and every declared symbol is solved.
    pub fn eval(&self, expr: &Sym) -> u64 {
        let value = expr.eval(&|name| {
            self.symbols
                .get(name)
                .and_then(|value| i64::try_from(*value).ok())
        });
        match value.and_then(|value| u64::try_from(value).ok()) {
            Some(value) => value,
            // The contract places totality on the caller: every symbol a
            // passed expression names was declared, and `validate` binds
            // every declared symbol. The signature is frozen, so an
            // out-of-contract expression has no representable answer.
            None => unreachable!(
                "the expression `{expr}` names an unsolved symbol or exceeds the size domain"
            ),
        }
    }
}

/// The proof-carrying resolution facts, built once by `validate` and
/// consumed positionally by the seal: no lookups, no fallible branches.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedFacts {
    /// The resolution's descent spine: the entry first, then every retained
    /// child of a selected strategy in schedule order, recursively (with
    /// statically empty repeats pruned exactly as the seal prunes them).
    /// Each entry proves the selection of an occurrence the resolution
    /// reaches; the seal takes entries in order as it descends.
    pub(crate) spine: Vec<(OccurrenceId, StrategyId)>,
    /// Dense scope choices, `[occurrence][strategy][choice var]`: every
    /// cell filled from a decoded scope decision in ChoiceVarId order.
    pub(crate) scopes: Vec<Vec<Vec<crate::residence::StorageScope>>>,
    /// Dense arena offsets, `[occurrence][strategy][residence][plane]`: a
    /// certified offset for every plane of a placement the arena packing
    /// certified, and an unread zero for every cell the seal never reads
    /// (only an Arena-placed storage reads its cell, and Arena placement
    /// proves the certification).
    pub(crate) offsets: Vec<Vec<Vec<Vec<u64>>>>,
}

/// One complete assignment: one value for every decision. `PartialEq`
/// only: the retained numerical assessment carries tolerance floats.
#[derive(Clone, Debug, PartialEq)]
pub struct CompleteAssignment {
    selections: BTreeMap<OccurrenceId, Option<StrategyId>>,
    values: SolvedValues,
    scopes: BTreeMap<(OccurrenceId, StrategyId, ChoiceVarId), crate::residence::StorageScope>,
    offsets: BTreeMap<(OccurrenceId, StrategyId, ResidenceId, u32), u64>,
    resolved: ResolvedFacts,
    optimal: bool,
    numerical: NumericalAssessment,
    estimated_cost: u64,
}

impl CompleteAssignment {
    pub(crate) fn new(
        selections: BTreeMap<OccurrenceId, Option<StrategyId>>,
        values: SolvedValues,
        scopes: BTreeMap<(OccurrenceId, StrategyId, ChoiceVarId), crate::residence::StorageScope>,
        offsets: BTreeMap<(OccurrenceId, StrategyId, ResidenceId, u32), u64>,
        resolved: ResolvedFacts,
        optimal: bool,
        numerical: NumericalAssessment,
        estimated_cost: u64,
    ) -> CompleteAssignment {
        CompleteAssignment {
            selections,
            values,
            scopes,
            offsets,
            resolved,
            optimal,
            numerical,
            estimated_cost,
        }
    }

    /// The proof-carrying resolution facts the seal consumes positionally.
    pub(crate) fn resolved(&self) -> &ResolvedFacts {
        &self.resolved
    }

    pub fn selections(&self) -> &BTreeMap<OccurrenceId, Option<StrategyId>> {
        &self.selections
    }
    pub fn values(&self) -> &SolvedValues {
        &self.values
    }
    pub fn scopes(
        &self,
    ) -> &BTreeMap<(OccurrenceId, StrategyId, ChoiceVarId), crate::residence::StorageScope> {
        &self.scopes
    }
    pub fn offsets(&self) -> &BTreeMap<(OccurrenceId, StrategyId, ResidenceId, u32), u64> {
        &self.offsets
    }
    pub fn optimal(&self) -> bool {
        self.optimal
    }
    pub fn numerical(&self) -> &NumericalAssessment {
        &self.numerical
    }
    /// The solved model's exact objective value for this assignment.
    pub fn estimated_cost(&self) -> u64 {
        self.estimated_cost
    }
}
