//! One global conditional planning model and one budgeted solve.
//!
//! The model includes implementation, physical strategy, tuning, child
//! activation, dispatch/resources, storage activation/interference/offsets,
//! capability, safety, numerical policy, and cost. The solver is the sole
//! assignment authority. There is no post-solve rejection: resolution accepts
//! only the solver's feasible assignment.

use magnitude_solver::{
    model::{Arithmetic, Constraint, Cost, LinearTerm, Literal},
    Algorithm, Budget as SolverBudget, Budgeted, Domain, FeasibleAssignment, FeasibleSolution,
    Limits as SolverLimits, Model, ModelBuilder, Options, Search, VarId,
};
use seismic_lang::{
    logical::{ChoiceId, LogicalProgram},
    precision::{NumericalAssessment, PrecisionPolicy},
    sym::{Atom, Sym},
};
use seismic_realization::executable::{
    EffectiveTargetProfile, ExecutableDialect, FeasiblePlanAssignment, InvariantReport,
    LaunchFacts, PhysicalStorageTemplateId, PlanFamily, ResolvedPlan, ScheduleStepTemplate,
    StorageActivation,
};
use std::{collections::BTreeMap, sync::Arc, time::Duration};

const INACTIVE: i64 = -1;
/// Upper bound of arena offsets in the model; the arena constraint is the
/// real limit, this only bounds the variable domain.
const OFFSET_MAX: i64 = i32::MAX as i64;

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
    CompilerBug(String),
    Solver(String),
}
impl std::fmt::Display for PlanningFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Infeasible => write!(f, "planning is infeasible"),
            Self::CompilerBug(reason) => write!(f, "compiler bug: {reason}"),
            Self::Solver(reason) => write!(f, "solver defect: {reason}"),
        }
    }
}
impl std::error::Error for PlanningFailure {}

impl From<InvariantReport> for PlanningFailure {
    fn from(report: InvariantReport) -> Self {
        Self::CompilerBug(report.0)
    }
}

pub struct Context<'a> {
    pub target: &'a EffectiveTargetProfile,
    pub precision: &'a PrecisionPolicy,
    pub numerical_evidence: &'a [NumericalEvidence],
}

/// Accepted numerical evidence, keyed to the complete logical / target /
/// assignment / precision identity.
#[derive(Clone, Debug)]
pub struct NumericalEvidence {
    pub logical: seismic_lang::logical::LogicalIdentity,
    pub entry: String,
    pub target: String,
    pub capability_fingerprint: String,
    pub toolchain_fingerprint: String,
    pub selections: BTreeMap<ChoiceId, (u32, u32)>,
    pub symbols: BTreeMap<String, i64>,
    pub assessment: NumericalAssessment,
}

/// One solve of the whole family: build the global model, obtain a feasible
/// assignment, spend the optimization budget, and resolve the best incumbent.
/// Resolution performs no legality check.
pub fn plan<D: ExecutableDialect>(
    logical: &LogicalProgram,
    family: &PlanFamily<D>,
    context: &Context<'_>,
    budget: Budget,
) -> Result<ResolvedPlan<D>, PlanningFailure> {
    if family.logical != logical.identity {
        return Err(PlanningFailure::CompilerBug(
            "the family belongs to another logical compilation".into(),
        ));
    }
    if family.target != logical.target {
        return Err(PlanningFailure::CompilerBug(
            "the family target differs from the logical specialization".into(),
        ));
    }
    let export = build_model(family, context)?;
    let (assignment, optimal) = solve(family, &export, budget)?;
    // Whole-plan numerical assessment.
    let assessment = composed_assessment(family, &assignment, context);
    family
        .resolve(&assignment, assessment, optimal)
        .map_err(PlanningFailure::from)
}

// ---------------------------------------------------------------------------
// Model construction
// ---------------------------------------------------------------------------

/// One tuning parameter's model variable and its declared domain.
struct ParameterVar {
    variable: VarId,
    lower: i64,
    upper: i64,
}

struct ChoiceExport {
    variable: VarId,
    active: VarId,
    /// One selected-indicator per physical alternative (list order).
    selected: Vec<VarId>,
    selected_by_identity: BTreeMap<(u32, u32), VarId>,
}

struct Export {
    model: Arc<Model>,
    choices: BTreeMap<ChoiceId, ChoiceExport>,
    parameters: Vec<(String, VarId)>,
    offsets: Vec<(PhysicalStorageTemplateId, VarId)>,
}

fn build_model<D: ExecutableDialect>(
    family: &PlanFamily<D>,
    context: &Context<'_>,
) -> Result<Export, PlanningFailure> {
    let mut builder = ModelBuilder::new();
    builder.units("estimated executable cost");

    // --- choice and alternative variables ----------------------------------
    let mut choices: BTreeMap<ChoiceId, ChoiceExport> = BTreeMap::new();
    let mut activation_parents: BTreeMap<ChoiceId, Vec<VarId>> = BTreeMap::new();
    for choice_id in family.choices.ids() {
        let physical = &family.choices[choice_id];
        let count = physical.alternatives.len() as i64;
        let mut domain: Vec<i64> = (0..count).collect();
        if choice_id != family.entry {
            domain.push(INACTIVE);
        }
        let variable = builder.variable(
            format!("choice {}", choice_id.0),
            Domain::set(domain.clone()),
        );
        let active = builder.variable(format!("choice {} active", choice_id.0), boolean());
        builder.constraint(Constraint::Table {
            variables: vec![variable, active],
            tuples: domain
                .iter()
                .map(|value| vec![*value, i64::from(*value != INACTIVE)])
                .collect(),
        });
        let selected: Vec<VarId> = physical
            .alternatives
            .iter()
            .enumerate()
            .map(|(ordinal, _)| {
                let indicator = builder.variable(
                    format!("choice {} alternative {ordinal}", choice_id.0),
                    boolean(),
                );
                builder.constraint(Constraint::Table {
                    variables: vec![variable, indicator],
                    tuples: domain
                        .iter()
                        .map(|value| vec![*value, i64::from(*value == ordinal as i64)])
                        .collect(),
                });
                indicator
            })
            .collect();
        let selected_by_identity = physical
            .alternatives
            .iter()
            .zip(selected.iter().copied())
            .map(|(alternative, variable)| {
                (
                    (
                        alternative.logical_alternative,
                        alternative.physical_alternative,
                    ),
                    variable,
                )
            })
            .collect();
        choices.insert(
            choice_id,
            ChoiceExport {
                variable,
                active,
                selected,
                selected_by_identity,
            },
        );
    }
    // Child activation from the schedules' call steps.
    for choice_id in family.choices.ids() {
        for alternative in family.choices[choice_id].alternatives.iter() {
            for step in flat_steps(&alternative.schedule) {
                if let ScheduleStepTemplate::Call(call) = step {
                    activation_parents.entry(call.choice).or_default().push(
                        choices[&choice_id].selected[alternative.physical_alternative as usize],
                    );
                }
            }
        }
    }
    for (choice_id, export) in &choices {
        if *choice_id == family.entry {
            if activation_parents.contains_key(choice_id) {
                return Err(PlanningFailure::CompilerBug(
                    "the entry choice has activation parents".into(),
                ));
            }
            builder.constraint(Constraint::InDomain {
                variable: export.active,
                domain: Domain::singleton(1),
            });
            continue;
        }
        let parents = activation_parents
            .get(choice_id)
            .cloned()
            .unwrap_or_default();
        if parents.is_empty() {
            // A choice no alternative invokes (its only caller fused the
            // interval) is unselectable: pin it inactive.
            builder.constraint(Constraint::InDomain {
                variable: export.variable,
                domain: Domain::singleton(INACTIVE),
            });
            continue;
        }
        // active = OR(parent selected): at least one parent, and active
        // implies some parent (both directions via linear inequalities).
        let mut at_least = vec![LinearTerm::new(export.active, -1)];
        at_least.extend(parents.iter().map(|parent| LinearTerm::new(*parent, 1)));
        builder.constraint(Constraint::LinearLe {
            terms: at_least,
            rhs: 0,
        });
        for parent in &parents {
            builder.constraint(Constraint::Implies {
                premise: Literal::new(*parent, 1),
                consequence: Literal::new(export.active, 1),
            });
        }
    }

    // --- tuning parameters ---------------------------------------------------
    let mut parameters: Vec<(String, VarId)> = Vec::new();
    let mut symbols: BTreeMap<String, ParameterVar> = BTreeMap::new();
    for parameter in family.parameters.iter() {
        let variable = builder.variable(
            format!("parameter {}", parameter.name),
            Domain::interval(parameter.lower, parameter.upper)
                .map_err(|e| PlanningFailure::Solver(e.to_string()))?,
        );
        symbols.insert(
            parameter.name.clone(),
            ParameterVar {
                variable,
                lower: parameter.lower,
                upper: parameter.upper,
            },
        );
        parameters.push((parameter.name.clone(), variable));
    }

    // --- storage activation, offsets, arena, interference --------------------
    let activations: Vec<StorageActivation> = family.model.storage.clone();
    let mut offsets: Vec<(PhysicalStorageTemplateId, VarId)> = Vec::new();
    let mut activation_variables = BTreeMap::new();
    for activation in activations.iter() {
        let template = family
            .storages
            .get(activation.storage)
            .ok_or_else(|| {
                PlanningFailure::CompilerBug(format!(
                    "storage#{:?} is absent from the family table",
                    activation.storage
                ))
            })?
            .clone();
        let alignment = template.alignment.max(1) as u64;
        let active = activation_variable(
            &mut builder,
            &choices,
            &template.active_if,
            &mut activation_variables,
        )?;
        let active_literal = Literal::new(active, 1);
        let offset = builder.variable(
            format!("offset {:?}", activation.storage),
            Domain::progression(0, OFFSET_MAX, alignment)
                .map_err(|e| PlanningFailure::Solver(e.to_string()))?,
        );
        builder.constraint(Constraint::InactiveValue {
            active: active_literal,
            variable: offset,
            inactive: 0,
        });
        let (bytes, _, _) = materialize(&mut builder, &template.bytes, &symbols)?;
        builder.guarded_constraint(
            vec![active_literal],
            Constraint::LinearLe {
                terms: vec![LinearTerm::new(offset, 1), LinearTerm::new(bytes, 1)],
                rhs: i128::from(context.target.limits.max_device_bytes.min(OFFSET_MAX)),
            },
        );
        offsets.push((activation.storage, offset));
    }
    // Lifetime interference: solver ordering disjunctions.
    for interference in &family.model.interference {
        let left = offsets
            .iter()
            .find(|(storage, _)| *storage == interference.left)
            .map(|(_, variable)| *variable);
        let right = offsets
            .iter()
            .find(|(storage, _)| *storage == interference.right)
            .map(|(_, variable)| *variable);
        let (Some(left_offset), Some(right_offset)) = (left, right) else {
            return Err(PlanningFailure::CompilerBug(
                "an interference pair names storage absent from the placement model".into(),
            ));
        };
        let left_template = family.storages.get(interference.left).cloned();
        let right_template = family.storages.get(interference.right).cloned();
        let (Some(left_template), Some(right_template)) = (left_template, right_template) else {
            return Err(PlanningFailure::CompilerBug(
                "an interference pair names storage absent from the family".into(),
            ));
        };
        let (left_bytes, _, _) = materialize(&mut builder, &left_template.bytes, &symbols)?;
        let (right_bytes, _, _) = materialize(&mut builder, &right_template.bytes, &symbols)?;
        let active_left = Literal::new(
            activation_variable(
                &mut builder,
                &choices,
                &left_template.active_if,
                &mut activation_variables,
            )?,
            1,
        );
        let active_right = Literal::new(
            activation_variable(
                &mut builder,
                &choices,
                &right_template.active_if,
                &mut activation_variables,
            )?,
            1,
        );
        let order_left = builder.variable("interference order left", boolean());
        let order_right = builder.variable("interference order right", boolean());
        builder.constraint(Constraint::ExactlyOne {
            variables: vec![order_left, order_right],
        });
        builder.guarded_constraint(
            vec![active_left, active_right, Literal::new(order_left, 1)],
            Constraint::LinearLe {
                terms: vec![
                    LinearTerm::new(left_offset, 1),
                    LinearTerm::new(left_bytes, 1),
                    LinearTerm::new(right_offset, -1),
                ],
                rhs: 0,
            },
        );
        builder.guarded_constraint(
            vec![active_left, active_right, Literal::new(order_right, 1)],
            Constraint::LinearLe {
                terms: vec![
                    LinearTerm::new(right_offset, 1),
                    LinearTerm::new(right_bytes, 1),
                    LinearTerm::new(left_offset, -1),
                ],
                rhs: 0,
            },
        );
    }

    // --- per-alternative hard constraints ---------------------------------------
    for facts in &family.model.alternatives {
        let Some(choice) = choices.get(&facts.choice) else {
            return Err(PlanningFailure::CompilerBug(
                "model facts name an absent choice".into(),
            ));
        };
        let ordinal = family.choices[facts.choice]
            .alternatives
            .iter()
            .position(|alternative| {
                alternative.logical_alternative == facts.logical_alternative
                    && alternative.physical_alternative == facts.physical_alternative
            })
            .ok_or_else(|| {
                PlanningFailure::CompilerBug("model facts name an absent alternative".into())
            })?;
        let selected = choice.selected[ordinal];
        // Cost: exact same object the resolver evaluates.
        let (cost, low, _) = materialize(&mut builder, &facts.cost, &symbols)?;
        if low < 0 {
            return Err(PlanningFailure::CompilerBug(
                "an alternative cost may be negative".into(),
            ));
        }
        builder.guarded_cost(
            vec![Literal::new(selected, 1)],
            Cost::Linear {
                constant: 0,
                terms: vec![LinearTerm::new(cost, 1)],
            },
        );
        // Capability coverage: an exact signature absent from the effective
        // target removes the alternative before solving.
        let uncovered = !facts
            .required_capabilities
            .is_subset(&context.target.effective_signatures);
        if uncovered {
            builder.constraint(Constraint::InDomain {
                variable: selected,
                domain: Domain::singleton(0),
            });
            continue;
        }
        for launch in &facts.launches {
            add_launch_constraints(&mut builder, selected, launch, &symbols, context.target)?;
        }
        // Numerical policy: Unknown (or uncovered) transfers need accepted
        // evidence; strict policies always retain ordered universal
        // strategies and exact math.
        if !matches!(context.precision, PrecisionPolicy::Unconstrained)
            && !matches!(
                facts.numerical,
                seismic_realization::NumericalTransfer::Exact
            )
        {
            let witnesses =
                numerical_witnesses(&mut builder, family, context, &choices, &parameters);
            if witnesses.is_empty() {
                builder.constraint(Constraint::InDomain {
                    variable: selected,
                    domain: Domain::singleton(0),
                });
            } else {
                let mut terms = vec![LinearTerm::new(selected, 1)];
                terms.extend(witnesses.iter().map(|w| LinearTerm::new(*w, -1)));
                builder.constraint(Constraint::LinearLe { terms, rhs: 0 });
            }
        }
    }

    let model = Arc::new(
        builder
            .build()
            .map_err(|e| PlanningFailure::Solver(e.to_string()))?,
    );
    Ok(Export {
        model,
        choices,
        parameters,
        offsets,
    })
}

fn add_launch_constraints(
    builder: &mut ModelBuilder,
    selected: VarId,
    launch: &LaunchFacts,
    symbols: &BTreeMap<String, ParameterVar>,
    target: &EffectiveTargetProfile,
) -> Result<(), PlanningFailure> {
    let guard = vec![Literal::new(selected, 1)];
    fn range(
        builder: &mut ModelBuilder,
        guard: &[Literal],
        expr: &Sym,
        low: i64,
        high: i64,
        symbols: &BTreeMap<String, ParameterVar>,
    ) -> Result<(), PlanningFailure> {
        let (variable, _, _) = materialize(builder, expr, symbols)?;
        builder.guarded_constraint(
            guard.to_vec(),
            Constraint::InDomain {
                variable,
                domain: Domain::interval(low, high)
                    .map_err(|e| PlanningFailure::Solver(e.to_string()))?,
            },
        );
        Ok(())
    }
    // 1 <= preferred_participants <= target.max_participants
    range(
        builder,
        &guard,
        &launch.preferred_participants,
        1,
        target.limits.max_participants,
        symbols,
    )?;
    for axis in 0..3 {
        // Launch conditions gate positive geometry for zero work: workgroup
        // counts may be zero only when there is no work (the runtime skips).
        range(
            builder,
            &guard,
            &launch.workgroups[axis],
            0,
            target.limits.max_workgroups_axis[axis],
            symbols,
        )?;
    }
    range(
        builder,
        &guard,
        &launch.workgroup_bytes,
        0,
        target.limits.max_workgroup_bytes,
        symbols,
    )?;
    range(
        builder,
        &guard,
        &launch.private_bytes_per_participant,
        0,
        target.limits.max_explicit_private_bytes,
        symbols,
    )?;
    if launch.direct_bindings > 0 {
        let bindings = i64::from(launch.direct_bindings);
        if bindings > target.limits.max_direct_bindings {
            builder.constraint(Constraint::InDomain {
                variable: selected,
                domain: Domain::singleton(0),
            });
        }
    }
    if launch.argument_table_bytes as i64 > target.limits.max_argument_table_bytes {
        builder.constraint(Constraint::InDomain {
            variable: selected,
            domain: Domain::singleton(0),
        });
    }
    Ok(())
}

/// Evidence witnesses for non-exact alternatives: each witness matches the
/// complete assignment identity (selections plus tuning values).
fn numerical_witnesses<D: ExecutableDialect>(
    builder: &mut ModelBuilder,
    family: &PlanFamily<D>,
    context: &Context<'_>,
    choices: &BTreeMap<ChoiceId, ChoiceExport>,
    parameters: &[(String, VarId)],
) -> Vec<VarId> {
    let mut witnesses = Vec::new();
    for evidence in context.numerical_evidence {
        if evidence.logical != family.logical
            || evidence.target != context.target.backend
            || evidence.capability_fingerprint != context.target.capability_fingerprint
            || evidence.toolchain_fingerprint != context.target.toolchain_fingerprint
            || !evidence.assessment.satisfies(context.precision)
        {
            continue;
        }
        let mut matches = Vec::new();
        let mut valid = true;
        for (choice_id, export) in choices {
            let wanted = match evidence.selections.get(choice_id) {
                None => INACTIVE,
                Some((logical, physical)) => family.choices[*choice_id]
                    .alternatives
                    .iter()
                    .position(|alternative| {
                        alternative.logical_alternative == *logical
                            && alternative.physical_alternative == *physical
                    })
                    .map(|ordinal| ordinal as i64)
                    .unwrap_or_else(|| {
                        valid = false;
                        INACTIVE
                    }),
            };
            matches.push(eq_indicator(builder, export.variable, wanted));
        }
        for (name, variable) in parameters {
            match evidence.symbols.get(name) {
                Some(value) => matches.push(eq_indicator(builder, *variable, *value)),
                None => valid = false,
            }
        }
        if !valid {
            continue;
        }
        let witness = builder.variable("complete numerical witness", boolean());
        builder.constraint(Constraint::BoolAnd {
            output: witness,
            inputs: matches,
        });
        witnesses.push(witness);
    }
    witnesses
}

// ---------------------------------------------------------------------------
// Solve (budget ordering)
// ---------------------------------------------------------------------------

fn solve<D: ExecutableDialect>(
    family: &PlanFamily<D>,
    export: &Export,
    budget: Budget,
) -> Result<(FeasiblePlanAssignment, bool), PlanningFailure> {
    let mut search = Search::new(
        Arc::clone(&export.model),
        Options {
            algorithm: Algorithm::Exact,
            ..Options::default()
        },
    )
    .map_err(|e| PlanningFailure::Solver(e.to_string()))?;
    // Feasibility is decided first; the budget limits optimization only.
    let outcome = search
        .advance_budgeted(SolverBudget {
            optimization: SolverLimits {
                work: budget.work.max(1),
                time: budget.time,
                memory_bytes: None,
            },
        })
        .map_err(|e| PlanningFailure::Solver(e.to_string()))?;
    let (solution, optimal) = match outcome {
        Budgeted::Optimal(solution) => (solution.feasible().clone(), true),
        Budgeted::Incumbent { solution, .. } => (solution, false),
        Budgeted::Infeasible => return Err(PlanningFailure::Infeasible),
        Budgeted::Suspended(progress) => match progress.incumbent {
            // Production has no incomplete/no-incumbent result: a suspension
            // with an incumbent is a budget stop, without one it is a bug.
            Some(solution) => (solution, false),
            None => {
                return Err(PlanningFailure::CompilerBug(
                    "the search suspended before feasibility was decided".into(),
                ));
            }
        },
    };
    let assignment = decode(family, export, solution)?;
    Ok((assignment, optimal))
}

fn decode<D: ExecutableDialect>(
    family: &PlanFamily<D>,
    export: &Export,
    solution: FeasibleSolution,
) -> Result<FeasiblePlanAssignment, PlanningFailure> {
    let values = solution.values();
    let mut selections: BTreeMap<ChoiceId, (u32, u32)> = BTreeMap::new();
    for (choice_id, choice) in &export.choices {
        let ordinal = *values
            .get(choice.variable.0)
            .ok_or_else(|| PlanningFailure::Solver("the solution omits a choice".into()))?;
        if ordinal != INACTIVE {
            let alternative = family.choices[*choice_id]
                .alternatives
                .iter()
                .nth(ordinal as usize)
                .ok_or_else(|| {
                    PlanningFailure::Solver("the solution selects an absent alternative".into())
                })?;
            selections.insert(
                *choice_id,
                (
                    alternative.logical_alternative,
                    alternative.physical_alternative,
                ),
            );
        }
    }
    let mut symbols: BTreeMap<String, i64> = BTreeMap::new();
    for (name, variable) in &export.parameters {
        symbols.insert(
            name.clone(),
            *values
                .get(variable.0)
                .ok_or_else(|| PlanningFailure::Solver("the solution omits a parameter".into()))?,
        );
    }
    let mut offsets: BTreeMap<PhysicalStorageTemplateId, u64> = BTreeMap::new();
    for (storage, variable) in &export.offsets {
        let value = *values
            .get(variable.0)
            .ok_or_else(|| PlanningFailure::Solver("the solution omits an offset".into()))?;
        let offset = u64::try_from(value)
            .map_err(|_| PlanningFailure::CompilerBug("a solved offset is negative".into()))?;
        offsets.insert(*storage, offset);
    }
    let decoded = seismic_realization::executable::DecodedPlanValues {
        selections,
        symbols,
        offsets,
    };
    let assignment = FeasibleAssignment::new(solution);
    FeasiblePlanAssignment::from_feasible(assignment, decoded).map_err(PlanningFailure::from)
}

fn composed_assessment<D: ExecutableDialect>(
    family: &PlanFamily<D>,
    assignment: &FeasiblePlanAssignment,
    context: &Context<'_>,
) -> NumericalAssessment {
    // Accepted evidence for this exact assignment, if any.
    for evidence in context.numerical_evidence {
        if evidence.logical == family.logical
            && evidence.target == context.target.backend
            && evidence.capability_fingerprint == context.target.capability_fingerprint
            && evidence.toolchain_fingerprint == context.target.toolchain_fingerprint
            && &evidence.selections == assignment.selections()
            && &evidence.symbols == assignment.symbols()
            && evidence.assessment.satisfies(context.precision)
        {
            return evidence.assessment.clone();
        }
    }
    // Compose the selected alternatives' transfers.
    let mut unknown: Option<String> = None;
    for (choice_id, (logical, physical)) in assignment.selections() {
        if let Some(choice) = family.choices.get(*choice_id) {
            if let Some(alternative) = choice.alternatives.iter().find(|alternative| {
                alternative.logical_alternative == *logical
                    && alternative.physical_alternative == *physical
            }) {
                if let seismic_realization::NumericalTransfer::Unknown { reason } =
                    &alternative.numerical
                {
                    unknown = Some(reason.clone());
                }
            }
        }
    }
    match unknown {
        Some(reason) => NumericalAssessment::unknown(reason),
        None => NumericalAssessment::exact(),
    }
}

// ---------------------------------------------------------------------------
// Symbolic helpers (checked arithmetic; overflow is infeasible, never wraps)
// ---------------------------------------------------------------------------

fn activation_variable(
    builder: &mut ModelBuilder,
    choices: &BTreeMap<ChoiceId, ChoiceExport>,
    activation: &seismic_realization::executable::ActivationLiteral,
    cache: &mut BTreeMap<seismic_realization::executable::ActivationLiteral, VarId>,
) -> Result<VarId, PlanningFailure> {
    if let Some(variable) = cache.get(activation) {
        return Ok(*variable);
    }
    let mut terms = Vec::with_capacity(activation.0.len());
    for term in &activation.0 {
        let choice = choices.get(&term.choice).ok_or_else(|| {
            PlanningFailure::CompilerBug(format!(
                "storage activation names choice {:?} absent from the family",
                term.choice
            ))
        })?;
        let variable = choice
            .selected_by_identity
            .get(&(term.logical_alternative, term.physical_alternative))
            .copied()
            .ok_or_else(|| {
                PlanningFailure::CompilerBug(format!(
                    "storage activation names absent alternative logical#{} physical#{} of choice {:?}",
                    term.logical_alternative, term.physical_alternative, term.choice
                ))
            })?;
        terms.push(variable);
    }
    let active = match terms.as_slice() {
        [] => builder.variable("unconditional storage activation", Domain::singleton(1)),
        [only] => *only,
        _ => {
            let output = builder.variable("storage activation conjunction", boolean());
            for term in &terms {
                builder.constraint(Constraint::LinearLe {
                    terms: vec![LinearTerm::new(output, 1), LinearTerm::new(*term, -1)],
                    rhs: 0,
                });
            }
            let mut all_implies_output = terms
                .iter()
                .map(|term| LinearTerm::new(*term, 1))
                .collect::<Vec<_>>();
            all_implies_output.push(LinearTerm::new(output, -1));
            builder.constraint(Constraint::LinearLe {
                terms: all_implies_output,
                rhs: i128::try_from(terms.len() - 1).map_err(|_| {
                    PlanningFailure::CompilerBug(
                        "storage activation conjunction exceeds the solver numeric range".into(),
                    )
                })?,
            });
            output
        }
    };
    cache.insert(activation.clone(), active);
    Ok(active)
}

fn flat_steps<D: ExecutableDialect>(
    schedule: &seismic_realization::executable::ScheduleTemplate<D>,
) -> Vec<&seismic_realization::executable::ScheduleStepTemplate<D>> {
    let mut steps = Vec::new();
    fn walk<'a, D: ExecutableDialect>(
        schedule: &'a seismic_realization::executable::ScheduleTemplate<D>,
        steps: &mut Vec<&'a seismic_realization::executable::ScheduleStepTemplate<D>>,
    ) {
        for step in schedule.steps.iter() {
            steps.push(step);
            match step {
                ScheduleStepTemplate::If(if_step) => {
                    walk(&if_step.then_schedule, steps);
                    walk(&if_step.else_schedule, steps);
                }
                ScheduleStepTemplate::Repeat(repeat) => {
                    walk(&repeat.body, steps);
                }
                _ => {}
            }
        }
    }
    walk(schedule, &mut steps);
    steps
}

fn eq_indicator(builder: &mut ModelBuilder, variable: VarId, wanted: i64) -> VarId {
    let output = builder.variable("assignment equality", boolean());
    builder.constraint(Constraint::Implies {
        premise: Literal::new(output, 1),
        consequence: Literal::new(variable, wanted),
    });
    builder.constraint(Constraint::Implies {
        premise: Literal::new(variable, wanted),
        consequence: Literal::new(output, 1),
    });
    output
}

fn boolean() -> Domain {
    Domain::interval(0, 1).expect("the boolean domain")
}

/// Materialize one symbolic expression as a model variable with exact
/// equality constraints. Products, aligned sums, and quotients use checked
/// symbolic arithmetic; overflow makes the alternative infeasible.
fn materialize(
    builder: &mut ModelBuilder,
    expression: &Sym,
    symbols: &BTreeMap<String, ParameterVar>,
) -> Result<(VarId, i64, i64), PlanningFailure> {
    fn atom(
        builder: &mut ModelBuilder,
        atom: &Atom,
        symbols: &BTreeMap<String, ParameterVar>,
    ) -> Result<(VarId, i64, i64), PlanningFailure> {
        match atom {
            Atom::Param(name) => {
                let parameter = symbols.get(name).ok_or_else(|| {
                    PlanningFailure::CompilerBug(format!(
                        "the plan expression references undeclared symbol `{name}`"
                    ))
                })?;
                Ok((parameter.variable, parameter.lower, parameter.upper))
            }
            Atom::Quot(numerator, denominator) | Atom::Rem(numerator, denominator) => {
                let (n, nl, nh) = materialize(builder, numerator, symbols)?;
                let (d, dl, dh) = materialize(builder, denominator, symbols)?;
                if nl < 0 || dl <= 0 {
                    return Err(PlanningFailure::CompilerBug(
                        "invalid symbolic division domain".into(),
                    ));
                }
                let qh = nh / dl;
                let rh = nh.min(dh.saturating_sub(1));
                let q = builder.variable(
                    "symbolic quotient",
                    Domain::interval(0, qh).map_err(|e| PlanningFailure::Solver(e.to_string()))?,
                );
                let r = builder.variable(
                    "symbolic remainder",
                    Domain::interval(0, rh).map_err(|e| PlanningFailure::Solver(e.to_string()))?,
                );
                builder.constraint(Constraint::Arithmetic(Arithmetic::DivRem {
                    numerator: n,
                    denominator: d,
                    quotient: q,
                    remainder: r,
                }));
                if matches!(atom, Atom::Quot(..)) {
                    Ok((q, 0, qh))
                } else {
                    Ok((r, 0, rh))
                }
            }
        }
    }
    let mut terms = Vec::new();
    let mut low = 0i128;
    let mut high = 0i128;
    for (monomial, coefficient) in expression.monomials() {
        let mut variable = builder.variable("monomial one", Domain::singleton(1));
        let (mut lo, mut hi) = (1i64, 1i64);
        for (factor, degree) in monomial {
            for _ in 0..*degree {
                let (right, rl, rh) = atom(builder, factor, symbols)?;
                let pl = i128::from(lo) * i128::from(rl.min(i64::MAX));
                let ph = i128::from(hi) * i128::from(rh.min(i64::MAX));
                let (pl, ph) = (pl.min(i128::from(i64::MAX)), ph.min(i128::from(i64::MAX)));
                let product = builder.variable(
                    "symbolic product",
                    Domain::interval(
                        i64::try_from(pl).map_err(|_| {
                            PlanningFailure::CompilerBug("symbolic product underflow".into())
                        })?,
                        i64::try_from(ph).map_err(|_| {
                            PlanningFailure::CompilerBug("symbolic product overflow".into())
                        })?,
                    )
                    .map_err(|e| PlanningFailure::Solver(e.to_string()))?,
                );
                builder.constraint(Constraint::Arithmetic(Arithmetic::Product {
                    left: variable,
                    right,
                    product,
                }));
                variable = product;
                lo = i64::try_from(pl).unwrap_or(i64::MIN);
                hi = i64::try_from(ph).unwrap_or(i64::MAX);
            }
        }
        terms.push((variable, coefficient));
        if coefficient >= 0 {
            low += i128::from(coefficient) * i128::from(lo);
            high += i128::from(coefficient) * i128::from(hi);
        } else {
            low += i128::from(coefficient) * i128::from(hi);
            high += i128::from(coefficient) * i128::from(lo);
        }
    }
    let low = i64::try_from(low.max(0))
        .map_err(|_| PlanningFailure::CompilerBug("symbolic value exceeds i64".into()))?;
    let high = i64::try_from(high.min(i128::from(i64::MAX)))
        .map_err(|_| PlanningFailure::CompilerBug("symbolic value exceeds i64".into()))?;
    let result = builder.variable(
        "symbolic expression",
        Domain::interval(low, high).map_err(|e| PlanningFailure::Solver(e.to_string()))?,
    );
    let mut equality = terms
        .iter()
        .map(|(variable, coefficient)| LinearTerm::new(*variable, *coefficient))
        .collect::<Vec<_>>();
    equality.push(LinearTerm::new(result, -1));
    equal_zero(builder, equality);
    Ok((result, low, high))
}

fn equal_zero(builder: &mut ModelBuilder, terms: Vec<LinearTerm>) {
    builder.constraint(Constraint::LinearLe {
        terms: terms.clone(),
        rhs: 0,
    });
    builder.constraint(Constraint::LinearLe {
        terms: terms
            .into_iter()
            .map(|term| LinearTerm::new(term.variable, -term.coefficient))
            .collect(),
        rhs: 0,
    });
}

// ---------------------------------------------------------------------------
// Synthetic test dialect
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::{
        logical::{construct, LogicalNode, LogicalNodeKind, RegionResult, TaskGraph},
        program::{compile, SourceFile},
        types::{DType, NonEmpty, TensorType, ValuePath},
    };
    use seismic_realization::executable::{
        BoundaryTemplates, ExecutorPredicateTemplate, ExecutorRangeTemplate, FusedStrategyTemplate,
        NodeRef, ObligationDisposition, PhysicalCarryTemplate, PlanValues, RegionPath, RegionStep,
        StateTransportTemplate,
    };
    use seismic_realization::NumericalTransfer;
    use seismic_realization::{
        dispatch::LinearIterationMap,
        executable::{
            AlternativeBuilder, CostEstimate, EffectiveTargetProfile, ExecutableDialect,
            HardResources, InvariantReport, Legalized, NativeResourceContract,
            PhysicalConsequences, PhysicalJoinTemplate, PhysicalPrimitive, PlanFamilyBuilder,
            ReductionStrategyTemplate, ReductionTopology, ResolvedStep, ResolvedTransport,
            TransportTemplate,
        },
    };
    use std::collections::BTreeMap;
    use std::collections::BTreeSet;

    // --- the synthetic dialect ---------------------------------------------

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum SynOp {
        Compute,
        Access,
        Predicate,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct SynDialect;
    impl seismic_realization::executable::sealed::Sealed for SynDialect {}

    impl ExecutableDialect for SynDialect {
        type Op = SynOp;
        type LayoutTemplate = u64;
        type ResolvedLayout = u64;

        fn legalize(p: &PhysicalPrimitive, _t: &EffectiveTargetProfile) -> Legalized<SynOp> {
            // Universal portable legalization: every registry primitive maps
            // to at least one synthetic opcode. `Inapplicable` here would be a
            // compiler bug detected by the family builder.
            let op = match &p.op {
                seismic_lang::logical::PrimitiveOp::Constant(_)
                | seismic_lang::logical::PrimitiveOp::RuntimeExtent(_) => SynOp::Compute,
                seismic_lang::logical::PrimitiveOp::Primitive(id) => {
                    if matches!(
                        id,
                        seismic_lang::intrinsics::PrimitiveId::ElementRead { .. }
                            | seismic_lang::intrinsics::PrimitiveId::ElementWrite { .. }
                            | seismic_lang::intrinsics::PrimitiveId::PackedRead(_)
                    ) {
                        SynOp::Access
                    } else {
                        SynOp::Compute
                    }
                }
                seismic_lang::logical::PrimitiveOp::Capability(_) => SynOp::Compute,
            };
            Legalized::Ops(NonEmpty::new(vec![op]).expect("one opcode"))
        }

        fn consequences(_op: &SynOp) -> PhysicalConsequences {
            PhysicalConsequences {
                hard: HardResources::default(),
                native_contract: NativeResourceContract {
                    max_resident_participants: (1, u64::MAX),
                    native_subgroup_width: None,
                },
                cost: CostEstimate(1),
                numerical: NumericalTransfer::Exact,
                capability: None,
            }
        }

        fn public_layout(tensor: &TensorType) -> u64 {
            match tensor.elem {
                seismic_lang::types::Elem::Dtype(dtype) => u64::from(dtype.bytes()),
                _ => 4,
            }
        }

        fn internal_layout(tensor: &TensorType) -> u64 {
            Self::public_layout(tensor)
        }

        fn resolve_layout(layout: &u64, _values: &PlanValues) -> Result<u64, InvariantReport> {
            Ok(*layout)
        }
    }

    // --- the universal synthetic strategy ----------------------------------

    /// One representative kernel: borrowed views, a moved owned result, an
    /// independent outer loop with a disjoint-write join, an ordered inner
    /// loop with a carried state, and one nested call.
    const KERNEL: &str = "fn add[M, N](x: &tensor[M, N] f32, y: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32:\n    let mut output = result\n    parallel for row in 0..M:\n        for col in 0..N:\n            output[row, col] = x[row, col] + y[row, col]\n    return output\n\nfn linear[M, N](x: &tensor[M, N] f32, y: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32:\n    return add(x, y, result)\n";

    fn check(sources: &[(&str, &str)]) -> Result<seismic_lang::sir::Program, String> {
        let files: Vec<SourceFile> = sources
            .iter()
            .map(|(path, text)| SourceFile {
                path: path.to_string(),
                text: text.to_string(),
            })
            .collect();
        compile(&files).map_err(|d| d.iter().map(|d| d.render()).collect::<Vec<_>>().join("\n"))
    }

    fn target() -> EffectiveTargetProfile {
        let mut effective_signatures = BTreeSet::new();
        effective_signatures.insert(seismic_lang::intrinsics::IntrinsicId {
            capability: seismic_lang::intrinsics::CapabilityId::new("test", "cap"),
            name: "all".into(),
        });
        EffectiveTargetProfile {
            backend: "cpu".into(),
            capability_fingerprint: "test-fingerprint".into(),
            toolchain_fingerprint: "test-toolchain".into(),
            effective_signatures,
            limits: seismic_realization::executable::TargetLimits {
                max_participants: 1024,
                max_workgroups_axis: [1024, 1, 1],
                max_workgroup_bytes: 32768,
                max_explicit_private_bytes: 32768,
                max_direct_bindings: 64,
                max_argument_table_bytes: 65536,
                max_device_bytes: 1 << 30,
            },
        }
    }

    fn supports_all(_: &seismic_lang::sir::IntrinsicUse) -> Result<(), String> {
        Ok(())
    }

    fn logical_target() -> seismic_lang::logical::EffectiveTargetIdentity {
        seismic_lang::logical::EffectiveTargetIdentity {
            backend: "cpu".into(),
            capability_fingerprint: "test-fingerprint".into(),
        }
    }

    fn shapes(entries: &[(&str, i64)]) -> BTreeMap<String, i64> {
        entries.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    fn legalize_node(
        _builder: &AlternativeBuilder<SynDialect>,
        node: &LogicalNode,
        target: &EffectiveTargetProfile,
    ) -> Legalized<SynOp> {
        let op = match &node.kind {
            LogicalNodeKind::Primitive(application) => application.op.clone(),
            _ => unreachable!("only primitive nodes legalize"),
        };
        let primitive = PhysicalPrimitive {
            op,
            inputs: Vec::new(),
            results: Vec::new(),
        };
        SynDialect::legalize(&primitive, target)
    }
    /// State transport of one logical storage in this alternative: its own
    /// template, or the caller's boundary placeholder.
    fn state_transport(
        builder: &AlternativeBuilder<SynDialect>,
        graph: &TaskGraph,
        storage: seismic_lang::logical::LogicalStorageId,
    ) -> StateTransportTemplate {
        if let Some(template) = builder.storage_of(storage) {
            StateTransportTemplate::Storage(template)
        } else {
            let leaf = match &graph.storages.get(storage).map(|s| s.origin.clone()) {
                Some(seismic_lang::logical::StorageOrigin::Parameter { ordinal, path, .. }) => {
                    seismic_realization::executable::BoundaryLeaf::Input {
                        param: *ordinal,
                        leaf: path.clone(),
                    }
                }
                Some(seismic_lang::logical::StorageOrigin::Result { path, .. }) => {
                    seismic_realization::executable::BoundaryLeaf::Result { leaf: path.clone() }
                }
                _ => seismic_realization::executable::BoundaryLeaf::Result {
                    leaf: ValuePath::default(),
                },
            };
            StateTransportTemplate::Boundary(leaf)
        }
    }

    /// Map every region of one alternative with the universal strategy:
    /// primitives and reductions launch; loops and conditionals become
    /// structured executor steps (or kernel-local control when wholly inside
    /// one launch — the synthetic strategy always uses executor steps);
    /// calls invoke nested plans; every obligation is discharged.
    fn map_region(
        builder: &mut AlternativeBuilder<SynDialect>,
        region: &RegionPath,
        target: &EffectiveTargetProfile,
    ) -> Result<(), String> {
        let graph = builder.graph().clone();
        let nodes = builder.region_nodes(region)?;
        for (node_id, node) in nodes {
            let node_ref = NodeRef {
                region: region.clone(),
                node: node_id,
            };
            if !builder.pending_nodes().contains(&node_ref) {
                // Already consumed (e.g. the call node of a cross-call fuse).
                continue;
            }
            match &node.kind {
                LogicalNodeKind::Primitive(_) => {
                    let legalized = legalize_node(builder, &node, target);
                    builder.map_primitive(node_ref, LinearIterationMap::serial(), legalized)?;
                }
                LogicalNodeKind::Reduction(_) => {
                    let result_value = node
                        .outputs
                        .first()
                        .map(|output| output.id)
                        .expect("the reduction has a result");
                    builder.map_reduction(
                        node_ref,
                        ReductionStrategyTemplate {
                            topology: ReductionTopology::SerialAxis {
                                axis: 0,
                                length: seismic_lang::types::ExtentExpr::Static(16),
                            },
                            iteration: LinearIterationMap::serial(),
                            ops: Legalized::Ops(NonEmpty::new(vec![SynOp::Compute]).expect("one")),
                            result: builder.transport_of(result_value)?,
                        },
                    )?;
                }
                LogicalNodeKind::If(if_node) => {
                    let condition = builder.transport_of(if_node.condition)?;
                    let then_region = {
                        let mut path = region.clone();
                        path.push(RegionStep::IfThen(node_id));
                        path
                    };
                    let else_region = {
                        let mut path = region.clone();
                        path.push(RegionStep::IfElse(node_id));
                        path
                    };
                    let joins = if_node
                        .joins
                        .iter()
                        .map(|slot| match slot {
                            seismic_lang::logical::JoinSlot::Value { joined, .. } => {
                                Ok(PhysicalJoinTemplate::Value {
                                    then: builder.transport_of(*joined)?,
                                    else_branch: builder.transport_of(*joined)?,
                                    joined: builder.transport_of(*joined)?,
                                })
                            }
                            seismic_lang::logical::JoinSlot::State { storage, .. } => {
                                Ok(PhysicalJoinTemplate::State {
                                    storage: builder.storage_of(*storage).ok_or_else(|| {
                                        "a state join must name storage of this alternative"
                                            .to_string()
                                    })?,
                                })
                            }
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    builder.schedule_if(
                        node_ref,
                        ExecutorPredicateTemplate { value: condition },
                        |then_builder| map_region(then_builder, &then_region, target),
                        |else_builder| map_region(else_builder, &else_region, target),
                        joins,
                    )?;
                }
                LogicalNodeKind::Loop(loop_node) => {
                    let body = {
                        let mut path = region.clone();
                        path.push(RegionStep::LoopBody(node_id));
                        path
                    };
                    let start = builder.transport_of(loop_node.range.start)?;
                    let end = builder.transport_of(loop_node.range.end)?;
                    let mut carries = Vec::new();
                    for slot in &loop_node.carried {
                        let transport = match slot.initial {
                            seismic_lang::logical::RegionInput::Value(initial) => {
                                let initial_transport = builder.transport_of(initial)?;
                                match initial_transport {
                                    TransportTemplate::Storage(_)
                                    | TransportTemplate::ExecutorScalar(_) => initial_transport,
                                    _ => {
                                        let dtype = node
                                            .outputs
                                            .first()
                                            .and_then(|output| output.ty.scalar_dtype())
                                            .unwrap_or(DType::I32);
                                        TransportTemplate::ExecutorScalar(
                                            seismic_realization::executable::ExecutorScalarTemplate {
                                                source:
                                                    seismic_realization::executable::ExecutorScalarSource::Slot(
                                                        builder.declare_executor_scalar_slot(dtype, "carry"),
                                                    ),
                                                dtype,
                                            },
                                        )
                                    }
                                }
                            }
                            seismic_lang::logical::RegionInput::State(token) => {
                                let storage = builder
                                    .logical_storage_of_token(token)
                                    .expect("the carried state has a storage");
                                match state_transport(builder, &graph, storage) {
                                    StateTransportTemplate::Storage(template) => {
                                        TransportTemplate::Storage(
                                            NonEmpty::new(vec![
                                                seismic_realization::executable::StorageViewTemplate {
                                                    storage: template,
                                                    access: seismic_lang::logical::Access::Exclusive,
                                                    transform: seismic_lang::logical::ViewTransform::Identity,
                                                },
                                            ])
                                            .expect("one plane"),
                                        )
                                    }
                                    StateTransportTemplate::Boundary(path) => {
                                        TransportTemplate::Boundary(path)
                                    }
                                }
                            }
                        };
                        carries.push(PhysicalCarryTemplate { transport });
                    }
                    builder.schedule_loop(
                        node_ref,
                        ExecutorRangeTemplate {
                            start,
                            end,
                            bound: loop_node.range.bound.clone(),
                        },
                        carries,
                        |body_builder| map_region(body_builder, &body, target),
                    )?;
                }
                LogicalNodeKind::Call(call_node) => {
                    use seismic_realization::executable::BoundaryLeaf;
                    let mut boundary = BoundaryTemplates::default();
                    for input in &call_node.boundary_inputs {
                        let input_leaf = BoundaryLeaf::Input {
                            param: input.param,
                            leaf: input.path.clone(),
                        };
                        match input.kind {
                            seismic_lang::logical::BoundaryInputKind::Value(value) => {
                                boundary
                                    .inputs
                                    .insert(input_leaf, builder.transport_of(value)?);
                            }
                            seismic_lang::logical::BoundaryInputKind::Shared {
                                value,
                                state: token,
                            }
                            | seismic_lang::logical::BoundaryInputKind::Exclusive {
                                value,
                                state: token,
                            }
                            | seismic_lang::logical::BoundaryInputKind::Move {
                                value,
                                state: token,
                            } => {
                                let storage = builder
                                    .logical_storage_of_token(token)
                                    .expect("the boundary state has a storage");
                                boundary.states.insert(
                                    input_leaf.clone(),
                                    state_transport(builder, &graph, storage),
                                );
                                boundary
                                    .inputs
                                    .insert(input_leaf, builder.transport_of(value)?);
                            }
                        }
                    }
                    for result in &call_node.boundary_results {
                        let result_leaf = BoundaryLeaf::Result {
                            leaf: result.path.clone(),
                        };
                        match &result.kind {
                            seismic_lang::logical::BoundaryResultKind::Value(value) => {
                                boundary
                                    .results
                                    .insert(result_leaf, builder.transport_of(*value)?);
                            }
                            seismic_lang::logical::BoundaryResultKind::Storage {
                                storage, ..
                            } => {
                                boundary.results.insert(
                                    result_leaf.clone(),
                                    match builder.storage_of(*storage) {
                                        Some(template) => TransportTemplate::Storage(
                                            NonEmpty::new(vec![
                                                seismic_realization::executable::StorageViewTemplate {
                                                    storage: template,
                                                    access: seismic_lang::logical::Access::Exclusive,
                                                    transform: seismic_lang::logical::ViewTransform::Identity,
                                                },
                                            ])
                                            .expect("one plane"),
                                        ),
                                        None => TransportTemplate::Boundary(result_leaf.clone()),
                                    },
                                );
                            }
                            seismic_lang::logical::BoundaryResultKind::State(token) => {
                                let storage = builder
                                    .logical_storage_of_token(*token)
                                    .expect("the result state has a storage");
                                boundary
                                    .states
                                    .insert(result_leaf, state_transport(builder, &graph, storage));
                            }
                        }
                    }
                    builder.invoke(node_ref, boundary)?;
                }
            }
        }
        Ok(())
    }

    /// Build the universal physical alternative of one (choice, logical
    /// alternative) and commit it.
    fn build_universal_alternative(
        family: &mut PlanFamilyBuilder<SynDialect>,
        choice: seismic_lang::logical::ChoiceId,
        logical_alternative: u32,
        target: &EffectiveTargetProfile,
    ) -> Result<(), String> {
        let mut builder = family.alternative(choice, logical_alternative)?;
        map_region(&mut builder, &Vec::new(), target)?;
        // Discharge every safety obligation (static proofs in the synthetic
        // dialect: all bounds are statically known here).
        for obligation in builder.pending_obligations() {
            builder.discharge(
                obligation,
                ObligationDisposition::StaticallyProved {
                    reason: "synthetic dialect".into(),
                },
            )?;
        }
        // Complete every boundary output with its recorded transport.
        let result_count = builder.graph().results.len();
        for ordinal in 0..result_count as u32 {
            let transport = match &builder.graph().results[ordinal as usize] {
                RegionResult::Value { id, .. } => builder.transport_of(*id)?,
                RegionResult::State { storage, .. } => match builder.storage_of(*storage) {
                    Some(template) => TransportTemplate::Storage(
                        NonEmpty::new(vec![seismic_realization::executable::StorageViewTemplate {
                            storage: template,
                            access: seismic_lang::logical::Access::Exclusive,
                            transform: seismic_lang::logical::ViewTransform::Identity,
                        }])
                        .expect("one plane"),
                    ),
                    None => TransportTemplate::Boundary(
                        seismic_realization::executable::BoundaryLeaf::Result {
                            leaf: ValuePath::default(),
                        },
                    ),
                },
            };
            builder.complete_result(ordinal, transport)?;
        }
        let alternative = builder.finish_alternative()?;
        family.add_alternative(choice, alternative)?;
        Ok(())
    }

    #[test]
    fn synthetic_dialect_round_trips_the_representative_kernel() {
        let program = check(&[("kernel.seismic", KERNEL)]).expect("the kernel checks");
        let logical = construct(
            &program,
            "linear",
            &logical_target(),
            &supports_all,
            shapes(&[("M", 4), ("N", 8)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        logical.verify().expect("the built program verifies");

        // Build the family: the entry choice and the call's occurrence choice.
        let entry_choice = logical.entry_choice;
        let call_choice = logical
            .choices
            .ids()
            .find(|choice| *choice != entry_choice)
            .expect("the call created its own choice");
        let target = target();
        let mut family = PlanFamilyBuilder::<SynDialect>::from_logical(&logical)
            .expect("the family builder opens");
        for choice in [entry_choice, call_choice] {
            let alternatives = logical
                .choice(choice)
                .alternatives
                .iter()
                .enumerate()
                .map(|(ordinal, _)| ordinal as u32)
                .collect::<Vec<_>>();
            for logical_alternative in alternatives {
                build_universal_alternative(&mut family, choice, logical_alternative, &target)
                    .expect("the universal alternative builds");
            }
        }
        let family = family.finish().expect("the family finishes");

        // One solve through the budgeted path.
        let context = Context {
            target: &target,
            precision: &PrecisionPolicy::Unconstrained,
            numerical_evidence: &[],
        };
        let plan = plan(&logical, &family, &context, Budget::default())
            .expect("the synthetic family solves and resolves");

        // The resolved artifact has one root ABI, nested calls, and no
        // logical nodes, choices, or unresolved symbols.
        assert!(!plan.abi.buffers.is_empty(), "the root ABI has buffers");
        assert!(
            plan.abi.buffers.iter().any(|buffer| matches!(
                buffer.role,
                seismic_realization::executable::AbiRole::Result
            )),
            "the root ABI allocates the entry result"
        );
        assert!(
            !plan.storage.is_empty(),
            "the global storage table is populated"
        );
        assert!(plan.internal_arena.bytes > 0, "the internal arena is sized");
        // Calls remain nested: the entry schedule contains exactly one call
        // whose body references the same global storage table.
        let mut call_count = 0;
        let mut launch_count = 0;
        fn walk<D: ExecutableDialect>(
            schedule: &seismic_realization::executable::ResolvedSchedule<D>,
            calls: &mut usize,
            launches: &mut usize,
        ) {
            for step in schedule.steps.iter() {
                match step {
                    ResolvedStep::Launch(_) => *launches += 1,
                    ResolvedStep::Call(call) => {
                        *calls += 1;
                        walk(&call.body.schedule, calls, launches);
                    }
                    ResolvedStep::If(if_step) => {
                        walk(&if_step.then_schedule, calls, launches);
                        walk(&if_step.else_schedule, calls, launches);
                    }
                    ResolvedStep::Repeat(repeat) => {
                        walk(&repeat.body, calls, launches);
                    }
                }
            }
        }
        walk(&plan.entry.schedule, &mut call_count, &mut launch_count);
        assert_eq!(call_count, 1, "the entry body is one nested call");
        assert!(launch_count > 0, "the callee launches");
        // Estimated cost is the sum of selected alternative costs.
        assert!(plan.estimated_cost > 0);
        // Every resolved storage id is unique (the root alone owns the table).
        let mut seen = BTreeSet::new();
        for id in plan.storage.ids() {
            assert!(seen.insert(id), "resolved storage ids are unique");
        }
        // Nested transports resolve to ids in the same global table.
        fn check_transports(
            transport: &ResolvedTransport,
            table: &BTreeSet<seismic_realization::executable::ResolvedStorageId>,
        ) {
            match transport {
                ResolvedTransport::Void | ResolvedTransport::Kernel(_) => {}
                ResolvedTransport::ExecutorScalar(_) => {}
                ResolvedTransport::Storage(views) => {
                    for view in views.iter() {
                        assert!(
                            table.contains(&view.storage),
                            "the nested transport names the global table"
                        );
                    }
                }
                ResolvedTransport::Tuple(items) => {
                    for item in items.iter() {
                        check_transports(item, table);
                    }
                }
            }
        }
        let table: BTreeSet<_> = plan.storage.ids().collect();
        let first_input = plan
            .entry
            .boundary
            .inputs
            .keys()
            .next()
            .cloned()
            .expect("the root input is bound");
        check_transports(
            plan.entry.boundary.inputs.get(&first_input).expect("bound"),
            &table,
        );
    }

    #[test]
    fn reductions_have_strategies_and_never_reach_scalar_emission() {
        let program = check(&[(
            "reduce.seismic",
            "fn sum[N](x: &tensor[N] f32) -> f32:\n    return reduce(f32(x), 0, sum)\n",
        )])
        .expect("the reductions check");
        let logical = construct(
            &program,
            "sum",
            &logical_target(),
            &supports_all,
            shapes(&[("N", 16)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        let entry_choice = logical.entry_choice;
        let target = target();
        let mut family = PlanFamilyBuilder::<SynDialect>::from_logical(&logical)
            .expect("the family builder opens");
        build_universal_alternative(&mut family, entry_choice, 0, &target)
            .expect("the reduction alternative builds");
        let family = family.finish().expect("the family finishes");
        let context = Context {
            target: &target,
            precision: &PrecisionPolicy::Unconstrained,
            numerical_evidence: &[],
        };
        let plan = plan(&logical, &family, &context, Budget::default())
            .expect("the reduction family solves and resolves");
        // The reduction is one launch with a serial (ascending) strategy and
        // an exact numerical transfer: it never reaches scalar emission.
        assert!(plan.estimated_cost > 0);
        assert!(matches!(
            plan.numerical.evidence,
            seismic_lang::precision::EvidenceClass::Exact
        ));
    }

    #[test]
    fn unconsumed_obligations_refuse_to_finish() {
        let program = check(&[("kernel.seismic", KERNEL)]).expect("the kernel checks");
        let logical = construct(
            &program,
            "linear",
            &logical_target(),
            &supports_all,
            shapes(&[("M", 4), ("N", 8)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        let family = PlanFamilyBuilder::<SynDialect>::from_logical(&logical)
            .expect("the family builder opens");
        let builder = family
            .alternative(logical.entry_choice, 0)
            .expect("the builder opens");
        // Finish without consuming anything: refused.
        assert!(builder.finish_alternative().is_err());
    }

    #[test]
    fn solver_tunable_participants_reach_the_resolved_geometry() {
        let program = check(&[(
            "reduce.seismic",
            "fn sum[N](x: &tensor[N] f32) -> f32:\n    return reduce(f32(x), 0, sum)\n",
        )])
        .expect("the reductions check");
        let logical = construct(
            &program,
            "sum",
            &logical_target(),
            &supports_all,
            shapes(&[("N", 16)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        let entry_choice = logical.entry_choice;
        let target = target();
        let mut family = PlanFamilyBuilder::<SynDialect>::from_logical(&logical)
            .expect("the family builder opens");
        let mut builder = family
            .alternative(entry_choice, 0)
            .expect("the alternative builder opens");

        // A grid-stride strategy names its solver-tunable participant count
        // through the hook: unique symbol, registered as a plan parameter.
        let participants = builder
            .solver_participants(1, 8)
            .expect("the symbol allocates");
        let parameter_name = participants
            .params()
            .into_iter()
            .next()
            .expect("the symbol names one parameter");
        let graph = builder.graph().clone();
        for (node_id, node) in graph.root.nodes.ids().zip(graph.root.nodes.iter()) {
            let node_ref = NodeRef {
                region: Vec::new(),
                node: node_id,
            };
            match &node.kind {
                LogicalNodeKind::Primitive(_) => {
                    let legalized = legalize_node(&builder, &node, &target);
                    builder
                        .map_primitive(node_ref, LinearIterationMap::serial(), legalized)
                        .expect("the primitive maps");
                }
                LogicalNodeKind::Reduction(_) => {
                    let result_value = node
                        .outputs
                        .first()
                        .map(|output| output.id)
                        .expect("the reduction has a result");
                    let result = builder
                        .transport_of(result_value)
                        .expect("the result transports");
                    let iteration = LinearIterationMap::linear(
                        &[seismic_lang::types::ExtentExpr::Static(16)],
                        &BTreeMap::new(),
                    )
                    .expect("the domain checks")
                    .with_participants(participants.clone());
                    builder
                        .map_reduction(
                            node_ref,
                            ReductionStrategyTemplate {
                                topology: ReductionTopology::SerialAxis {
                                    axis: 0,
                                    length: seismic_lang::types::ExtentExpr::Static(16),
                                },
                                iteration,
                                ops: Legalized::Ops(
                                    NonEmpty::new(vec![SynOp::Compute]).expect("one opcode"),
                                ),
                                result,
                            },
                        )
                        .expect("the reduction maps");
                }
                other => panic!("unexpected node {other:?}"),
            }
        }
        for obligation in builder.pending_obligations() {
            builder
                .discharge(
                    obligation,
                    ObligationDisposition::StaticallyProved {
                        reason: "synthetic dialect".into(),
                    },
                )
                .expect("the obligation discharges");
        }
        let result_count = builder.graph().results.len();
        for ordinal in 0..result_count as u32 {
            let transport = builder
                .transport_of(match &builder.graph().results[ordinal as usize] {
                    RegionResult::Value { id, .. } => *id,
                    RegionResult::State { .. } => unreachable!("sum returns a value"),
                })
                .expect("the result transports");
            builder
                .complete_result(ordinal, transport)
                .expect("the result completes");
        }
        let alternative = builder
            .finish_alternative()
            .expect("the alternative finishes");
        family
            .add_alternative(entry_choice, alternative)
            .expect("the alternative commits");
        let family = family.finish().expect("the family finishes");

        // The parameter is registered in the family with the declared domain.
        let parameter = family
            .parameters
            .iter()
            .find(|parameter| parameter.name == parameter_name)
            .expect("the participants parameter is registered");
        assert_eq!((parameter.lower, parameter.upper), (1, 8));

        // Solve through the budgeted path: the model constrains the symbol
        // like every other tuning parameter.
        let context = Context {
            target: &target,
            precision: &PrecisionPolicy::Unconstrained,
            numerical_evidence: &[],
        };
        let plan = plan(&logical, &family, &context, Budget::default())
            .expect("the family solves and resolves");

        // The resolved geometry reflects the assigned value.
        use seismic_realization::executable::{ExecutionExpr, ResolvedLaunch, ResolvedStep};
        fn find_launches<'a, D: ExecutableDialect>(
            schedule: &'a seismic_realization::executable::ResolvedSchedule<D>,
            found: &mut Vec<&'a ResolvedLaunch<D>>,
        ) {
            for step in schedule.steps.iter() {
                match step {
                    ResolvedStep::Launch(launch) => found.push(launch),
                    ResolvedStep::Call(call) => find_launches(&call.body.schedule, found),
                    ResolvedStep::If(if_step) => {
                        find_launches(&if_step.then_schedule, found);
                        find_launches(&if_step.else_schedule, found);
                    }
                    ResolvedStep::Repeat(repeat) => find_launches(&repeat.body, found),
                }
            }
        }
        let mut launches = Vec::new();
        find_launches(&plan.entry.schedule, &mut launches);
        // The reduction's launch is the one over the 16-element domain.
        let launch = launches
            .iter()
            .copied()
            .find(|launch| launch.work_items == ExecutionExpr::Const(16))
            .expect("the reduction launch is resolved");
        let ExecutionExpr::Const(assigned) = launch.geometry.participants_per_workgroup[0] else {
            panic!("the participant count is solved, not symbolic");
        };
        assert!(
            (1..=8).contains(&assigned),
            "the assigned participant count {assigned} lies in the declared domain"
        );
        // Workgroups cover the domain: ceil(16 / participants).
        assert_eq!(
            launch.geometry.workgroups[0],
            ExecutionExpr::CeilDiv(
                Box::new(ExecutionExpr::Const(16)),
                Box::new(ExecutionExpr::Const(assigned))
            )
        );
    }

    #[test]
    fn plan_parameter_names_are_family_unique() {
        let program = check(&[(
            "reduce.seismic",
            "fn sum[N](x: &tensor[N] f32) -> f32:\n    return reduce(f32(x), 0, sum)\n",
        )])
        .expect("the reductions check");
        let logical = construct(
            &program,
            "sum",
            &logical_target(),
            &supports_all,
            shapes(&[("N", 16)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        let target = target();
        let mut family = PlanFamilyBuilder::<SynDialect>::from_logical(&logical)
            .expect("the family builder opens");
        family
            .tuning_parameter("width", 1, 8)
            .expect("the family declares the parameter");
        // Two alternatives of the same family: names never collide, and a
        // repeated name is rejected instead of capturing the symbol.
        let mut first = family.alternative(logical.entry_choice, 0).unwrap();
        let a = first.solver_participants(1, 4).unwrap();
        let b = first.solver_participants(2, 8).unwrap();
        assert_ne!(a, b, "auto-allocated participant symbols are distinct");
        assert!(first.plan_parameter("width", 1, 4).is_err());
        drop(first);
        let _ = target;
    }

    // -----------------------------------------------------------------------
    // Follow-up turn 2: staging, barriers, computed ranges, cross-call fuse,
    // transfer injection, reference-based activation.
    // -----------------------------------------------------------------------

    use seismic_realization::executable::{
        BarrierScope, BoundaryLeaf, ExecutorComputedScalar, ExecutorScalarSource,
        ExecutorScalarTemplate, Replication, ResolvedLaunch, ResolvedScheduleRepeat,
        StorageLifetime, StorageScope,
    };
    use seismic_realization::CountExpr;

    fn build_sum_family_with(
        staging: bool,
        barrier: bool,
        noted: bool,
    ) -> (
        PlanFamily<SynDialect>,
        EffectiveTargetProfile,
        LogicalProgram,
    ) {
        let program = check(&[(
            "reduce.seismic",
            "fn sum[N](x: &tensor[N] f32) -> f32:\n    return reduce(f32(x), 0, sum)\n",
        )])
        .expect("the reductions check");
        let logical = construct(
            &program,
            "sum",
            &logical_target(),
            &supports_all,
            shapes(&[("N", 16)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        let target = target();
        let mut family = PlanFamilyBuilder::<SynDialect>::from_logical(&logical)
            .expect("the family builder opens");
        let mut builder = family
            .alternative(logical.entry_choice, 0)
            .expect("the alternative builder opens");
        if staging {
            builder
                .stage_storage(
                    StorageScope::Workgroup,
                    Sym::constant(1024),
                    4,
                    Replication::Once,
                    StorageLifetime::Always,
                )
                .expect("the staging allocation declares");
        }
        let nodes = builder
            .region_nodes(&Vec::new())
            .expect("the root region lists");
        for (node_id, node) in nodes {
            let node_ref = NodeRef {
                region: Vec::new(),
                node: node_id,
            };
            match &node.kind {
                LogicalNodeKind::Primitive(_) => {
                    let legalized = legalize_node(&builder, &node, &target);
                    builder
                        .map_primitive(node_ref, LinearIterationMap::serial(), legalized)
                        .expect("the primitive maps");
                }
                LogicalNodeKind::Reduction(_) => {
                    if barrier {
                        builder
                            .barrier(BarrierScope::Subgroup)
                            .expect("the barrier plans");
                    }
                    let result_value = node
                        .outputs
                        .first()
                        .map(|output| output.id)
                        .expect("the reduction has a result");
                    let result = builder
                        .transport_of(result_value)
                        .expect("the result transports");
                    builder
                        .map_reduction(
                            node_ref,
                            ReductionStrategyTemplate {
                                topology: ReductionTopology::SerialAxis {
                                    axis: 0,
                                    length: seismic_lang::types::ExtentExpr::Static(16),
                                },
                                iteration: LinearIterationMap::serial(),
                                ops: Legalized::Ops(
                                    NonEmpty::new(vec![SynOp::Compute]).expect("one opcode"),
                                ),
                                result,
                            },
                        )
                        .expect("the reduction maps");
                }
                other => panic!("unexpected node {other:?}"),
            }
        }
        if noted {
            builder.note_numerical(NumericalTransfer::Round {
                dtype: DType::F32,
                count: CountExpr::Const(2),
            });
            builder.note_numerical(NumericalTransfer::Round {
                dtype: DType::F16,
                count: CountExpr::Const(3),
            });
        }
        for obligation in builder.pending_obligations() {
            builder
                .discharge(
                    obligation,
                    ObligationDisposition::StaticallyProved {
                        reason: "synthetic dialect".into(),
                    },
                )
                .expect("the obligation discharges");
        }
        let result_value = match &builder.graph().results[0] {
            RegionResult::Value { id, .. } => *id,
            RegionResult::State { .. } => unreachable!("sum returns a value"),
        };
        let transport = builder
            .transport_of(result_value)
            .expect("the result transports");
        builder
            .complete_result(0, transport)
            .expect("the result completes");
        let alternative = builder
            .finish_alternative()
            .expect("the alternative finishes");
        family
            .add_alternative(logical.entry_choice, alternative)
            .expect("the alternative commits");
        let family = family.finish().expect("the family finishes");
        (family, target, logical)
    }

    #[test]
    fn workgroup_staging_reaches_consequences() {
        let (family, target, logical) = build_sum_family_with(true, false, false);
        // The staged allocation is a workgroup-scoped template in the family.
        assert!(
            family
                .storages
                .iter()
                .any(|template| template.scope == StorageScope::Workgroup),
            "the staged workgroup storage is registered"
        );
        // Exact hard resources: every launch carries the exact aligned
        // workgroup bytes.
        for facts in &family.model.alternatives {
            for launch in &facts.launches {
                assert_eq!(launch.workgroup_bytes.as_constant(), Some(1024));
            }
        }
        let context = Context {
            target: &target,
            precision: &PrecisionPolicy::Unconstrained,
            numerical_evidence: &[],
        };
        let plan = plan(&logical, &family, &context, Budget::default())
            .expect("the staged family solves and resolves");
        let launch = first_launch(&plan);
        assert!(
            !launch.kernel.workgroup_storage.is_empty(),
            "the kernel references its staged storage"
        );
        assert_eq!(launch.kernel.resources.workgroup_bytes, 1024);
    }

    #[test]
    fn planned_barrier_splits_phases_of_one_launch() {
        let (family, target, logical) = build_sum_family_with(false, true, false);
        // One launch, two mapped phases, one planned barrier between them.
        let context = Context {
            target: &target,
            precision: &PrecisionPolicy::Unconstrained,
            numerical_evidence: &[],
        };
        let plan = plan(&logical, &family, &context, Budget::default())
            .expect("the barrier family solves and resolves");
        let launch = first_launch(&plan);
        let kinds: Vec<bool> = launch
            .kernel
            .steps
            .iter()
            .map(|step| {
                matches!(
                    step,
                    seismic_realization::executable::ResolvedKernelStep::Barrier { .. }
                )
            })
            .collect();
        assert_eq!(kinds.iter().filter(|barrier| **barrier).count(), 1);
        let barrier_at = kinds.iter().position(|barrier| *barrier).unwrap();
        assert!(
            kinds.len() > barrier_at + 1 && !kinds[barrier_at + 1],
            "a mapped phase follows the planned barrier (got {kinds:?})"
        );
        assert!(!kinds[0], "the launch starts with a mapped phase");
        // The barrier is a model fact (a resource/cost of the launch).
        for facts in &family.model.alternatives {
            for launch_facts in &facts.launches {
                if launch_facts.launch_condition_work_items.as_constant() == Some(16) {
                    assert_eq!(launch_facts.barriers, 1);
                }
            }
        }
    }

    #[test]
    fn noted_transfers_compose_into_the_alternative() {
        let (family, _target, _logical) = build_sum_family_with(false, false, true);
        let alternative = &family
            .choices
            .iter()
            .next()
            .unwrap()
            .alternatives
            .iter()
            .next()
            .unwrap();
        // Round(F16, 3) composed onto Round(F32, 2): the larger unit
        // roundoff (f16) governs, the counts add.
        assert_eq!(
            alternative.numerical,
            NumericalTransfer::Round {
                dtype: DType::F16,
                count: CountExpr::Const(5),
            }
        );
    }

    fn first_launch<D: ExecutableDialect>(plan: &ResolvedPlan<D>) -> &ResolvedLaunch<D> {
        fn find<'a, D: ExecutableDialect>(
            schedule: &'a seismic_realization::executable::ResolvedSchedule<D>,
        ) -> Option<&'a ResolvedLaunch<D>> {
            for step in schedule.steps.iter() {
                match step {
                    ResolvedStep::Launch(launch) => return Some(launch),
                    ResolvedStep::Call(call) => {
                        if let Some(found) = find(&call.body.schedule) {
                            return Some(found);
                        }
                    }
                    ResolvedStep::If(if_step) => {
                        if let Some(found) = find(&if_step.then_schedule) {
                            return Some(found);
                        }
                        if let Some(found) = find(&if_step.else_schedule) {
                            return Some(found);
                        }
                    }
                    ResolvedStep::Repeat(repeat) => {
                        if let Some(found) = find(&repeat.body) {
                            return Some(found);
                        }
                    }
                }
            }
            None
        }
        find(&plan.entry.schedule).expect("the plan launches")
    }

    /// Map the `add` graph with computed (CeilDiv) repeat ranges: window
    /// counts name plan parameters through the Computed executor-scalar
    /// source.
    fn map_computed_region(
        builder: &mut AlternativeBuilder<SynDialect>,
        region: &RegionPath,
        target: &EffectiveTargetProfile,
        window: &str,
    ) -> Result<(), String> {
        let graph = builder.graph().clone();
        let nodes = builder.region_nodes(region)?;
        for (node_id, node) in nodes {
            let node_ref = NodeRef {
                region: region.clone(),
                node: node_id,
            };
            if !builder.pending_nodes().contains(&node_ref) {
                continue;
            }
            match &node.kind {
                LogicalNodeKind::Primitive(_) => {
                    let legalized = legalize_node(builder, &node, target);
                    builder.map_primitive(node_ref, LinearIterationMap::serial(), legalized)?;
                }
                LogicalNodeKind::Loop(loop_node) => {
                    let bound = loop_node
                        .range
                        .bound
                        .as_static()
                        .expect("the fixture uses static bounds");
                    let start = TransportTemplate::ExecutorScalar(ExecutorScalarTemplate {
                        source: ExecutorScalarSource::Computed(ExecutorComputedScalar::Const(0)),
                        dtype: DType::I32,
                    });
                    let end = TransportTemplate::ExecutorScalar(ExecutorScalarTemplate {
                        source: ExecutorScalarSource::Computed(ExecutorComputedScalar::CeilDiv(
                            Box::new(ExecutorComputedScalar::Const(bound as i64)),
                            Box::new(ExecutorComputedScalar::Param(window.to_string())),
                        )),
                        dtype: DType::I32,
                    });
                    let mut carries = Vec::new();
                    for slot in &loop_node.carried {
                        let transport = match slot.initial {
                            seismic_lang::logical::RegionInput::Value(initial) => {
                                match builder.transport_of(initial)? {
                                    TransportTemplate::Storage(_)
                                    | TransportTemplate::ExecutorScalar(_) => {
                                        builder.transport_of(initial)?
                                    }
                                    _ => {
                                        let dtype = node
                                            .outputs
                                            .first()
                                            .and_then(|output| output.ty.scalar_dtype())
                                            .unwrap_or(DType::I32);
                                        TransportTemplate::ExecutorScalar(ExecutorScalarTemplate {
                                            source: ExecutorScalarSource::Slot(
                                                builder
                                                    .declare_executor_scalar_slot(dtype, "carry"),
                                            ),
                                            dtype,
                                        })
                                    }
                                }
                            }
                            seismic_lang::logical::RegionInput::State(token) => {
                                let storage = builder
                                    .logical_storage_of_token(token)
                                    .expect("the carried state has a storage");
                                match state_transport(builder, &graph, storage) {
                                    StateTransportTemplate::Storage(template) => {
                                        TransportTemplate::Storage(
                                            NonEmpty::new(vec![
                                                seismic_realization::executable::StorageViewTemplate {
                                                    storage: template,
                                                    access: seismic_lang::logical::Access::Exclusive,
                                                    transform: seismic_lang::logical::ViewTransform::Identity,
                                                },
                                            ])
                                            .expect("one plane"),
                                        )
                                    }
                                    StateTransportTemplate::Boundary(leaf) => {
                                        TransportTemplate::Boundary(leaf)
                                    }
                                }
                            }
                        };
                        carries.push(PhysicalCarryTemplate { transport });
                    }
                    let mut body = region.clone();
                    body.push(RegionStep::LoopBody(node_id));
                    builder.schedule_loop(
                        node_ref,
                        ExecutorRangeTemplate {
                            start,
                            end,
                            bound: loop_node.range.bound.clone(),
                        },
                        carries,
                        |body_builder| map_computed_region(body_builder, &body, target, window),
                    )?;
                }
                other => panic!("unexpected node {other:?} in the computed-range fixture"),
            }
        }
        Ok(())
    }

    #[test]
    fn computed_ceildiv_repeat_ranges_resolve() {
        let program = check(&[("kernel.seismic", KERNEL)]).expect("the kernel checks");
        // The callee graph alone: loops over static domains.
        let logical = construct(
            &program,
            "add",
            &logical_target(),
            &supports_all,
            shapes(&[("M", 4), ("N", 8)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        let target = target();
        let mut family = PlanFamilyBuilder::<SynDialect>::from_logical(&logical)
            .expect("the family builder opens");
        let mut builder = family
            .alternative(logical.entry_choice, 0)
            .expect("the alternative builder opens");
        builder
            .plan_parameter("window", 1, 4)
            .expect("the window parameter registers");
        map_computed_region(&mut builder, &Vec::new(), &target, "window")
            .expect("the computed-range alternative builds");
        for obligation in builder.pending_obligations() {
            builder
                .discharge(
                    obligation,
                    ObligationDisposition::StaticallyProved {
                        reason: "synthetic dialect".into(),
                    },
                )
                .expect("the obligation discharges");
        }
        // The entry result of `add` is the moved parameter: complete it
        // against the parameter's own ABI buffer (the fixture's focus is the
        // ranges, not the result placement).
        let result_transport = TransportTemplate::Storage(
            NonEmpty::new(vec![seismic_realization::executable::StorageViewTemplate {
                storage: builder
                    .abi_storage_of(&BoundaryLeaf::Input {
                        param: 2,
                        leaf: ValuePath::default(),
                    })
                    .expect("the moved parameter has an ABI buffer"),
                access: seismic_lang::logical::Access::Exclusive,
                transform: seismic_lang::logical::ViewTransform::Identity,
            }])
            .expect("one plane"),
        );
        builder
            .complete_result(0, result_transport)
            .expect("the result completes");
        let alternative = builder
            .finish_alternative()
            .expect("the alternative finishes");
        family
            .add_alternative(logical.entry_choice, alternative)
            .expect("the alternative commits");
        let family = family.finish().expect("the family finishes");
        let context = Context {
            target: &target,
            precision: &PrecisionPolicy::Unconstrained,
            numerical_evidence: &[],
        };
        let plan = plan(&logical, &family, &context, Budget::default())
            .expect("the computed-range family solves and resolves");
        // Find the inner repeat (bound N = 8) and check its resolved range.
        fn find_repeat<'a, D: ExecutableDialect>(
            schedule: &'a seismic_realization::executable::ResolvedSchedule<D>,
            bound: u64,
        ) -> Option<&'a ResolvedScheduleRepeat<D>> {
            for step in schedule.steps.iter() {
                match step {
                    ResolvedStep::Repeat(repeat) => {
                        if let seismic_realization::executable::ExecutionExpr::Const(value) =
                            repeat.range.bound
                        {
                            if value == bound {
                                return Some(repeat);
                            }
                        }
                        if let Some(found) = find_repeat(&repeat.body, bound) {
                            return Some(found);
                        }
                    }
                    ResolvedStep::If(if_step) => {
                        if let Some(found) = find_repeat(&if_step.then_schedule, bound) {
                            return Some(found);
                        }
                        if let Some(found) = find_repeat(&if_step.else_schedule, bound) {
                            return Some(found);
                        }
                    }
                    ResolvedStep::Call(_) => {}
                    ResolvedStep::Launch(_) => {}
                }
            }
            None
        }
        let inner = find_repeat(&plan.entry.schedule, 8).expect("the inner repeat resolves");
        use seismic_realization::executable::{ExecutionExpr, ResolvedExecutorScalar as RES};
        assert_eq!(
            inner.range.start,
            RES::Computed {
                expr: ExecutionExpr::Const(0),
                dtype: DType::I32
            }
        );
        // end = ceil(8 / window) with the solved window value.
        match &inner.range.end {
            RES::Computed {
                expr: ExecutionExpr::CeilDiv(left, right),
                dtype,
            } => {
                assert_eq!(*dtype, DType::I32);
                assert_eq!(**left, ExecutionExpr::Const(8));
                let ExecutionExpr::Const(window) = **right else {
                    panic!("the window parameter is solved, not symbolic");
                };
                assert!(
                    (1..=4).contains(&window),
                    "the solved window {window} lies in its declared domain"
                );
            }
            other => panic!("the inner repeat end is a computed CeilDiv, got {other:?}"),
        }
    }

    #[test]
    fn cross_call_fusion_consumes_child_obligations_and_drops_the_call() {
        let program = check(&[("kernel.seismic", KERNEL)]).expect("the kernel checks");
        let logical = construct(
            &program,
            "linear",
            &logical_target(),
            &supports_all,
            shapes(&[("M", 4), ("N", 8)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        let entry_choice = logical.entry_choice;
        let call_choice = logical
            .choices
            .ids()
            .find(|choice| *choice != entry_choice)
            .expect("the call created its own choice");
        let target = target();
        let mut family = PlanFamilyBuilder::<SynDialect>::from_logical(&logical)
            .expect("the family builder opens");
        let child_graph = family
            .graph_of(call_choice, 0)
            .cloned()
            .expect("the child alternative's graph is introspectable");
        let mut builder = family
            .alternative(entry_choice, 0)
            .expect("the alternative builder opens");

        // Find the call node of the entry root.
        let call_ref = builder
            .region_nodes(&Vec::new())
            .expect("the root region lists")
            .into_iter()
            .find(|(_, node)| matches!(node.kind, LogicalNodeKind::Call(_)))
            .map(|(node_id, _)| NodeRef {
                region: Vec::new(),
                node: node_id,
            })
            .expect("the entry body is one call");
        // Build the fused boundary: tensor view values and states transport
        // directly to the caller's ABI-backed storage.
        let graph = builder.graph().clone();
        let call_node = match &graph
            .root
            .nodes
            .get(call_ref.node)
            .expect("the call node exists")
            .kind
        {
            LogicalNodeKind::Call(call_node) => call_node.clone(),
            _ => unreachable!("checked above"),
        };
        let mut boundary = BoundaryTemplates::default();
        for input in &call_node.boundary_inputs {
            let leaf = BoundaryLeaf::Input {
                param: input.param,
                leaf: input.path.clone(),
            };
            match input.kind {
                seismic_lang::logical::BoundaryInputKind::Value(value) => {
                    boundary
                        .inputs
                        .insert(leaf, builder.transport_of(value).unwrap());
                }
                seismic_lang::logical::BoundaryInputKind::Shared {
                    value,
                    state: token,
                }
                | seismic_lang::logical::BoundaryInputKind::Exclusive {
                    value,
                    state: token,
                }
                | seismic_lang::logical::BoundaryInputKind::Move {
                    value,
                    state: token,
                } => {
                    let storage = builder
                        .logical_storage_of_token(token)
                        .expect("the boundary state has a storage");
                    boundary
                        .states
                        .insert(leaf.clone(), state_transport(&builder, &graph, storage));
                    boundary
                        .inputs
                        .insert(leaf, builder.transport_of(value).unwrap());
                }
            }
        }
        for result in &call_node.boundary_results {
            let leaf = BoundaryLeaf::Result {
                leaf: result.path.clone(),
            };
            match &result.kind {
                seismic_lang::logical::BoundaryResultKind::Value(value) => {
                    boundary
                        .results
                        .insert(leaf, builder.transport_of(*value).unwrap());
                }
                seismic_lang::logical::BoundaryResultKind::Storage { storage, .. } => {
                    boundary.results.insert(
                        leaf,
                        TransportTemplate::Storage(
                            NonEmpty::new(vec![
                                seismic_realization::executable::StorageViewTemplate {
                                    storage: builder
                                        .storage_of(*storage)
                                        .expect("the occurrence-owned result template"),
                                    access: seismic_lang::logical::Access::Exclusive,
                                    transform: seismic_lang::logical::ViewTransform::Identity,
                                },
                            ])
                            .expect("one plane"),
                        ),
                    );
                }
                seismic_lang::logical::BoundaryResultKind::State(token) => {
                    let storage = builder
                        .logical_storage_of_token(*token)
                        .expect("the result state has a storage");
                    boundary
                        .states
                        .insert(leaf, state_transport(&builder, &graph, storage));
                }
            }
        }
        // The cross-call fuse: the child interval is inlined into this
        // alternative through boundary substitution; the child's pending
        // obligations become this alternative's own.
        builder
            .fuse_call(call_ref.clone(), &child_graph, boundary)
            .expect("the call fuses across the boundary");
        // Map the imported child regions (under the FusedCall path prefix).
        let fused_path = vec![RegionStep::FusedCall(call_ref.node)];
        map_region(&mut builder, &fused_path, &target).expect("the imported interval maps");
        // The caller's own remaining nodes map as usual.
        map_region(&mut builder, &Vec::new(), &target).expect("the caller maps");
        for obligation in builder.pending_obligations() {
            builder
                .discharge(
                    obligation,
                    ObligationDisposition::StaticallyProved {
                        reason: "synthetic dialect".into(),
                    },
                )
                .expect("the obligation discharges");
        }
        let result_count = builder.graph().results.len();
        for ordinal in 0..result_count as u32 {
            let transport = match &builder.graph().results[ordinal as usize] {
                RegionResult::Value { id, .. } => builder
                    .transport_of(*id)
                    .expect("the fused result transports"),
                RegionResult::State { .. } => unreachable!("linear returns a value"),
            };
            builder
                .complete_result(ordinal, transport)
                .expect("the fused result completes");
        }
        // Finishing proves both caller AND child pending sets are empty.
        let alternative = builder
            .finish_alternative()
            .expect("the fused alternative consumes caller and child obligations");
        family
            .add_alternative(entry_choice, alternative)
            .expect("the alternative commits");
        let family = family.finish().expect("the family finishes");
        let context = Context {
            target: &target,
            precision: &PrecisionPolicy::Unconstrained,
            numerical_evidence: &[],
        };
        let plan = plan(&logical, &family, &context, Budget::default())
            .expect("the fused family solves and resolves");
        // The call is gone: the schedule contains launches only, no Call step.
        fn count<D: ExecutableDialect>(
            schedule: &seismic_realization::executable::ResolvedSchedule<D>,
            calls: &mut usize,
            launches: &mut usize,
        ) {
            for step in schedule.steps.iter() {
                match step {
                    ResolvedStep::Launch(_) => *launches += 1,
                    ResolvedStep::Call(_) => *calls += 1,
                    ResolvedStep::If(if_step) => {
                        count(&if_step.then_schedule, calls, launches);
                        count(&if_step.else_schedule, calls, launches);
                    }
                    ResolvedStep::Repeat(repeat) => count(&repeat.body, calls, launches),
                }
            }
        }
        let mut calls = 0;
        let mut launches = 0;
        count(&plan.entry.schedule, &mut calls, &mut launches);
        assert_eq!(calls, 0, "the fused interval contains no call step");
        assert!(launches > 0, "the fused interval launches");
    }

    #[test]
    fn reference_based_activation_shrinks_the_fused_arena() {
        // The invoke-based family allocates the occurrence-owned call-result
        // arena storage (4 * 8 * f32 = 128 bytes).
        let program = check(&[("kernel.seismic", KERNEL)]).expect("the kernel checks");
        let logical = construct(
            &program,
            "linear",
            &logical_target(),
            &supports_all,
            shapes(&[("M", 4), ("N", 8)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        let entry_choice = logical.entry_choice;
        let call_choice = logical
            .choices
            .ids()
            .find(|choice| *choice != entry_choice)
            .expect("the call created its own choice");
        let target = target();
        let mut invoke_family = PlanFamilyBuilder::<SynDialect>::from_logical(&logical)
            .expect("the family builder opens");
        for choice in [entry_choice, call_choice] {
            for ordinal in 0..logical.choice(choice).alternatives.iter().count() as u32 {
                build_universal_alternative(&mut invoke_family, choice, ordinal, &target)
                    .expect("the invoke alternative builds");
            }
        }
        let invoke_family = invoke_family.finish().expect("the invoke family finishes");
        let context = Context {
            target: &target,
            precision: &PrecisionPolicy::Unconstrained,
            numerical_evidence: &[],
        };
        let invoke_plan = plan(&logical, &invoke_family, &context, Budget::default())
            .expect("the invoke family solves and resolves");
        let invoke_arena = invoke_plan.internal_arena.bytes;
        assert_eq!(
            invoke_arena, 128,
            "the invoke-based family materializes the call result in the arena"
        );

        // The fused family routes the call result directly to the public ABI
        // result buffer: the occurrence-owned arena storage is never
        // referenced and activates nothing.
        let mut fused_family = PlanFamilyBuilder::<SynDialect>::from_logical(&logical)
            .expect("the family builder opens");
        let child_graph = fused_family
            .graph_of(call_choice, 0)
            .cloned()
            .expect("the child graph is introspectable");
        let mut builder = fused_family
            .alternative(entry_choice, 0)
            .expect("the alternative builder opens");
        let call_ref = builder
            .region_nodes(&Vec::new())
            .expect("the root region lists")
            .into_iter()
            .find(|(_, node)| matches!(node.kind, LogicalNodeKind::Call(_)))
            .map(|(node_id, _)| NodeRef {
                region: Vec::new(),
                node: node_id,
            })
            .expect("the entry body is one call");
        let graph = builder.graph().clone();
        let call_node = match &graph
            .root
            .nodes
            .get(call_ref.node)
            .expect("the call node exists")
            .kind
        {
            LogicalNodeKind::Call(call_node) => call_node.clone(),
            _ => unreachable!("checked above"),
        };
        let mut boundary = BoundaryTemplates::default();
        for input in &call_node.boundary_inputs {
            let leaf = BoundaryLeaf::Input {
                param: input.param,
                leaf: input.path.clone(),
            };
            match input.kind {
                seismic_lang::logical::BoundaryInputKind::Value(value) => {
                    boundary
                        .inputs
                        .insert(leaf, builder.transport_of(value).unwrap());
                }
                seismic_lang::logical::BoundaryInputKind::Shared {
                    value,
                    state: token,
                }
                | seismic_lang::logical::BoundaryInputKind::Exclusive {
                    value,
                    state: token,
                }
                | seismic_lang::logical::BoundaryInputKind::Move {
                    value,
                    state: token,
                } => {
                    let storage = builder
                        .logical_storage_of_token(token)
                        .expect("the boundary state has a storage");
                    boundary
                        .states
                        .insert(leaf.clone(), state_transport(&builder, &graph, storage));
                    boundary
                        .inputs
                        .insert(leaf, builder.transport_of(value).unwrap());
                }
            }
        }
        let abi_result = builder
            .abi_storage_of(&BoundaryLeaf::Result {
                leaf: ValuePath::default(),
            })
            .expect("the entry result has a public ABI buffer");
        for result in &call_node.boundary_results {
            let leaf = BoundaryLeaf::Result {
                leaf: result.path.clone(),
            };
            match &result.kind {
                seismic_lang::logical::BoundaryResultKind::Value(value) => {
                    boundary
                        .results
                        .insert(leaf, builder.transport_of(*value).unwrap());
                }
                seismic_lang::logical::BoundaryResultKind::Storage { .. } => {
                    // Route the result directly to the public ABI buffer: the
                    // occurrence-owned arena storage is never referenced.
                    boundary.results.insert(
                        leaf,
                        TransportTemplate::Storage(
                            NonEmpty::new(vec![
                                seismic_realization::executable::StorageViewTemplate {
                                    storage: abi_result,
                                    access: seismic_lang::logical::Access::Exclusive,
                                    transform: seismic_lang::logical::ViewTransform::Identity,
                                },
                            ])
                            .expect("one plane"),
                        ),
                    );
                }
                seismic_lang::logical::BoundaryResultKind::State(token) => {
                    let storage = builder
                        .logical_storage_of_token(*token)
                        .expect("the result state has a storage");
                    boundary
                        .states
                        .insert(leaf, state_transport(&builder, &graph, storage));
                }
            }
        }
        builder
            .fuse_call(call_ref.clone(), &child_graph, boundary)
            .expect("the call fuses");
        let fused_path = vec![RegionStep::FusedCall(call_ref.node)];
        map_region(&mut builder, &fused_path, &target).expect("the imported interval maps");
        map_region(&mut builder, &Vec::new(), &target).expect("the caller maps");
        for obligation in builder.pending_obligations() {
            builder
                .discharge(
                    obligation,
                    ObligationDisposition::StaticallyProved {
                        reason: "synthetic dialect".into(),
                    },
                )
                .expect("the obligation discharges");
        }
        let result_count = builder.graph().results.len();
        for ordinal in 0..result_count as u32 {
            let transport = match &builder.graph().results[ordinal as usize] {
                RegionResult::Value { id, .. } => builder
                    .transport_of(*id)
                    .expect("the fused result transports"),
                RegionResult::State { .. } => unreachable!("linear returns a value"),
            };
            builder
                .complete_result(ordinal, transport)
                .expect("the fused result completes");
        }
        let alternative = builder
            .finish_alternative()
            .expect("the fused alternative finishes");
        fused_family
            .add_alternative(entry_choice, alternative)
            .expect("the alternative commits");
        let fused_family = fused_family.finish().expect("the fused family finishes");
        assert!(
            !fused_family
                .storages
                .iter()
                .any(|template| template.scope == StorageScope::DeviceArena),
            "the unreferenced arena template is dropped from the family table"
        );
        // And the fused family still solves and resolves end to end.
        let plan = plan(&logical, &fused_family, &context, Budget::default())
            .expect("the pruned family solves and resolves");
        assert_eq!(
            plan.internal_arena.bytes, 0,
            "the unreferenced arena storage activates nothing"
        );
        assert!(plan.estimated_cost > 0);
    }

    // -----------------------------------------------------------------------
    // Turn 3: retained predicates, view transforms, packed bytes, declared
    // status fields, retained kernel template ids.
    // -----------------------------------------------------------------------

    #[test]
    fn runtime_check_predicates_are_retained_in_the_status_binding() {
        let program = check(&[(
            "safety.seismic",
            "fn f[N](x: &tensor[N] f32, i: index[N]) -> f32:\n    return x[i] + 1.0 / x[0]\n",
        )])
        .expect("the sources check");
        let logical = construct(
            &program,
            "f",
            &logical_target(),
            &supports_all,
            shapes(&[("N", 4)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        let target = target();
        let mut family = PlanFamilyBuilder::<SynDialect>::from_logical(&logical)
            .expect("the family builder opens");
        let mut builder = family
            .alternative(logical.entry_choice, 0)
            .expect("the alternative builder opens");
        let nodes = builder
            .region_nodes(&Vec::new())
            .expect("the root region lists");
        for (node_id, node) in nodes {
            let node_ref = NodeRef {
                region: Vec::new(),
                node: node_id,
            };
            match &node.kind {
                LogicalNodeKind::Primitive(_) => {
                    let legalized = legalize_node(&builder, &node, &target);
                    builder
                        .map_primitive(node_ref, LinearIterationMap::serial(), legalized)
                        .expect("the primitive maps");
                }
                other => panic!("unexpected node {other:?}"),
            }
        }
        // One obligation discharged as a runtime check with real predicate
        // opcodes: they must survive into the resolved status binding.
        let obligation = builder
            .pending_obligations()
            .into_iter()
            .next()
            .expect("the element read created an obligation");
        builder
            .discharge(
                obligation,
                ObligationDisposition::RuntimeChecked {
                    predicate: Legalized::Ops(
                        NonEmpty::new(vec![SynOp::Predicate]).expect("one opcode"),
                    ),
                    inactive: seismic_realization::executable::InactiveBehavior::Skip,
                },
            )
            .expect("the runtime check discharges");
        for obligation in builder.pending_obligations() {
            builder
                .discharge(
                    obligation,
                    ObligationDisposition::StaticallyProved {
                        reason: "synthetic dialect".into(),
                    },
                )
                .expect("the obligation discharges");
        }
        let result_value = match &builder.graph().results[0] {
            RegionResult::Value { id, .. } => *id,
            RegionResult::State { .. } => unreachable!("sum returns a value"),
        };
        let transport = builder
            .transport_of(result_value)
            .expect("the result transports");
        builder
            .complete_result(0, transport)
            .expect("the result completes");
        let alternative = builder
            .finish_alternative()
            .expect("the alternative finishes");
        family
            .add_alternative(logical.entry_choice, alternative)
            .expect("the alternative commits");
        let family = family.finish().expect("the family finishes");
        let context = Context {
            target: &target,
            precision: &PrecisionPolicy::Unconstrained,
            numerical_evidence: &[],
        };
        let plan = plan(&logical, &family, &context, Budget::default())
            .expect("the family solves and resolves");
        let status = plan.abi.status.as_ref().expect("the status binding exists");
        let checked = status
            .fields
            .iter()
            .find(|field| field.predicate.is_some())
            .expect("the runtime check predicate is retained");
        assert_eq!(
            checked.predicate,
            Some(NonEmpty::new(vec![SynOp::Predicate]).expect("one opcode"))
        );
    }

    #[test]
    fn view_transforms_reach_resolved_transports() {
        let program = check(&[(
            "views.seismic",
            "fn f[N](x: &tensor[N, N] f32) -> tensor[N] f32:\n    let v = x[0:N, 0]\n    return to_owned(v)\n",
        )])
        .expect("the sources check");
        let logical = construct(
            &program,
            "f",
            &logical_target(),
            &supports_all,
            shapes(&[("N", 4)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        let target = target();
        let mut family = PlanFamilyBuilder::<SynDialect>::from_logical(&logical)
            .expect("the family builder opens");
        build_universal_alternative(&mut family, logical.entry_choice, 0, &target)
            .expect("the alternative builds");
        let family = family.finish().expect("the family finishes");
        let context = Context {
            target: &target,
            precision: &PrecisionPolicy::Unconstrained,
            numerical_evidence: &[],
        };
        let plan = plan(&logical, &family, &context, Budget::default())
            .expect("the family solves and resolves");
        // Some bound view of the sliced parameter carries the slice
        // transform, so backends address in storage coordinates.
        fn find_transform<D: ExecutableDialect>(
            schedule: &seismic_realization::executable::ResolvedSchedule<D>,
        ) -> Option<seismic_lang::logical::ViewTransform> {
            use seismic_realization::executable::{
                ResolvedKernelStep, ResolvedStep, ResolvedTransport,
            };
            for step in schedule.steps.iter() {
                match step {
                    ResolvedStep::Launch(launch) => {
                        for kernel_step in launch.kernel.steps.iter() {
                            if let ResolvedKernelStep::Mapped { bindings, .. } = kernel_step {
                                for (_, transport) in bindings {
                                    if let ResolvedTransport::Storage(views) = transport {
                                        for view in views.iter() {
                                            if !matches!(
                                                view.transform,
                                                seismic_lang::logical::ViewTransform::Identity
                                            ) {
                                                return Some(view.transform.clone());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    ResolvedStep::Call(call) => {
                        if let Some(found) = find_transform(&call.body.schedule) {
                            return Some(found);
                        }
                    }
                    ResolvedStep::If(if_step) => {
                        if let Some(found) = find_transform(&if_step.then_schedule) {
                            return Some(found);
                        }
                        if let Some(found) = find_transform(&if_step.else_schedule) {
                            return Some(found);
                        }
                    }
                    ResolvedStep::Repeat(repeat) => {
                        if let Some(found) = find_transform(&repeat.body) {
                            return Some(found);
                        }
                    }
                }
            }
            None
        }
        let transform =
            find_transform(&plan.entry.schedule).expect("a sliced view is bound in a launch");
        assert!(matches!(
            transform,
            seismic_lang::logical::ViewTransform::Slice { .. }
        ));
    }

    #[test]
    fn packed_storage_bytes_use_the_representation_planes() {
        use seismic_realization::executable::storage_bytes;
        let shape = TensorType::new(
            vec![seismic_lang::types::ExtentExpr::Static(256)],
            seismic_lang::types::Elem::Repr("q4g64".into()),
        );
        let bytes = storage_bytes(&shape).expect("the packed storage sizes");
        let total = bytes.as_constant().expect("static extents give a constant");
        // The authoritative total: the registry's own per-plane bytes over
        // the packed-axis value count (one row here).
        let expected: u64 = seismic_lang::repr::lookup("q4g64")
            .expect("the representation exists")
            .planes()
            .into_iter()
            .map(|plane| plane.bytes(256).expect("static values size every plane"))
            .sum();
        assert_eq!(
            total,
            i64::try_from(expected).expect("the plane total fits i64")
        );
        assert!(total > 0);
        // Dense storages keep their exact element sizing.
        let dense = TensorType::new(
            vec![seismic_lang::types::ExtentExpr::Static(16)],
            seismic_lang::types::Elem::Dtype(DType::F32),
        );
        assert_eq!(storage_bytes(&dense).unwrap().as_constant(), Some(64));
    }

    #[test]
    fn declared_status_fields_reach_the_status_binding() {
        let (family, target, logical) = build_sum_family_with(false, false, false);
        // Rebuild with one declared precondition field.
        let mut family = {
            // consume the built family's parts: rebuild from scratch instead
            drop(family);
            let program = check(&[(
                "reduce.seismic",
                "fn sum[N](x: &tensor[N] f32) -> f32:\n    return reduce(f32(x), 0, sum)\n",
            )])
            .expect("the reductions check");
            let logical = construct(
                &program,
                "sum",
                &logical_target(),
                &supports_all,
                shapes(&[("N", 16)]),
                BTreeMap::new(),
            )
            .expect("construction succeeds");
            let mut family = PlanFamilyBuilder::<SynDialect>::from_logical(&logical)
                .expect("the family builder opens");
            let mut builder = family
                .alternative(logical.entry_choice, 0)
                .expect("the alternative builder opens");
            let precondition = builder.status_field();
            let nodes = builder
                .region_nodes(&Vec::new())
                .expect("the root region lists");
            for (node_id, node) in nodes {
                let node_ref = NodeRef {
                    region: Vec::new(),
                    node: node_id,
                };
                match &node.kind {
                    LogicalNodeKind::Primitive(_) => {
                        let legalized = legalize_node(&builder, &node, &target);
                        builder
                            .map_primitive(node_ref, LinearIterationMap::serial(), legalized)
                            .expect("the primitive maps");
                    }
                    LogicalNodeKind::Reduction(_) => {
                        let result_value = node
                            .outputs
                            .first()
                            .map(|output| output.id)
                            .expect("the reduction has a result");
                        let result = builder
                            .transport_of(result_value)
                            .expect("the result transports");
                        builder
                            .map_reduction(
                                node_ref,
                                ReductionStrategyTemplate {
                                    topology: ReductionTopology::SerialAxis {
                                        axis: 0,
                                        length: seismic_lang::types::ExtentExpr::Static(16),
                                    },
                                    iteration: LinearIterationMap::serial(),
                                    ops: Legalized::Ops(
                                        NonEmpty::new(vec![SynOp::Compute]).expect("one opcode"),
                                    ),
                                    result,
                                },
                            )
                            .expect("the reduction maps");
                    }
                    other => panic!("unexpected node {other:?}"),
                }
            }
            for obligation in builder.pending_obligations() {
                builder
                    .discharge(
                        obligation,
                        ObligationDisposition::StaticallyProved {
                            reason: "synthetic dialect".into(),
                        },
                    )
                    .expect("the obligation discharges");
            }
            let result_value = match &builder.graph().results[0] {
                RegionResult::Value { id, .. } => *id,
                RegionResult::State { .. } => unreachable!("sum returns a value"),
            };
            let transport = builder
                .transport_of(result_value)
                .expect("the result transports");
            builder
                .complete_result(0, transport)
                .expect("the result completes");
            let alternative = builder
                .finish_alternative()
                .expect("the alternative finishes");
            family
                .add_alternative(logical.entry_choice, alternative)
                .expect("the alternative commits");
            family
        };
        let family = family.finish().expect("the family finishes");
        let context = Context {
            target: &target,
            precision: &PrecisionPolicy::Unconstrained,
            numerical_evidence: &[],
        };
        let program = check(&[(
            "reduce.seismic",
            "fn sum[N](x: &tensor[N] f32) -> f32:\n    return reduce(f32(x), 0, sum)\n",
        )])
        .expect("the reductions check");
        let logical = construct(
            &program,
            "sum",
            &logical_target(),
            &supports_all,
            shapes(&[("N", 16)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        let plan = plan(&logical, &family, &context, Budget::default())
            .expect("the family solves and resolves");
        let status = plan.abi.status.as_ref().expect("the status binding exists");
        // The declared precondition field is present and sized.
        assert!(status.bytes >= 4);
        assert!(
            status.fields.iter().any(|field| field.predicate.is_none()
                && field.node.node == seismic_lang::logical::NodeId(u32::MAX)),
            "the strategy-declared precondition field is retained"
        );
    }

    #[test]
    fn fused_kernel_template_ids_survive_resolution() {
        let source = "fn f[N](x: &tensor[N] f32) -> tensor[N] f32:\n    let v = f32(x)\n    let a = v * 2.0\n    let b = v + 1.0\n    return a + b\n";
        let program = check(&[("s.seismic", source)]).expect("the kernel checks");
        let logical = construct(
            &program,
            "f",
            &logical_target(),
            &supports_all,
            shapes(&[("N", 8)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        let target = target();
        let mut family = PlanFamilyBuilder::<SynDialect>::from_logical(&logical)
            .expect("the family builder opens");
        let mut builder = family
            .alternative(logical.entry_choice, 0)
            .expect("the alternative builder opens");
        // Fuse every root node: the shared operand's consumers are all in the
        // set, so it becomes a kernel-local value.
        let nodes = builder
            .region_nodes(&Vec::new())
            .expect("the root region lists");
        let node_refs: Vec<NodeRef> = nodes
            .into_iter()
            .map(|(node_id, _)| NodeRef {
                region: Vec::new(),
                node: node_id,
            })
            .collect();
        let ops = nodes_fused_ops(&builder, &node_refs, &target);
        builder
            .fuse(
                node_refs,
                FusedStrategyTemplate {
                    iteration: LinearIterationMap::linear(
                        &[seismic_lang::types::ExtentExpr::Static(8)],
                        &BTreeMap::new(),
                    )
                    .expect("the domain checks")
                    .with_participants(Sym::constant(1)),
                    ops: Legalized::Ops(ops),
                },
            )
            .expect("the fusion builds");
        for obligation in builder.pending_obligations() {
            builder
                .discharge(
                    obligation,
                    ObligationDisposition::StaticallyProved {
                        reason: "synthetic dialect".into(),
                    },
                )
                .expect("the obligation discharges");
        }
        let result_value = match &builder.graph().results[0] {
            RegionResult::Value { id, .. } => *id,
            RegionResult::State { .. } => unreachable!("f returns a value"),
        };
        let transport = builder
            .transport_of(result_value)
            .expect("the result transports");
        builder
            .complete_result(0, transport)
            .expect("the result completes");
        let alternative = builder
            .finish_alternative()
            .expect("the alternative finishes");
        family
            .add_alternative(logical.entry_choice, alternative)
            .expect("the alternative commits");
        let family = family.finish().expect("the family finishes");
        let context = Context {
            target: &target,
            precision: &PrecisionPolicy::Unconstrained,
            numerical_evidence: &[],
        };
        let plan = plan(&logical, &family, &context, Budget::default())
            .expect("the family solves and resolves");
        let launch = first_launch(&plan);
        // Kernel-local bindings resolve through the retained template map.
        use seismic_realization::executable::{ResolvedKernelStep, ResolvedTransport};
        let mut kernel_bindings = 0;
        for kernel_step in launch.kernel.steps.iter() {
            if let ResolvedKernelStep::Mapped { bindings, .. } = kernel_step {
                for (_, transport) in bindings {
                    if let ResolvedTransport::Kernel(resolved) = transport {
                        kernel_bindings += 1;
                        assert!(
                            launch.value_map.values().any(|id| *id == *resolved),
                            "the resolved kernel id {resolved:?} names a retained template"
                        );
                    }
                }
            }
        }
        assert!(
            kernel_bindings > 0,
            "the fused shared operand binds as kernel-local SSA"
        );
        assert!(!launch.value_map.is_empty());
    }

    /// One legalized opcode per fused node (the synthetic dialect maps every
    /// primitive to one opcode).
    fn nodes_fused_ops(
        builder: &AlternativeBuilder<SynDialect>,
        nodes: &[NodeRef],
        target: &EffectiveTargetProfile,
    ) -> NonEmpty<SynOp> {
        let mut ops = Vec::new();
        for node in nodes {
            let logical = builder
                .graph()
                .root
                .nodes
                .get(node.node)
                .cloned()
                .expect("the node exists");
            let legalized = legalize_node(builder, &logical, target);
            ops.extend(legalized.ops().expect("legalized").iter().cloned());
        }
        NonEmpty::new(ops).expect("one opcode per node")
    }

    // -----------------------------------------------------------------------
    // Turn 4: ownership-driven parameter access; owned/moved entry results
    // promote to public ABI result buffers.
    // -----------------------------------------------------------------------

    #[test]
    fn entry_parameter_access_follows_ownership() {
        // scale takes a shared `&` tensor, an exclusive `&mut` tensor, and a
        // scalar: the resolved root transports must carry the matching
        // access modes.
        let program = check(&[(
            "scale.seismic",
            "fn scale[N](x: &tensor[N] f32, y: &mut tensor[N] f32, a: f32):\n    parallel for i in 0..N:\n        y[i] = x[i] * a + 0.5 / a\n    return\n",
        )])
        .expect("the kernel checks");
        let logical = construct(
            &program,
            "scale",
            &logical_target(),
            &supports_all,
            shapes(&[("N", 24)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        let target = target();
        let mut family = PlanFamilyBuilder::<SynDialect>::from_logical(&logical)
            .expect("the family builder opens");
        build_universal_alternative(&mut family, logical.entry_choice, 0, &target)
            .expect("the alternative builds");
        let family = family.finish().expect("the family finishes");
        let context = Context {
            target: &target,
            precision: &PrecisionPolicy::Unconstrained,
            numerical_evidence: &[],
        };
        let plan = plan(&logical, &family, &context, Budget::default())
            .expect("the family solves and resolves");
        let x_transport = plan
            .entry
            .boundary
            .inputs
            .get(&BoundaryLeaf::Input {
                param: 0,
                leaf: ValuePath::default(),
            })
            .expect("x is bound");
        let y_transport = plan
            .entry
            .boundary
            .inputs
            .get(&BoundaryLeaf::Input {
                param: 1,
                leaf: ValuePath::default(),
            })
            .expect("y is bound");
        let access_of = |transport: &ResolvedTransport| match transport {
            ResolvedTransport::Storage(views) => views.first().access,
            _ => panic!("a tensor parameter binds storage"),
        };
        assert_eq!(
            access_of(x_transport),
            seismic_lang::logical::Access::Shared,
            "a shared borrow reads"
        );
        assert_eq!(
            access_of(y_transport),
            seismic_lang::logical::Access::Exclusive,
            "an exclusive borrow is a write path"
        );
    }

    #[test]
    fn owned_entry_results_resolve_to_public_abi_buffers() {
        // to_owned creates an owned storage; the returned tensor must be a
        // public ABI result buffer, and the arena must exclude it.
        let program = check(&[(
            "owned.seismic",
            "fn f[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n",
        )])
        .expect("the kernel checks");
        let logical = construct(
            &program,
            "f",
            &logical_target(),
            &supports_all,
            shapes(&[("N", 8)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        let target = target();
        let mut family = PlanFamilyBuilder::<SynDialect>::from_logical(&logical)
            .expect("the family builder opens");
        build_universal_alternative(&mut family, logical.entry_choice, 0, &target)
            .expect("the alternative builds");
        let family = family.finish().expect("the family finishes");
        let context = Context {
            target: &target,
            precision: &PrecisionPolicy::Unconstrained,
            numerical_evidence: &[],
        };
        let plan = plan(&logical, &family, &context, Budget::default())
            .expect("the family solves and resolves");
        let result_buffer = plan
            .abi
            .buffers
            .iter()
            .find(|buffer| {
                matches!(
                    buffer.role,
                    seismic_realization::executable::AbiRole::Result
                )
            })
            .expect("the owned result has a public ABI result buffer");
        assert_eq!(result_buffer.bytes, 32);
        // The result storage is ABI-placed, not arena.
        let result_storage = plan
            .storage
            .iter()
            .find(|storage| {
                matches!(
                    storage.placement,
                    seismic_realization::executable::ResolvedStoragePlacement::Abi {
                        binding,
                    } if binding == result_buffer.binding
                )
            })
            .expect("the result buffer names resolved storage");
        assert!(matches!(
            result_storage.placement,
            seismic_realization::executable::ResolvedStoragePlacement::Abi { .. }
        ));
        // The arena holds internals only: the to_owned chain's intermediate
        // (32 bytes) stays, the promoted result does not.
        assert_eq!(
            plan.internal_arena.bytes, 32,
            "the arena excludes the promoted result buffer"
        );
        assert!(
            plan.storage.iter().all(|storage| !matches!(
                storage.placement,
                seismic_realization::executable::ResolvedStoragePlacement::Arena { .. }
            ) || !matches!(
                storage.placement,
                seismic_realization::executable::ResolvedStoragePlacement::Abi {
                    binding,
                } if binding == result_buffer.binding
            )),
            "no arena storage carries the result binding"
        );
    }

    #[test]
    fn moved_parameter_results_get_their_own_result_buffer() {
        // A moved-in owned parameter returned as the result: the entry ABI
        // carries the parameter buffer AND a distinct public result buffer
        // for the result leaf (results are distinct from parameters).
        let program = check(&[(
            "fill.seismic",
            "fn fill[M](out: tensor[M] f32) -> tensor[M] f32:\n    let mut acc = out\n    parallel for i in 0..M:\n        acc[i] = f32(1.0)\n    return acc\n",
        )])
        .expect("the kernel checks");
        let logical = construct(
            &program,
            "fill",
            &logical_target(),
            &supports_all,
            shapes(&[("M", 4)]),
            BTreeMap::new(),
        )
        .expect("construction succeeds");
        let target = target();
        let mut family = PlanFamilyBuilder::<SynDialect>::from_logical(&logical)
            .expect("the family builder opens");
        build_universal_alternative(&mut family, logical.entry_choice, 0, &target)
            .expect("the alternative builds");
        let family = family.finish().expect("the family finishes");
        let context = Context {
            target: &target,
            precision: &PrecisionPolicy::Unconstrained,
            numerical_evidence: &[],
        };
        let plan = plan(&logical, &family, &context, Budget::default())
            .expect("the family solves and resolves");
        let roles: Vec<&str> = plan
            .abi
            .buffers
            .iter()
            .map(|buffer| match buffer.role {
                seismic_realization::executable::AbiRole::Parameter { .. } => "param",
                seismic_realization::executable::AbiRole::Result => "result",
            })
            .collect();
        assert_eq!(
            roles,
            vec!["param", "result"],
            "the parameter buffer and the result buffer are distinct ABI buffers"
        );
        let param_binding = plan
            .abi
            .buffers
            .iter()
            .find(|buffer| {
                matches!(
                    buffer.role,
                    seismic_realization::executable::AbiRole::Parameter { ordinal: 0 }
                )
            })
            .expect("the parameter buffer exists");
        let result_binding = plan
            .abi
            .buffers
            .iter()
            .find(|buffer| {
                matches!(
                    buffer.role,
                    seismic_realization::executable::AbiRole::Result
                )
            })
            .expect("the result buffer exists");
        assert_ne!(
            param_binding.binding, result_binding.binding,
            "results are distinct from parameters"
        );
        assert_eq!(plan.internal_arena.bytes, 0);
    }
}
