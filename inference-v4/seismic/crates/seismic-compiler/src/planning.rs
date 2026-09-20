//! Single-solve selection of a complete executable plan family.

use magnitude_solver::{
    model::{Arithmetic, Constraint, Cost, LinearTerm, Literal},
    Algorithm, Domain, Limits, Model, ModelBuilder, NeighborhoodOptions, Options, Outcome, Search,
    VarId,
};
use seismic_lang::{
    logical::{
        ChoiceId, LogicalCompilationIdentity, LogicalExtent, LogicalProgram,
        StructuralConstraintKind, StructuralSiteRef,
    },
    precision::{NumericalAssessment, PrecisionPolicy},
    sym::{Atom, Sym},
};
use seismic_realization::executable::{
    ExecutableDialect, ExecutableTargetProfile, PlanAlternativeFacts, PlanAssignment, PlanFamily,
    PlanSelection, PrecisionResolutionIdentity, ResolvedPlan,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Duration,
};

const INACTIVE: i64 = -1;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Strategy {
    #[default]
    Exact,
    Greedy,
}

#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub work: u64,
    pub time: Option<Duration>,
    pub strategy: Strategy,
}
impl Default for Budget {
    fn default() -> Self {
        Self {
            work: 200_000,
            time: Some(Duration::from_secs(2)),
            strategy: Strategy::Exact,
        }
    }
}

#[derive(Debug)]
pub enum PlanningError {
    Invalid(String),
    Infeasible(String),
    Incomplete(String),
    Solver(String),
}
impl std::fmt::Display for PlanningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(v) => write!(f, "invalid executable family: {v}"),
            Self::Infeasible(v) => write!(f, "executable planning is infeasible: {v}"),
            Self::Incomplete(v) => write!(f, "executable planning is incomplete: {v}"),
            Self::Solver(v) => write!(f, "executable solver defect: {v}"),
        }
    }
}
impl std::error::Error for PlanningError {}

pub struct Context<'a, D: ExecutableDialect> {
    pub target: &'a ExecutableTargetProfile<D::Capability>,
    pub precision: &'a PrecisionPolicy,
    pub numerical_evidence: &'a [NumericalEvidence],
}

#[derive(Clone, Debug, PartialEq)]
pub struct NumericalEvidence {
    pub logical: LogicalCompilationIdentity,
    pub entry: String,
    pub target: String,
    pub capability_fingerprint: String,
    pub toolchain_fingerprint: String,
    pub assignment: PlanAssignment,
    pub precision: PrecisionResolutionIdentity,
    pub assessment: NumericalAssessment,
}

struct AltExport {
    logical: u32,
    physical: u32,
    selected: VarId,
}
struct ChoiceExport {
    variable: VarId,
    active: VarId,
    alternatives: Vec<AltExport>,
}
struct SiteExport {
    symbol: String,
    variable: VarId,
    upper: i64,
}
struct Export {
    model: Arc<Model>,
    choices: BTreeMap<ChoiceId, ChoiceExport>,
    sites: Vec<SiteExport>,
}

pub fn plan<D: ExecutableDialect>(
    logical: &LogicalProgram,
    family: PlanFamily<D>,
    context: Context<'_, D>,
    budget: Budget,
) -> Result<ResolvedPlan<D>, PlanningError> {
    let export = build(logical, &family, &context)?;
    let solved = solve(export.model.clone(), budget)?;
    let assignment = make_assignment(logical, &export, &solved.values)?;
    let (precision, assessment) = precision_identity(logical, &assignment, &context)?;
    family
        .resolve(
            &assignment,
            context.target,
            precision,
            assessment,
            solved.optimal,
        )
        .map_err(|reason| {
            PlanningError::Invalid(format!(
                "solver selected an assignment rejected by resolution: {reason}"
            ))
        })
}

fn build<D: ExecutableDialect>(
    logical: &LogicalProgram,
    family: &PlanFamily<D>,
    context: &Context<'_, D>,
) -> Result<Export, PlanningError> {
    if family.logical_identity() != &logical.identity {
        return Err(PlanningError::Invalid(
            "family belongs to another logical compilation".into(),
        ));
    }
    if logical.target != context.target.target
        || logical.capability_fingerprint != context.target.capability_fingerprint
    {
        return Err(PlanningError::Invalid(
            "target profile differs from logical specialization".into(),
        ));
    }
    if logical.shapes.values().any(|v| *v < 0) {
        return Err(PlanningError::Invalid(
            "logical shape binding is negative".into(),
        ));
    }
    let facts = family.alternative_facts().map_err(PlanningError::Invalid)?;
    let mut builder = ModelBuilder::new();
    builder.units("estimated executable cost");
    let mut choices = BTreeMap::new();
    for choice in family.choices() {
        if choices.contains_key(&choice.logical_choice) {
            return Err(PlanningError::Invalid(format!(
                "family repeats choice#{}",
                choice.logical_choice.0
            )));
        }
        let mut domain = (0..choice.alternatives.len() as i64).collect::<Vec<_>>();
        if choice.logical_choice != family.entry() {
            domain.push(INACTIVE);
        }
        let variable = builder.variable(
            format!("choice {}", choice.logical_choice.0),
            Domain::set(domain.clone()),
        );
        let active = builder.variable(
            format!("choice {} active", choice.logical_choice.0),
            boolean(),
        );
        builder.constraint(Constraint::Table {
            variables: vec![variable, active],
            tuples: domain
                .iter()
                .map(|v| vec![*v, i64::from(*v != INACTIVE)])
                .collect(),
        });
        let alternatives = choice
            .alternatives
            .iter()
            .enumerate()
            .map(|(ordinal, alt)| {
                let selected = builder.variable(
                    format!("choice {} alternative {ordinal}", choice.logical_choice.0),
                    boolean(),
                );
                builder.constraint(Constraint::Table {
                    variables: vec![variable, selected],
                    tuples: domain
                        .iter()
                        .map(|v| vec![*v, i64::from(*v == ordinal as i64)])
                        .collect(),
                });
                AltExport {
                    logical: alt.logical_alternative(),
                    physical: alt.physical_alternative(),
                    selected,
                }
            })
            .collect();
        choices.insert(
            choice.logical_choice,
            ChoiceExport {
                variable,
                active,
                alternatives,
            },
        );
    }
    if !choices.contains_key(&family.entry()) {
        return Err(PlanningError::Invalid("entry choice is absent".into()));
    }
    for choice in family.choices() {
        if choice.logical_choice == family.entry() {
            if !choice.active_when.is_empty() {
                return Err(PlanningError::Invalid(
                    "entry choice has activation parents".into(),
                ));
            }
            continue;
        }
        if choice.active_when.is_empty() {
            return Err(PlanningError::Invalid(format!(
                "choice#{} has no activation path",
                choice.logical_choice.0
            )));
        }
        let mut terms = vec![LinearTerm::new(choices[&choice.logical_choice].active, 1)];
        for activation in &choice.active_when {
            terms.push(LinearTerm::new(
                logical_selected(
                    &mut builder,
                    &choices,
                    activation.parent,
                    activation.alternative,
                )?,
                -1,
            ));
        }
        equal_zero(&mut builder, terms);
    }
    let sites = add_structural(&mut builder, logical, &choices)?;
    let numeric = sites
        .iter()
        .map(|s| (s.symbol.clone(), s.variable))
        .collect::<BTreeMap<_, _>>();
    let bounds = sites
        .iter()
        .map(|s| (s.symbol.clone(), (1, s.upper)))
        .collect::<BTreeMap<_, _>>();
    let mut indexed = BTreeMap::new();
    for fact in facts {
        let key = (
            fact.choice,
            fact.logical_alternative,
            fact.physical_alternative,
        );
        if indexed.insert(key, fact).is_some() {
            return Err(PlanningError::Invalid(
                "family repeats facts for one alternative".into(),
            ));
        }
    }
    for choice in family.choices() {
        for (ordinal, alt) in choice.alternatives.iter().enumerate() {
            let key = (
                choice.logical_choice,
                alt.logical_alternative(),
                alt.physical_alternative(),
            );
            let fact = indexed
                .remove(&key)
                .ok_or_else(|| PlanningError::Invalid("alternative has no solver facts".into()))?;
            add_facts(
                &mut builder,
                choices[&choice.logical_choice].alternatives[ordinal].selected,
                alt.cost(),
                &fact,
                &logical.shapes,
                &numeric,
                &bounds,
                context.target,
            )?;
        }
    }
    if !indexed.is_empty() {
        return Err(PlanningError::Invalid(
            "facts name an absent alternative".into(),
        ));
    }
    add_numerical(&mut builder, logical, family, context, &choices, &sites)?;
    Ok(Export {
        model: Arc::new(
            builder
                .build()
                .map_err(|e| PlanningError::Solver(e.to_string()))?,
        ),
        choices,
        sites,
    })
}

fn add_structural(
    builder: &mut ModelBuilder,
    logical: &LogicalProgram,
    choices: &BTreeMap<ChoiceId, ChoiceExport>,
) -> Result<Vec<SiteExport>, PlanningError> {
    let mut sites = BTreeMap::<StructuralSiteRef, (i64, VarId)>::new();
    for extent in &logical.extents {
        let LogicalExtent::Structural {
            choice,
            alternative,
            site,
            upper_bound,
        } = extent
        else {
            continue;
        };
        if *upper_bound < 1 {
            return Err(PlanningError::Invalid(format!(
                "site#{site} has nonpositive bound"
            )));
        }
        let key = StructuralSiteRef {
            choice: *choice,
            alternative: *alternative,
            site: *site,
        };
        if sites.contains_key(&key) {
            return Err(PlanningError::Invalid(format!("site#{site} is repeated")));
        }
        let active = logical_selected(builder, choices, *choice, *alternative)?;
        let variable = builder.variable(
            format!("structural site {site}"),
            Domain::interval(1, *upper_bound).map_err(|e| PlanningError::Solver(e.to_string()))?,
        );
        builder.constraint(Constraint::InactiveValue {
            active: Literal::new(active, 1),
            variable,
            inactive: 1,
        });
        sites.insert(key, (*upper_bound, variable));
    }
    for c in &logical.structural_constraints {
        let (upper, variable) = sites
            .get(&c.site)
            .copied()
            .ok_or_else(|| PlanningError::Invalid("constraint names absent site".into()))?;
        let active = logical_selected(
            builder,
            choices,
            c.active_if.choice,
            c.active_if.alternative,
        )?;
        let guard = vec![Literal::new(active, 1)];
        match c.kind {
            StructuralConstraintKind::Multiple(unit) => {
                if unit <= 0 {
                    return Err(PlanningError::Invalid(
                        "structural multiple is nonpositive".into(),
                    ));
                }
                builder.guarded_constraint(
                    guard,
                    Constraint::InDomain {
                        variable,
                        domain: Domain::progression(unit, upper, unit as u64)
                            .map_err(|e| PlanningError::Solver(e.to_string()))?,
                    },
                );
            }
            StructuralConstraintKind::AtLeast(v) => {
                builder.guarded_constraint(
                    guard,
                    Constraint::LinearLe {
                        terms: vec![LinearTerm::new(variable, -1)],
                        rhs: -i128::from(v),
                    },
                );
            }
            StructuralConstraintKind::AtMost(v) => {
                builder.guarded_constraint(
                    guard,
                    Constraint::LinearLe {
                        terms: vec![LinearTerm::new(variable, 1)],
                        rhs: i128::from(v),
                    },
                );
            }
            StructuralConstraintKind::Equal(v) => {
                builder.guarded_constraint(
                    guard,
                    Constraint::InDomain {
                        variable,
                        domain: Domain::singleton(v),
                    },
                );
            }
            StructuralConstraintKind::Divides(v) => {
                if v < 0 {
                    return Err(PlanningError::Invalid(
                        "structural dividend is negative".into(),
                    ));
                }
                let numerator = builder.variable("structural dividend", Domain::singleton(v));
                let quotient = builder.variable(
                    "structural quotient",
                    Domain::interval(0, v.max(1)).unwrap(),
                );
                let remainder = builder.variable("structural remainder", Domain::singleton(0));
                builder.guarded_constraint(
                    guard,
                    Constraint::Arithmetic(Arithmetic::DivRem {
                        numerator,
                        denominator: variable,
                        quotient,
                        remainder,
                    }),
                );
            }
        }
    }
    for r in &logical.structural_refinements {
        let (_, divisor) = sites
            .get(&r.refinement)
            .copied()
            .ok_or_else(|| PlanningError::Invalid("refinement names absent divisor".into()))?;
        let (upper, dividend) = sites
            .get(&r.refined)
            .copied()
            .ok_or_else(|| PlanningError::Invalid("refinement names absent dividend".into()))?;
        let left = logical_selected(
            builder,
            choices,
            r.refinement.choice,
            r.refinement.alternative,
        )?;
        let right = logical_selected(builder, choices, r.refined.choice, r.refined.alternative)?;
        let active = builder.variable("refinement active", boolean());
        builder.constraint(Constraint::BoolAnd {
            output: active,
            inputs: vec![left, right],
        });
        let quotient = builder.variable(
            "refinement quotient",
            Domain::interval(0, upper.max(1)).unwrap(),
        );
        let remainder = builder.variable("refinement remainder", Domain::singleton(0));
        builder.guarded_constraint(
            vec![Literal::new(active, 1)],
            Constraint::Arithmetic(Arithmetic::DivRem {
                numerator: dividend,
                denominator: divisor,
                quotient,
                remainder,
            }),
        );
    }
    Ok(sites
        .into_iter()
        .map(|(site, (upper, variable))| SiteExport {
            symbol: format!("@site{}", site.site),
            variable,
            upper,
        })
        .collect())
}

fn add_facts<C: Ord>(
    builder: &mut ModelBuilder,
    selected: VarId,
    cost: &Sym,
    facts: &PlanAlternativeFacts<C>,
    known: &BTreeMap<String, i64>,
    numeric: &BTreeMap<String, VarId>,
    bounds: &BTreeMap<String, (i64, i64)>,
    target: &ExecutableTargetProfile<C>,
) -> Result<(), PlanningError> {
    let (cost, lo, _) = materialize(builder, cost, known, numeric, bounds)?;
    if lo < 0 {
        return Err(PlanningError::Invalid(
            "alternative cost may be negative".into(),
        ));
    }
    builder.guarded_cost(
        vec![Literal::new(selected, 1)],
        Cost::Linear {
            constant: 0,
            terms: vec![LinearTerm::new(cost, 1)],
        },
    );
    sym_range(
        builder,
        selected,
        &facts.aggregate_device_bytes,
        0,
        target.limits.max_device_bytes,
        known,
        numeric,
        bounds,
    )?;
    for allocation in &facts.allocations {
        sym_range(
            builder,
            selected,
            &allocation.bytes,
            0,
            target.limits.max_allocation_bytes,
            known,
            numeric,
            bounds,
        )?;
    }
    for launch in &facts.launches {
        if !launch.capabilities.is_subset(&target.capabilities)
            || launch.bindings > target.limits.max_bindings_per_launch
        {
            builder.constraint(Constraint::InDomain {
                variable: selected,
                domain: Domain::singleton(0),
            });
        }
        for axis in 0..3 {
            sym_range(
                builder,
                selected,
                &launch.workgroups[axis],
                1,
                target.limits.max_workgroups[axis],
                known,
                numeric,
                bounds,
            )?;
        }
        sym_range(
            builder,
            selected,
            &launch.participants_per_workgroup,
            1,
            target.limits.max_participants_per_workgroup,
            known,
            numeric,
            bounds,
        )?;
        sym_range(
            builder,
            selected,
            &launch.workgroup_bytes,
            0,
            target.limits.max_workgroup_bytes,
            known,
            numeric,
            bounds,
        )?;
        sym_range(
            builder,
            selected,
            &launch.private_bytes_per_participant,
            0,
            target.limits.max_private_bytes_per_participant,
            known,
            numeric,
            bounds,
        )?;
        sym_range(
            builder,
            selected,
            &launch.registers,
            0,
            target.limits.max_registers_per_kernel,
            known,
            numeric,
            bounds,
        )?;
    }
    Ok(())
}

fn add_numerical<D: ExecutableDialect>(
    builder: &mut ModelBuilder,
    logical: &LogicalProgram,
    family: &PlanFamily<D>,
    context: &Context<'_, D>,
    choices: &BTreeMap<ChoiceId, ChoiceExport>,
    sites: &[SiteExport],
) -> Result<(), PlanningError> {
    if matches!(context.precision, PrecisionPolicy::Unconstrained) {
        return Ok(());
    }
    let expected = logical
        .shapes
        .keys()
        .cloned()
        .chain(sites.iter().map(|s| s.symbol.clone()))
        .collect::<BTreeSet<_>>();
    let mut witnesses = Vec::new();
    for evidence in context.numerical_evidence {
        if !evidence_identity(logical, context, evidence)
            || !evidence.assessment.satisfies(context.precision)
        {
            continue;
        }
        if evidence
            .assignment
            .symbols()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>()
            != expected
        {
            continue;
        }
        if logical
            .shapes
            .iter()
            .any(|(k, v)| evidence.assignment.symbols().get(k) != Some(v))
        {
            continue;
        }
        let mut matches = Vec::new();
        let mut valid = !evidence
            .assignment
            .selections()
            .keys()
            .any(|c| !choices.contains_key(c));
        for (id, choice) in choices {
            let wanted = match evidence.assignment.selections().get(id) {
                None => INACTIVE,
                Some(s) => match choice.alternatives.iter().position(|a| {
                    a.logical == s.logical_alternative && a.physical == s.physical_alternative
                }) {
                    Some(v) => v as i64,
                    None => {
                        valid = false;
                        INACTIVE
                    }
                },
            };
            matches.push(eq_indicator(builder, choice.variable, wanted));
        }
        for site in sites {
            if let Some(wanted) = evidence.assignment.symbols().get(&site.symbol) {
                if !(1..=site.upper).contains(wanted) {
                    valid = false;
                } else {
                    matches.push(eq_indicator(builder, site.variable, *wanted));
                }
            } else {
                valid = false;
            }
        }
        if valid {
            let witness = builder.variable("complete numerical witness", boolean());
            builder.constraint(Constraint::BoolAnd {
                output: witness,
                inputs: matches,
            });
            witnesses.push(witness);
        }
    }
    let facts = family.alternative_facts().map_err(PlanningError::Invalid)?;
    for fact in facts {
        if !fact.launches.iter().any(|l| !l.numerical.is_empty()) {
            continue;
        }
        let selected = choices[&fact.choice]
            .alternatives
            .iter()
            .find(|a| {
                a.logical == fact.logical_alternative && a.physical == fact.physical_alternative
            })
            .ok_or_else(|| {
                PlanningError::Invalid("numerical facts name absent alternative".into())
            })?
            .selected;
        let mut terms = vec![LinearTerm::new(selected, 1)];
        terms.extend(witnesses.iter().map(|v| LinearTerm::new(*v, -1)));
        builder.constraint(Constraint::LinearLe { terms, rhs: 0 });
    }
    Ok(())
}

fn evidence_identity<D: ExecutableDialect>(
    logical: &LogicalProgram,
    context: &Context<'_, D>,
    e: &NumericalEvidence,
) -> bool {
    e.logical == logical.identity
        && e.entry == logical.entry
        && e.target == context.target.target
        && e.capability_fingerprint == context.target.capability_fingerprint
        && e.toolchain_fingerprint == context.target.toolchain_fingerprint
        && !e.precision.method_revision.is_empty()
        && !e.precision.evidence_domain.is_empty()
}

fn precision_identity<D: ExecutableDialect>(
    logical: &LogicalProgram,
    assignment: &PlanAssignment,
    context: &Context<'_, D>,
) -> Result<(PrecisionResolutionIdentity, Option<NumericalAssessment>), PlanningError> {
    let mut found = context.numerical_evidence.iter().filter(|e| {
        evidence_identity(logical, context, e)
            && &e.assignment == assignment
            && e.assessment.satisfies(context.precision)
    });
    let first = found.next();
    if found.next().is_some() {
        return Err(PlanningError::Invalid(
            "duplicate evidence for selected assignment".into(),
        ));
    }
    Ok(match first {
        Some(evidence) => (
            evidence.precision.clone(),
            Some(evidence.assessment.clone()),
        ),
        None => (
            PrecisionResolutionIdentity {
                method_revision: if matches!(context.precision, PrecisionPolicy::Unconstrained) {
                    "unconstrained"
                } else {
                    "reference-exact"
                }
                .into(),
                evidence_domain: "compiler".into(),
            },
            None,
        ),
    })
}

fn make_assignment(
    logical: &LogicalProgram,
    export: &Export,
    values: &[i64],
) -> Result<PlanAssignment, PlanningError> {
    let mut result = PlanAssignment::new();
    for (id, choice) in &export.choices {
        let ordinal = get(values, choice.variable)?;
        if ordinal != INACTIVE {
            let alt = choice
                .alternatives
                .get(ordinal as usize)
                .ok_or_else(|| PlanningError::Solver("invalid alternative ordinal".into()))?;
            result
                .select(
                    *id,
                    PlanSelection {
                        logical_alternative: alt.logical,
                        physical_alternative: alt.physical,
                    },
                )
                .map_err(PlanningError::Invalid)?;
        }
    }
    for (name, value) in &logical.shapes {
        result
            .bind_symbol(name.clone(), *value)
            .map_err(PlanningError::Invalid)?;
    }
    for site in &export.sites {
        result
            .bind_symbol(site.symbol.clone(), get(values, site.variable)?)
            .map_err(PlanningError::Invalid)?;
    }
    Ok(result)
}

fn logical_selected(
    builder: &mut ModelBuilder,
    choices: &BTreeMap<ChoiceId, ChoiceExport>,
    id: ChoiceId,
    logical: u32,
) -> Result<VarId, PlanningError> {
    let choice = choices
        .get(&id)
        .ok_or_else(|| PlanningError::Invalid("activation names absent choice".into()))?;
    let inputs = choice
        .alternatives
        .iter()
        .filter(|a| a.logical == logical)
        .map(|a| a.selected)
        .collect::<Vec<_>>();
    if inputs.is_empty() {
        return Err(PlanningError::Invalid(
            "activation names absent logical alternative".into(),
        ));
    }
    if inputs.len() == 1 {
        return Ok(inputs[0]);
    }
    let output = builder.variable("logical alternative selected", boolean());
    for input in &inputs {
        builder.constraint(Constraint::LinearLe {
            terms: vec![LinearTerm::new(*input, 1), LinearTerm::new(output, -1)],
            rhs: 0,
        });
    }
    let mut terms = vec![LinearTerm::new(output, 1)];
    terms.extend(inputs.iter().map(|v| LinearTerm::new(*v, -1)));
    builder.constraint(Constraint::LinearLe { terms, rhs: 0 });
    Ok(output)
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

fn sym_range(
    builder: &mut ModelBuilder,
    active: VarId,
    expression: &Sym,
    low: u64,
    high: u64,
    known: &BTreeMap<String, i64>,
    numeric: &BTreeMap<String, VarId>,
    bounds: &BTreeMap<String, (i64, i64)>,
) -> Result<(), PlanningError> {
    let (variable, _, _) = materialize(builder, expression, known, numeric, bounds)?;
    let low = i64::try_from(low)
        .map_err(|_| PlanningError::Invalid("resource lower bound exceeds i64".into()))?;
    let high = i64::try_from(high)
        .map_err(|_| PlanningError::Invalid("resource limit exceeds i64".into()))?;
    builder.guarded_constraint(
        vec![Literal::new(active, 1)],
        Constraint::InDomain {
            variable,
            domain: Domain::interval(low, high)
                .map_err(|e| PlanningError::Solver(e.to_string()))?,
        },
    );
    Ok(())
}

fn materialize(
    builder: &mut ModelBuilder,
    expression: &Sym,
    known: &BTreeMap<String, i64>,
    numeric: &BTreeMap<String, VarId>,
    bounds: &BTreeMap<String, (i64, i64)>,
) -> Result<(VarId, i64, i64), PlanningError> {
    fn atom(
        builder: &mut ModelBuilder,
        a: &Atom,
        known: &BTreeMap<String, i64>,
        numeric: &BTreeMap<String, VarId>,
        bounds: &BTreeMap<String, (i64, i64)>,
    ) -> Result<(VarId, i64, i64), PlanningError> {
        match a {
            Atom::Param(name) => {
                if let Some(v) = known.get(name) {
                    return Ok((
                        builder.variable(format!("symbol {name}"), Domain::singleton(*v)),
                        *v,
                        *v,
                    ));
                }
                let variable = *numeric.get(name).ok_or_else(|| {
                    PlanningError::Invalid(format!("unresolved executable symbol `{name}`"))
                })?;
                let (lo, hi) = bounds[name];
                Ok((variable, lo, hi))
            }
            Atom::Quot(n, d) | Atom::Rem(n, d) => {
                let (n, nl, nh) = materialize(builder, n, known, numeric, bounds)?;
                let (d, dl, dh) = materialize(builder, d, known, numeric, bounds)?;
                if nl < 0 || dl <= 0 {
                    return Err(PlanningError::Invalid(
                        "invalid symbolic division domain".into(),
                    ));
                }
                let qh = nh / dl;
                let rh = nh.min(dh.saturating_sub(1));
                let q = builder.variable("symbolic quotient", Domain::interval(0, qh).unwrap());
                let r = builder.variable("symbolic remainder", Domain::interval(0, rh).unwrap());
                builder.constraint(Constraint::Arithmetic(Arithmetic::DivRem {
                    numerator: n,
                    denominator: d,
                    quotient: q,
                    remainder: r,
                }));
                if matches!(a, Atom::Quot(..)) {
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
                let (right, rl, rh) = atom(builder, factor, known, numeric, bounds)?;
                let pl = checked(i128::from(lo) * i128::from(rl))?;
                let ph = checked(i128::from(hi) * i128::from(rh))?;
                let product = builder.variable(
                    "symbolic product",
                    Domain::interval(pl, ph).map_err(|e| PlanningError::Solver(e.to_string()))?,
                );
                builder.constraint(Constraint::Arithmetic(Arithmetic::Product {
                    left: variable,
                    right,
                    product,
                }));
                variable = product;
                lo = pl;
                hi = ph;
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
    let low = checked(low)?;
    let high = checked(high)?;
    let result = builder.variable(
        "symbolic expression",
        Domain::interval(low, high).map_err(|e| PlanningError::Solver(e.to_string()))?,
    );
    let mut equality = terms
        .into_iter()
        .map(|(v, c)| LinearTerm::new(v, c))
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
            .map(|t| LinearTerm::new(t.variable, -t.coefficient))
            .collect(),
        rhs: 0,
    });
}
fn checked(v: i128) -> Result<i64, PlanningError> {
    i64::try_from(v).map_err(|_| PlanningError::Invalid("symbolic value exceeds i64".into()))
}
fn boolean() -> Domain {
    Domain::interval(0, 1).expect("boolean domain")
}
fn get(values: &[i64], variable: VarId) -> Result<i64, PlanningError> {
    values
        .get(variable.0)
        .copied()
        .ok_or_else(|| PlanningError::Solver("solution omits variable".into()))
}

struct Solved {
    values: Vec<i64>,
    optimal: bool,
}

fn solve(model: Arc<Model>, budget: Budget) -> Result<Solved, PlanningError> {
    let algorithm = match budget.strategy {
        Strategy::Exact => Algorithm::Exact,
        Strategy::Greedy => Algorithm::Neighborhood(NeighborhoodOptions::default()),
    };
    let mut search = Search::new(
        model,
        Options {
            algorithm,
            ..Options::default()
        },
    )
    .map_err(|e| PlanningError::Solver(e.to_string()))?;
    match search
        .advance(Limits {
            work: budget.work.max(1),
            time: budget.time,
            memory_bytes: None,
        })
        .map_err(|e| PlanningError::Solver(e.to_string()))?
    {
        Outcome::Optimal(v) => Ok(Solved {
            values: v.feasible().values().to_vec(),
            optimal: true,
        }),
        Outcome::Infeasible => Err(PlanningError::Infeasible(
            "no assignment satisfies the executable family".into(),
        )),
        Outcome::Incomplete(v) => v
            .incumbent
            .map(|s| Solved {
                values: s.values().to_vec(),
                optimal: false,
            })
            .ok_or_else(|| {
                PlanningError::Incomplete("search produced no feasible assignment".into())
            }),
    }
}
