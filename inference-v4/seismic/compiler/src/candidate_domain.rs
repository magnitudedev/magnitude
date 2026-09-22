//! The sealed structural search space produced before native realization.
//!
//! A domain owns refined [`CandidateFamily`] values and the one authoritative
//! expression arena in which their choices and constraints were constructed.
//! It contains no native compiler, artifact registry, reflected description,
//! executor, or analytical profile.

use crate::errors::PreparationError;
use crate::expression::PlanningExpr;
use crate::numerics::{EvidenceCatalog, StructuralNumericalObligation};
use crate::preparation_budget::PreparationBudget;
use crate::refinement::{
    CandidateFamily, CandidateFamilyIdentity, ChoiceDeclaration, ConstructionAuthority,
    RefinementCompletion, RefinementLimits, RefinementRequest, RefinementSession,
};
use crate::target::{CompilerRegistry, TargetConstants};
use seismic_lang::entry::{CallSchema, LogicalEntry, SemanticEventManifest};
use seismic_lang::expr::{
    compiled::{CompiledPredicate, InvocationValues},
    AnyExpr, BoolExpr, DecisionId, EntryPredicate, ExprArena, PartialAssignment, SymbolValue,
};
use seismic_lang::ids::{ModuleHash, StableEntryId};
use seismic_lang::precision::PrecisionPolicy;
use seismic_target::{DeviceDescription, DeviceDescriptionIdentity, NumericalEnvironmentIdentity};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

static NEXT_DOMAIN_TOKEN: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetDomain {
    predicate: EntryPredicate,
    identity: [u8; 32],
}

impl TargetDomain {
    pub(crate) fn new(predicate: EntryPredicate, identity: [u8; 32]) -> Self {
        Self {
            predicate,
            identity,
        }
    }
    pub fn predicate(&self) -> EntryPredicate {
        self.predicate
    }
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

#[derive(Debug)]
pub struct NonEmpty<T> {
    items: Vec<T>,
}

impl<T> NonEmpty<T> {
    pub(crate) fn new(items: Vec<T>) -> Option<Self> {
        (!items.is_empty()).then_some(Self { items })
    }
    pub fn first(&self) -> &T {
        &self.items[0]
    }
    pub fn as_slice(&self) -> &[T] {
        &self.items
    }
    pub fn len(&self) -> usize {
        self.items.len()
    }
    pub fn is_empty(&self) -> bool {
        false
    }
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.items.iter()
    }
    pub(crate) fn into_vec(self) -> Vec<T> {
        self.items
    }
}

/// Unchecked search input. Only a [`CandidateDomain`] can construct one, so a
/// proposal is always associated with the domain API that will check it.
#[derive(Clone, Debug)]
pub struct CandidateProposal {
    domain: u64,
    family: CandidateFamilyIdentity,
    choices: Vec<(DecisionId, i64)>,
}

/// Canonical structural candidate name. Choices are in declaration order and
/// contain exactly the active choices; there is no public constructor.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CandidateCoordinate {
    domain: u64,
    family: CandidateFamilyIdentity,
    choices: Vec<(DecisionId, i64)>,
}

impl CandidateCoordinate {
    pub fn family(&self) -> &CandidateFamilyIdentity {
        &self.family
    }
    pub fn choices(&self) -> &[(DecisionId, i64)] {
        &self.choices
    }
}

#[derive(Clone, Debug)]
pub struct CandidatePoint {
    coordinate: CandidateCoordinate,
    invocation: InvocationValues,
}

impl CandidatePoint {
    pub fn new(coordinate: CandidateCoordinate, invocation: InvocationValues) -> Self {
        Self {
            coordinate,
            invocation,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConstraintOrigin {
    Invocation,
    SemanticApplicability,
    StructuralLegality,
    NumericalAdmissibility,
}

#[derive(Clone, Debug)]
pub struct DomainConstraint {
    origin: ConstraintOrigin,
    predicate: BoolExpr,
}

impl DomainConstraint {
    pub fn origin(&self) -> ConstraintOrigin {
        self.origin
    }
    pub fn predicate(&self) -> BoolExpr {
        self.predicate
    }
}

#[derive(Clone, Debug)]
pub struct ConstraintSet {
    conjuncts: Vec<DomainConstraint>,
    combined: BoolExpr,
}

impl ConstraintSet {
    pub fn conjuncts(&self) -> &[DomainConstraint] {
        &self.conjuncts
    }
    pub fn predicate(&self) -> BoolExpr {
        self.combined
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoordinateError {
    ForeignDomain,
    UnknownFamily,
    MissingChoice(DecisionId),
    DuplicateChoice(DecisionId),
    ForeignChoice(DecisionId),
    ValueOutsideAxis { decision: DecisionId, value: i64 },
    InactiveChoice(DecisionId),
    IndeterminateActivation(DecisionId),
    EmptyRegion,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MembershipError {
    Coordinate(CoordinateError),
    Invocation(seismic_lang::expr::EvalError),
}

#[derive(Debug)]
pub struct InvocationRegion {
    predicate: CompiledPredicate,
}

impl InvocationRegion {
    pub fn contains(
        &self,
        invocation: &InvocationValues,
    ) -> Result<bool, seismic_lang::expr::EvalError> {
        self.predicate.evaluate(invocation)
    }
}

#[derive(Debug)]
pub struct CandidateSlice<'a, B: seismic_target::TargetFamily> {
    coordinate: CandidateCoordinate,
    family: &'a Arc<CandidateFamily<B>>,
    constraints: &'a ConstraintSet,
    applicability: InvocationRegion,
    numerical: StructuralNumericalObligation,
}

pub(crate) struct CheckedPreparationCandidate<B: seismic_target::TargetFamily> {
    pub(crate) coordinate: CandidateCoordinate,
    pub(crate) family: Arc<CandidateFamily<B>>,
    pub(crate) constraint: BoolExpr,
    pub(crate) is_universal: bool,
}

impl<'a, B: seismic_target::TargetFamily> CandidateSlice<'a, B> {
    pub fn coordinate(&self) -> &CandidateCoordinate {
        &self.coordinate
    }
    pub fn applicability(&self) -> &InvocationRegion {
        &self.applicability
    }
    pub fn numerical_requirement(&self) -> StructuralNumericalObligation {
        self.numerical
    }
    pub fn executable(&self) -> crate::evaluation::TargetClosedExecutableView<'a, B> {
        crate::evaluation::TargetClosedExecutableView::new(
            self.family.as_ref(),
            self.constraints.conjuncts(),
        )
    }
}

#[derive(Debug)]
pub struct CandidateDomain<B: seismic_target::TargetFamily> {
    domain_token: u64,
    entry: StableEntryId,
    module: ModuleHash,
    schema: Arc<CallSchema>,
    semantic_events: Arc<SemanticEventManifest>,
    target_domain: TargetDomain,
    constants: TargetConstants,
    device: DeviceDescriptionIdentity,
    target: NumericalEnvironmentIdentity,
    evidence: Arc<EvidenceCatalog>,
    arena: ExprArena,
    universal: DomainCandidate<B>,
    optimized: Vec<DomainCandidate<B>>,
    precision: PrecisionPolicy,
    optimization_exhausted: bool,
}

impl<B: seismic_target::TargetFamily> CandidateDomain<B> {
    pub fn entry(&self) -> StableEntryId {
        self.entry
    }
    pub fn module(&self) -> ModuleHash {
        self.module
    }
    pub fn schema(&self) -> &Arc<CallSchema> {
        &self.schema
    }
    pub fn semantic_event_manifest(&self) -> &Arc<SemanticEventManifest> {
        &self.semantic_events
    }
    pub fn target_domain(&self) -> TargetDomain {
        self.target_domain
    }
    pub fn constants(&self) -> &TargetConstants {
        &self.constants
    }
    pub fn arena(&self) -> &ExprArena {
        &self.arena
    }
    pub fn precision(&self) -> &PrecisionPolicy {
        &self.precision
    }
    pub fn device_identity(&self) -> &DeviceDescriptionIdentity {
        &self.device
    }

    pub fn families(&self) -> impl Iterator<Item = CandidateFamilyView<'_, B>> {
        std::iter::once(CandidateFamilyView {
            candidate: &self.universal,
        })
        .chain(
            self.optimized
                .iter()
                .map(|candidate| CandidateFamilyView { candidate }),
        )
    }

    pub fn proposal(
        &self,
        family: CandidateFamilyIdentity,
        choices: Vec<(DecisionId, i64)>,
    ) -> CandidateProposal {
        CandidateProposal {
            domain: self.domain_token,
            family,
            choices,
        }
    }

    pub fn universal_proposal(&self) -> CandidateProposal {
        CandidateProposal {
            domain: self.domain_token,
            family: self.universal.family.identity().clone(),
            choices: Vec::new(),
        }
    }

    /// Validates a raw proposal and removes inactive choices and input-order
    /// differences, producing the one canonical coordinate.
    pub fn canonicalize(
        &self,
        proposal: CandidateProposal,
    ) -> Result<CandidateCoordinate, CoordinateError> {
        if proposal.domain != self.domain_token {
            return Err(CoordinateError::ForeignDomain);
        }
        let candidate = self.family(&proposal.family)?;
        let (_, choices) =
            canonical_choice_binding(&self.arena, candidate.family.choices(), &proposal.choices)?;
        Ok(CandidateCoordinate {
            domain: self.domain_token,
            family: proposal.family,
            choices,
        })
    }

    /// Checks the coordinate against the authoritative family and constraint
    /// relation. Since coordinates have no public constructor, failure here
    /// indicates stale data from a different domain snapshot.
    pub fn check(
        &self,
        coordinate: &CandidateCoordinate,
    ) -> Result<CandidateSlice<'_, B>, CoordinateError> {
        if coordinate.domain != self.domain_token {
            return Err(CoordinateError::ForeignDomain);
        }
        let candidate = self.family(&coordinate.family)?;
        let (mut fixed, canonical) =
            canonical_choice_binding(&self.arena, candidate.family.choices(), &coordinate.choices)?;
        if canonical != coordinate.choices {
            if let Some((decision, _)) = coordinate
                .choices
                .iter()
                .find(|choice| !canonical.contains(choice))
            {
                return Err(CoordinateError::InactiveChoice(*decision));
            }
            return Err(CoordinateError::MissingChoice(
                candidate
                    .family
                    .choices()
                    .iter()
                    .find(|choice| !canonical.iter().any(|(id, _)| *id == choice.decision()))
                    .map(ChoiceDeclaration::decision)
                    .unwrap_or_else(|| candidate.family.choices()[0].decision()),
            ));
        }
        for (symbol, value) in self.constants.bindings() {
            fixed.bind(*symbol, *value);
        }
        let applicability = self
            .arena
            .compile_bool_with(candidate.constraints.predicate(), &fixed);
        if applicability.reads().is_empty()
            && !applicability
                .evaluate(&InvocationValues::new())
                .unwrap_or(false)
        {
            return Err(CoordinateError::EmptyRegion);
        }
        Ok(CandidateSlice {
            coordinate: coordinate.clone(),
            family: &candidate.family,
            constraints: &candidate.constraints,
            applicability: InvocationRegion {
                predicate: applicability,
            },
            numerical: candidate.numerical,
        })
    }

    pub fn contains(&self, point: &CandidatePoint) -> Result<bool, MembershipError> {
        let slice = self
            .check(&point.coordinate)
            .map_err(MembershipError::Coordinate)?;
        slice
            .applicability
            .contains(&point.invocation)
            .map_err(MembershipError::Invocation)
    }

    pub(crate) fn checked_preparation_candidate(
        &self,
        coordinate: &CandidateCoordinate,
    ) -> Result<CheckedPreparationCandidate<B>, CoordinateError> {
        self.check(coordinate)?;
        let candidate = self.family(coordinate.family())?;
        Ok(CheckedPreparationCandidate {
            coordinate: coordinate.clone(),
            family: candidate.family.clone(),
            constraint: candidate.constraints.predicate(),
            is_universal: candidate.family.identity() == self.universal.family.identity(),
        })
    }

    pub(crate) fn arena_mut(&mut self) -> &mut ExprArena {
        &mut self.arena
    }

    pub(crate) fn numerical_context(
        &self,
    ) -> (
        PrecisionPolicy,
        Arc<EvidenceCatalog>,
        NumericalEnvironmentIdentity,
        TargetDomain,
        TargetConstants,
    ) {
        (
            self.precision.clone(),
            self.evidence.clone(),
            self.target.clone(),
            self.target_domain,
            self.constants.clone(),
        )
    }

    fn family(
        &self,
        identity: &CandidateFamilyIdentity,
    ) -> Result<&DomainCandidate<B>, CoordinateError> {
        if self.universal.family.identity() == identity {
            return Ok(&self.universal);
        }
        self.optimized
            .iter()
            .find(|candidate| candidate.family.identity() == identity)
            .ok_or(CoordinateError::UnknownFamily)
    }

    pub(crate) fn from_parts(parts: CandidateDomainParts<B>) -> Self {
        Self {
            domain_token: parts.domain_token,
            entry: parts.entry,
            module: parts.module,
            schema: parts.schema,
            semantic_events: parts.semantic_events,
            target_domain: parts.target_domain,
            constants: parts.constants,
            device: parts.device,
            target: parts.target,
            evidence: parts.evidence,
            arena: parts.arena,
            universal: parts.universal,
            optimized: parts.optimized,
            precision: parts.precision,
            optimization_exhausted: parts.optimization_exhausted,
        }
    }

    pub(crate) fn into_parts(self) -> CandidateDomainParts<B> {
        CandidateDomainParts {
            domain_token: self.domain_token,
            entry: self.entry,
            module: self.module,
            schema: self.schema,
            semantic_events: self.semantic_events,
            target_domain: self.target_domain,
            constants: self.constants,
            device: self.device,
            target: self.target,
            evidence: self.evidence,
            arena: self.arena,
            universal: self.universal,
            optimized: self.optimized,
            precision: self.precision,
            optimization_exhausted: self.optimization_exhausted,
        }
    }
}

pub struct CandidateFamilyView<'a, B: seismic_target::TargetFamily> {
    candidate: &'a DomainCandidate<B>,
}

impl<'a, B: seismic_target::TargetFamily> CandidateFamilyView<'a, B> {
    pub fn identity(&self) -> &CandidateFamilyIdentity {
        self.candidate.family.identity()
    }
    pub fn choices(&self) -> &'a [ChoiceDeclaration] {
        self.candidate.family.choices()
    }
    pub fn constraints(&self) -> &'a [DomainConstraint] {
        self.candidate.constraints.conjuncts()
    }
    pub fn numerical_requirement(&self) -> StructuralNumericalObligation {
        self.candidate.numerical
    }
    pub fn executable(&self) -> crate::evaluation::TargetClosedExecutableView<'a, B> {
        crate::evaluation::TargetClosedExecutableView::new(
            &self.candidate.family,
            self.candidate.constraints.conjuncts(),
        )
    }
}

pub(crate) struct CandidateDomainParts<B: seismic_target::TargetFamily> {
    pub domain_token: u64,
    pub entry: StableEntryId,
    pub module: ModuleHash,
    pub schema: Arc<CallSchema>,
    pub semantic_events: Arc<SemanticEventManifest>,
    pub target_domain: TargetDomain,
    pub constants: TargetConstants,
    pub device: DeviceDescriptionIdentity,
    pub target: NumericalEnvironmentIdentity,
    pub evidence: Arc<EvidenceCatalog>,
    pub arena: ExprArena,
    pub universal: DomainCandidate<B>,
    pub optimized: Vec<DomainCandidate<B>>,
    pub precision: PrecisionPolicy,
    pub optimization_exhausted: bool,
}

#[derive(Debug)]
pub(crate) struct DomainCandidate<B: seismic_target::TargetFamily> {
    pub(crate) family: Arc<CandidateFamily<B>>,
    pub(crate) constraints: ConstraintSet,
    pub(crate) numerical: StructuralNumericalObligation,
}

pub(crate) fn canonical_choice_binding(
    arena: &ExprArena,
    declarations: &[ChoiceDeclaration],
    choices: &[(DecisionId, i64)],
) -> Result<(PartialAssignment, Vec<(DecisionId, i64)>), CoordinateError> {
    let declared = declarations
        .iter()
        .map(ChoiceDeclaration::decision)
        .collect::<HashSet<_>>();
    let mut supplied = HashMap::new();
    for (decision, value) in choices {
        if !declared.contains(decision) {
            return Err(CoordinateError::ForeignChoice(*decision));
        }
        if supplied.insert(*decision, *value).is_some() {
            return Err(CoordinateError::DuplicateChoice(*decision));
        }
        if !arena.decision_domain(*decision).values().contains(value) {
            return Err(CoordinateError::ValueOutsideAxis {
                decision: *decision,
                value: *value,
            });
        }
    }

    let mut assignment = PartialAssignment::new();
    let mut canonical = Vec::new();
    for declaration in declarations {
        let active = arena.compile_bool_with(declaration.active_when(), &assignment);
        if !active.reads().is_empty() {
            return Err(CoordinateError::IndeterminateActivation(
                declaration.decision(),
            ));
        }
        let active = active
            .evaluate(&InvocationValues::new())
            .map_err(|_| CoordinateError::IndeterminateActivation(declaration.decision()))?;
        if !active {
            continue;
        }
        let decision = declaration.decision();
        let value = supplied
            .get(&decision)
            .copied()
            .ok_or(CoordinateError::MissingChoice(decision))?;
        assignment.bind(arena.decision_symbol(decision), SymbolValue::Int(value));
        canonical.push((decision, value));
    }
    Ok((assignment, canonical))
}

pub(crate) fn canonical_coordinate<B: seismic_target::TargetFamily>(
    domain: u64,
    arena: &ExprArena,
    family: &CandidateFamily<B>,
    choices: &[(DecisionId, i64)],
) -> Result<CandidateCoordinate, CoordinateError> {
    let (_, choices) = canonical_choice_binding(arena, family.choices(), choices)?;
    Ok(CandidateCoordinate {
        domain,
        family: family.identity().clone(),
        choices,
    })
}

pub(crate) fn construct_candidate_domain<T>(
    entry: LogicalEntry,
    device: &DeviceDescription<T>,
    registry: &CompilerRegistry<T>,
    precision: &PrecisionPolicy,
    evidence: &EvidenceCatalog,
    budget: &PreparationBudget,
) -> Result<CandidateDomain<T>, PreparationError>
where
    T: seismic_target::TargetFamily,
{
    internals::candidate_domain(entry, device, registry, precision, evidence, budget)
}

pub(crate) mod internals {
    use super::*;
    use seismic_lang::entry::{ParameterKind, ResultKind};
    use seismic_lang::expr::{CmpOp, NodeView, RootName};

    pub(super) fn candidate_domain<T>(
        entry: LogicalEntry,
        target: &DeviceDescription<T>,
        registry: &CompilerRegistry<T>,
        precision: &PrecisionPolicy,
        evidence: &EvidenceCatalog,
        budget: &PreparationBudget,
    ) -> Result<CandidateDomain<T>, PreparationError>
    where
        T: seismic_target::TargetFamily,
    {
        let semantic_events = Arc::new(entry.semantic_event_manifest());
        let seismic_lang::entry::LogicalEntryParts {
            identity,
            module,
            schema,
            domain,
            mut arena,
            program,
        } = entry.into_parts();
        let constants = crate::target::bind_target_constants(target, &mut arena);
        let target_domain = self::target_domain(&mut arena, &schema, domain, target)?;
        let refined = RefinementSession::new(RefinementLimits::from(budget)).refine(
            arena,
            RefinementRequest {
                program: &program,
                schema: &schema,
                target,
                registry,
                constants: &constants,
                precision,
            },
        )?;
        let (mut arena, universal_family, optimized_families, refinement_report) =
            refined.into_parts();
        // The universal schedule is made total from the declared target
        // contract before it enters the immutable structural domain. Native
        // reflection may later validate this contract, but it must never
        // rewrite a family after a coordinate has been selected.
        let universal_family =
            chunk_structural_universal(&mut arena, universal_family, target.limits().max_grid[0])?;
        let optimization_exhausted =
            !matches!(refinement_report.completion, RefinementCompletion::Complete)
                || !refinement_report.registered_factory_traversal_complete;

        if universal_family.authority != ConstructionAuthority::UniversalPortable
            || universal_family.numerical_role != seismic_lang::entry::NumericalRole::Reference
            || !universal_family.choices().is_empty()
            || !universal_family.numerical_transfer().is_exact()
        {
            return Err(PreparationError::UniversalClosure(
                "structural universal member lacks checked exact reference provenance".into(),
            ));
        }
        validate_structural_universal(
            &mut arena,
            &universal_family,
            target_domain.predicate().node(),
            &constants,
        )?;

        let target_node = target_domain.predicate().node();
        let universal = structural_candidate(&mut arena, universal_family, target_node, precision);
        let optimized = optimized_families
            .into_iter()
            .map(|family| structural_candidate(&mut arena, family, target_node, precision))
            .collect();

        Ok(CandidateDomain::from_parts(CandidateDomainParts {
            domain_token: NEXT_DOMAIN_TOKEN.fetch_add(1, Ordering::Relaxed),
            entry: identity,
            module,
            schema: Arc::new(schema),
            semantic_events,
            target_domain,
            constants,
            device: target.identity().clone(),
            target: target.numerical_environment_identity().clone(),
            evidence: Arc::new(evidence.clone()),
            arena,
            universal,
            optimized,
            precision: precision.clone(),
            optimization_exhausted,
        }))
    }

    fn chunk_structural_universal<T: seismic_target::TargetFamily>(
        arena: &mut ExprArena,
        family: CandidateFamily<T>,
        maximum_grid_x: u64,
    ) -> Result<CandidateFamily<T>, PreparationError> {
        if maximum_grid_x == 0 {
            return Err(PreparationError::UniversalClosure(
                "target contract has zero one-dimensional grid capacity".into(),
            ));
        }
        let mut parts = family.into_parts();
        let chunks = parts
            .executable
            .schedule()
            .launches()
            .iter()
            .enumerate()
            .filter_map(|(ordinal, launch)| {
                launch.parallel_extent.map(|_| {
                    (
                        parts.executable.schedule().launch_id(ordinal as u32),
                        arena.nat(maximum_grid_x),
                    )
                })
            })
            .collect::<Vec<_>>();
        parts.executable = parts.executable.chunk_semantic_launches(arena, chunks);
        Ok(CandidateFamily::from_parts(parts))
    }

    fn validate_structural_universal<T: seismic_target::TargetFamily>(
        arena: &mut ExprArena,
        family: &CandidateFamily<T>,
        target_domain: BoolExpr,
        constants: &TargetConstants,
    ) -> Result<(), PreparationError> {
        let mut fixed = PartialAssignment::new();
        for (symbol, value) in constants.bindings() {
            fixed.bind(*symbol, *value);
        }
        let target_domain = arena.partial(target_domain, &fixed);
        let semantic = arena.partial(family.semantic_coverage().node(), &fixed);
        let structural = arena.partial(family.hard_constraints(), &fixed);
        let coverage = arena.all(&[semantic, structural]);
        let total = arena.implies(target_domain, coverage);
        if !matches!(arena.view(AnyExpr::Bool(total)), NodeView::BoolConst(true)) {
            return Err(PreparationError::UniversalClosure(
                "structural universal member is not total over TargetDomain".into(),
            ));
        }
        Ok(())
    }

    fn structural_candidate<T: seismic_target::TargetFamily>(
        arena: &mut ExprArena,
        family: CandidateFamily<T>,
        target: BoolExpr,
        precision: &PrecisionPolicy,
    ) -> DomainCandidate<T> {
        let semantic = family.semantic_coverage().node();
        let hard = family.hard_constraints();
        let numerical =
            crate::numerics::structural_obligation(arena, family.numerical_transfer(), precision);
        let mut conjuncts = vec![
            DomainConstraint {
                origin: ConstraintOrigin::Invocation,
                predicate: target,
            },
            DomainConstraint {
                origin: ConstraintOrigin::SemanticApplicability,
                predicate: semantic,
            },
            DomainConstraint {
                origin: ConstraintOrigin::StructuralLegality,
                predicate: hard,
            },
        ];
        conjuncts.push(DomainConstraint {
            origin: ConstraintOrigin::NumericalAdmissibility,
            predicate: numerical.search_predicate(arena),
        });
        let combined = arena.all(
            &conjuncts
                .iter()
                .map(DomainConstraint::predicate)
                .collect::<Vec<_>>(),
        );
        DomainCandidate {
            family: Arc::new(family),
            constraints: ConstraintSet {
                conjuncts,
                combined,
            },
            numerical,
        }
    }

    pub(crate) fn planning_projection(
        arena: &mut ExprArena,
        predicate: BoolExpr,
    ) -> PlanningExpr<BoolExpr> {
        let mut terms = Vec::new();
        collect_planning_terms(arena, predicate, &mut terms);
        let node = arena.all(&terms);
        PlanningExpr::new(arena, node)
            .unwrap_or_else(|| panic!("planning projection retained an invocation expression"))
    }

    fn collect_planning_terms(arena: &ExprArena, predicate: BoolExpr, output: &mut Vec<BoolExpr>) {
        match arena.view(AnyExpr::Bool(predicate)) {
            NodeView::Binary {
                op: seismic_lang::expr::BinaryOp::And,
                lhs: AnyExpr::Bool(left),
                rhs: AnyExpr::Bool(right),
            } => {
                collect_planning_terms(arena, left, output);
                collect_planning_terms(arena, right, output);
            }
            NodeView::Nary {
                op: seismic_lang::expr::NaryOp::All,
                operands,
            } => {
                for operand in operands {
                    let AnyExpr::Bool(term) = *operand else {
                        panic!("Boolean conjunction contains a non-Boolean operand")
                    };
                    collect_planning_terms(arena, term, output);
                }
            }
            _ if PlanningExpr::new(arena, predicate).is_some() => output.push(predicate),
            _ => {}
        }
    }

    fn target_domain<B: seismic_target::TargetFamily>(
        arena: &mut ExprArena,
        schema: &CallSchema,
        entry: seismic_lang::entry::EntryDomain,
        target: &DeviceDescription<B>,
    ) -> Result<TargetDomain, PreparationError> {
        let max_index = if target.limits().max_index_bits >= 64 {
            u64::MAX
        } else {
            (1_u64 << target.limits().max_index_bits) - 1
        };
        let max_index = arena.nat(max_index);
        let max_allocation = arena.nat(target.limits().max_allocation_bytes);
        let mut terms = vec![entry.predicate().node()];
        let mut tensor = |representation,
                          axes: &[seismic_lang::expr::NatExpr]|
         -> Result<(), PreparationError> {
            if !target.dtypes().representations.contains(&representation) {
                return Err(PreparationError::TargetDomainUnrepresentable(format!(
                    "target {:?} does not support representation `{}`",
                    B::NAME,
                    seismic_lang::registry::representation_info(representation).name,
                )));
            }
            for axis in axes {
                terms.push(arena.nat_cmp(CmpOp::Le, *axis, max_index));
                terms.push(arena.side_conditions(AnyExpr::Nat(*axis)));
            }
            let bytes = seismic_ir::storage::tensor_bytes(arena, representation, axes);
            terms.push(arena.side_conditions(AnyExpr::Nat(bytes)));
            terms.push(arena.nat_cmp(CmpOp::Le, bytes, max_allocation));
            terms.push(arena.nat_cmp(CmpOp::Le, bytes, max_index));
            Ok(())
        };
        for parameter in schema.parameters() {
            if let ParameterKind::Tensor {
                representation,
                axes,
                ..
            } = &parameter.kind
            {
                tensor(*representation, axes)?;
            }
        }
        for result in schema.results() {
            if let ResultKind::Tensor {
                representation,
                axes,
            } = &result.kind
            {
                tensor(*representation, axes)?;
            }
        }
        let predicate = arena.all(&terms);
        let predicate = seismic_lang::expr::EntryPredicate::new(arena, predicate)
            .unwrap_or_else(|_| panic!("target-domain construction retained a compiler symbol"));
        let root = arena.root(RootName::Guard, AnyExpr::Bool(predicate.node()));
        let identity = arena.canonical_digest(&[root]).bytes();
        Ok(TargetDomain::new(predicate, identity))
    }
}

#[cfg(test)]
mod candidate_domain_tests {
    use super::*;
    use seismic_lang::expr::FiniteDomain;

    fn declarations() -> (
        ExprArena,
        Vec<ChoiceDeclaration>,
        DecisionId,
        DecisionId,
        DecisionId,
    ) {
        let mut arena = ExprArena::new();
        let parent = arena.decision(FiniteDomain::new(vec![0, 1]).unwrap());
        let child = arena.decision(FiniteDomain::new(vec![4, 8]).unwrap());
        let foreign = arena.decision(FiniteDomain::new(vec![7]).unwrap());
        let parent_active = arena.bool(true);
        let child_active = arena.decision_is(parent, 1);
        (
            arena,
            vec![
                ChoiceDeclaration {
                    decision: parent,
                    meaning: "algorithm",
                    active_when: parent_active,
                },
                ChoiceDeclaration {
                    decision: child,
                    meaning: "tile",
                    active_when: child_active,
                },
            ],
            parent,
            child,
            foreign,
        )
    }

    #[test]
    fn canonical_binding_omits_inactive_nested_choices() {
        let (arena, declarations, parent, child, _) = declarations();
        let (_, canonical) =
            canonical_choice_binding(&arena, &declarations, &[(child, 8), (parent, 0)]).unwrap();
        assert_eq!(canonical, vec![(parent, 0)]);
    }

    #[test]
    fn active_nested_choice_is_required_and_canonically_ordered() {
        let (arena, declarations, parent, child, _) = declarations();
        assert!(matches!(
            canonical_choice_binding(&arena, &declarations, &[(parent, 1)]),
            Err(CoordinateError::MissingChoice(id)) if id == child
        ));
        let (_, canonical) =
            canonical_choice_binding(&arena, &declarations, &[(child, 4), (parent, 1)]).unwrap();
        assert_eq!(canonical, vec![(parent, 1), (child, 4)]);
    }

    #[test]
    fn malformed_choice_sets_are_rejected() {
        let (arena, declarations, parent, child, foreign) = declarations();
        assert!(matches!(
            canonical_choice_binding(&arena, &declarations, &[(parent, 0), (parent, 0)]),
            Err(CoordinateError::DuplicateChoice(id)) if id == parent
        ));
        assert!(matches!(
            canonical_choice_binding(&arena, &declarations, &[(foreign, 7)]),
            Err(CoordinateError::ForeignChoice(id)) if id == foreign
        ));
        assert!(matches!(
            canonical_choice_binding(&arena, &declarations, &[(parent, 1), (child, 5)]),
            Err(CoordinateError::ValueOutsideAxis { decision: id, value: 5 }) if id == child
        ));
    }
}
