//! `CandidateDomain<B>`: the sealed symbolic domain of selectable candidates.
//!
//! `candidate_domain` consumes the logical entry, binds the opened machine into
//! the entry's arena, asks every registered factory (and the universal
//! portable factory) for closed implementations of the root function, and
//! seals their finite choice axes and one authoritative constraint relation.
//! Every member already owns reconciled native kernels; planning derives a
//! transient solver adapter later, after evaluation.
//!
//! W6 owns the internals.

use crate::errors::PreparationError;
use crate::expression::PlanningExpr;
use crate::implementation::{OptimizedImplementation, UniversalImplementation};
use crate::numerics::EvidenceCatalog;
use crate::preparation_budget::{PreparationBudget, PreparationBudgetTracker};
use crate::target::{CompilerRegistry, TargetConstants};
use seismic_lang::entry::{CallSchema, LogicalEntry, SemanticEventManifest};
use seismic_lang::expr::{
    compiled::{CompiledPredicate, InvocationValues},
    BoolExpr, DecisionId, EntryPredicate, ExprArena, PartialAssignment, SymbolValue,
};
use seismic_lang::ids::{ModuleHash, StableEntryId};
use seismic_lang::precision::PrecisionPolicy;
use seismic_target::NumericalEnvironmentIdentity;
use seismic_target::{DeviceDescription, DeviceDescriptionIdentity};
use std::sync::Arc;

/// The target domain (§10.1): entry domain intersected with target integer
/// and address representability and backend call-schema representability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetDomain {
    predicate: EntryPredicate,
    identity: [u8; 32],
}

#[cfg(test)]
mod candidate_domain_tests {
    use super::*;
    use seismic_lang::expr::FiniteDomain;

    fn fixture() -> (ExprArena, ChoiceAxis, DecisionId, DecisionId) {
        let mut arena = ExprArena::new();
        let decision = arena.decision(FiniteDomain::new(vec![1, 2]).unwrap());
        let foreign = arena.decision(FiniteDomain::new(vec![7]).unwrap());
        (
            arena,
            ChoiceAxis {
                decision,
                meaning: "tile",
                values: vec![1, 2],
            },
            decision,
            foreign,
        )
    }

    #[test]
    fn coordinate_binding_is_exact_and_checked() {
        let (arena, axis, decision, foreign) = fixture();
        assert!(matches!(
            exact_choice_binding(&arena, std::slice::from_ref(&axis), &[]),
            Err(CoordinateError::MissingChoice(id)) if id == decision
        ));
        assert!(matches!(
            exact_choice_binding(
                &arena,
                std::slice::from_ref(&axis),
                &[(decision, 1), (decision, 1)]
            ),
            Err(CoordinateError::DuplicateChoice(id)) if id == decision
        ));
        assert!(matches!(
            exact_choice_binding(&arena, std::slice::from_ref(&axis), &[(foreign, 7)]),
            Err(CoordinateError::ForeignChoice(id)) if id == foreign
        ));
        assert!(matches!(
            exact_choice_binding(&arena, std::slice::from_ref(&axis), &[(decision, 3)]),
            Err(CoordinateError::ValueOutsideAxis { decision: id, value: 3 }) if id == decision
        ));
    }

    #[test]
    fn authoritative_constraint_decides_membership_after_binding() {
        let (mut arena, axis, decision, _) = fixture();
        let predicate = arena.decision_is(decision, 2);
        for (value, expected) in [(1, false), (2, true)] {
            let fixed =
                exact_choice_binding(&arena, std::slice::from_ref(&axis), &[(decision, value)])
                    .unwrap();
            let compiled = arena.compile_bool_with(predicate, &fixed);
            assert!(compiled.reads().is_empty());
            assert_eq!(
                compiled.evaluate(&InvocationValues::new()).unwrap(),
                expected
            );
        }
    }
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

/// Non-empty vector with a private constructor.
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CandidateCoordinate {
    family: crate::implementation::ImplementationIdentity,
    choices: Vec<(DecisionId, i64)>,
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

impl CandidateCoordinate {
    pub fn new(
        family: crate::implementation::ImplementationIdentity,
        choices: Vec<(DecisionId, i64)>,
    ) -> Self {
        Self { family, choices }
    }
    pub fn family(&self) -> &crate::implementation::ImplementationIdentity {
        &self.family
    }
    pub fn choices(&self) -> &[(DecisionId, i64)] {
        &self.choices
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConstraintOrigin {
    Invocation,
    SemanticApplicability,
    NativeLegality,
    NumericalAdmissibility,
}

#[derive(Clone, Debug)]
pub struct DomainConstraint {
    origin: ConstraintOrigin,
    predicate: BoolExpr,
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

impl DomainConstraint {
    pub fn origin(&self) -> ConstraintOrigin {
        self.origin
    }
    pub fn predicate(&self) -> BoolExpr {
        self.predicate
    }
}

#[derive(Clone, Debug)]
pub struct ChoiceAxis {
    decision: DecisionId,
    meaning: &'static str,
    values: Vec<i64>,
}

impl ChoiceAxis {
    pub fn decision(&self) -> DecisionId {
        self.decision
    }
    pub fn meaning(&self) -> &'static str {
        self.meaning
    }
    pub fn values(&self) -> &[i64] {
        &self.values
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoordinateError {
    UnknownFamily,
    MissingChoice(DecisionId),
    DuplicateChoice(DecisionId),
    ForeignChoice(DecisionId),
    ValueOutsideAxis { decision: DecisionId, value: i64 },
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
    implementation: &'a crate::implementation::Implementation<B>,
    applicability: InvocationRegion,
}

impl<'a, B: seismic_target::TargetFamily> CandidateSlice<'a, B> {
    pub fn coordinate(&self) -> &CandidateCoordinate {
        &self.coordinate
    }
    pub fn applicability(&self) -> &InvocationRegion {
        &self.applicability
    }
    pub fn executable(&self) -> crate::evaluation::TargetClosedExecutableView<'a, B> {
        crate::evaluation::TargetClosedExecutableView::new(self.implementation, &[], &[])
    }
}

/// The candidate domain of one entry on one target under one precision policy.
#[derive(Debug)]
pub struct CandidateDomain<B: seismic_target::TargetFamily> {
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
    universal: UniversalImplementation<B>,
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

    pub(crate) fn from_parts(parts: CandidateDomainParts<B>) -> Self {
        Self {
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

    /// Consumes the space for freezing: the arena becomes shared and
    /// immutable.
    pub(crate) fn into_parts(self) -> CandidateDomainParts<B> {
        CandidateDomainParts {
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

    pub fn families(&self) -> impl Iterator<Item = CandidateFamilyView<'_, B>> {
        std::iter::once(CandidateFamilyView {
            implementation: self.universal.as_inner(),
            constraints: None,
            axes: &[],
        })
        .chain(self.optimized.iter().map(|candidate| CandidateFamilyView {
            implementation: candidate.implementation.as_inner(),
            constraints: Some(&candidate.constraints),
            axes: &candidate.axes,
        }))
    }

    pub fn bind(
        &self,
        coordinate: CandidateCoordinate,
    ) -> Result<CandidateSlice<'_, B>, CoordinateError> {
        let (implementation, axes, predicate) = self.family_parts(&coordinate.family)?;
        let mut fixed = exact_choice_binding(&self.arena, axes, &coordinate.choices)?;
        for (symbol, value) in self.constants.bindings() {
            fixed.bind(*symbol, *value);
        }
        let applicability = self.arena.compile_bool_with(predicate, &fixed);
        if applicability.reads().is_empty()
            && !applicability
                .evaluate(&InvocationValues::new())
                .unwrap_or(false)
        {
            return Err(CoordinateError::EmptyRegion);
        }
        Ok(CandidateSlice {
            coordinate,
            implementation,
            applicability: InvocationRegion {
                predicate: applicability,
            },
        })
    }

    pub fn contains(&self, point: &CandidatePoint) -> Result<bool, MembershipError> {
        let slice = self
            .bind(point.coordinate.clone())
            .map_err(MembershipError::Coordinate)?;
        slice
            .applicability
            .contains(&point.invocation)
            .map_err(MembershipError::Invocation)
    }

    fn family_parts(
        &self,
        identity: &crate::implementation::ImplementationIdentity,
    ) -> Result<
        (
            &crate::implementation::Implementation<B>,
            &[ChoiceAxis],
            BoolExpr,
        ),
        CoordinateError,
    > {
        if self.universal.as_inner().identity() == identity {
            return Ok((
                self.universal.as_inner(),
                &[],
                self.target_domain.predicate().node(),
            ));
        }
        self.optimized
            .iter()
            .find(|candidate| candidate.implementation.as_inner().identity() == identity)
            .map(|candidate| {
                (
                    candidate.implementation.as_inner(),
                    candidate.axes.as_slice(),
                    candidate.constraints.predicate(),
                )
            })
            .ok_or(CoordinateError::UnknownFamily)
    }
}

pub struct CandidateFamilyView<'a, B: seismic_target::TargetFamily> {
    implementation: &'a crate::implementation::Implementation<B>,
    constraints: Option<&'a ConstraintSet>,
    axes: &'a [ChoiceAxis],
}

impl<'a, B: seismic_target::TargetFamily> CandidateFamilyView<'a, B> {
    pub fn identity(&self) -> &crate::implementation::ImplementationIdentity {
        self.implementation.identity()
    }
    pub fn choices(&self) -> &'a [ChoiceAxis] {
        self.axes
    }
    pub fn constraints(&self) -> &'a [DomainConstraint] {
        self.constraints.map_or(&[], ConstraintSet::conjuncts)
    }
    pub fn executable(&self) -> crate::evaluation::TargetClosedExecutableView<'a, B> {
        crate::evaluation::TargetClosedExecutableView::new(
            self.implementation,
            self.axes,
            self.constraints.map_or(&[], ConstraintSet::conjuncts),
        )
    }
}

pub(crate) struct CandidateDomainParts<B: seismic_target::TargetFamily> {
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
    pub universal: UniversalImplementation<B>,
    pub optimized: Vec<DomainCandidate<B>>,
    pub precision: PrecisionPolicy,
    pub optimization_exhausted: bool,
}

/// One closed implementation together with the complete planning predicate
/// that admits it.  Keeping these facts in one object prevents the solver,
/// freezer and coverage builder from joining parallel tables by index.
#[derive(Debug)]
pub(crate) struct DomainCandidate<B: seismic_target::TargetFamily> {
    pub(crate) implementation: OptimizedImplementation<B>,
    pub(crate) axes: Vec<ChoiceAxis>,
    pub(crate) constraints: ConstraintSet,
}

fn exact_choice_binding(
    arena: &ExprArena,
    axes: &[ChoiceAxis],
    choices: &[(DecisionId, i64)],
) -> Result<PartialAssignment, CoordinateError> {
    let mut assignment = PartialAssignment::new();
    for (decision, value) in choices {
        let Some(axis) = axes.iter().find(|axis| axis.decision == *decision) else {
            return Err(CoordinateError::ForeignChoice(*decision));
        };
        if assignment.get(arena.decision_symbol(*decision)).is_some() {
            return Err(CoordinateError::DuplicateChoice(*decision));
        }
        if !axis.values.contains(value) {
            return Err(CoordinateError::ValueOutsideAxis {
                decision: *decision,
                value: *value,
            });
        }
        assignment.bind(arena.decision_symbol(*decision), SymbolValue::Int(*value));
    }
    if let Some(axis) = axes.iter().find(|axis| {
        assignment
            .get(arena.decision_symbol(axis.decision))
            .is_none()
    }) {
        return Err(CoordinateError::MissingChoice(axis.decision));
    }
    Ok(assignment)
}

/// The consuming transition from a checked logical entry to the one sealed
/// candidate domain. Native reconciliation precedes sealing; performance
/// evaluation does not.
pub(crate) fn construct_candidate_domain<T, C>(
    entry: LogicalEntry,
    device: &DeviceDescription<T>,
    registry: &CompilerRegistry<T>,
    compiler: &C,
    native_context: &C::Context,
    precision: &PrecisionPolicy,
    evidence: &EvidenceCatalog,
    budget: &PreparationBudget,
) -> Result<
    (
        CandidateDomain<T>,
        crate::realization::RealizationRegistry<T, C::Handle>,
    ),
    PreparationError,
>
where
    T: seismic_target::TargetFamily,
    C: seismic_target::NativeCompiler<T>,
{
    internals::candidate_domain(
        entry,
        device,
        registry,
        compiler,
        native_context,
        precision,
        evidence,
        budget,
    )
}

pub(crate) mod internals {
    use super::*;
    use crate::implementation::candidate_native_template_identities;
    use crate::numerics;
    use crate::realization::realize_candidate;
    use crate::refinement::{
        RefinementCompletion, RefinementLimits, RefinementRequest, RefinementSession,
    };
    use seismic_lang::entry::{ParameterKind, ResultKind};
    use seismic_lang::expr::{AnyExpr, CmpOp, NodeView, RootName};

    pub(super) fn candidate_domain<T, C>(
        entry: LogicalEntry,
        target: &DeviceDescription<T>,
        registry: &CompilerRegistry<T>,
        compiler: &C,
        native_context: &C::Context,
        precision: &PrecisionPolicy,
        evidence: &EvidenceCatalog,
        budget: &PreparationBudget,
    ) -> Result<
        (
            CandidateDomain<T>,
            crate::realization::RealizationRegistry<T, C::Handle>,
        ),
        PreparationError,
    >
    where
        T: seismic_target::TargetFamily,
        C: seismic_target::NativeCompiler<T>,
    {
        let mut realizations =
            crate::realization::RealizationRegistry::new(target.identity().clone());
        let semantic_events = Arc::new(entry.semantic_event_manifest());
        let seismic_lang::entry::LogicalEntryParts {
            identity,
            module,
            schema,
            domain,
            mut arena,
            program,
        } = entry.into_parts();

        // Target facts enter the arena exactly once, before any
        // implementation is constructed. Every later physical constraint
        // therefore refers to this single set of symbols and bindings.
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
        let refinement_exhausted =
            !matches!(refinement_report.completion, RefinementCompletion::Complete)
                || !refinement_report.registered_factory_traversal_complete;

        // Refinement has its own narrow budget.  Native/artifact/metadata and
        // later solver charges begin here, at the consuming realization seam.
        let mut tracker = PreparationBudgetTracker::new(budget.clone());
        let templates = candidate_native_template_identities(&universal_family, target);
        let within_templates = tracker.record_required_native_templates(&templates)?;
        let (universal, universal_realization, metrics) = realize_candidate(
            universal_family,
            compiler,
            native_context,
            &mut arena,
            target,
            registry,
            &constants,
        )?;
        realizations.insert(universal.identity().clone(), universal_realization)?;
        let retained_bytes = universal.retained_metadata_bytes();
        let within_artifact = tracker.record_required_native_artifact(metrics)?;
        let within_metadata = tracker.record_required_metadata(retained_bytes)?;
        let mut realization_open = within_templates && within_artifact && within_metadata;

        let mut closed_optimized = Vec::new();
        for family in optimized_families {
            if !realization_open {
                break;
            }
            let templates = candidate_native_template_identities(&family, target);
            let within_templates = tracker.record_native_templates(&templates);
            let (implementation, realization, metrics) = realize_candidate(
                family,
                compiler,
                native_context,
                &mut arena,
                target,
                registry,
                &constants,
            )?;
            realizations.insert(implementation.identity().clone(), realization)?;
            let retained_bytes = implementation.retained_metadata_bytes();
            let within_artifact = tracker.record_native_artifact(metrics);
            let within_metadata = tracker.charge_metadata(retained_bytes);
            // A family whose exact charge crosses a ceiling remains retained;
            // the ceiling only suppresses later realization.
            closed_optimized.push(OptimizedImplementation::from_closed(implementation));
            realization_open = within_templates && within_artifact && within_metadata;
        }
        let optimization_exhausted = refinement_exhausted || !realization_open;
        let universal = UniversalImplementation::from_closed_reference(
            universal,
            &mut arena,
            target_domain.predicate().node(),
            &constants,
        )?;

        let target_node = target_domain.predicate().node();
        let mut optimized = Vec::with_capacity(closed_optimized.len());
        for implementation in closed_optimized {
            let numerical = numerics::admissibility(
                &mut arena,
                implementation.as_inner().numerical_transfer(),
                precision,
                evidence,
                implementation.as_inner().identity(),
                implementation.as_inner().decisions(),
                target.numerical_environment_identity(),
                implementation.as_inner().native_numerical_identity(),
                target_domain.identity(),
            );
            let semantic = implementation.as_inner().semantic_coverage().node();
            // TargetDomain may contain scalar preconditions. Those are
            // invocation facts, not compile-time decisions, and are
            // validated before selection. Keep them in the executable
            // guard while exporting only target/decision/shape facts to
            // the finite solver.
            // The solver must choose only a physically admissible
            // implementation. Keeping hard constraints solely in the
            // runtime guard recreates the forbidden split-brain path in
            // which search selects a plan that execution must reject.
            let hard = implementation.as_inner().hard_constraints();
            let constraints = vec![
                DomainConstraint {
                    origin: ConstraintOrigin::Invocation,
                    predicate: target_node,
                },
                DomainConstraint {
                    origin: ConstraintOrigin::SemanticApplicability,
                    predicate: semantic,
                },
                DomainConstraint {
                    origin: ConstraintOrigin::NumericalAdmissibility,
                    predicate: numerical,
                },
                DomainConstraint {
                    origin: ConstraintOrigin::NativeLegality,
                    predicate: hard,
                },
            ];
            let combined = arena.all(
                &constraints
                    .iter()
                    .map(|constraint| constraint.predicate)
                    .collect::<Vec<_>>(),
            );
            let axes = implementation
                .as_inner()
                .decisions()
                .iter()
                .map(|(decision, meaning)| ChoiceAxis {
                    decision: *decision,
                    meaning,
                    values: arena.decision_domain(*decision).values().to_vec(),
                })
                .collect();
            optimized.push(DomainCandidate {
                implementation,
                axes,
                constraints: ConstraintSet {
                    conjuncts: constraints,
                    combined,
                },
            });
        }

        let domain = CandidateDomain::from_parts(CandidateDomainParts {
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
        });
        Ok((domain, realizations))
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
