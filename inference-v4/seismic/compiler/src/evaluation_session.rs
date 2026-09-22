//! Private mechanical preparation capabilities used by candidate evaluators.
//!
//! This module is the sole owner of native compiler access, realization,
//! numerical admission, budgets, finalization, and artifact lifetime. An
//! evaluator can realize one checked coordinate at a time and later publish
//! retained admissions with its invocation decision. Raw services and native
//! handles never cross this boundary.

use crate::candidate_domain::{CandidateCoordinate, CandidateDomain, NonEmpty};
use crate::errors::PreparationError;
use crate::evaluation::EvaluationIdentity;
use crate::executable::RetainedCandidate;
use crate::frozen::{freeze, FinalizationContext};
use crate::implementation::{validate_universal_implementation, Implementation};
use crate::planning::{PlanningCoverage, SelectionPolicy};
use crate::preparation_budget::{PlanningBudget, PreparationBudget, PreparationBudgetTracker};
use crate::prepared::{InvocationContract, SelectionFunction};
use crate::realization::{
    CanonicalRealizationRequest, NativeCandidateReconciler, NativeReconciliationError,
    RealizationOutcome, RealizationRegistry, RealizedNativeSet, Realizer, ReconciliationInput,
};
use crate::target::CompilerRegistry;
use seismic_lang::expr::compiled::CompiledPredicate;
use seismic_lang::expr::{ExprArena, PartialAssignment};
use seismic_target::NativeCompiler;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub(crate) struct EvaluationRequest<P> {
    coordinate: CandidateCoordinate,
    payload: P,
}

impl<P> EvaluationRequest<P> {
    pub(crate) fn new(coordinate: CandidateCoordinate, payload: P) -> Self {
        Self {
            coordinate,
            payload,
        }
    }
}

/// Opaque admission returned by checked realization. Evaluators may retain or
/// discard it, but cannot manufacture, inspect, or alter mechanical state.
pub(crate) struct AdmittedCandidate<T: seismic_target::TargetFamily, P> {
    session: u64,
    coordinate: CandidateCoordinate,
    universal: bool,
    pending: PendingCandidate<T, P>,
}

pub(crate) enum RealizationAdmission<T: seismic_target::TargetFamily, P> {
    Admitted(AdmittedCandidate<T, P>),
    Rejected,
    BudgetClosed,
}

/// Method-neutral input for compiling an evaluator's invocation decision.
/// The payload remains evaluator-owned; the guard and fixed bindings are the
/// exact admitted candidate facts produced by shared preparation.
pub(crate) struct PreparedSelection<P> {
    payload: P,
    guard: CompiledPredicate,
    fixed: PartialAssignment,
}

/// The exact retained prefix, in policy order. Its private construction keeps
/// selector inputs aligned with the candidates finalized by this session.
pub(crate) struct PreparedPortfolio<P> {
    selections: NonEmpty<PreparedSelection<P>>,
}

impl<P> PreparedPortfolio<P> {
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.selections.len()
    }

    pub(crate) fn map_payload<Q>(
        self,
        mut map: impl FnMut(P, &PartialAssignment) -> Q,
    ) -> PreparedPortfolio<Q> {
        let selections = self
            .selections
            .into_vec()
            .into_iter()
            .map(|selection| PreparedSelection {
                payload: map(selection.payload, &selection.fixed),
                guard: selection.guard,
                fixed: selection.fixed,
            })
            .collect();
        PreparedPortfolio {
            selections: NonEmpty::new(selections).expect("a prepared portfolio is nonempty"),
        }
    }

    pub(crate) fn into_guards_and_payloads(self) -> NonEmpty<(CompiledPredicate, P)> {
        NonEmpty::new(
            self.selections
                .into_vec()
                .into_iter()
                .map(|selection| (selection.guard, selection.payload))
                .collect(),
        )
        .expect("a prepared portfolio is nonempty")
    }

    #[cfg(test)]
    pub(crate) fn from_test_guards(guards: NonEmpty<CompiledPredicate>) -> PreparedPortfolio<()> {
        PreparedPortfolio {
            selections: NonEmpty::new(
                guards
                    .into_vec()
                    .into_iter()
                    .map(|guard| PreparedSelection {
                        payload: (),
                        guard,
                        fixed: PartialAssignment::new(),
                    })
                    .collect(),
            )
            .unwrap(),
        }
    }
}

struct ImplementationReconciler<'a, T: seismic_target::TargetFamily> {
    target: &'a seismic_target::DeviceDescription<T>,
    registry: &'a CompilerRegistry<T>,
}

impl<T: seismic_target::TargetFamily> NativeCandidateReconciler<T>
    for ImplementationReconciler<'_, T>
{
    type Output = Implementation<T>;

    fn reconcile(
        &mut self,
        arena: &mut ExprArena,
        input: ReconciliationInput<T>,
    ) -> Result<Self::Output, NativeReconciliationError> {
        let (family, assignment, native) = input.into_parts();
        crate::implementation::reconcile_candidate_with_descriptions(
            family,
            assignment,
            arena,
            self.target,
            self.registry,
            &native,
        )
        .map_err(NativeReconciliationError::Preparation)
    }
}

struct PendingCandidate<T: seismic_target::TargetFamily, P> {
    implementation: Arc<Implementation<T>>,
    native: Arc<RealizedNativeSet<T>>,
    guard: seismic_lang::expr::BoolExpr,
    numerical: crate::numerics::NumericalAssessment,
    payload: P,
}

pub(crate) struct EvaluationCompletion<T: seismic_target::TargetFamily, H> {
    pub(crate) entry: seismic_lang::ids::StableEntryId,
    pub(crate) module: seismic_lang::ids::ModuleHash,
    pub(crate) schema: Arc<seismic_lang::entry::CallSchema>,
    pub(crate) semantic_events: Arc<seismic_lang::entry::SemanticEventManifest>,
    pub(crate) device: seismic_target::DeviceDescriptionIdentity,
    pub(crate) evaluation: EvaluationIdentity,
    pub(crate) invocation: InvocationContract,
    pub(crate) coverage: PlanningCoverage,
    pub(crate) realizations: RealizationRegistry<T, H>,
}

struct ActiveEvaluation<'a, T, C>
where
    T: seismic_target::TargetFamily,
    C: NativeCompiler<T>,
{
    id: u64,
    domain: CandidateDomain<T>,
    evaluation: EvaluationIdentity,
    coverage: PlanningCoverage,
    realizer: Realizer<'a, T, C, ImplementationReconciler<'a, T>>,
    accounting: PreparationBudgetTracker,
    requested: HashSet<CandidateCoordinate>,
    universal_admitted: bool,
    optional_native_open: bool,
}

pub(crate) struct EvaluationSession<'a, T, C>
where
    T: seismic_target::TargetFamily,
    C: NativeCompiler<T>,
{
    registry: &'a CompilerRegistry<T>,
    compiler: &'a C,
    native_context: &'a C::Context,
    device: &'a seismic_target::DeviceDescription<T>,
    preparation_budget: &'a PreparationBudget,
    planning_budget: &'a PlanningBudget,
    opened: bool,
    active: Option<ActiveEvaluation<'a, T, C>>,
    completion: Option<EvaluationCompletion<T, C::Handle>>,
}

impl<'a, T, C> EvaluationSession<'a, T, C>
where
    T: seismic_target::TargetFamily,
    C: NativeCompiler<T>,
{
    pub(crate) fn new(
        registry: &'a CompilerRegistry<T>,
        compiler: &'a C,
        native_context: &'a C::Context,
        device: &'a seismic_target::DeviceDescription<T>,
        preparation_budget: &'a PreparationBudget,
        planning_budget: &'a PlanningBudget,
    ) -> Self {
        Self {
            registry,
            compiler,
            native_context,
            device,
            preparation_budget,
            planning_budget,
            opened: false,
            active: None,
            completion: None,
        }
    }

    pub(crate) fn search_budget(&self) -> &PlanningBudget {
        self.planning_budget
    }

    /// Opens exactly one domain for checked, incremental realization.
    pub(crate) fn begin(
        &mut self,
        domain: CandidateDomain<T>,
        evaluation: EvaluationIdentity,
        coverage: PlanningCoverage,
    ) -> Result<(), PreparationError> {
        if self.opened {
            return Err(PreparationError::InvalidCandidateDomain(
                "an evaluation session can be opened exactly once".into(),
            ));
        }
        if domain.device_identity() != self.device.identity()
            || evaluation.device() != self.device.identity()
        {
            return Err(PreparationError::InvalidCandidateDomain(
                "evaluation session device does not match the candidate domain".into(),
            ));
        }
        static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);
        let id = NEXT_SESSION.fetch_add(1, Ordering::Relaxed);
        let reconciler = ImplementationReconciler {
            target: self.device,
            registry: self.registry,
        };
        self.active = Some(ActiveEvaluation {
            id,
            domain,
            evaluation,
            coverage,
            realizer: Realizer::new(self.compiler, self.native_context, self.device, reconciler),
            accounting: PreparationBudgetTracker::new(self.preparation_budget.clone()),
            requested: HashSet::new(),
            universal_admitted: false,
            optional_native_open: true,
        });
        self.opened = true;
        Ok(())
    }

    /// Realizes and admits one coordinate that was checked by the authoritative
    /// domain. The universal coordinate must be requested first. Rejection and
    /// budget closure are explicit and never expose native state.
    pub(crate) fn realize_checked<P>(
        &mut self,
        request: EvaluationRequest<P>,
    ) -> Result<RealizationAdmission<T, P>, PreparationError> {
        let state = self.active.as_mut().ok_or_else(|| {
            PreparationError::InvalidCandidateDomain(
                "checked realization requires an open evaluation session".into(),
            )
        })?;
        let checked = state
            .domain
            .checked_preparation_candidate(&request.coordinate)
            .map_err(|error| {
                PreparationError::InvalidCandidateDomain(format!(
                    "evaluator requested an invalid coordinate: {error:?}"
                ))
            })?;
        if !state.universal_admitted && !checked.is_universal {
            return Err(PreparationError::InvalidCandidateDomain(
                "candidate evaluation must realize the universal coordinate first".into(),
            ));
        }
        if !state.requested.insert(checked.coordinate.clone()) {
            return Err(PreparationError::InvalidCandidateDomain(
                "candidate evaluation requested one coordinate more than once".into(),
            ));
        }
        if !checked.is_universal && !state.optional_native_open {
            return Ok(RealizationAdmission::BudgetClosed);
        }

        let realization = CanonicalRealizationRequest::from_checked_coordinate(
            checked.family,
            state.domain.arena(),
            &checked.coordinate,
        )
        .map_err(|error| {
            PreparationError::InvalidCandidateDomain(format!(
                "checked coordinate could not be realized: {error:?}"
            ))
        })?;
        let outcome = state
            .realizer
            .realize(state.domain.arena_mut(), realization)?;
        let (realized, rejection, metrics) = match outcome {
            RealizationOutcome::Ready {
                candidate,
                newly_formed,
            } => (Some(candidate), None, newly_formed),
            RealizationOutcome::Rejected {
                rejection,
                newly_formed,
            } => (None, Some(rejection), newly_formed),
        };
        let within_optional_budget = if checked.is_universal {
            state.accounting.record_required_native_artifact(metrics)?
        } else {
            state.accounting.record_native_artifact(metrics)
        };
        if !within_optional_budget {
            state.optional_native_open = false;
            if matches!(
                state.coverage.optimization,
                crate::planning::OptimizationCompletion::Complete
            ) {
                state.coverage.optimization = crate::planning::OptimizationCompletion::Limited(
                    crate::planning::PlanningLimit::NativeArtifacts,
                );
            }
        }
        let Some(realized) = realized else {
            if checked.is_universal {
                return Err(PreparationError::UniversalClosure(format!(
                    "native reconciliation rejected the covered implementation: {}",
                    rejection
                        .expect("rejected outcome carries its reason")
                        .reason()
                )));
            }
            return Ok(RealizationAdmission::Rejected);
        };

        let (precision, evidence, target, target_domain, constants) =
            state.domain.numerical_context();
        let implementation = realized.reconciled().clone();
        let decisions = implementation.decisions();
        let numerical = crate::numerics::admissibility(
            state.domain.arena_mut(),
            implementation.numerical_transfer(),
            &precision,
            &evidence,
            implementation.identity(),
            &decisions,
            &target,
            implementation.native_numerical_identity(),
            target_domain.identity(),
        );
        let guard = state.domain.arena_mut().all(&[
            checked.constraint,
            implementation.hard_constraints(),
            numerical,
        ]);
        let fixed = fixed_for_coordinate(state.domain.arena(), &constants, implementation.as_ref());
        let compiled_guard = state.domain.arena().compile_bool_with(guard, &fixed);
        if compiled_guard.reads().is_empty()
            && !compiled_guard
                .evaluate(&seismic_lang::expr::compiled::InvocationValues::new())
                .unwrap_or(false)
        {
            if checked.is_universal {
                return Err(PreparationError::UniversalClosure(
                    "covered implementation became inadmissible after native reconciliation".into(),
                ));
            }
            return Ok(RealizationAdmission::Rejected);
        }
        let assessment = crate::numerics::assess(
            state.domain.arena(),
            implementation.numerical_transfer(),
            &precision,
            &evidence,
            implementation.identity(),
            &target,
            implementation.native_numerical_identity(),
            target_domain.identity(),
            checked.coordinate.choices(),
        );
        if checked.is_universal {
            validate_universal_implementation(
                &implementation,
                state.domain.arena_mut(),
                target_domain.predicate().node(),
                &constants,
            )?;
            state.universal_admitted = true;
        }
        Ok(RealizationAdmission::Admitted(AdmittedCandidate {
            session: state.id,
            coordinate: checked.coordinate,
            universal: checked.is_universal,
            pending: PendingCandidate {
                implementation,
                native: realized.native().clone(),
                guard,
                numerical: assessment,
                payload: request.payload,
            },
        }))
    }

    /// Consumes retained admissions, seals the arena, compiles the exact
    /// portfolio, and publishes completion exactly once.
    pub(crate) fn publish<P, F>(
        &mut self,
        admitted: NonEmpty<AdmittedCandidate<T, P>>,
        build_selection: F,
    ) -> Result<SelectionPolicy<T>, PreparationError>
    where
        F: FnOnce(&ExprArena, PreparedPortfolio<P>) -> SelectionFunction,
    {
        let mut state = self.active.take().ok_or_else(|| {
            PreparationError::InvalidCandidateDomain(
                "publication requires an open evaluation session".into(),
            )
        })?;
        let admitted = admitted.into_vec();
        if !state.universal_admitted || !admitted[0].universal || admitted[0].session != state.id {
            return Err(PreparationError::InvalidCandidateDomain(
                "published candidates must begin with this session's universal admission".into(),
            ));
        }
        let mut retained_coordinates = HashSet::new();
        for candidate in &admitted {
            if candidate.session != state.id
                || !retained_coordinates.insert(candidate.coordinate.clone())
            {
                return Err(PreparationError::InvalidCandidateDomain(
                    "published admissions are foreign or duplicated".into(),
                ));
            }
        }

        let parts = state.domain.into_parts();
        let arena = Arc::new(parts.arena);
        let invocation = InvocationContract::compile(
            &arena,
            &parts.schema,
            parts.target_domain,
            &fixed_target_constants(&parts.constants),
        );
        let finalization = FinalizationContext {
            schema: parts.schema.clone(),
            device: parts.device.clone(),
            evaluation: state.evaluation.clone(),
            arena: arena.clone(),
            constants: parts.constants,
        };
        let mut registry_handles = state.realizer.into_registry();
        let mut variants: Vec<RetainedCandidate<T>> = Vec::with_capacity(admitted.len());
        let mut selections = Vec::with_capacity(admitted.len());
        let mut retained_metadata_bytes = 0_u64;
        for (index, admitted) in admitted.into_iter().enumerate() {
            let pending = admitted.pending;
            let fixed = fixed_for_coordinate(
                &arena,
                &finalization.constants,
                pending.implementation.as_ref(),
            );
            let guard = arena.compile_bool_with(pending.guard, &fixed);
            let implementation_identity = pending.implementation.identity().clone();
            let native = pending.native.clone();
            let candidate = crate::executable::compile_variant(freeze(
                &finalization,
                pending.implementation,
                pending.guard,
                pending.numerical,
            ));
            let candidate_bytes = candidate.retained_metadata_bytes();
            if index == 0 && candidate_bytes > self.planning_budget.required_retained_metadata_bytes
            {
                return Err(PreparationError::Planning(
                    crate::planning::PlanningError::BudgetExceeded {
                        resource: crate::planning::PlanningBudgetResource::RequiredRetainedMetadata,
                        required: candidate_bytes,
                        limit: self.planning_budget.required_retained_metadata_bytes,
                    },
                ));
            }
            let next_metadata = retained_metadata_bytes.saturating_add(candidate_bytes);
            if index > 0 && next_metadata > self.planning_budget.retained_metadata_bytes {
                if matches!(
                    state.coverage.optimization,
                    crate::planning::OptimizationCompletion::Complete
                ) {
                    state.coverage.optimization = crate::planning::OptimizationCompletion::Limited(
                        crate::planning::PlanningLimit::RetainedMetadata,
                    );
                }
                break;
            }
            retained_metadata_bytes = next_metadata;
            registry_handles.bind_implementation(implementation_identity, &native)?;
            variants.push(candidate);
            selections.push(PreparedSelection {
                payload: pending.payload,
                guard,
                fixed,
            });
        }
        state.coverage.budget.retained_metadata_bytes = retained_metadata_bytes;
        state.coverage.budget.executable_variants = variants.len() as u64;
        let variants =
            NonEmpty::new(variants).expect("covered policy retains its universal candidate");
        let selections = PreparedPortfolio {
            selections: NonEmpty::new(selections).expect("covered policy retains a selection case"),
        };
        let selection_function = build_selection(&arena, selections);
        let policy = SelectionPolicy {
            candidates: variants,
            selection_function,
        };
        let completion = EvaluationCompletion {
            entry: parts.entry,
            module: parts.module,
            schema: parts.schema,
            semantic_events: parts.semantic_events,
            device: parts.device,
            evaluation: state.evaluation,
            invocation,
            coverage: state.coverage,
            realizations: registry_handles,
        };
        assert!(
            self.completion.replace(completion).is_none(),
            "an evaluator may complete a preparation session exactly once"
        );
        Ok(policy)
    }

    pub(crate) fn take_completion(&mut self) -> Option<EvaluationCompletion<T, C::Handle>> {
        self.completion.take()
    }
}

fn fixed_target_constants(constants: &crate::target::TargetConstants) -> PartialAssignment {
    let mut fixed = PartialAssignment::new();
    for (symbol, value) in constants.bindings() {
        fixed.bind(*symbol, *value);
    }
    fixed
}

fn fixed_for_coordinate<T: seismic_target::TargetFamily>(
    arena: &ExprArena,
    constants: &crate::target::TargetConstants,
    implementation: &Implementation<T>,
) -> PartialAssignment {
    let mut fixed = fixed_target_constants(constants);
    for (decision, _) in implementation.decisions() {
        let symbol = arena.decision_symbol(decision);
        if let Some(value) = implementation.assignment().get(symbol) {
            fixed.bind(symbol, value);
        }
    }
    fixed
}

#[cfg(test)]
mod boundary_tests {
    use super::*;
    use crate::candidate_domain::DomainCandidate;
    use crate::evaluation::{CandidateEvaluator, EvaluationProvenance};
    use crate::numerics::EvidenceCatalog;
    use crate::planning::{OptimizationCompletion, PlanningBudgetReport, TargetCoverage};
    use crate::realization::demand_driven_tests::{
        choice_family, device, CountingCompiler, FakeTarget,
    };
    use crate::target::CompilerRegistryParts;
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};
    use seismic_lang::entry::ElementBindings;
    use seismic_lang::expr::compiled::InvocationValues;
    use seismic_lang::precision::PrecisionPolicy;

    fn registry() -> CompilerRegistry<FakeTarget> {
        CompilerRegistry::assemble(CompilerRegistryParts {
            capabilities: Vec::new(),
            structural_factories: Vec::new(),
            independent_launch_mode: (),
            cooperative_launch_mode: |_| None,
            native_launch_constraints: |_, _, _, _, _, _| Vec::new(),
            addressable_resources: |_| Vec::new(),
            emitted_intrinsics: Default::default(),
        })
    }

    fn domain(
        device: &seismic_target::DeviceDescription<FakeTarget>,
        registry: &CompilerRegistry<FakeTarget>,
    ) -> CandidateDomain<FakeTarget> {
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "session-fixture.seismic".into(),
            text: "fn probe(x: f32) -> f32:\n    return x\n".into(),
        }]))
        .unwrap();
        let entry = module
            .entry(
                module.entry_named("probe").unwrap(),
                &ElementBindings::default(),
            )
            .unwrap();
        crate::candidate_domain::construct_candidate_domain(
            entry,
            device,
            registry,
            &PrecisionPolicy::Exact,
            &EvidenceCatalog::default(),
            &PreparationBudget::default(),
        )
        .unwrap()
    }

    fn domain_with_optional(
        device: &seismic_target::DeviceDescription<FakeTarget>,
        registry: &CompilerRegistry<FakeTarget>,
    ) -> (CandidateDomain<FakeTarget>, CandidateCoordinate) {
        let mut parts = domain(device, registry).into_parts();
        let (family, decision, _) = choice_family(&mut parts.arena);
        let family_id = family.identity().clone();
        parts.optimized.push(DomainCandidate {
            family,
            constraints: parts.universal.constraints.clone(),
            numerical: parts.universal.numerical,
        });
        let domain = CandidateDomain::from_parts(parts);
        let optional = domain
            .canonicalize(domain.proposal(family_id, vec![(decision, 0)]))
            .unwrap();
        (domain, optional)
    }

    fn coverage() -> PlanningCoverage {
        PlanningCoverage {
            target: TargetCoverage::Exhaustive,
            optimization: OptimizationCompletion::Complete,
            budget: PlanningBudgetReport::default(),
        }
    }

    fn identity(device: &seismic_target::DeviceDescription<FakeTarget>) -> EvaluationIdentity {
        let provenance = EvaluationProvenance::new([4; 32], [5; 32]);
        EvaluationIdentity::new(device.identity().clone(), provenance)
    }

    fn select_first(_: &ExprArena, selections: PreparedPortfolio<()>) -> SelectionFunction {
        SelectionFunction::ordered_decision(selections, Vec::new())
    }

    struct FirstAdmitted;

    impl CandidateEvaluator<FakeTarget, CountingCompiler> for FirstAdmitted {
        fn evaluate(
            &self,
            domain: CandidateDomain<FakeTarget>,
            session: &mut EvaluationSession<'_, FakeTarget, CountingCompiler>,
        ) -> Result<SelectionPolicy<FakeTarget>, PreparationError> {
            let universal = domain.canonicalize(domain.universal_proposal()).unwrap();
            let device = session.device;
            session.begin(domain, identity(device), coverage())?;
            let RealizationAdmission::Admitted(universal) =
                session.realize_checked(EvaluationRequest::new(universal, ()))?
            else {
                panic!("universal candidate must be admitted")
            };
            session.publish(NonEmpty::new(vec![universal]).unwrap(), select_first)
        }
    }

    #[test]
    fn fake_evaluator_uses_checked_session_publication_and_exact_materialization() {
        let device = device();
        let registry = registry();
        let compiler = CountingCompiler::new();
        let domain = domain(&device, &registry);
        let preparation_budget = PreparationBudget::default();
        let planning_budget = PlanningBudget::default();
        let mut session = EvaluationSession::new(
            &registry,
            &compiler,
            &(),
            &device,
            &preparation_budget,
            &planning_budget,
        );
        let policy = FirstAdmitted.evaluate(domain, &mut session).unwrap();
        assert_eq!(policy.candidates.len(), 1);
        assert_eq!(
            policy
                .selection_function
                .apply(&InvocationValues::new())
                .as_usize(),
            0
        );
        let completion = session.take_completion().unwrap();
        assert_eq!(completion.coverage.target, TargetCoverage::Exhaustive);
        let prepared = crate::prepare::materialize(policy, completion);
        assert_eq!(prepared.variants().len(), 1);
        assert_eq!(prepared.select(&InvocationValues::new()).as_usize(), 0);
        assert!(session.take_completion().is_none());
        assert!(matches!(
            session.begin(
                self::domain(&device, &registry),
                identity(&device),
                coverage()
            ),
            Err(PreparationError::InvalidCandidateDomain(_))
        ));
    }

    #[test]
    fn checked_session_requires_universal_first_and_each_coordinate_once() {
        let device = device();
        let registry = registry();
        let compiler = CountingCompiler::new();
        let (domain, optional) = domain_with_optional(&device, &registry);
        let universal = domain.canonicalize(domain.universal_proposal()).unwrap();
        let preparation_budget = PreparationBudget::default();
        let planning_budget = PlanningBudget::default();
        let mut session = EvaluationSession::new(
            &registry,
            &compiler,
            &(),
            &device,
            &preparation_budget,
            &planning_budget,
        );
        session
            .begin(domain, identity(&device), coverage())
            .unwrap();
        assert!(matches!(
            session.realize_checked(EvaluationRequest::new(optional.clone(), ())),
            Err(PreparationError::InvalidCandidateDomain(_))
        ));
        assert_eq!(compiler.form_count(), 0);
        let admitted = session
            .realize_checked(EvaluationRequest::new(universal.clone(), ()))
            .unwrap();
        assert!(matches!(admitted, RealizationAdmission::Admitted(_)));
        assert!(matches!(
            session.realize_checked(EvaluationRequest::new(optional, ())),
            Ok(RealizationAdmission::Admitted(_))
        ));
        assert!(matches!(
            session.realize_checked(EvaluationRequest::new(universal, ())),
            Err(PreparationError::InvalidCandidateDomain(_))
        ));
    }

    #[test]
    fn admitted_ordinals_survive_publication_and_materialization() {
        let device = device();
        let registry = registry();
        let compiler = CountingCompiler::new();
        let (domain, optional) = domain_with_optional(&device, &registry);
        let universal = domain.canonicalize(domain.universal_proposal()).unwrap();
        let preparation_budget = PreparationBudget::default();
        let planning_budget = PlanningBudget::default();
        let mut session = EvaluationSession::new(
            &registry,
            &compiler,
            &(),
            &device,
            &preparation_budget,
            &planning_budget,
        );
        session
            .begin(domain, identity(&device), coverage())
            .unwrap();
        let RealizationAdmission::Admitted(universal) = session
            .realize_checked(EvaluationRequest::new(universal, ()))
            .unwrap()
        else {
            panic!("universal admission failed")
        };
        let RealizationAdmission::Admitted(optional) = session
            .realize_checked(EvaluationRequest::new(optional, ()))
            .unwrap()
        else {
            panic!("optional admission failed")
        };
        let policy = session
            .publish(
                NonEmpty::new(vec![universal, optional]).unwrap(),
                |_, selections| {
                    let mut selector_arena = ExprArena::new();
                    let always_node = selector_arena.bool(true);
                    let always = selector_arena.compile_bool(always_node);
                    SelectionFunction::ordered_decision(
                        selections,
                        vec![(always, crate::prepared::CandidateIndex::from_usize(1))],
                    )
                },
            )
            .unwrap();
        assert_eq!(policy.candidates.len(), 2);
        let prepared = crate::prepare::materialize(policy, session.take_completion().unwrap());
        assert_eq!(prepared.variants().len(), 2);
        assert_eq!(prepared.select(&InvocationValues::new()).as_usize(), 1);
    }

    #[test]
    fn selector_sees_only_candidates_retained_under_metadata_budget() {
        let device = device();
        let registry = registry();
        let compiler = CountingCompiler::new();
        let (domain, optional) = domain_with_optional(&device, &registry);
        let universal = domain.canonicalize(domain.universal_proposal()).unwrap();
        let preparation_budget = PreparationBudget::default();
        let planning_budget = PlanningBudget {
            retained_metadata_bytes: 0,
            ..PlanningBudget::default()
        };
        let mut session = EvaluationSession::new(
            &registry,
            &compiler,
            &(),
            &device,
            &preparation_budget,
            &planning_budget,
        );
        session
            .begin(domain, identity(&device), coverage())
            .unwrap();
        let RealizationAdmission::Admitted(universal) = session
            .realize_checked(EvaluationRequest::new(universal, ()))
            .unwrap()
        else {
            panic!("universal admission failed")
        };
        let RealizationAdmission::Admitted(optional) = session
            .realize_checked(EvaluationRequest::new(optional, ()))
            .unwrap()
        else {
            panic!("optional admission failed")
        };
        let policy = session
            .publish(
                NonEmpty::new(vec![universal, optional]).unwrap(),
                |arena, selections| {
                    assert_eq!(selections.len(), 1);
                    select_first(arena, selections)
                },
            )
            .unwrap();
        assert_eq!(policy.candidates.len(), 1);
        assert!(matches!(
            session.take_completion().unwrap().coverage.optimization,
            OptimizationCompletion::Limited(crate::planning::PlanningLimit::RetainedMetadata)
        ));
    }

    #[test]
    fn native_budget_closure_blocks_the_next_optional_compilation() {
        let device = device();
        let registry = registry();
        let compiler = CountingCompiler::new();
        let (domain, optional) = domain_with_optional(&device, &registry);
        let universal = domain.canonicalize(domain.universal_proposal()).unwrap();
        let preparation_budget = PreparationBudget {
            native_code_bytes: 0,
            ..PreparationBudget::default()
        };
        let planning_budget = PlanningBudget::default();
        let mut session = EvaluationSession::new(
            &registry,
            &compiler,
            &(),
            &device,
            &preparation_budget,
            &planning_budget,
        );
        session
            .begin(domain, identity(&device), coverage())
            .unwrap();
        let universal = session
            .realize_checked(EvaluationRequest::new(universal, ()))
            .unwrap();
        assert!(matches!(universal, RealizationAdmission::Admitted(_)));
        let before_optional = compiler.form_count();
        let optional = session
            .realize_checked(EvaluationRequest::new(optional, ()))
            .unwrap();
        assert!(matches!(optional, RealizationAdmission::BudgetClosed));
        assert_eq!(compiler.form_count(), before_optional);
    }

    #[test]
    fn publication_rejects_foreign_and_duplicate_admissions() {
        let device = device();
        let registry = registry();
        let compiler = CountingCompiler::new();
        let preparation_budget = PreparationBudget::default();
        let planning_budget = PlanningBudget::default();
        let mut first = EvaluationSession::new(
            &registry,
            &compiler,
            &(),
            &device,
            &preparation_budget,
            &planning_budget,
        );
        let mut second = EvaluationSession::new(
            &registry,
            &compiler,
            &(),
            &device,
            &preparation_budget,
            &planning_budget,
        );
        let first_domain = domain(&device, &registry);
        let first_coordinate = first_domain
            .canonicalize(first_domain.universal_proposal())
            .unwrap();
        first
            .begin(first_domain, identity(&device), coverage())
            .unwrap();
        let RealizationAdmission::Admitted(foreign) = first
            .realize_checked(EvaluationRequest::new(first_coordinate, ()))
            .unwrap()
        else {
            panic!("universal admission failed")
        };
        let second_domain = domain(&device, &registry);
        let second_coordinate = second_domain
            .canonicalize(second_domain.universal_proposal())
            .unwrap();
        second
            .begin(second_domain, identity(&device), coverage())
            .unwrap();
        let RealizationAdmission::Admitted(local) = second
            .realize_checked(EvaluationRequest::new(second_coordinate, ()))
            .unwrap()
        else {
            panic!("universal admission failed")
        };
        assert!(matches!(
            second.publish(NonEmpty::new(vec![local, foreign]).unwrap(), select_first),
            Err(PreparationError::InvalidCandidateDomain(_))
        ));
        assert!(second.take_completion().is_none());

        let mut third = EvaluationSession::new(
            &registry,
            &compiler,
            &(),
            &device,
            &preparation_budget,
            &planning_budget,
        );
        let third_domain = domain(&device, &registry);
        let third_coordinate = third_domain
            .canonicalize(third_domain.universal_proposal())
            .unwrap();
        third
            .begin(third_domain, identity(&device), coverage())
            .unwrap();
        let RealizationAdmission::Admitted(local) = third
            .realize_checked(EvaluationRequest::new(third_coordinate, ()))
            .unwrap()
        else {
            panic!("universal admission failed")
        };
        let duplicate = AdmittedCandidate {
            session: local.session,
            coordinate: local.coordinate.clone(),
            universal: local.universal,
            pending: PendingCandidate {
                implementation: local.pending.implementation.clone(),
                native: local.pending.native.clone(),
                guard: local.pending.guard,
                numerical: local.pending.numerical.clone(),
                payload: (),
            },
        };
        assert!(matches!(
            third.publish(NonEmpty::new(vec![local, duplicate]).unwrap(), select_first),
            Err(PreparationError::InvalidCandidateDomain(_))
        ));
        assert!(third.take_completion().is_none());
    }
}
