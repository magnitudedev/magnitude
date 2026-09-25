//! Preparation owns the structural domain and reusable executable candidates.
//! Evaluators decide retention; finalization only packages their exact policy.

use crate::candidate_domain::{CandidateCoordinate, CandidateDomain, NonEmpty};
use crate::errors::PreparationError;
use crate::evaluation::EvaluationIdentity;
use crate::executable::{ExecutableVariant, RetainedCandidate};
use crate::frozen::{freeze, CandidateContext};
use crate::implementation::{validate_universal_implementation, Implementation};
use crate::planning::{PlanningReport, SelectionPolicy};
use crate::preparation_budget::{PlanningBudget, PreparationBudget, PreparationBudgetTracker};
use crate::prepared::{InvocationContract, PreparedKernel, SelectionFunction};
use crate::realization::{
    CanonicalRealizationRequest, NativeCandidateReconciler, NativeReconciliationError,
    RealizationOutcome, Realizer, ReconciliationInput,
};
use crate::target::CompilerRegistry;
use seismic_lang::expr::compiled::CompiledPredicate;
use seismic_lang::expr::{ExprArena, PartialAssignment};
use seismic_native_target::NativeCompiler;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

/// A stable reference into one preparation, never a native handle or a policy index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PreparedCandidateId {
    preparation: u64,
    index: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RealizationAdmission {
    Ready(PreparedCandidateId),
    Rejected(std::sync::Arc<crate::realization::CandidateRejection>),
    Pending(RealizationPending),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RealizationPending {
    /// Numerical reasoning has not established an applicable region.
    NumericalAnalysis,
    /// Optional native formation exhausted its current preparation allowance.
    NativeBudget,
}

/// Method-neutral input for compiling an evaluator's invocation decision.
/// The payload remains evaluator-owned; the guard and fixed bindings are the
/// exact admitted candidate facts produced by shared preparation.
pub(crate) struct PreparedSelection<P> {
    candidate: PreparedCandidateId,
    payload: P,
    guard: CompiledPredicate,
    fixed: PartialAssignment,
}

/// Admitted candidate operands, in selector order. Consuming this portfolio
/// keeps each candidate's identity, guard and evaluator payload together.
pub struct PreparedPortfolio<P> {
    selections: NonEmpty<PreparedSelection<P>>,
}

impl<P> PreparedPortfolio<P> {
    /// Resolve an ordinal against this exact admitted portfolio.
    pub fn index(&self, ordinal: usize) -> Option<crate::prepared::CandidateIndex> {
        (ordinal < self.selections.len())
            .then(|| crate::prepared::CandidateIndex::from_usize(ordinal))
    }

    pub fn len(&self) -> usize {
        self.selections.len()
    }

    pub fn map_payload<Q>(
        self,
        mut map: impl FnMut(P, &PartialAssignment) -> Q,
    ) -> PreparedPortfolio<Q> {
        let selections = self
            .selections
            .into_vec()
            .into_iter()
            .map(|selection| PreparedSelection {
                candidate: selection.candidate,
                payload: map(selection.payload, &selection.fixed),
                guard: selection.guard,
                fixed: selection.fixed,
            })
            .collect();
        PreparedPortfolio {
            selections: NonEmpty::new(selections).expect("a prepared portfolio is nonempty"),
        }
    }

    pub(crate) fn into_operands(
        self,
    ) -> (
        NonEmpty<PreparedCandidateId>,
        NonEmpty<(CompiledPredicate, P)>,
    ) {
        let (candidates, values): (Vec<_>, Vec<_>) = self
            .selections
            .into_vec()
            .into_iter()
            .map(|selection| (selection.candidate, (selection.guard, selection.payload)))
            .unzip();
        (
            NonEmpty::new(candidates).expect("a prepared portfolio is nonempty"),
            NonEmpty::new(values).expect("a prepared portfolio is nonempty"),
        )
    }

    #[cfg(test)]
    pub(crate) fn from_test_guards(guards: NonEmpty<CompiledPredicate>) -> PreparedPortfolio<()> {
        PreparedPortfolio {
            selections: NonEmpty::new(
                guards
                    .into_vec()
                    .into_iter()
                    .enumerate()
                    .map(|(index, guard)| PreparedSelection {
                        candidate: PreparedCandidateId {
                            preparation: 0,
                            index,
                        },
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

struct ImplementationReconciler<'a, T: seismic_native_target::TargetFamily> {
    target: &'a seismic_native_target::DeviceDescription<T>,
    registry: &'a CompilerRegistry<T>,
}

impl<T: seismic_native_target::TargetFamily> NativeCandidateReconciler<T>
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
            native,
        )
        .map_err(NativeReconciliationError::Preparation)
    }
}

struct PreparedCandidate<T: seismic_native_target::TargetFamily, H> {
    universal: bool,
    fixed: PartialAssignment,
    retained: RetainedCandidate<T>,
    executable: ExecutableVariant<T, H>,
}

pub struct EvaluationSession<'a, T, C>
where
    T: seismic_native_target::TargetFamily,
    C: NativeCompiler<T>,
{
    id: u64,
    domain: CandidateDomain<'a, T>,
    invocation: std::sync::Arc<InvocationContract>,
    device: &'a seismic_native_target::DeviceDescription<T>,
    realizer: Realizer<'a, T, C, ImplementationReconciler<'a, T>>,
    accounting: PreparationBudgetTracker,
    planning_budget: PlanningBudget,
    construction_allowance: crate::candidate_domain::ConstructionAllowance,
    candidates: Vec<PreparedCandidate<T, C::Handle>>,
    coordinates: HashMap<CandidateCoordinate, PreparedCandidateId>,
    universal_admitted: bool,
    optional_native_open: bool,
}

impl<'a, T, C> EvaluationSession<'a, T, C>
where
    T: seismic_native_target::TargetFamily,
    C: NativeCompiler<T>,
{
    pub(crate) fn from_domain(
        domain: CandidateDomain<'a, T>,
        registry: &'a CompilerRegistry<T>,
        compiler: &'a C,
        native_context: &'a C::Context,
        device: &'a seismic_native_target::DeviceDescription<T>,
        preparation_budget: &PreparationBudget,
        planning_budget: &PlanningBudget,
    ) -> Result<Self, PreparationError> {
        if domain.device_identity() != device.identity() {
            return Err(PreparationError::InvalidCandidateDomain(
                "preparation device does not match the domain".into(),
            ));
        }
        static NEXT_PREPARATION: AtomicU64 = AtomicU64::new(1);
        let invocation = std::sync::Arc::new(InvocationContract::compile(&domain));
        let id = NEXT_PREPARATION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .expect("preparation identity space exhausted");
        Ok(Self {
            invocation,
            id,
            domain,
            device,
            realizer: Realizer::new(
                compiler,
                native_context,
                device,
                ImplementationReconciler {
                    target: device,
                    registry,
                },
            ),
            accounting: PreparationBudgetTracker::new(preparation_budget.clone()),
            planning_budget: planning_budget.clone(),
            construction_allowance: crate::candidate_domain::ConstructionAllowance {
                work_units: preparation_budget.construction_work_units,
                wall_time: preparation_budget.construction_wall_time,
            },
            candidates: Vec::new(),
            coordinates: HashMap::new(),
            universal_admitted: false,
            optional_native_open: true,
        })
    }

    /// Create the domain and its native services together. A strategy cannot
    /// substitute a different registry after construction has begun.
    pub fn new(
        entry: seismic_lang::entry::LogicalEntry,
        device: &'a seismic_native_target::DeviceDescription<T>,
        registry: &'a CompilerRegistry<T>,
        compiler: &'a C,
        native_context: &'a C::Context,
        precision: &seismic_lang::precision::PrecisionPolicy,
        preparation_budget: &PreparationBudget,
        planning_budget: &PlanningBudget,
    ) -> Result<Self, PreparationError> {
        let domain = crate::candidate_domain::construct_candidate_domain(
            entry,
            device,
            registry,
            precision,
        )?;
        Self::from_domain(
            domain,
            registry,
            compiler,
            native_context,
            device,
            preparation_budget,
            planning_budget,
        )
    }

    pub fn domain(&self) -> &CandidateDomain<'a, T> {
        &self.domain
    }
    pub(crate) fn domain_mut(&mut self) -> &mut CandidateDomain<'a, T> {
        &mut self.domain
    }
    /// Advance construction in this preparation's domain. The domain itself
    /// cannot be replaced independently of invocation and native ownership.
    pub fn advance(
        &mut self,
        coordinate: &crate::candidate_domain::ConstructionCoordinate,
        allowance: crate::candidate_domain::ConstructionAllowance,
    ) -> crate::candidate_domain::ConstructionAdvance {
        self.domain.advance(coordinate, allowance)
    }

    pub fn construction_allowance(&self) -> crate::candidate_domain::ConstructionAllowance {
        self.construction_allowance
    }
    pub fn search_budget(&self) -> &PlanningBudget {
        &self.planning_budget
    }

    pub fn inspect(
        &self,
        coordinate: &CandidateCoordinate,
    ) -> Result<crate::realization::NativeRequirements, PreparationError> {
        let checked = self
            .domain
            .checked_preparation_candidate(coordinate)
            .map_err(|error| {
                PreparationError::InvalidCandidateDomain(format!(
                    "invalid inspection coordinate: {error:?}"
                ))
            })?;
        let read = self
            .domain
            .read(&checked)
            .expect("candidate belongs to its domain");
        let request = CanonicalRealizationRequest::from_checked_candidate(&read);
        Ok(self.realizer.inspect(read.arena(), &request))
    }

    pub fn native_usage(&self) -> (u64, u64, bool) {
        let (code, metadata) = self.accounting.native_usage();
        (code, metadata, !self.optional_native_open)
    }

    pub fn extend_native_time(&mut self, additional: std::time::Duration) {
        self.accounting.extend_native_time(additional);
        self.optional_native_open = self.accounting.native_attempt_available();
    }

    pub fn realize_checked(
        &mut self,
        coordinate: &CandidateCoordinate,
    ) -> Result<RealizationAdmission, PreparationError> {
        // Stable semantic equality survives independent construction. Arena
        // ownership remains a separate prerequisite, including on cache hits.
        if !self.domain.owns_coordinate(coordinate) {
            return Err(PreparationError::InvalidCandidateDomain(
                "coordinate belongs to another domain owner".into(),
            ));
        }
        if let Some(candidate) = self.coordinates.get(coordinate) {
            return Ok(RealizationAdmission::Ready(*candidate));
        }
        let checked = self
            .domain
            .checked_preparation_candidate(coordinate)
            .map_err(|error| {
                PreparationError::InvalidCandidateDomain(format!(
                    "evaluator requested an invalid coordinate: {error:?}"
                ))
            })?;
        if !self.universal_admitted && !checked.is_universal() {
            return Err(PreparationError::InvalidCandidateDomain(
                "candidate evaluation must realize the universal coordinate first".into(),
            ));
        }
        let realization = {
            let read = self
                .domain
                .read(&checked)
                .expect("candidate belongs to its domain");
            CanonicalRealizationRequest::from_checked_candidate(&read)
        };
        let requirements = self.realizer.inspect(&self.domain.arena(), &realization);
        let needs_formation = !requirements.missing.is_empty();
        if !checked.is_universal() && !self.optional_native_open && needs_formation {
            return Ok(RealizationAdmission::Pending(
                RealizationPending::NativeBudget,
            ));
        }
        let attempt = self.realizer.realize(&mut self.domain.arena_mut(), realization);
        let outcome = match attempt {
            Ok(outcome) => outcome,
            Err(failure) => {
                // Earlier kernels in a multi-kernel request may already be
                // resident even though a later formation failed.
                let within_optional_budget = if checked.is_universal() {
                    self.accounting
                        .record_required_native_artifact(failure.newly_formed)?
                } else {
                    self.accounting.record_native_artifact(failure.newly_formed)
                };
                if !within_optional_budget {
                    self.optional_native_open = false;
                }
                return Err(failure.error);
            }
        };
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
        let within_optional_budget = if checked.is_universal() {
            self.accounting.record_required_native_artifact(metrics)?
        } else {
            self.accounting.record_native_artifact(metrics)
        };
        if !within_optional_budget {
            self.optional_native_open = false;
        }
        let Some(realized) = realized else {
            if checked.is_universal() {
                return Err(PreparationError::UniversalClosure(format!(
                    "native reconciliation rejected the covered implementation: {}",
                    rejection
                        .expect("rejected outcome carries its reason")
                        .reason()
                )));
            }
            return Ok(RealizationAdmission::Rejected(
                rejection.expect("rejected outcome carries its reason"),
            ));
        };

        let (precision, target_domain, constants) = self.domain.numerical_context();
        let implementation = realized.reconciled().clone();
        let fixed = crate::frozen::plan_assignment(&self.domain.arena(), &constants, &implementation);
        let safe_guard = {
            let mut arena = self.domain.arena_mut();
            let guard = arena.all(&[checked.constraint(), implementation.hard_constraints()]);
            arena.partial(guard, &fixed)
        };
        if checked.is_universal() {
            validate_universal_implementation(
                &implementation,
                &mut self.domain.arena_mut(),
                target_domain.predicate().node(),
                &constants,
            )?;
        }
        let compiled = {
            let arena = self.domain.arena();
            let context = CandidateContext {
                invocation: self.invocation.clone(),
                device: self.device.identity().clone(),
                arena: &arena,
                constants,
            };
            crate::executable::compile_variant(
                freeze(&context, implementation.clone(), fixed, safe_guard),
            )
        };
        let admission_policy = if checked.is_universal() { &seismic_lang::precision::PrecisionPolicy::Exact } else { &precision };
        let Some(retained) = compiled.accept_numerically(&mut self.domain.arena_mut(), admission_policy)
        else {
            return Ok(RealizationAdmission::Pending(RealizationPending::NumericalAnalysis));
        };
        let reference = PreparedCandidateId {
            preparation: self.id,
            index: self.candidates.len(),
        };
        let executable =
            crate::executable::materialize_variant(&retained, self.realizer.registry(), reference);
        let fixed = compiled.fixed().clone();
        self.universal_admitted |= checked.is_universal();
        self.coordinates
            .entry(coordinate.clone())
            .or_insert(reference);
        self.candidates.push(PreparedCandidate {
            universal: checked.is_universal(),
            fixed,
            retained,
            executable,
        });
        Ok(RealizationAdmission::Ready(reference))
    }

    fn candidate(
        &self,
        id: PreparedCandidateId,
    ) -> Result<&PreparedCandidate<T, C::Handle>, PreparationError> {
        if id.preparation != self.id {
            return Err(PreparationError::InvalidCandidateDomain(
                "candidate belongs to a different preparation".into(),
            ));
        }
        self.candidates.get(id.index).ok_or_else(|| {
            PreparationError::InvalidCandidateDomain("unknown prepared candidate".into())
        })
    }

    pub fn executable(
        &self,
        id: PreparedCandidateId,
    ) -> Result<&ExecutableVariant<T, C::Handle>, PreparationError> {
        Ok(&self.candidate(id)?.executable)
    }

    pub fn is_admitted(
        &self,
        id: PreparedCandidateId,
        values: &seismic_lang::expr::compiled::InvocationValues,
    ) -> Result<bool, PreparationError> {
        Ok(self
            .candidate(id)?
            .retained
            .guard()
            .evaluate(values)
            .expect("closed numerical guard"))
    }

    /// Compile candidate operands into the evaluator's selection, then retain
    /// exactly that selector's operand list. This never truncates the policy.
    pub fn policy<P>(
        &self,
        retained: NonEmpty<(PreparedCandidateId, P)>,
        build_selection: impl FnOnce(&ExprArena, PreparedPortfolio<P>) -> SelectionFunction,
    ) -> Result<SelectionPolicy, PreparationError> {
        let mut selections = Vec::new();
        for (id, payload) in retained.into_vec() {
            let candidate = self.candidate(id)?;
            selections.push(PreparedSelection {
                candidate: id,
                payload,
                guard: candidate.retained.guard().clone(),
                fixed: candidate.fixed.clone(),
            });
        }
        let selection = build_selection(
            &self.domain.arena(),
            PreparedPortfolio {
                selections: NonEmpty::new(selections).unwrap(),
            },
        );
        // The selector's operands define its ordinal space. Resolve those
        // same operands against preparation-owned retained candidates; no parallel list
        // supplied by a callback can change an ordinal's meaning.
        if !self
            .candidate(
                *selection
                    .candidate_operands()
                    .first()
                    .expect("a selector has a general operand"),
            )?
            .universal
        {
            return Err(PreparationError::InvalidCandidateDomain(
                "policy must begin with the universal candidate".into(),
            ));
        }
        let mut seen = HashSet::new();
        let mut bytes = 0u64;
        for id in selection.candidate_operands().iter().copied() {
            let candidate = self.candidate(id)?;
            if !seen.insert(id) {
                return Err(PreparationError::InvalidCandidateDomain(
                    "policy contains a duplicate candidate".into(),
                ));
            }
            bytes = bytes.saturating_add(candidate.retained.retained_metadata_bytes());
        }
        bytes = bytes.saturating_add(selection.retained_metadata_bytes());
        let required = selection.candidate_operands().len() == 1;
        let limit = if required {
            self.planning_budget.required_retained_metadata_bytes
        } else {
            self.planning_budget.retained_metadata_bytes
        };
        if bytes > limit {
            return Err(PreparationError::Planning(
                crate::planning::PlanningError::BudgetExceeded {
                    resource: if required {
                        crate::planning::PlanningBudgetResource::RequiredRetainedMetadata
                    } else {
                        crate::planning::PlanningBudgetResource::RetainedMetadata
                    },
                    required: bytes,
                    limit,
                },
            ));
        }
        if !required
            && selection.candidate_operands().len() as u64 > self.planning_budget.executable_variants
        {
            return Err(PreparationError::Planning(
                crate::planning::PlanningError::BudgetExceeded {
                    resource: crate::planning::PlanningBudgetResource::ExecutableVariants,
                    required: selection.candidate_operands().len() as u64,
                    limit: self.planning_budget.executable_variants,
                },
            ));
        }
        Ok(SelectionPolicy {
            selection_function: selection,
        })
    }

    /// Reports the native storage of this preparation's exact policy operands.
    pub fn retained_native_sizes(
        &self,
        policy: &SelectionPolicy,
    ) -> Result<(u64, u64), PreparationError> {
        let candidates = policy
            .candidate_ids()
            .iter()
            .map(|id| self.candidate(*id))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self.realizer.registry().retained_sizes(
            candidates
                .iter()
                .flat_map(|candidate| candidate.retained.native_instances()),
        ))
    }

    /// Packages existing executable bodies and handles. Preparation remains usable.
    pub(crate) fn finalize(
        &self,
        policy: SelectionPolicy,
        evaluation: EvaluationIdentity,
        mut report: PlanningReport,
    ) -> Result<PreparedKernel<T, C::Handle>, PreparationError> {
        if evaluation.device() != self.device.identity() {
            return Err(PreparationError::InvalidCandidateDomain(
                "result evaluation belongs to a different device".into(),
            ));
        }
        let mut variants = Vec::new();
        let mut bytes = policy.selection_function.retained_metadata_bytes();
        for id in policy.candidate_ids() {
            let retained = &self.candidate(*id)?.retained;
            bytes = bytes.saturating_add(retained.retained_metadata_bytes());
            variants.push(crate::executable::materialize_variant(
                retained,
                self.realizer.registry(),
                *id,
            ));
        }
        report.budget.executable_variants = variants.len() as u64;
        report.budget.retained_metadata_bytes = bytes;
        Ok(PreparedKernel::prepare(
            self.domain.semantic_event_manifest().clone(),
            self.device.identity().clone(),
            evaluation,
            self.invocation.clone(),
            policy.selection_function,
            NonEmpty::new(variants).unwrap(),
            report,
        ))
    }
}

#[cfg(test)]
pub(crate) mod boundary_tests {
    use super::*;
    use crate::evaluation::EvaluationProvenance;
    use crate::planning::{OptimizationCompletion, PlanningBudgetReport};
    use crate::realization::demand_driven_tests::{
        device, registry, CountingCompiler, FakeTarget,
    };
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};
    use seismic_lang::entry::ElementBindings;
    use seismic_lang::expr::compiled::InvocationValues;
    use seismic_lang::precision::PrecisionPolicy;

    fn domain<'ctx>(
        device: &'ctx seismic_native_target::DeviceDescription<FakeTarget>,
        registry: &'ctx CompilerRegistry<FakeTarget>,
    ) -> CandidateDomain<'ctx, FakeTarget> {
        domain_source(device, registry, "fn probe(x: f32) -> f32:\n    return x\n")
    }

    fn domain_source<'ctx>(
        device: &'ctx seismic_native_target::DeviceDescription<FakeTarget>,
        registry: &'ctx CompilerRegistry<FakeTarget>,
        source: &str,
    ) -> CandidateDomain<'ctx, FakeTarget> {
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "session-fixture.seismic".into(),
            text: source.into(),
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
        )
        .unwrap()
    }

    pub(crate) fn domain_with_optional<'ctx>(
        device: &'ctx seismic_native_target::DeviceDescription<FakeTarget>,
        registry: &'ctx CompilerRegistry<FakeTarget>,
    ) -> (CandidateDomain<'ctx, FakeTarget>, CandidateCoordinate) {
        authored_domain(device,registry,"fn probe(x: f32) -> f32:\n    return x\n\nlower probe(x: f32) -> f32 for cpu:\n    return x\n", PrecisionPolicy::Exact)
    }

    fn authored_domain<'ctx>(device:&'ctx seismic_native_target::DeviceDescription<FakeTarget>,registry:&'ctx CompilerRegistry<FakeTarget>,source:&str,precision:PrecisionPolicy) -> (CandidateDomain<'ctx,FakeTarget>,CandidateCoordinate) {
        use crate::candidate_domain::{BodyMapping,ConstructionCoordinate,ConstructionAllowance,Materialization};
        let module=check_source(SourceSet::new(vec![SourceFile{path:"session-outcome.seismic".into(),text:source.into()}])).unwrap();
        let entry=module.entry(module.entry_named("probe").unwrap(),&ElementBindings::default()).unwrap();
        let mut domain=crate::candidate_domain::construct_candidate_domain(entry,device,registry,&precision).unwrap();
        let selection=domain.root_selections().into_iter().find(|body|body.mapping==BodyMapping::Authored).unwrap();
        let Materialization::Ready(construction)=domain.advance(&ConstructionCoordinate::root(selection),ConstructionAllowance{work_units:100_000,wall_time:std::time::Duration::from_secs(30)}).state else {panic!("authored construction did not finish")};
        let coordinate=domain.canonicalize(domain.proposal(construction,Vec::new())).unwrap();
        (domain,coordinate)
    }

    fn report() -> PlanningReport {
        PlanningReport {
            optimization: OptimizationCompletion::Complete,
            budget: PlanningBudgetReport::default(),
        }
    }

    fn identity(device: &seismic_native_target::DeviceDescription<FakeTarget>) -> EvaluationIdentity {
        let provenance = EvaluationProvenance::new([4; 32], [5; 32]);
        EvaluationIdentity::new(device.identity().clone(), provenance)
    }

    fn general(
        session: &mut EvaluationSession<'_, FakeTarget, CountingCompiler>,
    ) -> PreparedCandidateId {
        let coordinate = session
            .domain()
            .canonicalize(session.domain().universal_proposal())
            .unwrap();
        let RealizationAdmission::Ready(candidate) = session.realize_checked(&coordinate).unwrap()
        else {
            panic!("general candidate rejected")
        };
        candidate
    }

    fn policy(
        session: &EvaluationSession<'_, FakeTarget, CountingCompiler>,
        candidates: Vec<PreparedCandidateId>,
    ) -> Result<SelectionPolicy, PreparationError> {
        session.policy(
            NonEmpty::new(candidates.into_iter().map(|id| (id, ())).collect()).unwrap(),
            |_, portfolio| SelectionFunction::ordered_decision(portfolio, Vec::new()),
        )
    }

    #[test]
    fn retained_bodies_follow_selector_operands_and_reject_foreign_operands() {
        let device = device();
        let registry = registry();
        let compiler = CountingCompiler::new();
        let (first_domain, optional) = domain_with_optional(&device, &registry);
        let mut first = EvaluationSession::from_domain(
            first_domain,
            &registry,
            &compiler,
            &(),
            &device,
            &PreparationBudget::default(),
            &PlanningBudget::default(),
        )
        .unwrap();
        let general_id = general(&mut first);
        let RealizationAdmission::Ready(optional_id) = first.realize_checked(&optional).unwrap()
        else {
            panic!("optional candidate")
        };
        let selector = policy(&first, vec![general_id]).unwrap().selection_function;
        let returned = first
            .policy(
                NonEmpty::new(vec![(general_id, ()), (optional_id, ())]).unwrap(),
                |_, _| selector,
            )
            .unwrap();
        assert_eq!(
            returned.candidate_ids().len(),
            1,
            "retained ordinals come from the returned selector's actual operands"
        );

        let mut second = EvaluationSession::from_domain(
            domain(&device, &registry),
            &registry,
            &compiler,
            &(),
            &device,
            &PreparationBudget::default(),
            &PlanningBudget::default(),
        )
        .unwrap();
        let second_general = general(&mut second);
        assert!(matches!(
            second.policy(
                NonEmpty::new(vec![(second_general, ())]).unwrap(),
                |_, _| returned.selection_function,
            ),
            Err(PreparationError::InvalidCandidateDomain(_))
        ));
    }

    #[test]
    fn semantic_coordinate_equality_does_not_bypass_session_ownership() {
        let device = device();
        let registry = registry();
        let compiler = CountingCompiler::new();
        let own = domain(&device, &registry);
        let other = domain(&device, &registry);
        let foreign = other.canonicalize(other.universal_proposal()).unwrap();
        assert_eq!(own.canonicalize(own.universal_proposal()).unwrap(), foreign);
        let mut session = EvaluationSession::from_domain(
            own,
            &registry,
            &compiler,
            &(),
            &device,
            &PreparationBudget::default(),
            &PlanningBudget::default(),
        )
        .unwrap();
        general(&mut session);
        assert!(matches!(
            session.realize_checked(&foreign),
            Err(PreparationError::InvalidCandidateDomain(_))
        ));
    }

    #[test]
    fn complete_helper_products_do_not_allocate_an_intermediate_tensor() {
        let mut description = crate::realization::demand_driven_tests::device_parts();
        description
            .dtypes
            .representations
            .insert(seismic_lang::registry::dense(
                seismic_lang::types::DType::F32,
            ));
        description.limits.max_allocation_bytes = 1024;
        description.limits.max_allocation_alignment = 16;
        description.limits.max_index_bits = 64;
        description.limits.max_bindings = 64;
        description.limits.max_argument_bytes = 4096;
        description.limits.max_workgroup_bytes = 4096;
        description.limits.max_grid = [1024; 3];
        let device = seismic_native_target::DeviceDescription::new(description).unwrap();
        let registry = registry();
        let domain = domain_source(&device, &registry,
            "fn twice(x: &tensor[4] f32) -> tensor[4] f32:\n    return x + x\n\nfn total(x: &tensor[4] f32) -> f32:\n    return reduce(x, 0, sum)\n\nfn probe(x: &tensor[4] f32) -> f32:\n    return total(twice(x))\n");
        let parts = domain.into_parts();
        assert_eq!(
            parts
                .materialized
                .first()
                .family
                .global_allocations()
                .allocations()
                .len(),
            1,
            "the ordinary helper path retains only the external input allocation"
        );
    }

    #[test]
    fn derived_outcome_drives_policy_and_explanation_without_empirical_promotion() {
        let device=device(); let registry=registry(); let compiler=CountingCompiler::new();
        for precision in [PrecisionPolicy::Exact, PrecisionPolicy::bounded(seismic_lang::precision::Tolerance::EXACT), PrecisionPolicy::Unconstrained] {
            let (domain,optional)=authored_domain(&device,&registry,"fn probe(x: f32) -> f32:\n    return x\n\nlower probe(x: f32) -> f32 for cpu:\n    return 3.0\n",precision.clone());
            let mut session=EvaluationSession::from_domain(domain,&registry,&compiler,&(),&device,&PreparationBudget::default(),&PlanningBudget::default()).unwrap();
            let universal=general(&mut session);
            let result=session.realize_checked(&optional).unwrap();
            if matches!(precision,PrecisionPolicy::Unconstrained) {
                let RealizationAdmission::Ready(id)=result else {panic!("floating-only difference must be admitted under Unconstrained")};
                let prepared=session.finalize(policy(&session,vec![universal,id]).unwrap(),identity(&device),report()).unwrap();
                assert!(matches!(prepared.variants().iter().nth(1).unwrap().numerical().regions[0].basis,crate::numerics::NumericalBasis::Unknown));
            } else {
                assert_eq!(result,RealizationAdmission::Pending(RealizationPending::NumericalAnalysis));
                assert_eq!(session.realize_checked(&optional).unwrap(),result,"reobservation cannot enlarge applicability");
            }
        }
    }

    #[test]
    fn reflected_descriptor_rejection_restricts_universal_applicability() {
        let device = device();
        let registry = registry();
        let compiler = CountingCompiler::new();
        compiler
            .reject_launch_descriptor
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let mut session = EvaluationSession::from_domain(
            domain_source(
                &device,
                &registry,
                "fn probe(x: f32) -> f32:\n    return x + x\n",
            ),
            &registry,
            &compiler,
            &(),
            &device,
            &PreparationBudget::default(),
            &PlanningBudget::default(),
        )
        .unwrap();
        let coordinate = session
            .domain()
            .canonicalize(session.domain().universal_proposal())
            .unwrap();
        assert!(matches!(
            session.realize_checked(&coordinate),
            Err(PreparationError::UniversalClosure(_))
        ));
    }

    #[test]
    fn executable_candidates_exist_before_publication_and_results_survive_preparation() {
        let device = device();
        let registry = registry();
        let compiler = CountingCompiler::new();
        let (domain, optional) = domain_with_optional(&device, &registry);
        let mut session = EvaluationSession::from_domain(
            domain,
            &registry,
            &compiler,
            &(),
            &device,
            &PreparationBudget::default(),
            &PlanningBudget::default(),
        )
        .unwrap();
        let general = general(&mut session);
        let first = session
            .finalize(
                policy(&session, vec![general]).unwrap(),
                identity(&device),
                report(),
            )
            .unwrap();
        let coordinate = session
            .domain()
            .canonicalize(session.domain().universal_proposal())
            .unwrap();
        let count = compiler.form_count();
        assert_eq!(
            session.realize_checked(&coordinate).unwrap(),
            RealizationAdmission::Ready(general)
        );
        assert_eq!(compiler.form_count(), count);
        assert!(session
            .executable(general)
            .unwrap()
            .guard()
            .evaluate(&InvocationValues::new())
            .unwrap());
        assert!(session.domain().check(&optional).is_ok());
        let RealizationAdmission::Ready(optional) = session.realize_checked(&optional).unwrap()
        else {
            panic!("optional rejected")
        };
        let second = session
            .finalize(
                policy(&session, vec![general, optional]).unwrap(),
                identity(&device),
                report(),
            )
            .unwrap();
        drop(session);
        assert_eq!(first.variants().len(), 1);
        assert_eq!(second.variants().len(), 2);
        assert_eq!(first.select(&InvocationValues::new()).as_usize(), 0);
    }

    #[test]
    fn finalization_resolves_exact_selector_operands_without_reformation() {
        let device = device();
        let registry = registry();
        let compiler = CountingCompiler::new();
        let (domain, optional) = domain_with_optional(&device, &registry);
        let mut session = EvaluationSession::from_domain(
            domain,
            &registry,
            &compiler,
            &(),
            &device,
            &PreparationBudget::default(),
            &PlanningBudget::default(),
        )
        .unwrap();
        let universal = general(&mut session);
        let RealizationAdmission::Ready(optional) = session.realize_checked(&optional).unwrap()
        else {
            panic!("optional formation should be ready")
        };
        let formed = compiler.form_count();
        let selected = policy(&session, vec![universal, optional]).unwrap();
        let prepared = session
            .finalize(selected, identity(&device), report())
            .unwrap();
        assert_eq!(compiler.form_count(), formed);
        assert_eq!(
            prepared
                .variants()
                .iter()
                .map(|variant| variant.prepared_candidate_id())
                .collect::<Vec<_>>(),
            vec![universal, optional]
        );
    }

    #[test]
    fn retention_is_rejected_without_truncating_or_consuming_preparation() {
        let device = device();
        let registry = registry();
        let compiler = CountingCompiler::new();
        let (domain, optional) = domain_with_optional(&device, &registry);
        let budget = PlanningBudget {
            retained_metadata_bytes: 0,
            ..PlanningBudget::default()
        };
        let mut session = EvaluationSession::from_domain(
            domain,
            &registry,
            &compiler,
            &(),
            &device,
            &PreparationBudget::default(),
            &budget,
        )
        .unwrap();
        let general = general(&mut session);
        let RealizationAdmission::Ready(optional) = session.realize_checked(&optional).unwrap()
        else {
            panic!("optional rejected")
        };
        assert!(matches!(
            policy(&session, vec![general, optional]),
            Err(PreparationError::Planning(_))
        ));
        assert!(policy(&session, vec![general]).is_ok());
    }

    #[test]
    fn foreign_and_duplicate_candidates_cannot_enter_policy() {
        let device = device();
        let registry = registry();
        let compiler = CountingCompiler::new();
        let mut first = EvaluationSession::from_domain(
            domain(&device, &registry),
            &registry,
            &compiler,
            &(),
            &device,
            &PreparationBudget::default(),
            &PlanningBudget::default(),
        )
        .unwrap();
        let mut second = EvaluationSession::from_domain(
            domain(&device, &registry),
            &registry,
            &compiler,
            &(),
            &device,
            &PreparationBudget::default(),
            &PlanningBudget::default(),
        )
        .unwrap();
        let a = general(&mut first);
        let b = general(&mut second);
        assert!(policy(&first, vec![a, a]).is_err());
        assert!(policy(&first, vec![a, b]).is_err());
        let foreign_policy = policy(&first, vec![a]).unwrap();
        assert!(second.retained_native_sizes(&foreign_policy).is_err());
        assert!(second
            .finalize(foreign_policy, identity(&device), report())
            .is_err());
    }

    #[test]
    fn closed_native_budget_still_reuses_resident_candidates() {
        let device = device();
        let registry = registry();
        let compiler = CountingCompiler::new();
        let (domain, optional) = domain_with_optional(&device, &registry);
        let budget = PreparationBudget {
            native_code_bytes: 0,
            ..PreparationBudget::default()
        };
        let mut session = EvaluationSession::from_domain(
            domain,
            &registry,
            &compiler,
            &(),
            &device,
            &budget,
            &PlanningBudget::default(),
        )
        .unwrap();
        let general = general(&mut session);
        let coordinate = session
            .domain()
            .canonicalize(session.domain().universal_proposal())
            .unwrap();
        assert_eq!(
            session.realize_checked(&coordinate).unwrap(),
            RealizationAdmission::Ready(general)
        );
        let count = compiler.form_count();
        assert_eq!(
            session.realize_checked(&optional).unwrap(),
            RealizationAdmission::Pending(RealizationPending::NativeBudget)
        );
        assert_eq!(compiler.form_count(), count);
    }
    #[test]
    fn unresolved_numerics_remain_structural_without_exposing_an_executable() {
        let device=device(); let registry=registry(); let compiler=CountingCompiler::new();
        let (domain,optional)=authored_domain(&device,&registry,"fn probe(x: u32) -> u32:\n    return x\n\nlower probe(x: u32) -> u32 for cpu:\n    return 3\n",PrecisionPolicy::Unconstrained);
        let mut session=EvaluationSession::from_domain(domain,&registry,&compiler,&(),&device,&PreparationBudget::default(),&PlanningBudget::default()).unwrap();
        let universal=general(&mut session);
        assert_eq!(session.realize_checked(&optional).unwrap(),RealizationAdmission::Pending(RealizationPending::NumericalAnalysis));
        let formed=compiler.form_count();
        assert_eq!(session.realize_checked(&optional).unwrap(),RealizationAdmission::Pending(RealizationPending::NumericalAnalysis));
        assert_eq!(compiler.form_count(),formed,"pending reasoning reuses native artifacts");
        assert!(!session.inspect(&optional).unwrap().rejected,"unknown is not native rejection");
        assert_eq!(session.candidates.len(),1,"unresolved candidates have no executable handle");
        policy(&session,vec![universal]).unwrap();
    }
}
