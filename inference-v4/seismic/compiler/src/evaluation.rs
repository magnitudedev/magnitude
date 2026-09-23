//! Method-independent evaluation over preparation-owned candidates.
//! Analytical models are private strategy data, not a domain transition.

pub(crate) mod construction;

use crate::candidate_domain::{CandidateDomain, DomainCandidate};
use crate::refinement::{ChoiceDeclaration, ConstructedCandidate, ConstructedCandidateIdentity};
use crate::target::ExecutionProfile;
use seismic_estimator::FactProvenance;
use seismic_lang::expr::{AnyExpr, ExprArena, SymbolKind};
use seismic_target::{DeviceDescription, DeviceDescriptionIdentity};
use sha2::Digest;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EvaluationIdentity {
    device: DeviceDescriptionIdentity,
    protocol: [u8; 32],
    evidence_fingerprint: [u8; 32],
}

impl EvaluationIdentity {
    pub fn new(device: DeviceDescriptionIdentity, provenance: EvaluationProvenance) -> Self {
        Self {
            device,
            protocol: provenance.protocol,
            evidence_fingerprint: provenance.evidence_fingerprint,
        }
    }

    pub fn device(&self) -> &DeviceDescriptionIdentity {
        &self.device
    }
    pub fn protocol(&self) -> [u8; 32] {
        self.protocol
    }
    pub fn evidence_fingerprint(&self) -> [u8; 32] {
        self.evidence_fingerprint
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EvaluationProvenance {
    protocol: [u8; 32],
    evidence_fingerprint: [u8; 32],
}

impl EvaluationProvenance {
    pub fn new(protocol: [u8; 32], evidence_fingerprint: [u8; 32]) -> Self {
        Self {
            protocol,
            evidence_fingerprint,
        }
    }
}

/// Read-only executable projection granted to evaluators. Native handles,
/// reflected artifacts, and execution services are intentionally absent.
pub struct TargetClosedExecutableView<'a, K: seismic_ir::target::PhysicalDialect> {
    identity: &'a ConstructedCandidateIdentity,
    execution: seismic_ir::execution::ClosedExecutionView<'a, K>,
    choices: &'a [ChoiceDeclaration],
    constraints: &'a [crate::candidate_domain::DomainConstraint],
}

impl<K: seismic_ir::target::PhysicalDialect> Copy for TargetClosedExecutableView<'_, K> {}

impl<K: seismic_ir::target::PhysicalDialect> Clone for TargetClosedExecutableView<'_, K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T: seismic_target::TargetFamily> TargetClosedExecutableView<'a, T> {
    pub(crate) fn new(
        family: &'a ConstructedCandidate<T>,
        constraints: &'a [crate::candidate_domain::DomainConstraint],
    ) -> Self {
        Self {
            identity: family.identity(),
            execution: family.executable().view(),
            choices: family.choices(),
            constraints,
        }
    }
}

impl<'a, K: seismic_ir::target::PhysicalDialect> TargetClosedExecutableView<'a, K> {
    pub fn identity(&self) -> &'a ConstructedCandidateIdentity {
        self.identity
    }
    pub fn execution(&self) -> seismic_ir::execution::ClosedExecutionView<'a, K> {
        self.execution
    }
    pub fn choices(&self) -> &'a [ChoiceDeclaration] {
        self.choices
    }
    pub fn constraints(&self) -> &'a [crate::candidate_domain::DomainConstraint] {
        self.constraints
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PerformanceQuantity {
    Latency,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MeasurementEndpoint {
    SubmissionThroughCompletion,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PerformanceObjective {
    pub quantity: PerformanceQuantity,
    pub endpoint: MeasurementEndpoint,
}

impl PerformanceObjective {
    pub const LATENCY: Self = Self {
        quantity: PerformanceQuantity::Latency,
        endpoint: MeasurementEndpoint::SubmissionThroughCompletion,
    };
}

#[derive(Clone, Debug)]
pub struct CandidatePerformanceModel {
    objective: PerformanceObjective,
    estimate: seismic_lang::expr::DurationExpr,
    uncertainty: CorrelatedUncertainty,
}

#[derive(Clone, Debug)]
pub struct CorrelatedUncertainty {
    pub maximum_relative_error_basis_points: u16,
    contributions: Vec<UncertaintyContribution>,
}

#[derive(Clone, Debug)]
pub struct UncertaintyContribution {
    region: [u8; 32],
    correlation: [u8; 32],
    guards: Vec<seismic_lang::expr::BoolExpr>,
    estimate: seismic_lang::expr::DurationExpr,
    evidence: FactProvenance,
}

impl CorrelatedUncertainty {
    pub fn new(
        maximum_relative_error_basis_points: u16,
        contributions: Vec<UncertaintyContribution>,
    ) -> Self {
        Self {
            maximum_relative_error_basis_points,
            contributions,
        }
    }
    pub fn contributions(&self) -> &[UncertaintyContribution] {
        &self.contributions
    }
}

impl UncertaintyContribution {
    pub fn new(
        region: [u8; 32],
        correlation: [u8; 32],
        guards: Vec<seismic_lang::expr::BoolExpr>,
        estimate: seismic_lang::expr::DurationExpr,
        evidence: FactProvenance,
    ) -> Self {
        Self {
            region,
            correlation,
            guards,
            estimate,
            evidence,
        }
    }
    pub fn region(&self) -> [u8; 32] {
        self.region
    }
    pub fn correlation(&self) -> [u8; 32] {
        self.correlation
    }
    pub fn guards(&self) -> &[seismic_lang::expr::BoolExpr] {
        &self.guards
    }
    pub fn estimate(&self) -> seismic_lang::expr::DurationExpr {
        self.estimate
    }
    pub fn evidence(&self) -> &FactProvenance {
        &self.evidence
    }
}

impl CandidatePerformanceModel {
    pub fn new(
        objective: PerformanceObjective,
        estimate: seismic_lang::expr::DurationExpr,
        uncertainty: CorrelatedUncertainty,
    ) -> Self {
        Self {
            objective,
            estimate,
            uncertainty,
        }
    }
    pub fn objective(&self) -> PerformanceObjective {
        self.objective
    }
    pub fn estimate(&self) -> seismic_lang::expr::DurationExpr {
        self.estimate
    }
    pub fn uncertainty(&self) -> &CorrelatedUncertainty {
        &self.uncertainty
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvaluationError {
    DeviceMismatch,
    InvalidModel(EvaluationModelError),
    ModelLimitation(seismic_estimator::ModelLimitation),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvaluationModelError {
    ObjectiveMismatch,
    UndeclaredSymbol,
}

#[derive(Debug)]
pub enum EvaluationMapError<E> {
    Evaluator(E),
    InvalidModel(EvaluationModelError),
}

pub use crate::evaluation_session::{
    EvaluationSession, PreparedCandidateId, PreparedPortfolio, RealizationAdmission,
    RealizationPending,
};
pub use crate::realization::{CandidateRejection, NativeArtifactRequestKey, NativeRequirements};

/// Common evaluator contract. Search, realization requests, retention, and
/// the closed invocation selection rule are one responsibility. Both
/// analytical and feedback implementations return this same policy type.
pub trait CandidateEvaluator<T, C>
where
    T: seismic_target::TargetFamily,
    C: seismic_target::NativeCompiler<T>,
{
    fn evaluate(
        &mut self,
        session: &mut EvaluationSession<'_, T, C>,
    ) -> Result<
        (
            crate::planning::SelectionPolicy,
            EvaluationIdentity,
            crate::planning::PlanningReport,
        ),
        crate::errors::PreparationError,
    >;
}

/// Immutable analytical closure for one opened target. Assembly derives the
/// exact required service vocabulary from the same pure definition whose
/// operation transfer is bound into the model, then seals device, observations,
/// and model together.
#[derive(Clone)]
pub struct AnalyticalEvaluationContext<T: seismic_target::TargetFamily> {
    device: Arc<DeviceDescription<T>>,
    profile: Arc<ExecutionProfile<T>>,
    model: Arc<dyn seismic_estimator::ExecutionModel<T> + Send + Sync>,
    model_revision: &'static str,
}

struct BoundAnalyticalModel<
    T: seismic_target::TargetFamily,
    D: seismic_estimator::AnalyticalModelDefinition<T>,
> {
    device: Arc<DeviceDescription<T>>,
    profile: Arc<ExecutionProfile<T>>,
    definition: D,
    services: Vec<(D::Service, seismic_estimator::AnalyticalServiceState)>,
}

impl<T, D> seismic_estimator::ServiceModel for BoundAnalyticalModel<T, D>
where
    T: seismic_target::TargetFamily,
    D: seismic_estimator::AnalyticalModelDefinition<T>,
{
    fn service(
        &self,
        service: seismic_estimator::ServiceClassId,
    ) -> &seismic_estimator::ServiceDefinition {
        self.profile.service_by_class(service)
    }

    fn maximum_relative_error_basis_points(&self) -> u16 {
        self.profile
            .composition_qualification()
            .maximum_relative_error_basis_points
    }
}

impl<T, D> seismic_estimator::ExecutionModel<T> for BoundAnalyticalModel<T, D>
where
    T: seismic_target::TargetFamily,
    D: seismic_estimator::AnalyticalModelDefinition<T>,
{
    fn emission_layout(
        &self,
        kernel: &seismic_ir::kernel::Kernel<T>,
    ) -> seismic_ir::target::KernelEmissionLayout {
        self.device.kernel_emission_layout(kernel)
    }

    fn operation_cost(
        &self,
        arena: &mut ExprArena,
        kernel: &seismic_ir::kernel::Kernel<T>,
        emission: &seismic_ir::target::KernelEmissionLayout,
        launch: &seismic_ir::schedule::Launch<T>,
        locals: &seismic_ir::storage::LaunchLocalLayout,
        op: seismic_ir::kernel::ops::ClosedOpView<'_, T>,
    ) -> Result<seismic_estimator::OperationCost, seismic_estimator::ModelLimitation> {
        self.definition
            .operation_cost(
                self.device.facts(),
                self.device.intrinsics(),
                arena,
                kernel,
                emission,
                launch,
                locals,
                op,
            )
            .map(|cost| cost.map_services(|service| {
                let state = self
                    .services
                    .iter()
                    .find(|(candidate, _)| *candidate == service)
                    .map(|(_, state)| *state)
                    .unwrap_or_else(|| {
                        panic!("analytical service vocabulary omitted an emitted variant")
                    });
                if !matches!(state, seismic_estimator::AnalyticalServiceState::Available) {
                    panic!("target-closed analytical operation emitted an unavailable service")
                }
                seismic_estimator::ServiceClassId::new(
                    seismic_estimator::AnalyticalService::stable_name(service),
                )
            }))
    }
}

impl<T: seismic_target::TargetFamily> AnalyticalEvaluationContext<T> {
    pub fn assemble<D>(
        parts: crate::target::ExecutionProfileParts<T>,
        definition: D,
    ) -> Result<Self, crate::errors::TargetError>
    where
        D: seismic_estimator::AnalyticalModelDefinition<T>,
    {
        let model_revision = definition.model_revision();
        if model_revision.is_empty() {
            return Err(crate::errors::TargetError::InvalidExecutionProfile(
                "analytical model revision is empty".into(),
            ));
        }
        let mut service_names = std::collections::BTreeSet::new();
        for service in <seismic_estimator::CoreService as seismic_estimator::AnalyticalService>::ALL
        {
            let name = seismic_estimator::AnalyticalService::stable_name(*service);
            if name.is_empty() || !service_names.insert(name) {
                return Err(crate::errors::TargetError::InvalidExecutionProfile(
                    format!("empty or duplicate core analytical service identity `{name}`"),
                ));
            }
        }
        for service in <D::Service as seismic_estimator::AnalyticalService>::ALL {
            let name = seismic_estimator::AnalyticalService::stable_name(*service);
            if name.is_empty() || !service_names.insert(name) {
                return Err(crate::errors::TargetError::InvalidExecutionProfile(format!(
                    "backend analytical service identity `{name}` is empty, duplicated, or collides with a core service"
                )));
            }
        }
        let device = parts.device().clone();
        let services: Vec<_> = <D::Service as seismic_estimator::AnalyticalService>::ALL
            .iter()
            .copied()
            .map(|service| {
                let state = definition.service_state(device.facts(), device.intrinsics(), service);
                (service, state)
            })
            .collect();
        let mut required: std::collections::BTreeSet<_> =
            <seismic_estimator::CoreService as seismic_estimator::AnalyticalService>::ALL
                .iter()
                .copied()
                .map(|service| {
                    seismic_estimator::ServiceClassId::new(
                        seismic_estimator::AnalyticalService::stable_name(service),
                    )
                })
                .collect();
        required.extend(services.iter().filter_map(|(service, state)| {
            matches!(state, seismic_estimator::AnalyticalServiceState::Available).then(|| {
                seismic_estimator::ServiceClassId::new(
                    seismic_estimator::AnalyticalService::stable_name(*service),
                )
            })
        }));
        let profile = Arc::new(ExecutionProfile::assemble(required, parts)?);
        let model = Arc::new(BoundAnalyticalModel {
            device: device.clone(),
            profile: profile.clone(),
            definition,
            services,
        });
        Ok(Self {
            device,
            profile,
            model,
            model_revision,
        })
    }

    pub fn device(&self) -> &DeviceDescription<T> {
        &self.device
    }

    pub fn device_arc(&self) -> Arc<DeviceDescription<T>> {
        self.device.clone()
    }

    pub fn is_bound_to(&self, device: &Arc<DeviceDescription<T>>) -> bool {
        Arc::ptr_eq(&self.device, device)
    }
}

impl<T: seismic_target::TargetFamily> std::fmt::Debug for AnalyticalEvaluationContext<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnalyticalEvaluationContext")
            .field("device", &self.device.identity())
            .field("profile", &self.profile.identity())
            .finish_non_exhaustive()
    }
}

/// Analytical strategy over one coherently assembled context. Candidate
/// construction and native admission remain supplied by the shared session.
pub struct AnalyticalEvaluator<'a, T: seismic_target::TargetFamily> {
    context: &'a AnalyticalEvaluationContext<T>,
}

impl<'a, T: seismic_target::TargetFamily> AnalyticalEvaluator<'a, T> {
    pub fn new(context: &'a AnalyticalEvaluationContext<T>) -> Self {
        Self { context }
    }
}

impl<T: seismic_target::TargetFamily> AnalyticalEvaluator<'_, T> {
    pub(crate) fn evaluate_domain(
        &self,
        domain: &mut CandidateDomain<'_, T>,
    ) -> Result<AnalyticalDomainModel<T>, EvaluationError> {
        if domain.device_identity() != self.context.device.identity() {
            return Err(EvaluationError::DeviceMismatch);
        }
        let mut protocol = sha2::Sha256::new();
        fn text(digest: &mut sha2::Sha256, tag: &'static [u8], value: &str) {
            digest.update((tag.len() as u64).to_le_bytes());
            digest.update(tag);
            digest.update((value.len() as u64).to_le_bytes());
            digest.update(value.as_bytes());
        }
        text(
            &mut protocol,
            b"protocol",
            "seismic-analytical-evaluator-protocol/v2",
        );
        text(
            &mut protocol,
            b"model-revision",
            self.context.model_revision,
        );
        text(
            &mut protocol,
            b"probe-suite-revision",
            self.context.profile.identity().probe_suite_revision,
        );
        let provenance = EvaluationProvenance {
            protocol: protocol.finalize().into(),
            evidence_fingerprint: self.context.profile.identity().fingerprint,
        };
        domain
            .try_evaluate_total(
                provenance,
                PerformanceObjective::LATENCY,
                |arena, executable| {
                    evaluate_executable(
                        self.context.model.as_ref(),
                        arena,
                        executable,
                    )
                },
            )
            .map_err(|error| match error {
                EvaluationMapError::Evaluator(error) => error,
                EvaluationMapError::InvalidModel(error) => EvaluationError::InvalidModel(error),
            })
    }
}

fn evaluate_executable<T: seismic_target::TargetFamily>(
    model: &(dyn seismic_estimator::ExecutionModel<T> + Send + Sync),
    arena: &mut ExprArena,
    executable: TargetClosedExecutableView<'_, T>,
) -> Result<CandidatePerformanceModel, EvaluationError> {
    let model = seismic_estimator::estimate(model, arena, executable.execution())
        .map_err(EvaluationError::ModelLimitation)?;
    Ok(CandidatePerformanceModel {
        objective: PerformanceObjective::LATENCY,
        estimate: model.estimate(),
        uncertainty: CorrelatedUncertainty {
            maximum_relative_error_basis_points: model.maximum_relative_error_basis_points(),
            contributions: model
                .contributions()
                .iter()
                .map(|contribution| {
                    let mut region = sha2::Sha256::new();
                    region.update(b"seismic-performance-region/v1");
                    match contribution.region().invocation() {
                        seismic_estimator::InvocationProvenance::KernelLaunch { ordinal } => {
                            region.update(b"kernel");
                            region.update(ordinal.to_le_bytes());
                        }
                        seismic_estimator::InvocationProvenance::HostSchedule => {
                            region.update(b"host");
                        }
                    }
                    let mut correlation = sha2::Sha256::new();
                    correlation.update(b"seismic-performance-correlation/v1");
                    correlation.update(contribution.correlation().stable_name().as_bytes());
                    UncertaintyContribution {
                        region: region.finalize().into(),
                        correlation: correlation.finalize().into(),
                        guards: contribution.region().guards().to_vec(),
                        estimate: contribution.duration(),
                        evidence: contribution.evidence().clone(),
                    }
                })
                .collect(),
        },
    })
}

impl<B: seismic_target::TargetFamily> CandidateDomain<'_, B> {
    /// Attach analytical expressions without taking ownership of the structural domain.
    pub(crate) fn try_evaluate_total<E>(
        &mut self,
        provenance: EvaluationProvenance,
        objective: PerformanceObjective,
        mut evaluate: impl FnMut(
            &mut ExprArena,
            TargetClosedExecutableView<'_, B>,
        ) -> Result<CandidatePerformanceModel, E>,
    ) -> Result<AnalyticalDomainModel<B>, EvaluationMapError<E>> {
        let identity = EvaluationIdentity::new(self.device_identity().clone(), provenance);
        let candidates = self.structural_candidates().cloned().collect::<Vec<_>>();
        let mut evaluated = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let view = TargetClosedExecutableView::new(
                &candidate.family,
                candidate.constraints.conjuncts(),
            );
            let performance =
                evaluate(&mut self.arena_mut(), view).map_err(EvaluationMapError::Evaluator)?;
            validate_performance_model(&self.arena(), view, objective, &performance)
                .map_err(EvaluationMapError::InvalidModel)?;
            evaluated.push(EvaluatedCandidate {
                candidate,
                performance,
            });
        }
        let universal = evaluated.remove(0);
        // The required member fixes ordinary allocation axes to their fresh
        // values. The same physical data also exposes its other assignments
        // to optional analytical search; this is not another construction.
        if !universal.candidate.family.choices().is_empty() {
            evaluated.insert(
                0,
                EvaluatedCandidate {
                    candidate: universal.candidate.clone(),
                    performance: universal.performance.clone(),
                },
            );
        }
        Ok(AnalyticalDomainModel {
            evaluation: identity,
            universal,
            optimized: evaluated,
            construction_complete: false,
        })
    }
}

fn validate_performance_model<K: seismic_ir::target::PhysicalDialect>(
    arena: &ExprArena,
    executable: TargetClosedExecutableView<'_, K>,
    objective: PerformanceObjective,
    model: &CandidatePerformanceModel,
) -> Result<(), EvaluationModelError> {
    if model.objective != objective {
        return Err(EvaluationModelError::ObjectiveMismatch);
    }
    let decisions = executable
        .choices()
        .iter()
        .map(|axis| axis.decision())
        .collect::<std::collections::HashSet<_>>();
    let mut expressions = vec![AnyExpr::Duration(model.estimate)];
    for contribution in model.uncertainty.contributions() {
        expressions.push(AnyExpr::Duration(contribution.estimate));
        expressions.extend(contribution.guards.iter().copied().map(AnyExpr::Bool));
    }
    for expression in expressions {
        for symbol in arena.free_symbols(expression) {
            let allowed = match arena.symbol_kind(symbol) {
                SymbolKind::CallDimension(_)
                | SymbolKind::CallScalar(_)
                | SymbolKind::TargetConstant(_) => true,
                SymbolKind::Decision(decision) => decisions.contains(&decision),
                _ => false,
            };
            if !allowed {
                return Err(EvaluationModelError::UndeclaredSymbol);
            }
        }
    }
    Ok(())
}

#[derive(Debug)]
pub(crate) struct AnalyticalDomainModel<B: seismic_target::TargetFamily> {
    pub(crate) evaluation: EvaluationIdentity,
    pub(crate) universal: EvaluatedCandidate<B>,
    pub(crate) optimized: Vec<EvaluatedCandidate<B>>,
    pub(crate) construction_complete: bool,
}

#[derive(Debug)]
pub(crate) struct EvaluatedCandidate<B: seismic_target::TargetFamily> {
    pub(crate) candidate: DomainCandidate<B>,
    pub(crate) performance: CandidatePerformanceModel,
}
