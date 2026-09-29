//! One complete model assessment: memory fit per domain, compatibility with
//! the measurement basis, and decode-speed estimates, all from headers and
//! the allocation-free execution plan draft.
//!
//! Order: the draft's decode demand is checked against the basis first (a
//! class outside it is `Incompatible`); then the standard workload's clean-load
//! charge is compared with every domain the load touches (`DoesNotFit` names
//! the domain with the largest deficit); only a fitting model is estimated.

use super::basis::{MeasurementBasis, MeasurementKey};
use super::demand::DecodeDemand;
use super::estimate::{estimate_performance, performance_depths, PerformanceEstimate};
use super::AssessmentError;
use crate::platform::{fit_capacities, DomainRole, MemoryReserves};
use crate::{
    AssessmentFitVerdict, AssessmentGraphResourceBounds, AssessmentHeaderBounds,
    AssessmentMemoryCharge, AssessmentMemoryTerms, ExecutionPlanDraft, ResourceCapacity,
    ResourcePlanner,
};
use magnitude_family_contracts::ModelDefinition;
use seismic::{DeviceTopology, HostMemoryStatus, MemoryPoolId};

/// Engine inputs to one assessment. No service profile or model identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssessmentRequest {
    /// The served context ceiling: the model's supported maximum.
    pub context_limit: u32,
    /// Requested decode-speed depths before filtering.
    pub performance_depths: Vec<u32>,
    pub reserves: MemoryReserves,
}

/// Fit of the standard workload in one memory domain the load touches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DomainFit {
    pub role: DomainRole,
    pub domain: MemoryPoolId,
    pub capacity_bytes: u64,
    pub required_bytes: u64,
    /// The domain's planning reserve.
    pub reserve_bytes: u64,
    /// `capacity − reserve − required`.
    pub remaining_bytes: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum IncompatibleReason {
    /// The plan needs classes the basis did not measure on this device.
    OutsideBasis {
        classes: Vec<(MeasurementKey, Option<String>)>,
    },
    /// Planning rejected the model's representation or topology.
    Unsupported { reason: String },
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExecutionAssessment {
    Fits {
        fit_context_tokens: u32,
        domains: Vec<DomainFit>,
        performance: Vec<PerformanceEstimate>,
    },
    DoesNotFit {
        fit_context_tokens: u32,
        domains: Vec<DomainFit>,
        limiting: MemoryPoolId,
        deficit_bytes: u64,
    },
    Incompatible {
        reason: IncompatibleReason,
    },
}

/// Basis-independent decode demand and the standard workload's checked memory charge.
/// The memory result is retained separately so an unsupported basis class keeps its
/// `Incompatible` verdict even when this model's resource certification also fails.
pub struct PreparedExecutionAssessment {
    demand: DecodeDemand,
    memory: Result<(u32, AssessmentMemoryCharge), AssessmentError>,
}

impl PreparedExecutionAssessment {
    /// The plain decode step's demand under the planned history codec.
    pub fn demand(&self) -> &DecodeDemand {
        &self.demand
    }
}

/// Perform model-specific arithmetic while the fixed generic basis is measured.
pub fn prepare_execution_assessment(
    definition: &ModelDefinition,
    draft: &ExecutionPlanDraft,
    context_limit: u32,
) -> Result<PreparedExecutionAssessment, AssessmentError> {
    if definition.decoder.context_limit != u64::from(context_limit) {
        return Err(AssessmentError::Plan(format!(
            "assessed context limit {} differs from the planned model's {}",
            context_limit, definition.decoder.context_limit
        )));
    }
    let policy = draft.policy();
    let demand = DecodeDemand::from_model(definition, draft.load(), policy.codec())?;
    let memory = prepare_memory_charge(definition, draft);
    Ok(PreparedExecutionAssessment { demand, memory })
}

fn prepare_memory_charge(
    definition: &ModelDefinition,
    draft: &ExecutionPlanDraft,
) -> Result<(u32, AssessmentMemoryCharge), AssessmentError> {
    let policy = draft.policy();
    let terms = AssessmentMemoryTerms::derive(
        definition,
        draft.load(),
        policy.selection(),
        policy.codec(),
        policy.method(),
        policy.limits(),
    )
    .map_err(AssessmentError::Memory)?;
    let fit_context_tokens = u32::try_from(terms.fit_depth)
        .map_err(|_| AssessmentError::Memory("fit depth exceeds u32".into()))?;
    let header = AssessmentHeaderBounds::derive(definition, draft.load(), policy.codec())
        .map_err(AssessmentError::Memory)?;
    let state = ResourcePlanner::state_plan(
        definition,
        draft.load(),
        policy.method(),
        policy.codec(),
        policy.limits(),
        ResourceCapacity {
            domain_bytes: draft.device().assessment_capacity_bytes(),
        },
    )
    .map_err(AssessmentError::Plan)?;
    let graph = AssessmentGraphResourceBounds::derive(
        definition,
        draft.load(),
        &state,
        policy.method(),
        policy.codec(),
        policy.limits(),
        draft.device().backend(),
    )
    .map_err(AssessmentError::Memory)?;
    let charge = header
        .with_graph_resource_bound(&graph)
        .and_then(|bounds| {
            state
                .fit_state_bytes(terms.fit_depth, terms.recurrent_banks)
                .and_then(|state_bytes| terms.charge(bounds, state_bytes))
        })
        .map_err(AssessmentError::Memory)?;
    Ok((fit_context_tokens, charge))
}

/// Join one prepared model with the basis and stable capacity. No graph or
/// model material is constructed here.
pub fn finish_execution_assessment(
    prepared: &PreparedExecutionAssessment,
    draft: &ExecutionPlanDraft,
    topology: &DeviceTopology,
    host: &HostMemoryStatus,
    basis: &MeasurementBasis,
    request: &AssessmentRequest,
) -> Result<ExecutionAssessment, AssessmentError> {
    let unmeasured = prepared.demand.unmeasured(basis);
    if !unmeasured.is_empty() {
        return Ok(ExecutionAssessment::Incompatible {
            reason: IncompatibleReason::OutsideBasis {
                classes: unmeasured,
            },
        });
    }
    let &(fit_context_tokens, charge) = prepared.memory.as_ref().map_err(Clone::clone)?;
    let device = topology
        .devices()
        .iter()
        .find(|device| device.selector == draft.device().selector())
        .ok_or_else(|| {
            AssessmentError::Memory(format!(
                "planned device {} is absent from the topology",
                draft.device().selector()
            ))
        })?;
    let capacities = fit_capacities(topology, device, host, &request.reserves)
        .map_err(|error| AssessmentError::Memory(error.to_string()))?;
    let fit = charge
        .assess_fit(&capacities)
        .map_err(AssessmentError::Memory)?;
    match fit.verdict {
        AssessmentFitVerdict::DoesNotFit {
            limiting,
            deficit_bytes,
        } => Ok(ExecutionAssessment::DoesNotFit {
            fit_context_tokens,
            domains: fit.domains,
            limiting,
            deficit_bytes,
        }),
        AssessmentFitVerdict::Fits => {
            let depths = performance_depths(request.context_limit, &request.performance_depths);
            let performance = estimate_performance(&prepared.demand, basis, &depths)?;
            Ok(ExecutionAssessment::Fits {
                fit_context_tokens,
                domains: fit.domains,
                performance,
            })
        }
    }
}

/// Assess one planned model against stable capacity and the basis. Opens no
/// device, reads no weight payload and allocates nothing.
pub fn assess_execution(
    definition: &ModelDefinition,
    draft: &ExecutionPlanDraft,
    topology: &DeviceTopology,
    host: &HostMemoryStatus,
    basis: &MeasurementBasis,
    request: &AssessmentRequest,
) -> Result<ExecutionAssessment, AssessmentError> {
    let prepared = prepare_execution_assessment(definition, draft, request.context_limit)?;
    finish_execution_assessment(&prepared, draft, topology, host, basis, request)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assessment::{
        BasisIdentity, ClassCost, ClassMeasurement, CostModel, HistoryCost, MeasurementKey,
        ProjectionCost, TermShape,
    };
    use crate::{
        ComponentSelection, ExecutionPath, ExecutionPlanner, PlannedMethod, ResourceLimits,
    };
    use magnitude_state::KvCodec;

    struct Environment {
        topology: std::sync::Arc<DeviceTopology>,
        host: HostMemoryStatus,
        draft: ExecutionPlanDraft,
        definition: ModelDefinition,
    }

    fn environment(context_limit: u64) -> Environment {
        let catalog = seismic::DeviceCatalog::discover().unwrap();
        let selected = crate::platform::select_device(
            &catalog,
            ExecutionPath::Native,
            crate::platform::DeviceRequest::Automatic,
            &MemoryReserves::standard(),
        )
        .unwrap();
        let mut definition = crate::planning::tests::fixture_definition();
        definition.decoder.context_limit = context_limit;
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let draft = ExecutionPlanner::prepare(
            &selected,
            &manifest,
            &definition,
            ComponentSelection {
                head: false,
                vision: false,
            },
            ExecutionPath::Native,
            PlannedMethod::Plain,
            KvCodec::Dense,
            ResourceLimits {
                max_launch_rows: 512,
                max_launch_slots: 32,
                max_projected_rows: 32,
                max_images_per_request: 1,
                lookahead: true,
            },
        )
        .unwrap();
        Environment {
            topology: catalog.topology(),
            host: catalog.host_memory_status().unwrap(),
            draft,
            definition,
        }
    }

    fn identity() -> BasisIdentity {
        BasisIdentity {
            engine_build: "test".into(),
            backend: "test".into(),
            device: "test".into(),
            protocol_version: crate::assessment::MEASUREMENT_PROTOCOL_VERSION,
        }
    }

    /// A basis holding every entry the fixture's decode needs: each term's
    /// cost key timed at 10 µs a launch, its weight format and its exact
    /// representation binding formed.
    fn complete_basis(environment: &Environment) -> MeasurementBasis {
        let demand = DecodeDemand::from_model(
            &environment.definition,
            environment.draft.load(),
            environment.draft.policy().codec(),
        )
        .unwrap();
        let measured = |model| ClassMeasurement::Measured {
            points: Vec::new(),
            cost: ClassCost {
                model,
                slow_factor: 1.1,
                fast_factor: 0.9,
                limited_evidence: false,
            },
        };
        let mut classes: Vec<(MeasurementKey, ClassMeasurement)> = Vec::new();
        let mut hold = |key: MeasurementKey, measurement: ClassMeasurement| {
            if classes.iter().all(|(known, _)| *known != key) {
                classes.push((key, measurement));
            }
        };
        for term in &demand.terms {
            let model = match term.shape {
                TermShape::Plain => CostModel::PerLaunch { seconds: 1e-5 },
                TermShape::Projection { weight, .. } => {
                    hold(
                        MeasurementKey::weight_format(weight, term.key.cost().bindings[0]),
                        measured(CostModel::PerByte {
                            seconds_per_byte: 1e-12,
                        }),
                    );
                    CostModel::Projection(ProjectionCost {
                        launch_seconds: 1e-5,
                        weight,
                        seconds_per_byte: vec![(1, 0.0)],
                    })
                }
                TermShape::Attention(reference) => CostModel::History(HistoryCost {
                    launch_seconds: 1e-5,
                    seconds_per_byte: 0.0,
                    reference,
                    kv_heads: vec![(reference.kv_heads, 1.0)],
                    group: vec![(reference.group, 1.0)],
                    width: vec![(reference.width, 1.0)],
                }),
            };
            hold(term.key.cost(), measured(model));
            if term.key.class.binds_representation() {
                hold(term.key.clone(), ClassMeasurement::Formed);
            }
        }
        MeasurementBasis {
            identity: identity(),
            classes,
        }
    }

    fn request(context_limit: u32, depths: &[u32]) -> AssessmentRequest {
        AssessmentRequest {
            context_limit,
            performance_depths: depths.to_vec(),
            reserves: MemoryReserves::standard(),
        }
    }

    #[test]
    fn classes_outside_the_basis_are_incompatible() {
        let environment = environment(128);
        let empty = MeasurementBasis {
            identity: identity(),
            classes: Vec::new(),
        };
        let assessment = assess_execution(
            &environment.definition,
            &environment.draft,
            &environment.topology,
            &environment.host,
            &empty,
            &request(128, &[64]),
        )
        .unwrap();
        let ExecutionAssessment::Incompatible {
            reason: IncompatibleReason::OutsideBasis { classes },
        } = assessment
        else {
            panic!("expected an outside-basis incompatibility, got {assessment:?}");
        };
        assert!(!classes.is_empty());
        assert!(classes.iter().all(|(_, reason)| reason.is_none()));

        let mut basis = complete_basis(&environment);
        let (key, _) = basis.classes.remove(0);
        basis.classes.push((
            key.clone(),
            ClassMeasurement::Unsupported {
                reason: "cannot form".into(),
            },
        ));
        assert_eq!(
            assess_execution(
                &environment.definition,
                &environment.draft,
                &environment.topology,
                &environment.host,
                &basis,
                &request(128, &[64]),
            )
            .unwrap(),
            ExecutionAssessment::Incompatible {
                reason: IncompatibleReason::OutsideBasis {
                    classes: vec![(key, Some("cannot form".into()))],
                },
            }
        );
    }

    #[test]
    fn fit_depth_is_independent_of_performance_depths() {
        let environment = environment(150_000);
        let basis = complete_basis(&environment);
        let assessment = assess_execution(
            &environment.definition,
            &environment.draft,
            &environment.topology,
            &environment.host,
            &basis,
            &request(150_000, &[25_000, 50_000, 50_000, 200_000]),
        )
        .unwrap();
        let ExecutionAssessment::Fits {
            fit_context_tokens,
            domains,
            performance,
        } = assessment
        else {
            panic!("the fixture fits every supported host, got {assessment:?}");
        };
        assert_eq!(fit_context_tokens, 100_000);
        assert_eq!(
            performance
                .iter()
                .map(|estimate| estimate.context_tokens)
                .collect::<Vec<_>>(),
            vec![25_000, 50_000, 150_000]
        );
        assert_eq!(domains[0].role, DomainRole::Allocation);
        assert!(domains.iter().all(|domain| domain.remaining_bytes >= 0));
        for domain in &domains {
            assert_eq!(
                i128::from(domain.remaining_bytes),
                i128::from(domain.capacity_bytes)
                    - i128::from(domain.reserve_bytes)
                    - i128::from(domain.required_bytes)
            );
        }
        // The charge depends on the fit depth only, not on the depths asked.
        let other = assess_execution(
            &environment.definition,
            &environment.draft,
            &environment.topology,
            &environment.host,
            &basis,
            &request(150_000, &[1_000]),
        )
        .unwrap();
        let ExecutionAssessment::Fits {
            domains: other_domains,
            ..
        } = other
        else {
            panic!("expected a fit");
        };
        assert_eq!(
            other_domains
                .iter()
                .map(|domain| domain.required_bytes)
                .collect::<Vec<_>>(),
            domains
                .iter()
                .map(|domain| domain.required_bytes)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_mismatched_context_limit_is_an_error() {
        let environment = environment(128);
        let basis = complete_basis(&environment);
        assert!(matches!(
            assess_execution(
                &environment.definition,
                &environment.draft,
                &environment.topology,
                &environment.host,
                &basis,
                &request(256, &[64]),
            ),
            Err(AssessmentError::Plan(_))
        ));
    }
}
