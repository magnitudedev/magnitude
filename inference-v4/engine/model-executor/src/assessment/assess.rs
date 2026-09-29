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
    AssessmentMemoryTerms, ExecutionPlanDraft, ResourceCapacity, ResourcePlanner,
};
use magnitude_model_contracts::ModelDefinition;
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
    if definition.geometry.context_limit != u64::from(request.context_limit) {
        return Err(AssessmentError::Plan(format!(
            "assessed context limit {} differs from the planned model's {}",
            request.context_limit, definition.geometry.context_limit
        )));
    }
    let policy = draft.policy();
    let demand = DecodeDemand::from_model(definition, draft.load(), policy.codec())?;
    let unmeasured = demand.unmeasured(basis);
    if !unmeasured.is_empty() {
        return Ok(ExecutionAssessment::Incompatible {
            reason: IncompatibleReason::OutsideBasis {
                classes: unmeasured
                    .into_iter()
                    .map(|(key, reason)| (key.clone(), reason.map(str::to_owned)))
                    .collect(),
            },
        });
    }

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
    // The same state plan production derives before sealing its graphs, so
    // the graph classes and slot multipliers are the production ones.
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
        .and_then(|bounds| terms.charge(bounds))
        .map_err(AssessmentError::Memory)?;
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
            let performance = estimate_performance(&demand, basis, &depths)?;
            Ok(ExecutionAssessment::Fits {
                fit_context_tokens,
                domains: fit.domains,
                performance,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assessment::{BasisIdentity, ClassCost, ClassMeasurement, CostModel, MeasuredPoint};
    use crate::{
        ComponentSelection, ExecutionPath, ExecutionPlanner, PlannedMethod, ResourceLimits,
    };
    use magnitude_model_state::KvCodec;

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
        definition.geometry.context_limit = context_limit;
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
                max_retained_entries: 128,
                active_requests: 1,
                in_flight_requests: 1,
                branch_checkpoints: 1,
                max_batch_rows: 512,
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

    /// A basis measuring every class the fixture's decode launches.
    fn complete_basis(environment: &Environment) -> MeasurementBasis {
        let demand = DecodeDemand::from_model(
            &environment.definition,
            environment.draft.load(),
            environment.draft.policy().codec(),
        )
        .unwrap();
        MeasurementBasis {
            identity: identity(),
            classes: demand
                .terms
                .iter()
                .map(|term| {
                    (
                        term.key.clone(),
                        ClassMeasurement::Measured {
                            points: vec![MeasuredPoint {
                                bytes: 1 << 20,
                                samples: vec![9e-6, 1e-5, 1.1e-5],
                            }],
                            cost: ClassCost {
                                model: CostModel::PerLaunch { seconds: 1e-5 },
                                slow_factor: 1.1,
                                fast_factor: 0.9,
                            },
                        },
                    )
                })
                .collect(),
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
