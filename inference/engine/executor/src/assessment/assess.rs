//! One complete model assessment: memory fit per domain and decode-speed
//! estimates, all from headers and the allocation-free execution plan draft.
//!
//! The standard workload's clean-load charge is compared with every domain
//! the load touches (`DoesNotFit` names the domain with the largest deficit);
//! only a fitting model is estimated, from its decode demand over the
//! device's bandwidth.

use super::bandwidth::DeviceBandwidth;
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
pub enum ExecutionAssessment {
    Fits {
        fit_context_tokens: u32,
        domains: Vec<DomainFit>,
        /// One estimate per requested depth, ascending.
        performance: Vec<PerformanceEstimate>,
    },
    DoesNotFit {
        fit_context_tokens: u32,
        domains: Vec<DomainFit>,
        limiting: MemoryPoolId,
        deficit_bytes: u64,
    },
}

/// Basis-independent decode demand and the standard workload's checked
/// memory charge.
pub struct PreparedExecutionAssessment {
    demand: DecodeDemand,
    fit_context_tokens: u32,
    charge: AssessmentMemoryCharge,
}

impl PreparedExecutionAssessment {
    /// The plain decode step's demand under the planned history codec.
    pub fn demand(&self) -> &DecodeDemand {
        &self.demand
    }
}

/// Perform model-specific arithmetic while the fixed generic basis is
/// measured. The memory charge builds every graph a load prepares, so this
/// fails with `AssessmentError::Graph` exactly when the model's program
/// cannot be built on the planned backend.
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
    .map_err(AssessmentError::Graph)?;
    let charge = header
        .with_graph_resource_bound(&graph)
        .and_then(|bounds| {
            state
                .fit_state_bytes(terms.fit_depth, terms.recurrent_banks)
                .and_then(|state_bytes| terms.charge(bounds, state_bytes))
        })
        .map_err(AssessmentError::Memory)?;
    Ok(PreparedExecutionAssessment {
        demand,
        fit_context_tokens,
        charge,
    })
}

/// Join one prepared model with stable capacity and the device's bandwidth.
/// No graph or model material is constructed here.
pub fn finish_execution_assessment(
    prepared: &PreparedExecutionAssessment,
    draft: &ExecutionPlanDraft,
    topology: &DeviceTopology,
    host: &HostMemoryStatus,
    bandwidth: DeviceBandwidth,
    request: &AssessmentRequest,
) -> Result<ExecutionAssessment, AssessmentError> {
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
    let fit = prepared
        .charge
        .assess_fit(&capacities)
        .map_err(AssessmentError::Memory)?;
    let fit_context_tokens = prepared.fit_context_tokens;
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
        AssessmentFitVerdict::Fits => Ok(ExecutionAssessment::Fits {
            fit_context_tokens,
            domains: fit.domains,
            performance: estimate_performance(
                &prepared.demand,
                bandwidth,
                &performance_depths(request.context_limit, &request.performance_depths),
            ),
        }),
    }
}

/// Assess one planned model against stable capacity and the device's
/// bandwidth. Opens no device, reads no weight payload and allocates
/// nothing.
pub fn assess_execution(
    definition: &ModelDefinition,
    draft: &ExecutionPlanDraft,
    topology: &DeviceTopology,
    host: &HostMemoryStatus,
    bandwidth: DeviceBandwidth,
    request: &AssessmentRequest,
) -> Result<ExecutionAssessment, AssessmentError> {
    let prepared = prepare_execution_assessment(definition, draft, request.context_limit)?;
    finish_execution_assessment(&prepared, draft, topology, host, bandwidth, request)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assessment::bandwidth::BandwidthSource;
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

    const BANDWIDTH: DeviceBandwidth = DeviceBandwidth {
        bytes_per_second: 100_000_000_000,
        source: BandwidthSource::Published,
    };

    fn request(context_limit: u32, depths: &[u32]) -> AssessmentRequest {
        AssessmentRequest {
            context_limit,
            performance_depths: depths.to_vec(),
            reserves: MemoryReserves::standard(),
        }
    }

    fn assess(environment: &Environment, request: &AssessmentRequest) -> ExecutionAssessment {
        assess_execution(
            &environment.definition,
            &environment.draft,
            &environment.topology,
            &environment.host,
            BANDWIDTH,
            request,
        )
        .unwrap()
    }

    #[test]
    fn fit_depth_is_independent_of_performance_depths() {
        let environment = environment(150_000);
        let ExecutionAssessment::Fits {
            fit_context_tokens,
            domains,
            performance,
        } = assess(
            &environment,
            &request(150_000, &[25_000, 50_000, 50_000, 200_000]),
        )
        else {
            panic!("the fixture fits every supported host");
        };
        assert_eq!(fit_context_tokens, 100_000);
        assert_eq!(
            performance
                .iter()
                .map(|estimate| estimate.context_tokens)
                .collect::<Vec<_>>(),
            vec![25_000, 50_000, 150_000]
        );
        assert!(performance
            .iter()
            .all(|estimate| estimate.tokens_per_second.is_finite()
                && estimate.tokens_per_second > 0.0));
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
        let ExecutionAssessment::Fits {
            domains: other_domains,
            ..
        } = assess(&environment, &request(150_000, &[1_000]))
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
        assert!(matches!(
            assess_execution(
                &environment.definition,
                &environment.draft,
                &environment.topology,
                &environment.host,
                BANDWIDTH,
                &request(256, &[64]),
            ),
            Err(AssessmentError::Plan(_))
        ));
    }
}
