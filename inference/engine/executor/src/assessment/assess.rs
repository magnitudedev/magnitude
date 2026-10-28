//! One complete model assessment, from headers and the allocation-free
//! execution plan draft, as two questions with one owner each: whether the
//! standard workload fits ([`assess_execution_fit`]: the clean-load charge
//! against every domain the load touches, `DoesNotFit` naming the domain
//! with the largest deficit), and how fast a fitting model decodes
//! ([`estimate_execution`]: its per-call decode demand over the executor's
//! measured kernel costs).

use super::demand::DecodeDemand;
use super::estimate::{estimate_decode, performance_depths, PerformanceEstimate, Pricing};
use super::executor_costs::ExecutorCosts;
use super::launches::Declarations;
use super::AssessmentError;
use crate::platform::{DomainRole, MemoryReserves};
use crate::{
    AssessmentFitVerdict, AssessmentGraphResourceBounds, AssessmentHeaderBounds,
    AssessmentMemoryCharge, AssessmentMemoryTerms, DeviceProcessBytes, ExecutionPlanDraft,
    HostProcessBytes, ResourceCapacity, ResourcePlanner,
};
use magnitude_family_contracts::ModelDefinition;
use seismic::{DeviceTopology, Executor, Status, StoreId};

/// Engine inputs to one assessment. No service profile or model identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssessmentRequest {
    /// The served context ceiling: the model's supported maximum.
    pub context_limit: u32,
    /// Requested decode-speed depths before filtering.
    pub performance_depths: Vec<u32>,
    /// The conversations decoding together: the serving profile's decode
    /// concurrency.
    pub sequences: u32,
    pub reserves: MemoryReserves,
}

/// Fit of the standard workload in one memory domain the load touches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DomainFit {
    pub role: DomainRole,
    pub domain: StoreId,
    /// What the executor's placements can hold in the store, within the
    /// store and the grants of the paths into it, with its planning
    /// reserve.
    pub capacity_bytes: u64,
    pub required_bytes: u64,
    /// The domain's planning reserve.
    pub reserve_bytes: u64,
    /// `capacity − reserve − required`.
    pub remaining_bytes: i64,
}

/// Whether the standard workload fits, each verdict with its own evidence.
#[derive(Clone, Debug, PartialEq)]
pub enum ExecutionAssessment {
    Fits(ExecutionFit),
    DoesNotFit(ExecutionShortfall),
}

/// The standard workload fits every domain the load touches.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionFit {
    pub fit_context_tokens: u32,
    pub domains: Vec<DomainFit>,
}

/// The standard workload exceeds some domain; `limiting` is the one with
/// the largest deficit.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionShortfall {
    pub fit_context_tokens: u32,
    pub domains: Vec<DomainFit>,
    pub limiting: StoreId,
    pub deficit_bytes: u64,
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

    /// The checked header and graph charge, shared with physical finalization.
    pub fn memory_charge(&self) -> &AssessmentMemoryCharge {
        &self.charge
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
    let backend = draft.device().backend();
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
    let header = AssessmentHeaderBounds::derive(draft.load()).map_err(AssessmentError::Memory)?;
    let state = ResourcePlanner::state_plan(
        definition,
        draft.load(),
        policy.method(),
        policy.codec(),
        policy.limits(),
        ResourceCapacity {
            domain_bytes: draft.device().assessment_capacity_bytes(),
            tensor_operations: crate::TensorOperations::of(draft.device().tensor_operations()),
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
        backend,
    )
    .map_err(AssessmentError::Graph)?;
    let host_process = HostProcessBytes::derive(
        definition,
        draft.load(),
        draft.programs(),
        policy.limits(),
        draft.device().backend(),
    )
    .map_err(AssessmentError::Memory)?;
    let device_process =
        DeviceProcessBytes::derive(draft.device().backend(), host_process.kernels)
            .map_err(AssessmentError::Memory)?;
    let charge = header
        .with_graph_resource_bound(&graph, host_process, device_process)
        .and_then(|bounds| {
            state
                .fit_state_bytes(terms.fit_depth, terms.recurrent_banks)
                .and_then(|state_bytes| {
                    terms.charge(bounds, state_bytes, state.startup_state_bytes()?)
                })
        })
        .map_err(AssessmentError::Memory)?;
    // The kernels a load prepares, at the statics it prepares them with:
    // what each call's launches are declared over.
    let declarations = Declarations::of_load(
        backend,
        draft.programs(),
        definition,
        draft.load(),
        crate::native::TuningLimits::of(policy.limits(), definition),
    )?;
    let demand = DecodeDemand::from_model(definition, draft.load(), policy.codec(), &declarations)?;
    Ok(PreparedExecutionAssessment {
        demand,
        fit_context_tokens,
        charge,
    })
}

/// The executor the draft plans on.
fn planned_executor<'a>(
    draft: &ExecutionPlanDraft,
    topology: &'a DeviceTopology,
) -> Result<&'a Executor, AssessmentError> {
    topology.resolve(draft.device().selector()).ok_or_else(|| {
        AssessmentError::Memory(format!(
            "planned device {} is absent from the topology",
            draft.device().selector()
        ))
    })
}

/// Whether the standard workload of one prepared model fits stable
/// capacity. No graph or model material is constructed here.
pub fn assess_execution_fit(
    prepared: &PreparedExecutionAssessment,
    draft: &ExecutionPlanDraft,
    topology: &DeviceTopology,
    status: &Status,
    request: &AssessmentRequest,
) -> Result<ExecutionAssessment, AssessmentError> {
    let device = planned_executor(draft, topology)?;
    let fit = prepared
        .charge
        .assess_fit(topology, device, status, &request.reserves)
        .map_err(AssessmentError::Memory)?;
    let fit_context_tokens = prepared.fit_context_tokens;
    Ok(match fit.verdict {
        AssessmentFitVerdict::DoesNotFit {
            limiting,
            deficit_bytes,
        } => ExecutionAssessment::DoesNotFit(ExecutionShortfall {
            fit_context_tokens,
            domains: fit.domains,
            limiting,
            deficit_bytes,
        }),
        AssessmentFitVerdict::Fits => ExecutionAssessment::Fits(ExecutionFit {
            fit_context_tokens,
            domains: fit.domains,
        }),
    })
}

/// What the draft's decode is priced with on `costs`: its resolved
/// lookahead policy.
pub fn decode_pricing<'a>(draft: &ExecutionPlanDraft, costs: &'a ExecutorCosts) -> Pricing<'a> {
    Pricing {
        costs,
        lookahead: draft.policy().limits().lookahead,
    }
}

/// One prepared model's decode speed at each requested depth, from the
/// planned executor's measured kernel costs.
pub fn estimate_execution(
    prepared: &PreparedExecutionAssessment,
    draft: &ExecutionPlanDraft,
    costs: &ExecutorCosts,
    request: &AssessmentRequest,
) -> Result<Vec<PerformanceEstimate>, AssessmentError> {
    estimate_decode(
        &prepared.demand,
        &decode_pricing(draft, costs),
        u64::from(request.sequences),
        &performance_depths(request.context_limit, &request.performance_depths),
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{
        ComponentSelection, ExecutionPath, ExecutionPlanner, PlannedMethod, ResourceLimits,
    };
    use magnitude_state::KvCodec;

    /// The limits the tests plan with.
    pub(crate) const LIMITS: ResourceLimits = ResourceLimits {
        max_launch_rows: 512,
        max_launch_slots: 32,
        max_selected_rows: 32,
        max_drafting_slots: 32,
        exported_logits_rows: 0,
        max_images_per_request: 1,
        max_image_cells: 0,
        lookahead: true,
    };

    struct Environment {
        topology: std::sync::Arc<DeviceTopology>,
        status: Status,
        draft: ExecutionPlanDraft,
        definition: ModelDefinition,
    }

    fn environment(context_limit: u64) -> Environment {
        environment_within(context_limit, u64::MAX)
    }

    /// The fixture planned on the host's device with at most
    /// `capacity_bytes` of its stable capacity.
    fn environment_within(context_limit: u64, capacity_bytes: u64) -> Environment {
        let inventory = std::sync::Arc::new(seismic::Inventory::discover(&[]).unwrap());
        let catalog = seismic::Devices::new(inventory.clone());
        let mut selected = crate::platform::select_device(
            &catalog,
            ExecutionPath::Native,
            crate::platform::DeviceRequest::Automatic,
            &MemoryReserves::standard(),
        )
        .unwrap();
        selected.assessment_capacity_bytes = selected.assessment_capacity_bytes.min(capacity_bytes);
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
            LIMITS,
        )
        .unwrap();
        Environment {
            topology: catalog.topology(),
            status: catalog.status(),
            draft,
            definition,
        }
    }

    fn request(context_limit: u32, depths: &[u32]) -> AssessmentRequest {
        AssessmentRequest {
            context_limit,
            performance_depths: depths.to_vec(),
            sequences: 1,
            reserves: MemoryReserves::standard(),
        }
    }

    fn assess(environment: &Environment, request: &AssessmentRequest) -> ExecutionAssessment {
        let prepared = prepare_execution_assessment(
            &environment.definition,
            &environment.draft,
            request.context_limit,
        )
        .unwrap();
        assess_execution_fit(
            &prepared,
            &environment.draft,
            &environment.topology,
            &environment.status,
            request,
        )
        .unwrap()
    }

    #[test]
    fn fit_depth_is_independent_of_performance_depths() {
        let environment = environment(150_000);
        let ExecutionAssessment::Fits(ExecutionFit {
            fit_context_tokens,
            domains,
        }) = assess(
            &environment,
            &request(150_000, &[25_000, 50_000, 50_000, 200_000]),
        )
        else {
            panic!("the fixture fits every supported host");
        };
        assert_eq!(fit_context_tokens, 100_000);
        assert!(domains
            .iter()
            .any(|domain| domain.role == DomainRole::Allocation));
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
        let ExecutionAssessment::Fits(ExecutionFit {
            domains: other_domains,
            ..
        }) = assess(&environment, &request(150_000, &[1_000]))
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

    /// The estimate covers the requested depths within the limit and the
    /// limit itself, slowing with depth.
    #[test]
    fn a_fitting_model_is_estimated_at_each_depth() {
        let environment = environment(150_000);
        let request = request(150_000, &[25_000, 50_000, 50_000, 200_000]);
        let prepared =
            prepare_execution_assessment(&environment.definition, &environment.draft, 150_000)
                .unwrap();
        let costs = crate::assessment::fixtures::costs();
        let estimates =
            estimate_execution(&prepared, &environment.draft, &costs, &request).unwrap();
        assert_eq!(
            estimates
                .iter()
                .map(|estimate| estimate.context_tokens)
                .collect::<Vec<_>>(),
            vec![25_000, 50_000, 150_000]
        );
        for pair in estimates.windows(2) {
            assert!(pair[0].tokens_per_second > pair[1].tokens_per_second);
        }
        assert!(estimates
            .iter()
            .all(|estimate| estimate.tokens_per_second.is_finite()));
    }

    /// A device whose domain cannot back one full context's history is
    /// assessed like any other, never failing to plan: the fit at the fit
    /// depth gives the verdict.
    #[test]
    fn a_context_beyond_the_domains_history_capacity_is_assessed() {
        // The fixture stores 256 bytes of history per token: 150,000 tokens
        // are 38.4 MB, beyond a 16 MiB domain.
        let environment = environment_within(150_000, 16 << 20);
        let request = request(150_000, &[25_000]);
        let prepared =
            prepare_execution_assessment(&environment.definition, &environment.draft, 150_000)
                .expect("a context the domain cannot back is not a planning failure");
        assert_eq!(prepared.fit_context_tokens, 100_000);
        assert!(assess_execution_fit(
            &prepared,
            &environment.draft,
            &environment.topology,
            &environment.status,
            &request,
        )
        .is_ok());
    }

    #[test]
    fn a_mismatched_context_limit_is_an_error() {
        let environment = environment(128);
        assert!(matches!(
            prepare_execution_assessment(&environment.definition, &environment.draft, 256),
            Err(AssessmentError::Plan(_))
        ));
    }
}
