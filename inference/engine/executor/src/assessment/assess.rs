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
    AssessmentMemoryBounds, AssessmentMemoryCharge, AssessmentMemoryTerms, DeviceProcessBytes,
    ExecutionPlanDraft, HostProcessBytes, PlannedMethod, ResourceCapacity, ResourcePlanner,
    StateResourcePlan,
};
use magnitude_family_contracts::ModelDefinition;
use seismic::{DeviceTopology, Executor, Status, StoreId};

/// Engine inputs to one assessment. No service profile or model identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssessmentRequest {
    /// The served context ceiling: the model's supported maximum.
    pub context_limit: u32,
    /// The contexts the model may be served at, largest first, none above
    /// the ceiling. The served context is the first at which the standard
    /// workload fits; a shortfall is reported at the last.
    pub served_contexts: Vec<u32>,
    /// Requested decode-speed depths before filtering by the served context.
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

/// The standard workload fits every domain the load touches at the served
/// context `fit_context_tokens`.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionFit {
    pub fit_context_tokens: u32,
    pub domains: Vec<DomainFit>,
}

/// The standard workload exceeds some domain at every served context;
/// `fit_context_tokens` is the smallest, and `limiting` the domain with the
/// largest deficit there.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionShortfall {
    pub fit_context_tokens: u32,
    pub domains: Vec<DomainFit>,
    pub limiting: StoreId,
    pub deficit_bytes: u64,
}

/// Basis-independent decode demand and the standard workload's checked
/// memory charge, at the context ceiling and at any served context below it.
/// The graphs and their bounds are prepared once at the ceiling; the state
/// plan is derived at each served context, as a load at that context plans
/// it.
pub struct PreparedExecutionAssessment {
    demand: DecodeDemand,
    fit_context_tokens: u32,
    charge: AssessmentMemoryCharge,
    terms: AssessmentMemoryTerms,
    bounds: AssessmentMemoryBounds,
    state: StateResourcePlan,
    method: PlannedMethod,
}

impl PreparedExecutionAssessment {
    /// The plain decode step's demand under the planned history codec.
    pub fn demand(&self) -> &DecodeDemand {
        &self.demand
    }

    /// The checked header and graph charge at the ceiling, shared with
    /// physical finalization.
    pub fn memory_charge(&self) -> &AssessmentMemoryCharge {
        &self.charge
    }

    /// The charge with the standard workload's one conversation at
    /// `context_tokens`, which must not exceed the ceiling.
    pub fn memory_charge_at(&self, context_tokens: u32) -> Result<AssessmentMemoryCharge, String> {
        if context_tokens == 0 || context_tokens > self.fit_context_tokens {
            return Err(format!(
                "served context {context_tokens} is outside the ceiling {}",
                self.fit_context_tokens
            ));
        }
        if context_tokens == self.fit_context_tokens {
            return Ok(self.charge);
        }
        let state = self.state.at_context(self.method, u64::from(context_tokens))?;
        let state_bytes =
            state.fit_state_bytes(u64::from(context_tokens), self.terms.recurrent_banks)?;
        self.terms
            .charge(self.bounds, state_bytes, state.startup_state_bytes()?)
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
        DeviceProcessBytes::derive(
            draft.device().backend(),
            host_process.kernels,
            draft.load().weights(),
        )
            .map_err(AssessmentError::Memory)?;
    let bounds = header
        .with_graph_resource_bound(&graph, host_process, device_process)
        .map_err(AssessmentError::Memory)?;
    let charge = state
        .fit_state_bytes(terms.fit_depth, terms.recurrent_banks)
        .and_then(|state_bytes| terms.charge(bounds, state_bytes, state.startup_state_bytes()?))
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
        terms,
        bounds,
        state,
        method: policy.method(),
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
/// capacity at one of the request's served contexts: the first, largest
/// first, at which it fits, or the shortfall at the smallest. Only the
/// conversation's state differs between them. No graph or model material is
/// constructed here.
pub fn assess_execution_fit(
    prepared: &PreparedExecutionAssessment,
    draft: &ExecutionPlanDraft,
    topology: &DeviceTopology,
    status: &Status,
    request: &AssessmentRequest,
) -> Result<ExecutionAssessment, AssessmentError> {
    if request.context_limit != prepared.fit_context_tokens {
        return Err(AssessmentError::Plan(format!(
            "assessed context limit {} differs from the prepared {}",
            request.context_limit, prepared.fit_context_tokens
        )));
    }
    if request.served_contexts.is_empty()
        || request
            .served_contexts
            .windows(2)
            .any(|pair| pair[0] <= pair[1])
    {
        return Err(AssessmentError::Plan(
            "served contexts must be nonempty and strictly decreasing".into(),
        ));
    }
    let device = planned_executor(draft, topology)?;
    let mut shortfall = None;
    for &fit_context_tokens in &request.served_contexts {
        let fit = prepared
            .memory_charge_at(fit_context_tokens)
            .and_then(|charge| charge.assess_fit(topology, device, status, &request.reserves))
            .map_err(AssessmentError::Memory)?;
        match fit.verdict {
            AssessmentFitVerdict::Fits => {
                return Ok(ExecutionAssessment::Fits(ExecutionFit {
                    fit_context_tokens,
                    domains: fit.domains,
                }));
            }
            AssessmentFitVerdict::DoesNotFit {
                limiting,
                deficit_bytes,
            } => {
                shortfall = Some(ExecutionShortfall {
                    fit_context_tokens,
                    domains: fit.domains,
                    limiting,
                    deficit_bytes,
                });
            }
        }
    }
    Ok(ExecutionAssessment::DoesNotFit(
        shortfall.expect("served contexts are nonempty"),
    ))
}

/// What the draft's decode is priced with on `costs`: its resolved
/// lookahead policy.
pub fn decode_pricing<'a>(draft: &ExecutionPlanDraft, costs: &'a ExecutorCosts) -> Pricing<'a> {
    Pricing {
        costs,
        lookahead: draft.policy().limits().lookahead,
    }
}

/// One prepared model's decode speed at each requested depth within the
/// context it is served at, from the planned executor's measured kernel
/// costs.
pub fn estimate_execution(
    prepared: &PreparedExecutionAssessment,
    draft: &ExecutionPlanDraft,
    costs: &ExecutorCosts,
    request: &AssessmentRequest,
    served_context: u32,
) -> Result<Vec<PerformanceEstimate>, AssessmentError> {
    estimate_decode(
        &prepared.demand,
        &decode_pricing(draft, costs),
        u64::from(request.sequences),
        &performance_depths(served_context, &request.performance_depths),
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{
        ComponentSelection, ExecutionPath, ExecutionPlanner, ResourceLimits,
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
            status: catalog.status(&catalog.topology(), selected.info.id()),
            topology: catalog.topology(),
            draft,
            definition,
        }
    }

    fn request(context_limit: u32, depths: &[u32]) -> AssessmentRequest {
        tiered(context_limit, &[context_limit], depths)
    }

    fn tiered(context_limit: u32, served_contexts: &[u32], depths: &[u32]) -> AssessmentRequest {
        AssessmentRequest {
            context_limit,
            served_contexts: served_contexts.to_vec(),
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
        assert_eq!(fit_context_tokens, 150_000);
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

    /// The estimate covers the requested depths within the served context
    /// and the served context itself, slowing with depth.
    #[test]
    fn a_fitting_model_is_estimated_at_each_depth() {
        let environment = environment(150_000);
        let request = request(150_000, &[25_000, 50_000, 50_000, 200_000]);
        let prepared =
            prepare_execution_assessment(&environment.definition, &environment.draft, 150_000)
                .unwrap();
        let costs = crate::assessment::fixtures::costs();
        let estimates =
            estimate_execution(&prepared, &environment.draft, &costs, &request, 150_000).unwrap();
        assert_eq!(
            estimates
                .iter()
                .map(|estimate| estimate.context_tokens)
                .collect::<Vec<_>>(),
            vec![25_000, 50_000, 150_000]
        );
        let served =
            estimate_execution(&prepared, &environment.draft, &costs, &request, 32_768).unwrap();
        assert_eq!(
            served
                .iter()
                .map(|estimate| estimate.context_tokens)
                .collect::<Vec<_>>(),
            vec![25_000, 32_768]
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
        assert_eq!(prepared.fit_context_tokens, 150_000);
        assert!(assess_execution_fit(
            &prepared,
            &environment.draft,
            &environment.topology,
            &environment.status,
            &request,
        )
        .is_ok());
    }

    /// A ceiling whose one conversation exceeds any host (the fixture's
    /// 256 history bytes per token make two billion tokens 512 GB) is served
    /// at the largest context below it that fits: it fits there, and the
    /// next larger does not.
    #[test]
    fn the_served_context_is_the_largest_that_fits() {
        const CEILING: u32 = 2_000_000_000;
        let environment = environment(u64::from(CEILING));
        let prepared =
            prepare_execution_assessment(&environment.definition, &environment.draft, CEILING)
                .unwrap();
        let mut contexts = vec![CEILING];
        let mut tier = 1u32 << 30;
        while tier >= 32_768 {
            contexts.push(tier);
            tier /= 2;
        }
        let request = tiered(CEILING, &contexts, &[25_000]);
        let fits = |context: u32| {
            let device = planned_executor(&environment.draft, &environment.topology).unwrap();
            prepared
                .memory_charge_at(context)
                .unwrap()
                .assess_fit(&environment.topology, device, &environment.status, &request.reserves)
                .unwrap()
                .verdict
                == AssessmentFitVerdict::Fits
        };
        let ExecutionAssessment::Fits(ExecutionFit {
            fit_context_tokens, ..
        }) = assess_execution_fit(
            &prepared,
            &environment.draft,
            &environment.topology,
            &environment.status,
            &request,
        )
        .unwrap()
        else {
            panic!("the fixture fits every supported host at 32K");
        };
        assert!(fit_context_tokens < CEILING);
        assert!(fits(fit_context_tokens));
        let larger = contexts
            .iter()
            .rev()
            .find(|context| **context > fit_context_tokens)
            .copied()
            .unwrap();
        assert!(!fits(larger));
    }

    /// When no served context fits, the shortfall is the smallest's.
    #[test]
    fn a_shortfall_is_reported_at_the_smallest_context() {
        const CEILING: u32 = 2_000_000_000;
        let environment = environment(u64::from(CEILING));
        let prepared =
            prepare_execution_assessment(&environment.definition, &environment.draft, CEILING)
                .unwrap();
        let ExecutionAssessment::DoesNotFit(shortfall) = assess_execution_fit(
            &prepared,
            &environment.draft,
            &environment.topology,
            &environment.status,
            &tiered(CEILING, &[CEILING, 1_900_000_000], &[]),
        )
        .unwrap()
        else {
            panic!("512 GB of history fits no host");
        };
        assert_eq!(shortfall.fit_context_tokens, 1_900_000_000);
        let device = planned_executor(&environment.draft, &environment.topology).unwrap();
        let AssessmentFitVerdict::DoesNotFit { deficit_bytes, .. } = prepared
            .memory_charge_at(1_900_000_000)
            .unwrap()
            .assess_fit(
                &environment.topology,
                device,
                &environment.status,
                &MemoryReserves::standard(),
            )
            .unwrap()
            .verdict
        else {
            panic!("expected a shortfall");
        };
        assert_eq!(shortfall.deficit_bytes, deficit_bytes);
    }

    /// The charge at a served context below the ceiling is exactly what a
    /// load prepared at that context charges. It need not grow with the
    /// context: a smaller one retains more prefixes, whose address tables
    /// the stores commit at startup.
    #[test]
    fn a_served_context_charge_bounds_a_load_at_that_context() {
        let ceiling = prepare_execution_assessment(
            &environment(262_144).definition,
            &environment(262_144).draft,
            262_144,
        )
        .unwrap();
        for context in [32_768u32, 65_536, 131_072, 262_144] {
            let at = environment(u64::from(context));
            let direct = *prepare_execution_assessment(&at.definition, &at.draft, context)
                .unwrap()
                .memory_charge();
            let tiered = ceiling.memory_charge_at(context).unwrap();
            assert_eq!(tiered, direct, "{context}");
            // Prefix retention is bounded by the same conversation at the
            // served context, not at the ceiling.
            let state = ceiling.state.at_context(ceiling.method, u64::from(context)).unwrap();
            assert_eq!(
                state.capacity().workload_state_bytes,
                state
                    .fit_state_bytes(u64::from(context), ceiling.terms.recurrent_banks)
                    .unwrap()
            );
            assert_eq!(tiered.state_bytes, state.capacity().workload_state_bytes);
        }
        assert!(ceiling.memory_charge_at(262_145).is_err());
        assert!(ceiling.memory_charge_at(0).is_err());
    }

    #[test]
    fn served_contexts_must_decrease() {
        let environment = environment(65_536);
        let prepared =
            prepare_execution_assessment(&environment.definition, &environment.draft, 65_536)
                .unwrap();
        for contexts in [&[][..], &[32_768, 65_536][..], &[65_536, 65_536][..]] {
            assert!(matches!(
                assess_execution_fit(
                    &prepared,
                    &environment.draft,
                    &environment.topology,
                    &environment.status,
                    &tiered(65_536, contexts, &[]),
                ),
                Err(AssessmentError::Plan(_))
            ));
        }
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
