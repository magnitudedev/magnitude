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
use super::misses::HeldSlots;
use super::AssessmentError;
use crate::platform::{DomainRole, MemoryReserves};
use crate::{
    AssessmentFitVerdict, AssessmentGraphResourceBounds, AssessmentHeaderBounds,
    AssessmentMemoryCharge, AssessmentMemoryTerms, DeviceProcessBytes, ExecutionPlanDraft,
    HostProcessBytes, ResourceCapacity, ResourcePlanner,
};
use magnitude_family_contracts::ModelDefinition;
use seismic::{DeviceTopology, Executor, Status, StoreId};
use std::num::NonZeroU64;

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

/// The standard workload fits every domain the load touches, held as
/// `residency` says.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionFit {
    pub fit_context_tokens: u32,
    pub domains: Vec<DomainFit>,
    pub residency: Residency,
}

/// How a fitting plan holds its routed experts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Residency {
    /// Every weight in memory.
    Resident,
    /// The routed experts streamed through an expert bank.
    Streamed(ExpertShares),
}

impl Residency {
    /// How `draft` holds its routed experts: a bank of every expert holds
    /// them as a resident load does.
    pub fn of(draft: &ExecutionPlanDraft) -> Self {
        match draft.load().expert_bank().filter(|bank| !bank.whole()) {
            None => Self::Resident,
            Some(bank) => Self::Streamed(ExpertShares::new(
                bank.floor(),
                bank.slots,
                bank.widest(),
                match bank.host {
                    crate::planning::HostTier::Linked(_) if bank.host_slots > 0 => ColdSource::HostTier { slots: bank.host_slots },
                    crate::planning::HostTier::Linked(_) => ColdSource::File(FileOnly::NoHostRoom),
                    crate::planning::HostTier::Unified => ColdSource::File(FileOnly::Unified),
                },
            )),
        }
    }
}

/// A streamed plan's expert bank: in every routed layer, `slots` of its
/// `experts` experts held, of which a load holds at least `floor`. A bank
/// keeps the same slots in every layer, so the shares are per layer; of a
/// model whose layers route over different counts, `experts` is the widest
/// layer's, whose share is the least. Where the experts the bank lacks come
/// from is `cold` ([`ColdSource`]). A streamed bank holds at least its floor
/// and fewer than every expert: `floor ≤ slots < experts`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertShares {
    floor: u64,
    slots: u64,
    experts: NonZeroU64,
    cold: ColdSource,
}

/// Where a streamed plan's bank finds the experts it lacks, as its plan
/// holds them (`ExpertBankPlan::host`, `host_slots`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ColdSource {
    /// A linked host tier of host RAM holding `slots` of each layer's
    /// experts at load (inclusive of the bank's), as many as host memory
    /// admits, growing as it lets it; the rest read from the model file.
    /// The floor's speed (`floor_performance`) is priced beside it.
    HostTier { slots: u64 },
    /// The model file, read on demand.
    File(FileOnly),
}

/// Why a streamed plan reads every cold expert from the model file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FileOnly {
    /// Unified memory: device storage is host RAM, so the bank is the one
    /// store, and its misses land from the file straight into it.
    Unified,
    /// A discrete device's linked tier that host memory admits no slot of
    /// beside the load.
    NoHostRoom,
}

impl ExpertShares {
    /// The shares of a bank of `slots` of `experts` experts, at least
    /// `floor`. A plan whose bank breaks `floor ≤ slots < experts` is a
    /// planning defect.
    pub fn new(floor: u64, slots: u64, experts: NonZeroU64, cold: ColdSource) -> Self {
        assert!(
            floor <= slots && slots < experts.get(),
            "a streamed bank holds at least its floor and fewer than every expert: {floor} ≤ {slots} < {experts}"
        );
        Self { floor, slots, experts, cold }
    }

    /// The slots a load holds at least.
    pub fn floor(&self) -> u64 {
        self.floor
    }

    /// The slots the plan holds.
    pub fn slots(&self) -> u64 {
        self.slots
    }

    /// The widest routed layer's experts.
    pub fn experts(&self) -> NonZeroU64 {
        self.experts
    }

    /// Where the experts the bank lacks come from.
    pub fn cold(&self) -> &ColdSource {
        &self.cold
    }

    /// What the plan holds of each routed layer's experts at `slots` bank
    /// slots: those, and its host tier as the plan sizes it at load and
    /// fills before decode (every expert where the host's memory holds
    /// them, the tier then complete; none without a tier). The holding a
    /// plan's estimate is of; a running load's is what it holds.
    pub fn held(&self, slots: u64) -> HeldSlots {
        let count = |value: u64| usize::try_from(value).expect("a layer's experts fit the host");
        let host = match self.cold {
            ColdSource::HostTier { slots } => slots,
            ColdSource::File(_) => 0,
        };
        HeldSlots { slots: count(slots), host: count(host) }
    }

    /// The share of each routed layer's experts a load holds at least.
    pub fn floor_share(&self) -> f64 {
        self.floor as f64 / self.experts.get() as f64
    }

    /// The share of each routed layer's experts the plan holds.
    pub fn share(&self) -> f64 {
        self.slots as f64 / self.experts.get() as f64
    }
}

/// The standard workload exceeds some domain; `limiting` is the one with
/// the largest deficit.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionShortfall {
    pub fit_context_tokens: u32,
    pub domains: Vec<DomainFit>,
    pub limiting: StoreId,
    pub deficit_bytes: NonZeroU64,
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
/// measured: the memory charge ([`execution_memory_charge`]) and the plain
/// decode step's demand ([`decode_demand`]).
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
    let (charge, fit_context_tokens) = charged(definition, draft, None)?;
    Ok(PreparedExecutionAssessment {
        demand: decode_demand(definition, draft)?,
        fit_context_tokens,
        charge,
    })
}

/// The standard workload's checked memory charge of `definition` planned
/// as `draft`, at the model's context. It builds every graph a load
/// prepares, so this fails with `AssessmentError::Graph` exactly when the
/// model's program cannot be built on the planned backend. What that
/// derives is kept in `cache`, where one is given
/// ([`AssessmentGraphResourceBounds::kept`]).
pub fn execution_memory_charge(
    definition: &ModelDefinition,
    draft: &ExecutionPlanDraft,
    cache: Option<&crate::KernelCache>,
) -> Result<AssessmentMemoryCharge, AssessmentError> {
    charged(definition, draft, cache).map(|(charge, _)| charge)
}

/// [`execution_memory_charge`] and the workload's fit depth in tokens.
fn charged(
    definition: &ModelDefinition,
    draft: &ExecutionPlanDraft,
    cache: Option<&crate::KernelCache>,
) -> Result<(AssessmentMemoryCharge, u32), AssessmentError> {
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
    let graph = AssessmentGraphResourceBounds::kept(
        cache,
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
        .with_graph_resource_bound(&graph, draft.device().graph_cost(), host_process, device_process)
        .and_then(|bounds| {
            terms.charge(
                bounds,
                state.capacity().workload_state_bytes,
                state.startup_state_bytes()?,
                state.loaded_state_bytes()?,
            )
        })
        .map_err(AssessmentError::Memory)?;
    Ok((charge, fit_context_tokens))
}

/// The plain decode step's demand of `definition` planned as `draft`,
/// under its planned history codec: over the kernels a load prepares, at
/// the statics it prepares them with, which each call's launches are
/// declared over.
pub fn decode_demand(definition: &ModelDefinition, draft: &ExecutionPlanDraft) -> Result<DecodeDemand, AssessmentError> {
    let policy = draft.policy();
    let declarations = Declarations::of_load(
        draft.device().backend(),
        draft.programs(),
        definition,
        draft.load(),
        crate::native::TuningLimits::of(policy.limits(), definition),
    )?;
    DecodeDemand::from_model(definition, draft.load(), policy.codec(), &declarations)
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

/// Whether the standard workload of one prepared model of `definition` fits
/// stable capacity. A plan that does not fit holds its experts as a load
/// does ([`ExecutionPlanDraft::with_held_experts`]) within the allocation
/// domain's stable capacity, and fits so; otherwise the plan's own shortfall
/// is the verdict. No graph or model material is constructed here.
pub fn assess_execution_fit(
    definition: &ModelDefinition,
    prepared: &PreparedExecutionAssessment,
    draft: &ExecutionPlanDraft,
    topology: &DeviceTopology,
    status: &Status,
    request: &AssessmentRequest,
) -> Result<ExecutionAssessment, AssessmentError> {
    let device = planned_executor(draft, topology)?;
    let fit_of = |charge: &AssessmentMemoryCharge| {
        charge
            .assess_fit(topology, device, status, &request.reserves)
            .map_err(AssessmentError::Memory)
    };
    let fit = fit_of(&prepared.charge)?;
    let (limiting, deficit_bytes) = match fit.verdict {
        AssessmentFitVerdict::Fits => {
            return Ok(ExecutionAssessment::Fits(ExecutionFit {
                fit_context_tokens: prepared.fit_context_tokens,
                domains: fit.domains,
                residency: Residency::of(draft),
            }));
        }
        AssessmentFitVerdict::DoesNotFit {
            limiting,
            deficit_bytes,
        } => (limiting, deficit_bytes),
    };
    let allocation = fit
        .domains
        .iter()
        .find(|domain| domain.role == DomainRole::Allocation)
        .ok_or_else(|| AssessmentError::Memory("the fit reaches no allocation domain".into()))?;
    let bound = allocation.capacity_bytes - allocation.reserve_bytes;
    let confirm = |candidate: &ExecutionPlanDraft| {
        let charge = execution_memory_charge(definition, candidate, None)?;
        let fit = fit_of(&charge)?;
        Ok::<_, AssessmentError>(match fit.verdict {
            AssessmentFitVerdict::Fits => Ok((charge, (prepared.fit_context_tokens, fit.domains))),
            AssessmentFitVerdict::DoesNotFit { deficit_bytes, .. } => Err(deficit_bytes),
        })
    };
    Ok(match draft.with_held_experts(definition, bound, confirm)? {
        Ok((held, (fit_context_tokens, domains))) => ExecutionAssessment::Fits(ExecutionFit {
            fit_context_tokens,
            domains,
            residency: Residency::of(&held),
        }),
        Err(_) => ExecutionAssessment::DoesNotFit(ExecutionShortfall {
            fit_context_tokens: prepared.fit_context_tokens,
            domains: fit.domains,
            limiting,
            deficit_bytes,
        }),
    })
}

/// What the draft's decode is priced with on `costs`: its resolved
/// lookahead policy.
pub fn decode_pricing<'a>(draft: &ExecutionPlanDraft, costs: &'a ExecutorCosts) -> Pricing<'a> {
    Pricing {
        costs,
        lookahead: draft.policy().limits().flights().pipelines(),
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
        ComponentSelection, ExecutionPath, ExecutionPlanner, PlannedMethod,
    };
    use magnitude_state::KvCodec;

    /// The limits the tests plan with.
    pub(crate) use crate::planning::fixtures::LIMITS;

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
        let mut definition = crate::planning::fixtures::fixture_definition();
        definition.decoder.context_limit = context_limit;
        let manifest = crate::planning::fixtures::fixture_manifest(&definition);
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
            &environment.definition,
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
            residency: Residency::Resident,
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
            &environment.definition,
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
