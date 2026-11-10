//! One complete model assessment, from headers and the allocation-free
//! execution plan draft, as two questions with one owner each: whether the
//! standard workload fits ([`assess_execution_fit`]: the clean-load charge
//! against every domain the load touches, `DoesNotFit` naming the domain
//! with the largest deficit), and how fast a fitting model decodes
//! ([`estimate_execution`]: its per-call decode demand over the executor's
//! measured kernel costs).

use super::demand::DecodeDemand;
use super::estimate::{estimate_decode, form_seconds, launch_seconds, performance_depths, PerformanceEstimate, Pricing};
use super::executor_costs::ExecutorCosts;
use super::launches::Declarations;
use super::misses::HeldSlots;
use super::AssessmentError;
use crate::memory::ClaimDemand;
use crate::platform::{DomainRole, MemoryReserves};
use crate::{
    AssessmentFitVerdict, AssessmentGraphResourceBounds, AssessmentHeaderBounds,
    AssessmentMemoryCharge, AssessmentMemoryTerms, ChargedDraft, DeviceProcessBytes, ExecutionPlanDraft, HeldCharge, HeldCharges,
    HostProcessBytes, ResourceCapacity, ResourcePlanner, UndraftedBounds,
};
use magnitude_family_contracts::ModelDefinition;
use seismic::{DeviceTopology, Executor, Status, StoreId};
use std::num::NonZeroU64;

/// Engine inputs to one assessment. No service profile or model identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssessmentRequest {
    /// The context the model is served at: its supported maximum or a
    /// served tier below it, the depth of the standard workload's
    /// conversations and the deepest decode estimated.
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

/// What the standard workload's required bytes hold, summed across every
/// domain the load touches (`AssessmentMemoryCharge::breakdown`): the
/// weights as the plan holds them (of a streamed plan, its bank and host
/// tier), the conversation's state at the served context, and everything
/// else the load holds (prepared programs and graphs, workspace, process
/// bytes and the loading peak).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBreakdown {
    pub weights_bytes: u64,
    pub context_bytes: u64,
    pub overhead_bytes: u64,
}

/// Where a fitting plan places the model on this machine before any load,
/// by tier, from the plan and its fit. `device_bytes` is what the fit
/// requires of the serving device's memory (its allocation domains: GPU
/// memory, or unified memory on Apple silicon), and `host_bytes` what the
/// plan holds in host RAM beside a discrete device: what the fit requires
/// of its staging domains (host tables' row caches, graphs, the process and
/// a streamed plan's host tier at its landing reserve), and the host tier's
/// slots beyond that reserve, which grow into the host memory the fit
/// leaves (`ModelLoadPlan::host_tier_bytes`); none on unified memory, where
/// host RAM is the device's memory. The two sum to the fit's required
/// bytes, its [`MemoryBreakdown`]'s weights, context and overhead, and that
/// growth. `disk_bytes` is what the plan reads from
/// the model file as steps name it rather than holding it
/// (`ModelLoadPlan::unheld_bytes`): the routed experts neither its bank nor
/// its host tier holds, and the rows of side tables (n-gram tables) beyond
/// their slots and row caches, as stored. A plan that holds every weight
/// whole reads none from disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelPlacement {
    pub device_bytes: u64,
    pub host_bytes: u64,
    pub disk_bytes: u64,
}

impl ModelPlacement {
    /// The placement of `load`, whose fit reached `domains`.
    pub fn of(load: &crate::ModelLoadPlan, domains: &[DomainFit]) -> Result<Self, String> {
        let required = |role: DomainRole| {
            domains
                .iter()
                .filter(|domain| domain.role == role)
                .try_fold(0u64, |total, domain| total.checked_add(domain.required_bytes))
                .ok_or_else(|| "required bytes overflow".to_owned())
        };
        Ok(Self {
            device_bytes: required(DomainRole::Allocation)?,
            host_bytes: required(DomainRole::Staging)?
                .checked_add(load.host_tier_bytes()?)
                .ok_or_else(|| "host bytes overflow".to_owned())?,
            disk_bytes: load.unheld_bytes()?,
        })
    }
}

/// Whether the standard workload fits, each verdict with its own evidence.
#[derive(Clone, Debug, PartialEq)]
pub enum ExecutionAssessment {
    Fits(ExecutionFit),
    DoesNotFit(ExecutionShortfall),
}

/// The standard workload fits every domain the load touches at the served
/// context `fit_context_tokens`, requiring `breakdown` across them, placed
/// by tier as `placement`, the plan's routed experts held in `bank`, whole
/// or streamed (none for a plan that banks no experts).
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionFit {
    pub fit_context_tokens: u32,
    pub domains: Vec<DomainFit>,
    pub breakdown: MemoryBreakdown,
    pub placement: ModelPlacement,
    pub bank: Option<crate::planning::ExpertBankPlan>,
}

impl ExecutionFit {
    /// How the fit holds its routed experts ([`Residency::of_bank`]).
    pub fn residency(&self) -> Residency {
        Residency::of_bank(self.bank.as_ref())
    }
}

/// How a fitting plan holds its routed experts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Residency {
    /// Every weight in memory.
    Resident,
    /// The routed experts streamed through this expert bank, the bank of
    /// the plan that fits: what the fit classed streamed is what its
    /// shares ([`ExpertShares::of`]) and speed are read from.
    Streamed(crate::planning::ExpertBankPlan),
}

impl Residency {
    /// How `draft` holds its routed experts: a bank of every expert holds
    /// them as a resident load does.
    pub fn of(draft: &ExecutionPlanDraft) -> Self {
        Self::of_bank(draft.load().expert_bank())
    }

    /// How a plan holding `bank` holds its routed experts: no bank, or one
    /// of every expert, holds them as a resident load does.
    pub fn of_bank(bank: Option<&crate::planning::ExpertBankPlan>) -> Self {
        match bank.filter(|bank| !bank.whole()) {
            None => Self::Resident,
            Some(bank) => Self::Streamed(bank.clone()),
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
    /// The shares of a streamed `bank`, which holds fewer than every
    /// expert: its floor, its slots and where its cold experts lie.
    pub fn of(bank: &crate::planning::ExpertBankPlan) -> Self {
        Self::new(
            bank.floor(),
            bank.slots,
            bank.widest(),
            match bank.host() {
                crate::planning::HostTier::Linked(_) if bank.host_slots > 0 => ColdSource::HostTier { slots: bank.host_slots },
                crate::planning::HostTier::Linked(_) => ColdSource::File(FileOnly::NoHostRoom),
                crate::planning::HostTier::Unified => ColdSource::File(FileOnly::Unified),
            },
        )
    }

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
    /// Of a plan that banks its experts, its demand holding them whole: the
    /// resident form its decode runs while its bank holds every expert. None
    /// of a plan that holds them whole, whose own demand is that form.
    whole: Option<DecodeDemand>,
    fit_context_tokens: u32,
    charge: AssessmentMemoryCharge,
    /// The charges planning's search made, which a fit's search of the
    /// draft charges from again ([`assess_execution_fit`]).
    held: HeldCharges,
}

impl PreparedExecutionAssessment {
    /// The plain decode step's demand under the planned history codec.
    pub fn demand(&self) -> &DecodeDemand {
        &self.demand
    }

    /// The plan's decode step at `depth` over `sequences` sequences in each
    /// form ([`form_seconds`]): its experts held whole, and as the plan holds
    /// them. A plan prepared whole has no streamed form of its own: a fit
    /// that streams it plans its bank anew (`assess_execution_fit`), and its
    /// streamed form is priced as its resident one.
    pub fn form_seconds(&self, pricing: &Pricing<'_>, sequences: u64, depth: u32) -> Result<super::FormSeconds, AssessmentError> {
        form_seconds(self.whole.as_ref().unwrap_or(&self.demand), &self.demand, pricing, sequences, depth)
    }

    /// The checked header and graph charge, shared with physical finalization.
    pub fn memory_charge(&self) -> &AssessmentMemoryCharge {
        &self.charge
    }
}

/// Perform model-specific arithmetic while the fixed generic basis is
/// measured: the plain decode step's demand ([`decode_demand`]), and of a
/// plan that banks its experts its demand holding them whole, beside the
/// memory charge planning confirmed of the draft ([`ChargedDraft`]).
pub fn prepare_execution_assessment<C: HeldCharge>(
    definition: &ModelDefinition,
    planned: &ChargedDraft<C>,
    context_limit: u32,
) -> Result<PreparedExecutionAssessment, AssessmentError> {
    if definition.decoder.context_limit != u64::from(context_limit) {
        return Err(AssessmentError::Plan(format!(
            "assessed context limit {} differs from the planned model's {}",
            context_limit, definition.decoder.context_limit
        )));
    }
    let fit_context_tokens = u32::try_from(crate::planning::fit_depth(definition))
        .map_err(|_| AssessmentError::Memory("fit depth exceeds u32".into()))?;
    let draft = planned.draft();
    let whole = match draft.load().expert_bank() {
        Some(_) => Some(decode_demand(definition, &draft.held_whole(definition)?)?),
        None => None,
    };
    Ok(PreparedExecutionAssessment {
        demand: decode_demand(definition, draft)?,
        whole,
        fit_context_tokens,
        charge: planned.charge().clone(),
        held: planned.held().charges(),
    })
}

/// The standard workload's checked memory charge of `definition` planned
/// as `draft`, at the model's context ([`StructuralCharge`] of the draft,
/// charged at its own bank). It builds every graph a load prepares, so this
/// fails with `AssessmentError::Graph` exactly when the model's program
/// cannot be built on the planned backend.
pub fn execution_memory_charge(
    definition: &ModelDefinition,
    draft: &ExecutionPlanDraft,
) -> Result<AssessmentMemoryCharge, AssessmentError> {
    StructuralCharge::of(definition, draft)?.charge(definition, draft)
}

/// [`execution_memory_charge`], with the bounds its certification gives of
/// the draft's target alone ([`StructuralCharge::with_undrafted`]).
pub fn execution_memory_charge_with_undrafted(
    definition: &ModelDefinition,
    draft: &ExecutionPlanDraft,
) -> Result<(AssessmentMemoryCharge, UndraftedBounds), AssessmentError> {
    let (structure, undrafted) = StructuralCharge::with_undrafted(definition, draft)?;
    Ok((structure.charge(definition, draft)?, undrafted))
}

/// The standard workload's checked memory charge of the target's plan
/// alone, `plain`, at its drafted plan's holding of experts, whose target
/// and vision families are that plan's `undrafted` ones
/// ([`StructuralCharge::undrafted`]).
pub fn undrafted_memory_charge(
    definition: &ModelDefinition,
    plain: &ExecutionPlanDraft,
    undrafted: UndraftedBounds,
) -> Result<AssessmentMemoryCharge, AssessmentError> {
    StructuralCharge::undrafted(definition, plain, undrafted)?.charge(definition, plain)
}

/// The terms of a draft's charge that its expert bank's slots leave as they
/// are: its state plan's bytes, its graph resource bound, which builds and
/// certifies every graph a load prepares, and its process bytes. A bank's
/// slots size only its record tensors' committed records
/// (`ExpertBankPlan::records_at`) and the banked stacks' resident bytes; its
/// records' extents, binding, conveyor, scratch and host tier, which the
/// program plan, the graphs and the kernels read, are of its structure
/// alone (`BankStructure`). So the charge of any draft that differs from
/// this one's only in its bank's slots is this structure with that draft's
/// weights and header bounds ([`Self::charge`]), and is derived once for
/// every size a bank search confirms.
#[derive(Debug)]
pub struct StructuralCharge {
    graph: AssessmentGraphResourceBounds,
    host_process: HostProcessBytes,
    device_process: DeviceProcessBytes,
    workload_state_bytes: u64,
    startup_state_bytes: u64,
    loaded_state_bytes: u64,
}

impl StructuralCharge {
    /// The structure of `draft`'s charge.
    pub fn of(definition: &ModelDefinition, draft: &ExecutionPlanDraft) -> Result<Self, AssessmentError> {
        Self::with_graph(definition, draft, |state| {
            let policy = draft.policy();
            AssessmentGraphResourceBounds::derive(
                definition,
                draft.load(),
                state,
                policy.method(),
                policy.codec(),
                policy.limits(),
                draft.device().backend(),
            )
            .map(|graph| (graph, ()))
        })
        .map(|(structure, ())| structure)
    }

    /// [`Self::of`], with the bounds of the draft's target family without
    /// its drafter's writes ([`AssessmentGraphResourceBounds::derive_with_undrafted`]),
    /// which [`Self::undrafted`] charges the target's plan alone with.
    pub fn with_undrafted(
        definition: &ModelDefinition,
        draft: &ExecutionPlanDraft,
    ) -> Result<(Self, UndraftedBounds), AssessmentError> {
        Self::with_graph(definition, draft, |state| {
            let policy = draft.policy();
            AssessmentGraphResourceBounds::derive_with_undrafted(
                definition,
                draft.load(),
                state,
                policy.method(),
                policy.codec(),
                policy.limits(),
                draft.device().backend(),
            )
        })
    }

    /// The structure of `plain`'s charge, the target's plan alone at a
    /// drafted plan's holding of its experts
    /// (`ExecutionPlanDraft::holding_as`), whose target and vision families'
    /// bounds are the drafted plan's `undrafted` ([`Self::with_undrafted`]):
    /// every other family of `plain` is certified here, those not again.
    pub fn undrafted(
        definition: &ModelDefinition,
        plain: &ExecutionPlanDraft,
        undrafted: UndraftedBounds,
    ) -> Result<Self, AssessmentError> {
        Self::with_graph(definition, plain, |state| {
            let policy = plain.policy();
            AssessmentGraphResourceBounds::of_target(
                definition,
                plain.load(),
                plain.programs(),
                state,
                policy.method(),
                policy.limits(),
                plain.device().backend(),
                undrafted.target,
                undrafted.vision,
            )
            .map(|graph| (graph, ()))
        })
        .map(|(structure, ())| structure)
    }

    /// The structure of `draft`'s charge with the graph bounds `graph`
    /// derives from its state plan, and what else `graph` gives.
    fn with_graph<T>(
        definition: &ModelDefinition,
        draft: &ExecutionPlanDraft,
        graph: impl FnOnce(&crate::StateResourcePlan) -> Result<(AssessmentGraphResourceBounds, T), crate::GraphError>,
    ) -> Result<(Self, T), AssessmentError> {
        let policy = draft.policy();
        let backend = draft.device().backend();
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
        let (graph, more) = graph(&state).map_err(AssessmentError::Graph)?;
        let host_process =
            HostProcessBytes::derive(definition, draft.load(), draft.programs(), policy.limits(), backend)
                .map_err(AssessmentError::Memory)?;
        let device_process = draft
            .load()
            .device_allocations()
            .and_then(|allocations| DeviceProcessBytes::derive(backend, host_process.kernels, allocations))
            .map_err(AssessmentError::Memory)?;
        let state_bytes = || -> Result<_, String> {
            Ok((state.capacity().workload_state_bytes, state.startup_state_bytes()?, state.loaded_state_bytes()?))
        };
        let (workload_state_bytes, startup_state_bytes, loaded_state_bytes) =
            state_bytes().map_err(AssessmentError::Memory)?;
        Ok((
            Self { graph, host_process, device_process, workload_state_bytes, startup_state_bytes, loaded_state_bytes },
            more,
        ))
    }

    /// The charge of `draft`, a draft of this structure: its weights'
    /// terms and header bounds beside the structure's.
    pub fn charge(
        &self,
        definition: &ModelDefinition,
        draft: &ExecutionPlanDraft,
    ) -> Result<AssessmentMemoryCharge, AssessmentError> {
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
        AssessmentHeaderBounds::derive(draft.load())
            .and_then(|header| {
                header.with_graph_resource_bound(
                    &self.graph,
                    draft.device().graph_cost(),
                    self.host_process,
                    self.device_process,
                )
            })
            .and_then(|bounds| {
                terms.charge(
                    bounds,
                    self.workload_state_bytes,
                    self.startup_state_bytes,
                    self.loaded_state_bytes,
                    group_class(draft, &self.graph)?,
                )
            })
            .map_err(AssessmentError::Memory)
    }
}

/// What a prompt group's class holds beyond the floor's
/// (`AssessmentMemoryCharge::group_class`) at the places each pass size of
/// `draft`'s ladder merges, up to its chain places, as a group opening
/// grants it (`ExecutorDomain::grant_group`, `ExecutorFamily::group_demand`):
/// the graph pools', graphs' and target buffers' (`graph`), the bank's
/// placed words of its places and each streamed table's stagings beyond
/// the floor's.
fn group_class(
    draft: &ExecutionPlanDraft,
    graph: &crate::AssessmentGraphResourceBounds,
) -> Result<crate::planning::ByPass<ClaimDemand>, String> {
    let limits = draft.policy().limits();
    crate::planning::ByPass::of(crate::PassLaunches::whole_chain(limits), |pass| {
        let places = pass.get();
        let pools = graph.group_class(limits, draft.device().graph_cost(), pass)?;
        let bank = match (draft.load().expert_bank(), places) {
            (Some(bank), 2..) => bank.placed_demand(
                u64::try_from(places).ok().and_then(std::num::NonZeroU64::new).ok_or("a pass's places exceed u64")?,
            )?,
            _ => ClaimDemand::default(),
        };
        let stagings = draft.load().weights().try_fold(0u64, |bytes, weight| {
            let grown = weight.staging_bytes(places)?.checked_sub(weight.staging_bytes(1)?);
            grown.and_then(|grown| bytes.checked_add(grown)).ok_or_else(|| "a group's stagings overflow".to_owned())
        })?;
        pools
            .checked_plus(bank)
            .and_then(|demand| demand.checked_plus(ClaimDemand { device: stagings, ..Default::default() }))
            .ok_or_else(|| "a prompt group's class overflows".to_owned())
    })
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
/// stable capacity: its experts held as a load holds them
/// ([`ExecutionPlanDraft::with_held_experts`]), placed whole where the
/// allocation domain's stable room holds them so, else streamed within it;
/// where neither fits, the prepared plan's own shortfall is the verdict.
/// Its plans are charged from the charges planning made ([`HeldCharges`]),
/// so a way of holding the experts planning charged is never charged from
/// its structure again. On a machine whose prices are not measured a
/// streamed fit is sized by memory alone
/// ([`crate::Unmeasured::SizedByMemory`]): fit is assessed, and speed is
/// what is pending.
pub fn assess_execution_fit(
    definition: &ModelDefinition,
    prepared: &PreparedExecutionAssessment,
    draft: &ExecutionPlanDraft,
    topology: &DeviceTopology,
    status: &Status,
    request: &AssessmentRequest,
    prices: &crate::assessment::Priced<crate::PlanPrices>,
) -> Result<ExecutionAssessment, AssessmentError> {
    let device = planned_executor(draft, topology)?;
    // Every plan is confirmed as a load's plan of it is (`StableRoom::fit`),
    // and settles into the same room: the allocation domain's stable fit;
    // every other store `Fit` places the plan in is its confirm's.
    let room = crate::planning::StableRoom { topology, executor: device, status, reserves: &request.reserves };
    let fit = room.fit(draft, &prepared.charge)?;
    // A prepared plan whose experts are whole (a bank of every expert at
    // its largest pass, or their own allocations) that fits is the plan
    // placement holds first.
    if matches!(fit.verdict, AssessmentFitVerdict::Fits) && draft.load().expert_bank().is_none_or(|bank| bank.whole()) {
        return Ok(ExecutionAssessment::Fits(ExecutionFit {
            fit_context_tokens: prepared.fit_context_tokens,
            breakdown: prepared.charge.breakdown(&fit.domains).map_err(AssessmentError::Memory)?,
            placement: ModelPlacement::of(draft.load(), &fit.domains).map_err(AssessmentError::Memory)?,
            domains: fit.domains,
            bank: draft.load().expert_bank().cloned(),
        }));
    }
    let bound = fit.settle_bound().map_err(AssessmentError::Memory)?;
    let confirm = |candidate: &ExecutionPlanDraft, charge: &AssessmentMemoryCharge| {
        let fit = room.fit(candidate, charge)?;
        Ok::<_, AssessmentError>(match fit.verdict {
            AssessmentFitVerdict::Fits => {
                Ok((charge.breakdown(&fit.domains).map_err(AssessmentError::Memory)?, fit.domains))
            }
            AssessmentFitVerdict::DoesNotFit { deficit_bytes, .. } => Err(deficit_bytes),
        })
    };
    let mut held = prepared.held.clone();
    Ok(match draft.with_held_experts(definition, crate::HeldBound::room(bound), prices, crate::Unmeasured::SizedByMemory, &mut held, room, confirm)? {
        Ok((held, (breakdown, domains))) => ExecutionAssessment::Fits(ExecutionFit {
            fit_context_tokens: prepared.fit_context_tokens,
            placement: ModelPlacement::of(held.draft().load(), &domains).map_err(AssessmentError::Memory)?,
            domains,
            breakdown,
            bank: held.draft().load().expert_bank().cloned(),
        }),
        Err(_) => match fit.verdict {
            AssessmentFitVerdict::DoesNotFit { limiting, deficit_bytes } => ExecutionAssessment::DoesNotFit(ExecutionShortfall {
                fit_context_tokens: prepared.fit_context_tokens,
                domains: fit.domains,
                limiting,
                deficit_bytes,
            }),
            AssessmentFitVerdict::Fits => {
                return Err(AssessmentError::Plan("the prepared plan fits, yet no holding of its experts does".into()));
            }
        },
    })
}

/// A banked plan's launch of some rows at some depth in each form
/// ([`crate::DecodeStep`]): `resident`'s demand, the plan holding its
/// experts whole, and `banked`'s, the plan as it banks them, whose streamed
/// layers decode in their streamed form ([`decode_demand`]), each on
/// `costs` under `resident`'s lookahead, which the two share; a decode step
/// within the decode row classes, a prefill launch beyond
/// ([`launch_seconds`]). Each launch is priced once: the step is a function
/// of its rows and depth alone, so a launch priced again reads what it
/// priced the first time.
pub fn decode_step(
    definition: &ModelDefinition,
    resident: &ExecutionPlanDraft,
    banked: &ExecutionPlanDraft,
    costs: ExecutorCosts,
) -> Result<Box<crate::DecodeStep>, AssessmentError> {
    let (whole, streamed) = (decode_demand(definition, resident)?, decode_demand(definition, banked)?);
    let lookahead = decode_pricing(resident, &costs).lookahead;
    let priced = std::cell::RefCell::new(std::collections::HashMap::new());
    Ok(Box::new(move |rows, depth| {
        if let Some(&seconds) = priced.borrow().get(&(rows, depth)) {
            return Ok(seconds);
        }
        let pricing = Pricing { costs: &costs, lookahead };
        let launch = |demand| launch_seconds(demand, &pricing, u64::from(rows), depth).map(|launch| launch.total());
        let seconds = super::FormSeconds { resident: launch(&whole)?, streamed: launch(&streamed)? };
        priced.borrow_mut().insert((rows, depth), seconds);
        Ok(seconds)
    }))
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

    impl Environment {
        /// The draft with its charge, as planning gives it.
        fn planned(&self) -> ChargedDraft {
            ChargedDraft::of(&self.definition, self.draft.clone()).unwrap()
        }
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
            &MemoryReserves::of_machine(None),
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
            reserves: MemoryReserves::of_machine(None),
        }
    }

    fn assess(environment: &Environment, request: &AssessmentRequest) -> ExecutionAssessment {
        let prepared = prepare_execution_assessment(
            &environment.definition,
            &environment.planned(),
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
            &crate::assessment::Priced::Measured(crate::assessment::synthetic::plan_prices()),
        )
        .unwrap()
    }

    #[test]
    fn fit_depth_is_independent_of_performance_depths() {
        let environment = environment(150_000);
        let ExecutionAssessment::Fits(fit) = assess(
            &environment,
            &request(150_000, &[25_000, 50_000, 50_000, 200_000]),
        )
        else {
            panic!("the fixture fits every supported host");
        };
        assert_eq!(fit.residency(), Residency::Resident);
        let ExecutionFit { fit_context_tokens, domains, breakdown, placement, .. } = fit;
        assert_eq!(fit_context_tokens, 150_000);
        // A resident plan of no side table reads nothing from disk, and
        // its tiers hold what its domains require.
        assert_eq!(placement.disk_bytes, 0);
        assert_eq!(
            placement.device_bytes + placement.host_bytes,
            breakdown.weights_bytes + breakdown.context_bytes + breakdown.overhead_bytes
        );
        assert!(breakdown.weights_bytes > 0 && breakdown.context_bytes > 0);
        assert_eq!(
            breakdown.weights_bytes + breakdown.context_bytes + breakdown.overhead_bytes,
            domains.iter().map(|domain| domain.required_bytes).sum::<u64>()
        );
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
            prepare_execution_assessment(&environment.definition, &environment.planned(), 150_000)
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
            prepare_execution_assessment(&environment.definition, &environment.planned(), 150_000)
                .expect("a context the domain cannot back is not a planning failure");
        assert_eq!(prepared.fit_context_tokens, 150_000);
        assert!(assess_execution_fit(
            &environment.definition,
            &prepared,
            &environment.draft,
            &environment.topology,
            &environment.status,
            &request,
            &crate::assessment::Priced::Measured(crate::assessment::synthetic::plan_prices()),
        )
        .is_ok());
    }

    #[test]
    fn a_mismatched_context_limit_is_an_error() {
        let environment = environment(128);
        assert!(matches!(
            prepare_execution_assessment(&environment.definition, &environment.planned(), 256),
            Err(AssessmentError::Plan(_))
        ));
    }
}
