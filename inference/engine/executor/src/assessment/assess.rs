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
use crate::platform::{DomainRole, MemoryReserves};
use crate::{
    AssessmentFitVerdict, AssessmentGraphResourceBounds, AssessmentHeaderBounds,
    AssessmentMemoryCharge, AssessmentMemoryTerms, DeviceProcessBytes, ExecutionPlanDraft,
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

/// How a fitting plan holds the model's weights as it runs, as its files
/// store them, so the four sum to the weights' stored bytes: `device_bytes`
/// held in the serving device's memory (GPU memory, or unified memory on
/// Apple silicon), `host_bytes` held in host RAM beside a discrete device,
/// `disk_bytes` the routed experts neither holds, which its misses stream
/// from the model file, and `lookup_disk_bytes` the rows of tables read by
/// row (an embedding) that the OS cannot keep in host RAM beside the plan, so
/// steps look up their few rows on disk. Rows host RAM has room for are held
/// there, in the OS file cache (`ModelLoadPlan::weight_placement`). What the
/// plan holds beside its weights (the conversation's state, graphs, buffers)
/// is its [`MemoryBreakdown`]'s, not its placement's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelPlacement {
    pub device_bytes: u64,
    pub host_bytes: u64,
    pub disk_bytes: u64,
    pub lookup_disk_bytes: u64,
}

impl ModelPlacement {
    /// The placement of `load`'s weights in a fit of `domains`: its table
    /// rows no slot holds go where the OS file cache holds them, in host RAM
    /// when its remaining room takes them all (the device's memory, of
    /// unified memory), else on disk.
    pub fn of(load: &crate::ModelLoadPlan, domains: &[DomainFit], topology: &DeviceTopology) -> Result<Self, String> {
        let crate::WeightPlacement { held, host, streamed, tables } = load.weight_placement()?;
        let host_ram = domains.iter().find(|domain| {
            topology.store(domain.domain).is_some_and(|store| matches!(store.kind(), seismic::StoreKind::HostRam))
        });
        let cached = host_ram.is_none_or(|domain| u64::try_from(domain.remaining_bytes).is_ok_and(|room| room >= tables));
        let overflow = || "placement bytes overflow".to_owned();
        Ok(match (cached, host_ram.map(|domain| domain.role)) {
            (false, _) => Self { device_bytes: held, host_bytes: host, disk_bytes: streamed, lookup_disk_bytes: tables },
            (true, Some(DomainRole::Allocation)) => {
                Self { device_bytes: held.checked_add(tables).ok_or_else(overflow)?, host_bytes: host, disk_bytes: streamed, lookup_disk_bytes: 0 }
            }
            (true, _) => Self { device_bytes: held, host_bytes: host.checked_add(tables).ok_or_else(overflow)?, disk_bytes: streamed, lookup_disk_bytes: 0 },
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
    /// `cache_bytes` is what the tier holds past its floor
    /// (`ExpertBankPlan::host_cache`): cache the load grows into free host
    /// RAM, which no store's required bytes count, as the fit charges the
    /// tier at its floor.
    HostTier { slots: u64, cache_bytes: u64 },
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
    /// A linked tier's cache was computed when the plan sized the tier at
    /// its slots, so computing it again cannot fail.
    pub fn of(bank: &crate::planning::ExpertBankPlan) -> Self {
        Self::new(
            bank.floor(),
            bank.slots,
            bank.widest(),
            match bank.host() {
                crate::planning::HostTier::Linked(_) if bank.host_slots > 0 => {
                    let cache_bytes = bank
                        .host_cache(bank.host_slots)
                        .expect("a planned host tier's cache was computed when the plan sized it");
                    ColdSource::HostTier { slots: bank.host_slots, cache_bytes }
                }
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
    /// slots: those, and its host tier as the plan sizes it at load and it
    /// fills (every expert where the host's memory holds them, the tier
    /// then complete; none without a tier). The holding a plan's estimate
    /// is of at steady state; a running load's is what it holds.
    pub fn held(&self, slots: u64) -> HeldSlots {
        let count = |value: u64| usize::try_from(value).expect("a layer's experts fit the host");
        let host = match self.cold {
            ColdSource::HostTier { slots, .. } => slots,
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

/// Basis-independent decode demand and the structure of the standard
/// workload's memory charge in each way of holding the experts
/// ([`crate::FloorForms`]), at the model's largest served context: every
/// fit is arithmetic over them ([`assess_execution_fit`]).
pub struct PreparedExecutionAssessment {
    demand: DecodeDemand,
    /// Of a plan that banks its experts, its demand holding them whole: the
    /// resident form its decode runs while its bank holds every expert. None
    /// of a plan that holds them whole, whose own demand is that form.
    whole: Option<DecodeDemand>,
    forms: crate::FloorForms,
}

impl PreparedExecutionAssessment {
    /// The plain decode step's demand under the planned history codec.
    pub fn demand(&self) -> &DecodeDemand {
        &self.demand
    }

    /// The form the demand and structure are read from
    /// ([`crate::FloorForms::primary`]).
    pub fn primary(&self) -> &crate::HeldForm {
        self.forms.primary()
    }

    /// The primary form's plan ([`Self::primary`]).
    pub fn draft(&self) -> &ExecutionPlanDraft {
        &self.primary().draft
    }

    /// The plan's decode step at `depth` over `sequences` sequences in each
    /// form ([`form_seconds`]): its experts held whole, and as the plan holds
    /// them.
    pub fn form_seconds(&self, pricing: &Pricing<'_>, sequences: u64, depth: u32) -> Result<super::FormSeconds, AssessmentError> {
        form_seconds(self.whole.as_ref().unwrap_or(&self.demand), &self.demand, pricing, sequences, depth)
    }

    /// The charge of the primary plan at `definition`'s context, its
    /// recorded graphs priced at `pricing` (the machine's graph cost as its
    /// records give it now, which the preparation does not read): what a
    /// load of it holds, shared with physical finalization.
    pub fn memory_charge(
        &self,
        definition: &ModelDefinition,
        pricing: crate::platform::GraphPricing,
    ) -> Result<AssessmentMemoryCharge, AssessmentError> {
        let primary = self.primary();
        primary.structure.charge(definition, &primary.draft.clone().with_graph_pricing(pricing))
    }
}

/// Perform model-specific arithmetic while the fixed generic basis is
/// measured: the plain decode step's demand ([`decode_demand`]) of the
/// forms' primary plan, and of a plan that banks its experts its demand
/// holding them whole, beside the forms' charge structures.
pub fn prepare_execution_assessment(
    definition: &ModelDefinition,
    forms: crate::FloorForms,
) -> Result<PreparedExecutionAssessment, AssessmentError> {
    let draft = &forms.primary().draft;
    let whole = match draft.load().expert_bank() {
        Some(_) => Some(decode_demand(definition, &draft.held_whole(definition)?)?),
        None => None,
    };
    Ok(PreparedExecutionAssessment { demand: decode_demand(definition, draft)?, whole, forms })
}

/// The standard workload's checked memory charge of `definition` planned
/// as `draft`, at the model's context ([`StructuralCharge`] of the draft,
/// charged at its own bank). It outlines every graph a load prepares, so
/// this fails with `AssessmentError::Graph` exactly when the model's program
/// cannot be built on the planned backend.
pub fn execution_memory_charge(
    definition: &ModelDefinition,
    draft: &ExecutionPlanDraft,
) -> Result<AssessmentMemoryCharge, AssessmentError> {
    StructuralCharge::of(definition, draft)?.charge(definition, draft)
}

/// The terms of a draft's charge that its expert bank's slots and the
/// served context leave as they are: its graph resource bound, which
/// outlines every graph a load prepares (no layout built), and its process
/// bytes. A bank's slots size only its record tensors' committed records
/// (`ExpertBankPlan::records_at`) and the banked stacks' resident bytes; its
/// records' extents, binding, conveyor, scratch and host tier, which the
/// program plan, the graphs and the kernels read, are of its structure
/// alone (`BankStructure`). The served context sizes only the state, which
/// [`Self::charge`] plans at the definition's context; the graphs outlined
/// at the largest served context bound every smaller one's. So the charge
/// of any draft that differs from this one's only in its bank's slots, at
/// any served context up to this one's, is this structure with that draft's
/// weights, state and header bounds ([`Self::charge`]).
#[derive(Debug)]
pub struct StructuralCharge {
    graph: AssessmentGraphResourceBounds,
    host_process: HostProcessBytes,
    device_process: DeviceProcessBytes,
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
    /// every other family of `plain` is outlined here, those not again.
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

    /// The structure of `draft`'s charge, a draft that differs from this
    /// structure's only in its pass (`ModelLoadPlan::pass`): its target
    /// family certified at that pass beside everything else as it is
    /// (`AssessmentGraphResourceBounds::at_pass`), which no pass moves.
    pub fn at_pass(&self, definition: &ModelDefinition, draft: &ExecutionPlanDraft) -> Result<Self, AssessmentError> {
        let policy = draft.policy();
        let graph = self
            .graph
            .at_pass(
                definition,
                draft.load(),
                &state_plan(definition, draft)?,
                policy.method(),
                policy.codec(),
                policy.limits(),
                draft.device().backend(),
            )
            .map_err(AssessmentError::Graph)?;
        Ok(Self { graph, host_process: self.host_process, device_process: self.device_process })
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
        let (graph, more) = graph(&state_plan(definition, draft)?).map_err(AssessmentError::Graph)?;
        let host_process =
            HostProcessBytes::derive(definition, draft.load(), draft.programs(), policy.limits(), backend)
                .map_err(AssessmentError::Memory)?;
        let device_process = draft
            .load()
            .device_allocations()
            .and_then(|allocations| DeviceProcessBytes::derive(backend, host_process.kernels, allocations))
            .map_err(AssessmentError::Memory)?;
        Ok((Self { graph, host_process, device_process }, more))
    }

    /// The charge of `draft`, a draft of this structure, at `definition`'s
    /// context: its weights' terms, its state planned there and its header
    /// bounds beside the structure's.
    pub fn charge(
        &self,
        definition: &ModelDefinition,
        draft: &ExecutionPlanDraft,
    ) -> Result<AssessmentMemoryCharge, AssessmentError> {
        let policy = draft.policy();
        let state = state_plan(definition, draft)?;
        let state_bytes = || -> Result<_, String> {
            Ok((state.capacity().workload_state_bytes, state.startup_state_bytes()?))
        };
        let (workload_state_bytes, startup_state_bytes) =
            state_bytes().map_err(AssessmentError::Memory)?;
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
                    workload_state_bytes,
                    startup_state_bytes,
                    self.graph.target.group_bytes.most(),
                )
            })
            .map_err(AssessmentError::Memory)
    }
}

/// The state plan of `definition` planned as `draft`, at its context.
fn state_plan(definition: &ModelDefinition, draft: &ExecutionPlanDraft) -> Result<crate::StateResourcePlan, AssessmentError> {
    let policy = draft.policy();
    ResourcePlanner::state_plan(
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
    .map_err(AssessmentError::Plan)
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

/// Whether the standard workload of one prepared model fits stable
/// capacity at `definition`'s context (a served context up to the one it
/// was prepared at), its experts held as a load holds them: whole where
/// the room holds them so (a bank of every expert, else their own
/// allocations), else streamed through the most bank slots the room holds,
/// each at its floor of places ([`crate::FloorForms`]); where none fits,
/// the shortfall of the least of them. Arithmetic over the prepared forms'
/// structures ([`StructuralCharge::charge`]), each plan confirmed as a
/// load's is (`StableRoom::fit`), its recorded graphs priced at `pricing`,
/// the machine's graph cost as its records give it now.
pub fn assess_execution_fit(
    definition: &ModelDefinition,
    prepared: &PreparedExecutionAssessment,
    pricing: crate::platform::GraphPricing,
    topology: &DeviceTopology,
    status: &Status,
    reserves: &MemoryReserves,
) -> Result<ExecutionAssessment, AssessmentError> {
    let forms = &prepared.forms;
    let device = planned_executor(prepared.draft(), topology)?;
    let room = crate::planning::StableRoom { topology, executor: device, status, reserves };
    let fit_context_tokens = u32::try_from(crate::planning::fit_depth(definition))
        .map_err(|_| AssessmentError::Memory("fit depth exceeds u32".into()))?;
    // The charge of `draft` of `form`'s structure and its fit.
    let charged = |form: &crate::HeldForm, draft: ExecutionPlanDraft| -> Result<_, AssessmentError> {
        let draft = draft.with_graph_pricing(pricing);
        let charge = form.structure.charge(definition, &draft)?;
        let fit = room.fit(&charge)?;
        Ok((draft, charge, fit))
    };
    let fits = |draft: &ExecutionPlanDraft, charge: &AssessmentMemoryCharge, fit: crate::AssessmentFit| -> Result<_, AssessmentError> {
        Ok(ExecutionAssessment::Fits(ExecutionFit {
            fit_context_tokens,
            breakdown: charge.breakdown(&fit.domains).map_err(AssessmentError::Memory)?,
            placement: ModelPlacement::of(draft.load(), &fit.domains, topology).map_err(AssessmentError::Memory)?,
            domains: fit.domains,
            bank: draft.load().expert_bank().cloned(),
        }))
    };
    let holds = |fit: &crate::AssessmentFit| matches!(fit.verdict, AssessmentFitVerdict::Fits);
    // Whole: a bank of every expert, else the experts' own allocations.
    if let Some(bank) = forms.bank().filter(|bank| bank.points.last() == Some(&bank.experts)) {
        let (draft, charge, fit) = charged(&bank.form, bank.at_slots(bank.experts)?)?;
        if holds(&fit) {
            return fits(&draft, &charge, fit);
        }
    }
    let resident = match forms.resident() {
        Some(resident) => {
            let (draft, charge, fit) = charged(resident, resident.draft.clone())?;
            if holds(&fit) {
                return fits(&draft, &charge, fit);
            }
            Some(fit)
        }
        None => None,
    };
    // Streamed: the most slots that fit, a bank's fit falling as its slots
    // do, beside the host tier host memory holds.
    let least = match forms.bank() {
        Some(bank) => {
            let at = |slots: u64| -> Result<_, AssessmentError> {
                let (_, charge, fit) = charged(&bank.form, bank.at_slots(slots)?)?;
                Ok((charge, fit))
            };
            let (floor_charge, floor_fit) = at(bank.points[0])?;
            if holds(&floor_fit) {
                let (mut fitting, mut short) = (0, bank.points.len());
                let mut held = (floor_charge, floor_fit);
                while short - fitting > 1 {
                    let middle = fitting + (short - fitting) / 2;
                    let (charge, fit) = at(bank.points[middle])?;
                    if holds(&fit) {
                        (fitting, held) = (middle, (charge, fit));
                    } else {
                        short = middle;
                    }
                }
                let slots = bank.points[fitting];
                let (charge, fit) = held;
                match bank.with_host_tier(slots, &charge)? {
                    Ok(draft) => return fits(&draft.with_graph_pricing(pricing), &charge, fit),
                    Err(reason) => {
                        return Err(AssessmentError::Plan(format!(
                            "the bank's floor fits, yet its experts cannot stream beside it: {reason}"
                        )));
                    }
                }
            }
            floor_fit
        }
        None => resident.expect("experts that cannot stream are held whole (`FloorForms::Resident`)"),
    };
    match least.verdict {
        AssessmentFitVerdict::DoesNotFit { limiting, deficit_bytes } => Ok(ExecutionAssessment::DoesNotFit(ExecutionShortfall {
            fit_context_tokens,
            domains: least.domains,
            limiting,
            deficit_bytes,
        })),
        AssessmentFitVerdict::Fits => unreachable!("a form that fits is held above"),
    }
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
    costs: &ExecutorCosts,
    request: &AssessmentRequest,
) -> Result<Vec<PerformanceEstimate>, AssessmentError> {
    estimate_decode(
        &prepared.demand,
        &decode_pricing(prepared.draft(), costs),
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
        /// The draft prepared as assessment prepares it: each way of holding
        /// its experts outlined at its floor.
        fn prepared(&self) -> PreparedExecutionAssessment {
            let forms = self
                .draft
                .floor_forms::<(), AssessmentError>(&self.definition, None, |draft| {
                    StructuralCharge::of(&self.definition, draft).map(|charge| (charge, ()))
                })
                .unwrap();
            prepare_execution_assessment(&self.definition, forms).unwrap()
        }

        /// The prepared draft's fit at `request`'s reserves.
        fn fit(&self, prepared: &PreparedExecutionAssessment, request: &AssessmentRequest) -> Result<ExecutionAssessment, AssessmentError> {
            assess_execution_fit(
                &self.definition,
                prepared,
                self.draft.device().graph_pricing(),
                &self.topology,
                &self.status,
                &request.reserves,
            )
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
        environment.fit(&environment.prepared(), request).unwrap()
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
        // A resident plan holds every weight in the device's memory: none in
        // a host tier, none read from disk.
        assert_eq!((placement.host_bytes, placement.disk_bytes), (0, 0));
        assert!(placement.device_bytes > 0);
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
        let prepared = environment.prepared();
        let costs = crate::assessment::fixtures::costs();
        let estimates = estimate_execution(&prepared, &costs, &request).unwrap();
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
        let fit_context_tokens = match environment.fit(&environment.prepared(), &request).unwrap() {
            ExecutionAssessment::Fits(fit) => fit.fit_context_tokens,
            ExecutionAssessment::DoesNotFit(shortfall) => shortfall.fit_context_tokens,
        };
        assert_eq!(fit_context_tokens, 150_000);
    }
}
