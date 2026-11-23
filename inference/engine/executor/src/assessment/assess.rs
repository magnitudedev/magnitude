//! One complete model assessment, from headers and the allocation-free
//! execution plan draft, as two questions with one owner each: whether the
//! standard workload fits ([`assess_execution_fit`]: the clean-load charge
//! against every domain the load touches, `DoesNotFit` naming the domain
//! with the largest deficit), and how fast a fitting model decodes
//! ([`estimate_execution`]: its per-call decode demand over the executor's
//! measured kernel costs).

use super::demand::{DecodeDemand, DraftDemand};
use super::estimate::{estimate_decode_speculative, performance_depths, PerformanceEstimate};
use magnitude_cost_model::{Census, DraftCensus, LaunchPrices, TargetCensus};
use seismic::measure::pricing::Pricing;
use super::executor_costs::ExecutorCosts;
use magnitude_cost_model::HeldSlots;
use super::AssessmentError;
use crate::memory::ClaimDemand;
use crate::platform::{DomainRole, ReleaseRate};
use crate::{
    AssessmentFitVerdict, AssessmentGraphResourceBounds, AssessmentHeaderBounds,
    AssessmentMemoryCharge, AttestedPrograms, DeviceProcessBytes, ExecutionPlanDraft,
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
    pub release: ReleaseRate,
}

/// Fit of the standard workload in one memory domain the load touches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DomainFit {
    pub role: DomainRole,
    pub domain: StoreId,
    /// What the executor's placements can hold in the store, within the
    /// store and the grants of the paths into it.
    pub capacity_bytes: u64,
    pub required_bytes: u64,
    /// `capacity − required`.
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
    /// when its remaining room beside any linked host expert tier cache takes
    /// them all (the device's memory, of unified memory), else on disk.
    pub fn of(load: &crate::ModelLoadPlan, domains: &[DomainFit], topology: &DeviceTopology) -> Result<Self, String> {
        let crate::WeightPlacement { held, host, streamed, tables } = load.weight_placement()?;
        let host_tier_cache = load
            .expert_bank()
            .map_or(Ok(0), |bank| bank.host_cache(bank.host_slots))?;
        Self::from_parts(held, host, streamed, tables, host_tier_cache, domains, topology)
    }

    pub(crate) fn of_bank(
        stored: u64,
        tables: u64,
        bank: Option<&crate::planning::ExpertBankPlan>,
        domains: &[DomainFit],
        topology: &DeviceTopology,
    ) -> Result<Self, String> {
        let host = bank.map_or(Ok(0), crate::planning::ExpertBankPlan::host_only_bytes)?;
        let streamed = bank.map_or(Ok(0), crate::planning::ExpertBankPlan::unheld_bytes)?;
        let host_tier_cache = bank.map_or(Ok(0), |bank| bank.host_cache(bank.host_slots))?;
        let held = [host, streamed, tables]
            .into_iter()
            .try_fold(stored, u64::checked_sub)
            .ok_or("a plan's tiers hold more than its weights")?;
        Self::from_parts(held, host, streamed, tables, host_tier_cache, domains, topology)
    }

    fn from_parts(
        held: u64,
        host: u64,
        streamed: u64,
        tables: u64,
        host_tier_cache: u64,
        domains: &[DomainFit],
        topology: &DeviceTopology,
    ) -> Result<Self, String> {
        let host_ram = domains.iter().find(|domain| {
            topology.store(domain.domain).is_some_and(|store| matches!(store.kind(), seismic::StoreKind::HostRam))
        });
        // `DomainFit::remaining_bytes` on HostRam only deducts the host tier's
        // required floor, not its elastic cache (`ExpertBankPlan::host_cache`).
        // Subtract `host_tier_cache` before deciding whether lookup tables also
        // fit in the OS file cache so host RAM is never double-spent between
        // the host expert tier and lookup tables (F-051).
        let cached = host_ram.is_none_or(|domain| {
            u64::try_from(domain.remaining_bytes)
                .is_ok_and(|room| room.saturating_sub(host_tier_cache) >= tables)
        });
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
    /// The streamed expert bank if this fit streams its routed experts,
    /// without cloning the bank.
    pub fn streamed_bank(&self) -> Option<&crate::planning::ExpertBankPlan> {
        self.bank.as_ref().filter(|bank| !bank.whole())
    }

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
                crate::planning::HostTier::Linked if bank.host_slots > 0 => {
                    let cache_bytes = bank
                        .host_cache(bank.host_slots)
                        .expect("a planned host tier's cache was computed when the plan sized it");
                    ColdSource::HostTier { slots: bank.host_slots, cache_bytes }
                }
                crate::planning::HostTier::Linked => ColdSource::File(FileOnly::NoHostRoom),
                crate::planning::HostTier::Unified(_) => ColdSource::File(FileOnly::Unified),
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

/// Precomputed decode demands for [`PreparedExecutionAssessment::from_demands`].
pub struct PreparedDemands {
    pub demand: DecodeDemand,
    pub whole: Option<DecodeDemand>,
    pub draft: Option<DraftDemand>,
}

/// Basis-independent decode demand and the structure of the standard
/// workload's memory charge in each way of holding the experts
/// ([`crate::FloorForms`]), at the model's largest served context: every
/// fit is arithmetic over them ([`assess_execution_fit`]).
pub struct PreparedExecutionAssessment {
    /// The definition the forms plan, at the largest served context.
    definition: ModelDefinition,
    demand: DecodeDemand,
    /// Of a plan that banks its experts, its demand holding them whole: the
    /// resident form its decode runs while its bank holds every expert. None
    /// of a plan that holds them whole, whose own demand is that form.
    whole: Option<DecodeDemand>,
    /// Of a drafted plan, its drafter's demand across certified proposal
    /// widths, retained from `SizingOutline` at zero extra outlining cost.
    draft: Option<DraftDemand>,
    forms: crate::FloorForms,
    target_census: std::sync::OnceLock<TargetCensus>,
    draft_census: std::sync::OnceLock<Option<DraftCensus>>,
}

impl PreparedExecutionAssessment {
    /// The definition the forms plan, at the largest served context.
    pub fn definition(&self) -> &ModelDefinition {
        &self.definition
    }

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

    /// Cached [`Self::censuses`], computed once per prepared execution.
    pub fn target_census(&self) -> &TargetCensus {
        self.target_census.get_or_init(|| match &self.whole {
            Some(whole) => {
                let (resident, streamed) = crate::assessment::pool::join(
                    || whole.census(),
                    || self.demand.census(),
                );
                TargetCensus::of(resident, Some(streamed))
            }
            None => TargetCensus::of(self.demand.census(), None),
        })
    }

    /// Cached drafter census, computed once per prepared execution (`None`
    /// when the plan does not draft).
    pub fn draft_census(&self) -> Option<&DraftCensus> {
        self.draft_census
            .get_or_init(|| self.draft.as_ref().map(DraftDemand::census))
            .as_ref()
    }

    /// Every census this preparation holds (target censuses plus any drafter
    /// width censuses), for [`ExecutorCosts::figures`].
    pub fn all_censuses(&self) -> Vec<&Census> {
        let mut censuses = self.target_census().censuses();
        if let Some(draft) = self.draft_census() {
            censuses.extend(draft.censuses());
        }
        censuses
    }

    /// The plan's censuses as they pair ([`TargetCensus::of`]): of a plan
    /// banking its experts, its census holding them whole beside its own;
    /// of one holding them whole, its own.
    pub fn censuses(&self) -> TargetCensus {
        self.target_census().clone()
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
        primary.structure.charge_with_pricing(definition, &primary.draft, pricing)
    }
    /// Compute the plain, whole-bank, and drafter decode demands from
    /// `definition` and `primary` without consuming `FloorForms`, so callers
    /// can overlap this with independent work such as `target_alone`.
    pub fn compute_decode_demands<T>(
        definition: &ModelDefinition,
        primary: &crate::HeldForm<T>,
    ) -> Result<PreparedDemands, AssessmentError> {
        let draft_plan = &primary.draft;
        let codec = draft_plan.policy().codec();
        let method = draft_plan.policy().method();
        let load = draft_plan.load();
        let graph = primary.structure.graph();
        let draft = DraftDemand::from_census(definition, method, graph)?;
        if load.expert_bank().is_some() {
            let (whole, demand) = crate::assessment::pool::join(
                || DecodeDemand::from_census(definition, load, codec, graph, true),
                || DecodeDemand::from_census(definition, load, codec, graph, false),
            );
            let whole = whole?;
            let demand = demand?;
            Ok(PreparedDemands { demand, whole: Some(whole), draft })
        } else {
            let demand = DecodeDemand::from_census(definition, load, codec, graph, false)?;
            Ok(PreparedDemands { demand, whole: None, draft })
        }
    }

    /// Assemble a [`PreparedExecutionAssessment`] from precomputed decode
    /// demands ([`Self::compute_decode_demands`]).
    pub fn from_demands(
        definition: ModelDefinition,
        mut forms: crate::FloorForms,
        demands: PreparedDemands,
    ) -> Box<Self> {
        forms.release_target_cache();
        Box::new(Self {
            definition,
            demand: demands.demand,
            whole: demands.whole,
            draft: demands.draft,
            forms,
            target_census: std::sync::OnceLock::new(),
            draft_census: std::sync::OnceLock::new(),
        })
    }
}

/// Perform model-specific arithmetic while the fixed generic basis is
/// measured: the plain decode step's demand ([`decode_demand`]) of the
/// forms' primary plan, and of a plan that banks its experts its demand
/// holding them whole, beside the forms' charge structures. Boxed: a
/// preparation holds a whole definition and its demands, and is moved by
/// value through deep frames, whose stacks hold every by-value copy
/// (`seismic_threads::STACK_BYTES`).
pub fn prepare_execution_assessment(
    definition: ModelDefinition,
    forms: crate::FloorForms,
) -> Result<Box<PreparedExecutionAssessment>, AssessmentError> {
    let demands = PreparedExecutionAssessment::compute_decode_demands(&definition, forms.primary())?;
    Ok(PreparedExecutionAssessment::from_demands(definition, forms, demands))
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
    /// Context limit at which `prepared_state_bytes` was planned.
    prepared_context: u64,
    /// `(workload_state_bytes, startup_state_bytes)` at `prepared_context`.
    prepared_state_bytes: (u64, u64),
    /// Precomputed context-invariant state layout for sizing state at any
    /// served tier without cloning `ModelDefinition` or re-deriving
    /// `ModelStateLayout`.
    state_sizer: std::sync::Arc<StateSizer>,
    /// Host-resident gathered tables (`load.host_table_bytes()`).
    host_table_bytes: u64,
    /// Upload windows invariant with bank slots and context.
    component_upload_bytes: u64,
    target_upload_peak_bytes: u64,
    /// Bank `(slots, host_slots)` for which `prepared_weight_demand`,
    /// `prepared_qualification_peak_bytes`, and `prepared_deferred_bank_bytes`
    /// were derived.
    prepared_bank_slots: Option<(u64, u64)>,
    prepared_weight_demand: ClaimDemand,
    prepared_qualification_peak_bytes: u64,
    prepared_deferred_bank_bytes: u64,
    /// Precomputed O(layers) slot-scaling model for banked plans so `assess_execution_fit`
    /// can evaluate any bank slot count without cloning `ExecutionPlanDraft` / `ModelLoadPlan`
    /// or re-scanning all weights.
    bank_slot_model: Option<BankSlotModel>,
    /// Precomputed `(stored_weight_bytes, unheld_table_bytes)` so `assess_execution_fit`
    /// can derive `ModelPlacement` at any bank `(slots, host_slots)` without cloning
    /// `ExecutionPlanDraft` or re-scanning all weights.
    base_weight_placement: Option<(u64, u64)>,
    /// Shared target-family layout cache (`Mutex<FamilyLayouts<SizingOutline>>`)
    /// threaded across `bank`, `resident`, `WindowedBankForm`, and `at_pass`
    /// so invariant entry, per-layer, and block-shape layouts are certified once.
    target_cache: Option<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
}

#[derive(Clone, Debug)]
struct StateSizer {
    target_history: Vec<magnitude_state::HistoryDomainLayout>,
    target_recurrent: Vec<magnitude_state::ComponentSpec>,
    head_history: Vec<magnitude_state::HistoryDomainLayout>,
    method: crate::PlannedMethod,
    limits: crate::ResourceLimits,
    capacity_bytes: ResourceCapacity,
    seeded_banks: Option<(u64, u32)>,
}

impl StateSizer {
    fn from_state(state: crate::StateResourcePlan, method: crate::PlannedMethod) -> Self {
        let (
            target_history,
            target_recurrent,
            head_history,
            limits,
            capacity_bytes,
            seeded_banks,
        ) = state.into_layout_parts();
        Self {
            target_history,
            target_recurrent,
            head_history,
            method,
            limits,
            capacity_bytes,
            seeded_banks,
        }
    }

    fn state_bytes_at_context(&self, context_limit: u64) -> Result<(u64, u64), AssessmentError> {
        ResourcePlanner::state_bytes_from_layout(
            &self.target_history,
            &self.target_recurrent,
            &self.head_history,
            context_limit,
            self.method,
            self.limits,
            self.capacity_bytes,
            self.seeded_banks,
        )
        .map_err(AssessmentError::Plan)
    }
}

/// Precomputed O(routed_layers) evaluator for banked `ModelLoadPlan` charges
/// at `(slots, places, paths)`: precomputes all slot/path/places-invariant
/// weight demands and qualification peak scopes once at preparation so fit
/// search evaluates any `(slots, places, paths)` in pure integer arithmetic
/// without cloning `ModelLoadPlan` or scanning all model weights.
#[derive(Clone, Debug)]
struct BankSlotModel {
    bank_template: crate::planning::ExpertBankPlan,
    /// Total `weight_demands` minus `bank.storage_demand()` (all non-banked
    /// weights plus invariant banked-weight/table metadata).
    base_weight_demand: ClaimDemand,
    /// `bank.storage_demand()` minus its `places`, `paths`, and `pooled`
    /// terms so `storage_demand` at any `(slots, places, paths)` is O(1).
    bank_storage_base: ClaimDemand,
    qual_peak: crate::native::QualificationPeakModel,
}

impl BankSlotModel {
    fn of(load: &crate::ModelLoadPlan) -> Result<Self, String> {
        let bank = load
            .expert_bank()
            .ok_or_else(|| "a bank slot model requires an expert bank".to_owned())?;
        let initial_total = ClaimDemand::checked_sum(load.weight_demands()?)
            .ok_or_else(|| "weight demand overflows".to_owned())?;
        let initial_storage = bank.storage_demand()?;
        let base_weight_demand = initial_total
            .checked_minus(initial_storage)
            .ok_or_else(|| "base weight demand underflows".to_owned())?;

        let initial_places_demand = bank.places_storage_demand(bank.places)?;
        let initial_paths_demand = bank.paths_layout_demand(bank.paths)?;
        let initial_pooled_demand = bank.pooled_demand_at(bank.slots, bank.paths)?;
        let bank_storage_base = initial_storage
            .checked_minus(initial_places_demand)
            .and_then(|d| d.checked_minus(initial_paths_demand))
            .and_then(|d| d.checked_minus(initial_pooled_demand))
            .ok_or_else(|| "bank storage base underflows".to_owned())?;

        let qual_peak = AttestedPrograms::qualification_peak_model(load)?;
        let mut bank_template = bank.clone();
        bank_template.ceiling = bank_template.widest().get();
        Ok(Self {
            bank_template,
            base_weight_demand,
            bank_storage_base,
            qual_peak,
        })
    }

    fn sum_banked_records(bank: &crate::planning::ExpertBankPlan) -> Result<(u64, Vec<u64>), String> {
        let mut total = 0u64;
        let mut per_layer = Vec::with_capacity(bank.layers().len());
        for index in 0..bank.layers().len() {
            let banked = bank.banked_bytes(index)?;
            let mut layer_sum = 0u64;
            for (_, &bytes) in banked.iter() {
                layer_sum = layer_sum
                    .checked_add(bytes)
                    .ok_or("banked record bytes overflow")?;
            }
            total = total
                .checked_add(layer_sum)
                .ok_or("banked record bytes overflow")?;
            per_layer.push(layer_sum);
        }
        Ok((total, per_layer))
    }

    fn bank_at(
        &self,
        slots: u64,
        places: u64,
        paths: crate::planning::BankPaths,
    ) -> Result<crate::planning::ExpertBankPlan, String> {
        let mut bank = self.bank_template.clone();
        let widest = bank.widest().get();
        bank.paths = paths;
        bank.places = places;
        if slots < bank.floor() || slots > widest {
            return Err(
                "an expert bank holds from its floor to its ceiling, at most every expert of each layer"
                    .to_owned(),
            );
        }
        bank.slots = slots;
        bank.ceiling = widest;
        bank.resident = slots == widest;
        Ok(bank)
    }

    fn weight_demand_for_bank(
        &self,
        bank: &crate::planning::ExpertBankPlan,
    ) -> Result<ClaimDemand, String> {
        let places_demand = bank.places_storage_demand(bank.places)?;
        let paths_demand = bank.paths_layout_demand(bank.paths)?;
        let pooled_demand = bank.pooled_demand_at(bank.slots, bank.paths)?;
        self.base_weight_demand
            .checked_plus(self.bank_storage_base)
            .and_then(|d| d.checked_plus(places_demand))
            .and_then(|d| d.checked_plus(paths_demand))
            .and_then(|d| d.checked_plus(pooled_demand))
            .ok_or_else(|| "weight demand overflows".to_owned())
    }

    #[cfg(test)]
    fn weight_demand_at(
        &self,
        _bank: &crate::planning::ExpertBankPlan,
        slots: u64,
        places: u64,
        paths: crate::planning::BankPaths,
    ) -> Result<ClaimDemand, String> {
        let bank = self.bank_at(slots, places, paths)?;
        self.weight_demand_for_bank(&bank)
    }

    fn charge_terms(
        &self,
        _bank: &crate::planning::ExpertBankPlan,
        slots: u64,
        places: u64,
        paths: crate::planning::BankPaths,
    ) -> Result<(ClaimDemand, u64, u64), String> {
        let bank = self.bank_at(slots, places, paths)?;
        let (_, per_layer) = Self::sum_banked_records(&bank)?;
        let weight_demand = self.weight_demand_for_bank(&bank)?;
        let qualification_peak = self.qual_peak.peak_bytes_from_layers(&per_layer)?;
        let deferred_bank_bytes = crate::planning::deferred_bank_bytes_of_bank(&bank)?;
        Ok((weight_demand, qualification_peak, deferred_bank_bytes))
    }
}

fn bank_slots_key(load: &crate::ModelLoadPlan) -> Option<(u64, u64)> {
    load.expert_bank().map(|bank| (bank.slots, bank.host_slots))
}

/// Opaque handle to a certified target-family layout cache, reusable across
/// plain and drafted plans of the same target.
#[derive(Clone)]
pub struct TargetLayoutCache(std::sync::Arc<dyn std::any::Any + Send + Sync>);

impl StructuralCharge {
    pub(crate) fn graph(&self) -> &AssessmentGraphResourceBounds {
        &self.graph
    }

    /// Shared target-family layout cache populated during certification, if any.
    pub fn target_cache(&self) -> Option<TargetLayoutCache> {
        self.target_cache
            .as_ref()
            .map(|cache| TargetLayoutCache(std::sync::Arc::clone(cache)))
    }

    /// Drop the target-family layout cache once certification has finished so
    /// retained preparations do not hold certified graph layouts in memory.
    pub fn release_target_cache(&mut self) {
        self.target_cache = None;
    }

    /// The structure of `draft`'s charge.
    pub fn of(definition: &ModelDefinition, draft: &ExecutionPlanDraft) -> Result<Self, AssessmentError> {
        Self::of_with_cache(definition, draft, None)
    }

    /// [`Self::of`], optionally reusing a target-family layout cache from
    /// another plan of the same target (e.g., a pre-planned plain target).
    pub fn of_with_cache(
        definition: &ModelDefinition,
        draft: &ExecutionPlanDraft,
        cache: Option<&TargetLayoutCache>,
    ) -> Result<Self, AssessmentError> {
        Self::with_graph(definition, draft, true, |state| {
            let policy = draft.policy();
            AssessmentGraphResourceBounds::derive_with_cache(
                definition,
                draft.load(),
                draft.programs(),
                state,
                policy.method(),
                policy.limits(),
                draft.device().backend(),
                cache.map(|c| &c.0),
            )
            .map(|(graph, cache)| (graph, Some(cache), ()))
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
        Self::with_undrafted_with_cache(definition, draft, None)
    }

    /// [`Self::with_undrafted`], optionally reusing a target-family layout cache
    /// from another plan of the same target.
    pub fn with_undrafted_with_cache(
        definition: &ModelDefinition,
        draft: &ExecutionPlanDraft,
        cache: Option<&TargetLayoutCache>,
    ) -> Result<(Self, UndraftedBounds), AssessmentError> {
        Self::with_graph(definition, draft, true, |state| {
            let policy = draft.policy();
            AssessmentGraphResourceBounds::derive_with_undrafted_and_cache(
                definition,
                draft.load(),
                draft.programs(),
                state,
                policy.method(),
                policy.limits(),
                draft.device().backend(),
                cache.map(|c| &c.0),
            )
            .map(|(graph, undrafted, cache)| (graph, Some(cache), undrafted))
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
        Self::with_graph(definition, plain, false, |state| {
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
                undrafted.readout,
            )
            .map(|graph| (graph, None, ()))
        })
        .map(|(structure, ())| structure)
    }

    /// The structure of `draft`'s charge, a draft that differs from this
    /// structure's only in how its target routed experts are held (banked vs
    /// resident, scratch window size, or pass): certifies only `draft`'s
    /// target family (reusing this structure's target-family layout cache,
    /// and its state copy family only if its recurrent bank capacity
    /// changed), reusing `readout`, `head`/`drafter`, and `vision` from
    /// `self`.
    pub fn reusing_non_target(
        &self,
        definition: &ModelDefinition,
        draft: &ExecutionPlanDraft,
    ) -> Result<Self, AssessmentError> {
        Self::with_graph(definition, draft, true, |state| {
            let policy = draft.policy();
            self.graph
                .derive_reusing_non_target_with_cache(
                    definition,
                    draft.load(),
                    draft.programs(),
                    state,
                    policy.method(),
                    policy.limits(),
                    draft.device().backend(),
                    self.target_cache.as_ref(),
                )
                .map(|(graph, cache)| (graph, Some(cache), ()))
        })
        .map(|(structure, ())| structure)
    }

    /// The structure of `draft`'s charge, a draft that differs from this
    /// structure's only in its pass (`ModelLoadPlan::pass`): its target
    /// family certified at that pass beside everything else as it is
    /// (`AssessmentGraphResourceBounds::at_pass_with_cache`), which no pass moves.
    pub fn idle_class_bytes(
        &self,
        places: usize,
        limits: crate::ResourceLimits,
        fronts: bool,
    ) -> Result<u64, AssessmentError> {
        self.graph
            .idle_class_bytes(places, limits, fronts)
            .map_err(AssessmentError::Plan)
    }

    pub fn at_pass(&self, definition: &ModelDefinition, draft: &ExecutionPlanDraft) -> Result<Self, AssessmentError> {
        let _scope =
            crate::programs::graph::sizing::SizingScope::enter_with_device(draft.device().properties());
        let policy = draft.policy();
        let (graph, cache) = self
            .graph
            .at_pass_with_cache(
                definition,
                draft.load(),
                draft.programs(),
                &state_plan(definition, draft)?,
                policy.method(),
                policy.limits(),
                draft.device().backend(),
                self.target_cache.as_ref(),
            )
            .map_err(AssessmentError::Graph)?;
        Ok(Self {
            graph,
            host_process: self.host_process,
            device_process: self.device_process,
            prepared_context: self.prepared_context,
            prepared_state_bytes: self.prepared_state_bytes,
            state_sizer: std::sync::Arc::clone(&self.state_sizer),
            host_table_bytes: self.host_table_bytes,
            component_upload_bytes: self.component_upload_bytes,
            target_upload_peak_bytes: self.target_upload_peak_bytes,
            prepared_bank_slots: self.prepared_bank_slots,
            prepared_weight_demand: self.prepared_weight_demand,
            prepared_qualification_peak_bytes: self.prepared_qualification_peak_bytes,
            prepared_deferred_bank_bytes: self.prepared_deferred_bank_bytes,
            bank_slot_model: self.bank_slot_model.clone(),
            base_weight_placement: self.base_weight_placement,
            target_cache: Some(cache),
        })
    }

    /// The structure of `draft`'s charge with the graph bounds `graph`
    /// derives from its state plan, and what else `graph` gives.
    fn with_graph<T>(
        definition: &ModelDefinition,
        draft: &ExecutionPlanDraft,
        build_fit_models: bool,
        graph: impl FnOnce(
            &crate::StateResourcePlan,
        ) -> Result<
            (
                AssessmentGraphResourceBounds,
                Option<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
                T,
            ),
            crate::GraphError,
        >,
    ) -> Result<(Self, T), AssessmentError> {
        let _scope =
            crate::programs::graph::sizing::SizingScope::enter_with_device(draft.device().properties());
        let policy = draft.policy();
        let backend = draft.device().backend();
        let load = draft.load();
        if policy.selection().head != policy.method().drafts() {
            return Err(AssessmentError::Memory(
                "assessment method and head selection disagree".into(),
            ));
        }
        if policy.selection().head != load.head().is_some()
            || policy.selection().vision != load.vision().is_some()
        {
            return Err(AssessmentError::Memory(
                "assessment selection disagrees with the load plan".into(),
            ));
        }
        let state = state_plan(definition, draft)?;
        let prepared_state_bytes = (
            state.capacity().workload_state_bytes,
            state.startup_state_bytes().map_err(AssessmentError::Memory)?,
        );
        let (graph, target_cache, more) = graph(&state).map_err(AssessmentError::Graph)?;
        let state_sizer = std::sync::Arc::new(StateSizer::from_state(state, draft.policy().method()));
        let host_process =
            HostProcessBytes::from_graph_bounds(definition, draft.programs(), &graph, backend)
                .map_err(AssessmentError::Memory)?;
        let device_process = load
            .device_allocations()
            .and_then(|allocations| DeviceProcessBytes::derive(backend, host_process.kernels, allocations))
            .map_err(AssessmentError::Memory)?;
        let host_table_bytes = load.host_table_bytes().map_err(AssessmentError::Memory)?;
        let header = AssessmentHeaderBounds::derive(load).map_err(AssessmentError::Memory)?;
        let prepared_weight_demand = load
            .weight_demands()
            .and_then(|demands| {
                ClaimDemand::checked_sum(demands).ok_or_else(|| "weight demand overflows".to_owned())
            })
            .map_err(AssessmentError::Memory)?;
        let bank_slot_model = if build_fit_models && load.expert_bank().is_some() {
            Some(BankSlotModel::of(load).map_err(AssessmentError::Memory)?)
        } else {
            None
        };
        let base_weight_placement = if build_fit_models {
            let crate::WeightPlacement { held, host, streamed, tables } =
                load.weight_placement().map_err(AssessmentError::Memory)?;
            let stored = held
                .checked_add(host)
                .and_then(|v| v.checked_add(streamed))
                .and_then(|v| v.checked_add(tables))
                .ok_or_else(|| AssessmentError::Memory("weight placement bytes overflow".to_owned()))?;
            Some((stored, tables))
        } else {
            None
        };
        let prepared_deferred_bank_bytes =
            crate::planning::deferred_bank_bytes(load).map_err(AssessmentError::Memory)?;
        Ok((
            Self {
                graph,
                host_process,
                device_process,
                prepared_context: definition.decoder.context_limit,
                prepared_state_bytes,
                state_sizer,
                host_table_bytes,
                component_upload_bytes: header.component_upload_bytes,
                target_upload_peak_bytes: header.target_upload_peak_bytes,
                prepared_bank_slots: bank_slots_key(load),
                prepared_weight_demand,
                prepared_qualification_peak_bytes: header.qualification_peak_bytes,
                prepared_deferred_bank_bytes,
                bank_slot_model,
                base_weight_placement,
                target_cache,
            },
            more,
        ))
    }

    fn state_bytes_at_context(&self, context_limit: u64) -> Result<(u64, u64), AssessmentError> {
        if context_limit == self.prepared_context {
            Ok(self.prepared_state_bytes)
        } else {
            self.state_sizer.state_bytes_at_context(context_limit)
        }
    }

    fn compose_charge(
        &self,
        state_bytes: (u64, u64),
        weight_demand: ClaimDemand,
        qualification_peak_bytes: u64,
        deferred_bank_bytes: u64,
        graph_cost: seismic::GraphCost,
    ) -> Result<AssessmentMemoryCharge, AssessmentError> {
        let header = AssessmentHeaderBounds {
            qualification_peak_bytes,
            target_upload_peak_bytes: self.target_upload_peak_bytes,
            component_upload_bytes: self.component_upload_bytes,
        };
        let bounds = header
            .with_graph_resource_bound(
                &self.graph,
                graph_cost,
                self.host_process,
                self.device_process,
            )
            .map_err(AssessmentError::Memory)?;
        let weight_bytes = weight_demand
            .checked_bytes()
            .ok_or_else(|| AssessmentError::Memory("weight demand overflows".to_owned()))?;
        AssessmentMemoryCharge::compose(
            weight_bytes,
            weight_demand,
            bounds,
            state_bytes.0,
            state_bytes.1,
            self.host_table_bytes,
            deferred_bank_bytes,
            self.graph.target.group_bytes.most(),
        )
        .map_err(AssessmentError::Memory)
    }

    /// The charge of `draft`, a draft of this structure, at `definition`'s
    /// context: reuses all form-invariant terms hoisted in [`Self::with_graph`]
    /// and only recomputes what varies with context or bank slots.
    pub fn charge(
        &self,
        definition: &ModelDefinition,
        draft: &ExecutionPlanDraft,
    ) -> Result<AssessmentMemoryCharge, AssessmentError> {
        self.charge_with_pricing(definition, draft, draft.device().graph_pricing())
    }

    /// [`Self::charge`] priced at `pricing` without cloning `draft`.
    pub fn charge_with_pricing(
        &self,
        definition: &ModelDefinition,
        draft: &ExecutionPlanDraft,
        pricing: crate::platform::GraphPricing,
    ) -> Result<AssessmentMemoryCharge, AssessmentError> {
        self.charge_at_context_with_pricing(definition.decoder.context_limit, draft, pricing)
    }

    /// [`Self::charge_with_pricing`] evaluated at `context_limit` without
    /// mutating or cloning `ModelDefinition`.
    pub fn charge_at_context_with_pricing(
        &self,
        context_limit: u64,
        draft: &ExecutionPlanDraft,
        pricing: crate::platform::GraphPricing,
    ) -> Result<AssessmentMemoryCharge, AssessmentError> {
        let state_bytes = self.state_bytes_at_context(context_limit)?;
        self.charge_with_state_and_pricing(state_bytes, draft, pricing)
    }

    fn charge_with_state_and_pricing(
        &self,
        state_bytes: (u64, u64),
        draft: &ExecutionPlanDraft,
        pricing: crate::platform::GraphPricing,
    ) -> Result<AssessmentMemoryCharge, AssessmentError> {
        let load = draft.load();
        let (weight_demand, qualification_peak_bytes, deferred_bank_bytes) =
            if bank_slots_key(load) == self.prepared_bank_slots {
                (
                    self.prepared_weight_demand,
                    self.prepared_qualification_peak_bytes,
                    self.prepared_deferred_bank_bytes,
                )
            } else {
                let weight_demand = load
                    .weight_demands()
                    .and_then(|demands| {
                        ClaimDemand::checked_sum(demands)
                            .ok_or_else(|| "weight demand overflows".to_owned())
                    })
                    .map_err(AssessmentError::Memory)?;
                let qualification_peak_bytes = AttestedPrograms::qualification_peak_bytes(load)
                    .map_err(AssessmentError::Memory)?;
                let deferred_bank_bytes = crate::planning::deferred_bank_bytes(load)
                    .map_err(AssessmentError::Memory)?;
                (weight_demand, qualification_peak_bytes, deferred_bank_bytes)
            };
        self.compose_charge(
            state_bytes,
            weight_demand,
            qualification_peak_bytes,
            deferred_bank_bytes,
            pricing.cost(),
        )
    }

    /// The charge of this banked structure at `(slots, places, paths)` and
    /// `pricing`, using precomputed `state_bytes` and `BankSlotModel` without
    /// cloning `draft`.
    fn charge_at_slots_places_and_paths(
        &self,
        state_bytes: (u64, u64),
        bank: &crate::planning::ExpertBankPlan,
        slots: u64,
        places: u64,
        paths: crate::planning::BankPaths,
        pricing: crate::platform::GraphPricing,
    ) -> Result<AssessmentMemoryCharge, AssessmentError> {
        let model = self
            .bank_slot_model
            .as_ref()
            .expect("invariant violated: charge_at_slots_places_and_paths requires a banked StructuralCharge");
        let (weight_demand, qualification_peak, deferred_bank_bytes) = model
            .charge_terms(bank, slots, places, paths)
            .map_err(AssessmentError::Memory)?;
        self.compose_charge(
            state_bytes,
            weight_demand,
            qualification_peak,
            deferred_bank_bytes,
            pricing.cost(),
        )
    }
}

fn resource_capacity(draft: &ExecutionPlanDraft) -> Result<ResourceCapacity, AssessmentError> {
    Ok(ResourceCapacity {
        domain_bytes: crate::MemoryBudget::of_stable(draft.device())
            .map_err(|error| AssessmentError::Plan(error.to_string()))?
            .capacity(),
        tensor_operations: crate::TensorOperations::of(draft.device().tensor_operations()),
        allocations: draft.device().allocations(),
    })
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
        resource_capacity(draft)?,
    )
    .map_err(AssessmentError::Plan)
}

/// The plain decode step's demand of `definition` planned as `draft`,
/// under its planned history codec, derived directly from `SizingDraft`'s
/// decode-regime launch census.
pub fn decode_demand(definition: &ModelDefinition, draft: &ExecutionPlanDraft) -> Result<DecodeDemand, AssessmentError> {
    let _scope =
        crate::programs::graph::sizing::SizingScope::enter_with_device(draft.device().properties());
    let policy = draft.policy();
    let capacity = resource_capacity(draft)?;
    super::demand::decode_demand_direct(
        definition,
        draft.load(),
        policy.codec(),
        policy.method(),
        policy.limits(),
        capacity,
        draft.device().backend(),
        false,
    )
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

/// Precomputed streamed-layer and executor-cost figures shared across every
/// served tier's bank path pricing and streamed decode estimation in one
/// assessment finish.
pub struct PricedBankContext {
    pub layers: magnitude_cost_model::StreamedLayers,
    pub figures: seismic::measure::pricing::Figures,
    pub read: magnitude_cost_model::ReadLine,
    pub lookahead: bool,
    pub num_layers: u64,
    pub least_move_bytes: u64,
}

impl PricedBankContext {
    /// Precompute the streamed layers and target census cost figures for
    /// `prepared` if it has an expert bank form. Returns `Ok(None)` for
    /// non-banked models or when `streamed_layers` is not yet measured.
    pub fn new(
        prepared: &PreparedExecutionAssessment,
        costs: &ExecutorCosts,
        figures: &super::StreamedFigures,
    ) -> Result<Option<Self>, AssessmentError> {
        let Some(bank) = prepared.forms.bank() else {
            return Ok(None);
        };
        let bank_plan = bank
            .form
            .draft
            .load()
            .expert_bank()
            .expect("a bank form holds an expert bank");
        let (layers, priced) = super::pool::join(
            || super::streamed_layers(bank_plan, costs, figures),
            || costs.figures(&prepared.all_censuses()),
        );
        let super::Priced::Measured(layers) = layers else {
            return Ok(None);
        };
        let least_move_bytes = bank_plan
            .least_move_bytes()
            .map_err(AssessmentError::Memory)?;
        Ok(Some(Self {
            layers,
            figures: priced,
            read: figures.read,
            lookahead: lookahead(prepared.draft()),
            num_layers: bank_plan.layers().len() as u64,
            least_move_bytes,
        }))
    }
}

/// O(1)-per-slot path record pricer derived from a prepared assessment's
/// cached `TargetCensus` and measured executor/streamed figures, matching
/// `magnitude_cost_model::BankPrices::paths` bit-for-bit without re-certifying
/// graphs.
struct BankPathPricer<'a> {
    layers: &'a magnitude_cost_model::StreamedLayers,
    read: magnitude_cost_model::ReadLine,
    compute: magnitude_cost_model::FormSeconds,
    num_layers: u64,
    least_move_bytes: u64,
    decode_rows: u64,
}

impl<'a> BankPathPricer<'a> {
    fn from_context(
        context_limit: u64,
        prepared: &PreparedExecutionAssessment,
        context: &'a PricedBankContext,
    ) -> Result<Self, AssessmentError> {
        let limits = prepared.draft().policy().limits();
        let prior = crate::planning::prior(context_limit, limits)?;
        let target = prepared.target_census();
        let pricing = Pricing {
            figures: &context.figures,
            lookahead: context.lookahead,
        };
        let compute = target.each(|census| {
            magnitude_cost_model::step_seconds(census, &pricing, 1, prior.decode_depth)
        })?;
        Ok(Self {
            layers: &context.layers,
            read: context.read,
            compute,
            num_layers: context.num_layers,
            least_move_bytes: context.least_move_bytes,
            decode_rows: crate::planning::decode_rows(limits),
        })
    }

    fn paths_at(&self, slots: u64) -> Result<crate::planning::BankPaths, AssessmentError> {
        let slots = usize::try_from(slots)
            .map_err(|_| AssessmentError::Memory("a bank's slots exceed usize".into()))?;
        let held = HeldSlots::bank(slots);
        let saturation_bytes = seismic_io::disk::FULL_DEPTH * seismic_io::disk::FULL_READ;
        let records = self
            .layers
            .path_records(
                self.read,
                self.compute,
                self.least_move_bytes,
                held,
                crate::planning::HINT_LEADS,
                self.num_layers,
                self.decode_rows,
                saturation_bytes,
            )
            .unwrap_or(magnitude_cost_model::PathRecords::NONE);
        Ok(crate::planning::BankPaths::Priced {
            line: self.read,
            records,
        })
    }
}

/// Clone `draft`'s `ExpertBankPlan` at `(slots, places, paths)` and size its
/// linked host tier against `room` beside `charge`, without cloning
/// `ModelLoadPlan` or `ProgramPlan`.
fn size_bank_with_host_tier(
    draft: &ExecutionPlanDraft,
    slots: u64,
    places: u64,
    paths: crate::planning::BankPaths,
    charge: &AssessmentMemoryCharge,
    budget: &crate::planning::MemoryBudget,
) -> Result<Result<crate::planning::ExpertBankPlan, crate::UnstreamedExperts>, AssessmentError> {
    use crate::planning::{HostTier, PassLaunches};

    let mut bank = draft
        .load()
        .expert_bank()
        .expect("a bank form holds an expert bank")
        .clone();
    bank.places = places;
    bank.paths = paths;
    if slots < bank.floor() || slots > bank.widest().get() {
        return Err(AssessmentError::Plan(
            "an expert bank holds from its floor to its ceiling, at most every expert of each layer".into(),
        ));
    }
    bank.slots = slots;
    bank.ceiling = slots;
    bank.resident = slots == bank.widest().get();
    bank.pass = PassLaunches::FLOOR;
    if draft.device().host_tier() != Some(HostTier::Linked) {
        return Ok(Ok(bank));
    }
    let Some(ceiling) = bank.host_ceiling().filter(|&ceiling| ceiling > 0 && !bank.whole()) else {
        return Ok(Ok(bank));
    };
    let left = |host_slots: u64| -> Result<i64, AssessmentError> {
        let growth = bank.host_cache(host_slots).map_err(AssessmentError::Memory)?;
        Ok(budget.host_left(draft, charge, growth)?)
    };
    if left(0)? < 0 {
        let reserve = bank.landing_records();
        let reason = format!(
            "host memory admits not even the tier's landing reserve of {reserve} records a group"
        );
        return Ok(Err(crate::UnstreamedExperts::Untiered {
            backend: draft.device().backend(),
            reason,
        }));
    }
    let most = if left(ceiling)? >= 0 {
        ceiling
    } else {
        let (mut lo, mut hi) = (0u64, ceiling);
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            if left(mid)? >= 0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    };
    bank.host_slots = most;
    Ok(Ok(bank))
}

/// Settle a streamed bank rung through the same `BankPaths` fixed-point as
/// runtime `ExecutionPlanDraft::with_paths`, binary-searching `floor..=ceiling_slots`
/// at each fixed `paths`. Returns `Ok(Ok((bank, charge, fit)))` when the rung
/// fits, or `Ok(Err(shortfall_fit))` when its floor does not fit.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn settle_bank_rung(
    draft: &ExecutionPlanDraft,
    structure: &StructuralCharge,
    ceiling_slots: u64,
    state_bytes: (u64, u64),
    tier_places: u64,
    path_pricer: Option<&BankPathPricer<'_>>,
    pricing: crate::platform::GraphPricing,
    topology: &DeviceTopology,
    executor: &seismic::Executor,
    budget: &crate::planning::MemoryBudget,
) -> Result<
    Result<(crate::planning::ExpertBankPlan, AssessmentMemoryCharge, crate::AssessmentFit), crate::AssessmentFit>,
    AssessmentError,
> {
    let bank_plan = draft
        .load()
        .expert_bank()
        .expect("a bank form holds an expert bank");
    let holds = |fit: &crate::AssessmentFit| matches!(fit.verdict, AssessmentFitVerdict::Fits);
    let mut paths = match path_pricer {
        Some(pricer) => pricer.paths_at(ceiling_slots)?,
        None => bank_plan.paths,
    };
    let mut seen = Vec::new();
    loop {
        let floor = bank_plan.floor_with_paths(paths);
        if floor > ceiling_slots {
            return Err(AssessmentError::Plan(format!(
                "priced bank floor ({floor}) exceeds ceiling ({ceiling_slots})"
            )));
        }
        let floor_charge = structure.charge_at_slots_places_and_paths(
            state_bytes,
            bank_plan,
            floor,
            tier_places,
            paths,
            pricing,
        )?;
        let floor_fit = budget.fit(topology, executor, &floor_charge)?;
        if !holds(&floor_fit) {
            return Ok(Err(floor_fit));
        }
        let (mut lo, mut short) = (floor, ceiling_slots + 1);
        let mut best = (floor_charge, floor_fit);
        while short - lo > 1 {
            let mid = lo + (short - lo) / 2;
            let charge = structure.charge_at_slots_places_and_paths(
                state_bytes,
                bank_plan,
                mid,
                tier_places,
                paths,
                pricing,
            )?;
            let fit = budget.fit(topology, executor, &charge)?;
            if holds(&fit) {
                lo = mid;
                best = (charge, fit);
            } else {
                short = mid;
            }
        }
        let (slots, (charge, fit)) = (lo, best);
        let next = match path_pricer {
            Some(pricer) => pricer.paths_at(slots)?,
            None => paths,
        };
        if next == paths {
            match size_bank_with_host_tier(draft, slots, tier_places, paths, &charge, budget)? {
                Ok(sized) => return Ok(Ok((sized, charge, fit))),
                Err(reason) => {
                    return Err(AssessmentError::Plan(format!(
                        "the bank's floor fits, yet its experts cannot stream beside it: {reason}"
                    )));
                }
            }
        }
        assert!(
            !seen.contains(&next),
            "invariant violated: a bank's path records and slots cycle through {seen:?} without a fixed point"
        );
        seen.push(paths);
        paths = next;
    }
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
) -> Result<ExecutionAssessment, AssessmentError> {
    assess_execution_fit_at_context(
        crate::planning::fit_depth(definition),
        prepared,
        pricing,
        topology,
        status,
        None,
    )
}

/// [`assess_execution_fit`] evaluated at `context_limit` without mutating or
/// cloning `ModelDefinition`.
///
/// Adjusts `places`-dependent bank parts (`FileTable`, `ScratchTable`,
/// `Routing`, `Keys`) to `prompt_places_at(limits, context_limit)`, and when
/// `bank_prices` is `Some((costs, figures))` prices `BankPaths::Priced`
/// staging/landing/reserve records via the same fixed-point as runtime
/// `ExecutionPlanDraft::with_paths`.
pub fn assess_execution_fit_at_context(
    context_limit: u64,
    prepared: &PreparedExecutionAssessment,
    pricing: crate::platform::GraphPricing,
    topology: &DeviceTopology,
    status: &Status,
    bank_prices: Option<(&ExecutorCosts, &super::StreamedFigures)>,
) -> Result<ExecutionAssessment, AssessmentError> {
    let owned_context = match bank_prices {
        Some((costs, figures)) => PricedBankContext::new(prepared, costs, figures)?,
        None => None,
    };
    assess_execution_fit_with_priced_bank(
        context_limit,
        prepared,
        pricing,
        topology,
        status,
        owned_context.as_ref(),
    )
}

/// [`assess_execution_fit_at_context`] reusing a precomputed
/// [`PricedBankContext`] across multiple served tiers so `streamed_layers` and
/// `costs.figures` are computed once per assessment finish.
pub fn assess_execution_fit_with_priced_bank(
    context_limit: u64,
    prepared: &PreparedExecutionAssessment,
    pricing: crate::platform::GraphPricing,
    topology: &DeviceTopology,
    status: &Status,
    priced_bank: Option<&PricedBankContext>,
) -> Result<ExecutionAssessment, AssessmentError> {
    let forms = &prepared.forms;
    let device = planned_executor(prepared.draft(), topology)?;
    let budget = crate::planning::MemoryBudget::stable(topology, device, status)?;
    let fit_context_tokens = u32::try_from(context_limit)
        .map_err(|_| AssessmentError::Memory("fit depth exceeds u32".into()))?;
    let fits_with_bank = |structure: &StructuralCharge,
                          load: &crate::ModelLoadPlan,
                          bank: Option<crate::planning::ExpertBankPlan>,
                          charge: &AssessmentMemoryCharge,
                          fit: crate::AssessmentFit|
     -> Result<_, AssessmentError> {
        let placement = match structure.base_weight_placement {
            Some((stored, tables)) => {
                ModelPlacement::of_bank(stored, tables, bank.as_ref(), &fit.domains, topology)
                    .map_err(AssessmentError::Memory)?
            }
            None => ModelPlacement::of(load, &fit.domains, topology)
                .map_err(AssessmentError::Memory)?,
        };
        Ok(ExecutionAssessment::Fits(ExecutionFit {
            fit_context_tokens,
            breakdown: charge.breakdown(&fit.domains).map_err(AssessmentError::Memory)?,
            placement,
            domains: fit.domains,
            bank,
        }))
    };
    let holds = |fit: &crate::AssessmentFit| matches!(fit.verdict, AssessmentFitVerdict::Fits);
    // `state_plan` depends only on `definition.decoder`, `load.head.is_some()`,
    // policy, and device — identical across bank, resident, and windowed forms.
    let primary_form = forms.primary();
    let state_bytes = primary_form
        .structure
        .state_bytes_at_context(context_limit)?;
    let tier_places = crate::planning::prompt_places_at(
        prepared.draft().policy().limits(),
        context_limit,
    )
    .max(1);
    let path_pricer = match (forms.bank(), priced_bank) {
        (Some(_), Some(context)) => Some(BankPathPricer::from_context(
            context_limit,
            prepared,
            context,
        )?),
        _ => None,
    };
    // Whole: a bank of every expert, else the experts' own allocations.
    if let Some(bank) = forms.bank().filter(|bank| bank.points.last() == Some(&bank.experts)) {
        let bank_plan = bank
            .form
            .draft
            .load()
            .expert_bank()
            .expect("a bank form holds an expert bank");
        let whole_paths = match &path_pricer {
            Some(pricer) => pricer.paths_at(bank.experts)?,
            None => bank_plan.paths,
        };
        let charge = bank.form.structure.charge_at_slots_places_and_paths(
            state_bytes,
            bank_plan,
            bank.experts,
            tier_places,
            whole_paths,
            pricing,
        )?;
        let fit = budget.fit(topology, device, &charge)?;
        if holds(&fit) {
            let sized = size_bank_with_host_tier(
                &bank.form.draft,
                bank.experts,
                tier_places,
                whole_paths,
                &charge,
                &budget,
            )?
            .expect("a whole bank needs no host tier");
            return fits_with_bank(&bank.form.structure, bank.form.draft.load(), Some(sized), &charge, fit);
        }
    }
    let resident = match forms.resident() {
        Some(resident) => {
            let charge = resident
                .structure
                .charge_with_state_and_pricing(state_bytes, &resident.draft, pricing)?;
            let fit = budget.fit(topology, device, &charge)?;
            if holds(&fit) {
                return fits_with_bank(&resident.structure, resident.draft.load(), None, &charge, fit);
            }
            Some(fit)
        }
        None => None,
    };
    // Streamed: the most slots that fit, a bank's fit falling as its slots
    // do, beside the host tier host memory holds; if the full-layer scratch
    // window's floor is short, walk the shared geometric `window_slots`
    // ladder (`BankForm::windowed`) for the largest fitting window.
    let least = match forms.bank() {
        Some(bank) => {
            let ceiling_slots = *bank
                .points
                .last()
                .expect("a bank form holds at least one slot point");
            let mut least_fit = match settle_bank_rung(
                &bank.form.draft,
                &bank.form.structure,
                ceiling_slots,
                state_bytes,
                tier_places,
                path_pricer.as_ref(),
                pricing,
                topology,
                device,
                &budget,
            )? {
                Ok((sized, charge, fit)) => {
                    return fits_with_bank(
                        &bank.form.structure,
                        bank.form.draft.load(),
                        Some(sized),
                        &charge,
                        fit,
                    );
                }
                Err(shortfall) => shortfall,
            };
            if !bank.windowed.is_empty() {
                let _scope = crate::programs::graph::sizing::SizingScope::enter_with_device(
                    bank.form.draft.device().properties(),
                );
                // `bank.windowed` is ordered by descending `window_slots`, so
                // floor memory falls monotonically toward the last rung. Probe
                // the smallest window first: if even it does not fit, no larger
                // window can fit, avoiding certifying intermediate rungs.
                let last_idx = bank.windowed.len() - 1;
                let eval_rung = |idx: usize| -> Result<
                    Option<
                        Result<
                            (
                                &StructuralCharge,
                                &crate::ModelLoadPlan,
                                crate::planning::ExpertBankPlan,
                                AssessmentMemoryCharge,
                                crate::AssessmentFit,
                            ),
                            crate::AssessmentFit,
                        >,
                    >,
                    AssessmentError,
                > {
                    let Some((rung, structure)) = bank.windowed[idx].structure(
                        &bank.form.draft,
                        bank.experts,
                        &prepared.definition,
                        &bank.form.structure,
                    )? else {
                        return Ok(None);
                    };
                    let win_ceiling = *rung
                        .points
                        .last()
                        .expect("a windowed bank rung holds at least one slot point");
                    match settle_bank_rung(
                        &rung.draft,
                        structure,
                        win_ceiling,
                        state_bytes,
                        tier_places,
                        path_pricer.as_ref(),
                        pricing,
                        topology,
                        device,
                        &budget,
                    )? {
                        Ok((sized, charge, fit)) => {
                            Ok(Some(Ok((structure, rung.draft.load(), sized, charge, fit))))
                        }
                        Err(shortfall) => Ok(Some(Err(shortfall))),
                    }
                };
                let mut smallest_fit = None;
                let mut smallest_idx = None;
                for idx in (0..=last_idx).rev() {
                    match eval_rung(idx)? {
                        Some(Ok(found)) => {
                            smallest_fit = Some(found);
                            smallest_idx = Some(idx);
                            break;
                        }
                        Some(Err(shortfall)) => {
                            if let (
                                AssessmentFitVerdict::DoesNotFit { deficit_bytes: cand, .. },
                                AssessmentFitVerdict::DoesNotFit { deficit_bytes: cur, .. },
                            ) = (&shortfall.verdict, &least_fit.verdict)
                            {
                                if cand < cur {
                                    least_fit = shortfall;
                                }
                            }
                            break;
                        }
                        None => continue,
                    }
                }
                if let (Some(mut best), Some(mut hi)) = (smallest_fit, smallest_idx) {
                    let mut lo = 0usize;
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        match eval_rung(mid)? {
                            Some(Ok(found)) => {
                                best = found;
                                hi = mid;
                            }
                            Some(Err(_)) | None => {
                                lo = mid + 1;
                            }
                        }
                    }
                    let (structure, load, sized, charge, fit) = best;
                    return fits_with_bank(structure, load, Some(sized), &charge, fit);
                }
            }
            least_fit
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

/// The drafter's decode demand for `definition` planned as `draft`, or
/// `Ok(None)` when `draft` does not draft.
pub fn draft_demand(
    definition: &ModelDefinition,
    draft: &ExecutionPlanDraft,
) -> Result<Option<DraftDemand>, AssessmentError> {
    if !draft.policy().method().drafts() {
        return Ok(None);
    }
    let _scope =
        crate::programs::graph::sizing::SizingScope::enter_with_device(draft.device().properties());
    let policy = draft.policy();
    let capacity = resource_capacity(draft)?;
    super::demand::draft_demand_direct(
        definition,
        draft.load(),
        policy.codec(),
        policy.method(),
        policy.limits(),
        capacity,
        draft.device().backend(),
    )
}

/// The prices of every launch of a load of `definition` planned as
/// `planned`, `resident` its plan holding its experts whole
/// ([`LaunchPrices`]): each plan's decode census ([`decode_demand`]) and,
/// when drafted, its drafter census ([`draft_demand`]), on `costs`' figures
/// under `resident`'s lookahead, which the two share.
pub fn launch_prices(
    definition: &ModelDefinition,
    resident: &ExecutionPlanDraft,
    planned: &ExecutionPlanDraft,
    costs: &ExecutorCosts,
) -> Result<LaunchPrices, AssessmentError> {
    let streamed = match planned.load().expert_bank() {
        Some(_) => Some(decode_demand(definition, planned)?.census()),
        None => None,
    };
    let target = TargetCensus::of(decode_demand(definition, resident)?.census(), streamed);
    let draft = draft_demand(definition, resident)?.map(|demand| demand.census());
    let mut censuses = target.censuses();
    if let Some(draft) = &draft {
        censuses.extend(draft.censuses());
    }
    let figures = costs.figures(&censuses);
    Ok(LaunchPrices::with_draft(target, draft, figures, lookahead(resident)))
}

/// Whether the draft's decode runs with lookahead: its resolved policy's.
pub fn lookahead(draft: &ExecutionPlanDraft) -> bool {
    draft.policy().limits().flights().pipelines()
}

/// One prepared model's resident decode speed at each requested depth, from
/// the planned executor's measured kernel costs.
///
/// A banked MoE model that fits resident executes its whole-resident form
/// (`TargetCensus::Banked { resident, .. }`), never its streamed form. When a
/// drafter is attached, speculative rounds are priced through
/// [`magnitude_cost_model::SpeculativePrice::resident`].
pub fn estimate_execution(
    prepared: &PreparedExecutionAssessment,
    costs: &ExecutorCosts,
    request: &AssessmentRequest,
) -> Result<Vec<PerformanceEstimate>, AssessmentError> {
    let census = match prepared.target_census() {
        TargetCensus::Resident(census) => census,
        TargetCensus::Banked { resident, .. } => resident,
    };
    let draft = prepared.draft_census();
    let mut censuses = vec![census];
    if let Some(draft) = draft {
        censuses.extend(draft.censuses());
    }
    let figures = costs.figures(&censuses);
    estimate_decode_speculative(
        census,
        draft,
        &Pricing { figures: &figures, lookahead: lookahead(prepared.draft()) },
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
        fn prepared(&self) -> Box<PreparedExecutionAssessment> {
            let forms = self
                .draft
                .floor_forms::<(), AssessmentError>(&self.definition, None, |draft| {
                    StructuralCharge::of(&self.definition, draft).map(|charge| (charge, ()))
                })
                .unwrap();
            prepare_execution_assessment(self.definition.clone(), forms).unwrap()
        }

        /// The prepared draft's fit.
        fn fit(&self, prepared: &PreparedExecutionAssessment) -> Result<ExecutionAssessment, AssessmentError> {
            assess_execution_fit(
                &self.definition,
                prepared,
                self.draft.device().graph_pricing(),
                &self.topology,
                &self.status,
            )
        }
    }

    fn environment(context_limit: u64) -> Environment {
        environment_within(context_limit, None)
    }

    /// The fixture planned on the host's device, selected as if the host
    /// had exactly `host_room` bytes free where it is given
    /// (`test_support::selected_with_room`).
    fn environment_within(context_limit: u64, host_room: Option<u64>) -> Environment {
        let inventory = std::sync::Arc::new(seismic::Inventory::discover(&[], seismic::ProgramStore::user()).unwrap());
        let catalog = seismic::Devices::new(inventory.clone());
        let selected = match host_room {
            Some(bytes) => crate::test_support::selected_with_room(&catalog, bytes),
            None => crate::platform::select_device(
                &catalog,
                ExecutionPath::Native,
                crate::platform::DeviceRequest::Automatic,
                &ReleaseRate::of_machine(None),
            )
            .unwrap(),
        };
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
            release: ReleaseRate::of_machine(None),
        }
    }

    /// The fit, whatever depths `_asked` names: it reads none.
    fn assess(environment: &Environment, _asked: &AssessmentRequest) -> ExecutionAssessment {
        environment.fit(&environment.prepared()).unwrap()
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
        let environment = environment_within(150_000, Some(16 << 20));
        let fit_context_tokens = match environment.fit(&environment.prepared()).unwrap() {
            ExecutionAssessment::Fits(fit) => fit.fit_context_tokens,
            ExecutionAssessment::DoesNotFit(shortfall) => shortfall.fit_context_tokens,
        };
        assert_eq!(fit_context_tokens, 150_000);
    }

    fn rtx3090_cuda(host: u64) -> (DeviceTopology, Status, crate::platform::SelectedDevice) {
        use crate::platform::GraphPricing;
        const GRAPHS: seismic::GraphCost = seismic::GraphCost::Pooled {
            graph: 92_663,
            launch: 8_102,
            granule: 2 << 20,
        };
        let topology = seismic::fixtures::recorded::rtx3090();
        let info = topology
            .executors()
            .iter()
            .find(|executor| executor.backend() == seismic::BackendName::Cuda)
            .unwrap()
            .clone();
        let status = Status {
            sample: seismic::Sample {
                taken: std::time::SystemTime::now(),
                host: crate::memory::ledger::tests::host_sample(u64::MAX, host),
                volumes: Vec::new(),
                executors: topology.executors().iter().map(|executor| executor.id()).collect(),
                grants: Vec::new(),
            },
            runtime: Default::default(),
        };
        let release = ReleaseRate::of_machine(None);
        let device = crate::platform::selected(&topology, &status, &release, info, GraphPricing::Measured(GRAPHS));
        (topology, status, device)
    }

    /// A 256K MoE model evaluated at a 32K tier charges
    /// `prompt_places_at(limits, 32768)` for `BankPart::{Keys, Routing,
    /// FileTable, ScratchTable}` (matching a fresh `ModelLoadPlan` at 32K)
    /// rather than freezing them at 256K.
    #[test]
    fn tier_scaled_bank_places_match_fresh_plan_at_smaller_context() {
        use crate::planning::fixtures::{
            gemma_moe_fixture_manifest, routed_fixture_definition, RoutedShape,
        };
        let (topology, status, device) = rtx3090_cuda(64 << 30);
        let shape = RoutedShape {
            layers: 4,
            experts: 64,
            selected: 8,
            hidden: 1024,
            intermediate: 512,
        };
        let prepare_at = |context_limit: u64| {
            let mut definition = routed_fixture_definition(shape);
            definition.decoder.context_limit = context_limit;
            let manifest = gemma_moe_fixture_manifest(&definition);
            let draft = ExecutionPlanner::prepare(
                &device,
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
            let forms = draft
                .floor_forms::<(), AssessmentError>(&definition, None, |draft| {
                    StructuralCharge::of(&definition, draft).map(|charge| (charge, ()))
                })
                .unwrap();
            prepare_execution_assessment(definition, forms).unwrap()
        };

        let prepared_256k = prepare_at(262_144);
        let prepared_32k = prepare_at(32_768);
        let places_256k = crate::planning::prompt_places_at(LIMITS, 262_144);
        let places_32k = crate::planning::prompt_places_at(LIMITS, 32_768);
        assert!(places_32k < places_256k);

        let ExecutionAssessment::Fits(fit_from_256k) = assess_execution_fit_at_context(
            32_768,
            &prepared_256k,
            device.graph_cost,
            &topology,
            &status,
            None,
        )
        .unwrap()
        else {
            panic!("the 4-layer MoE fixture fits the RTX 3090 at 32K");
        };
        let bank_from_256k = fit_from_256k
            .bank
            .expect("the routed fixture holds an expert bank");
        assert_eq!(bank_from_256k.places, places_32k);

        let bank_256k = prepared_256k
            .forms
            .bank()
            .unwrap()
            .form
            .draft
            .load()
            .expert_bank()
            .unwrap();
        let fresh_load_32k = prepared_32k.forms.bank().unwrap().form.draft.load();
        let fresh_bank_32k = fresh_load_32k.expert_bank().unwrap();
        assert_eq!(bank_256k.places, places_256k);
        assert_eq!(fresh_bank_32k.places, places_32k);
        assert!(
            bank_256k.places_storage_demand(places_32k).unwrap().shared
                < bank_256k.places_storage_demand(places_256k).unwrap().shared,
            "32K tier must charge strictly less for places-dependent bank parts than 256K"
        );
        assert_eq!(
            bank_256k
                .storage_demand_at(bank_from_256k.slots, places_32k, bank_from_256k.paths)
                .unwrap(),
            fresh_bank_32k
                .storage_demand_at(bank_from_256k.slots, fresh_bank_32k.places, bank_from_256k.paths)
                .unwrap(),
        );
        let model_256k = prepared_256k
            .forms
            .bank()
            .unwrap()
            .form
            .structure
            .bank_slot_model
            .as_ref()
            .unwrap();
        let tier_weight_demand = model_256k
            .weight_demand_at(bank_256k, bank_from_256k.slots, places_32k, bank_from_256k.paths)
            .unwrap();
        let fresh_weight_demand =
            ClaimDemand::checked_sum(fresh_load_32k.weight_demands().unwrap()).unwrap();
        assert_eq!(tier_weight_demand, fresh_weight_demand);
    }

    /// With measured executor/streamed figures,
    /// `assess_execution_fit_at_context` charges the exact same
    /// `BankPaths::Priced` staging/reserve/landing records as runtime
    /// `ExecutionPlanDraft::place` / `with_bank`.
    #[test]
    fn priced_assessment_matches_runtime_bank_paths_and_storage_demand() {
        use crate::planning::fixtures::{
            gemma_moe_fixture_manifest, routed_fixture_definition, RoutedShape,
        };
        let (topology, status, device) = rtx3090_cuda(64 << 30);
        let mut definition = routed_fixture_definition(RoutedShape {
            layers: 4,
            experts: 64,
            selected: 8,
            hidden: 1024,
            intermediate: 512,
        });
        definition.decoder.context_limit = 4096;
        let manifest = gemma_moe_fixture_manifest(&definition);
        let draft = ExecutionPlanner::prepare(
            &device,
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
        let forms = draft
            .floor_forms::<(), AssessmentError>(&definition, None, |draft| {
                StructuralCharge::of(&definition, draft).map(|charge| (charge, ()))
            })
            .unwrap();
        let prepared = prepare_execution_assessment(definition.clone(), forms).unwrap();

        let costs = crate::assessment::synthetic::costs();
        let figures = crate::assessment::synthetic::streamed_figures();
        let steps = costs.clone();
        let prices = crate::assessment::Priced::Measured(crate::PlanPrices::new(
            costs.clone(),
            figures.clone(),
            Box::new(move |def, banked| {
                let launch = crate::assessment::launch_prices(
                    def,
                    &banked.held_whole(def)?,
                    banked,
                    &steps,
                )?;
                Ok(Box::new(move |rows, depth| Ok(launch.launch(rows, depth)?)))
            }),
        ));

        let runtime = draft
            .clone()
            .with_bank::<AssessmentMemoryCharge>(&definition, None, &prices)
            .unwrap();
        let runtime_bank = runtime
            .draft()
            .load()
            .expert_bank()
            .expect("runtime plan holds an expert bank");
        assert!(
            matches!(
                runtime_bank.paths,
                crate::planning::BankPaths::Priced { records, .. }
                    if records.ahead > 0 && records.reserve > 0
            ),
            "synthetic prices must yield nonzero priced path records, got {:?}",
            runtime_bank.paths
        );

        let ExecutionAssessment::Fits(assessed_priced) = assess_execution_fit_at_context(
            4096,
            &prepared,
            device.graph_cost,
            &topology,
            &status,
            Some((&costs, &figures)),
        )
        .unwrap()
        else {
            panic!("priced assessment fits the RTX 3090");
        };
        let assessed_bank = assessed_priced
            .bank
            .expect("priced assessment holds an expert bank");
        assert_eq!(assessed_bank.paths, runtime_bank.paths);
        assert_eq!(assessed_bank.slots, runtime_bank.slots);
        assert_eq!(
            assessed_bank.storage_demand().unwrap(),
            runtime_bank.storage_demand().unwrap(),
        );
        let bank_form = prepared.forms.bank().unwrap();
        let prepared_bank = bank_form.form.draft.load().expert_bank().unwrap();
        let assessed_weight_demand = bank_form
            .form
            .structure
            .bank_slot_model
            .as_ref()
            .unwrap()
            .weight_demand_at(
                prepared_bank,
                assessed_bank.slots,
                assessed_bank.places,
                assessed_bank.paths,
            )
            .unwrap();
        let runtime_weight_demand =
            ClaimDemand::checked_sum(runtime.draft().load().weight_demands().unwrap()).unwrap();
        assert_eq!(assessed_weight_demand, runtime_weight_demand);

        let ExecutionAssessment::Fits(assessed_unpriced) = assess_execution_fit_at_context(
            4096,
            &prepared,
            device.graph_cost,
            &topology,
            &status,
            None,
        )
        .unwrap()
        else {
            panic!("unpriced assessment fits the RTX 3090");
        };
        assert_eq!(
            assessed_unpriced.bank.unwrap().paths,
            crate::planning::BankPaths::Unpriced
        );
    }

    /// A banked MoE model that fits resident must be priced using its
    /// `TargetCensus::Banked { resident, .. }` census, never its `streamed`
    /// census (which carries streaming-only calls and submission boundaries).
    #[test]
    fn resident_banked_moe_estimate_execution_uses_resident_census() {
        let env = environment(4096);
        let mut prepared = env.prepared();
        let resident_demand = prepared.demand.clone();
        let mut streamed_demand = resident_demand.clone();
        streamed_demand.boundaries = resident_demand.boundaries + 100;
        prepared.whole = Some(resident_demand.clone());
        prepared.demand = streamed_demand.clone();
        prepared.target_census = std::sync::OnceLock::new();

        let costs = crate::assessment::fixtures::costs();
        let req = request(4096, &[1024]);
        let estimates = estimate_execution(&prepared, &costs, &req).unwrap();
        let resident_census = resident_demand.census();
        let streamed_census = streamed_demand.census();
        let figures = costs.figures(&[&resident_census, &streamed_census]);
        let pricing = Pricing {
            figures: &figures,
            lookahead: lookahead(prepared.draft()),
        };
        let expected_resident =
            1.0 / magnitude_cost_model::step_seconds(&resident_census, &pricing, 1, 1024)
                .unwrap()
                .total();
        let streamed_tps =
            1.0 / magnitude_cost_model::step_seconds(&streamed_census, &pricing, 1, 1024)
                .unwrap()
                .total();
        assert!(
            expected_resident > streamed_tps,
            "streamed boundaries must make streamed slower than resident"
        );
        assert!(
            (estimates[0].tokens_per_second - expected_resident).abs() < 1e-9,
            "estimate_execution returned {:.3} tok/s, expected resident {:.3} (streamed was {:.3})",
            estimates[0].tokens_per_second,
            expected_resident,
            streamed_tps,
        );
    }

    /// Reproduces F-051 on `cliqa-r3-cuda` (30.97 GiB host RAM, A10G 23 GiB,
    /// `qwen3.8-flash-next:gguf:q2`): `DomainFit::remaining_bytes` on `HostRam`
    /// excludes the host expert tier's elastic cache (`28.93 GB`), so checking
    /// `remaining_bytes (28.96 GB) >= tables (28.43 GB)` without subtracting
    /// `host_tier_cache` double-spent host RAM and reported an impossible
    /// `host_bytes = 49.64 GB` on a 30.97 GiB host.
    #[test]
    fn placement_subtracts_host_tier_cache_before_caching_lookup_tables_in_ram() {
        let (topology, _, _) = rtx3090_cuda(33_258_901_504);
        let host_ram = topology.host_ram().id();

        const TOTAL_RAM: u64 = 33_258_901_504; // 30.97 GiB
        const REQUIRED_RAM: u64 = 4_303_246_443;
        const REMAINING_RAM: i64 = (TOTAL_RAM - REQUIRED_RAM) as i64; // 28.96 GB
        const DEVICE_HELD: u64 = 14_799_694_272;
        const HOST_ONLY_EXPERTS: u64 = 21_211_968_640;
        const HOST_TIER_CACHE: u64 = 28_930_211_840; // inclusive host tier cache
        const STREAMED_EXPERTS: u64 = 5_507_481_600;
        const LOOKUP_TABLES: u64 = 28_430_000_000;

        let tight_domains = [DomainFit {
            role: DomainRole::Staging,
            domain: host_ram,
            capacity_bytes: TOTAL_RAM,
            required_bytes: REQUIRED_RAM,
            remaining_bytes: REMAINING_RAM,
        }];
        let tight = ModelPlacement::from_parts(
            DEVICE_HELD,
            HOST_ONLY_EXPERTS,
            STREAMED_EXPERTS,
            LOOKUP_TABLES,
            HOST_TIER_CACHE,
            &tight_domains,
            &topology,
        )
        .unwrap();
        assert_eq!(tight.device_bytes, DEVICE_HELD);
        assert_eq!(tight.host_bytes, HOST_ONLY_EXPERTS, "host_bytes must not double-spend RAM with host tier cache");
        assert!(tight.host_bytes < TOTAL_RAM, "host_bytes ({}) must not exceed physical RAM ({TOTAL_RAM})", tight.host_bytes);
        assert_eq!(tight.disk_bytes, STREAMED_EXPERTS);
        assert_eq!(tight.lookup_disk_bytes, LOOKUP_TABLES, "lookup tables stay on SSD when host tier cache consumes RAM");

        // When host RAM genuinely fits BOTH host tier cache and lookup tables
        // (e.g. 128 GiB host), lookup tables are cached in host RAM.
        const LARGE_RAM: u64 = 128 * (1 << 30);
        let large_domains = [DomainFit {
            role: DomainRole::Staging,
            domain: host_ram,
            capacity_bytes: LARGE_RAM,
            required_bytes: REQUIRED_RAM,
            remaining_bytes: (LARGE_RAM - REQUIRED_RAM) as i64,
        }];
        let large = ModelPlacement::from_parts(
            DEVICE_HELD,
            HOST_ONLY_EXPERTS,
            STREAMED_EXPERTS,
            LOOKUP_TABLES,
            HOST_TIER_CACHE,
            &large_domains,
            &topology,
        )
        .unwrap();
        assert_eq!(large.host_bytes, HOST_ONLY_EXPERTS + LOOKUP_TABLES);
        assert_eq!(large.lookup_disk_bytes, 0);
    }
}
