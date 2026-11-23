//! One allocation-free, checked numerical startup decision.

use super::{
    ArtifactComponent, ArtifactComponentKind, AssessmentMemoryCharge, CapabilityPlan,
    ComponentPlan, ComponentSelection, HostTier, MAX_DRAFT_PROPOSALS, MemoryBudget, ModelLoadPlan, PassLaunches, PlannedMethod,
    ProgramPlan, ReadoutClassChoice, ResourceLimits, ResourcePlan, WeightPlan,
};
use crate::{
    ExecutionPath, UnstreamedExperts,
    assessment::{Priced, stage_timer},
    error::PlanError,
    platform::{FitReading, SelectedDevice},
};
use crate::experts::derived::slots_within;
use magnitude_artifacts::PackageManifest;
use magnitude_family_contracts::ModelDefinition;
use magnitude_state::KvCodec;
use seismic::{BackendName, ExecutorSelector};
use std::num::{NonZeroU64, NonZeroUsize};

/// The planned device by Seismic identity. The selector, not a name or
/// ordinal, is what the executing process resolves and opens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedDevice {
    selector: ExecutorSelector,
    backend: BackendName,
    name: String,
    tensor_operations: bool,
    /// Its largest single allocation, as discovery reports it: what an
    /// expert bank's lent allocations are bounded by.
    max_allocation_bytes: u64,
    /// How its host tier is held, where it has one
    /// (`SelectedDevice::host_tier`).
    host: Option<HostTier>,
    /// The executor and what its selection's fit questions read
    /// (`SelectedDevice::reading`): what the assessment's room asks of `Fit`
    /// ([`StableRoom::of`]).
    executor: seismic::Executor,
    reading: FitReading,
    /// What its recorded graphs are priced at for their census, and
    /// whether that is the device's own cost or its backend's bound
    /// (`SelectedDevice::graph_cost`).
    graph_pricing: crate::platform::GraphPricing,
}

impl PlannedDevice {
    /// How it makes its native allocations, as discovery reports them
    /// (`seismic::Limits::allocations`): what its pools' blocks are.
    pub fn allocations(&self) -> seismic::NativeAllocations {
        self.executor.limits().allocations()
    }

    pub fn selector(&self) -> ExecutorSelector {
        self.selector
    }

    pub fn backend(&self) -> BackendName {
        self.backend
    }

    pub fn properties(&self) -> Option<seismic::device::DeviceProperties> {
        self.executor.properties()
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether the device forms Metal tensor operations
    /// (`Executor::forms_tensor_operations`).
    pub fn tensor_operations(&self) -> bool {
        self.tensor_operations
    }

    /// How its host tier is held, where it has one
    /// (`SelectedDevice::host_tier`).
    pub fn host_tier(&self) -> Option<HostTier> {
        self.host
    }

    /// What its recorded graphs hold for their census.
    pub fn graph_cost(&self) -> seismic::GraphCost {
        self.graph_pricing.cost()
    }

    /// Whether its graphs are priced at the device's own cost or its
    /// backend's bound.
    pub fn graph_pricing(&self) -> crate::platform::GraphPricing {
        self.graph_pricing
    }

    pub fn topology(&self) -> &seismic::DeviceTopology {
        &self.reading.topology
    }

    pub fn executor(&self) -> &seismic::Executor {
        &self.executor
    }

    pub fn status(&self) -> &seismic::Status {
        &self.reading.status
    }
}

/// The most launches a prompt group chains for `context_limit` under `limits`:
/// every chunk of a whole-context prompt. Single owner shared by execution
/// planning, assessment tier sizing, and group-class reservation.
pub fn prompt_places_at(limits: ResourceLimits, context_limit: u64) -> u64 {
    context_limit.div_ceil(limits.max_launch_rows as u64)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedPolicy {
    path: ExecutionPath,
    method: PlannedMethod,
    codec: KvCodec,
    selection: ComponentSelection,
    limits: ResourceLimits,
}

impl ResolvedPolicy {
    pub fn path(&self) -> ExecutionPath {
        self.path
    }

    pub fn method(&self) -> PlannedMethod {
        self.method
    }

    pub fn codec(&self) -> KvCodec {
        self.codec
    }

    pub fn selection(&self) -> ComponentSelection {
        self.selection
    }

    pub fn limits(&self) -> ResourceLimits {
        self.limits
    }
}

/// The only startup value a program factory and allocator should consume.
/// Every field is immutable after the checked constructor returns.
///
/// Its device, load, program and resource plans are boxed, the draft's
/// boxes kept as it is admitted ([`ExecutionPlanDraft::admit`]): they are
/// nearly all of a plan's bytes, which every frame that moves or clones a
/// plan would otherwise hold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionPlan {
    device: Box<PlannedDevice>,
    components: ComponentPlan,
    load: Box<ModelLoadPlan>,
    programs: Box<ProgramPlan>,
    resources: Box<ResourcePlan>,
    capabilities: CapabilityPlan,
    policy: ResolvedPolicy,
}

/// Model topology on a backend: everything planning derives without a
/// device. A load's [`ExecutionPlanDraft`] is this plan on its selected
/// device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendPlan {
    backend: BackendName,
    components: ComponentPlan,
    load: ModelLoadPlan,
    programs: ProgramPlan,
    capabilities: CapabilityPlan,
    policy: ResolvedPolicy,
}

impl BackendPlan {
    pub fn backend(&self) -> BackendName {
        self.backend
    }

    pub fn programs(&self) -> &ProgramPlan {
        &self.programs
    }

    pub fn policy(&self) -> &ResolvedPolicy {
        &self.policy
    }

    pub fn load(&self) -> &ModelLoadPlan {
        &self.load
    }
}

/// Model topology and device choice before physical storage admission. Native
/// programs and their Seismic-owned graph storage can be prepared from this
/// value; only `admit` constructs an executable plan with resource ownership.
///
/// Its device, load and program plans are boxed: they are nine tenths of a
/// draft's bytes, which every frame that moves, clones or compares a draft
/// (each candidate of a placement search) would otherwise hold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionPlanDraft {
    device: Box<PlannedDevice>,
    components: ComponentPlan,
    load: Box<ModelLoadPlan>,
    programs: Box<ProgramPlan>,
    capabilities: CapabilityPlan,
    policy: ResolvedPolicy,
}

impl ExecutionPlanDraft {
    /// The draft of a load that traces its routes: the bank it streams
    /// through records each decode-form step's choices and keys, which every
    /// layer's route writes (`ExpertBankPlan::recording`).
    pub fn recording_routes(mut self) -> Self {
        self.load.recorded_routes = true;
        self
    }

    pub fn device(&self) -> &PlannedDevice {
        &self.device
    }

    pub fn programs(&self) -> &ProgramPlan {
        &self.programs
    }

    pub fn policy(&self) -> &ResolvedPolicy {
        &self.policy
    }

    pub fn load(&self) -> &ModelLoadPlan {
        &self.load
    }

    /// Store each readout scope in its prepared form (planes where its
    /// readers certify classes) before graph resources or persistent
    /// residency are admitted; a load `room` does not hold them in keeps
    /// every scope packed.
    pub fn finalize_readout(
        mut self,
        definition: &ModelDefinition,
        choices: &[ReadoutClassChoice],
        budget: &super::MemoryBudget,
    ) -> Result<Self, PlanError> {
        let canonical = self.clone();
        self.load = Box::new(
            self.load
                .finalize_readout(choices)
                .map_err(PlanError::InvalidDefinition)?,
        );
        *self.programs = self.load.program_plan(definition, self.policy.codec)?;
        if !choices.is_empty() && budget.fit_whole(&self, &self.memory_charge(definition)?)?.short.is_some() {
            return canonical.finalize_readout(definition, &[], budget);
        }
        Ok(self)
    }

    /// The draft whose expert bank, where it has one, holds a CPU site's
    /// words and buffers for exactly the streamed layers its qualification
    /// admitted one for or holds one awaiting its verdict (`sited` of the
    /// bank, `ExpertBankPlan::with_sites`),
    /// before graph resources or persistent residency are admitted: the
    /// charge follows admission.
    pub fn finalize_sites(
        mut self,
        sited: impl FnOnce(&super::ExpertBankPlan) -> Result<Vec<super::LayerSite>, String>,
    ) -> Result<Self, PlanError> {
        if let Some(bank) = self.load.bank.take() {
            let sited = sited(&bank).map_err(PlanError::ResourcePlanning)?;
            self.load.bank = Some(bank.with_sites(sited).map_err(PlanError::ResourcePlanning)?);
        }
        Ok(self)
    }

    /// Each streamed layer's request of its CPU site's verdict of the
    /// measurement job, where it holds a site: what a load prices the site
    /// on, and what a tool asks the job for before a load. The one
    /// derivation of the plan's sites: the kernels of the load's backend
    /// that serve one (`BankSite::served`), its form, and `profile`'s
    /// requirement of it (`site_requirement`: none where it approves no CPU
    /// site). By layer of the bank; none without one.
    pub fn site_requests(
        &self,
        profile: &crate::ResolvedNumericalProfile,
    ) -> Result<Vec<Option<crate::assessment::SiteRequest>>, String> {
        let Some(bank) = self.load.expert_bank() else {
            return Ok(Vec::new());
        };
        (0..bank.layers().len())
            .map(|layer| {
                let scope = bank.layers()[layer].scope;
                let index = bank.layers()[layer].sublayer();
                let (slot, binding) = self
                    .programs
                    .target()
                    .blocks()
                    .get(index.block as usize)
                    .and_then(|block| block.bank())
                    .ok_or_else(|| format!("{scope:?} has no bank entries"))?;
                if !bank.sited(layer) || !crate::native::BankSite::served(slot, binding, self.device.backend)? {
                    return Ok(None);
                }
                let form = crate::native::SiteForm::of(slot);
                Ok(profile
                    .site_requirement(form.activation.dtype())
                    .map(|requirement| crate::assessment::SiteRequest::of(&binding, form, requirement)))
            })
            .collect()
    }

    /// The draft's assessment charge at the model's context.
    fn memory_charge(&self, definition: &ModelDefinition) -> Result<AssessmentMemoryCharge, PlanError> {
        crate::assessment::execution_memory_charge(definition, self).map_err(charge_error)
    }

    /// What the draft allocates in the device's allocation domain at the
    /// model's context, derived from the single [`StoreDemands`] owner.
    pub(crate) fn allocation_bytes(&self, definition: &ModelDefinition) -> Result<u64, PlanError> {
        self.memory_charge(definition)?
            .store_demands(self.device.topology(), self.device.executor())
            .map(|demands| demands.allocation_bytes())
            .map_err(PlanError::ResourcePlanning)
    }

    /// The draft with its linked host tier (its records in host storage)
    /// sized at load: the most host slots' records
    /// (`ExpertBankPlan::host_charge`, the steps where it rises, up to every
    /// expert, `ExpertBankPlan::host_ceiling`) the room host RAM holds beside
    /// the load's charge at the tier's floor
    /// (`room`'s [`SettleRoom::host_left`]: the load's held reading, the
    /// assessment's stable fit); none past its floor beside a bank
    /// of every expert, which misses nothing. The tier's
    /// size is this load-time decision. A tier of every expert is read in at
    /// load behind every step's reads (`Prefetch::cover`); a smaller one
    /// starts empty and fills by use (file misses, the hints' reads and a
    /// prompt's seeds land into it).
    ///
    /// `charge` is the draft's, which holds the tier at its floor, as the
    /// bank commits it at load: a size's charge is it with the tier's growth
    /// past the floor in host storage beside it, the one term the size
    /// moves.
    ///
    /// Every file miss lands in a tier record, so the tier's floor, its
    /// landing reserve (`ExpertBankPlan::host_records` at no host slot: a
    /// request's misses and the hints' records, every staging record's
    /// landing at once), is what makes a step live: a reading that admits
    /// not even the floor leaves the experts unstreamed, with why.
    fn with_host_tier(
        mut self,
        charge: &AssessmentMemoryCharge,
        budget: &super::MemoryBudget,
    ) -> Result<Result<Self, UnstreamedExperts>, PlanError> {
        if self.device.host != Some(HostTier::Linked) {
            return Ok(Ok(self));
        }
        let Some(ceiling) = self.bank().host_ceiling().filter(|&ceiling| ceiling > 0 && !self.bank().whole()) else {
            return Ok(Ok(self));
        };
        let bank = self.bank();
        let host_charge = |slots: u64| bank.host_charge(slots).map_err(PlanError::ResourcePlanning);
        let left = |slots: u64| self.host_left(charge, budget, slots);
        if left(0)? < 0 {
            return Ok(Err(self.untiered_reason()));
        }
        // The most host memory admits.
        let points = rising(0, ceiling, &host_charge)?;
        let mut most = points[0];
        for &host in &points {
            if left(host)? < 0 {
                break;
            }
            most = host;
        }
        self.bank_mut().host_slots = most;
        Ok(Ok(self))
    }

    /// Clone only the draft's `ExpertBankPlan` resized to `(slots, slots)` at
    /// `PassLaunches::FLOOR` and size its linked host tier against `budget`
    /// beside `charge`, without cloning `ModelLoadPlan` or `ProgramPlan`.
    pub(crate) fn bank_at_slots_with_host_tier(
        &self,
        slots: u64,
        charge: &AssessmentMemoryCharge,
        budget: &super::MemoryBudget,
    ) -> Result<Result<super::ExpertBankPlan, UnstreamedExperts>, PlanError> {
        let mut bank = self.bank().clone();
        if slots < bank.floor() || slots > bank.widest().get() {
            return Err(PlanError::Topology(
                "an expert bank holds from its floor to its ceiling, at most every expert of each layer",
            ));
        }
        bank.slots = slots;
        bank.ceiling = slots;
        bank.resident = slots == bank.widest().get();
        bank.pass = PassLaunches::FLOOR;
        if self.device.host != Some(HostTier::Linked) {
            return Ok(Ok(bank));
        }
        let Some(ceiling) = bank.host_ceiling().filter(|&ceiling| ceiling > 0 && !bank.whole()) else {
            return Ok(Ok(bank));
        };
        let host_charge = |host_slots: u64| bank.host_charge(host_slots).map_err(PlanError::ResourcePlanning);
        let left = |host_slots: u64| -> Result<i64, PlanError> {
            let growth = bank.host_cache(host_slots).map_err(PlanError::ResourcePlanning)?;
            budget.host_left(self, charge, growth)
        };
        if left(0)? < 0 {
            let reserve = bank.landing_records();
            let reason = format!("host memory admits not even the tier's landing reserve of {reserve} records a group");
            return Ok(Err(UnstreamedExperts::Untiered { backend: self.device.backend, reason }));
        }
        let points = rising(0, ceiling, &host_charge)?;
        let mut most = points[0];
        for &host in &points {
            if left(host)? < 0 {
                break;
            }
            most = host;
        }
        bank.host_slots = most;
        Ok(Ok(bank))
    }

    /// The host RAM `charge`, the draft's with its tier at its floor, leaves
    /// in `budget` beside a tier of `slots`: negative where it does not fit.
    fn host_left(&self, charge: &AssessmentMemoryCharge, budget: &super::MemoryBudget, slots: u64) -> Result<i64, PlanError> {
        let growth = self.bank().host_cache(slots).map_err(PlanError::ResourcePlanning)?;
        budget.host_left(self, charge, growth)
    }

    /// Why the draft's experts cannot stream beside its linked host tier
    /// charged `charge` (its tier at its floor): host memory admits not
    /// even the tier's landing reserve. None where it does, or without a
    /// linked tier. Every file miss lands in a tier record, so the floor is
    /// what makes a step live, and a plan short of it is unstreamed, not
    /// short of device memory.
    fn untiered(&self, charge: &AssessmentMemoryCharge, budget: &super::MemoryBudget) -> Result<Option<UnstreamedExperts>, PlanError> {
        let tiered = self.bank().host_ceiling().is_some_and(|ceiling| ceiling > 0) && !self.bank().whole();
        if self.device.host != Some(HostTier::Linked) || !tiered {
            return Ok(None);
        }
        Ok((self.host_left(charge, budget, 0)? < 0).then(|| self.untiered_reason()))
    }

    /// The tier's landing reserve unadmitted, as a reason.
    fn untiered_reason(&self) -> UnstreamedExperts {
        let reserve = self.bank().landing_records();
        let reason = format!("host memory admits not even the tier's landing reserve of {reserve} records a group");
        UnstreamedExperts::Untiered { backend: self.device.backend, reason }
    }

    /// The draft's expert bank, which a streamed plan holds.
    fn bank_mut(&mut self) -> &mut super::ExpertBankPlan {
        self.load.bank.as_mut().expect("a streamed plan holds its bank")
    }

    /// The draft of an elastic load as its assessment plans it, in the
    /// stable fit's room ([`StableRoom`]), with its charge: its experts
    /// placed whole where they fit ([`Self::placed`]), else streamed at the
    /// pass the plan's `prices` settle ([`Unplaced::streamed`]). A load no
    /// plan holds is refused with why.
    pub fn with_bank<C: HeldCharge>(
        self,
        definition: &ModelDefinition,
        bound: Option<WeightBound>,
        prices: &Priced<super::PlanPrices>,
    ) -> Result<ChargedDraft<C>, PlanError> {
        let budget = super::MemoryBudget::of_stable(&self.device)?;
        match self.placed(definition, bound, prices, &budget)? {
            Placed::Whole(whole) => Ok(whole),
            Placed::Short(short) => short.streamed(definition, prices, &budget),
        }
    }

    /// A load's first stage, its placement: its experts whole, in a whole
    /// bank or as their own allocations, wherever `budget` holds them under
    /// the load's own `bound` ([`Self::place`]), decided from memory alone;
    /// or what streaming them starts from ([`Unplaced`]). The figures it
    /// decided on are reported (`the load's experts are placed`).
    pub fn placed<C: HeldCharge>(
        self,
        definition: &ModelDefinition,
        bound: Option<WeightBound>,
        prices: &Priced<super::PlanPrices>,
        budget: &super::MemoryBudget,
    ) -> Result<Placed<C>, PlanError> {
        let mut held = HeldCharges::default();
        let capacity = budget.capacity();
        let held_bound = HeldBound::of(bound, capacity);
        let Placement { figures, outcome } = self.place(definition, held_bound, prices, &mut held, budget, &mut load_confirm(budget))?;
        figures.report();
        Ok(match outcome {
            Placing::Whole { draft, charge, found: () } => Placed::Whole(ChargedDraft { draft, charge }),
            Placing::Short { ceiling, short } => Placed::Short(Unplaced { draft: self, ceiling, bound, held, short }),
            Placing::None(no) => return Err(refusal(no)),
        })
    }

    /// A load's experts streamed, whatever whole form `budget` holds them
    /// in: what a `PlanForm::Streamed` load streams from ([`Unplaced::streamed`]).
    /// A load whose experts cannot stream is refused with why.
    pub fn streaming<C: HeldCharge>(
        self,
        definition: &ModelDefinition,
        bound: Option<WeightBound>,
        budget: &super::MemoryBudget,
    ) -> Result<Unplaced<C>, PlanError> {
        let capacity = budget.capacity();
        match self.bank_ceiling(definition, HeldBound::of(bound, capacity), None)? {
            Ok(ceiling) => Ok(Unplaced { draft: self, ceiling, bound, held: HeldCharges::default(), short: None }),
            Err(no) => Err(refusal(no)),
        }
    }

    /// How a load holds its experts under `bound` ([`HeldBound`]), the one
    /// decision a load and its assessment share, in two stages: placed
    /// whole where `budget` holds them so ([`Self::place`]), from memory
    /// alone; else streamed, in the bank and host tier `budget` holds beside
    /// the plan's mandatory part, its arena among it, at the pass `prices`
    /// settle ([`Self::streamed_in`]). Each way of holding the
    /// experts is charged once ([`HeldCharges`]), and each plan confirmed by
    /// `confirm`, which judges a plan by its charge, giving what it found or
    /// the bytes the plan is short. The draft held comes with the charge it
    /// admitted.
    pub fn with_held_experts<T, E: From<PlanError>>(
        &self,
        definition: &ModelDefinition,
        bound: HeldBound,
        prices: &Priced<super::PlanPrices>,
        budget: &super::MemoryBudget,
        mut confirm: impl FnMut(&Self, &AssessmentMemoryCharge) -> Result<Result<T, NonZeroU64>, E>,
    ) -> Result<Result<(ChargedDraft, T), NoBank>, E> {
        let mut held = HeldCharges::<AssessmentMemoryCharge>::default();
        let ceiling = match self.place(definition, bound, prices, &mut held, budget, &mut confirm)?.outcome {
            Placing::Whole { draft, charge, found } => return Ok(Ok((ChargedDraft { draft, charge }, found))),
            Placing::Short { ceiling, .. } => ceiling,
            Placing::None(no) => return Ok(Err(no)),
        };
        Ok(self.streamed_in(ceiling, bound, definition, prices, &mut held, budget, &mut confirm)?
            .map(|(draft, charge, found)| (ChargedDraft { draft, charge }, found)))
    }

    fn place<C: HeldCharge, T, E: From<PlanError>>(
        &self,
        definition: &ModelDefinition,
        bound: HeldBound,
        prices: &Priced<super::PlanPrices>,
        held: &mut HeldCharges<C>,
        budget: &super::MemoryBudget,
        confirm: &mut impl FnMut(&Self, &AssessmentMemoryCharge) -> Result<Result<T, NonZeroU64>, E>,
    ) -> Result<Placement<C, T>, E> {
        let mut figures = PlacementFigures { pricing: self.device.graph_pricing, form: None, bank: None, resident: None };
        let placed = |figures, outcome| Ok(Placement { figures, outcome });
        let ceiling = match self.bank_ceiling(definition, bound, Some(prices))? {
            Ok(ceiling) => ceiling,
            Err(NoBank::Unstreamed(reason)) => {
                let Some(chains) = self.whole_chains(definition, bound)? else {
                    return placed(figures, Placing::None(NoBank::Unstreamed(reason)));
                };
                // The longest chain the budget holds, each as `confirm` decides it.
                let mut last_short = None;
                for places in chains {
                    let resident = self.held_whole_at(definition, places)?;
                    let charge = held.charge(definition, &resident)?;
                    let fit = budget.fit_whole(&resident, charge.charge())?;
                    figures.resident = Some(Need { fit: Some(fit), terms: NeedTerms::of(&resident, charge.charge())? });
                    if let Some(deficit_bytes) = fit.short {
                        last_short = Some(NoBank::Short {
                            deficit_bytes,
                            required: fit.limiting_required,
                            available: fit.limiting_available,
                        });
                    } else {
                        match confirm(&resident, charge.charge())? {
                            Ok(found) => {
                                figures.form = Some(WholeForm::Resident);
                                return placed(figures, Placing::Whole { draft: resident, charge, found });
                            }
                            Err(deficit_bytes) => {
                                last_short = Some(NoBank::Short {
                                    deficit_bytes,
                                    required: fit.limiting_required,
                                    available: fit.limiting_available,
                                });
                            }
                        }
                    }
                }
                return placed(figures, Placing::None(last_short.unwrap_or(NoBank::Unstreamed(reason))));
            }
            Err(no) => return placed(figures, Placing::None(no)),
        };
        if self.load.bank.as_ref().is_some_and(|bank| !bank.reaches_whole()) {
            return placed(figures, Placing::Short { ceiling, short: None });
        }
        let mut last_short = None;
        let experts = ceiling.whole.bank().slots;
        if ceiling.ceiling == experts {
            let mut fitting = Vec::new();
            for pass in super::pass_ladder(ceiling.places) {
                let draft = ceiling.whole.resized(experts, experts, pass)?;
                let charge = held.charge(definition, &draft)?;
                let fit = budget.fit_whole(&draft, charge.charge())?;
                figures.bank = Some(Need { fit: Some(fit), terms: NeedTerms::of(&draft, charge.charge())? });
                if fit.short.is_some() {
                    last_short = Some((fit.limiting_required, fit.limiting_available));
                    break;
                }
                fitting.push((draft, charge, fit));
            }
            for (draft, charge, fit) in fitting.into_iter().rev() {
                match confirm(&draft, charge.charge())? {
                    Ok(found) => {
                        figures.form = Some(WholeForm::Bank);
                        return placed(figures, Placing::Whole { draft, charge, found });
                    }
                    Err(_) => {
                        last_short = Some((fit.limiting_required, fit.limiting_available));
                    }
                }
            }
        }
        for places in self.whole_chains(definition, bound)?.into_iter().flatten() {
            let resident = self.held_whole_at(definition, places)?;
            let charge = held.charge(definition, &resident)?;
            let fit = budget.fit_whole(&resident, charge.charge())?;
            figures.resident = Some(Need { fit: Some(fit), terms: NeedTerms::of(&resident, charge.charge())? });
            if fit.short.is_some() {
                last_short = Some((fit.limiting_required, fit.limiting_available));
            } else {
                match confirm(&resident, charge.charge())? {
                    Ok(found) => {
                        figures.form = Some(WholeForm::Resident);
                        return placed(figures, Placing::Whole { draft: resident, charge, found });
                    }
                    Err(_) => {
                        last_short = Some((fit.limiting_required, fit.limiting_available));
                    }
                }
            }
        }
        placed(figures, Placing::Short { ceiling, short: last_short })
    }

    /// The second stage, for experts no whole plan holds: the bank the room
    /// holds beside the arena from `ceiling`, at the pass `prices` settle
    /// ([`Self::settle_bank`]), with the free records its model store's
    /// path needs at the slots it settles at ([`Self::with_paths`]), and
    /// its host tier ([`Self::with_host_tier`]),
    /// whose size leaves the charge as it is: the charge holds the tier at
    /// its floor. Experts the host cannot stream beside their tier are held
    /// as their own allocations within `bound`, whether the device holds
    /// them `confirm`'s, as experts that cannot stream are
    /// ([`Self::place`]).
    fn streamed_in<C: HeldCharge, T, E: From<PlanError>>(
        &self,
        ceiling: BankCeiling,
        bound: HeldBound,
        definition: &ModelDefinition,
        prices: &Priced<super::PlanPrices>,
        held: &mut HeldCharges<C>,
        budget: &super::MemoryBudget,
        confirm: &mut impl FnMut(&Self, &AssessmentMemoryCharge) -> Result<Result<T, NonZeroU64>, E>,
    ) -> Result<Result<(Self, C, T), NoBank>, E> {
        let (ceiling, settled) = match self.with_paths(ceiling, bound, definition, prices, held, budget, confirm)? {
            Ok(settled) => settled,
            Err(no) => return Ok(Err(no)),
        };
        let reason = match settled {
            Ok((draft, charge, found)) => match draft.with_host_tier(charge.charge(), budget)? {
                Ok(draft) => return Ok(Ok((draft, charge, found))),
                Err(reason) => reason,
            },
            Err(NoBank::Unstreamed(reason)) => reason,
            Err(no) => return Ok(Err(no)),
        };
        let Some(resident) = ceiling.whole.held_whole_within(definition, bound)? else {
            return Ok(Err(NoBank::Unstreamed(reason)));
        };
        let charge = held.charge(definition, &resident)?;
        Ok(match confirm(&resident, charge.charge())? {
            Ok(found) => Ok((resident, charge, found)),
            Err(_) => Err(NoBank::Unstreamed(reason)),
        })
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn with_paths<C: HeldCharge, T, E: From<PlanError>>(
        &self,
        mut ceiling: BankCeiling,
        bound: HeldBound,
        definition: &ModelDefinition,
        prices: &Priced<super::PlanPrices>,
        held: &mut HeldCharges<C>,
        budget: &super::MemoryBudget,
        confirm: &mut impl FnMut(&Self, &AssessmentMemoryCharge) -> Result<Result<T, NonZeroU64>, E>,
    ) -> Result<Result<(BankCeiling, Result<(Self, C, T), NoBank>), NoBank>, E> {
        let bank_prices = super::prices::measured(match prices {
            Priced::Measured(prices) => prices.of_bank(&ceiling.whole, definition)?,
            Priced::Pending(reason) => Priced::Pending(reason.clone()),
            Priced::Unavailable(reason) => Priced::Unavailable(reason.clone()),
        })?;
        if ceiling.whole.bank().paths == super::BankPaths::Unpriced {
            let whole = ceiling.whole.clone();
            ceiling = match self.rebuilt_from(&whole, definition, bound, CeilingPaths::Priced(&bank_prices), Ok)? {
                Ok(ceiling) => ceiling,
                Err(no) => return Ok(Err(no)),
            };
        }
        let initial = bank_paths(&bank_prices, ceiling.whole.bank(), ceiling.ceiling)?;
        let (settled_paths, outcome, settles) = settle_bank_paths::<_, E>(initial, |paths| {
            if ceiling.whole.bank().paths != paths {
                let whole = ceiling.whole.clone();
                match self.rebuilt_from(&whole, definition, bound, CeilingPaths::Fixed, |load| load.with_bank_paths(paths))? {
                    Ok(rebuilt) => ceiling = rebuilt,
                    Err(no) => return Ok((Err(no), None)),
                }
            }
            let settled = self.settled_in(&ceiling, bound, definition, prices, held, budget, confirm)?;
            let next = match &settled {
                Ok((draft, ..)) => Some(bank_paths(&bank_prices, draft.bank(), draft.bank().slots)?),
                Err(_) => None,
            };
            Ok((Ok(settled), next))
        })?;
        match outcome {
            Err(no) => Ok(Err(no)),
            Ok(settled) => {
                if let Ok((draft, ..)) = &settled {
                    let records = settled_paths.records();
                    tracing::info!(
                        slots = draft.bank().slots,
                        ahead = records.ahead,
                        reserve = records.reserve,
                        settles,
                        pass = draft.bank().pass.get(),
                        "the bank's path records are settled"
                    );
                }
                Ok(Ok((ceiling, settled)))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn settled_in<C: HeldCharge, T, E: From<PlanError>>(
        &self,
        ceiling: &BankCeiling,
        bound: HeldBound,
        definition: &ModelDefinition,
        prices: &Priced<super::PlanPrices>,
        held: &mut HeldCharges<C>,
        budget: &super::MemoryBudget,
        confirm: &mut impl FnMut(&Self, &AssessmentMemoryCharge) -> Result<Result<T, NonZeroU64>, E>,
    ) -> Result<Result<(Self, C, T), NoBank>, E> {
        let mut settled = Self::settle_bank(ceiling, definition, prices, held, budget, confirm)?;
        if matches!(settled, Err(NoBank::Short { .. })) {
            for &w in &ceiling.whole.bank().window_ladder()[1..] {
                let candidate = match self.with_window(&ceiling.whole, definition, bound, w, CeilingPaths::Fixed)? {
                    Ok(c) => Self::settle_bank(&c, definition, prices, held, budget, confirm)?,
                    Err(no) => Err(no),
                };
                match candidate {
                    Ok(found) => {
                        settled = Ok(found);
                        break;
                    }
                    Err(short @ NoBank::Short { deficit_bytes, .. }) => {
                        if let Err(NoBank::Short { deficit_bytes: least, .. }) = &settled {
                            if deficit_bytes < *least {
                                settled = Err(short);
                            }
                        }
                    }
                    Err(_) => {}
                }
            }
        }
        Ok(settled)
    }

    /// The structure every plan of the draft's bank is granted from: its
    /// bank built once, of every expert, and its ceiling, the most slots
    /// whose weights the device holds under `bound` (at most a banked
    /// draft's ceiling, which re-planning never raises), where the bank's
    /// curve ends, and its host tier's cap, which only a weight cap sets (at
    /// most a banked draft's, which re-planning never raises either); or
    /// why no bank streams under `bound`. Its floor, its program plan and
    /// everything but its slots and pass are of its structure alone, so each
    /// bank is that one resized ([`Self::resized`]), and its weights at any
    /// slots are that bank's less what its record reservations release
    /// (`ExpertBankPlan::resident_charge`). Its record tensors are reserved
    /// for every expert and commit their slots, each within the device's
    /// single allocation. Its prompt groups chain every chunk of a
    /// whole-context prompt (`with_chain`): a group's places are granted
    /// when it opens.
    fn bank_ceiling(&self, definition: &ModelDefinition, bound: HeldBound, prices: Option<&Priced<super::PlanPrices>>) -> Result<Result<BankCeiling, NoBank>, PlanError> {
        let substrate = match bank_substrate(self.device.backend, self.device.executor.limits().allocations(), self.device.host_tier(), definition) {
            Ok(substrate) => substrate,
            Err(error) => return unstreamed(error),
        };
        let places = self.prompt_places(definition);
        let whole = match super::bank::routed_experts(definition).and_then(|experts| {
            let load = self.load.clone().unbanked().with_expert_bank(
                definition,
                experts,
                None,
                places,
                self.policy.limits.flights(),
                substrate,
            )?;
            let programs = load.program_plan(definition, self.policy.codec)?;
            self.with_load_and_programs(load, programs).with_chain(places)
        }) {
            Ok(whole) => whole,
            Err(error) => return unstreamed(error),
        };
        let bank_prices = match prices {
            Some(Priced::Measured(prices)) => match prices.of_bank(&whole, definition)? {
                Priced::Measured(bank_prices) => Some(bank_prices),
                Priced::Pending(_) | Priced::Unavailable(_) => None,
            },
            _ => None,
        };
        let mode = bank_prices.as_ref().map_or(CeilingPaths::Fixed, CeilingPaths::Priced);
        let w_hi = self.load.bank.as_ref().map_or(whole.bank().window_slots(), |bank| bank.window_slots());
        let w_lo = whole.bank().round_records();
        let (whole, mut res) = match w_hi == whole.bank().window_slots() {
            true => match self.ceiling_of(whole, definition, bound, places, mode)? {
                Ok(found) => return Ok(Ok(found)),
                Err((whole, no)) => (whole, Err(no)),
            },
            false => {
                let res = self.with_window(&whole, definition, bound, w_hi, mode)?;
                (whole, res)
            }
        };
        if matches!(res, Err(NoBank::Bound { .. } | NoBank::Unstreamed(UnstreamedExperts::Allocation { .. }))) {
            for &w in &window_ladder(w_hi, w_lo)[1..] {
                match self.with_window(&whole, definition, bound, w, mode)? {
                    Ok(found) => {
                        res = Ok(found);
                        break;
                    }
                    Err(next) => res = Err(next),
                }
            }
        }
        Ok(res)
    }

    /// Clone this draft's metadata with a replacement `load` and `programs`,
    /// avoiding cloning and immediately dropping `self.load` and `self.programs`.
    fn with_load_and_programs(&self, load: ModelLoadPlan, programs: ProgramPlan) -> Self {
        Self {
            device: self.device.clone(),
            components: self.components.clone(),
            load: Box::new(load),
            programs: Box::new(programs),
            capabilities: self.capabilities,
            policy: self.policy.clone(),
        }
    }

    /// Clone this draft's metadata and programs with a replacement `load`,
    /// avoiding cloning and immediately dropping `self.load`.
    fn with_load(&self, load: ModelLoadPlan) -> Self {
        Self {
            device: self.device.clone(),
            components: self.components.clone(),
            load: Box::new(load),
            programs: self.programs.clone(),
            capabilities: self.capabilities,
            policy: self.policy.clone(),
        }
    }

    /// Rebuild `whole` with `window_slots` and compute its [`BankCeiling`]
    /// under `bound`, keeping its target program plan in sync.
    fn with_window(&self, whole: &Self, definition: &ModelDefinition, bound: HeldBound, window_slots: u64, mode: CeilingPaths<'_>) -> Result<Result<BankCeiling, NoBank>, PlanError> {
        self.rebuilt_from(whole, definition, bound, mode, |load| load.banked_as(definition, whole.bank(), window_slots))
    }

    /// `whole` with its load plan changed by `change` and its target
    /// program plan and chain planned again, and its [`BankCeiling`] under
    /// `bound`.
    fn rebuilt_from(
        &self,
        whole: &Self,
        definition: &ModelDefinition,
        bound: HeldBound,
        mode: CeilingPaths<'_>,
        change: impl FnOnce(ModelLoadPlan) -> Result<ModelLoadPlan, PlanError>,
    ) -> Result<Result<BankCeiling, NoBank>, PlanError> {
        let prior_ceiling = self.load.bank.as_ref().map(|bank| bank.ceiling);
        let prior_host_cap = self.load.bank.as_ref().and_then(|bank| bank.host_cap);
        whole.rebuilt_under(definition, bound, mode, prior_ceiling, prior_host_cap, change)
    }

    /// Rebuild `self` (a banked draft) with `window_slots` at `experts` total
    /// slots and compute its [`BankCeiling`] under `bound`, respecting any
    /// prior ceiling/host_cap.
    fn with_window_from_experts(
        &self,
        definition: &ModelDefinition,
        experts: u64,
        bound: HeldBound,
        window_slots: u64,
        prior_ceiling: Option<u64>,
        prior_host_cap: Option<u64>,
    ) -> Result<Result<BankCeiling, NoBank>, PlanError> {
        let bank = self.bank();
        let (places, flights, substrate, paths) =
            (bank.places, bank.flights, bank.substrate(), bank.paths);
        self.rebuilt_under(definition, bound, CeilingPaths::Fixed, prior_ceiling, prior_host_cap, |load| {
            load.unbanked()
                .with_expert_bank(definition, experts, Some(window_slots), places, flights, substrate)?
                .with_bank_paths(paths)
        })
    }

    /// Rebuild `self` (a whole-bank draft) with its load plan changed by
    /// `change`, its target program plan and chain planned again, and its
    /// [`BankCeiling`] under `bound`, respecting any prior ceiling/host_cap.
    fn rebuilt_under(
        &self,
        definition: &ModelDefinition,
        bound: HeldBound,
        mode: CeilingPaths<'_>,
        prior_ceiling: Option<u64>,
        prior_host_cap: Option<u64>,
        change: impl FnOnce(ModelLoadPlan) -> Result<ModelLoadPlan, PlanError>,
    ) -> Result<Result<BankCeiling, NoBank>, PlanError> {
        let places = self.bank().places;
        let load = change((*self.load).clone())?;
        let programs = load.program_plan(definition, self.policy.codec)?;
        let rebuilt = self.with_load_and_programs(load, programs).with_chain(places)?;
        Ok(rebuilt
            .ceiling_of_under(definition, bound, places, mode, prior_ceiling, prior_host_cap)?
            .map_err(|(_, no)| no))
    }

    /// Compute the [`BankCeiling`] of `whole` under `bound`, pricing each
    /// slot count's path records per `mode` so both refusal (`needed`) and
    /// admission (`ceiling`) evaluate the same function. Returns `whole` on
    /// failure so callers can retry smaller scratch windows without cloning
    /// on the happy path.
    fn ceiling_of(
        &self,
        whole: Self,
        definition: &ModelDefinition,
        bound: HeldBound,
        places: u64,
        mode: CeilingPaths<'_>,
    ) -> Result<Result<BankCeiling, (Self, NoBank)>, PlanError> {
        let prior_ceiling = self.load.bank.as_ref().map(|bank| bank.ceiling);
        let prior_host_cap = self.load.bank.as_ref().and_then(|bank| bank.host_cap);
        whole.ceiling_of_under(definition, bound, places, mode, prior_ceiling, prior_host_cap)
    }

    /// Compute the [`BankCeiling`] of `self` (a whole-bank draft) under
    /// `bound`, respecting any prior ceiling/host_cap.
    fn ceiling_of_under(
        mut self,
        definition: &ModelDefinition,
        bound: HeldBound,
        places: u64,
        mode: CeilingPaths<'_>,
        prior_ceiling: Option<u64>,
        prior_host_cap: Option<u64>,
    ) -> Result<Result<BankCeiling, (Self, NoBank)>, PlanError> {
        let (experts, floor) = (self.bank().slots, self.bank().floor());
        // A banked draft's re-planning never raises its ceiling.
        let top = prior_ceiling.unwrap_or(experts);
        // Its pooled records are granules of blocks the backend sizes within
        // its allocations; each lent allocation is one allocation.
        let (lent, most) = (self.bank().largest_lent().map_err(PlanError::ResourcePlanning)?, self.device.max_allocation_bytes);
        if lent > most {
            return Ok(Err((self, NoBank::Unstreamed(UnstreamedExperts::Allocation { lent, most }))));
        }
        // The load's own bound (a cap or a device room) below the floor's
        // weights holds no bank. The machine's room below them refuses
        // nothing here: the plan's settle finds it short.
        // The ceiling: the most slots whose weights the device holds under
        // `bound`, where the bank's curve ends, the floor at least: the bank
        // grows to it as memory frees, whatever the room at the load
        // (`settle_bank`).
        let device = bound.device();
        let (ceiling, ceiling_bytes, ceiling_paths) = match mode {
            CeilingPaths::Fixed => {
                // Paths are fixed at `self.bank().paths`, so borrow `self.load`
                // directly without cloning `ModelLoadPlan` or `ExpertBankPlan`,
                // and binary-search the monotone `resident_charge` curve.
                let bank = self.bank();
                let base = weight_bytes(&self.load)?
                    .checked_sub(bank.resident_charge(experts).map_err(PlanError::ResourcePlanning)?)
                    .ok_or(PlanError::Arithmetic("a bank's weights hold less than its record reservations' charge"))?;
                let bytes_at = |slots: u64| -> Result<u64, PlanError> {
                    let reserved = bank.resident_charge(slots).map_err(PlanError::ResourcePlanning)?;
                    base.checked_add(reserved)
                        .ok_or(PlanError::Arithmetic("a bank's weights overflow"))
                };
                let needed = bytes_at(floor)?;
                if let Some(limit) = bound.limit.filter(|limit| needed > limit.bytes()) {
                    return Ok(Err((self, NoBank::Bound { bound: limit, needed })));
                }
                let (ceiling, ceiling_bytes) = if floor >= top || needed > device {
                    (floor, needed)
                } else {
                    let hi_bytes = bytes_at(top)?;
                    if hi_bytes <= device {
                        (top, hi_bytes)
                    } else {
                        let (mut lo, mut lo_bytes, mut hi) = (floor, needed, top);
                        while lo + 1 < hi {
                            let mid = lo + (hi - lo) / 2;
                            let mid_bytes = bytes_at(mid)?;
                            if mid_bytes <= device {
                                lo = mid;
                                lo_bytes = mid_bytes;
                            } else {
                                hi = mid;
                            }
                        }
                        (lo, lo_bytes)
                    }
                };
                (ceiling, ceiling_bytes, bank.paths)
            }
            CeilingPaths::Priced(prices) => {
                // The weights of the bank at `slots` with its path records at those
                // slots, cached per distinct `BankPaths` (which varies monotonically
                // across only a handful of levels). Borrow `self.load` when `paths`
                // matches `self.bank().paths` to avoid cloning `ModelLoadPlan`.
                let mut cached: Option<(super::BankPaths, Option<super::ExpertBankPlan>, u64)> = None;
                let mut weights_at = |slots: u64| -> Result<(u64, super::BankPaths), PlanError> {
                    let paths = bank_paths(prices, self.bank(), slots)?;
                    if cached.as_ref().is_none_or(|(of, _, _)| *of != paths) {
                        if paths == self.bank().paths {
                            let base = weight_bytes(&self.load)?
                                .checked_sub(self.bank().resident_charge(experts).map_err(PlanError::ResourcePlanning)?)
                                .ok_or(PlanError::Arithmetic("a bank's weights hold less than its record reservations' charge"))?;
                            cached = Some((paths, None, base));
                        } else {
                            let load = (*self.load).clone().with_bank_paths(paths)?;
                            let base = weight_bytes(&load)?
                                .checked_sub(
                                    load.bank
                                        .as_ref()
                                        .expect("a streamed plan holds its bank")
                                        .resident_charge(experts)
                                        .map_err(PlanError::ResourcePlanning)?,
                                )
                                .ok_or(PlanError::Arithmetic("a bank's weights hold less than its record reservations' charge"))?;
                            let bank = load.bank.expect("a streamed plan holds its bank");
                            cached = Some((paths, Some(bank), base));
                        }
                    }
                    let (_, bank, base) = cached.as_ref().expect("just populated");
                    let bank = bank.as_ref().unwrap_or_else(|| self.bank());
                    let reserved = bank.resident_charge(slots).map_err(PlanError::ResourcePlanning)?;
                    let bytes = base
                        .checked_add(reserved)
                        .ok_or(PlanError::Arithmetic("a bank's weights overflow"))?;
                    Ok((bytes, paths))
                };
                let (needed, floor_paths) = weights_at(floor)?;
                if let Some(limit) = bound.limit.filter(|limit| needed > limit.bytes()) {
                    return Ok(Err((self, NoBank::Bound { bound: limit, needed })));
                }
                let mut ceiling = floor;
                let mut ceiling_bytes = needed;
                let mut ceiling_paths = floor_paths;
                while ceiling < top {
                    let (bytes, paths) = weights_at(ceiling + 1)?;
                    if bytes > device {
                        break;
                    }
                    ceiling += 1;
                    ceiling_bytes = bytes;
                    ceiling_paths = paths;
                }
                (ceiling, ceiling_bytes, ceiling_paths)
            }
        };
        if self.bank().paths != ceiling_paths {
            let load = (*self.load).clone().with_bank_paths(ceiling_paths)?;
            let programs = load.program_plan(definition, self.policy.codec)?;
            self.load = Box::new(load);
            self.programs = Box::new(programs);
        }
        let bank = self.bank();
        // A cap bounds every weight byte the load holds: a host tier holds
        // what it leaves beside the bank at its ceiling, its landing
        // reserve (the tier at no slots) aside, as staging is. A banked
        // draft's re-planning never raises the cap it was planned under.
        let capped = match (bound, bank.host_ceiling()) {
            (HeldBound { limit: Some(WeightBound::Cap(cap)), .. }, Some(_)) => {
                let left = cap - ceiling_bytes;
                let host = |slots: u64| bank.host_charge(slots).map_err(PlanError::ResourcePlanning);
                let (empty, widest) = (host(0)?, bank.widest().get());
                let cap = if host(widest)? - empty <= left {
                    widest
                } else {
                    let (mut lo, mut hi) = (0, widest);
                    while lo + 1 < hi {
                        let mid = lo + (hi - lo) / 2;
                        if host(mid)? - empty <= left {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    lo
                };
                Some(cap)
            }
            _ => None,
        };
        let host_cap = match (capped, prior_host_cap) {
            (Some(capped), Some(planned)) => Some(capped.min(planned)),
            (capped, planned) => capped.or(planned),
        };
        self.bank_mut().host_cap = host_cap;
        let places = usize::try_from(places)
            .ok()
            .and_then(NonZeroUsize::new)
            .map(PassLaunches::new)
            .ok_or(PlanError::Arithmetic("a group's places exceed usize"))?;
        Ok(Ok(BankCeiling { whole: self, ceiling, places }))
    }

    /// The draft's bank (of every expert, [`Self::bank_ceiling`]) at
    /// `slots` under `ceiling`, its passes of `pass` launches.
    fn resized(&self, slots: u64, ceiling: u64, pass: PassLaunches) -> Result<Self, PlanError> {
        let mut draft = self.with_load(self.load.clone().with_bank_slots(slots, ceiling)?);
        draft.bank_mut().pass = pass;
        Ok(draft)
    }

    /// Consume this draft and resize its bank in place, avoiding a `ModelLoadPlan` clone.
    fn into_resized(mut self, slots: u64, ceiling: u64, pass: PassLaunches) -> Result<Self, PlanError> {
        let load = (*self.load).with_bank_slots(slots, ceiling)?;
        self.load = Box::new(load);
        self.bank_mut().pass = pass;
        Ok(self)
    }

    /// The draft with its routed experts streamed through an expert bank
    /// (the same slots per layer in every group, at least the bank's floor
    /// and at most each layer's experts, or a banked draft's ceiling) beside
    /// its arena, at the pass its prices settle, with its charge and what
    /// `confirm` found of it; or why no bank's does.
    ///
    /// A pass's buffers are held from the load, beside the bank, so each
    /// pass size of the ladder up to the whole chain's places
    /// ([`super::pass_ladder`]) holds the bank its room leaves: a larger
    /// pass merges more launches, each prompt's routed reads fewer, and
    /// leaves fewer slots, each decode step missing more. Each pass size is
    /// priced by its mandatory part's charge: the smallest pass's structure,
    /// certified once, with the target family certified at that pass
    /// (`StructuralCharge::at_pass`), the one term a pass moves; the bank
    /// the most slots the room beside it holds. The plan holds the pass whose bank
    /// serves the workload prior in the least time a second: its decode
    /// steps a second times a step's seconds at the bank's slots, beside
    /// its requests a second times a prompt's routed seconds in passes of
    /// that size, its misses read at the most slots any pass leaves
    /// (`BankPrices`). Of passes priced alike, the larger, as fewer
    /// submissions of the same work. Only that pass is certified
    /// ([`Self::settle_bank_at`]), its mandatory part charged as priced.
    /// The plan, its certified family and its bound graphs all take the
    /// pass the plan holds (`ModelLoadPlan::pass`). Short at the smallest
    /// pass, the load is short.
    fn settle_bank<C: HeldCharge, T, E: From<PlanError>>(
        bank_ceiling: &BankCeiling,
        definition: &ModelDefinition,
        prices: &Priced<super::PlanPrices>,
        held: &mut HeldCharges<C>,
        budget: &super::MemoryBudget,
        confirm: &mut impl FnMut(&Self, &AssessmentMemoryCharge) -> Result<Result<T, NonZeroU64>, E>,
    ) -> Result<Result<(Self, C, T), NoBank>, E> {
        let BankCeiling { whole, ceiling, places } = bank_ceiling;
        let bank = whole.bank();
        let reserved = |slots| bank.resident_charge(slots).map_err(PlanError::ResourcePlanning);
        let points = rising(bank.floor(), *ceiling, &reserved)?;
        let mandatory = whole.resized(points[0], *ceiling, PassLaunches::FLOOR)?;
        let structure = crate::assessment::StructuralCharge::of(definition, &mandatory).map_err(charge_error)?;
        let floor_charge = structure.charge(definition, &mandatory).map_err(charge_error)?;
        if let Some(reason) = mandatory.untiered(&floor_charge, budget)? {
            return Ok(Err(NoBank::Unstreamed(reason)));
        }
        let prices = super::prices::measured(match prices {
            Priced::Measured(prices) => prices.of_bank(whole, definition)?,
            Priced::Pending(reason) => Priced::Pending(reason.clone()),
            Priced::Unavailable(reason) => Priced::Unavailable(reason.clone()),
        })?;
        let rates = prices.rates();
        let count = |slots: u64| usize::try_from(slots).map_err(|_| PlanError::Arithmetic("a bank's slots exceed usize"));
        let host = count(bank.host_ceiling().unwrap_or(0))?;
        let floor_bytes = floor_charge
            .store_demands(mandatory.device.topology(), mandatory.device.executor())
            .map_err(PlanError::ResourcePlanning)?
            .allocation_bytes();
        let floor_room = budget.room(&mandatory, &floor_charge)?;
        let fits_within = floor_bytes.saturating_add(floor_room);
        let mut most = None;
        let mut chosen: Option<(PassLaunches, u64, f64)> = None;
        for pass in super::pass_ladder(*places) {
            let mandatory = whole.resized(points[0], *ceiling, pass)?;
            let (charge, idle_class) = match pass == PassLaunches::FLOOR {
                true => (floor_charge.clone(), 0u64),
                false => {
                    let pass_structure = structure.at_pass(definition, &mandatory).map_err(charge_error)?;
                    let charge = pass_structure.charge(definition, &mandatory).map_err(charge_error)?;
                    let idle = pass_structure
                        .idle_class_bytes(pass.get(), mandatory.policy().limits(), mandatory.load().fronts())
                        .map_err(charge_error)?;
                    (charge, idle)
                }
            };
            let mandatory_bytes = charge
                .store_demands(mandatory.device.topology(), mandatory.device.executor())
                .map_err(PlanError::ResourcePlanning)?
                .allocation_bytes();
            // A larger pass leaves no more room.
            if mandatory_bytes > fits_within {
                break;
            }
            let free = budget.room(&mandatory, &charge)?;
            let slack = charge
                .bank_growth_slack(mandatory.device.topology(), mandatory.device.executor())
                .map_err(PlanError::ResourcePlanning)?;
            // Account for the upload-window slack released before bank growth
            // and the warm `IdleClass` (`CacheKind::GraphSlots`) bytes a
            // `pass`-place prompt group retains across decode so `settle_bank`
            // prices the true steady-state decode expert slots of each pass.
            let decode_free = free.saturating_add(slack).saturating_sub(idle_class);
            let slots = count(slots_within(decode_free, (points[0], *ceiling), &reserved)?)?;
            let held = magnitude_cost_model::HeldSlots { slots, host };
            // The smallest pass leaves the most slots.
            let most = *most.get_or_insert(held);
            let seconds = rates.decode_steps_per_s * prices.step_seconds(held)
                + rates.requests_per_s * prices.prompt_seconds(charge.held_pass.get() as u64, most);
            if chosen.as_ref().is_none_or(|(_, _, least)| seconds <= *least) {
                chosen = Some((pass, mandatory_bytes, seconds));
            }
        }
        let (pass, priced, _) = chosen.expect("invariant violated: the ladder holds the smallest pass");
        Self::settle_bank_at(bank_ceiling, pass, priced, definition, held, budget, confirm)
    }

    fn settle_bank_at<C: HeldCharge, T, E: From<PlanError>>(
        bank_ceiling: &BankCeiling,
        pass: PassLaunches,
        priced: u64,
        definition: &ModelDefinition,
        held: &mut HeldCharges<C>,
        budget: &super::MemoryBudget,
        confirm: &mut impl FnMut(&Self, &AssessmentMemoryCharge) -> Result<Result<T, NonZeroU64>, E>,
    ) -> Result<Result<(Self, C, T), NoBank>, E> {
        let BankCeiling { whole, ceiling, .. } = bank_ceiling;
        let ceiling = *ceiling;
        let bank = whole.bank();
        let reserved = |slots| bank.resident_charge(slots).map_err(PlanError::ResourcePlanning);
        let points = rising(bank.floor(), ceiling, &reserved)?;
        // The mandatory part: what the load holds whatever it releases.
        let mandatory = whole.resized(points[0], ceiling, pass)?;
        let floor_charge = held.charge(definition, &mandatory)?.charge().clone();
        // Host memory that admits not even the tier's floor streams nothing,
        // whatever the device holds.
        if let Some(reason) = mandatory.untiered(&floor_charge, budget)? {
            return Ok(Err(NoBank::Unstreamed(reason)));
        }
        let mandatory_bytes = floor_charge
            .store_demands(mandatory.device.topology(), mandatory.device.executor())
            .map_err(PlanError::ResourcePlanning)?
            .allocation_bytes();
        assert_eq!(mandatory_bytes, priced, "invariant violated: a pass of {pass:?} certifies to other bytes than it is priced at");
        let room = budget.room(&mandatory, &floor_charge)?;
        let short_error = |draft: &Self, charge: &AssessmentMemoryCharge, deficit_bytes: NonZeroU64| -> Result<NoBank, PlanError> {
            let fit = budget.fit_whole(draft, charge)?;
            Ok(NoBank::Short {
                deficit_bytes,
                required: fit.limiting_required,
                available: fit.limiting_available,
            })
        };
        // A draft of `slots` under `ceiling`, its charge, and what it leaves
        // of the room (`Ok`) or how far it exceeds it (`Err`).
        let charged = |held: &mut HeldCharges<C>, slots: u64, ceiling: u64| -> Result<_, E> {
            let draft = whole.resized(slots, ceiling, pass)?;
            let charge = held.charge(definition, &draft)?;
            let grown = charge
                .charge()
                .store_demands(draft.device.topology(), draft.device.executor())
                .map_err(PlanError::ResourcePlanning)?
                .allocation_bytes()
                .saturating_sub(mandatory_bytes);
            let left = match room.checked_sub(grown) {
                Some(left) => Ok(left),
                None => Err(grown - room),
            };
            Ok((draft, charge, left))
        };
        // The most slots whose reservation the room holds (including any slack
        // where Phase A or Phase B1 exceeds Phase B2 at the bank's floor), each
        // found short by its real charge giving up a step.
        let slack = floor_charge
            .bank_growth_slack(mandatory.device.topology(), mandatory.device.executor())
            .map_err(PlanError::ResourcePlanning)?;
        let most = slots_within(room.saturating_add(slack), (points[0], ceiling), &reserved)?;
        let mut at = points
            .iter()
            .position(|&slots| slots == most)
            .expect("invariant violated: the most slots a room holds end a charge level");
        loop {
            let slots = points[at];
            let (draft, charge) = match charged(held, slots, ceiling)? {
                (draft, charge, Ok(_)) => (draft, charge),
                (draft, charge, Err(excess)) if at == 0 => {
                    let deficit = NonZeroU64::new(excess).expect("a nonzero excess");
                    return Ok(Err(short_error(&draft, charge.charge(), deficit)?));
                }
                (_, _, Err(_)) => {
                    at -= 1;
                    continue;
                }
            };
            let deficit = match confirm(&draft, charge.charge())? {
                Ok(found) => return Ok(Ok((draft, charge, found))),
                Err(deficit) if at == 0 => {
                    return Ok(Err(short_error(&draft, charge.charge(), deficit)?));
                }
                Err(deficit) => deficit,
            };
            // The bank gives up at least the bytes the plan is short.
            let kept = reserved(slots)?.saturating_sub(deficit.get());
            let mut fewer = 0;
            for (index, &slots) in points[..at].iter().enumerate() {
                if reserved(slots)? <= kept {
                    fewer = index;
                }
            }
            at = fewer;
        }
    }

    /// The draft holding its experts whole ([`Self::held_whole`]), where
    /// the load's own bound holds its weights.
    fn held_whole_within(&self, definition: &ModelDefinition, bound: HeldBound) -> Result<Option<Self>, PlanError> {
        let whole = self.held_whole(definition)?;
        let bytes = weight_bytes(&whole.load)?;
        Ok(bound.limit.is_none_or(|limit| bytes <= limit.bytes()).then_some(whole))
    }

    /// The chains a plan holding its experts whole may run, longest first,
    /// where the load's own bound holds its weights (which no chain
    /// changes): a whole-context prompt's ([`Self::held_whole`]), then each
    /// half of the one before, down to one launch a group, as a load without
    /// routed experts runs. A shorter chain merges fewer chunks' passes and
    /// commits less: one launch-row region per launch it runs
    /// (`chain_rows`), and the graphs of the chains it seals. A placement
    /// holds the longest whose plan the room holds.
    fn whole_chains(&self, definition: &ModelDefinition, bound: HeldBound) -> Result<Option<Vec<u64>>, PlanError> {
        if let Some(limit) = bound.limit {
            let bytes = if self.load.bank.is_some() {
                weight_bytes(&self.load.clone().unbanked())?
            } else {
                weight_bytes(&self.load)?
            };
            if bytes > limit.bytes() {
                return Ok(None);
            }
        }
        let mut places = vec![self.whole_places(definition)?];
        while let Some(&last) = places.last().filter(|&&last| last > 1) {
            places.push(last.div_ceil(2));
        }
        Ok(Some(places))
    }

    /// The draft with its experts held whole: its prompt groups chain every
    /// chunk of a whole-context prompt, as a banked load's do (a chain is
    /// what merges their passes); one launch where the load has no routed
    /// experts, which have nothing to merge.
    pub(crate) fn held_whole(&self, definition: &ModelDefinition) -> Result<Self, PlanError> {
        self.held_whole_at(definition, self.whole_places(definition)?)
    }

    /// The most launches a group of a plan holding its experts whole
    /// chains ([`Self::held_whole`]).
    fn whole_places(&self, definition: &ModelDefinition) -> Result<u64, PlanError> {
        match super::bank::routed_experts(definition) {
            Ok(_) => Ok(self.prompt_places(definition)),
            Err(PlanError::Unstreamed(UnstreamedExperts::Unrouted)) => Ok(1),
            Err(error) => Err(error),
        }
    }

    /// The draft with its experts held whole, its prompt groups chaining up
    /// to `places` launches.
    fn held_whole_at(&self, definition: &ModelDefinition, places: u64) -> Result<Self, PlanError> {
        if self.load.bank.is_none() {
            return self.clone().with_chain(places);
        }
        let load = self.load.clone().unbanked();
        let programs = load.program_plan(definition, self.policy.codec)?;
        self.with_load_and_programs(load, programs).with_chain(places)
    }

    /// The draft holding its experts as `held` does, a plan of the same
    /// target that differs from this one only in its drafter (this one the
    /// target's plain plan, `held` the plan that drafts): through the same
    /// bank at the same slots, ceiling and host tier, or whole beside it. A
    /// bank is of the target's weights alone, so the bank built here is
    /// `held`'s, which is asserted. The one way to price a drafter: the two
    /// plans' charges at one holding of the experts.
    pub fn holding_as(&self, definition: &ModelDefinition, held: &Self) -> Result<Self, PlanError> {
        let draft = match held.load.bank.as_ref() {
            None => self.held_whole_at(definition, held.policy.limits.chain_capacity.get() as u64)?,
            Some(bank) => {
                let load = self.load.clone().banked_as(definition, bank, bank.window_slots())?;
                let programs = load.program_plan(definition, self.policy.codec)?;
                self.with_load_and_programs(load, programs).with_chain(bank.places)?
            }
        };
        assert_eq!(draft.load.bank, held.load.bank, "a plain plan's bank differs from its drafted plan's");
        assert_eq!(
            draft.policy.limits, held.policy.limits,
            "a plain plan's limits differ from its drafted plan's"
        );
        Ok(draft)
    }

    /// This drafted plan stripped of its drafter component, policy, and
    /// program slots, keeping its target weights, expert bank (if any), and
    /// limits intact: identical to `ExecutionPlanner::prepare` on the plain
    /// definition followed by [`Self::holding_as`], without re-planning every
    /// target weight from scratch.
    pub fn without_drafter(&self, plain_definition: &ModelDefinition) -> Result<Self, PlanError> {
        if plain_definition.draft.is_some() {
            return Err(PlanError::Topology(
                "without_drafter requires a plain model definition without a drafter",
            ));
        }
        let mut load = (*self.load).clone();
        load.head = None;
        load.target
            .retain(|weight| weight.component.kind != ArtifactComponentKind::Draft);
        load.unplaced_output.retain(|weight| {
            weight.component.kind != ArtifactComponentKind::Draft
                && !matches!(
                    weight.role.scope,
                    magnitude_family_contracts::WeightScope::HeadBlock(_)
                )
        });
        for weight in &mut load.target {
            if let super::Holding::Streamed(rows) = &mut weight.holding {
                if matches!(rows.fill, super::TableFill::Imported(_)) {
                    let slots = weight
                        .shape
                        .first()
                        .copied()
                        .and_then(|total| total.checked_sub(rows.landing_rows))
                        .ok_or(PlanError::Arithmetic("streamed table slots underflow"))?;
                    let plain_selected_rows =
                        magnitude_batching::row_class(self.policy.limits.max_selected_rows)
                            .ok_or_else(|| {
                                PlanError::InvalidDefinition(format!(
                                    "{} selected rows have no row class",
                                    self.policy.limits.max_selected_rows
                                ))
                            })? as u64;
                    let tile = weight.resident.tile_rows();
                    let landing_rows = plain_selected_rows.div_ceil(tile) * tile;
                    weight.shape[0] = slots + landing_rows;
                    weight.resident_bytes =
                        super::weights::representation_bytes(weight.resident, &weight.shape)
                            .map_err(PlanError::InvalidDefinition)?;
                    rows.landing_rows = landing_rows;
                    rows.fill = super::TableFill::Imported(super::TableFlights {
                        flights: self.policy.limits.flights(),
                        lanes: 1,
                        fronts: false,
                    });
                }
            }
        }
        load.readout_candidates.clear();
        load.rebuild_role_index();
        if self.policy.path == ExecutionPath::Native
            && crate::programs::graph::readout::reads_progressive_heads(self.device.backend)
                .map_err(PlanError::ResourcePlanning)?
        {
            load = load
                .with_readout_candidates(plain_definition)
                .map_err(PlanError::InvalidDefinition)?;
        }
        let programs = load.program_plan(plain_definition, self.policy.codec)?;
        Ok(Self {
            device: self.device.clone(),
            components: ComponentPlan {
                target: self.components.target,
                head: None,
                vision: self.components.vision,
            },
            load: Box::new(load),
            programs: Box::new(programs),
            capabilities: CapabilityPlan {
                max_draft_proposals: None,
                vision: self.capabilities.vision,
            },
            policy: ResolvedPolicy {
                path: self.policy.path,
                method: PlannedMethod::Plain,
                codec: self.policy.codec,
                selection: ComponentSelection {
                    head: false,
                    vision: self.policy.selection.vision,
                },
                limits: self.policy.limits,
            },
        })
    }

    /// A test draft whose target weights load through `substrate`'s expert
    /// bank at `slots` under `ceiling` (with `host_slots`, `host_cap`, and no
    /// CPU sites), keeping its resident `ProgramPlan` so CPU tests exercise
    /// banked residency (`import_whole`, empty and populated host-tier
    /// slices, and multi-slab home views) without needing a GPU.
    #[cfg(test)]
    pub(crate) fn test_bank_substrate(
        backend: seismic::BackendName,
        host: super::bank::HostTier,
    ) -> super::bank::BankSubstrate {
        super::bank::BankSubstrate {
            backend,
            allocations: seismic::NativeAllocations::EXACT,
            host,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_test_bank(
        &self,
        definition: &ModelDefinition,
        substrate: super::bank::BankSubstrate,
        slots: u64,
        ceiling: u64,
        host_slots: u64,
        host_cap: Option<u64>,
    ) -> Result<Self, PlanError> {
        let mut load = self
            .load
            .clone()
            .unbanked()
            .with_expert_bank(definition, ceiling, None, 1, self.policy.limits.flights(), substrate)?
            .with_bank_slots(slots, ceiling)?;
        let bank = load.bank.as_mut().expect("the load was just banked");
        bank.host_slots = host_slots;
        bank.host_cap = host_cap;
        let refused = (0..bank.layers().len())
            .map(|_| super::bank::LayerSite::Refused("test".into()))
            .collect();
        *bank = bank
            .clone()
            .with_sites(refused)
            .map_err(PlanError::ResourcePlanning)?;
        Ok(Self {
            load: Box::new(load),
            ..self.clone()
        })
    }

    /// The draft with prompt groups of up to `chain` launches.
    fn with_chain(mut self, chain: u64) -> Result<Self, PlanError> {
        self.policy.limits.chain_capacity = usize::try_from(chain)
            .ok()
            .and_then(NonZeroUsize::new)
            .ok_or(PlanError::Arithmetic("a prompt group's chain exceeds usize"))?;
        Ok(self)
    }

    /// The most launches a prompt group chains: every chunk of a
    /// whole-context prompt.
    fn prompt_places(&self, definition: &ModelDefinition) -> u64 {
        definition
            .decoder
            .context_limit
            .div_ceil(self.policy.limits.max_launch_rows as u64)
    }

    /// The bank of a draft that streams its experts.
    fn bank(&self) -> &super::ExpertBankPlan {
        self.load.bank.as_ref().expect("a streamed plan holds its bank")
    }

    /// The draft committing `slots` of its expert bank and `host_slots` of its
    /// host tier at the load, its ceiling kept ([`ModelLoadPlan::with_load_slots`]):
    /// what the target import's claim admits of the slots planned.
    pub fn with_load_slots(self, slots: u64, host_slots: u64) -> Result<Self, PlanError> {
        Ok(Self {
            load: Box::new(self.load.with_load_slots(slots, host_slots)?),
            ..self
        })
    }

    pub fn admit(self, resources: ResourcePlan) -> Result<ExecutionPlan, PlanError> {
        let planned_weight_bytes = resources
            .bytes()
            .target_weights
            .checked_add(resources.bytes().head_weights)
            .and_then(|bytes| bytes.checked_add(resources.bytes().vision_weights))
            .ok_or(PlanError::Arithmetic("resident weight byte count overflow"))?;
        if planned_weight_bytes == 0 {
            return Err(PlanError::Topology(
                "resource plan has no resident target weights",
            ));
        }
        Ok(ExecutionPlan {
            device: self.device,
            components: self.components,
            load: self.load,
            programs: self.programs,
            resources: Box::new(resources),
            capabilities: self.capabilities,
            policy: self.policy,
        })
    }
}

impl ExecutionPlan {
    /// The plan of a component that opens no expert bank
    /// ([`ModelLoadPlan::without_expert_bank`]).
    pub fn without_expert_bank(self) -> Self {
        Self { load: Box::new(self.load.without_expert_bank()), ..self }
    }

    /// The plan committing `slots` of its expert bank and `host_slots` of its
    /// host tier at the load, its ceiling kept ([`ModelLoadPlan::with_load_slots`]).
    pub fn with_load_slots(self, slots: u64, host_slots: u64) -> Result<Self, PlanError> {
        Ok(Self { load: Box::new(self.load.with_load_slots(slots, host_slots)?), ..self })
    }

    pub fn device(&self) -> &PlannedDevice {
        &self.device
    }

    pub fn components(&self) -> &ComponentPlan {
        &self.components
    }

    pub fn weights(&self) -> impl Iterator<Item = &WeightPlan> {
        self.load.weights()
    }

    pub fn load(&self) -> &ModelLoadPlan {
        &self.load
    }

    pub fn programs(&self) -> &ProgramPlan {
        &self.programs
    }

    pub fn resources(&self) -> &ResourcePlan {
        &self.resources
    }

    pub fn capabilities(&self) -> CapabilityPlan {
        self.capabilities
    }

    pub fn policy(&self) -> &ResolvedPolicy {
        &self.policy
    }

    /// Upper bound on uncompleted command buffers this load can keep live on
    /// the device's ordered command queue at once.
    pub fn command_buffer_capacity(&self) -> usize {
        command_buffer_capacity(&self.programs, &self.load, self.policy.limits())
    }
}

/// Exact upper bound on uncompleted command buffers a load can keep live on a
/// device's single ordered command queue at once.
///
/// Derived from the plan's own owners with zero chosen constants:
/// - Per-step target decode: `SubmissionChunks::count(blocks + 2)` base
///   submissions (entry + blocks + exit) + `host_tables` gate cuts + per
///   streamed layer `2` cuts (`1` host-gate `wait_event` cut + `1` AGX
///   indirect-dispatch cut) + `1` site-gate cut per CPU-sited layer.
/// - Pipelined target decode: `decode_step_cbs * limits.flights().groups()`.
/// - Streamed/chained prefill: `SubmissionChunks::count(places * (blocks + 2))`
///   base submissions + `places * host_tables` gate cuts + per streamed layer
///   `windows * (1 + 2 * window_rounds + 1 + has_host_tier) + 1` fill/cut CBs +
///   `(groups - 1) * decode_step_cbs` for any concurrent pipelined step.
/// - Concurrent non-target submissions from `limits.class_slots(places,
///   fronts)`: readout + drafter (if enabled) + vision (if enabled) + state +
///   `1` synchronous staging/service CB.
fn command_buffer_capacity(
    programs: &ProgramPlan,
    load: &ModelLoadPlan,
    limits: ResourceLimits,
) -> usize {
    use crate::programs::native_target::SubmissionChunks;

    let groups = limits.flights().groups();
    let blocks = programs.target().blocks().len();
    let host_tables = load.host_tables().len();
    let bank = load.expert_bank();
    let places = bank.map_or(limits.chain_capacity.get(), |bank| {
        usize::try_from(bank.places).unwrap_or(usize::MAX)
    });

    let runs_per_step = blocks.saturating_add(2);
    let decode_chunks = SubmissionChunks::count(runs_per_step);
    let streamed_decode_cuts = bank.map_or(0, |bank| {
        let layers = bank.layers().len();
        let sited = (0..layers).filter(|&layer| bank.sited(layer)).count();
        layers.saturating_mul(2).saturating_add(sited)
    });
    let decode_step_cbs = decode_chunks
        .saturating_add(host_tables)
        .saturating_add(streamed_decode_cuts);
    let decode_in_flight = decode_step_cbs.saturating_mul(groups);

    let chain_runs = places.saturating_mul(runs_per_step);
    let chain_chunks = SubmissionChunks::count(chain_runs);
    let streamed_chain_cuts = bank.map_or(0, |bank| {
        let layers = bank.layers().len();
        let windows = usize::try_from(bank.windows()).unwrap_or(usize::MAX);
        let rounds = usize::try_from(bank.window_rounds()).unwrap_or(usize::MAX);
        let tier_indirect = usize::from(bank.host_slots > 0);
        let per_window = 1usize
            .saturating_add(rounds.saturating_mul(2))
            .saturating_add(1)
            .saturating_add(tier_indirect);
        let per_layer = windows.saturating_mul(per_window).saturating_add(1);
        layers.saturating_mul(per_layer)
    });
    let chain_in_flight = chain_chunks
        .saturating_add(places.saturating_mul(host_tables))
        .saturating_add(streamed_chain_cuts)
        .saturating_add(decode_step_cbs.saturating_mul(groups.saturating_sub(1)));

    let target_cbs = decode_in_flight.max(chain_in_flight);

    let slots = limits.class_slots(places, load.fronts());
    let readout_cbs = slots.readout.activations;
    let drafter_cbs = if programs.head().is_some() || programs.draft().is_some() {
        slots.head.activations
    } else {
        0
    };
    let vision_cbs = usize::from(programs.vision().is_some());
    let state_cbs = slots.state.activations;
    let service_cbs = 1;

    target_cbs
        .saturating_add(readout_cbs)
        .saturating_add(drafter_cbs)
        .saturating_add(vision_cbs)
        .saturating_add(state_cbs)
        .saturating_add(service_cbs)
}

/// The charge of each way [`ExecutionPlanDraft::with_held_experts`] holds
/// a draft's experts, once charged: through a bank, at whatever slots, or
/// whole. A bank search's plans differ only in their slots, whose charges
/// follow from any one's ([`AssessmentMemoryCharge::at_bank_slots`]), and a
/// whole plan is the one its draft's experts held whole make at one chain,
/// so each way is charged from its structure
/// ([`crate::assessment::StructuralCharge`]) once in a search, as a `C` ([`HeldCharge`]).
#[derive(Clone, Debug)]
pub struct HeldCharges<C = AssessmentMemoryCharge> {
    /// A bank's, by the pass size it was certified at, whether it holds
    /// a resident form, and its scratch window slots (`W`).
    banked: Option<((PassLaunches, bool, u64), C)>,
    /// A whole plan's, by the launches its prompt groups chain: a chain
    /// sizes the target family's pass buffers, bound constants and graphs
    /// (`ExecutionPlanDraft::whole_chains`), so a whole plan of another
    /// chain is charged from its structure again.
    whole: Option<(NonZeroUsize, C)>,
}

impl<C> Default for HeldCharges<C> {
    fn default() -> Self {
        Self { banked: None, whole: None }
    }
}

/// What a search keeps of each way's charge ([`HeldCharges`]): the charge,
/// and what else its structure's one certification gives.
pub trait HeldCharge: Clone {
    /// The charge of `draft`, from its structure.
    fn of(definition: &ModelDefinition, draft: &ExecutionPlanDraft) -> Result<Self, PlanError>;

    fn charge(&self) -> &AssessmentMemoryCharge;

    /// This, of a draft of the same structure whose charge is `charge`.
    fn at(self, charge: AssessmentMemoryCharge) -> Self;
}

impl HeldCharge for AssessmentMemoryCharge {
    fn of(definition: &ModelDefinition, draft: &ExecutionPlanDraft) -> Result<Self, PlanError> {
        draft.memory_charge(definition)
    }

    fn charge(&self) -> &AssessmentMemoryCharge {
        self
    }

    fn at(self, charge: AssessmentMemoryCharge) -> Self {
        charge
    }
}

impl<C: HeldCharge> HeldCharges<C> {
    /// The charge of `draft`, of its way of holding its experts.
    fn charge(&mut self, definition: &ModelDefinition, draft: &ExecutionPlanDraft) -> Result<C, PlanError> {
        let Some(bank) = draft.load.bank.as_ref() else {
            let chain = draft.policy.limits.chain_capacity;
            return match &self.whole {
                Some((of, whole)) if *of == chain => Ok(whole.clone()),
                _ => Ok(self.whole.insert((chain, C::of(definition, draft)?)).1.clone()),
            };
        };
        let structure = (bank.pass, bank.resident, bank.window_slots());
        match &self.banked {
            Some((of, held)) if *of == structure => {
                let charge = held.charge().clone().at_bank_slots(&draft.load).map_err(PlanError::ResourcePlanning)?;
                Ok(held.clone().at(charge))
            }
            _ => Ok(self.banked.insert((structure, C::of(definition, draft)?)).1.clone()),
        }
    }
}

/// A charge's refusal as planning's: a kernel domain the graphs violate is
/// the plan's.
fn charge_error(error: crate::assessment::AssessmentError) -> PlanError {
    match error {
        crate::assessment::AssessmentError::Graph(crate::GraphError::KernelDomain(violation)) => {
            PlanError::KernelDomain(violation)
        }
        error => PlanError::ResourcePlanning(error.to_string()),
    }
}

/// A draft and the checked memory charge of it at the model's context
/// (`assessment::execution_memory_charge`): what planning confirmed of it,
/// which a load reads rather than charging it again.
#[derive(Clone, Debug)]
pub struct ChargedDraft<C = AssessmentMemoryCharge> {
    draft: ExecutionPlanDraft,
    charge: C,
}

impl ChargedDraft {
    /// `draft` with its charge (`assessment::execution_memory_charge`).
    pub fn of(definition: &ModelDefinition, draft: ExecutionPlanDraft) -> Result<Self, PlanError> {
        let charge = draft.memory_charge(definition)?;
        Ok(Self { draft, charge })
    }

}

impl<C: HeldCharge> ChargedDraft<C> {
    pub fn draft(&self) -> &ExecutionPlanDraft {
        &self.draft
    }

    pub fn charge(&self) -> &AssessmentMemoryCharge {
        self.charge.charge()
    }

    pub fn into_draft(self) -> ExecutionPlanDraft {
        self.draft
    }
}

/// Why a load's experts are held in no plan ([`ExecutionPlanDraft::with_held_experts`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NoBank {
    /// The plan's experts cannot stream.
    Unstreamed(UnstreamedExperts),
    /// The load's own `bound` is below the weights the plan `needed` at the
    /// bank's floor.
    Bound { bound: WeightBound, needed: u64 },
    /// The plan at the bank's floor is still `deficit_bytes` short (`required`
    /// against live `available` device room).
    Short {
        deficit_bytes: NonZeroU64,
        required: u64,
        available: u64,
    },
}

/// A planning refusal that the experts cannot stream as [`NoBank`]; any
/// other as the failure it is.
fn unstreamed<T, E: From<PlanError>>(error: PlanError) -> Result<Result<T, NoBank>, E> {
    match error {
        PlanError::Unstreamed(reason) => Ok(Err(NoBank::Unstreamed(reason))),
        error => Err(error.into()),
    }
}

/// A load's own bound on its weights, one kind at once
/// (`ModelPolicy::weight_bound`): a weight cap bounds every weight byte the
/// load holds, wherever it lies, its host tier's among them; a device room
/// bounds only what the device holds, so a discrete GPU's host tier is left
/// to host memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightBound {
    /// A cap on every weight byte the load holds, in bytes.
    Cap(u64),
    /// A bound on the weight bytes the device holds, in bytes.
    Room(u64),
}

impl WeightBound {
    /// The bound's bytes, whichever its kind.
    pub fn bytes(self) -> u64 {
        match self {
            Self::Cap(bytes) | Self::Room(bytes) => bytes,
        }
    }
}

impl std::fmt::Display for WeightBound {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cap(bytes) => write!(formatter, "a weight cap of {bytes} bytes"),
            Self::Room(bytes) => write!(formatter, "a device room of {bytes} bytes"),
        }
    }
}

/// What bounds the weights a plan holds its experts in: always the
/// device's room for them, which no plan's bank exceeds, and the load's own
/// bound where it has one ([`WeightBound`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeldBound {
    /// The device's room for the load's weights, in bytes.
    room: u64,
    /// The load's own bound (`ModelPolicy::weight_bound`).
    limit: Option<WeightBound>,
}

impl HeldBound {
    /// The bound of a load under its own `limit`, when it has one, on a
    /// device of `room` bytes for its weights.
    pub fn of(limit: Option<WeightBound>, room: u64) -> Self {
        Self { room, limit }
    }

    /// The bound of `room` bytes of the device for a load's weights alone.
    pub fn room(room: u64) -> Self {
        Self { room, limit: None }
    }

    /// The most weight bytes the device may hold: its room, or the load's
    /// own bound where it is less.
    fn device(self) -> u64 {
        self.limit.map_or(self.room, |limit| limit.bytes().min(self.room))
    }
}

/// The whole form a placement holds the experts in
/// ([`ExecutionPlanDraft::place`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WholeForm {
    /// A bank of every expert, which sheds slots under a memory press, as
    /// a streamed bank does while it serves.
    Bank,
    /// Each expert in its own allocation, where only that fits.
    Resident,
}

/// What a placement decided ([`ExecutionPlanDraft::place`]) and the figures
/// it decided on.
struct Placement<C, T> {
    figures: PlacementFigures,
    outcome: Placing<C, T>,
}

/// A placement's outcome: a whole plan, which holds no price; what a
/// streamed plan settles from; or why no plan holds the experts.
enum Placing<C, T> {
    Whole { draft: ExecutionPlanDraft, charge: C, found: T },
    Short { ceiling: BankCeiling, short: Option<(u64, u64)> },
    None(NoBank),
}

/// The figures a placement decided on: how the device's graphs are
/// priced, the form held, and, for each whole form tried (the bank at the
/// last pass tried, its smallest), its fit as the verdict that decided it
/// reads it and the terms of the charge that fit was of.
struct PlacementFigures {
    pricing: crate::platform::GraphPricing,
    form: Option<WholeForm>,
    bank: Option<Need>,
    resident: Option<Need>,
}

/// A whole form's fit, where `budget` judged it (none where `confirm` alone
/// did), and its charge's terms.
struct Need {
    fit: Option<super::WholeFit>,
    terms: NeedTerms,
}

/// The terms of a whole plan's charge (`AssessmentMemoryCharge`): its
/// weights and what of them lies in device storage, its state at the fit
/// depth, at startup and once loaded, its graph pools at its prompt groups'
/// chain places, its recorded
/// graphs in device and upload storage, its prepared resources, the
/// worker's device bytes, the startup and component import peaks, and the
/// allocation they compose to.
struct NeedTerms {
    weights: u64,
    weights_device: u64,
    state: u64,
    startup_state: u64,
    graph_pools: u64,
    chain_places: usize,
    recorded_graphs_device: u64,
    recorded_graphs_upload: u64,
    prepared: u64,
    process: u64,
    qualification_peak: u64,
    target_upload_peak: u64,
    component_upload: u64,
    allocation: u64,
}

impl NeedTerms {
    fn of(draft: &ExecutionPlanDraft, charge: &AssessmentMemoryCharge) -> Result<Self, PlanError> {
        let bounds = &charge.bounds;
        let allocation = charge
            .store_demands(draft.device.topology(), draft.device.executor())
            .map_err(PlanError::ResourcePlanning)?
            .allocation_bytes();
        let resource = |result: Result<u64, String>| result.map_err(PlanError::ResourcePlanning);
        Ok(Self {
            weights: charge.weight_demand.checked_bytes().ok_or(PlanError::Arithmetic("weight demand overflows"))?,
            weights_device: charge.weight_demand.device,
            state: charge.state_bytes,
            startup_state: charge.startup_state_bytes,
            graph_pools: bounds.graph_bytes,
            chain_places: draft.policy.limits.chain_capacity.get(),
            recorded_graphs_device: bounds.recorded_graph_bytes - bounds.recorded_graph_upload_bytes,
            recorded_graphs_upload: bounds.recorded_graph_upload_bytes,
            prepared: resource(bounds.prepared_resource_bytes())?,
            process: resource(bounds.device_process.bytes())?,
            qualification_peak: bounds.qualification_peak_bytes,
            target_upload_peak: bounds.target_upload_peak_bytes,
            component_upload: bounds.component_upload_bytes,
            allocation,
        })
    }
}

impl std::fmt::Display for NeedTerms {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            weights,
            weights_device,
            state,
            startup_state,
            graph_pools,
            chain_places,
            recorded_graphs_device,
            recorded_graphs_upload,
            prepared,
            process,
            qualification_peak,
            target_upload_peak,
            component_upload,
            allocation,
        } = self;
        write!(
            formatter,
            "weights {weights} ({weights_device} in device storage), state {state} (startup {startup_state}), \
             graph pools {graph_pools} at {chain_places} chain places, \
             recorded graphs {recorded_graphs_device} device and {recorded_graphs_upload} upload, prepared {prepared}, process {process}, \
             qualification {qualification_peak}, target import {target_upload_peak}, component import {component_upload}, allocation {allocation}"
        )
    }
}

impl PlacementFigures {
    /// One event of a load's placement.
    fn report(&self) {
        let fit = |need: &Option<Need>| need.as_ref().and_then(|need| need.fit);
        let available = fit(&self.bank).or(fit(&self.resident)).map(|fit| fit.available);
        tracing::info!(
            budget_available = ?available,
            whole_bank_bytes = ?fit(&self.bank).map(|fit| fit.needed),
            whole_bank_short = ?fit(&self.bank).and_then(|fit| fit.short).map(NonZeroU64::get),
            whole_bank_limiting = ?fit(&self.bank).and_then(|fit| fit.limiting),
            whole_bank_terms = %self.bank.as_ref().map_or("none".to_owned(), |need| need.terms.to_string()),
            resident_bytes = ?fit(&self.resident).map(|fit| fit.needed),
            resident_short = ?fit(&self.resident).and_then(|fit| fit.short).map(NonZeroU64::get),
            resident_limiting = ?fit(&self.resident).and_then(|fit| fit.limiting),
            resident_terms = %self.resident.as_ref().map_or("none".to_owned(), |need| need.terms.to_string()),
            graph_pricing = ?self.pricing,
            form = ?self.form,
            "the load's experts are placed"
        );
    }
}

/// A load's placement ([`ExecutionPlanDraft::placed`]): a whole plan, which
/// holds no price, or what streaming its experts settles from.
pub enum Placed<C = AssessmentMemoryCharge> {
    Whole(ChargedDraft<C>),
    Short(Unplaced<C>),
}

/// A load whose experts no whole plan holds: what its second stage
/// streams them from, on the machine's prices ([`Self::streamed`]).
pub struct Unplaced<C = AssessmentMemoryCharge> {
    draft: ExecutionPlanDraft,
    ceiling: BankCeiling,
    bound: Option<WeightBound>,
    held: HeldCharges<C>,
    /// The limiting store's `(required, available)` from whole placement when
    /// it did not fit, for `PlanForm::Whole` capacity errors.
    short: Option<(u64, u64)>,
}

impl<C: HeldCharge> Unplaced<C> {
    /// The load's plan before its bank.
    pub fn draft(&self) -> &ExecutionPlanDraft {
        &self.draft
    }

    /// Capacity error when a `PlanForm::Whole` load cannot fit the budget.
    pub fn whole_short_error(&self) -> PlanError {
        let (required, available) = self.short.unwrap_or((1, 0));
        PlanError::Resource(crate::CapacityError::new(
            crate::ResourceKind::DeviceMemory,
            required,
            available,
        ))
    }

    /// The load's second stage: its experts streamed through the bank and
    /// host tier `budget` holds beside its mandatory part, each plan
    /// confirmed by `budget`, at the pass the machine's `prices` settle. A
    /// load no plan holds is refused with why.
    pub fn streamed(
        self,
        definition: &ModelDefinition,
        prices: &Priced<super::PlanPrices>,
        budget: &super::MemoryBudget,
    ) -> Result<ChargedDraft<C>, PlanError> {
        let Self { draft, ceiling, bound, mut held, .. } = self;
        let capacity = budget.capacity();
        let streamed = draft.streamed_in(
            ceiling,
            HeldBound::of(bound, capacity),
            definition,
            prices,
            &mut held,
            budget,
            &mut load_confirm(budget),
        )?;
        match streamed {
            Ok((draft, charge, ())) => Ok(ChargedDraft { draft, charge }),
            Err(no) => Err(refusal(no)),
        }
    }
}

/// A load's confirm of each plan against `budget`: every plan, banked or
/// resident, must fit `budget`.
fn load_confirm<'a>(
    budget: &'a super::MemoryBudget,
) -> impl FnMut(&ExecutionPlanDraft, &AssessmentMemoryCharge) -> Result<Result<(), NonZeroU64>, PlanError> + 'a {
    move |draft: &ExecutionPlanDraft, charge: &AssessmentMemoryCharge| {
        Ok(budget.fit_whole(draft, charge)?.short.map_or(Ok(()), Err))
    }
}

/// A load's refusal where no plan holds its experts.
fn refusal(no: NoBank) -> PlanError {
    match no {
        NoBank::Unstreamed(reason) => PlanError::Unstreamed(reason),
        NoBank::Bound { bound, needed } => PlanError::WeightBound { bound, needed },
        NoBank::Short { required, available, .. } => PlanError::Resource(crate::CapacityError::new(
            crate::ResourceKind::DeviceMemory,
            required,
            available,
        )),
    }
}

/// The ways an assessment holds a draft's experts, each at its floor of
/// places (a pass of one launch, groups of one place): every place a load
/// adds beside the floor's is a group's class the load grants only when a
/// group opens, so a draft fits at some plan exactly when it fits at its
/// floor. Each form's structure is charged once
/// ([`crate::assessment::StructuralCharge`], an outline of every graph a
/// load prepares); every slot count, context and verdict is arithmetic over
/// it (`assessment::assess_execution_fit`). `T` is what else a form's
/// outline gives (a drafted plan's target alone).
///
/// Each form is boxed: a form carries a whole execution plan draft, and a
/// load passes its forms by value through deep frames, whose stacks hold
/// every by-value copy (`seismic_threads::STACK_BYTES`).
pub enum FloorForms<T = ()> {
    /// The experts streamed through a bank of every expert at the floor
    /// pass, at the most slots the load's own bound holds, and held whole
    /// as their own allocations at groups of one launch where that bound
    /// holds them so.
    Banked { bank: Box<BankForm<T>>, resident: Option<Box<HeldForm>> },
    /// Experts that cannot stream, held whole as their own allocations.
    Resident(Box<HeldForm<T>>),
}

/// One way of holding a draft's experts and its charge's structure.
pub struct HeldForm<T = ()> {
    pub draft: ExecutionPlanDraft,
    pub structure: crate::assessment::StructuralCharge,
    pub more: T,
}

/// A bank form ([`FloorForms::Banked`]): its draft at its ceiling, whose
/// charge at any slots is its structure's charge of [`Self::at_slots`],
/// beside smaller scratch-window rungs (`window_ladder`) materialized and
/// assessed lazily only when the full-layer scratch window's floor does not
/// fit.
pub struct BankForm<T = ()> {
    pub form: HeldForm<T>,
    /// The bank's experts: a bank of all of them holds them whole.
    pub experts: u64,
    /// The sizes the bank's charge steps through, its floor first and its
    /// ceiling last.
    pub points: Vec<u64>,
    /// Smaller `window_slots` rungs (`window_ladder`), descending down to
    /// `round_records`, each materialized lazily on demand.
    pub(crate) windowed: Vec<WindowedBankForm>,
}

/// Shared inputs for lazily materializing smaller `window_slots` rungs of a
/// [`BankForm`] only when the widest window's floor does not fit.
struct WindowedBankContext {
    bound: HeldBound,
    prior_ceiling: Option<u64>,
    prior_host_cap: Option<u64>,
    #[cfg(test)]
    whole: ExecutionPlanDraft,
    #[cfg(test)]
    definition: ModelDefinition,
}

/// Materialized rung of [`WindowedBankForm`].
pub(crate) struct WindowedBankRung {
    pub(crate) draft: ExecutionPlanDraft,
    pub(crate) points: Vec<u64>,
}

/// A smaller `window_slots` rung of [`BankForm`], materialized (`with_window`
/// + `rising`) and its target-family structure certified lazily
/// (`StructuralCharge::reusing_non_target`) only when needed.
pub(crate) struct WindowedBankForm {
    context: std::sync::Arc<WindowedBankContext>,
    window_slots: u64,
    rung: std::sync::OnceLock<Option<WindowedBankRung>>,
    structure: std::sync::OnceLock<crate::assessment::StructuralCharge>,
}

impl WindowedBankForm {
    pub(crate) fn materialize_from(
        &self,
        bank_draft: &ExecutionPlanDraft,
        experts: u64,
        definition: &ModelDefinition,
    ) -> Result<Option<&WindowedBankRung>, PlanError> {
        if let Some(rung) = self.rung.get() {
            return Ok(rung.as_ref());
        }
        let ctx = &*self.context;
        let materialized = match bank_draft.with_window_from_experts(
            definition,
            experts,
            ctx.bound,
            self.window_slots,
            ctx.prior_ceiling,
            ctx.prior_host_cap,
        )? {
            Ok(BankCeiling { whole: win_whole, ceiling: win_ceiling, .. }) => {
                let win_bank = win_whole.bank();
                let win_reserved = |slots| win_bank.resident_charge(slots).map_err(PlanError::ResourcePlanning);
                let win_points = rising(win_bank.floor(), win_ceiling, &win_reserved)?;
                let win_draft = win_whole.into_resized(win_ceiling, win_ceiling, PassLaunches::FLOOR)?;
                Some(WindowedBankRung {
                    draft: win_draft,
                    points: win_points,
                })
            }
            Err(_) => None,
        };
        let _ = self.rung.set(materialized);
        Ok(self.rung.get().expect("just initialized").as_ref())
    }

    pub(crate) fn structure(
        &self,
        bank_draft: &ExecutionPlanDraft,
        experts: u64,
        prepared_definition: &ModelDefinition,
        base: &crate::assessment::StructuralCharge,
    ) -> Result<Option<(&WindowedBankRung, &crate::assessment::StructuralCharge)>, crate::assessment::AssessmentError> {
        let Some(rung) = self.materialize_from(bank_draft, experts, prepared_definition)? else {
            return Ok(None);
        };
        if let Some(structure) = self.structure.get() {
            return Ok(Some((rung, structure)));
        }
        let mut certified = base.reusing_non_target(prepared_definition, &rung.draft)?;
        certified.release_target_cache();
        let _ = self.structure.set(certified);
        Ok(Some((rung, self.structure.get().expect("just initialized"))))
    }
}

#[cfg(test)]
impl std::ops::Deref for WindowedBankForm {
    type Target = WindowedBankRung;

    fn deref(&self) -> &Self::Target {
        let ctx = &*self.context;
        let experts = ctx.whole.bank().slots;
        self.materialize_from(&ctx.whole, experts, &ctx.definition)
            .expect("lazy windowed bank rung materializes")
            .expect("lazy windowed bank rung exists")
    }
}

impl<T> BankForm<T> {
    /// The form's draft with `slots` slots, a bank of at most that many.
    pub fn at_slots(&self, slots: u64) -> Result<ExecutionPlanDraft, PlanError> {
        self.form.draft.resized(slots, slots, PassLaunches::FLOOR)
    }

    /// The bank at `slots` with its linked host tier as host memory beside
    /// `charge` holds it (`ExecutionPlanDraft::with_host_tier`, sized by
    /// memory alone), or why its experts cannot stream beside it.
    pub fn with_host_tier(
        &self,
        slots: u64,
        charge: &AssessmentMemoryCharge,
        budget: &super::MemoryBudget,
    ) -> Result<Result<ExecutionPlanDraft, UnstreamedExperts>, PlanError> {
        self.at_slots(slots)?.with_host_tier(charge, budget)
    }
}

impl<T> FloorForms<T> {
    /// Drop certified target-family layout caches on all forms once
    /// preparation has finished.
    pub fn release_target_cache(&mut self) {
        match self {
            Self::Banked { bank, resident } => {
                bank.form.structure.release_target_cache();
                if let Some(resident) = resident {
                    resident.structure.release_target_cache();
                }
            }
            Self::Resident(resident) => {
                resident.structure.release_target_cache();
            }
        }
    }

    /// The forms without what else their outlines gave.
    pub fn structures(self) -> FloorForms {
        let form = |HeldForm { draft, structure, .. }: HeldForm<T>| HeldForm { draft, structure, more: () };
        match self {
            Self::Banked { bank, resident } => {
                let BankForm { form: held, experts, points, windowed } = *bank;
                FloorForms::Banked {
                    bank: Box::new(BankForm { form: form(held), experts, points, windowed }),
                    resident,
                }
            }
            Self::Resident(resident) => FloorForms::Resident(Box::new(form(*resident))),
        }
    }

    /// The form the plan's structure and demand are read from: its bank,
    /// else its experts whole.
    pub fn primary(&self) -> &HeldForm<T> {
        match self {
            Self::Banked { bank, .. } => &bank.form,
            Self::Resident(resident) => &**resident,
        }
    }

    /// Consume these forms, returning the primary form ([`Self::primary`]).
    pub fn into_primary(self) -> HeldForm<T> {
        match self {
            Self::Banked { bank, .. } => bank.form,
            Self::Resident(resident) => *resident,
        }
    }

    /// The bank form, where the experts stream.
    pub fn bank(&self) -> Option<&BankForm<T>> {
        match self {
            Self::Banked { bank, .. } => Some(&**bank),
            Self::Resident(_) => None,
        }
    }
}

impl FloorForms {
    /// The experts held whole as their own allocations, where the load's
    /// own bound holds them so.
    pub fn resident(&self) -> Option<&HeldForm> {
        match self {
            Self::Banked { resident, .. } => resident.as_deref(),
            Self::Resident(resident) => Some(&**resident),
        }
    }
}

impl ExecutionPlanDraft {
    /// The draft with its recorded graphs priced at `pricing`: the one
    /// term of its charge the machine's graph cost records move
    /// (`AssessmentMemoryBounds::recorded_graph_bytes`, a group's class).
    pub fn with_graph_pricing(mut self, pricing: crate::platform::GraphPricing) -> Self {
        self.device.graph_pricing = pricing;
        self
    }

    /// The draft's [`FloorForms`] under the load's own `limit`, its primary
    /// form's structure charged by `structure` (any secondary resident form
    /// reuses non-target family bounds from the bank form); or why no form
    /// holds its experts, as a load's placement refuses them.
    pub fn floor_forms<T: Sync, E: From<PlanError>>(
        &self,
        definition: &ModelDefinition,
        limit: Option<WeightBound>,
        structure: impl Fn(&Self) -> Result<(crate::assessment::StructuralCharge, T), E>,
    ) -> Result<FloorForms<T>, E> {
        let (forms, _) = self.floor_forms_inner(
            definition,
            limit,
            structure,
            None::<fn(&HeldForm<T>)>,
        )?;
        Ok(forms)
    }

    /// [`Self::floor_forms`], also running `after_primary` against the primary
    /// form as soon as it is ready: for a banked model with a secondary
    /// resident form, `after_primary` overlaps with certifying that resident
    /// form (`reusing_non_target`) on the assessment pool.
    pub fn floor_forms_with_primary<T: Sync, U: Send, E: From<PlanError>>(
        &self,
        definition: &ModelDefinition,
        limit: Option<WeightBound>,
        structure: impl Fn(&Self) -> Result<(crate::assessment::StructuralCharge, T), E>,
        after_primary: impl FnOnce(&HeldForm<T>) -> U + Send,
    ) -> Result<(FloorForms<T>, U), E> {
        let (forms, out) = self.floor_forms_inner(definition, limit, structure, Some(after_primary))?;
        Ok((forms, out.expect("Some(after_primary) always produces Some(U)")))
    }

    fn floor_forms_inner<T: Sync, U: Send, E: From<PlanError>>(
        &self,
        definition: &ModelDefinition,
        limit: Option<WeightBound>,
        structure: impl Fn(&Self) -> Result<(crate::assessment::StructuralCharge, T), E>,
        after_primary: Option<impl FnOnce(&HeldForm<T>) -> U + Send>,
    ) -> Result<(FloorForms<T>, Option<U>), E> {
        let bound = HeldBound::of(limit, super::MemoryBudget::of_stable(&self.device)?.capacity());
        let form = |draft: Self| -> Result<HeldForm<T>, E> {
            let (structure, more) = structure(&draft)?;
            Ok(HeldForm { draft, structure, more })
        };
        let bank = match stage_timer::time_stage(
            |t, ns| t.ff_bank_ceiling_ns += ns,
            || self.bank_ceiling(definition, bound, None),
        )? {
            Ok(BankCeiling { whole, ceiling, .. }) => {
                let bank = whole.bank();
                let reserved = |slots| bank.resident_charge(slots).map_err(PlanError::ResourcePlanning);
                let points = stage_timer::time_stage(
                    |t, ns| t.ff_rising_ns += ns,
                    || rising(bank.floor(), ceiling, &reserved),
                )?;
                let experts = bank.slots;
                let ladder = bank.window_ladder();
                let (draft, windowed) = if ladder.len() > 1 {
                    #[cfg(test)]
                    let (draft, context) = {
                        let draft = whole.resized(ceiling, ceiling, PassLaunches::FLOOR)?;
                        let context = std::sync::Arc::new(WindowedBankContext {
                            bound,
                            prior_ceiling: self.load.bank.as_ref().map(|bank| bank.ceiling),
                            prior_host_cap: self.load.bank.as_ref().and_then(|bank| bank.host_cap),
                            whole,
                            definition: definition.clone(),
                        });
                        (draft, context)
                    };
                    #[cfg(not(test))]
                    let (draft, context) = {
                        let draft = whole.into_resized(ceiling, ceiling, PassLaunches::FLOOR)?;
                        let context = std::sync::Arc::new(WindowedBankContext {
                            bound,
                            prior_ceiling: self.load.bank.as_ref().map(|bank| bank.ceiling),
                            prior_host_cap: self.load.bank.as_ref().and_then(|bank| bank.host_cap),
                        });
                        (draft, context)
                    };
                    let windowed = ladder[1..]
                        .iter()
                        .map(|&window_slots| WindowedBankForm {
                            context: std::sync::Arc::clone(&context),
                            window_slots,
                            rung: std::sync::OnceLock::new(),
                            structure: std::sync::OnceLock::new(),
                        })
                        .collect();
                    (draft, windowed)
                } else {
                    (whole.into_resized(ceiling, ceiling, PassLaunches::FLOOR)?, Vec::new())
                };
                Ok(BankForm { form: form(draft)?, experts, points, windowed })
            }
            Err(NoBank::Bound { bound, needed }) => return Err(PlanError::WeightBound { bound, needed }.into()),
            Err(NoBank::Unstreamed(reason)) => Err(reason),
            Err(no @ NoBank::Short { .. }) => unreachable!("a bank's ceiling asks no room, so it is never {no:?}"),
        };
        let has_resident = stage_timer::time_stage(
            |t, ns| t.ff_whole_chains_ns += ns,
            || self.whole_chains(definition, bound),
        )?
        .is_some();
        match (bank, has_resident) {
            (Ok(bank), true) => {
                let certify_resident = || -> Result<(Self, crate::assessment::StructuralCharge), E> {
                    let resident_draft = self.held_whole_at(definition, 1)?;
                    let resident_structure = bank
                        .form
                        .structure
                        .reusing_non_target(definition, &resident_draft)
                        .map_err(|error| E::from(charge_error(error)))?;
                    Ok((resident_draft, resident_structure))
                };
                let (primary_out, resident_res) = match after_primary {
                    Some(after_primary) => {
                        let bank_form = &bank.form;
                        let device_props = bank_form.draft.device().properties();
                        let (out, res) = crate::assessment::pool::join(
                            move || {
                                let _sizing =
                                    crate::programs::graph::sizing::SizingScope::enter_with_device(
                                        device_props,
                                    );
                                after_primary(bank_form)
                            },
                            certify_resident,
                        );
                        (Some(out), res)
                    }
                    None => (None, certify_resident()),
                };
                let (resident_draft, resident_structure) = resident_res?;
                Ok((
                    FloorForms::Banked {
                        bank: Box::new(bank),
                        resident: Some(Box::new(HeldForm {
                            draft: resident_draft,
                            structure: resident_structure,
                            more: (),
                        })),
                    },
                    primary_out,
                ))
            }
            (Ok(bank), false) => {
                let primary_out = after_primary.map(|f| f(&bank.form));
                Ok((FloorForms::Banked { bank: Box::new(bank), resident: None }, primary_out))
            }
            (Err(_), true) => {
                let resident = form(self.held_whole_at(definition, 1)?)?;
                let primary_out = after_primary.map(|f| f(&resident));
                Ok((FloorForms::Resident(Box::new(resident)), primary_out))
            }
            (Err(reason), false) => Err(PlanError::Unstreamed(reason).into()),
        }
    }
}

/// The path `bank` carries, priced on `prices` at `slots` held of each
/// layer (`BankPrices::paths`): its least move's stored bytes along the
/// stored read line.
fn bank_paths(prices: &magnitude_cost_model::BankPrices, bank: &super::ExpertBankPlan, slots: u64) -> Result<super::BankPaths, PlanError> {
    let bytes = bank.least_move_bytes().map_err(PlanError::ResourcePlanning)?;
    let slots = usize::try_from(slots).map_err(|_| PlanError::Arithmetic("a bank's slots exceed usize"))?;
    let saturation_bytes = seismic_io::disk::FULL_DEPTH * seismic_io::disk::FULL_READ;
    let records = prices
        .paths(bytes, magnitude_cost_model::HeldSlots::bank(slots), super::HINT_LEADS, saturation_bytes)
        .unwrap_or(magnitude_cost_model::PathRecords::NONE);
    Ok(super::BankPaths::Priced { line: prices.read(), records })
}

/// Componentwise covering envelope (`max` of `ahead` and `reserve`) over all
/// states in a detected `BankPaths` cycle (`cycle` plus `current`).
fn cycle_envelope(cycle: &[super::BankPaths], current: super::BankPaths) -> super::BankPaths {
    let mut ahead = current.records().ahead;
    let mut reserve = current.records().reserve;
    for path in cycle {
        let records = path.records();
        ahead = ahead.max(records.ahead);
        reserve = reserve.max(records.reserve);
    }
    match current {
        super::BankPaths::Priced { line, .. } => super::BankPaths::Priced {
            line,
            records: magnitude_cost_model::PathRecords { ahead, reserve },
        },
        super::BankPaths::Unpriced => super::BankPaths::Unpriced,
    }
}

/// Settle `paths` to a fixed point of `step`, or—when the discrete
/// anti-monotone mapping enters a cycle—to the covering envelope over the
/// cycle's states, settling once at that envelope.
fn settle_bank_paths<T, E>(
    initial: super::BankPaths,
    mut step: impl FnMut(super::BankPaths) -> Result<(T, Option<super::BankPaths>), E>,
) -> Result<(super::BankPaths, T, usize), E> {
    let mut paths = initial;
    let mut seen = Vec::new();
    loop {
        let (value, next) = step(paths)?;
        let Some(next) = next else {
            return Ok((paths, value, seen.len() + 1));
        };
        if next == paths {
            return Ok((paths, value, seen.len() + 1));
        }
        if let Some(cycle_start) = seen.iter().position(|&p| p == next) {
            let envelope = cycle_envelope(&seen[cycle_start..], paths);
            if envelope == paths {
                return Ok((paths, value, seen.len() + 1));
            }
            let (value, _) = step(envelope)?;
            return Ok((envelope, value, seen.len() + 2));
        }
        seen.push(paths);
        paths = next;
    }
}

/// How [`ExecutionPlanDraft::ceiling_of`] prices a bank's path records at
/// each slot count.
#[derive(Clone, Copy)]
enum CeilingPaths<'a> {
    /// Held fixed at the bank's own `paths` (an unpriced plan, or one
    /// iteration of [`ExecutionPlanDraft::with_paths`]).
    Fixed,
    /// Priced at each slot count from the machine's measured bank prices.
    Priced(&'a magnitude_cost_model::BankPrices),
}

/// What every plan of a bank is granted from ([`ExecutionPlanDraft::bank_ceiling`]).
struct BankCeiling {
    /// The draft with its bank of every expert.
    whole: ExecutionPlanDraft,
    /// The most slots its weights' bound holds.
    ceiling: u64,
    /// Its prompt groups' places: the most launches a pass covers.
    places: PassLaunches,
}

/// Geometric scratch-window ladder from `widest` (`P = 1`) down to `floor`
/// (`round_records`), halving `ceil(widest / P)` across `P = 2, 4, 8, ...`:
/// shared identically by runtime planning (`bank_ceiling`, `streamed_in`)
/// and assessment (`floor_forms`, `assess_execution_fit`).
fn window_ladder(widest: u64, floor: u64) -> Vec<u64> {
    let floor = floor.min(widest);
    let mut ladder = vec![widest];
    let mut p = 2u64;
    while *ladder.last().expect("ladder begins with widest") > floor {
        let next = widest.div_ceil(p).max(floor);
        if ladder.last() != Some(&next) {
            ladder.push(next);
        }
        p = p.saturating_mul(2);
    }
    ladder
}

impl super::ExpertBankPlan {
    /// The scratch-window ladder (`window_ladder`) from the bank's window
    /// down to its round's records: the rungs a load and its assessment
    /// walk below the bank, its own window first.
    pub fn window_ladder(&self) -> Vec<u64> {
        window_ladder(self.window_slots(), self.round_records())
    }
}

/// The sizes of `from..=to` a holding steps through: the most of each
/// `charge` level, `from`'s first and `to` last. A size whose charge is
/// its predecessor's costs nothing more, so it is held with it.
///
/// `charge` is monotonically non-decreasing on `from..=to`; each distinct
/// charge plateau is jumped in $O(\log(\text{to} - \text{cur}))$ queries.
fn rising(from: u64, to: u64, charge: &impl Fn(u64) -> Result<u64, PlanError>) -> Result<Vec<u64>, PlanError> {
    let mut points = Vec::new();
    let mut cur = from;
    let mut level = charge(cur)?;
    if from >= to {
        points.push(to);
        return Ok(points);
    }
    let top = charge(to)?;
    while level < top {
        let next_charge = charge(cur + 1)?;
        if next_charge > level {
            points.push(cur);
            cur += 1;
            level = next_charge;
            continue;
        }
        // Invariant: `charge(lo) <= level` and `charge(hi) == hi_charge > level`.
        let mut lo = cur + 1;
        let mut hi = to;
        let mut hi_charge = top;
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            let mid_charge = charge(mid)?;
            if mid_charge <= level {
                lo = mid;
            } else {
                hi = mid;
                hi_charge = mid_charge;
            }
        }
        points.push(lo);
        cur = hi;
        level = hi_charge;
    }
    points.push(to);
    Ok(points)
}

fn weight_bytes(load: &ModelLoadPlan) -> Result<u64, PlanError> {
    load.weight_bytes().map_err(PlanError::ResourcePlanning)
}

/// What a bank of `definition`'s experts streams on, of a device of
/// `backend` making its allocations as `allocations` and holding its host
/// tier `host`, or why they cannot stream: the
/// backend's entries ([`streams_experts`]) and a host tier, which every
/// miss lands in, so a bank streams only beside one.
pub(crate) fn bank_substrate(
    backend: BackendName,
    allocations: seismic::NativeAllocations,
    host: Option<HostTier>,
    definition: &ModelDefinition,
) -> Result<super::BankSubstrate, PlanError> {
    streams_experts(backend, definition)?;
    let Some(host) = host else {
        return Err(PlanError::Unstreamed(UnstreamedExperts::Untiered {
            backend,
            reason: "its backend places no host storage beside its device's own".to_owned(),
        }));
    };
    Ok(super::BankSubstrate { backend, allocations, host })
}

/// Refuse a backend without every entry of the bank of each routed form
/// the model has, or, beside a fused layer, without the shared-route decode
/// form its bank-served decode takes.
fn streams_experts(
    backend: BackendName,
    definition: &ModelDefinition,
) -> Result<(), PlanError> {
    use crate::native::{BankRouteForm, ExpertBankKernels};
    use crate::operators::routed::{fused_graph::DecodeForm, is_fused, routed_ffn};
    let refused = || PlanError::Unstreamed(crate::UnstreamedExperts::Backend(backend));
    let (mut has_fused, mut has_general) = (false, false);
    for (_, sublayer) in definition.decoder.sublayers() {
        if let Some(routed) = routed_ffn(&sublayer.op) {
            if is_fused(routed) {
                has_fused = true;
            } else {
                has_general = true;
            }
            if has_fused && has_general {
                break;
            }
        }
    }
    for (present, form) in [
        (has_fused, BankRouteForm::Fused),
        (has_general, BankRouteForm::General),
    ] {
        if !present {
            continue;
        }
        let declared = ExpertBankKernels::declared(backend, form)
            .map_err(|error| PlanError::ResourcePlanning(error.to_string()))?;
        let decodes = match form {
            BankRouteForm::Fused => {
                DecodeForm::of(backend).map_err(PlanError::ResourcePlanning)?
                    == DecodeForm::SharedRoute
            }
            BankRouteForm::General => true,
        };
        if !(declared && decodes) {
            return Err(refused());
        }
    }
    Ok(())
}

pub struct ExecutionPlanner;

impl ExecutionPlanner {
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        device: &SelectedDevice,
        manifest: &PackageManifest,
        definition: &ModelDefinition,
        selection: ComponentSelection,
        path: ExecutionPath,
        method: PlannedMethod,
        codec: KvCodec,
        limits: ResourceLimits,
    ) -> Result<ExecutionPlanDraft, PlanError> {
        let BackendPlan {
            backend,
            components,
            load,
            programs,
            capabilities,
            policy,
        } = Self::backend_plan(
            device.info.backend(),
            manifest,
            definition,
            selection,
            path,
            method,
            codec,
            limits,
        )?;
        Ok(ExecutionPlanDraft {
            device: Box::new(PlannedDevice {
                selector: device.info.selector(),
                backend,
                name: device.info.name().to_owned(),
                tensor_operations: device.info.forms_tensor_operations(),
                max_allocation_bytes: device.info.limits().max_allocation.bytes(),
                host: device.host_tier,
                executor: device.info.clone(),
                reading: device.reading.clone(),
                graph_pricing: device.graph_cost,
            }),
            components,
            load: Box::new(load),
            programs: Box::new(programs),
            capabilities,
            policy,
        })
    }

    /// The model's plan on `backend`, which is all the planner reads of a
    /// device: [`Self::prepare`] adds the selected device to it.
    #[allow(clippy::too_many_arguments)]
    pub fn backend_plan(
        backend: BackendName,
        manifest: &PackageManifest,
        definition: &ModelDefinition,
        selection: ComponentSelection,
        path: ExecutionPath,
        method: PlannedMethod,
        codec: KvCodec,
        limits: ResourceLimits,
    ) -> Result<BackendPlan, PlanError> {
        if path == ExecutionPath::Native && codec == KvCodec::RotatedK4V4 {
            return Err(PlanError::Unsupported("native rotated K4/V4 KV codec"));
        }
        if selection.head != method.drafts() {
            return Err(PlanError::Topology("method and head selection disagree"));
        }
        match method {
            PlannedMethod::Mtp { .. }
                if definition.head.is_none() || definition.draft.is_some() =>
            {
                return Err(PlanError::Topology(
                    "MTP drafts with the definition's head alone",
                ));
            }
            PlannedMethod::DFlash { .. }
                if definition.draft.is_none() || definition.head.is_some() =>
            {
                return Err(PlanError::Topology(
                    "DFlash drafts with the definition's separate draft alone",
                ));
            }
            _ => {}
        }
        let layout = super::resident_layout(path, backend);
        let load = stage_timer::time_stage(
            |t, ns| t.pr_load_derive_ns += ns,
            || ModelLoadPlan::derive(manifest, definition, selection, layout, method, limits),
        )
        .map_err(PlanError::InvalidDefinition)?;
        // Native entries import the weights, through record ports.
        let load = match path {
            ExecutionPath::Native => stage_timer::time_stage(
                |t, ns| t.pr_record_ports_ns += ns,
                || load.with_record_ports(backend),
            ),
            ExecutionPath::Planned => load,
        };
        // The head in progressive planes, where the backend reads them.
        let load = if path == ExecutionPath::Native
            && crate::programs::graph::readout::reads_progressive_heads(backend)
                .map_err(PlanError::ResourcePlanning)?
        {
            stage_timer::time_stage(
                |t, ns| t.pr_readout_candidates_ns += ns,
                || load.with_readout_candidates(definition),
            )
            .map_err(PlanError::InvalidDefinition)?
        } else {
            load
        };
        let programs = stage_timer::time_stage(
            |t, ns| t.pr_program_plan_ns += ns,
            || load.program_plan(definition, codec),
        )?;
        let target = ArtifactComponent {
            kind: ArtifactComponentKind::Target,
            identity: manifest.target.identity,
        };
        let vision = if selection.vision {
            manifest
                .projector
                .as_ref()
                .map(|projector| ArtifactComponent {
                    kind: ArtifactComponentKind::Projector,
                    identity: projector.identity,
                })
        } else {
            None
        };
        if selection.vision && vision.is_none() {
            return Err(PlanError::Topology(
                "enabled vision has no projector component",
            ));
        }
        // The head block chains on the device, so the draft width is bounded
        // by the sealed head graphs, not by the head's depth.
        let max_draft_proposals = match method {
            PlannedMethod::Plain => None,
            PlannedMethod::Mtp {
                greedy_proposals,
                sampled_proposals,
                ..
            } => {
                if greedy_proposals == 0
                    || sampled_proposals == 0
                    || greedy_proposals.max(sampled_proposals) > MAX_DRAFT_PROPOSALS
                    || method.widest() > greedy_proposals.max(sampled_proposals)
                {
                    return Err(PlanError::Unsupported("MTP proposal width"));
                }
                Some(greedy_proposals.max(sampled_proposals))
            }
            // One block pass drafts every proposal, so the width is bounded
            // by the draft's block and by one verification's rows.
            PlannedMethod::DFlash { proposals, .. } => {
                let most = definition.draft.as_ref().map_or(0, super::draft_proposals);
                if proposals == 0 || proposals > most || method.widest() > proposals {
                    return Err(PlanError::Unsupported("DFlash proposal width"));
                }
                Some(proposals)
            }
        };
        if programs.drafts_with_head() != matches!(method, PlannedMethod::Mtp { .. })
            || programs.draft().is_some() != matches!(method, PlannedMethod::DFlash { .. })
            || programs.vision().is_some() != selection.vision
        {
            return Err(PlanError::Topology(
                "enabled components and program slots disagree",
            ));
        }
        // The drafter's component: the target's embedded head, or the
        // separate draft.
        let head = match method {
            PlannedMethod::Plain => None,
            PlannedMethod::Mtp { .. } => Some(target),
            PlannedMethod::DFlash { .. } => Some(ArtifactComponent {
                kind: ArtifactComponentKind::Draft,
                identity: manifest
                    .draft
                    .as_ref()
                    .ok_or(PlanError::Topology("DFlash has no draft component"))?
                    .identity,
            }),
        };
        Ok(BackendPlan {
            backend,
            components: ComponentPlan {
                target,
                head,
                vision,
            },
            load,
            programs,
            capabilities: CapabilityPlan {
                max_draft_proposals,
                vision: selection.vision,
            },
            policy: ResolvedPolicy {
                path,
                method,
                codec,
                selection,
                limits,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No measured figures: a placement from memory alone, its bank's path
    /// unpriced.
    fn memory_alone() -> Priced<crate::planning::PlanPrices> {
        Priced::Unavailable("the test places from memory alone".into())
    }

    /// The plan's prices on the synthetic machine's measurements.
    fn synthetic() -> Priced<super::super::PlanPrices> {
        Priced::Measured(crate::assessment::synthetic::plan_prices())
    }

    /// The plan's prices before the machine's costs are measured.
    fn pending() -> Priced<super::super::PlanPrices> {
        Priced::Pending(crate::assessment::PendingPrices::standing(crate::assessment::PointRequest::Costs(crate::assessment::CostWalk::Required)))
    }

    /// A holding steps where its charge rises, each step holding the most
    /// sizes of its charge: a size that costs nothing more comes free.
    #[test]
    fn a_holding_steps_only_where_its_charge_rises() {
        let linear = |from: u64, to: u64, charge: &dyn Fn(u64) -> u64| {
            let mut points = Vec::new();
            let mut level = charge(from);
            for size in from + 1..=to {
                let next = charge(size);
                if next > level {
                    points.push(size - 1);
                    level = next;
                }
            }
            points.push(to);
            points
        };
        let check = |from: u64, to: u64, f: &dyn Fn(u64) -> u64| {
            let got = rising(from, to, &|s| Ok::<_, PlanError>(f(s))).unwrap();
            assert_eq!(got, linear(from, to, f), "mismatch for range {from}..={to}");
        };

        let charge = |slots: u64| Ok::<_, PlanError>(slots.div_ceil(4) * 100);
        assert_eq!(rising(2, 13, &charge).unwrap(), vec![4, 8, 12, 13]);
        assert_eq!(rising(5, 5, &charge).unwrap(), vec![5]);

        // Edge cases: from == to, constant, every-slot jump, plateau at start/end.
        check(0, 0, &|s| s * 10);
        check(7, 7, &|_| 42);
        check(1, 200, &|_| 100);
        check(1, 50, &|s| s * 10);
        check(0, 100, &|s| if s == 0 { 10 } else { 20 });
        check(0, 100, &|s| if s < 100 { 10 } else { 20 });
        check(1, 100, &|s| if s < 30 { 10 } else if s < 70 { 20 } else { 30 });
        for step in [1u64, 2, 3, 4, 7, 16, 64] {
            check(0, 257, &|s| s.div_ceil(step) * 100);
            check(3, 257, &|s| s.div_ceil(step) * 100);
        }
    }

    /// The recorded RTX 3090 with `host` bytes of host RAM free, CUDA's
    /// pooled graph memory as it measured it.
    fn rtx3090(host: u64) -> (seismic::DeviceTopology, seismic::Status, crate::platform::SelectedDevice) {
        recorded(seismic::fixtures::recorded::rtx3090(), BackendName::Cuda, host, RTX3090_CUDA_GRAPHS)
    }

    /// `topology`'s `backend` executor with `host` bytes of host RAM free,
    /// its graphs costing `cost`.
    fn recorded(
        topology: seismic::DeviceTopology,
        backend: BackendName,
        host: u64,
        cost: seismic::GraphCost,
    ) -> (seismic::DeviceTopology, seismic::Status, crate::platform::SelectedDevice) {
        use crate::platform::{GraphPricing, ReleaseRate};
        let info = topology.executors().iter().find(|executor| executor.backend() == backend).unwrap().clone();
        let status = seismic::Status {
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
        let device = crate::platform::selected(&topology, &status, &release, info, GraphPricing::Measured(cost));
        (topology, status, device)
    }

    /// [`recorded`] of the recorded RTX 3090 machine whose `backend` device
    /// grants `grant` bytes (`seismic::fixtures::rtx3090_granting`), the
    /// stable grant every plan and its assessment fit into
    /// (`FitMode::Stable`).
    fn rtx3090_granted(
        backend: BackendName,
        host: u64,
        cost: seismic::GraphCost,
        grant: u64,
    ) -> (seismic::DeviceTopology, seismic::Status, crate::platform::SelectedDevice) {
        recorded(seismic::fixtures::topology(&seismic::fixtures::rtx3090_granting(backend, grant)), backend, host, cost)
    }

    /// What the RTX 3090's recorded graphs hold on each backend: CUDA's
    /// pooled driver memory as the card measured it, and Vulkan's argument
    /// chunks as the recorded engine tests take them.
    const RTX3090_CUDA_GRAPHS: seismic::GraphCost = seismic::GraphCost::Pooled { graph: 92_663, launch: 8_102, granule: 2 << 20 };
    const RTX3090_VULKAN_GRAPHS: seismic::GraphCost = seismic::GraphCost::Chunked { launch: 0, chunk: 256 << 10 };

    /// The RTX 3090's graph cost on `backend`.
    fn rtx3090_graphs(backend: BackendName) -> seismic::GraphCost {
        match backend {
            BackendName::Cuda => RTX3090_CUDA_GRAPHS,
            BackendName::Vulkan => RTX3090_VULKAN_GRAPHS,
            other => panic!("the RTX 3090's graphs are measured on CUDA and Vulkan, not {other:?}"),
        }
    }

    /// Gemma 26B q4's routed shape on the recorded RTX 3090 (24 GB, CUDA,
    /// its graphs priced as the device measured them), uncapped, planned
    /// into the device's stable room: its bank holds every expert beside
    /// an arena whose pass merges at least 512 rows, the largest whose
    /// buffers fit beside the whole bank, the largest of all where the room
    /// holds that one's. The arena holds a group of one place; a group's
    /// class beyond it is claimed as the group opens, never charged at load
    /// (memory-failures items 7 and 27).
    #[test]
    fn gemma_q4_on_a_3090_keeps_its_bank_resident_beside_a_merged_pass() {
        use crate::planning::fixtures::{LIMITS, gemma_moe_fixture_definition, gemma_moe_fixture_manifest};
        use crate::platform::DomainRole;
        // Stable fit reads no headroom.
        let (topology, status, device) = rtx3090(0);
        let definition = gemma_moe_fixture_definition();
        let manifest = gemma_moe_fixture_manifest(&definition);
        let draft = ExecutionPlanner::prepare(
            &device,
            &manifest,
            &definition,
            ComponentSelection { head: false, vision: false },
            ExecutionPath::Native,
            PlannedMethod::Plain,
            KvCodec::Dense,
            LIMITS,
        )
        .unwrap();
        let executor = topology.resolve(draft.device().selector()).unwrap();
        let fit = crate::assessment::execution_memory_charge(&definition, &draft)
            .unwrap()
            .assess_fit(&topology, executor, &status)
            .unwrap();
        let allocation = fit.domains.iter().find(|domain| domain.role == DomainRole::Allocation).unwrap();
        let bound = allocation.capacity_bytes;
        let budget = crate::MemoryBudget::fixed(bound, None);
        let (planned, ()) = draft
            .with_held_experts(&definition, HeldBound::room(bound), &synthetic(), &budget, |_, _| Ok::<_, PlanError>(Ok(())))
            .unwrap()
            .unwrap_or_else(|no| panic!("Gemma 26B q4 streams on the RTX 3090: {no:?}"));
        let bank = planned.draft().load().expert_bank().expect("a routed load banks its experts");
        let allocated = planned.draft().allocation_bytes(&definition).unwrap();
        let limits = planned.draft().policy().limits();
        assert_eq!(
            (bank.slots, bank.ceiling),
            (128, 128),
            "the bank holds every expert: resident ({:?}, {} bytes a slot, {allocated} allocated of {bound})",
            bank.pass,
            bank.slot_bytes().unwrap(),
        );
        let region = *magnitude_batching::row_classes(limits.max_launch_rows).last().unwrap() as u64;
        assert!(bank.pass.get() as u64 * region >= 512, "{:?} of {region} rows", bank.pass);
        // The plan's charge at a pass size: its pass buffers, beside a
        // group of one place.
        let committed_at = |pass: PassLaunches| {
            let draft = planned.draft().resized(bank.slots, bank.ceiling, pass).unwrap();
            let charge = crate::assessment::execution_memory_charge(&definition, &draft).unwrap();
            assert!(charge.held_pass <= pass, "{:?} holds a pass above its budget {pass:?}", charge.held_pass);
            charge.store_demands(&topology, executor).unwrap().allocation_bytes()
        };
        assert!(committed_at(bank.pass) <= bound, "{:?} and its arena fit beside the whole bank", bank.pass);
        let ladder = super::super::pass_ladder(PassLaunches::new(NonZeroUsize::new(bank.places as usize).unwrap()));
        if let Some(&next) = ladder.iter().find(|&&pass| pass > bank.pass) {
            let with_class = committed_at(next);
            assert!(with_class > bound, "{next:?} and its arena ({with_class} bytes) fit the room ({bound}) beside the whole bank");
        }
        // With room for the largest pass and its arena beside the whole
        // bank, the largest pass is held.
        let largest = *ladder.last().unwrap();
        let wider = committed_at(largest).max(bound);
        let wider_budget = crate::MemoryBudget::fixed(wider, None);
        let (wide, ()) = draft
            .with_held_experts(&definition, HeldBound::room(wider), &synthetic(), &wider_budget, |_, _| Ok::<_, PlanError>(Ok(())))
            .unwrap()
            .unwrap();
        let widened = wide.draft().load().expert_bank().unwrap();
        assert_eq!((widened.slots, widened.ceiling), (128, 128), "the wider room keeps the whole bank");
        assert_eq!(widened.pass, largest, "{:?} of {} places", widened.pass, widened.places);
    }

    /// Qwen3.6 35B-A3B q4's routed shape (40 blocks of 256 experts choosing
    /// 8, hidden 2,048, intermediate 512: about 18.1 GB of Q4_0 expert
    /// stacks) on the recorded RTX 3090 (24 GB), planned as a load plans it
    /// (`with_bank`): the room holds every expert, so the bank is whole and
    /// the load never streams.
    #[test]
    fn qwen36_q4_on_an_unmeasured_3090_holds_every_expert() {
        use crate::planning::fixtures::{LIMITS, RoutedShape, gemma_moe_fixture_manifest, routed_fixture_definition};
        let (_, _, device) = rtx3090(64 << 30);
        let definition = routed_fixture_definition(RoutedShape { layers: 40, experts: 256, selected: 8, hidden: 2048, intermediate: 512 });
        let manifest = gemma_moe_fixture_manifest(&definition);
        let draft = ExecutionPlanner::prepare(
            &device,
            &manifest,
            &definition,
            ComponentSelection { head: false, vision: false },
            ExecutionPath::Native,
            PlannedMethod::Plain,
            KvCodec::Dense,
            LIMITS,
        )
        .unwrap();
        // The setup's precondition: the whole model's weights fit the
        // device with room to spare, as main's resident load of it does.
        let whole = draft.held_whole(&definition).unwrap();
        let bytes = weight_bytes(&whole.load).unwrap();
        let capacity = super::MemoryBudget::of_stable(draft.device()).unwrap().capacity();
        assert!(bytes < capacity - (2 << 30), "the fixture's {bytes} weight bytes fit the 3090's {capacity} with room");
        let planned = draft
            .with_bank::<AssessmentMemoryCharge>(&definition, None, &pending())
            .unwrap_or_else(|error| panic!("Qwen3.6 35B-A3B q4 is not planned whole on an unmeasured 3090: {error:?}"));
        let bank = planned.draft().load().expert_bank().expect("a routed load banks its experts");
        assert_eq!((bank.slots, bank.ceiling), (256, 256), "every expert is held");
        assert!(bank.whole());
        assert_eq!(bank.host_slots, 0, "a whole bank holds no host tier");
    }

    /// Qwen3.6 35B-A3B q4's routed shape at 4k on the RTX 3090's Vulkan
    /// executor, its device-local heap reported as the card's budget
    /// (`VK_EXT_memory_budget`, 25,336,446,976 bytes left, read on the
    /// card) and its graphs as the RTX 3090's are taken (`RTX3090_VULKAN_GRAPHS`), planned as the load
    /// plans it before the executor's costs are measured: the room holds its
    /// experts whole, so the load holds them whole, in a whole bank or none,
    /// never a partial bank that streams what the room holds.
    #[test]
    fn qwen36_q4_on_a_vulkan_3090_at_its_budget_holds_no_partial_bank() {
        use crate::planning::fixtures::{LIMITS, RoutedShape, gemma_moe_fixture_manifest, routed_fixture_definition};
        let (_, _, device) =
            rtx3090_granted(BackendName::Vulkan, 64 << 30, RTX3090_VULKAN_GRAPHS, 25_336_446_976);
        let mut definition = routed_fixture_definition(RoutedShape { layers: 40, experts: 256, selected: 8, hidden: 2048, intermediate: 512 });
        definition.decoder.context_limit = 4096;
        let manifest = gemma_moe_fixture_manifest(&definition);
        let draft = ExecutionPlanner::prepare(
            &device,
            &manifest,
            &definition,
            ComponentSelection { head: false, vision: false },
            ExecutionPath::Native,
            PlannedMethod::Plain,
            KvCodec::Dense,
            LIMITS,
        )
        .unwrap();
        // The setup's precondition: the experts held whole, charged as every
        // way of holding them is, fit the budget's stable room.
        let whole = draft.held_whole(&definition).unwrap();
        let mut held = HeldCharges::<AssessmentMemoryCharge>::default();
        let charge = held.charge(&definition, &whole).unwrap().charge().clone();
        let fit = super::MemoryBudget::of_stable(&whole.device).unwrap().fit(whole.device.topology(), whole.device.executor(), &charge).unwrap();
        assert!(
            matches!(fit.verdict, super::super::AssessmentFitVerdict::Fits),
            "the experts held whole fit the card's budget: {:?}",
            fit.verdict
        );
        // As the real load plans it: before the executor's costs are
        // measured.
        let planned = draft.clone().with_bank::<AssessmentMemoryCharge>(&definition, None, &pending()).unwrap();
        let held_as = |draft: &ExecutionPlanDraft| draft.load().expert_bank().map(|bank| (bank.slots, bank.ceiling));
        if let Some(held) = held_as(planned.draft()) {
            assert_eq!(held, (256, 256), "a bank of the experts the room holds whole is whole");
        }
        // Its assessment holds them as the load does: settled and confirmed
        // as `assess_execution_fit` settles and confirms it, from the draft
        // holding its experts as its largest plan does.
        let assessed = assessed_bank(&definition, &draft).map(|bank| (bank.slots, bank.ceiling));
        assert_eq!(assessed, held_as(planned.draft()), "the assessment holds the experts as the load does");
    }

    /// The stable room a plan fits into follows the grant the RTX 3090
    /// reports on each backend: a card planned at a grant is planned at it,
    /// not at the grant it recorded.
    #[test]
    fn the_stable_room_follows_the_grant_the_card_reports() {
        use crate::planning::fixtures::{LIMITS, RoutedShape, gemma_moe_fixture_manifest, routed_fixture_definition};
        let mut definition = routed_fixture_definition(RoutedShape { layers: 40, experts: 256, selected: 8, hidden: 2048, intermediate: 512 });
        definition.decoder.context_limit = 4096;
        let manifest = gemma_moe_fixture_manifest(&definition);
        for backend in [BackendName::Cuda, BackendName::Vulkan] {
            let granted = |grant| rtx3090_granted(backend, 64 << 30, rtx3090_graphs(backend), grant);
            let most = seismic::fixtures::rtx3090_most_granted(backend);
            let (_, _, device) = granted(most);
            let draft = ExecutionPlanner::prepare(
                &device,
                &manifest,
                &definition,
                ComponentSelection { head: false, vision: false },
                ExecutionPath::Native,
                PlannedMethod::Plain,
                KvCodec::Dense,
                LIMITS,
            )
            .unwrap();
            let whole = draft.held_whole(&definition).unwrap();
            let charge = HeldCharges::<AssessmentMemoryCharge>::default().charge(&definition, &whole).unwrap().charge().clone();
            let room = |grant| {
                let (topology, status, device) = granted(grant);
                super::MemoryBudget::stable(&topology, &device.info, &status)
                    .unwrap()
                    .fit_whole(&whole, &charge)
                    .unwrap()
                    .available
            };
            let (less, more) = (room(most - (1 << 30)), room(most));
            assert!(less < more && more <= most, "{backend:?}: the room follows the card's grant: {less} at 1 GiB less, {more} at {most}");
        }
    }

    /// Qwen3.6 35B-A3B q4, from its catalog header, on a CUDA RTX 3090 at
    /// the room its cold load reported (`the load's experts are placed`:
    /// 24,910,168,064 bytes, its graphs at the cost that load measured), at
    /// the two workloads it is loaded for.
    ///
    /// - The KL harness decodes its eight fixtures of 4,096 tokens together
    ///   (`--batch 8`), lookahead off: eight conversations at a fit depth of
    ///   4,097, each its history and its recurrent banks (an accepted bank,
    ///   three checkpoints the prefix cache retains, and two successor
    ///   banks), with the seed beside them. Lookahead off still takes two
    ///   successors: a prompt's chunks chained across the load's places
    ///   alternate two banks (`OwnedChainedAdvance`), the second written by
    ///   the chain's second chunk. Those 54 state granules pack into 27
    ///   two-granule CUDA pool blocks (~3.57 GB), which no whole plan holds
    ///   beside the weights, so the experts stream.
    /// - The service loads one conversation (`standard_service_limits`),
    ///   lookahead on (`ModelPolicy::default`), here at 4,096, and holds the
    ///   experts whole.
    #[test]
    fn qwen36_q4_on_a_cuda_3090_streams_the_kl_harness_load_and_holds_the_served_load_whole() {
        use magnitude_family_contracts::ModelFamily;
        const ROOM: u64 = 24_910_168_064;
        const GRAPHS: seismic::GraphCost = seismic::GraphCost::Pooled { graph: 98_687, launch: 10_144, granule: 2 << 20 };
        let identity = magnitude_artifacts::PackageIdentity { target: magnitude_artifacts::ArtifactIdentity([7; 32]), projector: None };
        let target = magnitude_family_common::headers::target("qwen3.6-35b-a3b:gguf:q4");
        let declared = magnitude_family_qwen35::Qwen35Family.inspect(&target, None, identity).unwrap();
        let manifest = magnitude_artifacts::PackageManifest {
            identity,
            target: magnitude_artifacts::ComponentManifest {
                files: vec![magnitude_artifacts::ComponentFile {
                    path: "qwen3.6-35b-a3b-q4.gguf".into(),
                    size: target.tensors.iter().map(|tensor| tensor.nbytes).sum(),
                }],
                identity: identity.target,
                tensors: target.tensors.clone(),
            },
            projector: None,
            draft: None,
        };
        // The limits the engine plans a load of 512 prefill rows with
        // (`resource_limits`): `decode` rows a decode step, `exported` rows
        // of logits.
        let limits = |conversations: usize, decode: usize, exported: usize, lookahead: bool| crate::ResourceLimits {
            max_launch_rows: 512,
            max_launch_slots: 512,
            max_selected_rows: decode,
            max_drafting_slots: decode,
            exported_logits_rows: exported,
            max_images_per_request: magnitude_artifacts::MAX_IMAGES_PER_REQUEST,
            max_image_cells: 0,
            lookahead,
            chain_capacity: NonZeroUsize::MIN,
            conversations: NonZeroUsize::new(conversations).unwrap(),
        };
        let (kl, served) = (limits(8, 8, 512, false), limits(1, 32, 0, true));
        let prepare = |device: &crate::platform::SelectedDevice, context: u64, limits| {
            let mut definition = declared.clone();
            definition.decoder.context_limit = context;
            let draft = ExecutionPlanner::prepare(
                device,
                &manifest,
                &definition,
                ComponentSelection { head: false, vision: false },
                ExecutionPath::Native,
                PlannedMethod::Plain,
                KvCodec::AffineK8V4,
                limits,
            )
            .unwrap();
            (definition, draft)
        };
        let granted = |grant| rtx3090_granted(BackendName::Cuda, 64 << 30, GRAPHS, grant);
        let most = seismic::fixtures::rtx3090_most_granted(BackendName::Cuda);
        let (_, _, card) = granted(most);
        let (definition, draft) = prepare(&card, 4097, kl);
        let whole = draft.held_whole(&definition).unwrap();
        let charge = HeldCharges::<AssessmentMemoryCharge>::default().charge(&definition, &whole).unwrap().charge().clone();
        let state = crate::ResourcePlanner::state_plan(
            &definition,
            whole.load(),
            whole.policy().method(),
            whole.policy().codec(),
            whole.policy().limits(),
            crate::ResourceCapacity {
                domain_bytes: super::MemoryBudget::of_stable(whole.device()).unwrap().capacity(),
                tensor_operations: crate::TensorOperations::of(whole.device().tensor_operations()),
                allocations: whole.device().allocations(),
            },
        )
        .unwrap();
        let target_state = state.target_state();
        assert_eq!(target_state.block.granules(target_state.granule), 2, "two 62.6 MiB state granules pack into 63 × 2 MiB on CUDA");
        assert_eq!(
            (charge.state_bytes, charge.startup_state_bytes),
            (
                target_state.slab_charge(54 * target_state.granule).unwrap() + target_state.table_bytes().unwrap() + target_state.block.absent,
                target_state.slab_charge(4 * target_state.granule).unwrap() + target_state.table_bytes().unwrap() + target_state.block.absent,
            ),
            "the KL load's state at the harness's limits"
        );
        // The card at the least grant whose stable room is the cold load's.
        let room = |grant| {
            let (topology, status, device) = granted(grant);
            super::MemoryBudget::stable(&topology, &device.info, &status)
                .unwrap()
                .fit_whole(&whole, &charge)
                .unwrap()
                .available
        };
        let (mut short, mut held) = (0u64, most);
        assert!(room(held) >= ROOM, "the card's room at its most grant holds the cold load's");
        while held - short > 1 {
            let mid = short + (held - short) / 2;
            match room(mid) >= ROOM {
                true => held = mid,
                false => short = mid,
            }
        }
        let (_, _, card) = granted(held);
        let (definition, draft) = prepare(&card, 4097, kl);
        let planned = draft.device().clone();
        assert!(
            matches!(draft.placed::<AssessmentMemoryCharge>(&definition, None, &memory_alone(), &super::MemoryBudget::of_stable(&planned).unwrap()).unwrap(), Placed::Short(_)),
            "the KL harness's eight conversations stream the experts"
        );
        let (definition, draft) = prepare(&card, 4096, served);
        let planned = draft.device().clone();
        match draft.placed::<AssessmentMemoryCharge>(&definition, None, &memory_alone(), &super::MemoryBudget::of_stable(&planned).unwrap()).unwrap() {
            Placed::Whole(placed) => {
                assert!(placed.draft.load().expert_bank().is_none_or(|bank| bank.whole()), "the served load holds every expert");
                assert_eq!(placed.draft.policy.limits.chain_capacity.get(), 8, "the served load chains every chunk of its 4,096-token prompt");
            }
            Placed::Short(_) => panic!("the served load's one conversation holds the experts whole"),
        }
    }

    /// The bank the draft's assessment holds its experts in where it fits
    /// (`assess_execution_fit` over its floor forms, by memory alone): none
    /// where it holds them whole as their own allocations.
    fn assessed_bank(definition: &ModelDefinition, draft: &ExecutionPlanDraft) -> Option<crate::ExpertBankPlan> {
        let forms = draft
            .floor_forms(definition, None, |draft| {
                crate::assessment::StructuralCharge::of(definition, draft).map(|charge| (charge, ())).map_err(charge_error)
            })
            .unwrap();
        let prepared = crate::assessment::prepare_execution_assessment(definition.clone(), forms).unwrap();
        let FitReading { topology, status, .. } = &draft.device.reading;
        match crate::assessment::assess_execution_fit(definition, &prepared, draft.device.graph_pricing, topology, status).unwrap() {
            crate::assessment::ExecutionAssessment::Fits(fit) => fit.bank,
            crate::assessment::ExecutionAssessment::DoesNotFit(shortfall) => panic!("the assessment finds no plan: {shortfall:?}"),
        }
    }

    /// Qwen3.6 35B-A3B q4's routed shape at 4k on the recorded RTX 3090, on
    /// CUDA and on Vulkan, its graphs as the RTX 3090's are taken (`rtx3090_graphs`), its device
    /// reporting memory between the least grant
    /// at which admission's fit holds a bank of every expert and the least at
    /// which every slot also fits the settle's room beside the bank's
    /// mandatory part (the bank at its floor), both derived here and
    /// asserted of the draft planned at that grant: the load and its
    /// assessment hold the bank whole, as admission fits it.
    #[test]
    fn a_bank_admission_holds_whole_is_held_whole_before_any_price() {
        use crate::planning::fixtures::{LIMITS, RoutedShape, gemma_moe_fixture_manifest, routed_fixture_definition};
        use super::super::AssessmentFitVerdict::Fits;
        let mut definition = routed_fixture_definition(RoutedShape { layers: 40, experts: 256, selected: 8, hidden: 2048, intermediate: 512 });
        definition.decoder.context_limit = 4096;
        let manifest = gemma_moe_fixture_manifest(&definition);
        for backend in [BackendName::Cuda, BackendName::Vulkan] {
            let granted = |grant: Option<u64>| match grant {
                Some(grant) => rtx3090_granted(backend, 64 << 30, rtx3090_graphs(backend), grant),
                None => recorded(seismic::fixtures::recorded::rtx3090(), backend, 64 << 30, rtx3090_graphs(backend)),
            };
            let prepare = |device: &crate::platform::SelectedDevice| {
                ExecutionPlanner::prepare(
                    device,
                    &manifest,
                    &definition,
                    ComponentSelection { head: false, vision: false },
                    ExecutionPath::Native,
                    PlannedMethod::Plain,
                    KvCodec::Dense,
                    LIMITS,
                )
                .unwrap()
            };
            // A bank of every expert at its smallest pass with its charge,
            // its mandatory part's charge, and what its slots hold beyond
            // the floor's.
            let whole_bank = |draft: &ExecutionPlanDraft| {
                let mut held = HeldCharges::<AssessmentMemoryCharge>::default();
                let bound = HeldBound::of(None, super::MemoryBudget::of_stable(&draft.device).unwrap().capacity());
                let BankCeiling { whole, ceiling, .. } = draft.bank_ceiling(&definition, bound, None).unwrap().unwrap();
                let experts = whole.bank().slots;
                assert_eq!(ceiling, experts, "{backend:?}: the bound holds a bank of every expert");
                let reserved = |slots| whole.bank().resident_charge(slots).map_err(PlanError::ResourcePlanning);
                let floor = rising(whole.bank().floor(), experts, &reserved).unwrap()[0];
                let bank = whole.resized(experts, experts, PassLaunches::FLOOR).unwrap();
                let mandatory = whole.resized(floor, experts, PassLaunches::FLOOR).unwrap();
                let charge = held.charge(&definition, &bank).unwrap();
                let floor_charge = held.charge(&definition, &mandatory).unwrap();
                (bank, charge, floor_charge, reserved(experts).unwrap() - reserved(floor).unwrap())
            };
            type WholeBank = (ExecutionPlanDraft, AssessmentMemoryCharge, AssessmentMemoryCharge, u64);
            // Whether the card granted `grant` holds the bank as admission
            // fits it, and as 280dd4f52's check did: every slot within the
            // settle's room beside the mandatory part, and the bank's charge
            // there, and admission's fit.
            let admits = |grant: u64, (_, charge, _, _): &WholeBank| {
                let (topology, status, device) = granted(Some(grant));
                let budget = super::MemoryBudget::stable(&topology, &device.info, &status).unwrap();
                matches!(budget.fit(&topology, &device.info, charge).unwrap().verdict, Fits)
            };
            let beside_mandatory = |grant: u64, (bank, charge, floor_charge, grown): &WholeBank| {
                let (topology, status, device) = granted(Some(grant));
                let budget = super::MemoryBudget::stable(&topology, &device.info, &status).unwrap();
                let mandatory = floor_charge.store_demands(&topology, &device.info).unwrap().allocation_bytes();
                let left = budget.room(bank, floor_charge).unwrap();
                let needed = charge.store_demands(&topology, &device.info).unwrap().allocation_bytes();
                *grown <= left && needed.saturating_sub(mandatory) <= left && matches!(budget.fit(&topology, &device.info, charge).unwrap().verdict, Fits)
            };
            // The least grant that holds the bank so.
            let least = |holds: &dyn Fn(u64) -> bool| {
                let (mut short, mut held) = (0u64, seismic::fixtures::rtx3090_most_granted(backend));
                assert!(holds(held), "{backend:?}: the card holds the bank at the most it grants");
                while held - short > 1 {
                    let mid = short + (held - short) / 2;
                    match holds(mid) {
                        true => held = mid,
                        false => short = mid,
                    }
                }
                held
            };
            let bank = whole_bank(&prepare(&granted(None).2));
            let (admitted, beside) = (least(&|grant| admits(grant, &bank)), least(&|grant| beside_mandatory(grant, &bank)));
            assert_eq!(
                admitted, beside,
                "{backend:?}: admission ({admitted}) and StableRoom::room beside mandatory ({beside}) must agree"
            );
            // Re-preparing at a tighter grant can shift state slab rounding slightly;
            // verify at a grant just above `admitted` (still well below full card grant)
            // that both unpriced `with_bank` and `assessed_bank` hold the bank whole.
            let grant = admitted.saturating_add(64 << 20).min(seismic::fixtures::rtx3090_most_granted(backend));
            let draft = prepare(&granted(Some(grant)).2);
            let bank = whole_bank(&draft);
            assert!(admits(grant, &bank), "{backend:?}: admission holds the bank at {grant} bytes");
            assert!(beside_mandatory(grant, &bank), "{backend:?}: the room beside its mandatory part also holds it at {grant} bytes");
            let whole = |planned: &ExecutionPlanDraft| planned.load().expert_bank().map(|bank| (bank.slots, bank.ceiling, bank.host_slots));
            let planned = draft
                .clone()
                .with_bank::<AssessmentMemoryCharge>(&definition, None, &pending())
                .unwrap_or_else(|error| panic!("{backend:?}: the load is held before any price is measured: {error:?}"));
            assert_eq!(whole(planned.draft()), Some((256, 256, 0)), "{backend:?}: the load holds the bank whole");
            let assessed = assessed_bank(&definition, &draft).map(|bank| (bank.slots, bank.ceiling, bank.host_slots));
            assert_eq!(assessed, Some((256, 256, 0)), "{backend:?}: its assessment holds the bank whole");
        }
    }

    /// Gemma 26B q4 on the recorded RTX 3090 with 64 GB of host RAM free,
    /// capped at its weights less half its experts, and the cap.
    fn capped_gemma() -> (ExecutionPlanDraft, u64) {
        use crate::planning::fixtures::{LIMITS, gemma_moe_fixture_definition, gemma_moe_fixture_manifest};
        let (_, _, device) = rtx3090(64 << 30);
        let definition = gemma_moe_fixture_definition();
        let manifest = gemma_moe_fixture_manifest(&definition);
        let draft = ExecutionPlanner::prepare(
            &device,
            &manifest,
            &definition,
            ComponentSelection { head: false, vision: false },
            ExecutionPath::Native,
            PlannedMethod::Plain,
            KvCodec::Dense,
            LIMITS,
        )
        .unwrap();
        let (weights, half) = half_the_experts(&draft, &definition);
        let cap = weights - half;
        (draft.with_bank::<AssessmentMemoryCharge>(&definition, Some(WeightBound::Cap(cap)), &synthetic()).unwrap().into_draft(), cap)
    }

    /// The weights of `draft` with its bank of every expert, and half its
    /// experts' slots' bytes: the bank's own figures, however a whole plan
    /// holds the experts (a whole bank, or allocations of their own, which
    /// bank none).
    fn half_the_experts(draft: &ExecutionPlanDraft, definition: &ModelDefinition) -> (u64, u64) {
        let bound = HeldBound::of(None, super::MemoryBudget::of_stable(draft.device()).unwrap().capacity());
        let BankCeiling { whole, .. } = draft.bank_ceiling(definition, bound, None).unwrap().expect("a routed load banks its experts");
        let bank = whole.bank();
        (whole.load().weight_bytes().unwrap(), bank.slot_bytes().unwrap() * (bank.widest().get() / 2))
    }

    /// A weight cap bounds every weight byte the load holds, its host tier's
    /// among them: Gemma 26B q4 capped at its weights less half its experts
    /// ([`capped_gemma`]) holds within the cap, its host tier ended by what
    /// the cap leaves beside the bank, not by the host's room.
    #[test]
    fn a_weight_cap_bounds_the_host_tier_beside_the_bank() {
        let (capped, cap) = capped_gemma();
        let held = capped.load().expert_bank().unwrap();
        assert!(!held.whole() && held.host_ceiling().is_some(), "half the experts stream through a host tier");
        let bytes = capped.load().weight_bytes().unwrap();
        assert!(bytes <= cap, "{bytes} weight bytes held under a cap of {cap}");
        // The cap, not the host's room, ends the tier.
        let ceiling = held.host_ceiling().unwrap();
        assert!(ceiling < held.widest().get() && held.host_slots <= ceiling, "{} host slots under a ceiling of {ceiling}", held.host_slots);
    }

    /// A host tier's first slots cost nothing beyond its floor (it is charged
    /// at least its landing reserve): the load still claims their backing,
    /// of no bytes, which the bank opens holding as its host slots require,
    /// even from weights that claim nothing in host RAM.
    #[test]
    fn a_host_tier_of_free_slots_claims_its_backing() {
        use crate::memory::ledger::{CommittedKind, DeviceLedgers, Ledgers, LoadShape};
        let mut load = capped_gemma().0.load().clone();
        load.bank.as_mut().expect("a routed load banks its experts").host_slots = 1;
        assert_eq!(load.host_tier_bytes().unwrap(), 0, "the tier's first slot costs nothing beyond its floor");
        let scope = Ledgers::new();
        let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
        let device = crate::test_support::open_backend(&catalog, BackendName::Cpu).expect("the CPU backend opens");
        let (ledgers, _) = DeviceLedgers::open_in(&scope, &device, std::sync::Arc::new(LoadShape { stores: Vec::new() }))
            .read_scripted(crate::memory::supply::Reading::device(64 << 30, 64 << 30, crate::domain::fixtures::NO_LAG));
        let mut weights = ledgers.changes().change(|keeping| ledgers.claim(&keeping.claims(), &device, CommittedKind::Load, crate::memory::ClaimDemand::default())).expect("nothing to claim");
        // The claim `BankCache::opened` requires of a bank with host slots.
        let backing = ledgers.host_backing(&mut weights, &load).unwrap();
        assert_eq!(backing.as_ref().map(crate::memory::ledger::Claim::bytes), Some(0), "a tier of host slots is claimed at no bytes");
    }

    /// A device room bounds only what the device holds: Gemma 26B q4 on the
    /// recorded RTX 3090, its device room its weights less half its
    /// experts, holds the bank a weight cap of the same bytes holds, while
    /// its host tier is left to host RAM: uncapped, it holds every expert
    /// the bank does not with 64 GB free, more than the cap's tier, none of
    /// it within the room.
    #[test]
    fn a_device_room_leaves_the_host_tier_to_host_memory() {
        use crate::planning::fixtures::{LIMITS, gemma_moe_fixture_definition, gemma_moe_fixture_manifest};
        let definition = gemma_moe_fixture_definition();
        let manifest = gemma_moe_fixture_manifest(&definition);
        let (_, _, device) = rtx3090(64 << 30);
        let draft = ExecutionPlanner::prepare(
            &device,
            &manifest,
            &definition,
            ComponentSelection { head: false, vision: false },
            ExecutionPath::Native,
            PlannedMethod::Plain,
            KvCodec::Dense,
            LIMITS,
        )
        .unwrap();
        let plan = |bound| draft.clone().with_bank::<AssessmentMemoryCharge>(&definition, Some(bound), &synthetic()).unwrap().into_draft();
        let (weights, half) = half_the_experts(&draft, &definition);
        let room = weights - half;
        let capped = plan(WeightBound::Cap(room));
        let roomy = plan(WeightBound::Room(room));
        let (capped, held) = (capped.load().expert_bank().unwrap(), roomy.load().expert_bank().unwrap());
        assert_eq!((held.slots, held.ceiling), (capped.slots, capped.ceiling), "the room bounds the device as the cap does");
        assert_eq!(held.host_cap, None, "no device room caps a host tier");
        let widest = held.widest().get();
        assert_eq!(held.host_ceiling(), Some(widest), "the tier may hold every expert");
        assert!(held.host_slots + held.slots >= widest, "64 GB of host RAM holds every expert the bank does not ({} host slots)", held.host_slots);
        assert!(capped.host_slots < held.host_slots, "the cap's tier ({}) is ended by the cap", capped.host_slots);
        let bytes = roomy.load().weight_bytes().unwrap();
        assert!(bytes > room, "{bytes} weight bytes held beside a device room of {room}: the tier lies outside it");
    }

    /// A streamed bank holds its arena's merged pass: Gemma 26B q4 on the
    /// recorded M4 Pro, uncapped (its structure's ceiling every expert) but
    /// with room beside its mandatory part for half its experts, streams
    /// the rest beside passes that merge launches. It commits only the forms
    /// it starts in: no decode block's resident form, since it starts
    /// banked.
    #[test]
    fn a_streamed_bank_holds_its_merged_pass() {
        use crate::planning::fixtures::{LIMITS, gemma_moe_fixture_definition, gemma_moe_fixture_manifest};
        let (_, _, device) = recorded(seismic::fixtures::recorded::m4_pro(), BackendName::Metal, 0, seismic::GraphCost::None);
        let definition = gemma_moe_fixture_definition();
        let manifest = gemma_moe_fixture_manifest(&definition);
        let draft = ExecutionPlanner::prepare(
            &device,
            &manifest,
            &definition,
            ComponentSelection { head: false, vision: false },
            ExecutionPath::Native,
            PlannedMethod::Plain,
            KvCodec::Dense,
            LIMITS,
        )
        .unwrap();
        let (_, half) = half_the_experts(&draft, &definition);
        let capacity = super::MemoryBudget::of_stable(draft.device()).unwrap().capacity();
        let BankCeiling { whole, ceiling, .. } = draft.bank_ceiling(&definition, HeldBound::of(None, capacity), None).unwrap().unwrap();
        let mandatory_bytes = whole
            .resized(whole.bank().floor(), ceiling, PassLaunches::FLOOR)
            .unwrap()
            .allocation_bytes(&definition)
            .unwrap();
        let budget = super::MemoryBudget::fixed(mandatory_bytes + half, None);
        let (planned, ()) = draft
            .with_held_experts(&definition, HeldBound::of(None, capacity), &synthetic(), &budget, |_, _| {
                Ok::<_, PlanError>(Ok(()))
            })
            .unwrap()
            .unwrap_or_else(|no| panic!("Gemma 26B q4 streams beside half its experts: {no:?}"));
        let held = planned.draft().load().expert_bank().unwrap();
        assert!(held.slots < held.widest().get(), "{} of {} slots", held.slots, held.widest());
        assert!(held.pass > PassLaunches::FLOOR, "{:?} beside {} streamed slots", held.pass, held.widest().get() - held.slots);
        assert_eq!(planned.charge().held_pass, held.pass, "settled pass must match held_pass");
        let load = planned.draft().load();
        let resident = load
            .target
            .iter()
            .filter(|weight| crate::operators::routed::streamed::resident_form(load, weight.role.scope))
            .count();
        assert_eq!(resident, 0, "a bank starting at {} of {} slots commits resident-form graphs", held.slots, held.widest());
    }

    /// A DeepSeek V4.1-sized routed model (40 layers of 384 experts
    /// choosing 8, 18.6 MB an expert, 285 GB of them) at a 64k context (a
    /// whole-context group of 128 places), prepared on `device`.
    fn far_larger(device: crate::platform::SelectedDevice) -> (ModelDefinition, ExecutionPlanDraft) {
        use crate::planning::fixtures::{LIMITS, RoutedShape, gemma_moe_fixture_manifest, routed_fixture_definition};
        let mut definition =
            routed_fixture_definition(RoutedShape { layers: 40, experts: 384, selected: 8, hidden: 4096, intermediate: 2688 });
        definition.decoder.context_limit = 65_536;
        let manifest = gemma_moe_fixture_manifest(&definition);
        let draft = ExecutionPlanner::prepare(
            &device,
            &manifest,
            &definition,
            ComponentSelection { head: false, vision: false },
            ExecutionPath::Native,
            PlannedMethod::Plain,
            KvCodec::Dense,
            LIMITS,
        )
        .unwrap();
        (definition, draft)
    }

    /// [`far_larger`]'s bank on the recorded M4 Pro as a plan of it under
    /// `bound` holds it, with room beside its mandatory part for 20 GB.
    fn far_larger_bank(bound: impl FnOnce(u64) -> HeldBound) -> super::super::ExpertBankPlan {
        let (_, _, device) = recorded(seismic::fixtures::recorded::m4_pro(), BackendName::Metal, 0, seismic::GraphCost::None);
        let (definition, draft) = far_larger(device);
        let bound = bound(super::MemoryBudget::of_stable(draft.device()).unwrap().capacity());
        let BankCeiling { whole, ceiling, .. } = draft.bank_ceiling(&definition, bound, None).unwrap().unwrap();
        let mandatory_bytes = whole
            .resized(whole.bank().floor(), ceiling, PassLaunches::FLOOR)
            .unwrap()
            .allocation_bytes(&definition)
            .unwrap();
        let budget = super::MemoryBudget::fixed(mandatory_bytes + FAR_ROOM, None);
        let (planned, ()) = draft
            .with_held_experts(
                &definition,
                bound,
                &synthetic(),
                &budget,
                |_, _| Ok::<_, PlanError>(Ok(())),
            )
            .unwrap()
            .unwrap_or_else(|no| panic!("a V4.1-sized model streams under {bound:?}: {no:?}"));
        planned.draft().load().expert_bank().unwrap().clone()
    }

    const FAR_ROOM: u64 = 20_000_000_000;

    /// On a machine whose prices are not measured, a load's plan of a bank
    /// waits for them.
    #[test]
    fn an_unmeasured_load_plan_waits_for_its_bank_prices() {
        let (_, _, device) = recorded(seismic::fixtures::recorded::m4_pro(), BackendName::Metal, 0, seismic::GraphCost::None);
        let (definition, draft) = far_larger(device);
        let capacity = super::MemoryBudget::of_stable(draft.device()).unwrap().capacity();
        let budget = super::MemoryBudget::fixed(FAR_ROOM, None);
        let waits = draft
            .with_held_experts(
                &definition,
                HeldBound::of(None, capacity),
                &pending(),
                &budget,
                |_, _| Ok::<_, PlanError>(Ok(())),
            )
            .err();
        assert!(matches!(waits, Some(PlanError::Pending(_))), "{waits:?}");
    }

    /// Capped at 29 GB, below a tenth of its experts' weights beside the
    /// rest, a model far larger than its machine streams a bank between
    /// what a step needs and that tenth: nothing a step can run on is
    /// refused. Its pass merges launches beside the bank: a pass of two
    /// holds the class of two places, not of the 128 the chain may hold.
    #[test]
    fn a_capped_load_far_larger_than_its_machine_streams_and_merges_its_passes() {
        let capped = far_larger_bank(|capacity| HeldBound::of(Some(WeightBound::Cap(29_000_000_000)), capacity));
        let tenth = capped.widest().get().div_ceil(10);
        let found = format!("{} slots (floor {}, a tenth {tenth}), ceiling {}, {:?} of {} places", capped.slots, capped.floor(), capped.ceiling, capped.pass, capped.places);
        assert!(capped.ceiling < tenth && capped.places == 128, "{found}");
        assert!(capped.pass > PassLaunches::FLOOR, "{found}");
    }

    /// A streamed bank settles its path records at its slots, carries them
    /// priced on the stored line, and its load logs them.
    #[test]
    fn a_streamed_bank_logs_the_path_records_it_settles_at() {
        let mut capped = None;
        let messages = logged(|| capped = Some(far_larger_bank(|capacity| HeldBound::of(Some(WeightBound::Cap(29_000_000_000)), capacity))));
        let capped = capped.expect("the load planned");
        let crate::planning::BankPaths::Priced { records, .. } = capped.paths else {
            panic!("a streamed bank's path is priced: {:?}", capped.paths);
        };
        let settled = messages.iter().filter(|message| message.contains("the bank's path records are settled")).collect::<Vec<_>>();
        let fields = format!("slots={} ahead={} reserve={}", capped.slots, records.ahead, records.reserve);
        assert!(settled.iter().any(|message| message.contains(&fields)), "{fields} in {messages:?}");
        assert!(records.ahead > 0 && records.reserve >= records.ahead, "{records:?}");
    }

    /// Every event `run` logs on this thread: its message and fields, as
    /// `message key=value ...`.
    fn logged(run: impl FnOnce()) -> Vec<String> {
        use std::sync::{Arc, Mutex};
        struct Fields(String);
        impl tracing::field::Visit for Fields {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                match field.name() {
                    "message" => self.0.insert_str(0, &format!("{value:?}")),
                    name => self.0.push_str(&format!(" {name}={value:?}")),
                }
            }
        }
        struct Events(Arc<Mutex<Vec<String>>>);
        impl tracing::Subscriber for Events {
            fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
                true
            }
            fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
                tracing::span::Id::from_u64(1)
            }
            fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
            fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
            fn event(&self, event: &tracing::Event<'_>) {
                let mut fields = Fields(String::new());
                event.record(&mut fields);
                self.0.lock().unwrap().push(fields.0);
            }
            fn enter(&self, _: &tracing::span::Id) {}
            fn exit(&self, _: &tracing::span::Id) {}
        }
        let events = Arc::new(Mutex::new(Vec::new()));
        tracing::subscriber::with_default(Events(events.clone()), run);
        let logged = events.lock().unwrap().clone();
        logged
    }

    /// Item 5: uncapped, with room beside its mandatory part for 20 GB at
    /// the load, a model far larger than its machine loads a bank within
    /// that room, but its ceiling is structural (the most slots the device
    /// holds), never the room: memory that frees after the load lets its
    /// target grow past the load-time room. It merges its passes beside it.
    #[test]
    fn an_uncapped_load_far_larger_than_its_machine_grows_past_its_load_room() {
        let uncapped = far_larger_bank(|capacity| HeldBound::of(None, capacity));
        let charge = |slots| uncapped.resident_charge(slots).unwrap();
        let (loaded, reach) = (charge(uncapped.slots) - charge(uncapped.floor()), charge(uncapped.ceiling) - charge(uncapped.floor()));
        let found = format!(
            "{} of {} slots ({loaded} bytes past the floor), ceiling {} ({reach} bytes) beside the load room's {FAR_ROOM}, {:?} of {} places",
            uncapped.slots,
            uncapped.widest(),
            uncapped.ceiling,
            uncapped.pass,
            uncapped.places,
        );
        assert!(loaded <= FAR_ROOM && reach > FAR_ROOM, "{found}");
        assert!(uncapped.pass > PassLaunches::FLOOR, "{found}");
    }

    /// A host tier's link line is host storage's routes of the executor's
    /// costs, a standing request of the job: a bank on the recorded RTX
    /// 3090 whose costs have not read them yet plans pending, waiting for
    /// the job as the load does, and prices once they are read.
    #[test]
    fn a_bank_plans_pending_until_its_costs_read_host_storage() {
        use crate::assessment::executor_costs::{Placement, RouteCosts};
        use crate::assessment::synthetic::{costs, kind, plan_prices_on};
        let (_, _, device) = rtx3090(64 << 30);
        let (definition, draft) = far_larger(device);
        let along_host = |activation| {
            let mut costs = costs();
            costs.placements.host = Placement::along(kind(1));
            costs.routes.push(RouteCosts { route: kind(1), activation, formats: Vec::new() });
            Priced::Measured(plan_prices_on(costs))
        };
        let unread = draft.clone().with_bank::<AssessmentMemoryCharge>(&definition, None, &along_host(None)).err();
        assert!(matches!(unread, Some(PlanError::Pending(_))), "{unread:?}");
        let read = draft.with_bank::<AssessmentMemoryCharge>(&definition, None, &along_host(Some(costs().first.activation)));
        let loaded = read.unwrap_or_else(|error| panic!("a bank prices once its costs read host storage: {error}")).into_draft();
        assert!(loaded.load().expert_bank().is_some());
    }

    /// Planned as a load is, uncapped on the machine's own room (the
    /// recorded RTX 3090, 24 GB, with 64 GB of host RAM free, too little
    /// for a tenth of its experts beside the rest), a model far larger than
    /// its machine streams a bank of the slots the device holds, merging its
    /// passes, never imported whole.
    #[test]
    fn an_uncapped_load_far_larger_than_its_machine_streams_never_whole() {
        let (_, _, device) = rtx3090(64 << 30);
        let (definition, draft) = far_larger(device);
        let loaded = draft
            .with_bank::<AssessmentMemoryCharge>(&definition, None, &synthetic())
            .unwrap_or_else(|error| panic!("a V4.1-sized model plans uncapped on an RTX 3090: {error}"))
            .into_draft();
        let streamed = loaded.load().expert_bank().expect("an uncapped load larger than its machine streams its experts");
        assert!(streamed.slots < streamed.widest().get(), "{} of {} slots", streamed.slots, streamed.widest());
        assert!(streamed.pass > PassLaunches::FLOOR, "{:?} of {} places", streamed.pass, streamed.places);
    }

    /// A plan holding its experts whole, each in its own allocation, that
    /// the memory free at its import cannot hold, streams them instead
    /// ([`ExecutionPlanDraft::streaming`]): the import's admission refuses the
    /// whole plan and fits the streamed plan's bank, from its floor, into
    /// the same room. A small routed model planned on the recorded M4 Pro,
    /// its imports admitted by the test host's ledgers, which another
    /// program presses to midway between the streamed floor's import and
    /// the whole plan's.
    #[test]
    fn a_whole_plan_the_import_cannot_hold_streams_its_bank_fitted_by_the_admission() {
        use crate::memory::ClaimDemand;
        use crate::planning::fixtures::{LIMITS, RoutedShape, gemma_moe_fixture_manifest, routed_fixture_definition};
        let (_, _, device) = recorded(seismic::fixtures::recorded::m4_pro(), BackendName::Metal, 0, seismic::GraphCost::None);
        let mut definition =
            routed_fixture_definition(RoutedShape { layers: 8, experts: 64, selected: 4, hidden: 1024, intermediate: 512 });
        definition.decoder.context_limit = 4096;
        let manifest = gemma_moe_fixture_manifest(&definition);
        let draft = ExecutionPlanner::prepare(
            &device,
            &manifest,
            &definition,
            ComponentSelection { head: false, vision: false },
            ExecutionPath::Native,
            PlannedMethod::Plain,
            KvCodec::Dense,
            LIMITS,
        )
        .unwrap();
        let whole = draft.held_whole(&definition).unwrap();
        assert!(whole.load().expert_bank().is_none(), "the whole plan holds each expert in its own allocation");
        let device = draft.device().clone();
        let budget = super::MemoryBudget::of_stable(&device).unwrap();
        let streamed = draft
            .streaming::<AssessmentMemoryCharge>(&definition, None, &budget)
            .unwrap()
            .streamed(&definition, &synthetic(), &budget)
            .unwrap()
            .into_draft();
        let bank = streamed.load().expert_bank().expect("the streamed plan banks its experts").clone();
        // Each import's storage, admitted as device bytes.
        let import = |load: &ModelLoadPlan| ClaimDemand { device: load.target_weight_demand().unwrap().bytes(), ..ClaimDemand::default() };
        let at = |slots: u64| Ok::<_, std::convert::Infallible>(import(&streamed.load().clone().with_load_slots(slots, bank.host_slots).unwrap()));
        let (whole_bytes, floor_bytes) = (import(whole.load()).device, at(bank.floor()).unwrap().device);
        assert!(floor_bytes < whole_bytes, "the streamed floor imports {floor_bytes} bytes, the whole plan {whole_bytes}");
        // Another program holds all but midway between the two imports.
        let target = floor_bytes + (whole_bytes - floor_bytes) / 2;
        let shape = std::sync::Arc::new(crate::memory::ledger::LoadShape { stores: Vec::new() });
        // The ledgers of the test host's CPU device, whose stores read as pressed.
        let catalog = seismic::Devices::discover(&[]).unwrap();
        let host = crate::test_support::open_backend(&catalog, BackendName::Cpu).unwrap();
        let (ledgers, reading) = crate::memory::ledger::DeviceLedgers::open_in(&crate::memory::ledger::Ledgers::new(), &host, shape)
            .read_scripted(crate::memory::supply::Reading::device(2 * whole_bytes, target, crate::domain::fixtures::NO_LAG));
        // The whole plan is refused for room ...
        let load = crate::memory::ledger::CommittedKind::Load;
        assert!(ledgers.changes().change(|keeping| ledgers.admit_most(&keeping.claims(), &reading, &host, load, 0, 0, |_| Ok::<_, std::convert::Infallible>(import(whole.load())))).unwrap().is_err());
        // ... and the streamed plan's bank is fitted into it: the most slots
        // that fit, and no more.
        let (slots, _held) = ledgers.changes().change(|keeping| ledgers.admit_most(&keeping.claims(), &reading, &host, load, bank.floor(), bank.slots, &at)).unwrap().unwrap();
        assert!(at(slots).unwrap().device <= target, "{slots} slots of {} fit the {target} bytes left", bank.slots);
        assert!(slots == bank.slots || at(slots + 1).unwrap().device > target, "{slots} slots are the most that fit");
    }

    /// When a device budget is below `W == widest`'s floor but above `W_min == round_records`'s
    /// floor, `with_held_experts` picks the largest `W ∈ [round_records, widest]` whose plan fits.
    #[test]
    fn tight_budget_shrinks_scratch_window_to_largest_fitting_w() {
        use crate::planning::fixtures::{LIMITS, RoutedShape, gemma_moe_fixture_manifest, routed_fixture_definition};
        let (_, _, device) = recorded(seismic::fixtures::recorded::m4_pro(), BackendName::Metal, 0, seismic::GraphCost::None);
        let mut definition =
            routed_fixture_definition(RoutedShape { layers: 4, experts: 64, selected: 4, hidden: 512, intermediate: 512 });
        definition.decoder.context_limit = 2048;
        let manifest = gemma_moe_fixture_manifest(&definition);
        let draft = ExecutionPlanner::prepare(
            &device,
            &manifest,
            &definition,
            ComponentSelection { head: false, vision: false },
            ExecutionPath::Native,
            PlannedMethod::Plain,
            KvCodec::Dense,
            LIMITS,
        )
        .unwrap();
        let budget = super::MemoryBudget::of_selected_stable(&device).unwrap();
        let full = draft
            .clone()
            .streaming::<AssessmentMemoryCharge>(&definition, None, &budget)
            .unwrap()
            .streamed(&definition, &synthetic(), &budget)
            .unwrap()
            .into_draft();
        let full_bank = full.load().expert_bank().unwrap();
        let widest = full_bank.widest().get();
        let round = full_bank.round_records();
        assert_eq!(full_bank.window_slots(), widest);

        // Compute the floor allocation bytes at W = widest and W = round_records.
        let full_floor_bytes = full
            .load()
            .clone()
            .with_load_slots(full_bank.floor(), full_bank.host_slots)
            .unwrap();
        let full_floor_draft = ExecutionPlanDraft {
            load: Box::new(full_floor_bytes),
            ..full.clone()
        };
        let full_floor_alloc = full_floor_draft.allocation_bytes(&definition).unwrap();

        let min_load = full.load().clone().banked_as(&definition, full_bank, round).unwrap();
        let min_floor_load = min_load
            .clone()
            .with_load_slots(min_load.expert_bank().unwrap().floor(), min_load.expert_bank().unwrap().host_slots)
            .unwrap();
        let min_floor_programs = min_floor_load
            .program_plan(&definition, full.policy().codec())
            .unwrap();
        let min_floor_alloc = ExecutionPlanDraft {
            load: Box::new(min_floor_load),
            programs: Box::new(min_floor_programs),
            ..full.clone()
        }
        .allocation_bytes(&definition)
        .unwrap();
        assert!(min_floor_alloc < full_floor_alloc, "smaller W reduces floor allocation: {min_floor_alloc} < {full_floor_alloc}");

        // Place a device memory cap halfway between W_min's floor and W_widest's floor,
        // with a weight cap at W_widest's floor weights so the unbanked resident plan
        // is excluded and the bank must settle by shrinking W.
        let cap = min_floor_alloc + (full_floor_alloc - min_floor_alloc) / 2;
        let weight_cap = full_floor_draft
            .memory_charge(&definition)
            .unwrap()
            .weight_demand
            .checked_bytes()
            .unwrap();
        let bound = HeldBound::of(Some(WeightBound::Cap(weight_cap)), budget.capacity());
        let (settled, _) = draft
            .with_held_experts(&definition, bound, &synthetic(), &budget, |_, charge| {
                let bytes = charge.store_demands(draft.device().topology(), draft.device().executor()).unwrap().allocation_bytes();
                Ok::<_, PlanError>(if bytes <= cap { Ok(()) } else { Err(NonZeroU64::new(bytes - cap).unwrap()) })
            })
            .unwrap()
            .unwrap();
        let chosen_bank = settled.draft().load().expert_bank().unwrap();
        assert!(
            chosen_bank.window_slots() >= round && chosen_bank.window_slots() < widest,
            "expected W in [{round}, {widest}), got {}",
            chosen_bank.window_slots()
        );
    }

    #[test]
    fn command_buffer_capacity_scales_with_streamed_layers_and_flight_depth() {
        use crate::planning::fixtures::{LIMITS, RoutedShape, gemma_moe_fixture_manifest, routed_fixture_definition};
        use crate::programs::native_target::SubmissionChunks;

        assert_eq!(SubmissionChunks::count(0), 0);
        assert_eq!(SubmissionChunks::count(1), 1);
        assert_eq!(SubmissionChunks::count(2), 2);
        assert_eq!(SubmissionChunks::count(3), 2);
        assert_eq!(SubmissionChunks::count(4), 3);
        assert_eq!(SubmissionChunks::count(7), 3);
        assert_eq!(SubmissionChunks::count(8), 4);
        assert_eq!(SubmissionChunks::count(50), 6);

        let (_, _, device) = recorded(seismic::fixtures::recorded::m4_pro(), BackendName::Metal, 0, seismic::GraphCost::None);
        let budget = super::MemoryBudget::of_selected_stable(&device).unwrap();

        let streamed_draft = |layers: u64, lookahead: bool| {
            let mut definition = routed_fixture_definition(RoutedShape {
                layers,
                experts: 16,
                selected: 2,
                hidden: 256,
                intermediate: 256,
            });
            definition.decoder.context_limit = 512;
            let manifest = gemma_moe_fixture_manifest(&definition);
            let limits = ResourceLimits { lookahead, ..LIMITS };
            let resident = ExecutionPlanner::prepare(
                &device,
                &manifest,
                &definition,
                ComponentSelection { head: false, vision: false },
                ExecutionPath::Native,
                PlannedMethod::Plain,
                KvCodec::Dense,
                limits,
            )
            .unwrap();
            let streamed = resident
                .clone()
                .streaming::<AssessmentMemoryCharge>(&definition, None, &budget)
                .unwrap()
                .streamed(&definition, &synthetic(), &budget)
                .unwrap()
                .into_draft();
            (resident, streamed)
        };

        let (resident_4, streamed_4_serial) = streamed_draft(4, false);
        let (_, streamed_4_pipelined) = streamed_draft(4, true);
        let (_, streamed_48_pipelined) = streamed_draft(48, true);
        let (_, streamed_160_pipelined) = streamed_draft(160, true);

        let cap = |draft: &ExecutionPlanDraft| {
            command_buffer_capacity(draft.programs(), draft.load(), draft.policy().limits())
        };

        let cap_resident_4 = cap(&resident_4);
        let cap_4_serial = cap(&streamed_4_serial);
        let cap_4_pipelined = cap(&streamed_4_pipelined);
        let cap_48_pipelined = cap(&streamed_48_pipelined);
        let cap_160_pipelined = cap(&streamed_160_pipelined);

        // Streamed loads require strictly more command buffers than resident
        // loads of the same depth.
        assert!(cap_4_serial > cap_resident_4, "{cap_4_serial} > {cap_resident_4}");
        // Deeper pipelining (`lookahead: true`, 2 groups vs 1) strictly grows
        // the derived bound.
        assert!(cap_4_pipelined > cap_4_serial, "{cap_4_pipelined} > {cap_4_serial}");
        // More streamed layers strictly grow the derived bound, including past
        // 128 layers with no hardcoded ceiling.
        assert!(cap_48_pipelined > cap_4_pipelined, "{cap_48_pipelined} > {cap_4_pipelined}");
        assert!(cap_160_pipelined > cap_48_pipelined, "{cap_160_pipelined} > {cap_48_pipelined}");

        // Never under-provisions a streamed load: covers at least
        // (submission_chunks(layers + 2) + 2 cuts/layer) * groups + service CB.
        for (draft, layers) in [
            (&streamed_4_serial, 4usize),
            (&streamed_4_pipelined, 4usize),
            (&streamed_48_pipelined, 48usize),
            (&streamed_160_pipelined, 160usize),
        ] {
            let groups = draft.policy().limits().flights().groups();
            let decode_minimum = (SubmissionChunks::count(layers + 2) + layers * 2) * groups + 1;
            let derived = cap(draft);
            assert!(
                derived >= decode_minimum,
                "layers={layers}, groups={groups}: derived {derived} < decode minimum {decode_minimum}"
            );
        }
    }

    #[test]
    fn settle_bank_paths_converges_to_covering_envelope_on_cycle() {
        let line = magnitude_cost_model::ReadLine::new(1e-6, 1e-10).unwrap();
        let priced = |ahead, reserve| {
            crate::planning::BankPaths::Priced {
                line,
                records: magnitude_cost_model::PathRecords { ahead, reserve },
            }
        };

        // 2-cycle p_high -> p_low -> p_high: closes while settled at p_low,
        // so settles once more at envelope p_high.
        let p_low = priced(2, 3);
        let p_high = priced(4, 6);
        let mut visited = Vec::new();
        let (settled, value, count) = settle_bank_paths::<_, PlanError>(p_high, |p| {
            visited.push(p);
            let next = if p == p_high { p_low } else { p_high };
            Ok((p.records(), Some(next)))
        })
        .unwrap();
        assert_eq!(settled, p_high);
        assert_eq!(value, p_high.records());
        assert_eq!(count, 3);
        assert_eq!(visited, vec![p_high, p_low, p_high]);

        // 2-cycle p_low -> p_high -> p_low: closes while already settled at
        // envelope p_high, so returns without an extra settle.
        let mut visited = Vec::new();
        let (settled, value, count) = settle_bank_paths::<_, PlanError>(p_low, |p| {
            visited.push(p);
            let next = if p == p_low { p_high } else { p_low };
            Ok((p.records(), Some(next)))
        })
        .unwrap();
        assert_eq!(settled, p_high);
        assert_eq!(value, p_high.records());
        assert_eq!(count, 2);
        assert_eq!(visited, vec![p_low, p_high]);

        // Incomparable 2-cycle {(2, 5), (4, 3)} -> covering envelope (4, 5).
        let p_a = priced(2, 5);
        let p_b = priced(4, 3);
        let envelope = priced(4, 5);
        let mut visited = Vec::new();
        let (settled, value, count) = settle_bank_paths::<_, PlanError>(p_a, |p| {
            visited.push(p);
            let next = if p == p_a { p_b } else { p_a };
            Ok((p.records(), Some(next)))
        })
        .unwrap();
        assert_eq!(settled, envelope);
        assert_eq!(value, envelope.records());
        assert_eq!(count, 3);
        assert_eq!(visited, vec![p_a, p_b, envelope]);
    }

    /// A `WholeForm::Resident` routed load retains its merged pass (`held_pass > 1`)
    /// when memory allows, and steps down `chain_capacity` / `held_pass` cleanly
    /// along `whole_chains` when room is tight so it never OOMs.
    #[test]
    fn resident_retains_merged_pass_and_steps_down_when_room_is_tight() {
        use crate::planning::fixtures::{LIMITS, RoutedShape, gemma_moe_fixture_manifest, routed_fixture_definition};
        let (topology, _, device) = recorded(seismic::fixtures::recorded::m4_pro(), BackendName::Cpu, 0, seismic::GraphCost::None);
        let mut definition = routed_fixture_definition(RoutedShape {
            layers: 4,
            experts: 32,
            selected: 4,
            hidden: 1024,
            intermediate: 512,
        });
        definition.decoder.context_limit = 8192;
        let manifest = gemma_moe_fixture_manifest(&definition);
        let draft = ExecutionPlanner::prepare(
            &device,
            &manifest,
            &definition,
            ComponentSelection { head: false, vision: false },
            ExecutionPath::Native,
            PlannedMethod::Plain,
            KvCodec::Dense,
            LIMITS,
        )
        .unwrap();
        let executor = draft.device().executor();
        let bytes_at = |places: u64| {
            let candidate = draft.held_whole_at(&definition, places).unwrap();
            let charge = candidate.memory_charge(&definition).unwrap();
            assert_eq!(charge.held_pass.get(), places as usize, "resident charge must honor places={places}");
            charge.store_demands(&topology, executor).unwrap().allocation_bytes()
        };
        let full_places = draft.whole_places(&definition).unwrap();
        assert!(full_places >= 8, "setup: expected multi-chunk prompt places, got {full_places}");
        let floor_bytes = bytes_at(1);
        let mid_places = full_places / 2;
        let mid_bytes = bytes_at(mid_places);
        let full_bytes = bytes_at(full_places);
        assert!(
            full_bytes > floor_bytes,
            "merged pass ({full_bytes} bytes at {full_places} places) must charge more than floor ({floor_bytes} bytes)",
        );

        let place_in = |room_bytes: u64| {
            let budget = super::MemoryBudget::fixed(room_bytes, None);
            draft
                .with_held_experts(
                    &definition,
                    HeldBound::room(room_bytes),
                    &synthetic(),
                    &budget,
                    |_, _| Ok::<_, PlanError>(Ok(())),
                )
                .unwrap()
                .unwrap()
                .0
        };

        // Generous room retains the full merged pass.
        let wide = place_in(full_bytes + (64 << 20));
        assert!(wide.draft().load().expert_bank().is_none(), "CPU load is WholeForm::Resident");
        assert_eq!(wide.draft().policy().limits().chain_capacity.get(), full_places as usize);
        assert_eq!(wide.charge().held_pass.get(), full_places as usize);

        // Tight room below `full_bytes` but holding `mid_bytes` steps down cleanly.
        if mid_bytes < full_bytes {
            let tight = place_in(mid_bytes);
            assert!(tight.draft().load().expert_bank().is_none());
            assert_eq!(tight.draft().policy().limits().chain_capacity.get(), mid_places as usize);
            assert_eq!(tight.charge().held_pass.get(), mid_places as usize);
            assert!(
                tight.charge().store_demands(&topology, executor).unwrap().allocation_bytes() <= mid_bytes,
                "stepped-down resident plan must fit within tight room",
            );
        }

        // Room equal to floor steps all the way down to 1 launch.
        let floor = place_in(floor_bytes);
        assert!(floor.draft().load().expert_bank().is_none());
        assert_eq!(floor.draft().policy().limits().chain_capacity.get(), 1);
        assert_eq!(floor.charge().held_pass, PassLaunches::FLOOR);
    }

    /// `NoBank::Short` reports the exact `(required, available)` from `WholeFit`
    /// against live `device_room`, never `capacity + deficit_bytes` / `capacity`.
    #[test]
    fn no_bank_short_reports_exact_required_and_live_device_room() {
        use crate::planning::fixtures::{LIMITS, RoutedShape, gemma_moe_fixture_manifest, routed_fixture_definition};

        let (_, _, device) = recorded(seismic::fixtures::recorded::m4_pro(), BackendName::Metal, 0, seismic::GraphCost::None);
        let definition = routed_fixture_definition(RoutedShape {
            layers: 4,
            experts: 32,
            selected: 4,
            hidden: 1024,
            intermediate: 512,
        });
        let manifest = gemma_moe_fixture_manifest(&definition);
        let draft = ExecutionPlanner::prepare(
            &device,
            &manifest,
            &definition,
            ComponentSelection { head: false, vision: false },
            ExecutionPath::Native,
            PlannedMethod::Plain,
            KvCodec::Dense,
            LIMITS,
        )
        .unwrap();

        let total_capacity = 64_u64 << 30; // 64 GiB total machine capacity
        let live_device_room = 1_u64 << 20; // 1 MiB live free room (far below floor)
        let mut budget = super::MemoryBudget::fixed(live_device_room, None);
        budget.device.capacity = total_capacity;
        let Placed::Short(unplaced) = draft.placed::<AssessmentMemoryCharge>(&definition, None, &synthetic(), &budget).unwrap() else {
            panic!("expected Placed::Short when live device_room is 1 MiB");
        };
        let err = match unplaced.streamed(&definition, &synthetic(), &budget) {
            Err(PlanError::Resource(err)) => err,
            Err(other) => panic!("expected PlanError::Resource, got {other:?}"),
            Ok(_) => panic!("expected refusal when live device_room is 1 MiB"),
        };
        assert_eq!(err.resource, crate::ResourceKind::DeviceMemory);
        assert_eq!(
            err.available, live_device_room,
            "available must be live device_room, not total machine capacity",
        );
        assert!(
            err.required > live_device_room && err.required < total_capacity,
            "required ({}) must be the actual floor need (> {} and << total capacity {})",
            err.required,
            live_device_room,
            total_capacity,
        );
    }

    /// An unstreamable (dense) model whose resident charge exceeds the live
    /// budget fails immediately at `placed()` with `PlanError::Resource`
    /// (`required`, `available`), never deferred past program preparation and
    /// graph sealing.
    #[test]
    fn unstreamable_model_fails_immediately_when_live_budget_is_short() {
        use crate::planning::fixtures::{LIMITS, fixture_definition, fixture_manifest};

        let (_, _, device) = recorded(seismic::fixtures::recorded::m4_pro(), BackendName::Metal, 0, seismic::GraphCost::None);
        let definition = fixture_definition();
        let manifest = fixture_manifest(&definition);
        let draft = ExecutionPlanner::prepare(
            &device,
            &manifest,
            &definition,
            ComponentSelection { head: false, vision: false },
            ExecutionPath::Native,
            PlannedMethod::Plain,
            KvCodec::Dense,
            LIMITS,
        )
        .unwrap();

        let live_available = 1_u64 << 10; // 1 KiB — far below resident charge
        let mut budget = super::MemoryBudget::fixed(live_available, None);
        budget.device.capacity = 36_u64 << 30;
        let err = match draft.placed::<AssessmentMemoryCharge>(&definition, None, &synthetic(), &budget) {
            Err(PlanError::Resource(err)) => err,
            Err(other) => panic!("expected PlanError::Resource for short unstreamable model, got {other:?}"),
            Ok(_) => panic!("unstreamable model must not be admitted when live budget is short"),
        };
        assert_eq!(err.resource, crate::ResourceKind::DeviceMemory);
        assert_eq!(err.available, live_available);
        assert!(err.required > live_available);
    }
}
