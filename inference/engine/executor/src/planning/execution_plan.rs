//! One allocation-free, checked numerical startup decision.

use super::{
    ArtifactComponent, ArtifactComponentKind, AssessmentMemoryCharge, CapabilityPlan,
    ComponentPlan, ComponentSelection, HostTier, MAX_DRAFT_PROPOSALS, ModelLoadPlan, PassLaunches, PlannedMethod,
    ProgramPlan, ReadoutClassChoice, ResourceLimits, ResourcePlan, WeightPlan,
};
use crate::{
    ExecutionPath, UnstreamedExperts,
    assessment::Priced,
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
    assessment_capacity_bytes: u64,
    tensor_operations: bool,
    /// Its largest single allocation and how it reserves, as discovery
    /// reports them: what an expert bank's reserved record tensors are
    /// bounded by, backed in and whether the host reads into them (none: it
    /// reserves nothing; why not where discovery could not read how).
    max_allocation_bytes: u64,
    reservation: Result<Option<seismic::Reservation>, String>,
    /// How its host tier is held, where it has one, or why it has none
    /// (`SelectedDevice::host_tier`).
    host: Result<Option<HostTier>, String>,
    /// The executor and what its selection's fit questions read
    /// (`SelectedDevice::reading`): what a plan's fit beside a linked host
    /// tier asks of `Fit` ([`ExecutionPlanDraft::with_host_tier`]).
    executor: seismic::Executor,
    reading: FitReading,
    /// What its recorded graphs are priced at for their census, and
    /// whether that is the device's own cost or its backend's bound
    /// (`SelectedDevice::graph_cost`).
    graph_pricing: crate::platform::GraphPricing,
}

impl PlannedDevice {
    pub fn selector(&self) -> ExecutorSelector {
        self.selector
    }

    pub fn backend(&self) -> BackendName {
        self.backend
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Stable capacity of the selected allocation domain at planning time.
    pub fn assessment_capacity_bytes(&self) -> u64 {
        self.assessment_capacity_bytes
    }

    /// Whether the device forms Metal tensor operations
    /// (`Executor::forms_tensor_operations`).
    pub fn tensor_operations(&self) -> bool {
        self.tensor_operations
    }

    /// How its host tier is held, where it has one, or why it has none
    /// (`SelectedDevice::host_tier`).
    pub fn host_tier(&self) -> Result<Option<HostTier>, &str> {
        self.host.as_ref().copied().map_err(String::as_str)
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionPlan {
    device: PlannedDevice,
    components: ComponentPlan,
    load: ModelLoadPlan,
    programs: ProgramPlan,
    resources: ResourcePlan,
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionPlanDraft {
    device: PlannedDevice,
    components: ComponentPlan,
    load: ModelLoadPlan,
    programs: ProgramPlan,
    capabilities: CapabilityPlan,
    policy: ResolvedPolicy,
}

impl ExecutionPlanDraft {
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
    /// residency are admitted; a load the planes do not fit keeps every
    /// scope packed.
    pub fn finalize_readout(
        mut self,
        definition: &ModelDefinition,
        choices: &[ReadoutClassChoice],
    ) -> Result<Self, PlanError> {
        let canonical = self.clone();
        self.load = self
            .load
            .finalize_readout(choices)
            .map_err(PlanError::InvalidDefinition)?;
        self.programs = self.load.program_plan(definition, self.policy.codec)?;
        if !choices.is_empty()
            && Self::allocated(&self.memory_charge(definition)?)? > self.device.assessment_capacity_bytes
        {
            return canonical.finalize_readout(definition, &[]);
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
    /// model's context, as its assessment charges it ([`Self::allocated`]).
    pub(crate) fn allocation_bytes(&self, definition: &ModelDefinition) -> Result<u64, PlanError> {
        Self::allocated(&self.memory_charge(definition)?)
    }

    /// What `charge` allocates in the device's allocation domain: all but a
    /// linked host tier's records, which lie in host storage
    /// ([`Self::with_host_tier`] sizes them against host RAM).
    fn allocated(charge: &AssessmentMemoryCharge) -> Result<u64, PlanError> {
        charge
            .allocation_bytes
            .checked_sub(charge.weight_demand.host)
            .ok_or(PlanError::Arithmetic("a host tier's records exceed the allocation charge"))
    }

    /// The draft with its linked host tier (its records in host storage)
    /// sized at load: the most host slots' records
    /// (`ExpertBankPlan::host_charge`, the steps where it rises, up to every
    /// expert, `ExpertBankPlan::host_ceiling`) the room host RAM holds beside
    /// the load's charge at the tier's floor
    /// (`AssessmentMemoryCharge::assess_fit`, `Fit` alone interpreting
    /// placements, stores and reserves); none past its floor beside a bank
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
    fn with_host_tier(mut self, charge: &AssessmentMemoryCharge) -> Result<Result<Self, UnstreamedExperts>, PlanError> {
        if !matches!(self.device.host, Ok(Some(HostTier::Linked(_)))) {
            return Ok(Ok(self));
        }
        let Some(ceiling) = self.bank().host_ceiling().filter(|&ceiling| ceiling > 0 && !self.bank().whole()) else {
            return Ok(Ok(self));
        };
        let bank = self.bank();
        let host_charge = |slots: u64| bank.host_charge(slots).map_err(PlanError::ResourcePlanning);
        let left = |slots: u64| self.host_left(charge, slots);
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

    /// The host RAM `charge`, the draft's with its tier at its floor, leaves
    /// beside a tier of `slots`: negative where it does not fit.
    fn host_left(&self, charge: &AssessmentMemoryCharge, slots: u64) -> Result<i64, PlanError> {
        let FitReading { topology, status, reserves } = &self.device.reading;
        let host_ram = topology.host_ram().id();
        let growth = self.bank().host_cache(slots).map_err(PlanError::ResourcePlanning)?;
        let fit = charge
            .clone()
            .with_weights(crate::memory::ClaimDemand { host: growth, ..Default::default() })
            .and_then(|charge| charge.assess_fit(topology, &self.device.executor, status, reserves))
            .map_err(PlanError::ResourcePlanning)?;
        let host = fit
            .domains
            .iter()
            .find(|domain| domain.domain == host_ram)
            .ok_or(PlanError::ResourcePlanning("a host tier's fit reads no host RAM domain".into()))?
            .remaining_bytes;
        Ok(match fit.verdict {
            super::AssessmentFitVerdict::Fits => host.max(0),
            super::AssessmentFitVerdict::DoesNotFit { deficit_bytes, .. } => {
                -i64::try_from(deficit_bytes.get()).map_err(|_| PlanError::Arithmetic("a host tier's deficit exceeds i64"))?
            }
        })
    }

    /// Why the draft's experts cannot stream beside its linked host tier
    /// charged `charge` (its tier at its floor): host memory admits not
    /// even the tier's landing reserve. None where it does, or without a
    /// linked tier. Every file miss lands in a tier record, so the floor is
    /// what makes a step live, and a plan short of it is unstreamed, not
    /// short of device memory.
    fn untiered(&self, charge: &AssessmentMemoryCharge) -> Result<Option<UnstreamedExperts>, PlanError> {
        let tiered = self.bank().host_ceiling().is_some_and(|ceiling| ceiling > 0) && !self.bank().whole();
        if !matches!(self.device.host, Ok(Some(HostTier::Linked(_)))) || !tiered {
            return Ok(None);
        }
        Ok((self.host_left(charge, 0)? < 0).then(|| self.untiered_reason()))
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

    /// The draft of an elastic load, with its charge: its experts placed
    /// whole where they fit ([`Self::placed`]), else streamed at the pass
    /// the plan's `prices` settle ([`Unplaced::streamed`]). A load no plan
    /// holds is refused with why.
    pub fn with_bank<C: HeldCharge>(
        self,
        definition: &ModelDefinition,
        bound: Option<WeightBound>,
        prices: &Priced<super::PlanPrices>,
    ) -> Result<ChargedDraft<C>, PlanError> {
        match self.placed(definition, bound)? {
            Placed::Whole(whole) => Ok(whole),
            Placed::Short(short) => short.streamed(definition, prices),
        }
    }

    /// A load's first stage, its placement: its experts whole, in a whole
    /// bank or as their own allocations, wherever the stable fit's room
    /// ([`StableRoom`]) holds them under the load's own `bound`
    /// ([`Self::place`]), decided from memory alone; or what streaming them
    /// starts from ([`Unplaced`]). The figures it decided on are reported
    /// (`the load's experts are placed`).
    pub fn placed<C: HeldCharge>(self, definition: &ModelDefinition, bound: Option<WeightBound>) -> Result<Placed<C>, PlanError> {
        let mut held = HeldCharges::default();
        let capacity = self.device.assessment_capacity_bytes;
        let room = StableRoom::of(&self.device);
        let Placement { figures, outcome } =
            self.place(definition, HeldBound::of(bound, capacity), &mut held, &room, &mut load_confirm(room, bound))?;
        figures.report();
        Ok(match outcome {
            Placing::Whole { draft, charge, found: () } => Placed::Whole(ChargedDraft { draft, charge }),
            Placing::Short(ceiling) => Placed::Short(Unplaced { draft: self, ceiling, bound, held }),
            Placing::None(no) => return Err(refusal(no, capacity)),
        })
    }

    /// A load's experts streamed, whatever whole form the stable room
    /// holds them in: what a load whose import claim refuses its whole plan
    /// streams from ([`Unplaced::streamed`]), its bank then fitted by that
    /// claim. A load whose experts cannot stream is refused with why.
    pub fn streaming<C: HeldCharge>(self, definition: &ModelDefinition, bound: Option<WeightBound>) -> Result<Unplaced<C>, PlanError> {
        let capacity = self.device.assessment_capacity_bytes;
        match self.bank_ceiling(definition, HeldBound::of(bound, capacity))? {
            Ok(ceiling) => Ok(Unplaced { draft: self, ceiling, bound, held: HeldCharges::default() }),
            Err(no) => Err(refusal(no, capacity)),
        }
    }

    /// How a load holds its experts under `bound` ([`HeldBound`]), the one
    /// decision a load and its assessment share, in two stages: placed
    /// whole where `room` holds them so ([`Self::place`]), from memory
    /// alone; else streamed, in the bank and host tier `room` holds beside
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
        room: impl SettleRoom,
        mut confirm: impl FnMut(&Self, &AssessmentMemoryCharge) -> Result<Result<T, NonZeroU64>, E>,
    ) -> Result<Result<(ChargedDraft, T), NoBank>, E> {
        let mut held = HeldCharges::<AssessmentMemoryCharge>::default();
        let ceiling = match self.place(definition, bound, &mut held, &room, &mut confirm)?.outcome {
            Placing::Whole { draft, charge, found } => return Ok(Ok((ChargedDraft { draft, charge }, found))),
            Placing::Short(ceiling) => ceiling,
            Placing::None(no) => return Ok(Err(no)),
        };
        Ok(Self::streamed_in(ceiling, bound, definition, prices, &mut held, &room, &mut confirm)?
            .map(|(draft, charge, found)| (ChargedDraft { draft, charge }, found)))
    }

    /// The first stage of every holding of the experts, a load's and its
    /// assessment's alike: whole, from memory alone, reading no price.
    /// Experts that can stream are held in a bank of every expert, the
    /// largest pass first, where the load's own bound holds every slot
    /// ([`Self::bank_ceiling`]) and `room` holds the bank as admission fits
    /// it ([`SettleRoom::fit_whole`]); else, as their own allocations
    /// ([`Self::held_whole`]), where the bound and `room` hold those; else
    /// they are short of a whole plan and stream ([`Placing::Short`]).
    /// Experts that cannot stream are held as their own allocations within
    /// the bound, whether the device holds them `confirm`'s. A banked
    /// draft whose growth stops short of every expert keeps its bank, as
    /// its ceiling never rises. Each whole plan is confirmed by `confirm`.
    fn place<C: HeldCharge, T, E: From<PlanError>>(
        &self,
        definition: &ModelDefinition,
        bound: HeldBound,
        held: &mut HeldCharges<C>,
        room: &impl SettleRoom,
        confirm: &mut impl FnMut(&Self, &AssessmentMemoryCharge) -> Result<Result<T, NonZeroU64>, E>,
    ) -> Result<Placement<C, T>, E> {
        let mut figures = PlacementFigures { pricing: self.device.graph_pricing, form: None, bank: None, resident: None };
        let placed = |figures, outcome| Ok(Placement { figures, outcome });
        let ceiling = match self.bank_ceiling(definition, bound)? {
            Ok(ceiling) => ceiling,
            Err(NoBank::Unstreamed(reason)) => {
                let Some(chains) = self.whole_chains(definition, bound)? else {
                    return placed(figures, Placing::None(NoBank::Unstreamed(reason)));
                };
                // The longest chain the room holds, else the shortest,
                // each as `confirm` decides it.
                let last = chains.len() - 1;
                for (at, places) in chains.into_iter().enumerate() {
                    let resident = self.held_whole_at(definition, places)?;
                    let charge = held.charge(definition, &resident)?;
                    let fit = room.fit_whole(charge.charge())?;
                    figures.resident = Some(Need { fit: Some(fit), terms: NeedTerms::of(&resident, charge.charge())? });
                    if fit.short.is_none() || at == last {
                        if let Ok(found) = confirm(&resident, charge.charge())? {
                            figures.form = Some(WholeForm::Resident);
                            return placed(figures, Placing::Whole { draft: resident, charge, found });
                        }
                    }
                }
                return placed(figures, Placing::None(NoBank::Unstreamed(reason)));
            }
            Err(no) => return placed(figures, Placing::None(no)),
        };
        if self.load.bank.as_ref().is_some_and(|bank| !bank.reaches_whole()) {
            return placed(figures, Placing::Short(ceiling));
        }
        let experts = ceiling.whole.bank().slots;
        if ceiling.ceiling == experts {
            for &pass in super::pass_ladder(ceiling.places).iter().rev() {
                let draft = ceiling.whole.resized(experts, experts, pass)?;
                let charge = held.charge(definition, &draft)?;
                let fit = room.fit_whole(charge.charge())?;
                figures.bank = Some(Need { fit: Some(fit), terms: NeedTerms::of(&draft, charge.charge())? });
                if fit.short.is_none() {
                    if let Ok(found) = confirm(&draft, charge.charge())? {
                        figures.form = Some(WholeForm::Bank);
                        return placed(figures, Placing::Whole { draft, charge, found });
                    }
                }
            }
        }
        for places in self.whole_chains(definition, bound)?.into_iter().flatten() {
            let resident = self.held_whole_at(definition, places)?;
            let charge = held.charge(definition, &resident)?;
            let fit = room.fit_whole(charge.charge())?;
            figures.resident = Some(Need { fit: Some(fit), terms: NeedTerms::of(&resident, charge.charge())? });
            if fit.short.is_none() {
                if let Ok(found) = confirm(&resident, charge.charge())? {
                    figures.form = Some(WholeForm::Resident);
                    return placed(figures, Placing::Whole { draft: resident, charge, found });
                }
            }
        }
        placed(figures, Placing::Short(ceiling))
    }

    /// The second stage, for experts no whole plan holds: the bank the room
    /// holds beside the arena from `ceiling`, at the pass `prices` settle
    /// ([`Self::settle_bank`]), and
    /// its host tier ([`Self::with_host_tier`]),
    /// whose size leaves the charge as it is: the charge holds the tier at
    /// its floor. Experts the host cannot stream beside their tier are held
    /// as their own allocations within `bound`, whether the device holds
    /// them `confirm`'s, as experts that cannot stream are
    /// ([`Self::place`]).
    fn streamed_in<C: HeldCharge, T, E: From<PlanError>>(
        ceiling: BankCeiling,
        bound: HeldBound,
        definition: &ModelDefinition,
        prices: &Priced<super::PlanPrices>,
        held: &mut HeldCharges<C>,
        room: &impl SettleRoom,
        confirm: &mut impl FnMut(&Self, &AssessmentMemoryCharge) -> Result<Result<T, NonZeroU64>, E>,
    ) -> Result<Result<(Self, C, T), NoBank>, E> {
        let reason = match Self::settle_bank(&ceiling, definition, prices, held, room, confirm)? {
            Ok((draft, charge, found)) => match draft.with_host_tier(charge.charge())? {
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
    fn bank_ceiling(&self, definition: &ModelDefinition, bound: HeldBound) -> Result<Result<BankCeiling, NoBank>, PlanError> {
        let substrate = match bank_substrate(
            self.device.backend,
            self.device.reservation.as_ref().copied().map_err(String::as_str),
            self.device.host_tier(),
            definition,
        ) {
            Ok(substrate) => substrate,
            Err(error) => return unstreamed(error),
        };
        let places = self.prompt_places(definition);
        let whole = match super::bank::routed_experts(definition).and_then(|experts| {
            let load = self.load.clone().unbanked().with_expert_bank(
                definition,
                experts,
                places,
                self.policy.limits.flights(),
                substrate,
            )?;
            let programs = load.program_plan(definition, self.policy.codec)?;
            Self { load, programs, ..self.clone() }.with_chain(places)
        }) {
            Ok(whole) => whole,
            Err(error) => return unstreamed(error),
        };
        let bank = whole.bank();
        let (experts, floor) = (bank.slots, bank.floor());
        // A banked draft's re-planning never raises its ceiling.
        let top = self.load.bank.as_ref().map_or(experts, |bank| bank.ceiling);
        let reserved = |slots| bank.resident_charge(slots).map_err(PlanError::ResourcePlanning);
        let most = bank.most_slots(self.device.max_allocation_bytes).map_err(PlanError::ResourcePlanning)?;
        if most < floor {
            return Ok(Err(NoBank::Unstreamed(UnstreamedExperts::Allocation { floor, most })));
        }
        // The weights of the bank at `slots`.
        let (whole_bytes, whole_charge) = (weight_bytes(&whole.load)?, reserved(experts)?);
        let weights_at = |slots| -> Result<u64, PlanError> {
            let reserved = reserved(slots)?;
            whole_bytes
                .checked_sub(whole_charge)
                .and_then(|bytes| bytes.checked_add(reserved))
                .ok_or(PlanError::Arithmetic("a bank's weights hold less than its record reservations' charge"))
        };
        // The load's own bound (a cap or a device room) below the floor's
        // weights holds no bank. The machine's room below them refuses
        // nothing here: the plan's settle finds it short.
        let needed = weights_at(floor)?;
        if let Some(limit) = bound.limit.filter(|limit| needed > limit.bytes()) {
            return Ok(Err(NoBank::Bound { bound: limit, needed }));
        }
        // The ceiling: the most slots whose weights the device holds under
        // `bound`, where the bank's curve ends, the floor at least: the bank
        // grows to it as memory frees, whatever the room at the load
        // (`settle_bank`).
        let device = bound.device();
        let mut ceiling = floor;
        while ceiling < top.min(most) && weights_at(ceiling + 1)? <= device {
            ceiling += 1;
        }
        // A cap bounds every weight byte the load holds: a host tier holds
        // what it leaves beside the bank at its ceiling, its landing
        // reserve (the tier at no slots) aside, as staging is. A banked
        // draft's re-planning never raises the cap it was planned under.
        let capped = match (bound, bank.host_ceiling()) {
            (HeldBound { limit: Some(WeightBound::Cap(cap)), .. }, Some(_)) => {
                let left = cap - weights_at(ceiling)?;
                let host = |slots: u64| bank.host_charge(slots).map_err(PlanError::ResourcePlanning);
                let (empty, widest) = (host(0)?, bank.widest().get());
                let mut cap = 0;
                while cap < widest && host(cap + 1)? - empty <= left {
                    cap += 1;
                }
                Some(cap)
            }
            _ => None,
        };
        let planned = self.load.bank.as_ref().and_then(|bank| bank.host_cap);
        let host_cap = match (capped, planned) {
            (Some(capped), Some(planned)) => Some(capped.min(planned)),
            (capped, planned) => capped.or(planned),
        };
        let mut whole = whole;
        whole.bank_mut().host_cap = host_cap;
        let places = usize::try_from(places)
            .ok()
            .and_then(NonZeroUsize::new)
            .map(PassLaunches::new)
            .ok_or(PlanError::Arithmetic("a group's places exceed usize"))?;
        Ok(Ok(BankCeiling { whole, ceiling, places }))
    }

    /// The draft's bank (of every expert, [`Self::bank_ceiling`]) at
    /// `slots` under `ceiling`, its passes of `pass` launches.
    fn resized(&self, slots: u64, ceiling: u64, pass: PassLaunches) -> Result<Self, PlanError> {
        let mut draft = Self { load: self.load.clone().with_bank_slots(slots, ceiling)?, ..self.clone() };
        draft.bank_mut().pass = pass;
        Ok(draft)
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
        room: &impl SettleRoom,
        confirm: &mut impl FnMut(&Self, &AssessmentMemoryCharge) -> Result<Result<T, NonZeroU64>, E>,
    ) -> Result<Result<(Self, C, T), NoBank>, E> {
        let BankCeiling { whole, ceiling, places } = bank_ceiling;
        let bank = whole.bank();
        let reserved = |slots| bank.resident_charge(slots).map_err(PlanError::ResourcePlanning);
        let points = rising(bank.floor(), *ceiling, &reserved)?;
        // The structure of the mandatory part at the smallest pass, certified
        // once: each pass size's charge is it with the target family
        // certified at that pass (`StructuralCharge::at_pass`), the one term
        // a pass moves.
        let mandatory = whole.resized(points[0], *ceiling, PassLaunches::FLOOR)?;
        let structure = crate::assessment::StructuralCharge::of(definition, &mandatory).map_err(charge_error)?;
        let floor_charge = structure.charge(definition, &mandatory).map_err(charge_error)?;
        if let Some(reason) = mandatory.untiered(&floor_charge)? {
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
        // The room beside the smallest pass's mandatory part, which a larger
        // pass's mandatory part must fit in.
        let floor_bytes = Self::allocated(&floor_charge)?;
        let floor_room = room.room(&floor_charge, floor_bytes)?;
        let fits_within = floor_bytes.saturating_add(floor_room.free.min(floor_room.ceiling));
        let mut most = None;
        let mut chosen: Option<(PassLaunches, u64, f64)> = None;
        for pass in super::pass_ladder(*places) {
            let mandatory = whole.resized(points[0], *ceiling, pass)?;
            let charge = match pass == PassLaunches::FLOOR {
                true => floor_charge.clone(),
                false => structure
                    .at_pass(definition, &mandatory)
                    .and_then(|structure| structure.charge(definition, &mandatory))
                    .map_err(charge_error)?,
            };
            let mandatory_bytes = Self::allocated(&charge)?;
            // A larger pass leaves no more room.
            if mandatory_bytes > fits_within {
                break;
            }
            let free = room.room(&charge, mandatory_bytes)?;
            let slots = count(slots_within(free.free.min(free.ceiling), (points[0], *ceiling), &reserved)?)?;
            // The smallest pass leaves the most slots.
            let most = *most.get_or_insert(crate::assessment::HeldSlots { slots, host });
            let seconds = rates.decode_steps_per_s * prices.step_seconds(crate::assessment::HeldSlots::bank(slots))
                + rates.requests_per_s * prices.prompt_seconds(pass, most);
            if chosen.as_ref().is_none_or(|(_, _, least)| seconds <= *least) {
                chosen = Some((pass, mandatory_bytes, seconds));
            }
        }
        let (pass, priced, _) = chosen.expect("invariant violated: the ladder holds the smallest pass");
        Self::settle_bank_at(bank_ceiling, pass, priced, definition, held, room, confirm)
    }

    /// The draft's bank (`bank_ceiling`) with passes of `pass` launches
    /// beside its arena, its mandatory part `priced` bytes as
    /// [`Self::settle_bank`] priced it, with its charge and what `confirm`
    /// found of it; or why no bank's does.
    ///
    /// Each plan is the bank's structure resized ([`Self::bank_ceiling`]).
    /// The arena, holding a group of one place (a group's class beyond it
    /// is claimed as the group opens), the pass's buffers and everything
    /// else the plan commits are its mandatory part ([`Self::allocated`]);
    /// the bank is what `room` holds beside it
    /// at the load: the most slots that fit, its floor at least. Its
    /// ceiling is structural, never the room: the bank's ceiling
    /// ([`Self::bank_ceiling`]), which the bank's target, read at every
    /// reading, grows to as memory frees (`experts::derived::target_slots`,
    /// which stops at the miss curve's saturation where that is smaller).
    /// Where `confirm` finds a plan short, the bank gives up the bytes it
    /// is short; short at its floor, the plan is short.
    fn settle_bank_at<C: HeldCharge, T, E: From<PlanError>>(
        bank_ceiling: &BankCeiling,
        pass: PassLaunches,
        priced: u64,
        definition: &ModelDefinition,
        held: &mut HeldCharges<C>,
        room: &impl SettleRoom,
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
        if let Some(reason) = mandatory.untiered(&floor_charge)? {
            return Ok(Err(NoBank::Unstreamed(reason)));
        }
        let mandatory_bytes = Self::allocated(&floor_charge)?;
        assert_eq!(mandatory_bytes, priced, "invariant violated: a pass of {pass:?} certifies to other bytes than it is priced at");
        let room = room.room(&floor_charge, mandatory_bytes)?;
        let room = room.free.min(room.ceiling);
        // A draft of `slots` under `ceiling`, its charge, and what it leaves
        // of the room (`Ok`) or how far it exceeds it (`Err`).
        let charged = |held: &mut HeldCharges<C>, slots: u64, ceiling: u64| -> Result<_, E> {
            let draft = whole.resized(slots, ceiling, pass)?;
            let charge = held.charge(definition, &draft)?;
            let grown = Self::allocated(charge.charge())?.saturating_sub(mandatory_bytes);
            let left = match room.checked_sub(grown) {
                Some(left) => Ok(left),
                None => Err(grown - room),
            };
            Ok((draft, charge, left))
        };
        // The most slots whose reservation the room holds, each found short
        // by its real charge giving up a step.
        let most = slots_within(room, (points[0], ceiling), &reserved)?;
        let mut at = points
            .iter()
            .position(|&slots| slots == most)
            .expect("invariant violated: the most slots a room holds end a charge level");
        loop {
            let slots = points[at];
            let (draft, charge) = match charged(held, slots, ceiling)? {
                (draft, charge, Ok(_)) => (draft, charge),
                (_, _, Err(excess)) if at == 0 => {
                    return Ok(Err(NoBank::Short { deficit_bytes: NonZeroU64::new(excess).expect("a nonzero excess") }));
                }
                (_, _, Err(_)) => {
                    at -= 1;
                    continue;
                }
            };
            let deficit = match confirm(&draft, charge.charge())? {
                Ok(found) => return Ok(Ok((draft, charge, found))),
                Err(deficit) if at == 0 => return Ok(Err(NoBank::Short { deficit_bytes: deficit })),
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
        let bytes = weight_bytes(&self.load.clone().unbanked())?;
        if bound.limit.is_some_and(|limit| bytes > limit.bytes()) {
            return Ok(None);
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
        let load = self.load.clone().unbanked();
        let programs = load.program_plan(definition, self.policy.codec)?;
        Self { load, programs, ..self.clone() }.with_chain(places)
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
                let mut load = self
                    .load
                    .clone()
                    .unbanked()
                    .with_expert_bank(definition, bank.slots, bank.places, bank.flights, bank.substrate())?
                    .with_bank_slots(bank.slots, bank.ceiling)?;
                let built = load.bank.as_mut().expect("the load was just banked");
                (built.host_slots, built.pass, built.resident) = (bank.host_slots, bank.pass, bank.resident);
                let programs = load.program_plan(definition, self.policy.codec)?;
                Self { load, programs, ..self.clone() }.with_chain(bank.places)?
            }
        };
        assert_eq!(draft.load.bank, held.load.bank, "a plain plan's bank differs from its drafted plan's");
        assert_eq!(
            draft.policy.limits, held.policy.limits,
            "a plain plan's limits differ from its drafted plan's"
        );
        Ok(draft)
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

    /// The draft committing `slots` of its expert bank at the load, its
    /// ceiling kept ([`ModelLoadPlan::with_load_slots`]): what the target
    /// import's claim admits of the slots planned.
    pub fn with_load_slots(self, slots: u64) -> Result<Self, PlanError> {
        Ok(Self { load: self.load.with_load_slots(slots)?, ..self })
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
            resources,
            capabilities: self.capabilities,
            policy: self.policy,
        })
    }
}

impl ExecutionPlan {
    /// The plan committing `slots` of its expert bank at the load, its
    /// ceiling kept ([`ModelLoadPlan::with_load_slots`]).
    pub fn with_load_slots(self, slots: u64) -> Result<Self, PlanError> {
        Ok(Self { load: self.load.with_load_slots(slots)?, ..self })
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
    /// A bank's, by the pass size it was certified at and whether it holds
    /// a resident form: a pass sizes the target family's pass buffers and
    /// graphs, and a bank planned whole has a resident form
    /// (`ExpertBankPlan::resident`), so a bank of another is charged from
    /// its structure again.
    banked: Option<((PassLaunches, bool), C)>,
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
        let structure = (bank.pass, bank.resident);
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
    /// The plan at the bank's floor is still `deficit_bytes` short.
    Short { deficit_bytes: NonZeroU64 },
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

/// The room a plan's settle grants into beside its mandatory part: what
/// releasable holdings may take (`free`), and what the device's stable fit
/// still allows (`ceiling`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanRoom {
    pub free: u64,
    pub ceiling: u64,
}

/// What a plan's settle grants into beside its mandatory part, of that
/// part's charge and its device-side bytes: a load's plan and its
/// assessment settle into the stable fit's ([`StableRoom`]); a test may give
/// a room of its own, as a function of the mandatory bytes.
pub trait SettleRoom {
    fn room(&self, mandatory: &AssessmentMemoryCharge, bytes: u64) -> Result<PlanRoom, PlanError>;
    /// The room's fit of a plan charged `charge` beside nothing else, as
    /// a whole plan is placed ([`ExecutionPlanDraft::place`]).
    fn fit_whole(&self, charge: &AssessmentMemoryCharge) -> Result<WholeFit, PlanError>;
}

/// A whole plan's fit ([`SettleRoom::fit_whole`]), as the one verdict that
/// decides it reads it: the room of the store its device storage is placed
/// on, what the plan requires there, and how far the plan is short and in
/// which role of store, if it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WholeFit {
    pub room: PlanRoom,
    pub needed: u64,
    pub short: Option<NonZeroU64>,
    pub limiting: Option<crate::platform::DomainRole>,
}

impl<F: Fn(u64) -> PlanRoom> SettleRoom for F {
    fn room(&self, _: &AssessmentMemoryCharge, bytes: u64) -> Result<PlanRoom, PlanError> {
        Ok(self(bytes))
    }

    /// The room beside nothing, against what the draft allocates as
    /// admission fits it.
    fn fit_whole(&self, charge: &AssessmentMemoryCharge) -> Result<WholeFit, PlanError> {
        let (room, needed) = (self(0), ExecutionPlanDraft::allocated(charge)?);
        let short = NonZeroU64::new(needed.saturating_sub(room.free.min(room.ceiling)));
        Ok(WholeFit { room, needed, short, limiting: short.map(|_| crate::platform::DomainRole::Allocation) })
    }
}

/// The stable fit's room, the one a load's plan and its assessment both
/// settle into: what the allocation domain's stable room
/// (`AssessmentFit::settle_bound`, the mandatory part's fit under the
/// device's reading) leaves beside the mandatory part, all of it free. What
/// the moment's reading leaves is the running load's memory pace's. Its
/// fit ([`Self::fit`]) is the one confirm of a plan against it.
#[derive(Clone, Copy)]
pub struct StableRoom<'a> {
    pub topology: &'a seismic::DeviceTopology,
    pub executor: &'a seismic::Executor,
    pub status: &'a seismic::Status,
    pub reserves: &'a crate::platform::MemoryReserves,
}

impl<'a> StableRoom<'a> {
    /// The room of `device` under the reading it was selected at.
    fn of(device: &'a PlannedDevice) -> Self {
        let FitReading { topology, status, reserves } = &device.reading;
        Self { topology, executor: &device.executor, status, reserves }
    }

    /// The stable fit of a draft charged `charge`, as the load commits it:
    /// its arena holds a group of one place, and a group's class beyond it
    /// is claimed as the group opens. The one confirm a load's plan and its
    /// assessment make.
    pub fn fit(&self, charge: &AssessmentMemoryCharge) -> Result<super::AssessmentFit, PlanError> {
        charge.assess_fit(self.topology, self.executor, self.status, self.reserves).map_err(PlanError::ResourcePlanning)
    }
}

impl SettleRoom for StableRoom<'_> {
    /// Admission's fit of the draft ([`Self::fit`]), in the allocation
    /// domain's stable room.
    fn fit_whole(&self, charge: &AssessmentMemoryCharge) -> Result<WholeFit, PlanError> {
        let fit = self.fit(charge)?;
        let allocation = fit
            .domains
            .iter()
            .find(|domain| domain.role == crate::platform::DomainRole::Allocation)
            .ok_or_else(|| PlanError::ResourcePlanning("the fit reaches no allocation domain".into()))?;
        let bound = allocation.capacity_bytes - allocation.reserve_bytes;
        let (short, limiting) = match fit.verdict {
            super::AssessmentFitVerdict::Fits => (None, None),
            super::AssessmentFitVerdict::DoesNotFit { limiting, deficit_bytes } => {
                (Some(deficit_bytes), fit.domains.iter().find(|domain| domain.domain == limiting).map(|domain| domain.role))
            }
        };
        Ok(WholeFit { room: PlanRoom { free: bound, ceiling: bound }, needed: allocation.required_bytes, short, limiting })
    }

    fn room(&self, mandatory: &AssessmentMemoryCharge, bytes: u64) -> Result<PlanRoom, PlanError> {
        let bound = mandatory
            .assess_fit(self.topology, self.executor, self.status, self.reserves)
            .and_then(|fit| fit.settle_bound())
            .map_err(PlanError::ResourcePlanning)?;
        let left = bound.saturating_sub(bytes);
        Ok(PlanRoom { free: left, ceiling: left })
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
    Short(BankCeiling),
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

/// A whole form's fit, where `room` judged it (none where `confirm` alone
/// did), and its charge's terms.
struct Need {
    fit: Option<WholeFit>,
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
    startup_additional: u64,
    component_upload: u64,
    allocation: u64,
}

impl NeedTerms {
    fn of(draft: &ExecutionPlanDraft, charge: &AssessmentMemoryCharge) -> Result<Self, PlanError> {
        let bounds = &charge.bounds;
        let allocation = ExecutionPlanDraft::allocated(charge)?;
        let resource = |result: Result<u64, String>| result.map_err(PlanError::ResourcePlanning);
        Ok(Self {
            weights: charge.weight_demand.checked_bytes().ok_or(PlanError::Arithmetic("weight demand overflows"))?,
            weights_device: charge.weight_demand.device.saturating_add(charge.weight_demand.reserved),
            state: charge.state_bytes,
            startup_state: charge.startup_state_bytes,
            graph_pools: bounds.graph_bytes,
            chain_places: draft.policy.limits.chain_capacity.get(),
            recorded_graphs_device: bounds.recorded_graph_bytes - bounds.recorded_graph_upload_bytes,
            recorded_graphs_upload: bounds.recorded_graph_upload_bytes,
            prepared: resource(bounds.prepared_resource_bytes())?,
            process: resource(bounds.device_process.bytes())?,
            startup_additional: bounds.startup_additional_bytes,
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
            startup_additional,
            component_upload,
            allocation,
        } = self;
        write!(
            formatter,
            "weights {weights} ({weights_device} in device storage), state {state} (startup {startup_state}), \
             graph pools {graph_pools} at {chain_places} chain places, \
             recorded graphs {recorded_graphs_device} device and {recorded_graphs_upload} upload, prepared {prepared}, process {process}, \
             startup import {startup_additional}, component import {component_upload}, allocation {allocation}"
        )
    }
}

impl PlacementFigures {
    /// One event of a load's placement.
    fn report(&self) {
        let fit = |need: &Option<Need>| need.as_ref().and_then(|need| need.fit);
        let room = fit(&self.bank).or(fit(&self.resident)).map(|fit| fit.room);
        tracing::info!(
            room_free = ?room.map(|room| room.free),
            room_ceiling = ?room.map(|room| room.ceiling),
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
}

impl<C: HeldCharge> Unplaced<C> {
    /// The load's plan before its bank.
    pub fn draft(&self) -> &ExecutionPlanDraft {
        &self.draft
    }

    /// The load's second stage: its experts streamed through the bank and
    /// host tier the stable fit's room holds beside its mandatory part, each
    /// plan confirmed as its assessment confirms it, at the pass the
    /// machine's `prices` settle. A load no plan holds is refused with why.
    pub fn streamed(self, definition: &ModelDefinition, prices: &Priced<super::PlanPrices>) -> Result<ChargedDraft<C>, PlanError> {
        let Self { draft, ceiling, bound, mut held } = self;
        let capacity = draft.device.assessment_capacity_bytes;
        let room = StableRoom::of(&draft.device);
        let streamed = ExecutionPlanDraft::streamed_in(
            ceiling,
            HeldBound::of(bound, capacity),
            definition,
            prices,
            &mut held,
            &room,
            &mut load_confirm(room, bound),
        )?;
        match streamed {
            Ok((draft, charge, ())) => Ok(ChargedDraft { draft, charge }),
            Err(no) => Err(refusal(no, capacity)),
        }
    }
}

/// A load's confirm of each plan, as its assessment confirms it
/// ([`StableRoom::fit`]), under its own `bound`: an unbounded load's experts
/// held whole are admitted here, whether the device holds them what
/// placement, assessment and admission find.
fn load_confirm<'a>(
    room: StableRoom<'a>,
    bound: Option<WeightBound>,
) -> impl FnMut(&ExecutionPlanDraft, &AssessmentMemoryCharge) -> Result<Result<(), NonZeroU64>, PlanError> + 'a {
    move |draft: &ExecutionPlanDraft, charge: &AssessmentMemoryCharge| {
        if bound.is_none() && draft.load.bank.is_none() {
            return Ok(Ok(()));
        }
        Ok(match room.fit(charge)?.verdict {
            super::AssessmentFitVerdict::Fits => Ok(()),
            super::AssessmentFitVerdict::DoesNotFit { deficit_bytes, .. } => Err(deficit_bytes),
        })
    }
}

/// A load's refusal, of a device of `capacity` bytes, where no plan holds
/// its experts.
fn refusal(no: NoBank, capacity: u64) -> PlanError {
    match no {
        NoBank::Unstreamed(reason) => PlanError::Unstreamed(reason),
        NoBank::Bound { bound, needed } => PlanError::WeightBound { bound, needed },
        NoBank::Short { deficit_bytes } => PlanError::Resource(crate::CapacityError {
            resource: crate::ResourceKind::DeviceMemory,
            required: capacity + deficit_bytes.get(),
            available: capacity,
        }),
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
pub enum FloorForms<T = ()> {
    /// The experts streamed through a bank of every expert at the floor
    /// pass, at the most slots the load's own bound holds, and held whole
    /// as their own allocations at groups of one launch where that bound
    /// holds them so.
    Banked { bank: BankForm<T>, resident: Option<HeldForm<T>> },
    /// Experts that cannot stream, held whole as their own allocations.
    Resident(HeldForm<T>),
}

/// One way of holding a draft's experts and its charge's structure.
pub struct HeldForm<T = ()> {
    pub draft: ExecutionPlanDraft,
    pub structure: crate::assessment::StructuralCharge,
    pub more: T,
}

/// A bank form ([`FloorForms::Banked`]): its draft at its ceiling, whose
/// charge at any slots is its structure's charge of [`Self::at_slots`].
pub struct BankForm<T = ()> {
    pub form: HeldForm<T>,
    /// The bank's experts: a bank of all of them holds them whole.
    pub experts: u64,
    /// The sizes the bank's charge steps through, its floor first and its
    /// ceiling last.
    pub points: Vec<u64>,
}

impl<T> BankForm<T> {
    /// The form's draft with `slots` slots, a bank of at most that many.
    pub fn at_slots(&self, slots: u64) -> Result<ExecutionPlanDraft, PlanError> {
        self.form.draft.resized(slots, slots, PassLaunches::FLOOR)
    }

    /// The bank at `slots` with its linked host tier as host memory beside
    /// `charge` holds it (`ExecutionPlanDraft::with_host_tier`, sized by
    /// memory alone), or why its experts cannot stream beside it.
    pub fn with_host_tier(&self, slots: u64, charge: &AssessmentMemoryCharge) -> Result<Result<ExecutionPlanDraft, UnstreamedExperts>, PlanError> {
        self.at_slots(slots)?.with_host_tier(charge)
    }
}

impl<T> FloorForms<T> {
    /// The forms without what else their outlines gave.
    pub fn structures(self) -> FloorForms {
        let form = |HeldForm { draft, structure, .. }: HeldForm<T>| HeldForm { draft, structure, more: () };
        match self {
            Self::Banked { bank: BankForm { form: held, experts, points }, resident } => {
                FloorForms::Banked { bank: BankForm { form: form(held), experts, points }, resident: resident.map(form) }
            }
            Self::Resident(resident) => FloorForms::Resident(form(resident)),
        }
    }

    /// The form the plan's structure and demand are read from: its bank,
    /// else its experts whole.
    pub fn primary(&self) -> &HeldForm<T> {
        match self {
            Self::Banked { bank, .. } => &bank.form,
            Self::Resident(resident) => resident,
        }
    }

    /// The bank form, where the experts stream.
    pub fn bank(&self) -> Option<&BankForm<T>> {
        match self {
            Self::Banked { bank, .. } => Some(bank),
            Self::Resident(_) => None,
        }
    }

    /// The experts held whole as their own allocations, where the load's
    /// own bound holds them so.
    pub fn resident(&self) -> Option<&HeldForm<T>> {
        match self {
            Self::Banked { resident, .. } => resident.as_ref(),
            Self::Resident(resident) => Some(resident),
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

    /// The draft's [`FloorForms`] under the load's own `limit`, each form's
    /// structure charged by `structure`; or why no form holds its experts,
    /// as a load's placement refuses them.
    pub fn floor_forms<T, E: From<PlanError>>(
        &self,
        definition: &ModelDefinition,
        limit: Option<WeightBound>,
        structure: impl Fn(&Self) -> Result<(crate::assessment::StructuralCharge, T), E>,
    ) -> Result<FloorForms<T>, E> {
        let bound = HeldBound::of(limit, self.device.assessment_capacity_bytes);
        let form = |draft: Self| -> Result<HeldForm<T>, E> {
            let (structure, more) = structure(&draft)?;
            Ok(HeldForm { draft, structure, more })
        };
        let bank = match self.bank_ceiling(definition, bound)? {
            Ok(BankCeiling { whole, ceiling, .. }) => {
                let bank = whole.bank();
                let reserved = |slots| bank.resident_charge(slots).map_err(PlanError::ResourcePlanning);
                let points = rising(bank.floor(), ceiling, &reserved)?;
                let experts = bank.slots;
                let draft = whole.resized(ceiling, ceiling, PassLaunches::FLOOR)?;
                Ok(BankForm { form: form(draft)?, experts, points })
            }
            Err(NoBank::Bound { bound, needed }) => return Err(PlanError::WeightBound { bound, needed }.into()),
            Err(NoBank::Unstreamed(reason)) => Err(reason),
            Err(no @ NoBank::Short { .. }) => unreachable!("a bank's ceiling asks no room, so it is never {no:?}"),
        };
        let resident = match self.whole_chains(definition, bound)? {
            Some(_) => Some(form(self.held_whole_at(definition, 1)?)?),
            None => None,
        };
        match (bank, resident) {
            (Ok(bank), resident) => Ok(FloorForms::Banked { bank, resident }),
            (Err(_), Some(resident)) => Ok(FloorForms::Resident(resident)),
            (Err(reason), None) => Err(PlanError::Unstreamed(reason).into()),
        }
    }
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

/// The sizes of `from..=to` a holding steps through: the most of each
/// `charge` level, `from`'s first and `to` last. A size whose charge is
/// its predecessor's costs nothing more, so it is held with it.
fn rising(from: u64, to: u64, charge: &impl Fn(u64) -> Result<u64, PlanError>) -> Result<Vec<u64>, PlanError> {
    let mut points = Vec::new();
    let mut level = charge(from)?;
    for size in from + 1..=to {
        let next = charge(size)?;
        if next > level {
            points.push(size - 1);
            level = next;
        }
    }
    points.push(to);
    Ok(points)
}

fn weight_bytes(load: &ModelLoadPlan) -> Result<u64, PlanError> {
    load.weight_bytes().map_err(PlanError::ResourcePlanning)
}

/// What a bank of `definition`'s experts streams on, of a device of
/// `backend` reserving `reservation` and holding its host tier `host`, as
/// each reads, or why they cannot stream: the backend's entries
/// ([`streams_experts`]), the device's reserved records, and a host tier,
/// which every miss lands in, so a bank streams only beside one.
pub(crate) fn bank_substrate(
    backend: BackendName,
    reservation: Result<Option<seismic::Reservation>, &str>,
    host: Result<Option<HostTier>, &str>,
    definition: &ModelDefinition,
) -> Result<super::BankSubstrate, PlanError> {
    streams_experts(backend, definition)?;
    let reservation = match reservation {
        Ok(Some(reservation)) => reservation,
        Ok(None) => return Err(PlanError::Unstreamed(UnstreamedExperts::Unreserved(backend))),
        Err(reason) => {
            let reason = reason.to_owned();
            return Err(PlanError::Unstreamed(UnstreamedExperts::Unreservable { backend, reason }));
        }
    };
    let reason = match host {
        Ok(Some(host)) => return Ok(super::BankSubstrate { backend, reservation, host }),
        Ok(None) => "its backend reserves no host storage beside its device's own",
        Err(reason) => reason,
    };
    Err(PlanError::Unstreamed(UnstreamedExperts::Untiered { backend, reason: reason.to_owned() }))
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
    let forms = definition
        .decoder
        .sublayers()
        .filter_map(|(_, sublayer)| routed_ffn(&sublayer.op))
        .map(|routed| match is_fused(routed) {
            true => BankRouteForm::Fused,
            false => BankRouteForm::General,
        })
        .collect::<std::collections::HashSet<_>>();
    for form in forms {
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
            device: PlannedDevice {
                selector: device.info.selector(),
                backend,
                name: device.info.name().to_owned(),
                assessment_capacity_bytes: device.assessment_capacity_bytes,
                tensor_operations: device.info.forms_tensor_operations(),
                max_allocation_bytes: device.info.limits().max_allocation.bytes(),
                reservation: device
                    .info
                    .reserving(match &device.host_tier {
                        Ok(Some(tier)) => tier.contract(crate::planning::BankTier::Bank),
                        _ => seismic::ReservedContract::DeviceOnly,
                    })
                    .map_err(ToString::to_string),
                host: device.host_tier.clone(),
                executor: device.info.clone(),
                reading: device.reading.clone(),
                graph_pricing: device.graph_cost,
            },
            components,
            load,
            programs,
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
        let load = ModelLoadPlan::derive(manifest, definition, selection, layout, method, limits)
            .map_err(PlanError::InvalidDefinition)?;
        // Native entries import the weights, through record ports.
        let load = match path {
            ExecutionPath::Native => load.with_record_ports(backend),
            ExecutionPath::Planned => load,
        };
        // The head in progressive planes, where the backend reads them.
        let load = if path == ExecutionPath::Native
            && crate::programs::graph::readout::reads_progressive_heads(backend)
                .map_err(PlanError::ResourcePlanning)?
        {
            load.with_readout_candidates(definition)
                .map_err(PlanError::InvalidDefinition)?
        } else {
            load
        };
        let programs = load.program_plan(definition, codec)?;
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
            // by the draft's block, not by sealed chains.
            PlannedMethod::DFlash { proposals, .. } => {
                let block = definition.draft.as_ref().map_or(
                    0,
                    magnitude_family_contracts::DraftDefinition::max_proposals,
                );
                if proposals == 0 || u64::from(proposals) > block || method.widest() > proposals {
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

    /// The plan's prices on the synthetic machine's measurements.
    fn synthetic() -> Priced<super::super::PlanPrices> {
        Priced::Measured(crate::assessment::synthetic::plan_prices())
    }

    /// The plan's prices before the machine's costs are measured.
    fn pending() -> Priced<super::super::PlanPrices> {
        Priced::Pending("the executor's costs are not measured yet".into())
    }

    /// A holding steps where its charge rises, each step holding the most
    /// sizes of its charge: a size that costs nothing more comes free.
    #[test]
    fn a_holding_steps_only_where_its_charge_rises() {
        let charge = |slots: u64| Ok::<_, PlanError>(slots.div_ceil(4) * 100);
        assert_eq!(rising(2, 13, &charge).unwrap(), vec![4, 8, 12, 13]);
        assert_eq!(rising(5, 5, &charge).unwrap(), vec![5]);
    }

    /// The recorded RTX 3090 with `host` bytes of host RAM free, CUDA's
    /// pooled graph memory as it measured it.
    fn rtx3090(host: u64) -> (seismic::DeviceTopology, seismic::Status, crate::platform::MemoryReserves, crate::platform::SelectedDevice) {
        recorded(seismic::fixtures::recorded::rtx3090(), BackendName::Cuda, host, RTX3090_CUDA_GRAPHS)
    }

    /// `topology`'s `backend` executor with `host` bytes of host RAM free,
    /// its graphs costing `cost`.
    fn recorded(
        topology: seismic::DeviceTopology,
        backend: BackendName,
        host: u64,
        cost: seismic::GraphCost,
    ) -> (seismic::DeviceTopology, seismic::Status, crate::platform::MemoryReserves, crate::platform::SelectedDevice) {
        use crate::platform::{GraphPricing, MemoryReserves};
        let info = topology.executors().iter().find(|executor| executor.backend() == backend).unwrap().clone();
        let status = seismic::Status {
            sample: seismic::Sample {
                taken: std::time::SystemTime::now(),
                host_ram: Ok(seismic::HostStatus {
                    allocatable: seismic::Reported::new(host, seismic::ReportedBy::LinuxMemAvailable),
                    reclaimable: 0,
                    limits: Vec::new(),
                    limit_visibility: seismic::LimitVisibility::Complete,
                    held: None,
                }),
                volumes: Vec::new(),
                executors: topology.executors().iter().map(|executor| executor.id()).collect(),
                grants: Vec::new(),
            },
            runtime: Default::default(),
        };
        let reserves = MemoryReserves::of_machine(None);
        let device = crate::platform::selected(&topology, &status, &reserves, info, GraphPricing::Measured(cost)).unwrap();
        (topology, status, reserves, device)
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
    ) -> (seismic::DeviceTopology, seismic::Status, crate::platform::MemoryReserves, crate::platform::SelectedDevice) {
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
        let (topology, status, reserves, device) = rtx3090(0);
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
            .assess_fit(&topology, executor, &status, &reserves)
            .unwrap();
        let allocation = fit.domains.iter().find(|domain| domain.role == DomainRole::Allocation).unwrap();
        let bound = allocation.capacity_bytes - allocation.reserve_bytes;
        let room = |mandatory: u64| {
            let left = bound.saturating_sub(mandatory);
            PlanRoom { free: left, ceiling: left }
        };
        let (planned, ()) = draft
            .with_held_experts(&definition, HeldBound::room(bound), &synthetic(), room, |_, _| Ok::<_, PlanError>(Ok(())))
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
            ExecutionPlanDraft::allocated(&charge).unwrap()
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
        let (wide, ()) = draft
            .with_held_experts(&definition, HeldBound::room(wider), &synthetic(), |mandatory: u64| {
                let left = wider.saturating_sub(mandatory);
                PlanRoom { free: left, ceiling: left }
            }, |_, _| Ok::<_, PlanError>(Ok(())))
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
        let (_, _, _, device) = rtx3090(64 << 30);
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
        assert!(
            bytes < draft.device().assessment_capacity_bytes() - (2 << 30),
            "the fixture's {bytes} weight bytes fit the 3090's {} with room",
            draft.device().assessment_capacity_bytes()
        );
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
        let (_, _, _, device) =
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
        let fit = StableRoom::of(&whole.device).fit(&charge).unwrap();
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
            let (_, _, _, device) = granted(most);
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
                let (topology, status, reserves, device) = granted(grant);
                StableRoom { topology: &topology, executor: &device.info, status: &status, reserves: &reserves }
                    .fit_whole(&charge)
                    .unwrap()
                    .room
                    .free
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
    ///   the chain's second chunk. That is 3,549,751,968 bytes, which no
    ///   whole plan holds beside the weights, so the experts stream. (The
    ///   cold load charged 23,520 bytes less in each of its three state
    ///   terms, so in the startup history slab they share, a limit of the
    ///   harness's load this plan does not reproduce.)
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
        let (_, _, _, card) = granted(most);
        let (definition, draft) = prepare(&card, 4097, kl);
        let whole = draft.held_whole(&definition).unwrap();
        let charge = HeldCharges::<AssessmentMemoryCharge>::default().charge(&definition, &whole).unwrap().charge().clone();
        assert_eq!(
            (charge.state_bytes, charge.startup_state_bytes),
            (3_549_751_968, 264_904_096),
            "the KL load's state at the harness's limits"
        );
        // The card at the least grant whose stable room is the cold load's.
        let room = |grant| {
            let (topology, status, reserves, device) = granted(grant);
            StableRoom { topology: &topology, executor: &device.info, status: &status, reserves: &reserves }
                .fit_whole(&charge)
                .unwrap()
                .room
                .free
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
        let (_, _, _, card) = granted(held);
        let (definition, draft) = prepare(&card, 4097, kl);
        assert!(
            matches!(draft.placed::<AssessmentMemoryCharge>(&definition, None).unwrap(), Placed::Short(_)),
            "the KL harness's eight conversations stream the experts"
        );
        let (definition, draft) = prepare(&card, 4096, served);
        match draft.placed::<AssessmentMemoryCharge>(&definition, None).unwrap() {
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
        let prepared = crate::assessment::prepare_execution_assessment(definition, forms).unwrap();
        let FitReading { topology, status, reserves } = &draft.device.reading;
        match crate::assessment::assess_execution_fit(definition, &prepared, draft.device.graph_pricing, topology, status, reserves).unwrap() {
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
                let bound = HeldBound::of(None, draft.device.assessment_capacity_bytes);
                let BankCeiling { whole, ceiling, .. } = draft.bank_ceiling(&definition, bound).unwrap().unwrap();
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
            let admits = |grant: u64, (bank, charge, _, _): &WholeBank| {
                let (topology, status, reserves, device) = granted(Some(grant));
                let room = StableRoom { topology: &topology, executor: &device.info, status: &status, reserves: &reserves };
                matches!(room.fit(charge).unwrap().verdict, Fits)
            };
            let beside_mandatory = |grant: u64, (bank, charge, floor_charge, grown): &WholeBank| {
                let (topology, status, reserves, device) = granted(Some(grant));
                let room = StableRoom { topology: &topology, executor: &device.info, status: &status, reserves: &reserves };
                let mandatory = ExecutionPlanDraft::allocated(floor_charge).unwrap();
                let beside = room.room(floor_charge, mandatory).unwrap();
                let left = beside.free.min(beside.ceiling);
                let needed = ExecutionPlanDraft::allocated(charge).unwrap();
                *grown <= left && needed.saturating_sub(mandatory) <= left && matches!(room.fit(charge).unwrap().verdict, Fits)
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
            let bank = whole_bank(&prepare(&granted(None).3));
            let (admitted, old) = (least(&|grant| admits(grant, &bank)), least(&|grant| beside_mandatory(grant, &bank)));
            assert!(
                admitted < old,
                "{backend:?}: admission holds the bank from {admitted} bytes, the room beside its mandatory part from {old}"
            );
            let grant = admitted + (old - admitted) / 2;
            // The setup's precondition, of the draft planned at the grant.
            let draft = prepare(&granted(Some(grant)).3);
            let bank = whole_bank(&draft);
            assert!(admits(grant, &bank), "{backend:?}: admission holds the bank at {grant} bytes");
            assert!(!beside_mandatory(grant, &bank), "{backend:?}: the room beside its mandatory part does not at {grant} bytes");
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
        let (_, _, _, device) = rtx3090(64 << 30);
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
        let plan = |bound| draft.clone().with_bank::<AssessmentMemoryCharge>(&definition, bound, &synthetic()).unwrap().into_draft();
        let whole = plan(None);
        let bank = whole.load().expert_bank().expect("a routed load banks its experts");
        let cap = whole.load().weight_bytes().unwrap() - bank.slot_bytes().unwrap() * (bank.widest().get() / 2);
        (plan(Some(WeightBound::Cap(cap))), cap)
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
        let ledgers = DeviceLedgers::open_in(&scope, &device, std::sync::Arc::new(LoadShape { stores: Vec::new() }));
        let mut weights = ledgers.claim(&device, CommittedKind::Load, crate::memory::ClaimDemand::default()).expect("nothing to claim");
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
        let plan = |host: u64, bound| {
            let (_, _, _, device) = rtx3090(host);
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
            draft.with_bank::<AssessmentMemoryCharge>(&definition, bound, &synthetic()).unwrap().into_draft()
        };
        let whole = plan(64 << 30, None);
        let bank = whole.load().expert_bank().expect("a routed load banks its experts");
        let room = whole.load().weight_bytes().unwrap() - bank.slot_bytes().unwrap() * (bank.widest().get() / 2);
        let capped = plan(64 << 30, Some(WeightBound::Cap(room)));
        let roomy = plan(64 << 30, Some(WeightBound::Room(room)));
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
        let (_, _, _, device) = recorded(seismic::fixtures::recorded::m4_pro(), BackendName::Metal, 0, seismic::GraphCost::None);
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
        let whole = draft.clone().with_bank::<AssessmentMemoryCharge>(&definition, None, &synthetic()).unwrap().into_draft();
        let bank = whole.load().expert_bank().expect("a routed load banks its experts");
        let half = bank.slot_bytes().unwrap() * (bank.widest().get() / 2);
        let room = |_: u64| PlanRoom { free: half, ceiling: half };
        let (planned, ()) = draft
            .with_held_experts(&definition, HeldBound::of(None, draft.device().assessment_capacity_bytes()), &synthetic(), room, |_, _| {
                Ok::<_, PlanError>(Ok(()))
            })
            .unwrap()
            .unwrap_or_else(|no| panic!("Gemma 26B q4 streams beside half its experts: {no:?}"));
        let held = planned.draft().load().expert_bank().unwrap();
        assert!(held.slots < held.widest().get(), "{} of {} slots", held.slots, held.widest());
        assert!(held.pass > PassLaunches::FLOOR, "{:?} beside {} streamed slots", held.pass, held.widest().get() - held.slots);
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
        let (_, _, _, device) = recorded(seismic::fixtures::recorded::m4_pro(), BackendName::Metal, 0, seismic::GraphCost::None);
        let (definition, draft) = far_larger(device);
        let bound = bound(draft.device().assessment_capacity_bytes());
        let (planned, ()) = draft
            .with_held_experts(
                &definition,
                bound,
                &synthetic(),
                |_: u64| PlanRoom { free: FAR_ROOM, ceiling: FAR_ROOM },
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
        let (_, _, _, device) = recorded(seismic::fixtures::recorded::m4_pro(), BackendName::Metal, 0, seismic::GraphCost::None);
        let (definition, draft) = far_larger(device);
        let capacity = draft.device().assessment_capacity_bytes();
        let waits = draft
            .with_held_experts(
                &definition,
                HeldBound::of(None, capacity),
                &pending(),
                |_: u64| PlanRoom { free: FAR_ROOM, ceiling: FAR_ROOM },
                |_, _| Ok::<_, PlanError>(Ok(())),
            )
            .err();
        assert!(matches!(waits, Some(PlanError::Unpriced { pending: true, .. })), "{waits:?}");
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
        let (_, _, _, device) = rtx3090(64 << 30);
        let (definition, draft) = far_larger(device);
        let along_host = |activation| {
            let mut costs = costs();
            costs.placements.host = Placement::along(kind(1));
            costs.routes.push(RouteCosts { route: kind(1), activation, formats: Vec::new() });
            Priced::Measured(plan_prices_on(costs))
        };
        let unread = draft.clone().with_bank::<AssessmentMemoryCharge>(&definition, None, &along_host(None)).err();
        assert!(matches!(unread, Some(PlanError::Unpriced { pending: true, .. })), "{unread:?}");
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
        let (_, _, _, device) = rtx3090(64 << 30);
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
        let (_, _, _, device) = recorded(seismic::fixtures::recorded::m4_pro(), BackendName::Metal, 0, seismic::GraphCost::None);
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
        let streamed = draft
            .streaming::<AssessmentMemoryCharge>(&definition, None)
            .unwrap()
            .streamed(&definition, &synthetic())
            .unwrap()
            .into_draft();
        let bank = streamed.load().expert_bank().expect("the streamed plan banks its experts").clone();
        // Each import's storage, admitted as device bytes.
        let import = |load: &ModelLoadPlan| ClaimDemand { device: load.target_weight_demand().unwrap().bytes(), ..ClaimDemand::default() };
        let at = |slots: u64| Ok::<_, std::convert::Infallible>(import(&streamed.load().clone().with_load_slots(slots).unwrap()));
        let (whole_bytes, floor_bytes) = (import(whole.load()).device, at(bank.floor()).unwrap().device);
        assert!(floor_bytes < whole_bytes, "the streamed floor imports {floor_bytes} bytes, the whole plan {whole_bytes}");
        // Another program holds all but midway between the two imports.
        let target = floor_bytes + (whole_bytes - floor_bytes) / 2;
        let shape = std::sync::Arc::new(crate::memory::ledger::LoadShape { stores: Vec::new() });
        // The ledgers of the test host's CPU device, whose stores read as pressed.
        let catalog = seismic::Devices::discover(&[]).unwrap();
        let host = crate::test_support::open_backend(&catalog, BackendName::Cpu).unwrap();
        let ledgers = crate::memory::ledger::DeviceLedgers::open_in(&crate::memory::ledger::Ledgers::new(), &host, shape);
        let now = std::time::Instant::now();
        for ledger in [ledgers.device(), ledgers.host()] {
            ledger.read(ledger.sampled(now), Ok(crate::memory::ledger::Available::new(2 * whole_bytes, target, None).unwrap()));
        }
        // The whole plan is refused for room ...
        let load = crate::memory::ledger::CommittedKind::Load;
        assert!(ledgers.admit_most(&host, load, 0, 0, |_| Ok::<_, std::convert::Infallible>(import(whole.load()))).unwrap().is_err());
        // ... and the streamed plan's bank is fitted into it: the most slots
        // that fit, and no more.
        let (slots, _held) = ledgers.admit_most(&host, load, bank.floor(), bank.slots, &at).unwrap().unwrap();
        assert!(at(slots).unwrap().device <= target, "{slots} slots of {} fit the {target} bytes left", bank.slots);
        assert!(slots == bank.slots || at(slots + 1).unwrap().device > target, "{slots} slots are the most that fit");
    }
}
