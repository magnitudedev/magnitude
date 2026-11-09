//! One allocation-free, checked numerical startup decision.

use super::{
    ArtifactComponent, ArtifactComponentKind, AssessmentMemoryCharge, CapabilityPlan,
    ComponentPlan, ComponentSelection, HostTier, MAX_DRAFT_PROPOSALS, ModelLoadPlan, PassLaunches, PlannedMethod,
    ProgramPlan, ReadoutClassChoice, ResourceLimits, ResourcePlan, UndraftedBounds, WeightPlan,
};
use crate::{
    ExecutionPath, UnstreamedExperts,
    error::PlanError,
    platform::{FitReading, SelectedDevice},
};
use magnitude_artifacts::PackageManifest;
use magnitude_family_contracts::ModelDefinition;
use magnitude_state::KvCodec;
use seismic::{BackendName, ExecutorSelector};
use crate::assessment::Priced;
use crate::domain::BankHolding;
use crate::domain::value::{Curve, Holding, Rate, Step, Supply, settle};
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
    /// What its recorded graphs hold for their census
    /// (`SelectedDevice::graph_cost`).
    graph_cost: seismic::GraphCost,
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
        self.graph_cost
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
    /// sized at load by the one settle of its holdings: each host slot's
    /// records (`ExpertBankPlan::host_charge`, the steps where it rises, up
    /// to every expert, `ExpertBankPlan::host_ceiling`) against the decode
    /// seconds it saves at the bank's settled slots (`prices`), into the
    /// room host RAM holds beside the load's charge at the tier's floor
    /// (`AssessmentMemoryCharge::assess_fit`, `Fit` alone interpreting
    /// placements, stores and reserves), confirmed there; none past its
    /// floor beside a bank of every expert, which misses nothing. The tier's
    /// size is this load-time decision; it is filled from the bank's open
    /// on, before decode needs it.
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
        prices: Option<&super::BankPrices>,
    ) -> Result<Result<Self, UnstreamedExperts>, PlanError> {
        if !matches!(self.device.host, Ok(Some(HostTier::Linked(_)))) {
            return Ok(Ok(self));
        }
        let Some(ceiling) = self.bank().host_ceiling().filter(|&ceiling| ceiling > 0 && !self.bank().whole()) else {
            return Ok(Ok(self));
        };
        let bank = self.bank();
        let host_charge = |slots: u64| bank.host_charge(slots).map_err(PlanError::ResourcePlanning);
        let left = |slots: u64| self.host_left(charge, slots);
        let free = left(0)?;
        if free < 0 {
            return Ok(Err(self.untiered_reason()));
        }
        let slots = usize::try_from(bank.slots).map_err(|_| PlanError::Arithmetic("a bank's slots exceed usize"))?;
        let points = rising(0, ceiling, &host_charge)?;
        // Unpriced (an assessment's, `Unmeasured::SizedByMemory`): the most
        // host memory admits.
        let Some(prices) = prices else {
            let mut most = points[0];
            for &host in &points {
                if left(host)? < 0 {
                    break;
                }
                most = host;
            }
            self.bank_mut().host_slots = most;
            return Ok(Ok(self));
        };
        let seconds = |host: u64| prices.step_seconds(crate::assessment::HeldSlots { slots, host: host as usize });
        let tier = stepped(Holding::Bank(BankHolding::Host), &points, &host_charge, &seconds, prices.rates().decode_steps_per_s)?;
        let granted = settle_confirmed(vec![tier], load_supply(PlanRoom { free: free as u64, ceiling: u64::MAX }), |granted| {
            let left = left(points[granted[0]])?;
            Ok::<_, PlanError>(NonZeroU64::new(left.min(0).unsigned_abs()).map_or(Ok(()), Err))
        })?;
        // The floor fits, so a settle that releases every step holds.
        let granted = granted.expect("a host tier's floor fits");
        self.bank_mut().host_slots = points[granted[0]];
        Ok(Ok(self))
    }

    /// The host RAM `charge`, the draft's with its tier at its floor, leaves
    /// beside a tier of `slots`: negative where it does not fit.
    fn host_left(&self, charge: &AssessmentMemoryCharge, slots: u64) -> Result<i64, PlanError> {
        let bank = self.bank();
        let host_charge = |slots: u64| bank.host_charge(slots).map_err(PlanError::ResourcePlanning);
        let FitReading { topology, status, reserves } = &self.device.reading;
        let host_ram = topology.host_ram().id();
        let growth = host_charge(slots)?
            .checked_sub(host_charge(0)?)
            .ok_or(PlanError::Arithmetic("a host tier is charged less than its floor"))?;
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

    /// The draft of an elastic load, with its charge: its experts held as
    /// [`Self::with_held_experts`] decides under the load's own `bound`,
    /// when given, on the plan's `prices`, into the stable fit's room beside
    /// the load's mandatory part ([`StableRoom`]), as its assessment
    /// settles, each way of holding them charged as a `C` ([`HeldCharge`]),
    /// never more of them on the device than its room ([`HeldBound`]). A
    /// load no plan holds is refused with why.
    pub fn with_bank<C: HeldCharge>(
        self,
        definition: &ModelDefinition,
        bound: Option<WeightBound>,
        prices: &Priced<super::PlanPrices>,
    ) -> Result<ChargedDraft<C>, PlanError> {
        let mut held = HeldCharges::default();
        let capacity = self.device.assessment_capacity_bytes;
        // What the load holds beyond its mandatory part settles into the
        // stable fit's room, as its assessment does ([`StableRoom`]), never
        // a reading of the moment, which is the running load's memory pace's.
        let room = StableRoom::of(&self.device);
        // Every plan is confirmed as its assessment confirms it
        // ([`StableRoom::fit`]).
        let confirm = |draft: &Self, charge: &AssessmentMemoryCharge| {
            // An unbounded load's experts that cannot stream are held whole:
            // whether the device holds them is what assessment and admission
            // find.
            if bound.is_none() && draft.load.bank.is_none() {
                return Ok(Ok(()));
            }
            Ok::<_, PlanError>(match room.fit(draft, charge)?.verdict {
                super::AssessmentFitVerdict::Fits => Ok(()),
                super::AssessmentFitVerdict::DoesNotFit { deficit_bytes, .. } => Err(deficit_bytes),
            })
        };
        match self.with_held_experts(definition, HeldBound::of(bound, capacity), prices, super::Unmeasured::Waits, &mut held, room, confirm)? {
            Ok((charged, ())) => Ok(charged),
            Err(NoBank::Unstreamed(reason)) => Err(PlanError::Unstreamed(reason)),
            Err(NoBank::Bound { bound, needed }) => Err(PlanError::WeightBound { bound, needed }),
            Err(NoBank::Short { deficit_bytes }) => Err(PlanError::Resource(crate::CapacityError {
                resource: crate::ResourceKind::DeviceMemory,
                required: capacity + deficit_bytes.get(),
                available: capacity,
            })),
        }
    }

    /// How a load holds its experts under `bound` ([`HeldBound`]), the one
    /// decision a load and its assessment share: in the bank, prompt room
    /// and host tier the plan's settle grants on `prices` into `room`
    /// beside the plan's mandatory part (`Self::with_largest_bank`,
    /// `Self::with_host_tier`), confirmed by `confirm`; or, where they cannot
    /// stream, whole, if `bound` holds them and `confirm` admits that. Each
    /// plan is charged from `held`, which charges each way of holding the
    /// experts once ([`HeldCharges`]); `confirm` judges a plan by its
    /// charge, giving what it found or the bytes the plan is short. The draft
    /// held comes with the charge it admitted.
    pub fn with_held_experts<C: HeldCharge, T, E: From<PlanError>>(
        &self,
        definition: &ModelDefinition,
        bound: HeldBound,
        prices: &Priced<super::PlanPrices>,
        unmeasured: super::Unmeasured,
        held: &mut HeldCharges<C>,
        room: impl SettleRoom,
        mut confirm: impl FnMut(&Self, &AssessmentMemoryCharge) -> Result<Result<T, NonZeroU64>, E>,
    ) -> Result<Result<(ChargedDraft<C>, T), NoBank>, E> {
        let reason = match self.with_largest_bank(definition, bound, prices, unmeasured, held, &room, &mut confirm)? {
            // The host tier's size leaves the charge as it is: the charge
            // holds the tier at its floor.
            Ok(Settled { draft, charge, found, prices }) => match draft.with_host_tier(charge.charge(), prices.as_ref())? {
                Ok(draft) => return Ok(Ok((ChargedDraft { draft, charge, held: held.clone() }, found))),
                Err(reason) => reason,
            },
            Err(NoBank::Unstreamed(reason)) => reason,
            Err(no) => return Ok(Err(no)),
        };
        // Experts that cannot stream are held whole, within the load's own
        // bound; whether the device holds them is `confirm`'s. Experts that
        // can stream are never held whole in place of a bank the room is
        // short of.
        let whole = self.held_whole(definition)?;
        let bytes = weight_bytes(&whole.load)?;
        if bound.limit.is_some_and(|limit| bytes > limit.bytes()) {
            return Ok(Err(NoBank::Unstreamed(reason)));
        }
        let charge = held.charge(definition, &whole)?;
        Ok(match confirm(&whole, charge.charge())? {
            Ok(found) => Ok((ChargedDraft { draft: whole, charge, held: held.clone() }, found)),
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
        let (reservation, host) = match bank_substrate(
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
                reservation,
                host,
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
        // `bound`, where the bank's curve ends, the floor at least. A plan
        // settled from here grows no further than its room admits
        // (`with_largest_bank`).
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

    /// The draft with its experts held as its largest plan holds them,
    /// whatever room the machine has: through its bank at its ceiling under
    /// the load's own `bound` ([`Self::bank_ceiling`]), with every place and
    /// a pass over them, beside a linked host tier of every expert; or, where they cannot stream,
    /// whole. What every plan's settle grants from, and what a settle
    /// grants where the room holds it all: an assessment's preparation,
    /// which reads the structure alone, is charged at it, and a plan that
    /// does not fit is settled on the machine's prices
    /// ([`Self::with_held_experts`]).
    pub fn with_ceiling_bank<C: HeldCharge>(self, definition: &ModelDefinition, limit: Option<WeightBound>) -> Result<ChargedDraft<C>, PlanError> {
        let mut held = HeldCharges::default();
        let bound = HeldBound::of(limit, self.device.assessment_capacity_bytes);
        let draft = match self.bank_ceiling(definition, bound)? {
            Ok(BankCeiling { whole, ceiling, places }) => {
                let mut draft = whole.resized(ceiling, ceiling, places)?;
                let linked = matches!(draft.device.host, Ok(Some(HostTier::Linked(_))));
                if let Some(host) = draft.bank().host_ceiling().filter(|_| linked && !draft.bank().whole()) {
                    draft.bank_mut().host_slots = host;
                }
                draft
            }
            Err(NoBank::Bound { bound, needed }) => return Err(PlanError::WeightBound { bound, needed }),
            Err(NoBank::Unstreamed(reason)) => {
                let whole = self.held_whole(definition)?;
                let bytes = weight_bytes(&whole.load)?;
                if limit.is_some_and(|limit| bytes > limit.bytes()) {
                    return Err(PlanError::Unstreamed(reason));
                }
                whole
            }
            Err(no @ NoBank::Short { .. }) => unreachable!("a bank's ceiling asks no room, so it is never {no:?}"),
        };
        let charge = held.charge(definition, &draft)?;
        Ok(ChargedDraft { draft, charge, held })
    }

    /// The draft with its routed experts streamed through an expert bank
    /// (the same slots per layer in every group, at least the bank's floor
    /// and at most each layer's experts, or a banked draft's ceiling) and
    /// its prompt room, as the plan's one settle grants them
    /// (`domain::value::settle`), with its charge and what `confirm` found
    /// of it; or why no bank's does.
    ///
    /// Each plan is the bank's structure resized ([`Self::bank_ceiling`]),
    /// whose ceiling under `bound` ends the settle's curve. The plan's own
    /// ceiling is the most of those slots `room` admits beside everything
    /// the plan holds: no plan holds a bank whose ceiling the machine cannot
    /// reach, so the serving heap weighs the bank's growth only as far as it
    /// can go.
    ///
    /// The plan's mandatory part is the bank at its floor with prompt passes
    /// of one launch and groups of one place. Beside it the settle grants,
    /// by the decode or prefill seconds each saves a second (`prices`, the
    /// workload prior's) per byte, into `room` of the mandatory part's
    /// allocation:
    /// - the bank's slots, a step where its charge rises
    ///   (`resident_charge`), each worth the decode seconds it saves;
    /// - the prompt room: each pass size (`pass_buffer_bytes`) beside the
    ///   class of the group places it merges
    ///   (`AssessmentMemoryCharge::with_group_class`), each worth the routed
    ///   prefill seconds its fewer passes save. A pass of several launches
    ///   needs a group's places to merge them, so each pass size comes with
    ///   its places, and none beyond them.
    ///
    /// The plan so granted is charged ([`HeldCharges`]) and confirmed: where
    /// its growth exceeds `room` or `confirm` finds it short, the settle
    /// runs again with the deficit owed, releasing the lowest rate per byte
    /// first; short with nothing left to release, the load is short.
    fn with_largest_bank<C: HeldCharge, T, E: From<PlanError>>(
        &self,
        definition: &ModelDefinition,
        bound: HeldBound,
        prices: &Priced<super::PlanPrices>,
        unmeasured: super::Unmeasured,
        held: &mut HeldCharges<C>,
        room: &impl SettleRoom,
        confirm: &mut impl FnMut(&Self, &AssessmentMemoryCharge) -> Result<Result<T, NonZeroU64>, E>,
    ) -> Result<Result<Settled<C, T>, NoBank>, E> {
        let BankCeiling { whole, ceiling, places } = match self.bank_ceiling(definition, bound)? {
            Ok(ceiling) => ceiling,
            Err(no) => return Ok(Err(no)),
        };
        let bank = whole.bank();
        let floor = bank.floor();
        let reserved = |slots| bank.resident_charge(slots).map_err(PlanError::ResourcePlanning);
        let points = rising(floor, ceiling, &reserved)?;
        let ladder = super::pass_ladder(places);
        // The mandatory part: what the load holds whatever it releases.
        let mandatory = whole.resized(points[0], ceiling, PassLaunches::FLOOR)?;
        let floor_charge = held.charge(definition, &mandatory)?.charge().clone();
        // Host memory that admits not even the tier's floor streams nothing,
        // whatever the device holds.
        if let Some(reason) = mandatory.untiered(&floor_charge)? {
            return Ok(Err(NoBank::Unstreamed(reason)));
        }
        let mandatory_bytes = Self::allocated(&floor_charge)?;
        // Each pass size's buffers, and the class of the group places it
        // merges with each that merges launches.
        let rungs = ladder
            .iter()
            .map(|&pass| {
                let buffers = crate::programs::native_target_graph::pass_buffer_bytes(
                    &whole.load,
                    &definition.decoder,
                    whole.policy.limits,
                    pass,
                )
                .map_err(PlanError::ResourcePlanning)?;
                let class = match pass > PassLaunches::FLOOR {
                    true => Self::allocated(&floor_charge.with_group_class(pass).map_err(PlanError::ResourcePlanning)?)?
                        .checked_sub(mandatory_bytes)
                        .ok_or(PlanError::Arithmetic("a prompt group's class is charged below nothing"))?,
                    false => 0,
                };
                Ok((pass, buffers + class))
            })
            .collect::<Result<Vec<_>, PlanError>>()?;
        let pass_bytes = |pass: PassLaunches| -> u64 {
            rungs.iter().find(|(rung, _)| *rung == pass).expect("a pass size of the ladder").1
        };
        // A pass's misses are read at the most the bank and its host tier
        // can hold: the least a merged pass saves, whatever the settle
        // grants the bank. The bank holds no more than its ceiling, nor
        // than the room beside the mandatory part admits.
        let room = room.room(&floor_charge, mandatory_bytes)?;
        let admitted = reserved(points[0])?.saturating_add(room.free.min(room.ceiling));
        let mut fits = points[0];
        for &slots in &points {
            if reserved(slots)? > admitted {
                break;
            }
            fits = slots;
        }
        let mut settled = None;
        // A draft of `slots` under `ceiling` and `pass`, its charge, and
        // how far what it needs beyond the mandatory part falls short of the
        // room (`Ok`, what it leaves) or exceeds it (`Err`).
        let mut charged = |slots: u64, ceiling: u64, pass: PassLaunches| -> Result<_, E> {
            let draft = whole.resized(slots, ceiling, pass)?;
            let charge = held.charge(definition, &draft)?;
            let needed = match draft.load.group_class_pass() {
                Some(pass) => Self::allocated(&charge.charge().with_group_class(pass).map_err(PlanError::ResourcePlanning)?)?,
                None => Self::allocated(charge.charge())?,
            };
            let (grown, room) = (needed.saturating_sub(mandatory_bytes), room.free.min(room.ceiling));
            Ok((draft, charge, room.checked_sub(grown).ok_or_else(|| grown - room)))
        };
        // The bank's ceiling beside a bank of `slots` that leaves `left` of
        // the room: the most slots the room admits beside everything the
        // plan holds, which the bank grows to while it serves and never
        // past. A ceiling the machine cannot reach would leave the bank's
        // growth always short of it, its last steps weighed against what
        // nothing can hold.
        let reach = |slots: u64, left: u64| -> Result<u64, PlanError> {
            let base = reserved(slots)?;
            let mut reach = slots;
            for &more in points.iter().filter(|&&more| more > slots) {
                if reserved(more)? - base > left {
                    break;
                }
                reach = more;
            }
            Ok(reach)
        };
        // A bank the room holds whole beside the mandatory part is held
        // whole, whatever the machine's prices or whether they are measured:
        // every expert resident misses nothing, so no price can make fewer
        // slots worth their bytes, and growth has nothing left to reach.
        // Every pass prices alike beside it, so the largest whose prompt
        // group's class fits beside it is held, the fewest submissions.
        if fits == bank.slots && ceiling == bank.slots {
            for &pass in ladder.iter().rev() {
                let (draft, charge, left) = charged(fits, fits, pass)?;
                if left.is_err() {
                    continue;
                }
                if let Ok(found) = confirm(&draft, charge.charge())? {
                    return Ok(Ok(Settled { draft, charge, found, prices: None }));
                }
            }
        }
        let prices = unmeasured.prices(match prices {
            Priced::Measured(prices) => prices.of_bank(&whole, definition)?,
            Priced::Pending(reason) => Priced::Pending(reason.clone()),
            Priced::Unavailable(reason) => Priced::Unavailable(reason.clone()),
        })?;
        // Unpriced (an assessment's, `Unmeasured::SizedByMemory`): the
        // largest bank the room admits at the floor pass that `confirm`
        // admits, by memory alone.
        let Some(prices) = prices else {
            let mut short = None;
            for &slots in points.iter().rev().filter(|&&slots| slots <= fits) {
                let left = match charged(slots, ceiling, PassLaunches::FLOOR)?.2 {
                    Ok(left) => left,
                    Err(excess) => {
                        short = short.or(NonZeroU64::new(excess));
                        continue;
                    }
                };
                let (draft, charge, _) = charged(slots, reach(slots, left)?, PassLaunches::FLOOR)?;
                match confirm(&draft, charge.charge())? {
                    Ok(found) => return Ok(Ok(Settled { draft, charge, found, prices: None })),
                    Err(deficit) => short = Some(deficit),
                }
            }
            return Ok(Err(NoBank::Short { deficit_bytes: short.expect("a bank has a floor, which is tried") }));
        };
        let seconds = |slots: u64| prices.step_seconds(crate::assessment::HeldSlots::bank(slots as usize));
        let rates = prices.rates();
        let count = |slots: u64| usize::try_from(slots).map_err(|_| PlanError::Arithmetic("a bank's slots exceed usize"));
        let most = crate::assessment::HeldSlots { slots: count(fits)?, host: count(bank.host_ceiling().unwrap_or(0))? };
        let curves = vec![
            stepped(Holding::Bank(BankHolding::Experts), &points, &reserved, &seconds, rates.decode_steps_per_s)?,
            super::pass_rows_curve(&ladder, 0, &pass_bytes, &|pass| prices.prompt_seconds(pass, most), rates.requests_per_s, 0.0),
        ];
        let granted = settle_confirmed(curves, load_supply(room), |granted| -> Result<Result<(), NonZeroU64>, E> {
            let slots = points[granted[0]];
            // A larger pass the prices cannot tell from the granted one
            // prices no slower: the largest of them that fits is held, as
            // fewer submissions of the same work, the granted bank beside it.
            let granted_seconds = prices.prompt_seconds(ladder[granted[1]], most);
            let mut tied = None;
            for &pass in ladder[granted[1] + 1..].iter().rev() {
                if prices.prompt_seconds(pass, most) > granted_seconds {
                    continue;
                }
                if let (_, _, Ok(left)) = charged(slots, ceiling, pass)? {
                    tied = Some((pass, left));
                    break;
                }
            }
            let (pass, left) = match tied {
                Some(tied) => tied,
                None => match charged(slots, ceiling, ladder[granted[1]])? {
                    (_, _, Ok(left)) => (ladder[granted[1]], left),
                    (_, _, Err(excess)) => return Ok(Err(NonZeroU64::new(excess).expect("a nonzero excess"))),
                },
            };
            let (draft, charge, _) = charged(slots, reach(slots, left)?, pass)?;
            Ok(confirm(&draft, charge.charge())?.map(|found| settled = Some((draft, charge, found))))
        })?;
        Ok(match granted {
            Ok(_) => {
                let (draft, charge, found) = settled.expect("a settle holds only once its plan is confirmed");
                Ok(Settled { draft, charge, found, prices: Some(prices) })
            }
            Err(deficit_bytes) => Err(NoBank::Short { deficit_bytes }),
        })
    }

    /// The draft with its experts held whole: its prompt groups chain every
    /// chunk of a whole-context prompt, as a banked load's do (a chain is
    /// what merges their passes); one launch where the load has no routed
    /// experts, which have nothing to merge.
    pub(crate) fn held_whole(&self, definition: &ModelDefinition) -> Result<Self, PlanError> {
        let places = match super::bank::routed_experts(definition) {
            Ok(_) => self.prompt_places(definition),
            Err(PlanError::Unstreamed(UnstreamedExperts::Unrouted)) => 1,
            Err(error) => return Err(error),
        };
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
            None => self.held_whole(definition)?,
            Some(bank) => {
                let mut load = self
                    .load
                    .clone()
                    .unbanked()
                    .with_expert_bank(definition, bank.slots, bank.places, bank.flights, bank.reservation(), bank.host())?
                    .with_bank_slots(bank.slots, bank.ceiling)?;
                let built = load.bank.as_mut().expect("the load was just banked");
                (built.host_slots, built.pass) = (bank.host_slots, bank.pass);
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
/// whole plan is the one its draft's experts held whole make, so each way is
/// charged from its structure ([`crate::assessment::StructuralCharge`]) once
/// however often one draft is searched, as a `C` ([`HeldCharge`]).
#[derive(Clone, Debug)]
pub struct HeldCharges<C = AssessmentMemoryCharge> {
    /// A bank's, by the pass size it was certified at and whether its
    /// ceiling reaches every expert: a pass sizes the target family's pass
    /// buffers and graphs, and a bank that can be whole has a resident form
    /// (`ExpertBankPlan::reaches_whole`), so a bank of another is charged
    /// from its structure again.
    banked: Option<((PassLaunches, bool), C)>,
    whole: Option<C>,
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

/// A drafted plan's charge, with the bounds its certification gives of the
/// target's plan alone (`UndraftedBounds`: its target family without the
/// drafter's writes, and the vision family both hold)
/// (`crate::assessment::execution_memory_charge_with_undrafted`): what
/// prices the drafter by the target's plan alone at the same holding of
/// experts (`crate::assessment::undrafted_memory_charge`). Those families
/// are of the plan's structure, so a bank's slots leave them as they are.
#[derive(Clone, Debug)]
pub struct DraftedCharge {
    pub charge: AssessmentMemoryCharge,
    pub undrafted: UndraftedBounds,
}

impl HeldCharge for DraftedCharge {
    fn of(definition: &ModelDefinition, draft: &ExecutionPlanDraft) -> Result<Self, PlanError> {
        let (charge, undrafted) =
            crate::assessment::execution_memory_charge_with_undrafted(definition, draft).map_err(charge_error)?;
        Ok(Self { charge, undrafted })
    }

    fn charge(&self) -> &AssessmentMemoryCharge {
        &self.charge
    }

    fn at(self, charge: AssessmentMemoryCharge) -> Self {
        Self { charge, ..self }
    }
}

impl<C: HeldCharge> HeldCharges<C> {
    /// The charge of `draft`, of its way of holding its experts.
    fn charge(&mut self, definition: &ModelDefinition, draft: &ExecutionPlanDraft) -> Result<C, PlanError> {
        let Some(bank) = draft.load.bank.as_ref() else {
            return match &self.whole {
                Some(whole) => Ok(whole.clone()),
                None => Ok(self.whole.insert(C::of(definition, draft)?).clone()),
            };
        };
        let structure = (bank.pass, bank.reaches_whole());
        match &self.banked {
            Some((of, held)) if *of == structure => {
                let charge = held.charge().clone().at_bank_slots(&draft.load).map_err(PlanError::ResourcePlanning)?;
                Ok(held.clone().at(charge))
            }
            _ => Ok(self.banked.insert((structure, C::of(definition, draft)?)).1.clone()),
        }
    }

    /// The charges themselves, without what else their certification gave.
    pub fn charges(&self) -> HeldCharges {
        HeldCharges {
            banked: self.banked.as_ref().map(|(structure, held)| (*structure, held.charge().clone())),
            whole: self.whole.as_ref().map(|held| held.charge().clone()),
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
/// which its assessment reads rather than charging it again, with the
/// charges its search made ([`HeldCharges`]), which a search of it again
/// reads rather than charging them again.
#[derive(Clone, Debug)]
pub struct ChargedDraft<C = AssessmentMemoryCharge> {
    draft: ExecutionPlanDraft,
    charge: C,
    held: HeldCharges<C>,
}

impl ChargedDraft {
    /// `draft` with its charge (`assessment::execution_memory_charge`),
    /// of no search: a search of it charges its own ways of holding its
    /// experts.
    pub fn of(definition: &ModelDefinition, draft: ExecutionPlanDraft) -> Result<Self, PlanError> {
        let charge = draft.memory_charge(definition)?;
        Ok(Self { draft, charge, held: HeldCharges::default() })
    }

}

impl<C: HeldCharge> ChargedDraft<C> {
    /// The charges this draft's search made.
    pub fn held(&self) -> &HeldCharges<C> {
        &self.held
    }

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

impl ChargedDraft<DraftedCharge> {
    /// The bounds this draft's certification gives of its target's plan
    /// alone ([`DraftedCharge`]).
    pub fn undrafted(&self) -> &UndraftedBounds {
        &self.charge.undrafted
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
}

impl<F: Fn(u64) -> PlanRoom> SettleRoom for F {
    fn room(&self, _: &AssessmentMemoryCharge, bytes: u64) -> Result<PlanRoom, PlanError> {
        Ok(self(bytes))
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

    /// The stable fit of `draft` charged `charge`: with its prompt groups'
    /// class of the places its pass merges where the load reserves it
    /// ([`super::ModelLoadPlan::group_class_pass`],
    /// `AssessmentMemoryCharge::with_group_class`), as the settle's room
    /// holds it. The one confirm a load's plan and its assessment make.
    pub fn fit(&self, draft: &ExecutionPlanDraft, charge: &AssessmentMemoryCharge) -> Result<super::AssessmentFit, PlanError> {
        let fit = |charge: &AssessmentMemoryCharge| charge.assess_fit(self.topology, self.executor, self.status, self.reserves);
        match draft.load.group_class_pass() {
            Some(pass) => charge.with_group_class(pass).and_then(|charge| fit(&charge)),
            None => fit(charge),
        }
        .map_err(PlanError::ResourcePlanning)
    }
}

impl SettleRoom for StableRoom<'_> {
    fn room(&self, mandatory: &AssessmentMemoryCharge, bytes: u64) -> Result<PlanRoom, PlanError> {
        let bound = mandatory
            .assess_fit(self.topology, self.executor, self.status, self.reserves)
            .and_then(|fit| fit.settle_bound())
            .map_err(PlanError::ResourcePlanning)?;
        let left = bound.saturating_sub(bytes);
        Ok(PlanRoom { free: left, ceiling: left })
    }
}

/// `room` as a load's settle reads it: nothing borrowed, nothing held yet
/// to exchange.
fn load_supply(room: PlanRoom) -> Supply {
    Supply {
        free: room.free,
        owed: 0,
        borrowable: 0,
        borrow_price: 0.0,
        ceiling: room.ceiling,
        horizon: std::time::Duration::ZERO,
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

/// What a bank's settle decided: the draft, its charge, what its confirm
/// found of it, and the prices its holdings were settled on.
struct Settled<C, T> {
    draft: ExecutionPlanDraft,
    charge: C,
    found: T,
    /// None where the bank is whole or an assessment sized it by memory
    /// alone.
    prices: Option<super::BankPrices>,
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

/// The curve of `holding` stepping through `points` ([`rising`]), none
/// held: each step the bytes `charge` rises by, worth what `seconds` falls
/// by, `per_second` times a second.
fn stepped(
    holding: Holding,
    points: &[u64],
    charge: &impl Fn(u64) -> Result<u64, PlanError>,
    seconds: &impl Fn(u64) -> f64,
    per_second: f64,
) -> Result<Curve<Holding>, PlanError> {
    let offered = points
        .windows(2)
        .map(|pair| {
            let bytes = charge(pair[1])?
                .checked_sub(charge(pair[0])?)
                .ok_or(PlanError::Arithmetic("a holding's charge fell as it grew"))?;
            let worth = Rate::new((seconds(pair[0]) - seconds(pair[1])).max(0.0) * per_second);
            Ok(Step { bytes, worth, move_seconds: 0.0 })
        })
        .collect::<Result<_, PlanError>>()?;
    Ok(Curve { holding, borrows: false, lends: false, held: Vec::new(), offered })
}

/// Settle `curves` (none holding a step) into `supply`, then confirm what
/// it grants by `check` (the plan's real charge at each curve's granted
/// steps, in order): where `check` finds it short by some bytes, settle
/// again with them owed, releasing the lowest rate per byte first, until
/// it holds; short with nothing granted left to release, the bytes it is
/// short.
fn settle_confirmed<E>(
    curves: Vec<Curve<Holding>>,
    supply: Supply,
    mut check: impl FnMut(&[usize]) -> Result<Result<(), NonZeroU64>, E>,
) -> Result<Result<Vec<usize>, NonZeroU64>, E> {
    let count = |steps: &[(Holding, usize)], holding: Holding| {
        steps.iter().filter(|(of, _)| *of == holding).map(|(_, steps)| steps).sum::<usize>()
    };
    let settled = settle(&curves, supply);
    let mut granted: Vec<usize> = curves.iter().map(|curve| count(&settled.granted, curve.holding)).collect();
    loop {
        let deficit = match check(&granted)? {
            Ok(()) => return Ok(Ok(granted)),
            Err(deficit) => deficit,
        };
        let holding = curves
            .iter()
            .zip(&granted)
            .map(|(curve, &steps)| Curve {
                holding: curve.holding,
                borrows: false,
                lends: false,
                held: curve.offered[..steps].iter().rev().copied().collect(),
                offered: Vec::new(),
            })
            .collect::<Vec<_>>();
        let owed = Supply { free: 0, owed: deficit.get(), ceiling: u64::MAX, ..supply };
        let released = settle(&holding, owed).released;
        if released.is_empty() {
            return Ok(Err(deficit));
        }
        for (steps, curve) in granted.iter_mut().zip(&holding) {
            *steps -= count(&released, curve.holding);
        }
    }
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
) -> Result<(seismic::Reservation, HostTier), PlanError> {
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
        Ok(Some(host)) => return Ok((reservation, host)),
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
                max_allocation_bytes: device.info.limits().max_allocation.value,
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
                graph_cost: device.graph_cost.cost(),
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

    /// A holding steps where its charge rises, each step holding the most
    /// sizes of its charge: a size that costs nothing more comes free.
    #[test]
    fn a_holding_steps_only_where_its_charge_rises() {
        let charge = |slots: u64| Ok::<_, PlanError>(slots.div_ceil(4) * 100);
        assert_eq!(rising(2, 13, &charge).unwrap(), vec![4, 8, 12, 13]);
        assert_eq!(rising(5, 5, &charge).unwrap(), vec![5]);
    }

    /// A plan found short after its settle is settled again with the
    /// deficit owed, releasing the lowest rate per byte first, until it
    /// holds; with nothing left to release, it is short by what remains.
    #[test]
    fn a_short_plan_releases_its_least_worth_first() {
        let points = [0, 1, 2, 3];
        let charge = |slots: u64| Ok::<_, PlanError>(slots * 10);
        let bank = stepped(Holding::Bank(BankHolding::Experts), &points, &charge, &|slots| 10.0 - slots as f64 * 3.0, 1.0).unwrap();
        let pass = Curve { holding: Holding::PassRows, borrows: false, lends: false, held: Vec::new(), offered: vec![Step { bytes: 10, worth: Rate::new(0.5), move_seconds: 0.0 }] };
        let room = load_supply(PlanRoom { free: 40, ceiling: u64::MAX });
        // The real charge is 5 bytes over the estimate for each step held.
        let check = |granted: &[usize]| -> Result<Result<(), NonZeroU64>, PlanError> {
            let held = (granted[0] + granted[1]) as u64;
            Ok(NonZeroU64::new((held * 15).saturating_sub(40)).map_or(Ok(()), Err))
        };
        let granted = settle_confirmed(vec![bank.clone(), pass.clone()], room, check).unwrap().unwrap();
        // Four steps fit the estimate; the real charge holds two, and the
        // pass's step is worth the least.
        assert_eq!(granted, vec![2, 0]);
        let never = |_: &[usize]| -> Result<Result<(), NonZeroU64>, PlanError> { Ok(Err(NonZeroU64::MIN)) };
        assert_eq!(settle_confirmed(vec![bank, pass], room, never).unwrap(), Err(NonZeroU64::MIN));
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
                    obtainable: seismic::Reported::new(host, seismic::ReportedBy::LinuxMemAvailable),
                    limits: Vec::new(),
                    limit_visibility: seismic::LimitVisibility::Complete,
                    displacement: None,
                    kernel_pressure: None,
                    free_target: None,
                    held: None,
                    displaced_bytes: None,
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
    /// reports `grant` bytes of memory, the stable grant every plan and its
    /// assessment fit into (`FitMode::Stable`): CUDA's usable memory, or
    /// Vulkan's device-local heap, as the heap and each of its memory types
    /// report it. The device path's grant is asserted to be it.
    fn rtx3090_granted(
        backend: BackendName,
        host: u64,
        cost: seismic::GraphCost,
        grant: u64,
    ) -> (seismic::DeviceTopology, seismic::Status, crate::platform::MemoryReserves, crate::platform::SelectedDevice) {
        let mut reports = seismic::fixtures::rtx3090();
        match backend {
            BackendName::Cuda => reports.cuda[0].memory = seismic::fixtures::CudaMemory::Own { usable: Ok(grant) },
            BackendName::Vulkan => {
                let vulkan = &mut reports.vulkan[0];
                let heap = vulkan.memory.heaps.iter().position(|heap| heap.device_local).expect("the RTX 3090 has a device-local heap");
                vulkan.memory.heaps[heap].bytes = grant;
                for kind in &mut vulkan.kinds {
                    if let seismic::KindMemory::Heap { index, bytes, .. } = &mut kind.memory {
                        if *index == heap {
                            *bytes = grant;
                        }
                    }
                }
            }
            other => panic!("the RTX 3090 reports no device grant on {other:?}"),
        }
        let topology = seismic::fixtures::topology(&reports);
        let info = topology.executors().iter().find(|executor| executor.backend() == backend).unwrap();
        let path = info.paths(seismic::Placing::Fixed(seismic::Contract::DeviceOnly))[0];
        assert_eq!(
            topology.access_path(path).and_then(|path| path.grant()).map(|granted| granted.value),
            Some(grant),
            "{backend:?}: the device path's stable grant is the memory its device reports"
        );
        recorded(topology, backend, host, cost)
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
    /// its graphs priced as the device measured them), uncapped, planned on
    /// synthetic prices into the device's stable room. The bank's last
    /// slots are worth far more per byte than the larger merged passes of
    /// its whole-context prompt group, so the settle keeps the bank
    /// resident (every slot) and a pass of at least 512 rows beside it,
    /// whose prefill is no slower than 512-row passes'. Every pass ties
    /// beside the whole bank, so the plan holds the largest one whose prompt
    /// group's class of the places it merges fits beside it, the largest
    /// of all where the room holds that one's class.
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
        let prices = Priced::Measured(crate::assessment::synthetic::plan_prices());
        let (planned, ()) = draft
            .with_held_experts(&definition, HeldBound::room(bound), &prices, crate::Unmeasured::Waits, &mut HeldCharges::<AssessmentMemoryCharge>::default(), room, |_, _| {
                Ok::<_, PlanError>(Ok(()))
            })
            .unwrap()
            .unwrap_or_else(|no| panic!("Gemma 26B q4 streams on the RTX 3090: {no:?}"));
        let bank = planned.draft().load().expert_bank().expect("a routed load banks its experts");
        let allocated = planned.draft().allocation_bytes(&definition).unwrap();
        let limits = planned.draft().policy().limits();
        let Priced::Measured(plan_prices) = &prices else { unreachable!("the prices are measured") };
        let Priced::Measured(priced) = plan_prices.of_bank(planned.draft(), &definition).unwrap() else {
            unreachable!("synthetic costs price the bank")
        };
        let step = |slots: usize| priced.step_seconds(crate::assessment::HeldSlots::bank(slots));
        let whole = crate::assessment::HeldSlots::bank(bank.slots as usize);
        let prompt = |launches: usize| priced.prompt_seconds(PassLaunches::new(NonZeroUsize::new(launches).unwrap()), whole);
        assert_eq!(
            (bank.slots, bank.ceiling),
            (128, 128),
            "the bank holds every expert: resident ({:?}, {} bytes a slot, {allocated} allocated of {bound}; \
             step at 124/126/127/128 slots {:.6}/{:.6}/{:.6}/{:.6} s; prompt at 1/2/16/32 launches {:.4}/{:.4}/{:.4}/{:.4} s; rates {:?})",
            bank.pass,
            bank.slot_bytes().unwrap(),
            step(124),
            step(126),
            step(127),
            step(128),
            prompt(1),
            prompt(2),
            prompt(16),
            prompt(32),
            priced.rates(),
        );
        let region = *magnitude_batching::row_classes(limits.max_launch_rows).last().unwrap() as u64;
        assert!(bank.pass.get() as u64 * region >= 512, "{:?} of {region} rows", bank.pass);
        // Every pass size prices alike beside a whole bank, compute-bound
        // from 512 rows on: the largest whose class of the places it merges
        // fits is held, the fewest submissions.
        let allocated_at = |pass: PassLaunches| {
            let draft = planned.draft().resized(bank.slots, bank.ceiling, pass).unwrap();
            let charge = crate::assessment::execution_memory_charge(&definition, &draft).unwrap();
            match draft.load().group_class_pass() {
                Some(pass) => ExecutionPlanDraft::allocated(&charge.with_group_class(pass).unwrap()).unwrap(),
                None => ExecutionPlanDraft::allocated(&charge).unwrap(),
            }
        };
        assert!(allocated_at(bank.pass) <= bound, "{:?} and its class fit beside the whole bank", bank.pass);
        let ladder = super::super::pass_ladder(PassLaunches::new(NonZeroUsize::new(bank.places as usize).unwrap()));
        if let Some(&next) = ladder.iter().find(|&&pass| pass > bank.pass) {
            let with_class = allocated_at(next);
            assert!(with_class > bound, "{next:?} and its class ({with_class} bytes) fit the room ({bound}) beside the whole bank");
        }
        // With room for the largest pass and its class beside the whole
        // bank, the largest pass is held.
        let largest = *ladder.last().unwrap();
        let wider = allocated_at(largest).max(bound);
        let (wide, ()) = draft
            .with_held_experts(&definition, HeldBound::room(wider), &prices, crate::Unmeasured::Waits, &mut HeldCharges::<AssessmentMemoryCharge>::default(), |mandatory: u64| {
                let left = wider.saturating_sub(mandatory);
                PlanRoom { free: left, ceiling: left }
            }, |_, _| Ok::<_, PlanError>(Ok(())))
            .unwrap()
            .unwrap();
        let widened = wide.draft().load().expert_bank().unwrap();
        assert_eq!((widened.slots, widened.ceiling), (128, 128), "the wider room keeps the whole bank");
        assert_eq!(
            widened.pass,
            *super::super::pass_ladder(PassLaunches::new(NonZeroUsize::new(widened.places as usize).unwrap())).last().unwrap(),
            "{:?} of {} places",
            widened.pass,
            widened.places
        );
        let Priced::Measured(prices) = &prices else { unreachable!("the prices are measured") };
        let Priced::Measured(prices) = prices.of_bank(planned.draft(), &definition).unwrap() else {
            unreachable!("synthetic costs price the bank")
        };
        assert!(
            prices.prompt_seconds(bank.pass, whole) <= prices.prompt_seconds(PassLaunches::FLOOR, whole),
            "{:?}'s prefill is slower than 512-row passes'",
            bank.pass
        );
    }

    /// Qwen3.6 35B-A3B q4's routed shape (40 blocks of 256 experts choosing
    /// 8, hidden 2,048, intermediate 512: about 18.1 GB of Q4_0 expert
    /// stacks) on the recorded RTX 3090 (24 GB), planned as a load plans it
    /// (`with_bank`) on a machine with nothing measured: the room holds
    /// every expert, so the bank is whole without the prices, and the load
    /// never waits for them or streams.
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
        let prices = Priced::Pending("the executor's costs are not measured yet".into());
        let planned = draft
            .with_bank::<AssessmentMemoryCharge>(&definition, None, &prices)
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
        let fit = StableRoom::of(&whole.device).fit(&whole, &charge).unwrap();
        assert!(
            matches!(fit.verdict, super::super::AssessmentFitVerdict::Fits),
            "the experts held whole fit the card's budget: {:?}",
            fit.verdict
        );
        // As the real load plans it: before the executor's costs are
        // measured.
        let prices = Priced::Pending("the executor's costs are not measured yet".into());
        let planned = draft.with_bank::<AssessmentMemoryCharge>(&definition, None, &prices).unwrap();
        if let Some(bank) = planned.draft().load().expert_bank() {
            assert_eq!((bank.slots, bank.ceiling), (256, 256), "a bank of the experts the room holds whole is whole");
        }
    }

    /// The draft's experts held as `assess_execution_fit` holds them, prices
    /// pending: from its largest plan (`with_ceiling_bank`), settled into the
    /// stable fit's bound and confirmed by its fit, sized by memory alone.
    fn assessed_as_fit(definition: &ModelDefinition, draft: ExecutionPlanDraft) -> ChargedDraft {
        let ceiling = draft.with_ceiling_bank::<AssessmentMemoryCharge>(definition, None).unwrap();
        let room = StableRoom::of(&ceiling.draft.device);
        let fit = room.fit(&ceiling.draft, ceiling.charge.charge()).unwrap();
        if matches!(fit.verdict, super::super::AssessmentFitVerdict::Fits) {
            return ceiling;
        }
        let confirm = |candidate: &ExecutionPlanDraft, charge: &AssessmentMemoryCharge| {
            Ok::<_, PlanError>(match room.fit(candidate, charge)?.verdict {
                super::super::AssessmentFitVerdict::Fits => Ok(()),
                super::super::AssessmentFitVerdict::DoesNotFit { deficit_bytes, .. } => Err(deficit_bytes),
            })
        };
        let pending = Priced::Pending("the executor's costs are not measured yet".into());
        let mut held = ceiling.held.clone();
        let (assessed, ()) = ceiling
            .draft
            .with_held_experts(definition, HeldBound::room(fit.settle_bound().unwrap()), &pending, crate::Unmeasured::SizedByMemory, &mut held, room, confirm)
            .unwrap()
            .unwrap_or_else(|no| panic!("the assessment finds no plan: {no:?}"));
        assessed
    }

    /// Qwen3.6 35B-A3B q4's routed shape at 4k on the recorded RTX 3090, on
    /// CUDA and on Vulkan, its graphs as the RTX 3090's are taken (`rtx3090_graphs`), its device
    /// reporting memory between the least grant
    /// at which admission's fit holds a bank of every expert and the least at
    /// which every slot also fits the settle's room beside the bank's
    /// mandatory part (the bank at its floor), both derived here and
    /// asserted of the draft planned at that grant: before any price is
    /// measured, the load and its assessment hold the bank whole, as
    /// admission fits it, never waiting on the prices a partial bank
    /// settles on.
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
                matches!(room.fit(bank, charge).unwrap().verdict, Fits)
            };
            let beside_mandatory = |grant: u64, (bank, charge, floor_charge, grown): &WholeBank| {
                let (topology, status, reserves, device) = granted(Some(grant));
                let room = StableRoom { topology: &topology, executor: &device.info, status: &status, reserves: &reserves };
                let mandatory = ExecutionPlanDraft::allocated(floor_charge).unwrap();
                let beside = room.room(floor_charge, mandatory).unwrap();
                let left = beside.free.min(beside.ceiling);
                let needed = ExecutionPlanDraft::allocated(charge).unwrap();
                *grown <= left && needed.saturating_sub(mandatory) <= left && matches!(room.fit(bank, charge).unwrap().verdict, Fits)
            };
            // The least grant that holds the bank so.
            let least = |holds: &dyn Fn(u64) -> bool| {
                let (mut short, mut held) = (0u64, 64u64 << 30);
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
            let pending = Priced::Pending("the executor's costs are not measured yet".into());
            let whole = |planned: &ExecutionPlanDraft| planned.load().expert_bank().map(|bank| (bank.slots, bank.ceiling, bank.host_slots));
            let planned = draft
                .clone()
                .with_bank::<AssessmentMemoryCharge>(&definition, None, &pending)
                .unwrap_or_else(|error| panic!("{backend:?}: the load is held before any price is measured: {error:?}"));
            assert_eq!(whole(planned.draft()), Some((256, 256, 0)), "{backend:?}: the load holds the bank whole");
            let assessed = assessed_as_fit(&definition, draft);
            assert_eq!(whole(assessed.draft()), Some((256, 256, 0)), "{backend:?}: its assessment holds the bank whole");
        }
    }

    /// A weight cap bounds every weight byte the load holds, its host tier's
    /// among them: Gemma 26B q4 on the recorded RTX 3090 with 64 GB of host
    /// RAM free, capped at its weights less half its experts, holds within
    /// the cap, its host tier ended by what the cap leaves beside the bank,
    /// not by the host's room.
    #[test]
    fn a_weight_cap_bounds_the_host_tier_beside_the_bank() {
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
        let prices = Priced::Measured(crate::assessment::synthetic::plan_prices());
        let plan = |bound| draft.clone().with_bank::<AssessmentMemoryCharge>(&definition, bound, &prices).unwrap().into_draft();
        let whole = plan(None);
        let bank = whole.load().expert_bank().expect("a routed load banks its experts");
        let cap = whole.load().weight_bytes().unwrap() - bank.slot_bytes().unwrap() * (bank.widest().get() / 2);
        let capped = plan(Some(WeightBound::Cap(cap)));
        let held = capped.load().expert_bank().unwrap();
        assert!(!held.whole() && held.host_ceiling().is_some(), "half the experts stream through a host tier");
        let bytes = capped.load().weight_bytes().unwrap();
        assert!(bytes <= cap, "{bytes} weight bytes held under a cap of {cap}");
        // The cap, not the host's room, ends the tier.
        let ceiling = held.host_ceiling().unwrap();
        assert!(ceiling < held.widest().get() && held.host_slots <= ceiling, "{} host slots under a ceiling of {ceiling}", held.host_slots);
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
        let prices = Priced::Measured(crate::assessment::synthetic::plan_prices());
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
            draft.with_bank::<AssessmentMemoryCharge>(&definition, bound, &prices).unwrap().into_draft()
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

    /// A merged pass's saving is its misses' reads at the most the bank
    /// can hold, which is no more than the room admits, whatever its
    /// structure's ceiling: Gemma 26B q4 on the recorded M4 Pro, uncapped
    /// (its structure's ceiling every expert) but with room beside its
    /// mandatory part for half its experts, reads the rest from storage, so
    /// passes that read them fewer times save storage reads and the settle
    /// merges launches.
    #[test]
    fn a_pass_saves_the_misses_of_the_bank_the_room_admits() {
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
        let prices = Priced::Measured(crate::assessment::synthetic::plan_prices());
        let whole = draft.clone().with_bank::<AssessmentMemoryCharge>(&definition, None, &prices).unwrap().into_draft();
        let bank = whole.load().expert_bank().expect("a routed load banks its experts");
        let half = bank.slot_bytes().unwrap() * (bank.widest().get() / 2);
        let room = |_: u64| PlanRoom { free: half, ceiling: half };
        let (planned, ()) = draft
            .with_held_experts(&definition, HeldBound::of(None, draft.device().assessment_capacity_bytes()), &prices, crate::Unmeasured::Waits, &mut HeldCharges::<AssessmentMemoryCharge>::default(), room, |_, _| {
                Ok::<_, PlanError>(Ok(()))
            })
            .unwrap()
            .unwrap_or_else(|no| panic!("Gemma 26B q4 streams beside half its experts: {no:?}"));
        let held = planned.draft().load().expert_bank().unwrap();
        assert!(held.slots < held.widest().get(), "{} of {} slots", held.slots, held.widest());
        assert!(held.pass > PassLaunches::FLOOR, "{:?} beside {} streamed slots", held.pass, held.widest().get() - held.slots);
    }

    /// A DeepSeek V4.1-sized routed model (40 layers of 384 experts
    /// choosing 8, 18.6 MB an expert, 285 GB of them) at a 64k context (a
    /// whole-context group of 128 places), prepared on `device`, and its
    /// plan prices.
    fn far_larger(device: crate::platform::SelectedDevice) -> (ModelDefinition, ExecutionPlanDraft, Priced<crate::PlanPrices>) {
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
        (definition, draft, Priced::Measured(crate::assessment::synthetic::plan_prices()))
    }

    /// [`far_larger`]'s bank on the recorded M4 Pro as a plan of it under
    /// `bound` holds it, with room beside its mandatory part for 20 GB.
    fn far_larger_bank(bound: impl FnOnce(u64) -> HeldBound) -> super::super::ExpertBankPlan {
        let (_, _, _, device) = recorded(seismic::fixtures::recorded::m4_pro(), BackendName::Metal, 0, seismic::GraphCost::None);
        let (definition, draft, prices) = far_larger(device);
        let bound = bound(draft.device().assessment_capacity_bytes());
        let (planned, ()) = draft
            .with_held_experts(
                &definition,
                bound,
                &prices,
                crate::Unmeasured::Waits,
                &mut HeldCharges::<AssessmentMemoryCharge>::default(),
                |_: u64| PlanRoom { free: FAR_ROOM, ceiling: FAR_ROOM },
                |_, _| Ok::<_, PlanError>(Ok(())),
            )
            .unwrap()
            .unwrap_or_else(|no| panic!("a V4.1-sized model streams under {bound:?}: {no:?}"));
        planned.draft().load().expert_bank().unwrap().clone()
    }

    const FAR_ROOM: u64 = 20_000_000_000;

    /// On a machine whose prices are not measured, an assessment's plan
    /// sizes the bank by memory alone, past its floor and within the room
    /// beside its mandatory part, and a load's plan waits for them.
    #[test]
    fn an_unmeasured_plan_sizes_its_bank_by_memory_for_an_assessment_and_waits_for_a_load() {
        let (_, _, _, device) = recorded(seismic::fixtures::recorded::m4_pro(), BackendName::Metal, 0, seismic::GraphCost::None);
        let (definition, draft, _) = far_larger(device);
        let pending = Priced::Pending("the job has not run".into());
        let capacity = draft.device().assessment_capacity_bytes();
        let plan = |unmeasured| {
            draft.with_held_experts(
                &definition,
                HeldBound::of(None, capacity),
                &pending,
                unmeasured,
                &mut HeldCharges::<AssessmentMemoryCharge>::default(),
                |_: u64| PlanRoom { free: FAR_ROOM, ceiling: FAR_ROOM },
                |_, _| Ok::<_, PlanError>(Ok(())),
            )
        };
        let waits = plan(crate::Unmeasured::Waits).err();
        assert!(matches!(waits, Some(PlanError::Unpriced { pending: true, .. })), "{waits:?}");
        let (sized, ()) = plan(crate::Unmeasured::SizedByMemory)
            .unwrap()
            .unwrap_or_else(|no| panic!("an unmeasured assessment sizes a bank by memory: {no:?}"));
        let bank = sized.draft().load().expert_bank().unwrap();
        let grown = |slots| bank.resident_charge(slots).unwrap() - bank.resident_charge(bank.floor()).unwrap();
        let found = format!("{} slots (floor {}, ceiling {}), {} bytes past the floor", bank.slots, bank.floor(), bank.ceiling, grown(bank.slots));
        assert!(bank.slots > bank.floor() && grown(bank.slots) <= FAR_ROOM, "{found}");
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

    /// Uncapped, with room beside its mandatory part for 20 GB, a model far
    /// larger than its machine holds a bank whose ceiling that room admits,
    /// not every expert, and merges its passes beside it.
    #[test]
    fn an_uncapped_load_far_larger_than_its_machine_grows_only_to_its_room() {
        let uncapped = far_larger_bank(|capacity| HeldBound::of(None, capacity));
        let grown = uncapped.resident_charge(uncapped.ceiling).unwrap() - uncapped.resident_charge(uncapped.floor()).unwrap();
        let found = format!(
            "{} of {} slots, ceiling {} ({grown} bytes past the floor beside the room's {FAR_ROOM}), {:?} of {} places",
            uncapped.slots,
            uncapped.widest(),
            uncapped.ceiling,
            uncapped.pass,
            uncapped.places,
        );
        assert!(uncapped.ceiling < uncapped.widest().get() && grown <= FAR_ROOM, "{found}");
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
        let (definition, draft, _) = far_larger(device);
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
        let (definition, draft, prices) = far_larger(device);
        let loaded = draft
            .with_bank::<AssessmentMemoryCharge>(&definition, None, &prices)
            .unwrap_or_else(|error| panic!("a V4.1-sized model plans uncapped on an RTX 3090: {error}"))
            .into_draft();
        let streamed = loaded.load().expert_bank().expect("an uncapped load larger than its machine streams its experts");
        assert!(streamed.slots < streamed.widest().get(), "{} of {} slots", streamed.slots, streamed.widest());
        assert!(streamed.pass > PassLaunches::FLOOR, "{:?} of {} places", streamed.pass, streamed.places);
    }
}
