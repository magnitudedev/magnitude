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

    /// Retain only empirically selected optional readout views before graph
    /// resources or persistent residency are admitted. The canonical packed
    /// projection remains available to every class.
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
        prices: &super::BankPrices,
    ) -> Result<Result<Self, UnstreamedExperts>, PlanError> {
        if !matches!(self.device.host, Ok(Some(HostTier::Linked(_)))) {
            return Ok(Ok(self));
        }
        let Some(ceiling) = self.bank().host_ceiling().filter(|_| !self.bank().whole()) else {
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
        let points = rising(0, ceiling.get(), &host_charge)?;
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
    /// [`Self::with_held_experts`] decides under `cap`, when given, on the
    /// plan's `prices`, into the stable fit's room beside the load's
    /// mandatory part ([`StableRoom`]), as its assessment settles,
    /// each way of holding them charged as a `C` ([`HeldCharge`]). A capped
    /// load no plan holds is refused with why. An uncapped one is held whole
    /// whatever it needs, its shortfall what assessment and admission find.
    pub fn with_bank<C: HeldCharge>(
        self,
        definition: &ModelDefinition,
        cap: Option<u64>,
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
            // An uncapped load's experts that cannot stream are held whole
            // whatever that needs.
            if cap.is_none() && draft.load.bank.is_none() {
                return Ok(Ok(()));
            }
            Ok::<_, PlanError>(match room.fit(draft, charge)?.verdict {
                super::AssessmentFitVerdict::Fits => Ok(()),
                super::AssessmentFitVerdict::DoesNotFit { deficit_bytes, .. } => Err(deficit_bytes),
            })
        };
        match (self.with_held_experts(definition, cap.unwrap_or(u64::MAX), prices, &mut held, room, confirm)?, cap) {
            (Ok((charged, ())), _) => Ok(charged),
            (Err(_), None) => {
                let draft = self.held_whole(definition)?;
                let charge = held.charge(definition, &draft)?;
                Ok(ChargedDraft { draft, charge, held })
            }
            (Err(NoBank::Unstreamed(reason)), Some(_)) => Err(PlanError::Unstreamed(reason)),
            (Err(NoBank::Bound { needed }), Some(cap)) => Err(PlanError::WeightCap { cap, needed }),
            (Err(NoBank::Short { deficit_bytes }), Some(_)) => Err(PlanError::Resource(crate::CapacityError {
                resource: crate::ResourceKind::DeviceMemory,
                required: capacity + deficit_bytes.get(),
                available: capacity,
            })),
        }
    }

    /// How a load holds its experts under `bound` of weights, the one
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
        bound: u64,
        prices: &Priced<super::PlanPrices>,
        held: &mut HeldCharges<C>,
        room: impl SettleRoom,
        mut confirm: impl FnMut(&Self, &AssessmentMemoryCharge) -> Result<Result<T, NonZeroU64>, E>,
    ) -> Result<Result<(ChargedDraft<C>, T), NoBank>, E> {
        let reason = match self.with_largest_bank(definition, bound, prices, held, &room, &mut confirm)? {
            // The host tier's size leaves the charge as it is: the charge
            // holds the tier at its floor.
            Ok(Settled { draft, charge, found, prices }) => match draft.with_host_tier(charge.charge(), &prices)? {
                Ok(draft) => return Ok(Ok((ChargedDraft { draft, charge, held: held.clone() }, found))),
                Err(reason) => reason,
            },
            Err(NoBank::Unstreamed(reason)) => reason,
            Err(no) => return Ok(Err(no)),
        };
        let whole = self.held_whole(definition)?;
        if weight_bytes(&whole.load)? > bound {
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
    /// whose weights `bound` holds (at most a banked draft's ceiling, which
    /// re-planning never raises), where the bank's curve ends; or why no
    /// bank streams under `bound`. Its floor, its program plan and
    /// everything but its slots and pass are of its structure alone, so each
    /// bank is that one resized ([`Self::resized`]), and its weights at any
    /// slots are that bank's less what its record reservations release
    /// (`ExpertBankPlan::resident_charge`). Its record tensors are reserved
    /// for every expert and commit their slots, each within the device's
    /// single allocation. Its prompt groups chain every chunk of a
    /// whole-context prompt (`with_chain`): a group's places are granted
    /// when it opens.
    fn bank_ceiling(&self, definition: &ModelDefinition, bound: u64) -> Result<Result<BankCeiling, NoBank>, PlanError> {
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
        let needed = weights_at(floor)?;
        if needed > bound {
            return Ok(Err(NoBank::Bound { needed }));
        }
        // The ceiling: the most slots whose weights `bound` holds, where the
        // bank's curve ends. Every bank planned from here grows to it,
        // however few slots the machine admits now.
        let mut ceiling = floor;
        while ceiling < top.min(most) && weights_at(ceiling + 1)? <= bound {
            ceiling += 1;
        }
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
    /// `cap` ([`Self::bank_ceiling`]), with every place and a pass over
    /// them, beside a linked host tier of every expert; or, where they cannot stream,
    /// whole. What every plan's settle grants from, and what a settle
    /// grants where the room holds it all: an assessment's preparation,
    /// which reads the structure alone, is charged at it, and a plan that
    /// does not fit is settled on the machine's prices
    /// ([`Self::with_held_experts`]).
    pub fn with_ceiling_bank<C: HeldCharge>(self, definition: &ModelDefinition, cap: Option<u64>) -> Result<ChargedDraft<C>, PlanError> {
        let mut held = HeldCharges::default();
        let bound = cap.unwrap_or(u64::MAX);
        let draft = match (self.bank_ceiling(definition, bound)?, cap) {
            (Ok(BankCeiling { whole, ceiling, places }), _) => {
                let mut draft = whole.resized(ceiling, ceiling, places)?;
                let linked = matches!(draft.device.host, Ok(Some(HostTier::Linked(_))));
                if let Some(host) = draft.bank().host_ceiling().filter(|_| linked && !draft.bank().whole()) {
                    draft.bank_mut().host_slots = host.get();
                }
                draft
            }
            (Err(NoBank::Bound { needed }), Some(cap)) => return Err(PlanError::WeightCap { cap, needed }),
            (Err(NoBank::Unstreamed(reason)), _) => {
                let whole = self.held_whole(definition)?;
                if weight_bytes(&whole.load)? > bound {
                    return Err(PlanError::Unstreamed(reason));
                }
                whole
            }
            (Err(no), _) => unreachable!("a bank's ceiling asks no room, so it is never {no:?}"),
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
    /// which grows to its ceiling under `bound`, the end of its curve.
    ///
    /// The plan's mandatory part is the bank at its floor with prompt passes
    /// of one launch and groups of one place. Beside it the settle grants,
    /// by the decode or prefill seconds each saves a second (`prices`, the
    /// workload prior's) per byte, into `room` of the mandatory part's
    /// allocation:
    /// - the bank's slots, a step where its charge rises
    ///   (`resident_charge`), each worth the decode seconds it saves;
    /// - the prompt room: groups of the plan's chain places (their class,
    ///   `AssessmentMemoryCharge::with_group_class`) with passes of two
    ///   launches, then each larger pass size (`pass_buffer_bytes`), each
    ///   worth the routed prefill seconds its fewer passes save. A pass of
    ///   several launches needs a group's places to merge them, so places
    ///   come with the first such pass.
    ///
    /// The plan so granted is charged ([`HeldCharges`]) and confirmed: where
    /// its growth exceeds `room` or `confirm` finds it short, the settle
    /// runs again with the deficit owed, releasing the lowest rate per byte
    /// first; short with nothing left to release, the load is short.
    fn with_largest_bank<C: HeldCharge, T, E: From<PlanError>>(
        &self,
        definition: &ModelDefinition,
        bound: u64,
        prices: &Priced<super::PlanPrices>,
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
        let prices = match prices {
            Priced::Measured(prices) => prices.of_bank(&whole, definition)?,
            Priced::Pending(reason) => return Err(PlanError::Unpriced { pending: true, reason: reason.clone() }.into()),
            Priced::Unavailable(reason) => return Err(PlanError::Unpriced { pending: false, reason: reason.clone() }.into()),
        };
        let points = rising(floor, ceiling, &reserved)?;
        let ladder = super::pass_ladder(places);
        let resized = |slots: u64, pass: PassLaunches| whole.resized(slots, ceiling, pass);
        // The mandatory part: what the load holds whatever it releases.
        let mandatory = resized(points[0], PassLaunches::FLOOR)?;
        let floor_charge = *held.charge(definition, &mandatory)?.charge();
        // Host memory that admits not even the tier's floor streams nothing,
        // whatever the device holds.
        if let Some(reason) = mandatory.untiered(&floor_charge)? {
            return Ok(Err(NoBank::Unstreamed(reason)));
        }
        let mandatory_bytes = Self::allocated(&floor_charge)?;
        let class = Self::allocated(&floor_charge.with_group_class().map_err(PlanError::ResourcePlanning)?)?
            .checked_sub(mandatory_bytes)
            .ok_or(PlanError::Arithmetic("a prompt group's class is charged below nothing"))?;
        // Each pass size's buffers, and a group's class with the first pass
        // that merges launches.
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
                Ok((pass, buffers + if pass > PassLaunches::FLOOR { class } else { 0 }))
            })
            .collect::<Result<Vec<_>, PlanError>>()?;
        let pass_bytes = |pass: PassLaunches| -> u64 {
            rungs.iter().find(|(rung, _)| *rung == pass).expect("a pass size of the ladder").1
        };
        let seconds = |slots: u64| prices.step_seconds(crate::assessment::HeldSlots::bank(slots as usize));
        let rates = prices.rates();
        let curves = vec![
            stepped(Holding::Bank(BankHolding::Experts), &points, &reserved, &seconds, rates.decode_steps_per_s)?,
            super::pass_rows_curve(&ladder, 0, &pass_bytes, &|pass| prices.prompt_seconds(pass), rates.requests_per_s, 0.0),
        ];
        let room = room.room(&floor_charge, mandatory_bytes)?;
        let mut settled = None;
        let granted = settle_confirmed(curves, load_supply(room), |granted| -> Result<Result<(), NonZeroU64>, E> {
            let (slots, pass) = (points[granted[0]], ladder[granted[1]]);
            let draft = resized(slots, pass)?;
            let charge = held.charge(definition, &draft)?;
            let needed = match draft.holds_group_class() {
                true => Self::allocated(&charge.charge().with_group_class().map_err(PlanError::ResourcePlanning)?)?,
                false => Self::allocated(charge.charge())?,
            };
            let over = needed.saturating_sub(mandatory_bytes).saturating_sub(room.free.min(room.ceiling));
            if let Some(over) = NonZeroU64::new(over) {
                return Ok(Err(over));
            }
            Ok(confirm(&draft, charge.charge())?.map(|found| settled = Some((draft, charge, found))))
        })?;
        Ok(match granted {
            Ok(_) => {
                let (draft, charge, found) = settled.expect("a settle holds only once its plan is confirmed");
                Ok(Settled { draft, charge, found, prices })
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

    /// Whether the draft leaves room for its prompt groups' class of its
    /// chain places (`AssessmentMemoryCharge::with_group_class`): where its
    /// settle granted passes of several launches, which a group's places
    /// merge.
    pub fn holds_group_class(&self) -> bool {
        self.load.bank.as_ref().is_some_and(|bank| bank.pass > PassLaunches::FLOOR)
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
    /// A bank's, by the pass size it was certified at: a pass sizes the
    /// target family's pass buffers and graphs, so a bank of another is
    /// charged from its structure again.
    banked: Option<(PassLaunches, C)>,
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
        match &self.banked {
            Some((pass, held)) if *pass == bank.pass => {
                let charge = held.charge().at_bank_slots(&draft.load).map_err(PlanError::ResourcePlanning)?;
                Ok(held.clone().at(charge))
            }
            _ => Ok(self.banked.insert((bank.pass, C::of(definition, draft)?)).1.clone()),
        }
    }

    /// The charges themselves, without what else their certification gave.
    pub fn charges(&self) -> HeldCharges {
        HeldCharges {
            banked: self.banked.as_ref().map(|(pass, held)| (*pass, *held.charge())),
            whole: self.whole.as_ref().map(|held| *held.charge()),
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
    /// The bound is below the weights the plan `needed` at the bank's floor.
    Bound { needed: u64 },
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
    /// class of the plan's chain places where the load reserves it
    /// ([`ExecutionPlanDraft::holds_group_class`],
    /// `AssessmentMemoryCharge::with_group_class`), as the settle's room
    /// holds it. The one confirm a load's plan and its assessment make.
    pub fn fit(&self, draft: &ExecutionPlanDraft, charge: &AssessmentMemoryCharge) -> Result<super::AssessmentFit, PlanError> {
        let charge = match draft.holds_group_class() {
            true => charge.clone().with_group_class().map_err(PlanError::ResourcePlanning)?,
            false => charge.clone(),
        };
        charge.assess_fit(self.topology, self.executor, self.status, self.reserves).map_err(PlanError::ResourcePlanning)
    }
}

impl SettleRoom for StableRoom<'_> {
    fn room(&self, mandatory: &AssessmentMemoryCharge, bytes: u64) -> Result<PlanRoom, PlanError> {
        let bound = mandatory
            .clone()
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
    prices: super::BankPrices,
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
    Ok(Curve { holding, borrows: false, held: Vec::new(), offered })
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
        let pass = Curve { holding: Holding::PassRows, borrows: false, held: Vec::new(), offered: vec![Step { bytes: 10, worth: Rate::new(0.5), move_seconds: 0.0 }] };
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

    /// Gemma 26B q4's routed shape on the recorded RTX 3090 (24 GB, CUDA,
    /// its graphs priced as the device measured them), uncapped, planned on
    /// synthetic prices into the device's stable room. The bank's last
    /// slots are worth far more per byte than the larger merged passes of
    /// its whole-context prompt group, so the settle keeps the bank
    /// resident (every slot) and a pass of at least 512 rows beside it,
    /// whose prefill is no slower than 512-row passes'.
    #[test]
    fn gemma_q4_on_a_3090_keeps_its_bank_resident_beside_a_merged_pass() {
        use crate::planning::fixtures::{LIMITS, gemma_moe_fixture_definition, gemma_moe_fixture_manifest};
        use crate::platform::{DomainRole, GraphPricing, MemoryReserves};
        let topology = seismic::fixtures::recorded::rtx3090();
        // Stable fit reads no headroom.
        let status = seismic::Status {
            sample: seismic::Sample {
                taken: std::time::SystemTime::now(),
                host_ram: Ok(seismic::HostStatus {
                    allocatable: seismic::Reported::new(0, seismic::ReportedBy::LinuxMemAvailable),
                    obtainable: seismic::Reported::new(0, seismic::ReportedBy::LinuxMemAvailable),
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
        let info = topology.executors().iter().find(|executor| executor.backend() == BackendName::Cuda).unwrap().clone();
        // CUDA's pooled driver memory as the RTX 3090 measured it.
        let cost = seismic::GraphCost::Pooled { graph: 92_663, launch: 8_102, granule: 2 << 20 };
        let device = crate::platform::selected(&topology, &status, &reserves, info, GraphPricing::Measured(cost)).unwrap();
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
            .with_held_experts(&definition, bound, &prices, &mut HeldCharges::<AssessmentMemoryCharge>::default(), room, |_, _| {
                Ok::<_, PlanError>(Ok(()))
            })
            .unwrap()
            .unwrap_or_else(|no| panic!("Gemma 26B q4 streams on the RTX 3090: {no:?}"));
        let bank = planned.draft().load().expert_bank().expect("a routed load banks its experts");
        let allocated = planned.draft().allocation_bytes(&definition).unwrap();
        let limits = planned.draft().policy().limits();
        let Priced::Measured(plan_prices) = &prices else { unreachable!("the prices are measured") };
        let priced = plan_prices.of_bank(planned.draft(), &definition).unwrap();
        let step = |slots: usize| priced.step_seconds(crate::assessment::HeldSlots::bank(slots));
        let prompt = |launches: usize| priced.prompt_seconds(PassLaunches::new(NonZeroUsize::new(launches).unwrap()));
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
        let Priced::Measured(prices) = &prices else { unreachable!("the prices are measured") };
        let prices = prices.of_bank(planned.draft(), &definition).unwrap();
        assert!(
            prices.prompt_seconds(bank.pass) <= prices.prompt_seconds(PassLaunches::FLOOR),
            "{:?}'s prefill is slower than 512-row passes'",
            bank.pass
        );
    }
}
