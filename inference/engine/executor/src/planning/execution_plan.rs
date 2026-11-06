//! One allocation-free, checked numerical startup decision.

use super::{
    ArtifactComponent, ArtifactComponentKind, AssessmentMemoryCharge, CapabilityPlan,
    ComponentPlan, ComponentSelection, HostTier, MAX_DRAFT_PROPOSALS, ModelLoadPlan, PlannedMethod,
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

    /// The draft with its host tier sized at load, where the tier is linked
    /// (its records in host storage): the most host slots, up to every
    /// expert (`ExpertBankPlan::host_ceiling`), whose charge the selection's
    /// reading admits (`AssessmentMemoryCharge::assess_fit`, `Fit` alone
    /// interpreting placements, stores and reserves), the charge of every
    /// size at least that of fewer; none past its floor where not even one
    /// is admitted, nor beside a bank of every expert, which misses nothing
    /// (a shrink of it is a change of the memory budget, which the tier's
    /// growth follows). The tier's size is this load-time decision; it is
    /// filled from the bank's open on, before decode needs it.
    ///
    /// `charge` is the draft's ([`Self::memory_charge`]), which holds the
    /// tier at its floor, as the bank commits it at load: a size's charge is
    /// it with the tier's growth past the floor
    /// (`ExpertBankPlan::host_charge`) in host storage beside it, the one
    /// term the size moves.
    ///
    /// Every file miss lands in a tier record, so the tier's floor, its
    /// landing reserve (`ExpertBankPlan::host_records` at no host slot: a
    /// request's misses and the hints' records, every staging record's
    /// landing at once), is what makes a step live: a reading that admits
    /// not even the floor leaves the experts unstreamed, with why.
    fn with_host_tier(
        mut self,
        charge: &AssessmentMemoryCharge,
    ) -> Result<Result<Self, UnstreamedExperts>, PlanError> {
        if !matches!(self.device.host, Ok(Some(HostTier::Linked(_)))) {
            return Ok(Ok(self));
        }
        let Some(ceiling) = self.bank().host_ceiling().filter(|_| !self.bank().whole()) else {
            return Ok(Ok(self));
        };
        let bank = self.bank();
        let floor = bank.host_charge(0).map_err(PlanError::ResourcePlanning)?;
        let FitReading { topology, status, reserves } = &self.device.reading;
        // Whether the reading admits the charge beside a tier of `slots`.
        let admits = |slots: u64| -> Result<bool, PlanError> {
            let growth = bank
                .host_charge(slots)
                .map_err(PlanError::ResourcePlanning)?
                .checked_sub(floor)
                .ok_or(PlanError::Arithmetic("a host tier is charged less than its floor"))?;
            let fit = charge
                .with_weights(crate::memory::ClaimDemand { host: growth, ..Default::default() })
                .and_then(|charge| charge.assess_fit(topology, &self.device.executor, status, reserves))
                .map_err(PlanError::ResourcePlanning)?;
            Ok(fit.verdict == super::AssessmentFitVerdict::Fits)
        };
        if !admits(0)? {
            let reserve = bank.landing_records();
            let reason = format!("host memory admits not even the tier's landing reserve of {reserve} records a group");
            return Ok(Err(UnstreamedExperts::Untiered { backend: self.device.backend, reason }));
        }
        // The most slots admitted lies in [admitted, refused): the floor is
        // admitted.
        let (mut admitted, mut refused) = (0, ceiling.get() + 1);
        while refused - admitted > 1 {
            let slots = admitted + (refused - admitted) / 2;
            match admits(slots)? {
                true => admitted = slots,
                false => refused = slots,
            }
        }
        self.bank_mut().host_slots = admitted;
        Ok(Ok(self))
    }

    /// The draft's expert bank, which a streamed plan holds.
    fn bank_mut(&mut self) -> &mut super::ExpertBankPlan {
        self.load.bank.as_mut().expect("a streamed plan holds its bank")
    }

    /// The draft of an elastic load, with its charge: its experts held as
    /// [`Self::with_held_experts`] decides under `cap`, when given, its
    /// charges confirmed against the device's assessment capacity, each way
    /// of holding them charged as a `C` ([`HeldCharge`]). A capped load no
    /// plan holds is refused with why. An uncapped one is held whole
    /// whatever it needs, its shortfall what assessment and admission find.
    pub fn with_bank<C: HeldCharge>(
        self,
        definition: &ModelDefinition,
        cap: Option<u64>,
    ) -> Result<ChargedDraft<C>, PlanError> {
        let mut held = HeldCharges::default();
        let capacity = self.device.assessment_capacity_bytes;
        // A bank beyond its floor is admitted beside its prompt groups' class
        // of the planned places (`AssessmentMemoryCharge::with_group_class`);
        // at its floor, which the load needs to run, as the rest of the load
        // is.
        let confirm = |draft: &Self, charge: &AssessmentMemoryCharge| {
            // An uncapped load's experts that cannot stream are held whole
            // whatever that needs.
            if cap.is_none() && draft.load.bank.is_none() {
                return Ok(Ok(()));
            }
            let needed = match draft.beyond_floor() {
                true => Self::allocated(&charge.clone().with_group_class().map_err(PlanError::ResourcePlanning)?)?,
                false => Self::allocated(charge)?,
            };
            let excess = needed.saturating_sub(capacity);
            Ok::<_, PlanError>(NonZeroU64::new(excess).map_or(Ok(()), Err))
        };
        match (self.with_held_experts(definition, cap.unwrap_or(u64::MAX), &mut held, confirm)?, cap) {
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
    /// decision a load and its assessment share: in the largest bank that
    /// `bound` holds and `confirm` admits (`Self::with_largest_bank`),
    /// beside the host tier that bank's charge leaves room for
    /// (`Self::with_host_tier`); or, where they cannot stream, whole, if
    /// `bound` holds them and `confirm` admits that. Each plan is charged
    /// from `held`, which charges each way of holding the experts once
    /// ([`HeldCharges`]); `confirm` judges a plan by its charge, giving what
    /// it found or the bytes the plan is short. The draft held comes with
    /// the charge it admitted.
    pub fn with_held_experts<C: HeldCharge, T, E: From<PlanError>>(
        &self,
        definition: &ModelDefinition,
        bound: u64,
        held: &mut HeldCharges<C>,
        mut confirm: impl FnMut(&Self, &AssessmentMemoryCharge) -> Result<Result<T, NonZeroU64>, E>,
    ) -> Result<Result<(ChargedDraft<C>, T), NoBank>, E> {
        let mut charged = |draft: &Self| -> Result<Result<(C, T), NonZeroU64>, E> {
            let charge = held.charge(definition, draft)?;
            Ok(confirm(draft, charge.charge())?.map(|found| (charge, found)))
        };
        let reason = match self.with_largest_bank(definition, bound, &mut charged)? {
            // The host tier's size leaves the charge as it is: the charge
            // holds the tier at its floor.
            Ok((draft, (charge, found))) => match draft.with_host_tier(charge.charge())? {
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
        Ok(match charged(&whole)? {
            Ok((charge, found)) => Ok((ChargedDraft { draft: whole, charge, held: held.clone() }, found)),
            Err(_) => Err(NoBank::Unstreamed(reason)),
        })
    }

    /// The draft with its routed experts streamed through the largest expert
    /// bank (the same slots per layer in every group, at most each layer's
    /// experts, or a banked draft's ceiling, and at least the bank's floor)
    /// whose weights `bound` holds and whose plan `confirm` admits, with
    /// what `confirm` found of it; or why no bank's does. The bank grows to
    /// the most slots `bound` holds, its ceiling. Its record tensors are
    /// reserved for every expert and commit their slots, each within the
    /// device's single allocation. The bank is built once, of every expert:
    /// its floor, its program plan and everything but its slots are of its
    /// structure alone, so each bank searched is that one resized
    /// (`ModelLoadPlan::with_bank_slots`), and its weights at any slots are
    /// that bank's less what its record reservations release
    /// (`ExpertBankPlan::resident_charge`), which finds the ceiling by
    /// arithmetic. A plan `confirm` finds short by some bytes takes slots
    /// away by that shortfall, down to the floor: each pass ends below the
    /// last. Its prompt groups chain every chunk of a whole-context prompt
    /// (`with_chain`), which the plan charges at their floor of one place: a
    /// group's places are granted when it opens. The one search for a bank's
    /// slots: under a weight cap, and under a machine's capacity.
    fn with_largest_bank<T, E: From<PlanError>>(
        &self,
        definition: &ModelDefinition,
        bound: u64,
        mut confirm: impl FnMut(&Self) -> Result<Result<T, NonZeroU64>, E>,
    ) -> Result<Result<(Self, T), NoBank>, E> {
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
        let (slot_bytes, most) = (
            bank.slot_bytes().map_err(PlanError::ResourcePlanning)?,
            bank.most_slots(self.device.max_allocation_bytes)
                .map_err(PlanError::ResourcePlanning)?,
        );
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
        let Some(room) = bound.checked_sub(needed) else {
            return Ok(Err(NoBank::Bound { needed }));
        };
        // The most slots whose weights `bound` holds, searched from the
        // estimate a slot's bytes give. The estimate is near but not exact:
        // the granules backing each record tensor charge up to one more than
        // its records, and `slot_bytes` over-states a slot of a group still
        // holding its decode request's records (`ExpertBankPlan::records_at`).
        let estimate = (floor + room / slot_bytes).min(top).min(most);
        let ceiling = most_admitted(floor, top.min(most), estimate, |slots| Ok(weights_at(slots)? <= bound))?;
        // Every bank planned from here grows to the ceiling, however few
        // slots the machine admits now.
        let resized = |slots| -> Result<Self, PlanError> {
            Ok(Self { load: whole.load.clone().with_bank_slots(slots, ceiling)?, ..whole.clone() })
        };
        let (mut slots, mut draft) = (ceiling, resized(ceiling)?);
        assert_eq!(
            weight_bytes(&draft.load)?,
            weights_at(ceiling)?,
            "a bank's weights moved other than by its record reservations' charge"
        );
        let found = loop {
            match confirm(&draft)? {
                Ok(found) => break found,
                Err(deficit_bytes) if slots == floor => {
                    return Ok(Err(NoBank::Short { deficit_bytes }));
                }
                Err(deficit_bytes) => {
                    // The shallowest commitment releasing the deficit: charges
                    // plateau over record counts and granules, never linear in
                    // the slots (`ExpertBankPlan::resident_charge`).
                    let bank = draft.bank();
                    let Some(next) = super::releasing(slots, floor, deficit_bytes.get(), |at| bank.resident_charge(at))
                        .map_err(PlanError::ResourcePlanning)?
                    else {
                        return Ok(Err(NoBank::Short { deficit_bytes }));
                    };
                    slots = next;
                    draft = resized(slots)?;
                }
            }
        };
        Ok(Ok((draft, found)))
    }

    /// The draft with its experts held whole: its prompt groups chain every
    /// chunk of a whole-context prompt, as a banked load's do (a chain is
    /// what merges their passes); one launch where the load has no routed
    /// experts, which have nothing to merge.
    fn held_whole(&self, definition: &ModelDefinition) -> Result<Self, PlanError> {
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
                load.bank.as_mut().expect("the load was just banked").host_slots = bank.host_slots;
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

    /// Whether the draft holds a bank beyond its floor: one whose slots past
    /// the floor leave room for its prompt groups' class
    /// (`AssessmentMemoryCharge::with_group_class`).
    pub fn beyond_floor(&self) -> bool {
        self.load.bank.as_ref().is_some_and(|bank| bank.slots > bank.floor())
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
    banked: Option<C>,
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
        let banked = draft.load.bank.is_some();
        let held = match banked {
            true => &mut self.banked,
            false => &mut self.whole,
        };
        match held {
            Some(held) if banked => {
                let charge = held.charge().at_bank_slots(&draft.load).map_err(PlanError::ResourcePlanning)?;
                Ok(held.clone().at(charge))
            }
            Some(held) => Ok(held.clone()),
            None => Ok(held.insert(C::of(definition, draft)?).clone()),
        }
    }

    /// The charges themselves, without what else their certification gave.
    pub fn charges(&self) -> HeldCharges {
        HeldCharges {
            banked: self.banked.as_ref().map(|held| *held.charge()),
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

/// The most of `floor..=top` that `admits`, which admits `floor` and a
/// prefix of the range, searched from `guess` (in the range): by steps
/// doubling away from it until one bounds the answer, then by halving. The
/// last count admitted is the one returned: each admitted probe lies above
/// every earlier one.
fn most_admitted(
    floor: u64,
    top: u64,
    guess: u64,
    mut admits: impl FnMut(u64) -> Result<bool, PlanError>,
) -> Result<u64, PlanError> {
    // `low` is admitted; `high` is past `top` or not admitted.
    let (mut low, mut high) = (floor, top + 1);
    if guess > floor {
        match admits(guess)? {
            true => low = guess,
            false => high = guess,
        }
    }
    let mut step = 1;
    if low == guess {
        while let Some(probe) = low.checked_add(step).filter(|&probe| probe < high) {
            if !admits(probe)? {
                high = probe;
                break;
            }
            (low, step) = (probe, step * 2);
        }
    } else {
        while let Some(probe) = high.checked_sub(step).filter(|&probe| probe > low) {
            if admits(probe)? {
                low = probe;
                break;
            }
            (high, step) = (probe, step * 2);
        }
    }
    while high - low > 1 {
        let probe = low + (high - low) / 2;
        match admits(probe)? {
            true => low = probe,
            false => high = probe,
        }
    }
    Ok(low)
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
    use super::most_admitted;

    /// From any guess, the search finds the most admitted count, the last
    /// count it admits is that one, and an exact guess takes two probes.
    #[test]
    fn the_most_admitted_count_is_found_from_any_guess() {
        let (floor, top) = (3, 40);
        for most in floor..=top {
            for guess in floor..=top {
                let (mut probes, mut last) = (0, None);
                let found = most_admitted(floor, top, guess, |count| {
                    probes += 1;
                    let admitted = count <= most;
                    if admitted {
                        last = Some(count);
                    }
                    Ok(admitted)
                })
                .unwrap();
                assert_eq!(found, most, "guess {guess}");
                assert!(last.is_none_or(|last| last == most) && (last.is_some() || most == floor));
                assert!(probes <= 2 + 2 * (top - floor).ilog2() as usize, "{probes} probes");
                if guess == most && most > floor && most < top {
                    assert_eq!(probes, 2);
                }
            }
        }
    }
}
