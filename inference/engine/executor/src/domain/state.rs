//! state lifecycle for the executor domain.

use super::*;
use crate::planning::graph_demand;
use magnitude_state::{
    GrowthChoice, Holder, RowDemand, ShrinkPolicy, StateHoldingCensus, StoreCopy,
};

/// The storage of `kind` this executor places first: its store.
pub(super) fn storage_store(topology: &seismic::DeviceTopology, executor: &seismic::Executor, kind: seismic::Contract) -> Option<seismic::StoreId> {
    executor.paths(seismic::Placing::Fixed(kind)).first().and_then(|path| topology.access_path(*path)).map(|path| path.store())
}

/// What the process holds at one moment, read on each side of a release.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryMoment {
    /// What the device's allocations charge on its store.
    pub charged: u64,
    /// What the device claims there: its allocations, or what its backend
    /// holds natively where that is more (block slack, recorded graphs). A
    /// release of backend holdings shows here and not in `charged`.
    pub claimed: u64,
    /// The process footprint, where the platform reports one.
    pub footprint: Option<u64>,
}

/// What a shrink of the state's surplus came to ([`ExecutorDomain::shrink_state`]):
/// the physical bytes released at once, and the idle graph slots' release
/// off the submitting thread, which a later boundary finishes
/// (`ExecutorDomain::advance_slot_release`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateShrink {
    pub released: u64,
    pub slots: SlotRelease,
}

/// Where the idle graph slots' release stands after a shrink of the state's
/// surplus.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotRelease {
    /// No slot was idle.
    None,
    /// This shrink began releasing them.
    Began,
    /// One begun before still runs: this shrink released nothing more.
    Running,
}

/// A comparison with Seismic's logical allocation charge. `unattributed`
/// remains explicit for storage without a classified owner. `stores` also
/// includes physical backend holdings (block slack and native resources),
/// classified as compute.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryChargeReconciliation {
    pub stores: Vec<StoreAllocation>,
    pub charged: u64,
    pub target_state: StateHoldingCensus,
    pub head_state: Option<StateHoldingCensus>,
    pub graph_pools: u64,
    pub owned_features: u64,
    pub target_weights: u64,
    pub optional_weights: u64,
    pub prepared_programs: u64,
    pub bound_constants: u64,
    pub external_pins: u64,
    pub unattributed: u64,
}

impl MemoryChargeReconciliation {
    pub fn complete(&self) -> bool {
        self.unattributed == 0
    }
}

/// One store's charge by the class the census reports it in (integration
/// spec §9.2).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StoreAllocation {
    pub store: seismic::StoreIdentity,
    /// Target weights, bound graph constants, the pristine recurrent seed
    /// and host-resident tables.
    pub model_bytes: u64,
    /// Per-conversation state and the features it owns, external views of
    /// released state included.
    pub context_bytes: u64,
    /// Every charge no classified holder makes: prepared programs, graph
    /// pools and the backend's own holdings (block slack, native resources).
    pub compute_bytes: u64,
    /// Optional components (MTP head, vision), resident or dormant.
    pub auxiliary_bytes: u64,
}

/// The class of a holder the census visits. Compute has no holder: it is
/// what a store is charged beyond every visited holder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HolderClass {
    Model,
    Context,
    Auxiliary,
}

/// What visited holders hold in one store, by class.
#[derive(Clone, Copy, Debug, Default)]
struct HeldBytes {
    model: u64,
    context: u64,
    auxiliary: u64,
}

impl HeldBytes {
    fn of(&mut self, class: HolderClass) -> &mut u64 {
        match class {
            HolderClass::Model => &mut self.model,
            HolderClass::Context => &mut self.context,
            HolderClass::Auxiliary => &mut self.auxiliary,
        }
    }
}

/// The one classification of a device's charges: every holder's physical
/// allocations, each counted once whatever holds it, by store and class.
/// Holders are visited through callbacks that return nothing, so the
/// census keeps the first failure and [`StoreCensus::finish`] reports it.
#[derive(Default)]
struct StoreCensus {
    seen: std::collections::BTreeSet<u64>,
    stores: std::collections::BTreeMap<seismic::StoreId, HeldBytes>,
    failure: Option<String>,
}

impl StoreCensus {
    fn fail(&mut self, failure: impl Into<String>) {
        self.failure.get_or_insert_with(|| failure.into());
    }

    fn observe(&mut self, observer: &seismic::TensorStorageObserver, class: HolderClass) {
        if !self.seen.insert(observer.identity()) {
            return;
        }
        for (store, bytes) in observer.store_charges().into_iter().flatten() {
            let held = self.stores.entry(store).or_default().of(class);
            match held.checked_add(bytes) {
                Some(total) => *held = total,
                None => self.fail("store census overflows"),
            }
        }
    }

    fn tensor(&mut self, tensor: &Tensor, class: HolderClass) {
        self.observe(&tensor.observe_storage(), class);
    }

    /// Move the permanent seed's bytes `charges`, part of its state store's
    /// context storage, to the model class.
    fn seed(&mut self, charges: Result<Vec<(seismic::StoreId, u64)>, String>) {
        let charges = match charges {
            Ok(charges) => charges,
            Err(error) => return self.fail(error),
        };
        for (store, bytes) in charges {
            let Some(held) = self.stores.get_mut(&store) else {
                return self.fail("seed is absent from its state store");
            };
            match (
                held.context.checked_sub(bytes),
                held.model.checked_add(bytes),
            ) {
                (Some(context), Some(model)) => {
                    held.context = context;
                    held.model = model;
                }
                _ => return self.fail("seed exceeds its state store's charge"),
            }
        }
    }

    /// Each store `device` charges, its compute bytes what no visited holder
    /// holds there.
    fn finish(mut self, device: &seismic::Device) -> Result<Vec<StoreAllocation>, String> {
        if let Some(failure) = self.failure {
            return Err(failure);
        }
        let topology = device.topology();
        let allocations = device
            .ledgers()
            .own_claims()
            .into_iter()
            .map(|(store, charged)| {
                let held = self.stores.remove(&store).unwrap_or_default();
                let classified = held
                    .model
                    .checked_add(held.context)
                    .and_then(|bytes| bytes.checked_add(held.auxiliary))
                    .ok_or("store census overflows")?;
                Ok(StoreAllocation {
                    store: topology
                        .store_identity(store)
                        .expect("charged store belongs to the opened topology"),
                    model_bytes: held.model,
                    context_bytes: held.context,
                    compute_bytes: charged
                        .checked_sub(classified)
                        .ok_or("classified bytes exceed the store's charge")?,
                    auxiliary_bytes: held.auxiliary,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        if !self.stores.is_empty() {
            return Err("a holder's allocation is in a store the device does not charge".into());
        }
        Ok(allocations)
    }
}

impl<F: ProgramFamily> ExecutorDomain<F> {
    pub(super) fn allocate_boundary(&mut self) -> Result<FeatureRef, DomainError> {
        let head = self.definition.head.as_ref();
        let decoder = &self.definition.decoder;
        let width = head.map_or(decoder.hidden, |head| head.conditioning_width(decoder));
        let element = if head.is_some_and(|head| head.residual_conditioning()) {
            seismic::Element::f32()
        } else {
            seismic::Element::dense(crate::planning::activation_dtype(decoder.activation_dtype))
        };
        let bytes = self.head_conditioning_bytes() as u64;
        let mut held = self.claim_growth(
            crate::memory::ledger::CommittedKind::Component,
            crate::memory::ClaimDemand { device: bytes, ..Default::default() },
        )?;
        let tensor = Tensor::zeros(self.domain.device(), element, &[1, width]);
        let tensor = tensor.map_err(|error| DomainError::Input(error.to_string()))?;
        held.allocated();
        self.domain
            .publish_features(tensor, held.into_claims())
            .map_err(|error| DomainError::Input(error.to_string()))
    }

    /// Shrink compactions of the target and head stores.
    pub fn state_compactions(
        &self,
    ) -> (
        magnitude_state::Compactions,
        Option<magnitude_state::Compactions>,
    ) {
        (
            self.target_store.compactions(),
            self.head_store.as_ref().map(|store| store.compactions()),
        )
    }
    /// Whether no request is resident: only then may the optional
    /// components, which its steps draft and encode with, be released.
    pub(super) fn optional_idle(&self) -> bool {
        self.target.is_empty() && self.head.is_empty() && self.input.is_empty()
    }

    /// Release the bound optional `components` after the owner has drained
    /// all requests; the queued steps, which draft with them, are orphaned
    /// first. The measured ledger delta is the only released-byte credit;
    /// shared target tensors remain cached by the head loader.
    pub fn release_idle_optional_components(
        &mut self,
        bindings: &mut StateBindings,
        components: &[super::OptionalComponent],
    ) -> Result<u64, String> {
        if !self.optional_idle() || components.is_empty() {
            return Ok(0);
        }
        self.orphan_queue(bindings)
            .map_err(|error| error.to_string())?;
        let before = self
            .domain
            .device()
            .device_ledger()
            .charged;
        for &component in components {
            match component {
                super::OptionalComponent::Drafter => {
                    self.unbind(component);
                    self.head_loader.as_ref().map(|loader| loader.unload())
                }
                super::OptionalComponent::Vision => {
                    self.unbind(component);
                    self.vision_loader.as_ref().map(|loader| loader.unload())
                }
            };
        }
        let after = self
            .domain
            .device()
            .device_ledger()
            .charged;
        before
            .checked_sub(after)
            .ok_or_else(|| "optional component release increased Seismic's charge".to_owned())
    }

    pub fn reconcile_memory_charge(
        &self,
        retained: &[&ResumeState],
    ) -> Result<MemoryChargeReconciliation, String> {
        let (target_state, head_state) = self.state_holding_census(retained)?;
        let state = target_state
            .total()
            .checked_add(head_state.map_or(0, StateHoldingCensus::total))
            .ok_or("state charge sum overflows")?;
        // The graph pools' storage, and what the device's graph budget
        // costs, which its slots' holds charge to the ledger.
        let device = self.domain.device();
        let graph_holds = device.graph_cost().bytes(device.graph_budget()).ok_or("graph budget charge overflows")?;
        let graph_pools = self
            .resources
            .committed_bytes()?
            .checked_add(graph_holds.device)
            .and_then(|bytes| bytes.checked_add(graph_holds.upload))
            .ok_or("graph charge sum overflows")?;
        let owned_features = self
            .domain
            .owned_feature_bytes()
            .map_err(|error| error.to_string())?;
        let target_weights = self.family.target_weight_bytes()?;
        // Vision has its own cache; the head shares the target's.
        let head_weights = self.head_weight_bytes()?;
        let vision_weights = self
            .vision_loader
            .as_ref()
            .map(|loader| loader.own_bytes())
            .transpose()?
            .unwrap_or(0);
        let optional_weights = head_weights
            .checked_add(vision_weights)
            .ok_or("optional resident charge overflows")?;
        let prepared_programs = self.family.prepared_program_bytes();
        let bound_constants = self.family.bound_constant_bytes()?;
        let external_pins = self
            .target_store
            .external_pinned_bytes()
            .map_err(|error| error.to_string())?
            .checked_add(
                self.head_store
                    .as_ref()
                    .map(|store| store.external_pinned_bytes())
                    .transpose()
                    .map_err(|error| error.to_string())?
                    .unwrap_or(0),
            )
            .ok_or("external pin charge overflows")?;
        let classified = state
            .checked_add(graph_pools)
            .and_then(|bytes| bytes.checked_add(owned_features))
            .and_then(|bytes| bytes.checked_add(target_weights))
            .and_then(|bytes| bytes.checked_add(optional_weights))
            .and_then(|bytes| bytes.checked_add(prepared_programs))
            .and_then(|bytes| bytes.checked_add(bound_constants))
            .and_then(|bytes| bytes.checked_add(external_pins))
            .ok_or("classified charge sum overflows")?;
        let charged = device
            .ledgers()
            .own
            .iter()
            .map(|(_, ledger)| ledger.charged)
            .sum::<u64>();
        let unattributed = charged
            .checked_sub(classified)
            .ok_or("classified holdings exceed Seismic's charge")?;
        let stores = self.store_census()?;
        Ok(MemoryChargeReconciliation {
            stores,
            charged,
            target_state,
            head_state,
            graph_pools,
            owned_features,
            target_weights,
            optional_weights,
            prepared_programs,
            bound_constants,
            external_pins,
            unattributed,
        })
    }

    /// Every store the device charges, by class: the one place a holder is
    /// assigned its class.
    fn store_census(&self) -> Result<Vec<StoreAllocation>, String> {
        let mut census = StoreCensus::default();
        self.family
            .visit_model_storage(&mut |tensor| census.tensor(tensor, HolderClass::Model));
        for store in std::iter::once(&self.target_store).chain(&self.head_store) {
            store.visit_storage(&mut |observer| census.observe(observer, HolderClass::Context));
            census.seed(
                store
                    .seed_store_charges()
                    .map_err(|error| error.to_string()),
            );
        }
        self.domain.visit_owned_feature_storage(&mut |observer| {
            census.observe(observer, HolderClass::Context)
        });
        if let Some(loader) = &self.head_loader {
            loader.visit_storage(&mut |tensor| census.tensor(tensor, HolderClass::Auxiliary));
        }
        if let Some(loader) = &self.vision_loader {
            loader.visit_storage(&mut |tensor| census.tensor(tensor, HolderClass::Auxiliary));
        }
        census.finish(self.domain.device())
    }

    /// Account for each state arena from resident requests, retained resume
    /// states, and the store's registered submitted transactions.
    /// Unregistered omissions remain an error rather than being mislabeled as
    /// surplus.
    pub fn state_holding_census(
        &self,
        retained: &[&ResumeState],
    ) -> Result<(StateHoldingCensus, Option<StateHoldingCensus>), String> {
        let target_live = self.target.values().map(Holder::State).collect::<Vec<_>>();
        let target_retained = retained
            .iter()
            .map(|state| Holder::Checkpoint(&state.target))
            .collect::<Vec<_>>();
        let target = self
            .target_store
            .holding_census(&target_live, &target_retained, &[])
            .map_err(|error| error.to_string())?;
        let head = self
            .head_store
            .as_ref()
            .map(|store| {
                let live = self.head.values().map(Holder::State).collect::<Vec<_>>();
                let retained = retained
                    .iter()
                    .map(|state| {
                        state
                            .head
                            .as_ref()
                            .map(Holder::Checkpoint)
                            .ok_or("retained resume state has no head state")
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                store
                    .holding_census(&live, &retained, &[])
                    .map_err(|error| error.to_string())
            })
            .transpose()?;
        Ok((target, head))
    }

    /// Commit state backing for a launch before its capacity check: the rows
    /// and successor banks its operations will claim, placed so histories
    /// keep growing in place. A refused minimum grant returns a byte deficit.
    /// Histories at the segment limit are repacked first. With a `ceiling`,
    /// growth that would take the stores' committed state past it grows
    /// nothing and returns [`DomainError::StateCeiling`].
    pub fn provision(
        &mut self,
        bindings: &mut StateBindings,
        operations: &[Operation],
        ceiling: Option<u64>,
    ) -> Result<(), DomainError> {
        fn primed(operation: &Operation) -> Option<&Priming> {
            match operation {
                Operation::Forward { prime, .. } => prime.as_ref(),
                _ => None,
            }
        }
        // A group claiming the front queued step needs no rows of its own.
        // The step it queues next is optional and is queued only into space
        // already bound: growing now would change bindings under the claimed
        // step. Any other group resolves the queue first: its tentative rows
        // would otherwise read as another history's in every layout
        // question below (blocked tails, in-place rows, growth).
        if self.claim_slots(operations).is_some() {
            return Ok(());
        }
        self.orphan_queue(bindings)?;
        self.close_group()?;
        self.open_group(operations)?;
        for operation in operations {
            if let Operation::Forward { request, .. } | Operation::Head { request, .. } = operation
            {
                self.relocate_tails(bindings, *request)?;
            }
        }
        let (mut target, mut target_queued) = (Vec::new(), Vec::new());
        let (mut head, mut head_queued) = (Vec::new(), Vec::new());
        // The steps queued behind the group (`behind`, within the arena's
        // places) each follow the one before by their rows. A step takes a
        // successor bank, and the steps behind it one more: a pipelined
        // successor its own, a chain's chunks the two they alternate
        // (`OwnedChainedAdvance`).
        let behind = self.behind(operations);
        // The rows each sequence reaches through this group
        // (`claimed_through`), committed on the ledger before any store
        // grows.
        let method = self.execution.policy().method();
        let mut state_rows = Vec::new();
        let mut target_banks = 0usize;
        let mut head_banks = 0usize;
        for operation in operations {
            let rows = operation.row_count();
            match operation {
                Operation::Forward { request, prime, .. } => {
                    let queued = behind.of(*request).collect::<Vec<_>>();
                    // A primed chunk's drafter entry, and the entries of the
                    // chunks queued behind it. Queued rows are backed now and
                    // admitted by their own steps' successors.
                    let entry = prime.as_ref().map_or(0, |prime| prime.tokens.len());
                    let entries = std::iter::once(prime.as_ref())
                        .chain(queued.iter().copied().map(primed))
                        .flatten()
                        .map(|prime| prime.tokens.len())
                        .collect::<Vec<_>>();
                    if let (false, Some(state)) = (entries.is_empty(), self.head.get(request)) {
                        let ahead = entries.iter().sum::<usize>() - entry;
                        head.extend(state.demands(entry));
                        head_queued.extend(state.queued_demands(entry, ahead));
                        head_banks += entries.len();
                    }
                    if let Some(state) = self.target.get(request) {
                        let ahead = queued.iter().copied().map(Operation::row_count).sum::<usize>();
                        state_rows.push((*request, claimed_through(state.position(), rows + ahead, method)));
                        target.extend(state.demands(rows));
                        target_queued.extend(state.queued_demands(rows, ahead));
                        target_banks += 1 + queued.len().min(1);
                    }
                }
                Operation::Head { request, .. } => {
                    if let Some(state) = self.head.get(request) {
                        head.extend(state.demands(rows));
                        head_banks += 1;
                    }
                }
                _ => {}
            }
        }
        if let Some(ceiling) = ceiling {
            let mut growth = 0u64;
            if target_banks != 0 {
                growth = bindings
                    .store(false)?
                    .growth_claim(&target, &target_queued, target_banks)?
                    .minimum_bytes;
            }
            if head_banks != 0 && self.head_store.is_some() {
                growth = growth
                    .checked_add(
                        bindings
                            .store(true)?
                            .growth_claim(&head, &head_queued, head_banks)?
                            .minimum_bytes,
                    )
                    .ok_or_else(|| DomainError::invariant("state growth overflows"))?;
            }
            let committed = self.committed_state_bytes();
            if growth != 0 && committed.saturating_add(growth) > ceiling {
                return Err(DomainError::StateCeiling {
                    committed,
                    growth,
                    ceiling,
                });
            }
        }
        for (sequence, through) in state_rows {
            self.claim_state_rows(sequence, through)?;
        }
        // Stores without history domains still grow banks.
        if target_banks != 0 {
            self.grant_state_growth(bindings, false, &target, &target_queued, target_banks)?;
        }
        if head_banks != 0 && self.head_store.is_some() {
            self.grant_state_growth(bindings, true, &head, &head_queued, head_banks)?;
        }
        self.settle_state_slabs();
        let lane = match operations.first() {
            Some(Operation::Forward { .. }) => ReservationLane::Target,
            Some(Operation::Head { .. }) => ReservationLane::Head,
            Some(Operation::Encode { image, .. }) => {
                let class = self
                    .image_class(image.patches())
                    .map_err(DomainError::Input)?;
                return self.provision_vision(class as u64);
            }
            None => return Ok(()),
        };
        self.provision_output(lane, super::projected_rows(operations))
    }

    /// The state the stores commit: every history and bank slab they back,
    /// whether a holder references it or not.
    pub fn committed_state_bytes(&self) -> u64 {
        self.target_store.committed_bytes()
            + self
                .head_store
                .as_ref()
                .map_or(0, |store| store.committed_bytes())
    }

    /// Grow the vision storage of `patch_rows`'s class by an activation and
    /// an output slot when none is free: an encode claims both of its image's
    /// class, an activation with the memory its encode's graph takes once
    /// recorded ([`graph_demand`]). Each slot is claimed first like any
    /// growth ([`Self::claim_growth`]) and keeps its claim until released;
    /// a refusal is a demand deficit for the owner's release order.
    fn provision_vision(&mut self, patch_rows: u64) -> Result<(), DomainError> {
        let pool = self
            .resources
            .vision_graph()
            .ok_or_else(|| DomainError::invariant("vision graph is absent"))?;
        let device = self.domain.device();
        let activation = (pool.available_workspace(patch_rows) == 0)
            .then(|| {
                let graphs = graph_demand(device.graph_cost(), device.graph_budget(), pool.activation_graphs(patch_rows)?).map_err(DomainError::invariant)?;
                crate::memory::ClaimDemand { device: pool.activation_bytes(patch_rows)?, ..Default::default() }
                    .checked_plus(graphs)
                    .ok_or_else(|| DomainError::invariant("a vision activation's demand overflows"))
            })
            .transpose()?;
        let output = (pool.available_output(patch_rows) == 0)
            .then(|| pool.output_bytes(patch_rows))
            .transpose()?
            .map(|bytes| crate::memory::ClaimDemand { device: bytes, ..Default::default() });
        if let Some(demand) = activation {
            self.grow_vision(demand, |pool, held| pool.grow_activation(patch_rows, held))?;
        }
        if let Some(demand) = output {
            self.grow_vision(demand, |pool, held| pool.grow_output(patch_rows, held))?;
        }
        Ok(())
    }

    fn grow_vision(
        &mut self,
        demand: crate::memory::ClaimDemand,
        grow: impl FnOnce(&mut crate::VisionGraphPool, crate::memory::ledger::Held) -> Result<(), crate::resources::VisionGrowth>,
    ) -> Result<(), DomainError> {
        let bytes = demand.bytes();
        let held = self.claim_growth(crate::memory::ledger::CommittedKind::Component, demand)?;
        let grown = grow(
            self.resources
                .vision_graph_mut()
                .expect("the vision graph was found above"),
            held,
        );
        grown.map_err(|growth| match growth {
            crate::resources::VisionGrowth::Class(error) => {
                DomainError::invariant(format!("vision class was not prepared: {error}"))
            }
            crate::resources::VisionGrowth::Device(error) => {
                growth_failure(error, bytes, "vision growth")
            }
        })
    }

    /// Grow `lane`'s retained-output pool by one slot when none is free, so
    /// the launch finds its output: a request keeps its last published
    /// output after its launch completes, so the pool follows the live
    /// requests. A target launch's pool is of the readout class that holds
    /// its `projected` rows (`AllocatedResources::retained_outputs_mut`).
    /// The slot is claimed first like any growth ([`Self::claim_growth`])
    /// and keeps its claim until released; a refusal is a demand deficit
    /// for the owner's release order.
    pub(super) fn provision_output(&mut self, lane: ReservationLane, projected: usize) -> Result<(), DomainError> {
        let Some(pool) = self.resources.retained_outputs_mut(lane, projected) else {
            return Ok(());
        };
        if pool.available_output() > 0 {
            return Ok(());
        }
        let bytes = pool.output_slot_bytes();
        let held = self.claim_growth(
            crate::memory::ledger::CommittedKind::Outputs,
            crate::memory::ClaimDemand { device: bytes, ..Default::default() },
        )?;
        self.resources
            .retained_outputs_mut(lane, projected)
            .expect("the pool was found above")
            .grow_claimed_output(held)
            .map_err(|error| growth_failure(error, bytes, "graph output growth"))
    }

    /// Commit `sequence`'s state rows through row `through` on the device's
    /// ledger (`Need::StateRows`), growing its claim by the pages its rows
    /// past those it holds occupy: a provision repeated, or one whose rows its sequence
    /// rewinds below, takes nothing again. Where the ledger names sequences
    /// of this load to preempt, their bytes are reserved to it and this
    /// returns [`DomainError::Preempt`]; once the owner has evicted them
    /// (each dropping its claim), the next provision completes the growth it
    /// waited for. A preemption whose sequences still stand is let go.
    /// Refused past the budget, or where the ledger names sequences this
    /// load cannot evict, nothing is taken.
    fn claim_state_rows(&mut self, sequence: RequestId, through: u64) -> Result<(), DomainError> {
        use crate::memory::ledger::{Need, Refused};
        if let Some((waiting, waited, preemption)) = self.preempting.take() {
            let evicted = preemption.sequences().iter().all(|named| !self.state_claims.contains_key(named));
            match self.state_claims.get_mut(&waiting) {
                Some(state) if evicted => {
                    state.claim.complete(preemption);
                    state.through = state.through.max(waited);
                }
                // Its sequences stand, or its own closed: let go.
                _ => drop(preemption),
            }
        }
        let state = self
            .state_claims
            .get_mut(&sequence)
            .ok_or_else(|| DomainError::invariant("a resident sequence holds no state claim"))?;
        if through <= state.through {
            return Ok(());
        }
        match state.claim.grow(Need::StateRows { sequence, from: state.through, to: through }) {
            Ok(()) => {
                state.through = through;
                Ok(())
            }
            Err(Refused::Short(short)) => Err(DomainError::short(short, self.resources.ledgers())),
            Err(Refused::Preempt(preemption)) => {
                let sequences = preemption.sequences().to_vec();
                if !sequences.iter().all(|named| self.state_claims.contains_key(named)) {
                    // Another load's sequences: this load cannot evict them,
                    // so the growth is short of what their state would give.
                    return Err(DomainError::short(preemption.short(), self.resources.ledgers()));
                }
                self.preempting = Some((sequence, through, preemption));
                Err(DomainError::Preempt(sequences))
            }
        }
    }

    /// Let go of `sequence`'s state claim, as it closes or is evicted, with
    /// the preemption its growth waits on: kept, that preemption would hold
    /// what it took and the bytes of every sequence it named as each left,
    /// for a growth no provision completes.
    pub(super) fn drop_state_claim(&mut self, sequence: RequestId) {
        self.state_claims.remove(&sequence);
        if self.preempting.as_ref().is_some_and(|(waiting, ..)| *waiting == sequence) {
            self.preempting = None;
        }
    }

    /// Grant elastic state growth of the target or `head` store from a fresh
    /// domain observation. Growth changes bindings, so the queued steps are
    /// orphaned first; its release may itself free what the growth needed.
    /// Seismic's charge ledger remains authoritative: its limit enforces the
    /// grant on each added slab allocation. The slabs' bytes are the rows
    /// the sequences claimed ([`Self::claim_state_rows`]), allocated on
    /// their claims, and only what they add past those is held anew
    /// ([`StateSlabs`]). `queued` are the rows of the
    /// steps queued behind the group's, backed now and admitted by their own
    /// steps.
    pub(super) fn grant_state_growth(
        &mut self,
        bindings: &mut StateBindings,
        head: bool,
        demands: &[RowDemand],
        queued: &[RowDemand],
        banks: usize,
    ) -> Result<(), DomainError> {
        // No backing changes when neither plan adds a slab: the group's
        // demands only admit slabs already backed, so the charge stands as
        // last read.
        let store = bindings.store(head)?;
        let claim = store.growth_claim(demands, queued, banks)?;
        if claim.minimum_bytes == 0 && claim.preferred_bytes == 0 {
            store.provision_with_growth(demands, queued, banks, GrowthChoice::Minimum)?;
            self.settle_state_slabs();
            return Ok(());
        }
        for choice in [GrowthChoice::Preferred, GrowthChoice::Minimum] {
            let claim = store.growth_claim(demands, queued, banks)?;
            let required = match choice {
                GrowthChoice::Preferred => claim.preferred_bytes,
                GrowthChoice::Minimum => claim.minimum_bytes,
            };
            // What the slabs add past the rows the sequences claimed is held
            // before the store performs its fallible physical operation;
            // the preferred growth falls back to the minimum where it is
            // not.
            let ledger = self.resources.ledgers().device();
            let claims = self.state_claims.values().map(|state| &state.claim);
            if let Err(short) = self.state_slabs.reserve(ledger, required, claims) {
                if choice == GrowthChoice::Preferred {
                    continue;
                }
                return Err(DomainError::short(short, self.resources.ledgers()));
            }
            let grown = store.provision_with_growth(demands, queued, banks, choice);
            self.settle_state_slabs();
            match grown {
                Ok(()) => return Ok(()),
                Err(error)
                    if choice == GrowthChoice::Preferred
                        && matches!(
                            &error,
                            magnitude_state::Error::Tensor(seismic::TensorError::Execution(execution))
                                if execution.capacity_refusal().is_some()
                                    || matches!(execution, seismic::ExecutionError::AllocationFailed(_))
                        ) =>
                {
                    continue;
                }
                Err(error) => return Err(error.into()),
            }
        }
        unreachable!("minimum state growth either succeeds or returns a deficit")
    }

    /// Claim `demand`, the new charge of one allocation of `kind`, by the
    /// device's ledgers as they read last
    /// (`memory::ledger::DeviceLedgers::claim`), before the allocation is
    /// made; its holder marks it allocated once made and keeps it until the
    /// allocation is freed. Refused as a capacity deficit of the store short
    /// of it, for the owner's release order, or as blind where that store's
    /// reading failed.
    pub(super) fn claim_growth(
        &self,
        kind: crate::memory::ledger::CommittedKind,
        demand: crate::memory::ClaimDemand,
    ) -> Result<crate::memory::ledger::Held, DomainError> {
        let ledgers = self.resources.ledgers();
        ledgers.claim(self.domain.device(), kind, demand).map_err(|short| DomainError::short(short, ledgers))
    }

    /// Hold a head just bound for the class held, what it demands beyond
    /// the floor charged with its weights
    /// (`ProgramFamily::bound_head_demand`) claimed first
    /// ([`Self::claim_growth`]) and kept with its binding. A refused claim
    /// unbinds the head.
    pub(super) fn hold_bound_head(&mut self) -> Result<(), DomainError> {
        let places = self.resources.group_places().get();
        let demand = self.family.bound_head_demand(places).map_err(DomainError::Input)?;
        if demand == crate::memory::ClaimDemand::default() {
            return Ok(());
        }
        let mut held = match self.claim_growth(crate::memory::ledger::CommittedKind::Component, demand) {
            Ok(held) => held,
            Err(refused) => {
                self.unbind(super::OptionalComponent::Drafter);
                return Err(refused);
            }
        };
        if let Err(error) = self.family.hold_bound_head(places) {
            self.unbind(super::OptionalComponent::Drafter);
            return Err(DomainError::invariant(error));
        }
        held.allocated();
        self.head_binding.extend(held.into_claims());
        Ok(())
    }

    /// Unbind `component`, releasing what its binding claimed.
    fn unbind(&mut self, component: super::OptionalComponent) {
        self.family.unbind_optional(component);
        match component {
            super::OptionalComponent::Drafter => self.head_binding.clear(),
            super::OptionalComponent::Vision => self.vision_binding.clear(),
        }
    }


    /// Before a request's next launch reserves rows, relocate the partial
    /// last page of each of its target and head histories that a sibling
    /// fork continued (`magnitude_state::OwnedTailRelocation`), so every
    /// history keeps growing within its span limit. A relocation copies
    /// less than one page into one free page; without memory for that page
    /// the request fails with a capacity error before any launch.
    /// Moving rows changes bindings: the queued steps are orphaned before
    /// the first relocation.
    pub(super) fn relocate_tails(
        &mut self,
        bindings: &mut StateBindings,
        request: RequestId,
    ) -> Result<(), DomainError> {
        if !self.family.state_is_bound() {
            return Ok(());
        }
        self.relocate_store_tails(bindings, request, false)?;
        self.relocate_store_tails(bindings, request, true)
    }

    fn relocate_store_tails(
        &mut self,
        bindings: &mut StateBindings,
        request: RequestId,
        head: bool,
    ) -> Result<(), DomainError> {
        let store = if head {
            match self.head_store.clone() {
                Some(store) => store,
                None => return Ok(()),
            }
        } else {
            self.target_store.clone()
        };
        // `provision` resolved the queued steps: only another history can
        // block.
        let Some(state) = self.lane_states(head).get(&request) else {
            return Ok(());
        };
        let demands = state.relocation_demands();
        if demands.is_empty() {
            return Ok(());
        }
        self.grant_state_growth(bindings, head, &demands, &[], 0)?;
        for demand in demands {
            let state = self
                .lane_states(head)
                .remove(&request)
                .expect("state checked above");
            let relocation = match OwnedTailRelocation::prepare(state, demand.domain) {
                Ok(relocation) => relocation,
                Err((state, error)) => {
                    self.lane_states(head).insert(request, state);
                    return Err(error.into());
                }
            };
            if let Err(error) = self.submit_store_copy(&store, relocation.copy()) {
                self.lane_states(head).insert(request, relocation.abort());
                return Err(error);
            }
            self.lane_states(head).insert(request, relocation.commit());
        }
        Ok(())
    }

    /// The accepted states of the head or target lane.
    fn lane_states(&mut self, head: bool) -> &mut BTreeMap<RequestId, SequenceState> {
        if head {
            &mut self.head
        } else {
            &mut self.target
        }
    }

    /// Commit the successor banks a newly opened request needs. Free banks
    /// change no bindings and leave the queued steps running.
    pub fn provision_open(&mut self, bindings: &mut StateBindings) -> Result<(), DomainError> {
        self.orphan_queue(bindings)?;
        let requirements = self.open_requirements();
        self.grant_state_growth(bindings, false, &[], &[], requirements.target_banks())?;
        if self.head_store.is_some() {
            self.grant_state_growth(bindings, true, &[], &[], requirements.head_banks())?;
        }
        Ok(())
    }

    /// Release empty state slabs at idle, keeping one spare per store. In
    /// Reclaim, also compact sparse occupied slabs into already held space.
    /// Idle graph slots are released off this thread, `wake` completed once they
    /// are ([`Self::release_idle_graph_slots`]). The physical bytes released
    /// now, and where the slots' release stands, which a later boundary
    /// finishes ([`Self::advance_slot_release`]). The queued steps
    /// are orphaned first: their launch bindings pin slab placement.
    pub fn shrink_state(
        &mut self,
        bindings: &mut StateBindings,
        policy: ShrinkPolicy,
        wake: crate::CompletionWake,
    ) -> Result<StateShrink, DomainError> {
        self.orphan_queue(bindings)?;
        let mut released = self.shrink_store(bindings, false, policy)?;
        if self.head_store.is_some() {
            released += self.shrink_store(bindings, true, policy)?;
        }
        self.settle_state_slabs();
        let slots = self.release_idle_graph_slots(wake);
        Ok(StateShrink { released, slots })
    }

    /// At a flight boundary: the idle graph slots' release, once the load's
    /// release thread has ended it (`AllocatedResources::released_slots`);
    /// none while it runs.
    pub fn advance_slot_release(&mut self) -> Result<Option<crate::resources::SlotsReleased>, DomainError> {
        let Some(released) = self.resources.released_slots().map_err(DomainError::invariant)? else {
            return Ok(None);
        };
        Ok(Some(released))
    }

    /// Claim the prompt-group class of the group opening with `operations`:
    /// a lone prompt chunk with planned chunks behind it takes a place for
    /// each, within the load's chain capacity (a chain runs over as many
    /// passes as it needs), as many as yield at once. Each
    /// count from the most down is claimed whole or not at all
    /// ([`crate::memory::ledger::Ledger::claim_units`]): its device-side
    /// bytes on the device's ledger and its host bytes on host RAM's,
    /// never preempting a sequence and never refused; with none, the group
    /// runs with one place. Its pools' slots and the family's group buffers
    /// are then held; a device that refuses them leaves one place. Held
    /// until the group closes ([`Self::close_group`]); one still held, its
    /// close waiting for a gap, serves the group as it stands. Nothing is claimed
    /// while the measurement job's claim stands, nor while a step of the
    /// bank runs (it takes a group's places only between flights).
    pub(super) fn open_group(&mut self, operations: &[Operation]) -> Result<(), DomainError> {
        // A class whose close waited for a gap between flights serves this
        // group as it stands.
        if self.resources.group_places() > std::num::NonZeroUsize::MIN {
            return Ok(());
        }
        let capacity = self.chain_capacity().get();
        let chunks = match (operations, self.behind_within(operations, capacity)) {
            ([_], super::queue::Behind::Chain { chunks }) => chunks.len() + 1,
            _ => return Ok(()),
        };
        if self.measurement_claim.is_some() {
            return Ok(());
        }
        let Some(quiet) = self.quiet() else {
            return Ok(());
        };
        // Idle slots a claim took cache of are released first
        // (`follow_cache_targets`): their memory is the claim's once freed.
        if self.idle_slots_taken() {
            return Ok(());
        }
        // A closed group's idle slots, which this group's class takes up
        // again: their cache is let go and the class's claim in each store
        // holds what idles there beside what it grows (`class_demand` counts
        // only slots not yet allocated), so it covers every slot beyond the
        // floor, as its close turns them cache again. Opened with none, they
        // idle again.
        let ledgers = self.resources.ledgers();
        let idle = [ledgers.device(), ledgers.host()].map(|ledger| {
            let store = ledger.store();
            let bytes = self.idle_class.iter().filter(|cached| cached.store() == store).map(crate::memory::ledger::Cached::bytes).sum::<u64>();
            self.idle_class.retain(|cached| cached.store() != store);
            bytes
        });
        assert!(self.idle_class.is_empty(), "invariant violated: idle slots' cache in a store the device's ledgers do not keep");
        let opened = self.claim_group_class(chunks, idle, quiet);
        if self.group_class.is_none() {
            let ledgers = self.resources.ledgers();
            for (ledger, bytes) in [ledgers.device(), ledgers.host()].into_iter().zip(idle).filter(|(_, bytes)| *bytes > 0) {
                let cached = ledger
                    .cache(crate::memory::ledger::CacheKind::GraphSlots, bytes)
                    .expect("invariant violated: the idle slots' cache let go a moment ago fits again");
                self.idle_class.push(cached);
            }
        }
        opened
    }

    /// The class of `chunks` places or fewer, the most whose claims yield
    /// at once ([`Self::open_group`]), its claims holding `idle` bytes of
    /// idle slots beside the class's growth, the device's store's and host
    /// RAM's apart.
    fn claim_group_class(&mut self, chunks: usize, idle: [u64; 2], quiet: super::Quiet) -> Result<(), DomainError> {
        let device = self.domain.device();
        for places in (2..=chunks).rev() {
            let demand = class_demand(&self.execution, &mut self.resources, &self.family, device, places)?;
            let [idle_device, idle_host] = idle;
            let beside = demand
                .checked_bytes()
                .and_then(|bytes| bytes.checked_sub(demand.host))
                .and_then(|bytes| bytes.checked_add(idle_device))
                .ok_or_else(|| DomainError::invariant("a group's class overflows"))?;
            let host = demand.host.checked_add(idle_host).ok_or_else(|| DomainError::invariant("a group's class overflows"))?;
            let ledgers = self.resources.ledgers();
            let mut yielded = true;
            let mut claims = [(ledgers.device(), beside), (ledgers.host(), host)].map(|(ledger, bytes)| {
                let unit = std::num::NonZeroU64::new(bytes)?;
                let (claim, units) = ledger.claim_units(crate::memory::ledger::CommittedKind::GroupClass, unit, 1);
                yielded &= units == 1;
                Some(claim)
            });
            if !yielded {
                continue;
            }
            let limits = self.execution.policy().limits();
            let grown = std::num::NonZeroUsize::new(places).expect("a class of two places or more");
            let held = self
                .resources
                .grow_class(grown, limits.class_slots(places))
                .and_then(|grown| self.family.hold_group(places, quiet).map(|()| grown))
                .map(|grown| self.resources.hold_class(grown));
            return match held {
                Ok(()) => {
                    // Allocated whole: the pools' slots and the group's
                    // buffers are made, those a closed group left idle
                    // among them, whose cache the claims now cover.
                    for claim in claims.iter_mut().flatten() {
                        claim.allocated(claim.bytes() - claim.allocated_bytes());
                    }
                    self.group_class = Some(GroupClass { places, claims });
                    Ok(())
                }
                Err(
                    crate::resources::AllocationError::Tensor(error)
                    | crate::resources::AllocationError::Workflow(seismic::WorkflowError::TensorView(error)),
                ) => match growth_failure(error, beside, "a prompt group's class") {
                    // The device refused the bytes: the group runs with one
                    // place, what was allocated idling until released.
                    DomainError::Capacity(_) => Ok(()),
                    error => Err(error),
                },
                Err(error) => Err(DomainError::invariant(error.to_string())),
            };
        }
        Ok(())
    }

    /// Whether a claim took any of a closed group's idle slots' cache in
    /// process: the slots then still hold memory the ledger lends, so they
    /// are released at once (`follow_cache_targets`).
    pub(super) fn idle_slots_taken(&self) -> bool {
        self.idle_class.iter().any(|cached| cached.releasing() > 0)
    }

    /// Close the group whose class is held, once no chain of it is under
    /// way: its group buffers go back to the floor's, its pools hold one
    /// place again (their slots beyond it idle until released,
    /// `AllocatedResources::release_idle_slots`), and its claims turn into
    /// the idle slots' cache (`CacheKind::GraphSlots`). Only between flights ([`Self::quiet`]); otherwise the next close
    /// does it.
    pub(super) fn close_group(&mut self) -> Result<(), DomainError> {
        if self.resources.group_places() == std::num::NonZeroUsize::MIN
            || matches!(self.queue, super::queue::Queue::Chain(_) | super::queue::Queue::Opening(_))
        {
            return Ok(());
        }
        let Some(quiet) = self.quiet() else {
            return Ok(());
        };
        // The group's buffers go back to the OS now: its claims shrink by
        // them, by the store each lies in, and what the pools keep idle
        // turns cache.
        let GroupClass { places, claims } = self.group_class.take().expect("invariant violated: a held class without its claims");
        let freed = self.family.group_demand(places).map_err(DomainError::invariant)?;
        let freed = [freed.checked_bytes().map(|bytes| bytes - freed.host), Some(freed.host)];
        self.family.hold_group(1, quiet).map_err(|error| DomainError::invariant(error.to_string()))?;
        self.resources.release_class();
        for (claim, freed) in claims.into_iter().zip(freed) {
            let freed = freed.ok_or_else(|| DomainError::invariant("a group's buffers overflow"))?;
            let Some(mut claim) = claim else {
                assert_eq!(freed, 0, "invariant violated: a group's buffers lie in a store its class claimed nothing of");
                continue;
            };
            claim.shrink(freed);
            self.idle_class.push(claim.into_cache(crate::memory::ledger::CacheKind::GraphSlots));

        }
        Ok(())
    }

    /// Claim `bytes` of device memory for the service's measurement job
    /// beside the model, in place of any claim before, which is dropped
    /// first: a committed claim on the device's ledger
    /// (`memory::ledger::Need::Measurement`), which the bank's cache yields
    /// to; and the bank's own traffic is held (`ElasticBank::hold_traffic`),
    /// so the job measures beside none of the load's copies or reads. A
    /// growth or the host tier's backing running ends at its next slot
    /// (`ElasticBank::stop_growths`): held, its slots would wait for the
    /// claim's release, which waits for the job's end. Refused, as the
    /// device's store is blind or by what committed memory it may not take
    /// exceeds the budget by (`memory::ledger::Short`), nothing is held.
    pub fn claim_measurement_room(&mut self, bytes: u64) -> Result<(), crate::memory::ledger::Short> {
        use crate::memory::ledger::{Need, Refused};
        self.measurement_claim = None;
        let claim = match self.resources.ledgers().device().claim(Need::Measurement { bytes }) {
            Ok(claim) => claim,
            Err(Refused::Short(short)) => return Err(short),
            Err(Refused::Preempt(preemption)) => {
                panic!("invariant violated: a measurement's claim waits on preempting {:?}; only state preempts", preemption.sequences())
            }
        };
        if let Some(bank) = self.family.expert_bank() {
            bank.stop_growths();
        }
        let traffic = self.hold_expert_traffic();
        self.measurement_claim = Some(MeasurementClaim { _claim: claim, _traffic: traffic });
        Ok(())
    }

    /// Hold the expert bank's own traffic until the returned hold drops
    /// (`ElasticBank::hold_traffic`): what a measurement beside the load,
    /// the job's or a tool's in its own process, holds while it runs. None
    /// without a bank, which has no traffic.
    pub fn hold_expert_traffic(&self) -> Option<TrafficHold> {
        self.family.expert_bank().map(|bank| bank.hold_traffic())
    }

    /// Whether the expert bank's traffic is held and none of it runs; a
    /// load without a bank has none.
    pub fn expert_traffic_quiet(&self) -> bool {
        self.family.expert_bank().is_none_or(|bank| bank.traffic_quiet())
    }

    /// Wait until the expert bank's traffic, held, is quiet
    /// (`ElasticBank::wait_traffic_quiet`); at once without a bank.
    pub fn wait_expert_traffic_quiet(&self) -> Result<(), DomainError> {
        self.family.expert_bank().map_or(Ok(()), |bank| bank.wait_traffic_quiet())
    }

    /// The measurement job ended: its committed claim is released, and the
    /// bank's traffic runs again.
    pub fn release_measurement_room(&mut self) {
        // Its claim and its hold of the bank's traffic drop with it.
        self.measurement_claim = None;
    }

    /// The store `kind` storage is placed in first; none where the
    /// executor places none.
    pub(super) fn store_of(&self, kind: seismic::Contract) -> Option<seismic::StoreId> {
        let device = self.domain.device();
        storage_store(device.topology(), device.info(), kind)
    }


    /// What the process holds now (`MemoryMoment`).
    pub fn memory_moment(&self) -> MemoryMoment {
        self.memory_probe().moment()
    }

    /// The places of the prompt-group class held now: one, or an open
    /// group's ([`Self::open_group`]).
    pub fn group_places(&self) -> std::num::NonZeroUsize {
        self.resources.group_places()
    }

    /// Bound one of the device's ledgers through the reader's right to
    /// bound them, the one holder while the reader lives.
    #[cfg(test)]
    pub(crate) fn bound(&self, ledger: seismic::Ledger, bound: Option<u64>) -> Result<(), seismic::LimitError> {
        self.memory.borrow().bounds().bound(ledger, bound)
    }

    /// The slots the expert bank commits now; none without an elastic bank.
    pub fn expert_bank_slots(&self) -> Option<u64> {
        self.family.expert_bank().map(|bank| bank.slots())
    }

    /// The most rows a verify carries while each streamed expert it misses
    /// is read once for all of them: the decode form's; none without an
    /// elastic bank, where every expert is resident.
    pub fn streamed_verify_rows(&self) -> Option<usize> {
        self.family
            .expert_bank()
            .map(|_| crate::operators::routed::fused_graph::DECODE_ROWS as usize)
    }

    /// Where the load's routed experts are now (`ElasticBank::residency`);
    /// none without an elastic bank.
    pub fn expert_residency(&self) -> Result<Option<super::ExpertResidency>, DomainError> {
        self.family.expert_bank().map(|bank| bank.residency()).transpose()
    }

    /// Starve the expert bank's host for `duration` from now
    /// (`ExpertBank::starve`): a test's way to end the waits of the steps
    /// it submits meanwhile before their answers.
    #[cfg(test)]
    pub(crate) fn starve_expert_host(&self, duration: std::time::Duration) {
        let starved = self.family.expert_bank().is_some_and(|bank| bank.starve(duration));
        assert!(starved, "a streamed load's host is starved");
    }

    /// At a flight boundary: advance the expert bank's running resize
    /// (`ElasticBank::advance`), `wake` completed as it progresses. A
    /// finished one is returned, its credit Seismic's measured change of the
    /// bank's charge, with what it held the submitting thread by source; a
    /// growth's heap claim is released and the static
    /// holding, which charges the bank, follows either.
    pub fn advance_expert_bank(
        &mut self,
        wake: crate::CompletionWake,
    ) -> Result<Option<(super::BankResized, super::ThreadHeld)>, DomainError> {
        let started = std::time::Instant::now();
        let Some(finished) = self.family.expert_bank().map(|bank| bank.advance(wake)).transpose()? else {
            return Ok(None);
        };
        // A cut asked beside a chain may be answered here, beginning.
        self.note_running_cut();
        let Some(finished) = finished else {
            self.resize_held.hold(started, super::HeldSource::Advance);
            return Ok(None);
        };
        let bank = self.family.expert_bank().expect("a resize finished of the load's expert bank");
        let holding = finished.holding();
        let tier = bank.holding_storage(holding)?;
        let held = self.resize_held.finish(started, tier, holding);
        Ok(Some((finished, held)))
    }

    /// Record `resize`, a finished resize of the expert bank's `holding`
    /// ([`Self::advance_expert_bank`]), as that holding's last
    /// (`ElasticBank::record_resize`).
    pub fn record_expert_resize(&self, holding: super::BankHolding, resize: super::BankResize) {
        self.family
            .expert_bank()
            .expect("a finished resize is of the load's expert bank")
            .record_resize(holding, resize);
    }

    /// Release graph activations and output slots grown beyond the arena's
    /// class that no request or launch holds, on the load's release thread
    /// (`AllocatedResources::release_idle_slots`), `wake` completed once
    /// they are released: whether a release began; while one runs nothing
    /// more is released until a boundary finishes it.
    fn release_idle_graph_slots(&mut self, wake: crate::CompletionWake) -> SlotRelease {
        if self.resources.releasing_slots() {
            return SlotRelease::Running;
        }
        match self.resources.release_idle_slots(std::mem::take(&mut self.idle_class), wake) {
            true => SlotRelease::Began,
            false => SlotRelease::None,
        }
    }

    /// Free empty slabs and compact into held slabs without a growth claim.
    /// Synchronize the static holding after Seismic reports the released bytes.
    fn shrink_store(
        &mut self,
        bindings: &mut StateBindings,
        head: bool,
        policy: ShrinkPolicy,
    ) -> Result<u64, DomainError> {
        // A live request's steps in flight (its next, and the one queued
        // behind it where the load pipelines or chains) each append a row
        // and take a successor bank: room Reclaim keeps, as no surplus.
        let steps = self.execution.policy().limits().target_successor_banks();
        let live = if head { &self.head } else { &self.target };
        let next = magnitude_state::NextSteps {
            rows: live.values().flat_map(|state| state.demands(steps)).collect(),
            banks: steps * live.len(),
        };
        let bindings = bindings.store(head)?;
        let store = Rc::clone(bindings);
        let released =
            bindings.shrink_keeping(policy, &next, |_, copy| self.submit_store_copy(&store, &copy));
        released
    }

    fn submit_store_copy(
        &mut self,
        store: &Rc<StateStore>,
        copy: &StoreCopy,
    ) -> Result<(), DomainError> {
        let max_rows = self.execution.policy().limits().max_launch_rows;
        for start in (0..copy.rows()).step_by(max_rows) {
            let end = (start + max_rows).min(copy.rows());
            let chunk = copy.chunk(start, end);
            let workspace = self
                .resources
                .state_graph()
                .acquire_workspace()
                .map_err(DomainError::Capacity)?;
            let class_rows = crate::batching::row_class(chunk.rows())
                .ok_or_else(|| DomainError::invariant("store copy has no prepared row class"))?;
            let batch = crate::batching::ValidatedStateBatch::copy(
                chunk.copies().to_vec(),
                chunk.row_capacity(),
                class_rows,
            )
            .map_err(|error| DomainError::invariant(error.to_string()))?;
            let inputs = StateLaunchInputs::new(batch, StateWork::StoreCopy(chunk), workspace);
            let launch = ValidatedStateLaunch::new(inputs, store, None, self.domain.id())
                .map_err(|(_, error)| DomainError::invariant(error.to_string()))?;
            let submission = self
                .family
                .submit_state(launch)
                .map_err(|(error, _)| DomainError::from(error))?;
            submission.finish().map_err(DomainError::Device)?;
        }
        Ok(())
    }
}

/// The row a sequence's state claim reaches through a group that advances
/// it `rows` (its own and those queued behind it) from `position`: those
/// rows, and the most speculative rows its drafter adds past them
/// (`PlannedMethod::draft_rows`, none undrafted), so no row a step writes
/// is outside its claim.
pub(crate) fn claimed_through(position: usize, rows: usize, method: crate::PlannedMethod) -> u64 {
    (position + rows + method.draft_rows()) as u64
}

/// A prompt group's class held beyond one place ([`ExecutorDomain::open_group`]):
/// its places, and its committed claims, of its device-side bytes on the
/// device's ledger and of its host bytes on host RAM's (none of a store it
/// takes nothing of).
pub(super) struct GroupClass {
    pub(super) places: usize,
    pub(super) claims: [Option<crate::memory::ledger::Claim>; 2],
}

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// Settle the device's ledger with the state slabs the stores hold now
    /// ([`StateSlabs::settle`]): after every change to the slabs or to the
    /// sequences' row claims.
    pub(super) fn settle_state_slabs(&mut self) {
        let backed = self.committed_state_bytes();
        let ledger = self.resources.ledgers().device();
        let claims = self.state_claims.values_mut().map(|state| &mut state.claim);
        self.state_slabs.settle(ledger, backed, claims);
    }
}

/// The state slabs the stores grew past those the load allocated as it
/// started, which the load's record holds, on the device's ledger. A
/// slab's bytes are the rows the sequences claimed by pages
/// ([`ExecutorDomain::claim_state_rows`]): they are marked allocated on
/// those claims, and only the slabs' excess past them is held, as spare
/// granules allocated at once. No slab byte is admitted beside the rows
/// that claimed it, and none reads as other programs' memory.
pub(super) struct StateSlabs {
    /// The stores' slab bytes as the load started.
    floor: u64,
    /// The grown slabs' bytes no sequence's rows claim.
    spare: Vec<crate::memory::ledger::Cached>,
    /// What a growth adds past the rows claimed, held before the stores
    /// allocate it and allocated as it exists: cache once settled
    /// ([`Self::settle`]), never counted as made while it is not.
    pending: Option<crate::memory::ledger::Claim>,
}

impl StateSlabs {
    /// The stores' slabs as the load started, `floor` bytes.
    pub(super) fn new(floor: u64) -> Self {
        Self { floor, spare: Vec::new(), pending: None }
    }

    /// Hold, before the stores allocate a growth of `bytes`, what it adds
    /// past the rows `claims` hold that no slab backs yet, as a claim
    /// allocated only as the slabs exist ([`Self::settle`] makes it spare).
    /// Refused, as its store refuses it, where free room lacks it; nothing
    /// is held then.
    pub(super) fn reserve<'a>(
        &mut self,
        ledger: &crate::memory::ledger::Ledger,
        bytes: u64,
        claims: impl Iterator<Item = &'a crate::memory::ledger::Claim>,
    ) -> Result<(), crate::memory::ledger::Short> {
        let unbacked = claims.map(|claim| claim.bytes() - claim.allocated_bytes()).sum::<u64>();
        let required = bytes.saturating_sub(unbacked);
        if required == 0 {
            return Ok(());
        }
        assert!(self.pending.is_none(), "invariant violated: a state growth is reserved while another waits to settle");
        self.pending = Some(ledger.backing(required)?);
        Ok(())
    }

    /// Settle with `backed`, the slab bytes the stores hold now. Of what
    /// they grew past the floor, each claim in `claims` (the earliest
    /// sequence first) is allocated for the rows it holds, and the rest is
    /// spare: what claims took of the spare in process and the slabs hold
    /// no more is freed first (their rows, or memory the slabs gave back),
    /// and what spare holds is cut or added to. Spare is added only from
    /// free room: where the room is held already (a preemption the evicted
    /// rows went to), it is not held twice.
    pub(super) fn settle<'a>(
        &mut self,
        ledger: &crate::memory::ledger::Ledger,
        backed: u64,
        claims: impl Iterator<Item = &'a mut crate::memory::ledger::Claim>,
    ) {
        let mut grown = backed.saturating_sub(self.floor);
        let claims: Vec<_> = claims
            .map(|claim| {
                let rows = claim.bytes().min(grown);
                grown -= rows;
                (claim, rows)
            })
            .collect();
        // Freed before the rows are allocated, so no reading counts the
        // memory twice.
        let releasing = self.spare.iter().map(crate::memory::ledger::Cached::releasing).sum::<u64>();
        let held = self.spare.iter().map(crate::memory::ledger::Cached::bytes).sum::<u64>();
        let gone = (held + releasing).saturating_sub(grown).min(releasing);
        let mut freed = gone;
        for spare in &mut self.spare {
            let paid = spare.releasing().min(freed);
            spare.released(paid);
            freed -= paid;
        }
        for (claim, rows) in claims {
            let has = claim.allocated_bytes();
            match rows.cmp(&has) {
                std::cmp::Ordering::Greater => claim.allocated(rows - has),
                std::cmp::Ordering::Less => claim.deallocated(has - rows),
                std::cmp::Ordering::Equal => {}
            }
        }
        // What claims took and the slabs still hold stays allocated,
        // releasing, until the slabs give it back: spare holds the rest.
        let grown = grown - (releasing - gone);

        let mut held = held;
        let mut at = self.spare.len();
        while held > grown {
            at -= 1;
            let has = self.spare[at].bytes();
            let cut = (held - grown).min(has);
            match cut == has && self.spare[at].releasing() == 0 {
                true => drop(self.spare.remove(at)),
                false => drop(self.spare[at].split(cut, crate::memory::ledger::CacheKind::Spare)),
            }
            held -= cut;
        }

        // What the stores allocated of the reserved growth is spare, as it
        // exists; what they did not allocate is free.
        if let Some(pending) = self.pending.take() {
            let backed = grown.saturating_sub(held).min(pending.bytes());
            if backed > 0 {
                self.spare.push(pending.into_backed_cache(backed, crate::memory::ledger::CacheKind::Spare));
                held += backed;
            }
        }
        if grown > held {
            // Refused where the room is held already: not held twice.
            self.spare.extend(ledger.cache(crate::memory::ledger::CacheKind::Spare, grown - held).ok());
        }
    }
}

/// A resident sequence's committed state rows ([`ExecutorDomain::claim_state_rows`]):
/// its claim on the device's ledger, and the row it holds them through.
pub(super) struct StateClaim {
    pub(super) claim: crate::memory::ledger::Claim,
    pub(super) through: u64,
}

/// The arena's committed claim as the domain is made: the load's own
/// graph scratch and the target graphs' bound constants (the floor's group
/// buffers and the held pass's buffers, which hold each step's routed
/// output until its back combines it, among them), carved from what the
/// load recorded allocating as it started (`recorded`), with groups of one
/// place. A group's class beyond it is claimed as the group opens
/// (`ExecutorDomain::open_group`).
pub(super) fn claim_arena<F: ProgramFamily>(
    execution: &ExecutionPlan,
    resources: &AllocatedResources,
    family: &F,
    recorded: &mut crate::memory::ledger::Recorded,
) -> Result<crate::memory::ledger::ArenaClaim, DomainError> {
    let constants = family.target_constant_bytes().map_err(DomainError::invariant)?;
    let allocated = execution
        .resources()
        .bytes()
        .scratch
        .checked_add(constants)
        .ok_or_else(|| DomainError::invariant("the arena's bytes overflow"))?;
    Ok(recorded.claim_arena(resources.ledgers().device(), allocated))
}

/// What growing the prompt-group class from the floor's one place to
/// `places` claims: the target's group buffers and the bank's placed words
/// (`ProgramFamily::group_demand`), the pools' slots
/// (`AllocatedResources::class_bytes`), and the graphs those slots make
/// as they are allocated, priced by the device's graph cost beyond its
/// budget (`graph_demand`).
fn class_demand<F: ProgramFamily>(
    execution: &ExecutionPlan,
    resources: &mut AllocatedResources,
    family: &F,
    device: &seismic::Device,
    places: usize,
) -> Result<crate::memory::ClaimDemand, DomainError> {
    let slots = execution.policy().limits().class_slots(places);
    let pools = resources.class_bytes(slots).map_err(DomainError::invariant)?;
    let graphs = resources.class_graphs(slots).map_err(DomainError::invariant)?;
    let graphs = graph_demand(device.graph_cost(), device.graph_budget(), graphs).map_err(DomainError::invariant)?;
    family
        .group_demand(places)
        .map_err(DomainError::Input)?
        .checked_plus(crate::memory::ClaimDemand { device: pools, ..Default::default() })
        .and_then(|demand| demand.checked_plus(graphs))
        .ok_or_else(|| DomainError::invariant("a group's demand overflows"))
}

/// A failed slot allocation of `bytes`: a capacity deficit when the device
/// refused the bytes, otherwise a device failure.
fn growth_failure(error: seismic::TensorError, bytes: u64, growth: &str) -> DomainError {
    match error {
        seismic::TensorError::Execution(ref execution)
            if execution.capacity_refusal().is_some()
                || matches!(execution, seismic::ExecutionError::AllocationFailed(_)) =>
        {
            DomainError::Capacity(CapacityError {
                resource: ResourceKind::DeviceMemory,
                required: bytes,
                available: execution.capacity_refusal().map_or(0, |(_, available)| available),
            })
        }
        other => DomainError::Device(crate::DeviceError::Execution(format!("{growth}: {other}"))),
    }
}

/// The measurement job's committed claim on the device's ledger, and its
/// hold of the bank's own traffic, which stands while the claim does and
/// drops with it.
pub(super) struct MeasurementClaim {
    _claim: crate::memory::ledger::Claim,
    _traffic: Option<TrafficHold>,
}

#[cfg(test)]
#[path = "claims_tests.rs"]
mod claims_tests;

#[cfg(test)]
mod group_demand_tests {
    use super::*;

    /// A class's graphs demand what the device's graph budget costs with
    /// them beyond what it costs now: on a pooled driver, the granules the
    /// class adds, none where the budget's last granule holds them; on
    /// Vulkan, command memory in device storage and argument chunks in
    /// upload storage; nothing where recording holds no device memory.
    #[test]
    fn a_class_demands_the_granules_its_graphs_add() {
        let pooled = seismic::GraphCost::Pooled { graph: 0, launch: 1000, granule: 1 << 20 };
        let launches = |count: u64| seismic::GraphCensus { graphs: 1, launches: count, arguments: 8 * count, largest: 8 };
        // 1,000 launches price at 1,000,000 bytes: one granule.
        let budget = launches(1000);
        let demand = graph_demand(pooled, budget, launches(40)).unwrap();
        assert_eq!(demand, crate::memory::ClaimDemand::default());
        let demand = graph_demand(pooled, budget, launches(1000)).unwrap();
        assert_eq!(demand, crate::memory::ClaimDemand { device: 1 << 20, ..Default::default() });
        let chunked = seismic::GraphCost::Chunked { launch: 64, chunk: 1024 };
        let demand = graph_demand(chunked, launches(0), launches(10)).unwrap();
        assert_eq!(demand, crate::memory::ClaimDemand { device: 640, upload: 2048, ..Default::default() });
        let demand = graph_demand(seismic::GraphCost::None, budget, launches(1000)).unwrap();
        assert_eq!(demand, crate::memory::ClaimDemand::default());
    }
}

