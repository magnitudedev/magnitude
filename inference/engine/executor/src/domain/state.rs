//! state lifecycle for the executor domain.

use super::*;
use crate::assessment::{DecodeWaiting, Places};
use crate::memory::{ClaimId, HoldingClass};
use magnitude_state::{
    GrowthChoice, Holder, RowDemand, ShrinkPolicy, StateHoldingCensus, StoreCopy,
};

/// The storage of `kind` this executor places first: its store.
fn storage_store(topology: &seismic::DeviceTopology, executor: &seismic::Executor, kind: seismic::Contract) -> Option<seismic::StoreId> {
    executor.paths(seismic::Placing::Fixed(kind)).first().and_then(|path| topology.access_path(*path)).map(|path| path.store())
}

/// The bytes of `demand` that land in the store the bank's device storage
/// lies in: what a group's places weigh against the bank's slots there.
fn beside_bank_bytes(topology: &seismic::DeviceTopology, executor: &seismic::Executor, demand: crate::memory::ClaimDemand) -> Result<u64, DomainError> {
    let store = storage_store(topology, executor, seismic::Contract::DeviceOnly)
        .ok_or_else(|| DomainError::invariant("a bank has no device storage placement"))?;
    let mut bytes = 0u64;
    for kind in crate::memory::CLAIMED {
        if storage_store(topology, executor, kind) == Some(store) {
            bytes = bytes.checked_add(demand.storage_bytes(kind).ok_or_else(|| DomainError::invariant("group storage overflows"))?)
                .ok_or_else(|| DomainError::invariant("group store demand overflows"))?;
        }
    }
    // Staging is either already included in device demand on unified RAM,
    // or belongs to a distinct host store (Fit::host_staging).
    Ok(bytes)
}

/// What recording the graphs of `more` takes on a device of `cost` whose
/// graph budget is `budget` (`seismic::Device::graph_budget`): the memory
/// the budget with them costs beyond what it costs now, in the storage the
/// graphs hold it in. A device's graphs are priced as a whole, so the
/// difference counts the granules `more` adds.
pub(crate) fn graph_demand(
    cost: seismic::GraphCost,
    budget: seismic::GraphCensus,
    more: seismic::GraphCensus,
) -> Result<crate::memory::ClaimDemand, DomainError> {
    let priced = |census| cost.bytes(census).ok_or_else(|| DomainError::invariant("a graph budget's memory overflows"));
    let grown = budget.plus(more).ok_or_else(|| DomainError::invariant("a graph budget overflows"))?;
    let bytes = priced(grown)?.beyond(priced(budget)?);
    Ok(crate::memory::ClaimDemand { device: bytes.device, upload: bytes.upload, ..Default::default() })
}

/// What a prompt's places are worth beyond what memory gave its group
/// (`ExecutorDomain::grant_group`): `places` for `request`'s chunks in hand,
/// and the slots the bank shrinks to for their bytes, at flight boundaries
/// (`ExecutorDomain::places_shrink`). Its memory lands in a later group's
/// room, so a later group of the request takes it rather than weighing
/// again on a reading that may not show it yet. It holds until a group of
/// the request holds those places or no more is in hand, or the request is
/// released or closes; while it holds, the bank grows nothing.
#[derive(Clone, Copy, Debug)]
pub(super) struct PlacesWant {
    pub(super) request: RequestId,
    pub(super) places: usize,
    pub(super) slots: u64,
}

/// Where the room the service's measurement job claims beside the model
/// stands at the latest probe (`ExecutorDomain::measurement_room`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeasurementRoom {
    /// What opportunistic growth may take now holds the claim.
    Made,
    /// The bytes the bank's shrink must still free, which what it holds
    /// above its floor covers.
    Short(u64),
    /// Short by more than the bank holds above its floor
    /// (`releasable`, none without a bank): no shrink makes the room.
    Beyond { short: u64, releasable: u64 },
    /// No probe has read memory yet.
    Unread,
    /// The room is made, and the bank's own copies and reads begun before
    /// the claim are still running: the job would measure beside them.
    Draining,
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
        let claim = self.claim_device_growth(bytes, 0, HoldingClass::Live)?;
        let tensor = Tensor::zeros(self.domain.device(), element, &[1, width]);
        self.release_claim(claim);
        let tensor = tensor.map_err(|error| DomainError::Input(error.to_string()))?;
        self.domain
            .publish_features(tensor)
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
    /// Release bound optional components after the owner has drained all
    /// requests; the queued steps, which draft with them, are orphaned
    /// first. The measured ledger delta is the only released-byte credit;
    /// shared target tensors remain cached by the head loader.
    pub fn release_idle_optional_components(
        &mut self,
        bindings: &mut StateBindings,
    ) -> Result<u64, String> {
        if !self.target.is_empty() || !self.head.is_empty() || !self.input.is_empty() {
            return Ok(0);
        }
        self.orphan_queue(bindings)
            .map_err(|error| error.to_string())?;
        let before = self
            .domain
            .device()
            .device_ledger()
            .charged;
        self.family.unbind_optional();
        if let Some(loader) = &self.head_loader {
            loader.unload();
        }
        if let Some(loader) = &self.vision_loader {
            loader.unload();
        }
        self.refresh_memory().map_err(|error| error.to_string())?;
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

    /// The peak new charge of importing and binding the optional component a
    /// group needs, when it is not bound, and the part a dedicated device
    /// stages through host RAM. A lazy import can briefly hold an upload
    /// tensor alongside the resident weights.
    pub(super) fn component_growth(
        &self,
        operations: &[Operation],
    ) -> Result<Option<(u64, u64)>, DomainError> {
        let head = matches!(operations.first(), Some(Operation::Head { .. }))
            || operations
                .iter()
                .any(|operation| matches!(operation, Operation::Forward { prime: Some(_), .. }));
        let (resident_bytes, upload_peak) = match operations.first() {
            Some(_) if head && !self.family.head_is_bound() => {
                let binding_constants = self
                    .head_loader
                    .as_ref()
                    .ok_or_else(|| DomainError::Input("head component is disabled".into()))?
                    .binding_constant_bytes()
                    .map_err(DomainError::Input)?;
                (
                    self.execution.resources().bytes().head_weights,
                    self.execution
                        .load()
                        .head_upload_peak_bytes()?
                        .max(binding_constants),
                )
            }
            Some(Operation::Encode { .. }) if !self.family.vision_is_bound() => (
                self.execution.resources().bytes().vision_weights,
                self.execution.load().vision_upload_peak_bytes()?,
            ),
            _ => return Ok(None),
        };
        let required = resident_bytes
            .checked_add(upload_peak)
            .ok_or_else(|| DomainError::Input("component import claim overflows".into()))?;
        Ok(Some((required, upload_peak)))
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
        for operation in operations {
            if let Operation::Forward { request, .. } | Operation::Head { request, .. } = operation
            {
                self.relocate_tails(bindings, *request)?;
            }
        }
        let (mut target, mut target_queued) = (Vec::new(), Vec::new());
        let (mut head, mut head_queued) = (Vec::new(), Vec::new());
        // The steps queued behind the group (`behind`, within the class it is
        // granted) each follow the one before by their rows. A step takes a
        // successor bank, and the steps behind it one more: a pipelined
        // successor its own, a chain's chunks the two they alternate
        // (`OwnedChainedAdvance`).
        self.grant_group(operations)?;
        let behind = self.behind(operations);
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
        // Stores without history domains still grow banks.
        if target_banks != 0 {
            self.grant_state_growth(bindings, false, &target, &target_queued, target_banks)?;
        }
        if head_banks != 0 && self.head_store.is_some() {
            self.grant_state_growth(bindings, true, &head, &head_queued, head_banks)?;
        }
        // The import itself runs under a claim at reservation; admit its peak
        // here so a deficit reaches the owner's release order first.
        if let Some((required, staged)) = self.component_growth(operations)? {
            self.decide_device_growth(required, staged)?;
        }
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
    /// recorded ([`graph_demand`]). Each charge is claimed from the heap like
    /// any growth; a refusal is a demand deficit for the owner's release
    /// order.
    fn provision_vision(&mut self, patch_rows: u64) -> Result<(), DomainError> {
        let pool = self
            .resources
            .vision_graph()
            .ok_or_else(|| DomainError::invariant("vision graph is absent"))?;
        let device = self.domain.device();
        let activation = (pool.available_workspace(patch_rows) == 0)
            .then(|| {
                let graphs = graph_demand(device.graph_cost(), device.graph_budget(), pool.activation_graphs(patch_rows)?)?;
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
            self.grow_vision(demand, |pool| pool.grow_activation(patch_rows))?;
        }
        if let Some(demand) = output {
            self.grow_vision(demand, |pool| pool.grow_output(patch_rows))?;
        }
        Ok(())
    }

    fn grow_vision(
        &mut self,
        demand: crate::memory::ClaimDemand,
        grow: impl FnOnce(&mut crate::VisionGraphPool) -> Result<(), crate::resources::VisionGrowth>,
    ) -> Result<(), DomainError> {
        let bytes = demand.bytes();
        let claim = self.memory.borrow_mut().claim_storage(demand, HoldingClass::Live)?;
        let grown = grow(
            self.resources
                .vision_graph_mut()
                .expect("the vision graph was found above"),
        );
        self.release_claim(claim);
        grown.map_err(|growth| match growth {
            crate::resources::VisionGrowth::Class(error) => {
                DomainError::invariant(format!("vision class was not prepared: {error}"))
            }
            crate::resources::VisionGrowth::Device(error) => {
                growth_failure(error, bytes, "vision growth")
            }
        })?;
        self.refresh_memory()
    }

    /// Grow `lane`'s retained-output pool by one slot when none is free, so
    /// the launch finds its output: a request keeps its last published
    /// output after its launch completes, so the pool follows the live
    /// requests. A target launch's pool is of the readout class that holds
    /// its `projected` rows (`AllocatedResources::retained_outputs_mut`).
    /// The slot's charge is claimed from the heap like any state growth; a
    /// refusal is a demand deficit for the owner's release order.
    pub(super) fn provision_output(&mut self, lane: ReservationLane, projected: usize) -> Result<(), DomainError> {
        let Some(pool) = self.resources.retained_outputs_mut(lane, projected) else {
            return Ok(());
        };
        if pool.available_output() > 0 {
            return Ok(());
        }
        let bytes = pool.output_slot_bytes();
        let claim = self.claim_device_growth(bytes, 0, HoldingClass::Live)?;
        let grown = self
            .resources
            .retained_outputs_mut(lane, projected)
            .expect("the pool was found above")
            .grow_output();
        self.release_claim(claim);
        grown.map_err(|error| growth_failure(error, bytes, "graph output growth"))?;
        self.refresh_memory()
    }

    /// Grant elastic state growth of the target or `head` store from a fresh
    /// domain observation. Growth changes bindings, so the queued steps are
    /// orphaned first; its release may itself free what the growth needed.
    /// Seismic's charge ledger remains authoritative: its limit enforces the
    /// grant on each added slab allocation. `queued` are the rows of the
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
        // demands only admit slabs already backed. Reservation observes
        // memory again before launching the selected group.
        let store = bindings.store(head)?;
        let claim = store.growth_claim(demands, queued, banks)?;
        if claim.minimum_bytes == 0 && claim.preferred_bytes == 0 {
            return Ok(store.provision_with_growth(
                demands,
                queued,
                banks,
                GrowthChoice::Minimum,
            )?);
        }
        for choice in [GrowthChoice::Preferred, GrowthChoice::Minimum] {
            let claim = store.growth_claim(demands, queued, banks)?;
            let required = match choice {
                GrowthChoice::Preferred => claim.preferred_bytes,
                GrowthChoice::Minimum => claim.minimum_bytes,
            };
            // Keep the peak claimed while the store performs its fallible
            // physical operation. Seismic measures the resulting charge; the
            // existing static holding is then resized from committed backing
            // instead of creating another holding for the same bytes.
            let claim = if required == 0 {
                None
            } else {
                match self.claim_device_growth(required, 0, HoldingClass::Live) {
                    Ok(claim) => Some(claim),
                    Err(_) if choice == GrowthChoice::Preferred => continue,
                    Err(error) => return Err(error),
                }
            };
            let provisioned = store.provision_with_growth(demands, queued, banks, choice);
            if let Some(claim) = claim {
                self.release_claim(claim);
            }
            match provisioned {
                Ok(()) => {
                    self.refresh_memory()?;
                    return Ok(());
                }
                Err(error)
                    if choice == GrowthChoice::Preferred
                        && matches!(
                            error,
                            magnitude_state::Error::Tensor(seismic::TensorError::Execution(
                                seismic::ExecutionError::AllocationCapacity { .. }
                                    | seismic::ExecutionError::AllocationFailed(_)
                            ))
                        ) =>
                {
                    continue;
                }
                Err(error) => return Err(error.into()),
            }
        }
        unreachable!("minimum state growth either succeeds or returns a deficit")
    }

    /// Observe every domain the device uses and refresh the Seismic
    /// allocation ceiling: `Reclaim` while any domain's headroom is at or
    /// below its planning reserve, `Blind` when an observation fails.
    /// It classifies the bands alone: claims granted to running work are
    /// not admitted again, so memory the host takes from under a running
    /// growth is the band it puts the domain in, never a refusal.
    /// Its readings are the ones the memory pace is measured from
    /// (`MemoryPace`): the owner probes every `reading_interval`, and again
    /// after each release while memory is short.
    pub fn probe_memory(&mut self, reading_interval: std::time::Duration) -> Result<(), DomainError> {
        let readings = self.observe_memory()?;
        let host = self.host_witness(&readings);
        self.pace.borrow_mut().observe(
            &readings,
            &self.domain.device().ledgers().own_claims(),
            host,
            !self.resize_held.runs(),
            reading_interval,
        );
        match readings
            .into_iter()
            .find(|reading| reading.band == crate::platform::MemoryBand::Reclaim)
        {
            Some(reading) => Err(DomainError::Reclaim(reading)),
            None => Ok(()),
        }
    }

    /// Host RAM's witness of the process's releases at a probe that read
    /// `readings` (`MemoryPace`): what the system counts the process holding
    /// out of free memory, which falls by a release as free memory rises by
    /// it. A Metal device's allocations are wired through its residency set,
    /// so the host's wired pages (`seismic::HostHeld`): a released buffer
    /// leaves the footprint at once, but its pages are free only once the
    /// driver unwires them at its next tick (on an M4 Pro, 0.47 s after a
    /// 138 MB slot's release free and wired memory moved together; the
    /// first slot a reserved buffer released was not unwired within 3 s).
    /// The count is the host's, the process's own wired size counting none
    /// of the residency set's: another program wiring memory in the span
    /// holds a release unseen until the credit lapses, and one unwiring
    /// shows it early. Any other device's allocations are pages of the
    /// process, so its footprint (`seismic::ProcessMemory`), which the
    /// system frees as they leave it (a GB10's fell by each bank slot within
    /// its release). None where the system reports neither.
    fn host_witness(&self, readings: &[crate::platform::DomainReading]) -> Option<u64> {
        match self.domain.device().backend() {
            seismic::BackendName::Metal => readings
                .iter()
                .find(|reading| reading.identity == seismic::StoreIdentity::HostRam)
                .and_then(|reading| reading.held)
                .map(|held| held.wired_bytes),
            _ => self.memory.borrow().footprint(),
        }
    }

    /// Claim the peak new charge of one physical operation from the heap:
    /// `required` device bytes, of which a dedicated device stages `staged`
    /// through host RAM. The caller releases the claim once Seismic has
    /// measured the operation's result.
    pub(super) fn claim_device_growth(
        &mut self,
        required: u64,
        staged: u64,
        class: HoldingClass,
    ) -> Result<ClaimId, DomainError> {
        Ok(self.memory.borrow_mut().claim(required, staged, class)?)
    }

    pub(super) fn release_claim(&mut self, claim: ClaimId) {
        self.memory.borrow_mut().release(claim);
    }

    /// Hold a head just bound for the class held, under a claim for what it
    /// demands beyond the floor charged with its weights
    /// (`ProgramFamily::bound_head_demand`). A refused claim unbinds the
    /// head: no allocation escapes the heap's charge.
    pub(super) fn hold_bound_head(&mut self) -> Result<(), DomainError> {
        let places = self.resources.group_places().get();
        let demand = self.family.bound_head_demand(places).map_err(DomainError::Input)?;
        if demand == crate::memory::ClaimDemand::default() {
            return Ok(());
        }
        let (_, unborn) = self.opportunistic_room(seismic::Contract::DeviceOnly);
        let claim = match self.memory.borrow_mut().claim_demand(demand, unborn, HoldingClass::Live) {
            Ok(claim) => claim,
            Err(refusal) => {
                self.family.unbind_head();
                return Err(refusal.into());
            }
        };
        let held = self.family.hold_bound_head(places);
        self.release_claim(claim);
        self.refresh_memory()?;
        held.map_err(|error| {
            self.family.unbind_head();
            DomainError::invariant(error)
        })
    }

    /// Whether the heap would grant a peak of `required` device bytes, of
    /// which a dedicated device stages `staged` bytes through host RAM, from
    /// a fresh observation (see [`DeviceHeap::check`]).
    pub(super) fn decide_device_growth(
        &mut self,
        required: u64,
        staged: u64,
    ) -> Result<(), DomainError> {
        Ok(self.memory.borrow_mut().check(required, staged)?)
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
        self.refresh_memory()?;
        Ok(())
    }

    /// Release empty state slabs at idle, keeping one spare per store. In
    /// Reclaim, also compact sparse occupied slabs into already held space,
    /// and release the held prompt-group class down to the floor. Idle
    /// graph slots are released off this thread, `wake` completed once they
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
        let (buffers, slots) = self.release_idle_graph_slots(policy, wake)?;
        Ok(StateShrink { released: released + buffers, slots })
    }

    /// At a flight boundary: the idle graph slots' release, once the load's
    /// release thread has ended it (`AllocatedResources::released_slots`),
    /// the heap then reading the charge it left; none while it runs.
    pub fn advance_slot_release(&mut self) -> Result<Option<crate::resources::SlotsReleased>, DomainError> {
        let Some(released) = self.resources.released_slots().map_err(DomainError::invariant)? else {
            return Ok(None);
        };
        self.refresh_memory()?;
        Ok(Some(released))
    }

    /// Grant the prompt group opening with `operations` its class: the
    /// places of every graph slot its chain's launches take (`behind`, which
    /// runs at most that many). The places its request's chunks in hand
    /// (the group's planned chunks, within the load's chain capacity) are
    /// worth ([`Self::weigh_places`]), or the most fewer whose class fits:
    /// below what opportunistic growth may take now (`Places::admitted`, of
    /// [`Self::opportunistic_room`]) and granted by the heap beside the
    /// workload's unborn state, which refuses every growth in Reclaim. The
    /// class held already serves any smaller one, and nothing shrinks here:
    /// places worth the bank's memory leave a want its shrink serves
    /// (`PlacesWant`). The one decision of a group's places; its pools'
    /// slots and buffers are held together or not at all. A class's bytes
    /// are its group buffers, its places' bank words, its pools' slots and
    /// the graphs each slot makes as it is allocated, priced by the
    /// device's graph cost beyond its budget (`graph_demand`).
    pub(super) fn grant_group(&mut self, operations: &[Operation]) -> Result<(), DomainError> {
        self.retire_places_want(operations);
        let capacity = self.chain_capacity().get();
        let (request, chunks) = match (operations, self.behind_within(operations, capacity)) {
            ([operation], super::queue::Behind::Chain { chunks }) => {
                // `behind_within` holds a chain within `capacity` launches.
                (operation.request(), chunks.len() + 1)
            }
            _ => return Ok(()),
        };
        let floor = self.resources.group_places();
        let held = floor.get();
        let limits = self.execution.policy().limits();
        let (room, unborn) = self.opportunistic_room(seismic::Contract::DeviceOnly);
        // What the bank can still grow by is not the places' to take: a
        // bank grows to its ceiling (every expert, or a weight cap's bound)
        // before places grow into its room, and no further.
        let shortfall = self.family.expert_bank().map(|bank| bank.growth_shortfall()).transpose()?.unwrap_or(0);
        let room = room.map(|(room, margin)| (room.saturating_sub(shortfall), margin));
        let wait = self.pace.borrow().wait_bound();
        // The bytes of a class of each count of places from those held.
        let device = self.domain.device();
        let demands = (held..=chunks.max(held))
            .map(|places| {
                let slots = limits.class_slots(places);
                let pools = self.resources.class_bytes(slots).map_err(DomainError::invariant)?;
                let graphs = self.resources.class_graphs(slots).map_err(DomainError::invariant)?;
                let graphs = graph_demand(device.graph_cost(), device.graph_budget(), graphs)?;
                self.family.group_demand(places).map_err(DomainError::Input)?
                    .checked_plus(crate::memory::ClaimDemand { device: pools, ..Default::default() })
                    .and_then(|demand| demand.checked_plus(graphs))
                    .ok_or_else(|| DomainError::invariant("a group's demand overflows"))
            })
            .collect::<Result<Vec<_>, DomainError>>()?;
        let bytes = demands.iter()
            .map(|&demand| beside_bank_bytes(device.topology(), device.info(), demand))
            .collect::<Result<Vec<_>, DomainError>>()?;
        // Without a reading only the heap bounds a grant: nothing is short.
        let places = Places {
            chunks,
            held,
            bytes: &bytes,
            room: room.map_or(u64::MAX, |(room, _)| room),
            wait,
        };
        // A want the request's earlier group left holds until a group holds
        // its places (`PlacesWant`); any other group weighs its own.
        let want = self
            .places_want
            .filter(|want| want.request == request && want.places.min(chunks) > held);
        let worth = match want {
            Some(want) => want.places.min(chunks),
            None => {
                let (worth, slots) = self.weigh_places(places)?;
                self.places_want = slots.map(|slots| PlacesWant { request, places: worth, slots });
                worth
            }
        };
        // What room admits now; the rest waits for the bank's shrink.
        let admitted = worth.min(places.admitted(|_| false).0);
        let counts = demands.iter().copied().zip(bytes.iter().copied()).enumerate()
            .take(admitted.saturating_sub(held).saturating_add(1));
        for (more, (demand, bytes)) in counts.skip(1).rev() {
            let grown = floor.saturating_add(more);
            let places = grown.get();
            let claim = match self.memory.borrow_mut().claim_demand(demand, unborn, HoldingClass::Live) {
                Ok(claim) => claim,
                Err(ClaimRefusal::Deficit { .. }) => continue,
                // Nothing grows in Reclaim, nor blind.
                Err(ClaimRefusal::Reclaim(_) | ClaimRefusal::Blind(_)) => return Ok(()),
                Err(refusal) => return Err(refusal.into()),
            };
            // The pools' slots, then the buffers, each keeping what was held
            // on a failure; the class is held only once both are.
            let held = self
                .resources
                .grow_class(grown, limits.class_slots(places))
                .and_then(|grown| self.family.hold_group(places).map(|()| grown))
                .map(|grown| self.resources.hold_class(grown));
            self.release_claim(claim);
            if held.is_ok() {
                self.retire_places_want(operations);
                let wait = wait.map_or("none".to_owned(), |wait| format!("{wait:.1} s"));
                tracing::info!(
                    "prompt groups hold {places} places, {bytes} bytes more; wait bound {wait}"
                );
            }
            self.refresh_memory()?;
            return match held.map_err(|error| match error {
                crate::resources::AllocationError::Tensor(error)
                | crate::resources::AllocationError::Workflow(seismic::WorkflowError::TensorView(error)) => {
                    growth_failure(error, bytes, "prompt group class")
                }
                error => DomainError::invariant(error.to_string()),
            }) {
                // The device refused the bytes: the class held serves.
                Err(DomainError::Capacity(_)) => Ok(()),
                held => held,
            };
        }
        Ok(())
    }

    /// Retire the prompt's places want where it ends (`PlacesWant`): the
    /// class held holds its places, or the group opening with `operations`
    /// holds its request's last chunk in hand or none (its operation plans
    /// no chunk after it: the last prompt chunk, or decode).
    fn retire_places_want(&mut self, operations: &[Operation]) {
        let held = self.resources.group_places().get();
        self.places_want = self.places_want.filter(|want| {
            want.places > held
                && operations
                    .iter()
                    .filter(|operation| operation.request() == want.request)
                    .all(|operation| self.planned_after(operation).is_some())
        });
    }

    /// The places `places`' chunks in hand are worth, and the slots the
    /// bank shrinks to for them when fewer than it holds: by value per byte
    /// within the wait bound (`GroupReads::grant`), on the decode the
    /// load's latest miss record prices while decode waits (`decode_owed`),
    /// each group's passes by the executor's costs where measured. Before
    /// the load has measured its groups' reads, and on a load without a
    /// bank, whose places compete with no slot, or with a whole bank, which
    /// gives none up, every chunk, as a group runs to capacity until the
    /// run has measured its layers (`Telemetry::group_size`).
    fn weigh_places(&self, places: Places<'_>) -> Result<(usize, Option<u64>), DomainError> {
        let chunks = places.chunks;
        let Some(bank) = self.family.expert_bank() else {
            return Ok((chunks, None));
        };
        // A whole bank gives up no slot for places: they grow only into
        // room beside it.
        if bank.whole() {
            return Ok((chunks, None));
        }
        let Some((reads, slots)) = bank.group_reads()? else {
            return Ok((chunks, None));
        };
        let pass = self.experts.as_ref().and_then(|measures| measures.pass.measured());
        let record = self.decode_owed.and_then(|_| bank.latest_record());
        // Decode waiting is priced on the resident step; a load that could
        // not estimate it says why.
        let resident = match (&record, &self.experts) {
            (Some(_), Some(measures)) => Some(measures.resident.measured().map(|resident| &**resident).ok_or_else(|| {
                DomainError::invariant(format!(
                    "decode waits on an unpriced resident step: {}",
                    measures.resident.why().unwrap_or_default()
                ))
            })?),
            _ => None,
        };
        let grant = match (self.decode_owed, record, resident) {
            (Some(owed), Some(record), Some(resident)) => {
                // A record is due only with a step that misses nothing, so one
                // that prices none breaks an invariant (`ElasticBank::growth`).
                let priced = |error: crate::assessment::MissRecordError| {
                    DomainError::invariant(format!("the miss record prices no step: {error}"))
                };
                let unpriced = |error: crate::assessment::AssessmentError| {
                    DomainError::invariant(format!("the resident step: {error}"))
                };
                let compute = record.compute(resident).map_err(unpriced)?.map_err(priced)?;
                // The host tier beside the bank stays as it is.
                let host = bank.residency()?.host.map_or(0, |host| host.slots);
                let count = |value: u64| usize::try_from(value).map_err(|_| DomainError::invariant("bank slots exceed the host"));
                let host = count(host)?;
                // Decode runs beside the prompt's groups.
                let mut step = |slots: u64| -> Result<f64, DomainError> {
                    let held = crate::assessment::HeldSlots { slots: count(slots)?, host };
                    let step = record.step_seconds(held, record.rows().map_err(priced)?, compute).map_err(priced)?;
                    Ok(step.total_seconds)
                };
                reads.grant(slots, places, pass, Some(DecodeWaiting { owed, step: &mut step }))?
            }
            _ => reads.grant::<DomainError>(slots, places, pass, None)?,
        };
        Ok((grant.places, (grant.slots < slots.now).then_some(grant.slots)))
    }

    /// The bytes the bank gives up now for a prompt's places
    /// (`PlacesWant`): its slots beyond the want's; none without a want.
    pub fn places_shrink(&self) -> Result<Option<u64>, DomainError> {
        let (Some(want), Some(bank), Some(plan)) =
            (self.places_want, self.family.expert_bank(), self.execution.load().expert_bank())
        else {
            return Ok(None);
        };
        match bank.slots().saturating_sub(want.slots) {
            0 => Ok(None),
            excess => excess
                .checked_mul(plan.slot_bytes().map_err(DomainError::invariant)?)
                .map(Some)
                .ok_or_else(|| DomainError::invariant("the bank's shrink for places overflows")),
        }
    }

    /// Claim `bytes` of device memory for the service's measurement job
    /// beside the model, in place of any claim before: it is held out of
    /// what opportunistic growth may take ([`Self::opportunistic_room`],
    /// [`Self::growth_room`]), so neither the bank's slots nor a group's
    /// places take it, and the
    /// bank's shrink makes it ([`Self::measurement_room`]); and the bank's
    /// own traffic is held (`ElasticBank::hold_traffic`), so the job
    /// measures beside none of the load's copies or reads. A growth or the
    /// host tier's backing running ends at its next slot
    /// (`ElasticBank::stop_growths`), and none is
    /// offered while the claim stands ([`Self::expert_bank_growths`]): held,
    /// its slots would wait for the claim's release, which waits for its
    /// end.
    pub fn claim_measurement_room(&mut self, bytes: u64) {
        if let Some(bank) = self.family.expert_bank() {
            bank.stop_growths();
        }
        let traffic = self.hold_expert_traffic();
        self.measurement_claim = Some(MeasurementClaim { bytes, _traffic: traffic });
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

    /// The measurement job ended: the bank may grow into its room again,
    /// and its traffic runs again.
    pub fn release_measurement_room(&mut self) {
        // Its hold of the bank's traffic drops with it.
        self.measurement_claim = None;
    }

    /// Where the measurement job's claim stands at the latest probe: made
    /// once the room beside the workload ([`Self::workload_room`]) holds
    /// it, else short by the rest, within or beyond what the bank holds
    /// above its floor ([`Self::bank_above_floor`]), and once made, still
    /// draining while the bank's traffic begun before the claim runs
    /// (`ElasticBank::traffic_quiet`); none without a claim.
    pub fn measurement_room(&self) -> Result<Option<MeasurementRoom>, DomainError> {
        let Some(bytes) = self.measurement_claim.as_ref().map(|claim| claim.bytes) else {
            return Ok(None);
        };
        let quiet = self.expert_traffic_quiet();
        Ok(Some(match self.workload_room(seismic::Contract::DeviceOnly).0 {
            None => MeasurementRoom::Unread,
            Some((room, _)) if room >= bytes && !quiet => MeasurementRoom::Draining,
            Some((room, _)) if room >= bytes => MeasurementRoom::Made,
            Some((room, _)) => match (bytes - room, self.bank_above_floor()?) {
                (short, releasable) if short <= releasable => MeasurementRoom::Short(short),
                (short, releasable) => MeasurementRoom::Beyond { short, releasable },
            },
        }))
    }

    /// The bytes the expert bank's slots above its floor hold: its records'
    /// charge (`ExpertResidency::stores`), each slot an equal share of it;
    /// none without a bank.
    fn bank_above_floor(&self) -> Result<u64, DomainError> {
        let Some(residency) = self.expert_residency()? else {
            return Ok(0);
        };
        let charged = residency.stores.iter().map(|(_, bytes)| u128::from(*bytes)).sum::<u128>();
        let above = u128::from(residency.slots.saturating_sub(residency.floor));
        let held = charged * above / u128::from(residency.slots.max(1));
        u64::try_from(held).map_err(|_| DomainError::invariant("the bank's charge exceeds the host"))
    }

    /// What a group's places may take now in `kind` storage: the room beside
    /// the workload there ([`Self::workload_room`]) less, in device
    /// storage's store, the measurement job's claim; and the workload's
    /// unborn state there.
    fn opportunistic_room(&self, kind: seismic::Contract) -> (Option<(u64, u64)>, u64) {
        let (room, unborn) = self.workload_room(kind);
        let claimed = self.measurement_beside(kind);
        (room.map(|(room, margin)| (room.saturating_sub(claimed), margin)), unborn)
    }

    /// The measurement job's claim in `kind` storage's store: none but in
    /// device storage's.
    fn measurement_beside(&self, kind: seismic::Contract) -> u64 {
        match self.beside_device(kind) {
            true => self.measurement_claim.as_ref().map_or(0, |claim| claim.bytes),
            false => 0,
        }
    }

    /// The most bytes a growth step of the expert bank in `kind` storage may
    /// back now, and the unborn state its claim leaves beside it: what the
    /// heap's claim of them admits (`DeviceHeap::reserved_room`, the one test
    /// the claim then passes, beside the workload's unborn state), less its
    /// store's fluctuation (`MemoryPace::fluctuation`) and, in device
    /// storage's store, the measurement job's claim. Refused in the Reclaim
    /// band, or blind.
    fn growth_room(&self, kind: crate::planning::BankTier) -> Result<(u64, u64), ClaimRefusal> {
        let unborn = self.unborn(kind.storage());
        let room = self.memory.borrow_mut().reserved_room(kind, unborn)?;
        let fluctuation = self.store_of(kind.storage()).map_or(0, |store| self.pace.borrow().fluctuation(store));
        let claimed = self.measurement_beside(kind.storage());
        Ok((room.saturating_sub(fluctuation).saturating_sub(claimed), unborn))
    }

    /// What the plan charged in `kind` storage's store that is not held yet,
    /// so no growth takes it: the planned workload's state the stores have
    /// not committed (`StateCapacityPlan::workload_state_bytes` less the
    /// stores' committed slabs), so the workload's conversations grow into
    /// the room the plan charged them, and the planned weights of each
    /// optional component not bound yet (the head's, the projector's, each
    /// imported at its first use, [`Self::component_growth`]), so that use
    /// finds the room the plan charged it. A bound component's weights are
    /// the ledger's holding (`classify_memory`, which holds only completed
    /// imports), so each is charged once, here or there. None
    /// in another store (a discrete device's host memory). The memory the
    /// device's graph budget costs is held: its live slots' holds charge it
    /// to the ledger.
    fn unborn(&self, kind: seismic::Contract) -> u64 {
        if !self.beside_device(kind) {
            return 0;
        }
        let bytes = self.execution.resources().bytes();
        unborn_bytes(
            self.execution.resources().capacity().workload_state_bytes,
            self.committed_state_bytes(),
            [(self.family.head_is_bound(), bytes.head_weights), (self.family.vision_is_bound(), bytes.vision_weights)],
        )
    }

    /// The room beside the workload in `kind` storage's store: what a growth
    /// step leaves above its planning line's fluctuation
    /// (`MemoryPace::growth_room`, with that margin; none without a reading
    /// of it) less, in device storage's store, the planned workload's state
    /// the stores have not committed yet; and that unborn state, which the
    /// heap's claim leaves beside the growth (`DeviceHeap::claim_demand`,
    /// [`Self::unborn`]).
    fn workload_room(&self, kind: seismic::Contract) -> (Option<(u64, u64)>, u64) {
        let unborn = self.unborn(kind);
        let room = self
            .store_of(kind)
            .and_then(|store| self.pace.borrow().growth_room(store))
            .map(|(room, margin)| (room.saturating_sub(unborn), margin));
        (room, unborn)
    }

    /// The bank's reclaim target for this storage kind from the latest
    /// probe. The release event reads this same owner before releasing; it
    /// neither samples memory again nor changes the pacing decision.
    pub fn expert_reclaim_deficit(&self, kind: seismic::Contract) -> u64 {
        match self.store_of(kind).map(|store| self.shortfall(store)) {
            Some(super::Shortfall::Short(bytes)) => bytes,
            _ => 0,
        }
    }

    /// What `store` is short of now (`MemoryPace::shortfall`).
    fn shortfall(&self, store: seismic::StoreId) -> super::Shortfall {
        let charged = self
            .domain
            .device()
            .ledgers()
            .own_claims()
            .into_iter()
            .filter(|(of, _)| *of == store)
            .map(|(_, bytes)| bytes)
            .sum();
        self.pace.borrow().shortfall(store, charged)
    }

    /// Whether all the bank's stores are short of is released already, and
    /// only the host has yet to show it: the bank's rung waits, as does
    /// everything costlier, for a later reading.
    pub fn expert_reclaim_covered(&self) -> bool {
        let shortfalls = crate::planning::BankTier::ALL.map(|kind| self.store_of(kind.storage()).map(|store| self.shortfall(store)));
        let mut read = shortfalls.iter().flatten();
        read.clone().any(|shortfall| *shortfall == super::Shortfall::Covered)
            && !read.any(|shortfall| matches!(shortfall, super::Shortfall::Short(_)))
    }

    /// Take everything released as shown (`MemoryPace::lapse_credit`).
    pub fn lapse_expert_reclaim_credit(&self) {
        self.pace.borrow_mut().lapse_credit();
    }

    /// The store `kind` storage is placed in first; none where the
    /// executor places none.
    pub(super) fn store_of(&self, kind: seismic::Contract) -> Option<seismic::StoreId> {
        let device = self.domain.device();
        storage_store(device.topology(), device.info(), kind)
    }

    /// Whether `kind` storage lies in device storage's store, beside the
    /// workload's state (on unified memory, any).
    fn beside_device(&self, kind: seismic::Contract) -> bool {
        self.store_of(kind) == self.store_of(seismic::Contract::DeviceOnly)
    }

    /// What the process holds now (`MemoryMoment`).
    pub fn memory_moment(&self) -> MemoryMoment {
        let ledger = self.domain.device().device_ledger();
        MemoryMoment {
            charged: ledger.charged,
            claimed: ledger.claim,
            footprint: seismic::ProcessMemory::read().footprint_bytes,
        }
    }

    /// The places of the prompt-group class held now (`grant_group`).
    pub fn group_places(&self) -> std::num::NonZeroUsize {
        self.resources.group_places()
    }

    /// The slots the expert bank commits now; none without an elastic bank.
    pub fn expert_bank_slots(&self) -> Option<u64> {
        self.family.expert_bank().map(|bank| bank.slots())
    }

    /// Where the load's routed experts are now (`ElasticBank::residency`);
    /// none without an elastic bank.
    pub fn expert_residency(&self) -> Result<Option<super::ExpertResidency>, DomainError> {
        self.family.expert_bank().map(|bank| bank.residency()).transpose()
    }

    /// Wait until the expert bank's host tier is filled
    /// (`ElasticBank::wait_tier_filled`); at once without a bank or tier.
    pub fn wait_expert_tier_filled(&self) -> Result<(), DomainError> {
        self.family.expert_bank().map_or(Ok(()), |bank| bank.wait_tier_filled())
    }

    /// Starve the expert bank's host for `duration` from now
    /// (`ExpertBank::starve`): a test's way to end the waits of the steps
    /// it submits meanwhile before their answers.
    #[cfg(test)]
    pub(crate) fn starve_expert_host(&self, duration: std::time::Duration) {
        let starved = self.family.expert_bank().is_some_and(|bank| bank.starve(duration));
        assert!(starved, "a streamed load's host is starved");
    }

    /// The release ladder's bank rung: one shrink step
    /// ([`Self::shrink_expert_bank_of`]) covering what the latest probe says
    /// each storage kind's domain is short until the next can take effect
    /// (`MemoryPace::shortfall`): the ladder probes before each release.
    pub fn shrink_expert_bank(
        &mut self,
        bindings: &mut StateBindings,
        wake: crate::CompletionWake,
    ) -> Result<super::BankStep, DomainError> {
        let deficits = crate::planning::BankTier::ALL.map(|kind| (kind, self.expert_reclaim_deficit(kind.storage())));
        let deficit = move |kind| deficits.iter().find(|(of, _)| *of == kind).map_or(0, |&(_, bytes)| bytes);
        self.shrink_expert_bank_of(bindings, &deficit, wake)
    }

    /// [`Self::shrink_expert_bank_of`] covering `deficit` bytes of device
    /// storage: what a prompt group's places take.
    pub fn shrink_expert_bank_by(
        &mut self,
        bindings: &mut StateBindings,
        deficit: u64,
        wake: crate::CompletionWake,
    ) -> Result<super::BankStep, DomainError> {
        let device = move |kind| if kind == crate::planning::BankTier::Bank { deficit } else { 0 };
        self.shrink_expert_bank_of(bindings, &device, wake)
    }

    /// Begin one shrink step of the bank toward the floor by the slots
    /// covering `deficit` bytes of a holding's storage kind
    /// (`ElasticBank::shrink`), the queued steps orphaned first, as they may
    /// name records it retires; its memory is released at later flight
    /// boundaries ([`Self::advance_expert_bank`]), `wake` completed as it
    /// progresses. The shrink's step records no graph: the decode launch's
    /// were recorded as its slots were made (`NativeGraphPool::grow`). A
    /// load without an elastic bank is at its limit.
    fn shrink_expert_bank_of(
        &mut self,
        bindings: &mut StateBindings,
        deficit: &dyn Fn(crate::planning::BankTier) -> u64,
        wake: crate::CompletionWake,
    ) -> Result<super::BankStep, DomainError> {
        let started = std::time::Instant::now();
        let Some(bank) = self.family.expert_bank() else {
            return Ok(super::BankStep::AtLimit);
        };
        let resident = self.experts.as_ref().and_then(|measures| measures.resident.measured().map(|resident| &**resident));
        let moves = self.experts.as_ref().and_then(|measures| measures.moves.measured());
        let step = bank.shrink(resident, moves, deficit, &mut || self.queue.orphan(self.experts.as_ref(), bindings), wake)?;
        match step {
            super::BankStep::Begun(tier) => {
                self.resize_held.begin(tier, started);
                self.shrink_step = true;
            }
            _ => self.resize_held.hold(started, super::HeldSource::Decision),
        }
        Ok(step)
    }

    /// The storage kinds whose next growth step of the expert bank
    /// (`ElasticBank::growth`, on the resident step this load was planned
    /// with, each to its target within its store's room, `growth_target`)
    /// pays, each store's own: a holding no resize has measured takes its
    /// first step, which measures what a step holds the submitting thread;
    /// after that, a step pays where what it saves the decode the load's
    /// latest miss record counted exceeds what its holding's last resize
    /// held the thread (`growth_target` names the rule's two gaps); one that
    /// does not pay grows nothing. Each store's step is offered, paying or
    /// not ([`super::GrowthOffer`]). Device memory and host memory are
    /// separate rooms, so neither's step waits on the other's: they come in
    /// turn, the kind after the last to grow first. Without a resident step
    /// nothing grows in device memory, nor while a prompt's places want
    /// device memory from the bank (`PlacesWant`); the host tier grows
    /// regardless, its step always paying; nothing grows in a store the heap
    /// reads in Reclaim or blind, nor at all beside the measurement job's
    /// claim ([`Self::claim_measurement_room`]).
    pub fn expert_bank_growths(&self) -> Result<Vec<super::GrowthOffer>, DomainError> {
        let resident = self.experts.as_ref().and_then(|measures| measures.resident.measured().map(|resident| &**resident));
        let Some(bank) = self.family.expert_bank().filter(|_| self.measurement_claim.is_none()) else {
            return Ok(Vec::new());
        };
        let held = &self.resize_held.last;
        let mut kinds = crate::planning::BankTier::ALL;
        if self.grown_last == Some(crate::planning::BankTier::Bank) {
            kinds.reverse();
        }
        let mut offers = Vec::new();
        for kind in kinds {
            // A prompt's places want device memory, which a device step
            // would take; and a device step is priced on the resident step.
            if kind == crate::planning::BankTier::Bank && (self.places_want.is_some() || resident.is_none()) {
                continue;
            }
            let room = match self.growth_room(kind) {
                Ok((room, _)) => room,
                // Nothing grows in Reclaim, nor blind.
                Err(ClaimRefusal::Reclaim(_) | ClaimRefusal::Blind(_)) => continue,
                Err(refusal) => return Err(refusal.into()),
            };
            let Some(growth) = bank.growth(resident, &|holding| held.contains_key(&holding), kind, room)? else {
                continue;
            };
            // The host tier grows to every expert wherever there is room:
            // nothing prices it.
            let pays = growth.holding == super::BankHolding::Host
                || held.get(&growth.holding).is_none_or(|&held| growth.saving.is_some_and(|saved| saved > held));
            offers.push(super::GrowthOffer { growth, pays });
        }
        Ok(offers)
    }

    /// Begin one growth step of the expert bank in `kind` storage
    /// (`ElasticBank::grow`): its new bytes, which its reservation backs
    /// with fresh pages of the device's store or host RAM
    /// (`DeviceHeap::claim_reserved`), are claimed from the heap first, and
    /// held until the growth finishes ([`Self::advance_expert_bank`]); a
    /// heap that refuses the claim (`Reclaim`, `Capacity`, `Blind`) grows
    /// nothing; the claim holds the step to what the device's grants leave
    /// (Metal's working set), the step alone, as the bank's chunks are held
    /// once. The step is sized within what that claim admits now
    /// ([`Self::growth_room`]), so a refusal of it is the machine's change
    /// between the two readings, which the next decision reads afresh:
    /// where not even one slot fits, nothing grows.
    pub fn grow_expert_bank(
        &mut self,
        kind: crate::planning::BankTier,
        wake: crate::CompletionWake,
    ) -> Result<super::BankStep, DomainError> {
        let started = std::time::Instant::now();
        let resident = self.experts.as_ref().and_then(|measures| measures.resident.measured().map(|resident| &**resident));
        let Some(bank) = self.family.expert_bank() else {
            return Ok(super::BankStep::AtLimit);
        };
        let held = &self.resize_held.last;
        let measured = |holding| held.contains_key(&holding);
        let (room, unborn) = self.growth_room(kind)?;
        let Some(super::Growth { bytes, .. }) = bank.growth(resident, &measured, kind, room)? else {
            return Ok(super::BankStep::AtLimit);
        };
        let claim = self
            .memory
            .borrow_mut()
            .claim_reserved(kind, bytes, unborn, HoldingClass::Model)?;
        let grown = bank.grow(resident, &measured, kind, room, wake);
        match grown {
            Ok(super::BankStep::Begun(tier)) => {
                self.bank_claim = Some(super::GrowthClaim { id: claim, kind, bytes });
                self.grown_last = Some(kind);
                self.resize_held.begin(tier, started);
                Ok(super::BankStep::Begun(tier))
            }
            other => {
                self.release_claim(claim);
                other
            }
        }
    }

    /// The model has loaded: the displacement its load caused is accepted,
    /// and its growth reads what can be taken without displacing more of
    /// another program's data from now on (`DeviceHeap::serve`).
    pub fn serve(&mut self) -> Result<(), DomainError> {
        self.memory
            .borrow_mut()
            .serve()
            .map_err(|error| DomainError::Blind(error.to_string()))
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
        let Some(bank) = self.family.expert_bank() else {
            return Ok(None);
        };
        let Some(finished) = bank.advance(wake)? else {
            self.resize_held.hold(started, super::HeldSource::Advance);
            return Ok(None);
        };
        let holding = finished.holding();
        let tier = bank.holding_storage(holding)?;
        let held = self.resize_held.finish(started, tier, holding);
        match &finished {
            super::BankResized::Shrunk { times, .. } => self.pace.borrow_mut().released(times),
            // A growth's claim was the heap's hold on what it backs, which
            // Seismic now charges; a shrink beside it leaves it held.
            super::BankResized::Grown { .. } | super::BankResized::Stopped { .. } | super::BankResized::GrowthRefused { .. } => {
                if let Some(claim) = self.bank_claim.take() {
                    self.release_claim(claim.id);
                }
            }
        }
        // The heap classifies against the charge it last read.
        self.refresh_memory()?;
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

    /// Release graph activations and output slots grown beyond the held
    /// prompt-group class that no request or launch holds, and in Reclaim
    /// the held class down to the floor of one place (`grant_group`): its
    /// group buffers now, credited with the charge of the model storage the
    /// release drops whole, as Seismic measured it before, and the slots on
    /// the load's release thread
    /// (`AllocatedResources::release_idle_slots`), `wake` completed once
    /// they are released. The bytes released now, and the slots' release;
    /// while one runs nothing more is released, the class and its slots
    /// together, until a boundary finishes it.
    fn release_idle_graph_slots(&mut self, policy: ShrinkPolicy, wake: crate::CompletionWake) -> Result<(u64, SlotRelease), DomainError> {
        if self.resources.releasing_slots() {
            return Ok((0, SlotRelease::Running));
        }
        let reclaim = policy == ShrinkPolicy::Reclaim;
        // The class held, buffers and pools together, down to the floor.
        let buffers = reclaim && self.resources.group_places() > std::num::NonZeroUsize::MIN;
        let released = match buffers {
            false => 0,
            true => {
                // Every model allocation and its charge now, each once. Only
                // this thread drops a model allocation (one released off it
                // only shrinks a reservation, or is no model storage's), so
                // those released by the time it returns are what it freed:
                // no concurrent release or growth moves the credit.
                let mut held = std::collections::HashMap::new();
                self.family.visit_model_storage(&mut |tensor| {
                    let observer = tensor.observe_storage();
                    if let Some(charged) = observer.charged_bytes() {
                        held.entry(observer.identity()).or_insert((observer, charged));
                    }
                });
                // A group of one holds the floor's alone, which allocates none.
                self.family.hold_group(1).map_err(|error| DomainError::invariant(error.to_string()))?;
                self.refresh_memory()?;
                held.into_values().filter(|(observer, _)| observer.charged_bytes().is_none()).map(|(_, charged)| charged).sum()
            }
        };
        let slots = match self.resources.release_idle_slots(reclaim, wake) {
            true => SlotRelease::Began,
            false => SlotRelease::None,
        };
        Ok((released, slots))
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
        self.refresh_memory()?;
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

/// A failed slot allocation of `bytes`: a capacity deficit when the device
/// refused the bytes, otherwise a device failure.
fn growth_failure(error: seismic::TensorError, bytes: u64, growth: &str) -> DomainError {
    match error {
        seismic::TensorError::Execution(
            seismic::ExecutionError::AllocationCapacity { .. }
            | seismic::ExecutionError::AllocationFailed(_),
        ) => DomainError::Capacity(CapacityError {
            resource: ResourceKind::DeviceMemory,
            required: bytes,
            available: 0,
        }),
        other => DomainError::Device(crate::DeviceError::Execution(format!("{growth}: {other}"))),
    }
}

/// The measurement job's claim beside the model: the device bytes it
/// claims, and its hold of the bank's own traffic, which stands while the
/// claim does and drops with it.
pub(super) struct MeasurementClaim {
    pub(super) bytes: u64,
    pub(super) _traffic: Option<TrafficHold>,
}

/// What the plan charged that is not held yet (`ExecutorDomain::unborn`):
/// the planned workload state past what the stores committed, and each
/// optional component's planned weights while it is unbound (once bound,
/// its weights are the ledger's optional holding, so each is charged once).
fn unborn_bytes(workload: u64, committed: u64, components: [(bool, u64); 2]) -> u64 {
    let unbound = components.iter().filter(|&&(bound, _)| !bound).map(|&(_, weights)| weights).sum::<u64>();
    workload.saturating_sub(committed) + unbound
}

#[cfg(test)]
mod group_demand_tests {
    use super::*;

    #[test]
    fn group_forecast_projects_onto_the_banks_actual_store() {
        let demand = crate::memory::ClaimDemand { device: 10, reserved: 20, upload: 30, shared: 0, host: 40, staged: 7 };
        for (topology, backend, expected) in [
            (seismic::fixtures::recorded::m4_pro(), seismic::BackendName::Metal, 100),
            (seismic::fixtures::recorded::rtx3090(), seismic::BackendName::Cuda, 30),
        ] {
            let executor = topology.executors().iter().find(|executor| executor.backend() == backend).unwrap();
            assert_eq!(beside_bank_bytes(&topology, executor, demand).unwrap(), expected);
        }
    }

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

#[cfg(test)]
mod unborn_tests {
    use super::unborn_bytes;

    /// An unbound component's planned weights stay out of growth beside the
    /// workload's unborn state; a bound one's are the ledger's, not held
    /// out again.
    #[test]
    fn an_unbound_components_planned_weights_are_charged_once() {
        assert_eq!(unborn_bytes(10, 4, [(false, 3), (false, 5)]), 6 + 8, "both components unbound");
        assert_eq!(unborn_bytes(10, 4, [(true, 3), (false, 5)]), 6 + 5, "the head bound, the projector not");
        assert_eq!(unborn_bytes(10, 12, [(true, 3), (true, 5)]), 0, "everything held");
    }
}
