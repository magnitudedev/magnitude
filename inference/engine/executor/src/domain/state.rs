//! state lifecycle for the executor domain.

use super::*;
use crate::memory::{ClaimId, HoldingClass};
use magnitude_state::{
    GrowthChoice, Holder, RowDemand, ShrinkPolicy, StateHoldingCensus, StoreCopy,
};

/// A comparison with Seismic's live charge. `unattributed` remains explicit
/// for any storage whose owner has not yet been classified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryChargeReconciliation {
    pub charged: u64,
    pub target_state: StateHoldingCensus,
    pub head_state: Option<StateHoldingCensus>,
    pub graph_pools: u64,
    pub owned_media: u64,
    pub target_weights: u64,
    pub optional_weights: u64,
    pub prepared_programs: u64,
    pub bound_constants: u64,
    pub external_pins: u64,
    pub unattributed: u64,
}

impl MemoryChargeReconciliation {
    pub fn complete(self) -> bool {
        self.unattributed == 0
    }
}

impl<F: ProgramFamily> ExecutorDomain<F> {
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
    /// requests; a queued lookahead, which drafts with them, is orphaned
    /// first. The measured ledger delta is the only released-byte credit;
    /// shared target tensors remain cached by the head loader.
    pub fn release_idle_optional_components(
        &mut self,
        bindings: &mut StateBindings<F>,
    ) -> Result<u64, String> {
        if !self.target.is_empty() || !self.head.is_empty() || !self.input.is_empty() {
            return Ok(0);
        }
        self.orphan_lookahead(bindings)
            .map_err(|error| error.to_string())?;
        let before = self.domain.device().memory_usage().charged;
        self.family.unbind_optional();
        if let Some(loader) = &self.head_loader {
            loader.unload();
        }
        if let Some(loader) = &self.vision_loader {
            loader.unload();
        }
        self.sync_optional_holding()?;
        let after = self.domain.device().memory_usage().charged;
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
        let graph_pools = self.resources.committed_bytes()?;
        let owned_media = self.owned_media_bytes(retained)?;
        let target_weights = self.family.target_weight_bytes()?;
        // The head loader inherits the target cache so tied weights are
        // represented by the same allocation. Vision has its own cache.
        let head_weights = self
            .head_loader
            .as_ref()
            .map(|loader| {
                loader
                    .resident_bytes()?
                    .checked_sub(target_weights)
                    .ok_or("head residency cache omits a target allocation")
            })
            .transpose()?
            .unwrap_or(0);
        let vision_weights = self
            .vision_loader
            .as_ref()
            .map(|loader| loader.resident_bytes())
            .transpose()?
            .unwrap_or(0);
        let optional_weights = head_weights
            .checked_add(vision_weights)
            .ok_or("optional resident charge overflows")?;
        let prepared_programs = self.family.prepared_program_bytes()?;
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
            .and_then(|bytes| bytes.checked_add(owned_media))
            .and_then(|bytes| bytes.checked_add(target_weights))
            .and_then(|bytes| bytes.checked_add(optional_weights))
            .and_then(|bytes| bytes.checked_add(prepared_programs))
            .and_then(|bytes| bytes.checked_add(bound_constants))
            .and_then(|bytes| bytes.checked_add(external_pins))
            .ok_or("classified charge sum overflows")?;
        let charged = self.domain.device().memory_usage().charged;
        let unattributed = charged
            .checked_sub(classified)
            .ok_or("classified holdings exceed Seismic's charge")?;
        Ok(MemoryChargeReconciliation {
            charged,
            target_state,
            head_state,
            graph_pools,
            owned_media,
            target_weights,
            optional_weights,
            prepared_programs,
            bound_constants,
            external_pins,
            unattributed,
        })
    }

    /// Count each standalone feature allocation once across request inputs
    /// and retained resume states. Graph-backed features already belong to a
    /// sealed output arena in `graph_pools` and must not be counted again.
    fn owned_media_bytes(&self, retained: &[&ResumeState]) -> Result<u64, String> {
        let features = self
            .input
            .values()
            .flat_map(|input| {
                input
                    .images
                    .values()
                    .filter_map(|image| image.features.as_ref())
            })
            .chain(retained.iter().flat_map(|state| state.features.values()));
        let mut seen: Vec<&Tensor> = Vec::new();
        let mut bytes = 0u64;
        for feature in features {
            if feature.allocation().is_graph_backed() {
                continue;
            }
            let tensor = feature
                .allocation()
                .tensor()
                .map_err(|error| error.to_string())?;
            if seen.iter().any(|other| other.shares_allocation(tensor)) {
                continue;
            }
            bytes = bytes
                .checked_add(tensor.storage_bytes())
                .ok_or("owned media charge overflows")?;
            seen.push(tensor);
        }
        Ok(bytes)
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
    /// Histories at the segment limit are repacked first.
    pub fn provision(
        &mut self,
        bindings: &mut StateBindings<F>,
        operations: &[Operation],
    ) -> Result<(), DomainError> {
        fn primed(operation: &Operation) -> Option<&Priming> {
            match operation {
                Operation::Forward { prime, .. } => prime.as_ref(),
                _ => None,
            }
        }
        // A group claiming the queued lookahead needs no rows of its own. The
        // step it queues next is optional and is queued only into space
        // already bound: growing now would change bindings under the claimed
        // step. Any other group resolves the lookahead first: its tentative
        // rows would otherwise read as another history's in every layout
        // question below (blocked tails, in-place rows, growth).
        if self.claim_slots(bindings, operations).is_some() {
            return Ok(());
        }
        self.orphan_lookahead(bindings)?;
        for operation in operations {
            if let Operation::Forward { request, .. } | Operation::Head { request, .. } = operation
            {
                self.relocate_tails(bindings, *request)?;
            }
        }
        let mut target = Vec::new();
        let mut head = Vec::new();
        // A continuable step also queues its lookahead, which follows it by
        // one row and one bank.
        let lookahead = self.execution.policy().limits().lookahead;
        let mut target_banks = 0usize;
        let mut head_banks = 0usize;
        for operation in operations {
            let rows = operation.row_count();
            match operation {
                Operation::Forward { request, prime, .. } => {
                    let next = lookahead
                        .then(|| self.successor_of(operation))
                        .flatten()
                        .map(|(next, _)| next);
                    // A primed chunk's drafter entry, and the entry of the
                    // chunk queued behind it.
                    let entries = [prime.as_ref(), next.as_ref().and_then(primed)]
                        .into_iter()
                        .flatten()
                        .map(|prime| prime.tokens.len())
                        .collect::<Vec<_>>();
                    if let (false, Some(state)) = (entries.is_empty(), self.head.get(request)) {
                        head.extend(state.demands(entries.iter().sum()));
                        head_banks += entries.len();
                    }
                    if let Some(state) = self.target.get(request) {
                        let ahead = next.as_ref().map_or(0, Operation::row_count);
                        target.extend(state.demands(rows + ahead));
                        target_banks += 1 + usize::from(ahead != 0);
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
        // Stores without history domains still grow banks.
        if target_banks != 0 {
            self.grant_state_growth(bindings, false, &target, target_banks)?;
        }
        if head_banks != 0 && self.head_store.is_some() {
            self.grant_state_growth(bindings, true, &head, head_banks)?;
        }
        // The import itself runs under a claim at reservation; admit its peak
        // here so a deficit reaches the owner's release order first.
        if let Some((required, staged)) = self.component_growth(operations)? {
            self.decide_device_growth(required, staged)?;
        }
        let lane = match operations.first() {
            Some(Operation::Forward { .. }) => ReservationLane::Target,
            Some(Operation::Head { .. }) => ReservationLane::Head,
            Some(Operation::Encode { .. }) => ReservationLane::Vision,
            None => return Ok(()),
        };
        self.provision_activation(lane)?;
        self.provision_output(lane)
    }

    /// Grow an on-demand family's activations by one when none is free: a
    /// family that holds none from startup (vision) claims its upload regions
    /// when a launch first needs them. The charge is claimed from the heap
    /// like any growth; a refusal is a demand deficit for the owner's
    /// release order.
    fn provision_activation(&mut self, lane: ReservationLane) -> Result<(), DomainError> {
        let Some(pool) = self.resources.activations_mut(lane) else {
            return Ok(());
        };
        if !pool.activates_on_demand() || pool.available_workspace() > 0 {
            return Ok(());
        }
        let bytes = pool.activation_slot_bytes();
        let claim = self.claim_device_growth(bytes, 0, HoldingClass::Live)?;
        let grown = self
            .resources
            .activations_mut(lane)
            .expect("the pool was found above")
            .grow_activation();
        self.release_claim(claim);
        grown.map_err(|error| match error {
            seismic::WorkflowError::TensorView(seismic::TensorError::Execution(
                seismic::ExecutionError::AllocationCapacity { .. }
                | seismic::ExecutionError::AllocationFailed(_),
            )) => DomainError::Capacity(CapacityError {
                resource: ResourceKind::DeviceMemory,
                required: bytes,
                available: 0,
            }),
            other => DomainError::Device(crate::DeviceError::Execution(format!(
                "graph activation growth: {other}"
            ))),
        })?;
        self.refresh_memory()?;
        self.sync_static_holding().map_err(DomainError::Input)
    }

    /// Grow `lane`'s retained-output pool by one slot when none is free, so
    /// the launch finds its output: a request keeps its last published
    /// output after its launch completes, so the pool follows the live
    /// requests. The slot's charge is claimed from the heap like any state
    /// growth; a refusal is a demand deficit for the owner's release order.
    pub(super) fn provision_output(&mut self, lane: ReservationLane) -> Result<(), DomainError> {
        let Some(pool) = self.resources.retained_outputs_mut(lane) else {
            return Ok(());
        };
        if pool.available_output() > 0 {
            return Ok(());
        }
        let bytes = pool.output_slot_bytes();
        let claim = self.claim_device_growth(bytes, 0, HoldingClass::Live)?;
        let grown = self
            .resources
            .retained_outputs_mut(lane)
            .expect("the pool was found above")
            .grow_output();
        self.release_claim(claim);
        grown.map_err(|error| match error {
            seismic::TensorError::Execution(
                seismic::ExecutionError::AllocationCapacity { .. }
                | seismic::ExecutionError::AllocationFailed(_),
            ) => DomainError::Capacity(CapacityError {
                resource: ResourceKind::DeviceMemory,
                required: bytes,
                available: 0,
            }),
            other => DomainError::Device(crate::DeviceError::Execution(format!(
                "graph output growth: {other}"
            ))),
        })?;
        self.refresh_memory()?;
        self.sync_static_holding().map_err(DomainError::Input)
    }

    /// Grant elastic state growth of the target or `head` store from a fresh
    /// domain observation. Growth changes bindings, so a queued lookahead is
    /// orphaned first; its release may itself free what the growth needed.
    /// Seismic's charge ledger remains authoritative: its limit enforces the
    /// grant on each added slab allocation.
    pub(super) fn grant_state_growth(
        &mut self,
        bindings: &mut StateBindings<F>,
        head: bool,
        demands: &[RowDemand],
        banks: usize,
    ) -> Result<(), DomainError> {
        // No backing changes when neither plan adds a slab. Reservation
        // observes memory again before launching the selected group.
        let claim = bindings.store(head)?.growth_claim(demands, banks)?;
        if claim.minimum_bytes == 0 && claim.preferred_bytes == 0 {
            return Ok(());
        }
        let store = bindings.store(head)?;
        for choice in [GrowthChoice::Preferred, GrowthChoice::Minimum] {
            let claim = store.growth_claim(demands, banks)?;
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
            let provisioned = store.provision_with_growth(demands, banks, choice);
            if let Some(claim) = claim {
                self.release_claim(claim);
            }
            match provisioned {
                Ok(()) => {
                    self.refresh_memory()?;
                    self.sync_static_holding().map_err(DomainError::Input)?;
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
                    continue
                }
                Err(error) => return Err(error.into()),
            }
        }
        unreachable!("minimum state growth either succeeds or returns a deficit")
    }

    /// Observe every domain the device uses and refresh the Seismic
    /// allocation ceiling: `Reclaim` while any domain's headroom is at or
    /// below its planning reserve, `Blind` when an observation fails.
    pub fn probe_memory(&mut self) -> Result<(), DomainError> {
        self.decide_device_growth(0, 0)
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
    /// Moving rows changes bindings: a queued lookahead is orphaned before
    /// the first relocation.
    pub(super) fn relocate_tails(
        &mut self,
        bindings: &mut StateBindings<F>,
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
        bindings: &mut StateBindings<F>,
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
        // `provision` resolved the lookahead: only another history can block.
        let Some(state) = self.lane_states(head).get(&request) else {
            return Ok(());
        };
        let demands = state.relocation_demands();
        if demands.is_empty() {
            return Ok(());
        }
        self.grant_state_growth(bindings, head, &demands, 0)?;
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
    /// change no bindings and leave a queued lookahead running.
    pub fn provision_open(&mut self, bindings: &mut StateBindings<F>) -> Result<(), DomainError> {
        self.orphan_lookahead(bindings)?;
        let requirements = self.open_requirements();
        self.grant_state_growth(bindings, false, &[], requirements.target_banks())?;
        if self.head_store.is_some() {
            self.grant_state_growth(bindings, true, &[], requirements.head_banks())?;
        }
        self.sync_static_holding().map_err(DomainError::Input)?;
        Ok(())
    }

    /// Release empty state slabs at idle, keeping one spare per store. In
    /// Reclaim, also compact sparse occupied slabs into already held space.
    /// Returns the physical bytes released. A queued lookahead is orphaned
    /// first: its launch binding pins slab placement.
    pub fn shrink_state(
        &mut self,
        bindings: &mut StateBindings<F>,
        policy: ShrinkPolicy,
    ) -> Result<u64, DomainError> {
        self.orphan_lookahead(bindings)?;
        let mut released = self.shrink_store(bindings, false, policy)?;
        if self.head_store.is_some() {
            released += self.shrink_store(bindings, true, policy)?;
        }
        Ok(released + self.release_idle_graph_slots()?)
    }

    /// Release graph activations and output slots grown beyond the startup
    /// counts that no request or launch holds, crediting only Seismic's
    /// measured decrease.
    fn release_idle_graph_slots(&mut self) -> Result<u64, DomainError> {
        let before = self.domain.device().memory_usage().charged;
        if self.resources.release_idle_slots() == 0 {
            return Ok(0);
        }
        self.sync_static_holding().map_err(DomainError::Input)?;
        before
            .checked_sub(self.domain.device().memory_usage().charged)
            .ok_or_else(|| DomainError::invariant("graph slot release increased Seismic's charge"))
    }

    /// Free empty slabs and compact into held slabs without a growth claim.
    /// Synchronize the static holding after Seismic reports the released bytes.
    fn shrink_store(
        &mut self,
        bindings: &mut StateBindings<F>,
        head: bool,
        policy: ShrinkPolicy,
    ) -> Result<u64, DomainError> {
        let bindings = bindings.store(head)?;
        let store = Rc::clone(bindings);
        let released =
            bindings.shrink_with(policy, |_, copy| self.submit_store_copy(&store, &copy));
        self.sync_static_holding().map_err(DomainError::Input)?;
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
