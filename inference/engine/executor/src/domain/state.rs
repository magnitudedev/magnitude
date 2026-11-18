//! state lifecycle for the executor domain.

use super::*;
use crate::planning::graph_demand;
use magnitude_state::{GrowthChoice, RowDemand, StateStore, StoreCopy};

/// The storage of `kind` this executor places first: its store.
pub(super) fn storage_store(topology: &seismic::DeviceTopology, executor: &seismic::Executor, kind: seismic::Contract) -> Option<seismic::StoreId> {
    executor.paths(kind).first().and_then(|path| topology.access_path(*path)).map(|path| path.store())
}

/// What a yield of the state's spare slabs came to
/// ([`ExecutorDomain::yield_state`]): the bytes of the granules the stores
/// released, and the bytes the state's pool returned to the OS for them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateYield {
    pub released: u64,
    pub returned: u64,
}

/// What a shrink of the state's surplus came to ([`ExecutorDomain::shrink_state`]):
/// the state's yield, and the idle graph slots' release off the submitting
/// thread, which a later boundary finishes
/// (`ExecutorDomain::advance_slot_release`).
#[cfg(any(test, feature = "fixtures"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateShrink {
    pub state: StateYield,
    pub slots: SlotRelease,
}

/// Where the idle graph slots' release stands after a shrink of the state's
/// surplus.
#[cfg(any(test, feature = "fixtures"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotRelease {
    /// No slot was idle.
    None,
    /// This shrink began releasing them.
    Began,
    /// One begun before still runs: this shrink released nothing more.
    Running,
}

/// What one state store commits and what its holders occupy of it, read
/// from the store's own record of its claims.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoreStateBytes {
    pub committed: u64,
    pub occupied: u64,
}

impl StoreStateBytes {
    fn of(store: &StateStore) -> Self {
        Self { committed: store.committed_bytes(), occupied: store.occupied_bytes() }
    }
}

/// The state stores' bytes: the target's and, where the drafter keeps its
/// own history, the drafter's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateBytes {
    pub target: StoreStateBytes,
    pub head: Option<StoreStateBytes>,
}

impl StateBytes {
    pub fn committed(self) -> u64 {
        self.target.committed + self.head.map_or(0, |head| head.committed)
    }

    pub fn occupied(self) -> u64 {
        self.target.occupied + self.head.map_or(0, |head| head.occupied)
    }
}

/// What a load holds as it ends, by the plan's terms beside what the
/// device's ledgers hold allocated (`allocated`, the one count of the
/// process's memory): the load's memory record. The terms describe the
/// allocation and are never summed against it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadCharge {
    pub allocated: u64,
    pub state: StateBytes,
    pub graph_pools: u64,
    pub owned_features: u64,
    pub target_weights: u64,
    pub optional_weights: u64,
    pub prepared_programs: u64,
    pub bound_constants: u64,
}

/// One store's allocation by class (integration spec §9.2), its ledger's
/// ([`crate::memory::ledger::Ledger::charge_classes`]).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StoreAllocation {
    pub store: seismic::StoreIdentity,
    /// The load's weights and the expert bank's records, their pool's
    /// spare included.
    pub model_bytes: u64,
    /// Sequences' state, the state floor and retained prefixes: the
    /// state's pool, its spare included.
    pub context_bytes: u64,
    /// Everything no other class holds: the arena, graph slots, group
    /// classes, scratch, outputs and the device's own memory.
    pub compute_bytes: u64,
    /// Optional components (MTP head, vision), resident or dormant.
    pub auxiliary_bytes: u64,
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
        let changes = self.resources.ledgers().changes().clone();
        changes.change(|keeping| {
            let mut held = self.claim_growth(
                &keeping.claims(),
                crate::memory::ledger::CommittedKind::Component,
                crate::memory::ClaimDemand { device: bytes, ..Default::default() },
            )?;
            let device = self.domain.device();
            let tensor = held
                .allocate(device, |charges| Tensor::zeros(charges, element, &[1, width]))
                .map_err(|error| DomainError::Input(error.to_string()))?;
            self.domain
                .publish_features(tensor, held.into_claims())
                .map_err(|error| DomainError::Input(error.to_string()))
        })
    }

    /// Whether no request is resident: only then may the optional
    /// components, which its steps draft and encode with, be released.
    pub(super) fn optional_idle(&self) -> bool {
        self.target.is_empty() && self.head.is_empty() && self.input.is_empty()
    }

    /// Value each resident request's reservation by what its replay costs
    /// per byte it holds (`Reservation::value`): the seconds one launch of
    /// its written rows takes resident from an empty history, on the load's
    /// measured step (`LoadCosts::step`), over the reservation's bytes. A
    /// shortage's relief preempts the request worth least first. None is
    /// valued before the load's step is measured; one that wrote no rows
    /// replays nothing and keeps no value.
    pub fn value_state(&self) -> Result<(), DomainError> {
        let Some(step) = self.load_costs().map(|costs| &costs.step) else {
            return Ok(());
        };
        for (sequence, state) in &self.state_claims {
            let Some(bytes) = std::num::NonZeroU64::new(state.bytes()).filter(|_| state.through() > 0) else {
                continue;
            };
            let rows = u32::try_from(state.through()).unwrap_or_else(|_| panic!("invariant violated: {sequence:?} writes rows past a launch's count"));
            let replay = step(rows, 0).map_err(|error| DomainError::invariant(format!("the replay of {sequence:?}: {error}")))?.resident;
            state.value(crate::memory::ledger::Value::per_byte(replay, bytes));
        }
        Ok(())
    }

    /// A test's release of the bound optional `components` once no request
    /// is resident, as the ledger's taking them makes it
    /// (`ExecutorDomain::release_yielded`): each idles as cache and is
    /// unloaded whole. What the device's ledger holds less is the
    /// released-byte credit; shared target tensors remain cached by the head loader.
    #[cfg(any(test, feature = "fixtures"))]
    pub fn release_idle_optional_components(
        &mut self,
        _bindings: &mut StateBindings,
        components: &[super::OptionalComponent],
    ) -> Result<u64, String> {
        if !self.optional_idle() || components.is_empty() {
            return Ok(0);
        }
        let before = self.held_bytes();
        self.idle_optional();
        let (released, kept): (Vec<_>, Vec<_>) =
            std::mem::take(&mut self.idle_components).into_iter().partition(|idle| components.contains(&idle.component));
        self.idle_components = kept;
        for idle in released {
            self.unload_component(idle.component, idle);
        }
        before
            .checked_sub(self.held_bytes())
            .ok_or_else(|| "optional component release increased what the process holds".to_owned())
    }

    /// The state stores' bytes, each read from its own record.
    pub fn state_bytes(&self) -> StateBytes {
        StateBytes {
            target: StoreStateBytes::of(&self.target_store),
            head: self.head_store.as_deref().map(StoreStateBytes::of),
        }
    }

    /// What each store's ledger holds allocated, by class
    /// ([`crate::memory::ledger::Ledger::charge_classes`]), each read in one
    /// hold: the classes sum to what the store has allocated.
    pub fn store_allocations(&self) -> Vec<StoreAllocation> {
        self.store_ledgers()
            .map(|ledger| {
                let classes = ledger.charge_classes();
                StoreAllocation {
                    store: ledger.store(),
                    model_bytes: classes.model,
                    context_bytes: classes.context,
                    compute_bytes: classes.compute,
                    auxiliary_bytes: classes.auxiliary,
                }
            })
            .collect()
    }

    /// Each store's ledger, the device's first.
    fn store_ledgers(&self) -> impl Iterator<Item = &crate::memory::ledger::Ledger> {
        let ledgers = self.resources.ledgers();
        let device = ledgers.device();
        std::iter::once(device).chain((ledgers.host().store() != device.store()).then(|| ledgers.host()))
    }

    /// What the load holds as it ends, by the plan's terms
    /// ([`LoadCharge`]).
    pub fn load_charge(&self) -> Result<LoadCharge, String> {
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
        let allocated = self.store_ledgers().map(|ledger| ledger.charge_classes().allocated()).sum::<u64>();
        Ok(LoadCharge {
            allocated,
            state: self.state_bytes(),
            graph_pools,
            owned_features,
            target_weights,
            optional_weights,
            prepared_programs,
            bound_constants,
        })
    }

    /// Commit state backing for a launch before its capacity check: the rows
    /// and successor banks its operations will claim, placed so histories
    /// keep growing in place. A refused minimum grant returns a byte deficit.
    /// Histories at the segment limit are repacked first. With a `ceiling`,
    /// growth that would take the stores' committed state past it grows
    /// nothing and returns [`DomainError::StateCeiling`]. A request's rows
    /// draw within its reservation, extended here, at the group's boundary,
    /// where the group reaches past it, so none waits on memory mid-launch;
    /// an extension short of memory is a capacity refusal the owner
    /// relieves, preempting others, or waits on. Every other
    /// allocation the group needs is made here too, where a refusal parks
    /// or refuses its requests alone: the optional components it binds,
    /// after the state's backing, and its priming boundary, kept for its
    /// reservation ([`Self::reserve`]), which then allocates nothing.
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
        // Any other group resolves the queue: a chain under way concludes,
        // its holder's state the one its reconciled chunks reached, from
        // which a group of its holder's that is not its queued chunk
        // (lowered at other rows than the chain planned) runs.
        self.conclude_chain()?;
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
        // The rows each request reaches through this group
        // (`claimed_through`), within its reservation.
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
                    .store(false)
                    .growth_claim(&target, &target_queued, target_banks)?
                    .minimum_bytes;
            }
            if head_banks != 0 && self.head_store.is_some() {
                growth = growth
                    .checked_add(
                        bindings
                            .store(true)
                            .growth_claim(&head, &head_queued, head_banks)?
                            .minimum_bytes,
                    )
                    .expect("invariant violated: state growth overflows");
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
        // A request whose group reaches past its reservation extends it
        // first, a launch's rows ahead (`RequestRows::reach`), and backs
        // the rows it added: short, the group is refused for memory here,
        // at its boundary, before any of its rows is written.
        // The extension settles the slabs in the same change, so the room it
        // reserved is committed room the backing draws on, never held anew.
        let rule = crate::memory::ledger::RequestRows::of(&self.execution);
        for &(sequence, through) in &state_rows {
            let held = self
                .state_claims
                .get(&sequence)
                .expect("invariant violated: a resident request holds no reservation")
                .rows();
            if through > held {
                self.change_state(|this, keeping| {
                    let reservation = this.state_claims.get_mut(&sequence).expect("a resident request's reservation, read above");
                    this.resources.ledgers().device().extend(&keeping.claims(), &this.request_slot, reservation, rule.reach(through))
                })
                .map_err(DomainError::request_short)?;
                self.back_reservation(bindings, sequence)?;
            }
        }
        for (sequence, through) in state_rows {
            self.state_claims
                .get_mut(&sequence)
                .expect("invariant violated: a resident request holds no reservation")
                .grow(through);
        }
        // Stores without history domains still grow banks.
        if target_banks != 0 {
            self.grant_state_growth(bindings, false, &target, &target_queued, target_banks)?;
        }
        if head_banks != 0 && self.head_store.is_some() {
            self.grant_state_growth(bindings, true, &head, &head_queued, head_banks)?;
        }
        self.change_state(|_, _| ());
        let Some(lane) = operations.first().map(ReservationLane::of) else {
            return Ok(());
        };
        self.bind_component(lane)?;
        // A primed chunk's drafter entry runs on the head.
        if operations.iter().any(|operation| matches!(operation, Operation::Forward { prime: Some(prime), .. } if !prime.tokens.is_empty())) {
            self.bind_component(ReservationLane::Head)?;
        }
        let bounded = operations.iter().any(|operation| matches!(operation, Operation::Forward { prime: Some(Priming { boundary: true, .. }), .. }));
        match (bounded, self.boundary.is_some()) {
            (false, _) => self.boundary = None,
            (true, false) => self.boundary = Some(self.allocate_boundary()?),
            (true, true) => {}
        }
        match operations.first() {
            Some(Operation::Encode { image, .. }) => {
                let class = self
                    .image_class(image.patches())
                    .map_err(DomainError::Input)?;
                self.provision_vision(class as u64)
            }
            _ => self.provision_output(lane, super::projected_rows(operations)),
        }
    }

    /// The state the stores' holders reference: the rows and banks some
    /// sequence or checkpoint holds (`StateStore::referenced_bytes`).
    fn referenced_state_bytes(&self) -> u64 {
        self.target_store.referenced_bytes() + self.head_store.as_ref().map_or(0, |store| store.referenced_bytes())
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
            .expect("invariant violated: a vision provision on a load without a vision graph");
        let device = self.domain.device();
        let activation = (pool.available_workspace(patch_rows) == 0)
            .then(|| {
                let graphs = graph_demand(device.graph_cost(), device.graph_budget(), pool.activation_graphs(patch_rows)?).map_err(DomainError::invariant)?;
                let demand = crate::memory::ClaimDemand { device: pool.activation_bytes(patch_rows)?, ..Default::default() }
                    .checked_plus(graphs)
                    .expect("invariant violated: a vision activation's demand overflows");
                Ok::<_, DomainError>(demand)
            })
            .transpose()?;
        let output = (pool.available_output(patch_rows) == 0)
            .then(|| pool.output_bytes(patch_rows))
            .transpose()?
            .map(|bytes| crate::memory::ClaimDemand { device: bytes, ..Default::default() });
        if let Some(demand) = activation {
            self.grow_vision(demand, |pool, device, held| pool.grow_activation(device, patch_rows, held))?;
        }
        if let Some(demand) = output {
            self.grow_vision(demand, |pool, device, held| pool.grow_output(device, patch_rows, held))?;
        }
        Ok(())
    }

    fn grow_vision(
        &mut self,
        demand: crate::memory::ClaimDemand,
        grow: impl FnOnce(&mut crate::VisionGraphPool, &seismic::Device, crate::memory::ledger::Held) -> Result<(), crate::resources::VisionGrowth>,
    ) -> Result<(), DomainError> {
        let bytes = demand.bytes();
        let changes = self.resources.ledgers().changes().clone();
        let grown = changes.change(|keeping| {
            let held = self.claim_growth(&keeping.claims(), crate::memory::ledger::CommittedKind::Component, demand)?;
            Ok::<_, DomainError>(grow(
                self.resources
                    .vision_graph_mut()
                    .expect("the vision graph was found above"),
                self.domain.device(),
                held,
            ))
        })?;
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
        let changes = self.resources.ledgers().changes().clone();
        changes.change(|keeping| {
            let held = self.claim_growth(
                &keeping.claims(),
                crate::memory::ledger::CommittedKind::Outputs,
                crate::memory::ClaimDemand { device: bytes, ..Default::default() },
            )?;
            self.resources
                .retained_outputs_mut(lane, projected)
                .expect("the pool was found above")
                .grow_claimed_output(self.domain.device(), held)
                .map_err(|error| growth_failure(error, bytes, "graph output growth"))
        })
    }

    /// Let go of `sequence`'s reservation, as it closes or is evicted.
    /// Back every row `request`'s reservation holds past its position with
    /// slabs now, in both stores: rows backed and admitted by no step, as
    /// queued rows are. So the OS holds every row the request has reserved,
    /// as it opens and as it extends, within the room its reservation
    /// committed, and no row it writes allocates mid-launch.
    pub(super) fn back_reservation(&mut self, bindings: &mut StateBindings, request: RequestId) -> Result<(), DomainError> {
        let unwritten = self.unwritten_rows(request);
        let target = self
            .target
            .get(&request)
            .expect("invariant violated: a reserved request holds no target state")
            .queued_demands(0, unwritten);
        self.grant_state_growth(bindings, false, &[], &target, 0)?;
        if let (Some(head), true) = (self.head.get(&request), self.head_store.is_some()) {
            let head = head.queued_demands(0, unwritten);
            self.grant_state_growth(bindings, true, &[], &head, 0)?;
        }
        Ok(())
    }

    /// Whether a request other than `own` holds state on the device's
    /// ledger: resident, in a flight or not, so its end or eviction frees
    /// memory.
    pub fn holds_state_beside(&self, own: &std::collections::BTreeSet<RequestId>) -> bool {
        self.state_claims.keys().any(|request| !own.contains(request))
    }

    /// The rows `request`'s reservation holds past its target position.
    fn unwritten_rows(&self, request: RequestId) -> usize {
        let reservation = self.state_claims.get(&request).expect("invariant violated: a resident request holds no reservation");
        let position = self.target.get(&request).map_or(0, SequenceState::position) as u64;
        usize::try_from(reservation.rows().saturating_sub(position)).expect("invariant violated: a reservation's rows exceed the host's")
    }

    pub(super) fn drop_state_claim(&mut self, sequence: RequestId) {
        self.state_claims.remove(&sequence);
    }

    /// Grant elastic state growth of the target or `head` store from a fresh
    /// domain observation. Growth changes bindings, so the queued steps are
    /// orphaned first; its release may itself free what the growth needed.
    /// Seismic's charge ledger remains authoritative: its limit enforces the
    /// grant on each added slab allocation. The slabs draw on the committed
    /// room the request slot and the reservations hold, and only what they
    /// add past it is held anew ([`StateSlabs`]). `queued` are the rows of the
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
        let store = bindings.store(head);
        let claim = store.growth_claim(demands, queued, banks)?;
        if claim.minimum_bytes == 0 && claim.preferred_bytes == 0 {
            self.change_state(|_, _| store.provision_with_growth(demands, queued, banks, GrowthChoice::Minimum))?;
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
            let changes = self.resources.ledgers().changes().clone();
            let ledger = self.resources.ledgers().device();
            if let Err(short) = changes.change(|keeping| self.state_slabs.reserve(&keeping.claims(), ledger, required)) {
                if choice == GrowthChoice::Preferred {
                    continue;
                }
                return Err(DomainError::short(short, self.resources.ledgers()));
            }
            let grown = self.change_state(|_, _| store.provision_with_growth(demands, queued, banks, choice));
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
    /// (`memory::ledger::DeviceLedgers::claim`), in `claiming`, before the
    /// allocation is made in the same change; its holder marks it allocated
    /// once made and keeps it until the allocation is freed. Refused as a
    /// capacity deficit of the store short of it, for the owner's release
    /// order, or as blind where that store's reading failed.
    pub(super) fn claim_growth(
        &self,
        claiming: &crate::memory::ledger::Claiming,
        kind: crate::memory::ledger::CommittedKind,
        demand: crate::memory::ClaimDemand,
    ) -> Result<crate::memory::ledger::Held, DomainError> {
        let ledgers = self.resources.ledgers();
        ledgers.claim(claiming, self.domain.device(), kind, demand).map_err(|short| DomainError::short(short, ledgers))
    }

    /// Hold a head just bound for the class held, what it demands beyond
    /// the floor charged with its weights
    /// (`ProgramFamily::bound_head_demand`) claimed first
    /// ([`Self::claim_growth`]) in `claiming`, the binding's change, and
    /// kept with its binding. A refused claim unbinds the head.
    pub(super) fn hold_bound_head(&mut self, claiming: &crate::memory::ledger::Claiming) -> Result<(), DomainError> {
        let places = self.group_places().get();
        let demand = self.family.bound_head_demand(places).map_err(DomainError::Input)?;
        if demand == crate::memory::ClaimDemand::default() {
            return Ok(());
        }
        let mut held = match self.claim_growth(claiming, crate::memory::ledger::CommittedKind::Component, demand) {
            Ok(held) => held,
            Err(refused) => {
                self.unbind(super::OptionalComponent::Drafter);
                return Err(refused);
            }
        };
        let device = self.domain.device().clone();
        let bound = held.allocate(&device, |charges| self.family.hold_bound_head(charges, places));
        if let Err(error) = bound {
            self.unbind(super::OptionalComponent::Drafter);
            return Err(DomainError::invariant(error));
        }
        self.head_binding.extend(held.into_claims());
        Ok(())
    }

    /// Unbind `component`, releasing what its binding claimed.
    pub(super) fn unbind(&mut self, component: super::OptionalComponent) {
        // The binding's free and its claims' are one change.
        let changes = self.resources.ledgers().changes().clone();
        changes.change(|_| {
            self.family.unbind_optional(component);
            match component {
                super::OptionalComponent::Drafter => self.head_binding.clear(),
                super::OptionalComponent::Vision => self.vision_binding.clear(),
            }
        });
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
    pub fn provision_open(&mut self, unchained: &mut Unchained<'_>) -> Result<(), DomainError> {
        self.conclude_chain()?;
        let requirements = self.open_requirements();
        self.grant_state_growth(unchained.bindings(), false, &[], &[], requirements.target_banks())?;
        if self.head_store.is_some() {
            self.grant_state_growth(unchained.bindings(), true, &[], &[], requirements.head_banks())?;
        }
        Ok(())
    }

    /// A test's shrink of the state's surplus apart from the ledger: the
    /// load frees its spare slabs and idle slots only as the ledger takes
    /// them (`ExecutorDomain::release_yielded`). The state yields its slabs
    /// beyond the live requests' next steps ([`Self::yield_state`]), and
    /// idle graph slots are released off this thread, `wake` completed once
    /// they are ([`Self::release_idle_graph_slots`]): the state's yield, and
    /// where the slots' release stands, which a later boundary finishes
    /// ([`Self::advance_slot_release`]).
    #[cfg(any(test, feature = "fixtures"))]
    pub fn shrink_state(
        &mut self,
        bindings: &mut StateBindings,
        wake: crate::CompletionWake,
    ) -> Result<StateShrink, DomainError> {
        let mut unchained = self.unchained(bindings).expect("invariant violated: a test shrinks the state beside a chain");
        let state = self.yield_state(&mut unchained)?;
        let slots = self.release_idle_graph_slots(wake);
        Ok(StateShrink { state, slots })
    }

    /// Release both stores' slabs beyond the live requests' next steps
    /// ([`Self::shrink_store`]), moving no row or bank, and return what
    /// they freed to the OS through the state's pool, which empties its
    /// emptiest blocks where whole blocks fall short of it
    /// (`seismic::Pool::return_freed`): the one copy. The queued steps are
    /// orphaned first: their launch bindings pin slab placement.
    pub(super) fn yield_state(&mut self, unchained: &mut Unchained<'_>) -> Result<StateYield, DomainError> {
        self.conclude_chain()?;
        let bindings = unchained.bindings();
        let pool = bindings.target.pool().clone();
        self.change_state(|this, _| {
            let mut released = this.shrink_store(bindings, false)?;
            if this.head_store.is_some() {
                released += this.shrink_store(bindings, true)?;
            }
            let returned = pool
                .return_freed(released)
                .map_err(|error| DomainError::invariant(format!("the state's pool could not return its slabs' blocks: {error}")))?;
            Ok(StateYield { released, returned })
        })
    }

    /// Whether the load's release thread is releasing idle graph slots: a
    /// change under way whose end frees memory.
    pub fn releasing_slots(&self) -> bool {
        self.resources.releasing_slots()
    }

    /// At a flight boundary: the idle graph slots' release, once the load's
    /// release thread has ended it (`AllocatedResources::released_slots`);
    /// none while it runs.
    pub fn advance_slot_release(&mut self) -> Option<crate::resources::SlotsReleased> {
        self.resources.released_slots()
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
        if self.group_class.is_some() {
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
        // A closed group's idle slots, which this group's class takes up
        // again, are kept in the change's keep phase, their allocation kept
        // (`IdleClass::reclaim`): the class's claims then hold only what it
        // grows (`class_demand` counts only slots not yet allocated). Idle
        // slots an earlier claim took are released first
        // (`follow_cache_targets`): their memory is that claim's once freed,
        // so the group runs with one place until then. Opened with none, the
        // slots idle as cache again. One change: no reading sees them free.
        let changes = self.resources.ledgers().changes().clone();
        changes.change(|keeping| {
            let Some(idle) = self.idle_class.reclaim(&keeping) else {
                return Ok(());
            };
            let opened = self.claim_group_class(&keeping.claims(), chunks, quiet);
            match &mut self.group_class {
                Some(class) => class.idle.extend(idle),
                None => self.idle_class.idle(idle),
            }
            opened
        })
    }

    /// The class of `chunks` places or fewer, the most whose claims yield
    /// at once ([`Self::open_group`]): claims of the class's growth, the
    /// device's store's and host RAM's apart, in `claiming`.
    fn claim_group_class(&mut self, claiming: &crate::memory::ledger::Claiming, chunks: usize, quiet: super::Quiet) -> Result<(), DomainError> {
        let device = self.domain.device().clone();
        for places in (2..=chunks).rev() {
            let demand = class_demand(&self.execution, &mut self.resources, &self.family, &device, places)?;
            let ledgers = self.resources.ledgers();
            // Each store's part of the class, as its allocations lie: a
            // discrete GPU's slots hold their upload regions in host RAM.
            let (on_device, on_host) = ledgers.placed(&device, demand);
            let mut yielded = true;
            let claims = [(ledgers.device(), on_device), (ledgers.host(), on_host)]
                .into_iter()
                .filter_map(|(ledger, bytes)| {
                    let unit = std::num::NonZeroU64::new(bytes)?;
                    let (claim, units) = ledger.claim_units(claiming, crate::memory::ledger::CommittedKind::GroupClass, unit, 1);
                    yielded &= units == 1;
                    Some(claim)
                })
                .collect();
            if !yielded {
                continue;
            }
            let limits = self.execution.policy().limits();
            let grown = std::num::NonZeroUsize::new(places).expect("a class of two places or more");
            // The class's allocations and their record are one change: the
            // pools' slots and the group's buffers are made, each charged to
            // the claim of its store.
            let mut class = ledgers.holding(crate::memory::ledger::CommittedKind::GroupClass, claims);
            let held = class.allocate(&device, |charges| {
                self.resources
                    .grow_class(charges, limits.class_slots(places))
                    .and_then(|slots| self.family.hold_group(charges, grown, quiet).map(|()| slots))
                    .map(|slots| self.resources.hold_class(slots))
            });
            return match held {
                Ok(()) => {
                    // Claimed at the class's priced demand: each claim now
                    // holds what its allocations charged, so its graphs'
                    // price, which CUDA charges to the device's own entry,
                    // is not held twice, and the class turns cache whole
                    // as it closes.
                    let mut claims = class.into_claims();
                    claims.iter_mut().for_each(crate::memory::ledger::Claim::fit_allocated);
                    self.group_class = Some(GroupClass { places: grown, claims, idle: Vec::new() });
                    Ok(())
                }
                Err(
                    crate::resources::AllocationError::Tensor(error)
                    | crate::resources::AllocationError::Workflow(seismic::WorkflowError::TensorView(error)),
                ) => match growth_failure(error, on_device, "a prompt group's class") {
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

    /// Close the group whose class is held, once no chain of it is under
    /// way: its group buffers go back to the floor's, its pools hold one
    /// place again (their slots beyond it idle until released,
    /// `AllocatedResources::release_idle_slots`), and its claims turn into
    /// the idle slots' cache (`CacheKind::GraphSlots`). Only between flights ([`Self::quiet`]); otherwise the next close
    /// does it.
    pub(super) fn close_group(&mut self) -> Result<(), DomainError> {
        if matches!(self.queue, super::queue::Queue::Chain(_) | super::queue::Queue::Opening(_)) {
            return Ok(());
        }
        // The held class, its claims with it, is the one record that a
        // group holds more than one place: the pools hold its places from
        // its open to its close alone.
        let (GroupClass { claims, idle, .. }, quiet) = match (self.group_class.take(), self.quiet()) {
            (Some(class), Some(quiet)) => (class, quiet),
            (class, _) => {
                self.group_class = class;
                return Ok(());
            }
        };
        // The buffers' free, which Seismic returns to the claims that
        // charged them, and the claims' turn to cache are one change.
        let changes = self.resources.ledgers().changes().clone();
        changes.change(|_| {
            self.family.release_group(quiet).map_err(DomainError::invariant)?;
            self.resources.release_class();
            self.idle_class.idle(claims);
            self.idle_class.idle(idle);
            Ok::<(), DomainError>(())
        })
    }

    /// The relief that covers `lacks` bytes of `resource`, the memory of
    /// the store a claim was refused in, taking what `allow` admits
    /// (`memory::ledger::Ledger::plan_relief`); none where nothing covers it
    /// or the resource is no store's.
    pub fn plan_relief(&self, resource: ResourceKind, lacks: u64, allow: crate::memory::ledger::Relief) -> Option<crate::memory::ledger::ReliefPlan> {
        use crate::memory::ledger::Frees;
        let ledgers = self.resources.ledgers();
        let (ledger, frees) = match resource {
            ResourceKind::RequestState => (ledgers.device(), Frees::RequestState),
            ResourceKind::DeviceMemory => (ledgers.device(), Frees::Room),
            ResourceKind::HostStaging => (ledgers.host(), Frees::Room),
            _ => return None,
        };
        ledger.plan_relief(lacks, allow, frees)
    }

    /// Carry out `plan` whole in its store (`memory::ledger::Ledger::execute`):
    /// its cache yields now, each holder freeing it at its next release; its
    /// requests are the owner's to evict.
    pub fn execute_relief(&self, plan: crate::memory::ledger::ReliefPlan) -> crate::memory::ledger::Owed {
        let ledgers = self.resources.ledgers();
        match ledgers.keeping(&plan.store()) {
            Some(ledger) => ledger.change(|keeping| ledger.execute(&keeping.claims(), plan)),
            None => panic!("invariant violated: a relief plan of a store the load does not reach"),
        }
    }

    /// Hold `bytes` of device memory for the service's measurement job
    /// beside the model, in place of any grant before, which is dropped
    /// first: a grant on the device's ledger
    /// (`memory::ledger::GrantRole::Measurement`), taken of free room and
    /// the cache worth nothing, never the bank's or the retained prefixes';
    /// the job's footprint is credited against it once its process is named
    /// ([`Self::attach_measurement_process`]), so the job's growth is never
    /// read as others'. The bank's own traffic is held
    /// (`ElasticBank::hold_traffic`), so the job measures beside none of the
    /// load's copies or reads. A growth or the host tier's backing running
    /// ends at its next slot (`ElasticBank::stop_growths`): held, its slots
    /// would wait for the grant's release, which waits for the job's end.
    /// Short (`memory::ledger::Short`), nothing is held, and the job waits
    /// for the next quiet period.
    pub fn claim_measurement_room(&mut self, bytes: u64) -> Result<(), crate::memory::ledger::Short> {
        self.measurement_claim = None;
        let ledger = self.resources.ledgers().device();
        let grant = ledger.change(|keeping| ledger.grant(&keeping.claims(), crate::memory::ledger::GrantRole::Measurement, bytes))?;
        if let Some(bank) = self.family.expert_bank() {
            bank.stop_growths();
        }
        let traffic = self.hold_expert_traffic();
        self.measurement_claim = Some(MeasurementClaim { grant, _traffic: traffic });
        Ok(())
    }

    /// The measurement job holding the room runs as `process`: its
    /// footprint is credited against its grant from now until the room is
    /// released. Nothing where no grant is held (it was released first).
    pub fn attach_measurement_process(&mut self, process: seismic::ProcessId) {
        if let Some(claim) = &self.measurement_claim {
            claim.grant.attach(process);
        }
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


    /// The places of the prompt-group class held now: one, or an open
    /// group's ([`Self::open_group`]), read from its held class alone: a
    /// chain runs at most that many launches.
    pub fn group_places(&self) -> std::num::NonZeroUsize {
        self.group_class.as_ref().map_or(std::num::NonZeroUsize::MIN, |class| class.places)
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
    #[cfg(any(test, feature = "fixtures"))]
    fn release_idle_graph_slots(&mut self, wake: crate::CompletionWake) -> SlotRelease {
        if self.resources.releasing_slots() {
            return SlotRelease::Running;
        }
        match self.resources.release_idle_slots(self.idle_class.release(), wake) {
            true => SlotRelease::Began,
            false => SlotRelease::None,
        }
    }

    /// Release one store's slabs beyond the room its live requests'
    /// reservations and next steps claim
    /// (`magnitude_state::StoreBindings::shrink`), moving no row or bank:
    /// the bytes of the granules released.
    fn shrink_store(&mut self, bindings: &mut StateBindings, head: bool) -> Result<u64, DomainError> {
        // A live request's steps in flight (its next, and the one queued
        // behind it where the load pipelines or chains) each append a row
        // and take a successor bank, and every row its reservation holds is
        // backed from its open ([`Self::back_reservation`]): room the
        // shrink keeps, as no surplus.
        let steps = self.execution.policy().limits().target_successor_banks();
        let live = if head { &self.head } else { &self.target };
        let mut rows = Vec::new();
        for (request, state) in live {
            rows.extend(state.demands(self.unwritten_rows(*request).max(steps)));
        }
        let next = magnitude_state::NextSteps { rows, banks: steps * live.len() };
        Ok(bindings.store(head).shrink(&next)?)
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
                .expect("invariant violated: a store copy has no prepared row class");
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

/// The bytes of the load `execution`'s request slot
/// (`CommittedKind::RequestSlot`): one request's opening reservation, of a
/// request about to write no rows (`RequestRows::opening`), by the page
/// rule its reservations are sized by (`LoadShape::state_bytes`); at least
/// the state stores' first slabs, which construction backs as granules
/// within it.
pub fn request_slot_bytes(execution: &ExecutionPlan) -> Result<u64, String> {
    let resources = execution.resources();
    let initial = resources
        .head_state()
        .map_or(Ok(0), |state| state.initial_slab_bytes())
        .and_then(|head| resources.target_state().initial_slab_bytes().map(|target| target + head))?;
    let opening = crate::memory::ledger::LoadShape::of(execution).state_bytes(crate::memory::ledger::RequestRows::of(execution).opening(0));
    Ok(opening.max(initial))
}

/// The row a sequence's state claim reaches through a group that advances
/// it `rows` (its own and those queued behind it) from `position`: those
/// rows, and the most speculative rows its drafter adds past them
/// (`PlannedMethod::draft_rows`, none undrafted), so no row a step writes
/// is outside its claim.
pub(crate) fn claimed_through(position: usize, rows: usize, method: crate::PlannedMethod) -> u64 {
    (position + rows + method.draft_rows()) as u64
}

/// A closed group's class's idle graph slots, beyond one place, until
/// their pools release them: their one owner. Each is cache
/// (`CacheKind::GraphSlots`), its allocation kept, which any claim of the
/// process may take in process; the slots are then released, never
/// reused, as their memory is that claim's once freed
/// ([`ExecutorDomain::follow_cache_targets`]). A class that opens again
/// keeps them first ([`Self::reclaim`]).
#[derive(Default)]
pub(super) struct IdleClass(Vec<crate::memory::ledger::Cached>);

impl IdleClass {
    /// Idle `claims`, a class's committed in place, as cache.
    pub(super) fn idle(&mut self, claims: impl IntoIterator<Item = crate::memory::ledger::Claim>) {
        self.0.extend(claims.into_iter().map(|claim| claim.into_cache(crate::memory::ledger::CacheKind::GraphSlots)));
    }

    /// Whether a claim took any of the slots' cache in process: they then
    /// hold memory the ledger lent, and are released at once.
    pub(super) fn taken(&self) -> bool {
        self.0.iter().any(|cached| cached.releasing() > 0)
    }

    /// Every slot's cache, handed to their release
    /// (`AllocatedResources::release_idle_slots`).
    pub(super) fn release(&mut self) -> Vec<crate::memory::ledger::Cached> {
        std::mem::take(&mut self.0)
    }

    /// The slots kept whole for a class that takes them up again, in its
    /// change's keep phase ([`crate::memory::ledger::Keeping::keep`]):
    /// none, every slot left idle, where an earlier claim took any of them.
    fn reclaim(&mut self, keeping: &crate::memory::ledger::Keeping) -> Option<Vec<crate::memory::ledger::Claim>> {
        match keeping.keep(self.release(), crate::memory::ledger::CommittedKind::GroupClass) {
            Ok(claims) => Some(claims),
            Err(idle) => {
                self.0 = idle;
                None
            }
        }
    }
}

/// A prompt group's class held beyond one place ([`ExecutorDomain::open_group`]):
/// its places, the one record of them, and its committed claims, of its
/// device-side bytes on the device's ledger and of its host bytes on host
/// RAM's (none of a store it takes nothing of).
pub(super) struct GroupClass {
    /// Its places, two or more: the pools' slots and the group's buffers
    /// are held for them from its open to its close.
    pub(super) places: std::num::NonZeroUsize,
    /// Its claims, one in each store its allocations lie in.
    pub(super) claims: Vec<crate::memory::ledger::Claim>,
    /// The idle slots of a closed group the class took up again, committed
    /// in place.
    pub(super) idle: Vec<crate::memory::ledger::Claim>,
}

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// `change`, a change to the state slabs or to the requests'
    /// reservations, handed the change in its keep phase, and the device's
    /// ledger settled with the slabs the stores hold after it
    /// ([`StateSlabs::settle`]), as one change
    /// (`memory::ledger::Changes::change`).
    pub(super) fn change_state<R>(&mut self, change: impl FnOnce(&mut Self, crate::memory::ledger::Keeping<'_>) -> R) -> R {
        let changes = self.resources.ledgers().changes().clone();
        changes.change(|keeping| {
            let changed = change(self, keeping);
            // The slabs alone: the stores' address tables are committed by
            // their own claim, never room the reservations' rows fill.
            let backed = self.target_store.backed_slab_bytes() + self.head_store.as_ref().map_or(0, |store| store.backed_slab_bytes());
            let referenced = self.referenced_state_bytes();
            let ledger = self.resources.ledgers().device();
            let written = self
                .state_claims
                .iter()
                .map(|(sequence, state)| ledger.bytes(&crate::memory::ledger::Need::Request { sequence: *sequence, rows: state.through() }))
                .sum::<u64>();
            let committed = self.state_claims.values().map(crate::memory::ledger::Reservation::bytes).sum::<u64>() + self.request_slot.bytes();
            self.state_slabs.settle(ledger, backed, referenced, written, committed);
            changed
        })
    }
}

/// The state slabs the stores hold, against the committed room the load's
/// request slot and its requests' reservations hold on the device's ledger.
/// The pool's entry allocates every slab once
/// (`memory::ledger::Ledger::pool_block`); the holdings over its granules
/// move only what is held. Slabs the requests' written rows occupy, and
/// spare slabs within the committed room, are that room's; what holders
/// still reference beyond the written rows (retained prefixes) is
/// [`CacheKind::Prefix`](crate::memory::ledger::CacheKind::Prefix), and the
/// spare slabs past the committed room are spare granules. No slab byte is
/// held twice, and none reads as other programs' memory.
pub(super) struct StateSlabs {
    /// The committed room no slab backed at the last settle: what a growth
    /// draws on before anything is held anew.
    unbacked: u64,
    /// The spare slabs past the committed room.
    spare: SlabCache,
    /// The slabs retained prefixes alone reference.
    prefix: SlabCache,
    /// What a growth adds past the committed room, held before the stores
    /// allocate it in the pool: spare once settled ([`Self::settle`]).
    pending: Option<crate::memory::ledger::Claim>,
}

/// One kind of the slabs' cache: its holdings over the pool's granules, in
/// the pieces they were held in, and what yields took of it that the slabs
/// still hold. A yield lowers what the pieces hold at once (the ledger's
/// pooled cache); the granules stay the slabs' until they give them back.
struct SlabCache {
    kind: crate::memory::ledger::CacheKind,
    held: Vec<crate::memory::ledger::Cached>,
    /// What the pieces held after the last settle.
    settled: u64,
    /// What yields took that the slabs still hold.
    owed: u64,
}

impl SlabCache {
    fn new(kind: crate::memory::ledger::CacheKind) -> Self {
        Self { kind, held: Vec::new(), settled: 0, owed: 0 }
    }

    fn held(&self) -> u64 {
        self.held.iter().map(crate::memory::ledger::Cached::bytes).sum()
    }

    /// What yields took of its cache that the slabs still hold, before or
    /// after [`Self::settle`] folds the latest yield in.
    fn owed(&self) -> u64 {
        self.owed + self.settled.saturating_sub(self.held())
    }

    /// Settle with `slabs`, the bytes the slabs hold of its kind now: what
    /// yields took since the last settle is owed, the slabs gone pay what
    /// is owed first, and the pieces hold the rest, cut (the latest first)
    /// or held anew from the pool's spare granules.
    fn settle(&mut self, ledger: &crate::memory::ledger::Ledger, slabs: u64) {
        self.owed = self.owed().min(slabs);
        let bytes = slabs - self.owed;
        let mut held = self.held();
        while held > bytes {
            let at = self.held.len() - 1;
            let has = self.held[at].bytes();
            let cut = (held - bytes).min(has);
            match cut == has {
                true => drop(self.held.remove(at)),
                false => drop(self.held[at].split(cut, self.kind)),
            }
            held -= cut;
        }
        // Refused where the room is held already, so not held twice.
        if bytes > held {
            self.held.extend(ledger.cache(self.kind, bytes - held).ok());
        }
        self.held.retain(|cached| cached.bytes() > 0);
        self.settled = self.held();
    }
}

impl Default for StateSlabs {
    fn default() -> Self {
        use crate::memory::ledger::CacheKind;
        Self { unbacked: 0, spare: SlabCache::new(CacheKind::Spare), prefix: SlabCache::new(CacheKind::Prefix), pending: None }
    }
}

impl StateSlabs {
    /// Hold, before the stores allocate a growth of `bytes`, what it adds
    /// past the committed room no slab backs yet, as a pooled claim in
    /// `claiming` ([`Self::settle`] makes it spare). Refused, as its store refuses it,
    /// where free room lacks it; nothing is held then. A request's rows
    /// within its reservation add nothing.
    pub(super) fn reserve(
        &mut self,
        claiming: &crate::memory::ledger::Claiming,
        ledger: &crate::memory::ledger::Ledger,
        bytes: u64,
    ) -> Result<(), crate::memory::ledger::Short> {
        let required = bytes.saturating_sub(self.unbacked);
        if required == 0 {
            return Ok(());
        }
        assert!(self.pending.is_none(), "invariant violated: a state growth is reserved while another waits to settle");
        self.pending = Some(ledger.backing(claiming, crate::memory::ledger::PoolClass::State, required)?);
        Ok(())
    }

    /// What yields took of the retained prefixes' cache that the slabs
    /// still hold: what their holder evicts entries for.
    pub(super) fn prefix_owed(&self) -> u64 {
        self.prefix.owed()
    }

    /// Value the retained prefixes' cache (`Cached::value`): what the entry
    /// their holder evicts first saves ([`ExecutorDomain::value_cache`]).
    pub(super) fn value_prefix(&mut self, value: crate::memory::ledger::Value) {
        self.prefix.held.iter().for_each(|cached| cached.value(value));
    }

    /// What yields took of the spare slabs that the stores still hold:
    /// what releasing their empty slabs frees.
    pub(super) fn spare_owed(&self) -> u64 {
        self.spare.owed()
    }

    /// Settle with `backed`, the slab bytes the stores hold now in the
    /// pool, of which holders reference `referenced`, the requests' written
    /// rows `written` as their reservations count them, and `committed`, the room the
    /// request slot and the reservations hold. The written rows' slabs are
    /// the committed room's; what holders reference beyond them is the
    /// retained prefixes'; the spare slabs fill the committed room first,
    /// and the rest is spare granules. The reserved growth turns spare, and
    /// each cache settles ([`SlabCache::settle`]).
    pub(super) fn settle(&mut self, ledger: &crate::memory::ledger::Ledger, backed: u64, referenced: u64, written: u64, committed: u64) {
        let written = written.min(backed);
        let prefix = referenced.saturating_sub(written).min(backed - written);
        let spare = backed - written - prefix;
        let room = committed.saturating_sub(written);
        let within = spare.min(room);
        self.unbacked = room - within;
        if let Some(pending) = self.pending.take() {
            let cached = pending.into_backed_cache(crate::memory::ledger::CacheKind::Spare);
            self.spare.held.push(cached);
        }
        self.prefix.settle(ledger, prefix);
        self.spare.settle(ledger, spare - within);
    }
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
    let demand = family
        .group_demand(places)
        .map_err(DomainError::Input)?
        .checked_plus(crate::memory::ClaimDemand { device: pools, ..Default::default() })
        .and_then(|demand| demand.checked_plus(graphs))
        .expect("invariant violated: a group's demand overflows");
    Ok(demand)
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

/// The measurement job's grant on the device's ledger, and its hold of the
/// bank's own traffic, which stands while the grant does and drops with it.
pub(super) struct MeasurementClaim {
    grant: crate::memory::ledger::Grant,
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

