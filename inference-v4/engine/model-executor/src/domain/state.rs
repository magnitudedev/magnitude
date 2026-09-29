//! state lifecycle for the executor domain.

use super::*;
use magnitude_model_state::{
    GrowthChoice, Holder, RowDemand, StateHoldingCensus, StateReclaimShrinkShape,
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
    pub fn reclaim_state_shape(
        &self,
    ) -> Result<(StateReclaimShrinkShape, Option<StateReclaimShrinkShape>), String> {
        Ok((
            self.target_store
                .reclaim_shrink_shape()
                .map_err(|error| error.to_string())?,
            self.head_store
                .as_ref()
                .map(|store| store.reclaim_shrink_shape())
                .transpose()
                .map_err(|error| error.to_string())?,
        ))
    }
    /// Release bound optional components after the owner has drained all
    /// requests. The measured ledger delta is the only released-byte credit;
    /// shared target tensors remain cached by the head loader.
    pub fn release_idle_optional_components(&mut self) -> Result<u64, String> {
        if !self.target.is_empty()
            || !self.head.is_empty()
            || !self.input.is_empty()
            || self.lookahead.is_some()
        {
            return Ok(0);
        }
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
        live_checkpoints: &[&DomainCheckpoint],
        retained: &[&DomainCheckpoint],
    ) -> Result<MemoryChargeReconciliation, String> {
        let (target_state, head_state) = self.state_holding_census(live_checkpoints, retained)?;
        let state = target_state
            .total()
            .checked_add(head_state.map_or(0, StateHoldingCensus::total))
            .ok_or("state charge sum overflows")?;
        let graph_pools = self.resources.committed_bytes()?;
        let owned_media = self.owned_media_bytes(live_checkpoints, retained)?;
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

    /// Count each standalone feature allocation once across live requests and
    /// checkpoints. Graph-backed features already belong to a sealed output
    /// arena in `graph_pools` and must not be counted again.
    fn owned_media_bytes(
        &self,
        live_checkpoints: &[&DomainCheckpoint],
        retained: &[&DomainCheckpoint],
    ) -> Result<u64, String> {
        let inputs = self.input.values().chain(
            live_checkpoints
                .iter()
                .chain(retained.iter())
                .filter_map(|checkpoint| checkpoint.input.as_ref()),
        );
        let mut seen: Vec<&Tensor> = Vec::new();
        let mut bytes = 0u64;
        for feature in inputs.flat_map(|input| {
            input
                .images
                .values()
                .filter_map(|image| image.features.as_ref())
        }) {
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

    /// Account for each state arena from current requests, checkpoints, and
    /// the store's registered submitted transactions. Unregistered omissions
    /// remain an error rather than being mislabeled as surplus.
    pub fn state_holding_census(
        &self,
        live_checkpoints: &[&DomainCheckpoint],
        retained: &[&DomainCheckpoint],
    ) -> Result<(StateHoldingCensus, Option<StateHoldingCensus>), String> {
        let target_live = self
            .target
            .values()
            .map(Holder::State)
            .chain(
                live_checkpoints
                    .iter()
                    .map(|checkpoint| Holder::Checkpoint(&checkpoint.target)),
            )
            .collect::<Vec<_>>();
        let target_retained = retained
            .iter()
            .map(|checkpoint| Holder::Checkpoint(&checkpoint.target))
            .collect::<Vec<_>>();
        let target = self
            .target_store
            .holding_census(&target_live, &target_retained, &[])
            .map_err(|error| error.to_string())?;
        let head = self
            .head_store
            .as_ref()
            .map(|store| {
                let live = self
                    .head
                    .values()
                    .map(Holder::State)
                    .chain(
                        live_checkpoints
                            .iter()
                            .map(|checkpoint| {
                                checkpoint
                                    .head
                                    .as_ref()
                                    .map(Holder::Checkpoint)
                                    .ok_or("live checkpoint has no head state")
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    )
                    .collect::<Vec<_>>();
                let retained = retained
                    .iter()
                    .map(|checkpoint| {
                        checkpoint
                            .head
                            .as_ref()
                            .map(Holder::Checkpoint)
                            .ok_or("retained checkpoint has no head state")
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                store
                    .holding_census(&live, &retained, &[])
                    .map_err(|error| error.to_string())
            })
            .transpose()?;
        Ok((target, head))
    }

    /// A lazy component import can briefly hold an upload tensor alongside
    /// the resident weights. Admit that peak before the importer allocates.
    pub(super) fn grant_component_growth(
        &mut self,
        resident_bytes: u64,
        upload_peak: u64,
    ) -> Result<(), DomainError> {
        let required = resident_bytes
            .checked_add(upload_peak)
            .ok_or_else(|| DomainError::Input("component import claim overflows".into()))?;
        self.grant_device_growth(required, upload_peak)
    }

    /// Commit state backing for a launch before its capacity check: the rows
    /// and successor banks its operations will claim, placed so histories
    /// keep growing in place. A refused minimum grant returns a byte deficit.
    /// Histories at the segment limit are repacked first.
    pub fn provision(&mut self, operations: &[Operation]) -> Result<(), DomainError> {
        // A group claiming the queued lookahead needs nothing; any other
        // group first waits for the lookahead and releases it.
        if self.claim_slots(operations).is_some() {
            return Ok(());
        }
        self.orphan_lookahead()?;
        for operation in operations {
            if let Operation::Forward { request, .. } = operation {
                self.compact_target(*request)?;
            }
        }
        let mut target = Vec::new();
        let mut head = Vec::new();
        // A continuable step also queues its lookahead, which follows it by
        // one row and one bank.
        let lookahead = self.execution.policy().limits().lookahead;
        let mut target_banks = 0usize;
        for operation in operations {
            let rows = operation.row_count();
            match operation {
                Operation::Forward { request, .. } => {
                    if let Some(state) = self.target.get(request) {
                        let ahead = usize::from(
                            lookahead && lookahead::continuation_of(operation).is_some(),
                        );
                        target.push(state.demand(rows + ahead));
                        target_banks += 1 + ahead;
                    }
                }
                Operation::Head { request, .. } => {
                    if let Some(state) = self.head.get(request) {
                        head.push(state.demand(rows));
                    }
                }
                _ => {}
            }
        }
        if !target.is_empty() {
            self.grant_state_growth(self.target_store.clone(), &target, target_banks)?;
        }
        if !head.is_empty() {
            if let Some(store) = self.head_store.clone() {
                self.grant_state_growth(store, &head, head.len())?;
            }
        }
        match operations.first() {
            Some(Operation::Head { .. }) if !self.family.head_is_bound() => {
                let binding_constants = self
                    .head_loader
                    .as_ref()
                    .ok_or_else(|| DomainError::Input("head component is disabled".into()))?
                    .binding_constant_bytes()
                    .map_err(DomainError::Input)?;
                self.grant_component_growth(
                    self.execution.resources().bytes().head_weights,
                    self.execution
                        .load()
                        .head_upload_peak_bytes()?
                        .max(binding_constants),
                )?;
            }
            Some(Operation::Encode { .. }) if !self.family.vision_is_bound() => {
                self.grant_component_growth(
                    self.execution.resources().bytes().vision_weights,
                    self.execution.load().vision_upload_peak_bytes()?,
                )?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Grant elastic state growth from a fresh domain observation. Seismic's
    /// charge ledger remains authoritative: its limit enforces the grant on
    /// every physical allocation, including a temporary reallocation peak.
    fn grant_state_growth(
        &mut self,
        store: Rc<StateStore>,
        demands: &[RowDemand],
        banks: usize,
    ) -> Result<(), DomainError> {
        for choice in [GrowthChoice::Preferred, GrowthChoice::Minimum] {
            let claim = store.growth_claim(demands, banks)?;
            let required = match choice {
                GrowthChoice::Preferred => claim.preferred_bytes,
                GrowthChoice::Minimum => claim.minimum_bytes,
            };
            if required != 0 {
                if let Err(error) = self.grant_device_growth(required, 0) {
                    if choice == GrowthChoice::Preferred {
                        continue;
                    }
                    return Err(error);
                }
            }
            // Keep the peak reserved while the store performs its fallible
            // physical operation. Seismic measures the resulting charge; the
            // existing static holding is then resized from committed backing
            // instead of creating another holding for the same bytes.
            let claim = if required == 0 {
                None
            } else {
                Some(
                    self.memory
                        .borrow_mut()
                        .claim(crate::memory::MemoryNeed {
                            minimum_bytes: required,
                            preferred_bytes: required,
                            class: crate::memory::HoldingClass::Model,
                        })
                        .map_err(|error| {
                            DomainError::Blind(format!("state peak claim rejected: {error:?}"))
                        })?
                        .id,
                )
            };
            let provisioned = store.provision_with_growth(demands, banks, choice);
            if let Some(claim) = claim {
                self.memory
                    .borrow_mut()
                    .cancel_claim(claim)
                    .map_err(|error| {
                        DomainError::invariant(format!("state peak claim lost: {error:?}"))
                    })?;
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
                            magnitude_model_state::Error::Tensor(seismic::TensorError::Execution(
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
        self.grant_device_growth(0, 0)
    }

    /// Grant one peak physical claim of `required` device bytes, of which a
    /// dedicated device stages `staged` bytes through host RAM. The readings
    /// exclude existing charges, so only the added peak is compared with
    /// each domain's ceiling above its planning reserve.
    pub(super) fn grant_device_growth(
        &mut self,
        required: u64,
        staged: u64,
    ) -> Result<(), DomainError> {
        // Seismic enforces the allocation ceiling; the engine heap owns the
        // admission decision over the same readings.
        let readings = self.refresh_memory()?;
        let action = self
            .memory
            .borrow()
            .decide(crate::memory::MemoryNeed {
                minimum_bytes: required,
                preferred_bytes: required,
                class: crate::memory::HoldingClass::Surplus,
            })
            .map_err(|error| DomainError::Blind(format!("memory policy: {error:?}")))?;
        match action {
            crate::memory::MemoryAction::Grant { bytes } if bytes >= required => {}
            crate::memory::MemoryAction::Reject { available, .. } => {
                return Err(crate::CapacityError {
                    resource: crate::ResourceKind::DeviceMemory,
                    required,
                    available,
                }
                .into())
            }
            crate::memory::MemoryAction::Grant { bytes } => {
                return Err(crate::CapacityError {
                    resource: crate::ResourceKind::DeviceMemory,
                    required,
                    available: bytes,
                }
                .into())
            }
            crate::memory::MemoryAction::Wait => return Err(DomainError::Reclaim),
        }
        // A host-backed device has no staging domain: its staged bytes are
        // part of `required` on the same host domain.
        if let Some(staging) = readings
            .iter()
            .find(|reading| reading.role == DomainRole::Staging)
        {
            if staged > staging.ceiling_bytes {
                return Err(crate::CapacityError {
                    resource: crate::ResourceKind::HostStaging,
                    required: staged,
                    available: staging.ceiling_bytes,
                }
                .into());
            }
        }
        Ok(())
    }

    /// Repack the target history of a request at the visible segment limit
    /// into one contiguous run with the bit-exact state copy program, so its
    /// next launch cannot exceed the limit. Rare by construction (placement
    /// keeps histories in few runs), so the copy completes before the state
    /// publishes the new run and the launch proceeds.
    pub(super) fn compact_target(&mut self, request: RequestId) -> Result<(), DomainError> {
        let Some(state) = self.target.get(&request) else {
            return Ok(());
        };
        if !state.compaction_needed() || !self.family.state_is_bound() {
            return Ok(());
        }
        let rows = state
            .history_ranges()
            .iter()
            .map(|(_, count)| count)
            .sum::<usize>();
        self.grant_state_growth(
            self.target_store.clone(),
            &[magnitude_model_state::RowDemand { after: None, rows }],
            0,
        )?;
        let state = self.target.remove(&request).expect("state checked above");
        // One copy launch of at most the largest prepared copy class.
        let max_rows = self.execution.policy().limits().max_batch_rows;
        let compaction = match OwnedCompaction::prepare(state, max_rows) {
            Ok(OwnedCompactionPreparation::Ready(compaction)) => compaction,
            Ok(
                OwnedCompactionPreparation::NotNeeded(state)
                | OwnedCompactionPreparation::Deferred { state, .. },
            ) => {
                self.target.insert(request, state);
                return Ok(());
            }
            Err((state, error)) => {
                self.target.insert(request, state);
                return Err(error.into());
            }
        };
        let workspace = match self.resources.state_graph().acquire_workspace() {
            Ok(workspace) => workspace,
            Err(error) => {
                self.target.insert(request, compaction.abort());
                return Err(DomainError::Capacity(error));
            }
        };
        let Some(class_rows) = crate::batching::row_class(compaction.rows()) else {
            self.target.insert(request, compaction.abort());
            return Err(self.fatal_invariant("a copy within the batch bound has no row class"));
        };
        let batch = match crate::batching::ValidatedStateBatch::copy(
            compaction.copies().to_vec(),
            self.target_store.history_capacity(),
            class_rows,
        ) {
            Ok(batch) => batch,
            Err(error) => {
                self.target.insert(request, compaction.abort());
                return Err(DomainError::invariant(error.to_string()));
            }
        };
        let inputs = StateLaunchInputs::new(batch, StateWork::Copy(compaction), workspace);
        let launch =
            match ValidatedStateLaunch::new(inputs, &self.target_store, None, self.domain.id()) {
                Ok(launch) => launch,
                Err((_, error)) => return Err(self.fatal_invariant(error.to_string())),
            };
        let submission = match self.family.submit_state(launch) {
            Ok(submission) => submission,
            Err((error, _)) => {
                let failure = DomainError::from(error);
                self.fatal = Some(failure.clone());
                return Err(failure);
            }
        };
        let completed = match submission.finish() {
            Ok(completed) => completed,
            Err(error) => {
                let failure = DomainError::Device(error);
                self.fatal = Some(failure.clone());
                return Err(failure);
            }
        };
        let (core, _) = completed.into_parts();
        let (_, work) = core.into_parts();
        let StateWork::Copy(compaction) = work else {
            return Err(self.fatal_invariant("state copy returned another maintenance kind"));
        };
        self.target.insert(request, compaction.commit());
        Ok(())
    }

    /// Commit the successor banks a newly opened request needs.
    pub fn provision_open(&mut self) -> Result<(), DomainError> {
        let requirements = self.open_requirements();
        self.grant_state_growth(self.target_store.clone(), &[], requirements.target_banks())?;
        if let Some(store) = self.head_store.clone() {
            self.grant_state_growth(store, &[], requirements.head_banks())?;
        }
        self.sync_static_holding().map_err(DomainError::Input)?;
        Ok(())
    }

    /// Release committed state backing beyond what the stores need (see
    /// [`StateStore::shrink`]): at idle with hysteresis, under a memory
    /// deficit or Reclaim fully. Returns the physical bytes released.
    /// A queued lookahead is released first: it holds a state transaction,
    /// under which nothing shrinks.
    pub fn shrink_state(
        &mut self,
        policy: magnitude_model_state::ShrinkPolicy,
    ) -> Result<u64, DomainError> {
        self.orphan_lookahead()?;
        let mut released = self.target_store.shrink(policy)?;
        if let Some(store) = &self.head_store {
            released += store.shrink(policy)?;
        }
        self.sync_static_holding().map_err(DomainError::Input)?;
        Ok(released)
    }
}
