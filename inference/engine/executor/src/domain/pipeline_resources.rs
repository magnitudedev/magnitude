//! The prefix's allocation authority within the same ordinary request domain.
//! Numerical execution stays in PipelineNativeFamily; this is not another Owner.
use super::*;
use crate::pipeline::{ExecutableStage, PipelineRefusal, StageAllocation};

pub(super) struct PipelinePrefixOwner {
    pub(super) memory: RefCell<DeviceHeap>,
    pub(super) store: Rc<StateStore>,
    pub(super) arena: seismic::NativeExecutionArena,
    pub(super) pool: crate::NativeGraphPool,
    pub(super) state_pool: crate::NativeGraphPool,
    holding: HoldingId,
    target_weights: u64,
    prepared_programs: u64,
    bound_constants: u64,
}

pub(super) fn validate_binding_owner(
    device: &seismic::Device,
    expected: &Rc<StateStore>,
    bindings: &StoreBindings,
) -> Result<(), PipelineRefusal> {
    if !expected.belongs_to_device(device)
        || !bindings.belongs_to_device(device)
        || !Rc::ptr_eq(bindings, expected)
    {
        return Err(PipelineRefusal::ForeignDevice);
    }
    Ok(())
}

pub(super) fn validate_prefix_allocation(
    stage: &ExecutableStage,
    allocation: &StageAllocation,
) -> Result<(), PipelineRefusal> {
    let device = allocation.memory.heap().device();
    if !device.same_device(stage.device()) {
        return Err(PipelineRefusal::ForeignDevice);
    }
    validate_binding_owner(device, stage.state(), &allocation.bindings)?;
    let plan = allocation.execution.resources();
    if allocation.readout_pool.is_some()
        || !stage.assignment().owns_entry()
        || stage.assignment().owns_readout()
        || allocation.arena.bytes() != plan.arena_bytes()
        || plan.head_graph().is_some()
        || plan.vision_graph().is_some()
    {
        return Err(PipelineRefusal::UnsupportedProfile);
    }
    for (pool, charge) in [
        (&allocation.pool, plan.target_graph()),
        (&allocation.state_pool, plan.state_graph()),
    ] {
        if pool.domain() != &stage.domain
            || !pool.belongs_to(device)
            || !pool.uses_arena(&allocation.arena)
            || pool.committed_bytes() != charge.committed_bytes
        {
            return Err(PipelineRefusal::Preparation(
                "prefix pool differs from admitted device, domain, arena or footprint".into(),
            ));
        }
    }
    Ok(())
}

impl PipelinePrefixOwner {
    pub(super) fn adopt(
        stage: &ExecutableStage,
        allocation: StageAllocation,
    ) -> Result<(Self, StoreBindings), PipelineRefusal> {
        validate_prefix_allocation(stage, &allocation)?;
        let target_weights = stage
            .residency
            .resident_bytes()
            .map_err(|e| PipelineRefusal::Preparation(e.into()))?;
        let prepared_programs = stage
            .programs
            .device_storage_bytes()
            .map_err(|e| PipelineRefusal::Preparation(e.into()))?;
        let bound_constants = stage.constant_bytes()?;
        let StageAllocation {
            execution: _,
            arena,
            memory,
            bindings,
            pool,
            readout_pool: _,
            state_pool,
        } = allocation;
        let mut memory = memory.finish();
        let bytes = [
            target_weights,
            prepared_programs,
            bound_constants,
            arena.bytes(),
            pool.committed_bytes(),
            state_pool.committed_bytes(),
            bindings.committed_bytes(),
            bindings
                .external_pinned_bytes()
                .map_err(|e| PipelineRefusal::Preparation(e.to_string()))?,
        ]
        .into_iter()
        .try_fold(0u64, u64::checked_add)
        .ok_or_else(|| PipelineRefusal::Preparation("prefix charge overflows".into()))?;
        if bytes != memory.device().memory_usage().charged {
            return Err(PipelineRefusal::Preparation(
                "prefix adoption omits a physical allocation".into(),
            ));
        }
        // Observe physical storage before classifying it, exactly as the normal
        // domain does. Releasing the startup claim freed no physical storage.
        memory
            .refresh()
            .map_err(|e| PipelineRefusal::Preparation(e.to_string()))?;
        let holding = memory
            .heap_mut()
            .insert(bytes, HoldingClass::Model)
            .map_err(|e| PipelineRefusal::Preparation(format!("prefix holding: {e:?}")))?;
        Ok((
            Self {
                memory: RefCell::new(memory),
                store: Rc::clone(&bindings),
                arena,
                pool,
                state_pool,
                holding,
                target_weights,
                prepared_programs,
                bound_constants,
            },
            bindings,
        ))
    }

    pub(super) fn provision(
        &self,
        bindings: &mut StoreBindings,
        demands: &[RowDemand],
        banks: usize,
    ) -> Result<(), DomainError> {
        validate_binding_owner(self.memory.borrow().device(), &self.store, bindings)
            .map_err(|e| DomainError::invariant(e.to_string()))?;
        self.memory.borrow_mut().check(0, 0)?;
        if super::state::grant_store_growth(&self.memory, bindings, demands, banks)? {
            self.refresh()?;
        }
        Ok(())
    }

    fn reconcile(&self, live: &[Holder<'_>]) -> Result<MemoryChargeReconciliation, String> {
        let target_state = self
            .store
            .holding_census(live, &[], &[])
            .map_err(|e| e.to_string())?;
        let graph_pools = [
            self.arena.bytes(),
            self.pool.committed_bytes(),
            self.state_pool.committed_bytes(),
        ]
        .into_iter()
        .try_fold(0u64, u64::checked_add)
        .ok_or("prefix graph pool charge overflows")?;
        let external_pins = self
            .store
            .external_pinned_bytes()
            .map_err(|e| e.to_string())?;
        let classified = [
            target_state.total(),
            graph_pools,
            self.target_weights,
            self.prepared_programs,
            self.bound_constants,
            external_pins,
        ]
        .into_iter()
        .try_fold(0u64, u64::checked_add)
        .ok_or("prefix classified charge overflows")?;
        let charged = self.memory.borrow().device().memory_usage().charged;
        let unattributed = charged
            .checked_sub(classified)
            .ok_or("prefix classified holdings exceed Seismic's charge")?;
        Ok(MemoryChargeReconciliation {
            charged,
            target_state,
            head_state: None,
            graph_pools,
            activation_transfer: 0,
            owned_media: 0,
            target_weights: self.target_weights,
            optional_weights: 0,
            prepared_programs: self.prepared_programs,
            bound_constants: self.bound_constants,
            external_pins,
            unattributed,
        })
    }

    pub(super) fn refresh(&self) -> Result<Vec<DomainReading>, DomainError> {
        let mut memory = self.memory.borrow_mut();
        let readings = memory
            .refresh()
            .map_err(|e| DomainError::Blind(e.to_string()))?;
        // State backing may grow or retire while the immutable numerical family
        // keeps the same identity. Charge actual committed backing, not startup.
        let bytes = [
            self.target_weights,
            self.prepared_programs,
            self.bound_constants,
            self.arena.bytes(),
            self.pool.committed_bytes(),
            self.state_pool.committed_bytes(),
            self.store.committed_bytes(),
            self.store.external_pinned_bytes()?,
        ]
        .into_iter()
        .try_fold(0u64, u64::checked_add)
        .ok_or_else(|| DomainError::invariant("prefix static charge overflows"))?;
        memory
            .heap_mut()
            .resize(self.holding, bytes)
            .map_err(|e| DomainError::Input(format!("prefix holding update: {e:?}")))?;
        Ok(readings)
    }
}

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// Independent physical classification; this never adds two GPU budgets.
    pub fn pipeline_prefix_memory_charge(
        &self,
    ) -> Result<Option<MemoryChargeReconciliation>, String> {
        self.pipeline_owner
            .as_ref()
            .map(|owner| {
                let live = self
                    .pipeline_prefix
                    .values()
                    .map(Holder::State)
                    .collect::<Vec<_>>();
                owner.reconcile(&live)
            })
            .transpose()
    }
}
