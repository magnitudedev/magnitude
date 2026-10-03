use super::graphs::{BoundStageGraphs, PreparedStageGraphs};
use super::{PipelineRefusal, StageAssignment};
use super::transfer::{ActivationBuffer, ActivationContract};
use crate::{
    memory::{ClaimId, HoldingClass},
    AttestedPrograms, DeviceHeap, NativeGraphPool, ResidencyStore, ResourceDomainId,
    StateResourcePlan,
};
use magnitude_artifacts::Package;
use magnitude_state::{StateStore, StoreBindings};
use seismic::{BackendName, Device};
use std::{collections::HashMap, rc::Rc};

/// Claimed local startup preparation, consumed into execution and allocation
/// ownership once. Neither side recreates an allocation when the Owner adopts it.
pub struct StageResources {
    pub(crate) executable: ExecutableStage,
    pub(crate) allocation: StageAllocation,
}
/// Device-bound numerical resources. The mutable store binding right and heap
/// are deliberately not part of this value; those belong to the request Owner.
pub struct ExecutableStage {
    pub(crate) assignment: StageAssignment,
    pub(crate) device: Rc<Device>,
    pub(crate) programs: Rc<AttestedPrograms>,
    pub(crate) residency: ResidencyStore,
    pub(crate) state_graphs: Rc<crate::PreparedStateCopyGraphs>,
    pub(crate) store: Rc<StateStore>,
    pub(crate) state_plan: StateResourcePlan,
    pub(crate) domain: ResourceDomainId,
    pub(crate) graphs: BoundStageGraphs,
    pub(crate) handoff: Option<std::cell::RefCell<ActivationBuffer>>,
}
/// Unique allocation authorities transferred into the ordinary domain. This is
/// not a second request lifecycle and cannot execute or publish a model step.
pub struct StageAllocation {
    pub(crate) execution: Rc<crate::ExecutionPlan>,
    pub(crate) arena: seismic::NativeExecutionArena,
    pub(crate) memory: StageStartupHeap,
    pub(crate) bindings: StoreBindings,
    pub(crate) pool: NativeGraphPool,
    pub(crate) readout_pool: Option<NativeGraphPool>,
    pub(crate) state_pool: NativeGraphPool,
}
impl StageResources {
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        assignment: StageAssignment,
        programs: Rc<AttestedPrograms>,
        draft: crate::ExecutionPlanDraft,
        mut heap: DeviceHeap,
        state_plan: StateResourcePlan,
        domain: ResourceDomainId,
        package: &Package,
    ) -> Result<Self, PipelineRefusal> {
        let device = heap.device().clone();
        if device.backend() != BackendName::Cuda
            || !programs.belongs_to(&device)
            || draft.device().selector() != device.info().selector
        {
            return Err(PipelineRefusal::ForeignDevice);
        }
        if package.identity() != assignment.definition().artifact_identity
            || !state_plan.is_for_stage(
                assignment.definition(),
                assignment.view().global_range(),
                draft.load(),
            )
        {
            return Err(PipelineRefusal::ForeignModel);
        }
        crate::resident_weights::validate_definition_package(assignment.definition(), package)
            .map_err(|e| PipelineRefusal::Preparation(e.to_string()))?;
        let load = draft.load().clone();
        let prepared =
            PreparedStageGraphs::prepare(&device, &programs, &assignment, &load, &state_plan)?;
        let charge = prepared.charge;
        let readout_charge = prepared.readout_charge;
        // All local families share one arena; independent output/upload pools
        // remain separately charged. Do not add workspace maxima together.
        let assigned = assignment.weights(&load)?;
        let weight_bytes = assignment.resident_bytes(&load)?;
        let rows = magnitude_batching::row_classes(state_plan.limits().max_launch_rows)
            .into_iter()
            .map(|r| r as u64)
            .collect::<Vec<_>>();
        let state_graphs = Rc::new(
            crate::PreparedStateCopyGraphs::prepare(
                &device,
                programs.stage_state(),
                crate::programs::native_state::state_copy_classes(
                    state_plan.target_state(),
                    None,
                    &rows,
                )
                .map_err(PipelineRefusal::Preparation)?,
            )
            .map_err(|e| PipelineRefusal::Preparation(e.to_string()))?,
        );
        let family = state_graphs.family();
        let state_charge = crate::NativeGraphCharge::from_checked(
            seismic::NativeGraphStorageBytes {
                workspace: family.workspace_bytes(),
                output: family.output_bytes(),
                upload: family.upload_bytes(),
            },
            1,
            crate::GraphSlots {
                activations: 1,
                output: 1,
            },
        )
        .map_err(PipelineRefusal::Preparation)?;
        let activation_contract = ActivationContract::new(
            state_plan.limits().max_launch_rows as u64,
            assignment.view().decoder().hidden,
        )?;
        let activation_bytes =
            activation_contract.bytes(state_plan.limits().max_launch_rows as u64)?;
        let handoff_bytes = if assignment.owns_entry() {
            0
        } else {
            activation_bytes
        };
        let upload = load
            .target_upload_peak_bytes()
            .map_err(PipelineRefusal::Preparation)?;
        let host_peak = upload
            .checked_add(activation_bytes)
            .ok_or_else(|| PipelineRefusal::Preparation("host staging charge overflows".into()))?;
        let empty = crate::NativeGraphCharge::from_checked(
            seismic::NativeGraphStorageBytes {
                workspace: 0,
                output: 0,
                upload: 0,
            },
            1,
            crate::GraphSlots {
                activations: 0,
                output: 0,
            },
        )
        .map_err(PipelineRefusal::Preparation)?;
        let plan = crate::ResourcePlanner::stage_plan_with_state(
            state_plan.clone(),
            &assignment,
            crate::planning::StageResourceCharges {
                target: charge,
                readout: readout_charge.unwrap_or(empty),
                state: state_charge,
                constant_bytes: prepared.constants_bytes,
                program_bytes: programs
                    .device_storage_bytes()
                    .map_err(|e| PipelineRefusal::Preparation(e.into()))?,
                activation_bytes: handoff_bytes,
            },
        )
        .map_err(PipelineRefusal::Plan)?;
        let workspace = plan.arena_bytes();
        let peak = plan.startup_peak_bytes();
        let execution = draft.admit(plan.clone()).map_err(PipelineRefusal::Plan)?;
        let mut residency = ResidencyStore::new(
            device.clone(),
            programs.clone(),
            execution.clone(),
            domain.clone(),
        )
        .map_err(|e| PipelineRefusal::Preparation(e.to_string()))?;
        let claim = heap
            .claim(peak, host_peak, HoldingClass::InFlight)
            .map_err(|refusal| PipelineRefusal::Memory {
                refusal,
                allocation: peak,
                staged: host_peak,
            })?;
        let result = (|| {
            let mut weights = HashMap::new();
            for planned in assigned {
                let weight = residency
                    .import_gguf(
                        package.target(),
                        &planned.descriptor,
                        crate::operators::resident_dtype(
                            planned.role.kind,
                            crate::resident_weights::activation_dtype(
                                assignment.definition().decoder.activation_dtype,
                            ),
                        ),
                    )
                    .map_err(|e| PipelineRefusal::Preparation(e.to_string()))?;
                if !weight.belongs_to(&device) {
                    return Err(PipelineRefusal::ForeignDevice);
                }
                if weights.insert(planned.role, weight).is_some() {
                    return Err(PipelineRefusal::InvalidWeightRole(planned.role));
                }
            }
            if residency
                .resident_bytes()
                .map_err(|e| PipelineRefusal::Preparation(e.into()))?
                != weight_bytes
            {
                return Err(PipelineRefusal::Preparation(
                    "assigned resident storage differs from local plan".into(),
                ));
            }
            let bindings = state_plan
                .target_state()
                .allocate(device.clone())
                .map_err(PipelineRefusal::Preparation)?;
            let arena = device
                .execution_arena(workspace)
                .map_err(|e| PipelineRefusal::Preparation(e.to_string()))?;
            let pool = NativeGraphPool::new(domain.clone(), &prepared.family, charge, &arena)
                .map_err(|e| PipelineRefusal::Preparation(e.to_string()))?;
            let readout_pool = prepared
                .readout
                .as_ref()
                .zip(readout_charge)
                .map(|(graphs, charge)| {
                    NativeGraphPool::new(domain.clone(), graphs.family(), charge, &arena)
                        .map_err(|e| PipelineRefusal::Preparation(e.to_string()))
                })
                .transpose()?;
            let state_pool =
                NativeGraphPool::new(domain.clone(), state_graphs.family(), state_charge, &arena)
                    .map_err(|e| PipelineRefusal::Preparation(e.to_string()))?;
            let graphs = prepared.bind(&device, &weights)?;
            let handoff = if assignment.owns_entry() {
                None
            } else {
                Some(std::cell::RefCell::new(ActivationBuffer::allocate(
                    &device,
                    activation_contract,
                )?))
            };
            Ok((
                weights,
                bindings,
                pool,
                readout_pool,
                graphs,
                handoff,
                state_pool,
                arena,
            ))
        })();
        match result {
            Ok((_weights, bindings, pool, readout_pool, graphs, handoff, state_pool, arena)) => {
                Ok(Self {
                    executable: ExecutableStage {
                        assignment,
                        device,
                        programs,
                        residency,
                        state_graphs,
                        store: Rc::clone(&bindings),
                        state_plan,
                        domain,
                        graphs,
                        handoff,
                    },
                    allocation: StageAllocation {
                        execution: Rc::new(execution),
                        arena,
                        memory: StageStartupHeap {
                            heap: Some(heap),
                            claim: Some(claim),
                        },
                        bindings,
                        pool,
                        readout_pool,
                        state_pool,
                    },
                })
            }
            Err(error) => {
                heap.release(claim);
                Err(error)
            }
        }
    }
    pub(crate) fn into_parts(self) -> (ExecutableStage, StageAllocation) {
        (self.executable, self.allocation)
    }
}
impl ExecutableStage {
    pub fn assignment(&self) -> &StageAssignment {
        &self.assignment
    }
    pub fn device(&self) -> &Rc<Device> {
        &self.device
    }
    pub fn state(&self) -> &Rc<StateStore> {
        &self.store
    }
    pub(crate) fn activation_transfer_bytes(&self) -> u64 {
        self.handoff
            .as_ref()
            .map_or(0, |buffer| buffer.borrow().storage_bytes())
    }
    pub(crate) fn constant_bytes(&self) -> Result<u64, PipelineRefusal> {
        self.graphs.constants.iter().try_fold(0u64, |n, t| {
            n.checked_add(t.storage_bytes()).ok_or_else(|| {
                PipelineRefusal::Preparation("stage constant charge overflows".into())
            })
        })
    }
}
impl StageAllocation {
    /// The admitted local plan; reading it transfers no allocation authority.
    pub fn resource_plan(&self) -> &crate::ResourcePlan {
        self.execution.resources()
    }

    /// Adopt the suffix's existing resources without allocating a second arena,
    /// recreating its store, reopening its device, or importing another weight.
    pub(crate) fn into_suffix_domain(
        self,
    ) -> Result<
        (
            Rc<crate::ExecutionPlan>,
            DeviceHeap,
            StoreBindings,
            crate::AllocatedResources,
        ),
        PipelineRefusal,
    > {
        let Self {
            execution,
            arena,
            memory,
            bindings,
            pool,
            readout_pool,
            state_pool,
        } = self;
        let readout = readout_pool.ok_or(PipelineRefusal::UnsupportedProfile)?;
        let resources = crate::AllocatedResources::adopt_stage_pools(
            memory.heap().device(),
            execution.resources(),
            arena,
            pool,
            readout,
            state_pool,
        )
        .map_err(PipelineRefusal::Preparation)?;
        if !bindings.belongs_to_device(memory.heap().device()) {
            return Err(PipelineRefusal::ForeignDevice);
        }
        Ok((execution, memory.finish(), bindings, resources))
    }
}
/// Preparation retains its fitting claim until the executor adopts the local
/// resources. Keeping RAII on this member allows the stage's other owners to
/// move, without copying allocations, into the normal domain lifecycle.
pub(crate) struct StageStartupHeap {
    heap: Option<DeviceHeap>,
    claim: Option<ClaimId>,
}
impl StageStartupHeap {
    /// Release only the transient startup claim. Physical storage remains held
    /// by its existing owners, and the exact heap moves into the loaded domain.
    pub(crate) fn finish(mut self) -> DeviceHeap {
        if let Some(claim) = self.claim.take() {
            self.heap.as_mut().expect("stage heap owned").release(claim);
        }
        self.heap.take().expect("stage heap consumed once")
    }
    pub(crate) fn heap(&self) -> &DeviceHeap {
        self.heap.as_ref().expect("stage heap owned")
    }
}
impl Drop for StageStartupHeap {
    fn drop(&mut self) {
        if let Some(claim) = self.claim.take() {
            self.heap.as_mut().expect("stage heap owned").release(claim);
        }
    }
}

/// Serial local graph families reuse one arena; outputs and uploads are not
/// aliases and their individual holdings must remain charged.
#[cfg(test)]
fn local_graph_charge(
    blocks: crate::NativeGraphCharge,
    readout: Option<crate::NativeGraphCharge>,
    constants: u64,
) -> Result<(u64, u64), PipelineRefusal> {
    let workspace = blocks
        .workspace_bytes
        .max(readout.map_or(0, |c| c.workspace_bytes));
    let total = workspace
        .checked_add(blocks.committed_bytes)
        .and_then(|n| n.checked_add(readout.map_or(0, |c| c.committed_bytes)))
        .and_then(|n| n.checked_add(constants))
        .ok_or_else(|| PipelineRefusal::Preparation("stage graph charge overflows".into()))?;
    Ok((workspace, total))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn arena_identity_refuses_equal_capacity_and_foreign_opened_devices() {
        let catalog = seismic::DeviceCatalog::discover().unwrap();
        let device = catalog.open_backend(BackendName::Cpu).unwrap();
        let a = device.execution_arena(4096).unwrap();
        let b = device.execution_arena(4096).unwrap();
        assert_eq!(a.bytes(), b.bytes());
        assert!(a.same_arena(&a.clone()));
        assert!(!a.same_arena(&b));
        let foreign = seismic::DeviceCatalog::discover()
            .unwrap()
            .open_backend(BackendName::Cpu)
            .unwrap();
        assert_eq!(device.info().selector, foreign.info().selector);
        assert!(!a.same_arena(&foreign.execution_arena(4096).unwrap()));
        let zero = device.execution_arena(0).unwrap();
        assert!(zero.same_arena(&zero.clone()));
        assert!(!zero.same_arena(&foreign.execution_arena(0).unwrap()));
        assert!(!zero.same_arena(&a));
    }
    #[test]
    fn startup_adoption_moves_exact_heap_and_releases_only_its_claim() {
        let catalog = seismic::DeviceCatalog::discover().unwrap();
        let device = Rc::new(catalog.open_backend(BackendName::Cpu).unwrap());
        let mut heap = DeviceHeap::open(
            catalog,
            crate::platform::MemoryReserves::standard(),
            device.clone(),
        )
        .unwrap();
        let claim = heap.claim(4096, 0, HoldingClass::InFlight).unwrap();
        let storage = seismic::Tensor::zeros(&device, seismic::Element::f32(), &[1024]).unwrap();
        let charged = device.memory_usage().charged;
        let startup = StageStartupHeap {
            heap: Some(heap),
            claim: Some(claim),
        };
        assert_eq!(startup.heap().heap().claims().count(), 1);
        let mut adopted = startup.finish();
        assert!(adopted.device().same_device(&device));
        assert_eq!(adopted.heap().claims().count(), 0);
        assert_eq!(device.memory_usage().charged, charged);
        assert!(storage.belongs_to(adopted.device()));
        adopted.refresh().unwrap();
        adopted
            .heap_mut()
            .insert(charged, HoldingClass::Model)
            .unwrap();
        adopted.refresh().unwrap();
        let growth = adopted.claim(4096, 0, HoldingClass::InFlight).unwrap();
        adopted.release(growth);
        assert_eq!(device.memory_usage().charged, charged);
    }
    #[test]
    fn endpoint_pools_share_workspace_but_not_output_or_upload_charge() {
        let charge = |workspace, output, upload, slots| {
            crate::NativeGraphCharge::from_checked(
                seismic::NativeGraphStorageBytes {
                    workspace,
                    output,
                    upload,
                },
                1,
                crate::GraphSlots {
                    activations: 1,
                    output: slots,
                },
            )
            .unwrap()
        };
        let blocks = charge(100, 20, 7, 2);
        let readout = charge(150, 30, 11, 1);
        assert_eq!(local_graph_charge(blocks, None, 13).unwrap(), (100, 160));
        assert_eq!(
            local_graph_charge(blocks, Some(readout), 13).unwrap(),
            (150, 251)
        );
        assert_eq!(
            local_graph_charge(readout, Some(blocks), 13).unwrap(),
            (150, 251)
        );
        assert!(local_graph_charge(blocks, Some(readout), u64::MAX).is_err());
    }
}
