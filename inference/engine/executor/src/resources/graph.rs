//! Bounded Seismic graph storage owned by the executor resource plan.

use crate::{CapacityError, InvariantError, ResourceDomainId, ResourceKind};
use seismic::{
    NativeExecutionArena, NativeGraphFamily, NativeGraphFamilyOutputSlot, NativeGraphFamilySlot,
    NativeGraphOutputs, NativeGraphPlan, Tensor, WorkflowTensor,
};
use std::{cell::RefCell, rc::Rc};

/// One family's graph storage. Activations cover the launches of the family
/// in flight at once; each holds its upload regions and binds the device's
/// one workspace arena. They are fixed at startup, except for a family that
/// starts with none (vision), which claims one when a launch first needs it
/// and releases it at idle. Output slots outlive their
/// launches (a request retains its last published features), so their count
/// follows the live requests: the pool starts with what one request needs
/// and grows one slot at a time under a heap claim, and slots idle beyond
/// that start are released as surplus.
pub struct NativeGraphPool {
    domain: ResourceDomainId,
    family: NativeGraphFamily,
    arena: NativeExecutionArena,
    upload_regions: usize,
    /// Upload regions of one activation.
    activation_bytes: u64,
    /// Startup activations. A family with none (vision) claims its
    /// activation when a launch first needs it and releases it when idle.
    minimum_activations: usize,
    /// Activations allocated now, lent or free.
    activations: usize,
    output_bytes: u64,
    minimum_outputs: usize,
    /// Output slots allocated now, lent or free.
    outputs: usize,
    workspace: Rc<RefCell<Vec<NativeGraphFamilySlot>>>,
    output: Rc<RefCell<Vec<NativeGraphFamilyOutputSlot>>>,
}

impl NativeGraphPool {
    /// The startup slots of `charge`: its activations in `arena`, each with
    /// `upload_regions` upload regions, and its startup output slots.
    pub(crate) fn new(
        domain: ResourceDomainId,
        family: &NativeGraphFamily,
        charge: crate::NativeGraphCharge,
        arena: &NativeExecutionArena,
    ) -> Result<Self, super::AllocationError> {
        let workspace = (0..charge.activations)
            .map(|_| family.new_slot_in(arena, charge.upload_regions))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| super::AllocationError::Device(error.to_string()))?;
        let output = (0..charge.output_slots)
            .map(|_| family.new_output_slot())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| super::AllocationError::Device(error.to_string()))?;
        let activation_bytes = charge
            .upload_bytes
            .checked_mul(charge.upload_regions as u64)
            .ok_or_else(|| {
                super::AllocationError::Plan(crate::InvariantError {
                    context: "graph pool",
                    detail: "activation byte count overflows".into(),
                })
            })?;
        Ok(Self {
            domain,
            family: family.clone(),
            arena: arena.clone(),
            upload_regions: charge.upload_regions,
            activation_bytes,
            minimum_activations: charge.activations,
            activations: charge.activations,
            output_bytes: charge.output_bytes,
            minimum_outputs: charge.output_slots,
            outputs: charge.output_slots,
            workspace: Rc::new(RefCell::new(workspace)),
            output: Rc::new(RefCell::new(output)),
        })
    }

    pub fn domain(&self) -> &ResourceDomainId {
        &self.domain
    }

    /// Physical arenas owned by this pool, including slots currently lent
    /// to a submission or a published output view.
    pub fn committed_bytes(&self) -> u64 {
        self.activation_bytes * self.activations as u64 + self.output_bytes * self.outputs as u64
    }

    /// The charge of one more output slot.
    pub fn output_slot_bytes(&self) -> u64 {
        self.output_bytes
    }

    /// Whether the family claims its activation on demand: it holds none
    /// from startup, so a launch that finds none free grows one.
    pub fn activates_on_demand(&self) -> bool {
        self.minimum_activations == 0
    }

    /// The charge of one more activation: its upload regions. Its workspace
    /// is the shared arena, already charged.
    pub fn activation_slot_bytes(&self) -> u64 {
        self.activation_bytes
    }

    /// Allocate one more activation in the arena. The caller holds a heap
    /// claim for [`Self::activation_slot_bytes`] across this call.
    pub(crate) fn grow_activation(&mut self) -> Result<(), seismic::WorkflowError> {
        let slot = self.family.new_slot_in(&self.arena, self.upload_regions)?;
        self.workspace.borrow_mut().push(slot);
        self.activations += 1;
        Ok(())
    }

    /// Release free activations beyond the startup count; returns how many
    /// were released. A lent activation stays until its launch completes.
    pub(crate) fn release_idle_activations(&mut self) -> usize {
        let mut free = self.workspace.borrow_mut();
        let releasable = self
            .activations
            .saturating_sub(self.minimum_activations)
            .min(free.len());
        let kept = free.len() - releasable;
        free.truncate(kept);
        self.activations -= releasable;
        releasable
    }

    pub fn available_workspace(&self) -> usize {
        self.workspace.borrow().len()
    }

    pub fn available_output(&self) -> usize {
        self.output.borrow().len()
    }

    /// Allocate one more output slot. The caller holds a heap claim for
    /// [`Self::output_slot_bytes`] across this call.
    pub(crate) fn grow_output(&mut self) -> Result<(), seismic::TensorError> {
        let slot = self.family.new_output_slot()?;
        self.output.borrow_mut().push(slot);
        self.outputs += 1;
        Ok(())
    }

    /// Release free output slots beyond the startup count; returns how many
    /// were released. Lent slots stay until their last view drops.
    pub(crate) fn release_idle_outputs(&mut self) -> usize {
        let mut free = self.output.borrow_mut();
        let releasable = self
            .outputs
            .saturating_sub(self.minimum_outputs)
            .min(free.len());
        let kept = free.len() - releasable;
        free.truncate(kept);
        self.outputs -= releasable;
        releasable
    }

    pub fn acquire_workspace(&self) -> Result<NativeGraphWorkspaceLease, CapacityError> {
        let slot = self.workspace.borrow_mut().pop().ok_or(CapacityError {
            resource: ResourceKind::Workspace,
            required: 1,
            available: 0,
        })?;
        Ok(NativeGraphWorkspaceLease {
            domain: self.domain.clone(),
            slot: Some(slot),
            free: self.workspace.clone(),
        })
    }

    pub fn acquire_output(&self) -> Result<NativeGraphOutputLease, CapacityError> {
        let slot = self.output.borrow_mut().pop().ok_or(CapacityError {
            resource: ResourceKind::Output,
            required: 1,
            available: 0,
        })?;
        Ok(NativeGraphOutputLease {
            domain: self.domain.clone(),
            slot: Some(slot),
            free: self.output.clone(),
        })
    }
}

pub struct NativeGraphWorkspaceLease {
    domain: ResourceDomainId,
    slot: Option<NativeGraphFamilySlot>,
    free: Rc<RefCell<Vec<NativeGraphFamilySlot>>>,
}

impl NativeGraphWorkspaceLease {
    pub fn domain(&self) -> &ResourceDomainId {
        &self.domain
    }

    pub fn slot_mut(&mut self) -> &mut NativeGraphFamilySlot {
        self.slot.as_mut().expect("graph workspace lease is live")
    }
}

impl Drop for NativeGraphWorkspaceLease {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            self.free.borrow_mut().push(slot);
        }
    }
}

/// An output reservation owns one family arena. Activation transfers it into
/// a one-shot Seismic output value; successful work returns that value after
/// every exported tensor view has been released.
pub struct NativeGraphOutputLease {
    domain: ResourceDomainId,
    slot: Option<NativeGraphFamilyOutputSlot>,
    free: Rc<RefCell<Vec<NativeGraphFamilyOutputSlot>>>,
}

impl NativeGraphOutputLease {
    pub fn domain(&self) -> &ResourceDomainId {
        &self.domain
    }

    pub fn activate(
        &mut self,
        plan: &NativeGraphPlan,
    ) -> Result<NativeGraphOutputs, InvariantError> {
        let slot = self.slot.take().expect("graph output lease is live");
        slot.activate(plan).map_err(|error| InvariantError {
            context: "target graph output",
            detail: error.to_string(),
        })
    }

    pub fn recycle(&mut self, outputs: NativeGraphOutputs) -> Result<(), InvariantError> {
        if self.slot.is_some() {
            return Err(InvariantError {
                context: "target graph output",
                detail: "output arena was already recycled".into(),
            });
        }
        self.slot = Some(outputs.recycle().map_err(|error| InvariantError {
            context: "target graph output",
            detail: error.to_string(),
        })?);
        Ok(())
    }

    /// Publish a completed graph's outputs. Every exported tensor returned
    /// through this owner retains the physical output arena. Final drop
    /// recycles it into the same admitted pool.
    pub fn publish(self, outputs: NativeGraphOutputs) -> GraphOutputOwner {
        assert!(self.slot.is_none(), "graph output was not activated");
        GraphOutputOwner {
            claim: Rc::new(GraphOutputClaim {
                outputs: Some(outputs),
                free: self.free.clone(),
            }),
        }
    }
}

impl Drop for NativeGraphOutputLease {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            self.free.borrow_mut().push(slot);
        }
    }
}

struct GraphOutputClaim {
    outputs: Option<NativeGraphOutputs>,
    free: Rc<RefCell<Vec<NativeGraphFamilyOutputSlot>>>,
}

impl Drop for GraphOutputClaim {
    fn drop(&mut self) {
        let Some(outputs) = self.outputs.take() else {
            return;
        };
        let slot = outputs
            .recycle()
            .expect("a graph output view outlived its last ownership claim");
        self.free.borrow_mut().push(slot);
    }
}

#[derive(Clone)]
pub struct GraphOutputOwner {
    claim: Rc<GraphOutputClaim>,
}

impl GraphOutputOwner {
    pub fn tensor(&self, result: &WorkflowTensor) -> Option<GraphOutputTensor> {
        let tensor = self.claim.outputs.as_ref()?.exported(result)?;
        Some(GraphOutputTensor {
            tensor,
            claim: self.claim.clone(),
        })
    }
}

pub struct GraphOutputTensor {
    // Rust drops fields in declaration order. Release this tensor handle
    // before the final claim attempts to recycle the output arena.
    tensor: Tensor,
    claim: Rc<GraphOutputClaim>,
}

impl Clone for GraphOutputTensor {
    fn clone(&self) -> Self {
        Self {
            tensor: self.tensor.clone(),
            claim: self.claim.clone(),
        }
    }
}

impl GraphOutputTensor {
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    pub fn slice_leading(&self, start: u64, end: u64) -> Result<Self, seismic::TensorError> {
        Ok(Self {
            tensor: self.tensor.slice_leading(start, end)?,
            claim: self.claim.clone(),
        })
    }
}

pub type TargetGraphPool = NativeGraphPool;
pub type TargetGraphWorkspaceLease = NativeGraphWorkspaceLease;
pub type TargetGraphOutputLease = NativeGraphOutputLease;
