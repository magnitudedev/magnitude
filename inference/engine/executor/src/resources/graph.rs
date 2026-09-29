//! Bounded Seismic graph storage owned by the executor resource plan.

use crate::{CapacityError, InvariantError, ResourceDomainId, ResourceKind};
use seismic::{
    NativeGraphFamily, NativeGraphFamilyOutputSlot, NativeGraphFamilySlot, NativeGraphOutputs,
    NativeGraphPlan, Tensor, WorkflowTensor,
};
use std::{cell::RefCell, rc::Rc};

/// One family's graph storage. Workspace slots cover the launches of the
/// family in flight at once and are fixed at startup. Output slots outlive
/// their launches (a request retains its last published features), so their
/// count follows the live requests: the pool starts with what one request
/// needs and grows one slot at a time under a heap claim, and slots idle
/// beyond that start are released as surplus.
pub struct NativeGraphPool {
    domain: ResourceDomainId,
    family: NativeGraphFamily,
    workspace_bytes: u64,
    output_bytes: u64,
    minimum_outputs: usize,
    /// Output slots allocated now, lent or free.
    outputs: usize,
    workspace: Rc<RefCell<Vec<NativeGraphFamilySlot>>>,
    output: Rc<RefCell<Vec<NativeGraphFamilyOutputSlot>>>,
}

impl NativeGraphPool {
    /// The startup slots of `charge`: its workspace slots, each with
    /// `upload_regions` upload regions, and its startup output slots.
    pub(crate) fn new(
        domain: ResourceDomainId,
        family: &NativeGraphFamily,
        charge: crate::NativeGraphCharge,
    ) -> Result<Self, super::AllocationError> {
        let workspace = (0..charge.workspace_slots)
            .map(|_| family.new_slot(charge.upload_regions))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| super::AllocationError::Device(error.to_string()))?;
        let output = (0..charge.output_slots)
            .map(|_| family.new_output_slot())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| super::AllocationError::Device(error.to_string()))?;
        let workspace_bytes = charge
            .committed_bytes
            .checked_sub(charge.output_bytes * charge.output_slots as u64)
            .ok_or_else(|| {
                super::AllocationError::Plan(crate::InvariantError {
                    context: "graph pool",
                    detail: "planned charge is below its output slots".into(),
                })
            })?;
        Ok(Self {
            domain,
            family: family.clone(),
            workspace_bytes,
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
        self.workspace_bytes + self.output_bytes * self.outputs as u64
    }

    /// The charge of one more output slot.
    pub fn output_slot_bytes(&self) -> u64 {
        self.output_bytes
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
