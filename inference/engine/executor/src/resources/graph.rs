//! Bounded Seismic graph storage owned by the executor resource plan.

use crate::{CapacityError, InvariantError, ResourceDomainId, ResourceKind};
use seismic::{
    NativeExecutionArena, NativeGraphFamily, NativeGraphFamilyOutputSlot, NativeGraphFamilySlot,
    NativeGraphOutputs, NativeGraphPlan, Tensor, WorkflowTensor,
};
use std::{cell::RefCell, rc::Rc};

/// One family's graph storage. Activations cover the launches of the family
/// in flight at once; each holds its upload regions and binds the device's
/// one workspace arena. They are fixed at startup. Output slots outlive their
/// launches (a request retains its last published features), so their count
/// follows the live requests: the pool starts with what one request needs
/// and grows one slot at a time under a heap claim, and slots idle beyond
/// that start are released as surplus.
pub struct NativeGraphPool {
    domain: ResourceDomainId,
    family: NativeGraphFamily,
    /// Upload regions of one activation.
    activation_bytes: u64,
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
            activation_bytes,
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

/// Vision storage, one family per class: an encode's workspace and its
/// image's output slot are of its image's class, so a small image never holds
/// the largest class's storage. Vision holds nothing until an image arrives;
/// an encode claims an activation and an output slot of its class under the
/// heap, the image holds its slot until its features are released, and free
/// slots are released when idle.
pub struct VisionGraphPool {
    domain: ResourceDomainId,
    upload_regions: usize,
    classes: Vec<VisionGraphClass>,
}

struct VisionGraphClass {
    patch_rows: u64,
    family: NativeGraphFamily,
    activations: usize,
    outputs: usize,
    workspace: Rc<RefCell<Vec<NativeGraphFamilySlot>>>,
    output: Rc<RefCell<Vec<NativeGraphFamilyOutputSlot>>>,
}

impl VisionGraphPool {
    pub(crate) fn new<'a>(
        domain: ResourceDomainId,
        upload_regions: usize,
        classes: impl IntoIterator<Item = (u64, &'a NativeGraphFamily)>,
    ) -> Self {
        Self {
            domain,
            upload_regions,
            classes: classes
                .into_iter()
                .map(|(patch_rows, family)| VisionGraphClass {
                    patch_rows,
                    family: family.clone(),
                    activations: 0,
                    outputs: 0,
                    workspace: Rc::new(RefCell::new(Vec::new())),
                    output: Rc::new(RefCell::new(Vec::new())),
                })
                .collect(),
        }
    }

    fn class(&self, patch_rows: u64) -> Result<&VisionGraphClass, CapacityError> {
        self.classes
            .iter()
            .find(|class| class.patch_rows == patch_rows)
            .ok_or(CapacityError {
                resource: ResourceKind::Workspace,
                required: 1,
                available: 0,
            })
    }

    fn class_mut(&mut self, patch_rows: u64) -> Result<&mut VisionGraphClass, CapacityError> {
        self.classes
            .iter_mut()
            .find(|class| class.patch_rows == patch_rows)
            .ok_or(CapacityError {
                resource: ResourceKind::Workspace,
                required: 1,
                available: 0,
            })
    }

    /// The charge of one activation of the class of `patch_rows`: its own
    /// workspace and its upload regions.
    pub fn activation_bytes(&self, patch_rows: u64) -> Result<u64, CapacityError> {
        let family = &self.class(patch_rows)?.family;
        Ok(family.workspace_bytes() + family.upload_bytes() * self.upload_regions as u64)
    }

    /// The charge of one output slot of the class of `patch_rows`.
    pub fn output_bytes(&self, patch_rows: u64) -> Result<u64, CapacityError> {
        Ok(self.class(patch_rows)?.family.output_bytes())
    }

    pub fn available_workspace(&self, patch_rows: u64) -> usize {
        self.class(patch_rows)
            .map_or(0, |class| class.workspace.borrow().len())
    }

    pub fn available_output(&self, patch_rows: u64) -> usize {
        self.class(patch_rows)
            .map_or(0, |class| class.output.borrow().len())
    }

    /// Allocate one more activation of the class of `patch_rows`. The caller
    /// holds a heap claim for [`Self::activation_bytes`] across this call.
    pub(crate) fn grow_activation(&mut self, patch_rows: u64) -> Result<(), VisionGrowth> {
        let regions = self.upload_regions;
        let class = self.class_mut(patch_rows).map_err(VisionGrowth::Class)?;
        let slot = class
            .family
            .new_slot(regions)
            .map_err(VisionGrowth::Device)?;
        class.workspace.borrow_mut().push(slot);
        class.activations += 1;
        Ok(())
    }

    /// Allocate one more output slot of the class of `patch_rows`. The
    /// caller holds a heap claim for [`Self::output_bytes`] across this call.
    pub(crate) fn grow_output(&mut self, patch_rows: u64) -> Result<(), VisionGrowth> {
        let class = self.class_mut(patch_rows).map_err(VisionGrowth::Class)?;
        let slot = class
            .family
            .new_output_slot()
            .map_err(VisionGrowth::Device)?;
        class.output.borrow_mut().push(slot);
        class.outputs += 1;
        Ok(())
    }

    pub fn acquire_workspace(
        &self,
        patch_rows: u64,
    ) -> Result<NativeGraphWorkspaceLease, CapacityError> {
        let class = self.class(patch_rows)?;
        let slot = class.workspace.borrow_mut().pop().ok_or(CapacityError {
            resource: ResourceKind::Workspace,
            required: 1,
            available: 0,
        })?;
        Ok(NativeGraphWorkspaceLease {
            domain: self.domain.clone(),
            slot: Some(slot),
            free: class.workspace.clone(),
        })
    }

    pub fn acquire_output(&self, patch_rows: u64) -> Result<NativeGraphOutputLease, CapacityError> {
        let class = self.class(patch_rows)?;
        let slot = class.output.borrow_mut().pop().ok_or(CapacityError {
            resource: ResourceKind::Output,
            required: 1,
            available: 0,
        })?;
        Ok(NativeGraphOutputLease {
            domain: self.domain.clone(),
            slot: Some(slot),
            free: class.output.clone(),
        })
    }

    /// Release every free activation and output slot; returns how many were
    /// released. A lent activation stays until its encode completes, a lent
    /// output slot until its image's features are released.
    pub(crate) fn release_idle(&mut self) -> usize {
        self.classes
            .iter_mut()
            .map(|class| {
                let activations = class.workspace.borrow_mut().drain(..).count();
                let outputs = class.output.borrow_mut().drain(..).count();
                class.activations -= activations;
                class.outputs -= outputs;
                activations + outputs
            })
            .sum()
    }

    /// Storage allocated now, lent or free.
    pub fn committed_bytes(&self) -> u64 {
        self.classes
            .iter()
            .map(|class| {
                let family = &class.family;
                let activation =
                    family.workspace_bytes() + family.upload_bytes() * self.upload_regions as u64;
                activation * class.activations as u64 + family.output_bytes() * class.outputs as u64
            })
            .sum()
    }
}

/// A failed vision slot growth: a class the load did not prepare, or the
/// device's refusal.
pub(crate) enum VisionGrowth {
    Class(CapacityError),
    Device(seismic::TensorError),
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
