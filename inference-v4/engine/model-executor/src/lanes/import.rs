//! One-shot planned resident import with an owned destination tensor.

use crate::{
    ImportArtifactTensor, ImportWorkspaceLease, InvariantError, PoolClass, ResourceDomainId,
    WeightPlan, WeightStorageIdentity,
};
use seismic::{Device, Tensor};

/// A persistent destination allocated once from an admitted weight plan.
/// Dropping it before residency publication releases the uncommitted memory.
pub struct ResidentWeightSlot {
    identity: WeightStorageIdentity,
    tensor: Tensor,
}

impl ResidentWeightSlot {
    pub fn new(plan: &WeightPlan, device: &Device, tensor: Tensor) -> Result<Self, InvariantError> {
        if !tensor.belongs_to(device)
            || tensor.element() != plan.resident
            || tensor.extents() != plan.shape
            || tensor.byte_len() != plan.resident_bytes
        {
            return Err(InvariantError {
                context: "resident weight slot",
                detail: format!(
                    "{} ({:?}): planned element {:?}, shape {:?}, bytes {}; actual element {:?}, shape {:?}, bytes {}, same device {}",
                    plan.descriptor.name,
                    plan.role,
                    plan.resident,
                    plan.shape,
                    plan.resident_bytes,
                    tensor.element(),
                    tensor.extents(),
                    tensor.byte_len(),
                    tensor.belongs_to(device),
                ),
            });
        }
        Ok(Self {
            identity: plan.storage_identity(),
            tensor,
        })
    }

    pub fn identity(&self) -> &WeightStorageIdentity {
        &self.identity
    }
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }
    pub fn tensor_mut(&mut self) -> &mut Tensor {
        &mut self.tensor
    }
    pub fn into_tensor(self) -> Tensor {
        self.tensor
    }
}

pub struct ImportLaunchInputs {
    plan: WeightPlan,
    source: ImportArtifactTensor,
    workspace: ImportWorkspaceLease,
    destination: ResidentWeightSlot,
}

impl ImportLaunchInputs {
    pub fn new(
        plan: WeightPlan,
        source: ImportArtifactTensor,
        workspace: ImportWorkspaceLease,
        destination: ResidentWeightSlot,
    ) -> Self {
        Self {
            plan,
            source,
            workspace,
            destination,
        }
    }
}

pub struct ValidatedImportLaunch {
    core: ImportLaunchCore,
    workspace: ImportWorkspaceLease,
    destination: ResidentWeightSlot,
}

impl ValidatedImportLaunch {
    pub fn new(
        inputs: ImportLaunchInputs,
        domain: &ResourceDomainId,
    ) -> Result<Self, (ImportLaunchInputs, InvariantError)> {
        let count = inputs
            .plan
            .shape
            .iter()
            .try_fold(1u64, |count, extent| count.checked_mul(*extent));
        let upload = inputs.workspace.upload();
        let upload_matches = upload.element() == inputs.plan.source
            && count.is_some_and(|count| upload.extents() == [count])
            && upload.byte_len() == inputs.plan.source_bytes
            && upload.storage_bytes() == inputs.workspace.bytes();
        let valid = inputs.source.artifact() == inputs.plan.component.identity
            && inputs.source.stored().shape() == inputs.plan.shape
            && inputs.source.stored().source_element() == Some(inputs.plan.source)
            && inputs.source.stored().source_bytes() == inputs.plan.source_bytes
            && inputs.destination.identity() == &inputs.plan.storage_identity()
            && inputs.workspace.domain() == domain
            && inputs.workspace.class() == (PoolClass::Import {
                bytes: inputs.plan.source_bytes,
            })
            && inputs.workspace.bytes() == inputs.plan.source_bytes
            && upload_matches;
        if !valid {
            return Err((
                inputs,
                InvariantError {
                    context: "import launch",
                    detail: "source, destination, or upload lease differs from admitted weight"
                        .into(),
                },
            ));
        }
        let ImportLaunchInputs {
            plan,
            source,
            workspace,
            destination,
        } = inputs;
        Ok(Self {
            core: ImportLaunchCore {
                plan,
                source,
                domain: domain.clone(),
            },
            workspace,
            destination,
        })
    }

    pub fn domain(&self) -> &ResourceDomainId {
        &self.core.domain
    }
    pub fn class(&self) -> PoolClass {
        PoolClass::Import {
            bytes: self.core.plan.source_bytes,
        }
    }
    pub(crate) fn submission_parts_mut(
        &mut self,
    ) -> (
        &ImportLaunchCore,
        &mut ImportWorkspaceLease,
        &mut ResidentWeightSlot,
    ) {
        (&self.core, &mut self.workspace, &mut self.destination)
    }
    pub(crate) fn into_submission_parts(
        self,
    ) -> (ImportLaunchCore, ImportWorkspaceLease, ResidentWeightSlot) {
        (self.core, self.workspace, self.destination)
    }
}

pub struct ImportLaunchCore {
    plan: WeightPlan,
    source: ImportArtifactTensor,
    domain: ResourceDomainId,
}

impl ImportLaunchCore {
    pub fn plan(&self) -> &WeightPlan {
        &self.plan
    }
    pub fn source(&self) -> &ImportArtifactTensor {
        &self.source
    }
    pub fn into_parts(self) -> (WeightPlan, ImportArtifactTensor) {
        (self.plan, self.source)
    }
}
