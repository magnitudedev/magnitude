//! Sole join of prepared vision controls and their planned device resources.

use crate::{
    InvariantError, NativeGraphOutputLease, NativeGraphWorkspaceLease, PoolClass, ResourceDomainId,
};
use magnitude_model_batching::ValidatedVisionBatch;

pub struct VisionLaunchInputs {
    batch: ValidatedVisionBatch,
    workspace: NativeGraphWorkspaceLease,
    output: Option<NativeGraphOutputLease>,
}

impl VisionLaunchInputs {
    pub fn new(
        batch: ValidatedVisionBatch,
        workspace: NativeGraphWorkspaceLease,
        output: NativeGraphOutputLease,
    ) -> Self {
        Self {
            batch,
            workspace,
            output: Some(output),
        }
    }
}

pub struct ValidatedVisionLaunch {
    core: VisionLaunchCore,
    workspace: NativeGraphWorkspaceLease,
    output: Option<NativeGraphOutputLease>,
}

impl ValidatedVisionLaunch {
    pub fn new(
        inputs: VisionLaunchInputs,
        domain: &ResourceDomainId,
    ) -> Result<Self, (VisionLaunchInputs, InvariantError)> {
        if inputs.workspace.domain() != domain
            || inputs
                .output
                .as_ref()
                .is_none_or(|output| output.domain() != domain)
        {
            return Err((
                inputs,
                InvariantError {
                    context: "vision launch",
                    detail: "workspace or output lease differs from vision domain/class".into(),
                },
            ));
        }
        let VisionLaunchInputs {
            batch,
            workspace,
            output,
        } = inputs;
        Ok(Self {
            core: VisionLaunchCore {
                batch,
                domain: domain.clone(),
            },
            workspace,
            output,
        })
    }

    pub fn domain(&self) -> &ResourceDomainId {
        &self.core.domain
    }
    pub fn class(&self) -> PoolClass {
        PoolClass::Vision {
            patch_rows: self.core.batch.class_patch_rows(),
        }
    }
    pub(crate) fn submission_parts(
        &self,
    ) -> (
        &VisionLaunchCore,
        &NativeGraphWorkspaceLease,
        &Option<NativeGraphOutputLease>,
    ) {
        (&self.core, &self.workspace, &self.output)
    }
    pub(crate) fn execution_parts_mut(
        &mut self,
    ) -> (
        &VisionLaunchCore,
        &mut NativeGraphWorkspaceLease,
        &mut Option<NativeGraphOutputLease>,
    ) {
        (&self.core, &mut self.workspace, &mut self.output)
    }
    pub(crate) fn into_submission_parts(
        self,
    ) -> (
        VisionLaunchCore,
        NativeGraphWorkspaceLease,
        Option<NativeGraphOutputLease>,
    ) {
        (self.core, self.workspace, self.output)
    }
}

pub struct VisionLaunchCore {
    batch: ValidatedVisionBatch,
    domain: ResourceDomainId,
}

impl VisionLaunchCore {
    pub fn batch(&self) -> &ValidatedVisionBatch {
        &self.batch
    }
    pub fn into_batch(self) -> ValidatedVisionBatch {
        self.batch
    }
}
