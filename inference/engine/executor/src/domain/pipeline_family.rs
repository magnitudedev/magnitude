//! The normal concrete ProgramFamily adapter for the already checked executor.
//! Allocation and request admission stay with the existing domain/Owner.
use super::ProgramFamily;
use crate::{
    pipeline::{TwoStageCudaPipeline, ValidatedPipelineLaunch},
    programs::{ReadySubmission, StateProgram},
    programs::native_state::NativeStateProgram,
    HeadLaunchCore, ResidentHead, ResidentVision, StateLaunchCore, SubmitError, TargetLaunchCore,
    TargetLaunchWorkspace, TargetOutput, ValidatedHeadLaunch, ValidatedStateLaunch,
    ValidatedTargetLaunch, ValidatedVisionLaunch, VisionLaunchCore,
};
use magnitude_family_contracts::ModelDefinition;
use magnitude_state::{OwnedStateAdvance, TentativeAdvance};

pub struct PipelineNativeFamily {
    executor: TwoStageCudaPipeline,
    state: [NativeStateProgram; 2],
}
impl PipelineNativeFamily {
    pub fn new(executor: TwoStageCudaPipeline) -> Self {
        let state = executor
            .stages
            .each_ref()
            .map(|stage| NativeStateProgram::new(stage.state_graphs.clone()));
        Self { executor, state }
    }
    pub fn executor(&self) -> &TwoStageCudaPipeline {
        &self.executor
    }

    /// Move both prepared allocation authorities into the normal domain. This
    /// preserves the unique binding rights and both local allocation domains.
    pub fn into_domain(
        self,
        allocations: [crate::pipeline::StageAllocation; 2],
    ) -> Result<
        (super::ExecutorDomain<Self>, super::StateBindings<Self>),
        crate::pipeline::PipelineRefusal,
    > {
        use crate::pipeline::PipelineRefusal;
        let [prefix, suffix] = allocations;
        super::pipeline_resources::validate_prefix_allocation(&self.executor.stages[0], &prefix)?;
        let stage = &self.executor.stages[1];
        if !suffix.memory.heap().device().same_device(&stage.device)
            || suffix.pool.domain() != &stage.domain
        {
            return Err(PipelineRefusal::ForeignDevice);
        }
        super::pipeline_resources::validate_binding_owner(
            &stage.device,
            &stage.store,
            &suffix.bindings,
        )?;
        let definition = stage.assignment.original_model().clone();
        let (prefix_owner, prefix_bindings) =
            super::pipeline_resources::PipelinePrefixOwner::adopt(
                &self.executor.stages[0],
                prefix,
            )?;
        let (execution, memory, bindings, resources) = suffix.into_suffix_domain()?;
        let (mut domain, mut bindings) = super::ExecutorDomain::with_family(
            execution, definition, resources, memory, bindings, None, None, None, self,
        );
        domain.pipeline_owner = Some(prefix_owner);
        bindings.pipeline_prefix = Some(prefix_bindings);
        domain
            .register_allocated_holdings()
            .map_err(PipelineRefusal::Preparation)?;
        Ok((domain, bindings))
    }
}
fn unsupported(lane: &str) -> SubmitError {
    SubmitError::Invariant(crate::InvariantError {
        context: "pipeline program family",
        detail: format!("explicit plain pipeline does not support {lane}"),
    })
}
impl ProgramFamily for PipelineNativeFamily {
    type TargetSubmission =
        ReadySubmission<TargetLaunchCore, [TargetLaunchWorkspace; 2], TargetOutput>;
    type HeadSubmission = ReadySubmission<HeadLaunchCore, (), Option<crate::GraphOutputTensor>>;
    type VisionSubmission = ReadySubmission<VisionLaunchCore, (), crate::GraphOutputTensor>;
    type StateSubmission = <NativeStateProgram as StateProgram>::Submission;
    fn bind_head(&mut self, _: ResidentHead, _: &ModelDefinition) -> Result<(), String> {
        Err("explicit pipeline excludes heads".into())
    }
    fn bind_vision(&mut self, _: ResidentVision, _: &ModelDefinition) -> Result<(), String> {
        Err("explicit pipeline excludes vision".into())
    }
    fn head_is_bound(&self) -> bool {
        false
    }
    fn vision_is_bound(&self) -> bool {
        false
    }
    fn state_is_bound(&self) -> bool {
        true
    }
    fn unbind_optional(&mut self) {}
    // The ordinary domain owns suffix accounting; prefix remains independently
    // classified in its own heap, never counted as suffix storage.
    fn target_weight_bytes(&self) -> Result<u64, &'static str> {
        self.executor.stages[1].residency.resident_bytes()
    }
    fn prepared_program_bytes(&self) -> Result<u64, &'static str> {
        self.executor.stages[1].programs.device_storage_bytes()
    }
    fn target_constant_bytes(&self) -> Result<u64, &'static str> {
        self.executor.stages[1]
            .constant_bytes()
            .map_err(|_| "stage constant charge overflow")
    }
    fn activation_transfer_bytes(&self) -> u64 {
        self.executor.stages[1].activation_transfer_bytes()
    }
    fn submit_pipeline_target(
        &mut self,
        launch: ValidatedPipelineLaunch,
    ) -> Result<(Self::TargetSubmission, OwnedStateAdvance), SubmitError> {
        let complete = self.executor.execute(launch)?;
        let ([prefix, suffix], workspace, output) = complete.into_parts();
        let (_, mut advances, _, _) = prefix.into_parts();
        let Some(TentativeAdvance::Accepted(prefix)) = advances.pop() else {
            return Err(unsupported("nonordinary prefix reconciliation"));
        };
        if !advances.is_empty() {
            return Err(unsupported("multiple prefix requests"));
        }
        // Both device stages are complete. Keep both sets of leases until the
        // ordinary submission finishes; return prefix state for the same target
        // flight to carry into joint reconciliation. No state is accepted here.
        Ok((ReadySubmission::new(suffix, workspace, output), prefix))
    }
    fn submit_target(
        &mut self,
        launch: ValidatedTargetLaunch,
    ) -> Result<Self::TargetSubmission, (SubmitError, ValidatedTargetLaunch)> {
        Err((unsupported("single-stage target submission"), launch))
    }
    fn submit_head(
        &mut self,
        launch: ValidatedHeadLaunch,
    ) -> Result<Self::HeadSubmission, (SubmitError, ValidatedHeadLaunch)> {
        Err((unsupported("head execution"), launch))
    }
    fn submit_vision(
        &mut self,
        launch: ValidatedVisionLaunch,
    ) -> Result<Self::VisionSubmission, (SubmitError, ValidatedVisionLaunch)> {
        Err((unsupported("vision execution"), launch))
    }
    fn submit_state(
        &mut self,
        launch: ValidatedStateLaunch,
    ) -> Result<Self::StateSubmission, (SubmitError, ValidatedStateLaunch)> {
        let Some(index) = self
            .executor
            .stages
            .iter()
            .position(|stage| launch.domain() == &stage.domain)
        else {
            return Err((unsupported("foreign state resource domain"), launch));
        };
        self.state[index].submit(launch)
    }
}
