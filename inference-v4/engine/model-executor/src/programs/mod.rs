//! Typed callable numerical programs and their submission lifecycle.

pub(crate) mod native_head;
pub(crate) mod native_import;
pub(crate) mod native_state;
pub(crate) mod native_target;
pub(crate) mod native_target_graph;
pub(crate) mod native_target_readout_graph;
pub(crate) mod native_vision;
mod submission;

pub use native_head::PreparedHeadGraphs;
pub use native_state::PreparedStateCopyGraphs;
pub use native_target::TargetReadoutGraphResult;
pub use native_target_graph::PreparedTargetGraphs;
pub use native_target_readout_graph::PreparedTargetReadoutGraphs;
pub use native_vision::PreparedVisionGraphs;
pub use submission::{CompletedWork, ProgramSubmission, ReadySubmission};

use crate::{
    GraphOutputTensor, HeadLaunchCore, ImportLaunchCore, ProjectLaunchCore, ResidentWeightSlot,
    StateLaunchCore, SubmitError, TargetLaunchCore, ValidatedHeadLaunch, ValidatedImportLaunch,
    ValidatedProjectionLaunch, ValidatedStateLaunch, ValidatedTargetLaunch, ValidatedVisionLaunch,
    VisionLaunchCore,
};

/// Submit one ready native graph run and wait for its outcome. Programs
/// that chain several runs submit them without waiting instead and check
/// every completion once all are queued.
pub(crate) fn run_graph(
    ready: seismic::ReadyNativeGraphRun<'_>,
) -> Result<seismic::NativeGraphOutputs, seismic::CallError> {
    let (outputs, completion) = ready.submit()?;
    completion.wait()?;
    Ok(outputs)
}

pub type CompletedTargetWork = CompletedWork<TargetLaunchCore, Option<TargetReadoutGraphResult>>;
pub type CompletedHeadWork = CompletedWork<HeadLaunchCore, GraphOutputTensor>;
pub struct ProjectGraphOutput {
    pub(crate) logits: GraphOutputTensor,
    pub(crate) selected: GraphOutputTensor,
}
pub type CompletedProjectWork = CompletedWork<ProjectLaunchCore, ProjectGraphOutput>;
pub type CompletedVisionWork = CompletedWork<VisionLaunchCore, GraphOutputTensor>;
pub type CompletedStateWork = CompletedWork<StateLaunchCore, ()>;
pub type CompletedImportWork = CompletedWork<ImportLaunchCore, ResidentWeightSlot>;

/// One typed numerical lane. Native implementations return an already-ready
/// submission; compiler-planned implementations may remain pending.
pub trait TargetProgram {
    type Submission: ProgramSubmission<CompletedWork = CompletedTargetWork>;

    /// A rejected submission returns its still-owned launch so the domain can
    /// terminate the group without losing state ownership.
    fn submit(
        &mut self,
        launch: ValidatedTargetLaunch,
    ) -> Result<Self::Submission, (SubmitError, ValidatedTargetLaunch)>;
}

pub trait HeadProgram {
    type Submission: ProgramSubmission<CompletedWork = CompletedHeadWork>;
    type ProjectSubmission: ProgramSubmission<CompletedWork = CompletedProjectWork>;

    fn submit(
        &mut self,
        launch: ValidatedHeadLaunch,
    ) -> Result<Self::Submission, (SubmitError, ValidatedHeadLaunch)>;
    fn project(
        &mut self,
        launch: ValidatedProjectionLaunch,
    ) -> Result<Self::ProjectSubmission, (SubmitError, ValidatedProjectionLaunch)>;
}

pub trait VisionProgram {
    type Submission: ProgramSubmission<CompletedWork = CompletedVisionWork>;

    fn submit(
        &mut self,
        launch: ValidatedVisionLaunch,
    ) -> Result<Self::Submission, (SubmitError, ValidatedVisionLaunch)>;
}

pub trait StateProgram {
    type Submission: ProgramSubmission<CompletedWork = CompletedStateWork>;

    fn submit(
        &mut self,
        launch: ValidatedStateLaunch,
    ) -> Result<Self::Submission, (SubmitError, ValidatedStateLaunch)>;
}

pub trait ImportProgram {
    type Submission: ProgramSubmission<CompletedWork = CompletedImportWork>;

    fn submit(
        &mut self,
        launch: ValidatedImportLaunch,
    ) -> Result<Self::Submission, (SubmitError, ValidatedImportLaunch)>;
}
