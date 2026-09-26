//! Typed callable numerical programs and their submission lifecycle.

pub(crate) mod graph;
pub(crate) mod native_constants;
pub(crate) mod native_head;
pub(crate) mod native_import;
pub(crate) mod native_state;
pub(crate) mod native_target;
pub(crate) mod native_target_graph;
pub(crate) mod native_vision;
mod submission;

pub use graph::readout::PreparedTargetReadoutGraphs;
pub use native_head::PreparedHeadGraphs;
pub use native_state::PreparedStateCopyGraphs;
pub use native_target::{CommitSpan, TargetOutput, TargetReadoutGraphResult};
pub use native_target_graph::{PreparedTargetGraphs, SealReport};
pub use native_vision::PreparedVisionGraphs;
pub use submission::{
    CompletedWork, DeviceSubmission, ProgramSubmission, ReadySubmission, SubmittedTarget,
};

use crate::{
    GraphOutputTensor, HeadLaunchCore, ImportLaunchCore, ResidentWeightSlot, StateLaunchCore,
    SubmitError, TargetLaunchCore, ValidatedHeadLaunch, ValidatedImportLaunch,
    ValidatedStateLaunch, ValidatedTargetLaunch, ValidatedVisionLaunch, VisionLaunchCore,
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

pub type CompletedTargetWork = CompletedWork<TargetLaunchCore, TargetOutput>;
/// A head's selections, `[steps, slot class, 2]` (token, status) rows in
/// step-major order; absent for a causal-only head.
pub type CompletedHeadWork = CompletedWork<HeadLaunchCore, Option<GraphOutputTensor>>;
pub type CompletedVisionWork = CompletedWork<VisionLaunchCore, GraphOutputTensor>;
pub type CompletedStateWork = CompletedWork<StateLaunchCore, ()>;
pub type CompletedImportWork = CompletedWork<ImportLaunchCore, ResidentWeightSlot>;

/// One typed numerical lane. A submission may still be executing on the
/// device when it is returned; its completion reports when it is not.
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

    fn submit(
        &mut self,
        launch: ValidatedHeadLaunch,
    ) -> Result<Self::Submission, (SubmitError, ValidatedHeadLaunch)>;
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
