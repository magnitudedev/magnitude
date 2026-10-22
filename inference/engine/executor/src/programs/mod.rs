//! Typed callable numerical programs and their submission lifecycle.

pub(crate) mod graph;
pub(crate) mod invocation_demand;
pub(crate) mod native_constants;
pub(crate) mod native_draft;
pub(crate) mod native_drafter;
pub(crate) mod native_head;
pub(crate) mod native_import;
pub(crate) mod native_shared_head;
pub(crate) mod native_state;
pub(crate) mod native_target;
pub(crate) mod native_target_graph;
pub(crate) mod native_vision;
mod submission;

pub use graph::GraphError;
pub use graph::readout::PreparedTargetReadoutGraphs;
pub use native_draft::PreparedDraftGraphs;
pub use native_drafter::PreparedDrafterGraphs;
pub use native_head::PreparedHeadGraphs;
pub use native_state::PreparedStateCopyGraphs;
pub use native_target::{CommitSpan, StoredRead, TargetOutput, TargetReadoutGraphResult};
pub use native_target_graph::{
    PlannedSealedGraphs, PreparedTargetGraphs, SealReport, SealedGraphsHeap,
};
pub use native_vision::PreparedVisionGraphs;
pub use submission::{
    CompletedWork, DeviceSubmission, ProgramSubmission, ReadySubmission, SubmittedHead,
    SubmittedTarget,
};

use crate::{
    GraphOutputTensor, HeadLaunchCore, ImportLaunchCore, InvariantError,
    ResidentWeightSlot, StateLaunchCore, SubmitError, TargetLaunchCore, ValidatedHeadLaunch,
    ValidatedImportLaunch, ValidatedStateLaunch, ValidatedTargetLaunch, ValidatedVisionLaunch,
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

pub type CompletedTargetWork = CompletedWork<TargetLaunchCore, TargetOutput>;
pub type CompletedHeadWork = CompletedWork<HeadLaunchCore, HeadOutput>;

/// What a head step leaves the host.
pub struct HeadOutput {
    /// The step's selections, `[steps, slot class, 2]` (token, status) rows
    /// in step-major order; absent for a causal-only head.
    pub selections: Option<GraphOutputTensor>,
    /// The step's landed reads of table rows its device work selected
    /// (`DeviceTable::read_selected`), completed once the step completed.
    pub landed_reads: Vec<crate::device_table::SubmittedRead>,
}

impl HeadOutput {
    /// The step completed: its landed reads bring the rows they landed into
    /// slots of their own (`SubmittedRead::complete`). Fails when one could
    /// not land its rows: the step read other rows.
    pub fn complete_landed_reads(&mut self) -> Result<(), SubmitError> {
        std::mem::take(&mut self.landed_reads)
            .into_iter()
            .try_for_each(crate::device_table::SubmittedRead::complete)
            .map_err(|detail| {
                SubmitError::Invariant(InvariantError {
                    context: "landed table rows",
                    detail,
                })
            })
    }
}
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
