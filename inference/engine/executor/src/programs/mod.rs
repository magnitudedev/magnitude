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
    DeviceError, GraphOutputTensor, HeadLaunchCore, ImportLaunchCore, InvariantError,
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
    /// The step's read of table rows its device work selected, which the
    /// host observes once the step completed.
    pub selected_rows: Option<SelectedRows>,
}

/// A head step's read of table rows its device work selected
/// (`ResidentTable::read_selected`), and where the host finds those rows.
pub struct SelectedRows {
    read: crate::device_table::SelectedRead,
    rows: SelectedRowsSource,
}

enum SelectedRowsSource {
    /// The step's selected tokens: a chained pass embeds each one.
    Selections,
    /// `[.., 2]` i32 rows the step exported, the first column a token (a
    /// corrector's candidates).
    Exported(GraphOutputTensor),
}

impl SelectedRows {
    /// A read of the tokens the step selects.
    pub(crate) fn selections(read: crate::device_table::SelectedRead) -> Self {
        Self {
            read,
            rows: SelectedRowsSource::Selections,
        }
    }

    /// A read of the tokens of the step's exported `rows`.
    pub(crate) fn exported(
        read: crate::device_table::SelectedRead,
        rows: GraphOutputTensor,
    ) -> Self {
        Self {
            read,
            rows: SelectedRowsSource::Exported(rows),
        }
    }

    /// Observe the rows the read selected once the step completed
    /// (`SelectedRead::observe`): `selected`, the step's selected tokens, or
    /// the tokens of its exported rows.
    pub fn observe(&self, selected: &[crate::TokenId]) -> Result<(), SubmitError> {
        let invalid = |detail: String| {
            SubmitError::Invariant(InvariantError {
                context: "selected table rows",
                detail,
            })
        };
        let rows = match &self.rows {
            SelectedRowsSource::Selections => {
                selected.iter().map(|token| u64::from(token.0)).collect()
            }
            SelectedRowsSource::Exported(rows) => rows
                .tensor()
                .read_to_host()
                .map_err(|error| SubmitError::Device(DeviceError::Transfer(error.to_string())))?
                .chunks_exact(8)
                .map(|row| {
                    let token = i32::from_le_bytes(row[0..4].try_into().expect("four token bytes"));
                    u64::try_from(token)
                        .map_err(|_| invalid(format!("selected row {token} is negative")))
                })
                .collect::<Result<Vec<_>, _>>()?,
        };
        self.read.observe(&rows).map_err(invalid)
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
