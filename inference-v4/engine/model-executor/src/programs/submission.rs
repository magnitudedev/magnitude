use crate::{
    completion::{Completed, Completion},
    error::DeviceError,
};

/// A program owns its launch, state reconciliation payload, workspace, and
/// outputs until completion is observed and physical work is finished.
pub trait ProgramSubmission {
    type CompletedWork;

    fn completion(&mut self) -> &mut dyn Completion;
    fn finish(self) -> Result<Self::CompletedWork, DeviceError>;
}

/// Physical output plus the launch's owned reconciliation payload. The launch
/// remains unavailable to callers until `finish` consumes its submission.
pub struct CompletedWork<L, O> {
    launch: L,
    output: O,
}

impl<L, O> CompletedWork<L, O> {
    pub fn new(launch: L, output: O) -> Self {
        Self { launch, output }
    }

    pub fn into_parts(self) -> (L, O) {
        (self.launch, self.output)
    }
}

/// The native direct path has already completed when it returns this value.
/// A planned program may instead provide a pending implementation of the same
/// trait without changing its caller. Both retain the launch and leases.
pub struct ReadySubmission<L, S, O> {
    completion: Completed,
    launch: L,
    workspace: S,
    output: O,
}

impl<L, S, O> ReadySubmission<L, S, O> {
    pub fn new(launch: L, workspace: S, output: O) -> Self {
        Self {
            completion: Completed::success(),
            launch,
            workspace,
            output,
        }
    }
}

impl<L, S, O> ProgramSubmission for ReadySubmission<L, S, O> {
    type CompletedWork = CompletedWork<L, O>;

    fn completion(&mut self) -> &mut dyn Completion {
        &mut self.completion
    }

    fn finish(mut self) -> Result<Self::CompletedWork, DeviceError> {
        self.completion.result()?;
        drop(self.workspace);
        Ok(CompletedWork {
            launch: self.launch,
            output: self.output,
        })
    }
}
