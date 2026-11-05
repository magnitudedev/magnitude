//! Physical completion of issued native work, independent of its encoding
//! and of the tensors it uses.

use crate::api::CallError;
use seismic_compiler::errors::ExecutionError;

/// Completion of submitted device work, as allocation fences observe it.
/// The work ends complete or faulted, the device running none of it any
/// more either way; a fault reaches its submitter and what reads what the
/// work wrote, while a release of what it used needs only its end.
pub(crate) trait DeviceCompletion: Send + Sync {
    /// Whether the work completed, without waiting: `Ok(false)` while it
    /// runs, its fault once it ended without completing.
    fn completed(&self) -> Result<bool, ExecutionError>;
    /// Block until the work ends: its fault where it did not complete.
    fn ended(&self) -> Result<(), ExecutionError>;
    /// Block until the work ends, to release what it used: a fault ends it
    /// as completion does, and reaches its submitter and whoever orders
    /// after it, so a release needs only the end.
    fn retire(&self) {
        match self.ended() {
            Ok(()) | Err(_) => {}
        }
    }
    /// The native work this is the completion of, which work of another
    /// native queue waits for on the device (`Allocation::queue_order`);
    /// none for other work.
    fn native(&self) -> Option<&RouteSubmission> {
        None
    }
}

/// Work one backend committed: what an encoder issues, whole or as the
/// prefix it committed before failing.
pub(crate) enum Committed {
    Cpu {
        outcome: Result<(), ExecutionError>,
        /// Execution interval on the [`crate::native::trace::host_seconds`] clock.
        interval: (f64, f64),
    },
    #[cfg(target_os = "macos")]
    Metal(seismic_metal::DirectSubmission),
    Cuda(seismic_cuda::direct::DirectSubmission),
    #[cfg(not(target_os = "macos"))]
    Vulkan(crate::native::vulkan::VulkanSubmission),
}

impl DeviceCompletion for Committed {
    fn completed(&self) -> Result<bool, ExecutionError> {
        match self {
            Self::Cpu { outcome, .. } => outcome.clone().map(|()| true),
            #[cfg(target_os = "macos")]
            Self::Metal(submission) => submission.completed(),
            Self::Cuda(submission) => submission.completed(),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(submission) => submission.completed(),
        }
    }
    fn ended(&self) -> Result<(), ExecutionError> {
        match self {
            Self::Cpu { outcome, .. } => outcome.clone(),
            #[cfg(target_os = "macos")]
            Self::Metal(submission) => submission.finish(),
            Self::Cuda(submission) => submission.finish(),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(submission) => submission.finish(),
        }
    }
}

impl Committed {
    /// Device execution time of the completed work.
    pub(crate) fn device_seconds(&self) -> Result<f64, ExecutionError> {
        match self {
            Self::Cpu { interval, .. } => Ok(interval.1 - interval.0),
            #[cfg(target_os = "macos")]
            Self::Metal(submission) => submission.device_seconds(),
            Self::Cuda(submission) => submission.device_seconds(),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(submission) => submission.device_seconds(),
        }
    }
}

/// Issuing stopped after a prefix was committed: the prefix runs on as the
/// submission's work, and its outcome reports the issue. Only this
/// exceptional outcome owns the allocation.
pub(crate) struct IssuedFailure {
    pub(crate) submitted: Committed,
    pub(crate) issue: CallError,
}

impl IssuedFailure {
    pub(crate) fn outcome(&self) -> CallError {
        match self.submitted.ended() {
            Ok(()) => self.issue.clone(),
            Err(fence) => CallError::Unfenced { issue: Box::new(self.issue.clone()), fence },
        }
    }
}

/// One unit of issued native work: committed whole, or a committed prefix
/// whose issue failed.
pub(crate) enum RouteSubmission {
    Committed(Committed),
    Failed(Box<IssuedFailure>),
}

impl RouteSubmission {
    pub(crate) fn committed(&self) -> &Committed {
        match self {
            Self::Committed(committed) => committed,
            Self::Failed(failure) => &failure.submitted,
        }
    }

    /// Wait for completion and report the work's outcome: a failed issue's
    /// error once its committed prefix has completed.
    pub(crate) fn finish(&self) -> Result<(), CallError> {
        match self {
            Self::Committed(committed) => committed.ended().map_err(CallError::Execution),
            Self::Failed(failure) => Err(failure.outcome()),
        }
    }
}

impl DeviceCompletion for RouteSubmission {
    fn completed(&self) -> Result<bool, ExecutionError> {
        self.committed().completed()
    }
    fn ended(&self) -> Result<(), ExecutionError> {
        self.committed().ended()
    }
    fn native(&self) -> Option<&RouteSubmission> {
        Some(self)
    }
}
