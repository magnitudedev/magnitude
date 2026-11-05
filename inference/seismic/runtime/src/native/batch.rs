//! Ordered tensor-result native calls encoded as one device submission.

use super::*;

/// A preparation-time batch of independent native calls on one device. Each
/// call keeps its own arguments and result allocations; the calls share one
/// scratch arena because a launch waits for the earlier launches over the
/// bytes it reaches.
pub(crate) struct NativeTensorBatch {
    device: Arc<DeviceInner>,
    calls: Vec<(Arc<NativePrepared>, NativeBoundCall)>,
    scratch: ScratchArena,
}

impl NativeTensorBatch {
    pub(crate) fn new(device: Arc<DeviceInner>) -> Self {
        Self {
            device,
            calls: Vec::new(),
            scratch: ScratchArena::default(),
        }
    }

    pub(crate) fn push(
        &mut self,
        kernel: &Arc<NativePrepared>,
        args: EncodedArgs,
        outputs: EncodedOutputs,
    ) -> Result<(), CallError> {
        if !Arc::ptr_eq(&self.device, &kernel.public_device) {
            return Err(CallError::Execution(ExecutionError::SubmissionFailed(
                "native batch call belongs to another device".into(),
            )));
        }
        if kernel
            .results
            .iter()
            .any(|result| !matches!(result, NativeResult::Tensor { .. }))
        {
            return Err(CallError::Workflow(
                crate::api::WorkflowError::HostBoundaryRequired,
            ));
        }
        let call = kernel.prepare_call(&mut self.scratch, args, Some(outputs), super::Parking::Unanswered)?;
        self.calls.push((kernel.clone(), call));
        Ok(())
    }

    pub(crate) fn submit(mut self) -> Result<NativeTensorBatchCompletion, CallError> {
        if self.calls.is_empty() {
            return Err(CallError::Workflow(crate::api::WorkflowError::Empty));
        }
        let access = merge_access(
            self.calls
                .iter()
                .flat_map(|(_, call)| call.access.iter().cloned()),
        );
        let retained: Arc<dyn std::any::Any + Send + Sync> = Arc::new(
            self.calls
                .iter()
                .map(|(kernel, _)| kernel.clone())
                .collect::<Vec<_>>(),
        );
        let submission = submit(&BatchCalls(&self.calls), &access, retained, 1, false, super::NativeQueue::Compute)?;
        // The fences are recorded: the calls' views, and their leases, end
        // here; a slab freed or a region shrunk from now on is released
        // physically once the batch completes (allocation fences).
        for (_, call) in &mut self.calls {
            call.release_views();
        }
        Ok(NativeTensorBatchCompletion {
            submission,
            _calls: self.calls,
            finished: false,
        })
    }
}

struct BatchCalls<'a>(&'a [(Arc<NativePrepared>, NativeBoundCall)]);

impl DispatchList for BatchCalls<'_> {
    fn count(&self) -> usize {
        self.0.len()
    }

    fn dispatch<'s>(
        &'s self,
        index: usize,
        buffers: &mut Vec<(&'s Allocation, u64)>,
    ) -> Dispatch<'s> {
        let (kernel, call) = &self.0[index];
        kernel.bind(
            call.buffers
                .iter()
                .map(|(allocation, offset, _)| (&**allocation, *offset)),
            buffers,
        );
        Dispatch {
            kernel,
            words: &call.words,
            launches: &call.launches.geometry,
            rounds: call.launches.rounds,
            representations: &call.representations,
        }
    }

    fn reaches<'s>(&'s self, index: usize, reaches: &mut Vec<hazard::Reach<'s>>) {
        self.0[index].1.reaches(reaches);
    }

    fn encodes(&self, index: usize, ordinal: usize) -> bool {
        let (kernel, call) = &self.0[index];
        call.encodes(kernel, ordinal)
    }

    fn runs(&self) -> Option<Vec<super::PlanRun>> {
        None
    }
}

/// Keeps every submitted call's source, result, scratch and implementation
/// alive until the device has finished its batch.
pub(crate) struct NativeTensorBatchCompletion {
    submission: NativeSubmission,
    _calls: Vec<(Arc<NativePrepared>, NativeBoundCall)>,
    finished: bool,
}

impl NativeTensorBatchCompletion {
    /// Whether the batch ended, without waiting: completed, or faulted, the
    /// outcome [`Self::wait`] reports.
    pub(crate) fn is_complete(&self) -> bool {
        self.submission.is_complete()
    }

    pub(crate) fn wait(mut self) -> Result<(), CallError> {
        let result = self.submission.wait();
        self.finished = true;
        result
    }
}

impl Drop for NativeTensorBatchCompletion {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.submission.wait();
        }
    }
}
