//! Ordered tensor-result native calls encoded as one device submission.

use super::*;

/// A preparation-time batch of independent native calls on one device. Each
/// call keeps its own arguments and result allocations; implementations may
/// share scratch because their launches execute in submission order.
pub(crate) struct NativeTensorBatch {
    device: Arc<DeviceInner>,
    calls: Vec<(Arc<NativePrepared>, NativeBoundCall)>,
}

impl NativeTensorBatch {
    pub(crate) fn new(device: Arc<DeviceInner>) -> Self {
        Self {
            device,
            calls: Vec::new(),
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
        let call = {
            let mut standalone = kernel
                .standalone
                .lock()
                .expect("native standalone-call lock poisoned");
            kernel.prepare_call(&mut standalone, args, Some(outputs))?
        };
        self.calls.push((kernel.clone(), call));
        Ok(())
    }

    pub(crate) fn submit(self) -> Result<NativeTensorBatchCompletion, CallError> {
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
        let submission = submit(&BatchCalls(&self.calls), &access, retained, 1)?;
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
        buffers.extend(
            call.buffers
                .iter()
                .map(|(allocation, offset)| (&**allocation, *offset)),
        );
        Dispatch {
            kernel,
            words: &call.words,
            word_bytes: &call.word_bytes,
            launches: &call.launches,
            representations: &call.representations,
        }
    }

    fn plans(&self) -> Option<Vec<u64>> {
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
