//! Work as `seismic-measure` times it: a formed kernel's calls
//! ([`KernelPoint`]), each argument set of a rotation validated and placed
//! once and every pass resubmitting them.

use super::{CallError, NativeBoundCall, NativePrepared, NativeSubmission, StandaloneCalls};
use crate::api::kernel::{EncodedArgs, EncodedOutputs};
use crate::api::tensor::TensorInner;
use seismic_measure::{Gauge, Pending, Work};
use std::collections::HashSet;
use std::sync::Arc;

/// A formed kernel's calls over a rotation of argument sets, each placed
/// once with results of its own: a pass is one call of each, in order.
pub struct KernelPoint {
    kernel: Arc<NativePrepared>,
    calls: Vec<NativeBoundCall>,
    arguments: Vec<EncodedArgs>,
    rotation_bytes: u64,
}

impl KernelPoint {
    /// Validate and place every call of `rotation`. Results of each argument
    /// set are allocated once and reused by every pass, or taken from
    /// `outputs` (and added to it) where a caller shares results among the
    /// placements of one point; scratch is the standalone arena.
    pub(crate) fn place(
        kernel: &Arc<NativePrepared>,
        rotation: Vec<EncodedArgs>,
        mut outputs: Option<&mut Vec<Vec<Arc<TensorInner>>>>,
    ) -> Result<Self, CallError> {
        if rotation.is_empty() {
            return Err(CallError::Workflow(crate::api::WorkflowError::Empty));
        }
        let arguments = rotation.clone();
        let calls = {
            let mut standalone = kernel
                .standalone
                .lock()
                .expect("native standalone-call lock poisoned");
            rotation
                .into_iter()
                .enumerate()
                .map(|(index, args)| {
                    let supplied =
                        outputs
                            .as_ref()
                            .and_then(|pool| pool.get(index))
                            .map(|tensors| {
                                let mut encoded = EncodedOutputs::new();
                                for tensor in tensors {
                                    encoded.push_tensor(tensor.clone());
                                }
                                encoded
                            });
                    let call = kernel.prepare_call(&mut standalone, args, supplied)?;
                    if let Some(pool) = outputs.as_mut() {
                        if index == pool.len() {
                            pool.push(call.results.clone());
                        }
                    }
                    Ok(call)
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        let mut distinct = HashSet::new();
        for call in &calls {
            for (allocation, _) in &call.access {
                distinct.insert((allocation.identity(), allocation.bytes()));
            }
        }
        Ok(Self {
            kernel: kernel.clone(),
            calls,
            arguments,
            rotation_bytes: distinct.into_iter().map(|(_, bytes)| bytes).sum(),
        })
    }

    /// Distinct bytes the rotation's calls read and write.
    pub fn rotation_bytes(&self) -> u64 {
        self.rotation_bytes
    }

    /// The formed kernel the point calls.
    pub(crate) fn kernel(&self) -> &Arc<NativePrepared> {
        &self.kernel
    }

    /// The placed calls, one per argument set.
    pub(super) fn calls(&self) -> &[NativeBoundCall] {
        &self.calls
    }

    /// The argument sets of the rotation.
    pub(crate) fn arguments(&self) -> &[EncodedArgs] {
        &self.arguments
    }
}

impl Pending<CallError> for NativeSubmission {
    fn seconds(&self) -> Result<f64, CallError> {
        self.device_seconds()
    }

    fn wait(&self) -> Result<(), CallError> {
        NativeSubmission::wait(self)
    }
}

impl Gauge<CallError> for KernelPoint {
    fn submit(&self, passes: usize) -> Result<Box<dyn Pending<CallError>>, CallError> {
        Ok(Box::new(Work::submit(self, passes)?))
    }
}

impl Work for KernelPoint {
    type Error = CallError;
    type Pending = NativeSubmission;

    /// `passes` passes over the rotation as one unit of device work.
    /// Writable state is not restored between passes: every configuration
    /// is timed the same way.
    fn submit(&self, passes: usize) -> Result<NativeSubmission, CallError> {
        StandaloneCalls {
            kernel: &self.kernel,
            calls: &self.calls,
        }
        .submit(passes)
    }

    fn calls_per_pass(&self) -> usize {
        self.calls.len()
    }

    /// The same calls placed again, with results of their own.
    fn gauge(&self) -> Result<Box<dyn Gauge<CallError>>, CallError> {
        Ok(Box::new(Self::place(
            &self.kernel,
            self.arguments.clone(),
            None,
        )?))
    }
}
