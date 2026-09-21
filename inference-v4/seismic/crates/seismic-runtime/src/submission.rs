//! Submission of prepared invocations (package R1).
//!
//! A `Submission` holds prepared invocations in source order and executes
//! them through the backend executors over direct sealed handles and dense
//! validated bindings. Each invocation's artifact yields one paired
//! `SealedBackend` bundle — device, physical plan, and native artifact of the
//! same compilation — so no executor ever receives halves assembled by hand.
//! The only data-dependent semantic failure is a named retained guard
//! (`SafetyViolation`); every other failure is external, and backend executor
//! failures map straight through the shared taxonomy. No raw binding reaches
//! an executor.

use crate::invocation::PreparedInvocation;
use crate::plan::{InvocationResults, ScalarResult, SealedBackend};
use crate::ExecutionObservation;
use seismic_realization::failure::{
    ExecutionFailure, ExternalFailure, ExternalStage, InvalidInvocation,
};
use seismic_realization::ids::{BufferSlot, DenseIndex};
#[cfg(target_os = "macos")]
use seismic_realization::invocation::InvocationValues;
#[cfg(target_os = "macos")]
use std::sync::Arc;

#[derive(Default)]
pub struct Submission {
    invocations: Vec<PreparedInvocation>,
}

impl Submission {
    pub fn single(invocation: PreparedInvocation) -> Submission {
        Submission {
            invocations: vec![invocation],
        }
    }

    pub fn append(&mut self, mut other: Submission) {
        self.invocations.append(&mut other.invocations);
    }

    pub fn len(&self) -> usize {
        self.invocations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.invocations.is_empty()
    }

    /// Execute every invocation in source order; returns the results of
    /// each in the same order.
    pub fn execute(self) -> Result<Vec<InvocationResults>, ExecutionFailure> {
        self.invocations
            .iter()
            .map(|invocation| execute_one(invocation).map(|(results, _)| results))
            .collect()
    }

    /// Execute with per-invocation wall-time observation.
    pub fn execute_observed(
        self,
    ) -> Result<Vec<(InvocationResults, ExecutionObservation)>, ExecutionFailure> {
        self.invocations
            .iter()
            .map(|invocation| {
                let started = std::time::Instant::now();
                let (results, launches) = execute_one(invocation)?;
                Ok((
                    results,
                    ExecutionObservation {
                        host_seconds: started.elapsed().as_secs_f64(),
                        launches,
                    },
                ))
            })
            .collect()
    }
}

/// Execute one prepared invocation through its artifact's backend executor.
fn execute_one(
    invocation: &PreparedInvocation,
) -> Result<(InvocationResults, Vec<seismic_realization::physical::LaunchExecution>), ExecutionFailure> {
    match invocation.artifact().backend() {
        SealedBackend::Cpu {
            device: cpu,
            physical,
            native,
        } => {
            let buffers = invocation.buffers();
            let mut pointers: Vec<*mut u8> = Vec::with_capacity(buffers.len());
            for (index, validated) in buffers.iter().enumerate() {
                // Every bound buffer was validated against the CPU device's
                // domain, so its storage is the host arm; any other storage
                // kind is a wrong-device buffer.
                let pointer = match validated.buffer.host_pointer() {
                    Some(pointer) => pointer,
                    None => {
                        return Err(wrong_device(BufferSlot::from_index(index)))
                    }
                };
                pointers.push(pointer);
            }
            let mut workers = cpu
                .workers
                .try_borrow_mut()
                .map_err(|_| execution_contention())?;
            let outputs = seismic_cpu::CpuExecutor::new(&mut workers).execute(
                &seismic_cpu::ExecutionInputs {
                    physical,
                    native,
                    buffers: &pointers,
                    values: invocation.values(),
                },
            )?;
            Ok((
                InvocationResults {
                    planes: invocation.results().to_vec(),
                    scalars: outputs
                        .scalars
                        .into_iter()
                        .map(|output| ScalarResult {
                            path: output.path,
                            endpoint: output.endpoint,
                            dtype: output.dtype,
                            value: output.value,
                        })
                        .collect(),
                },
                Vec::new(),
            ))
        }
        #[cfg(target_os = "macos")]
        SealedBackend::Metal { native } => {
            let surface = MetalSurface {
                native,
                invocation,
            };
            let outcome = seismic_metal::runtime::Executor::new().execute(&surface)?;
            Ok((
                InvocationResults {
                    planes: invocation.results().to_vec(),
                    scalars: outcome
                        .scalars
                        .into_iter()
                        .map(|output| ScalarResult {
                            path: output.path,
                            endpoint: output.endpoint,
                            dtype: output.dtype,
                            value: output.value,
                        })
                        .collect(),
                },
                outcome.launches,
            ))
        }
        SealedBackend::Cuda {
            device,
            physical,
            native,
        } => {
            let executor =
                seismic_cuda::runtime::Executor::new(device, native, physical);
            let buffers = invocation.buffers();
            let mut pointers: Vec<u64> = Vec::with_capacity(buffers.len());
            for (index, validated) in buffers.iter().enumerate() {
                // Every bound buffer was validated against the CUDA device's
                // domain, so its storage is the device arm; any other storage
                // kind is a wrong-device buffer.
                let pointer = match validated.buffer.device_pointer() {
                    Some(pointer) => pointer,
                    None => {
                        return Err(wrong_device(BufferSlot::from_index(index)))
                    }
                };
                pointers.push(pointer);
            }
            let bridge = seismic_cuda::runtime::Prepared {
                values: invocation.values(),
                // Dense by `BufferSlot`: the table was built in slot order.
                buffer_pointer: &|slot: BufferSlot| pointers[slot.index()],
            };
            let outcome = executor.execute(&bridge)?;
            let scalars = physical
                .result_fields()
                .iter()
                .map(|(index, field)| ScalarResult {
                    path: field.path.clone(),
                    endpoint: field.endpoint,
                    dtype: field.dtype,
                    value: seismic_cuda::runtime::decode_result_word(
                        outcome.result_words[index.index()],
                        field.dtype,
                    ),
                })
                .collect();
            Ok((
                InvocationResults {
                    planes: invocation.results().to_vec(),
                    scalars,
                },
                Vec::new(),
            ))
        }
    }
}

/// The R1 invocation surface the Metal executor consumes. The native artifact
/// comes from the invocation's sealed Metal arm, never from a hand-assembled
/// pair.
#[cfg(target_os = "macos")]
struct MetalSurface<'a> {
    native: &'a Arc<seismic_metal::native::NativeArtifact>,
    invocation: &'a PreparedInvocation,
}

#[cfg(target_os = "macos")]
impl seismic_metal::runtime::InvocationSurface for MetalSurface<'_> {
    fn metal_artifact(&self) -> &seismic_metal::native::NativeArtifact {
        self.native
    }

    fn metal_buffer(&self, slot: BufferSlot) -> Option<&seismic_metal::runtime::Buffer> {
        self.invocation.buffer(slot).buffer.as_metal()
    }

    fn values(&self) -> &InvocationValues {
        self.invocation.values()
    }

    fn result_planes(&self) -> &[seismic_metal::runtime::OwnedResultPlane] {
        &self.invocation.metal_result_planes
    }
}

/// A bound buffer whose storage kind disagrees with the executing device.
fn wrong_device(slot: BufferSlot) -> ExecutionFailure {
    ExecutionFailure::Invocation(InvalidInvocation::BufferDevice { slot })
}

/// The host device is already executing one submission.
fn execution_contention() -> ExecutionFailure {
    ExecutionFailure::External(ExternalFailure {
        stage: ExternalStage::Submission,
        detail: "the CPU device is already executing".into(),
    })
}
