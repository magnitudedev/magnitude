//! Compilation and execution of explicitly authored top-level Metal
//! implementations. This path deliberately consumes source plus a closed ABI;
//! it does not construct compiler kernels, plans, schedules, or portfolios.

use crate::{MetalBuffer, MetalDevice};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCommonCounterSetTimestamp, MTLComputeCommandEncoder, MTLComputePassDescriptor,
    MTLComputePipelineState, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor,
    MTLCounterSamplingPoint, MTLCounterSet, MTLDevice, MTLDispatchType, MTLLibrary, MTLSize,
    MTLStorageMode,
};
use seismic_compiler::errors::ExecutionError;
use seismic_target::NativeCompilationError;

/// Metal's `setBytes` limit, which bounds a direct entry's argument words.
pub const DIRECT_WORD_BYTES_LIMIT: usize = 4096;

/// Metal's buffer argument table size, which bounds a direct entry's
/// buffers, argument words and scalar-result slots together.
pub const DIRECT_BUFFER_SLOTS: usize = 31;

/// One kernel function of an authored Metal library, compiled into a
/// pipeline.
pub struct DirectPipeline {
    state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

// Metal pipeline states are immutable and documented as thread-safe.
unsafe impl Send for DirectPipeline {}
unsafe impl Sync for DirectPipeline {}

impl DirectPipeline {
    /// Compile `source` once and form a pipeline for each named kernel, in
    /// the given order.
    pub fn compile_all(
        device: &MetalDevice,
        source: &str,
        kernels: &[&str],
    ) -> Result<Vec<Self>, NativeCompilationError> {
        let source = NSString::from_str(source);
        let options = objc2_metal::MTLCompileOptions::new();
        #[allow(deprecated)]
        options.setFastMathEnabled(false);
        options.setMathMode(objc2_metal::MTLMathMode::Safe);
        options.setMathFloatingPointFunctions(objc2_metal::MTLMathFloatingPointFunctions::Precise);
        let library = device
            .handle()
            .raw()
            .newLibraryWithSource_options_error(&source, Some(&options))
            .map_err(|error| NativeCompilationError::ToolchainFailure(error.to_string()))?;
        kernels
            .iter()
            .map(|kernel| {
                let name = NSString::from_str(kernel);
                let function = library.newFunctionWithName(&name).ok_or_else(|| {
                    NativeCompilationError::MalformedToolchainOutput(format!(
                        "Metal library does not define kernel `{kernel}`"
                    ))
                })?;
                let state = device
                    .handle()
                    .raw()
                    .newComputePipelineStateWithFunction_error(&function)
                    .map_err(|error| NativeCompilationError::ToolchainFailure(error.to_string()))?;
                Ok(Self { state })
            })
            .collect()
    }

    pub fn max_threads_per_threadgroup(&self) -> u64 {
        self.state.maxTotalThreadsPerThreadgroup() as u64
    }

    /// Threadgroup memory the compiled function declares statically.
    pub fn static_threadgroup_bytes(&self) -> u64 {
        self.state.staticThreadgroupMemoryLength() as u64
    }
}

/// One direct launch as encoded.
pub struct DirectLaunch<'a> {
    pub pipeline: &'a DirectPipeline,
    pub buffers: &'a [(&'a MetalBuffer, u64)],
    pub words: &'a [u8],
    /// Scalar-result slots, bound after the words.
    pub scalar_results: (&'a MetalBuffer, u64),
    pub threadgroups: [u64; 3],
    pub threads_per_threadgroup: [u64; 3],
    pub threadgroup_bytes: u64,
}

/// The host monotonic clock in seconds, in the time base of Metal's
/// command-buffer `GPUStartTime`/`GPUEndTime` (mach absolute time).
pub fn host_seconds() -> f64 {
    unsafe extern "C" {
        fn clock_gettime_nsec_np(clock_id: std::ffi::c_int) -> u64;
    }
    const CLOCK_UPTIME_RAW: std::ffi::c_int = 8;
    // SAFETY: a pure libc clock read.
    unsafe { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) as f64 / 1e9 }
}

/// A GPU timestamp correlated with the [`host_seconds`] clock.
#[derive(Clone, Copy)]
struct TimestampPair {
    gpu: u64,
    host: f64,
}

fn sample_timestamps(device: &MetalDevice) -> TimestampPair {
    let mut cpu = 0u64;
    let mut gpu = 0u64;
    // `sampleTimestamps` pairs the GPU clock with a CPU clock of its own;
    // the host clock read beside it is the time base every interval uses.
    let host = host_seconds();
    // SAFETY: both pointers are valid for one write.
    unsafe {
        device.handle().raw().sampleTimestamps_gpuTimestamp(
            std::ptr::NonNull::from(&mut cpu),
            std::ptr::NonNull::from(&mut gpu),
        )
    };
    TimestampPair { gpu, host }
}

/// A device buffer of GPU timestamp samples shared by the timed batches of
/// one measurement. Metal bounds how many sample buffers may exist, so timed
/// batches take ranges of one buffer instead of allocating their own.
pub struct LaunchTimestamps {
    samples: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    capacity: usize,
    /// First unreserved sample.
    next: std::sync::Mutex<usize>,
}

// The sample buffer is written by the device and resolved after completion.
unsafe impl Send for LaunchTimestamps {}
unsafe impl Sync for LaunchTimestamps {}

impl LaunchTimestamps {
    /// A buffer of `samples` timestamp samples (two per timed launch).
    pub fn new(device: &MetalDevice, samples: usize) -> Result<Self, ExecutionError> {
        let failed = |detail: &str| ExecutionError::SubmissionFailed(detail.into());
        let raw = device.handle().raw();
        if !raw.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary) {
            return Err(failed(
                "Metal device does not sample counters at stage boundaries",
            ));
        }
        let timestamp_set = raw
            .counterSets()
            .and_then(|sets| {
                // SAFETY: a static Metal string constant.
                let wanted = unsafe { MTLCommonCounterSetTimestamp };
                sets.iter().find(|set| &*set.name() == wanted)
            })
            .ok_or_else(|| failed("Metal device has no timestamp counter set"))?;
        let descriptor = MTLCounterSampleBufferDescriptor::new();
        descriptor.setCounterSet(Some(&timestamp_set));
        descriptor.setStorageMode(MTLStorageMode::Shared);
        // SAFETY: Metal validates the count when forming the buffer.
        unsafe { descriptor.setSampleCount(samples) };
        let buffer = raw
            .newCounterSampleBufferWithDescriptor_error(&descriptor)
            .map_err(|error| {
                ExecutionError::SubmissionFailed(format!(
                    "Metal refused a {samples}-sample timestamp buffer: {}",
                    error.localizedDescription()
                ))
            })?;
        Ok(Self {
            samples: buffer,
            capacity: samples,
            next: std::sync::Mutex::new(0),
        })
    }

    fn reserve(&self, samples: usize) -> Result<usize, ExecutionError> {
        let mut next = self.next.lock().expect("timestamp cursor lock is never poisoned");
        let first = *next;
        if first + samples > self.capacity {
            return Err(ExecutionError::SubmissionFailed(format!(
                "timed launches need {} timestamp samples; the buffer holds {}",
                first + samples,
                self.capacity
            )));
        }
        *next += samples;
        Ok(first)
    }

    /// Release every range. Only valid once every timed batch that reserved
    /// one has been resolved.
    pub fn reset(&self) {
        *self.next.lock().expect("timestamp cursor lock is never poisoned") = 0;
    }
}

/// How a batch places its launches in compute encoders.
enum Encoding {
    /// Every launch in one serial encoder: the production form.
    Serial(Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>),
    /// Every launch in its own encoder, with the encoder's start and end
    /// timestamps sampled at stage boundaries into its range of
    /// `timestamps`. Encoders of one command buffer run in order over
    /// tracked resources.
    Timed {
        timestamps: std::sync::Arc<LaunchTimestamps>,
        /// First sample of this batch's range.
        first: usize,
        capacity: usize,
        /// Per `encode` call, the launch's sample pair (absent when the
        /// launch was empty and encoded nothing).
        launches: Vec<Option<usize>>,
        encoded: usize,
        calibration: TimestampPair,
    },
}

/// Ordered direct launches encoded into one command buffer. The production
/// form uses one serial compute encoder, which orders every launch after the
/// previous one's writes.
pub struct DirectBatch {
    device: MetalDevice,
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoding: Encoding,
}

fn command_buffer(
    device: &MetalDevice,
) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>, ExecutionError> {
    device.queue().commandBuffer().ok_or_else(|| {
        ExecutionError::SubmissionFailed("Metal could not create a native command buffer".into())
    })
}

impl DirectBatch {
    pub fn new(device: &MetalDevice) -> Result<Self, ExecutionError> {
        let command = command_buffer(device)?;
        let encoder = command.computeCommandEncoder().ok_or_else(|| {
            ExecutionError::SubmissionFailed(
                "Metal could not create a native compute encoder".into(),
            )
        })?;
        Ok(Self {
            device: device.clone(),
            command,
            encoding: Encoding::Serial(encoder),
        })
    }

    /// A measurement batch of at most `launches` launches: each launch gets
    /// its own encoder, whose GPU start and end are timestamped. Encoder
    /// boundaries add device time, so a timed batch attributes time to
    /// launches but does not measure the production step.
    pub fn timed(
        device: &MetalDevice,
        timestamps: &std::sync::Arc<LaunchTimestamps>,
        launches: usize,
    ) -> Result<Self, ExecutionError> {
        let first = timestamps.reserve(2 * launches)?;
        Ok(Self {
            device: device.clone(),
            command: command_buffer(device)?,
            encoding: Encoding::Timed {
                timestamps: timestamps.clone(),
                first,
                capacity: launches,
                launches: Vec::new(),
                encoded: 0,
                calibration: sample_timestamps(device),
            },
        })
    }

    /// Record a launch that does no device work (inactive, or an empty
    /// grid): nothing is encoded, and a timed batch keeps its place with no
    /// interval.
    pub fn skip(&mut self) {
        if let Encoding::Timed { launches, .. } = &mut self.encoding {
            launches.push(None);
        }
    }

    pub fn encode(&mut self, launch: &DirectLaunch<'_>) -> Result<(), ExecutionError> {
        if launch.threadgroups.contains(&0) || launch.threads_per_threadgroup.contains(&0) {
            self.skip();
            return Ok(());
        }
        let threads = launch
            .threads_per_threadgroup
            .iter()
            .try_fold(1u64, |product, value| product.checked_mul(*value))
            .ok_or_else(|| {
                ExecutionError::SubmissionFailed("native threadgroup size overflowed".into())
            })?;
        if threads > launch.pipeline.max_threads_per_threadgroup() {
            return Err(ExecutionError::SubmissionFailed(format!(
                "native launch requests {threads} threads per threadgroup, but the pipeline allows {}",
                launch.pipeline.max_threads_per_threadgroup()
            )));
        }
        if launch.words.len() > DIRECT_WORD_BYTES_LIMIT {
            return Err(ExecutionError::SubmissionFailed(
                "native ABI words exceed Metal setBytes limit".into(),
            ));
        }
        match &mut self.encoding {
            Encoding::Serial(encoder) => encode_launch(encoder, launch),
            Encoding::Timed {
                timestamps,
                first,
                capacity,
                launches,
                encoded,
                ..
            } => {
                if *encoded == *capacity {
                    return Err(ExecutionError::SubmissionFailed(
                        "timed Metal batch exceeds its declared launch count".into(),
                    ));
                }
                let pass = MTLComputePassDescriptor::computePassDescriptor();
                pass.setDispatchType(MTLDispatchType::Serial);
                // SAFETY: index 0 always exists; sample indices lie inside
                // the batch's reserved range.
                unsafe {
                    let attachment = pass.sampleBufferAttachments().objectAtIndexedSubscript(0);
                    attachment.setSampleBuffer(Some(&timestamps.samples));
                    attachment.setStartOfEncoderSampleIndex(*first + 2 * *encoded);
                    attachment.setEndOfEncoderSampleIndex(*first + 2 * *encoded + 1);
                }
                let encoder = self
                    .command
                    .computeCommandEncoderWithDescriptor(&pass)
                    .ok_or_else(|| {
                        ExecutionError::SubmissionFailed(
                            "Metal could not create a timed compute encoder".into(),
                        )
                    })?;
                encode_launch(&encoder, launch);
                encoder.endEncoding();
                launches.push(Some(*encoded));
                *encoded += 1;
            }
        }
        Ok(())
    }

    /// Commit without waiting. The queue runs command buffers in commit
    /// order.
    pub fn commit(self) -> DirectSubmission {
        let timing = match self.encoding {
            Encoding::Serial(encoder) => {
                encoder.endEncoding();
                None
            }
            Encoding::Timed {
                timestamps,
                first,
                launches,
                encoded,
                calibration,
                ..
            } => Some(LaunchTiming {
                timestamps,
                first,
                launches,
                encoded,
                calibration,
            }),
        };
        self.command.commit();
        DirectSubmission {
            device: self.device,
            command: self.command,
            timing,
        }
    }
}

fn encode_launch(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, launch: &DirectLaunch<'_>) {
    encoder.setComputePipelineState(&launch.pipeline.state);
    for (index, (buffer, offset)) in launch.buffers.iter().enumerate() {
        unsafe { encoder.setBuffer_offset_atIndex(Some(buffer.raw()), *offset as usize, index) }
    }
    let zero = 0u8;
    let words = if launch.words.is_empty() {
        std::slice::from_ref(&zero)
    } else {
        launch.words
    };
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&words[0]).cast(),
            words.len(),
            launch.buffers.len(),
        );
        encoder.setBuffer_offset_atIndex(
            Some(launch.scalar_results.0.raw()),
            launch.scalar_results.1 as usize,
            launch.buffers.len() + 1,
        );
    }
    if launch.threadgroup_bytes != 0 {
        // Metal requires threadgroup memory lengths in multiples of 16.
        let bytes = launch.threadgroup_bytes.div_ceil(16) * 16;
        unsafe { encoder.setThreadgroupMemoryLength_atIndex(bytes as usize, 0) };
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: launch.threadgroups[0] as usize,
            height: launch.threadgroups[1] as usize,
            depth: launch.threadgroups[2] as usize,
        },
        MTLSize {
            width: launch.threads_per_threadgroup[0] as usize,
            height: launch.threads_per_threadgroup[1] as usize,
            depth: launch.threads_per_threadgroup[2] as usize,
        },
    );
}

struct LaunchTiming {
    timestamps: std::sync::Arc<LaunchTimestamps>,
    first: usize,
    launches: Vec<Option<usize>>,
    encoded: usize,
    calibration: TimestampPair,
}

/// How often a host wait polls a command buffer's status.
///
/// A host blocked in `waitUntilCompleted` leaves its CPU idle for the whole
/// command buffer, and Apple silicon then runs the GPU's memory-bound work
/// measurably slower: on an M4 Pro, 4B decode takes 3.5% more device time
/// than while a host thread polls at this interval (every kernel 2–4%).
/// Polling costs about 0.8 W of CPU (about 3% more energy per decoded
/// token) and observes completion about 20 µs sooner than the blocking
/// wait's wake-up. Polling every 100 µs keeps only a third of the gain.
/// Measurements: `specs/26-09-23/benchmark-results.md`, "Metal host wait".
const WAIT_POLL: std::time::Duration = std::time::Duration::from_micros(20);

/// A committed direct command buffer.
pub struct DirectSubmission {
    device: MetalDevice,
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    timing: Option<LaunchTiming>,
}

// Waiting on and querying a committed command buffer is thread-safe.
unsafe impl Send for DirectSubmission {}
unsafe impl Sync for DirectSubmission {}

impl DirectSubmission {
    pub fn is_complete(&self) -> bool {
        matches!(
            self.command.status(),
            MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
        )
    }

    /// Wait for completion by polling the status every [`WAIT_POLL`].
    pub fn wait_complete(&self) {
        while !self.is_complete() {
            std::thread::sleep(WAIT_POLL);
        }
    }

    /// Wait and report the command buffer's outcome.
    pub fn finish(&self) -> Result<(), ExecutionError> {
        self.wait_complete();
        match self.command.error() {
            Some(error) => Err(ExecutionError::SubmissionFailed(
                error.localizedDescription().to_string(),
            )),
            None => Ok(()),
        }
    }

    /// Device execution time of the completed command buffer.
    pub fn device_seconds(&self) -> f64 {
        self.command.GPUEndTime() - self.command.GPUStartTime()
    }

    /// The completed command buffer's device execution interval on the
    /// [`host_seconds`] clock.
    pub fn device_interval(&self) -> (f64, f64) {
        (self.command.GPUStartTime(), self.command.GPUEndTime())
    }

    /// For a timed batch, each `encode` call's device interval on the
    /// [`host_seconds`] clock, in encode order (`None` for an empty launch).
    /// `None` for a production batch. Call after completion.
    pub fn launch_intervals(&self) -> Result<Option<Vec<Option<(f64, f64)>>>, ExecutionError> {
        let Some(timing) = &self.timing else {
            return Ok(None);
        };
        let failed = |detail: &str| ExecutionError::SubmissionFailed(detail.into());
        if !self.is_complete() {
            return Err(failed("timed Metal batch is read before completion"));
        }
        let samples = if timing.encoded == 0 {
            Vec::new()
        } else {
            // SAFETY: the range lies inside the batch's reserved samples.
            let data = unsafe {
                timing
                    .timestamps
                    .samples
                    .resolveCounterRange(NSRange::new(timing.first, 2 * timing.encoded))
            }
            .ok_or_else(|| failed("Metal did not resolve the timed batch's samples"))?;
            data.to_vec()
                .chunks_exact(8)
                .map(|bytes| u64::from_le_bytes(bytes.try_into().expect("eight bytes")))
                .collect::<Vec<_>>()
        };
        // GPU ticks map linearly onto the host clock between the pair taken
        // when the batch was formed and one taken now.
        let early = timing.calibration;
        let late = sample_timestamps(&self.device);
        if late.gpu <= early.gpu {
            return Err(failed("Metal GPU timestamps did not advance"));
        }
        let seconds_per_tick = (late.host - early.host) / (late.gpu - early.gpu) as f64;
        let host = |tick: u64| early.host + (tick as f64 - early.gpu as f64) * seconds_per_tick;
        timing
            .launches
            .iter()
            .map(|slot| {
                slot.map(|index| {
                    let (start, end) = (samples[2 * index], samples[2 * index + 1]);
                    // Metal reports a sample it could not take as all ones.
                    if start == u64::MAX || end == u64::MAX || end < start {
                        return Err(failed("Metal dropped a timed launch's timestamp"));
                    }
                    Ok((host(start), host(end)))
                })
                .transpose()
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some)
    }
}
