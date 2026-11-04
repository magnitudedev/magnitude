//! Execution of explicitly authored top-level Metal implementations, whose
//! pipelines the Metal toolchain forms. This path deliberately consumes
//! source plus a closed ABI; it does not construct compiler kernels, plans,
//! schedules, or portfolios.

use crate::{MetalBuffer, MetalDevice};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCommonCounterSetTimestamp, MTLComputeCommandEncoder, MTLComputePassDescriptor,
    MTLComputePipelineState, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor,
    MTLCounterSamplingPoint, MTLCounterSet, MTLDevice, MTLDispatchType, MTLEvent, MTLSharedEvent,
    MTLSize, MTLStorageMode,
};
use seismic_compiler::errors::ExecutionError;

/// Metal's `setBytes` limit, which bounds a direct entry's argument words.
pub const DIRECT_WORD_BYTES_LIMIT: usize = 4096;

/// Metal's buffer argument table size, which bounds a direct entry's
/// buffers, argument words and scalar-result slots together.
pub const DIRECT_BUFFER_SLOTS: usize = 31;

/// One kernel function of an authored Metal library, compiled into a
/// pipeline.
#[derive(Clone)]
pub struct DirectPipeline {
    state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

// Metal pipeline states are immutable and documented as thread-safe.
unsafe impl Send for DirectPipeline {}
unsafe impl Sync for DirectPipeline {}

impl DirectPipeline {
    pub(crate) fn from_state(state: Retained<ProtocolObject<dyn MTLComputePipelineState>>) -> Self {
        Self { state }
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
    /// Scalar-result slots, bound after the words; nothing is bound there
    /// for an entry without scalar results.
    pub scalar_results: Option<(&'a MetalBuffer, u64)>,
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
    crate::pooled(|| unsafe {
        device.handle().raw().sampleTimestamps_gpuTimestamp(
            std::ptr::NonNull::from(&mut cpu),
            std::ptr::NonNull::from(&mut gpu),
        )
    });
    TimestampPair { gpu, host }
}

type Page = Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>;

/// The process's GPU timestamp pages, shared by every timed batch: Metal
/// sample buffers of [`PAGE_SAMPLES`] samples each. Metal bounds how many a
/// process may hold (32 on an M4 Pro, macOS 15.7.9, whatever their size,
/// another process's not counting: `tests/sample_buffers.rs`), and only this
/// pool forms them, so it forms a page when a lease finds none free, until
/// Metal first refuses one: the pages formed then are the process's
/// allowance, and the pool's size. Each command buffer of a timed batch
/// leases one page for its launches' start and end samples; when the
/// command buffer completes, its completion handler resolves the page's
/// ticks into host memory and frees the page. Timed work of any length
/// therefore holds at most the pool.
pub struct LaunchTimestamps {
    /// The device whose command buffers sample the pages.
    registry_id: u64,
    /// The device's timestamp counter set, which every page samples.
    counters: Retained<ProtocolObject<dyn MTLCounterSet>>,
    pool: std::sync::Mutex<Pool>,
    /// Signalled when a page is freed or a part's ticks are resolved.
    changed: std::sync::Condvar,
}

/// Samples of one page: Metal's largest sample buffer, 32 KiB of 8-byte
/// timestamps (even: a launch's pair lies in one page).
const PAGE_SAMPLES: usize = 4_096;

/// The pool, formed by the first timed work and kept for the process: its
/// pages are the process's whole allowance.
static PROCESS_POOL: std::sync::Mutex<Option<&'static LaunchTimestamps>> =
    std::sync::Mutex::new(None);

struct Pool {
    free: Vec<Page>,
    /// The pages formed: once `full`, the pool's size.
    formed: usize,
    /// Whether Metal refused a page: no more are formed.
    full: bool,
}

/// A committed part's resolved ticks, set by its command buffer's
/// completion handler: `None` when Metal did not resolve them.
type PartTicks = std::sync::Arc<std::sync::OnceLock<Option<Vec<u64>>>>;

/// The page leased by a timed batch's open command buffer and the samples
/// its launches took. Dropped uncommitted (the batch failed or was
/// abandoned), the page returns to the pool: nothing will run on it.
struct Lease {
    timestamps: &'static LaunchTimestamps,
    /// Taken only when the command buffer commits.
    page: Option<Page>,
    samples: usize,
}

impl Lease {
    fn full(&self) -> bool {
        self.samples == PAGE_SAMPLES
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        if let Some(page) = self.page.take() {
            self.timestamps.lock().free.push(page);
            self.timestamps.changed.notify_all();
        }
    }
}

// The sample buffers are written by the device and resolved after completion.
unsafe impl Send for LaunchTimestamps {}
unsafe impl Sync for LaunchTimestamps {}

impl LaunchTimestamps {
    /// The process's pool, formed on first use with its first page: a
    /// device that cannot sample timestamps fails here, as does one whose
    /// first page Metal refuses, and so does a second Metal device, since
    /// the pages sample the first one's command buffers.
    pub fn shared(device: &MetalDevice) -> Result<&'static Self, ExecutionError> {
        let registry_id = device.handle().registry_id();
        let mut shared = PROCESS_POOL
            .lock()
            .expect("process timestamp pool lock is never poisoned");
        let pool = match *shared {
            Some(pool) => pool,
            None => {
                let counters = crate::pooled(|| Self::counters(device))?;
                let first = crate::pooled(|| Self::form(device, &counters)).map_err(|refusal| {
                    ExecutionError::SubmissionFailed(format!(
                        "Metal refused the process's first {PAGE_SAMPLES}-sample timestamp buffer: {refusal}"
                    ))
                })?;
                let pool: &'static Self = Box::leak(Box::new(Self {
                    registry_id,
                    counters,
                    pool: std::sync::Mutex::new(Pool {
                        free: vec![first],
                        formed: 1,
                        full: false,
                    }),
                    changed: std::sync::Condvar::new(),
                }));
                *shared = Some(pool);
                pool
            }
        };
        if pool.registry_id != registry_id {
            return Err(ExecutionError::SubmissionFailed(
                "the process's Metal timestamp pages sample another device's work".into(),
            ));
        }
        Ok(pool)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Pool> {
        self.pool
            .lock()
            .expect("timestamp pool lock is never poisoned")
    }

    /// The device's timestamp counter set, sampled at stage boundaries.
    fn counters(
        device: &MetalDevice,
    ) -> Result<Retained<ProtocolObject<dyn MTLCounterSet>>, ExecutionError> {
        let failed = |detail: &str| ExecutionError::SubmissionFailed(detail.into());
        let raw = device.handle().raw();
        if !raw.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary) {
            return Err(failed(
                "Metal device does not sample counters at stage boundaries",
            ));
        }
        raw.counterSets()
            .and_then(|sets| {
                // SAFETY: a static Metal string constant.
                let wanted = unsafe { MTLCommonCounterSetTimestamp };
                sets.iter().find(|set| &*set.name() == wanted)
            })
            .ok_or_else(|| failed("Metal device has no timestamp counter set"))
    }

    /// A new page, or Metal's refusal.
    fn form(
        device: &MetalDevice,
        counters: &ProtocolObject<dyn MTLCounterSet>,
    ) -> Result<Page, String> {
        let descriptor = MTLCounterSampleBufferDescriptor::new();
        descriptor.setCounterSet(Some(counters));
        descriptor.setStorageMode(MTLStorageMode::Shared);
        // SAFETY: Metal validates the count when forming the buffer.
        unsafe { descriptor.setSampleCount(PAGE_SAMPLES) };
        device
            .handle()
            .raw()
            .newCounterSampleBufferWithDescriptor_error(&descriptor)
            .map_err(|error| error.localizedDescription().to_string())
    }

    /// A free page, or a new one until Metal refuses one. With neither,
    /// the wait ends when a committed part's command buffer completes (its
    /// completion handler frees its page) or a command buffer still being
    /// encoded commits or drops its lease. The leasing batch holds no page
    /// of its own uncommitted: a command buffer leases once, at its first
    /// launch, so its batch's earlier pages are all committed and can
    /// complete. When `blocked`, the count of the leasing batch's committed
    /// parts waiting behind a host gate of its own not yet opened, is every
    /// page formed, no page frees until a gate opens, which the encoding
    /// thread waiting here never could ([`DirectBatch::wait_event`]): the
    /// lease is refused.
    fn lease(
        &'static self,
        device: &MetalDevice,
        blocked: impl Fn() -> usize,
    ) -> Result<Lease, ExecutionError> {
        let mut pool = self.lock();
        let page = loop {
            if let Some(page) = pool.free.pop() {
                break page;
            }
            if !pool.full {
                match crate::pooled(|| Self::form(device, &self.counters)) {
                    Ok(page) => {
                        pool.formed += 1;
                        break page;
                    }
                    // The refusal is the process's bound: the pages formed
                    // so far are the pool.
                    Err(_) => pool.full = true,
                }
                continue;
            }
            if blocked() == pool.formed {
                return Err(ExecutionError::SubmissionFailed(format!(
                    "timestamp pool exhausted: all {} pages are in flight behind a host gate \
                     of this batch that is not yet open",
                    pool.formed
                )));
            }
            pool = self
                .changed
                .wait(pool)
                .expect("timestamp pool lock is never poisoned");
        };
        Ok(Lease {
            timestamps: self,
            page: Some(page),
            samples: 0,
        })
    }

    /// `lease`'s command buffer is about to be committed: its completion
    /// handler resolves the page's samples and frees the page. The ticks it
    /// will resolve.
    fn commit(
        &'static self,
        mut lease: Lease,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> PartTicks {
        let ticks = PartTicks::default();
        let page = lease.page.take().expect("a lease holds its page until committed");
        let samples = lease.samples;
        let resolved = ticks.clone();
        let handler = block2::RcBlock::new(
            move |_: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                // SAFETY: the range lies inside the page; the device wrote
                // it before completing.
                let values = crate::pooled(|| unsafe {
                    page.resolveCounterRange(NSRange::new(0, samples))
                        .map(|data| {
                            data.to_vec()
                                .chunks_exact(8)
                                .map(|bytes| {
                                    u64::from_le_bytes(bytes.try_into().expect("eight bytes"))
                                })
                                .collect()
                        })
                });
                let mut pool = self.lock();
                resolved
                    .set(values)
                    .expect("a command buffer completes once");
                pool.free.push(page.clone());
                drop(pool);
                self.changed.notify_all();
            },
        );
        // SAFETY: Metal copies the block, which owns everything it uses.
        crate::pooled(|| unsafe { command.addCompletedHandler(block2::RcBlock::as_ptr(&handler)) });
        ticks
    }

    /// A committed part's ticks, once its completion handler has resolved
    /// them: the caller has seen the command buffer complete, and Metal runs
    /// the handler on completion.
    fn resolved<'a>(&self, ticks: &'a PartTicks) -> Option<&'a [u64]> {
        let mut pool = self.lock();
        while ticks.get().is_none() {
            pool = self
                .changed
                .wait(pool)
                .expect("timestamp pool lock is never poisoned");
        }
        ticks.get().expect("resolved above").as_deref()
    }
}

/// How a batch places its launches in compute encoders.
enum Encoding {
    /// Every launch between two event commands in one serial encoder: the
    /// production form. `None` after an event command, until the next
    /// launch opens the next encoder.
    Serial(Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>),
    /// Every launch in its own encoder, with the encoder's start and end
    /// timestamps sampled at stage boundaries into the page its command
    /// buffer leased from `timestamps`; each encoder runs after the one
    /// before ([`ordered`]). A command buffer whose page is full is
    /// committed and the next leases another.
    Timed {
        timestamps: &'static LaunchTimestamps,
        /// The current command buffer's page, from its first launch.
        lease: Option<Lease>,
        /// The committed command buffers' ticks, in order: together the
        /// sample pairs of the launches encoded before the current one's.
        parts: Vec<PartTicks>,
        /// The batch's host gates, in order.
        gates: Vec<Gate>,
        /// Per `encode` call, the launch's sample pair in the batch's order
        /// (absent when the launch was empty and encoded nothing).
        launches: Vec<Option<usize>>,
        encoded: usize,
        calibration: TimestampPair,
    },
}

/// A host gate of a timed batch ([`DirectBatch::wait_event`]): the parts
/// from `first_part` on run after it, so their pages free only once
/// `event` reaches `value`.
struct Gate {
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    value: u64,
    first_part: usize,
}

/// The parts of `parts` that wait behind a gate of `gates` not yet open:
/// those from the first such gate's on, uncompleted.
fn blocked_parts(gates: &[Gate], parts: &[PartTicks]) -> usize {
    gates
        .iter()
        .find(|gate| crate::pooled(|| gate.event.signaledValue()) < gate.value)
        .map_or(0, |gate| {
            parts[gate.first_part..]
                .iter()
                .filter(|ticks| ticks.get().is_none())
                .count()
        })
}

impl Encoding {
    /// `command` is about to be committed: a timed batch's page goes in
    /// flight with it.
    fn committing(&mut self, command: &ProtocolObject<dyn MTLCommandBuffer>) {
        if let Self::Timed {
            timestamps,
            lease,
            parts,
            ..
        } = self
        {
            if let Some(lease) = lease.take() {
                parts.push(timestamps.commit(lease, command));
            }
        }
    }
}

/// Ordered direct launches encoded into command buffers: one, and one more
/// after each host gate ([`DirectBatch::wait_event`]). The production form
/// uses one serial compute encoder, which orders every launch after the
/// previous one's writes. A batch ends in [`DirectBatch::commit`], or in
/// [`DirectBatch::abandon`] after a failure: either hands the caller every
/// command buffer it committed, which run on the caller's storage, as a
/// submission to fence.
pub struct DirectBatch {
    device: MetalDevice,
    /// The command buffers committed before the current one, in order.
    committed: Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    /// Whether anything is encoded in `command`.
    used: bool,
    encoding: Encoding,
}

fn complete(command: &ProtocolObject<dyn MTLCommandBuffer>) -> bool {
    matches!(
        crate::pooled(|| command.status()),
        MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
    )
}

fn command_buffer(
    device: &MetalDevice,
) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>, ExecutionError> {
    crate::pooled(|| device.queue().commandBuffer()).ok_or_else(|| {
        ExecutionError::SubmissionFailed("Metal could not create a native command buffer".into())
    })
}

fn serial_encoder(
    command: &ProtocolObject<dyn MTLCommandBuffer>,
) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>, ExecutionError> {
    crate::pooled(|| command.computeCommandEncoder()).ok_or_else(|| {
        ExecutionError::SubmissionFailed("Metal could not create a native compute encoder".into())
    })
}

/// A 64-bit value the host and the device's command buffers both advance
/// and wait on (`MTLSharedEvent`).
pub struct SharedEvent {
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
}

// A shared event is a thread-safe Metal object.
unsafe impl Send for SharedEvent {}
unsafe impl Sync for SharedEvent {}

impl SharedEvent {
    /// An event at 0.
    pub fn new(device: &MetalDevice) -> Result<Self, ExecutionError> {
        crate::pooled(|| device.handle().raw().newSharedEvent())
            .map(|event| Self { event })
            .ok_or_else(|| {
                ExecutionError::AllocationFailed("Metal could not create a shared event".into())
            })
    }

    fn as_event(&self) -> &ProtocolObject<dyn MTLEvent> {
        ProtocolObject::from_ref(&*self.event)
    }

    /// Set the event to `value` from the host.
    pub fn signal(&self, value: u64) {
        crate::pooled(|| self.event.setSignaledValue(value));
    }

    pub fn value(&self) -> u64 {
        crate::pooled(|| self.event.signaledValue())
    }

    /// Wait at most `timeout` for the event to reach `value`; whether it
    /// did.
    pub fn wait(&self, value: u64, timeout: std::time::Duration) -> bool {
        let milliseconds = u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX);
        crate::pooled(|| {
            self.event.signaledValue() >= value
                || self
                    .event
                    .waitUntilSignaledValue_timeoutMS(value, milliseconds)
        })
    }
}

impl DirectBatch {
    pub fn new(device: &MetalDevice) -> Result<Self, ExecutionError> {
        let command = command_buffer(device)?;
        let encoder = serial_encoder(&command)?;
        Ok(Self {
            device: device.clone(),
            committed: Vec::new(),
            command,
            used: false,
            encoding: Encoding::Serial(Some(encoder)),
        })
    }

    /// The host gate: the work encoded after this waits until `event`
    /// reaches `value`, and reads what the host wrote before raising it.
    /// The work encoded before is committed in its own command buffer, and
    /// the next begins with the wait (`encodeWaitForEvent`). Within one
    /// command buffer a load after the wait can return bytes an earlier
    /// load in the buffer read before the host rewrote them: chained
    /// prompts' n-gram table reads behind their gates saw stale rows in 1 of
    /// 16 runs on a CPU-loaded M4 Pro, and none in 48 with a command buffer
    /// begun before each read (2026-10-24). A wait before anything is
    /// encoded begins the current command buffer.
    ///
    /// The gate's opener must not be the thread encoding the batch: work
    /// behind the gate cannot complete before it opens. A timed batch whose
    /// every timestamp page is in flight behind a gate of its own not yet
    /// open fails [`Self::encode`] ("timestamp pool exhausted") rather than
    /// wait for a page that cannot free.
    pub fn wait_event(&mut self, event: &SharedEvent, value: u64) -> Result<(), ExecutionError> {
        self.end_serial();
        if self.used {
            self.cut()?;
        }
        crate::pooled(|| {
            self.command
                .encodeWaitForEvent_value(event.as_event(), value)
        });
        if let Encoding::Timed { parts, gates, .. } = &mut self.encoding {
            gates.push(Gate {
                event: event.event.clone(),
                value,
                first_part: parts.len(),
            });
        }
        Ok(())
    }

    /// Commit the current command buffer and continue in a new one, which
    /// the queue runs after it. Encoders end before this.
    fn cut(&mut self) -> Result<(), ExecutionError> {
        // Formed before the commit: a failure commits nothing.
        let next = command_buffer(&self.device)?;
        let previous = std::mem::replace(&mut self.command, next);
        self.encoding.committing(&previous);
        crate::pooled(|| previous.commit());
        self.committed.push(previous);
        Ok(())
    }

    /// `event` reaches `value` once the work encoded before it completes
    /// (`encodeSignalEvent`).
    pub fn signal_event(&mut self, event: &SharedEvent, value: u64) {
        self.end_serial();
        self.used = true;
        crate::pooled(|| {
            self.command
                .encodeSignalEvent_value(event.as_event(), value)
        });
    }

    /// End the open serial encoder: an event command is encoded between
    /// encoders.
    fn end_serial(&mut self) {
        if let Encoding::Serial(encoder) = &mut self.encoding {
            if let Some(encoder) = encoder.take() {
                crate::pooled(|| encoder.endEncoding());
            }
        }
    }

    /// A measurement batch: each launch gets its own encoder, whose GPU
    /// start and end are timestamped, and the batch commits a command
    /// buffer each time one fills its page of samples. Encoder boundaries
    /// add device time, so a timed batch attributes time to launches but
    /// does not measure the production step.
    pub fn timed(
        device: &MetalDevice,
        timestamps: &'static LaunchTimestamps,
    ) -> Result<Self, ExecutionError> {
        Ok(Self {
            device: device.clone(),
            committed: Vec::new(),
            command: command_buffer(device)?,
            used: false,
            encoding: Encoding::Timed {
                timestamps,
                lease: None,
                parts: Vec::new(),
                gates: Vec::new(),
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
        if let Encoding::Timed {
            lease: Some(lease), ..
        } = &self.encoding
        {
            if lease.full() {
                self.cut()?;
            }
        }
        self.used = true;
        let command = &self.command;
        let device = &self.device;
        match &mut self.encoding {
            Encoding::Serial(encoder) => {
                let encoder = match encoder {
                    Some(encoder) => encoder,
                    None => encoder.insert(serial_encoder(command)?),
                };
                crate::pooled(|| encode_launch(encoder, launch))
            }
            Encoding::Timed {
                timestamps,
                lease,
                parts,
                gates,
                launches,
                encoded,
                ..
            } => {
                let lease = match lease {
                    Some(lease) => lease,
                    None => lease.insert(timestamps.lease(device, || blocked_parts(gates, parts))?),
                };
                let page = lease.page.as_ref().expect("a lease holds its page until committed");
                let offset = lease.samples;
                crate::pooled(|| {
                    let pass = MTLComputePassDescriptor::computePassDescriptor();
                    pass.setDispatchType(MTLDispatchType::Serial);
                    // SAFETY: index 0 always exists; the pair lies inside
                    // the leased page, which is not full.
                    unsafe {
                        let attachment =
                            pass.sampleBufferAttachments().objectAtIndexedSubscript(0);
                        attachment.setSampleBuffer(Some(page));
                        attachment.setStartOfEncoderSampleIndex(offset);
                        attachment.setEndOfEncoderSampleIndex(offset + 1);
                    }
                    let encoder = command
                        .computeCommandEncoderWithDescriptor(&pass)
                        .ok_or_else(|| {
                            ExecutionError::SubmissionFailed(
                                "Metal could not create a timed compute encoder".into(),
                            )
                        })?;
                    encode_launch(&encoder, launch);
                    encoder.endEncoding();
                    Ok(())
                })?;
                lease.samples += 2;
                launches.push(Some(*encoded));
                *encoded += 1;
            }
        }
        Ok(())
    }

    /// Commit without waiting. The queue runs command buffers in commit
    /// order.
    pub fn commit(mut self) -> DirectSubmission {
        self.end_serial();
        self.encoding.committing(&self.command);
        crate::pooled(|| self.command.commit());
        let timing = match self.encoding {
            Encoding::Serial(_) => None,
            Encoding::Timed {
                timestamps,
                parts,
                launches,
                calibration,
                ..
            } => Some(LaunchTiming {
                timestamps,
                parts,
                launches,
                calibration,
            }),
        };
        let mut commands = self.committed;
        commands.push(self.command);
        DirectSubmission {
            device: self.device,
            commands,
            timing,
        }
    }

    /// End a batch that failed: the current command buffer is dropped
    /// uncommitted, and the command buffers committed before it, which run
    /// on, form the returned submission (`None` when there are none).
    pub fn abandon(mut self) -> Option<DirectSubmission> {
        self.end_serial();
        (!self.committed.is_empty()).then(|| DirectSubmission {
            device: self.device,
            commands: self.committed,
            timing: None,
        })
    }
}

fn encode_launch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    launch: &DirectLaunch<'_>,
) {
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
            launch.scalar_results.map(|(buffer, _)| buffer.raw()),
            launch.scalar_results.map_or(0, |(_, offset)| offset as usize),
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
    timestamps: &'static LaunchTimestamps,
    parts: Vec<PartTicks>,
    launches: Vec<Option<usize>>,
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

/// A committed direct batch: its command buffers, in commit order.
pub struct DirectSubmission {
    device: MetalDevice,
    commands: Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    timing: Option<LaunchTiming>,
}

// Waiting on and querying a committed command buffer is thread-safe.
unsafe impl Send for DirectSubmission {}
unsafe impl Sync for DirectSubmission {}

impl DirectSubmission {
    /// Whether every command buffer has completed.
    pub fn is_complete(&self) -> bool {
        self.commands.iter().all(|command| complete(command))
    }

    /// Wait for completion by polling the status every [`WAIT_POLL`].
    pub fn wait_complete(&self) {
        while !self.is_complete() {
            std::thread::sleep(WAIT_POLL);
        }
    }

    /// Wait and report the outcome: the first command buffer's failure.
    pub fn finish(&self) -> Result<(), ExecutionError> {
        self.wait_complete();
        crate::pooled(|| {
            self.commands
                .iter()
                .find_map(|command| command.error())
                .map_or(Ok(()), |error| {
                    Err(ExecutionError::SubmissionFailed(
                        error.localizedDescription().to_string(),
                    ))
                })
        })
    }

    /// Device execution time of the completed submission.
    pub fn device_seconds(&self) -> f64 {
        let (start, end) = self.device_interval();
        end - start
    }

    /// The completed submission's device execution interval on the
    /// [`host_seconds`] clock: from its first command buffer's start to its
    /// last's end, gate waits included, as within one command buffer.
    pub fn device_interval(&self) -> (f64, f64) {
        let (first, last) = (
            self.commands.first().expect("a submission has a command buffer"),
            self.commands.last().expect("a submission has a command buffer"),
        );
        crate::pooled(|| (first.GPUStartTime(), last.GPUEndTime()))
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
        let mut samples = Vec::new();
        for part in &timing.parts {
            samples.extend_from_slice(
                timing
                    .timestamps
                    .resolved(part)
                    .ok_or_else(|| failed("Metal did not resolve the timed batch's samples"))?,
            );
        }
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
