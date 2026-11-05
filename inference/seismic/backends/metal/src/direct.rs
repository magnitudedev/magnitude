//! Execution of explicitly authored top-level Metal implementations, whose
//! pipelines the Metal toolchain forms. This path deliberately consumes
//! source plus a closed ABI; it does not construct compiler kernels, plans,
//! schedules, or portfolios.

use crate::{Binding, MetalBuffer, MetalDevice};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCommonCounterSetTimestamp, MTLComputeCommandEncoder, MTLComputePassDescriptor,
    MTLComputePipelineState, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor,
    MTLCounterSamplingPoint, MTLCounterSet, MTLDevice, MTLDispatchType, MTLEvent,
    MTLResourceUsage, MTLSharedEvent, MTLSize, MTLStorageMode,
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
    pub buffers: &'a [(Binding<'a>, u64)],
    pub words: &'a [u8],
    /// Scalar-result slots, bound after the words; nothing is bound there
    /// for an entry without scalar results.
    pub scalar_results: Option<(&'a MetalBuffer, u64)>,
    /// The groups on each axis; with `grid`, the most the grid holds.
    pub threadgroups: [u64; 3],
    /// Three 32-bit group counts an earlier launch wrote, at this offset of
    /// this buffer: dispatched as they are when the launch runs.
    pub grid: Option<(Binding<'a>, u64)>,
    pub threads_per_threadgroup: [u64; 3],
    pub threadgroup_bytes: u64,
}

impl DirectLaunch<'_> {
    /// Whether the launch's grid is empty: it does no device work and
    /// encodes nothing.
    pub fn is_empty(&self) -> bool {
        self.threadgroups.contains(&0) || self.threads_per_threadgroup.contains(&0)
    }
}

pub use crate::timing::host_seconds;
use crate::timing::SubmissionTime;

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
    /// The production form: each group of launches ([`DirectBatch::group`])
    /// in a concurrent encoder of its own, except that consecutive groups of
    /// one launch share a serial encoder, which runs them in order. Each
    /// encoder runs after the one before ([`ordered`]). `open` is `None`
    /// after an event command or a group boundary, until the next launch
    /// opens an encoder of the `next` dispatch type.
    Grouped {
        open: Option<(
            Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
            MTLDispatchType,
        )>,
        next: MTLDispatchType,
        /// Whether each command's device interval is observed
        /// ([`DirectBatch::observed`]).
        observed: bool,
    },
    /// Every launch in its own encoder, with the encoder's start and end
    /// timestamps sampled at stage boundaries into the page its command
    /// buffer leased from `timestamps`; each encoder runs after the one
    /// before ([`ordered`]). A command buffer whose page is full is
    /// committed and the next leases another.
    Timed {
        timing: Box<LaunchTiming>,
        /// The current command buffer's page, from its first launch.
        lease: Option<Lease>,
        /// The batch's host gates, in order.
        gates: Vec<Gate>,
        encoded: usize,
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
    /// Commit `command`: a timed batch's page goes in flight with it, and
    /// an observed one's envelope starts immediately before the commit.
    fn commit(&mut self, command: Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> CommittedCommand {
        let observed = match self {
            Self::Grouped { observed, .. } => *observed,
            Self::Timed { timing, lease, .. } => {
                if let Some(lease) = lease.take() {
                    timing.parts.push(timing.timestamps.commit(lease, &command));
                }
                true
            }
        };
        let time = observed.then(SubmissionTime::before_commit);
        crate::pooled(|| command.commit());
        CommittedCommand { command, time }
    }

    /// The launch attribution of a timed batch.
    fn launch_timing(self) -> Option<Box<LaunchTiming>> {
        match self {
            Self::Grouped { .. } => None,
            Self::Timed { timing, .. } => Some(timing),
        }
    }
}

/// A committed command buffer, with the lower end of its submission
/// envelope where its batch observes device intervals.
struct CommittedCommand {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    time: Option<SubmissionTime>,
}

/// Direct launches encoded into command buffers: one, and one more after
/// each host gate ([`DirectBatch::wait_event`]). The production form runs
/// each group of launches concurrently and the groups one after another
/// ([`DirectBatch::group`]). A batch ends in [`DirectBatch::commit`], or in
/// [`DirectBatch::abandon`] after a failure: either hands the caller every
/// command buffer it committed, which run on the caller's storage, as a
/// submission to fence.
pub struct DirectBatch {
    device: MetalDevice,
    /// The command buffers committed before the current one, in order.
    committed: Vec<CommittedCommand>,
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    /// Whether anything is encoded in `command`.
    used: bool,
    encoding: Encoding,
}

/// Whether the command buffer ended: completed, or failed.
fn ended(command: &ProtocolObject<dyn MTLCommandBuffer>) -> bool {
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

/// Declare `encoder` a writer of the device's order buffer, which no launch
/// binds. The buffer is tracked (created by the device), and for a tracked
/// resource `useResource:usage:` protects against data hazards: the encoder
/// starts after every earlier encoder that declared it has completed, so
/// the queue's direct encoders run one after another whatever their
/// dispatches bind.
fn ordered(
    device: &MetalDevice,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
) {
    encoder.useResource_usage(
        ProtocolObject::from_ref(device.order()),
        MTLResourceUsage::Read | MTLResourceUsage::Write,
    );
}

fn grouped_encoder(
    device: &MetalDevice,
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    dispatch: MTLDispatchType,
) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>, ExecutionError> {
    crate::pooled(|| {
        let encoder = command.computeCommandEncoderWithDispatchType(dispatch)?;
        ordered(device, &encoder);
        Some(encoder)
    })
    .ok_or_else(|| {
        ExecutionError::SubmissionFailed("Metal could not create a native compute encoder".into())
    })
}

/// A 64-bit value the host or the device's command buffers advance
/// and the other waits on (`MTLSharedEvent`).
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
    /// A production batch. Each launch belongs to a group its caller begins
    /// ([`Self::group`]) before encoding it.
    pub fn new(device: &MetalDevice) -> Result<Self, ExecutionError> {
        Self::grouped(device, false)
    }

    /// Production encoding with command-buffer interval observation. Unlike
    /// launch attribution, this does not change encoder boundaries.
    pub fn observed(device: &MetalDevice) -> Result<Self, ExecutionError> {
        Self::grouped(device, true)
    }

    fn grouped(device: &MetalDevice, observed: bool) -> Result<Self, ExecutionError> {
        Ok(Self {
            device: device.clone(),
            committed: Vec::new(),
            command: command_buffer(device)?,
            used: false,
            encoding: Encoding::Grouped {
                open: None,
                // The first group sets it.
                next: MTLDispatchType::Serial,
                observed,
            },
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
        self.end_open();
        if self.used {
            self.cut()?;
        }
        crate::pooled(|| {
            self.command
                .encodeWaitForEvent_value(event.as_event(), value)
        });
        if let Encoding::Timed { timing, gates, .. } = &mut self.encoding {
            gates.push(Gate {
                event: event.event.clone(),
                value,
                first_part: timing.parts.len(),
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
        let committed = self.encoding.commit(previous);
        self.committed.push(committed);
        Ok(())
    }

    /// `event` reaches `value` once the work encoded before it completes
    /// (`encodeSignalEvent`).
    pub fn signal_event(&mut self, event: &SharedEvent, value: u64) {
        self.end_open();
        self.used = true;
        crate::pooled(|| {
            self.command
                .encodeSignalEvent_value(event.as_event(), value)
        });
    }

    /// End the open encoder: an event command is encoded between encoders.
    fn end_open(&mut self) {
        if let Encoding::Grouped { open, .. } = &mut self.encoding {
            if let Some((encoder, _)) = open.take() {
                crate::pooled(|| encoder.endEncoding());
            }
        }
    }

    /// The next `launches` launches (those that do device work) form a
    /// group: none depends on another, and every launch encoded before them
    /// completes before any of them starts. A group of several runs in a
    /// concurrent encoder; a group of one joins an open serial encoder, whose
    /// dispatches run in order, or opens one. Measured per dependent launch
    /// against a single serial encoder: an encoder per launch costs about
    /// 0.2 µs more at 64 KiB, a barrier in one concurrent encoder 0.5 µs
    /// (`tests/barrier.rs`). A timed batch's launches each have an encoder
    /// of their own already.
    pub fn group(&mut self, launches: usize) {
        let Encoding::Grouped { open, next, .. } = &mut self.encoding else {
            return;
        };
        let dispatch = if launches > 1 {
            MTLDispatchType::Concurrent
        } else {
            MTLDispatchType::Serial
        };
        if dispatch == MTLDispatchType::Serial
            && open
                .as_ref()
                .is_some_and(|(_, open)| *open == MTLDispatchType::Serial)
        {
            return;
        }
        *next = dispatch;
        self.end_open();
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
                timing: Box::new(LaunchTiming {
                    timestamps,
                    parts: Vec::new(),
                    launches: Vec::new(),
                    calibration: sample_timestamps(device),
                }),
                lease: None,
                gates: Vec::new(),
                encoded: 0,
            },
        })
    }

    /// Record a launch that does no device work (inactive, or an empty
    /// grid): nothing is encoded, and a timed batch keeps its place with no
    /// interval.
    pub fn skip(&mut self) {
        if let Encoding::Timed { timing, .. } = &mut self.encoding {
            timing.launches.push(None);
        }
    }

    pub fn encode(&mut self, launch: &DirectLaunch<'_>) -> Result<(), ExecutionError> {
        if launch.is_empty() {
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
            Encoding::Grouped { open, next, .. } => {
                let (encoder, _) = match open {
                    Some(open) => open,
                    None => open.insert((grouped_encoder(device, command, *next)?, *next)),
                };
                crate::pooled(|| encode_launch(encoder, launch))
            }
            Encoding::Timed { timing, lease, gates, encoded } => {
                let LaunchTiming { timestamps, parts, launches, .. } = &mut **timing;
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
                    ordered(device, &encoder);
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
        self.end_open();
        let last = self.encoding.commit(self.command);
        let mut commands = self.committed;
        commands.push(last);
        DirectSubmission {
            device: self.device,
            commands,
            timing: self.encoding.launch_timing(),
        }
    }

    /// End a batch that failed: the current command buffer is dropped
    /// uncommitted, and the command buffers committed before it, which run
    /// on, form the returned submission (`None` when there are none).
    pub fn abandon(mut self) -> Option<DirectSubmission> {
        self.end_open();
        (!self.committed.is_empty()).then(|| DirectSubmission {
            device: self.device,
            commands: self.committed,
            // Launch attribution covers the whole batch; the committed
            // prefix keeps only its commands' envelopes.
            timing: None,
        })
    }
}

fn encode_launch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    launch: &DirectLaunch<'_>,
) {
    encoder.setComputePipelineState(&launch.pipeline.state);
    for (index, (binding, offset)) in launch.buffers.iter().enumerate() {
        binding.with(|buffer| unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), *offset as usize, index) });
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
    let threads = MTLSize {
        width: launch.threads_per_threadgroup[0] as usize,
        height: launch.threads_per_threadgroup[1] as usize,
        depth: launch.threads_per_threadgroup[2] as usize,
    };
    match launch.grid {
        // SAFETY: the grid is three 32-bit words at a 4-byte-aligned offset
        // of one of the launch's own bindings, set above.
        Some((binding, offset)) => binding.with(|buffer| unsafe {
            encoder.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                buffer,
                offset as usize,
                threads,
            )
        }),
        None => encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: launch.threadgroups[0] as usize,
                height: launch.threadgroups[1] as usize,
                depth: launch.threadgroups[2] as usize,
            },
            threads,
        ),
    }
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
const WAIT_POLL: std::time::Duration = std::time::Duration::from_micros(20);

/// A committed direct batch: its command buffers, in commit order.
pub struct DirectSubmission {
    device: MetalDevice,
    commands: Vec<CommittedCommand>,
    /// A timed batch's launch attribution.
    timing: Option<Box<LaunchTiming>>,
}

// Waiting on and querying a committed command buffer is thread-safe.
unsafe impl Send for DirectSubmission {}
unsafe impl Sync for DirectSubmission {}

impl DirectSubmission {
    /// Whether the work completed, without waiting: `Ok(false)` while a
    /// command buffer runs, and the first one's failure once every one
    /// ended and one failed.
    pub fn completed(&self) -> Result<bool, ExecutionError> {
        if !self.commands.iter().all(|committed| ended(&committed.command)) {
            return Ok(false);
        }
        crate::pooled(|| {
            self.commands
                .iter()
                .find_map(|committed| committed.command.error())
                .map_or(Ok(true), |error| {
                    Err(ExecutionError::SubmissionFailed(
                        error.localizedDescription().to_string(),
                    ))
                })
        })
    }

    /// Wait for the work to end, polling its status every [`WAIT_POLL`],
    /// and report its outcome: the first command buffer's failure.
    pub fn finish(&self) -> Result<(), ExecutionError> {
        while !self.completed()? {
            std::thread::sleep(WAIT_POLL);
        }
        Ok(())
    }

    /// Device execution time when interval observation was requested with
    /// [`DirectBatch::observed`] or [`DirectBatch::timed`]. Invalid driver
    /// timestamps (including missing zero-work timestamps) are errors.
    pub fn device_seconds(&self) -> Result<f64, ExecutionError> {
        let (start, end) = self.device_interval()?;
        Ok(end - start)
    }

    /// The completed submission's device execution interval on the
    /// [`host_seconds`] clock: from its first command buffer's start to its
    /// last's end, gate waits included, as within one command buffer.
    /// Observation must have been requested before submission.
    pub fn device_interval(&self) -> Result<(f64, f64), ExecutionError> {
        self.finish()?;
        crate::pooled(|| {
            let mut span = None;
            for CommittedCommand { command, time } in &self.commands {
                let time = time.as_ref().ok_or_else(|| ExecutionError::SubmissionFailed(
                    "Metal submission interval observation was not requested".into()))?;
                let interval = time.interval(command.GPUStartTime(), command.GPUEndTime())
                    .map_err(ExecutionError::SubmissionFailed)?;
                span = Some(match span {
                    None => interval,
                    Some((start, _)) => (start, interval.1),
                });
            }
            Ok(span.expect("a submission has a command buffer"))
        })
    }

    /// For a timed batch, each `encode` call's device interval on the
    /// [`host_seconds`] clock, in encode order (`None` for an empty launch).
    /// `None` for a production batch. Call after completion.
    pub fn launch_intervals(&self) -> Result<Option<Vec<Option<(f64, f64)>>>, ExecutionError> {
        let Some(timing) = self.timing.as_deref() else {
            return Ok(None);
        };
        let failed = |detail: &str| ExecutionError::SubmissionFailed(detail.into());
        if !self.completed()? {
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
