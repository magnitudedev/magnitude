//! Execution of explicitly authored top-level Metal implementations, whose
//! pipelines the Metal toolchain forms. This path deliberately consumes
//! source plus a closed ABI; it does not construct compiler kernels, plans,
//! schedules, or portfolios.

use crate::{MetalBuffer, MetalDevice};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePassDescriptor,
    MTLComputePipelineState, MTLCounterDontSample, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor,
    MTLDevice, MTLDispatchType, MTLEvent,
    MTLResourceUsage, MTLSharedEvent, MTLSize, MTLStorageMode,
};
use seismic_compiler::direct_launch::{
    DirectLaunchContract, DirectLaunchRequest, LaunchGrid, ValidatedLaunch, WorkgroupContract,
};
use seismic_compiler::errors::ExecutionError;

/// Metal's `setBytes` limit, which bounds a direct entry's argument words.
pub const DIRECT_WORD_BYTES_LIMIT: usize = 4096;

/// Metal's buffer argument table size, which bounds a direct entry's
/// buffers, argument words and scalar-result slots together.
pub const DIRECT_BUFFER_SLOTS: usize = crate::device_info::ARGUMENT_TABLE_ENTRIES as usize;

/// One kernel function of an authored Metal library, compiled into a
/// pipeline.
#[derive(Clone)]
pub struct DirectPipeline {
    pub(crate) state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
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
    /// The groups on each axis; with `grid`, the most the grid holds.
    pub threadgroups: [u64; 3],
    /// Three 32-bit group counts an earlier launch wrote, at this offset of
    /// this buffer: dispatched as they are when the launch runs.
    pub grid: Option<(&'a MetalBuffer, u64)>,
    pub threads_per_threadgroup: [u64; 3],
    pub threadgroup_bytes: u64,
}

impl DirectLaunch<'_> {
    /// Whether the launch's grid or threadgroup is empty: it does no device
    /// work and encodes nothing. An indirect launch's grid comes from its GPU
    /// buffer, so only an empty threadgroup makes it empty on the host.
    pub fn is_empty(&self) -> bool {
        (self.grid.is_none() && self.threadgroups.contains(&0))
            || self.threads_per_threadgroup.contains(&0)
    }
}

pub use crate::timing::host_seconds;
use crate::timing::checked_interval;

/// A GPU timestamp correlated with the [`host_seconds`] clock: the host
/// clock read around `sampleTimestamps`, at the midpoint, is the time base
/// every interval uses, and the GPU sample lies within `half_gap` of it.
#[derive(Clone, Copy)]
pub(crate) struct TimestampPair {
    gpu: u64,
    host: f64,
    half_gap: f64,
}

pub(crate) fn sample_timestamps(device: &MetalDevice) -> TimestampPair {
    let mut cpu = 0u64;
    let mut gpu = 0u64;
    // `sampleTimestamps` pairs the GPU clock with a CPU clock of its own;
    // the host clock is read around it instead.
    let before = host_seconds();
    // SAFETY: both pointers are valid for one write.
    crate::pooled(|| unsafe {
        device.handle().raw().sampleTimestamps_gpuTimestamp(
            std::ptr::NonNull::from(&mut cpu),
            std::ptr::NonNull::from(&mut gpu),
        )
    });
    let after = host_seconds();
    TimestampPair { gpu, host: (before + after) / 2.0, half_gap: (after - before) / 2.0 }
}

/// The [`host_seconds`] clock of a GPU tick: linear between the pair taken
/// when a batch was formed and one taken once it completed.
pub(crate) struct HostClock {
    early: TimestampPair,
    seconds_per_tick: f64,
    /// How far a mapped tick between the two pairs may lie from the host
    /// time it was taken at: the larger of their half-gaps, since each
    /// pair's host time is within its half-gap of its GPU sample.
    pub(crate) uncertainty: f64,
}

impl HostClock {
    pub(crate) fn seconds(&self, tick: u64) -> f64 {
        self.early.host + (tick as f64 - self.early.gpu as f64) * self.seconds_per_tick
    }
}

/// The host clock of the GPU ticks taken since `early`, from a pair taken
/// now.
pub(crate) fn host_clock(device: &MetalDevice, early: TimestampPair) -> Result<HostClock, ExecutionError> {
    let late = sample_timestamps(device);
    if late.gpu <= early.gpu {
        return Err(ExecutionError::SubmissionFailed("Metal GPU timestamps did not advance".into()));
    }
    Ok(HostClock {
        early,
        seconds_per_tick: (late.host - early.host) / (late.gpu - early.gpu) as f64,
        uncertainty: early.half_gap.max(late.half_gap),
    })
}

type Page = Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>;

/// The process's GPU timestamp pages, shared by every timed and observed
/// batch: Metal sample buffers of [`PAGE_SAMPLES`] samples each. Metal
/// bounds how many a process may hold (32 on an M4 Pro, macOS 15.7.9,
/// whatever their size, another process's not counting:
/// `tests/sample_buffers.rs`), and only this
/// pool forms them, so it forms a page when a lease finds none free, until
/// Metal first refuses one: the pages formed then are the process's
/// allowance, and the pool's size. Each command buffer of a timed batch
/// leases one page for its launches' start and end samples; when the
/// command buffer completes, its completion handler resolves the page's
/// ticks into host memory and frees the page. Timed work of any length
/// therefore holds at most the pool. Observed batches share span pages
/// ([`SpanPair`]), taken from the pool as they are needed and returned once
/// every pair of a page drops.
pub struct LaunchTimestamps {
    /// The device whose command buffers sample the pages.
    registry_id: u64,
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
    /// The pages currently leased in pairs to observed batches; a slot
    /// returns its page to `free` once every pair of it drops.
    spans: Vec<Option<SpanPage>>,
}

impl Pool {
    fn active_span_pages(&self) -> usize {
        self.spans.iter().filter(|slot| slot.is_some()).count()
    }
}

/// A page whose samples observed batches lease in pairs: its free pairs.
struct SpanPage {
    page: Page,
    free: Vec<usize>,
}

/// An observed batch's pair of samples: its first encoder samples its
/// start into the first, and every encoder its end into the second, so
/// that once the batch completes they hold the start of its first encoder
/// and the end of its last (the queue's direct encoders run one after
/// another, [`ordered`]). Each command buffer the batch commits holds the
/// pair until it completes, the batch and its submission until dropped;
/// once none does, the pair returns to its page.
pub(crate) struct SpanPair {
    timestamps: &'static LaunchTimestamps,
    sample_buffer: Page,
    /// The page's index among the pool's span pages.
    page: usize,
    pair: usize,
}

// The sample buffer is written by the device and resolved after completion.
unsafe impl Send for SpanPair {}
unsafe impl Sync for SpanPair {}

impl SpanPair {
    fn start_index(&self) -> usize {
        2 * self.pair
    }

    fn end_index(&self) -> usize {
        2 * self.pair + 1
    }

    /// The span's start and end ticks, once its batch completed.
    pub(crate) fn resolve(&self) -> Result<(u64, u64), ExecutionError> {
        let failed = |detail: &str| ExecutionError::SubmissionFailed(detail.into());
        // SAFETY: the pair lies inside the page; the device wrote it before
        // its batch completed.
        let ticks = crate::pooled(|| unsafe {
            self.sample_buffer
                .resolveCounterRange(NSRange::new(self.start_index(), 2))
                .map(|data| data.to_vec())
        })
        .ok_or_else(|| failed("Metal did not resolve an observed batch's samples"))?;
        let tick = |bytes: &[u8]| u64::from_le_bytes(bytes.try_into().expect("eight bytes"));
        match (tick(&ticks[..8]), tick(&ticks[8..16])) {
            // Metal reports a sample it could not take as all ones.
            (u64::MAX, _) | (_, u64::MAX) => Err(failed("Metal dropped an observed batch's timestamp")),
            span => Ok(span),
        }
    }
}

impl SpanPair {
    /// `command`, about to be committed, writes the pair: it is not leased
    /// again before `command` completes, whoever else drops it first.
    pub(crate) fn hold_until_completed(
        self: &std::sync::Arc<Self>,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
    ) {
        let held = self.clone();
        let handler = block2::RcBlock::new(
            move |_: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                let _ = &held;
            },
        );
        // SAFETY: Metal copies the block, which owns the pair and releases
        // it once the handler has run.
        crate::pooled(|| unsafe { command.addCompletedHandler(block2::RcBlock::as_ptr(&handler)) });
    }
}

impl Drop for SpanPair {
    fn drop(&mut self) {
        let mut pool = self.timestamps.lock();
        let slot = pool.spans[self.page]
            .as_mut()
            .expect("live span pair references live span page");
        slot.free.push(self.pair);
        if slot.free.len() == PAGE_SAMPLES / 2 {
            let retired = pool.spans[self.page].take().expect("checked above");
            pool.free.push(retired.page);
        }
        drop(pool);
        self.timestamps.changed.notify_all();
    }
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
    /// The process's pool, formed on first use with its first page (every
    /// open device samples timestamps, `MetalDevice::open`): a device whose
    /// first page Metal refuses fails here, and so does a second Metal device, since
    /// the pages sample the first one's command buffers.
    pub fn shared(device: &MetalDevice) -> Result<&'static Self, ExecutionError> {
        let registry_id = device.handle().registry_id();
        let mut shared = PROCESS_POOL
            .lock()
            .expect("process timestamp pool lock is never poisoned");
        let pool = match *shared {
            Some(pool) => pool,
            None => {
                let first = crate::pooled(|| Self::form(device)).map_err(|refusal| {
                    ExecutionError::SubmissionFailed(format!(
                        "Metal refused the process's first {PAGE_SAMPLES}-sample timestamp buffer: {refusal}"
                    ))
                })?;
                let pool: &'static Self = Box::leak(Box::new(Self {
                    registry_id,
                    pool: std::sync::Mutex::new(Pool {
                        free: vec![first],
                        formed: 1,
                        full: false,
                        spans: Vec::new(),
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

    /// A new page, or Metal's refusal.
    fn form(device: &MetalDevice) -> Result<Page, String> {
        let descriptor = MTLCounterSampleBufferDescriptor::new();
        descriptor.setCounterSet(Some(device.timestamp_counters()));
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
                match crate::pooled(|| Self::form(device)) {
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
            let leasable = pool.formed - pool.active_span_pages();
            if leasable > 0 && blocked() == leasable {
                return Err(ExecutionError::SubmissionFailed(format!(
                    "timestamp pool exhausted: all {leasable} leasable pages are in flight behind \
                     a host gate of this batch that is not yet open",
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

    /// A free pair of a span page, taking a page from the pool for spans
    /// when every span page's pairs are leased. With neither, the wait ends
    /// when a pair frees (the command buffers that sampled into it
    /// completed) or a timed batch's page does. When `blocked`, the count of
    /// span pairs held by this batch itself, is every pair across every
    /// formed page, no pair can free while this thread waits: the lease is
    /// refused.
    pub(crate) fn span(
        &'static self,
        device: &MetalDevice,
        blocked: impl Fn() -> usize,
    ) -> Result<SpanPair, ExecutionError> {
        let mut pool = self.lock();
        loop {
            let free = pool
                .spans
                .iter_mut()
                .enumerate()
                .find_map(|(page, slot)| {
                    let spans = slot.as_mut()?;
                    spans.free.pop().map(|pair| (page, pair, spans.page.clone()))
                });
            if let Some((page, pair, sample_buffer)) = free {
                return Ok(SpanPair { timestamps: self, sample_buffer, page, pair });
            }
            let page = match pool.free.pop() {
                Some(page) => page,
                None if !pool.full => match crate::pooled(|| Self::form(device)) {
                    Ok(page) => {
                        pool.formed += 1;
                        page
                    }
                    Err(_) => {
                        pool.full = true;
                        continue;
                    }
                },
                None => {
                    let span_pages = pool.active_span_pages();
                    let total_pairs = span_pages * (PAGE_SAMPLES / 2);
                    if span_pages == pool.formed && blocked() == total_pairs {
                        return Err(ExecutionError::SubmissionFailed(format!(
                            "timestamp pool exhausted: all {total_pairs} span pairs across \
                             {span_pages} pages are held by this batch",
                        )));
                    }
                    pool = self
                        .changed
                        .wait(pool)
                        .expect("timestamp pool lock is never poisoned");
                    continue;
                }
            };
            let span_page = SpanPage { page, free: (0..PAGE_SAMPLES / 2).rev().collect() };
            if let Some(empty) = pool.spans.iter_mut().find(|slot| slot.is_none()) {
                *empty = Some(span_page);
            } else {
                pool.spans.push(Some(span_page));
            }
        }
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

/// The dispatch types of the compute encoders a command buffer holds so
/// far. An indirect dispatch is encoded only in a command buffer whose
/// encoders are all of its own encoder's type ([`Held::admits`]): the AGX
/// driver (M4 Pro, `AGXMetalG16X`, macOS 15.7) keeps an indirect dispatch's
/// threadgroup-optimization state in the command buffer's compute context
/// across encoders, by dispatch type, and an indirect dispatch in an
/// encoder of one type after an encoder of the other writes or reads
/// address 0 (`insertIndirectTGOptKernel` in a serial encoder after a
/// concurrent one, `executeIndirectKernelWithThreadgroupOptimization` in a
/// concurrent one after a serial one) and kills the process. Encoders of
/// one type throughout a command buffer, at any length or order of direct
/// and indirect dispatches, and a new command buffer, never did
/// (2026-11-17).
#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum Held {
    #[default]
    Nothing,
    Only(MTLDispatchType),
    Both,
}

impl Held {
    /// What the command buffer holds once it opens an encoder of `kind`.
    fn with(self, kind: MTLDispatchType) -> Self {
        match self {
            Self::Nothing => Self::Only(kind),
            Self::Only(held) if held == kind => self,
            Self::Only(_) | Self::Both => Self::Both,
        }
    }

    /// Whether an indirect dispatch may be encoded in an encoder of `kind`
    /// of this command buffer.
    fn admits(self, kind: MTLDispatchType) -> bool {
        self == Self::Nothing || self == Self::Only(kind)
    }
}

/// Encoder state shared by grouped production batches (observed or not): each
/// group of launches ([`DirectBatch::group`]) runs in a concurrent encoder of
/// its own, except that consecutive groups of one launch share a serial
/// encoder, which runs them in order. Each encoder runs after the one before
/// ([`ordered`]). `open` is `None` after an event command or a group boundary,
/// until the next launch opens an encoder of the `next` dispatch type. An
/// indirect launch the current command buffer does not admit ([`Held`]) begins
/// the next.
struct GroupedState {
    open: Option<(
        Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        MTLDispatchType,
    )>,
    next: MTLDispatchType,
    /// The current command buffer's encoders' types.
    held: Held,
}

impl GroupedState {
    fn new() -> Self {
        Self {
            open: None,
            // The first group sets it.
            next: MTLDispatchType::Serial,
            held: Held::Nothing,
        }
    }

    fn end_open(&mut self) {
        if let Some((encoder, _)) = self.open.take() {
            crate::pooled(|| encoder.endEncoding());
        }
    }

    fn group(&mut self, launches: usize) {
        let dispatch = if launches > 1 {
            MTLDispatchType::Concurrent
        } else {
            MTLDispatchType::Serial
        };
        if dispatch == MTLDispatchType::Serial
            && self
                .open
                .as_ref()
                .is_some_and(|(_, open)| *open == MTLDispatchType::Serial)
        {
            return;
        }
        self.next = dispatch;
        self.end_open();
    }

    fn encoder(
        &mut self,
        open: impl FnOnce(
            MTLDispatchType,
        ) -> Result<
            Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
            ExecutionError,
        >,
    ) -> Result<&ProtocolObject<dyn MTLComputeCommandEncoder>, ExecutionError> {
        if self.open.is_none() {
            let encoder = open(self.next)?;
            self.held = self.held.with(self.next);
            self.open = Some((encoder, self.next));
        }
        let (encoder, _) = self.open.as_ref().expect("grouped encoder is open");
        Ok(&**encoder)
    }
}

/// How a batch places its launches in compute encoders and observes them.
enum BatchMode {
    /// The production form: grouped encoders, no timestamp observation.
    Grouped(GroupedState),
    /// Production grouped encoding whose device interval is observed
    /// ([`DirectBatch::observed`]): each command buffer leases a [`SpanPair`]
    /// at its first encoder.
    Observed {
        grouped: GroupedState,
        timestamps: &'static LaunchTimestamps,
        /// The pair the current command buffer leased at its first encoder.
        span: Option<std::sync::Arc<SpanPair>>,
        recording: Recording<std::sync::Arc<SpanPair>>,
    },
    /// Every launch in its own encoder, with the encoder's start and end
    /// timestamps sampled at stage boundaries into the page its command
    /// buffer leased from `timing.timestamps`; each encoder runs after the
    /// one before ([`ordered`]). A command buffer whose page is full is
    /// committed and the next leases another.
    Timed {
        timing: Box<LaunchTiming>,
        /// The current command buffer's page, from its first launch.
        lease: Option<Lease>,
        /// The batch's host gates, in order.
        gates: Vec<Gate>,
        encoded: usize,
        recording: Recording<usize>,
    },
}

/// What an observed or timed batch records of each command buffer it
/// commits, read once the batch completes. `samples` is `None` when the
/// command buffer encoded no launch.
struct CommandRecord<S> {
    /// The host clock immediately before its commit.
    committed: f64,
    /// The values of the host gates it waited on before its first launch.
    gates: Vec<u64>,
    samples: Option<S>,
}

/// What an observed or timed batch records while it is encoded.
struct Recording<S> {
    /// The GPU and host clocks when the batch was formed.
    calibration: TimestampPair,
    committed: Vec<CommandRecord<S>>,
    /// The gates the current command buffer waits on.
    gates: Vec<u64>,
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

impl<S> Recording<S> {
    fn new(device: &MetalDevice) -> Self {
        Self { calibration: sample_timestamps(device), committed: Vec::new(), gates: Vec::new() }
    }

    fn record(&mut self, samples: Option<S>) {
        self.committed.push(CommandRecord {
            committed: host_seconds(),
            gates: std::mem::take(&mut self.gates),
            samples,
        });
    }
}

impl BatchMode {
    fn grouped_mut(&mut self) -> Option<&mut GroupedState> {
        match self {
            Self::Grouped(grouped) | Self::Observed { grouped, .. } => Some(grouped),
            Self::Timed { .. } => None,
        }
    }

    /// Commit `command`, recording it in an observed or timed batch: the
    /// samples of its launches go in flight with it.
    fn commit(
        &mut self,
        command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    ) -> Retained<ProtocolObject<dyn MTLCommandBuffer>> {
        match self {
            Self::Grouped(grouped) => {
                grouped.held = Held::Nothing;
            }
            Self::Observed { grouped, span, recording, .. } => {
                grouped.held = Held::Nothing;
                let samples = span.take().map(|pair| {
                    pair.hold_until_completed(&command);
                    pair
                });
                recording.record(samples);
            }
            Self::Timed { timing, lease, recording, .. } => {
                let samples = lease.take().map(|lease| {
                    timing.parts.push(timing.timestamps.commit(lease, &command));
                    timing.parts.len() - 1
                });
                recording.record(samples);
            }
        }
        crate::pooled(|| command.commit());
        command
    }

    /// Consume the mode into the submission's observation, if any.
    fn into_observed(self) -> Option<Observed> {
        match self {
            Self::Grouped(_) => None,
            Self::Observed { recording, .. } => Some(Observed::Span {
                calibration: recording.calibration,
                commands: recording.committed,
            }),
            Self::Timed { timing, recording, .. } => Some(Observed::Timed {
                calibration: recording.calibration,
                commands: recording.committed,
                timing,
            }),
        }
    }
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
    committed: Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    /// The open uncommitted command buffer (`None` only after `commit` takes it).
    command: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    /// Whether anything is encoded in `command`.
    used: bool,
    mode: BatchMode,
}

impl Drop for DirectBatch {
    /// Close any open encoder on the uncommitted command buffer and wait for
    /// any already-committed prefix to complete so an unwind or early drop
    /// never leaves an open encoder or frees caller storage while in flight.
    fn drop(&mut self) {
        self.end_open();
        for command in self.committed.drain(..) {
            crate::pooled(|| command.waitUntilCompleted());
        }
    }
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
    device.command_buffer().ok_or_else(|| {
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

/// A compute pass of `dispatch` type whose encoder samples its end into
/// `pair`, and its start too when `start`: the pair then holds the encoder's
/// interval, or, over encoders that run one after another and each sample
/// their end, the first one's start and the last one's end.
pub(crate) fn sampled_pass(
    dispatch: MTLDispatchType,
    pair: &SpanPair,
    start: bool,
) -> Retained<MTLComputePassDescriptor> {
    let pass = MTLComputePassDescriptor::computePassDescriptor();
    pass.setDispatchType(dispatch);
    // SAFETY: index 0 always exists; both samples lie inside the pair's
    // page.
    unsafe {
        let attachment = pass.sampleBufferAttachments().objectAtIndexedSubscript(0);
        attachment.setSampleBuffer(Some(&pair.sample_buffer));
        attachment.setStartOfEncoderSampleIndex(if start { pair.start_index() } else { MTLCounterDontSample });
        attachment.setEndOfEncoderSampleIndex(pair.end_index());
    }
    pass
}

fn open_compute_encoder(
    device: &MetalDevice,
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    kind: crate::device::EncoderKind<'_>,
) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>, ExecutionError> {
    crate::pooled(|| {
        let encoder = device.compute_encoder(command, kind)?;
        ordered(device, &encoder);
        Some(encoder)
    })
    .ok_or_else(|| {
        ExecutionError::SubmissionFailed("Metal could not create a native compute encoder".into())
    })
}

/// A grouped encoder of `dispatch` type for an observed batch. Its command
/// buffer leases a span pair at its first encoder, which samples its start;
/// each of its encoders samples its end.
fn observed_encoder(
    device: &MetalDevice,
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    dispatch: MTLDispatchType,
    timestamps: &'static LaunchTimestamps,
    span: &mut Option<std::sync::Arc<SpanPair>>,
    held_spans: usize,
) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>, ExecutionError> {
    let (pass, leased) = match span.as_ref() {
        Some(pair) => (crate::pooled(|| sampled_pass(dispatch, pair, false)), None),
        None => {
            let pair = std::sync::Arc::new(timestamps.span(device, || held_spans)?);
            let pass = crate::pooled(|| sampled_pass(dispatch, &pair, true));
            (pass, Some(pair))
        }
    };
    let encoder = open_compute_encoder(device, command, crate::device::EncoderKind::Pass(&pass))?;
    if let Some(pair) = leased {
        *span = Some(pair);
    }
    Ok(encoder)
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
        Self::with_mode(device, BatchMode::Grouped(GroupedState::new()))
    }

    /// Production encoding whose device interval is observed: from its
    /// first launch's start to its last launch's end, sampled on the GPU
    /// (each command buffer's first encoder's start and every encoder's end,
    /// into a pair of the process's timestamp pages, [`SpanPair`]). Unlike
    /// launch attribution, this does not change encoder boundaries.
    ///
    /// Not the command buffers' `GPUStartTime`: in a streamed prefill on an
    /// M4 Pro (macOS 15.7.9, 2026-11-17) the second of an observed batch's
    /// four command buffers reported a start 0.2 s before the batch's first
    /// commit, while the first, ahead of it on the queue, started after it.
    pub fn observed(device: &MetalDevice) -> Result<Self, ExecutionError> {
        let timestamps = LaunchTimestamps::shared(device)?;
        Self::with_mode(
            device,
            BatchMode::Observed {
                grouped: GroupedState::new(),
                timestamps,
                span: None,
                recording: Recording::new(device),
            },
        )
    }

    fn with_mode(device: &MetalDevice, mode: BatchMode) -> Result<Self, ExecutionError> {
        Ok(Self {
            device: device.clone(),
            committed: Vec::new(),
            command: Some(command_buffer(device)?),
            used: false,
            mode,
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
                .as_ref()
                .expect("DirectBatch holds its command buffer until consumed")
                .encodeWaitForEvent_value(event.as_event(), value)
        });
        match &mut self.mode {
            BatchMode::Grouped(_) => {}
            BatchMode::Observed { recording, .. } => {
                recording.gates.push(value);
            }
            BatchMode::Timed { timing, gates, recording, .. } => {
                recording.gates.push(value);
                gates.push(Gate {
                    event: event.event.clone(),
                    value,
                    first_part: timing.parts.len(),
                });
            }
        }
        Ok(())
    }

    /// Commit the current command buffer and continue in a new one, which
    /// the queue runs after it. Encoders end before this.
    fn cut(&mut self) -> Result<(), ExecutionError> {
        // Formed before the commit: a failure commits nothing.
        let next = command_buffer(&self.device)?;
        let previous = self
            .command
            .replace(next)
            .expect("DirectBatch holds its command buffer until consumed");
        let committed = self.mode.commit(previous);
        self.committed.push(committed);
        self.used = false;
        Ok(())
    }

    /// `event` reaches `value` once the work encoded before it completes
    /// (`encodeSignalEvent`).
    pub fn signal_event(&mut self, event: &SharedEvent, value: u64) {
        self.end_open();
        self.used = true;
        crate::pooled(|| {
            self.command
                .as_ref()
                .expect("DirectBatch holds its command buffer until consumed")
                .encodeSignalEvent_value(event.as_event(), value)
        });
    }

    /// End the open encoder: an event command is encoded between encoders.
    fn end_open(&mut self) {
        if let Some(grouped) = self.mode.grouped_mut() {
            grouped.end_open();
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
        if let Some(grouped) = self.mode.grouped_mut() {
            grouped.group(launches);
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
        Self::with_mode(
            device,
            BatchMode::Timed {
                timing: Box::new(LaunchTiming {
                    timestamps,
                    parts: Vec::new(),
                    launches: Vec::new(),
                }),
                lease: None,
                gates: Vec::new(),
                encoded: 0,
                recording: Recording::new(device),
            },
        )
    }

    /// Record a launch that does no device work (inactive, or an empty
    /// grid): nothing is encoded, and a timed batch keeps its place with no
    /// interval.
    pub fn skip(&mut self) {
        if let BatchMode::Timed { timing, .. } = &mut self.mode {
            timing.launches.push(None);
        }
    }

    pub fn encode(&mut self, launch: &DirectLaunch<'_>) -> Result<(), ExecutionError> {
        if launch.is_empty() {
            self.skip();
            return Ok(());
        }
        let dynamic_shared_bytes =
            crate::checked_threadgroup_memory_bytes(launch.threadgroup_bytes).ok_or_else(|| {
                ExecutionError::SubmissionFailed(
                    "native threadgroup memory size overflowed".into(),
                )
            })?;
        let raw = self.device.handle().raw();
        let max_size = raw.maxThreadsPerThreadgroup();
        let contract = DirectLaunchContract {
            max_buffer_slots: Some(DIRECT_BUFFER_SLOTS),
            max_grid: [crate::device_info::MSL_GRID_INDEX_MAX; 3],
            workgroup: WorkgroupContract::Dynamic {
                max_size: [
                    max_size.width as u64,
                    max_size.height as u64,
                    max_size.depth as u64,
                ],
                max_threads: launch.pipeline.max_threads_per_threadgroup(),
                max_shared_bytes: raw.maxThreadgroupMemoryLength() as u64,
            },
            max_param_bytes: Some(DIRECT_WORD_BYTES_LIMIT as u64),
        };
        let buffer_slots = launch.buffers.len().checked_add(2).ok_or_else(|| {
            ExecutionError::SubmissionFailed("native buffer slot count overflowed".into())
        })?;
        let request = DirectLaunchRequest {
            buffer_slots,
            grid: match launch.grid {
                Some(_) => LaunchGrid::Indirect,
                None => LaunchGrid::Direct(launch.threadgroups),
            },
            group: Some(launch.threads_per_threadgroup),
            static_shared_bytes: launch.pipeline.static_threadgroup_bytes(),
            dynamic_shared_bytes,
            param_bytes: launch.words.len() as u64,
        };
        if matches!(contract.validate(&request)?, ValidatedLaunch::Empty) {
            self.skip();
            return Ok(());
        }
        match &self.mode {
            BatchMode::Timed { lease: Some(lease), .. } if lease.full() => {
                self.cut()?;
            }
            // An indirect launch in a command buffer that holds an encoder of
            // the other type than its own begins the next command buffer.
            BatchMode::Grouped(grouped) | BatchMode::Observed { grouped, .. }
                if launch.grid.is_some() && !grouped.held.admits(grouped.next) =>
            {
                self.end_open();
                self.cut()?;
            }
            _ => {}
        }
        let command = self
            .command
            .as_ref()
            .expect("DirectBatch holds its command buffer until consumed");
        let device = &self.device;
        match &mut self.mode {
            BatchMode::Grouped(grouped) => {
                let encoder = grouped.encoder(|dispatch| {
                    open_compute_encoder(
                        device,
                        command,
                        crate::device::EncoderKind::Dispatch(dispatch),
                    )
                })?;
                crate::pooled(|| encode_launch(encoder, launch));
            }
            BatchMode::Observed { grouped, timestamps, span, recording } => {
                let encoder = grouped.encoder(|dispatch| {
                    let held_spans = recording
                        .committed
                        .iter()
                        .filter(|record| record.samples.is_some())
                        .count();
                    observed_encoder(device, command, dispatch, *timestamps, span, held_spans)
                })?;
                crate::pooled(|| encode_launch(encoder, launch));
            }
            BatchMode::Timed { timing, lease, gates, encoded, .. } => {
                let LaunchTiming { timestamps, parts, launches, .. } = &mut **timing;
                let leased = match lease {
                    Some(_) => None,
                    None => Some(timestamps.lease(device, || blocked_parts(gates, parts))?),
                };
                let active = leased.as_ref().or(lease.as_ref()).expect("a timed launch has a lease");
                let page = active.page.as_ref().expect("a lease holds its page until committed");
                let offset = active.samples;
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
                    let encoder = device
                        .compute_encoder(&command, crate::device::EncoderKind::Pass(&pass))
                        .ok_or_else(|| {
                            ExecutionError::SubmissionFailed(
                                "Metal could not create a timed compute encoder".into(),
                            )
                        })?;
                    ordered(device, &encoder);
                    encode_launch(&encoder, launch);
                    encoder.endEncoding();
                    Ok::<_, ExecutionError>(())
                })?;
                let lease = match leased {
                    Some(new_lease) => lease.insert(new_lease),
                    None => lease.as_mut().expect("a timed launch has a lease"),
                };
                lease.samples += 2;
                launches.push(Some(*encoded));
                *encoded += 1;
            }
        }
        self.used = true;
        Ok(())
    }

    /// Commit without waiting. The queue runs command buffers in commit
    /// order.
    pub fn commit(mut self) -> DirectSubmission {
        self.end_open();
        let command = self
            .command
            .take()
            .expect("DirectBatch holds its command buffer until consumed");
        let last = self.mode.commit(command);
        let mut commands = std::mem::take(&mut self.committed);
        commands.push(last);
        let mode = std::mem::replace(&mut self.mode, BatchMode::Grouped(GroupedState::new()));
        DirectSubmission {
            device: self.device.clone(),
            commands,
            observed: mode.into_observed(),
        }
    }

    /// End a batch that failed: the current command buffer is dropped
    /// uncommitted, and the command buffers committed before it, which run
    /// on, form the returned submission (`None` when there are none).
    pub fn abandon(mut self) -> Option<DirectSubmission> {
        self.end_open();
        let commands = std::mem::take(&mut self.committed);
        // The batch's observation covers the whole batch: its committed
        // prefix, whose outcome is the failure, observes nothing.
        (!commands.is_empty()).then(|| DirectSubmission {
            device: self.device.clone(),
            commands,
            observed: None,
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
    unsafe { crate::set_threadgroup_memory_length(encoder, launch.threadgroup_bytes, 0) };
    let threads = MTLSize {
        width: launch.threads_per_threadgroup[0] as usize,
        height: launch.threads_per_threadgroup[1] as usize,
        depth: launch.threads_per_threadgroup[2] as usize,
    };
    match launch.grid {
        // SAFETY: the grid is three 32-bit words at a 4-byte-aligned offset
        // of one of the launch's own bindings, set above.
        Some((buffer, offset)) => unsafe {
            encoder.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                buffer.raw(),
                offset as usize,
                threads,
            )
        },
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
    commands: Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    /// What an observed or timed batch recorded.
    observed: Option<Observed>,
}

/// What an observed or timed batch recorded of its command buffers, one
/// record each, in commit order: GPU timestamp samples, the one source of
/// every Metal device interval.
enum Observed {
    /// An observed production batch ([`DirectBatch::observed`]): each command
    /// buffer's leased [`SpanPair`], when it encoded a launch.
    Span {
        /// The GPU and host clocks when the batch was formed.
        calibration: TimestampPair,
        commands: Vec<CommandRecord<std::sync::Arc<SpanPair>>>,
    },
    /// A timed measurement batch ([`DirectBatch::timed`]): each command
    /// buffer's part index into `timing.parts`, when it encoded a launch.
    Timed {
        /// The GPU and host clocks when the batch was formed.
        calibration: TimestampPair,
        commands: Vec<CommandRecord<usize>>,
        timing: Box<LaunchTiming>,
    },
}

fn resolve_interval<S>(
    commands: &[CommandRecord<S>],
    host: &HostClock,
    mut resolve: impl FnMut(&S) -> Result<(u64, u64), ExecutionError>,
) -> Result<(f64, f64), ExecutionError> {
    let failed = |detail: String| ExecutionError::SubmissionFailed(detail);
    let sampled = commands
        .iter()
        .map(|command| {
            let Some(samples) = &command.samples else {
                return Ok(None);
            };
            let (start, end) = resolve(samples)?;
            Ok(Some((host.seconds(start), host.seconds(end))))
        })
        .collect::<Result<Vec<_>, ExecutionError>>()?;
    let mut launched = sampled.iter().flatten();
    let first = launched.next().ok_or_else(|| {
        failed("an observed Metal batch that encoded no launch has no device interval".into())
    })?;
    let end = launched.last().unwrap_or(first).1;
    checked_interval(
        commands[0].committed,
        first.0,
        end,
        host_seconds(),
        host.uncertainty,
    )
    .map_err(|error| {
        let count = commands.len();
        let details = commands
            .iter()
            .zip(&sampled)
            .enumerate()
            .map(|(index, (command, sampled))| {
                let sampled = match sampled {
                    Some((start, end)) => format!("sampled [{start}, {end}]"),
                    None => "no launch".into(),
                };
                format!(
                    "command buffer {} of {count}: committed at {}, after gates {:?}, {sampled}",
                    index + 1,
                    command.committed,
                    command.gates
                )
            })
            .collect::<Vec<_>>();
        failed(format!("{error}; {}", details.join("; ")))
    })
}

// Waiting on and querying a committed command buffer is thread-safe.
unsafe impl Send for DirectSubmission {}
unsafe impl Sync for DirectSubmission {}

impl Drop for DirectSubmission {
    /// Wait for any uncompleted command buffers to finish before dropping so
    /// backing `MetalBuffer`s, anonymous `Mapping`s, and timestamp pages are
    /// never freed while GPU work referencing them is still in flight.
    fn drop(&mut self) {
        if !self.commands.is_empty() {
            crate::pooled(|| {
                for command in self.commands.drain(..) {
                    if !matches!(
                        command.status(),
                        MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
                    ) {
                        command.waitUntilCompleted();
                    }
                }
            });
        }
    }
}

impl DirectSubmission {
    /// Whether the work completed, without waiting: `Ok(false)` while a
    /// command buffer runs, and the first one's failure once every one
    /// ended and one failed.
    pub fn completed(&self) -> Result<bool, ExecutionError> {
        if !self.commands.iter().all(|command| ended(command)) {
            return Ok(false);
        }
        crate::pooled(|| {
            self.commands
                .iter()
                .find_map(|command| command.error())
                .map_or(Ok(true), |error| {
                    Err(self
                        .device
                        .handle()
                        .command_buffer_error(&error, ExecutionError::SubmissionFailed))
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
    /// [`DirectBatch::observed`] or [`DirectBatch::timed`]. Missing or
    /// invalid timestamps (a batch that encoded no launch has none) are
    /// errors.
    pub fn device_seconds(&self) -> Result<f64, ExecutionError> {
        let (start, end) = self.device_interval()?;
        Ok(end - start)
    }

    /// The completed submission's device execution interval on the
    /// [`host_seconds`] clock, from GPU timestamp samples: from its first
    /// launch's start to its last launch's end, gate waits between them
    /// included. Observation must have been requested before submission.
    /// An interval outside the submission's envelope (from just before its
    /// first commit to now) is an error, not device time, and names each
    /// command buffer's commit, gates and samples.
    pub fn device_interval(&self) -> Result<(f64, f64), ExecutionError> {
        self.finish()?;
        let failed = |detail: String| ExecutionError::SubmissionFailed(detail);
        let observed = self
            .observed
            .as_ref()
            .ok_or_else(|| failed("Metal submission interval observation was not requested".into()))?;
        match observed {
            Observed::Span { calibration, commands } => {
                let host = host_clock(&self.device, *calibration)?;
                resolve_interval(commands, &host, |pair| pair.resolve())
            }
            Observed::Timed { calibration, commands, timing } => {
                let host = host_clock(&self.device, *calibration)?;
                resolve_interval(commands, &host, |&part| {
                    let ticks = timing
                        .timestamps
                        .resolved(&timing.parts[part])
                        .ok_or_else(|| {
                            failed("Metal did not resolve the timed batch's samples".into())
                        })?;
                    match (ticks.first(), ticks.last()) {
                        (Some(&start), Some(&end)) if start != u64::MAX && end != u64::MAX => {
                            Ok((start, end))
                        }
                        _ => Err(failed("Metal dropped a timed launch's timestamp".into())),
                    }
                })
            }
        }
    }

    /// For a timed batch, each `encode` call's device interval on the
    /// [`host_seconds`] clock, in encode order (`None` for an empty launch).
    /// `None` for any other batch. Call after completion.
    pub fn launch_intervals(&self) -> Result<Option<Vec<Option<(f64, f64)>>>, ExecutionError> {
        let Some(Observed::Timed { calibration, timing, .. }) = &self.observed else {
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
        let host = host_clock(&self.device, *calibration)?;
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
                    Ok((host.seconds(start), host.seconds(end)))
                })
                .transpose()
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Span pages return to the free pool once every leased pair of a page
    /// drops, so leasing and retiring spans past pool capacity never exhausts
    /// timed `lease()` calls; and when a batch itself holds every span pair
    /// across every formed page, `span()` refuses with an error instead of
    /// hanging forever on `changed`.
    #[test]
    fn span_pages_retire_to_pool_and_exhaustion_refuses_instead_of_hanging() {
        let device = MetalDevice::open(
            crate::DeviceHandle::system_default().unwrap(),
            std::sync::Arc::new(seismic_memory::entry::Tally::unbounded(
                seismic_topology::StoreIdentity::HostRam,
            )),
        )
        .unwrap();
        let page = crate::pooled(|| LaunchTimestamps::form(&device)).unwrap();
        let timestamps: &'static LaunchTimestamps = Box::leak(Box::new(LaunchTimestamps {
            registry_id: device.handle().registry_id(),
            pool: std::sync::Mutex::new(Pool {
                free: vec![page],
                formed: 1,
                full: true,
                spans: Vec::new(),
            }),
            changed: std::sync::Condvar::new(),
        }));
        let pairs_per_page = PAGE_SAMPLES / 2;

        // Lease and retire more than pool capacity across two full rounds.
        for _ in 0..2 {
            let mut held = Vec::with_capacity(pairs_per_page);
            for count in 0..pairs_per_page {
                held.push(timestamps.span(&device, || count).unwrap());
            }
            // All pairs across the only formed page are held by this batch:
            // requesting one more must refuse rather than hang.
            let refused = timestamps.span(&device, || held.len());
            assert!(
                matches!(refused, Err(ExecutionError::SubmissionFailed(ref msg)) if msg.contains("timestamp pool exhausted")),
                "expected refusal when all span pairs are held, got {}",
                refused.err().map_or_else(|| "Ok(SpanPair)".into(), |e| e.to_string()),
            );
            drop(held);

            // Once all pairs drop, the span page returns to `free` and a
            // timed `lease()` succeeds (no spurious "behind a host gate").
            let lease = timestamps.lease(&device, || 0).unwrap();
            drop(lease);
        }
    }

    /// Dropping a `DirectBatch` without `commit()` or `abandon()` ends any
    /// open compute encoder and waits for any already-committed prefix to
    /// finish (verified via a signaled `SharedEvent` in the committed prefix).
    #[test]
    fn dropping_direct_batch_ends_open_encoder_and_waits_committed_prefix() {
        let Ok(handle) = crate::DeviceHandle::system_default() else {
            return;
        };
        let device = MetalDevice::open(handle, crate::test_support::store()).unwrap();
        let done = SharedEvent::new(&device).unwrap();
        let gate = SharedEvent::new(&device).unwrap();
        {
            let mut batch = DirectBatch::new(&device).unwrap();
            // Signal `done = 1` in the first command buffer, then cut via
            // `wait_event` so that first command buffer is committed while a
            // second remains uncommitted.
            batch.signal_event(&done, 1);
            batch.wait_event(&gate, 1).unwrap();
            assert_eq!(batch.committed.len(), 1);
            // Open an encoder on the uncommitted second command buffer so Drop
            // must call `endEncoding()` before releasing it.
            let command = batch.command.as_ref().unwrap();
            let encoder = open_compute_encoder(
                &device,
                command,
                crate::device::EncoderKind::Dispatch(MTLDispatchType::Serial),
            )
            .unwrap();
            if let BatchMode::Grouped(grouped) = &mut batch.mode {
                grouped.open = Some((encoder, MTLDispatchType::Serial));
            }
            // Drop `batch` without commit() or abandon(): must end the open
            // encoder and wait for the committed prefix (`done == 1`).
        }
        assert_eq!(
            done.value(),
            1,
            "DirectBatch::drop must wait for already-committed command buffers"
        );
    }
}
