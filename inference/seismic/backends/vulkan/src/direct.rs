//! Execution of formed modules (§8.3–§8.7), the shape of CUDA's `direct`:
//! batches of launches recorded into one primary command buffer and
//! submitted to the device's queue, graphs recorded once into a secondary
//! command buffer and replayed, and their completion and timing.
//!
//! Every launch reads its arguments from an argument block in upload
//! storage: its buffer addresses, the scalar-result address, then its words.
//! The launch's only push constant is the block's address.

use crate::device::{
    barrier, call, Device, Handed, Recording, Retired, RetiredCommand, ARGUMENT_CHUNK_BYTES,
};
use crate::formation::DirectModule;
use crate::memory::{Buffer, Range};
use ash::vk::{self, Handle};
use seismic_compiler::errors::ExecutionError;
use seismic_native_target::{GateOpening, ParkAnswer};
use seismic_compiler::graph_cost::GraphCensus;
use std::sync::atomic::{AtomicU64, Ordering};

/// One launch as issued. Its workgroup size and group memory are the
/// pipeline's; only the group counts vary per launch.
pub struct DirectLaunch<'a> {
    pub module: &'a DirectModule,
    pub function: usize,
    /// Buffer arguments in ABI order, as `(buffer, byte offset)`.
    pub buffers: &'a [(&'a Buffer, u64)],
    /// The argument words, little-endian.
    pub words: &'a [u8],
    /// Scalar-result slots, addressed after the buffers; address 0 for an
    /// entry without scalar results.
    pub scalar_results: Option<(&'a Buffer, u64)>,
    /// The group counts; with `grid`, the most the grid holds.
    pub groups: [u64; 3],
    /// The three 32-bit group counts an earlier launch wrote, as `(buffer,
    /// byte offset)`, dispatched indirectly.
    pub grid: Option<(&'a Buffer, u64)>,
}

/// What fixes a launch's device work apart from the values it reads: its
/// pipeline and group counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LaunchKind {
    pipeline: u64,
    groups: [u64; 3],
}

impl DirectLaunch<'_> {
    pub fn kind(&self) -> LaunchKind {
        LaunchKind {
            pipeline: self.module.pipeline_handle(self.function).as_raw(),
            groups: self.groups,
        }
    }

    /// Whether the launch does no device work (an empty grid): it is
    /// recorded as nothing.
    pub fn is_empty(&self) -> bool {
        self.groups.contains(&0)
    }

    /// The launch's argument block: each buffer's address, the
    /// scalar-result address, then the words.
    pub fn argument_block(&self) -> Vec<u8> {
        let mut block = Vec::with_capacity((self.buffers.len() + 1) * 8 + self.words.len());
        for (buffer, offset) in self.buffers {
            block.extend_from_slice(&(buffer.address() + offset).to_le_bytes());
        }
        let scalar_results = self
            .scalar_results
            .map_or(0, |(buffer, offset)| buffer.address() + offset);
        block.extend_from_slice(&scalar_results.to_le_bytes());
        block.extend_from_slice(self.words);
        block
    }
}

/// Where a recorded launch's argument block lies in its graph's chunks, and
/// the bytes it holds now.
struct Block {
    chunk: usize,
    offset: u64,
    bytes: Vec<u8>,
}

/// Argument blocks written into upload chunks for one command buffer.
struct Arguments {
    chunks: Vec<Range>,
    /// Bytes used in the last chunk.
    used: u64,
}

impl Arguments {
    fn new() -> Self {
        Self {
            chunks: Vec::new(),
            used: ARGUMENT_CHUNK_BYTES,
        }
    }

    /// Write the block of `launch` and return its device address.
    fn write(
        &mut self,
        device: &Device,
        recording: &mut Recording,
        launch: &DirectLaunch<'_>,
    ) -> Result<u64, ExecutionError> {
        self.write_block(device, recording, launch).map(|(address, _)| address)
    }

    /// Write the block of `launch`; its device address, and where it lies
    /// with the bytes it holds.
    fn write_block(
        &mut self,
        device: &Device,
        recording: &mut Recording,
        launch: &DirectLaunch<'_>,
    ) -> Result<(u64, Block), ExecutionError> {
        let block = launch.argument_block();
        let bytes = argument_block_bytes(launch.buffers.len() as u64, launch.words.len() as u64);
        if bytes > ARGUMENT_CHUNK_BYTES {
            return Err(ExecutionError::SubmissionFailed(format!(
                "a launch's argument block needs {bytes} bytes; a chunk holds {ARGUMENT_CHUNK_BYTES}"
            )));
        }
        if self.used + bytes > ARGUMENT_CHUNK_BYTES {
            self.chunks.push(device.argument_chunk(recording)?);
            self.used = 0;
        }
        let start = self.used;
        let chunk = self.chunks.last().expect("a chunk was taken above");
        chunk.write_mapped(start, &block);
        self.used = start + bytes;
        let address = chunk.address() + start;
        Ok((address, Block { chunk: self.chunks.len() - 1, offset: start, bytes: block }))
    }
}

/// The bytes `launch`'s argument block takes in its chunk, as Vulkan's
/// layout counts them ([`seismic_compiler::graph_cost::argument_bytes`]).
pub(crate) const fn argument_block_bytes(buffers: u64, words: u64) -> u64 {
    seismic_compiler::graph_cost::argument_bytes(seismic_lang::registry::BackendName::Vulkan, buffers, words)
}

/// Record one launch: its barrier after the previous launch's writes, the
/// pipeline, the argument block address, the dispatch.
unsafe fn record_launch(
    device: &Device,
    command: vk::CommandBuffer,
    launch: &DirectLaunch<'_>,
    address: u64,
    groups: [u32; 3],
    first: bool,
) {
    let raw = &device.inner.device;
    if !first {
        compute_barrier(raw, command);
    }
    raw.cmd_bind_pipeline(
        command,
        vk::PipelineBindPoint::COMPUTE,
        launch.module.pipeline_handle(launch.function),
    );
    raw.cmd_push_constants(
        command,
        device.inner.layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        &address.to_le_bytes(),
    );
    match launch.grid {
        Some((buffer, offset)) => {
            let range = buffer.range();
            raw.cmd_dispatch_indirect(command, range.raw(), range.offset() + offset);
        }
        None => raw.cmd_dispatch(command, groups[0], groups[1], groups[2]),
    }
}

/// Each launch follows the previous launch's writes, as on a CUDA stream,
/// its indirect dispatch's read of a grid among them.
unsafe fn compute_barrier(raw: &ash::Device, command: vk::CommandBuffer) {
    barrier(
        raw,
        command,
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_WRITE,
        vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::DRAW_INDIRECT,
        vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::INDIRECT_COMMAND_READ,
    );
}

/// The group counts of a non-empty launch, checked against the device.
fn groups(device: &Device, launch: &DirectLaunch<'_>) -> Result<Option<[u32; 3]>, ExecutionError> {
    if launch.groups.contains(&0) {
        return Ok(None);
    }
    let limit = device.facts().limits.max_group_count;
    let mut groups = [0u32; 3];
    for axis in 0..3 {
        if launch.groups[axis] > limit[axis] {
            return Err(ExecutionError::SubmissionFailed(format!(
                "native launch requests {} groups on axis {axis}; the device allows {}",
                launch.groups[axis], limit[axis]
            )));
        }
        groups[axis] = launch.groups[axis] as u32;
    }
    Ok(Some(groups))
}

fn begin(
    device: &Device,
    command: vk::CommandBuffer,
    flags: vk::CommandBufferUsageFlags,
    inheritance: Option<&vk::CommandBufferInheritanceInfo<'_>>,
) -> Result<(), ExecutionError> {
    let mut info = vk::CommandBufferBeginInfo::default().flags(flags);
    if let Some(inheritance) = inheritance {
        info = info.inheritance_info(inheritance);
    }
    call(
        unsafe { device.inner.device.begin_command_buffer(command, &info) },
        "vkBeginCommandBuffer",
    )
}

/// Forms a [`DirectGraph`]: the launches of one submission, in order, with
/// their argument blocks fixed, in one or more segments.
pub struct DirectGraphBuilder {
    device: Device,
    /// Heap usage and held memory when recording began.
    mark: (u64, u64),
    /// Becomes the graph; dropping an abandoned builder retires it (and the
    /// chunks it wrote) unsubmitted.
    graph: Option<GraphInner>,
    arguments: Arguments,
    /// Each launch's argument block, in launch order.
    blocks: Vec<Block>,
    launched: bool,
    /// The launches recorded so far ([`DirectGraph::census`]).
    launches: GraphCensus,
}

impl DirectGraphBuilder {
    pub fn new(device: &Device) -> Result<Self, ExecutionError> {
        let mut builder = Self {
            device: device.clone(),
            mark: device.inner.unheld_mark(),
            graph: Some(GraphInner {
                device: device.clone(),
                commands: Vec::new(),
                arguments: Vec::new(),
                last_use: AtomicU64::new(0),
            }),
            arguments: Arguments::new(),
            blocks: Vec::new(),
            launched: false,
            launches: GraphCensus::default(),
        };
        builder.begin_segment()?;
        Ok(builder)
    }

    fn graph(&mut self) -> &mut GraphInner {
        self.graph.as_mut().expect("a live builder holds its graph")
    }

    fn command(&self) -> vk::CommandBuffer {
        *self
            .graph
            .as_ref()
            .expect("a live builder holds its graph")
            .commands
            .last()
            .expect("a live builder has begun a segment")
    }

    /// Allocate and begin the secondary buffer of the next segment.
    fn begin_segment(&mut self) -> Result<(), ExecutionError> {
        let device = self.device.clone();
        let mut recording = device.recording()?;
        let command = device.command(&mut recording, vk::CommandBufferLevel::SECONDARY)?;
        self.graph().commands.push(command);
        let inheritance = vk::CommandBufferInheritanceInfo::default();
        let begun = begin(
            &device,
            command,
            vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
            Some(&inheritance),
        );
        // Released before an error drops the builder, whose graph retires
        // under this lock.
        drop(recording);
        begun
    }

    fn end_segment(&self) -> Result<(), ExecutionError> {
        let recording = self.device.locked();
        let ended = call(
            unsafe { self.device.inner.device.end_command_buffer(self.command()) },
            "vkEndCommandBuffer",
        );
        drop(recording);
        ended
    }

    /// End the current segment and begin the next: launches added from now
    /// on can be replayed apart from those added before, in their own queue
    /// submission. Segments share the graph's argument blocks, and a
    /// segment's first launch still follows the previous launch's writes.
    pub fn segment(&mut self) -> Result<(), ExecutionError> {
        self.end_segment()?;
        self.begin_segment()
    }

    /// Append a launch after every launch added before it; an empty grid
    /// adds nothing.
    pub fn launch(&mut self, launch: &DirectLaunch<'_>) -> Result<(), ExecutionError> {
        let Some(groups) = groups(&self.device, launch)? else {
            return Ok(());
        };
        let mut recording = self.device.recording()?;
        let (address, block) = self.arguments.write_block(&self.device, &mut recording, launch)?;
        self.blocks.push(block);
        unsafe {
            record_launch(
                &self.device,
                self.command(),
                launch,
                address,
                groups,
                !self.launched,
            )
        };
        self.launched = true;
        let census = GraphCensus::launch(argument_block_bytes(launch.buffers.len() as u64, launch.words.len() as u64));
        self.launches = self
            .launches
            .plus(census)
            .ok_or_else(|| ExecutionError::SubmissionFailed("a graph's launch census overflows".into()))?;
        Ok(())
    }

    /// Finish recording. The device memory recording took beyond blocks
    /// (command memory) and its argument chunks are the graph's; the device
    /// keeps both after the graph is dropped, for the graphs recorded after
    /// it (command buffers it reuses, chunks it hands out again), so they
    /// stay held. Only the command memory is the driver's: the chunks are
    /// charged as the upload blocks they lie in.
    pub fn instantiate(mut self) -> Result<DirectGraph, ExecutionError> {
        self.end_segment()?;
        let mut graph = self.graph.take().expect("a live builder holds its graph");
        graph.arguments = std::mem::take(&mut self.arguments.chunks);
        let inner = &self.device.inner;
        let command = inner.unheld_since(self.mark);
        inner.graphs.fetch_add(command, Ordering::Relaxed);
        let arguments = graph.arguments.iter().map(Range::bytes).sum::<u64>();
        Ok(DirectGraph {
            inner: std::sync::Arc::new(graph),
            blocks: std::mem::take(&mut self.blocks),
            device_bytes: command + arguments,
            census: GraphCensus::graph(self.launches)
                .ok_or_else(|| ExecutionError::SubmissionFailed("a graph's census overflows".into()))?,
        })
    }
}

impl Drop for DirectGraphBuilder {
    fn drop(&mut self) {
        if let Some(graph) = &mut self.graph {
            graph.arguments.append(&mut self.arguments.chunks);
        }
    }
}

/// Recorded secondary command buffers (its segments, in launch order) of
/// launches with fixed arguments. One replay replaces its launches'
/// individual recording.
pub struct DirectGraph {
    inner: std::sync::Arc<GraphInner>,
    /// Each launch's argument block as the device holds it once the work
    /// queued so far ran ([`DirectBatch::rebind`]).
    blocks: Vec<Block>,
    device_bytes: u64,
    /// What the graph is made of, as its memory follows it.
    census: GraphCensus,
}

impl DirectGraph {
    pub fn segments(&self) -> usize {
        self.inner.commands.len()
    }

    /// Its launches (each not [`DirectLaunch::is_empty`]).
    pub fn launches(&self) -> usize {
        self.blocks.len()
    }

    /// Forget what its argument blocks hold, after a rewrite queued in a
    /// batch that never committed ([`DirectBatch::rebind`]): the next
    /// rebind rewrites every block.
    pub fn unbind(&mut self) {
        for block in &mut self.blocks {
            block.bytes.clear();
        }
    }

    /// Its launches and their argument blocks' bytes
    /// ([`argument_block_bytes`]).
    pub fn census(&self) -> GraphCensus {
        self.census
    }

    /// The driver's command memory recording it took: what it holds less
    /// its argument chunks.
    pub(crate) fn command_bytes(&self) -> u64 {
        self.device_bytes - self.inner.arguments.iter().map(Range::bytes).sum::<u64>()
    }
}


/// Shared by the graph and the submissions replaying it, so its buffers and
/// argument blocks outlive every replay.
struct GraphInner {
    device: Device,
    commands: Vec<vk::CommandBuffer>,
    arguments: Vec<Range>,
    /// Timeline value of the latest submission that replays it.
    last_use: AtomicU64,
}

impl Drop for GraphInner {
    fn drop(&mut self) {
        let value = self.last_use.load(Ordering::Acquire);
        let mut arguments = std::mem::take(&mut self.arguments);
        for command in self.commands.drain(..) {
            self.device.retire(Retired {
                value,
                command: RetiredCommand::Secondary(command),
                queries: None,
                arguments: std::mem::take(&mut arguments),
            });
        }
    }
}

/// Launches recorded into one primary command buffer and submitted to the
/// device's queue. A leading barrier orders the batch after every earlier
/// submission (its writes to every later read, indirect grids included);
/// each launch follows the previous launch's writes; a trailing barrier
/// makes results visible to the host. Whether it writes device timestamps is
/// its type ([`Timing`]): a [`Measured`] batch resets its queries and
/// stamps its start and end (and, timed, each launch), an [`Unmeasured`]
/// one writes none.
pub struct DirectBatch<T: Timing = Measured> {
    device: Device,
    command: vk::CommandBuffer,
    timing: T,
    arguments: Arguments,
    /// Whether a dispatch or replay has been recorded (the next one needs a
    /// barrier).
    launched: bool,
    /// Graphs replayed by this batch; the submission keeps them alive and
    /// records itself as their last use.
    replayed: Vec<std::sync::Arc<GraphInner>>,
    /// Host timelines the batch's queue submission waits on before it runs,
    /// and the ones it signals once it completes, with their values.
    waits: Vec<(vk::Semaphore, u64)>,
    signals: Vec<(vk::Semaphore, u64)>,
    /// Set by `commit`; an uncommitted batch returns its resources on drop.
    committed: bool,
}

mod sealed {
    pub trait Sealed {}
}

/// Whether a batch writes device timestamps (see [`DirectBatch`]).
pub trait Timing: sealed::Sealed + Sized {
    /// What its submission keeps of them.
    type Stamps;

    /// Its queries, taken under the recording lock.
    #[doc(hidden)]
    fn open(device: &Device, recording: &mut Recording, launches: Option<usize>) -> Result<Self, ExecutionError>;

    /// Its query pool, returned with the batch.
    #[doc(hidden)]
    fn queries(&self) -> Option<(vk::QueryPool, u32)>;

    /// Reset its queries, before the leading barrier.
    #[doc(hidden)]
    fn reset(&self, raw: &ash::Device, command: vk::CommandBuffer);

    /// Stamp the batch's start, after the leading barrier.
    #[doc(hidden)]
    fn start(&mut self, raw: &ash::Device, command: vk::CommandBuffer);

    /// Stamp the batch's end, after the trailing barrier, and keep what its
    /// submission reads.
    #[doc(hidden)]
    fn end(&mut self, raw: &ash::Device, command: vk::CommandBuffer) -> Self::Stamps;

    /// For a timed batch, stamp a launch's start, after the previous
    /// launch's writes where one was recorded, and keep its mark: whether
    /// it did (the launch then needs no barrier of its own).
    #[doc(hidden)]
    fn mark(&mut self, raw: &ash::Device, command: vk::CommandBuffer, launched: bool) -> bool;

    /// For a timed batch, keep an empty launch's place.
    #[doc(hidden)]
    fn skip(&mut self);

    /// Whether the batch times each launch.
    #[doc(hidden)]
    fn timed(&self) -> bool;
}

/// Device timestamps at marks of work on the device's queue
/// ([`DirectBatch::stamp`]): a query of a pool of their own per mark, each
/// reset and written as the queue passes its mark, so the device time
/// between two marks is read once the work has passed both. Marks may lie
/// in different batches of one queue: its timestamps share one domain.
pub struct Stamps {
    device: Device,
    pool: vk::QueryPool,
    marks: u32,
}

impl Stamps {
    /// The stamps of `marks` marks on `device`.
    pub fn new(device: &Device, marks: usize) -> Result<Self, ExecutionError> {
        let marks = u32::try_from(marks).map_err(|_| ExecutionError::SubmissionFailed("too many timestamp marks".into()))?;
        let info = vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(marks.max(1));
        let pool = call(unsafe { device.inner.device.create_query_pool(&info, None) }, "vkCreateQueryPool")?;
        Ok(Self { device: device.clone(), pool, marks })
    }

    /// The device seconds from mark `from` to mark `to`, once the queue has
    /// passed both (waited for here).
    pub fn seconds(&self, from: usize, to: usize) -> Result<f64, ExecutionError> {
        let mut values = vec![0u64; self.marks as usize];
        call(
            unsafe {
                self.device.inner.device.get_query_pool_results(
                    self.pool,
                    0,
                    &mut values,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
            },
            "vkGetQueryPoolResults",
        )?;
        let limits = self.device.facts().limits;
        let ticks = values[to].wrapping_sub(values[from]) & mask(limits.timestamp_valid_bits);
        Ok(ticks as f64 * limits.timestamp_period / 1e9)
    }
}

impl Drop for Stamps {
    fn drop(&mut self) {
        // The work that wrote them has completed or never ran: its owner
        // retires it before the stamps it holds.
        unsafe { self.device.inner.device.destroy_query_pool(self.pool, None) };
    }
}

/// A batch that writes its start and end timestamps, and, timed, one before
/// each launch: its submission's device interval is known.
pub struct Measured {
    queries: (vk::QueryPool, u32),
    /// Queries written so far (the first is the batch start).
    written: u32,
    /// For a timed batch, per `launch` call, the query written just before
    /// the launch (absent for an empty launch).
    marks: Option<Vec<Option<u32>>>,
}

/// A batch that writes no timestamp and resets no query.
pub struct Unmeasured;

/// What a measured submission reads: its end query (the start is query 0),
/// and a timed batch's marks.
pub struct MeasuredStamps {
    end: u32,
    marks: Option<Vec<Option<u32>>>,
}

impl sealed::Sealed for Measured {}
impl sealed::Sealed for Unmeasured {}

impl Measured {
    fn stamp(&mut self, raw: &ash::Device, command: vk::CommandBuffer) -> u32 {
        let query = self.written;
        unsafe { raw.cmd_write_timestamp2(command, vk::PipelineStageFlags2::ALL_COMMANDS, self.queries.0, query) };
        self.written += 1;
        query
    }
}

impl Timing for Measured {
    type Stamps = MeasuredStamps;

    fn open(device: &Device, recording: &mut Recording, launches: Option<usize>) -> Result<Self, ExecutionError> {
        let queries = u32::try_from(launches.unwrap_or(0) + 2)
            .map_err(|_| ExecutionError::SubmissionFailed("too many timed launches".into()))?;
        Ok(Self {
            queries: device.queries(recording, queries)?,
            written: 0,
            marks: launches.map(Vec::with_capacity),
        })
    }

    fn queries(&self) -> Option<(vk::QueryPool, u32)> {
        Some(self.queries)
    }

    fn reset(&self, raw: &ash::Device, command: vk::CommandBuffer) {
        unsafe { raw.cmd_reset_query_pool(command, self.queries.0, 0, self.queries.1) };
    }

    fn start(&mut self, raw: &ash::Device, command: vk::CommandBuffer) {
        self.stamp(raw, command);
    }

    fn end(&mut self, raw: &ash::Device, command: vk::CommandBuffer) -> MeasuredStamps {
        MeasuredStamps { end: self.stamp(raw, command), marks: self.marks.take() }
    }

    fn mark(&mut self, raw: &ash::Device, command: vk::CommandBuffer, launched: bool) -> bool {
        if self.marks.is_none() {
            return false;
        }
        if launched {
            unsafe { compute_barrier(raw, command) };
        }
        let mark = self.stamp(raw, command);
        self.marks.as_mut().expect("a timed batch has marks").push(Some(mark));
        true
    }

    fn skip(&mut self) {
        if let Some(marks) = &mut self.marks {
            marks.push(None);
        }
    }

    fn timed(&self) -> bool {
        self.marks.is_some()
    }
}

impl Timing for Unmeasured {
    type Stamps = ();

    fn open(_: &Device, _: &mut Recording, launches: Option<usize>) -> Result<Self, ExecutionError> {
        assert!(launches.is_none(), "an unmeasured batch times no launch");
        Ok(Self)
    }

    fn queries(&self) -> Option<(vk::QueryPool, u32)> {
        None
    }

    fn reset(&self, _: &ash::Device, _: vk::CommandBuffer) {}

    fn start(&mut self, _: &ash::Device, _: vk::CommandBuffer) {}

    fn end(&mut self, _: &ash::Device, _: vk::CommandBuffer) {}

    fn mark(&mut self, _: &ash::Device, _: vk::CommandBuffer, _: bool) -> bool {
        false
    }

    fn skip(&mut self) {}

    fn timed(&self) -> bool {
        false
    }
}

/// Which side raises a host timeline, so a batch can only wait on what the
/// host raises and only signal what the host waits on: a gate no device
/// signal can satisfy, by type.
pub trait Side: sealed::Sealed {}

/// The host raises the timeline ([`HostTimeline::signal`]) and the device's
/// submissions wait on it ([`DirectBatch::wait_timeline`]): a gate.
pub enum Gate {}

/// The device's submissions raise the timeline
/// ([`DirectBatch::signal_timeline`]) and the host waits on it
/// ([`HostTimeline::wait`]).
pub enum Reached {}

impl sealed::Sealed for Gate {}
impl sealed::Sealed for Reached {}
impl Side for Gate {}
impl Side for Reached {}

/// A timeline semaphore between the host and the device's queue
/// submissions, raised by side `S`.
pub struct HostTimeline<S: Side> {
    device: Device,
    semaphore: vk::Semaphore,
    side: std::marker::PhantomData<fn() -> S>,
}

impl<S: Side> HostTimeline<S> {
    /// A timeline at 0.
    pub fn new(device: &Device) -> Result<Self, ExecutionError> {
        let mut timeline = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let semaphore = call(
            unsafe {
                device.inner.device.create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut timeline),
                    None,
                )
            },
            "vkCreateSemaphore",
        )?;
        Ok(Self {
            device: device.clone(),
            semaphore,
            side: std::marker::PhantomData,
        })
    }

    /// Raise the timeline to `value` from the host; a value it has reached
    /// already changes nothing. A device's timeline ([`Reached`]) is raised
    /// so only by a host that plays the device (a test). Precondition: no
    /// other signal of the timeline runs at once (`vkSignalSemaphore`'s
    /// value must exceed the semaphore's, which a concurrent signal may
    /// pass between this one's read and its signal).
    ///
    /// Host writes made before the signal are visible to the device work
    /// that waits on `value` (`DirectBatch::wait_timeline`):
    /// - the writes are to `HOST_COHERENT` memory (every upload allocation,
    ///   `Memory::new`, which host-shared tensors are), so they are
    ///   available in the host domain without `vkFlushMappedMemoryRanges`
    ///   (spec, "Host Access to Device Memory Objects": a flush is required
    ///   only of memory without `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`);
    /// - `vkSignalSemaphore`'s first synchronization scope includes every
    ///   host operation that happened-before it (spec, "Semaphore Signaling",
    ///   `vkSignalSemaphore`), and the waiting batch's semaphore wait
    ///   operation's second synchronization and access scopes include every
    ///   command of the batch and all its device memory accesses, at the
    ///   stages of its wait mask (spec, "Semaphore Waiting"), which is
    ///   `COMPUTE_SHADER | ALL_COMMANDS` (`Device::submit_with`).
    pub fn signal(&self, value: u64) -> Result<(), ExecutionError> {
        self.raise(value, GateOpening::Signalled)
    }

    fn raise(&self, value: u64, opening: GateOpening) -> Result<(), ExecutionError> {
        if self.value()? >= value {
            return Ok(());
        }
        let info = vk::SemaphoreSignalInfo::default()
            .semaphore(self.semaphore)
            .value(value);
        call(
            unsafe { self.device.inner.device.signal_semaphore(&info) },
            "vkSignalSemaphore",
        )?;
        // The submissions held on this gate go to the queue now
        // (`Device::submit_with`).
        self.device.release(opening);
        Ok(())
    }

    /// The value reached; the device's loss where it cannot say.
    pub fn value(&self) -> Result<u64, ExecutionError> {
        call(
            unsafe {
                self.device
                    .inner
                    .device
                    .get_semaphore_counter_value(self.semaphore)
            },
            "vkGetSemaphoreCounterValue",
        )
    }

    /// Wait at most `timeout` for the timeline to reach `value`; whether it
    /// did, or the device's loss where it cannot say.
    pub fn wait(&self, value: u64, timeout: std::time::Duration) -> Result<bool, ExecutionError> {
        let semaphores = [self.semaphore];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        let nanoseconds = u64::try_from(timeout.as_nanos()).unwrap_or(u64::MAX);
        match unsafe { self.device.inner.device.wait_semaphores(&info, nanoseconds) } {
            Ok(()) => Ok(true),
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(error) => call(Err(error), "vkWaitSemaphores"),
        }
    }
}

impl HostTimeline<Gate> {
    /// [`Self::signal`] as a park's answer: the submission waiting on it in
    /// the driver records when it opened and `answer`
    /// ([`DirectSubmission::opened`]).
    pub fn answer(&self, value: u64, answer: ParkAnswer) -> Result<(), ExecutionError> {
        self.raise(value, GateOpening::Answered(answer))
    }
}

impl<S: Side> Drop for HostTimeline<S> {
    /// A submission may still wait on or signal the semaphore: every one
    /// submitted so far completes first. Precondition: every value a
    /// submission waits on was signaled, or is signaled by one before it.
    /// Where the wait fails (the device is lost) the semaphore is never
    /// destroyed: a submission may still use it.
    fn drop(&mut self) {
        if self.device.ended(self.device.submitted()).is_ok() {
            unsafe { self.device.inner.device.destroy_semaphore(self.semaphore, None) };
        }
    }
}

impl DirectBatch<Measured> {
    /// A batch that stamps its start and end.
    pub fn new(device: &Device) -> Result<Self, ExecutionError> {
        Self::create(device, None)
    }

    /// A measurement batch that writes a timestamp before every launch, so
    /// each launch's device interval is known; `launches` bounds the calls
    /// of `launch` and `skip`. A timed batch attributes time but does not
    /// measure production, and never replays.
    pub fn timed(device: &Device, launches: usize) -> Result<Self, ExecutionError> {
        Self::create(device, Some(launches))
    }
}

impl DirectBatch<Unmeasured> {
    /// A batch that writes no timestamp.
    pub fn unmeasured(device: &Device) -> Result<Self, ExecutionError> {
        Self::create(device, None)
    }
}

impl<T: Timing> DirectBatch<T> {
    fn create(device: &Device, launches: Option<usize>) -> Result<Self, ExecutionError> {
        let mut recording = device.recording()?;
        let command = device.command(&mut recording, vk::CommandBufferLevel::PRIMARY)?;
        let timing = match T::open(device, &mut recording, launches) {
            Ok(timing) => timing,
            Err(error) => {
                recording.free_command(command);
                return Err(error);
            }
        };
        let mut batch = Self {
            device: device.clone(),
            command,
            timing,
            arguments: Arguments::new(),
            launched: false,
            replayed: Vec::new(),
            waits: Vec::new(),
            signals: Vec::new(),
            committed: false,
        };
        let begun = begin(
            device,
            command,
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            None,
        );
        if begun.is_ok() {
            let raw = &device.inner.device;
            batch.timing.reset(raw, command);
            // After every earlier submission's writes (whose trailing
            // barrier reaches only the host), for every read and write of
            // this one, an indirect grid's included.
            unsafe {
                barrier(
                    raw,
                    command,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    vk::AccessFlags2::MEMORY_WRITE,
                    vk::PipelineStageFlags2::COMPUTE_SHADER
                        | vk::PipelineStageFlags2::TRANSFER
                        | vk::PipelineStageFlags2::DRAW_INDIRECT,
                    vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                );
            }
            batch.timing.start(raw, command);
        }
        // Released before an error drops the batch, which retires under it.
        drop(recording);
        begun.map(|()| batch)
    }

    /// The batch's queue submission waits, before any of its work, until
    /// `timeline` reaches `value`: the host gate, which only the host raises
    /// ([`HostTimeline::signal`]); until then the submission waits in the
    /// driver, or, behind another that does, on the host, as every one
    /// after it does (`Device::submit_with`). Precondition:
    /// nothing is recorded in the batch yet.
    pub fn wait_timeline(&mut self, timeline: &HostTimeline<Gate>, value: u64) {
        assert!(
            !self.launched,
            "DirectBatch::wait_timeline precondition: the batch records nothing yet"
        );
        self.waits.push((timeline.semaphore, value));
    }

    /// `timeline` reaches `value` once the batch's queue submission
    /// completes. Precondition: the batch records nothing after this.
    pub fn signal_timeline(&mut self, timeline: &HostTimeline<Reached>, value: u64) {
        self.signals.push((timeline.semaphore, value));
    }

    /// Record a launch that does no device work (inactive, or an empty
    /// grid): nothing is recorded, and a timed batch keeps its place with no
    /// interval.
    pub fn skip(&mut self) {
        self.timing.skip();
    }

    pub fn launch(&mut self, launch: &DirectLaunch<'_>) -> Result<(), ExecutionError> {
        let Some(groups) = groups(&self.device, launch)? else {
            self.skip();
            return Ok(());
        };
        let device = self.device.clone();
        let mut recording = device.recording()?;
        let address = self.arguments.write(&device, &mut recording, launch)?;
        if self.timing.mark(&device.inner.device, self.command, self.launched) {
            unsafe { record_launch(&device, self.command, launch, address, groups, true) };
        } else {
            unsafe {
                record_launch(
                    &device,
                    self.command,
                    launch,
                    address,
                    groups,
                    !self.launched,
                )
            };
        }
        self.launched = true;
        Ok(())
    }

    /// Rewrite, in queue order, the argument blocks of `graph` whose bytes
    /// differ from `blocks` (each launch's [`DirectLaunch::argument_block`],
    /// in launch order), so the graph's launches replayed after this in the
    /// batch, or in a later submission, take them: a graph is recorded once
    /// for its shape and bound to the storage of each replay. The rewrite
    /// follows every earlier use of the blocks (the batch's leading barrier
    /// orders it after earlier submissions, a barrier here after this
    /// batch's launches) and precedes every later one. The bytes are
    /// carried in the command buffer (`vkCmdUpdateBuffer`): nothing is
    /// written through the mapping, which work in flight may still read.
    pub fn rebind(&mut self, graph: &mut DirectGraph, blocks: Vec<Vec<u8>>) -> Result<(), ExecutionError> {
        assert!(
            !self.timing.timed(),
            "DirectBatch::rebind precondition: a timed batch times each launch"
        );
        assert_eq!(blocks.len(), graph.blocks.len(), "a shape's launches are its graph's");
        if graph.blocks.iter().zip(&blocks).all(|(held, block)| held.bytes == *block) {
            return Ok(());
        }
        let _recording = self.device.locked();
        let raw = &self.device.inner.device;
        unsafe {
            if self.launched {
                barrier(
                    raw,
                    self.command,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_READ,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                );
            }
            for (held, block) in graph.blocks.iter_mut().zip(blocks) {
                if held.bytes != block {
                    let chunk = &graph.inner.arguments[held.chunk];
                    raw.cmd_update_buffer(self.command, chunk.raw(), chunk.offset() + held.offset, &block);
                    held.bytes = block;
                }
            }
            barrier(
                raw,
                self.command,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ,
            );
        }
        self.launched = true;
        if !self
            .replayed
            .iter()
            .any(|replayed| std::sync::Arc::ptr_eq(replayed, &graph.inner))
        {
            self.replayed.push(graph.inner.clone());
        }
        Ok(())
    }

    /// Execute every launch of `graph` in its order.
    pub fn replay(&mut self, graph: &DirectGraph) -> Result<(), ExecutionError> {
        self.execute(graph, &graph.inner.commands)
    }

    /// Stamp mark `mark` of `stamps` once the queue's preceding work
    /// completes ([`Stamps`]).
    pub fn stamp(&mut self, stamps: &Stamps, mark: usize) {
        let _recording = self.device.locked();
        let raw = &self.device.inner.device;
        let query = u32::try_from(mark).expect("a mark of its stamps");
        unsafe {
            raw.cmd_reset_query_pool(self.command, stamps.pool, query, 1);
            raw.cmd_write_timestamp2(self.command, vk::PipelineStageFlags2::ALL_COMMANDS, stamps.pool, query);
        }
    }

    /// Execute the launches of the segments `segments` of `graph`, in order.
    pub fn replay_segments(
        &mut self,
        graph: &DirectGraph,
        segments: std::ops::Range<usize>,
    ) -> Result<(), ExecutionError> {
        self.execute(graph, &graph.inner.commands[segments])
    }

    fn execute(
        &mut self,
        graph: &DirectGraph,
        commands: &[vk::CommandBuffer],
    ) -> Result<(), ExecutionError> {
        assert!(
            !self.timing.timed(),
            "DirectBatch::replay precondition: a timed batch times each launch"
        );
        let _recording = self.device.locked();
        let raw = &self.device.inner.device;
        unsafe {
            if self.launched {
                compute_barrier(raw, self.command);
            }
            raw.cmd_execute_commands(self.command, commands);
        }
        self.launched = true;
        if !self
            .replayed
            .iter()
            .any(|replayed| std::sync::Arc::ptr_eq(replayed, &graph.inner))
        {
            self.replayed.push(graph.inner.clone());
        }
        Ok(())
    }

    /// Submit without waiting.
    pub fn commit(self) -> Result<DirectSubmission<T>, ExecutionError> {
        self.commit_by(Device::submit_with)
    }

    /// [`Self::commit`] through `submit`, which hands the recorded batch to
    /// the queue (`Device::submit_with`, or a probe's own).
    fn commit_by(
        mut self,
        submit: impl FnOnce(&Device, vk::CommandBuffer, &[(vk::Semaphore, u64)], &[(vk::Semaphore, u64)], Handed) -> Result<u64, ExecutionError>,
    ) -> Result<DirectSubmission<T>, ExecutionError> {
        let device = self.device.clone();
        let recording = device.recording()?;
        let raw = &device.inner.device;
        // The batch's writes to the host. A later submission's reads follow
        // them by its own leading barrier.
        unsafe {
            barrier(
                raw,
                self.command,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::HOST,
                vk::AccessFlags2::HOST_READ,
            );
        }
        let stamps = self.timing.end(raw, self.command);
        let ended = call(
            unsafe { raw.end_command_buffer(self.command) },
            "vkEndCommandBuffer",
        );
        drop(recording);
        ended?;
        let handed = Handed::default();
        let value = submit(&self.device, self.command, &self.waits, &self.signals, handed.clone())?;
        for graph in &self.replayed {
            graph.last_use.fetch_max(value, Ordering::AcqRel);
        }
        self.committed = true;
        Ok(DirectSubmission {
            device: self.device.clone(),
            value,
            retired: Some(Retired {
                value,
                command: RetiredCommand::Primary(self.command),
                queries: self.timing.queries(),
                arguments: std::mem::take(&mut self.arguments.chunks),
            }),
            _graphs: std::mem::take(&mut self.replayed),
            stamps,
            handed,
        })
    }
}

impl<T: Timing> Drop for DirectBatch<T> {
    fn drop(&mut self) {
        if !self.committed {
            // Nothing was submitted: everything is reusable at once.
            self.device.retire(Retired {
                value: 0,
                command: RetiredCommand::Primary(self.command),
                queries: self.timing.queries(),
                arguments: std::mem::take(&mut self.arguments.chunks),
            });
        }
    }
}

/// The samples of a [`TimelineAnchor`].
const ANCHOR_SAMPLES: usize = 16;

/// A batch's device timestamps paired with the host clock: the origin that
/// places device times on the host timeline. Recorded on the idle queue,
/// polled to its completion and paired with the host clock read at once: a
/// blocking wait returns tens of microseconds after the value is reached,
/// which would place every device time as late. The pair read soonest after
/// its batch's end, of [`ANCHOR_SAMPLES`], is the anchor.
pub struct TimelineAnchor {
    ticks: u64,
    host_seconds: f64,
    period: f64,
    mask: u64,
}

impl TimelineAnchor {
    pub fn record(device: &Device, host_clock: impl Fn() -> f64) -> Result<Self, ExecutionError> {
        let limits = device.facts().limits;
        let (period, mask) = (limits.timestamp_period, mask(limits.timestamp_valid_bits));
        let mut anchor: Option<Self> = None;
        for _ in 0..ANCHOR_SAMPLES {
            let submission = DirectBatch::new(device)?.commit()?;
            while !submission.completed()? {
                std::hint::spin_loop();
            }
            let host_seconds = host_clock();
            let sample = Self { ticks: submission.timestamps(2)?[1], host_seconds, period, mask };
            // Read before the anchor so far places its end: read sooner
            // after it.
            if anchor.as_ref().is_none_or(|anchor| sample.host_seconds < anchor.place(sample.ticks)) {
                anchor = Some(sample);
            }
        }
        Ok(anchor.expect("an anchor takes samples"))
    }

    fn place(&self, ticks: u64) -> f64 {
        let elapsed = ticks.wrapping_sub(self.ticks) & self.mask;
        // A tick before the anchor wraps to a huge value: take it negative.
        let signed = if elapsed > self.mask / 2 {
            -(((self.ticks.wrapping_sub(ticks)) & self.mask) as f64)
        } else {
            elapsed as f64
        };
        self.host_seconds + signed * self.period / 1e9
    }
}

fn mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// Submitted launches, observed through the device's timeline semaphore; a
/// [`Measured`] one's timestamps read once it completes.
pub struct DirectSubmission<T: Timing = Measured> {
    device: Device,
    value: u64,
    /// Returned to the device for reuse when this submission is dropped.
    retired: Option<Retired>,
    /// The graphs it replays, alive until it is.
    _graphs: Vec<std::sync::Arc<GraphInner>>,
    stamps: T::Stamps,
    handed: Handed,
}

impl<T: Timing> DirectSubmission<T> {
    /// When the batch was handed to the queue, before and after the call
    /// that took it: at its commit, or, where it was held behind a host
    /// gate, when the gate's raising released it (`None` while it is held,
    /// or where the queue refused it).
    pub fn handed(&self) -> Option<(std::time::Instant, std::time::Instant)> {
        self.handed.handed.get().copied()
    }

    /// When the host raised the last gate the batch was handed to the queue
    /// waiting on, and why; `None` where it was handed over with its gates
    /// raised (or none), or the queue refused it.
    pub fn opened(&self) -> Option<(std::time::Instant, GateOpening)> {
        self.handed.opened.get().copied()
    }

    /// Whether the batch completed, without waiting: `Ok(false)` while it
    /// runs, and its error where the queue refused it or the device is
    /// lost.
    pub fn completed(&self) -> Result<bool, ExecutionError> {
        self.device.reached(self.value)
    }

    /// Wait for the batch to end and report its outcome.
    pub fn finish(&self) -> Result<(), ExecutionError> {
        self.device.wait(self.value)
    }

}

impl DirectSubmission<Measured> {
    /// The first `count` timestamps of the completed batch.
    fn timestamps(&self, count: u32) -> Result<Vec<u64>, ExecutionError> {
        self.finish()?;
        let (pool, _) = self
            .retired
            .as_ref()
            .and_then(|retired| retired.queries)
            .expect("a live submission holds its queries");
        let mut values = vec![0u64; count as usize];
        call(
            unsafe {
                self.device.inner.device.get_query_pool_results(
                    pool,
                    0,
                    &mut values,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
            },
            "vkGetQueryPoolResults",
        )?;
        Ok(values)
    }

    /// Device execution time between the batch's start and end timestamps.
    pub fn device_seconds(&self) -> Result<f64, ExecutionError> {
        let end = self.stamps.end;
        let values = self.timestamps(end + 1)?;
        let limits = self.device.facts().limits;
        let ticks =
            values[end as usize].wrapping_sub(values[0]) & mask(limits.timestamp_valid_bits);
        Ok(ticks as f64 * limits.timestamp_period / 1e9)
    }

    /// The completed batch's device interval on the anchor's host clock.
    pub fn device_interval(&self, anchor: &TimelineAnchor) -> Result<(f64, f64), ExecutionError> {
        let end = self.stamps.end;
        let values = self.timestamps(end + 1)?;
        Ok((anchor.place(values[0]), anchor.place(values[end as usize])))
    }

    /// For a timed batch, each `launch` call's device interval on the
    /// anchor's host clock, in launch order (`None` for an empty launch). A
    /// launch ends where the next recorded launch (or the batch) ends.
    /// `None` for a production batch.
    pub fn launch_intervals(
        &self,
        anchor: &TimelineAnchor,
    ) -> Result<Option<Vec<Option<(f64, f64)>>>, ExecutionError> {
        let Some(marks) = &self.stamps.marks else {
            return Ok(None);
        };
        let values = self.timestamps(self.stamps.end + 1)?;
        let mut intervals = Vec::with_capacity(marks.len());
        for (index, mark) in marks.iter().enumerate() {
            let Some(mark) = mark else {
                intervals.push(None);
                continue;
            };
            let end = marks[index + 1..]
                .iter()
                .flatten()
                .next()
                .copied()
                .unwrap_or(self.stamps.end);
            intervals.push(Some((
                anchor.place(values[*mark as usize]),
                anchor.place(values[end as usize]),
            )));
        }
        Ok(Some(intervals))
    }
}

impl<T: Timing> Drop for DirectSubmission<T> {
    fn drop(&mut self) {
        if let Some(retired) = self.retired.take() {
            self.device.retire(retired);
        }
    }
}


#[cfg(test)]
mod park_tests {
    //! A submission behind a host gate the host has not raised (a park)
    //! never reaches the driver: on NVIDIA's driver, with one in the queue,
    //! `vkQueueSubmit2` blocks after ~2,000 more submissions (2,182 plain,
    //! 2,046 that each park; RTX 3090, 2026-11-02), and so does every
    //! queue's, the second's included, until the host raises the gate; a
    //! host whose raising waits on a submission then never raises it.
    //! `vkSignalSemaphore` and counter queries still return.

    use super::*;
    use crate::transfer::{TransferEnd, TransferItem, TransferQueue, TransferRun};
    use std::sync::{mpsc, Arc};
    use std::time::{Duration, Instant};

    /// Submissions queued behind the park: past the driver's depth.
    const DEPTH: u64 = 4096;
    /// The longest any step here may take.
    const BOUND: Duration = Duration::from_secs(5);

    /// Run `call` on a thread of its own and require it to return within
    /// [`BOUND`]; what it returned.
    fn returns<R: Send + 'static>(label: &str, call: impl FnOnce() -> R + Send + 'static) -> R {
        let (send, receive) = mpsc::channel();
        let start = Instant::now();
        std::thread::spawn(move || {
            let _ = send.send(call());
        });
        match receive.recv_timeout(BOUND) {
            Ok(returned) => {
                eprintln!("  {label}: {:?}", start.elapsed());
                returned
            }
            Err(_) => {
                eprintln!("{label}: BLOCKED past {BOUND:?}");
                // A driver call that never returns: no unwinding ends it.
                std::process::abort();
            }
        }
    }

    /// A park, then `DEPTH` submissions behind it (each itself a park on a
    /// second timeline where `chained`), a copy on the second queue, then
    /// the answers: every call returns, and the work completes in order.
    fn case(device: &Device, chained: bool) {
        eprintln!("{}:", if chained { "submissions that each park" } else { "plain submissions" });
        let (arrived, park) = (Arc::new(HostTimeline::new(device).unwrap()), Arc::new(HostTimeline::new(device).unwrap()));
        let (chain_arrived, chain) = (Arc::new(HostTimeline::new(device).unwrap()), Arc::new(HostTimeline::new(device).unwrap()));
        let mut reached = DirectBatch::new(device).unwrap();
        reached.signal_timeline(&arrived, 1);
        let _reached = reached.commit().unwrap();
        let mut held = DirectBatch::new(device).unwrap();
        held.wait_timeline(&park, 2);
        let held = held.commit().unwrap();
        assert!(arrived.wait(1, BOUND).unwrap(), "the device reaches the park");

        let submissions = returns("queued behind the park", {
            let (device, chain, chain_arrived) = (device.clone(), chain.clone(), chain_arrived.clone());
            move || {
                (0..DEPTH)
                    .map(|index| {
                        let mut batch = DirectBatch::new(&device).unwrap();
                        if chained {
                            // Park `index` of the chain: the device reaches
                            // 2·index + 1, the next waits for 2·index + 2.
                            if index > 0 {
                                batch.wait_timeline(&chain, 2 * index);
                            }
                            batch.signal_timeline(&chain_arrived, 2 * index + 1);
                        }
                        batch.commit().unwrap()
                    })
                    .collect::<Vec<_>>()
            }
        });
        if device.has_side_queue() {
            let (queue, source, destination) = (
                TransferQueue::new(device).unwrap(),
                device.allocate_upload(4096, 256, None).unwrap(),
                device.allocate_upload(4096, 256, None).unwrap(),
            );
            assert!(returns("a copy on the second queue while parked", move || {
                let run = TransferRun { source: TransferEnd::Buffer(&source, 0), destination: TransferEnd::Buffer(&destination, 0), bytes: 4096 };
                queue.copy(&[TransferItem::Run(run)], 1).unwrap();
                queue.wait(1, BOUND).unwrap()
            }));
        }
        assert!(!held.completed().unwrap(), "the parked submission waits for its answer");
        returns("the park's answer", {
            let park = park.clone();
            move || park.signal(2).unwrap()
        });
        if chained {
            returns("the chain's answers, each as the device reaches it", {
                let (chain, chain_arrived) = (chain.clone(), chain_arrived.clone());
                move || {
                    for index in 0..DEPTH {
                        assert!(chain_arrived.wait(2 * index + 1, BOUND).unwrap(), "the device reaches park {index}");
                        chain.signal(2 * index + 2).unwrap();
                    }
                }
            });
        }
        returns("the work completes", move || {
            held.finish().unwrap();
            for submission in &submissions {
                submission.finish().unwrap();
            }
        });
    }

    /// How the keeper's wait on the park timeline (`HostTimeline::wait`,
    /// `vkWaitSemaphores`) spends its thread while the device works toward
    /// the value, by how long the device takes: the share of the wait the
    /// thread ran (1.0 is a spin), and how long after the device reached
    /// the value (as a polling thread first reads it) it returned. A
    /// measurement, which asserts nothing of its figures: run it by name
    /// with `--ignored`. Its thread's run time is Linux's
    /// (`/proc/thread-self/schedstat`).
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "a measurement: run it by name with --ignored"]
    fn the_keepers_wait_is_measured() {
        const WAITS: u64 = 100;
        let Ok(devices) = crate::discover() else {
            return;
        };
        // This thread's run time so far (`/proc/thread-self/schedstat`).
        let ran = || -> f64 {
            let text = std::fs::read_to_string("/proc/thread-self/schedstat").expect("schedstat");
            text.split_whitespace().next().unwrap().parse::<f64>().unwrap() / 1e9
        };
        for description in devices.into_iter().filter(|description| description.floor.is_ok()) {
            let device = Device::open(description.facts.uuid, crate::unbounded_accounts(&description.facts), &Default::default()).expect("a device meeting the floor opens");
            for work in [Duration::from_micros(50), Duration::from_micros(300), Duration::from_millis(2)] {
                let gate = Arc::new(HostTimeline::new(&device).unwrap());
                let reached = Arc::new(HostTimeline::new(&device).unwrap());
                let (mut ran_share, mut late) = (0.0, 0.0);
                for index in 1..=WAITS {
                    let mut batch = DirectBatch::new(&device).unwrap();
                    batch.wait_timeline(&gate, index);
                    batch.signal_timeline(&reached, index);
                    let submission = batch.commit().unwrap();
                    let (raiser, poller) = {
                        let (gate, reached) = (gate.clone(), reached.clone());
                        let raiser = std::thread::spawn(move || {
                            let until = Instant::now() + work;
                            while Instant::now() < until {
                                std::hint::spin_loop();
                            }
                            gate.signal(index).unwrap();
                        });
                        let poller = std::thread::spawn(move || {
                            while reached.value().unwrap() < index {
                                std::hint::spin_loop();
                            }
                            Instant::now()
                        });
                        (raiser, poller)
                    };
                    let (start, cpu) = (Instant::now(), ran());
                    assert!(reached.wait(index, BOUND).unwrap(), "the device reaches {index}");
                    let (returned, spent) = (Instant::now(), ran() - cpu);
                    raiser.join().unwrap();
                    let seen = poller.join().unwrap();
                    ran_share += spent / (returned - start).as_secs_f64();
                    late += returned.saturating_duration_since(seen).as_secs_f64();
                    submission.finish().unwrap();
                }
                eprintln!(
                    "{}: device reaching the value {work:?} after the wait began: the wait ran {:.0}% of it and returned {:.1} µs after a polling thread saw it",
                    description.facts.name,
                    100.0 * ran_share / WAITS as f64,
                    late / WAITS as f64 * 1e6
                );
            }
        }
    }

    /// What a park costs the device's timeline: `PARKS` empty submissions,
    /// each after a park the host answers as the keeper does (a blocking
    /// wait on the timeline, then the answer), against the same submissions
    /// ungated. Prints the line the cost function prices a park at: the
    /// difference per park, which holds the submission boundary, the host's
    /// wake and its answer's submission. A measurement: run it by name with
    /// `--ignored`.
    #[test]
    #[ignore = "a measurement: run it by name with --ignored"]
    fn a_parks_cost_is_measured() {
        const PARKS: u64 = 2000;
        let Ok(devices) = crate::discover() else {
            return;
        };
        for description in devices.into_iter().filter(|description| description.floor.is_ok()) {
            let device = Device::open(description.facts.uuid, crate::unbounded_accounts(&description.facts), &Default::default()).expect("a device meeting the floor opens");
            let run = |parked: bool| {
                let (reached, gate) = (Arc::new(HostTimeline::new(&device).unwrap()), Arc::new(HostTimeline::new(&device).unwrap()));
                let answerer = parked.then(|| {
                    let (reached, gate) = (reached.clone(), gate.clone());
                    std::thread::spawn(move || {
                        for index in 0..PARKS {
                            assert!(reached.wait(2 * index + 1, BOUND).unwrap(), "the device reaches park {index}");
                            gate.signal(2 * index + 2).unwrap();
                        }
                    })
                });
                let start = Instant::now();
                let submissions = (0..=PARKS)
                    .map(|index| {
                        let mut batch = DirectBatch::new(&device).unwrap();
                        if parked && index > 0 {
                            batch.wait_timeline(&gate, 2 * index);
                        }
                        if parked && index < PARKS {
                            batch.signal_timeline(&reached, 2 * index + 1);
                        }
                        batch.commit().unwrap()
                    })
                    .collect::<Vec<_>>();
                submissions.last().unwrap().finish().unwrap();
                let seconds = start.elapsed().as_secs_f64();
                if let Some(answerer) = answerer {
                    answerer.join().unwrap();
                }
                seconds
            };
            // Warm the queue and the driver, then the median of five each.
            run(false);
            run(true);
            let median = |parked: bool| {
                let mut seconds = (0..5).map(|_| run(parked)).collect::<Vec<_>>();
                seconds.sort_by(f64::total_cmp);
                seconds[2]
            };
            let (plain, parked) = (median(false), median(true));
            eprintln!(
                "{}: {PARKS} submissions {:.1} µs each ungated, {:.1} µs each after a park: a park costs {:.1} µs",
                description.facts.name,
                plain / PARKS as f64 * 1e6,
                parked / PARKS as f64 * 1e6,
                (parked - plain) / PARKS as f64 * 1e6
            );
        }
    }

    #[test]
    fn a_park_holds_what_follows_it_on_the_host() {
        for description in crate::test_devices() {
            eprintln!("{}:", description.facts.name);
            let device = Device::open(description.facts.uuid, crate::unbounded_accounts(&description.facts), &Default::default()).expect("a device meeting the floor opens");
            case(&device, false);
            case(&device, true);
        }
    }
}
