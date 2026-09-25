//! Execution of formed modules (§8.3–§8.7), the shape of CUDA's `direct`:
//! batches of launches recorded into one primary command buffer and
//! submitted to the device's queue, graphs recorded once into a secondary
//! command buffer and replayed, and their completion and timing.
//!
//! Every launch reads its arguments from an argument block in upload
//! storage: its buffer addresses, the scalar-result address, then its words.
//! The launch's only push constant is the block's address.

use crate::device::{
    barrier, call, Device, Recording, Retired, RetiredCommand, ARGUMENT_CHUNK_BYTES,
};
use crate::formation::DirectModule;
use crate::memory::{Buffer, Range};
use ash::vk;
use seismic_compiler::errors::ExecutionError;
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
    pub scalar_results: (&'a Buffer, u64),
    pub groups: [u64; 3],
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
        let mut block = Vec::with_capacity((launch.buffers.len() + 1) * 8 + launch.words.len());
        for (buffer, offset) in launch.buffers {
            block.extend_from_slice(&(buffer.address() + offset).to_le_bytes());
        }
        block.extend_from_slice(
            &(launch.scalar_results.0.address() + launch.scalar_results.1).to_le_bytes(),
        );
        block.extend_from_slice(launch.words);
        let bytes = block.len() as u64;
        if bytes > ARGUMENT_CHUNK_BYTES {
            return Err(ExecutionError::SubmissionFailed(format!(
                "a launch's argument block needs {bytes} bytes; a chunk holds {ARGUMENT_CHUNK_BYTES}"
            )));
        }
        let start = self.used.next_multiple_of(16);
        if start + bytes > ARGUMENT_CHUNK_BYTES {
            self.chunks.push(device.argument_chunk(recording)?);
            self.used = 0;
        }
        let start = self.used.next_multiple_of(16);
        let chunk = self.chunks.last().expect("a chunk was taken above");
        chunk.write_mapped(start, &block);
        self.used = start + bytes;
        Ok(chunk.address() + start)
    }
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
    raw.cmd_dispatch(command, groups[0], groups[1], groups[2]);
}

/// Each launch follows the previous launch's writes, as on a CUDA stream.
unsafe fn compute_barrier(raw: &ash::Device, command: vk::CommandBuffer) {
    barrier(
        raw,
        command,
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_WRITE,
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
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
/// their argument blocks fixed.
pub struct DirectGraphBuilder {
    device: Device,
    /// Becomes the graph; dropping an abandoned builder retires it (and the
    /// chunks it wrote) unsubmitted.
    graph: Option<GraphInner>,
    arguments: Arguments,
    launched: bool,
}

impl DirectGraphBuilder {
    pub fn new(device: &Device) -> Result<Self, ExecutionError> {
        let mut recording = device.recording();
        let command = device.command(&mut recording, vk::CommandBufferLevel::SECONDARY)?;
        let builder = Self {
            device: device.clone(),
            graph: Some(GraphInner {
                device: device.clone(),
                command,
                arguments: Vec::new(),
                last_use: AtomicU64::new(0),
            }),
            arguments: Arguments::new(),
            launched: false,
        };
        let inheritance = vk::CommandBufferInheritanceInfo::default();
        let begun = begin(
            device,
            command,
            vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
            Some(&inheritance),
        );
        // Released before an error drops the builder, whose graph retires
        // under this lock.
        drop(recording);
        begun.map(|()| builder)
    }

    fn command(&self) -> vk::CommandBuffer {
        self.graph
            .as_ref()
            .expect("a live builder holds its graph")
            .command
    }

    /// Append a launch after every launch added before it; an empty grid
    /// adds nothing.
    pub fn launch(&mut self, launch: &DirectLaunch<'_>) -> Result<(), ExecutionError> {
        let Some(groups) = groups(&self.device, launch)? else {
            return Ok(());
        };
        let mut recording = self.device.recording();
        let address = self.arguments.write(&self.device, &mut recording, launch)?;
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
        Ok(())
    }

    pub fn instantiate(mut self) -> Result<DirectGraph, ExecutionError> {
        let mut graph = self.graph.take().expect("a live builder holds its graph");
        graph.arguments = std::mem::take(&mut self.arguments.chunks);
        let recording = self.device.recording();
        let ended = call(
            unsafe { self.device.inner.device.end_command_buffer(graph.command) },
            "vkEndCommandBuffer",
        );
        drop(recording);
        ended.map(|()| DirectGraph {
            inner: std::sync::Arc::new(graph),
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

/// A recorded secondary command buffer of launches with fixed arguments.
/// One replay replaces its launches' individual recording.
pub struct DirectGraph {
    inner: std::sync::Arc<GraphInner>,
}

/// Shared by the graph and the submissions replaying it, so its buffer and
/// argument blocks outlive every replay.
struct GraphInner {
    device: Device,
    command: vk::CommandBuffer,
    arguments: Vec<Range>,
    /// Timeline value of the latest submission that replays it.
    last_use: AtomicU64,
}

impl Drop for GraphInner {
    fn drop(&mut self) {
        self.device.retire(Retired {
            value: self.last_use.load(Ordering::Acquire),
            command: RetiredCommand::Secondary(self.command),
            queries: None,
            arguments: std::mem::take(&mut self.arguments),
        });
    }
}

/// Launches recorded into one primary command buffer and submitted to the
/// device's queue. A leading barrier orders the batch after every earlier
/// submission; each launch follows the previous launch's writes; a trailing
/// barrier makes results visible to the host.
pub struct DirectBatch {
    device: Device,
    command: vk::CommandBuffer,
    queries: (vk::QueryPool, u32),
    /// Queries written so far (the first is the batch start).
    written: u32,
    /// For a timed batch, per `launch` call, the query written just before
    /// the launch (absent for an empty launch).
    marks: Option<Vec<Option<u32>>>,
    arguments: Arguments,
    /// Whether a dispatch or replay has been recorded (the next one needs a
    /// barrier).
    launched: bool,
    /// Graphs replayed by this batch; the submission keeps them alive and
    /// records itself as their last use.
    replayed: Vec<std::sync::Arc<GraphInner>>,
    /// Set by `commit`; an uncommitted batch returns its resources on drop.
    committed: bool,
}

impl DirectBatch {
    pub fn new(device: &Device) -> Result<Self, ExecutionError> {
        Self::create(device, 2, None)
    }

    /// A measurement batch that writes a timestamp before every launch, so
    /// each launch's device interval is known; `launches` bounds the calls
    /// of `launch` and `skip`. A timed batch attributes time but does not
    /// measure production, and never replays.
    pub fn timed(device: &Device, launches: usize) -> Result<Self, ExecutionError> {
        let queries = u32::try_from(launches + 2)
            .map_err(|_| ExecutionError::SubmissionFailed("too many timed launches".into()))?;
        Self::create(device, queries, Some(Vec::with_capacity(launches)))
    }

    fn create(
        device: &Device,
        queries: u32,
        marks: Option<Vec<Option<u32>>>,
    ) -> Result<Self, ExecutionError> {
        let mut recording = device.recording();
        let command = device.command(&mut recording, vk::CommandBufferLevel::PRIMARY)?;
        let pool = match device.queries(&mut recording, queries) {
            Ok(pool) => pool,
            Err(error) => {
                recording.free_command(command);
                return Err(error);
            }
        };
        let mut batch = Self {
            device: device.clone(),
            command,
            queries: pool,
            written: 0,
            marks,
            arguments: Arguments::new(),
            launched: false,
            replayed: Vec::new(),
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
            unsafe {
                raw.cmd_reset_query_pool(command, pool.0, 0, pool.1);
                barrier(
                    raw,
                    command,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    vk::AccessFlags2::MEMORY_WRITE,
                    vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                );
            }
            batch.timestamp();
        }
        // Released before an error drops the batch, which retires under it.
        drop(recording);
        begun.map(|()| batch)
    }

    /// Write the next timestamp; the recording lock is held.
    fn timestamp(&mut self) -> u32 {
        let query = self.written;
        unsafe {
            self.device.inner.device.cmd_write_timestamp2(
                self.command,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                self.queries.0,
                query,
            )
        };
        self.written += 1;
        query
    }

    /// Record a launch that does no device work (inactive, or an empty
    /// grid): nothing is recorded, and a timed batch keeps its place with no
    /// interval.
    pub fn skip(&mut self) {
        if let Some(marks) = &mut self.marks {
            marks.push(None);
        }
    }

    pub fn launch(&mut self, launch: &DirectLaunch<'_>) -> Result<(), ExecutionError> {
        let Some(groups) = groups(&self.device, launch)? else {
            self.skip();
            return Ok(());
        };
        let device = self.device.clone();
        let mut recording = device.recording();
        let address = self.arguments.write(&device, &mut recording, launch)?;
        if self.marks.is_some() {
            if self.launched {
                unsafe { compute_barrier(&device.inner.device, self.command) };
            }
            let mark = self.timestamp();
            self.marks
                .as_mut()
                .expect("a timed batch has marks")
                .push(Some(mark));
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

    /// Execute every launch of `graph` in its order.
    pub fn replay(&mut self, graph: &DirectGraph) -> Result<(), ExecutionError> {
        assert!(
            self.marks.is_none(),
            "DirectBatch::replay precondition: a timed batch times each launch"
        );
        let _recording = self.device.recording();
        let raw = &self.device.inner.device;
        unsafe {
            if self.launched {
                compute_barrier(raw, self.command);
            }
            raw.cmd_execute_commands(self.command, &[graph.inner.command]);
        }
        self.launched = true;
        self.replayed.push(graph.inner.clone());
        Ok(())
    }

    /// Submit without waiting.
    pub fn commit(mut self) -> Result<DirectSubmission, ExecutionError> {
        let device = self.device.clone();
        let recording = device.recording();
        let raw = &device.inner.device;
        unsafe {
            barrier(
                raw,
                self.command,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::HOST | vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::HOST_READ
                    | vk::AccessFlags2::MEMORY_READ
                    | vk::AccessFlags2::MEMORY_WRITE,
            );
        }
        let end = self.timestamp();
        let ended = call(
            unsafe { raw.end_command_buffer(self.command) },
            "vkEndCommandBuffer",
        );
        drop(recording);
        ended?;
        let value = self.device.submit(self.command)?;
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
                queries: Some(self.queries),
                arguments: std::mem::take(&mut self.arguments.chunks),
            }),
            _graphs: std::mem::take(&mut self.replayed),
            end,
            marks: self.marks.take(),
        })
    }
}

impl Drop for DirectBatch {
    fn drop(&mut self) {
        if !self.committed {
            // Nothing was submitted: everything is reusable at once.
            self.device.retire(Retired {
                value: 0,
                command: RetiredCommand::Primary(self.command),
                queries: Some(self.queries),
                arguments: std::mem::take(&mut self.arguments.chunks),
            });
        }
    }
}

/// A batch's device timestamps paired with the host clock: the origin that
/// places device times on the host timeline. Recorded on the idle queue,
/// waited on, and paired with the host clock read just after.
pub struct TimelineAnchor {
    ticks: u64,
    host_seconds: f64,
    period: f64,
    mask: u64,
}

impl TimelineAnchor {
    pub fn record(
        device: &Device,
        host_clock: impl FnOnce() -> f64,
    ) -> Result<Self, ExecutionError> {
        let batch = DirectBatch::new(device)?;
        let submission = batch.commit()?;
        submission.finish()?;
        let host_seconds = host_clock();
        let ticks = submission.timestamps(2)?[1];
        let limits = device.facts().limits;
        Ok(Self {
            ticks,
            host_seconds,
            period: limits.timestamp_period,
            mask: mask(limits.timestamp_valid_bits),
        })
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

/// Submitted launches, observed through the device's timeline semaphore.
pub struct DirectSubmission {
    device: Device,
    value: u64,
    /// Returned to the device for reuse when this submission is dropped.
    retired: Option<Retired>,
    /// The graphs it replays, alive until it is.
    _graphs: Vec<std::sync::Arc<GraphInner>>,
    /// Query of the batch's end timestamp (the start is query 0).
    end: u32,
    marks: Option<Vec<Option<u32>>>,
}

impl DirectSubmission {
    pub fn is_complete(&self) -> bool {
        self.device.completed() >= self.value
    }

    pub fn wait_complete(&self) {
        let _ = self.device.wait(self.value);
    }

    pub fn finish(&self) -> Result<(), ExecutionError> {
        self.device.wait(self.value)
    }

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
        let values = self.timestamps(self.end + 1)?;
        let limits = self.device.facts().limits;
        let ticks =
            values[self.end as usize].wrapping_sub(values[0]) & mask(limits.timestamp_valid_bits);
        Ok(ticks as f64 * limits.timestamp_period / 1e9)
    }

    /// The completed batch's device interval on the anchor's host clock.
    pub fn device_interval(&self, anchor: &TimelineAnchor) -> Result<(f64, f64), ExecutionError> {
        let values = self.timestamps(self.end + 1)?;
        Ok((
            anchor.place(values[0]),
            anchor.place(values[self.end as usize]),
        ))
    }

    /// For a timed batch, each `launch` call's device interval on the
    /// anchor's host clock, in launch order (`None` for an empty launch). A
    /// launch ends where the next recorded launch (or the batch) ends.
    /// `None` for a production batch.
    pub fn launch_intervals(
        &self,
        anchor: &TimelineAnchor,
    ) -> Result<Option<Vec<Option<(f64, f64)>>>, ExecutionError> {
        let Some(marks) = &self.marks else {
            return Ok(None);
        };
        let values = self.timestamps(self.end + 1)?;
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
                .unwrap_or(self.end);
            intervals.push(Some((
                anchor.place(values[*mark as usize]),
                anchor.place(values[end as usize]),
            )));
        }
        Ok(Some(intervals))
    }
}

impl Drop for DirectSubmission {
    fn drop(&mut self) {
        if let Some(retired) = self.retired.take() {
            self.device.retire(retired);
        }
    }
}
