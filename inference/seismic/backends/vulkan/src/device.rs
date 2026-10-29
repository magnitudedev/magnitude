//! One opened Vulkan device (§8.1–§8.5): the logical device with exactly
//! the floor's features plus the gated ones it has, one queue (compute-only
//! when the device has such a family), a timeline semaphore that numbers
//! every submission, the block allocator, and host access to storage.

use crate::facts::{self, Description, Facts};
use crate::instance::{Instance, LoaderError};
use crate::memory::{Buffer, Memory, Range, StorageKind};
use ash::vk;
use seismic_compiler::errors::ExecutionError;
use std::ffi::CStr;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

/// Map a Vulkan call's failure: device loss is its own error.
pub(crate) fn call<T>(result: Result<T, vk::Result>, what: &str) -> Result<T, ExecutionError> {
    result.map_err(|error| match error {
        vk::Result::ERROR_DEVICE_LOST => ExecutionError::DeviceLost(format!("{what}: {error}")),
        other => ExecutionError::SubmissionFailed(format!("{what} failed: {other}")),
    })
}

/// Why a device could not be opened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpenError {
    Loader(LoaderError),
    /// No physical device has the requested UUID any more.
    Missing,
    /// The device does not meet the floor.
    Floor(String),
    /// Creating the device or one of its objects failed.
    Creation(String),
}

impl OpenError {
    /// Why the device does not meet the floor, when that is the error: the
    /// reason discovery reports as the device's unavailability.
    pub fn floor_reason(&self) -> Option<String> {
        match self {
            Self::Floor(reason) => Some(reason.clone()),
            Self::Loader(_) | Self::Missing | Self::Creation(_) => None,
        }
    }
}

impl fmt::Display for OpenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Loader(error) => write!(f, "{error}"),
            Self::Missing => f.write_str("the Vulkan device is no longer present"),
            Self::Creation(reason) => write!(f, "opening the Vulkan device failed: {reason}"),
            Self::Floor(reason) => write!(f, "the Vulkan device does not meet the floor: {reason}"),
        }
    }
}

impl std::error::Error for OpenError {}

/// Staging of transfers to and from unmapped device storage.
struct Staging {
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    range: Range,
}

/// Bytes of the staging range; larger transfers go in pieces.
const STAGING_BYTES: u64 = 16 << 20;

/// Resources of one submission (or of a dropped graph), reusable once the
/// timeline reaches `value`.
pub(crate) struct Retired {
    pub(crate) value: u64,
    pub(crate) command: RetiredCommand,
    pub(crate) queries: Option<(vk::QueryPool, u32)>,
    pub(crate) arguments: Vec<Range>,
}

pub(crate) enum RetiredCommand {
    /// A batch's primary buffer, pooled for reuse.
    Primary(vk::CommandBuffer),
    /// A graph's secondary buffer, freed.
    Secondary(vk::CommandBuffer),
}

/// Submission-side state recorded under one lock: command buffers, query
/// pools and argument chunks of finished submissions awaiting reuse.
pub(crate) struct Recording {
    pub(crate) pool: vk::CommandPool,
    free_commands: Vec<vk::CommandBuffer>,
    free_queries: Vec<(vk::QueryPool, u32)>,
    free_chunks: Vec<Range>,
    retired: Vec<Retired>,
}

impl Recording {
    /// Return an unused primary command buffer.
    pub(crate) fn free_command(&mut self, command: vk::CommandBuffer) {
        self.free_commands.push(command);
    }
}

/// Bytes of one argument chunk.
pub(crate) const ARGUMENT_CHUNK_BYTES: u64 = 256 << 10;

pub(crate) struct Queue {
    raw: vk::Queue,
    next: u64,
}

pub(crate) struct Inner {
    pub(crate) device: ash::Device,
    pub(crate) facts: Facts,
    pub(crate) memory: Memory,
    /// The tables buffer (`seismic_lang::source::VULKAN_TABLES`), every
    /// launch's second push constant.
    pub(crate) tables: Range,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) cache: vk::PipelineCache,
    pub(crate) timeline: vk::Semaphore,
    queue: Mutex<Queue>,
    recording: Mutex<Recording>,
    staging: Mutex<Option<Staging>>,
    /// Device memory the driver took for this device outside every block:
    /// the logical device itself, pipelines, shader scratch.
    driver: std::sync::atomic::AtomicU64,
    driver_account: Mutex<Option<std::sync::Arc<dyn seismic_compiler::executable::MemoryAccount>>>,
    driver_holds: Mutex<Vec<Box<dyn seismic_compiler::executable::MemoryHold>>>,
    /// Device memory recorded graphs took (their command memory), measured
    /// while each is recorded; the device keeps it for the command buffers
    /// it reuses.
    pub(crate) graphs: std::sync::atomic::AtomicU64,
    physical: vk::PhysicalDevice,
    instance: &'static Instance,
}

/// An opened Vulkan device.
#[derive(Clone)]
pub struct Device {
    pub(crate) inner: Arc<Inner>,
}

/// Memory-budget sample of the device-local heap (`VK_EXT_memory_budget`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBudget {
    pub heap_budget_bytes: u64,
    pub heap_usage_bytes: u64,
}

/// The process's Vulkan devices: the one logical device of each opened
/// physical device.
#[derive(Default)]
struct Registry {
    opened: Vec<Device>,
}

fn registry() -> MutexGuard<'static, Registry> {
    static REGISTRY: std::sync::OnceLock<Mutex<Registry>> = std::sync::OnceLock::new();
    REGISTRY
        .get_or_init(Mutex::default)
        .lock()
        .expect("the device registry lock is never poisoned")
}

impl Inner {
    /// This process's usage of the heaps device storage is allocated from.
    fn heap_usage(&self) -> u64 {
        memory_budget(self.instance, self.physical, &self.facts).heap_usage_bytes
    }

    /// Run a driver operation that may take device memory no allocation
    /// charges (pipeline creation; recording and submitting work, for which
    /// the driver may grow its command memory and shader scratch), counting
    /// the growth of the process's heap usage across it, less blocks it
    /// created, as driver memory the device holds.
    pub(crate) fn measured<R>(&self, operation: impl FnOnce() -> R) -> R {
        thread_local! {
            static MEASURING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        }
        // An operation measured inside another is part of the outer one.
        if MEASURING.with(|measuring| measuring.replace(true)) {
            return operation();
        }
        let start = self.unheld_mark();
        let result = operation();
        self.record_driver_memory(&self.driver, self.unheld_since(start));
        MEASURING.with(|measuring| measuring.set(false));
        result
    }

    /// Count `bytes` the driver took for this device in `counter` (`driver`,
    /// or `graphs` for a recorded graph) and charge them to the driver
    /// memory account, held until the device is destroyed. The driver has
    /// already taken them, so they are recorded, never refused; before an
    /// account is set they are only counted, and the account takes them
    /// when it is set. Both happen under the account's lock, so an account
    /// set meanwhile takes them once.
    pub(crate) fn record_driver_memory(&self, counter: &std::sync::atomic::AtomicU64, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let account = self
            .driver_account
            .lock()
            .expect("driver account lock poisoned");
        counter.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
        if let Some(account) = account.as_ref() {
            self.driver_holds
                .lock()
                .expect("driver accounting lock poisoned")
                .push(account.record(bytes));
        }
    }

    /// The process's heap usage with the blocks and recorded graphs this
    /// device holds, to measure what an operation takes beyond them.
    pub(crate) fn unheld_mark(&self) -> (u64, u64) {
        (
            self.heap_usage(),
            self.memory.device_heap_block_bytes()
                + self.graphs.load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Growth of the heap usage since `mark` that neither a new block nor a
    /// recorded graph accounts for.
    pub(crate) fn unheld_since(&self, (usage, held): (u64, u64)) -> u64 {
        let grown = self.heap_usage().saturating_sub(usage);
        let held = (self.memory.device_heap_block_bytes()
            + self.graphs.load(std::sync::atomic::Ordering::Relaxed))
        .saturating_sub(held);
        grown.saturating_sub(held)
    }
}

impl Device {
    /// The account the device's staging blocks are charged to: host RAM's.
    /// The logical device is the process's, so a later open replaces the
    /// account for later blocks; existing blocks keep their charges.
    pub fn set_host_memory_account(
        &self,
        account: Arc<dyn seismic_compiler::executable::MemoryAccount>,
    ) {
        self.inner.memory.set_host_memory_account(account);
    }
    /// Open the physical device with this UUID, revalidating it against
    /// the floor.
    ///
    /// A physical device has one logical device for the process, created by
    /// the first open and shared by every later one, like CUDA's primary
    /// context: NVIDIA 580 deadlocks when one logical device is destroyed
    /// while another compiles pipelines, which independent opens in one
    /// process would do.
    pub fn open(uuid: [u8; 16]) -> Result<Self, OpenError> {
        let mut registry = registry();
        if let Some(device) = registry
            .opened
            .iter()
            .find(|device| device.facts().uuid == uuid)
        {
            return Ok(device.clone());
        }
        let instance = crate::instance::instance().map_err(OpenError::Loader)?;
        let (physical, Description { facts, floor, .. }) = instance
            .physical_devices()
            .map_err(OpenError::Loader)?
            .into_iter()
            .map(|physical| (physical, facts::describe(instance, physical)))
            .find(|(_, description)| description.facts.uuid == uuid)
            .ok_or(OpenError::Missing)?;
        floor.map_err(OpenError::Floor)?;
        let device = Self {
            inner: Arc::new(Inner::create(instance, physical, facts)?),
        };
        device
            .write_tables()
            .map_err(|error| OpenError::Creation(error.to_string()))?;
        registry.opened.push(device.clone());
        Ok(device)
    }

    pub fn facts(&self) -> &Facts {
        &self.inner.facts
    }

    /// Device-local storage. `room`, when given, bounds the device usage the
    /// allocation may add, block slack included.
    pub fn allocate(
        &self,
        bytes: u64,
        alignment: u64,
        room: Option<u64>,
    ) -> Result<Buffer, ExecutionError> {
        self.allocate_kind(StorageKind::Device, bytes, alignment, room)
    }

    /// Storage the host writes before each submission with plain memory
    /// writes that never wait for device work; `room` as for
    /// [`Device::allocate`].
    pub fn allocate_upload(
        &self,
        bytes: u64,
        alignment: u64,
        room: Option<u64>,
    ) -> Result<Buffer, ExecutionError> {
        self.allocate_kind(StorageKind::Upload, bytes, alignment, room)
    }

    fn allocate_kind(
        &self,
        kind: StorageKind,
        bytes: u64,
        alignment: u64,
        room: Option<u64>,
    ) -> Result<Buffer, ExecutionError> {
        let range = self
            .inner
            .memory
            .allocate(&self.inner.device, kind, bytes, alignment, room)?;
        Ok(Buffer::new(self.inner.clone(), range))
    }

    /// The accounts new device and upload blocks are charged to, by storage
    /// kind and heap (the topology's access paths). Replaces the accounts
    /// for later blocks; existing blocks keep their charges.
    pub fn set_allocation_accounts(&self, accounts: Vec<crate::AllocationAccount>) {
        self.inner.memory.set_allocation_accounts(accounts);
    }

    /// The account the driver's memory for this device is recorded to. The
    /// first account set also takes what the driver and recorded graphs took
    /// before it.
    pub fn set_driver_memory_account(
        &self,
        account: Arc<dyn seismic_compiler::executable::MemoryAccount>,
    ) {
        let mut slot = self
            .inner
            .driver_account
            .lock()
            .expect("driver account lock poisoned");
        let first = slot.is_none();
        let account = slot.insert(account);
        // What was counted before, read under the lock every count is made
        // under, so nothing is taken twice.
        let counted = self.inner.driver.load(std::sync::atomic::Ordering::Relaxed)
            + self.inner.graphs.load(std::sync::atomic::Ordering::Relaxed);
        if first && counted != 0 {
            self.inner
                .driver_holds
                .lock()
                .expect("driver accounting lock poisoned")
                .push(account.record(counted));
        }
    }

    /// Host write: through the mapping when the storage is host-visible,
    /// else through staging and a transfer on the queue, waited on. The
    /// caller orders it after device work using the storage.
    pub fn write(&self, buffer: &Buffer, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError> {
        let range = buffer.range();
        if range.mapped().is_some() {
            range.write_mapped(offset, bytes);
            return Ok(());
        }
        let mut done = 0usize;
        while done < bytes.len() {
            let length = (bytes.len() - done).min(STAGING_BYTES as usize);
            let _staging = self.transfer(
                |staging| {
                    staging.write_mapped(0, &bytes[done..done + length]);
                    vk::BufferCopy {
                        src_offset: staging.offset(),
                        dst_offset: range.offset() + offset + done as u64,
                        size: length as u64,
                    }
                },
                range,
                true,
            )?;
            done += length;
        }
        Ok(())
    }

    /// Fill the tables buffer, before the device is shared: by
    /// `vkCmdUpdateBuffer`, which takes no staging (a staging block taken
    /// before the device has its accounts would be charged to none).
    fn write_tables(&self) -> Result<(), ExecutionError> {
        /// What one `vkCmdUpdateBuffer` writes at most.
        const UPDATE_BYTES: usize = 65536;
        let tables = &self.inner.tables;
        let device = &self.inner.device;
        let command = {
            let mut recording = self.recording();
            self.command(&mut recording, vk::CommandBufferLevel::PRIMARY)?
        };
        unsafe {
            call(
                device.begin_command_buffer(
                    command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                ),
                "vkBeginCommandBuffer",
            )?;
            for (index, chunk) in seismic_lang::source::VULKAN_TABLES
                .chunks(UPDATE_BYTES)
                .enumerate()
            {
                device.cmd_update_buffer(
                    command,
                    tables.raw(),
                    tables.offset() + (index * UPDATE_BYTES) as u64,
                    chunk,
                );
            }
            barrier(
                device,
                command,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_READ,
            );
            call(device.end_command_buffer(command), "vkEndCommandBuffer")?;
        }
        let value = self.submit(command)?;
        self.wait(value)?;
        self.recording().free_command(command);
        Ok(())
    }

    /// Host read; see [`Device::write`].
    pub fn read(
        &self,
        buffer: &Buffer,
        offset: u64,
        into: &mut [u8],
    ) -> Result<(), ExecutionError> {
        let range = buffer.range();
        if range.mapped().is_some() {
            range.read_mapped(offset, into);
            return Ok(());
        }
        let mut done = 0usize;
        while done < into.len() {
            let length = (into.len() - done).min(STAGING_BYTES as usize);
            let staging = self.transfer(
                |staging| vk::BufferCopy {
                    src_offset: range.offset() + offset + done as u64,
                    dst_offset: staging.offset(),
                    size: length as u64,
                },
                range,
                false,
            )?;
            staging
                .as_ref()
                .expect("a transfer leaves its staging in place")
                .range
                .read_mapped(0, &mut into[done..done + length]);
            done += length;
        }
        Ok(())
    }

    /// One staged copy between `range` and the staging range, submitted and
    /// waited on. `upload` copies staging to `range`, else `range` to
    /// staging. Returns the staging lock so a read can copy out.
    fn transfer(
        &self,
        region: impl FnOnce(&Range) -> vk::BufferCopy,
        range: &Range,
        upload: bool,
    ) -> Result<MutexGuard<'_, Option<Staging>>, ExecutionError> {
        let inner = &self.inner;
        let device = &inner.device;
        let mut guard = inner
            .staging
            .lock()
            .expect("staging lock is never poisoned");
        if guard.is_none() {
            *guard = Some(inner.create_staging()?);
        }
        let staging = guard.as_mut().expect("staging was created above");
        let copy = region(&staging.range);
        let (source, destination) = if upload {
            (staging.range.raw(), range.raw())
        } else {
            (range.raw(), staging.range.raw())
        };
        let command = staging.command;
        unsafe {
            call(
                device.reset_command_buffer(command, vk::CommandBufferResetFlags::empty()),
                "vkResetCommandBuffer",
            )?;
            call(
                device.begin_command_buffer(
                    command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                ),
                "vkBeginCommandBuffer",
            )?;
            barrier(
                device,
                command,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
            );
            device.cmd_copy_buffer(command, source, destination, &[copy]);
            barrier(
                device,
                command,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::HOST | vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::HOST_READ
                    | vk::AccessFlags2::MEMORY_READ
                    | vk::AccessFlags2::MEMORY_WRITE,
            );
            call(device.end_command_buffer(command), "vkEndCommandBuffer")?;
        }
        let value = self.submit(command)?;
        self.wait(value)?;
        Ok(guard)
    }

    /// Submit `command`, signalling the timeline with the next value.
    pub(crate) fn submit(&self, command: vk::CommandBuffer) -> Result<u64, ExecutionError> {
        self.submit_with(command, &[], &[])
    }

    /// Submit `command` once every timeline semaphore of `waits` reaches its
    /// value, signalling the device timeline with the next value and each
    /// of `signals` with its own once the command completes.
    pub(crate) fn submit_with(
        &self,
        command: vk::CommandBuffer,
        waits: &[(vk::Semaphore, u64)],
        signals: &[(vk::Semaphore, u64)],
    ) -> Result<u64, ExecutionError> {
        let inner = &self.inner;
        let mut queue = inner.queue.lock().expect("queue lock is never poisoned");
        let value = queue.next;
        let semaphore = |stages: vk::PipelineStageFlags2| {
            move |(semaphore, value): (vk::Semaphore, u64)| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(semaphore)
                    .value(value)
                    .stage_mask(stages)
            }
        };
        let signal = std::iter::once((inner.timeline, value))
            .chain(signals.iter().copied())
            .map(semaphore(vk::PipelineStageFlags2::ALL_COMMANDS))
            .collect::<Vec<_>>();
        // A host gate's wait: no stage of the batch, the compute shaders
        // that read what the host wrote before signaling it included,
        // starts before it (spec, "Semaphore Waiting": the wait's second
        // synchronization scope is limited to the stages of its mask).
        let wait = waits
            .iter()
            .copied()
            .map(semaphore(
                vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::ALL_COMMANDS,
            ))
            .collect::<Vec<_>>();
        let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(command)];
        let submit = vk::SubmitInfo2::default()
            .command_buffer_infos(&commands)
            .wait_semaphore_infos(&wait)
            .signal_semaphore_infos(&signal);
        inner.measured(|| {
            call(
                unsafe {
                    inner
                        .device
                        .queue_submit2(queue.raw, &[submit], vk::Fence::null())
                },
                "vkQueueSubmit2",
            )
        })?;
        queue.next += 1;
        Ok(value)
    }

    /// The device timeline value of the latest submission.
    pub(crate) fn submitted(&self) -> u64 {
        self.inner
            .queue
            .lock()
            .expect("queue lock is never poisoned")
            .next
            - 1
    }

    /// The timeline value every completed submission has reached. A failed
    /// query means the work cannot make progress: it reads as complete.
    pub(crate) fn completed(&self) -> u64 {
        unsafe {
            self.inner
                .device
                .get_semaphore_counter_value(self.inner.timeline)
        }
        .unwrap_or(u64::MAX)
    }

    pub(crate) fn wait(&self, value: u64) -> Result<(), ExecutionError> {
        let semaphores = [self.inner.timeline];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        call(
            unsafe { self.inner.device.wait_semaphores(&info, u64::MAX) },
            "vkWaitSemaphores",
        )
    }

    pub(crate) fn recording(&self) -> MutexGuard<'_, Recording> {
        let mut recording = self
            .inner
            .recording
            .lock()
            .expect("recording lock is never poisoned");
        let completed = self.completed();
        let mut index = 0;
        while index < recording.retired.len() {
            if recording.retired[index].value <= completed {
                let retired = recording.retired.swap_remove(index);
                match retired.command {
                    RetiredCommand::Primary(command) => recording.free_commands.push(command),
                    RetiredCommand::Secondary(command) => unsafe {
                        self.inner
                            .device
                            .free_command_buffers(recording.pool, &[command])
                    },
                }
                recording.free_queries.extend(retired.queries);
                recording.free_chunks.extend(retired.arguments);
            } else {
                index += 1;
            }
        }
        recording
    }

    /// A primary command buffer, reset and ready to begin.
    pub(crate) fn command(
        &self,
        recording: &mut Recording,
        level: vk::CommandBufferLevel,
    ) -> Result<vk::CommandBuffer, ExecutionError> {
        if level == vk::CommandBufferLevel::PRIMARY {
            if let Some(command) = recording.free_commands.pop() {
                call(
                    unsafe {
                        self.inner
                            .device
                            .reset_command_buffer(command, vk::CommandBufferResetFlags::empty())
                    },
                    "vkResetCommandBuffer",
                )?;
                return Ok(command);
            }
        }
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(recording.pool)
            .level(level)
            .command_buffer_count(1);
        call(
            unsafe { self.inner.device.allocate_command_buffers(&info) },
            "vkAllocateCommandBuffers",
        )
        .map(|commands| commands[0])
    }

    /// A timestamp query pool of at least `queries` queries.
    pub(crate) fn queries(
        &self,
        recording: &mut Recording,
        queries: u32,
    ) -> Result<(vk::QueryPool, u32), ExecutionError> {
        if let Some(index) = recording
            .free_queries
            .iter()
            .position(|(_, capacity)| *capacity >= queries)
        {
            return Ok(recording.free_queries.swap_remove(index));
        }
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(queries);
        call(
            unsafe { self.inner.device.create_query_pool(&info, None) },
            "vkCreateQueryPool",
        )
        .map(|pool| (pool, queries))
    }

    /// An argument chunk in upload storage.
    pub(crate) fn argument_chunk(
        &self,
        recording: &mut Recording,
    ) -> Result<Range, ExecutionError> {
        match recording.free_chunks.pop() {
            Some(chunk) => Ok(chunk),
            None => self.inner.memory.allocate_internal(
                &self.inner.device,
                StorageKind::Upload,
                ARGUMENT_CHUNK_BYTES,
                256,
            ),
        }
    }

    pub(crate) fn retire(&self, retired: Retired) {
        self.inner
            .recording
            .lock()
            .expect("recording lock is never poisoned")
            .retired
            .push(retired);
    }

    /// Record and submit device work with `operation`, counting the growth
    /// of the process's heap usage across it that no block accounts for as
    /// driver memory the device holds (command memory, shader scratch).
    pub fn measured<R>(&self, operation: impl FnOnce() -> R) -> R {
        self.inner.measured(operation)
    }

    /// What the blocks of `storage` hold beyond their ranges: held by this
    /// process, charged to no allocation, and free for later allocations.
    pub fn unallocated_bytes(&self, storage: crate::PlacedStorage) -> u64 {
        self.inner.memory.free_bytes(storage.into())
    }

    /// What the device holds for `storage` beyond its allocations: its
    /// blocks' free space and the device's own ranges in them (argument
    /// chunks, in upload storage) and, for device storage, the driver's
    /// memory for the device and its recorded graphs (measured; see
    /// `Inner::measured`), which the driver memory account charges there.
    pub fn held_beyond_allocations(&self, storage: crate::PlacedStorage) -> u64 {
        let memory = &self.inner.memory;
        let kind: StorageKind = storage.into();
        let driver = match storage {
            crate::PlacedStorage::Device => {
                self.inner.driver.load(std::sync::atomic::Ordering::Relaxed)
                    + self.inner.graphs.load(std::sync::atomic::Ordering::Relaxed)
            }
            crate::PlacedStorage::Upload => 0,
        };
        memory.free_bytes(kind) + memory.internal_bytes(kind) + driver
    }

    /// `VK_EXT_memory_budget` of the heaps device storage is allocated
    /// from.
    pub fn memory_budget(&self) -> MemoryBudget {
        memory_budget(self.inner.instance, self.inner.physical, &self.inner.facts)
    }
}

/// `VK_EXT_memory_budget` of the heaps `facts`' device storage is allocated
/// from, together. The budget is what the driver grants this process of the
/// heaps, its own usage included.
pub(crate) fn memory_budget(
    instance: &Instance,
    physical: vk::PhysicalDevice,
    facts: &Facts,
) -> MemoryBudget {
    let budgets = heap_budgets(instance, physical);
    crate::Allocation::heaps(&facts.memory, &facts.allocation.device)
        .iter()
        .map(|&heap| budgets[heap as usize])
        .fold(
            MemoryBudget {
                heap_budget_bytes: 0,
                heap_usage_bytes: 0,
            },
            |sum, heap| MemoryBudget {
                heap_budget_bytes: sum.heap_budget_bytes + heap.budget,
                heap_usage_bytes: sum.heap_usage_bytes + heap.usage,
            },
        )
}

/// One heap's `VK_EXT_memory_budget`: what the driver grants this process
/// of the heap, and what the process uses of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeapBudget {
    /// `heapBudget`: what the driver grants this process of the heap, its
    /// own usage included.
    pub budget: u64,
    /// `heapUsage`: what this process uses of the heap.
    pub usage: u64,
}

/// Every heap's budget of `physical`, a device meeting the floor (which
/// requires the extension); a physical-device query needs no logical
/// device.
pub(crate) fn heap_budgets(instance: &Instance, physical: vk::PhysicalDevice) -> Vec<HeapBudget> {
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut properties = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
    unsafe {
        instance
            .raw()
            .get_physical_device_memory_properties2(physical, &mut properties)
    };
    (0..properties.memory_properties.memory_heap_count as usize)
        .map(|heap| HeapBudget {
            budget: budget.heap_budget[heap],
            usage: budget.heap_usage[heap],
        })
        .collect()
}

/// One global memory barrier.
pub(crate) unsafe fn barrier(
    device: &ash::Device,
    command: vk::CommandBuffer,
    source_stage: vk::PipelineStageFlags2,
    source_access: vk::AccessFlags2,
    destination_stage: vk::PipelineStageFlags2,
    destination_access: vk::AccessFlags2,
) {
    let barriers = [vk::MemoryBarrier2::default()
        .src_stage_mask(source_stage)
        .src_access_mask(source_access)
        .dst_stage_mask(destination_stage)
        .dst_access_mask(destination_access)];
    device.cmd_pipeline_barrier2(
        command,
        &vk::DependencyInfo::default().memory_barriers(&barriers),
    );
}

fn creation<T>(result: Result<T, vk::Result>, what: &str) -> Result<T, OpenError> {
    result.map_err(|error| OpenError::Creation(format!("{what} failed: {error}")))
}

impl Inner {
    fn create(
        instance: &'static Instance,
        physical: vk::PhysicalDevice,
        facts: Facts,
    ) -> Result<Self, OpenError> {
        let usage_before = memory_budget(instance, physical, &facts).heap_usage_bytes;
        let raw = instance.raw();
        let mut extensions: Vec<&CStr> = facts::FLOOR_EXTENSIONS.to_vec();
        if facts.matrix {
            extensions.push(ash::khr::cooperative_matrix::NAME);
        }
        if facts.f32_atomic_add {
            extensions.push(ash::ext::shader_atomic_float::NAME);
        }
        if facts.shader_fma.float32 {
            extensions.push(facts::SHADER_FMA_EXTENSION);
        }
        let extension_names = extensions
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();
        let mut shader_fma = facts::ShaderFmaFeatures::default();
        shader_fma.float16 = facts.shader_fma.float16.into();
        shader_fma.float32 = facts.shader_fma.float32.into();
        shader_fma.float64 = facts.shader_fma.float64.into();

        let mut explicit = vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR::default()
            .workgroup_memory_explicit_layout(true)
            .workgroup_memory_explicit_layout_scalar_block_layout(true)
            .workgroup_memory_explicit_layout8_bit_access(true)
            .workgroup_memory_explicit_layout16_bit_access(true);
        let mut cooperative =
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        let mut atomic_float = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default()
            .shader_buffer_float32_atomic_add(true);
        let mut f11 =
            vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .shader_int8(true)
            .shader_float16(facts.float16)
            .storage_buffer8_bit_access(true)
            .scalar_block_layout(true)
            .timeline_semaphore(true)
            .vulkan_memory_model(true)
            .vulkan_memory_model_device_scope(true)
            .shader_subgroup_extended_types(true)
            .shader_shared_int64_atomics(facts.shared_int64_atomics);
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(true)
            .maintenance4(true)
            .pipeline_creation_cache_control(true)
            .subgroup_size_control(true)
            .compute_full_subgroups(true)
            .shader_integer_dot_product(true);
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .features(
                vk::PhysicalDeviceFeatures::default()
                    .shader_int64(true)
                    .shader_int16(true),
            )
            .push_next(&mut f11)
            .push_next(&mut f12)
            .push_next(&mut f13)
            .push_next(&mut explicit);
        if facts.matrix {
            features = features.push_next(&mut cooperative);
        }
        if facts.f32_atomic_add {
            features = features.push_next(&mut atomic_float);
        }
        if facts.shader_fma.float32 {
            features = features.push_next(&mut shader_fma);
        }
        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(facts.queue_family)
            .queue_priorities(&priorities)];
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&extension_names)
            .push_next(&mut features);
        let device = creation(
            unsafe { raw.create_device(physical, &info, None) },
            "vkCreateDevice",
        )?;
        // From here every created object is destroyed by `Drop` (null
        // handles are ignored), so partial construction cleans up.
        let queue = unsafe { device.get_device_queue(facts.queue_family, 0) };
        let memory = match Memory::new(&facts, instance, physical) {
            Ok(memory) => memory,
            Err(reason) => {
                unsafe { device.destroy_device(None) };
                return Err(OpenError::Creation(reason.into()));
            }
        };
        let tables =
            match memory.resident(&device, seismic_lang::source::VULKAN_TABLES.len() as u64) {
                Ok(tables) => tables,
                Err(error) => {
                    unsafe { device.destroy_device(None) };
                    return Err(OpenError::Creation(error.to_string()));
                }
            };
        let mut inner = Self {
            device,
            facts,
            memory,
            tables,
            layout: vk::PipelineLayout::null(),
            cache: vk::PipelineCache::null(),
            timeline: vk::Semaphore::null(),
            queue: Mutex::new(Queue {
                raw: queue,
                next: 1,
            }),
            recording: Mutex::new(Recording {
                pool: vk::CommandPool::null(),
                free_commands: Vec::new(),
                free_queries: Vec::new(),
                free_chunks: Vec::new(),
                retired: Vec::new(),
            }),
            staging: Mutex::new(None),
            driver: std::sync::atomic::AtomicU64::new(0),
            driver_account: Mutex::new(None),
            driver_holds: Mutex::new(Vec::new()),
            graphs: std::sync::atomic::AtomicU64::new(0),
            physical,
            instance,
        };
        let device = &inner.device;
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(16)];
        inner.layout = creation(
            unsafe {
                device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges),
                    None,
                )
            },
            "vkCreatePipelineLayout",
        )?;
        inner.cache = creation(
            unsafe { device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None) },
            "vkCreatePipelineCache",
        )?;
        let mut timeline = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        inner.timeline = creation(
            unsafe {
                device.create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut timeline),
                    None,
                )
            },
            "vkCreateSemaphore",
        )?;
        let pool = creation(
            unsafe {
                device.create_command_pool(&command_pool_info(inner.facts.queue_family), None)
            },
            "vkCreateCommandPool",
        )?;
        inner
            .recording
            .get_mut()
            .expect("recording lock is never poisoned")
            .pool = pool;
        // What creating the device took of the heap, before any block.
        inner.driver.store(
            inner.heap_usage().saturating_sub(usage_before),
            std::sync::atomic::Ordering::Relaxed,
        );
        Ok(inner)
    }

    fn create_staging(&self) -> Result<Staging, ExecutionError> {
        let device = &self.device;
        let pool = call(
            unsafe {
                device.create_command_pool(&command_pool_info(self.facts.queue_family), None)
            },
            "vkCreateCommandPool",
        )?;
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command = match call(
            unsafe { device.allocate_command_buffers(&info) },
            "vkAllocateCommandBuffers",
        ) {
            Ok(commands) => commands[0],
            Err(error) => {
                unsafe { device.destroy_command_pool(pool, None) };
                return Err(error);
            }
        };
        let range = match self
            .memory
            .allocate_internal(device, StorageKind::Staging, STAGING_BYTES, 256)
        {
            Ok(range) => range,
            Err(error) => {
                unsafe { device.destroy_command_pool(pool, None) };
                return Err(error);
            }
        };
        Ok(Staging {
            pool,
            command,
            range,
        })
    }
}

fn command_pool_info<'a>(family: u32) -> vk::CommandPoolCreateInfo<'a> {
    vk::CommandPoolCreateInfo::default()
        .queue_family_index(family)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
}

impl Drop for Inner {
    fn drop(&mut self) {
        let device = &self.device;
        unsafe {
            let _ = device.device_wait_idle();
            let recording = self
                .recording
                .get_mut()
                .expect("recording lock is never poisoned");
            for retired in recording.retired.drain(..) {
                recording.free_queries.extend(retired.queries);
            }
            for (pool, _) in recording.free_queries.drain(..) {
                device.destroy_query_pool(pool, None);
            }
            device.destroy_command_pool(recording.pool, None);
            if let Some(staging) = self
                .staging
                .get_mut()
                .expect("staging lock is never poisoned")
                .take()
            {
                device.destroy_command_pool(staging.pool, None);
            }
            self.memory.destroy_resident(device, &self.tables);
            self.memory.destroy(device);
            device.destroy_semaphore(self.timeline, None);
            device.destroy_pipeline_cache(self.cache, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_device(None);
        }
    }
}
