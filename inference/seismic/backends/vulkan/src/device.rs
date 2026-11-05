//! One opened Vulkan device (§8.1–§8.5): the logical device with exactly
//! the floor's features plus the gated ones it has, one queue (compute-only
//! when the device has such a family) and, where the family has one, a
//! second that only zeroes reservations' chunks, a timeline semaphore that
//! numbers every submission, the block allocator, and host access to
//! storage.

use crate::facts::{self, Description, Facts};
use crate::instance::{Instance, LoaderError};
use crate::memory::{Buffer, Memory, Pool, Range};
use ash::vk;
use seismic_compiler::errors::ExecutionError;
use seismic_native_target::GateOpening;
use std::ffi::CStr;
use std::fmt;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError, RwLock, RwLockReadGuard, Weak};
use std::time::Instant;

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
    /// The device timeline value the next submission takes.
    next: u64,
    /// Submissions taken but not yet given to the queue, in order, behind
    /// the one [`Self::waiting`] (`Device::submit_with`).
    held: std::collections::VecDeque<Held>,
    /// The gates of the last submission handed to the queue, while the host
    /// has not raised them all, and its stamps: the one submission waiting
    /// on the host in the driver, with nothing behind it
    /// ([`Inner::release`]).
    waiting: Option<(Vec<(vk::Semaphore, u64)>, Handed)>,
    /// The queue refused the submission of this value: it and every later
    /// one fail with its refusal's error.
    refused: Option<(u64, Refusal)>,
}

impl Queue {
    /// The queue's refusal of `value`, where it refused it or one before
    /// it held ([`Inner::refuse`]).
    fn refused(&self, value: u64) -> Result<(), ExecutionError> {
        match &self.refused {
            Some((from, refusal)) if *from <= value => Err(refusal.error().clone()),
            _ => Ok(()),
        }
    }

    /// The device's loss, where the queue refused `value` or one before it
    /// held and could not raise the timeline past them: nothing will ever
    /// reach `value`.
    fn lost(&self, value: u64) -> Result<(), ExecutionError> {
        match &self.refused {
            Some((from, Refusal::Lost(lost))) if *from <= value => Err(lost.clone()),
            _ => Ok(()),
        }
    }
}

/// How the queue refused submissions ([`Inner::refuse`]).
enum Refusal {
    /// The device timeline was raised past them (or none took a value),
    /// so their waits end, with this error.
    Raised(ExecutionError),
    /// The queue refused raising the timeline too: the device is lost, and
    /// waits on them end at once with its loss.
    Lost(ExecutionError),
}

impl Refusal {
    fn error(&self) -> &ExecutionError {
        match self {
            Self::Raised(error) | Self::Lost(error) => error,
        }
    }
}

/// A submission's way to the device ([`Stamps`]), shared with its
/// `DirectSubmission`.
pub(crate) type Handed = Arc<Stamps>;

/// When a submission reached the device's queue, and, for one handed over
/// while a gate it waits on was shut, when the host raised its last gate.
#[derive(Default)]
pub(crate) struct Stamps {
    /// The instants before and after the `vkQueueSubmit2` that took it,
    /// whether at its commit or at the release of the gate it was held
    /// behind, set once that call succeeded; never for one the queue
    /// refused.
    pub(crate) handed: OnceLock<(Instant, Instant)>,
    /// When the host raised the last of its gates it was handed over
    /// waiting on, and why; never for one handed over with
    /// its gates raised.
    pub(crate) opened: OnceLock<(Instant, GateOpening)>,
}

/// A submission held on the host until its gates are raised.
struct Held {
    command: vk::CommandBuffer,
    value: u64,
    waits: Vec<(vk::Semaphore, u64)>,
    signals: Vec<(vk::Semaphore, u64)>,
    handed: Handed,
}

impl Inner {
    /// Give the queue every held submission, in order, up to and including
    /// the first whose gates are not all raised, unless one handed to it
    /// still waits on the host. That one waits on the host inside the
    /// driver, so the device starts it the moment the host raises its gate,
    /// whether before or after the device got there, rather than once the
    /// host has handed it over; every later one stays held until that gate
    /// is raised. A driver blocks the queue's submissions on a "wait before
    /// signal" only once enough work is queued behind one (NVIDIA:
    /// `direct::park_tests`); with nothing behind the one waiting, no driver
    /// call waits on a gate. Its gates stay in its wait: the host's writes
    /// before raising them are ordered before its work as before.
    /// `opening` is why the host raised the gate this release follows.
    /// A gate query the device fails refuses the first submission held and
    /// every one behind it (or, with none held, the next taken): the device
    /// cannot make progress, and no gate is read as raised.
    fn release(&self, queue: &mut Queue, opening: GateOpening) {
        let raised = |gates: &[(vk::Semaphore, u64)]| -> Result<bool, ExecutionError> {
            for &(semaphore, value) in gates {
                let reached = call(
                    unsafe { self.device.get_semaphore_counter_value(semaphore) },
                    "vkGetSemaphoreCounterValue",
                )?;
                if reached < value {
                    return Ok(false);
                }
            }
            Ok(true)
        };
        if let Some((gates, _)) = &queue.waiting {
            match raised(gates) {
                Ok(true) => {
                    let (_, stamps) = queue.waiting.take().expect("a waiting submission");
                    stamps.opened.set((Instant::now(), opening)).expect("a waiting submission opens once");
                }
                Ok(false) => {}
                Err(failed) => {
                    let value = queue.held.front().map_or(queue.next, |held| held.value);
                    return self.refuse(queue, value, failed);
                }
            }
        }
        while let Some(held) = queue.held.front() {
            if queue.waiting.is_some() {
                return;
            }
            match raised(&held.waits) {
                Ok(true) => {}
                Ok(false) => queue.waiting = Some((held.waits.clone(), held.handed.clone())),
                Err(failed) => {
                    let value = held.value;
                    return self.refuse(queue, value, failed);
                }
            }
            let held = queue.held.pop_front().expect("a held submission is first");
            self.handover.notify_all();
            if let Err(refused) = self.submit_now(queue.raw, &held) {
                self.refuse(queue, held.value, refused);
                return;
            }
        }
    }

    fn submit_now(&self, raw: vk::Queue, held: &Held) -> Result<(), ExecutionError> {
        let semaphore = |stages: vk::PipelineStageFlags2| {
            move |&(semaphore, value): &(vk::Semaphore, u64)| {
                vk::SemaphoreSubmitInfo::default().semaphore(semaphore).value(value).stage_mask(stages)
            }
        };
        let signal = std::iter::once(&(self.timeline, held.value))
            .chain(&held.signals)
            .map(semaphore(vk::PipelineStageFlags2::ALL_COMMANDS))
            .collect::<Vec<_>>();
        // A host gate's wait: no stage of the batch, the compute shaders
        // that read what the host wrote before signaling it included,
        // starts before it (Vulkan spec, "Semaphore Waiting": the wait's second
        // synchronization scope is limited to the stages of its mask).
        let wait = held
            .waits
            .iter()
            .map(semaphore(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::ALL_COMMANDS))
            .collect::<Vec<_>>();
        let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(held.command)];
        let submit = vk::SubmitInfo2::default()
            .command_buffer_infos(&commands)
            .wait_semaphore_infos(&wait)
            .signal_semaphore_infos(&signal);
        let handing = Instant::now();
        call(
            unsafe { self.device.queue_submit2(raw, &[submit], vk::Fence::null()) },
            "vkQueueSubmit2",
        )?;
        held.handed
            .handed
            .set((handing, Instant::now()))
            .expect("a submission is handed to the queue once");
        Ok(())
    }

    /// The queue refused the submission of `value`: it and every one held
    /// behind it never run. Where any of them took a value (`value` below
    /// `queue.next`), one empty submission raises the device timeline past
    /// them (after the work submitted before, which the queue still
    /// completes, a submission waiting on the host once the host raises
    /// its gate), so every wait on them ends and reports `refused`
    /// (`Device::wait`); if the queue refuses that too, the device is lost,
    /// and every wait on them ends at once with its loss. Refusing the next
    /// value, with nothing held, records the refusal alone: every value
    /// taken belongs to a submission handed over, which signals it itself,
    /// and a timeline signal must exceed the value it raises from.
    fn refuse(&self, queue: &mut Queue, value: u64, refused: ExecutionError) {
        queue.held.clear();
        if value == queue.next {
            queue.refused = Some((value, Refusal::Raised(refused)));
            self.handover.notify_all();
            return;
        }
        let last = queue.next - 1;
        let signal = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.timeline)
            .value(last)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let submit = vk::SubmitInfo2::default().signal_semaphore_infos(&signal);
        let refusal = match call(
            unsafe { self.device.queue_submit2(queue.raw, &[submit], vk::Fence::null()) },
            "vkQueueSubmit2",
        ) {
            Ok(()) => Refusal::Raised(refused),
            Err(lost) => Refusal::Lost(ExecutionError::DeviceLost(format!(
                "{refused}, and raising the timeline past the refused submissions then failed: {lost}"
            ))),
        };
        queue.refused = Some((value, refusal));
        self.handover.notify_all();
    }
}

/// The second queue and what its zero fills record into, formed at its
/// first fill.
struct Side {
    raw: vk::Queue,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
}

pub(crate) struct Inner {
    pub(crate) device: ash::Device,
    pub(crate) facts: Facts,
    pub(crate) memory: Memory,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) timeline: vk::Semaphore,
    queue: Mutex<Queue>,
    /// Notified whenever held submissions are handed to the queue or
    /// refused: a wait on one held waits on it before the driver does
    /// (`Device::ended`).
    handover: Condvar,
    /// The family's second queue, where it has one: reservations' chunks
    /// are zeroed on it (`Inner::zero`) beside work the first runs.
    side: Option<Mutex<Side>>,
    /// `VK_EXT_external_memory_host`'s functions, where the device imports
    /// host memory (`Facts::host_import_alignment`).
    host_import: Option<ash::ext::external_memory_host::Device>,
    /// `VK_KHR_map_memory2`'s functions, where the device maps memory at a
    /// placed host address (`Facts::placed_map_alignment`).
    placed_maps: Option<ash::khr::map_memory2::Device>,
    recording: Mutex<Recording>,
    staging: Mutex<Option<Staging>>,
    /// Device memory recorded graphs took (their command memory), measured
    /// while each is recorded: what each recording's own measure leaves to
    /// the others recorded beside it (`DirectGraphBuilder`). The driver's
    /// memory, which `Memory`'s reconciliation charges, holds it.
    pub(crate) graphs: std::sync::atomic::AtomicU64,
    /// Dropped after the logical device is destroyed (`Inner::drop`).
    instance: Arc<Instance>,
    /// Ends the destruction once every other field has dropped, last as it
    /// is declared last, on a panic too.
    destroyed: Ends,
}

/// An opened Vulkan device.
#[derive(Clone)]
pub struct Device {
    pub(crate) inner: Arc<Inner>,
    /// What the memory allocated through this open is charged to.
    pub(crate) accounts: Arc<crate::memory::MemoryAccounts>,
}

/// Memory-budget sample of the device-local heap (`VK_EXT_memory_budget`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBudget {
    pub heap_budget_bytes: u64,
    pub heap_usage_bytes: u64,
}

/// The process's Vulkan devices: the one logical device of each physical
/// device that something holds, by its UUID, and when it is destroyed.
#[derive(Default)]
struct Registry {
    opened: Vec<([u8; 16], Weak<Inner>, Arc<Destroyed>)>,
}

/// Whether a logical device's destruction has ended (`Inner::drop`): an
/// open of its physical device that finds it no longer held waits for that
/// before creating another. A flag, so a panic that poisoned its lock left
/// nothing inconsistent.
#[derive(Default)]
struct Destroyed {
    ended: Mutex<bool>,
    changed: Condvar,
}

impl Destroyed {
    fn ended(&self) -> bool {
        *self.ended.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn end(&self) {
        *self.ended.lock().unwrap_or_else(PoisonError::into_inner) = true;
        self.changed.notify_all();
    }

    // A raw condition-variable wait in the load path (a device open);
    // adopt the bottom-layer wait primitive here once seismic can use it.
    fn wait(&self) {
        let mut ended = self.ended.lock().unwrap_or_else(PoisonError::into_inner);
        while !*ended {
            ended = self.changed.wait(ended).unwrap_or_else(PoisonError::into_inner);
        }
    }
}

/// A logical device's last field: its drop ends the destruction.
struct Ends(Arc<Destroyed>);

impl Drop for Ends {
    fn drop(&mut self) {
        self.0.end();
    }
}

/// Held shared while the process creates pipelines on any of its logical
/// devices, and exclusively while it destroys one: NVIDIA 580 deadlocks when
/// one logical device is destroyed while another compiles pipelines, which
/// two of a process's devices would otherwise do, whatever their physical
/// devices.
static PIPELINES: RwLock<()> = RwLock::new(());

/// Hold off every device's destruction while pipelines are created.
pub(crate) fn creating_pipelines() -> RwLockReadGuard<'static, ()> {
    PIPELINES.read().unwrap_or_else(PoisonError::into_inner)
}

fn registry() -> MutexGuard<'static, Registry> {
    static REGISTRY: std::sync::OnceLock<Mutex<Registry>> = std::sync::OnceLock::new();
    REGISTRY
        .get_or_init(Mutex::default)
        .lock()
        .expect("the device registry lock is never poisoned")
}

impl Inner {
    /// What the driver holds beyond every block, and what recorded graphs
    /// took, now: the mark a graph's recording measures its command memory
    /// from (`DirectGraphBuilder`), read from `Memory`'s exact snapshot so
    /// no block in transition is ever counted as a graph's.
    pub(crate) fn unheld_mark(&self) -> (u64, u64) {
        (self.memory.unheld().1, self.graphs.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// What the driver took since `mark` that no graph recorded meanwhile
    /// accounts for.
    pub(crate) fn unheld_since(&self, (unheld, graphs): (u64, u64)) -> u64 {
        let grown = self.memory.unheld().1.saturating_sub(unheld);
        let recorded = self.graphs.load(std::sync::atomic::Ordering::Relaxed).saturating_sub(graphs);
        grown.saturating_sub(recorded)
    }
}

impl Device {
    /// Open the physical device with this UUID, revalidating it against
    /// the floor.
    ///
    /// A physical device has one logical device at a time, created by an
    /// open when none is held and shared by every open while one is, like
    /// CUDA's primary context. The last holder of a device (its `Device`s,
    /// buffers, chunks and reservations) destroys it, never while any of
    /// the process's devices creates pipelines ([`creating_pipelines`]), and
    /// an open meanwhile waits for that destruction to end, without the
    /// registry, before creating another.
    ///
    /// What is allocated through the returned device (buffers, reservations
    /// and their growth, its staging and argument chunks) is charged to
    /// `accounts`, and the driver's memory its creation and admissions find
    /// to their driver account: an open sharing a held logical device
    /// charges its own to its own, and what others allocated keeps theirs.
    pub fn open(uuid: [u8; 16], accounts: crate::memory::MemoryAccounts) -> Result<Self, OpenError> {
        let accounts = Arc::new(accounts);
        let mut registry = loop {
            let mut registry = registry();
            let Some(index) = registry.opened.iter().position(|(opened, _, _)| *opened == uuid) else {
                break registry;
            };
            if let Some(inner) = registry.opened[index].1.upgrade() {
                return Ok(Self { inner, accounts });
            }
            let destroyed = registry.opened[index].2.clone();
            if destroyed.ended() {
                registry.opened.swap_remove(index);
                break registry;
            }
            drop(registry);
            destroyed.wait();
        };
        let instance = crate::instance::instance().map_err(OpenError::Loader)?;
        let Description {
            handle,
            facts,
            floor,
        } = instance
            .physical_devices()
            .map_err(OpenError::Loader)?
            .into_iter()
            .map(|physical| facts::describe(&instance, physical))
            .find(|description| description.facts.uuid == uuid)
            .ok_or(OpenError::Missing)?;
        floor.map_err(OpenError::Floor)?;
        let inner = Arc::new(Inner::create(instance, handle.physical, facts, &*accounts.driver)?);
        registry.opened.push((uuid, Arc::downgrade(&inner), inner.destroyed.0.clone()));
        Ok(Self { inner, accounts })
    }

    pub fn facts(&self) -> &Facts {
        &self.inner.facts
    }

    /// Storage in the backend's own device placement, for what it allocates
    /// beside the topology's. `room`, when given, bounds the device usage
    /// the allocation may add, block slack included.
    pub fn allocate(
        &self,
        bytes: u64,
        alignment: u64,
        room: Option<u64>,
    ) -> Result<Buffer, ExecutionError> {
        self.allocate_in(self.inner.memory.device().clone(), bytes, alignment, room)
    }

    /// Storage in the backend's own upload placement (record tables), which
    /// the host writes with plain memory writes that never wait for device
    /// work; `room` as for [`Device::allocate`].
    pub fn allocate_upload(
        &self,
        bytes: u64,
        alignment: u64,
        room: Option<u64>,
    ) -> Result<Buffer, ExecutionError> {
        self.allocate_in(self.inner.memory.upload().clone(), bytes, alignment, room)
    }

    /// Storage in `kinds`, in order: an allocation lands on the first with
    /// room. `room` as for [`Device::allocate`].
    pub fn allocate_kinds(
        &self,
        kinds: &[u32],
        bytes: u64,
        alignment: u64,
        room: Option<u64>,
    ) -> Result<Buffer, ExecutionError> {
        self.allocate_in(self.inner.memory.pool(kinds)?, bytes, alignment, room)
    }

    fn allocate_in(
        &self,
        pool: Arc<Pool>,
        bytes: u64,
        alignment: u64,
        room: Option<u64>,
    ) -> Result<Buffer, ExecutionError> {
        let range = self
            .inner
            .memory
            .allocate(&self.inner.device, &pool, bytes, alignment, room, &self.accounts)?;
        Ok(Buffer::new(self.inner.clone(), range))
    }

    /// Host write: through the mapping when the storage is host-visible,
    /// else through staging and a transfer on the queue, waited on. The
    /// caller orders it after device work using the storage.
    pub fn write(&self, buffer: &Buffer, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError> {
        self.write_range(buffer.range(), offset, bytes)
    }

    /// [`Device::write`] of a range of a block.
    pub(crate) fn write_range(&self, range: &Range, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError> {
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

    /// Host read; see [`Device::write`].
    pub fn read(
        &self,
        buffer: &Buffer,
        offset: u64,
        into: &mut [u8],
    ) -> Result<(), ExecutionError> {
        self.read_range(buffer.range(), offset, into)
    }

    /// [`Device::read`] of a range of a block.
    pub(crate) fn read_range(&self, range: &Range, offset: u64, into: &mut [u8]) -> Result<(), ExecutionError> {
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
            *guard = Some(inner.create_staging(&self.accounts)?);
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
        self.submit_with(command, &[], &[], Handed::default())
    }

    /// Submit `command` once every host gate of `waits` (a timeline the
    /// host raises, `HostTimeline::signal`) reaches its value, after every
    /// earlier submission, signalling the device timeline with the next
    /// value and each of `signals` with its own once the command completes;
    /// `handed` takes the instant the queue takes it.
    ///
    /// The queue holds at most one submission whose gate is shut, with
    /// nothing behind it: the first such one goes to the queue, so the
    /// device starts it the moment the host raises its gate; every later
    /// one is held on the host, with every one behind it, until that gate
    /// is raised ([`Inner::release`]). A driver that takes a submission
    /// waiting on a value no signal operation is pending for (a "wait
    /// before signal") may block every queue's `vkQueueSubmit2` once enough
    /// work is queued behind it, and with it whichever host thread would
    /// raise the gate (NVIDIA: `direct::park_tests`); with none queued
    /// behind the one waiting, no driver call waits on a gate.
    ///
    /// The hold is order, not a means to answer: whatever the host needs to
    /// raise a gate must never be submitted to this queue, where it would be
    /// held behind that gate. An answer's work runs on the host, on the
    /// second queue (`TransferQueue`), which no hold delays, or behind the
    /// gate it opens; never as a staged copy of this queue
    /// ([`Self::transfer`]: a device tensor's host read or write).
    pub(crate) fn submit_with(
        &self,
        command: vk::CommandBuffer,
        waits: &[(vk::Semaphore, u64)],
        signals: &[(vk::Semaphore, u64)],
        handed: Handed,
    ) -> Result<u64, ExecutionError> {
        let mut queue = self.queue();
        if let Some((_, refusal)) = &queue.refused {
            return Err(refusal.error().clone());
        }
        let value = queue.next;
        queue.next += 1;
        queue.held.push_back(Held {
            command,
            value,
            waits: waits.to_vec(),
            signals: signals.to_vec(),
            handed,
        });
        // Taking a submission raises no gate: one found raised here was
        // raised by a signal whose own release saw it first.
        self.inner.release(&mut queue, GateOpening::Signalled);
        queue.refused(value).map(|()| value)
    }

    /// Submit the held submissions whose gates the host has raised, in
    /// order: after a host signal raised one, for `opening`.
    pub(crate) fn release(&self, opening: GateOpening) {
        let mut queue = self.queue();
        self.inner.release(&mut queue, opening);
    }

    /// The device timeline value of the latest submission.
    pub(crate) fn submitted(&self) -> u64 {
        self.queue().next - 1
    }

    /// The timeline value every completed submission has reached; the
    /// device's loss where it cannot say.
    fn completed(&self) -> Result<u64, ExecutionError> {
        call(
            unsafe {
                self.inner
                    .device
                    .get_semaphore_counter_value(self.inner.timeline)
            },
            "vkGetSemaphoreCounterValue",
        )
    }

    /// Whether the submission that took `value` completed, without waiting:
    /// `Ok(false)` while it runs; its error where the queue refused it or
    /// one before it held ([`Self::submit_with`]), or the device is lost.
    pub(crate) fn reached(&self, value: u64) -> Result<bool, ExecutionError> {
        self.queue().lost(value)?;
        if self.completed()? < value {
            return Ok(false);
        }
        self.queue().refused(value).map(|()| true)
    }

    /// Wait for the submission that took `value` to end; its error where
    /// the queue refused it or one before it held ([`Self::submit_with`]),
    /// or the device is lost.
    pub(crate) fn wait(&self, value: u64) -> Result<(), ExecutionError> {
        self.ended(value)?;
        self.queue().refused(value)
    }

    /// Wait for the submission that took `value` to end, run or refused,
    /// so the device uses nothing it used any more; the device's loss
    /// where it cannot say. The driver waits only on a submission handed
    /// to the queue, or refused with the timeline raised past it: one held
    /// is waited on here until it is either, so a refusal that loses the
    /// device ends the wait with its loss ([`Inner::refuse`]).
    pub(crate) fn ended(&self, value: u64) -> Result<(), ExecutionError> {
        self.inner
            .handover
            .wait_while(self.queue(), |queue| {
                queue.held.front().is_some_and(|held| held.value <= value)
            })
            .expect("queue lock is never poisoned")
            .lost(value)?;
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

    fn queue(&self) -> MutexGuard<'_, Queue> {
        self.inner.queue.lock().expect("queue lock is never poisoned")
    }

    /// The recording lock, without retiring what completed submissions
    /// used: for recording that takes nothing they free.
    pub(crate) fn locked(&self) -> MutexGuard<'_, Recording> {
        self.inner
            .recording
            .lock()
            .expect("recording lock is never poisoned")
    }

    /// The recording lock, with what completed submissions used retired
    /// for reuse; the device's loss where it cannot say which completed.
    pub(crate) fn recording(&self) -> Result<MutexGuard<'_, Recording>, ExecutionError> {
        let mut recording = self.locked();
        let completed = self.completed()?;
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
        Ok(recording)
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
                self.inner.memory.upload(),
                ARGUMENT_CHUNK_BYTES,
                256,
                &self.accounts,
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

    /// The driver memory charged, as last reconciled (I9, `Memory`).
    pub fn driver_charged(&self) -> u64 {
        self.inner.memory.driver_bytes()
    }

    /// The driver memory held now, read exactly (no block in transition),
    /// without charging it: what the next admission's reconciliation will
    /// find at least.
    pub fn driver_held(&self) -> u64 {
        self.inner.memory.unheld().1
    }

    /// What the runtime's ledger of the device's store runs under its
    /// decision lock before each admission against the store: the driver's
    /// memory reconciled (`Memory::reconcile_for_admission`, I9). It holds
    /// the device weakly: the store's accounts are held by the device.
    pub fn admission(&self) -> Arc<dyn Fn() + Send + Sync> {
        let (device, account) = (Arc::downgrade(&self.inner), self.accounts.driver.clone());
        Arc::new(move || {
            if let Some(inner) = device.upgrade() {
                inner.memory.reconcile_for_admission(&*account);
            }
        })
    }

    /// What the blocks of `kinds` hold beyond their ranges: held by this
    /// process, charged to no allocation, and free for later allocations.
    pub fn unallocated_bytes(&self, kinds: &[u32]) -> u64 {
        let memory = &self.inner.memory;
        memory.pooled(kinds).map_or(0, |pool| memory.free_bytes(&pool))
    }

    /// What the device holds for `kinds` beyond their allocations: their
    /// blocks' free space and the device's own ranges in them (argument
    /// chunks, in upload storage).
    pub fn held_beyond_allocations(&self, kinds: &[u32]) -> u64 {
        let memory = &self.inner.memory;
        memory.pooled(kinds).map_or(0, |pool| memory.free_bytes(&pool) + memory.internal_bytes(&pool))
    }

    /// The driver's memory for the device and its recorded graphs,
    /// reconciled first (`Memory::reconcile`), which the driver memory
    /// account charges to device storage.
    pub fn driver_bytes_reconciled(&self) -> u64 {
        let memory = &self.inner.memory;
        memory.reconcile(&*self.accounts.driver);
        memory.driver_bytes()
    }

    /// `VK_EXT_memory_budget` of the heaps device storage is allocated
    /// from.
    pub fn memory_budget(&self) -> MemoryBudget {
        self.inner.memory.memory_budget()
    }
}

/// `VK_EXT_memory_budget` of the heaps `facts`' device storage is allocated
/// from, together. The budget is what the driver grants this process of the
/// heaps, its own usage included.
pub(crate) fn memory_budget(instance: &Instance, physical: vk::PhysicalDevice, heaps: &[u32]) -> MemoryBudget {
    let budgets = heap_budgets(instance, physical);
    heaps
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

/// An unopened physical device. It holds its instance, so the handle is
/// valid for as long as it is held.
#[derive(Clone)]
pub struct DeviceHandle {
    pub(crate) instance: Arc<Instance>,
    pub(crate) physical: vk::PhysicalDevice,
}

impl DeviceHandle {
    /// Every heap's `VK_EXT_memory_budget` budget and usage, in heap order,
    /// read without a logical device. The extension is in the floor, so
    /// every device meeting it reports budgets.
    pub fn heap_budgets(&self) -> Vec<HeapBudget> {
        heap_budgets(&self.instance, self.physical)
    }
}

impl std::fmt::Debug for DeviceHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("DeviceHandle").field(&self.physical).finish()
    }
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
        instance: Arc<Instance>,
        physical: vk::PhysicalDevice,
        facts: Facts,
        driver_account: &dyn seismic_compiler::executable::MemoryAccount,
    ) -> Result<Self, OpenError> {
        let usage_before = memory_budget(
            &instance,
            physical,
            &facts.allocation.device_heaps,
        )
        .heap_usage_bytes;
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
        if facts.host_import_alignment.is_some() {
            extensions.push(ash::ext::external_memory_host::NAME);
        }
        if facts.placed_map_alignment.is_some() {
            extensions.push(ash::khr::map_memory2::NAME);
            extensions.push(ash::ext::map_memory_placed::NAME);
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
            .shader_buffer_int64_atomics(true)
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
        let mut placed =
            vk::PhysicalDeviceMapMemoryPlacedFeaturesEXT::default().memory_map_placed(true).memory_unmap_reserve(true);
        if facts.placed_map_alignment.is_some() {
            features = features.push_next(&mut placed);
        }
        let priorities = [1.0f32; 2];
        // The second queue, of the first's family or of a family of its own.
        let side_family = facts.side_family;
        let shares_family = side_family == Some(facts.queue_family);
        let mut queues = vec![vk::DeviceQueueCreateInfo::default()
            .queue_family_index(facts.queue_family)
            .queue_priorities(&priorities[..if shares_family { 2 } else { 1 }])];
        if let Some(family) = side_family.filter(|_| !shares_family) {
            queues.push(vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities[..1]));
        }
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
        let side = side_family.map(|family| {
            Mutex::new(Side {
                raw: unsafe { device.get_device_queue(family, u32::from(shares_family)) },
                pool: vk::CommandPool::null(),
                command: vk::CommandBuffer::null(),
                fence: vk::Fence::null(),
            })
        });
        let host_import = facts
            .host_import_alignment
            .map(|_| ash::ext::external_memory_host::Device::new(raw, &device));
        let placed_maps = facts.placed_map_alignment.map(|_| ash::khr::map_memory2::Device::new(raw, &device));
        let memory = match Memory::new(&facts, instance.clone(), physical, usage_before) {
            Ok(memory) => memory,
            Err(reason) => {
                let _destroying = PIPELINES.write().unwrap_or_else(PoisonError::into_inner);
                unsafe { device.destroy_device(None) };
                return Err(OpenError::Creation(reason.into()));
            }
        };
        let mut inner = Self {
            device,
            facts,
            memory,
            layout: vk::PipelineLayout::null(),
            timeline: vk::Semaphore::null(),
            queue: Mutex::new(Queue {
                raw: queue,
                next: 1,
                held: std::collections::VecDeque::new(),
                waiting: None,
                refused: None,
            }),
            handover: Condvar::new(),
            side,
            host_import,
            placed_maps,
            recording: Mutex::new(Recording {
                pool: vk::CommandPool::null(),
                free_commands: Vec::new(),
                free_queries: Vec::new(),
                free_chunks: Vec::new(),
                retired: Vec::new(),
            }),
            staging: Mutex::new(None),
            graphs: std::sync::atomic::AtomicU64::new(0),
            instance,
            destroyed: Ends(Arc::default()),
        };
        let device = &inner.device;
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(8)];
        inner.layout = creation(
            unsafe {
                device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges),
                    None,
                )
            },
            "vkCreatePipelineLayout",
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
        // What creating the device took of the heap.
        inner.memory.reconcile(driver_account);
        Ok(inner)
    }

    fn create_staging(&self, accounts: &crate::memory::MemoryAccounts) -> Result<Staging, ExecutionError> {
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
            .allocate_internal(device, self.memory.staging(), STAGING_BYTES, 256, accounts)
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

impl Device {
    /// Whether the second queue is open (`Facts::side_family`): what zeroes
    /// reservations' chunks and runs transfers beside the first's work.
    pub(crate) fn has_side_queue(&self) -> bool {
        self.inner.side.is_some()
    }

    /// Submit `submits` to the second queue, with no fence: their own
    /// semaphores signal their completion.
    pub(crate) fn submit_side(&self, submits: &[vk::SubmitInfo2<'_>]) -> Result<(), ExecutionError> {
        let side = self
            .inner
            .side
            .as_ref()
            .expect("a transfer queue exists only beside a second queue")
            .lock()
            .expect("side queue lock is never poisoned");
        call(unsafe { self.inner.device.queue_submit2(side.raw, submits, vk::Fence::null()) }, "vkQueueSubmit2")
    }
}

impl Inner {
    /// Write zero over each `(range, offset, bytes)` (whole words, with
    /// `offset` within the range's block) on the second queue, waited on:
    /// beside work the first queue runs or holds parked, which never waits
    /// for it. The fence's signal makes the zeros available, and the
    /// leading barrier of every later submission
    /// (`direct::compute_barrier`) makes them visible to its work.
    ///
    /// # Panics
    /// On a device without a second queue: only a device reserving device
    /// storage zeroes, which it does only with one
    /// (`Facts::reservation`).
    pub(crate) fn zero(&self, regions: &[(&Range, u64, u64)]) -> Result<(), ExecutionError> {
        let device = &self.device;
        let mut side = self
            .side
            .as_ref()
            .expect("a device reserving device storage has a second queue")
            .lock()
            .expect("side queue lock is never poisoned");
        // Formed whole or not at all: the pool, then its command buffer and
        // the fence, each kept only once all are formed (`Drop` destroys a
        // formed pool and fence).
        if side.fence == vk::Fence::null() {
            let pool = call(
                unsafe { device.create_command_pool(&command_pool_info(self.facts.side_family.expect("a second queue has a family")), None) },
                "vkCreateCommandPool",
            )?;
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let formed = call(unsafe { device.allocate_command_buffers(&info) }, "vkAllocateCommandBuffers")
                .and_then(|commands| {
                    call(unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }, "vkCreateFence")
                        .map(|fence| (commands[0], fence))
                });
            match formed {
                Ok((command, fence)) => (side.pool, side.command, side.fence) = (pool, command, fence),
                Err(error) => {
                    unsafe { device.destroy_command_pool(pool, None) };
                    return Err(error);
                }
            }
        }
        let command = side.command;
        unsafe {
            call(
                device.reset_command_buffer(command, vk::CommandBufferResetFlags::empty()),
                "vkResetCommandBuffer",
            )?;
            call(
                device.begin_command_buffer(
                    command,
                    &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                ),
                "vkBeginCommandBuffer",
            )?;
            for &(range, offset, bytes) in regions {
                device.cmd_fill_buffer(command, range.raw(), range.offset() + offset, bytes, 0);
            }
            call(device.end_command_buffer(command), "vkEndCommandBuffer")?;
            let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(command)];
            call(
                device.queue_submit2(
                    side.raw,
                    &[vk::SubmitInfo2::default().command_buffer_infos(&commands)],
                    side.fence,
                ),
                "vkQueueSubmit2",
            )?;
            let waited = call(device.wait_for_fences(&[side.fence], true, u64::MAX), "vkWaitForFences");
            call(device.reset_fences(&[side.fence]), "vkResetFences")?;
            waited
        }
    }

    /// The memory types the host memory at `pointer` may be imported as
    /// (`vkGetMemoryHostPointerPropertiesEXT`).
    ///
    /// # Panics
    /// On a device that imports no host memory: only a device reserving
    /// host storage imports, which it does only where it can
    /// (`Facts::reservation`).
    /// Map `memory` whole at `at`, a place of a reserved host range
    /// (`VK_EXT_map_memory_placed`), replacing what the range held there.
    ///
    /// # Panics
    ///
    /// On a device that maps no memory at a placed address: only a
    /// reservation the device backs as placed chunks maps one
    /// (`Facts::reservation`).
    pub(crate) fn map_placed(&self, memory: vk::DeviceMemory, at: std::ptr::NonNull<u8>) -> Result<(), ExecutionError> {
        let maps = self.placed_maps.as_ref().expect("a device backing placed chunks maps memory in place");
        let mut placed = vk::MemoryMapPlacedInfoEXT::default().placed_address(at.as_ptr().cast());
        let info = vk::MemoryMapInfoKHR::default()
            .flags(vk::MemoryMapFlags::PLACED_EXT)
            .memory(memory)
            .offset(0)
            .size(vk::WHOLE_SIZE)
            .push_next(&mut placed);
        let mut mapped = std::ptr::null_mut();
        call(
            unsafe { (maps.fp().map_memory2_khr)(self.device.handle(), &info, &mut mapped) }.result(),
            "vkMapMemory2KHR",
        )?;
        assert_eq!(mapped.cast::<u8>(), at.as_ptr(), "a placed mapping lies where it was placed");
        Ok(())
    }

    /// Unmap `memory`, mapped in place ([`Self::map_placed`]), leaving its
    /// host range reserved (`VK_MEMORY_UNMAP_RESERVE_BIT_EXT`), so nothing
    /// else maps there before the range is released.
    pub(crate) fn unmap_placed(&self, memory: vk::DeviceMemory) {
        let maps = self.placed_maps.as_ref().expect("a placed mapping's device maps memory in place");
        let info = vk::MemoryUnmapInfoKHR::default().flags(vk::MemoryUnmapFlagsKHR::RESERVE_EXT).memory(memory);
        call(unsafe { (maps.fp().unmap_memory2_khr)(self.device.handle(), &info) }.result(), "vkUnmapMemory2KHR")
            .expect("a placed mapping unmaps");
    }

    pub(crate) fn importable(&self, pointer: std::ptr::NonNull<u8>) -> Result<u32, ExecutionError> {
        let import = self
            .host_import
            .as_ref()
            .expect("a device reserving host storage imports host memory");
        let mut properties = vk::MemoryHostPointerPropertiesEXT::default();
        call(
            unsafe {
                (import.fp().get_memory_host_pointer_properties_ext)(
                    self.device.handle(),
                    vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
                    pointer.as_ptr().cast(),
                    &mut properties,
                )
            }
            .result(),
            "vkGetMemoryHostPointerPropertiesEXT",
        )?;
        Ok(properties.memory_type_bits)
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
            if let Some(side) = self.side.as_mut() {
                let side = side.get_mut().expect("side queue lock is never poisoned");
                device.destroy_fence(side.fence, None);
                device.destroy_command_pool(side.pool, None);
            }
            self.memory.destroy(device);
            device.destroy_semaphore(self.timeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            let _destroying = PIPELINES.write().unwrap_or_else(PoisonError::into_inner);
            device.destroy_device(None);
        }
    }
}

#[cfg(test)]
mod reconcile_tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    const BLOCK: u64 = 256 << 20;

    /// The opened devices meeting the floor.
    fn devices() -> Vec<Device> {
        let Ok(devices) = crate::discover() else {
            return Vec::new();
        };
        devices
            .into_iter()
            .filter(|description| description.floor.is_ok())
            .map(|description| Device::open(description.facts.uuid, crate::test_accounts(&description.facts)).expect("a device meeting the floor opens"))
            .collect()
    }

    /// A reconciliation under block churn is exact both ways: pooled
    /// blocks and chunks of 256 MiB created and freed on one thread while
    /// another reconciles in a loop leave the charge short of no driver
    /// memory (it covers what the driver holds once churn ends) and holding
    /// no block (it grew by far less than one).
    #[test]
    fn a_reconciliation_under_churn_is_exact() {
        for device in devices() {
            let memory = &device.inner.memory;
            memory.reconcile(&*device.accounts.driver);
            let before = memory.driver_bytes();
            let stop = Arc::new(AtomicBool::new(false));
            let reconciler = {
                let (device, stop) = (device.clone(), stop.clone());
                std::thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        device.inner.memory.reconcile(&*device.accounts.driver);
                    }
                })
            };
            for _ in 0..20 {
                drop(device.allocate(BLOCK, 256, None).expect("a pooled block allocates"));
                let chunk = memory.chunk(&device.inner.device, memory.device(), BLOCK, crate::memory::Mapping::Pool, &device.accounts).expect("a chunk allocates");
                memory.free(&device.inner.device, &chunk);
            }
            stop.store(true, Ordering::Relaxed);
            reconciler.join().unwrap();
            memory.reconcile(&*device.accounts.driver);
            let (_, unheld) = memory.unheld();
            let charged = memory.driver_bytes();
            assert!(charged >= unheld, "under-charged: {charged} charged, the driver holds {unheld}");
            assert!(charged - before < BLOCK / 2, "a block charged as the driver's: grew {}", charged - before);
        }
    }
}
