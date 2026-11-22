//! One opened Vulkan device (§8.1–§8.5): the logical device with exactly
//! the floor's features plus the gated ones it has, one queue (compute-only
//! when the device has such a family) and, where the family has one, a
//! second that runs transfers and zeroes pool blocks, a timeline semaphore that
//! numbers every submission, the block allocator, and host access to
//! storage.

use crate::device_info::{self, Description, DeviceInfo, FormationTarget, Probed, Properties};
use crate::instance::{Instance, LoaderError};
use crate::memory::{Buffer, DevicePools, Memory, MemoryStores, Pool, Range};
use ash::vk;
use seismic_compiler::errors::ExecutionError;
use seismic_memory::entry::{Charge, Entry};
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

/// Map a Vulkan call that allocates, maps, or backs device/host memory or
/// command objects: device loss stays `DeviceLost`, memory/object exhaustion
/// maps to `DeviceRefused`, and any other error goes through `fallback`.
pub(crate) fn memory_call<T>(
    result: Result<T, vk::Result>,
    what: &str,
    required: u64,
    memory: &Memory,
    fallback: fn(String) -> ExecutionError,
) -> Result<T, ExecutionError> {
    result.map_err(|error| {
        classify_vk_error(
            error,
            what,
            required,
            || memory.memory_budget(),
            || (memory.object_limit(), memory.objects()),
            fallback,
        )
    })
}

fn classify_vk_error(
    error: vk::Result,
    what: &str,
    required: u64,
    budget: impl FnOnce() -> crate::MemoryBudget,
    objects: impl FnOnce() -> (u64, u64),
    fallback: fn(String) -> ExecutionError,
) -> ExecutionError {
    match error {
        vk::Result::ERROR_DEVICE_LOST => ExecutionError::DeviceLost(format!("{what}: {error}")),
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
            let budget = budget();
            ExecutionError::DeviceRefused {
                required,
                reading: seismic_compiler::errors::RefusalReading::HeapBudget {
                    budget: budget.heap_budget_bytes,
                    usage: budget.heap_usage_bytes,
                },
            }
        }
        vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_MEMORY_MAP_FAILED => {
            ExecutionError::DeviceRefused {
                required,
                reading: seismic_compiler::errors::RefusalReading::Host,
            }
        }
        vk::Result::ERROR_TOO_MANY_OBJECTS => {
            let (limit, held) = objects();
            ExecutionError::DeviceRefused {
                required,
                reading: seismic_compiler::errors::RefusalReading::MemoryObjects { limit, held },
            }
        }
        other => fallback(format!("{what} failed: {other}")),
    }
}

/// Structural retention cap for submission free lists (`free_commands`,
/// `free_queries`, `free_chunks`): one ready resource for the next submission
/// plus one per active in-flight submission (`recording.retired.len()`).
/// Steady-state pipelines stay warm without allocator churn, while burst
/// overflow is released back to Vulkan as flights drain.
fn free_cap(in_flight: usize) -> usize {
    in_flight.saturating_add(1)
}

/// Drain excess entries beyond `cap` from the tail so steady-state low-index
/// entries remain pooled and overflow entries are released first.
fn trim_to_cap<T>(pool: &mut Vec<T>, cap: usize) -> Vec<T> {
    if pool.len() > cap {
        pool.drain(cap..).collect()
    } else {
        Vec::new()
    }
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
    /// Creating the device or one of its objects failed for want of memory
    /// (`VK_ERROR_OUT_OF_DEVICE_MEMORY`, `VK_ERROR_OUT_OF_HOST_MEMORY`), or
    /// as another holds what it needs (`VK_ERROR_TOO_MANY_OBJECTS`): opening
    /// it later may succeed.
    Busy(String),
}

impl OpenError {
    /// An open's own allocation or submission that failed: for want of
    /// memory, [`Self::Busy`].
    fn of(error: ExecutionError) -> Self {
        match error.capacity_refusal() {
            Some(_) => Self::Busy(error.to_string()),
            None => Self::Creation(error.to_string()),
        }
    }

    /// Why the device does not meet the floor, when that is the error: the
    /// reason discovery reports as the device's unavailability.
    pub fn floor_reason(&self) -> Option<String> {
        match self {
            Self::Floor(reason) => Some(reason.clone()),
            Self::Loader(_) | Self::Missing | Self::Creation(_) | Self::Busy(_) => None,
        }
    }
}

impl fmt::Display for OpenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Loader(error) => write!(f, "{error}"),
            Self::Missing => f.write_str("the Vulkan device is no longer present"),
            Self::Creation(reason) | Self::Busy(reason) => write!(f, "opening the Vulkan device failed: {reason}"),
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
    /// Timeline value of the last submitted transfer using this staging
    /// command buffer and range; 0 when no submission is in flight.
    last_value: u64,
}

/// One step of unmapped device storage staged on the first queue
/// (`Device::transfer`).
enum Staged {
    /// A copy from the staging range into the storage.
    Upload(vk::BufferCopy),
    /// A copy from the storage into the staging range.
    Download(vk::BufferCopy),
    /// Zero over the whole storage, whose bytes are whole words.
    Zero,
}

/// Bytes of the staging range; larger transfers go in pieces.
const STAGING_BYTES: u64 = 16 << 20;

/// Resources of one submission (or of a dropped graph), reusable once the
/// timeline reaches `value`.
pub(crate) struct Retired {
    pub(crate) value: u64,
    pub(crate) command: Option<RetiredCommand>,
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
/// pools and retired submissions awaiting completion.
pub(crate) struct Recording {
    pub(crate) pool: vk::CommandPool,
    free_commands: Vec<vk::CommandBuffer>,
    free_queries: Vec<(vk::QueryPool, u32)>,
    retired: Vec<(Retired, Weak<Opened>)>,
}

impl Recording {
    /// Return an unused primary command buffer, freeing excess beyond the
    /// active in-flight retention cap (`free_cap`).
    pub(crate) fn free_command(&mut self, device: &ash::Device, command: vk::CommandBuffer) {
        self.free_commands.push(command);
        let excess = trim_to_cap(&mut self.free_commands, free_cap(self.retired.len()));
        if !excess.is_empty() {
            unsafe { device.free_command_buffers(self.pool, &excess) };
        }
    }
}

/// A pooled primary command buffer the device's own work records, retired
/// for reuse when it drops, on every path: at once while nothing was
/// submitted, else once the timeline reaches the submission's value.
struct Primary<'a> {
    device: &'a Device,
    command: vk::CommandBuffer,
    /// The submission's timeline value; 0 until it is submitted.
    value: u64,
}

impl<'a> Primary<'a> {
    fn take(device: &'a Device) -> Result<Self, ExecutionError> {
        let command = device.command(&mut *device.recording()?, vk::CommandBufferLevel::PRIMARY)?;
        Ok(Self { device, command, value: 0 })
    }
}

impl Drop for Primary<'_> {
    fn drop(&mut self) {
        self.device.retire(Retired {
            value: self.value,
            command: Some(RetiredCommand::Primary(self.command)),
            queries: None,
            arguments: Vec::new(),
        });
    }
}

/// Bytes of one argument chunk.
pub(crate) const ARGUMENT_CHUNK_BYTES: u64 = 256 << 10;

/// Maximum submissions held on the host behind a shut gate before new batches
/// wait on `handover` for room. Large enough for an entire multi-layer model
/// step to enqueue behind its first park, small enough to keep live driver
/// command buffers (and their kernel IB BOs / mappings) bounded so flooding
/// behind a park applies backpressure instead of exhausting driver pools.
pub(crate) const MAX_HELD_SUBMISSIONS: usize = 512;

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

/// Block while `queue.held` is at or above `limit` until a release drains
/// entries, the queue refuses, or the device is lost.
fn wait_for_held_room(
    queue: &Mutex<Queue>,
    handover: &Condvar,
    lost: &OnceLock<ExecutionError>,
    limit: usize,
) -> Result<(), ExecutionError> {
    let mut queue = queue.lock().expect("queue lock is never poisoned");
    loop {
        if let Some(error) = lost.get() {
            return Err(error.clone());
        }
        if let Some((_, refusal)) = &queue.refused {
            return Err(refusal.error().clone());
        }
        if queue.held.len() < limit {
            return Ok(());
        }
        queue = handover
            .wait(queue)
            .expect("queue lock is never poisoned");
    }
}

/// Advance `queue`'s waiting and held submissions using the provided gate
/// query, submit, handover-notify, and refusal hooks.
///
/// `queue.waiting` is only recorded after `submit` succeeds, and is cleared on
/// gate-query failure so a refused submission or failed query never leaves a
/// `HostTimeline` semaphore handle in `queue.waiting` past its owner's drop.
fn release_queue(
    queue: &mut Queue,
    opening: GateOpening,
    mut raised: impl FnMut(&[(vk::Semaphore, u64)]) -> Result<bool, ExecutionError>,
    mut submit: impl FnMut(&Held) -> Result<(), ExecutionError>,
    mut notify: impl FnMut(),
    mut refuse: impl FnMut(&mut Queue, u64, ExecutionError),
) {
    if let Some((gates, _)) = &queue.waiting {
        match raised(gates) {
            Ok(true) => {
                let (_, stamps) = queue.waiting.take().expect("a waiting submission");
                stamps
                    .opened
                    .set((Instant::now(), opening))
                    .expect("a waiting submission opens once");
            }
            Ok(false) => {}
            Err(failed) => {
                queue.waiting = None;
                let value = queue.held.front().map_or(queue.next, |held| held.value);
                refuse(queue, value, failed);
                return;
            }
        }
    }
    while let Some(held) = queue.held.front() {
        if queue.waiting.is_some() {
            return;
        }
        let gates_raised = match raised(&held.waits) {
            Ok(raised) => raised,
            Err(failed) => {
                let value = held.value;
                refuse(queue, value, failed);
                return;
            }
        };
        let held = queue.held.pop_front().expect("a held submission is first");
        notify();
        if let Err(refused) = submit(&held) {
            refuse(queue, held.value, refused);
            return;
        }
        if !gates_raised {
            queue.waiting = Some((held.waits, held.handed));
            return;
        }
    }
}

/// Whether a staging command pool and buffer range are safe to destroy when an
/// [`Opened`] drops: either no transfer is in flight (`last_value == 0`), the
/// timeline wait confirmed completion, or the device is lost (Vulkan permits
/// destroying objects after `VK_ERROR_DEVICE_LOST`). On a non-`DeviceLost`
/// wait failure, the submission may still be in flight, so destruction is
/// skipped to prevent Vulkan UB/UAF.
fn staging_safe_to_destroy(
    last_value: u64,
    wait: impl FnOnce(u64) -> Result<(), ExecutionError>,
) -> bool {
    last_value == 0
        || match wait(last_value) {
            Ok(()) | Err(ExecutionError::DeviceLost(_)) => true,
            Err(_) => false,
        }
}

/// Wait for and reset a side-queue fence if dirty, clearing `fence_dirty` only
/// once both wait and reset succeed. If `wait` fails, `reset` is skipped so a
/// fence with a pending signal operation is never reset.
fn clean_dirty_fence(
    fence_dirty: &mut bool,
    wait: impl FnOnce() -> Result<(), ExecutionError>,
    reset: impl FnOnce() -> Result<(), ExecutionError>,
) -> Result<(), ExecutionError> {
    if !*fence_dirty {
        return Ok(());
    }
    wait()?;
    reset()?;
    *fence_dirty = false;
    Ok(())
}

impl Inner {
    /// Whether any Vulkan call on this logical device has observed
    /// `VK_ERROR_DEVICE_LOST`.
    pub(crate) fn is_lost(&self) -> bool {
        self.lost.get().is_some()
    }

    /// Return the recorded device loss if this logical device has been lost.
    fn check_lost(&self) -> Result<(), ExecutionError> {
        match self.lost.get() {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }

    /// Record device loss under `queue`, draining held submissions and waking
    /// every thread waiting on `handover`.
    fn lose_locked(&self, queue: &mut Queue, error: ExecutionError) -> ExecutionError {
        let first = self.lost.get_or_init(|| error).clone();
        queue.held.clear();
        queue.waiting = None;
        queue.refused = Some((0, Refusal::Lost(first.clone())));
        self.handover.notify_all();
        first
    }

    /// Record device loss on this logical device, waking any blocked waiters.
    pub(crate) fn lose(&self, error: ExecutionError) -> ExecutionError {
        let mut queue = self.queue.lock().expect("queue lock is never poisoned");
        self.lose_locked(&mut queue, error)
    }

    /// Execute one Vulkan call against this logical device, enforcing the
    /// single device-wide lost state on entry and transitioning to lost on
    /// `VK_ERROR_DEVICE_LOST`. Must not be called while holding `self.queue`.
    pub(crate) fn vk<T>(
        &self,
        what: &str,
        f: impl FnOnce(&ash::Device) -> Result<T, vk::Result>,
    ) -> Result<T, ExecutionError> {
        self.check_lost()?;
        match f(&self.device) {
            Ok(value) => Ok(value),
            Err(vk::Result::ERROR_DEVICE_LOST) => {
                let error = ExecutionError::DeviceLost(format!("{what}: {}", vk::Result::ERROR_DEVICE_LOST));
                Err(self.lose(error))
            }
            Err(other) => Err(ExecutionError::SubmissionFailed(format!("{what} failed: {other}"))),
        }
    }

    /// Wait at most `timeout` for timeline `semaphore` to reach `value`,
    /// enforcing the single device-wide lost state before, after timeout,
    /// and on error. Must not be called while holding `self.queue`.
    pub(crate) fn wait_semaphore(
        &self,
        semaphore: vk::Semaphore,
        value: u64,
        timeout: std::time::Duration,
    ) -> Result<bool, ExecutionError> {
        self.check_lost()?;
        let semaphores = [semaphore];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        let nanoseconds = u64::try_from(timeout.as_nanos()).unwrap_or(u64::MAX);
        match unsafe { self.device.wait_semaphores(&info, nanoseconds) } {
            Ok(()) => Ok(true),
            Err(vk::Result::TIMEOUT) => {
                self.check_lost()?;
                Ok(false)
            }
            Err(vk::Result::ERROR_DEVICE_LOST) => {
                let error = ExecutionError::DeviceLost(format!(
                    "vkWaitSemaphores: {}",
                    vk::Result::ERROR_DEVICE_LOST
                ));
                Err(self.lose(error))
            }
            Err(other) => Err(ExecutionError::SubmissionFailed(format!(
                "vkWaitSemaphores failed: {other}"
            ))),
        }
    }

    /// Wait for the submission that took `value` to end, run or refused,
    /// so the device uses nothing it used any more.
    pub(crate) fn ended(&self, value: u64) -> Result<(), ExecutionError> {
        let queue = self.queue.lock().expect("queue lock is never poisoned");
        #[cfg(test)]
        if queue.held.front().is_some_and(|held| held.value <= value) {
            if let Some(notify) = self
                .on_handover_wait
                .lock()
                .expect("handover hook lock is never poisoned")
                .take()
            {
                let _ = notify.send(());
            }
        }
        self.handover
            .wait_while(queue, |queue| {
                !self.is_lost() && queue.held.front().is_some_and(|held| held.value <= value)
            })
            .expect("queue lock is never poisoned")
            .lost(value)?;
        let semaphores = [self.timeline];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        self.vk("vkWaitSemaphores", |device| unsafe {
            device.wait_semaphores(&info, u64::MAX)
        })
    }

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
        if self.is_lost() {
            return;
        }
        let raw = queue.raw;
        release_queue(
            queue,
            opening,
            |gates| {
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
            },
            |held| self.submit_now(raw, held),
            || self.handover.notify_all(),
            |queue, value, error| self.refuse(queue, value, error),
        );
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
        if matches!(refused, ExecutionError::DeviceLost(_)) {
            self.lose_locked(queue, refused);
            return;
        }
        queue.held.clear();
        queue.waiting = None;
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
        match call(
            unsafe { self.device.queue_submit2(queue.raw, &[submit], vk::Fence::null()) },
            "vkQueueSubmit2",
        ) {
            Ok(()) => {
                queue.refused = Some((value, Refusal::Raised(refused)));
                self.handover.notify_all();
            }
            Err(lost) => {
                self.lose_locked(
                    queue,
                    ExecutionError::DeviceLost(format!(
                        "{refused}, and raising the timeline past the refused submissions then failed: {lost}"
                    )),
                );
            }
        }
    }
}

/// The second queue and what its zero fills record into, formed at its
/// first fill.
struct Side {
    raw: vk::Queue,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    /// Set once `queue_submit2` submits `command` with `fence`, cleared only
    /// after both `wait_for_fences` and `reset_fences` succeed. While true,
    /// `command` may be in flight or `fence` may still be signaled.
    fence_dirty: bool,
}

pub(crate) struct Inner {
    pub(crate) device: ash::Device,
    pub(crate) info: DeviceInfo,
    pub(crate) properties: Properties,
    pub(crate) memory: Memory,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) timeline: vk::Semaphore,
    queue: Mutex<Queue>,
    /// Notified whenever held submissions are handed to the queue or
    /// refused: a wait on one held waits on it before the driver does
    /// (`Device::ended`).
    handover: Condvar,
    /// The first `VK_ERROR_DEVICE_LOST` observed on this logical device, if
    /// any: device loss in Vulkan is device-wide and irreversible, so every
    /// queue, timeline and waiter observes it immediately without blocking.
    lost: OnceLock<ExecutionError>,
    /// Test hook fired under `self.queue` right before `Device::ended` blocks
    /// on `self.handover`, so device-loss unblock tests synchronize without
    /// sleeping.
    #[cfg(test)]
    pub(crate) on_handover_wait: Mutex<Option<std::sync::mpsc::Sender<()>>>,
    /// The family's second queue, where it has one: pool blocks are zeroed
    /// on it (`Inner::zero`) beside work the first runs.
    side: Option<Mutex<Side>>,
    /// `VK_KHR_pipeline_executable_properties`' functions, where formed
    /// pipelines report their statistics (`DeviceInfo::pipeline_statistics`).
    pub(crate) executables: Option<ash::khr::pipeline_executable_properties::Device>,
    recording: Mutex<Recording>,
    /// Device memory recorded graphs took (their command memory), measured
    /// while each is recorded: what each recording's own measure leaves to
    /// the others recorded beside it (`DirectGraphBuilder`). The driver's
    /// memory, which `Memory`'s reconciliation charges, holds it.
    pub(crate) graphs: std::sync::atomic::AtomicU64,
    /// The tables buffer (`seismic_lang::source::VULKAN_TABLES`) in the
    /// device placement, every launch's second push constant: the logical
    /// device's own memory, like the device object and its pipelines, a
    /// block the driver's memory holds ([`Memory::driver_block`]), which
    /// each reconciliation charges to every live open's own entry and no
    /// ledger admits. Small: the tables' bytes (about 38 KiB of ggml grids,
    /// codebooks and conversion tables), rounded up to the device's least
    /// memory object (its heaps over `maxMemoryAllocationCount`). Written
    /// before any open of the device is returned ([`Device::open`]) and
    /// released as the device is destroyed (`Inner::drop`).
    pub(crate) tables: Range,
    /// Dropped after the logical device is destroyed (`Inner::drop`).
    instance: Arc<Instance>,
    /// Ends the destruction once every other field has dropped, last as it
    /// is declared last, on a panic too.
    destroyed: Ends,
}

/// Per-open state of a [`Device`]: its stores, pools, transfer staging,
/// reusable argument chunks, and reconciled driver-memory charge. Dropping
/// the last clone of an open releases its staging, argument chunks, and
/// driver charge from its [`MemoryStores`] without disturbing other opens
/// sharing the logical device.
pub(crate) struct Opened {
    inner: Arc<Inner>,
    pub(crate) stores: MemoryStores,
    pools: DevicePools,
    staging: Mutex<Option<Staging>>,
    free_chunks: Mutex<Vec<Range>>,
    driver_charge: Mutex<Option<Charge>>,
}

impl Drop for Opened {
    fn drop(&mut self) {
        self.inner.retire_completed_best_effort();
        if let Some(staging) = self
            .staging
            .get_mut()
            .expect("staging lock is never poisoned")
            .take()
        {
            if staging_safe_to_destroy(staging.last_value, |value| self.inner.ended(value)) {
                unsafe { self.inner.device.destroy_command_pool(staging.pool, None) };
                self.inner.memory.release(&self.inner.device, &staging.range);
            }
        }
        for chunk in self
            .free_chunks
            .get_mut()
            .expect("free chunks lock is never poisoned")
            .drain(..)
        {
            self.inner.memory.release(&self.inner.device, &chunk);
        }
        if let Some(mut charge) = self
            .driver_charge
            .get_mut()
            .expect("driver charge lock poisoned")
            .take()
        {
            let bytes = charge.bytes();
            if bytes > 0 {
                let change = charge.entry().change();
                charge.shrink(&change, bytes);
            }
        }
    }
}

/// An opened Vulkan device.
#[derive(Clone)]
pub struct Device {
    pub(crate) inner: Arc<Inner>,
    opened: Arc<Opened>,
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
        (self.memory.unheld().1.driver, self.graphs.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// What the driver took since `mark` that no graph recorded meanwhile
    /// accounts for.
    pub(crate) fn unheld_since(&self, (unheld, graphs): (u64, u64)) -> u64 {
        let grown = self.memory.unheld().1.driver.saturating_sub(unheld);
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
    /// CUDA's primary context. The last holder of a device (its `Device`s
    /// and buffers) destroys it, never while any of
    /// the process's devices creates pipelines ([`creating_pipelines`]), and
    /// an open meanwhile waits for that destruction to end, without the
    /// registry, before creating another.
    ///
    /// What is allocated through the returned device lies in `stores`: each
    /// buffer charged to the entry its allocation names, its blocks' free
    /// space, its staging and argument chunks to their store's own entry,
    /// and the driver's memory, as each reconciliation finds it
    /// ([`Device::reconcile_driver_memory`]), to the own entry of the
    /// device's store ([`MemoryStores::own`]): an open
    /// sharing a held logical device charges its own to its own, and what
    /// others allocated keeps theirs.
    ///
    /// The device's properties take its probes as `probed` holds them, what
    /// the measurement job stored for the device and driver version; each
    /// probe not run takes its safe value. Opening runs no probe.
    pub fn open(uuid: [u8; 16], stores: MemoryStores, probed: &Probed) -> Result<Self, OpenError> {
        let mut registry = loop {
            let mut registry = registry();
            let Some(index) = registry.opened.iter().position(|(opened, _, _)| *opened == uuid) else {
                break registry;
            };
            if let Some(inner) = registry.opened[index].1.upgrade() {
                if !inner.is_lost() {
                    drop(registry);
                    return Self::opening(inner, stores);
                }
                registry.opened.swap_remove(index);
                break registry;
            }
            let destroyed = registry.opened[index].2.clone();
            if destroyed.ended() {
                registry.opened.swap_remove(index);
                break registry;
            }
            drop(registry);
            destroyed.wait();
        };
        let device = Self::create_device(uuid, stores, probed)?;
        registry.opened.push((uuid, Arc::downgrade(&device.inner), device.inner.destroyed.0.clone()));
        drop(registry);
        Ok(device)
    }

    /// Open a dedicated logical device not registered in the process registry,
    /// for tests that simulate device loss without affecting sibling tests.
    #[cfg(test)]
    pub(crate) fn open_isolated(uuid: [u8; 16], stores: MemoryStores, probed: &Probed) -> Result<Self, OpenError> {
        Self::create_device(uuid, stores, probed)
    }

    fn create_device(uuid: [u8; 16], stores: MemoryStores, probed: &Probed) -> Result<Self, OpenError> {
        let instance = crate::instance::instance().map_err(OpenError::Loader)?;
        let description = instance
            .physical_devices()
            .map_err(OpenError::Loader)?
            .into_iter()
            .map(|physical| device_info::describe(&instance, physical))
            .collect::<Result<Vec<_>, _>>()
            .map_err(OpenError::Loader)?
            .into_iter()
            .find(|description| description.info.uuid == uuid)
            .ok_or(OpenError::Missing)?;
        let properties = description.properties(probed).map_err(OpenError::Floor)?;
        let Description { handle, info, .. } = description;
        let inner = Arc::new(Inner::create(instance, handle.physical, info, properties)?);
        // The logical device's tables are written once, before it is
        // registered: no other open finds it until they are.
        let device = Self::opening(inner, stores)?;
        device.write_tables().map_err(OpenError::of)?;
        Ok(device)
    }

    /// A new open of `inner` charging `stores`: the one place an [`Opened`]
    /// is made.
    fn opening(inner: Arc<Inner>, stores: MemoryStores) -> Result<Self, OpenError> {
        let pools = DevicePools::new(&inner.memory).map_err(OpenError::Creation)?;
        let opened = Arc::new(Opened {
            inner: inner.clone(),
            stores,
            pools,
            staging: Mutex::new(None),
            free_chunks: Mutex::new(Vec::new()),
            driver_charge: Mutex::new(None),
        });
        Ok(Self { inner, opened })
    }

    /// The address of the logical device's tables buffer, every launch's
    /// second push constant.
    pub(crate) fn tables(&self) -> u64 {
        self.inner.tables.address()
    }

    /// Write the tables buffer's bytes, once, as the logical device is
    /// created ([`Device::open`]): through its mapping where the storage is
    /// host-visible, else by `vkCmdUpdateBuffer` on the queue, waited on,
    /// which takes no staging.
    fn write_tables(&self) -> Result<(), ExecutionError> {
        let tables = &self.inner.tables;
        let bytes = seismic_lang::source::VULKAN_TABLES;
        if tables.mapped().is_some() {
            tables.write_mapped(0, bytes);
            return Ok(());
        }
        let device = &self.inner.device;
        let mut command = Primary::take(self)?;
        let raw = command.command;
        {
            let _recording = self.locked();
            unsafe {
                call(
                    device.begin_command_buffer(
                        raw,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    ),
                    "vkBeginCommandBuffer",
                )?;
                cmd_update_buffer(device, raw, tables.raw(), tables.offset(), bytes);
                barrier(
                    device,
                    raw,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    vk::AccessFlags2::MEMORY_READ,
                );
                call(device.end_command_buffer(raw), "vkEndCommandBuffer")?;
            }
        }
        command.value = self.submit(raw)?;
        self.wait(command.value)
    }

    pub fn info(&self) -> &DeviceInfo {
        &self.inner.info
    }

    /// The device's properties (`seismic_lang::device::vulkan`).
    pub fn properties(&self) -> &Properties {
        &self.inner.properties
    }

    /// The target this device forms sources in.
    pub fn formation(&self) -> FormationTarget {
        self.inner.info.formation(&self.inner.properties)
    }

    /// Storage in the backend's own device placement, for what it allocates
    /// beside the topology's, charged to `entry`.
    pub fn allocate(
        &self,
        bytes: u64,
        alignment: u64,
        entry: &Arc<dyn Entry>,
    ) -> Result<Buffer, ExecutionError> {
        self.allocate_in(self.opened.pools.device().clone(), bytes, alignment, entry)
    }

    /// Storage in the backend's own upload placement (record tables), which
    /// the host writes with plain memory writes that never wait for device
    /// work, charged to `entry`.
    pub fn allocate_upload(
        &self,
        bytes: u64,
        alignment: u64,
        entry: &Arc<dyn Entry>,
    ) -> Result<Buffer, ExecutionError> {
        self.allocate_in(self.opened.pools.upload().clone(), bytes, alignment, entry)
    }

    /// Storage in `kinds`, in order (kinds of one store): an allocation
    /// lands on the first with room, charged to `entry`; the buffer names
    /// the kind it landed on ([`Buffer::kind`]).
    pub fn allocate_kinds(
        &self,
        kinds: &[u32],
        bytes: u64,
        alignment: u64,
        entry: &Arc<dyn Entry>,
    ) -> Result<Buffer, ExecutionError> {
        self.allocate_in(self.opened.pools.pool(&self.inner.memory, kinds)?, bytes, alignment, entry)
    }

    /// A range of `pool`, its bytes carved from its block for `entry`.
    fn allocate_in(
        &self,
        pool: Arc<Pool>,
        bytes: u64,
        alignment: u64,
        entry: &Arc<dyn Entry>,
    ) -> Result<Buffer, ExecutionError> {
        let range = self
            .inner
            .memory
            .allocate(&self.inner.device, &pool, bytes, alignment, &self.opened.stores)?;
        let carved = range.carve(entry);
        Ok(Buffer::new(self.inner.clone(), range, Some(carved)))
    }

    /// A pool block (`seismic_memory::pool`): one memory object and
    /// the buffer bound to it, of memory type `kind`, never shared, every
    /// byte zero and backed, charged exactly to `entry` (its bytes rounded
    /// to whole words and the driver's rounding included) and returned to
    /// the driver when it drops. Zeroed through its mapping where its type
    /// is mappable, else on the second queue, beside work the first runs
    /// or holds parked; on a device without one, on the first queue behind
    /// its submissions.
    pub fn allocate_block(&self, kind: u32, bytes: u64, entry: &Arc<dyn Entry>, fill: seismic_memory::SlabFill) -> Result<Buffer, ExecutionError> {
        let range = self.inner.memory.block(&self.inner.device, kind, bytes, entry)?;
        // Freed by its drop where zeroing fails.
        let buffer = Buffer::new(self.inner.clone(), range, None);
        if fill == seismic_memory::SlabFill::Zeroed {
            let range = buffer.range();
            match range.mapped() {
                // SAFETY: the block's own mapping, which nothing else reaches yet.
                Some(base) => unsafe { std::ptr::write_bytes(base, 0, range.bytes() as usize) },
                None if self.inner.side.is_some() => self.inner.zero(&[(range, 0, range.bytes())])?,
                None => drop(self.transfer(range, |_| Staged::Zero)?),
            }
        }
        Ok(buffer)
    }

    /// The own entry of the store of the device's own storage
    /// ([`MemoryStores::own`]).
    pub(crate) fn own(&self) -> Arc<dyn Entry> {
        self.opened.stores.own(&self.inner.memory)
    }


    /// Host write: through the mapping when the storage is host-visible,
    /// else through staging and a transfer on the queue, waited on. The
    /// caller orders it after device work using the storage.
    pub fn write(&self, buffer: &Buffer, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError> {
        self.write_range(buffer.range(), offset, bytes)
    }

    /// [`Device::write`] of a range of a block.
    pub(crate) fn write_range(&self, range: &Range, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError> {
        range.check_span(offset, bytes.len() as u64);
        if range.mapped().is_some() {
            range.write_mapped(offset, bytes);
            return Ok(());
        }
        let mut done = 0usize;
        while done < bytes.len() {
            let length = (bytes.len() - done).min(STAGING_BYTES as usize);
            let _staging = self.transfer(range, |staging| {
                staging.write_mapped(0, &bytes[done..done + length]);
                Staged::Upload(vk::BufferCopy {
                    src_offset: staging.offset(),
                    dst_offset: range.offset() + offset + done as u64,
                    size: length as u64,
                })
            })?;
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
        range.check_span(offset, into.len() as u64);
        if range.mapped().is_some() {
            range.read_mapped(offset, into);
            return Ok(());
        }
        let mut done = 0usize;
        while done < into.len() {
            let length = (into.len() - done).min(STAGING_BYTES as usize);
            let staging = self.transfer(range, |staging| {
                Staged::Download(vk::BufferCopy {
                    src_offset: range.offset() + offset + done as u64,
                    dst_offset: staging.offset(),
                    size: length as u64,
                })
            })?;
            staging
                .as_ref()
                .expect("a transfer leaves its staging in place")
                .range
                .read_mapped(0, &mut into[done..done + length]);
            done += length;
        }
        Ok(())
    }

    /// One staged step of `range`, as `step` gives it from the staging
    /// range ([`Staged`]), submitted and waited on. Returns the staging lock
    /// so a read can copy out.
    fn transfer(
        &self,
        range: &Range,
        step: impl FnOnce(&Range) -> Staged,
    ) -> Result<MutexGuard<'_, Option<Staging>>, ExecutionError> {
        let inner = &self.inner;
        let device = &inner.device;
        let mut guard = self
            .opened
            .staging
            .lock()
            .expect("staging lock is never poisoned");
        if guard.is_none() {
            *guard = Some(inner.create_staging(&self.opened.pools, &self.opened.stores)?);
        }
        let staging = guard.as_mut().expect("staging was created above");
        if staging.last_value != 0 {
            self.ended(staging.last_value)?;
            staging.last_value = 0;
        }
        let step = step(&staging.range);
        let command = staging.command;
        inner.vk("vkResetCommandBuffer", |device| unsafe {
            device.reset_command_buffer(command, vk::CommandBufferResetFlags::empty())
        })?;
        inner.vk("vkBeginCommandBuffer", |device| unsafe {
            device.begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        })?;
        unsafe {
            barrier(
                device,
                command,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
            );
            match step {
                Staged::Upload(copy) => device.cmd_copy_buffer(command, staging.range.raw(), range.raw(), &[copy]),
                Staged::Download(copy) => device.cmd_copy_buffer(command, range.raw(), staging.range.raw(), &[copy]),
                Staged::Zero => device.cmd_fill_buffer(command, range.raw(), range.offset(), range.bytes(), 0),
            }
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
        }
        inner.vk("vkEndCommandBuffer", |device| unsafe {
            device.end_command_buffer(command)
        })?;
        let value = self.submit(command)?;
        staging.last_value = value;
        self.ended(value)?;
        staging.last_value = 0;
        self.queue().refused(value)?;
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
        self.inner.vk("vkGetSemaphoreCounterValue", |device| unsafe {
            device.get_semaphore_counter_value(self.inner.timeline)
        })
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
    /// is waited on here until it is either, so a refusal or device loss
    /// anywhere on the logical device ends the wait with its loss
    /// ([`Inner::refuse`], [`Inner::lose`]).
    pub(crate) fn ended(&self, value: u64) -> Result<(), ExecutionError> {
        self.inner.ended(value)
    }

    fn queue(&self) -> MutexGuard<'_, Queue> {
        self.inner.queue.lock().expect("queue lock is never poisoned")
    }

    /// Wait while the host hold behind a shut gate is at capacity
    /// ([`MAX_HELD_SUBMISSIONS`]) before allocating another batch's command
    /// buffer and query pool.
    pub(crate) fn wait_held_room(&self) -> Result<(), ExecutionError> {
        wait_for_held_room(
            &self.inner.queue,
            &self.inner.handover,
            &self.inner.lost,
            MAX_HELD_SUBMISSIONS,
        )
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
        self.inner.retire_up_to(&mut recording, completed);
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
                memory_call(
                    unsafe {
                        self.inner
                            .device
                            .reset_command_buffer(command, vk::CommandBufferResetFlags::empty())
                    },
                    "vkResetCommandBuffer",
                    0,
                    &self.inner.memory,
                    ExecutionError::SubmissionFailed,
                )?;
                return Ok(command);
            }
        }
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(recording.pool)
            .level(level)
            .command_buffer_count(1);
        memory_call(
            unsafe { self.inner.device.allocate_command_buffers(&info) },
            "vkAllocateCommandBuffers",
            0,
            &self.inner.memory,
            ExecutionError::SubmissionFailed,
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
        memory_call(
            unsafe { self.inner.device.create_query_pool(&info, None) },
            "vkCreateQueryPool",
            0,
            &self.inner.memory,
            ExecutionError::SubmissionFailed,
        )
        .map(|pool| (pool, queries))
    }

    /// An argument chunk in upload storage.
    pub(crate) fn argument_chunk(
        &self,
        _recording: &mut Recording,
    ) -> Result<Range, ExecutionError> {
        let reused = self
            .opened
            .free_chunks
            .lock()
            .expect("free chunks lock is never poisoned")
            .pop();
        match reused {
            Some(chunk) => Ok(chunk),
            None => self.inner.memory.allocate(
                &self.inner.device,
                self.opened.pools.upload(),
                ARGUMENT_CHUNK_BYTES,
                256,
                &self.opened.stores,
            ),
        }
    }

    pub(crate) fn retire(&self, retired: Retired) {
        self.inner
            .recording
            .lock()
            .expect("recording lock is never poisoned")
            .retired
            .push((retired, Arc::downgrade(&self.opened)));
    }

    /// Settle the charge of the driver's own memory for the device (the
    /// logical device, pipelines, command, query and graph memory) to what
    /// the driver holds now, on the own entry of the store of the device's
    /// own storage ([`MemoryStores::own`]): grown or shrunk, never refused.
    /// The embedder's ledger runs it before each reading of the store.
    /// Returns what the driver grants the process of the device heaps
    /// beyond its usage (budget less usage), read in the one
    /// `VK_EXT_memory_budget` query the charge was settled from: the grant
    /// and the usage the ledger counts are of one instant, so a driver that
    /// shows a free late shows it late in both, and the freed bytes stay
    /// counted as the driver's until it does.
    pub fn reconcile_driver_memory(&self) -> u64 {
        self.inner.memory.reconcile(&self.own(), &self.opened.driver_charge)
    }

    /// The driver memory charged, as last reconciled (`DriverMemory`).
    pub fn driver_charged(&self) -> u64 {
        self.opened
            .driver_charge
            .lock()
            .expect("driver charge lock poisoned")
            .as_ref()
            .map_or(0, Charge::bytes)
    }

    /// The driver memory held now, read exactly (no block in transition),
    /// without charging it: what the next reconciliation will find at
    /// least.
    pub fn driver_held(&self) -> u64 {
        self.inner.memory.unheld().1.driver
    }

    /// `VK_EXT_memory_budget` of the heaps device storage is allocated
    /// from.
    pub fn memory_budget(&self) -> MemoryBudget {
        self.inner.memory.memory_budget()
    }
}

/// `VK_EXT_memory_budget` of the heaps the device's storage is allocated
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

/// Maximum byte length of a single `vkCmdUpdateBuffer` call
/// (`VUID-vkCmdUpdateBuffer-dataSize-00036`).
const UPDATE_BUFFER_MAX_BYTES: usize = 65536;

/// Split `bytes` written at `offset` into `(chunk_offset, chunk)` pieces each
/// within `vkCmdUpdateBuffer`'s 65536-byte limit.
fn update_buffer_chunks(offset: u64, bytes: &[u8]) -> impl Iterator<Item = (u64, &[u8])> {
    assert_eq!(
        offset % 4,
        0,
        "vkCmdUpdateBuffer precondition: dstOffset ({offset}) must be a multiple of 4"
    );
    assert_eq!(
        bytes.len() % 4,
        0,
        "vkCmdUpdateBuffer precondition: dataSize ({}) must be a multiple of 4",
        bytes.len()
    );
    bytes
        .chunks(UPDATE_BUFFER_MAX_BYTES)
        .enumerate()
        .map(move |(index, chunk)| (offset + (index * UPDATE_BUFFER_MAX_BYTES) as u64, chunk))
}

/// Record one or more `vkCmdUpdateBuffer` commands to write `bytes` into
/// `buffer` starting at `offset`, chunked to the 65536-byte Vulkan limit
/// (`VUID-vkCmdUpdateBuffer-dataSize-00036`).
pub(crate) unsafe fn cmd_update_buffer(
    device: &ash::Device,
    command: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: u64,
    bytes: &[u8],
) {
    for (chunk_offset, chunk) in update_buffer_chunks(offset, bytes) {
        device.cmd_update_buffer(command, buffer, chunk_offset, chunk);
    }
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
    result.map_err(|error| {
        let reason = format!("{what} failed: {error}");
        match error {
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_TOO_MANY_OBJECTS => {
                OpenError::Busy(reason)
            }
            _ => OpenError::Creation(reason),
        }
    })
}

impl Inner {
    fn create(
        instance: Arc<Instance>,
        physical: vk::PhysicalDevice,
        info: DeviceInfo,
        properties: Properties,
    ) -> Result<Self, OpenError> {
        let target = info.formation(&properties);
        let usage_before = memory_budget(
            &instance,
            physical,
            &info.allocation.device_heaps,
        )
        .heap_usage_bytes;
        let raw = instance.raw();
        let mut extensions: Vec<&CStr> = device_info::FLOOR_EXTENSIONS.to_vec();
        // Enabled wherever the device supports it, whether or not formation
        // takes the path, so the measurement job's probes reach it.
        let matrix = properties.matrix.holds();
        if matrix {
            extensions.push(ash::khr::cooperative_matrix::NAME);
        }
        if target.f32_atomic_add {
            extensions.push(ash::ext::shader_atomic_float::NAME);
        }
        if properties.fma_32.holds() {
            extensions.push(device_info::SHADER_FMA_EXTENSION);
        }
        if target.subgroup_uniform_control_flow {
            extensions.push(ash::khr::shader_subgroup_uniform_control_flow::NAME);
        }
        if target.maximal_reconvergence {
            extensions.push(ash::khr::shader_maximal_reconvergence::NAME);
        }
        if info.pipeline_statistics {
            extensions.push(ash::khr::pipeline_executable_properties::NAME);
        }
        let extension_names = extensions
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();
        let mut shader_fma = device_info::ShaderFmaFeatures::default();
        shader_fma.float16 = info.shader_fma().float16.into();
        shader_fma.float32 = properties.fma_32.holds().into();
        shader_fma.float64 = info.shader_fma().float64.into();

        let mut explicit = vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR::default()
            .workgroup_memory_explicit_layout(true)
            .workgroup_memory_explicit_layout_scalar_block_layout(true)
            .workgroup_memory_explicit_layout8_bit_access(true)
            .workgroup_memory_explicit_layout16_bit_access(true);
        let mut cooperative =
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        let mut atomic_float = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default()
            .shader_buffer_float32_atomic_add(true);
        let mut uniform_control_flow = vk::PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR::default()
            .shader_subgroup_uniform_control_flow(true);
        let mut maximal_reconvergence =
            vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR::default().shader_maximal_reconvergence(true);
        let mut f11 =
            vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .shader_int8(true)
            .shader_float16(target.float16)
            .storage_buffer8_bit_access(true)
            .scalar_block_layout(true)
            .timeline_semaphore(true)
            .vulkan_memory_model(true)
            .vulkan_memory_model_device_scope(true)
            .shader_subgroup_extended_types(true)
            .shader_buffer_int64_atomics(true)
            .shader_shared_int64_atomics(target.shared_int64_atomics);
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
        if matrix {
            features = features.push_next(&mut cooperative);
        }
        if target.f32_atomic_add {
            features = features.push_next(&mut atomic_float);
        }
        if properties.fma_32.holds() {
            features = features.push_next(&mut shader_fma);
        }
        if target.subgroup_uniform_control_flow {
            features = features.push_next(&mut uniform_control_flow);
        }
        if target.maximal_reconvergence {
            features = features.push_next(&mut maximal_reconvergence);
        }
        let mut executables =
            vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default().pipeline_executable_info(true);
        if info.pipeline_statistics {
            features = features.push_next(&mut executables);
        }
        let priorities = [1.0f32; 2];
        // The second queue, of the first's family or of a family of its own.
        let side_family = info.side_family;
        let shares_family = side_family == Some(info.queue_family);
        let mut queues = vec![vk::DeviceQueueCreateInfo::default()
            .queue_family_index(info.queue_family)
            .queue_priorities(&priorities[..if shares_family { 2 } else { 1 }])];
        if let Some(family) = side_family.filter(|_| !shares_family) {
            queues.push(vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities[..1]));
        }
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&extension_names)
            .push_next(&mut features);
        let device = creation(
            unsafe { raw.create_device(physical, &create_info, None) },
            "vkCreateDevice",
        )?;
        // From here every created object is destroyed by `Drop` (null
        // handles are ignored), so partial construction cleans up.
        let queue = unsafe { device.get_device_queue(info.queue_family, 0) };
        let side = side_family.map(|family| {
            Mutex::new(Side {
                raw: unsafe { device.get_device_queue(family, u32::from(shares_family)) },
                pool: vk::CommandPool::null(),
                command: vk::CommandBuffer::null(),
                fence: vk::Fence::null(),
                fence_dirty: false,
            })
        });
        let executables = info
            .pipeline_statistics
            .then(|| ash::khr::pipeline_executable_properties::Device::new(raw, &device));
        // The tables: the device's own block on the device placement's
        // first kind, held as the driver's memory (`Inner::tables`).
        let formed = Memory::new(&info, instance.clone(), physical, usage_before)
            .map_err(|reason| OpenError::Creation(reason.into()))
            .and_then(|memory| {
                let tables = memory
                    .driver_block(&device, memory.device_kind(), seismic_lang::source::VULKAN_TABLES.len() as u64)
                    .map_err(OpenError::of)?;
                Ok((memory, tables))
            });
        let (memory, tables) = match formed {
            Ok(formed) => formed,
            Err(error) => {
                let _destroying = PIPELINES.write().unwrap_or_else(PoisonError::into_inner);
                unsafe { device.destroy_device(None) };
                return Err(error);
            }
        };
        let mut inner = Self {
            device,
            info,
            properties,
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
            lost: OnceLock::new(),
            #[cfg(test)]
            on_handover_wait: Mutex::new(None),
            side,
            executables,
            recording: Mutex::new(Recording {
                pool: vk::CommandPool::null(),
                free_commands: Vec::new(),
                free_queries: Vec::new(),
                retired: Vec::new(),
            }),
            graphs: std::sync::atomic::AtomicU64::new(0),
            tables,
            instance,
            destroyed: Ends(Arc::default()),
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
                device.create_command_pool(&command_pool_info(inner.info.queue_family), None)
            },
            "vkCreateCommandPool",
        )?;
        inner
            .recording
            .get_mut()
            .expect("recording lock is never poisoned")
            .pool = pool;
        Ok(inner)
    }

    fn retire_up_to(&self, recording: &mut Recording, completed: u64) {
        let mut touched_opens: Vec<Arc<Opened>> = Vec::new();
        let mut index = 0;
        while index < recording.retired.len() {
            if recording.retired[index].0.value <= completed {
                let (retired, owner) = recording.retired.swap_remove(index);
                match retired.command {
                    Some(RetiredCommand::Primary(command)) => recording.free_commands.push(command),
                    Some(RetiredCommand::Secondary(command)) => unsafe {
                        self.device
                            .free_command_buffers(recording.pool, &[command])
                    },
                    None => {}
                }
                recording.free_queries.extend(retired.queries);
                match owner.upgrade() {
                    Some(opened) => {
                        if !retired.arguments.is_empty() {
                            opened
                                .free_chunks
                                .lock()
                                .expect("free chunks lock is never poisoned")
                                .extend(retired.arguments);
                            if !touched_opens.iter().any(|existing| Arc::ptr_eq(existing, &opened)) {
                                touched_opens.push(opened);
                            }
                        }
                    }
                    None => {
                        for chunk in retired.arguments {
                            self.memory.release(&self.device, &chunk);
                        }
                    }
                }
            } else {
                index += 1;
            }
        }
        let cap = free_cap(recording.retired.len());
        let excess_commands = trim_to_cap(&mut recording.free_commands, cap);
        if !excess_commands.is_empty() {
            unsafe {
                self.device
                    .free_command_buffers(recording.pool, &excess_commands)
            };
        }
        for (pool, _) in trim_to_cap(&mut recording.free_queries, cap) {
            unsafe { self.device.destroy_query_pool(pool, None) };
        }
        for opened in touched_opens {
            let excess_chunks = {
                let mut free = opened
                    .free_chunks
                    .lock()
                    .expect("free chunks lock is never poisoned");
                trim_to_cap(&mut free, cap)
            };
            for chunk in excess_chunks {
                self.memory.release(&self.device, &chunk);
            }
        }
    }

    fn retire_completed_best_effort(&self) {
        let Ok(completed) = (unsafe { self.device.get_semaphore_counter_value(self.timeline) }) else {
            return;
        };
        let mut recording = self
            .recording
            .lock()
            .expect("recording lock is never poisoned");
        self.retire_up_to(&mut recording, completed);
    }

    fn create_staging(&self, pools: &DevicePools, stores: &MemoryStores) -> Result<Staging, ExecutionError> {
        let device = &self.device;
        let pool = call(
            unsafe {
                device.create_command_pool(&command_pool_info(self.info.queue_family), None)
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
            .allocate(device, pools.staging(), STAGING_BYTES, 256, stores)
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
            last_value: 0,
        })
    }
}

impl Device {
    /// Whether the second queue is open (`DeviceInfo::side_family`): what zeroes
    /// pool blocks and runs transfers beside the first's work.
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
        self.inner.vk("vkQueueSubmit2", |device| unsafe {
            device.queue_submit2(side.raw, submits, vk::Fence::null())
        })
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
    /// On a device without a second queue, which zeroes on its first
    /// (`Device::allocate_block`).
    pub(crate) fn zero(&self, regions: &[(&Range, u64, u64)]) -> Result<(), ExecutionError> {
        self.check_lost()?;
        let device = &self.device;
        let mut side = self
            .side
            .as_ref()
            .expect("only a device with a second queue zeroes on it")
            .lock()
            .expect("side queue lock is never poisoned");
        // Formed whole or not at all: the pool, then its command buffer and
        // the fence, each kept only once all are formed (`Drop` destroys a
        // formed pool and fence).
        if side.fence == vk::Fence::null() {
            let pool = call(
                unsafe { device.create_command_pool(&command_pool_info(self.info.side_family.expect("a second queue has a family")), None) },
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
        let fence = side.fence;
        if let Err(error) = clean_dirty_fence(
            &mut side.fence_dirty,
            || {
                self.vk("vkWaitForFences", |device| unsafe {
                    device.wait_for_fences(&[fence], true, u64::MAX)
                })
            },
            || self.vk("vkResetFences", |device| unsafe { device.reset_fences(&[fence]) }),
        ) {
            return Err(match error {
                ExecutionError::DeviceLost(_) => error,
                other => self.lose(ExecutionError::DeviceLost(format!(
                    "side-queue fence recovery failed: {other}"
                ))),
            });
        }
        let command = side.command;
        self.vk("vkResetCommandBuffer", |device| unsafe {
            device.reset_command_buffer(command, vk::CommandBufferResetFlags::empty())
        })?;
        self.vk("vkBeginCommandBuffer", |device| unsafe {
            device.begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        })?;
        unsafe {
            for &(range, offset, bytes) in regions {
                device.cmd_fill_buffer(command, range.raw(), range.offset() + offset, bytes, 0);
            }
        }
        self.vk("vkEndCommandBuffer", |device| unsafe {
            device.end_command_buffer(command)
        })?;
        let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(command)];
        self.vk("vkQueueSubmit2", |device| unsafe {
            device.queue_submit2(
                side.raw,
                &[vk::SubmitInfo2::default().command_buffer_infos(&commands)],
                fence,
            )
        })?;
        side.fence_dirty = true;
        clean_dirty_fence(
            &mut side.fence_dirty,
            || {
                self.vk("vkWaitForFences", |device| unsafe {
                    device.wait_for_fences(&[fence], true, u64::MAX)
                })
            },
            || self.vk("vkResetFences", |device| unsafe { device.reset_fences(&[fence]) }),
        )
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
            if !self.is_lost() {
                let _ = device.device_wait_idle();
            }
            let recording = self
                .recording
                .get_mut()
                .expect("recording lock is never poisoned");
            for (retired, _) in recording.retired.drain(..) {
                recording.free_queries.extend(retired.queries);
                for chunk in retired.arguments {
                    self.memory.release(device, &chunk);
                }
            }
            for (pool, _) in recording.free_queries.drain(..) {
                device.destroy_query_pool(pool, None);
            }
            device.destroy_command_pool(recording.pool, None);
            if let Some(side) = self.side.as_mut() {
                let side = side.get_mut().expect("side queue lock is never poisoned");
                device.destroy_fence(side.fence, None);
                device.destroy_command_pool(side.pool, None);
            }
            self.memory.release(device, &self.tables);
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
    use ash::vk::Handle;
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
            .map(|description| Device::open(description.info.uuid, crate::unbounded_stores(&description.info), &Default::default()).expect("a device meeting the floor opens"))
            .collect()
    }

    /// A reconciliation under block churn is exact both ways: pooled
    /// blocks and blocks of their own of 256 MiB created and freed on one
    /// thread while another reconciles in a loop leave the charge at what
    /// the driver holds once churn ends, holding no block (it moved by far
    /// less than one).
    #[test]
    fn a_reconciliation_under_churn_is_exact() {
        for device in devices() {
            let memory = &device.inner.memory;
            let driver = device.own();
            device.reconcile_driver_memory();
            let before = device.driver_charged();
            let stop = Arc::new(AtomicBool::new(false));
            let reconciler = {
                let (device, stop) = (device.clone(), stop.clone());
                std::thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        device.reconcile_driver_memory();
                    }
                })
            };
            let kind = device.opened.pools.device().kinds()[0];
            for _ in 0..20 {
                drop(device.allocate(BLOCK, 256, &driver).expect("a pooled block allocates"));
                drop(device.allocate_block(kind, BLOCK, &driver, seismic_memory::SlabFill::Zeroed).expect("a block of its own allocates"));
            }
            stop.store(true, Ordering::Relaxed);
            reconciler.join().unwrap();
            device.reconcile_driver_memory();
            let unheld = memory.unheld().1.driver;
            let charged = device.driver_charged();
            assert_eq!(charged, unheld, "the charge is what the driver holds");
            assert!(charged.abs_diff(before) < BLOCK / 2, "a block charged as the driver's: moved {}", charged.abs_diff(before));
        }
    }

    #[test]
    fn classify_vk_error_maps_all_oom_and_device_loss_codes() {
        use seismic_compiler::errors::RefusalReading;

        let budget = || crate::MemoryBudget {
            heap_budget_bytes: 1000,
            heap_usage_bytes: 900,
        };
        let objects = || (4096, 4096);

        let lost = classify_vk_error(
            vk::Result::ERROR_DEVICE_LOST,
            "vkCreateBuffer",
            64,
            budget,
            objects,
            ExecutionError::AllocationFailed,
        );
        assert!(matches!(lost, ExecutionError::DeviceLost(_)));

        let dev_oom = classify_vk_error(
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            "vkAllocateMemory",
            64,
            budget,
            objects,
            ExecutionError::AllocationFailed,
        );
        assert!(dev_oom.capacity_refusal().is_some(), "{dev_oom:?}");
        assert!(matches!(
            dev_oom,
            ExecutionError::DeviceRefused {
                required: 64,
                reading: RefusalReading::HeapBudget { budget: 1000, usage: 900 },
            }
        ));

        for host_code in [vk::Result::ERROR_OUT_OF_HOST_MEMORY, vk::Result::ERROR_MEMORY_MAP_FAILED] {
            let host_oom = classify_vk_error(
                host_code,
                "vkMapMemory",
                64,
                budget,
                objects,
                ExecutionError::AllocationFailed,
            );
            assert!(host_oom.capacity_refusal().is_some(), "{host_oom:?}");
            assert!(matches!(
                host_oom,
                ExecutionError::DeviceRefused { required: 64, reading: RefusalReading::Host }
            ));
        }

        let too_many = classify_vk_error(
            vk::Result::ERROR_TOO_MANY_OBJECTS,
            "vkAllocateMemory",
            64,
            budget,
            objects,
            ExecutionError::AllocationFailed,
        );
        assert!(too_many.capacity_refusal().is_some(), "{too_many:?}");
        assert!(matches!(
            too_many,
            ExecutionError::DeviceRefused {
                required: 64,
                reading: RefusalReading::MemoryObjects { limit: 4096, held: 4096 },
            }
        ));

        let other = classify_vk_error(
            vk::Result::ERROR_INITIALIZATION_FAILED,
            "vkCreateQueryPool",
            0,
            budget,
            objects,
            ExecutionError::SubmissionFailed,
        );
        assert!(other.capacity_refusal().is_none(), "{other:?}");
        assert!(matches!(other, ExecutionError::SubmissionFailed(_)));
    }

    #[test]
    fn free_cap_and_trim_to_cap_retain_steady_state_prefix_and_evict_overflow_tail() {
        assert_eq!(free_cap(0), 1);
        assert_eq!(free_cap(3), 4);

        let mut pool = vec![10, 20, 30, 40, 50];
        let evicted = trim_to_cap(&mut pool, free_cap(1));
        assert_eq!(pool, vec![10, 20]);
        assert_eq!(evicted, vec![30, 40, 50]);

        let none = trim_to_cap(&mut pool, free_cap(2));
        assert!(none.is_empty());
        assert_eq!(pool, vec![10, 20]);
    }

    #[test]
    fn update_buffer_chunks_splits_at_vulkan_64kib_limit() {
        assert_eq!(update_buffer_chunks(0, &[]).count(), 0);

        let small = vec![0xABu8; 16];
        let chunks: Vec<_> = update_buffer_chunks(64, &small).collect();
        assert_eq!(chunks, vec![(64, small.as_slice())]);

        let exact = vec![0xCDu8; UPDATE_BUFFER_MAX_BYTES];
        let chunks: Vec<_> = update_buffer_chunks(0, &exact).collect();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], (0, exact.as_slice()));

        let large: Vec<u8> = (0..(UPDATE_BUFFER_MAX_BYTES * 2 + 16))
            .map(|i| (i % 251) as u8)
            .collect();
        let chunks: Vec<_> = update_buffer_chunks(128, &large).collect();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].0, 128);
        assert_eq!(chunks[0].1.len(), UPDATE_BUFFER_MAX_BYTES);
        assert_eq!(chunks[1].0, 128 + UPDATE_BUFFER_MAX_BYTES as u64);
        assert_eq!(chunks[1].1.len(), UPDATE_BUFFER_MAX_BYTES);
        assert_eq!(chunks[2].0, 128 + (UPDATE_BUFFER_MAX_BYTES * 2) as u64);
        assert_eq!(chunks[2].1.len(), 16);
        let reassembled: Vec<u8> = chunks.into_iter().flat_map(|(_, c)| c.iter().copied()).collect();
        assert_eq!(reassembled, large);
    }

    #[test]
    #[should_panic(expected = "dataSize")]
    fn update_buffer_chunks_rejects_unaligned_length() {
        let _ = update_buffer_chunks(0, &[1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "dstOffset")]
    fn update_buffer_chunks_rejects_unaligned_offset() {
        let _ = update_buffer_chunks(2, &[1, 2, 3, 4]);
    }

    fn test_queue(held: Vec<Held>) -> Queue {
        Queue {
            raw: vk::Queue::null(),
            next: held.last().map_or(1, |h| h.value + 1),
            held: held.into(),
            waiting: None,
            refused: None,
        }
    }

    fn test_held(value: u64, gates: Vec<(vk::Semaphore, u64)>) -> Held {
        Held {
            command: vk::CommandBuffer::null(),
            value,
            waits: gates,
            signals: Vec::new(),
            handed: Handed::default(),
        }
    }

    #[test]
    fn release_queue_leaves_waiting_none_when_submit_fails_with_non_device_lost() {
        let gate = (vk::Semaphore::from_raw(0x1234), 1);
        let mut queue = test_queue(vec![test_held(1, vec![gate])]);
        let mut refused_value = None;

        release_queue(
            &mut queue,
            GateOpening::Signalled,
            |_| Ok(false),
            |_| Err(ExecutionError::SubmissionFailed("transient host oom".into())),
            || {},
            |queue, value, error| {
                queue.held.clear();
                queue.refused = Some((value, Refusal::Raised(error)));
                refused_value = Some(value);
            },
        );

        assert_eq!(refused_value, Some(1));
        assert!(queue.waiting.is_none(), "failed submit must not leave gate semaphore in queue.waiting");
        assert!(queue.held.is_empty());
    }

    #[test]
    fn release_queue_clears_waiting_when_raised_query_fails_with_non_device_lost() {
        let gate = (vk::Semaphore::from_raw(0x5678), 1);
        let mut queue = test_queue(vec![test_held(2, Vec::new())]);
        queue.waiting = Some((vec![gate], Handed::default()));
        let mut refused_value = None;

        release_queue(
            &mut queue,
            GateOpening::Signalled,
            |_| Err(ExecutionError::SubmissionFailed("query failed".into())),
            |_| panic!("submit must not run when waiting gate query fails"),
            || {},
            |queue, value, error| {
                queue.held.clear();
                queue.refused = Some((value, Refusal::Raised(error)));
                refused_value = Some(value);
            },
        );

        assert_eq!(refused_value, Some(2));
        assert!(queue.waiting.is_none(), "failed gate query must clear queue.waiting");
    }

    #[test]
    fn release_queue_sets_waiting_only_after_submit_succeeds() {
        let gate = (vk::Semaphore::from_raw(0x9abc), 1);
        let mut queue = test_queue(vec![test_held(1, vec![gate]), test_held(2, Vec::new())]);
        let mut submitted = Vec::new();

        release_queue(
            &mut queue,
            GateOpening::Signalled,
            |gates| Ok(gates.is_empty()),
            |held| {
                submitted.push(held.value);
                Ok(())
            },
            || {},
            |_, _, _| panic!("unexpected refusal"),
        );

        assert_eq!(submitted, vec![1]);
        assert!(queue.waiting.is_some());
        assert_eq!(queue.held.len(), 1);
        assert_eq!(queue.held[0].value, 2);
    }

    #[test]
    fn staging_safe_to_destroy_skips_destruction_on_non_device_lost_wait_failure() {
        assert!(staging_safe_to_destroy(0, |_| panic!("zero last_value must not wait")));
        assert!(staging_safe_to_destroy(5, |v| {
            assert_eq!(v, 5);
            Ok(())
        }));
        assert!(staging_safe_to_destroy(5, |_| Err(ExecutionError::DeviceLost("lost".into()))));
        assert!(!staging_safe_to_destroy(5, |_| Err(ExecutionError::SubmissionFailed("wait failed".into()))));
    }

    #[test]
    fn clean_dirty_fence_clears_flag_only_when_both_wait_and_reset_succeed() {
        let mut dirty = false;
        assert!(clean_dirty_fence(
            &mut dirty,
            || panic!("clean fence must not wait"),
            || panic!("clean fence must not reset"),
        )
        .is_ok());
        assert!(!dirty);

        // Wait fails -> reset must not be called, flag stays dirty.
        let mut dirty = true;
        assert!(clean_dirty_fence(
            &mut dirty,
            || Err(ExecutionError::SubmissionFailed("wait failed".into())),
            || panic!("reset must not run if wait failed"),
        )
        .is_err());
        assert!(dirty);

        // Wait succeeds, reset fails -> flag stays dirty.
        let mut dirty = true;
        assert!(clean_dirty_fence(
            &mut dirty,
            || Ok(()),
            || Err(ExecutionError::SubmissionFailed("reset failed".into())),
        )
        .is_err());
        assert!(dirty);

        // Both succeed -> flag cleared.
        let mut dirty = true;
        assert!(clean_dirty_fence(&mut dirty, || Ok(()), || Ok(())).is_ok());
        assert!(!dirty);
    }

    #[test]
    fn wait_for_held_room_blocks_at_limit_and_wakes_on_release_and_loss() {
        let queue = Arc::new(Mutex::new(test_queue(vec![test_held(1, Vec::new()), test_held(2, Vec::new())])));
        let handover = Arc::new(Condvar::new());
        let lost = Arc::new(OnceLock::<ExecutionError>::new());

        // Below limit returns immediately.
        assert!(wait_for_held_room(&queue, &handover, &lost, 3).is_ok());

        // At limit blocks until an entry is drained and handover is notified.
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let waiter = {
            let (queue, handover, lost) = (Arc::clone(&queue), Arc::clone(&handover), Arc::clone(&lost));
            std::thread::spawn(move || {
                let _ = done_tx.send(wait_for_held_room(&queue, &handover, &lost, 2));
            })
        };
        assert!(done_rx.recv_timeout(std::time::Duration::from_millis(20)).is_err());
        {
            let mut guard = queue.lock().unwrap();
            guard.held.pop_front();
            handover.notify_all();
        }
        assert!(
            done_rx
                .recv_timeout(std::time::Duration::from_secs(2))
                .expect("release must unblock wait_for_held_room")
                .is_ok()
        );
        waiter.join().unwrap();

        // At limit wakes with DeviceLost if the device is lost.
        {
            let mut guard = queue.lock().unwrap();
            guard.held.push_back(test_held(3, Vec::new()));
        }
        let (lost_tx, lost_rx) = std::sync::mpsc::channel();
        let waiter = {
            let (queue, handover, lost) = (Arc::clone(&queue), Arc::clone(&handover), Arc::clone(&lost));
            std::thread::spawn(move || {
                let _ = lost_tx.send(wait_for_held_room(&queue, &handover, &lost, 2));
            })
        };
        assert!(lost_rx.recv_timeout(std::time::Duration::from_millis(20)).is_err());
        {
            let _guard = queue.lock().unwrap();
            let _ = lost.set(ExecutionError::DeviceLost("lost".into()));
            handover.notify_all();
        }
        let result = lost_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("device loss must unblock wait_for_held_room");
        assert!(matches!(result, Err(ExecutionError::DeviceLost(_))));
        waiter.join().unwrap();
    }
}
