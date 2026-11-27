//! Transfers: copies between host memory and a device's storage (either
//! way) that run beside its compute, queued in batches the host numbers and
//! observes complete.
//!
//! A transfer is what moves bytes between storage the host addresses and
//! storage it does not. Where the host addresses both ends it copies in
//! place and forms no transfer. Per backend:
//!
//! | Backend | Transfer |
//! |---|---|
//! | CUDA | a stream of its own, one copy a run by where its ends lie (`cuMemcpyHtoDAsync`, `cuMemcpyDtoHAsync`, `cuMemcpyDtoDAsync`; host ends pinned and mapped), then `cuStreamWriteValue32` of the batch's value into a mapped word (`seismic_cuda::transfer`) |
//! | CPU | the device's memory is the host's: each run is copied on the calling thread, complete on return |
//! | Metal | none: every storage is host-addressed (shared or wired), so the host writes in place ([`TransferError::InPlace`]) |
//! | Vulkan | `vkCmdCopyBuffer` regions on the compute family's second queue, signalling a timeline semaphore of its own with the batch's value (`seismic_vulkan::transfer`); none on a family of one queue, which reserves no device storage |
//!
//! A batch is a sequence of runs and signals: a signal sets a 4-byte word of
//! the device's storage to its value once every run of the batch before it
//! has landed, so device work can wait for one record of a batch by an
//! equality on its word (CUDA: `cuStreamWriteValue32` on the copy stream;
//! Vulkan: `vkCmdFillBuffer` behind a barrier over the copies before it;
//! CPU: a release store after the copies).
//!
//! Batches complete in the order they were queued. The caller orders a
//! transfer against device work: it queues one only into bytes no queued
//! work reaches, and lets work read them only once it has seen the batch
//! complete (a host gate it answers after [`TransferInner::completed`]).
//! The source's bytes stay untouched until then. Each batch is recorded as
//! a device use of its runs' sources and destinations, on the device's
//! transfer queue: host access to them, and their storage's release or
//! unbacking, wait for it as for compute work.

use crate::api::device::DeviceInner;
use crate::api::tensor::TensorInner;
use crate::backends::OpenedKind;
use crate::completion::{Committed, DeviceCompletion, RouteSubmission};
use crate::driver::{typed_buffer, DeviceQueue};
use seismic_compiler::errors::ExecutionError;
pub use seismic_io::{DeviceId, FenceCapability, FenceError, ProducerId, TimelineObserver, TimelinePoint};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Why a device forms no transfer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransferError {
    /// The backend's every storage is host-addressed: the host writes in
    /// place, and nothing transfers.
    InPlace,
    /// A run's or a signal's tensors are another device's, or lie outside
    /// the bytes their views reach.
    Run(String),
    /// Refused before any item of the batch was queued on the device (e.g.
    /// CUDA event-allocation failure, invalid item, adoption wait failure, or
    /// queue already faulted). No stream work was issued for this batch and no
    /// fence was installed.
    RefusedBeforeIssue(ExecutionError),
    /// A device/driver failure that may have occurred after earlier items of
    /// the batch were already queued on the transfer stream.
    Execution(ExecutionError),
}

impl TransferError {
    /// Whether this failure was definitely refused before any item of the batch
    /// reached the device transfer queue (so no stream work was issued for this
    /// call and no fence was installed).
    pub fn refused_before_issue(&self) -> bool {
        matches!(self, Self::InPlace | Self::Run(_) | Self::RefusedBeforeIssue(_))
    }
}

impl fmt::Display for TransferError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InPlace => formatter.write_str("the device's storage is host-addressed: the host writes it in place"),
            Self::Run(why) => write!(formatter, "a transfer run is invalid: {why}"),
            Self::RefusedBeforeIssue(error) => write!(formatter, "a transfer was refused before issue: {error}"),
            Self::Execution(error) => write!(formatter, "a transfer failed: {error}"),
        }
    }
}

impl std::error::Error for TransferError {}

/// One copy of a batch: `bytes` bytes of `source` from `source_offset`
/// into `destination` from `destination_offset`, both offsets within the
/// bytes the tensors' views reach: host memory into the device's storage,
/// or back.
pub struct TransferRun<'a> {
    pub source: &'a TensorInner,
    pub source_offset: u64,
    pub destination: &'a TensorInner,
    pub destination_offset: u64,
    pub bytes: u64,
}

/// One step of a batch, in order: a copy, or a signal: the i32 word at byte
/// `offset` of `word`'s view (this device's storage) set to `value` once
/// every run of the batch before it has landed.
pub enum TransferItem<'a> {
    Run(TransferRun<'a>),
    Signal { word: &'a TensorInner, offset: u64, value: i32 },
}

enum TransferKind {
    /// The device's memory is the host's: each batch is copied as queued;
    /// `done` is the last batch's value.
    Cpu { done: AtomicU64 },
    Cuda(seismic_cuda::transfer::TransferQueue),
    #[cfg(not(target_os = "macos"))]
    Vulkan(Arc<seismic_vulkan::transfer::TransferQueue>),
}

/// The queue batches run on, and how their completion is observed. Shared
/// with the fences its batches leave on the allocations they reach.
struct Queue {
    kind: TransferKind,
}

impl Queue {
    /// Whether batch `value` is complete, or why it never will be: the
    /// queue faulted or the device is lost. Each backend reports that only
    /// once no copy runs any more: its queue has drained, or its context or
    /// device is lost.
    fn completed(&self, value: u64) -> Result<bool, ExecutionError> {
        match &self.kind {
            TransferKind::Cpu { done } => Ok(done.load(Ordering::Acquire) >= value),
            TransferKind::Cuda(queue) => queue.completed(value as u32),
            #[cfg(not(target_os = "macos"))]
            TransferKind::Vulkan(queue) => queue.reached(value),
        }
    }

    /// Wait at most `timeout` for batch `value`, sleeping until the device
    /// completes it: a CPU batch completed as it was queued.
    fn wait(&self, value: u64, timeout: Duration) -> Result<bool, ExecutionError> {
        match &self.kind {
            TransferKind::Cpu { .. } => self.completed(value),
            TransferKind::Cuda(queue) => queue.wait(value as u32, timeout),
            #[cfg(not(target_os = "macos"))]
            TransferKind::Vulkan(queue) => queue.wait(value, timeout),
        }
    }
}

/// A queued batch, as the allocations its runs read and write observe it:
/// complete, or faulted once no copy of it runs ([`Queue::completed`]).
struct BatchFence {
    queue: Arc<Queue>,
    value: u64,
}

impl DeviceCompletion for BatchFence {
    fn completed(&self) -> Result<bool, ExecutionError> {
        self.queue.completed(self.value)
    }

    fn ended(&self) -> Result<(), ExecutionError> {
        while !self.queue.wait(self.value, Duration::from_secs(1))? {
            std::thread::yield_now();
        }
        Ok(())
    }
}

/// Allocate a process-unique monotonic producer queue identifier for a
/// transfer or fill-copy queue.
pub(super) fn next_queue_id() -> u64 {
    static NEXT_QUEUE_ID: AtomicU64 = AtomicU64::new(1);
    NEXT_QUEUE_ID.fetch_add(1, Ordering::Relaxed)
}

/// Process-unique identity of one submitted transfer batch: scoped by opened
/// device, producer queue instance, and monotonic batch generation/value so
/// two batches from different queues or devices with the same numeric `u64`
/// value are never treated as equal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TransferSyncId {
    pub device_id: u64,
    pub producer_queue_id: u64,
    pub value: u64,
}

#[derive(Clone)]
enum TransferSyncHandle {
    /// Host-only timeline (CPU transfer or CPU compute submission).
    Cpu(Option<Arc<seismic_io::HostTimeline>>),
    /// CUDA transfer-stream completion event (and optional backing queue for host wait).
    Cuda {
        event: Option<Arc<seismic_cuda::Event>>,
        queue: Option<Arc<Queue>>,
    },
    /// CUDA compute-stream completion event retained inside `RouteSubmission`.
    CudaSubmission(Arc<RouteSubmission>),
    /// Metal 64-bit monotonic `MTLSharedEvent` shared across compute/blit/host.
    #[cfg(target_os = "macos")]
    Metal(Arc<seismic_metal::SharedEvent>),
    /// Vulkan transfer-queue `VK_KHR_timeline_semaphore`.
    #[cfg(not(target_os = "macos"))]
    Vulkan(Arc<seismic_vulkan::transfer::TransferQueue>),
    /// Vulkan compute-queue `VK_KHR_timeline_semaphore` retained inside `RouteSubmission`.
    #[cfg(not(target_os = "macos"))]
    VulkanSubmission {
        semaphore: seismic_vulkan::vk::Semaphore,
        route: Arc<RouteSubmission>,
    },
}

/// Unified typed timeline fence handle (`TimelineFence`) bridging
/// `seismic-runtime` queues (CUDA `CUevent` pool, Metal `MTLSharedEvent`,
/// Vulkan `VK_KHR_timeline_semaphore`, CPU `HostOnly`) with `seismic-io` v2's
/// [`TimelinePoint`] and [`TimelineObserver`].
#[derive(Clone)]
pub struct TransferSync {
    id: TransferSyncId,
    handle: TransferSyncHandle,
}

/// Unified alias for [`TransferSync`] matching the `seismic-io` v2 architecture.
pub type TimelineFence = TransferSync;

impl TransferSync {
    /// The unique `(device_id, producer_queue_id, value)` identity of this
    /// transfer batch or compute submission.
    pub fn id(&self) -> TransferSyncId {
        self.id
    }

    /// Typed [`TimelinePoint`] carrying `(DeviceId, ProducerId, value, capability)`.
    /// CPU fences and unrecorded CUDA handles are explicitly typed
    /// [`FenceCapability::HostOnly`]; GPU-waitable handles are
    /// [`FenceCapability::DeviceAndHost`].
    pub fn timeline_point(&self) -> TimelinePoint {
        let capability = match &self.handle {
            TransferSyncHandle::Cpu(_) | TransferSyncHandle::Cuda { event: None, .. } => {
                FenceCapability::HostOnly
            }
            TransferSyncHandle::Cuda { event: Some(_), .. }
            | TransferSyncHandle::CudaSubmission(_) => FenceCapability::DeviceAndHost,
            #[cfg(target_os = "macos")]
            TransferSyncHandle::Metal(_) => FenceCapability::DeviceAndHost,
            #[cfg(not(target_os = "macos"))]
            TransferSyncHandle::Vulkan(_) | TransferSyncHandle::VulkanSubmission { .. } => {
                FenceCapability::DeviceAndHost
            }
        };
        TimelinePoint::new(
            DeviceId(self.id.device_id),
            ProducerId(self.id.producer_queue_id),
            self.id.value,
            capability,
        )
    }

    /// Whether `self` and `other` are one batch's handle, so following
    /// one follows the other. Two batches from different devices, different
    /// producer queues, or different generations are never equal.
    pub fn same_batch(&self, other: &Self) -> bool {
        self.id == other.id
    }

    /// Validate that this fence belongs to `device` and carries a completion
    /// handle supported by `device`'s backend/queue. Rejects foreign-device,
    /// `HostOnly`-on-GPU, or unsupported handles loudly rather than silently
    /// ignoring them.
    pub(super) fn validate_for_device(&self, device: &DeviceInner) -> Result<(), ExecutionError> {
        let expected = device.device_id();
        if self.id.device_id != expected {
            return Err(ExecutionError::SubmissionFailed(format!(
                "transfer sync {:?} belongs to device {}, not device {expected}",
                self.id, self.id.device_id,
            )));
        }
        // On GPU backends, enforce `TimelinePoint::validate_for_device_wait` so
        // `FenceCapability::HostOnly` is rejected by construction.
        if !matches!(device.kind, OpenedKind::Cpu(_)) {
            self.timeline_point()
                .validate_for_device_wait(DeviceId(expected))
                .map_err(|err| ExecutionError::SubmissionFailed(err.to_string()))?;
        }
        match (&device.kind, &self.handle) {
            (OpenedKind::Cpu(_), TransferSyncHandle::Cpu(_)) => Ok(()),
            (OpenedKind::Cuda(_), TransferSyncHandle::Cuda { event: Some(_), .. })
            | (OpenedKind::Cuda(_), TransferSyncHandle::CudaSubmission(_)) => Ok(()),
            #[cfg(target_os = "macos")]
            (OpenedKind::Metal(_), TransferSyncHandle::Metal(_)) => Ok(()),
            #[cfg(not(target_os = "macos"))]
            (OpenedKind::Vulkan(_), TransferSyncHandle::Vulkan(_))
            | (OpenedKind::Vulkan(_), TransferSyncHandle::VulkanSubmission { .. }) => Ok(()),
            _ => Err(ExecutionError::SubmissionFailed(format!(
                "transfer sync {:?} is not supported on backend {:?}",
                self.id,
                device.info().backend(),
            ))),
        }
    }

    /// Construct a host-only CPU fence already reached at `value`.
    pub(super) fn from_cpu_reached(device_id: u64, producer_queue_id: u64, value: u64) -> Self {
        let tl = seismic_io::HostTimeline::new(DeviceId(device_id), ProducerId(producer_queue_id));
        tl.advance(value);
        Self {
            id: TransferSyncId {
                device_id,
                producer_queue_id,
                value,
            },
            handle: TransferSyncHandle::Cpu(Some(tl)),
        }
    }

    /// Construct a `TransferSync` from a committed compute [`RouteSubmission`].
    pub(super) fn from_compute_submission(
        device: &DeviceInner,
        route: &Arc<RouteSubmission>,
        producer_queue_id: u64,
        value: u64,
        #[cfg(target_os = "macos")] metal_event: Option<Arc<seismic_metal::SharedEvent>>,
    ) -> Self {
        let device_id = device.device_id();
        match route.committed() {
            Committed::Cpu { .. } => Self::from_cpu_reached(device_id, producer_queue_id, value),
            Committed::Cuda(_) => Self {
                id: TransferSyncId {
                    device_id,
                    producer_queue_id,
                    value,
                },
                handle: TransferSyncHandle::CudaSubmission(Arc::clone(route)),
            },
            #[cfg(target_os = "macos")]
            Committed::Metal(_) => Self {
                id: TransferSyncId {
                    device_id,
                    producer_queue_id,
                    value,
                },
                handle: TransferSyncHandle::Metal(
                    metal_event.expect("Metal signaled submission must attach SharedEvent"),
                ),
            },
            #[cfg(not(target_os = "macos"))]
            Committed::Vulkan(submission) => {
                if let Some((semaphore, sem_val, _)) = submission.ready_timeline_wait() {
                    Self {
                        id: TransferSyncId {
                            device_id,
                            producer_queue_id,
                            value: sem_val,
                        },
                        handle: TransferSyncHandle::VulkanSubmission {
                            semaphore,
                            route: Arc::clone(route),
                        },
                    }
                } else {
                    Self::from_cpu_reached(device_id, producer_queue_id, value)
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    pub(super) fn from_metal_event(
        device_id: u64,
        producer_queue_id: u64,
        value: u64,
        event: Arc<seismic_metal::SharedEvent>,
    ) -> Self {
        Self {
            id: TransferSyncId {
                device_id,
                producer_queue_id,
                value,
            },
            handle: TransferSyncHandle::Metal(event),
        }
    }

    #[cfg(target_os = "macos")]
    pub(super) fn metal_wait(&self) -> Option<(&seismic_metal::SharedEvent, u64)> {
        match &self.handle {
            TransferSyncHandle::Metal(event) => Some((event.as_ref(), self.id.value)),
            _ => None,
        }
    }

    #[cfg(all(test, not(target_os = "macos")))]
    pub(super) fn cpu_for_test(device_id: u64, value: u64) -> Self {
        Self::from_cpu_reached(device_id, 1, value)
    }

    pub(super) fn cuda_event(&self) -> Option<&seismic_cuda::Event> {
        match &self.handle {
            TransferSyncHandle::Cuda { event, .. } => event.as_deref(),
            TransferSyncHandle::CudaSubmission(route) => match route.committed() {
                Committed::Cuda(sub) => Some(sub.completion_event()),
                _ => None,
            },
            _ => None,
        }
    }

    /// On Vulkan, the `(semaphore, value, owner)` timeline wait that signals
    /// when this batch completes (allocation-free fast path; `owner` retains
    /// the backing queue/submission alive until the waiting submission retires).
    #[cfg(not(target_os = "macos"))]
    pub(super) fn vulkan_wait(
        &self,
    ) -> Option<(seismic_vulkan::vk::Semaphore, u64, Arc<dyn std::any::Any + Send + Sync>)> {
        match &self.handle {
            TransferSyncHandle::Vulkan(queue) => Some((
                queue.semaphore(),
                self.id.value,
                Arc::clone(queue) as Arc<dyn std::any::Any + Send + Sync>,
            )),
            TransferSyncHandle::VulkanSubmission { semaphore, route } => Some((
                *semaphore,
                self.id.value,
                Arc::clone(route) as Arc<dyn std::any::Any + Send + Sync>,
            )),
            _ => None,
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub(super) fn from_vulkan_queue(
        device_id: u64,
        producer_queue_id: u64,
        value: u64,
        queue: Arc<seismic_vulkan::transfer::TransferQueue>,
    ) -> Self {
        Self {
            id: TransferSyncId { device_id, producer_queue_id, value },
            handle: TransferSyncHandle::Vulkan(queue),
        }
    }

    /// Non-blocking host check whether this fence has been reached.
    pub fn is_reached(&self) -> Result<bool, FenceError> {
        <Self as TimelineObserver>::is_reached(self, self.timeline_point())
    }

    /// Block the host up to `timeout` waiting for this fence to be reached.
    pub fn wait(&self, timeout: Duration) -> Result<bool, FenceError> {
        <Self as TimelineObserver>::wait(self, self.timeline_point(), timeout)
    }
}

impl TimelineObserver for TransferSync {
    fn is_reached(&self, point: TimelinePoint) -> Result<bool, FenceError> {
        let expected = self.timeline_point();
        if point.device != expected.device {
            return Err(FenceError::DeviceMismatch {
                expected: expected.device,
                got: point.device,
            });
        }
        if point.producer != expected.producer {
            return Err(FenceError::UnknownProducer {
                device: point.device,
                producer: point.producer,
            });
        }
        match &self.handle {
            TransferSyncHandle::Cpu(Some(tl)) => tl.is_reached(point),
            TransferSyncHandle::Cpu(None) => Ok(true),
            TransferSyncHandle::Cuda { queue: Some(q), .. } => {
                q.completed(point.value).map_err(|e| FenceError::Faulted(e.to_string()))
            }
            TransferSyncHandle::Cuda { event: Some(ev), queue: None } => {
                ev.query().map_err(|e| FenceError::Faulted(e.to_string()))
            }
            TransferSyncHandle::Cuda { event: None, queue: None } => Ok(false),
            TransferSyncHandle::CudaSubmission(route) => {
                route.completed().map_err(|e| FenceError::Faulted(e.to_string()))
            }
            #[cfg(target_os = "macos")]
            TransferSyncHandle::Metal(event) => Ok(event.value() >= point.value),
            #[cfg(not(target_os = "macos"))]
            TransferSyncHandle::Vulkan(queue) => {
                queue.reached(point.value).map_err(|e| FenceError::Faulted(e.to_string()))
            }
            #[cfg(not(target_os = "macos"))]
            TransferSyncHandle::VulkanSubmission { route, .. } => {
                route.completed().map_err(|e| FenceError::Faulted(e.to_string()))
            }
        }
    }

    fn wait(&self, point: TimelinePoint, timeout: Duration) -> Result<bool, FenceError> {
        let expected = self.timeline_point();
        if point.device != expected.device {
            return Err(FenceError::DeviceMismatch {
                expected: expected.device,
                got: point.device,
            });
        }
        if point.producer != expected.producer {
            return Err(FenceError::UnknownProducer {
                device: point.device,
                producer: point.producer,
            });
        }
        match &self.handle {
            TransferSyncHandle::Cpu(Some(tl)) => tl.wait(point, timeout),
            TransferSyncHandle::Cpu(None) => Ok(true),
            TransferSyncHandle::Cuda { queue: Some(q), .. } => {
                q.wait(point.value, timeout).map_err(|e| FenceError::Faulted(e.to_string()))
            }
            TransferSyncHandle::Cuda { event: Some(ev), queue: None } => {
                ev.synchronize().map_err(|e| FenceError::Faulted(e.to_string()))?;
                Ok(true)
            }
            TransferSyncHandle::Cuda { event: None, queue: None } => Ok(false),
            TransferSyncHandle::CudaSubmission(route) => {
                if route.completed().map_err(|e| FenceError::Faulted(e.to_string()))? {
                    return Ok(true);
                }
                route.ended().map_err(|e| FenceError::Faulted(e.to_string()))?;
                Ok(true)
            }
            #[cfg(target_os = "macos")]
            TransferSyncHandle::Metal(event) => Ok(event.wait(point.value, timeout)),
            #[cfg(not(target_os = "macos"))]
            TransferSyncHandle::Vulkan(queue) => {
                queue.wait(point.value, timeout).map_err(|e| FenceError::Faulted(e.to_string()))
            }
            #[cfg(not(target_os = "macos"))]
            TransferSyncHandle::VulkanSubmission { route, .. } => {
                if route.completed().map_err(|e| FenceError::Faulted(e.to_string()))? {
                    return Ok(true);
                }
                route.ended().map_err(|e| FenceError::Faulted(e.to_string()))?;
                Ok(true)
            }
        }
    }
}

/// Monotonic process-wide ID generator for independent [`TransferInner`] queues
/// so each CUDA/Vulkan/CPU transfer stream tracks its own allocation fence slot
/// (`DeviceQueue::Transfer(queue_id)`) and reverse-order completion on stream B
/// can never overwrite or erase stream A's fence.
static NEXT_TRANSFER_QUEUE_ID: AtomicU64 = AtomicU64::new(1);

/// One device's transfer queue.
pub struct TransferInner {
    device: Arc<DeviceInner>,
    producer_queue_id: u64,
    queue: Arc<Queue>,
    /// Unique producer queue ID for `DeviceQueue::Transfer(queue_id)`.
    queue_id: u64,
    /// The value the next batch takes: batches are numbered here alone,
    /// every backend's queue signalling the value it is given.
    next: AtomicU64,
    /// Queues batches one at a time, so values reach the device in order.
    issuing: Mutex<()>,
}

impl TransferInner {
    /// The transfer queue of `device`.
    pub fn new(device: &Arc<DeviceInner>) -> Result<Self, TransferError> {
        let kind = match &device.kind {
            OpenedKind::Cpu(_) => TransferKind::Cpu { done: AtomicU64::new(0) },
            OpenedKind::Cuda(opened) => TransferKind::Cuda(
                seismic_cuda::transfer::TransferQueue::new(opened.service()).map_err(TransferError::Execution)?,
            ),
            #[cfg(target_os = "macos")]
            OpenedKind::Metal(_) => return Err(TransferError::InPlace),
            #[cfg(not(target_os = "macos"))]
            OpenedKind::Vulkan(opened) => TransferKind::Vulkan(Arc::new(
                seismic_vulkan::transfer::TransferQueue::new(opened.service()).map_err(TransferError::Execution)?,
            )),
        };
        Ok(Self {
            device: device.clone(),
            producer_queue_id: next_queue_id(),
            queue: Arc::new(Queue { kind }),
            queue_id: NEXT_TRANSFER_QUEUE_ID.fetch_add(1, Ordering::Relaxed),
            next: AtomicU64::new(1),
            issuing: Mutex::new(()),
        })
    }

    /// Queue `items` as one batch: the value it completes at
    /// ([`Self::completed`]). An invalid batch is refused before anything is
    /// queued; once queued, a failure is the queue's (`completed` reports
    /// it). The caller queues it only once the device work that read the
    /// bytes it overwrites has ended (the module documentation), so no batch
    /// waits on device work. The
    /// batch is a device use of every run's source (read) and destination
    /// (written) and every signal's word (written): host access to them,
    /// and their storage's release, wait for it.
    pub fn copy(&self, items: &[TransferItem<'_>]) -> Result<u64, TransferError> {
        self.copy_inner(items, &[], false).map(|(value, _)| value)
    }

    /// [`Self::copy`], also returning a [`TransferSync`] event handle so a
    /// [`super::graph::NativeGraphSequence`] can wait on the batch via
    /// cross-stream event synchronization (`cuStreamWaitEvent` on CUDA,
    /// timeline semaphore wait on Vulkan).
    pub fn copy_synced(&self, items: &[TransferItem<'_>]) -> Result<(u64, TransferSync), TransferError> {
        self.copy_inner(items, &[], true)
    }

    /// [`Self::copy_synced`], first ordering this transfer batch after every
    /// [`TransferSync`] fence in `wait_fences` (`cuStreamWaitEvent` on CUDA,
    /// `vkQueueSubmit2` timeline semaphore wait on Vulkan, host wait on CPU)
    /// and returning this batch's signal [`TransferSync`]. Rejects foreign
    /// devices or `HostOnly` fences on GPU backends as
    /// [`TransferError::RefusedBeforeIssue`].
    pub fn copy_after(
        &self,
        items: &[TransferItem<'_>],
        wait_fences: &[&TransferSync],
    ) -> Result<(u64, TransferSync), TransferError> {
        self.copy_inner(items, wait_fences, true)
    }

    /// Raise `gate` to `value` once all previously queued copies on this
    /// transfer queue have landed. On CUDA this emits `cuStreamWriteValue32`
    /// directly on the copy stream (zero host wait on DMA completion); on
    /// other backends it waits for the latest queued batch and raises `gate`
    /// from the host.
    pub fn signal_gate(
        &self,
        gate: &super::timeline::TimelineHost<super::timeline::Gate>,
        value: u64,
    ) -> Result<(), TransferError> {
        if gate.inner().device().device_id() != self.device.device_id() {
            return Err(TransferError::RefusedBeforeIssue(ExecutionError::SubmissionFailed(
                format!(
                    "timeline gate belongs to device {}, not transfer device {}",
                    gate.inner().device().device_id(),
                    self.device.device_id(),
                ),
            )));
        }
        let _issuing = self.issuing.lock().expect("transfer queue lock is never poisoned");
        match (&self.queue.kind, gate.inner().kind()) {
            (TransferKind::Cuda(queue), super::timeline::TimelineKind::Cuda(sides)) => queue
                .write_word(sides.gate.device_address(), value as u32)
                .map_err(TransferError::Execution),
            _ => {
                let latest = self.next.load(Ordering::Relaxed).saturating_sub(1);
                if latest > 0 {
                    while !self
                        .queue
                        .wait(latest, Duration::from_secs(1))
                        .map_err(TransferError::Execution)?
                    {
                        std::thread::yield_now();
                    }
                }
                gate.signal(value).map_err(TransferError::Execution)
            }
        }
    }

    fn copy_inner(
        &self,
        items: &[TransferItem<'_>],
        wait_fences: &[&TransferSync],
        record_event: bool,
    ) -> Result<(u64, TransferSync), TransferError> {
        seismic_memory::assert_blocking_allocation_allowed("TransferInner::copy");
        for fence in wait_fences {
            fence
                .validate_for_device(&self.device)
                .map_err(TransferError::RefusedBeforeIssue)?;
        }
        for item in items {
            match item {
                TransferItem::Run(run) => self.check(run)?,
                TransferItem::Signal { word, offset, .. } => self.check_signal(word, *offset)?,
            }
        }
        // Any view of a slab moved by `Slots::adopt` must wait for its
        // compute-queue `Copy(from -> to)` before the unordered transfer
        // stream reads or overwrites it. Done here (outside `residency` and
        // `NoBlockingAllocationGuard`, and before `self.issuing` / device
        // permits) so `Slots::region_view` stays strictly non-blocking.
        for item in items {
            match item {
                TransferItem::Run(run) => {
                    run.source.wait_adopted().map_err(TransferError::RefusedBeforeIssue)?;
                    run.destination.wait_adopted().map_err(TransferError::RefusedBeforeIssue)?;
                }
                TransferItem::Signal { word, .. } => {
                    word.wait_adopted().map_err(TransferError::RefusedBeforeIssue)?;
                }
            }
        }
        // Host access is excluded while the batch is queued, in access-group
        // order as native submissions take theirs.
        let access = crate::native::merge_access(items.iter().flat_map(|item| match item {
            TransferItem::Run(run) => {
                vec![(run.source.allocation().clone(), false), (run.destination.allocation().clone(), true)]
            }
            TransferItem::Signal { word, .. } => vec![(word.allocation().clone(), true)],
        }));
        let permits = access
            .iter()
            .map(|(allocation, write)| allocation.acquire_for_device(*write))
            .collect::<Vec<_>>();
        let _issuing = self.issuing.lock().expect("transfer queue lock is never poisoned");
        // A batch takes its value once queued: a refused one takes none.
        let value = self.next.load(Ordering::Relaxed);
        let handle = match &self.queue.kind {
            TransferKind::Cpu { done } => {
                // Every item's addresses first: a batch is refused before any
                // of it is copied.
                enum Step {
                    Copy(*const u8, *mut u8, usize),
                    Signal(*mut u8, i32),
                }
                let read_at = |tensor: &TensorInner, offset: u64| {
                    let base = typed_buffer::<seismic_cpu::Cpu, seismic_cpu::Executor>(tensor.allocation())
                        .data_pointer()
                        .cast_const();
                    // SAFETY: within the tensor's view, checked above.
                    unsafe { base.add((tensor.byte_offset() + offset) as usize) }
                };
                let write_at = |tensor: &TensorInner, offset: u64| {
                    let base = tensor
                        .allocation()
                        .storage()
                        .host_address()
                        .ok_or_else(|| TransferError::Run("a CPU tensor without host memory".into()))?;
                    // SAFETY: within the tensor's view, checked above.
                    Ok::<_, TransferError>(unsafe { base.as_ptr().add((tensor.byte_offset() + offset) as usize) })
                };
                let steps = items
                    .iter()
                    .map(|item| match item {
                        TransferItem::Run(run) => Ok(Step::Copy(
                            read_at(run.source, run.source_offset),
                            write_at(run.destination, run.destination_offset)?,
                            run.bytes as usize,
                        )),
                        TransferItem::Signal { word, offset, value } => Ok(Step::Signal(write_at(word, *offset)?, *value)),
                    })
                    .collect::<Result<Vec<_>, TransferError>>()?;
                for fence in wait_fences {
                    fence
                        .wait(Duration::from_secs(60))
                        .map_err(|e| TransferError::RefusedBeforeIssue(ExecutionError::SubmissionFailed(e.to_string())))?;
                }
                for step in steps {
                    match step {
                        // SAFETY: both ranges lie in live views (checked);
                        // the caller keeps any other access off them
                        // meanwhile.
                        Step::Copy(source, destination, bytes) => unsafe { std::ptr::copy(source, destination, bytes) },
                        // SAFETY: an aligned word of a live view (checked),
                        // which device work reads atomically.
                        Step::Signal(word, value) => unsafe {
                            std::sync::atomic::AtomicI32::from_ptr(word.cast()).store(value, Ordering::Release)
                        },
                    }
                }
                done.store(value, Ordering::Release);
                let tl = seismic_io::HostTimeline::new(
                    DeviceId(self.device.device_id()),
                    ProducerId(self.producer_queue_id),
                );
                tl.advance(value);
                TransferSyncHandle::Cpu(Some(tl))
            }
            TransferKind::Cuda(queue) => {
                type Cuda = seismic_cuda::Cuda;
                type Executor = seismic_cuda::Executor;
                let wait_events: Vec<&seismic_cuda::Event> = wait_fences
                    .iter()
                    .map(|fence| {
                        fence.cuda_event().ok_or_else(|| {
                            TransferError::RefusedBeforeIssue(ExecutionError::SubmissionFailed(
                                format!("wait fence {:?} carries no CUDA event", fence.id()),
                            ))
                        })
                    })
                    .collect::<Result<_, _>>()?;
                let items = items
                    .iter()
                    .map(|item| match item {
                        TransferItem::Run(run) => seismic_cuda::transfer::TransferItem::Run(seismic_cuda::transfer::TransferRun {
                            source: typed_buffer::<Cuda, Executor>(run.source.allocation()),
                            source_offset: run.source.byte_offset() + run.source_offset,
                            destination: typed_buffer::<Cuda, Executor>(run.destination.allocation()),
                            destination_offset: run.destination.byte_offset() + run.destination_offset,
                            bytes: run.bytes,
                        }),
                        TransferItem::Signal { word, offset, value } => seismic_cuda::transfer::TransferItem::Signal {
                            word: typed_buffer::<Cuda, Executor>(word.allocation()),
                            offset: word.byte_offset() + offset,
                            value: *value as u32,
                        },
                    })
                    .collect::<Vec<_>>();
                // The word holds a value's low 32 bits, compared cyclically.
                let result = queue.copy_after_events(&items, value as u32, &wait_events, record_event);
                let cuda_event = match self.handle_cuda_copy_outcome(result, value, &access) {
                    Ok(event) => event,
                    Err(error) => {
                        drop(permits);
                        return Err(error);
                    }
                };
                TransferSyncHandle::Cuda {
                    event: cuda_event,
                    queue: Some(Arc::clone(&self.queue)),
                }
            }
            #[cfg(not(target_os = "macos"))]
            TransferKind::Vulkan(queue) => {
                use crate::backends::vulkan_transfer_end;
                let mut wait_sems: Vec<(seismic_vulkan::vk::Semaphore, u64)> =
                    Vec::with_capacity(wait_fences.len());
                let mut wait_owners: Vec<Arc<dyn std::any::Any + Send + Sync>> =
                    Vec::with_capacity(wait_fences.len());
                for fence in wait_fences {
                    let Some((sem, val, owner)) = fence.vulkan_wait() else {
                        return Err(TransferError::RefusedBeforeIssue(ExecutionError::SubmissionFailed(
                            format!("wait fence {:?} carries no Vulkan timeline semaphore", fence.id()),
                        )));
                    };
                    wait_sems.push((sem, val));
                    wait_owners.push(owner);
                }
                let items = items
                    .iter()
                    .map(|item| match item {
                        TransferItem::Run(run) => seismic_vulkan::transfer::TransferItem::Run(seismic_vulkan::transfer::TransferRun {
                            source: vulkan_transfer_end(run.source.allocation(), run.source.byte_offset() + run.source_offset),
                            destination: vulkan_transfer_end(
                                run.destination.allocation(),
                                run.destination.byte_offset() + run.destination_offset,
                            ),
                            bytes: run.bytes,
                        }),
                        TransferItem::Signal { word, offset, value } => seismic_vulkan::transfer::TransferItem::Signal {
                            word: vulkan_transfer_end(word.allocation(), word.byte_offset() + offset),
                            value: *value as u32,
                        },
                    })
                    .collect::<Vec<_>>();
                queue.copy_after(&items, value, &wait_sems, wait_owners).map_err(TransferError::Execution)?;
                TransferSyncHandle::Vulkan(Arc::clone(queue))
            }
        };
        self.next.store(value + 1, Ordering::Relaxed);
        let fence: Arc<dyn DeviceCompletion> = Arc::new(BatchFence { queue: self.queue.clone(), value });
        for (allocation, write) in &access {
            allocation.record_device_use(DeviceQueue::Transfer(self.queue_id), fence.clone(), *write);
        }
        drop(permits);
        let sync = TransferSync {
            id: TransferSyncId {
                device_id: self.device.device_id(),
                producer_queue_id: self.producer_queue_id,
                value,
            },
            handle,
        };
        Ok((value, sync))
    }

    /// Translate a CUDA transfer enqueue outcome:
    /// - `RefusedBeforeIssue`: nothing was queued on the transfer stream, so
    ///   neither consume `value` nor install a [`BatchFence`] (otherwise
    ///   allocation retirement would wait forever on an unsignaled value on a
    ///   healthy idle queue).
    /// - `PossiblyIssued`: earlier copies of the batch may already be in flight
    ///   on the now-faulted stream, so advance `next` and record a
    ///   [`BatchFence`] on every accessed allocation so host access and storage
    ///   release wait for the faulted stream to drain or context loss.
    fn handle_cuda_copy_outcome(
        &self,
        result: Result<Option<Arc<seismic_cuda::Event>>, seismic_cuda::transfer::CopyError>,
        value: u64,
        access: &[(Arc<crate::driver::Allocation>, bool)],
    ) -> Result<Option<Arc<seismic_cuda::Event>>, TransferError> {
        match result {
            Ok(event) => Ok(event),
            Err(seismic_cuda::transfer::CopyError::RefusedBeforeIssue(error)) => {
                Err(TransferError::RefusedBeforeIssue(error))
            }
            Err(seismic_cuda::transfer::CopyError::PossiblyIssued(error)) => {
                self.next.store(value + 1, Ordering::Relaxed);
                let fence: Arc<dyn DeviceCompletion> = Arc::new(BatchFence { queue: self.queue.clone(), value });
                for (allocation, write) in access {
                    allocation.record_device_use(DeviceQueue::Transfer(self.queue_id), fence.clone(), *write);
                }
                Err(TransferError::Execution(error))
            }
        }
    }

    /// A run copies between live views of this device's tensors, each
    /// within its view's bytes, never a slab view directly and never into
    /// read-only storage.
    fn check(&self, run: &TransferRun<'_>) -> Result<(), TransferError> {
        if run.source.is_slab_view() || run.destination.is_slab_view() {
            return Err(TransferError::Run("a run cannot transfer a slab view directly".into()));
        }
        if run.destination.allocation().storage().read_only() {
            return Err(TransferError::Run("a run cannot write to a read-only tensor".into()));
        }
        let within =
            |tensor: &TensorInner, offset: u64| offset.checked_add(run.bytes).is_some_and(|end| end <= tensor.byte_len());
        if !within(run.source, run.source_offset) || !within(run.destination, run.destination_offset) {
            return Err(TransferError::Run("a run lies outside its tensors".into()));
        }
        if !Arc::ptr_eq(run.source.device(), &self.device) || !Arc::ptr_eq(run.destination.device(), &self.device) {
            return Err(TransferError::Run("a run's tensors are another device's".into()));
        }
        Ok(())
    }

    /// A signal's word is an aligned i32 of a live writable view of this
    /// device's, never a slab view.
    fn check_signal(&self, word: &TensorInner, offset: u64) -> Result<(), TransferError> {
        if word.is_slab_view() {
            return Err(TransferError::Run("a signal's word cannot lie in a slab view".into()));
        }
        if word.allocation().storage().read_only() {
            return Err(TransferError::Run("a signal's word cannot lie in a read-only tensor".into()));
        }
        if offset.checked_add(4).is_none_or(|end| end > word.byte_len()) {
            return Err(TransferError::Run("a signal's word lies outside its tensor".into()));
        }
        if (word.byte_offset() + offset) % 4 != 0 {
            return Err(TransferError::Run("a signal's word is not 4-byte aligned".into()));
        }
        if !Arc::ptr_eq(word.device(), &self.device) {
            return Err(TransferError::Run("a signal's word is another device's".into()));
        }
        Ok(())
    }

    /// Whether the batch that took `value` is complete, its bytes visible
    /// to device work begun after this returns true; why no batch can
    /// complete (a copy faulted, or the device is lost).
    pub fn completed(&self, value: u64) -> Result<bool, TransferError> {
        self.queue.completed(value).map_err(TransferError::Execution)
    }

    /// Wait at most `timeout` for the batch that took `value`; whether it
    /// completed, or why none can. The thread sleeps until the batch
    /// completes, whatever other queues of the device still run.
    pub fn wait(&self, value: u64, timeout: Duration) -> Result<bool, TransferError> {
        seismic_memory::assert_blocking_allocation_allowed("TransferInner::wait");
        self.queue.wait(value, timeout).map_err(TransferError::Execution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Arc<DeviceInner> {
        let devices = crate::devices::Devices::discover(&[]).unwrap();
        devices.open_backend(seismic_lang::registry::BackendName::Cpu).unwrap()
    }

    #[test]
    fn a_cpu_batch_copies_its_runs_and_completes_on_return() {
        let device = cpu();
        let transfer = TransferInner::new(&device).unwrap();
        let tensor = |bytes: &[u8]| {
            let representation = seismic_lang::registry::dense(seismic_lang::types::DType::I32);
            crate::api::tensor::TensorInner::from_host(&crate::memory::Charges::tally(&device), representation, &[bytes.len() as u64 / 4], bytes).unwrap()
        };
        let source = tensor(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let destination = tensor(&[0; 8]);
        let words = tensor(&[0; 8]);
        let value = transfer
            .copy(
                &[
                    TransferItem::Run(TransferRun { source: &source, source_offset: 0, destination: &destination, destination_offset: 4, bytes: 2 }),
                    TransferItem::Signal { word: &words, offset: 4, value: 5 },
                    TransferItem::Run(TransferRun { source: &source, source_offset: 6, destination: &destination, destination_offset: 0, bytes: 2 }),
                ],
            )
            .unwrap();
        assert!(transfer.completed(value).unwrap());
        assert!(!transfer.completed(value + 1).unwrap());
        // The batch is one fence, recorded on its source and destination.
        assert_eq!(Arc::strong_count(&transfer.queue), 2);
        assert_eq!(destination.read_to_host().unwrap(), [7, 8, 0, 0, 1, 2, 0, 0]);
        assert_eq!(words.read_to_host().unwrap(), [0, 0, 0, 0, 5, 0, 0, 0], "the signal's word holds its value");
        // A run past its tensor, or a signal past or within a word, is
        // refused, queuing nothing: a batch with a valid run before it
        // writes nothing either.
        let past = TransferRun { source: &source, source_offset: 7, destination: &destination, destination_offset: 0, bytes: 2 };
        assert!(matches!(transfer.copy(&[TransferItem::Run(past)]), Err(TransferError::Run(_))));
        for offset in [6, 8] {
            let first = TransferRun { source: &source, source_offset: 0, destination: &destination, destination_offset: 0, bytes: 2 };
            let signal = TransferItem::Signal { word: &words, offset, value: 1 };
            assert!(matches!(transfer.copy(&[TransferItem::Run(first), signal]), Err(TransferError::Run(_))));
        }
        assert_eq!(destination.read_to_host().unwrap(), [7, 8, 0, 0, 1, 2, 0, 0], "a refused batch wrote nothing");
        assert!(!transfer.completed(value + 1).unwrap(), "a refused batch takes no value");
    }

    #[test]
    fn transfer_rejects_slab_views_and_read_only_destinations() {
        let device = cpu();
        let transfer = TransferInner::new(&device).unwrap();
        let charges = crate::memory::Charges::tally(&device);
        let i32s = seismic_lang::registry::dense(seismic_lang::types::DType::I32);
        let writable = TensorInner::from_host(&charges, i32s, &[4], &[1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap();

        // Build a read-only mapped host region tensor.
        let backing = Arc::new([9u8, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0]);
        // SAFETY: `backing` is an immutable Arc<[u8; 16]> owned by the region for its full lifetime.
        let region = unsafe {
            crate::api::HostRegion::new(
                std::ptr::NonNull::new(backing.as_ptr() as *mut u8).unwrap(),
                backing.len(),
                backing.clone(),
            )
        };
        let mapped = crate::api::tensor::MappedRegionInner::new(&device, region).unwrap();
        let read_only = mapped.tensor(i32s, &[4], 0).unwrap();

        // Copying FROM a read-only tensor into a writable tensor succeeds.
        let value = transfer
            .copy(&[TransferItem::Run(TransferRun {
                source: &read_only,
                source_offset: 0,
                destination: &writable,
                destination_offset: 0,
                bytes: 4,
            })])
            .unwrap();
        assert!(transfer.completed(value).unwrap());
        assert_eq!(&writable.read_to_host().unwrap()[..4], &[9, 10, 11, 12]);

        // Writing TO a read-only tensor (copy destination or signal word) is refused.
        assert!(matches!(
            transfer.copy(&[TransferItem::Run(TransferRun {
                source: &writable,
                source_offset: 0,
                destination: &read_only,
                destination_offset: 0,
                bytes: 4,
            })]),
            Err(TransferError::Run(_))
        ));
        assert!(matches!(
            transfer.copy(&[TransferItem::Signal { word: &read_only, offset: 0, value: 1 }]),
            Err(TransferError::Run(_))
        ));

        // Build a logical slab view and verify source, destination, and signal are all refused.
        let pool_charges = crate::pool::PoolCharges::one(&charges);
        let pool = crate::pool::Pool::open(&pool_charges, seismic_topology::Contract::DeviceOnly, 1 << 16, 1 << 18).unwrap();
        let geometry = crate::pool::SlabGeometry {
            rows_per_slab: 4,
            logical_rows: 16,
            capacity: 4,
            slab_bytes: crate::pool::BLOCK_ALIGNMENT,
            slab_alignment: crate::pool::BLOCK_ALIGNMENT,
            regions: vec![(i32s, vec![2], 0)],
        };
        let store = crate::pool::SlabStore::in_pool(&charges, &pool, geometry, &[]).unwrap();
        store.add_slab(crate::pool::SlabFill::Zeroed).unwrap();
        let slab_view = store.logical_region(0).unwrap();
        assert!(slab_view.is_slab_view());

        assert!(matches!(
            transfer.copy(&[TransferItem::Run(TransferRun {
                source: &slab_view,
                source_offset: 0,
                destination: &writable,
                destination_offset: 0,
                bytes: 4,
            })]),
            Err(TransferError::Run(_))
        ));
        assert!(matches!(
            transfer.copy(&[TransferItem::Run(TransferRun {
                source: &writable,
                source_offset: 0,
                destination: &slab_view,
                destination_offset: 0,
                bytes: 4,
            })]),
            Err(TransferError::Run(_))
        ));
        assert!(matches!(
            transfer.copy(&[TransferItem::Signal { word: &slab_view, offset: 0, value: 1 }]),
            Err(TransferError::Run(_))
        ));
    }

    #[test]
    fn cuda_refused_before_issue_installs_no_unsignaled_fence() {
        let device = cpu();
        let transfer = TransferInner::new(&device).unwrap();
        let charges = crate::memory::Charges::tally(&device);
        let i32s = seismic_lang::registry::dense(seismic_lang::types::DType::I32);
        let source = TensorInner::from_host(&charges, i32s, &[2], &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let destination = TensorInner::from_host(&charges, i32s, &[2], &[0; 8]).unwrap();
        let access = vec![(source.allocation().clone(), false), (destination.allocation().clone(), true)];

        // Pre-issue refusal (invalid item, event-alloc failure, or already-faulted queue):
        // must NOT advance `next` and must NOT record a `BatchFence` on any allocation.
        let value = transfer.next.load(Ordering::Relaxed);
        let refused = transfer.handle_cuda_copy_outcome(
            Err(seismic_cuda::transfer::CopyError::RefusedBeforeIssue(
                ExecutionError::SubmissionFailed("event alloc failed".into()),
            )),
            value,
            &access,
        );
        assert!(matches!(refused, Err(TransferError::RefusedBeforeIssue(_))));
        assert!(matches!(refused, Err(ref e) if e.refused_before_issue()));
        assert_eq!(transfer.next.load(Ordering::Relaxed), value, "RefusedBeforeIssue must not consume a batch value");
        assert_eq!(
            Arc::strong_count(&transfer.queue),
            1,
            "RefusedBeforeIssue must not install a BatchFence on allocations"
        );
        // Host read must return immediately without hanging on an unsignaled fence.
        assert_eq!(destination.read_to_host().unwrap(), [0; 8]);

        // Mid-batch failure (`PossiblyIssued`): must advance `next` and install a
        // `BatchFence` on every accessed allocation so release waits for stream drain.
        let issued = transfer.handle_cuda_copy_outcome(
            Err(seismic_cuda::transfer::CopyError::PossiblyIssued(
                ExecutionError::SubmissionFailed("mid-batch enqueue fault".into()),
            )),
            value,
            &access,
        );
        assert!(matches!(issued, Err(TransferError::Execution(_))));
        assert!(matches!(issued, Err(ref e) if !e.refused_before_issue()));
        assert_eq!(transfer.next.load(Ordering::Relaxed), value + 1, "PossiblyIssued must advance batch value");
        assert_eq!(
            Arc::strong_count(&transfer.queue),
            2,
            "PossiblyIssued must install a BatchFence on accessed allocations"
        );
        // Signal the synthetic CPU queue value before scope exit so tensor drop
        // does not wait indefinitely on the fence installed by `PossiblyIssued`.
        if let TransferKind::Cpu { done } = &transfer.queue.kind {
            done.store(value, Ordering::Release);
        }
    }

    /// Proof case #3: `TransferSync` identity is scoped by
    /// `(device_id, producer_queue_id, value)`. Two batches with the same
    /// numeric `u64` timeline value on different queues or different opened
    /// devices are NEVER treated as equal (`same_batch == false`), even on
    /// backends without CUDA events.
    #[test]
    fn transfer_sync_same_numeric_value_across_queues_and_devices_is_not_equal() {
        let device_a = cpu();
        let device_b = cpu();
        assert_ne!(device_a.device_id(), device_b.device_id());

        let queue_a1 = TransferInner::new(&device_a).unwrap();
        let queue_a2 = TransferInner::new(&device_a).unwrap();
        let queue_b1 = TransferInner::new(&device_b).unwrap();

        let make_pair = |device: &Arc<DeviceInner>| {
            let charges = crate::memory::Charges::tally(device);
            let i32s = seismic_lang::registry::dense(seismic_lang::types::DType::I32);
            let src = TensorInner::from_host(&charges, i32s, &[2], &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
            let dst = TensorInner::from_host(&charges, i32s, &[2], &[0; 8]).unwrap();
            (src, dst)
        };
        let (src_a, dst_a) = make_pair(&device_a);
        let (src_b, dst_b) = make_pair(&device_b);
        fn run<'a>(src: &'a TensorInner, dst: &'a TensorInner) -> [TransferItem<'a>; 1] {
            [TransferItem::Run(TransferRun {
                source: src,
                source_offset: 0,
                destination: dst,
                destination_offset: 0,
                bytes: 4,
            })]
        }

        let (v_a1_first, sync_a1_first) = queue_a1.copy_synced(&run(&src_a, &dst_a)).unwrap();
        let (v_a1_second, sync_a1_second) = queue_a1.copy_synced(&run(&src_a, &dst_a)).unwrap();
        let (v_a2_first, sync_a2_first) = queue_a2.copy_synced(&run(&src_a, &dst_a)).unwrap();
        let (v_b1_first, sync_b1_first) = queue_b1.copy_synced(&run(&src_b, &dst_b)).unwrap();

        assert_eq!(v_a1_first, 1);
        assert_eq!(v_a2_first, 1);
        assert_eq!(v_b1_first, 1);
        assert_eq!(v_a1_second, 2);

        // Same batch clone -> equal.
        assert!(sync_a1_first.same_batch(&sync_a1_first.clone()));
        // Same queue, different generation -> NOT equal.
        assert!(!sync_a1_first.same_batch(&sync_a1_second));
        // Different queue on same device, SAME numeric u64 value (1) -> NOT equal.
        assert!(!sync_a1_first.same_batch(&sync_a2_first));
        assert_ne!(sync_a1_first.id().producer_queue_id, sync_a2_first.id().producer_queue_id);
        // Different device, SAME numeric u64 value (1) -> NOT equal.
        assert!(!sync_a1_first.same_batch(&sync_b1_first));
        assert_ne!(sync_a1_first.id().device_id, sync_b1_first.id().device_id);
    }

    /// Proof cases #1 & #2 at the cross-queue fence layer:
    /// 1. Delayed fill before consumer read: while a `Fill` write is still in
    ///    flight, a `Compute` read on that slot receives the fill fence via
    ///    `queue_order` so it waits for producer publication.
    /// 2. Overlapping readers prohibiting slot reuse until the final reader
    ///    retires: while two `Compute` readers overlap on the slot, a `Fill`
    ///    write (slot reuse) is blocked after reader 1 finishes and only
    ///    unblocked once the final reader (reader 2) retires.
    #[test]
    fn delayed_fill_orders_before_consumer_and_overlapping_readers_block_slot_reuse() {
        use crate::native::NativeQueue;
        use std::sync::atomic::AtomicBool;

        struct ControlledFence(Arc<AtomicBool>);
        impl DeviceCompletion for ControlledFence {
            fn completed(&self) -> Result<bool, ExecutionError> {
                Ok(self.0.load(Ordering::Acquire))
            }
            fn ended(&self) -> Result<(), ExecutionError> {
                assert!(self.0.load(Ordering::Acquire), "unexpected blocking wait in unit test");
                Ok(())
            }
        }

        let device = cpu();
        let charges = crate::memory::Charges::tally(&device);
        let i32s = seismic_lang::registry::dense(seismic_lang::types::DType::I32);
        let slot = TensorInner::from_host(&charges, i32s, &[4], &[0; 16]).unwrap();
        let alloc = slot.allocation();

        // Proof #1: delayed fill write on NativeQueue::Fill blocks Compute read.
        let fill_done = Arc::new(AtomicBool::new(false));
        let fill_fence: Arc<dyn DeviceCompletion> = Arc::new(ControlledFence(Arc::clone(&fill_done)));
        alloc.record_device_use(DeviceQueue::Native(NativeQueue::Fill), Arc::clone(&fill_fence), true);

        let compute_waits = alloc.queue_order(NativeQueue::Compute, false).unwrap();
        assert_eq!(compute_waits.len(), 1, "consumer read must wait on delayed fill producer");
        assert!(Arc::ptr_eq(&compute_waits[0], &fill_fence));

        // Once the delayed fill completes, consumer read has no pending fill fence.
        fill_done.store(true, Ordering::Release);
        assert!(alloc.queue_order(NativeQueue::Compute, false).unwrap().is_empty());

        // Proof #2: two overlapping Compute readers prohibit Fill slot reuse
        // until the FINAL reader retires.
        let reader1_done = Arc::new(AtomicBool::new(false));
        let reader2_done = Arc::new(AtomicBool::new(false));
        let reader1: Arc<dyn DeviceCompletion> = Arc::new(ControlledFence(Arc::clone(&reader1_done)));
        let reader2: Arc<dyn DeviceCompletion> = Arc::new(ControlledFence(Arc::clone(&reader2_done)));
        alloc.record_device_use(DeviceQueue::Native(NativeQueue::Compute), Arc::clone(&reader1), false);
        alloc.record_device_use(DeviceQueue::Native(NativeQueue::Compute), Arc::clone(&reader2), false);

        // Both readers in flight -> Fill write must wait.
        assert!(!alloc.queue_order(NativeQueue::Fill, true).unwrap().is_empty());

        // Reader 1 completes, reader 2 still active -> Fill slot reuse STILL blocked!
        reader1_done.store(true, Ordering::Release);
        let remaining = alloc.queue_order(NativeQueue::Fill, true).unwrap();
        assert_eq!(remaining.len(), 1, "slot reuse must remain blocked until final reader retires");
        assert!(Arc::ptr_eq(&remaining[0], &reader2));

        // Final reader (reader 2) retires -> slot reuse now unblocked!
        reader2_done.store(true, Ordering::Release);
        assert!(alloc.queue_order(NativeQueue::Fill, true).unwrap().is_empty());
    }

    /// `NativeGraphSequence::follow_transfer` and `TransferSync::validate_for_device`
    /// must reject foreign-device and unsupported `TransferSync` handles loudly
    /// rather than silently accepting or ignoring them.
    #[test]
    fn follow_transfer_rejects_foreign_device_and_unsupported_handle() {
        use crate::native::graph::NativeGraphSequence;
        use crate::native::NativeQueue;

        let device_a = cpu();
        let device_b = cpu();
        assert_ne!(device_a.device_id(), device_b.device_id());

        let queue_a = TransferInner::new(&device_a).unwrap();
        let queue_b = TransferInner::new(&device_b).unwrap();

        let make_pair = |device: &Arc<DeviceInner>| {
            let charges = crate::memory::Charges::tally(device);
            let i32s = seismic_lang::registry::dense(seismic_lang::types::DType::I32);
            let src = TensorInner::from_host(&charges, i32s, &[2], &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
            let dst = TensorInner::from_host(&charges, i32s, &[2], &[0; 8]).unwrap();
            (src, dst)
        };
        let (src_a, dst_a) = make_pair(&device_a);
        let (src_b, dst_b) = make_pair(&device_b);
        fn run<'a>(src: &'a TensorInner, dst: &'a TensorInner) -> [TransferItem<'a>; 1] {
            [TransferItem::Run(TransferRun {
                source: src,
                source_offset: 0,
                destination: dst,
                destination_offset: 0,
                bytes: 4,
            })]
        }
        let (_, sync_a) = queue_a.copy_synced(&run(&src_a, &dst_a)).unwrap();
        let (_, sync_b) = queue_b.copy_synced(&run(&src_b, &dst_b)).unwrap();

        // Same-device, matching handle -> accepted.
        let mut seq_a = NativeGraphSequence::new(&device_a, NativeQueue::Compute);
        assert!(seq_a.follow_transfer(sync_a).is_ok());

        // Foreign-device handle -> rejected loudly.
        let err = seq_a
            .follow_transfer(sync_b)
            .expect_err("foreign-device TransferSync must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("belongs to device"),
            "unexpected foreign-device error: {msg}"
        );

        // Unsupported handle (e.g., CUDA handle without a recorded event on a
        // CPU device, even with a matching device_id) -> rejected loudly.
        let unsupported = TransferSync {
            id: TransferSyncId {
                device_id: device_a.device_id(),
                producer_queue_id: 99,
                value: 1,
            },
            handle: TransferSyncHandle::Cuda {
                event: None,
                queue: None,
            },
        };
        let err = seq_a
            .follow_transfer(unsupported)
            .expect_err("unsupported TransferSync handle must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("is not supported on backend"),
            "unexpected unsupported-handle error: {msg}"
        );
    }

    /// Stage 1: `TransferSync` bridges directly to `seismic_io::TimelinePoint`
    /// (`HostOnly` on CPU, `DeviceAndHost` on Metal/GPU), implements
    /// `TimelineObserver` (usable directly by `IoPlane::reserve_bus_until_fence`),
    /// and `TransferInner::copy_after` orders a transfer after upstream fences
    /// while rejecting foreign-device handles as `RefusedBeforeIssue`.
    #[test]
    fn timeline_point_bridge_copy_after_and_io_plane_integration() {
        use crate::devices::Devices;
        use crate::memory::Charges;
        use crate::native::graph::NativeGraphSequence;
        use crate::native::NativeQueue;
        use seismic_lang::registry::BackendName;

        let device_a = cpu();
        let device_b = cpu();

        let alloc = |dev: &Arc<DeviceInner>| {
            let charges = Charges::tally(dev);
            let i32s = seismic_lang::registry::dense(seismic_lang::types::DType::I32);
            TensorInner::from_host(&charges, i32s, &[1], &[1, 2, 3, 4]).unwrap()
        };
        let src_a = alloc(&device_a);
        let mid_a = alloc(&device_a);
        let dst_a = alloc(&device_a);
        let src_b = alloc(&device_b);
        let dst_b = alloc(&device_b);

        let queue_a = TransferInner::new(&device_a).unwrap();
        let queue_b = TransferInner::new(&device_b).unwrap();

        let (_, sync1) = queue_a
            .copy_synced(&[TransferItem::Run(TransferRun {
                source: &src_a,
                source_offset: 0,
                destination: &mid_a,
                destination_offset: 0,
                bytes: 4,
            })])
            .unwrap();

        // CPU TransferSync exposes `FenceCapability::HostOnly` and is reached.
        let pt1 = sync1.timeline_point();
        assert_eq!(pt1.device, DeviceId(device_a.device_id()));
        assert_eq!(pt1.capability, FenceCapability::HostOnly);
        assert!(sync1.is_reached().unwrap());
        assert!(sync1.wait(Duration::from_millis(10)).unwrap());

        // `copy_after` chains a second transfer after `sync1`.
        let (_, sync2) = queue_a
            .copy_after(
                &[TransferItem::Run(TransferRun {
                    source: &mid_a,
                    source_offset: 0,
                    destination: &dst_a,
                    destination_offset: 0,
                    bytes: 4,
                })],
                &[&sync1],
            )
            .unwrap();
        assert_eq!(sync2.timeline_point().value, pt1.value + 1);
        assert!(sync2.is_reached().unwrap());

        // Foreign-device wait fence is rejected as `RefusedBeforeIssue`.
        let (_, sync_b) = queue_b
            .copy_synced(&[TransferItem::Run(TransferRun {
                source: &src_b,
                source_offset: 0,
                destination: &dst_b,
                destination_offset: 0,
                bytes: 4,
            })])
            .unwrap();
        let err = queue_a
            .copy_after(
                &[TransferItem::Run(TransferRun {
                    source: &mid_a,
                    source_offset: 0,
                    destination: &dst_a,
                    destination_offset: 0,
                    bytes: 4,
                })],
                &[&sync_b],
            )
            .err()
            .expect("foreign-device wait_fence must fail");
        assert!(err.refused_before_issue(), "foreign wait fence must be RefusedBeforeIssue: {err}");

        // `NativeGraphSequence::wait_fence` accepts same-device fence.
        let mut seq = NativeGraphSequence::new(&device_a, NativeQueue::Compute);
        assert!(seq.wait_fence(sync2.clone()).is_ok());

        // `TransferSync` plugs directly into `seismic_io::IoPlane::reserve_bus_until_fence`.
        let ddr = seismic_io::SharedBusId::HostDdr(seismic_io::NumaNodeId(0));
        let fabric = seismic_io::Fabric::build(seismic_io::FabricSpec {
            revision: 1,
            buses: vec![seismic_io::SharedBusSpec {
                id: ddr,
                peak_bps: 60e9,
                uncertainty_rel: 0.03,
            }],
            links: Vec::new(),
        })
        .unwrap();
        let plane = seismic_io::IoPlane::new(fabric);
        let _res = plane
            .reserve_bus_until_fence(
                ddr,
                10e9,
                sync2.timeline_point(),
                Arc::new(sync2),
            )
            .unwrap();
        plane.poll_fence_reservations();
        assert_eq!(plane.bus_ledger(ddr).unwrap().active_reservations, 0);

        // On macOS Metal: `MTLSharedEvent` produces `DeviceAndHost` and rejects
        // `HostOnly` CPU fences on GPU `wait_fence`.
        #[cfg(target_os = "macos")]
        {
            let devices = Devices::discover(&[]).unwrap();
            if let Ok(metal_exec) = devices.executor_of(BackendName::Metal) {
                if let Ok(metal_dev) = devices.open(metal_exec) {
                let OpenedKind::Metal(opened) = &metal_dev.kind else {
                    unreachable!();
                };
                let ev = Arc::new(seismic_metal::SharedEvent::new(opened.service()).unwrap());
                let metal_fence = TransferSync::from_metal_event(
                    metal_dev.device_id(),
                    next_queue_id(),
                    1,
                    Arc::clone(&ev),
                );
                assert_eq!(metal_fence.timeline_point().capability, FenceCapability::DeviceAndHost);
                assert!(!metal_fence.is_reached().unwrap());
                ev.signal(1);
                assert!(metal_fence.is_reached().unwrap());

                let mut mseq = NativeGraphSequence::new(&metal_dev, NativeQueue::Compute);
                assert!(mseq.wait_fence(metal_fence).is_ok());

                // HostOnly CPU fence (even with matching device_id) is rejected on Metal!
                let host_only = TransferSync::from_cpu_reached(metal_dev.device_id(), 77, 1);
                assert!(
                    mseq.wait_fence(host_only).is_err(),
                    "Metal GPU wait_fence must reject HostOnly fence"
                );
            }
            }
        }
    }

    #[test]
    fn cuda_transfer_signal_gate_unblocks_compute_wait_word_alternating() {
        let devices = crate::devices::Devices::discover(&[]).unwrap();
        let Ok(exec) = devices.executor_of(seismic_lang::registry::BackendName::Cuda) else {
            return;
        };
        let device = devices.open(exec).unwrap();
        let OpenedKind::Cuda(opened) = &device.kind else {
            unreachable!();
        };
        let transfer = TransferInner::new(&device).unwrap();
        let gate = crate::native::timeline::TimelineHost::<crate::native::timeline::Gate>::new(&device)
            .unwrap();
        let super::super::timeline::TimelineKind::Cuda(sides) = gate.inner().kind() else {
            unreachable!();
        };
        let charges = crate::memory::Charges::tally(&device);
        let i32s = seismic_lang::registry::dense(seismic_lang::types::DType::I32);
        let elems = 64 * 1024u64;
        let bytes = (elems * 4) as usize;
        let dev_src = TensorInner::from_host(&charges, i32s, &[elems], &vec![0x5Au8; bytes]).unwrap();
        let dev_dst = TensorInner::from_host(&charges, i32s, &[elems], &vec![0u8; bytes]).unwrap();

        // Pre-queue 32 alternating compute waits ahead of copy-stream and host signals,
        // mirroring `DecodeReactor` + `StepSubmitter` across >8 layers.
        let mut completions = Vec::new();
        for step in 1u64..=32 {
            let mut batch = seismic_cuda::direct::DirectBatch::unobserved(
                opened.service(),
                seismic_cuda::NativeStream::Compute,
            )
            .unwrap();
            batch.wait_word(&sides.gate, step as u32).unwrap();
            let committed = batch.commit().unwrap();
            completions.push((step, committed));
        }
        for (step, committed) in completions {
            if step % 2 == 1 {
                let (_val, _sync) = transfer
                    .copy_synced(&[TransferItem::Run(TransferRun {
                        source: &dev_src,
                        source_offset: 0,
                        destination: &dev_dst,
                        destination_offset: 0,
                        bytes: bytes as u64,
                    })])
                    .unwrap();
                transfer.signal_gate(&gate, step).unwrap();
            } else {
                gate.signal(step).unwrap();
            }
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            while !committed.completed().unwrap() {
                assert!(
                    std::time::Instant::now() < deadline,
                    "CUDA Compute wait_word hung at step {step}, gate word={}",
                    sides.gate.load()
                );
                std::thread::yield_now();
            }
        }
        assert_eq!(sides.gate.load(), 32);

        // Also test whether `opened.service().drain()` (`cuEventSynchronize`) on Thread A
        // while `Compute` is blocked on `wait_word(33)` allows Thread B to call
        // `transfer.copy_synced` + `transfer.signal_gate(33)` without deadlocking on
        // the CUDA driver context mutex!
        let mut batch = seismic_cuda::direct::DirectBatch::unobserved(
            opened.service(),
            seismic_cuda::NativeStream::Compute,
        )
        .unwrap();
        batch.wait_word(&sides.gate, 33).unwrap();
        let _committed = batch.commit().unwrap();
        let dev_clone = opened.service().clone();
        let drainer = std::thread::spawn(move || dev_clone.drain());
        std::thread::sleep(Duration::from_millis(50));
        let (_val, _sync) = transfer
            .copy_synced(&[TransferItem::Run(TransferRun {
                source: &dev_src,
                source_offset: 0,
                destination: &dev_dst,
                destination_offset: 0,
                bytes: bytes as u64,
            })])
            .unwrap();
        transfer.signal_gate(&gate, 33).unwrap();
        drainer.join().unwrap().unwrap();
        assert_eq!(sides.gate.load(), 33);
    }
}
