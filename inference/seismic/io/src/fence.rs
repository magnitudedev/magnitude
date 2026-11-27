//! Typed hardware/host timeline synchronization primitives shared between
//! `seismic-io` v2 and `seismic-runtime`.
//!
//! Every [`TimelinePoint`] identifies an exact point on ONE ordered producer
//! queue of ONE device: `(DeviceId, ProducerId, u64)`. Two independent
//! producer streams NEVER advance a shared scalar by `max`.

use crate::fabric::{DeviceId, ProducerId};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// Whether a [`TimelinePoint`] can be waited on by GPU queues or only by the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FenceCapability {
    /// Waitable by GPU queues on `device` AND by the host
    /// (CUDA `CUevent` pool, Metal `MTLSharedEvent`, Vulkan `VK_KHR_timeline_semaphore`).
    DeviceAndHost,
    /// Host-waitable only (e.g., CPU `AtomicU64` / NVMe completion).
    /// Passing this to a GPU `wait_fence` is rejected as an invariant error.
    HostOnly,
}

/// Identifies an exact point on ONE ordered producer queue of ONE device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TimelinePoint {
    pub device: DeviceId,
    pub producer: ProducerId,
    pub value: u64,
    pub capability: FenceCapability,
}

impl TimelinePoint {
    #[inline]
    pub const fn new(
        device: DeviceId,
        producer: ProducerId,
        value: u64,
        capability: FenceCapability,
    ) -> Self {
        Self {
            device,
            producer,
            value,
            capability,
        }
    }

    /// Validate that `self` may be waited on by a GPU queue on `device`.
    /// Rejects `FenceCapability::HostOnly` and foreign devices loudly.
    pub fn validate_for_device_wait(&self, device: DeviceId) -> Result<(), FenceError> {
        if self.device != device {
            return Err(FenceError::DeviceMismatch {
                expected: device,
                got: self.device,
            });
        }
        if self.capability != FenceCapability::DeviceAndHost {
            return Err(FenceError::HostOnlyOnDeviceWait(*self));
        }
        Ok(())
    }
}

/// Why a timeline wait or validation failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FenceError {
    /// Attempted to enqueue a `HostOnly` timeline point on a GPU queue wait.
    HostOnlyOnDeviceWait(TimelinePoint),
    /// Attempted to wait on a timeline point belonging to a different device.
    DeviceMismatch { expected: DeviceId, got: DeviceId },
    /// The producer queue for this timeline point is unknown to this observer.
    UnknownProducer { device: DeviceId, producer: ProducerId },
    /// The producer queue faulted; no future point on it will be reached.
    Faulted(String),
}

impl fmt::Display for FenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostOnlyOnDeviceWait(pt) => {
                write!(f, "cannot wait on HostOnly timeline point {pt:?} from a device queue")
            }
            Self::DeviceMismatch { expected, got } => {
                write!(f, "timeline point belongs to device {got:?}, not {expected:?}")
            }
            Self::UnknownProducer { device, producer } => {
                write!(f, "unknown timeline producer ({device:?}, {producer:?})")
            }
            Self::Faulted(reason) => write!(f, "timeline producer faulted: {reason}"),
        }
    }
}

impl std::error::Error for FenceError {}

/// Backend-agnostic observer of [`TimelinePoint`] completion.
///
/// `seismic-runtime` implements this for CUDA (`CUevent` generation pools),
/// Metal (`MTLSharedEvent`), and Vulkan (`VK_KHR_timeline_semaphore`), while
/// [`HostTimeline`] and [`MultiProducerTimeline`] provide host-only / multi-stream
/// implementations directly inside `seismic-io`.
pub trait TimelineObserver: Send + Sync {
    /// Non-blocking check whether `point` has been reached.
    fn is_reached(&self, point: TimelinePoint) -> Result<bool, FenceError>;

    /// Block up to `timeout` waiting for `point` to be reached.
    fn wait(&self, point: TimelinePoint, timeout: Duration) -> Result<bool, FenceError>;
}

/// Single-producer CPU/host timeline backed by `AtomicU64` + `Condvar`,
/// explicitly tagged [`FenceCapability::HostOnly`].
pub struct HostTimeline {
    device: DeviceId,
    producer: ProducerId,
    completed: AtomicU64,
    state: Mutex<HostTimelineState>,
    changed: Condvar,
}

#[derive(Default)]
struct HostTimelineState {
    completed: u64,
    fault: Option<String>,
}

impl HostTimeline {
    pub fn new(device: DeviceId, producer: ProducerId) -> Arc<Self> {
        Arc::new(Self {
            device,
            producer,
            completed: AtomicU64::new(0),
            state: Mutex::new(HostTimelineState::default()),
            changed: Condvar::new(),
        })
    }

    pub fn device(&self) -> DeviceId {
        self.device
    }

    pub fn producer(&self) -> ProducerId {
        self.producer
    }

    /// Create a [`TimelinePoint`] for `value` on this host-only producer.
    pub fn point(&self, value: u64) -> TimelinePoint {
        TimelinePoint::new(self.device, self.producer, value, FenceCapability::HostOnly)
    }

    /// Current monotonic completed value.
    pub fn current(&self) -> u64 {
        self.completed.load(Ordering::Acquire)
    }

    /// Advance this producer's monotonic completed watermark to `value`.
    /// Panics if `value` moves backwards (invariant violation on a single ordered producer).
    pub fn advance(&self, value: u64) {
        let mut state = self.state.lock().expect("host timeline");
        assert!(
            value >= state.completed,
            "HostTimeline({:?}, {:?}) regressed from {} to {value}",
            self.device,
            self.producer,
            state.completed
        );
        state.completed = value;
        self.completed.store(value, Ordering::Release);
        drop(state);
        self.changed.notify_all();
    }

    /// Mark this producer faulted and wake all waiters.
    pub fn fault(&self, reason: impl Into<String>) {
        let mut state = self.state.lock().expect("host timeline");
        if state.fault.is_none() {
            state.fault = Some(reason.into());
        }
        drop(state);
        self.changed.notify_all();
    }
}

impl TimelineObserver for HostTimeline {
    fn is_reached(&self, point: TimelinePoint) -> Result<bool, FenceError> {
        if point.device != self.device {
            return Err(FenceError::DeviceMismatch {
                expected: self.device,
                got: point.device,
            });
        }
        if point.producer != self.producer {
            return Err(FenceError::UnknownProducer {
                device: point.device,
                producer: point.producer,
            });
        }
        if self.completed.load(Ordering::Acquire) >= point.value {
            return Ok(true);
        }
        let state = self.state.lock().expect("host timeline");
        if state.completed >= point.value {
            return Ok(true);
        }
        if let Some(reason) = &state.fault {
            return Err(FenceError::Faulted(reason.clone()));
        }
        Ok(false)
    }

    fn wait(&self, point: TimelinePoint, timeout: Duration) -> Result<bool, FenceError> {
        if self.is_reached(point)? {
            return Ok(true);
        }
        let deadline = Instant::now() + timeout;
        let mut state = self.state.lock().expect("host timeline");
        loop {
            if state.completed >= point.value {
                return Ok(true);
            }
            if let Some(reason) = &state.fault {
                return Err(FenceError::Faulted(reason.clone()));
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(false);
            }
            let (guard, _) = self
                .changed
                .wait_timeout(state, remaining)
                .expect("host timeline");
            state = guard;
        }
    }
}

/// Per-`(DeviceId, ProducerId)` timeline registry supporting arbitrary
/// independent producer streams (including `DeviceAndHost` timelines in
/// deterministic replay or backend wrappers). Independent producers NEVER
/// share or overwrite each other's monotonic watermarks.
#[derive(Default)]
pub struct MultiProducerTimeline {
    producers: Mutex<HashMap<(DeviceId, ProducerId), ProducerEntry>>,
    changed: Condvar,
}

#[derive(Default)]
struct ProducerEntry {
    completed: u64,
    fault: Option<String>,
}

impl MultiProducerTimeline {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Register a `(device, producer)` stream starting at watermark `0`.
    pub fn register(&self, device: DeviceId, producer: ProducerId) {
        let mut map = self.producers.lock().expect("multi-producer timeline");
        map.entry((device, producer)).or_default();
    }

    /// Advance `(device, producer)` independently to `value`.
    pub fn advance(&self, device: DeviceId, producer: ProducerId, value: u64) {
        let mut map = self.producers.lock().expect("multi-producer timeline");
        let entry = map.entry((device, producer)).or_default();
        assert!(
            value >= entry.completed,
            "producer ({device:?}, {producer:?}) regressed from {} to {value}",
            entry.completed
        );
        entry.completed = value;
        drop(map);
        self.changed.notify_all();
    }

    /// Fault `(device, producer)` independently without affecting other producers.
    pub fn fault(&self, device: DeviceId, producer: ProducerId, reason: impl Into<String>) {
        let mut map = self.producers.lock().expect("multi-producer timeline");
        let entry = map.entry((device, producer)).or_default();
        if entry.fault.is_none() {
            entry.fault = Some(reason.into());
        }
        drop(map);
        self.changed.notify_all();
    }

    /// Current watermark for `(device, producer)`.
    pub fn current(&self, device: DeviceId, producer: ProducerId) -> Option<u64> {
        self.producers
            .lock()
            .expect("multi-producer timeline")
            .get(&(device, producer))
            .map(|e| e.completed)
    }
}

impl TimelineObserver for MultiProducerTimeline {
    fn is_reached(&self, point: TimelinePoint) -> Result<bool, FenceError> {
        let map = self.producers.lock().expect("multi-producer timeline");
        let Some(entry) = map.get(&(point.device, point.producer)) else {
            return Err(FenceError::UnknownProducer {
                device: point.device,
                producer: point.producer,
            });
        };
        if entry.completed >= point.value {
            return Ok(true);
        }
        if let Some(reason) = &entry.fault {
            return Err(FenceError::Faulted(reason.clone()));
        }
        Ok(false)
    }

    fn wait(&self, point: TimelinePoint, timeout: Duration) -> Result<bool, FenceError> {
        let deadline = Instant::now() + timeout;
        let mut map = self.producers.lock().expect("multi-producer timeline");
        loop {
            let Some(entry) = map.get(&(point.device, point.producer)) else {
                return Err(FenceError::UnknownProducer {
                    device: point.device,
                    producer: point.producer,
                });
            };
            if entry.completed >= point.value {
                return Ok(true);
            }
            if let Some(reason) = &entry.fault {
                return Err(FenceError::Faulted(reason.clone()));
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(false);
            }
            let (guard, _) = self
                .changed
                .wait_timeout(map, remaining)
                .expect("multi-producer timeline");
            map = guard;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_only_rejected_for_device_wait() {
        let dev = DeviceId(1);
        let host_pt = TimelinePoint::new(dev, ProducerId(1), 4, FenceCapability::HostOnly);
        assert_eq!(
            host_pt.validate_for_device_wait(dev),
            Err(FenceError::HostOnlyOnDeviceWait(host_pt))
        );

        let gpu_pt = TimelinePoint::new(dev, ProducerId(2), 4, FenceCapability::DeviceAndHost);
        assert_eq!(gpu_pt.validate_for_device_wait(dev), Ok(()));
        assert_eq!(
            gpu_pt.validate_for_device_wait(DeviceId(2)),
            Err(FenceError::DeviceMismatch {
                expected: DeviceId(2),
                got: dev,
            })
        );
    }

    #[test]
    fn independent_producers_never_corrupt_each_other() {
        let tl = MultiProducerTimeline::new();
        let dev = DeviceId(0);
        let a = ProducerId(1);
        let b = ProducerId(2);
        tl.register(dev, a);
        tl.register(dev, b);

        let pt_a = TimelinePoint::new(dev, a, 10, FenceCapability::DeviceAndHost);
        let pt_b = TimelinePoint::new(dev, b, 2, FenceCapability::DeviceAndHost);

        // Stream B completes to 20 before Stream A completes to 10.
        tl.advance(dev, b, 20);
        assert!(tl.is_reached(pt_b).unwrap());
        assert!(!tl.is_reached(pt_a).unwrap(), "stream B must not advance stream A");

        tl.advance(dev, a, 10);
        assert!(tl.is_reached(pt_a).unwrap());
    }
}
