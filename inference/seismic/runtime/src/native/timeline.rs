//! Host timelines: a monotonic value one device's native work and the host
//! both advance and wait on.
//!
//! A sealed graph carries timeline nodes between its kernel nodes
//! (`NativeGraphDraft::wait_timeline`, `signal_timeline`) and each run binds
//! every node to a timeline and a value:
//!
//! - **wait** (the host gate): the device runs nothing after the node until
//!   the timeline reaches the value, which the host sets (`Timeline::signal`)
//!   once what the work after the gate reads is in place;
//! - **signal**: the timeline reaches the value once the device has completed
//!   the work before the node, its writes visible to the host
//!   (`Timeline::wait`).
//!
//! Per backend:
//!
//! | Backend | Timeline | Wait | Signal |
//! |---|---|---|---|
//! | Metal | `MTLSharedEvent` | `encodeWaitForEvent` | `encodeSignalEvent` |
//! | CUDA | a 32-bit word in mapped pinned host memory | `cuStreamWaitValue32` (cyclic `>=`) | `cuStreamWriteValue32` |
//! | Vulkan | a timeline semaphore | the queue submission after the node waits on it | the queue submission before the node signals it |
//! | CPU | a value under a mutex | the run blocks before the node's launches | set when the launches before it return |
//!
//! Values only grow. CUDA compares the low 32 bits cyclically, so a wait or
//! a reached value is exact while the values in flight lie within 2^31 of
//! one another.

use crate::api::device::DeviceInner;
use crate::api::TensorError;
use crate::backends::OpenedKind;
use seismic_compiler::errors::ExecutionError;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// One device's host timeline.
pub struct TimelineInner {
    device: Arc<DeviceInner>,
    kind: TimelineKind,
}

pub(crate) enum TimelineKind {
    Cpu(CpuTimeline),
    #[cfg(target_os = "macos")]
    Metal(seismic_metal::SharedEvent),
    Cuda(seismic_cuda::direct::HostWord),
    #[cfg(not(target_os = "macos"))]
    Vulkan(seismic_vulkan::direct::HostTimeline),
}

/// The CPU backend's timeline: its runs execute on the submitting thread,
/// so a wait blocks that thread.
pub(crate) struct CpuTimeline {
    value: Mutex<u64>,
    changed: Condvar,
}

impl CpuTimeline {
    pub(crate) fn signal(&self, value: u64) {
        let mut current = self.value.lock().expect("timeline lock is never poisoned");
        if value > *current {
            *current = value;
            self.changed.notify_all();
        }
    }

    fn reached(&self, value: u64) -> bool {
        *self.value.lock().expect("timeline lock is never poisoned") >= value
    }

    /// Block until the timeline reaches `value`: a CPU run's host gate.
    pub(crate) fn wait_unbounded(&self, value: u64) {
        let current = self.value.lock().expect("timeline lock is never poisoned");
        let _reached = self
            .changed
            .wait_while(current, |current| *current < value)
            .expect("timeline lock is never poisoned");
    }

    pub(crate) fn wait(&self, value: u64, timeout: Duration) -> bool {
        let current = self.value.lock().expect("timeline lock is never poisoned");
        let (current, _) = self
            .changed
            .wait_timeout_while(current, timeout, |current| *current < value)
            .expect("timeline lock is never poisoned");
        *current >= value
    }
}

/// Whether 32-bit word `word` has reached `value`, compared as CUDA's
/// `CU_STREAM_WAIT_VALUE_GEQ` compares.
fn word_reached(word: u32, value: u64) -> bool {
    (word.wrapping_sub(value as u32) as i32) >= 0
}

impl TimelineInner {
    /// A timeline at 0 on `device`.
    pub fn new(device: &Arc<DeviceInner>) -> Result<Self, TensorError> {
        let kind = match &device.kind {
            OpenedKind::Cpu(_) => TimelineKind::Cpu(CpuTimeline {
                value: Mutex::new(0),
                changed: Condvar::new(),
            }),
            #[cfg(target_os = "macos")]
            OpenedKind::Metal(opened) => TimelineKind::Metal(
                seismic_metal::SharedEvent::new(opened.service())
                    .map_err(TensorError::Execution)?,
            ),
            OpenedKind::Cuda(opened) => TimelineKind::Cuda(
                seismic_cuda::direct::HostWord::new(opened.service())
                    .map_err(TensorError::Execution)?,
            ),
            #[cfg(not(target_os = "macos"))]
            OpenedKind::Vulkan(opened) => TimelineKind::Vulkan(
                seismic_vulkan::direct::HostTimeline::new(opened.service())
                    .map_err(TensorError::Execution)?,
            ),
        };
        Ok(Self {
            device: device.clone(),
            kind,
        })
    }

    pub(crate) fn device(&self) -> &Arc<DeviceInner> {
        &self.device
    }

    pub(crate) fn kind(&self) -> &TimelineKind {
        &self.kind
    }

    /// Raise the timeline to `value` from the host, after every host write
    /// before it; a value it has reached already changes nothing.
    pub fn signal(&self, value: u64) -> Result<(), ExecutionError> {
        match &self.kind {
            TimelineKind::Cpu(timeline) => timeline.signal(value),
            #[cfg(target_os = "macos")]
            TimelineKind::Metal(event) => {
                if event.value() < value {
                    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                    event.signal(value);
                }
            }
            TimelineKind::Cuda(word) => {
                if !word_reached(word.load(), value) {
                    word.store(value as u32);
                }
            }
            #[cfg(not(target_os = "macos"))]
            TimelineKind::Vulkan(timeline) => {
                std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                timeline.signal(value)?;
            }
        }
        Ok(())
    }

    /// Whether the timeline has reached `value`.
    pub fn reached(&self, value: u64) -> bool {
        match &self.kind {
            TimelineKind::Cpu(timeline) => timeline.reached(value),
            #[cfg(target_os = "macos")]
            TimelineKind::Metal(event) => event.value() >= value,
            TimelineKind::Cuda(word) => word_reached(word.load(), value),
            #[cfg(not(target_os = "macos"))]
            TimelineKind::Vulkan(timeline) => timeline.value() >= value,
        }
    }

    /// Wait at most `timeout` for the timeline to reach `value`, after which
    /// the host sees every device write before the signal; whether it did.
    pub fn wait(&self, value: u64, timeout: Duration) -> bool {
        let reached = match &self.kind {
            TimelineKind::Cpu(timeline) => timeline.wait(value, timeout),
            #[cfg(target_os = "macos")]
            TimelineKind::Metal(event) => event.wait(value, timeout),
            // A mapped word has no blocking wait: poll it, yielding.
            TimelineKind::Cuda(word) => {
                let start = Instant::now();
                loop {
                    if word_reached(word.load(), value) {
                        break true;
                    }
                    if start.elapsed() >= timeout {
                        break false;
                    }
                    std::thread::yield_now();
                }
            }
            #[cfg(not(target_os = "macos"))]
            TimelineKind::Vulkan(timeline) => timeline.wait(value, timeout),
        };
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        reached
    }
}

/// What one timeline node of a submitted run does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimelineNode {
    /// The device waits until the timeline reaches the bound value.
    Wait,
    /// The timeline reaches the bound value once the work before completes.
    Signal,
}

/// A timeline node bound to its timeline and value for one run.
#[derive(Clone)]
pub(crate) struct TimelineOperation {
    pub(crate) node: TimelineNode,
    pub(crate) timeline: Arc<TimelineInner>,
    pub(crate) value: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_word_is_compared_cyclically() {
        assert!(word_reached(5, 5));
        assert!(word_reached(6, 5));
        assert!(!word_reached(4, 5));
        // Past 2^32 the low words wrap and still compare.
        assert!(word_reached(2, (1u64 << 32) + 1));
        assert!(!word_reached(u32::MAX, (1u64 << 32) + 1));
    }

    #[test]
    fn a_cpu_timeline_wakes_a_waiter() {
        let timeline = Arc::new(CpuTimeline {
            value: Mutex::new(0),
            changed: Condvar::new(),
        });
        assert!(!timeline.wait(1, Duration::from_millis(1)));
        let signaler = timeline.clone();
        let thread = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(5));
            signaler.signal(3);
        });
        assert!(timeline.wait(2, Duration::from_secs(10)));
        thread.join().unwrap();
        // A lower value never lowers it.
        timeline.signal(1);
        assert!(timeline.reached(3));
    }
}
