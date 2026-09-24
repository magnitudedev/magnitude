//! Submission tracing for measurement. While a trace is active on a device,
//! every native submission on that device is recorded with its host encode
//! interval and, once complete, its device interval on the same host clock.
//! At launch detail each launch runs in its own timed unit (Metal encoder,
//! CUDA event pair, Vulkan timestamp pair), so device time is attributed to
//! entries; that changes the device work and is for attribution runs only.

use super::RouteSubmission;
use crate::api::device::DeviceInner;
use crate::backends::OpenedKind;
use seismic_compiler::errors::ExecutionError;
use std::sync::{Arc, Mutex};

/// How much a trace records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceDetail {
    /// Production submissions, timed per submission.
    Submissions,
    /// Every launch in its own timed unit.
    Launches,
}

#[derive(Debug)]
pub enum TraceError {
    /// Another trace is already active on the device.
    AlreadyActive,
    Execution(ExecutionError),
}

impl std::fmt::Display for TraceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyActive => formatter.write_str("a submission trace is already active"),
            Self::Execution(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for TraceError {}

/// The host monotonic clock (seconds) every traced time is expressed in. On
/// macOS it is the time base of Metal's command-buffer GPU times.
pub fn host_seconds() -> f64 {
    #[cfg(target_os = "macos")]
    {
        seismic_metal::host_seconds()
    }
    #[cfg(not(target_os = "macos"))]
    {
        static ORIGIN: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        ORIGIN
            .get_or_init(std::time::Instant::now)
            .elapsed()
            .as_secs_f64()
    }
}

/// One launch of a traced submission.
#[derive(Clone, Debug)]
pub struct TracedLaunch {
    /// Entry name of the call the launch belongs to.
    pub entry: String,
    /// Index of the launch within its call (multi-launch entries).
    pub launch: usize,
    /// Device interval; `None` for an empty launch.
    pub device: Option<(f64, f64)>,
}

/// One completed traced submission. Times are [`host_seconds`].
#[derive(Clone, Debug)]
pub struct TracedSubmission {
    /// Host time when encoding began.
    pub encode_start: f64,
    /// Host time when the submission was committed to the device queue.
    pub committed: f64,
    /// Device execution interval.
    pub device: (f64, f64),
    /// Launches in order. Device intervals are present at launch detail.
    pub launches: Vec<TracedLaunch>,
}

struct Pending {
    route: Arc<RouteSubmission>,
    labels: Vec<(String, usize)>,
    encode_start: f64,
    committed: f64,
}

/// Timestamp samples one launch-detail trace may hold between collections
/// (two per launch): Metal's largest sample buffer, 32 KiB of 8-byte
/// timestamps.
const LAUNCH_SAMPLES: usize = 4_096;

enum Timeline {
    /// Device intervals are already host times (CPU, and Metal at
    /// submission detail).
    Host,
    /// Metal at launch detail: the trace's shared timestamp samples.
    #[cfg(target_os = "macos")]
    Metal(Arc<seismic_metal::LaunchTimestamps>),
    Cuda(seismic_cuda::direct::TimelineAnchor),
    #[cfg(not(target_os = "macos"))]
    Vulkan(seismic_vulkan::direct::TimelineAnchor),
}

pub(crate) struct TraceSink {
    detail: TraceDetail,
    timeline: Timeline,
    pending: Mutex<Vec<Pending>>,
}

impl TraceSink {
    pub(super) fn detail(&self) -> TraceDetail {
        self.detail
    }

    /// The timestamp samples of a Metal launch-detail trace.
    #[cfg(target_os = "macos")]
    pub(super) fn metal_timestamps(&self) -> &Arc<seismic_metal::LaunchTimestamps> {
        match &self.timeline {
            Timeline::Metal(timestamps) => timestamps,
            _ => unreachable!("a Metal launch-detail trace always holds timestamp samples"),
        }
    }

    pub(super) fn record(
        &self,
        route: Arc<RouteSubmission>,
        labels: Vec<(String, usize)>,
        encode_start: f64,
        committed: f64,
    ) {
        self.pending
            .lock()
            .expect("trace sink lock is never poisoned")
            .push(Pending {
                route,
                labels,
                encode_start,
                committed,
            });
    }
}

/// An active trace. Dropping it stops recording.
pub struct SubmissionTrace {
    device: Arc<DeviceInner>,
    sink: Arc<TraceSink>,
}

impl SubmissionTrace {
    pub fn start(device: &Arc<DeviceInner>, detail: TraceDetail) -> Result<Self, TraceError> {
        let timeline = match &device.kind {
            OpenedKind::Cuda(opened) => Timeline::Cuda(
                seismic_cuda::direct::TimelineAnchor::record(opened.service(), host_seconds)
                    .map_err(TraceError::Execution)?,
            ),
            #[cfg(not(target_os = "macos"))]
            OpenedKind::Vulkan(opened) => Timeline::Vulkan(
                seismic_vulkan::direct::TimelineAnchor::record(opened.service(), host_seconds)
                    .map_err(TraceError::Execution)?,
            ),
            #[cfg(target_os = "macos")]
            OpenedKind::Metal(opened) if detail == TraceDetail::Launches => Timeline::Metal(
                Arc::new(
                    seismic_metal::LaunchTimestamps::new(opened.service(), LAUNCH_SAMPLES)
                        .map_err(TraceError::Execution)?,
                ),
            ),
            _ => Timeline::Host,
        };
        let sink = Arc::new(TraceSink {
            detail,
            timeline,
            pending: Mutex::new(Vec::new()),
        });
        let mut active = device.trace.lock().expect("trace slot lock is never poisoned");
        if active.is_some() {
            return Err(TraceError::AlreadyActive);
        }
        *active = Some(sink.clone());
        Ok(Self {
            device: device.clone(),
            sink,
        })
    }

    /// Wait for every submission recorded so far and return them in
    /// submission order, leaving the trace active and empty. Launch-detail
    /// traces reuse their timestamp samples afterwards, so no submission on
    /// the device may be in flight outside the trace (single-submitter use).
    pub fn collect(&self) -> Result<Vec<TracedSubmission>, TraceError> {
        let pending = std::mem::take(
            &mut *self
                .sink
                .pending
                .lock()
                .expect("trace sink lock is never poisoned"),
        );
        let resolved = pending
            .into_iter()
            .map(|pending| self.resolve(pending))
            .collect();
        #[cfg(target_os = "macos")]
        if let Timeline::Metal(timestamps) = &self.sink.timeline {
            timestamps.reset();
        }
        resolved
    }

    fn resolve(&self, pending: Pending) -> Result<TracedSubmission, TraceError> {
        let execution = TraceError::Execution;
        let (device, intervals) = match (&*pending.route, &self.sink.timeline) {
            (RouteSubmission::Cpu { outcome, interval }, _) => {
                outcome.clone().map_err(execution)?;
                (*interval, None)
            }
            #[cfg(target_os = "macos")]
            (RouteSubmission::Metal(submission), _) => {
                submission.finish().map_err(execution)?;
                (
                    submission.device_interval(),
                    submission.launch_intervals().map_err(execution)?,
                )
            }
            (RouteSubmission::Cuda(submission), Timeline::Cuda(anchor)) => {
                submission.finish().map_err(execution)?;
                (
                    submission.device_interval(anchor).map_err(execution)?,
                    submission.launch_intervals(anchor).map_err(execution)?,
                )
            }
            (RouteSubmission::Cuda(_), _) => {
                unreachable!("a CUDA device trace always holds a CUDA timeline")
            }
            #[cfg(not(target_os = "macos"))]
            (RouteSubmission::Vulkan(submission), Timeline::Vulkan(anchor)) => {
                submission.finish().map_err(execution)?;
                (
                    submission.device_interval(anchor).map_err(execution)?,
                    submission.launch_intervals(anchor).map_err(execution)?,
                )
            }
            #[cfg(not(target_os = "macos"))]
            (RouteSubmission::Vulkan(_), _) => {
                unreachable!("a Vulkan device trace always holds a Vulkan timeline")
            }
        };
        if intervals
            .as_ref()
            .is_some_and(|intervals| intervals.len() != pending.labels.len())
        {
            return Err(TraceError::Execution(ExecutionError::SubmissionFailed(
                "timed submission launch count differs from its calls".into(),
            )));
        }
        let launches = pending
            .labels
            .into_iter()
            .enumerate()
            .map(|(index, (entry, launch))| TracedLaunch {
                entry,
                launch,
                device: intervals.as_ref().and_then(|intervals| intervals[index]),
            })
            .collect();
        Ok(TracedSubmission {
            encode_start: pending.encode_start,
            committed: pending.committed,
            device,
            launches,
        })
    }
}

impl Drop for SubmissionTrace {
    fn drop(&mut self) {
        let mut active = self
            .device
            .trace
            .lock()
            .expect("trace slot lock is never poisoned");
        if active
            .as_ref()
            .is_some_and(|sink| Arc::ptr_eq(sink, &self.sink))
        {
            *active = None;
        }
    }
}
