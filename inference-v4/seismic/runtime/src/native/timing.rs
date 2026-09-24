//! Device timing of a prepared native implementation at one or more
//! workloads ("points"), each cycling through its own argument sets.
//!
//! A point's calls are validated and placed once; every sample resubmits
//! them. Timing is device time (command-buffer timestamps on Metal, stream
//! events on CUDA), so host latency never enters a sample, and submissions
//! need not wait for one another: every sample of every point is submitted
//! back to back and read afterwards, one host round trip per batch instead
//! of one per sample.
//!
//! The first pass over a point's rotation calibrates the repetitions per
//! sample so that a sample covers at least `min_sample_seconds` of device
//! time. When one pass already does, the calibrating pass is itself the
//! first sample: device time excludes host-side first-use costs, and
//! measured calibrating passes match later samples (M4 Max, 2026-09-24), so
//! a separate warm-up would only spend device time.

use super::{
    median, median_of, CallError, MeasureOptions, Measurement, NativeBoundCall, NativePrepared,
    NativeSubmission, StandaloneCalls,
};
use crate::api::kernel::EncodedArgs;
use std::collections::BTreeMap;
use std::sync::Arc;

/// One point's calls, placed once and submitted for every sample.
pub(crate) struct PointTiming {
    kernel: Arc<NativePrepared>,
    calls: Vec<NativeBoundCall>,
    /// Passes over the rotation per sample; `None` until calibrated.
    repetitions: Option<usize>,
    rotation_bytes: u64,
    samples: Vec<f64>,
}

/// A submitted sample of one point.
pub(crate) struct PendingSample {
    submission: NativeSubmission,
    calls: usize,
}

impl PendingSample {
    /// Device seconds per call, once the sample completes.
    pub(crate) fn seconds(self) -> Result<f64, CallError> {
        Ok(self.submission.device_seconds()? / self.calls as f64)
    }
}

impl PointTiming {
    /// Validate and place every call of `rotation`. Results of each argument
    /// set are allocated once and reused by every sample; scratch is the
    /// standalone arena.
    pub(crate) fn new(
        kernel: &Arc<NativePrepared>,
        rotation: Vec<EncodedArgs>,
    ) -> Result<Self, CallError> {
        if rotation.is_empty() {
            return Err(CallError::Workflow(crate::api::WorkflowError::Empty));
        }
        let calls = {
            let mut standalone = kernel
                .standalone
                .lock()
                .expect("native standalone-call lock poisoned");
            rotation
                .into_iter()
                .map(|args| kernel.prepare_call(&mut standalone, args, None))
                .collect::<Result<Vec<_>, _>>()?
        };
        let mut distinct = BTreeMap::new();
        for call in &calls {
            for (allocation, _) in &call.access {
                distinct.insert(allocation.identity(), allocation.bytes());
            }
        }
        Ok(Self {
            kernel: kernel.clone(),
            calls,
            repetitions: None,
            rotation_bytes: distinct.values().sum(),
            samples: Vec::new(),
        })
    }

    /// Submit one sample (one pass until calibrated).
    pub(crate) fn submit(&self) -> Result<PendingSample, CallError> {
        let repetitions = self.repetitions.unwrap_or(1);
        let submission = StandaloneCalls {
            kernel: &self.kernel,
            calls: &self.calls,
        }
        .submit(repetitions)?;
        Ok(PendingSample {
            submission,
            calls: repetitions * self.calls.len(),
        })
    }

    /// Record a completed sample; the first calibrates.
    pub(crate) fn record(&mut self, seconds: f64, min_sample_seconds: f64) {
        if self.repetitions.is_some() {
            self.samples.push(seconds);
            return;
        }
        let pass = seconds * self.calls.len() as f64;
        let repetitions = if pass > 0.0 {
            ((min_sample_seconds / pass).ceil() as usize).max(1)
        } else {
            1
        };
        self.repetitions = Some(repetitions);
        if repetitions == 1 {
            self.samples.push(seconds);
        }
    }

    /// The samples so far as a measurement.
    pub(crate) fn measurement(&self) -> Measurement {
        let median = median(&self.samples);
        let deviation = median_of(
            self.samples
                .iter()
                .map(|sample| (sample - median).abs())
                .collect(),
        );
        Measurement {
            samples: self.samples.clone(),
            median,
            deviation,
            repetitions: self.repetitions.unwrap_or(1) * self.calls.len(),
            rotation_bytes: self.rotation_bytes,
        }
    }
}

/// A point whose sample failed, and why.
pub(crate) struct PointFailure {
    pub(crate) point: usize,
    pub(crate) error: CallError,
}

/// Submit the pending samples in order, then read each into its point.
fn collect(
    points: &mut [PointTiming],
    order: &[usize],
    min_sample_seconds: f64,
) -> Result<(), PointFailure> {
    let pending = order
        .iter()
        .map(|&point| {
            points[point]
                .submit()
                .map(|sample| (point, sample))
                .map_err(|error| PointFailure { point, error })
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (point, sample) in pending {
        let seconds = sample
            .seconds()
            .map_err(|error| PointFailure { point, error })?;
        points[point].record(seconds, min_sample_seconds);
    }
    Ok(())
}

/// Bring every point to at least `options.samples` samples: an uncalibrated
/// point first gets its calibrating pass, then samples are taken round by
/// round (each point once per round), all submitted before any is read.
pub(crate) fn sample(points: &mut [PointTiming], options: &MeasureOptions) -> Result<(), PointFailure> {
    let uncalibrated = (0..points.len())
        .filter(|&point| points[point].repetitions.is_none())
        .collect::<Vec<_>>();
    collect(points, &uncalibrated, options.min_sample_seconds)?;
    let missing = points
        .iter()
        .map(|point| options.samples.saturating_sub(point.samples.len()))
        .collect::<Vec<_>>();
    let rounds = missing.iter().copied().max().unwrap_or(0);
    let order = (0..rounds)
        .flat_map(|round| (0..missing.len()).filter(|&point| round < missing[point]).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    collect(points, &order, options.min_sample_seconds)
}

impl NativePrepared {
    /// Measure calls cycling through `rotation`.
    pub(crate) fn measure(
        self: &Arc<Self>,
        rotation: Vec<EncodedArgs>,
        options: &MeasureOptions,
    ) -> Result<Measurement, CallError> {
        if options.samples == 0 {
            return Err(CallError::Workflow(crate::api::WorkflowError::Empty));
        }
        let mut points = [PointTiming::new(self, rotation)?];
        sample(&mut points, options).map_err(|failure| failure.error)?;
        Ok(points[0].measurement())
    }
}
