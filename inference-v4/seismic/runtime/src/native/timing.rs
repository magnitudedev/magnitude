//! Device timing of a prepared native implementation at one or more
//! workloads ("points"), each cycling through its own argument sets.
//!
//! A point's calls are validated and placed once; every sample resubmits
//! them. Timing is device time (command-buffer timestamps on Metal, stream
//! events on CUDA), so host latency never enters a sample. Each sample
//! completes before the next is submitted: a Metal queue runs command
//! buffers without a hazard between them concurrently, so samples of
//! different points submitted back to back overlap and each one's interval
//! includes the other's work (measured on an M4 Pro, 2026-09-24: a decode
//! GEMV's samples read 185 µs next to a 4-row point's and 67 µs alone).
//!
//! The first pass over a point's rotation calibrates the repetitions per
//! sample so that a sample covers at least `min_sample_seconds` of device
//! time. It is never a sample: it pays first-use device costs.
//!
//! A device that idled (while configurations were formed, for example) runs
//! at a low clock until it has been busy for a while: on an M4 Pro the first
//! configuration measured after forming a batch read 2–4× its time for its
//! first 4–15 samples, and sometimes for all of them. [`warm`] keeps the
//! device busy until its speed stops changing before measuring.

use super::{
    median, median_of, CallError, MeasureOptions, Measurement, NativeBoundCall, NativePrepared,
    NativeSubmission, StandaloneCalls,
};
use crate::api::kernel::{EncodedArgs, EncodedOutputs};
use crate::api::tensor::TensorInner;
use std::collections::HashSet;
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
        Self::place(kernel, rotation, None)
    }

    /// Place a candidate against a point's shared result storage. Group
    /// sweeps submit one timing at a time and wait for its completion, so a
    /// later candidate can safely write the same buffers.
    pub(crate) fn reusing_outputs(
        kernel: &Arc<NativePrepared>,
        rotation: Vec<EncodedArgs>,
        outputs: &mut Vec<Vec<Arc<TensorInner>>>,
    ) -> Result<Self, CallError> {
        Self::place(kernel, rotation, Some(outputs))
    }

    fn place(
        kernel: &Arc<NativePrepared>,
        rotation: Vec<EncodedArgs>,
        mut outputs: Option<&mut Vec<Vec<Arc<TensorInner>>>>,
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
                .enumerate()
                .map(|(index, args)| {
                    let supplied =
                        outputs
                            .as_ref()
                            .and_then(|pool| pool.get(index))
                            .map(|tensors| {
                                let mut encoded = EncodedOutputs::new();
                                for tensor in tensors {
                                    encoded.push_tensor(tensor.clone());
                                }
                                encoded
                            });
                    let call = kernel.prepare_call(&mut standalone, args, supplied)?;
                    if let Some(pool) = outputs.as_mut() {
                        if index == pool.len() {
                            pool.push(call.results.clone());
                        }
                    }
                    Ok(call)
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        let mut distinct = HashSet::new();
        for call in &calls {
            for (allocation, _) in &call.access {
                distinct.insert((allocation.identity(), allocation.bytes()));
            }
        }
        Ok(Self {
            kernel: kernel.clone(),
            calls,
            repetitions: None,
            rotation_bytes: distinct.into_iter().map(|(_, bytes)| bytes).sum(),
            samples: Vec::new(),
        })
    }

    /// Declaration ordinals of the launches that do work at this point: active
    /// (`when`) with a nonempty grid in some call of the rotation.
    pub(crate) fn working_launches(&self) -> Vec<usize> {
        let mut working = self
            .calls
            .iter()
            .flat_map(|call| {
                call.launches
                    .iter()
                    .enumerate()
                    .filter(|(_, geometry)| {
                        geometry.is_some_and(|geometry| {
                            geometry.groups.iter().all(|groups| *groups > 0)
                        })
                    })
                    .map(|(ordinal, _)| ordinal)
            })
            .collect::<Vec<_>>();
        working.sort_unstable();
        working.dedup();
        working
    }

    /// Submit one sample (one pass until calibrated).
    pub(crate) fn submit(&self) -> Result<PendingSample, CallError> {
        self.submit_passes(self.repetitions.unwrap_or(1))
    }

    /// Submit `passes` passes over the rotation as one unit of device work.
    fn submit_passes(&self, passes: usize) -> Result<PendingSample, CallError> {
        let submission = StandaloneCalls {
            kernel: &self.kernel,
            calls: &self.calls,
        }
        .submit(passes)?;
        Ok(PendingSample {
            submission,
            calls: passes * self.calls.len(),
        })
    }

    /// Record a completed sample; the first calibrates and is not kept (it
    /// pays first-use costs: on an M4 Pro, twice the later samples).
    pub(crate) fn record(&mut self, seconds: f64, min_sample_seconds: f64) {
        if self.repetitions.is_some() {
            self.samples.push(seconds);
            return;
        }
        let pass = seconds * self.calls.len() as f64;
        self.repetitions = Some(if pass > 0.0 {
            ((min_sample_seconds / pass).ceil() as usize).max(1)
        } else {
            1
        });
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

/// Continuous device work every warm-up does first: an idle device's clock
/// rises over tens of milliseconds of load, holding at intermediate levels.
const WARM_MIN_SECONDS: f64 = 0.05;
/// Device time of one warm-up chunk after the first.
const WARM_CHUNK_SECONDS: f64 = 0.01;
/// Consecutive warm-up chunks must agree this closely per pass.
const WARM_AGREEMENT: f64 = 0.02;
/// Device time after which warming stops whether or not chunks agree.
const WARM_LIMIT_SECONDS: f64 = 0.5;

/// Keep the device busy with `point`'s calls until it runs at its sustained
/// clock: [`WARM_MIN_SECONDS`] of work, then chunks of
/// [`WARM_CHUNK_SECONDS`] until two consecutive chunks take the same time
/// per pass within [`WARM_AGREEMENT`], at most [`WARM_LIMIT_SECONDS`].
pub(crate) fn warm(point: &PointTiming) -> Result<(), CallError> {
    let pass = point.submit_passes(1)?.submission.device_seconds()?;
    let passes = |seconds: f64| {
        if pass > 0.0 {
            ((seconds / pass).ceil() as usize).max(1)
        } else {
            1
        }
    };
    let mut spent = point
        .submit_passes(passes(WARM_MIN_SECONDS))?
        .submission
        .device_seconds()?;
    let chunk = passes(WARM_CHUNK_SECONDS);
    let mut previous = f64::INFINITY;
    while spent < WARM_LIMIT_SECONDS {
        let seconds = point.submit_passes(chunk)?.submission.device_seconds()?;
        spent += seconds;
        if (seconds - previous).abs() <= WARM_AGREEMENT * previous {
            break;
        }
        previous = seconds;
    }
    Ok(())
}

/// A point whose sample failed, and why.
pub(crate) struct PointFailure {
    pub(crate) point: usize,
    pub(crate) error: CallError,
}

/// Take one sample of each point of `order` in turn, each completing before
/// the next is submitted.
fn collect(
    points: &mut [PointTiming],
    order: &[usize],
    min_sample_seconds: f64,
) -> Result<(), PointFailure> {
    for &point in order {
        let seconds = points[point]
            .submit()
            .and_then(PendingSample::seconds)
            .map_err(|error| PointFailure { point, error })?;
        points[point].record(seconds, min_sample_seconds);
    }
    Ok(())
}

/// Bring every point to at least `options.samples` samples: an uncalibrated
/// point first gets its calibrating pass, then samples are taken round by
/// round (each point once per round).
pub(crate) fn sample(
    points: &mut [PointTiming],
    options: &MeasureOptions,
) -> Result<(), PointFailure> {
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
        .flat_map(|round| {
            (0..missing.len())
                .filter(|&point| round < missing[point])
                .collect::<Vec<_>>()
        })
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
        warm(&points[0])?;
        sample(&mut points, options).map_err(|failure| failure.error)?;
        Ok(points[0].measurement())
    }
}
