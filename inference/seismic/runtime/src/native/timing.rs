//! Device timing of a prepared native implementation at one or more
//! workloads ("points"), each cycling through its own argument sets.
//!
//! A point's calls are validated and placed once; every sample resubmits
//! them. Timing is device time (command-buffer timestamps on Metal, stream
//! events on CUDA), so host latency never enters a sample. Each sample
//! completes before the next placement's is submitted: a Metal queue runs
//! command buffers without a hazard between them concurrently, so samples
//! of different points submitted back to back overlap and each one's
//! interval includes the other's work (measured on an M4 Pro, 2026-09-24: a
//! decode GEMV's samples read 185 µs next to a 4-row point's and 67 µs
//! alone). Submissions of one placement run in turn, and those are queued.
//!
//! # What a sample is
//!
//! A sample is one submission of a placement's passes, queued behind one
//! pass of the same placement that is not timed: its *lead*. What a
//! placement's work costs when the device comes to it from something else
//! is not what it costs in service, where the same work follows itself: the
//! first pass after other work read 11% to 18% long on an M4 Pro and on a
//! GB10 (a 32-row attention prefill beside its 512-row point, a one-row
//! GEMV beside its 512-row point: 2026-10-10), and the first after the host
//! held the device for 20 ms to 80 ms (a candidate's numerical witnesses)
//! read 14% to 64% long on the M4 Pro, 15% to 35% on the GB10; the second
//! pass read as every later one. The lead pays that, the device goes from
//! it into the sample without a pause, and a sample then reads the same
//! whatever ran before it.
//!
//! A placement's first sample of a session calibrates it: the pass behind
//! its lead gives the repetitions for which a sample covers at least
//! `min_sample_seconds` of device time, and is itself the sample when that
//! is one pass. The lead also pays the first-use device costs of what a
//! placement is the first to run (launches no earlier point of the kernel
//! ran, scratch no earlier call touched): a pass that kept those read a
//! token-packing GEMM at 1.6× its time at 512 rows on an M4 Pro, and the
//! faster configuration lost its confirmation to its defaults.
//!
//! `min_sample_seconds` is not a matter of noise, which more rounds would
//! answer: every submission carries a fixed device cost besides its
//! kernel's, and a sample shorter than that cost measures the submission.
//! Both sides of a comparison would carry it, and their ratio would read
//! nearer one than the kernels are, in every round alike. Repetitions make
//! the fixed cost a small part of the sample; no number of rounds does.
//!
//! How many samples a session takes is its [`Rounds`]' decision, round by
//! round: a comparison's rounds end when it is resolved
//! ([`super::sampling`]), a measurement of a point's own time takes the
//! count it was asked for ([`Counted`]).
//!
//! # The state of the device
//!
//! A device's speed is not a property of the work alone. An Apple GPU runs
//! at a performance state its governor moves by how busy the device is, and
//! the states are far apart: on an M4 Pro (2026-10-10) the lowest runs a
//! kernel at 3.7× to 4.7× the time of the highest, with steps near 2.5×,
//! 1.6× and 1.3× between, and not every kernel by the same factor (a
//! two-row GEMV 3.7×, its 512-row point 4.7×): the ratio of two
//! configurations is itself a property of the state, so sampling both in
//! one round does not make a reading at a lower state the reading serving
//! would see. The device falls to the lowest after 100 ms without work (not
//! after 50 ms), and climbs back in 33 ms to 47 ms of work that leaves it
//! no pause. Between those, the governor holds: samples of 200 µs to 1 ms,
//! each complete before the next is submitted, keep the device busy for 60%
//! to 88% of the time, and at that it neither climbs (300 ms of them after
//! an idle read 4.4× throughout) nor falls (the same samples after
//! sustained work read 1.0× throughout). A session of such samples
//! therefore runs at whatever state the load's history left, and a
//! transition can fall anywhere in it: two readings of one configuration
//! differed by 5% at the median and 35% at the 90th percentile within one
//! load, and the defaults' own time at 8 of 105 points by more than 30%
//! between two loads. A GB10 has no such states: its samples read the same
//! at every duty and after every idle.
//!
//! So a session reads at one state, the device's *sustained clock*: the one
//! it reaches under work without pause, which is also the one serving runs
//! at. The state is known by a gauge: work whose reading at the sustained
//! clock has been taken, which is the work of the unit whose sessions it
//! raises the device for and no other's ([`SustainedClock`], [`Clock`]): a
//! unit's first session establishes it, with the unit's default
//! configuration at the first point it times. A session first
//! *raises* the device: gauge readings, each queued behind the one before
//! it so the device is never without work, until one reads no slower than
//! the gauge's reference takes a sustained clock to read ([`Reference`]).
//! The first reading of a raise follows other work and reads long (6% at
//! the median), so a raise at the sustained clock takes about two readings;
//! below it none passes until the device has climbed, the nearest state
//! being 1.07× away. The session's samples follow at once, each behind its
//! lead, and the pause between two of them (the host's turn, 130 µs on the
//! M4 Pro) is one the governor holds through.
//!
//! There are two kinds of evidence that a device is at its sustained clock,
//! and no third. One is saturation for as long as it takes to get there
//! from any state: the lowest state holds for 7 ms to 25 ms of saturating
//! work before it moves, within that it is as steady as the highest, no
//! reading tells them apart, and how long the hold is is a property of the
//! governor the device does not report. The first reference is therefore
//! taken by saturating over doubling lengths ([`establish`]), as many of
//! them as a comparison needs rounds to resolve, and then until the device
//! stops getting faster. What that took is what a raise is given (its
//! *allowance*). The other is a reading no slower than a sustained clock is
//! known to read, which ends a raise at once and is only as good as what is
//! known.
//!
//! What is known is not one number: the gauge's level differs between
//! saturations by more than its readings spread within one. Against the
//! median of seven consecutive readings 2% of a raise's readings passed on
//! an M4 Pro, where the device was at its sustained clock in 78% and 81% of
//! the raises of two loads; on a GB10 consecutive readings repeat to the
//! 32 ns step of its timer, a reference of one saturation has no width at
//! all, and a later saturation a few steps above it yields no passing
//! reading however long it runs (2026-10-10: 21 raises of one load ran out
//! their allowance on a device that was at its clock throughout). So a
//! reference is every reading taken with the device known to be at a
//! sustained clock, in every saturation of the load, and it only grows. A
//! level slower than every one of them cannot be told from a state below
//! the clock by reading it, so such a saturation costs one allowance; it
//! is then at a sustained clock by the first kind of evidence, its level
//! joins the reference, and no level costs an allowance twice.

use super::deadline::{Deadline, WorkError};
use super::sampling::Shuffle;
use super::{
    CallError, MeasureOptions, Measurement, NativeBoundCall, NativePrepared, NativeSubmission,
    StandaloneCalls, median, median_of,
};
use crate::api::kernel::{EncodedArgs, EncodedOutputs};
use crate::api::tensor::TensorInner;
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// What one class of operation has taken in this load: its wall times, in
/// seconds, or what it added to the bytes held, in bytes. A class is priced
/// only from its own observations: their mean. A class never observed has
/// no price.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Observed {
    seconds: f64,
    count: usize,
    longest: f64,
}

impl Observed {
    pub fn observe(&mut self, seconds: f64) {
        self.seconds += seconds;
        self.count += 1;
        self.longest = self.longest.max(seconds);
    }

    /// The class's price: the mean of its observations.
    pub fn price(&self) -> Option<f64> {
        (self.count > 0).then(|| self.seconds / self.count as f64)
    }

    /// The longest single operation of the class.
    pub fn longest(&self) -> f64 {
        self.longest
    }

    /// Everything the class's operations have taken.
    pub fn total(&self) -> f64 {
        self.seconds
    }
}

/// What calls of one kind have taken in this load, each with how many units
/// of work it did (a compile call and the entries it formed): a cost of the
/// call itself and a cost of each unit, the least-squares line through the
/// observations. A call of the toolchain pays for its start, its parse of
/// the source and its link whatever it forms, and for each entry's code
/// besides: one mean over calls of different sizes prices a small call as a
/// large one (NVRTC, 2026-10-10: compiles of 3 to 48 entries took 0.3 to
/// 4.3 s in one load).
///
/// Until calls of two sizes are observed the line cannot be told from a
/// constant, and the price is the mean; neither term is ever negative.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Linear {
    count: f64,
    units: f64,
    seconds: f64,
    units_squared: f64,
    units_seconds: f64,
    longest: f64,
}

impl Linear {
    pub fn observe(&mut self, units: usize, seconds: f64) {
        let units = units as f64;
        self.count += 1.;
        self.units += units;
        self.seconds += seconds;
        self.units_squared += units * units;
        self.units_seconds += units * seconds;
        self.longest = self.longest.max(seconds);
    }

    /// The cost of a call and of each unit; none before an observation.
    fn terms(&self) -> Option<(f64, f64)> {
        if self.count == 0. {
            return None;
        }
        let mean = self.seconds / self.count;
        let spread = self.count * self.units_squared - self.units * self.units;
        // Sizes that differ by less than rounding are one size.
        if spread <= self.units_squared * f64::EPSILON * self.count {
            return Some((mean, 0.));
        }
        let each = (self.count * self.units_seconds - self.units * self.seconds) / spread;
        let call = (self.seconds - each * self.units) / self.count;
        Some(if each < 0. {
            (mean, 0.)
        } else if call < 0. {
            (0., self.units_seconds / self.units_squared)
        } else {
            (call, each)
        })
    }

    /// The price of a call that does `units` units.
    pub fn price(&self, units: usize) -> Option<f64> {
        self.terms().map(|(call, each)| call + each * units as f64)
    }

    /// The cost of each unit, once calls of two sizes were observed.
    pub fn each(&self) -> Option<f64> {
        self.terms()
            .map(|(_, each)| each)
            .filter(|each| *each > 0.)
    }

    /// The mean call, and the mean of its units.
    pub fn mean(&self) -> Option<(f64, f64)> {
        (self.count > 0.).then(|| (self.seconds / self.count, self.units / self.count))
    }

    pub fn longest(&self) -> f64 {
        self.longest
    }

    pub fn total(&self) -> f64 {
        self.seconds
    }
}

/// The bytes a whole piece of work adds, at its most, to what the load
/// holds when it starts: priced like its seconds ([`Quote`]), as the sum of
/// its operations' class prices and how many of its operations belong to a
/// class this load has not observed. A piece that needs nothing more than
/// is held already has a footprint of nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Footprint {
    pub priced: u64,
    pub unpriced: usize,
}

impl Footprint {
    /// Add `times` operations of a class priced `price` bytes.
    pub fn add(&mut self, price: Option<f64>, times: usize) {
        match price {
            Some(bytes) => {
                self.priced = self
                    .priced
                    .saturating_add((bytes * times as f64).ceil() as u64)
            }
            None => self.unpriced += times,
        }
    }

    pub fn and(mut self, other: Footprint) -> Footprint {
        self.priced = self.priced.saturating_add(other.priced);
        self.unpriced += other.unpriced;
        self
    }

    /// What the work is taken to add: its priced bytes, with each unpriced
    /// operation at `largest`, the most a single piece of work of this load
    /// has added.
    pub fn forecast(&self, largest: u64) -> u64 {
        u64::try_from(self.unpriced)
            .ok()
            .and_then(|unpriced| largest.checked_mul(unpriced))
            .map_or(u64::MAX, |unpriced| self.priced.saturating_add(unpriced))
    }

    /// The admission rule for bytes, the one for seconds ([`Quote::fits`])
    /// over again. Work starts only when what it is taken to add fits
    /// `free`, the bytes its device's memory plan leaves for it. A device
    /// without a plan (`None`) bounds nothing: what is known of it is that
    /// no limit is enforced, and no number is put in its place.
    pub fn fits(&self, free: Option<u64>, largest: u64) -> bool {
        free.is_none_or(|free| self.forecast(largest) <= free)
    }
}

/// The operation classes of one point, each observed as wall time: placing
/// a kernel's calls there and restoring its state, one observed invocation
/// of one argument set (a numerical witness), one timed submission (a
/// calibrating pass or a sample), and warming the device with its calls.
/// Device medians remain the performance objective, never a price.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct PointPrices {
    pub placement: Observed,
    pub witness: Observed,
    pub sample: Observed,
    pub warm: Observed,
}

impl PointPrices {
    pub fn longest(&self) -> f64 {
        [self.placement, self.witness, self.sample, self.warm]
            .iter()
            .map(Observed::longest)
            .fold(0., f64::max)
    }
}

/// The price of a whole piece of work before it starts: the sum of its
/// operations' class prices, and how many of its operations belong to a
/// class this load has not observed.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Quote {
    pub priced: Duration,
    pub unpriced: usize,
}

impl Quote {
    /// Add `times` operations of one class, priced `price`. A class with
    /// no price is one thing not known, however many of its operations the
    /// work has: what is not known is what an operation of the class takes,
    /// once, and an unknown for each operation held a piece of seven
    /// challengers to seven times the longest operation of any kind the
    /// load had seen (a 4.3 s compile) and shut units that had not started
    /// out of the last 23 s of a budget.
    pub fn add(&mut self, price: Option<f64>, times: usize) {
        match price {
            Some(seconds) => self.priced += Duration::from_secs_f64(seconds * times as f64),
            None => self.unpriced += usize::from(times > 0),
        }
    }

    /// Add operations of a class priced `price` whose number is not known
    /// before they run, at the number `expected` of them.
    pub fn expect(&mut self, price: Option<f64>, expected: f64) {
        match price {
            Some(seconds) => self.priced += Duration::from_secs_f64(seconds * expected),
            None => self.unpriced += usize::from(expected > 0.),
        }
    }

    pub fn and(mut self, other: Quote) -> Quote {
        self.priced += other.priced;
        self.unpriced += other.unpriced;
        self
    }

    /// The forecast of the work's wall time; none while any of it is
    /// unpriced.
    pub fn forecast(&self) -> Option<Duration> {
        (self.unpriced == 0).then_some(self.priced)
    }

    /// The one admission rule. Work starts only when its priced operations
    /// fit the remaining budget; an unpriced operation is late, and its
    /// work is not started, once what remains after the priced operations
    /// is less than `longest`, the longest single operation this load has
    /// completed, for each unpriced operation: the load then holds direct
    /// evidence that an operation can outlast what is left, and none that
    /// this one will not.
    pub fn fits(&self, remaining: Duration, longest: Duration) -> bool {
        u32::try_from(self.unpriced)
            .ok()
            .and_then(|unpriced| longest.checked_mul(unpriced))
            .and_then(|unpriced| self.priced.checked_add(unpriced))
            .is_some_and(|needed| needed <= remaining)
    }
}

/// One point's calls, placed once and submitted for every sample.
pub(crate) struct PointTiming<'a> {
    kernel: Arc<NativePrepared>,
    calls: Vec<NativeBoundCall>,
    arguments: Vec<EncodedArgs>,
    pub(super) initialize: Option<super::validation::Initializer<'a>>,
    /// Passes over the rotation per sample; `None` until calibrated.
    repetitions: Option<usize>,
    pass_seconds: Option<f64>,
    rotation_bytes: u64,
    samples: Vec<f64>,
    placement_seconds: f64,
    prices: Rc<Cell<PointPrices>>,
}

/// Result storage of each point's argument sets, shared by every placement at
/// that point in one tuning run. Samples complete one at a time, so the
/// configurations placed at a point can write the same results. Numerical
/// observation clears that storage before each candidate's first invocation.
/// Points are named by label, so any subset of a run's points finds its own
/// storage.
#[derive(Default)]
pub(crate) struct OutputPool(HashMap<String, Vec<Vec<Arc<TensorInner>>>>);

impl OutputPool {
    /// The storage of the argument sets of the point labeled `point`.
    pub(crate) fn at(&mut self, point: &str) -> &mut Vec<Vec<Arc<TensorInner>>> {
        self.0.entry(point.to_owned()).or_default()
    }
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

impl<'a> PointTiming<'a> {
    /// Validate and place every call of `rotation`. Results of each argument
    /// set are allocated once and reused by every sample; scratch is the
    /// standalone arena.
    pub(crate) fn new(
        kernel: &Arc<NativePrepared>,
        rotation: Vec<EncodedArgs>,
    ) -> Result<Self, CallError> {
        Self::place(kernel, rotation, None)
    }

    /// Place a configuration against a point's shared result storage
    /// ([`OutputPool`]).
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
        let began = Instant::now();
        if rotation.is_empty() {
            return Err(CallError::Workflow(crate::api::WorkflowError::Empty));
        }
        let arguments = rotation.clone();
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
            arguments,
            initialize: None,
            repetitions: None,
            pass_seconds: None,
            rotation_bytes: distinct.into_iter().map(|(_, bytes)| bytes).sum(),
            samples: Vec::new(),
            placement_seconds: began.elapsed().as_secs_f64(),
            prices: Rc::new(Cell::new(PointPrices::default())),
        })
    }

    /// Observe this placement's operations in its point's classes.
    pub(super) fn track(&mut self, prices: Rc<Cell<PointPrices>>) {
        self.prices = prices;
    }

    /// The placement is complete: placing its calls and `restoring` its
    /// point's state are one observation of the placement class.
    pub(super) fn placed(&self, restoring: Duration) {
        self.observe(|prices| {
            prices
                .placement
                .observe(self.placement_seconds + restoring.as_secs_f64())
        });
    }

    fn observe(&self, update: impl FnOnce(&mut PointPrices)) {
        let mut prices = self.prices.get();
        update(&mut prices);
        self.prices.set(prices);
    }

    /// Declaration ordinals of the launches that do work at this point: active
    /// (`when`) with a nonempty grid in some call of the rotation.
    pub(crate) fn working_launches(&self) -> Vec<usize> {
        let mut working = self
            .calls
            .iter()
            .flat_map(|call| {
                call.launches
                    .geometry
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

    /// Submit `passes` passes over the rotation as one unit of device work.
    /// Writable state is not restored between timed passes: every
    /// configuration is timed the same way, and only the first, validated
    /// invocation ([`PointTiming::observe_first`]) needs pristine state.
    fn submit_passes(
        &self,
        passes: usize,
        deadline: Deadline,
    ) -> Result<PendingSample, WorkError<CallError>> {
        deadline.check(Duration::from_secs_f64(
            self.pass_seconds.unwrap_or(0.) * passes as f64,
        ))?;
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

    pub(super) fn prepared(&self) -> &Arc<NativePrepared> {
        &self.kernel
    }

    /// Run every call of the rotation once from pristine state and hand
    /// what it published to `observe`: the placement's numerical
    /// observations, before any timed pass or subsequent candidate can
    /// overwrite the result and state buffers. Nothing here is timed: each
    /// call runs alone with the host before and after it, which is not how
    /// a sample runs (module documentation).
    pub(super) fn observe_first(
        &mut self,
        deadline: Deadline,
        mut observe: impl FnMut(
            usize,
            Vec<crate::api::kernel::DecodedValue>,
            &EncodedArgs,
        ) -> Result<(), super::tune::Exclusion>,
    ) -> Result<(), WorkError<super::tune::Exclusion>> {
        for (rotation, call) in self.calls.iter().enumerate() {
            deadline.check(Duration::ZERO)?;
            let witness = Instant::now();
            if let Some(initialize) = &self.initialize {
                initialize.borrow_mut()()
                    .map_err(|e| super::tune::Exclusion::Execution(e.to_string()))?;
            }
            // Match fresh-call zeroed result storage. A pooled output must not
            // let a missing write inherit the preceding candidate's correct bytes.
            // Initialization remains outside the submitted device interval.
            for result in &call.results {
                let bytes = usize::try_from(result.byte_len()).map_err(|_| {
                    super::tune::Exclusion::Execution("result bytes exceed usize".into())
                })?;
                result
                    .write_from_host(&vec![0; bytes])
                    .map_err(|error| super::tune::Exclusion::Execution(error.to_string()))?;
            }
            self.kernel
                .reset_scalar_results()
                .map_err(|e| super::tune::Exclusion::Execution(e.to_string()))?;
            deadline.check(Duration::ZERO)?;
            StandaloneCalls {
                kernel: &self.kernel,
                calls: std::slice::from_ref(call),
            }
            .submit(1)
            .and_then(|submission| submission.wait())
            .map_err(|e| super::tune::Exclusion::Execution(e.to_string()))?;
            let actual = self
                .kernel
                .read_call_results(call)
                .map_err(|e| super::tune::Exclusion::Execution(e.to_string()))?;
            let observed = observe(rotation, actual.into_values(), &self.arguments[rotation]);
            // Restoring state, the call, its readback and its observation
            // are one witness.
            self.observe(|prices| prices.witness.observe(witness.elapsed().as_secs_f64()));
            observed?;
        }
        Ok(())
    }

    /// Calibrate the placement without sampling it: one pass behind its
    /// lead gives the repetitions for which a sample covers at least
    /// `min_sample_seconds` of device time.
    ///
    /// The two sides of a comparison are sampled with the same repetitions
    /// ([`sample`]). Every submission carries a fixed device cost besides
    /// its kernel's, which is a larger part of a sample of fewer
    /// repetitions, so sides calibrated apart read apart by more than
    /// their kernels differ, for a whole session and by a different amount
    /// in the next: where a session gave the served choice fewer
    /// repetitions relative to a challenger's than another session did, the
    /// same challenger read 8% to 11% faster on an M4 Pro and 4% faster on
    /// a GB10 (prefill units, two complete recordings of each, 2026-10-10),
    /// and where the repetitions agreed the two sessions agreed to 1%.
    fn calibrate(
        &mut self,
        min_sample_seconds: f64,
        deadline: Deadline,
    ) -> Result<usize, WorkError<CallError>> {
        let lead = self.submit_passes(1, deadline)?;
        let measured = self
            .submit_passes(1, deadline)
            .and_then(|sample| Ok(sample.seconds()? * self.calls.len() as f64));
        let drained = lead.submission.wait();
        let seconds = measured?;
        drained?;
        self.pass_seconds = Some(seconds);
        let repetitions = if seconds > 0.0 {
            ((min_sample_seconds / seconds).ceil() as usize).max(1)
        } else {
            1
        };
        self.repetitions = Some(repetitions);
        Ok(repetitions)
    }

    /// Take one sample: its device seconds per call. The sample is queued
    /// behind its lead, one pass that is not timed (module documentation).
    /// An uncalibrated placement's sample is first one pass, which gives
    /// its repetitions, and is the sample itself when they are one;
    /// otherwise the sample follows it at once.
    fn sample(
        &mut self,
        min_sample_seconds: f64,
        deadline: Deadline,
    ) -> Result<f64, WorkError<CallError>> {
        let lead = self.submit_passes(1, deadline)?;
        let measured = (|| -> Result<f64, WorkError<CallError>> {
            if self.pass_seconds.is_none() {
                // A challenger can inherit the incumbent's repetitions,
                // but not its latency. Price the already submitted lead
                // before admitting that first repeated batch.
                self.pass_seconds = Some(lead.submission.device_seconds()?);
            }
            let repetitions = match self.repetitions {
                Some(repetitions) => repetitions,
                None => {
                    let pass = self.submit_passes(1, deadline)?.seconds()?;
                    let seconds = pass * self.calls.len() as f64;
                    self.pass_seconds = Some(seconds);
                    let repetitions = if seconds > 0.0 {
                        ((min_sample_seconds / seconds).ceil() as usize).max(1)
                    } else {
                        1
                    };
                    self.repetitions = Some(repetitions);
                    if repetitions == 1 {
                        return Ok(pass);
                    }
                    // The calibrating pass just ran: it leads the sample.
                    repetitions
                }
            };
            let seconds = self.submit_passes(repetitions, deadline)?.seconds()?;
            // PendingSample already divides by repetitions and call count.
            // Multiplying by the rotation length recovers one pass.
            self.pass_seconds = Some(seconds * self.calls.len() as f64);
            Ok(seconds)
        })();
        let drained = lead.submission.wait();
        let seconds = measured?;
        drained?;
        Ok(seconds)
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
    pub(crate) error: WorkError<CallError>,
}

/// What a sampling session takes of its slots (the placed points it was
/// given, by position), decided round by round.
pub(crate) trait Rounds {
    /// The slots the next round samples, in groups; none ends the session.
    /// The slots of a group are compared with one another (a challenger
    /// and the choice it is sampled beside), and are sampled one after
    /// another: what disturbs a device for a moment (another process's
    /// work, the host's memory traffic: on a GB10 one sample in ten read
    /// 15% to 35% long, several in a row) then falls on both sides of a
    /// comparison or on neither.
    fn next(&mut self) -> Vec<Vec<usize>>;
    /// A round's samples: each slot sampled, with its device seconds per
    /// call.
    fn sampled(&mut self, round: &[(usize, f64)]);
}

/// Every slot sampled a given number of times: a measurement of each
/// point's own time, for a caller that named the count.
pub(crate) struct Counted {
    slots: Vec<usize>,
    rounds: usize,
}

impl Counted {
    /// `samples` of each of the first `slots` slots.
    pub(crate) fn new(slots: usize, samples: usize) -> Self {
        Self {
            slots: (0..slots).collect(),
            rounds: samples,
        }
    }
}

impl Rounds for Counted {
    fn next(&mut self) -> Vec<Vec<usize>> {
        if self.rounds == 0 {
            Vec::new()
        } else {
            self.slots.iter().map(|slot| vec![*slot]).collect()
        }
    }

    fn sampled(&mut self, _: &[(usize, f64)]) {
        self.rounds -= 1;
    }
}

/// Take and keep one sample of each slot of `order` in turn, each
/// completing before the next slot's is submitted: each slot with its
/// device seconds per call.
fn collect(
    points: &mut [PointTiming],
    order: &[usize],
    min_sample_seconds: f64,
    deadline: Deadline,
) -> Result<Vec<(usize, f64)>, PointFailure> {
    let mut round = Vec::with_capacity(order.len());
    for &point in order {
        let began = Instant::now();
        let seconds = points[point]
            .sample(min_sample_seconds, deadline)
            .map_err(|error| PointFailure { point, error })?;
        points[point].observe(|prices| prices.sample.observe(began.elapsed().as_secs_f64()));
        points[point].samples.push(seconds);
        round.push((point, seconds));
    }
    Ok(round)
}

/// Readings of a gauge under saturation: each is queued while the one
/// before it runs, so the device has work from the first reading to the
/// last and beyond it (the reading queued behind the last one read is still
/// to run when its reader goes on).
struct Saturation<'g> {
    kernel: &'g Arc<NativePrepared>,
    calls: &'g [NativeBoundCall],
    ahead: Option<(usize, NativeSubmission)>,
    deadline: Deadline,
}

impl Saturation<'_> {
    fn submit(&self, passes: usize) -> Result<NativeSubmission, WorkError<CallError>> {
        self.deadline.check(Duration::ZERO)?;
        Ok(StandaloneCalls {
            kernel: self.kernel,
            calls: self.calls,
        }
        .submit(passes)?)
    }

    /// The device seconds of the next reading of `passes` passes.
    fn read(&mut self, passes: usize) -> Result<f64, WorkError<CallError>> {
        self.deadline.check(Duration::ZERO)?;
        let reading = match self.ahead.take() {
            Some((queued, reading)) if queued == passes => reading,
            // A reading of another length queued ahead runs as work alone.
            _ => self.submit(passes)?,
        };
        let queued = self.submit(passes);
        match queued {
            Ok(queued) => {
                self.ahead = Some((passes, queued));
                Ok(reading.device_seconds()?)
            }
            Err(error) => {
                // Expiry between readings still drains the admitted one.
                reading.wait()?;
                Err(error)
            }
        }
    }

    /// The reading still queued completes: a placement's work runs beside
    /// the gauge's on a Metal queue, not behind it, so nothing is sampled
    /// while a reading is in flight. What follows is the host's turn, a
    /// pause the device holds its clock through.
    ///
    /// That reading is one of the device at the clock its reader left it
    /// at, with no pause since: its passes and its device seconds.
    fn finish(self) -> Result<Option<(usize, f64)>, CallError> {
        match self.ahead {
            Some((passes, queued)) => Ok(Some((passes, queued.device_seconds()?))),
            None => Ok(None),
        }
    }
}

/// The level of a saturation whose device is at a sustained clock: readings
/// taken now, until one adds nothing to their range (the fewest that say
/// how far the gauge's readings spread within the saturation), in ascending
/// order. Their median is the level: within one saturation a disturbed
/// reading is slower and never faster, and the median is not moved by one.
fn settled<E>(mut read: impl FnMut() -> Result<f64, E>) -> Result<Vec<f64>, E> {
    let mut readings: Vec<f64> = Vec::new();
    loop {
        let reading = read()?;
        let widens = readings.first().is_none_or(|fastest| reading < *fastest)
            || readings.last().is_none_or(|slowest| reading > *slowest);
        let at = readings.partition_point(|taken| *taken <= reading);
        readings.insert(at, reading);
        if !widens {
            return Ok(readings);
        }
    }
}

/// What a gauge reads with its device at a sustained clock, from every
/// saturation of the load (module documentation).
///
/// Two things are kept, because slowness is evidence of two kinds. A single
/// reading that is slow may only have been disturbed, so single readings
/// say how slow a sustained clock reads on their fast side alone: as far
/// above their median as the fastest lies below it. A level a whole
/// saturation held ([`settled`]) is slow because the device was, and is
/// believed as it stands.
///
/// Nothing here is ever taken out. An established level is one the device
/// sustained under saturation for as long as an establishment takes, so it
/// is a sustained clock whatever the device reads later: throttled then,
/// or sharing the device, and faster since. What a raise tells apart is a
/// device still climbing from one at a sustained level, and the candidates
/// of a comparison are sampled against their default inside the same
/// rounds, so a bound left looser by a slow episode biases no comparison,
/// while forgetting the level would buy one more allowance to learn it
/// again and nothing else.
#[derive(Clone, Debug)]
struct Reference {
    /// Every reading taken with the device known to be at a sustained
    /// clock, in ascending order: an establishment's, the reading queued
    /// behind the one that ended each raise ([`Saturation::finish`]), and
    /// those an expired raise settled on.
    readings: Vec<f64>,
    /// The level of each saturation that established one, in the order
    /// they were established: the establishment's, then one for each raise
    /// that ran out its allowance.
    levels: Vec<f64>,
}

impl Reference {
    /// A reference of one established saturation, from its `readings`.
    fn established(readings: Vec<f64>) -> Self {
        let mut reference = Self {
            readings: Vec::new(),
            levels: Vec::new(),
        };
        reference.establish(readings);
        reference
    }

    /// One more saturation that held a sustained clock for as long as an
    /// establishment takes: its `readings`. Its level is returned.
    fn establish(&mut self, readings: Vec<f64>) -> f64 {
        let level = median_of(readings.clone());
        for reading in readings {
            self.observe(reading);
        }
        self.levels.push(level);
        level
    }

    /// One more reading of the device at a sustained clock.
    fn observe(&mut self, reading: f64) {
        let at = self.readings.partition_point(|taken| *taken <= reading);
        self.readings.insert(at, reading);
    }

    fn median(&self) -> f64 {
        median_of(self.readings.clone())
    }

    /// The slowest a reading of the device at a sustained clock is taken
    /// to be: what a raise's reading is held to. The slower of the slowest
    /// established level and of what the single readings say on their fast
    /// side, their median as far again as their fastest lies below it.
    fn bound(&self) -> f64 {
        let median = self.median();
        self.levels
            .iter()
            .copied()
            .fold(median + (median - self.readings[0]), f64::max)
    }
}

/// How a raise ended, or that the record is of an establishment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RaiseOutcome {
    /// A gauge was placed and its first level established.
    Established,
    /// A reading was no slower than the bound.
    Passed,
    /// The raise ran out its allowance and its level was established.
    Expired,
}

/// What one raise of a device did, or one establishment of its gauge.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct Raise {
    pub outcome: RaiseOutcome,
    /// Gauge readings taken, the level's among them.
    pub readings: usize,
    /// Device seconds of a reading: the one that passed, or the level the
    /// saturation established.
    pub level: f64,
    /// What a reading was held to; none for an establishment.
    pub bound: Option<f64>,
    /// Device seconds of the first reading taken.
    pub first: f64,
    /// Host seconds taken.
    pub seconds: f64,
    /// The first sample of the session that followed a raise, as its
    /// point's record states it (device seconds per call): which samples
    /// were taken behind this raise.
    pub sample: Option<f64>,
}

/// The passes of a gauge's reading and its reference, for a gauge whose
/// device is in a state not known.
///
/// The device is saturated in stages, each two readings of one length. The
/// first stage's readings take `seconds` of device time and each stage's
/// twice the one before, until `stages` stages are taken; from then on
/// stages of the last length are taken until one's second reading is not
/// faster than its first. A reading's passes are those that take its length
/// at the speed the reading before it showed, so the lengths are of time,
/// and do not shrink as the device speeds up. Readings of `seconds` follow
/// until one is not faster than the one before it either: a device that
/// leaves a state as the last stage ends is still getting faster while a
/// level would be read (GB10, 2026-10-10: a stage of 12.8 ms read a pass at
/// over 200 µs, the six readings of the level behind it fell to a median
/// of 344 µs, and the reading after them took 24 µs; that level bounded
/// nothing for the rest of the load). The saturation's level is of
/// readings of `seconds` taken then ([`settled`]): those readings are
/// returned with their passes and with every reading the establishment
/// took.
///
/// A device below its sustained clock climbs under saturation, but not at
/// once: its lowest state holds first, and reads as steadily as the highest
/// (module documentation). Two readings that agree therefore show the
/// sustained clock only when they are long against that hold, and how long
/// it is the device does not say. Doubling the length makes the question
/// one of scale and not of a duration: over `stages` stages the readings
/// span a range of 2^(`stages` − 1) in length, and the claim given up is
/// "the device is still getting faster", of which each stage is one round.
/// `stages` is the number of rounds a comparison of this load needs to
/// resolve ([`super::sampling::Tally::fewest`]): the same evidence is asked
/// of the device's state as of a challenger's lead. The length does not
/// double further: a stage at the sustained clock is flat half the time,
/// and stages that doubled until one was would have no expected end.
///
/// The states between the lowest and the highest hold too, each for a few
/// milliseconds of saturation, and a stage that ended inside one would
/// take it for the sustained clock. Nothing read later takes that level
/// out again ([`Reference`]): the count of stages is what stands against
/// it.
fn establish<E>(
    mut read: impl FnMut(usize) -> Result<f64, E>,
    seconds: f64,
    stages: u32,
) -> Result<(usize, Vec<f64>, usize), E> {
    let mut taken = 0;
    let mut read = |passes: usize| {
        taken += 1;
        read(passes)
    };
    // Seconds per pass, as the last reading showed them.
    let mut pass = read(1)?;
    let mut reading = |length: f64, pass: &mut f64| -> Result<f64, E> {
        let passes = if *pass > 0. {
            ((length / *pass).ceil() as usize).max(1)
        } else {
            1
        };
        *pass = read(passes)? / passes as f64;
        Ok(*pass)
    };
    let mut length = seconds;
    let mut stage = 1;
    loop {
        let (first, second) = (reading(length, &mut pass)?, reading(length, &mut pass)?);
        if stage < stages {
            stage += 1;
            length *= 2.;
        } else if second >= first {
            break;
        }
    }
    // The same question at the length of a reading, before a level is
    // taken at it: readings of `seconds` until one is not faster than the
    // one before.
    let mut before = pass;
    loop {
        let after = reading(seconds, &mut pass)?;
        if after >= before {
            break;
        }
        before = after;
    }
    let passes = if pass > 0. {
        ((seconds / pass).ceil() as usize).max(1)
    } else {
        1
    };
    let readings = settled(|| read(passes))?;
    Ok((passes, readings, taken))
}

/// Raise a device to a sustained clock: readings of the gauge under
/// saturation until one is no slower than `reference` takes a sustained
/// clock to read ([`Reference::bound`]), or until `expired` says the raise
/// has saturated the device for as long as an establishment takes. Either
/// way the device is at a sustained clock when this returns, and the raise
/// took `readings` readings.
///
/// A raise that expires has found the work slower than every level the
/// reference knows through an allowance of saturation. It is not yet a
/// level: work that is slow as a session begins may still be coming down
/// when the allowance ends (GB10, 2026-10-10: a projection's own passes read
/// 605 µs at the start of a session and 222 µs, their level, 0.12 s to
/// 0.24 s of saturation later, with an allowance of 0.1 s; levels taken
/// where the allowance ended, at 496 µs to 730 µs, left the unit's bound at
/// 2.9× its level). So the readings go on until one is not faster than the
/// one before, as an establishment's do, and what the saturation then
/// holds is an establishment by another name: its level is settled and
/// joins the reference ([`Reference::establish`]), and the same level
/// passes from then on. Every level in a reference is therefore one its
/// own readings showed to have stopped falling, which is why none is ever
/// taken out. No second establishment follows and nothing is replaced;
/// a reference that started over after each expiry would again know one
/// saturation only, and the next one above it would expire in turn.
///
/// The bound is not the reference's median. What a raise tells apart is
/// the sustained clock and the state below it, 1.07× and more away, and a
/// median has no width: where the gauge read 1.6% above its reference, 2%
/// of the readings of a device at its sustained clock were no slower than
/// the median, and the raises that found the device there took 22 readings
/// at the median and up to 655 (M4 Pro, 2026-10-10: 4.6 s of a 60 s load).
fn raise<E>(
    mut read: impl FnMut() -> Result<f64, E>,
    reference: &mut Reference,
    mut expired: impl FnMut() -> bool,
) -> Result<(RaiseOutcome, usize, f64, f64, f64), E> {
    let bound = reference.bound();
    let mut readings = 0;
    let mut first = None;
    loop {
        let reading = read()?;
        readings += 1;
        let first = *first.get_or_insert(reading);
        if reading <= bound {
            return Ok((RaiseOutcome::Passed, readings, reading, bound, first));
        }
        if expired() {
            // The allowance says when to ask, not what the answer is: the
            // readings go on while each is faster than the one before it,
            // as an establishment's do, and a level is taken only where
            // they have stopped coming down. One that comes down to the
            // bound on the way ends the raise as any other does.
            let mut before = reading;
            loop {
                let after = read()?;
                readings += 1;
                if after <= bound {
                    return Ok((RaiseOutcome::Passed, readings, after, bound, first));
                }
                if after >= before {
                    break;
                }
                before = after;
            }
            let settled = settled(&mut read)?;
            readings += settled.len();
            let level = reference.establish(settled);
            return Ok((RaiseOutcome::Expired, readings, level, bound, first));
        }
    }
}

/// What a clock knows once its work has been established.
struct Gauged {
    /// The passes of one reading.
    passes: usize,
    /// What a reading takes with the device at a sustained clock.
    reference: Reference,
    /// The host seconds the establishment took, and nothing else: what a
    /// raise is given. A raise does not change it.
    allowance: f64,
}

/// A device's sustained clock as one kind of work reads it (module
/// documentation): the levels that work has held with the device at a
/// sustained clock, and what its establishment took.
///
/// The work is its holder's own: what a level says of the device, it says
/// for the work it was read with and for nothing else. A gauge of other
/// work read twice its level and more for the sessions of four units of a
/// load whose own samples were what they were in every other load, and did
/// so whether it wrote its arguments or shared none of them with any unit
/// (GB10, 2026-10-10): a bound taken from it either ran out an allowance
/// for work that was at its level, or, once that level was established,
/// bounded nothing. So a unit of a search gauges with its own work
/// ([`Clock`]), and a comparison run by a caller with the work it compares.
pub struct SustainedClock {
    /// The device seconds of a reading: a sample's.
    seconds: f64,
    /// The stages of an establishment ([`establish`]).
    stages: u32,
    gauged: Option<Gauged>,
}

impl SustainedClock {
    /// A clock of work not yet established, whose readings cover `seconds`
    /// of device time and whose establishment takes `stages` stages: the
    /// rounds a comparison needs to resolve.
    pub fn new(seconds: f64, stages: u32) -> Self {
        Self {
            seconds,
            stages,
            gauged: None,
        }
    }

    /// Bring the device to a sustained clock for work that follows at
    /// once. `read` runs the clock's work the passes it is given, with
    /// nothing between them and the device never without work from one
    /// call to the next, and returns their device seconds.
    ///
    /// The first raise is the work's establishment ([`establish`]); every
    /// later one reads until a reading is no slower than the work's
    /// reference takes a sustained clock to read, or for as long as the
    /// establishment took ([`raise`]). What it did is returned.
    pub fn raise<E>(&mut self, mut read: impl FnMut(usize) -> Result<f64, E>) -> Result<Raise, E> {
        let began = Instant::now();
        let Some(gauged) = &mut self.gauged else {
            let (passes, settled, readings) = establish(&mut read, self.seconds, self.stages)?;
            let allowance = began.elapsed().as_secs_f64();
            let reference = Reference::established(settled);
            let level = reference.levels[0];
            self.gauged = Some(Gauged {
                passes,
                reference,
                allowance,
            });
            return Ok(Raise {
                outcome: RaiseOutcome::Established,
                readings,
                level,
                bound: None,
                first: level,
                seconds: allowance,
                sample: None,
            });
        };
        let (passes, allowance) = (gauged.passes, gauged.allowance);
        let (outcome, readings, level, bound, first) = raise(
            || read(passes),
            &mut gauged.reference,
            || began.elapsed().as_secs_f64() > allowance,
        )?;
        Ok(Raise {
            outcome,
            readings,
            level,
            bound: Some(bound),
            first,
            seconds: began.elapsed().as_secs_f64(),
            sample: None,
        })
    }

    /// One more reading of `passes` passes taken with the device at a
    /// sustained clock: it joins the reference when it is a reading's
    /// length.
    fn observe(&mut self, passes: usize, reading: f64) {
        if let Some(gauged) = &mut self.gauged {
            if gauged.passes == passes {
                gauged.reference.observe(reading);
            }
        }
    }
}

/// The sustained clock of one unit of a search: the unit's own work, placed
/// once, and what it reads ([`SustainedClock`]). The unit holds it from its
/// first session to its last.
pub(crate) struct Clock {
    kernel: Arc<NativePrepared>,
    calls: Vec<NativeBoundCall>,
    sustained: Mutex<SustainedClock>,
    /// What the clock's raises and its establishment did since they were
    /// last taken ([`Clock::raises`]).
    raises: Mutex<Vec<Raise>>,
}

impl Clock {
    /// What the clock's raises and its establishment did since this was
    /// last asked: the record of the piece of work that ran them.
    pub(crate) fn raises(&self) -> Vec<Raise> {
        std::mem::take(&mut *self.raises.lock().expect("raise record lock is never poisoned"))
    }

    /// `seconds` is a sample of the session that followed the clock's last
    /// raise: its first, when that raise has none yet ([`Raise::sample`]).
    fn followed(&self, seconds: f64) {
        let mut raises = self.raises.lock().expect("raise record lock is never poisoned");
        if let Some(raise) = raises.last_mut() {
            raise.sample.get_or_insert(seconds);
        }
    }

    /// The clock its holder keeps in `held` from its first session on: the
    /// one it holds, and otherwise one whose work is `point`'s, placed now
    /// with results of its own. A unit's first session is of its default
    /// configuration at the first point it discovers, so that is the work
    /// a unit gauges with, beside the unit's own resident inputs: a unit's
    /// points are declared in ascending cost and discovered in that order
    /// ([`super::tune::PointSpec::cost`]), so it is the lightest point the
    /// unit's census times, by declaration and not by a measurement.
    /// `stages` is the rounds a comparison needs to resolve.
    pub(crate) fn hold(
        held: &mut Option<Arc<Clock>>,
        point: &PointTiming,
        min_sample_seconds: f64,
        stages: u32,
    ) -> Result<Arc<Clock>, CallError> {
        if let Some(clock) = held {
            return Ok(clock.clone());
        }
        let clock = Arc::new(Clock {
            kernel: point.kernel.clone(),
            calls: PointTiming::new(&point.kernel, point.arguments.clone())?.calls,
            sustained: Mutex::new(SustainedClock::new(min_sample_seconds, stages)),
            raises: Mutex::new(Vec::new()),
        });
        *held = Some(clock.clone());
        Ok(clock)
    }

    /// Bring the device to a sustained clock for the session that follows
    /// at once, with the clock's work queued reading behind reading
    /// ([`SustainedClock::raise`]). What it did is recorded
    /// ([`Clock::raises`]).
    pub(crate) fn raise(&self, deadline: Deadline) -> Result<(), WorkError<CallError>> {
        let mut sustained = self.sustained.lock().expect("clock lock is never poisoned");
        let mut saturation = Saturation {
            kernel: &self.kernel,
            calls: &self.calls,
            ahead: None,
            deadline,
        };
        let raised = sustained.raise(|passes| saturation.read(passes));
        // The reading queued behind the last one read is of the device at
        // a sustained clock, whatever the readings before it were.
        if let Some((passes, seconds)) = saturation.finish()? {
            sustained.observe(passes, seconds);
        }
        let raise = raised?;
        self.raises
            .lock()
            .expect("raise record lock is never poisoned")
            .push(raise);
        Ok(())
    }
}

/// Raise `clock`'s device for a session over `points`, whose first slot's
/// point the raise is observed at (its `warm` class). The session's
/// [`sample`] follows at once.
pub(crate) fn warm(
    clock: &Clock,
    points: &[PointTiming],
    deadline: Deadline,
) -> Result<(), WorkError<CallError>> {
    let began = Instant::now();
    let raised = clock.raise(deadline);
    if let Some(first) = points.first() {
        first.observe(|prices| prices.warm.observe(began.elapsed().as_secs_f64()));
    }
    raised
}

/// Run a sampling session over `points`, on a device its caller has raised
/// with `clock` ([`warm`]; none for a measurement that holds no clock):
/// samples are taken round by round, each round the groups `rounds` names
/// in a random order, and each group's slots in a random order of their own
/// (`shuffle`, which its caller keeps from session to session: the sessions
/// of one comparison must not all open in the same order), until it names
/// none or the cutoff interrupts it. Only complete rounds reach the
/// comparison; a prefix of a matched round is never sampling evidence.
pub(crate) fn sample(
    points: &mut [PointTiming],
    min_sample_seconds: f64,
    rounds: &mut dyn Rounds,
    shuffle: &mut Shuffle,
    clock: Option<&Clock>,
    deadline: Deadline,
) -> Result<(), PointFailure> {
    loop {
        let mut groups = rounds.next();
        if groups.is_empty() {
            return Ok(());
        }
        // The slots of a group are sampled with the repetitions of its
        // first, the choice the others are compared with.
        for group in groups.iter().filter(|group| group.len() > 1) {
            let first = group[0];
            let repetitions = match points[first].repetitions {
                Some(repetitions) => repetitions,
                None => points[first]
                    .calibrate(min_sample_seconds, deadline)
                    .map_err(|error| PointFailure {
                        point: first,
                        error,
                    })?,
            };
            for &slot in &group[1..] {
                points[slot].repetitions.get_or_insert(repetitions);
            }
        }
        let mut turns = (0..groups.len()).collect::<Vec<_>>();
        shuffle.shuffle(&mut turns);
        let mut order = Vec::new();
        for turn in turns {
            shuffle.shuffle(&mut groups[turn]);
            order.append(&mut groups[turn]);
        }
        let round = collect(points, &order, min_sample_seconds, deadline)?;
        if let (Some(clock), Some(&(_, seconds))) = (clock, round.first()) {
            clock.followed(seconds);
        }
        rounds.sampled(&round);
    }
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
        // A measurement on its own is of no load and holds no clock: it
        // reads at the state its own samples leave the device in.
        sample(
            &mut points,
            options.min_sample_seconds,
            &mut Counted::new(1, options.samples),
            &mut Shuffle::default(),
            None,
            Deadline::default(),
        )
        .map_err(|failure| match failure.error {
            WorkError::Failed(error) => error,
            WorkError::Expired => unreachable!("a standalone measurement has no deadline"),
        })?;
        Ok(points[0].measurement())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    /// A synthetic device with a governed clock like the one probed on an
    /// M4 Pro: a pass takes `PASS` at the sustained clock; below it the
    /// device runs `LOWEST` times slower, holds there for `hold` seconds of
    /// saturating work, and climbs over `CLIMB` more through the steps
    /// probed. Every reading carries a right-skewed jitter of 1%, and the
    /// device is saturated from its first reading on, as a gauge's is.
    struct Device {
        /// Seconds of saturating work since the device was at its lowest.
        saturated: f64,
        hold: f64,
        /// How much slower the sustained clock itself has become.
        throttled: f64,
        elapsed: f64,
        readings: usize,
        jitter: u64,
    }

    const PASS: f64 = 100e-6;
    const LOWEST: f64 = 4.5;
    const CLIMB: f64 = 0.015;

    impl Device {
        fn lowest(hold: f64) -> Self {
            Self {
                saturated: 0.,
                hold,
                throttled: 1.,
                elapsed: 0.,
                readings: 0,
                jitter: 0x9e37_79b9_7f4a_7c15,
            }
        }

        fn sustained() -> Self {
            Self {
                saturated: 1.,
                ..Self::lowest(0.02)
            }
        }

        fn slowdown(&self) -> f64 {
            let climbed = (self.saturated - self.hold) / CLIMB;
            self.throttled
                * if climbed < 0. {
                    LOWEST
                } else if climbed < 1. / 3. {
                    2.5
                } else if climbed < 2. / 3. {
                    1.6
                } else if climbed < 1. {
                    1.3
                } else {
                    1.
                }
        }

        fn uniform(&mut self) -> f64 {
            self.jitter ^= self.jitter >> 12;
            self.jitter ^= self.jitter << 25;
            self.jitter ^= self.jitter >> 27;
            ((self.jitter.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 34) as f64 + 0.5)
                / f64::from(1u32 << 30)
        }

        fn read(&mut self, passes: usize) -> Result<f64, Infallible> {
            let mut seconds = 0.;
            for _ in 0..passes {
                let pass = PASS * self.slowdown();
                self.saturated += pass;
                seconds += pass;
            }
            let seconds = seconds * (1. - 0.01 * self.uniform().ln());
            self.elapsed += seconds;
            self.readings += 1;
            Ok(seconds)
        }
    }

    /// A reading's device seconds, and its passes at the sustained clock.
    const SECONDS: f64 = 200e-6;
    const PASSES: usize = 2;
    const STAGES: u32 = 7;

    /// A gauge established on `device`: its reference.
    fn established(device: &mut Device, stages: u32) -> Reference {
        let (passes, settled, readings) =
            establish(|passes| device.read(passes), SECONDS, stages).unwrap();
        // A reading's passes are calibrated where the establishment ended.
        assert_eq!(passes, PASSES);
        assert_eq!(readings, device.readings);
        Reference::established(settled)
    }

    #[test]
    fn a_reference_is_the_sustained_clock_wherever_the_device_began() {
        // From its lowest state, for holds from none to the longest probed
        // (25 ms), and from the sustained clock itself.
        for mut device in [
            Device::lowest(0.),
            Device::lowest(0.007),
            Device::lowest(0.025),
            Device::sustained(),
        ] {
            let hold = device.hold;
            let reference = established(&mut device, STAGES);
            println!(
                "hold {:.0} ms: reference {:.1} µs in {:.0} ms of saturation",
                hold * 1e3,
                reference.median() * 1e6,
                device.elapsed * 1e3
            );
            assert!(
                (reference.median() / SECONDS - 1.).abs() < 0.03,
                "hold {hold}: {reference:?}"
            );
            assert_eq!(reference.levels, [reference.median()]);
            assert!(reference.bound() >= reference.median());
            assert!(device.elapsed < 0.5, "{}", device.elapsed);
        }
    }

    /// What the stages are for: saturation that ends inside the lowest
    /// state's hold reads that state as steady, and takes it for the
    /// sustained clock.
    #[test]
    fn an_establishment_shorter_than_the_hold_reads_the_lowest_state() {
        let mut device = Device::lowest(0.025);
        let (passes, settled, _) = establish(|passes| device.read(passes), SECONDS, 3).unwrap();
        let reference = Reference::established(settled);
        assert!(
            reference.median() > 4. * passes as f64 * PASS,
            "{reference:?}"
        );
    }

    /// A raise as a clock runs it on `device`, given `allowance` seconds
    /// of saturation: how it ended and its readings. The reading behind
    /// the one that ended it is observed, and is not one of the raise's.
    fn raised(
        device: &mut Device,
        reference: &mut Reference,
        allowance: f64,
    ) -> (RaiseOutcome, usize) {
        let began = Cell::new(device.elapsed);
        let now = Cell::new(device.elapsed);
        let (outcome, readings, _, _, _) = raise(
            || {
                let seconds = device.read(PASSES)?;
                now.set(now.get() + seconds);
                Ok::<_, Infallible>(seconds)
            },
            reference,
            || now.get() - began.get() > allowance,
        )
        .unwrap();
        reference.observe(device.read(PASSES).unwrap());
        (outcome, readings)
    }

    /// A clock's first raise establishes its work, wherever the device
    /// began; every later one is held to what that established.
    #[test]
    fn a_clock_establishes_its_work_with_its_first_raise() {
        let mut device = Device::lowest(0.007);
        let mut clock = SustainedClock::new(SECONDS, STAGES);
        let established = clock.raise(|passes| device.read(passes)).unwrap();
        assert_eq!(established.outcome, RaiseOutcome::Established);
        assert_eq!((established.bound, established.sample), (None, None));
        assert_eq!(established.readings, device.readings);
        assert!((established.level / SECONDS - 1.).abs() < 0.03, "{established:?}");
        let raised = clock.raise(|passes| device.read(passes)).unwrap();
        assert_ne!(raised.outcome, RaiseOutcome::Established);
        assert!(raised.bound.is_some_and(|bound| bound >= established.level), "{raised:?}");
    }

    /// A saturation's level is of readings taken until one adds nothing to
    /// their range.
    #[test]
    fn a_level_is_of_readings_taken_until_one_adds_nothing_to_their_range() {
        let mut readings = [1.00, 1.02, 0.99, 1.03, 1.01, 0.5].into_iter();
        let settled = settled(|| Ok::<_, Infallible>(readings.next().unwrap())).unwrap();
        assert_eq!(settled, [0.99, 1.00, 1.01, 1.02, 1.03]);
        assert_eq!(readings.next(), Some(0.5));
        let reference = Reference::established(settled);
        assert_eq!(reference.levels, [1.01]);
    }

    /// Single readings bound a raise on their fast side: no further above
    /// their median than their fastest is below it.
    #[test]
    fn a_raise_ends_with_a_reading_no_slower_than_the_bound() {
        let mut reference = Reference::established(vec![1.00, 1.01, 1.01, 1.02]);
        assert_eq!(reference.levels, [1.01]);
        assert!((reference.bound() - 1.02).abs() < 1e-12);
        // Above the median and within the bound: it ends the raise, where
        // a reading asked to be no slower than the median would not.
        let mut readings = [1.05, 1.021, 1.015, 1.0].into_iter();
        let raised = raise(
            || Ok::<_, Infallible>(readings.next().unwrap()),
            &mut reference,
            || false,
        )
        .unwrap();
        assert_eq!(raised.0, RaiseOutcome::Passed);
        assert_eq!((raised.1, raised.2), (3, 1.015));
        assert_eq!(reference.levels, [1.01]);
    }

    /// A saturation slower than every level the reference knows costs one
    /// allowance, and its level never costs one again: on a device whose
    /// readings repeat exactly within a saturation (a GB10's do, to the
    /// step of its timer) and whose level differs between saturations, the
    /// raises that expire are those that find a new slowest level.
    #[test]
    fn a_level_expires_a_raise_once() {
        let levels = [1.000, 1.003, 1.001, 1.003, 0.998, 1.004, 1.002, 1.004, 1.003, 0.999];
        let allowance = 5;
        let mut reference = Reference::established(vec![levels[0]; 2]);
        assert_eq!(reference.bound(), levels[0]);
        let mut slowest = levels[0];
        let mut expired = Vec::new();
        for (saturation, level) in levels.into_iter().enumerate().skip(1) {
            let mut left = allowance;
            let (outcome, readings, read, bound, first) = raise(
                || Ok::<_, Infallible>(level),
                &mut reference,
                || {
                    left -= 1;
                    left == 0
                },
            )
            .unwrap();
            assert!(bound >= slowest, "{bound} {slowest}");
            assert_eq!((read, first), (level, level));
            if level > bound {
                // The allowance, the reading that is not faster than the
                // one before it, then the two equal readings a level of
                // this saturation settles on.
                assert_eq!((outcome, readings), (RaiseOutcome::Expired, allowance + 3));
                expired.push(saturation);
            } else {
                assert_eq!((outcome, readings), (RaiseOutcome::Passed, 1));
            }
            // The reading queued behind the raise's last.
            reference.observe(level);
            slowest = slowest.max(level);
            assert!(reference.bound() >= slowest);
        }
        // No more raises expired than levels were a new slowest (1.003 and
        // 1.004, each once): a level seen passes.
        assert!(expired.len() <= 2, "{expired:?}");
        assert_eq!(expired[0], 1);
        assert_eq!(reference.levels.len(), 1 + expired.len());
    }

    /// A level is not taken while the readings are still coming down: work
    /// that is slow when its allowance runs out and falling ends its raise
    /// at its own level, and no level joins the reference.
    #[test]
    fn a_raise_that_expires_while_falling_takes_no_level() {
        let mut reference = Reference::established(vec![1.00, 1.00]);
        let mut reading = 3.0;
        let mut left = 4;
        let (outcome, readings, level, bound, first) = raise(
            || {
                reading = f64::max(reading - 0.25, 1.0);
                Ok::<_, Infallible>(reading)
            },
            &mut reference,
            || {
                left -= 1;
                left == 0
            },
        )
        .unwrap();
        // 2.75 down to 1.00 in steps of 0.25: eight readings, the allowance
        // gone after four of them.
        assert_eq!((outcome, readings), (RaiseOutcome::Passed, 8));
        assert_eq!((first, level, bound), (2.75, 1.0, 1.0));
        assert_eq!(reference.levels, [1.0]);
        // Falling to a plateau above every level known, it settles there.
        let (mut reading, mut left) = (3.0, 2);
        let (outcome, _, level, _, _) = raise(
            || {
                reading = f64::max(reading - 0.25, 1.5);
                Ok::<_, Infallible>(reading)
            },
            &mut reference,
            || {
                left -= 1;
                left == 0
            },
        )
        .unwrap();
        assert_eq!((outcome, level), (RaiseOutcome::Expired, 1.5));
        assert_eq!(reference.levels, [1.0, 1.5]);
    }

    /// An established level is never taken out: a device that sustained a
    /// slower clock for an allowance and is faster since is raised by a
    /// reading at either.
    #[test]
    fn an_established_level_is_kept() {
        let mut device = Device::sustained();
        let mut reference = established(&mut device, STAGES);
        let allowance = device.elapsed;
        device.throttled = 1.2;
        let (outcome, _) = raised(&mut device, &mut reference, allowance);
        assert_eq!(outcome, RaiseOutcome::Expired);
        assert_eq!(reference.levels.len(), 2);
        assert!(
            (reference.levels[1] / (1.2 * reference.levels[0]) - 1.).abs() < 0.03,
            "{reference:?}"
        );
        // The throttled clock is a sustained one now: a raise passes.
        let (outcome, _) = raised(&mut device, &mut reference, allowance);
        assert_eq!(outcome, RaiseOutcome::Passed);
        device.throttled = 1.;
        for _ in 0..200 {
            let (outcome, readings) = raised(&mut device, &mut reference, allowance);
            assert_eq!((outcome, readings), (RaiseOutcome::Passed, 1));
        }
        assert_eq!(reference.levels.len(), 2);
        assert!(reference.bound() >= reference.levels[1]);
    }

    /// At the sustained clock a raise takes a few readings and does not
    /// run out its allowance: a level of three readings is their median,
    /// which a third of later readings are no slower than on average, and
    /// the reference grows by a reading with each raise. (On a device the
    /// first reading of a raise follows other work and reads long; this
    /// one's does not.)
    #[test]
    fn a_raise_at_the_sustained_clock_takes_a_few_readings() {
        let mut device = Device::sustained();
        let (references, raises) = (400, 50);
        let (mut readings, mut expired) = (0, 0);
        for _ in 0..references {
            device.elapsed = 0.;
            device.readings = 0;
            let mut reference = established(&mut device, STAGES);
            let allowance = device.elapsed;
            for _ in 0..raises {
                let (outcome, taken) = raised(&mut device, &mut reference, allowance);
                readings += taken;
                expired += usize::from(outcome == RaiseOutcome::Expired);
            }
        }
        let mean = readings as f64 / f64::from(references * raises);
        println!("readings of a raise at the sustained clock: {mean:.2}; expired {expired}");
        assert!((1.0..3.0).contains(&mean), "{mean}");
        // A level of a few readings is now and then among the fastest the
        // gauge gives, and the raise after it runs out its allowance.
        assert!(expired * 100 < references as usize, "{expired}");
    }

    #[test]
    fn a_raise_from_the_lowest_state_ends_at_the_sustained_clock() {
        let mut device = Device::sustained();
        let mut reference = established(&mut device, STAGES);
        let allowance = device.elapsed;
        // The device idles: it is at its lowest again.
        device.saturated = 0.;
        device.elapsed = 0.;
        let (outcome, _) = raised(&mut device, &mut reference, allowance);
        assert_eq!(outcome, RaiseOutcome::Passed);
        assert!(device.slowdown() == 1., "{}", device.slowdown());
        println!("raise from the lowest state: {:.0} ms", device.elapsed * 1e3);
        assert!(device.elapsed < 0.1, "{}", device.elapsed);
        assert_eq!(reference.levels.len(), 1);
    }
}
