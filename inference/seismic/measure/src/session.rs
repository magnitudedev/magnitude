//! The one timing session: a placed [`Work`] sampled round by round, each
//! sample one submission of its passes queued behind a pass that is not
//! timed.
//!
//! # What a sample is
//!
//! A sample is one submission of a slot's passes, queued behind one pass of
//! the same work that is not timed: its *lead*. What work costs when the
//! device comes to it from something else is not what it costs in service,
//! where the same work follows itself: the first pass after other work read
//! 11% to 18% long on an M4 Pro and on a GB10 (a 32-row attention prefill
//! beside its 512-row point, a one-row GEMV beside its 512-row point:
//! 2026-10-10), and the first after the host held the device for 20 ms to
//! 80 ms (a candidate's numerical witnesses) read 14% to 64% long on the M4
//! Pro, 15% to 35% on the GB10; the second pass read as every later one.
//! The lead pays that, the device goes from it into the sample without a
//! pause, and a sample then reads the same whatever ran before it. Each
//! sample completes before the next slot's is submitted: a Metal queue runs
//! command buffers without a hazard between them concurrently, so samples
//! of different slots submitted back to back overlap and each one's
//! interval includes the other's work (measured on an M4 Pro, 2026-09-24: a
//! decode GEMV's samples read 185 µs next to a 4-row point's and 67 µs
//! alone).
//!
//! A slot's first sample of a session calibrates it: the pass behind its
//! lead gives the repetitions for which a sample covers at least
//! `min_sample_seconds` of device time, and is itself the sample when that
//! is one pass. The lead also pays the first-use device costs of what a
//! slot is the first to run (launches no earlier point of the kernel ran,
//! scratch no earlier call touched): a pass that kept those read a
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
//! ([`crate::statistics`]), and a counted session takes the count it was
//! asked for ([`Counted`]).
//!
//! # What a pass reads
//!
//! A sample repeats passes of one work, so whatever a pass reads that fits
//! the device's last-level cache is read from the cache from the second
//! pass on. Served work never finds it there: a decode step streams every
//! layer's weights and states once, and between two calls of one kernel
//! everything else the step reads goes through the cache. Passes the cache
//! holds time the work at the cache's bandwidth and not at memory's, and
//! configurations that stream memory differently read alike: on a GB10
//! (24 MiB of L2, 2026-10-16) a router GEMV's four-layer rotation read 12 to
//! 20 µs per call, and a state-space step's three row mappings read within
//! 1% of one another; with every pass reading four times the cache, the
//! router read 22 to 35 µs and the step 44.5, 47.0 and 48.9 µs.
//!
//! So a sample shorter than `min_sample_seconds` alone repeats *cold*
//! passes ([`Work::submit_cold`]): the work's calls, then its copies'
//! (the same work over storage of its own, which the work's owner sizes to
//! the cache), so that no call finds its bytes left by an earlier one.
//! Calibration times a pass alone, behind a lead alone; a pass that covers
//! `min_sample_seconds` by itself is the sample ([`Passes::Alone`]), behind
//! a lead alone. Such a pass is long beside what a cache can give it: the
//! most a warm cache saves a pass is reading what it holds once, which on
//! the GB10 (273 GB/s) is 92 µs, under 5% of a 2 ms pass; copies would
//! multiply its cost instead (a 60 s tuning budget on an M4 Pro measured 3
//! configurations of a 64k-token attention prefill in 31 s with them, 8 in
//! 12 s without, 2026-10-17). A shorter one is repeated cold, as many passes
//! as cover the minimum ([`Passes::Cold`]), each sample behind a cold lead;
//! the first, which follows the calibrating pass at once, follows a cold
//! pass too where the work has copies.
//!
//! # One work alone
//!
//! A measurement of one work's own time ([`crate::measure`]) samples one
//! slot, so nothing but the work itself need come between its samples, and
//! each sample's lead is the sample before it: every later sample is
//! submitted before the one before it is read, and the device goes from
//! one into the next without a pause. Only the first sample is behind a
//! lead of its own. Where the stopping rule is read after a sample
//! ([`Stopping`]), whether a next sample is taken is not known before that
//! sample is read, so what follows it is a lead: the next sample follows
//! the lead if the rule continues, and the lead is waited for and nothing
//! more is taken if it stops. Every sample counted is one the rule read,
//! no sample is taken beyond where it stops, and a lead is submitted only
//! where the rule may stop: never more passes than a lead per sample would
//! take. Work timed on the host completes
//! inside its submission, so the sample before is complete when the next
//! is submitted, as a lead of its own would be.

use crate::clock::Clock;
use crate::deadline::{Deadline, WorkError};
use crate::statistics::{
    fewest_bounding, median, median_interval, median_of, next_reading, Shuffle, Timing,
};
use crate::work::{Pending, Work};
use std::time::{Duration, Instant};

/// The passes of work whose pass takes `pass_seconds` that cover at least
/// `length` seconds: at least one, and one where a pass read no time.
pub(crate) fn passes_covering(length: f64, pass_seconds: f64) -> usize {
    if pass_seconds > 0. {
        ((length / pass_seconds).ceil() as usize).max(1)
    } else {
        1
    }
}

/// What one sample of a slot submits, behind a lead of one of the same
/// (module documentation, "What a pass reads").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Passes {
    /// One pass of the work alone: it covers `min_sample_seconds` by
    /// itself, or reads no time.
    Alone,
    /// This many cold passes ([`Work::submit_cold`]): a pass alone is
    /// shorter than `min_sample_seconds`.
    Cold(usize),
}

/// One placed [`Work`] and what its session has sampled of it.
pub struct Slot<W: Work> {
    work: W,
    /// What a sample submits; `None` until calibrated.
    passes: Option<Passes>,
    /// Device seconds of one call, as the last submission read gave them.
    call_seconds: Option<f64>,
    samples: Vec<f64>,
}

/// What one calibrating pass gave: its seconds per call, and what a sample
/// covering the minimum device time submits.
struct Calibration {
    per_call: f64,
    passes: Passes,
}

/// A submitted sample of one slot.
struct PendingSample<W: Work> {
    pending: W::Pending,
    calls: usize,
}

impl<W: Work> PendingSample<W> {
    /// Seconds per call, once the sample completes.
    fn seconds(self) -> Result<f64, W::Error> {
        Ok(self.pending.seconds()? / self.calls as f64)
    }

    /// Wait for the submission without timing it: a lead, or what is
    /// drained after a failure.
    fn wait(self) -> Result<(), W::Error> {
        self.pending.wait()
    }
}

/// A sample begun behind its lead: read already (an uncalibrated slot's
/// calibrating pass, where that is the sample), or submitted, behind a cold
/// pass that evicts what the calibrating pass left cached where one ran.
enum Begun<W: Work> {
    Read(f64),
    Submitted {
        sample: PendingSample<W>,
        evicting: Option<PendingSample<W>>,
    },
}

/// What a slot sampled alone submits behind a sample before reading it.
enum Follow<'a, W: Work> {
    /// The next sample, where the stopping rule certainly continues.
    Sample(PendingSample<W>),
    /// A lead, where the rule is read: the next sample follows it if the
    /// rule continues.
    Lead(PendingSample<W>, &'a mut Resolution),
    /// Nothing, where the rule certainly stops.
    Nothing,
}

impl<W: Work> Follow<'_, W> {
    /// Wait for what was submitted, untimed.
    fn wait(self) -> Result<(), W::Error> {
        match self {
            Self::Sample(pending) | Self::Lead(pending, _) => pending.wait(),
            Self::Nothing => Ok(()),
        }
    }
}

/// `outcome` of work submitted behind a lead, given the lead's own outcome
/// (`drained`, waited for after it): the work's failure first, then the
/// lead's.
fn behind<T, E>(
    outcome: Result<T, WorkError<E>>,
    drained: Result<(), E>,
) -> Result<T, WorkError<E>> {
    let value = outcome?;
    drained?;
    Ok(value)
}

impl<W: Work> Slot<W> {
    /// A slot of `work`, not yet calibrated or sampled.
    pub fn new(work: W) -> Self {
        Self {
            work,
            passes: None,
            call_seconds: None,
            samples: Vec::new(),
        }
    }

    /// The work the slot times.
    pub fn work(&self) -> &W {
        &self.work
    }

    /// The work the slot times, to be changed between sessions.
    pub fn work_mut(&mut self) -> &mut W {
        &mut self.work
    }

    /// Seconds per call of every sample taken.
    pub fn samples(&self) -> &[f64] {
        &self.samples
    }

    /// What a sample submits, once calibrated.
    pub fn passes(&self) -> Option<Passes> {
        self.passes
    }

    /// The calls a submission of `passes` makes.
    fn calls(&self, passes: Passes) -> usize {
        match passes {
            Passes::Alone => self.work.calls_per_pass(),
            Passes::Cold(passes) => passes * self.work.cold_calls_per_pass(),
        }
    }

    /// Submit `passes` as one unit of device work.
    fn submit(
        &self,
        passes: Passes,
        deadline: Deadline,
    ) -> Result<PendingSample<W>, WorkError<W::Error>> {
        let calls = self.calls(passes);
        deadline.check(Duration::from_secs_f64(
            self.call_seconds.unwrap_or(0.) * calls as f64,
        ))?;
        let pending = match passes {
            Passes::Alone => self.work.submit(1)?,
            Passes::Cold(passes) => self.work.submit_cold(passes)?,
        };
        Ok(PendingSample { pending, calls })
    }

    /// The lead of the slot's next sample: one pass of the form its samples
    /// take; of an uncalibrated slot, a pass alone, which its calibrating
    /// pass follows.
    fn lead(&self, deadline: Deadline) -> Result<PendingSample<W>, WorkError<W::Error>> {
        match self.passes {
            Some(Passes::Cold(_)) => self.submit(Passes::Cold(1), deadline),
            Some(Passes::Alone) | None => self.submit(Passes::Alone, deadline),
        }
    }

    /// Calibrate the slot without sampling it: one pass behind its lead
    /// gives what a sample covering at least `min_sample_seconds` of device
    /// time submits.
    ///
    /// The two sides of a comparison are sampled with the same passes
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
    ) -> Result<Passes, WorkError<W::Error>> {
        let lead = self.submit(Passes::Alone, deadline)?;
        let measured = self.calibrating_pass(min_sample_seconds, deadline);
        let calibration = behind(measured, lead.wait())?;
        Ok(self.calibrated(calibration))
    }

    /// The calibrating pass, behind a lead its caller submitted: one pass
    /// alone, whose time gives what a sample covering `min_sample_seconds`
    /// submits. A pass that covers it is a sample by itself: the most a
    /// cache its lead left warm saves it is reading what the cache holds
    /// once, beside a pass at least as long as the minimum. One shorter is
    /// repeated cold, as many passes as cover it.
    fn calibrating_pass(
        &self,
        min_sample_seconds: f64,
        deadline: Deadline,
    ) -> Result<Calibration, WorkError<W::Error>> {
        let per_call = self.submit(Passes::Alone, deadline)?.seconds()?;
        let pass = per_call * self.work.calls_per_pass() as f64;
        let passes = match passes_covering(min_sample_seconds, pass) {
            1 => Passes::Alone,
            _ => Passes::Cold(passes_covering(
                min_sample_seconds,
                per_call * self.work.cold_calls_per_pass() as f64,
            )),
        };
        Ok(Calibration { per_call, passes })
    }

    /// Record `calibration` as the slot's: its call time and passes.
    fn calibrated(&mut self, calibration: Calibration) -> Passes {
        self.call_seconds = Some(calibration.per_call);
        self.passes = Some(calibration.passes);
        calibration.passes
    }

    /// Begin a sample behind `lead`, a pass submitted before it: the
    /// sample, and the passes it takes. An uncalibrated slot's sample is
    /// first one pass alone, which gives its passes, and is the sample
    /// itself when they are [`Passes::Alone`]; otherwise the sample follows
    /// it at once, behind a cold pass where the work has copies to evict
    /// the bytes the calibrating pass left cached.
    fn begin(
        &mut self,
        lead: &PendingSample<W>,
        min_sample_seconds: f64,
        deadline: Deadline,
    ) -> Result<(Begun<W>, Passes), WorkError<W::Error>> {
        if self.call_seconds.is_none() {
            // A challenger can inherit the incumbent's passes, but not its
            // latency. Price the already submitted lead before admitting
            // that first repeated batch.
            self.call_seconds = Some(lead.pending.seconds()? / lead.calls as f64);
        }
        let (passes, evicting) = match self.passes {
            Some(passes) => (passes, None),
            None => {
                let calibration = self.calibrating_pass(min_sample_seconds, deadline)?;
                let pass = calibration.per_call;
                let passes = self.calibrated(calibration);
                if passes == Passes::Alone {
                    return Ok((Begun::Read(pass), passes));
                }
                // The calibrating pass just ran, alone: it leads a sample of
                // work without copies; copies run behind it first.
                let evicting = (self.work.cold_calls_per_pass() > self.work.calls_per_pass())
                    .then(|| self.submit(Passes::Cold(1), deadline))
                    .transpose()?;
                (passes, evicting)
            }
        };
        let sample = match self.submit(passes, deadline) {
            Ok(sample) => sample,
            Err(error) => {
                return behind(Err(error), evicting.map_or(Ok(()), PendingSample::wait));
            }
        };
        Ok((Begun::Submitted { sample, evicting }, passes))
    }

    /// A begun sample's seconds per call, once it completes. A submitted
    /// sample's also price the slot's calls.
    fn read(&mut self, sample: Begun<W>) -> Result<f64, WorkError<W::Error>> {
        match sample {
            Begun::Read(seconds) => Ok(seconds),
            Begun::Submitted { sample, evicting } => {
                let seconds = sample.seconds();
                let evicted = evicting.map_or(Ok(()), PendingSample::wait);
                let seconds = behind(seconds.map_err(WorkError::from), evicted)?;
                self.call_seconds = Some(seconds);
                Ok(seconds)
            }
        }
    }

    /// Take one sample: its seconds per call. The sample is queued behind
    /// its lead, one pass that is not timed (module documentation).
    fn sample(
        &mut self,
        min_sample_seconds: f64,
        deadline: Deadline,
    ) -> Result<f64, WorkError<W::Error>> {
        let lead = self.lead(deadline)?;
        let measured = self
            .begin(&lead, min_sample_seconds, deadline)
            .and_then(|(sample, _)| self.read(sample));
        behind(measured, lead.wait())
    }

    /// Sample the slot alone until `stopping` ends it, each sample queued
    /// behind the one before it: a measurement of the work's own time,
    /// raised before by `clock`, the clock it records its first sample
    /// with (module documentation, "One work alone").
    pub(crate) fn sample_alone(
        &mut self,
        min_sample_seconds: f64,
        stopping: &mut Stopping,
        clock: Option<&Clock<W::Error>>,
        deadline: Deadline,
    ) -> Result<(), WorkError<W::Error>> {
        let lead = self.lead(deadline)?;
        let begun = self.begin(&lead, min_sample_seconds, deadline);
        let (mut sample, passes) = behind(begun, lead.wait())?;
        loop {
            // Submitted before the sample is read, so that the device goes
            // from the sample into it without a pause.
            let follow = match stopping.after(self.samples.len() + 1) {
                After::Continues => self.submit(passes, deadline).map(Follow::Sample),
                After::Reads(resolution) => self
                    .lead(deadline)
                    .map(|lead| Follow::Lead(lead, resolution)),
                After::Stops => Ok(Follow::Nothing),
            };
            let read = self.read(sample);
            // The sample has completed: nothing else is in flight where
            // what follows it failed to submit.
            let follow = match follow {
                Ok(follow) => follow,
                Err(error) => return read.and(Err(error)),
            };
            let seconds = match read {
                Ok(seconds) => seconds,
                Err(error) => return behind(Err(error), follow.wait()),
            };
            self.samples.push(seconds);
            if let (Some(clock), 1) = (clock, self.samples.len()) {
                clock.followed(seconds);
            }
            sample = match follow {
                Follow::Sample(next) => Begun::Submitted {
                    sample: next,
                    evicting: None,
                },
                Follow::Lead(lead, resolution) => {
                    if resolution.stops(&self.samples) {
                        return behind(Ok(()), lead.wait());
                    }
                    let next = self.submit(passes, deadline);
                    Begun::Submitted {
                        sample: behind(next, lead.wait())?,
                        evicting: None,
                    }
                }
                Follow::Nothing => return Ok(()),
            };
        }
    }

    /// The samples so far as a measurement: with the anytime-valid
    /// interval for their median at `error_rate` where they were sampled to
    /// the stopping rule at that rate, and none where their count was
    /// decided otherwise (counted, or matched rounds).
    pub fn measurement(&self, error_rate: Option<f64>) -> Measurement {
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
            repetitions: self.calls(self.passes.unwrap_or(Passes::Alone)),
            interval: error_rate.and_then(|rate| median_interval(&self.samples, rate)),
        }
    }
}

/// A slot of a session as its holder keeps it: the slot, and what its
/// holder observes of the session's host time.
pub trait Slotted {
    /// The work the slot times.
    type Work: Work;
    /// The slot.
    fn slot(&self) -> &Slot<Self::Work>;
    /// The slot, to be sampled.
    fn slot_mut(&mut self) -> &mut Slot<Self::Work>;
    /// A sample of the slot took `seconds` of host time.
    fn sampled(&self, _seconds: f64) {}
    /// Raising the device for a session opened by this slot took `seconds`
    /// of host time.
    fn warmed(&self, _seconds: f64) {}
}

impl<W: Work> Slotted for Slot<W> {
    type Work = W;
    fn slot(&self) -> &Slot<W> {
        self
    }
    fn slot_mut(&mut self) -> &mut Slot<W> {
        self
    }
}

/// The samples of one work: seconds per call of every sample, their median
/// and median absolute deviation, the calls a sample's submission makes,
/// and the anytime-valid interval for the median where the measurement was
/// sampled to a stopping rule.
#[derive(Clone, Debug, PartialEq)]
pub struct Measurement {
    /// Seconds per call of every sample, in the order taken.
    pub samples: Vec<f64>,
    /// Median of the samples.
    pub median: f64,
    /// Median absolute deviation of the samples.
    pub deviation: f64,
    /// Calls per sample submission.
    pub repetitions: usize,
    /// The anytime-valid interval for the median at the error rate the
    /// samples were sampled to; none where no stopping rule decided their
    /// count, or while too few samples bound one.
    pub interval: Option<(f64, f64)>,
}

impl Measurement {
    /// The measurement as the time it records.
    pub fn timing(&self) -> Timing {
        Timing {
            median: self.median,
            interval: self.interval,
            samples: u32::try_from(self.samples.len()).expect("fewer than 2^32 samples"),
        }
    }
}

/// A slot whose sample failed, and why.
pub struct SlotFailure<E> {
    /// The failed slot's position among the session's slots.
    pub slot: usize,
    /// Why its sample failed, or that the session's cutoff passed.
    pub error: WorkError<E>,
}

/// What a sampling session takes of its slots (the placed works it was
/// given, by position), decided round by round.
pub trait Rounds {
    /// The slots the next round samples, in groups; none ends the session.
    /// The slots of a group are compared with one another (a challenger
    /// and the choice it is sampled beside), and are sampled one after
    /// another: what disturbs a device for a moment (another process's
    /// work, the host's memory traffic: on a GB10 one sample in ten read
    /// 15% to 35% long, several in a row) then falls on both sides of a
    /// comparison or on neither.
    fn next(&mut self) -> Vec<Vec<usize>>;
    /// A round's samples: each slot sampled, with its seconds per call.
    fn sampled(&mut self, round: &[(usize, f64)]);
}

/// Every slot sampled a given number of times: a measurement of each
/// slot's own time, for a caller that named the count.
pub struct Counted {
    slots: Vec<usize>,
    rounds: usize,
}

impl Counted {
    /// `samples` of each of the first `slots` slots.
    pub fn new(slots: usize, samples: usize) -> Self {
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

/// When a slot sampled alone stops ([`Slot::sample_alone`]): after a count
/// of samples its caller named, or once its median is resolved
/// ([`Timing::resolved`] at the error rate), the rule read at counts
/// growing by a sixteenth. The fewest samples a resolved median takes
/// follow from the error rate, and its bound holds however many are taken:
/// a measurement for a caller that names how precise, not how many.
pub(crate) enum Stopping {
    /// After this many samples.
    Count(usize),
    /// Once the median is resolved.
    Resolved(Resolution),
}

/// The stopping rule of a resolved median: its error rate, and the count
/// it is next read at.
pub(crate) struct Resolution {
    error_rate: f64,
    next: usize,
}

/// What the stopping rule does once a given count of samples is read.
enum After<'a> {
    /// Sampling certainly continues.
    Continues,
    /// The rule is read.
    Reads(&'a mut Resolution),
    /// Sampling certainly stops.
    Stops,
}

impl Stopping {
    /// Sampling until the median is resolved at `error_rate`. The rule's
    /// readings are those of counts growing by a sixteenth from one; none
    /// of them below the fewest samples that bound a median can resolve
    /// it, so the first reading is the first at or above them.
    pub(crate) fn resolved(error_rate: f64) -> Self {
        let fewest = fewest_bounding(error_rate);
        let mut next = 1;
        while next < fewest {
            next = next_reading(next);
        }
        Self::Resolved(Resolution { error_rate, next })
    }

    /// What the rule does once `count` samples are read.
    fn after(&mut self, count: usize) -> After<'_> {
        match self {
            Self::Count(samples) if count < *samples => After::Continues,
            Self::Count(_) => After::Stops,
            Self::Resolved(resolution) if count == resolution.next => After::Reads(resolution),
            Self::Resolved(_) => After::Continues,
        }
    }
}

impl Resolution {
    /// Whether sampling stops at `samples`, a count the rule is read at.
    fn stops(&mut self, samples: &[f64]) -> bool {
        self.next = next_reading(samples.len());
        Timing::of(samples, self.error_rate).resolved()
    }
}

/// Take and keep one sample of each slot of `order` in turn, each
/// completing before the next slot's is submitted: each slot with its
/// seconds per call.
fn collect<S: Slotted>(
    slots: &mut [S],
    order: &[usize],
    min_sample_seconds: f64,
    deadline: Deadline,
) -> Result<Vec<(usize, f64)>, SlotFailure<<S::Work as Work>::Error>> {
    let mut round = Vec::with_capacity(order.len());
    for &slot in order {
        let began = Instant::now();
        let seconds = slots[slot]
            .slot_mut()
            .sample(min_sample_seconds, deadline)
            .map_err(|error| SlotFailure { slot, error })?;
        slots[slot].sampled(began.elapsed().as_secs_f64());
        slots[slot].slot_mut().samples.push(seconds);
        round.push((slot, seconds));
    }
    Ok(round)
}

/// Run a sampling session over `slots`, on a device its caller has raised
/// with `clock` ([`crate::clock::warm`]; none for a session that holds no
/// clock): samples are taken round by round, each round the groups `rounds`
/// names in a random order, and each group's slots in a random order of
/// their own (`shuffle`, which its caller keeps from session to session: the
/// sessions of one comparison must not all open in the same order), until
/// it names none or the cutoff interrupts it. Only complete rounds reach
/// the comparison; a prefix of a matched round is never sampling evidence.
pub fn sample<S: Slotted>(
    slots: &mut [S],
    min_sample_seconds: f64,
    rounds: &mut dyn Rounds,
    shuffle: &mut Shuffle,
    clock: Option<&Clock<<S::Work as Work>::Error>>,
    deadline: Deadline,
) -> Result<(), SlotFailure<<S::Work as Work>::Error>> {
    loop {
        let mut groups = rounds.next();
        if groups.is_empty() {
            return Ok(());
        }
        // The slots of a group are sampled with the passes of its first,
        // the choice the others are compared with.
        for group in groups.iter().filter(|group| group.len() > 1) {
            let first = group[0];
            let passes = match slots[first].slot().passes {
                Some(passes) => passes,
                None => slots[first]
                    .slot_mut()
                    .calibrate(min_sample_seconds, deadline)
                    .map_err(|error| SlotFailure { slot: first, error })?,
            };
            for &slot in &group[1..] {
                slots[slot].slot_mut().passes.get_or_insert(passes);
            }
        }
        let mut turns = (0..groups.len()).collect::<Vec<_>>();
        shuffle.shuffle(&mut turns);
        let mut order = Vec::new();
        for turn in turns {
            shuffle.shuffle(&mut groups[turn]);
            order.append(&mut groups[turn]);
        }
        let round = collect(slots, &order, min_sample_seconds, deadline)?;
        if let (Some(clock), Some(&(_, seconds))) = (clock, round.first()) {
            clock.followed(seconds);
        }
        rounds.sampled(&round);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::{Clock, warm};
    use crate::work::Gauge;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// What the scripted works were asked, in order: each submission as
    /// (work, passes), cold where its passes were, a gauge's work numbered
    /// its work's plus 100, and each read and wait of a submission, by its
    /// place among them.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Event {
        Submitted(usize, usize),
        SubmittedCold(usize, usize),
        Read(usize),
        Waited(usize),
    }

    type Log = Rc<RefCell<Vec<Event>>>;

    /// Every submission of `log`, as (work, passes).
    fn submissions(log: &Log) -> Vec<(usize, usize)> {
        log.borrow()
            .iter()
            .filter_map(|event| match *event {
                Event::Submitted(work, passes) => Some((work, passes)),
                _ => None,
            })
            .collect()
    }

    /// Work whose pass takes `pass` seconds on a device with one clock,
    /// each submission's passes longer by up to `spread` of it, as a fixed
    /// sequence of submissions gives.
    struct Scripted {
        id: usize,
        pass: f64,
        spread: f64,
        /// Copies a cold pass runs after the work's own call.
        copies: usize,
        log: Log,
    }

    impl Scripted {
        fn new(id: usize, pass: f64, log: &Log) -> Self {
            Self {
                id,
                pass,
                spread: 0.,
                copies: 0,
                log: log.clone(),
            }
        }

        /// A submission of `calls` calls, logged as `event`.
        fn submitted(&self, calls: usize, event: Event) -> Read {
            let submission = self
                .log
                .borrow()
                .iter()
                .filter(|event| matches!(event, Event::Submitted(..) | Event::SubmittedCold(..)))
                .count();
            self.log.borrow_mut().push(event);
            let jitter = (submission * 7919 % 101) as f64 / 101.;
            Read {
                seconds: self.pass * calls as f64 * (1. + self.spread * jitter),
                submission,
                log: self.log.clone(),
            }
        }
    }

    struct Read {
        seconds: f64,
        submission: usize,
        log: Log,
    }

    impl Pending<()> for Read {
        fn seconds(&self) -> Result<f64, ()> {
            self.log.borrow_mut().push(Event::Read(self.submission));
            Ok(self.seconds)
        }
        fn wait(&self) -> Result<(), ()> {
            self.log.borrow_mut().push(Event::Waited(self.submission));
            Ok(())
        }
    }

    impl Gauge<()> for Scripted {
        fn submit(&self, passes: usize) -> Result<Box<dyn Pending<()>>, ()> {
            Ok(Box::new(Work::submit(self, passes)?))
        }
    }

    impl Work for Scripted {
        type Error = ();
        type Pending = Read;
        type Clock = ();
        fn submit(&self, passes: usize) -> Result<Read, ()> {
            Ok(self.submitted(passes, Event::Submitted(self.id, passes)))
        }
        fn calls_per_pass(&self) -> usize {
            1
        }
        fn submit_cold(&self, passes: usize) -> Result<Read, ()> {
            if self.copies == 0 {
                return Work::submit(self, passes);
            }
            Ok(self.submitted(
                passes * self.cold_calls_per_pass(),
                Event::SubmittedCold(self.id, passes),
            ))
        }
        fn cold_calls_per_pass(&self) -> usize {
            1 + self.copies
        }
        fn clock(&self) -> Option<()> {
            Some(())
        }
        fn gauge(&self) -> Result<Box<dyn Gauge<()>>, ()> {
            Ok(Box::new(Scripted {
                id: self.id + 100,
                ..Scripted::new(self.id, self.pass, &self.log)
            }))
        }
    }

    /// The establishment of the gauge of work whose pass is 0.1 ms: a pass
    /// behind a lead, then stages doubling from a 2 ms reading over the
    /// seven stages a lead takes at one in twenty, then the level's
    /// readings.
    fn establishment() -> Vec<(usize, usize)> {
        let mut expected = vec![(100, 1), (100, 1)];
        for passes in [20, 40, 80, 160, 320, 640, 1280] {
            expected.extend([(100, passes); 3]);
        }
        expected.extend([(100, 20); 4]);
        expected
    }

    /// The session pinned on scripted work: the order and size of every
    /// submission, and what each raise of the clock recorded, for a
    /// counted measurement. A change to either is a change to what every
    /// measurement takes.
    #[test]
    fn a_counted_session_submits_and_raises_as_recorded() {
        let log = Log::default();
        let mut slots = [1e-4, 3e-4]
            .into_iter()
            .enumerate()
            .map(|(id, pass)| Slot::new(Scripted::new(id, pass, &log)))
            .collect::<Vec<_>>();
        let clock = Clock::new(slots[0].work(), 2e-3, 0.05).unwrap();
        warm(&clock, &slots, Deadline::default()).unwrap();
        sample(
            &mut slots,
            2e-3,
            &mut Counted::new(2, 3),
            &mut Shuffle::default(),
            Some(&clock),
            Deadline::default(),
        )
        .unwrap_or_else(|_| panic!("scripted work does not fail"));
        warm(&clock, &slots, Deadline::default()).unwrap();
        let mut expected = establishment();
        // Each slot's first sample: its lead, its calibrating pass, then the
        // sample of the repetitions that pass gave (one pass of slot 1 is
        // 0.3 ms: 7 of them cover 2 ms). Later samples: a lead and the sample.
        expected.extend([(0, 1), (0, 1), (0, 20), (1, 1), (1, 1), (1, 7)]);
        expected.extend([
            (1, 1),
            (1, 7),
            (0, 1),
            (0, 20),
            (0, 1),
            (0, 20),
            (1, 1),
            (1, 7),
        ]);
        // The raise after the session: one reading, which passes.
        expected.extend([(100, 20), (100, 20)]);
        assert_eq!(submissions(&log), expected);
        let raises = clock
            .raises()
            .iter()
            .map(|raise| {
                format!(
                    "{:?} {} {:e} {:?} {:e} {:?}",
                    raise.outcome,
                    raise.readings,
                    raise.level,
                    raise.bound,
                    raise.first,
                    raise.sample
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            raises,
            [
                "Established 18 2e-3 None 2e-3 Some(0.0001)",
                "Passed 1 2e-3 Some(0.002) 2e-3 None",
            ]
        );
        let measurements = slots
            .iter()
            .map(|slot| slot.measurement(None))
            .collect::<Vec<_>>();
        assert_eq!(measurements[0].samples, [1e-4; 3]);
        assert_eq!(measurements[0].repetitions, 20);
        assert_eq!(measurements[1].samples, [3e-4; 3]);
        assert_eq!(measurements[1].repetitions, 7);
    }

    /// A work sampled alone a counted number of times is behind one lead:
    /// each later sample is submitted before the one before it is read, and
    /// nothing follows the last.
    #[test]
    fn a_work_alone_is_led_once_and_each_sample_follows_the_one_before() {
        use Event::{Read, Submitted, Waited};
        let log = Log::default();
        let mut slot = Slot::new(Scripted::new(0, 1e-4, &log));
        slot.sample_alone(2e-3, &mut Stopping::Count(4), None, Deadline::default())
            .unwrap_or_else(|_| panic!("scripted work does not fail"));
        assert_eq!(
            *log.borrow(),
            [
                // The lead, priced; the calibrating pass behind it, whose
                // 0.1 ms give 20 passes to cover 2 ms; the first sample
                // behind it; the lead drained.
                Submitted(0, 1),
                Read(0),
                Submitted(0, 1),
                Read(1),
                Submitted(0, 20),
                Waited(0),
                // Each later sample submitted before the one before it is
                // read.
                Submitted(0, 20),
                Read(2),
                Submitted(0, 20),
                Read(3),
                Submitted(0, 20),
                Read(4),
                Read(5),
            ]
        );
        assert_eq!(slot.samples(), [1e-4; 4]);
    }

    /// A work sampled alone until its median is resolved stops at the
    /// count the rule stops at read sample by sample from one, with a lead
    /// behind a sample only where the rule is read: from the fewest samples
    /// that bound a median on. Its first sample is the one its clock's
    /// raise records.
    #[test]
    fn a_work_alone_stops_where_its_rule_does_with_leads_only_where_it_is_read() {
        const RATE: f64 = 0.05;
        let log = Log::default();
        let mut slot = Slot::new(Scripted {
            spread: 0.3,
            ..Scripted::new(0, 1e-4, &log)
        });
        let clock = Clock::new(slot.work(), 2e-3, RATE).unwrap();
        warm(&clock, std::slice::from_ref(&slot), Deadline::default()).unwrap();
        slot.sample_alone(
            2e-3,
            &mut Stopping::resolved(RATE),
            Some(&clock),
            Deadline::default(),
        )
        .unwrap_or_else(|_| panic!("scripted work does not fail"));
        let samples = slot.samples();
        let mut reading = 1;
        let mut readings = Vec::new();
        while !Timing::of(&samples[..reading], RATE).resolved() {
            readings.push(reading);
            reading = next_reading(reading);
        }
        readings.push(reading);
        assert_eq!(reading, samples.len());
        // The rule was read and continued before it stopped.
        assert!(samples.len() > fewest_bounding(RATE));
        let Some(Passes::Cold(repetitions)) = slot.passes() else {
            panic!("a pass shorter than the minimum sample is repeated");
        };
        let mut expected = vec![(0, 1), (0, 1), (0, repetitions)];
        for count in 1..=samples.len() {
            if count >= fewest_bounding(RATE) && readings.contains(&count) {
                expected.push((0, 1));
            }
            if count < samples.len() {
                expected.push((0, repetitions));
            }
        }
        let submitted = submissions(&log)
            .into_iter()
            .filter(|&(work, _)| work == 0)
            .collect::<Vec<_>>();
        assert_eq!(submitted, expected);
        assert_eq!(clock.raises()[0].sample, Some(samples[0]));
    }

    /// A pass shorter than the minimum sample is sampled cold: calibrated
    /// alone behind a lead alone, then repeated cold behind a cold pass
    /// that evicts what calibration left cached, and every later sample
    /// behind a cold lead. A pass that covers the minimum is the sample,
    /// alone, and its copies never run.
    #[test]
    fn short_passes_are_sampled_cold_and_long_ones_alone() {
        let log = Log::default();
        let mut short = Slot::new(Scripted {
            copies: 3,
            ..Scripted::new(0, 1e-4, &log)
        });
        // A pass alone of 0.1 ms; a cold pass of four calls, 0.4 ms, five
        // of which cover 2 ms. Seconds are per call.
        for _ in 0..2 {
            assert_eq!(short.sample(2e-3, Deadline::default()).unwrap(), 1e-4);
        }
        assert_eq!(short.passes(), Some(Passes::Cold(5)));
        assert_eq!(short.calls(Passes::Cold(5)), 20);
        let mut long = Slot::new(Scripted {
            copies: 3,
            ..Scripted::new(1, 3e-3, &log)
        });
        long.sample(2e-3, Deadline::default()).unwrap();
        long.sample(2e-3, Deadline::default()).unwrap();
        assert_eq!(long.passes(), Some(Passes::Alone));
        let submitted = log
            .borrow()
            .iter()
            .filter_map(|event| match *event {
                Event::Submitted(work, passes) => Some((work, false, passes)),
                Event::SubmittedCold(work, passes) => Some((work, true, passes)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            submitted,
            [
                (0, false, 1),
                (0, false, 1),
                (0, true, 1),
                (0, true, 5),
                (0, true, 1),
                (0, true, 5),
                (1, false, 1),
                (1, false, 1),
                (1, false, 1),
                (1, false, 1),
            ]
        );
    }
}
