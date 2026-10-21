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

/// One placed [`Work`] and what its session has sampled of it.
pub struct Slot<W: Work> {
    work: W,
    /// Passes over the work per sample; `None` until calibrated.
    repetitions: Option<usize>,
    pass_seconds: Option<f64>,
    samples: Vec<f64>,
}

/// What one calibrating pass gave: its seconds per call and per pass, and
/// the repetitions a sample covering the minimum device time takes.
struct Calibration {
    per_call: f64,
    pass_seconds: f64,
    repetitions: usize,
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
/// calibrating pass, where that is the sample), or submitted.
enum Begun<W: Work> {
    Read(f64),
    Submitted(PendingSample<W>),
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
            repetitions: None,
            pass_seconds: None,
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

    /// The passes a sample takes, once calibrated.
    pub fn repetitions(&self) -> Option<usize> {
        self.repetitions
    }

    /// Submit `passes` passes over the work as one unit of device work.
    fn submit_passes(
        &self,
        passes: usize,
        deadline: Deadline,
    ) -> Result<PendingSample<W>, WorkError<W::Error>> {
        deadline.check(Duration::from_secs_f64(
            self.pass_seconds.unwrap_or(0.) * passes as f64,
        ))?;
        let pending = self.work.submit(passes)?;
        Ok(PendingSample {
            pending,
            calls: passes * self.work.calls_per_pass(),
        })
    }

    /// Calibrate the slot without sampling it: one pass behind its lead
    /// gives the repetitions for which a sample covers at least
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
    ) -> Result<usize, WorkError<W::Error>> {
        let lead = self.submit_passes(1, deadline)?;
        let measured = self.calibrating_pass(min_sample_seconds, deadline);
        let calibration = behind(measured, lead.wait())?;
        Ok(self.calibrated(calibration))
    }

    /// The calibrating pass, behind a lead its caller submitted: one pass,
    /// whose time gives the repetitions for which a sample covers at least
    /// `min_sample_seconds`.
    fn calibrating_pass(
        &self,
        min_sample_seconds: f64,
        deadline: Deadline,
    ) -> Result<Calibration, WorkError<W::Error>> {
        let per_call = self.submit_passes(1, deadline)?.seconds()?;
        let pass_seconds = per_call * self.work.calls_per_pass() as f64;
        Ok(Calibration {
            per_call,
            pass_seconds,
            repetitions: passes_covering(min_sample_seconds, pass_seconds),
        })
    }

    /// Record `calibration` as the slot's: its pass time and repetitions.
    fn calibrated(&mut self, calibration: Calibration) -> usize {
        self.pass_seconds = Some(calibration.pass_seconds);
        self.repetitions = Some(calibration.repetitions);
        calibration.repetitions
    }

    /// Begin a sample behind `lead`, a pass of the work submitted before
    /// it: the sample, and the repetitions it takes. An uncalibrated slot's
    /// sample is first one pass, which gives its repetitions, and is the
    /// sample itself when they are one; otherwise the sample follows it at
    /// once.
    fn begin(
        &mut self,
        lead: &PendingSample<W>,
        min_sample_seconds: f64,
        deadline: Deadline,
    ) -> Result<(Begun<W>, usize), WorkError<W::Error>> {
        if self.pass_seconds.is_none() {
            // A challenger can inherit the incumbent's repetitions, but not
            // its latency. Price the already submitted lead before
            // admitting that first repeated batch.
            self.pass_seconds = Some(lead.pending.seconds()?);
        }
        let repetitions = match self.repetitions {
            Some(repetitions) => repetitions,
            None => {
                let calibration = self.calibrating_pass(min_sample_seconds, deadline)?;
                let pass = calibration.per_call;
                let repetitions = self.calibrated(calibration);
                if repetitions == 1 {
                    return Ok((Begun::Read(pass), repetitions));
                }
                // The calibrating pass just ran: it leads the sample.
                repetitions
            }
        };
        let sample = self.submit_passes(repetitions, deadline)?;
        Ok((Begun::Submitted(sample), repetitions))
    }

    /// A begun sample's seconds per call, once it completes. A submitted
    /// sample's also price the slot's pass.
    fn read(&mut self, sample: Begun<W>) -> Result<f64, WorkError<W::Error>> {
        match sample {
            Begun::Read(seconds) => Ok(seconds),
            Begun::Submitted(pending) => {
                let seconds = pending.seconds()?;
                // A pending sample already divides by repetitions and call
                // count. Multiplying by the calls of a pass recovers one
                // pass.
                self.pass_seconds = Some(seconds * self.work.calls_per_pass() as f64);
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
        let lead = self.submit_passes(1, deadline)?;
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
        let lead = self.submit_passes(1, deadline)?;
        let begun = self.begin(&lead, min_sample_seconds, deadline);
        let (mut sample, repetitions) = behind(begun, lead.wait())?;
        // The device seconds of one sample: its passes' calls, at a call's
        // seconds.
        let calls = (repetitions * self.work.calls_per_pass()) as f64;
        loop {
            // Submitted before the sample is read, so that the device goes
            // from the sample into it without a pause. The sample is
            // expected to take as long as the last pass priced.
            let spent = calls * self.samples.iter().sum::<f64>();
            let expected = self.pass_seconds.unwrap_or(0.) * repetitions as f64;
            let follow = match stopping.after(self.samples.len() + 1, spent + expected) {
                After::Continues => self
                    .submit_passes(repetitions, deadline)
                    .map(Follow::Sample),
                After::Reads(resolution) => self
                    .submit_passes(1, deadline)
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
                Follow::Sample(next) => Begun::Submitted(next),
                Follow::Lead(lead, resolution) => {
                    let spent = calls * self.samples.iter().sum::<f64>();
                    if resolution.stops(&self.samples, spent) {
                        return behind(Ok(()), lead.wait());
                    }
                    let next = self.submit_passes(repetitions, deadline);
                    Begun::Submitted(behind(next, lead.wait())?)
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
            repetitions: self.repetitions.unwrap_or(1) * self.work.calls_per_pass(),
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
/// growing by a sixteenth, or, where the caller gave a budget, once its
/// samples cover that much device time, whichever comes first. The fewest
/// samples a resolved median takes follow from the error rate, and its
/// bound holds however many are taken: a measurement for a caller that
/// names how precise, not how many. A budget bounds how long that takes
/// where a sample is long: the rule is also read before the sample that is
/// expected to reach it, so at most one sample and two leads follow the
/// budget, and a median the budget ends before it is resolved has no
/// interval below the fewest samples that bound one.
pub(crate) enum Stopping {
    /// After this many samples.
    Count(usize),
    /// Once the median is resolved.
    Resolved(Resolution),
}

/// The stopping rule of a resolved median: its error rate, the count it is
/// next read at, and the device seconds its samples may cover.
pub(crate) struct Resolution {
    error_rate: f64,
    next: usize,
    budget: Option<f64>,
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
        Self::Resolved(Resolution {
            error_rate,
            next,
            budget: None,
        })
    }

    /// Sampling until the median is resolved at `error_rate` or the samples
    /// cover `seconds` of device time, whichever comes first.
    pub(crate) fn within(error_rate: f64, seconds: f64) -> Self {
        let mut stopping = Self::resolved(error_rate);
        if let Self::Resolved(resolution) = &mut stopping {
            resolution.budget = Some(seconds);
        }
        stopping
    }

    /// What the rule does once `count` samples are read, which are expected
    /// to cover `expected` seconds of device time.
    fn after(&mut self, count: usize, expected: f64) -> After<'_> {
        match self {
            Self::Count(samples) if count < *samples => After::Continues,
            Self::Count(_) => After::Stops,
            Self::Resolved(resolution)
                if count == resolution.next || resolution.reaches(expected) =>
            {
                After::Reads(resolution)
            }
            Self::Resolved(_) => After::Continues,
        }
    }
}

impl Resolution {
    /// Whether samples covering `seconds` of device time reach the budget.
    fn reaches(&self, seconds: f64) -> bool {
        self.budget.is_some_and(|budget| seconds >= budget)
    }

    /// Whether sampling stops at `samples`, a count the rule is read at,
    /// which cover `seconds` of device time: the median is resolved, or the
    /// samples have reached the budget. A reading the budget asked for
    /// leaves the rule's own counts as they are.
    fn stops(&mut self, samples: &[f64], seconds: f64) -> bool {
        if samples.len() >= self.next {
            self.next = next_reading(samples.len());
        }
        Timing::of(samples, self.error_rate).resolved() || self.reaches(seconds)
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
        // The slots of a group are sampled with the repetitions of its
        // first, the choice the others are compared with.
        for group in groups.iter().filter(|group| group.len() > 1) {
            let first = group[0];
            let repetitions = match slots[first].slot().repetitions {
                Some(repetitions) => repetitions,
                None => slots[first]
                    .slot_mut()
                    .calibrate(min_sample_seconds, deadline)
                    .map_err(|error| SlotFailure { slot: first, error })?,
            };
            for &slot in &group[1..] {
                slots[slot]
                    .slot_mut()
                    .repetitions
                    .get_or_insert(repetitions);
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
    /// (work, passes), a gauge's work numbered its work's plus 100, and
    /// each read and wait of a submission, by its place among them.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Event {
        Submitted(usize, usize),
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
        log: Log,
    }

    impl Scripted {
        fn new(id: usize, pass: f64, log: &Log) -> Self {
            Self {
                id,
                pass,
                spread: 0.,
                log: log.clone(),
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
            let submission = submissions(&self.log).len();
            self.log
                .borrow_mut()
                .push(Event::Submitted(self.id, passes));
            let jitter = (submission * 7919 % 101) as f64 / 101.;
            Ok(Read {
                seconds: self.pass * passes as f64 * (1. + self.spread * jitter),
                submission,
                log: self.log.clone(),
            })
        }
        fn calls_per_pass(&self) -> usize {
            1
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
        let repetitions = slot.repetitions().unwrap();
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

    /// A work whose samples are long stops once they cover its budget,
    /// before its median is resolved: the rule is read before the sample
    /// expected to reach the budget, so a lead follows that sample and
    /// nothing more, and the median has no interval. A work whose samples
    /// are short is resolved within it, as without one.
    #[test]
    fn a_budget_ends_long_samples_before_the_median_is_resolved() {
        const RATE: f64 = 0.05;
        let log = Log::default();
        let mut slot = Slot::new(Scripted::new(0, 1., &log));
        slot.sample_alone(
            1e-3,
            &mut Stopping::within(RATE, 2.5),
            None,
            Deadline::default(),
        )
        .unwrap_or_else(|_| panic!("scripted work does not fail"));
        assert_eq!(slot.samples(), [1.; 3]);
        // Its lead, the calibrating pass (the first sample), two samples,
        // and the lead behind the last.
        assert_eq!(submissions(&log), [(0, 1); 5]);
        assert_eq!(slot.measurement(Some(RATE)).interval, None);

        let log = Log::default();
        let mut slot = Slot::new(Scripted::new(0, 1e-4, &log));
        slot.sample_alone(
            1e-3,
            &mut Stopping::within(RATE, 2.5),
            None,
            Deadline::default(),
        )
        .unwrap_or_else(|_| panic!("scripted work does not fail"));
        assert_eq!(slot.samples().len(), fewest_bounding(RATE));
        assert!(slot.measurement(Some(RATE)).timing().resolved());
    }
}
