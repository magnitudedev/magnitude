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
//! ([`crate::statistics`]), and a measurement of a point's own time takes
//! the count it was asked for ([`Counted`]).

use crate::clock::Clock;
use crate::deadline::{Deadline, WorkError};
use crate::statistics::{Shuffle, median, median_of};
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
        let drained = lead.pending.wait();
        let calibration = measured?;
        drained?;
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

    /// Take one sample: its seconds per call. The sample is queued behind
    /// its lead, one pass that is not timed (module documentation). An
    /// uncalibrated slot's sample is first one pass, which gives its
    /// repetitions, and is the sample itself when they are one; otherwise
    /// the sample follows it at once.
    fn sample(
        &mut self,
        min_sample_seconds: f64,
        deadline: Deadline,
    ) -> Result<f64, WorkError<W::Error>> {
        let lead = self.submit_passes(1, deadline)?;
        let measured = (|| -> Result<f64, WorkError<W::Error>> {
            if self.pass_seconds.is_none() {
                // A challenger can inherit the incumbent's repetitions,
                // but not its latency. Price the already submitted lead
                // before admitting that first repeated batch.
                self.pass_seconds = Some(lead.pending.seconds()?);
            }
            let repetitions = match self.repetitions {
                Some(repetitions) => repetitions,
                None => {
                    let calibration = self.calibrating_pass(min_sample_seconds, deadline)?;
                    let pass = calibration.per_call;
                    let repetitions = self.calibrated(calibration);
                    if repetitions == 1 {
                        return Ok(pass);
                    }
                    // The calibrating pass just ran: it leads the sample.
                    repetitions
                }
            };
            let seconds = self.submit_passes(repetitions, deadline)?.seconds()?;
            // A pending sample already divides by repetitions and call
            // count. Multiplying by the calls of a pass recovers one pass.
            self.pass_seconds = Some(seconds * self.work.calls_per_pass() as f64);
            Ok(seconds)
        })();
        let drained = lead.pending.wait();
        let seconds = measured?;
        drained?;
        Ok(seconds)
    }

    /// The samples so far as a measurement.
    pub fn measurement(&self) -> Measurement {
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
/// and median absolute deviation, and the calls a sample's submission
/// makes.
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

    /// Every submission of the scripted works, as (work, passes); a gauge is
    /// its work's number plus 100.
    type Log = Rc<RefCell<Vec<(usize, usize)>>>;

    /// Work whose pass takes `pass` seconds on a device with one clock.
    struct Scripted {
        id: usize,
        pass: f64,
        log: Log,
    }

    struct Read(f64);

    impl Pending<()> for Read {
        fn seconds(&self) -> Result<f64, ()> {
            Ok(self.0)
        }
        fn wait(&self) -> Result<(), ()> {
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
        fn submit(&self, passes: usize) -> Result<Read, ()> {
            self.log.borrow_mut().push((self.id, passes));
            Ok(Read(self.pass * passes as f64))
        }
        fn calls_per_pass(&self) -> usize {
            1
        }
        fn gauge(&self) -> Result<Box<dyn Gauge<()>>, ()> {
            Ok(Box::new(Scripted {
                id: self.id + 100,
                pass: self.pass,
                log: self.log.clone(),
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
            .map(|(id, pass)| {
                Slot::new(Scripted {
                    id,
                    pass,
                    log: log.clone(),
                })
            })
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
        assert_eq!(*log.borrow(), expected);
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
            .map(|slot| slot.measurement())
            .collect::<Vec<_>>();
        assert_eq!(measurements[0].samples, [1e-4; 3]);
        assert_eq!(measurements[0].repetitions, 20);
        assert_eq!(measurements[1].samples, [3e-4; 3]);
        assert_eq!(measurements[1].repetitions, 7);
    }
}
