//! The sustained clock a session is sampled at.
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
//! clock has been taken, which is the work of the holder whose sessions it
//! raises the device for and no other's ([`Clock`]): a
//! holder's first session establishes it, with that session's first work. A
//! session first *raises* the device: gauge readings, each queued behind
//! the one before it so the device is never without work, until one reads
//! no slower than the gauge's reference takes a sustained clock to read
//! (`Reference`). The first reading of a raise follows other work and
//! reads long (6% at the median), so a raise at the sustained clock takes
//! about two readings; below it none passes until the device has climbed,
//! the nearest state being 1.07× away. The session's samples follow at
//! once, each behind its lead, and the pause between two of them (the
//! host's turn, 130 µs on the M4 Pro) is one the governor holds through.
//!
//! There are two kinds of evidence that a device is at its sustained clock,
//! and no third. One is saturation for as long as it takes to get there
//! from any state: the lowest state holds for 7 ms to 25 ms of saturating
//! work before it moves, within that it is as steady as the highest, no
//! reading tells them apart, and how long the hold is is a property of the
//! governor the device does not report. The first reference is therefore
//! taken by saturating over doubling lengths (`establish`), as many of
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

use crate::deadline::{Deadline, WorkError};
use crate::session::{Slot, Slotted, passes_covering};
use crate::statistics::{Tally, median_of};
use crate::work::{Gauge, Pending, Work};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Readings of a gauge under saturation: each is queued while the one
/// before it runs, so the device has work from the first reading to the
/// last and beyond it (the reading queued behind the last one read is still
/// to run when its reader goes on).
struct Saturation<'g, E> {
    gauge: &'g dyn Gauge<E>,
    ahead: Option<(usize, Box<dyn Pending<E>>)>,
    deadline: Deadline,
}

impl<E> Saturation<'_, E> {
    fn submit(&self, passes: usize) -> Result<Box<dyn Pending<E>>, WorkError<E>> {
        self.deadline.check(Duration::ZERO)?;
        Ok(self.gauge.submit(passes)?)
    }

    /// The seconds of the next reading of `passes` passes.
    fn read(&mut self, passes: usize) -> Result<f64, WorkError<E>> {
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
                Ok(reading.seconds()?)
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
    /// at, with no pause since: its passes and its seconds.
    fn finish(self) -> Result<Option<(usize, f64)>, E> {
        match self.ahead {
            Some((passes, queued)) => Ok(Some((passes, queued.seconds()?))),
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
    /// How it ended.
    pub outcome: RaiseOutcome,
    /// Gauge readings taken, the level's among them.
    pub readings: usize,
    /// Seconds of a reading: the one that passed, or the level the
    /// saturation established.
    pub level: f64,
    /// What a reading was held to; none for an establishment.
    pub bound: Option<f64>,
    /// Seconds of the first reading taken.
    pub first: f64,
    /// Host seconds taken.
    pub seconds: f64,
    /// The first sample of the session that followed a raise, as its
    /// slot's record states it (seconds per call): which samples were taken
    /// behind this raise.
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
/// resolve ([`Tally::fewest`]): the same evidence is asked of the device's
/// state as of a challenger's lead. The length does not double further: a
/// stage at the sustained clock is flat half the time, and stages that
/// doubled until one was would have no expected end.
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
        let passes = passes_covering(length, *pass);
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
    let passes = passes_covering(seconds, pass);
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
    /// The seconds of a reading: a sample's.
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
    /// call to the next, and returns their seconds.
    ///
    /// The first raise is the work's establishment ([`establish`]); every
    /// later one reads until a reading is no slower than the work's
    /// reference takes a sustained clock to read, or for as long as the
    /// establishment took ([`raise`]). What it did is returned.
    pub fn raise<E>(
        &mut self,
        mut read: impl FnMut(usize) -> Result<f64, E>,
    ) -> Result<Raise, E> {
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

/// The sustained clock of one holder: the holder's own work, placed once
/// as a gauge, and what it reads. The holder keeps it from its first
/// session to its last.
pub struct Clock<E> {
    gauge: Box<dyn Gauge<E>>,
    sustained: Mutex<SustainedClock>,
    /// What the clock's raises and its establishment did since they were
    /// last taken ([`Clock::raises`]).
    raises: Mutex<Vec<Raise>>,
}

impl<E> Clock<E> {
    /// What the clock's raises and its establishment did since this was
    /// last asked: the record of the piece of work that ran them.
    pub fn raises(&self) -> Vec<Raise> {
        std::mem::take(
            &mut *self
                .raises
                .lock()
                .expect("raise record lock is never poisoned"),
        )
    }

    /// `seconds` is a sample of the session that followed the clock's last
    /// raise: its first, when that raise has none yet ([`Raise::sample`]).
    pub(crate) fn followed(&self, seconds: f64) {
        let mut raises = self
            .raises
            .lock()
            .expect("raise record lock is never poisoned");
        if let Some(raise) = raises.last_mut() {
            raise.sample.get_or_insert(seconds);
        }
    }

    /// A clock whose work is `work`'s, placed now as a gauge of its own
    /// ([`Work::gauge`]), with readings of `min_sample_seconds` of device
    /// time. Its establishment asks of the device's state the evidence a
    /// lead takes at `error_rate`: its stages are the rounds a comparison
    /// at that rate needs to resolve ([`Tally::fewest`]).
    pub fn new<W: Work<Error = E>>(
        work: &W,
        min_sample_seconds: f64,
        error_rate: f64,
    ) -> Result<Self, E> {
        Ok(Clock {
            gauge: work.gauge()?,
            sustained: Mutex::new(SustainedClock::new(
                min_sample_seconds,
                Tally::default().fewest(error_rate),
            )),
            raises: Mutex::new(Vec::new()),
        })
    }

    /// The clock its holder keeps in `held` from its first session on: the
    /// one it holds, and otherwise a new one whose work is `slot`'s
    /// ([`Clock::new`]). A tuning unit's first session is of its default
    /// configuration at the first point it discovers, so that is the work a
    /// unit gauges with.
    pub fn hold<W: Work<Error = E>>(
        held: &mut Option<Arc<Clock<E>>>,
        slot: &Slot<W>,
        min_sample_seconds: f64,
        error_rate: f64,
    ) -> Result<Arc<Clock<E>>, E> {
        if let Some(clock) = held {
            return Ok(clock.clone());
        }
        let clock = Arc::new(Clock::new(slot.work(), min_sample_seconds, error_rate)?);
        *held = Some(clock.clone());
        Ok(clock)
    }

    /// Bring the device to a sustained clock for the session that follows
    /// at once, with the clock's work queued reading behind reading: the
    /// first raise is the work's establishment, every later one reads until
    /// a reading passes the work's reference or for as long as the
    /// establishment took (module documentation). What it did is recorded
    /// ([`Clock::raises`]).
    pub fn raise(&self, deadline: Deadline) -> Result<(), WorkError<E>> {
        let mut sustained = self.sustained.lock().expect("clock lock is never poisoned");
        let mut saturation = Saturation {
            gauge: &*self.gauge,
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

/// Raise `clock`'s device for a session over `slots`, whose first slot the
/// raise is observed at ([`Slotted::warmed`]). The session's
/// [`crate::session::sample`] follows at once.
pub fn warm<S: Slotted>(
    clock: &Clock<<S::Work as Work>::Error>,
    slots: &[S],
    deadline: Deadline,
) -> Result<(), WorkError<<S::Work as Work>::Error>> {
    let began = Instant::now();
    let raised = clock.raise(deadline);
    if let Some(first) = slots.first() {
        first.warmed(began.elapsed().as_secs_f64());
    }
    raised
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
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
        assert!(
            (established.level / SECONDS - 1.).abs() < 0.03,
            "{established:?}"
        );
        let raised = clock.raise(|passes| device.read(passes)).unwrap();
        assert_ne!(raised.outcome, RaiseOutcome::Established);
        assert!(
            raised.bound.is_some_and(|bound| bound >= established.level),
            "{raised:?}"
        );
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
        let levels = [
            1.000, 1.003, 1.001, 1.003, 0.998, 1.004, 1.002, 1.004, 1.003, 0.999,
        ];
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
        println!(
            "raise from the lowest state: {:.0} ms",
            device.elapsed * 1e3
        );
        assert!(device.elapsed < 0.1, "{}", device.elapsed);
        assert_eq!(reference.levels.len(), 1);
    }
}
