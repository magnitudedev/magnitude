//! Measuring a point: [`measure`], the one way to time a [`Work`] on its
//! own, and [`measure_all`], which keeps what was measured as [`Record`]s
//! under the caller's keys.

use crate::clock::{raise_held, HeldClocks};
use crate::deadline::{Deadline, WorkError};
use crate::session::{Measurement, Slot, Stopping};
use crate::statistics::{fewest_bounding, Timing};
use crate::work::Work;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::num::NonZeroUsize;

/// When a measurement stops sampling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Stop {
    /// After this many samples.
    Count(NonZeroUsize),
    /// Once [`Timing::resolved`] holds at `error_rate`.
    Resolved { error_rate: f64 },
    /// Once [`Timing::resolved`] holds at `error_rate`, or once sampling has
    /// taken `seconds` of wall time, whichever comes first, after one sample
    /// at least. The budget bounds how many samples a point takes, never
    /// whether it has one, and a median it ends after one sample has no
    /// interval.
    Within { error_rate: f64, seconds: f64 },
    /// As [`Stop::Within`], the budget unable to end sampling before `floor`
    /// samples are read: a point's median always rests on at least `floor`
    /// samples.
    Floor {
        error_rate: f64,
        seconds: f64,
        floor: NonZeroUsize,
    },
    /// As [`Stop::Within`], the budget unable to end sampling before the
    /// fewest samples that bound the median ([`fewest_bounding`]): a point
    /// whose figure needs an interval has one whatever a call takes.
    Bounded { error_rate: f64, seconds: f64 },
}

/// How a point is measured: when sampling stops, and the device time a
/// sample covers at least (its repetitions follow from it). Nothing is
/// defaulted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Measure {
    /// When sampling stops.
    pub stop: Stop,
    /// The device time a sample covers at least.
    pub min_sample_seconds: f64,
}

impl Measure {
    /// This measurement with a [`Stop::Within`] or [`Stop::Floor`] made
    /// [`Stop::Bounded`]. Another stop is unchanged.
    pub fn bounded(mut self) -> Self {
        match self.stop {
            Stop::Within { error_rate, seconds } | Stop::Floor { error_rate, seconds, .. } => {
                self.stop = Stop::Bounded { error_rate, seconds };
            }
            Stop::Count(_) | Stop::Resolved { .. } | Stop::Bounded { .. } => {}
        }
        self
    }
}

/// Measure `work`: raised first to its sustained clock where `clocks`
/// holds one for it (none for a measurement of the device as it is), then
/// sampled as `how` says, with no cutoff. The one way to time a point: a
/// point begun is timed to its end, its samples as many as `how` stops at
/// and at least one, whatever the time its host has left; when a point
/// begins is its host's ([`measure_all`]).
pub fn measure<W: Work>(
    work: W,
    how: &Measure,
    clocks: Option<&mut HeldClocks<W::Clock, W::Error>>,
) -> Result<Measurement, W::Error> {
    // With no cutoff, nothing that times the point can expire.
    let uncut = |error| match error {
        WorkError::Failed(error) => error,
        WorkError::Expired => unreachable!("invariant violated: measuring without a cutoff expired"),
    };
    let deadline = Deadline::default();
    let mut slot = Slot::new(work);
    let clock = match clocks {
        Some(clocks) => clocks.hold(&slot)?,
        None => None,
    };
    raise_held(clock.as_deref(), std::slice::from_ref(&slot), deadline).map_err(uncut)?;
    let (mut stopping, error_rate) = match how.stop {
        Stop::Count(samples) => (Stopping::Count(samples.get()), None),
        Stop::Resolved { error_rate } => (Stopping::resolved(error_rate), Some(error_rate)),
        Stop::Within { error_rate, seconds } => (Stopping::within(error_rate, seconds, 1), Some(error_rate)),
        Stop::Floor { error_rate, seconds, floor } => (Stopping::within(error_rate, seconds, floor.get()), Some(error_rate)),
        Stop::Bounded { error_rate, seconds } => (
            Stopping::within(error_rate, seconds, fewest_bounding(error_rate)),
            Some(error_rate),
        ),
    };
    slot.sample_alone(how.min_sample_seconds, &mut stopping, clock.as_deref(), deadline)
        .map_err(uncut)?;
    Ok(slot.measurement(error_rate))
}

/// Why a point has no measured time: the reason a host gave for not
/// measuring it (a timing failure that ended its jobs), or why its work
/// could not be formed, as text of at most [`Reason::BYTES`] bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Reason(String);

impl Reason {
    /// The most bytes of a reason: 4 KiB. A reason is a message shown to a
    /// person (a driver's error with its context, or a host's note of why
    /// it skipped a point), and a record is read whole before it is
    /// decoded, so the bound keeps every record's read small; a longer
    /// message is cut at a character boundary.
    pub const BYTES: usize = 4096;

    /// `text`, cut to at most [`Reason::BYTES`] bytes at a character
    /// boundary.
    pub fn new(text: impl Into<String>) -> Self {
        let mut text = text.into();
        if text.len() > Self::BYTES {
            let mut end = Self::BYTES;
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            text.truncate(end);
        }
        Self(text)
    }

    /// The reason's text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Reason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// What is known of a point under its key: its measured time, or that it
/// cannot be measured and why. A record's validity is its key; a host never
/// evicts one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Record {
    /// The point's measured time.
    Measured(Timing),
    /// The point is not measured under this key, for `reason`.
    Unobtainable { reason: Reason },
}

impl Record {
    /// The most bytes a record is written as in JSON, derived from the
    /// widest of each kind: a timing whose numbers are all the longest a
    /// float is written as (`-2.2250738585072014e-308`, 24 characters) with
    /// the most samples, and a reason of [`Reason::BYTES`] control
    /// characters, each written as a six-character escape. What a store
    /// bounds a read of a record by.
    pub fn largest() -> u64 {
        let widest = -f64::MIN_POSITIVE;
        let measured = Record::Measured(Timing {
            median: widest,
            interval: Some((widest, widest)),
            samples: u32::MAX,
        });
        let unobtainable = Record::Unobtainable {
            reason: Reason("\u{1}".repeat(Reason::BYTES)),
        };
        [measured, unobtainable]
            .iter()
            .map(|record| {
                serde_json::to_vec(record)
                    .expect("a record serializes")
                    .len() as u64
            })
            .max()
            .expect("two records")
    }
}

/// Why a [`RecordStore`] could not read or write a record: an absent
/// record is not an error, an unreadable one is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoreError(pub String);

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for StoreError {}

/// Persistence of records by key, which the host implements (as it does
/// for formed artifacts): reads bounded by [`Record::largest`], writes
/// atomic.
pub trait RecordStore<K> {
    /// The record stored under `key`; none when none is.
    fn load(&self, key: &K) -> Result<Option<Record>, StoreError>;
    /// Store `record` under `key`, replacing what was stored there.
    fn store(&self, key: &K, record: &Record) -> Result<(), StoreError>;
}

/// What a host decides of a point before it is timed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Attempt {
    /// Form and time the point.
    Proceed,
    /// Do not time the point; it is recorded unobtainable for `Reason`.
    Skip(Reason),
}

/// Why a point's work was not formed.
#[derive(Debug)]
pub enum Unformed<G> {
    /// It cannot be formed under its key (its kernels do not form, which is
    /// deterministic for the key): it is recorded unobtainable.
    Never(G),
    /// It could not be formed now (its operands could not be allocated, say):
    /// nothing is recorded, as for a timing that fails, and whether the
    /// failure is permanent is the host's decision ([`Attempt::Skip`]).
    Now(G),
}

/// What became of one point of [`measure_all`].
#[derive(Debug)]
pub enum Outcome<E, G> {
    /// Its record was stored already.
    Loaded(Record),
    /// It was measured now, and its record stored.
    Measured(Timing),
    /// The host skipped it, and it was recorded unobtainable.
    Skipped(Reason),
    /// Its work cannot be formed under its key ([`Unformed::Never`]), and
    /// it was recorded unobtainable with the failure.
    Unformed(G),
    /// Its work could not be formed now ([`Unformed::Now`]), and nothing was
    /// recorded: a later request attempts it again, and whether the failure
    /// is permanent is the host's decision ([`Attempt::Skip`]).
    Unmade(G),
    /// Timing it failed, and nothing was recorded: a later request attempts
    /// it again, and whether the failure is permanent is the host's
    /// decision ([`Attempt::Skip`]).
    Failed(E),
    /// It was not begun: it was reached once the cutoff for beginning
    /// points had passed, and nothing is recorded. A point begun never
    /// expires: it is formed and timed to its end ([`measure`]).
    Expired,
    /// Its record could not be read or written.
    Store(StoreError),
}

/// Measure every point of `points` that `store` holds no record of, as
/// `how` says, with `clocks` holding the sustained clocks. `deadline` is
/// when its host stops beginning points: a point reached once it has passed
/// is neither attempted nor formed, and one begun is timed to its end
/// ([`measure`]), so whether a point is read, and what is recorded of it,
/// never follows from how long its host took to reach it. Each point's work
/// is formed by its own `F` only after `attempting` says to proceed, so a
/// host's attempt covers forming it as well as timing it. A point that
/// fails does not stop the rest. This is the one writer of records: a
/// point measured is recorded as its timing, and one skipped or whose work
/// cannot be formed under its key ([`Unformed::Never`]) as unobtainable with
/// the reason. One whose work could not be formed now ([`Unformed::Now`]) or
/// whose timing fails is not recorded: either can fail for a moment's
/// reason, and only its host knows when one is permanent. Nor is one that
/// expired.
pub fn measure_all<K, W, G, F>(
    points: Vec<(K, F)>,
    how: &Measure,
    deadline: Deadline,
    clocks: &mut HeldClocks<W::Clock, W::Error>,
    store: &dyn RecordStore<K>,
    attempting: &mut dyn FnMut(&K) -> Attempt,
) -> Vec<(K, Outcome<W::Error, G>)>
where
    W: Work,
    G: fmt::Display,
    F: FnOnce() -> Result<W, Unformed<G>>,
{
    points
        .into_iter()
        .map(|(key, form)| {
            let outcome = match store.load(&key) {
                Ok(Some(record)) => Outcome::Loaded(record),
                Ok(None) if deadline.expired() => Outcome::Expired,
                Ok(None) => attempt(&key, form, how, clocks, store, attempting),
                Err(error) => Outcome::Store(error),
            };
            (key, outcome)
        })
        .collect()
}

/// One point of [`measure_all`] that no record holds: attempted, formed,
/// timed and recorded.
fn attempt<K, W, G, F>(
    key: &K,
    form: F,
    how: &Measure,
    clocks: &mut HeldClocks<W::Clock, W::Error>,
    store: &dyn RecordStore<K>,
    attempting: &mut dyn FnMut(&K) -> Attempt,
) -> Outcome<W::Error, G>
where
    W: Work,
    G: fmt::Display,
    F: FnOnce() -> Result<W, Unformed<G>>,
{
    let unobtainable = |reason: Reason| Some(Record::Unobtainable { reason });
    let (record, outcome) = match attempting(key) {
        Attempt::Skip(reason) => (unobtainable(reason.clone()), Outcome::Skipped(reason)),
        // Forming the work is part of its attempt.
        Attempt::Proceed => match form() {
            Err(Unformed::Never(error)) => (
                unobtainable(Reason::new(error.to_string())),
                Outcome::Unformed(error),
            ),
            Err(Unformed::Now(error)) => (None, Outcome::Unmade(error)),
            Ok(work) => match measure(work, how, Some(clocks)) {
                Ok(measurement) => {
                    let timing = measurement.timing();
                    (Some(Record::Measured(timing)), Outcome::Measured(timing))
                }
                Err(error) => (None, Outcome::Failed(error)),
            },
        },
    };
    match record.map_or(Ok(()), |record| store.store(key, &record)) {
        Ok(()) => outcome,
        Err(error) => Outcome::Store(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::work::{Gauge, Pending};
    use std::cell::RefCell;
    use std::collections::HashMap;

    /// Work whose passes take the seconds it holds each, or whose every
    /// submission fails with the error it holds, without a clock.
    struct Fixed(Result<f64, &'static str>);

    struct Done(f64);

    impl Pending<&'static str> for Done {
        fn seconds(&self) -> Result<f64, &'static str> {
            Ok(self.0)
        }
        fn wait(&self) -> Result<(), &'static str> {
            Ok(())
        }
    }

    impl Work for Fixed {
        type Error = &'static str;
        type Pending = Done;
        type Clock = ();
        fn submit(&self, passes: usize) -> Result<Done, &'static str> {
            self.0.map(|seconds| Done(seconds * passes as f64))
        }
        fn calls_per_pass(&self) -> usize {
            1
        }
        fn submit_cold(&self, passes: usize) -> Result<Done, &'static str> {
            self.submit(passes)
        }
        fn cold_calls_per_pass(&self) -> usize {
            1
        }
        fn clock(&self) -> Option<()> {
            None
        }
        fn gauge(&self) -> Result<Box<dyn Gauge<&'static str>>, &'static str> {
            unreachable!("work without a clock is never gauged")
        }
    }

    #[derive(Default)]
    struct Memory(RefCell<HashMap<&'static str, Record>>);

    impl RecordStore<&'static str> for Memory {
        fn load(&self, key: &&'static str) -> Result<Option<Record>, StoreError> {
            Ok(self.0.borrow().get(key).cloned())
        }
        fn store(&self, key: &&'static str, record: &Record) -> Result<(), StoreError> {
            self.0.borrow_mut().insert(key, record.clone());
            Ok(())
        }
    }

    /// Sampling stops once the median is resolved within the tolerance: 8
    /// samples at one in twenty, the fewest that bound it.
    #[test]
    fn sampling_stops_when_the_median_is_resolved() {
        let how = Measure {
            stop: Stop::Resolved { error_rate: 0.05 },
            min_sample_seconds: 0.,
        };
        let steady = measure(Fixed(Ok(1.)), &how, None).unwrap();
        assert!(steady.timing().resolved());
        assert_eq!(steady.samples.len(), 8);
    }

    /// A budget spent after one sample ends sampling there with no interval;
    /// bounded, it cannot end it before the fewest samples that bound the
    /// median, so whatever a call takes (here a budget of nothing) the point
    /// has an interval; another stop is unchanged.
    #[test]
    fn a_bounded_budget_cannot_end_sampling_before_an_interval_exists() {
        let within = Measure {
            stop: Stop::Within { error_rate: 0.05, seconds: 0. },
            min_sample_seconds: 0.,
        };
        let spent = measure(Fixed(Ok(1.)), &within, None).unwrap();
        assert_eq!(spent.samples.len(), 1);
        assert!(spent.timing().interval.is_none());
        let bounded = measure(Fixed(Ok(1.)), &within.bounded(), None).unwrap();
        assert_eq!(bounded.samples.len(), fewest_bounding(0.05));
        assert!(bounded.timing().interval.is_some());
        let counted = Measure { stop: Stop::Count(NonZeroUsize::new(3).unwrap()), min_sample_seconds: 0. };
        assert_eq!(counted.bounded(), counted);
        let floored = Measure {
            stop: Stop::Floor {
                error_rate: 0.05,
                seconds: 0.,
                floor: NonZeroUsize::new(3).unwrap(),
            },
            min_sample_seconds: 0.,
        };
        assert_eq!(measure(Fixed(Ok(1.)), &floored, None).unwrap().samples.len(), 3);
        assert_eq!(
            measure(Fixed(Ok(1.)), &floored.bounded(), None).unwrap().samples.len(),
            fewest_bounding(0.05),
        );
    }

    /// A stored point is loaded and neither attempted nor formed; the others
    /// are attempted before they are formed. A skipped point and one that
    /// can never be formed are recorded unobtainable, one measured is
    /// recorded as its timing, and one not formed now or whose timing fails
    /// is not recorded.
    #[test]
    fn points_are_formed_only_once_attempted_and_only_lasting_outcomes_are_recorded() {
        let store = Memory::default();
        let stored = Record::Measured(Timing::of(&[1.], 0.05));
        store.store(&"stored", &stored).unwrap();
        let formed = RefCell::new(Vec::new());
        let form = |key: &'static str, work: Result<Fixed, Unformed<&'static str>>| {
            let formed = &formed;
            (key, move || {
                formed.borrow_mut().push(key);
                work
            })
        };
        let points = vec![
            form("stored", Ok(Fixed(Ok(1.)))),
            form("skipped", Ok(Fixed(Ok(1.)))),
            form("unformed", Err(Unformed::Never("no kernel"))),
            form("unmade", Err(Unformed::Now("out of memory"))),
            form("measured", Ok(Fixed(Ok(2e-3)))),
            form("failed", Ok(Fixed(Err("device lost")))),
        ];
        let mut attempted = Vec::new();
        let how = Measure {
            stop: Stop::Count(NonZeroUsize::new(3).unwrap()),
            min_sample_seconds: 0.,
        };
        let outcomes = measure_all(
            points,
            &how,
            Deadline::default(),
            &mut HeldClocks::new(0.05, 1e-3),
            &store,
            &mut |key| {
                attempted.push(*key);
                if *key == "skipped" {
                    Attempt::Skip(Reason::new("ended two jobs"))
                } else {
                    Attempt::Proceed
                }
            },
        );
        assert_eq!(
            attempted,
            ["skipped", "unformed", "unmade", "measured", "failed"]
        );
        assert_eq!(
            *formed.borrow(),
            ["unformed", "unmade", "measured", "failed"]
        );
        assert!(matches!(
            &outcomes[0],
            ("stored", Outcome::Loaded(record)) if *record == stored
        ));
        assert!(matches!(&outcomes[1], ("skipped", Outcome::Skipped(_))));
        assert!(matches!(
            &outcomes[2],
            ("unformed", Outcome::Unformed("no kernel"))
        ));
        assert!(matches!(
            &outcomes[3],
            ("unmade", Outcome::Unmade("out of memory"))
        ));
        assert!(matches!(
            &outcomes[4],
            ("measured", Outcome::Measured(timing)) if timing.median == 2e-3
        ));
        assert!(matches!(
            &outcomes[5],
            ("failed", Outcome::Failed("device lost"))
        ));
        let records = store.0.borrow();
        assert_eq!(
            records["unformed"],
            Record::Unobtainable {
                reason: Reason::new("no kernel")
            }
        );
        assert!(matches!(records["skipped"], Record::Unobtainable { .. }));
        assert!(matches!(records["measured"], Record::Measured(_)));
        assert!(!records.contains_key("unmade"));
        assert!(!records.contains_key("failed"));
    }

    /// Once the cutoff has passed a point no record holds is neither
    /// attempted nor formed: it is not begun, and expires. A stored point is
    /// still loaded.
    #[test]
    fn a_passed_deadline_expires_every_point_no_record_holds() {
        let store = Memory::default();
        let stored = Record::Measured(Timing::of(&[1.], 0.05));
        store.store(&"stored", &stored).unwrap();
        let how = Measure {
            stop: Stop::Count(NonZeroUsize::new(3).unwrap()),
            min_sample_seconds: 0.,
        };
        let mut attempted = Vec::new();
        let passed = Deadline::new(Some(std::time::Instant::now()));
        let work: fn() -> Result<Fixed, Unformed<&'static str>> = || Ok(Fixed(Ok(1.)));
        let outcomes = measure_all(
            vec![("stored", work), ("late", work)],
            &how,
            passed,
            &mut HeldClocks::new(0.05, 1e-3),
            &store,
            &mut |key| {
                attempted.push(*key);
                Attempt::Proceed
            },
        );
        assert!(matches!(&outcomes[0], ("stored", Outcome::Loaded(_))));
        assert!(matches!(&outcomes[1], ("late", Outcome::Expired)));
        assert!(attempted.is_empty(), "{attempted:?}");
        assert_eq!(store.0.borrow().len(), 1);
    }

    /// A point begun before the cutoff is formed and timed to its end and
    /// recorded, however long its host takes to make it (a cold compile, a
    /// busy host): here its making outlasts the cutoff, and it still reads
    /// every sample its stop asks for.
    #[test]
    fn a_point_begun_before_the_cutoff_is_timed_to_its_end() {
        let store = Memory::default();
        let how = Measure {
            stop: Stop::Count(NonZeroUsize::new(3).unwrap()),
            min_sample_seconds: 0.,
        };
        let cutoff = std::time::Duration::from_millis(5);
        let deadline = Deadline::new(Some(std::time::Instant::now() + cutoff));
        let slow = move || {
            std::thread::sleep(4 * cutoff);
            Ok::<_, Unformed<&'static str>>(Fixed(Ok(1.)))
        };
        let outcomes = measure_all(
            vec![("slow", slow)],
            &how,
            deadline,
            &mut HeldClocks::new(0.05, 1e-3),
            &store,
            &mut |_| Attempt::Proceed,
        );
        assert!(deadline.expired(), "the cutoff passed while the point was made");
        assert!(matches!(&outcomes[0], ("slow", Outcome::Measured(timing)) if timing.samples == 3), "{:?}", outcomes[0].1);
        assert!(matches!(store.0.borrow()["slow"], Record::Measured(_)));
    }
}
