//! How one preparation's tuning time is divided among its units.
//!
//! The time is planned before it is spent, so that tuning ends inside it
//! because the planned work fits, not because work was cut off.
//!
//! Breadth comes before depth. Every unit is first started, the unit with
//! the largest share of step time first: its defaults are measured, and its
//! form starts when the plan has room. Each unit states what its start will
//! cost: its planning and measurements, and the programs it must form. The
//! plan prices a program at what forming one has taken so far in this
//! preparation (a cold pipeline cache shows up here) and scales the units'
//! own estimates by how the starts so far compared with theirs. A unit
//! gets its form starts while they fit together with a defaults-only start
//! and the conclusion of every unit after it; else a defaults-only start
//! while that fits; else it keeps its defaults unmeasured. The plan is made
//! again before each start, from what the starts before it took.
//!
//! The breadth is finished first: a unit started on its defaults alone is
//! owed its form starts, and gets the step that measures them, largest
//! share first, before any slice is dealt by weight (a search of three
//! configurations is ended there whatever its share).
//!
//! The time left is then dealt in slices to the units whose searches are
//! not finished, by the step time each still takes: a unit's share of step
//! time at its defaults times its best cost so far relative to them, which
//! is what further search can recover. What one of a unit's measurements
//! costs takes no part: a unit whose measurements are slow (long prefill
//! points, programs slow to form) gets its share of the time and fewer
//! measurements in it, not less time for each of them. Each slice goes to
//! the unit that with it will have received the least time per unit of that
//! weight, so the units' time follows it from the first slice on (a unit of
//! a hundredth of the step is not refined before the unit of a quarter of
//! it has had twenty-five slices, and a round of first slices does not eat
//! the time on a device whose every step is slow), a unit that overruns a
//! slice waits for the others to catch up, and the time a finished unit
//! does not need goes to the rest.
//!
//! Every started unit can be concluded (its finalists confirmed, its choice
//! validated) within its reserve, its own estimate of that cost. A reserve
//! is a bound, not what the conclusion takes: a unit with nothing to
//! confirm concludes in no time. So when the time left only covers the
//! reserves of the units not yet concluded, one unit is concluded, the one
//! that has received the most for its weight, and what its conclusion left
//! of its reserve goes back to the others, which are refined on; and so on
//! until every unit is concluded, the last when the time left covers only
//! its own reserve. A unit whose search finishes is concluded at once.
//!
//! A slice is dealt only while what a unit cannot interrupt still fits
//! before the reserves: the longest a refinement of any unit has run past
//! its slice (a chunk's programs being formed and measured). A step that
//! runs into the reserves takes the time of the conclusions after it, and
//! a conclusion without its time gives up its choice. The first refinement
//! of a unit started on its defaults alone measures its form starts
//! whatever the slice: that step is held to the plan's price of those
//! starts instead, and says nothing of the slices after it.
//!
//! Under the plan lies a guarantee: every step is given the instant by
//! which it must return with what it has. A step that reaches it, or ends
//! after it, is a failure of the plan, and is reported as one.

use std::time::Duration;

/// Time since the tuning began.
pub(crate) trait Clock {
    fn now(&self) -> Duration;
}

/// Where a unit's search stands, as the allocation reads it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Progress {
    /// The search ran to its own end.
    pub finished: bool,
    /// Its best configuration so far relative to its defaults.
    pub cost: f64,
}

/// What a unit's start measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Depth {
    /// The defaults and every form's start.
    Forms,
    /// The defaults alone; the first refinement measures the form starts.
    Defaults,
}

/// A unit's estimate of its start and conclusion, before it starts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Estimate {
    /// Planning the search and measuring the defaults, formation aside.
    pub defaults: Duration,
    /// Measuring every form's start, formation aside.
    pub forms: Duration,
    /// Programs the defaults need formed, and the form starts besides.
    pub default_programs: usize,
    pub form_programs: usize,
    /// Concluding the unit once started.
    pub conclusion: Duration,
}

/// What a start did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Started {
    pub progress: Progress,
    /// Programs formed, and what forming them took.
    pub programs: usize,
    pub forming: Duration,
    /// The start reached its hard limit before its planned work was done.
    pub cut: bool,
}

/// One tuning unit's search. Instants are times since the tuning began.
pub(crate) trait Unit {
    type Context;
    type Failure;
    /// The unit's share of expected step time at its defaults.
    fn share(&self) -> f64;
    fn estimate(&self) -> Estimate;
    /// Measure the defaults and, at [`Depth::Forms`], every form's start.
    /// `until` is the hard limit: the start returns by it with what it has.
    fn start(
        &mut self,
        context: &mut Self::Context,
        depth: Depth,
        until: Duration,
    ) -> Result<Started, Self::Failure>;
    /// Continue the search until `slice`, or until what remains before
    /// `until` only covers the unit's reserve.
    fn refine(
        &mut self,
        context: &mut Self::Context,
        slice: Duration,
        until: Duration,
    ) -> Result<Progress, Self::Failure>;
    /// What concluding the unit will take.
    fn reserve(&self) -> Duration;
    /// Confirm and validate the unit's choice, by `until`. A unit never
    /// started keeps its defaults.
    fn conclude(
        &mut self,
        context: &mut Self::Context,
        until: Duration,
    ) -> Result<(), Self::Failure>;
}

/// What the plan gave a unit's start, and what it took.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StartRecord {
    pub unit: usize,
    /// None: neither start fit, and the unit keeps its defaults unmeasured.
    pub depth: Option<Depth>,
    /// What the plan expected the start it gave to take, and a start with
    /// the form starts.
    pub estimated: Duration,
    pub with_forms: Duration,
    /// The time the plan had for this start and everything after it.
    pub available: Duration,
    pub actual: Duration,
    pub cut: bool,
}

/// A step that ended after the instant it was given.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Overrun {
    pub unit: usize,
    pub step: &'static str,
    pub by: Duration,
}

/// The plan as it was made and what happened: for the log.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Report {
    pub starts: Vec<StartRecord>,
    /// When the starts ended, the last refinement slice ended and the last
    /// conclusion ended.
    pub started: Duration,
    pub refined: Duration,
    pub concluded: Duration,
    /// The conclusion reserves when the first unit was concluded for want
    /// of time.
    pub reserved: Duration,
    pub slices: usize,
    pub overruns: Vec<Overrun>,
}

#[derive(Clone, Copy)]
enum State {
    Waiting,
    Open {
        progress: Progress,
        received: Duration,
        /// The unit's next refinement measures its form starts, at this
        /// price by the plan.
        starts: Option<Duration>,
    },
    Concluded,
}

/// What the open units other than `except` need to conclude.
fn reserved<U: Unit>(units: &[U], states: &[State], except: Option<usize>) -> Duration {
    units
        .iter()
        .zip(states)
        .enumerate()
        .filter(|(unit, (_, state))| Some(*unit) != except && matches!(state, State::Open { .. }))
        .map(|(_, (unit, _))| unit.reserve())
        .sum()
}

/// What the starts so far say of the units' estimates.
#[derive(Default)]
struct Observed {
    /// Programs formed and what forming them took.
    programs: usize,
    forming: Duration,
    /// What the starts took besides forming, and what their units estimated.
    spent: Duration,
    estimated: Duration,
}

impl Observed {
    /// The plan's price of a start of `estimate` at `depth`: the unit's
    /// estimate scaled by how the starts so far compared with theirs, and
    /// its programs at what forming one has taken so far.
    fn price(&self, estimate: &Estimate, depth: Depth) -> Duration {
        let (work, programs) = match depth {
            Depth::Defaults => (estimate.defaults, estimate.default_programs),
            Depth::Forms => (
                estimate.defaults + estimate.forms,
                estimate.default_programs + estimate.form_programs,
            ),
        };
        let scale = if self.estimated.is_zero() {
            1.
        } else {
            self.spent.as_secs_f64() / self.estimated.as_secs_f64()
        };
        let program = if self.programs == 0 {
            0.
        } else {
            self.forming.as_secs_f64() / self.programs as f64
        };
        Duration::from_secs_f64(work.as_secs_f64() * scale + programs as f64 * program)
    }

    fn conclusion(&self, estimate: &Estimate) -> Duration {
        estimate.conclusion
    }
}

struct Run<'r, U: Unit, C: Clock> {
    units: &'r mut [U],
    states: Vec<State>,
    clock: &'r C,
    total: Duration,
    report: Report,
}

impl<U: Unit, C: Clock> Run<'_, U, C> {
    /// Record a step of `unit` that ended after `until`.
    fn check(&mut self, unit: usize, step: &'static str, until: Duration) {
        let now = self.clock.now();
        if now > until {
            self.report.overruns.push(Overrun {
                unit,
                step,
                by: now - until,
            });
        }
    }

    fn conclude(&mut self, context: &mut U::Context, unit: usize) -> Result<(), U::Failure> {
        let until = self
            .total
            .saturating_sub(reserved(self.units, &self.states, Some(unit)));
        self.units[unit].conclude(context, until)?;
        self.states[unit] = State::Concluded;
        self.check(unit, "conclusion", until);
        Ok(())
    }
}

/// Tune `units` within `total`, refining in slices of `slice`.
pub(crate) fn allocate<U: Unit>(
    units: &mut [U],
    context: &mut U::Context,
    clock: &impl Clock,
    total: Duration,
    slice: Duration,
) -> Result<Report, U::Failure> {
    let mut run = Run {
        states: vec![State::Waiting; units.len()],
        units,
        clock,
        total,
        report: Report::default(),
    };
    // Largest share first; units of equal share in preparation order.
    let mut order = (0..run.units.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| run.units[*right].share().total_cmp(&run.units[*left].share()));
    let estimates = run.units.iter().map(Unit::estimate).collect::<Vec<_>>();
    let mut observed = Observed::default();
    for (position, &unit) in order.iter().enumerate() {
        let now = clock.now();
        // What is left for this start and everything after it: the time
        // less concluding the units already started.
        let available = total
            .saturating_sub(now)
            .saturating_sub(reserved(run.units, &run.states, None));
        // Every later unit keeps room for its defaults and its conclusion.
        let later = order[position + 1..]
            .iter()
            .map(|later| {
                observed.price(&estimates[*later], Depth::Defaults)
                    + observed.conclusion(&estimates[*later])
            })
            .sum::<Duration>();
        let conclusion = observed.conclusion(&estimates[unit]);
        let with_forms = observed.price(&estimates[unit], Depth::Forms);
        let defaults = observed.price(&estimates[unit], Depth::Defaults);
        let depth = if with_forms + conclusion + later <= available {
            Some(Depth::Forms)
        } else if defaults + conclusion <= available {
            Some(Depth::Defaults)
        } else {
            None
        };
        let mut record = StartRecord {
            unit,
            depth,
            estimated: match depth {
                Some(Depth::Forms) => with_forms,
                Some(Depth::Defaults) => defaults,
                None => Duration::ZERO,
            },
            with_forms,
            available,
            actual: Duration::ZERO,
            cut: false,
        };
        if let Some(depth) = depth {
            let until = total.saturating_sub(reserved(run.units, &run.states, None) + conclusion);
            let started = run.units[unit].start(context, depth, until)?;
            let actual = clock.now().saturating_sub(now);
            observed.programs += started.programs;
            observed.forming += started.forming;
            observed.spent += actual.saturating_sub(started.forming);
            observed.estimated += match depth {
                Depth::Forms => estimates[unit].defaults + estimates[unit].forms,
                Depth::Defaults => estimates[unit].defaults,
            };
            record.actual = actual;
            record.cut = started.cut;
            run.states[unit] = State::Open {
                progress: started.progress,
                received: Duration::ZERO,
                starts: (depth == Depth::Defaults).then(|| with_forms.saturating_sub(defaults)),
            };
            run.check(unit, "start", until);
        }
        run.report.starts.push(record);
    }
    run.report.started = clock.now();
    for &unit in &order {
        if matches!(run.states[unit], State::Open { progress, .. } if progress.finished) {
            run.conclude(context, unit)?;
        }
    }
    // The longest an ordinary refinement has run past the end of its slice:
    // what a unit cannot interrupt.
    let mut overshoot = Duration::ZERO;
    let mut short = None;
    run.report.refined = clock.now();
    loop {
        let now = clock.now();
        let reserve = reserved(run.units, &run.states, None);
        // Time received, with one more slice, per unit of step time still
        // taken.
        let served = |unit: usize| match run.states[unit] {
            State::Open {
                progress, received, ..
            } if !progress.finished => {
                let weight = run.units[unit].share() * progress.cost;
                Some((received + slice).as_secs_f64() / weight.max(f64::MIN_POSITIVE))
            }
            _ => None,
        };
        // What the unit's next refinement cannot interrupt.
        let guard = |unit: usize| match run.states[unit] {
            State::Open {
                starts: Some(price),
                ..
            } => price,
            State::Open { .. } => overshoot,
            State::Waiting | State::Concluded => Duration::ZERO,
        };
        let open = order
            .iter()
            .filter_map(|unit| served(*unit).map(|served| (*unit, served)))
            .collect::<Vec<_>>();
        let fitting = open
            .iter()
            .filter(|(unit, _)| now + reserve + guard(*unit) < total)
            .copied()
            .collect::<Vec<_>>();
        // A unit owed its form starts first, largest share first; then the
        // unit a slice leaves least served.
        let owed = fitting.iter().find(|(unit, _)| {
            matches!(run.states[*unit], State::Open { starts: Some(_), .. })
        });
        let Some((unit, _)) = owed
            .or_else(|| {
                fitting
                    .iter()
                    .min_by(|left, right| left.1.total_cmp(&right.1))
            })
            .copied()
        else {
            // No unit's slice fits before the reserves. Conclude the unit
            // that has received the most for its weight: what its conclusion
            // leaves of its reserve is dealt to the rest.
            let Some((unit, _)) = open
                .iter()
                .max_by(|left, right| left.1.total_cmp(&right.1))
                .copied()
            else {
                break;
            };
            short.get_or_insert(reserve);
            run.conclude(context, unit)?;
            continue;
        };
        let until = total.saturating_sub(reserved(run.units, &run.states, Some(unit)));
        let end = (now + slice).min(until.saturating_sub(guard(unit)));
        let progress = run.units[unit].refine(context, end, until)?;
        run.report.slices += 1;
        let over = clock.now().saturating_sub(end);
        run.check(
            unit,
            "refinement",
            total.saturating_sub(reserved(run.units, &run.states, None)),
        );
        let State::Open {
            received, starts, ..
        } = run.states[unit]
        else {
            unreachable!("the unit refined is open");
        };
        // The step that measured the form starts is not a slice like the
        // ones after it.
        if starts.is_none() {
            overshoot = overshoot.max(over);
        }
        run.states[unit] = State::Open {
            progress,
            received: received + clock.now().saturating_sub(now),
            starts: None,
        };
        run.report.refined = clock.now();
        if progress.finished {
            run.conclude(context, unit)?;
        }
    }
    run.report.reserved = short.unwrap_or_else(|| reserved(run.units, &run.states, None));
    for unit in 0..run.units.len() {
        if matches!(run.states[unit], State::Waiting) {
            run.conclude(context, unit)?;
        }
    }
    run.report.concluded = clock.now();
    Ok(run.report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    #[derive(Clone, Default)]
    struct Fake(Rc<Cell<Duration>>);

    impl Fake {
        fn pass(&self, time: Duration) {
            self.0.set(self.0.get() + time);
        }
    }

    impl Clock for Fake {
        fn now(&self) -> Duration {
            self.0.get()
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Step {
        Start(Depth),
        Refine,
        Conclude,
    }

    /// A unit whose search is `work` of measurement, each measurement
    /// taking `step`. Its start measures its defaults in `defaults` and its
    /// form starts in `forms`, forming `programs` programs for each at
    /// `forming` a program; concluding takes `confirm`. It estimates each of
    /// those at `believed` times what they take. Its best cost falls from 1
    /// to `floor` as the work is done.
    struct Synthetic {
        name: usize,
        clock: Fake,
        share: f64,
        defaults: Duration,
        forms: Duration,
        programs: usize,
        forming: Duration,
        believed: f64,
        step: Duration,
        work: Duration,
        confirm: Duration,
        /// What the unit reserves for its conclusion, when not `confirm`.
        reserved: Option<Duration>,
        floor: f64,
        started: Option<Depth>,
        cut: bool,
        done: Duration,
        concluded: Option<Duration>,
    }

    type Log = RefCell<Vec<(usize, Step, Duration)>>;

    const SECOND: Duration = Duration::from_secs(1);

    impl Synthetic {
        fn new(name: usize, clock: &Fake, share: f64, work: u64) -> Self {
            Self {
                name,
                clock: clock.clone(),
                share,
                defaults: SECOND / 2,
                forms: SECOND / 2,
                programs: 0,
                forming: Duration::ZERO,
                believed: 1.,
                step: Duration::from_millis(100),
                work: Duration::from_secs(work),
                confirm: Duration::from_millis(500),
                reserved: None,
                floor: 0.5,
                started: None,
                cut: false,
                done: Duration::ZERO,
                concluded: None,
            }
        }

        fn progress(&self) -> Progress {
            let done = self.done.as_secs_f64() / self.work.as_secs_f64().max(f64::MIN_POSITIVE);
            Progress {
                finished: self.done >= self.work,
                cost: 1. - (1. - self.floor) * done.min(1.),
            }
        }

        /// Spend `time` in steps, stopping at the hard limit; whether all
        /// of it was spent.
        fn spend(&self, time: Duration, until: Duration) -> bool {
            let mut spent = Duration::ZERO;
            while spent < time {
                if self.clock.now() >= until {
                    return false;
                }
                let step = self.step.min(time - spent);
                self.clock.pass(step);
                spent += step;
            }
            true
        }
    }

    impl Unit for Synthetic {
        type Context = Log;
        type Failure = String;

        fn share(&self) -> f64 {
            self.share
        }

        fn estimate(&self) -> Estimate {
            Estimate {
                defaults: self.defaults.mul_f64(self.believed),
                forms: self.forms.mul_f64(self.believed),
                default_programs: self.programs,
                form_programs: self.programs,
                conclusion: self.confirm,
            }
        }

        fn start(&mut self, log: &mut Log, depth: Depth, until: Duration) -> Result<Started, String> {
            log.borrow_mut()
                .push((self.name, Step::Start(depth), self.clock.now()));
            let programs = match depth {
                Depth::Defaults => self.programs,
                Depth::Forms => 2 * self.programs,
            };
            let forming = self.forming * programs as u32;
            let work = match depth {
                Depth::Defaults => self.defaults,
                Depth::Forms => self.defaults + self.forms,
            };
            let complete = self.spend(forming + work, until);
            self.started = Some(depth);
            self.cut = !complete;
            Ok(Started {
                progress: self.progress(),
                programs,
                forming,
                cut: !complete,
            })
        }

        fn refine(
            &mut self,
            log: &mut Log,
            slice: Duration,
            until: Duration,
        ) -> Result<Progress, String> {
            assert!(self.started.is_some() && self.concluded.is_none());
            log.borrow_mut()
                .push((self.name, Step::Refine, self.clock.now()));
            // A measurement begun is finished: a slice can overrun by one.
            while self.done < self.work
                && self.clock.now() < slice
                && self.clock.now() + self.reserve() < until
            {
                self.clock.pass(self.step);
                self.done += self.step;
            }
            Ok(self.progress())
        }

        fn reserve(&self) -> Duration {
            if self.started.is_some() && self.concluded.is_none() {
                self.reserved.unwrap_or(self.confirm)
            } else {
                Duration::ZERO
            }
        }

        fn conclude(&mut self, log: &mut Log, _until: Duration) -> Result<(), String> {
            log.borrow_mut()
                .push((self.name, Step::Conclude, self.clock.now()));
            if self.started.is_some() {
                self.clock.pass(self.confirm);
            }
            self.concluded = Some(self.clock.now());
            Ok(())
        }
    }

    const SLICE: Duration = Duration::from_millis(500);

    fn run(
        units: &mut [Synthetic],
        clock: &Fake,
        total: u64,
    ) -> (Vec<(usize, Step, Duration)>, Report) {
        let mut log = Log::default();
        let report = allocate(units, &mut log, clock, Duration::from_secs(total), SLICE).unwrap();
        (log.into_inner(), report)
    }

    fn depths(report: &Report) -> Vec<(usize, Option<Depth>)> {
        report
            .starts
            .iter()
            .map(|start| (start.unit, start.depth))
            .collect()
    }

    /// A breadth phase that fits: every unit gets its form starts, largest
    /// share first, refinement fills the rest by step time left, and tuning
    /// ends inside the time with no step cut and none late.
    #[test]
    fn a_breadth_phase_that_fits_starts_every_form_and_refinement_fills_the_rest() {
        let clock = Fake::default();
        // Preparation order is not share order; the searches are far longer
        // than the time.
        let mut units = [0.1, 0.5, 0.15, 0.25]
            .into_iter()
            .enumerate()
            .map(|(name, share)| Synthetic {
                floor: 1.,
                ..Synthetic::new(name, &clock, share, 1000)
            })
            .collect::<Vec<_>>();
        let (log, report) = run(&mut units, &clock, 60);
        assert_eq!(
            depths(&report),
            [1, 3, 2, 0].map(|unit| (unit, Some(Depth::Forms)))
        );
        assert!(log[..4]
            .iter()
            .all(|(_, step, _)| *step == Step::Start(Depth::Forms)));
        assert!(log[4..]
            .iter()
            .all(|(_, step, _)| !matches!(step, Step::Start(_))));
        // The plan's estimates were what the starts took.
        assert!(report
            .starts
            .iter()
            .all(|start| start.estimated == start.actual && !start.cut));
        assert!(report.overruns.is_empty());
        assert_eq!(report.started, Duration::from_secs(4));
        // Refinement ran until only the four conclusions were left, and
        // they ended with the time.
        assert_eq!(report.refined, Duration::from_secs(58));
        assert_eq!(report.reserved, Duration::from_secs(2));
        assert_eq!(report.concluded, Duration::from_secs(60));
        // Its time followed the shares: the searches found nothing, and
        // their measurements cost the same.
        let refined = units
            .iter()
            .map(|unit| unit.done.as_secs_f64())
            .collect::<Vec<_>>();
        for (unit, share) in [0.1, 0.5, 0.15, 0.25].into_iter().enumerate() {
            assert!(
                (refined[unit] / 54. - share).abs() < 0.02,
                "unit {unit}: {refined:?}"
            );
        }
        // Nothing is refined once conclusions begin.
        let first = log
            .iter()
            .position(|(_, step, _)| *step == Step::Conclude)
            .unwrap();
        assert!(log[first..]
            .iter()
            .all(|(_, step, _)| *step == Step::Conclude));
    }

    /// A breadth phase that does not fit (a cold pipeline cache: every
    /// program takes a second to form): the largest units get their form
    /// starts while the rest keep room for their defaults, no step is cut,
    /// and tuning ends inside the time.
    #[test]
    fn a_breadth_phase_that_does_not_fit_gives_form_starts_by_share() {
        let clock = Fake::default();
        let mut units = [0.05, 0.4, 0.1, 0.3, 0.15]
            .into_iter()
            .enumerate()
            .map(|(name, share)| Synthetic {
                programs: 4,
                forming: SECOND,
                floor: 1.,
                ..Synthetic::new(name, &clock, share, 1000)
            })
            .collect::<Vec<_>>();
        // Defaults-only: 4 programs and half a second, 4.5 s; with forms 9 s;
        // five of the latter and their conclusions are 47.5 s.
        let (log, report) = run(&mut units, &clock, 40);
        // The first start is planned before any program was formed, at the
        // units' estimates alone, and shows what a program costs; from the
        // second on the plan prices them. Three units' form starts fit with
        // the defaults of the two after them.
        assert_eq!(
            depths(&report),
            [
                (1, Some(Depth::Forms)),
                (3, Some(Depth::Forms)),
                (4, Some(Depth::Forms)),
                (2, Some(Depth::Defaults)),
                (0, Some(Depth::Defaults)),
            ]
        );
        assert!(report.starts[1..]
            .iter()
            .all(|start| start.estimated == start.actual));
        assert!(report.starts.iter().all(|start| !start.cut));
        assert!(report.overruns.is_empty(), "{:?}", report.overruns);
        assert!(report.concluded <= Duration::from_secs(40));
        assert!(units.iter().all(|unit| unit.concluded.is_some() && !unit.cut));
        // What was left went to refinement.
        assert!(report.refined > report.started);
        assert!(log.iter().any(|(_, step, _)| *step == Step::Refine));
    }

    /// Time for no more than some units' defaults: the smallest units keep
    /// their defaults unmeasured, and no start is begun that cannot end.
    #[test]
    fn units_whose_defaults_do_not_fit_are_not_started() {
        let clock = Fake::default();
        let mut units = [0.1, 0.5, 0.4]
            .into_iter()
            .enumerate()
            .map(|(name, share)| Synthetic {
                defaults: SECOND,
                forms: SECOND,
                ..Synthetic::new(name, &clock, share, 1000)
            })
            .collect::<Vec<_>>();
        let (_, report) = run(&mut units, &clock, 3);
        assert_eq!(
            depths(&report),
            [
                (1, Some(Depth::Defaults)),
                (2, Some(Depth::Defaults)),
                (0, None)
            ]
        );
        assert!(units.iter().all(|unit| unit.concluded.is_some()));
        assert!(report.overruns.is_empty());
        assert_eq!(report.concluded, Duration::from_secs(3));
    }

    /// Estimates wrong by a factor of two: the first start shows it, the
    /// plan for the rest is made from what it took, and tuning still ends
    /// inside the time.
    #[test]
    fn estimates_wrong_by_half_are_corrected_from_what_the_starts_took() {
        let clock = Fake::default();
        let mut units = [0.4, 0.3, 0.2, 0.1]
            .into_iter()
            .enumerate()
            .map(|(name, share)| Synthetic {
                defaults: SECOND * 2,
                forms: SECOND * 2,
                believed: 0.5,
                floor: 1.,
                ..Synthetic::new(name, &clock, share, 1000)
            })
            .collect::<Vec<_>>();
        // By the units' estimates four full starts and their conclusions
        // take 10 s; they take 18.
        let (_, report) = run(&mut units, &clock, 13);
        assert_eq!(report.starts[0].depth, Some(Depth::Forms));
        assert_eq!(report.starts[0].estimated, SECOND * 2);
        assert_eq!(report.starts[0].actual, SECOND * 4);
        // From the second start on the plan prices starts at twice the
        // units' estimates, and gives what fits.
        assert!(report.starts[1..]
            .iter()
            .all(|start| start.estimated == start.actual));
        assert_eq!(
            depths(&report)[1..],
            [
                (1, Some(Depth::Defaults)),
                (2, Some(Depth::Defaults)),
                (3, Some(Depth::Defaults))
            ]
        );
        assert!(report.overruns.is_empty(), "{:?}", report.overruns);
        assert!(report.concluded <= Duration::from_secs(13));
    }

    /// A start that takes far longer than its estimate and the time left:
    /// the hard limit ends it, the cut is reported, and tuning ends at the
    /// deadline with every unit concluded.
    #[test]
    fn a_start_far_longer_than_the_time_is_cut_at_the_deadline_and_reported() {
        let clock = Fake::default();
        let mut units = [
            Synthetic::new(0, &clock, 0.2, 1000),
            Synthetic {
                // Believed to take a tenth of a second; takes a hundred.
                defaults: SECOND * 100,
                believed: 0.001,
                ..Synthetic::new(1, &clock, 0.5, 1000)
            },
            Synthetic::new(2, &clock, 0.3, 1000),
        ];
        let (_, report) = run(&mut units, &clock, 10);
        let cut = report.starts[0];
        assert_eq!((cut.unit, cut.cut), (1, true));
        assert!(cut.actual > cut.estimated * 10);
        // The start returned at its limit: the time less its own
        // conclusion.
        assert_eq!(cut.actual, Duration::from_millis(9500));
        // Nothing else fit; every unit holds a choice, and tuning ended
        // with the time.
        assert_eq!(
            depths(&report)[1..],
            [(2, None), (0, None)]
        );
        assert!(units.iter().all(|unit| unit.concluded.is_some()));
        assert!(report.overruns.is_empty());
        assert_eq!(report.concluded, Duration::from_secs(10));
    }

    #[test]
    fn refinement_follows_the_step_time_each_unit_still_takes() {
        // A unit that has recovered most of its time yields to one that has
        // not: equal shares, one search cutting its cost to a fifth.
        let clock = Fake::default();
        let mut units = [
            Synthetic {
                floor: 1.,
                ..Synthetic::new(0, &clock, 0.5, 1000)
            },
            Synthetic {
                floor: 0.2,
                ..Synthetic::new(1, &clock, 0.5, 20)
            },
        ];
        run(&mut units, &clock, 33);
        assert!(units[0].done > units[1].done);
        // Equal shares and gains: the unit whose measurements cost four
        // times as much gets the same time, and a quarter of the
        // measurements in it.
        let clock = Fake::default();
        let mut units = [
            Synthetic {
                step: Duration::from_millis(400),
                floor: 1.,
                ..Synthetic::new(0, &clock, 0.5, 1000)
            },
            Synthetic {
                floor: 1.,
                ..Synthetic::new(1, &clock, 0.5, 1000)
            },
        ];
        run(&mut units, &clock, 53);
        let (slow, fast) = (units[0].done.as_secs_f64(), units[1].done.as_secs_f64());
        assert!((fast / slow - 1.).abs() < 0.1, "{slow} {fast}");
    }

    /// Reserves far above what the conclusions take (units with nothing to
    /// confirm): a unit is concluded when the reserves fill the time left,
    /// what its conclusion did not use is dealt to the rest, and tuning uses
    /// the time instead of ending with the reserves unspent.
    #[test]
    fn a_reserve_a_conclusion_does_not_use_is_dealt_to_the_units_still_open() {
        let clock = Fake::default();
        let mut units = [0.5, 0.3, 0.2]
            .into_iter()
            .enumerate()
            .map(|(name, share)| Synthetic {
                confirm: Duration::from_millis(100),
                reserved: Some(SECOND * 5),
                floor: 1.,
                ..Synthetic::new(name, &clock, share, 1000)
            })
            .collect::<Vec<_>>();
        let (log, report) = run(&mut units, &clock, 30);
        // The three reserves are half the time: the first unit is concluded
        // with 15 s left.
        let first = log
            .iter()
            .position(|(_, step, _)| *step == Step::Conclude)
            .unwrap();
        assert!(log[first].2 <= Duration::from_secs(15), "{:?}", log[first]);
        assert_eq!(report.reserved, SECOND * 15);
        // The others were refined on after it, and the last unit until only
        // its own reserve was left.
        assert!(log[first + 1..]
            .iter()
            .any(|(_, step, _)| *step == Step::Refine));
        assert!(report.refined >= Duration::from_millis(24_500), "{report:?}");
        assert!(report.concluded <= Duration::from_secs(30));
        assert!(report.overruns.is_empty(), "{:?}", report.overruns);
        assert!(units.iter().all(|unit| unit.concluded.is_some()));
        // Refined 3 s of the first 15 by the old rule, the units now hold
        // most of the time.
        let refined = units.iter().map(|unit| unit.done).sum::<Duration>();
        assert!(refined >= Duration::from_secs(20), "{refined:?}");
    }

    /// Slow steps (every slice runs seconds past its end) and many small
    /// units: the time still follows the shares, the small units' first
    /// slices do not take it from the large unit.
    #[test]
    fn a_round_of_first_slices_does_not_take_the_time_from_the_largest_unit() {
        let clock = Fake::default();
        let mut shares = vec![0.5];
        shares.extend([0.05; 10]);
        let mut units = shares
            .into_iter()
            .enumerate()
            .map(|(name, share)| Synthetic {
                step: SECOND * 2,
                floor: 1.,
                ..Synthetic::new(name, &clock, share, 1000)
            })
            .collect::<Vec<_>>();
        run(&mut units, &clock, 60);
        // Half the step: about half of the refinement, not one slice in
        // eleven.
        let refined = units.iter().map(|unit| unit.done).sum::<Duration>();
        let largest = units[0].done.as_secs_f64() / refined.as_secs_f64();
        assert!(largest > 0.35, "{largest}: {refined:?}");
    }

    #[test]
    fn finished_searches_conclude_at_once_and_their_time_goes_to_the_rest() {
        let clock = Fake::default();
        let mut units = [
            Synthetic::new(0, &clock, 0.4, 2),
            Synthetic::new(1, &clock, 0.4, 1000),
            Synthetic::new(2, &clock, 0.2, 3),
        ];
        let (log, report) = run(&mut units, &clock, 60);
        // The short searches ran to their ends and concluded then, not at
        // the end of the time.
        for unit in [0, 2] {
            assert_eq!(units[unit].done, units[unit].work);
            assert!(units[unit].concluded.unwrap() < Duration::from_secs(30));
            let concluded = log
                .iter()
                .position(|(name, step, _)| *name == unit && *step == Step::Conclude)
                .unwrap();
            assert!(log[concluded + 1..].iter().all(|(name, _, _)| *name != unit));
        }
        // The long one got everything else: 60 s less three starts, three
        // conclusions and the two short searches.
        assert!(units[1].done >= Duration::from_secs(50), "{:?}", units[1].done);
        assert!(report.concluded <= Duration::from_secs(60));
        assert!(report.overruns.is_empty());
    }

    #[test]
    fn with_time_for_everything_every_search_runs_to_its_end() {
        let clock = Fake::default();
        let mut units = [0.2, 0.5, 0.3]
            .into_iter()
            .enumerate()
            .map(|(name, share)| Synthetic::new(name, &clock, share, 4 + name as u64))
            .collect::<Vec<_>>();
        let (log, report) = run(&mut units, &clock, 600);
        assert!(units.iter().all(|unit| unit.done == unit.work));
        // Starts, the searches and the conclusions: nothing waits for the
        // time to end.
        assert_eq!(report.concluded, Duration::from_millis(3000 + 15000 + 1500));
        assert_eq!(
            log.iter()
                .filter(|(_, step, _)| *step == Step::Conclude)
                .count(),
            3
        );
    }
}
