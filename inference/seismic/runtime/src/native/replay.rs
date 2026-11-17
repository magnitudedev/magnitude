//! Replay of the search against a recorded survey (tuning spec §E2).
//!
//! A survey ([`super::tune::Strategy::Survey`]) records, at each of its
//! points, the work each assignment of the parameters that decide the work
//! there runs, with every sample of every work ([`super::survey`]); a
//! configuration's measurement is composed from the work it runs at each
//! point. A replay runs the production search ([`GroupSearch`]) over the
//! record many times, planned as the unit's search plans it (the record's
//! [`SurveyRecord::plan`]): the record's domain is the domain the search
//! asks ([`plan::Domain`]: what it admits is what the survey recorded at
//! every point), and each simulated measurement draws samples with
//! replacement from the recorded samples of the work the configuration runs
//! at each point, as many as the search would take, so the recorded noise
//! is reproduced without modeling it. Nothing is produced per
//! configuration of the domain, which is the product of the launches'
//! domains: a configuration exists when the search asks for it, and the
//! record answers it point by point.
//!
//! What the order asked for is then converted as a unit converts its leads:
//! the cheapest measured challenger whose cost is below the served choice's
//! is re-measured with it under matched draws and replaces it when
//! its rounds resolve a lead; one that does not gives way to the next. A unit also
//! holds a lead back until it exceeds the error its conversions have
//! observed; a replay converts every nominal lead, which is what a unit
//! does before it has observed any.
//!
//! The *true* time of a work is the median of all its recorded samples, and
//! a configuration's true cost is the production objective over the true
//! times of the work it runs ([`Cost::relative`]): a sum over the points,
//! so the true best is found as the survey finds its choice
//! ([`SurveyIndex::best`]). Each replay reports the chosen configuration's
//! true cost relative to the true best, how fast the search converges, and,
//! per point, how much the configuration that is best overall loses to the
//! work best at that point alone (the gap a per-size launch could recover).
//!
//! Both modes minimize the same invocation-demand elapsed-service objective.
//! [`Objective::Keyed`] shares identical physical point evidence; [`Objective::Separate`]
//! samples each configuration independently.

use super::admission::admit;
use super::plan::{self, ParameterAddress, PlanError, Subsets};
use super::search::{Cost, LaunchKey, ParameterValues, SearchSettings};
use super::survey::{SurveyIndex, SurveyRecord, WorkOutcome};
use super::tune::{GroupSearch, PointMeasurement, TuningMethod, TuningResult};
use seismic_lang::checked::NativeSpecialization;
use seismic_measure::{median, Tally, Verdict};
use std::cell::RefCell;
use std::collections::HashMap;

/// How the replayed evaluator costs configurations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Objective {
    /// Global expected service relative to the same canonical demand, with
    /// one measurement per physical point/key.
    Keyed,
    /// The same objective with independent samples for every
    /// configuration: no point's measurement is shared.
    Separate,
}

/// One work a point recorded that can serve.
#[derive(Clone, Debug)]
struct RecordedWork<'r> {
    key: &'r LaunchKey,
    samples: &'r [f64],
    /// The median of every recorded sample.
    time: f64,
}

/// A survey as the replay reads it. A configuration is held as its
/// assignment: its value of each surveyed parameter, in the record's
/// order ([`SurveyIndex`]).
pub struct Recording<'r> {
    record: &'r SurveyRecord,
    index: SurveyIndex<'r>,
    /// Each surveyed parameter's address, in the record's order.
    addresses: Vec<ParameterAddress>,
    /// Per point, per work there: what it recorded; none when it cannot
    /// serve (the survey excluded it, or the record's requirement does not
    /// admit it).
    works: Vec<Vec<Option<RecordedWork<'r>>>>,
    weights: Vec<f64>,
    labels: Vec<String>,
    /// The defaults' true time at each point: the reference of every cost.
    reference: Vec<f64>,
    /// The configuration of least true cost, and that cost.
    best: (Vec<u64>, f64),
}

#[derive(Debug)]
pub enum RecordingError {
    /// The result is not a survey's.
    NotASurvey,
    /// A measured work lacks samples at a point.
    Empty { point: String },
    /// The survey could not measure the defaults.
    NoMeasuredDefaults,
}

impl std::fmt::Display for RecordingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotASurvey => f.write_str("the result is not a survey's"),
            Self::Empty { point } => {
                write!(f, "a surveyed work has no samples at `{point}`")
            }
            Self::NoMeasuredDefaults => f.write_str("the survey did not measure the defaults"),
        }
    }
}

impl std::error::Error for RecordingError {}

impl<'r> Recording<'r> {
    pub fn new(result: &'r TuningResult) -> Result<Self, RecordingError> {
        let TuningMethod::Survey { record, .. } = &result.method else {
            return Err(RecordingError::NotASurvey);
        };
        let index = SurveyIndex::new(record);
        let addresses = record
            .parameters
            .iter()
            .map(|parameter| match parameter.launch {
                None => ParameterAddress::Entry(parameter.name.clone()),
                Some(ordinal) => ParameterAddress::Launch { ordinal, name: parameter.name.clone() },
            })
            .collect();
        let works = record
            .points
            .iter()
            .map(|point| {
                point
                    .work
                    .iter()
                    .map(|work| match &work.outcome {
                        WorkOutcome::Excluded(_) => Ok(None),
                        // A survey times work whatever its verdict; work the
                        // record's requirement does not admit cannot serve.
                        WorkOutcome::Measured(_)
                            if work.observation.is_some_and(|observation| {
                                admit(&result.observations[observation], &result.requirement).is_err()
                            }) =>
                        {
                            Ok(None)
                        }
                        WorkOutcome::Measured(measurement) if measurement.samples.is_empty() => {
                            Err(RecordingError::Empty { point: measurement.point.clone() })
                        }
                        WorkOutcome::Measured(measurement) => Ok(Some(RecordedWork {
                            key: &measurement.key,
                            samples: &measurement.samples,
                            time: median(&measurement.samples),
                        })),
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let weights = result.points.iter().map(|point| point.weight).collect::<Vec<_>>();
        let labels = result.points.iter().map(|point| point.label.clone()).collect();
        let mut recording = Self {
            record,
            index,
            addresses,
            works,
            weights,
            labels,
            reference: Vec::new(),
            best: (Vec::new(), f64::INFINITY),
        };
        // The defaults are the reference of every cost.
        recording.reference = recording
            .served(recording.index.defaults())
            .ok_or(RecordingError::NoMeasuredDefaults)?
            .iter()
            .map(|work| work.time)
            .collect();
        // The true cost is a sum over the points of each one's demand times
        // its time, over the defaults' (a constant): the least is the
        // least sum.
        let best = recording
            .index
            .best(|point, work| {
                let time = recording.works[point][work].as_ref()?.time;
                Some(if recording.weights[point] > 0. { recording.weights[point] * time } else { 0. })
            })
            .expect("the defaults serve at every point");
        let best = recording.index.assignment(&best).expect("a configuration of the record values every parameter");
        recording.best = (best.clone(), recording.truth(&best));
        Ok(recording)
    }

    /// The configurations of the record's domain.
    pub fn len(&self) -> u128 {
        self.index.count(|_, _| true)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The configurations of the record's domain that run, at every point,
    /// measured work the record's requirement admits.
    pub fn measured(&self) -> u128 {
        self.index.count(|point, work| self.works[point][work].is_some())
    }

    /// The parameter values of the configuration of least true cost.
    pub fn best(&self) -> ParameterValues {
        self.values(&self.best.0)
    }

    /// `assignment`'s values by name: an entry parameter's own, a launch's
    /// `L{launch}.{name}`.
    fn values(&self, assignment: &[u64]) -> ParameterValues {
        self.addresses
            .iter()
            .zip(assignment)
            .map(|(address, value)| {
                let name = match address {
                    ParameterAddress::Entry(name) => name.clone(),
                    ParameterAddress::Launch { ordinal, name } => format!("L{ordinal}.{name}"),
                };
                (name, *value)
            })
            .collect()
    }

    /// The work `assignment` runs at each point; none when one of it cannot
    /// serve. `assignment` is of the record's domain.
    fn served(&self, assignment: &[u64]) -> Option<Vec<&RecordedWork<'r>>> {
        (0..self.labels.len())
            .map(|point| {
                let work = self.index.work_at(point, assignment).expect("a configuration of the record's domain");
                self.works[point][work].as_ref()
            })
            .collect()
    }

    /// The true cost of `assignment` (infinite when it cannot serve).
    fn truth(&self, assignment: &[u64]) -> f64 {
        match self.served(assignment) {
            None => f64::INFINITY,
            Some(works) => Cost::relative(
                works.iter().map(|work| work.key.clone()).collect(),
                &self.weights,
                &works.iter().map(|work| work.time).collect::<Vec<_>>(),
                &self.reference,
            )
            .total(),
        }
    }

    /// True cost of `assignment` relative to the true best (0 = the best).
    fn excess(&self, assignment: &[u64]) -> f64 {
        self.truth(assignment) / self.best.1 - 1.0
    }

    /// Per point: the work best there alone, and how much slower the
    /// overall best configuration is at that point.
    pub fn gaps(&self) -> Vec<PointGap> {
        let overall = self.served(&self.best.0).expect("the best configuration serves");
        self.labels
            .iter()
            .enumerate()
            .map(|(point, label)| {
                let (local, time) = self.works[point]
                    .iter()
                    .enumerate()
                    .filter_map(|(work, recorded)| recorded.as_ref().map(|recorded| (work, recorded.time)))
                    .min_by(|left, right| left.1.total_cmp(&right.1))
                    .expect("the defaults' work serves");
                let local = self
                    .index
                    .assignment(&self.record.points[point].work[local].configuration)
                    .expect("a work's configuration values every parameter");
                PointGap {
                    label: label.clone(),
                    weight: self.weights[point],
                    shared_seconds: overall[point].time,
                    local_seconds: time,
                    local: self.values(&local),
                }
            })
            .collect()
    }

    /// The position of `address` among the surveyed parameters. The
    /// record's plan names only surveyed parameters.
    fn coordinate(&self, address: &ParameterAddress) -> usize {
        self.addresses
            .iter()
            .position(|surveyed| surveyed == address)
            .expect("the record's plan names only surveyed parameters")
    }

    /// `specialization`'s assignment, into `assignment`.
    fn assign(&self, specialization: &NativeSpecialization, assignment: &mut Vec<u64>) {
        assignment.clear();
        assignment.extend(self.addresses.iter().map(|address| address.value(specialization)));
    }

    /// The configuration of `assignment`.
    fn specialization(&self, assignment: &[u64]) -> NativeSpecialization {
        self.index.configuration(assignment).specialization()
    }

    /// The assignments of the record's domain that are `base` with each of
    /// `coordinates` changed to another surveyed value, strictly after the
    /// values `after` of them when given, in the declaration's walk order
    /// (`coordinates` in the order given, each over its values in
    /// declaration order, the last fastest), until `visit` returns false.
    /// A prefix is not extended once a point all of whose changed
    /// parameters it assigns holds no row of it: the walk follows what the
    /// record holds, not the product of the coordinates' values.
    fn changes(
        &self,
        base: &[u64],
        coordinates: &[usize],
        after: Option<&[u64]>,
        mut visit: impl FnMut(&[u64]) -> bool,
    ) {
        let parameters = &self.record.parameters;
        // A coordinate held at a value the survey did not take is of no
        // configuration of the domain.
        let held = (0..base.len())
            .filter(|coordinate| !coordinates.contains(coordinate))
            .all(|coordinate| parameters[coordinate].values.contains(&base[coordinate]));
        if !held {
            return;
        }
        // Each point is decided at the depth that assigns the last of its
        // changed parameters; one that reads none of them, at once.
        let mut decided = vec![Vec::new(); coordinates.len() + 1];
        for (point, surveyed) in self.record.points.iter().enumerate() {
            let depth = coordinates
                .iter()
                .rposition(|coordinate| surveyed.leading.contains(coordinate))
                .map_or(0, |depth| depth + 1);
            decided[depth].push(point);
        }
        let holds = |assignment: &[u64], depth: usize| {
            decided[depth].iter().all(|&point| self.index.work_at(point, assignment).is_some())
        };
        if !holds(base, 0) {
            return;
        }
        let after = after.map(|after| {
            coordinates
                .iter()
                .zip(after)
                .map(|(&coordinate, value)| ordinal(&parameters[coordinate].values, *value))
                .collect::<Vec<_>>()
        });
        let mut assignment = base.to_vec();
        // The walk, as a stack of each depth's next ordinal; `tight` while
        // the prefix equals `after`'s.
        let mut next = vec![0; coordinates.len()];
        let mut tight = vec![after.is_some(); coordinates.len() + 1];
        if let (Some(after), Some(first)) = (&after, next.first_mut()) {
            *first = after[0];
        }
        if coordinates.is_empty() {
            // `base` itself, which no `after` precedes.
            if after.is_none() {
                visit(base);
            }
            return;
        }
        let mut depth = 0;
        loop {
            let coordinate = coordinates[depth];
            let values = &parameters[coordinate].values;
            let Some(at) = (next[depth]..values.len()).find(|&at| values[at] != base[coordinate]) else {
                // This depth is done: back up.
                assignment[coordinate] = base[coordinate];
                if depth == 0 {
                    return;
                }
                depth -= 1;
                continue;
            };
            next[depth] = at + 1;
            assignment[coordinate] = values[at];
            let tight_here = tight[depth] && after.as_ref().is_some_and(|after| after[depth] == at);
            if !holds(&assignment, depth + 1) {
                continue;
            }
            if depth + 1 == coordinates.len() {
                // Strictly after `after`: its own assignment is skipped.
                if !tight_here && !visit(&assignment) {
                    return;
                }
                continue;
            }
            depth += 1;
            tight[depth] = tight_here;
            next[depth] = match &after {
                Some(after) if tight_here => after[depth],
                _ => 0,
            };
        }
    }
}

/// Where `value` stands in `values`.
fn ordinal(values: &[u64], value: u64) -> usize {
    values.iter().position(|declared| *declared == value).expect("a surveyed value")
}

/// A recording as the domain of a search: it admits what the survey
/// recorded at every point, and a configuration is forecast to run the
/// work the survey recorded it running.
struct Replayed<'a, 'r> {
    recording: &'a Recording<'r>,
    objective: Objective,
    /// The assignment each question is answered at, reused across them.
    assignment: RefCell<Vec<u64>>,
}

impl<'a, 'r> Replayed<'a, 'r> {
    fn new(recording: &'a Recording<'r>, objective: Objective) -> Self {
        Self {
            recording,
            objective,
            assignment: RefCell::new(Vec::with_capacity(recording.addresses.len())),
        }
    }

    /// The key `assignment` is measured under at `point`: the recorded
    /// one, or under independent sampling one no other configuration has.
    /// A configuration that cannot serve has none recorded: it is asked
    /// for, and found excluded.
    fn key(&self, assignment: &[u64], point: usize) -> LaunchKey {
        let recorded = (self.objective == Objective::Keyed)
            .then(|| self.recording.served(assignment))
            .flatten();
        match recorded {
            Some(works) => works[point].key.clone(),
            None => LaunchKey {
                launches: Vec::new(),
                values: self.recording.values(assignment),
            },
        }
    }
}

impl plan::Domain for Replayed<'_, '_> {
    fn place(
        &self,
        base: &NativeSpecialization,
        parameters: &[ParameterAddress],
        values: &[u64],
    ) -> Result<Option<NativeSpecialization>, PlanError> {
        let recording = self.recording;
        let mut moved = self.assignment.borrow_mut();
        recording.assign(base, &mut moved);
        for (address, value) in parameters.iter().zip(values) {
            moved[recording.coordinate(address)] = *value;
        }
        Ok(recording.index.holds(&moved).then(|| recording.specialization(&moved)))
    }

    /// The configurations of the record's domain that change from `base`
    /// exactly `parameters`, in the declaration's walk order
    /// ([`Recording::changes`]).
    fn admissible(
        &self,
        base: &NativeSpecialization,
        parameters: &[ParameterAddress],
        after: Option<&[u64]>,
        limit: usize,
    ) -> Result<Vec<Vec<u64>>, PlanError> {
        let recording = self.recording;
        let coordinates = parameters.iter().map(|address| recording.coordinate(address)).collect::<Vec<_>>();
        let mut base_assignment = Vec::new();
        recording.assign(base, &mut base_assignment);
        let mut admitted = Vec::new();
        if limit > 0 {
            recording.changes(&base_assignment, &coordinates, after, |assignment| {
                admitted.push(coordinates.iter().map(|&coordinate| assignment[coordinate]).collect());
                admitted.len() < limit
            });
        }
        Ok(admitted)
    }

    /// The definition of [`plan::TuningPartition::nearest`] over the
    /// record's domain: of the configurations with `parameters` at `values`,
    /// the one that changes the fewest other parameters from `base`, then
    /// the fewest steps, then the earliest. Only the parameters of the
    /// points that hold no row of `base` so changed can make it held, so
    /// only those are tried, one changed, then two, to the first number
    /// that holds any.
    fn nearest(
        &self,
        base: &NativeSpecialization,
        parameters: &[ParameterAddress],
        values: &[u64],
    ) -> Result<Option<NativeSpecialization>, PlanError> {
        let recording = self.recording;
        let mut fixed = Vec::new();
        recording.assign(base, &mut fixed);
        let held = parameters.iter().map(|address| recording.coordinate(address)).collect::<Vec<_>>();
        for (&coordinate, value) in held.iter().zip(values) {
            fixed[coordinate] = *value;
        }
        if recording.index.holds(&fixed) {
            return Ok(Some(recording.specialization(&fixed)));
        }
        let failing = (0..recording.labels.len())
            .filter(|&point| recording.index.work_at(point, &fixed).is_none())
            .collect::<Vec<_>>();
        // No configuration has `values` when one is not surveyed, or when a
        // point holds no row with them.
        let surveyed = held
            .iter()
            .zip(values)
            .all(|(&coordinate, value)| recording.record.parameters[coordinate].values.contains(value));
        let rowed = failing.iter().all(|&point| {
            let surveyed = &recording.record.points[point];
            surveyed.rows.iter().any(|row| {
                surveyed
                    .leading
                    .iter()
                    .zip(&row.values)
                    .all(|(coordinate, value)| !held.contains(coordinate) || fixed[*coordinate] == *value)
            })
        });
        if !surveyed || !rowed {
            return Ok(None);
        }
        let mut free = failing
            .iter()
            .flat_map(|&point| recording.record.points[point].leading.iter().copied())
            .filter(|coordinate| !held.contains(coordinate))
            .collect::<Vec<_>>();
        free.sort_unstable();
        free.dedup();
        let declared = |coordinate: usize| recording.record.parameters[coordinate].values.as_slice();
        let mut nearest = None::<((usize, usize), Vec<u64>)>;
        for moved in Subsets::nearest_first(free.len()) {
            if nearest.as_ref().is_some_and(|((changed, _), _)| *changed < moved.len()) {
                break;
            }
            let coordinates = moved.iter().map(|&position| free[position]).collect::<Vec<_>>();
            recording.changes(&fixed, &coordinates, None, |candidate| {
                let distance = plan::distance_of(
                    coordinates.iter().map(|&coordinate| (declared(coordinate), fixed[coordinate], candidate[coordinate])),
                );
                if nearest.as_ref().is_none_or(|(least, _)| distance < *least) {
                    nearest = Some((distance, candidate.to_vec()));
                }
                true
            });
        }
        Ok(nearest.map(|(_, nearest)| recording.specialization(&nearest)))
    }

    fn forecast(
        &self,
        point: usize,
        _baseline: &LaunchKey,
        specialization: &NativeSpecialization,
    ) -> LaunchKey {
        let mut assignment = Vec::new();
        self.recording.assign(specialization, &mut assignment);
        self.key(&assignment, point)
    }
}

/// How much one configuration for every point loses at one point.
#[derive(Clone, Debug)]
pub struct PointGap {
    pub label: String,
    pub weight: f64,
    /// True time of the overall best configuration here.
    pub shared_seconds: f64,
    /// True time of the configuration best here alone.
    pub local_seconds: f64,
    pub local: ParameterValues,
}

impl PointGap {
    pub fn gap(&self) -> f64 {
        self.shared_seconds / self.local_seconds - 1.0
    }
}

/// A deterministic generator for sample draws (xorshift64*).
struct Draws(u64);

impl Draws {
    fn index(&mut self, below: usize) -> usize {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 33) as usize % below
    }
}

/// Measures recorded configurations by drawing recorded samples.
struct Recorded<'a, 'r> {
    domain: Replayed<'a, 'r>,
    draws: Draws,
}

impl Recorded<'_, '_> {
    /// One round of the comparison of `challenger` with `served`: a draw
    /// of each at every point where they run different work, as the
    /// round's `d` ([`super::sampling`]).
    fn round(&mut self, challenger: &[u64], served: &[u64]) -> f64 {
        let recording = self.domain.recording;
        let works = |assignment: &[u64]| recording.served(assignment).expect("a measured configuration serves");
        let (challenging, serving) = (works(challenger), works(served));
        let mut difference = 0.;
        for point in 0..recording.labels.len() {
            if self.domain.key(challenger, point) != self.domain.key(served, point) {
                let mut draw = |work: &RecordedWork| work.samples[self.draws.index(work.samples.len())];
                difference += recording.weights[point] * (draw(challenging[point]) - draw(serving[point]));
            }
        }
        difference
    }

    /// The configuration of `assignment` measured at every point, one draw
    /// per new measurement, as a search measures a challenger in its first
    /// round; `shared` holds the measurements this pass may reuse
    /// ([`Objective::Keyed`]). None when it cannot serve.
    fn measure(
        &mut self,
        assignment: &[u64],
        shared: &mut HashMap<(usize, LaunchKey), f64>,
    ) -> Option<Vec<PointMeasurement>> {
        let recording = self.domain.recording;
        let works = recording.served(assignment)?;
        Some(
            works
                .iter()
                .enumerate()
                .map(|(point, work)| {
                    let key = self.domain.key(assignment, point);
                    let id = (point, key.clone());
                    let seconds = match shared.get(&id) {
                        Some(seconds) => *seconds,
                        None => {
                            let seconds = work.samples[self.draws.index(work.samples.len())];
                            if self.domain.objective == Objective::Keyed {
                                shared.insert(id, seconds);
                            }
                            seconds
                        }
                    };
                    PointMeasurement {
                        point: recording.labels[point].clone(),
                        key,
                        median_seconds: seconds,
                        deviation_seconds: 0.,
                        samples: Vec::new(),
                        repetitions: 1,
                        rotation_bytes: 0,
                    }
                })
                .collect(),
        )
    }

    /// The cost of `measured` relative to `reference`.
    fn cost(&self, measured: &[PointMeasurement], reference: &[PointMeasurement]) -> Cost {
        let medians = |points: &[PointMeasurement]| {
            points
                .iter()
                .map(|point| point.median_seconds)
                .collect::<Vec<_>>()
        };
        Cost::relative(
            measured.iter().map(|point| point.key.clone()).collect(),
            &self.domain.recording.weights,
            &medians(measured),
            &medians(reference),
        )
    }
}

/// The outcome of replaying one recording many times.
#[derive(Clone, Debug)]
pub struct Replay {
    /// Per run: the chosen configuration's true cost over the true best's,
    /// minus one.
    pub chosen: Vec<f64>,
    /// Per run: configurations the search evaluated.
    pub evaluated: Vec<usize>,
    /// Per run, after each evaluation: the excess of the configuration the
    /// search estimated cheapest so far.
    pub curves: Vec<Vec<f64>>,
}

impl Replay {
    /// The fraction of runs whose choice is within `excess` of the best.
    pub fn within(&self, excess: f64) -> f64 {
        self.chosen
            .iter()
            .filter(|chosen| **chosen <= excess)
            .count() as f64
            / self.chosen.len() as f64
    }

    /// The smallest evaluation count after which at least `fraction` of the
    /// runs had reached a configuration within `excess` of the best; `None`
    /// when the runs never get there.
    pub fn needed(&self, excess: f64, fraction: f64) -> Option<usize> {
        let longest = self.curves.iter().map(Vec::len).max().unwrap_or(0);
        (1..=longest).find(|&count| {
            let reached = self
                .curves
                .iter()
                .filter(|curve| {
                    curve
                        .get(count - 1)
                        .or(curve.last())
                        .is_some_and(|excess_at| *excess_at <= excess)
                })
                .count();
            reached as f64 >= fraction * self.curves.len() as f64
        })
    }

    /// The `quantile` of the chosen excess over the runs.
    pub fn quantile(&self, quantile: f64) -> f64 {
        let mut chosen = self.chosen.clone();
        chosen.sort_by(f64::total_cmp);
        chosen[((chosen.len() - 1) as f64 * quantile).round() as usize]
    }
}

/// Replay the search `runs` times over `recording`, each run's time
/// covering `evaluations` configurations (the defaults among them) and
/// drawing from its own seed. Validation always passes on a survey record,
/// which holds only validated configurations.
pub fn replay(
    recording: &Recording,
    evaluations: usize,
    settings: &SearchSettings,
    objective: Objective,
    runs: usize,
) -> Replay {
    let mut replay = Replay {
        chosen: Vec::with_capacity(runs),
        evaluated: Vec::with_capacity(runs),
        curves: Vec::with_capacity(runs),
    };
    let defaults = recording.index.defaults();
    for run in 0..runs {
        let mut evaluator = Recorded {
            domain: Replayed::new(recording, objective),
            draws: Draws((run as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1),
        };
        // Search measurements by point and key, shared as a unit's are.
        let mut shared = HashMap::new();
        let reference = evaluator
            .measure(defaults, &mut shared)
            .expect("a recording's defaults were measured");
        let mut search = GroupSearch::new(
            recording.record.plan.clone(),
            recording.specialization(defaults),
            reference.clone(),
            recording.weights.clone(),
            Vec::new(),
        );
        // Every configuration measured, with its measurement and its cost
        // as the search estimates it.
        let mut measured = vec![(
            defaults.to_vec(),
            reference.clone(),
            evaluator.cost(&reference, &reference),
        )];
        let mut cheapest = 0;
        let mut curve = vec![recording.excess(defaults)];
        let mut evaluated = 1;
        'search: while let Some(piece) = search
            .plan(&evaluator.domain)
            .expect("a record admits or refuses a configuration, and never fails")
        {
            for challenger in piece.challengers() {
                if evaluated >= evaluations {
                    break 'search;
                }
                evaluated += 1;
                let mut assignment = Vec::new();
                recording.assign(challenger, &mut assignment);
                assert!(
                    recording.index.holds(&assignment),
                    "the search asks only for what the record admits"
                );
                let points = evaluator.measure(&assignment, &mut shared);
                search.record(challenger, points.clone());
                if let Some(points) = points {
                    let cost = evaluator.cost(&points, &reference);
                    if cost.total() < measured[cheapest].2.total() {
                        cheapest = measured.len();
                    }
                    measured.push((assignment, points, cost));
                }
                curve.push(recording.excess(&measured[cheapest].0));
            }
        }
        // Convert the leads as a unit does: the cheapest challenger whose
        // measured cost is below the served choice's is compared
        // with it in matched rounds of draws, the fewest in which a lead
        // can resolve, and serves when they resolve one. A unit gives a
        // lead its rounds left unresolved more of them while they are worth
        // their time; a replay, which has no time, drops it.
        let mut served = measured.remove(0);
        loop {
            let leader = measured
                .iter()
                .enumerate()
                .filter(|(_, (_, _, cost))| cost.improves_on(&served.2))
                .min_by(|left, right| left.1.2.total().total_cmp(&right.1.2.total()))
                .map(|(position, _)| position);
            let Some(leader) = leader else {
                break;
            };
            let leader = measured.remove(leader);
            let mut tally = Tally::default();
            for _ in 0..tally.fewest(settings.error_rate) {
                let difference = evaluator.round(&leader.0, &served.0);
                tally.observe(difference);
                if difference >= 0. {
                    break;
                }
            }
            if tally.verdict(settings.error_rate) == Some(Verdict::Leads) {
                served = leader;
            }
        }
        replay.chosen.push(recording.excess(&served.0));
        replay.evaluated.push(evaluated);
        replay.curves.push(curve);
    }
    replay
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::test_support::{declared, CountedDomain, V41_ATTENTION_DECODE};
    use crate::native::survey::{surveyed_parameters, SurveyRecord, SurveyedPoint, SurveyedRow, SurveyedWork, WorkOutcome};
    use crate::native::tune::{Configuration, DeclaredParameter, TuningTime};
    use std::collections::BTreeSet;

    /// The plan of a program formed whole over the entry parameters
    /// `parameters`, launch 0 working at each of `labels`.
    fn whole(parameters: &[DeclaredParameter], labels: &[&str]) -> plan::TuningPartition {
        plan::whole(
            &parameters
                .iter()
                .map(|parameter| (parameter.name.clone(), parameter.values.clone(), parameter.form))
                .collect::<Vec<_>>(),
            1,
            labels
                .iter()
                .map(|label| plan::PlannedPoint { label: (*label).into(), active: vec![0] })
                .collect(),
        )
    }

    /// A configuration of entry parameters `values`.
    fn configuration(values: &ParameterValues) -> Configuration {
        Configuration {
            statics: Default::default(),
            params: values.clone(),
            launches: Vec::new(),
        }
    }

    /// The result of a survey at `points` (label, weight) recording
    /// `record`.
    fn surveyed(points: &[(&str, f64)], parameters: Vec<DeclaredParameter>, record: SurveyRecord) -> TuningResult {
        TuningResult {
            disposition: crate::native::tune::TuningDisposition::QualifiedCandidate,
            tuning_identity: String::new(),
            entry: "test".into(),
            backend: "cpu".into(),
            points: points
                .iter()
                .map(|(label, weight)| crate::native::tune::PointRecord {
                    label: (*label).into(),
                    weight: *weight,
                    class: None,
                })
                .collect(),
            requirement: crate::native::tune::NumericalRequirement::BitExact,
            observations: Vec::new(),
            served: Vec::new(),
            reselection: None,
            implementation_identity: String::new(),
            parameters,
            configurations: Vec::new(),
            overall: record.defaults.clone(),
            method: TuningMethod::Survey { samples: 15, record },
            time: TuningTime::default(),
            coverage: None,
            default_launches: Vec::new(),
        }
    }

    /// One point's record: each of `rows` (its configuration's values of
    /// the entry parameters `parameters`, every one leading) runs the work
    /// `work` gives it, one work per distinct key.
    fn point(
        parameters: &[DeclaredParameter],
        rows: &[ParameterValues],
        work: impl Fn(&ParameterValues) -> (LaunchKey, f64, Vec<f64>),
        label: &str,
    ) -> SurveyedPoint {
        let mut point = SurveyedPoint {
            leading: (0..parameters.len()).collect(),
            rows: Vec::new(),
            work: Vec::new(),
        };
        let mut found = HashMap::new();
        for values in rows {
            let (key, seconds, samples) = work(values);
            let index = *found.entry(key.clone()).or_insert_with(|| {
                point.work.push(SurveyedWork {
                    configuration: configuration(values),
                    artifact: Some(String::new()),
                    observation: None,
                    outcome: WorkOutcome::Measured(PointMeasurement {
                        point: label.into(),
                        key,
                        median_seconds: seconds,
                        deviation_seconds: 0.0,
                        samples,
                        repetitions: 1,
                        rotation_bytes: 0,
                    }),
                });
                point.work.len() - 1
            });
            point.rows.push(SurveyedRow {
                values: parameters.iter().map(|parameter| values[&parameter.name]).collect(),
                work: index,
            });
        }
        point
    }

    /// A survey of `A` (read by the first point) and `B` (read by the
    /// second, which runs a hundred times longer and is noisy), its second
    /// point's samples spread by `spread` of its time a step.
    fn survey(spread: f64) -> TuningResult {
        let rows = [1u64, 2, 3, 4]
            .into_iter()
            .flat_map(|a| [1u64, 2].map(|b| [("A".to_string(), a), ("B".to_string(), b)].into_iter().collect()))
            .collect::<Vec<ParameterValues>>();
        let read = |name: &str, values: &ParameterValues, base: f64, spread: f64| {
            (
                LaunchKey {
                    launches: vec![0],
                    values: [(name.to_string(), values[name])].into_iter().collect(),
                },
                base,
                (0..15).map(|sample| base * (1.0 + spread * ((sample % 5) as f64 - 2.0))).collect(),
            )
        };
        let parameters = ["A", "B"]
            .into_iter()
            .zip([vec![1, 2, 3, 4], vec![1, 2]])
            .map(|(name, values)| DeclaredParameter {
                name: name.into(),
                launch: None,
                arithmetic: false,
                form: false,
                values,
            })
            .collect::<Vec<_>>();
        let m1 = |values: &ParameterValues| {
            read("A", values, 1e-4 * (1.0 - 0.1 * (values["A"] - 1) as f64), 0.001)
        };
        let m256 = |values: &ParameterValues| {
            read("B", values, 1e-2 * if values["B"] == 1 { 1.0 } else { 1.001 }, spread)
        };
        let record = SurveyRecord {
            parameters: parameters.clone(),
            defaults: configuration(&rows[0]),
            formed: rows.len(),
            points: vec![point(&parameters, &rows, m1, "m1"), point(&parameters, &rows, m256, "m256")],
            plan: whole(&parameters, &["m1", "m256"]),
        };
        surveyed(&[("m1", 0.4), ("m256", 0.6)], parameters, record)
    }

    fn settings() -> SearchSettings {
        SearchSettings { error_rate: 0.05 }
    }

    #[test]
    fn global_service_objective_and_physical_sample_reuse_are_independent() {
        let survey = survey(0.02);
        let recording = Recording::new(&survey).unwrap();
        let evaluator = |objective| Recorded {
            domain: Replayed::new(&recording, objective),
            draws: Draws(0x9e3779b97f4a7c15),
        };
        let (defaults, best) = (recording.index.defaults(), recording.best.0.clone());
        assert_eq!(recording.best()["A"], 4);
        // The noisy region a hundred times longer dominates actual
        // service: the whole gain of the best configuration is under 1%.
        let gain = 1.0 - recording.truth(&best);
        let baseline = 0.4 * 1e-4 + 0.6 * 1e-2;
        assert!((gain - (0.4 * 3e-5 / baseline)).abs() < 1e-12);
        assert!(gain < 0.01);
        let mut keyed = evaluator(Objective::Keyed);
        let mut shared = HashMap::new();
        let first = keyed.measure(defaults, &mut shared).unwrap();
        let rival = keyed.measure(&best, &mut shared).unwrap();
        assert_eq!(
            first[1].median_seconds, rival[1].median_seconds,
            "unchanged physical B work has one observation"
        );
        assert_eq!(shared.len(), 3, "only the changed A point adds evidence");
        // The shared point cancels: the cost tells them apart by A alone.
        assert!(keyed.cost(&rival, &first).improves_on(&keyed.cost(&first, &first)));
        let mut separate = evaluator(Objective::Separate);
        let mut independent = HashMap::new();
        let first = separate.measure(defaults, &mut independent).unwrap();
        let rival = separate.measure(&best, &mut independent).unwrap();
        assert!(
            independent.is_empty(),
            "independent sampling never reuses a physical observation"
        );
        assert_ne!(first[1].key, rival[1].key);
    }

    /// The replay runs the production order over the record as its domain:
    /// with evaluations for the whole record it reaches the true best, the
    /// defaults alone are the choice of a run that measures nothing else,
    /// and what it asks for is the record's configurations, each at most
    /// once.
    #[test]
    fn a_replay_runs_the_production_order_over_the_record() {
        // Both points read within a twentieth of a percent: the choice is
        // the order's, not the noise's.
        let survey = survey(0.0001);
        let recording = Recording::new(&survey).unwrap();
        let defaults = recording.index.defaults();
        let whole = replay(&recording, recording.len() as usize, &settings(), Objective::Keyed, 50);
        // The defaults, every other value of each parameter from them, the
        // leader's one unmeasured neighbour, and then the rest of the
        // record: a search ends when its domain is exhausted.
        assert!(whole.evaluated.iter().all(|evaluated| *evaluated == 8), "{:?}", whole.evaluated);
        assert_eq!(whole.within(0.0), 1.0, "{:?}", whole.chosen);
        let none = replay(&recording, 1, &settings(), Objective::Keyed, 5);
        assert!(none.evaluated.iter().all(|evaluated| *evaluated == 1));
        assert!(none.chosen.iter().all(|chosen| *chosen == recording.excess(defaults)));
        // The record is the domain: the nearest configuration with B at 2
        // from the defaults is the one that moves nothing else.
        let domain = Replayed::new(&recording, Objective::Keyed);
        let moved = plan::Domain::nearest(
            &domain,
            &recording.specialization(defaults),
            &[ParameterAddress::Entry("B".into())],
            &[2],
        )
        .unwrap()
        .unwrap();
        assert_eq!((moved.param("A"), moved.param("B")), (Some(1), Some(2)));
        assert_eq!(
            plan::Domain::place(
                &domain,
                &recording.specialization(defaults),
                &[ParameterAddress::Entry("A".into())],
                &[9],
            ),
            Ok(None)
        );
    }

    /// A survey of an entry parameter `E` and two launches' `X` and `Y`,
    /// formed separately: the first point runs launch 0 (it reads `E` and
    /// `X`), the second launch 1 (`E` and `Y`), the third neither (`E`
    /// alone); `E` 1 with `Y` 3 is not recorded, and `E` 0 with `X` 3 is
    /// excluded. Every sample of a work is its time, distinct per work.
    fn launches() -> TuningResult {
        let parameter = |name: &str, launch, values: Vec<u64>| DeclaredParameter {
            name: name.into(),
            launch,
            arithmetic: false,
            form: false,
            values,
        };
        let parameters = vec![
            parameter("E", None, vec![0, 1]),
            parameter("X", Some(0), vec![1, 2, 3]),
            parameter("Y", Some(1), vec![2, 3, 1]),
        ];
        let leading = [vec![0, 1], vec![0, 2], vec![0]];
        let labels = ["m1", "m8", "m64"];
        let points = leading
            .iter()
            .enumerate()
            .map(|(point, leading)| {
                let mut surveyed = SurveyedPoint { leading: leading.clone(), rows: Vec::new(), work: Vec::new() };
                let mut rows = vec![Vec::new()];
                for &position in leading {
                    rows = rows
                        .into_iter()
                        .flat_map(|prefix: Vec<u64>| {
                            parameters[position].values.iter().map(move |value| {
                                let mut row = prefix.clone();
                                row.push(*value);
                                row
                            })
                        })
                        .collect();
                }
                for values in rows.into_iter().filter(|values| !(point == 1 && values == &[1, 3])) {
                    let mut configuration = Configuration {
                        statics: Default::default(),
                        params: [("E".to_string(), 0)].into_iter().collect(),
                        launches: vec![[("X".to_string(), 1)].into_iter().collect(), [("Y".to_string(), 2)].into_iter().collect()],
                    };
                    for (&position, value) in leading.iter().zip(&values) {
                        let parameter = &parameters[position];
                        match parameter.launch {
                            None => configuration.params.insert(parameter.name.clone(), *value),
                            Some(launch) => configuration.launches[launch].insert(parameter.name.clone(), *value),
                        };
                    }
                    let seconds = 1e-4 * (1.0 + point as f64) * (1.0 + 0.37 * values[0] as f64 - 0.11 * values.iter().skip(1).map(|value| (*value as f64 - 2.2).powi(2)).sum::<f64>());
                    let key = LaunchKey {
                        launches: (point < 2).then_some(point).into_iter().collect(),
                        values: leading.iter().zip(&values).map(|(position, value)| (format!("{position}"), *value)).collect(),
                    };
                    surveyed.work.push(SurveyedWork {
                        configuration,
                        artifact: None,
                        observation: None,
                        outcome: if point == 0 && values == [0, 3] {
                            WorkOutcome::Excluded(crate::native::tune::Exclusion::Formation("refused".into()))
                        } else {
                            WorkOutcome::Measured(PointMeasurement {
                                point: labels[point].into(),
                                key,
                                median_seconds: seconds,
                                deviation_seconds: 0.,
                                samples: vec![seconds; 5],
                                repetitions: 1,
                                rotation_bytes: 0,
                            })
                        },
                    });
                    surveyed.rows.push(SurveyedRow { values, work: surveyed.work.len() - 1 });
                }
                surveyed
            })
            .collect();
        let address = |position: usize| match parameters[position].launch {
            None => ParameterAddress::Entry(parameters[position].name.clone()),
            Some(ordinal) => ParameterAddress::Launch { ordinal, name: parameters[position].name.clone() },
        };
        // The search's plan: the launches share `E`, one group.
        let plan = plan::TuningPartition {
            groups: vec![plan::Group {
                launches: vec![0, 1],
                parameters: (0..3).map(address).collect(),
                values: parameters.iter().map(|parameter| parameter.values.clone()).collect(),
                forms: vec![false; 3],
            }],
            boundary: Vec::new(),
            boundary_values: Vec::new(),
            boundary_forms: Vec::new(),
            points: labels
                .iter()
                .enumerate()
                .map(|(point, label)| plan::PlannedPoint { label: (*label).into(), active: (point < 2).then_some(point).into_iter().collect() })
                .collect(),
        };
        let record = SurveyRecord {
            parameters: parameters.clone(),
            defaults: Configuration {
                statics: Default::default(),
                params: [("E".to_string(), 0)].into_iter().collect(),
                launches: vec![[("X".to_string(), 1)].into_iter().collect(), [("Y".to_string(), 2)].into_iter().collect()],
            },
            formed: 0,
            points,
            plan,
        };
        surveyed(&[("m1", 0.5), ("m8", 0.3), ("m64", 0.2)], parameters, record)
    }

    /// The assignments of `coordinates` (positions of `E`, `X`, `Y`) that
    /// change each from `base` to another value, in walk order: `values`
    /// gives each position's values in declaration order.
    fn brute_changes(values: &[Vec<u64>], base: &[u64], coordinates: &[usize]) -> Vec<Vec<u64>> {
        let mut every = vec![base.to_vec()];
        for &coordinate in coordinates {
            every = every
                .into_iter()
                .flat_map(|prefix| {
                    values[coordinate].iter().filter(|value| **value != base[coordinate]).map(move |value| {
                        let mut assignment = prefix.clone();
                        assignment[coordinate] = *value;
                        assignment
                    })
                })
                .collect();
        }
        every
    }

    /// Over a record whose launch parameters vary, the replay's domain is
    /// every configuration whose work at each point the survey recorded,
    /// costed from that work: its count, its true best and what it admits
    /// are those of brute force over the product of the values, and the
    /// true best is the survey's own choice ([`SurveyIndex::best`]). A
    /// replay with evaluations for the whole domain reaches it every run.
    #[test]
    fn a_replay_over_varying_launch_parameters_costs_each_configuration_from_its_work() {
        let survey = launches();
        let recording = Recording::new(&survey).unwrap();
        let values = recording.record.parameters.iter().map(|parameter| parameter.values.clone()).collect::<Vec<_>>();
        let every = brute_changes(&values, &[9, 9, 9], &[0, 1, 2]);
        let held = every.iter().filter(|assignment| recording.index.holds(assignment)).cloned().collect::<Vec<_>>();
        assert_eq!(held.len(), 2 * 3 * 3 - 3, "E 1 with Y 3 is outside the domain");
        assert_eq!(recording.len(), held.len() as u128);
        let serving = held.iter().filter(|assignment| recording.served(assignment).is_some()).collect::<Vec<_>>();
        assert_eq!(recording.measured(), serving.len() as u128);
        assert_eq!(serving.len(), held.len() - 3, "E 0 with X 3 is excluded");
        // The brute-force best: the first of the least true cost, each
        // configuration's cost composed from its work at each point.
        let truth = |assignment: &[u64]| {
            let works = (0..3)
                .map(|point| {
                    match &recording.record.points[point].work[recording.index.work_at(point, assignment).unwrap()].outcome {
                        WorkOutcome::Measured(measurement) => Some(measurement.median_seconds),
                        WorkOutcome::Excluded(_) => None,
                    }
                })
                .collect::<Option<Vec<_>>>();
            works.map_or(f64::INFINITY, |times| times.iter().zip(&recording.weights).map(|(time, weight)| time * weight).sum())
        };
        let least = held.iter().map(|assignment| truth(assignment)).fold(f64::INFINITY, f64::min);
        let brute = held.iter().find(|assignment| truth(assignment) == least).unwrap();
        assert_eq!(&recording.best.0, brute);
        assert_ne!(&recording.best.0, recording.index.defaults(), "the best moves from the defaults");
        // The survey's choice by its own cost is the same configuration.
        let chosen = recording
            .index
            .best(|point, work| match &recording.record.points[point].work[work].outcome {
                WorkOutcome::Measured(measurement) => Some(recording.weights[point] * measurement.median_seconds),
                WorkOutcome::Excluded(_) => None,
            })
            .unwrap();
        assert_eq!(recording.index.assignment(&chosen).as_ref(), Some(brute));
        // What the domain admits changing each set of parameters, a batch
        // of one at a time, is brute force's in walk order.
        let domain = Replayed::new(&recording, Objective::Keyed);
        let addresses = recording.addresses.clone();
        for base in &held {
            let specialization = recording.specialization(base);
            for coordinates in [vec![0], vec![1], vec![2], vec![1, 2], vec![2, 0], vec![0, 1, 2]] {
                let parameters = coordinates.iter().map(|&coordinate| addresses[coordinate].clone()).collect::<Vec<_>>();
                let expected = brute_changes(&values, base, &coordinates)
                    .into_iter()
                    .filter(|assignment| recording.index.holds(assignment))
                    .map(|assignment| coordinates.iter().map(|&coordinate| assignment[coordinate]).collect::<Vec<_>>())
                    .collect::<Vec<_>>();
                let mut paged = Vec::new();
                while let [next] = plan::Domain::admissible(&domain, &specialization, &parameters, paged.last().map(Vec::as_slice), 1)
                    .unwrap()
                    .as_slice()
                {
                    paged.push(next.clone());
                }
                assert_eq!(paged, expected, "{base:?} changing {coordinates:?}");
                assert_eq!(plan::Domain::admissible(&domain, &specialization, &parameters, None, usize::MAX).unwrap(), expected);
            }
            // The nearest with `Y` at each value: brute force's least
            // distance; of those as near, the one changing the parameters
            // declared earliest, at their earliest values.
            for y in [2, 3, 1] {
                let nearest = plan::Domain::nearest(&domain, &specialization, &[addresses[2].clone()], &[y]).unwrap();
                let expected = held
                    .iter()
                    .filter(|candidate| candidate[2] == y)
                    .min_by_key(|candidate| {
                        let changed = (0..2).filter(|&coordinate| candidate[coordinate] != base[coordinate]).collect::<Vec<_>>();
                        let ordinals = changed
                            .iter()
                            .map(|&coordinate| ordinal(&values[coordinate], candidate[coordinate]))
                            .collect::<Vec<_>>();
                        (
                            plan::distance_of((0..2).map(|coordinate| (values[coordinate].as_slice(), base[coordinate], candidate[coordinate]))),
                            changed,
                            ordinals,
                        )
                    });
                let nearest = nearest.map(|nearest| {
                    let mut assignment = Vec::new();
                    recording.assign(&nearest, &mut assignment);
                    assignment
                });
                assert_eq!(nearest.as_ref(), expected, "{base:?} with Y {y}");
            }
        }
        let whole = replay(&recording, recording.len() as usize, &settings(), Objective::Keyed, 20);
        assert_eq!(whole.within(0.0), 1.0, "{:?}", whole.chosen);
    }

    /// V4.1's Metal attention_decode as a survey recorded it
    /// ([`V41_ATTENTION_DECODE`]): eight parameters whose declared lists
    /// name 129,024 configurations, every one the declaration admits
    /// recorded at one point.
    fn attention_decode() -> TuningResult {
        let implementation = declared(V41_ATTENTION_DECODE);
        let defaults = implementation
            .default_specialization_without_device(&NativeSpecialization::new()).unwrap()
            .unwrap();
        let mut admitted = Vec::new();
        implementation
            .walk_admissible(&NativeSpecialization::new(), |configuration| {
                admitted.push(configuration.params().clone());
                true
            })
            .unwrap();
        let work = |values: &ParameterValues| {
            let seconds = 5e-5
                * (1.0 + (values["SPAN"] as f64).log2() / 16.0 + (values["PARTS"] as f64 / 64.0 - 1.0).abs() / 8.0);
            (LaunchKey { launches: vec![0], values: values.clone() }, seconds, vec![seconds; 15])
        };
        let parameters = surveyed_parameters(&implementation);
        let record = SurveyRecord {
            parameters: parameters.clone(),
            defaults: Configuration::of(&defaults),
            formed: admitted.len(),
            points: vec![point(&parameters, &admitted, work, "m1")],
            plan: whole(&parameters, &["m1"]),
        };
        surveyed(&[("m1", 1.0)], parameters, record)
    }

    /// Planning a whole search of V4.1's Metal attention_decode over its
    /// survey places at most a few hundred configurations, not its
    /// declared product: the rest of a group is walked over the 28 the
    /// record admits. Placing all 129,024 again in each of a survey
    /// report's eight thousand runs spun its replay for over ten minutes.
    #[test]
    fn planning_a_survey_replay_places_what_the_record_admits_not_the_declared_product() {
        let survey = attention_decode();
        let recording = Recording::new(&survey).unwrap();
        assert_eq!(recording.len(), 28);
        // The replay as a survey report runs it, its domain counted.
        let counted = CountedDomain::new(Replayed::new(&recording, Objective::Keyed));
        let mut evaluator = Recorded {
            domain: Replayed::new(&recording, Objective::Keyed),
            draws: Draws(1),
        };
        let mut shared = HashMap::new();
        let defaults = recording.index.defaults();
        let reference = evaluator.measure(defaults, &mut shared).unwrap();
        let mut search = GroupSearch::new(
            recording.record.plan.clone(),
            recording.specialization(defaults),
            reference,
            recording.weights.clone(),
            Vec::new(),
        );
        let mut asked = BTreeSet::new();
        while let Some(piece) = search.plan(&counted).unwrap() {
            for challenger in piece.challengers() {
                let mut assignment = Vec::new();
                recording.assign(challenger, &mut assignment);
                assert!(recording.index.holds(&assignment), "the search asks only for what the record admits");
                let points = evaluator.measure(&assignment, &mut shared);
                asked.insert(assignment);
                search.record(challenger, points);
            }
        }
        assert_eq!(asked.len(), 27, "every other recorded configuration is asked for");
        assert!(counted.placed.get() <= 500, "{} placed", counted.placed.get());
    }

    /// Allocations made by the current thread, counted by this test
    /// binary's allocator.
    mod counted {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;

        thread_local! {
            static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
        }

        struct Counting;

        unsafe impl GlobalAlloc for Counting {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
                unsafe { System.alloc(layout) }
            }

            unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
                unsafe { System.dealloc(pointer, layout) }
            }
        }

        #[global_allocator]
        static ALLOCATOR: Counting = Counting;

        pub fn allocations() -> usize {
            ALLOCATIONS.with(Cell::get)
        }
    }

    /// A search asks the record for configurations a step from what it
    /// holds, most of which a survey did not record: asking for one the
    /// record does not hold allocates nothing, and one it holds costs only
    /// the configuration returned.
    #[test]
    fn asking_the_record_allocates_only_the_configuration_it_answers() {
        let survey = survey(0.02);
        let recording = Recording::new(&survey).unwrap();
        let domain = Replayed::new(&recording, Objective::Keyed);
        let defaults = recording.specialization(recording.index.defaults());
        let parameters = [
            ParameterAddress::Entry("A".into()),
            ParameterAddress::Entry("B".into()),
        ];
        let ask = |values: &[u64]| {
            plan::Domain::place(&domain, &defaults, &parameters, values)
                .expect("a record admits or refuses")
        };
        let before = counted::allocations();
        for a in 5..1005 {
            assert!(ask(&[a, 2]).is_none());
        }
        assert_eq!(counted::allocations(), before, "a refused configuration allocates");
        let held = ask(&[4, 2]).expect("the survey recorded A=4, B=2");
        assert_eq!((held.param("A"), held.param("B")), (Some(4), Some(2)));
    }

    #[test]
    fn gaps_compare_the_shared_best_with_each_point_alone() {
        let survey = survey(0.02);
        let gaps = Recording::new(&survey).unwrap().gaps();
        assert_eq!(gaps.len(), 2);
        assert!(gaps.iter().all(|gap| gap.gap().abs() < 1e-12), "{gaps:?}");
    }
}
