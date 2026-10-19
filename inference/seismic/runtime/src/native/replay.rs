//! Replay of the search against a recorded survey (tuning spec §E2).
//!
//! A survey ([`super::tune::Strategy::Survey`]) records every admissible
//! configuration of one entry instance with every sample of every point. A
//! replay runs the production search order ([`GroupSearch`]) over that
//! record many times: the record is the domain the order asks
//! ([`plan::Domain`]: what it admits is what the survey recorded), and each
//! simulated measurement draws samples with replacement from the recorded
//! samples of the configuration's point, as many as the search would take,
//! so the recorded noise is reproduced without modeling it.
//!
//! What the order asked for is then converted as a unit converts its leads:
//! the cheapest measured challenger whose cost is below the served choice's
//! is re-measured with it under matched draws and replaces it when
//! its rounds resolve a lead; one that does not gives way to the next. A unit also
//! holds a lead back until it exceeds the error its conversions have
//! observed; a replay converts every nominal lead, which is what a unit
//! does before it has observed any.
//!
//! The *true* time of a configuration at a point is the median of all its
//! recorded samples there, and its true cost is the production objective
//! over true times ([`Cost::relative`]). Each replay reports
//! the chosen configuration's true cost relative to the true best, how fast
//! the search converges, and, per point, how much the configuration that is
//! best overall loses to the one best at that point alone (the gap a
//! per-size launch could recover).
//!
//! Both modes minimize the same invocation-demand elapsed-service objective.
//! [`Objective::Keyed`] shares identical physical point evidence; [`Objective::Separate`]
//! samples each configuration independently.

use super::plan::{self, ParameterAddress, PlanError, PlannedPoint};
use super::search::{Cost, LaunchKey, ParameterValues, SearchSettings};
use super::tune::{Configuration, GroupSearch, Outcome, PointMeasurement, TuningResult};
use seismic_lang::checked::NativeSpecialization;
use seismic_measure::{median, Tally, Verdict};
use std::collections::{BTreeSet, HashMap};

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

/// One point of a recorded configuration.
#[derive(Clone, Debug)]
struct RecordedPoint {
    key: LaunchKey,
    samples: Vec<f64>,
    /// The median of every recorded sample.
    time: f64,
}

/// A survey as the replay reads it.
#[derive(Clone, Debug)]
pub struct Recording {
    /// The surveyed parameters: name, declared values, whether they select
    /// a form.
    parameters: Vec<(String, Vec<u64>, bool)>,
    /// Every recorded configuration, in the survey's order: the defaults
    /// first.
    configurations: Vec<NativeSpecialization>,
    positions: HashMap<Configuration, usize>,
    weights: Vec<f64>,
    labels: Vec<String>,
    /// Per configuration; `None` when the survey excluded it.
    points: Vec<Option<Vec<RecordedPoint>>>,
    /// True cost per configuration (excluded: infinite).
    truth: Vec<f64>,
}

#[derive(Debug)]
pub enum RecordingError {
    /// A measured configuration lacks samples at a point.
    Empty { point: String },
    /// The survey could not measure the defaults.
    NoMeasuredDefaults,
}

impl std::fmt::Display for RecordingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty { point } => {
                write!(f, "a surveyed configuration has no samples at `{point}`")
            }
            Self::NoMeasuredDefaults => f.write_str("the survey did not measure the defaults"),
        }
    }
}

impl std::error::Error for RecordingError {}

impl Recording {
    pub fn new(result: &TuningResult) -> Result<Self, RecordingError> {
        let parameters = result
            .parameters
            .iter()
            .map(|parameter| {
                (
                    parameter.name.clone(),
                    parameter.values.clone(),
                    parameter.form,
                )
            })
            .collect::<Vec<_>>();
        let mut points = Vec::with_capacity(result.configurations.len());
        for record in &result.configurations {
            points.push(match &record.outcome {
                Outcome::Excluded(_) => None,
                // A survey times a configuration whatever its verdict; one
                // the record's requirement does not admit cannot be chosen.
                Outcome::Measured { .. } if result.admitted(record).is_err() => None,
                Outcome::Measured { points, .. } => Some(
                    points
                        .iter()
                        .map(|point| {
                            if point.samples.is_empty() {
                                return Err(RecordingError::Empty {
                                    point: point.point.clone(),
                                });
                            }
                            Ok(RecordedPoint {
                                key: point.key.clone(),
                                samples: point.samples.clone(),
                                time: median(&point.samples),
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                ),
            });
        }
        let weights = result
            .points
            .iter()
            .map(|point| point.weight)
            .collect::<Vec<_>>();
        let labels = result
            .points
            .iter()
            .map(|point| point.label.clone())
            .collect();
        // The defaults are the reference of every cost.
        let reference = points
            .first()
            .and_then(Option::as_ref)
            .ok_or(RecordingError::NoMeasuredDefaults)?
            .iter()
            .map(|point| point.time)
            .collect::<Vec<_>>();
        let truth = points
            .iter()
            .map(|recorded| match recorded {
                None => f64::INFINITY,
                Some(recorded) => Cost::relative(
                    recorded.iter().map(|point| point.key.clone()).collect(),
                    &weights,
                    &recorded.iter().map(|point| point.time).collect::<Vec<_>>(),
                    &reference,
                )
                .total(),
            })
            .collect();
        Ok(Self {
            parameters,
            configurations: result
                .configurations
                .iter()
                .map(|record| record.configuration.specialization())
                .collect(),
            positions: result
                .configurations
                .iter()
                .enumerate()
                .map(|(index, record)| (record.configuration.clone(), index))
                .collect(),
            weights,
            labels,
            points,
            truth,
        })
    }

    /// Configurations the survey recorded: every admissible one of the
    /// domain it was asked for.
    pub fn len(&self) -> usize {
        self.configurations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.configurations.is_empty()
    }

    /// The parameter values of configuration `index`.
    pub fn values(&self, index: usize) -> ParameterValues {
        self.configurations[index].params().clone()
    }

    /// Admissible configurations the survey measured.
    pub fn measured(&self) -> usize {
        self.points.iter().filter(|points| points.is_some()).count()
    }

    /// The configuration with the least true cost.
    pub fn best(&self) -> usize {
        (0..self.truth.len())
            .min_by(|left, right| self.truth[*left].total_cmp(&self.truth[*right]))
            .expect("the record holds the defaults")
    }

    /// True cost of `index` relative to the true best (0 = the best).
    pub fn excess(&self, index: usize) -> f64 {
        self.truth[index] / self.truth[self.best()] - 1.0
    }

    /// Per point: the configuration best there alone, and how much slower
    /// the overall best configuration is at that point.
    pub fn gaps(&self) -> Vec<PointGap> {
        let best = self.best();
        let overall = self.points[best]
            .as_ref()
            .expect("the best configuration was measured");
        self.labels
            .iter()
            .enumerate()
            .map(|(point, label)| {
                let (local, time) = self
                    .points
                    .iter()
                    .enumerate()
                    .filter_map(|(index, points)| {
                        points.as_ref().map(|points| (index, points[point].time))
                    })
                    .min_by(|left, right| left.1.total_cmp(&right.1))
                    .expect("the defaults were measured");
                PointGap {
                    label: label.clone(),
                    weight: self.weights[point],
                    shared_seconds: overall[point].time,
                    local_seconds: time,
                    local: self.values(local),
                }
            })
            .collect()
    }

    /// The surveyed domain as the search plans it: a survey measures
    /// programs formed whole, one group over all of it.
    fn partition(&self) -> plan::TuningPartition {
        let launches = |point: usize| {
            self.points
                .iter()
                .flatten()
                .flat_map(|points| points[point].key.launches.iter().copied())
                .collect::<BTreeSet<_>>()
        };
        let points = self
            .labels
            .iter()
            .enumerate()
            .map(|(point, label)| PlannedPoint {
                label: label.clone(),
                active: launches(point).into_iter().collect(),
            })
            .collect::<Vec<_>>();
        let working = points
            .iter()
            .flat_map(|point| point.active.iter().copied())
            .max()
            .map_or(0, |last| last + 1);
        plan::whole(&self.parameters, working, points)
    }

    /// `base` with `parameters` at `values`.
    fn moved(
        base: &NativeSpecialization,
        parameters: &[ParameterAddress],
        values: &[u64],
    ) -> NativeSpecialization {
        parameters
            .iter()
            .zip(values)
            .fold(base.clone(), |moved, (address, value)| address.set(moved, *value))
    }
}

/// A recording as the domain of a search: it admits what the survey
/// recorded, and a configuration is forecast to run what the survey
/// recorded it running.
struct Replayed<'r> {
    recording: &'r Recording,
    objective: Objective,
}

impl Replayed<'_> {
    /// The key configuration `index` is measured under at `point`: the
    /// recorded one, or under independent sampling one no other
    /// configuration has. An excluded configuration has none recorded: it
    /// is asked for, and found excluded.
    fn key(
        &self,
        index: Option<usize>,
        point: usize,
        specialization: &NativeSpecialization,
    ) -> LaunchKey {
        let recorded = index
            .and_then(|index| self.recording.points[index].as_ref())
            .filter(|_| self.objective == Objective::Keyed);
        match recorded {
            Some(points) => points[point].key.clone(),
            None => LaunchKey {
                launches: Vec::new(),
                values: specialization.params().clone(),
            },
        }
    }
}

impl plan::Domain for Replayed<'_> {
    fn place(
        &self,
        base: &NativeSpecialization,
        parameters: &[ParameterAddress],
        values: &[u64],
    ) -> Result<Option<NativeSpecialization>, PlanError> {
        let moved = Recording::moved(base, parameters, values);
        Ok(self
            .recording
            .positions
            .contains_key(&Configuration::of(&moved))
            .then_some(moved))
    }

    /// The definition of [`plan::TuningPartition::nearest`] over the
    /// recorded configurations, whose order is the declaration's.
    fn nearest(
        &self,
        base: &NativeSpecialization,
        parameters: &[ParameterAddress],
        values: &[u64],
    ) -> Result<Option<NativeSpecialization>, PlanError> {
        let moved = Recording::moved(base, parameters, values);
        let declared = self
            .recording
            .parameters
            .iter()
            .map(|(name, values, _)| (ParameterAddress::Entry(name.clone()), values.as_slice()))
            .collect::<Vec<_>>();
        Ok(self
            .recording
            .configurations
            .iter()
            .filter(|candidate| {
                parameters
                    .iter()
                    .zip(values)
                    .all(|(address, value)| address.value(candidate) == *value)
            })
            .min_by_key(|candidate| plan::distance(&declared, &moved, candidate))
            .cloned())
    }

    fn forecast(
        &self,
        point: usize,
        _baseline: &LaunchKey,
        specialization: &NativeSpecialization,
    ) -> LaunchKey {
        let index = self
            .recording
            .positions
            .get(&Configuration::of(specialization))
            .copied();
        self.key(index, point, specialization)
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
struct Recorded<'r> {
    domain: Replayed<'r>,
    draws: Draws,
}

impl Recorded<'_> {
    /// One recorded sample of configuration `index` at `point`.
    fn draw(&mut self, index: usize, point: usize) -> f64 {
        let samples = &self.domain.recording.points[index]
            .as_ref()
            .expect("a measured configuration is drawn from")[point]
            .samples;
        samples[self.draws.index(samples.len())]
    }

    /// One round of the comparison of `challenger` with `served`: a draw
    /// of each at every point where they run different work, as the
    /// round's `d` ([`super::sampling`]).
    fn round(&mut self, challenger: usize, served: usize) -> f64 {
        let recording = self.domain.recording;
        let mut difference = 0.;
        for point in 0..recording.labels.len() {
            let key = |index: usize| {
                self.domain
                    .key(Some(index), point, &recording.configurations[index])
            };
            if key(challenger) != key(served) {
                difference += recording.weights[point]
                    * (self.draw(challenger, point) - self.draw(served, point));
            }
        }
        difference
    }

    /// Configuration `index` measured at every point, one draw per new
    /// measurement, as a search measures a challenger in its first round;
    /// `shared` holds the measurements this pass may reuse
    /// ([`Objective::Keyed`]). None when the survey excluded it.
    fn measure(
        &mut self,
        index: usize,
        shared: &mut HashMap<(usize, LaunchKey), f64>,
    ) -> Option<Vec<PointMeasurement>> {
        let recording = self.domain.recording;
        let points = recording.points[index].as_ref()?;
        Some(
            points
                .iter()
                .enumerate()
                .map(|(point, recorded)| {
                    let key =
                        self.domain
                            .key(Some(index), point, &recording.configurations[index]);
                    let id = (point, key.clone());
                    let seconds = match shared.get(&id) {
                        Some(seconds) => *seconds,
                        None => {
                            let seconds =
                                recorded.samples[self.draws.index(recorded.samples.len())];
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
    for run in 0..runs {
        let mut evaluator = Recorded {
            domain: Replayed {
                recording,
                objective,
            },
            draws: Draws((run as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1),
        };
        // Search measurements by point and key, shared as a unit's are.
        let mut shared = HashMap::new();
        let reference = evaluator
            .measure(0, &mut shared)
            .expect("a recording's defaults were measured");
        let mut search = GroupSearch::new(
            recording.partition(),
            recording.configurations[0].clone(),
            reference.clone(),
            recording.weights.clone(),
            Vec::new(),
        );
        // Every configuration measured, with its measurement and its cost
        // as the search estimates it.
        let mut measured = vec![(0, reference.clone(), evaluator.cost(&reference, &reference))];
        let mut cheapest = 0;
        let mut curve = vec![recording.excess(0)];
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
                let index = recording.positions[&Configuration::of(challenger)];
                let points = evaluator.measure(index, &mut shared);
                search.record(challenger, points.clone());
                if let Some(points) = points {
                    let cost = evaluator.cost(&points, &reference);
                    if cost.total() < measured[cheapest].2.total() {
                        cheapest = measured.len();
                    }
                    measured.push((index, points, cost));
                }
                curve.push(recording.excess(measured[cheapest].0));
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
                let difference = evaluator.round(leader.0, served.0);
                tally.observe(difference);
                if difference >= 0. {
                    break;
                }
            }
            if tally.verdict(settings.error_rate) == Some(Verdict::Leads) {
                served = leader;
            }
        }
        replay.chosen.push(recording.excess(served.0));
        replay.evaluated.push(evaluated);
        replay.curves.push(curve);
    }
    replay
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::tune::{
        ConfigurationRecord, DeclaredParameter, TuningMethod, TuningTime,
    };

    /// A survey of `A` (read by the first point) and `B` (read by the
    /// second, which runs a hundred times longer and is noisy).
    fn recording() -> Recording {
        recording_spreading(0.02)
    }

    /// The same survey with the second point's samples spread by `spread`
    /// of its time a step.
    fn recording_spreading(spread: f64) -> Recording {
        let mut configurations = Vec::new();
        for a in [1u64, 2, 3, 4] {
            for b in [1u64, 2] {
                let values: ParameterValues = [("A".to_string(), a), ("B".to_string(), b)]
                    .into_iter()
                    .collect();
                let point = |label: &str, name: &str, base: f64, spread: f64| PointMeasurement {
                    point: label.into(),
                    key: LaunchKey {
                        launches: vec![0],
                        values: [(name.to_string(), values[name])].into_iter().collect(),
                    },
                    median_seconds: base,
                    deviation_seconds: 0.0,
                    samples: (0..15)
                        .map(|sample| base * (1.0 + spread * ((sample % 5) as f64 - 2.0)))
                        .collect(),
                    repetitions: 1,
                    rotation_bytes: 0,
                };
                configurations.push(ConfigurationRecord {
                    formation: None,
                    configuration: Configuration {
                        statics: Default::default(),
                        params: values.clone(),
                        launches: Vec::new(),
                    },
                    artifact: Some(String::new()),
                    observations: Vec::new(),
                    outcome: Outcome::Measured {
                        points: vec![
                            point("m1", "A", 1e-4 * (1.0 - 0.1 * (a - 1) as f64), 0.001),
                            point("m256", "B", 1e-2 * if b == 1 { 1.0 } else { 1.001 }, spread),
                        ],
                        confirmed: Vec::new(),
                    },
                });
            }
        }
        let result = TuningResult {
            disposition: crate::native::tune::TuningDisposition::QualifiedCandidate,
            tuning_identity: String::new(),
            entry: "test".into(),
            backend: "cpu".into(),
            points: [("m1", 0.4), ("m256", 0.6)]
                .into_iter()
                .map(|(label, weight)| crate::native::tune::PointRecord {
                    label: label.into(),
                    weight,
                    class: None,
                })
                .collect(),
            requirement: crate::native::tune::NumericalRequirement::BitExact,
            measurement_identity: String::new(),
            observations: Vec::new(),
            served: Vec::new(),
            reselection: None,
            implementation_identity: String::new(),
            parameters: vec![
                DeclaredParameter {
                    name: "A".into(),
                    launch: None,
                    arithmetic: false,
                    form: false,
                    values: vec![1, 2, 3, 4],
                },
                DeclaredParameter {
                    name: "B".into(),
                    launch: None,
                    arithmetic: false,
                    form: false,
                    values: vec![1, 2],
                },
            ],
            configurations,
            overall: Configuration {
                statics: Default::default(),
                params: Default::default(),
                launches: Vec::new(),
            },
            method: TuningMethod::Survey { samples: 15 },
            time: TuningTime::default(),
            coverage: None,
        };
        Recording::new(&result).unwrap()
    }

    fn settings() -> SearchSettings {
        SearchSettings { error_rate: 0.05 }
    }

    #[test]
    fn global_service_objective_and_physical_sample_reuse_are_independent() {
        let recording = recording();
        let evaluator = |objective| Recorded {
            domain: Replayed {
                recording: &recording,
                objective,
            },
            draws: Draws(0x9e3779b97f4a7c15),
        };
        let best = recording.best();
        assert_eq!(recording.values(best)["A"], 4);
        // The noisy region a hundred times longer dominates actual
        // service: the whole gain of the best configuration is under 1%.
        let gain = 1.0 - recording.truth[best];
        let baseline = 0.4 * 1e-4 + 0.6 * 1e-2;
        assert!((gain - (0.4 * 3e-5 / baseline)).abs() < 1e-12);
        assert!(gain < 0.01);
        let mut keyed = evaluator(Objective::Keyed);
        let mut shared = HashMap::new();
        let first = keyed.measure(0, &mut shared).unwrap();
        let rival = keyed.measure(best, &mut shared).unwrap();
        assert_eq!(
            first[1].median_seconds, rival[1].median_seconds,
            "unchanged physical B work has one observation"
        );
        assert_eq!(shared.len(), 3, "only the changed A point adds evidence");
        // The shared point cancels: the cost tells them apart by A alone.
        assert!(keyed.cost(&rival, &first).improves_on(&keyed.cost(&first, &first)));
        let mut separate = evaluator(Objective::Separate);
        let mut independent = HashMap::new();
        let first = separate.measure(0, &mut independent).unwrap();
        let rival = separate.measure(best, &mut independent).unwrap();
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
        let recording = recording_spreading(0.0001);
        let whole = replay(&recording, recording.len(), &settings(), Objective::Keyed, 50);
        // The defaults, every other value of each parameter from them, the
        // leader's one unmeasured neighbour, and then the rest of the
        // record: a search ends when its domain is exhausted.
        assert!(whole.evaluated.iter().all(|evaluated| *evaluated == 8), "{:?}", whole.evaluated);
        assert_eq!(whole.within(0.0), 1.0, "{:?}", whole.chosen);
        let none = replay(&recording, 1, &settings(), Objective::Keyed, 5);
        assert!(none.evaluated.iter().all(|evaluated| *evaluated == 1));
        assert!(none.chosen.iter().all(|chosen| *chosen == recording.excess(0)));
        // The record is the domain: the nearest configuration with B at 2
        // from the defaults is the one that moves nothing else.
        let domain = Replayed {
            recording: &recording,
            objective: Objective::Keyed,
        };
        let moved = plan::Domain::nearest(
            &domain,
            &recording.configurations[0],
            &[ParameterAddress::Entry("B".into())],
            &[2],
        )
        .unwrap()
        .unwrap();
        assert_eq!((moved.param("A"), moved.param("B")), (Some(1), Some(2)));
        assert_eq!(
            plan::Domain::place(
                &domain,
                &recording.configurations[0],
                &[ParameterAddress::Entry("A".into())],
                &[9],
            ),
            Ok(None)
        );
    }

    #[test]
    fn gaps_compare_the_shared_best_with_each_point_alone() {
        let gaps = recording().gaps();
        assert_eq!(gaps.len(), 2);
        assert!(gaps.iter().all(|gap| gap.gap().abs() < 1e-12), "{gaps:?}");
    }
}
