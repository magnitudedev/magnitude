//! Replay of the search against a recorded survey (tuning spec §E2).
//!
//! A survey ([`super::tune::Strategy::Survey`]) records every admissible
//! configuration of one entry instance with every sample of every point. A
//! replay runs the production [`search::search`] over that record many
//! times: each simulated measurement draws samples with replacement from the
//! recorded samples of the configuration's point, as many as the search
//! would take, so the recorded noise is reproduced without modeling it.
//!
//! The *true* time of a configuration at a point is the median of all its
//! recorded samples there, and its true cost is the production objective
//! over true times ([`Cost::relative`]). Each replay reports
//! the chosen configuration's true cost relative to the true best, how fast
//! the search converges, and, per point, how much the configuration that is
//! best overall loses to the one best at that point alone (the gap a
//! per-size launch could recover).
//!
//! [`Objective::Separate`] replays the previous objective for comparison:
//! absolute weighted medians, every point measured afresh for every
//! configuration, margins relative to the whole cost.

use super::search::{
    self, Cost, Evaluator, ParameterValues, PointKey, SearchSettings, SearchSpace,
};
use super::tune::{Exclusion, Outcome, TuningResult};
use std::collections::HashMap;

/// How the replayed evaluator costs configurations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Objective {
    /// The production objective: per point relative to the defaults, one
    /// measurement per point and key.
    Keyed,
    /// The previous objective: `Σ weight × median` in seconds, every point
    /// of every configuration measured separately.
    Separate,
}

/// One point of a recorded configuration.
#[derive(Clone, Debug)]
struct RecordedPoint {
    key: PointKey,
    samples: Vec<f64>,
    /// The median of every recorded sample.
    time: f64,
}

/// A survey as the replay reads it.
#[derive(Clone, Debug)]
pub struct Recording {
    space: SearchSpace,
    weights: Vec<f64>,
    classes: Vec<usize>,
    labels: Vec<String>,
    /// Per configuration index; `None` when the survey excluded it.
    points: Vec<Option<Vec<RecordedPoint>>>,
    /// True cost per configuration index (excluded: infinite).
    truth: Vec<f64>,
}

#[derive(Debug)]
pub enum RecordingError {
    /// The record's configurations do not form its declared space.
    Space(String),
    /// A measured configuration lacks samples at a point.
    Empty { point: String },
    /// The survey could not measure the defaults.
    DefaultExcluded,
}

impl std::fmt::Display for RecordingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Space(detail) => write!(
                f,
                "the survey's configurations do not form its space: {detail}"
            ),
            Self::Empty { point } => {
                write!(f, "a surveyed configuration has no samples at `{point}`")
            }
            Self::DefaultExcluded => f.write_str("the survey excluded the defaults"),
        }
    }
}

impl std::error::Error for RecordingError {}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values[middle]
    } else {
        (values[middle - 1] + values[middle]) / 2.0
    }
}

impl Recording {
    pub fn new(result: &TuningResult) -> Result<Self, RecordingError> {
        let declared = result
            .parameters
            .iter()
            .map(|parameter| (parameter.name.clone(), parameter.values.clone()))
            .collect::<Vec<_>>();
        let admissible = result
            .configurations
            .iter()
            .map(|record| record.configuration.params.clone())
            .collect::<Vec<_>>();
        let space = SearchSpace::new(&declared, &admissible)
            .map_err(|error| RecordingError::Space(error.to_string()))?;
        let mut points = vec![None; space.len()];
        for record in &result.configurations {
            let index = space
                .index_of(&record.configuration.params)
                .expect("every recorded configuration is in the space built from them");
            points[index] = match &record.outcome {
                Outcome::Excluded(_) => None,
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
                                time: median(&mut point.samples.clone()),
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                ),
            };
        }
        let weights = result
            .points
            .iter()
            .map(|point| point.weight)
            .collect::<Vec<_>>();
        let classes = search::classes(result.points.iter().map(|point| point.class.as_deref()));
        let labels = result
            .points
            .iter()
            .map(|point| point.label.clone())
            .collect();
        let reference = points[space.default_index()]
            .as_ref()
            .ok_or(RecordingError::DefaultExcluded)?
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
                    &classes,
                    &recorded.iter().map(|point| point.time).collect::<Vec<_>>(),
                    &reference,
                )
                .total(),
            })
            .collect();
        Ok(Self {
            space,
            weights,
            classes,
            labels,
            points,
            truth,
        })
    }

    pub fn space(&self) -> &SearchSpace {
        &self.space
    }

    /// Admissible configurations the survey measured.
    pub fn measured(&self) -> usize {
        self.points.iter().filter(|points| points.is_some()).count()
    }

    /// The configuration with the least true cost.
    pub fn best(&self) -> usize {
        (0..self.truth.len())
            .min_by(|left, right| self.truth[*left].total_cmp(&self.truth[*right]))
            .expect("the space holds the defaults")
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
                    local: self.space.values(local),
                }
            })
            .collect()
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
    recording: &'r Recording,
    objective: Objective,
    settings: &'r SearchSettings,
    draws: Draws,
    /// Search medians by point and key ([`Objective::Keyed`]).
    measured: HashMap<(usize, PointKey), f64>,
    /// The defaults' search medians.
    reference: Option<Vec<f64>>,
    /// After each evaluation: the true cost of the configuration the search
    /// estimates cheapest so far.
    curve: Vec<f64>,
    cheapest: Option<(f64, usize)>,
}

impl Recorded<'_> {
    fn draw(&mut self, point: &RecordedPoint, samples: usize) -> f64 {
        let mut drawn = (0..samples.max(1))
            .map(|_| point.samples[self.draws.index(point.samples.len())])
            .collect::<Vec<_>>();
        median(&mut drawn)
    }

    fn excluded() -> Exclusion {
        Exclusion::Measurement {
            point: String::new(),
            detail: "excluded by the survey".into(),
        }
    }

    /// The cost of `index` from `medians` at its points.
    fn cost(&self, index: usize, medians: &[f64], reference: &[f64]) -> Cost {
        let points = self.recording.points[index]
            .as_ref()
            .expect("a costed configuration was measured");
        match self.objective {
            Objective::Keyed => Cost::relative(
                points.iter().map(|point| point.key.clone()).collect(),
                &self.recording.weights,
                &self.recording.classes,
                medians,
                reference,
            ),
            // Every configuration's points differ from every other's.
            Objective::Separate => Cost::new(
                self.recording
                    .weights
                    .iter()
                    .zip(medians)
                    .map(|(weight, median)| {
                        (
                            PointKey {
                                launches: Vec::new(),
                                values: self.recording.space.values(index),
                            },
                            weight * median,
                        )
                    })
                    .collect(),
            ),
        }
    }

    /// Medians of `index` at every point, drawing `samples` per new
    /// measurement; `shared` holds the measurements this pass may reuse.
    fn medians(
        &mut self,
        index: usize,
        samples: usize,
        shared: &mut HashMap<(usize, PointKey), f64>,
    ) -> Option<Vec<f64>> {
        let recording = self.recording;
        let points = recording.points[index].as_ref()?;
        Some(
            points
                .iter()
                .enumerate()
                .map(|(point, recorded)| match self.objective {
                    Objective::Separate => self.draw(recorded, samples),
                    Objective::Keyed => {
                        let id = (point, recorded.key.clone());
                        match shared.get(&id) {
                            Some(median) => *median,
                            None => {
                                let median = self.draw(recorded, samples);
                                shared.insert(id, median);
                                median
                            }
                        }
                    }
                })
                .collect(),
        )
    }

    fn reference_of(&self, medians: &[f64]) -> Vec<f64> {
        match self.objective {
            Objective::Keyed => medians.to_vec(),
            Objective::Separate => vec![1.0; medians.len()],
        }
    }
}

impl Evaluator for Recorded<'_> {
    fn evaluate(&mut self, batch: &[usize]) -> Vec<Result<Cost, Exclusion>> {
        let mut measured = std::mem::take(&mut self.measured);
        let results = batch
            .iter()
            .map(|&index| {
                let Some(medians) = self.medians(index, self.settings.samples, &mut measured)
                else {
                    self.curve.push(
                        self.cheapest
                            .map_or(f64::INFINITY, |(_, best)| self.recording.excess(best)),
                    );
                    return Err(Self::excluded());
                };
                if index == self.recording.space.default_index() {
                    self.reference = Some(self.reference_of(&medians));
                }
                let reference = self
                    .reference
                    .as_ref()
                    .expect("the defaults are evaluated first");
                let cost = self.cost(index, &medians, reference);
                if self.cheapest.is_none_or(|(total, _)| cost.total() < total) {
                    self.cheapest = Some((cost.total(), index));
                }
                let (_, best) = self.cheapest.expect("a configuration was costed");
                self.curve.push(self.recording.excess(best));
                Ok(cost)
            })
            .collect();
        self.measured = measured;
        results
    }

    fn confirm(&mut self, finalists: &[usize]) -> Vec<Result<Cost, Exclusion>> {
        let mut shared = HashMap::new();
        let medians = finalists
            .iter()
            .map(|&index| self.medians(index, self.settings.confirmation_samples, &mut shared))
            .collect::<Vec<_>>();
        let Some(reference) = medians[0]
            .as_ref()
            .map(|medians| self.reference_of(medians))
        else {
            return finalists.iter().map(|_| Err(Self::excluded())).collect();
        };
        finalists
            .iter()
            .zip(medians)
            .map(|(&index, medians)| {
                medians
                    .map(|medians| self.cost(index, &medians, &reference))
                    .ok_or_else(Self::excluded)
            })
            .collect()
    }

    fn expired(&self) -> bool {
        false
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

/// Replay the search `runs` times over `recording` with `budget`
/// configurations, each run drawing from its own seed. The choice is the
/// first configuration of the search's ranking (validation always passes on
/// a survey record, which holds only validated configurations).
pub fn replay(
    recording: &Recording,
    budget: usize,
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
            recording,
            objective,
            settings,
            draws: Draws((run as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1),
            measured: HashMap::new(),
            reference: None,
            curve: Vec::new(),
            cheapest: None,
        };
        let trace = search::search(&recording.space, &[], budget, settings, &mut evaluator);
        replay.chosen.push(recording.excess(trace.ranking[0]));
        replay.evaluated.push(trace.evaluated.len());
        replay.curves.push(evaluator.curve);
    }
    replay
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::tune::{
        Configuration, ConfigurationRecord, DeclaredParameter, PointMeasurement, TuningMethod,
        TuningTime, Validation,
    };

    /// A survey of `A` (read by the first point) and `B` (read by the
    /// second, which runs a hundred times longer and is noisy).
    fn recording() -> Recording {
        let mut configurations = Vec::new();
        for a in [1u64, 2, 3, 4] {
            for b in [1u64, 2] {
                let values: ParameterValues = [("A".to_string(), a), ("B".to_string(), b)]
                    .into_iter()
                    .collect();
                let point = |label: &str, name: &str, base: f64, spread: f64| PointMeasurement {
                    point: label.into(),
                    key: PointKey {
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
                    configuration: Configuration {
                        statics: Default::default(),
                        params: values.clone(),
                        launches: Vec::new(),
                    },
                    outcome: Outcome::Measured {
                        artifact: String::new(),
                        points: vec![
                            point("m1", "A", 1e-4 * (1.0 - 0.1 * (a - 1) as f64), 0.001),
                            point("m256", "B", 1e-2 * if b == 1 { 1.0 } else { 1.001 }, 0.02),
                        ],
                        confirmed: Vec::new(),
                        validated: true,
                    },
                });
            }
        }
        let result = TuningResult {
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
            validation: Validation::BitExact,
            parameters: vec![
                DeclaredParameter {
                    name: "A".into(),
                    launch: None,
                    arithmetic: false,
                    values: vec![1, 2, 3, 4],
                },
                DeclaredParameter {
                    name: "B".into(),
                    launch: None,
                    arithmetic: false,
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
        };
        Recording::new(&result).unwrap()
    }

    fn settings() -> SearchSettings {
        SearchSettings {
            improvement: 0.01,
            restarts: 2,
            confirmed: 3,
            default_margin: 0.02,
            samples: 3,
            confirmation_samples: 7,
        }
    }

    #[test]
    fn the_keyed_objective_finds_the_decode_optimum_the_separate_one_misses() {
        let recording = recording();
        assert_eq!(recording.space().values(recording.best())["A"], 4);
        let keyed = replay(&recording, 8, &settings(), Objective::Keyed, 200);
        // `B` differs by 0.1% under 2% noise: either choice is within 1%.
        assert!(keyed.within(0.01) >= 0.95, "{:?}", keyed.quantile(0.95));
        let separate = replay(&recording, 8, &settings(), Objective::Separate, 200);
        assert!(separate.within(0.01) < 0.5, "{}", separate.within(0.01));
    }

    #[test]
    fn gaps_compare_the_shared_best_with_each_point_alone() {
        let gaps = recording().gaps();
        assert_eq!(gaps.len(), 2);
        assert!(gaps.iter().all(|gap| gap.gap().abs() < 1e-12), "{gaps:?}");
    }
}
