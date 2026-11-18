//! A survey's record ([`super::tune::Strategy::Survey`]): the work every
//! surveyed configuration runs at each tuning point, measured once.
//!
//! What runs at a point is decided by the entry parameters and by the
//! parameters of the launches that can run there; the parameters of every
//! other launch change nothing there. So a survey measures, at each point,
//! each distinct assignment of those parameters that some surveyed
//! configuration takes, and no more: the work it records is the sum of the
//! points' counts, not the product of the domains. A whole configuration's
//! cost at a point is the work its assignment there runs.
//!
//! The record's domain is every configuration whose assignment at each
//! point is one the record holds there. Each `where` conjunct reads the
//! entry parameters and at most one launch's, and every launch whose
//! parameters vary can run at some point, so each conjunct is decided by
//! one point's assignment, which a surveyed configuration takes: every
//! configuration of the domain is admissible, and every surveyed one is in
//! it.

use super::plan::TuningPartition;
use super::tune::{Configuration, ConfigurationRecord, DeclaredParameter, Exclusion, Outcome, PointMeasurement};
use seismic_lang::checked::NativeImplementation;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};

/// What a survey measured: the work each surveyed configuration runs at
/// each point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurveyRecord {
    /// Every surveyed parameter: the entry parameters, then each launch's,
    /// in declaration order, each with the values surveyed (one for a
    /// parameter no point can make read).
    pub parameters: Vec<DeclaredParameter>,
    /// The defaults: the reference of every cost.
    pub defaults: Configuration,
    /// The configurations formed and measured: the defaults, and one
    /// standing for each work no configuration formed before it ran.
    pub formed: usize,
    /// Per tuning point, in the points' order.
    pub points: Vec<SurveyedPoint>,
    /// The plan the unit's search orders the domain by: what a replay of
    /// the search over this record plans ([`super::replay`]).
    pub plan: TuningPartition,
}

/// What a survey measured at one point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurveyedPoint {
    /// The parameters that decide the work here, as positions in
    /// [`SurveyRecord::parameters`], in its order: every entry parameter,
    /// and every parameter of each launch that can run here.
    pub leading: Vec<usize>,
    /// Every assignment of `leading` a surveyed configuration takes, with
    /// the work it runs here.
    pub rows: Vec<SurveyedRow>,
    /// The distinct work the rows run here.
    pub work: Vec<SurveyedWork>,
}

/// One assignment of a point's leading parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurveyedRow {
    /// The value of each leading parameter, in `leading`'s order.
    pub values: Vec<u64>,
    /// The work it runs, an index into [`SurveyedPoint::work`].
    pub work: usize,
}

/// One work at one point, measured in one configuration that runs it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurveyedWork {
    /// The configuration it was measured in.
    pub configuration: Configuration,
    /// The artifact formed for that configuration; none when none was.
    pub artifact: Option<String>,
    /// Its observation against the defaults here, an index into
    /// [`super::tune::TuningResult::observations`]; none for the defaults'
    /// own artifact, the reference.
    pub observation: Option<usize>,
    /// Its measurement here, or why the configuration that stands for it
    /// could not be formed or measured.
    pub outcome: WorkOutcome,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum WorkOutcome {
    Measured(PointMeasurement),
    Excluded(Exclusion),
}

/// Every parameter of `implementation` as a survey records it: the entry
/// parameters, then each launch's, in declaration order.
pub(crate) fn surveyed_parameters(implementation: &NativeImplementation) -> Vec<DeclaredParameter> {
    let declared = |launch: Option<usize>, parameter: &seismic_lang::checked::NativeParameter| DeclaredParameter {
        name: parameter.name.clone(),
        launch,
        arithmetic: parameter.arithmetic,
        form: parameter.form,
        values: parameter.values.clone(),
    };
    implementation
        .params
        .iter()
        .map(|parameter| declared(None, parameter))
        .chain(implementation.launches.iter().enumerate().flat_map(|(launch, declaration)| {
            declaration.params.iter().map(move |parameter| declared(Some(launch), parameter))
        }))
        .collect()
}

/// `configuration`'s value of `parameter`.
fn value_of(configuration: &Configuration, parameter: &DeclaredParameter) -> Option<u64> {
    match parameter.launch {
        None => configuration.params.get(&parameter.name).copied(),
        Some(launch) => configuration.launches.get(launch)?.get(&parameter.name).copied(),
    }
}

/// `configuration` with `parameter` at `value`.
fn set(configuration: &mut Configuration, parameter: &DeclaredParameter, value: u64) {
    match parameter.launch {
        None => {
            configuration.params.insert(parameter.name.clone(), value);
        }
        Some(launch) => {
            if configuration.launches.len() <= launch {
                configuration.launches.resize_with(launch + 1, BTreeMap::new);
            }
            configuration.launches[launch].insert(parameter.name.clone(), value);
        }
    }
}

/// A record read: each point's rows by their values, and the record's
/// domain factored into what can be chosen apart.
///
/// A configuration is held as an assignment: its value of each surveyed
/// parameter, in [`SurveyRecord::parameters`]' order. With the entry
/// parameters held, the launches fall into groups no point reads two of:
/// the points that read a group's parameters read no other launch's, and
/// the rest read the entry parameters alone. The domain is therefore each
/// entry assignment the record holds times the product of its groups'
/// assignments, and whatever sums over the points (a cost, a count) is
/// found per group apart: in the sum of the groups' assignments, never in
/// the product of every launch's domain.
pub struct SurveyIndex<'r> {
    record: &'r SurveyRecord,
    rows: Vec<HashMap<&'r [u64], usize>>,
    /// The positions of the entry parameters, in order.
    entries: Vec<usize>,
    /// The groups of launch parameters chosen apart.
    groups: Vec<LaunchGroup>,
    /// The points that read no launch's parameters.
    entry_points: Vec<usize>,
    /// The defaults' assignment.
    defaults: Vec<u64>,
    /// The values a row is looked up by, reused across lookups.
    scratch: RefCell<Vec<u64>>,
}

/// Launches some point reads together, transitively.
struct LaunchGroup {
    /// Their parameters' positions, ascending.
    positions: Vec<usize>,
    /// The points that read them.
    points: Vec<usize>,
}

impl<'r> SurveyIndex<'r> {
    pub fn new(record: &'r SurveyRecord) -> Self {
        let rows = record
            .points
            .iter()
            .map(|point| point.rows.iter().map(|row| (row.values.as_slice(), row.work)).collect())
            .collect();
        let entries = (0..record.parameters.len())
            .filter(|&position| record.parameters[position].launch.is_none())
            .collect();
        let launch_of = |position: usize| record.parameters[position].launch;
        // The launches each point reads parameters of.
        let read = record
            .points
            .iter()
            .map(|point| {
                let mut launches = point.leading.iter().filter_map(|&position| launch_of(position)).collect::<Vec<_>>();
                launches.dedup();
                launches
            })
            .collect::<Vec<_>>();
        // Launches some point reads together share a group.
        let mut group = BTreeMap::<usize, usize>::new();
        for launches in &read {
            for &launch in launches {
                let next = group.len();
                group.entry(launch).or_insert(next);
            }
        }
        let mut parent = (0..group.len()).collect::<Vec<_>>();
        fn root(parent: &mut [usize], mut at: usize) -> usize {
            while parent[at] != at {
                parent[at] = parent[parent[at]];
                at = parent[at];
            }
            at
        }
        for launches in &read {
            for pair in launches.windows(2) {
                let (left, right) = (root(&mut parent, group[&pair[0]]), root(&mut parent, group[&pair[1]]));
                parent[left] = right;
            }
        }
        let mut groups = BTreeMap::<usize, LaunchGroup>::new();
        for (point, launches) in read.iter().enumerate() {
            if let Some(&launch) = launches.first() {
                let at = root(&mut parent, group[&launch]);
                let members = groups.entry(at).or_insert_with(|| LaunchGroup { positions: Vec::new(), points: Vec::new() });
                members.points.push(point);
                for &position in &record.points[point].leading {
                    if launch_of(position).is_some() && !members.positions.contains(&position) {
                        members.positions.push(position);
                    }
                }
            }
        }
        let groups = groups
            .into_values()
            .map(|mut group| {
                group.positions.sort_unstable();
                group
            })
            .collect();
        let entry_points = (0..record.points.len()).filter(|&point| read[point].is_empty()).collect();
        let defaults = record
            .parameters
            .iter()
            .map(|parameter| value_of(&record.defaults, parameter).expect("the defaults value every surveyed parameter"))
            .collect();
        Self { record, rows, entries, groups, entry_points, defaults, scratch: RefCell::new(Vec::new()) }
    }

    /// `configuration`'s assignment; none when it lacks a surveyed
    /// parameter.
    pub fn assignment(&self, configuration: &Configuration) -> Option<Vec<u64>> {
        self.record.parameters.iter().map(|parameter| value_of(configuration, parameter)).collect()
    }

    /// The configuration of `assignment`: the defaults with every surveyed
    /// parameter at its value there.
    pub fn configuration(&self, assignment: &[u64]) -> Configuration {
        let mut configuration = self.record.defaults.clone();
        for (parameter, value) in self.record.parameters.iter().zip(assignment) {
            set(&mut configuration, parameter, *value);
        }
        configuration
    }

    /// The defaults' assignment.
    pub fn defaults(&self) -> &[u64] {
        &self.defaults
    }

    /// The work `assignment` runs at `point`, as an index into the point's
    /// [`SurveyedPoint::work`]; none when the record holds no row of its
    /// assignment there. Allocates nothing.
    pub fn work_at(&self, point: usize, assignment: &[u64]) -> Option<usize> {
        let mut values = self.scratch.borrow_mut();
        values.clear();
        values.extend(self.record.points[point].leading.iter().map(|&position| assignment[position]));
        self.rows[point].get(values.as_slice()).copied()
    }

    /// Whether `assignment` is of the record's domain: each value surveyed,
    /// and a row of it at every point. Allocates nothing.
    pub fn holds(&self, assignment: &[u64]) -> bool {
        self.record.parameters.iter().zip(assignment).all(|(parameter, value)| parameter.values.contains(value))
            && (0..self.record.points.len()).all(|point| self.work_at(point, assignment).is_some())
    }

    /// The work `configuration` runs at `point`; none when the record
    /// holds no configuration with its assignment there.
    pub fn work(&self, point: usize, configuration: &Configuration) -> Option<&'r SurveyedWork> {
        let assignment = self.assignment(configuration)?;
        self.work_at(point, &assignment).map(|work| &self.record.points[point].work[work])
    }

    /// Whether `configuration` is of the record's domain.
    pub fn contains(&self, configuration: &Configuration) -> bool {
        self.assignment(configuration).is_some_and(|assignment| self.holds(&assignment))
    }

    /// `configuration`'s record, composed from the work it runs at each
    /// point: its measurement there and its observation, or the first
    /// point's exclusion. None when it is not of the record's domain.
    pub fn composed(&self, configuration: &Configuration) -> Option<ConfigurationRecord> {
        let work = (0..self.record.points.len())
            .map(|point| self.work(point, configuration))
            .collect::<Option<Vec<_>>>()?;
        let outcome = match work.iter().find_map(|work| match &work.outcome {
            WorkOutcome::Excluded(exclusion) => Some(exclusion.clone()),
            WorkOutcome::Measured(_) => None,
        }) {
            Some(exclusion) => Outcome::Excluded(exclusion),
            None => Outcome::Measured {
                points: work
                    .iter()
                    .map(|work| match &work.outcome {
                        WorkOutcome::Measured(measurement) => measurement.clone(),
                        WorkOutcome::Excluded(_) => unreachable!("an exclusion was found above"),
                    })
                    .collect(),
                confirmed: Vec::new(),
            },
        };
        Some(ConfigurationRecord {
            configuration: configuration.clone(),
            // Formed only when it is the configuration some work was
            // measured in.
            artifact: work
                .iter()
                .find(|work| work.configuration == *configuration)
                .and_then(|work| work.artifact.clone()),
            observations: work.iter().filter_map(|work| work.observation).collect(),
            outcome,
            formation: None,
            launches: Vec::new(),
        })
    }

    /// The assignments of the entry parameters the record's domain holds,
    /// in the declaration's walk order: those every point holds a row of.
    fn entry_assignments(&self) -> Vec<Vec<u64>> {
        let record = self.record;
        let Some(first) = record.points.first() else {
            return Vec::new();
        };
        // Each point's leading parameters hold every entry parameter.
        let projected = |point: &SurveyedPoint, values: &[u64]| {
            point
                .leading
                .iter()
                .zip(values)
                .filter(|(position, _)| record.parameters[**position].launch.is_none())
                .map(|(_, value)| *value)
                .collect::<Vec<_>>()
        };
        let held = record
            .points
            .iter()
            .map(|point| {
                point
                    .rows
                    .iter()
                    .map(|row| projected(point, &row.values))
                    .collect::<std::collections::HashSet<_>>()
            })
            .collect::<Vec<_>>();
        let ordinals = |values: &[u64]| {
            self.entries
                .iter()
                .zip(values)
                .map(|(&position, value)| ordinal(&record.parameters[position], *value))
                .collect::<Vec<_>>()
        };
        first
            .rows
            .iter()
            .map(|row| projected(first, &row.values))
            .filter(|values| held.iter().all(|point| point.contains(values)))
            .map(|values| (ordinals(&values), values))
            .collect::<BTreeMap<_, _>>()
            .into_values()
            .collect()
    }

    /// Each entry assignment of the domain in walk order, as the defaults'
    /// assignment with the entry parameters at it.
    fn entry_bases(&self) -> impl Iterator<Item = Vec<u64>> + '_ {
        self.entry_assignments().into_iter().map(|entry| {
            let mut base = self.defaults.clone();
            for (&position, value) in self.entries.iter().zip(entry) {
                base[position] = value;
            }
            base
        })
    }

    /// Each assignment of `group`'s parameters over their surveyed values,
    /// written into `assignment`, in walk order (the last position
    /// fastest): `visit` sees `assignment` with the group at each.
    fn walk_group(&self, group: &LaunchGroup, assignment: &mut [u64], mut visit: impl FnMut(&[u64])) {
        let values = |index: usize| &self.record.parameters[group.positions[index]].values;
        let mut ordinals = vec![0; group.positions.len()];
        loop {
            for (index, &position) in group.positions.iter().enumerate() {
                assignment[position] = values(index)[ordinals[index]];
            }
            visit(assignment);
            let Some(carry) = (0..ordinals.len()).rev().find(|&index| ordinals[index] + 1 < values(index).len()) else {
                return;
            };
            ordinals[carry] += 1;
            ordinals[carry + 1..].iter_mut().for_each(|ordinal| *ordinal = 0);
        }
    }

    /// The sum over `points` of `cost` of the work `assignment` runs at
    /// each; none when one has no row or its work cannot serve.
    fn cost_at(
        &self,
        points: &[usize],
        assignment: &[u64],
        cost: &impl Fn(usize, usize) -> Option<f64>,
    ) -> Option<f64> {
        points.iter().map(|&point| cost(point, self.work_at(point, assignment)?)).sum()
    }

    /// How many configurations of the record's domain run, at every point,
    /// work that `serves` (given the point and the work's index there).
    pub fn count(&self, serves: impl Fn(usize, usize) -> bool) -> u128 {
        let cost = |point: usize, work: usize| serves(point, work).then_some(0.);
        self.entry_bases()
            .filter(|base| self.cost_at(&self.entry_points, base, &cost).is_some())
            .map(|mut base| {
                self.groups
                    .iter()
                    .map(|group| {
                        let mut count = 0u128;
                        self.walk_group(group, &mut base, |assignment| {
                            count += u128::from(self.cost_at(&group.points, assignment, &cost).is_some());
                        });
                        count
                    })
                    .product::<u128>()
            })
            .sum()
    }

    /// The configuration of the record's domain whose work costs least,
    /// `cost` giving each work's cost at its point, by the point and the
    /// work's index there (none where it cannot serve), the first in the declaration's walk order of those that cost
    /// that; none when no configuration has work that serves at every
    /// point. The cost is a sum over the points, so each group's least is
    /// found apart from the others.
    pub fn best(&self, cost: impl Fn(usize, usize) -> Option<f64>) -> Option<Configuration> {
        let mut best: Option<(f64, Vec<u64>)> = None;
        for mut base in self.entry_bases() {
            let Some(mut total) = self.cost_at(&self.entry_points, &base, &cost) else {
                continue;
            };
            let mut served = true;
            for group in &self.groups {
                // The group's assignments in walk order, the first of the
                // least cost kept.
                let mut least: Option<(f64, Vec<u64>)> = None;
                self.walk_group(group, &mut base, |assignment| {
                    if let Some(cost) = self.cost_at(&group.points, assignment, &cost) {
                        if least.as_ref().is_none_or(|(least, _)| cost < *least) {
                            least = Some((cost, group.positions.iter().map(|&position| assignment[position]).collect()));
                        }
                    }
                });
                let Some((cost, chosen)) = least else {
                    served = false;
                    break;
                };
                total += cost;
                for (&position, value) in group.positions.iter().zip(chosen) {
                    base[position] = value;
                }
            }
            if served && best.as_ref().is_none_or(|(least, _)| total < *least) {
                best = Some((total, base));
            }
        }
        best.map(|(_, assignment)| self.configuration(&assignment))
    }
}

/// Where `value` stands in `parameter`'s surveyed values.
fn ordinal(parameter: &DeclaredParameter, value: u64) -> usize {
    parameter
        .values
        .iter()
        .position(|surveyed| *surveyed == value)
        .expect("a surveyed configuration takes a surveyed value")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::search::LaunchKey;

    /// A record over the entry parameter `E` and two launches' `X` and `Y`:
    /// the first point reads `E` and `X`, the second `E` and `Y`, the third
    /// `E` alone; some assignments are not surveyed. Each work's time is
    /// `time(point, values)`.
    fn record(time: impl Fn(usize, &[u64]) -> f64) -> SurveyRecord {
        let parameter = |name: &str, launch, values: Vec<u64>| DeclaredParameter {
            name: name.into(),
            launch,
            arithmetic: false,
            form: false,
            values,
        };
        let parameters = vec![
            parameter("E", None, vec![0, 1]),
            parameter("X", Some(0), vec![1, 2]),
            parameter("Y", Some(1), vec![3, 1, 2]),
        ];
        let leading = [vec![0, 1], vec![0, 2], vec![0]];
        let points = leading
            .iter()
            .enumerate()
            .map(|(point, leading)| {
                let mut surveyed = SurveyedPoint {
                    leading: leading.clone(),
                    rows: Vec::new(),
                    work: Vec::new(),
                };
                let mut values = vec![Vec::new()];
                for &position in leading {
                    values = values
                        .into_iter()
                        .flat_map(|prefix: Vec<u64>| {
                            parameters[position].values.iter().map(move |value| {
                                let mut values = prefix.clone();
                                values.push(*value);
                                values
                            })
                        })
                        .collect();
                }
                // `E` 1 with `Y` 2 is not surveyed.
                for values in values.into_iter().filter(|values| !(point == 1 && values == &[1, 2])) {
                    surveyed.work.push(SurveyedWork {
                        configuration: Configuration {
                            statics: BTreeMap::new(),
                            params: BTreeMap::new(),
                            launches: Vec::new(),
                        },
                        artifact: None,
                        observation: None,
                        outcome: WorkOutcome::Measured(PointMeasurement {
                            point: format!("p{point}"),
                            key: LaunchKey { launches: Vec::new(), values: BTreeMap::new() },
                            median_seconds: time(point, &values),
                            deviation_seconds: 0.,
                            samples: Vec::new(),
                            repetitions: 1,
                            rotation_bytes: 0,
                        }),
                    });
                    surveyed.rows.push(SurveyedRow { values, work: surveyed.work.len() - 1 });
                }
                surveyed
            })
            .collect();
        let mut defaults = Configuration {
            statics: BTreeMap::new(),
            params: BTreeMap::new(),
            launches: Vec::new(),
        };
        for (position, value) in [(0, 0), (1, 1), (2, 3)] {
            set(&mut defaults, &parameters[position], value);
        }
        SurveyRecord { parameters, defaults, formed: 0, points, plan: crate::native::plan::whole(&[], 2, Vec::new()) }
    }

    /// Every configuration of `record`'s domain in walk order, by brute force.
    fn every(record: &SurveyRecord) -> Vec<Configuration> {
        let index = SurveyIndex::new(record);
        let mut every = Vec::new();
        for e in [0, 1] {
            for x in [1, 2] {
                for y in [3, 1, 2] {
                    let mut configuration = record.defaults.clone();
                    for (position, value) in [(0, e), (1, x), (2, y)] {
                        set(&mut configuration, &record.parameters[position], value);
                    }
                    if index.contains(&configuration) {
                        every.push(configuration);
                    }
                }
            }
        }
        every
    }

    /// The least cost the record's domain holds, the first in walk order of
    /// those that cost it, is the brute force's over every configuration of
    /// the domain: the launches no point reads together are chosen apart.
    #[test]
    fn the_best_configuration_is_the_least_of_the_whole_domain() {
        let times: [&dyn Fn(usize, &[u64]) -> f64; 4] = [
            &|point, values| (point + 1) as f64 * values.iter().sum::<u64>() as f64,
            &|point, values| ((point * 7 + values.iter().map(|value| value * 3 + 1).product::<u64>() as usize) % 5) as f64,
            &|_, _| 1.,
            &|point, values| if point == 1 && values == [0, 2] { 0. } else { 10. - values[0] as f64 },
        ];
        for time in times {
            let record = record(time);
            let index = SurveyIndex::new(&record);
            let every = every(&record);
            assert_eq!(every.len(), 10, "E 1 with Y 2 is outside the domain");
            let cost = |configuration: &Configuration| {
                (0..record.points.len())
                    .map(|point| match &index.work(point, configuration).unwrap().outcome {
                        WorkOutcome::Measured(measurement) => measurement.median_seconds,
                        WorkOutcome::Excluded(_) => unreachable!("every work is measured"),
                    })
                    .sum::<f64>()
            };
            let least = every.iter().map(cost).fold(f64::INFINITY, f64::min);
            let expected = every.iter().find(|configuration| cost(configuration) == least).unwrap();
            let best = index.best(|point, work| match &record.points[point].work[work].outcome {
                WorkOutcome::Measured(measurement) => Some(measurement.median_seconds),
                WorkOutcome::Excluded(_) => None,
            });
            assert_eq!(index.count(|_, _| true), every.len() as u128);
            assert_eq!(best.as_ref(), Some(expected));
        }
    }
}
