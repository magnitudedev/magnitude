//! Time-bounded local search over a native implementation's declared
//! parameter space, as a pure function of what an [`Evaluator`] reports.
//!
//! The search walks the grid of admissible configurations: each parameter's
//! values in numeric order, neighbors one step apart in one parameter. It
//! starts from the defaults, one configuration per other value of each form
//! parameter (the admissible configuration with that value nearest the
//! defaults) and the consumer's hints. From the cheapest start of each form,
//! cheapest form first, it moves to the best neighbor while that improves
//! the cost by more than `improvement`; then, at each local minimum, it
//! restarts from the unvisited configuration farthest from everything
//! visited. It stops when every configuration was visited, `restarts`
//! consecutive restarts failed to improve the best cost, or the evaluator's
//! time ran out. The `confirmed` cheapest
//! configurations and the defaults are then re-measured, alternating, and
//! ranked by those costs; a finalist the evaluator cannot confirm (its
//! re-measurement failed or is not trustworthy) leaves the ranking, and the
//! defaults rank first unless the leader beats them by `default_margin` or
//! could not be confirmed themselves. When the defaults passed the search but
//! their re-measurement did not, they are the choice; only defaults that
//! failed the search itself leave the ranking to confirmed finalists.
//!
//! The objective ([`Cost`]) is per point: each tuning point contributes its
//! weight (share of step time) times the configuration's time there relative
//! to the defaults' time there, so a point's vote does not scale with how
//! long its workload runs. Points of one class (the same rows at different
//! history lengths) split their class's weight by real time instead
//! ([`Cost::relative`]). A point votes only between configurations that
//! run different work there ([`PointKey`]); configurations with the same key
//! at a point share one measurement of it, and margins are taken relative to
//! the points that tell two configurations apart.
//!
//! Given the evaluator's costs and where its time ran out, the procedure is
//! deterministic, so a replay of recorded surveys (tuning spec §E2) can run
//! this same code with a recorded evaluator.

use super::tune::Exclusion;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// A configuration's parameter values by name.
pub type ParameterValues = BTreeMap<String, u64>;

/// What a configuration's measurement at one point depends on: the launches
/// active there (by declaration ordinal) and the values of the parameters
/// those launches read. Configurations with the same key at a point run the
/// same work there: the point cannot tell them apart, so it is measured once
/// for all of them and does not vote between them.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PointKey {
    pub launches: Vec<usize>,
    pub values: ParameterValues,
}

/// One configuration's cost: per tuning point, its key and its weighted
/// relative time `weight × median / defaults' median`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cost {
    points: Vec<(PointKey, f64)>,
}

impl Cost {
    /// `points` in the tuning points' order.
    pub fn new(points: Vec<(PointKey, f64)>) -> Self {
        Self { points }
    }

    /// The cost of measurements at keyed points relative to the reference
    /// (the defaults' medians), which costs `Σ weights`.
    ///
    /// Each class of points ([`classes`]) carries its summed weight `W`,
    /// split among its points in proportion to `weight × reference`: point
    /// `p` costs `W × weight_p × median_p / Σ_class weight × reference`. The
    /// points of a class are variants of one workload (the same rows at
    /// different history lengths) occurring as often as their weights say,
    /// so each counts by its share of the class's real time: a 90 ms variant
    /// outweighs a 0.1 ms one. A point in a class of its own costs
    /// `weight × median / reference`, its time relative to the defaults'.
    pub fn relative(
        keys: Vec<PointKey>,
        weights: &[f64],
        classes: &[usize],
        medians: &[f64],
        reference: &[f64],
    ) -> Self {
        let count = classes.iter().copied().max().map_or(0, |class| class + 1);
        let mut weight = vec![0.0; count];
        let mut time = vec![0.0; count];
        for (point, &class) in classes.iter().enumerate() {
            weight[class] += weights[point];
            time[class] += weights[point] * reference[point];
        }
        Self::new(
            keys.into_iter()
                .enumerate()
                .map(|(point, key)| {
                    let class = classes[point];
                    (
                        key,
                        weight[class] * weights[point] * medians[point] / time[class],
                    )
                })
                .collect(),
        )
    }

    pub fn total(&self) -> f64 {
        self.points.iter().map(|(_, cost)| cost).sum()
    }

    /// Whether this cost beats `other` by more than `margin` of what the
    /// points that tell the two apart cost `other`. Points with equal keys
    /// share one measurement and cancel; a margin of the whole cost would let
    /// the points that cannot tell them apart dilute it.
    pub fn improves_on(&self, other: &Cost, margin: f64) -> bool {
        let (mine, theirs) = self
            .points
            .iter()
            .zip(&other.points)
            .filter(|((key, _), (other, _))| key != other)
            .fold((0.0, 0.0), |(mine, theirs), ((_, cost), (_, other))| {
                (mine + cost, theirs + other)
            });
        mine < theirs * (1.0 - margin)
    }
}

/// Class indices of points named by class (`None`: a class of its own), in
/// order of first appearance.
pub fn classes<'n>(names: impl IntoIterator<Item = Option<&'n str>>) -> Vec<usize> {
    let mut named: Vec<(&str, usize)> = Vec::new();
    let mut count = 0;
    names
        .into_iter()
        .map(|name| {
            if let Some(&(_, class)) =
                name.and_then(|name| named.iter().find(|(known, _)| *known == name))
            {
                return class;
            }
            let class = count;
            count += 1;
            named.extend(name.map(|name| (name, class)));
            class
        })
        .collect()
}

/// Search constants. Part of a tuning result's identity: changing any of
/// them changes what the search may choose.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchSettings {
    /// ε: the relative cost improvement a move or a restart must exceed.
    pub improvement: f64,
    /// R: consecutive non-improving restarts that end the search.
    pub restarts: usize,
    /// K: the cheapest configurations besides the defaults confirmed.
    pub confirmed: usize,
    /// δ: the relative margin by which the confirmed leader must beat the
    /// defaults to rank above them.
    pub default_margin: f64,
    /// Samples per point while searching.
    pub samples: usize,
    /// Samples per point of each confirmed configuration.
    pub confirmation_samples: usize,
}

/// One declared parameter.
#[derive(Clone, Debug)]
pub struct SearchParameter {
    pub name: String,
    /// Its values in declaration order; the first is the default.
    pub values: Vec<u64>,
    /// Its values select structurally different forms of the implementation
    /// (declared `form`), each searched from a start of its own: one form's
    /// defaults tell nothing of another's.
    pub form: bool,
}

/// A declared parameter space and its admissible configurations.
#[derive(Clone, Debug)]
pub struct SearchSpace {
    /// Parameter names in declaration order, each with its values in numeric
    /// order.
    parameters: Vec<(String, Vec<u64>)>,
    /// Positions of the form parameters.
    forms: Vec<usize>,
    /// Admissible configurations, as a value index per parameter.
    configurations: Vec<Vec<u32>>,
    /// Each admissible configuration's index.
    positions: HashMap<Vec<u32>, usize>,
    /// Index of the all-defaults configuration.
    default: usize,
}

/// Why a declared space cannot be searched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SearchSpaceError {
    /// A configuration values a parameter the declaration lacks, lacks one
    /// it declares, or takes an undeclared value.
    Undeclared(ParameterValues),
    /// The defaults are not among the admissible configurations.
    DefaultInadmissible,
}

impl std::fmt::Display for SearchSpaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Undeclared(values) => {
                write!(
                    f,
                    "configuration {values:?} does not match the declared parameters"
                )
            }
            Self::DefaultInadmissible => {
                f.write_str("the all-defaults configuration is inadmissible")
            }
        }
    }
}

impl std::error::Error for SearchSpaceError {}

impl SearchSpace {
    /// `declared` lists the parameters in declaration order; `admissible`
    /// lists the configurations the declaration's `where` admits.
    pub fn new(
        declared: &[SearchParameter],
        admissible: &[ParameterValues],
    ) -> Result<Self, SearchSpaceError> {
        let parameters = declared
            .iter()
            .map(|parameter| {
                let mut sorted = parameter.values.clone();
                sorted.sort_unstable();
                sorted.dedup();
                (parameter.name.clone(), sorted)
            })
            .collect::<Vec<_>>();
        let forms = (0..declared.len())
            .filter(|position| declared[*position].form)
            .collect();
        let coordinates = |values: &ParameterValues| -> Result<Vec<u32>, SearchSpaceError> {
            if values.len() != parameters.len() {
                return Err(SearchSpaceError::Undeclared(values.clone()));
            }
            parameters
                .iter()
                .map(|(name, sorted)| {
                    values
                        .get(name)
                        .and_then(|value| sorted.binary_search(value).ok())
                        .map(|index| index as u32)
                        .ok_or_else(|| SearchSpaceError::Undeclared(values.clone()))
                })
                .collect()
        };
        let configurations = admissible
            .iter()
            .map(coordinates)
            .collect::<Result<Vec<_>, _>>()?;
        let defaults = declared
            .iter()
            .map(|parameter| (parameter.name.clone(), parameter.values[0]))
            .collect::<ParameterValues>();
        let positions = configurations
            .iter()
            .enumerate()
            .map(|(index, coordinates)| (coordinates.clone(), index))
            .collect::<HashMap<_, _>>();
        let default = *positions
            .get(&coordinates(&defaults)?)
            .ok_or(SearchSpaceError::DefaultInadmissible)?;
        Ok(Self {
            parameters,
            forms,
            configurations,
            positions,
            default,
        })
    }

    pub fn len(&self) -> usize {
        self.configurations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.configurations.is_empty()
    }

    pub fn default_index(&self) -> usize {
        self.default
    }

    /// The parameter values of configuration `index`.
    pub fn values(&self, index: usize) -> ParameterValues {
        self.parameters
            .iter()
            .zip(&self.configurations[index])
            .map(|((name, values), step)| (name.clone(), values[*step as usize]))
            .collect()
    }

    /// The index of the configuration with `values`, when admissible.
    pub fn index_of(&self, values: &ParameterValues) -> Option<usize> {
        let coordinates = self
            .parameters
            .iter()
            .map(|(name, sorted)| {
                values
                    .get(name)
                    .and_then(|value| sorted.binary_search(value).ok())
                    .map(|index| index as u32)
            })
            .collect::<Option<Vec<_>>>()?;
        (values.len() == self.parameters.len())
            .then(|| self.positions.get(&coordinates).copied())
            .flatten()
    }

    fn distance(&self, left: usize, right: usize) -> u32 {
        self.configurations[left]
            .iter()
            .zip(&self.configurations[right])
            .map(|(left, right)| left.abs_diff(*right))
            .sum()
    }

    /// The form of configuration `index`: its form parameters' values.
    fn form(&self, index: usize) -> Vec<u32> {
        self.forms
            .iter()
            .map(|position| self.configurations[index][*position])
            .collect()
    }

    /// One start per other value of each form parameter: the admissible
    /// configuration with that value nearest the defaults (ties go to the
    /// lowest index). A value no configuration admits has none.
    pub fn form_starts(&self) -> Vec<usize> {
        self.form_starts_with(&[])
    }

    /// Prefer an admissible consumer hint for a form value when one exists.
    /// This lets a workload choose where to begin within a structural form
    /// without spending an extra measurement beyond that form's first start.
    fn form_starts_with(&self, hints: &[usize]) -> Vec<usize> {
        let defaults = &self.configurations[self.default];
        let mut starts = Vec::new();
        for &position in &self.forms {
            for value in 0..self.parameters[position].1.len() as u32 {
                if value == defaults[position] {
                    continue;
                }
                let hinted = hints.iter().copied().find(|index| {
                    *index < self.len() && self.configurations[*index][position] == value
                });
                starts.extend(hinted.or_else(|| {
                    (0..self.len())
                        .filter(|index| self.configurations[*index][position] == value)
                        .min_by_key(|index| self.distance(*index, self.default))
                }));
            }
        }
        starts
    }
}

/// Measures configurations of a [`SearchSpace`] by index.
pub trait Evaluator {
    /// Form and measure a batch; one cost per configuration, in order. A
    /// configuration that cannot be formed or run is excluded. The search's
    /// first batch starts with the defaults, the reference of every
    /// [`Cost`]. The evaluator answers at least the first configuration and
    /// stops early, answering a prefix, when its time runs out.
    fn evaluate(&mut self, batch: &[usize]) -> Vec<Result<Cost, Exclusion>>;
    /// Re-measure `finalists` (every one evaluated before; the defaults
    /// first, the reference of the new costs), alternating between them
    /// sample by sample; their new costs, in order.
    fn confirm(&mut self, finalists: &[usize]) -> Vec<Result<Cost, Exclusion>>;
    /// Whether the evaluator's time has run out. The search then ends with
    /// what it has reached.
    fn expired(&self) -> bool;
}

/// How far behind the leader a configuration's search cost may be and still
/// be confirmed: past this, timing noise cannot reverse the ranking. It is
/// the spread at which confirmation already rejects a configuration's
/// samples as unstable.
pub(super) const CONTENDING: f64 = 0.10;

/// Why the search stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchStop {
    /// Every admissible configuration was evaluated.
    Exhausted,
    /// The allowed consecutive restarts found nothing better.
    Converged,
    /// The time ran out; the result is the best found so far.
    Expired,
    /// The unit's time could not cover the points every candidate must
    /// pass; it keeps its defaults without searching.
    Unaffordable,
}

/// Everything one search did.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchTrace {
    /// Every configuration evaluated, in order, with its search cost.
    pub evaluated: Vec<(usize, Result<Cost, Exclusion>)>,
    /// The finalists re-measured, with their confirmed costs.
    pub confirmed: Vec<(usize, Result<Cost, Exclusion>)>,
    /// Confirmed eligible configurations, cheapest first (subject to the default margin).
    pub ranking: Vec<usize>,
    pub stop: SearchStop,
}

struct State<'s, E> {
    space: &'s SearchSpace,
    evaluator: &'s mut E,
    /// The cost of every evaluated configuration; `None` when excluded.
    costs: HashMap<usize, Option<Cost>>,
    evaluated: Vec<(usize, Result<Cost, Exclusion>)>,
    /// Least distance from each configuration to any evaluated one.
    nearest: Vec<u32>,
    stop: Option<SearchStop>,
}

/// Order of total costs; an excluded configuration is last.
fn total(cost: Option<&Cost>) -> f64 {
    cost.map_or(f64::INFINITY, Cost::total)
}

/// Whether `cost` beats `over` by more than `margin`; an excluded
/// configuration beats nothing, and anything measured beats one.
fn improves(cost: Option<&Cost>, over: Option<&Cost>, margin: f64) -> bool {
    match (cost, over) {
        (Some(cost), Some(over)) => cost.improves_on(over, margin),
        (Some(_), None) => true,
        (None, _) => false,
    }
}

impl<E: Evaluator> State<'_, E> {
    fn cost(&self, index: usize) -> Option<&Cost> {
        self.costs[&index].as_ref()
    }

    fn record(&mut self, index: usize, result: Result<Cost, Exclusion>) {
        self.costs.insert(index, result.as_ref().ok().cloned());
        for (other, nearest) in self.nearest.iter_mut().enumerate() {
            *nearest = (*nearest).min(self.space.distance(index, other));
        }
        self.evaluated.push((index, result));
    }

    fn visited(&self, index: usize) -> bool {
        self.costs.contains_key(&index)
    }

    /// Evaluate the unvisited configurations of `batch` in order, as far as
    /// the evaluator's time reaches. Sets the stop reason when nothing more
    /// may be evaluated.
    fn evaluate(&mut self, batch: &[usize]) {
        if self.stop.is_some() {
            return;
        }
        if self.evaluator.expired() {
            self.stop = Some(SearchStop::Expired);
            return;
        }
        let mut fresh = Vec::new();
        for &index in batch {
            if !self.visited(index) && !fresh.contains(&index) {
                fresh.push(index);
            }
        }
        if fresh.is_empty() {
            return;
        }
        let results = self.evaluator.evaluate(&fresh);
        assert!(
            !results.is_empty() && results.len() <= fresh.len(),
            "an evaluator answers a nonempty prefix of the batch"
        );
        let answered = results.len();
        for (index, result) in fresh.iter().copied().zip(results) {
            self.record(index, result);
        }
        if self.evaluated.len() == self.space.len() {
            self.stop = Some(SearchStop::Exhausted);
        } else if answered < fresh.len() {
            self.stop = Some(SearchStop::Expired);
        }
    }

    /// Neighbors of `index` in fixed order: parameters in declaration
    /// order, one step down before one step up.
    fn neighbors(&self, index: usize) -> Vec<usize> {
        let space = self.space;
        let origin = &space.configurations[index];
        let mut neighbors = Vec::new();
        for (parameter, (_, values)) in space.parameters.iter().enumerate() {
            let step = origin[parameter];
            let moves = [
                step.checked_sub(1),
                (step as usize + 1 < values.len()).then_some(step + 1),
            ];
            for target in moves.into_iter().flatten() {
                let mut coordinates = origin.clone();
                coordinates[parameter] = target;
                if let Some(neighbor) = space.positions.get(&coordinates) {
                    neighbors.push(*neighbor);
                }
            }
        }
        neighbors
    }

    /// The cheapest of `candidates` that were evaluated; ties go to the
    /// first.
    fn cheapest(&self, candidates: impl IntoIterator<Item = usize>) -> Option<usize> {
        candidates
            .into_iter()
            .filter(|index| self.visited(*index))
            .min_by(|left, right| total(self.cost(*left)).total_cmp(&total(self.cost(*right))))
    }

    /// Descend from `current` until no neighbor improves on it by more
    /// than the settings' improvement.
    fn descend(&mut self, mut current: usize, improvement: f64) {
        loop {
            let neighbors = self.neighbors(current);
            self.evaluate(&neighbors);
            match self.cheapest(neighbors) {
                Some(best) if improves(self.cost(best), self.cost(current), improvement) => {
                    current = best;
                }
                _ => return,
            }
            if self.stop.is_some() {
                return;
            }
        }
    }

    /// The cheapest configuration evaluated so far.
    fn best(&self) -> usize {
        self.cheapest(self.evaluated.iter().map(|(index, _)| *index))
            .expect("the defaults were evaluated")
    }

    /// The unvisited configuration farthest from every visited one; ties go
    /// to the lowest index.
    fn farthest(&self) -> Option<usize> {
        (0..self.space.len())
            .filter(|index| !self.visited(*index))
            .fold(None, |best: Option<usize>, index| match best {
                Some(best) if self.nearest[best] >= self.nearest[index] => Some(best),
                _ => Some(index),
            })
    }
}

/// Search `space` from the defaults, its form starts and the `start`
/// configurations. A hint for a structural form replaces that form's usual
/// nearest-default start, so it uses the same measurement slot. The defaults
/// are always evaluated; everything else as far as the evaluator's time
/// reaches.
pub fn search(
    space: &SearchSpace,
    start: &[usize],
    settings: &SearchSettings,
    evaluator: &mut impl Evaluator,
) -> SearchTrace {
    let mut state = State {
        space,
        evaluator,
        costs: HashMap::new(),
        evaluated: Vec::new(),
        nearest: vec![u32::MAX; space.len()],
        stop: None,
    };
    let default = space.default;
    let starts = std::iter::once(default)
        .chain(space.form_starts_with(start))
        .chain(start.iter().copied().filter(|index| *index < space.len()))
        .collect::<Vec<_>>();
    // The defaults are the reference of every cost: evaluated even when the
    // time has already run out.
    if state.evaluator.expired() {
        state.stop = Some(SearchStop::Expired);
        let results = state.evaluator.evaluate(&[default]);
        let result = results
            .into_iter()
            .next()
            .expect("an evaluator answers the first configuration");
        state.record(default, result);
    } else {
        state.evaluate(&starts);
    }
    // The cheapest measured start of each form, cheapest form first.
    let mut descents: Vec<usize> = Vec::new();
    for index in starts
        .iter()
        .copied()
        .filter(|index| state.visited(*index) && state.cost(*index).is_some())
    {
        match descents
            .iter_mut()
            .find(|other| space.form(**other) == space.form(index))
        {
            Some(other) => *other = state.cheapest([*other, index]).expect("both evaluated"),
            None => descents.push(index),
        }
    }
    descents.sort_by(|left, right| total(state.cost(*left)).total_cmp(&total(state.cost(*right))));
    for current in descents {
        if state.stop.is_some() {
            break;
        }
        state.descend(current, settings.improvement);
    }
    let mut failed_restarts = 0;
    while state.stop.is_none() && failed_restarts < settings.restarts {
        let before = state.best();
        let Some(restart) = state.farthest() else {
            break;
        };
        state.evaluate(&[restart]);
        if state.visited(restart) && state.stop.is_none() {
            state.descend(restart, settings.improvement);
        }
        let after = state.best();
        if improves(state.cost(after), state.cost(before), settings.improvement) {
            failed_restarts = 0;
        } else {
            failed_restarts += 1;
        }
    }
    let stop = state.stop.unwrap_or(SearchStop::Converged);
    let State {
        evaluated,
        costs,
        evaluator,
        ..
    } = state;

    // Confirm the cheapest measured configurations against the defaults.
    let mut cheapest = evaluated
        .iter()
        .filter(|(index, result)| *index != default && result.is_ok())
        .map(|(index, _)| *index)
        .collect::<Vec<_>>();
    cheapest.sort_by(|left, right| {
        total(costs[left].as_ref()).total_cmp(&total(costs[right].as_ref()))
    });
    cheapest.truncate(settings.confirmed);
    // Only configurations within noise of the leader can change the ranking
    // when re-measured; the rest are not confirmed. The defaults always are:
    // they are the reference of the confirmed costs.
    let leader = cheapest.first().map(|index| total(costs[index].as_ref()));
    let contends = |index: &usize| {
        leader.is_none_or(|leader| total(costs[index].as_ref()) <= leader * (1. + CONTENDING))
    };
    cheapest.retain(contends);
    let finalists = costs
        .get(&default)
        .and_then(|cost| cost.as_ref())
        .map(|_| default)
        .into_iter()
        .chain(cheapest)
        .collect::<Vec<_>>();
    // Confirmation ranks the finalists against the defaults; the defaults
    // alone (every other candidate failed, or the safety stop came first)
    // have nothing to rank against and remain the choice unconfirmed.
    let confirmed = if finalists.iter().any(|index| *index != default) {
        let results = evaluator.confirm(&finalists);
        assert_eq!(
            results.len(),
            finalists.len(),
            "an evaluator confirms every finalist"
        );
        finalists.iter().copied().zip(results).collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let confirmed_cost = |index: usize| {
        confirmed
            .iter()
            .find(|(candidate, _)| *candidate == index)
            .and_then(|(_, result)| result.as_ref().ok())
    };
    let ranked = || {
        let mut ranked: Vec<_> = finalists
            .iter()
            .copied()
            .filter(|index| confirmed_cost(*index).is_some())
            .collect();
        ranked.sort_by(|left, right| {
            total(confirmed_cost(*left)).total_cmp(&total(confirmed_cost(*right)))
        });
        ranked
    };
    let ranking = match confirmed_cost(default) {
        Some(reference) => {
            let mut ranked = ranked();
            let leader = ranked[0];
            let leader_cost = confirmed_cost(leader).expect("ranked finalists were confirmed");
            // A leader that does not improve on the defaults by the margin
            // gives way to them.
            if leader != default && !leader_cost.improves_on(reference, settings.default_margin) {
                ranked.retain(|index| *index != default);
                ranked.insert(0, default);
            }
            ranked
        }
        // Re-measuring the defaults failed, but they passed validation and
        // measurement in the search: timing noise is not a numerical
        // verdict, so the defaults are the choice.
        None if finalists.contains(&default) => vec![default],
        // The defaults themselves failed: only confirmed finalists can rank.
        None => ranked(),
    };
    SearchTrace {
        evaluated,
        confirmed,
        ranking,
        stop,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn space(declared: &[(&str, &[u64])]) -> SearchSpace {
        forms_space(declared, &[], |_| true)
    }

    /// The configurations of `declared` that `admits`, with `forms` declared
    /// form parameters.
    fn forms_space(
        declared: &[(&str, &[u64])],
        forms: &[&str],
        admits: impl Fn(&ParameterValues) -> bool,
    ) -> SearchSpace {
        let declared = declared
            .iter()
            .map(|(name, values)| SearchParameter {
                name: name.to_string(),
                values: values.to_vec(),
                form: forms.contains(name),
            })
            .collect::<Vec<_>>();
        let mut admissible = vec![ParameterValues::new()];
        for parameter in &declared {
            admissible = admissible
                .into_iter()
                .flat_map(|base| {
                    parameter.values.iter().map(move |value| {
                        let mut next = base.clone();
                        next.insert(parameter.name.clone(), *value);
                        next
                    })
                })
                .collect();
        }
        admissible.retain(|values| admits(values));
        SearchSpace::new(&declared, &admissible).unwrap()
    }

    /// A deterministic evaluator over a cost function of parameter values,
    /// whose time runs out after `limit` evaluations.
    struct Exact<'s, F> {
        space: &'s SearchSpace,
        cost: F,
        evaluations: Vec<usize>,
        limit: usize,
    }

    impl<F: Fn(&ParameterValues) -> f64> Exact<'_, F> {
        /// One point that every parameter changes.
        fn cost(&self, index: usize) -> Result<Cost, Exclusion> {
            let values = self.space.values(index);
            let cost = (self.cost)(&values);
            Ok(Cost::new(vec![(
                PointKey {
                    launches: vec![0],
                    values,
                },
                cost,
            )]))
        }
    }

    impl<F: Fn(&ParameterValues) -> f64> Evaluator for Exact<'_, F> {
        fn evaluate(&mut self, batch: &[usize]) -> Vec<Result<Cost, Exclusion>> {
            let answered = batch
                .len()
                .min(self.limit.saturating_sub(self.evaluations.len()).max(1));
            self.evaluations.extend_from_slice(&batch[..answered]);
            batch[..answered]
                .iter()
                .map(|index| self.cost(*index))
                .collect()
        }
        fn confirm(&mut self, finalists: &[usize]) -> Vec<Result<Cost, Exclusion>> {
            finalists.iter().map(|index| self.cost(*index)).collect()
        }
        fn expired(&self) -> bool {
            self.evaluations.len() >= self.limit
        }
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
    fn descends_to_an_interacting_optimum_the_separable_search_misses() {
        // Cost is lowest where A · B = 16; single-parameter moves from the
        // defaults (A 1, B 1) each only halve the product's distance.
        let space = space(&[("A", &[1, 2, 4, 8, 16]), ("B", &[1, 2, 4, 8, 16])]);
        let mut evaluator = Exact {
            space: &space,
            cost: |values: &ParameterValues| {
                let product = (values["A"] * values["B"]) as f64;
                1.0 + (product.log2() - 4.0).abs() + 0.01 * values["A"] as f64
            },
            evaluations: Vec::new(),
            limit: usize::MAX,
        };
        let trace = search(&space, &[], &settings(), &mut evaluator);
        let chosen = space.values(trace.ranking[0]);
        assert_eq!(chosen["A"] * chosen["B"], 16, "{chosen:?}");
        assert_eq!(chosen["A"], 1, "{chosen:?}");
    }

    /// Two points: the first reads only `A`, the second only `B` and costs
    /// nine times as much.
    struct Split<'s> {
        space: &'s SearchSpace,
    }

    impl Split<'_> {
        fn cost(&self, index: usize) -> Result<Cost, Exclusion> {
            let values = self.space.values(index);
            let key = |name: &str, launch: usize| PointKey {
                launches: vec![launch],
                values: [(name.to_string(), values[name])].into_iter().collect(),
            };
            let small = 0.1 * (1.0 - 0.03 * (values["A"] - 1) as f64);
            let large = 0.9 * if values["B"] == 1 { 1.0 } else { 1.5 };
            Ok(Cost::new(vec![(key("A", 0), small), (key("B", 1), large)]))
        }
    }

    impl Evaluator for Split<'_> {
        fn evaluate(&mut self, batch: &[usize]) -> Vec<Result<Cost, Exclusion>> {
            batch.iter().map(|index| self.cost(*index)).collect()
        }
        fn confirm(&mut self, finalists: &[usize]) -> Vec<Result<Cost, Exclusion>> {
            finalists.iter().map(|index| self.cost(*index)).collect()
        }
        fn expired(&self) -> bool {
            false
        }
    }

    #[test]
    fn margins_are_relative_to_the_points_that_tell_configurations_apart() {
        // Each step of `A` gains 3% at a point holding a tenth of the cost:
        // 0.3% of the whole, below the 1% improvement, but 3% of the only
        // point that tells the configurations apart.
        let space = space(&[("A", &[1, 2, 3, 4]), ("B", &[1, 2])]);
        let mut evaluator = Split { space: &space };
        let trace = search(&space, &[], &settings(), &mut evaluator);
        let chosen = space.values(trace.ranking[0]);
        assert_eq!((chosen["A"], chosen["B"]), (4, 1), "{chosen:?}");
    }

    #[test]
    fn a_small_space_is_covered_exactly_and_never_past_its_time() {
        let space = space(&[("A", &[4, 2, 8])]);
        let mut evaluator = Exact {
            space: &space,
            cost: |values: &ParameterValues| values["A"] as f64,
            evaluations: Vec::new(),
            limit: usize::MAX,
        };
        let trace = search(&space, &[], &settings(), &mut evaluator);
        assert_eq!(trace.stop, SearchStop::Exhausted);
        assert_eq!(trace.evaluated.len(), 3);
        assert_eq!(space.values(trace.ranking[0])["A"], 2);

        let space = self::space(&[("A", &[1, 2, 3, 4, 5, 6, 7, 8]), ("B", &[1, 2, 3, 4])]);
        let mut evaluator = Exact {
            space: &space,
            cost: |values: &ParameterValues| (values["A"] + values["B"]) as f64,
            evaluations: Vec::new(),
            limit: 5,
        };
        let trace = search(&space, &[], &settings(), &mut evaluator);
        assert_eq!(trace.stop, SearchStop::Expired);
        assert_eq!(evaluator.evaluations.len(), 5);
    }

    /// Searches like [`Exact`] with cost `A`, but cannot confirm `A` = 1.
    struct Unconfirmable<'s> {
        space: &'s SearchSpace,
    }

    impl Unconfirmable<'_> {
        fn cost(&self, index: usize) -> Result<Cost, Exclusion> {
            let values = self.space.values(index);
            let cost = values["A"] as f64;
            Ok(Cost::new(vec![(
                PointKey {
                    launches: vec![0],
                    values,
                },
                cost,
            )]))
        }
    }

    impl Evaluator for Unconfirmable<'_> {
        fn evaluate(&mut self, batch: &[usize]) -> Vec<Result<Cost, Exclusion>> {
            batch.iter().map(|index| self.cost(*index)).collect()
        }
        fn confirm(&mut self, finalists: &[usize]) -> Vec<Result<Cost, Exclusion>> {
            finalists
                .iter()
                .map(|index| match self.space.values(*index)["A"] {
                    1 => Err(Exclusion::Measurement {
                        point: "m1".into(),
                        detail: "unstable".into(),
                    }),
                    _ => self.cost(*index),
                })
                .collect()
        }
        fn expired(&self) -> bool {
            false
        }
    }

    #[test]
    fn a_finalist_that_cannot_be_confirmed_leaves_the_ranking() {
        // A 1 leads beyond noise of the rest, so only it and the defaults
        // (A 4) are confirmed; it cannot be, and the defaults are the choice.
        let space = space(&[("A", &[4, 1, 2, 3])]);
        let trace = search(
            &space,
            &[],
            &settings(),
            &mut Unconfirmable { space: &space },
        );
        let ranked = trace
            .ranking
            .iter()
            .map(|index| space.values(*index)["A"])
            .collect::<Vec<_>>();
        assert_eq!(ranked, vec![4]);
    }

    #[test]
    fn defaults_that_passed_the_search_stay_the_choice_when_unconfirmed() {
        // The defaults (A 1) are evaluated but cannot be re-measured.
        let space = space(&[("A", &[1, 2, 3, 4])]);
        let trace = search(
            &space,
            &[],
            &settings(),
            &mut Unconfirmable { space: &space },
        );
        assert_eq!(trace.ranking, vec![space.default_index()]);
        assert_eq!(space.values(trace.ranking[0])["A"], 1);
    }

    #[test]
    fn a_restart_reaches_an_optimum_past_a_ridge() {
        // The defaults (A 1) are a local minimum; the best (A 8) lies past a
        // ridge that only a restart from the farthest configuration reaches.
        let space = space(&[("A", &[1, 2, 3, 4, 5, 6, 7, 8])]);
        let mut evaluator = Exact {
            space: &space,
            cost: |values: &ParameterValues| match values["A"] {
                1 => 1.0,
                8 => 0.5,
                a => 2.0 + a as f64,
            },
            evaluations: Vec::new(),
            limit: usize::MAX,
        };
        let trace = search(&space, &[], &settings(), &mut evaluator);
        assert_eq!(space.values(trace.ranking[0])["A"], 8);
    }

    #[test]
    fn points_of_a_class_count_by_their_real_time() {
        let key = |launch: usize| PointKey {
            launches: vec![launch],
            values: ParameterValues::new(),
        };
        let keys = || (0..3).map(key).collect::<Vec<_>>();
        // m32 at a short and a long history (one class, equal frequency),
        // and a decode point of its own.
        let weights = [0.25, 0.25, 0.5];
        let classes = classes([Some("m32"), Some("m32"), None]);
        assert_eq!(classes, [0, 0, 1]);
        let reference = [0.13e-3, 90e-3, 0.06e-3];
        // 15% faster at the short history, 7% slower at the long one.
        let candidate = [0.13e-3 * 0.85, 90e-3 * 1.07, 0.06e-3];
        let cost = Cost::relative(keys(), &weights, &classes, &candidate, &reference);
        let defaults = Cost::relative(keys(), &weights, &classes, &reference, &reference);
        assert!((defaults.total() - 1.0).abs() < 1e-12);
        assert!(cost.total() > defaults.total(), "{cost:?}");
        // Each point in a class of its own: the short history's win counts
        // as much as the long one's loss, and the candidate wins.
        let own = super::classes([None, None, None]);
        let cost = Cost::relative(keys(), &weights, &own, &candidate, &reference);
        assert!(cost.total() < 1.0, "{cost:?}");
    }

    #[test]
    fn the_defaults_win_ties_within_the_margin() {
        let space = space(&[("A", &[2, 1])]);
        let mut evaluator = Exact {
            space: &space,
            cost: |values: &ParameterValues| if values["A"] == 1 { 0.99 } else { 1.0 },
            evaluations: Vec::new(),
            limit: usize::MAX,
        };
        let trace = search(&space, &[], &settings(), &mut evaluator);
        assert_eq!(trace.ranking[0], space.default_index());
    }

    #[test]
    fn a_start_hint_is_evaluated_with_the_defaults() {
        let space = space(&[("A", &[1, 2, 3, 4, 5, 6, 7, 8, 9])]);
        let hint = space
            .index_of(&[("A".to_string(), 9)].into_iter().collect())
            .unwrap();
        let mut evaluator = Exact {
            space: &space,
            cost: |values: &ParameterValues| 10.0 - values["A"] as f64,
            evaluations: Vec::new(),
            limit: usize::MAX,
        };
        let trace = search(&space, &[hint], &settings(), &mut evaluator);
        assert_eq!(evaluator.evaluations[..2], [space.default_index(), hint]);
        assert_eq!(trace.ranking[0], hint);
    }

    #[test]
    fn each_form_is_searched_from_a_start_of_its_own() {
        // The matrix form (M 1) needs K ≥ 2, so the defaults have no M
        // neighbor. Its nearest start costs more than the defaults, but
        // tuned along A it beats every vector configuration.
        let space = forms_space(
            &[
                ("A", &[1, 2, 3, 4, 5, 6, 7, 8]),
                ("K", &[1, 2, 3, 4]),
                ("M", &[0, 1]),
            ],
            &["M"],
            |values| values["M"] == 0 || values["K"] >= 2,
        );
        let start = space
            .index_of(
                &[("A", 1), ("K", 2), ("M", 1)]
                    .into_iter()
                    .map(|(name, value)| (name.to_string(), value))
                    .collect(),
            )
            .unwrap();
        assert_eq!(space.form_starts(), vec![start]);
        let mut evaluator = Exact {
            space: &space,
            cost: |values: &ParameterValues| match values["M"] {
                0 => 1.0 + 0.01 * (values["A"] + values["K"]) as f64,
                _ => 0.9 + 0.1 * (8 - values["A"]) as f64 + 0.01 * values["K"] as f64,
            },
            evaluations: Vec::new(),
            limit: usize::MAX,
        };
        let trace = search(&space, &[], &settings(), &mut evaluator);
        assert_eq!(evaluator.evaluations[..2], [space.default_index(), start]);
        let chosen = space.values(trace.ranking[0]);
        assert_eq!((chosen["M"], chosen["A"]), (1, 8), "{chosen:?}");
    }

    #[test]
    fn hinted_form_start_uses_its_first_measurement() {
        let space = forms_space(
            &[
                ("PARTS", &[16, 32, 64, 128]),
                ("MATRIX", &[0, 1]),
                ("KEYWISE", &[0, 1]),
            ],
            &["MATRIX", "KEYWISE"],
            |values| values["MATRIX"] == 0 || values["KEYWISE"] == 0,
        );
        let hint = space
            .index_of(
                &[
                    ("PARTS".to_owned(), 128),
                    ("MATRIX".to_owned(), 0),
                    ("KEYWISE".to_owned(), 1),
                ]
                .into_iter()
                .collect(),
            )
            .unwrap();
        let matrix = space.form_starts()[0];
        let mut evaluator = Exact {
            space: &space,
            cost: |values: &ParameterValues| {
                if values["KEYWISE"] == 1 && values["PARTS"] == 128 {
                    0.5
                } else {
                    1.0
                }
            },
            evaluations: Vec::new(),
            limit: usize::MAX,
        };
        let trace = search(&space, &[hint], &settings(), &mut evaluator);
        assert_eq!(
            evaluator.evaluations[..3],
            [space.default_index(), matrix, hint]
        );
        assert_eq!(trace.ranking[0], hint);
    }
}
