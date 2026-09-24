//! Budgeted local search over a native implementation's declared parameter
//! space, as a pure function of what an [`Evaluator`] reports.
//!
//! The search walks the grid of admissible configurations: each parameter's
//! values in numeric order, neighbors one step apart in one parameter. From
//! the best starting configuration it moves to the best neighbor while that
//! improves the cost by more than `improvement`; at a local minimum it
//! restarts from the unvisited configuration farthest from everything
//! visited. It stops when the budget is spent, every configuration was
//! visited, `restarts` consecutive restarts failed to improve the best cost,
//! or the evaluator reports its deadline passed. The `confirmed` cheapest
//! configurations and the defaults are then re-measured, alternating, and
//! ranked by those costs; the defaults rank first unless the leader beats
//! them by `default_margin`. Validation walks that ranking.
//!
//! Given the evaluator's costs the procedure is deterministic, so a replay of
//! recorded surveys (tuning spec §E2) can run this same code with a recorded
//! evaluator.

use super::tune::Exclusion;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// A configuration's parameter values by name.
pub type ParameterValues = BTreeMap<String, u64>;

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

/// A declared parameter space and its admissible configurations.
#[derive(Clone, Debug)]
pub struct SearchSpace {
    /// Parameter names in declaration order, each with its values in numeric
    /// order.
    parameters: Vec<(String, Vec<u64>)>,
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
                write!(f, "configuration {values:?} does not match the declared parameters")
            }
            Self::DefaultInadmissible => f.write_str("the all-defaults configuration is inadmissible"),
        }
    }
}

impl std::error::Error for SearchSpaceError {}

impl SearchSpace {
    /// `declared` lists each parameter's values in declaration order (the
    /// first is its default); `admissible` lists the configurations the
    /// declaration's `where` admits.
    pub fn new(
        declared: &[(String, Vec<u64>)],
        admissible: &[ParameterValues],
    ) -> Result<Self, SearchSpaceError> {
        let parameters = declared
            .iter()
            .map(|(name, values)| {
                let mut sorted = values.clone();
                sorted.sort_unstable();
                sorted.dedup();
                (name.clone(), sorted)
            })
            .collect::<Vec<_>>();
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
            .map(|(name, values)| (name.clone(), values[0]))
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
}

/// Measures configurations of a [`SearchSpace`] by index.
pub trait Evaluator {
    /// Form and measure a batch; one weighted cost per configuration, in
    /// order. A configuration that cannot be formed or run is excluded.
    fn evaluate(&mut self, batch: &[usize]) -> Vec<Result<f64, Exclusion>>;
    /// Re-measure `finalists` (every one evaluated before), alternating
    /// between them sample by sample; their new costs, in order.
    fn confirm(&mut self, finalists: &[usize]) -> Vec<Result<f64, Exclusion>>;
    /// Whether the safety stop has passed. The search then ends with what it
    /// has reached.
    fn expired(&self) -> bool;
}

/// Why the search stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchStop {
    /// The budget was spent.
    Budget,
    /// Every admissible configuration was evaluated.
    Exhausted,
    /// The allowed consecutive restarts found nothing better.
    Converged,
    /// The safety stop passed; the result is the best found so far.
    Expired,
}

/// Everything one search did.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchTrace {
    /// Every configuration evaluated, in order, with its search cost.
    pub evaluated: Vec<(usize, Result<f64, Exclusion>)>,
    /// The finalists re-measured, with their confirmed costs.
    pub confirmed: Vec<(usize, Result<f64, Exclusion>)>,
    /// The order validation tries configurations in. It holds the defaults,
    /// which always validate, so validation never runs past it.
    pub ranking: Vec<usize>,
    pub stop: SearchStop,
}

struct State<'s, E> {
    space: &'s SearchSpace,
    evaluator: &'s mut E,
    budget: usize,
    costs: HashMap<usize, f64>,
    evaluated: Vec<(usize, Result<f64, Exclusion>)>,
    /// Least distance from each configuration to any evaluated one.
    nearest: Vec<u32>,
    stop: Option<SearchStop>,
}

impl<E: Evaluator> State<'_, E> {
    fn cost(&self, index: usize) -> f64 {
        self.costs[&index]
    }

    fn visited(&self, index: usize) -> bool {
        self.costs.contains_key(&index)
    }

    /// Evaluate the unvisited configurations of `batch` that the budget
    /// admits, in order. Sets the stop reason when nothing more may be
    /// evaluated.
    fn evaluate(&mut self, batch: &[usize]) {
        if self.stop.is_some() {
            return;
        }
        if self.evaluator.expired() {
            self.stop = Some(SearchStop::Expired);
            return;
        }
        let remaining = self.budget - self.evaluated.len();
        let mut fresh = Vec::new();
        for &index in batch {
            if !self.visited(index) && !fresh.contains(&index) {
                fresh.push(index);
            }
        }
        fresh.truncate(remaining);
        if fresh.is_empty() {
            return;
        }
        let results = self.evaluator.evaluate(&fresh);
        assert_eq!(results.len(), fresh.len(), "an evaluator answers every configuration");
        for (index, result) in fresh.into_iter().zip(results) {
            let cost = result.as_ref().map_or(f64::INFINITY, |cost| *cost);
            self.costs.insert(index, cost);
            for (other, nearest) in self.nearest.iter_mut().enumerate() {
                *nearest = (*nearest).min(self.space.distance(index, other));
            }
            self.evaluated.push((index, result));
        }
        if self.evaluated.len() == self.space.len() {
            self.stop = Some(SearchStop::Exhausted);
        } else if self.evaluated.len() == self.budget {
            self.stop = Some(SearchStop::Budget);
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
            let moves = [step.checked_sub(1), (step as usize + 1 < values.len()).then_some(step + 1)];
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

    fn improves(&self, cost: f64, over: f64, margin: f64) -> bool {
        cost < over * (1.0 - margin)
    }

    /// Descend from `current` until no neighbor improves on it by more
    /// than the settings' improvement.
    fn descend(&mut self, mut current: usize, improvement: f64) {
        loop {
            let neighbors = self.neighbors(current);
            self.evaluate(&neighbors);
            let best = neighbors
                .iter()
                .copied()
                .filter(|neighbor| self.visited(*neighbor))
                .min_by(|left, right| self.cost(*left).total_cmp(&self.cost(*right)));
            match best {
                Some(best) if self.improves(self.cost(best), self.cost(current), improvement) => {
                    current = best;
                }
                _ => return,
            }
            if self.stop.is_some() {
                return;
            }
        }
    }

    fn best_cost(&self) -> f64 {
        self.costs.values().copied().fold(f64::INFINITY, f64::min)
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

/// Search `space` from the defaults and the `start` configurations (for
/// example the winner of the same declaration at other element bindings),
/// evaluating at most `budget` configurations (at least the defaults).
pub fn search(
    space: &SearchSpace,
    start: &[usize],
    budget: usize,
    settings: &SearchSettings,
    evaluator: &mut impl Evaluator,
) -> SearchTrace {
    let mut state = State {
        space,
        evaluator,
        budget: budget.clamp(1, space.len()),
        costs: HashMap::new(),
        evaluated: Vec::new(),
        nearest: vec![u32::MAX; space.len()],
        stop: None,
    };
    let default = space.default;
    let starts = std::iter::once(default)
        .chain(start.iter().copied().filter(|index| *index < space.len()))
        .collect::<Vec<_>>();
    // The defaults are the validation reference: evaluated even past the
    // safety stop.
    if state.evaluator.expired() {
        state.stop = Some(SearchStop::Expired);
        let results = state.evaluator.evaluate(&[default]);
        let result = results.into_iter().next().expect("an evaluator answers every configuration");
        state
            .costs
            .insert(default, result.as_ref().map_or(f64::INFINITY, |cost| *cost));
        state.evaluated.push((default, result));
    } else {
        state.evaluate(&starts);
    }
    let current = starts
        .iter()
        .copied()
        .filter(|index| state.visited(*index))
        .min_by(|left, right| state.cost(*left).total_cmp(&state.cost(*right)))
        .expect("the defaults were evaluated");
    if state.stop.is_none() {
        state.descend(current, settings.improvement);
    }
    let mut failed_restarts = 0;
    while state.stop.is_none() && failed_restarts < settings.restarts {
        let before = state.best_cost();
        let Some(restart) = state.farthest() else {
            break;
        };
        state.evaluate(&[restart]);
        if state.visited(restart) && state.stop.is_none() {
            state.descend(restart, settings.improvement);
        }
        if state.improves(state.best_cost(), before, settings.improvement) {
            failed_restarts = 0;
        } else {
            failed_restarts += 1;
        }
    }
    let stop = state.stop.unwrap_or(SearchStop::Converged);
    let State {
        evaluated, costs, evaluator, ..
    } = state;

    // Confirm the cheapest measured configurations against the defaults.
    let mut cheapest = evaluated
        .iter()
        .filter(|(index, result)| *index != default && result.is_ok())
        .map(|(index, _)| *index)
        .collect::<Vec<_>>();
    cheapest.sort_by(|left, right| costs[left].total_cmp(&costs[right]));
    cheapest.truncate(settings.confirmed);
    let finalists = std::iter::once(default).chain(cheapest).collect::<Vec<_>>();
    let confirmed = if finalists.len() > 1 {
        let results = evaluator.confirm(&finalists);
        assert_eq!(results.len(), finalists.len(), "an evaluator confirms every finalist");
        finalists.iter().copied().zip(results).collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let confirmed_cost = |index: usize| {
        confirmed
            .iter()
            .find(|(candidate, _)| *candidate == index)
            .and_then(|(_, result)| result.as_ref().ok().copied())
    };
    let ranking = match confirmed_cost(default) {
        Some(reference) => {
            let mut ranked = finalists
                .iter()
                .copied()
                .filter(|index| confirmed_cost(*index).is_some())
                .collect::<Vec<_>>();
            ranked.sort_by(|left, right| {
                confirmed_cost(*left)
                    .expect("ranked finalists were confirmed")
                    .total_cmp(&confirmed_cost(*right).expect("ranked finalists were confirmed"))
            });
            let leader = ranked[0];
            let leader_cost = confirmed_cost(leader).expect("ranked finalists were confirmed");
            if leader != default && leader_cost >= reference * (1.0 - settings.default_margin) {
                ranked.retain(|index| *index != default);
                ranked.insert(0, default);
            }
            ranked
        }
        // Nothing but the defaults was measured, or re-measuring the
        // defaults failed: the defaults are the choice.
        None => vec![default],
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
        let declared = declared
            .iter()
            .map(|(name, values)| (name.to_string(), values.to_vec()))
            .collect::<Vec<_>>();
        let mut admissible = vec![ParameterValues::new()];
        for (name, values) in &declared {
            admissible = admissible
                .into_iter()
                .flat_map(|base| {
                    values.iter().map(move |value| {
                        let mut next = base.clone();
                        next.insert(name.clone(), *value);
                        next
                    })
                })
                .collect();
        }
        SearchSpace::new(&declared, &admissible).unwrap()
    }

    /// A deterministic evaluator over a cost function of parameter values.
    struct Exact<'s, F> {
        space: &'s SearchSpace,
        cost: F,
        evaluations: Vec<usize>,
    }

    impl<F: Fn(&ParameterValues) -> f64> Evaluator for Exact<'_, F> {
        fn evaluate(&mut self, batch: &[usize]) -> Vec<Result<f64, Exclusion>> {
            self.evaluations.extend_from_slice(batch);
            batch
                .iter()
                .map(|index| Ok((self.cost)(&self.space.values(*index))))
                .collect()
        }
        fn confirm(&mut self, finalists: &[usize]) -> Vec<Result<f64, Exclusion>> {
            finalists
                .iter()
                .map(|index| Ok((self.cost)(&self.space.values(*index))))
                .collect()
        }
        fn expired(&self) -> bool {
            false
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
        };
        let trace = search(&space, &[], 25, &settings(), &mut evaluator);
        let chosen = space.values(trace.ranking[0]);
        assert_eq!(chosen["A"] * chosen["B"], 16, "{chosen:?}");
        assert_eq!(chosen["A"], 1, "{chosen:?}");
    }

    #[test]
    fn a_small_space_is_covered_exactly_and_never_past_its_budget() {
        let space = space(&[("A", &[4, 2, 8])]);
        let mut evaluator = Exact {
            space: &space,
            cost: |values: &ParameterValues| values["A"] as f64,
            evaluations: Vec::new(),
        };
        let trace = search(&space, &[], 100, &settings(), &mut evaluator);
        assert_eq!(trace.stop, SearchStop::Exhausted);
        assert_eq!(trace.evaluated.len(), 3);
        assert_eq!(space.values(trace.ranking[0])["A"], 2);

        let space = self::space(&[("A", &[1, 2, 3, 4, 5, 6, 7, 8]), ("B", &[1, 2, 3, 4])]);
        let mut evaluator = Exact {
            space: &space,
            cost: |values: &ParameterValues| (values["A"] + values["B"]) as f64,
            evaluations: Vec::new(),
        };
        let trace = search(&space, &[], 5, &settings(), &mut evaluator);
        assert_eq!(trace.stop, SearchStop::Budget);
        assert_eq!(evaluator.evaluations.len(), 5);
    }

    #[test]
    fn the_defaults_win_ties_within_the_margin() {
        let space = space(&[("A", &[2, 1])]);
        let mut evaluator = Exact {
            space: &space,
            cost: |values: &ParameterValues| if values["A"] == 1 { 0.99 } else { 1.0 },
            evaluations: Vec::new(),
        };
        let trace = search(&space, &[], 10, &settings(), &mut evaluator);
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
        };
        let trace = search(&space, &[hint], 2, &settings(), &mut evaluator);
        assert_eq!(evaluator.evaluations, vec![space.default_index(), hint]);
        assert_eq!(trace.ranking[0], hint);
    }
}
