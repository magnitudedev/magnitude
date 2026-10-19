//! What a native search ranks configurations by: the objective ([`Cost`]),
//! what tells two configurations apart at a point ([`LaunchKey`]), and the
//! constants a search is run with ([`SearchSettings`]).
//!
//! The objective is expected elapsed service: each tuning point contributes
//! its invocation demand (launches per turn) times the configuration's time
//! there, and the sum over points is taken relative to the defaults'
//! service under the same demand ([`Cost::relative`]). A point's vote
//! therefore scales with how long its workload runs, a point without
//! demand contributes nothing, and classes of points do not renormalize
//! it. A point votes only between configurations that run
//! different work there ([`LaunchKey`]); configurations with the same key at
//! a point share one measurement of it, and two configurations are
//! compared over the points that tell them apart.
//!
//! The order in which a search asks its domain for configurations is
//! [`super::tune::GroupSearch`], over the plan of that domain
//! ([`super::plan`]): one order for every implementation, and the one a
//! replay of a recorded survey runs ([`super::replay`]).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A configuration's parameter values by name.
pub type ParameterValues = BTreeMap<String, u64>;

/// What a configuration's measurement at one point depends on: the launches
/// active there (by declaration ordinal) and the values of the parameters
/// those launches read. Configurations with the same key at a point run the
/// same work there: the point cannot tell them apart, so it is measured once
/// for all of them and does not vote between them.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LaunchKey {
    pub launches: Vec<usize>,
    pub values: ParameterValues,
}

/// One configuration's expected service: each point's physical key and its
/// demand-weighted contribution relative to the complete canonical workload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cost {
    points: Vec<(LaunchKey, f64)>,
}

impl Cost {
    /// `points` in the tuning points' order.
    pub fn new(points: Vec<(LaunchKey, f64)>) -> Self {
        Self { points }
    }

    /// Expected elapsed service relative to the same invocation demand at
    /// the reference. Region identities do not independently renormalize cost.
    pub fn relative(
        keys: Vec<LaunchKey>,
        demand: &[f64],
        medians: &[f64],
        reference: &[f64],
    ) -> Self {
        let baseline = demand
            .iter()
            .zip(reference)
            .filter(|(demand, _)| **demand > 0.0)
            .map(|(demand, seconds)| demand * seconds)
            .sum::<f64>();
        Self::new(
            keys.into_iter()
                .enumerate()
                .map(|(point, key)| {
                    (
                        key,
                        if baseline > 0.0 && demand[point] > 0.0 {
                            demand[point] * medians[point] / baseline
                        } else {
                            0.0
                        },
                    )
                })
                .collect(),
        )
    }

    pub fn total(&self) -> f64 {
        self.points.iter().map(|(_, cost)| cost).sum()
    }

    /// Whether this cost is below `other` over the points that tell the
    /// two apart. Points with equal keys share one measurement and cancel,
    /// so two configurations that no point tells apart do not improve on
    /// one another.
    pub fn improves_on(&self, other: &Cost) -> bool {
        let (mine, theirs) = self
            .points
            .iter()
            .zip(&other.points)
            .filter(|((key, _), (other, _))| key != other)
            .fold((0.0, 0.0), |(mine, theirs), ((_, cost), (_, other))| {
                (mine + cost, theirs + other)
            });
        mine < theirs
    }
}

/// Search constants. Part of a tuning result's identity: changing one
/// changes what the search may choose.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchSettings {
    /// α: how often a resolved comparison is wrong, the one place that is
    /// said. Of the comparisons of a configuration that is not faster than
    /// the served choice, at most this fraction resolve as a lead, and the
    /// configuration is promoted. Of the comparisons of one that is not
    /// slower, at most this fraction resolve as a trail, and it is
    /// dropped. Both hold over every round a comparison
    /// is ever given ([`super::sampling`]). A smaller rate takes more
    /// rounds to resolve anything: the fewest in which a lead can resolve
    /// are the first `n` with `(2^(n+1) − 1) / (n + 1) ≥ 1 / α`, seven at
    /// 0.05 and nine at 0.01.
    pub error_rate: f64,
}

/// Whether a search reached its own end, and what stopped one that did
/// not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchStop {
    /// The search asked for nothing more: every configuration it asked its
    /// domain for was measured.
    Exhausted,
    /// The search did not reach its own end; the result is the best found
    /// so far.
    Expired,
    /// The search did not reach its own end because no piece it still
    /// offered fit the memory its load's tuning may hold, with nothing in
    /// flight and everything held for nothing returned; the result is the
    /// best found so far. Its owner's memory admission says so: a search
    /// knows its own end and nothing of the room it was given.
    NoRoom,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn costs_are_compared_over_the_points_that_tell_configurations_apart() {
        let key = |name: &str, value: u64, launch: usize| LaunchKey {
            launches: vec![launch],
            values: [(name.to_string(), value)].into_iter().collect(),
        };
        let cost = |a: u64, small: f64, shared: f64| {
            Cost::new(vec![(key("A", a, 0), small), (key("B", 1, 1), shared)])
        };
        // Any gain at the point that tells them apart improves, however
        // small a share of the whole cost that point holds; equal or
        // greater cost there does not.
        assert!(cost(2, 0.0995, 0.9).improves_on(&cost(1, 0.1, 0.9)));
        assert!(!cost(2, 0.1, 0.9).improves_on(&cost(1, 0.1, 0.9)));
        assert!(!cost(2, 0.1005, 0.9).improves_on(&cost(1, 0.1, 0.9)));
        // The point with equal keys cancels: its recorded cost does not
        // decide the comparison in either direction.
        assert!(cost(2, 0.0995, 0.95).improves_on(&cost(1, 0.1, 0.9)));
        assert!(!cost(2, 0.1005, 0.5).improves_on(&cost(1, 0.1, 0.9)));
        // Equal keys everywhere: nothing tells them apart, nothing improves.
        assert!(!cost(1, 0.05, 0.9).improves_on(&cost(1, 0.1, 0.9)));
    }

    #[test]
    fn zero_demand_witnesses_have_no_service_vote() {
        let keys = || {
            vec![
                LaunchKey {
                    launches: vec![0],
                    values: ParameterValues::new()
                };
                2
            ]
        };
        let empty = Cost::relative(keys(), &[0.0, 0.0], &[f64::NAN, 0.0], &[0.0, f64::INFINITY]);
        assert_eq!(empty.total(), 0.0);
        let demanded = Cost::relative(keys(), &[1.0, 0.0], &[2.0, f64::NAN], &[1.0, f64::INFINITY]);
        assert_eq!(demanded.total(), 2.0);
    }

    #[test]
    fn invocation_demand_ranks_global_elapsed_service() {
        let key = |launch: usize| LaunchKey {
            launches: vec![launch],
            values: ParameterValues::new(),
        };
        let keys = || (0..3).map(key).collect::<Vec<_>>();
        // m32 at a short and a long history (one class, equal frequency),
        // and a decode point of its own.
        let weights = [0.25, 0.25, 0.5];
        let reference = [0.13e-3, 90e-3, 0.06e-3];
        // 15% faster at the short history, 7% slower at the long one.
        let candidate = [0.13e-3 * 0.85, 90e-3 * 1.07, 0.06e-3];
        let cost = Cost::relative(keys(), &weights, &candidate, &reference);
        let defaults = Cost::relative(keys(), &weights, &reference, &reference);
        assert!((defaults.total() - 1.0).abs() < 1e-12);
        assert!(cost.total() > defaults.total(), "{cost:?}");
        // Relabeling regions cannot reverse the expected elapsed-service objective.
        assert!(cost.total() > 1.0);
    }
}
