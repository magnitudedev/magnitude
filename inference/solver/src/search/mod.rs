//! Selectable search implementations over the same typed model and outcomes.
use std::time::Duration;
mod engine;
mod knowledge;
mod neighborhood;
mod repair;
mod structure;
use crate::result::Budgeted;
use crate::{FeasibleSolution, Model, Outcome, Result, Solution};
use std::sync::Arc;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Algorithm {
    Exact,
    Neighborhood(NeighborhoodOptions),
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NeighborhoodOptions {
    pub seed: u64,
    pub max_neighborhood_variables: usize,
    pub repair_work: u64,
    pub population_size: usize,
    pub restart_after: u64,
    /// Enables forced proposals, uphill acceptance and population exploration.
    /// Restarts use `restart_after` independently; use u64::MAX to disable them.
    /// False is the joint-greedy ablation; width one is coordinate greedy.
    pub exploration: bool,
}
impl Default for NeighborhoodOptions {
    fn default() -> Self {
        Self {
            seed: 0,
            max_neighborhood_variables: 8,
            repair_work: 4096,
            population_size: 4,
            restart_after: 32,
            exploration: true,
        }
    }
}

enum Implementation {
    Exact(engine::Search),
    Neighborhood(neighborhood::Search),
}

pub struct Search {
    implementation: Implementation,
}
impl Search {
    pub fn new(model: Arc<Model>, options: Options) -> Result<Self> {
        let implementation = match &options.algorithm {
            Algorithm::Exact => Implementation::Exact(engine::Search::new(model, options)?),
            Algorithm::Neighborhood(config) => Implementation::Neighborhood(
                neighborhood::Search::new(model, options.clone(), config.clone())?,
            ),
        };
        Ok(Self { implementation })
    }
    /// Installs a validated feasible solution of this search's exact model as
    /// the incumbent without restarting search. Returns whether it became the
    /// new incumbent; a weaker assignment is retained but not installed.
    pub fn install_incumbent(&mut self, incumbent: FeasibleSolution) -> Result<bool> {
        match &mut self.implementation {
            Implementation::Exact(search) => search.install_incumbent(incumbent),
            Implementation::Neighborhood(search) => search.install_incumbent(incumbent),
        }
    }
    /// Guides search with a preferred complete assignment of the model's
    /// original domains. It is a search preference, never a feasibility claim.
    pub fn prefer(&mut self, values: &[i64]) -> Result<()> {
        match &mut self.implementation {
            Implementation::Exact(search) => search.prefer(values),
            Implementation::Neighborhood(search) => search.prefer(values),
        }
    }
    /// One budgeted solve advance. The budget limits optimization only:
    /// feasibility is decided first regardless of work and time, then the
    /// remaining budget is spent improving the incumbent, which is returned
    /// with an optimality flag. Work and time never suspend the search before
    /// feasibility is decided.
    pub fn advance_budgeted(&mut self, budget: Budget) -> Result<Budgeted> {
        match &mut self.implementation {
            Implementation::Exact(search) => search.advance_budgeted(budget),
            Implementation::Neighborhood(search) => search.advance_budgeted(budget),
        }
    }
    pub fn advance(&mut self, limits: Limits) -> Result<Outcome> {
        match &mut self.implementation {
            Implementation::Exact(search) => Ok(match search.advance(limits)? {
                engine::Outcome::Optimal(feasible) => Outcome::Optimal(Solution { feasible }),
                engine::Outcome::Infeasible => Outcome::Infeasible,
                engine::Outcome::Incomplete(progress) => Outcome::Incomplete(progress),
            }),
            Implementation::Neighborhood(search) => search.advance(limits),
        }
    }
    pub fn stats(&self) -> &Stats {
        match &self.implementation {
            Implementation::Exact(s) => s.stats(),
            Implementation::Neighborhood(s) => s.stats(),
        }
    }
    pub fn model(&self) -> &Arc<Model> {
        match &self.implementation {
            Implementation::Exact(s) => s.model(),
            Implementation::Neighborhood(s) => s.model(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Policy {
    DepthFirst,
    BestFirst,
}
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Options {
    pub algorithm: Algorithm,
    pub policy: Policy,
    pub memoize: bool,
    pub decompose: bool,
    pub bounds: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            algorithm: Algorithm::Exact,
            policy: Policy::DepthFirst,
            memoize: true,
            decompose: true,
            bounds: true,
        }
    }
}
#[derive(Clone, Debug)]
pub struct Limits {
    /// Incremental number of propagation/analysis/branch work units per advance.
    pub work: u64,
    pub time: Option<Duration>,
    /// Retained search storage budget; excludes immutable input model and allocator overhead.
    pub memory_bytes: Option<usize>,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            work: 100_000,
            time: None,
            memory_bytes: None,
        }
    }
}
/// Budget for one budgeted solve advance. It limits optimization only: the
/// search decides feasibility first regardless of work and time, so there is
/// no budgeted outcome without an incumbent except a memory or coverage
/// suspension. The retained-memory cap also bounds the feasibility phase.
#[derive(Clone, Debug)]
pub struct Budget {
    /// Limits spent only after a feasible incumbent exists.
    pub optimization: Limits,
}
impl Default for Budget {
    fn default() -> Self {
        Self {
            optimization: Limits::default(),
        }
    }
}
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Stats {
    pub work: u64,
    pub nodes: u64,
    pub branches: u64,
    pub materializations: u64,
    pub decompositions: u64,
    pub memo_hits: u64,
    /// Completed residual proofs reused under structural equality and full domains.
    pub structural_hits: u64,
    /// New completed proofs admitted to the bounded shared cache.
    pub proof_stores: u64,
    /// Regions excluded by a learned conflict under compatible premises.
    pub conflict_hits: u64,
    /// New infeasibility proofs retained for stronger residual contexts.
    pub conflicts_learned: u64,
    /// Number of cached regions reclaimed under memory pressure. Exact ancestor
    /// summaries preserve their proofs; any later memo miss recomputes the
    /// region. Live pending coverage is never discarded.
    pub evictions: u64,
    /// Current retained unsolved OR children excluded by compatible incumbents.
    pub pruned: u64,
    pub propagations: u64,
    pub assessments: u64,
    pub max_frontier: usize,
    pub max_context_variables: usize,
    /// Rectangular context-domain upper estimate, not a count of legal outcomes.
    pub max_context_log10: f64,
    pub retained_bytes: usize,
    /// Maximum observed retained bytes at accounting boundaries.
    pub peak_retained_bytes: usize,
    pub first_feasible_seconds: Option<f64>,
    pub winner_found_seconds: Option<f64>,
    pub solve_seconds: f64,
    pub neighborhood_attempts: u64,
    /// Sum of requested primary widths for local neighborhoods, before
    /// deduplication and dependent-variable release. Excludes whole-model restarts.
    pub requested_variables: u64,
    pub max_requested_variables: usize,
    pub released_variables: u64,
    pub max_released_variables: usize,
    pub successful_repairs: u64,
    pub failed_repairs: u64,
    pub exhausted_repairs: u64,
    pub improving_moves: u64,
    pub accepted_moves: u64,
    pub restarts: u64,
    pub repair_work: u64,
}
