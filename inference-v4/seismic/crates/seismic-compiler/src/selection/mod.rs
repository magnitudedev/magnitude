//! Joint selection of authored implementations, contiguous fusion groups and numerical
//! sites (spec sections 9-12). The family comes from `seismic-lang`; the backend adds
//! domains, legality, fusion intervals and local cost factors; `magnitude-solver`
//! searches; the witness is validated, instantiated and handed to the backend to realize.
//!
//! Nothing here or in a backend hook ranks alternatives by profitability outside the
//! solver objective, and nothing after selection changes the witness.
mod analyze;
mod backend;
mod export;
mod greedy;
/// Target-neutral halves of the backend hooks (domains, intervals, limits, factors, seed).
pub mod mapping;
/// Derived quantities over numerical site values.
pub mod quantity;
mod search;
/// The shared structural walk of candidate bodies, parameterized by a backend `Accounting`.
pub mod structure;

pub use analyze::{analyze, analyze_with, SearchAnalysis};
pub use backend::{Backend, Constraint, Factor, Interval, IntervalRef};
pub use search::{replay, select, Budget, Strategy};

use seismic_lang::family::{Family, Witness};
use std::sync::Arc;
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProofStatus {
    /// Complete checked execution; search ended by budget or with open obligations.
    Feasible,
    /// Additionally optimal over the stated family under the stated estimate model.
    ModelOptimal,
}

/// The executable boundary: a checked feasible witness with its realization.
#[derive(Debug)]
pub struct Selected<E> {
    pub execution: E,
    pub family: Arc<Family>,
    pub witness: Witness,
    /// Estimated cost of `witness` in the backend's estimate units.
    pub estimate: u64,
    /// The validated constructive seed and its estimate, retained for comparison.
    pub seed: Witness,
    pub seed_estimate: u64,
    pub status: ProofStatus,
    /// Identity of the estimate model, e.g. `metal-estimate-unqualified-v0`.
    pub estimate_model: String,
    /// Lower bound proved by the solver over the exported family, in the same units.
    pub lower_bound: u64,
    pub unresolved: Vec<String>,
    /// Wall time of every selection phase of this entry.
    pub timings: Timings,
    pub search: SearchStats,
}

/// Wall time per selection phase. `search` is the solver (or the greedy sweeps); it
/// excludes `seed`, whose validation is charged to `seed`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Timings {
    /// `family::construct`.
    pub family: Duration,
    /// `bind_structure` + `constraints` + `intervals` + `factors`.
    pub backend_hooks: Duration,
    /// Tabulation of constraints and factors into the solver model.
    pub export: Duration,
    /// `Backend::seed` and its validation against the exported model.
    pub seed: Duration,
    pub search: Duration,
    pub instantiate: Duration,
    pub realize: Duration,
}

impl Timings {
    /// Everything before instantiation: what choosing the witness cost.
    pub fn solve(&self) -> Duration {
        self.family + self.backend_hooks + self.export + self.seed + self.search
    }
}

/// One solver phase as `magnitude-solver` reports it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Phase {
    pub time: Duration,
    pub work: u64,
    pub nodes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchStats {
    pub strategy: Strategy,
    /// Variables and factors of the exported model.
    pub variables: usize,
    pub factors: usize,
    /// `Strategy::Exact`: the exact slice, and the neighborhood slice when it ran.
    pub exact: Phase,
    pub neighborhood: Option<Phase>,
    /// `Strategy::Greedy`: completed sweeps and validated trial assignments.
    pub greedy_sweeps: u32,
    pub greedy_trials: u64,
}

/// Distinct outcomes (spec section 13.3). None of these recommends or performs a fallback.
#[derive(Debug)]
pub enum SelectionError {
    /// Type, effect, ownership or declaration error in the linked program.
    InvalidSource(String),
    /// No applicable implementation covers a reached call on this target.
    MissingCoverage(String),
    /// The backend has no mapping for an otherwise meaningful structure.
    UnsupportedMapping(String),
    /// Required representation/participation/lifetime interfaces disagree.
    IncompatibleComposition(String),
    /// The exported joint family is proved to have no solution.
    Infeasible,
    /// Family/interface construction stopped with unexplored obligations.
    ConstructionIncomplete(Vec<String>),
    /// Budget ended without any checked usable configuration.
    SelectionIncomplete(String),
    /// A required quantity or estimate has no supported derivation.
    AnalysisUnavailable(String),
    /// Reconstruction disagreed with the witness: a compiler defect.
    Reconstruction(String),
}

impl std::fmt::Display for SelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SelectionError::InvalidSource(m) => write!(f, "invalid source: {m}"),
            SelectionError::MissingCoverage(m) => write!(f, "missing target coverage: {m}"),
            SelectionError::UnsupportedMapping(m) => {
                write!(f, "unsupported structural mapping: {m}")
            }
            SelectionError::IncompatibleComposition(m) => {
                write!(f, "incompatible composition: {m}")
            }
            SelectionError::Infeasible => {
                write!(f, "no feasible configuration in the stated family")
            }
            SelectionError::ConstructionIncomplete(o) => {
                write!(f, "construction incomplete: {}", o.join("; "))
            }
            SelectionError::SelectionIncomplete(m) => write!(f, "selection incomplete: {m}"),
            SelectionError::AnalysisUnavailable(m) => write!(f, "analysis unavailable: {m}"),
            SelectionError::Reconstruction(m) => write!(f, "reconstruction defect: {m}"),
        }
    }
}

impl std::error::Error for SelectionError {}
