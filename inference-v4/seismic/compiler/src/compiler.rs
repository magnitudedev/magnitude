//! The Seismic compiler core (spec §2.3).
//!
//! Owns: semantic refinement, implementation builders and the factory boundary,
//! the candidate domain, total evaluation, transient solver adapter,
//! numerical analysis, freezing, exact coverage and portfolio
//! construction, the executable representation, the backend contract, and
//! the complete error taxonomy.
//!
//! The only artifact progression is
//! `CheckedModule -> LogicalEntry -> CandidateDomain<B>
//!  -> CandidateEvaluator -> SelectionPolicy<B> -> PreparedKernel<B, H>`.
//! Evaluator-internal estimation and solver state are not artifact boundaries.
//!
//! Planning authority and raw solver witnesses are intentionally absent from
//! the public API:
//!
//! ```compile_fail
//! use seismic_compiler::expression::PlanningExpr;
//! ```
//!
//! ```compile_fail
//! use seismic_compiler::solve::RawAssignment;
//! ```

pub mod candidate_domain;
pub mod errors;
pub mod evaluation;
pub mod executable;
pub mod implementation;
pub mod numerics;
pub mod planning;
pub mod preparation_budget;
pub mod prepare;
pub mod prepared;
pub mod refinement;
pub mod solve;
pub mod target;

mod evaluation_session;
mod expression;
mod frozen;
mod portable;
mod realization;

pub use planning::{
    OptimizationCompletion, PlanningBudgetReport, PlanningBudgetResource, PlanningCoverage,
    PlanningError, PlanningInfeasibleReport, PlanningLimit, SelectionPolicy, TargetCoverage,
};
pub use preparation_budget::{PlanningBudget, PreparationBudget};
pub use prepare::prepare_analytically;
pub use prepared::{CandidateIndex, SelectionFunction};
