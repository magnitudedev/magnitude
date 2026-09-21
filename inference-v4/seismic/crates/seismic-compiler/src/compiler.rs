//! The Seismic compiler core (spec §2.3).
//!
//! Owns: typed kernel IR, implementation builders and the factory boundary,
//! storage topology, parametric schedules, the plan space and solver
//! adapter, numerical analysis, freezing, exact coverage and portfolio
//! construction, the executable representation, the backend contract, and
//! the complete error taxonomy.
//!
//! The only artifact progression is
//! `CheckedModule -> LogicalEntry -> PlanSpace<B> -> FrozenPlan<B>
//!  -> ExecutableVariant<B> -> PreparedKernel<B>`.
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

pub mod errors;
pub mod executable;
pub mod frozen;
pub mod implementation;
pub mod kernel;
pub mod numerics;
pub mod plan_space;
pub mod portfolio;
pub mod preparation_budget;
pub mod prepared;
pub mod repr;
pub mod schedule;
pub mod solve;
pub mod storage;
pub mod target;

mod expression;
mod identity;
mod portable;

pub use plan_space::plan_space;
pub use portfolio::prepare_kernel;
pub use preparation_budget::PreparationBudget;
