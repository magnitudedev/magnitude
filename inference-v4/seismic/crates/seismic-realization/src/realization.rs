//! Closed strategy/route/residence/kernel/physical types and the sole core
//! formation and sealing implementation.
//!
//! Public surface: sealed read-only artifacts, the mapping catalog traits,
//! and exactly these transitions:
//!
//! ```text
//! form_plan_space(&LogicalProgram, &EffectiveTargetProfile, &impl MappingCatalog<D>)
//!     -> Result<PlanSpace<D>, CompilerDefect>
//! PlanSpace::solver_model(&self, &NumericalContext) -> SolverModelView
//! PlanSpace::resolve(self, CompleteAssignment) -> PhysicalPlan<D>
//! ```
//!
//! All mutable construction state is private to `formation`. There is no
//! public plan-space, strategy, kernel-block, or physical-plan builder.

pub mod consequences;
pub mod dispatch;
pub mod failure;
pub mod ids;
pub mod invocation;
pub mod kernel;
pub mod numerics;
pub mod occurrence;
pub mod physical;
pub mod plan_space;
pub mod residence;
pub mod routes;
pub mod strategy;
pub mod target;

mod formation;

pub use plan_space::form_plan_space;
