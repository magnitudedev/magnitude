//! Structured combinatorial optimization with selectable exact and neighborhood search.
pub mod composition;
pub mod model;
pub mod result;
pub mod scheduling;
pub mod search;
pub use model::Domain;
pub use model::{FactorId, Model, ModelBuilder, VarId};
pub use result::{Budgeted, FeasibleAssignment, FeasibleSolution, Outcome, Progress, Solution};
pub use search::{Algorithm, Budget, Limits, NeighborhoodOptions, Options, Policy, Search, Stats};

pub use model::{Error, Result};
pub mod bounds;
pub mod propagation;

mod conflicts;
mod decomposition;
