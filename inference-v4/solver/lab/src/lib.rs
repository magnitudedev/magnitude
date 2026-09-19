//! Reproducible domain-independent experiments for the exact solver.
//!
//! Generated costs are mathematical fixtures, not measured hardware costs.

pub mod experiment;
pub mod generators;
pub mod reference;
pub mod reporting;
pub mod runner;

pub type LabResult<T> = Result<T, Box<dyn std::error::Error>>;
