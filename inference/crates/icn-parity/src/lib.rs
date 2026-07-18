//! Primitive parity orchestration that is independent of the llama.cpp Rust fork.
//!
//! The crate treats parity cases as neutral data, executes reference and
//! candidate producers in bounded subprocesses, and compares their evidence
//! outside both engine processes.

pub mod assets;
pub mod compare;
pub mod decode;
pub mod digest;
pub mod model;
pub mod models;
pub mod process;
mod protocol;
pub mod provenance;
pub mod runner;
pub mod store;
pub mod tools;

pub use assets::{AssetRepository, ValidationReport};
pub use compare::compare_evidence;
pub use model::{CaseDefinition, ComparisonRecord, EvidenceRecord};
pub use runner::{RunOptions, RunSummary, run_profile};
