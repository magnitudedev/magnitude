//! Reusable controlled benchmark runner for streaming Chat Completions endpoints.

mod client;
mod model;
mod report;
mod runner;
mod stats;
mod suite;

pub use model::{
    BenchmarkComparison, BenchmarkError, BenchmarkRun, BenchmarkRunConfig, ExperimentResult,
    FixtureCatalog, MemoryEvidence, Profile, ProgressEvent, RunOutcome, TargetConfig, TargetKind,
    TargetResult,
};
pub use report::{comparison_markdown, run_markdown};
pub use runner::{BenchmarkRunner, ProgressCallback, compare_evidence};
pub use suite::{load_fixtures, load_profile, validate_assets};
