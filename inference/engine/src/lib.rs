//! The inference engine library: host resolution, read-only load preview,
//! the numerical worker and its protocol, and the typed outcomes of each.
//! Common crates own artifacts, model families, execution, generation,
//! scheduling, chat, and templates; this crate binds them.
pub mod assessment;
pub mod census;
pub mod chat;
pub mod composition;
pub mod error;
mod execution;
pub use execution::build_native_domain;
pub mod families;
pub mod generation;
pub mod host;
pub mod inputs;
pub mod invocation;
pub mod options;
pub mod planning;
pub mod preview;
pub mod telemetry;
pub mod worker;
