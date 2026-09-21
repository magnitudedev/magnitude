//! Rust inference engine. Artifact interpretation and model policy retain V3
//! semantics; numerical execution belongs to Seismic, not this loading layer.
pub mod chat;
pub mod error;
pub mod execution;
pub mod generation;
pub mod inputs;
pub mod models;
/// The only compiler owner above the runtime (package E1).
pub mod preparation;
pub mod service;
pub mod serving;
pub mod state;
pub mod telemetry;
pub mod weights;
pub use error::Error;
