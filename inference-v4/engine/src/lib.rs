//! Rust inference engine. Artifact interpretation and model policy retain V3
//! semantics; numerical execution belongs to Seismic, not this loading layer.
pub mod chat;
pub mod error;
pub mod generation;
pub mod inputs;
pub mod kernels;
pub mod models;
pub mod service;
pub mod serving;
pub mod state;
pub mod telemetry;
pub mod weights;
pub use error::Error;
