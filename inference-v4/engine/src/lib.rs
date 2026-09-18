//! Rust inference engine. Artifact interpretation and model policy retain V3
//! semantics; numerical execution belongs to Seismic, not this loading layer.
pub mod models;
pub mod weights;
pub mod state;
pub mod execution;
pub mod chat;
pub mod inputs;
pub mod generation;
pub mod service;
pub mod serving;
