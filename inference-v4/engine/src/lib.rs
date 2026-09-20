//! Rust inference engine. Artifact interpretation and model policy retain V3
//! semantics; numerical execution belongs to Seismic, not this loading layer.
pub mod chat;
pub mod execution;
pub mod generation;
pub mod inputs;
pub mod models;
pub mod service;
pub mod serving;
pub mod state;
pub mod weights;
