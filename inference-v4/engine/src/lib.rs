//! Inference V4 composition facade. Common crates own artifacts, model state,
//! execution, generation, service, chat, and templates; this crate only binds
//! them into a host-facing engine.
pub mod chat;
pub mod composition;
mod execution;
pub mod generation;
pub mod inputs;
pub mod options;
pub mod service;
pub mod serving;
pub mod telemetry;
