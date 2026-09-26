//! Host-facing generation intent and publication types.
//!
//! Live generation, executor operations, method state, and reconciliation
//! machinery remain owned by the subordinate generation/service crates. The
//! root facade exports only values a host needs to prepare a request or consume
//! its result.

pub use magnitude_generation::{
    grammar, Constraint, DetailedUsage, FinishReason, GenerationSeed, MethodChoice, Options,
    OutputToken, Sampling, Shaping, TokenId, Usage,
};
