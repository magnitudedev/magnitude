//! Metal backend: sealed `MetalOp` dialect, strategies over the common family
//! builder, mechanical MSL emission, and the macOS runtime.
//!
//! The backend owns exactly: the `MetalDialect` legalization/consequence
//! contract, the universal and capability strategies that consume logical
//! alternatives through `PlanFamilyBuilder`/`AlternativeBuilder`, the MSL
//! printer that exhaustively accepts `MetalOp`, and runtime glue over the
//! retained structured `ResolvedStep` tree. Boundaries name caller storage;
//! there is no assembly alias union. Threadgroup/private storage and argument
//! tables come only from the resolved plan.

#[path = "msl_new.rs"]
pub mod msl;
pub mod physical;
pub mod target;

#[path = "mapping_new.rs"]
pub mod mapping;

#[cfg(target_os = "macos")]
pub mod runtime;
