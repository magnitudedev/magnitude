//! Metal backend: MSL printer and runtime.

#[path = "msl_new.rs"]
pub mod msl;
pub mod target;

#[cfg(target_os = "macos")]
pub mod runtime;

#[path = "mapping_new.rs"]
pub mod mapping;
pub mod physical;
