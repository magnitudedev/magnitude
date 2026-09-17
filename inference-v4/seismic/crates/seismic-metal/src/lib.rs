//! Metal backend: MSL printer and runtime.

pub mod msl;
#[cfg(target_os = "macos")]
pub mod plan_exec;
#[cfg(target_os = "macos")]
pub mod runtime;
