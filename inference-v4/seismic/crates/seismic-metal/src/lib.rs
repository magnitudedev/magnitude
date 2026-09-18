//! Metal backend: MSL printer and runtime.

pub mod msl;
pub mod execution;
pub mod memory;
pub mod storage;
pub mod reduction;
pub mod family;
#[cfg(target_os = "macos")]
pub mod plan_exec;
#[cfg(target_os = "macos")]
pub mod runtime;

pub mod choices;

pub mod collective;

pub mod support;

pub mod model;
pub mod tuning;
pub mod terminal;
