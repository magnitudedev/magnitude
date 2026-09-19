//! Metal backend: MSL printer and runtime.

pub mod msl;
pub mod execution;
pub mod memory;
pub mod storage;
pub mod reduction;

#[cfg(target_os = "macos")]
pub mod runtime;

pub mod collective;

pub mod support;

pub mod terminal;
pub mod mapping;
