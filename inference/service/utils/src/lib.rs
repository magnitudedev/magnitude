//! Shared infrastructure utilities for ICN crates.

pub mod blocking;
pub mod file_cache;
pub mod owner;
pub mod process;
pub mod sparse_file;

#[cfg(windows)]
pub mod windows_process;
