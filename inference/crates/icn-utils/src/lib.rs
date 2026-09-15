//! Shared infrastructure utilities for ICN crates.

pub mod file_cache;
pub mod sparse_file;

#[cfg(windows)]
pub mod windows_process;
