//! Shared infrastructure utilities for ICN crates.

#![cfg_attr(
    test,
    allow(
        clippy::disallowed_methods,
        reason = "tests start threads of their own; the code they test starts its threads through seismic-threads"
    )
)]

pub mod blocking;
pub mod file_cache;
pub mod owner;
pub mod process;
pub mod sparse_file;

#[cfg(windows)]
pub mod windows_process;
