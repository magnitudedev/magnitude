//! The threads this process may run at once: its share of the machine.
//!
//! The process's auxiliary pools (memory fault-in, native-target workers,
//! repack, formation) size themselves by [`share`], never by the machine: a
//! process the service starts is handed its share at startup
//! ([`set_share`]), the serving worker all of the machine's cores and
//! background work fewer, so its auxiliary pools leave the serving worker
//! room. A device's own pools (the CPU device's, the expert-compute pool)
//! never read the share: their size is part of the device's identity, which
//! a measurement sizes and serving then opens, so they take the machine as
//! the process sees it. A process handed none (a tool, a test) takes its
//! [`quota`].

use std::num::NonZeroUsize;
use std::sync::OnceLock;

/// The share, fixed at its first read or its one setting, whichever is
/// first.
static SHARE: OnceLock<NonZeroUsize> = OnceLock::new();

/// The threads this process may run at once: what [`set_share`] set, or,
/// where nothing set it before this first read, the process's [`quota`].
pub fn share() -> NonZeroUsize {
    *SHARE.get_or_init(quota)
}

/// Set this process's share, at its startup, before any pool reads it.
///
/// # Panics
/// Once the share is set or read: a pool sized by another share runs
/// beside the pools sized by this one, and the process takes more of the
/// machine than it was handed.
pub fn set_share(share: NonZeroUsize) {
    if SHARE.set(share).is_err() {
        panic!("invariant violated: the process's thread share is set once, at startup, before any pool reads it");
    }
}

/// The CPUs the process may keep busy at once: its affinity, under a
/// cgroup's CPU quota (`cpu.max`), as `available_parallelism` reads them;
/// one where the OS reports none.
pub fn quota() -> NonZeroUsize {
    std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN)
}
