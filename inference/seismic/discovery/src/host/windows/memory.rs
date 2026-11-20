//! Host RAM sample on Windows: `GlobalMemoryStatusEx`, PDH memory counters,
//! NUMA node free lists, and job-object limits.

use super::super::{
    beyond, binding, resolve, ColdPages, Counted, HostSample, Lag, OwnReclaimed, PressureLevel,
    Scope, Supply,
};
use super::limits::job_limits;
use super::Sys;
use std::time::{Duration, Instant};
use windows_sys::Win32::System::Performance::{PDH_HCOUNTER, PDH_HQUERY};

/// The performance counters a sample reads, collected together.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Counters {
    /// `\Memory\Free & Zero Page List Bytes`.
    pub(super) free_zero: u64,
    /// `\Memory\Standby Cache {Core, Normal Priority, Reserve} Bytes`.
    pub(super) standby: u64,
}

/// The open query of [`Counters`], opened once.
pub(super) struct Query {
    pub(super) query: PDH_HQUERY,
    /// Free & zero and the three standby lists.
    pub(super) counters: [PDH_HCOUNTER; 4],
}

// SAFETY: PDH's handles are process-wide; the query is only collected under
// its lock.
unsafe impl Send for Query {}

pub(super) const COUNTER_PATHS: [&str; 4] = [
    "\\Memory\\Free & Zero Page List Bytes",
    "\\Memory\\Standby Cache Core Bytes",
    "\\Memory\\Standby Cache Normal Priority Bytes",
    "\\Memory\\Standby Cache Reserve Bytes",
];

/// Host RAM now: the free and zero lists and the standby lists
/// ([`lists`]), bounded by the commit this process has left
/// (`GlobalMemoryStatusEx`) and by its own job's limits ([`job_limits`]);
/// the low-memory notification. Windows reports no `harm`: no unprivileged
/// counter separates reclaimed-page refaults from cold file reads.
pub(crate) fn sample(sys: &impl Sys) -> HostSample {
    let at = Instant::now();
    let status = sys.memory_status();
    let counters = sys.counters();
    let (free_zero, standby) = lists(sys, counters.as_ref(), status.ullAvailPhys);
    let limits = job_limits(sys, &status);
    let pressure = match sys.low_memory() {
        true => PressureLevel::Warn,
        false => PressureLevel::Normal,
    };
    let counted_at = Instant::now();
    // What the lists hold, within the commit this process can still make:
    // a page taken is committed first.
    let room = (free_zero + standby).min(status.ullAvailPageFile);
    let free = free_zero.min(room);
    let host = Scope {
        total: status.ullTotalPhys,
        supply: Supply {
            free_beyond_target: beyond(free, 0),
            clean_cache: room - free,
            cold_reclaimable: 0,
        },
    };
    // A job's commit and working set count a growth in the call, whichever
    // scope binds.
    let scope = binding(host, &limits).scope;
    HostSample {
        at,
        page_bytes: sys.page(),
        total: scope.total,
        supply: scope.supply,
        counted: Counted {
            at: counted_at,
            lag: counted_at.duration_since(at),
        },
        // The working-set manager trims cold pages to the standby list,
        // which `clean_cache` counts.
        cold: ColdPages {
            machine: 0,
            own: None,
        },
        own_reclaimed: own_trimmed(sys),
        wired: None,
        harm: None,
        pressure: Some(pressure),
        // A free decommits in the call, and a growth commits in it.
        lag: Lag {
            release: Duration::ZERO,
            growth: Duration::ZERO,
        },
    }
}

/// The free and zero lists and the standby lists, in bytes: the
/// performance counters' (`counters`); where they are unanswered, the free
/// and zero lists the sum of every memory node's available bytes
/// (`GetNumaAvailableMemoryNodeEx`, which on a machine of more than one
/// node counts exactly those lists), within `available` (`ullAvailPhys`,
/// the free, zero and standby lists together), and the standby lists the
/// rest of `available`. On a machine of one node, whose available bytes
/// include its standby lists, all of `available` then reads as free.
fn lists(sys: &impl Sys, counters: Option<&Counters>, available: u64) -> (u64, u64) {
    resolve(
        &[
            ((), &|| counters.map(|c| (c.free_zero, c.standby))),
            ((), &|| {
                let mut answered = false;
                let mut sum = 0u64;
                for node in 0..=sys.highest_node() {
                    if let Some(bytes) = sys.numa_available(node) {
                        answered = true;
                        sum = sum.saturating_add(bytes);
                    }
                }
                answered.then(|| {
                    let free_zero = sum.min(available);
                    (free_zero, available - free_zero)
                })
            }),
        ],
        (),
        || (0, available),
    )
    .0
}

/// This process's private pages out of its working set: its private commit
/// less its private working set ([`HostSample::own_reclaimed`]); none
/// before Windows 10 1809, which reports no private working set.
fn own_trimmed(sys: &impl Sys) -> Option<OwnReclaimed> {
    let private_working_set = sys.private_working_set()?;
    // A copy-on-write page of a mapped image turns private in the working
    // set with its commit charged to the image's section, so the private
    // working set may exceed the private commit: none is then out of it.
    // An upper bound ([`OwnReclaimed::exact`]), and a trimmed page waits in
    // RAM on the modified or standby list, which no figure here separates:
    // none is counted stored.
    Some(OwnReclaimed {
        bytes: (sys.memory_counters().PagefileUsage as u64).saturating_sub(private_working_set),
        stored: 0,
        exact: false,
    })
}
