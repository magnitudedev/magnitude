//! Host RAM sample on macOS: live `vm.*` sysctls, `host_statistics64`
//! snapshots paired with task paging, and Apple Silicon IOGPU wired limits.

use super::super::{
    beyond, resolve, ColdPages, Counted, Harm, HostSample, Lag, OwnReclaimed, PressureLevel,
    Supply, Widened, WiredFigures, WiredLimit,
};
use super::{capacity, sysctl_always, sysctl_count, Sys};
use std::time::{Duration, Instant};

/// XNU's `HOST_STATISTICS_TIME_WINDOW` (`osfmk/kern/host.c`): within a
/// window of it, once ordinary processes have asked a flavor more often than
/// the window's random allowance (2 to 10 requests), `host_statistics64`
/// serves the snapshot the latest ordinary request took. The inactive,
/// decompression and swap-in counts have no live sysctl, nor does the wired
/// count before macOS 26 (macOS 15.7.9 defines no `vm.page_wired_count`),
/// so they show the machine as late as this.
pub(super) const STATISTICS_WINDOW: Duration = Duration::from_secs(1);

/// The bytes of `statistics`, zeroed before the kernel wrote it, padding
/// included: two snapshots are the same exactly where these are.
fn snapshot_bytes(
    statistics: &libc::vm_statistics64,
) -> [u8; std::mem::size_of::<libc::vm_statistics64>()] {
    // SAFETY: a plain C structure of integers, read as its bytes.
    unsafe { std::mem::transmute_copy(statistics) }
}

/// The counters' snapshot a sample read, with what it pairs with.
#[derive(Clone, Copy)]
pub(super) struct Snapshot {
    pub(super) statistics: libc::vm_statistics64,
    pub(super) counted: Counted,
    /// This process's own paging, read as the snapshot was first served.
    pub(super) own: u64,
    /// The file pages the OS keeps ([`file_floor`]), in pages.
    pub(super) file_floor: u64,
}

/// The file pages the OS keeps while it compresses others' memory, as the
/// snapshot `statistics` after `before` shows them: where the OS compressed
/// more pages between the two than were free above its free-page target
/// `free_target` (`free_count` excludes `speculative_count` as
/// `vm.page_free_count` does), it chose compressing over every file page still
/// standing, so those are its floor (a mapped library or binary in use);
/// when compression stops and free pages recover comfortably above target, the
/// temporary floor decays rather than latching permanently; clamped to the
/// snapshot's external file pages.
pub(super) fn file_floor(
    before: Option<&Snapshot>,
    statistics: &libc::vm_statistics64,
    free_target: u64,
) -> u64 {
    let Some(before) = before else { return 0 };
    let compressed =
        u64::from(statistics.compressions).saturating_sub(u64::from(before.statistics.compressions));
    let external = u64::from(statistics.external_page_count);
    let free = u64::from(statistics.free_count).saturating_sub(u64::from(statistics.speculative_count))
        + u64::from(statistics.purgeable_count);
    let floor = if compressed > free.saturating_sub(free_target) {
        external
    } else if compressed == 0 && free >= free_target.saturating_mul(4) {
        0
    } else if compressed == 0 && free > free_target.saturating_mul(2) {
        before.file_floor / 2
    } else {
        before.file_floor
    };
    floor.min(external)
}

/// The snapshot the latest sample was served, and when that sample began.
pub(super) struct Served {
    pub(super) snapshot: Snapshot,
    pub(super) began: Instant,
}

/// The process's samples' snapshots, in order.
static SERVED: std::sync::Mutex<Option<Served>> = std::sync::Mutex::new(None);

/// The VM's counts and this process's own paging as one snapshot, read in a
/// sample that began at `began`, with the task's figures read beside it,
/// live.
///
/// A snapshot differing from the one the sample before was served was taken
/// after that sample began (the kernel serves the latest one taken), within
/// [`STATISTICS_WINDOW`] of now, and before now: its counters count every
/// event before the later of the two bounds, none after now, and this
/// process's paging is read just before now, beside it. One the same as the
/// sample before's is that snapshot, served again, or the machine's counts
/// unchanged since, this process's paging among them: it keeps the instant
/// and the paging it was first read with.
fn snapshot(sys: &impl Sys, began: Instant, page: u64, free_target: u64) -> (Snapshot, TaskVm) {
    // A holder that panicked left the last snapshot whole: it is replaced
    // in one store.
    let mut served = SERVED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let statistics = sys.vm_statistics();
    // Read on every sample, so its 32-bit decompressions widen across
    // every wrap; after the machine's, so a page between the two counts as
    // the process's alone, never another program's.
    let task = sys.task_vm();
    let snapshot = paired(
        served.take(),
        statistics,
        Instant::now(),
        own_paging(&task, page),
        free_target,
    );
    *served = Some(Served { snapshot, began });
    (snapshot, task)
}

/// The snapshot `statistics`, served at `at` beside this process's paging
/// `own`, after the sample `before` ([`snapshot`]), its file floor judged
/// against the free-page target `free_target` ([`file_floor`]).
pub(super) fn paired(
    before: Option<Served>,
    statistics: libc::vm_statistics64,
    at: Instant,
    own: u64,
    free_target: u64,
) -> Snapshot {
    match before {
        Some(Served { snapshot, .. })
            if snapshot_bytes(&snapshot.statistics) == snapshot_bytes(&statistics) =>
        {
            if at.saturating_duration_since(snapshot.counted.at) >= STATISTICS_WINDOW {
                // Beyond XNU's 1s cache window, identical statistics show that no
                // event occurred; advance `counted.at` so journal settlement never stalls.
                Snapshot {
                    counted: Counted { at, lag: STATISTICS_WINDOW },
                    own,
                    ..snapshot
                }
            } else {
                snapshot
            }
        }
        before => {
            let lag = before.as_ref().map_or(STATISTICS_WINDOW, |before| {
                at.duration_since(before.began).min(STATISTICS_WINDOW)
            });
            let file_floor = file_floor(
                before.as_ref().map(|before| &before.snapshot),
                &statistics,
                free_target,
            );
            Snapshot {
                statistics,
                counted: Counted { at, lag },
                own,
                file_floor,
            }
        }
    }
}

/// `task_vm_info` (`TASK_VM_INFO`) through revision 7, owned by
/// `seismic_native_target::process::macos::TaskVmInfo`.
pub(crate) use seismic_native_target::process::macos::TaskVmInfo as TaskVm;

/// The process's task's decompressions, widened across their wraps.
static OWN_DECOMPRESSIONS: Widened = Widened::new();

/// This process's own paging over the harm counters' events: its pages
/// decompressed (`task_vm_info.decompressions`) and swapped in
/// (`ledger_swapins`, in bytes), as pages of `page`.
///
/// # Panics
/// Where the swap-ins read negative.
fn own_paging(vm: &TaskVm, page: u64) -> u64 {
    let ledger_swapins = vm.ledger_swapins;
    let swapins = u64::try_from(ledger_swapins).unwrap_or_else(|_| {
        panic!("invariant violated: task_vm_info.ledger_swapins is {ledger_swapins}")
    });
    OWN_DECOMPRESSIONS.of(vm.decompressions as u32) + swapins / page
}

/// Free swap (`vm.swapusage`'s `xsu_avail`); none where it is not reported.
///
/// # Panics
/// Where it reports another size than its structure's.
pub(super) fn swap_free(sys: &impl Sys) -> u64 {
    resolve(
        &[((), &|| {
            let bytes = sys.sysctl("vm.swapusage")?;
            assert_eq!(
                bytes.len(),
                std::mem::size_of::<libc::xsw_usage>(),
                "invariant violated: sysctl vm.swapusage returned {} bytes",
                bytes.len()
            );
            // SAFETY: the structure's bytes, as the OS wrote them.
            let usage: libc::xsw_usage = unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast()) };
            Some(usage.xsu_avail)
        })],
        (),
        || 0,
    )
    .0
}

/// `kern.memorystatus_vm_pressure_level`: 1 normal, 2 warn, 4 critical;
/// none where it is not reported.
///
/// # Panics
/// Where it reads another level.
pub(super) fn pressure_level(sys: &impl Sys) -> Option<PressureLevel> {
    Some(
        match sysctl_count(sys, "kern.memorystatus_vm_pressure_level")? {
            1 => PressureLevel::Normal,
            2 => PressureLevel::Warn,
            4 => PressureLevel::Critical,
            level => panic!(
                "invariant violated: kern.memorystatus_vm_pressure_level is {level}"
            ),
        },
    )
}

/// The ceiling on host RAM the GPU may wire, where the GPU wires host RAM:
/// Apple silicon's IOGPU stack defines `iogpu.wired_limit_mb` (macOS 14+,
/// in MiB) or `debug.iogpu.wired_limit` (macOS 13, in bytes). The first of
/// them set (non-zero) is the limit; neither set is the OS's default
/// ([`WiredLimit::OsDefault`]). None where the OS defines neither: the GPU
/// wires nothing through IOGPU (an Intel Mac: `ENOENT` on macOS 15.7.9
/// x86_64), and the sample has no wired figures.
pub(super) fn wired_limit(sys: &impl Sys) -> Option<WiredLimit> {
    let megabytes = sysctl_count(sys, "iogpu.wired_limit_mb");
    let bytes = sysctl_count(sys, "debug.iogpu.wired_limit");
    if megabytes.is_none() && bytes.is_none() {
        return None;
    }
    Some(
        resolve(
            &[
                ((), &|| {
                    megabytes
                        .filter(|megabytes| *megabytes > 0)
                        .map(|megabytes| WiredLimit::Set(megabytes << 20))
                }),
                ((), &|| {
                    bytes.filter(|bytes| *bytes > 0).map(WiredLimit::Set)
                }),
            ],
            (),
            || WiredLimit::OsDefault,
        )
        .0,
    )
}

/// `bytes` times `numerator / denominator`, never more than `bytes`, so it
/// fits `bytes`'s width whatever the ratio.
fn share(bytes: u64, numerator: u64, denominator: u64) -> u64 {
    (u128::from(bytes) * u128::from(numerator) / u128::from(denominator))
        .min(u128::from(bytes)) as u64
}

/// What `stored` compressed bytes holding `logical` bytes save of each byte
/// compressed, applied to `bytes`: none while the compressor stores nothing.
pub(super) fn compressed_saving(bytes: u64, stored: u64, logical: u64) -> u64 {
    match logical {
        0 => 0,
        logical => share(bytes, logical.saturating_sub(stored), logical),
    }
}

/// What `bytes` compressed bytes occupy in a compressor storing `stored`
/// bytes for `logical` bytes, at its ratio: none while it stores nothing,
/// and never more than `bytes`.
pub(super) fn compressed_stored(bytes: u64, stored: u64, logical: u64) -> u64 {
    match logical {
        0 => 0,
        logical => share(bytes, stored, logical),
    }
}

/// Host RAM now: the levels from the live `vm.*` sysctls (the free and
/// purgeable counts, the free target, free swap, the pressure level and the
/// GPU's wired limit), and the counts no sysctl reads from
/// `host_statistics64`'s snapshot, with this process's own paging beside it
/// ([`snapshot`]). The clean cache, the cold pages and what compressing them
/// saves are all that one snapshot's (its file-backed pages above the floor
/// the OS keeps, its inactive pages, the pages the compressor occupies and
/// the pages it holds), so no page counts in two parts and the saving is the
/// compressor's as it stood when those pages were counted; free swap, where
/// the rest can go, is the live figure. macOS enforces no process
/// memory limit (`RLIMIT_AS`/`RLIMIT_DATA` bind no allocation), so the host
/// is the scope.
///
/// Every macOS reports `hw.memsize`, `vm.page_free_count` and
/// `vm.vm_page_free_target`. Where the rest is not reported: no purgeable
/// pages or free swap (`vm.page_purgeable_count`, `vm.swapusage`), no
/// pressure level, and no wired figures without the IOGPU stack
/// ([`wired_limit`]).
///
/// # Panics
/// Where `hw.memsize`, `vm.page_free_count` or `vm.vm_page_free_target` is
/// not reported.
pub(crate) fn sample(sys: &impl Sys) -> HostSample {
    let at = Instant::now();
    let page = sys.page();
    let pages = |count: u64| {
        count
            .checked_mul(page)
            .unwrap_or_else(|| panic!("invariant violated: {count} pages overflow bytes"))
    };
    let total = capacity(sys).value;
    let free = sysctl_always(sys, "vm.page_free_count");
    let purgeable = resolve(
        &[((), &|| sysctl_count(sys, "vm.page_purgeable_count"))],
        (),
        || 0,
    )
    .0;
    let free_target_raw = sysctl_always(sys, "vm.vm_page_free_target");
    // Preserve at least a small OS reserve cushion on multi-GB Macs where XNU's
    // raw free_target (4000 pages = 64 MiB on 64 GiB) leaves almost no buffer
    // against background app spikes before hitting the compressor.
    let reserve_pages = (total / page / 256).min((256 << 20) / page);
    let free_target = if free_target_raw > 0 {
        free_target_raw.max(reserve_pages)
    } else {
        0
    };
    let external_live = sysctl_count(sys, "vm.page_pageable_external_count");
    let xpmapped_min = sysctl_count(sys, "vm.vm_page_xpmapped_min").unwrap_or(0);
    let wired_limit = wired_limit(sys);
    let swap = swap_free(sys);
    let pressure = pressure_level(sys);
    let (
        Snapshot {
            statistics,
            counted,
            own,
            file_floor,
        },
        task,
    ) = snapshot(sys, at, page, free_target_raw);
    // Free and volatile purgeable pages (purged at no one's cost) less the
    // free-page target, which the pageout daemon refills by compressing,
    // swapping and evicting other pages: pages taken below it are never free
    // without displacing someone, and a fall below it is a fall all the
    // same.
    let free_beyond_target = beyond(pages(free + purgeable), pages(free_target));
    // Inactive anonymous pages (`inactive` minus non-speculative file-backed
    // pages and volatile purgeable pages, which are already counted 100% in
    // `free_beyond_target`), computed within the one snapshot so supply can
    // never exceed `total` nor double-count volatile pages.
    let external_snap = u64::from(statistics.external_page_count);
    let non_spec_external = external_snap.saturating_sub(u64::from(statistics.speculative_count));
    let cold = pages(
        u64::from(statistics.inactive_count)
            .saturating_sub(non_spec_external)
            .saturating_sub(purgeable),
    );
    // Live file-backed pages (`vm.page_pageable_external_count` when present,
    // preventing 1s clock skew against `vm.page_free_count`), minus the larger
    // of XNU's executable-mapped floor (`vm.vm_page_xpmapped_min`) and any
    // dynamic compression floor (`file_floor`).
    let external = external_live.unwrap_or(external_snap);
    let protected_file = file_floor.max(xpmapped_min);
    let clean_cache = pages(external.saturating_sub(protected_file));
    let stored = pages(u64::from(statistics.compressor_page_count));
    let logical = pages(statistics.total_uncompressed_pages_in_compressor);
    // `free_beyond_target` and `clean_cache` come from live `vm.*` sysctls while
    // `cold` comes from the 1s `host_statistics64` snapshot; bound `cold_reclaimable`
    // so an app exiting inside that window never causes `supply.total() > total`.
    let immediate = u64::try_from(free_beyond_target.max(0))
        .unwrap_or(0)
        .saturating_add(clean_cache);
    let cold_reclaimable = cold
        .min(compressed_saving(cold, stored, logical).saturating_add(swap))
        .min(total.saturating_sub(immediate));
    HostSample {
        at,
        page_bytes: page,
        total,
        supply: Supply {
            free_beyond_target,
            clean_cache,
            cold_reclaimable,
        },
        counted,
        cold: ColdPages {
            machine: cold,
            own: None,
        },
        own_reclaimed: Some(OwnReclaimed {
            bytes: task.compressed,
            stored: compressed_stored(task.compressed, stored, logical),
            exact: true,
        }),
        wired: wired_limit.map(|limit| WiredFigures {
            limit,
            wire_count: pages(u64::from(statistics.wire_count)),
        }),
        harm: Some(Harm {
            machine: statistics.decompressions.saturating_add(statistics.swapins),
            own,
        }),
        pressure,
        // A release of anonymous memory returns its pages in the call, and
        // a growth counts at its first touch.
        lag: Lag {
            release: Duration::ZERO,
            growth: Duration::ZERO,
        },
    }
}
