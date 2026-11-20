//! The host: RAM capacity, the OS build, the CPU, and host RAM's sample at
//! one instant ([`HostSample`]) within the process limits that bound every
//! path into it. Host modules never load GPU drivers.

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "windows")]
mod windows;

#[cfg(target_os = "linux")]
use linux as platform;
#[cfg(target_os = "macos")]
use macos as platform;
#[cfg(target_os = "windows")]
use windows as platform;

#[cfg(target_os = "linux")]
pub(crate) use linux::pressure_files;
pub(crate) use platform::{LiveSys, Sys};
#[cfg(all(test, target_os = "linux"))]
pub(crate) use linux::Tree;

use seismic_topology::{CacheSize, Firmware, HostNode, PciLocation, Processors, Reported};
use std::fmt;
#[cfg(any(not(target_os = "linux"), test))]
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// One OS read: the value, or `None` when the source is missing or refused
/// ([`seismic_memory::unanswered`]). Any other error (`EIO`, invalid UTF-8)
/// is an invariant violation and panics.
pub(crate) fn read<T>(what: impl fmt::Display, result: std::io::Result<T>) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(error) if seismic_memory::unanswered(&error) => None,
        Err(error) => panic!("invariant violated: {what}: {error}"),
    }
}

pub(crate) use seismic_topology::resolve;

/// The host's diagnostics for the hardware report: on Linux, a container
/// whose own cgroup sets no memory limit, so limits above it are not
/// visible (the one line of its kind).
pub(crate) fn diagnostics(sys: &impl Sys) -> Vec<seismic_topology::Diagnostic> {
    platform::diagnostics(sys)
}

/// Host RAM capacity: RAM the OS can use.
pub(crate) fn capacity(sys: &impl Sys) -> Reported<u64> {
    platform::capacity(sys)
}

/// The OS version and build, as the OS reports it: the driver of the CPU
/// executor and of Metal.
pub(crate) fn os_build(sys: &impl Sys) -> String {
    platform::os_build(sys)
}

/// Human-readable CPU brand string (e.g. `"Apple M4 Pro"`, `"AMD EPYC 7763"`),
/// falling back to `"CPU (<arch>)"` when no source answers.
pub(crate) fn cpu_model(sys: &impl Sys) -> String {
    platform::cpu_model(sys)
}

/// The CPU as the OS reports it: the platform's identifying CPU fields,
/// then `;isa=<tier>` for the highest native tier it supports, and
/// `;cache=<bytes>` for its reported cache when non-zero.
pub(crate) fn cpu_identity(sys: &impl Sys, cache: CacheSize) -> String {
    let mut identity = platform::cpu_identity(sys);
    if let Some(tier) = seismic_native_cpu::Tier::detected() {
        identity.push_str(&format!(";isa={}", tier.name()));
    }
    if cache.bytes > 0 {
        identity.push_str(&format!(";cache={}", cache.bytes));
    }
    identity
}

/// The CPU's last-level cache size and whether it is the full LLC before RAM.
pub(crate) fn last_level_cache(sys: &impl Sys) -> CacheSize {
    platform::last_level_cache(sys)
}

/// The system's firmware strings (`seismic_topology::SystemIdentity`).
pub(crate) fn firmware(sys: &impl Sys) -> Firmware {
    platform::firmware(sys)
}

/// The host's processors (`seismic_topology::Processors`).
pub(crate) fn processors(sys: &impl Sys) -> Processors {
    platform::processors(sys)
}

/// The memory node of the PCI device at `location`.
pub(crate) fn device_node(sys: &impl Sys, location: PciLocation) -> HostNode {
    platform::device_node(sys, location)
}

/// Place every thread of the process on `cpus`
/// (`crate::processors::place_process`).
pub(crate) fn place_process(cpus: &[usize]) -> Result<(), String> {
    platform::place_process(cpus)
}

/// Place the calling thread on `cpus` (`crate::processors::pin`).
pub(crate) fn pin(cpus: &[usize]) -> Result<(), String> {
    platform::pin(cpus)
}

/// Host RAM at one instant ([`HostSample`]), read now: every figure from
/// its sources in order, or its documented value where none answers.
pub fn host_sample() -> HostSample {
    sample(&LiveSys)
}

/// Host RAM at one instant from `sys` ([`host_sample`]).
pub(crate) fn sample(sys: &impl Sys) -> HostSample {
    platform::sample(sys)
}

/// Host RAM at one instant: what the OS can give a taker without writing
/// anyone's dirty data ([`Supply`]), the total it is of, the OS's pressure
/// level, and the machine's counters ([`Counted`]): other programs' cold
/// pages, the harm that shows where their working sets lie with this
/// process's own share of it, and on macOS the wired count the GPU's
/// ceiling reads. Every figure is the machine's, ours included: the memory
/// ledger attributes its own changes ([`Lag`] says how late each shows in
/// the supply, [`Counted`] what the counters show).
///
/// Its total is the least of the host's RAM and each limit on this
/// process ([`Limit`]: a Linux cgroup's, `RLIMIT_AS`, its memory nodes', a
/// Windows job's), read again at every sample, and its supply the least
/// any of them leaves ([`binding`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostSample {
    /// When the reading of the live figures (`total`, `supply`,
    /// `pressure`, `wired.limit`) began.
    pub at: Instant,
    /// The page size `cold`, `harm` and `wired` count in.
    pub page_bytes: u64,
    /// Bytes of the binding scope.
    pub total: u64,
    pub supply: Supply,
    /// When the counters below show the machine.
    pub counted: Counted,
    /// Cold anonymous pages machine-wide and, where the OS isolates it
    /// (Linux solo cgroup v2), this process's own share. A counter ([`Counted`]).
    pub cold: ColdPages,
    /// This process's own anonymous memory the OS holds outside RAM at the
    /// sample, compressed or swapped ([`OwnReclaimed`]): what of the
    /// process's allocations `total − supply` no longer shows. Read live
    /// with `supply`, never a counter.
    /// The OS reports it directly, where a resident figure would not
    /// separate the process's allocations from what else is mapped into it
    /// (a macOS task's resident size includes the Metal buffers and files
    /// it maps; a cgroup's `memory.current` includes its page cache, which
    /// `supply` counts).
    /// - macOS: `task_vm_info.compressed`, the task's `internal_compressed`
    ///   ledger: a page compressed is debited from the task's resident
    ///   anonymous pages and credited here at a whole page, and stays here
    ///   while its compressor segment is swapped out. Wired pages are never
    ///   compressed.
    /// - Linux: `/proc/self/status`'s `VmSwap`, the process's anonymous
    ///   pages in swap, zswap's included (the mm's `MM_SWAPENTS`); none
    ///   where the kernel reports no `VmSwap` (no `CONFIG_SWAP`, a sandbox's
    ///   own `/proc`). It is
    ///   the mm's counter as `/proc` reads it: per-thread caches of up to 64
    ///   events before Linux 6.2, a per-CPU counter's approximate read
    ///   since, so it may be off by those batches either way. A driver's
    ///   pinned pages (CUDA's page-locked host memory) are never swapped.
    /// - Windows: the process's private commit (`PrivateUsage`) less its
    ///   private working set (`PrivateWorkingSetSize`): private pages
    ///   trimmed out of the working set, an upper bound of those out of
    ///   RAM, since a trimmed page waits on the modified list until it is
    ///   written to the page file, and a committed page never touched
    ///   counts too. None before Windows 10 1809, which reports no private
    ///   working set.
    pub own_reclaimed: Option<OwnReclaimed>,
    /// The GPU's wired figures: on macOS only, where the GPU wires host RAM.
    pub wired: Option<WiredFigures>,
    /// The machine's harm and this process's share of it, counted together;
    /// none where the OS counts no share of this process's in the machine's
    /// unit ([`Harm`]).
    pub harm: Option<Harm>,
    /// The OS's pressure level; none where the OS reports none (Linux built
    /// or booted without PSI, `psi=0`).
    pub pressure: Option<PressureLevel>,
    pub lag: Lag,
}

/// What the OS can give a taker without writing anyone's dirty data, in
/// bytes, in its three parts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Supply {
    /// Free pages less the floor the OS keeps free by reclaiming, negative
    /// where the OS has fallen below that floor (pages taken of its reserve
    /// that its reclaim has not yet refilled), so a fall of free memory
    /// shows whole, above the floor or below it: macOS
    /// `vm.page_free_count + vm.page_purgeable_count` less
    /// `vm.vm_page_free_target`; Linux `MemFree` less the zones' low
    /// watermarks (`/proc/zoneinfo`); Windows the free and zero lists, the
    /// OS keeping no floor of them. In a binding limit, the limit less its
    /// usage.
    pub free_beyond_target: i64,
    /// File pages no one has dirtied, which the OS drops without writing:
    /// macOS file-backed pages (`external_page_count` of a
    /// `host_statistics64` snapshot; macOS counts no dirty file page) above
    /// those it keeps while it compresses others' memory, the file pages
    /// standing where it last compressed more than was free above its
    /// target (none before);
    /// Linux `Active(file) + Inactive(file) −
    /// Dirty − Writeback`, within a cgroup its `active_file + inactive_file −
    /// file_dirty − file_writeback`; Windows the standby lists.
    pub clean_cache: u64,
    /// What compressing or swapping the cold anonymous pages frees, bounded
    /// by where they can go: macOS [`HostSample::cold`] times what the
    /// compressor saves of a page (`1 − compressor_page_count /
    /// total_uncompressed_pages_in_compressor`, of the snapshot `cold` is
    /// counted in, none while it stores nothing) plus free swap
    /// (`vm.swapusage`), never more than the cold pages; Linux `Inactive(anon)` within `SwapFree`; none on Windows
    /// ([`HostSample::cold`]) and none within a binding limit.
    pub cold_reclaimable: u64,
}

impl Supply {
    /// The three parts together, signed: below zero where the OS is short
    /// of its own floor by more than its cache and cold pages give.
    pub fn balance(&self) -> i64 {
        let rest = i64::try_from(self.clean_cache.saturating_add(self.cold_reclaimable)).expect("a store's bytes fit i64");
        self.free_beyond_target.saturating_add(rest)
    }

    /// What a taker can have now: the balance, none where it is short.
    pub fn total(&self) -> u64 {
        self.balance().max(0).unsigned_abs()
    }
}

/// `free` less `floor`, signed: what [`Supply::free_beyond_target`] holds.
pub(crate) fn beyond(free: u64, floor: u64) -> i64 {
    let signed = |bytes: u64| i64::try_from(bytes).expect("a store's bytes fit i64");
    signed(free) - signed(floor)
}

/// When a sample's counters show the machine, by the OS's mechanism that
/// serves them: every event at least `lag` before `at` shows in them, and
/// none after `at`. The counters are `cold`, `own_cold`, `harm` and the
/// wired count, read together, this process's share of `harm` just after
/// the machine's, so an event between the two is counted as its own and
/// never as another program's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Counted {
    pub at: Instant,
    /// - macOS: how long before `at` the snapshot `host_statistics64`
    ///   served may have been taken. It serves an ordinary process the
    ///   latest snapshot any ordinary process took while a window of XNU's
    ///   `HOST_STATISTICS_TIME_WINDOW` (1 s) holds more requests than the
    ///   window's random allowance, so a snapshot that differs from the one
    ///   the sample before was served was taken after that sample began, and
    ///   within the window: `lag` is the lesser of the two, and a snapshot
    ///   served again keeps the `at` and `lag` it was first read with.
    /// - Linux: the slower of the folds the counters pass through, the
    ///   per-CPU counts `/proc` sums (`vm.stat_interval`, its timer rounded
    ///   up to the next whole second by `round_jiffies_relative`) and, where
    ///   `harm` or `own_cold` is a cgroup's, memcg's periodic statistics
    ///   flush (`FLUSH_TIME`, 2 s, `mm/memcontrol.c`), from when the reading
    ///   began.
    /// - Windows: since the reading began, the counters being the kernel's
    ///   own counts, read in the call.
    pub lag: Duration,
}

/// This process's own anonymous bytes the OS holds outside RAM, compressed
/// or swapped ([`HostSample::own_reclaimed`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OwnReclaimed {
    /// The bytes out of RAM, counted at their uncompressed size.
    pub bytes: u64,
    /// Of `bytes`, what the OS's compressed store (macOS's compressor,
    /// Linux's zswap) holds of them in RAM, at its compressed size: in
    /// what the machine uses, and in no figure of what the process holds.
    /// None where the OS gives no figure (Linux outside a cgroup of this
    /// process alone or without zswap, Windows): the process's compressed
    /// pages then read as other programs' memory, by at most that store's
    /// size.
    /// - macOS: `bytes` at the compressor's machine-wide ratio
    ///   (`compressor_page_count / total_uncompressed_pages_in_compressor`),
    ///   the task's own ratio not being reported.
    /// - Linux: the own cgroup's `memory.stat` `zswap`.
    pub stored: u64,
    /// Whether `bytes` is the OS's exact count (macOS's `internal_compressed`
    /// ledger) or approximate (Linux's batched swap count, Windows's upper
    /// bound): where it is exact, a process's bytes the machine's use does
    /// not hold beyond it can only be an accounting error.
    pub exact: bool,
}

impl OwnReclaimed {
    /// None of the process's bytes out of RAM, exactly.
    pub const NONE: Self = Self { bytes: 0, stored: 0, exact: true };
}

/// Cold anonymous bytes machine-wide and this process's own share where the
/// OS isolates it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ColdPages {
    /// Machine-wide cold anonymous bytes (0 where the OS has no cold queue,
    /// e.g. Windows).
    pub machine: u64,
    /// This process's own share of `machine`: Linux's `inactive_anon` of this
    /// process's own cgroup v2 (`memory.stat`); `None` on macOS, Windows, and
    /// Linux in a shared cgroup or cgroup v1.
    pub own: Option<u64>,
}

/// macOS's wired figures: the system's GPU wired ceiling and what is wired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WiredFigures {
    pub limit: WiredLimit,
    /// Bytes wired machine-wide, ours included (`wire_count`): a counter
    /// ([`Counted`]).
    pub wire_count: u64,
}

/// The ceiling on host RAM Apple silicon's GPU may wire, as the IOGPU
/// stack's sysctls state it: `iogpu.wired_limit_mb` (macOS 14+, in MiB),
/// then `debug.iogpu.wired_limit` (macOS 13, in bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WiredLimit {
    /// Neither set (zero): the OS's default, which is no number the host
    /// reads; the device's `recommendedMaxWorkingSetSize` reports it, and
    /// the memory ledger reads it there.
    OsDefault,
    /// Set, in bytes.
    Set(u64),
}

/// The machine's harm: programs bringing back what the OS reclaimed from
/// them, cumulative and never decreasing (an OS count narrower than 64 bits
/// is widened across its wraps), and this process's share of the same
/// events in the same unit, so its own paging never reads as another
/// program's. One count of each OS:
/// - macOS: pages decompressed and swapped in, the machine's
///   (`HOST_VM_INFO64`'s `decompressions + swapins`) and the task's
///   (`task_vm_info`'s `decompressions` plus its `ledger_swapins` bytes in
///   pages).
/// - Linux: pages refaulted, the machine's (`/proc/vmstat`'s
///   `workingset_refault_anon + workingset_refault_file`,
///   `workingset_refault` before Linux 5.9) and those of this process's
///   own cgroup v2 (its `memory.stat`'s same counts). No count is kept per
///   process, so a cgroup's counts are this process's only where the
///   cgroup has a memory controller, lists this process alone and has no
///   descendant; in a cgroup shared with any other process, an ancestor's,
///   or without a cgroup v2 memory controller, there is no share, and no
///   harm.
/// - Windows: hard faults, each resolved by one paging read: the machine's
///   paging reads (`\Memory\Page Reads/sec`'s count) and the process's
///   `HardFaultCount`. Windows counts no process's pages brought in
///   (`\Memory\Pages Input/sec` has no per-process count), so the unit
///   is a read, of at least one page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Harm {
    pub machine: u64,
    pub own: u64,
}

/// The OS's memory pressure level now.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PressureLevel {
    Normal,
    /// macOS `kern.memorystatus_vm_pressure_level` warn; Linux PSI `some
    /// avg10` above zero; Windows the low-memory resource notification.
    Warn,
    /// macOS critical; Linux PSI `full avg10` above zero. Windows reports
    /// no level past low memory.
    Critical,
}

/// How late this process's own changes to host RAM show in a sample's
/// supply, by the OS's own mechanisms. Only the host's: a GPU allocator's
/// release on unified memory shows by that allocator's mechanism, which its
/// backend states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lag {
    /// How late this process's release shows: at once everywhere (`munmap`
    /// and `free` return the pages in the call; a cgroup's `memory.current`
    /// and a Windows job's commit uncharge them in it).
    pub release: Duration,
    /// How late this process's growth shows: at once on the host and in a
    /// Windows job (first touch, wiring and commit count in the call).
    /// Where a Linux cgroup's supply binds ([`binding`]), its clean cache is
    /// read from `memory.stat`, which shows the cache a growth displaced
    /// within memcg's periodic statistics flush (`FLUSH_TIME`, 2 s,
    /// `mm/memcontrol.c`): a growth shows whole within it.
    pub growth: Duration,
}

/// A 32-bit cumulative OS count widened to 64 bits across its wraps: every
/// reading of the count goes through one, so no wrap is missed while a
/// reading is taken more often than the count can wrap (4.3 billion events).
#[cfg(any(not(target_os = "linux"), test))]
pub(crate) struct Widened(Mutex<Option<(u32, u64)>>);

#[cfg(any(not(target_os = "linux"), test))]
impl Widened {
    pub(crate) const fn new() -> Self {
        Self(Mutex::new(None))
    }

    /// The count `raw` reads, widened.
    pub(crate) fn of(&self, raw: u32) -> u64 {
        let mut last = self.0.lock().expect("a widened count's lock is never poisoned");
        let wide = match *last {
            None => u64::from(raw),
            Some((before, wide)) => wide + u64::from(raw.wrapping_sub(before)),
        };
        *last = Some((raw, wide));
        wide
    }
}

/// A scope's total and supply: the host's, or a limit's.
#[cfg(any(not(target_os = "macos"), test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Scope {
    pub(crate) total: u64,
    pub(crate) supply: Supply,
}

/// The binding scope of a sample ([`binding`]), and which scope its supply
/// is.
#[cfg(any(not(target_os = "macos"), test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Binding {
    pub(crate) scope: Scope,
    /// The index in `limits` of the limit whose supply binds; none where the
    /// host's does. How late a growth shows in the supply is that scope's,
    /// so of scopes leaving the same supply the earliest binds, the host's
    /// first: a growth shows in the least supply as soon as in any of
    /// theirs.
    pub(crate) supplier: Option<usize>,
}

/// The binding scope of `host` and each limit of `limits`:
/// the least total of them all and the least supply any of them leaves.
/// Each bounds what this process may take, so both are bound by every one
/// at once: whichever leaves the least supply, the total never moves with
/// it, and the supply moves only as the scopes' own figures move.
#[cfg(any(not(target_os = "macos"), test))]
pub(crate) fn binding(host: Scope, limits: &[Limit]) -> Binding {
    limits.iter().map(|limit| limit.scope(host.total)).enumerate().fold(
        Binding { scope: host, supplier: None },
        |least, (index, bound)| {
            let (supply, supplier) = match bound.supply.balance() < least.scope.supply.balance() {
                true => (bound.supply, Some(index)),
                false => (least.scope.supply, least.supplier),
            };
            Binding { scope: Scope { total: least.scope.total.min(bound.total), supply }, supplier }
        },
    )
}

/// A limit on this process's memory, read, with the usage it is enforced
/// against. A limit no source shows is no limit: the host's figures and the
/// process's rlimits bound it.
#[cfg(any(not(target_os = "macos"), test))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Limit {
    /// Which limit it is.
    pub(crate) kind: ProcessLimitKind,
    /// The limit.
    pub(crate) limit_bytes: u64,
    /// The usage the limit is enforced against.
    pub(crate) used_bytes: u64,
    /// Clean file cache the OS can reclaim within this limit. Zero for a
    /// limit of address space or commit, where reclaiming physical pages
    /// does not release the accounted resource.
    pub(crate) reclaimable_bytes: u64,
}

#[cfg(any(not(target_os = "macos"), test))]
impl Limit {
    /// The limit as a scope of host RAM of `capacity` bytes: its room
    /// beyond its usage, negative where the usage overran it (cgroup v2's
    /// `memory.high` lets it, reclaiming and throttling past it), and the
    /// clean file cache within it, never more than its usage (the counters
    /// are read apart), so its supply never exceeds the limit.
    fn scope(&self, capacity: u64) -> Scope {
        Scope {
            total: self.limit_bytes.min(capacity),
            supply: Supply {
                free_beyond_target: beyond(self.limit_bytes, self.used_bytes),
                clean_cache: self.reclaimable_bytes.min(self.used_bytes),
                cold_reclaimable: 0,
            },
        }
    }
}

/// An OS limit on this process's memory.
#[cfg(any(not(target_os = "macos"), test))]
#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(dead_code)] // each OS builds only its own kinds; a cgroup is named for Debug
pub(crate) enum ProcessLimitKind {
    /// cgroup v1: `memory.stat`'s `hierarchical_memory_limit` of this
    /// process's cgroup, the least limit of it and every ancestor (past any
    /// namespace), enforced against its `memory.usage_in_bytes`.
    CgroupV1 { cgroup: String },
    /// cgroup v2 `memory.max` (or the lower `memory.high`, past which the
    /// kernel reclaims and throttles) of this process's cgroup or a visible
    /// ancestor, enforced against that cgroup's `memory.current`.
    CgroupV2 { cgroup: String },
    /// `RLIMIT_AS`, enforced against the process's `VmSize`.
    AddressSpace,
    /// The memory nodes `Mems_allowed` confines the process to, a strict
    /// subset of those with memory: their total, enforced against their
    /// used memory.
    AllowedNodes { nodes: Vec<usize> },
    /// The commit limit of the Windows job this process runs in (its job's
    /// or its process limit), enforced against the commit it counts.
    WindowsJobCommit,
    /// A Windows job's working-set limit (`MaximumWorkingSetSize`): a
    /// physical-memory scope, past which pages are trimmed.
    WindowsJobWorkingSet,
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;

    fn cgroup(limit: u64, used: u64, reclaimable: u64) -> Limit {
        Limit {
            kind: ProcessLimitKind::CgroupV2 { cgroup: "/load".into() },
            limit_bytes: limit,
            used_bytes: used,
            reclaimable_bytes: reclaimable,
        }
    }

    fn host(total: u64, free: u64, clean: u64, cold: u64) -> Scope {
        Scope { total, supply: Supply { free_beyond_target: beyond(free, 0), clean_cache: clean, cold_reclaimable: cold } }
    }

    /// A cgroup's file cache is supply within it, all of it the kernel can
    /// reclaim, not only its inactive part; it binds where its supply is
    /// the least.
    #[test]
    fn a_cgroups_reclaimable_file_cache_is_supply() {
        let within = binding(host(64 * GB, 40 * GB, 0, 0), &[cgroup(32 * GB, 30 * GB, 10 * GB)]);
        assert_eq!(within, Binding { scope: host(32 * GB, 2 * GB, 10 * GB, 0), supplier: Some(0) });
    }

    /// The host's supply binds where it is less than every limit's, cold
    /// pages included, and the total stays the least of them.
    #[test]
    fn the_least_supply_binds_within_the_least_total() {
        let machine = host(64 * GB, 8 * GB, 4 * GB, 6 * GB);
        assert_eq!(
            binding(machine, &[cgroup(32 * GB, 10 * GB, 0)]),
            Binding { scope: host(32 * GB, 8 * GB, 4 * GB, 6 * GB), supplier: None },
            "the host's supply binds, so a growth shows in it at once"
        );
    }

    /// The scope never flips: as the host's supply falls past the cgroup's,
    /// the total holds and the supply falls with the host's alone. Where the
    /// two supplies are equal the host's binds, as it shows a growth at once
    /// and the cgroup's only within memcg's flush.
    #[test]
    fn the_scope_holds_as_the_least_supply_moves() {
        let limit = [cgroup(32 * GB, 20 * GB, 0)];
        let readings: Vec<Binding> = [16, 13, 12, 11, 8].map(|free| binding(host(64 * GB, free * GB, 0, 0), &limit)).into();
        assert!(readings.iter().all(|binding| binding.scope.total == 32 * GB));
        assert_eq!(readings.iter().map(|binding| binding.scope.supply.total() / GB).collect::<Vec<_>>(), [12, 12, 12, 11, 8]);
        assert_eq!(
            readings.iter().map(|binding| binding.supplier).collect::<Vec<_>>(),
            [Some(0), Some(0), None, None, None],
            "the scope whose supply binds is named with it"
        );
    }

    /// A cgroup v2 usage past its `memory.high` is an overrun its supply
    /// shows, less its clean cache, not saturated to none: it binds below
    /// a host with room.
    #[test]
    fn a_cgroups_overrun_is_negative_supply() {
        let overrun = binding(host(64 * GB, 40 * GB, 0, 0), &[cgroup(32 * GB, 35 * GB, GB)]);
        let supply = Supply { free_beyond_target: -3 * GB as i64, clean_cache: GB, cold_reclaimable: 0 };
        assert_eq!(overrun, Binding { scope: Scope { total: 32 * GB, supply }, supplier: Some(0) });
        assert_eq!(supply.balance(), -2 * GB as i64);
    }

    /// A 32-bit count read across its wrap keeps rising.
    #[test]
    fn a_widened_count_rises_across_its_wrap() {
        let count = Widened::new();
        assert_eq!(count.of(u32::MAX - 1), u64::from(u32::MAX - 1));
        assert_eq!(count.of(3), u64::from(u32::MAX) + 4);
        assert_eq!(count.of(3), u64::from(u32::MAX) + 4);
    }
}
