//! macOS 15+: `sysctl hw.memsize`, the kernel's live `vm.*` counts, the
//! counts only `host_statistics64` reads, and the task's own paging, each
//! read through [`Facts`].

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use super::{read, Counted, Harm, HostSample, Lag, OwnCold, OwnReclaimed, PressureLevel, Supply, Widened, WiredFigures, WiredLimit};
use seismic_topology::{Reported, ReportedBy};
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

/// Every OS read discovery makes on macOS, its one owner. A `sysctl` name
/// and `available_parallelism` may not answer ([`read`]); the Mach calls,
/// the page size and the OS build answer on every release from macOS 15.
pub(crate) trait Facts {
    /// Every byte `sysctlbyname(name)` reports; none where the OS defines
    /// no such name on this machine (`ENOENT`: an Intel Mac's
    /// `iogpu.wired_limit_mb`), defines it but reports no value for this
    /// machine (`EINVAL`: `hw.perflevel0.l3cachesize` on an M4 Pro), or
    /// refuses the read.
    fn sysctl(&self, name: &str) -> Option<Vec<u8>>;
    /// The VM's counts (`HOST_VM_INFO64`), as `host_statistics64` serves
    /// them: fresh, or a snapshot taken within [`STATISTICS_WINDOW`].
    fn vm_statistics(&self) -> libc::vm_statistics64;
    /// This process's task's `task_vm_info`, through revision 6 at least.
    fn task_vm(&self) -> TaskVm;
    /// `sysconf(_SC_PAGESIZE)`.
    fn page(&self) -> u64;
    /// The OS version and build as Metal's owner reads it: the build ships
    /// the Metal compiler.
    fn os_build(&self) -> String;
    /// The CPUs the process may keep busy at once
    /// (`seismic_threads::quota`, their one owner: `available_parallelism`,
    /// its affinity under a cgroup's CPU quota, or one).
    fn parallelism(&self) -> NonZeroUsize;
}

/// The running OS's facts.
pub(crate) struct Os;

/// `sysctlbyname(name)` into `buffer`, or its length alone without one: the
/// bytes the OS wrote, or would write.
fn sysctl_into(name: &std::ffi::CStr, buffer: Option<&mut [u8]>) -> std::io::Result<usize> {
    let (pointer, mut length): (*mut libc::c_void, usize) = match buffer {
        Some(buffer) => (buffer.as_mut_ptr().cast(), buffer.len()),
        None => (std::ptr::null_mut(), 0),
    };
    let status = unsafe { libc::sysctlbyname(name.as_ptr(), pointer, &mut length, std::ptr::null_mut(), 0) };
    match status {
        0 => Ok(length),
        _ => Err(std::io::Error::last_os_error()),
    }
}

impl Facts for Os {
    fn sysctl(&self, name: &str) -> Option<Vec<u8>> {
        let text = std::ffi::CString::new(name).unwrap_or_else(|_| panic!("invariant violated: sysctl name {name:?} holds a NUL"));
        // `EINVAL`: a name the OS defines and reports no value of here.
        let answered = |result: std::io::Result<usize>| match result {
            Err(error) if error.raw_os_error() == Some(libc::EINVAL) => None,
            result => read(format_args!("sysctl {name}"), result),
        };
        let mut bytes = vec![0u8; answered(sysctl_into(&text, None))?];
        let length = answered(sysctl_into(&text, Some(&mut bytes)))?;
        bytes.truncate(length);
        Some(bytes)
    }

    fn vm_statistics(&self) -> libc::vm_statistics64 {
        let mut statistics: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
        let mut count = libc::HOST_VM_INFO64_COUNT;
        #[allow(deprecated)]
        let host = unsafe { libc::mach_host_self() };
        let status = unsafe {
            libc::host_statistics64(host, libc::HOST_VM_INFO64, (&mut statistics as *mut libc::vm_statistics64).cast(), &mut count)
        };
        assert_eq!(status, libc::KERN_SUCCESS, "invariant violated: host_statistics64 HOST_VM_INFO64 returned {status}");
        statistics
    }

    fn task_vm(&self) -> TaskVm {
        let mut vm = TaskVm::default();
        let mut count = TASK_VM_INFO_REV7_COUNT;
        #[allow(deprecated)]
        let task = unsafe { libc::mach_task_self() };
        let status = unsafe { libc::task_info(task, TASK_VM_INFO, (&mut vm as *mut TaskVm).cast(), &mut count) };
        assert_eq!(status, libc::KERN_SUCCESS, "invariant violated: task_info TASK_VM_INFO returned {status}");
        assert!(count >= TASK_VM_INFO_REV6_COUNT, "invariant violated: task_vm_info has {count} words, without decompressions and ledger_swapins");
        vm
    }

    fn page(&self) -> u64 {
        u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
            .ok()
            .filter(|size| *size > 0)
            .unwrap_or_else(|| panic!("invariant violated: sysconf(_SC_PAGESIZE): {}", std::io::Error::last_os_error()))
    }

    fn os_build(&self) -> String {
        seismic_metal::profile::operating_system()
    }

    fn parallelism(&self) -> NonZeroUsize {
        seismic_threads::quota()
    }
}

/// The count `sysctl` `name` reports, zero included: a 32- or 64-bit
/// integer by name and OS release.
///
/// # Panics
/// Where it reports another width.
fn sysctl_count(facts: &impl Facts, name: &str) -> Option<u64> {
    let bytes = facts.sysctl(name)?;
    Some(match *bytes.as_slice() {
        [a, b, c, d] => u64::from(u32::from_ne_bytes([a, b, c, d])),
        [a, b, c, d, e, f, g, h] => u64::from_ne_bytes([a, b, c, d, e, f, g, h]),
        _ => panic!("invariant violated: sysctl {name} returned {} bytes", bytes.len()),
    })
}

/// The count `sysctl` `name` reports, none where it reports none or zero.
fn sysctl_u64(facts: &impl Facts, name: &str) -> Option<u64> {
    sysctl_count(facts, name).filter(|value| *value > 0)
}

/// A count the OS reports on every release from macOS 15.
///
/// # Panics
/// Where it reports none.
fn sysctl_always(facts: &impl Facts, name: &str) -> u64 {
    sysctl_count(facts, name).unwrap_or_else(|| panic!("invariant violated: macOS reads no sysctl {name}"))
}

/// `sysctl`'s brand string, model and CPU counts, each the OS reports; the
/// architecture alone (`unknown;arch=<arch>`) where it reports none.
pub(super) fn cpu_identity(facts: &impl Facts) -> String {
    let fields = ["machdep.cpu.brand_string", "hw.model", "hw.ncpu", "hw.physicalcpu"]
        .into_iter()
        .filter_map(|key| Some((key, facts.sysctl(key)?)))
        .collect::<Vec<_>>();
    match fields.is_empty() {
        true => format!("unknown;arch={}", std::env::consts::ARCH),
        false => serde_json::to_string(&fields).unwrap_or_else(|error| panic!("invariant violated: sysctl fields as JSON: {error}")),
    }
}

pub(super) fn os_build(facts: &impl Facts) -> String {
    facts.os_build()
}

/// The CPU's last-level caches: the highest level each core type reports
/// (`hw.perflevel<n>.*` for each of `hw.nperflevels`), summed over the core
/// types and their instances, the L3 where the OS reports one, else the
/// L2, of `l<k>cachesize` bytes, one instance per `cpusperl<k>` of its
/// `logicalcpu`. Where a core type's cache or sharing is not reported, the
/// largest L2 any reports (a core type's or `hw.l2cachesize`); else none.
pub(super) fn last_level_cache(facts: &impl Facts) -> Option<Reported<u64>> {
    let levels = sysctl_u64(facts, "hw.nperflevels").unwrap_or(0);
    let field = |level: u64, name: &str| sysctl_u64(facts, &format!("hw.perflevel{level}.{name}"));
    let highest = |level: u64| {
        let logical = field(level, "logicalcpu")?;
        let (bytes, sharing) = [("l3cachesize", "cpusperl3"), ("l2cachesize", "cpusperl2")]
            .into_iter()
            .find_map(|(size, sharing)| Some((field(level, size)?, field(level, sharing))))?;
        Some(bytes * logical.div_ceil(sharing?))
    };
    let summed = (levels > 0).then(|| (0..levels).map(highest).sum::<Option<u64>>()).flatten();
    let bytes = summed.or_else(|| (0..levels).filter_map(|level| field(level, "l2cachesize")).chain(sysctl_u64(facts, "hw.l2cachesize")).max())?;
    Some(Reported::new(bytes, ReportedBy::MacOsSysctlCache))
}

/// `hw.memsize`: installed RAM, which every macOS reports.
///
/// # Panics
/// Where it reports none, zero, or not 8 bytes.
pub(super) fn capacity(facts: &impl Facts) -> Reported<u64> {
    let bytes = facts.sysctl("hw.memsize").unwrap_or_else(|| panic!("invariant violated: macOS reads no sysctl hw.memsize"));
    let bytes = <[u8; 8]>::try_from(bytes.as_slice())
        .unwrap_or_else(|_| panic!("invariant violated: sysctl hw.memsize returned {} bytes, not 8", bytes.len()));
    match u64::from_ne_bytes(bytes) {
        0 => panic!("invariant violated: sysctl hw.memsize reported zero bytes"),
        bytes => Reported::new(bytes, ReportedBy::MacOsMemsize),
    }
}

/// macOS reports no condition of the host for the hardware report.
pub(super) fn diagnostics(_facts: &impl Facts) -> Vec<seismic_topology::Diagnostic> {
    Vec::new()
}

/// XNU's `HOST_STATISTICS_TIME_WINDOW` (`osfmk/kern/host.c`): within a
/// window of it, once ordinary processes have asked a flavor more often than
/// the window's random allowance (2 to 10 requests), `host_statistics64`
/// serves the snapshot the latest ordinary request took. The inactive,
/// decompression and swap-in counts have no live sysctl, nor does the wired
/// count before macOS 26 (macOS 15.7.9 defines no `vm.page_wired_count`),
/// so they show the machine as late as this.
const STATISTICS_WINDOW: Duration = Duration::from_secs(1);

/// The bytes of `statistics`, zeroed before the kernel wrote it, padding
/// included: two snapshots are the same exactly where these are.
fn snapshot_bytes(statistics: &libc::vm_statistics64) -> [u8; std::mem::size_of::<libc::vm_statistics64>()] {
    // SAFETY: a plain C structure of integers, read as its bytes.
    unsafe { std::mem::transmute_copy(statistics) }
}

/// The counters' snapshot a sample read, with what it pairs with.
#[derive(Clone, Copy)]
struct Snapshot {
    statistics: libc::vm_statistics64,
    counted: Counted,
    /// This process's own paging, read as the snapshot was first served.
    own: u64,
    /// The file pages the OS keeps ([`file_floor`]), in pages.
    file_floor: u64,
}

/// The file pages the OS keeps while it compresses others' memory, as the
/// snapshot `statistics` after `before` shows them: where the OS compressed
/// more pages between the two than were free above its free-page target
/// `free_target`, it chose compressing over every file page still standing,
/// so those are its floor (a mapped library or binary in use); otherwise
/// the floor `before` showed, none before the OS first chose so. Compression
/// with more free than it compressed is the OS's own, not a press's, and
/// moves no floor.
fn file_floor(before: Option<&Snapshot>, statistics: &libc::vm_statistics64, free_target: u64) -> u64 {
    let Some(before) = before else { return 0 };
    let compressed = u64::from(statistics.compressions).saturating_sub(u64::from(before.statistics.compressions));
    let free = u64::from(statistics.free_count) + u64::from(statistics.purgeable_count);
    match compressed > free.saturating_sub(free_target) {
        true => u64::from(statistics.external_page_count),
        false => before.file_floor,
    }
}

/// The snapshot the latest sample was served, and when that sample began.
struct Served {
    snapshot: Snapshot,
    began: Instant,
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
fn snapshot(facts: &impl Facts, began: Instant, page: u64, free_target: u64) -> (Snapshot, TaskVm) {
    // A holder that panicked left the last snapshot whole: it is replaced
    // in one store.
    let mut served = SERVED.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let statistics = facts.vm_statistics();
    // Read on every sample, so its 32-bit decompressions widen across
    // every wrap; after the machine's, so a page between the two counts as
    // the process's alone, never another program's.
    let task = facts.task_vm();
    let snapshot = paired(served.take(), statistics, Instant::now(), own_paging(&task, page), free_target);
    *served = Some(Served { snapshot, began });
    (snapshot, task)
}

/// The snapshot `statistics`, served at `at` beside this process's paging
/// `own`, after the sample `before` ([`snapshot`]), its file floor judged
/// against the free-page target `free_target` ([`file_floor`]).
fn paired(before: Option<Served>, statistics: libc::vm_statistics64, at: Instant, own: u64, free_target: u64) -> Snapshot {
    match before {
        Some(Served { snapshot, .. }) if snapshot_bytes(&snapshot.statistics) == snapshot_bytes(&statistics) => snapshot,
        before => {
            let lag = before.as_ref().map_or(STATISTICS_WINDOW, |before| at.duration_since(before.began).min(STATISTICS_WINDOW));
            let file_floor = file_floor(before.as_ref().map(|before| &before.snapshot), &statistics, free_target);
            Snapshot { statistics, counted: Counted { at, lag }, own, file_floor }
        }
    }
}

/// `task_vm_info` (`TASK_VM_INFO`) through revision 7, laid out as the
/// kernel writes it and read in part.
#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct TaskVm {
    virtual_size: u64,
    region_count: i32,
    page_size: i32,
    resident_size: u64,
    resident_size_peak: u64,
    device: u64,
    device_peak: u64,
    internal: u64,
    internal_peak: u64,
    external: u64,
    external_peak: u64,
    reusable: u64,
    reusable_peak: u64,
    purgeable_volatile_pmap: u64,
    purgeable_volatile_resident: u64,
    purgeable_volatile_virtual: u64,
    compressed: u64,
    compressed_peak: u64,
    compressed_lifetime: u64,
    phys_footprint: u64,
    min_address: u64,
    max_address: u64,
    /// `ledger_phys_footprint_peak` and the twenty purgeable and tag
    /// ledgers after it.
    ledgers: [i64; 21],
    limit_bytes_remaining: u64,
    decompressions: i32,
    ledger_swapins: i64,
    ledger_tag_neural_nofootprint_total: i64,
    ledger_tag_neural_nofootprint_peak: i64,
}

const TASK_VM_INFO: libc::task_flavor_t = 22;
/// `TASK_VM_INFO_REV7_COUNT`: the structure in natural-sized words.
const TASK_VM_INFO_REV7_COUNT: libc::mach_msg_type_number_t =
    (std::mem::size_of::<TaskVm>() / std::mem::size_of::<libc::natural_t>()) as libc::mach_msg_type_number_t;
/// `TASK_VM_INFO_REV6_COUNT`: through `ledger_swapins`.
const TASK_VM_INFO_REV6_COUNT: libc::mach_msg_type_number_t = TASK_VM_INFO_REV7_COUNT - 4;

/// The process's task's decompressions, widened across their wraps.
static OWN_DECOMPRESSIONS: Widened = Widened::new();

/// This process's own paging over the harm counters' events: its pages
/// decompressed (`task_vm_info.decompressions`) and swapped in
/// (`ledger_swapins`, in bytes), as pages of `page`.
///
/// # Panics
/// Where the swap-ins read negative.
fn own_paging(vm: &TaskVm, page: u64) -> u64 {
    let swapins = u64::try_from(vm.ledger_swapins).unwrap_or_else(|_| panic!("invariant violated: task_vm_info.ledger_swapins is {}", vm.ledger_swapins));
    OWN_DECOMPRESSIONS.of(vm.decompressions as u32) + swapins / page
}

/// Free swap (`vm.swapusage`'s `xsu_avail`); none where it is not reported.
///
/// # Panics
/// Where it reports another size than its structure's.
fn swap_free(facts: &impl Facts) -> u64 {
    let Some(bytes) = facts.sysctl("vm.swapusage") else { return 0 };
    assert_eq!(bytes.len(), std::mem::size_of::<libc::xsw_usage>(), "invariant violated: sysctl vm.swapusage returned {} bytes", bytes.len());
    // SAFETY: the structure's bytes, as the OS wrote them.
    let usage: libc::xsw_usage = unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast()) };
    usage.xsu_avail
}

/// `kern.memorystatus_vm_pressure_level`: 1 normal, 2 warn, 4 critical;
/// none where it is not reported.
///
/// # Panics
/// Where it reads another level.
fn pressure_level(facts: &impl Facts) -> Option<PressureLevel> {
    Some(match sysctl_count(facts, "kern.memorystatus_vm_pressure_level")? {
        1 => PressureLevel::Normal,
        2 => PressureLevel::Warn,
        4 => PressureLevel::Critical,
        level => panic!("invariant violated: kern.memorystatus_vm_pressure_level is {level}"),
    })
}

/// The ceiling on host RAM the GPU may wire, where the GPU wires host RAM:
/// Apple silicon's IOGPU stack defines `iogpu.wired_limit_mb` (macOS 14+,
/// in MiB) or `debug.iogpu.wired_limit` (macOS 13, in bytes). The first of
/// them set (non-zero) is the limit; neither set is the OS's default
/// ([`WiredLimit::OsDefault`]). None where the OS defines neither: the GPU
/// wires nothing through IOGPU (an Intel Mac: `ENOENT` on macOS 15.7.9
/// x86_64), and the sample has no wired figures.
fn wired_limit(facts: &impl Facts) -> Option<WiredLimit> {
    let megabytes = sysctl_count(facts, "iogpu.wired_limit_mb");
    let bytes = sysctl_count(facts, "debug.iogpu.wired_limit");
    if megabytes.is_none() && bytes.is_none() {
        return None;
    }
    let set = megabytes.filter(|megabytes| *megabytes > 0).map(|megabytes| megabytes << 20).or(bytes.filter(|bytes| *bytes > 0));
    Some(set.map_or(WiredLimit::OsDefault, WiredLimit::Set))
}

/// `bytes` times `numerator / denominator`, never more than `bytes`, so it
/// fits `bytes`'s width whatever the ratio.
fn share(bytes: u64, numerator: u64, denominator: u64) -> u64 {
    (u128::from(bytes) * u128::from(numerator) / u128::from(denominator)).min(u128::from(bytes)) as u64
}

/// What `stored` compressed bytes holding `logical` bytes save of each byte
/// compressed, applied to `bytes`: none while the compressor stores nothing.
fn compressed_saving(bytes: u64, stored: u64, logical: u64) -> u64 {
    match logical {
        0 => 0,
        logical => share(bytes, logical.saturating_sub(stored), logical),
    }
}

/// What `bytes` compressed bytes occupy in a compressor storing `stored`
/// bytes for `logical` bytes, at its ratio: none while it stores nothing,
/// and never more than `bytes`.
fn compressed_stored(bytes: u64, stored: u64, logical: u64) -> u64 {
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
pub(super) fn sample(facts: &impl Facts) -> HostSample {
    let at = Instant::now();
    let page = facts.page();
    let pages = |count: u64| count.checked_mul(page).unwrap_or_else(|| panic!("invariant violated: {count} pages overflow bytes"));
    let total = capacity(facts).value;
    let free = sysctl_always(facts, "vm.page_free_count");
    let purgeable = sysctl_count(facts, "vm.page_purgeable_count").unwrap_or(0);
    let free_target = sysctl_always(facts, "vm.vm_page_free_target");
    let wired_limit = wired_limit(facts);
    let swap = swap_free(facts);
    let pressure = pressure_level(facts);
    let (Snapshot { statistics, counted, own, file_floor }, task) = snapshot(facts, at, page, free_target);
    // Free and volatile purgeable pages (purged at no one's cost) less the
    // free-page target, which the pageout daemon refills by compressing,
    // swapping and evicting other pages: pages taken below it are never free
    // without displacing someone, and a fall below it is a fall all the
    // same.
    let free_beyond_target = super::beyond(pages(free + purgeable), pages(free_target));
    // The inactive queue is not split by kind: at least its pages that no
    // file backs, both counts of the one snapshot.
    let cold = pages(u64::from(statistics.inactive_count).saturating_sub(u64::from(statistics.external_page_count)));
    // The file pages above the floor the OS keeps: a press takes them
    // before the OS compresses anyone's memory.
    let clean_cache = pages(u64::from(statistics.external_page_count).saturating_sub(file_floor));
    let stored = pages(u64::from(statistics.compressor_page_count));
    let logical = pages(statistics.total_uncompressed_pages_in_compressor);
    HostSample {
        at,
        page_bytes: page,
        total,
        supply: Supply {
            free_beyond_target,
            clean_cache,
            cold_reclaimable: cold.min(compressed_saving(cold, stored, logical).saturating_add(swap)),
        },
        counted,
        cold,
        own_cold: OwnCold::Unobserved,
        own_reclaimed: Some(OwnReclaimed { bytes: task.compressed, stored: compressed_stored(task.compressed, stored, logical), exact: true }),
        wired: wired_limit.map(|limit| WiredFigures { limit, wire_count: pages(u64::from(statistics.wire_count)) }),
        harm: Some(Harm { machine: statistics.decompressions.saturating_add(statistics.swapins), own }),
        pressure,
        // A release of anonymous memory returns its pages in the call, and
        // a growth counts at its first touch.
        lag: Lag { release: Duration::ZERO, growth: Duration::ZERO },
    }
}

/// One memory node of every core: per performance level (`hw.nperflevels`,
/// level 0 the performance cores and every later level efficiency cores, as
/// many levels as the OS reports), its `physicalcpu` cores of
/// `logicalcpu / physicalcpu` threads each. Where the levels are not
/// reported, every core is one performance level: the `hw.physicalcpu`
/// cores of `hw.logicalcpu` threads, or, where those are not reported
/// either, the flat topology of the quota's CPUs
/// (`crate::processors::flat`). A CPU is its ordinal,
/// performance cores first, and the process may run on every one
/// ([`crate::ProcessorsReportedBy::MacOsSysctl`]).
///
/// # Panics
/// Where a level's threads do not divide over its cores.
pub(super) fn processors(facts: &impl Facts) -> crate::Processors {
    let reported = |name: &str| sysctl_u64(facts, name);
    let levels = reported("hw.nperflevels")
        .and_then(|levels| {
            (0..levels)
                .map(|level| {
                    let field = |name: &str| reported(&format!("hw.perflevel{level}.{name}"));
                    Some((field("physicalcpu")?, field("logicalcpu")?, level == 0))
                })
                .collect::<Option<Vec<_>>>()
        })
        .or_else(|| Some(vec![(reported("hw.physicalcpu")?, reported("hw.logicalcpu")?, true)]));
    let (nodes, every, by) = match levels {
        Some(levels) => {
            let mut cores = Vec::new();
            let mut next = 0usize;
            for (physical, logical, performance) in levels {
                assert!(logical % physical == 0, "invariant violated: {logical} logical CPUs over {physical} cores");
                let threads = usize::try_from(logical / physical).unwrap_or_else(|_| panic!("invariant violated: {logical} logical CPUs"));
                for _ in 0..physical {
                    cores.push(crate::Core { threads: (next..next + threads).collect(), performance });
                    next += threads;
                }
            }
            (vec![crate::MemoryNode { id: 0, memory: true, cores }], (0..next).collect(), crate::ProcessorsReportedBy::MacOsSysctl)
        }
        None => {
            let every = (0..facts.parallelism().get()).collect();
            (crate::processors::flat(&every), every, crate::ProcessorsReportedBy::Flat)
        }
    };
    crate::Processors::new(nodes, every, facts.parallelism(), by).unwrap_or_else(|error| panic!("invariant violated: the processors the OS reports: {error}"))
}

/// macOS places no thread on a core it names: its scheduler places every
/// thread, so nothing is placed.
pub(super) fn place_process(_cpus: &[usize]) -> Result<(), String> {
    Ok(())
}

/// macOS places no thread on a core it names: its scheduler places every
/// thread, so nothing is placed.
pub(super) fn pin(_cpus: &[usize]) -> Result<(), String> {
    Ok(())
}

/// macOS reads no device's memory node.
pub(super) fn device_node(_facts: &impl Facts, _location: crate::PciLocation) -> crate::DeviceNode {
    crate::DeviceNode::Unknown
}

/// The machine's names (`crate::machine`): Apple and `hw.model` (an opaque
/// identifier such as `Mac16,5`), and `system_profiler`'s marketing name
/// (`SPHardwareDataType`'s `machine_name`, such as "MacBook Pro"). macOS
/// reports no enclosure type. A field it does not report reads as nothing.
pub(super) fn machine() -> crate::machine::Firmware {
    let name = std::process::Command::new("/usr/sbin/system_profiler")
        .args(["SPHardwareDataType", "-json"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| serde_json::from_slice::<serde_json::Value>(&output.stdout).ok())
        .and_then(|document| {
            let name = document.get("SPHardwareDataType")?.as_array()?.first()?.get("machine_name")?.as_str()?;
            Some(name.as_bytes().to_vec())
        });
    crate::machine::Firmware {
        manufacturer: b"Apple".to_vec(),
        model: Os.sysctl("hw.model").unwrap_or_default(),
        name: name.unwrap_or_default(),
        ..crate::machine::Firmware::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    const GB: u64 = 1 << 30;

    /// A Mac as a test states it: the sysctls it reports, every other name
    /// unreported. Its Mach counts are zero: a sample reads them of the
    /// running OS alone.
    struct Mac {
        sysctls: BTreeMap<String, Vec<u8>>,
        parallelism: NonZeroUsize,
    }

    impl Mac {
        /// A Mac of 64 GB and a quota of one CPU, reporting nothing else.
        fn new() -> Self {
            Self { sysctls: BTreeMap::new(), parallelism: NonZeroUsize::MIN }.count("hw.memsize", 64 * GB)
        }

        /// The Mac with `name` reporting `value`, a 64-bit count.
        fn count(mut self, name: &str, value: u64) -> Self {
            self.sysctls.insert(name.into(), value.to_ne_bytes().to_vec());
            self
        }

        /// The Mac with `counts` reported, each `(name, value)`.
        fn counts(self, counts: &[(&str, u64)]) -> Self {
            counts.iter().fold(self, |mac, (name, value)| mac.count(name, *value))
        }
    }

    impl Facts for Mac {
        fn sysctl(&self, name: &str) -> Option<Vec<u8>> {
            self.sysctls.get(name).cloned()
        }
        fn vm_statistics(&self) -> libc::vm_statistics64 {
            unsafe { std::mem::zeroed() }
        }
        fn task_vm(&self) -> TaskVm {
            TaskVm::default()
        }
        fn page(&self) -> u64 {
            16 << 10
        }
        fn os_build(&self) -> String {
            "macOS 15.7".into()
        }
        fn parallelism(&self) -> NonZeroUsize {
            self.parallelism
        }
    }

    /// A compressor storing 1 GB in 4 GB saves three quarters of what it
    /// compresses; one storing nothing is not counted to save anything.
    #[test]
    fn the_compressor_saves_what_it_does_not_store() {
        assert_eq!(compressed_saving(8 << 30, 1 << 30, 4 << 30), 6 << 30);
        assert_eq!(compressed_saving(8 << 30, 0, 0), 0);
        assert_eq!(compressed_stored(8 << 30, 1 << 30, 4 << 30), 2 << 30);
        assert_eq!(compressed_stored(8 << 30, 0, 0), 0);
    }

    /// A snapshot served again keeps the instant and the paging it was
    /// first read with; a new one counts from when the sample before began,
    /// never more than the window before it, with the paging read beside it.
    #[test]
    fn a_snapshot_pairs_with_the_paging_read_when_it_was_first_served() {
        let statistics = |decompressions| libc::vm_statistics64 { decompressions, ..unsafe { std::mem::zeroed() } };
        let start = Instant::now();
        let first = paired(None, statistics(10), start, 3, 0);
        assert_eq!((first.counted, first.own), (Counted { at: start, lag: STATISTICS_WINDOW }, 3));
        let again = paired(Some(Served { snapshot: first, began: start }), statistics(10), start + Duration::from_millis(40), 5, 0);
        assert_eq!((again.counted, again.own), (Counted { at: start, lag: STATISTICS_WINDOW }, 3));
        let began = start + Duration::from_millis(40);
        let fresh = paired(Some(Served { snapshot: again, began }), statistics(12), start + Duration::from_millis(70), 6, 0);
        assert_eq!((fresh.counted, fresh.own), (Counted { at: start + Duration::from_millis(70), lag: Duration::from_millis(30) }, 6));
        let late = paired(Some(Served { snapshot: fresh, began }), statistics(13), start + Duration::from_secs(5), 7, 0);
        assert_eq!(late.counted, Counted { at: start + Duration::from_secs(5), lag: STATISTICS_WINDOW });
    }

    /// The file floor is the file pages standing where the OS compressed
    /// more than was free above its target, kept until it next does so;
    /// compression beside more free memory moves none, and none stands
    /// before the OS first chose to compress.
    #[test]
    fn the_file_floor_is_what_the_os_keeps_as_it_compresses() {
        let statistics = |compressions, free, external| libc::vm_statistics64 {
            compressions,
            free_count: free,
            external_page_count: external,
            ..unsafe { std::mem::zeroed() }
        };
        let snapshot = |statistics, file_floor| Snapshot { statistics, counted: Counted { at: Instant::now(), lag: STATISTICS_WINDOW }, own: 0, file_floor };
        let target = 4000;
        assert_eq!(file_floor(None, &statistics(0, 100_000, 120_000), target), 0);
        let rest = snapshot(statistics(10, 100_000, 120_000), 0);
        assert_eq!(file_floor(Some(&rest), &statistics(5_000, 100_000, 120_000), target), 0);
        let pressed = statistics(30_000, 4_500, 95_000);
        assert_eq!(file_floor(Some(&rest), &pressed, target), 95_000);
        let kept = snapshot(pressed, 95_000);
        assert_eq!(file_floor(Some(&kept), &statistics(30_000, 60_000, 140_000), target), 95_000);
        assert_eq!(file_floor(Some(&kept), &statistics(60_000, 4_200, 88_000), target), 88_000);
    }

    /// `iogpu.wired_limit_mb` of zero is the OS's default, which the ledger
    /// reads as the device's working set; a limit set is read in bytes.
    #[test]
    fn a_wired_limit_of_zero_is_the_os_default() {
        assert_eq!(wired_limit(&Mac::new().count("iogpu.wired_limit_mb", 0)), Some(WiredLimit::OsDefault));
        assert_eq!(wired_limit(&Mac::new().count("iogpu.wired_limit_mb", 40_960)), Some(WiredLimit::Set(40 * GB)));
    }

    /// macOS 13's `debug.iogpu.wired_limit`, in bytes, is the limit where
    /// `iogpu.wired_limit_mb` is not defined; zero is the OS's default.
    #[test]
    fn macos_13s_wired_limit_is_read_in_bytes() {
        assert_eq!(wired_limit(&Mac::new().count("debug.iogpu.wired_limit", 36 * GB)), Some(WiredLimit::Set(36 * GB)));
        assert_eq!(wired_limit(&Mac::new().count("debug.iogpu.wired_limit", 0)), Some(WiredLimit::OsDefault));
    }

    /// A Mac reporting none of the optional sysctls (an Intel Mac defines
    /// no IOGPU sysctl) reads each one's documented value: no wired
    /// figures, free swap or pressure level. (A sample is read of the
    /// running OS alone: its snapshot and the task's widened count are the
    /// process's.)
    #[test]
    fn the_optional_sysctls_unreported_read_their_documented_values() {
        let mac = Mac::new();
        assert_eq!(wired_limit(&mac), None, "no IOGPU stack wires host RAM");
        assert_eq!(swap_free(&mac), 0);
        assert_eq!(pressure_level(&mac), None);
        assert_eq!(capacity(&mac).value, 64 * GB);
    }

    /// Without performance levels every core is one performance level: the
    /// physical cores, or each CPU of the quota.
    #[test]
    fn without_performance_levels_every_core_is_one_performance_level() {
        let intel = processors(&Mac::new().counts(&[("hw.physicalcpu", 6), ("hw.logicalcpu", 12)]));
        assert_eq!((intel.by(), intel.performance_cores()), (crate::ProcessorsReportedBy::MacOsSysctl, 6));
        assert!(intel.nodes()[0].cores.iter().all(|core| core.threads.len() == 2));
        let counted = processors(&Mac { parallelism: NonZeroUsize::new(4).unwrap(), ..Mac::new() });
        assert_eq!((counted.by(), counted.performance_cores()), (crate::ProcessorsReportedBy::Flat, 4));
        assert_eq!(processors(&Mac::new()).performance_cores(), 1);
    }

    /// Three performance levels (an Apple M6 reports `hw.nperflevels` 3):
    /// level 0 is the performance cores, the rest efficiency cores.
    #[test]
    fn of_three_performance_levels_the_first_is_the_performance_cores() {
        let mac = Mac::new().counts(&[
            ("hw.nperflevels", 3),
            ("hw.perflevel0.physicalcpu", 4),
            ("hw.perflevel0.logicalcpu", 4),
            ("hw.perflevel1.physicalcpu", 6),
            ("hw.perflevel1.logicalcpu", 6),
            ("hw.perflevel2.physicalcpu", 2),
            ("hw.perflevel2.logicalcpu", 2),
        ]);
        let processors = processors(&mac);
        assert_eq!(processors.nodes()[0].cores.len(), 12);
        assert_eq!(processors.performance_cores(), 4);
        assert!(processors.nodes()[0].cores[..4].iter().all(|core| core.performance));
    }

    /// The last-level caches are the highest level each core type reports,
    /// over its instances; where a core type's sharing is not reported, the
    /// largest L2 reported; with no cache reported, none.
    #[test]
    fn the_last_level_cache_falls_back_to_the_largest_l2() {
        let levels = Mac::new().counts(&[
            ("hw.nperflevels", 2),
            ("hw.perflevel0.logicalcpu", 10),
            ("hw.perflevel0.l2cachesize", 16 << 20),
            ("hw.perflevel0.cpusperl2", 5),
            ("hw.perflevel1.logicalcpu", 4),
            ("hw.perflevel1.l2cachesize", 4 << 20),
        ]);
        assert_eq!(last_level_cache(&levels).map(|cache| cache.value), Some(16 << 20), "no sharing for level 1: the largest L2");
        let shared = levels.count("hw.perflevel1.cpusperl2", 4);
        assert_eq!(last_level_cache(&shared).map(|cache| cache.value), Some((2 * 16 + 4) << 20));
        assert_eq!(last_level_cache(&Mac::new().count("hw.l2cachesize", 256 << 10)).map(|cache| cache.value), Some(256 << 10));
        assert_eq!(last_level_cache(&Mac::new()), None);
    }

    /// A Mac reporting none of the CPU's identifying sysctls is identified
    /// by its architecture alone.
    #[test]
    fn a_cpu_reporting_no_identity_is_its_architecture() {
        assert_eq!(cpu_identity(&Mac::new()), format!("unknown;arch={}", std::env::consts::ARCH));
    }

    /// The machine's sample reads every figure, within the machine.
    #[test]
    fn the_host_sample_reads_within_the_machine() {
        let sample = sample(&Os);
        assert!(sample.supply.total() <= sample.total);
        assert!(sample.supply.cold_reclaimable <= sample.cold && sample.cold <= sample.total);
        if let Some(wired) = sample.wired {
            assert!(wired.wire_count > 0 && wired.wire_count < sample.total);
        }
        let harm = sample.harm.expect("macOS counts the task's own paging");
        assert!(harm.own <= harm.machine);
        assert!(sample.own_reclaimed.is_some_and(|own| own.bytes < sample.total && own.stored <= own.bytes && own.exact), "macOS reports the task's compressed bytes");
        assert!(sample.counted.at <= Instant::now() && sample.counted.lag <= STATISTICS_WINDOW);
    }
}
