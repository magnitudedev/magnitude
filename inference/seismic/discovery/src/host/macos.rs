//! macOS 15+: `sysctl hw.memsize`, the kernel's live `vm.*` counts, the
//! counts only `host_statistics64` reads, and the task's own paging.

use super::{Counted, Harm, HostSample, Lag, OwnCold, OwnReclaimed, PressureLevel, Supply, Widened, WiredFigures, WiredLimit};
use seismic_topology::{Reported, ReportedBy};
use std::io;
use std::time::{Duration, Instant};

/// `sysctlbyname(name)` into `buffer`, or its length alone without one: the
/// bytes the OS wrote, or would write.
fn sysctl(name: &str, buffer: Option<&mut [u8]>) -> io::Result<usize> {
    let name =
        std::ffi::CString::new(name).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    let (pointer, mut length): (*mut libc::c_void, usize) = match buffer {
        Some(buffer) => (buffer.as_mut_ptr().cast(), buffer.len()),
        None => (std::ptr::null_mut(), 0),
    };
    let status =
        unsafe { libc::sysctlbyname(name.as_ptr(), pointer, &mut length, std::ptr::null_mut(), 0) };
    if status != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(length)
}

fn failed(name: &str, error: io::Error) -> String {
    format!("sysctl {name}: {error}")
}

/// Every byte `sysctl` `name` reports, whatever its length.
fn sysctl_bytes(name: &str) -> Result<Vec<u8>, String> {
    let length = sysctl(name, None).map_err(|error| failed(name, error))?;
    let mut bytes = vec![0u8; length];
    let length = sysctl(name, Some(&mut bytes)).map_err(|error| failed(name, error))?;
    bytes.truncate(length);
    Ok(bytes)
}

/// The value `sysctl` `name` reports, which is exactly `N` bytes long.
fn sysctl_exact<const N: usize>(name: &str) -> Result<[u8; N], String> {
    let mut bytes = [0u8; N];
    let length = sysctl(name, Some(&mut bytes)).map_err(|error| failed(name, error))?;
    if length != N {
        return Err(format!("sysctl {name} returned {length} bytes, not {N}"));
    }
    Ok(bytes)
}

/// The integer `sysctl` `name` reports, none where the OS defines no such
/// name, reports no value for it, or reports zero; any other failure is an
/// error.
fn sysctl_u64(name: &str) -> Result<Option<u64>, String> {
    let mut bytes = [0u8; 8];
    let length = match sysctl(name, Some(&mut bytes)) {
        Ok(length) => length,
        // `ENOENT`: the OS defines no such name. `EINVAL`: it defines the
        // name but reports no value for this machine, as macOS 15 does for
        // `hw.perflevel0.l3cachesize` and `hw.l3cachesize` on an M4 Pro
        // (read on an M4 Pro, where `hw.nonexistent` is `ENOENT`), though
        // the buffer holds any 32- or 64-bit value.
        Err(error) if matches!(error.raw_os_error(), Some(libc::ENOENT | libc::EINVAL)) => {
            return Ok(None);
        }
        Err(error) => return Err(failed(name, error)),
    };
    // The values are 32- or 64-bit integers by name and OS release.
    let value = match length {
        4 => u64::from(u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
        8 => u64::from_ne_bytes(bytes),
        _ => return Err(format!("sysctl {name} returned {length} bytes")),
    };
    Ok((value > 0).then_some(value))
}

pub(super) fn cpu_identity() -> Result<String, String> {
    let mut fields = Vec::new();
    for key in [
        "machdep.cpu.brand_string",
        "hw.model",
        "hw.ncpu",
        "hw.physicalcpu",
    ] {
        fields.push((key, sysctl_bytes(key)?));
    }
    serde_json::to_string(&fields).map_err(|error| error.to_string())
}

/// The OS version and build as Metal's owner reads it: the build ships the
/// Metal compiler.
pub(super) fn os_build() -> Result<String, String> {
    Ok(seismic_metal::profile::operating_system())
}

/// The last-level caches of every core type (`hw.perflevel<n>.*` for each
/// of `hw.nperflevels`), summed over their instances: per core type, the
/// L3 where the OS reports one, else the L2, of `l<k>cachesize` bytes, one
/// instance per `cpusperl<k>` of its `logicalcpu`. A core type whose cache
/// or sharing the OS reports none of leaves the caches unknown; any other
/// failure is an error.
pub(super) fn last_level_cache() -> Result<Option<Reported<u64>>, String> {
    let Some(levels) = sysctl_u64("hw.nperflevels")? else {
        return Ok(None);
    };
    let mut total = 0_u64;
    for level in 0..levels {
        let field = |name: &str| sysctl_u64(&format!("hw.perflevel{level}.{name}"));
        let Some(logical) = field("logicalcpu")? else {
            return Ok(None);
        };
        let mut cache = None;
        for (size, sharing) in [("l3cachesize", "cpusperl3"), ("l2cachesize", "cpusperl2")] {
            if let Some(bytes) = field(size)? {
                cache = Some((bytes, field(sharing)?));
                break;
            }
        }
        let Some((bytes, Some(sharing))) = cache else {
            return Ok(None);
        };
        total += bytes * logical.div_ceil(sharing);
    }
    Ok(Some(Reported::new(total, ReportedBy::MacOsSysctlCache)))
}

pub(super) fn capacity() -> Result<Reported<u64>, String> {
    let bytes = u64::from_ne_bytes(sysctl_exact("hw.memsize")?);
    if bytes == 0 {
        return Err("sysctl hw.memsize reported zero bytes".into());
    }
    Ok(Reported::new(bytes, ReportedBy::MacOsMemsize))
}

/// XNU's `HOST_STATISTICS_TIME_WINDOW` (`osfmk/kern/host.c`): within a
/// window of it, once ordinary processes have asked a flavor more often than
/// the window's random allowance (2 to 10 requests), `host_statistics64`
/// serves the snapshot the latest ordinary request took. The inactive,
/// decompression and swap-in counts have no live sysctl, nor does the wired
/// count before macOS 26 (macOS 15.7.9 defines no `vm.page_wired_count`),
/// so they show the machine as late as this.
const STATISTICS_WINDOW: Duration = Duration::from_secs(1);

/// The count `sysctl` `name` reports, zero included: a 32- or 64-bit
/// integer by name and OS release.
///
/// # Panics
/// Where the read fails: the reader's names are each release's from 15 on,
/// so one missing is the invariant's violation, never a zero or another
/// source's figure.
fn sysctl_count(name: &str) -> u64 {
    let mut bytes = [0u8; 8];
    let length = sysctl(name, Some(&mut bytes)).unwrap_or_else(|error| {
        let version = sysctl_bytes("kern.osproductversion")
            .map(|bytes| String::from_utf8_lossy(&bytes).trim_end_matches('\0').to_owned())
            .unwrap_or_else(|error| format!("of unknown version ({error})"));
        panic!("invariant violated: macOS {version} reads no sysctl {name}: {error}")
    });
    match length {
        4 => u64::from(u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
        8 => u64::from_ne_bytes(bytes),
        _ => panic!("invariant violated: sysctl {name} returned {length} bytes"),
    }
}

/// The VM's counts (`HOST_VM_INFO64`), as `host_statistics64` serves them:
/// fresh, or a snapshot taken within [`STATISTICS_WINDOW`].
fn vm_statistics() -> libc::vm_statistics64 {
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

/// The bytes of `statistics`, zeroed before the kernel wrote it, padding
/// included: two snapshots are the same exactly where these are.
fn snapshot_bytes(statistics: &libc::vm_statistics64) -> [u8; std::mem::size_of::<libc::vm_statistics64>()] {
    // SAFETY: a plain C structure of integers, read as its bytes.
    unsafe { std::mem::transmute_copy(statistics) }
}

/// The counters' snapshot a sample read, with what it pairs with.
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
/// sample that began at `began`, with the task's figures read beside it
/// ([`task_vm`]), live.
///
/// A snapshot differing from the one the sample before was served was taken
/// after that sample began (the kernel serves the latest one taken), within
/// [`STATISTICS_WINDOW`] of now, and before now: its counters count every
/// event before the later of the two bounds, none after now, and this
/// process's paging is read just before now, beside it. One the same as the sample
/// before's is that snapshot, served again, or the machine's counts
/// unchanged since, this process's paging among them: it keeps the instant
/// and the paging it was first read with.
fn snapshot(began: Instant, page: u64, free_target: u64) -> (std::sync::MutexGuard<'static, Option<Served>>, TaskVm) {
    let mut served = SERVED.lock().expect("the served snapshot's lock is never poisoned");
    let statistics = vm_statistics();
    // Read on every sample, so its 32-bit decompressions widen across
    // every wrap; after the machine's, so a page between the two counts as
    // the process's alone, never another program's.
    let task = task_vm();
    let own = own_paging(&task, page);
    let at = Instant::now();
    let snapshot = paired(served.take(), statistics, at, own, free_target);
    *served = Some(Served { snapshot, began });
    (served, task)
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
#[derive(Default)]
struct TaskVm {
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

/// This process's task's `task_vm_info` now.
///
/// # Panics
/// Where the kernel reports no revision-6 `task_vm_info`, which every
/// release from macOS 15 does.
fn task_vm() -> TaskVm {
    let mut vm = TaskVm::default();
    let mut count = TASK_VM_INFO_REV7_COUNT;
    #[allow(deprecated)]
    let task = unsafe { libc::mach_task_self() };
    let status = unsafe { libc::task_info(task, TASK_VM_INFO, (&mut vm as *mut TaskVm).cast(), &mut count) };
    assert_eq!(status, libc::KERN_SUCCESS, "invariant violated: task_info TASK_VM_INFO returned {status}");
    assert!(
        count >= TASK_VM_INFO_REV6_COUNT,
        "invariant violated: task_vm_info has {count} words, without decompressions and ledger_swapins"
    );
    vm
}

/// This process's own paging over the harm counters' events: its pages
/// decompressed (`task_vm_info.decompressions`) and swapped in
/// (`ledger_swapins`, in bytes), as pages of `page`.
fn own_paging(vm: &TaskVm, page: u64) -> u64 {
    let swapins = u64::try_from(vm.ledger_swapins).expect("invariant violated: task_vm_info.ledger_swapins is negative");
    OWN_DECOMPRESSIONS.of(vm.decompressions as u32) + swapins / page
}

/// Free swap (`vm.swapusage`'s `xsu_avail`).
fn swap_free() -> u64 {
    let mut usage: libc::xsw_usage = unsafe { std::mem::zeroed() };
    let bytes = unsafe { std::slice::from_raw_parts_mut((&mut usage as *mut libc::xsw_usage).cast::<u8>(), std::mem::size_of::<libc::xsw_usage>()) };
    let length = sysctl("vm.swapusage", Some(bytes)).unwrap_or_else(|error| panic!("invariant violated: sysctl vm.swapusage: {error}"));
    assert_eq!(length, std::mem::size_of::<libc::xsw_usage>(), "invariant violated: sysctl vm.swapusage returned {length} bytes");
    usage.xsu_avail
}

/// `kern.memorystatus_vm_pressure_level`: 1 normal, 2 warn, 4 critical.
fn pressure_level() -> PressureLevel {
    match sysctl_count("kern.memorystatus_vm_pressure_level") {
        1 => PressureLevel::Normal,
        2 => PressureLevel::Warn,
        4 => PressureLevel::Critical,
        level => panic!("invariant violated: kern.memorystatus_vm_pressure_level is {level}"),
    }
}

/// What `stored` compressed bytes holding `logical` bytes save of each byte
/// compressed, applied to `bytes`: none while the compressor stores nothing.
fn compressed_saving(bytes: u64, stored: u64, logical: u64) -> u64 {
    match logical {
        0 => 0,
        logical => u64::try_from(u128::from(bytes) * u128::from(logical.saturating_sub(stored)) / u128::from(logical))
            .expect("a share of a byte count fits u64"),
    }
}

/// What `bytes` compressed bytes occupy in a compressor storing `stored`
/// bytes for `logical` bytes, at its ratio: none while it stores nothing,
/// and never more than `bytes`.
fn compressed_stored(bytes: u64, stored: u64, logical: u64) -> u64 {
    match logical {
        0 => 0,
        logical => u64::try_from(u128::from(bytes) * u128::from(stored) / u128::from(logical))
            .expect("a share of a byte count fits u64")
            .min(bytes),
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
pub(super) fn sample() -> HostSample {
    let at = Instant::now();
    let page = u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
        .ok()
        .filter(|size| *size > 0)
        .expect("invariant violated: sysconf(_SC_PAGESIZE) reads no page size");
    let pages = |count: u64| count.checked_mul(page).expect("invariant violated: a page count overflows bytes");
    let total = capacity().unwrap_or_else(|error| panic!("invariant violated: {error}")).value;
    let free = sysctl_count("vm.page_free_count");
    let purgeable = sysctl_count("vm.page_purgeable_count");
    let free_target = sysctl_count("vm.vm_page_free_target");
    let wired_limit = match sysctl_count("iogpu.wired_limit_mb") {
        0 => WiredLimit::OsDefault,
        megabytes => WiredLimit::Set(megabytes << 20),
    };
    let swap = swap_free();
    let pressure = pressure_level();
    let (served, task) = snapshot(at, page, free_target);
    let Snapshot { statistics, counted, own, file_floor } = &served.as_ref().expect("a sample was just served").snapshot;
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
    let clean_cache = pages(u64::from(statistics.external_page_count).saturating_sub(*file_floor));
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
        counted: *counted,
        cold,
        own_cold: OwnCold::Unobserved,
        own_reclaimed: Some(OwnReclaimed { bytes: task.compressed, stored: compressed_stored(task.compressed, stored, logical), exact: true }),
        wired: Some(WiredFigures { limit: wired_limit, wire_count: pages(u64::from(statistics.wire_count)) }),
        harm: Some(Harm { machine: statistics.decompressions.saturating_add(statistics.swapins), own: *own }),
        pressure: Some(pressure),
        // A release of anonymous memory returns its pages in the call, and
        // a growth counts at its first touch.
        lag: Lag { release: Duration::ZERO, growth: Duration::ZERO },
    }
}

#[cfg(test)]
mod memory_tests {
    use super::*;

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

    /// The machine's sample reads every figure, within the machine.
    #[test]
    fn the_host_sample_reads_within_the_machine() {
        let sample = sample();
        assert!(sample.supply.total() <= sample.total);
        assert!(sample.supply.cold_reclaimable <= sample.cold && sample.cold <= sample.total);
        let wired = sample.wired.expect("macOS reads its wired figures");
        assert!(wired.wire_count > 0 && wired.wire_count < sample.total);
        let harm = sample.harm.expect("macOS counts the task's own paging");
        assert!(harm.own <= harm.machine);
        assert!(sample.own_reclaimed.is_some_and(|own| own.bytes < sample.total && own.stored <= own.bytes && own.exact), "macOS reports the task's compressed bytes");
        assert!(sample.counted.at <= Instant::now() && sample.counted.lag <= STATISTICS_WINDOW);
    }
}

/// One memory node of every core: per performance level (`hw.nperflevels`,
/// level 0 the performance cores), its `physicalcpu` cores of
/// `logicalcpu / physicalcpu` threads each; with no levels (Intel), the
/// `hw.physicalcpu` cores of `hw.logicalcpu` threads, every one a
/// performance core. A CPU is its ordinal, performance cores first, and the
/// process may run on every one ([`crate::ProcessorsReportedBy::MacOsSysctl`]).
pub(super) fn processors() -> Result<crate::Processors, String> {
    let reported =
        |name: &str| sysctl_u64(name)?.ok_or_else(|| format!("sysctl {name} reports no value"));
    let levels = match sysctl_u64("hw.nperflevels")? {
        Some(levels) => (0..levels)
            .map(|level| {
                let field = |name: &str| reported(&format!("hw.perflevel{level}.{name}"));
                Ok((field("physicalcpu")?, field("logicalcpu")?, level == 0))
            })
            .collect::<Result<Vec<_>, String>>()?,
        None => vec![(
            reported("hw.physicalcpu")?,
            reported("hw.logicalcpu")?,
            true,
        )],
    };
    let mut cores = Vec::new();
    let mut next = 0usize;
    for (physical, logical, performance) in levels {
        if logical % physical != 0 {
            return Err(format!("{logical} logical CPUs over {physical} cores"));
        }
        let threads = usize::try_from(logical / physical).map_err(|error| error.to_string())?;
        for _ in 0..physical {
            cores.push(crate::Core {
                threads: (next..next + threads).collect(),
                performance,
            });
            next += threads;
        }
    }
    crate::Processors::new(
        vec![crate::MemoryNode {
            id: 0,
            memory: true,
            cores,
        }],
        (0..next).collect(),
        crate::ProcessorsReportedBy::MacOsSysctl,
    )
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
pub(super) fn device_node(_location: crate::PciLocation) -> Result<crate::DeviceNode, String> {
    Ok(crate::DeviceNode::Unknown)
}
