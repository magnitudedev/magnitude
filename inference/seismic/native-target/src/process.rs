//! What the process holds, by what holds it, and the one place freed pages
//! are returned to the system.
//!
//! A device's ledger counts what Seismic charged. The process also holds
//! what no ledger counts: its own heap, what the allocator keeps of freed
//! heap, and what a toolchain or a device framework keeps for itself. A
//! bound on what a load holds is a bound on all of it, so all of it is read
//! here, by the parts the system reports.

/// One reading of the process's memory. A part the system does not report
/// is none.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProcessMemory {
    /// The physical footprint: what memory pressure and the peak-footprint
    /// accounting count.
    pub footprint_bytes: Option<u64>,
    /// Heap blocks in use: what the process's own code and the frameworks
    /// it calls have allocated and not freed, by the sizes allocated. Pages
    /// of a block that were never written are not footprint, so this can
    /// exceed the footprint (447 MB of footprint held 518 MB of live blocks
    /// at the start of one load's tuning): its changes say what the heap
    /// took and gave back, not what the footprint did.
    pub malloc_live_bytes: Option<u64>,
    /// Footprint the system's ledger tags as graphics. A Metal buffer is
    /// tagged so until it joins a residency set; the buffers Seismic
    /// allocates are wired through one and are not (a load holding 2.8 GB
    /// on the device read zero here), so on Metal the device's allocations
    /// are footprint that no part of this reading separates from the rest.
    pub graphics_bytes: Option<u64>,
    /// Resident pages of mapped files: outside the footprint, and reclaimed
    /// by the system without swapping.
    pub mapped_bytes: Option<u64>,
}

impl ProcessMemory {
    /// The footprint neither live heap nor graphics accounts for: freed heap
    /// the allocator has not returned, stacks and page tables, and on Metal
    /// the device's wired allocations and what the framework keeps.
    pub fn rest_bytes(&self) -> Option<u64> {
        Some(
            self.footprint_bytes?
                .saturating_sub(self.malloc_live_bytes?)
                .saturating_sub(self.graphics_bytes?),
        )
    }

    /// The footprint now, read without walking the heap. Where the system
    /// keeps no high-water mark of the footprint, every reading is also a
    /// sample of it ([`ProcessMemory::footprint_peak`]).
    pub fn footprint_now() -> Option<u64> {
        #[cfg(target_os = "macos")]
        {
            Some(macos::task_vm_info()?.phys_footprint)
        }
        #[cfg(target_os = "linux")]
        {
            let footprint = linux::footprint_bytes()?;
            SAMPLED_PEAK.fetch_max(footprint, std::sync::atomic::Ordering::Relaxed);
            Some(footprint)
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        None
    }

    /// Whether [`ProcessMemory::footprint_peak`] is the most of the
    /// readings this process took of its footprint, not a mark the system
    /// keeps: what the footprint was between two readings is then in no
    /// peak. True on Linux, whose one high-water mark (`VmHWM`) is of the
    /// whole resident set, resident pages of mapped files included, which
    /// the footprint is not.
    pub const PEAK_IS_SAMPLED: bool = cfg!(not(target_os = "macos"));

    /// The most the footprint has been in the life of the process, in the
    /// footprint's own quantity: on macOS the system's mark; on Linux the
    /// most of every reading taken ([`ProcessMemory::footprint_now`]:
    /// around each piece of a search, when a worker thread ends a job and
    /// when a compile holds its program and what it formed), because the
    /// kernel keeps no mark of anonymous memory alone
    /// ([`ProcessMemory::PEAK_IS_SAMPLED`]).
    pub fn footprint_peak() -> Option<u64> {
        #[cfg(target_os = "macos")]
        {
            macos::lifetime_peak_footprint()
        }
        #[cfg(target_os = "linux")]
        {
            let now = Self::footprint_now()?;
            Some(now.max(SAMPLED_PEAK.load(std::sync::atomic::Ordering::Relaxed)))
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        None
    }

    /// The most the system counted resident of the process in its life,
    /// where the system keeps it: what a process's peak accounting reports
    /// when it exits. On Linux that is the whole resident set (`VmHWM`),
    /// resident pages of mapped files included, so it is not the
    /// footprint's peak ([`ProcessMemory::footprint_peak`]). It cannot be
    /// started again, so it bounds a span of the process only from the
    /// span's end on.
    pub fn peak_footprint() -> Option<u64> {
        #[cfg(target_os = "macos")]
        {
            macos::lifetime_peak_footprint()
        }
        #[cfg(target_os = "linux")]
        {
            linux::peak_resident_bytes()
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        None
    }

    pub fn read() -> Self {
        #[cfg(target_os = "macos")]
        {
            let mut reading = Self::default();
            if let Some(info) = macos::task_vm_info() {
                reading.footprint_bytes = Some(info.phys_footprint);
                reading.graphics_bytes = u64::try_from(info.ledger_tag_graphics_footprint).ok();
                reading.mapped_bytes = Some(info.external);
            }
            reading.malloc_live_bytes = Some(macos::malloc_live_bytes());
            reading
        }
        #[cfg(target_os = "linux")]
        {
            let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
                return Self::default();
            };
            let bytes = |field: &str| {
                status
                    .lines()
                    .find_map(|line| line.strip_prefix(field))?
                    .trim()
                    .strip_suffix("kB")?
                    .trim()
                    .parse::<u64>()
                    .ok()
                    .map(|kibibytes| kibibytes * 1024)
            };
            Self {
                footprint_bytes: bytes("RssAnon:")
                    .zip(bytes("RssShmem:"))
                    .map(|(anonymous, shared)| anonymous + shared),
                malloc_live_bytes: linux::malloc_live_bytes(),
                // A device with memory of its own holds its allocations
                // outside the process; nothing of the footprint is tagged.
                graphics_bytes: Some(0),
                mapped_bytes: bytes("RssFile:"),
            }
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        Self::default()
    }
}

/// The most of the footprint's readings on a system that keeps no mark of
/// it ([`ProcessMemory::footprint_peak`]).
#[cfg(target_os = "linux")]
static SAMPLED_PEAK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(target_os = "linux")]
mod linux {
    /// The most the resident set has been in the life of the process
    /// (`VmHWM`): the kernel keeps no high-water mark of the anonymous part
    /// alone, so this one also counts resident pages of mapped files, which
    /// the footprint does not.
    pub(super) fn peak_resident_bytes() -> Option<u64> {
        status_bytes("VmHWM:")
    }

    /// The footprint: resident anonymous and shared memory, in one reading
    /// of the process's status.
    pub(super) fn footprint_bytes() -> Option<u64> {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        Some(field_bytes(&status, "RssAnon:")? + field_bytes(&status, "RssShmem:")?)
    }

    /// A field of `/proc/self/status` the kernel states in kibibytes.
    pub(super) fn status_bytes(field: &str) -> Option<u64> {
        field_bytes(&std::fs::read_to_string("/proc/self/status").ok()?, field)
    }

    fn field_bytes(status: &str, field: &str) -> Option<u64> {
        status
            .lines()
            .find_map(|line| line.strip_prefix(field))?
            .trim()
            .strip_suffix("kB")?
            .trim()
            .parse::<u64>()
            .ok()
            .map(|kibibytes| kibibytes * 1024)
    }

    /// Heap blocks in use, by the sizes allocated: what glibc's arenas
    /// hold in use and what it mapped for single large blocks. None with
    /// another C library, which reports neither.
    pub(super) fn malloc_live_bytes() -> Option<u64> {
        #[cfg(target_env = "gnu")]
        {
            /// `struct mallinfo2` of `<malloc.h>`.
            #[repr(C)]
            struct Mallinfo2 {
                arena: usize,
                ordblks: usize,
                smblks: usize,
                hblks: usize,
                hblkhd: usize,
                usmblks: usize,
                fsmblks: usize,
                uordblks: usize,
                fordblks: usize,
                keepcost: usize,
            }
            unsafe extern "C" {
                fn mallinfo2() -> Mallinfo2;
            }
            // SAFETY: `mallinfo2` reads the allocator's own counters under
            // its locks and returns them by value.
            let info = unsafe { mallinfo2() };
            Some((info.uordblks + info.hblkhd) as u64)
        }
        #[cfg(not(target_env = "gnu"))]
        None
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use std::ffi::{c_int, c_uint, c_void};

    /// `TASK_VM_INFO` through its graphics ledger tags (within revision 3
    /// of the structure), as `<mach/task_info.h>` lays it out, packed to
    /// four bytes.
    #[repr(C, packed(4))]
    #[derive(Clone, Copy, Default)]
    pub(super) struct TaskVmInfo {
        virtual_size: u64,
        region_count: i32,
        page_size: i32,
        resident_size: u64,
        resident_size_peak: u64,
        device: u64,
        device_peak: u64,
        internal: u64,
        internal_peak: u64,
        pub(super) external: u64,
        external_peak: u64,
        reusable: u64,
        reusable_peak: u64,
        purgeable_volatile_pmap: u64,
        purgeable_volatile_resident: u64,
        purgeable_volatile_virtual: u64,
        compressed: u64,
        compressed_peak: u64,
        compressed_lifetime: u64,
        pub(super) phys_footprint: u64,
        min_address: u64,
        max_address: u64,
        ledger_phys_footprint_peak: i64,
        ledger_purgeable_nonvolatile: i64,
        ledger_purgeable_novolatile_compressed: i64,
        ledger_purgeable_volatile: i64,
        ledger_purgeable_volatile_compressed: i64,
        ledger_tag_network_nonvolatile: i64,
        ledger_tag_network_nonvolatile_compressed: i64,
        ledger_tag_network_volatile: i64,
        ledger_tag_network_volatile_compressed: i64,
        ledger_tag_media_footprint: i64,
        ledger_tag_media_footprint_compressed: i64,
        ledger_tag_media_nofootprint: i64,
        ledger_tag_media_nofootprint_compressed: i64,
        pub(super) ledger_tag_graphics_footprint: i64,
        ledger_tag_graphics_footprint_compressed: i64,
        ledger_tag_graphics_nofootprint: i64,
        ledger_tag_graphics_nofootprint_compressed: i64,
    }

    /// `malloc_statistics_t` of `<malloc/malloc.h>`.
    #[repr(C)]
    #[derive(Default)]
    struct MallocStatistics {
        blocks_in_use: c_uint,
        size_in_use: usize,
        max_size_in_use: usize,
        size_allocated: usize,
    }

    unsafe extern "C" {
        static mach_task_self_: c_uint;
        fn task_info(
            task: c_uint,
            flavor: c_uint,
            info: *mut c_int,
            count: *mut c_uint,
        ) -> c_int;
        fn malloc_zone_statistics(zone: *mut c_void, statistics: *mut MallocStatistics);
        fn malloc_zone_pressure_relief(zone: *mut c_void, goal: usize) -> usize;
    }

    pub(super) fn task_vm_info() -> Option<TaskVmInfo> {
        const TASK_VM_INFO: c_uint = 22;
        let mut info = TaskVmInfo::default();
        let mut count =
            (std::mem::size_of::<TaskVmInfo>() / std::mem::size_of::<c_uint>()) as c_uint;
        // SAFETY: `task_info` writes at most `count` 32-bit words of the
        // flavor's structure, which `TaskVmInfo` is a prefix of.
        let result = unsafe {
            task_info(
                mach_task_self_,
                TASK_VM_INFO,
                (&raw mut info).cast(),
                &mut count,
            )
        };
        (result == 0).then_some(info)
    }

    /// `ri_lifetime_max_phys_footprint` of `rusage_info_v4`
    /// (`<sys/resource.h>`): the figure a process's peak-footprint
    /// accounting reports. The task ledger's own peak is of an interval the
    /// system restarts, and read below the footprint of the same instant.
    pub(super) fn lifetime_peak_footprint() -> Option<u64> {
        const RUSAGE_INFO_V4: c_int = 4;
        /// The structure in 64-bit words: a 16-byte UUID, then 35 counters.
        const WORDS: usize = 2 + 35;
        /// `ri_lifetime_max_phys_footprint`: the 29th counter.
        const LIFETIME_MAX_PHYS_FOOTPRINT: usize = 2 + 28;
        unsafe extern "C" {
            fn getpid() -> c_int;
            fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *mut c_void) -> c_int;
        }
        let mut usage = [0u64; WORDS];
        // SAFETY: `proc_pid_rusage` writes a whole `rusage_info_v4`, which
        // `usage` is the size and alignment of, when it returns zero.
        let result =
            unsafe { proc_pid_rusage(getpid(), RUSAGE_INFO_V4, usage.as_mut_ptr().cast()) };
        (result == 0).then_some(usage[LIFETIME_MAX_PHYS_FOOTPRINT])
    }

    pub(super) fn malloc_live_bytes() -> u64 {
        let mut statistics = MallocStatistics::default();
        // SAFETY: fills the statistics it is given; a null zone sums every
        // zone.
        unsafe { malloc_zone_statistics(std::ptr::null_mut(), &mut statistics) };
        statistics.size_in_use as u64
    }

    pub(super) fn relieve() {
        // SAFETY: a null zone relieves every zone, and a goal of zero asks
        // for everything that can be returned; it frees nothing in use.
        unsafe { malloc_zone_pressure_relief(std::ptr::null_mut(), 0) };
    }
}

/// Return to the system the pages the allocator keeps of memory this
/// process has freed, so that they stop counting in its footprint.
///
/// A load's tuning allocates and frees much on many threads (rendered
/// sources, compiles, prepared kernels, observations), and the allocator
/// keeps the freed pages dirty for reuse: footprint that holds nothing.
/// Measured on an M4 Max (2026-10-09): compiling and releasing 512 Metal
/// programs of the engine's dense output projection left +18.5 MiB of
/// footprint with live heap unchanged, and `malloc_zone_pressure_relief`
/// returned all but 2.5 MiB of it (what Metal keeps for itself) in 7 ms; of
/// the +8.9 MiB twelve whole kernels left through the engine's own path, it
/// returned 7.3 MiB. On glibc the same is `malloc_trim`.
///
/// Returns what the footprint fell by.
pub fn return_freed_pages() -> u64 {
    let before = ProcessMemory::read().footprint_bytes;
    release_freed_heap();
    before
        .zip(ProcessMemory::read().footprint_bytes)
        .map_or(0, |(before, after)| before.saturating_sub(after))
}

/// [`return_freed_pages`] without the two readings of the process: for a
/// caller that returns pages often and counts nothing.
pub fn release_freed_heap() {
    #[cfg(target_os = "macos")]
    macos::relieve();
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        unsafe extern "C" {
            fn malloc_trim(pad: usize) -> std::ffi::c_int;
        }
        // SAFETY: returns free heap to the system and frees nothing in use.
        unsafe { malloc_trim(0) };
    }
}

/// Held by a test while it reads the process's memory as its own work's:
/// the tests of one binary share the process.
#[cfg(test)]
pub(crate) static MEASURING: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    /// The reading follows the heap: bytes allocated and touched are live
    /// heap and footprint, and freed they are neither once returned.
    #[test]
    #[cfg(target_os = "macos")]
    fn a_reading_follows_the_heap_and_freed_pages_are_returned() {
        const BLOCK: usize = 16 << 10;
        const BYTES: usize = 64 << 20;
        let _measuring = MEASURING.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let before = ProcessMemory::read();
        // Small blocks: freed, their pages stay with the allocator.
        let blocks = (0..BYTES / BLOCK)
            .map(|block| vec![(block % 255) as u8 + 1; BLOCK])
            .collect::<Vec<_>>();
        let held = ProcessMemory::read();
        let grew = |part: fn(&ProcessMemory) -> Option<u64>| {
            part(&held).unwrap() as i64 - part(&before).unwrap() as i64
        };
        // Other tests of this process allocate and free meanwhile.
        assert!(grew(|reading| reading.malloc_live_bytes) >= BYTES as i64 / 2);
        assert!(grew(|reading| reading.footprint_bytes) >= BYTES as i64 / 2);
        assert!(std::hint::black_box(&blocks).iter().all(|block| block[0] > 0));
        drop(blocks);
        return_freed_pages();
        let after = ProcessMemory::read();
        let left = after.footprint_bytes.unwrap() as i64 - before.footprint_bytes.unwrap() as i64;
        assert!(left < BYTES as i64 / 8, "{left} bytes of freed heap stay");
        assert!(
            after.malloc_live_bytes.unwrap() < held.malloc_live_bytes.unwrap() - BYTES as u64 / 2
        );
        // The life's peak is no less than any reading taken in it.
        assert!(ProcessMemory::peak_footprint().unwrap() >= held.footprint_bytes.unwrap());
    }
}
