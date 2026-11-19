//! Windows x64: `GlobalMemoryStatusEx`, the memory performance counters,
//! the memory nodes' free pages, job-object memory limits, hard faults and
//! the low-memory resource notification, each read through [`Facts`].
//!
//! Physical RAM, commit limit and page files remain separate. The commit
//! figures are process-scoped ("the maximum this process can commit"):
//! within a job, they are the job's.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use super::{read, Counted, Harm, HostSample, Lag, Limit, OwnCold, OwnReclaimed, PressureLevel, ProcessLimitKind, Supply, Widened};
use seismic_topology::{Reported, ReportedBy};
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};
use windows_sys::Wdk::System::SystemServices::RtlGetVersion;
use windows_sys::Win32::Foundation::RtlNtStatusToDosError;
use windows_sys::Win32::System::JobObjects::{
    IsProcessInJob, JobObjectExtendedLimitInformation, QueryInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_PROCESS_MEMORY, JOB_OBJECT_LIMIT_WORKINGSET,
};
use windows_sys::Win32::System::Memory::QueryMemoryResourceNotification;
use windows_sys::Win32::System::Performance::{PDH_HCOUNTER, PDH_HQUERY};
use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX2};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, LOGICAL_PROCESSOR_RELATIONSHIP, MEMORYSTATUSEX, OSVERSIONINFOW};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// Every OS read discovery makes on Windows, its one owner. A read some
/// machines refuse or lack is the value or none ([`read`]: the performance
/// counters to an account outside Performance Monitor Users, the private
/// working set before Windows 10 1809, a job's limits); a call that answers
/// on every Windows is its value, and its failure an invariant violation.
pub(crate) trait Facts {
    /// `GlobalMemoryStatusEx`.
    fn memory_status(&self) -> MEMORYSTATUSEX;
    /// `RtlGetVersion`: the true version and build, which `GetVersionEx`
    /// reports only to a manifested process.
    fn os_version(&self) -> OSVERSIONINFOW;
    /// The processor's identifying fields ([`super::cpu_identity`]):
    /// `GetNativeSystemInfo`'s architecture, count, level and revision with
    /// CPUID's vendor, signature and feature, and brand leaves.
    fn cpu_identity(&self) -> String;
    /// The records `GetLogicalProcessorInformationEx(relationship)` writes,
    /// each `Size` bytes, in a buffer aligned for them.
    fn processor_records(&self, relationship: LOGICAL_PROCESSOR_RELATIONSHIP) -> Option<Vec<u64>>;
    /// `GetNumaHighestNodeNumber`.
    fn highest_node(&self) -> u16;
    /// `GetNumaAvailableMemoryNodeEx(node)`: bytes available on it; none
    /// where `node` numbers no node (node numbers need not be contiguous).
    fn numa_available(&self, node: u16) -> Option<u64>;
    /// `GetProcessGroupAffinity`: the processor groups the process runs in.
    fn process_groups(&self) -> Option<Vec<u16>>;
    /// `GetProcessAffinityMask`'s process mask.
    fn affinity_mask(&self) -> Option<usize>;
    /// The memory performance counters, collected together.
    fn counters(&self) -> Option<Counters>;
    /// This process's hard faults since it started: its
    /// `SYSTEM_PROCESS_INFORMATION` entry's `HardFaultCount`.
    fn hard_faults(&self) -> Option<u32>;
    /// `K32GetProcessMemoryInfo` of this process: its working set and its
    /// private commit (`PagefileUsage`, the same figure as `PrivateUsage`).
    fn memory_counters(&self) -> PROCESS_MEMORY_COUNTERS;
    /// This process's private working set (`PROCESS_MEMORY_COUNTERS_EX2`'s
    /// `PrivateWorkingSetSize`); none before Windows 10 1809 (build 17763),
    /// which has no such structure.
    fn private_working_set(&self) -> Option<u64>;
    /// Whether the low-memory resource notification is signalled now.
    fn low_memory(&self) -> bool;
    /// The extended limits of the job this process runs in, its own
    /// (`QueryInformationJobObject` of no handle reads the immediate job,
    /// never a parent's); none outside a job.
    fn job(&self) -> Option<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>;
    /// The system's page (`GetSystemInfo`'s `dwPageSize`, through the one
    /// owner of the host page, `seismic_memory::host_page`).
    fn page(&self) -> u64;
    /// The CPUs the process may keep busy at once
    /// (`seismic_threads::quota`, their one owner: `available_parallelism`,
    /// its affinity under a cgroup's CPU quota, or one).
    fn parallelism(&self) -> NonZeroUsize;
}

/// The running OS's facts.
pub(crate) struct Os;

/// The last OS error as the outcome of the read `what`.
fn last_error<T>(what: &str) -> Option<T> {
    read(what, Err(std::io::Error::last_os_error()))
}

/// A Win32 call that answers on every Windows: `answered` true, or an
/// invariant violation naming `what`.
fn always(what: &str, answered: bool) {
    assert!(answered, "invariant violated: {what}: {}", std::io::Error::last_os_error());
}

impl Facts for Os {
    fn memory_status(&self) -> MEMORYSTATUSEX {
        let mut status = MEMORYSTATUSEX { dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32, ..Default::default() };
        always("GlobalMemoryStatusEx", unsafe { GlobalMemoryStatusEx(&mut status) } != 0);
        status
    }

    fn os_version(&self) -> OSVERSIONINFOW {
        let mut version = OSVERSIONINFOW { dwOSVersionInfoSize: std::mem::size_of::<OSVERSIONINFOW>() as u32, ..unsafe { std::mem::zeroed() } };
        let status = unsafe { RtlGetVersion(&mut version) };
        assert_eq!(status, 0, "invariant violated: RtlGetVersion returned NTSTATUS {status:#x}");
        version
    }

    fn cpu_identity(&self) -> String {
        use windows_sys::Win32::System::SystemInformation::{GetNativeSystemInfo, SYSTEM_INFO};
        let mut system: SYSTEM_INFO = unsafe { std::mem::zeroed() };
        unsafe { GetNativeSystemInfo(&mut system) };
        let architecture = unsafe { system.Anonymous.Anonymous.wProcessorArchitecture };
        // CPUID's vendor, model/stepping and brand leaves identify the Windows
        // x64 backend's processor; GetNativeSystemInfo supplies its width.
        let mut cpu: Vec<(u32, u32, u32, u32, u32)> = Vec::new();
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::__cpuid;
            let max = __cpuid(0).eax;
            for leaf in [0, 1] {
                if leaf <= max {
                    let report = __cpuid(leaf);
                    // Leaf 1 EBX contains the executing core's APIC ID. It is
                    // not part of the machine-wide hardware identity.
                    let ebx = if leaf == 0 { report.ebx } else { 0 };
                    cpu.push((leaf, report.eax, ebx, report.ecx, report.edx));
                }
            }
            let extended = __cpuid(0x8000_0000).eax;
            for leaf in 0x8000_0002..=0x8000_0004 {
                if leaf <= extended {
                    let report = __cpuid(leaf);
                    cpu.push((leaf, report.eax, report.ebx, report.ecx, report.edx));
                }
            }
        }
        serde_json::to_string(&(architecture, system.dwNumberOfProcessors, system.wProcessorLevel, system.wProcessorRevision, cpu))
            .unwrap_or_else(|error| panic!("invariant violated: integers serialize: {error}"))
    }

    fn processor_records(&self, relationship: LOGICAL_PROCESSOR_RELATIONSHIP) -> Option<Vec<u64>> {
        use windows_sys::Win32::System::SystemInformation::GetLogicalProcessorInformationEx;
        let mut length = 0u32;
        unsafe { GetLogicalProcessorInformationEx(relationship, std::ptr::null_mut(), &mut length) };
        let mut buffer = vec![0u64; (length as usize).div_ceil(8)];
        if unsafe { GetLogicalProcessorInformationEx(relationship, buffer.as_mut_ptr().cast(), &mut length) } == 0 {
            return last_error(&format!("GetLogicalProcessorInformationEx({relationship})"));
        }
        buffer.truncate((length as usize).div_ceil(8));
        Some(buffer)
    }

    fn highest_node(&self) -> u16 {
        use windows_sys::Win32::System::Threading::GetNumaHighestNodeNumber;
        let mut highest = 0u32;
        always("GetNumaHighestNodeNumber", unsafe { GetNumaHighestNodeNumber(&mut highest) } != 0);
        u16::try_from(highest).unwrap_or_else(|_| panic!("invariant violated: NUMA node {highest} beyond u16"))
    }

    fn numa_available(&self, node: u16) -> Option<u64> {
        use windows_sys::Win32::Foundation::ERROR_INVALID_PARAMETER;
        use windows_sys::Win32::System::Threading::GetNumaAvailableMemoryNodeEx;
        let mut available = 0u64;
        if unsafe { GetNumaAvailableMemoryNodeEx(node, &mut available) } != 0 {
            return Some(available);
        }
        let error = std::io::Error::last_os_error();
        match error.raw_os_error() == Some(ERROR_INVALID_PARAMETER as i32) {
            // A number naming no node.
            true => None,
            false => read(format!("GetNumaAvailableMemoryNodeEx({node})"), Err(error)),
        }
    }

    fn process_groups(&self) -> Option<Vec<u16>> {
        use windows_sys::Win32::System::Threading::GetProcessGroupAffinity;
        let process = unsafe { GetCurrentProcess() };
        let mut count = 0u16;
        unsafe { GetProcessGroupAffinity(process, &mut count, std::ptr::null_mut()) };
        let mut groups = vec![0u16; usize::from(count)];
        match unsafe { GetProcessGroupAffinity(process, &mut count, groups.as_mut_ptr()) } {
            0 => last_error("GetProcessGroupAffinity"),
            _ => Some(groups),
        }
    }

    fn affinity_mask(&self) -> Option<usize> {
        use windows_sys::Win32::System::Threading::GetProcessAffinityMask;
        let (mut mask, mut system) = (0usize, 0usize);
        match unsafe { GetProcessAffinityMask(GetCurrentProcess(), &mut mask, &mut system) } {
            0 => last_error("GetProcessAffinityMask"),
            _ => Some(mask),
        }
    }

    fn counters(&self) -> Option<Counters> {
        use windows_sys::Win32::System::Performance::{
            PdhAddEnglishCounterW, PdhCollectQueryData, PdhGetRawCounterValue, PdhOpenQueryW, PDH_CSTATUS_NEW_DATA,
            PDH_CSTATUS_VALID_DATA, PDH_RAW_COUNTER,
        };
        // Opened once: an account refused the counters stays refused.
        static QUERY: std::sync::OnceLock<Option<std::sync::Mutex<Query>>> = std::sync::OnceLock::new();
        let query = QUERY.get_or_init(|| {
            let mut query = std::ptr::null_mut();
            pdh("PdhOpenQueryW", unsafe { PdhOpenQueryW(std::ptr::null(), 0, &mut query) })?;
            let mut counters = [std::ptr::null_mut(); 5];
            for (counter, path) in counters.iter_mut().zip(COUNTER_PATHS) {
                let wide: Vec<u16> = path.encode_utf16().chain(std::iter::once(0)).collect();
                pdh(path, unsafe { PdhAddEnglishCounterW(query, wide.as_ptr(), 0, counter) })?;
            }
            Some(std::sync::Mutex::new(Query { query, counters }))
        });
        // A collector that panicked left the query's handles whole.
        let query = query.as_ref()?.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        pdh("PdhCollectQueryData", unsafe { PdhCollectQueryData(query.query) })?;
        let mut values = [0u64; 5];
        for ((value, counter), path) in values.iter_mut().zip(query.counters).zip(COUNTER_PATHS) {
            let mut raw: PDH_RAW_COUNTER = unsafe { std::mem::zeroed() };
            pdh(path, unsafe { PdhGetRawCounterValue(counter, std::ptr::null_mut(), &mut raw) })?;
            if raw.CStatus != PDH_CSTATUS_VALID_DATA && raw.CStatus != PDH_CSTATUS_NEW_DATA {
                pdh(path, raw.CStatus)?;
            }
            *value = u64::try_from(raw.FirstValue).unwrap_or_else(|_| panic!("invariant violated: {path} reads {}", raw.FirstValue));
        }
        let [free_zero, core, normal, reserve, reads] = values;
        // `Page Reads/sec` counts in 32 bits (`PERF_COUNTER_COUNTER`).
        Some(Counters { free_zero, standby: core + normal + reserve, page_reads: reads as u32 })
    }

    /// `HardFaultCount` at offset 0x10 on x64, `UniqueProcessId` at 0x50,
    /// the one per-process count of them.
    ///
    /// # Panics
    /// Where the table the OS wrote is cut short or lists no entry for this
    /// process.
    fn hard_faults(&self) -> Option<u32> {
        use windows_sys::Wdk::System::SystemInformation::{NtQuerySystemInformation, SystemProcessInformation};
        use windows_sys::Win32::Foundation::STATUS_INFO_LENGTH_MISMATCH;
        let mut buffer = vec![0u64; 1 << 16];
        loop {
            let bytes = u32::try_from(buffer.len() * 8).unwrap_or_else(|_| panic!("invariant violated: the process table outgrows u32 bytes"));
            let mut returned = 0u32;
            let status = unsafe { NtQuerySystemInformation(SystemProcessInformation, buffer.as_mut_ptr().cast(), bytes, &mut returned) };
            if status == STATUS_INFO_LENGTH_MISMATCH {
                buffer.resize((returned as usize).div_ceil(8) * 2, 0);
                continue;
            }
            if status < 0 {
                let error = std::io::Error::from_raw_os_error(unsafe { RtlNtStatusToDosError(status) } as i32);
                return read("NtQuerySystemInformation(SystemProcessInformation)", Err(error));
            }
            break;
        }
        // SAFETY: the buffer's bytes, as the OS wrote them.
        let table: &[u8] = unsafe { std::slice::from_raw_parts(buffer.as_ptr().cast(), buffer.len() * 8) };
        let field = |at: usize, width: usize| -> u64 {
            let bytes = table.get(at..at + width).unwrap_or_else(|| panic!("invariant violated: the process table is cut short"));
            let mut value = [0u8; 8];
            value[..width].copy_from_slice(bytes);
            u64::from_le_bytes(value)
        };
        let here = u64::from(std::process::id());
        let mut at = 0usize;
        loop {
            if field(at + 0x50, 8) == here {
                return Some(field(at + 0x10, 4) as u32);
            }
            match field(at, 4) {
                0 => panic!("invariant violated: the process table lists no entry for this process"),
                next => at += next as usize,
            }
        }
    }

    fn memory_counters(&self) -> PROCESS_MEMORY_COUNTERS {
        let mut counters = PROCESS_MEMORY_COUNTERS { cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32, ..Default::default() };
        always("GetProcessMemoryInfo", unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) } != 0);
        counters
    }

    fn private_working_set(&self) -> Option<u64> {
        /// Windows 10 1809, the first build that knows the structure.
        const EX2: u32 = 17763;
        if self.os_version().dwBuildNumber < EX2 {
            return None;
        }
        let mut counters = PROCESS_MEMORY_COUNTERS_EX2 { cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX2>() as u32, ..Default::default() };
        let answered = unsafe {
            K32GetProcessMemoryInfo(GetCurrentProcess(), (&mut counters as *mut PROCESS_MEMORY_COUNTERS_EX2).cast(), counters.cb)
        };
        always("GetProcessMemoryInfo(PROCESS_MEMORY_COUNTERS_EX2)", answered != 0);
        Some(counters.PrivateWorkingSetSize as u64)
    }

    fn low_memory(&self) -> bool {
        let mut state = 0;
        let low = crate::pressure::low_memory_notification();
        always("QueryMemoryResourceNotification", unsafe { QueryMemoryResourceNotification(low, &mut state) } != 0);
        state != 0
    }

    fn job(&self) -> Option<JOBOBJECT_EXTENDED_LIMIT_INFORMATION> {
        let mut in_job = 0;
        if unsafe { IsProcessInJob(GetCurrentProcess(), std::ptr::null_mut(), &mut in_job) } == 0 {
            return last_error("IsProcessInJob");
        }
        if in_job == 0 {
            return None;
        }
        let mut information = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        let queried = unsafe {
            QueryInformationJobObject(
                std::ptr::null_mut(),
                JobObjectExtendedLimitInformation,
                (&mut information as *mut JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                std::ptr::null_mut(),
            )
        };
        match queried {
            0 => last_error("QueryInformationJobObject(JobObjectExtendedLimitInformation)"),
            _ => Some(information),
        }
    }

    fn page(&self) -> u64 {
        static PAGE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        *PAGE.get_or_init(|| seismic_memory::host_page().unwrap_or_else(|error| panic!("invariant violated: the system's page: {error}")))
    }

    fn parallelism(&self) -> NonZeroUsize {
        seismic_threads::quota()
    }
}

/// A PDH call's `status` as a read ([`read`]): an account refused the
/// counters (`PDH_ACCESS_DENIED`) or a counter or object the system does
/// not provide (its provider disabled or its registration damaged) is none;
/// any other failure an invariant violation.
fn pdh(what: &str, status: u32) -> Option<()> {
    use windows_sys::Win32::System::Performance::{PDH_ACCESS_DENIED, PDH_CSTATUS_NO_COUNTER, PDH_CSTATUS_NO_OBJECT, PDH_NO_DATA};
    let outcome = match status {
        0 => Ok(()),
        PDH_ACCESS_DENIED => Err(std::io::ErrorKind::PermissionDenied.into()),
        PDH_CSTATUS_NO_OBJECT | PDH_CSTATUS_NO_COUNTER | PDH_NO_DATA => Err(std::io::ErrorKind::NotFound.into()),
        status => Err(std::io::Error::other(format!("PDH status {status:#x}"))),
    };
    read(what, outcome)
}

/// The OS's diagnostics for the hardware report: none on Windows.
pub(super) fn diagnostics(_facts: &impl Facts) -> Vec<seismic_topology::Diagnostic> {
    Vec::new()
}

pub(super) fn cpu_identity(facts: &impl Facts) -> String {
    facts.cpu_identity()
}

/// `GlobalMemoryStatusEx`'s `ullTotalPhys`.
pub(super) fn capacity(facts: &impl Facts) -> Reported<u64> {
    Reported::new(facts.memory_status().ullTotalPhys, ReportedBy::WindowsTotalPhys)
}

/// `RtlGetVersion`'s version and build.
pub(super) fn os_build(facts: &impl Facts) -> String {
    let version = facts.os_version();
    format!("Windows {}.{}.{}", version.dwMajorVersion, version.dwMinorVersion, version.dwBuildNumber)
}

/// `GetLogicalProcessorInformationEx(RelationCache)`: the highest-level data
/// or unified caches, one record per instance, summed; none where the OS
/// describes none or refuses the records.
pub(super) fn last_level_cache(facts: &impl Facts) -> Option<Reported<u64>> {
    use windows_sys::Win32::System::SystemInformation::{CacheData, CacheUnified, RelationCache};
    // The highest level seen, and its instances' bytes together.
    let mut last: Option<(u8, u64)> = None;
    each_record(&facts.processor_records(RelationCache)?, |record| {
        if record.Relationship == RelationCache {
            // SAFETY: a `RelationCache` record's union is its cache.
            let cache = unsafe { &record.Anonymous.Cache };
            if cache.Type == CacheData || cache.Type == CacheUnified {
                let size = u64::from(cache.CacheSize);
                last = match last {
                    Some((level, total)) if level == cache.Level => Some((level, total + size)),
                    Some((level, _)) if level > cache.Level => last,
                    _ => Some((cache.Level, size)),
                };
            }
        }
    });
    last.map(|(_, total)| Reported::new(total, ReportedBy::WindowsProcessorCache))
}

/// Each record of `buffer` ([`Facts::processor_records`]), in order.
fn each_record(buffer: &[u64], mut each: impl FnMut(&windows_sys::Win32::System::SystemInformation::SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)) {
    let (bytes, length) = (buffer.as_ptr().cast::<u8>(), buffer.len() * 8);
    let mut offset = 0usize;
    while offset < length {
        // SAFETY: the OS wrote consecutive records, each `Size` bytes long.
        let record = unsafe { &*(bytes.add(offset) as *const windows_sys::Win32::System::SystemInformation::SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) };
        if record.Size == 0 {
            break;
        }
        each(record);
        offset += record.Size as usize;
    }
}

/// The CPUs of `masks`, each its group's 64 times its group plus its bit.
fn group_cpus(masks: &[windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY]) -> Vec<usize> {
    let mut cpus = masks
        .iter()
        .flat_map(|mask| {
            (0..usize::BITS as usize).filter(move |bit| mask.Mask >> bit & 1 == 1).map(move |bit| usize::from(mask.Group) * 64 + bit)
        })
        .collect::<Vec<_>>();
    cpus.sort_unstable();
    cpus
}

/// Each NUMA node (`RelationNumaNode`), with memory where
/// `GetNumaAvailableMemoryNodeEx` reports any available on it, its cores
/// (`RelationProcessorCore`) those whose threads lie in it, each a
/// performance core of the highest `EfficiencyClass`; the process may run
/// on the CPUs of its groups (`GetProcessGroupAffinity`), within its
/// affinity mask where it has one group (`GetProcessAffinityMask`). Where
/// any of it is unanswered, the flat topology
/// (`crate::processors::flat`) of the quota's CPUs.
pub(super) fn processors(facts: &impl Facts) -> crate::Processors {
    let (nodes, allowed, by) = match reported_processors(facts) {
        Some((nodes, allowed)) => (nodes, allowed, crate::ProcessorsReportedBy::WindowsProcessorInformation),
        None => {
            let every = (0..facts.parallelism().get()).collect();
            (crate::processors::flat(&every), every, crate::ProcessorsReportedBy::Flat)
        }
    };
    crate::Processors::new(nodes, allowed, facts.parallelism(), by).unwrap_or_else(|error| panic!("invariant violated: the processors the OS reports: {error}"))
}

/// The nodes and allowed CPUs the OS reports ([`processors`]), none where
/// any of them is unanswered.
fn reported_processors(facts: &impl Facts) -> Option<(Vec<crate::MemoryNode>, std::collections::BTreeSet<usize>)> {
    use windows_sys::Win32::System::SystemInformation::{RelationNumaNode, RelationProcessorCore};
    let mut cores = Vec::new();
    each_record(&facts.processor_records(RelationProcessorCore)?, |record| {
        // SAFETY: a `RelationProcessorCore` record's union is its processor.
        let core = unsafe { &record.Anonymous.Processor };
        // SAFETY: the record holds `GroupCount` masks.
        let masks = unsafe { std::slice::from_raw_parts(core.GroupMask.as_ptr(), usize::from(core.GroupCount)) };
        cores.push((group_cpus(masks), core.EfficiencyClass));
    });
    let fastest = cores.iter().map(|(_, class)| *class).max()?;
    let mut nodes = Vec::new();
    let mut unanswered = false;
    each_record(&facts.processor_records(RelationNumaNode)?, |record| {
        // SAFETY: a `RelationNumaNode` record's union is its node, holding
        // `GroupCount` masks (one where it reports none).
        let node = unsafe { &record.Anonymous.NumaNode };
        let masks = unsafe { std::slice::from_raw_parts(node.Anonymous.GroupMasks.as_ptr(), usize::from(node.GroupCount).max(1)) };
        let cpus = group_cpus(masks);
        let number = u16::try_from(node.NodeNumber).unwrap_or_else(|_| panic!("invariant violated: NUMA node {}", node.NodeNumber));
        let Some(available) = facts.numa_available(number) else {
            unanswered = true;
            return;
        };
        let cores = cores
            .iter()
            .filter(|(threads, _)| threads.first().is_some_and(|first| cpus.contains(first)))
            .map(|(threads, class)| crate::Core { threads: threads.clone(), performance: *class == fastest })
            .collect();
        nodes.push(crate::MemoryNode { id: usize::from(number), memory: available > 0, cores });
    });
    if unanswered {
        return None;
    }
    let every = cores.iter().flat_map(|(threads, _)| threads.iter().copied());
    let allowed = match facts.process_groups()?.as_slice() {
        [group] => {
            let mask = facts.affinity_mask()?;
            every.filter(|cpu| cpu / 64 == usize::from(*group) && mask >> (cpu % 64) & 1 == 1).collect()
        }
        groups => every.filter(|cpu| groups.iter().any(|group| cpu / 64 == usize::from(*group))).collect(),
    };
    Some((nodes, allowed))
}

/// Nothing: the process's affinity mask would bound every thread's,
/// [`pin`]'s workers among them, and the process's threads are not
/// enumerated here.
pub(super) fn place_process(_cpus: &[usize]) -> Result<(), String> {
    Ok(())
}

/// `SetThreadGroupAffinity` of the calling thread to `cpus`: a thread runs
/// in one processor group, so to those of the group holding the most of
/// them (the first such).
pub(super) fn pin(cpus: &[usize]) -> Result<(), String> {
    use windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY;
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadGroupAffinity};
    let in_group = |group: usize| cpus.iter().filter(|&&cpu| cpu / 64 == group).count();
    let group = cpus
        .iter()
        .map(|&cpu| cpu / 64)
        .max_by_key(|&group| (in_group(group), std::cmp::Reverse(group)))
        .ok_or("a thread placed on no CPU")?;
    let mask = cpus.iter().filter(|&&cpu| cpu / 64 == group).fold(0usize, |mask, &cpu| mask | 1 << (cpu % 64));
    let group = u16::try_from(group).map_err(|_| format!("CPUs {cpus:?} beyond every group"))?;
    let affinity = GROUP_AFFINITY { Mask: mask, Group: group, Reserved: [0; 3] };
    match unsafe { SetThreadGroupAffinity(GetCurrentThread(), &affinity, std::ptr::null_mut()) } {
        0 => Err(format!("SetThreadGroupAffinity to CPUs {cpus:?}: {}", std::io::Error::last_os_error())),
        _ => Ok(()),
    }
}

/// Host RAM now: the free and zero lists and the standby lists
/// ([`lists`]), bounded by the commit this process has left
/// (`GlobalMemoryStatusEx`) and by its own job's limits ([`job_limits`]);
/// the machine's hard-fault page reads and this process's hard faults,
/// each one paging read ([`Harm`]), none without the performance counters;
/// the low-memory notification.
pub(super) fn sample(facts: &impl Facts) -> HostSample {
    let at = Instant::now();
    let status = facts.memory_status();
    let counters = facts.counters();
    let (free_zero, standby) = lists(facts, counters.as_ref(), status.ullAvailPhys);
    let limits = job_limits(facts, &status);
    let pressure = match facts.low_memory() {
        true => PressureLevel::Warn,
        false => PressureLevel::Normal,
    };
    // The process's own read after the machine's, so a fault between the
    // two counts as its own alone: never another program's. Both are the
    // kernel's own counts, read in the call: they show every event before
    // the reading began.
    let harm = counters
        .as_ref()
        .zip(facts.hard_faults())
        .map(|(counters, own)| Harm { machine: PAGE_READS.of(counters.page_reads), own: OWN_HARD_FAULTS.of(own) });
    let counted_at = Instant::now();
    // What the lists hold, within the commit this process can still make:
    // a page taken is committed first.
    let room = (free_zero + standby).min(status.ullAvailPageFile);
    let free = free_zero.min(room);
    let host = super::Scope {
        total: status.ullTotalPhys,
        supply: Supply { free_beyond_target: super::beyond(free, 0), clean_cache: room - free, cold_reclaimable: 0 },
    };
    // A job's commit and working set count a growth in the call, whichever
    // scope binds.
    let scope = super::binding(host, &limits).scope;
    HostSample {
        at,
        page_bytes: facts.page(),
        total: scope.total,
        supply: scope.supply,
        counted: Counted { at: counted_at, lag: counted_at.duration_since(at) },
        // The working-set manager trims cold pages to the standby list,
        // which `clean_cache` counts.
        cold: 0,
        own_cold: OwnCold::Unobserved,
        own_reclaimed: own_trimmed(facts),
        wired: None,
        harm,
        pressure: Some(pressure),
        // A free decommits in the call, and a growth commits in it.
        lag: Lag { release: Duration::ZERO, growth: Duration::ZERO },
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
fn lists(facts: &impl Facts, counters: Option<&Counters>, available: u64) -> (u64, u64) {
    if let Some(counters) = counters {
        return (counters.free_zero, counters.standby);
    }
    let free_zero = (0..=facts.highest_node()).filter_map(|node| facts.numa_available(node)).sum::<u64>().min(available);
    (free_zero, available - free_zero)
}

/// The performance counters a sample reads, collected together.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Counters {
    /// `\Memory\Free & Zero Page List Bytes`.
    free_zero: u64,
    /// `\Memory\Standby Cache {Core, Normal Priority, Reserve} Bytes`.
    standby: u64,
    /// `\Memory\Page Reads/sec`'s raw count: hard-fault reads since boot,
    /// in 32 bits.
    page_reads: u32,
}

/// The open query of [`Counters`], opened once.
struct Query {
    query: PDH_HQUERY,
    /// Free & zero, the three standby lists, page reads.
    counters: [PDH_HCOUNTER; 5],
}

// SAFETY: PDH's handles are process-wide; the query is only collected under
// its lock.
unsafe impl Send for Query {}

const COUNTER_PATHS: [&str; 5] = [
    "\\Memory\\Free & Zero Page List Bytes",
    "\\Memory\\Standby Cache Core Bytes",
    "\\Memory\\Standby Cache Normal Priority Bytes",
    "\\Memory\\Standby Cache Reserve Bytes",
    "\\Memory\\Page Reads/sec",
];

static PAGE_READS: Widened = Widened::new();
static OWN_HARD_FAULTS: Widened = Widened::new();

/// This process's private pages out of its working set: its private commit
/// less its private working set ([`HostSample::own_reclaimed`]); none
/// before Windows 10 1809, which reports no private working set.
fn own_trimmed(facts: &impl Facts) -> Option<OwnReclaimed> {
    let private_working_set = facts.private_working_set()?;
    // A copy-on-write page of a mapped image turns private in the working
    // set with its commit charged to the image's section, so the private
    // working set may exceed the private commit: none is then out of it.
    // An upper bound ([`OwnReclaimed::exact`]), and a trimmed page waits in
    // RAM on the modified or standby list, which no figure here separates:
    // none is counted stored.
    Some(OwnReclaimed {
        bytes: (facts.memory_counters().PagefileUsage as u64).saturating_sub(private_working_set),
        stored: 0,
        exact: false,
    })
}

/// The memory limits of this process's own job (a parent job's are not
/// read: no handle to it is held), each against the usage it is enforced
/// on; none outside a job, or where the job's limits are refused:
/// - `JOB_OBJECT_LIMIT_JOB_MEMORY`: `JobMemoryLimit`, the commit of every
///   process of the job, against the commit `GlobalMemoryStatusEx` counts
///   against the least commit limit on this process (`ullTotalPageFile −
///   ullAvailPageFile`, its limit "for the system or the current process,
///   whichever is smaller"): the job's committed total wherever the job's
///   limit is that least one, and where it is not, the system's, whose
///   room is then no more than the job's.
/// - `JOB_OBJECT_LIMIT_PROCESS_MEMORY`: `ProcessMemoryLimit`, against this
///   process's private commit.
/// - `JOB_OBJECT_LIMIT_WORKINGSET`: `MaximumWorkingSetSize`, against this
///   process's working set: a physical-memory scope, past which the OS
///   trims the process's pages rather than failing its allocation.
fn job_limits(facts: &impl Facts, status: &MEMORYSTATUSEX) -> Vec<Limit> {
    let Some(job) = facts.job() else {
        return Vec::new();
    };
    let flags = job.BasicLimitInformation.LimitFlags;
    let set = |flag: u32| flags & flag != 0;
    let process = facts.memory_counters();
    let mut limits = Vec::new();
    if set(JOB_OBJECT_LIMIT_JOB_MEMORY) {
        limits.push(Limit {
            kind: ProcessLimitKind::WindowsJobCommit,
            limit_bytes: job.JobMemoryLimit as u64,
            used_bytes: status.ullTotalPageFile.saturating_sub(status.ullAvailPageFile),
            reclaimable_bytes: 0,
        });
    }
    if set(JOB_OBJECT_LIMIT_PROCESS_MEMORY) {
        limits.push(Limit {
            kind: ProcessLimitKind::WindowsJobCommit,
            limit_bytes: job.ProcessMemoryLimit as u64,
            used_bytes: process.PagefileUsage as u64,
            reclaimable_bytes: 0,
        });
    }
    if set(JOB_OBJECT_LIMIT_WORKINGSET) {
        limits.push(Limit {
            kind: ProcessLimitKind::WindowsJobWorkingSet,
            limit_bytes: job.BasicLimitInformation.MaximumWorkingSetSize as u64,
            used_bytes: process.WorkingSetSize as u64,
            reclaimable_bytes: 0,
        });
    }
    limits
}

/// No device's memory node is read here.
pub(super) fn device_node(_facts: &impl Facts, _location: crate::PciLocation) -> crate::DeviceNode {
    crate::DeviceNode::Unknown
}

/// The machine's names (`crate::machine`): the SMBIOS table
/// `GetSystemFirmwareTable` returns (provider `RSMB`), after its eight-byte
/// header (calling method, versions, then the table's length). A table the
/// OS does not return reads as nothing.
pub(super) fn machine() -> crate::machine::Firmware {
    use windows_sys::Win32::System::SystemInformation::GetSystemFirmwareTable;
    const RSMB: u32 = u32::from_be_bytes(*b"RSMB");
    let size = unsafe { GetSystemFirmwareTable(RSMB, 0, std::ptr::null_mut(), 0) };
    if !(8..=1 << 20).contains(&size) {
        return crate::machine::Firmware::default();
    }
    let mut data = vec![0u8; size as usize];
    if unsafe { GetSystemFirmwareTable(RSMB, 0, data.as_mut_ptr().cast(), size) } != size {
        return crate::machine::Firmware::default();
    }
    let length = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    data.get(8..8 + length).map_or_else(crate::machine::Firmware::default, crate::machine::parse_smbios)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;

    /// A Windows machine's facts for one case: 64 GiB, 40 GiB of it
    /// available, a commit room of 100 GiB, no job, two memory nodes, a
    /// process of 3 GiB private commit and 2 GiB working set.
    struct Machine {
        status: MEMORYSTATUSEX,
        counters: Option<Counters>,
        nodes: Vec<Option<u64>>,
        job: Option<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>,
        process: PROCESS_MEMORY_COUNTERS,
        records: Option<Vec<u64>>,
        parallelism: NonZeroUsize,
    }

    impl Machine {
        fn new() -> Self {
            Self {
                status: MEMORYSTATUSEX {
                    ullTotalPhys: 64 * GB,
                    ullAvailPhys: 40 * GB,
                    ullTotalPageFile: 120 * GB,
                    ullAvailPageFile: 100 * GB,
                    ..Default::default()
                },
                counters: Some(Counters { free_zero: 10 * GB, standby: 30 * GB, page_reads: 7 }),
                nodes: vec![Some(6 * GB), Some(4 * GB)],
                job: None,
                process: PROCESS_MEMORY_COUNTERS { PagefileUsage: 3 * GB as usize, WorkingSetSize: 2 * GB as usize, ..Default::default() },
                records: None,
                parallelism: NonZeroUsize::new(4).unwrap(),
            }
        }

        fn in_job(mut self, flags: u32, edit: impl FnOnce(&mut JOBOBJECT_EXTENDED_LIMIT_INFORMATION)) -> Self {
            let mut job = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            job.BasicLimitInformation.LimitFlags = flags;
            edit(&mut job);
            self.job = Some(job);
            self
        }
    }

    impl Facts for Machine {
        fn memory_status(&self) -> MEMORYSTATUSEX {
            self.status
        }
        fn os_version(&self) -> OSVERSIONINFOW {
            OSVERSIONINFOW { dwMajorVersion: 10, dwBuildNumber: 26100, ..unsafe { std::mem::zeroed() } }
        }
        fn cpu_identity(&self) -> String {
            "cpu".into()
        }
        fn processor_records(&self, _relationship: LOGICAL_PROCESSOR_RELATIONSHIP) -> Option<Vec<u64>> {
            self.records.clone()
        }
        fn highest_node(&self) -> u16 {
            (self.nodes.len() - 1) as u16
        }
        fn numa_available(&self, node: u16) -> Option<u64> {
            self.nodes[usize::from(node)]
        }
        fn process_groups(&self) -> Option<Vec<u16>> {
            None
        }
        fn affinity_mask(&self) -> Option<usize> {
            None
        }
        fn counters(&self) -> Option<Counters> {
            self.counters
        }
        fn hard_faults(&self) -> Option<u32> {
            Some(3)
        }
        fn memory_counters(&self) -> PROCESS_MEMORY_COUNTERS {
            self.process
        }
        fn private_working_set(&self) -> Option<u64> {
            Some(GB)
        }
        fn low_memory(&self) -> bool {
            false
        }
        fn job(&self) -> Option<JOBOBJECT_EXTENDED_LIMIT_INFORMATION> {
            self.job
        }
        fn page(&self) -> u64 {
            4096
        }
        fn parallelism(&self) -> NonZeroUsize {
            self.parallelism
        }
    }

    fn supply(free: u64, clean: u64) -> Supply {
        Supply { free_beyond_target: super::super::beyond(free, 0), clean_cache: clean, cold_reclaimable: 0 }
    }

    /// The performance counters split the available lists, and count the
    /// machine's page reads beside this process's hard faults.
    #[test]
    fn the_counters_split_free_from_standby() {
        let sample = sample(&Machine::new());
        assert_eq!((sample.total, sample.supply), (64 * GB, supply(10 * GB, 30 * GB)));
        assert!(sample.harm.is_some());
    }

    /// Without the performance counters (an account outside Performance
    /// Monitor Users), the free and zero lists are the nodes' available
    /// bytes summed, the standby lists the rest of `ullAvailPhys`, and no
    /// harm is counted.
    #[test]
    fn without_the_counters_free_is_the_nodes_and_standby_the_rest() {
        let taken = sample(&Machine { counters: None, ..Machine::new() });
        assert_eq!((taken.total, taken.supply, taken.harm), (64 * GB, supply(10 * GB, 30 * GB), None));
        // One node's available bytes include its standby lists: all free.
        let one_node = Machine { counters: None, nodes: vec![Some(40 * GB)], ..Machine::new() };
        assert_eq!(sample(&one_node).supply, supply(40 * GB, 0));
        // A number naming no node adds nothing.
        let sparse = Machine { counters: None, nodes: vec![Some(5 * GB), None, Some(2 * GB)], ..Machine::new() };
        assert_eq!(sample(&sparse).supply, supply(7 * GB, 33 * GB));
    }

    /// Outside a job, no limit: the host's total and supply.
    #[test]
    fn outside_a_job_there_is_no_limit() {
        let machine = Machine::new();
        assert!(job_limits(&machine, &machine.status).is_empty());
        assert_eq!(sample(&machine).total, 64 * GB);
    }

    /// A job-wide commit limit (a process-isolated container's `--memory`)
    /// is the scope's total, its room the commit the job has left
    /// (`ullTotalPageFile − ullAvailPageFile` used of it), never an unknown.
    #[test]
    fn a_job_commit_limit_bounds_the_scope() {
        let mut machine = Machine::new().in_job(JOB_OBJECT_LIMIT_JOB_MEMORY, |job| job.JobMemoryLimit = 8 * GB as usize);
        machine.status.ullTotalPageFile = 8 * GB;
        machine.status.ullAvailPageFile = 2 * GB;
        assert_eq!(
            job_limits(&machine, &machine.status),
            [Limit { kind: ProcessLimitKind::WindowsJobCommit, limit_bytes: 8 * GB, used_bytes: 6 * GB, reclaimable_bytes: 0 }]
        );
        let sample = sample(&machine);
        assert_eq!((sample.total, sample.supply.total()), (8 * GB, 2 * GB));
    }

    /// A job's per-process commit limit binds against this process's
    /// private commit.
    #[test]
    fn a_job_process_limit_binds_against_private_commit() {
        let machine = Machine::new().in_job(JOB_OBJECT_LIMIT_PROCESS_MEMORY, |job| job.ProcessMemoryLimit = 4 * GB as usize);
        let sample = sample(&machine);
        assert_eq!((sample.total, sample.supply), (4 * GB, supply(GB, 0)));
    }

    /// A job's working-set limit is a physical scope against this process's
    /// working set.
    #[test]
    fn a_job_working_set_limit_is_a_physical_scope() {
        let machine = Machine::new().in_job(JOB_OBJECT_LIMIT_WORKINGSET, |job| {
            job.BasicLimitInformation.MaximumWorkingSetSize = 5 * GB as usize
        });
        assert_eq!(
            job_limits(&machine, &machine.status),
            [Limit { kind: ProcessLimitKind::WindowsJobWorkingSet, limit_bytes: 5 * GB, used_bytes: 2 * GB, reclaimable_bytes: 0 }]
        );
        let sample = sample(&machine);
        assert_eq!((sample.total, sample.supply), (5 * GB, supply(3 * GB, 0)));
    }

    /// Without the processor records, the flat topology of the quota's
    /// CPUs.
    #[test]
    fn without_the_records_the_processors_are_flat() {
        let flat = processors(&Machine::new());
        assert_eq!((flat.by(), flat.nodes()[0].cores.len()), (crate::ProcessorsReportedBy::Flat, 4));
    }
}
