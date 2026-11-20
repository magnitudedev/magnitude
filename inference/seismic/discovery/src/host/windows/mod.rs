//! Windows x64: `GlobalMemoryStatusEx`, the memory performance counters,
//! the memory nodes' free pages, job-object memory limits, hard faults and
//! the low-memory resource notification, each read through [`Sys`].
//!
//! Physical RAM, commit limit and page files remain separate. The commit
//! figures are process-scoped ("the maximum this process can commit"):
//! within a job, they are the job's.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use super::{read, resolve};
use memory::{Counters, Query, COUNTER_PATHS};
use seismic_topology::{CacheSize, Reported, ReportedBy};
use std::num::NonZeroUsize;
use windows_sys::Wdk::System::SystemServices::RtlGetVersion;
use windows_sys::Win32::System::JobObjects::{
    IsProcessInJob, JobObjectExtendedLimitInformation, QueryInformationJobObject,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
};
use windows_sys::Win32::System::Memory::QueryMemoryResourceNotification;
use windows_sys::Win32::System::ProcessStatus::{
    K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX2,
};
use windows_sys::Win32::System::SystemInformation::{
    GlobalMemoryStatusEx, LOGICAL_PROCESSOR_RELATIONSHIP, MEMORYSTATUSEX, OSVERSIONINFOW,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// Every OS read discovery makes on Windows, its one owner. A read some
/// machines refuse or lack is the value or none ([`read`]: the performance
/// counters to an account outside Performance Monitor Users, the private
/// working set before Windows 10 1809, a job's limits); a call that answers
/// on every Windows is its value, and its failure an invariant violation.
pub(crate) trait Sys {
    /// `GlobalMemoryStatusEx`.
    fn memory_status(&self) -> MEMORYSTATUSEX;
    /// `RtlGetVersion`: the true version and build, which `GetVersionEx`
    /// reports only to a manifested process.
    fn os_version(&self) -> OSVERSIONINFOW;
    /// The processor's identifying fields ([`super::cpu_identity`]):
    /// `GetNativeSystemInfo`'s architecture, count, level and revision with
    /// CPUID's vendor, signature and feature, and brand leaves.
    fn cpu_identity(&self) -> String;
    /// Human-readable CPU brand string from CPUID extended leaves; none when
    /// unavailable.
    fn cpu_brand(&self) -> Option<String>;
    /// The records `GetLogicalProcessorInformationEx(relationship)` writes,
    /// each `Size` bytes, in a buffer aligned for them with exact byte length.
    fn processor_records(&self, relationship: LOGICAL_PROCESSOR_RELATIONSHIP) -> Option<topology::ProcessorRecords>;
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
    /// The raw SMBIOS structure table (`GetSystemFirmwareTable(RSMB)` after
    /// its eight-byte header); none where the OS returns none.
    fn smbios_table(&self) -> Option<Vec<u8>>;
    /// The system's page (`GetSystemInfo`'s `dwPageSize`, through the one
    /// owner of the host page, `seismic_memory::host_page`).
    fn page(&self) -> u64;
    /// The CPUs the process may keep busy at once
    /// (`seismic_threads::quota`, their one owner: `available_parallelism`,
    /// its affinity under a cgroup's CPU quota, or one).
    fn parallelism(&self) -> NonZeroUsize;
}

/// The running OS's readings.
pub(crate) struct LiveSys;

/// The last OS error as the outcome of the read `what`.
fn last_error<T>(what: &str) -> Option<T> {
    read(what, Err(std::io::Error::last_os_error()))
}

/// A Win32 call that answers on every Windows: `answered` true, or an
/// invariant violation naming `what`.
fn always(what: &str, answered: bool) {
    assert!(
        answered,
        "invariant violated: {what}: {}",
        std::io::Error::last_os_error()
    );
}

/// CPUID brand string (leaves `0x8000_0002..=0x8000_0004`), trimmed; none
/// where the CPU does not support the extended brand leaves or on non-x86_64.
fn cpuid_brand_string() -> Option<String> {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::__cpuid;
        if __cpuid(0x8000_0000).eax < 0x8000_0004 {
            return None;
        }
        let mut bytes = [0u8; 48];
        for (i, leaf) in (0x8000_0002..=0x8000_0004).enumerate() {
            let report = __cpuid(leaf);
            let base = i * 16;
            bytes[base..base + 4].copy_from_slice(&report.eax.to_le_bytes());
            bytes[base + 4..base + 8].copy_from_slice(&report.ebx.to_le_bytes());
            bytes[base + 8..base + 12].copy_from_slice(&report.ecx.to_le_bytes());
            bytes[base + 12..base + 16].copy_from_slice(&report.edx.to_le_bytes());
        }
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        let text = String::from_utf8_lossy(&bytes[..end]).trim().to_owned();
        (!text.is_empty()).then_some(text)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        None
    }
}

impl Sys for LiveSys {
    fn memory_status(&self) -> MEMORYSTATUSEX {
        let mut status = MEMORYSTATUSEX {
            dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
            ..Default::default()
        };
        always(
            "GlobalMemoryStatusEx",
            unsafe { GlobalMemoryStatusEx(&mut status) } != 0,
        );
        status
    }

    fn os_version(&self) -> OSVERSIONINFOW {
        let mut version = OSVERSIONINFOW {
            dwOSVersionInfoSize: std::mem::size_of::<OSVERSIONINFOW>() as u32,
            ..unsafe { std::mem::zeroed() }
        };
        let status = unsafe { RtlGetVersion(&mut version) };
        assert_eq!(
            status, 0,
            "invariant violated: RtlGetVersion returned NTSTATUS {status:#x}"
        );
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
        serde_json::to_string(&(
            architecture,
            system.dwNumberOfProcessors,
            system.wProcessorLevel,
            system.wProcessorRevision,
            cpu,
        ))
        .unwrap_or_else(|error| panic!("invariant violated: integers serialize: {error}"))
    }

    fn cpu_brand(&self) -> Option<String> {
        cpuid_brand_string()
    }

    fn processor_records(
        &self,
        relationship: LOGICAL_PROCESSOR_RELATIONSHIP,
    ) -> Option<topology::ProcessorRecords> {
        use windows_sys::Win32::Foundation::{ERROR_INSUFFICIENT_BUFFER, ERROR_INVALID_PARAMETER};
        use windows_sys::Win32::System::SystemInformation::{
            GetLogicalProcessorInformationEx, RelationNumaNodeEx,
        };
        let mut length = 0u32;
        if unsafe {
            GetLogicalProcessorInformationEx(relationship, std::ptr::null_mut(), &mut length)
        } != 0
        {
            return Some(topology::ProcessorRecords::new(Vec::new(), 0));
        }
        let first_err = std::io::Error::last_os_error();
        if first_err.raw_os_error() != Some(ERROR_INSUFFICIENT_BUFFER as i32) {
            if relationship == RelationNumaNodeEx
                && first_err.raw_os_error() == Some(ERROR_INVALID_PARAMETER as i32)
            {
                return None;
            }
            return read(format!("GetLogicalProcessorInformationEx({relationship})"), Err(first_err));
        }
        loop {
            let mut buffer = vec![0u64; (length as usize).div_ceil(8)];
            if unsafe {
                GetLogicalProcessorInformationEx(
                    relationship,
                    buffer.as_mut_ptr().cast(),
                    &mut length,
                )
            } != 0
            {
                let byte_len = length as usize;
                buffer.truncate(byte_len.div_ceil(8));
                return Some(topology::ProcessorRecords::new(buffer, byte_len));
            }
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() != Some(ERROR_INSUFFICIENT_BUFFER as i32) {
                return read(format!("GetLogicalProcessorInformationEx({relationship})"), Err(error));
            }
        }
    }

    fn highest_node(&self) -> u16 {
        use windows_sys::Win32::System::Threading::GetNumaHighestNodeNumber;
        let mut highest = 0u32;
        always(
            "GetNumaHighestNodeNumber",
            unsafe { GetNumaHighestNodeNumber(&mut highest) } != 0,
        );
        u16::try_from(highest)
            .unwrap_or_else(|_| panic!("invariant violated: NUMA node {highest} beyond u16"))
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
        use windows_sys::Win32::Foundation::ERROR_INSUFFICIENT_BUFFER;
        use windows_sys::Win32::System::Threading::GetProcessGroupAffinity;
        let process = unsafe { GetCurrentProcess() };
        let mut count = 0u16;
        if unsafe { GetProcessGroupAffinity(process, &mut count, std::ptr::null_mut()) } != 0 {
            return Some(Vec::new());
        }
        let first_err = std::io::Error::last_os_error();
        if first_err.raw_os_error() != Some(ERROR_INSUFFICIENT_BUFFER as i32) {
            return read("GetProcessGroupAffinity", Err(first_err));
        }
        loop {
            let mut groups = vec![0u16; usize::from(count)];
            if unsafe { GetProcessGroupAffinity(process, &mut count, groups.as_mut_ptr()) } != 0 {
                groups.truncate(usize::from(count));
                return Some(groups);
            }
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() != Some(ERROR_INSUFFICIENT_BUFFER as i32) {
                return read("GetProcessGroupAffinity", Err(error));
            }
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
            PdhAddEnglishCounterW, PdhCloseQuery, PdhCollectQueryData, PdhGetRawCounterValue,
            PdhOpenQueryW, PDH_CSTATUS_NEW_DATA, PDH_CSTATUS_VALID_DATA, PDH_HQUERY,
            PDH_RAW_COUNTER,
        };
        struct QueryGuard(PDH_HQUERY);
        impl Drop for QueryGuard {
            fn drop(&mut self) {
                if !self.0.is_null() {
                    // SAFETY: `self.0` was opened by `PdhOpenQueryW` and not yet moved out.
                    unsafe { PdhCloseQuery(self.0) };
                }
            }
        }
        // Opened once: an account refused the counters stays refused.
        static QUERY: std::sync::OnceLock<Option<std::sync::Mutex<Query>>> =
            std::sync::OnceLock::new();
        let query = QUERY.get_or_init(|| {
            let mut raw_query = std::ptr::null_mut();
            pdh("PdhOpenQueryW", unsafe {
                PdhOpenQueryW(std::ptr::null(), 0, &mut raw_query)
            })?;
            let mut guard = QueryGuard(raw_query);
            let mut counters = [std::ptr::null_mut(); 4];
            for (counter, path) in counters.iter_mut().zip(COUNTER_PATHS) {
                let wide: Vec<u16> = path.encode_utf16().chain(std::iter::once(0)).collect();
                pdh(path, unsafe {
                    PdhAddEnglishCounterW(guard.0, wide.as_ptr(), 0, counter)
                })?;
            }
            let query = std::mem::replace(&mut guard.0, std::ptr::null_mut());
            Some(std::sync::Mutex::new(Query { query, counters }))
        });
        // A collector that panicked left the query's handles whole.
        let query = query
            .as_ref()?
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        pdh("PdhCollectQueryData", unsafe {
            PdhCollectQueryData(query.query)
        })?;
        let mut values = [0u64; 4];
        for ((value, counter), path) in values.iter_mut().zip(query.counters).zip(COUNTER_PATHS) {
            let mut raw: PDH_RAW_COUNTER = unsafe { std::mem::zeroed() };
            pdh(path, unsafe {
                PdhGetRawCounterValue(counter, std::ptr::null_mut(), &mut raw)
            })?;
            if raw.CStatus != PDH_CSTATUS_VALID_DATA && raw.CStatus != PDH_CSTATUS_NEW_DATA {
                pdh(path, raw.CStatus)?;
            }
            *value = u64::try_from(raw.FirstValue)
                .unwrap_or_else(|_| panic!("invariant violated: {path} reads {}", raw.FirstValue));
        }
        let [free_zero, standby_core, normal, reserve] = values;
        Some(Counters {
            free_zero,
            standby_core,
            standby_reclaimable: normal + reserve,
        })
    }

    fn memory_counters(&self) -> PROCESS_MEMORY_COUNTERS {
        let mut counters = PROCESS_MEMORY_COUNTERS {
            cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
            ..Default::default()
        };
        always(
            "GetProcessMemoryInfo",
            unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) }
                != 0,
        );
        counters
    }

    fn private_working_set(&self) -> Option<u64> {
        /// Windows 10 1809, the first build that knows the structure.
        const EX2: u32 = 17763;
        if self.os_version().dwBuildNumber < EX2 {
            return None;
        }
        let mut counters = PROCESS_MEMORY_COUNTERS_EX2 {
            cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX2>() as u32,
            ..Default::default()
        };
        let answered = unsafe {
            K32GetProcessMemoryInfo(
                GetCurrentProcess(),
                (&mut counters as *mut PROCESS_MEMORY_COUNTERS_EX2).cast(),
                counters.cb,
            )
        };
        always(
            "GetProcessMemoryInfo(PROCESS_MEMORY_COUNTERS_EX2)",
            answered != 0,
        );
        Some(counters.PrivateWorkingSetSize as u64)
    }

    fn low_memory(&self) -> bool {
        let mut state = 0;
        let low = crate::pressure::low_memory_notification();
        always(
            "QueryMemoryResourceNotification",
            unsafe { QueryMemoryResourceNotification(low, &mut state) } != 0,
        );
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

    fn smbios_table(&self) -> Option<Vec<u8>> {
        use windows_sys::Win32::System::SystemInformation::GetSystemFirmwareTable;
        const RSMB: u32 = u32::from_be_bytes(*b"RSMB");
        let size = unsafe { GetSystemFirmwareTable(RSMB, 0, std::ptr::null_mut(), 0) };
        if !(8..=1 << 20).contains(&size) {
            return None;
        }
        let mut data = vec![0u8; size as usize];
        if unsafe { GetSystemFirmwareTable(RSMB, 0, data.as_mut_ptr().cast(), size) } != size {
            return None;
        }
        let length = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        data.get(8..8 + length).map(<[u8]>::to_vec)
    }

    fn page(&self) -> u64 {
        static PAGE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        *PAGE.get_or_init(|| {
            seismic_memory::host_page()
                .unwrap_or_else(|error| panic!("invariant violated: the system's page: {error}"))
        })
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
    use windows_sys::Win32::System::Performance::{
        PDH_ACCESS_DENIED, PDH_CSTATUS_NO_COUNTER, PDH_CSTATUS_NO_OBJECT, PDH_NO_DATA,
    };
    let outcome = match status {
        0 => Ok(()),
        PDH_ACCESS_DENIED => Err(std::io::ErrorKind::PermissionDenied.into()),
        PDH_CSTATUS_NO_OBJECT | PDH_CSTATUS_NO_COUNTER | PDH_NO_DATA => {
            Err(std::io::ErrorKind::NotFound.into())
        }
        status => Err(std::io::Error::other(format!("PDH status {status:#x}"))),
    };
    read(what, outcome)
}

/// The OS's diagnostics for the hardware report: none on Windows.
pub(super) fn diagnostics(_sys: &impl Sys) -> Vec<seismic_topology::Diagnostic> {
    Vec::new()
}

pub(super) fn cpu_identity(sys: &impl Sys) -> String {
    sys.cpu_identity()
}

/// Human-readable CPU brand string from CPUID, falling back to `"CPU (<arch>)"`.
pub(super) fn cpu_model(sys: &impl Sys) -> String {
    resolve(
        &[((), &|| sys.cpu_brand())],
        (),
        || format!("CPU ({})", std::env::consts::ARCH),
    )
    .0
}

/// `GlobalMemoryStatusEx`'s `ullTotalPhys`.
pub(super) fn capacity(sys: &impl Sys) -> Reported<u64> {
    Reported::new(
        sys.memory_status().ullTotalPhys,
        ReportedBy::WindowsTotalPhys,
    )
}

/// `RtlGetVersion`'s version and build.
pub(super) fn os_build(sys: &impl Sys) -> String {
    let version = sys.os_version();
    format!(
        "Windows {}.{}.{}",
        version.dwMajorVersion, version.dwMinorVersion, version.dwBuildNumber
    )
}

/// `GetLogicalProcessorInformationEx(RelationCache)`: the highest-level data
/// or unified caches, one record per instance, summed; [`CacheSize::UNREPORTED`]
/// where the OS describes none or refuses the records.
pub(super) fn last_level_cache(sys: &impl Sys) -> CacheSize {
    let (bytes, is_last_level) = resolve(
        &[(true, &|| relation_cache_bytes(sys))],
        false,
        || 0,
    );
    CacheSize {
        bytes,
        is_last_level,
    }
}

fn relation_cache_bytes(sys: &impl Sys) -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{CacheData, CacheUnified, RelationCache};
    let records = sys.processor_records(RelationCache)?;
    // The highest level seen, and its instances' bytes together.
    let mut last: Option<(u8, u64)> = None;
    records.caches(|cache| {
        if cache.Type == CacheData || cache.Type == CacheUnified {
            let size = u64::from(cache.CacheSize);
            last = match last {
                Some((level, total)) if level == cache.Level => Some((level, total + size)),
                Some((level, _)) if level > cache.Level => last,
                _ => Some((cache.Level, size)),
            };
        }
    });
    last.map(|(_, total)| total)
}

mod limits;
mod memory;
mod topology;
#[cfg(test)]
mod tests;

pub(crate) use memory::sample;
pub(crate) use topology::{device_node, firmware, pin, place_process, processors};
