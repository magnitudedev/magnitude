//! Windows x64: `GlobalMemoryStatusEx` and job-object memory limits.
//!
//! Physical RAM, commit limit and page files remain separate. The commit
//! figures are process-scoped ("the maximum this process can commit").

use super::{HostSample, LimitVisibility, ObservationError, ProcessLimitKind, ProcessMemoryLimit};
use crate::model::{Reported, ReportedBy};
use windows_sys::Wdk::System::SystemServices::RtlGetVersion;
use windows_sys::Win32::System::JobObjects::{
    IsProcessInJob, JobObjectExtendedLimitInformation, QueryInformationJobObject,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_JOB_MEMORY,
    JOB_OBJECT_LIMIT_PROCESS_MEMORY,
};
use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
use windows_sys::Win32::System::SystemInformation::{
    GlobalMemoryStatusEx, MEMORYSTATUSEX, OSVERSIONINFOW,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

pub(super) fn cpu_identity() -> Result<String, String> {
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
    .map_err(|error| error.to_string())
}

fn memory_status() -> Result<MEMORYSTATUSEX, String> {
    let mut status = MEMORYSTATUSEX {
        dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
        return Err(format!(
            "GlobalMemoryStatusEx failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(status)
}

pub(super) fn capacity() -> Result<Reported<u64>, String> {
    let status = memory_status()?;
    Ok(Reported::new(
        status.ullTotalPhys,
        ReportedBy::WindowsTotalPhys,
    ))
}

/// `RtlGetVersion`: the true version and build, which `GetVersionEx` reports
/// only to a manifested process.
pub(super) fn os_build() -> Result<String, String> {
    let mut version = OSVERSIONINFOW {
        dwOSVersionInfoSize: std::mem::size_of::<OSVERSIONINFOW>() as u32,
        ..unsafe { std::mem::zeroed() }
    };
    let status = unsafe { RtlGetVersion(&mut version) };
    if status != 0 {
        return Err(format!("RtlGetVersion returned NTSTATUS {status:#x}"));
    }
    Ok(format!(
        "Windows {}.{}.{}",
        version.dwMajorVersion, version.dwMinorVersion, version.dwBuildNumber
    ))
}

/// `GetLogicalProcessorInformationEx(RelationCache)`: the highest-level data
/// or unified caches, one record per instance, summed; `None` when the OS
/// describes none. A failed call is an error.
pub(super) fn last_level_cache() -> Result<Option<Reported<u64>>, String> {
    use windows_sys::Win32::System::SystemInformation::{
        CacheData, CacheUnified, GetLogicalProcessorInformationEx, RelationCache,
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };
    let mut length = 0u32;
    unsafe { GetLogicalProcessorInformationEx(RelationCache, std::ptr::null_mut(), &mut length) };
    // Records are variable-length; a u64 buffer aligns each.
    let mut buffer = vec![0u64; (length as usize).div_ceil(8)];
    if unsafe {
        GetLogicalProcessorInformationEx(RelationCache, buffer.as_mut_ptr().cast(), &mut length)
    } == 0
    {
        return Err(format!(
            "GetLogicalProcessorInformationEx(RelationCache) failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    let bytes = buffer.as_ptr().cast::<u8>();
    let mut offset = 0usize;
    // The highest level seen, and its instances' bytes together.
    let mut last: Option<(u8, u64)> = None;
    while offset < length as usize {
        // SAFETY: the OS wrote `length` bytes of consecutive records, each
        // `Size` bytes long.
        let record =
            unsafe { &*(bytes.add(offset) as *const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) };
        if record.Relationship == RelationCache {
            let cache = unsafe { &record.Anonymous.Cache };
            if matches!(cache.Type, CacheData | CacheUnified) {
                let size = u64::from(cache.CacheSize);
                last = match last {
                    Some((level, total)) if level == cache.Level => Some((level, total + size)),
                    Some((level, _)) if level > cache.Level => last,
                    _ => Some((cache.Level, size)),
                };
            }
        }
        offset += record.Size as usize;
    }
    Ok(last.map(|(_, total)| Reported::new(total, ReportedBy::WindowsProcessorCache)))
}

pub(super) fn sample() -> Result<HostSample, ObservationError> {
    let status = memory_status().map_err(ObservationError::Failed)?;
    Ok(HostSample {
        headroom: Reported::new(
            status.ullAvailPhys.min(status.ullAvailPageFile),
            ReportedBy::WindowsPhysicalAndCommit,
        ),
        limits: job_limits()?,
        limit_visibility: LimitVisibility::Complete,
        displacement: None,
        kernel_pressure: None,
    })
}

fn job_limits() -> Result<Vec<ProcessMemoryLimit>, ObservationError> {
    let process = unsafe { GetCurrentProcess() };
    let mut in_job = 0;
    if unsafe { IsProcessInJob(process, std::ptr::null_mut(), &mut in_job) } == 0 {
        return Err(ObservationError::Failed(format!(
            "IsProcessInJob failed: {}",
            std::io::Error::last_os_error()
        )));
    }
    if in_job == 0 {
        return Ok(Vec::new());
    }
    let mut information: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
    let queried = unsafe {
        QueryInformationJobObject(
            std::ptr::null_mut(),
            JobObjectExtendedLimitInformation,
            (&mut information as *mut JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            std::ptr::null_mut(),
        )
    };
    if queried == 0 {
        return Err(ObservationError::Failed(format!(
            "QueryInformationJobObject failed: {}",
            std::io::Error::last_os_error()
        )));
    }
    let flags = information.BasicLimitInformation.LimitFlags;
    if flags & JOB_OBJECT_LIMIT_JOB_MEMORY != 0 {
        // No documented API reports the job's current committed total, so the
        // remaining job-wide allowance cannot be established.
        return Err(ObservationError::Unsupported(
            "a job-wide committed-memory limit applies, but its current usage is not observable"
                .into(),
        ));
    }
    if flags & JOB_OBJECT_LIMIT_PROCESS_MEMORY == 0 {
        return Ok(Vec::new());
    }
    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    if unsafe { K32GetProcessMemoryInfo(process, &mut counters, counters.cb) } == 0 {
        return Err(ObservationError::Failed(format!(
            "GetProcessMemoryInfo failed: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok(vec![ProcessMemoryLimit {
        kind: ProcessLimitKind::WindowsJobProcess,
        limit_bytes: information.ProcessMemoryLimit as u64,
        used_bytes: counters.PagefileUsage as u64,
        reclaimable_bytes: 0,
    }])
}
