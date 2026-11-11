//! Windows x64: `GlobalMemoryStatusEx` and job-object memory limits.
//!
//! Physical RAM, commit limit and page files remain separate. The commit
//! figures are process-scoped ("the maximum this process can commit").

use super::{HostStatus, LimitVisibility, ObservationError, ProcessLimitKind, ProcessMemoryLimit};
use seismic_topology::{Reported, ReportedBy};
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

/// The records `GetLogicalProcessorInformationEx(relationship)` writes,
/// each `Size` bytes, in a buffer aligned for them.
fn processor_records(
    relationship: windows_sys::Win32::System::SystemInformation::LOGICAL_PROCESSOR_RELATIONSHIP,
) -> Result<Vec<u64>, String> {
    use windows_sys::Win32::System::SystemInformation::GetLogicalProcessorInformationEx;
    let mut length = 0u32;
    unsafe { GetLogicalProcessorInformationEx(relationship, std::ptr::null_mut(), &mut length) };
    let mut buffer = vec![0u64; (length as usize).div_ceil(8)];
    if unsafe {
        GetLogicalProcessorInformationEx(relationship, buffer.as_mut_ptr().cast(), &mut length)
    } == 0
    {
        return Err(format!(
            "GetLogicalProcessorInformationEx({relationship}) failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    buffer.truncate((length as usize).div_ceil(8));
    Ok(buffer)
}

/// Each record of `buffer` ([`processor_records`]), in order.
fn each_record(
    buffer: &[u64],
    mut each: impl FnMut(
        &windows_sys::Win32::System::SystemInformation::SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    ),
) {
    let (bytes, length) = (buffer.as_ptr().cast::<u8>(), buffer.len() * 8);
    let mut offset = 0usize;
    while offset < length {
        // SAFETY: the OS wrote consecutive records, each `Size` bytes long.
        let record = unsafe {
            &*(bytes.add(offset) as *const windows_sys::Win32::System::SystemInformation::SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)
        };
        if record.Size == 0 {
            break;
        }
        each(record);
        offset += record.Size as usize;
    }
}

/// The CPUs of `masks`, each its group's 64 times its group plus its bit.
fn group_cpus(
    masks: &[windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY],
) -> Vec<usize> {
    let mut cpus = masks
        .iter()
        .flat_map(|mask| {
            (0..usize::BITS as usize)
                .filter(move |bit| mask.Mask >> bit & 1 == 1)
                .map(move |bit| usize::from(mask.Group) * 64 + bit)
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
/// affinity mask where it has one group (`GetProcessAffinityMask`).
pub(super) fn processors() -> Result<crate::Processors, String> {
    use windows_sys::Win32::System::SystemInformation::{RelationNumaNode, RelationProcessorCore};
    use windows_sys::Win32::System::Threading::{
        GetNumaAvailableMemoryNodeEx, GetProcessAffinityMask, GetProcessGroupAffinity,
    };
    let mut cores = Vec::new();
    each_record(&processor_records(RelationProcessorCore)?, |record| {
        // SAFETY: a `RelationProcessorCore` record's union is its processor.
        let core = unsafe { &record.Anonymous.Processor };
        // SAFETY: the record holds `GroupCount` masks.
        let masks = unsafe {
            std::slice::from_raw_parts(core.GroupMask.as_ptr(), usize::from(core.GroupCount))
        };
        cores.push((group_cpus(masks), core.EfficiencyClass));
    });
    let fastest = cores
        .iter()
        .map(|(_, class)| *class)
        .max()
        .ok_or("Windows reports no processor core")?;
    let mut nodes = Vec::new();
    let mut failed = None;
    each_record(&processor_records(RelationNumaNode)?, |record| {
        // SAFETY: a `RelationNumaNode` record's union is its node, holding
        // `GroupCount` masks (one where it reports none).
        let node = unsafe { &record.Anonymous.NumaNode };
        let masks = unsafe {
            std::slice::from_raw_parts(
                node.Anonymous.GroupMasks.as_ptr(),
                usize::from(node.GroupCount).max(1),
            )
        };
        let cpus = group_cpus(masks);
        let mut available = 0u64;
        let Ok(number) = u16::try_from(node.NodeNumber) else {
            failed.get_or_insert(format!("NUMA node {}", node.NodeNumber));
            return;
        };
        if unsafe { GetNumaAvailableMemoryNodeEx(number, &mut available) } == 0 {
            failed.get_or_insert(format!(
                "GetNumaAvailableMemoryNodeEx({number}): {}",
                std::io::Error::last_os_error()
            ));
            return;
        }
        let cores = cores
            .iter()
            .filter(|(threads, _)| threads.first().is_some_and(|first| cpus.contains(first)))
            .map(|(threads, class)| crate::Core {
                threads: threads.clone(),
                performance: *class == fastest,
            })
            .collect();
        nodes.push(crate::MemoryNode {
            id: usize::from(number),
            memory: available > 0,
            cores,
        });
    });
    if let Some(failed) = failed {
        return Err(failed);
    }
    let process = unsafe { GetCurrentProcess() };
    let mut count = 0u16;
    unsafe { GetProcessGroupAffinity(process, &mut count, std::ptr::null_mut()) };
    let mut groups = vec![0u16; usize::from(count)];
    if unsafe { GetProcessGroupAffinity(process, &mut count, groups.as_mut_ptr()) } == 0 {
        return Err(format!(
            "GetProcessGroupAffinity: {}",
            std::io::Error::last_os_error()
        ));
    }
    let every = cores
        .iter()
        .flat_map(|(threads, _)| threads.iter().copied());
    let allowed = match groups.as_slice() {
        [group] => {
            let (mut mask, mut system) = (0usize, 0usize);
            if unsafe { GetProcessAffinityMask(process, &mut mask, &mut system) } == 0 {
                return Err(format!(
                    "GetProcessAffinityMask: {}",
                    std::io::Error::last_os_error()
                ));
            }
            every
                .filter(|cpu| cpu / 64 == usize::from(*group) && mask >> (cpu % 64) & 1 == 1)
                .collect()
        }
        groups => every
            .filter(|cpu| groups.iter().any(|group| cpu / 64 == usize::from(*group)))
            .collect(),
    };
    crate::Processors::new(
        nodes,
        allowed,
        crate::ProcessorsReportedBy::WindowsProcessorInformation,
    )
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
    let mask = cpus
        .iter()
        .filter(|&&cpu| cpu / 64 == group)
        .fold(0usize, |mask, &cpu| mask | 1 << (cpu % 64));
    let group = u16::try_from(group).map_err(|_| format!("CPUs {cpus:?} beyond every group"))?;
    let affinity = GROUP_AFFINITY {
        Mask: mask,
        Group: group,
        Reserved: [0; 3],
    };
    match unsafe { SetThreadGroupAffinity(GetCurrentThread(), &affinity, std::ptr::null_mut()) } {
        0 => Err(format!(
            "SetThreadGroupAffinity to CPUs {cpus:?}: {}",
            std::io::Error::last_os_error()
        )),
        _ => Ok(()),
    }
}

pub(super) fn sample() -> Result<HostStatus, ObservationError> {
    let status = memory_status().map_err(ObservationError::Failed)?;
    Ok(HostStatus {
        // Available physical memory is standby and free pages, neither of
        // which holds another program's working set.
        allocatable: Reported::new(
            status.ullAvailPhys.min(status.ullAvailPageFile),
            ReportedBy::WindowsPhysicalAndCommit,
        ),
        reclaimable: 0,
        limits: job_limits()?,
        limit_visibility: LimitVisibility::Complete,
        held: None,
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

/// No device's memory node is read here.
pub(super) fn device_node(_location: crate::PciLocation) -> Result<crate::DeviceNode, String> {
    Ok(crate::DeviceNode::Unknown)
}
