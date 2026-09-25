//! Windows x64: `GlobalMemoryStatusEx` and job-object memory limits.
//!
//! Physical RAM, commit limit and page files remain separate. The commit
//! figures are process-scoped ("the maximum this process can commit").

use super::{
    HeadroomBasis, HeadroomEstimate, HostCapacity, HostMeasurements, HostMemoryStatus,
    LimitVisibility, ProcessLimitKind, ProcessMemoryLimit,
};
use crate::devices::{CapacityBasis, ObservationError};
use std::time::SystemTime;
use windows_sys::Win32::System::JobObjects::{
    IsProcessInJob, JobObjectExtendedLimitInformation, QueryInformationJobObject,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_JOB_MEMORY,
    JOB_OBJECT_LIMIT_PROCESS_MEMORY,
};
use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

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

pub(super) fn capacity() -> Result<HostCapacity, String> {
    let status = memory_status()?;
    Ok(HostCapacity {
        bytes: status.ullTotalPhys,
        basis: CapacityBasis::OsUsableRam,
    })
}

pub(super) fn status() -> Result<HostMemoryStatus, ObservationError> {
    let status = memory_status().map_err(ObservationError::Failed)?;
    let sampled_at = SystemTime::now();
    Ok(HostMemoryStatus {
        sampled_at,
        measurements: HostMeasurements::Windows {
            available_physical_bytes: status.ullAvailPhys,
            commit_limit_bytes: status.ullTotalPageFile,
            available_commit_bytes: status.ullAvailPageFile,
        },
        headroom: HeadroomEstimate {
            bytes: status.ullAvailPhys.min(status.ullAvailPageFile),
            basis: HeadroomBasis::WindowsPhysicalAndCommit,
        },
        pressure: None,
        limits: job_limits()?,
        limit_visibility: LimitVisibility::Complete,
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
    }])
}
