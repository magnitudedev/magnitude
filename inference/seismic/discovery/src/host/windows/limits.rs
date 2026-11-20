//! Windows job-object memory limits.

use super::super::{Limit, ProcessLimitKind};
use super::Sys;
use windows_sys::Win32::System::JobObjects::{
    JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_PROCESS_MEMORY, JOB_OBJECT_LIMIT_WORKINGSET,
};
use windows_sys::Win32::System::SystemInformation::MEMORYSTATUSEX;

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
pub(super) fn job_limits(sys: &impl Sys, status: &MEMORYSTATUSEX) -> Vec<Limit> {
    let Some(job) = sys.job() else {
        return Vec::new();
    };
    let flags = job.BasicLimitInformation.LimitFlags;
    let set = |flag: u32| flags & flag != 0;
    let process = sys.memory_counters();
    let mut limits = Vec::new();
    if set(JOB_OBJECT_LIMIT_JOB_MEMORY) {
        limits.push(Limit {
            kind: ProcessLimitKind::WindowsJobCommit,
            limit_bytes: job.JobMemoryLimit as u64,
            used_bytes: status
                .ullTotalPageFile
                .saturating_sub(status.ullAvailPageFile),
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
