//! The one host platform implementation: host RAM capacity and scoped host
//! memory observations. Host modules never load GPU drivers.

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

use super::{CapacityBasis, ObservationError};
use std::time::SystemTime;

/// Established host RAM capacity and the definition of that figure.
pub(crate) struct HostCapacity {
    pub(crate) bytes: u64,
    pub(crate) basis: CapacityBasis,
}

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
pub(crate) fn capacity() -> Result<HostCapacity, String> {
    platform::capacity()
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
pub(crate) fn capacity() -> Result<HostCapacity, String> {
    Err("host memory discovery is not implemented for this operating system".into())
}

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
pub(crate) fn status() -> Result<HostMemoryStatus, ObservationError> {
    platform::status()
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
pub(crate) fn status() -> Result<HostMemoryStatus, ObservationError> {
    Err(ObservationError::Unsupported(
        "host memory observation is not implemented for this operating system".into(),
    ))
}

/// One sampled host memory observation. Its fields share one sample time;
/// device observations are separate samples, not one atomic snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostMemoryStatus {
    pub sampled_at: SystemTime,
    /// Native counters exactly as the platform names them.
    pub measurements: HostMeasurements,
    /// The platform's defined headroom estimate. It is an estimate of
    /// allocatable RAM, never exact physical availability.
    pub headroom: HeadroomEstimate,
    /// Process and environment limits that apply to this process.
    pub limits: Vec<ProcessMemoryLimit>,
    /// Whether every applicable limit could be observed.
    pub limit_visibility: LimitVisibility,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostMeasurements {
    /// Mach `HOST_VM_INFO64` page counters and the page size they use.
    MacOs {
        page_size_bytes: u64,
        /// Free pages, including speculative pages.
        free_pages: u64,
        speculative_pages: u64,
        active_pages: u64,
        inactive_pages: u64,
        wired_pages: u64,
        purgeable_pages: u64,
        /// Pages occupied by the compressor (compressed storage, not the
        /// uncompressed size it holds).
        compressor_pages: u64,
        /// File-backed pages.
        external_pages: u64,
        /// Anonymous pages.
        internal_pages: u64,
    },
    /// `/proc/meminfo`.
    Linux {
        mem_free_bytes: u64,
        /// The kernel's estimate of memory available for new workloads.
        mem_available_bytes: u64,
    },
    /// `GlobalMemoryStatusEx`.
    Windows {
        available_physical_bytes: u64,
        /// Commit limit (physical RAM plus page files) for this process.
        commit_limit_bytes: u64,
        /// Commit this process can still make.
        available_commit_bytes: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadroomEstimate {
    pub bytes: u64,
    pub basis: HeadroomBasis,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadroomBasis {
    /// macOS: (free + inactive pages) × page size. Inactive pages can be
    /// reclaimed or compressed; wired, active and compressor pages cannot be
    /// assumed reclaimable.
    MachFreeAndInactivePages,
    /// Linux `MemAvailable`.
    LinuxMemAvailable,
    /// Windows: the lesser of available physical memory and available commit.
    WindowsPhysicalAndCommit,
}

/// One applicable limit with the usage it is enforced against.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessMemoryLimit {
    pub kind: ProcessLimitKind,
    pub limit_bytes: u64,
    pub used_bytes: u64,
}

impl ProcessMemoryLimit {
    pub fn remaining_bytes(&self) -> u64 {
        self.limit_bytes.saturating_sub(self.used_bytes)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProcessLimitKind {
    /// cgroup v2 `memory.max` of this process's cgroup or a visible
    /// ancestor, enforced against that cgroup's `memory.current`.
    CgroupV2 { cgroup: String },
    /// `RLIMIT_AS`, enforced against the process's virtual size.
    AddressSpace,
    /// `RLIMIT_DATA`, enforced against the process's data segment size.
    DataSegment,
    /// A Windows job's per-process committed-memory limit, enforced against
    /// this process's private commit.
    WindowsJobProcess,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitVisibility {
    /// Every limit applicable to this process was observed.
    Complete,
    /// This process's cgroup namespace hides ancestors above its root.
    /// Hidden ancestor limits cannot be presumed unlimited.
    CgroupAncestorsHidden,
}
