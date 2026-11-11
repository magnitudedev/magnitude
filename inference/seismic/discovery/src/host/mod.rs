//! The host: RAM capacity, the OS build, and host RAM status with the
//! process limits that bound every path into host RAM. Host
//! modules never load GPU drivers.

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

use seismic_topology::Reported;
use std::fmt;

/// Host RAM capacity: RAM the OS can use.
pub(crate) fn capacity() -> Result<Reported<u64>, String> {
    platform::capacity()
}

/// The OS version and build, as the OS reports it: the driver of the CPU
/// executor and of Metal.
pub(crate) fn os_build() -> Result<String, String> {
    platform::os_build()
}

/// The CPU as the OS reports it: the platform's identifying CPU fields
/// (Linux `/proc/cpuinfo`'s model, feature and revision fields for every
/// processor; macOS `sysctl`'s brand string, model and CPU counts;
/// Windows' processor architecture, count, level and revision with its
/// CPUID vendor, signature and feature, and brand leaves), then
/// `;isa=<tier>` for the highest native tier it supports, where it supports
/// one, and `;cache=<bytes>` for its last-level `cache`
/// ([`last_level_cache`]), where the OS reports one.
pub(crate) fn cpu_identity(cache: Option<u64>) -> Result<String, String> {
    let mut identity = platform::cpu_identity()?;
    if let Some(tier) = seismic_native_cpu::Tier::detected() {
        identity.push_str(&format!(";isa={}", tier.name()));
    }
    if let Some(cache) = cache {
        identity.push_str(&format!(";cache={cache}"));
    }
    Ok(identity)
}

/// The CPU's last-level caches: the highest-level data or unified caches the
/// OS reports, each instance's size as the OS reports it summed over the
/// instances (no memory-side cache, which no OS reports): part of the CPU's
/// identity, and the host's reported cache. `None` only where the OS
/// reports no such cache, or no size or sharing for one; a cache the OS
/// reports but that cannot be read is an error, never a lower level
/// standing in for it.
pub(crate) fn last_level_cache() -> Result<Option<Reported<u64>>, String> {
    platform::last_level_cache()
}

/// The host's processors (`crate::processors`).
pub(crate) fn processors() -> Result<crate::Processors, String> {
    platform::processors()
}

/// The memory node of the PCI device at `location`
/// (`crate::processors::device_node`).
pub(crate) fn device_node(location: crate::PciLocation) -> Result<crate::DeviceNode, String> {
    platform::device_node(location)
}

/// Place every thread of the process on `cpus`
/// (`crate::processors::place_process`).
pub(crate) fn place_process(cpus: &[usize]) -> Result<(), String> {
    platform::place_process(cpus)
}

/// Place the calling thread on `cpus` (`crate::processors::pin`).
pub(crate) fn pin(cpus: &[usize]) -> Result<(), String> {
    platform::pin(cpus)
}

/// One host RAM status read now: what every claim of host RAM and every
/// observation between claims reads, so a decision never stands on a
/// reading older than itself.
pub(crate) fn status() -> Result<HostStatus, ObservationError> {
    platform::sample()
}

/// Host RAM this process can take now under `reclaim`: one reading's
/// [`HostStatus::allocatable_bytes`]. What a one-off taker of host RAM that
/// must not press the host bounds itself by ([`Reclaim::Nothing`]).
pub fn host_room(reclaim: Reclaim) -> Result<u64, ObservationError> {
    Ok(status()?.allocatable_bytes(reclaim))
}

/// Host RAM's one available reading: pages the OS can hand out without
/// swapping or compressing anyone, and the total they are of. The only host
/// memory figure the memory ledger reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostAvailable {
    pub total: u64,
    pub available: u64,
    /// Clean file-backed pages beyond `available` that the OS drops once for
    /// a taker ([`HostStatus::reclaimable`]); `available + reclaimable ≤
    /// total`. Zero where `available` already counts them.
    pub reclaimable: u64,
}

/// Host RAM's available reading now, the same definition on every OS: its
/// own figure (macOS `free_count − speculative_count + purgeable_count` less `vm.vm_page_free_target`,
/// Linux `MemAvailable`, Windows `ullAvailPhys`), bounded by the cgroup or job
/// limit the process lies in where one is set and binds harder.
pub fn host_available() -> Result<HostAvailable, ObservationError> {
    let capacity = capacity().map_err(ObservationError::Failed)?.value;
    let sample = status()?;
    let within = available_within(capacity, sample.allocatable.value, &sample.limits);
    Ok(HostAvailable { reclaimable: sample.reclaimable.min(within.total.saturating_sub(within.available)), ..within })
}

/// What `available` bytes of `capacity` leave the process within `limits`:
/// the binding one of the host's figure and each physical-memory limit's
/// room (its limit less the usage it cannot reclaim), each with the total
/// it is of.
fn available_within(capacity: u64, available: u64, limits: &[ProcessMemoryLimit]) -> HostAvailable {
    limits
        .iter()
        .filter(|limit| {
            matches!(
                limit.kind,
                ProcessLimitKind::CgroupV1 { .. } | ProcessLimitKind::CgroupV2 { .. } | ProcessLimitKind::WindowsJobProcess
            )
        })
        .map(|limit| HostAvailable { total: limit.limit_bytes.min(capacity), available: limit.remaining_bytes(), reclaimable: 0 })
        .fold(HostAvailable { total: capacity, available, reclaimable: 0 }, |least, bound| {
            if bound.available < least.available { bound } else { least }
        })
}

/// RAM the OS holds that it cannot make available to a process, so not
/// counted in [`HostStatus::allocatable`]; on a platform that counts it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostHeld {
    /// Wired memory: pinned by the kernel and drivers, GPU buffers included
    /// on unified memory.
    pub wired_bytes: u64,
    /// Physical memory the compressor occupies.
    pub compressed_bytes: u64,
}

/// What a taker of host RAM may count as room beyond the free memory above
/// the OS's floor, stated by every caller of a reading: a growing process
/// that must never displace another program's data takes
/// [`Self::Nothing`]; a load, which is the user's own action, takes the
/// clean file cache once ([`Self::CleanFileCache`]), as it did before a
/// growth was told apart from a load.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Reclaim {
    /// Only what the OS hands out without reclaiming anything: the free
    /// room of [`HostStatus::allocatable`].
    Nothing,
    /// Also the clean file-backed pages the OS drops once for a taker
    /// ([`HostStatus::reclaimable`]).
    CleanFileCache,
}

/// Host RAM's status in one sample.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostStatus {
    /// RAM this process can take without the OS displacing another
    /// program's data (compressing or swapping it): what a process that
    /// grows and gives back while others run may take. An estimate, never
    /// exact physical availability. The OSes differ in what that includes:
    /// Linux reclaims file cache within its watermarks before it swaps, so
    /// its `MemAvailable` counts reclaimable file cache; macOS reclaims by
    /// page age across anonymous and file-backed pages alike, so only its
    /// free and purgeable pages above the free-page target its pageout
    /// daemon refills are taken without displacing anyone.
    pub allocatable: Reported<u64>,
    /// Clean file-backed pages the OS drops once for a taker but only beside
    /// the age-ordered reclaim of other programs' data, so `allocatable`
    /// leaves them out: macOS's file-backed pages (`external_page_count`).
    /// Zero where `allocatable` already counts the file cache the kernel
    /// reclaims first (Linux's `MemAvailable`, Windows's `ullAvailPhys`).
    /// Only a load counts them ([`Reclaim::CleanFileCache`]).
    pub reclaimable: u64,
    /// The OS limits on this process: grants on host RAM that every path
    /// into it shares.
    pub limits: Vec<ProcessMemoryLimit>,
    /// Whether every applicable limit could be observed.
    pub limit_visibility: LimitVisibility,
    /// What the OS holds that `allocatable` does not count ([`HostHeld`]).
    /// `None` on a platform that counts none.
    pub held: Option<HostHeld>,
}

impl HostStatus {
    /// [`Self::allocatable`], and with [`Reclaim::CleanFileCache`] the
    /// [`Self::reclaimable`] pages too, within every process limit.
    pub fn allocatable_bytes(&self, reclaim: Reclaim) -> u64 {
        let room = match reclaim {
            Reclaim::Nothing => self.allocatable.value,
            Reclaim::CleanFileCache => self.allocatable.value.saturating_add(self.reclaimable),
        };
        self.limits.iter().fold(room, |bytes, limit| bytes.min(limit.remaining_bytes()))
    }
}

/// Why a status reading could not be taken.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ObservationError {
    /// This platform or configuration offers no qualified facility for the
    /// observation.
    Unsupported(String),
    /// The native query failed.
    Failed(String),
}

impl fmt::Display for ObservationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(message) => write!(f, "observation unsupported: {message}"),
            Self::Failed(message) => write!(f, "observation failed: {message}"),
        }
    }
}

impl std::error::Error for ObservationError {}

/// One applicable limit with the usage it is enforced against.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessMemoryLimit {
    /// Which limit it is.
    pub kind: ProcessLimitKind,
    /// The limit.
    pub limit_bytes: u64,
    /// The usage the limit is enforced against.
    pub used_bytes: u64,
    /// Clean inactive file cache the OS can reclaim within this limit. Zero
    /// for address-space, data-segment and committed-memory limits, where
    /// reclaiming physical pages does not release the accounted resource.
    pub reclaimable_bytes: u64,
}

impl ProcessMemoryLimit {
    /// What the limit has left: the limit less the usage the OS cannot
    /// reclaim within it.
    pub fn remaining_bytes(&self) -> u64 {
        self.limit_bytes
            .saturating_sub(self.used_bytes.saturating_sub(self.reclaimable_bytes))
    }
}

/// An OS limit on this process's memory.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum ProcessLimitKind {
    /// cgroup v1 `memory.limit_in_bytes` of this process's cgroup or a
    /// visible ancestor in the v1 memory hierarchy, enforced against that
    /// cgroup's `memory.usage_in_bytes`.
    CgroupV1 { cgroup: String },
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

/// Whether a sample's process limits are all the limits that apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum LimitVisibility {
    /// Every limit applicable to this process was observed.
    Complete,
    /// This process's cgroup namespace hides ancestors above its root.
    /// Hidden ancestor limits cannot be presumed unlimited.
    CgroupAncestorsHidden,
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    /// macOS's allocatable memory is its free pages less read-ahead and
    /// its purgeable pages, above the free-page target: never its
    /// file-backed pages, which the OS reclaims only beside idle programs'
    /// anonymous pages.
    #[test]
    fn macos_allocatable_is_free_and_purgeable_pages_above_the_target() {
        let reading = macos::vm_reading().unwrap();
        let pages = &reading.statistics;
        let allocatable = (u64::from(pages.free_count) - u64::from(pages.speculative_count)
            + u64::from(pages.purgeable_count))
        .saturating_sub(reading.free_target);
        let sample = macos::sample_of(&reading).unwrap();
        assert_eq!(
            sample.allocatable,
            Reported::new(
                allocatable * reading.page_size,
                seismic_topology::ReportedBy::MachAllocatablePages
            )
        );
        assert_eq!(sample.reclaimable, u64::from(pages.external_page_count) * reading.page_size);
    }
}

#[cfg(test)]
mod available_tests {
    use super::*;

    const GB: u64 = 1 << 30;

    fn cgroup(limit: u64, used: u64, reclaimable: u64) -> ProcessMemoryLimit {
        ProcessMemoryLimit {
            kind: ProcessLimitKind::CgroupV2 { cgroup: "/load".into() },
            limit_bytes: limit,
            used_bytes: used,
            reclaimable_bytes: reclaimable,
        }
    }

    /// Item 59: a cgroup's file cache is available within it, all of it
    /// the kernel can reclaim, not only its inactive part.
    #[test]
    fn a_cgroups_reclaimable_file_cache_is_available() {
        let within = available_within(64 * GB, 40 * GB, &[cgroup(32 * GB, 30 * GB, 10 * GB)]);
        assert_eq!(within, HostAvailable { total: 32 * GB, available: 12 * GB, reclaimable: 0 });
    }

    /// Only a load counts the clean file cache as room, and every process
    /// limit binds both.
    #[test]
    fn only_a_load_counts_the_clean_file_cache() {
        let status = HostStatus {
            allocatable: Reported::new(4 * GB, seismic_topology::ReportedBy::MachAllocatablePages),
            reclaimable: 6 * GB,
            limits: Vec::new(),
            limit_visibility: LimitVisibility::Complete,
            held: None,
        };
        assert_eq!(status.allocatable_bytes(Reclaim::Nothing), 4 * GB);
        assert_eq!(status.allocatable_bytes(Reclaim::CleanFileCache), 10 * GB);
        let limited = HostStatus { limits: vec![cgroup(32 * GB, 30 * GB, 0)], ..status };
        assert_eq!(limited.allocatable_bytes(Reclaim::CleanFileCache), 2 * GB);
    }

    /// The host's own figure binds where it is lower than every limit's
    /// room, and a virtual-memory limit never bounds physical memory.
    #[test]
    fn the_binding_bound_gives_the_reading() {
        let address_space = ProcessMemoryLimit {
            kind: ProcessLimitKind::AddressSpace,
            limit_bytes: 1 * GB,
            used_bytes: 1 * GB,
            reclaimable_bytes: 0,
        };
        let within = available_within(64 * GB, 8 * GB, &[cgroup(32 * GB, 10 * GB, 0), address_space]);
        assert_eq!(within, HostAvailable { total: 64 * GB, available: 8 * GB, reclaimable: 0 });
    }
}
