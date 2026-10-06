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
use std::time::{Duration, Instant, SystemTime};

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

/// One host memory observation. `history` holds the displacement windows of
/// the samples taken through it; the sample closes a window when it is due.
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
pub(crate) fn status(
    history: &mut DisplacementHistory,
) -> Result<HostMemoryStatus, ObservationError> {
    let sample = platform::sample()?;
    Ok(HostMemoryStatus {
        sampled_at: sample.sampled_at,
        measurements: sample.measurements,
        headroom: sample.headroom,
        limits: sample.limits,
        limit_visibility: sample.limit_visibility,
        displacement: sample
            .displacement
            .map(|counters| history.record(Instant::now(), counters)),
        kernel_pressure: sample.kernel_pressure,
    })
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
pub(crate) fn status(
    _history: &mut DisplacementHistory,
) -> Result<HostMemoryStatus, ObservationError> {
    Err(ObservationError::Unsupported(
        "host memory observation is not implemented for this operating system".into(),
    ))
}

/// One platform sample, before the displacement history of the catalog that
/// took it is applied.
#[cfg_attr(
    not(any(target_os = "linux", target_os = "macos", target_os = "windows")),
    allow(dead_code)
)]
pub(crate) struct HostSample {
    pub(crate) sampled_at: SystemTime,
    pub(crate) measurements: HostMeasurements,
    pub(crate) headroom: HeadroomEstimate,
    pub(crate) limits: Vec<ProcessMemoryLimit>,
    pub(crate) limit_visibility: LimitVisibility,
    /// Cumulative displacement, on a platform that counts it.
    pub(crate) displacement: Option<DisplacementCounters>,
    pub(crate) kernel_pressure: Option<KernelPressure>,
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
    /// Displacement of programs' memory over the recent windows of the
    /// catalog that took this sample. `None` on a platform that counts none.
    pub displacement: Option<HostDisplacement>,
    /// The kernel's own pressure classification. `None` on a platform
    /// without one.
    pub kernel_pressure: Option<KernelPressure>,
}

/// A program's pages leaving RAM under memory demand, counted in pages
/// since boot: compressed in memory, faulted back from compression, written
/// to swap, and read back from swap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) struct DisplacementCounters {
    pub(crate) compressed_pages: u64,
    pub(crate) decompressed_pages: u64,
    pub(crate) swapped_out_pages: u64,
    pub(crate) swapped_in_pages: u64,
}

/// Displacement over one completed window: the span between two host samples
/// at least 500 ms apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DisplacementWindow {
    pub duration: Duration,
    pub compressed_pages: u64,
    pub decompressed_pages: u64,
    pub swapped_out_pages: u64,
    pub swapped_in_pages: u64,
}

/// The two most recent completed displacement windows. They are contiguous:
/// `previous` ends where `latest` begins.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostDisplacement {
    pub latest: Option<DisplacementWindow>,
    pub previous: Option<DisplacementWindow>,
}

/// The least span of one displacement window. Rates over a shorter span are
/// dominated by when the kernel happened to run its pageout work.
const MIN_DISPLACEMENT_WINDOW: Duration = Duration::from_millis(500);

/// The completed displacement windows of one catalog's host samples. Windows
/// tumble: the first sample at least [`MIN_DISPLACEMENT_WINDOW`] after a
/// window opened completes it and opens the next.
#[derive(Debug, Default)]
pub(crate) struct DisplacementHistory {
    open: Option<(Instant, DisplacementCounters)>,
    completed: HostDisplacement,
}

impl DisplacementHistory {
    pub(crate) fn record(&mut self, at: Instant, counters: DisplacementCounters) -> HostDisplacement {
        let Some((opened_at, opening)) = self.open else {
            self.open = Some((at, counters));
            return self.completed;
        };
        let duration = at.saturating_duration_since(opened_at);
        if duration >= MIN_DISPLACEMENT_WINDOW {
            self.completed = HostDisplacement {
                latest: Some(DisplacementWindow {
                    duration,
                    compressed_pages: counters
                        .compressed_pages
                        .saturating_sub(opening.compressed_pages),
                    decompressed_pages: counters
                        .decompressed_pages
                        .saturating_sub(opening.decompressed_pages),
                    swapped_out_pages: counters
                        .swapped_out_pages
                        .saturating_sub(opening.swapped_out_pages),
                    swapped_in_pages: counters
                        .swapped_in_pages
                        .saturating_sub(opening.swapped_in_pages),
                }),
                previous: self.completed.latest,
            };
            self.open = Some((at, counters));
        }
        self.completed
    }
}

/// A kernel's classification of system memory pressure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelPressure {
    Normal,
    Warning,
    Critical,
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
    /// macOS: (free + active + inactive pages) × page size, the kernel's
    /// own definition of available memory: every page that is neither wired
    /// nor occupied by the compressor. Taking a page another program holds
    /// displaces it; the displacement windows report that cost.
    MachMovablePages,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitVisibility {
    /// Every limit applicable to this process was observed.
    Complete,
    /// This process's cgroup namespace hides ancestors above its root.
    /// Hidden ancestor limits cannot be presumed unlimited.
    CgroupAncestorsHidden,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counters(compressed: u64, swapped_out: u64) -> DisplacementCounters {
        DisplacementCounters {
            compressed_pages: compressed,
            swapped_out_pages: swapped_out,
            ..DisplacementCounters::default()
        }
    }

    #[test]
    fn a_window_completes_at_the_first_sample_past_the_minimum_span() {
        let start = Instant::now();
        let mut history = DisplacementHistory::default();
        assert_eq!(history.record(start, counters(10, 0)), HostDisplacement::default());
        // Samples inside the open window complete nothing.
        let early = history.record(start + Duration::from_millis(499), counters(40, 0));
        assert_eq!(early, HostDisplacement::default());
        let first = history.record(start + Duration::from_millis(700), counters(110, 3));
        let latest = first.latest.unwrap();
        assert_eq!(latest.duration, Duration::from_millis(700));
        assert_eq!(latest.compressed_pages, 100);
        assert_eq!(latest.swapped_out_pages, 3);
        assert_eq!(first.previous, None);
    }

    #[test]
    fn completed_windows_are_contiguous_and_keep_the_previous_one() {
        let start = Instant::now();
        let mut history = DisplacementHistory::default();
        history.record(start, counters(0, 0));
        history.record(start + Duration::from_millis(500), counters(5, 1));
        let second = history.record(start + Duration::from_millis(1100), counters(5, 4));
        assert_eq!(second.previous.unwrap().swapped_out_pages, 1);
        let latest = second.latest.unwrap();
        assert_eq!(latest.duration, Duration::from_millis(600));
        assert_eq!(latest.compressed_pages, 0);
        assert_eq!(latest.swapped_out_pages, 3);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_headroom_is_movable_memory_and_its_samples_form_displacement_windows() {
        let mut history = DisplacementHistory::default();
        let first = status(&mut history).unwrap();
        let HostMeasurements::MacOs {
            page_size_bytes,
            free_pages,
            active_pages,
            inactive_pages,
            ..
        } = first.measurements
        else {
            panic!("macOS reports Mach measurements");
        };
        assert_eq!(first.headroom.basis, HeadroomBasis::MachMovablePages);
        assert_eq!(
            first.headroom.bytes,
            (free_pages + active_pages + inactive_pages) * page_size_bytes
        );
        assert!(first.kernel_pressure.is_some());
        assert_eq!(first.displacement, Some(HostDisplacement::default()));
        std::thread::sleep(MIN_DISPLACEMENT_WINDOW);
        let second = status(&mut history).unwrap();
        let window = second.displacement.unwrap().latest.unwrap();
        assert!(window.duration >= MIN_DISPLACEMENT_WINDOW);
    }

    #[test]
    fn a_counter_that_went_backwards_reports_no_displacement() {
        let start = Instant::now();
        let mut history = DisplacementHistory::default();
        history.record(start, counters(100, 100));
        let window = history
            .record(start + Duration::from_secs(1), counters(0, 0))
            .latest
            .unwrap();
        assert_eq!(window.compressed_pages, 0);
        assert_eq!(window.swapped_out_pages, 0);
    }
}
