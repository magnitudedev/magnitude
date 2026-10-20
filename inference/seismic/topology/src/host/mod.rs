//! The host: RAM capacity, the OS build, and host RAM status with the
//! process limits that bound every path into host RAM (spec §2.4). Host
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

use crate::model::Reported;
use std::fmt;
use std::time::{Duration, Instant};

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
/// one.
pub(crate) fn cpu_identity() -> Result<String, String> {
    let mut identity = platform::cpu_identity()?;
    if let Some(tier) = seismic_native_cpu::Tier::detected() {
        identity.push_str(&format!(";isa={}", tier.name()));
    }
    Ok(identity)
}

/// One host RAM status. `history` holds the displacement windows of the
/// samples taken through it; the sample closes a window when it is due.
pub(crate) fn status(history: &mut DisplacementHistory) -> Result<HostStatus, ObservationError> {
    let sample = platform::sample()?;
    Ok(HostStatus {
        headroom: sample.headroom,
        limits: sample.limits,
        limit_visibility: sample.limit_visibility,
        displacement: sample
            .displacement
            .map(|counters| history.record(Instant::now(), counters)),
        kernel_pressure: sample.kernel_pressure,
    })
}

/// One platform sample, before the displacement history of the inventory
/// that took it is applied.
pub(crate) struct HostSample {
    pub(crate) headroom: Reported<u64>,
    pub(crate) limits: Vec<ProcessMemoryLimit>,
    pub(crate) limit_visibility: LimitVisibility,
    /// Cumulative displacement, on a platform that counts it.
    pub(crate) displacement: Option<DisplacementCounters>,
    pub(crate) kernel_pressure: Option<KernelPressure>,
}

/// Host RAM's status in one sample.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostStatus {
    /// The OS's estimate of RAM this process can still take. An estimate of
    /// allocatable RAM, never exact physical availability.
    pub headroom: Reported<u64>,
    /// The OS limits on this process: grants on host RAM that every path
    /// into it shares.
    pub limits: Vec<ProcessMemoryLimit>,
    /// Whether every applicable limit could be observed.
    pub limit_visibility: LimitVisibility,
    /// Displacement of programs' memory over the recent windows of the
    /// inventory that took this sample. `None` on a platform that counts
    /// none.
    pub displacement: Option<HostDisplacement>,
    /// The kernel's own pressure classification. `None` on a platform
    /// without one.
    pub kernel_pressure: Option<KernelPressure>,
}

impl HostStatus {
    /// What host RAM has left for this process: the OS's availability
    /// within every process limit.
    pub fn available(&self) -> u64 {
        self.limits
            .iter()
            .fold(self.headroom.value, |bytes, limit| {
                bytes.min(limit.remaining_bytes())
            })
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
    /// The span between the window's two samples.
    pub duration: Duration,
    /// Pages compressed in memory within the window.
    pub compressed_pages: u64,
    /// Pages faulted back from compression within the window.
    pub decompressed_pages: u64,
    /// Pages written to swap within the window.
    pub swapped_out_pages: u64,
    /// Pages read back from swap within the window.
    pub swapped_in_pages: u64,
}

/// The completed displacement windows of the last [`DISPLACEMENT_HISTORY`],
/// most recent first. They are contiguous: each ends where the one before it
/// in the list begins.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HostDisplacement {
    /// The windows, most recent first.
    pub windows: Vec<DisplacementWindow>,
}

/// The least span of one displacement window. Rates over a shorter span are
/// dominated by when the kernel happened to run its pageout work.
const MIN_DISPLACEMENT_WINDOW: Duration = Duration::from_millis(500);

/// How far back completed windows are kept: at least the longest span a
/// memory policy judges displacement over.
pub const DISPLACEMENT_HISTORY: Duration = Duration::from_secs(60);

/// The completed displacement windows of one inventory's host samples.
/// Windows tumble: the first sample at least [`MIN_DISPLACEMENT_WINDOW`]
/// after a window opened completes it and opens the next. A window that
/// ended more than [`DISPLACEMENT_HISTORY`] ago is dropped.
#[derive(Debug, Default)]
pub(crate) struct DisplacementHistory {
    open: Option<(Instant, DisplacementCounters)>,
    completed: HostDisplacement,
}

impl DisplacementHistory {
    pub(crate) fn record(
        &mut self,
        at: Instant,
        counters: DisplacementCounters,
    ) -> HostDisplacement {
        let Some((opened_at, opening)) = self.open else {
            self.open = Some((at, counters));
            return self.completed.clone();
        };
        let duration = at.saturating_duration_since(opened_at);
        if duration >= MIN_DISPLACEMENT_WINDOW {
            self.completed.windows.insert(
                0,
                DisplacementWindow {
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
                },
            );
            // Windows are contiguous, so the ones after the first that ends
            // beyond the history are older still.
            let mut ended = at;
            let kept = self
                .completed
                .windows
                .iter()
                .take_while(|window| {
                    let within = at.saturating_duration_since(ended) <= DISPLACEMENT_HISTORY;
                    ended = ended.checked_sub(window.duration).unwrap_or(ended);
                    within
                })
                .count();
            self.completed.windows.truncate(kept);
            self.open = Some((at, counters));
        }
        self.completed.clone()
    }
}

/// A kernel's classification of system memory pressure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelPressure {
    /// The kernel reports no memory pressure.
    Normal,
    /// The kernel warns of memory pressure.
    Warning,
    /// The kernel reports critical memory pressure.
    Critical,
}

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

/// Whether a sample's process limits are all the limits that apply.
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
        assert_eq!(
            history.record(start, counters(10, 0)),
            HostDisplacement::default()
        );
        // Samples inside the open window complete nothing.
        let early = history.record(start + Duration::from_millis(499), counters(40, 0));
        assert_eq!(early, HostDisplacement::default());
        let first = history.record(start + Duration::from_millis(700), counters(110, 3));
        let [latest] = first.windows[..] else {
            panic!("one window completed");
        };
        assert_eq!(latest.duration, Duration::from_millis(700));
        assert_eq!(latest.compressed_pages, 100);
        assert_eq!(latest.swapped_out_pages, 3);
    }

    #[test]
    fn completed_windows_are_contiguous_most_recent_first() {
        let start = Instant::now();
        let mut history = DisplacementHistory::default();
        history.record(start, counters(0, 0));
        history.record(start + Duration::from_millis(500), counters(5, 1));
        let second = history.record(start + Duration::from_millis(1100), counters(5, 4));
        let [latest, previous] = second.windows[..] else {
            panic!("two windows completed");
        };
        assert_eq!(previous.swapped_out_pages, 1);
        assert_eq!(latest.duration, Duration::from_millis(600));
        assert_eq!(latest.compressed_pages, 0);
        assert_eq!(latest.swapped_out_pages, 3);
    }

    #[test]
    fn windows_that_ended_beyond_the_history_are_dropped() {
        let start = Instant::now();
        let mut history = DisplacementHistory::default();
        let mut at = start;
        history.record(at, counters(0, 0));
        let mut displacement = HostDisplacement::default();
        for _ in 0..200 {
            at += Duration::from_millis(500);
            displacement = history.record(at, counters(0, 0));
        }
        // Every kept window ended within the history; the next older one
        // would not have.
        let kept = displacement.windows.len() as u32;
        assert_eq!(
            Duration::from_millis(500) * (kept - 1),
            DISPLACEMENT_HISTORY
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_headroom_is_movable_memory_and_its_samples_form_displacement_windows() {
        let reading = macos::vm_reading().unwrap();
        let pages = &reading.statistics;
        let movable = u64::from(pages.free_count)
            + u64::from(pages.active_count)
            + u64::from(pages.inactive_count);
        assert_eq!(
            macos::sample_of(&reading).unwrap().headroom,
            Reported::new(
                movable * reading.page_size,
                crate::model::ReportedBy::MachMovablePages
            )
        );
        let mut history = DisplacementHistory::default();
        let first = status(&mut history).unwrap();
        assert!(first.kernel_pressure.is_some());
        assert_eq!(first.displacement, Some(HostDisplacement::default()));
        std::thread::sleep(MIN_DISPLACEMENT_WINDOW);
        let second = status(&mut history).unwrap();
        let window = second.displacement.unwrap().windows[0];
        assert!(window.duration >= MIN_DISPLACEMENT_WINDOW);
    }

    #[test]
    fn a_counter_that_went_backwards_reports_no_displacement() {
        let start = Instant::now();
        let mut history = DisplacementHistory::default();
        history.record(start, counters(100, 100));
        let window = history
            .record(start + Duration::from_secs(1), counters(0, 0))
            .windows[0];
        assert_eq!(window.compressed_pages, 0);
        assert_eq!(window.swapped_out_pages, 0);
    }
}
