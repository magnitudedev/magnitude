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
use std::sync::{Arc, Condvar, Mutex};
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

/// One host RAM status read now, with the displacement windows `sampler`
/// has formed: what every claim of host RAM and every observation between
/// claims reads, so a decision never stands on a reading older than itself.
pub(crate) fn status(sampler: &HostSampler) -> Result<HostStatus, ObservationError> {
    sampler.status(platform::sample()?)
}

/// The cadence the host's displacement counters are sampled at: the 1 s
/// samples of the M1 Ultra traces the memory policy's distress rules were
/// judged on. Every displacement window is about this long however often a
/// consumer reads the host, so how often it reads changes no verdict.
pub const DISPLACEMENT_CADENCE: Duration = Duration::from_secs(1);

/// A reading of the host's cumulative displacement counters. Each platform
/// declares whether it counts displacement as its
/// `DISPLACEMENT_COUNTERS: Option<ReadCounters>`: the reading where it
/// does, none where it does not.
type ReadCounters = fn() -> Result<DisplacementCounters, ObservationError>;

/// The displacement history of one inventory, on a platform that counts
/// displacement: its counters, read by a thread every
/// [`DISPLACEMENT_CADENCE`] from its start to the sampler's drop, which
/// stops and joins it. Where the platform counts none there is nothing to
/// sample and no thread.
pub(crate) struct HostSampler {
    running: Option<(Arc<Sampled>, std::thread::JoinHandle<()>)>,
}

struct Sampled {
    read: ReadCounters,
    state: Mutex<SampledState>,
    /// Signalled when the sampler is dropped.
    stop: Condvar,
}

struct SampledState {
    history: DisplacementHistory,
    /// The last tick's failed counter reading: the windows are unknown
    /// until a reading succeeds again, which opens a fresh window.
    failed: Option<ObservationError>,
    stopped: bool,
}

impl HostSampler {
    /// Start sampling, from a first reading now.
    pub(crate) fn start() -> Result<Self, ObservationError> {
        let Some(read) = platform::DISPLACEMENT_COUNTERS else {
            return Ok(Self { running: None });
        };
        let mut history = DisplacementHistory::default();
        history.record(Instant::now(), read()?);
        let sampled = Arc::new(Sampled {
            read,
            state: Mutex::new(SampledState {
                history,
                failed: None,
                stopped: false,
            }),
            stop: Condvar::new(),
        });
        let thread = std::thread::Builder::new()
            .name("seismic-host".into())
            .spawn({
                let sampled = sampled.clone();
                move || sampled.run()
            })
            .map_err(|error| ObservationError::Failed(format!("host sampler thread: {error}")))?;
        Ok(Self {
            running: Some((sampled, thread)),
        })
    }

    /// The status of host RAM `sample` with this sampler's displacement
    /// windows: the completed windows of the last [`DISPLACEMENT_HISTORY`],
    /// none on a platform that counts no displacement.
    fn status(&self, sample: HostSample) -> Result<HostStatus, ObservationError> {
        let displacement = match &self.running {
            None => None,
            Some((sampled, _)) => {
                let state = sampled.lock();
                match &state.failed {
                    Some(error) => return Err(error.clone()),
                    None => Some(state.history.completed.clone()),
                }
            }
        };
        Ok(HostStatus {
            allocatable: sample.allocatable,
            obtainable: sample.obtainable,
            limits: sample.limits,
            limit_visibility: sample.limit_visibility,
            displacement,
            kernel_pressure: sample.kernel_pressure,
            free_target: sample.free_target,
            held: sample.held,
            displaced_bytes: sample.displaced_bytes,
        })
    }
}

impl Sampled {
    fn lock(&self) -> std::sync::MutexGuard<'_, SampledState> {
        self.state
            .lock()
            .expect("displacement sampler lock poisoned")
    }

    fn run(&self) {
        let mut state = self.lock();
        loop {
            let due = Instant::now() + DISPLACEMENT_CADENCE;
            while !state.stopped {
                let now = Instant::now();
                if now >= due {
                    break;
                }
                state = self
                    .stop
                    .wait_timeout(state, due - now)
                    .expect("displacement sampler lock poisoned")
                    .0;
            }
            if state.stopped {
                return;
            }
            drop(state);
            let reading = (self.read)();
            state = self.lock();
            match reading {
                Ok(counters) => {
                    state.history.record(Instant::now(), counters);
                    state.failed = None;
                }
                Err(error) => {
                    state.history.open = None;
                    state.failed = Some(error);
                }
            }
        }
    }
}

impl Drop for HostSampler {
    fn drop(&mut self) {
        if let Some((sampled, thread)) = self.running.take() {
            sampled.lock().stopped = true;
            sampled.stop.notify_all();
            thread.join().expect("the host sampler does not panic");
        }
    }
}

/// One platform sample, before the displacement history of the inventory
/// that took it is applied.
pub(crate) struct HostSample {
    pub(crate) allocatable: Reported<u64>,
    pub(crate) obtainable: Reported<u64>,
    pub(crate) limits: Vec<ProcessMemoryLimit>,
    pub(crate) limit_visibility: LimitVisibility,
    pub(crate) kernel_pressure: Option<KernelPressure>,
    pub(crate) free_target: Option<u64>,
    pub(crate) held: Option<HostHeld>,
    pub(crate) displaced_bytes: Option<u64>,
}

/// RAM the OS holds that it cannot make available to a process, so counted
/// in neither [`HostStatus::allocatable`] nor [`HostStatus::obtainable`];
/// on a platform that counts it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostHeld {
    /// Wired memory: pinned by the kernel and drivers, GPU buffers included
    /// on unified memory.
    pub wired_bytes: u64,
    /// Physical memory the compressor occupies.
    pub compressed_bytes: u64,
}

/// Host RAM's status in one sample.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostStatus {
    /// RAM this process can take without the OS displacing another
    /// program's data (compressing or swapping it): what a process that
    /// grows and gives back while others run may take. An estimate, never
    /// exact physical availability.
    pub allocatable: Reported<u64>,
    /// RAM the OS can make available to this process, displacing other
    /// programs' data if it must: what an explicit request to load may
    /// take. At least `allocatable`.
    pub obtainable: Reported<u64>,
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
    /// The free memory the kernel keeps for itself and reclaims toward,
    /// which `allocatable` still counts: below it, taking memory makes the
    /// kernel reclaim, and where nothing cheap is left to reclaim, compress
    /// or swap another program's data. The line a load's reserve starts
    /// from. `Some(0)` where `allocatable` already leaves it out; `None` on
    /// a platform that reports no such line.
    pub free_target: Option<u64>,
    /// What the OS holds that neither `allocatable` nor `obtainable`
    /// counts ([`HostHeld`]). `None` on a platform that counts none.
    pub held: Option<HostHeld>,
    /// RAM the OS made free by displacing programs' data that is displaced
    /// still, each displaced page counted once: what the data would take
    /// in RAM as itself, less what it takes there compressed. `allocatable` counts this RAM as free, though it is free only
    /// while that data stays out. `None` on a platform that counts none.
    pub displaced_bytes: Option<u64>,
}

impl HostStatus {
    /// [`Self::allocatable`] within every process limit.
    pub fn allocatable_bytes(&self) -> u64 {
        self.within_limits(self.allocatable.value)
    }

    /// [`Self::obtainable`] within every process limit.
    pub fn obtainable_bytes(&self) -> u64 {
        self.within_limits(self.obtainable.value)
    }

    fn within_limits(&self, bytes: u64) -> u64 {
        self.limits
            .iter()
            .fold(bytes, |bytes, limit| bytes.min(limit.remaining_bytes()))
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

/// Displacement over one completed window: the span between two successive
/// readings of the host's counters, at least one [`DISPLACEMENT_CADENCE`]
/// apart.
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
    /// When the window ended: the reading that completed it.
    pub ended: Instant,
}

/// The completed displacement windows of the last [`DISPLACEMENT_HISTORY`],
/// most recent first. They are contiguous: each ends where the one before it
/// in the list begins.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HostDisplacement {
    /// The windows, most recent first.
    pub windows: Vec<DisplacementWindow>,
}

/// How far back completed windows are kept: at least the longest span a
/// memory policy judges displacement over.
pub const DISPLACEMENT_HISTORY: Duration = Duration::from_secs(60);

/// The completed displacement windows of one inventory's counter readings.
/// Windows tumble: each reading completes the open window and opens the
/// next, so a window is as long as the sampler's cadence makes it
/// ([`DISPLACEMENT_CADENCE`]). A window that ended more than
/// [`DISPLACEMENT_HISTORY`] ago is dropped.
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
                ended: at,
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

    /// The first reading opens a window; each next one completes it, however
    /// soon: the sampler's cadence alone sets a window's span.
    #[test]
    fn each_reading_completes_the_open_window() {
        let start = Instant::now();
        let mut history = DisplacementHistory::default();
        assert_eq!(
            history.record(start, counters(10, 0)),
            HostDisplacement::default()
        );
        let first = history.record(start + Duration::from_millis(100), counters(110, 3));
        let [latest] = first.windows[..] else {
            panic!("one window completed");
        };
        assert_eq!(latest.duration, Duration::from_millis(100));
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
    fn macos_room_is_read_with_and_without_compression_and_its_samples_form_displacement_windows() {
        let reading = macos::vm_reading().unwrap();
        let pages = &reading.statistics;
        let allocatable = u64::from(pages.free_count) - u64::from(pages.speculative_count)
            + u64::from(pages.external_page_count)
            + u64::from(pages.purgeable_count);
        let obtainable = u64::from(pages.free_count)
            + u64::from(pages.active_count)
            + u64::from(pages.inactive_count);
        let sample = macos::sample_of(&reading).unwrap();
        assert_eq!(
            sample.allocatable,
            Reported::new(
                allocatable * reading.page_size,
                seismic_topology::ReportedBy::MachAllocatablePages
            )
        );
        assert_eq!(
            sample.obtainable,
            Reported::new(
                obtainable * reading.page_size,
                seismic_topology::ReportedBy::MachMovablePages
            )
        );
        let started = Instant::now();
        let sampler = HostSampler::start().unwrap();
        let first = status(&sampler).unwrap();
        assert!(first.kernel_pressure.is_some());
        assert_eq!(first.displacement, Some(HostDisplacement::default()));
        // However often a reader reads, the windows are the sampler's: each
        // at least one cadence long, since a tick waits a whole cadence
        // after the sample before it, so no more of them than cadences have
        // passed.
        let windows = loop {
            let windows = status(&sampler).unwrap().displacement.unwrap().windows;
            let elapsed = started.elapsed();
            assert!(
                windows.len() as u128 <= elapsed.as_nanos() / DISPLACEMENT_CADENCE.as_nanos(),
                "{windows:?} after {elapsed:?}"
            );
            if windows.len() >= 2 {
                break windows;
            }
            std::thread::sleep(DISPLACEMENT_CADENCE / 10);
        };
        assert!(
            windows
                .iter()
                .all(|window| window.duration >= DISPLACEMENT_CADENCE),
            "{windows:?}"
        );
        // Dropping the sampler stops and joins its thread.
        drop(sampler);
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
