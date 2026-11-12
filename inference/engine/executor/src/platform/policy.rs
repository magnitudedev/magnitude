//! The engine's sole fit policy over the topology's stores, grants and
//! placements ([`Fit`]).
//!
//! Seismic's topology reports stores, their capacities, the access paths
//! executors place storage along and the grants on them, and a status sample
//! reads each store's headroom and what is left of each grant; Seismic
//! enforces the limits it is given and holds no reserve policy. The host
//! (service or CLI) chooses the threshold policy, [`MemoryReserves`], keyed
//! by store kind, and this module applies it to every store a load reaches:
//!
//! - Stable fit compares a workload with each store's capacity, bounded by
//!   the applicable process limits for host RAM and by the grants of the
//!   paths it is placed along, less the store's planning reserve. It uses no
//!   current free memory. An executor's stable fit for a storage kind is the
//!   sum along its placement of each grant within its store's fit, a store
//!   reached by several paths or executors counted once.
//! - Every live claim must leave each store's observed headroom above the
//!   planning reserve: the claimable ceiling is `headroom − planning`, and
//!   along a placement each grant's remaining reading bounds it too. Bytes
//!   Seismic already charged are absent from observed headroom and are
//!   never subtracted again. No engine claim crosses that line.
//!
//! Process limits are the visible ones. A container's own cgroup limit is
//! visible; limits of cgroups above it may be hidden, and are then bounded
//! only by host headroom, which already reflects the host's memory.

use seismic::{
    AccessPathId, DeviceTopology, Executor, ExecutorId, Contract, HostStatus, ObservationError, Status,
    Store, StoreId, StoreKind,
};
use std::{
    collections::BTreeMap,
    fmt,
    time::{Duration, Instant},
};

/// Host RAM's fit as `host` reads it: its capacity within the process
/// limits `host` reports, and its allocatable memory as headroom.
fn host_ram_fit(store: &Store, host: &HostStatus, reserves: &MemoryReserves) -> StoreFit {
    let capacity_bytes = host
        .limits
        .iter()
        .fold(store.capacity().value, |bytes, limit| {
            bytes.min(limit.limit_bytes)
        });
    // A fit decides whether a load may take the store: a load's own action,
    // which takes the clean file cache once (`Reclaim::CleanFileCache`).
    store_fit(
        store,
        reserves.thresholds(store),
        capacity_bytes,
        host.allocatable_bytes(seismic::Reclaim::CleanFileCache),
        MemoryConstraint::HostRam,
    )
}

/// `store`'s fit from its capacity within process limits and its headroom,
/// each less its planning reserve, its thresholds `thresholds`
/// ([`MemoryReserves::thresholds`]).
fn store_fit(
    store: &Store,
    thresholds: DomainThresholds,
    capacity_bytes: u64,
    headroom_bytes: u64,
    constraint: MemoryConstraint,
) -> StoreFit {
    StoreFit {
        store: store.id(),
        capacity_bytes,
        stable_bytes: capacity_bytes.saturating_sub(thresholds.planning_bytes),
        headroom_bytes,
        ceiling_bytes: thresholds.ceiling(headroom_bytes),
        thresholds,
        constraint,
    }
}

/// The interval a load reads memory at: the engine's probe, and what the
/// rest of the host may take between two readings, before a release can be
/// decided.
pub const READING_INTERVAL: Duration = Duration::from_millis(100);

/// The page-fill probe's samples, each its own cold region
/// ([`page_fill`]).
const PAGE_FILL_SAMPLES: usize = 3;

/// The bytes one round of the page-release probe takes and frees, at most
/// ([`page_release`]).
const PAGE_RELEASE_ROUND_BYTES: usize = 256 << 20;

/// The page-release probe's rounds a sample times back to back, and its
/// samples: one round alone varies twofold between runs on an idle
/// machine, the median over five samples of four does not
/// ([`page_release`]).
const PAGE_RELEASE_ROUNDS_PER_SAMPLE: usize = 4;
const PAGE_RELEASE_SAMPLES: usize = 5;

/// A store's margin over its line by its terms ([`StoreMotion::margin`]):
/// the fluctuation; the take over a reading interval, `t·interval`; the
/// sustained take over a release's latency, `s·latency`; and the deficit a
/// take outrunning the releasing grows, `max(0, s − r)·B/r`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MarginTerms {
    pub fluctuation: u64,
    pub take: u64,
    pub latency: u64,
    pub releasing: u64,
}

impl MarginTerms {
    /// The margin: its terms' sum.
    pub fn total(self) -> u64 {
        self.fluctuation.saturating_add(self.take).saturating_add(self.latency).saturating_add(self.releasing)
    }
}

/// How other programs move a store's available bytes, as measured: the
/// steady fluctuation (the largest fall between consecutive readings with
/// nothing of ours written or asked) and the fastest rate another program
/// takes it at. Its margin ([`Self::margin`]) is what a planning line keeps
/// above the store's own line: a load's on host RAM or device memory, a
/// download's on a volume ([`super::VolumeReserve`]).
///
/// The take rate is a measured, positive rate: before any program was seen
/// taking the store, the fastest one program can take it at, measured of
/// the machine (host RAM's page-fill rate, [`Self::of_machine`]; a volume's
/// sequential write rate, [`super::VolumeReserve`]). So no margin is zero.
/// That fastest rate is a burst's: what one program takes before a reading
/// notices it. What another program sustains over time is `sustained`, the
/// fastest take seen across readings, none before one is seen.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "MotionFigures")]
pub struct StoreMotion {
    fluctuation: u64,
    take_rate: f64,
    /// Written whether or not it is known: the worker transport's encoding
    /// is positional (postcard), so an omitted field misreads every field
    /// after it. JSON writes `null`, which `MotionFigures` reads as none.
    sustained: Option<f64>,
}

/// A take rate is finite by construction ([`StoreMotion::new`]).
impl Eq for StoreMotion {}

/// [`StoreMotion`]'s figures as serialized, checked as [`StoreMotion::new`]
/// checks them.
#[derive(serde::Deserialize)]
struct MotionFigures {
    fluctuation: u64,
    take_rate: f64,
    #[serde(default)]
    sustained: Option<f64>,
}

/// Whether `rate` is a measured rate: finite and positive.
fn is_rate(rate: f64) -> bool {
    rate.is_finite() && rate > 0.0
}

impl TryFrom<MotionFigures> for StoreMotion {
    type Error = String;

    fn try_from(figures: MotionFigures) -> Result<Self, String> {
        match (is_rate(figures.take_rate), figures.sustained.is_none_or(is_rate)) {
            (true, true) => Ok(Self { fluctuation: figures.fluctuation, take_rate: figures.take_rate, sustained: figures.sustained }),
            _ => Err(format!(
                "take rates are measured and positive, not {} and {:?}",
                figures.take_rate, figures.sustained
            )),
        }
    }
}

impl StoreMotion {
    /// A motion measured across readings: `fluctuation`, and another
    /// program seen taking the store at `take_rate`, sustained.
    ///
    /// # Panics
    /// Where `take_rate` (bytes a second) is not a positive finite rate:
    /// what bounds a take is measured, never zero.
    pub fn new(fluctuation: u64, take_rate: f64) -> Self {
        assert!(is_rate(take_rate), "a take rate is measured and positive, not {take_rate}");
        Self { fluctuation, take_rate, sustained: Some(take_rate) }
    }

    /// A store's motion before any reading of it: the fastest one program
    /// can take it at, `take_rate`, a burst no program was seen sustain.
    ///
    /// # Panics
    /// Where `take_rate` is not a positive finite rate.
    pub fn prior(take_rate: f64) -> Self {
        assert!(is_rate(take_rate), "a take rate is measured and positive, not {take_rate}");
        Self { fluctuation: 0, take_rate, sustained: None }
    }

    /// The machine's own motion before any reading of it: no measured
    /// fluctuation, and the rate a fresh program takes host RAM at over one
    /// reading interval ([`page_fill`]): what it faults of fresh pages in
    /// the interval, over the interval, so the take over an interval is
    /// those bytes. No program on one thread takes memory from the OS
    /// faster than the machine faults it fresh pages, so a load whose line
    /// holds that take over a reading interval is not crossed by one
    /// program's ask before the load notices it.
    ///
    /// A generic figure of the machine, measured once: where `machine`
    /// names a kernel cache and the topology of the machine it serves, the
    /// rate that cache keeps for the host's identity
    /// (`KernelCache::machine_figures`, keyed by
    /// `PhysicalIdentity::Host`), measured and stored there when it keeps
    /// none; elsewhere, measured once per process.
    pub fn of_machine(machine: Option<(&std::path::Path, &DeviceTopology)>) -> Self {
        Self::prior(page_rate(machine, "page-fill-interval", |probe| probe.fill))
    }

    /// The margin a planning line holds above its store's own line, for a
    /// load that notices a take within a reading `interval` and whose
    /// first release lands `latency` after it decides one, then releases
    /// `releasing`'s bytes at its rate (none of a store nothing of ours
    /// releases in, a volume's):
    ///
    /// `fluctuation + t·interval + s·latency + max(0, s − r)·B / r`
    ///
    /// for the take rate `t`, the sustained take `s` (none before one is
    /// seen), release rate `r` and releasable bytes `B`: one program takes
    /// at most `t` before the next reading notices it, so the rest of the
    /// host takes `t·interval` by then; what it goes on taking while the
    /// release lands is what it is seen to sustain, `s·latency`; and while
    /// it sustains a take faster than the load releases, the deficit grows
    /// at `s − r` for the `B / r` releasing runs. A burst is no sustained
    /// take: before one is seen across readings, neither the latency nor
    /// the releasing adds anything. So the latency, the load's own, moves
    /// the line only by what other programs are seen taking: a release of
    /// the load's that took long never raises the line the next release is
    /// decided at while nothing else moves, and no release of its own drives
    /// another. Held for the `B / r` seconds a release of a 25.2 GB bank
    /// runs at 3.1 GB/s, an M4 Pro's 28.8 GB/s page-fill burst would be a
    /// 208 GB take of a 51.5 GB store; held over a 200 ms release latency,
    /// 5.8 GB a 48 GB M4 Pro's line rose by at each slow release, until the
    /// line outran every release and the load unloaded with nothing else
    /// taking memory.
    ///
    /// The rest of the host never takes more of a store than it holds, so
    /// the margin of a store a load releases in is within the store's
    /// bytes. A margin past them is a measurement that cannot be: an
    /// invariant violation reported with every input, a panic in debug and
    /// test builds.
    pub fn margin(&self, interval: Duration, latency: Duration, releasing: Option<Releasing>) -> u64 {
        self.terms(interval, latency, releasing).total()
    }

    /// [`Self::margin`] by its terms.
    pub fn terms(&self, interval: Duration, latency: Duration, releasing: Option<Releasing>) -> MarginTerms {
        let sustained = self.sustained.unwrap_or(0.0);
        let lead = MarginTerms {
            fluctuation: self.fluctuation,
            take: (self.take_rate * interval.as_secs_f64()) as u64,
            latency: (sustained * latency.as_secs_f64()) as u64,
            releasing: 0,
        };
        let Some(releasing) = releasing else {
            return lead;
        };
        let growth = (sustained - releasing.rate.0).max(0.0) / releasing.rate.0;
        let terms = MarginTerms { releasing: (growth * releasing.bytes as f64) as u64, ..lead };
        let margin = terms.total();
        if margin > releasing.store_bytes {
            tracing::error!(
                margin,
                store_bytes = releasing.store_bytes,
                fluctuation = self.fluctuation,
                take_rate = self.take_rate,
                sustained = ?self.sustained,
                release_rate = releasing.rate.bytes_per_second(),
                releasable = releasing.bytes,
                interval = ?interval,
                latency = ?latency,
                "invariant violated: a memory margin exceeds the store it is held in"
            );
            if cfg!(any(debug_assertions, test)) {
                panic!(
                    "invariant violated: a margin of {margin} bytes in a store of {} ({self:?}, {releasing:?}, interval {interval:?}, latency {latency:?})",
                    releasing.store_bytes
                );
            }
        }
        terms
    }

    pub fn fluctuation(&self) -> u64 {
        self.fluctuation
    }

    pub fn take_rate(&self) -> f64 {
        self.take_rate
    }

    /// The larger of each figure: what a later measurement adds to an
    /// earlier one, which it never lowers.
    pub fn with(self, other: Self) -> Self {
        let sustained = match (self.sustained, other.sustained) {
            (Some(one), Some(other)) => Some(one.max(other)),
            (one, other) => one.or(other),
        };
        Self { fluctuation: self.fluctuation.max(other.fluctuation), take_rate: self.take_rate.max(other.take_rate), sustained }
    }
}

/// The rate a load releases memory at, measured and positive: a finished
/// release's bytes over the span they took to release (its wait for queued
/// steps apart, `ResizeTimes::releasing`), the slowest
/// measured; before any, the machine's page-release rate
/// ([`Self::of_machine`]).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "f64", into = "f64")]
pub struct ReleaseRate(f64);

/// A release rate is finite by construction ([`ReleaseRate::new`]).
impl Eq for ReleaseRate {}

impl TryFrom<f64> for ReleaseRate {
    type Error = String;

    fn try_from(rate: f64) -> Result<Self, String> {
        match rate.is_finite() && rate > 0.0 {
            true => Ok(Self(rate)),
            false => Err(format!("a release rate is measured and positive, not {rate}")),
        }
    }
}

impl From<ReleaseRate> for f64 {
    fn from(rate: ReleaseRate) -> f64 {
        rate.0
    }
}

impl ReleaseRate {
    /// # Panics
    /// Where `bytes_per_second` is not a positive finite rate.
    pub fn new(bytes_per_second: f64) -> Self {
        Self::try_from(bytes_per_second).unwrap_or_else(|error| panic!("{error}"))
    }

    /// The rate the machine returns pages to the OS at, as a release is
    /// counted on to ([`page_release`]): a generic figure of the machine,
    /// kept and measured as [`StoreMotion::of_machine`]'s page-fill rate is.
    pub fn of_machine(machine: Option<(&std::path::Path, &DeviceTopology)>) -> Self {
        Self::new(page_rate(machine, "page-release-median", |probe| probe.release))
    }

    /// The slower of two rates: what a later measurement adds, which never
    /// makes a margin smaller.
    pub fn with(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    pub fn bytes_per_second(&self) -> f64 {
        self.0
    }
}

/// What a load releases in a store of `store_bytes`: `bytes`, at `rate`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Releasing {
    pub rate: ReleaseRate,
    pub bytes: u64,
    pub store_bytes: u64,
}

/// The bytes a serving load can release in host RAM and in its device's own
/// memory: its bank's beyond the bank's floor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Releasable {
    pub host: u64,
    pub device: u64,
}

/// A rate the machine's figures keep: host RAM's page-fill and page-release
/// rates, a volume's sequential write rate.
#[derive(serde::Serialize, serde::Deserialize)]
struct MachineRate {
    bytes_per_second: f64,
}

impl seismic::measure::Stored for MachineRate {
    const SCHEMA: &'static str = "machine-rate-v1";
}

/// The most bytes a stored [`MachineRate`] is read as.
const MACHINE_RATE_RECORD_BYTES: u64 = 4096;

/// A generic figure of the machine, measured once: where `figure` names a
/// kernel cache and the figure's key there, the rate its machine figures
/// (`KernelCache::machine_figures`) keep under that key, `measure`d and
/// stored there when they keep none; elsewhere, `measure`d.
pub(super) fn machine_rate<E>(
    figure: Option<(&std::path::Path, String)>,
    measure: impl FnOnce() -> Result<f64, E>,
) -> Result<f64, E> {
    let Some((cache, key)) = figure else { return measure() };
    let figures = crate::kernel_cache::KernelCache::machine_figures(cache);
    match figures.load::<MachineRate>(&key, MACHINE_RATE_RECORD_BYTES) {
        Ok(Some(MachineRate { bytes_per_second })) if bytes_per_second.is_finite() && bytes_per_second > 0.0 => {
            return Ok(bytes_per_second);
        }
        Ok(_) => {}
        Err(error) => tracing::warn!("machine figures: the {key} rate stored is unreadable: {error}"),
    }
    let bytes_per_second = measure()?;
    if let Err(error) = figures.store(&key, &MachineRate { bytes_per_second }, MACHINE_RATE_RECORD_BYTES) {
        tracing::warn!("machine figures: the {key} rate could not be stored: {error}");
    }
    Ok(bytes_per_second)
}

/// The host's identity in `topology` (`PhysicalIdentity::Host`): its CPU
/// as the OS reports it and its RAM, what a machine's figures are keyed by.
fn host_identity(topology: &DeviceTopology) -> Option<String> {
    topology.physical_devices().iter().find_map(|device| match device.identity() {
        seismic::PhysicalIdentity::Host { machine } => Some(machine.clone()),
        _ => None,
    })
}

/// The machine's page probe: the rate a fresh program takes host RAM at
/// over a reading interval ([`page_fill`]) and the rate the machine returns
/// pages to the OS at ([`page_release`]), in bytes a second; whether the
/// host's room cut the fill short (`censored`); and what bounded it: the
/// room it read, and the bytes each fill sample took, every one returned
/// before the next, so the largest is the probe's peak.
#[derive(Clone, Copy, Debug)]
pub struct PageProbe {
    pub fill: f64,
    pub release: f64,
    pub censored: bool,
    pub room: u64,
    pub fill_samples: [usize; PAGE_FILL_SAMPLES],
}

/// A figure the page probe took under a cut it could not measure past: a
/// lower bound of the figure, used but never kept.
struct Censored(f64);

/// The page probe's `figure` of the machine (`page-fill-interval`,
/// `page-release-median`, named by their methods, so a figure another
/// method stored is never read as theirs), kept under `{figure}:{host}`
/// ([`machine_rate`]), but for a censored probe's, which is used and not
/// kept, so a later process probes again; the probe runs at most once per
/// process.
fn page_rate(machine: Option<(&std::path::Path, &DeviceTopology)>, figure: &str, rate: fn(PageProbe) -> f64) -> f64 {
    static PROBED: std::sync::OnceLock<PageProbe> = std::sync::OnceLock::new();
    let key = machine.and_then(|(cache, topology)| Some((cache, format!("{figure}:{}", host_identity(topology)?))));
    machine_rate(key, || {
        let probe = *PROBED.get_or_init(PageProbe::run);
        if probe.censored { Err(Censored(rate(probe))) } else { Ok(rate(probe)) }
    })
    .unwrap_or_else(|Censored(rate)| rate)
}

impl PageProbe {
    /// The page probe, run, bounded by half of host RAM's room
    /// (`seismic::host_room`; a page where it cannot be read), so it never
    /// causes the pressure it guards against: a burst
    /// the reserve guards against and the probe must both fit in that room,
    /// so the probe takes at most the share a burst would leave. A fill the
    /// bound cut short is censored: the take over an interval is then more
    /// than half the room, the machine already near pressure, and the
    /// bound's rate a lower bound of it, reported as an event, used and
    /// never kept. It understates the take only while the room is under
    /// twice a burst, and a later probe, in more room, corrects it.
    pub fn run() -> Self {
        let page = host_page();
        let room = seismic::host_room(seismic::Reclaim::Nothing).unwrap_or_else(|error| {
            tracing::warn!(%error, "the page probe could not read host RAM's room, and takes a page at most");
            0
        });
        let bound = usize::try_from(room / 2).unwrap_or(usize::MAX).max(page);
        let filled = page_fill(bound, page);
        let (fill, censored) = (filled.rate(), filled.censored());
        if censored {
            tracing::warn!(
                fill_bytes_per_second = fill,
                room_bytes = room,
                "the page-fill probe filled the host's room before its reading interval ended: \
                 its rate is a lower bound, used and not kept"
            );
        }
        let release = page_release(bound.min(PAGE_RELEASE_ROUND_BYTES));
        Self { fill, release, censored, room, fill_samples: filled.0.map(|(bytes, _)| bytes) }
    }
}

/// The host's page size.
fn host_page() -> usize {
    #[cfg(unix)]
    {
        // SAFETY: `sysconf` reads a constant of the system.
        usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).expect("the page size is positive")
    }
    #[cfg(not(unix))]
    {
        4096
    }
}

/// What a fresh program takes of host RAM within one reading interval:
/// [`PAGE_FILL_SAMPLES`] samples, each a fresh region of at most `bound`
/// bytes the process has never touched (cold: a recycled page faults
/// faster than a fresh one, which overstates the take), faulted a page at a
/// time from its start until [`READING_INTERVAL`] elapses or the bound is
/// reached, then returned before the next.
fn page_fill(bound: usize, page: usize) -> PageFill {
    PageFill(std::array::from_fn(|_| ColdRegion::map(bound).fill(page, READING_INTERVAL)))
}

/// The page-fill probe's samples: each one's bytes faulted, and whether
/// the bound cut it short of the interval.
struct PageFill([(usize, bool); PAGE_FILL_SAMPLES]);

impl PageFill {
    /// The median sample's bytes over the interval.
    fn rate(&self) -> f64 {
        let mut bytes = self.0.map(|(bytes, _)| bytes);
        bytes.sort_unstable();
        bytes[PAGE_FILL_SAMPLES / 2] as f64 / READING_INTERVAL.as_secs_f64()
    }

    /// Whether the bound cut any sample short: its rate is then a lower
    /// bound.
    fn censored(&self) -> bool {
        self.0.iter().any(|&(_, cut)| cut)
    }
}

/// Fresh address space of the OS's own, mapped for the page-fill probe and
/// returned to it whole on drop: never memory an allocator recycles.
struct ColdRegion {
    base: *mut u8,
    bytes: usize,
}

impl ColdRegion {
    fn map(bytes: usize) -> Self {
        #[cfg(unix)]
        // SAFETY: an anonymous private mapping at an address the system
        // chooses refers to no existing memory; its pages fault in on
        // first touch.
        let base = unsafe {
            let base = libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            );
            assert!(base != libc::MAP_FAILED, "the page-fill probe could not map {bytes} bytes: {}", std::io::Error::last_os_error());
            base.cast::<u8>()
        };
        // An allocation this large is the OS's own fresh pages on Windows,
        // returned to it when freed.
        #[cfg(not(unix))]
        // SAFETY: a non-zero layout, freed with the same layout on drop.
        let base = unsafe {
            let base = std::alloc::alloc(std::alloc::Layout::from_size_align(bytes, 4096).expect("the probe's layout"));
            assert!(!base.is_null(), "the page-fill probe could not take {bytes} bytes");
            base
        };
        Self { base, bytes }
    }

    /// A byte written to each page from the start, one at a time, until
    /// `interval` elapses or the region ends: the bytes faulted, and
    /// whether the region's end cut the interval short.
    fn fill(&self, page: usize, interval: Duration) -> (usize, bool) {
        // The clock is read every this many pages, a small part of their
        // faults' time.
        const PAGES_PER_READ: usize = 16;
        let started = Instant::now();
        let mut offset = 0;
        while offset < self.bytes {
            if (offset / page) % PAGES_PER_READ == 0 && started.elapsed() >= interval {
                return (offset, false);
            }
            // SAFETY: within the region, which only this probe touches.
            unsafe { std::ptr::write_volatile(self.base.add(offset), 1) };
            offset += page;
        }
        (self.bytes, started.elapsed() < interval)
    }
}

impl Drop for ColdRegion {
    fn drop(&mut self) {
        #[cfg(unix)]
        // SAFETY: the whole mapping `map` made, which nothing refers to.
        {
            let unmapped = unsafe { libc::munmap(self.base.cast(), self.bytes) };
            debug_assert_eq!(unmapped, 0, "a whole mapping is unmapped");
        }
        #[cfg(not(unix))]
        // SAFETY: the allocation `map` made, with its layout.
        unsafe {
            std::alloc::dealloc(self.base, std::alloc::Layout::from_size_align(self.bytes, 4096).expect("the probe's layout"))
        };
    }
}

/// The rate the machine returns pages to the OS at, as a release is
/// counted on to: one round discarded, then [`PAGE_RELEASE_SAMPLES`]
/// samples of [`PAGE_RELEASE_ROUNDS_PER_SAMPLE`] rounds back to back of
/// `bytes` each ([`page_round`]), the median sample's rate. A release is
/// help counted on, so a smaller rate is the safe side: the median, never
/// the fastest.
fn page_release(bytes: usize) -> f64 {
    page_round(bytes);
    let mut samples = (0..PAGE_RELEASE_SAMPLES)
        .map(|_| {
            let released = (0..PAGE_RELEASE_ROUNDS_PER_SAMPLE).map(|_| page_round(bytes)).sum::<Duration>();
            (PAGE_RELEASE_ROUNDS_PER_SAMPLE * bytes) as f64 / released.as_secs_f64().max(f64::MIN_POSITIVE)
        })
        .collect::<Vec<_>>();
    samples.sort_by(f64::total_cmp);
    samples[PAGE_RELEASE_SAMPLES / 2]
}

/// One round of the page-release probe: a byte written to every page of
/// `bytes` of new memory, then the memory freed, the free timed.
fn page_round(bytes: usize) -> Duration {
    let page = 4096;
    let layout = std::alloc::Layout::from_size_align(bytes, page).expect("the probe's layout");
    // SAFETY: a non-zero layout; every write lies within it, and it is
    // freed with the same layout.
    unsafe {
        let memory = std::alloc::alloc(layout);
        assert!(!memory.is_null(), "the page-release probe could not take {bytes} bytes");
        for offset in (0..bytes).step_by(page) {
            std::ptr::write_volatile(memory.add(offset), 1);
        }
        let started = Instant::now();
        std::alloc::dealloc(memory, layout);
        started.elapsed()
    }
}

/// A store's motion as a reader that observes it continuously measures it
/// (the service's memory guard, a volume's writer): readings folded in as
/// they come, beside the prior (the machine's page-fill rate, a volume's
/// write rate), which bounds a take until a faster one is seen.
///
/// What other programs took is judged on available bytes plus what the
/// reading counts as our own outside them (`own`: what this host's loads
/// hold, what a writer wrote): our import, allocation or write lowers the
/// one and raises the other by the same bytes, so only another program's
/// take lowers the sum. A reading
/// that counts no such holding (`None`) measures nothing, nor is it
/// measured against.
#[derive(Clone, Copy, Debug)]
pub struct MotionMeter {
    prior: StoreMotion,
    /// The latest reading's time, and the available memory plus own holding
    /// it counted, if it counted one.
    last: Option<(Instant, Option<u64>)>,
    fluctuation: u64,
    fastest: f64,
}

impl MotionMeter {
    pub fn new(prior: StoreMotion) -> Self {
        Self { prior, last: None, fluctuation: 0, fastest: 0.0 }
    }

    /// Fold in a reading at `at` of `available` bytes, beside `own` bytes
    /// this host's own loads hold outside them, where the reading counts
    /// them. A fall of the sum within a reading interval is steady
    /// fluctuation; over a longer span, a take at its rate where it exceeds
    /// the fluctuation.
    pub fn read(&mut self, at: Instant, available: u64, own: Option<u64>) {
        let judged = own.map(|own| available.saturating_add(own));
        if let (Some((then, Some(was))), Some(now)) = (self.last, judged) {
            let fall = was.saturating_sub(now);
            let span = at.duration_since(then);
            if span <= READING_INTERVAL {
                self.fluctuation = self.fluctuation.max(fall);
            } else if fall > self.fluctuation {
                self.fastest = self.fastest.max(fall as f64 / span.as_secs_f64());
            }
        }
        self.last = Some((at, judged));
    }

    /// The motion measured so far, never below the prior.
    pub fn motion(&self) -> StoreMotion {
        let measured = (self.fastest > 0.0).then(|| StoreMotion::new(self.fluctuation, self.fastest));
        measured.map_or(StoreMotion { fluctuation: self.fluctuation, ..self.prior }, |measured| self.prior.with(measured))
    }
}

/// What each memory domain keeps free beyond what a load holds, derived and
/// never a fraction of capacity: a store's planning line is its margin.
/// Host RAM's headroom is what the OS gives without displacing another
/// program's data ([`HostStatus::allocatable`]), and a device's driver keeps
/// its own reserve outside what it grants or reports free, so neither keeps
/// a line of its own beneath the margin:
/// - host RAM's margin is its motion's ([`StoreMotion::margin`]) over a
///   reading interval and a release's latency: none before a plan
///   estimates one, the plan's decode step (a bank shrink installs between
///   steps), then what the load measured;
/// - a device's own memory's margin is its motion's once a load measured
///   it; before that, none: the driver reports what other processes hold of
///   it at every reading.
///
/// A volume's reserve is not these: it belongs to the volume's writer
/// ([`super::VolumeReserve`]), which reads the volume's motion as it writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoryReserves {
    host: StoreMotion,
    device: Option<StoreMotion>,
    interval: Duration,
    latency: Duration,
    release: ReleaseRate,
    releasable: Releasable,
}

/// One domain's thresholds. Stable fit and every engine claim keep headroom
/// above `planning_bytes`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DomainThresholds {
    pub planning_bytes: u64,
}

impl DomainThresholds {
    /// What a store with `headroom_bytes` left lets a claim take: its
    /// headroom above the planning reserve.
    pub fn ceiling(&self, headroom_bytes: u64) -> u64 {
        headroom_bytes.saturating_sub(self.planning_bytes)
    }
}

impl MemoryReserves {
    /// The reserves of a host moving as `host` measures, a load releasing
    /// at `release`: read every [`READING_INTERVAL`], no release latency
    /// estimated yet, nothing releasable known.
    pub fn new(host: StoreMotion, release: ReleaseRate) -> Self {
        Self { host, device: None, interval: READING_INTERVAL, latency: Duration::ZERO, release, releasable: Releasable::default() }
    }

    /// The reserves of this machine before any reading of its motion or a
    /// load's release ([`StoreMotion::of_machine`],
    /// [`ReleaseRate::of_machine`]).
    pub fn of_machine(machine: Option<(&std::path::Path, &DeviceTopology)>) -> Self {
        Self::new(StoreMotion::of_machine(machine), ReleaseRate::of_machine(machine))
    }

    /// These reserves, a release taking `latency`: the plan's estimated
    /// decode step, or a load's measured release.
    pub fn with_latency(self, latency: Duration) -> Self {
        Self { latency, ..self }
    }

    /// These reserves with what a serving load measured: host RAM's motion
    /// and a device's own memory's, each never below what was known.
    pub fn measured(self, host: Option<StoreMotion>, device: Option<StoreMotion>) -> Self {
        let device = match (self.device, device) {
            (Some(known), Some(measured)) => Some(known.with(measured)),
            (known, measured) => known.or(measured),
        };
        Self { host: host.map_or(self.host, |measured| self.host.with(measured)), device, ..self }
    }

    /// These reserves with what a serving load measured of its releases:
    /// their rate, never above what was known, and the bytes it can
    /// release now.
    pub fn releasing(self, release: Option<ReleaseRate>, releasable: Releasable) -> Self {
        Self { release: release.map_or(self.release, |measured| self.release.with(measured)), releasable, ..self }
    }

    /// Host RAM's motion.
    pub fn host_motion(&self) -> StoreMotion {
        self.host
    }

    /// The rate a load releases at.
    pub fn release_rate(&self) -> ReleaseRate {
        self.release
    }

    /// The margin over `host_ram`'s line ([`StoreMotion::margin`]).
    #[cfg(test)]
    pub fn host_margin(&self, host_ram: &Store) -> u64 {
        self.thresholds(host_ram).planning_bytes
    }

    /// The thresholds of host RAM or a device's own memory: its margin
    /// ([`Self::margin_terms`]).
    ///
    /// # Panics
    /// For a volume, whose reserve is its writer's
    /// ([`super::VolumeReserves`]).
    pub fn thresholds(&self, store: &Store) -> DomainThresholds {
        DomainThresholds { planning_bytes: self.margin_terms(store).total() }
    }

    /// The margin over `store`'s line by its terms ([`StoreMotion::terms`]):
    /// none for a device's own memory before its motion is measured.
    ///
    /// # Panics
    /// For a volume, whose reserve is its writer's
    /// ([`super::VolumeReserves`]).
    pub fn margin_terms(&self, store: &Store) -> MarginTerms {
        let (motion, bytes) = match store.kind() {
            StoreKind::HostRam => (Some(self.host), self.releasable.host),
            StoreKind::DeviceMemory(_) => (self.device, self.releasable.device),
            StoreKind::Volume(_) => panic!("a volume's reserve is its writer's (`VolumeReserves`), never a load's"),
        };
        let releasing = Releasing { rate: self.release, bytes, store_bytes: store.capacity().value };
        motion.map_or_else(MarginTerms::default, |motion| motion.terms(self.interval, self.latency, Some(releasing)))
    }

    /// Stable identity of the reserves assessments read, for assessment
    /// cache keys: the rule, not its measured motion, which moves with the
    /// host from reading to reading and which a load reads live.
    pub fn identity(&self) -> String {
        "reserves-v5".to_owned()
    }
}

/// Why fit could not answer: never a verdict that a demand does not fit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryPolicyError {
    /// A required observation is unavailable; missing observations never
    /// authorize allocation.
    Observation(ObservationError),
    /// The fit subject belongs to another topology revision.
    ForeignIdentifier,
    /// Storage demanded of an executor that places none of that kind: an
    /// unavailable executor.
    Unplaced {
        executor: ExecutorId,
        kind: Contract,
    },
}

impl fmt::Display for MemoryPolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Observation(error) => error.fmt(f),
            Self::ForeignIdentifier => {
                f.write_str("fit subject belongs to another topology revision")
            }
            Self::Unplaced { executor, kind } => {
                write!(f, "executor {executor:?} places no {kind:?} storage")
            }
        }
    }
}

impl std::error::Error for MemoryPolicyError {}

/// How long memory must stay free (headroom above the planning reserve) to
/// count as recovered.
pub const RECOVERY_STABLE_TIME: Duration = Duration::from_secs(5);

/// Whether memory has recovered (stayed free): free at every reading
/// for [`RECOVERY_STABLE_TIME`]. What a service closed by memory pressure
/// reopens load admission on. `T` is the caller's clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Recovery<T> {
    free_since: Option<T>,
}

impl<T: Copy> Recovery<T> {
    pub const fn new() -> Self {
        Self { free_since: None }
    }

    /// A reading at `now`, memory `free` or not; `elapsed(since, now)` the
    /// clock's time between two readings. Whether memory has recovered.
    pub fn observe(&mut self, free: bool, now: T, elapsed: impl FnOnce(T, T) -> Duration) -> bool {
        if !free {
            self.free_since = None;
            return false;
        }
        let since = *self.free_since.get_or_insert(now);
        elapsed(since, now) >= RECOVERY_STABLE_TIME
    }
}

/// What a store holds for an executor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DomainRole {
    /// The executor's device storage is placed on it.
    Allocation,
    /// Host-side work (staging, host tables) and nothing of device storage.
    Staging,
}

impl DomainRole {
    /// A store `executor`'s device storage is placed on holds its
    /// allocations; any other store it charges holds host-side work.
    pub fn of(topology: &DeviceTopology, executor: &Executor, store: StoreId) -> Self {
        let allocates = executor.paths(seismic::Placing::Fixed(Contract::DeviceOnly)).iter().any(|id| {
            topology
                .access_path(*id)
                .expect("placement belongs to its topology")
                .store()
                == store
        });
        if allocates {
            Self::Allocation
        } else {
            Self::Staging
        }
    }
}

/// What bounds a fit: a store's own room or an access path's grant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryConstraint {
    /// Host RAM and its applicable process limits.
    HostRam,
    /// A device memory store: memory the OS does not manage.
    DeviceLocal,
    /// What the executor's driver lets this process place along an access
    /// path (CUDA's usable memory, a Vulkan heap's budget, Metal's working
    /// set), as reported and, where a reading exists, what is left of it.
    Grant,
    /// A volume's space available to this process.
    Volume,
}

impl fmt::Display for MemoryConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::HostRam => "host RAM",
            Self::DeviceLocal => "device memory",
            Self::Grant => "executor placement grant",
            Self::Volume => "volume space",
        })
    }
}

/// One store's fit: its capacity within process limits and its live
/// headroom, each less the planning reserve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoreFit {
    pub store: StoreId,
    /// The store's capacity, within the process limits for host RAM.
    pub capacity_bytes: u64,
    /// Capacity less the planning reserve: what metadata-only fit plans on.
    pub stable_bytes: u64,
    /// Observed available bytes, bounded by every applicable process limit.
    /// Existing Seismic charges are already absent from it.
    pub headroom_bytes: u64,
    /// Headroom less the planning reserve: what a live claim may take.
    pub ceiling_bytes: u64,
    pub thresholds: DomainThresholds,
    /// What bounds the store's room.
    pub constraint: MemoryConstraint,
}

/// What an executor's placement of one storage kind holds, stable and live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlacementFit {
    /// Along the placement, each grant within its store's stable fit.
    pub stable_bytes: u64,
    /// Along the placement, what is left of each grant within its store's
    /// live ceiling: what a live claim may take.
    pub ceiling_bytes: u64,
}

/// Whether fit plans on stable capacity or on live headroom.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitMode {
    /// Capacity within process limits and grants, less the planning
    /// reserve: no current free memory.
    Stable,
    /// Observed headroom and what is left of each grant, less the planning
    /// reserve.
    Live,
}

/// Bytes of one storage kind on one executor. Demands are placed in order,
/// sharing store and path budgets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StorageDemand {
    /// The executor whose placement the bytes go along.
    pub executor: ExecutorId,
    /// The storage kind, whose placement it is.
    pub kind: Contract,
    pub bytes: u64,
}

/// A demand that does not fit, by the store where it ran out of room.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FitRefusal {
    /// The last store the demand reached.
    pub store: StoreId,
    /// The demand's bytes.
    pub required_bytes: u64,
    /// Its bytes no store or grant had room for.
    pub missing_bytes: u64,
    /// What ran out: the store's room or a grant.
    pub constraint: MemoryConstraint,
}

/// Where a demand's bytes landed, and what of it did not fit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DemandFit {
    /// The bytes placed in each store, in store order.
    pub placed: Vec<(StoreId, u64)>,
    /// Each demand some of whose bytes did not fit.
    pub refused: Vec<FitRefusal>,
}

impl DemandFit {
    /// Whether every byte of the demand was placed.
    pub fn fits(&self) -> bool {
        self.refused.is_empty()
    }
}

/// Which of host RAM's room a fit question reads: what the OS gives without
/// compressing or swapping another program's data
/// ([`HostStatus::allocatable`]), the one measure every question reads.
/// [`Fit::new`] takes it for the assessment's questions, which name it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostRoom {
    /// Allocatable memory as headroom.
    Allocatable,
}

/// The sole interpretation of stores and grants for engine fit questions.
pub struct Fit<'a> {
    topology: &'a DeviceTopology,
    status: &'a Status,
    reserves: &'a MemoryReserves,
    /// The reserves of the volumes this process writes, where it reads them.
    volumes: Option<&'a super::VolumeReserves>,
}

impl<'a> Fit<'a> {
    /// Fit over `topology`'s stores and grants as `status` reads them, with
    /// `reserves`, host RAM's room `room`.
    pub fn new(
        topology: &'a DeviceTopology,
        status: &'a Status,
        reserves: &'a MemoryReserves,
        room: HostRoom,
    ) -> Self {
        let HostRoom::Allocatable = room;
        Self {
            topology,
            status,
            reserves,
            volumes: None,
        }
    }

    /// This fit, reading volumes against `volumes`' reserves: a volume
    /// this process does not write keeps nothing free for it.
    pub fn with_volumes(self, volumes: &'a super::VolumeReserves) -> Self {
        Self { volumes: Some(volumes), ..self }
    }

    /// Store `id`'s fit: its capacity within process limits (host RAM's)
    /// and its headroom ([`Status::headroom`], within what is left of every
    /// grant on it), each less its planning reserve.
    pub fn store(&self, id: StoreId) -> Result<StoreFit, MemoryPolicyError> {
        let store = self
            .topology
            .store(id)
            .ok_or(MemoryPolicyError::ForeignIdentifier)?;
        let (constraint, thresholds) = match store.kind() {
            StoreKind::HostRam => {
                let host = self
                    .status
                    .sample
                    .host_ram
                    .as_ref()
                    .map_err(|error| MemoryPolicyError::Observation(error.clone()))?;
                return Ok(host_ram_fit(store, host, self.reserves));
            }
            // A device's own memory has no line of its own: the driver keeps
            // its reserve outside what it reports free.
            StoreKind::DeviceMemory(_) => (MemoryConstraint::DeviceLocal, self.reserves.thresholds(store)),
            // Nor has a volume (`super::volume`): it keeps the margin its
            // writes need, and none where this process writes nothing.
            StoreKind::Volume(volume) => {
                let margin = self.volumes.and_then(|volumes| volumes.margin(volume)).unwrap_or(0);
                (MemoryConstraint::Volume, DomainThresholds { planning_bytes: margin })
            }
        };
        let headroom_bytes = self
            .status
            .headroom(store, seismic::Reclaim::CleanFileCache)
            .map_err(MemoryPolicyError::Observation)?;
        // A store's own figure knows only this process; what is left of a
        // grant on it is the driver's, which other processes' memory lowers.
        let headroom_bytes = self
            .topology
            .access_paths()
            .iter()
            .filter(|path| path.store() == id && self.status.sample.executors.contains(&path.executor()))
            .filter_map(|path| self.status.grant_left(path))
            .try_fold(headroom_bytes, |least, left| match left {
                Ok(left) => Ok(least.min(left.value)),
                Err(error) => Err(MemoryPolicyError::Observation(error.clone())),
            })?;
        Ok(store_fit(store, thresholds, store.capacity().value, headroom_bytes, constraint))
    }

    /// Host RAM's fit from a host sample alone, as [`Fit::store`] gives it
    /// from a whole status: for a reader that opens no executor (the
    /// residency supervisor), which has only `host`.
    pub fn host_ram(topology: &DeviceTopology, host: &HostStatus, reserves: &MemoryReserves) -> StoreFit {
        host_ram_fit(topology.host_ram(), host, reserves)
    }

    /// What access path `id`'s grant leaves a claim: the reported grant,
    /// within what a live reading says is left of it, less its store's
    /// planning reserve. A grant bounds its store as a store's headroom does,
    /// so it keeps the same reserve: a grant filled to its end fails the
    /// device's own work beside the charges (Metal's submission memory beyond
    /// its working set loses the device). An ungranted path is bounded by its
    /// store alone.
    fn grant(&self, id: AccessPathId, mode: FitMode) -> Result<u64, MemoryPolicyError> {
        let path = self
            .topology
            .access_path(id)
            .ok_or(MemoryPolicyError::ForeignIdentifier)?;
        let Some(grant) = path.grant() else {
            return Ok(u64::MAX);
        };
        let left = match mode {
            FitMode::Stable => grant.value,
            FitMode::Live => match self.status.grant_left(path) {
                Some(Ok(left)) => grant.value.min(left.value),
                Some(Err(error)) => return Err(MemoryPolicyError::Observation(error.clone())),
                // An unopened executor has only its reported stable grant.
                None => grant.value,
            },
        };
        let reserve = self.store(path.store())?.thresholds.planning_bytes;
        Ok(left.saturating_sub(reserve))
    }

    /// What `executor`'s placement of `kind` storage holds: the sum along
    /// its paths of each grant within its store's fit.
    pub fn placement(
        &self,
        executor: &Executor,
        kind: Contract,
    ) -> Result<PlacementFit, MemoryPolicyError> {
        self.storage_beside(executor, &[kind], &[])
    }

    /// What `executor`'s placements of `kinds` hold together beside the
    /// direct demands `direct`, which take their stores' room first. A
    /// store or grant several kinds reach is counted once.
    pub fn storage_beside(
        &self,
        executor: &Executor,
        kinds: &[Contract],
        direct: &[(StoreId, u64)],
    ) -> Result<PlacementFit, MemoryPolicyError> {
        let demands: Vec<_> = kinds
            .iter()
            .map(|&kind| StorageDemand {
                executor: executor.id(),
                kind,
                bytes: u64::MAX,
            })
            .collect();
        let available = |mode| -> Result<u64, MemoryPolicyError> {
            let (_, storage) = self.place(&demands, direct, mode)?;
            Ok(storage.into_iter().sum())
        };
        Ok(PlacementFit {
            stable_bytes: available(FitMode::Stable)?,
            ceiling_bytes: available(FitMode::Live)?,
        })
    }

    /// The host RAM demand of `staged` bytes that `executor`'s backend
    /// stages while it fills device storage. Transfer staging is in host
    /// RAM on every backend (§2.3), and every claim's staged bytes are part
    /// of its device storage demand. Where host RAM is a store of the
    /// executor's device placement ([`DomainRole::Allocation`]), that demand
    /// already holds them in host RAM; only where host RAM is the
    /// executor's staging store are they a host demand of their own.
    pub fn host_staging(&self, executor: &Executor, staged: u64) -> u64 {
        match DomainRole::of(self.topology, executor, self.topology.host_ram().id()) {
            DomainRole::Allocation => 0,
            DomainRole::Staging => staged,
        }
    }

    /// Direct demands (host heap, staging, volume downloads) and executor
    /// storage share one budget per store and one budget per access grant.
    pub fn demand(
        &self,
        storage: &[StorageDemand],
        direct: &[(StoreId, u64)],
        mode: FitMode,
    ) -> Result<DemandFit, MemoryPolicyError> {
        self.place(storage, direct, mode).map(|(fit, _)| fit)
    }

    /// [`Self::demand`], with the bytes placed of each storage demand in
    /// order.
    fn place(
        &self,
        storage: &[StorageDemand],
        direct: &[(StoreId, u64)],
        mode: FitMode,
    ) -> Result<(DemandFit, Vec<u64>), MemoryPolicyError> {
        let room_of = |fit: &StoreFit| match mode {
            FitMode::Stable => fit.stable_bytes,
            FitMode::Live => fit.ceiling_bytes,
        };
        let mut stores = BTreeMap::<StoreId, u64>::new();
        let mut grants = BTreeMap::<AccessPathId, u64>::new();
        let mut placed = BTreeMap::<StoreId, u64>::new();
        let mut placed_storage = Vec::with_capacity(storage.len());
        let mut refused = Vec::new();
        for &(store, bytes) in direct {
            let fit = self.store(store)?;
            let room = stores.entry(store).or_insert(room_of(&fit));
            let take = bytes.min(*room);
            *room -= take;
            *placed.entry(store).or_default() += take;
            if take < bytes {
                refused.push(FitRefusal {
                    store,
                    required_bytes: bytes,
                    missing_bytes: bytes - take,
                    constraint: fit.constraint,
                });
            }
        }
        for demand in storage {
            let executor = self
                .topology
                .executor(demand.executor)
                .ok_or(MemoryPolicyError::ForeignIdentifier)?;
            let mut left = demand.bytes;
            let mut last = None;
            for &id in executor.paths(seismic::Placing::Fixed(demand.kind)) {
                let path = self
                    .topology
                    .access_path(id)
                    .expect("placement belongs to its topology");
                let fit = self.store(path.store())?;
                let room = stores.entry(path.store()).or_insert(room_of(&fit));
                let grant = grants.entry(id).or_insert(self.grant(id, mode)?);
                let constraint = if *room <= *grant {
                    fit.constraint
                } else {
                    MemoryConstraint::Grant
                };
                let take = left.min(*room).min(*grant);
                *room -= take;
                *grant -= take;
                left -= take;
                *placed.entry(path.store()).or_default() += take;
                last = Some((path.store(), constraint));
                if left == 0 {
                    break;
                }
            }
            placed_storage.push(demand.bytes - left);
            if left != 0 {
                let (store, constraint) = last.ok_or(MemoryPolicyError::Unplaced {
                    executor: demand.executor,
                    kind: demand.kind,
                })?;
                refused.push(FitRefusal {
                    store,
                    required_bytes: demand.bytes,
                    missing_bytes: left,
                    constraint,
                });
            }
        }
        let fit = DemandFit {
            placed: placed.into_iter().collect(),
            refused,
        };
        Ok((fit, placed_storage))
    }
}

/// One fresh observation of a store an opened executor uses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DomainReading {
    /// What the store holds for the executor.
    pub role: DomainRole,
    pub domain: StoreId,
    /// Observed available bytes, bounded by every applicable process limit.
    /// Existing Seismic charges are already absent from it.
    pub headroom_bytes: u64,
    /// The store across revisions, as other processes name it.
    pub identity: seismic::StoreIdentity,
    /// What bounds `ceiling_bytes`.
    pub constraint: MemoryConstraint,
    pub thresholds: DomainThresholds,
    /// Additional bytes the engine may still claim in the store: its
    /// headroom above the planning reserve.
    pub ceiling_bytes: u64,
    /// What the OS held outside headroom, for host RAM on a platform that
    /// counts it.
    pub held: Option<seismic::HostHeld>,
}

impl fmt::Display for DomainReading {
    /// The reading a claim's room was judged on.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} headroom {} bytes against a planning reserve of {}",
            self.constraint, self.headroom_bytes, self.thresholds.planning_bytes
        )?;
        if let Some(held) = self.held {
            write!(
                f,
                "; the host holds {} bytes wired and {} compressed",
                held.wired_bytes, held.compressed_bytes
            )?;
        }
        Ok(())
    }
}

/// Fresh store readings for the executor's placements and host-side work.
/// Allocation roles identify stores reached by device storage, not a
/// promise that device storage occupies only one store.
pub fn observe_domains(
    devices: &seismic::Devices,
    device: &seismic::Device,
    reserves: &MemoryReserves,
) -> Result<Vec<DomainReading>, MemoryPolicyError> {
    let status = devices.device_status(device);
    domain_readings(device, &status, reserves)
}

/// The readings of every store `device` places storage on, and host RAM.
pub(crate) fn domain_readings(
    device: &seismic::Device,
    status: &Status,
    reserves: &MemoryReserves,
) -> Result<Vec<DomainReading>, MemoryPolicyError> {
    let topology = device.topology();
    let fit = Fit::new(topology, status, reserves, HostRoom::Allocatable);
    let stores: std::collections::BTreeSet<_> = crate::memory::CLAIMED
        .iter()
        .flat_map(|kind| device.info().paths(seismic::Placing::Fixed(*kind)))
        .map(|id| {
            topology
                .access_path(*id)
                .expect("placement belongs to its topology")
                .store()
        })
        .chain(std::iter::once(topology.host_ram().id()))
        .collect();
    let readings: Result<Vec<_>, MemoryPolicyError> = stores
        .into_iter()
        .map(|store| {
            let reading = fit.store(store)?;
            // Host RAM's fit read the host sample; other stores have none.
            let host = (reading.constraint == MemoryConstraint::HostRam).then(|| {
                status
                    .sample
                    .host_ram
                    .as_ref()
                    .expect("host RAM's fit read its sample")
            });
            Ok(DomainReading {
                role: DomainRole::of(topology, device.info(), store),
                domain: store,
                headroom_bytes: reading.headroom_bytes,
                identity: topology
                    .store_identity(store)
                    .expect("observed store belongs to topology"),
                constraint: reading.constraint,
                thresholds: reading.thresholds,
                ceiling_bytes: reading.ceiling_bytes,
                held: host.and_then(|host| host.held),
            })
        })
        .collect();
    // Opt-in investigation trace: the exact sample and readings that decide
    // the ceiling, rather than a sample taken again after a failed worker
    // has exited.
    if std::env::var_os("MAGNITUDE_TRACE_MEMORY_BUDGET").is_some() {
        eprintln!(
            "magnitude-engine: memory budget pid={} device={} charged={:?} status={status:?} \
             readings={readings:?}",
            std::process::id(),
            device.info().selector(),
            device.ledgers().own_claims(),
        );
    }
    readings
}

#[cfg(test)]
mod tests {
    //! Fit over the recorded machines' topologies
    //! (`seismic::fixtures::recorded`). Status
    //! values (headroom, charges, what is left of a grant) are each test's
    //! input; every structural value is the machine's.
    use super::*;
    use seismic::fixtures::recorded;
    use seismic::{
        BackendName, LimitVisibility, ProcessLimitKind, ProcessMemoryLimit,
        Reported, ReportedBy, RuntimeStatus, Sample,
    };
    use std::time::SystemTime;

    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;

    const LINUX: ReportedBy = ReportedBy::LinuxMemAvailable;
    const MACOS: ReportedBy = ReportedBy::MachAllocatablePages;

    /// Memory recovers once free for the stable time at every reading; one
    /// reading that is not free starts it over.
    #[test]
    fn memory_recovers_once_it_stays_free() {
        let stable = RECOVERY_STABLE_TIME.as_secs();
        let elapsed = |since: u64, now: u64| Duration::from_secs(now - since);
        let mut recovery = Recovery::new();
        assert!(!recovery.observe(true, 0, elapsed));
        assert!(!recovery.observe(false, stable, elapsed));
        assert!(!recovery.observe(true, stable + 1, elapsed));
        assert!(!recovery.observe(true, 2 * stable, elapsed));
        assert!(recovery.observe(true, 2 * stable + 1, elapsed));
    }
    const WINDOWS: ReportedBy = ReportedBy::WindowsPhysicalAndCommit;

    /// Host RAM with `headroom` bytes this process can take, as `by`
    /// reports it, without limits.
    fn host(by: ReportedBy, headroom: u64) -> HostStatus {
        HostStatus {
            allocatable: Reported::new(headroom, by),
            reclaimable: 0,
            limits: Vec::new(),
            limit_visibility: LimitVisibility::Complete,
            held: None,
        }
    }

    /// Reserves whose host motion holds exactly `margin` bytes over a
    /// reading interval: one byte taken an interval, the rest fluctuation.
    fn holding(margin: u64) -> MemoryReserves {
        MemoryReserves::new(StoreMotion::new(margin - 1, 10.0), crate::platform::ReleaseRate::new(1e9))
    }

    /// The reserves these tests read where no figure depends on them: a
    /// host whose motion holds 2 GiB.
    fn reserves() -> MemoryReserves {
        holding(2 * GIB)
    }

    /// Status of an unopened machine: `host` alone, no grant read and no
    /// charge.
    fn status(host: HostStatus) -> Status {
        Status {
            sample: Sample {
                taken: SystemTime::now(),
                host_ram: Ok(host),
                volumes: Vec::new(),
                executors: Vec::new(),
                grants: Vec::new(),
            },
            runtime: RuntimeStatus::default(),
        }
    }

    /// Status of an unopened machine sampled for every executor of
    /// `topology`: `host` alone, no grant read and no charge.
    fn sampled(topology: &DeviceTopology, host: HostStatus) -> Status {
        let mut status = status(host);
        status.sample.executors = topology
            .executors()
            .iter()
            .map(|executor| executor.id())
            .collect();
        status
    }

    /// The first executor of `backend`.
    fn executor(topology: &DeviceTopology, backend: BackendName) -> &Executor {
        topology
            .executors()
            .iter()
            .find(|executor| executor.backend() == backend)
            .unwrap()
    }

    /// The machine's one device memory store.
    fn device_memory(topology: &DeviceTopology) -> StoreId {
        let stores = topology
            .stores()
            .iter()
            .filter(|store| matches!(store.kind(), StoreKind::DeviceMemory(_)))
            .map(|store| store.id())
            .collect::<Vec<_>>();
        let [store] = stores[..] else {
            panic!("the machine has one device memory store");
        };
        store
    }

    /// What the reported grant of the `index`th path of `executor`'s
    /// placement of `kind` lets a claim take: the grant less its store's
    /// planning reserve.
    fn grant_room(
        topology: &DeviceTopology,
        executor: &Executor,
        kind: Contract,
        index: usize,
    ) -> u64 {
        let path = topology
            .access_path(executor.paths(seismic::Placing::Fixed(kind))[index])
            .unwrap();
        let store = topology.store(path.store()).unwrap();
        let reserve = reserves().thresholds(store).planning_bytes;
        path.grant().unwrap().value - reserve
    }

    fn demand(executor: &Executor, kind: Contract, bytes: u64) -> StorageDemand {
        StorageDemand {
            executor: executor.id(),
            kind,
            bytes,
        }
    }

    /// The bytes `fit` placed on `store`.
    fn placed(fit: &DemandFit, store: StoreId) -> u64 {
        fit.placed
            .iter()
            .filter(|(on, _)| *on == store)
            .map(|(_, bytes)| bytes)
            .sum()
    }

    /// The instant the meter tests read from.
    fn epoch() -> Instant {
        static EPOCH: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        *EPOCH.get_or_init(Instant::now)
    }

    /// Another process's memory on a discrete device reaches its store's
    /// fit through what the driver says is left of the grant, which the
    /// process's own charges cannot show.
    #[test]
    fn a_stores_headroom_follows_what_is_left_of_its_grants() {
        let reserves = reserves();
        let rtx3090 = recorded::rtx3090();
        let cuda = executor(&rtx3090, BackendName::Cuda);
        let path = rtx3090.access_path(cuda.paths(seismic::Placing::Fixed(Contract::DeviceOnly))[0]).unwrap();
        let mut opened = sampled(&rtx3090, host(LINUX, 40 * GIB));
        let store = path.store();
        let room = |status: &Status| Fit::new(&rtx3090, status, &reserves, HostRoom::Allocatable).store(store).unwrap().headroom_bytes;
        let capacity = room(&opened);
        opened.runtime.grants = vec![(path.id(), Ok(Reported::new(GIB, ReportedBy::CudaFree)))];
        assert!(capacity > GIB);
        assert_eq!(room(&opened), GIB);
    }

    /// Each store's planning line is its motion's margin, never a fraction
    /// of capacity: a device's own memory has no margin until a load
    /// measured its motion; a volume's is its writer's. The margin is the
    /// fluctuation plus what the host takes at its take rate over a reading
    /// interval and a release's latency, so a positive take rate leaves no
    /// reserve without one; and a measurement only raises it.
    #[test]
    fn reserves_are_the_motions_margin() {
        // 10 GB a second, over 0.1 s, then over 0.1 s and a 0.4 s release.
        let motion = StoreMotion::new(GIB, 10e9);
        let unplanned = MemoryReserves::new(motion, crate::platform::ReleaseRate::new(1e9));
        let planned = unplanned.with_latency(Duration::from_millis(400));
        let m4_pro = recorded::m4_pro();
        let host_ram = m4_pro.host_ram();
        assert_eq!(unplanned.host_margin(host_ram), GIB + 1_000_000_000);
        assert_eq!(planned.host_margin(host_ram), GIB + 5_000_000_000);
        // A host of another size keeps the same line.
        let lunar_lake = recorded::lunar_lake();
        assert_eq!(planned.thresholds(lunar_lake.host_ram()), planned.thresholds(host_ram));
        // A slower measurement lowers nothing; a faster one raises the margin.
        let slower = planned.measured(Some(StoreMotion::new(0, 1e9)), None);
        assert_eq!(slower.host_margin(host_ram), planned.host_margin(host_ram));
        let faster = planned.measured(Some(StoreMotion::new(0, 20e9)), None);
        assert_eq!(faster.host_margin(host_ram), GIB + 10_000_000_000);
        let rtx3090 = recorded::rtx3090();
        let device = rtx3090.store(device_memory(&rtx3090)).unwrap();
        assert_eq!(planned.thresholds(device), DomainThresholds { planning_bytes: 0 });
        let measured = planned.measured(None, Some(StoreMotion::new(0, 1e9)));
        assert_eq!(measured.thresholds(device), DomainThresholds { planning_bytes: 500_000_000 });
    }

    /// The margin over the take before a first release lands, and while
    /// the host takes faster than the load releases, the deficit's growth
    /// over the releasing: none where the load releases as fast, growing
    /// with the bytes released where it does not, and smaller the faster
    /// the release.
    #[test]
    fn the_margin_covers_a_take_outrunning_the_release() {
        let interval = Duration::from_millis(100);
        let latency = Duration::from_millis(100);
        let motion = StoreMotion::new(MIB, 4e9);
        let lead = MIB + 800_000_000;
        let releasing = |rate: f64, bytes: u64| Some(Releasing { rate: ReleaseRate::new(rate), bytes, store_bytes: 64 * GIB });
        assert_eq!(motion.margin(interval, latency, None), lead);
        for bytes in [0, GIB, 10 * GIB] {
            assert_eq!(motion.margin(interval, latency, releasing(4e9, bytes)), lead, "t ≤ r: no shortfall");
            assert_eq!(motion.margin(interval, latency, releasing(8e9, bytes)), lead);
        }
        // t = 2r: the deficit grows by a byte per byte released.
        assert_eq!(motion.margin(interval, latency, releasing(2e9, GIB)), lead + GIB);
        assert_eq!(motion.margin(interval, latency, releasing(2e9, 4 * GIB)), lead + 4 * GIB);
        assert!(motion.margin(interval, latency, releasing(3e9, 4 * GIB)) < motion.margin(interval, latency, releasing(2e9, 4 * GIB)));
        // A measured release only ever slows the rate the reserves hold.
        let reserves = MemoryReserves::new(motion, ReleaseRate::new(8e9)).with_latency(latency);
        let serving = reserves.releasing(None, Releasable { host: 4 * GIB, device: 0 });
        let m4_pro = recorded::m4_pro();
        let host_ram = m4_pro.host_ram();
        assert_eq!(serving.host_margin(host_ram), lead);
        assert_eq!(serving.releasing(Some(ReleaseRate::new(2e9)), Releasable { host: 4 * GIB, device: 0 }).host_margin(host_ram), lead + 4 * GIB);
        assert_eq!(serving.releasing(Some(ReleaseRate::new(16e9)), Releasable { host: 4 * GIB, device: 0 }).release_rate(), ReleaseRate::new(8e9));
    }

    /// A burst is no sustained take: before another program is seen
    /// sustaining a take, the margin holds the burst only until a reading
    /// notices it, whatever the load's release latency. The inputs an
    /// uncapped Flash Next load on a 48 GiB M4 Pro read: a 28.8 GB/s
    /// page-fill rate, releases at 3.1 GB/s, 25.2 GB of its bank
    /// releasable; the burst held over the releasing would be a 208 GB
    /// margin, and over a 198 ms release latency 5.7 GB more.
    #[test]
    fn a_burst_takes_only_until_a_reading_notices_it() {
        let interval = Duration::from_millis(100);
        let store_bytes = 48 * GIB;
        let releasing = Some(Releasing { rate: ReleaseRate::new(3.1e9), bytes: 25_200_000_000, store_bytes });
        let burst = StoreMotion::prior(28.8e9);
        let lead = (28.8e9 * interval.as_secs_f64()) as u64;
        for latency in [Duration::ZERO, Duration::from_micros(6_048), Duration::from_millis(198)] {
            assert_eq!(burst.margin(interval, latency, releasing), lead, "{latency:?}");
        }
        assert!(lead < store_bytes);
    }

    /// The load's own release latency moves its line only by what another
    /// program is seen to sustain over it. On a 48 GB M4 Pro holding 25 GB
    /// for another program that took nothing more, Flash Next's bank
    /// released 63→36 slots with an 80 ms latency and 36→20 with 198 ms;
    /// at the page-fill rate over each latency the host line rose from
    /// 3.79 GB to 9.90 GB, past the 9.46 GB the releases had freed, and the
    /// load unloaded for memory pressure no other program made.
    #[test]
    fn a_release_latency_raises_the_line_only_by_a_sustained_take() {
        let m4_pro = recorded::m4_pro();
        let host_ram = m4_pro.host_ram();
        let opened = MemoryReserves::new(StoreMotion::prior(28.8e9), ReleaseRate::new(3.1e9))
            .releasing(None, Releasable { host: 4 * GIB, device: 0 });
        let line = opened.host_margin(host_ram);
        for latency in [Duration::from_millis(80), Duration::from_millis(198)] {
            let released = opened.with_latency(latency).releasing(Some(ReleaseRate::new(17e9)), Releasable { host: 4 * GIB, device: 0 });
            assert_eq!(released.host_margin(host_ram), line, "nothing else moves: {latency:?} raises nothing");
            // Another program seen taking 1 GB/s goes on taking over it.
            let taking = released.measured(Some(StoreMotion::new(0, 1e9)), None);
            let without = opened.measured(Some(StoreMotion::new(0, 1e9)), None);
            assert_eq!(taking.host_margin(host_ram), without.host_margin(host_ram) + (1e9 * latency.as_secs_f64()) as u64);
        }
    }

    /// A take another program sustains across readings faster than the load
    /// releases grows the deficit over the releasing: at twice the release
    /// rate, a byte per byte released, on the burst's lead over a reading
    /// interval and the sustained take's over the release latency.
    #[test]
    fn a_sustained_take_grows_the_margin_over_the_releasing() {
        let interval = Duration::from_millis(100);
        let latency = Duration::from_millis(100);
        let releasing = Some(Releasing { rate: ReleaseRate::new(3.1e9), bytes: 25_200_000_000, store_bytes: 48 * GIB });
        let measured = StoreMotion::prior(28.8e9).with(StoreMotion::new(0, 6.2e9));
        assert_eq!(measured.take_rate(), 28.8e9, "the burst stays the fastest seen");
        assert_eq!(measured.margin(interval, latency, releasing), 2_880_000_000 + 620_000_000 + 25_200_000_000);
    }

    /// A fill the region's end cuts short of its interval is censored, its
    /// bytes the region's; one the interval ends is not. The probe takes at
    /// most half the room it read, a sample at a time.
    #[test]
    fn a_page_fill_cut_by_its_bound_is_censored() {
        let page = host_page();
        assert_eq!(ColdRegion::map(16 * page).fill(page, Duration::from_secs(10)), (16 * page, true));
        assert_eq!(ColdRegion::map(1 << 30).fill(page, Duration::ZERO), (0, false));
        let probe = PageProbe::run();
        let bound = usize::try_from(probe.room / 2).unwrap().max(page);
        assert!(probe.fill_samples.iter().all(|&bytes| bytes <= bound), "{probe:?}");
    }

    /// The rest of the host never takes more of a store than it holds: a
    /// margin past it is a measurement that cannot be, an invariant
    /// violation (a panic in test builds). A 114 MB/s release of a 20 GB
    /// bank against a sustained 10 GB/s take, as a release rate read over
    /// its wait for queued steps made it, is a 1.7 TB margin in a 48 GB
    /// store.
    #[test]
    fn a_margin_past_its_store_is_an_invariant_violation() {
        let motion = StoreMotion::new(0, 10e9);
        let interval = Duration::from_millis(100);
        let latency = Duration::from_millis(165);
        let store_bytes = 48 * GIB;
        let releasing = |rate: f64| Some(Releasing { rate: ReleaseRate::new(rate), bytes: 20_000_000_000, store_bytes });
        assert!(motion.margin(interval, latency, releasing(17e9)) < store_bytes, "a release at its work's rate");
        let impossible = std::panic::catch_unwind(|| motion.margin(interval, latency, releasing(114e6)));
        let message = impossible.expect_err("a margin past its store panics in test builds");
        let message = message.downcast_ref::<String>().expect("a formatted invariant message");
        assert!(message.starts_with("invariant violated: a margin of"), "{message}");
    }

    /// A take rate is measured and positive: no motion without one is
    /// constructible, and a serialized one is refused.
    #[test]
    fn no_motion_without_a_take_rate() {
        assert!(std::panic::catch_unwind(|| StoreMotion::new(GIB, 0.0)).is_err());
        assert!(std::panic::catch_unwind(|| StoreMotion::new(GIB, f64::NAN)).is_err());
        assert!(serde_json::from_value::<StoreMotion>(serde_json::json!({ "fluctuation": 0, "take_rate": 0.0 })).is_err());
        let m4_pro = recorded::m4_pro();
        assert!(MemoryReserves::new(StoreMotion::new(0, 1e9), crate::platform::ReleaseRate::new(1e9)).host_margin(m4_pro.host_ram()) > 0);
    }

    /// The machine's page-fill rate is a positive rate, measured once: per
    /// process without a cache, once per machine with one.
    #[test]
    fn the_machine_fills_pages_at_a_measured_rate() {
        let machine = StoreMotion::of_machine(None);
        assert!(machine.take_rate() > 0.0 && machine.take_rate().is_finite());
        assert_eq!(StoreMotion::of_machine(None), machine);
        assert_eq!(machine.fluctuation(), 0);
        // A cache keeps the machine's rate, which a later reader reads
        // rather than measuring again.
        let cache = std::env::temp_dir().join(format!("page-fill-{}", std::process::id()));
        let topology = recorded::m4_pro();
        let stored = StoreMotion::of_machine(Some((&cache, &topology)));
        crate::kernel_cache::KernelCache::machine_figures(&cache)
            .store(&format!("page-fill-interval:{}", host_identity(&topology).unwrap()), &MachineRate { bytes_per_second: 7e9 }, MACHINE_RATE_RECORD_BYTES)
            .unwrap();
        assert!(stored.take_rate() > 0.0);
        assert_eq!(StoreMotion::of_machine(Some((&cache, &topology))).take_rate(), 7e9);
        std::fs::remove_dir_all(&cache).unwrap();
    }

    /// The meter folds a continuous observer's readings, judged on available
    /// memory plus what the host's own loads hold: a fall within a reading
    /// interval is fluctuation, a fall over a longer span a take at its
    /// rate; a reading that counts no own holding measures nothing; and the
    /// motion is never below the prior.
    #[test]
    fn the_meter_measures_fluctuation_and_takes() {
        let start = epoch();
        let at = |millis: u64| start + Duration::from_millis(millis);
        let mut meter = MotionMeter::new(StoreMotion::new(0, 5e9));
        meter.read(at(0), 20 * GIB, Some(0));
        meter.read(at(100), 20 * GIB - MIB, Some(0));
        assert_eq!(meter.motion(), StoreMotion::new(MIB, 5e9));
        // 4 GiB over a second, slower than the prior; then a reading that
        // counts no own holding, and one after it.
        meter.read(at(1_100), 16 * GIB - MIB, Some(0));
        meter.read(at(2_100), 4 * GIB, None);
        meter.read(at(3_100), 0, Some(0));
        assert_eq!(meter.motion(), StoreMotion::new(MIB, 5e9));
        // 10 GiB over a second, faster than the prior.
        let mut fast = MotionMeter::new(StoreMotion::new(0, GIB as f64));
        fast.read(at(0), 20 * GIB, Some(0));
        fast.read(at(1_000), 10 * GIB, Some(0));
        assert_eq!(fast.motion(), StoreMotion::new(0, 10.0 * GIB as f64));
    }

    /// The host's own load taking memory is no other program's take: its
    /// holding rises by what available memory falls, over any span.
    #[test]
    fn the_meter_measures_no_take_of_the_hosts_own_loads() {
        let start = epoch();
        let at = |millis: u64| start + Duration::from_millis(millis);
        let mut meter = MotionMeter::new(StoreMotion::new(0, GIB as f64));
        for (step, held) in [0, 4, 12, 20].into_iter().enumerate() {
            meter.read(at(step as u64 * 1_000), 30 * GIB - held * GIB, Some(held * GIB));
        }
        assert_eq!(meter.motion(), StoreMotion::new(0, GIB as f64));
    }

    /// Host RAM's stable fit is its capacity less the planning reserve, and
    /// its ceiling its headroom less the reserve, none at or below it.
    #[test]
    fn host_ram_ceiling_follows_the_planning_reserve() {
        let topology = recorded::rtx3090();
        let id = topology.host_ram().id();
        let capacity = topology.host_ram().capacity().value;
        let line = 2 * GIB;
        let reserves = holding(line);
        let reading = |headroom| {
            Fit::new(&topology, &status(host(LINUX, headroom)), &reserves, HostRoom::Allocatable)
                .store(id)
                .unwrap()
        };
        let above = reading(16 * GIB);
        assert_eq!(above.thresholds.planning_bytes, line);
        assert_eq!(above.stable_bytes, capacity - line);
        assert_eq!(above.ceiling_bytes, 16 * GIB - line);
        assert_eq!(above.constraint, MemoryConstraint::HostRam);
        assert_eq!(reading(line).ceiling_bytes, 0);
        // Stable fit does not follow headroom.
        let below = reading(line / 2);
        assert_eq!(below.ceiling_bytes, 0);
        assert_eq!(below.stable_bytes, above.stable_bytes);
    }

    /// Host RAM that could not be observed is an observation error, never a
    /// default.
    #[test]
    fn host_ram_without_an_observation_has_no_fit() {
        let topology = recorded::rtx3090();
        let mut unobserved = status(host(LINUX, 16 * GIB));
        let error = ObservationError::Failed("sample failed".into());
        unobserved.sample.host_ram = Err(error.clone());
        let reserves = reserves();
        let fit = Fit::new(&topology, &unobserved, &reserves, HostRoom::Allocatable);
        assert_eq!(
            fit.store(topology.host_ram().id()),
            Err(MemoryPolicyError::Observation(error.clone()))
        );
        let cpu = executor(&topology, BackendName::Cpu);
        assert_eq!(
            fit.placement(cpu, Contract::DeviceOnly),
            Err(MemoryPolicyError::Observation(error))
        );
    }

    /// A store an executor's device storage is placed on holds its
    /// allocations; any other store it charges holds host-side work.
    #[test]
    fn domain_roles_follow_the_device_storage_placement() {
        let rtx3090 = recorded::rtx3090();
        let (gpu, host_ram) = (device_memory(&rtx3090), rtx3090.host_ram().id());
        for backend in [BackendName::Cuda, BackendName::Vulkan] {
            let executor = executor(&rtx3090, backend);
            assert_eq!(
                DomainRole::of(&rtx3090, executor, gpu),
                DomainRole::Allocation
            );
            assert_eq!(
                DomainRole::of(&rtx3090, executor, host_ram),
                DomainRole::Staging
            );
        }
        let cpu = executor(&rtx3090, BackendName::Cpu);
        assert_eq!(
            DomainRole::of(&rtx3090, cpu, host_ram),
            DomainRole::Allocation
        );
        // GB10's CUDA allocates in host RAM.
        let gb10 = recorded::gb10();
        let cuda = executor(&gb10, BackendName::Cuda);
        assert_eq!(
            DomainRole::of(&gb10, cuda, gb10.host_ram().id()),
            DomainRole::Allocation
        );
        // Device storage over two paths allocates in both stores.
        let strix = recorded::strix_halo_windows();
        let vulkan = executor(&strix, BackendName::Vulkan);
        for store in [device_memory(&strix), strix.host_ram().id()] {
            assert_eq!(
                DomainRole::of(&strix, vulkan, store),
                DomainRole::Allocation
            );
        }
    }

    /// On the M4 Pro, Metal's stable fit is the smaller of its recommended
    /// working set and host RAM's capacity, less host RAM's planning reserve:
    /// the working set's while host RAM is larger, host RAM's fit under a
    /// process limit below it. A working set filled to its end loses the
    /// device, so the reserve stays inside it too.
    #[test]
    fn metal_fit_respects_the_recommended_working_set() {
        let topology = recorded::m4_pro();
        let reserves = reserves();
        let metal = executor(&topology, BackendName::Metal);
        let working_set = grant_room(&topology, metal, Contract::DeviceOnly, 0);
        let unlimited = sampled(&topology, host(MACOS, 16 * GIB));
        let fit = Fit::new(&topology, &unlimited, &reserves, HostRoom::Allocatable);
        let host_fit = fit.store(topology.host_ram().id()).unwrap().stable_bytes;
        assert!(working_set < host_fit);
        for kind in crate::memory::CLAIMED {
            assert_eq!(
                fit.placement(metal, kind).unwrap().stable_bytes,
                working_set
            );
        }
        // The CPU's host RAM is ungranted: host RAM's fit.
        let cpu = executor(&topology, BackendName::Cpu);
        assert_eq!(
            fit.placement(cpu, Contract::DeviceOnly)
                .unwrap()
                .stable_bytes,
            host_fit
        );

        let mut limited = host(MACOS, 16 * GIB);
        limited.limits = vec![ProcessMemoryLimit {
            kind: ProcessLimitKind::DataSegment,
            limit_bytes: 32 * GIB,
            used_bytes: 0,
            reclaimable_bytes: 0,
        }];
        let limited = sampled(&topology, limited);
        let fit = Fit::new(&topology, &limited, &reserves, HostRoom::Allocatable);
        let host_fit = fit.store(topology.host_ram().id()).unwrap().stable_bytes;
        assert!(host_fit < working_set);
        assert_eq!(
            fit.placement(metal, Contract::DeviceOnly)
                .unwrap()
                .stable_bytes,
            host_fit
        );
    }

    /// Metal's live ceiling is the smaller of the working set left (the
    /// sample's grant status) and host RAM's ceiling, each less host RAM's
    /// planning reserve. The working set bounds what Metal may claim, never
    /// host RAM's own ceiling.
    #[test]
    fn metal_working_set_bounds_metals_ceiling_not_host_rams() {
        let topology = recorded::m4_pro();
        let reserves = reserves();
        let metal = executor(&topology, BackendName::Metal);
        let path = metal.paths(seismic::Placing::Fixed(Contract::DeviceOnly))[0];
        let host_ram = topology.host_ram().id();
        let mut sampled = sampled(&topology, host(MACOS, 20 * GIB));
        let with_left = |sampled: &mut Status, left| {
            sampled.sample.grants = vec![(
                path,
                Ok(Reported::new(left, ReportedBy::MetalWorkingSetUnallocated)),
            )];
        };
        // Working set left 4 GiB above host RAM's planning reserve, which
        // stays inside it, is tighter than host RAM's ceiling.
        let reserve = Fit::new(&topology, &sampled, &reserves, HostRoom::Allocatable)
            .store(host_ram)
            .unwrap()
            .thresholds
            .planning_bytes;
        with_left(&mut sampled, 4 * GIB + reserve);
        let fit = Fit::new(&topology, &sampled, &reserves, HostRoom::Allocatable);
        let host_ceiling = fit.store(host_ram).unwrap().ceiling_bytes;
        assert!(4 * GIB < host_ceiling);
        let room = fit.placement(metal, Contract::DeviceOnly).unwrap();
        assert_eq!(room.ceiling_bytes, 4 * GIB);
        // The sample bounds the ceiling, not stable fit.
        assert_eq!(
            room.stable_bytes,
            grant_room(&topology, metal, Contract::DeviceOnly, 0)
        );
        let over = fit
            .demand(
                &[demand(metal, Contract::DeviceOnly, 5 * GIB)],
                &[],
                FitMode::Live,
            )
            .unwrap();
        assert_eq!(
            over.refused,
            [FitRefusal {
                store: host_ram,
                required_bytes: 5 * GIB,
                missing_bytes: GIB,
                constraint: MemoryConstraint::Grant,
            }]
        );
        // Host RAM's ceiling is tighter than the working set left.
        with_left(&mut sampled, 30 * GIB);
        let fit = Fit::new(&topology, &sampled, &reserves, HostRoom::Allocatable);
        let room = fit.placement(metal, Contract::DeviceOnly).unwrap();
        assert_eq!(room.ceiling_bytes, host_ceiling);
        let over = fit
            .demand(
                &[demand(metal, Contract::DeviceOnly, host_ceiling + 1)],
                &[],
                FitMode::Live,
            )
            .unwrap();
        assert_eq!(over.refused[0].constraint, MemoryConstraint::HostRam);
        // A spent working set grants nothing but leaves host RAM's ceiling.
        with_left(&mut sampled, 0);
        let fit = Fit::new(&topology, &sampled, &reserves, HostRoom::Allocatable);
        assert_eq!(
            fit.placement(metal, Contract::DeviceOnly)
                .unwrap()
                .ceiling_bytes,
            0
        );
        assert_eq!(fit.store(host_ram).unwrap().ceiling_bytes, host_ceiling);
    }

    /// An integrated GPU (GB10's CUDA and Vulkan, Strix Halo's and Lunar
    /// Lake's Vulkan) fits the smaller of its grant's room and host RAM's
    /// fit, and a demand beyond it is refused by whichever is smaller.
    #[test]
    fn an_integrated_gpu_is_bounded_by_the_smaller_of_its_grant_and_host_ram() {
        let reserves = reserves();
        for (topology, backend) in [
            (recorded::gb10(), BackendName::Cuda),
            (recorded::gb10(), BackendName::Vulkan),
            (recorded::strix_halo_linux(), BackendName::Vulkan),
            (recorded::lunar_lake(), BackendName::Vulkan),
        ] {
            let unopened = sampled(&topology, host(LINUX, 8 * GIB));
            let fit = Fit::new(&topology, &unopened, &reserves, HostRoom::Allocatable);
            let host_fit = fit.store(topology.host_ram().id()).unwrap().stable_bytes;
            let executor = executor(&topology, backend);
            let grant = grant_room(&topology, executor, Contract::DeviceOnly, 0);
            let room = fit.placement(executor, Contract::DeviceOnly).unwrap();
            let label = format!("{backend:?} on {}", executor.name());
            assert_eq!(room.stable_bytes, grant.min(host_fit), "{label}");
            let over = fit
                .demand(
                    &[demand(executor, Contract::DeviceOnly, room.stable_bytes + 1)],
                    &[],
                    FitMode::Stable,
                )
                .unwrap();
            let bound = if host_fit <= grant {
                MemoryConstraint::HostRam
            } else {
                MemoryConstraint::Grant
            };
            assert_eq!(over.refused[0].constraint, bound, "{label}");
        }
    }

    /// On GB10 every charge is host RAM's: CUDA's and Vulkan's device
    /// storage and direct host work spend host RAM's one fit.
    #[test]
    fn gb10_executors_and_host_work_share_host_ram() {
        let topology = recorded::gb10();
        let reserves = reserves();
        let unopened = status(host(LINUX, 8 * GIB));
        let fit = Fit::new(&topology, &unopened, &reserves, HostRoom::Allocatable);
        let host_ram = topology.host_ram().id();
        let host_fit = fit.store(host_ram).unwrap().stable_bytes;
        let cuda = executor(&topology, BackendName::Cuda);
        let vulkan = executor(&topology, BackendName::Vulkan);
        let (work, half) = (GIB, (host_fit - GIB) / 2);
        let result = fit
            .demand(
                &[
                    demand(cuda, Contract::DeviceOnly, half),
                    demand(vulkan, Contract::DeviceOnly, host_fit - work - half + 1),
                ],
                &[(host_ram, work)],
                FitMode::Stable,
            )
            .unwrap();
        assert_eq!(result.placed, [(host_ram, host_fit)]);
        assert_eq!(
            result.refused,
            [FitRefusal {
                store: host_ram,
                required_bytes: host_fit - work - half + 1,
                missing_bytes: 1,
                constraint: MemoryConstraint::HostRam,
            }]
        );
    }

    /// CUDA's free figure for an integrated device is no bound (§2.5): its
    /// live grant is its total less what it holds, less host RAM's planning
    /// reserve, and host RAM's ceiling bounds it below that.
    #[test]
    fn integrated_cuda_is_bounded_by_host_ram_and_what_it_holds() {
        let topology = recorded::gb10();
        let reserves = reserves();
        let cuda = executor(&topology, BackendName::Cuda);
        let path = cuda.paths(seismic::Placing::Fixed(Contract::DeviceOnly))[0];
        let total = topology.access_path(path).unwrap().grant().unwrap().value;
        let mut opened = sampled(&topology, host(LINUX, 40 * GIB));
        let holding = |opened: &mut Status, held| {
            opened.runtime.grants = vec![(
                path,
                Ok(Reported::new(total - held, ReportedBy::CudaTotalMemory)),
            )];
        };
        holding(&mut opened, 0);
        let fit = Fit::new(&topology, &opened, &reserves, HostRoom::Allocatable);
        let host_ceiling = fit.store(topology.host_ram().id()).unwrap().ceiling_bytes;
        assert_eq!(
            fit.placement(cuda, Contract::DeviceOnly)
                .unwrap()
                .ceiling_bytes,
            host_ceiling
        );
        // Holding all but 8 GiB above host RAM's planning reserve of its
        // total leaves it 8 GiB, below host RAM's ceiling.
        let reserve = fit
            .store(topology.host_ram().id())
            .unwrap()
            .thresholds
            .planning_bytes;
        holding(&mut opened, total - 8 * GIB - reserve);
        let fit = Fit::new(&topology, &opened, &reserves, HostRoom::Allocatable);
        assert!(8 * GIB < host_ceiling);
        assert_eq!(
            fit.placement(cuda, Contract::DeviceOnly)
                .unwrap()
                .ceiling_bytes,
            8 * GIB
        );
    }

    /// Two executors on one store: each of the RTX 3090's CUDA and Vulkan
    /// fits the smaller of its grant's room and its memory's fit, and a
    /// demand split across both cannot exceed that fit once; a charge to
    /// the store bounds both executors' ceilings.
    #[test]
    fn cuda_and_vulkan_on_one_device_share_its_fit() {
        let topology = recorded::rtx3090();
        let reserves = reserves();
        let gpu = device_memory(&topology);
        let capacity = topology.store(gpu).unwrap().capacity().value;
        let mut sampled = sampled(&topology, host(LINUX, 16 * GIB));
        let fit = Fit::new(&topology, &sampled, &reserves, HostRoom::Allocatable);
        let own = fit.store(gpu).unwrap();
        assert_eq!(own.constraint, MemoryConstraint::DeviceLocal);
        assert_eq!(own.stable_bytes, capacity - own.thresholds.planning_bytes);
        let cuda = executor(&topology, BackendName::Cuda);
        let vulkan = executor(&topology, BackendName::Vulkan);
        for executor in [cuda, vulkan] {
            let grant = grant_room(&topology, executor, Contract::DeviceOnly, 0);
            assert_eq!(
                fit.placement(executor, Contract::DeviceOnly)
                    .unwrap()
                    .stable_bytes,
                grant.min(own.stable_bytes)
            );
        }
        let half = own.stable_bytes / 2;
        let split = fit
            .demand(
                &[
                    demand(cuda, Contract::DeviceOnly, half),
                    demand(vulkan, Contract::DeviceOnly, own.stable_bytes - half + 1),
                ],
                &[],
                FitMode::Stable,
            )
            .unwrap();
        assert_eq!(split.placed, [(gpu, own.stable_bytes)]);
        assert_eq!(
            split.refused,
            [FitRefusal {
                store: gpu,
                required_bytes: own.stable_bytes - half + 1,
                missing_bytes: 1,
                constraint: MemoryConstraint::DeviceLocal,
            }]
        );
        // 8 GiB charged by either executor is the store's ledger.
        sampled.runtime.charges = vec![(gpu, 8 * GIB)];
        let fit = Fit::new(&topology, &sampled, &reserves, HostRoom::Allocatable);
        for executor in [cuda, vulkan] {
            assert_eq!(
                fit.placement(executor, Contract::DeviceOnly)
                    .unwrap()
                    .ceiling_bytes,
                capacity - 8 * GIB - own.thresholds.planning_bytes
            );
        }
    }

    /// The A10-8Q slice keeps no host reserve above its claims: before a
    /// load measured its motion its driver's own reserve is all it keeps,
    /// so its stable fit is its capacity, and a charge leaving no headroom
    /// leaves no ceiling.
    #[test]
    fn a_small_dedicated_device_keeps_no_host_reserve_above_its_claims() {
        let topology = recorded::a10_8q();
        let reserves = reserves();
        let slice = device_memory(&topology);
        let capacity = topology.store(slice).unwrap().capacity().value;
        let mut sampled = sampled(&topology, host(LINUX, 16 * GIB));
        let fit = Fit::new(&topology, &sampled, &reserves, HostRoom::Allocatable);
        let own = fit.store(slice).unwrap();
        assert_eq!(own.thresholds.planning_bytes, 0);
        assert_eq!(own.stable_bytes, capacity);
        // CUDA's usable memory is below the slice's fit; Vulkan's heap, the
        // slice's capacity, is above it.
        let cuda = executor(&topology, BackendName::Cuda);
        let usable = grant_room(&topology, cuda, Contract::DeviceOnly, 0);
        assert!(usable < own.stable_bytes);
        assert_eq!(
            fit.placement(cuda, Contract::DeviceOnly)
                .unwrap()
                .stable_bytes,
            usable
        );
        let vulkan = executor(&topology, BackendName::Vulkan);
        assert_eq!(
            fit.placement(vulkan, Contract::DeviceOnly)
                .unwrap()
                .stable_bytes,
            own.stable_bytes
        );
        sampled.runtime.charges = vec![(slice, capacity - 2 * GIB)];
        let charged = Fit::new(&topology, &sampled, &reserves, HostRoom::Allocatable)
            .store(slice)
            .unwrap();
        assert_eq!(charged.ceiling_bytes, 2 * GIB);
        sampled.runtime.charges = vec![(slice, capacity)];
        let at_line = Fit::new(&topology, &sampled, &reserves, HostRoom::Allocatable)
            .store(slice)
            .unwrap();
        assert_eq!(at_line.ceiling_bytes, 0);
    }

    /// A shared path's grant is one budget for every storage kind placed on
    /// it: the RTX 3090's Vulkan places device and upload storage (ReBAR)
    /// on one path, whose budget left bounds both together.
    #[test]
    fn remaining_path_grant_is_shared_by_device_and_upload() {
        let topology = recorded::rtx3090();
        let reserves = reserves();
        let vulkan = executor(&topology, BackendName::Vulkan);
        let path = vulkan.paths(seismic::Placing::Fixed(Contract::DeviceOnly))[0];
        assert_eq!(vulkan.paths(seismic::Placing::Fixed(Contract::Upload)), [path]);
        let mut sampled = sampled(&topology, host(LINUX, 16 * GIB));
        let gpu = device_memory(&topology);
        let reserve = Fit::new(&topology, &sampled, &reserves, HostRoom::Allocatable)
            .store(gpu)
            .unwrap()
            .thresholds
            .planning_bytes;
        // A budget left 10 GiB above the memory's planning reserve.
        sampled.sample.grants = vec![(
            path,
            Ok(Reported::new(10 * GIB + reserve, ReportedBy::VulkanHeapBudget)),
        )];
        let fit = Fit::new(&topology, &sampled, &reserves, HostRoom::Allocatable);
        assert_eq!(fit.store(gpu).unwrap().ceiling_bytes, 10 * GIB);
        let result = fit
            .demand(
                &[
                    demand(vulkan, Contract::DeviceOnly, 7 * GIB),
                    demand(vulkan, Contract::Upload, 4 * GIB),
                ],
                &[],
                FitMode::Live,
            )
            .unwrap();
        assert_eq!(result.placed, [(gpu, 10 * GIB)]);
        assert_eq!(
            result.refused,
            [FitRefusal {
                store: gpu,
                required_bytes: 4 * GIB,
                missing_bytes: GIB,
                constraint: MemoryConstraint::DeviceLocal,
            }]
        );
    }

    /// A placement over two paths: Strix Halo on Windows (#207) places
    /// device storage on its own memory up to that store's fit, then on
    /// host RAM under the heaps' total beyond it. A demand larger than its
    /// own memory continues onto host RAM; beyond both, the refusal names
    /// the path that ran out: its grant, or host RAM when other work has
    /// taken host RAM's fit.
    #[test]
    fn a_placement_over_two_paths_continues_onto_host_ram() {
        let topology = recorded::strix_halo_windows();
        let reserves = reserves();
        let unopened = sampled(&topology, host(WINDOWS, 16 * GIB));
        let fit = Fit::new(&topology, &unopened, &reserves, HostRoom::Allocatable);
        let (radeon, host_ram) = (device_memory(&topology), topology.host_ram().id());
        let own_fit = fit.store(radeon).unwrap().stable_bytes;
        let host_fit = fit.store(host_ram).unwrap().stable_bytes;
        let vulkan = executor(&topology, BackendName::Vulkan);
        // The own-memory path's grant is its store's capacity, whose room is
        // its fit; the host path's room is below host RAM's fit.
        assert_eq!(grant_room(&topology, vulkan, Contract::DeviceOnly, 0), own_fit);
        let beyond = grant_room(&topology, vulkan, Contract::DeviceOnly, 1);
        assert!(beyond < host_fit);
        let room = fit.placement(vulkan, Contract::DeviceOnly).unwrap();
        assert_eq!(room.stable_bytes, own_fit + beyond);

        let spilled = fit
            .demand(
                &[demand(vulkan, Contract::DeviceOnly, own_fit + GIB)],
                &[],
                FitMode::Stable,
            )
            .unwrap();
        assert!(spilled.fits());
        assert_eq!(
            (placed(&spilled, radeon), placed(&spilled, host_ram)),
            (own_fit, GIB)
        );

        let over = fit
            .demand(
                &[demand(vulkan, Contract::DeviceOnly, own_fit + beyond + 1)],
                &[],
                FitMode::Stable,
            )
            .unwrap();
        assert_eq!(
            over.refused,
            [FitRefusal {
                store: host_ram,
                required_bytes: own_fit + beyond + 1,
                missing_bytes: 1,
                constraint: MemoryConstraint::Grant,
            }]
        );

        // Host work leaving 1 GiB of host RAM's fit.
        let crowded = fit
            .demand(
                &[demand(vulkan, Contract::DeviceOnly, own_fit + 2 * GIB)],
                &[(host_ram, host_fit - GIB)],
                FitMode::Stable,
            )
            .unwrap();
        assert_eq!(placed(&crowded, radeon), own_fit);
        assert_eq!(placed(&crowded, host_ram), host_fit);
        assert_eq!(
            crowded.refused,
            [FitRefusal {
                store: host_ram,
                required_bytes: own_fit + 2 * GIB,
                missing_bytes: GIB,
                constraint: MemoryConstraint::HostRam,
            }]
        );
    }

    /// A container limited to a few GiB of a large host keeps its line and
    /// measured margin within what it may use, never a fraction of the
    /// whole host's RAM, which would exceed the limit and leave nothing to
    /// fit.
    #[test]
    fn a_limited_container_keeps_its_line_within_its_limit() {
        let topology = recorded::rtx3090();
        let reserves = holding(GIB);
        let host_ram = topology.host_ram();
        let mut limited = host(LINUX, 3 * GIB);
        limited.limits = vec![ProcessMemoryLimit {
            kind: ProcessLimitKind::CgroupV2 {
                cgroup: "/container".into(),
            },
            limit_bytes: 4 * GIB,
            used_bytes: GIB,
            reclaimable_bytes: 0,
        }];
        let limited = status(limited);
        let reading = Fit::new(&topology, &limited, &reserves, HostRoom::Allocatable).store(host_ram.id()).unwrap();
        assert_eq!(reading.capacity_bytes, 4 * GIB);
        assert_eq!(reading.thresholds, reserves.thresholds(host_ram));
        assert_eq!(reading.stable_bytes, 3 * GIB);
        assert_eq!(
            Fit::host_ram(&topology, &limited.sample.host_ram.clone().unwrap(), &reserves).thresholds,
            reading.thresholds
        );
    }

    /// Process limits are grants on host RAM that every path into it
    /// shares: a cgroup limit bounds host RAM's stable fit by its limit and
    /// its headroom by what the limit leaves, for the CPU and for CUDA's
    /// pinned upload alike, though the cgroup's ancestors are hidden.
    #[test]
    fn process_limits_bound_host_headroom() {
        let topology = recorded::rtx3090();
        let reserves = reserves();
        let host_ram = topology.host_ram().id();
        let mut limited = host(LINUX, 40 * GIB);
        limited.limits = vec![ProcessMemoryLimit {
            kind: ProcessLimitKind::CgroupV2 {
                cgroup: "/worker".into(),
            },
            limit_bytes: 32 * GIB,
            used_bytes: 16 * GIB,
            reclaimable_bytes: 0,
        }];
        let limited_status = status(limited.clone());
        let fit = Fit::new(&topology, &limited_status, &reserves, HostRoom::Allocatable);
        let reading = fit.store(host_ram).unwrap();
        assert_eq!(reading.capacity_bytes, 32 * GIB);
        assert_eq!(reading.stable_bytes, 32 * GIB - reading.thresholds.planning_bytes);
        assert_eq!(reading.headroom_bytes, 16 * GIB);
        assert_eq!(reading.ceiling_bytes, 16 * GIB - reading.thresholds.planning_bytes);
        let cpu = executor(&topology, BackendName::Cpu);
        let cuda = executor(&topology, BackendName::Cuda);
        for (executor, kind) in [(cpu, Contract::DeviceOnly), (cuda, Contract::Upload)] {
            let room = fit.placement(executor, kind).unwrap();
            assert_eq!(
                (room.stable_bytes, room.ceiling_bytes),
                (reading.stable_bytes, reading.ceiling_bytes)
            );
        }
        // A container's own limit bounds it though its ancestors are hidden.
        let mut contained = limited;
        contained.limit_visibility = LimitVisibility::CgroupAncestorsHidden;
        let contained = status(contained);
        assert_eq!(
            Fit::new(&topology, &contained, &reserves, HostRoom::Allocatable)
                .store(host_ram)
                .unwrap(),
            reading
        );
    }

    /// Clean file cache a cgroup can reclaim counts as headroom within its
    /// limit; anonymous usage and the planning reserve still bound claims.
    #[test]
    fn cgroup_file_cache_is_available_but_anonymous_usage_and_reserves_still_bound_claims() {
        let topology = recorded::rtx3090();
        let reserves = reserves();
        let host_ram = topology.host_ram().id();
        let reading = |reclaimable_bytes| {
            let mut sampled = host(LINUX, 40 * GIB);
            sampled.limits = vec![ProcessMemoryLimit {
                kind: ProcessLimitKind::CgroupV2 {
                    cgroup: "/worker".into(),
                },
                limit_bytes: 32 * GIB,
                used_bytes: 30 * GIB,
                reclaimable_bytes,
            }];
            Fit::new(&topology, &status(sampled), &reserves, HostRoom::Allocatable)
                .store(host_ram)
                .unwrap()
        };
        let cached = reading(8 * GIB);
        assert_eq!(cached.headroom_bytes, 10 * GIB);
        assert_eq!(cached.ceiling_bytes, 10 * GIB - cached.thresholds.planning_bytes);
        let anonymous = reading(0);
        assert_eq!(anonymous.headroom_bytes, 2 * GIB);
        assert_eq!(anonymous.ceiling_bytes, 0);
    }

    /// Direct host work and every storage kind of an executor on host RAM
    /// spend host RAM's one fit.
    #[test]
    fn storage_kinds_and_direct_work_spend_one_host_budget() {
        let topology = recorded::m4_pro();
        let reserves = reserves();
        let unopened = status(host(MACOS, 16 * GIB));
        let fit = Fit::new(&topology, &unopened, &reserves, HostRoom::Allocatable);
        let host_ram = topology.host_ram().id();
        let room = fit.store(host_ram).unwrap().stable_bytes;
        let cpu = executor(&topology, BackendName::Cpu);
        let result = fit
            .demand(
                &[
                    demand(cpu, Contract::DeviceOnly, room - 20),
                    demand(cpu, Contract::Upload, 20),
                ],
                &[(host_ram, 7)],
                FitMode::Stable,
            )
            .unwrap();
        assert_eq!(result.placed, [(host_ram, room)]);
        assert_eq!(
            result.refused,
            [FitRefusal {
                store: host_ram,
                required_bytes: 20,
                missing_bytes: 7,
                constraint: MemoryConstraint::HostRam,
            }]
        );
    }

    /// Storage demanded of an executor that places none (Metal on a device
    /// whose memory is not the host's) is refused as unplaced, never fitted
    /// as zero.
    #[test]
    fn demand_on_an_unavailable_executor_is_unplaced() {
        let topology = recorded::m4_pro_without_unified_memory();
        let reserves = reserves();
        let unopened = status(host(MACOS, 16 * GIB));
        let fit = Fit::new(&topology, &unopened, &reserves, HostRoom::Allocatable);
        let metal = executor(&topology, BackendName::Metal);
        let unplaced = MemoryPolicyError::Unplaced {
            executor: metal.id(),
            kind: Contract::DeviceOnly,
        };
        assert_eq!(
            fit.demand(
                &[demand(metal, Contract::DeviceOnly, 1)],
                &[],
                FitMode::Stable
            ),
            Err(unplaced.clone())
        );
        assert_eq!(fit.placement(metal, Contract::DeviceOnly), Err(unplaced));
    }

    /// Fit answers only for nodes of its own topology revision.
    #[test]
    fn a_subject_of_another_revision_is_refused() {
        let (topology, other) = (recorded::m4_pro(), recorded::m4_pro());
        let reserves = reserves();
        let unopened = status(host(MACOS, 16 * GIB));
        let fit = Fit::new(&topology, &unopened, &reserves, HostRoom::Allocatable);
        assert_eq!(
            fit.store(other.host_ram().id()),
            Err(MemoryPolicyError::ForeignIdentifier)
        );
        let cpu = executor(&other, BackendName::Cpu);
        assert_eq!(
            fit.demand(&[demand(cpu, Contract::DeviceOnly, 1)], &[], FitMode::Stable),
            Err(MemoryPolicyError::ForeignIdentifier)
        );
    }
}
