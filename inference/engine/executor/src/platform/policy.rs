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
//!   never subtracted again.
//! - Headroom at or below the planning reserve is the Reclaim band. No
//!   engine claim crosses that line.
//! - A host that reports displacement is also in Reclaim while it is in
//!   distress ([`HostDistress`]): programs' memory has been swapped out or
//!   thrashed through the compressor for a sustained span, whoever caused
//!   it.
//!
//! Process limits are the visible ones. A container's own cgroup limit is
//! visible; limits of cgroups above it may be hidden, and are then bounded
//! only by host headroom, which already reflects the host's memory.

use seismic::{
    AccessPathId, DISPLACEMENT_HISTORY, DeviceTopology, DisplacementWindow, Executor, ExecutorId,
    Contract, HostDisplacement, HostStatus, KernelPressure, LimitError, ObservationError, Status, Store, StoreId,
    StoreKind,
};
use std::{
    collections::BTreeMap,
    fmt,
    time::{Duration, Instant},
};

/// Harm to other programs that host headroom does not show: their memory is
/// leaving RAM faster than the system can absorb. A host in distress is in
/// the Reclaim band whatever its headroom.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostDistress {
    /// Pages were written to swap in windows covering most of the last
    /// [`DISPLACEMENT_SPAN`] ([`sustained`]).
    Swapping,
    /// Windows covering most of the last [`DISPLACEMENT_SPAN`] compressed
    /// and decompressed pages at the rates the kernel itself treats as
    /// thrashing ([`sustained`]).
    Thrashing,
    /// The kernel classifies system memory pressure as critical.
    KernelCritical,
}

impl fmt::Display for HostDistress {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Swapping => "the system is swapping",
            Self::Thrashing => "the system is thrashing compressed memory",
            Self::KernelCritical => "the kernel reports critical memory pressure",
        })
    }
}

/// Host RAM about to run out: the hosting service kills its worker on the
/// first reading that shows it. Where the OS reports its own pressure level
/// (macOS `kern.memorystatus_vm_pressure_level`, the level `memorystatus`
/// acts on before it kills processes), only critical pressure is; elsewhere,
/// obtainable headroom at or below the emergency reserve. Headroom alone is
/// no emergency where the OS reports a level: what can be taken without
/// displacing other programs' data runs low long before memory is
/// exhausted, while the OS can still compress.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostEmergency {
    /// The kernel classifies system memory pressure as critical.
    KernelCritical,
    /// On a host that reports no pressure level, obtainable headroom at or
    /// below the emergency reserve.
    EmergencyReserve,
}

impl fmt::Display for HostEmergency {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::KernelCritical => "the kernel reports critical memory pressure",
            Self::EmergencyReserve => "system memory headroom reached the emergency reserve",
        })
    }
}

/// XNU's compressor thrashing rule (`compute_swapout_target_age`): at least
/// 20 compressions and 20 decompressions per 10 ms, and more than 50
/// decompressions per 10 ms.
const THRASHING_COMPRESSIONS_PER_SECOND: u64 = 2_000;
const THRASHING_DECOMPRESSIONS_PER_SECOND: u64 = 5_000;

/// The span displacement is judged over. Displacement harms other programs'
/// speed, not the system's survival: headroom at the planning and emergency
/// reserves and critical kernel pressure are the out-of-memory backstops,
/// and they act on their first observation. A load wires its weights
/// for as long as its import runs, and on a host whose memory is full of
/// compressible data that wiring compresses other programs' pages and
/// faults some back for most of the import: about 10 s for a 22 GB import on
/// an M4 Max, and 14 to 16 s for the same import on an M1 Ultra whose memory
/// a program filled. The span is
/// about twice the longest; displacement that fills it is the system not
/// absorbing demand, whoever caused it.
pub const DISPLACEMENT_SPAN: Duration = Duration::from_secs(30);

/// How much of [`DISPLACEMENT_SPAN`] displaced windows must cover. Genuine
/// thrashing measured on the same Ultra left a quiet second every 20 to 40 s
/// (`notes.md`), which a rule requiring every window would read as recovery.
pub const DISPLACED_WITHIN_SPAN: Duration = Duration::from_secs(25);

const _: () = assert!(DISPLACEMENT_SPAN.as_secs() <= DISPLACEMENT_HISTORY.as_secs());

fn rate_reaches(pages: u64, window: &DisplacementWindow, per_second: u64) -> bool {
    u128::from(pages) * 1_000 >= u128::from(per_second) * window.duration.as_millis()
}

fn thrashed(window: &DisplacementWindow) -> bool {
    rate_reaches(
        window.compressed_pages,
        window,
        THRASHING_COMPRESSIONS_PER_SECOND,
    ) && rate_reaches(
        window.decompressed_pages,
        window,
        THRASHING_DECOMPRESSIONS_PER_SECOND,
    )
}

/// Whether windows showing displacement by `displaced` cover at least
/// [`DISPLACED_WITHIN_SPAN`] of the last [`DISPLACEMENT_SPAN`]. Windows are
/// contiguous, most recent first; time before the first window counts as
/// not displaced. They are the host sampler's, each about
/// [`seismic::DISPLACEMENT_CADENCE`] long whoever reads the host and however
/// often, the cadence of the traces these rules were judged on.
fn sustained(windows: &[DisplacementWindow], displaced: fn(&DisplacementWindow) -> bool) -> bool {
    let mut span = Duration::ZERO;
    let mut covered = Duration::ZERO;
    for window in windows {
        if span >= DISPLACEMENT_SPAN {
            break;
        }
        // The oldest window counts only up to the span's start.
        let within = window.duration.min(DISPLACEMENT_SPAN - span);
        span += within;
        if displaced(window) {
            covered += within;
        }
    }
    covered >= DISPLACED_WITHIN_SPAN
}

/// The distress `host` reports, most severe first. A host that reports no
/// displacement and no kernel pressure is never in distress. Displacement
/// is distress only when it is sustained over [`DISPLACEMENT_SPAN`]: a load's
/// own import, or a burst of another program's demand, is the system
/// absorbing demand.
fn host_distress(host: &HostStatus) -> Option<HostDistress> {
    if host.kernel_pressure == Some(KernelPressure::Critical) {
        return Some(HostDistress::KernelCritical);
    }
    let windows = &host.displacement.as_ref()?.windows;
    if sustained(windows, thrashed) {
        Some(HostDistress::Thrashing)
    } else if sustained(windows, |window| window.swapped_out_pages > 0) {
        Some(HostDistress::Swapping)
    } else {
        None
    }
}

/// Whether `host` shows displacement now, judged against `baseline`: the
/// kernel's pressure at warning or above, or the latest displacement window,
/// ended since the expert bank last grew, swapping out pages faster than
/// every window since then that was not itself displacement now
/// ([`DisplacementBaseline`]). Compression alone is not: a bank holding
/// borrowed room keeps the OS compressing others as their pages age, and a
/// compressed page costs its program a decompression, a swapped one a read
/// from disk. A host that reports neither never does.
fn displacing(host: &HostStatus, baseline: &DisplacementBaseline) -> bool {
    if matches!(host.kernel_pressure, Some(KernelPressure::Warning | KernelPressure::Critical)) {
        return true;
    }
    host.displacement
        .as_ref()
        .and_then(|displacement| displacement.windows.first())
        .is_some_and(|latest| latest.ended > baseline.since && baseline.exceeded_by(latest))
}

/// Host RAM's fit as `host` reads it: its capacity within the process
/// limits `host` reports, its `room` as headroom and free headroom, and the
/// distress `host` reports.
fn host_ram_fit(store: &Store, host: &HostStatus, reserves: &MemoryReserves, room: HostRoom) -> StoreFit {
    let capacity_bytes = host
        .limits
        .iter()
        .fold(store.capacity().value, |bytes, limit| {
            bytes.min(limit.limit_bytes)
        });
    let serving = match room {
        HostRoom::Load(LoadStage::Serving(serving)) => Some(serving),
        _ => None,
    };
    // While the load loads, its own displacement is accepted.
    let displaced = match (room, host.displaced_bytes) {
        (HostRoom::Allocatable | HostRoom::Load(LoadStage::Loading), _) | (HostRoom::Load(LoadStage::ServingUncounted), None) => 0,
        (HostRoom::Load(LoadStage::Serving(serving)), Some(now)) => now.saturating_sub(serving.displaced_bytes),
        (HostRoom::Load(stage), _) => panic!(
            "invariant violated: the host's displacement reading changed presence while a load served ({stage:?})"
        ),
    };
    let displacing = serving.and_then(|serving| serving.baseline).is_some_and(|baseline| displacing(host, &baseline));
    let thresholds = reserves.host_thresholds(store, host);
    let fit = match serving.and_then(|serving| serving.lent) {
        // Allocatable counts as free the RAM the OS made by displacing other
        // programs' data since, which is free only while that data stays out.
        None => store_fit(
            store,
            thresholds,
            capacity_bytes,
            host.allocatable_bytes().saturating_sub(displaced),
            host_distress(host),
            false,
            MemoryConstraint::HostRam,
        ),
        // The expert bank holds room borrowed beyond the hold line, which
        // the OS gave it by compressing other programs' data: allocatable
        // memory then sits near nothing for as long as the bank holds it, so
        // its level tells another program's demand from the bank's own no
        // more. The load's claims take back what is lent of it, within what
        // the OS can obtain, and displacement now is the hold.
        Some(lent) => store_fit(
            store,
            thresholds,
            capacity_bytes,
            host.obtainable_bytes().min(thresholds.planning_bytes.saturating_add(lent)),
            host_distress(host),
            displacing,
            MemoryConstraint::HostRam,
        ),
    };
    StoreFit {
        borrowable_bytes: serving.filter(|serving| serving.borrows).map(|_| host.obtainable_bytes()),
        displacing,
        ..fit
    }
}

/// `store`'s fit from its capacity within process limits, its headroom and
/// free headroom, and its host's distress: each less its planning reserve,
/// in the Reclaim band at or below the reserve, while the host is in
/// distress, or where `displacing` (a hold on room borrowed beyond the
/// hold line, [`Serving::lent`]), its thresholds `thresholds`
/// ([`MemoryReserves::thresholds`]).
fn store_fit(
    store: &Store,
    thresholds: DomainThresholds,
    capacity_bytes: u64,
    headroom_bytes: u64,
    distress: Option<HostDistress>,
    displacing: bool,
    constraint: MemoryConstraint,
) -> StoreFit {
    StoreFit {
        store: store.id(),
        capacity_bytes,
        stable_bytes: capacity_bytes.saturating_sub(thresholds.planning_bytes),
        headroom_bytes,
        ceiling_bytes: thresholds.ceiling(headroom_bytes),
        thresholds,
        distress,
        borrowable_bytes: None,
        displacing: false,
        constraint,
        band: if headroom_bytes > thresholds.planning_bytes && distress.is_none() && !displacing {
            MemoryBand::Normal
        } else {
            MemoryBand::Reclaim
        },
    }
}

/// The interval a load reads memory at: the engine's probe, and what the
/// rest of the host may take between two readings, before a release can be
/// decided.
pub const READING_INTERVAL: Duration = Duration::from_millis(100);

/// The bytes of fresh memory the page-fill probe takes ([`StoreMotion::of_machine`]).
const PAGE_FILL_PROBE_BYTES: usize = 256 << 20;

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
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "MotionFigures")]
pub struct StoreMotion {
    fluctuation: u64,
    take_rate: f64,
}

/// A take rate is finite by construction ([`StoreMotion::new`]).
impl Eq for StoreMotion {}

/// [`StoreMotion`]'s figures as serialized, checked as [`StoreMotion::new`]
/// checks them.
#[derive(serde::Deserialize)]
struct MotionFigures {
    fluctuation: u64,
    take_rate: f64,
}

impl TryFrom<MotionFigures> for StoreMotion {
    type Error = String;

    fn try_from(figures: MotionFigures) -> Result<Self, String> {
        match figures.take_rate.is_finite() && figures.take_rate > 0.0 {
            true => Ok(Self { fluctuation: figures.fluctuation, take_rate: figures.take_rate }),
            false => Err(format!("a take rate is measured and positive, not {}", figures.take_rate)),
        }
    }
}

impl StoreMotion {
    /// # Panics
    /// Where `take_rate` (bytes a second) is not a positive finite rate:
    /// what bounds a take is measured, never zero.
    pub fn new(fluctuation: u64, take_rate: f64) -> Self {
        assert!(take_rate.is_finite() && take_rate > 0.0, "a take rate is measured and positive, not {take_rate}");
        Self { fluctuation, take_rate }
    }

    /// The machine's own motion before any reading of it: no measured
    /// fluctuation, and the rate the machine fills fresh pages at (a byte
    /// written to every page of [`PAGE_FILL_PROBE_BYTES`] of new memory,
    /// timed, then freed). No program on one thread takes memory from the OS
    /// faster than the machine fills its pages, so a load whose line holds
    /// that rate over a release's latency is not crossed by one program's
    /// ask before the release lands.
    ///
    /// A generic figure of the machine, measured once: where `machine`
    /// names a kernel cache and the topology of the machine it serves, the
    /// rate that cache keeps for the host's identity
    /// (`KernelCache::machine_figures`, keyed by
    /// `PhysicalIdentity::Host`), measured and stored there when it keeps
    /// none; elsewhere, measured once per process.
    pub fn of_machine(machine: Option<(&std::path::Path, &DeviceTopology)>) -> Self {
        Self::new(0, page_rate(machine, "page-fill", |probe| probe.fill))
    }

    /// The margin a planning line holds above its store's own line, for a
    /// load that notices a take within a reading `interval` and whose
    /// first release lands `latency` after it decides one, then releases
    /// `releasing`'s bytes at its rate (none of a store nothing of ours
    /// releases in, a volume's):
    ///
    /// `fluctuation + t·(interval + latency) + max(0, t − r)·B / r`
    ///
    /// for the take rate `t`, release rate `r` and releasable bytes `B`:
    /// the rest of the host takes `t·(interval + latency)` before the first
    /// release lands, and while it takes faster than the load releases, the
    /// deficit grows at `t − r` for the `B / r` releasing runs.
    ///
    /// The rest of the host never takes more of a store than it holds, so
    /// the margin of a store a load releases in is at most the store's
    /// bytes. Its terms are measured rates, of other programs and of the
    /// load's releases: a margin past the store is a measurement that
    /// cannot be, reported as an invariant violation with every input
    /// (a panic in debug and test builds). A release build then holds the
    /// take before the first release lands, never a line past the store
    /// that would read every reading as Reclaim and drive the bank to its
    /// floor.
    pub fn margin(&self, interval: Duration, latency: Duration, releasing: Option<Releasing>) -> u64 {
        let lead = self.lead(interval, latency);
        let Some(releasing) = releasing else {
            return lead;
        };
        let shortfall = (self.shortfall_ratio(releasing.rate) * releasing.bytes as f64) as u64;
        let margin = lead.saturating_add(shortfall);
        if margin <= releasing.store_bytes {
            return margin;
        }
        tracing::error!(
            margin,
            store_bytes = releasing.store_bytes,
            fluctuation = self.fluctuation,
            take_rate = self.take_rate,
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
        lead.min(releasing.store_bytes)
    }

    /// The fluctuation and what the host takes before a first release
    /// lands: `fluctuation + t·(interval + latency)`.
    fn lead(&self, interval: Duration, latency: Duration) -> u64 {
        self.fluctuation.saturating_add((self.take_rate * (interval + latency).as_secs_f64()) as u64)
    }

    /// `g = max(0, t − r) / r`: what the deficit grows by per byte released.
    fn shortfall_ratio(&self, release: ReleaseRate) -> f64 {
        (self.take_rate - release.0).max(0.0) / release.0
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
        Self::new(self.fluctuation.max(other.fluctuation), self.take_rate.max(other.take_rate))
    }
}

/// The rate a load releases memory at, measured and positive: a finished
/// release's bytes over its span from decision to effect, the slowest
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

    /// The rate the machine returns fresh pages to the OS at (the
    /// page-fill probe's memory freed, timed), the fastest any release
    /// returns memory: a generic figure of the machine, kept and measured
    /// as [`StoreMotion::of_machine`]'s page-fill rate is.
    pub fn of_machine(machine: Option<(&std::path::Path, &DeviceTopology)>) -> Self {
        Self::new(page_rate(machine, "page-release", |probe| probe.release))
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

/// The machine's page probe: the rates it fills fresh pages at and returns
/// them to the OS at, in bytes a second.
#[derive(Clone, Copy)]
struct PageProbe {
    fill: f64,
    release: f64,
}

/// The page probe's `figure` of the machine (`page-fill`, `page-release`),
/// kept under `{figure}:{host}` ([`machine_rate`]); the probe runs at most
/// once per process.
fn page_rate(machine: Option<(&std::path::Path, &DeviceTopology)>, figure: &str, rate: fn(PageProbe) -> f64) -> f64 {
    static PROBED: std::sync::OnceLock<PageProbe> = std::sync::OnceLock::new();
    let key = machine.and_then(|(cache, topology)| Some((cache, format!("{figure}:{}", host_identity(topology)?))));
    machine_rate(key, || Ok::<_, std::convert::Infallible>(rate(*PROBED.get_or_init(page_probe)))).unwrap_or_else(|never| match never {})
}

/// The page probe, run: a byte written to every page of
/// [`PAGE_FILL_PROBE_BYTES`] of new memory, timed, then the memory freed,
/// timed.
fn page_probe() -> PageProbe {
    let page = 4096;
    let layout = std::alloc::Layout::from_size_align(PAGE_FILL_PROBE_BYTES, page).expect("the probe's layout");
    // SAFETY: a non-zero layout; every write lies within it, and it is
    // freed with the same layout.
    unsafe {
        let memory = std::alloc::alloc(layout);
        assert!(!memory.is_null(), "the page-fill probe could not take {PAGE_FILL_PROBE_BYTES} bytes");
        let started = Instant::now();
        for offset in (0..PAGE_FILL_PROBE_BYTES).step_by(page) {
            std::ptr::write_volatile(memory.add(offset), 1);
        }
        let filled = started.elapsed().as_secs_f64();
        let started = Instant::now();
        std::alloc::dealloc(memory, layout);
        let released = started.elapsed().as_secs_f64();
        let rate = |seconds: f64| PAGE_FILL_PROBE_BYTES as f64 / seconds.max(f64::MIN_POSITIVE);
        PageProbe { fill: rate(filled), release: rate(released) }
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
/// never a fraction of capacity. A store's planning line is its own line
/// plus a margin; its emergency line is its own line alone:
/// - host RAM's own line is the kernel's free target
///   ([`HostStatus::free_target`]): below it, taking memory makes the kernel
///   reclaim for it, compressing or swapping other programs' data once
///   nothing cheap is left;
/// - a device's own memory has none: its driver keeps its own reserve
///   outside what it grants or reports free;
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
/// above `planning_bytes`; on a host that reports no pressure level, host
/// RAM at or below `emergency_bytes` is an emergency ([`HostEmergency`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DomainThresholds {
    pub planning_bytes: u64,
    pub emergency_bytes: u64,
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
    pub fn host_margin(&self, host_ram: &Store) -> u64 {
        let releasing = Releasing { rate: self.release, bytes: self.releasable.host, store_bytes: host_ram.capacity().value };
        self.host.margin(self.interval, self.latency, Some(releasing))
    }

    /// The thresholds of a store of `kind` whose own line is `line` bytes:
    /// for host RAM, the kernel's free target ([`HostStatus::free_target`],
    /// none where the OS reports none); for a device's own memory, none.
    ///
    /// # Panics
    /// For a volume, whose reserve is its writer's
    /// ([`super::VolumeReserves`]).
    pub fn thresholds(&self, store: &Store, line: u64) -> DomainThresholds {
        let margin = match store.kind() {
            StoreKind::HostRam => self.host_margin(store),
            StoreKind::DeviceMemory(_) => self.device.map_or(0, |device| {
                let releasing =
                    Releasing { rate: self.release, bytes: self.releasable.device, store_bytes: store.capacity().value };
                device.margin(self.interval, self.latency, Some(releasing))
            }),
            StoreKind::Volume(_) => panic!("a volume's reserve is its writer's (`VolumeReserves`), never a load's"),
        };
        DomainThresholds { planning_bytes: line.saturating_add(margin), emergency_bytes: line }
    }

    /// Host RAM's thresholds as `host` reads its line.
    pub fn host_thresholds(&self, host_ram: &Store, host: &HostStatus) -> DomainThresholds {
        self.thresholds(host_ram, host.free_target.unwrap_or(0))
    }

    /// Stable identity of the reserves assessments read, for assessment
    /// cache keys: the rule, not its measured motion, which moves with the
    /// host from reading to reading and which a load reads live.
    pub fn identity(&self) -> String {
        "reserves-v4".to_owned()
    }
}

/// Why fit could not answer: never a verdict that a demand does not fit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryPolicyError {
    /// A required observation is unavailable; missing observations never
    /// authorize allocation.
    Observation(ObservationError),
    /// A limit was asked of the device on a store or path not its own.
    Limit(LimitError),
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
            Self::Limit(error) => error.fmt(f),
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

/// How long memory must stay free (headroom above the planning reserve, no
/// distress) to count as recovered.
pub const RECOVERY_STABLE_TIME: Duration = Duration::from_secs(5);

/// Whether memory has recovered (stayed free): free at every reading
/// for [`RECOVERY_STABLE_TIME`]. What a service closed by memory pressure
/// reopens load admission on, and an elastic expert bank grows on. `T` is
/// the caller's clock.
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

/// The band of one observed store. No engine claim takes headroom to the
/// planning reserve; other programs can, and on a host that reports
/// displacement the engine's own load can put the host in distress.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryBand {
    /// Headroom above the planning reserve and no host distress.
    Normal,
    /// Headroom at or below the planning reserve, or a host in distress:
    /// pause growth and release.
    Reclaim,
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
    /// The host's distress, for host RAM; other stores have none.
    pub distress: Option<HostDistress>,
    /// For host RAM read by a load serving on a host that reports
    /// displacement: what the OS can obtain for it, displacing other
    /// programs' data ([`HostStatus::obtainable`]), the line room borrowed
    /// beyond the hold line is held within ([`FitMode::Borrow`]). None
    /// elsewhere: a host that reports no displacement lends no such room,
    /// as nothing would tell another program's demand from the bank's own.
    pub borrowable_bytes: Option<u64>,
    /// For host RAM read by a serving load: whether the host shows
    /// displacement now against the load's baseline ([`Serving::baseline`]).
    pub displacing: bool,
    pub band: MemoryBand,
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
    /// [`Self::Live`], host RAM's room up to the line the expert bank
    /// borrows within ([`StoreFit::borrowable_bytes`]) where it has one: the
    /// bank's growth into room borrowed beyond the hold line.
    Borrow,
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

/// Which of host RAM's room a fit question reads ([`HostStatus`]). Every
/// question reads one measure, what the OS gives without compressing or
/// swapping another program's data ([`HostStatus::allocatable`]): free
/// pages, file pages it drops and purgeable pages it purges. A load's
/// band, shortfall and claims hold at it, so another program's ask that
/// takes it below the planning reserve is answered by the load's releases,
/// not by the OS displacing that program's data; and a load is admitted
/// and planned on the same figure, so nothing admitted reads Reclaim as it
/// serves. What the OS could obtain by compressing others
/// ([`HostStatus::obtainable`]) is reported, never planned on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostRoom {
    /// A question of no loaded model, and a load's admission: allocatable
    /// memory as headroom and free headroom alike.
    Allocatable,
    /// A load's, from its opening through serving: allocatable memory as
    /// its headroom; its free headroom, which its growth reads, less what
    /// the OS displaced of other programs' data since the load began to
    /// serve ([`LoadStage`]), so it never grows into RAM the OS freed by
    /// compressing them; and while its expert bank holds room borrowed
    /// beyond that line, the bank's hold on it ([`Serving::lent`]).
    Load(LoadStage),
}

impl HostRoom {
    /// A load's room once it serves on `host` at `now`: its growth counts
    /// displacement from what `host` reports displaced now, where the OS
    /// reports it ([`HostStatus::displaced_bytes`]), and judges
    /// displacement now from the windows after `now`.
    pub fn serving(host: &HostStatus, now: Instant) -> Self {
        Self::Load(match host.displaced_bytes {
            Some(displaced_bytes) => LoadStage::Serving(Serving {
                displaced_bytes,
                baseline: Some(DisplacementBaseline::since(now)),
                borrows: false,
                lent: None,
            }),
            None => LoadStage::ServingUncounted,
        })
    }
}

/// Where a load is, as its growth's free headroom reads displacement.
/// Whether the OS reports displacement is the platform's, fixed for the
/// process: a serving load whose reading appears or disappears is an
/// invariant violation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadStage {
    /// Loading: its own displacement is accepted.
    Loading,
    /// Serving where the OS reports displacement ([`Serving`]).
    Serving(Serving),
    /// Serving where the OS reports no displacement: none is counted, and
    /// no room is borrowed beyond the hold line.
    ServingUncounted,
}

/// A load serving where the OS reports displacement.
///
/// Its expert bank may hold slots beyond the hold line (allocatable memory
/// less the planning reserve) up to what the OS can obtain by compressing
/// other programs' data less the reserve ([`FitMode::Borrow`]): borrowed
/// room, given back first. While the bank holds any, allocatable memory sits
/// near nothing (the OS keeps its free pages by compressing others), so the
/// hold on it is displacement happening now ([`displacing`]), not a level:
/// a hold on the level would read Reclaim for as long as the bank holds
/// borrowed room, give it back, take it again once memory stayed free, and
/// so on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Serving {
    /// What the OS reported displaced as the load began to serve, or as its
    /// bank's last growth into borrowed room ended: what is displaced
    /// beyond it is not free. A growth into borrowed room displaces by
    /// design, and once that room is given back what it displaced stays
    /// out until its programs read it again, which no release of the load's
    /// makes free.
    pub displaced_bytes: u64,
    /// What displacement now is judged against; none while the expert bank
    /// grows, whose own backing displaces.
    pub baseline: Option<DisplacementBaseline>,
    /// Whether the load holds an expert bank that borrows room beyond the
    /// hold line (the one store's): only its room reads the line it borrows
    /// within ([`StoreFit::borrowable_bytes`]).
    pub borrows: bool,
    /// While the expert bank holds borrowed room: the bytes of it the load's
    /// claims may still take back. Host RAM's headroom is then what the OS
    /// can obtain, within the planning reserve plus these bytes, and its
    /// band is Reclaim while the host shows displacement now. None while
    /// the bank holds none: the hold line reads allocatable memory.
    pub lent: Option<u64>,
}

/// What displacement now is judged against on a host that reports it: the
/// fastest swap-out rate of the displacement windows ended
/// since the expert bank last grew (`since`), each window folded in once
/// ([`Self::fold`]) and only where it was not itself displacement now. Its
/// first window seeds it: the windows right after a growth show what the
/// host does with the bank at its new size, and a window faster than every
/// one of them is another program's demand. Kernel pressure at warning or
/// above is displacement now whatever the rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DisplacementBaseline {
    since: Instant,
    /// The end of the last window folded in.
    folded: Instant,
    /// Pages swapped out a second. None before a first window.
    swapped: Option<u64>,
}

impl DisplacementBaseline {
    /// A baseline of the windows ending after `since`, none yet.
    pub fn since(since: Instant) -> Self {
        Self { since, folded: since, swapped: None }
    }

    /// Whether `window` swaps out pages faster than every window folded
    /// in; none before a first.
    fn exceeded_by(&self, window: &DisplacementWindow) -> bool {
        self.swapped.is_some_and(|swapped| exceeds(window.swapped_out_pages, window, swapped))
    }

    /// Fold in `displacement`'s windows ended since the last folded, oldest
    /// first: each that is not displacement now raises the rate to its
    /// own.
    pub fn fold(&mut self, displacement: &HostDisplacement) {
        let folded = self.folded;
        for window in displacement.windows.iter().rev().filter(|window| window.ended > folded) {
            self.folded = window.ended;
            if self.exceeded_by(window) {
                continue;
            }
            let rate = per_second(window.swapped_out_pages, window);
            self.swapped = Some(self.swapped.map_or(rate, |swapped| swapped.max(rate)));
        }
    }
}

/// Whether `pages` within `window` exceed `per_second`.
fn exceeds(pages: u64, window: &DisplacementWindow, per_second: u64) -> bool {
    u128::from(pages) * 1_000 > u128::from(per_second) * window.duration.as_millis()
}

/// `pages` within `window` as pages a second, rounded up.
fn per_second(pages: u64, window: &DisplacementWindow) -> u64 {
    let millis = window.duration.as_millis().max(1);
    u64::try_from((u128::from(pages) * 1_000).div_ceil(millis)).unwrap_or(u64::MAX)
}

/// The sole interpretation of stores and grants for engine fit questions.
pub struct Fit<'a> {
    topology: &'a DeviceTopology,
    status: &'a Status,
    reserves: &'a MemoryReserves,
    room: HostRoom,
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
        Self {
            topology,
            status,
            reserves,
            room,
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
    /// grant on it), each less its planning reserve, in the Reclaim band at or below the reserve or while the
    /// host is in distress.
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
                return Ok(host_ram_fit(store, host, self.reserves, self.room));
            }
            // A device's own memory has no line of its own: the driver keeps
            // its reserve outside what it reports free.
            StoreKind::DeviceMemory(_) => (MemoryConstraint::DeviceLocal, self.reserves.thresholds(store, 0)),
            // Nor has a volume (`super::volume`): it keeps the margin its
            // writes need, and none where this process writes nothing.
            StoreKind::Volume(volume) => {
                let margin = self.volumes.and_then(|volumes| volumes.margin(volume)).unwrap_or(0);
                (MemoryConstraint::Volume, DomainThresholds { planning_bytes: margin, emergency_bytes: 0 })
            }
        };
        let headroom_bytes = self
            .status
            .headroom(store)
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
        Ok(store_fit(
            store,
            thresholds,
            store.capacity().value,
            headroom_bytes,
            None,
            false,
            constraint,
        ))
    }

    /// Host RAM's fit from a host sample alone, as [`Fit::store`] gives it
    /// from a whole status: for a reader that opens no executor (the
    /// residency supervisor), which has only `host`.
    pub fn host_ram(
        topology: &DeviceTopology,
        host: &HostStatus,
        reserves: &MemoryReserves,
        room: HostRoom,
    ) -> StoreFit {
        host_ram_fit(topology.host_ram(), host, reserves, room)
    }

    /// Whether host RAM is about to run out as `host` reads it, and why
    /// ([`HostEmergency`]): for the residency supervisor, which has only
    /// `host`.
    pub fn host_emergency(
        topology: &DeviceTopology,
        host: &HostStatus,
        reserves: &MemoryReserves,
    ) -> Option<HostEmergency> {
        match host.kernel_pressure {
            Some(KernelPressure::Critical) => Some(HostEmergency::KernelCritical),
            Some(KernelPressure::Normal | KernelPressure::Warning) => None,
            None => {
                let thresholds = reserves.host_thresholds(topology.host_ram(), host);
                (host.obtainable_bytes() <= thresholds.emergency_bytes)
                    .then_some(HostEmergency::EmergencyReserve)
            }
        }
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
            FitMode::Live | FitMode::Borrow => match self.status.grant_left(path) {
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
            FitMode::Borrow => fit.borrowable_bytes.map_or(fit.ceiling_bytes, |bytes| fit.thresholds.ceiling(bytes)),
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
    /// The host's distress, for host RAM; a device's own memory has none.
    pub distress: Option<HostDistress>,
    /// What the OS held outside headroom, for host RAM on a platform that
    /// counts it.
    pub held: Option<seismic::HostHeld>,
    /// The displacement windows `distress` was judged on, for host RAM on a
    /// platform that counts displacement.
    pub displacement: Option<seismic::HostDisplacement>,
    /// The line room borrowed beyond the hold line is held within, for host
    /// RAM read by a load serving where the OS reports displacement
    /// ([`StoreFit::borrowable_bytes`]).
    pub borrowable_bytes: Option<u64>,
    /// Whether host RAM shows displacement now ([`StoreFit::displacing`]).
    pub displacing: bool,
    pub band: MemoryBand,
}

impl DomainReading {
    /// The room Seismic's bound on this store gives beyond its charge: its
    /// live room, host RAM's up to the line its expert bank borrows within
    /// where it has one ([`FitMode::Borrow`]), and in the store the device's
    /// storage is placed on ([`DomainRole::Allocation`]) never less than
    /// `planned`, what the load's plan holds and has not allocated yet (its
    /// workload's state, its unbound components, its prompt-group class).
    /// That was admitted with the load, so a line that rises past the live
    /// room shrinks what grows beyond the plan (the Reclaim band's
    /// release), never a planned holding's room.
    fn bound_room(&self, planned: u64) -> u64 {
        let live = self
            .borrowable_bytes
            .map_or(self.ceiling_bytes, |bytes| self.ceiling_bytes.max(self.thresholds.ceiling(bytes)));
        match self.role {
            DomainRole::Allocation => live.max(planned),
            DomainRole::Staging => live,
        }
    }
}

impl fmt::Display for DomainReading {
    /// The reading a band was judged on.
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

/// Fresh store readings for the executor's placements and host-side work,
/// host RAM's room `room`. Allocation roles identify stores reached by
/// device storage, not a promise that device storage occupies only one
/// store.
pub fn observe_domains(
    devices: &seismic::Devices,
    device: &seismic::Device,
    reserves: &MemoryReserves,
    room: HostRoom,
) -> Result<Vec<DomainReading>, MemoryPolicyError> {
    let status = devices.status(device.topology(), device.info().id());
    domain_readings(device, &status, reserves, room)
}

/// The readings of every store `device` places storage on, and host RAM,
/// host RAM's room `room`.
fn domain_readings(
    device: &seismic::Device,
    status: &Status,
    reserves: &MemoryReserves,
    room: HostRoom,
) -> Result<Vec<DomainReading>, MemoryPolicyError> {
    let topology = device.topology();
    let fit = Fit::new(topology, status, reserves, room);
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
                distress: reading.distress,
                held: host.and_then(|host| host.held),
                displacement: host.and_then(|host| host.displacement.clone()),
                borrowable_bytes: reading.borrowable_bytes,
                displacing: reading.displacing,
                band: reading.band,
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

/// Bound `device`'s stores and paths from the status sample `status`: what
/// each has charged plus its live room, and in the store its device storage
/// is placed on never less than `planned`, the load's planned holdings not
/// yet allocated. A failed reading holds the device's own ledger on every
/// store it reaches to what it claims: missing observations never authorize
/// allocation. Host RAM's room is `room`, the one the claims checked against
/// these bounds read.
pub(crate) fn refresh_device_ceiling(
    device: &seismic::Device,
    status: &Status,
    reserves: &MemoryReserves,
    room: HostRoom,
    planned: u64,
) -> Result<Vec<DomainReading>, MemoryPolicyError> {
    let result = set_device_ceiling(device, status, reserves, room, planned);
    if result.is_err() {
        for (store, ledger) in device.ledgers().own {
            device
                .bound(seismic::Ledger::Own(store), Some(ledger.claim))
                .expect("a device bounds its own ledgers");
        }
    }
    result
}

/// Bound `device`'s stores and paths from `status`: what each has charged
/// plus its room ([`DomainReading::bound_room`]): the bank's growth into
/// borrowed room recommits into it, and every other claim the heap holds
/// to its live ceiling first. The readings it bound them from.
fn set_device_ceiling(
    device: &seismic::Device,
    status: &Status,
    reserves: &MemoryReserves,
    room: HostRoom,
    planned: u64,
) -> Result<Vec<DomainReading>, MemoryPolicyError> {
    let fit = Fit::new(device.topology(), status, reserves, room);
    let readings = domain_readings(device, status, reserves, room)?;
    for reading in &readings {
        let limit = status.charged(reading.domain).saturating_add(reading.bound_room(planned));
        device
            .bound(seismic::Ledger::Store(reading.domain), Some(limit))
            .map_err(MemoryPolicyError::Limit)?;
    }
    let charges: BTreeMap<_, _> = device.ledgers().paths.into_iter().map(|path| (path.path, path.claim)).collect();
    let paths: std::collections::BTreeSet<_> = crate::memory::CLAIMED
        .iter()
        .flat_map(|kind| device.info().paths(seismic::Placing::Fixed(*kind)))
        .copied()
        .collect();
    for path in paths {
        let left = fit.grant(path, FitMode::Live)?;
        let charged = charges.get(&path).copied().unwrap_or(0);
        device
            .bound(seismic::Ledger::Path(path), Some(charged.saturating_add(left)))
            .map_err(MemoryPolicyError::Limit)?;
    }
    Ok(readings)
}

#[cfg(test)]
mod tests {
    //! Fit over the recorded machines' topologies
    //! (`seismic::fixtures::recorded`), and host distress. Status
    //! values (headroom, charges, what is left of a grant) are each test's
    //! input; every structural value is the machine's.
    use super::*;
    use seismic::fixtures::recorded;
    use seismic::{
        BackendName, HostDisplacement, LimitVisibility, ProcessLimitKind, ProcessMemoryLimit,
        Reported, ReportedBy, RuntimeStatus, Sample,
    };
    use std::time::SystemTime;

    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;

    const LINUX: ReportedBy = ReportedBy::LinuxMemAvailable;
    const MACOS: ReportedBy = ReportedBy::MachMovablePages;

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
    /// reports it, without limits, displacement counters or kernel pressure.
    fn host(by: ReportedBy, headroom: u64) -> HostStatus {
        HostStatus {
            allocatable: Reported::new(headroom, by),
            obtainable: Reported::new(headroom, by),
            limits: Vec::new(),
            limit_visibility: LimitVisibility::Complete,
            displacement: None,
            kernel_pressure: None,
            free_target: None,
            held: None,
            displaced_bytes: None,
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
        let reserve = reserves().thresholds(store, 0).planning_bytes;
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

    fn window(
        millis: u64,
        compressed: u64,
        decompressed: u64,
        swapped_out: u64,
    ) -> DisplacementWindow {
        DisplacementWindow {
            duration: Duration::from_millis(millis),
            compressed_pages: compressed,
            decompressed_pages: decompressed,
            swapped_out_pages: swapped_out,
            swapped_in_pages: 0,
            ended: epoch(),
        }
    }

    /// The instant every test window ends at but the ones a test places.
    fn epoch() -> Instant {
        static EPOCH: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        *EPOCH.get_or_init(Instant::now)
    }

    /// A macOS host whose displacement windows are `windows`, most recent
    /// first.
    fn displaced(windows: Vec<DisplacementWindow>) -> HostStatus {
        HostStatus {
            displacement: Some(HostDisplacement { windows }),
            kernel_pressure: Some(KernelPressure::Normal),
            ..host(MACOS, 32 * GIB)
        }
    }

    /// Enough windows like `window` to span `span`.
    fn lasting(window: DisplacementWindow, span: Duration) -> Vec<DisplacementWindow> {
        let count = span.as_millis().div_ceil(window.duration.as_millis());
        vec![window; usize::try_from(count).unwrap()]
    }

    /// 2,000 compressions/s and 5,000 decompressions/s over 500 ms.
    fn thrashing_window() -> DisplacementWindow {
        window(500, 1_000, 2_500, 0)
    }

    fn quiet_window() -> DisplacementWindow {
        window(500, 0, 2_500, 0)
    }

    /// `displaced` worth of windows like `active`, most recent first, then
    /// quiet ones to fill the span.
    fn within_span(active: DisplacementWindow, displaced: Duration) -> Vec<DisplacementWindow> {
        let mut windows = lasting(active, displaced);
        windows.extend(lasting(quiet_window(), DISPLACEMENT_SPAN - displaced));
        windows
    }

    const HALF_SECOND: Duration = Duration::from_millis(500);

    /// A host that reports no displacement and no kernel pressure is never
    /// in distress, whatever its headroom.
    #[test]
    fn a_host_without_displacement_counters_is_never_in_distress() {
        assert_eq!(host_distress(&host(LINUX, 0)), None);
    }

    /// Compression without decompression, and decompression without
    /// compression, are not thrashing.
    #[test]
    fn compression_alone_is_not_distress() {
        // Other programs' idle data being compressed is the allowed cost of a load.
        let compressing = displaced(lasting(window(500, 1_000_000, 0, 0), DISPLACEMENT_SPAN));
        assert_eq!(host_distress(&compressing), None);
        // Programs faulting compressed pages back without new compression is recovery.
        let recovering = displaced(lasting(window(500, 0, 1_000_000, 0), DISPLACEMENT_SPAN));
        assert_eq!(host_distress(&recovering), None);
    }

    /// Thrashing is both of XNU's rates in windows covering at least
    /// [`DISPLACED_WITHIN_SPAN`] of the last [`DISPLACEMENT_SPAN`].
    #[test]
    fn thrashing_needs_both_rates_over_most_of_the_span() {
        let at_rate = thrashing_window();
        let enough = within_span(at_rate, DISPLACED_WITHIN_SPAN);
        assert_eq!(
            host_distress(&displaced(enough.clone())),
            Some(HostDistress::Thrashing)
        );
        // The quiet part may lie anywhere in the span.
        let mut quiet_first = enough.clone();
        quiet_first.reverse();
        assert_eq!(
            host_distress(&displaced(quiet_first)),
            Some(HostDistress::Thrashing)
        );
        // A burst covering less, however intense, is absorbed demand.
        let short = within_span(at_rate, DISPLACED_WITHIN_SPAN - HALF_SECOND);
        assert_eq!(host_distress(&displaced(short)), None);
        // Windows older than the span do not count, and a history shorter
        // than the span counts its missing time as quiet.
        let mut stale = within_span(at_rate, DISPLACED_WITHIN_SPAN - HALF_SECOND);
        stale.extend(lasting(at_rate, DISPLACEMENT_SPAN));
        assert_eq!(host_distress(&displaced(stale)), None);
        let young = lasting(at_rate, DISPLACED_WITHIN_SPAN - HALF_SECOND);
        assert_eq!(host_distress(&displaced(young)), None);

        for below in [
            window(500, 1_000, 2_499, 0),
            window(500, 999, 2_500, 0),
            // The same page counts over a longer window are a lower rate.
            window(1_000, 1_000, 2_500, 0),
        ] {
            assert_eq!(
                host_distress(&displaced(lasting(below, DISPLACEMENT_SPAN))),
                None
            );
        }
    }

    /// Traces of 1 s samples measured on an M1 Ultra: a load's import into
    /// filled memory is absorbed demand, a program cycling a working set
    /// larger than the memory left to it is thrashing.
    #[test]
    fn displacement_measured_on_the_ultra_is_judged_as_it_was_seen() {
        // Most recent first; T reached both thrashing rates.
        let samples = |trace: &str| {
            trace
                .chars()
                .rev()
                .flat_map(|sample| {
                    let window = if sample == 'T' {
                        thrashing_window()
                    } else {
                        quiet_window()
                    };
                    [window, window]
                })
                .collect::<Vec<_>>()
        };
        // A 22 GB import into memory a program had filled (0.2.8 refused
        // it), then the program's working set settling around the model.
        let import = samples(
            "................TTTTTTTTTTTTTTTT.......TTTTTT.....TTTTTTTTT....TTTTTTT..TTTT..",
        );
        for end in 1..=import.len() {
            assert_eq!(
                host_distress(&displaced(import[import.len() - end..].to_vec())),
                None
            );
        }
        // A program cycling a working set larger than the memory the model
        // left it.
        let thrashing = samples("TTTTTTTTTTTTTTTTTTTT.TTTTTTTTTTTTTTTTTTTTTTTTTT.TTTTTTTTTTTT");
        assert_eq!(
            host_distress(&displaced(thrashing)),
            Some(HostDistress::Thrashing)
        );
    }

    /// Swapping is swap-outs in windows covering most of the span; a burst
    /// is not.
    #[test]
    fn swapping_needs_swap_outs_over_most_of_the_span() {
        let swapping = within_span(window(500, 0, 0, 1), DISPLACED_WITHIN_SPAN);
        assert_eq!(
            host_distress(&displaced(swapping)),
            Some(HostDistress::Swapping)
        );
        let burst = within_span(window(500, 0, 0, 40), DISPLACED_WITHIN_SPAN - HALF_SECOND);
        assert_eq!(host_distress(&displaced(burst)), None);
        // One swap-out in a span of the sampler's windows marks only its own.
        let mut stray = lasting(window(1_000, 0, 0, 0), DISPLACEMENT_SPAN);
        stray[7].swapped_out_pages = 1;
        assert_eq!(host_distress(&displaced(stray)), None);
    }

    /// Critical kernel pressure is distress on its first observation;
    /// warning is not.
    #[test]
    fn critical_kernel_pressure_is_distress_without_any_window() {
        let critical = HostStatus {
            kernel_pressure: Some(KernelPressure::Critical),
            ..displaced(Vec::new())
        };
        assert_eq!(host_distress(&critical), Some(HostDistress::KernelCritical));
        let warning = HostStatus {
            kernel_pressure: Some(KernelPressure::Warning),
            ..displaced(Vec::new())
        };
        assert_eq!(host_distress(&warning), None);
    }

    /// Where the OS reports its pressure level, only critical pressure is an
    /// emergency, whatever the headroom; where it reports none, obtainable
    /// headroom at or below the emergency reserve, the kernel's own line,
    /// is.
    #[test]
    fn host_emergency_is_critical_pressure_where_reported_and_the_emergency_reserve_elsewhere() {
        let topology = recorded::rtx3090();
        let reserves = holding(2 * GIB);
        let emergency_bytes = GIB;
        let emergency = |headroom, kernel_pressure| {
            let host = HostStatus {
                kernel_pressure,
                free_target: Some(emergency_bytes),
                ..host(LINUX, headroom)
            };
            Fit::host_emergency(&topology, &host, &reserves)
        };
        assert_eq!(emergency(emergency_bytes + 1, None), None);
        assert_eq!(
            emergency(emergency_bytes, None),
            Some(HostEmergency::EmergencyReserve)
        );
        assert_eq!(emergency(0, Some(KernelPressure::Normal)), None);
        assert_eq!(emergency(0, Some(KernelPressure::Warning)), None);
        assert_eq!(
            emergency(64 * GIB, Some(KernelPressure::Critical)),
            Some(HostEmergency::KernelCritical)
        );
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

    /// A host in distress puts host RAM in Reclaim whatever its headroom,
    /// which fit still reports as observed; distress is host RAM's, so a
    /// device memory store never carries it.
    #[test]
    fn host_distress_puts_host_domains_in_reclaim_whatever_their_headroom() {
        let reserves = reserves();
        let swapping = status(displaced(lasting(window(500, 0, 0, 1), DISPLACEMENT_SPAN)));
        let m4_pro = recorded::m4_pro();
        let host_ram = Fit::new(&m4_pro, &swapping, &reserves, HostRoom::Allocatable)
            .store(m4_pro.host_ram().id())
            .unwrap();
        assert_eq!(host_ram.band, MemoryBand::Reclaim);
        assert_eq!(host_ram.distress, Some(HostDistress::Swapping));
        assert_eq!(host_ram.headroom_bytes, 32 * GIB);
        assert_eq!(host_ram.ceiling_bytes, 32 * GIB - host_ram.thresholds.planning_bytes);

        let rtx3090 = recorded::rtx3090();
        let fit = Fit::new(&rtx3090, &swapping, &reserves, HostRoom::Allocatable);
        let own = fit.store(device_memory(&rtx3090)).unwrap();
        assert_eq!(own.distress, None);
        assert_eq!(own.band, MemoryBand::Normal);
        assert_eq!(
            fit.store(rtx3090.host_ram().id()).unwrap().band,
            MemoryBand::Reclaim
        );

        // Thrashing for less than the span leaves the band to headroom.
        let burst = within_span(thrashing_window(), DISPLACED_WITHIN_SPAN - HALF_SECOND);
        let absorbed = status(displaced(burst));
        let host_ram = Fit::new(&m4_pro, &absorbed, &reserves, HostRoom::Allocatable)
            .store(m4_pro.host_ram().id())
            .unwrap();
        assert_eq!(host_ram.distress, None);
        assert_eq!(host_ram.band, MemoryBand::Normal);
    }

    /// A line that rises past what the host has left never starves what the
    /// plan holds and has not allocated yet: with another program taking
    /// everything above the planning reserve, host RAM reads Reclaim (the
    /// bank releases first) and has no live room, yet the bound on the store
    /// the device's storage is placed on still admits the plan's unheld
    /// 6 GiB; a staging store's bound keeps no plan.
    #[test]
    fn a_rising_line_keeps_the_plans_unheld_holdings_admissible() {
        let reserves = reserves();
        let m4_pro = recorded::m4_pro();
        let pressed = Fit::new(&m4_pro, &status(host(ReportedBy::MachAllocatablePages, 0)), &reserves, HostRoom::Allocatable)
            .store(m4_pro.host_ram().id())
            .unwrap();
        assert_eq!(pressed.band, MemoryBand::Reclaim);
        assert_eq!(pressed.ceiling_bytes, 0);
        let reading = |role| DomainReading {
            role,
            domain: pressed.store,
            headroom_bytes: pressed.headroom_bytes,
            identity: m4_pro.store_identity(pressed.store).unwrap(),
            constraint: pressed.constraint,
            thresholds: pressed.thresholds,
            ceiling_bytes: pressed.ceiling_bytes,
            distress: pressed.distress,
            held: None,
            displacement: None,
            borrowable_bytes: pressed.borrowable_bytes,
            displacing: pressed.displacing,
            band: pressed.band,
        };
        assert_eq!(reading(DomainRole::Allocation).bound_room(6 * GIB), 6 * GIB);
        assert_eq!(reading(DomainRole::Allocation).bound_room(0), 0, "with nothing planned, no live room");
        assert_eq!(reading(DomainRole::Staging).bound_room(6 * GIB), 0);
    }

    /// A load's room leaves out what the OS displaced from other programs
    /// since it began to serve: 32 GiB allocatable of which 6 GiB exist
    /// because 6 GiB more of another program's data went to swap or the
    /// compressor is 26 GiB, its band's and its growth's alike;
    /// displacement it began beside, or that came back, takes nothing, and
    /// while it loads its own displacement is accepted.
    #[test]
    fn a_loads_growth_reads_no_room_made_by_displacing_another_program() {
        let reserves = reserves();
        let m4_pro = recorded::m4_pro();
        let fit = |displaced_bytes, room| {
            let host = HostStatus { displaced_bytes, ..host(ReportedBy::MachAllocatablePages, 32 * GIB) };
            Fit::new(&m4_pro, &status(host), &reserves, room).store(m4_pro.host_ram().id()).unwrap()
        };
        let free = |displaced_bytes, room| fit(displaced_bytes, room).headroom_bytes;
        let served = HostRoom::Load(LoadStage::Serving(Serving { displaced_bytes: 4 * GIB, baseline: None, borrows: false, lent: None }));
        assert_eq!(free(Some(10 * GIB), served), 26 * GIB);
        assert_eq!(free(Some(4 * GIB), served), 32 * GIB);
        assert_eq!(free(Some(GIB), served), 32 * GIB);
        assert_eq!(free(None, HostRoom::Load(LoadStage::ServingUncounted)), 32 * GIB, "a platform that counts no displacement");
        assert_eq!(free(Some(10 * GIB), HostRoom::Load(LoadStage::Loading)), 32 * GIB, "loading");
        assert_eq!(free(Some(10 * GIB), HostRoom::Allocatable), 32 * GIB);
    }

    /// Whether the OS reports displacement is the platform's: a reading that
    /// disappears while a load serves is an invariant violation, never read
    /// as nothing displaced.
    #[test]
    #[should_panic(expected = "invariant violated")]
    fn a_displacement_reading_that_disappears_while_serving_is_a_violation() {
        let m4_pro = recorded::m4_pro();
        let host = HostStatus { displaced_bytes: None, ..host(ReportedBy::MachAllocatablePages, 32 * GIB) };
        let served = HostRoom::Load(LoadStage::Serving(Serving { displaced_bytes: 4 * GIB, baseline: None, borrows: false, lent: None }));
        Fit::new(&m4_pro, &status(host), &reserves(), served).store(m4_pro.host_ram().id()).unwrap();
    }

    /// The host of a load that compressed other programs' data to fit, as
    /// it begins to serve (an M4 Max, 64 GB, 47.8 GB wired and 10.7 GB
    /// compressed): 5.4 GB can be taken without compressing anyone, 10 GB
    /// is obtainable by compressing more, against a planning reserve of
    /// 6.87 GB. The load reads Reclaim: what the OS could take from other
    /// programs is theirs, and another program's ask lands only in what is
    /// allocatable, so the load releases toward the line rather than the
    /// OS compressing that program. Thrashing is Reclaim as well.
    #[test]
    fn a_load_holds_at_what_the_os_gives_without_compressing_anyone() {
        const GB: u64 = 1_000_000_000;
        // The kernel's free target, and a margin the load measured.
        let target = 4_000 * 16_384;
        let reserves = holding(6_870_000_000 - target);
        let mut m4_max = seismic::fixtures::m4_pro();
        m4_max.host.value = 64 * GIB;
        let m4_max = seismic::fixtures::topology(&m4_max);
        let host_ram = m4_max.host_ram();
        let host = |allocatable: u64| HostStatus {
            allocatable: Reported::new(allocatable, ReportedBy::MachAllocatablePages),
            obtainable: Reported::new(10 * GB, ReportedBy::MachMovablePages),
            displaced_bytes: Some(9 * GB),
            free_target: Some(target),
            ..host(MACOS, 0)
        };
        let planning = reserves.host_thresholds(host_ram, &host(0)).planning_bytes;
        assert_eq!(planning, 6_870_000_000);
        let served = HostRoom::Load(LoadStage::Serving(Serving { displaced_bytes: 9 * GB, baseline: None, borrows: false, lent: None }));
        let reading = |host: HostStatus| host_ram_fit(host_ram, &host, &reserves, served);

        let short = reading(host(5_400_000_000));
        assert_eq!((short.band, short.distress), (MemoryBand::Reclaim, None));
        assert_eq!(short.headroom_bytes, 5_400_000_000);
        assert_eq!(short.ceiling_bytes, 0);

        let above = reading(host(planning + 2 * GB));
        assert_eq!(above.band, MemoryBand::Normal);
        assert_eq!(above.ceiling_bytes, 2 * GB);

        let thrashing = displaced(lasting(thrashing_window(), DISPLACEMENT_SPAN)).displacement;
        let distressed = reading(HostStatus { displacement: thrashing, ..host(planning + 2 * GB) });
        assert_eq!((distressed.band, distressed.distress), (MemoryBand::Reclaim, Some(HostDistress::Thrashing)));
    }

    /// The same Mac's load, its expert bank holding room borrowed beyond
    /// the hold line: allocatable memory sits at 1 GB (the OS keeps its free
    /// pages by compressing others for the bank), which reads Reclaim on the
    /// line alone, and 20 GB is obtainable. Holding borrowed room, the load
    /// reads Normal on that level, its claims taking at most what is lent of
    /// it, and Reclaim once the host shows displacement now: a window
    /// swapping out faster than every one since the bank last grew, or the
    /// kernel's pressure at warning. The bank's growth reads what is
    /// obtainable above the planning reserve.
    #[test]
    fn a_bank_holding_borrowed_room_is_held_on_displacement_now_not_the_level() {
        const GB: u64 = 1_000_000_000;
        let reserves = reserves();
        let mut m4_max = seismic::fixtures::m4_pro();
        m4_max.host.value = 64 * GIB;
        let m4_max = seismic::fixtures::topology(&m4_max);
        let host_ram = m4_max.host_ram();
        let planning = reserves.thresholds(host_ram, 0).planning_bytes;
        let grown = epoch();
        let second = |seconds: u64| grown + Duration::from_secs(seconds);
        let at = |ended: Instant, swapped: u64| DisplacementWindow { ended, ..window(1_000, 0, 0, swapped) };
        // Two windows since the bank grew: 3,000 then 2,000 pages swapped a
        // second.
        let mut baseline = DisplacementBaseline::since(grown);
        baseline.fold(&HostDisplacement { windows: vec![at(second(2), 2_000), at(second(1), 3_000)] });
        let host = |latest: DisplacementWindow, pressure: KernelPressure| HostStatus {
            allocatable: Reported::new(GB, ReportedBy::MachAllocatablePages),
            obtainable: Reported::new(20 * GB, ReportedBy::MachMovablePages),
            displaced_bytes: Some(9 * GB),
            displacement: Some(HostDisplacement { windows: vec![latest, at(second(2), 2_000), at(second(1), 3_000)] }),
            kernel_pressure: Some(pressure),
            ..host(MACOS, 0)
        };
        let room = |lent| HostRoom::Load(LoadStage::Serving(Serving { displaced_bytes: 9 * GB, baseline: Some(baseline), borrows: true, lent }));
        let reading = |host: HostStatus, lent| host_ram_fit(host_ram, &host, &reserves, room(lent));

        let steady = host(at(second(3), 3_000), KernelPressure::Normal);
        assert_eq!(reading(steady.clone(), None).band, MemoryBand::Reclaim, "the level alone reads the bank's own borrowing");
        let held = reading(steady.clone(), Some(4 * GB));
        assert_eq!((held.band, held.displacing), (MemoryBand::Normal, false));
        assert_eq!(held.ceiling_bytes, 4 * GB, "claims take at most what is lent");
        assert_eq!(held.borrowable_bytes, Some(20 * GB));
        assert_eq!(reading(steady.clone(), Some(0)).band, MemoryBand::Reclaim, "nothing left lent");

        let asked = reading(host(at(second(3), 3_001), KernelPressure::Normal), Some(4 * GB));
        assert_eq!((asked.band, asked.displacing), (MemoryBand::Reclaim, true), "faster than every window since");
        let warned = reading(host(at(second(3), 0), KernelPressure::Warning), Some(4 * GB));
        assert_eq!((warned.band, warned.displacing), (MemoryBand::Reclaim, true), "the kernel's warning");
        let growing = HostRoom::Load(LoadStage::Serving(Serving { displaced_bytes: 9 * GB, baseline: None, borrows: true, lent: Some(4 * GB) }));
        let during = host_ram_fit(host_ram, &host(at(second(3), 9_000), KernelPressure::Warning), &reserves, growing);
        assert_eq!((during.band, during.displacing), (MemoryBand::Normal, false), "nothing is judged while the bank grows");

        let sampled = status(steady);
        let borrow = Fit::new(&m4_max, &sampled, &reserves, room(Some(4 * GB)));
        let cpu = executor(&m4_max, BackendName::Cpu).id();
        let demand = |mode| borrow.demand(&[StorageDemand { executor: cpu, kind: Contract::DeviceOnly, bytes: 10 * GB }], &[], mode).unwrap();
        assert!(!demand(FitMode::Live).fits(), "beyond what is lent");
        assert!(20 * GB - planning > 10 * GB);
        assert!(demand(FitMode::Borrow).fits(), "the bank's growth reads what is obtainable");
    }

    /// A window that is itself displacement now never raises the baseline,
    /// and each window is folded in once: the baseline is the fastest of the
    /// steady windows since the bank last grew, seeded by the first.
    #[test]
    fn displacement_now_never_raises_its_own_baseline() {
        let since = epoch();
        let at = |seconds: u64, swapped: u64| DisplacementWindow { ended: since + Duration::from_secs(seconds), ..window(1_000, 0, 0, swapped) };
        let mut baseline = DisplacementBaseline::since(since);
        let windows = |windows: Vec<DisplacementWindow>| HostDisplacement { windows };
        baseline.fold(&windows(vec![at(0, 99_000)]));
        assert_eq!(baseline.swapped, None, "a window ended as the bank grew is not one since");
        baseline.fold(&windows(vec![at(1, 500)]));
        assert_eq!(baseline.swapped, Some(500), "the first window seeds it");
        baseline.fold(&windows(vec![at(2, 80_000), at(1, 500)]));
        assert_eq!(baseline.swapped, Some(500), "displacement now is not folded in");
        baseline.fold(&windows(vec![at(3, 400), at(2, 80_000), at(1, 500)]));
        assert_eq!(baseline.swapped, Some(500));
        baseline.fold(&windows(vec![at(4, 700), at(3, 400), at(2, 80_000)]));
        assert_eq!(baseline.swapped, Some(500), "faster than every one since: displacement now");
    }

    /// Each store's planning line is its own line plus its motion's margin,
    /// its emergency line its own line alone, and neither a fraction of
    /// capacity: host RAM's own line is the kernel's free target (none where
    /// the OS leaves it out of what it reports allocatable, as Linux's
    /// `MemAvailable` does), a device's own memory has none and no margin
    /// until a load measured its motion; a volume's is its writer's. The
    /// margin is the fluctuation plus what the host takes at its take rate
    /// over a reading interval and a release's latency, so a positive take
    /// rate leaves no reserve without one; and a measurement only raises it.
    #[test]
    fn reserves_are_the_stores_own_line_plus_the_motions_margin() {
        let target = 4_000 * 16_384;
        let macos = HostStatus { free_target: Some(target), ..host(MACOS, 0) };
        let linux = HostStatus { free_target: Some(0), ..host(LINUX, 0) };
        // 10 GB a second, over 0.1 s, then over 0.1 s and a 0.4 s release.
        let motion = StoreMotion::new(GIB, 10e9);
        let unplanned = MemoryReserves::new(motion, crate::platform::ReleaseRate::new(1e9));
        let planned = unplanned.with_latency(Duration::from_millis(400));
        let m4_pro = recorded::m4_pro();
        let host_ram = m4_pro.host_ram();
        assert_eq!(unplanned.host_margin(host_ram), GIB + 1_000_000_000);
        assert_eq!(planned.host_margin(host_ram), GIB + 5_000_000_000);
        assert_eq!(
            planned.host_thresholds(host_ram, &macos),
            DomainThresholds { planning_bytes: target + planned.host_margin(host_ram), emergency_bytes: target }
        );
        assert_eq!(
            planned.host_thresholds(host_ram, &linux),
            DomainThresholds { planning_bytes: planned.host_margin(host_ram), emergency_bytes: 0 }
        );
        // A host of another size keeps the same line.
        let lunar_lake = recorded::lunar_lake();
        assert_eq!(planned.host_thresholds(lunar_lake.host_ram(), &macos), planned.host_thresholds(host_ram, &macos));
        // A slower measurement lowers nothing; a faster one raises the margin.
        let slower = planned.measured(Some(StoreMotion::new(0, 1e9)), None);
        assert_eq!(slower.host_margin(host_ram), planned.host_margin(host_ram));
        let faster = planned.measured(Some(StoreMotion::new(0, 20e9)), None);
        assert_eq!(faster.host_margin(host_ram), GIB + 10_000_000_000);
        let rtx3090 = recorded::rtx3090();
        let device = rtx3090.store(device_memory(&rtx3090)).unwrap();
        assert_eq!(planned.thresholds(device, 0), DomainThresholds { planning_bytes: 0, emergency_bytes: 0 });
        let measured = planned.measured(None, Some(StoreMotion::new(0, 1e9)));
        assert_eq!(measured.thresholds(device, 0), DomainThresholds { planning_bytes: 500_000_000, emergency_bytes: 0 });
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

    /// The rest of the host never takes more of a store than it holds: a
    /// margin past it is a measurement that cannot be, an invariant
    /// violation (a panic in test builds), never a line that would hold the
    /// whole store. A 114 MB/s release of a 20 GB bank against a 10 GB/s
    /// take, as a release rate read over its wait for queued steps made it,
    /// is a 1.7 TB margin in a 48 GB store.
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
            .store(&format!("page-fill:{}", host_identity(&topology).unwrap()), &MachineRate { bytes_per_second: 7e9 }, MACHINE_RATE_RECORD_BYTES)
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

    /// Host RAM's stable fit is its capacity less the planning reserve, its
    /// ceiling its headroom less the reserve, and it is in Reclaim at or
    /// below the reserve.
    #[test]
    fn host_ram_band_and_ceiling_follow_the_planning_reserve() {
        let topology = recorded::rtx3090();
        let id = topology.host_ram().id();
        let capacity = topology.host_ram().capacity().value;
        let line = 2 * GIB;
        let reserves = holding(line - GIB);
        let reading = |headroom| {
            let host = HostStatus { free_target: Some(GIB), ..host(LINUX, headroom) };
            Fit::new(&topology, &status(host), &reserves, HostRoom::Allocatable)
                .store(id)
                .unwrap()
        };
        let normal = reading(16 * GIB);
        assert_eq!(normal.thresholds.planning_bytes, line);
        assert_eq!(normal.stable_bytes, capacity - line);
        assert_eq!(normal.ceiling_bytes, 16 * GIB - line);
        assert_eq!(normal.constraint, MemoryConstraint::HostRam);
        assert_eq!(normal.band, MemoryBand::Normal);
        // Exactly at the planning reserve is already Reclaim.
        let at_line = reading(line);
        assert_eq!(at_line.band, MemoryBand::Reclaim);
        assert_eq!(at_line.ceiling_bytes, 0);
        // Below the emergency reserve the engine still reports Reclaim; the
        // kill belongs to the service. Stable fit does not follow headroom.
        let below = reading(normal.thresholds.emergency_bytes / 2);
        assert_eq!(below.band, MemoryBand::Reclaim);
        assert_eq!(below.ceiling_bytes, 0);
        assert_eq!(below.stable_bytes, normal.stable_bytes);
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
    /// planning reserve. The working set
    /// bounds what may be claimed; it never puts host RAM in Reclaim.
    #[test]
    fn metal_working_set_bounds_the_ceiling_not_the_band() {
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
        assert_eq!(fit.store(host_ram).unwrap().band, MemoryBand::Normal);
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
        // A spent working set grants nothing but leaves host RAM Normal.
        with_left(&mut sampled, 0);
        let fit = Fit::new(&topology, &sampled, &reserves, HostRoom::Allocatable);
        assert_eq!(
            fit.placement(metal, Contract::DeviceOnly)
                .unwrap()
                .ceiling_bytes,
            0
        );
        assert_eq!(fit.store(host_ram).unwrap().band, MemoryBand::Normal);
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
    /// is Reclaim.
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
        assert_eq!(charged.band, MemoryBand::Normal);
        sampled.runtime.charges = vec![(slice, capacity)];
        let at_line = Fit::new(&topology, &sampled, &reserves, HostRoom::Allocatable)
            .store(slice)
            .unwrap();
        assert_eq!(at_line.ceiling_bytes, 0);
        assert_eq!(at_line.band, MemoryBand::Reclaim);
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
        limited.free_target = Some(0);
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
        assert_eq!(reading.thresholds, reserves.thresholds(host_ram, 0));
        assert_eq!(reading.stable_bytes, 3 * GIB);
        assert_eq!(
            Fit::host_ram(&topology, &limited.sample.host_ram.clone().unwrap(), &reserves, HostRoom::Allocatable).thresholds,
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
        assert_eq!(cached.band, MemoryBand::Normal);
        let anonymous = reading(0);
        assert_eq!(anonymous.headroom_bytes, 2 * GIB);
        assert_eq!(anonymous.ceiling_bytes, 0);
        assert_eq!(anonymous.band, MemoryBand::Reclaim);
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
