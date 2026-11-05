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
    Contract, HostStatus, KernelPressure, LimitError, ObservationError, Status, Store, StoreId,
    StoreKind,
};
use std::{collections::BTreeMap, fmt, time::Duration};

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

/// Host RAM's fit as `host` reads it: its capacity within the process
/// limits `host` reports, its `room` as headroom, and the distress `host`
/// reports.
fn host_ram_fit(store: &Store, host: &HostStatus, reserves: &MemoryReserves, room: HostRoom) -> StoreFit {
    let capacity_bytes = host
        .limits
        .iter()
        .fold(store.capacity().value, |bytes, limit| {
            bytes.min(limit.limit_bytes)
        });
    store_fit(
        store,
        reserves,
        capacity_bytes,
        match room {
            HostRoom::Allocatable => host.allocatable_bytes(),
            HostRoom::Obtainable => host.obtainable_bytes(),
            HostRoom::Served { displaced } => host
                .allocatable_bytes()
                .saturating_sub(host.displaced_bytes.map_or(0, |now| now.saturating_sub(displaced))),
        },
        host_distress(host),
        MemoryConstraint::HostRam,
    )
}

/// `store`'s fit from its capacity within process limits, its headroom and
/// its host's distress: each less its planning reserve, in the Reclaim band
/// at or below the reserve or while the host is in distress. The reserves are
/// fractions of the capacity within limits, not of the whole store: a
/// container limited to a few GiB of a large host keeps a reserve of those.
fn store_fit(
    store: &Store,
    reserves: &MemoryReserves,
    capacity_bytes: u64,
    headroom_bytes: u64,
    distress: Option<HostDistress>,
    constraint: MemoryConstraint,
) -> StoreFit {
    let thresholds = reserves.for_domain(store.kind(), capacity_bytes);
    StoreFit {
        store: store.id(),
        capacity_bytes,
        stable_bytes: capacity_bytes.saturating_sub(thresholds.planning_bytes),
        headroom_bytes,
        ceiling_bytes: thresholds.ceiling(headroom_bytes),
        thresholds,
        distress,
        constraint,
        band: if headroom_bytes > thresholds.planning_bytes && distress.is_none() {
            MemoryBand::Normal
        } else {
            MemoryBand::Reclaim
        },
    }
}

const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;

/// The threshold policy, keyed by store kind: how much of each memory
/// domain a loaded model may not use, and how much of each volume a
/// download may not use. The host (service or CLI) chooses it and passes it
/// to the engine; the engine never defaults it. `standard` is the only place
/// the values live.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoryReserves {
    host: KindReserves,
    device: KindReserves,
    /// The bytes every volume keeps free.
    volume_bytes: u64,
}

/// The planning and emergency reserves of one kind of memory domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct KindReserves {
    planning: Reserve,
    emergency: Reserve,
}

/// `max(capacity / fraction_divisor, floor_bytes)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct Reserve {
    fraction_divisor: u64,
    floor_bytes: u64,
}

impl Reserve {
    fn of(&self, capacity_bytes: u64) -> u64 {
        (capacity_bytes / self.fraction_divisor).max(self.floor_bytes)
    }
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
    /// What a store with `headroom_bytes` left lets a live claim take: its
    /// headroom above the planning reserve.
    pub fn ceiling(&self, headroom_bytes: u64) -> u64 {
        headroom_bytes.saturating_sub(self.planning_bytes)
    }
}

impl MemoryReserves {
    /// Host RAM, unified memory included: planning reserve
    /// `max(capacity / 10, 2 GiB)` and emergency reserve
    /// `max(capacity / 20, 1 GiB)`. A dedicated device's own memory, whose
    /// capacity is already what its driver grants this process: planning
    /// reserve `max(capacity / 20, 512 MiB)` and emergency reserve
    /// `max(capacity / 40, 256 MiB)`. A volume: planning reserve 2 GiB.
    pub fn standard() -> Self {
        Self {
            host: KindReserves {
                planning: Reserve {
                    fraction_divisor: 10,
                    floor_bytes: 2 * GIB,
                },
                emergency: Reserve {
                    fraction_divisor: 20,
                    floor_bytes: GIB,
                },
            },
            device: KindReserves {
                planning: Reserve {
                    fraction_divisor: 20,
                    floor_bytes: 512 * MIB,
                },
                emergency: Reserve {
                    fraction_divisor: 40,
                    floor_bytes: 256 * MIB,
                },
            },
            volume_bytes: 2 * GIB,
        }
    }

    /// The thresholds of a store of `kind` with `capacity_bytes`. A volume
    /// has no emergency reserve: no worker grows on one, so none is killed
    /// for its space.
    pub fn for_domain(&self, kind: &StoreKind, capacity_bytes: u64) -> DomainThresholds {
        let reserves = match kind {
            StoreKind::HostRam => self.host,
            StoreKind::DeviceMemory(_) => self.device,
            StoreKind::Volume(_) => return self.volume_thresholds(),
        };
        DomainThresholds {
            planning_bytes: reserves.planning.of(capacity_bytes),
            emergency_bytes: reserves.emergency.of(capacity_bytes),
        }
    }

    /// Every volume's thresholds: it keeps `volume_bytes` free, which neither
    /// a download nor any other demand on it may take.
    pub fn volume_thresholds(&self) -> DomainThresholds {
        DomainThresholds {
            planning_bytes: self.volume_bytes,
            emergency_bytes: 0,
        }
    }

    /// Stable identity of the reserves assessments read, the host and device
    /// memory reserves, for assessment cache keys.
    pub fn identity(&self) -> String {
        let reserve =
            |reserve: Reserve| format!("{}:{}", reserve.fraction_divisor, reserve.floor_bytes);
        format!(
            "reserves-v2:{}:{}:{}:{}",
            reserve(self.host.planning),
            reserve(self.host.emergency),
            reserve(self.device.planning),
            reserve(self.device.emergency)
        )
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

/// Which of host RAM's room a fit question reads ([`HostStatus`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostRoom {
    /// What can be taken without the OS displacing another program's data
    /// ([`HostStatus::allocatable`]): what a load that grows and gives back
    /// while others run, its Reclaim band and its claims, read.
    Allocatable,
    /// What the OS can make available, displacing other programs' data if
    /// it must ([`HostStatus::obtainable`]): what an explicit load's
    /// admission reads.
    Obtainable,
    /// What a serving load may take without displacing any more of other
    /// programs' data than the OS held displaced as it began to serve
    /// (`displaced` bytes, [`HostStatus::displaced_bytes`]): `Allocatable`
    /// less the displacement since, which made free RAM that is free only
    /// while another program's data stays out.
    Served { displaced: u64 },
}

/// The sole interpretation of stores and grants for engine fit questions.
pub struct Fit<'a> {
    topology: &'a DeviceTopology,
    status: &'a Status,
    reserves: &'a MemoryReserves,
    room: HostRoom,
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
        }
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
        let constraint = match store.kind() {
            StoreKind::HostRam => {
                let host = self
                    .status
                    .sample
                    .host_ram
                    .as_ref()
                    .map_err(|error| MemoryPolicyError::Observation(error.clone()))?;
                return Ok(host_ram_fit(store, host, self.reserves, self.room));
            }
            StoreKind::DeviceMemory(_) => MemoryConstraint::DeviceLocal,
            StoreKind::Volume(_) => MemoryConstraint::Volume,
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
            self.reserves,
            store.capacity().value,
            headroom_bytes,
            None,
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
                let store = topology.host_ram();
                let thresholds = reserves.for_domain(store.kind(), store.capacity().value);
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
    /// The host's distress, for host RAM; a device's own memory has none.
    pub distress: Option<HostDistress>,
    /// What the OS held outside headroom, for host RAM on a platform that
    /// counts it.
    pub held: Option<seismic::HostHeld>,
    /// The displacement windows `distress` was judged on, for host RAM on a
    /// platform that counts displacement.
    pub displacement: Option<seismic::HostDisplacement>,
    pub band: MemoryBand,
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

/// Fresh store readings for the executor's placements and host-side work.
/// Allocation roles identify stores reached by device storage, not a
/// promise that device storage occupies only one store.
pub fn observe_domains(
    devices: &seismic::Devices,
    device: &seismic::Device,
    reserves: &MemoryReserves,
) -> Result<Vec<DomainReading>, MemoryPolicyError> {
    let status = devices.status(device.topology(), device.info().id());
    domain_readings(device, &status, reserves, HostRoom::Allocatable)
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
/// each has charged plus its live room. A failed reading holds the device's
/// own ledger on every store it reaches to what it claims: missing
/// observations never authorize allocation. Host RAM's room is `room`, the
/// one the claims checked against these bounds read.
pub(crate) fn refresh_device_ceiling(
    device: &seismic::Device,
    status: &Status,
    reserves: &MemoryReserves,
    room: HostRoom,
) -> Result<Vec<DomainReading>, MemoryPolicyError> {
    let result = set_device_ceiling(device, status, reserves, room);
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
/// plus its live room. The readings it bound them from.
fn set_device_ceiling(
    device: &seismic::Device,
    status: &Status,
    reserves: &MemoryReserves,
    room: HostRoom,
) -> Result<Vec<DomainReading>, MemoryPolicyError> {
    let fit = Fit::new(device.topology(), status, reserves, room);
    let readings = domain_readings(device, status, reserves, room)?;
    for reading in &readings {
        let limit = status
            .charged(reading.domain)
            .saturating_add(reading.ceiling_bytes);
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
            held: None,
            displaced_bytes: None,
        }
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
        let reserve = MemoryReserves::standard()
            .for_domain(store.kind(), store.capacity().value)
            .planning_bytes;
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
        }
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
    /// headroom at or below the emergency reserve is.
    #[test]
    fn host_emergency_is_critical_pressure_where_reported_and_the_emergency_reserve_elsewhere() {
        let topology = recorded::rtx3090();
        let reserves = MemoryReserves::standard();
        let store = topology.host_ram();
        let emergency_bytes = reserves
            .for_domain(store.kind(), store.capacity().value)
            .emergency_bytes;
        let emergency = |headroom, kernel_pressure| {
            let host = HostStatus {
                kernel_pressure,
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
        let reserves = MemoryReserves::standard();
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
        let reserves = MemoryReserves::standard();
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

    /// A serving load's room leaves out what the OS displaced from other
    /// programs since it began serving: 32 GiB free of which 6 GiB exist
    /// because 6 GiB more of another program's data went to swap or the
    /// compressor is 26 GiB; displacement it began beside, or that came back,
    /// takes nothing.
    #[test]
    fn a_serving_load_reads_no_room_made_by_displacing_another_program() {
        let reserves = MemoryReserves::standard();
        let m4_pro = recorded::m4_pro();
        let headroom = |displaced_bytes, room| {
            let host = HostStatus { displaced_bytes, ..host(ReportedBy::MachAllocatablePages, 32 * GIB) };
            Fit::new(&m4_pro, &status(host), &reserves, room).store(m4_pro.host_ram().id()).unwrap().headroom_bytes
        };
        let served = HostRoom::Served { displaced: 4 * GIB };
        assert_eq!(headroom(Some(10 * GIB), served), 26 * GIB);
        assert_eq!(headroom(Some(4 * GIB), served), 32 * GIB);
        assert_eq!(headroom(Some(GIB), served), 32 * GIB);
        assert_eq!(headroom(None, served), 32 * GIB, "a platform that counts no displacement");
        assert_eq!(headroom(Some(10 * GIB), HostRoom::Allocatable), 32 * GIB);
    }

    /// Each reserve is the larger of its fraction of the store's capacity
    /// and its floor: host RAM a tenth (2 GiB floor) to plan and a
    /// twentieth (1 GiB) for emergency, device memory a twentieth (512 MiB)
    /// and a fortieth (256 MiB).
    #[test]
    fn standard_reserves_use_the_larger_of_fraction_and_floor() {
        let reserves = MemoryReserves::standard();
        let thresholds = |topology: &DeviceTopology, store: StoreId| {
            let store = topology.store(store).unwrap();
            reserves.for_domain(store.kind(), store.capacity().value)
        };
        // Lunar Lake's 14.8 GB: both floors.
        let lunar_lake = recorded::lunar_lake();
        assert_eq!(
            thresholds(&lunar_lake, lunar_lake.host_ram().id()),
            DomainThresholds {
                planning_bytes: 2 * GIB,
                emergency_bytes: GIB,
            }
        );
        // The M4 Pro's 48 GiB: both fractions.
        let m4_pro = recorded::m4_pro();
        let capacity = m4_pro.host_ram().capacity().value;
        assert_eq!(
            thresholds(&m4_pro, m4_pro.host_ram().id()),
            DomainThresholds {
                planning_bytes: capacity / 10,
                emergency_bytes: capacity / 20,
            }
        );
        // The A10-8Q slice's 7.6 GB: the device floors.
        let a10 = recorded::a10_8q();
        assert_eq!(
            thresholds(&a10, device_memory(&a10)),
            DomainThresholds {
                planning_bytes: 512 * MIB,
                emergency_bytes: 256 * MIB,
            }
        );
        // The RTX 3090's 25.3 GB: the device fractions.
        let rtx3090 = recorded::rtx3090();
        let capacity = rtx3090
            .store(device_memory(&rtx3090))
            .unwrap()
            .capacity()
            .value;
        assert_eq!(
            thresholds(&rtx3090, device_memory(&rtx3090)),
            DomainThresholds {
                planning_bytes: capacity / 20,
                emergency_bytes: capacity / 40,
            }
        );
        // Every volume keeps 2 GiB free, whatever its size, and has no
        // emergency reserve.
        let volume = StoreKind::Volume(seismic::VolumeId {
            device: 1,
            uuid: None,
        });
        for capacity in [64 * GIB, 4096 * GIB] {
            assert_eq!(
                reserves.for_domain(&volume, capacity),
                DomainThresholds {
                    planning_bytes: 2 * GIB,
                    emergency_bytes: 0,
                }
            );
        }
    }

    /// Host RAM's stable fit is its capacity less the planning reserve, its
    /// ceiling its headroom less the reserve, and it is in Reclaim at or
    /// below the reserve.
    #[test]
    fn host_ram_band_and_ceiling_follow_the_planning_reserve() {
        let topology = recorded::rtx3090();
        let id = topology.host_ram().id();
        let capacity = topology.host_ram().capacity().value;
        let reserves = MemoryReserves::standard();
        let reading = |headroom| {
            Fit::new(&topology, &status(host(LINUX, headroom)), &reserves, HostRoom::Allocatable)
                .store(id)
                .unwrap()
        };
        let normal = reading(16 * GIB);
        assert_eq!(normal.thresholds.planning_bytes, capacity / 10);
        assert_eq!(normal.stable_bytes, capacity - capacity / 10);
        assert_eq!(normal.ceiling_bytes, 16 * GIB - capacity / 10);
        assert_eq!(normal.constraint, MemoryConstraint::HostRam);
        assert_eq!(normal.band, MemoryBand::Normal);
        // Exactly at the planning reserve is already Reclaim.
        let at_line = reading(capacity / 10);
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
        let reserves = MemoryReserves::standard();
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
        let reserves = MemoryReserves::standard();
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
        let reserves = MemoryReserves::standard();
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
        let reserves = MemoryReserves::standard();
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
        let reserves = MemoryReserves::standard();
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
        let reserves = MemoryReserves::standard();
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
        let reserves = MemoryReserves::standard();
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

    /// The A10-8Q slice keeps the 512 MiB device floor above its claims, not
    /// the 2 GiB host floor: its stable fit is its capacity less 512 MiB,
    /// and a charge leaving headroom at the reserve is Reclaim.
    #[test]
    fn a_small_dedicated_device_keeps_512_mib_above_its_claims() {
        let topology = recorded::a10_8q();
        let reserves = MemoryReserves::standard();
        let slice = device_memory(&topology);
        let capacity = topology.store(slice).unwrap().capacity().value;
        let mut sampled = sampled(&topology, host(LINUX, 16 * GIB));
        let fit = Fit::new(&topology, &sampled, &reserves, HostRoom::Allocatable);
        let own = fit.store(slice).unwrap();
        assert_eq!(own.thresholds.planning_bytes, 512 * MIB);
        assert_eq!(own.stable_bytes, capacity - 512 * MIB);
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
        assert_eq!(charged.ceiling_bytes, 2 * GIB - 512 * MIB);
        assert_eq!(charged.band, MemoryBand::Normal);
        sampled.runtime.charges = vec![(slice, capacity - 512 * MIB)];
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
        let reserves = MemoryReserves::standard();
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
        let reserves = MemoryReserves::standard();
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

    /// Process limits are grants on host RAM that every path into it
    /// shares: a cgroup limit bounds host RAM's stable fit by its limit and
    /// A container limited to a few GiB of a large host keeps reserves of
    /// what it may use: a fraction of the whole host's RAM would exceed the
    /// limit and leave nothing to fit.
    #[test]
    fn host_reserves_are_fractions_of_the_capacity_within_limits() {
        let topology = recorded::rtx3090();
        let reserves = MemoryReserves::standard();
        let host_ram = topology.host_ram();
        assert!(host_ram.capacity().value / 10 > 4 * GIB);
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
        assert_eq!(reading.thresholds, reserves.for_domain(host_ram.kind(), 4 * GIB));
        assert_eq!(reading.stable_bytes, 2 * GIB);
        assert_eq!(
            Fit::host_ram(&topology, &limited.sample.host_ram.clone().unwrap(), &reserves, HostRoom::Allocatable).thresholds,
            reading.thresholds
        );
    }

    /// its headroom by what the limit leaves, for the CPU and for CUDA's
    /// pinned upload alike, though the cgroup's ancestors are hidden.
    #[test]
    fn process_limits_bound_host_headroom() {
        let topology = recorded::rtx3090();
        let reserves = MemoryReserves::standard();
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
        let reserves = MemoryReserves::standard();
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
        let reserves = MemoryReserves::standard();
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
        let reserves = MemoryReserves::standard();
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
        let reserves = MemoryReserves::standard();
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
