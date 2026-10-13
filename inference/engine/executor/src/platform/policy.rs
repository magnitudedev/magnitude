//! The engine's memory policy over Seismic facts and observations.
//!
//! Seismic establishes pools, capacities and scoped observations and enforces
//! the limits it is given; it holds no reserve policy. The host (service or
//! CLI) chooses the threshold policy, [`MemoryReserves`], and this module
//! applies it to every memory domain a load uses:
//!
//! - Stable fit compares a workload with the domain's capacity, bounded by
//!   applicable process limits (and, on Metal, the recommended working set),
//!   less the domain's planning reserve. It uses no current free memory.
//! - Every live claim must leave the domain's observed headroom above the
//!   planning reserve: the claimable ceiling is `headroom − planning`. Bytes
//!   Seismic already charged are absent from observed headroom and are never
//!   subtracted again.
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
    DISPLACEMENT_HISTORY, DeviceCatalog, DeviceInfo, DeviceMeasurements, DeviceMemory,
    DeviceMemoryInfo, DeviceSelector, DeviceTopology, DisplacementWindow, HostMemoryStatus,
    KernelPressure, MemoryPoolId, MemoryPoolKind, ObservationError,
};
use std::fmt;
use std::time::Duration;

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
/// not displaced.
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
pub fn host_distress(host: &HostMemoryStatus) -> Option<HostDistress> {
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

/// The constraint that bounds a domain's claimable ceiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryConstraint {
    /// Host RAM and its applicable process limits.
    HostRam,
    /// A device-local pool.
    DeviceLocal,
    /// The device memory its driver lets this process use, as discovery
    /// measured it: the engine's charge never exceeds stable fit capacity.
    DeviceCapacity,
    /// Metal's recommended working set for the device.
    DeviceWorkingSet,
}

impl fmt::Display for MemoryConstraint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::HostRam => "host RAM",
            Self::DeviceLocal => "device memory",
            Self::DeviceCapacity => "the device memory this process may use",
            Self::DeviceWorkingSet => "the device's recommended working set",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryPolicyError {
    /// Seismic reports the device's backing as not normalized.
    UnsupportedBacking { device: String, reason: String },
    /// A required observation is unavailable; missing observations never
    /// authorize allocation.
    Observation(ObservationError),
    /// The device's observation does not belong to its established backing.
    MismatchedObservation { device: String },
}

impl fmt::Display for MemoryPolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedBacking { device, reason } => {
                write!(
                    formatter,
                    "device {device} has no qualified memory backing: {reason}"
                )
            }
            Self::Observation(error) => write!(formatter, "{error}"),
            Self::MismatchedObservation { device } => write!(
                formatter,
                "device {device} reported an observation for a different memory backing"
            ),
        }
    }
}

impl std::error::Error for MemoryPolicyError {}

fn established(device: &DeviceInfo) -> Result<&DeviceMemoryInfo, MemoryPolicyError> {
    match &device.memory {
        DeviceMemory::Established(memory) => Ok(memory),
        DeviceMemory::Unsupported { reason } => Err(MemoryPolicyError::UnsupportedBacking {
            device: device.selector.to_string(),
            reason: reason.clone(),
        }),
    }
}

const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;

/// The threshold policy: how much of each memory domain a loaded model may
/// not use. The host (service or CLI) chooses it and passes it to the engine;
/// the engine never defaults it. `standard` is the only place the values live.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoryReserves {
    host: KindReserves,
    device: KindReserves,
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
/// above `planning_bytes`; the hosting service kills the worker at or below
/// `emergency_bytes`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DomainThresholds {
    pub planning_bytes: u64,
    pub emergency_bytes: u64,
}

impl MemoryReserves {
    /// Host RAM, unified memory included: planning reserve
    /// `max(capacity / 10, 2 GiB)` and emergency reserve
    /// `max(capacity / 20, 1 GiB)`. A dedicated device's own memory, whose
    /// capacity is already what its driver grants this process: planning
    /// reserve `max(capacity / 20, 512 MiB)` and emergency reserve
    /// `max(capacity / 40, 256 MiB)`.
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
        }
    }

    /// The thresholds of a domain of `kind` with `capacity_bytes`.
    pub fn for_domain(&self, kind: MemoryPoolKind, capacity_bytes: u64) -> DomainThresholds {
        let reserves = match kind {
            MemoryPoolKind::HostRam => self.host,
            MemoryPoolKind::DeviceLocal => self.device,
        };
        DomainThresholds {
            planning_bytes: reserves.planning.of(capacity_bytes),
            emergency_bytes: reserves.emergency.of(capacity_bytes),
        }
    }

    /// Stable identity of the policy for assessment cache keys.
    pub fn identity(&self) -> String {
        let reserve = |reserve: Reserve| format!("{}:{}", reserve.fraction_divisor, reserve.floor_bytes);
        format!(
            "reserves-v2:{}:{}:{}:{}",
            reserve(self.host.planning),
            reserve(self.host.emergency),
            reserve(self.device.planning),
            reserve(self.device.emergency)
        )
    }
}

/// Stable fit capacity of one memory domain a load touches. A workload fits
/// the domain when `required ≤ capacity − reserve`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FitCapacity {
    pub domain: MemoryPoolId,
    pub kind: MemoryPoolKind,
    /// Pool capacity bounded by process limits and, on Metal, the recommended
    /// working set.
    pub capacity_bytes: u64,
    /// The domain's planning reserve.
    pub reserve_bytes: u64,
}

impl FitCapacity {
    /// The most a stable workload may use of this domain.
    pub fn fit_bytes(&self) -> u64 {
        self.capacity_bytes.saturating_sub(self.reserve_bytes)
    }
}

/// What a domain is used for by one load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DomainRole {
    /// The device's allocation domain: weights, state, workspace.
    Allocation,
    /// Host RAM used by a dedicated device's staged uploads.
    Staging,
}

/// Stable fit capacities of every domain a load on `device` touches: its
/// allocation domain first, then host RAM for staging when the device is
/// dedicated. Uses no live availability.
pub fn fit_capacities(
    topology: &DeviceTopology,
    device: &DeviceInfo,
    host: &HostMemoryStatus,
    reserves: &MemoryReserves,
) -> Result<Vec<(DomainRole, FitCapacity)>, MemoryPolicyError> {
    let memory = established(device)?;
    let host_pool = topology.host_pool();
    let host_capacity = host
        .limits
        .iter()
        .map(|limit| limit.limit_bytes)
        .fold(host_pool.capacity_bytes, u64::min);
    let host_fit = FitCapacity {
        domain: host_pool.id,
        kind: MemoryPoolKind::HostRam,
        capacity_bytes: host_capacity,
        reserve_bytes: reserves
            .for_domain(MemoryPoolKind::HostRam, host_pool.capacity_bytes)
            .planning_bytes,
    };
    if memory.allocates_host_memory() {
        let capacity_bytes = device
            .recommended_working_set_bytes()
            .map_or(host_fit.capacity_bytes, |working_set| {
                host_fit.capacity_bytes.min(working_set)
            });
        return Ok(vec![(
            DomainRole::Allocation,
            FitCapacity {
                capacity_bytes,
                ..host_fit
            },
        )]);
    }
    let pool = topology
        .pool(memory.allocation_pool)
        .expect("a device's pools belong to its topology");
    Ok(vec![
        (
            DomainRole::Allocation,
            FitCapacity {
                domain: pool.id,
                kind: pool.kind,
                capacity_bytes: pool.capacity_bytes,
                reserve_bytes: reserves.for_domain(pool.kind, pool.capacity_bytes).planning_bytes,
            },
        ),
        (DomainRole::Staging, host_fit),
    ])
}

/// The band of one observed domain. No engine claim takes headroom to the
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

/// One fresh observation of a domain a loaded device uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DomainReading {
    pub role: DomainRole,
    pub domain: MemoryPoolId,
    /// Observed available bytes, bounded by every applicable process limit.
    /// Existing Seismic charges are already absent from it.
    pub headroom_bytes: u64,
    /// What bounds `ceiling_bytes`.
    pub constraint: MemoryConstraint,
    pub thresholds: DomainThresholds,
    /// Additional bytes the engine may still claim: headroom above the
    /// planning reserve and, on Metal, the room left in the working set
    /// above the planning reserve.
    pub ceiling_bytes: u64,
    /// The host's distress, for a host RAM domain; a dedicated device's own
    /// memory has none.
    pub distress: Option<HostDistress>,
    pub band: MemoryBand,
}

impl DomainReading {
    fn new(
        role: DomainRole,
        domain: MemoryPoolId,
        headroom_bytes: u64,
        constraint: MemoryConstraint,
        thresholds: DomainThresholds,
        distress: Option<HostDistress>,
    ) -> Self {
        Self {
            role,
            domain,
            headroom_bytes,
            constraint,
            thresholds,
            ceiling_bytes: headroom_bytes.saturating_sub(thresholds.planning_bytes),
            distress,
            band: if headroom_bytes > thresholds.planning_bytes && distress.is_none() {
                MemoryBand::Normal
            } else {
                MemoryBand::Reclaim
            },
        }
    }

    /// Add memory the engine already holds that a claim can take without
    /// the device holding more (its blocks' free space); the band stays the
    /// domain's.
    fn with_unallocated(mut self, bytes: u64) -> Self {
        self.ceiling_bytes += bytes;
        self
    }

    /// Bound the claimable ceiling by a further device constraint with
    /// `remaining_bytes` left, keeping the domain's planning reserve inside
    /// it too; the band stays the domain's.
    fn bounded(mut self, remaining_bytes: u64, constraint: MemoryConstraint) -> Self {
        let bytes = remaining_bytes.saturating_sub(self.thresholds.planning_bytes);
        if bytes < self.ceiling_bytes {
            self.ceiling_bytes = bytes;
            self.constraint = constraint;
        }
        self
    }
}

/// The most restrictive band among `readings`: a load is Normal only while
/// every domain it uses is.
pub fn band_of(readings: &[DomainReading]) -> MemoryBand {
    if readings
        .iter()
        .all(|reading| reading.band == MemoryBand::Normal)
    {
        MemoryBand::Normal
    } else {
        MemoryBand::Reclaim
    }
}

/// A dedicated device's own memory domain, the device memory the engine
/// holds in it (Seismic's `held`: its charges and what its backend holds
/// beyond them) and the part of that its allocations can still take (the
/// free space of the backend's blocks).
#[derive(Clone, Copy, Debug)]
struct DedicatedDomain {
    id: MemoryPoolId,
    capacity_bytes: u64,
    held_bytes: u64,
    unallocated_bytes: u64,
}

/// The pure core of [`observe_domains`]: readings from one host sample and
/// one device sample. `dedicated` is the device's own pool, or `None` when
/// the device allocates host RAM.
fn domain_readings(
    selector: DeviceSelector,
    host: &HostMemoryStatus,
    host_pool: MemoryPoolId,
    host_capacity_bytes: u64,
    dedicated: Option<DedicatedDomain>,
    measurements: &DeviceMeasurements,
    reserves: &MemoryReserves,
) -> Result<Vec<DomainReading>, MemoryPolicyError> {
    let host_headroom = host
        .limits
        .iter()
        .map(|limit| limit.remaining_bytes())
        .fold(host.headroom.bytes, u64::min);
    let host_thresholds = reserves.for_domain(MemoryPoolKind::HostRam, host_capacity_bytes);
    let host_distress = host_distress(host);
    let mismatched = || MemoryPolicyError::MismatchedObservation {
        device: selector.to_string(),
    };
    let Some(dedicated) = dedicated else {
        let reading = DomainReading::new(
            DomainRole::Allocation,
            host_pool,
            host_headroom,
            MemoryConstraint::HostRam,
            host_thresholds,
            host_distress,
        );
        return Ok(vec![match *measurements {
            // Metal's working set bounds claims on the same host domain, less
            // the planning reserve as stable fit plans it: the driver's own
            // submission memory lies outside the charge, and a working set
            // filled to its end fails a submission (device lost) where a
            // refused claim would have released. Host headroom decides the
            // band.
            DeviceMeasurements::Metal {
                recommended_working_set_bytes,
                current_allocated_bytes,
            } => reading.bounded(
                recommended_working_set_bytes.saturating_sub(current_allocated_bytes),
                MemoryConstraint::DeviceWorkingSet,
            ),
            // An integrated (host-backed) device's driver-reported free bytes
            // exclude reclaimable page cache, so they are not an allocation
            // bound; host headroom is.
            DeviceMeasurements::Host
            | DeviceMeasurements::Cuda { .. }
            | DeviceMeasurements::Vulkan { .. } => reading,
        }]);
    };
    let headroom = match *measurements {
        DeviceMeasurements::Cuda { free_bytes, .. } => free_bytes,
        // A dedicated Vulkan device may use its memory budget less what this
        // process already uses of the heap.
        DeviceMeasurements::Vulkan {
            heap_budget_bytes,
            heap_usage_bytes,
        } => heap_budget_bytes.saturating_sub(heap_usage_bytes),
        DeviceMeasurements::Host | DeviceMeasurements::Metal { .. } => return Err(mismatched()),
    };
    // Claims are bounded by the stable capacity less the device memory the
    // engine holds, as well as by headroom, so what the engine holds never
    // exceeds stable fit capacity whatever a driver's live figure later
    // reports (a Vulkan budget may grow).
    Ok(vec![
        DomainReading::new(
            DomainRole::Allocation,
            dedicated.id,
            headroom,
            MemoryConstraint::DeviceLocal,
            reserves.for_domain(MemoryPoolKind::DeviceLocal, dedicated.capacity_bytes),
            None,
        )
        .bounded(
            dedicated.capacity_bytes.saturating_sub(dedicated.held_bytes),
            MemoryConstraint::DeviceCapacity,
        )
        .with_unallocated(dedicated.unallocated_bytes),
        DomainReading::new(
            DomainRole::Staging,
            host_pool,
            host_headroom,
            MemoryConstraint::HostRam,
            host_thresholds,
            host_distress,
        ),
    ])
}

/// Fresh readings of every domain a load on `device` uses: the allocation
/// domain first, then host RAM for staging when the device is dedicated.
/// Any failure is the Blind case: missing observations never authorize
/// allocation.
pub fn observe_domains(
    catalog: &DeviceCatalog,
    device: &seismic::Device,
    reserves: &MemoryReserves,
) -> Result<Vec<DomainReading>, MemoryPolicyError> {
    let info = device.info();
    let memory = established(info)?;
    let topology = catalog.topology();
    let host = catalog
        .host_memory_status()
        .map_err(MemoryPolicyError::Observation)?;
    let status = device
        .memory_status()
        .map_err(MemoryPolicyError::Observation)?;
    let host_pool = topology.host_pool();
    let usage = device.memory_usage();
    let dedicated = (!memory.allocates_host_memory()).then(|| {
        let pool = topology
            .pool(memory.allocation_pool)
            .expect("a device's pools belong to its topology");
        DedicatedDomain {
            id: pool.id,
            capacity_bytes: pool.capacity_bytes,
            held_bytes: usage.held,
            unallocated_bytes: usage.unallocated,
        }
    });
    domain_readings(
        info.selector,
        &host,
        host_pool.id,
        host_pool.capacity_bytes,
        dedicated,
        &status.measurements,
        reserves,
    )
}

/// Refresh Seismic's enforced ceiling on `device` from fresh readings: the
/// current charge plus the allocation domain's ceiling. A failed
/// observation revokes every unspent grant.
pub fn refresh_device_ceiling(
    catalog: &DeviceCatalog,
    device: &seismic::Device,
    reserves: &MemoryReserves,
) -> Result<Vec<DomainReading>, MemoryPolicyError> {
    let charged = device.memory_usage().charged;
    match observe_domains(catalog, device, reserves) {
        Ok(readings) => {
            let allocation = readings
                .iter()
                .find(|reading| reading.role == DomainRole::Allocation)
                .expect("observe_domains reports the allocation domain");
            device.set_memory_limit(Some(charged.saturating_add(allocation.ceiling_bytes)));
            Ok(readings)
        }
        Err(error) => {
            device.set_memory_limit(Some(charged));
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::{
        HeadroomBasis, HeadroomEstimate, HostDisplacement, HostMeasurements, LimitVisibility,
        ProcessLimitKind, ProcessMemoryLimit,
    };
    use std::time::SystemTime;

    /// A real pool identity to label synthetic readings with; the values
    /// under test never come from the device.
    fn pool() -> MemoryPoolId {
        DeviceCatalog::discover().unwrap().topology().host_pool().id
    }

    fn host(headroom: u64, limits: Vec<ProcessMemoryLimit>) -> HostMemoryStatus {
        HostMemoryStatus {
            sampled_at: SystemTime::now(),
            measurements: HostMeasurements::Linux {
                mem_free_bytes: headroom,
                mem_available_bytes: headroom,
            },
            headroom: HeadroomEstimate {
                bytes: headroom,
                basis: HeadroomBasis::LinuxMemAvailable,
            },
            limits,
            limit_visibility: LimitVisibility::Complete,
            displacement: None,
            kernel_pressure: None,
        }
    }

    fn window(
        millis: u64,
        compressed: u64,
        decompressed: u64,
        swapped_out: u64,
    ) -> DisplacementWindow {
        DisplacementWindow {
            duration: std::time::Duration::from_millis(millis),
            compressed_pages: compressed,
            decompressed_pages: decompressed,
            swapped_out_pages: swapped_out,
            swapped_in_pages: 0,
        }
    }

    /// A host whose displacement windows are `windows`, most recent first.
    fn displaced(windows: Vec<DisplacementWindow>) -> HostMemoryStatus {
        HostMemoryStatus {
            displacement: Some(HostDisplacement { windows }),
            kernel_pressure: Some(KernelPressure::Normal),
            ..host(32 * GIB, Vec::new())
        }
    }

    /// Enough 500 ms windows like `window` to span `span`.
    fn lasting(window: DisplacementWindow, span: std::time::Duration) -> Vec<DisplacementWindow> {
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

    /// `displaced` windows of `active`, most recent first, then quiet ones
    /// to fill the span.
    fn within_span(
        active: DisplacementWindow,
        displaced: std::time::Duration,
    ) -> Vec<DisplacementWindow> {
        let mut windows = lasting(active, displaced);
        windows.extend(lasting(quiet_window(), DISPLACEMENT_SPAN - displaced));
        windows
    }

    const HALF_SECOND: std::time::Duration = std::time::Duration::from_millis(500);

    #[test]
    fn a_host_without_displacement_counters_is_never_in_distress() {
        assert_eq!(host_distress(&host(0, Vec::new())), None);
    }

    #[test]
    fn compression_alone_is_not_distress() {
        // Other programs' idle data being compressed is the allowed cost of a load.
        let compressing = displaced(lasting(window(500, 1_000_000, 0, 0), DISPLACEMENT_SPAN));
        assert_eq!(host_distress(&compressing), None);
        // Programs faulting compressed pages back without new compression is recovery.
        let recovering = displaced(lasting(window(500, 0, 1_000_000, 0), DISPLACEMENT_SPAN));
        assert_eq!(host_distress(&recovering), None);
    }

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

    #[test]
    fn displacement_measured_on_the_ultra_is_judged_as_it_was_seen() {
        // 1 s samples measured on an M1 Ultra, most
        // recent first; T reached both thrashing rates.
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

    #[test]
    fn swapping_needs_swap_outs_over_most_of_the_span() {
        let swapping = within_span(window(500, 0, 0, 1), DISPLACED_WITHIN_SPAN);
        assert_eq!(
            host_distress(&displaced(swapping)),
            Some(HostDistress::Swapping)
        );
        let burst = within_span(window(500, 0, 0, 40), DISPLACED_WITHIN_SPAN - HALF_SECOND);
        assert_eq!(host_distress(&displaced(burst)), None);
    }

    #[test]
    fn critical_kernel_pressure_is_distress_without_any_window() {
        let critical = HostMemoryStatus {
            kernel_pressure: Some(KernelPressure::Critical),
            ..displaced(Vec::new())
        };
        assert_eq!(host_distress(&critical), Some(HostDistress::KernelCritical));
        let warning = HostMemoryStatus {
            kernel_pressure: Some(KernelPressure::Warning),
            ..displaced(Vec::new())
        };
        assert_eq!(host_distress(&warning), None);
    }

    #[test]
    fn host_distress_puts_host_domains_in_reclaim_whatever_their_headroom() {
        let swapping = displaced(lasting(window(500, 0, 0, 1), DISPLACEMENT_SPAN));
        let host_backed = readings(&swapping, 32 * GIB, None, DeviceMeasurements::Host).unwrap();
        assert_eq!(host_backed[0].band, MemoryBand::Reclaim);
        assert_eq!(host_backed[0].distress, Some(HostDistress::Swapping));
        // Headroom and ceiling are still reported as observed.
        assert_eq!(host_backed[0].headroom_bytes, 32 * GIB);

        // A dedicated device's own memory has no distress; its staging domain does.
        let dedicated = readings(
            &swapping,
            32 * GIB,
            Some(24 * GIB),
            DeviceMeasurements::Cuda {
                free_bytes: 20 * GIB,
                total_bytes: 24 * GIB,
            },
        )
        .unwrap();
        assert_eq!(dedicated[0].band, MemoryBand::Normal);
        assert_eq!(dedicated[0].distress, None);
        assert_eq!(dedicated[1].band, MemoryBand::Reclaim);
        assert_eq!(band_of(&dedicated), MemoryBand::Reclaim);

        // Thrashing for less than the span leaves the band to headroom.
        let burst = within_span(thrashing_window(), DISPLACED_WITHIN_SPAN - HALF_SECOND);
        let absorbed =
            readings(&displaced(burst), 32 * GIB, None, DeviceMeasurements::Host).unwrap();
        assert_eq!(absorbed[0].band, MemoryBand::Normal);
    }

    fn readings(
        host: &HostMemoryStatus,
        host_capacity: u64,
        dedicated_capacity: Option<u64>,
        measurements: DeviceMeasurements,
    ) -> Result<Vec<DomainReading>, MemoryPolicyError> {
        let id = pool();
        domain_readings(
            DeviceSelector::HostCpu,
            host,
            id,
            host_capacity,
            dedicated_capacity.map(|capacity_bytes| DedicatedDomain {
                id,
                capacity_bytes,
                held_bytes: 0,
                unallocated_bytes: 0,
            }),
            &measurements,
            &MemoryReserves::standard(),
        )
    }

    #[test]
    fn standard_reserves_use_the_larger_of_fraction_and_floor() {
        let reserves = MemoryReserves::standard();
        let host = |capacity| reserves.for_domain(MemoryPoolKind::HostRam, capacity);
        assert_eq!(
            host(16 * GIB),
            DomainThresholds {
                planning_bytes: 2 * GIB,
                emergency_bytes: GIB,
            }
        );
        assert_eq!(
            host(64 * GIB),
            DomainThresholds {
                planning_bytes: 64 * GIB / 10,
                emergency_bytes: 64 * GIB / 20,
            }
        );
        let device = |capacity| reserves.for_domain(MemoryPoolKind::DeviceLocal, capacity);
        assert_eq!(
            device(6 * GIB),
            DomainThresholds {
                planning_bytes: 512 * MIB,
                emergency_bytes: 256 * MIB,
            }
        );
        assert_eq!(
            device(22 * GIB),
            DomainThresholds {
                planning_bytes: 22 * GIB / 20,
                emergency_bytes: 22 * GIB / 40,
            }
        );
    }

    /// The A10-8Q vGPU slice: WDDM's per-process budget less the CUDA
    /// context leaves 6,298 MiB of its 8 GiB. The device keeps 512 MiB above any claim, not the
    /// 2 GiB host floor that left the engine under 4 GiB.
    #[test]
    fn a_small_dedicated_device_keeps_512_mib_above_its_claims() {
        let capacity = 6_298 * MIB;
        let reading = readings(
            &host(64 * GIB, vec![]),
            128 * GIB,
            Some(capacity),
            DeviceMeasurements::Cuda {
                free_bytes: capacity - 2 * GIB,
                total_bytes: 8 * GIB,
            },
        )
        .unwrap();
        assert_eq!(reading[0].thresholds.planning_bytes, 512 * MIB);
        assert_eq!(reading[0].ceiling_bytes, capacity - 2 * GIB - 512 * MIB);
        assert_eq!(reading[0].band, MemoryBand::Normal);
        let at_line = readings(
            &host(64 * GIB, vec![]),
            128 * GIB,
            Some(capacity),
            DeviceMeasurements::Cuda {
                free_bytes: 512 * MIB,
                total_bytes: 8 * GIB,
            },
        )
        .unwrap();
        assert_eq!(at_line[0].band, MemoryBand::Reclaim);
        assert_eq!(at_line[0].ceiling_bytes, 0);
    }

    /// A live figure above the capacity (a grown Vulkan budget) does not
    /// enlarge what the engine may claim: its charge stays within stable fit
    /// capacity, which is what assessment promised.
    #[test]
    fn dedicated_claims_stay_within_the_granted_capacity() {
        let (capacity, charged) = (6 * GIB, 4 * GIB);
        let reserve = 512 * MIB;
        let reading = |free_bytes| {
            domain_readings(
                DeviceSelector::HostCpu,
                &host(64 * GIB, vec![]),
                pool(),
                128 * GIB,
                Some(DedicatedDomain {
                    id: pool(),
                    capacity_bytes: capacity,
                    held_bytes: charged,
                    unallocated_bytes: 0,
                }),
                &DeviceMeasurements::Cuda {
                    free_bytes,
                    total_bytes: 8 * GIB,
                },
                &MemoryReserves::standard(),
            )
            .unwrap()[0]
        };
        // Headroom 3 GiB above the charge, but the capacity leaves 2 GiB.
        let freed = reading(3 * GIB);
        assert_eq!(freed.ceiling_bytes, capacity - charged - reserve);
        assert_eq!(freed.constraint, MemoryConstraint::DeviceCapacity);
        assert_eq!(freed.band, MemoryBand::Normal);
        // Untracked driver growth: headroom is tighter than the capacity.
        let grown = reading(GIB);
        assert_eq!(grown.ceiling_bytes, GIB - reserve);
        assert_eq!(grown.constraint, MemoryConstraint::DeviceLocal);
        // A charge at stable fit capacity grants nothing more, but only
        // headroom decides the band.
        let full = domain_readings(
            DeviceSelector::HostCpu,
            &host(64 * GIB, vec![]),
            pool(),
            128 * GIB,
            Some(DedicatedDomain {
                id: pool(),
                capacity_bytes: capacity,
                held_bytes: capacity - reserve,
                unallocated_bytes: 0,
            }),
            &DeviceMeasurements::Cuda {
                free_bytes: 2 * GIB,
                total_bytes: 8 * GIB,
            },
            &MemoryReserves::standard(),
        )
        .unwrap()[0];
        assert_eq!(full.ceiling_bytes, 0);
        assert_eq!(full.band, MemoryBand::Normal);
        // Free space the device's blocks already hold is claimable on top:
        // an allocation into it adds nothing to what the device holds.
        let blocks = domain_readings(
            DeviceSelector::HostCpu,
            &host(64 * GIB, vec![]),
            pool(),
            128 * GIB,
            Some(DedicatedDomain {
                id: pool(),
                capacity_bytes: capacity,
                held_bytes: charged,
                unallocated_bytes: GIB / 2,
            }),
            &DeviceMeasurements::Cuda {
                free_bytes: 3 * GIB,
                total_bytes: 8 * GIB,
            },
            &MemoryReserves::standard(),
        )
        .unwrap()[0];
        assert_eq!(blocks.ceiling_bytes, capacity - charged - reserve + GIB / 2);
    }

    #[test]
    fn host_ram_band_and_ceiling_follow_the_planning_reserve() {
        let normal = readings(
            &host(5 * GIB, vec![]),
            16 * GIB,
            None,
            DeviceMeasurements::Host,
        )
        .unwrap();
        assert_eq!(normal.len(), 1);
        assert_eq!(normal[0].role, DomainRole::Allocation);
        assert_eq!(normal[0].band, MemoryBand::Normal);
        assert_eq!(normal[0].ceiling_bytes, 3 * GIB);
        assert_eq!(normal[0].constraint, MemoryConstraint::HostRam);
        // Exactly at the planning reserve is already Reclaim.
        let at_line = readings(
            &host(2 * GIB, vec![]),
            16 * GIB,
            None,
            DeviceMeasurements::Host,
        )
        .unwrap();
        assert_eq!(at_line[0].band, MemoryBand::Reclaim);
        assert_eq!(at_line[0].ceiling_bytes, 0);
        // Below the emergency reserve the engine still reports Reclaim; the
        // kill belongs to the service.
        let below = readings(
            &host(GIB / 2, vec![]),
            16 * GIB,
            None,
            DeviceMeasurements::Host,
        )
        .unwrap();
        assert_eq!(below[0].band, MemoryBand::Reclaim);
        assert_eq!(below[0].ceiling_bytes, 0);
    }

    #[test]
    fn process_limits_bound_host_headroom() {
        let limited = host(
            12 * GIB,
            vec![ProcessMemoryLimit {
                kind: ProcessLimitKind::CgroupV2 {
                    cgroup: "/worker".into(),
                },
                limit_bytes: 8 * GIB,
                used_bytes: 4 * GIB,
            }],
        );
        let reading = readings(&limited, 16 * GIB, None, DeviceMeasurements::Host).unwrap();
        assert_eq!(reading[0].headroom_bytes, 4 * GIB);
        assert_eq!(reading[0].ceiling_bytes, 2 * GIB);
        // A container's own limit bounds it though its ancestors are hidden.
        let mut contained = limited.clone();
        contained.limit_visibility = LimitVisibility::CgroupAncestorsHidden;
        assert_eq!(
            readings(&contained, 16 * GIB, None, DeviceMeasurements::Host).unwrap(),
            reading
        );
    }

    #[test]
    fn metal_working_set_bounds_the_ceiling_not_the_band() {
        let metal = |allocated| DeviceMeasurements::Metal {
            recommended_working_set_bytes: 12 * GIB,
            current_allocated_bytes: allocated,
        };
        // Working-set room above the 2 GiB reserve (1 GiB) is tighter than
        // the host ceiling (4 GiB).
        let bounded = readings(&host(6 * GIB, vec![]), 16 * GIB, None, metal(9 * GIB)).unwrap();
        assert_eq!(bounded[0].band, MemoryBand::Normal);
        assert_eq!(bounded[0].ceiling_bytes, GIB);
        assert_eq!(bounded[0].constraint, MemoryConstraint::DeviceWorkingSet);
        // Host headroom (ceiling 4 GiB) is tighter than working-set room.
        let host_bound = readings(&host(6 * GIB, vec![]), 16 * GIB, None, metal(GIB)).unwrap();
        assert_eq!(host_bound[0].ceiling_bytes, 4 * GIB);
        assert_eq!(host_bound[0].constraint, MemoryConstraint::HostRam);
        // A working set filled to its reserve grants nothing but is not
        // Reclaim by itself: the claim is refused and the owner releases.
        let full = readings(&host(6 * GIB, vec![]), 16 * GIB, None, metal(10 * GIB)).unwrap();
        assert_eq!(full[0].band, MemoryBand::Normal);
        assert_eq!(full[0].ceiling_bytes, 0);
        assert_eq!(full[0].constraint, MemoryConstraint::DeviceWorkingSet);
        let reclaim = readings(&host(GIB, vec![]), 16 * GIB, None, metal(GIB)).unwrap();
        assert_eq!(reclaim[0].band, MemoryBand::Reclaim);
        assert_eq!(reclaim[0].ceiling_bytes, 0);
    }

    /// A 24 GiB Mac whose movable memory stays large while the engine's
    /// wired state grows: claims stop at stable fit (working set less the
    /// planning reserve), never at the working set's end, where the driver's
    /// submission memory no longer fits and the device is lost.
    #[test]
    fn metal_claims_stop_at_stable_fit_below_the_working_set() {
        let working_set = 16 * GIB;
        let reserve = MemoryReserves::standard()
            .for_domain(MemoryPoolKind::HostRam, 24 * GIB)
            .planning_bytes;
        let stable_fit = working_set - reserve;
        for allocated in [
            4 * GIB,
            10 * GIB,
            stable_fit - 1,
            stable_fit,
            working_set - GIB / 2,
            working_set,
        ] {
            let reading = readings(
                &host(14 * GIB, vec![]),
                24 * GIB,
                None,
                DeviceMeasurements::Metal {
                    recommended_working_set_bytes: working_set,
                    current_allocated_bytes: allocated,
                },
            )
            .unwrap();
            assert_eq!(reading[0].band, MemoryBand::Normal);
            assert_eq!(
                reading[0].ceiling_bytes,
                stable_fit.saturating_sub(allocated),
                "allocated {allocated}"
            );
            assert_eq!(reading[0].constraint, MemoryConstraint::DeviceWorkingSet);
        }
    }

    #[test]
    fn integrated_cuda_is_bounded_by_host_ram_not_driver_free_bytes() {
        let reading = readings(
            &host(10 * GIB, vec![]),
            32 * GIB,
            None,
            DeviceMeasurements::Cuda {
                free_bytes: GIB,
                total_bytes: 32 * GIB,
            },
        )
        .unwrap();
        assert_eq!(reading.len(), 1);
        // Host planning reserve is 3.2 GiB of 32 GiB.
        assert_eq!(reading[0].ceiling_bytes, 10 * GIB - 32 * GIB / 10);
        assert_eq!(reading[0].band, MemoryBand::Normal);
    }

    #[test]
    fn dedicated_cuda_reads_its_own_pool_and_host_staging() {
        let cuda = |free| DeviceMeasurements::Cuda {
            free_bytes: free,
            total_bytes: 24 * GIB,
        };
        let normal = readings(
            &host(8 * GIB, vec![]),
            64 * GIB,
            Some(24 * GIB),
            cuda(10 * GIB),
        )
        .unwrap();
        assert_eq!(normal.len(), 2);
        let (allocation, staging) = (normal[0], normal[1]);
        assert_eq!(allocation.role, DomainRole::Allocation);
        assert_eq!(allocation.constraint, MemoryConstraint::DeviceLocal);
        // 24 GiB card: planning reserve 1.2 GiB.
        assert_eq!(allocation.thresholds.planning_bytes, 24 * GIB / 20);
        assert_eq!(allocation.ceiling_bytes, 10 * GIB - 24 * GIB / 20);
        assert_eq!(allocation.band, MemoryBand::Normal);
        assert_eq!(staging.role, DomainRole::Staging);
        assert_eq!(staging.constraint, MemoryConstraint::HostRam);
        // 64 GiB host: planning reserve 6.4 GiB.
        assert_eq!(staging.ceiling_bytes, 8 * GIB - 64 * GIB / 10);
        assert_eq!(staging.band, MemoryBand::Normal);
        assert_eq!(band_of(&normal), MemoryBand::Normal);

        let card_reclaim = readings(
            &host(8 * GIB, vec![]),
            64 * GIB,
            Some(24 * GIB),
            cuda(GIB),
        )
        .unwrap();
        assert_eq!(card_reclaim[0].band, MemoryBand::Reclaim);
        assert_eq!(card_reclaim[0].ceiling_bytes, 0);
        assert_eq!(band_of(&card_reclaim), MemoryBand::Reclaim);
    }

    #[test]
    fn staging_reclaim_holds_a_dedicated_load_in_reclaim() {
        let staged = readings(
            &host(6 * GIB, vec![]),
            64 * GIB,
            Some(24 * GIB),
            DeviceMeasurements::Cuda {
                free_bytes: 20 * GIB,
                total_bytes: 24 * GIB,
            },
        )
        .unwrap();
        assert_eq!(staged[0].band, MemoryBand::Normal);
        assert_eq!(staged[1].role, DomainRole::Staging);
        assert_eq!(staged[1].band, MemoryBand::Reclaim);
        assert_eq!(staged[1].ceiling_bytes, 0);
        assert_eq!(band_of(&staged), MemoryBand::Reclaim);
    }

    #[test]
    fn dedicated_vulkan_uses_budget_less_usage() {
        let vulkan = |usage| DeviceMeasurements::Vulkan {
            heap_budget_bytes: 14 * GIB,
            heap_usage_bytes: usage,
        };
        let normal = readings(
            &host(8 * GIB, vec![]),
            32 * GIB,
            Some(16 * GIB),
            vulkan(6 * GIB),
        )
        .unwrap();
        assert_eq!(normal[0].headroom_bytes, 8 * GIB);
        // 16 GiB heap budget: planning reserve 819 MiB.
        assert_eq!(normal[0].ceiling_bytes, 8 * GIB - 16 * GIB / 20);
        assert_eq!(normal[0].band, MemoryBand::Normal);
        let reclaim = readings(
            &host(8 * GIB, vec![]),
            32 * GIB,
            Some(16 * GIB),
            vulkan(13 * GIB + GIB / 4),
        )
        .unwrap();
        assert_eq!(reclaim[0].band, MemoryBand::Reclaim);
        assert_eq!(reclaim[0].ceiling_bytes, 0);
        // Usage past the budget saturates rather than wrapping.
        let over = readings(
            &host(8 * GIB, vec![]),
            32 * GIB,
            Some(16 * GIB),
            vulkan(15 * GIB),
        )
        .unwrap();
        assert_eq!(over[0].headroom_bytes, 0);
    }

    #[test]
    fn a_dedicated_device_never_reports_host_measurements() {
        for measurements in [
            DeviceMeasurements::Host,
            DeviceMeasurements::Metal {
                recommended_working_set_bytes: GIB,
                current_allocated_bytes: 0,
            },
        ] {
            assert!(matches!(
                readings(
                    &host(8 * GIB, vec![]),
                    32 * GIB,
                    Some(16 * GIB),
                    measurements
                ),
                Err(MemoryPolicyError::MismatchedObservation { .. })
            ));
        }
    }

    #[test]
    fn host_backed_fit_subtracts_the_planning_reserve() {
        let catalog = DeviceCatalog::discover().unwrap();
        let topology = catalog.topology();
        let host = catalog.host_memory_status().unwrap();
        let cpu = topology
            .devices()
            .iter()
            .find(|device| device.selector == DeviceSelector::HostCpu)
            .unwrap();
        let reserves = MemoryReserves::standard();
        let capacity = topology.host_pool().capacity_bytes;
        let bounded = host
            .limits
            .iter()
            .map(|limit| limit.limit_bytes)
            .fold(capacity, u64::min);
        let fits = fit_capacities(&topology, cpu, &host, &reserves).unwrap();
        assert_eq!(fits.len(), 1);
        let (role, fit) = fits[0];
        assert_eq!(role, DomainRole::Allocation);
        assert_eq!(fit.capacity_bytes, bounded);
        assert_eq!(
            fit.fit_bytes(),
            bounded
                - reserves
                    .for_domain(MemoryPoolKind::HostRam, capacity)
                    .planning_bytes
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_fit_respects_the_recommended_working_set() {
        let catalog = DeviceCatalog::discover().unwrap();
        let topology = catalog.topology();
        let host = catalog.host_memory_status().unwrap();
        let Some(metal) = topology
            .devices()
            .iter()
            .find(|device| device.backend == seismic::BackendName::Metal)
        else {
            return;
        };
        let reserves = MemoryReserves::standard();
        let host_capacity = host
            .limits
            .iter()
            .map(|limit| limit.limit_bytes)
            .fold(topology.host_pool().capacity_bytes, u64::min);
        let working_set = metal.recommended_working_set_bytes().unwrap();
        let fits = fit_capacities(&topology, metal, &host, &reserves).unwrap();
        assert_eq!(fits.len(), 1);
        assert_eq!(fits[0].1.capacity_bytes, host_capacity.min(working_set));
        assert_eq!(
            fits[0].1.fit_bytes(),
            host_capacity.min(working_set)
                - reserves
                    .for_domain(MemoryPoolKind::HostRam, topology.host_pool().capacity_bytes)
                    .planning_bytes
        );
    }
}
