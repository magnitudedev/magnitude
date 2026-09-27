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
//! - Headroom at or below the planning reserve is the Reclaim band. Only
//!   other programs can cause it, since no engine claim crosses the line.

use seismic::{
    DeviceCatalog, DeviceInfo, DeviceMeasurements, DeviceMemory, DeviceMemoryInfo, DeviceSelector,
    DeviceTopology, HostMemoryStatus, LimitVisibility, MemoryPoolId, MemoryPoolKind,
    ObservationError,
};
use std::fmt;

/// The constraint that bounds a domain's claimable ceiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryConstraint {
    /// Host RAM and its applicable process limits.
    HostRam,
    /// A device-local pool.
    DeviceLocal,
    /// Metal's recommended working set for the device.
    DeviceWorkingSet,
}

impl fmt::Display for MemoryConstraint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::HostRam => "host RAM",
            Self::DeviceLocal => "device memory",
            Self::DeviceWorkingSet => "the device's recommended working set",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryPolicyError {
    /// Seismic reports the device's backing as not normalized.
    UnsupportedBacking { device: String, reason: String },
    /// Some applicable limits are hidden and cannot be presumed unlimited.
    HiddenLimits,
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
                write!(formatter, "device {device} has no qualified memory backing: {reason}")
            }
            Self::HiddenLimits => formatter.write_str(
                "cgroup ancestors of this process are hidden; their memory limits cannot be established",
            ),
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

fn visible_limits(host: &HostMemoryStatus) -> Result<(), MemoryPolicyError> {
    match host.limit_visibility {
        LimitVisibility::Complete => Ok(()),
        LimitVisibility::CgroupAncestorsHidden => Err(MemoryPolicyError::HiddenLimits),
    }
}

const GIB: u64 = 1024 * 1024 * 1024;

/// The threshold policy: how much of each memory domain a loaded model may
/// not use. The host (service or CLI) chooses it and passes it to the engine;
/// the engine never defaults it. `standard` is the only place the values live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryReserves {
    planning_fraction_divisor: u64,
    planning_floor_bytes: u64,
    emergency_fraction_divisor: u64,
    emergency_floor_bytes: u64,
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
    /// Planning reserve `max(capacity / 10, 2 GiB)`; emergency reserve
    /// `max(capacity / 20, 1 GiB)`.
    pub fn standard() -> Self {
        Self {
            planning_fraction_divisor: 10,
            planning_floor_bytes: 2 * GIB,
            emergency_fraction_divisor: 20,
            emergency_floor_bytes: GIB,
        }
    }

    pub fn for_domain(&self, capacity_bytes: u64) -> DomainThresholds {
        DomainThresholds {
            planning_bytes: (capacity_bytes / self.planning_fraction_divisor)
                .max(self.planning_floor_bytes),
            emergency_bytes: (capacity_bytes / self.emergency_fraction_divisor)
                .max(self.emergency_floor_bytes),
        }
    }

    /// Stable identity of the policy for assessment cache keys.
    pub fn identity(&self) -> String {
        format!(
            "reserves-v1:{}:{}:{}:{}",
            self.planning_fraction_divisor,
            self.planning_floor_bytes,
            self.emergency_fraction_divisor,
            self.emergency_floor_bytes
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
    visible_limits(host)?;
    let host_capacity = host
        .limits
        .iter()
        .map(|limit| limit.limit_bytes)
        .fold(host_pool.capacity_bytes, u64::min);
    let host_fit = FitCapacity {
        domain: host_pool.id,
        kind: MemoryPoolKind::HostRam,
        capacity_bytes: host_capacity,
        reserve_bytes: reserves.for_domain(host_pool.capacity_bytes).planning_bytes,
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
                reserve_bytes: reserves.for_domain(pool.capacity_bytes).planning_bytes,
            },
        ),
        (DomainRole::Staging, host_fit),
    ])
}

/// The band of one observed domain. Only other processes can move a domain
/// into `Reclaim`, since every engine claim keeps headroom above the
/// planning reserve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryBand {
    /// Headroom above the planning reserve.
    Normal,
    /// Headroom at or below the planning reserve: pause growth and release.
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
    /// planning reserve and, on Metal, the room left in the working set.
    pub ceiling_bytes: u64,
    pub band: MemoryBand,
}

impl DomainReading {
    fn new(
        role: DomainRole,
        domain: MemoryPoolId,
        headroom_bytes: u64,
        constraint: MemoryConstraint,
        thresholds: DomainThresholds,
    ) -> Self {
        Self {
            role,
            domain,
            headroom_bytes,
            constraint,
            thresholds,
            ceiling_bytes: headroom_bytes.saturating_sub(thresholds.planning_bytes),
            band: if headroom_bytes > thresholds.planning_bytes {
                MemoryBand::Normal
            } else {
                MemoryBand::Reclaim
            },
        }
    }

    /// Bound the claimable ceiling by a further device constraint without a
    /// reserve of its own; the band stays the domain's.
    fn bounded(mut self, bytes: u64, constraint: MemoryConstraint) -> Self {
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

/// A dedicated device's own memory domain.
#[derive(Clone, Copy, Debug)]
struct DedicatedDomain {
    id: MemoryPoolId,
    capacity_bytes: u64,
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
    visible_limits(host)?;
    let host_headroom = host
        .limits
        .iter()
        .map(|limit| limit.remaining_bytes())
        .fold(host.headroom.bytes, u64::min);
    let host_thresholds = reserves.for_domain(host_capacity_bytes);
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
        );
        return Ok(vec![match *measurements {
            // Metal's working set bounds claims on the same host domain
            // without a reserve of its own; host headroom decides the band.
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
    Ok(vec![
        DomainReading::new(
            DomainRole::Allocation,
            dedicated.id,
            headroom,
            MemoryConstraint::DeviceLocal,
            reserves.for_domain(dedicated.capacity_bytes),
        ),
        DomainReading::new(
            DomainRole::Staging,
            host_pool,
            host_headroom,
            MemoryConstraint::HostRam,
            host_thresholds,
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
    let dedicated = (!memory.allocates_host_memory()).then(|| {
        let pool = topology
            .pool(memory.allocation_pool)
            .expect("a device's pools belong to its topology");
        DedicatedDomain {
            id: pool.id,
            capacity_bytes: pool.capacity_bytes,
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
        HeadroomBasis, HeadroomEstimate, HostMeasurements, ProcessLimitKind, ProcessMemoryLimit,
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
        }
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
            dedicated_capacity.map(|capacity_bytes| DedicatedDomain { id, capacity_bytes }),
            &measurements,
            &MemoryReserves::standard(),
        )
    }

    #[test]
    fn standard_reserves_use_the_larger_of_fraction_and_floor() {
        let reserves = MemoryReserves::standard();
        assert_eq!(
            reserves.for_domain(16 * GIB),
            DomainThresholds {
                planning_bytes: 2 * GIB,
                emergency_bytes: GIB,
            }
        );
        assert_eq!(
            reserves.for_domain(64 * GIB),
            DomainThresholds {
                planning_bytes: 64 * GIB / 10,
                emergency_bytes: 64 * GIB / 20,
            }
        );
    }

    #[test]
    fn host_ram_band_and_ceiling_follow_the_planning_reserve() {
        let normal = readings(&host(5 * GIB, vec![]), 16 * GIB, None, DeviceMeasurements::Host)
            .unwrap();
        assert_eq!(normal.len(), 1);
        assert_eq!(normal[0].role, DomainRole::Allocation);
        assert_eq!(normal[0].band, MemoryBand::Normal);
        assert_eq!(normal[0].ceiling_bytes, 3 * GIB);
        assert_eq!(normal[0].constraint, MemoryConstraint::HostRam);
        // Exactly at the planning reserve is already Reclaim.
        let at_line = readings(&host(2 * GIB, vec![]), 16 * GIB, None, DeviceMeasurements::Host)
            .unwrap();
        assert_eq!(at_line[0].band, MemoryBand::Reclaim);
        assert_eq!(at_line[0].ceiling_bytes, 0);
        // Below the emergency reserve the engine still reports Reclaim; the
        // kill belongs to the service.
        let below = readings(&host(GIB / 2, vec![]), 16 * GIB, None, DeviceMeasurements::Host)
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
        let mut hidden = host(12 * GIB, vec![]);
        hidden.limit_visibility = LimitVisibility::CgroupAncestorsHidden;
        assert_eq!(
            readings(&hidden, 16 * GIB, None, DeviceMeasurements::Host),
            Err(MemoryPolicyError::HiddenLimits)
        );
    }

    #[test]
    fn metal_working_set_bounds_the_ceiling_not_the_band() {
        let metal = |allocated| DeviceMeasurements::Metal {
            recommended_working_set_bytes: 12 * GIB,
            current_allocated_bytes: allocated,
        };
        // Working-set room (1 GiB) is tighter than host ceiling (4 GiB).
        let bounded = readings(&host(6 * GIB, vec![]), 16 * GIB, None, metal(11 * GIB)).unwrap();
        assert_eq!(bounded[0].band, MemoryBand::Normal);
        assert_eq!(bounded[0].ceiling_bytes, GIB);
        assert_eq!(bounded[0].constraint, MemoryConstraint::DeviceWorkingSet);
        // Host headroom (ceiling 4 GiB) is tighter than working-set room.
        let host_bound = readings(&host(6 * GIB, vec![]), 16 * GIB, None, metal(GIB)).unwrap();
        assert_eq!(host_bound[0].ceiling_bytes, 4 * GIB);
        assert_eq!(host_bound[0].constraint, MemoryConstraint::HostRam);
        // A full working set grants nothing but is not Reclaim by itself.
        let full = readings(&host(6 * GIB, vec![]), 16 * GIB, None, metal(12 * GIB)).unwrap();
        assert_eq!(full[0].band, MemoryBand::Normal);
        assert_eq!(full[0].ceiling_bytes, 0);
        let reclaim = readings(&host(GIB, vec![]), 16 * GIB, None, metal(GIB)).unwrap();
        assert_eq!(reclaim[0].band, MemoryBand::Reclaim);
        assert_eq!(reclaim[0].ceiling_bytes, 0);
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
        let normal = readings(&host(8 * GIB, vec![]), 64 * GIB, Some(24 * GIB), cuda(10 * GIB))
            .unwrap();
        assert_eq!(normal.len(), 2);
        let (allocation, staging) = (normal[0], normal[1]);
        assert_eq!(allocation.role, DomainRole::Allocation);
        assert_eq!(allocation.constraint, MemoryConstraint::DeviceLocal);
        // 24 GiB card: planning reserve 2.4 GiB.
        assert_eq!(allocation.thresholds.planning_bytes, 24 * GIB / 10);
        assert_eq!(allocation.ceiling_bytes, 10 * GIB - 24 * GIB / 10);
        assert_eq!(allocation.band, MemoryBand::Normal);
        assert_eq!(staging.role, DomainRole::Staging);
        assert_eq!(staging.constraint, MemoryConstraint::HostRam);
        // 64 GiB host: planning reserve 6.4 GiB.
        assert_eq!(staging.ceiling_bytes, 8 * GIB - 64 * GIB / 10);
        assert_eq!(staging.band, MemoryBand::Normal);
        assert_eq!(band_of(&normal), MemoryBand::Normal);

        let card_reclaim =
            readings(&host(8 * GIB, vec![]), 64 * GIB, Some(24 * GIB), cuda(2 * GIB)).unwrap();
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
        let normal =
            readings(&host(8 * GIB, vec![]), 32 * GIB, Some(16 * GIB), vulkan(6 * GIB)).unwrap();
        assert_eq!(normal[0].headroom_bytes, 8 * GIB);
        assert_eq!(normal[0].ceiling_bytes, 6 * GIB);
        assert_eq!(normal[0].band, MemoryBand::Normal);
        let reclaim =
            readings(&host(8 * GIB, vec![]), 32 * GIB, Some(16 * GIB), vulkan(13 * GIB)).unwrap();
        assert_eq!(reclaim[0].band, MemoryBand::Reclaim);
        assert_eq!(reclaim[0].ceiling_bytes, 0);
        // Usage past the budget saturates rather than wrapping.
        let over =
            readings(&host(8 * GIB, vec![]), 32 * GIB, Some(16 * GIB), vulkan(15 * GIB)).unwrap();
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
                readings(&host(8 * GIB, vec![]), 32 * GIB, Some(16 * GIB), measurements),
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
            bounded - reserves.for_domain(capacity).planning_bytes
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
                    .for_domain(topology.host_pool().capacity_bytes)
                    .planning_bytes
        );
    }
}
