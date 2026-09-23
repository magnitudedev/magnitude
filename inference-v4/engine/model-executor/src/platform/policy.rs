//! The engine's memory policy over Seismic facts and observations.
//!
//! Seismic establishes pools, capacities and scoped observations and enforces
//! the limits it is given; it holds no reserve policy. This module applies
//! the selected integrated policy:
//!
//! - Assessment compares a workload with stable capacity minus the planning
//!   reserve. It uses no current free memory.
//! - Load (and any incremental growth) requires the additional allocation to
//!   fit current headroom minus the planning reserve and every applicable
//!   device/process limit. Bytes Seismic already charged are already absent
//!   from observed headroom and are never subtracted again.
//! - Unified backing has one host reserve; a device-local pool has its own.
//!
//! The host emergency threshold (max(5% of RAM, 1 GiB)) belongs to service
//! pressure supervision and is never an allocation reserve.

use seismic::{
    DeviceCatalog, DeviceInfo, DeviceMeasurements, DeviceMemory, DeviceMemoryInfo, DeviceTopology,
    HostMemoryStatus, LimitVisibility, MemoryPoolKind, ObservationError,
};
use std::fmt;

const GIB: u64 = 1 << 30;

/// Dedicated device planning reserve, once per device-local pool.
pub const DEDICATED_PLANNING_RESERVE: u64 = 1536 << 20;

/// Host planning reserve: max(10% of physical RAM, 2 GiB).
pub fn host_planning_reserve(host_capacity_bytes: u64) -> u64 {
    (host_capacity_bytes / 10).max(2 * GIB)
}

/// The constraint an admission check failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryConstraint {
    /// Host RAM after the host planning reserve.
    HostRam,
    /// A device-local pool after the dedicated planning reserve.
    DeviceLocal,
    /// Metal's recommended working set for the device.
    DeviceWorkingSet,
}

impl fmt::Display for MemoryConstraint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::HostRam => "host RAM after the planning reserve",
            Self::DeviceLocal => "device memory after the planning reserve",
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
    Insufficient {
        constraint: MemoryConstraint,
        required: u64,
        available: u64,
    },
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
            Self::Insufficient {
                constraint,
                required,
                available,
            } => write!(
                formatter,
                "{required} bytes do not fit {constraint}: {available} bytes available"
            ),
        }
    }
}

impl std::error::Error for MemoryPolicyError {}

fn established<'a>(device: &'a DeviceInfo) -> Result<&'a DeviceMemoryInfo, MemoryPolicyError> {
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

/// Stable planning capacity of a device for assessment: its allocation pool
/// bounded by applicable process limits, minus that pool's planning reserve.
pub fn assessment_capacity(
    topology: &DeviceTopology,
    device: &DeviceInfo,
    host: &HostMemoryStatus,
) -> Result<u64, MemoryPolicyError> {
    let memory = established(device)?;
    let pool = topology
        .pool(memory.allocation_pool)
        .expect("a device's pools belong to its topology");
    Ok(match pool.kind {
        MemoryPoolKind::HostRam => {
            visible_limits(host)?;
            let bounded = host
                .limits
                .iter()
                .map(|limit| limit.limit_bytes)
                .fold(pool.capacity_bytes, u64::min);
            bounded.saturating_sub(host_planning_reserve(pool.capacity_bytes))
        }
        MemoryPoolKind::DeviceLocal => pool
            .capacity_bytes
            .saturating_sub(DEDICATED_PLANNING_RESERVE),
    })
}

/// Admits growing an opened device's Seismic charges to `budget_bytes`
/// against fresh scoped observations sampled in this process.
pub fn admit_growth(
    catalog: &DeviceCatalog,
    device: &seismic::Device,
    budget_bytes: u64,
) -> Result<(), MemoryPolicyError> {
    let info = device.info();
    let memory = established(info)?;
    let usage = device.memory_usage();
    let additional = budget_bytes.saturating_sub(usage.charged);
    let insufficient = |constraint, available| MemoryPolicyError::Insufficient {
        constraint,
        required: additional,
        available,
    };
    let status = device
        .memory_status()
        .map_err(MemoryPolicyError::Observation)?;
    if memory.allocates_host_memory() {
        let host = catalog
            .host_memory_status()
            .map_err(MemoryPolicyError::Observation)?;
        visible_limits(&host)?;
        let headroom = host
            .limits
            .iter()
            .map(|limit| limit.remaining_bytes())
            .fold(host.headroom.bytes, u64::min);
        // The host pool is unique; its capacity defines the host reserve.
        let host_capacity = catalog.topology().host_pool().capacity_bytes;
        let available = headroom.saturating_sub(host_planning_reserve(host_capacity));
        if additional > available {
            return Err(insufficient(MemoryConstraint::HostRam, available));
        }
    }
    match status.measurements {
        DeviceMeasurements::Host => {
            if !memory.allocates_host_memory() {
                return Err(MemoryPolicyError::MismatchedObservation {
                    device: info.selector.to_string(),
                });
            }
        }
        DeviceMeasurements::Metal {
            recommended_working_set_bytes,
            current_allocated_bytes,
        } => {
            let available =
                recommended_working_set_bytes.saturating_sub(current_allocated_bytes);
            if additional > available {
                return Err(insufficient(MemoryConstraint::DeviceWorkingSet, available));
            }
        }
        DeviceMeasurements::Cuda { free_bytes, .. } => {
            if memory.allocates_host_memory() {
                return Err(MemoryPolicyError::MismatchedObservation {
                    device: info.selector.to_string(),
                });
            }
            let available = free_bytes.saturating_sub(DEDICATED_PLANNING_RESERVE);
            if additional > available {
                return Err(insufficient(MemoryConstraint::DeviceLocal, available));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_planning_reserve_is_the_larger_of_ten_percent_and_two_gib() {
        assert_eq!(host_planning_reserve(8 * GIB), 2 * GIB);
        assert_eq!(host_planning_reserve(64 * GIB), 64 * GIB / 10);
        assert_eq!(host_planning_reserve(20 * GIB), 2 * GIB);
    }

    #[test]
    fn host_backed_assessment_subtracts_one_host_reserve() {
        let catalog = DeviceCatalog::discover().unwrap();
        let topology = catalog.topology();
        let host = catalog.host_memory_status().unwrap();
        let cpu = topology
            .devices()
            .iter()
            .find(|device| device.selector == seismic::DeviceSelector::HostCpu)
            .unwrap();
        let capacity = topology.host_pool().capacity_bytes;
        let bounded = host
            .limits
            .iter()
            .map(|limit| limit.limit_bytes)
            .fold(capacity, u64::min);
        assert_eq!(
            assessment_capacity(&topology, cpu, &host),
            Ok(bounded.saturating_sub(host_planning_reserve(capacity)))
        );
    }

    #[test]
    fn growth_already_charged_is_not_subtracted_twice() {
        let catalog = DeviceCatalog::discover().unwrap();
        let device = catalog.open_backend(seismic::BackendName::Cpu).unwrap();
        let tensor = seismic::Tensor::zeros(&device, seismic::Element::f32(), &[1 << 20]).unwrap();
        let charged = device.memory_usage().charged;
        // Growing to exactly the charged total requests no additional bytes.
        assert_eq!(admit_growth(&catalog, &device, charged), Ok(()));
        drop(tensor);
    }
}
