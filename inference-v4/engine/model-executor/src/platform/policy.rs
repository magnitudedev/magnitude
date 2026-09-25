//! The engine's memory policy over Seismic facts and observations.
//!
//! Seismic establishes pools, capacities and scoped observations and enforces
//! the limits it is given; it holds no reserve policy. This module applies
//! the selected integrated policy:
//!
//! - Assessment compares a workload with the domain's total capacity, bounded
//!   by applicable process limits. It uses no current free memory.
//! - Load (and any incremental growth) requires the additional allocation to
//!   fit current headroom and every applicable device/process limit. Bytes
//!   Seismic already charged are absent from observed headroom and are never
//!   subtracted again.

use seismic::{
    DeviceCatalog, DeviceInfo, DeviceMeasurements, DeviceMemory, DeviceMemoryInfo, DeviceTopology,
    HostMemoryStatus, LimitVisibility, MemoryPoolKind, ObservationError, PressureLevel,
};
use std::fmt;

/// The constraint an admission check failed.
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
    /// The OS reports memory stalls severe enough to pause growth.
    Pressure(PressureLevel),
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
            Self::Pressure(level) => write!(formatter, "platform memory pressure is {level:?}"),
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
/// bounded by applicable process limits and device working-set advice.
pub fn assessment_capacity(
    topology: &DeviceTopology,
    device: &DeviceInfo,
    host: &HostMemoryStatus,
) -> Result<u64, MemoryPolicyError> {
    let memory = established(device)?;
    let pool = topology
        .pool(memory.allocation_pool)
        .expect("a device's pools belong to its topology");
    let capacity = match pool.kind {
        MemoryPoolKind::HostRam => {
            visible_limits(host)?;
            let bounded = host
                .limits
                .iter()
                .map(|limit| limit.limit_bytes)
                .fold(pool.capacity_bytes, u64::min);
            bounded
        }
        MemoryPoolKind::DeviceLocal => pool.capacity_bytes,
    };
    Ok(device
        .recommended_working_set_bytes()
        .map_or(capacity, |working_set| capacity.min(working_set)))
}

/// Fresh headroom for one opened device. The returned bytes are additional
/// bytes: the device's existing Seismic charges are already absent from each
/// observation. The tightest applicable domain constraint wins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrowthAvailability {
    pub bytes: u64,
    pub constraint: MemoryConstraint,
}

pub fn growth_availability(
    catalog: &DeviceCatalog,
    device: &seismic::Device,
) -> Result<GrowthAvailability, MemoryPolicyError> {
    let info = device.info();
    let memory = established(info)?;
    let status = device
        .memory_status()
        .map_err(MemoryPolicyError::Observation)?;
    let mut available = if memory.allocates_host_memory() {
        let host = catalog
            .host_memory_status()
            .map_err(MemoryPolicyError::Observation)?;
        visible_limits(&host)?;
        if let Some(level @ (PressureLevel::Pressure | PressureLevel::Emergency)) = host.pressure {
            return Err(MemoryPolicyError::Pressure(level));
        }
        let headroom = host
            .limits
            .iter()
            .map(|limit| limit.remaining_bytes())
            .fold(host.headroom.bytes, u64::min);
        GrowthAvailability {
            bytes: headroom,
            constraint: MemoryConstraint::HostRam,
        }
    } else {
        GrowthAvailability {
            bytes: u64::MAX,
            constraint: MemoryConstraint::DeviceLocal,
        }
    };
    let mut tighten = |bytes, constraint| {
        if bytes < available.bytes {
            available = GrowthAvailability { bytes, constraint };
        }
    };
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
            let available = recommended_working_set_bytes.saturating_sub(current_allocated_bytes);
            tighten(available, MemoryConstraint::DeviceWorkingSet);
        }
        // A dedicated device's pool is bounded by the driver's free bytes. An
        // integrated (host-backed) device allocates host RAM, bounded by the
        // host check above; its driver-reported free bytes exclude
        // reclaimable page cache, so they are not an allocation bound.
        DeviceMeasurements::Cuda { free_bytes, .. } => {
            if !memory.allocates_host_memory() {
                tighten(free_bytes, MemoryConstraint::DeviceLocal);
            }
        }
        // A dedicated Vulkan device is bounded by its memory budget less what
        // this process already uses of the heap.
        DeviceMeasurements::Vulkan {
            heap_budget_bytes,
            heap_usage_bytes,
        } => {
            if !memory.allocates_host_memory() {
                let available = heap_budget_bytes.saturating_sub(heap_usage_bytes);
                tighten(available, MemoryConstraint::DeviceLocal);
            }
        }
    }
    Ok(available)
}

/// Refresh Seismic's enforced ceiling from one required live observation.
/// The observation already excludes existing charges, so the ceiling is the
/// current charge plus the newly available bytes.
pub fn refresh_allocation_ceiling(
    catalog: &DeviceCatalog,
    device: &seismic::Device,
) -> Result<GrowthAvailability, MemoryPolicyError> {
    let charged = device.memory_usage().charged;
    let available = match growth_availability(catalog, device) {
        Ok(available) => available,
        Err(error) => {
            // Blind observations and platform pressure both revoke unspent
            // grants immediately, including one from an earlier reading.
            device.set_memory_limit(Some(charged));
            return Err(error);
        }
    };
    device.set_memory_limit(Some(charged.saturating_add(available.bytes)));
    Ok(available)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_backed_assessment_uses_the_domain_and_process_capacity() {
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
        assert_eq!(assessment_capacity(&topology, cpu, &host), Ok(bounded));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_assessment_respects_the_recommended_working_set() {
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
        let host_capacity = host
            .limits
            .iter()
            .map(|limit| limit.limit_bytes)
            .fold(topology.host_pool().capacity_bytes, u64::min);
        let working_set = metal.recommended_working_set_bytes().unwrap();
        assert_eq!(
            assessment_capacity(&topology, metal, &host),
            Ok(host_capacity.min(working_set))
        );
    }
}
