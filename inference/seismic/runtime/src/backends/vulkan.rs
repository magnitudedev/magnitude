//! The Vulkan backend's composition (Vulkan backend spec §9): a native-only
//! opened device with no compiler target, discovery from the device floor,
//! and storage behind the shared allocation core.

use super::{Descriptor, DiscoveredDevice, DiscoveredMemory};
use crate::devices::{
    Availability, CapacityBasis, DeviceInfo, DeviceKind, DeviceSelector, LedgerKey, OpenError,
};
use crate::driver::{self, Allocation, AllocationLimits, Storage};
use crate::memory::{MemoryDomain, MemoryUsage};
use crate::native::abi::vulkan::VulkanFeatures;
use seismic_compiler::errors::{ExecutionError, TargetError};
use seismic_compiler::prepared::DeviceIdentity;
use seismic_lang::registry::BackendName;
use std::any::Any;
use std::sync::Arc;

/// Largest alignment an allocation may request; the block allocator admits
/// any power of two up to its block size.
const MAX_ALLOCATION_ALIGNMENT: u64 = 4096;

/// A discovered physical device. Integrated GPUs and software devices
/// allocate from host RAM; a discrete GPU has its device-local heap. Under
/// WDDM the driver grants a process the heap's budget, the operating
/// system's per-process budget, which other programs' use does not change;
/// elsewhere the budget follows other programs, and the heap, which already
/// excludes memory the driver reserves, is the capacity. A device below the
/// floor reports no budget and is never opened; its heap stands in.
pub(super) fn discovered(description: seismic_vulkan::Description) -> DiscoveredDevice {
    let facts = description.facts;
    let memory = if facts.device_type == seismic_vulkan::DeviceType::Discrete {
        let granted = if cfg!(target_os = "windows") {
            super::granted_bytes(BackendName::Vulkan, facts.uuid, || {
                description
                    .heap_budget_bytes
                    .ok_or_else(|| "the device reports no heap budget".to_owned())
            })
            .unwrap_or(facts.device_local_bytes)
        } else {
            facts.device_local_bytes
        };
        DiscoveredMemory::Dedicated {
            capacity_bytes: granted.min(facts.device_local_bytes),
            total_bytes: facts.device_local_bytes,
            basis: if cfg!(target_os = "windows") {
                CapacityBasis::VulkanHeapBudget
            } else {
                CapacityBasis::VulkanDeviceLocalHeap
            },
            ledger: LedgerKey::PhysicalGpu(facts.uuid),
            max_allocation_bytes: facts.limits.max_allocation_bytes,
            shared_device_local: facts.uploads_device_local(),
        }
    } else {
        DiscoveredMemory::Host {
            max_allocation_bytes: facts.limits.max_allocation_bytes,
        }
    };
    DiscoveredDevice {
        selector: DeviceSelector::Vulkan { uuid: facts.uuid },
        name: facts.name.clone(),
        kind: if facts.is_gpu() {
            DeviceKind::Gpu
        } else {
            DeviceKind::Cpu
        },
        backend: BackendName::Vulkan,
        availability: match description.floor {
            Ok(()) => Availability::Available,
            Err(reason) => Availability::Unavailable { reason },
        },
        memory,
        memory_bandwidth: None,
        descriptor: Descriptor::Vulkan { uuid: facts.uuid },
    }
}

/// An opened Vulkan device: the service, its facts, and its memory domain.
pub(crate) struct VulkanOpened {
    identity: DeviceIdentity,
    service: seismic_vulkan::Device,
    memory: Arc<MemoryDomain>,
}

impl VulkanOpened {
    pub(super) fn open(
        uuid: [u8; 16],
        info: &DeviceInfo,
        memory: Arc<MemoryDomain>,
    ) -> Result<Self, OpenError> {
        let service = seismic_vulkan::Device::open(uuid).map_err(|error| match error {
            seismic_vulkan::OpenError::Missing => OpenError::IdentityChanged(info.selector),
            other => OpenError::Backend(TargetError::DeviceUnavailable(other.to_string())),
        })?;
        Ok(Self {
            identity: driver::fresh_device_identity(),
            service,
            memory,
        })
    }

    pub(crate) fn identity(&self) -> DeviceIdentity {
        self.identity
    }

    pub(crate) fn service(&self) -> &seismic_vulkan::Device {
        &self.service
    }

    /// The device features the generated prefix exposes.
    pub(crate) fn features(&self) -> VulkanFeatures {
        VulkanFeatures::of(&self.service.facts().formation())
    }

    pub(crate) fn memory_usage(&self) -> MemoryUsage {
        let mut usage = self.memory.usage();
        usage.unallocated = self.service.unallocated_bytes();
        usage.held += self.service.held_beyond_allocations();
        usage
    }

    pub(crate) fn set_memory_limit(&self, limit: Option<u64>) {
        self.memory.set_limit(limit)
    }

    pub(crate) fn restart_memory_high_water(&self) {
        self.memory.restart_high_water()
    }

    fn allocate_with(
        &self,
        bytes: u64,
        alignment: u64,
        make: impl FnOnce(&seismic_vulkan::Device, Option<u64>) -> Result<seismic_vulkan::Buffer, ExecutionError>,
    ) -> Result<Arc<Allocation>, ExecutionError> {
        let limits = AllocationLimits {
            max_allocation_bytes: self.service.facts().limits.max_allocation_bytes,
            max_allocation_alignment: MAX_ALLOCATION_ALIGNMENT,
        };
        // The device usage this allocation may add: the limit's room before
        // its own charge.
        let usage = self.memory.usage();
        let room = usage.limit.map(|limit| limit.saturating_sub(usage.charged));
        let mut reservation = driver::reserve(&self.memory, bytes)?;
        driver::allocate_reserved(limits, bytes, alignment, &mut reservation, || {
            Ok(Box::new(VulkanStorage {
                service: self.service.clone(),
                buffer: make(&self.service, room)?,
            }))
        })
    }

    pub(crate) fn allocate(
        &self,
        bytes: u64,
        alignment: u64,
    ) -> Result<Arc<Allocation>, ExecutionError> {
        self.allocate_with(bytes, alignment, |service, room| {
            service.allocate(bytes, alignment, room)
        })
    }

    pub(crate) fn allocate_upload(
        &self,
        bytes: u64,
        alignment: u64,
    ) -> Result<Arc<Allocation>, ExecutionError> {
        self.allocate_with(bytes, alignment, |service, room| {
            service.allocate_upload(bytes, alignment, room)
        })
    }
}

struct VulkanStorage {
    service: seismic_vulkan::Device,
    buffer: seismic_vulkan::Buffer,
}

impl Storage for VulkanStorage {
    fn read(&self, offset: u64, into: &mut [u8]) -> Result<(), ExecutionError> {
        self.service.read(&self.buffer, offset, into)
    }
    fn write(&self, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError> {
        self.service.write(&self.buffer, offset, bytes)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The Vulkan buffer of an allocation of a Vulkan device.
pub(crate) fn vulkan_buffer(allocation: &Allocation) -> &seismic_vulkan::Buffer {
    &allocation
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .unwrap_or_else(|| panic!("Tensor allocation backend invariant violated after successful WrongDevice validation"))
        .buffer
}
