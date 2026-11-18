//! The Vulkan backend's composition (Vulkan backend spec §9): a native-only
//! opened device with no compiler target, discovery from the device floor,
//! and storage behind the shared allocation core.

use crate::devices::OpenError;
use crate::driver::{self, Allocation, AllocationLimits, Storage};
use crate::memory::DeviceStores;
use crate::native::abi::vulkan::VulkanFeatures;
use seismic_compiler::errors::{ExecutionError, TargetError};
use seismic_compiler::prepared::DeviceIdentity;
use seismic_topology::Executor;
use std::any::Any;
use std::sync::Arc;

/// Largest alignment an allocation may request; the block allocator admits
/// any power of two up to its block size.
const MAX_ALLOCATION_ALIGNMENT: u64 = 4096;

/// An opened Vulkan device: the service, its facts, and the stores its
/// storage kinds lie in.
pub(crate) struct VulkanOpened {
    identity: DeviceIdentity,
    service: seismic_vulkan::Device,
    memory: Arc<DeviceStores>,
}

impl VulkanOpened {
    pub(super) fn open(
        uuid: [u8; 16],
        info: &Executor,
        memory: Arc<DeviceStores>,
        probed: &seismic_vulkan::Probed,
    ) -> Result<Self, OpenError> {
        let stores = seismic_vulkan::MemoryStores {
            // The store of every kind the device's placements allocate in.
            kinds: memory.kind_stores(),
            host: memory.host_store(),
        };
        let service = seismic_vulkan::Device::open(uuid, stores, probed).map_err(|error| match error {
            seismic_vulkan::OpenError::Missing => OpenError::IdentityChanged(info.selector()),
            busy @ seismic_vulkan::OpenError::Busy(_) => OpenError::Backend(TargetError::DeviceBusy(busy.to_string())),
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
        VulkanFeatures::of(&self.service.formation())
    }

    /// A buffer charged as `charge` says, landing on the first of its kinds
    /// with room, ordered by
    /// `access`'s hazard domain where given; it lies along the path of the
    /// kind it landed in ([`seismic_vulkan::Buffer::kind`]).
    pub(crate) fn allocate_contract(
        &self,
        charge: crate::memory::AllocationCharge<'_>,
        bytes: u64,
        alignment: u64,
        access: Option<Arc<driver::AccessGroup>>,
    ) -> Result<Arc<Allocation>, ExecutionError> {
        let crate::memory::AllocationCharge { entry, kinds } = charge;
        driver::allocate_in(self.allocation_limits(), &self.memory, bytes, alignment, access, || {
            let buffer = self.service.allocate_kinds(kinds, bytes, alignment, entry)?;
            let kind = buffer.kind();
            let storage: Box<dyn Storage> = Box::new(VulkanStorage { service: self.service.clone(), buffer: Arc::new(buffer) });
            Ok((kind, storage))
        })
    }

    /// One block of a granule pool, one memory object of `kind` and the
    /// buffer bound to it, lying along the kind's path
    /// (`driver::Opened::allocate_block`).
    pub(crate) fn allocate_block(
        &self,
        kind: u32,
        bytes: u64,
        entry: &Arc<dyn crate::memory::Entry>,
        access: Arc<driver::AccessGroup>,
    ) -> Result<Arc<Allocation>, ExecutionError> {
        self.allocation_limits().admit_bytes(bytes)?;
        let buffer = self.service.allocate_block(kind, bytes, entry)?;
        Ok(Allocation::new_in(
            driver::fresh_allocation_identity(),
            bytes,
            self.memory.path_of(kind),
            Box::new(VulkanStorage { service: self.service.clone(), buffer: Arc::new(buffer) }),
            Some(access),
        ))
    }


    fn allocation_limits(&self) -> AllocationLimits {
        AllocationLimits {
            max_allocation_bytes: self.service.facts().limits.max_allocation_bytes,
            max_allocation_alignment: MAX_ALLOCATION_ALIGNMENT,
        }
    }

}

/// Storage of one buffer.
struct VulkanStorage {
    service: seismic_vulkan::Device,
    buffer: Arc<seismic_vulkan::Buffer>,
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
    fn host_address(&self) -> Option<std::ptr::NonNull<u8>> {
        self.buffer.host_address()
    }
}

/// The device address of byte `offset` of a Vulkan allocation.
pub(crate) fn vulkan_address(allocation: &Allocation, offset: u64) -> u64 {
    vulkan_buffer(allocation).address() + offset
}

/// Byte `offset` of a Vulkan allocation as a transfer reaches it.
pub(crate) fn vulkan_transfer_end(allocation: &Allocation, offset: u64) -> seismic_vulkan::transfer::TransferEnd<'_> {
    seismic_vulkan::transfer::TransferEnd(vulkan_buffer(allocation), offset)
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

/// The stores a test opens a device of `facts` with: one unbounded store
/// for every kind and the host.
#[cfg(test)]
pub(crate) fn test_stores(facts: &seismic_vulkan::Facts) -> seismic_vulkan::MemoryStores {
    let store: Arc<dyn crate::memory::Store> =
        Arc::new(seismic_memory::entry::Tally::unbounded(seismic_topology::StoreIdentity::HostRam));
    let kinds = (0..facts.memory.types.len() as u32).map(|kind| (kind, store.clone())).collect();
    seismic_vulkan::MemoryStores { kinds, host: store }
}
