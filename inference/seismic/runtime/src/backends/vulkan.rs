//! The Vulkan backend's composition (Vulkan backend spec §9): a native-only
//! opened device with no compiler target, discovery from the device floor,
//! and storage behind the shared allocation core.

use crate::devices::OpenError;
use crate::driver::{self, Allocation, AllocationLimits, Storage};
use crate::memory::{ExecutorMemory, MemoryDomain, MemoryUsage};
use crate::native::abi::vulkan::VulkanFeatures;
use seismic_compiler::errors::{ExecutionError, TargetError};
use seismic_compiler::prepared::DeviceIdentity;
use seismic_topology::Executor;
use std::any::Any;
use std::sync::Arc;

/// Largest alignment an allocation may request; the block allocator admits
/// any power of two up to its block size.
const MAX_ALLOCATION_ALIGNMENT: u64 = 4096;

/// An opened Vulkan device: the service, its facts, and its memory domain.
pub(crate) struct VulkanOpened {
    identity: DeviceIdentity,
    service: seismic_vulkan::Device,
    memory: Arc<MemoryDomain>,
    upload: Arc<MemoryDomain>,
}

impl VulkanOpened {
    pub(super) fn open(
        uuid: [u8; 16],
        info: &Executor,
        topology: &seismic_topology::DeviceTopology,
        memory: Arc<ExecutorMemory>,
    ) -> Result<Self, OpenError> {
        let service = seismic_vulkan::Device::open(uuid).map_err(|error| match error {
            seismic_vulkan::OpenError::Missing => OpenError::IdentityChanged(info.selector()),
            other => OpenError::Backend(TargetError::DeviceUnavailable(other.to_string())),
        })?;
        let account =
            |domain: Arc<MemoryDomain>| -> Arc<dyn seismic_compiler::executable::MemoryAccount> {
                Arc::new(crate::memory::MemoryAccount(domain))
            };
        service.set_host_memory_account(account(memory.staging.clone()));
        // A placement of one path per heap (a discrete device) charges each
        // heap's blocks to that path; a placement whose paths partition one
        // allocation route (a device whose memory is the host's) charges
        // every block to the whole placement, which spends its paths in order.
        let mut accounts = Vec::new();
        for (storage, kind) in [
            (
                seismic_vulkan::PlacedStorage::Device,
                seismic_topology::StorageKind::Device,
            ),
            (
                seismic_vulkan::PlacedStorage::Upload,
                seismic_topology::StorageKind::Upload,
            ),
        ] {
            let domain = match storage {
                seismic_vulkan::PlacedStorage::Device => &memory.device,
                seismic_vulkan::PlacedStorage::Upload => &memory.upload,
            };
            let heaps = info
                .placement(kind)
                .iter()
                .map(|id| {
                    let path = topology
                        .access_path(*id)
                        .expect("placement belongs to its topology");
                    match path.route() {
                        seismic_topology::AccessRoute::VulkanHeap(heap) => Some((*id, heap)),
                        seismic_topology::AccessRoute::Device
                        | seismic_topology::AccessRoute::Upload => None,
                    }
                })
                .collect::<Option<Vec<_>>>();
            match heaps {
                Some(heaps) => {
                    for (id, heap) in heaps {
                        let path = memory
                            .path(id)
                            .expect("an executor's memory has a scope for each of its paths");
                        accounts.push(seismic_vulkan::AllocationAccount {
                            storage,
                            heap: Some(heap),
                            account: account(domain.physical_on(&path.scope)),
                        });
                    }
                }
                None => {
                    accounts.push(seismic_vulkan::AllocationAccount {
                        storage,
                        heap: None,
                        account: account(domain.physical()),
                    });
                }
            }
        }
        service.set_allocation_accounts(accounts);
        service.set_driver_memory_account(account(memory.device.physical()));
        Ok(Self {
            identity: driver::fresh_device_identity(),
            service,
            upload: memory.upload.clone(),
            memory: memory.device.clone(),
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

    /// The usage of `storage`: its domain's charges, with what the backend
    /// holds for it beyond them.
    pub(crate) fn usage(&self, storage: seismic_vulkan::PlacedStorage) -> MemoryUsage {
        let domain = match storage {
            seismic_vulkan::PlacedStorage::Device => &self.memory,
            seismic_vulkan::PlacedStorage::Upload => &self.upload,
        };
        let mut usage = domain.usage();
        usage.unallocated = self.service.unallocated_bytes(storage);
        usage.held += self.service.held_beyond_allocations(storage);
        usage
    }

    pub(crate) fn set_limit(&self, kind: seismic_topology::StorageKind, limit: Option<u64>) {
        match kind {
            seismic_topology::StorageKind::Device => self.memory.set_limit(limit),
            seismic_topology::StorageKind::Upload => self.upload.set_limit(limit),
        }
    }

    pub(crate) fn restart_memory_high_water(&self) {
        self.memory.restart_high_water()
    }

    fn allocate_with(
        &self,
        memory: &Arc<MemoryDomain>,
        bytes: u64,
        alignment: u64,
        make: impl FnOnce(
            &seismic_vulkan::Device,
            Option<u64>,
        ) -> Result<seismic_vulkan::Buffer, ExecutionError>,
    ) -> Result<Arc<Allocation>, ExecutionError> {
        let limits = AllocationLimits {
            max_allocation_bytes: self.service.facts().limits.max_allocation_bytes,
            max_allocation_alignment: MAX_ALLOCATION_ALIGNMENT,
        };
        // The device usage this allocation may add: the limit's room before
        // its own charge.
        let usage = memory.usage();
        let room = usage.limit.map(|limit| limit.saturating_sub(usage.charged));
        let mut reservation = driver::reserve(memory, bytes)?;
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
        self.allocate_with(&self.memory, bytes, alignment, |service, room| {
            service.allocate(bytes, alignment, room)
        })
    }

    pub(crate) fn allocate_upload(
        &self,
        bytes: u64,
        alignment: u64,
    ) -> Result<Arc<Allocation>, ExecutionError> {
        self.allocate_with(&self.upload, bytes, alignment, |service, room| {
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
