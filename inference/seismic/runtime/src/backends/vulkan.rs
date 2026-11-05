//! The Vulkan backend's composition (Vulkan backend spec §9): a native-only
//! opened device with no compiler target, discovery from the device floor,
//! and storage behind the shared allocation core.

use crate::devices::OpenError;
use crate::driver::{self, Allocation, AllocationLimits, Storage};
use crate::memory::{DeviceLedgers, Placement};
use crate::native::abi::vulkan::VulkanFeatures;
use seismic_compiler::errors::{ExecutionError, TargetError};
use seismic_compiler::prepared::DeviceIdentity;
use seismic_topology::Executor;
use std::any::Any;
use std::sync::Arc;

/// Largest alignment an allocation may request; the block allocator admits
/// any power of two up to its block size.
const MAX_ALLOCATION_ALIGNMENT: u64 = 4096;

/// An opened Vulkan device: the service, its facts, and its accounting, a
/// placement per storage kind.
pub(crate) struct VulkanOpened {
    identity: DeviceIdentity,
    service: seismic_vulkan::Device,
    memory: Arc<DeviceLedgers>,
}

impl VulkanOpened {
    pub(super) fn open(
        uuid: [u8; 16],
        info: &Executor,
        memory: Arc<DeviceLedgers>,
        probed: &seismic_vulkan::Probed,
    ) -> Result<Self, OpenError> {
        let account =
            |placement: Arc<Placement>| -> Arc<dyn seismic_compiler::executable::MemoryAccount> {
                Arc::new(crate::memory::MemoryAccount(placement))
            };
        let accounts = seismic_vulkan::MemoryAccounts {
            host: account(memory.staging.clone()),
            // Every block is charged along its kind's paths.
            kinds: memory
                .kind_placements()
                .iter()
                .map(|(kind, placement)| seismic_vulkan::AllocationAccount { kind: *kind, account: account(placement.physical()) })
                .collect(),
            driver: account(memory.device().physical()),
        };
        let service = seismic_vulkan::Device::open(uuid, accounts, probed).map_err(|error| match error {
            seismic_vulkan::OpenError::Missing => OpenError::IdentityChanged(info.selector()),
            other => OpenError::Backend(TargetError::DeviceUnavailable(other.to_string())),
        })?;
        // Every admission against the device's store decides from the
        // driver's memory reconciled (I9, `seismic_vulkan` memory).
        memory.device().add_store_admission(service.facts().uuid, service.admission());
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

    /// What the backend holds for fixed buffers of `contract` beyond their
    /// allocations; `DeviceOnly`'s includes the driver's memory, which its
    /// account charges there.
    pub(crate) fn backend_held(&self, contract: seismic_topology::Contract) -> crate::devices::BackendHeld {
        let kinds = self.memory.kinds(seismic_topology::Placing::Fixed(contract));
        let driver = match contract {
            seismic_topology::Contract::DeviceOnly => self.service.driver_bytes_reconciled(),
            _ => 0,
        };
        crate::devices::BackendHeld {
            unallocated: self.service.unallocated_bytes(kinds),
            beyond: self.service.held_beyond_allocations(kinds) + driver,
        }
    }

    fn allocate_with(
        &self,
        memory: &Arc<Placement>,
        bytes: u64,
        alignment: u64,
        access: Option<Arc<driver::AccessGroup>>,
        make: impl FnOnce(
            &seismic_vulkan::Device,
            Option<u64>,
        ) -> Result<seismic_vulkan::Buffer, ExecutionError>,
    ) -> Result<Arc<Allocation>, ExecutionError> {
        let limits = self.allocation_limits();
        // The device memory this allocation may add: what the ledgers let
        // the backend hold beyond what it holds now.
        let room = Some(memory.physical().available());
        let mut reservation = driver::reserve(memory, bytes)?;
        driver::allocate_reserved_in(limits, bytes, alignment, &mut reservation, access, || {
            Ok(Box::new(VulkanStorage {
                service: self.service.clone(),
                buffer: Arc::new(make(&self.service, room)?),
            }))
        })
    }

    fn allocation_limits(&self) -> AllocationLimits {
        AllocationLimits {
            max_allocation_bytes: self.service.facts().limits.max_allocation_bytes,
            max_allocation_alignment: MAX_ALLOCATION_ALIGNMENT,
        }
    }

    /// Zero-filled storage of a reserved tensor of `reservation`'s records,
    /// its leading `committed` bytes backed in chunks, each its own memory
    /// object, reached through its record table
    /// (`seismic_vulkan::reservation`), and the resizer that resizes it in
    /// place: charged as every backend's reservation is
    /// ([`driver::allocate_reservation`]).
    pub(crate) fn allocate_reservation(
        &self,
        reservation: driver::ReservedRange,
        committed: u64,
        alignment: u64,
    ) -> Result<driver::Resizer, ExecutionError> {
        driver::allocate_reservation(self.allocation_limits(), &self.memory, reservation, committed, alignment, || {
            let (buffer, resizer) = self.service.allocate_reserved(
                self.memory.kinds(seismic_topology::Placing::Reserved(reservation.contract)),
                committed,
                reservation.bytes,
                reservation.span.granule,
                reservation.record.get(),
            )?;
            let table = buffer.table();
            let bytes = table.bytes();
            let table: Box<dyn Storage> = Box::new(VulkanStorage {
                service: self.service.clone(),
                buffer: table,
            });
            let storage: Box<dyn Storage> = Box::new(VulkanReserved(buffer));
            Ok((storage, super::ReservedResizer::Vulkan(resizer), Some((bytes, table))))
        })
    }

    pub(crate) fn allocate(
        &self,
        bytes: u64,
        alignment: u64,
    ) -> Result<Arc<Allocation>, ExecutionError> {
        self.allocate_in(bytes, alignment, None)
    }

    pub(crate) fn allocate_in(
        &self,
        bytes: u64,
        alignment: u64,
        access: Option<Arc<driver::AccessGroup>>,
    ) -> Result<Arc<Allocation>, ExecutionError> {
        let kinds = self.memory.kinds(seismic_topology::Placing::Fixed(seismic_topology::Contract::DeviceOnly));
        self.allocate_with(self.memory.device(), bytes, alignment, access, |service, room| {
            service.allocate_kinds(kinds, bytes, alignment, room)
        })
    }

    /// A fixed buffer of a mapped `contract`, in the kinds its placement
    /// names (`OpenedKind::allocate_mapped`).
    pub(crate) fn allocate_mapped(
        &self,
        contract: seismic_topology::Contract,
        bytes: u64,
        alignment: u64,
    ) -> Result<Arc<Allocation>, ExecutionError> {
        let placing = seismic_topology::Placing::Fixed(contract);
        let kinds = self.memory.kinds(placing);
        self.allocate_with(self.memory.placement(placing), bytes, alignment, None, |service, room| {
            service.allocate_kinds(kinds, bytes, alignment, room)
        })
    }
}

/// Storage of one buffer: an allocation's own, or the record table a
/// reservation keeps and binds through ([`VulkanReserved`]), shared with it.
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

/// Vulkan's reserved storage: chunks device work reaches only through
/// record tables, which the host reads and writes chunk by chunk, in place
/// where they are imported host pages. It is no [`VulkanStorage`], so no
/// binding path can bind it.
struct VulkanReserved(seismic_vulkan::ReservedBuffer);

impl Storage for VulkanReserved {
    fn read(&self, offset: u64, into: &mut [u8]) -> Result<(), ExecutionError> {
        self.0.read(offset, into)
    }
    fn write(&self, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError> {
        self.0.write(offset, bytes)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn host_address(&self) -> Option<std::ptr::NonNull<u8>> {
        self.0.host_base()
    }
}

/// The device address of byte `offset` of a Vulkan allocation (of reserved
/// storage, a byte of a backed chunk).
pub(crate) fn vulkan_address(allocation: &Allocation, offset: u64) -> u64 {
    match allocation.storage().as_any().downcast_ref::<VulkanReserved>() {
        Some(VulkanReserved(reserved)) => reserved.address(offset),
        None => vulkan_buffer(allocation).address() + offset,
    }
}

/// Byte `offset` of a Vulkan allocation as a transfer reaches it: its
/// buffer's, or a reservation's committed bytes chunk by chunk.
pub(crate) fn vulkan_transfer_end(allocation: &Allocation, offset: u64) -> seismic_vulkan::transfer::TransferEnd<'_> {
    match allocation.storage().as_any().downcast_ref::<VulkanReserved>() {
        Some(VulkanReserved(reserved)) => seismic_vulkan::transfer::TransferEnd::Reserved(reserved, offset),
        None => seismic_vulkan::transfer::TransferEnd::Buffer(vulkan_buffer(allocation), offset),
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

#[cfg(test)]
mod tests {
    use crate::backends::OpenedKind;
    use seismic_lang::registry::BackendName;
    use seismic_vulkan::direct::{DirectBatch, HostTimeline};

    /// An admission against the device's store, through the runtime's real
    /// ledger, decides from the driver's memory as it is at the decision:
    /// after submissions a host gate held and its raising released (the
    /// parks keeper's path), which reconcile nothing, a new block's
    /// admission leaves the charge at least what the driver held just
    /// before it.
    #[test]
    fn an_admission_sees_the_drivers_growth_since_any_reading() {
        let Ok(devices) = crate::devices::Devices::discover(&[]) else {
            return;
        };
        let Ok(device) = devices.open_backend(BackendName::Vulkan) else {
            eprintln!("no Vulkan device");
            return;
        };
        let OpenedKind::Vulkan(opened) = &device.kind else {
            unreachable!("a Vulkan device opens the Vulkan backend")
        };
        let service = opened.service();
        let gate = HostTimeline::new(service).unwrap();
        let submissions = (1..=256u64)
            .map(|value| {
                let mut batch = DirectBatch::new(service).unwrap();
                batch.wait_timeline(&gate, value);
                batch.commit().unwrap()
            })
            .collect::<Vec<_>>();
        gate.signal(256).unwrap();
        for submission in &submissions {
            submission.finish().unwrap();
        }
        let held = service.driver_held();
        let f32 = seismic_lang::registry::representation("f32").expect("f32 is registered");
        let _block = crate::api::tensor::TensorInner::zeros(&device, f32, &[64 << 20]).expect("a 256 MiB tensor allocates");
        assert!(service.driver_charged() >= held, "the admission missed the driver's growth");
    }
}

/// The accounts a test opens a device of `facts` with: a placement on a
/// store of its own, unbounded, for every kind.
#[cfg(test)]
pub(crate) fn test_accounts(facts: &seismic_vulkan::Facts) -> seismic_vulkan::MemoryAccounts {
    let account = || -> Arc<dyn seismic_compiler::executable::MemoryAccount> {
        Arc::new(crate::memory::MemoryAccount(Placement::on(crate::memory::StoreLedger::new())))
    };
    let kinds = (0..facts.memory.types.len() as u32)
        .chain([seismic_vulkan::topology::IMPORTED_KIND])
        .map(|kind| seismic_vulkan::AllocationAccount { kind, account: account() })
        .collect();
    seismic_vulkan::MemoryAccounts { host: account(), kinds, driver: account() }
}
