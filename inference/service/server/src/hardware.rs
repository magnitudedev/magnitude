//! `GET /api/v1/hardware` (integration spec §5.4, §9.4; memory-reserves spec §5.5): the service
//! process's own Seismic topology (its physical devices with their executors and the serving
//! device, the stores, and the access and transfer paths with their grants)
//! and its status as the engine's fit reads them, merged with the resident worker's readings. The
//! service opens no executor, so only a loaded worker reads a device store's live free bytes.
//! No capacity, placement or backend rule is recreated here.

use std::sync::Arc;
use std::time::UNIX_EPOCH;

use futures_util::future::BoxFuture;
use magnitude_engine::worker::protocol::MemoryObservation;
use magnitude_executor::platform::Fit;
use magnitude_service_contracts::{
    ExecutionBackend, HardwareAccessPath, HardwareBackendDiagnostic, HardwareDeviceId,
    HardwareDeviceKind, HardwareDeviceMemory, HardwareDiagnosticSource, HardwareExecutor,
    HardwareExecutorAvailability, HardwareFormFactor, HardwareGrant, HardwareHostRam, HardwareMachineIdentity,
    HardwarePhysicalDevice, HardwarePhysicalDeviceId, HardwareProvider, HardwareReportedBy,
    HardwareSnapshot, HardwareStore, HardwareTransferPath, HardwareVolume, InventoryError,
};
use seismic::{
    Availability, BackendName, Contract, DeviceMemory, DeviceTopology, DiagnosticSource,
    FormFactor, Inventory, PhysicalIdentity, ReportedBy, RuntimeStatus, Status, StoreKind,
};
use sha2::{Digest, Sha256};

use crate::assessment::environment::ServingDevice;
use crate::build_identity;
use crate::model_store::ServedVolumes;
use crate::memory_domains::{store_domain_id, store_identity_id};
use crate::residency::controller::ModelInstances;

/// The engine backend a Seismic backend names.
pub fn execution_backend(backend: BackendName) -> ExecutionBackend {
    match backend {
        BackendName::Cpu => ExecutionBackend::Cpu,
        BackendName::Metal => ExecutionBackend::Metal,
        BackendName::Cuda => ExecutionBackend::Cuda,
        BackendName::Vulkan => ExecutionBackend::Vulkan,
    }
}

/// The resident worker's memory observations, when a model is loaded.
pub trait ResidentMemory: Send + Sync + 'static {
    fn observe(&self) -> BoxFuture<'_, Option<MemoryObservation>>;
}

impl ResidentMemory for ModelInstances {
    fn observe(&self) -> BoxFuture<'_, Option<MemoryObservation>> {
        Box::pin(self.resident_memory())
    }
}

/// The hardware endpoint's provider.
pub struct HardwareInventory {
    inventory: Arc<Inventory>,
    volumes: Arc<ServedVolumes>,
    serving: ServingDevice,
    resident: Arc<dyn ResidentMemory>,
}

impl HardwareInventory {
    /// Every snapshot reads `inventory`'s current topology, a fresh status sample and the
    /// resident worker's fresh observations, and reports `serving` as the device the service
    /// serves on.
    pub fn new(
        inventory: Arc<Inventory>,
        volumes: ServedVolumes,
        serving: ServingDevice,
        resident: Arc<dyn ResidentMemory>,
    ) -> Result<Self, String> {
        Ok(Self {
            inventory,
            volumes: Arc::new(volumes),
            serving,
            resident,
        })
    }

    /// The snapshot of `inventory`'s current topology from one sample, which
    /// needs no opened executor, with a resident worker's device memory
    /// readings.
    fn snapshot_with(
        inventory: &Inventory,
        volumes: &ServedVolumes,
        serving: ServingDevice,
        resident: Option<&MemoryObservation>,
    ) -> Result<HardwareSnapshot, InventoryError> {
        let topology = inventory.topology();
        let every = topology
            .executors()
            .iter()
            .map(|executor| executor.id())
            .collect::<Vec<_>>();
        let status = Status {
            sample: inventory.sample(&topology, &every),
            runtime: RuntimeStatus::default(),
        };
        let mut snapshot = snapshot(&topology, &status, volumes, serving)?;
        if let Some(resident) = resident {
            merge_resident(&mut snapshot.device_memory, resident);
        }
        Ok(snapshot)
    }
}

/// A resident worker's readings of device memory, which only an opened
/// executor knows; the OS reports host RAM and volumes to the sample.
fn merge_resident(device_memory: &mut [HardwareDeviceMemory], resident: &MemoryObservation) {
    for reading in &resident.stores {
        let id = store_identity_id(&reading.store);
        if let Some(memory) = device_memory
            .iter_mut()
            .find(|memory| memory.store.id == id)
        {
            memory.current_free_bytes = Some(reading.headroom_bytes);
        }
    }
}

/// The wire name of the interface a value was reported by.
pub fn reported_by(by: ReportedBy) -> HardwareReportedBy {
    match by {
        ReportedBy::LinuxMemTotal | ReportedBy::LinuxSysinfo => HardwareReportedBy::LinuxMemTotal,
        ReportedBy::MacOsMemsize => HardwareReportedBy::MacOsMemsize,
        ReportedBy::WindowsTotalPhys => HardwareReportedBy::WindowsTotalPhys,
        ReportedBy::CpuAddressSpace => HardwareReportedBy::CpuAddressSpace,
        ReportedBy::MetalMaxBufferLength => HardwareReportedBy::MetalMaxBufferLength,
        ReportedBy::MetalRecommendedWorkingSet => HardwareReportedBy::MetalRecommendedWorkingSet,
        ReportedBy::MetalWorkingSetUnallocated => HardwareReportedBy::MetalWorkingSetUnallocated,
        ReportedBy::CudaTotalMemory => HardwareReportedBy::CudaTotalMemory,
        ReportedBy::CudaUsableMemory => HardwareReportedBy::CudaUsableMemory,
        ReportedBy::CudaFree => HardwareReportedBy::CudaFree,
        ReportedBy::VulkanHeap => HardwareReportedBy::VulkanHeap,
        ReportedBy::VulkanHeapBudget => HardwareReportedBy::VulkanHeapBudget,
        ReportedBy::VulkanHeaps => HardwareReportedBy::VulkanHeaps,
        ReportedBy::VulkanHeapsBeyondDedicated => HardwareReportedBy::VulkanHeapsBeyondDedicated,
        ReportedBy::VulkanMaxAllocation => HardwareReportedBy::VulkanMaxAllocation,
        ReportedBy::CudaTotalBeyondDedicated => HardwareReportedBy::CudaTotalBeyondDedicated,
        ReportedBy::DxgiDedicatedMemory => HardwareReportedBy::DxgiDedicatedMemory,
        ReportedBy::FilesystemSize => HardwareReportedBy::FilesystemSize,
        ReportedBy::FilesystemAvailable => HardwareReportedBy::FilesystemAvailable,
    }
}

impl HardwareProvider for HardwareInventory {
    fn snapshot(&self) -> BoxFuture<'_, Result<HardwareSnapshot, InventoryError>> {
        let inventory = Arc::clone(&self.inventory);
        let volumes = Arc::clone(&self.volumes);
        let serving = self.serving;
        Box::pin(async move {
            let resident = self.resident.observe().await;
            magnitude_service_utils::blocking::run(move || {
                Self::snapshot_with(&inventory, &volumes, serving, resident.as_ref())
            })
            .await
        })
    }
}

fn observation_error(error: impl std::fmt::Display) -> InventoryError {
    InventoryError::ModelOperation {
        code: "memory_observation_unavailable".to_owned(),
        message: error.to_string(),
        retryable: true,
    }
}

fn snapshot(
    topology: &DeviceTopology,
    status: &Status,
    served: &ServedVolumes,
    serving: ServingDevice,
) -> Result<HardwareSnapshot, InventoryError> {
    let fit = Fit::new(topology, status).with_volumes(&served.reserves);
    let path_index = |id| {
        let index = topology
            .access_paths()
            .iter()
            .position(|path| path.id() == id)
            .expect("placement names a topology path");
        u32::try_from(index).expect("topology path index fits u32")
    };
    let executor_id = |id| {
        let executor = topology
            .executor(id)
            .expect("path names a topology executor");
        HardwareDeviceId::new(executor.selector().to_string())
    };
    let placement = |executor: &seismic::Executor, kind| -> Vec<u32> {
        executor
            .paths(kind)
            .iter()
            .copied()
            .map(path_index)
            .collect()
    };
    let hardware_executor = |executor: &seismic::Executor| -> Result<_, InventoryError> {
        let availability = match executor.availability() {
            Availability::Available => HardwareExecutorAvailability::Available {
                stable_capacity_bytes: fit
                    .placement(executor, Contract::DeviceOnly)
                    .map_err(observation_error)?
                    .stable_bytes,
            },
            Availability::Unavailable { reason } => HardwareExecutorAvailability::Unavailable {
                reason: reason.clone(),
            },
        };
        Ok(HardwareExecutor {
            id: executor_id(executor.id()),
            backend: execution_backend(executor.backend()),
            name: executor.name().to_owned(),
            description: format!("{} ({})", executor.name(), executor.backend().as_str()),
            device_placement: placement(executor, Contract::DeviceOnly),
            upload_placement: placement(executor, Contract::Upload),
            availability,
        })
    };
    let physical_devices = topology
        .physical_devices()
        .iter()
        .map(|physical| {
            let executors = topology
                .executors()
                .iter()
                .filter(|executor| executor.physical() == physical.id())
                .collect::<Vec<_>>();
            // The device's kind is what discovery decided of it once, from all its backends'
            // reports: the host is the CPU, and a GPU is integrated when its memory is the
            // host's.
            let kind = match (physical.identity(), physical.memory()) {
                (PhysicalIdentity::Host { .. }, _) => HardwareDeviceKind::Cpu,
                (_, DeviceMemory::Host) => HardwareDeviceKind::IntegratedGpu,
                (_, DeviceMemory::Own) => HardwareDeviceKind::Gpu,
                (_, DeviceMemory::Undecided) => HardwareDeviceKind::Unknown,
            };
            Ok(HardwarePhysicalDevice {
                id: HardwarePhysicalDeviceId::new(physical.identity().to_string()),
                kind,
                executors: executors
                    .into_iter()
                    .map(&hardware_executor)
                    .collect::<Result<Vec<_>, InventoryError>>()?,
            })
        })
        .collect::<Result<Vec<_>, InventoryError>>()?;
    let store = |store: &seismic::Store| -> Result<(HardwareStore, u64), InventoryError> {
        let reading = fit.store(store.id()).map_err(observation_error)?;
        let hardware = HardwareStore {
            id: store_domain_id(topology, store.id()),
            total_capacity_bytes: store.capacity().value,
            capacity_basis: reported_by(store.capacity().by),
            allocation_capacity_bytes: reading.capacity_bytes,
        };
        Ok((hardware, reading.headroom_bytes))
    };
    let host_ram = {
        let (store, current_free_bytes) = store(topology.host_ram())?;
        HardwareHostRam { store, current_free_bytes }
    };
    let mut device_memory = Vec::new();
    let mut volumes = Vec::new();
    for each in topology.stores() {
        match each.kind() {
            StoreKind::HostRam => {}
            // Its headroom is what opened executors leave, which only a
            // resident worker reads (`merge_resident`).
            StoreKind::DeviceMemory(device) => device_memory.push(HardwareDeviceMemory {
                physical_device_id: HardwarePhysicalDeviceId::new(
                    topology
                        .physical_device(*device)
                        .expect("a device memory store's owner belongs to its topology")
                        .identity()
                        .to_string(),
                ),
                store: store(each)?.0,
                current_free_bytes: None,
            }),
            StoreKind::Volume(volume) => {
                let (store, current_free_bytes) = store(each)?;
                volumes.push(HardwareVolume {
                    store,
                    current_free_bytes,
                    roles: served.roles(volume),
                });
            }
        }
    }
    let access_paths = topology
        .access_paths()
        .iter()
        .map(|path| {
            let remaining = status
                .grant_left(path)
                .transpose()
                .map_err(observation_error)?;
            Ok(HardwareAccessPath {
                id: path_index(path.id()),
                executor: executor_id(path.executor()),
                store: store_domain_id(topology, path.store()),
                grant: path.grant().map(|grant| HardwareGrant {
                    bytes: grant.bytes.value,
                    basis: reported_by(grant.bytes.by),
                    remaining_bytes: remaining.map(|left| left.value),
                }),
            })
        })
        .collect::<Result<Vec<_>, InventoryError>>()?;
    let transfer_paths = topology
        .transfer_paths()
        .iter()
        .map(|path| HardwareTransferPath {
            from: store_domain_id(topology, path.from()),
            to: store_domain_id(topology, path.to()),
            executor: executor_id(path.by()),
        })
        .collect();
    let backend_diagnostics = topology
        .diagnostics()
        .iter()
        .map(|diagnostic| HardwareBackendDiagnostic {
            source: match diagnostic.source {
                DiagnosticSource::Backend(backend) => HardwareDiagnosticSource::Backend {
                    backend: execution_backend(backend),
                },
                DiagnosticSource::Adapters => HardwareDiagnosticSource::Adapters,
                DiagnosticSource::Host => HardwareDiagnosticSource::Host,
            },
            message: diagnostic.message.clone(),
        })
        .collect();
    let internal = |error: &dyn std::fmt::Display| InventoryError::Internal(error.to_string());
    let captured_at = status
        .sample
        .taken
        .duration_since(UNIX_EPOCH)
        .map_err(|error| internal(&error))?
        .as_millis();
    let host = topology.host();
    let cpu_pool_cores = host
        .processors
        .cores()
        .map_err(InventoryError::Internal)?
        .get();
    Ok(HardwareSnapshot {
        captured_at: u64::try_from(captured_at).map_err(|error| internal(&error))?,
        platform: std::env::consts::OS.to_owned(),
        architecture: std::env::consts::ARCH.to_owned(),
        system_product_name: host.system.name.clone(),
        machine_identity: host
            .system
            .identity
            .as_ref()
            .map(|identity| HardwareMachineIdentity {
                manufacturer: identity.manufacturer.clone(),
                model: identity.model.clone(),
                family: identity.family.clone(),
                version: identity.version.clone(),
            }),
        form_factor: match host.system.form_factor {
            FormFactor::Portable => HardwareFormFactor::Portable,
            FormFactor::Desktop => HardwareFormFactor::Desktop,
            FormFactor::AllInOne => HardwareFormFactor::AllInOne,
            FormFactor::MiniPc => HardwareFormFactor::MiniPc,
            FormFactor::Server => HardwareFormFactor::Server,
            FormFactor::Unknown => HardwareFormFactor::Unknown,
        },
        cpu_model: (!host.cpu_model.is_empty()).then(|| host.cpu_model.clone()),
        chip_cores: host.processors.chip_cores(),
        logical_cpus: host.processors.logical_cpus(),
        cpu_pool_cores,
        native_build: build_identity::native_build(),
        backend_diagnostics,
        topology_fingerprint: format!("{:x}", Sha256::digest(topology.normalized().as_bytes())),
        physical_devices,
        selected_executor: HardwareDeviceId::new(serving.selector().to_string()),
        host_ram,
        device_memory,
        volumes,
        access_paths,
        transfer_paths,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_executor::platform::VolumeReserves;
    use magnitude_service_contracts::HardwareVolumeRole;

    #[test]
    fn resident_readings_update_only_their_physical_store() {
        use magnitude_engine::census::{AllocationCensus, ServingCensus};
        use magnitude_engine::worker::protocol::StoreHeadroom;
        use seismic::StoreIdentity;
        let gpu = StoreIdentity::DeviceMemory(PhysicalIdentity::Uuid([1; 16]));
        let other = StoreIdentity::DeviceMemory(PhysicalIdentity::Uuid([2; 16]));
        let memory = |identity: &StoreIdentity| HardwareDeviceMemory {
            physical_device_id: HardwarePhysicalDeviceId::new(identity.to_string()),
            store: HardwareStore {
                id: store_identity_id(identity),
                total_capacity_bytes: 1024,
                capacity_basis: HardwareReportedBy::CudaTotalMemory,
                allocation_capacity_bytes: 1024,
            },
            current_free_bytes: None,
        };
        let mut device_memory = vec![memory(&gpu), memory(&other)];
        let observation = MemoryObservation {
            census: AllocationCensus {
                stores: Vec::new(),
                serving: ServingCensus {
                    group_places: std::num::NonZeroUsize::MIN,
                    group_capacity: std::num::NonZeroUsize::MIN,
                    experts: None,
                },
            },
            stores: vec![
                StoreHeadroom {
                    store: StoreIdentity::HostRam,
                    headroom_bytes: 3,
                },
                StoreHeadroom {
                    store: gpu,
                    headroom_bytes: 7,
                },
            ],
        };
        merge_resident(&mut device_memory, &observation);
        assert_eq!(device_memory[0].current_free_bytes, Some(7));
        assert_eq!(device_memory[1].current_free_bytes, None);
    }

    /// A volume the service keeps a path on carries that path's role, and only that volume.
    #[test]
    fn served_volumes_carry_their_roles() {
        let directory = std::env::temp_dir().join(format!("served-volume-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let inventory = Inventory::discover(&[directory.clone()], seismic::ProgramStore::user()).unwrap();
        let mut served = ServedVolumes::new(Arc::new(VolumeReserves::new(None)));
        served.tag(&directory, HardwareVolumeRole::ModelStore).unwrap();
        served.tag(&directory, HardwareVolumeRole::KernelCache).unwrap();
        served.tag(&directory, HardwareVolumeRole::ModelStore).unwrap();
        let observed = HardwareInventory::snapshot_with(
            &inventory,
            &served,
            ServingDevice::of(seismic::ExecutorSelector::HostCpu),
            None,
        )
        .unwrap();
        std::fs::remove_dir_all(&directory).unwrap();
        assert_eq!(observed.volumes.len(), 1);
        assert_eq!(
            observed.volumes[0].roles,
            vec![HardwareVolumeRole::ModelStore, HardwareVolumeRole::KernelCache]
        );
    }

    #[test]
    fn snapshot_keeps_physical_identity_and_all_placement_references() {
        let inventory = Inventory::discover(&[], seismic::ProgramStore::user()).unwrap();
        let topology = inventory.topology();
        let serving = ServingDevice::of(seismic::ExecutorSelector::HostCpu);
        let observed = HardwareInventory::snapshot_with(
            &inventory,
            &ServedVolumes::new(Arc::new(VolumeReserves::new(None))),
            serving,
            None,
        )
        .unwrap();
        assert_eq!(
            observed.physical_devices.len(),
            topology.physical_devices().len()
        );
        assert_eq!(
            observed.device_memory.len() + observed.volumes.len() + 1,
            topology.stores().len()
        );
        let stores = std::iter::once(&observed.host_ram.store)
            .chain(observed.device_memory.iter().map(|memory| &memory.store))
            .chain(observed.volumes.iter().map(|volume| &volume.store))
            .collect::<Vec<_>>();
        for physical in &observed.physical_devices {
            for executor in &physical.executors {
                for path in executor
                    .device_placement
                    .iter()
                    .chain(&executor.upload_placement)
                {
                    let path = &observed.access_paths[*path as usize];
                    assert_eq!(path.executor, executor.id);
                    assert!(stores.iter().any(|s| s.id == path.store));
                }
            }
        }
        // The report names the serving device it is given, whatever the automatic selection.
        assert_eq!(observed.selected_executor.as_str(), "host-cpu");
        assert!(
            observed
                .physical_devices
                .iter()
                .flat_map(|physical| &physical.executors)
                .any(|executor| executor.id == observed.selected_executor)
        );
    }
}
