//! `GET /api/v1/hardware` (integration spec §5.4, §9.4; memory-reserves spec §5.5): the service
//! process's own Seismic topology (its physical devices with their executors and the serving
//! device, the stores, and the access and transfer paths with their grants)
//! and its status under the service's reserve
//! policy, as the engine's fit reads them, merged with the resident worker's readings. The
//! service opens no executor, so only a loaded worker reads a device store's live free bytes.
//! No capacity, placement or backend rule is recreated here.

use std::sync::Arc;
use std::time::UNIX_EPOCH;

use futures_util::future::BoxFuture;
use magnitude_engine::worker::protocol::MemoryObservation;
use magnitude_executor::platform::{Fit, MemoryReserves};
use magnitude_service_contracts::{
    ExecutionBackend, HardwareAccessPath, HardwareBackendDiagnostic, HardwareDeviceId,
    HardwareDeviceKind, HardwareDeviceMemory, HardwareDiagnosticSource, HardwareExecutor,
    HardwareExecutorAvailability, HardwareGrant, HardwareHostRam, HardwareObservedStore,
    HardwarePhysicalDevice, HardwarePhysicalDeviceId, HardwareProvider, HardwareReportedBy,
    HardwareSnapshot, HardwareStore, HardwareTransferPath, InventoryError,
};
use seismic::{
    Availability, BackendName, DeviceMemory, DeviceTopology, DiagnosticSource, Inventory,
    PhysicalIdentity, ReportedBy, RuntimeStatus, Status, StorageKind, StoreKind,
};
use sha2::{Digest, Sha256};

use crate::assessment::environment::ServingDevice;
use crate::build_identity;
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

/// Descriptive facts about the machine that do not change while the service runs.
struct MachineDescription {
    system_product_name: Option<String>,
    cpu_model: Option<String>,
    physical_cores: Option<usize>,
    logical_cores: usize,
}

impl MachineDescription {
    fn discover() -> Self {
        let system = sysinfo::System::new_with_specifics(
            sysinfo::RefreshKind::nothing().with_cpu(sysinfo::CpuRefreshKind::nothing()),
        );
        Self {
            system_product_name: system_product_name(),
            cpu_model: system
                .cpus()
                .first()
                .map(|cpu| cpu.brand().trim().to_owned())
                .filter(|brand| !brand.is_empty()),
            physical_cores: physical_cores(),
            logical_cores: std::thread::available_parallelism().map_or(1, |cores| cores.get()),
        }
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
    reserves: MemoryReserves,
    serving: ServingDevice,
    machine: Arc<MachineDescription>,
    resident: Arc<dyn ResidentMemory>,
}

impl HardwareInventory {
    /// Describe the machine once; every snapshot then reads `inventory`'s current topology, a
    /// fresh status sample and the resident worker's fresh observations, and reports `serving`
    /// as the device the service serves on.
    pub fn new(
        inventory: Arc<Inventory>,
        reserves: MemoryReserves,
        serving: ServingDevice,
        resident: Arc<dyn ResidentMemory>,
    ) -> Self {
        Self {
            inventory,
            reserves,
            serving,
            machine: Arc::new(MachineDescription::discover()),
            resident,
        }
    }

    /// The snapshot of `inventory`'s current topology from one sample, which
    /// needs no opened executor, with a resident worker's device memory
    /// readings.
    fn snapshot_with(
        inventory: &Inventory,
        reserves: &MemoryReserves,
        serving: ServingDevice,
        machine: &MachineDescription,
        resident: Option<&MemoryObservation>,
    ) -> Result<HardwareSnapshot, InventoryError> {
        let topology = inventory.topology();
        let status = Status {
            sample: inventory.sample(&topology),
            runtime: RuntimeStatus::default(),
        };
        let mut snapshot = snapshot(&topology, &status, reserves, serving, machine)?;
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
        ReportedBy::LinuxMemTotal => HardwareReportedBy::LinuxMemTotal,
        ReportedBy::MacOsMemsize => HardwareReportedBy::MacOsMemsize,
        ReportedBy::WindowsTotalPhys => HardwareReportedBy::WindowsTotalPhys,
        ReportedBy::LinuxMemAvailable => HardwareReportedBy::LinuxMemAvailable,
        ReportedBy::MachMovablePages => HardwareReportedBy::MachMovablePages,
        ReportedBy::WindowsPhysicalAndCommit => HardwareReportedBy::WindowsPhysicalAndCommit,
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
        ReportedBy::MetalWorkingSetBeyondDedicated => {
            HardwareReportedBy::MetalWorkingSetBeyondDedicated
        }
        ReportedBy::DxgiDedicatedMemory => HardwareReportedBy::DxgiDedicatedMemory,
        ReportedBy::LinuxSysfsCache => HardwareReportedBy::LinuxSysfsCache,
        ReportedBy::MacOsSysctlCache => HardwareReportedBy::MacOsSysctlCache,
        ReportedBy::WindowsProcessorCache => HardwareReportedBy::WindowsProcessorCache,
        ReportedBy::CudaL2Cache => HardwareReportedBy::CudaL2Cache,
        ReportedBy::FilesystemSize => HardwareReportedBy::FilesystemSize,
        ReportedBy::FilesystemAvailable => HardwareReportedBy::FilesystemAvailable,
    }
}

impl HardwareProvider for HardwareInventory {
    fn snapshot(&self) -> BoxFuture<'_, Result<HardwareSnapshot, InventoryError>> {
        let inventory = Arc::clone(&self.inventory);
        let reserves = self.reserves;
        let serving = self.serving;
        let machine = Arc::clone(&self.machine);
        Box::pin(async move {
            let resident = self.resident.observe().await;
            crate::spawn_blocking_traced(move || {
                Self::snapshot_with(&inventory, &reserves, serving, &machine, resident.as_ref())
            })
            .await
            .map_err(|error| {
                InventoryError::Internal(format!("hardware snapshot task failed: {error}"))
            })?
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
    reserves: &MemoryReserves,
    serving: ServingDevice,
    machine: &MachineDescription,
) -> Result<HardwareSnapshot, InventoryError> {
    let fit = Fit::new(topology, status, reserves);
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
            .placement(kind)
            .iter()
            .copied()
            .map(path_index)
            .collect()
    };
    let hardware_executor = |executor: &seismic::Executor| -> Result<_, InventoryError> {
        let availability = match executor.availability() {
            Availability::Available => HardwareExecutorAvailability::Available {
                stable_capacity_bytes: fit
                    .placement(executor, StorageKind::Device)
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
            device_placement: placement(executor, StorageKind::Device),
            upload_placement: placement(executor, StorageKind::Upload),
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
            stable_capacity_bytes: reading.stable_bytes,
            allocation_capacity_bytes: reading.capacity_bytes,
            planning_reserve_bytes: reading.thresholds.planning_bytes,
            emergency_reserve_bytes: reading.thresholds.emergency_bytes,
        };
        Ok((hardware, reading.headroom_bytes))
    };
    let host_ram = {
        let (store, current_free_bytes) = store(topology.host_ram())?;
        let host = status.sample.host_ram.as_ref().map_err(observation_error)?;
        HardwareHostRam {
            store,
            available_bytes: host.headroom.value,
            current_free_bytes,
        }
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
            StoreKind::Volume(_) => {
                let (store, current_free_bytes) = store(each)?;
                volumes.push(HardwareObservedStore {
                    store,
                    current_free_bytes,
                });
            }
        }
    }
    let access_paths = topology
        .access_paths()
        .iter()
        .map(|path| {
            let remaining = status
                .grant_left(path.id())
                .transpose()
                .map_err(observation_error)?;
            Ok(HardwareAccessPath {
                id: path_index(path.id()),
                executor: executor_id(path.executor()),
                store: store_domain_id(topology, path.store()),
                grant: path.grant().map(|grant| HardwareGrant {
                    bytes: grant.value,
                    basis: reported_by(grant.by),
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
    Ok(HardwareSnapshot {
        captured_at: u64::try_from(captured_at).map_err(|error| internal(&error))?,
        platform: std::env::consts::OS.to_owned(),
        architecture: std::env::consts::ARCH.to_owned(),
        system_product_name: machine.system_product_name.clone(),
        cpu_model: machine.cpu_model.clone(),
        physical_cores: machine.physical_cores,
        logical_cores: machine.logical_cores,
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

fn system_product_name() -> Option<String> {
    match std::env::consts::OS {
        "linux" => [
            "/sys/devices/virtual/dmi/id/product_name",
            "/sys/class/dmi/id/product_name",
            "/proc/device-tree/model",
        ]
        .into_iter()
        .find_map(|path| {
            std::fs::read(path)
                .ok()
                .and_then(|value| normalize_product_name(&value))
        }),
        "macos" => std::process::Command::new("/usr/sbin/system_profiler")
            .args(["SPHardwareDataType", "-json"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| serde_json::from_slice::<serde_json::Value>(&output.stdout).ok())
            .and_then(|document| {
                document
                    .get("SPHardwareDataType")?
                    .as_array()?
                    .first()?
                    .get("machine_name")?
                    .as_str()
                    .and_then(|name| normalize_product_name(name.as_bytes()))
            }),
        _ => None,
    }
}

fn normalize_product_name(value: &[u8]) -> Option<String> {
    let name = String::from_utf8_lossy(value)
        .replace('_', " ")
        .trim_matches(|character: char| character == '\0' || character.is_whitespace())
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let placeholder = [
        "default string",
        "not specified",
        "system product name",
        "to be filled by o.e.m.",
        "unknown",
    ]
    .iter()
    .any(|placeholder| name.eq_ignore_ascii_case(placeholder));
    (!name.is_empty() && !placeholder).then_some(name)
}

#[cfg(not(target_os = "linux"))]
fn physical_cores() -> Option<usize> {
    // sysctl on macOS, GetLogicalProcessorInformationEx on Windows.
    sysinfo::System::physical_core_count().filter(|cores| *cores > 0)
}

#[cfg(target_os = "linux")]
fn physical_cores() -> Option<usize> {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| parse_linux_physical_cores(&text))
}

/// Distinct (package, core) pairs. ARM and virtualized `/proc/cpuinfo` may omit topology; their
/// logical processor records are never counted as physical cores.
#[cfg(any(target_os = "linux", test))]
fn parse_linux_physical_cores(text: &str) -> Option<usize> {
    let mut cores = std::collections::BTreeSet::new();
    for record in text.split("\n\n") {
        let mut processor = false;
        let mut package = None;
        let mut core = None;
        for line in record.lines() {
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            match key.trim() {
                "processor" => processor = true,
                "physical id" => package = Some(value.trim().parse::<u32>().ok()?),
                "core id" => core = Some(value.trim().parse::<u32>().ok()?),
                _ => {}
            }
        }
        if processor {
            cores.insert((package?, core?));
        }
    }
    (!cores.is_empty()).then_some(cores.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_cores_separate_smt_threads_and_sockets() {
        let mut text = String::new();
        for socket in 0..2 {
            for core in 0..4 {
                for thread in 0..2 {
                    text.push_str(&format!(
                        "processor : {}\nphysical id : {socket}\ncore id : {core}\n\n",
                        socket * 8 + core * 2 + thread
                    ));
                }
            }
        }
        assert_eq!(parse_linux_physical_cores(&text), Some(8));
        for incomplete in ["", "processor : 0\n", "processor : 0\ncore id : 0"] {
            assert_eq!(parse_linux_physical_cores(incomplete), None);
        }
    }

    #[test]
    fn product_name_placeholders_are_absent() {
        assert_eq!(normalize_product_name(b"Default string\0"), None);
        assert_eq!(
            normalize_product_name(b"MacBook_Pro  "),
            Some("MacBook Pro".to_owned())
        );
    }

    #[test]
    fn resident_readings_update_only_their_physical_store() {
        use magnitude_engine::census::AllocationCensus;
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
                stable_capacity_bytes: 768,
                allocation_capacity_bytes: 1024,
                planning_reserve_bytes: 256,
                emergency_reserve_bytes: 64,
            },
            current_free_bytes: None,
        };
        let mut device_memory = vec![memory(&gpu), memory(&other)];
        let observation = MemoryObservation {
            census: AllocationCensus { stores: Vec::new() },
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

    #[test]
    fn snapshot_keeps_physical_identity_and_all_placement_references() {
        let inventory = Inventory::discover(&[]).unwrap();
        let topology = inventory.topology();
        let serving = ServingDevice::of(seismic::ExecutorSelector::HostCpu);
        let observed = HardwareInventory::snapshot_with(
            &inventory,
            &MemoryReserves::standard(),
            serving,
            &MachineDescription::discover(),
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
        assert!(observed.host_ram.current_free_bytes <= observed.host_ram.available_bytes);
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
