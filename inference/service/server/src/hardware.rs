//! `GET /api/v1/hardware` (integration spec §5.4, §9.4; memory-reserves spec §5.5): the service
//! process's own Seismic topology and host memory status under the service's reserve policy,
//! merged with the resident worker's observations. The service opens no device, so only a loaded
//! worker observes a dedicated device's live free bytes.

use std::sync::Arc;
use std::time::UNIX_EPOCH;

use futures_util::future::BoxFuture;
use magnitude_engine::census::MemoryDomain;
use magnitude_engine::worker::protocol::MemoryObservation;
use magnitude_executor::platform::{MemoryReserves, unusable_reason};
use magnitude_service_contracts::{
    ExecutionBackend, HardwareDevice, HardwareDeviceId, HardwareDeviceKind,
    HardwareDeviceMemoryLimit, HardwareDeviceMemoryLimitKind, HardwareMemoryDomain,
    HardwareMemoryDomainKind, HardwareProvider, HardwareSnapshot, HardwareSystemMemory,
    InventoryError,
};
use seismic::{
    BackendName, DeviceCatalog, DeviceInfo, DeviceKind, DeviceMemory, DeviceTopology,
    HostMemoryStatus,
};
use sha2::{Digest, Sha256};

use crate::build_identity;
use crate::memory_domains::{census_domain_id, pool_domain_id};
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
    catalog: Arc<DeviceCatalog>,
    reserves: MemoryReserves,
    machine: Arc<MachineDescription>,
    resident: Arc<dyn ResidentMemory>,
}

impl HardwareInventory {
    /// Describe the machine once; every snapshot then reads the catalog's topology, a fresh host
    /// memory status and the resident worker's fresh observations.
    pub fn new(
        catalog: Arc<DeviceCatalog>,
        reserves: MemoryReserves,
        resident: Arc<dyn ResidentMemory>,
    ) -> Self {
        Self {
            catalog,
            reserves,
            machine: Arc::new(MachineDescription::discover()),
            resident,
        }
    }

    fn snapshot_with(
        catalog: &DeviceCatalog,
        reserves: &MemoryReserves,
        machine: &MachineDescription,
        resident: Option<&MemoryObservation>,
    ) -> Result<HardwareSnapshot, InventoryError> {
        let host = catalog.host_memory_status().map_err(|error| {
            InventoryError::ModelOperation {
                code: "memory_observation_unavailable".to_owned(),
                message: format!("system memory observation failed: {error}"),
                retryable: true,
            }
        })?;
        let topology = catalog.topology();
        let mut snapshot = snapshot(&topology, &host, reserves, machine);
        if let Some(resident) = resident {
            merge_resident(&mut snapshot, &topology, resident);
        }
        Ok(snapshot)
    }
}

impl HardwareProvider for HardwareInventory {
    fn snapshot(&self) -> BoxFuture<'_, Result<HardwareSnapshot, InventoryError>> {
        let catalog = Arc::clone(&self.catalog);
        let reserves = self.reserves;
        let machine = Arc::clone(&self.machine);
        Box::pin(async move {
            let resident = self.resident.observe().await;
            crate::spawn_blocking_traced(move || {
                Self::snapshot_with(&catalog, &reserves, &machine, resident.as_ref())
            })
            .await
            .map_err(|error| InventoryError::Internal(format!("hardware snapshot task failed: {error}")))?
        })
    }
}

/// Merge the resident worker's readings: a dedicated domain's live free bytes are observable
/// only by the process that has the device open. Host RAM keeps the service's own sample, taken
/// from the same Seismic source.
fn merge_resident(
    snapshot: &mut HardwareSnapshot,
    topology: &DeviceTopology,
    resident: &MemoryObservation,
) {
    for reading in &resident.domains {
        let MemoryDomain::DeviceLocal { .. } = reading.domain else {
            continue;
        };
        let id = census_domain_id(topology, &reading.domain);
        let domain = snapshot
            .memory_domains
            .iter_mut()
            .find(|domain| domain.id == id)
            .expect("the snapshot lists every dedicated pool of its topology");
        domain.current_free_bytes = Some(reading.headroom_bytes);
    }
}

fn snapshot(
    topology: &DeviceTopology,
    host: &HostMemoryStatus,
    reserves: &MemoryReserves,
    machine: &MachineDescription,
) -> HardwareSnapshot {
    let host_pool = topology.host_pool();
    let host_thresholds = reserves.for_domain(host_pool.capacity_bytes);
    let allocation_capacity_bytes = host
        .limits
        .iter()
        .map(|limit| limit.limit_bytes)
        .fold(host_pool.capacity_bytes, u64::min);
    let allocation_headroom_bytes = host
        .limits
        .iter()
        .map(|limit| limit.remaining_bytes())
        .fold(host.headroom.bytes, u64::min);
    let host_backed = |device: &DeviceInfo| match &device.memory {
        DeviceMemory::Established(memory) => memory.allocates_host_memory(),
        DeviceMemory::Unsupported { .. } => device.kind == DeviceKind::Cpu,
    };
    let working_set = topology
        .devices()
        .iter()
        .filter_map(DeviceInfo::recommended_working_set_bytes)
        .min();
    let system_stable_capacity = working_set
        .map_or(allocation_capacity_bytes, |working_set| {
            allocation_capacity_bytes.min(working_set)
        })
        .saturating_sub(host_thresholds.planning_bytes);
    let system_devices = topology
        .devices()
        .iter()
        .filter(|device| host_backed(device))
        .map(|device| {
            public_device(
                device,
                device
                    .recommended_working_set_bytes()
                    .map(|total| HardwareDeviceMemoryLimit {
                        kind: HardwareDeviceMemoryLimitKind::RecommendedWorkingSet,
                        total_bytes: total,
                        stable_bytes: total
                            .min(allocation_capacity_bytes)
                            .saturating_sub(host_thresholds.planning_bytes),
                        current_free_bytes: None,
                    }),
            )
        })
        .collect::<Vec<_>>();
    let unified = topology
        .devices()
        .iter()
        .any(|device| device.kind == DeviceKind::Gpu && host_backed(device));
    let mut memory_domains = vec![HardwareMemoryDomain {
        id: pool_domain_id(topology, host_pool.id),
        kind: if unified {
            HardwareMemoryDomainKind::UnifiedMemory
        } else {
            HardwareMemoryDomainKind::System
        },
        total_capacity_bytes: host_pool.capacity_bytes,
        stable_capacity_bytes: system_stable_capacity,
        current_free_bytes: Some(allocation_headroom_bytes),
        shares_system_memory: true,
        devices: system_devices,
    }];
    for device in topology.devices().iter().filter(|device| !host_backed(device)) {
        let (id, total_capacity_bytes) = match &device.memory {
            DeviceMemory::Established(memory) => {
                let pool = topology
                    .pool(memory.allocation_pool)
                    .expect("a device's pools belong to its topology");
                (pool_domain_id(topology, pool.id), pool.capacity_bytes)
            }
            // A device whose memory Seismic cannot normalize has no assessable capacity; its
            // unavailable reason says why.
            DeviceMemory::Unsupported { .. } => (
                magnitude_service_contracts::MemoryDomainId::new(device.selector.to_string()),
                0,
            ),
        };
        if let Some(domain) = memory_domains.iter_mut().find(|domain| domain.id == id) {
            domain.devices.push(public_device(device, None));
            continue;
        }
        memory_domains.push(HardwareMemoryDomain {
            id,
            kind: HardwareMemoryDomainKind::PhysicalDevice,
            total_capacity_bytes,
            stable_capacity_bytes: total_capacity_bytes
                .saturating_sub(reserves.for_domain(total_capacity_bytes).planning_bytes),
            current_free_bytes: None,
            shares_system_memory: false,
            devices: vec![public_device(device, None)],
        });
    }
    HardwareSnapshot {
        captured_at: host
            .sampled_at
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)),
        platform: std::env::consts::OS.to_owned(),
        architecture: std::env::consts::ARCH.to_owned(),
        system_product_name: machine.system_product_name.clone(),
        cpu_model: machine.cpu_model.clone(),
        physical_cores: machine.physical_cores,
        logical_cores: machine.logical_cores,
        system_memory: HardwareSystemMemory {
            physical_capacity_bytes: host_pool.capacity_bytes,
            physical_available_bytes: host.headroom.bytes,
            allocation_capacity_bytes,
            allocation_headroom_bytes,
            assess_reserve_bytes: host_thresholds.planning_bytes,
            abort_reserve_bytes: host_thresholds.emergency_bytes,
        },
        native_build: build_identity::native_build(),
        enabled_backends: build_identity::compiled_backends(),
        topology_fingerprint: topology_fingerprint(&memory_domains),
        memory_domains,
    }
}

fn public_device(device: &DeviceInfo, memory_limit: Option<HardwareDeviceMemoryLimit>) -> HardwareDevice {
    let backend = execution_backend(device.backend);
    HardwareDevice {
        id: HardwareDeviceId::new(device.selector.to_string()),
        backend,
        name: device.name.clone(),
        description: format!("{} ({})", device.name, device.backend.as_str()),
        kind: match (&device.kind, &device.memory) {
            (DeviceKind::Cpu, _) => HardwareDeviceKind::Cpu,
            (DeviceKind::Gpu, DeviceMemory::Established(memory))
                if memory.allocates_host_memory() =>
            {
                HardwareDeviceKind::IntegratedGpu
            }
            (DeviceKind::Gpu, _) => HardwareDeviceKind::Gpu,
        },
        memory_limit,
        unavailable_reason: unusable_reason(device).map(str::to_owned),
    }
}

/// The identity of the stable topology: domains, capacities and devices, without live values.
fn topology_fingerprint(domains: &[HardwareMemoryDomain]) -> String {
    let mut digest = Sha256::new();
    for domain in domains {
        digest.update(domain.id.as_str().as_bytes());
        digest.update(format!("{:?}", domain.kind).as_bytes());
        digest.update(domain.total_capacity_bytes.to_le_bytes());
        digest.update(domain.stable_capacity_bytes.to_le_bytes());
        for device in &domain.devices {
            digest.update(device.id.as_str().as_bytes());
            digest.update(format!("{:?}{:?}", device.backend, device.kind).as_bytes());
            if let Some(limit) = &device.memory_limit {
                digest.update(limit.total_bytes.to_le_bytes());
                digest.update(limit.stable_bytes.to_le_bytes());
            }
            digest.update(device.unavailable_reason.as_deref().unwrap_or("").as_bytes());
        }
    }
    format!("{:x}", digest.finalize())
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

    /// The live snapshot on this machine: one `system` domain, every device named by its
    /// selector, reserves from the standard policy.
    #[test]
    fn live_snapshot_names_domains_and_devices_by_selector() {
        let catalog = DeviceCatalog::discover().expect("device discovery");
        let snapshot = HardwareInventory::snapshot_with(
            &catalog,
            &MemoryReserves::standard(),
            &MachineDescription::discover(),
            None,
        )
        .expect("hardware snapshot");
        let system = &snapshot.memory_domains[0];
        assert!(system.id.is_system());
        let thresholds =
            MemoryReserves::standard().for_domain(system.total_capacity_bytes);
        assert_eq!(snapshot.system_memory.assess_reserve_bytes, thresholds.planning_bytes);
        assert_eq!(snapshot.system_memory.abort_reserve_bytes, thresholds.emergency_bytes);
        assert!(system.stable_capacity_bytes <= system.total_capacity_bytes - thresholds.planning_bytes);
        let topology = catalog.topology();
        let listed = snapshot
            .memory_domains
            .iter()
            .flat_map(|domain| &domain.devices)
            .count();
        assert_eq!(listed, topology.devices().len());
        for device in topology.devices() {
            assert!(snapshot.memory_domains.iter().flat_map(|domain| &domain.devices).any(
                |public| public.id.as_str() == device.selector.to_string()
                    && public.unavailable_reason.as_deref() == unusable_reason(device)
            ));
        }
        assert!(serde_json::to_string(&snapshot).is_ok());
    }

    /// A resident worker's readings give every dedicated domain it observes live free bytes;
    /// the system domain keeps the service's own sample.
    #[test]
    fn resident_readings_merge_into_dedicated_domains() {
        use magnitude_engine::census::AllocationCensus;
        use magnitude_engine::worker::protocol::DomainHeadroom;
        let catalog = DeviceCatalog::discover().expect("device discovery");
        let topology = catalog.topology();
        let mut snapshot = HardwareInventory::snapshot_with(
            &catalog,
            &MemoryReserves::standard(),
            &MachineDescription::discover(),
            None,
        )
        .expect("hardware snapshot");
        let system_free = snapshot.memory_domains[0].current_free_bytes;
        let dedicated = topology
            .devices()
            .iter()
            .filter(|device| {
                matches!(&device.memory, DeviceMemory::Established(memory) if !memory.allocates_host_memory())
            })
            .map(|device| MemoryDomain::DeviceLocal {
                device: device.selector,
            })
            .collect::<Vec<_>>();
        let resident = MemoryObservation {
            census: AllocationCensus { domains: Vec::new() },
            domains: dedicated
                .iter()
                .copied()
                .chain([MemoryDomain::HostRam])
                .map(|domain| DomainHeadroom {
                    domain,
                    headroom_bytes: 7,
                })
                .collect(),
        };
        merge_resident(&mut snapshot, &topology, &resident);
        assert_eq!(snapshot.memory_domains[0].current_free_bytes, system_free);
        for domain in &dedicated {
            let id = census_domain_id(&topology, domain);
            let public = snapshot
                .memory_domains
                .iter()
                .find(|public| public.id == id)
                .expect("dedicated domain");
            assert_eq!(public.current_free_bytes, Some(7));
        }
    }
}
