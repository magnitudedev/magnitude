//! Discovery (spec §2.3): what the backend crates and the OS report, and the
//! rules that turn those reports into a topology revision.
//!
//! [`gather`] reads the reports; [`build`] applies the placement and grant
//! rules to them. The rules branch only on what the reports state (whether
//! a device's memory is the host's, its heaps, the dedicated memory the OS
//! reports for its adapter, the capabilities its backend requires), never
//! on a vendor, driver or model.
//!
//! Whose memory a device's is, `build` decides once per physical device from
//! every backend's report of it ([`device_memory`]); every executor of the
//! device then places by that one decision.

use crate::host;
use crate::model::{
    AccessRoute, Availability, Builder, DeviceMemory, DeviceTopology, Diagnostic, DiagnosticSource,
    ExecutorBuilder, ExecutorKind, ExecutorReport, GrantSource, Limits, OwnPath, PhysicalDeviceId,
    PhysicalIdentity, Reported, ReportedBy, Revision, StorageKind, StoreId, StoreKind, VolumeId,
};
use crate::selector::ExecutorSelector;
use seismic_lang::registry::BackendName;
use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// Everything discovery reads, before the rules apply.
#[derive(Debug)]
pub(crate) struct Reports {
    /// Host RAM capacity.
    pub(crate) host: Reported<u64>,
    /// The OS build: the CPU's and Metal's driver.
    pub(crate) os_build: String,
    pub(crate) cpu: CpuReport,
    pub(crate) metal: Vec<MetalReport>,
    pub(crate) cuda: Vec<CudaReport>,
    pub(crate) vulkan: Vec<VulkanReport>,
    /// The OS's display adapters (DXGI); empty where the OS has no adapter
    /// interface.
    pub(crate) adapters: Vec<AdapterReport>,
    pub(crate) volumes: Vec<VolumeReport>,
    pub(crate) diagnostics: Vec<Diagnostic>,
}

#[derive(Debug)]
pub(crate) struct CpuReport {
    pub(crate) name: String,
    /// The CPU as the OS reports it: model, instruction set and caches.
    pub(crate) identity: String,
    /// Its last-level caches, summed over their instances
    /// ([`host::last_level_cache`]).
    pub(crate) cache: Option<Reported<u64>>,
    pub(crate) availability: Availability,
    pub(crate) max_allocation: u64,
}

#[derive(Debug)]
pub(crate) struct MetalReport {
    pub(crate) registry_id: u64,
    pub(crate) name: String,
    /// `MTLDevice.hasUnifiedMemory`: its memory is the host's.
    pub(crate) unified: bool,
    /// What the backend's kernels require: a compiler accepting one of the
    /// language versions it emits (`seismic_metal::profile::language`).
    pub(crate) admission: Result<(), String>,
    /// `seismic_metal::profile::forms_tensor_operations`, for a device the
    /// backend admits.
    pub(crate) tensor_operations: bool,
    pub(crate) working_set: u64,
    pub(crate) max_allocation: u64,
}

#[derive(Debug)]
pub(crate) struct CudaReport {
    pub(crate) uuid: [u8; 16],
    /// `cuDeviceGetLuid`, where the driver reports one (WDDM): what matches
    /// the device to the dedicated memory the OS reports for its adapter.
    pub(crate) luid: Option<[u8; 8]>,
    pub(crate) name: String,
    /// `cuDriverGetVersion`.
    pub(crate) driver: String,
    /// The compute-capability and driver floors of the backend's kernels.
    pub(crate) admission: Result<(), String>,
    /// `cuDeviceTotalMem`.
    pub(crate) total: u64,
    pub(crate) max_allocation: u64,
    /// `CU_DEVICE_ATTRIBUTE_INTEGRATED`, which describes the device.
    pub(crate) integrated: bool,
    /// `CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE`; none where it reads zero.
    pub(crate) l2: Option<u64>,
    pub(crate) memory: CudaMemory,
    /// The CUDA devices whose memory it reads and writes directly
    /// (`cuDeviceCanAccessPeer`).
    pub(crate) peers: Vec<[u8; 16]>,
}

/// Whose memory a CUDA device reports its memory is.
#[derive(Debug)]
pub(crate) enum CudaMemory {
    /// The host's: an integrated device (`CU_DEVICE_ATTRIBUTE_INTEGRATED`),
    /// or one whose memory is an OS NUMA node
    /// (`CU_DEVICE_ATTRIBUTE_NUMA_CONFIG`).
    Host,
    /// Its own, of which the device lets this process use `usable`.
    Own { usable: Result<u64, String> },
}

#[derive(Debug)]
pub(crate) struct VulkanReport {
    pub(crate) uuid: [u8; 16],
    /// Its device type, which describes it.
    pub(crate) kind: ExecutorKind,
    /// Whether its memory is the host's, as its device type states it
    /// (`seismic_vulkan::DeviceType::memory_is_host`); none for a type that
    /// states neither.
    pub(crate) memory_is_host: Option<bool>,
    pub(crate) name: String,
    /// The device floor: `Err` names the first missing item.
    pub(crate) floor: Result<(), String>,
    /// `VkDriverId` and the driver version.
    pub(crate) driver: String,
    pub(crate) max_allocation: u64,
    pub(crate) memory: VulkanMemory,
}

/// A Vulkan device's memory as the backend reports it.
#[derive(Debug)]
pub(crate) struct VulkanMemory {
    /// `VkPhysicalDeviceIDProperties::deviceLUID`, when valid: what matches
    /// the device to the dedicated memory the OS reports for its adapter.
    pub(crate) luid: Option<[u8; 8]>,
    pub(crate) heaps: Vec<VulkanHeap>,
    /// For each storage kind, the heaps the backend allocates it from, in
    /// order.
    pub(crate) allocates: [Vec<usize>; StorageKind::COUNT],
}

#[derive(Debug)]
pub(crate) struct VulkanHeap {
    pub(crate) bytes: u64,
    /// `VK_MEMORY_HEAP_DEVICE_LOCAL_BIT`.
    pub(crate) device_local: bool,
}

/// One display adapter as the OS reports it.
#[derive(Debug)]
pub(crate) struct AdapterReport {
    /// The adapter's LUID, in `LUID` memory layout.
    pub(crate) luid: [u8; 8],
    /// Memory the OS reports as dedicated to the adapter, which it does not
    /// manage.
    pub(crate) dedicated: u64,
}

#[derive(Debug)]
pub(crate) struct VolumeReport {
    pub(crate) path: PathBuf,
    pub(crate) id: VolumeId,
    pub(crate) size: Reported<u64>,
}

/// Why no topology could be discovered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiscoveryError {
    /// Host RAM capacity could not be established; no topology is complete
    /// without it.
    HostMemory(String),
    /// The OS build, the CPU's and Metal's driver, could not be read.
    OsBuild(String),
    /// The CPU as the OS reports it could not be read.
    CpuIdentity(String),
    /// A cache the OS reports for the CPU, part of its identity, could not
    /// be read.
    CpuCache(String),
    /// A served path's volume could not be identified or sized.
    Volume { path: PathBuf, message: String },
}

impl fmt::Display for DiscoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostMemory(message) => write!(f, "host memory discovery failed: {message}"),
            Self::OsBuild(message) => write!(f, "the OS build could not be read: {message}"),
            Self::CpuIdentity(message) => {
                write!(f, "the CPU identity could not be read: {message}")
            }
            Self::CpuCache(message) => write!(f, "the CPU's caches could not be read: {message}"),
            Self::Volume { path, message } => {
                write!(
                    f,
                    "the volume of {} could not be read: {message}",
                    path.display()
                )
            }
        }
    }
}

impl std::error::Error for DiscoveryError {}

fn availability(admission: Result<(), String>) -> Availability {
    match admission {
        Ok(()) => Availability::Available,
        Err(reason) => Availability::Unavailable { reason },
    }
}

/// Whether the memory of the GPU of `uuid` is the host's, as each of its
/// views that states it reports it.
fn stated_memory(reports: &Reports, uuid: [u8; 16]) -> Vec<bool> {
    reports
        .cuda
        .iter()
        .filter(|cuda| cuda.uuid == uuid)
        .map(|cuda| matches!(cuda.memory, CudaMemory::Host))
        .chain(
            reports
                .vulkan
                .iter()
                .filter(|vulkan| vulkan.uuid == uuid)
                .filter_map(|vulkan| vulkan.memory_is_host),
        )
        .collect()
}

/// Whose memory the GPU of `uuid` is, from every report that states it: the
/// host's or its own when they all agree, undecided when they disagree or
/// none states it.
fn device_memory(reports: &Reports, uuid: [u8; 16]) -> DeviceMemory {
    let stated = stated_memory(reports, uuid);
    match stated.first() {
        Some(&host) if stated.iter().all(|&each| each == host) => {
            if host {
                DeviceMemory::Host
            } else {
                DeviceMemory::Own
            }
        }
        _ => DeviceMemory::Undecided,
    }
}

/// The last-level cache of the GPU of `uuid`, as its CUDA view reports it:
/// Vulkan reports no cache size, so a device without a CUDA view has none.
fn device_cache(reports: &Reports, uuid: [u8; 16]) -> Option<Reported<u64>> {
    reports
        .cuda
        .iter()
        .filter(|cuda| cuda.uuid == uuid)
        .find_map(|cuda| cuda.l2)
        .map(|bytes| Reported::new(bytes, ReportedBy::CudaL2Cache))
}

/// The OS adapter LUID of the GPU of `uuid`, as any of its views reports it.
fn adapter_luid(reports: &Reports, uuid: [u8; 16]) -> Option<[u8; 8]> {
    reports
        .vulkan
        .iter()
        .filter(|vulkan| vulkan.uuid == uuid)
        .find_map(|vulkan| vulkan.memory.luid)
        .or_else(|| {
            reports
                .cuda
                .iter()
                .filter(|cuda| cuda.uuid == uuid)
                .find_map(|cuda| cuda.luid)
        })
}

/// Why an executor of the GPU of `uuid`, whose memory is undecided, cannot
/// run: its views state its memory differently, or none states it.
fn undecided(reports: &Reports, uuid: [u8; 16]) -> String {
    if stated_memory(reports, uuid).is_empty() {
        "no backend reports whether its memory is the host's or its own".to_owned()
    } else {
        "its backends report its memory differently: as the host's and as its own".to_owned()
    }
}

/// The revision `revision` of the machine `reports` describe.
pub(crate) fn build(revision: Revision, reports: &Reports) -> DeviceTopology {
    let machine = format!("{};ram={}", reports.cpu.identity, reports.host.value);
    let mut builder = Builder::new(
        revision,
        PhysicalIdentity::Host { machine },
        reports.cpu.cache,
        reports.host,
    );
    let host = builder.host();
    let host_ram = builder.host_ram();

    // Physical devices, by the identity their drivers share (a GPU's device
    // UUID, which its CUDA and Vulkan views report alike), each with whose
    // memory it is.
    for metal in &reports.metal {
        let memory = if metal.unified {
            DeviceMemory::Host
        } else {
            DeviceMemory::Own
        };
        builder.physical(PhysicalIdentity::Registry(metal.registry_id), memory, None);
    }
    for uuid in reports
        .cuda
        .iter()
        .map(|cuda| cuda.uuid)
        .chain(reports.vulkan.iter().map(|vulkan| vulkan.uuid))
    {
        builder.physical(
            PhysicalIdentity::Uuid(uuid),
            device_memory(reports, uuid),
            device_cache(reports, uuid),
        );
    }

    // Each GPU's own memory. A device whose memory is its own has its
    // device-local memory, as each report states it, of which the smallest
    // report is the store's capacity; a device whose memory is the host's
    // has the dedicated memory the OS reports for its adapter.
    let mut own: Vec<(PhysicalIdentity, Reported<u64>)> = Vec::new();
    for cuda in &reports.cuda {
        if device_memory(reports, cuda.uuid) == DeviceMemory::Own {
            own.push((
                PhysicalIdentity::Uuid(cuda.uuid),
                Reported::new(cuda.total, ReportedBy::CudaTotalMemory),
            ));
        }
    }
    for vulkan in &reports.vulkan {
        if device_memory(reports, vulkan.uuid) != DeviceMemory::Own {
            continue;
        }
        let memory = &vulkan.memory;
        let local = memory.allocates[StorageKind::Device as usize]
            .iter()
            .map(|&heap| &memory.heaps[heap])
            .filter(|heap| heap.device_local)
            .map(|heap| heap.bytes)
            .reduce(|total, bytes| total + bytes);
        if let Some(bytes) = local {
            own.push((
                PhysicalIdentity::Uuid(vulkan.uuid),
                Reported::new(bytes, ReportedBy::VulkanHeap),
            ));
        }
    }
    let devices = builder.topology().physical_devices().to_vec();
    for device in devices {
        let capacity = match (device.memory(), device.identity()) {
            (DeviceMemory::Host, PhysicalIdentity::Uuid(uuid)) => {
                dedicated(&reports.adapters, adapter_luid(reports, *uuid))
                    .map(|bytes| Reported::new(bytes, ReportedBy::DxgiDedicatedMemory))
            }
            (DeviceMemory::Own, identity) => own
                .iter()
                .filter(|(of, _)| of == identity)
                .map(|(_, reported)| *reported)
                .min_by_key(|reported| reported.value),
            _ => None,
        };
        if let Some(capacity) = capacity {
            builder.store(StoreKind::DeviceMemory(device.id()), capacity);
        }
    }

    for volume in &reports.volumes {
        let kind = StoreKind::Volume(volume.id.clone());
        let store = match builder.find_store(&kind) {
            Some(store) => store,
            None => builder.store(kind, volume.size),
        };
        builder.serve(volume.path.clone(), store);
    }

    // CPU: both kinds on host RAM, ungranted.
    let cpu = {
        let mut executor = builder.executor(ExecutorReport {
            physical: host,
            selector: ExecutorSelector::HostCpu,
            kind: ExecutorKind::Cpu,
            name: reports.cpu.name.clone(),
            availability: reports.cpu.availability.clone(),
            driver: reports.os_build.clone(),
            limits: Limits {
                max_allocation: Reported::new(
                    reports.cpu.max_allocation,
                    ReportedBy::CpuAddressSpace,
                ),
            },
            tensor_operations: false,
        });
        if reports.cpu.availability == Availability::Available {
            let path = executor.ungranted(host_ram, AccessRoute::Device);
            place_all(&mut executor, &[path]);
        }
        executor.id()
    };

    // Metal: a device whose memory is the host's places as every such
    // device does, under its recommended working set.
    for metal in &reports.metal {
        let physical = builder.device(&PhysicalIdentity::Registry(metal.registry_id));
        let own = own_memory(&builder, physical);
        let admission = metal.admission.clone().and_then(|()| {
            if metal.unified {
                Ok(())
            } else {
                Err("its memory is not the host's, and Metal reports no size for it".into())
            }
        });
        let available = admission.is_ok();
        let mut executor = builder.executor(ExecutorReport {
            physical,
            selector: ExecutorSelector::Metal {
                registry_id: metal.registry_id,
            },
            kind: if metal.unified {
                ExecutorKind::IntegratedGpu
            } else {
                ExecutorKind::DiscreteGpu
            },
            name: metal.name.clone(),
            availability: availability(admission),
            driver: reports.os_build.clone(),
            limits: Limits {
                max_allocation: Reported::new(
                    metal.max_allocation,
                    ReportedBy::MetalMaxBufferLength,
                ),
            },
            tensor_operations: metal.tensor_operations,
        });
        if available {
            place_host_class(
                &mut executor,
                own,
                host_ram,
                Reported::new(metal.working_set, ReportedBy::MetalRecommendedWorkingSet),
                ReportedBy::MetalWorkingSetBeyondDedicated,
                |before| {
                    Some(GrantSource::MetalWorkingSet {
                        registry_id: metal.registry_id,
                        before,
                    })
                },
            );
        }
    }

    // CUDA: a device whose memory is its own places device storage on it
    // under its usable memory and upload storage on host RAM (pinned),
    // ungranted; one whose memory is the host's places as every such device
    // does, under its total.
    for cuda in &reports.cuda {
        let memory = device_memory(reports, cuda.uuid);
        let physical = builder.device(&PhysicalIdentity::Uuid(cuda.uuid));
        let own = own_memory(&builder, physical);
        // CUDA's own report is one of those the device's memory is decided
        // from, so a decided device's memory is the one CUDA reports.
        let placement = cuda.admission.clone().and_then(|()| {
            if memory == DeviceMemory::Undecided {
                return Err(undecided(reports, cuda.uuid));
            }
            match &cuda.memory {
                CudaMemory::Host => Ok(None),
                CudaMemory::Own { usable: Ok(usable) } => Ok(Some(*usable)),
                CudaMemory::Own { usable: Err(error) } => {
                    Err(format!("its CUDA context could not be created: {error}"))
                }
            }
        });
        let mut executor = builder.executor(ExecutorReport {
            physical,
            selector: ExecutorSelector::Cuda { uuid: cuda.uuid },
            kind: if cuda.integrated {
                ExecutorKind::IntegratedGpu
            } else {
                ExecutorKind::DiscreteGpu
            },
            name: cuda.name.clone(),
            availability: availability(placement.clone().map(|_| ())),
            driver: cuda.driver.clone(),
            limits: Limits {
                max_allocation: Reported::new(cuda.max_allocation, ReportedBy::CudaTotalMemory),
            },
            tensor_operations: false,
        });
        match placement {
            Ok(None) => place_host_class(
                &mut executor,
                own,
                host_ram,
                Reported::new(cuda.total, ReportedBy::CudaTotalMemory),
                ReportedBy::CudaTotalBeyondDedicated,
                // What is left of these grants only the opened executor
                // knows (`RuntimeStatus`).
                |_| None,
            ),
            Ok(Some(usable)) => {
                let device = executor.granted(
                    own.expect("a CUDA device whose memory is its own has a store")
                        .0,
                    Reported::new(usable, ReportedBy::CudaUsableMemory),
                    // What is left of it only the opened executor knows.
                    None,
                    AccessRoute::Device,
                );
                let pinned = executor.ungranted(host_ram, AccessRoute::Upload);
                executor.place(StorageKind::Device, &[device]);
                executor.place(StorageKind::Upload, &[pinned]);
            }
            Err(_) => {}
        }
    }

    for vulkan in &reports.vulkan {
        let memory = device_memory(reports, vulkan.uuid);
        let physical = builder.device(&PhysicalIdentity::Uuid(vulkan.uuid));
        let own = own_memory(&builder, physical);
        let heaps = &vulkan.memory;
        let placement = vulkan.floor.clone().and_then(|()| match memory {
            DeviceMemory::Host => Ok(Decided::Host),
            DeviceMemory::Own => Ok(Decided::Own),
            DeviceMemory::Undecided => Err(undecided(reports, vulkan.uuid)),
        });
        let mut executor = builder.executor(ExecutorReport {
            physical,
            selector: ExecutorSelector::Vulkan { uuid: vulkan.uuid },
            kind: vulkan.kind,
            name: vulkan.name.clone(),
            availability: availability(placement.clone().map(|_| ())),
            driver: vulkan.driver.clone(),
            limits: Limits {
                max_allocation: Reported::new(
                    vulkan.max_allocation,
                    ReportedBy::VulkanMaxAllocation,
                ),
            },
            tensor_operations: false,
        });
        match placement {
            // A device whose memory is its own: one access path per heap the
            // backend allocates from, a device-local heap's to its own
            // memory granted the heap's size, another heap's to host RAM,
            // ungranted.
            Ok(Decided::Own) => {
                let mut paths: HashMap<usize, OwnPath> = HashMap::new();
                for kind in StorageKind::ALL {
                    let placement = heaps.allocates[kind as usize]
                        .iter()
                        .map(|&heap| {
                            *paths.entry(heap).or_insert_with(|| {
                                let route = AccessRoute::VulkanHeap(heap as u32);
                                if !heaps.heaps[heap].device_local {
                                    return executor.ungranted(host_ram, route);
                                }
                                executor.granted(
                                    own.expect("a device-local heap is its own memory").0,
                                    Reported::new(heaps.heaps[heap].bytes, ReportedBy::VulkanHeap),
                                    Some(GrantSource::VulkanHeaps {
                                        uuid: vulkan.uuid,
                                        heaps: vec![heap],
                                        before: 0,
                                    }),
                                    route,
                                )
                            })
                        })
                        .collect::<Vec<_>>();
                    executor.place(kind, &placement);
                }
            }
            // A device whose memory is the host's: its heaps together state
            // one total, split across heaps only as a presentation (§4); the
            // heaps' live budget is attributed along the placement.
            Ok(Decided::Host) => {
                let total = heaps.heaps.iter().map(|heap| heap.bytes).sum::<u64>();
                let all = (0..heaps.heaps.len()).collect::<Vec<_>>();
                place_host_class(
                    &mut executor,
                    own,
                    host_ram,
                    Reported::new(total, ReportedBy::VulkanHeaps),
                    ReportedBy::VulkanHeapsBeyondDedicated,
                    |before| {
                        Some(GrantSource::VulkanHeaps {
                            uuid: vulkan.uuid,
                            heaps: all.clone(),
                            before,
                        })
                    },
                );
            }
            Err(_) => {}
        }
    }

    // Transfer paths: each executor's copies between host RAM and the other
    // stores it reaches, the CPU's file reads from each volume, and a GPU's
    // copies to another whose memory it reports peer access to.
    let mut transfers = Vec::new();
    let topology = builder.topology();
    for executor in topology.executors() {
        let mut reached: Vec<StoreId> = Vec::new();
        for path in topology.access_paths() {
            if path.executor() == executor.id()
                && path.store() != host_ram
                && !reached.contains(&path.store())
            {
                reached.push(path.store());
            }
        }
        for store in reached {
            transfers.push((host_ram, store, executor.id()));
            transfers.push((store, host_ram, executor.id()));
        }
    }
    if reports.cpu.availability == Availability::Available {
        for store in topology.stores() {
            if let StoreKind::Volume(_) = store.kind() {
                transfers.push((store.id(), host_ram, cpu));
            }
        }
    }
    let own_store = |uuid: [u8; 16]| {
        let device = topology
            .physical_devices()
            .iter()
            .find(|device| *device.identity() == PhysicalIdentity::Uuid(uuid))?;
        topology
            .stores()
            .iter()
            .find(|store| *store.kind() == StoreKind::DeviceMemory(device.id()))
            .map(|store| store.id())
    };
    for cuda in &reports.cuda {
        let executor = topology.executors().iter().find(|executor| {
            executor.selector() == ExecutorSelector::Cuda { uuid: cuda.uuid }
                && *executor.availability() == Availability::Available
        });
        let (Some(executor), Some(from)) = (executor, own_store(cuda.uuid)) else {
            continue;
        };
        for &peer in &cuda.peers {
            if let Some(to) = own_store(peer) {
                transfers.push((from, to, executor.id()));
                transfers.push((to, from, executor.id()));
            }
        }
    }
    for (from, to, by) in transfers {
        builder.transfer(from, to, by);
    }

    for diagnostic in &reports.diagnostics {
        builder.diagnostic(diagnostic.clone());
    }
    builder.finish()
}

/// Whose memory a device's is, once decided.
#[derive(Clone, Copy)]
enum Decided {
    Host,
    Own,
}

/// The store of `physical`'s own memory and its capacity, when it has one.
fn own_memory(builder: &Builder, physical: PhysicalDeviceId) -> Option<(StoreId, Reported<u64>)> {
    let store = builder.find_store(&StoreKind::DeviceMemory(physical))?;
    let capacity = builder
        .topology()
        .store(store)
        .expect("a found store is of the revision")
        .capacity();
    Some((store, capacity))
}

fn place_all(executor: &mut ExecutorBuilder<'_>, paths: &[OwnPath]) {
    for kind in StorageKind::ALL {
        executor.place(kind, paths);
    }
}

/// Place both kinds of storage of an executor of a device whose memory is
/// the host's: on its own memory `own` (the store of the dedicated memory
/// the OS reports for its adapter, and that memory's size), where it has
/// some, under the lesser of it and the executor's `total`, then on host
/// RAM under what `total` leaves beyond it (`beyond`). `status` names each
/// path's live grant source from the bytes placed ahead of it.
fn place_host_class(
    executor: &mut ExecutorBuilder<'_>,
    own: Option<(StoreId, Reported<u64>)>,
    host_ram: StoreId,
    total: Reported<u64>,
    beyond: ReportedBy,
    status: impl Fn(u64) -> Option<GrantSource>,
) {
    let mut placement = Vec::new();
    let mut before = 0;
    let mut grant =
        |executor: &mut ExecutorBuilder<'_>, store: StoreId, reported: Reported<u64>| {
            let path = executor.granted(store, reported, status(before), AccessRoute::Device);
            before += reported.value;
            placement.push(path);
        };
    match own {
        Some((store, dedicated)) => {
            grant(
                executor,
                store,
                if dedicated.value <= total.value {
                    dedicated
                } else {
                    total
                },
            );
            if total.value > dedicated.value {
                grant(
                    executor,
                    host_ram,
                    Reported::new(total.value - dedicated.value, beyond),
                );
            }
        }
        None => grant(executor, host_ram, total),
    }
    place_all(executor, &placement);
}

/// The dedicated memory the OS reports for the adapter of `luid`, when it
/// reports some.
fn dedicated(adapters: &[AdapterReport], luid: Option<[u8; 8]>) -> Option<u64> {
    let luid = luid?;
    adapters
        .iter()
        .find(|adapter| adapter.luid == luid)
        .map(|adapter| adapter.dedicated)
        .filter(|bytes| *bytes > 0)
}

/// What discovery reads, with the native handles status sampling reads
/// again.
pub(crate) struct Gathered {
    pub(crate) reports: Reports,
    pub(crate) natives: Natives,
}

/// Unopened native handles a sample reads.
#[derive(Default)]
pub(crate) struct Natives {
    #[cfg(target_os = "macos")]
    pub(crate) metal: Vec<seismic_metal::DeviceHandle>,
}

/// The diagnostic of a backend whose enumeration succeeded with no device:
/// every backend yields devices or a diagnostic.
fn no_device(backend: BackendName) -> Diagnostic {
    Diagnostic {
        source: DiagnosticSource::Backend(backend),
        message: format!("{} reports no device", backend.as_str()),
    }
}

/// Read the backend crates and the OS. Only cheap native queries run, except
/// what the backends themselves need to establish availability, grants and
/// capability: a Metal device's language and tensor-operation probes
/// (compiles), a Vulkan device's numerical probes, and a discrete CUDA
/// device's usable memory, which needs its context (the last two once per
/// process).
pub(crate) fn gather(paths: &[PathBuf]) -> Result<Gathered, DiscoveryError> {
    let host = host::capacity().map_err(DiscoveryError::HostMemory)?;
    let os_build = host::os_build().map_err(DiscoveryError::OsBuild)?;
    let mut diagnostics = Vec::new();

    let cache = host::last_level_cache().map_err(DiscoveryError::CpuCache)?;
    let cpu = CpuReport {
        name: format!("Host CPU ({})", std::env::consts::ARCH),
        identity: host::cpu_identity(cache.map(|cache| cache.value))
            .map_err(DiscoveryError::CpuIdentity)?,
        cache,
        availability: cpu_availability(seismic_native_cpu::Tier::detected()),
        max_allocation: seismic_cpu::MAX_ALLOCATION_BYTES,
    };

    #[cfg(target_os = "macos")]
    let natives = Natives {
        metal: seismic_metal::DeviceHandle::discover(),
    };
    #[cfg(target_os = "macos")]
    if natives.metal.is_empty() {
        diagnostics.push(no_device(BackendName::Metal));
    }
    #[cfg(target_os = "macos")]
    let metal = natives
        .metal
        .iter()
        .map(|handle| {
            let admission = seismic_metal::profile::language(handle)
                .map(|_| ())
                .map_err(|error| error.to_string());
            MetalReport {
                registry_id: handle.registry_id(),
                name: handle.name(),
                unified: handle.has_unified_memory(),
                tensor_operations: admission.is_ok()
                    && seismic_metal::profile::forms_tensor_operations(handle),
                admission,
                working_set: handle.recommended_working_set_bytes(),
                max_allocation: handle.max_allocation_bytes(),
            }
        })
        .collect();
    #[cfg(not(target_os = "macos"))]
    let (natives, metal) = (Natives::default(), Vec::new());

    let cuda = cuda_reports(&mut diagnostics);

    #[cfg(not(target_os = "macos"))]
    let vulkan = match seismic_vulkan::discover() {
        Err(error) => {
            diagnostics.push(Diagnostic {
                source: DiagnosticSource::Backend(BackendName::Vulkan),
                message: error.to_string(),
            });
            Vec::new()
        }
        Ok(descriptions) => {
            if descriptions.is_empty() {
                diagnostics.push(no_device(BackendName::Vulkan));
            }
            descriptions.iter().map(vulkan_report).collect()
        }
    };
    // Vulkan is not built on macOS (MoltenVK is not supported).
    #[cfg(target_os = "macos")]
    let vulkan = {
        diagnostics.push(Diagnostic {
            source: DiagnosticSource::Backend(BackendName::Vulkan),
            message: "this build has no Vulkan runtime".into(),
        });
        Vec::new()
    };

    #[cfg(windows)]
    let adapters = match crate::adapters::adapters() {
        Ok(adapters) => adapters,
        Err(message) => {
            diagnostics.push(Diagnostic {
                source: DiagnosticSource::Adapters,
                message,
            });
            Vec::new()
        }
    };
    #[cfg(not(windows))]
    let adapters = Vec::new();

    let volumes = paths
        .iter()
        .map(|path| {
            let failed = |error: std::io::Error| DiscoveryError::Volume {
                path: path.clone(),
                message: error.to_string(),
            };
            Ok(VolumeReport {
                path: path.clone(),
                id: crate::volume::identify(path).map_err(failed)?,
                size: crate::volume::size(path).map_err(failed)?,
            })
        })
        .collect::<Result<_, DiscoveryError>>()?;

    Ok(Gathered {
        reports: Reports {
            host,
            os_build,
            cpu,
            metal,
            cuda,
            vulkan,
            adapters,
            volumes,
            diagnostics,
        },
        natives,
    })
}

/// Every CUDA device's report, and which of the others each reads directly.
fn cuda_reports(diagnostics: &mut Vec<Diagnostic>) -> Vec<CudaReport> {
    let diagnostic = |message: String| Diagnostic {
        source: DiagnosticSource::Backend(BackendName::Cuda),
        message,
    };
    let count = match seismic_cuda::device_count() {
        Ok(0) => {
            diagnostics.push(no_device(BackendName::Cuda));
            return Vec::new();
        }
        Ok(count) => count,
        Err(error) => {
            diagnostics.push(diagnostic(error.to_string()));
            return Vec::new();
        }
    };
    let mut described = Vec::new();
    for ordinal in 0..count {
        match seismic_cuda::describe(ordinal) {
            Ok(descriptor) => described.push(descriptor),
            Err(error) => {
                diagnostics.push(diagnostic(format!("device ordinal {ordinal}: {error}")))
            }
        }
    }
    let mut reports = Vec::new();
    for descriptor in &described {
        let mut peers = Vec::new();
        for peer in &described {
            if peer.ordinal == descriptor.ordinal {
                continue;
            }
            match seismic_cuda::can_access_peer(descriptor.ordinal, peer.ordinal) {
                Ok(true) => peers.push(peer.uuid),
                Ok(false) => {}
                Err(error) => diagnostics.push(diagnostic(format!(
                    "peer access of ordinal {} to {}: {error}",
                    descriptor.ordinal, peer.ordinal
                ))),
            }
        }
        let (ordinal, uuid) = (descriptor.ordinal, descriptor.uuid);
        reports.push(CudaReport {
            uuid,
            luid: descriptor.luid,
            name: descriptor.name.clone(),
            driver: format!("cuda driver {}", descriptor.driver_api),
            admission: descriptor.admission().map_err(|error| error.to_string()),
            total: descriptor.total_memory_bytes,
            max_allocation: seismic_cuda::max_allocation_bytes(descriptor.total_memory_bytes),
            integrated: descriptor.integrated,
            l2: (descriptor.l2_cache_bytes > 0).then_some(descriptor.l2_cache_bytes),
            memory: if descriptor.integrated || descriptor.numa_node {
                CudaMemory::Host
            } else {
                CudaMemory::Own {
                    usable: usable_at_first_read(uuid, || {
                        seismic_cuda::usable_memory_bytes(ordinal, uuid)
                            .map_err(|error| error.to_string())
                    }),
                }
            },
            peers,
        });
    }
    reports
}

/// The Vulkan backend's description of one device.
#[cfg(not(target_os = "macos"))]
fn vulkan_report(description: &seismic_vulkan::Description) -> VulkanReport {
    let device = &description.facts;
    let heaps = |types: &[u32]| {
        seismic_vulkan::Allocation::heaps(&device.memory, types)
            .into_iter()
            .map(|heap| heap as usize)
            .collect()
    };
    let kind = match device.device_type {
        seismic_vulkan::DeviceType::Cpu => ExecutorKind::Cpu,
        seismic_vulkan::DeviceType::Integrated => ExecutorKind::IntegratedGpu,
        seismic_vulkan::DeviceType::Discrete => ExecutorKind::DiscreteGpu,
        seismic_vulkan::DeviceType::Other => ExecutorKind::Other,
    };
    VulkanReport {
        uuid: device.uuid,
        kind,
        memory_is_host: (device.device_type != seismic_vulkan::DeviceType::Other)
            .then(|| device.device_type.memory_is_host()),
        name: device.name.clone(),
        floor: description.floor.clone(),
        driver: format!(
            "vulkan driver {} version {}",
            device.driver_id, device.driver_version
        ),
        max_allocation: device.limits.max_allocation_bytes,
        memory: VulkanMemory {
            luid: device.luid,
            heaps: device
                .memory
                .heaps
                .iter()
                .map(|heap| VulkanHeap {
                    bytes: heap.bytes,
                    device_local: heap.device_local,
                })
                .collect(),
            allocates: [
                heaps(&device.allocation.device),
                heaps(&device.allocation.upload),
            ],
        },
    }
}

/// Discovery must enforce the same minimum instruction set as native
/// specialization. Otherwise assessment promises a CPU that cannot form even
/// the first import kernel (for example QEMU's default virtual CPU).
pub(crate) fn cpu_availability(tier: Option<seismic_native_cpu::Tier>) -> Availability {
    match tier {
        Some(_) => Availability::Available,
        None => Availability::Unavailable {
            reason: format!(
                "CPU execution requires {} ({}) or higher; \
                 expose these CPU features to the virtual machine",
                seismic_native_cpu::Tier::all()[0].name(),
                seismic_native_cpu::Tier::all()[0].features(),
            ),
        },
    }
}

/// The memory a discrete CUDA device lets this process use, as the driver
/// reports it once this process first creates the device's context
/// (`seismic_cuda::usable_memory_bytes`), read once per process and
/// device: the first read precedes any allocation of this process, so a
/// later discovery (a refresh, or another inventory) keeps it rather than
/// reading the process's own holdings. A failed read is not kept.
fn usable_at_first_read(
    uuid: [u8; 16],
    read: impl FnOnce() -> Result<u64, String>,
) -> Result<u64, String> {
    static USABLE: OnceLock<Mutex<HashMap<[u8; 16], u64>>> = OnceLock::new();
    let mut usable = USABLE
        .get_or_init(Mutex::default)
        .lock()
        .expect("the usable-memory registry lock is never poisoned");
    if let Some(bytes) = usable.get(&uuid) {
        return Ok(*bytes);
    }
    let bytes = read()?;
    usable.insert(uuid, bytes);
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::{cpu_availability, usable_at_first_read};
    use crate::model::Availability;

    #[test]
    fn cpu_discovery_enforces_the_native_instruction_set_floor() {
        let Availability::Unavailable { reason } = cpu_availability(None) else {
            panic!("a CPU with no native instruction-set tier was admitted");
        };
        assert!(reason.contains(seismic_native_cpu::Tier::all()[0].features()));
        for tier in seismic_native_cpu::Tier::all() {
            assert_eq!(cpu_availability(Some(*tier)), Availability::Available);
        }
    }

    /// The first successful read stands for the process; after a failure it
    /// is read again.
    #[test]
    fn usable_memory_is_kept_for_the_process_and_a_failure_is_not() {
        let uuid = [0xa5; 16];
        let busy = || Err("busy".to_owned());
        assert_eq!(usable_at_first_read(uuid, busy), Err("busy".to_owned()));
        assert_eq!(usable_at_first_read(uuid, || Ok(6 << 30)), Ok(6 << 30));
        assert_eq!(usable_at_first_read(uuid, || Ok(1 << 30)), Ok(6 << 30));
    }
}
