//! Topologies built from recorded driver reports for each situation of the
//! spec's §3, their placements and grants, revisions, and identities. The
//! recorded machines and their sources are `crate::fixtures`'.

use crate::discovery::{Gathered, Natives, Reports, VolumeReport, VulkanReport};
use crate::fixtures::{
    a10_8q, gb10, linux_host, lunar_lake, m4_pro, m4_pro_without_unified_memory, rtx3090,
    rtx3090_cuda, rtx3090_host, rtx3090_uuid, rtx3090_vulkan, ryzen_igpu_and_rtx4070,
    strix_halo_linux, strix_halo_linux_vulkan, strix_halo_windows, topology, uuid,
    A10_8Q_CUDA_USABLE, A10_8Q_VULKAN_HEAP, GB, GB10_CUDA_TOTAL, GB10_UUID, GB10_VULKAN_HEAP,
    LUNAR_LAKE_HEAP, LUNAR_LAKE_UUID, M4_PRO_REGISTRY, M4_PRO_WORKING_SET, RTX3090_CUDA_TOTAL,
    RTX3090_CUDA_USABLE, RTX3090_HOST, RTX3090_VOLUME, RTX3090_VULKAN_DRIVER, RTX3090_VULKAN_HEAP,
    RTX4070_HEAP, RTX4070_UUID, RYZEN_IGPU_HEAPS, RYZEN_IGPU_UUID, STRIX_HALO_LINUX_HEAPS,
    STRIX_HALO_LINUX_UUID, STRIX_HALO_WINDOWS_DEDICATED, STRIX_HALO_WINDOWS_HEAPS,
    STRIX_HALO_WINDOWS_LUID, STRIX_HALO_WINDOWS_UUID,
};
use crate::model::{
    AccessRoute, Availability, DeviceMemory, DeviceTopology, Executor, PhysicalDeviceId,
    PhysicalIdentity, Reported, ReportedBy, Revision, StorageKind, StoreId, StoreKind, VolumeId,
};
use crate::selector::ExecutorSelector;
use crate::Inventory;
use seismic_lang::registry::BackendName;
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The executor of `backend` on the physical device of `identity`.
fn executor<'a>(
    topology: &'a DeviceTopology,
    backend: BackendName,
    identity: &PhysicalIdentity,
) -> &'a Executor {
    topology
        .executors()
        .iter()
        .find(|executor| {
            executor.backend() == backend
                && topology
                    .physical_device(executor.physical())
                    .unwrap()
                    .identity()
                    == identity
        })
        .unwrap()
}

/// Where a placement lands: each path's store and grant.
fn placement(
    topology: &DeviceTopology,
    executor: &Executor,
    kind: StorageKind,
) -> Vec<(StoreKind, Option<Reported<u64>>)> {
    executor
        .placement(kind)
        .iter()
        .map(|&id| {
            let path = topology.access_path(id).unwrap();
            (
                topology.store(path.store()).unwrap().kind().clone(),
                path.grant(),
            )
        })
        .collect()
}

fn device_memory(topology: &DeviceTopology, uuid: [u8; 16]) -> StoreKind {
    let executor = topology
        .resolve(ExecutorSelector::Vulkan { uuid })
        .or_else(|| topology.resolve(ExecutorSelector::Cuda { uuid }))
        .unwrap();
    StoreKind::DeviceMemory(executor.physical())
}

fn store_of(topology: &DeviceTopology, kind: &StoreKind) -> StoreId {
    topology
        .stores()
        .iter()
        .find(|store| store.kind() == kind)
        .unwrap()
        .id()
}

fn reported(value: u64, by: ReportedBy) -> Option<Reported<u64>> {
    Some(Reported::new(value, by))
}

/// The volume store the inventory serves `path` on.
fn volume(topology: &DeviceTopology, path: &Path) -> Option<StoreId> {
    topology
        .stores()
        .iter()
        .map(|store| store.id())
        .find(|&store| topology.volume_paths(store).any(|served| served == path))
}

/// The topology as text: what each row's test produced.
fn describe(topology: &DeviceTopology) -> String {
    let store = |id: StoreId| match topology.store(id).unwrap().kind() {
        StoreKind::HostRam => "host".to_owned(),
        StoreKind::DeviceMemory(device) => format!("device{}", physical_index(topology, *device)),
        StoreKind::Volume(volume) => format!("volume {}", volume.device),
    };
    let mut text = String::new();
    for each in topology.stores() {
        writeln!(
            text,
            "  store {}: {} ({:?})",
            store(each.id()),
            gb(each.capacity().value),
            each.capacity().by
        )
        .unwrap();
    }
    for executor in topology.executors() {
        write!(
            text,
            "  {:?} on device{} ({:?} memory) [{}]",
            executor.backend(),
            physical_index(topology, executor.physical()),
            topology
                .physical_device(executor.physical())
                .unwrap()
                .memory(),
            executor.driver(),
        )
        .unwrap();
        if let Availability::Unavailable { reason } = executor.availability() {
            writeln!(text, ": unavailable ({reason})").unwrap();
            continue;
        }
        for kind in StorageKind::ALL {
            let places = executor
                .placement(kind)
                .iter()
                .map(|&id| {
                    let path = topology.access_path(id).unwrap();
                    match path.grant() {
                        Some(grant) => format!(
                            "{} under {} ({:?})",
                            store(path.store()),
                            gb(grant.value),
                            grant.by
                        ),
                        None => format!("{} ungranted", store(path.store())),
                    }
                })
                .collect::<Vec<_>>();
            write!(text, "; {kind:?} -> [{}]", places.join(", ")).unwrap();
        }
        writeln!(text).unwrap();
    }
    for path in topology.transfer_paths() {
        writeln!(
            text,
            "  transfer {} -> {} by {:?}",
            store(path.from()),
            store(path.to()),
            topology.executor(path.by()).unwrap().backend()
        )
        .unwrap();
    }
    text
}

fn physical_index(topology: &DeviceTopology, id: PhysicalDeviceId) -> usize {
    topology
        .physical_devices()
        .iter()
        .position(|device| device.id() == id)
        .unwrap()
}

fn gb(bytes: u64) -> String {
    format!("{:.2} GB ({bytes})", bytes as f64 / GB as f64)
}

fn show(row: &str, topology: &DeviceTopology) {
    eprintln!("{row}\n{}", describe(topology));
}

/// Every path names nodes of its revision, and every placement names the
/// executor's own access paths.
fn assert_consistent(topology: &DeviceTopology) {
    for path in topology.access_paths() {
        assert!(topology.executor(path.executor()).is_some());
        assert!(topology.store(path.store()).is_some());
    }
    for path in topology.transfer_paths() {
        assert!(topology.store(path.from()).is_some() && topology.store(path.to()).is_some());
        assert!(topology.executor(path.by()).is_some());
    }
    for executor in topology.executors() {
        for kind in StorageKind::ALL {
            for &id in executor.placement(kind) {
                assert_eq!(topology.access_path(id).unwrap().executor(), executor.id());
            }
        }
    }
}

#[test]
fn m4_pro_places_metal_on_host_ram_under_its_working_set() {
    let topology = topology(&m4_pro());
    show("M4 Pro, 48 GB", &topology);
    assert_consistent(&topology);
    assert_eq!(topology.stores().len(), 1);
    let registry = PhysicalIdentity::Registry(M4_PRO_REGISTRY);
    let metal = executor(&topology, BackendName::Metal, &registry);
    let cpu = topology.resolve(ExecutorSelector::HostCpu).unwrap();
    let working_set = reported(M4_PRO_WORKING_SET, ReportedBy::MetalRecommendedWorkingSet);
    for kind in StorageKind::ALL {
        assert_eq!(
            placement(&topology, metal, kind),
            [(StoreKind::HostRam, working_set)]
        );
        assert_eq!(
            placement(&topology, cpu, kind),
            [(StoreKind::HostRam, None)]
        );
    }
    assert!(topology.transfer_paths().is_empty());
}

#[test]
fn rtx3090_cuda_and_vulkan_share_its_memory_under_their_own_grants() {
    let gpu = rtx3090_uuid();
    let topology = topology(&rtx3090());
    show("Linux, RTX 3090", &topology);
    assert_consistent(&topology);
    let own = device_memory(&topology, gpu);
    let store = topology.store(store_of(&topology, &own)).unwrap();
    // The smallest report of the memory: CUDA's total, below Vulkan's heap.
    assert_eq!(
        store.capacity(),
        Reported::new(RTX3090_CUDA_TOTAL, ReportedBy::CudaTotalMemory)
    );
    let cuda = executor(&topology, BackendName::Cuda, &PhysicalIdentity::Uuid(gpu));
    let vulkan = executor(&topology, BackendName::Vulkan, &PhysicalIdentity::Uuid(gpu));
    assert_eq!(cuda.physical(), vulkan.physical());
    assert_eq!(
        placement(&topology, cuda, StorageKind::Device),
        [(
            own.clone(),
            reported(RTX3090_CUDA_USABLE, ReportedBy::CudaUsableMemory)
        )]
    );
    assert_eq!(
        placement(&topology, cuda, StorageKind::Upload),
        [(StoreKind::HostRam, None)]
    );
    // ReBAR: the device-local host-visible type is on the 24 GiB heap, so
    // upload storage shares device storage's path.
    for kind in StorageKind::ALL {
        assert_eq!(
            placement(&topology, vulkan, kind),
            [(
                own.clone(),
                reported(RTX3090_VULKAN_HEAP, ReportedBy::VulkanHeap)
            )]
        );
    }
    assert_eq!(
        vulkan.placement(StorageKind::Device),
        vulkan.placement(StorageKind::Upload)
    );
    assert_eq!(topology.transfer_paths().len(), 4);
}

#[test]
fn gb10_places_every_executor_on_host_ram() {
    let topology = topology(&gb10());
    show("GB10, 128 GB", &topology);
    assert_consistent(&topology);
    assert_eq!(topology.stores().len(), 1);
    let gpu = PhysicalIdentity::Uuid(GB10_UUID);
    let cuda = executor(&topology, BackendName::Cuda, &gpu);
    let vulkan = executor(&topology, BackendName::Vulkan, &gpu);
    for kind in StorageKind::ALL {
        // CUDA's grant is its total, which on sparky is host RAM's capacity.
        assert_eq!(
            placement(&topology, cuda, kind),
            [(
                StoreKind::HostRam,
                reported(GB10_CUDA_TOTAL, ReportedBy::CudaTotalMemory)
            )]
        );
        assert_eq!(
            placement(&topology, vulkan, kind),
            [(
                StoreKind::HostRam,
                reported(GB10_VULKAN_HEAP, ReportedBy::VulkanHeaps)
            )]
        );
    }
    assert!(topology.transfer_paths().is_empty());
}

#[test]
fn strix_halo_on_linux_is_granted_its_heaps_together() {
    let topology = topology(&strix_halo_linux());
    show("Strix Halo, Linux (#199; heaps of report 42266)", &topology);
    assert_consistent(&topology);
    assert_eq!(topology.stores().len(), 1);
    let gpu = PhysicalIdentity::Uuid(uuid(STRIX_HALO_LINUX_UUID));
    let vulkan = executor(&topology, BackendName::Vulkan, &gpu);
    for kind in StorageKind::ALL {
        assert_eq!(
            placement(&topology, vulkan, kind),
            [(
                StoreKind::HostRam,
                reported(STRIX_HALO_LINUX_HEAPS.iter().sum(), ReportedBy::VulkanHeaps)
            )]
        );
    }
}

#[test]
fn strix_halo_on_windows_places_its_dedicated_memory_first() {
    let topology = topology(&strix_halo_windows());
    show(
        "Strix Halo, Windows (#207; heaps of report 52227)",
        &topology,
    );
    assert_consistent(&topology);
    let gpu = uuid(STRIX_HALO_WINDOWS_UUID);
    let radeon = device_memory(&topology, gpu);
    assert_eq!(
        topology
            .store(store_of(&topology, &radeon))
            .unwrap()
            .capacity(),
        Reported::new(
            STRIX_HALO_WINDOWS_DEDICATED,
            ReportedBy::DxgiDedicatedMemory
        )
    );
    let vulkan = topology
        .resolve(ExecutorSelector::Vulkan { uuid: gpu })
        .unwrap();
    assert_eq!(
        topology
            .physical_device(vulkan.physical())
            .unwrap()
            .identity(),
        &PhysicalIdentity::Uuid(gpu)
    );
    let total: u64 = STRIX_HALO_WINDOWS_HEAPS.iter().sum();
    for kind in StorageKind::ALL {
        assert_eq!(
            placement(&topology, vulkan, kind),
            [
                (
                    radeon.clone(),
                    reported(
                        STRIX_HALO_WINDOWS_DEDICATED,
                        ReportedBy::DxgiDedicatedMemory
                    )
                ),
                (
                    StoreKind::HostRam,
                    reported(
                        total - STRIX_HALO_WINDOWS_DEDICATED,
                        ReportedBy::VulkanHeapsBeyondDedicated
                    )
                ),
            ]
        );
    }
    assert_eq!(topology.transfer_paths().len(), 2);
}

#[test]
fn lunar_lake_is_granted_its_one_heap() {
    let topology = topology(&lunar_lake());
    show("Lunar Lake (#206; heap of report 52450)", &topology);
    assert_consistent(&topology);
    assert_eq!(topology.stores().len(), 1);
    let gpu = PhysicalIdentity::Uuid(uuid(LUNAR_LAKE_UUID));
    let vulkan = executor(&topology, BackendName::Vulkan, &gpu);
    assert_eq!(
        placement(&topology, vulkan, StorageKind::Device),
        [(
            StoreKind::HostRam,
            reported(LUNAR_LAKE_HEAP, ReportedBy::VulkanHeaps)
        )]
    );
}

#[test]
fn an_igpu_and_a_discrete_gpu_are_both_physical_devices() {
    let (igpu, gpu) = (uuid(RYZEN_IGPU_UUID), uuid(RTX4070_UUID));
    let topology = topology(&ryzen_igpu_and_rtx4070());
    show("Ryzen iGPU + RTX 4070 (reports 39430, 51857)", &topology);
    assert_consistent(&topology);
    assert_eq!(topology.physical_devices().len(), 3);
    assert_eq!(topology.stores().len(), 2);
    let on_igpu = executor(
        &topology,
        BackendName::Vulkan,
        &PhysicalIdentity::Uuid(igpu),
    );
    assert_eq!(
        placement(&topology, on_igpu, StorageKind::Device),
        [(
            StoreKind::HostRam,
            reported(RYZEN_IGPU_HEAPS.iter().sum(), ReportedBy::VulkanHeaps)
        )]
    );
    let on_gpu = executor(&topology, BackendName::Vulkan, &PhysicalIdentity::Uuid(gpu));
    assert_eq!(
        placement(&topology, on_gpu, StorageKind::Device),
        [(
            device_memory(&topology, gpu),
            reported(RTX4070_HEAP, ReportedBy::VulkanHeap)
        )]
    );
}

/// Two RTX 3090s, each the rtx3090 host's, with and without the peer
/// access CUDA reports: the host has one GPU, so the peer report is the
/// rule's input here, not a recording.
#[test]
fn gpus_copy_to_each_other_where_cuda_reports_peer_access() {
    let (first, second) = (rtx3090_uuid(), uuid("D37CB696763FCACE0D8DCBA6B3D100D1"));
    for peered in [false, true] {
        let mut reports = linux_host(RTX3090_HOST);
        let peers = |other| if peered { vec![other] } else { Vec::new() };
        reports.cuda.push(rtx3090_cuda(first, peers(second)));
        reports.cuda.push(rtx3090_cuda(second, peers(first)));
        reports.vulkan.push(rtx3090_vulkan(first));
        reports.vulkan.push(rtx3090_vulkan(second));
        let topology = topology(&reports);
        show(&format!("2 x RTX 3090, peer access {peered}"), &topology);
        assert_consistent(&topology);
        assert_eq!(topology.stores().len(), 3);
        for each in [first, second] {
            let own = device_memory(&topology, each);
            for backend in [BackendName::Cuda, BackendName::Vulkan] {
                let executor = executor(&topology, backend, &PhysicalIdentity::Uuid(each));
                assert_eq!(
                    placement(&topology, executor, StorageKind::Device)[0].0,
                    own
                );
            }
        }
        let host = topology.host_ram().id();
        let peer_paths = topology
            .transfer_paths()
            .iter()
            .filter(|path| path.from() != host && path.to() != host)
            .count();
        // Each CUDA executor copies both ways between its memory and its
        // peer's.
        assert_eq!(peer_paths, if peered { 4 } else { 0 });
    }
}

#[test]
fn a_discrete_gpu_has_an_access_path_to_host_ram() {
    let gpu = rtx3090_uuid();
    let mut reports = rtx3090_host();
    reports.cuda.push(rtx3090_cuda(gpu, Vec::new()));
    let topology = topology(&reports);
    let cuda = executor(&topology, BackendName::Cuda, &PhysicalIdentity::Uuid(gpu));
    assert!(topology
        .access_paths()
        .iter()
        .any(|path| path.executor() == cuda.id() && path.store() == topology.host_ram().id()));
}

/// The rtx3090 host's evaluation volume, served at two paths.
fn rtx3090_volume(path: &str) -> VolumeReport {
    let (device, uuid, size) = RTX3090_VOLUME;
    VolumeReport {
        path: PathBuf::from(path),
        id: VolumeId {
            device,
            uuid: uuid.map(str::to_owned),
        },
        size: Reported::new(size, ReportedBy::FilesystemSize),
    }
}

#[test]
fn a_served_volume_is_read_into_host_ram_by_the_cpu() {
    let mut reports = rtx3090();
    reports.volumes.push(rtx3090_volume("/models"));
    reports.volumes.push(rtx3090_volume("/cache"));
    let topology = topology(&reports);
    show("Streaming", &topology);
    assert_consistent(&topology);
    let store = volume(&topology, Path::new("/models")).unwrap();
    assert_eq!(volume(&topology, Path::new("/cache")), Some(store));
    let cpu = topology.resolve(ExecutorSelector::HostCpu).unwrap();
    let reads = topology
        .transfer_paths()
        .iter()
        .filter(|path| path.from() == store)
        .collect::<Vec<_>>();
    assert_eq!(reads.len(), 1);
    assert_eq!(
        (reads[0].to(), reads[0].by()),
        (topology.host_ram().id(), cpu.id())
    );
}

#[test]
fn a_machine_has_one_host_ram_store() {
    // A container's cgroup limit is a grant on host RAM, read with its
    // status (the rtx3090 host runs under one, and the cgroup fixtures in
    // `host::linux` read them): the structure is the machine's, and the one
    // host RAM store is what the limit bounds.
    let topology = topology(&rtx3090());
    show("Container (rtx3090)", &topology);
    let host_ram = topology
        .stores()
        .iter()
        .filter(|store| *store.kind() == StoreKind::HostRam)
        .map(|store| store.id())
        .collect::<Vec<_>>();
    assert_eq!(host_ram, [topology.host_ram().id()]);
}

#[test]
fn a_cpu_without_the_instruction_set_floor_is_unavailable_with_its_reason() {
    let mut reports = rtx3090_host();
    // What discovery reads where the CPU reports no native tier.
    reports.cpu.availability = crate::discovery::cpu_availability(None);
    let topology = topology(&reports);
    show("CPU-only VM without AVX", &topology);
    let cpu = topology.resolve(ExecutorSelector::HostCpu).unwrap();
    assert!(matches!(
        cpu.availability(),
        Availability::Unavailable { .. }
    ));
    assert!(topology.access_paths().is_empty());
}

#[test]
fn a_vgpu_slice_is_the_device_its_apis_expose() {
    let slice = rtx3090_uuid();
    let topology = topology(&a10_8q());
    show("A10-8Q vGPU slice", &topology);
    assert_consistent(&topology);
    let own = device_memory(&topology, slice);
    assert_eq!(
        topology
            .store(store_of(&topology, &own))
            .unwrap()
            .capacity(),
        Reported::new(A10_8Q_VULKAN_HEAP, ReportedBy::VulkanHeap)
    );
    let cuda = executor(&topology, BackendName::Cuda, &PhysicalIdentity::Uuid(slice));
    let vulkan = executor(
        &topology,
        BackendName::Vulkan,
        &PhysicalIdentity::Uuid(slice),
    );
    assert_eq!(
        placement(&topology, cuda, StorageKind::Device),
        [(
            own.clone(),
            reported(A10_8Q_CUDA_USABLE, ReportedBy::CudaUsableMemory)
        )]
    );
    assert_eq!(
        placement(&topology, vulkan, StorageKind::Device),
        [(own, reported(A10_8Q_VULKAN_HEAP, ReportedBy::VulkanHeap))]
    );
}

/// A Metal device whose memory is not the host's is unavailable, and places
/// nothing.
#[test]
fn a_metal_device_without_unified_memory_is_unavailable_with_its_reason() {
    let topology = topology(&m4_pro_without_unified_memory());
    show("Discrete Metal GPU", &topology);
    let registry = PhysicalIdentity::Registry(M4_PRO_REGISTRY);
    let metal = executor(&topology, BackendName::Metal, &registry);
    assert!(matches!(
        metal.availability(),
        Availability::Unavailable { .. }
    ));
    assert!(metal.placement(StorageKind::Device).is_empty());
    let cpu = topology.resolve(ExecutorSelector::HostCpu).unwrap();
    assert_eq!(
        placement(&topology, cpu, StorageKind::Device),
        [(StoreKind::HostRam, None)]
    );
}

/// The backend's allocation rule on the recorded layouts: device storage on
/// the heap of the type it allocates from.
#[test]
fn vulkan_device_storage_reads_its_memory_types_heap() {
    let allocates = |report: &VulkanReport| report.memory.allocates.clone();
    // Discrete with ReBAR: both kinds on the device-local heap.
    assert_eq!(
        allocates(&rtx3090_vulkan(rtx3090_uuid())),
        [vec![0], vec![0]]
    );
    // RADV: the device-local heap, then the GTT heap.
    assert_eq!(
        allocates(&strix_halo_linux_vulkan()),
        [vec![1, 0], vec![1, 0]]
    );
}

#[test]
fn integrated_accounting_grants_share_a_route_but_discrete_heaps_are_selectable() {
    let integrated = topology(&strix_halo_windows());
    let gpu = integrated
        .executors()
        .iter()
        .find(|executor| executor.backend() == BackendName::Vulkan)
        .unwrap();
    let routes = gpu
        .placement(StorageKind::Device)
        .iter()
        .map(|id| integrated.access_path(*id).unwrap().route())
        .collect::<Vec<_>>();
    assert_eq!(routes, [AccessRoute::Device, AccessRoute::Device]);
    let mut reports = strix_halo_windows();
    reports.vulkan = vec![rtx3090_vulkan(rtx3090_uuid())];
    let discrete = topology(&reports);
    let vulkan = discrete.access_paths().iter().filter(|path| {
        discrete.executor(path.executor()).unwrap().backend() == BackendName::Vulkan
    });
    for path in vulkan {
        assert!(matches!(path.route(), AccessRoute::VulkanHeap(_)));
    }
}

fn gathered(reports: Reports) -> Gathered {
    Gathered {
        reports,
        natives: Natives::default(),
    }
}

/// Every identifier `topology` holds.
fn identifiers(topology: &DeviceTopology) -> Vec<Revision> {
    let mut revisions = Vec::new();
    for device in topology.physical_devices() {
        revisions.push(device.id().revision());
    }
    for store in topology.stores() {
        revisions.push(store.id().revision());
        if let StoreKind::DeviceMemory(device) = store.kind() {
            revisions.push(device.revision());
        }
    }
    for executor in topology.executors() {
        revisions.push(executor.id().revision());
        revisions.push(executor.physical().revision());
        for kind in StorageKind::ALL {
            revisions.extend(executor.placement(kind).iter().map(|id| id.revision()));
        }
    }
    for path in topology.access_paths() {
        revisions.extend([
            path.id().revision(),
            path.executor().revision(),
            path.store().revision(),
        ]);
    }
    for path in topology.transfer_paths() {
        revisions.extend([
            path.id().revision(),
            path.from().revision(),
            path.to().revision(),
            path.by().revision(),
        ]);
    }
    revisions
}

#[test]
fn a_changed_refresh_leaves_no_identifier_at_its_old_revision() {
    // A BIOS change of the carve-out: RADV's device-local heap grows by it.
    let strix = |carve_out: u64| {
        let mut reports = strix_halo_linux();
        reports.vulkan[0].memory.heaps[1].bytes += carve_out;
        reports.volumes.push(rtx3090_volume("/models"));
        gathered(reports)
    };
    let inventory = Inventory::from_gathered(vec![PathBuf::from("/models")], strix(0));
    let before = inventory.topology();
    // The same machine keeps its revision and identifiers.
    let same = inventory.advance(strix(0));
    assert!(Arc::ptr_eq(&before, &same));
    let rebuilt = topology(&strix(0).reports);
    assert_ne!(rebuilt.revision(), before.revision());
    assert_eq!(rebuilt.normalized(), before.normalized());
    let after = inventory.advance(strix(1 << 30));
    assert_ne!(after.revision(), before.revision());
    assert_ne!(after.normalized(), before.normalized());
    assert!(identifiers(&after)
        .into_iter()
        .all(|revision| revision == after.revision()));
    assert!(identifiers(&before)
        .into_iter()
        .all(|revision| revision == before.revision()));
    let gpu = PhysicalIdentity::Uuid(uuid(STRIX_HALO_LINUX_UUID));
    let old = executor(&before, BackendName::Vulkan, &gpu);
    assert!(after.executor(old.id()).is_none());
    assert!(after.store(before.host_ram().id()).is_none());
    assert!(after
        .access_path(old.placement(StorageKind::Device)[0])
        .is_none());
    assert_eq!(
        volume(&after, Path::new("/models")).unwrap().revision(),
        after.revision()
    );
    assert_consistent(&after);
}

/// CUDA and Vulkan views reporting one device UUID are one physical device,
/// each keeping its own selector.
#[test]
fn views_sharing_a_device_uuid_are_one_physical_device() {
    let topology = topology(&rtx3090());
    let gpu = rtx3090_uuid();
    let cuda = topology
        .resolve(ExecutorSelector::Cuda { uuid: gpu })
        .unwrap();
    let vulkan = topology
        .resolve(ExecutorSelector::Vulkan { uuid: gpu })
        .unwrap();
    assert_eq!(cuda.physical(), vulkan.physical());
    assert_eq!(
        topology
            .physical_device(cuda.physical())
            .unwrap()
            .identity(),
        &PhysicalIdentity::Uuid(gpu)
    );
    assert_eq!(topology.physical_devices().len(), 2);
    for executor in topology.executors() {
        assert_eq!(
            executor.selector().to_string().parse(),
            Ok(executor.selector())
        );
    }
}

/// Whose memory a device's is, discovery decides once from every view's
/// report: views that disagree leave it undecided and make every executor
/// of the device unavailable, with the reason.
#[test]
fn views_that_disagree_on_whose_memory_it_is_make_the_device_unavailable() {
    let mut reports = gb10();
    let topology_agreed = topology(&reports);
    let gpu = PhysicalIdentity::Uuid(GB10_UUID);
    let device = |topology: &DeviceTopology| {
        let cuda = executor(topology, BackendName::Cuda, &gpu);
        topology.physical_device(cuda.physical()).unwrap().memory()
    };
    assert_eq!(device(&topology_agreed), DeviceMemory::Host);
    reports.vulkan[0].memory_is_host = Some(false);
    let disputed = topology(&reports);
    assert_eq!(device(&disputed), DeviceMemory::Undecided);
    for backend in [BackendName::Cuda, BackendName::Vulkan] {
        let executor = executor(&disputed, backend, &gpu);
        assert!(matches!(
            executor.availability(),
            Availability::Unavailable { reason } if reason.contains("differently")
        ));
        assert!(executor.placement(StorageKind::Device).is_empty());
    }
}

/// A device none of whose views states whose memory it is stays undecided,
/// and its executors say that none states it rather than that they disagree.
#[test]
fn a_device_no_view_states_the_memory_of_is_undecided_for_that_reason() {
    let mut reports = gb10();
    reports.cuda.clear();
    reports.vulkan[0].memory_is_host = None;
    let topology = topology(&reports);
    let gpu = PhysicalIdentity::Uuid(GB10_UUID);
    let vulkan = executor(&topology, BackendName::Vulkan, &gpu);
    assert_eq!(
        topology
            .physical_device(vulkan.physical())
            .unwrap()
            .memory(),
        DeviceMemory::Undecided
    );
    assert!(matches!(
        vulkan.availability(),
        Availability::Unavailable { reason } if reason.contains("no backend reports")
    ));
    assert!(vulkan.placement(StorageKind::Device).is_empty());
}

/// The dedicated memory the OS reports for an adapter applies to every view
/// of a device whose memory is the host's, CUDA's included, matched by its
/// LUID: it is placed first, under the lesser of it and the view's own
/// total, then host RAM under what the total leaves.
#[test]
fn every_view_of_a_device_whose_memory_is_the_hosts_places_its_dedicated_memory_first() {
    let mut reports = gb10();
    let luid = [0x42, 0x01, 0, 0, 0, 0, 0, 0];
    let dedicated = 32 * GB;
    reports
        .adapters
        .push(crate::discovery::AdapterReport { luid, dedicated });
    reports.cuda[0].luid = Some(luid);
    let topology = topology(&reports);
    assert_consistent(&topology);
    let own = device_memory(&topology, GB10_UUID);
    let gpu = PhysicalIdentity::Uuid(GB10_UUID);
    let cuda = executor(&topology, BackendName::Cuda, &gpu);
    let vulkan = executor(&topology, BackendName::Vulkan, &gpu);
    assert_eq!(
        placement(&topology, cuda, StorageKind::Device),
        [
            (
                own.clone(),
                reported(dedicated, ReportedBy::DxgiDedicatedMemory)
            ),
            (
                StoreKind::HostRam,
                reported(
                    GB10_CUDA_TOTAL - dedicated,
                    ReportedBy::CudaTotalBeyondDedicated
                )
            ),
        ]
    );
    assert_eq!(
        placement(&topology, vulkan, StorageKind::Device),
        [
            (own, reported(dedicated, ReportedBy::DxgiDedicatedMemory)),
            (
                StoreKind::HostRam,
                reported(
                    GB10_VULKAN_HEAP - dedicated,
                    ReportedBy::VulkanHeapsBeyondDedicated
                )
            ),
        ]
    );
}

/// WDDM issues an adapter a new LUID each boot. The LUID only matches the
/// adapter's dedicated memory, so a reboot changes neither the graph nor a
/// measurement key.
#[test]
fn a_new_adapter_luid_changes_no_identity() {
    let reboot = |luid: [u8; 8]| {
        let mut reports = strix_halo_windows();
        reports.adapters[0].luid = luid;
        reports.vulkan[0].memory.luid = Some(luid);
        topology(&reports)
    };
    let before = reboot(STRIX_HALO_WINDOWS_LUID);
    let after = reboot([0x7f, 0x42, 0x02, 0, 0, 0, 0, 0]);
    assert_eq!(before.normalized(), after.normalized());
    let gpu = uuid(STRIX_HALO_WINDOWS_UUID);
    let identity = |topology: &DeviceTopology| {
        let vulkan = topology
            .resolve(ExecutorSelector::Vulkan { uuid: gpu })
            .unwrap();
        topology
            .physical_device(vulkan.physical())
            .unwrap()
            .identity()
            .clone()
    };
    assert_eq!(identity(&before), PhysicalIdentity::Uuid(gpu));
    assert_eq!(identity(&after), PhysicalIdentity::Uuid(gpu));
}

/// The CPU's identity is its reported hardware, not only the OS build: a
/// change of either is a new executor identity, and so a new measurement
/// key.
#[test]
fn cpu_hardware_changes_change_the_cpu_executor_identity() {
    let mut reports = m4_pro();
    let identity = |reports: &Reports| {
        let topology = topology(reports);
        let cpu = topology.resolve(ExecutorSelector::HostCpu).unwrap().id();
        topology.executor_identity(cpu).unwrap()
    };
    let before = identity(&reports);
    assert_eq!(identity(&reports), before);
    reports.cpu.identity.push_str(";processor-revision-changed");
    assert_ne!(identity(&reports), before);
}

/// A device's cache is what an API reports for it: the host's is its CPU's
/// last-level total, a GPU's is its CUDA L2 for its CUDA and Vulkan views
/// alike, and a device no API reports one for has none.
#[test]
fn a_device_carries_the_cache_its_apis_report() {
    let topology = topology(&gb10());
    let cache = |identity: &PhysicalIdentity| {
        topology
            .physical_devices()
            .iter()
            .find(|device| device.identity() == identity)
            .unwrap()
            .cache()
    };
    let host = topology.physical_devices()[0].identity().clone();
    assert!(matches!(host, PhysicalIdentity::Host { .. }));
    assert_eq!(
        cache(&host),
        reported(crate::fixtures::GB10_CPU_CACHE, ReportedBy::LinuxSysfsCache)
    );
    assert_eq!(
        cache(&PhysicalIdentity::Uuid(GB10_UUID)),
        reported(crate::fixtures::GB10_CUDA_L2, ReportedBy::CudaL2Cache)
    );
    let mut vulkan_only = gb10();
    vulkan_only.cuda.clear();
    let topology = crate::fixtures::topology(&vulkan_only);
    let gpu = topology
        .physical_devices()
        .iter()
        .find(|device| *device.identity() == PhysicalIdentity::Uuid(GB10_UUID))
        .unwrap();
    assert_eq!(gpu.cache(), None);
}

/// A driver update changes the identity of that driver's executor only.
#[test]
fn a_driver_update_changes_only_its_executor_identity() {
    let gpu = PhysicalIdentity::Uuid(rtx3090_uuid());
    let identities = |driver: &str| {
        let topology = rtx3090_topology(driver);
        [BackendName::Cpu, BackendName::Cuda, BackendName::Vulkan].map(|backend| {
            let executor = match backend {
                BackendName::Cpu => topology.resolve(ExecutorSelector::HostCpu).unwrap(),
                backend => executor(&topology, backend, &gpu),
            };
            topology.executor_identity(executor.id()).unwrap()
        })
    };
    let [cpu, cuda, vulkan] = identities(RTX3090_VULKAN_DRIVER);
    let [cpu_after, cuda_after, vulkan_after] = identities("vulkan driver 4 version 616.92.0");
    assert_eq!((cpu, cuda), (cpu_after, cuda_after));
    assert_ne!(vulkan, vulkan_after);
    assert_eq!(vulkan_after.driver, "vulkan driver 4 version 616.92.0");
    assert_eq!(vulkan.physical, gpu);
}

/// The rtx3090 machine with Vulkan driver `driver`.
fn rtx3090_topology(driver: &str) -> Arc<DeviceTopology> {
    let mut reports = rtx3090();
    reports.vulkan[0].driver = driver.into();
    Arc::new(topology(&reports))
}

/// The machine this runs on: its reports, the topology they make, and one
/// sample of it.
#[test]
fn the_machine_is_discovered_and_sampled() {
    let directory = std::env::temp_dir();
    let gathered = crate::discovery::gather(&[directory.clone()]).unwrap();
    eprintln!("{:#?}", gathered.reports);
    let inventory = Inventory::from_gathered(vec![directory.clone()], gathered);
    let topology = inventory.topology();
    show("This machine", &topology);
    assert_consistent(&topology);
    let cpu = topology.resolve(ExecutorSelector::HostCpu).unwrap();
    if *cpu.availability() == Availability::Available {
        assert_eq!(
            placement(&topology, cpu, StorageKind::Device),
            [(StoreKind::HostRam, None)]
        );
    }
    let served = volume(&topology, &directory).unwrap();
    let sample = inventory.sample(&topology);
    assert!(sample.host_ram.as_ref().unwrap().headroom.value > 0);
    assert_eq!(sample.volumes.len(), 1);
    assert_eq!(sample.volumes[0].0, served);
    assert!(sample.volumes[0].1.as_ref().unwrap().value > 0);
    // Every grant whose status needs no opened executor is read.
    assert_eq!(sample.grants.len(), topology.grant_sources().count());
    assert!(sample.grants.iter().all(|(_, left)| left.is_ok()));
    eprintln!("{sample:#?}");
    assert_eq!(inventory.refresh().unwrap().revision(), topology.revision());
}
