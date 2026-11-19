//! Topologies built from recorded driver reports for each machine shape,
//! their placements and grants, revisions, and identities. The
//! recorded machines and their sources are `crate::fixtures`'.

use crate::fixtures::{
    a10_8q, assert_consistent, executor, gb10, linux_host, lunar_lake, m4_pro,
    m4_pro_without_unified_memory, placement, rtx3090, rtx3090_cuda, rtx3090_granting, rtx3090_host,
    rtx3090_most_granted, rtx3090_uuid, rtx3090_volume, rtx3090_vulkan, ryzen_igpu_and_rtx4070,
    show, strix_halo_linux, strix_halo_windows, topology, uuid, volume, A10_8Q_CUDA_USABLE, A10_8Q_VULKAN_HEAP, GB,
    GB10_CUDA_TOTAL, GB10_UUID, GB10_VULKAN_HEAP, LUNAR_LAKE_HEAP, LUNAR_LAKE_UUID,
    M4_PRO_REGISTRY, M4_PRO_WORKING_SET, RTX3090_CUDA_TOTAL, RTX3090_CUDA_USABLE, RTX3090_HOST,
    RTX3090_VULKAN_DRIVER, RTX3090_VULKAN_HEAP, RTX4070_HEAP, RTX4070_UUID, RYZEN_IGPU_HEAPS,
    RYZEN_IGPU_UUID, STRIX_HALO_LINUX_HEAPS, STRIX_HALO_LINUX_UUID, STRIX_HALO_WINDOWS_DEDICATED,
    STRIX_HALO_WINDOWS_HEAPS, STRIX_HALO_WINDOWS_LUID, STRIX_HALO_WINDOWS_UUID,
};
use crate::model::{
    Availability, CacheSize, DeviceMemory, DeviceTopology, Grant, GrantKind, PhysicalIdentity,
    Reported, ReportedBy, StoreId, StoreKind,
};
use crate::placement::Contract;
use crate::processors::HostNode;
use crate::reports::Reports;
use crate::selector::ExecutorSelector;
use crate::system::{FormFactor, HardwareIdentity, SystemIdentity};
use seismic_lang::registry::BackendName;
use std::path::Path;
use std::sync::Arc;

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

fn grant(value: u64, by: ReportedBy, kind: GrantKind) -> Option<Grant> {
    Some(Grant {
        bytes: Reported::new(value, by),
        kind,
    })
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
    let working_set = grant(
        M4_PRO_WORKING_SET,
        ReportedBy::MetalRecommendedWorkingSet,
        GrantKind::Wired,
    );
    for kind in Contract::ALL {
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

/// A machine granting a backend's device storage other bytes than it
/// recorded grants them on that backend's device path, the grant stable fit
/// plans against, and leaves the other backend's as recorded: a test that
/// plans a card at a grant plans at it.
#[test]
fn rtx3090_granting_grants_the_backends_device_path_alone() {
    let gpu = PhysicalIdentity::Uuid(rtx3090_uuid());
    let grant_of = |topology: &DeviceTopology, backend| {
        placement(topology, executor(topology, backend, &gpu), Contract::DeviceOnly)
            .into_iter()
            .map(|(_, grant)| grant.map(|grant| grant.bytes.value))
            .collect::<Vec<_>>()
    };
    for (backend, other, recorded) in [
        (BackendName::Cuda, BackendName::Vulkan, RTX3090_VULKAN_HEAP),
        (BackendName::Vulkan, BackendName::Cuda, RTX3090_CUDA_USABLE),
    ] {
        for grant in [20 * GB, rtx3090_most_granted(backend) - GB] {
            let topology = topology(&rtx3090_granting(backend, grant));
            assert_consistent(&topology);
            assert_eq!(grant_of(&topology, backend), [Some(grant)], "{backend:?} at {grant}");
            assert_eq!(grant_of(&topology, other), [Some(recorded)], "{other:?} beside {backend:?} at {grant}");
        }
    }
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
        placement(&topology, cuda, Contract::DeviceOnly),
        [(
            own.clone(),
            grant(
                RTX3090_CUDA_USABLE,
                ReportedBy::CudaUsableMemory,
                GrantKind::DeviceOpened,
            )
        )]
    );
    for kind in [Contract::Upload, Contract::Shared] {
        assert_eq!(
            placement(&topology, cuda, kind),
            [(StoreKind::HostRam, None)]
        );
    }
    assert_eq!(
        cuda.paths(Contract::Upload),
        cuda.paths(Contract::Shared)
    );
    // ReBAR: the device-local host-visible type is on the 24 GiB heap, so
    // upload storage shares device storage's path; host storage is host RAM
    // whatever upload storage maps to.
    for kind in [Contract::DeviceOnly, Contract::Upload] {
        assert_eq!(
            placement(&topology, vulkan, kind),
            [(
                own.clone(),
                grant(
                    RTX3090_VULKAN_HEAP,
                    ReportedBy::VulkanHeap,
                    GrantKind::DeviceUnopened,
                )
            )]
        );
    }
    assert_eq!(
        vulkan.paths(Contract::DeviceOnly),
        vulkan.paths(Contract::Upload)
    );
    assert_eq!(
        placement(&topology, vulkan, Contract::Shared),
        [(StoreKind::HostRam, None)]
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
    for kind in Contract::ALL {
        // CUDA's grant is its total, which on a GB10 is host RAM's capacity.
        assert_eq!(
            placement(&topology, cuda, kind),
            [(
                StoreKind::HostRam,
                grant(
                    GB10_CUDA_TOTAL,
                    ReportedBy::CudaTotalMemory,
                    GrantKind::HostLimit,
                )
            )]
        );
        assert_eq!(
            placement(&topology, vulkan, kind),
            [(
                StoreKind::HostRam,
                grant(
                    GB10_VULKAN_HEAP,
                    ReportedBy::VulkanHeaps,
                    GrantKind::HostBudget,
                )
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
    for kind in Contract::ALL {
        assert_eq!(
            placement(&topology, vulkan, kind),
            [(
                StoreKind::HostRam,
                grant(
                    STRIX_HALO_LINUX_HEAPS.iter().sum(),
                    ReportedBy::VulkanHeaps,
                    GrantKind::HostBudget,
                )
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
    for kind in Contract::ALL {
        assert_eq!(
            placement(&topology, vulkan, kind),
            [
                (
                    radeon.clone(),
                    grant(
                        STRIX_HALO_WINDOWS_DEDICATED,
                        ReportedBy::DxgiDedicatedMemory,
                        GrantKind::HostBudget,
                    )
                ),
                (
                    StoreKind::HostRam,
                    grant(
                        total - STRIX_HALO_WINDOWS_DEDICATED,
                        ReportedBy::VulkanHeapsBeyondDedicated,
                        GrantKind::HostBudget,
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
        placement(&topology, vulkan, Contract::DeviceOnly),
        [(
            StoreKind::HostRam,
            grant(
                LUNAR_LAKE_HEAP,
                ReportedBy::VulkanHeaps,
                GrantKind::HostBudget,
            )
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
        placement(&topology, on_igpu, Contract::DeviceOnly),
        [(
            StoreKind::HostRam,
            grant(
                RYZEN_IGPU_HEAPS.iter().sum(),
                ReportedBy::VulkanHeaps,
                GrantKind::HostBudget,
            )
        )]
    );
    let on_gpu = executor(&topology, BackendName::Vulkan, &PhysicalIdentity::Uuid(gpu));
    assert_eq!(
        placement(&topology, on_gpu, Contract::DeviceOnly),
        [(
            device_memory(&topology, gpu),
            grant(
                RTX4070_HEAP,
                ReportedBy::VulkanHeap,
                GrantKind::DeviceUnopened,
            )
        )]
    );
}

/// Two RTX 3090s, each the RTX 3090 machine's: each gets its own device
/// memory store, and CUDA and Vulkan place device storage on it.
#[test]
fn two_discrete_gpus_have_separate_device_memory_stores() {
    let (first, second) = (rtx3090_uuid(), uuid("D37CB696763FCACE0D8DCBA6B3D100D1"));
    let mut reports = linux_host(RTX3090_HOST);
    reports.cuda.push(rtx3090_cuda(first));
    reports.cuda.push(rtx3090_cuda(second));
    reports.vulkan.push(rtx3090_vulkan(first));
    reports.vulkan.push(rtx3090_vulkan(second));
    let topology = topology(&reports);
    show("2 x RTX 3090", &topology);
    assert_consistent(&topology);
    assert_eq!(topology.stores().len(), 3);
    for each in [first, second] {
        let own = device_memory(&topology, each);
        for backend in [BackendName::Cuda, BackendName::Vulkan] {
            let executor = executor(&topology, backend, &PhysicalIdentity::Uuid(each));
            assert_eq!(
                placement(&topology, executor, Contract::DeviceOnly)[0].0,
                own
            );
        }
    }
}

/// The native kinds a buffer of `contract` is allocated from.
fn natives(
    topology: &DeviceTopology,
    executor: &crate::Executor,
    contract: Contract,
) -> Vec<u32> {
    executor
        .placement(contract)
        .iter()
        .map(|id| topology.kind(*id).unwrap().native().id)
        .collect()
}

/// The rule's outcomes on the recorded topologies:
/// each contract in each shape, by the backend's kind identifiers.
#[test]
fn the_rule_places_every_contract_as_the_spec_states() {
    let reports = rtx3090();
    let discrete = topology(&reports);
    let uuid = PhysicalIdentity::Uuid(rtx3090_uuid());
    let cuda = executor(&discrete, BackendName::Cuda, &uuid);
    let of = |topology, executor| Contract::ALL.map(|contract| natives(topology, executor, contract));
    assert_eq!(
        of(&discrete, cuda),
        [vec![0], vec![1], vec![1], vec![1]]
    );
    let vulkan = executor(&discrete, BackendName::Vulkan, &uuid);
    assert_eq!(
        of(&discrete, vulkan),
        [vec![0], vec![2], vec![5], vec![5]]
    );
    assert!(discrete.first_store(cuda, Contract::Shared) != discrete.first_store(cuda, Contract::DeviceOnly));
    assert!(!discrete.shared_ranges_lie_in_device_memory(cuda));

    let reports = gb10();
    let unified = topology(&reports);
    let uuid = PhysicalIdentity::Uuid(crate::fixtures::GB10_UUID);
    let cuda = executor(&unified, BackendName::Cuda, &uuid);
    assert_eq!(
        of(&unified, cuda),
        [vec![0], vec![1], vec![1], vec![1]]
    );
    assert!(unified.shared_ranges_lie_in_device_memory(cuda));
    let vulkan = executor(&unified, BackendName::Vulkan, &uuid);
    assert_eq!(
        of(&unified, vulkan),
        [vec![0], vec![2], vec![3], vec![3]]
    );
    assert!(unified.shared_ranges_lie_in_device_memory(vulkan));

    let reports = m4_pro();
    let apple = topology(&reports);
    let metal = apple
        .executors()
        .iter()
        .find(|executor| executor.backend() == BackendName::Metal)
        .unwrap();
    assert_eq!(
        of(&apple, metal),
        [vec![0], vec![0], vec![0], vec![0]]
    );
    assert!(apple.shared_ranges_lie_in_device_memory(metal));
}

/// A device-local mappable heap of at most `WINDOW_BYTES` counts as host
/// RAM: upload storage takes the host heap's first type, not the window.
#[test]
fn a_small_bar_counts_as_host_ram() {
    let mut reports = rtx3090_host();
    let mut vulkan = rtx3090_vulkan(rtx3090_uuid());
    vulkan.memory.heaps.push(crate::VulkanHeap {
        bytes: crate::WINDOW_BYTES,
        device_local: true,
    });
    let window = vulkan.memory.heaps.len() - 1;
    vulkan.kinds[2].memory = crate::KindMemory::Heap { index: window, device_local: true, bytes: crate::WINDOW_BYTES };
    reports.vulkan.push(vulkan);
    let topology = topology(&reports);
    let vulkan = executor(
        &topology,
        BackendName::Vulkan,
        &PhysicalIdentity::Uuid(rtx3090_uuid()),
    );
    assert_eq!(
        natives(&topology, vulkan, Contract::Upload),
        [4]
    );
}

#[test]
fn a_discrete_gpu_has_an_access_path_to_host_ram() {
    let gpu = rtx3090_uuid();
    let mut reports = rtx3090_host();
    reports.cuda.push(rtx3090_cuda(gpu));
    let topology = topology(&reports);
    let cuda = executor(&topology, BackendName::Cuda, &PhysicalIdentity::Uuid(gpu));
    assert!(topology
        .access_paths()
        .iter()
        .any(|path| path.executor() == cuda.id() && path.store() == topology.host_ram().id()));
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
    // status (the RTX 3090 machine runs under one, and the cgroup fixtures in
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
        placement(&topology, cuda, Contract::DeviceOnly),
        [(
            own.clone(),
            grant(
                A10_8Q_CUDA_USABLE,
                ReportedBy::CudaUsableMemory,
                GrantKind::DeviceOpened,
            )
        )]
    );
    assert_eq!(
        placement(&topology, vulkan, Contract::DeviceOnly),
        [(
            own,
            grant(
                A10_8Q_VULKAN_HEAP,
                ReportedBy::VulkanHeap,
                GrantKind::DeviceUnopened,
            )
        )]
    );
}

/// A Metal device whose memory is its own runs as every such device does:
/// device work's own buffers in its private storage, in that memory under
/// the working set Metal reports as its size, and what the host maps in
/// shared storage, in host RAM.
#[test]
fn a_metal_device_without_unified_memory_places_in_its_own_memory() {
    let topology = topology(&m4_pro_without_unified_memory());
    show("Discrete Metal GPU", &topology);
    let registry = PhysicalIdentity::Registry(M4_PRO_REGISTRY);
    let metal = executor(&topology, BackendName::Metal, &registry);
    assert_eq!(*metal.availability(), Availability::Available);
    let own = StoreKind::DeviceMemory(metal.physical());
    assert_eq!(
        placement(&topology, metal, Contract::DeviceOnly),
        [(
            own,
            grant(
                M4_PRO_WORKING_SET,
                ReportedBy::MetalRecommendedWorkingSet,
                GrantKind::DeviceUnopened,
            )
        )]
    );
    for contract in [Contract::Upload, Contract::Shared, Contract::Words] {
        assert_eq!(placement(&topology, metal, contract), [(StoreKind::HostRam, None)], "{contract:?}");
    }
    assert_consistent(&topology);
}

/// An integrated device's kinds are each charged along every path of its
/// placement, the heaps' one total presented across them; a discrete
/// device's kinds each along the path of the heap they lie on, and its host
/// storage's kinds (host-visible types, imported host pages) along its one
/// path to host RAM.
#[test]
fn integrated_kinds_share_their_paths_but_discrete_kinds_lie_on_their_heaps() {
    let integrated = topology(&strix_halo_windows());
    let gpu = integrated
        .executors()
        .iter()
        .find(|executor| executor.backend() == BackendName::Vulkan)
        .unwrap();
    let placement = gpu.paths(Contract::DeviceOnly);
    assert_eq!(placement.len(), 2);
    for id in gpu.placement(Contract::DeviceOnly) {
        assert_eq!(integrated.kind(*id).unwrap().paths(), placement);
    }
    let mut reports = strix_halo_windows();
    reports.vulkan = vec![rtx3090_vulkan(rtx3090_uuid())];
    let discrete = topology(&reports);
    let vulkan = discrete
        .executors()
        .iter()
        .find(|executor| executor.backend() == BackendName::Vulkan)
        .unwrap();
    let paths = |contract| {
        vulkan
            .placement(contract)
            .iter()
            .map(|id| discrete.kind(*id).unwrap().paths().to_vec())
            .collect::<Vec<_>>()
    };
    let [device] = vulkan.paths(Contract::DeviceOnly) else {
        panic!("one device storage path")
    };
    assert_eq!(paths(Contract::DeviceOnly), [vec![*device]]);
    assert_eq!(
        paths(Contract::Upload),
        [vec![*device]],
        "ReBAR: the device-local heap"
    );
    let [host] = vulkan.paths(Contract::Shared) else {
        panic!("one host storage path")
    };
    assert_eq!(paths(Contract::Shared), [vec![*host]]);
    assert_eq!(
        discrete
            .store(discrete.access_path(*host).unwrap().store())
            .unwrap()
            .kind(),
        &StoreKind::HostRam
    );
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

/// A software Vulkan device (lavapipe) beside a GPU's CUDA and Vulkan views
/// is an executor of the host, not a second physical device: the machine
/// has one GPU, and the software device owns no memory of its own.
#[test]
fn a_software_vulkan_device_is_an_executor_of_the_host() {
    let mut reports = rtx3090();
    let software = uuid("6d65736132342e302e340000000000ff");
    let mut lavapipe = crate::fixtures::vulkan(
        software,
        "llvmpipe (LLVM 17.0.6, 256 bits)",
        "vulkan driver MESA_LLVMPIPE version 24.0.9",
        true,
        &[(RTX3090_HOST, true)],
        &[&["DL|HV|HC|C"]],
        1 << 31,
        None,
    );
    lavapipe.kind = crate::model::ExecutorKind::Cpu;
    lavapipe.floor = Err("missing VK_KHR_workgroup_memory_explicit_layout".into());
    reports.vulkan.push(lavapipe);
    let topology = topology(&reports);

    let gpus = topology
        .physical_devices()
        .iter()
        .filter(|physical| !matches!(physical.identity(), PhysicalIdentity::Host { .. }))
        .count();
    assert_eq!(gpus, 1);
    let software = topology
        .resolve(ExecutorSelector::Vulkan { uuid: software })
        .unwrap();
    assert!(matches!(
        topology
            .physical_device(software.physical())
            .unwrap()
            .identity(),
        PhysicalIdentity::Host { .. }
    ));
    assert_eq!(
        topology
            .stores()
            .iter()
            .filter(|store| matches!(store.kind(), StoreKind::DeviceMemory(_)))
            .count(),
        1
    );
    assert_consistent(&topology);
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
        assert!(executor
            .paths(Contract::DeviceOnly)
            .is_empty());
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
    assert!(vulkan.paths(Contract::DeviceOnly).is_empty());
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
        .push(crate::reports::AdapterReport { luid, dedicated });
    reports.cuda[0].luid = Some(luid);
    let topology = topology(&reports);
    assert_consistent(&topology);
    let own = device_memory(&topology, GB10_UUID);
    let gpu = PhysicalIdentity::Uuid(GB10_UUID);
    let cuda = executor(&topology, BackendName::Cuda, &gpu);
    let vulkan = executor(&topology, BackendName::Vulkan, &gpu);
    assert_eq!(
        placement(&topology, cuda, Contract::DeviceOnly),
        [
            (
                own.clone(),
                grant(
                    dedicated,
                    ReportedBy::DxgiDedicatedMemory,
                    GrantKind::HostLimit,
                )
            ),
            (
                StoreKind::HostRam,
                grant(
                    GB10_CUDA_TOTAL - dedicated,
                    ReportedBy::CudaTotalBeyondDedicated,
                    GrantKind::HostLimit,
                )
            ),
        ]
    );
    assert_eq!(
        placement(&topology, vulkan, Contract::DeviceOnly),
        [
            (
                own,
                grant(
                    dedicated,
                    ReportedBy::DxgiDedicatedMemory,
                    GrantKind::HostBudget,
                )
            ),
            (
                StoreKind::HostRam,
                grant(
                    GB10_VULKAN_HEAP - dedicated,
                    ReportedBy::VulkanHeapsBeyondDedicated,
                    GrantKind::HostBudget,
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
/// alike, and a device no API reports one for gets [`CacheSize::UNREPORTED`].
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
    let host = topology.host_device().identity().clone();
    assert!(matches!(host, PhysicalIdentity::Host { .. }));
    assert_eq!(
        cache(&host),
        CacheSize {
            bytes: crate::fixtures::GB10_CPU_CACHE,
            is_last_level: true,
        }
    );
    assert_eq!(
        cache(&PhysicalIdentity::Uuid(GB10_UUID)),
        CacheSize {
            bytes: crate::fixtures::GB10_CUDA_L2,
            is_last_level: false,
        }
    );
    let mac = crate::fixtures::topology(&m4_pro());
    assert_eq!(
        mac.host_device().cache(),
        CacheSize {
            bytes: crate::fixtures::M4_PRO_CPU_CACHE,
            is_last_level: false,
        }
    );
    let mut vulkan_only = gb10();
    vulkan_only.cuda.clear();
    let topology = crate::fixtures::topology(&vulkan_only);
    let gpu = topology
        .physical_devices()
        .iter()
        .find(|device| *device.identity() == PhysicalIdentity::Uuid(GB10_UUID))
        .unwrap();
    assert_eq!(gpu.cache(), CacheSize::UNREPORTED);
}

/// Static host metadata is carried on `topology.host()`, and `topology.host_device()`
/// returns the host physical device.
#[test]
fn host_metadata_and_host_device_are_exposed_on_topology() {
    let mut reports = m4_pro();
    reports.system = SystemIdentity {
        identity: Some(HardwareIdentity {
            manufacturer: "Apple".into(),
            model: "Mac16,7".into(),
            family: Some("MacBook Pro".into()),
            version: None,
        }),
        form_factor: FormFactor::Portable,
        name: Some("MacBook Pro".into()),
    };
    let topology = topology(&reports);
    let host = topology.host();
    assert_eq!(
        host.identity,
        format!("{};ram={}", reports.cpu.identity, reports.host.value)
    );
    assert_eq!(host.cpu_model, "Apple M4 Pro");
    assert_eq!(host.os_build, crate::fixtures::M4_PRO_OS);
    assert_eq!(host.page_bytes, 16384);
    assert_eq!(host.processors, reports.processors);
    assert_eq!(host.system, reports.system);

    let device = topology.host_device();
    assert_eq!(device, &topology.physical_devices()[0]);
    assert_eq!(
        device.identity(),
        &PhysicalIdentity::Host {
            machine: host.identity.clone(),
        }
    );
    assert_eq!(device.memory(), DeviceMemory::Host);
    assert_eq!(device.host_node(), HostNode::Any);
}

/// Every backend and memory-ownership combination receives its semantic `GrantKind`.
#[test]
fn grant_kind_matches_each_backend_and_memory_ownership() {
    let grant_kind = |topology: &DeviceTopology, backend: BackendName, identity: &PhysicalIdentity| {
        let exec = executor(topology, backend, identity);
        placement(topology, exec, Contract::DeviceOnly)
            .into_iter()
            .map(|(_, grant)| grant.unwrap().kind)
            .collect::<Vec<_>>()
    };

    let metal_id = PhysicalIdentity::Registry(M4_PRO_REGISTRY);
    assert_eq!(
        grant_kind(&topology(&m4_pro()), BackendName::Metal, &metal_id),
        [GrantKind::Wired]
    );
    assert_eq!(
        grant_kind(
            &topology(&m4_pro_without_unified_memory()),
            BackendName::Metal,
            &metal_id,
        ),
        [GrantKind::DeviceUnopened]
    );

    let gb10_id = PhysicalIdentity::Uuid(GB10_UUID);
    let gb10_topology = topology(&gb10());
    assert_eq!(
        grant_kind(&gb10_topology, BackendName::Cuda, &gb10_id),
        [GrantKind::HostLimit]
    );
    assert_eq!(
        grant_kind(&gb10_topology, BackendName::Vulkan, &gb10_id),
        [GrantKind::HostBudget]
    );

    let rtx_id = PhysicalIdentity::Uuid(rtx3090_uuid());
    let rtx_topology = topology(&rtx3090());
    assert_eq!(
        grant_kind(&rtx_topology, BackendName::Cuda, &rtx_id),
        [GrantKind::DeviceOpened]
    );
    assert_eq!(
        grant_kind(&rtx_topology, BackendName::Vulkan, &rtx_id),
        [GrantKind::DeviceUnopened]
    );
}

/// Host NUMA affinity reported on CUDA or Vulkan views propagates onto the physical device.
#[test]
fn host_node_propagates_from_cuda_and_vulkan_reports() {
    let gpu = PhysicalIdentity::Uuid(rtx3090_uuid());
    let node_of = |reports: &Reports| {
        let topology = topology(reports);
        topology
            .physical_devices()
            .iter()
            .find(|device| device.identity() == &gpu)
            .unwrap()
            .host_node()
    };

    let mut reports = rtx3090();
    assert_eq!(node_of(&reports), HostNode::Any);

    reports.cuda[0].host_node = HostNode::Node(1);
    assert_eq!(node_of(&reports), HostNode::Node(1));

    reports.cuda[0].host_node = HostNode::Any;
    reports.vulkan[0].host_node = HostNode::Node(2);
    assert_eq!(node_of(&reports), HostNode::Node(2));

    reports.cuda.clear();
    assert_eq!(node_of(&reports), HostNode::Node(2));
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
