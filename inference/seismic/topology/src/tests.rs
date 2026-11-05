//! Topologies built from recorded driver reports for each machine shape,
//! their placements and grants, revisions, and identities. The
//! recorded machines and their sources are `crate::fixtures`'.

use crate::fixtures::{
    a10_8q, assert_consistent, executor, gb10, linux_host, lunar_lake, m4_pro,
    m4_pro_without_unified_memory, placement, rtx3090, rtx3090_cuda, rtx3090_host, rtx3090_uuid,
    rtx3090_volume, rtx3090_vulkan, ryzen_igpu_and_rtx4070, show, strix_halo_linux,
    strix_halo_windows, topology, uuid, volume, A10_8Q_CUDA_USABLE, A10_8Q_VULKAN_HEAP, GB,
    GB10_CUDA_TOTAL, GB10_UUID, GB10_VULKAN_HEAP, LUNAR_LAKE_HEAP, LUNAR_LAKE_UUID,
    M4_PRO_REGISTRY, M4_PRO_WORKING_SET, RTX3090_CUDA_TOTAL, RTX3090_CUDA_USABLE, RTX3090_HOST,
    RTX3090_VULKAN_DRIVER, RTX3090_VULKAN_HEAP, RTX4070_HEAP, RTX4070_UUID, RYZEN_IGPU_HEAPS,
    RYZEN_IGPU_UUID, STRIX_HALO_LINUX_HEAPS, STRIX_HALO_LINUX_UUID, STRIX_HALO_WINDOWS_DEDICATED,
    STRIX_HALO_WINDOWS_HEAPS, STRIX_HALO_WINDOWS_LUID, STRIX_HALO_WINDOWS_UUID,
};
use crate::model::{
    Availability, DeviceMemory, DeviceTopology, PhysicalIdentity, Reported, ReportedBy,
    Reservation, ReservedRecords, StoreId, StoreKind,
};
use crate::placement::{Contract, Placing, ReservedContract};
use crate::reports::Reports;
use crate::selector::ExecutorSelector;
use seismic_lang::registry::BackendName;
use std::num::NonZeroU64;
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

fn reported(value: u64, by: ReportedBy) -> Option<Reported<u64>> {
    Some(Reported::new(value, by))
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

/// A backing past the bytes a `u64` counts is none, as is the charge
/// holding it, where the bytes themselves fit.
#[test]
fn a_backing_past_u64_is_none() {
    let reservation = crate::Reservation {
        granule: 1 << 21,
        host_addressed: true,
        chunks: None,
    };
    let last = u64::MAX - (u64::MAX % reservation.granule);
    assert_eq!(reservation.backing(last), Some(last));
    assert_eq!(reservation.backing(last + 1), None);
    assert_eq!(
        reservation.charge(ReservedRecords::of(NonZeroU64::MIN), 1, last + 1),
        None
    );
}

/// A chunked reservation's chunk is the least of whole records and whole
/// chunk units no smaller than its granule, so its backing holds less than
/// one such chunk past the committed records. At the M4 Pro's Metal
/// reservation, for the records Qwen3.6-35B-A3B Q4 banks (gate and up
/// Q4_K, down Q5_K and Q6_K as Metal holds them), each chunk is three or
/// four records of about 2.3 MB, where the least multiple of granule and
/// record was 32 or 64 records of up to 54 MiB.
#[test]
fn a_chunked_reservation_backs_the_least_chunk_of_whole_records() {
    let topology = topology(&m4_pro());
    let metal = executor(
        &topology,
        BackendName::Metal,
        &PhysicalIdentity::Registry(M4_PRO_REGISTRY),
    );
    let reservation = metal.reserving(ReservedContract::DeviceOnly).unwrap().unwrap();
    let unit = reservation.chunks.unwrap().unit;
    for (record, records) in [(589_824u64, 4u64), (753_664, 3), (884_736, 3)] {
        let record = NonZeroU64::new(record).unwrap();
        let chunks = reservation.of_records(ReservedRecords::of(record));
        let record = record.get();
        let chunk = chunks.granule;
        assert_eq!(chunk, records * record, "{record}");
        assert_eq!(
            (chunk % unit, chunk >= reservation.granule),
            (0, true),
            "{record}"
        );
        // One record fewer is under the granule.
        assert!(chunk - record < reservation.granule, "{record}");
        // The rule is its own fixed point.
        assert_eq!(
            chunks.of_records(ReservedRecords::of(NonZeroU64::new(record).unwrap())),
            chunks
        );
        // 238 slots of 37 layers' records are backed by less than a chunk
        // past them.
        let bytes = 238 * 37 * record;
        assert!(chunks.backing(bytes).unwrap() - bytes < chunk, "{record}");
    }
    // A record of whole granules is its own chunk.
    let granules = NonZeroU64::new(3 * reservation.granule).unwrap();
    assert_eq!(
        reservation
            .of_records(ReservedRecords::of(granules))
            .granule,
        granules.get()
    );
}

/// A range whose records carry another form's larger views (a conveyor's
/// stored records packed in a bank's) is chunked so each view lies whole in
/// a chunk: the least chunk of whole records and units no smaller than the
/// largest view, which the packing then admits.
#[test]
fn a_chunk_holds_the_largest_view_its_records_carry() {
    let topology = topology(&m4_pro());
    let metal = executor(
        &topology,
        BackendName::Metal,
        &PhysicalIdentity::Registry(M4_PRO_REGISTRY),
    );
    let reservation = metal.reserving(ReservedContract::DeviceOnly).unwrap().unwrap();
    let record = NonZeroU64::new(2_359_296).unwrap();
    let view = NonZeroU64::new(2_752_512).unwrap();
    let alone = reservation.of_records(ReservedRecords::of(record));
    assert_eq!(
        alone.packing(view.get()),
        None,
        "a chunk of one record holds no larger view"
    );
    let carrying = reservation.of_records(ReservedRecords::of(record).carrying(view));
    assert_eq!(carrying.granule, 2 * record.get());
    assert!(carrying.packing(view.get()).is_some());
    // A view no larger than the granule leaves the chunk as it was.
    let small = NonZeroU64::new(4096).unwrap();
    assert_eq!(
        reservation.of_records(ReservedRecords::of(record).carrying(small)),
        alone
    );
}

/// Records pack only where a chunk holds one whole: a record of no bytes,
/// or one larger than a chunk, has no packing.
#[test]
fn only_a_record_a_chunk_holds_whole_packs() {
    let topology = topology(&m4_pro());
    let metal = executor(
        &topology,
        BackendName::Metal,
        &PhysicalIdentity::Registry(M4_PRO_REGISTRY),
    );
    let chunks = metal
        .reserving(ReservedContract::DeviceOnly)
        .unwrap()
        .unwrap()
        .of_records(ReservedRecords::of(NonZeroU64::new(589_824).unwrap()));
    assert_eq!(
        (chunks.packing(0), chunks.packing(chunks.granule + 1)),
        (None, None)
    );
    let packing = chunks.packing(589_824).unwrap();
    // Four records fill the chunk; the fifth starts the next.
    assert_eq!(
        (packing.end(0, 4), packing.at(0, 4)),
        (chunks.granule, chunks.granule)
    );
    // A record that does not fit the rest of a chunk starts the next.
    assert_eq!(packing.at(chunks.granule - 1, 0), chunks.granule);
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
            reported(RTX3090_CUDA_USABLE, ReportedBy::CudaUsableMemory)
        )]
    );
    for kind in [Contract::Upload, Contract::Shared] {
        assert_eq!(
            placement(&topology, cuda, kind),
            [(StoreKind::HostRam, None)]
        );
    }
    assert_eq!(
        cuda.paths(Placing::Fixed(Contract::Upload)),
        cuda.paths(Placing::Fixed(Contract::Shared))
    );
    // ReBAR: the device-local host-visible type is on the 24 GiB heap, so
    // upload storage shares device storage's path; host storage is host RAM
    // whatever upload storage maps to.
    for kind in [Contract::DeviceOnly, Contract::Upload] {
        assert_eq!(
            placement(&topology, vulkan, kind),
            [(
                own.clone(),
                reported(RTX3090_VULKAN_HEAP, ReportedBy::VulkanHeap)
            )]
        );
    }
    assert_eq!(
        vulkan.paths(Placing::Fixed(Contract::DeviceOnly)),
        vulkan.paths(Placing::Fixed(Contract::Upload))
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
    for kind in Contract::ALL {
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
    for kind in Contract::ALL {
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
        placement(&topology, vulkan, Contract::DeviceOnly),
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
        placement(&topology, on_igpu, Contract::DeviceOnly),
        [(
            StoreKind::HostRam,
            reported(RYZEN_IGPU_HEAPS.iter().sum(), ReportedBy::VulkanHeaps)
        )]
    );
    let on_gpu = executor(&topology, BackendName::Vulkan, &PhysicalIdentity::Uuid(gpu));
    assert_eq!(
        placement(&topology, on_gpu, Contract::DeviceOnly),
        [(
            device_memory(&topology, gpu),
            reported(RTX4070_HEAP, ReportedBy::VulkanHeap)
        )]
    );
}

/// Two RTX 3090s, each the RTX 3090 machine's, with and without the peer
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
                    placement(&topology, executor, Contract::DeviceOnly)[0].0,
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

/// A reservation granule discovery could not read is the executor's
/// reservation state, with why, not a storage kind it reserves none of;
/// the other kind's stays as recorded.
#[test]
fn an_unread_reservation_granule_is_unreservable_with_its_reason() {
    let gpu = rtx3090_uuid();
    let mut reports = rtx3090_host();
    let mut cuda = rtx3090_cuda(gpu, Vec::new());
    let host = cuda
        .kinds
        .iter()
        .find(|kind| kind.id == 3)
        .unwrap()
        .reservation
        .clone();
    let unread = crate::Unreservable::Unread("the granule read failed".into());
    cuda.kinds[0].reservation = Err(unread.clone());
    reports.cuda.push(cuda);
    let topology = topology(&reports);
    let cuda = executor(&topology, BackendName::Cuda, &PhysicalIdentity::Uuid(gpu));
    assert_eq!(
        cuda.reserving(ReservedContract::DeviceOnly),
        Err(&unread)
    );
    assert_eq!(cuda.reserving(ReservedContract::Shared), host.as_ref().copied());
}

/// Why a Vulkan device reserves no device or host storage (one queue, no
/// host import) is its executor's reservation state, as its report says.
#[test]
fn a_vulkan_device_says_why_it_reserves_nothing() {
    let mut reports = rtx3090_host();
    let mut vulkan = rtx3090_vulkan(rtx3090_uuid());
    for kind in &mut vulkan.kinds {
        match kind.id {
            crate::fixtures::VULKAN_IMPORTED_KIND => {
                kind.reservation = Err(crate::Unreservable::NoHostImport)
            }
            _ if kind.reservation != Ok(None) => kind.reservation = Err(crate::Unreservable::OneQueue { queues: 1 }),
            _ => {}
        }
    }
    reports.vulkan.push(vulkan);
    let topology = topology(&reports);
    let vulkan = executor(
        &topology,
        BackendName::Vulkan,
        &PhysicalIdentity::Uuid(rtx3090_uuid()),
    );
    assert_eq!(vulkan.reserving(ReservedContract::DeviceOnly), Err(&crate::Unreservable::OneQueue { queues: 1 }));
    assert_eq!(vulkan.reserving(ReservedContract::Shared), Err(&crate::Unreservable::NoHostImport));
}

/// The native kinds a buffer placed as `placing` is allocated from.
fn natives(
    topology: &DeviceTopology,
    executor: &crate::Executor,
    placing: Placing,
) -> Vec<u32> {
    executor
        .placement(placing)
        .iter()
        .map(|id| topology.kind(*id).unwrap().native().id)
        .collect()
}

/// The rule's outcomes on the recorded topologies:
/// each contract in each shape, by the backend's kind identifiers.
#[test]
fn the_rule_places_every_contract_as_the_spec_states() {
    use ReservedContract::{DeviceOnly, Shared};
    let imported = crate::fixtures::VULKAN_IMPORTED_KIND;
    let reports = rtx3090();
    let discrete = topology(&reports);
    let uuid = PhysicalIdentity::Uuid(rtx3090_uuid());
    let cuda = executor(&discrete, BackendName::Cuda, &uuid);
    let of = |topology, executor| Placing::ALL.map(|placing| natives(topology, executor, placing));
    assert_eq!(
        of(&discrete, cuda),
        [vec![0], vec![0], vec![1], vec![1], vec![3], vec![1]]
    );
    let vulkan = executor(&discrete, BackendName::Vulkan, &uuid);
    assert_eq!(
        of(&discrete, vulkan),
        [vec![0], vec![0], vec![2], vec![5], vec![imported], vec![5]]
    );
    assert!(
        discrete.first_store(cuda, Placing::Reserved(Shared))
            != discrete.first_store(cuda, Placing::Reserved(DeviceOnly))
    );
    assert!(!discrete.shared_ranges_lie_in_device_memory(cuda));

    let mut reports = gb10();
    let placed = Reservation {
        granule: 2 << 20,
        host_addressed: true,
        chunks: None,
    };
    for kind in &mut reports.vulkan[0].kinds {
        if kind.reservation != Ok(None) {
            kind.reservation = Ok(Some(placed));
        }
    }
    let unified = topology(&reports);
    let uuid = PhysicalIdentity::Uuid(crate::fixtures::GB10_UUID);
    let cuda = executor(&unified, BackendName::Cuda, &uuid);
    assert_eq!(
        of(&unified, cuda),
        [vec![0], vec![2], vec![1], vec![1], vec![2], vec![1]]
    );
    assert!(unified.shared_ranges_lie_in_device_memory(cuda));
    let vulkan = executor(&unified, BackendName::Vulkan, &uuid);
    assert_eq!(
        of(&unified, vulkan),
        [vec![0], vec![2], vec![2], vec![3], vec![3], vec![3]]
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
        [vec![0], vec![0], vec![0], vec![0], vec![0], vec![0]]
    );
    assert!(apple.shared_ranges_lie_in_device_memory(metal));
}

/// Without placed mapping, a unified device's only kind that reserves is
/// imported host pages (no chunked device memory is reported there).
#[test]
fn a_unified_device_without_placed_mapping_reserves_imported_pages() {
    let mut reports = gb10();
    let pages = Reservation {
        granule: 2 << 20,
        host_addressed: true,
        chunks: None,
    };
    let vulkan = &mut reports.vulkan[0];
    for kind in &mut vulkan.kinds {
        kind.reservation = Ok(None);
    }
    vulkan.kinds.push(crate::KindReport {
        id: crate::fixtures::VULKAN_IMPORTED_KIND,
        name: "imported host pages".into(),
        reach: crate::Reach::Mapped {
            reads: crate::HostReads::Cached,
        },
        obtained: crate::Obtained::ImportedHostPages,
        memory: crate::KindMemory::HostRam,
        reservation: Ok(Some(pages)),
    });
    let unified = topology(&reports);
    let uuid = PhysicalIdentity::Uuid(crate::fixtures::GB10_UUID);
    let vulkan = executor(&unified, BackendName::Vulkan, &uuid);
    let imported = vec![crate::fixtures::VULKAN_IMPORTED_KIND];
    assert_eq!(
        natives(&unified, vulkan, Placing::Reserved(ReservedContract::DeviceOnly)),
        imported
    );
    assert_eq!(
        natives(&unified, vulkan, Placing::Reserved(ReservedContract::Shared)),
        imported
    );
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
        natives(&topology, vulkan, Placing::Fixed(Contract::Upload)),
        [4]
    );
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
            reported(A10_8Q_CUDA_USABLE, ReportedBy::CudaUsableMemory)
        )]
    );
    assert_eq!(
        placement(&topology, vulkan, Contract::DeviceOnly),
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
    assert!(metal.paths(Placing::Fixed(Contract::DeviceOnly)).is_empty());
    let cpu = topology.resolve(ExecutorSelector::HostCpu).unwrap();
    assert_eq!(
        placement(&topology, cpu, Contract::DeviceOnly),
        [(StoreKind::HostRam, None)]
    );
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
    let placement = gpu.paths(Placing::Fixed(Contract::DeviceOnly));
    assert_eq!(placement.len(), 2);
    for id in gpu.placement(Placing::Fixed(Contract::DeviceOnly)) {
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
            .placement(Placing::Fixed(contract))
            .iter()
            .map(|id| discrete.kind(*id).unwrap().paths().to_vec())
            .collect::<Vec<_>>()
    };
    let [device] = vulkan.paths(Placing::Fixed(Contract::DeviceOnly)) else {
        panic!("one device storage path")
    };
    assert_eq!(paths(Contract::DeviceOnly), [vec![*device]]);
    assert_eq!(
        paths(Contract::Upload),
        [vec![*device]],
        "ReBAR: the device-local heap"
    );
    let [host] = vulkan.paths(Placing::Fixed(Contract::Shared)) else {
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
    let reserved = vulkan.placement(Placing::Reserved(ReservedContract::Shared));
    assert_eq!(reserved.len(), 1);
    let imported = discrete.kind(reserved[0]).unwrap();
    assert_eq!(imported.obtained(), crate::Obtained::ImportedHostPages);
    assert_eq!(imported.paths(), [*host]);
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
        crate::fixtures::unrecorded_reserving(),
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
            .paths(Placing::Fixed(Contract::DeviceOnly))
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
    assert!(vulkan.paths(Placing::Fixed(Contract::DeviceOnly)).is_empty());
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
        placement(&topology, vulkan, Contract::DeviceOnly),
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

#[test]
fn reservation_charges_preserve_table_placement_and_total() {
    use crate::{Reservation, ReservationChunks, TableCharge};
    let reservation = |table| Reservation {
        granule: 4096,
        host_addressed: true,
        chunks: Some(ReservationChunks { unit: 4096, table }),
    };
    let record = NonZeroU64::new(4096).unwrap();
    let upload = reservation(Contract::Upload);
    let initial = upload
        .charges(ReservedContract::Shared, ReservedRecords::of(record), 8, 8192)
        .unwrap();
    let smaller = upload
        .charges(ReservedContract::Shared, ReservedRecords::of(record), 8, 4096)
        .unwrap();
    assert_eq!(initial.backing, 8192);
    assert_eq!(
        initial.table,
        Some(TableCharge::Separate {
            placement: Contract::Upload,
            bytes: 64
        })
    );
    assert_eq!(
        initial.total(),
        upload.charge(ReservedRecords::of(record), 8, 8192)
    );
    assert_eq!(smaller.table, initial.table);
    assert_eq!(initial.total().unwrap() - smaller.total().unwrap(), 4096);
    assert!(upload
        .charges(
            ReservedContract::Shared,
            ReservedRecords::of(record),
            u64::MAX,
            4096
        )
        .is_none());
    // A table in the backing's own storage is charged with it.
    let device = reservation(Contract::DeviceOnly)
        .charges(ReservedContract::DeviceOnly, ReservedRecords::of(record), 8, 8192)
        .unwrap();
    assert_eq!(
        device.table,
        Some(TableCharge::WithBacking {
            placement: Contract::DeviceOnly,
            bytes: 64
        })
    );
}
