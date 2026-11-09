//! Recorded machines: the driver reports of the machine shapes the topology
//! handles, as a machine of ours or a public report recorded them, and the topologies discovery
//! builds from them ([`recorded`]). This crate's tests and other crates'
//! tests (through the `fixtures` feature) read the machines here.
//!
//! Every recorded value names its source: a driver report on our hardware,
//! a Vulkan Hardware Database report
//! (vulkan.gpuinfo.org/displayreport.php?id=N), or the issue that reported
//! the machine. A Vulkan device records its report's memory types, heap by
//! heap (a report gives no global order), from which the placement rule
//! decides.

use crate::kind::{HostReads, KindMemory, KindReport, Obtained, Reach, Reserving};
use crate::model::{
    Availability, DeviceTopology, Executor, ExecutorKind, PhysicalDeviceId, PhysicalIdentity,
    Reported, ReportedBy, Reservation, Revision, StoreId, StoreKind, VolumeId,
};
use crate::reports::{
    build, AdapterReport, CpuReport, CudaMemory, CudaReport, MetalReport, Reports, VolumeReport,
    VulkanHeap, VulkanMemory, VulkanReport,
};
use seismic_lang::registry::BackendName;
use std::fmt::Write;
use std::path::{Path, PathBuf};

/// The kind identifier a recorded Vulkan report gives imported host pages,
/// as the Vulkan backend reports them (its tests hold its kinds to these
/// recordings); every memory type's is its index.
pub const VULKAN_IMPORTED_KIND: u32 = u32::MAX;

pub const GB: u64 = 1_000_000_000;

/// The CPU backend's reservation in `granule`s, as it declares it
/// (`seismic_cpu::topology`'s `RESERVATIONS_HOST_ADDRESSED` and `RESERVATION_CHUNKS`):
/// host-addressed, in no chunks.
fn cpu_reservation(granule: u64) -> Reservation {
    Reservation {
        granule,
        host_addressed: true,
        chunks: None,
    }
}

/// The CPU backend's largest allocation on a 64-bit host
/// (`seismic_cpu::MAX_ALLOCATION_BYTES`): `isize::MAX` down to its 4 KiB
/// scratch alignment.
const CPU_MAX_ALLOCATION: u64 = 0x7fff_ffff_ffff_f000;

/// CUDA's reservation in `granule`s, as it declares it: in no chunks,
/// host-addressed unless it is a discrete device's own memory, which only
/// the device addresses (`seismic_cuda::topology`).
fn cuda_reservation(granule: u64, host_addressed: bool) -> Reservation {
    Reservation {
        granule,
        host_addressed,
        chunks: None,
    }
}

/// A Vulkan reservation as the backend reports it
/// (`seismic_vulkan::ReservationForm`): chunks of `unit`, addressed through
/// a record table in upload storage.
fn vulkan_reservation(granule: u64, unit: u64, host_addressed: bool) -> Reservation {
    Reservation {
        granule,
        host_addressed,
        chunks: Some(crate::ReservationChunks {
            unit,
            table: crate::Contract::Upload,
            whole: false,
        }),
    }
}

/// The host memory the CPU backend allocates, its one kind
/// (`seismic_cpu::kind`).
fn cpu_kind(granule: u64) -> KindReport {
    KindReport {
        id: 0,
        name: "host memory".into(),
        reach: Reach::Mapped {
            reads: HostReads::Cached,
        },
        obtained: Obtained::Allocated,
        memory: KindMemory::HostRam,
        reservation: Ok(Some(cpu_reservation(granule))),
    }
}

/// Metal's shared storage mode, its one kind (`seismic_metal::kind`),
/// reserved as `reservation`.
fn metal_kind(reservation: Reservation) -> KindReport {
    KindReport {
        id: 0,
        name: "shared storage".into(),
        reach: Reach::Mapped {
            reads: HostReads::Cached,
        },
        obtained: Obtained::Allocated,
        memory: KindMemory::HostRam,
        reservation: Ok(Some(reservation)),
    }
}

/// CUDA's kinds, as it declares them (`seismic_cuda::topology::kinds`) for
/// a device whose memory is the host's (`in_host`) or its own: device memory
/// and pinned host memory always; host-node memory on a device whose memory
/// is the host's, and registered host pages on one whose memory is its own,
/// where each is read to reserve. `created` is how memory CUDA creates
/// reserves, `registered` how registered host pages do.
fn cuda_kinds(in_host: bool, created: Reserving, registered: Reserving) -> Vec<KindReport> {
    let mapped = Reach::Mapped {
        reads: HostReads::Cached,
    };
    let kind = |id, name: &str, reach, obtained, memory, reservation| KindReport {
        id,
        name: name.into(),
        reach,
        obtained,
        memory,
        reservation,
    };
    let own = if in_host { Ok(None) } else { created.clone() };
    let mut kinds = vec![
        kind(
            0,
            "device memory",
            Reach::Device,
            Obtained::Allocated,
            KindMemory::DeviceMemory,
            own,
        ),
        kind(
            1,
            "pinned host memory",
            mapped,
            Obtained::Allocated,
            KindMemory::HostRam,
            Ok(None),
        ),
    ];
    if in_host && created != Ok(None) {
        kinds.push(kind(
            2,
            "host-node memory",
            mapped,
            Obtained::Allocated,
            KindMemory::HostRam,
            created,
        ));
    }
    if !in_host && registered != Ok(None) {
        kinds.push(kind(
            3,
            "registered host pages",
            mapped,
            Obtained::ImportedHostPages,
            KindMemory::HostRam,
            registered,
        ));
    }
    kinds
}

/// `report` with the kinds CUDA declares for it ([`cuda_kinds`]).
fn cuda(mut report: CudaReport, created: Reserving, registered: Reserving) -> CudaReport {
    report.kinds = cuda_kinds(
        matches!(report.memory, CudaMemory::Host),
        created,
        registered,
    );
    report
}

/// A recorded Vulkan memory layout: each heap's memory types' flags (`DL`,
/// `HV`, `HC`, `C`), heap by heap.
pub type Layout<'a> = &'a [&'a [&'a str]];

/// How a recorded Vulkan device reserves: the memory types that reserve
/// (a device whose memory is its own, its device-local types' chunks; one
/// whose memory is the host's, its mappable types' placed mappings), and
/// imported host pages, reported only where they reserve or why not.
pub struct VulkanReserving {
    pub types: Reserving,
    pub imported: Reserving,
}

/// The Vulkan kinds of `layout` on `heaps`, reserving as `reserving`
/// records, as the backend reports them (`seismic_vulkan`'s topology holds
/// itself to these): each memory type a kind on its heap, the host mapping
/// a mappable type, through its caches where it is host-cached; and
/// imported host pages, where they reserve or say why not.
fn recorded_vulkan_kinds(
    layout: Layout<'_>,
    heaps: &[(u64, bool)],
    memory_is_host: bool,
    reserving: VulkanReserving,
) -> Vec<KindReport> {
    let has = |flags: &str, flag: &str| flags.split('|').any(|each| each == flag);
    let mappable = |flags: &str| has(flags, "HV") && has(flags, "HC");
    let types = layout.iter().enumerate().flat_map(|(heap, types)| types.iter().map(move |flags| (heap, *flags)));
    let mut kinds = (0..)
        .zip(types)
        .map(|(id, (heap, flags))| {
            let (bytes, device_local) = heaps[heap];
            KindReport {
                id,
                name: format!("memory type {id}"),
                reach: match (mappable(flags), has(flags, "C")) {
                    (true, true) => Reach::Mapped { reads: HostReads::Cached },
                    (true, false) => Reach::Mapped { reads: HostReads::Uncached },
                    (false, _) => Reach::Device,
                },
                obtained: Obtained::Allocated,
                memory: KindMemory::Heap { index: heap, device_local, bytes },
                reservation: match if memory_is_host { mappable(flags) } else { has(flags, "DL") } {
                    true => reserving.types.clone(),
                    false => Ok(None),
                },
            }
        })
        .collect::<Vec<_>>();
    if reserving.imported != Ok(None) {
        kinds.push(KindReport {
            id: VULKAN_IMPORTED_KIND,
            name: "imported host pages".into(),
            reach: Reach::Mapped { reads: HostReads::Cached },
            obtained: Obtained::ImportedHostPages,
            memory: KindMemory::HostRam,
            reservation: reserving.imported,
        });
    }
    kinds
}

/// How a recorded Vulkan device whose report records no queue count or
/// host import reserves: by neither, which decide it.
pub fn unrecorded_reserving() -> VulkanReserving {
    let unrecorded = || Err(crate::Unreservable::Unrecorded);
    VulkanReserving {
        types: unrecorded(),
        imported: unrecorded(),
    }
}

/// The registration granule of an x86-64 Linux host's 4 KiB pages
/// (`seismic_reservation::registration_granule`): its 2 MiB large page, or with
/// none, the span of one last-level page table (4096 · 4096 / 8 bytes),
/// the same 2 MiB.
const X86_REGISTRATION_GRANULE: u64 = X86_PAGE * (X86_PAGE / 8);

/// The topology of each recorded machine, each built at a revision of its
/// own.
pub mod recorded {
    use crate::model::DeviceTopology;

    /// macOS on an M4 Pro, 48 GB: CPU and Metal on host RAM, Metal
    /// under its recommended working set.
    pub fn m4_pro() -> DeviceTopology {
        super::topology(&super::m4_pro())
    }

    /// The m4-pro report with `hasUnifiedMemory` false: the rule's input for
    /// a discrete Metal GPU (no host of ours has one), whose Metal executor
    /// is unavailable.
    pub fn m4_pro_without_unified_memory() -> DeviceTopology {
        super::topology(&super::m4_pro_without_unified_memory())
    }

    /// rtx3090 (Linux, RTX 3090): CPU, and CUDA and Vulkan on the 3090.
    pub fn rtx3090() -> DeviceTopology {
        super::topology(&super::rtx3090())
    }

    /// Linux on a GB10: CPU, and CUDA and Vulkan on host RAM.
    pub fn gb10() -> DeviceTopology {
        super::topology(&super::gb10())
    }

    /// Strix Halo on Linux (#199, RADV heaps of report 42266).
    pub fn strix_halo_linux() -> DeviceTopology {
        super::topology(&super::strix_halo_linux())
    }

    /// Strix Halo on Windows (#207, heaps of report 52227, 96 GB dedicated).
    pub fn strix_halo_windows() -> DeviceTopology {
        super::topology(&super::strix_halo_windows())
    }

    /// Lunar Lake (#206, ANV heap of report 52450).
    pub fn lunar_lake() -> DeviceTopology {
        super::topology(&super::lunar_lake())
    }

    /// A Ryzen iGPU (report 39430) and an RTX 4070 (report 51857).
    pub fn ryzen_igpu_and_rtx4070() -> DeviceTopology {
        super::topology(&super::ryzen_igpu_and_rtx4070())
    }

    /// The A10-8Q vGPU slice: CUDA and Vulkan on an 8 GiB partition.
    pub fn a10_8q() -> DeviceTopology {
        super::topology(&super::a10_8q())
    }
}

pub fn topology(reports: &Reports) -> DeviceTopology {
    build(Revision::next(), reports)
}

// Linux on an RTX 3090.
pub const RTX3090_HOST: u64 = 50_437_746_688;
pub const RTX3090_OS: &str = "Linux";
pub const RTX3090_CUDA_TOTAL: u64 = 25_298_141_184;
pub const RTX3090_CUDA_USABLE: u64 = 25_024_135_168;
/// The RTX 3090's L2, as CUDA reported it.
pub const RTX3090_CUDA_L2: u64 = 6_291_456;
pub const RTX3090_CUDA_DRIVER: &str = "cuda driver 12.8";
pub const RTX3090_VOLUME: (u64, Option<&str>, u64) = (82, None, 128_849_018_880);
// Report 52089 (NVIDIA 615.71.09, Linux).
pub const RTX3090_UUID: &str = "D37CB696763FCACE0D8DCBA6B3D100D0";
pub const RTX3090_VULKAN_DRIVER: &str = "vulkan driver 4 version 615.71.9";
pub const RTX3090_VULKAN_HEAP: u64 = 25_769_803_776;
/// NVIDIA's `min(maxMemoryAllocationSize, maxBufferSize)`: 2^64 − 1 and
/// 2^40 alike in reports 52089 (RTX 3090) and 51857 (RTX 4070), of driver
/// 615.71.09, and 52552 (GB10).
pub const NVIDIA_VULKAN_MAX_ALLOCATION: u64 = 1_099_511_627_776;

/// The CPU executor of a host, as discovery reads it there.
fn cpu() -> CpuReport {
    CpuReport {
        name: "Host CPU (x86_64)".into(),
        identity: "recorded x86_64 host".into(),
        cache: None,
        availability: Availability::Available,
        max_allocation: CPU_MAX_ALLOCATION,
        kind: cpu_kind(X86_PAGE),
    }
}

/// The host page of the recorded x86-64 hosts and of GB10's kernel (not a
/// 64K-page build), and of macOS on Apple silicon: what the CPU backend
/// backs a reservation in.
const X86_PAGE: u64 = 4096;
const GB10_PAGE: u64 = 4096;
const APPLE_SILICON_PAGE: u64 = 16_384;
/// The 2 MiB granule Metal (`seismic_metal::device::RESERVATION_GRANULE`)
/// and the recorded CUDA drivers (`cuMemGetAllocationGranularity`, the
/// reserved-tensor charge tests on a GB10) back a reservation in.
const DEVICE_GRANULE: u64 = 2 << 20;

/// The granule CUDA reserves host-NUMA memory in on GB10, where discovery
/// reserves device storage (its memory is the host's), as its driver
/// reports it (`cuMemGetAllocationGranularity`; held to the hardware by
/// `the_gb10_fixtures_host_granule_is_the_drivers`).
pub const GB10_HOST_GRANULE: u64 = 2 << 20;

fn reports(host: Reported<u64>, os_build: &str, cpu: CpuReport) -> Reports {
    Reports {
        host,
        os_build: os_build.into(),
        cpu,
        metal: Vec::new(),
        cuda: Vec::new(),
        vulkan: Vec::new(),
        adapters: Vec::new(),
        volumes: Vec::new(),
        diagnostics: Vec::new(),
    }
}

/// The RTX 3090 machine's host: its RAM, OS and CPU.
pub fn rtx3090_host() -> Reports {
    reports(
        Reported::new(RTX3090_HOST, ReportedBy::LinuxMemTotal),
        RTX3090_OS,
        cpu(),
    )
}

/// A Linux host whose RAM an issue reports, with rtx3090's OS and CPU
/// standing in for the parts the issue does not report.
pub fn linux_host(bytes: u64) -> Reports {
    let mut reports = rtx3090_host();
    reports.host = Reported::new(bytes, ReportedBy::LinuxMemTotal);
    reports
}

pub fn uuid(hex: &str) -> [u8; 16] {
    let mut uuid = [0u8; 16];
    for (index, byte) in uuid.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).unwrap();
    }
    uuid
}

/// A Vulkan device meeting the floor, as the backend reports it: its heaps
/// (each one's size and whether it is device-local), its memory `layout`,
/// and whose largest allocation is `max_allocation`: the report's
/// `min(maxMemoryAllocationSize, maxBufferSize)`, as the backend reads it.
pub fn vulkan(
    uuid: [u8; 16],
    name: &str,
    driver: &str,
    memory_is_host: bool,
    heaps: &[(u64, bool)],
    layout: Layout<'_>,
    max_allocation: u64,
    luid: Option<[u8; 8]>,
    reserving: VulkanReserving,
) -> VulkanReport {
    VulkanReport {
        uuid,
        kind: if memory_is_host {
            ExecutorKind::IntegratedGpu
        } else {
            ExecutorKind::DiscreteGpu
        },
        memory_is_host: Some(memory_is_host),
        name: name.into(),
        floor: Ok(()),
        driver: driver.into(),
        max_allocation,
        memory: VulkanMemory {
            luid,
            heaps: heaps
                .iter()
                .map(|&(bytes, device_local)| VulkanHeap {
                    bytes,
                    device_local,
                })
                .collect(),
        },
        kinds: recorded_vulkan_kinds(layout, heaps, memory_is_host, reserving),
    }
}

/// Report 52089's layout (RTX 3090).
pub const RTX3090_LAYOUT: Layout<'static> = &[&["DL", "DL", "DL|HV|HC"], &["", "HV|HC", "HV|HC|C"]];
/// Report 52552's layout (GB10): one heap holding every type.
pub const GB10_LAYOUT: Layout<'static> = &[&["DL", "DL", "DL|HV|HC", "DL|HV|HC|C"]];
/// RADV's layout, reports 42266 (Strix Halo) and 39430 (Ryzen iGPU): the
/// GTT heap, then the device-local heap.
pub const RADV_LAYOUT: Layout<'static> = &[
    &["HV|HC", "HV|HC|C", "HV|HC|C", "HV|HC", "HV|HC|C"],
    &["DL", "DL", "DL|HV|HC", "DL|HV|HC", "DL", "DL|HV|HC"],
];
/// Report 52227's layout (Strix Halo on Windows).
pub const STRIX_HALO_WINDOWS_LAYOUT: Layout<'static> = &[
    &[
        "HV|HC", "HV|HC|C", "HV|HC", "HV|HC|C", "HV|HC", "HV|HC|C", "HV|HC", "HV|HC|C",
    ],
    &[
        "DL", "DL|HV|HC", "DL", "DL|HV|HC", "DL", "DL|HV|HC", "DL", "DL|HV|HC",
    ],
];
/// Report 52450's layout (Lunar Lake): one heap.
pub const LUNAR_LAKE_LAYOUT: Layout<'static> =
    &[&["DL", "DL|HV|HC", "DL|HV|HC|C", "DL|HV|HC", "DL|HV|HC|C"]];
/// Report 51857's layout (RTX 4070).
pub const RTX4070_LAYOUT: Layout<'static> = &[&["DL", "DL|HV|HC"], &["", "HV|HC", "HV|HC|C"]];
/// The A10-8Q vGPU slice's one heap, as its record gives it.
pub const A10_8Q_LAYOUT: Layout<'static> = &[&["DL", "DL|HV|HC"]];

/// RTX 3090, report 52089 (NVIDIA 615.71.09, Linux).
pub fn rtx3090_vulkan(uuid: [u8; 16]) -> VulkanReport {
    rtx3090_vulkan_of(uuid, RTX3090_VULKAN_HEAP)
}

/// Report 52089 with a device-local heap of `heap` bytes, each memory type
/// on it reporting it as the report does.
fn rtx3090_vulkan_of(uuid: [u8; 16], heap: u64) -> VulkanReport {
    vulkan(
        uuid,
        "NVIDIA GeForce RTX 3090",
        RTX3090_VULKAN_DRIVER,
        false,
        &[(heap, true), (101_123_254_272, false)],
        RTX3090_LAYOUT,
        NVIDIA_VULKAN_MAX_ALLOCATION,
        None,
        // vulkaninfo on an RTX 3090, 2026-11-02: the compute-only family's
        // 8 queues (device storage in fill-unit chunks), host import at
        // 4 KiB within the host's 4 KiB page (host storage in host pages),
        // and maxMemoryAllocationCount 2^32 − 1, which leaves both at the
        // 2 MiB least chunk.
        VulkanReserving {
            types: Ok(Some(vulkan_reservation(DEVICE_GRANULE, 4, false))),
            imported: Ok(Some(vulkan_reservation(DEVICE_GRANULE, X86_PAGE, true))),
        },
    )
}

/// CUDA's view of the RTX 3090 machine's GPU.
pub fn rtx3090_cuda(uuid: [u8; 16], peers: Vec<[u8; 16]>) -> CudaReport {
    rtx3090_cuda_of(uuid, peers, RTX3090_CUDA_USABLE)
}

/// CUDA's view of the RTX 3090 letting this process use `usable` bytes.
fn rtx3090_cuda_of(uuid: [u8; 16], peers: Vec<[u8; 16]>, usable: u64) -> CudaReport {
    cuda(
        CudaReport {
            uuid,
            luid: None,
            name: "NVIDIA GeForce RTX 3090".into(),
            driver: RTX3090_CUDA_DRIVER.into(),
            admission: Ok(()),
            total: RTX3090_CUDA_TOTAL,
            max_allocation: RTX3090_CUDA_TOTAL,
            integrated: false,
            l2: Some(RTX3090_CUDA_L2),
            memory: CudaMemory::Own {
                usable: Ok(usable),
            },
            peers,
            kinds: Vec::new(),
        },
        Ok(Some(cuda_reservation(DEVICE_GRANULE, false))),
        // Registered host pages (CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM
        // on the RTX 3090), in the x86-64 host's registration granule.
        Ok(Some(cuda_reservation(X86_REGISTRATION_GRANULE, true))),
    )
}

/// The RTX 3090's UUID, as report 52089 gives it.
pub fn rtx3090_uuid() -> [u8; 16] {
    uuid(RTX3090_UUID)
}

/// The rtx3090 machine: its host, and the CUDA and Vulkan views of its RTX
/// 3090.
pub fn rtx3090() -> Reports {
    let gpu = rtx3090_uuid();
    let mut reports = rtx3090_host();
    reports.cuda.push(rtx3090_cuda(gpu, Vec::new()));
    reports.vulkan.push(rtx3090_vulkan(gpu));
    reports
}

/// The most the RTX 3090 grants `backend`'s device storage: the memory
/// the backend reports of the card, CUDA's total or Vulkan's device-local
/// heap.
pub fn rtx3090_most_granted(backend: BackendName) -> u64 {
    match backend {
        BackendName::Cuda => RTX3090_CUDA_TOTAL,
        BackendName::Vulkan => RTX3090_VULKAN_HEAP,
        other => panic!("the RTX 3090 grants device storage on CUDA and Vulkan, not {other:?}"),
    }
}

/// The rtx3090 machine with its RTX 3090 granting `grant` bytes to
/// `backend`'s device storage, the grant stable fit plans against: CUDA's
/// usable memory, or Vulkan's device-local heap, as the heap and each
/// memory type on it report it. The other backend reports as recorded. A
/// grant is of the memory the backend reports of the card
/// ([`rtx3090_most_granted`]).
pub fn rtx3090_granting(backend: BackendName, grant: u64) -> Reports {
    let most = rtx3090_most_granted(backend);
    assert!(grant <= most, "the RTX 3090 grants {backend:?} no more than its {most} bytes");
    let gpu = rtx3090_uuid();
    let mut reports = rtx3090_host();
    let (usable, heap) = match backend {
        BackendName::Vulkan => (RTX3090_CUDA_USABLE, grant),
        _ => (grant, RTX3090_VULKAN_HEAP),
    };
    reports.cuda.push(rtx3090_cuda_of(gpu, Vec::new(), usable));
    reports.vulkan.push(rtx3090_vulkan_of(gpu, heap));
    reports
}

// macOS on an M4 Pro, 48 GB.
pub const M4_PRO_HOST: u64 = 51_539_607_552;
pub const M4_PRO_OS: &str = "Version 15.7.9 (Build 24G830)";
pub const M4_PRO_REGISTRY: u64 = 4_294_968_578;
pub const M4_PRO_WORKING_SET: u64 = 38_654_705_664;
pub const M4_PRO_MAX_BUFFER: u64 = 28_991_029_248;
/// Two 16 MiB performance-cluster L2s and one 4 MiB efficiency-cluster L2,
/// as macOS reported them.
pub const M4_PRO_CPU_CACHE: u64 = 37_748_736;

pub fn m4_pro() -> Reports {
    let mut reports = reports(
        Reported::new(M4_PRO_HOST, ReportedBy::MacOsMemsize),
        M4_PRO_OS,
        CpuReport {
            name: "Host CPU (aarch64)".into(),
            identity: "Apple M4 Pro".into(),
            cache: Some(Reported::new(
                M4_PRO_CPU_CACHE,
                ReportedBy::MacOsSysctlCache,
            )),
            availability: Availability::Available,
            max_allocation: CPU_MAX_ALLOCATION,
            kind: cpu_kind(APPLE_SILICON_PAGE),
        },
    );
    reports.metal.push(MetalReport {
        registry_id: M4_PRO_REGISTRY,
        name: "Apple M4 Pro".into(),
        unified: true,
        admission: Ok(()),
        // The M4 Pro's GPU is of Apple GPU family 9, below the
        // tensor-operation floor (family 10).
        tensor_operations: false,
        working_set: M4_PRO_WORKING_SET,
        max_allocation: M4_PRO_MAX_BUFFER,
        // Recorded: `seismic_metal::device::RESERVATIONS_HOST_ADDRESSED` and
        // `RESERVATION_CHUNKS`, which exist only where Metal does.
        kind: metal_kind(Reservation {
            granule: DEVICE_GRANULE,
            host_addressed: true,
            chunks: Some(crate::ReservationChunks {
                unit: APPLE_SILICON_PAGE,
                table: crate::Contract::DeviceOnly,
                whole: true,
            }),
        }),
    });
    reports
}

/// The m4-pro report with `hasUnifiedMemory` false, the rule's input (no
/// host of ours has a discrete Metal GPU).
pub fn m4_pro_without_unified_memory() -> Reports {
    let mut reports = m4_pro();
    reports.metal[0].unified = false;
    reports
}

// Linux on a GB10; the Vulkan layout's types are report 52552's (one heap holding every type).
pub const GB10_HOST: u64 = 130_663_231_488;
pub const GB10_UUID: [u8; 16] = [
    126, 167, 120, 13, 84, 128, 7, 101, 28, 72, 157, 41, 75, 23, 164, 144,
];
pub const GB10_CUDA_TOTAL: u64 = 130_663_231_488;
pub const GB10_CUDA_DRIVER: &str = "cuda driver 13.0";
pub const GB10_VULKAN_HEAP: u64 = 97_997_423_616;
pub const GB10_VULKAN_DRIVER: &str = "vulkan driver 4 version 2435301568";
pub const GB10_OS: &str =
    "Linux 6.17.0-1026-nvidia #26-Ubuntu SMP PREEMPT_DYNAMIC Thu Jun 25 00:57:17 UTC 2026";
/// The CPU's last-level caches together, and the GPU's L2, as Linux and
/// CUDA reported them.
pub const GB10_CPU_CACHE: u64 = 25_165_824;
pub const GB10_CUDA_L2: u64 = 25_165_824;

pub fn gb10() -> Reports {
    let mut reports = reports(
        Reported::new(GB10_HOST, ReportedBy::LinuxMemTotal),
        GB10_OS,
        CpuReport {
            name: "Host CPU (aarch64)".into(),
            identity: "NVIDIA GB10".into(),
            cache: Some(Reported::new(GB10_CPU_CACHE, ReportedBy::LinuxSysfsCache)),
            availability: Availability::Available,
            max_allocation: CPU_MAX_ALLOCATION,
            kind: cpu_kind(GB10_PAGE),
        },
    );
    reports.cuda.push(cuda(
        CudaReport {
            uuid: GB10_UUID,
            luid: None,
            name: "NVIDIA GB10".into(),
            driver: GB10_CUDA_DRIVER.into(),
            admission: Ok(()),
            total: GB10_CUDA_TOTAL,
            max_allocation: GB10_CUDA_TOTAL,
            integrated: true,
            l2: Some(GB10_CUDA_L2),
            memory: CudaMemory::Host,
            peers: Vec::new(),
            kinds: Vec::new(),
        },
        // Reserved in host-node memory, which the host addresses.
        Ok(Some(cuda_reservation(GB10_HOST_GRANULE, true))),
        Ok(None),
    ));
    reports.vulkan.push(vulkan(
        GB10_UUID,
        "NVIDIA GB10",
        GB10_VULKAN_DRIVER,
        true,
        &[(GB10_VULKAN_HEAP, true)],
        GB10_LAYOUT,
        NVIDIA_VULKAN_MAX_ALLOCATION,
        None,
        unrecorded_reserving(),
    ));
    reports
}

// Strix Halo on Linux, RADV: report 42266 (Mesa 25.0.7), whose heaps are
// RADV's split of GTT plus carve-out. The RAM is #199's ("58GB of
// RAM available"); the report's machine is a larger one, whose RAM the
// database does not record.
pub const STRIX_HALO_LINUX_HOST: u64 = 58 * GB;
pub const STRIX_HALO_LINUX_UUID: &str = "00000000C60000000000000000000000";
pub const STRIX_HALO_LINUX_HEAPS: [u64; 2] = [39_904_485_376, 79_808_970_752];
/// RADV's `maxMemoryAllocationSize` and `maxBufferSize`, both 2^32 − 4 in
/// reports 42266 (Strix Halo) and 39430 (Ryzen iGPU).
pub const RADV_MAX_ALLOCATION: u64 = 4_294_967_292;

pub fn strix_halo_linux_vulkan() -> VulkanReport {
    let [gtt, device_local] = STRIX_HALO_LINUX_HEAPS;
    vulkan(
        uuid(STRIX_HALO_LINUX_UUID),
        "AMD Radeon Graphics (RADV GFX1151)",
        "vulkan driver 3 version 25.0.7",
        true,
        &[(gtt, false), (device_local, true)],
        RADV_LAYOUT,
        RADV_MAX_ALLOCATION,
        None,
        unrecorded_reserving(),
    )
}

pub fn strix_halo_linux() -> Reports {
    let mut reports = linux_host(STRIX_HALO_LINUX_HOST);
    reports.vulkan.push(strix_halo_linux_vulkan());
    reports
}

// Strix Halo on Windows: report 52227 (AMD 26.9.2, Radeon 8060S), its LUID
// 313A010000000000. #207's machine (128 GB, "96 assigned to gpu", Windows
// reporting "32Gb unified memory") supplies the DXGI dedicated memory and
// host RAM, in GB as the issue states them.
pub const STRIX_HALO_WINDOWS_HOST: u64 = 32 * GB;
pub const STRIX_HALO_WINDOWS_DEDICATED: u64 = 96 * GB;
pub const STRIX_HALO_WINDOWS_UUID: &str = "00000000BC0000000000000000000000";
pub const STRIX_HALO_WINDOWS_LUID: [u8; 8] = [0x31, 0x3A, 0x01, 0, 0, 0, 0, 0];
pub const STRIX_HALO_WINDOWS_HEAPS: [u64; 2] = [36_692_557_824, 73_385_115_648];
/// The OS as report 52227 states it (Windows 11, x86_64). The report gives
/// no build number, which discovery's `Windows <major>.<minor>.<build>`
/// carries.
pub const STRIX_HALO_WINDOWS_OS: &str = "Windows 11";
/// AMD's Windows driver's `maxMemoryAllocationSize` and `maxBufferSize`,
/// both 2^31 in report 52227.
pub const STRIX_HALO_WINDOWS_MAX_ALLOCATION: u64 = 2_147_483_648;

/// A Windows host whose RAM and OS a record reports, with an x86_64 CPU.
pub fn windows_host(bytes: u64, os_build: &str) -> Reports {
    reports(
        Reported::new(bytes, ReportedBy::WindowsTotalPhys),
        os_build,
        cpu(),
    )
}

pub fn strix_halo_windows() -> Reports {
    let [shared, device_local] = STRIX_HALO_WINDOWS_HEAPS;
    let mut reports = windows_host(STRIX_HALO_WINDOWS_HOST, STRIX_HALO_WINDOWS_OS);
    reports.adapters.push(AdapterReport {
        luid: STRIX_HALO_WINDOWS_LUID,
        dedicated: STRIX_HALO_WINDOWS_DEDICATED,
    });
    reports.vulkan.push(vulkan(
        uuid(STRIX_HALO_WINDOWS_UUID),
        "AMD Radeon(TM) 8060S Graphics",
        "vulkan driver 1 version 2.0.406",
        true,
        &[(shared, false), (device_local, true)],
        STRIX_HALO_WINDOWS_LAYOUT,
        STRIX_HALO_WINDOWS_MAX_ALLOCATION,
        Some(STRIX_HALO_WINDOWS_LUID),
        unrecorded_reserving(),
    ));
    reports
}

// Lunar Lake: report 52450 (Mesa 26.2.3, ANV), one heap. The RAM is #206's
// ("14.8 GB unified memory"); the report's machine is another, whose RAM
// the database does not record.
pub const LUNAR_LAKE_HOST: u64 = 14_800_000_000;
pub const LUNAR_LAKE_UUID: &str = "8680A064040000000002000000000000";
pub const LUNAR_LAKE_HEAP: u64 = 24_869_332_992;
/// ANV's `maxBufferSize`, 2^32, below its `maxMemoryAllocationSize` (the
/// heap's size) in report 52450.
pub const LUNAR_LAKE_MAX_ALLOCATION: u64 = 4_294_967_296;

pub fn lunar_lake() -> Reports {
    let mut reports = linux_host(LUNAR_LAKE_HOST);
    reports.vulkan.push(vulkan(
        uuid(LUNAR_LAKE_UUID),
        "Intel(R) Graphics (LNL)",
        "vulkan driver 6 version 26.2.3",
        true,
        &[(LUNAR_LAKE_HEAP, true)],
        LUNAR_LAKE_LAYOUT,
        LUNAR_LAKE_MAX_ALLOCATION,
        None,
        unrecorded_reserving(),
    ));
    reports
}

// A Ryzen iGPU, report 39430 (RADV GFX1103_R1, Mesa 24.2.8), and an RTX
// 4070, report 51857 (NVIDIA 615.71.09, Linux), on the RTX 3090 machine's RAM:
// no report pairs them with RAM or with the 4070's CUDA view, so the 4070
// is its Vulkan view alone.
pub const RYZEN_IGPU_UUID: &str = "00000000C40000000000000000000000";
pub const RYZEN_IGPU_HEAPS: [u64; 2] = [11_902_001_152, 23_804_002_304];
pub const RTX4070_UUID: &str = "BC93928CFFA26FA92292A87860C54187";
pub const RTX4070_HEAP: u64 = 12_878_610_432;

pub fn ryzen_igpu_and_rtx4070() -> Reports {
    let [gtt, device_local] = RYZEN_IGPU_HEAPS;
    let mut reports = rtx3090_host();
    reports.vulkan.push(vulkan(
        uuid(RYZEN_IGPU_UUID),
        "AMD Radeon Graphics (RADV GFX1103_R1)",
        "vulkan driver 3 version 24.2.8",
        true,
        &[(gtt, false), (device_local, true)],
        // RADV's layout, as Strix Halo's.
        RADV_LAYOUT,
        RADV_MAX_ALLOCATION,
        None,
        unrecorded_reserving(),
    ));
    reports.vulkan.push(vulkan(
        uuid(RTX4070_UUID),
        "NVIDIA GeForce RTX 4070",
        "vulkan driver 4 version 615.71.9",
        false,
        &[(RTX4070_HEAP, true), (25_065_572_352, false)],
        RTX4070_LAYOUT,
        NVIDIA_VULKAN_MAX_ALLOCATION,
        None,
        unrecorded_reserving(),
    ));
    reports
}

// The A10-8Q vGPU slice (CUDA and Vulkan under WDDM): CUDA's usable memory
// and total, and Vulkan's heap, as commit 157ab9413 recorded them in the
// runtime catalog's test "The A10-8Q vGPU slice"
// (`seismic/runtime/src/devices/catalog.rs` at that commit). The record
// gives no host, UUID, drivers, L2 or Vulkan limits, and the Vulkan
// Hardware Database holds no A10-8Q report: the RTX 3090 machine's host,
// the RTX 3090's UUID and drivers, and NVIDIA's Vulkan allocation limit stand in,
// and the slice has no L2, which no record gives.
pub const A10_8Q_CUDA_TOTAL: u64 = 8_589_475_840;
pub const A10_8Q_CUDA_USABLE: u64 = 6_603_640_832;
pub const A10_8Q_VULKAN_HEAP: u64 = 7_643_791_360;

pub fn a10_8q() -> Reports {
    let slice = rtx3090_uuid();
    let mut reports = rtx3090_host();
    reports.cuda.push(cuda(
        CudaReport {
            uuid: slice,
            luid: None,
            name: "NVIDIA A10-8Q".into(),
            driver: RTX3090_CUDA_DRIVER.into(),
            admission: Ok(()),
            total: A10_8Q_CUDA_TOTAL,
            max_allocation: A10_8Q_CUDA_TOTAL,
            integrated: false,
            l2: None,
            memory: CudaMemory::Own {
                usable: Ok(A10_8Q_CUDA_USABLE),
            },
            peers: Vec::new(),
            kinds: Vec::new(),
        },
        Ok(Some(cuda_reservation(DEVICE_GRANULE, false))),
        // Registered pages' granule is not recorded.
        Ok(None),
    ));
    reports.vulkan.push(vulkan(
        slice,
        "NVIDIA A10-8Q",
        RTX3090_VULKAN_DRIVER,
        false,
        &[(A10_8Q_VULKAN_HEAP, true)],
        A10_8Q_LAYOUT,
        NVIDIA_VULKAN_MAX_ALLOCATION,
        None,
        unrecorded_reserving(),
    ));
    reports
}

// Reading topologies in tests: this crate's, and the inventory's in
// `seismic-discovery`.

/// The executor of `backend` on the physical device of `identity`.
pub fn executor<'a>(
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
pub fn placement(
    topology: &DeviceTopology,
    executor: &Executor,
    contract: crate::Contract,
) -> Vec<(StoreKind, Option<Reported<u64>>)> {
    executor
        .paths(crate::Placing::Fixed(contract))
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

/// The volume store the inventory serves `path` on.
pub fn volume(topology: &DeviceTopology, path: &Path) -> Option<StoreId> {
    topology
        .stores()
        .iter()
        .map(|store| store.id())
        .find(|&store| topology.volume_paths(store).any(|served| served == path))
}

/// The topology as text: what each row's test produced.
pub fn describe(topology: &DeviceTopology) -> String {
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
        for contract in crate::Contract::ALL {
            let places = executor
                .paths(crate::Placing::Fixed(contract))
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
            write!(text, "; {contract:?} -> [{}]", places.join(", ")).unwrap();
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

pub fn show(row: &str, topology: &DeviceTopology) {
    eprintln!("{row}\n{}", describe(topology));
}

/// Every path names nodes of its revision, and every placement names the
/// executor's own access paths.
pub fn assert_consistent(topology: &DeviceTopology) {
    for path in topology.access_paths() {
        assert!(topology.executor(path.executor()).is_some());
        assert!(topology.store(path.store()).is_some());
    }
    for path in topology.transfer_paths() {
        assert!(topology.store(path.from()).is_some() && topology.store(path.to()).is_some());
        assert!(topology.executor(path.by()).is_some());
    }
    for executor in topology.executors() {
        for placing in crate::Placing::ALL {
            for &id in executor.paths(placing) {
                assert_eq!(topology.access_path(id).unwrap().executor(), executor.id());
            }
        }
    }
}

/// The RTX 3090 host's evaluation volume, served at two paths.
pub fn rtx3090_volume(path: &str) -> VolumeReport {
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
