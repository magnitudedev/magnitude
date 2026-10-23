//! Recorded machines: the driver reports of the spec's §3 situations that a
//! host of ours or a public report recorded, and the topologies discovery
//! builds from them ([`recorded`]). This crate's tests and other crates'
//! tests (through the `fixtures` feature) read the machines here.
//!
//! Every recorded value names its source: a run on our hosts
//! (an evaluation run id), a Vulkan Hardware Database report
//! (vulkan.gpuinfo.org/displayreport.php?id=N), or the issue that reported
//! the machine. The database's reports list memory types under their heap
//! and do not give the types' global order; each layout here lists them heap
//! by heap, which the allocation rule reads only within a heap.

use crate::discovery::{
    build, AdapterReport, CpuReport, CudaMemory, CudaReport, MetalReport, Reports, VulkanHeap,
    VulkanMemory, VulkanReport,
};
use crate::model::{Availability, DeviceTopology, ExecutorKind, Reported, ReportedBy, Revision};

pub(crate) const GB: u64 = 1_000_000_000;

/// The topology of each recorded machine, each built at a revision of its
/// own.
pub mod recorded {
    use crate::model::DeviceTopology;

    /// m4-pro-04 (macOS, M4 Pro, 48 GB): CPU and Metal on host RAM, Metal
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

    /// sparky (Linux, GB10): CPU, and CUDA and Vulkan on host RAM.
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

pub(crate) fn topology(reports: &Reports) -> DeviceTopology {
    build(Revision::next(), reports)
}

// rtx3090 (Linux, RTX 3090), run 20261018T194135Z-cargo.test-0f6481 and
// the live run recorded below.
pub(crate) const RTX3090_HOST: u64 = 50_437_746_688;
pub(crate) const RTX3090_OS: &str = "Linux";
pub(crate) const RTX3090_CUDA_TOTAL: u64 = 25_298_141_184;
pub(crate) const RTX3090_CUDA_USABLE: u64 = 25_024_135_168;
/// The RTX 3090's L2, as rtx3090 reported it (run
/// 20261019T040157Z-fix.seismic-088b6b).
pub(crate) const RTX3090_CUDA_L2: u64 = 6_291_456;
pub(crate) const RTX3090_CUDA_DRIVER: &str = "cuda driver 12.8";
#[cfg(test)]
pub(crate) const RTX3090_VOLUME: (u64, Option<&str>, u64) = (82, None, 128_849_018_880);
// Report 52089 (NVIDIA 615.71.09, Linux).
pub(crate) const RTX3090_UUID: &str = "D37CB696763FCACE0D8DCBA6B3D100D0";
pub(crate) const RTX3090_VULKAN_DRIVER: &str = "vulkan driver 4 version 615.71.9";
pub(crate) const RTX3090_VULKAN_HEAP: u64 = 25_769_803_776;
/// NVIDIA's `min(maxMemoryAllocationSize, maxBufferSize)`: 2^64 − 1 and
/// 2^40 alike in reports 52089 (RTX 3090) and 51857 (RTX 4070), of driver
/// 615.71.09, and 52552 (GB10).
pub(crate) const NVIDIA_VULKAN_MAX_ALLOCATION: u64 = 1_099_511_627_776;

/// The CPU executor of a host, as discovery reads it there.
fn cpu() -> CpuReport {
    CpuReport {
        name: "Host CPU (x86_64)".into(),
        identity: "recorded x86_64 host".into(),
        cache: None,
        availability: Availability::Available,
        max_allocation: seismic_cpu::MAX_ALLOCATION_BYTES,
    }
}

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

/// The rtx3090 host: its RAM, OS and CPU.
pub(crate) fn rtx3090_host() -> Reports {
    reports(
        Reported::new(RTX3090_HOST, ReportedBy::LinuxMemTotal),
        RTX3090_OS,
        cpu(),
    )
}

/// A Linux host whose RAM an issue reports, with rtx3090's OS and CPU
/// standing in for the parts the issue does not report.
pub(crate) fn linux_host(bytes: u64) -> Reports {
    let mut reports = rtx3090_host();
    reports.host = Reported::new(bytes, ReportedBy::LinuxMemTotal);
    reports
}

pub(crate) fn uuid(hex: &str) -> [u8; 16] {
    let mut uuid = [0u8; 16];
    for (index, byte) in uuid.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).unwrap();
    }
    uuid
}

/// A memory layout from a report: each heap's size, whether it is
/// device-local, and its memory types' flags (`DL`, `HV`, `HC`, `C`).
pub(crate) fn layout(heaps: &[(u64, bool, &[&str])]) -> seismic_vulkan::MemoryLayout {
    let mut layout = seismic_vulkan::MemoryLayout {
        heaps: Vec::new(),
        types: Vec::new(),
    };
    for (heap, &(bytes, device_local, types)) in heaps.iter().enumerate() {
        layout.heaps.push(seismic_vulkan::MemoryHeap {
            bytes,
            device_local,
        });
        for flags in types {
            let has = |flag: &str| flags.split('|').any(|each| each == flag);
            layout.types.push(seismic_vulkan::MemoryType {
                heap: heap as u32,
                device_local: has("DL"),
                host_visible: has("HV"),
                host_coherent: has("HC"),
                host_cached: has("C"),
            });
        }
    }
    layout
}

/// A Vulkan device meeting the floor, allocating as the backend does
/// (`seismic_vulkan::Allocation::of`), whose largest allocation is
/// `max_allocation`: the report's `min(maxMemoryAllocationSize,
/// maxBufferSize)`, as the backend reads it.
pub(crate) fn vulkan(
    uuid: [u8; 16],
    name: &str,
    driver: &str,
    memory_is_host: bool,
    layout: seismic_vulkan::MemoryLayout,
    max_allocation: u64,
    luid: Option<[u8; 8]>,
) -> VulkanReport {
    let allocation = seismic_vulkan::Allocation::of(&layout, memory_is_host);
    let heaps = |types: &[u32]| {
        seismic_vulkan::Allocation::heaps(&layout, types)
            .into_iter()
            .map(|heap| heap as usize)
            .collect()
    };
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
            heaps: layout
                .heaps
                .iter()
                .map(|heap| VulkanHeap {
                    bytes: heap.bytes,
                    device_local: heap.device_local,
                })
                .collect(),
            allocates: [heaps(&allocation.device), heaps(&allocation.upload)],
        },
    }
}

/// RTX 3090, report 52089 (NVIDIA 615.71.09, Linux).
pub(crate) fn rtx3090_vulkan(uuid: [u8; 16]) -> VulkanReport {
    vulkan(
        uuid,
        "NVIDIA GeForce RTX 3090",
        RTX3090_VULKAN_DRIVER,
        false,
        layout(&[
            (RTX3090_VULKAN_HEAP, true, &["DL", "DL", "DL|HV|HC"]),
            (101_123_254_272, false, &["", "HV|HC", "HV|HC|C"]),
        ]),
        NVIDIA_VULKAN_MAX_ALLOCATION,
        None,
    )
}

/// The rtx3090 host's CUDA view of its RTX 3090.
pub(crate) fn rtx3090_cuda(uuid: [u8; 16], peers: Vec<[u8; 16]>) -> CudaReport {
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
            usable: Ok(RTX3090_CUDA_USABLE),
        },
        peers,
    }
}

/// The RTX 3090's UUID, as report 52089 gives it.
pub(crate) fn rtx3090_uuid() -> [u8; 16] {
    uuid(RTX3090_UUID)
}

/// The rtx3090 machine: its host, and the CUDA and Vulkan views of its RTX
/// 3090.
pub(crate) fn rtx3090() -> Reports {
    let gpu = rtx3090_uuid();
    let mut reports = rtx3090_host();
    reports.cuda.push(rtx3090_cuda(gpu, Vec::new()));
    reports.vulkan.push(rtx3090_vulkan(gpu));
    reports
}

// m4-pro-04 (macOS, M4 Pro, 48 GB), run 20261018T200619Z-cargo.test-f7c5b0.
pub(crate) const M4_PRO_HOST: u64 = 51_539_607_552;
pub(crate) const M4_PRO_OS: &str = "Version 15.7.9 (Build 24G830)";
pub(crate) const M4_PRO_REGISTRY: u64 = 4_294_968_578;
pub(crate) const M4_PRO_WORKING_SET: u64 = 38_654_705_664;
pub(crate) const M4_PRO_MAX_BUFFER: u64 = 28_991_029_248;
/// Two 16 MiB performance-cluster L2s and one 4 MiB efficiency-cluster L2,
/// as m4-pro-01 reported them (run 20261019T040157Z-fix.seismic-24bcae).
pub(crate) const M4_PRO_CPU_CACHE: u64 = 37_748_736;

pub(crate) fn m4_pro() -> Reports {
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
            max_allocation: seismic_cpu::MAX_ALLOCATION_BYTES,
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
    });
    reports
}

/// The m4-pro report with `hasUnifiedMemory` false, the rule's input (no
/// host of ours has a discrete Metal GPU).
pub(crate) fn m4_pro_without_unified_memory() -> Reports {
    let mut reports = m4_pro();
    reports.metal[0].unified = false;
    reports
}

// sparky (Linux, GB10), run 20261018T200619Z-cargo.test-1cc1e4; the
// Vulkan layout's types are report 52552's (one heap holding every type).
pub(crate) const GB10_HOST: u64 = 130_663_231_488;
pub(crate) const GB10_UUID: [u8; 16] = [
    126, 167, 120, 13, 84, 128, 7, 101, 28, 72, 157, 41, 75, 23, 164, 144,
];
pub(crate) const GB10_CUDA_TOTAL: u64 = 130_663_231_488;
pub(crate) const GB10_CUDA_DRIVER: &str = "cuda driver 13.0";
pub(crate) const GB10_VULKAN_HEAP: u64 = 97_997_423_616;
pub(crate) const GB10_VULKAN_DRIVER: &str = "vulkan driver 4 version 2435301568";
pub(crate) const GB10_OS: &str =
    "Linux 6.17.0-1026-nvidia #26-Ubuntu SMP PREEMPT_DYNAMIC Thu Jun 25 00:57:17 UTC 2026";
/// The CPU's last-level caches together, and the GPU's L2, as sparky
/// reported them (run 20261019T040157Z-fix.seismic-24d66f).
pub(crate) const GB10_CPU_CACHE: u64 = 25_165_824;
pub(crate) const GB10_CUDA_L2: u64 = 25_165_824;

pub(crate) fn gb10() -> Reports {
    let mut reports = reports(
        Reported::new(GB10_HOST, ReportedBy::LinuxMemTotal),
        GB10_OS,
        CpuReport {
            name: "Host CPU (aarch64)".into(),
            identity: "NVIDIA GB10".into(),
            cache: Some(Reported::new(GB10_CPU_CACHE, ReportedBy::LinuxSysfsCache)),
            availability: Availability::Available,
            max_allocation: seismic_cpu::MAX_ALLOCATION_BYTES,
        },
    );
    reports.cuda.push(CudaReport {
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
    });
    reports.vulkan.push(vulkan(
        GB10_UUID,
        "NVIDIA GB10",
        GB10_VULKAN_DRIVER,
        true,
        layout(&[(
            GB10_VULKAN_HEAP,
            true,
            &["DL", "DL", "DL|HV|HC", "DL|HV|HC|C"],
        )]),
        NVIDIA_VULKAN_MAX_ALLOCATION,
        None,
    ));
    reports
}

// Strix Halo on Linux, RADV: report 42266 (Mesa 25.0.7), whose heaps are
// RADV's split of GTT plus carve-out (§4.1). The RAM is #199's ("58GB of
// RAM available"); the report's machine is a larger one, whose RAM the
// database does not record.
pub(crate) const STRIX_HALO_LINUX_HOST: u64 = 58 * GB;
pub(crate) const STRIX_HALO_LINUX_UUID: &str = "00000000C60000000000000000000000";
pub(crate) const STRIX_HALO_LINUX_HEAPS: [u64; 2] = [39_904_485_376, 79_808_970_752];
/// RADV's `maxMemoryAllocationSize` and `maxBufferSize`, both 2^32 − 4 in
/// reports 42266 (Strix Halo) and 39430 (Ryzen iGPU).
pub(crate) const RADV_MAX_ALLOCATION: u64 = 4_294_967_292;

pub(crate) fn strix_halo_linux_vulkan() -> VulkanReport {
    let [gtt, device_local] = STRIX_HALO_LINUX_HEAPS;
    vulkan(
        uuid(STRIX_HALO_LINUX_UUID),
        "AMD Radeon Graphics (RADV GFX1151)",
        "vulkan driver 3 version 25.0.7",
        true,
        layout(&[
            (
                gtt,
                false,
                &["HV|HC", "HV|HC|C", "HV|HC|C", "HV|HC", "HV|HC|C"],
            ),
            (
                device_local,
                true,
                &["DL", "DL", "DL|HV|HC", "DL|HV|HC", "DL", "DL|HV|HC"],
            ),
        ]),
        RADV_MAX_ALLOCATION,
        None,
    )
}

pub(crate) fn strix_halo_linux() -> Reports {
    let mut reports = linux_host(STRIX_HALO_LINUX_HOST);
    reports.vulkan.push(strix_halo_linux_vulkan());
    reports
}

// Strix Halo on Windows: report 52227 (AMD 26.9.2, Radeon 8060S), its LUID
// 313A010000000000. #207's machine (128 GB, "96 assigned to gpu", Windows
// reporting "32Gb unified memory") supplies the DXGI dedicated memory and
// host RAM, in GB as the issue states them.
pub(crate) const STRIX_HALO_WINDOWS_HOST: u64 = 32 * GB;
pub(crate) const STRIX_HALO_WINDOWS_DEDICATED: u64 = 96 * GB;
pub(crate) const STRIX_HALO_WINDOWS_UUID: &str = "00000000BC0000000000000000000000";
pub(crate) const STRIX_HALO_WINDOWS_LUID: [u8; 8] = [0x31, 0x3A, 0x01, 0, 0, 0, 0, 0];
pub(crate) const STRIX_HALO_WINDOWS_HEAPS: [u64; 2] = [36_692_557_824, 73_385_115_648];
/// The OS as report 52227 states it (Windows 11, x86_64). The report gives
/// no build number, which discovery's `Windows <major>.<minor>.<build>`
/// carries.
pub(crate) const STRIX_HALO_WINDOWS_OS: &str = "Windows 11";
/// AMD's Windows driver's `maxMemoryAllocationSize` and `maxBufferSize`,
/// both 2^31 in report 52227.
pub(crate) const STRIX_HALO_WINDOWS_MAX_ALLOCATION: u64 = 2_147_483_648;

/// A Windows host whose RAM and OS a record reports, with an x86_64 CPU.
pub(crate) fn windows_host(bytes: u64, os_build: &str) -> Reports {
    reports(
        Reported::new(bytes, ReportedBy::WindowsTotalPhys),
        os_build,
        cpu(),
    )
}

pub(crate) fn strix_halo_windows() -> Reports {
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
        layout(&[
            (
                shared,
                false,
                &[
                    "HV|HC", "HV|HC|C", "HV|HC", "HV|HC|C", "HV|HC", "HV|HC|C", "HV|HC", "HV|HC|C",
                ],
            ),
            (
                device_local,
                true,
                &[
                    "DL", "DL|HV|HC", "DL", "DL|HV|HC", "DL", "DL|HV|HC", "DL", "DL|HV|HC",
                ],
            ),
        ]),
        STRIX_HALO_WINDOWS_MAX_ALLOCATION,
        Some(STRIX_HALO_WINDOWS_LUID),
    ));
    reports
}

// Lunar Lake: report 52450 (Mesa 26.2.3, ANV), one heap. The RAM is #206's
// ("14.8 GB unified memory"); the report's machine is another, whose RAM
// the database does not record.
pub(crate) const LUNAR_LAKE_HOST: u64 = 14_800_000_000;
pub(crate) const LUNAR_LAKE_UUID: &str = "8680A064040000000002000000000000";
pub(crate) const LUNAR_LAKE_HEAP: u64 = 24_869_332_992;
/// ANV's `maxBufferSize`, 2^32, below its `maxMemoryAllocationSize` (the
/// heap's size) in report 52450.
pub(crate) const LUNAR_LAKE_MAX_ALLOCATION: u64 = 4_294_967_296;

pub(crate) fn lunar_lake() -> Reports {
    let mut reports = linux_host(LUNAR_LAKE_HOST);
    reports.vulkan.push(vulkan(
        uuid(LUNAR_LAKE_UUID),
        "Intel(R) Graphics (LNL)",
        "vulkan driver 6 version 26.2.3",
        true,
        layout(&[(
            LUNAR_LAKE_HEAP,
            true,
            &["DL", "DL|HV|HC", "DL|HV|HC|C", "DL|HV|HC", "DL|HV|HC|C"],
        )]),
        LUNAR_LAKE_MAX_ALLOCATION,
        None,
    ));
    reports
}

// A Ryzen iGPU, report 39430 (RADV GFX1103_R1, Mesa 24.2.8), and an RTX
// 4070, report 51857 (NVIDIA 615.71.09, Linux), on the rtx3090 host's RAM:
// no report pairs them with RAM or with the 4070's CUDA view, so the 4070
// is its Vulkan view alone.
pub(crate) const RYZEN_IGPU_UUID: &str = "00000000C40000000000000000000000";
pub(crate) const RYZEN_IGPU_HEAPS: [u64; 2] = [11_902_001_152, 23_804_002_304];
pub(crate) const RTX4070_UUID: &str = "BC93928CFFA26FA92292A87860C54187";
pub(crate) const RTX4070_HEAP: u64 = 12_878_610_432;

pub(crate) fn ryzen_igpu_and_rtx4070() -> Reports {
    let [gtt, device_local] = RYZEN_IGPU_HEAPS;
    let mut reports = rtx3090_host();
    reports.vulkan.push(vulkan(
        uuid(RYZEN_IGPU_UUID),
        "AMD Radeon Graphics (RADV GFX1103_R1)",
        "vulkan driver 3 version 24.2.8",
        true,
        layout(&[
            (
                gtt,
                false,
                &["HV|HC", "HV|HC|C", "HV|HC|C", "HV|HC", "HV|HC|C"],
            ),
            (
                device_local,
                true,
                &["DL", "DL", "DL|HV|HC", "DL|HV|HC", "DL", "DL|HV|HC"],
            ),
        ]),
        RADV_MAX_ALLOCATION,
        None,
    ));
    reports.vulkan.push(vulkan(
        uuid(RTX4070_UUID),
        "NVIDIA GeForce RTX 4070",
        "vulkan driver 4 version 615.71.9",
        false,
        layout(&[
            (RTX4070_HEAP, true, &["DL", "DL|HV|HC"]),
            (25_065_572_352, false, &["", "HV|HC", "HV|HC|C"]),
        ]),
        NVIDIA_VULKAN_MAX_ALLOCATION,
        None,
    ));
    reports
}

// The A10-8Q vGPU slice (CUDA and Vulkan under WDDM): CUDA's usable memory
// and total, and Vulkan's heap, as commit 157ab9413 recorded them in the
// runtime catalog's test "The A10-8Q vGPU slice"
// (`seismic/runtime/src/devices/catalog.rs` at that commit). The record
// gives no host, UUID, drivers, L2 or Vulkan limits, and the Vulkan
// Hardware Database holds no A10-8Q report: the rtx3090 host, the RTX
// 3090's UUID and drivers, and NVIDIA's Vulkan allocation limit stand in,
// and the slice has no L2, which no record gives.
pub(crate) const A10_8Q_CUDA_TOTAL: u64 = 8_589_475_840;
pub(crate) const A10_8Q_CUDA_USABLE: u64 = 6_603_640_832;
pub(crate) const A10_8Q_VULKAN_HEAP: u64 = 7_643_791_360;

pub(crate) fn a10_8q() -> Reports {
    let slice = rtx3090_uuid();
    let mut reports = rtx3090_host();
    reports.cuda.push(CudaReport {
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
    });
    reports.vulkan.push(vulkan(
        slice,
        "NVIDIA A10-8Q",
        RTX3090_VULKAN_DRIVER,
        false,
        layout(&[(A10_8Q_VULKAN_HEAP, true, &["DL", "DL|HV|HC"])]),
        NVIDIA_VULKAN_MAX_ALLOCATION,
        None,
    ));
    reports
}
