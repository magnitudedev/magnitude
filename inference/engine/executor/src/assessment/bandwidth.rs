//! A device's memory bandwidth, from what discovery already knows: the
//! driver's report, a published specification matched by device name, or
//! the device's class. Nothing runs on the device.

use seismic::{DeviceInfo, DeviceKind, DeviceMemory, DeviceTopology};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceBandwidth {
    pub bytes_per_second: u64,
    pub source: BandwidthSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BandwidthSource {
    /// The driver reported the memory clock and bus width.
    Reported,
    /// A published specification matched the device's name.
    Published,
    /// The low end of the device's class: its name is not in the table.
    Assumed(DeviceClass),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceClass {
    /// A GPU with its own memory.
    DedicatedGpu,
    /// A GPU allocating host memory.
    IntegratedGpu,
    Cpu,
}

impl DeviceClass {
    /// A GPU whose memory backing is not established takes the lower GPU
    /// class.
    fn of(device: &DeviceInfo) -> Self {
        match (device.kind, &device.memory) {
            (DeviceKind::Cpu, _) => Self::Cpu,
            (DeviceKind::Gpu, DeviceMemory::Established(memory))
                if !memory.allocates_host_memory() =>
            {
                Self::DedicatedGpu
            }
            (DeviceKind::Gpu, _) => Self::IntegratedGpu,
        }
    }

    /// The low end of the class, so an unrecognized device's speed errs
    /// slow.
    fn assumed_bytes_per_second(self) -> u64 {
        match self {
            Self::DedicatedGpu => 200 * GB,
            Self::IntegratedGpu => 70 * GB,
            Self::Cpu => 40 * GB,
        }
    }
}

const GB: u64 = 1_000_000_000;
const MB: u64 = 1_000_000;

/// The bandwidth of `device`. Always resolves: the driver's report, else the
/// published specification of its name, else its class. `cpu_cores`
/// selects between an Apple chip's bins, which share one name.
pub fn resolve_bandwidth(
    topology: &DeviceTopology,
    device: &DeviceInfo,
    cpu_cores: u32,
) -> DeviceBandwidth {
    if let Some(bytes_per_second) = device.memory_bandwidth {
        return DeviceBandwidth {
            bytes_per_second,
            source: BandwidthSource::Reported,
        };
    }
    // The memory the device allocates from: its own for a discrete GPU, host RAM for a unified
    // one.
    let memory_bytes = match &device.memory {
        DeviceMemory::Established(memory) => topology
            .pool(memory.allocation_pool)
            .map(|pool| pool.capacity_bytes),
        DeviceMemory::Unsupported { .. } => None,
    };
    if let Some(bytes_per_second) = published(&device.name, cpu_cores, memory_bytes) {
        return DeviceBandwidth {
            bytes_per_second,
            source: BandwidthSource::Published,
        };
    }
    let class = DeviceClass::of(device);
    DeviceBandwidth {
        bytes_per_second: class.assumed_bytes_per_second(),
        source: BandwidthSource::Assumed(class),
    }
}

/// How a published figure depends on the device's configuration. Where one
/// name covers configurations the device's facts cannot tell apart, the
/// table lists the lowest, so the estimate errs slow.
enum Bandwidth {
    Fixed(u64),
    /// Chip bins sharing one name, by CPU core count. A count the table
    /// does not list takes the lowest bin.
    ByCpuCores(&'static [(u32, u64)]),
    /// Configurations sharing one name, by the memory the device allocates
    /// from, in GiB. The nearest listed size applies: a driver reports
    /// slightly less than the board's memory.
    ByMemoryGib(&'static [(u64, u64)]),
}

struct PublishedBandwidth {
    /// Normalized device names (`normalize_name`).
    names: &'static [&'static str],
    /// Megabytes per second.
    bandwidth: Bandwidth,
}

const fn fixed(names: &'static [&'static str], megabytes: u64) -> PublishedBandwidth {
    PublishedBandwidth {
        names,
        bandwidth: Bandwidth::Fixed(megabytes),
    }
}

fn published(name: &str, cpu_cores: u32, memory_bytes: Option<u64>) -> Option<u64> {
    let name = normalize_name(name);
    let entry = PUBLISHED
        .iter()
        .find(|entry| entry.names.iter().any(|candidate| *candidate == name))?;
    let megabytes = match entry.bandwidth {
        Bandwidth::Fixed(megabytes) => megabytes,
        Bandwidth::ByCpuCores(bins) => bins
            .iter()
            .find(|(cores, _)| *cores == cpu_cores)
            .or_else(|| bins.iter().min_by_key(|(_, megabytes)| *megabytes))
            .map(|(_, megabytes)| *megabytes)?,
        Bandwidth::ByMemoryGib(configurations) => {
            let bytes = memory_bytes?;
            configurations
                .iter()
                .min_by_key(|(gib, _)| (gib << 30).abs_diff(bytes))
                .map(|(_, megabytes)| *megabytes)?
        }
    };
    Some(megabytes * MB)
}

/// Lowercase, collapse whitespace, drop trademark marks, every trailing
/// parenthetical (a RADV or ANV driver suffix, a Windows shared-memory size),
/// the vendor and brand prefix and a trailing "graphics". Model suffixes
/// (Ti, SUPER, XT, XTX, Laptop GPU, Max, Pro, Ultra) are identity and stay.
pub fn normalize_name(name: &str) -> String {
    let mut name = name
        .to_lowercase()
        .replace("(r)", "")
        .replace("(tm)", "")
        .replace(['®', '™'], "");
    while let Some(open) = name.trim_end().strip_suffix(')').and_then(|body| body.rfind('(')) {
        name.truncate(open);
    }
    let mut words = name.split_whitespace().collect::<Vec<_>>();
    for prefixes in [["nvidia", "amd", "intel"].as_slice(), &["geforce", "radeon"]] {
        if words.len() > 1 && prefixes.contains(&words[0]) {
            words.remove(0);
        }
    }
    if words.len() > 1 && words.last() == Some(&"graphics") {
        words.pop();
    }
    words.join(" ")
}

/// Published peak memory bandwidth in megabytes per second, keyed by the
/// name each driver reports. Figures are the vendor's where the vendor
/// publishes one; otherwise TechPowerUp's (TPU), Notebookcheck's (NBC), or
/// derived as memory transfer rate × bus width (derived).
const PUBLISHED: &[PublishedBandwidth] = &[
    // Apple silicon, by MTLDevice.name. https://support.apple.com/en-us/111900 (M1 Max, Ultra),
    // /111902 (M1 Pro), /111340 (M2 Pro, Max), /111835 (M2 Ultra), /117735 (M3), /117736 (M3
    // Pro, Max), /122211 (M3 Ultra), /121555 (M4), /121553 (M4 Pro, Max), /125405 (M5),
    // /126318 (M5 Pro, Max), /128107 (M5 Ultra), /128108 (M6).
    // M1: LPDDR4X-4266 on 128 bits (derived; Apple publishes none).
    fixed(&["apple m1"], 68_250),
    fixed(&["apple m1 pro"], 200_000),
    fixed(&["apple m1 max"], 400_000),
    fixed(&["apple m1 ultra"], 800_000),
    fixed(&["apple m2"], 100_000),
    fixed(&["apple m2 pro"], 200_000),
    fixed(&["apple m2 max"], 400_000),
    fixed(&["apple m2 ultra"], 800_000),
    fixed(&["apple m3"], 100_000),
    fixed(&["apple m3 pro"], 150_000),
    PublishedBandwidth {
        names: &["apple m3 max"],
        bandwidth: Bandwidth::ByCpuCores(&[(14, 300_000), (16, 400_000)]),
    },
    fixed(&["apple m3 ultra"], 819_000),
    fixed(&["apple m4"], 120_000),
    fixed(&["apple m4 pro"], 273_000),
    PublishedBandwidth {
        names: &["apple m4 max"],
        bandwidth: Bandwidth::ByCpuCores(&[(14, 410_000), (16, 546_000)]),
    },
    fixed(&["apple m5"], 153_000),
    fixed(&["apple m5 pro"], 307_000),
    // Both bins have 18 CPU cores; the 40-core GPU bin reaches 614 GB/s.
    fixed(&["apple m5 max"], 460_000),
    fixed(&["apple m5 ultra"], 1_200_000),
    PublishedBandwidth {
        names: &["apple m6"],
        bandwidth: Bandwidth::ByMemoryGib(&[(16, 153_000), (24, 170_000), (32, 170_000)]),
    },
    // AMD Radeon desktop, raw DRAM bandwidth.
    // https://www.amd.com/en/products/specifications/graphics.html
    fixed(&["rx 6400"], 128_000),
    fixed(&["rx 6500 xt"], 144_000),
    fixed(&["rx 6600"], 224_000),
    fixed(&["rx 6600 xt"], 256_000),
    fixed(&["rx 6650 xt"], 280_000),
    fixed(&["rx 6700", "rx 6750 gre 10gb"], 320_000),
    fixed(&["rx 6700 xt", "rx 6750 gre 12gb"], 384_000),
    fixed(&["rx 6750 xt"], 432_000),
    fixed(&["rx 6800", "rx 6800 xt", "rx 6900 xt"], 512_000),
    fixed(&["rx 6950 xt"], 576_000),
    fixed(&["rx 7400"], 173_000),
    fixed(&["rx 7600", "rx 7600 xt", "rx 7650 gre"], 288_000),
    fixed(&["rx 7700", "rx 7800 xt"], 624_000),
    fixed(&["rx 7700 xt"], 432_000),
    fixed(&["rx 7900 gre"], 576_000),
    fixed(&["rx 7900 xt"], 800_000),
    fixed(&["rx 7900 xtx"], 960_000),
    PublishedBandwidth {
        names: &["rx 9050"],
        bandwidth: Bandwidth::ByMemoryGib(&[(4, 144_000), (8, 288_000)]),
    },
    fixed(&["rx 9060"], 288_000),
    fixed(&["rx 9060 xt"], 320_000),
    fixed(&["rx 9070 gre"], 432_000),
    fixed(&["rx 9070", "rx 9070 xt"], 640_000),
    // AMD Radeon mobile.
    fixed(&["rx 6300m"], 64_000),
    fixed(&["rx 6450m", "rx 6500m", "rx 6550s"], 128_000),
    fixed(&["rx 6550m"], 144_000),
    fixed(&["rx 6600m", "rx 6600s", "rx 6700s"], 224_000),
    fixed(&["rx 6650m", "rx 6650m xt", "rx 6800s", "rx 7600m", "rx 7600s"], 256_000),
    fixed(&["rx 6700m"], 320_000),
    fixed(&["rx 6800m"], 384_000),
    fixed(&["rx 6850m xt", "rx 7800m"], 432_000),
    fixed(&["rx 7600m xt", "rx 7700s"], 288_000),
    fixed(&["rx 7900m"], 576_000),
    // AMD Radeon workstation.
    // https://www.amd.com/en/products/specifications/professional-graphics.html
    // W7500: AMD's datasheet (172.8) over its product page (256).
    fixed(&["pro w7400", "pro w7500"], 172_800),
    fixed(&["pro w7600"], 288_000),
    fixed(&["pro w7700"], 576_000),
    PublishedBandwidth {
        names: &["pro w7800"],
        bandwidth: Bandwidth::ByMemoryGib(&[(32, 576_000), (48, 864_000)]),
    },
    fixed(&["pro w7900"], 864_000),
    fixed(&["ai pro r9700", "ai pro r9700s", "ai pro r9600d"], 640_000),
    // AMD integrated, sharing system memory: the lowest memory each ships
    // with. Ryzen AI Max: LPDDR5X-8000 on 256 bits (8040S on 128).
    fixed(&["8060s", "8050s"], 256_000),
    fixed(&["8040s"], 128_000),
    // Ryzen AI 300 / 400 with DDR5-5600 on 128 bits (up to 136.5 with
    // LPDDR5X-8533).
    fixed(&["890m", "880m", "860m", "840m", "820m"], 89_600),
    // Ryzen 7040 / 8040 laptops and the 8000G desktops (DDR5-5200).
    fixed(&["780m", "760m", "740m"], 83_200),
    // Intel Arc discrete. https://www.intel.com/content/www/us/en/products/details/discrete-gpus/arc.html
    fixed(&["arc a310"], 124_000),
    fixed(&["arc a380"], 186_000),
    fixed(&["arc a580", "arc a750", "arc a770m"], 512_000),
    PublishedBandwidth {
        names: &["arc a770"],
        bandwidth: Bandwidth::ByMemoryGib(&[(8, 512_000), (16, 560_000)]),
    },
    fixed(&["arc a350m", "arc a370m"], 112_000),
    fixed(&["arc a530m", "arc a550m", "arc pro b50"], 224_000),
    fixed(&["arc a570m"], 256_000),
    fixed(&["arc a730m"], 336_000),
    fixed(&["arc pro a40/a50"], 192_000),
    fixed(&["arc pro a60"], 384_000),
    fixed(&["arc b570"], 380_000),
    fixed(&["arc b580", "arc pro b60"], 456_000),
    fixed(&["arc pro b65", "arc pro b70"], 608_000),
    // Intel integrated (derived on 128 bits): Lunar Lake's on-package
    // LPDDR5X-8533; Panther Lake B390's LPDDR5X-9600; otherwise DDR5 (B370
    // at 7200, 140T at 6400, Meteor Lake Arc at 5600).
    fixed(&["arc 140v gpu", "arc 130v gpu"], 136_500),
    fixed(&["arc b390 gpu"], 153_600),
    fixed(&["arc b370 gpu"], 115_200),
    fixed(&["arc 140t gpu"], 102_400),
    fixed(&["arc"], 89_600),
    // NVIDIA through Vulkan (CUDA reports its own bandwidth), and Turing
    // cards below the CUDA backend's floor. RTX 50: NVIDIA
    // (https://www.nvidia.com/en-us/geforce/graphics-cards/compare/); RTX
    // 20–40: TPU; laptops: the part's maximum (NBC, TPU).
    fixed(&["rtx 2060"], 336_000),
    fixed(&["rtx 2060 super", "rtx 2070", "rtx 2070 super", "rtx 2080"], 448_000),
    // The desktop "SUPER" (495.9) and laptop "Super" (448) share the name.
    fixed(&["rtx 2080 super"], 448_000),
    fixed(&["rtx 2080 ti"], 616_000),
    PublishedBandwidth {
        names: &["rtx 3050"],
        bandwidth: Bandwidth::ByMemoryGib(&[(6, 168_000), (8, 224_000)]),
    },
    PublishedBandwidth {
        names: &["rtx 3060"],
        bandwidth: Bandwidth::ByMemoryGib(&[(8, 240_000), (12, 360_000)]),
    },
    // GDDR6 (448) and GDDR6X (608.3) boards share the name and size.
    fixed(&["rtx 3060 ti", "rtx 3070"], 448_000),
    fixed(&["rtx 3070 ti"], 608_300),
    PublishedBandwidth {
        names: &["rtx 3080"],
        bandwidth: Bandwidth::ByMemoryGib(&[(10, 760_300), (12, 912_400)]),
    },
    fixed(&["rtx 3080 ti"], 912_400),
    fixed(&["rtx 3090"], 936_200),
    fixed(&["rtx 3090 ti"], 1_008_300),
    fixed(&["rtx 4060"], 272_000),
    fixed(&["rtx 4060 ti"], 288_000),
    // GDDR6 (480) and GDDR6X (504.2) boards share the name and size.
    fixed(&["rtx 4070"], 480_000),
    fixed(&["rtx 4070 super", "rtx 4070 ti"], 504_200),
    fixed(&["rtx 4070 ti super"], 672_300),
    fixed(&["rtx 4080"], 716_800),
    fixed(&["rtx 4080 super"], 736_300),
    fixed(&["rtx 4090", "rtx 4090 d"], 1_008_000),
    fixed(&["rtx 5050"], 320_000),
    fixed(&["rtx 5060", "rtx 5060 ti"], 448_000),
    fixed(&["rtx 5070"], 672_000),
    fixed(&["rtx 5070 ti"], 896_000),
    fixed(&["rtx 5080"], 960_000),
    fixed(&["rtx 5090", "rtx 5090 d"], 1_792_000),
    fixed(&["rtx 5090 d v2"], 1_344_000),
    fixed(&["rtx 2050"], 112_000),
    fixed(&["rtx 2060 with max-q design"], 264_000),
    fixed(
        &["rtx 2070 with max-q design", "rtx 2080 with max-q design"],
        384_000,
    ),
    fixed(
        &["rtx 2070 super with max-q design", "rtx 2080 super with max-q design"],
        352_000,
    ),
    fixed(&["rtx 3050 6gb laptop gpu"], 168_000),
    fixed(
        &["rtx 3050 laptop gpu", "rtx 3050 ti laptop gpu", "rtx 3050ti laptop gpu", "rtx 4050 laptop gpu"],
        192_000,
    ),
    fixed(&["rtx 3060 laptop gpu"], 288_000),
    // Max-Q parts share the name (3070: 384–448, 3080: 384–448).
    fixed(&["rtx 3070 laptop gpu", "rtx 3080 laptop gpu"], 384_000),
    fixed(&["rtx 3070 ti laptop gpu"], 448_000),
    fixed(&["rtx 3080 ti laptop gpu"], 512_000),
    fixed(&["rtx 4060 laptop gpu", "rtx 4070 laptop gpu"], 256_000),
    fixed(&["rtx 4080 laptop gpu"], 432_000),
    fixed(&["rtx 4090 laptop gpu"], 576_000),
    fixed(
        &["rtx 5050 laptop gpu", "rtx 5060 laptop gpu", "rtx 5070 laptop gpu"],
        384_000,
    ),
    fixed(&["rtx 5070 ti laptop gpu"], 672_000),
    fixed(&["rtx 5080 laptop gpu", "rtx 5090 laptop gpu"], 896_000),
    // NVIDIA data center, workstation and embedded. https://www.nvidia.com/en-us/data-center/
    fixed(&["gb10"], 273_000),
    fixed(&["a100-pcie-40gb", "a100-sxm4-40gb"], 1_555_000),
    fixed(&["a100 80gb pcie"], 1_935_000),
    fixed(&["a100-sxm4-80gb"], 2_039_000),
    fixed(&["h100 pcie"], 2_000_000),
    fixed(&["h100 80gb hbm3"], 3_350_000),
    fixed(&["h100 nvl"], 3_900_000),
    fixed(&["h200", "h200 nvl"], 4_800_000),
    fixed(&["l40s", "l40"], 864_000),
    fixed(&["l4"], 300_000),
    fixed(&["a10"], 600_000),
    fixed(&["a40"], 696_000),
    fixed(&["rtx a6000"], 768_000),
    fixed(&["rtx 6000 ada generation"], 960_000),
    fixed(&["rtx 5000 ada generation"], 576_000),
    fixed(&["rtx 4500 ada generation"], 432_000),
    fixed(&["rtx 4000 ada generation"], 360_000),
    fixed(
        &[
            "rtx pro 6000 blackwell workstation edition",
            "rtx pro 6000 blackwell max-q workstation edition",
        ],
        1_792_000,
    ),
    fixed(&["rtx pro 6000 blackwell server edition"], 1_597_000),
    fixed(&["rtx pro 5000 blackwell"], 1_344_000),
    fixed(&["rtx pro 4500 blackwell"], 896_000),
    fixed(&["rtx pro 4000 blackwell"], 672_000),
    fixed(&["b200", "b300 sxm6 ac"], 8_000_000),
    fixed(&["gb300"], 7_100_000),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn driver_spellings_normalize_to_one_name() {
        for spelling in [
            "NVIDIA GeForce RTX 4090",
            "GeForce RTX 4090",
            "RTX 4090",
            "NVIDIA  GeForce  RTX 4090",
        ] {
            assert_eq!(normalize_name(spelling), "rtx 4090");
        }
        assert_eq!(normalize_name("AMD Radeon RX 7900 XTX (RADV NAVI31)"), "rx 7900 xtx");
        assert_eq!(normalize_name("AMD Radeon RX 9070 XT"), "rx 9070 xt");
        assert_eq!(normalize_name("AMD Radeon(TM) 8060S Graphics"), "8060s");
        assert_eq!(normalize_name("AMD Radeon 780M Graphics (RADV PHOENIX)"), "780m");
        assert_eq!(normalize_name("Intel(R) Arc(TM) B580 Graphics"), "arc b580");
        assert_eq!(normalize_name("Intel(R) Arc(tm) A770 Graphics (DG2)"), "arc a770");
        assert_eq!(normalize_name("Intel(R) Arc(TM) 140V GPU (16GB)"), "arc 140v gpu");
        assert_eq!(normalize_name("Apple M4 Max"), "apple m4 max");
        assert_eq!(
            normalize_name("NVIDIA GeForce RTX 4070 Laptop GPU"),
            "rtx 4070 laptop gpu"
        );
    }

    #[test]
    fn every_table_name_is_normalized_and_unique() {
        let mut seen = std::collections::BTreeSet::new();
        for entry in PUBLISHED {
            for name in entry.names {
                assert_eq!(normalize_name(name), *name);
                assert!(seen.insert(*name), "{name} is listed twice");
            }
        }
    }

    #[test]
    fn the_reported_machines_resolve() {
        // magnitudedev/magnitude#142: M1 Max, M2 Max and an RX 9070 XT.
        assert_eq!(published("Apple M1 Max", 10, None), Some(400 * GB));
        assert_eq!(published("Apple M2 Max", 12, None), Some(400 * GB));
        assert_eq!(published("AMD Radeon RX 9070 XT", 16, Some(16 << 30)), Some(640 * GB));
    }

    #[test]
    fn apple_bins_follow_cpu_cores_and_default_to_the_lowest() {
        assert_eq!(published("Apple M4 Max", 16, None), Some(546 * GB));
        assert_eq!(published("Apple M4 Max", 14, None), Some(410 * GB));
        assert_eq!(published("Apple M4 Max", 12, None), Some(410 * GB));
        assert_eq!(published("Apple M6", 12, Some(24 << 30)), Some(170 * GB));
    }

    #[test]
    fn configurations_follow_the_nearest_memory_size() {
        assert_eq!(
            published("NVIDIA GeForce RTX 3060", 8, Some((12 << 30) - 300 * MB)),
            Some(360 * GB)
        );
        assert_eq!(published("NVIDIA GeForce RTX 3060", 8, Some(8 << 30)), Some(240 * GB));
        assert_eq!(published("NVIDIA GeForce RTX 3060", 8, None), None);
    }

    #[test]
    fn an_unknown_name_is_not_published() {
        assert_eq!(published("Mystery GPU 9000", 8, None), None);
        assert_eq!(published("Host CPU (aarch64)", 8, None), None);
    }
}
