use super::super::{Counted, WiredLimit};
use super::memory::{
    compressed_saving, compressed_stored, file_floor, paired, pressure_level, swap_free,
    wired_limit, Served, Snapshot, STATISTICS_WINDOW,
};
use super::*;
use seismic_topology::{flat_nodes, CacheSize, HostNode, PciLocation};
use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

const GB: u64 = 1 << 30;

/// A Mac as a test states it: the sysctls it reports, every other name
/// unreported. Its Mach counts are zero: a sample reads them of the
/// running OS alone.
struct Mac {
    sysctls: BTreeMap<String, Vec<u8>>,
    parallelism: NonZeroUsize,
    profiler_json: Option<Vec<u8>>,
    vm_wire_count: u32,
    iogpu_in_use: Option<u64>,
}

impl Mac {
    /// A Mac of 64 GB and a quota of one CPU, reporting nothing else.
    fn new() -> Self {
        Self {
            sysctls: BTreeMap::new(),
            parallelism: NonZeroUsize::MIN,
            profiler_json: None,
            vm_wire_count: 0,
            iogpu_in_use: None,
        }
        .count("hw.memsize", 64 * GB)
    }

    /// The Mac with `name` reporting `value`, a 64-bit count.
    fn count(mut self, name: &str, value: u64) -> Self {
        self.sysctls.insert(name.into(), value.to_ne_bytes().to_vec());
        self
    }

    /// The Mac with `counts` reported, each `(name, value)`.
    fn counts(self, counts: &[(&str, u64)]) -> Self {
        counts
            .iter()
            .fold(self, |mac, (name, value)| mac.count(name, *value))
    }

    /// The Mac with `name` reporting the NUL-terminated string `value`.
    fn text(mut self, name: &str, value: &str) -> Self {
        let mut bytes = value.as_bytes().to_vec();
        bytes.push(0);
        self.sysctls.insert(name.into(), bytes);
        self
    }
}

impl Sys for Mac {
    fn sysctl(&self, name: &str) -> Option<Vec<u8>> {
        self.sysctls.get(name).cloned()
    }
    fn vm_statistics(&self) -> libc::vm_statistics64 {
        libc::vm_statistics64 {
            wire_count: self.vm_wire_count,
            ..unsafe { std::mem::zeroed() }
        }
    }
    fn task_vm(&self) -> TaskVm {
        TaskVm::default()
    }
    fn page(&self) -> u64 {
        16 << 10
    }
    fn parallelism(&self) -> NonZeroUsize {
        self.parallelism
    }
    fn hardware_profiler_json(&self) -> Option<Vec<u8>> {
        self.profiler_json.clone()
    }
    fn iogpu_in_use_memory(&self) -> Option<u64> {
        self.iogpu_in_use
    }
}

/// A compressor storing 1 GB in 4 GB saves three quarters of what it
/// compresses; one storing nothing is not counted to save anything.
#[test]
fn the_compressor_saves_what_it_does_not_store() {
    assert_eq!(compressed_saving(8 << 30, 1 << 30, 4 << 30), 6 << 30);
    assert_eq!(compressed_saving(8 << 30, 0, 0), 0);
    assert_eq!(compressed_stored(8 << 30, 1 << 30, 4 << 30), 2 << 30);
    assert_eq!(compressed_stored(8 << 30, 0, 0), 0);
}

/// A snapshot served again keeps the instant and the paging it was
/// first read with; a new one counts from when the sample before began,
/// never more than the window before it, with the paging read beside it.
#[test]
fn a_snapshot_pairs_with_the_paging_read_when_it_was_first_served() {
    let statistics = |decompressions| libc::vm_statistics64 {
        decompressions,
        ..unsafe { std::mem::zeroed() }
    };
    let start = Instant::now();
    let first = paired(None, statistics(10), start, 3, 0);
    assert_eq!(
        (first.counted, first.own),
        (
            Counted {
                at: start,
                lag: STATISTICS_WINDOW
            },
            3
        )
    );
    let again = paired(
        Some(Served {
            snapshot: first,
            began: start,
        }),
        statistics(10),
        start + Duration::from_millis(40),
        5,
        0,
    );
    assert_eq!(
        (again.counted, again.own),
        (
            Counted {
                at: start,
                lag: STATISTICS_WINDOW
            },
            3
        )
    );
    let began = start + Duration::from_millis(40);
    let fresh = paired(
        Some(Served {
            snapshot: again,
            began,
        }),
        statistics(12),
        start + Duration::from_millis(70),
        6,
        0,
    );
    assert_eq!(
        (fresh.counted, fresh.own),
        (
            Counted {
                at: start + Duration::from_millis(70),
                lag: Duration::from_millis(30)
            },
            6
        )
    );
    let late = paired(
        Some(Served {
            snapshot: fresh,
            began,
        }),
        statistics(13),
        start + Duration::from_secs(5),
        7,
        0,
    );
    assert_eq!(
        late.counted,
        Counted {
            at: start + Duration::from_secs(5),
            lag: STATISTICS_WINDOW
        }
    );
}

/// The file floor is the file pages standing where the OS compressed
/// more than was free above its target (excluding speculative pages as
/// `vm.page_free_count` does), held while compressions or tight free persist,
/// decaying and releasing once compressions cease and free recovers
/// comfortably above target, and clamped to current external pages.
#[test]
fn the_file_floor_is_what_the_os_keeps_as_it_compresses() {
    let statistics = |compressions, free, external| libc::vm_statistics64 {
        compressions,
        free_count: free,
        external_page_count: external,
        ..unsafe { std::mem::zeroed() }
    };
    let snapshot = |statistics, file_floor| Snapshot {
        statistics,
        counted: Counted {
            at: Instant::now(),
            lag: STATISTICS_WINDOW,
        },
        own: 0,
        file_floor,
    };
    let target = 4000;
    assert_eq!(file_floor(None, &statistics(0, 100_000, 120_000), target), 0);
    let rest = snapshot(statistics(10, 100_000, 120_000), 0);
    assert_eq!(
        file_floor(Some(&rest), &statistics(5_000, 100_000, 120_000), target),
        0
    );
    let pressed = statistics(30_000, 4_500, 95_000);
    assert_eq!(file_floor(Some(&rest), &pressed, target), 95_000);
    let kept = snapshot(pressed, 95_000);
    // Still tight near target without new compressions: floor stays standing,
    // clamped to current external pages.
    assert_eq!(
        file_floor(Some(&kept), &statistics(30_000, 7_000, 140_000), target),
        95_000
    );
    assert_eq!(
        file_floor(Some(&kept), &statistics(30_000, 7_000, 80_000), target),
        80_000
    );
    // Modest calm recovery (2x..4x target) halves the temporary floor; full
    // calm recovery (>= 4x target) clears it completely so a past transient
    // spike never permanently blacklists gigabytes of file cache.
    assert_eq!(
        file_floor(Some(&kept), &statistics(30_000, 10_000, 140_000), target),
        47_500
    );
    assert_eq!(
        file_floor(Some(&kept), &statistics(30_000, 60_000, 140_000), target),
        0
    );
    assert_eq!(
        file_floor(Some(&kept), &statistics(60_000, 4_200, 88_000), target),
        88_000
    );
}

/// Live `vm.page_pageable_external_count` and `vm.vm_page_xpmapped_min`
/// yield skew-free clean cache above the kernel's executable floor.
#[test]
fn live_external_respects_xpmapped_min() {
    let mac = Mac::new().counts(&[
        ("vm.page_free_count", 100_000),
        ("vm.vm_page_free_target", 4_000),
        ("vm.page_pageable_external_count", 250_000),
        ("vm.vm_page_xpmapped_min", 100_000),
    ]);
    let s = sample(&mac);
    assert_eq!(s.supply.clean_cache, 150_000 * (16 << 10));
}

/// `iogpu.wired_limit_mb` of zero is the OS's default, which the ledger
/// reads as the device's working set; a limit set is read in bytes.
#[test]
fn a_wired_limit_of_zero_is_the_os_default() {
    assert_eq!(
        wired_limit(&Mac::new().count("iogpu.wired_limit_mb", 0)),
        Some(WiredLimit::OsDefault)
    );
    assert_eq!(
        wired_limit(&Mac::new().count("iogpu.wired_limit_mb", 40_960)),
        Some(WiredLimit::Set(40 * GB))
    );
}

/// macOS 13's `debug.iogpu.wired_limit`, in bytes, is the limit where
/// `iogpu.wired_limit_mb` is not defined; zero is the OS's default.
#[test]
fn macos_13s_wired_limit_is_read_in_bytes() {
    assert_eq!(
        wired_limit(&Mac::new().count("debug.iogpu.wired_limit", 36 * GB)),
        Some(WiredLimit::Set(36 * GB))
    );
    assert_eq!(
        wired_limit(&Mac::new().count("debug.iogpu.wired_limit", 0)),
        Some(WiredLimit::OsDefault)
    );
}

/// A Mac reporting none of the optional sysctls (an Intel Mac defines
/// no IOGPU sysctl) reads each one's documented value: no wired
/// figures, free swap or pressure level.
#[test]
fn the_optional_sysctls_unreported_read_their_documented_values() {
    let mac = Mac::new();
    assert_eq!(wired_limit(&mac), None, "no IOGPU stack wires host RAM");
    assert_eq!(swap_free(&mac), 0);
    assert_eq!(pressure_level(&mac), None);
    assert_eq!(capacity(&mac).value, 64 * GB);
    assert_eq!(
        device_node(
            &mac,
            PciLocation {
                domain: 0,
                bus: 0,
                device: 0
            }
        ),
        HostNode::Any
    );
}

/// Without performance levels every core is one performance level: the
/// physical cores, or each CPU of the quota.
#[test]
fn without_performance_levels_every_core_is_one_performance_level() {
    let intel = processors(&Mac::new().counts(&[("hw.physicalcpu", 6), ("hw.logicalcpu", 12)]));
    assert_eq!(intel.performance_cores(), 6);
    assert!(intel.nodes()[0]
        .cores
        .iter()
        .all(|core| core.threads.len() == 2));
    let counted = processors(&Mac {
        parallelism: NonZeroUsize::new(4).unwrap(),
        ..Mac::new()
    });
    assert_eq!(counted.performance_cores(), 4);
    assert_eq!(counted.nodes(), flat_nodes(counted.allowed()));
    assert_eq!(processors(&Mac::new()).performance_cores(), 1);
}

/// Three performance levels (an Apple M6 reports `hw.nperflevels` 3):
/// level 0 is the performance cores, the rest efficiency cores.
#[test]
fn of_three_performance_levels_the_first_is_the_performance_cores() {
    let mac = Mac::new().counts(&[
        ("hw.nperflevels", 3),
        ("hw.perflevel0.physicalcpu", 4),
        ("hw.perflevel0.logicalcpu", 4),
        ("hw.perflevel1.physicalcpu", 6),
        ("hw.perflevel1.logicalcpu", 6),
        ("hw.perflevel2.physicalcpu", 2),
        ("hw.perflevel2.logicalcpu", 2),
    ]);
    let processors = processors(&mac);
    assert_eq!(processors.nodes()[0].cores.len(), 12);
    assert_eq!(processors.performance_cores(), 4);
    assert!(processors.nodes()[0].cores[..4]
        .iter()
        .all(|core| core.performance));
}

/// The reported caches are the highest level each core type reports,
/// over its instances; where a core type's sharing is not reported, the
/// largest L2 reported; with no cache reported, `CacheSize::UNREPORTED`.
#[test]
fn the_last_level_cache_falls_back_to_the_largest_l2() {
    let levels = Mac::new().counts(&[
        ("hw.nperflevels", 2),
        ("hw.perflevel0.logicalcpu", 10),
        ("hw.perflevel0.l2cachesize", 16 << 20),
        ("hw.perflevel0.cpusperl2", 5),
        ("hw.perflevel1.logicalcpu", 4),
        ("hw.perflevel1.l2cachesize", 4 << 20),
    ]);
    assert_eq!(
        last_level_cache(&levels),
        CacheSize {
            bytes: 16 << 20,
            is_last_level: false,
        },
        "no sharing for level 1: the largest L2"
    );
    let shared = levels.count("hw.perflevel1.cpusperl2", 4);
    assert_eq!(
        last_level_cache(&shared),
        CacheSize {
            bytes: (2 * 16 + 4) << 20,
            is_last_level: false,
        }
    );
    assert_eq!(
        last_level_cache(&Mac::new().count("hw.l2cachesize", 256 << 10)),
        CacheSize {
            bytes: 256 << 10,
            is_last_level: false,
        }
    );
    assert_eq!(last_level_cache(&Mac::new()), CacheSize::UNREPORTED);
}

/// A Mac reporting none of the CPU's identifying sysctls is identified
/// by its architecture alone.
#[test]
fn a_cpu_reporting_no_identity_is_its_architecture() {
    assert_eq!(
        cpu_identity(&Mac::new()),
        format!("unknown;arch={}", std::env::consts::ARCH)
    );
}

#[test]
fn os_build_and_cpu_model_read_sysctls_with_fallbacks() {
    let bare = Mac::new();
    assert_eq!(os_build(&bare), "macOS");
    assert_eq!(cpu_model(&bare), format!("CPU ({})", std::env::consts::ARCH));
    let mac = bare
        .text("kern.osproductversion", "15.7.9")
        .text("kern.osversion", "24G830")
        .text("machdep.cpu.brand_string", "Apple M4 Pro");
    assert_eq!(os_build(&mac), "macOS 15.7.9 (24G830)");
    assert_eq!(cpu_model(&mac), "Apple M4 Pro");
}

#[test]
fn firmware_reads_hw_model_and_system_profiler_json() {
    let mac = Mac {
        profiler_json: Some(
            br#"{"SPHardwareDataType":[{"machine_name":"MacBook Pro"}]}"#.to_vec(),
        ),
        ..Mac::new().text("hw.model", "Mac16,11")
    };
    let fw = firmware(&mac);
    assert_eq!(fw.manufacturer, b"Apple");
    assert_eq!(fw.model, b"Mac16,11\0");
    assert_eq!(fw.name, b"MacBook Pro");
}

/// The machine's sample reads every figure, within the machine.
#[test]
fn the_host_sample_reads_within_the_machine() {
    let sample = sample(&LiveSys);
    assert!(sample.supply.total() <= sample.total);
    assert!(sample.supply.cold_reclaimable <= sample.cold.machine && sample.cold.machine <= sample.total);
    assert_eq!(sample.cold.own, None);
    if let Some(wired) = sample.wired {
        assert!(wired.wire_count > 0 && wired.wire_count < sample.total);
    }
    let harm = sample.harm.expect("macOS counts the task's own paging");
    assert!(harm.own <= harm.machine);
    assert!(
        sample
            .own_reclaimed
            .is_some_and(|own| own.bytes < sample.total && own.stored <= own.bytes && own.exact),
        "macOS reports the task's compressed bytes"
    );
    assert!(sample.counted.at <= Instant::now() && sample.counted.lag <= STATISTICS_WINDOW);
}

/// `WiredFigures::wire_count` prefers IOGPU's `"In use system memory"` when
/// available and falls back to `vm_statistics64.wire_count` otherwise.
#[test]
fn wired_count_prefers_iogpu_in_use_memory_and_falls_back_to_vm_wire_count() {
    let base = Mac::new().counts(&[
        ("vm.page_free_count", 100_000),
        ("vm.vm_page_free_target", 4_000),
        ("iogpu.wired_limit_mb", 0),
    ]);
    let with_iogpu = Mac {
        vm_wire_count: 120_000,
        iogpu_in_use: Some(256 << 20),
        ..base
    };
    let wired = sample(&with_iogpu)
        .wired
        .expect("iogpu.wired_limit_mb reports wired figures");
    assert_eq!(wired.limit, WiredLimit::OsDefault);
    assert_eq!(wired.wire_count, 256 << 20);

    let fallback = Mac {
        iogpu_in_use: None,
        ..with_iogpu
    };
    let wired_fallback = sample(&fallback)
        .wired
        .expect("fallback still reports wired figures");
    assert_eq!(wired_fallback.wire_count, 120_000 * (16 << 10));
}

/// On Apple Silicon, IORegistry `AGXAccelerator` → `PerformanceStatistics` →
/// `"In use system memory"` returns `Some(bytes)` strictly below machine-wide
/// `vm_statistics64.wire_count` (excluding XNU kernel wired pages).
#[test]
fn iogpu_in_use_memory_excludes_kernel_wired_pages() {
    let stats = LiveSys.vm_statistics();
    let vm_wired = u64::from(stats.wire_count) * LiveSys.page();
    let iogpu = LiveSys.iogpu_in_use_memory();
    if cfg!(target_arch = "aarch64") && wired_limit(&LiveSys).is_some() {
        let bytes = iogpu.expect("Apple Silicon AGXAccelerator reports In use system memory");
        assert!(
            bytes < vm_wired,
            "IOGPU in-use system memory ({bytes}) should be less than machine-wide vm_statistics64.wire_count ({vm_wired})"
        );
    } else if let Some(bytes) = iogpu {
        assert!(bytes < vm_wired);
    }
}
