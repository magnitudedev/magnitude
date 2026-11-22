use super::super::{ColdPages, Limit, PressureLevel, ProcessLimitKind};
use super::limits::{diagnostics, limits, own_cgroup};
use super::memory::{host_total, sample};
use super::*;
use seismic_topology::ReportedBy;
use std::time::Duration;

const GB: u64 = 1 << 30;

/// A machine's OS readings over a fixture tree of its own, removed on drop.
struct Machine(Tree);

impl Machine {
    fn new(name: &str) -> Self {
        let root = std::env::temp_dir().join(format!("seismic-linux-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        Self(Tree::new(root))
    }

    /// A machine whose process is in the cgroup `cgroup` (a
    /// `/proc/self/cgroup` text) with the mounts `mountinfo`.
    fn in_cgroup(name: &str, cgroup: &str, mountinfo: &str) -> Self {
        let machine = Self::new(name);
        machine.file("/proc/self/cgroup", cgroup);
        machine.file("/proc/self/mountinfo", mountinfo);
        machine
    }

    fn refuse(mut self, path: &str) -> Self {
        self.0.refused.push(PathBuf::from(path));
        self
    }
}

impl std::ops::Deref for Machine {
    type Target = Tree;
    fn deref(&self) -> &Tree {
        &self.0
    }
}

impl std::ops::DerefMut for Machine {
    fn deref_mut(&mut self) -> &mut Tree {
        &mut self.0
    }
}

impl Drop for Machine {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0.root);
    }
}

fn cgroup_v2(cgroup: &str, limit_bytes: u64, used_bytes: u64) -> Limit {
    Limit {
        kind: ProcessLimitKind::CgroupV2 { cgroup: cgroup.into() },
        limit_bytes,
        used_bytes,
        reclaimable_bytes: 0,
    }
}

/// The v1 page counter's maximum as `hierarchical_memory_limit` shows it
/// for no limit, of 4 KiB pages.
fn v1_unlimited() -> u64 {
    i64::MAX as u64 / 4096 * 4096
}

const V2_MOUNT: &str = "30 25 0:26 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime shared:4 - cgroup2 cgroup2 rw,nsdelegate\n";
const V1_MOUNTS: &str = "\
24 21 0:22 / /sys/fs/cgroup ro,nosuid,nodev,noexec shared:9 - tmpfs tmpfs ro,mode=755
33 24 0:29 / /sys/fs/cgroup/memory rw,nosuid,nodev,noexec,relatime shared:15 - cgroup cgroup rw,memory
34 24 0:30 / /sys/fs/cgroup/cpu,cpuacct rw,nosuid,nodev,noexec,relatime shared:16 - cgroup cgroup rw,cpu,cpuacct
";

/// 16 GiB with 8 GiB free, as `/proc/meminfo` writes it.
const MEMINFO: &str = "MemTotal:       16777216 kB\nMemFree:         8388608 kB\n";

// Fallbacks: each reading past its last source is its documented value.

/// The floor is the zones' low watermarks summed, in pages.
#[test]
fn the_floor_is_the_zones_low_watermarks() {
    let machine = Machine::new("floor-zoneinfo");
    machine.file("/proc/meminfo", MEMINFO);
    machine.file(
        "/proc/zoneinfo",
        "Node 0, zone   Normal\n  pages free     3\n        min      10\n        low      12\n        high     14\n        protection: (0, 0)\nNode 0, zone DMA32\n        low      5\n",
    );
    let s = sample(&*machine);
    assert_eq!(s.supply.free_beyond_target, (8 * GB - 17 * 4096) as i64);
    assert_eq!(s.reclaim_band, 2 * 4096);
}

/// Where `/proc/zoneinfo` is refused, the floor is the kernel's formula
/// from its tunables: `min_free + max(min_free / 4, total × scale /
/// 10000)`.
#[test]
fn the_floor_is_the_kernels_formula_where_zoneinfo_is_refused() {
    let machine = Machine::new("floor-formula").refuse("/proc/zoneinfo");
    machine.file("/proc/meminfo", MEMINFO);
    machine.file("/proc/sys/vm/min_free_kbytes", "67584\n");
    machine.file("/proc/sys/vm/watermark_scale_factor", "10\n");
    let min_free = 67584 * 1024;
    let distance = (min_free / 4).max(16 * GB * 10 / 10_000);
    let floor = min_free + distance;
    assert_eq!(floor, 86_507_520);
    let s = sample(&*machine);
    assert_eq!(s.supply.free_beyond_target, (8 * GB - floor) as i64);
    assert_eq!(s.reclaim_band, distance);
}

/// Where the tunables are refused too, the formula takes the kernel's
/// defaults: `min_free_kbytes` = `int_sqrt(16 GiB in KiB × 16)` = 16384
/// KiB, scale factor 10.
#[test]
fn the_floor_takes_the_kernels_defaults_where_the_tunables_are_refused() {
    let machine = Machine::new("floor-defaults")
        .refuse("/proc/zoneinfo")
        .refuse("/proc/sys/vm");
    machine.file("/proc/meminfo", MEMINFO);
    machine.file("/proc/sys/vm/min_free_kbytes", "67584\n");
    let floor = 16384 * 1024 + 16 * GB * 10 / 10_000;
    assert_eq!(sample(&*machine).supply.free_beyond_target, (8 * GB - floor) as i64);
}

/// Where `/proc/meminfo` is missing, `MemTotal` and `MemFree` are
/// `sysinfo`'s, tagged with `ReportedBy::LinuxSysinfo`.
#[test]
fn total_and_free_are_sysinfos_where_meminfo_is_missing() {
    let mut machine = Machine::new("sysinfo");
    machine.sysinfo = (32 * GB, 20 * GB);
    machine.file("/proc/zoneinfo", "Node 0, zone   Normal\n        low      1\n");
    let total = host_total(&*machine);
    assert_eq!((total.value, total.by), (32 * GB, ReportedBy::LinuxSysinfo));
    let sample = sample(&*machine);
    assert_eq!(sample.total, 32 * GB);
    assert_eq!(sample.supply.free_beyond_target, (20 * GB - 4096) as i64);
}

/// Clean cache, cold pages and free swap are 0 where `/proc/meminfo` has
/// none of them, or not all of the cache's four counts.
#[test]
fn clean_cache_cold_and_swap_are_zero_where_meminfo_has_none() {
    let machine = Machine::new("no-cache");
    machine.file("/proc/meminfo", MEMINFO);
    let sample_of = |meminfo: &str| {
        machine.file("/proc/meminfo", meminfo);
        sample(&*machine)
    };
    let bare = sample_of(MEMINFO);
    assert_eq!(
        (bare.supply.clean_cache, bare.cold.machine, bare.supply.cold_reclaimable),
        (0, 0, 0)
    );
    let no_dirty = sample_of(&format!(
        "{MEMINFO}Active(file): 100 kB\nInactive(file): 100 kB\nWriteback: 0 kB\nInactive(anon): 50 kB\n"
    ));
    assert_eq!(
        (no_dirty.supply.clean_cache, no_dirty.cold.machine, no_dirty.supply.cold_reclaimable),
        (0, 50 * 1024, 0)
    );
    let whole = sample_of(&format!(
        "{MEMINFO}Active(file): 100 kB\nInactive(file): 100 kB\nDirty: 20 kB\nWriteback: 10 kB\nInactive(anon): 50 kB\nSwapFree: 30 kB\n"
    ));
    assert_eq!(
        (whole.supply.clean_cache, whole.cold.machine, whole.supply.cold_reclaimable),
        (170 * 1024, 50 * 1024, 30 * 1024)
    );
    let with_shmem = sample_of(&format!(
        "{MEMINFO}Active(file): 100 kB\nInactive(file): 100 kB\nDirty: 20 kB\nWriteback: 10 kB\nShmem: 80 kB\nMapped: 80 kB\n"
    ));
    assert_eq!(
        with_shmem.supply.clean_cache,
        170 * 1024,
        "Shmem is on anon/unevictable LRU, never in Active(file)+Inactive(file); mapped Shmem must not reduce clean file cache"
    );
    let with_mapped = sample_of(&format!(
        "{MEMINFO}Active(file): 100 kB\nInactive(file): 100 kB\nDirty: 20 kB\nWriteback: 10 kB\nShmem: 50 kB\nMapped: 90 kB\n"
    ));
    assert_eq!(
        with_mapped.supply.clean_cache,
        130 * 1024,
        "only mapped regular-file pages (Mapped - Shmem = 40 kB) are excluded from clean cache"
    );
}

/// With no PSI anywhere there is no pressure level; with the machine's
/// refused, the cgroup's is it; with both, the higher. Sub-percent micro
/// stalls remain Normal; sustained >= 1% some stall is Warn, >= 0.5% full is
/// Critical.
#[test]
fn pressure_is_none_without_psi_and_the_cgroups_without_the_machines() {
    let psi = |some: &str, full: &str| {
        format!("some avg10={some} avg60=0.00 avg300=0.00 total=1\nfull avg10={full} avg60=0.00 avg300=0.00 total=0\n")
    };
    let machine = Machine::in_cgroup("pressure", "0::/app\n", V2_MOUNT);
    machine.file("/proc/meminfo", MEMINFO);
    assert_eq!(sample(&*machine).pressure, None);
    machine.file("/sys/fs/cgroup/app/memory.pressure", &psi("0.12", "0.00"));
    assert_eq!(sample(&*machine).pressure, Some(PressureLevel::Normal));
    machine.file("/sys/fs/cgroup/app/memory.pressure", &psi("1.50", "0.00"));
    assert_eq!(sample(&*machine).pressure, Some(PressureLevel::Warn));
    machine.file("/proc/pressure/memory", &psi("0.00", "0.00"));
    assert_eq!(sample(&*machine).pressure, Some(PressureLevel::Warn));
    machine.file("/proc/pressure/memory", &psi("1.50", "0.60"));
    assert_eq!(sample(&*machine).pressure, Some(PressureLevel::Critical));
}

/// Without `/proc/vmstat` there is no harm, even with the own cgroup's
/// refaults.
#[test]
fn harm_is_none_without_vmstat() {
    let machine = Machine::in_cgroup("no-harm", "0::/app.scope\n", V2_MOUNT);
    machine.file("/proc/meminfo", MEMINFO);
    machine.file(
        "/sys/fs/cgroup/app.scope/memory.stat",
        "workingset_refault_anon 5\nworkingset_refault_file 6\n",
    );
    machine.file(
        "/sys/fs/cgroup/app.scope/cgroup.procs",
        &format!("{}\n", std::process::id()),
    );
    machine.file("/sys/fs/cgroup/app.scope/cgroup.stat", "nr_descendants 0\n");
    assert_eq!(sample(&*machine).harm, None);
    machine.file(
        "/proc/vmstat",
        "workingset_refault_anon 30\nworkingset_refault_file 40\n",
    );
    assert_eq!(
        sample(&*machine).harm,
        Some(super::super::Harm { machine: 70, own: 11 })
    );
}

/// Where `vm.stat_interval` is missing it is the kernel's default, 1 s:
/// the counters lag by it rounded up to the next second, 2 s.
#[test]
fn stat_interval_is_one_second_where_missing() {
    let machine = Machine::new("stat-interval");
    machine.file("/proc/meminfo", MEMINFO);
    let lag = sample(&*machine).counted.lag;
    assert!(lag >= Duration::from_secs(2) && lag < Duration::from_secs(3), "{lag:?}");
}

/// Where no cgroup hierarchy is mounted, no cgroup limits the process,
/// and no container line is reported.
#[test]
fn no_cgroup_mount_gives_no_limit() {
    let machine = Machine::in_cgroup("unmounted", "0::/docker/abc\n", "");
    assert!(limits(&*machine).is_empty());
    assert!(diagnostics(&*machine).is_empty());
    let v1 = Machine::in_cgroup("unmounted-v1", "2:memory:/docker/abc\n", "");
    assert!(limits(&*v1).is_empty());
}

// Defects fixed.

/// cgroup v2: every visible level that sets a limit is one, innermost
/// first, each the lower of its `memory.max` and `memory.high`; the root
/// (no `memory.max`) none.
#[test]
fn v2_reports_every_limited_level_up_to_the_root() {
    let machine = Machine::in_cgroup(
        "v2-levels",
        "0::/user.slice/user-1.slice/app.scope\n",
        V2_MOUNT,
    );
    let level = |path: &str, max: &str, high: &str, current: &str| {
        machine.file(&format!("/sys/fs/cgroup{path}/memory.max"), max);
        machine.file(&format!("/sys/fs/cgroup{path}/memory.high"), high);
        machine.file(&format!("/sys/fs/cgroup{path}/memory.current"), current);
    };
    level("/user.slice/user-1.slice/app.scope", "max\n", "max\n", "400\n");
    level("/user.slice/user-1.slice", "5000\n", "4000\n", "900\n");
    level("/user.slice", "8000\n", "max\n", "1200\n");
    assert_eq!(
        limits(&*machine),
        [
            cgroup_v2("/user.slice/user-1.slice", 4000, 900),
            cgroup_v2("/user.slice", 8000, 1200)
        ]
    );
}

/// A level whose usage is missing is no limit, never an unknown one.
#[test]
fn a_v2_level_without_its_usage_is_no_limit() {
    let machine = Machine::in_cgroup("v2-no-usage", "0::/app\n", V2_MOUNT);
    machine.file("/sys/fs/cgroup/app/memory.max", "1000\n");
    assert!(limits(&*machine).is_empty());
}

/// cgroup v1: the own cgroup's `hierarchical_memory_limit`, an ancestor's
/// below its own (past any namespace), against the usage of the level that
/// sets it, which its siblings share, with that level's hierarchical clean
/// cache.
#[test]
fn v1_reads_the_hierarchical_limit_below_the_own_levels() {
    let machine = Machine::in_cgroup(
        "v1-hierarchical",
        "4:memory:/system.slice/svc.service\n3:cpu,cpuacct:/system.slice/svc.service\n1:name=systemd:/system.slice/svc.service\n",
        V1_MOUNTS,
    );
    let own = "/sys/fs/cgroup/memory/system.slice/svc.service";
    let slice = "/sys/fs/cgroup/memory/system.slice";
    machine.file(&format!("{own}/memory.limit_in_bytes"), &v1_unlimited().to_string());
    machine.file(&format!("{own}/memory.usage_in_bytes"), "1000\n");
    machine.file(
        &format!("{own}/memory.stat"),
        "hierarchical_memory_limit 3000\ntotal_active_file 1\n",
    );
    machine.file(&format!("{slice}/memory.limit_in_bytes"), "3000\n");
    machine.file(&format!("{slice}/memory.usage_in_bytes"), "2500\n");
    machine.file(
        &format!("{slice}/memory.stat"),
        "hierarchical_memory_limit 3000\ntotal_active_file 200\ntotal_inactive_file 400\ntotal_dirty 100\ntotal_writeback 50\n",
    );
    assert_eq!(
        limits(&*machine),
        [Limit {
            kind: ProcessLimitKind::CgroupV1 {
                cgroup: "/system.slice".into()
            },
            limit_bytes: 3000,
            used_bytes: 2500,
            reclaimable_bytes: 450,
        }]
    );
    machine.file(
        &format!("{own}/memory.stat"),
        &format!("hierarchical_memory_limit {}\n", v1_unlimited()),
    );
    assert!(limits(&*machine).is_empty(), "no limit up the hierarchy");
}

/// A hybrid host's memory limits are its v1 hierarchy's, a container's
/// v1 mount rooted at its own cgroup.
#[test]
fn a_hybrid_container_reads_its_v1_cgroup() {
    let machine = Machine::in_cgroup(
        "hybrid-container",
        "13:misc:/docker/abc\n2:memory:/docker/abc\n1:name=systemd:/docker/abc\n0::/docker/abc\n",
        "1294 1287 0:78 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime - tmpfs tmpfs rw,mode=755\n\
         1296 1294 0:34 /docker/abc /sys/fs/cgroup/memory ro,nosuid,nodev,noexec,relatime master:16 - cgroup cgroup rw,memory\n",
    );
    machine.file("/sys/fs/cgroup/memory/memory.usage_in_bytes", "1500\n");
    machine.file("/sys/fs/cgroup/memory/memory.stat", "hierarchical_memory_limit 4000\n");
    assert_eq!(
        limits(&*machine),
        [Limit {
            kind: ProcessLimitKind::CgroupV1 { cgroup: "/".into() },
            limit_bytes: 4000,
            used_bytes: 1500,
            reclaimable_bytes: 0,
        }]
    );
}

/// A cgroup v2 limit credits only clean file cache, never more than its
/// usage, and binds the sample where it leaves the least.
#[test]
fn v2_headroom_credits_only_clean_file_cache() {
    let machine = Machine::in_cgroup("v2-cache", "0::/worker\n", V2_MOUNT);
    machine.file("/sys/fs/cgroup/worker/memory.max", "1000\n");
    machine.file("/sys/fs/cgroup/worker/memory.current", "900\n");
    for (stat, reclaimable, remaining) in [
        ("active_file 200\ninactive_file 400\nfile_dirty 100\nfile_writeback 50\nanon 300\n", 450, 550),
        ("active_file 200\ninactive_file 400\nfile_dirty 100\nfile_writeback 50\nfile_mapped 150\n", 300, 400),
        ("active_file 300\ninactive_file 300\nfile_dirty 700\nfile_writeback 0\n", 0, 100),
        ("active_file 0\ninactive_file 600\nfile_writeback 0\n", 0, 100),
        ("active_file 600\ninactive_file 600\nfile_dirty 0\nfile_writeback 0\n", 900, 1000),
    ] {
        machine.file("/sys/fs/cgroup/worker/memory.stat", stat);
        let limits = limits(&*machine);
        assert_eq!((limits[0].used_bytes, limits[0].reclaimable_bytes), (900, reclaimable));
        assert_eq!(limits[0].scope(u64::MAX).supply.balance(), remaining);
    }
    // A real multi-GiB container (e.g. 29.2 GiB rtx5090-bench with swap=0)
    // reserves a 2 GiB OOM-safety floor below `memory.max` and excludes `file_mapped`.
    let limit = 30 * GB;
    let current = 20 * GB;
    machine.file("/sys/fs/cgroup/worker/memory.max", &format!("{limit}\n"));
    machine.file("/sys/fs/cgroup/worker/memory.current", &format!("{current}\n"));
    machine.file(
        "/sys/fs/cgroup/worker/memory.stat",
        &format!(
            "active_file {}\ninactive_file {}\nfile_dirty 0\nfile_writeback 0\nfile_mapped {}\n",
            2 * GB,
            2 * GB,
            GB / 2
        ),
    );
    let container = &limits(&*machine)[0];
    assert_eq!(container.used_bytes, current + 2 * GB, "2 GiB OOM-safety floor reserved");
    assert_eq!(container.reclaimable_bytes, 4 * GB - GB / 2, "file_mapped excluded from clean cache");
    assert_eq!(container.scope(u64::MAX).supply.balance(), (limit - current - 2 * GB + 4 * GB - GB / 2) as i64);
    // Small real container (128 MiB) scales proportionally without exceeding the limit.
    let small_limit = 128 * 1024 * 1024;
    machine.file("/sys/fs/cgroup/worker/memory.max", &format!("{small_limit}\n"));
    machine.file("/sys/fs/cgroup/worker/memory.current", "0\n");
    machine.file("/sys/fs/cgroup/worker/memory.stat", "active_file 0\ninactive_file 0\nfile_dirty 0\nfile_writeback 0\n");
    assert_eq!(limits(&*machine)[0].used_bytes, small_limit / 12);
}

/// `RLIMIT_AS` is a limit enforced against `VmSize`.
#[test]
fn rlimit_as_is_a_limit_against_vmsize() {
    let mut machine = Machine::new("rlimit-as");
    machine.address_space_limit = Some(8 * GB);
    machine.file("/proc/self/status", "VmPeak:\t 2097152 kB\nVmSize:\t 1048576 kB\n");
    assert_eq!(
        limits(&*machine),
        [Limit {
            kind: ProcessLimitKind::AddressSpace,
            limit_bytes: 8 * GB,
            used_bytes: GB,
            reclaimable_bytes: 0,
        }]
    );
    machine.address_space_limit = None;
    assert!(limits(&*machine).is_empty());
}

/// `cpuset.mems` allowing a strict subset of the nodes with memory limits
/// the process to theirs, their free pages above their own zones' low
/// watermarks; allowing every one, to none.
#[test]
fn cpuset_mems_a_strict_subset_limits_to_those_nodes() {
    let machine = Machine::new("cpuset-mems");
    machine.file("/sys/devices/system/node/has_memory", "0-2\n");
    machine.file(
        "/proc/zoneinfo",
        "Node 0, zone   Normal\n        low      1000\nNode 1, zone   Normal\n        low      100\nNode 2, zone   Normal\n        low      50\n",
    );
    machine.file(
        "/sys/devices/system/node/node1/meminfo",
        "Node 1 MemTotal:       16777216 kB\nNode 1 MemFree:         4194304 kB\nNode 1 Active(file):      102400 kB\nNode 1 Inactive(file):    102400 kB\nNode 1 Dirty:                  0 kB\nNode 1 Writeback:              0 kB\n",
    );
    machine.file(
        "/sys/devices/system/node/node2/meminfo",
        "Node 2 MemTotal:       16777216 kB\nNode 2 MemFree:        12582912 kB\n",
    );
    machine.file("/proc/self/status", "Mems_allowed:\t00000006\nMems_allowed_list:\t1-2\n");
    assert_eq!(
        limits(&*machine),
        [Limit {
            kind: ProcessLimitKind::AllowedNodes { nodes: vec![1, 2] },
            limit_bytes: 32 * GB,
            used_bytes: 16 * GB + 150 * 4096,
            reclaimable_bytes: 200 * 1024 * 1024,
        }]
    );
    machine.file("/proc/self/status", "Mems_allowed_list:\t0-2\n");
    assert!(limits(&*machine).is_empty());
}

/// A private namespace's root setting no memory limit is reported once;
/// one with rtx3090-b's 79.3 GB `memory.max` is a limit, not reported.
#[test]
fn an_unlimited_container_root_is_reported() {
    let machine = Machine::in_cgroup("container", "0::/\n", V2_MOUNT);
    machine.file("/sys/fs/cgroup/memory.max", "max\n");
    machine.file("/sys/fs/cgroup/memory.high", "max\n");
    machine.file("/sys/fs/cgroup/memory.current", "1000\n");
    let reported = diagnostics(&*machine);
    assert_eq!(reported.len(), 1);
    assert_eq!(reported[0].source, seismic_topology::DiagnosticSource::Host);
    assert!(reported[0].message.starts_with("container memory limit not set"));
    assert!(limits(&*machine).is_empty());
    machine.file("/sys/fs/cgroup/memory.max", "79337357312\n");
    assert!(diagnostics(&*machine).is_empty());
    assert_eq!(limits(&*machine), [cgroup_v2("/", 79_337_357_312, 1000 + 2 * GB)]);
}

/// The host's own root cgroup (no `memory.max`) is no container.
#[test]
fn the_hosts_root_cgroup_is_not_reported() {
    let machine = Machine::in_cgroup("host-root", "0::/\n", V2_MOUNT);
    assert!(diagnostics(&*machine).is_empty());
}

// The own cgroup.

/// A cgroup v2 of this process alone is its own: its refaults, cold and
/// zswapped pages are this process's.
#[test]
fn a_cgroup_of_this_process_alone_counts_its_own_paging_and_cold_pages() {
    let machine = Machine::in_cgroup("own", "0::/app.scope\n", V2_MOUNT);
    machine.file("/proc/meminfo", MEMINFO);
    machine.file(
        "/proc/vmstat",
        "workingset_refault_anon 30\nworkingset_refault_file 40\n",
    );
    machine.file("/proc/self/status", "VmSwap:\t     100 kB\n");
    machine.file(
        "/sys/fs/cgroup/app.scope/memory.stat",
        "inactive_anon 300\nzswap 70\nworkingset_refault_anon 5\nworkingset_refault_file 6\n",
    );
    machine.file(
        "/sys/fs/cgroup/app.scope/cgroup.procs",
        &format!("{}\n", std::process::id()),
    );
    machine.file(
        "/sys/fs/cgroup/app.scope/cgroup.stat",
        "nr_descendants 0\nnr_dying_descendants 0\n",
    );
    assert_eq!(
        own_cgroup(&*machine),
        Some(PathBuf::from("/sys/fs/cgroup/app.scope"))
    );
    let sample = sample(&*machine);
    assert_eq!(
        sample.harm,
        Some(super::super::Harm { machine: 70, own: 11 })
    );
    assert_eq!(sample.cold, ColdPages { machine: 0, own: Some(300) });
    assert_eq!(
        sample.own_reclaimed.map(|own| (own.bytes, own.stored)),
        Some((100 * 1024, 70))
    );
}

/// A cgroup this process shares, by another member or a descendant, or
/// under cgroup v1 alone, is not its own: no count of it is this
/// process's.
#[test]
fn a_shared_cgroup_is_not_the_processs_own() {
    let pid = std::process::id();
    let own = |name: &str, procs: String, descendants: u32| {
        let machine = Machine::in_cgroup(name, "0::/session-2.scope\n", V2_MOUNT);
        machine.file("/sys/fs/cgroup/session-2.scope/cgroup.procs", &procs);
        machine.file(
            "/sys/fs/cgroup/session-2.scope/cgroup.stat",
            &format!("nr_descendants {descendants}\n"),
        );
        own_cgroup(&*machine)
    };
    assert!(own("shared-sole", format!("{pid}\n"), 0).is_some());
    assert_eq!(
        own("shared-member", format!("{pid}\n1\n"), 0),
        None,
        "another process in the cgroup"
    );
    assert_eq!(
        own("shared-descendant", format!("{pid}\n"), 1),
        None,
        "a descendant's counts in the cgroup's"
    );
    let v1 = Machine::in_cgroup("shared-v1", "4:memory:/svc\n", V1_MOUNTS);
    v1.file("/sys/fs/cgroup/memory/svc/cgroup.procs", &format!("{pid}\n"));
    assert_eq!(own_cgroup(&*v1), None);
}

/// `cpu_model` reads `/proc/cpuinfo`'s `model name`, falling back through ARM MIDR,
/// `/proc/cpuinfo`'s `Hardware`, `/proc/device-tree/model`, and finally `CPU (<arch>)`.
#[test]
fn cpu_model_reads_cpuinfo_or_falls_back() {
    let machine = Machine::new("cpu-model");
    assert_eq!(cpu_model(&*machine), format!("CPU ({})", std::env::consts::ARCH));
    machine.file("/proc/cpuinfo", "processor\t: 0\nmodel name\t: AMD EPYC 7763 64-Core Processor\n");
    assert_eq!(cpu_model(&*machine), "AMD EPYC 7763 64-Core Processor");
}

#[test]
fn cpu_model_decodes_arm_midr_graviton_and_big_little() {
    let graviton4 = Machine::new("cpu-model-graviton4");
    graviton4.file(
        "/proc/cpuinfo",
        "\
processor\t: 0
BogoMIPS\t: 2100.00
Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics
CPU implementer\t: 0x41
CPU architecture: 8
CPU variant\t: 0x0
CPU part\t: 0xd4f
CPU revision\t: 1

processor\t: 1
CPU implementer\t: 0x41
CPU architecture: 8
CPU variant\t: 0x0
CPU part\t: 0xd4f
CPU revision\t: 1
",
    );
    assert_eq!(cpu_model(&*graviton4), "ARM Neoverse-V2");

    let graviton2 = Machine::new("cpu-model-graviton2");
    graviton2.file(
        "/proc/cpuinfo",
        "processor\t: 0\nCPU implementer\t: 0x41\nCPU part\t: 0xd0c\n",
    );
    assert_eq!(cpu_model(&*graviton2), "ARM Neoverse-N1");

    let big_little = Machine::new("cpu-model-big-little");
    big_little.file(
        "/proc/cpuinfo",
        "\
processor\t: 0
CPU implementer\t: 0x41
CPU part\t: 0xd05

processor\t: 1
CPU implementer\t: 0x41
CPU part\t: 0xd05

processor\t: 2
CPU implementer\t: 0x41
CPU part\t: 0xd0b

processor\t: 3
CPU implementer\t: 0x41
CPU part\t: 0xd0b
",
    );
    assert_eq!(cpu_model(&*big_little), "ARM Cortex-A55 / ARM Cortex-A76");
}

#[test]
fn cpu_model_falls_back_to_hardware_and_device_tree_model() {
    let dt = Machine::new("cpu-model-device-tree");
    dt.file(
        "/proc/cpuinfo",
        "processor\t: 0\nCPU implementer\t: 0x99\nCPU part\t: 0x999\n",
    );
    dt.file("/proc/device-tree/model", "Raspberry Pi 5 Model B Rev 1.0\0");
    assert_eq!(cpu_model(&*dt), "Raspberry Pi 5 Model B Rev 1.0");

    let hw = Machine::new("cpu-model-hardware");
    hw.file(
        "/proc/cpuinfo",
        "processor\t: 0\nCPU implementer\t: 0x99\nCPU part\t: 0x999\nHardware\t: BCM2712\n",
    );
    hw.file("/proc/device-tree/model", "Raspberry Pi 5 Model B Rev 1.0\0");
    assert_eq!(cpu_model(&*hw), "BCM2712");
}

/// `last_level_cache` returns `CacheSize::UNREPORTED` when cacheinfo is absent.
#[test]
fn last_level_cache_is_unreported_without_sysfs() {
    let machine = Machine::new("no-cacheinfo");
    assert_eq!(last_level_cache(&*machine), CacheSize::UNREPORTED);
}

/// A binding cgroup charges `memory.current` synchronously: `lag.growth` is
/// `Duration::ZERO` when its clean file cache is 0 OR when `free_beyond_target > 0`
/// (headroom below `memory.max - floor`), and `FLUSH_TIME` (2 s) only when
/// `free_beyond_target <= 0` and clean cache from `memory.stat` can be evicted.
#[test]
fn cgroup_growth_lag_is_zero_without_clean_cache() {
    let machine = Machine::in_cgroup("cgroup-lag", "0::/worker\n", V2_MOUNT);
    machine.file("/proc/meminfo", MEMINFO);
    machine.file("/sys/fs/cgroup/worker/memory.max", &format!("{}\n", 4 * GB));
    machine.file("/sys/fs/cgroup/worker/memory.current", &format!("{}\n", GB));
    machine.file(
        "/sys/fs/cgroup/worker/memory.stat",
        "active_file 0\ninactive_file 0\nfile_dirty 0\nfile_writeback 0\n",
    );
    let no_cache = sample(&*machine);
    assert_eq!(no_cache.total, 4 * GB, "cgroup binds over host");
    assert_eq!(no_cache.supply.clean_cache, 0);
    assert_eq!(no_cache.lag.growth, Duration::ZERO);

    // Clean cache present, but `memory.current` (1 GiB) is well below
    // `memory.max - floor` (4 GiB - 341 MiB), so `free_beyond_target > 0`
    // and growth charges `memory.current` synchronously without evicting cache.
    machine.file(
        "/sys/fs/cgroup/worker/memory.stat",
        "active_file 4096\ninactive_file 0\nfile_dirty 0\nfile_writeback 0\n",
    );
    let with_headroom = sample(&*machine);
    assert_eq!(with_headroom.total, 4 * GB, "cgroup binds over host");
    assert_eq!(with_headroom.supply.clean_cache, 4096);
    assert!(with_headroom.supply.free_beyond_target > 0);
    assert_eq!(with_headroom.lag.growth, Duration::ZERO);

    // Once `memory.current` reaches `memory.max - floor` (`free_beyond_target <= 0`)
    // and clean cache is present, growth can evict cache whose drop in
    // `memory.stat` lags within `FLUSH_TIME` (2 s).
    machine.file("/sys/fs/cgroup/worker/memory.current", &format!("{}\n", 4 * GB));
    let at_limit = sample(&*machine);
    assert_eq!(at_limit.total, 4 * GB, "cgroup binds over host");
    assert_eq!(at_limit.supply.clean_cache, 4096);
    assert!(at_limit.supply.free_beyond_target <= 0);
    assert_eq!(at_limit.lag.growth, Duration::from_secs(2));
}

/// Regression: CUDA `cuMemHostAlloc` allocates multi-GiB `shmem` that Linux
/// accounts on the anon/unevictable LRU (never `active_file + inactive_file`)
/// while still counting mapped `shmem` in `file_mapped`. `v2_limit` must
/// subtract only regular-file mappings (`file_mapped - shmem`), never `shmem`
/// itself or `shmem` twice.
#[test]
fn v2_clean_cache_excludes_only_regular_file_mappings_not_shmem() {
    let machine = Machine::in_cgroup("v2-cuda-shmem", "0::/worker\n", V2_MOUNT);
    let limit = 45 * GB;
    let current = 40 * GB;
    let shmem = 9 * GB;
    let regular_mapped = 512 * 1024 * 1024;
    let file_mapped = shmem + regular_mapped;
    let active_file = 6 * GB;
    let inactive_file = 12 * GB;
    machine.file("/sys/fs/cgroup/worker/memory.max", &format!("{limit}\n"));
    machine.file("/sys/fs/cgroup/worker/memory.current", &format!("{current}\n"));
    machine.file(
        "/sys/fs/cgroup/worker/memory.stat",
        &format!(
            "active_file {active_file}\ninactive_file {inactive_file}\nfile_dirty 0\nfile_writeback 0\nshmem {shmem}\nfile_mapped {file_mapped}\n"
        ),
    );
    let container = &limits(&*machine)[0];
    assert_eq!(
        container.reclaimable_bytes,
        active_file + inactive_file - regular_mapped,
        "9 GiB cuMemHostAlloc shmem must not be subtracted from 18 GiB droppable file cache"
    );
}

/// Reproduces `a56890` on a 64 GiB Vast.ai Linux container (`memory.max = 60 GiB`,
/// `memory.swap.max = 0`, physical `MemTotal = 64 GiB`, host `SwapFree = 8 GiB`):
/// 1. `v2_limit` reads `memory.swap.max = 0` and reserves `min(limit / 12, 4 GiB)`
///    instead of capping at 2 GiB (leaving room for uncharged serving/driver RSS
///    and `file_mapped` fault-ins that cannot swap).
/// 2. `binding()` bounds immediate non-swapping supply (`free_beyond_target +
///    clean_cache`) by the physical host even when host `SwapFree` inflates the
///    host's total `balance()` above the cgroup's.
#[test]
fn swapless_v2_cgroup_reserves_swapless_floor_and_never_masks_host_immediate_exhaustion() {
    let machine = Machine::in_cgroup("v2-swapless-64g", "0::/worker\n", V2_MOUNT);
    let host_total_kib = 64 * 1024 * 1024;
    // Physical host has only 1 GiB free + 0.5 GiB clean cache, plus 8 GiB swap-backed cold anon.
    machine.file(
        "/proc/meminfo",
        &format!(
            "MemTotal: {host_total_kib} kB\n\
             MemFree: 1048576 kB\n\
             Active(file): 262144 kB\n\
             Inactive(file): 262144 kB\n\
             Dirty: 0 kB\n\
             Writeback: 0 kB\n\
             Inactive(anon): 8388608 kB\n\
             SwapFree: 8388608 kB\n"
        ),
    );
    machine.file("/proc/zoneinfo", "Node 0, zone Normal\n        low 1\n");
    let limit = 60 * GB;
    let current = 52 * GB;
    machine.file("/sys/fs/cgroup/worker/memory.max", &format!("{limit}\n"));
    machine.file("/sys/fs/cgroup/worker/memory.current", &format!("{current}\n"));
    machine.file("/sys/fs/cgroup/worker/memory.swap.max", "0\n");
    machine.file(
        "/sys/fs/cgroup/worker/memory.stat",
        "active_file 0\ninactive_file 0\nfile_dirty 0\nfile_writeback 0\n",
    );

    // 1. Verify swapless cgroup_floor is 4 GiB (`min(60 GiB / 12, 4 GiB)`), not 2 GiB.
    let container = &limits(&*machine)[0];
    assert_eq!(container.used_bytes, current + 4 * GB);

    // 2. Verify sample() does NOT let cgroup's 4 GiB immediate supply hide
    //    the physical host's 1.5 GiB immediate supply behind 8 GiB host swap.
    let s = sample(&*machine);
    assert_eq!(s.total, limit);
    assert_eq!(
        s.supply.free_beyond_target + s.supply.clean_cache as i64,
        (GB + GB / 2 - 4096) as i64,
        "immediate non-swapping supply must stay bounded by the physical host (1.5 GiB - 1 page low watermark)"
    );
    assert_eq!(
        s.supply.balance(),
        (4 * GB) as i64,
        "total pageable balance stays bounded by the cgroup (4 GiB)"
    );
}
