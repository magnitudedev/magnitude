//! GNU Linux: `/proc/meminfo`, `/proc/zoneinfo`, `/proc/vmstat`, PSI and
//! cgroup memory limits and counts.
//!
//! `MemTotal` is OS-usable RAM, not installed RAM; `MemAvailable` is the
//! kernel's estimate. cgroup limits are read from this process's cgroup and
//! every visible ancestor, in whichever hierarchy the memory controller is
//! bound to: a v1 `memory` hierarchy (legacy and hybrid hosts) or the v2
//! unified hierarchy. Ancestors hidden by a cgroup namespace or a container's
//! mount are reported as such, never presumed unlimited.

use super::{Counted, Harm, HostSample, Lag, ObservationError, OwnCold, OwnReclaimed, PressureLevel, ProcessLimitKind, ProcessMemoryLimit, Supply};
use std::time::{Duration, Instant};
use seismic_topology::{Reported, ReportedBy};
use std::path::{Path, PathBuf};

/// The `/proc/cpuinfo` fields that identify a processor (x86's and Arm's):
/// every processor's, heterogeneous cores included, and no live clock,
/// utilization or calibration counter.
const CPU_IDENTITY_FIELDS: [&str; 16] = [
    "processor",
    "vendor_id",
    "cpu family",
    "model",
    "model name",
    "stepping",
    "flags",
    "Features",
    "CPU implementer",
    "CPU architecture",
    "CPU variant",
    "CPU part",
    "CPU revision",
    "Hardware",
    "Revision",
    "Serial",
];

pub(super) fn cpu_identity() -> Result<String, String> {
    let report = std::fs::read_to_string("/proc/cpuinfo").map_err(|error| error.to_string())?;
    let fields = report
        .lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(':')?;
            let key = key.trim();
            CPU_IDENTITY_FIELDS
                .contains(&key)
                .then(|| format!("{key}={}", value.trim()))
        })
        .collect::<std::collections::BTreeSet<_>>();
    if fields.is_empty() {
        return Err("/proc/cpuinfo reports no CPU identity".into());
    }
    Ok(fields.into_iter().collect::<Vec<_>>().join(";"))
}

pub(super) fn capacity() -> Result<Reported<u64>, String> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").map_err(|error| error.to_string())?;
    let bytes = kib_field(&meminfo, "MemTotal:")
        .ok_or_else(|| "/proc/meminfo MemTotal is absent or invalid".to_owned())?;
    Ok(Reported::new(bytes, ReportedBy::LinuxMemTotal))
}

/// `uname(2)`: the kernel's name, release and version.
pub(super) fn os_build() -> Result<String, String> {
    let mut name = unsafe { std::mem::zeroed::<libc::utsname>() };
    if unsafe { libc::uname(&mut name) } != 0 {
        return Err(format!("uname failed: {}", std::io::Error::last_os_error()));
    }
    // SAFETY: `uname` fills each field with a NUL-terminated string.
    let text = |field: &[libc::c_char]| {
        unsafe { std::ffi::CStr::from_ptr(field.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    };
    Ok(format!(
        "{} {} {}",
        text(&name.sysname),
        text(&name.release),
        text(&name.version)
    ))
}

/// One cache cacheinfo describes for one CPU: its `level`, its `size`
/// (`<n>K`, as the kernel writes it) and the CPUs sharing it
/// (`shared_cpu_list`), each none where the kernel leaves it out.
struct CpuCache {
    level: Option<u32>,
    bytes: Option<u64>,
    shared: Option<String>,
}

/// `/sys/devices/system/cpu/cpu*/cache/index*`: the `Data` or `Unified`
/// caches of the highest `level` over every CPU, each instance (its
/// `shared_cpu_list`) counted once, their sizes summed. The kernel leaves
/// out an attribute it does not know (a type, level or size of zero), and a
/// CPU it describes no caches of has no `cache` directory, so an absent
/// directory or attribute is the OS reporting none: with a level, or a last
/// level's size or sharing, unknown, the last-level caches are unknown. Any
/// other read failure, or a value that does not parse, is an error.
pub(super) fn last_level_cache() -> Result<Option<Reported<u64>>, String> {
    let root = Path::new("/sys/devices/system/cpu");
    let failed = |error: std::io::Error| format!("{}: {error}", root.display());
    let listing = std::fs::read_dir(root).map_err(failed)?;
    let mut caches = Vec::new();
    for cpu in listing {
        let cpu = cpu.map_err(failed)?;
        let name = cpu.file_name();
        let is_cpu = name
            .to_str()
            .and_then(|name| name.strip_prefix("cpu"))
            .is_some_and(|index| !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()));
        if is_cpu {
            cpu_caches(&cpu.path().join("cache"), &mut caches)?;
        }
    }
    let Some(levels) = caches
        .iter()
        .map(|cache| cache.level)
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    let Some(&last) = levels.iter().max() else {
        return Ok(None);
    };
    let mut instances = std::collections::BTreeMap::new();
    for cache in caches.iter().filter(|cache| cache.level == Some(last)) {
        let (Some(shared), Some(bytes)) = (&cache.shared, cache.bytes) else {
            return Ok(None);
        };
        instances.insert(shared.clone(), bytes);
    }
    Ok(Some(Reported::new(
        instances.values().sum(),
        ReportedBy::LinuxSysfsCache,
    )))
}

/// The data and unified caches of one CPU's cacheinfo directory `root`.
fn cpu_caches(root: &Path, caches: &mut Vec<CpuCache>) -> Result<(), String> {
    let listing = match std::fs::read_dir(root) {
        Ok(listing) => listing,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(format!("{}: {error}", root.display())),
    };
    for entry in listing {
        let entry = entry.map_err(|error| format!("{}: {error}", root.display()))?;
        if !entry.file_name().to_string_lossy().starts_with("index") {
            continue;
        }
        let index = entry.path();
        let field = |name: &str| -> Result<Option<String>, String> {
            let path = index.join(name);
            match std::fs::read_to_string(&path) {
                Ok(text) => Ok(Some(text.trim().to_owned())),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(error) => Err(format!("{}: {error}", path.display())),
            }
        };
        let unparsed = |name: &str, value: &str| {
            format!(
                "{}: {value:?} is not a cache {name}",
                index.join(name).display()
            )
        };
        if !matches!(field("type")?.as_deref(), Some("Data" | "Unified")) {
            continue;
        }
        let level = field("level")?
            .map(|level| level.parse::<u32>().map_err(|_| unparsed("level", &level)))
            .transpose()?;
        let bytes = field("size")?
            .map(|size| {
                size.strip_suffix('K')
                    .and_then(|kib| kib.parse::<u64>().ok())
                    .and_then(|kib| kib.checked_mul(1024))
                    .ok_or_else(|| unparsed("size", &size))
            })
            .transpose()?;
        caches.push(CpuCache {
            level,
            bytes,
            shared: field("shared_cpu_list")?,
        });
    }
    Ok(())
}

/// Host RAM now: `/proc/meminfo`'s free, file and anonymous counts above the
/// zones' low watermarks (`/proc/zoneinfo`), within this process's cgroup
/// limits ([`super::binding`]); the machine's refaults (`/proc/vmstat`) and,
/// where this process's own cgroup holds it alone ([`Cgroups::own`]), that
/// cgroup's (`memory.stat`); PSI (`/proc/pressure/memory`, and the limiting
/// cgroup's `memory.pressure`), where the kernel keeps it. The cgroups are
/// found again at every sample, so a process moved to another cgroup is
/// read in the one it is in.
///
/// # Panics
/// Where a figure cannot be read, naming it.
pub(super) fn sample() -> HostSample {
    let at = Instant::now();
    let page = page_size();
    let meminfo = read("/proc/meminfo").unwrap_or_else(|error| panic!("invariant violated: {error}"));
    let field = |name: &str| {
        kib_field(&meminfo, name).unwrap_or_else(|| panic!("invariant violated: /proc/meminfo has no {name}"))
    };
    let cgroups = Cgroups::of_process();
    let limits = cgroups.as_ref().map_or_else(Vec::new, |cgroups| {
        cgroups.limits().unwrap_or_else(|error| panic!("invariant violated: this process's cgroup limits: {error}"))
    });
    let cold = field("Inactive(anon):");
    let host = super::Scope {
        total: field("MemTotal:"),
        supply: Supply {
            free_beyond_target: super::beyond(field("MemFree:"), low_watermarks(page)),
            clean_cache: (field("Active(file):") + field("Inactive(file):"))
                .saturating_sub(field("Dirty:") + field("Writeback:")),
            cold_reclaimable: cold.min(field("SwapFree:")),
        },
    };
    let binding = super::binding(host, &limits.iter().map(|(_, limit)| limit.clone()).collect::<Vec<_>>());
    let pressure = psi_level(Path::new("/proc/pressure/memory")).map(|machine| {
        let limiting = limits.iter().find(|(_, limit)| matches!(limit.kind, ProcessLimitKind::CgroupV2 { .. }));
        // A cgroup whose PSI is off (`cgroup.pressure` 0) reports none.
        limiting.and_then(|(directory, _)| psi_level(&directory.join("memory.pressure"))).map_or(machine, |within| machine.max(within))
    });
    let machine = refaults(&read("/proc/vmstat").unwrap_or_else(|error| panic!("invariant violated: {error}")));
    let own = cgroups.as_ref().and_then(|cgroups| {
        cgroups.own().unwrap_or_else(|error| panic!("invariant violated: this process's own cgroup: {error}"))
    });
    // Read after the machine's, so an event between the two counts as this
    // process's alone: never another program's.
    let harm = own.as_deref().map(|cgroup| Harm { machine, own: cgroup_refaults(cgroup) });
    let own_cold = own.as_deref().map_or(OwnCold::Unobserved, own_cold);
    let own_reclaimed = own_swapped().map(|bytes| OwnReclaimed { bytes, stored: own.as_deref().map_or(0, own_zswapped), exact: false });
    // The counters show every event their folds had passed as the reading
    // began.
    let counted_at = Instant::now();
    let counted = Counted { at: counted_at, lag: counters_lag(own.is_some()) + counted_at.duration_since(at) };
    HostSample {
        at,
        page_bytes: page,
        total: binding.scope.total,
        supply: binding.scope.supply,
        counted,
        cold,
        own_cold,
        own_reclaimed,
        wired: None,
        harm,
        pressure,
        lag: Lag {
            // A free returns its pages in the call, and a cgroup's
            // `memory.current` uncharges them in it.
            release: Duration::ZERO,
            // The host's supply shows a growth in the call; a cgroup's
            // clean cache, read from its `memory.stat`, within memcg's
            // flush, and only where that cgroup's supply binds.
            growth: match binding.supplier {
                None => Duration::ZERO,
                Some(_) => FLUSH_TIME,
            },
        },
    }
}

/// memcg's periodic statistics flush (`FLUSH_TIME`, `mm/memcontrol.c`,
/// Linux 5.15 on): a cgroup's `memory.stat` shows an event within it. A
/// read may flush sooner, past a count of pending updates, never later.
const FLUSH_TIME: Duration = Duration::from_secs(2);

/// How late the counters show an event ([`Counted::lag`]): the per-CPU
/// counts `/proc` sums fold within `vm.stat_interval`, their timer rounded
/// up to the next whole second (`round_jiffies_relative`, `mm/vmstat.c`);
/// a cgroup's, where the counters read one (`cgroup`), within memcg's flush.
fn counters_lag(cgroup: bool) -> Duration {
    let fold = stat_interval() + Duration::from_secs(1);
    match cgroup {
        true => fold.max(FLUSH_TIME),
        false => fold,
    }
}

/// The PSI files a pressure trigger watches: the machine's, and the
/// innermost cgroup v2 limiting this process's ([`crate::pressure`]).
pub(crate) fn pressure_files() -> Vec<PathBuf> {
    let limiting = Cgroups::of_process().and_then(|cgroups| {
        let limits = cgroups.limits().unwrap_or_else(|error| panic!("invariant violated: this process's cgroup limits: {error}"));
        limits.into_iter().find(|(_, limit)| matches!(limit.kind, ProcessLimitKind::CgroupV2 { .. })).map(|(directory, _)| directory)
    });
    std::iter::once(PathBuf::from("/proc/pressure/memory"))
        .chain(limiting.map(|directory| directory.join("memory.pressure")))
        .collect()
}

fn page_size() -> u64 {
    u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
        .ok()
        .filter(|size| *size > 0)
        .expect("invariant violated: sysconf(_SC_PAGESIZE) reads no page size")
}

/// The zones' low watermarks together, in bytes of `page` (`/proc/zoneinfo`):
/// kswapd wakes to reclaim once free pages fall below them, so free pages
/// above them are taken without anyone's reclaim.
fn low_watermarks(page: u64) -> u64 {
    let zoneinfo = read("/proc/zoneinfo").unwrap_or_else(|error| panic!("invariant violated: {error}"));
    zones_low(&zoneinfo) * page
}

fn zones_low(zoneinfo: &str) -> u64 {
    zoneinfo
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            (fields.next() == Some("low")).then(|| {
                fields
                    .next()
                    .and_then(|value| value.parse::<u64>().ok())
                    .unwrap_or_else(|| panic!("invariant violated: /proc/zoneinfo reads {line:?}"))
            })
        })
        .sum()
}

/// The pages refaulted machine-wide (`/proc/vmstat`): anonymous and file
/// since Linux 5.9, the one count before.
fn refaults(vmstat: &str) -> u64 {
    refault_count(vmstat).unwrap_or_else(|| panic!("invariant violated: /proc/vmstat has no workingset refault counts"))
}

/// The pages refaulted in `cgroup`, as its `memory.stat` counts them.
fn cgroup_refaults(cgroup: &Path) -> u64 {
    let path = cgroup.join("memory.stat");
    let stat = read(&path).unwrap_or_else(|error| panic!("invariant violated: {error}"));
    refault_count(&stat).unwrap_or_else(|| panic!("invariant violated: {} has no workingset refault counts", path.display()))
}

/// The refaults a `/proc/vmstat` or `memory.stat` text counts, under the
/// same names in both: anonymous and file since Linux 5.9, the one count
/// before; none where it counts neither.
fn refault_count(text: &str) -> Option<u64> {
    let count = |name: &str| {
        text.lines().find_map(|line| {
            let mut fields = line.split_whitespace();
            (fields.next() == Some(name)).then(|| {
                fields
                    .next()
                    .and_then(|value| value.parse::<u64>().ok())
                    .unwrap_or_else(|| panic!("invariant violated: a refault count reads {line:?}"))
            })
        })
    };
    match (count("workingset_refault_anon"), count("workingset_refault_file"), count("workingset_refault")) {
        (Some(anon), Some(file), _) => Some(anon + file),
        (None, None, all) => all,
        (anon, file, _) => panic!("invariant violated: refault counts read anon {anon:?} and file {file:?}"),
    }
}

/// `vm.stat_interval`: how often the kernel folds the per-CPU counts into
/// the machine's.
fn stat_interval() -> Duration {
    let path = "/proc/sys/vm/stat_interval";
    let value = read(path).unwrap_or_else(|error| panic!("invariant violated: {error}"));
    Duration::from_secs(value.trim().parse().unwrap_or_else(|error| panic!("invariant violated: {path} reads {value:?}: {error}")))
}

/// The pressure level a PSI file `path` reports: critical where some task
/// stalled with every task waiting (`full avg10`), warn where some task
/// stalled (`some avg10`), over the last ten seconds. None where the kernel
/// keeps no PSI (built without it, or booted `psi=0`, where the file is
/// absent or refuses reads with `EOPNOTSUPP`).
///
/// # Panics
/// Where the file cannot be read otherwise, or reads no level.
fn psi_level(path: &Path) -> Option<PressureLevel> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound || error.raw_os_error() == Some(libc::EOPNOTSUPP) => {
            return None;
        }
        Err(error) => panic!("invariant violated: {}: {error}", path.display()),
    };
    Some(parse_psi(&text).unwrap_or_else(|| panic!("invariant violated: {} reads {text:?}", path.display())))
}

fn parse_psi(text: &str) -> Option<PressureLevel> {
    let avg10 = |kind: &str| -> Option<f64> {
        text.lines()
            .find_map(|line| line.strip_prefix(kind))?
            .split_whitespace()
            .find_map(|field| field.strip_prefix("avg10="))?
            .parse()
            .ok()
    };
    Some(match (avg10("some ")?, avg10("full ")?) {
        (_, full) if full > 0.0 => PressureLevel::Critical,
        (some, _) if some > 0.0 => PressureLevel::Warn,
        _ => PressureLevel::Normal,
    })
}

/// This process's anonymous bytes in swap, zswap's included
/// (`/proc/self/status`'s `VmSwap`); none where the kernel reports no
/// `VmSwap` (built without `CONFIG_SWAP`, or a sandbox's own `/proc`): the
/// process's bytes out of RAM are then unreported, never a failed reading.
fn own_swapped() -> Option<u64> {
    let status = read("/proc/self/status").unwrap_or_else(|error| panic!("invariant violated: {error}"));
    kib_field(&status, "VmSwap:")
}

/// The RAM zswap holds of this process's own swapped pages, at their
/// compressed size: the `zswap` of its own cgroup ([`Cgroups::own`]); none
/// where the kernel counts no zswap (before 5.19, or built without it).
fn own_zswapped(cgroup: &Path) -> u64 {
    let path = cgroup.join("memory.stat");
    let stat = read(&path).unwrap_or_else(|error| panic!("invariant violated: {error}"));
    stat_field(&path, &stat, "zswap").unwrap_or_else(|error| panic!("invariant violated: {error}")).unwrap_or(0)
}

/// This process's cold anonymous bytes: the `inactive_anon` of its own
/// cgroup ([`Cgroups::own`]).
fn own_cold(cgroup: &Path) -> OwnCold {
    let path = cgroup.join("memory.stat");
    let stat = read(&path).unwrap_or_else(|error| panic!("invariant violated: {error}"));
    OwnCold::Bytes(
        stat_field(&path, &stat, "inactive_anon")
            .unwrap_or_else(|error| panic!("invariant violated: {error}"))
            .unwrap_or_else(|| panic!("invariant violated: {} has no inactive_anon", path.display())),
    )
}

/// The value of `key` in the `memory.stat` text `stat` of `path`.
fn stat_field(path: &Path, stat: &str, key: &str) -> Result<Option<u64>, ObservationError> {
    stat.lines()
        .find_map(|line| {
            let mut fields = line.split_whitespace();
            (fields.next() == Some(key)).then(|| parse_bytes(path, fields.next().unwrap_or("")))
        })
        .transpose()
}

fn read(path: impl AsRef<Path>) -> Result<String, ObservationError> {
    let path = path.as_ref();
    std::fs::read_to_string(path)
        .map_err(|error| ObservationError::Failed(format!("{}: {error}", path.display())))
}

fn kib_field(text: &str, name: &str) -> Option<u64> {
    text.lines()
        .find_map(|line| line.strip_prefix(name))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .and_then(|kib| kib.checked_mul(1024))
}

/// The cgroup hierarchy this process's memory controller is bound to.
///
/// A controller is attached to exactly one hierarchy. On a legacy or hybrid
/// host the v1 `memory` hierarchy carries every memory limit and the unified
/// hierarchy carries none, so the v1 hierarchy is chosen whenever
/// `/proc/self/cgroup` lists it.
enum MemoryHierarchy<'a> {
    V1 {
        path: &'a str,
    },
    V2 {
        path: &'a str,
    },
    /// No memory controller is enabled: no cgroup can limit memory.
    None,
}

fn memory_hierarchy(membership: &str) -> MemoryHierarchy<'_> {
    let mut unified = None;
    for line in membership.lines() {
        let mut fields = line.splitn(3, ':');
        let (Some(_), Some(controllers), Some(path)) =
            (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        if controllers
            .split(',')
            .any(|controller| controller == "memory")
        {
            return MemoryHierarchy::V1 { path };
        }
        if controllers.is_empty() {
            unified = Some(path);
        }
    }
    unified.map_or(MemoryHierarchy::None, |path| MemoryHierarchy::V2 { path })
}

/// This process's cgroup and each visible ancestor, in the hierarchy its
/// memory controller is bound to, as found at one sample: the scope that
/// sample binds ([`super::binding`]).
struct Cgroups {
    hierarchy: Hierarchy,
    /// From this process's cgroup up to the mount point: each directory and
    /// its path relative to the mount.
    chain: Vec<(PathBuf, String)>,
}

enum Hierarchy {
    /// The v1 `memory` hierarchy, its "no limit" figure with it
    /// ([`v1_unlimited_bytes`]).
    V1 { unlimited: u64 },
    V2,
}

impl Cgroups {
    /// This process's now, read at each call so that a process moved to
    /// another cgroup is read in the one it is in: none where no memory
    /// controller is enabled.
    ///
    /// # Panics
    /// Where they cannot be read.
    fn of_process() -> Option<Self> {
        Self::find(Path::new("/")).unwrap_or_else(|error| panic!("invariant violated: this process's cgroups: {error}"))
    }

    /// This process's cgroups under the filesystem root `root` (`/`
    /// outside tests). Ancestors a cgroup namespace or a container's mount
    /// hides cannot be read, and bound nothing here.
    fn find(root: &Path) -> Result<Option<Self>, ObservationError> {
        let membership = read(root.join("proc/self/cgroup"))?;
        let (hierarchy, mount) = match memory_hierarchy(&membership) {
            MemoryHierarchy::V1 { path } => {
                let mount = cgroup_mount(root, path, |filesystem, options| {
                    filesystem == "cgroup" && options.split(',').any(|option| option == "memory")
                })?
                .ok_or_else(|| ObservationError::Unsupported("this process's cgroup v1 memory hierarchy is not mounted".into()))?;
                (Hierarchy::V1 { unlimited: v1_unlimited_bytes()? }, mount)
            }
            MemoryHierarchy::V2 { path } => {
                let mount = cgroup_mount(root, path, |filesystem, _| filesystem == "cgroup2")?
                    .ok_or_else(|| ObservationError::Unsupported("this process's cgroup2 hierarchy is not mounted".into()))?;
                (Hierarchy::V2, mount)
            }
            MemoryHierarchy::None => return Ok(None),
        };
        let mut chain = Vec::new();
        walk_cgroups(&mount, |directory, cgroup| {
            chain.push((directory.to_path_buf(), cgroup));
            Ok(())
        })?;
        Ok(Some(Self { hierarchy, chain }))
    }

    /// Every limit of the chain now, innermost first, each with its
    /// directory.
    fn limits(&self) -> Result<Vec<(PathBuf, ProcessMemoryLimit)>, ObservationError> {
        let mut limits = Vec::new();
        for (directory, cgroup) in &self.chain {
            let limit = match self.hierarchy {
                Hierarchy::V2 => v2_limit(directory)?.map(|limit_bytes| -> Result<_, ObservationError> {
                    Ok(ProcessMemoryLimit {
                        kind: ProcessLimitKind::CgroupV2 { cgroup: cgroup.clone() },
                        limit_bytes,
                        used_bytes: read_bytes(&directory.join("memory.current"))?,
                        reclaimable_bytes: file_cache_headroom(
                            directory,
                            ["active_file", "inactive_file", "file_dirty", "file_writeback"],
                        )?,
                    })
                }),
                Hierarchy::V1 { unlimited } => {
                    let limit_bytes = read_bytes(&directory.join("memory.limit_in_bytes"))?;
                    (limit_bytes < unlimited).then(|| -> Result<_, ObservationError> {
                        Ok(ProcessMemoryLimit {
                            kind: ProcessLimitKind::CgroupV1 { cgroup: cgroup.clone() },
                            limit_bytes,
                            used_bytes: read_bytes(&directory.join("memory.usage_in_bytes"))?,
                            reclaimable_bytes: file_cache_headroom(
                                directory,
                                ["total_active_file", "total_inactive_file", "total_dirty", "total_writeback"],
                            )?,
                        })
                    })
                }
            };
            if let Some(limit) = limit {
                limits.push((directory.clone(), limit?));
            }
        }
        Ok(limits)
    }

    /// The cgroup whose counts are this process's own, the one owner of
    /// that rule for every count read from it (harm's refaults, the cold
    /// pages): this process's own cgroup v2, where it has a memory
    /// controller (its `memory.stat`), lists this process alone
    /// (`cgroup.procs`) and has no descendant (`cgroup.stat`'s
    /// `nr_descendants`), whose processes its hierarchical counts would
    /// include. None otherwise: a cgroup shared with other processes (a
    /// `session-N.scope`, `user.slice`, a container's root) counts their
    /// events too, and an ancestor counts its other children's, so no count
    /// of it is this process's. None under cgroup v1, whose counts this
    /// reader does not take.
    fn own(&self) -> Result<Option<PathBuf>, ObservationError> {
        let Hierarchy::V2 = self.hierarchy else {
            return Ok(None);
        };
        let (directory, _) = self.chain.first().expect("a cgroup chain starts at this process's cgroup");
        if !present(&directory.join("memory.stat"))? {
            return Ok(None);
        }
        let pid = std::process::id().to_string();
        if !read(directory.join("cgroup.procs"))?.lines().eq([pid.as_str()]) {
            return Ok(None);
        }
        let path = directory.join("cgroup.stat");
        let stat = read(&path)?;
        let descendants = stat_field(&path, &stat, "nr_descendants")?
            .ok_or_else(|| ObservationError::Failed(format!("{} has no nr_descendants", path.display())))?;
        Ok((descendants == 0).then(|| directory.clone()))
    }
}

/// Whether the file `path` exists: an error other than its absence is no
/// answer to that.
fn present(path: &Path) -> Result<bool, ObservationError> {
    path.try_exists().map_err(|error| ObservationError::Failed(format!("{}: {error}", path.display())))
}

/// A cgroup v2's memory limit: the lower of `memory.max` and `memory.high`
/// (past which the kernel reclaims the cgroup and throttles it), none where
/// both are `max` or the cgroup has no memory controller.
fn v2_limit(directory: &Path) -> Result<Option<u64>, ObservationError> {
    let mut least = None;
    for name in ["memory.max", "memory.high"] {
        let file = directory.join(name);
        if !present(&file)? {
            continue;
        }
        let value = read(&file)?;
        let value = value.trim();
        if value != "max" {
            let bytes = parse_bytes(&file, value)?;
            least = Some(least.map_or(bytes, |least: u64| least.min(bytes)));
        }
    }
    Ok(least)
}

/// Cgroup usage includes reclaimable page cache, unlike Linux MemAvailable.
/// Credit filesystem pages on either LRU list, conservatively excluding all
/// dirty and writeback pages: a charge at the limit reclaims active cache too
/// (a v1 docker cgroup on Linux gave up all of its 58.7 GB at an unchanged
/// allocation rate, with no failed charge or OOM kill). Anonymous memory,
/// tmpfs/shmem (on the anon lists) and kernel allocations receive no credit.
/// V1 counters must be hierarchical, like memory.usage_in_bytes, so ancestor
/// limits include their children's cache. Missing counters cannot establish
/// reclaimability.
fn file_cache_headroom(directory: &Path, keys: [&str; 4]) -> Result<u64, ObservationError> {
    let path = directory.join("memory.stat");
    let stat = read(&path)?;
    let mut values = [None; 4];
    for line in stat.lines() {
        let mut fields = line.split_whitespace();
        let Some(key) = fields.next() else { continue };
        if let Some(index) = keys.iter().position(|candidate| *candidate == key) {
            let value = fields.next().unwrap_or("");
            values[index] = Some(parse_bytes(&path, value)?);
        }
    }
    Ok(match values {
        [Some(active), Some(inactive), Some(dirty), Some(writeback)] => active
            .saturating_add(inactive)
            .saturating_sub(dirty.saturating_add(writeback)),
        _ => 0,
    })
}

/// The v1 memory controller reports "no limit" as its page counter maximum,
/// `LONG_MAX` rounded down to the page size.
fn v1_unlimited_bytes() -> Result<u64, ObservationError> {
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page = u64::try_from(page)
        .ok()
        .filter(|page| *page > 0)
        .ok_or_else(|| ObservationError::Failed("sysconf(_SC_PAGESIZE) failed".into()))?;
    Ok(i64::MAX as u64 / page * page)
}

/// A mounted cgroup hierarchy and this process's cgroup directory within it.
struct CgroupMount {
    /// Mount point, under the filesystem root.
    point: PathBuf,
    /// This process's cgroup directory.
    cgroup: PathBuf,
}

/// The first mount matching `accepts(filesystem type, super options)` whose
/// root contains `path`.
fn cgroup_mount(
    root: &Path,
    path: &str,
    accepts: impl Fn(&str, &str) -> bool,
) -> Result<Option<CgroupMount>, ObservationError> {
    let mountinfo = read(root.join("proc/self/mountinfo"))?;
    let mut outside = None;
    for line in mountinfo.lines() {
        let Some((mount, filesystem)) = line.split_once(" - ") else {
            continue;
        };
        let mut filesystem = filesystem.split_whitespace();
        let (Some(kind), Some(_source), Some(options)) =
            (filesystem.next(), filesystem.next(), filesystem.next())
        else {
            continue;
        };
        if !accepts(kind, options) {
            continue;
        }
        let mut fields = mount.split_whitespace();
        let (Some(mounted), Some(point)) = (fields.nth(3), fields.next()) else {
            continue;
        };
        let mounted = unescape(mounted);
        let point = root.join(unescape(point).trim_start_matches('/'));
        let Some(relative) = within(path, &mounted) else {
            outside = Some(mounted);
            continue;
        };
        let cgroup = point.join(relative);
        return Ok(Some(CgroupMount { point, cgroup }));
    }
    match outside {
        Some(mounted) => Err(ObservationError::Unsupported(format!(
            "cgroup {path} is outside the visible mount root {mounted}"
        ))),
        None => Ok(None),
    }
}

/// `path` relative to the mounted hierarchy path `mounted`, if inside it.
fn within<'a>(path: &'a str, mounted: &str) -> Option<&'a str> {
    let rest = path.strip_prefix(mounted.trim_end_matches('/'))?;
    (rest.is_empty() || rest.starts_with('/')).then(|| rest.trim_start_matches('/'))
}

/// Visits this process's cgroup and each visible ancestor up to the mount
/// point, with the cgroup's path relative to the mount.
fn walk_cgroups(
    mount: &CgroupMount,
    mut visit: impl FnMut(&Path, String) -> Result<(), ObservationError>,
) -> Result<(), ObservationError> {
    let mut directory = mount.cgroup.clone();
    loop {
        let cgroup = directory
            .strip_prefix(&mount.point)
            .expect("cgroup walk stays inside its mount")
            .display()
            .to_string();
        visit(&directory, format!("/{cgroup}"))?;
        if directory == mount.point {
            return Ok(());
        }
        directory = directory
            .parent()
            .expect("cgroup walk stays inside its mount")
            .to_path_buf();
    }
}

fn read_bytes(path: &Path) -> Result<u64, ObservationError> {
    parse_bytes(path, read(path)?.trim())
}

fn parse_bytes(path: &Path, value: &str) -> Result<u64, ObservationError> {
    value
        .parse::<u64>()
        .map_err(|_| ObservationError::Failed(format!("{} contains {value}", path.display())))
}

/// `crate::processors::sysfs_nodes` over `/sys/devices`, with the CPUs
/// `sched_getaffinity` lets the process run on.
pub(super) fn processors() -> Result<crate::Processors, String> {
    let nodes = crate::processors::sysfs_nodes(Path::new("/sys/devices"))?;
    // SAFETY: a zeroed set the kernel fills for the calling thread.
    let allowed = unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return Err(format!(
                "sched_getaffinity: {}",
                std::io::Error::last_os_error()
            ));
        }
        (0..libc::CPU_SETSIZE as usize)
            .filter(|cpu| libc::CPU_ISSET(*cpu, &set))
            .collect()
    };
    crate::Processors::new(nodes, allowed, crate::ProcessorsReportedBy::LinuxSysfs)
}

/// `cpus` as a CPU set.
fn cpu_set(cpus: &[usize]) -> Result<libc::cpu_set_t, String> {
    if let Some(cpu) = cpus.iter().find(|&&cpu| cpu >= libc::CPU_SETSIZE as usize) {
        return Err(format!("CPU {cpu} is beyond a CPU set"));
    }
    // SAFETY: a zeroed set holding `cpus`.
    Ok(unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        cpus.iter().for_each(|&cpu| libc::CPU_SET(cpu, &mut set));
        set
    })
}

/// `sched_setaffinity` of thread `tid` of the process to `set`; a thread
/// that exited needs none.
fn place_thread(tid: libc::pid_t, set: &libc::cpu_set_t, cpus: &[usize]) -> Result<(), String> {
    // SAFETY: a CPU set, applied to thread `tid` of this process.
    if unsafe { libc::sched_setaffinity(tid, std::mem::size_of::<libc::cpu_set_t>(), set) } != 0 {
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::ESRCH) {
            return Err(format!(
                "sched_setaffinity of thread {tid} to CPUs {cpus:?}: {error}"
            ));
        }
    }
    Ok(())
}

/// The threads [`pin`] placed, by id, while they run: [`place_process`]
/// places none of them.
static PINNED: std::sync::Mutex<std::collections::BTreeSet<libc::pid_t>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());

/// The calling thread's entry in [`PINNED`], taken out as it exits.
struct Pinned(libc::pid_t);

impl Drop for Pinned {
    fn drop(&mut self) {
        PINNED.lock().expect("the pinned threads").remove(&self.0);
    }
}

thread_local! {
    static PIN: std::cell::OnceCell<Pinned> = const { std::cell::OnceCell::new() };
}

/// `sched_setaffinity` of every thread of the process (`/proc/self/task`)
/// but those [`pin`] placed to `cpus`, pass after pass until one finds no
/// thread it had not placed: a thread started during a pass by one not yet
/// placed is placed by the next.
pub(super) fn place_process(cpus: &[usize]) -> Result<(), String> {
    let set = cpu_set(cpus)?;
    let mut placed = PINNED.lock().expect("the pinned threads").clone();
    loop {
        let tasks = std::fs::read_dir("/proc/self/task")
            .map_err(|error| format!("/proc/self/task: {error}"))?;
        let mut new = false;
        for task in tasks {
            let name = task
                .map_err(|error| format!("/proc/self/task: {error}"))?
                .file_name();
            let tid = name
                .to_str()
                .and_then(|name| name.parse::<libc::pid_t>().ok())
                .ok_or_else(|| format!("/proc/self/task/{name:?} is no thread"))?;
            if placed.insert(tid) {
                new = true;
                place_thread(tid, &set, cpus)?;
            }
        }
        if !new {
            return Ok(());
        }
    }
}

/// `sched_setaffinity` of the calling thread to `cpus`, which [`place_process`]
/// then leaves where it is.
pub(super) fn pin(cpus: &[usize]) -> Result<(), String> {
    let set = cpu_set(cpus)?;
    // SAFETY: a CPU set, applied to the calling thread.
    let placed =
        unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) };
    match placed {
        0 => {
            // SAFETY: the calling thread's id.
            let tid = unsafe { libc::syscall(libc::SYS_gettid) } as libc::pid_t;
            PIN.with(|pin| {
                pin.get_or_init(|| {
                    PINNED.lock().expect("the pinned threads").insert(tid);
                    Pinned(tid)
                });
            });
            Ok(())
        }
        _ => Err(format!(
            "sched_setaffinity to CPUs {cpus:?}: {}",
            std::io::Error::last_os_error()
        )),
    }
}

/// mountinfo escapes space, tab, newline and backslash as octal.
fn unescape(field: &str) -> String {
    field
        .replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fixture filesystem root holding `/proc/self/{cgroup,mountinfo}` and
    /// cgroup files, removed on drop.
    struct Fixture(PathBuf);

    impl Fixture {
        fn new(name: &str, cgroup: &str, mountinfo: &str) -> Self {
            let root =
                std::env::temp_dir().join(format!("seismic-cgroup-{}-{name}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            let fixture = Self(root);
            fixture.write("proc/self/cgroup", cgroup);
            fixture.write("proc/self/mountinfo", mountinfo);
            fixture
        }

        fn write(&self, path: &str, contents: &str) {
            let path = self.0.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, contents).unwrap();
        }

        fn limits(&self) -> Result<Vec<ProcessMemoryLimit>, ObservationError> {
            Ok(match self.cgroups()? {
                Some(cgroups) => cgroups.limits()?.into_iter().map(|(_, limit)| limit).collect(),
                None => Vec::new(),
            })
        }

        fn cgroups(&self) -> Result<Option<Cgroups>, ObservationError> {
            Cgroups::find(&self.0)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn limit(kind: ProcessLimitKind, limit_bytes: u64, used_bytes: u64) -> ProcessMemoryLimit {
        ProcessMemoryLimit {
            kind,
            limit_bytes,
            used_bytes,
            reclaimable_bytes: 0,
        }
    }

    const V2_HOST_MOUNT: &str = "30 25 0:26 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime shared:4 - cgroup2 cgroup2 rw,nsdelegate\n";
    const V1_HOST_MOUNTS: &str = "\
24 21 0:22 / /sys/fs/cgroup ro,nosuid,nodev,noexec shared:9 - tmpfs tmpfs ro,mode=755
33 24 0:29 / /sys/fs/cgroup/memory rw,nosuid,nodev,noexec,relatime shared:15 - cgroup cgroup rw,memory
34 24 0:30 / /sys/fs/cgroup/cpu,cpuacct rw,nosuid,nodev,noexec,relatime shared:16 - cgroup cgroup rw,cpu,cpuacct
";

    #[test]
    fn v2_host_reports_every_limited_cgroup_up_to_the_root() {
        let fixture = Fixture::new("v2-host", "0::/user.slice/app.scope\n", V2_HOST_MOUNT);
        fixture.write("sys/fs/cgroup/user.slice/app.scope/memory.max", "1000\n");
        fixture.write("sys/fs/cgroup/user.slice/app.scope/memory.current", "400\n");
        fixture.write("sys/fs/cgroup/user.slice/app.scope/memory.stat", "");
        fixture.write("sys/fs/cgroup/user.slice/memory.max", "max\n");
        fixture.write("sys/fs/cgroup/user.slice/memory.current", "900\n");
        fixture.write("sys/fs/cgroup/user.slice/memory.stat", "");
        let limits = fixture.limits().unwrap();
        assert_eq!(
            limits,
            [limit(
                ProcessLimitKind::CgroupV2 {
                    cgroup: "/user.slice/app.scope".into()
                },
                1000,
                400
            )]
        );
    }

    #[test]
    fn a_v2_namespace_root_gives_its_own_limit() {
        let fixture = Fixture::new("v2-namespace", "0::/\n", V2_HOST_MOUNT);
        fixture.write("sys/fs/cgroup/memory.max", "2000\n");
        fixture.write("sys/fs/cgroup/memory.current", "500\n");
        fixture.write("sys/fs/cgroup/memory.stat", "");
        let limits = fixture.limits().unwrap();
        assert_eq!(
            limits,
            [limit(
                ProcessLimitKind::CgroupV2 { cgroup: "/".into() },
                2000,
                500
            )]
        );
    }

    #[test]
    fn v1_host_reads_the_memory_hierarchy_and_skips_unlimited_cgroups() {
        let unlimited = v1_unlimited_bytes().unwrap().to_string();
        let fixture = Fixture::new(
            "v1-host",
            "4:memory:/system.slice/svc.service\n3:cpu,cpuacct:/system.slice/svc.service\n1:name=systemd:/system.slice/svc.service\n",
            V1_HOST_MOUNTS,
        );
        let memory = "sys/fs/cgroup/memory";
        fixture.write(
            &format!("{memory}/system.slice/svc.service/memory.stat"),
            "",
        );
        fixture.write(
            &format!("{memory}/system.slice/svc.service/memory.limit_in_bytes"),
            "3000\n",
        );
        fixture.write(
            &format!("{memory}/system.slice/svc.service/memory.usage_in_bytes"),
            "1000\n",
        );
        fixture.write(
            &format!("{memory}/system.slice/memory.limit_in_bytes"),
            &unlimited,
        );
        fixture.write(
            &format!("{memory}/system.slice/memory.usage_in_bytes"),
            "5000\n",
        );
        fixture.write(&format!("{memory}/memory.limit_in_bytes"), &unlimited);
        fixture.write(&format!("{memory}/memory.usage_in_bytes"), "9000\n");
        let limits = fixture.limits().unwrap();
        assert_eq!(
            limits,
            [limit(
                ProcessLimitKind::CgroupV1 {
                    cgroup: "/system.slice/svc.service".into()
                },
                3000,
                1000
            )]
        );
    }

    #[test]
    fn hybrid_host_takes_memory_limits_from_the_v1_hierarchy() {
        let unlimited = v1_unlimited_bytes().unwrap().to_string();
        let mounts = format!(
            "{V1_HOST_MOUNTS}35 24 0:31 / /sys/fs/cgroup/unified rw,nosuid,nodev,noexec shared:10 - cgroup2 cgroup2 rw,nsdelegate\n"
        );
        let fixture = Fixture::new(
            "hybrid-host",
            "4:memory:/user.slice\n0::/user.slice/app.scope\n",
            &mounts,
        );
        fixture.write(
            "sys/fs/cgroup/unified/user.slice/app.scope/cgroup.procs",
            "1\n",
        );
        fixture.write("sys/fs/cgroup/memory/user.slice/memory.stat", "");
        fixture.write(
            "sys/fs/cgroup/memory/user.slice/memory.limit_in_bytes",
            "6000\n",
        );
        fixture.write(
            "sys/fs/cgroup/memory/user.slice/memory.usage_in_bytes",
            "2000\n",
        );
        fixture.write("sys/fs/cgroup/memory/memory.limit_in_bytes", &unlimited);
        fixture.write("sys/fs/cgroup/memory/memory.usage_in_bytes", "9000\n");
        fixture.write("sys/fs/cgroup/memory/memory.stat", "");
        let limits = fixture.limits().unwrap();
        assert_eq!(
            limits,
            [limit(
                ProcessLimitKind::CgroupV1 {
                    cgroup: "/user.slice".into()
                },
                6000,
                2000
            )]
        );
    }

    #[test]
    fn v2_limits_take_the_lower_of_max_and_high() {
        // Past `memory.high` the kernel reclaims the cgroup and throttles
        // it: the cgroup's limit is the lower of it and `memory.max`, and
        // the innermost limited cgroup is the one whose pressure is read.
        let fixture = Fixture::new("v2-high", "0::/user.slice/app.scope\n", V2_HOST_MOUNT);
        fixture.write("sys/fs/cgroup/user.slice/app.scope/memory.max", "1000\n");
        fixture.write("sys/fs/cgroup/user.slice/app.scope/memory.high", "800\n");
        fixture.write("sys/fs/cgroup/user.slice/app.scope/memory.current", "790\n");
        fixture.write("sys/fs/cgroup/user.slice/app.scope/memory.stat", "");
        fixture.write("sys/fs/cgroup/user.slice/memory.max", "max\n");
        fixture.write("sys/fs/cgroup/user.slice/memory.high", "max\n");
        fixture.write("sys/fs/cgroup/user.slice/memory.current", "750\n");
        fixture.write("sys/fs/cgroup/user.slice/memory.stat", "");
        let limits = fixture.limits().unwrap();
        assert_eq!(
            limits,
            [limit(
                ProcessLimitKind::CgroupV2 {
                    cgroup: "/user.slice/app.scope".into()
                },
                800,
                790
            )]
        );
        assert_eq!(limits[0].scope(u64::MAX).supply.balance(), 10);
        let cgroups = fixture.cgroups().unwrap().unwrap();
        let own = fixture.0.join("sys/fs/cgroup/user.slice/app.scope");
        assert_eq!(cgroups.limits().unwrap()[0].0, own);
    }

    /// The machine's figures parse as the kernel writes them.
    #[test]
    fn proc_figures_parse() {
        assert_eq!(zones_low("Node 0, zone   Normal\n  pages free     3\n        min      10\n        low      12\n        high     14\nNode 0, zone DMA32\n        low      5\n"), 17);
        assert_eq!(refaults("workingset_refault_anon 3\nworkingset_refault_file 4\n"), 7);
        assert_eq!(refaults("workingset_refault 9\n"), 9);
        assert_eq!(refault_count("anon 900\nfile 30\n"), None);
        let psi = |some: &str, full: &str| parse_psi(&format!("some avg10={some} avg60=0.00 avg300=0.00 total=1\nfull avg10={full} avg60=0.00 avg300=0.00 total=0\n"));
        assert_eq!(psi("0.00", "0.00"), Some(PressureLevel::Normal));
        assert_eq!(psi("0.12", "0.00"), Some(PressureLevel::Warn));
        assert_eq!(psi("0.12", "0.01"), Some(PressureLevel::Critical));
    }

    /// A cgroup v2 of this process alone, with a memory controller and no
    /// descendant, is its own: its refaults and cold pages are this
    /// process's, in the machine's unit.
    #[test]
    fn a_cgroup_of_this_process_alone_counts_its_own_paging_and_cold_pages() {
        let fixture = Fixture::new("own", "0::/app.scope\n", V2_HOST_MOUNT);
        fixture.write("sys/fs/cgroup/app.scope/memory.stat", "inactive_anon 300\nzswap 70\nworkingset_refault_anon 5\nworkingset_refault_file 6\n");
        fixture.write("sys/fs/cgroup/app.scope/cgroup.procs", &format!("{}\n", std::process::id()));
        fixture.write("sys/fs/cgroup/app.scope/cgroup.stat", "nr_descendants 0\nnr_dying_descendants 0\n");
        let own = fixture.cgroups().unwrap().unwrap().own().unwrap().expect("a cgroup of this process alone is its own");
        assert_eq!(own, fixture.0.join("sys/fs/cgroup/app.scope"));
        assert_eq!(cgroup_refaults(&own), 11);
        assert_eq!(own_cold(&own), OwnCold::Bytes(300));
        assert_eq!(own_zswapped(&own), 70);
    }

    /// A cgroup this process shares, by another member or a descendant
    /// whose counts its own include, an ancestor holding its memory
    /// controller, and a cgroup v1 hierarchy count no events as this
    /// process's: harm and own cold pages are then unobserved, never
    /// another program's read as ours.
    #[test]
    fn a_shared_cgroup_counts_nothing_as_this_processs() {
        let pid = std::process::id();
        let own = |name: &str, procs: String, descendants: u32| {
            let fixture = Fixture::new(name, "0::/session-2.scope\n", V2_HOST_MOUNT);
            fixture.write("sys/fs/cgroup/session-2.scope/memory.stat", "workingset_refault_anon 5\nworkingset_refault_file 6\n");
            fixture.write("sys/fs/cgroup/session-2.scope/cgroup.procs", &procs);
            fixture.write("sys/fs/cgroup/session-2.scope/cgroup.stat", &format!("nr_descendants {descendants}\n"));
            fixture.cgroups().unwrap().unwrap().own().unwrap()
        };
        assert!(own("shared-sole", format!("{pid}\n"), 0).is_some());
        assert_eq!(own("shared-member", format!("{pid}\n1\n"), 0), None, "another process in the cgroup");
        assert_eq!(own("shared-descendant", format!("{pid}\n"), 1), None, "a descendant's counts in the cgroup's");
        let ancestor = Fixture::new("shared-ancestor", "0::/user.slice/app.scope\n", V2_HOST_MOUNT);
        ancestor.write("sys/fs/cgroup/user.slice/memory.stat", "workingset_refault_anon 5\nworkingset_refault_file 6\n");
        ancestor.write("sys/fs/cgroup/user.slice/app.scope/cgroup.procs", &format!("{pid}\n"));
        assert_eq!(ancestor.cgroups().unwrap().unwrap().own().unwrap(), None, "the memory controller is an ancestor's");
        let v1 = Fixture::new("shared-v1", "4:memory:/svc\n", V1_HOST_MOUNTS);
        v1.write("sys/fs/cgroup/memory/svc/memory.stat", "total_rss 1\n");
        assert_eq!(v1.cgroups().unwrap().unwrap().own().unwrap(), None);
    }

    /// The level reads as the kernel writes it, and a kernel keeping no PSI
    /// reports none.
    #[test]
    fn a_kernel_without_psi_reports_no_level() {
        let fixture = Fixture::new("psi", "0::/\n", V2_HOST_MOUNT);
        fixture.write("pressure", "some avg10=0.50 avg60=0.00 avg300=0.00 total=9\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n");
        assert_eq!(psi_level(&fixture.0.join("pressure")), Some(PressureLevel::Warn));
        assert_eq!(psi_level(&fixture.0.join("absent")), None);
    }

    #[test]
    fn hybrid_container_reads_its_v1_cgroup_and_hides_ancestors() {
        // A container on a hybrid host: the unified hierarchy is listed but
        // not mounted, and the v1 memory mount is rooted at the container.
        let fixture = Fixture::new(
            "hybrid-container",
            "13:misc:/docker/abc\n2:memory:/docker/abc\n1:name=systemd:/docker/abc\n0::/docker/abc\n",
            "1294 1287 0:78 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime - tmpfs tmpfs rw,mode=755\n\
             1296 1294 0:34 /docker/abc /sys/fs/cgroup/memory ro,nosuid,nodev,noexec,relatime master:16 - cgroup cgroup rw,memory\n",
        );
        fixture.write("sys/fs/cgroup/memory/memory.limit_in_bytes", "4000\n");
        fixture.write("sys/fs/cgroup/memory/memory.usage_in_bytes", "1500\n");
        fixture.write("sys/fs/cgroup/memory/memory.stat", "");
        let limits = fixture.limits().unwrap();
        assert_eq!(
            limits,
            [limit(
                ProcessLimitKind::CgroupV1 { cgroup: "/".into() },
                4000,
                1500
            )]
        );
    }

    #[test]
    fn v2_headroom_credits_only_clean_file_cache() {
        let fixture = Fixture::new("v2-cache", "0::/worker\n", V2_HOST_MOUNT);
        fixture.write("sys/fs/cgroup/worker/memory.max", "1000\n");
        fixture.write("sys/fs/cgroup/worker/memory.current", "900\n");
        for (stat, reclaimable, remaining) in [
            (
                "active_file 200\ninactive_file 400\nfile_dirty 100\nfile_writeback 50\nanon 300\n",
                450,
                550,
            ),
            (
                "active_file 0\ninactive_file 0\nfile_dirty 0\nfile_writeback 0\nanon 900\n",
                0,
                100,
            ),
            // Dirty pages may lie on either list.
            (
                "active_file 300\ninactive_file 300\nfile_dirty 700\nfile_writeback 0\n",
                0,
                100,
            ),
            (
                "active_file 0\ninactive_file 600\nfile_writeback 0\n",
                0,
                100,
            ),
            (
                "inactive_file 600\nfile_dirty 0\nfile_writeback 0\n",
                0,
                100,
            ),
            // Non-atomic counters can briefly disagree; never grant more
            // than the limit even when the cache counter exceeds usage.
            (
                "active_file 600\ninactive_file 600\nfile_dirty 0\nfile_writeback 0\n",
                1200,
                1000,
            ),
        ] {
            fixture.write("sys/fs/cgroup/worker/memory.stat", stat);
            let limits = fixture.limits().unwrap();
            assert_eq!(limits[0].used_bytes, 900);
            assert_eq!(limits[0].reclaimable_bytes, reclaimable);
            assert_eq!(limits[0].scope(u64::MAX).supply.balance(), remaining);
        }
    }

    #[test]
    fn v1_ancestor_cache_uses_hierarchical_counters() {
        let fixture = Fixture::new("v1-cache", "4:memory:/parent/worker\n", V1_HOST_MOUNTS);
        fixture.write(
            "sys/fs/cgroup/memory/parent/worker/memory.limit_in_bytes",
            &v1_unlimited_bytes().unwrap().to_string(),
        );
        fixture.write(
            "sys/fs/cgroup/memory/parent/memory.limit_in_bytes",
            "1000\n",
        );
        fixture.write("sys/fs/cgroup/memory/parent/memory.usage_in_bytes", "900\n");
        fixture.write("sys/fs/cgroup/memory/parent/memory.stat",
            "active_file 0\ninactive_file 0\ndirty 0\nwriteback 0\ntotal_active_file 200\ntotal_inactive_file 400\ntotal_dirty 100\ntotal_writeback 50\n");
        fixture.write(
            "sys/fs/cgroup/memory/memory.limit_in_bytes",
            &v1_unlimited_bytes().unwrap().to_string(),
        );
        let limits = fixture.limits().unwrap();
        assert_eq!(limits[0].reclaimable_bytes, 450);
        assert_eq!(limits[0].scope(u64::MAX).supply.balance(), 550);
    }

    #[test]
    fn no_memory_controller_admits_no_cgroup_limit() {
        let fixture = Fixture::new(
            "no-memory",
            "3:cpu,cpuacct:/\n1:name=systemd:/\n",
            V1_HOST_MOUNTS,
        );
        let limits = fixture.limits().unwrap();
        assert!(limits.is_empty());
    }

    #[test]
    fn a_cgroup_outside_every_visible_mount_is_unsupported() {
        let fixture = Fixture::new(
            "outside",
            "2:memory:/docker/other\n",
            "1296 1294 0:34 /docker/abc /sys/fs/cgroup/memory ro - cgroup cgroup rw,memory\n",
        );
        assert!(matches!(
            fixture.limits(),
            Err(ObservationError::Unsupported(_))
        ));
    }

    #[test]
    fn an_unmounted_hierarchy_is_unsupported_not_unlimited() {
        let fixture = Fixture::new("unmounted", "2:memory:/docker/abc\n", "");
        assert!(matches!(
            fixture.limits(),
            Err(ObservationError::Unsupported(_))
        ));
    }
}

/// The PCI function's `numa_node` (`crate::processors::device_node`).
pub(super) fn device_node(location: crate::PciLocation) -> Result<crate::DeviceNode, String> {
    let crate::PciLocation {
        domain,
        bus,
        device,
    } = location;
    let path = format!("/sys/bus/pci/devices/{domain:04x}:{bus:02x}:{device:02x}.0/numa_node");
    let text = std::fs::read_to_string(&path).map_err(|error| format!("{path}: {error}"))?;
    crate::processors::parse_device_node(&text).map_err(|error| format!("{path}: {error}"))
}
