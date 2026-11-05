//! GNU Linux: `/proc/meminfo`, `getrlimit` and cgroup memory limits.
//!
//! `MemTotal` is OS-usable RAM, not installed RAM; `MemAvailable` is the
//! kernel's estimate. cgroup limits are read from this process's cgroup and
//! every visible ancestor, in whichever hierarchy the memory controller is
//! bound to: a v1 `memory` hierarchy (legacy and hybrid hosts) or the v2
//! unified hierarchy. Ancestors hidden by a cgroup namespace or a container's
//! mount are reported as such, never presumed unlimited.

use super::{HostSample, LimitVisibility, ObservationError, ProcessLimitKind, ProcessMemoryLimit};
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

pub(super) fn sample() -> Result<HostSample, ObservationError> {
    let meminfo = read("/proc/meminfo")?;
    let mem_available_bytes = kib_field(&meminfo, "MemAvailable:").ok_or_else(|| {
        ObservationError::Unsupported(
            "/proc/meminfo has no MemAvailable estimate (Linux 3.14+ required)".into(),
        )
    })?;
    let mut limits = resource_limits()?;
    let limit_visibility = cgroup_limits(Path::new("/"), &mut limits)?;
    Ok(HostSample {
        // `MemAvailable` is what can be taken without swapping: page cache
        // and reclaimable slab, not other programs' anonymous memory. Linux
        // reports no larger figure that swapping would make available, so
        // a load obtains what any growth may take.
        allocatable: Reported::new(mem_available_bytes, ReportedBy::LinuxMemAvailable),
        obtainable: Reported::new(mem_available_bytes, ReportedBy::LinuxMemAvailable),
        limits,
        limit_visibility,
        kernel_pressure: None,
        held: None,
        // The swap in use, and what zswap saves of the pages it holds
        // (`Zswapped` uncompressed, `Zswap` compressed, where the kernel
        // has it).
        displaced_bytes: Some(displaced_bytes(&meminfo)?),
    })
}

fn displaced_bytes(meminfo: &str) -> Result<u64, ObservationError> {
    let field = |name| {
        kib_field(meminfo, name)
            .ok_or_else(|| ObservationError::Failed(format!("/proc/meminfo has no {name}")))
    };
    let swapped = field("SwapTotal:")?.saturating_sub(field("SwapFree:")?);
    let zswap_saved = kib_field(meminfo, "Zswapped:")
        .zip(kib_field(meminfo, "Zswap:"))
        .map_or(0, |(uncompressed, compressed)| {
            uncompressed.saturating_sub(compressed)
        });
    Ok(swapped + zswap_saved)
}

/// Linux counts no displacement here: `MemAvailable` is the kernel's
/// estimate of what is usable without swapping, so no displacement is read
/// beside it.
pub(super) const DISPLACEMENT_COUNTERS: Option<super::ReadCounters> = None;

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

fn resource_limits() -> Result<Vec<ProcessMemoryLimit>, ObservationError> {
    let mut limits = Vec::new();
    let mut status = None;
    for (resource, kind, usage) in [
        (libc::RLIMIT_AS, ProcessLimitKind::AddressSpace, "VmSize:"),
        (libc::RLIMIT_DATA, ProcessLimitKind::DataSegment, "VmData:"),
    ] {
        let mut limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        if unsafe { libc::getrlimit(resource, &mut limit) } != 0 {
            return Err(ObservationError::Failed(format!(
                "getrlimit failed: {}",
                std::io::Error::last_os_error()
            )));
        }
        if limit.rlim_cur == libc::RLIM_INFINITY {
            continue;
        }
        if status.is_none() {
            status = Some(read("/proc/self/status")?);
        }
        let status = status.as_deref().expect("process status was just read");
        let used_bytes = kib_field(status, usage).ok_or_else(|| {
            ObservationError::Failed(format!("/proc/self/status {usage} is absent"))
        })?;
        limits.push(ProcessMemoryLimit {
            kind,
            limit_bytes: limit.rlim_cur,
            used_bytes,
            reclaimable_bytes: 0,
        });
    }
    Ok(limits)
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

/// Appends every visible cgroup memory limit, from this process's cgroup up
/// to the root of the mounted hierarchy. `root` is the filesystem root the
/// proc and cgroup files are read under (`/` outside tests).
fn cgroup_limits(
    root: &Path,
    limits: &mut Vec<ProcessMemoryLimit>,
) -> Result<LimitVisibility, ObservationError> {
    let membership = read(root.join("proc/self/cgroup"))?;
    match memory_hierarchy(&membership) {
        MemoryHierarchy::V1 { path } => cgroup_v1_limits(root, path, limits),
        MemoryHierarchy::V2 { path } => cgroup_v2_limits(root, path, limits),
        MemoryHierarchy::None => Ok(LimitVisibility::Complete),
    }
}

fn cgroup_v2_limits(
    root: &Path,
    path: &str,
    limits: &mut Vec<ProcessMemoryLimit>,
) -> Result<LimitVisibility, ObservationError> {
    let mount =
        cgroup_mount(root, path, |filesystem, _| filesystem == "cgroup2")?.ok_or_else(|| {
            ObservationError::Unsupported("this process's cgroup2 hierarchy is not mounted".into())
        })?;
    walk_cgroups(&mount, |directory, cgroup| {
        let max = directory.join("memory.max");
        if !max.exists() {
            return Ok(());
        }
        let value = read(&max)?;
        let value = value.trim();
        if value == "max" {
            return Ok(());
        }
        limits.push(ProcessMemoryLimit {
            kind: ProcessLimitKind::CgroupV2 { cgroup },
            limit_bytes: parse_bytes(&max, value)?,
            used_bytes: read_bytes(&directory.join("memory.current"))?,
            reclaimable_bytes: file_cache_headroom(
                directory,
                [
                    "active_file",
                    "inactive_file",
                    "file_dirty",
                    "file_writeback",
                ],
            )?,
        });
        Ok(())
    })?;
    // `cgroup.type` exists only on non-root cgroups: a visible top that has
    // it is a namespace or bind-mount root with hidden ancestors.
    let visibility = if mount.point.join("cgroup.type").exists() {
        LimitVisibility::CgroupAncestorsHidden
    } else {
        LimitVisibility::Complete
    };
    Ok(visibility)
}

fn cgroup_v1_limits(
    root: &Path,
    path: &str,
    limits: &mut Vec<ProcessMemoryLimit>,
) -> Result<LimitVisibility, ObservationError> {
    let mount = cgroup_mount(root, path, |filesystem, options| {
        filesystem == "cgroup" && options.split(',').any(|option| option == "memory")
    })?
    .ok_or_else(|| {
        ObservationError::Unsupported(
            "this process's cgroup v1 memory hierarchy is not mounted".into(),
        )
    })?;
    let unlimited = v1_unlimited_bytes()?;
    walk_cgroups(&mount, |directory, cgroup| {
        let limit = directory.join("memory.limit_in_bytes");
        let limit_bytes = read_bytes(&limit)?;
        if limit_bytes >= unlimited {
            return Ok(());
        }
        limits.push(ProcessMemoryLimit {
            kind: ProcessLimitKind::CgroupV1 { cgroup },
            limit_bytes,
            used_bytes: read_bytes(&directory.join("memory.usage_in_bytes"))?,
            reclaimable_bytes: file_cache_headroom(
                directory,
                [
                    "total_active_file",
                    "total_inactive_file",
                    "total_dirty",
                    "total_writeback",
                ],
            )?,
        });
        Ok(())
    })?;
    // A v1 mount exposes its hierarchy from the mounted root down; a root
    // other than the hierarchy's own (a container's cgroup) hides ancestors.
    let visibility = if mount.root == "/" {
        LimitVisibility::Complete
    } else {
        LimitVisibility::CgroupAncestorsHidden
    };
    Ok(visibility)
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
    /// The hierarchy path mounted at `point` (`/` for the hierarchy root).
    root: String,
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
        return Ok(Some(CgroupMount {
            point,
            root: mounted,
            cgroup,
        }));
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

        fn limits(&self) -> Result<(Vec<ProcessMemoryLimit>, LimitVisibility), ObservationError> {
            let mut limits = Vec::new();
            let visibility = cgroup_limits(&self.0, &mut limits)?;
            Ok((limits, visibility))
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
        let (limits, visibility) = fixture.limits().unwrap();
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
        assert_eq!(visibility, LimitVisibility::Complete);
    }

    #[test]
    fn v2_namespace_root_hides_its_ancestors() {
        let fixture = Fixture::new("v2-namespace", "0::/\n", V2_HOST_MOUNT);
        fixture.write("sys/fs/cgroup/cgroup.type", "domain\n");
        fixture.write("sys/fs/cgroup/memory.max", "2000\n");
        fixture.write("sys/fs/cgroup/memory.current", "500\n");
        fixture.write("sys/fs/cgroup/memory.stat", "");
        let (limits, visibility) = fixture.limits().unwrap();
        assert_eq!(
            limits,
            [limit(
                ProcessLimitKind::CgroupV2 { cgroup: "/".into() },
                2000,
                500
            )]
        );
        assert_eq!(visibility, LimitVisibility::CgroupAncestorsHidden);
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
        let (limits, visibility) = fixture.limits().unwrap();
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
        assert_eq!(visibility, LimitVisibility::Complete);
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
        let (limits, visibility) = fixture.limits().unwrap();
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
        assert_eq!(visibility, LimitVisibility::Complete);
    }

    #[test]
    fn v2_limits_come_from_max_alone() {
        // `memory.high` is a throttling threshold, not a limit: only
        // `memory.max` bounds this process, enforced against `memory.current`.
        let fixture = Fixture::new("v2-high", "0::/user.slice/app.scope\n", V2_HOST_MOUNT);
        fixture.write("sys/fs/cgroup/user.slice/app.scope/memory.max", "1000\n");
        fixture.write("sys/fs/cgroup/user.slice/app.scope/memory.high", "800\n");
        fixture.write("sys/fs/cgroup/user.slice/app.scope/memory.current", "810\n");
        fixture.write("sys/fs/cgroup/user.slice/app.scope/memory.stat", "");
        fixture.write("sys/fs/cgroup/user.slice/memory.max", "max\n");
        fixture.write("sys/fs/cgroup/user.slice/memory.high", "700\n");
        fixture.write("sys/fs/cgroup/user.slice/memory.current", "750\n");
        fixture.write("sys/fs/cgroup/user.slice/memory.stat", "");
        let (limits, visibility) = fixture.limits().unwrap();
        assert_eq!(
            limits,
            [limit(
                ProcessLimitKind::CgroupV2 {
                    cgroup: "/user.slice/app.scope".into()
                },
                1000,
                810
            )]
        );
        assert_eq!(visibility, LimitVisibility::Complete);
        assert_eq!(limits[0].remaining_bytes(), 190);
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
        let (limits, visibility) = fixture.limits().unwrap();
        assert_eq!(
            limits,
            [limit(
                ProcessLimitKind::CgroupV1 { cgroup: "/".into() },
                4000,
                1500
            )]
        );
        assert_eq!(visibility, LimitVisibility::CgroupAncestorsHidden);
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
            let (limits, _) = fixture.limits().unwrap();
            assert_eq!(limits[0].used_bytes, 900);
            assert_eq!(limits[0].reclaimable_bytes, reclaimable);
            assert_eq!(limits[0].remaining_bytes(), remaining);
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
        let (limits, _) = fixture.limits().unwrap();
        assert_eq!(limits[0].reclaimable_bytes, 450);
        assert_eq!(limits[0].remaining_bytes(), 550);
    }

    #[test]
    fn no_memory_controller_admits_no_cgroup_limit() {
        let fixture = Fixture::new(
            "no-memory",
            "3:cpu,cpuacct:/\n1:name=systemd:/\n",
            V1_HOST_MOUNTS,
        );
        let (limits, visibility) = fixture.limits().unwrap();
        assert!(limits.is_empty());
        assert_eq!(visibility, LimitVisibility::Complete);
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
