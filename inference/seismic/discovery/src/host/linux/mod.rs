//! GNU Linux: `/proc/meminfo`, `/proc/zoneinfo`, `/proc/vmstat`, PSI, the
//! process's memory limits (cgroups, `RLIMIT_AS`, `cpuset.mems`) and its
//! processors, each read through [`Sys`] from a fixed list of sources,
//! with each reading's documented value where none answers.
//!
//! `MemTotal` is OS-usable RAM, not installed RAM. A limit no source shows
//! (a cgroup hierarchy no mount exposes) is no limit: the host's figures
//! and `RLIMIT_AS` bound the process.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use super::{read, resolve};
use seismic_topology::{CacheSize, Reported};
use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;

/// Every OS read discovery makes on Linux, its one owner: the files of
/// `/proc` and `/sys` by absolute path, and the calls that read no file.
/// Each is the value, or none where the source is missing or refused
/// ([`read`]): the reader then goes to the reading's next source.
pub(crate) trait Sys {
    /// The text of the file at the absolute `path`.
    fn text(&self, path: &Path) -> Option<String>;
    /// The raw bytes of the file at the absolute `path`.
    fn bytes(&self, path: &Path) -> Option<Vec<u8>>;
    /// The names in the directory at the absolute `path`.
    fn entries(&self, path: &Path) -> Option<Vec<String>>;
    /// `uname(2)`: the kernel's name, release and version.
    fn uname(&self) -> Option<String>;
    /// The host page size, through [`seismic_memory::host_page`].
    fn page(&self) -> u64;
    /// The CPUs `sched_getaffinity` lets the process run on.
    fn affinity(&self) -> Option<BTreeSet<usize>>;
    /// The CPUs the process may keep busy at once
    /// (`seismic_threads::quota`, their one owner: `available_parallelism`,
    /// its affinity under a cgroup's CPU quota, or one).
    fn parallelism(&self) -> NonZeroUsize;
    /// `getrlimit(RLIMIT_AS)`'s soft limit, in bytes; none where it is
    /// infinite.
    fn address_space_limit(&self) -> Option<u64>;
    /// `sysinfo(2)`: `totalram` and `freeram` in bytes, the second source
    /// of `MemTotal` and `MemFree`.
    fn sysinfo(&self) -> (u64, u64);
}

/// The running kernel's readings.
pub(crate) struct LiveSys;

/// The last OS error as the outcome of the read `what`.
fn last_error<T>(what: &str) -> Option<T> {
    read(what, Err(std::io::Error::last_os_error()))
}

impl Sys for LiveSys {
    fn text(&self, path: &Path) -> Option<String> {
        read(path.display(), std::fs::read_to_string(path))
    }

    fn bytes(&self, path: &Path) -> Option<Vec<u8>> {
        read(path.display(), std::fs::read(path))
    }

    fn entries(&self, path: &Path) -> Option<Vec<String>> {
        read(
            path.display(),
            std::fs::read_dir(path).and_then(|entries| {
                entries
                    .map(|entry| entry.map(|entry| entry.file_name().to_string_lossy().into_owned()))
                    .collect()
            }),
        )
    }

    fn uname(&self) -> Option<String> {
        let mut name = unsafe { std::mem::zeroed::<libc::utsname>() };
        if unsafe { libc::uname(&mut name) } != 0 {
            return last_error("uname");
        }
        // SAFETY: `uname` fills each field with a NUL-terminated string.
        let text = |field: &[libc::c_char]| {
            unsafe { std::ffi::CStr::from_ptr(field.as_ptr()) }
                .to_string_lossy()
                .into_owned()
        };
        Some(format!(
            "{} {} {}",
            text(&name.sysname),
            text(&name.release),
            text(&name.version)
        ))
    }

    fn page(&self) -> u64 {
        static PAGE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        *PAGE.get_or_init(|| {
            seismic_memory::host_page()
                .unwrap_or_else(|error| panic!("invariant violated: the host page: {error}"))
        })
    }

    fn affinity(&self) -> Option<BTreeSet<usize>> {
        // SAFETY: a zeroed set the kernel fills for the calling thread.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
                return last_error("sched_getaffinity");
            }
            Some(
                (0..libc::CPU_SETSIZE as usize)
                    .filter(|cpu| libc::CPU_ISSET(*cpu, &set))
                    .collect(),
            )
        }
    }

    fn parallelism(&self) -> NonZeroUsize {
        seismic_threads::quota()
    }

    fn address_space_limit(&self) -> Option<u64> {
        let mut limit = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
        if unsafe { libc::getrlimit(libc::RLIMIT_AS, &mut limit) } != 0 {
            return last_error("getrlimit(RLIMIT_AS)");
        }
        (limit.rlim_cur != libc::RLIM_INFINITY).then_some(limit.rlim_cur)
    }

    fn sysinfo(&self) -> (u64, u64) {
        let mut info = unsafe { std::mem::zeroed::<libc::sysinfo>() };
        // SAFETY: `sysinfo` fills the struct; it fails only for a bad pointer.
        let status = unsafe { libc::sysinfo(&mut info) };
        assert_eq!(status, 0, "invariant violated: sysinfo: {}", std::io::Error::last_os_error());
        let unit = u64::from(info.mem_unit);
        (u64::from(info.totalram) * unit, u64::from(info.freeram) * unit)
    }
}

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

/// Human-readable CPU brand string from `/proc/cpuinfo`'s `model name`,
/// falling back to `"CPU (<arch>)"`.
pub(super) fn cpu_model(sys: &impl Sys) -> String {
    resolve(
        &[((), &|| {
            sys.text(Path::new("/proc/cpuinfo"))?
                .lines()
                .find_map(|line| {
                    let (key, value) = line.split_once(':')?;
                    (key.trim() == "model name").then(|| value.trim().to_owned())
                })
                .filter(|name| !name.is_empty())
        })],
        (),
        || format!("CPU ({})", std::env::consts::ARCH),
    )
    .0
}

/// `/proc/cpuinfo`'s identifying fields; where it is unreadable or names
/// none, the architecture alone (`unknown;arch=<arch>`).
pub(super) fn cpu_identity(sys: &impl Sys) -> String {
    resolve(
        &[((), &|| {
            let fields = sys
                .text(Path::new("/proc/cpuinfo"))?
                .lines()
                .filter_map(|line| {
                    let (key, value) = line.split_once(':')?;
                    let key = key.trim();
                    CPU_IDENTITY_FIELDS
                        .contains(&key)
                        .then(|| format!("{key}={}", value.trim()))
                })
                .collect::<BTreeSet<_>>();
            (!fields.is_empty()).then(|| fields.into_iter().collect::<Vec<_>>().join(";"))
        })],
        (),
        || format!("unknown;arch={}", std::env::consts::ARCH),
    )
    .0
}

/// Host RAM's capacity: the host's total ([`memory::host_total`]).
pub(super) fn capacity(sys: &impl Sys) -> Reported<u64> {
    memory::host_total(sys)
}

/// `uname(2)`: the kernel's name, release and version; `Linux` where it
/// fails.
pub(super) fn os_build(sys: &impl Sys) -> String {
    resolve(&[((), &|| sys.uname())], (), || "Linux".into()).0
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
/// level's size or sharing, unknown, the largest L2; where there is none
/// of those either (cacheinfo unreadable), [`CacheSize::UNREPORTED`].
///
/// # Panics
/// Where an attribute present does not parse.
pub(super) fn last_level_cache(sys: &impl Sys) -> CacheSize {
    let root = Path::new("/sys/devices/system/cpu");
    let mut caches = Vec::new();
    if let Some(entries) = sys.entries(root) {
        for name in entries {
            let is_cpu = name
                .strip_prefix("cpu")
                .is_some_and(|index| !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()));
            if is_cpu {
                cpu_caches(sys, &root.join(name).join("cache"), &mut caches);
            }
        }
    }
    let last_level = || {
        let levels = caches
            .iter()
            .map(|cache| cache.level)
            .collect::<Option<Vec<_>>>()?;
        let &last = levels.iter().max()?;
        let mut instances = std::collections::BTreeMap::new();
        for cache in caches.iter().filter(|cache| cache.level == Some(last)) {
            instances.insert(cache.shared.clone()?, cache.bytes?);
        }
        Some(instances.values().sum())
    };
    let largest_l2 = || {
        caches
            .iter()
            .filter(|cache| cache.level == Some(2))
            .filter_map(|cache| cache.bytes)
            .max()
    };
    let (bytes, is_last_level) = resolve(
        &[(true, &last_level), (false, &largest_l2)],
        false,
        || 0,
    );
    CacheSize {
        bytes,
        is_last_level,
    }
}

/// The data and unified caches of one CPU's cacheinfo directory `root`:
/// none where it is absent.
fn cpu_caches(sys: &impl Sys, root: &Path, caches: &mut Vec<CpuCache>) {
    let Some(entries) = sys.entries(root) else {
        return;
    };
    for name in entries.into_iter().filter(|name| name.starts_with("index")) {
        let index = root.join(name);
        let field = |name: &str| sys.text(&index.join(name)).map(|text| text.trim().to_owned());
        let unparsed = |name: &str, value: &str| {
            format!(
                "invariant violated: {}: {value:?} is not a cache {name}",
                index.join(name).display()
            )
        };
        if !matches!(field("type").as_deref(), Some("Data" | "Unified")) {
            continue;
        }
        let level = field("level").map(|level| {
            level
                .parse::<u32>()
                .unwrap_or_else(|_| panic!("{}", unparsed("level", &level)))
        });
        let bytes = field("size").map(|size| {
            size.strip_suffix('K')
                .and_then(|kib| kib.parse::<u64>().ok())
                .and_then(|kib| kib.checked_mul(1024))
                .unwrap_or_else(|| panic!("{}", unparsed("size", &size)))
        });
        caches.push(CpuCache {
            level,
            bytes,
            shared: field("shared_cpu_list"),
        });
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

/// A Linux machine's OS readings for tests: a tree of files under `root`
/// standing for `/`, every path under one of `refused` refused, and the
/// calls' answers. [`Tree::new`] is a four-CPU machine with nothing in its
/// tree and no rlimit; each test writes what its case shows.
#[cfg(test)]
pub(crate) struct Tree {
    pub(crate) root: PathBuf,
    pub(crate) refused: Vec<PathBuf>,
    pub(crate) uname: Option<String>,
    pub(crate) page: u64,
    pub(crate) affinity: Option<BTreeSet<usize>>,
    pub(crate) parallelism: NonZeroUsize,
    pub(crate) address_space_limit: Option<u64>,
    /// `sysinfo`'s total and free bytes.
    pub(crate) sysinfo: (u64, u64),
}

#[cfg(test)]
impl Tree {
    pub(crate) fn new(root: PathBuf) -> Self {
        Self {
            root,
            refused: Vec::new(),
            uname: Some("Linux 6.8.0-49-generic #49-Ubuntu SMP".into()),
            page: 4096,
            affinity: Some((0..4).collect()),
            parallelism: NonZeroUsize::new(4).unwrap(),
            address_space_limit: None,
            sysinfo: (16 << 30, 8 << 30),
        }
    }

    /// Write `text` at the absolute `path` in the tree.
    pub(crate) fn file(&self, path: &str, text: &str) {
        let at = self.under(Path::new(path));
        std::fs::create_dir_all(at.parent().unwrap()).unwrap();
        std::fs::write(at, text).unwrap();
    }

    fn under(&self, path: &Path) -> PathBuf {
        self.root.join(path.strip_prefix("/").unwrap_or(path))
    }

    fn refuses(&self, path: &Path) -> bool {
        self.refused.iter().any(|refused| path.starts_with(refused))
    }
}

#[cfg(test)]
impl Sys for Tree {
    fn text(&self, path: &Path) -> Option<String> {
        (!self.refuses(path))
            .then(|| LiveSys.text(&self.under(path)))
            .flatten()
    }
    fn bytes(&self, path: &Path) -> Option<Vec<u8>> {
        (!self.refuses(path))
            .then(|| LiveSys.bytes(&self.under(path)))
            .flatten()
    }
    fn entries(&self, path: &Path) -> Option<Vec<String>> {
        (!self.refuses(path))
            .then(|| LiveSys.entries(&self.under(path)))
            .flatten()
    }
    fn uname(&self) -> Option<String> {
        self.uname.clone()
    }
    fn page(&self) -> u64 {
        self.page
    }
    fn affinity(&self) -> Option<BTreeSet<usize>> {
        self.affinity.clone()
    }
    fn parallelism(&self) -> NonZeroUsize {
        self.parallelism
    }
    fn address_space_limit(&self) -> Option<u64> {
        self.address_space_limit
    }
    fn sysinfo(&self) -> (u64, u64) {
        self.sysinfo
    }
}

mod limits;
mod memory;
mod topology;
#[cfg(test)]
mod tests;

pub(crate) use limits::diagnostics;
pub(crate) use memory::sample;
pub(crate) use topology::{device_node, firmware, pin, place_process, pressure_files, processors};
