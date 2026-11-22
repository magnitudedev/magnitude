//! The limits on this process's memory below the host's figures: its
//! cgroups' (v2 every visible level, v1 the hierarchical limit), its
//! address space's (`RLIMIT_AS`) and its memory nodes' (`cpuset.mems`).
//! A limit any of whose sources is missing or refused is no limit: the
//! host's figures and the others bound the process. The one report of a
//! limit not seen is [`diagnostics`]: a container whose own cgroup sets
//! none.

use super::super::{resolve, Limit, ProcessLimitKind};
use super::memory::{clean_cache, kib_field, parse_bytes, stat_field};
use super::topology::parse_range_list;
use super::{unescape, Sys};
use seismic_topology::{Diagnostic, DiagnosticSource};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Every limit on this process's memory now, cgroups' innermost first
/// ([`process_limits`] of the cgroups it is in now).
#[cfg(test)]
pub(super) fn limits(sys: &impl Sys) -> Vec<Limit> {
    process_limits(sys, &Cgroups::of_process(sys))
}

/// Every limit on this process's memory now, within `cgroups`: theirs
/// innermost first, then `RLIMIT_AS`'s and the memory nodes'.
pub(super) fn process_limits(sys: &impl Sys, cgroups: &Cgroups) -> Vec<Limit> {
    let status = sys.text(Path::new("/proc/self/status"));
    let mut limits = cgroups.limits(sys);
    limits.extend(address_space(sys, status.as_deref()));
    limits.extend(status.as_deref().and_then(|status| allowed_nodes(sys, status)));
    limits
}

/// The cgroup v2 holding this process alone ([`Cgroups::own`]), found now.
#[cfg(test)]
pub(super) fn own_cgroup(sys: &impl Sys) -> Option<PathBuf> {
    Cgroups::of_process(sys).own(sys)
}

/// "container memory limit not set", exactly where this process's cgroup
/// v2 mount is a private cgroup namespace's root (the mount point's own
/// `memory.max` exists: the true root has none) and no level from this
/// process's cgroup up to it sets a limit (every `memory.max` and
/// `memory.high` is `max`). A limit set above a namespace is invisible
/// inside it, so the host's RAM is the bound this process sees; a cgroup
/// v1 limit is read past namespaces (`hierarchical_memory_limit`), so
/// needs no line.
pub(crate) fn diagnostics(sys: &impl Sys) -> Vec<Diagnostic> {
    let Some(v2) = Cgroups::of_process(sys).v2 else {
        return Vec::new();
    };
    let namespace = sys.text(&v2.point.join("memory.max")).is_some();
    let unlimited = || {
        v2.levels().all(|(directory, _)| {
            ["memory.max", "memory.high"].iter().all(|name| {
                sys.text(&directory.join(name))
                    .is_none_or(|value| value.trim() == "max")
            })
        })
    };
    match namespace && unlimited() {
        true => vec![Diagnostic {
            source: DiagnosticSource::Host,
            message: "container memory limit not set: this container's cgroup sets no memory.max or memory.high, \
                      so any limit above it is invisible and the host's RAM is taken as the bound; set one \
                      (Docker `--memory`, Kubernetes `resources.limits.memory`)"
                .into(),
        }],
        false => Vec::new(),
    }
}

/// This process's cgroups, found once: the v1 `memory` hierarchy's, where
/// `/proc/self/cgroup` lists the memory controller in one (legacy and
/// hybrid hosts), and the unified (v2) hierarchy's, each where its
/// hierarchy is mounted with this process's cgroup inside its root. A
/// controller is bound to one hierarchy, so on a hybrid host the v2
/// cgroups have no memory files and every memory limit is the v1
/// hierarchy's.
pub(super) struct Cgroups {
    v1: Option<CgroupMount>,
    v2: Option<CgroupMount>,
}

impl Cgroups {
    /// This process's now (`/proc/self/cgroup`, `/proc/self/mountinfo`),
    /// read at each call so that a process moved to another cgroup is read
    /// in the one it is in; none where either is missing (a kernel without
    /// cgroups, a sandbox's `/proc`) or the hierarchy is not mounted.
    pub(super) fn of_process(sys: &impl Sys) -> Self {
        let (Some(membership), Some(mountinfo)) = (
            sys.text(Path::new("/proc/self/cgroup")),
            sys.text(Path::new("/proc/self/mountinfo")),
        ) else {
            return Self { v1: None, v2: None };
        };
        let listed = |memory: bool| {
            membership.lines().find_map(|line| {
                let mut fields = line.splitn(3, ':');
                let (_, controllers, path) = (fields.next()?, fields.next()?, fields.next()?);
                let found = match memory {
                    true => controllers.split(',').any(|controller| controller == "memory"),
                    false => controllers.is_empty(),
                };
                found.then_some(path)
            })
        };
        Self {
            v1: listed(true).and_then(|path| {
                cgroup_mount(&mountinfo, path, |filesystem, options| {
                    filesystem == "cgroup" && options.split(',').any(|option| option == "memory")
                })
            }),
            v2: listed(false)
                .and_then(|path| cgroup_mount(&mountinfo, path, |filesystem, _| filesystem == "cgroup2")),
        }
    }

    /// This process's cgroup v2 directory: whose `memory.pressure` holds
    /// its stalls.
    pub(super) fn unified(&self) -> Option<&Path> {
        self.v2.as_ref().map(|v2| v2.cgroup.as_path())
    }

    /// The cgroup whose counts are this process's own, the one owner of
    /// that rule for every count read from it (harm's refaults, the cold
    /// and zswapped pages, each from its `memory.stat`, none where it has
    /// no memory controller): this process's cgroup v2 where it lists this
    /// process alone (`cgroup.procs`) and has no descendant (`cgroup.stat`'s
    /// `nr_descendants`), whose processes its hierarchical counts would
    /// include. None otherwise: a cgroup shared with other processes (a
    /// `session-N.scope`, `user.slice`, a container's root) counts their
    /// events too. None under cgroup v1 alone, and none where either file
    /// is missing.
    pub(super) fn own(&self, sys: &impl Sys) -> Option<PathBuf> {
        let directory = self.unified()?;
        let pid = std::process::id().to_string();
        if !sys
            .text(&directory.join("cgroup.procs"))?
            .lines()
            .eq([pid.as_str()])
        {
            return None;
        }
        let path = directory.join("cgroup.stat");
        let descendants = stat_field(&path, &sys.text(&path)?, "nr_descendants")?;
        (descendants == 0).then(|| directory.to_path_buf())
    }

    /// The cgroups' limits now, innermost first: the v1 hierarchical limit,
    /// and each v2 level's from this process's cgroup up to the mount.
    fn limits(&self, sys: &impl Sys) -> Vec<Limit> {
        let v1 = self.v1.as_ref().and_then(|v1| v1_limit(sys, v1));
        let v2 = self
            .v2
            .iter()
            .flat_map(|v2| v2.levels())
            .filter_map(|(directory, cgroup)| v2_limit(sys, &directory, cgroup));
        v1.into_iter().chain(v2).collect()
    }
}

/// OOM-safety reserve below a Linux cgroup memory limit (`memory.max` /
/// `memory.high` / `hierarchical_memory_limit`): unlike host RAM (which has
/// zone `low` watermarks and kswapd), a cgroup limit has no kernel watermark
/// and OOM-kills immediately on overrun.
///
/// Reserves `min(limit / 12, 2 GiB)` when the cgroup can spill anonymous
/// bursts to swap, and `min(limit / 12, 4 GiB)` when swap is explicitly
/// disabled (`memory.swap.max == 0` in cgroup v2 or `memsw == limit` in
/// cgroup v1), where uncharged serving/driver RSS and mapped executable
/// fault-ins cannot page out and otherwise thrash `file_mapped` until
/// OOM-kill. Reserves `0` on tiny synthetic limits (`< 64 MiB`).
pub(super) fn cgroup_floor(limit_bytes: u64, swap_disabled: bool) -> u64 {
    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;
    if limit_bytes < 64 * MIB {
        return 0;
    }
    let cap = if swap_disabled { 4 * GIB } else { 2 * GIB };
    (limit_bytes / 12).min(cap)
}

/// The v1 limit of this process's cgroup: its `memory.stat`'s
/// `hierarchical_memory_limit`, the least `memory.limit_in_bytes` of it and
/// every ancestor, which the kernel computes past any cgroup namespace or
/// mount root. It is enforced against the usage of the level that sets it,
/// which its siblings share: the visible level whose own
/// `memory.limit_in_bytes` is that limit, else (the level is above the
/// mount) the mount's root, the outermost visible; its clean file cache
/// from that level's hierarchical (`total_`) counts, excluding mapped
/// regular-file pages (`total_mapped_file − total_shmem`). None where the
/// limit is the page counter's maximum ([`v1_unlimited_bytes`]) or a file
/// is missing.
fn v1_limit(sys: &impl Sys, mount: &CgroupMount) -> Option<Limit> {
    let own = mount.cgroup.join("memory.stat");
    let limit_bytes = stat_field(&own, &sys.text(&own)?, "hierarchical_memory_limit")?;
    if limit_bytes >= v1_unlimited_bytes(sys.page()) {
        return None;
    }
    let levels = mount
        .levels()
        .map(|(directory, _)| directory)
        .collect::<Vec<_>>();
    let setter = levels
        .iter()
        .find(|directory| read_bytes(sys, &directory.join("memory.limit_in_bytes")) == Some(limit_bytes))
        .or(levels.last())?;
    let raw_used = read_bytes(sys, &setter.join("memory.usage_in_bytes"))?;
    let path = setter.join("memory.stat");
    let stat = sys.text(&path)?;
    let cache = [
        "total_active_file",
        "total_inactive_file",
        "total_dirty",
        "total_writeback",
    ];
    let mapped_file = stat_field(&path, &stat, "total_mapped_file")
        .unwrap_or(0)
        .saturating_sub(stat_field(&path, &stat, "total_shmem").unwrap_or(0));
    let reclaimable_bytes = clean_cache(
        cache.map(|key| stat_field(&path, &stat, key)),
        mapped_file,
    )
    .min(raw_used);
    let swap_disabled =
        read_bytes(sys, &setter.join("memory.memsw.limit_in_bytes")) == Some(limit_bytes);
    Some(Limit {
        kind: ProcessLimitKind::CgroupV1 {
            cgroup: mount.name(setter),
        },
        limit_bytes,
        used_bytes: raw_used.saturating_add(cgroup_floor(limit_bytes, swap_disabled)),
        reclaimable_bytes,
    })
}

/// A cgroup v2 level's limit: the lower of `memory.max` and `memory.high`
/// (past which the kernel reclaims the cgroup and throttles it), enforced
/// against its `memory.current` plus [`cgroup_floor`]; its clean file cache
/// from its `memory.stat`, excluding mapped regular-file pages
/// (`file_mapped − shmem`, executables and shared libraries). None where both
/// are `max` or missing (the root, or no memory controller), or where
/// `memory.current` is missing.
fn v2_limit(sys: &impl Sys, directory: &Path, cgroup: String) -> Option<Limit> {
    let limit_bytes = ["memory.max", "memory.high"]
        .into_iter()
        .filter_map(|name| {
            let file = directory.join(name);
            let value = sys.text(&file)?;
            let value = value.trim();
            (value != "max").then(|| parse_bytes(&file, value))
        })
        .min()?;
    let raw_used = read_bytes(sys, &directory.join("memory.current"))?;
    let path = directory.join("memory.stat");
    let cache = ["active_file", "inactive_file", "file_dirty", "file_writeback"];
    let stat = sys.text(&path);
    let mapped_file = stat
        .as_deref()
        .and_then(|stat| stat_field(&path, stat, "file_mapped"))
        .unwrap_or(0)
        .saturating_sub(
            stat.as_deref()
                .and_then(|stat| stat_field(&path, stat, "shmem"))
                .unwrap_or(0),
        );
    let reclaimable_bytes = clean_cache(
        cache.map(|key| stat.as_deref().and_then(|stat| stat_field(&path, stat, key))),
        mapped_file,
    )
    .min(raw_used);
    let swap_disabled = sys
        .text(&directory.join("memory.swap.max"))
        .is_some_and(|value| value.trim() == "0");
    Some(Limit {
        kind: ProcessLimitKind::CgroupV2 { cgroup },
        limit_bytes,
        used_bytes: raw_used.saturating_add(cgroup_floor(limit_bytes, swap_disabled)),
        reclaimable_bytes,
    })
}

/// `RLIMIT_AS`, enforced against the process's address space: `VmSize` of
/// `status` (the text of `/proc/self/status`), else the first field of
/// `/proc/self/statm` (the same count, in pages); none where it is
/// infinite. Address space is no physical page, so none of it is reclaimed
/// cache.
///
/// # Panics
/// Where neither `status` nor `statm` gives the size: every Linux reports
/// a process's own address space.
fn address_space(sys: &impl Sys, status: Option<&str>) -> Option<Limit> {
    let limit_bytes = sys.address_space_limit()?;
    let used_bytes = resolve(
        &[
            ((), &|| status.and_then(|status| kib_field(status, "VmSize:"))),
            ((), &|| {
                let statm = sys.text(Path::new("/proc/self/statm"))?;
                let pages = statm
                    .split_whitespace()
                    .next()
                    .and_then(|size| size.parse::<u64>().ok())
                    .unwrap_or_else(|| {
                        panic!("invariant violated: /proc/self/statm reads {statm:?}")
                    });
                Some(pages * sys.page())
            }),
        ],
        (),
        || panic!("invariant violated: no /proc/self/status or statm"),
    )
    .0;
    Some(Limit {
        kind: ProcessLimitKind::AddressSpace,
        limit_bytes,
        used_bytes,
        reclaimable_bytes: 0,
    })
}

/// The memory nodes `cpuset.mems` confines this process to
/// (`Mems_allowed_list` of `status`, the text of `/proc/self/status`) where
/// they are a strict subset of the nodes with memory
/// (`/sys/devices/system/node/has_memory`): every allocation is placed in
/// them, so their `MemTotal` is the limit, used but for their `MemFree`
/// above their reclaim floor ([`super::memory::nodes_floor`]), their clean
/// file cache reclaimable (each node's
/// `/sys/devices/system/node/node<N>/meminfo`). None where every node with
/// memory is allowed or a file is missing.
///
/// # Panics
/// Where no allowed node has memory: the kernel keeps a task's allowed
/// nodes within those with memory.
fn allowed_nodes(sys: &impl Sys, status: &str) -> Option<Limit> {
    let allowed = node_set(
        "Mems_allowed_list",
        status
            .lines()
            .find_map(|line| line.strip_prefix("Mems_allowed_list:"))?,
    );
    let has_memory = "/sys/devices/system/node/has_memory";
    let with_memory = node_set(has_memory, &sys.text(Path::new(has_memory))?);
    let nodes = allowed.intersection(&with_memory).copied().collect::<Vec<_>>();
    assert!(
        !nodes.is_empty(),
        "invariant violated: allowed nodes {allowed:?} hold none of those with memory, {with_memory:?}"
    );
    if nodes.len() == with_memory.len() {
        return None;
    }
    let (mut total, mut free, mut cache) = (0, 0, 0);
    for node in &nodes {
        // `Node <N> <field>: <kB> kB`: each line read as `/proc/meminfo`'s.
        let text = sys.text(&PathBuf::from(format!(
            "/sys/devices/system/node/node{node}/meminfo"
        )))?;
        let meminfo = text
            .lines()
            .filter_map(|line| line.splitn(3, ' ').nth(2))
            .collect::<Vec<_>>()
            .join("\n");
        total += kib_field(&meminfo, "MemTotal:")?;
        free += kib_field(&meminfo, "MemFree:")?;
        cache += clean_cache(
            ["Active(file):", "Inactive(file):", "Dirty:", "Writeback:"]
                .map(|name| kib_field(&meminfo, name)),
            kib_field(&meminfo, "Mapped:")
                .unwrap_or(0)
                .saturating_sub(kib_field(&meminfo, "Shmem:").unwrap_or(0)),
        );
    }
    let floor = super::memory::nodes_floor(sys, &nodes, total);
    Some(Limit {
        kind: ProcessLimitKind::AllowedNodes { nodes },
        limit_bytes: total,
        used_bytes: total.saturating_sub(free).saturating_add(floor),
        reclaimable_bytes: cache,
    })
}

/// A kernel node list (`0-3,5`) `what` reads.
///
/// # Panics
/// Where it does not parse.
fn node_set(what: &str, text: &str) -> BTreeSet<usize> {
    parse_range_list(text)
        .unwrap_or_else(|error| panic!("invariant violated: {what} reads {text:?}: {error}"))
        .into_iter()
        .collect()
}

/// The v1 memory controller reports "no limit" as its page counter maximum,
/// `LONG_MAX` rounded down to the page size `page`.
fn v1_unlimited_bytes(page: u64) -> u64 {
    i64::MAX as u64 / page * page
}

/// A mounted cgroup hierarchy and this process's cgroup in it.
struct CgroupMount {
    /// The mount point.
    point: PathBuf,
    /// This process's cgroup directory.
    cgroup: PathBuf,
}

impl CgroupMount {
    /// The cgroup at `directory`, a directory within the mount, named by its
    /// path from the mount's root.
    fn name(&self, directory: &Path) -> String {
        match directory.strip_prefix(&self.point) {
            Ok(relative) => format!("/{}", relative.display()),
            Err(_) => unreachable!("a cgroup within its mount"),
        }
    }

    /// This process's cgroup and each ancestor up to the mount point, each
    /// with its name.
    fn levels(&self) -> impl Iterator<Item = (PathBuf, String)> + '_ {
        self.cgroup
            .ancestors()
            .take_while(|directory| directory.starts_with(&self.point))
            .map(|directory| (directory.to_path_buf(), self.name(directory)))
    }
}

/// The first mount of `mountinfo` matching `accepts(filesystem type, super
/// options)` whose root contains the cgroup `path`; none where none is
/// visible.
fn cgroup_mount(
    mountinfo: &str,
    path: &str,
    accepts: impl Fn(&str, &str) -> bool,
) -> Option<CgroupMount> {
    mountinfo.lines().find_map(|line| {
        let (mount, filesystem) = line.split_once(" - ")?;
        let mut filesystem = filesystem.split_whitespace();
        let (kind, _source, options) = (filesystem.next()?, filesystem.next()?, filesystem.next()?);
        if !accepts(kind, options) {
            return None;
        }
        let mut fields = mount.split_whitespace();
        let (mounted, point) = (
            unescape(fields.nth(3)?),
            PathBuf::from(unescape(fields.next()?)),
        );
        let rest = path.strip_prefix(mounted.trim_end_matches('/'))?;
        (rest.is_empty() || rest.starts_with('/')).then(|| CgroupMount {
            cgroup: point.join(rest.trim_start_matches('/')),
            point,
        })
    })
}

/// The figure the file `path` holds, none where it is missing.
///
/// # Panics
/// Where it does not parse.
fn read_bytes(sys: &impl Sys, path: &Path) -> Option<u64> {
    Some(parse_bytes(path, sys.text(path)?.trim()))
}
