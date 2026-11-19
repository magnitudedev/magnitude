//! The limits on this process's memory below the host's figures: its
//! cgroups' (v2 every visible level, v1 the hierarchical limit), its
//! address space's (`RLIMIT_AS`) and its memory nodes' (`cpuset.mems`).
//! A limit any of whose sources is missing or refused is no limit: the
//! host's figures and the others bound the process. The one report of a
//! limit not seen is [`diagnostics`]: a container whose own cgroup sets
//! none.

use super::super::{Limit, ProcessLimitKind};
use super::memory::{clean_cache, kib_field, page, parse_bytes, stat_field};
use super::{unescape, Facts};
use seismic_topology::{Diagnostic, DiagnosticSource};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Every limit on this process's memory now, cgroups' innermost first
/// ([`process_limits`] of the cgroups it is in now).
#[cfg(test)]
pub(super) fn limits(facts: &impl Facts) -> Vec<Limit> {
    process_limits(facts, &Cgroups::of_process(facts))
}

/// Every limit on this process's memory now, within `cgroups`: theirs
/// innermost first, then `RLIMIT_AS`'s and the memory nodes'.
pub(super) fn process_limits(facts: &impl Facts, cgroups: &Cgroups) -> Vec<Limit> {
    let status = facts.text(Path::new("/proc/self/status"));
    let mut limits = cgroups.limits(facts);
    limits.extend(address_space(facts, status.as_deref()));
    limits.extend(status.as_deref().and_then(|status| allowed_nodes(facts, status)));
    limits
}

/// The cgroup v2 holding this process alone ([`Cgroups::own`]), found now.
#[cfg(test)]
pub(super) fn own_cgroup(facts: &impl Facts) -> Option<PathBuf> {
    Cgroups::of_process(facts).own(facts)
}

/// "container memory limit not set", exactly where this process's cgroup
/// v2 mount is a private cgroup namespace's root (the mount point's own
/// `memory.max` exists: the true root has none) and no level from this
/// process's cgroup up to it sets a limit (every `memory.max` and
/// `memory.high` is `max`). A limit set above a namespace is invisible
/// inside it, so the host's RAM is the bound this process sees; a cgroup
/// v1 limit is read past namespaces (`hierarchical_memory_limit`), so
/// needs no line.
pub(crate) fn diagnostics(facts: &impl Facts) -> Vec<Diagnostic> {
    let Some(v2) = Cgroups::of_process(facts).v2 else { return Vec::new() };
    let namespace = facts.text(&v2.point.join("memory.max")).is_some();
    let unlimited = || {
        v2.levels().all(|(directory, _)| {
            ["memory.max", "memory.high"].iter().all(|name| facts.text(&directory.join(name)).is_none_or(|value| value.trim() == "max"))
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
    pub(super) fn of_process(facts: &impl Facts) -> Self {
        let (Some(membership), Some(mountinfo)) =
            (facts.text(Path::new("/proc/self/cgroup")), facts.text(Path::new("/proc/self/mountinfo")))
        else {
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
            v2: listed(false).and_then(|path| cgroup_mount(&mountinfo, path, |filesystem, _| filesystem == "cgroup2")),
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
    pub(super) fn own(&self, facts: &impl Facts) -> Option<PathBuf> {
        let directory = self.unified()?;
        let pid = std::process::id().to_string();
        if !facts.text(&directory.join("cgroup.procs"))?.lines().eq([pid.as_str()]) {
            return None;
        }
        let path = directory.join("cgroup.stat");
        let descendants = stat_field(&path, &facts.text(&path)?, "nr_descendants")?;
        (descendants == 0).then(|| directory.to_path_buf())
    }

    /// The cgroups' limits now, innermost first: the v1 hierarchical limit,
    /// and each v2 level's from this process's cgroup up to the mount.
    fn limits(&self, facts: &impl Facts) -> Vec<Limit> {
        let v1 = self.v1.as_ref().and_then(|v1| v1_limit(facts, v1));
        let v2 = self.v2.iter().flat_map(|v2| v2.levels()).filter_map(|(directory, cgroup)| v2_limit(facts, &directory, cgroup));
        v1.into_iter().chain(v2).collect()
    }
}

/// The v1 limit of this process's cgroup: its `memory.stat`'s
/// `hierarchical_memory_limit`, the least `memory.limit_in_bytes` of it and
/// every ancestor, which the kernel computes past any cgroup namespace or
/// mount root. It is enforced against the usage of the level that sets it,
/// which its siblings share: the visible level whose own
/// `memory.limit_in_bytes` is that limit, else (the level is above the
/// mount) the mount's root, the outermost visible; its clean file cache
/// from that level's hierarchical (`total_`) counts. None where the limit
/// is the page counter's maximum ([`v1_unlimited_bytes`]) or a file is
/// missing.
fn v1_limit(facts: &impl Facts, mount: &CgroupMount) -> Option<Limit> {
    let own = mount.cgroup.join("memory.stat");
    let limit_bytes = stat_field(&own, &facts.text(&own)?, "hierarchical_memory_limit")?;
    if limit_bytes >= v1_unlimited_bytes(page(facts)) {
        return None;
    }
    let levels = mount.levels().map(|(directory, _)| directory).collect::<Vec<_>>();
    let setter = levels
        .iter()
        .find(|directory| read_bytes(facts, &directory.join("memory.limit_in_bytes")) == Some(limit_bytes))
        .or(levels.last())?;
    let used_bytes = read_bytes(facts, &setter.join("memory.usage_in_bytes"))?;
    let path = setter.join("memory.stat");
    let stat = facts.text(&path)?;
    let cache = ["total_active_file", "total_inactive_file", "total_dirty", "total_writeback"];
    Some(Limit {
        kind: ProcessLimitKind::CgroupV1 { cgroup: mount.name(setter) },
        limit_bytes,
        used_bytes,
        reclaimable_bytes: clean_cache(cache.map(|key| stat_field(&path, &stat, key))),
    })
}

/// A cgroup v2 level's limit: the lower of `memory.max` and `memory.high`
/// (past which the kernel reclaims the cgroup and throttles it), enforced
/// against its `memory.current`; its clean file cache from its
/// `memory.stat`. None where both are `max` or missing (the root, or no
/// memory controller), or where `memory.current` is missing.
fn v2_limit(facts: &impl Facts, directory: &Path, cgroup: String) -> Option<Limit> {
    let limit_bytes = ["memory.max", "memory.high"]
        .into_iter()
        .filter_map(|name| {
            let file = directory.join(name);
            let value = facts.text(&file)?;
            let value = value.trim();
            (value != "max").then(|| parse_bytes(&file, value))
        })
        .min()?;
    let used_bytes = read_bytes(facts, &directory.join("memory.current"))?;
    let path = directory.join("memory.stat");
    let cache = ["active_file", "inactive_file", "file_dirty", "file_writeback"];
    let reclaimable_bytes = facts.text(&path).map_or(0, |stat| clean_cache(cache.map(|key| stat_field(&path, &stat, key))));
    Some(Limit { kind: ProcessLimitKind::CgroupV2 { cgroup }, limit_bytes, used_bytes, reclaimable_bytes })
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
fn address_space(facts: &impl Facts, status: Option<&str>) -> Option<Limit> {
    let limit_bytes = facts.address_space_limit()?;
    let used_bytes = status.and_then(|status| kib_field(status, "VmSize:")).unwrap_or_else(|| {
        let statm = facts.text(Path::new("/proc/self/statm")).unwrap_or_else(|| panic!("invariant violated: no /proc/self/status or statm"));
        let pages = statm.split_whitespace().next().and_then(|size| size.parse::<u64>().ok());
        pages.unwrap_or_else(|| panic!("invariant violated: /proc/self/statm reads {statm:?}")) * page(facts)
    });
    Some(Limit { kind: ProcessLimitKind::AddressSpace, limit_bytes, used_bytes, reclaimable_bytes: 0 })
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
fn allowed_nodes(facts: &impl Facts, status: &str) -> Option<Limit> {
    let allowed = node_list("Mems_allowed_list", status.lines().find_map(|line| line.strip_prefix("Mems_allowed_list:"))?);
    let has_memory = "/sys/devices/system/node/has_memory";
    let with_memory = node_list(has_memory, &facts.text(Path::new(has_memory))?);
    let nodes = allowed.intersection(&with_memory).copied().collect::<Vec<_>>();
    assert!(!nodes.is_empty(), "invariant violated: allowed nodes {allowed:?} hold none of those with memory, {with_memory:?}");
    if nodes.len() == with_memory.len() {
        return None;
    }
    let (mut total, mut free, mut cache) = (0, 0, 0);
    for node in &nodes {
        // `Node <N> <field>: <kB> kB`: each line read as `/proc/meminfo`'s.
        let text = facts.text(&PathBuf::from(format!("/sys/devices/system/node/node{node}/meminfo")))?;
        let meminfo = text.lines().filter_map(|line| line.splitn(3, ' ').nth(2)).collect::<Vec<_>>().join("\n");
        total += kib_field(&meminfo, "MemTotal:")?;
        free += kib_field(&meminfo, "MemFree:")?;
        cache += clean_cache(["Active(file):", "Inactive(file):", "Dirty:", "Writeback:"].map(|name| kib_field(&meminfo, name)));
    }
    let floor = super::memory::nodes_floor(facts, &nodes, total);
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
fn node_list(what: &str, text: &str) -> BTreeSet<usize> {
    let number = |value: &str| value.parse::<usize>().unwrap_or_else(|_| panic!("invariant violated: {what} reads {text:?}"));
    text.trim()
        .split(',')
        .filter(|range| !range.is_empty())
        .flat_map(|range| match range.split_once('-') {
            Some((first, last)) => number(first)..=number(last),
            None => number(range)..=number(range),
        })
        .collect()
}

/// The v1 memory controller reports "no limit" as its page counter maximum,
/// `LONG_MAX` rounded down to the page size `page`.
fn v1_unlimited_bytes(page: u64) -> u64 {
    i64::MAX as u64 / page * page
}

/// A mounted cgroup hierarchy and this process's cgroup in it.
struct CgroupMount {
    /// This process's cgroup as `/proc/self/cgroup` names it.
    path: String,
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
fn cgroup_mount(mountinfo: &str, path: &str, accepts: impl Fn(&str, &str) -> bool) -> Option<CgroupMount> {
    mountinfo.lines().find_map(|line| {
        let (mount, filesystem) = line.split_once(" - ")?;
        let mut filesystem = filesystem.split_whitespace();
        let (kind, _source, options) = (filesystem.next()?, filesystem.next()?, filesystem.next()?);
        if !accepts(kind, options) {
            return None;
        }
        let mut fields = mount.split_whitespace();
        let (mounted, point) = (unescape(fields.nth(3)?), PathBuf::from(unescape(fields.next()?)));
        let rest = path.strip_prefix(mounted.trim_end_matches('/'))?;
        (rest.is_empty() || rest.starts_with('/')).then(|| CgroupMount {
            path: path.to_owned(),
            cgroup: point.join(rest.trim_start_matches('/')),
            point,
        })
    })
}

/// The figure the file `path` holds, none where it is missing.
///
/// # Panics
/// Where it does not parse.
fn read_bytes(facts: &impl Facts, path: &Path) -> Option<u64> {
    Some(parse_bytes(path, facts.text(path)?.trim()))
}
