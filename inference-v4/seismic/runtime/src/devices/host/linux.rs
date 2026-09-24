//! GNU Linux: `/proc/meminfo`, `getrlimit` and cgroup memory limits.
//!
//! `MemTotal` is OS-usable RAM, not installed RAM; `MemAvailable` is the
//! kernel's estimate. cgroup limits are read from this process's cgroup and
//! every visible ancestor, in whichever hierarchy the memory controller is
//! bound to: a v1 `memory` hierarchy (legacy and hybrid hosts) or the v2
//! unified hierarchy. Ancestors hidden by a cgroup namespace or a container's
//! mount are reported as such, never presumed unlimited.

use super::{
    HeadroomBasis, HeadroomEstimate, HostCapacity, HostMeasurements, HostMemoryStatus,
    LimitVisibility, ProcessLimitKind, ProcessMemoryLimit,
};
use crate::devices::{CapacityBasis, ObservationError};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

pub(super) fn capacity() -> Result<HostCapacity, String> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").map_err(|error| error.to_string())?;
    let bytes = kib_field(&meminfo, "MemTotal:")
        .ok_or_else(|| "/proc/meminfo MemTotal is absent or invalid".to_owned())?;
    Ok(HostCapacity {
        bytes,
        basis: CapacityBasis::OsUsableRam,
    })
}

pub(super) fn status() -> Result<HostMemoryStatus, ObservationError> {
    let meminfo = read("/proc/meminfo")?;
    let sampled_at = SystemTime::now();
    let mem_free_bytes = kib_field(&meminfo, "MemFree:")
        .ok_or_else(|| ObservationError::Failed("/proc/meminfo MemFree is absent".into()))?;
    let mem_available_bytes = kib_field(&meminfo, "MemAvailable:").ok_or_else(|| {
        ObservationError::Unsupported(
            "/proc/meminfo has no MemAvailable estimate (Linux 3.14+ required)".into(),
        )
    })?;
    let mut limits = resource_limits()?;
    let limit_visibility = cgroup_limits(Path::new("/"), &mut limits)?;
    Ok(HostMemoryStatus {
        sampled_at,
        measurements: HostMeasurements::Linux {
            mem_free_bytes,
            mem_available_bytes,
        },
        headroom: HeadroomEstimate {
            bytes: mem_available_bytes,
            basis: HeadroomBasis::LinuxMemAvailable,
        },
        limits,
        limit_visibility,
    })
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
    V1 { path: &'a str },
    V2 { path: &'a str },
    /// No memory controller is enabled: no cgroup can limit memory.
    None,
}

fn memory_hierarchy(membership: &str) -> MemoryHierarchy<'_> {
    let mut unified = None;
    for line in membership.lines() {
        let mut fields = line.splitn(3, ':');
        let (Some(_), Some(controllers), Some(path)) = (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        if controllers.split(',').any(|controller| controller == "memory") {
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
    let mount = cgroup_mount(root, path, |filesystem, _| filesystem == "cgroup2")?.ok_or_else(|| {
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
        });
        Ok(())
    })?;
    // `cgroup.type` exists only on non-root cgroups: a visible top that has
    // it is a namespace or bind-mount root with hidden ancestors.
    Ok(if mount.point.join("cgroup.type").exists() {
        LimitVisibility::CgroupAncestorsHidden
    } else {
        LimitVisibility::Complete
    })
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
        });
        Ok(())
    })?;
    // A v1 mount exposes its hierarchy from the mounted root down; a root
    // other than the hierarchy's own (a container's cgroup) hides ancestors.
    Ok(if mount.root == "/" {
        LimitVisibility::Complete
    } else {
        LimitVisibility::CgroupAncestorsHidden
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
            let root = std::env::temp_dir()
                .join(format!("seismic-cgroup-{}-{name}", std::process::id()));
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
        }
    }

    const V2_HOST_MOUNT: &str =
        "30 25 0:26 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime shared:4 - cgroup2 cgroup2 rw,nsdelegate\n";
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
        fixture.write("sys/fs/cgroup/user.slice/memory.max", "max\n");
        fixture.write("sys/fs/cgroup/user.slice/memory.current", "900\n");
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
        let (limits, visibility) = fixture.limits().unwrap();
        assert_eq!(
            limits,
            [limit(ProcessLimitKind::CgroupV2 { cgroup: "/".into() }, 2000, 500)]
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
        fixture.write(&format!("{memory}/system.slice/svc.service/memory.limit_in_bytes"), "3000\n");
        fixture.write(&format!("{memory}/system.slice/svc.service/memory.usage_in_bytes"), "1000\n");
        fixture.write(&format!("{memory}/system.slice/memory.limit_in_bytes"), &unlimited);
        fixture.write(&format!("{memory}/system.slice/memory.usage_in_bytes"), "5000\n");
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
        fixture.write("sys/fs/cgroup/unified/user.slice/app.scope/cgroup.procs", "1\n");
        fixture.write("sys/fs/cgroup/memory/user.slice/memory.limit_in_bytes", "6000\n");
        fixture.write("sys/fs/cgroup/memory/user.slice/memory.usage_in_bytes", "2000\n");
        fixture.write("sys/fs/cgroup/memory/memory.limit_in_bytes", &unlimited);
        fixture.write("sys/fs/cgroup/memory/memory.usage_in_bytes", "9000\n");
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
        let (limits, visibility) = fixture.limits().unwrap();
        assert_eq!(
            limits,
            [limit(ProcessLimitKind::CgroupV1 { cgroup: "/".into() }, 4000, 1500)]
        );
        assert_eq!(visibility, LimitVisibility::CgroupAncestorsHidden);
    }

    #[test]
    fn no_memory_controller_admits_no_cgroup_limit() {
        let fixture = Fixture::new("no-memory", "3:cpu,cpuacct:/\n1:name=systemd:/\n", V1_HOST_MOUNTS);
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
        assert!(matches!(fixture.limits(), Err(ObservationError::Unsupported(_))));
    }

    #[test]
    fn an_unmounted_hierarchy_is_unsupported_not_unlimited() {
        let fixture = Fixture::new("unmounted", "2:memory:/docker/abc\n", "");
        assert!(matches!(fixture.limits(), Err(ObservationError::Unsupported(_))));
    }
}
