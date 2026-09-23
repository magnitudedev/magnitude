//! GNU Linux: `/proc/meminfo`, `getrlimit` and cgroup v2 limits.
//!
//! `MemTotal` is OS-usable RAM, not installed RAM; `MemAvailable` is the
//! kernel's estimate. cgroup limits are read from this process's cgroup and
//! every visible ancestor; ancestors hidden by a cgroup namespace are
//! reported as such, never presumed unlimited.

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
    let limit_visibility = cgroup_limits(&mut limits)?;
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

/// Appends every visible cgroup v2 `memory.max` from this process's cgroup
/// up to the visible hierarchy root.
fn cgroup_limits(limits: &mut Vec<ProcessMemoryLimit>) -> Result<LimitVisibility, ObservationError> {
    let membership = read("/proc/self/cgroup")?;
    let Some(path) = membership.lines().find_map(|line| line.strip_prefix("0::")) else {
        let v1_memory = membership.lines().any(|line| {
            line.split(':')
                .nth(1)
                .is_some_and(|controllers| controllers.split(',').any(|c| c == "memory"))
        });
        if v1_memory {
            return Err(ObservationError::Unsupported(
                "cgroup v1 memory controller limits are not supported".into(),
            ));
        }
        return Ok(LimitVisibility::Complete);
    };
    let (mount, root) = cgroup2_mount()?;
    let relative = path.strip_prefix(root.as_str()).ok_or_else(|| {
        ObservationError::Unsupported(format!(
            "cgroup {path} is outside the visible cgroup2 mount root {root}"
        ))
    })?;
    let mut directory = mount.join(relative.trim_start_matches('/'));
    loop {
        let max = directory.join("memory.max");
        if max.exists() {
            let value = read(&max)?;
            let value = value.trim();
            if value != "max" {
                let limit_bytes = value.parse::<u64>().map_err(|_| {
                    ObservationError::Failed(format!("{} contains {value}", max.display()))
                })?;
                let current = directory.join("memory.current");
                let used = read(&current)?;
                let used_bytes = used.trim().parse::<u64>().map_err(|_| {
                    ObservationError::Failed(format!("{} is invalid", current.display()))
                })?;
                let cgroup = directory
                    .strip_prefix(&mount)
                    .expect("cgroup walk stays inside its mount")
                    .display()
                    .to_string();
                limits.push(ProcessMemoryLimit {
                    kind: ProcessLimitKind::CgroupV2 {
                        cgroup: format!("/{cgroup}"),
                    },
                    limit_bytes,
                    used_bytes,
                });
            }
        }
        if directory == mount {
            break;
        }
        directory = directory
            .parent()
            .expect("cgroup walk stays inside its mount")
            .to_path_buf();
    }
    // `cgroup.type` exists only on non-root cgroups: a visible top that has
    // it is a namespace or bind-mount root with hidden ancestors.
    Ok(if mount.join("cgroup.type").exists() {
        LimitVisibility::CgroupAncestorsHidden
    } else {
        LimitVisibility::Complete
    })
}

/// The cgroup2 mount point and the hierarchy path mounted there.
fn cgroup2_mount() -> Result<(PathBuf, String), ObservationError> {
    let mountinfo = read("/proc/self/mountinfo")?;
    mountinfo
        .lines()
        .find_map(|line| {
            let (mount, filesystem) = line.split_once(" - ")?;
            if filesystem.split_whitespace().next()? != "cgroup2" {
                return None;
            }
            let mut fields = mount.split_whitespace();
            let root = fields.nth(3)?;
            let point = fields.next()?;
            Some((PathBuf::from(unescape(point)), unescape(root)))
        })
        .ok_or_else(|| {
            ObservationError::Unsupported("this process's cgroup2 hierarchy is not mounted".into())
        })
}

/// mountinfo escapes space, tab, newline and backslash as octal.
fn unescape(field: &str) -> String {
    field
        .replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
}
