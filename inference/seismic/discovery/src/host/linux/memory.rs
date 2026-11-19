//! Host RAM: its total and its sample at one instant, every figure from its
//! sources in order, or its documented value where none answers.

use super::limits::{self, Cgroups};
use super::Facts;
use super::super::{beyond, binding, Counted, Harm, HostSample, Lag, OwnCold, OwnReclaimed, PressureLevel, ProcessLimitKind, Scope, Supply};
use seismic_topology::{Reported, ReportedBy};
use std::path::Path;
use std::time::{Duration, Instant};

/// Host RAM's total: `/proc/meminfo`'s `MemTotal`, else `sysinfo(2)`'s
/// `totalram`, the same kernel count (`si_meminfo`), which always answers.
pub(super) fn host_total(facts: &impl Facts) -> Reported<u64> {
    Reported::new(total_of(facts, facts.text(Path::new("/proc/meminfo")).as_deref()), ReportedBy::LinuxMemTotal)
}

/// `meminfo`'s `MemTotal`, else `sysinfo(2)`'s `totalram` ([`host_total`]).
fn total_of(facts: &impl Facts, meminfo: Option<&str>) -> u64 {
    meminfo_field(meminfo, "MemTotal:").unwrap_or_else(|| facts.sysinfo().0)
}

/// Host RAM now, each figure from its sources in order:
/// - `MemTotal`, `MemFree`: `/proc/meminfo`, else `sysinfo(2)`.
/// - The floor the kernel keeps free: [`reclaim_floor`].
/// - Clean cache (`Active(file) + Inactive(file) − Dirty − Writeback`), cold
///   anonymous pages (`Inactive(anon)`) and free swap (`SwapFree`):
///   `/proc/meminfo`, 0 where it has none.
/// - Harm: `/proc/vmstat`'s refaults with this process's own cgroup's
///   ([`Cgroups::own`]) read just after; none where either is missing.
/// - Pressure: [`pressure`].
/// - The counters' lag: [`counters_lag`].
/// - Own cold and zswapped pages: the own cgroup's `memory.stat`
///   (unobserved and 0 where it has none); own swapped bytes:
///   `/proc/self/status`'s `VmSwap`, none where it has none.
///
/// The whole is bound by every limit on this process
/// ([`limits::process_limits`], [`super::super::binding`]). The cgroups
/// are found again at every sample, so a process moved to another cgroup
/// is read in the one it is in.
pub(crate) fn sample(facts: &impl Facts) -> HostSample {
    let at = Instant::now();
    let page = page(facts);
    let meminfo = facts.text(Path::new("/proc/meminfo"));
    let field = |name: &str| meminfo_field(meminfo.as_deref(), name);
    let total = total_of(facts, meminfo.as_deref());
    let free = field("MemFree:").unwrap_or_else(|| facts.sysinfo().1);
    let cgroups = Cgroups::of_process(facts);
    let limits = limits::process_limits(facts, &cgroups);
    let cold = field("Inactive(anon):").unwrap_or(0);
    let clean_cache = clean_cache(["Active(file):", "Inactive(file):", "Dirty:", "Writeback:"].map(field));
    let cold_reclaimable = cold.min(field("SwapFree:").unwrap_or(0));
    let pressure = pressure(facts, &cgroups);
    let machine = facts.text(Path::new("/proc/vmstat")).and_then(|vmstat| refault_count(&vmstat));
    // Read after the machine's, so an event between the two counts as this
    // process's alone: never another program's.
    let own = cgroups.own(facts).and_then(|cgroup| {
        let path = cgroup.join("memory.stat");
        facts.text(&path).map(|stat| (path, stat))
    });
    let own_field = |key: &str| own.as_ref().and_then(|(path, stat)| stat_field(path, stat, key));
    let harm = machine.zip(own.as_ref().and_then(|(_, stat)| refault_count(stat))).map(|(machine, own)| Harm { machine, own });
    let own_cold = own_field("inactive_anon").map_or(OwnCold::Unobserved, OwnCold::Bytes);
    let own_reclaimed = facts
        .text(Path::new("/proc/self/status"))
        .and_then(|status| kib_field(&status, "VmSwap:"))
        .map(|bytes| OwnReclaimed { bytes, stored: own_field("zswap").unwrap_or(0), exact: false });
    let host = Scope {
        total,
        supply: Supply { free_beyond_target: beyond(free, reclaim_floor(facts, total, page)), clean_cache, cold_reclaimable },
    };
    let binding = binding(host, &limits);
    // The counters show every event their folds had passed as the reading
    // began.
    let counted_at = Instant::now();
    let counted = Counted { at: counted_at, lag: counters_lag(facts, own.is_some()) + counted_at.duration_since(at) };
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
            // The host's supply, a node's and the address space show a
            // growth in the call; a cgroup's clean cache, read from its
            // `memory.stat`, within memcg's flush, and only where that
            // cgroup's supply binds.
            growth: match binding.supplier.map(|index| &limits[index].kind) {
                Some(ProcessLimitKind::CgroupV1 { .. } | ProcessLimitKind::CgroupV2 { .. }) => FLUSH_TIME,
                _ => Duration::ZERO,
            },
        },
    }
}

/// memcg's periodic statistics flush (`FLUSH_TIME`, `mm/memcontrol.c`,
/// Linux 5.15 on): a cgroup's `memory.stat` shows an event within it. A
/// read may flush sooner, past a count of pending updates, never later.
const FLUSH_TIME: Duration = Duration::from_secs(2);

/// `sysconf(_SC_PAGESIZE)`.
///
/// # Panics
/// Where it does not answer: it answers on every Linux.
pub(super) fn page(facts: &impl Facts) -> u64 {
    facts.page().unwrap_or_else(|| panic!("invariant violated: sysconf(_SC_PAGESIZE) does not answer"))
}

/// How late the counters show an event ([`Counted::lag`]): the per-CPU
/// counts `/proc` sums fold within `vm.stat_interval` ([`stat_interval`]),
/// their timer rounded up to the next whole second
/// (`round_jiffies_relative`, `mm/vmstat.c`); a cgroup's, where the
/// counters read one (`cgroup`), within memcg's flush.
fn counters_lag(facts: &impl Facts, cgroup: bool) -> Duration {
    let fold = stat_interval(facts) + Duration::from_secs(1);
    match cgroup {
        true => fold.max(FLUSH_TIME),
        false => fold,
    }
}

/// `vm.stat_interval` (`/proc/sys/vm/stat_interval`): how often the kernel
/// folds the per-CPU counts into the machine's; 1 s, the kernel's default,
/// where it is missing.
///
/// # Panics
/// Where it does not parse.
fn stat_interval(facts: &impl Facts) -> Duration {
    let path = "/proc/sys/vm/stat_interval";
    facts.text(Path::new(path)).map_or(Duration::from_secs(1), |value| {
        Duration::from_secs(value.trim().parse().unwrap_or_else(|error| panic!("invariant violated: {path} reads {value:?}: {error}")))
    })
}

/// The floor the kernel keeps free, in bytes: kswapd wakes to reclaim once
/// free pages fall below the zones' low watermarks, so free pages above
/// them are taken without anyone's reclaim. The sum of every zone's `low`
/// in `/proc/zoneinfo`, in pages of `page` bytes; where it is missing or
/// states no watermark, the kernel's own formula from the host's `total`
/// bytes ([`watermark_formula`]).
fn reclaim_floor(facts: &impl Facts, total: u64, page: u64) -> u64 {
    match facts.text(Path::new("/proc/zoneinfo")).map(|zoneinfo| zones_low(&zoneinfo, |_| true)) {
        Some(low) if low > 0 => low * page,
        _ => watermark_formula(facts, total),
    }
}

/// The floor the kernel keeps free on the memory nodes `nodes`, of
/// `nodes_total` bytes: the low watermarks of their zones
/// (`/proc/zoneinfo`'s `Node <N>, zone <name>` sections); where those are
/// missing, the nodes' share of the host's ([`watermark_formula`]), each
/// node's zones set in proportion to its managed pages.
pub(super) fn nodes_floor(facts: &impl Facts, nodes: &[usize], nodes_total: u64) -> u64 {
    let page = page(facts);
    match facts.text(Path::new("/proc/zoneinfo")).map(|zoneinfo| zones_low(&zoneinfo, |node| nodes.contains(&node))) {
        Some(low) if low > 0 => low * page,
        _ => {
            let total = host_total(facts).value;
            let share = u128::from(watermark_formula(facts, total)) * u128::from(nodes_total) / u128::from(total.max(1));
            u64::try_from(share).unwrap_or(u64::MAX)
        }
    }
}

/// The zones' low watermarks summed, in bytes, as `mm/page_alloc.c` sets
/// them for a host of `total` bytes.
///
/// `__setup_per_zone_wmarks` splits `min_free_kbytes` (`pages_min`) over
/// the zones in proportion to their managed pages (`min_z = pages_min ×
/// managed_z / lowmem`), then sets each zone's `low = min_z + max(min_z /
/// 4, managed_z × watermark_scale_factor / 10000)`. As `min_z` is
/// proportional to `managed_z`, the same term wins in every zone, so over
/// the zones `Σ low = min_free + max(min_free / 4, total × scale / 10000)`
/// with `total` the managed pages, which `MemTotal` counts.
///
/// The tunables, each from its file else the kernel's default:
/// `/proc/sys/vm/min_free_kbytes`, else `calculate_min_free_kbytes`'s
/// `int_sqrt(lowmem_kbytes × 16)` clamped to [128, 262144] KiB, `lowmem`
/// taken as `total` (on 64-bit hosts every zone is lowmem; the kernel's
/// count less the high watermarks differs by under a percent);
/// `/proc/sys/vm/watermark_scale_factor`, else 10.
///
/// # Panics
/// Where a tunable does not parse.
fn watermark_formula(facts: &impl Facts, total: u64) -> u64 {
    let tunable = |path: &str| {
        facts.text(Path::new(path)).map(|value| {
            value.trim().parse::<u64>().unwrap_or_else(|error| panic!("invariant violated: {path} reads {value:?}: {error}"))
        })
    };
    let min_free = tunable("/proc/sys/vm/min_free_kbytes").unwrap_or_else(|| (total / 1024 * 16).isqrt().clamp(128, 262_144)) * 1024;
    let scale = tunable("/proc/sys/vm/watermark_scale_factor").unwrap_or(10);
    let scaled = u64::try_from(u128::from(total) * u128::from(scale) / 10_000).unwrap_or(u64::MAX);
    min_free.saturating_add((min_free / 4).max(scaled))
}

/// The `low` of every zone of a `/proc/zoneinfo` text on a node `keep`
/// keeps, in pages.
///
/// # Panics
/// Where a node header or a `low` does not parse.
fn zones_low(zoneinfo: &str, keep: impl Fn(usize) -> bool) -> u64 {
    let mut kept = false;
    zoneinfo
        .lines()
        .filter_map(|line| {
            if let Some(header) = line.strip_prefix("Node ") {
                let node = header.split(',').next().and_then(|node| node.trim().parse::<usize>().ok());
                kept = keep(node.unwrap_or_else(|| panic!("invariant violated: /proc/zoneinfo reads {line:?}")));
                return None;
            }
            let mut fields = line.split_whitespace();
            (kept && fields.next() == Some("low")).then(|| {
                fields
                    .next()
                    .and_then(|value| value.parse::<u64>().ok())
                    .unwrap_or_else(|| panic!("invariant violated: /proc/zoneinfo reads {line:?}"))
            })
        })
        .sum()
}

/// The pressure level: the higher of the machine's PSI
/// (`/proc/pressure/memory`) and this process's cgroup v2's
/// (`memory.pressure`), each where it answers; none where neither does (a
/// kernel built without PSI, booted `psi=0`, a cgroup with
/// `cgroup.pressure` 0, a sandbox's `/proc`).
fn pressure(facts: &impl Facts, cgroups: &Cgroups) -> Option<PressureLevel> {
    let machine = psi_level(facts, Path::new("/proc/pressure/memory"));
    let cgroup = cgroups.unified().and_then(|directory| psi_level(facts, &directory.join("memory.pressure")));
    machine.max(cgroup)
}

/// The pressure level a PSI file `path` reports: critical where some task
/// stalled with every task waiting (`full avg10`), warn where some task
/// stalled (`some avg10`), over the last ten seconds; none where it is
/// missing.
///
/// # Panics
/// Where the file reads no level.
fn psi_level(facts: &impl Facts, path: &Path) -> Option<PressureLevel> {
    let text = facts.text(path)?;
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

/// The refaults a `/proc/vmstat` or `memory.stat` text counts, under the
/// same names in both: anonymous and file since Linux 5.9, the one count
/// before; none where it counts neither.
///
/// # Panics
/// Where it counts one of anonymous and file without the other, or a count
/// does not parse.
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

/// Clean file cache from its four counts, active and inactive file pages
/// less dirty and writeback ones (which may lie on either list): 0 where
/// any is missing, which then establishes none. Active cache counts too: a
/// charge at a cgroup's limit reclaims it (a v1 docker cgroup gave up all
/// of its 58.7 GB at an unchanged allocation rate, with no failed charge or
/// OOM kill); anonymous pages, tmpfs and kernel allocations never.
pub(super) fn clean_cache([active, inactive, dirty, writeback]: [Option<u64>; 4]) -> u64 {
    match (active, inactive, dirty, writeback) {
        (Some(active), Some(inactive), Some(dirty), Some(writeback)) => {
            active.saturating_add(inactive).saturating_sub(dirty.saturating_add(writeback))
        }
        _ => 0,
    }
}

/// The field `name` of a `/proc/meminfo` text, none where the text or the
/// field is missing.
fn meminfo_field(meminfo: Option<&str>, name: &str) -> Option<u64> {
    kib_field(meminfo?, name)
}

/// The `name` (`Key:`) field of a `<Key>: <n> kB` text, in bytes.
pub(super) fn kib_field(text: &str, name: &str) -> Option<u64> {
    text.lines()
        .find_map(|line| line.strip_prefix(name))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .and_then(|kib| kib.checked_mul(1024))
}

/// The value of `key` in the `memory.stat` text `stat` of `path`.
///
/// # Panics
/// Where it does not parse.
pub(super) fn stat_field(path: &Path, stat: &str, key: &str) -> Option<u64> {
    stat.lines().find_map(|line| {
        let mut fields = line.split_whitespace();
        (fields.next() == Some(key)).then(|| parse_bytes(path, fields.next().unwrap_or("")))
    })
}

/// The figure `value` the file `path` holds.
///
/// # Panics
/// Where it does not parse.
pub(super) fn parse_bytes(path: &Path, value: &str) -> u64 {
    value.parse::<u64>().unwrap_or_else(|_| panic!("invariant violated: {} contains {value:?}", path.display()))
}
