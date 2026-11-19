//! The host's processors and the placement of threads on them, a PCI
//! device's memory node, the machine's firmware names, and the PSI files a
//! pressure trigger watches.

use super::Facts;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The PSI files a pressure trigger watches (`crate::pressure`), the ones a
/// sample's pressure level reads: the machine's (`/proc/pressure/memory`)
/// and this process's cgroup v2's `memory.pressure`
/// ([`super::limits::Cgroups::unified`]) where it has one. Each file whose
/// trigger the kernel refuses is left unwatched alone.
pub(crate) fn pressure_files() -> Vec<PathBuf> {
    let cgroups = super::limits::Cgroups::of_process(&super::Os);
    std::iter::once(PathBuf::from("/proc/pressure/memory"))
        .chain(cgroups.unified().map(|cgroup| cgroup.join("memory.pressure")))
        .collect()
}

/// The host's processors (`crate::processors`):
/// - its nodes: sysfs's ([`crate::processors::sysfs_nodes`]); where sysfs
///   reports none, one node holding the CPUs the process may run on
///   ([`crate::processors::flat`]);
/// - the CPUs the process may run on: its affinity (`sched_getaffinity`);
///   where the OS states none, as many as its quota, numbered from 0;
/// - its quota ([`Facts::parallelism`], `seismic_threads::quota`): the
///   affinity under the cgroup's CPU quota (`cpu.max`), which a container's
///   `--cpus` sets apart from its affinity (20.48 CPUs of a 128-CPU
///   affinity).
///
/// # Panics
/// Where the OS reports processors that contradict themselves, or an
/// affinity of no CPU.
pub(crate) fn processors(facts: &impl Facts) -> crate::Processors {
    let parallelism = facts.parallelism();
    let allowed = facts
        .affinity()
        .unwrap_or_else(|| (0..parallelism.get()).collect());
    let (nodes, by) = match crate::processors::sysfs_nodes(facts) {
        Some(nodes) => (nodes, crate::ProcessorsReportedBy::LinuxSysfs),
        None => (crate::processors::flat(&allowed), crate::ProcessorsReportedBy::Flat),
    };
    crate::Processors::new(nodes, allowed, parallelism, by).unwrap_or_else(|error| panic!("invariant violated: the processors the OS reports: {error}"))
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
        PINNED.lock().unwrap_or_else(std::sync::PoisonError::into_inner).remove(&self.0);
    }
}

thread_local! {
    static PIN: std::cell::OnceCell<Pinned> = const { std::cell::OnceCell::new() };
}

/// `sched_setaffinity` of every thread of the process (`/proc/self/task`)
/// but those [`pin`] placed to `cpus`, pass after pass until one finds no
/// thread it had not placed: a thread started during a pass by one not yet
/// placed is placed by the next.
pub(crate) fn place_process(cpus: &[usize]) -> Result<(), String> {
    let set = cpu_set(cpus)?;
    let mut placed = PINNED.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
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
pub(crate) fn pin(cpus: &[usize]) -> Result<(), String> {
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
                    PINNED.lock().unwrap_or_else(std::sync::PoisonError::into_inner).insert(tid);
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

/// The PCI function's `numa_node` (`crate::processors::device_node`): that
/// node where it is one (0 or more); no preference
/// ([`crate::DeviceNode::Unknown`]) where it is `-1` (firmware giving no
/// proximity, a single-node host), absent (WSL2's GPU, on a paravirtual bus)
/// or refused.
///
/// # Panics
/// Where the `numa_node` present names no node.
pub(crate) fn device_node(facts: &impl Facts, location: crate::PciLocation) -> crate::DeviceNode {
    let crate::PciLocation { domain, bus, device } = location;
    let path = format!("/sys/bus/pci/devices/{domain:04x}:{bus:02x}:{device:02x}.0/numa_node");
    facts.text(Path::new(&path)).map_or(crate::DeviceNode::Unknown, |text| {
        crate::processors::parse_device_node(&text).unwrap_or_else(|error| panic!("invariant violated: {path}: {error}"))
    })
}

/// The machine's names (`crate::machine`): DMI's system and chassis fields
/// (`/sys/devices/virtual/dmi/id`), and a board's device-tree model
/// (`/proc/device-tree/model`) as its name where DMI names no product. A field
/// missing or refused ([`crate::host::read`]) reads as nothing.
pub(crate) fn machine() -> crate::machine::Firmware {
    let read = |path: &str| crate::host::read(path, std::fs::read(path)).unwrap_or_default();
    let dmi = |field: &str| read(&format!("/sys/devices/virtual/dmi/id/{field}"));
    let model = dmi("product_name");
    let name = match crate::machine::label(&model) {
        Some(_) => Vec::new(),
        None => read("/proc/device-tree/model"),
    };
    crate::machine::Firmware {
        manufacturer: dmi("sys_vendor"),
        model,
        family: dmi("product_family"),
        version: dmi("product_version"),
        chassis_type: String::from_utf8_lossy(&dmi("chassis_type")).trim().parse().ok(),
        name,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::Tree;
    use std::num::NonZeroUsize;

    /// A machine's facts with nothing in its tree, under a root of its own.
    fn tree(name: &str) -> Tree {
        let root = std::env::temp_dir().join(format!("seismic-topology-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        Tree::new(root)
    }

    /// Without sysfs's nodes, the processors are one node holding the CPUs
    /// the process may run on, each its own serving core.
    #[test]
    fn no_sysfs_nodes_is_one_node_of_the_allowed_cpus() {
        let mut facts = tree("flat");
        facts.affinity = Some(BTreeSet::from([2, 3, 5]));
        let processors = processors(&facts);
        assert_eq!(processors.by(), crate::ProcessorsReportedBy::Flat);
        assert_eq!(processors.nodes(), crate::processors::flat(&BTreeSet::from([2, 3, 5])));
        assert_eq!(processors.allowed(), &BTreeSet::from([2, 3, 5]));
    }

    /// Without an affinity, the process may run on `available_parallelism`'s
    /// count of CPUs, numbered from 0.
    #[test]
    fn no_affinity_is_available_parallelism() {
        let mut facts = tree("parallelism");
        facts.affinity = None;
        facts.parallelism = NonZeroUsize::new(6).unwrap();
        let processors = processors(&facts);
        assert_eq!(processors.allowed(), &(0..6).collect::<BTreeSet<_>>());
        assert_eq!(processors.cores().unwrap().get(), 6);
    }

    /// A cgroup's CPU quota below the affinity caps the pool: rtx3090-b's
    /// container runs on an affinity of 128 CPUs under `cpu.max` 2048000
    /// 100000 (20.48 CPUs), which `available_parallelism` reads as 20.
    #[test]
    fn a_cpu_quota_below_the_affinity_caps_the_pool() {
        let mut facts = tree("quota");
        facts.affinity = Some((0..128).collect());
        facts.parallelism = NonZeroUsize::new(20).unwrap();
        let processors = processors(&facts);
        assert_eq!(processors.performance_cores(), 128);
        assert_eq!(processors.cores().unwrap().get(), 20);
    }

    /// A PCI `numa_node` of 0 or more is that node; `-1`, absent or refused
    /// is no preference.
    #[test]
    fn a_numa_node_of_minus_one_absent_or_refused_is_no_preference() {
        let mut facts = tree("numa-node");
        let at = |bus| crate::PciLocation { domain: 0, bus, device: 0 };
        facts.file("/sys/bus/pci/devices/0000:41:00.0/numa_node", "1\n");
        facts.file("/sys/bus/pci/devices/0000:42:00.0/numa_node", "-1\n");
        facts.file("/sys/bus/pci/devices/0000:43:00.0/numa_node", "0\n");
        facts.refused.push(PathBuf::from("/sys/bus/pci/devices/0000:43:00.0"));
        assert_eq!(device_node(&facts, at(0x41)), crate::DeviceNode::Node(1));
        assert_eq!(device_node(&facts, at(0x42)), crate::DeviceNode::Unknown);
        assert_eq!(device_node(&facts, at(0x43)), crate::DeviceNode::Unknown, "refused");
        assert_eq!(device_node(&facts, at(0x44)), crate::DeviceNode::Unknown, "absent");
        std::fs::remove_dir_all(&facts.root).unwrap();
    }
}
