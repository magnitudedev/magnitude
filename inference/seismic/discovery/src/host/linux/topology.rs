//! The host's processors and the placement of threads on them, a PCI
//! device's memory node, the machine's firmware names, and the PSI files a
//! pressure trigger watches.

use super::super::resolve;
use super::Sys;
use seismic_topology::{
    flat_nodes, Core, Firmware, HostNode, MemoryNode, PciLocation, Processors,
};
use std::path::{Path, PathBuf};

/// The PSI files a pressure trigger watches (`crate::pressure`), the ones a
/// sample's pressure level reads: the machine's (`/proc/pressure/memory`)
/// and this process's cgroup v2's `memory.pressure`
/// ([`super::limits::Cgroups::unified`]) where it has one. Each file whose
/// trigger the kernel refuses is left unwatched alone.
pub(crate) fn pressure_files(sys: &impl Sys) -> Vec<PathBuf> {
    let cgroups = super::limits::Cgroups::of_process(sys);
    std::iter::once(PathBuf::from("/proc/pressure/memory"))
        .chain(cgroups.unified().map(|cgroup| cgroup.join("memory.pressure")))
        .collect()
}

/// A Linux CPU or node range list (`0-3,8,10-11`), every part parsed or an
/// error.
pub(super) fn parse_range_list(list: &str) -> Result<Vec<usize>, String> {
    let number = |text: &str| {
        text.parse::<usize>()
            .map_err(|_| format!("{list:?} is no range list"))
    };
    let mut values = Vec::new();
    for part in list.trim().split(',').filter(|part| !part.is_empty()) {
        match part.split_once('-') {
            Some((low, high)) => {
                let (low, high) = (number(low)?, number(high)?);
                if low > high {
                    return Err(format!("{list:?} is no range list"));
                }
                values.extend(low..=high);
            }
            None => values.push(number(part)?),
        }
    }
    Ok(values)
}

/// A Linux PCI `numa_node` file's contents: `-1` is no preference ([`HostNode::Any`]).
fn parse_device_node(text: &str) -> Result<HostNode, String> {
    match text.trim().parse::<i64>() {
        Ok(-1) => Ok(HostNode::Any),
        Ok(node) => usize::try_from(node)
            .map(HostNode::Node)
            .map_err(|_| format!("{text:?} is no node")),
        Err(_) => Err(format!("{text:?} is no node")),
    }
}

/// The nodes sysfs reports (`/sys/devices`): each online node (one node
/// of every online CPU where the kernel has no nodes: no `system/node`),
/// whether it has memory, and its online CPUs grouped into cores by their
/// thread siblings. A hybrid part lists its performance CPUs
/// (`cpu_core/cpus`); where it lists none, the kernel states no core kinds
/// and every core serves. None where any of it is absent (a sandbox's
/// `/sys`, a kernel without CPU topology): a part of the topology is no
/// topology.
///
/// # Panics
/// Where a list present does not parse.
fn sysfs_nodes(sys: &impl Sys) -> Option<Vec<MemoryNode>> {
    let list = |path: &str| -> Option<Vec<usize>> {
        let path = Path::new("/sys/devices").join(path);
        let text = sys.text(&path)?;
        Some(
            parse_range_list(&text)
                .unwrap_or_else(|error| panic!("invariant violated: {}: {error}", path.display())),
        )
    };
    let online = list("system/cpu/online")?;
    let performance = list("cpu_core/cpus");
    let nodes: Vec<(usize, bool, Vec<usize>)> = match list("system/node/online") {
        Some(indices) => {
            let memory = list("system/node/has_memory")?;
            indices
                .into_iter()
                .map(|index| {
                    Some((
                        index,
                        memory.contains(&index),
                        list(&format!("system/node/node{index}/cpulist"))?,
                    ))
                })
                .collect::<Option<_>>()?
        }
        None => vec![(0, true, online.clone())],
    };
    nodes
        .into_iter()
        .map(|(id, memory, cpus)| {
            let mut cores: Vec<Core> = Vec::new();
            for cpu in cpus.into_iter().filter(|cpu| online.contains(cpu)) {
                let mut threads =
                    list(&format!("system/cpu/cpu{cpu}/topology/thread_siblings_list"))?;
                threads.retain(|thread| online.contains(thread));
                threads.sort_unstable();
                if !cores.iter().any(|core| core.threads == threads) {
                    let performance = performance
                        .as_ref()
                        .is_none_or(|performance| performance.contains(&cpu));
                    cores.push(Core { threads, performance });
                }
            }
            Some(MemoryNode { id, memory, cores })
        })
        .collect()
}

/// The host's processors (`crate::processors`):
/// - its nodes: sysfs's ([`sysfs_nodes`]); where sysfs
///   reports none, one node holding the CPUs the process may run on
///   ([`flat_nodes`]);
/// - the CPUs the process may run on: its affinity (`sched_getaffinity`);
///   where the OS states none, as many as its quota, numbered from 0;
/// - its quota ([`Sys::parallelism`], `seismic_threads::quota`): the
///   affinity under the cgroup's CPU quota (`cpu.max`), which a container's
///   `--cpus` sets apart from its affinity (20.48 CPUs of a 128-CPU
///   affinity).
///
/// # Panics
/// Where the OS reports processors that contradict themselves, or an
/// affinity of no CPU.
pub(crate) fn processors(sys: &impl Sys) -> Processors {
    let parallelism = sys.parallelism();
    let allowed = resolve(
        &[((), &|| sys.affinity())],
        (),
        || (0..parallelism.get()).collect(),
    )
    .0;
    let nodes = resolve(&[((), &|| sysfs_nodes(sys))], (), || flat_nodes(&allowed)).0;
    Processors::new(nodes, allowed, parallelism)
        .unwrap_or_else(|error| panic!("invariant violated: the processors the OS reports: {error}"))
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
        PINNED
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.0);
    }
}

thread_local! {
    static PIN: std::cell::OnceCell<Pinned> = const { std::cell::OnceCell::new() };
}

/// `sched_setaffinity` of every thread of the process (`/proc/self/task`)
/// but those [`pin`] placed to `cpus`, pass after pass until one finds no
/// thread it had not placed: a thread started during a pass by one not yet
/// placed is placed by the next. Serialized with [`pin`] under [`PINNED`]
/// so a concurrent `pin` is never overwritten.
pub(crate) fn place_process(cpus: &[usize]) -> Result<(), String> {
    let set = cpu_set(cpus)?;
    let pinned = PINNED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut placed = pinned.clone();
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
/// then leaves where it is. Serialized with [`place_process`] under [`PINNED`].
pub(crate) fn pin(cpus: &[usize]) -> Result<(), String> {
    let set = cpu_set(cpus)?;
    let mut pinned = PINNED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // SAFETY: a CPU set, applied to the calling thread.
    let placed =
        unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) };
    match placed {
        0 => {
            // SAFETY: the calling thread's id.
            let tid = unsafe { libc::syscall(libc::SYS_gettid) } as libc::pid_t;
            pinned.insert(tid);
            drop(pinned);
            PIN.with(|pin| {
                pin.get_or_init(|| Pinned(tid));
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
/// node where it is one (0 or more); no preference ([`HostNode::Any`]) where
/// it is `-1` (firmware giving no proximity, a single-node host), absent
/// (WSL2's GPU, on a paravirtual bus) or refused.
///
/// # Panics
/// Where the `numa_node` present names no node.
pub(crate) fn device_node(sys: &impl Sys, location: PciLocation) -> HostNode {
    let PciLocation { domain, bus, device } = location;
    let path = format!("/sys/bus/pci/devices/{domain:04x}:{bus:02x}:{device:02x}.0/numa_node");
    resolve(
        &[((), &|| {
            let text = sys.text(Path::new(&path))?;
            Some(
                parse_device_node(&text)
                    .unwrap_or_else(|error| panic!("invariant violated: {path}: {error}")),
            )
        })],
        (),
        || HostNode::Any,
    )
    .0
}

/// The machine's names (`seismic_topology::SystemIdentity`): DMI's system and
/// chassis fields (`/sys/devices/virtual/dmi/id`), and a board's device-tree
/// model (`/proc/device-tree/model`) as its name where DMI names no product.
/// A field missing or refused reads as nothing.
pub(crate) fn firmware(sys: &impl Sys) -> Firmware {
    let read = |path: &str| resolve(&[((), &|| sys.bytes(Path::new(path)))], (), Vec::new).0;
    let dmi = |field: &str| read(&format!("/sys/devices/virtual/dmi/id/{field}"));
    let model = dmi("product_name");
    let name = match seismic_topology::label(&model) {
        Some(_) => Vec::new(),
        None => read("/proc/device-tree/model"),
    };
    Firmware {
        manufacturer: dmi("sys_vendor"),
        model,
        family: dmi("product_family"),
        version: dmi("product_version"),
        chassis_type: String::from_utf8_lossy(&dmi("chassis_type"))
            .trim()
            .parse()
            .ok(),
        name,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::Tree;
    use std::collections::BTreeSet;
    use std::num::NonZeroUsize;

    /// A machine's fixture with nothing in its tree, under a root of its own.
    fn tree(name: &str) -> Tree {
        let root = std::env::temp_dir().join(format!(
            "seismic-topology-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        Tree::new(root)
    }

    fn sysfs(name: &str, files: &[(&str, &str)]) -> Tree {
        let tree = tree(name);
        for (path, contents) in files {
            tree.file(&format!("/sys/devices/{path}"), contents);
        }
        tree
    }

    fn quota(cpus: usize) -> NonZeroUsize {
        NonZeroUsize::new(cpus).unwrap()
    }

    #[test]
    fn a_device_node_of_minus_one_is_any() {
        assert_eq!(parse_device_node("1\n"), Ok(HostNode::Node(1)));
        assert_eq!(parse_device_node("-1\n"), Ok(HostNode::Any));
        assert!(parse_device_node("x").is_err());
    }

    #[test]
    fn lists_parse_strictly() {
        assert_eq!(
            parse_range_list("0-3,8,10-11\n"),
            Ok(vec![0, 1, 2, 3, 8, 10, 11])
        );
        assert_eq!(parse_range_list(""), Ok(vec![]));
        assert!(
            parse_range_list("0-x").is_err(),
            "a bad bound is an error, never zero"
        );
        assert!(parse_range_list("3-1").is_err());
    }

    #[test]
    fn nodes_without_memory_and_second_threads_serve_no_core() {
        // Two nodes, the second without memory; two cores a node, each of
        // two threads (CPU n and n + 4).
        let mut files = vec![
            ("system/cpu/online", "0-7"),
            ("system/node/online", "0-1"),
            ("system/node/has_memory", "0"),
            ("system/node/node0/cpulist", "0-1,4-5"),
            ("system/node/node1/cpulist", "2-3,6-7"),
        ];
        let siblings = ["0,4", "1,5", "2,6", "3,7", "0,4", "1,5", "2,6", "3,7"];
        let paths = (0..8)
            .map(|cpu| format!("system/cpu/cpu{cpu}/topology/thread_siblings_list"))
            .collect::<Vec<_>>();
        files.extend(paths.iter().map(String::as_str).zip(siblings));
        let root = sysfs("numa", &files);
        let nodes = sysfs_nodes(&root).unwrap();
        let core = |threads: &[usize]| Core {
            threads: threads.to_vec(),
            performance: true,
        };
        assert_eq!(
            nodes,
            vec![
                MemoryNode {
                    id: 0,
                    memory: true,
                    cores: vec![core(&[0, 4]), core(&[1, 5])]
                },
                MemoryNode {
                    id: 1,
                    memory: false,
                    cores: vec![core(&[2, 6]), core(&[3, 7])]
                },
            ]
        );
        // The process may run on CPUs 1 to 7: core [0, 4] serves on 4.
        let processors =
            Processors::new(nodes, (1..8).collect(), quota(7))
                .unwrap();
        assert_eq!(
            processors.memory_cores(),
            vec![(0, vec![vec![4], vec![1, 5]])]
        );
        assert_eq!(processors.performance_cores(), 4);
        std::fs::remove_dir_all(&root.root).unwrap();
    }

    #[test]
    fn a_hybrid_part_counts_its_performance_cores_and_a_kernel_without_nodes_has_one() {
        let mut files = vec![("system/cpu/online", "0-3"), ("cpu_core/cpus", "0-1")];
        let paths = (0..4)
            .map(|cpu| format!("system/cpu/cpu{cpu}/topology/thread_siblings_list"))
            .collect::<Vec<_>>();
        files.extend(
            paths
                .iter()
                .map(String::as_str)
                .zip(["0-1", "0-1", "2", "3"]),
        );
        let root = sysfs("hybrid", &files);
        let nodes = sysfs_nodes(&root).unwrap();
        assert_eq!(
            nodes,
            vec![MemoryNode {
                id: 0,
                memory: true,
                cores: vec![
                    Core {
                        threads: vec![0, 1],
                        performance: true
                    },
                    Core {
                        threads: vec![2],
                        performance: false
                    },
                    Core {
                        threads: vec![3],
                        performance: false
                    },
                ],
            }]
        );
        let processors =
            Processors::new(nodes, (0..4).collect(), quota(4))
                .unwrap();
        assert_eq!(
            (processors.performance_cores(), processors.memory_cores()),
            (1, vec![(0, vec![vec![0, 1]])])
        );
        std::fs::remove_dir_all(&root.root).unwrap();
    }

    /// A topology sysfs reports in part is none: the process's CPUs are
    /// then flat, each its own core.
    #[test]
    fn a_partial_report_is_a_flat_topology() {
        let root = sysfs(
            "partial",
            &[
                ("system/cpu/online", "0-1"),
                ("system/cpu/cpu0/topology/thread_siblings_list", "0"),
            ],
        );
        assert_eq!(sysfs_nodes(&root), None, "CPU 1's siblings are unreported");
        let processors = processors(&root);
        assert_eq!(processors.nodes(), flat_nodes(processors.allowed()));
        assert_eq!(processors.performance_cores(), processors.allowed().len());
        std::fs::remove_dir_all(&root.root).unwrap();
    }

    /// Without sysfs's nodes, the processors are one node holding the CPUs
    /// the process may run on, each its own serving core.
    #[test]
    fn no_sysfs_nodes_is_one_node_of_the_allowed_cpus() {
        let mut sys = tree("flat");
        sys.affinity = Some(BTreeSet::from([2, 3, 5]));
        let processors = processors(&sys);
        assert_eq!(processors.nodes(), flat_nodes(&BTreeSet::from([2, 3, 5])));
        assert_eq!(processors.allowed(), &BTreeSet::from([2, 3, 5]));
    }

    /// Without an affinity, the process may run on `available_parallelism`'s
    /// count of CPUs, numbered from 0.
    #[test]
    fn no_affinity_is_available_parallelism() {
        let mut sys = tree("parallelism");
        sys.affinity = None;
        sys.parallelism = NonZeroUsize::new(6).unwrap();
        let processors = processors(&sys);
        assert_eq!(processors.allowed(), &(0..6).collect::<BTreeSet<_>>());
        assert_eq!(processors.cores().unwrap().get(), 6);
    }

    /// A cgroup's CPU quota below the affinity caps the pool: rtx3090-b's
    /// container runs on an affinity of 128 CPUs under `cpu.max` 2048000
    /// 100000 (20.48 CPUs), which `available_parallelism` reads as 20.
    #[test]
    fn a_cpu_quota_below_the_affinity_caps_the_pool() {
        let mut sys = tree("quota");
        sys.affinity = Some((0..128).collect());
        sys.parallelism = NonZeroUsize::new(20).unwrap();
        let processors = processors(&sys);
        assert_eq!(processors.performance_cores(), 128);
        assert_eq!(processors.cores().unwrap().get(), 20);
    }

    /// A PCI `numa_node` of 0 or more is that node; `-1`, absent or refused
    /// is no preference.
    #[test]
    fn a_numa_node_of_minus_one_absent_or_refused_is_no_preference() {
        let mut sys = tree("numa-node");
        let at = |bus| PciLocation {
            domain: 0,
            bus,
            device: 0,
        };
        sys.file("/sys/bus/pci/devices/0000:41:00.0/numa_node", "1\n");
        sys.file("/sys/bus/pci/devices/0000:42:00.0/numa_node", "-1\n");
        sys.file("/sys/bus/pci/devices/0000:43:00.0/numa_node", "0\n");
        sys.refused
            .push(PathBuf::from("/sys/bus/pci/devices/0000:43:00.0"));
        assert_eq!(device_node(&sys, at(0x41)), HostNode::Node(1));
        assert_eq!(device_node(&sys, at(0x42)), HostNode::Any);
        assert_eq!(device_node(&sys, at(0x43)), HostNode::Any, "refused");
        assert_eq!(device_node(&sys, at(0x44)), HostNode::Any, "absent");
        std::fs::remove_dir_all(&sys.root).unwrap();
    }

    #[test]
    fn firmware_reads_dmi_and_falls_back_to_device_tree() {
        let sys = tree("firmware");
        sys.file("/sys/devices/virtual/dmi/id/sys_vendor", "Lenovo\n");
        sys.file("/sys/devices/virtual/dmi/id/product_name", "21HD\n");
        sys.file("/sys/devices/virtual/dmi/id/chassis_type", "10\n");
        let fw = firmware(&sys);
        assert_eq!(fw.manufacturer, b"Lenovo\n");
        assert_eq!(fw.model, b"21HD\n");
        assert_eq!(fw.chassis_type, Some(10));
        assert!(fw.name.is_empty());
        std::fs::remove_dir_all(&sys.root).unwrap();
    }
}
