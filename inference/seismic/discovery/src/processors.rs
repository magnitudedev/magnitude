//! The host's processors as the OS reports them: its memory nodes, each
//! node's cores with their hardware threads and kind, and the CPUs this
//! process may run on; and the placement of a thread on one CPU.
//!
//! Every fact is the OS's, read when [`Processors::discover`] runs (the
//! process's affinity is its own and can change, so it is no part of a
//! topology revision); nothing is presumed where the OS reports nothing,
//! and a report that cannot be read or parsed is an error.

use std::collections::BTreeSet;

/// The host's processors (module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Processors {
    nodes: Vec<MemoryNode>,
    /// The CPUs this process may run on.
    allowed: BTreeSet<usize>,
    by: ProcessorsReportedBy,
}

/// One memory node: its number, the OS's (what a device's node names),
/// whether memory is attached to it, and its cores.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryNode {
    pub id: usize,
    pub memory: bool,
    pub cores: Vec<Core>,
}

/// One core: its hardware threads (CPUs, by the numbers a thread is placed
/// by, ascending) and whether it is a performance core.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Core {
    pub threads: Vec<usize>,
    pub performance: bool,
}

/// The API that reported the processors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessorsReportedBy {
    /// Linux sysfs: `/sys/devices/system/node` (`online`, `has_memory`,
    /// each node's `cpulist`), `/sys/devices/system/cpu` (`online`, each
    /// CPU's `topology/thread_siblings_list`) and a hybrid part's
    /// performance CPUs (`/sys/devices/cpu_core/cpus`); the affinity is
    /// `sched_getaffinity`.
    LinuxSysfs,
    /// macOS `sysctl`: one memory node; each performance level's
    /// (`hw.nperflevels`, `hw.perflevel<n>.{physicalcpu,logicalcpu}`, level
    /// 0 the performance cores) or, with no levels, `hw.{physical,logical}cpu`.
    /// macOS numbers no CPU a thread is placed by: a CPU is its ordinal, and
    /// the process may run on every one.
    MacOsSysctl,
    /// Windows `GetLogicalProcessorInformationEx` (`RelationNumaNode`,
    /// `RelationProcessorCore` with its `EfficiencyClass`, the highest the
    /// performance cores), `GetNumaAvailableMemoryNodeEx` and the process's
    /// group affinity; a CPU is its group's 64 times its group plus its bit.
    WindowsProcessorInformation,
}

impl Processors {
    /// The processors as the OS reports them now.
    pub fn discover() -> Result<Self, String> {
        crate::host::processors()
    }

    /// `nodes`, of which the process may run on `allowed`, as `by` reported
    /// them: every core's threads ascending and non-empty, no CPU twice.
    /// Public so a consumer's host tests state a recorded machine.
    pub fn new(
        nodes: Vec<MemoryNode>,
        allowed: BTreeSet<usize>,
        by: ProcessorsReportedBy,
    ) -> Result<Self, String> {
        let mut seen = BTreeSet::new();
        for core in nodes.iter().flat_map(|node| &node.cores) {
            if core.threads.is_empty() || !core.threads.is_sorted() {
                return Err(format!("a core of threads {:?}", core.threads));
            }
            if let Some(cpu) = core.threads.iter().find(|&&cpu| !seen.insert(cpu)) {
                return Err(format!("CPU {cpu} reported in two cores"));
            }
        }
        Ok(Self { nodes, allowed, by })
    }

    pub fn nodes(&self) -> &[MemoryNode] {
        &self.nodes
    }

    pub fn by(&self) -> ProcessorsReportedBy {
        self.by
    }

    /// The threads of `core` the process may run on.
    fn allowed_threads(&self, core: &Core) -> Vec<usize> {
        core.threads
            .iter()
            .copied()
            .filter(|cpu| self.allowed.contains(cpu))
            .collect()
    }

    /// The performance cores the process may run on (on any of their
    /// threads): what a pool of one thread a core can occupy, where a second
    /// hardware thread on a saturated core, or a slower core, delays every
    /// barrier.
    pub fn performance_cores(&self) -> usize {
        self.nodes
            .iter()
            .flat_map(|node| &node.cores)
            .filter(|core| core.performance && !self.allowed_threads(core).is_empty())
            .count()
    }

    /// The CPUs this process may run on.
    pub fn allowed(&self) -> &BTreeSet<usize> {
        &self.allowed
    }

    /// Each memory node with memory attached, its performance cores the
    /// process may run on, each as its threads the process may run on
    /// (ascending: the first is where one thread a core runs); a node with
    /// none is left out. A node of CPUs alone (the 2990WX's two dies without
    /// memory) reads every byte across the fabric.
    pub fn memory_cores(&self) -> Vec<(usize, Vec<Vec<usize>>)> {
        self.nodes
            .iter()
            .filter(|node| node.memory)
            .map(|node| {
                let cores = node
                    .cores
                    .iter()
                    .filter(|core| core.performance)
                    .map(|core| self.allowed_threads(core));
                (
                    node.id,
                    cores
                        .filter(|threads| !threads.is_empty())
                        .collect::<Vec<_>>(),
                )
            })
            .filter(|(_, cores)| !cores.is_empty())
            .collect()
    }
}

/// Place the calling thread on `cpus` (numbers [`Processors`] reports, at
/// least one): it runs on those alone. macOS places no thread on a core it
/// names (its scheduler places every thread), so there the call places
/// nothing; elsewhere an OS that refuses is an error.
pub fn pin(cpus: &[usize]) -> Result<(), String> {
    if cpus.is_empty() {
        return Err("a thread placed on no CPU".into());
    }
    crate::host::pin(cpus)
}

/// Whether a [`ProcessPlacement`] is held: one at a time in a process.
static PLACED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The process's placement ([`place_process`]): while it is held, every
/// thread of the process but those [`pin`] placed runs where it placed
/// them; as it drops, every such thread is placed back on the CPUs the
/// process could run on as it was claimed, the threads started meanwhile
/// among them. One is held at a time in a process: its holder owns every
/// thread's CPUs, and what the process may run on reads as its placement
/// until it drops.
#[must_use = "the process's placement lasts as long as it is held"]
pub struct ProcessPlacement {
    every: Vec<usize>,
}

impl Drop for ProcessPlacement {
    fn drop(&mut self) {
        crate::host::place_process(&self.every).expect("the process's threads back on every CPU it could run on");
        PLACED.store(false, std::sync::atomic::Ordering::Release);
    }
}

/// Place every thread of the process now on `cpus` (numbers [`Processors`]
/// reports, at least one) but those [`pin`] placed, which keep their CPUs,
/// until the returned placement drops; a thread it starts later inherits
/// its starter's placement, so threads a placed thread starts run there
/// too, and one may then be placed elsewhere with [`pin`]. Linux places
/// each of the process's threads (`/proc/self/task`, `sched_setaffinity`),
/// again until a pass finds none it had not placed (one started meanwhile
/// by a thread not yet placed); macOS and Windows place nothing here
/// (macOS's scheduler places every thread; Windows' process mask would
/// bound [`pin`]).
///
/// # Panics
/// While another placement is held: two owners of every thread's CPUs in
/// one process would each size its work from what the other left it, and
/// restore what it found.
pub fn place_process(cpus: &[usize]) -> Result<ProcessPlacement, String> {
    if cpus.is_empty() {
        return Err("a process placed on no CPU".into());
    }
    assert!(
        !PLACED.swap(true, std::sync::atomic::Ordering::AcqRel),
        "the process is placed already: one placement owns every thread's CPUs at a time (a second serving executor in one process)"
    );
    let placed = Processors::discover()
        .map(|processors| processors.allowed().iter().copied().collect::<Vec<_>>())
        .and_then(|every| crate::host::place_process(cpus).map(|()| ProcessPlacement { every }));
    if placed.is_err() {
        PLACED.store(false, std::sync::atomic::Ordering::Release);
    }
    placed
}

/// The memory node a device is attached to, as the OS reports its bus
/// location's: `Node(n)` names [`MemoryNode::id`]; `Unknown` where the OS
/// reports none (Linux's `-1`, a host without NUMA), where the OS here reads
/// no device's node (macOS, Windows), or where the device reports no bus
/// location (a CPU device, Metal, Vulkan).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceNode {
    Node(usize),
    Unknown,
}

/// A PCI function's bus location (domain, bus, device; function 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PciLocation {
    pub domain: u32,
    pub bus: u32,
    pub device: u32,
}

/// The memory node of the device at `location` ([`DeviceNode`]): Linux's
/// `/sys/bus/pci/devices/<location>.0/numa_node`.
pub fn device_node(location: PciLocation) -> Result<DeviceNode, String> {
    crate::host::device_node(location)
}

/// A Linux PCI `numa_node` file's contents: `-1` is no node.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn parse_device_node(text: &str) -> Result<DeviceNode, String> {
    match text.trim().parse::<i64>() {
        Ok(-1) => Ok(DeviceNode::Unknown),
        Ok(node) => usize::try_from(node)
            .map(DeviceNode::Node)
            .map_err(|_| format!("{text:?} is no node")),
        Err(_) => Err(format!("{text:?} is no node")),
    }
}

/// A Linux CPU or node list (`0-3,8,10-11`), every part parsed or an
/// error.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn parse_list(list: &str) -> Result<Vec<usize>, String> {
    let number = |text: &str| {
        text.parse::<usize>()
            .map_err(|_| format!("{list:?} is no CPU list"))
    };
    let mut values = Vec::new();
    for part in list.trim().split(',').filter(|part| !part.is_empty()) {
        match part.split_once('-') {
            Some((low, high)) => {
                let (low, high) = (number(low)?, number(high)?);
                if low > high {
                    return Err(format!("{list:?} is no CPU list"));
                }
                values.extend(low..=high);
            }
            None => values.push(number(part)?),
        }
    }
    Ok(values)
}

/// The nodes sysfs under `devices` (`/sys/devices`) reports: each online
/// node (one node of every online CPU where the kernel has no nodes: no
/// `system/node`), whether it has memory, and its online CPUs grouped into
/// cores by their thread siblings, each a performance core unless the part
/// lists its performance CPUs (`cpu_core/cpus`) without it.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn sysfs_nodes(devices: &std::path::Path) -> Result<Vec<MemoryNode>, String> {
    let read = |path: std::path::PathBuf| {
        std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))
    };
    let list = |path: std::path::PathBuf| {
        parse_list(&read(path.clone())?).map_err(|error| format!("{}: {error}", path.display()))
    };
    let online = list(devices.join("system/cpu/online"))?;
    let performance = match std::fs::read_to_string(devices.join("cpu_core/cpus")) {
        Ok(cpus) => Some(parse_list(&cpus)?),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(format!(
                "{}: {error}",
                devices.join("cpu_core/cpus").display()
            ))
        }
    };
    let node = devices.join("system/node");
    let nodes: Vec<(usize, bool, Vec<usize>)> = if node.exists() {
        let memory = list(node.join("has_memory"))?;
        list(node.join("online"))?
            .into_iter()
            .map(|index| {
                Ok((
                    index,
                    memory.contains(&index),
                    list(node.join(format!("node{index}/cpulist")))?,
                ))
            })
            .collect::<Result<_, String>>()?
    } else {
        vec![(0, true, online.clone())]
    };
    nodes
        .into_iter()
        .map(|(id, memory, cpus)| {
            let mut cores: Vec<Core> = Vec::new();
            for cpu in cpus.into_iter().filter(|cpu| online.contains(cpu)) {
                let mut threads = list(
                    devices.join(format!("system/cpu/cpu{cpu}/topology/thread_siblings_list")),
                )?;
                threads.retain(|thread| online.contains(thread));
                threads.sort_unstable();
                if !cores.iter().any(|core| core.threads == threads) {
                    let performance = performance
                        .as_ref()
                        .is_none_or(|performance| performance.contains(&cpu));
                    cores.push(Core {
                        threads,
                        performance,
                    });
                }
            }
            Ok(MemoryNode { id, memory, cores })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sysfs tree under a fresh directory of `files` (path, contents).
    fn sysfs(name: &str, files: &[(&str, &str)]) -> std::path::PathBuf {
        let root =
            std::env::temp_dir().join(format!("seismic-processors-{}-{name}", std::process::id()));
        for (path, contents) in files {
            let path = root.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, contents).unwrap();
        }
        root
    }

    #[test]
    fn a_device_node_of_minus_one_is_unknown() {
        assert_eq!(parse_device_node("1\n"), Ok(DeviceNode::Node(1)));
        assert_eq!(parse_device_node("-1\n"), Ok(DeviceNode::Unknown));
        assert!(parse_device_node("x").is_err());
    }

    #[test]
    fn lists_parse_strictly() {
        assert_eq!(parse_list("0-3,8,10-11\n"), Ok(vec![0, 1, 2, 3, 8, 10, 11]));
        assert_eq!(parse_list(""), Ok(vec![]));
        assert!(
            parse_list("0-x").is_err(),
            "a bad bound is an error, never zero"
        );
        assert!(parse_list("3-1").is_err());
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
            Processors::new(nodes, (1..8).collect(), ProcessorsReportedBy::LinuxSysfs).unwrap();
        assert_eq!(
            processors.memory_cores(),
            vec![(0, vec![vec![4], vec![1, 5]])]
        );
        assert_eq!(processors.performance_cores(), 4);
        std::fs::remove_dir_all(root).unwrap();
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
            Processors::new(nodes, (0..4).collect(), ProcessorsReportedBy::LinuxSysfs).unwrap();
        assert_eq!(
            (processors.performance_cores(), processors.memory_cores()),
            (1, vec![(0, vec![vec![0, 1]])])
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn an_unreadable_report_is_an_error() {
        let root = sysfs(
            "unreadable",
            &[
                ("system/cpu/online", "0-1"),
                ("system/cpu/cpu0/topology/thread_siblings_list", "0"),
            ],
        );
        assert!(
            sysfs_nodes(&root).is_err(),
            "CPU 1's siblings are unreported"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn this_host_reports_its_processors() {
        let processors = Processors::discover().unwrap();
        assert!(processors.performance_cores() >= 1);
        assert!(!processors.memory_cores().is_empty());
        let cpus = processors
            .memory_cores()
            .into_iter()
            .flat_map(|(_, cores)| cores)
            .flatten()
            .collect::<Vec<_>>();
        pin(&cpus[..1]).unwrap();
        pin(&cpus).unwrap();
        assert!(pin(&[]).is_err());
    }
}
