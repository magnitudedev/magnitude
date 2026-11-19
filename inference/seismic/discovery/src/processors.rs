//! The host's processors as the OS reports them: its memory nodes, each
//! node's cores with their hardware threads and kind, and the CPUs this
//! process may run on; and the placement of a thread on one CPU.
//!
//! Every fact is the OS's, read when [`Processors::discover`] runs (the
//! process's affinity and quota are its own and can change, so they are no
//! part of a topology revision). Where the OS's report is absent the
//! processors are the flat topology of the CPUs the process may run on
//! ([`ProcessorsReportedBy::Flat`]), never one presumed; a report present
//! that does not parse, or contradicts itself, panics.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

/// The host's processors (module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Processors {
    nodes: Vec<MemoryNode>,
    /// The CPUs this process may run on.
    allowed: BTreeSet<usize>,
    /// The CPUs this process may keep busy at once ([`Self::quota`]).
    quota: NonZeroUsize,
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
    /// `sched_getaffinity`, the quota `available_parallelism`.
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
    /// None of the above could be read (a sandbox's `/sys`, an OS refusing
    /// the calls): one memory node of every CPU the process may run on
    /// (where the OS states none, `available_parallelism`'s count of them,
    /// numbered from 0; where neither answers, CPU 0), each its own core and
    /// every one serving ([`flat`]). Nothing here is the OS's topology; nodes, siblings and
    /// core kinds are unknown.
    Flat,
}

impl Processors {
    /// The processors as the OS reports them now: each fact from its
    /// sources in order, or its documented value where none answers.
    pub fn discover() -> Self {
        crate::host::processors(&crate::host::Os)
    }

    /// `nodes`, of which the process may run on `allowed` and keep `quota`
    /// busy at once, as `by` reported them: every core's threads ascending
    /// and non-empty, no CPU twice. Public so a consumer's host tests state
    /// a recorded machine.
    pub fn new(
        nodes: Vec<MemoryNode>,
        allowed: BTreeSet<usize>,
        quota: NonZeroUsize,
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
        Ok(Self { nodes, allowed, quota, by })
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

    /// The CPUs this process may keep busy at once, however many it may
    /// run on: `available_parallelism`, which on Linux is its affinity under
    /// its cgroup's CPU quota (`cpu.max`: a container's `--cpus` caps it
    /// without narrowing the affinity); where that is unanswered, the CPUs
    /// it may run on. The one cap every pool of the process's CPUs is sized
    /// under.
    pub fn quota(&self) -> NonZeroUsize {
        self.quota
    }

    /// The performance cores the process may keep busy at once: those it
    /// may run on ([`Self::performance_cores`]) under its [`Self::quota`].
    /// The machine's cores as one process sees them: the CPU device's pool.
    pub fn cores(&self) -> Result<NonZeroUsize, String> {
        NonZeroUsize::new(self.performance_cores())
            .map(|cores| cores.min(self.quota))
            .ok_or_else(|| "the process may run on no performance core".to_owned())
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
    let every = Processors::discover().allowed().iter().copied().collect::<Vec<_>>();
    let placed = crate::host::place_process(cpus).map(|()| ProcessPlacement { every });
    if placed.is_err() {
        PLACED.store(false, std::sync::atomic::Ordering::Release);
    }
    placed
}

/// The memory node a device is attached to, as the OS reports its bus
/// location's: `Node(n)` names [`MemoryNode::id`]; `Unknown`, no
/// preference, where the OS reports none (Linux's `-1`, a host without
/// NUMA, or no `numa_node` at all or one refused, as WSL2 gives its GPU),
/// where the OS here reads no device's node (macOS, Windows), or where the
/// device reports no bus location (a CPU device, Metal, Vulkan).
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
/// `/sys/bus/pci/devices/<location>.0/numa_node`, or no preference.
pub fn device_node(location: PciLocation) -> DeviceNode {
    crate::host::device_node(&crate::host::Os, location)
}

/// The topology of `cpus` where the OS reports none
/// ([`ProcessorsReportedBy::Flat`]): one node with memory, each CPU its own
/// core, every one serving.
pub(crate) fn flat(cpus: &BTreeSet<usize>) -> Vec<MemoryNode> {
    vec![MemoryNode { id: 0, memory: true, cores: cpus.iter().map(|&cpu| Core { threads: vec![cpu], performance: true }).collect() }]
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
#[cfg(target_os = "linux")]
pub(crate) fn sysfs_nodes(facts: &impl crate::host::Facts) -> Option<Vec<MemoryNode>> {
    use std::path::Path;
    let list = |path: &str| -> Option<Vec<usize>> {
        let path = Path::new("/sys/devices").join(path);
        let text = facts.text(&path)?;
        Some(parse_list(&text).unwrap_or_else(|error| panic!("invariant violated: {}: {error}", path.display())))
    };
    let online = list("system/cpu/online")?;
    let performance = list("cpu_core/cpus");
    let nodes: Vec<(usize, bool, Vec<usize>)> = match list("system/node/online") {
        Some(indices) => {
            let memory = list("system/node/has_memory")?;
            indices
                .into_iter()
                .map(|index| Some((index, memory.contains(&index), list(&format!("system/node/node{index}/cpulist"))?)))
                .collect::<Option<_>>()?
        }
        None => vec![(0, true, online.clone())],
    };
    nodes
        .into_iter()
        .map(|(id, memory, cpus)| {
            let mut cores: Vec<Core> = Vec::new();
            for cpu in cpus.into_iter().filter(|cpu| online.contains(cpu)) {
                let mut threads = list(&format!("system/cpu/cpu{cpu}/topology/thread_siblings_list"))?;
                threads.retain(|thread| online.contains(thread));
                threads.sort_unstable();
                if !cores.iter().any(|core| core.threads == threads) {
                    let performance = performance.as_ref().is_none_or(|performance| performance.contains(&cpu));
                    cores.push(Core { threads, performance });
                }
            }
            Some(MemoryNode { id, memory, cores })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sysfs tree of `files` (path under `/sys/devices`, contents) under a
    /// fresh directory, of a machine of four CPUs ([`crate::host::Tree::new`]).
    #[cfg(target_os = "linux")]
    fn sysfs(name: &str, files: &[(&str, &str)]) -> crate::host::Tree {
        let root =
            std::env::temp_dir().join(format!("seismic-processors-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let tree = crate::host::Tree::new(root);
        for (path, contents) in files {
            tree.file(&format!("/sys/devices/{path}"), contents);
        }
        tree
    }

    /// A quota of `cpus`.
    fn quota(cpus: usize) -> NonZeroUsize {
        NonZeroUsize::new(cpus).unwrap()
    }

    /// The pool is the performance cores the process may run on, capped by
    /// its quota where that is fewer, never raised by it.
    #[test]
    fn the_pool_is_the_performance_cores_under_the_quota() {
        let nodes = flat(&(0..128).collect());
        let pool = |cap| Processors::new(nodes.clone(), (0..128).collect(), quota(cap), ProcessorsReportedBy::Flat).unwrap().cores().unwrap().get();
        assert_eq!(pool(20), 20);
        assert_eq!(pool(128), 128);
        let few = Processors::new(nodes, (0..8).collect(), quota(128), ProcessorsReportedBy::Flat).unwrap();
        assert_eq!(few.cores().unwrap().get(), 8);
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

    #[cfg(target_os = "linux")]
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
            Processors::new(nodes, (1..8).collect(), quota(7), ProcessorsReportedBy::LinuxSysfs).unwrap();
        assert_eq!(
            processors.memory_cores(),
            vec![(0, vec![vec![4], vec![1, 5]])]
        );
        assert_eq!(processors.performance_cores(), 4);
        std::fs::remove_dir_all(&root.root).unwrap();
    }

    #[cfg(target_os = "linux")]
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
            Processors::new(nodes, (0..4).collect(), quota(4), ProcessorsReportedBy::LinuxSysfs).unwrap();
        assert_eq!(
            (processors.performance_cores(), processors.memory_cores()),
            (1, vec![(0, vec![vec![0, 1]])])
        );
        std::fs::remove_dir_all(&root.root).unwrap();
    }

    /// A topology sysfs reports in part is none: the process's CPUs are
    /// then flat, each its own core.
    #[cfg(target_os = "linux")]
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
        let processors = crate::host::processors(&root);
        assert_eq!(processors.by(), ProcessorsReportedBy::Flat);
        assert_eq!(processors.performance_cores(), processors.allowed().len());
        std::fs::remove_dir_all(&root.root).unwrap();
    }

    #[test]
    fn this_host_reports_its_processors() {
        let processors = Processors::discover();
        assert!(processors.performance_cores() >= 1);
        let cores = processors.cores().unwrap().get();
        assert_eq!(cores, processors.performance_cores().min(processors.quota().get()));
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
