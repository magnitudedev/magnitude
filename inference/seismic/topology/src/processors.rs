//! The host's processors as the OS reports them: its memory nodes, each
//! node's cores with their hardware threads and kind, the CPUs this process
//! may run on, and each GPU's host NUMA node affinity.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::num::NonZeroUsize;

/// The host's processors: its memory nodes, each node's cores with their
/// hardware threads and kind, the CPUs this process may run on, and the
/// thread quota it may keep busy at once.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Processors {
    nodes: Vec<MemoryNode>,
    /// The CPUs this process may run on.
    allowed: BTreeSet<usize>,
    /// The CPUs this process may keep busy at once ([`Self::quota`]).
    quota: NonZeroUsize,
}

/// One memory node: its number, the OS's (what a device's [`HostNode`]
/// names), whether memory is attached to it, and its cores.
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

impl Processors {
    /// `nodes`, of which the process may run on `allowed` and keep `quota`
    /// busy at once: every core's threads ascending and non-empty, no CPU
    /// twice.
    pub fn new(
        nodes: Vec<MemoryNode>,
        allowed: BTreeSet<usize>,
        quota: NonZeroUsize,
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
        Ok(Self {
            nodes,
            allowed,
            quota,
        })
    }

    /// A flat topology of `cpus` where no OS topology API answered: one memory
    /// node with memory attached, each CPU its own performance core.
    pub fn flat(cpus: BTreeSet<usize>, quota: NonZeroUsize) -> Self {
        let nodes = flat_nodes(&cpus);
        Self {
            nodes,
            allowed: cpus,
            quota,
        }
    }

    pub fn nodes(&self) -> &[MemoryNode] {
        &self.nodes
    }

    /// Total physical cores across all nodes in the topology.
    pub fn chip_cores(&self) -> usize {
        self.nodes.iter().map(|node| node.cores.len()).sum()
    }

    /// Total logical CPUs (hardware threads) across all nodes in the topology.
    pub fn logical_cpus(&self) -> usize {
        self.nodes
            .iter()
            .flat_map(|node| &node.cores)
            .map(|core| core.threads.len())
            .sum()
    }

    /// The threads of `core` the process may run on.
    fn allowed_threads(&self, core: &Core) -> Vec<usize> {
        core.threads
            .iter()
            .copied()
            .filter(|cpu| self.allowed.contains(cpu))
            .collect()
    }

    /// The performance cores the process may run on (on any of their threads).
    pub fn performance_cores(&self) -> usize {
        self.nodes
            .iter()
            .flat_map(|node| &node.cores)
            .filter(|core| core.performance && !self.allowed_threads(core).is_empty())
            .count()
    }

    /// The CPUs this process may keep busy at once.
    pub fn quota(&self) -> NonZeroUsize {
        self.quota
    }

    /// The performance cores the process may keep busy at once: those it may
    /// run on ([`Self::performance_cores`]) under its [`Self::quota`].
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
    /// (ascending); a node with none is left out.
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

/// Build the single-node flat layout for `cpus`.
pub fn flat_nodes(cpus: &BTreeSet<usize>) -> Vec<MemoryNode> {
    vec![MemoryNode {
        id: 0,
        memory: true,
        cores: cpus
            .iter()
            .map(|&cpu| Core {
                threads: vec![cpu],
                performance: true,
            })
            .collect(),
    }]
}

/// The host NUMA memory node a physical device is attached to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HostNode {
    /// Device is attached to host memory node `n` ([`MemoryNode::id`]).
    Node(usize),
    /// No specific NUMA node affinity (single-node host, integrated device,
    /// OS without PCI NUMA reporting, or PCI NUMA node `-1` / unreadable).
    #[default]
    Any,
}

/// A PCI function's bus location (domain, bus, device; function 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PciLocation {
    pub domain: u32,
    pub bus: u32,
    pub device: u32,
}
