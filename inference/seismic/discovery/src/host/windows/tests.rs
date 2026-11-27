use super::super::{beyond, ColdPages, Limit, PressureLevel, ProcessLimitKind, Supply};
use super::limits::job_limits;
use super::memory::{windows_floor, Counters};
use super::*;
use seismic_topology::{flat_nodes, CacheSize, HostNode, PciLocation};
use windows_sys::Win32::System::JobObjects::{
    JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_PROCESS_MEMORY, JOB_OBJECT_LIMIT_WORKINGSET,
};

const GB: u64 = 1 << 30;
const MB: u64 = 1 << 20;

/// A Windows machine's fixture for one case: 64 GiB, 40 GiB of it
/// available, a commit room of 100 GiB, no job, two memory nodes, a
/// process of 3 GiB private commit and 2 GiB working set.
struct Machine {
    status: MEMORYSTATUSEX,
    counters: Option<Counters>,
    nodes: Vec<Option<u64>>,
    job: Option<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>,
    process: PROCESS_MEMORY_COUNTERS,
    records: Vec<(LOGICAL_PROCESSOR_RELATIONSHIP, topology::ProcessorRecords)>,
    groups: Option<Vec<u16>>,
    affinity: Option<usize>,
    brand: Option<String>,
    smbios: Option<Vec<u8>>,
    parallelism: NonZeroUsize,
    build: u32,
}

impl Machine {
    fn new() -> Self {
        Self {
            status: MEMORYSTATUSEX {
                ullTotalPhys: 64 * GB,
                ullAvailPhys: 40 * GB,
                ullTotalPageFile: 120 * GB,
                ullAvailPageFile: 100 * GB,
                ..Default::default()
            },
            counters: Some(Counters {
                free_zero: 10 * GB,
                standby_core: 2 * GB,
                standby_reclaimable: 28 * GB,
            }),
            nodes: vec![Some(6 * GB), Some(4 * GB)],
            job: None,
            process: PROCESS_MEMORY_COUNTERS {
                PagefileUsage: 3 * GB as usize,
                WorkingSetSize: 2 * GB as usize,
                ..Default::default()
            },
            records: Vec::new(),
            groups: None,
            affinity: None,
            brand: None,
            smbios: None,
            parallelism: NonZeroUsize::new(4).unwrap(),
            build: 26100,
        }
    }

    fn in_job(
        mut self,
        flags: u32,
        edit: impl FnOnce(&mut JOBOBJECT_EXTENDED_LIMIT_INFORMATION),
    ) -> Self {
        let mut job = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        job.BasicLimitInformation.LimitFlags = flags;
        edit(&mut job);
        self.job = Some(job);
        self
    }
}

impl Sys for Machine {
    fn memory_status(&self) -> MEMORYSTATUSEX {
        self.status
    }
    fn os_version(&self) -> OSVERSIONINFOW {
        OSVERSIONINFOW {
            dwMajorVersion: 10,
            dwBuildNumber: self.build,
            ..unsafe { std::mem::zeroed() }
        }
    }
    fn cpu_identity(&self) -> String {
        "cpu".into()
    }
    fn cpu_brand(&self) -> Option<String> {
        self.brand.clone()
    }
    fn processor_records(&self, relationship: LOGICAL_PROCESSOR_RELATIONSHIP) -> Option<topology::ProcessorRecords> {
        self.records
            .iter()
            .find(|(rel, _)| *rel == relationship)
            .map(|(_, records)| records.clone())
    }
    fn highest_node(&self) -> u16 {
        (self.nodes.len() - 1) as u16
    }
    fn numa_available(&self, node: u16) -> Option<u64> {
        self.nodes[usize::from(node)]
    }
    fn process_groups(&self) -> Option<Vec<u16>> {
        self.groups.clone()
    }
    fn affinity_mask(&self) -> Option<usize> {
        self.affinity
    }
    fn counters(&self) -> Option<Counters> {
        self.counters
    }
    fn memory_counters(&self) -> PROCESS_MEMORY_COUNTERS {
        self.process
    }
    fn low_memory(&self) -> bool {
        false
    }
    fn job(&self) -> Option<JOBOBJECT_EXTENDED_LIMIT_INFORMATION> {
        self.job
    }
    fn smbios_table(&self) -> Option<Vec<u8>> {
        self.smbios.clone()
    }
    fn page(&self) -> u64 {
        4096
    }
    fn parallelism(&self) -> NonZeroUsize {
        self.parallelism
    }
}

fn supply(free: u64, floor: u64, clean: u64) -> Supply {
    Supply {
        free_beyond_target: beyond(free, floor),
        clean_cache: clean,
        cold_reclaimable: 0,
    }
}

/// The performance counters split free+zero from reclaimable standby while
/// keeping `Standby Cache Core` protected and reserving Windows' safety floor;
/// Windows reports no harm because no unprivileged counter separates
/// reclaimed-page refaults from cold file reads.
#[test]
fn the_counters_split_free_from_standby() {
    let sample = sample(&Machine::new());
    let floor = windows_floor(64 * GB);
    assert_eq!(floor, 512 * MB);
    assert_eq!(
        (sample.total, sample.supply, sample.cold, sample.harm),
        (
            64 * GB,
            supply(10 * GB, floor, 28 * GB),
            ColdPages {
                machine: 0,
                own: None
            },
            None,
        )
    );
}

/// Exhausting system commit below the safety floor signals pressure even
/// before `LowMemoryResourceNotification` triggers.
#[test]
fn tight_commit_signals_pressure_proactively() {
    let mut tight = Machine::new();
    let floor = windows_floor(64 * GB);
    tight.status.ullAvailPageFile = floor;
    assert_eq!(sample(&tight).pressure, Some(PressureLevel::Warn));
    tight.status.ullAvailPageFile = floor / 2;
    assert_eq!(sample(&tight).pressure, Some(PressureLevel::Critical));
}

/// Windows never misreports `PagefileUsage - PrivateWorkingSetSize` as trimmed
/// host RAM: WDDM backs `cuMemAlloc` VRAM in `PagefileUsage`, which would
/// otherwise report gigabytes of fake trimmed host memory.
#[test]
fn wddm_pagefile_commit_never_reports_fake_own_reclaimed() {
    let mut machine = Machine::new();
    machine.process.PagefileUsage = 16 * GB as usize;
    machine.process.WorkingSetSize = 2 * GB as usize;
    assert_eq!(sample(&machine).own_reclaimed, None);
}

/// Without the performance counters (an account outside Performance
/// Monitor Users), the free and zero lists are the nodes' available
/// bytes summed, the standby lists the rest of `ullAvailPhys`, less the
/// host safety floor, and no harm is counted.
#[test]
fn without_the_counters_free_is_the_nodes_and_standby_the_rest() {
    let floor = windows_floor(64 * GB);
    let taken = sample(&Machine {
        counters: None,
        ..Machine::new()
    });
    assert_eq!(
        (taken.total, taken.supply, taken.harm),
        (64 * GB, supply(10 * GB, floor, 30 * GB), None)
    );
    // One node's available bytes include its standby lists: all free.
    let one_node = Machine {
        counters: None,
        nodes: vec![Some(40 * GB)],
        ..Machine::new()
    };
    assert_eq!(sample(&one_node).supply, supply(40 * GB, floor, 0));
    // A number naming no node adds nothing.
    let sparse = Machine {
        counters: None,
        nodes: vec![Some(5 * GB), None, Some(2 * GB)],
        ..Machine::new()
    };
    assert_eq!(sample(&sparse).supply, supply(7 * GB, floor, 33 * GB));
}

/// Outside a job, no limit: the host's total and supply.
#[test]
fn outside_a_job_there_is_no_limit() {
    let machine = Machine::new();
    assert!(job_limits(&machine, &machine.status).is_empty());
    assert_eq!(sample(&machine).total, 64 * GB);
}

/// A job-wide commit limit (a process-isolated container's `--memory`)
/// is the scope's total, its room the commit the job has left
/// (`ullTotalPageFile − ullAvailPageFile` used of it), never an unknown.
#[test]
fn a_job_commit_limit_bounds_the_scope() {
    let mut machine = Machine::new()
        .in_job(JOB_OBJECT_LIMIT_JOB_MEMORY, |job| job.JobMemoryLimit = 8 * GB as usize);
    machine.status.ullTotalPageFile = 8 * GB;
    machine.status.ullAvailPageFile = 2 * GB;
    assert_eq!(
        job_limits(&machine, &machine.status),
        [Limit {
            kind: ProcessLimitKind::WindowsJobCommit,
            limit_bytes: 8 * GB,
            used_bytes: 6 * GB,
            reclaimable_bytes: 0,
        }]
    );
    let sample = sample(&machine);
    assert_eq!((sample.total, sample.supply.total()), (8 * GB, 2 * GB));
}

/// A job's per-process commit limit binds against this process's
/// private commit.
#[test]
fn a_job_process_limit_binds_against_private_commit() {
    let machine = Machine::new().in_job(JOB_OBJECT_LIMIT_PROCESS_MEMORY, |job| {
        job.ProcessMemoryLimit = 4 * GB as usize
    });
    let sample = sample(&machine);
    assert_eq!((sample.total, sample.supply), (4 * GB, supply(GB, 0, 0)));
}

/// A job's working-set limit is a physical scope against this process's
/// working set.
#[test]
fn a_job_working_set_limit_is_a_physical_scope() {
    let machine = Machine::new().in_job(JOB_OBJECT_LIMIT_WORKINGSET, |job| {
        job.BasicLimitInformation.MaximumWorkingSetSize = 5 * GB as usize
    });
    assert_eq!(
        job_limits(&machine, &machine.status),
        [Limit {
            kind: ProcessLimitKind::WindowsJobWorkingSet,
            limit_bytes: 5 * GB,
            used_bytes: 2 * GB,
            reclaimable_bytes: 0,
        }]
    );
    let sample = sample(&machine);
    assert_eq!((sample.total, sample.supply), (5 * GB, supply(3 * GB, 0, 0)));
}

/// Without the processor records, the flat topology of the quota's
/// CPUs, and `CacheSize::UNREPORTED` for cache.
#[test]
fn without_the_records_the_processors_are_flat() {
    let machine = Machine::new();
    let flat = processors(&machine);
    assert_eq!(flat.nodes()[0].cores.len(), 4);
    assert_eq!(flat.nodes(), flat_nodes(flat.allowed()));
    assert_eq!(last_level_cache(&machine), CacheSize::UNREPORTED);
    assert_eq!(
        device_node(
            &machine,
            PciLocation {
                domain: 0,
                bus: 0,
                device: 0
            }
        ),
        HostNode::Any
    );
    assert_eq!(
        cpu_model(&machine),
        format!("CPU ({})", std::env::consts::ARCH)
    );
    let named = Machine {
        brand: Some("AMD Ryzen 9 9950X".into()),
        ..Machine::new()
    };
    assert_eq!(cpu_model(&named), "AMD Ryzen 9 9950X");
}

/// On Windows 10 22H2 (build 19045), which fails `RelationNumaNodeEx`, the
/// nodes come from `RelationNumaNode` (whose records there carry
/// `GroupCount == 0`), with the real cores and their threads.
#[test]
fn before_numa_node_ex_the_nodes_come_from_numa_node() {
    use windows_sys::Win32::System::SystemInformation::{
        GROUP_AFFINITY, NUMA_NODE_RELATIONSHIP, NUMA_NODE_RELATIONSHIP_0, PROCESSOR_RELATIONSHIP,
        RelationNumaNode, RelationProcessorCore, SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };

    fn pack(items: &[SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX]) -> topology::ProcessorRecords {
        let byte_len = std::mem::size_of_val(items);
        let mut words = vec![0u64; byte_len.div_ceil(8)];
        // SAFETY: `items` is a contiguous slice of `byte_len` initialized bytes and `words`
        // holds at least `byte_len` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                items.as_ptr().cast::<u8>(),
                words.as_mut_ptr().cast::<u8>(),
                byte_len,
            );
        }
        topology::ProcessorRecords::new(words, byte_len)
    }

    let record_size = std::mem::size_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>() as u32;
    let core_record = |mask: usize| {
        // SAFETY: `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX` is a C ABI struct/union where
        // all-zeroes is a valid bit pattern, ensuring union padding bytes are initialized.
        let mut record: SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX = unsafe { std::mem::zeroed() };
        record.Relationship = RelationProcessorCore;
        record.Size = record_size;
        record.Anonymous.Processor = PROCESSOR_RELATIONSHIP {
            Flags: 1,
            EfficiencyClass: 0,
            Reserved: [0; 20],
            GroupCount: 1,
            GroupMask: [GROUP_AFFINITY {
                Mask: mask,
                Group: 0,
                Reserved: [0; 3],
            }],
        };
        record
    };
    let numa_record = {
        // SAFETY: all-zeroes initializes all union padding bytes before populating `NumaNode`.
        let mut record: SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX = unsafe { std::mem::zeroed() };
        record.Relationship = RelationNumaNode;
        record.Size = record_size;
        record.Anonymous.NumaNode = NUMA_NODE_RELATIONSHIP {
            NodeNumber: 0,
            Reserved: [0; 18],
            // Windows 10 `RelationNumaNode` sets `GroupCount = 0` with one `GroupMask`.
            GroupCount: 0,
            Anonymous: NUMA_NODE_RELATIONSHIP_0 {
                GroupMask: GROUP_AFFINITY {
                    Mask: 0b1111,
                    Group: 0,
                    Reserved: [0; 3],
                },
            },
        };
        record
    };

    let machine = Machine {
        nodes: vec![Some(16 * GB)],
        records: vec![
            (RelationProcessorCore, pack(&[core_record(0b0011), core_record(0b1100)])),
            (RelationNumaNode, pack(&[numa_record])),
        ],
        groups: Some(vec![0]),
        affinity: Some(0b1111),
        build: 19045,
        ..Machine::new()
    };

    let topo = processors(&machine);
    assert_eq!(topo.nodes().len(), 1);
    assert_eq!(topo.nodes()[0].id, 0);
    assert!(topo.nodes()[0].memory);
    assert_eq!(topo.nodes()[0].cores.len(), 2);
    assert_eq!(topo.nodes()[0].cores[0].threads, vec![0, 1]);
    assert_eq!(topo.nodes()[0].cores[1].threads, vec![2, 3]);
    assert_eq!(topo.allowed().iter().copied().collect::<Vec<_>>(), vec![0, 1, 2, 3]);
}

/// `RelationNumaNodeEx` only on builds that know it ([`topology::NUMA_NODE_EX_BUILD`]).
#[test]
fn numa_nodes_come_from_the_relationship_the_build_knows() {
    use windows_sys::Win32::System::SystemInformation::{RelationNumaNode, RelationNumaNodeEx};
    let on = |build| topology::numa_relationship(&Machine { build, ..Machine::new() });
    assert_eq!(on(19045), RelationNumaNode);
    assert_eq!(on(topology::NUMA_NODE_EX_BUILD - 1), RelationNumaNode);
    assert_eq!(on(topology::NUMA_NODE_EX_BUILD), RelationNumaNodeEx);
    assert_eq!(on(26100), RelationNumaNodeEx);
}
