use super::super::{beyond, ColdPages, Limit, ProcessLimitKind, Supply};
use super::limits::job_limits;
use super::memory::Counters;
use super::*;
use seismic_topology::{flat_nodes, CacheSize, HostNode, PciLocation};
use windows_sys::Win32::System::JobObjects::{
    JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_PROCESS_MEMORY, JOB_OBJECT_LIMIT_WORKINGSET,
};

const GB: u64 = 1 << 30;

/// A Windows machine's fixture for one case: 64 GiB, 40 GiB of it
/// available, a commit room of 100 GiB, no job, two memory nodes, a
/// process of 3 GiB private commit and 2 GiB working set.
struct Machine {
    status: MEMORYSTATUSEX,
    counters: Option<Counters>,
    nodes: Vec<Option<u64>>,
    job: Option<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>,
    process: PROCESS_MEMORY_COUNTERS,
    records: Option<topology::ProcessorRecords>,
    brand: Option<String>,
    smbios: Option<Vec<u8>>,
    parallelism: NonZeroUsize,
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
                standby: 30 * GB,
            }),
            nodes: vec![Some(6 * GB), Some(4 * GB)],
            job: None,
            process: PROCESS_MEMORY_COUNTERS {
                PagefileUsage: 3 * GB as usize,
                WorkingSetSize: 2 * GB as usize,
                ..Default::default()
            },
            records: None,
            brand: None,
            smbios: None,
            parallelism: NonZeroUsize::new(4).unwrap(),
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
            dwBuildNumber: 26100,
            ..unsafe { std::mem::zeroed() }
        }
    }
    fn cpu_identity(&self) -> String {
        "cpu".into()
    }
    fn cpu_brand(&self) -> Option<String> {
        self.brand.clone()
    }
    fn processor_records(&self, _relationship: LOGICAL_PROCESSOR_RELATIONSHIP) -> Option<topology::ProcessorRecords> {
        self.records.clone()
    }
    fn highest_node(&self) -> u16 {
        (self.nodes.len() - 1) as u16
    }
    fn numa_available(&self, node: u16) -> Option<u64> {
        self.nodes[usize::from(node)]
    }
    fn process_groups(&self) -> Option<Vec<u16>> {
        None
    }
    fn affinity_mask(&self) -> Option<usize> {
        None
    }
    fn counters(&self) -> Option<Counters> {
        self.counters
    }
    fn memory_counters(&self) -> PROCESS_MEMORY_COUNTERS {
        self.process
    }
    fn private_working_set(&self) -> Option<u64> {
        Some(GB)
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

fn supply(free: u64, clean: u64) -> Supply {
    Supply {
        free_beyond_target: beyond(free, 0),
        clean_cache: clean,
        cold_reclaimable: 0,
    }
}

/// The performance counters split the available lists; Windows reports no
/// harm because no unprivileged counter separates reclaimed-page refaults
/// from cold file reads.
#[test]
fn the_counters_split_free_from_standby() {
    let sample = sample(&Machine::new());
    assert_eq!(
        (sample.total, sample.supply, sample.cold, sample.harm),
        (
            64 * GB,
            supply(10 * GB, 30 * GB),
            ColdPages {
                machine: 0,
                own: None
            },
            None,
        )
    );
}

/// Without the performance counters (an account outside Performance
/// Monitor Users), the free and zero lists are the nodes' available
/// bytes summed, the standby lists the rest of `ullAvailPhys`, and no
/// harm is counted.
#[test]
fn without_the_counters_free_is_the_nodes_and_standby_the_rest() {
    let taken = sample(&Machine {
        counters: None,
        ..Machine::new()
    });
    assert_eq!(
        (taken.total, taken.supply, taken.harm),
        (64 * GB, supply(10 * GB, 30 * GB), None)
    );
    // One node's available bytes include its standby lists: all free.
    let one_node = Machine {
        counters: None,
        nodes: vec![Some(40 * GB)],
        ..Machine::new()
    };
    assert_eq!(sample(&one_node).supply, supply(40 * GB, 0));
    // A number naming no node adds nothing.
    let sparse = Machine {
        counters: None,
        nodes: vec![Some(5 * GB), None, Some(2 * GB)],
        ..Machine::new()
    };
    assert_eq!(sample(&sparse).supply, supply(7 * GB, 33 * GB));
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
    assert_eq!((sample.total, sample.supply), (4 * GB, supply(GB, 0)));
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
    assert_eq!((sample.total, sample.supply), (5 * GB, supply(3 * GB, 0)));
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
