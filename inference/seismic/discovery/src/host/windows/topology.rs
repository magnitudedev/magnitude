//! Windows processors, thread pinning, device NUMA node stub, and SMBIOS
//! firmware identity.

use super::super::resolve;
use super::Sys;
use seismic_topology::{
    flat_nodes, Core, Firmware, HostNode, MemoryNode, PciLocation, Processors,
};
use std::collections::BTreeSet;
use windows_sys::Win32::System::SystemInformation::{
    CACHE_RELATIONSHIP, GROUP_AFFINITY, LOGICAL_PROCESSOR_RELATIONSHIP, NUMA_NODE_RELATIONSHIP,
    PROCESSOR_RELATIONSHIP, RelationCache, RelationNumaNode, RelationNumaNodeEx,
    RelationProcessorCore, SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
};

/// Aligned, exact-length buffer of `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX`
/// records returned by `GetLogicalProcessorInformationEx`.
#[derive(Clone)]
pub(crate) struct ProcessorRecords {
    words: Vec<u64>,
    byte_len: usize,
}

impl ProcessorRecords {
    pub(crate) fn new(words: Vec<u64>, byte_len: usize) -> Self {
        assert!(
            byte_len <= words.len().saturating_mul(std::mem::size_of::<u64>()),
            "invariant violated: ProcessorRecords byte_len {byte_len} exceeds backing buffer"
        );
        Self { words, byte_len }
    }

    /// Walk each valid `(Relationship, ptr, size)` header in the buffer using
    /// raw pointers so we never construct an over-wide `&SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX`
    /// reference over a shorter variant or trailing tail.
    fn each_raw(
        &self,
        mut each: impl FnMut(
            LOGICAL_PROCESSOR_RELATIONSHIP,
            *const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
            usize,
        ),
    ) {
        const HEADER_BYTES: usize = 8;
        let align = std::mem::align_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>();
        let base = self.words.as_ptr().cast::<u8>();
        let mut offset = 0usize;
        while offset.saturating_add(HEADER_BYTES) <= self.byte_len && offset % align == 0 {
            // SAFETY: `words` is 8-byte aligned and each OS record is aligned to
            // at least 4 bytes; `offset + HEADER_BYTES <= self.byte_len`.
            let record = unsafe { base.add(offset).cast::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>() };
            let relationship = unsafe { std::ptr::addr_of!((*record).Relationship).read_unaligned() };
            let size = unsafe { std::ptr::addr_of!((*record).Size).read_unaligned() } as usize;
            if size < HEADER_BYTES || offset.saturating_add(size) > self.byte_len {
                break;
            }
            each(relationship, record, size);
            offset += size;
        }
    }

    /// Visit every `RelationCache` record whose byte size covers `CACHE_RELATIONSHIP`.
    pub(super) fn caches(&self, mut each: impl FnMut(&CACHE_RELATIONSHIP)) {
        let union_offset = std::mem::offset_of!(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Anonymous);
        let needed = union_offset + std::mem::size_of::<CACHE_RELATIONSHIP>();
        self.each_raw(|relationship, record, size| {
            if relationship == RelationCache && size >= needed {
                // SAFETY: `relationship == RelationCache` and `size` covers the full
                // `CACHE_RELATIONSHIP` payload inside the allocation.
                let cache = unsafe { &*std::ptr::addr_of!((*record).Anonymous.Cache) };
                each(cache);
            }
        });
    }

    /// Visit every `RelationProcessorCore` record, yielding `(efficiency_class, group_masks)`.
    fn processor_cores(&self, mut each: impl FnMut(u8, &[GROUP_AFFINITY])) {
        let union_offset = std::mem::offset_of!(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Anonymous);
        let masks_offset = union_offset + std::mem::offset_of!(PROCESSOR_RELATIONSHIP, GroupMask);
        let min_size = union_offset + std::mem::size_of::<PROCESSOR_RELATIONSHIP>();
        self.each_raw(|relationship, record, size| {
            if relationship != RelationProcessorCore || size < min_size {
                return;
            }
            // SAFETY: `relationship == RelationProcessorCore` and `size` covers the
            // fixed header of `PROCESSOR_RELATIONSHIP`.
            let core = unsafe { &*std::ptr::addr_of!((*record).Anonymous.Processor) };
            let count = usize::from(core.GroupCount);
            let masks_end = masks_offset.saturating_add(count.saturating_mul(std::mem::size_of::<GROUP_AFFINITY>()));
            if masks_end <= size {
                // SAFETY: `count` `GROUP_AFFINITY` entries lie strictly within `size`.
                let masks = unsafe { std::slice::from_raw_parts(core.GroupMask.as_ptr(), count) };
                each(core.EfficiencyClass, masks);
            }
        });
    }

    /// Visit every `RelationNumaNode` / `RelationNumaNodeEx` record, yielding
    /// `(node_number, group_masks)`.
    fn numa_nodes(&self, mut each: impl FnMut(u32, &[GROUP_AFFINITY])) {
        let union_offset = std::mem::offset_of!(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Anonymous);
        let masks_offset = union_offset + std::mem::offset_of!(NUMA_NODE_RELATIONSHIP, Anonymous);
        let min_size = union_offset + std::mem::size_of::<NUMA_NODE_RELATIONSHIP>();
        self.each_raw(|relationship, record, size| {
            if (relationship != RelationNumaNode && relationship != RelationNumaNodeEx)
                || size < min_size
            {
                return;
            }
            // SAFETY: `relationship` is a NUMA node record and `size` covers
            // the fixed header of `NUMA_NODE_RELATIONSHIP`.
            let node = unsafe { &*std::ptr::addr_of!((*record).Anonymous.NumaNode) };
            let count = usize::from(node.GroupCount).max(1);
            let masks_end = masks_offset.saturating_add(count.saturating_mul(std::mem::size_of::<GROUP_AFFINITY>()));
            if masks_end <= size {
                // SAFETY: `count` `GROUP_AFFINITY` entries lie strictly within `size`.
                let masks = unsafe {
                    std::slice::from_raw_parts(node.Anonymous.GroupMasks.as_ptr(), count)
                };
                each(node.NodeNumber, masks);
            }
        });
    }
}

/// The CPUs of `masks`, each its group's 64 times its group plus its bit.
fn group_cpus(masks: &[GROUP_AFFINITY]) -> Vec<usize> {
    let mut cpus = masks
        .iter()
        .flat_map(|mask| {
            (0..usize::BITS as usize)
                .filter(move |bit| mask.Mask >> bit & 1 == 1)
                .map(move |bit| usize::from(mask.Group) * 64 + bit)
        })
        .collect::<Vec<_>>();
    cpus.sort_unstable();
    cpus
}

/// Each NUMA node (`RelationNumaNodeEx`, falling back to `RelationNumaNode` on
/// older Windows), with memory where `GetNumaAvailableMemoryNodeEx` answers
/// for it (static node memory presence, independent of momentary free bytes
/// under pressure), its cores (`RelationProcessorCore`) those whose threads
/// lie in it, each a performance core of the highest `EfficiencyClass`; the
/// process may run on the CPUs of its groups (`GetProcessGroupAffinity`),
/// within its affinity mask where it has one group (`GetProcessAffinityMask`).
/// Where any of it is unanswered, the flat topology ([`flat_nodes`]) of the
/// quota's CPUs.
pub(crate) fn processors(sys: &impl Sys) -> Processors {
    let (nodes, allowed) = resolve(
        &[((), &|| reported_processors(sys))],
        (),
        || {
            let every = (0..sys.parallelism().get()).collect();
            (flat_nodes(&every), every)
        },
    )
    .0;
    Processors::new(nodes, allowed, sys.parallelism())
        .unwrap_or_else(|error| panic!("invariant violated: the processors the OS reports: {error}"))
}

/// The nodes and allowed CPUs the OS reports ([`processors`]), none where
/// any of them is unanswered.
fn reported_processors(sys: &impl Sys) -> Option<(Vec<MemoryNode>, BTreeSet<usize>)> {
    let mut cores = Vec::new();
    sys.processor_records(RelationProcessorCore)?
        .processor_cores(|efficiency_class, masks| {
            cores.push((group_cpus(masks), efficiency_class));
        });
    let fastest = cores.iter().map(|(_, class)| *class).max()?;
    let numa_records = sys
        .processor_records(RelationNumaNodeEx)
        .or_else(|| sys.processor_records(RelationNumaNode))?;
    let mut nodes = Vec::new();
    numa_records.numa_nodes(|node_number, masks| {
        let cpus = group_cpus(masks);
        let number = u16::try_from(node_number)
            .unwrap_or_else(|_| panic!("invariant violated: NUMA node {node_number}"));
        let cores = cores
            .iter()
            .filter(|(threads, _)| threads.first().is_some_and(|first| cpus.contains(first)))
            .map(|(threads, class)| Core {
                threads: threads.clone(),
                performance: *class == fastest,
            })
            .collect();
        nodes.push(MemoryNode {
            id: usize::from(number),
            memory: sys.numa_available(number).is_some(),
            cores,
        });
    });
    let every = cores.iter().flat_map(|(threads, _)| threads.iter().copied());
    let allowed = match sys.process_groups()?.as_slice() {
        [group] => {
            let mask = sys.affinity_mask()?;
            every
                .filter(|cpu| cpu / 64 == usize::from(*group) && mask >> (cpu % 64) & 1 == 1)
                .collect()
        }
        groups => every
            .filter(|cpu| groups.iter().any(|group| cpu / 64 == usize::from(*group)))
            .collect(),
    };
    Some((nodes, allowed))
}

/// Nothing: the process's affinity mask would bound every thread's,
/// [`pin`]'s workers among them, and the process's threads are not
/// enumerated here.
pub(crate) fn place_process(_cpus: &[usize]) -> Result<(), String> {
    Ok(())
}

/// `SetThreadGroupAffinity` of the calling thread to `cpus`: a thread runs
/// in one processor group, so to those of the group holding the most of
/// them (the first such).
pub(crate) fn pin(cpus: &[usize]) -> Result<(), String> {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadGroupAffinity};
    let in_group = |group: usize| cpus.iter().filter(|&&cpu| cpu / 64 == group).count();
    let group = cpus
        .iter()
        .map(|&cpu| cpu / 64)
        .max_by_key(|&group| (in_group(group), std::cmp::Reverse(group)))
        .ok_or("a thread placed on no CPU")?;
    let mask = cpus
        .iter()
        .filter(|&&cpu| cpu / 64 == group)
        .fold(0usize, |mask, &cpu| mask | 1 << (cpu % 64));
    let group = u16::try_from(group).map_err(|_| format!("CPUs {cpus:?} beyond every group"))?;
    let affinity = GROUP_AFFINITY {
        Mask: mask,
        Group: group,
        Reserved: [0; 3],
    };
    match unsafe { SetThreadGroupAffinity(GetCurrentThread(), &affinity, std::ptr::null_mut()) } {
        0 => Err(format!(
            "SetThreadGroupAffinity to CPUs {cpus:?}: {}",
            std::io::Error::last_os_error()
        )),
        _ => Ok(()),
    }
}

/// No device's memory node is read here.
pub(crate) fn device_node(_sys: &impl Sys, _location: PciLocation) -> HostNode {
    HostNode::Any
}

/// The machine's names (`seismic_topology::SystemIdentity`): the SMBIOS
/// table `GetSystemFirmwareTable` returns (provider `RSMB`), parsed via
/// [`seismic_topology::parse_smbios`].
pub(crate) fn firmware(sys: &impl Sys) -> Firmware {
    resolve(
        &[((), &|| {
            sys.smbios_table()
                .as_deref()
                .map(seismic_topology::parse_smbios)
        })],
        (),
        Firmware::default,
    )
    .0
}
