//! Windows processors, thread pinning, device NUMA node stub, and SMBIOS
//! firmware identity.

use super::super::resolve;
use super::Sys;
use seismic_topology::{
    flat_nodes, Core, Firmware, HostNode, MemoryNode, PciLocation, Processors,
};
use std::collections::BTreeSet;
use windows_sys::Win32::System::SystemInformation::{
    GROUP_AFFINITY, SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
};

/// Each record of `buffer` ([`Sys::processor_records`]), in order.
pub(super) fn each_record(
    buffer: &[u64],
    mut each: impl FnMut(&SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX),
) {
    let (bytes, length) = (buffer.as_ptr().cast::<u8>(), buffer.len() * 8);
    let mut offset = 0usize;
    while offset < length {
        // SAFETY: the OS wrote consecutive records, each `Size` bytes long.
        let record = unsafe {
            &*(bytes.add(offset) as *const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)
        };
        if record.Size == 0 {
            break;
        }
        each(record);
        offset += record.Size as usize;
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

/// Each NUMA node (`RelationNumaNode`), with memory where
/// `GetNumaAvailableMemoryNodeEx` reports any available on it, its cores
/// (`RelationProcessorCore`) those whose threads lie in it, each a
/// performance core of the highest `EfficiencyClass`; the process may run
/// on the CPUs of its groups (`GetProcessGroupAffinity`), within its
/// affinity mask where it has one group (`GetProcessAffinityMask`). Where
/// any of it is unanswered, the flat topology ([`flat_nodes`]) of the
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
    use windows_sys::Win32::System::SystemInformation::{RelationNumaNode, RelationProcessorCore};
    let mut cores = Vec::new();
    each_record(&sys.processor_records(RelationProcessorCore)?, |record| {
        // SAFETY: a `RelationProcessorCore` record's union is its processor.
        let core = unsafe { &record.Anonymous.Processor };
        // SAFETY: the record holds `GroupCount` masks.
        let masks = unsafe {
            std::slice::from_raw_parts(core.GroupMask.as_ptr(), usize::from(core.GroupCount))
        };
        cores.push((group_cpus(masks), core.EfficiencyClass));
    });
    let fastest = cores.iter().map(|(_, class)| *class).max()?;
    let mut nodes = Vec::new();
    let mut unanswered = false;
    each_record(&sys.processor_records(RelationNumaNode)?, |record| {
        // SAFETY: a `RelationNumaNode` record's union is its node, holding
        // `GroupCount` masks (one where it reports none).
        let node = unsafe { &record.Anonymous.NumaNode };
        let masks = unsafe {
            std::slice::from_raw_parts(
                node.Anonymous.GroupMasks.as_ptr(),
                usize::from(node.GroupCount).max(1),
            )
        };
        let cpus = group_cpus(masks);
        let number = u16::try_from(node.NodeNumber)
            .unwrap_or_else(|_| panic!("invariant violated: NUMA node {}", node.NodeNumber));
        let Some(available) = sys.numa_available(number) else {
            unanswered = true;
            return;
        };
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
            memory: available > 0,
            cores,
        });
    });
    if unanswered {
        return None;
    }
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
