//! macOS processors, thread placement stubs, device NUMA node stub, and
//! firmware/marketing identity.

use super::super::resolve;
use super::{sysctl_u64, Sys};
use seismic_topology::{
    flat_nodes, Core, Firmware, HostNode, MemoryNode, PciLocation, Processors,
};

/// One memory node of every core: per performance level (`hw.nperflevels`,
/// level 0 the performance cores and every later level efficiency cores, as
/// many levels as the OS reports), its `physicalcpu` cores of
/// `logicalcpu / physicalcpu` threads each. Where the levels are not
/// reported, every core is one performance level: the `hw.physicalcpu`
/// cores of `hw.logicalcpu` threads, or, where those are not reported
/// either, the flat topology of the quota's CPUs ([`flat_nodes`]). A CPU is
/// its ordinal, performance cores first, and the process may run on every
/// one.
///
/// # Panics
/// Where a level's threads do not divide over its cores.
pub(crate) fn processors(sys: &impl Sys) -> Processors {
    let reported = |name: &str| sysctl_u64(sys, name);
    let from_levels = |levels: Vec<(u64, u64, bool)>| {
        let mut cores = Vec::new();
        let mut next = 0usize;
        for (physical, logical, performance) in levels {
            assert!(
                logical % physical == 0,
                "invariant violated: {logical} logical CPUs over {physical} cores"
            );
            let threads = usize::try_from(logical / physical)
                .unwrap_or_else(|_| panic!("invariant violated: {logical} logical CPUs"));
            for _ in 0..physical {
                cores.push(Core {
                    threads: (next..next + threads).collect(),
                    performance,
                });
                next += threads;
            }
        }
        (
            vec![MemoryNode {
                id: 0,
                memory: true,
                cores,
            }],
            (0..next).collect(),
        )
    };
    let (nodes, every) = resolve(
        &[
            ((), &|| {
                let count = reported("hw.nperflevels")?;
                let levels = (0..count)
                    .map(|level| {
                        let field = |name: &str| reported(&format!("hw.perflevel{level}.{name}"));
                        Some((field("physicalcpu")?, field("logicalcpu")?, level == 0))
                    })
                    .collect::<Option<Vec<_>>>()?;
                Some(from_levels(levels))
            }),
            ((), &|| {
                let physical = reported("hw.physicalcpu")?;
                let logical = reported("hw.logicalcpu")?;
                Some(from_levels(vec![(physical, logical, true)]))
            }),
        ],
        (),
        || {
            let every = (0..sys.parallelism().get()).collect();
            (flat_nodes(&every), every)
        },
    )
    .0;
    Processors::new(nodes, every, sys.parallelism())
        .unwrap_or_else(|error| panic!("invariant violated: the processors the OS reports: {error}"))
}

/// macOS places no thread on a core it names: its scheduler places every
/// thread, so nothing is placed.
pub(crate) fn place_process(_cpus: &[usize]) -> Result<(), String> {
    Ok(())
}

/// macOS places no thread on a core it names: its scheduler places every
/// thread, so nothing is placed.
pub(crate) fn pin(_cpus: &[usize]) -> Result<(), String> {
    Ok(())
}

/// macOS reads no device's memory node.
pub(crate) fn device_node(_sys: &impl Sys, _location: PciLocation) -> HostNode {
    HostNode::Any
}

/// The machine's names (`seismic_topology::SystemIdentity`): Apple and
/// `hw.model` (an opaque identifier such as `Mac16,5`), and
/// `system_profiler`'s marketing name (`SPHardwareDataType`'s
/// `machine_name`, such as "MacBook Pro"). macOS reports no enclosure type.
/// A field it does not report reads as nothing.
pub(crate) fn firmware(sys: &impl Sys) -> Firmware {
    let model = resolve(&[((), &|| sys.sysctl("hw.model"))], (), Vec::new).0;
    let name = resolve(
        &[((), &|| {
            let stdout = sys.hardware_profiler_json()?;
            let document = serde_json::from_slice::<serde_json::Value>(&stdout).ok()?;
            let name = document
                .get("SPHardwareDataType")?
                .as_array()?
                .first()?
                .get("machine_name")?
                .as_str()?;
            Some(name.as_bytes().to_vec())
        })],
        (),
        Vec::new,
    )
    .0;
    Firmware {
        manufacturer: b"Apple".to_vec(),
        model,
        name,
        ..Firmware::default()
    }
}
