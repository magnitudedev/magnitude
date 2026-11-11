//! The inventory over recorded machines (`seismic_topology::fixtures`): its
//! refreshes and samples, and the machine this runs on.

use crate::gather::{Gathered, Natives};
use crate::Inventory;
use seismic_topology::fixtures::{
    assert_consistent, executor, placement, rtx3090, rtx3090_host, rtx3090_uuid, rtx3090_volume,
    show, strix_halo_linux, topology, uuid, volume, STRIX_HALO_LINUX_UUID,
};
use seismic_topology::{
    Availability, BackendName, Contract, DeviceTopology, ExecutorSelector, PhysicalIdentity,
    Placing, Reports, Revision, StoreKind,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn gathered(reports: Reports) -> Gathered {
    Gathered {
        reports,
        natives: Natives::default(),
    }
}

/// Every identifier `topology` holds.
fn identifiers(topology: &DeviceTopology) -> Vec<Revision> {
    let mut revisions = Vec::new();
    for device in topology.physical_devices() {
        revisions.push(device.id().revision());
    }
    for store in topology.stores() {
        revisions.push(store.id().revision());
        if let StoreKind::DeviceMemory(device) = store.kind() {
            revisions.push(device.revision());
        }
    }
    for executor in topology.executors() {
        revisions.push(executor.id().revision());
        revisions.push(executor.physical().revision());
        for kind in Contract::ALL {
            revisions.extend(
                executor
                    .paths(Placing::Fixed(kind))
                    .iter()
                    .map(|id| id.revision()),
            );
        }
    }
    for path in topology.access_paths() {
        revisions.extend([
            path.id().revision(),
            path.executor().revision(),
            path.store().revision(),
        ]);
    }
    for path in topology.transfer_paths() {
        revisions.extend([
            path.id().revision(),
            path.from().revision(),
            path.to().revision(),
            path.by().revision(),
        ]);
    }
    revisions
}

#[test]
fn a_changed_refresh_leaves_no_identifier_at_its_old_revision() {
    // A BIOS change of the carve-out: RADV's device-local heap grows by it.
    let strix = |carve_out: u64| {
        let mut reports = strix_halo_linux();
        reports.vulkan[0].memory.heaps[1].bytes += carve_out;
        reports.volumes.push(rtx3090_volume("/models"));
        gathered(reports)
    };
    let inventory = Inventory::from_gathered(vec![PathBuf::from("/models")], strix(0));
    let before = inventory.topology();
    // The same machine keeps its revision and identifiers.
    let same = inventory.advance(strix(0));
    assert!(Arc::ptr_eq(&before, &same));
    let rebuilt = topology(&strix(0).reports);
    assert_ne!(rebuilt.revision(), before.revision());
    assert_eq!(rebuilt.normalized(), before.normalized());
    let after = inventory.advance(strix(1 << 30));
    assert_ne!(after.revision(), before.revision());
    assert_ne!(after.normalized(), before.normalized());
    assert!(identifiers(&after)
        .into_iter()
        .all(|revision| revision == after.revision()));
    assert!(identifiers(&before)
        .into_iter()
        .all(|revision| revision == before.revision()));
    let gpu = PhysicalIdentity::Uuid(uuid(STRIX_HALO_LINUX_UUID));
    let old = executor(&before, BackendName::Vulkan, &gpu);
    assert!(after.executor(old.id()).is_none());
    assert!(after.store(before.host_ram().id()).is_none());
    assert!(after
        .access_path(old.paths(Placing::Fixed(Contract::DeviceOnly))[0])
        .is_none());
    assert_eq!(
        volume(&after, Path::new("/models")).unwrap().revision(),
        after.revision()
    );
    assert_consistent(&after);
}

/// A sample reads the grants of the executors it is for alone: the RTX
/// 3090's CUDA executor reads no Vulkan heap budget, though its Vulkan
/// executor on the same device has grants a heap budget reads.
#[test]
fn a_sample_reads_the_grants_of_its_executors_alone() {
    let inventory = Inventory::from_gathered(Vec::new(), gathered(rtx3090()));
    let topology = inventory.topology();
    let gpu = PhysicalIdentity::Uuid(rtx3090_uuid());
    let cuda = executor(&topology, BackendName::Cuda, &gpu).id();
    let vulkan = executor(&topology, BackendName::Vulkan, &gpu).id();
    let sourced = |of| {
        topology
            .grant_sources()
            .filter(|(path, _, _)| path.executor() == of)
            .map(|(path, _, _)| path.id())
            .collect::<Vec<_>>()
    };
    assert!(sourced(cuda).is_empty());
    assert!(!sourced(vulkan).is_empty());
    let read = |executors: &[_]| {
        inventory
            .sample(&topology, executors)
            .grants
            .into_iter()
            .map(|(path, _)| path)
            .collect::<Vec<_>>()
    };
    assert!(read(&[cuda]).is_empty());
    assert_eq!(read(&[vulkan]), sourced(vulkan));
    assert_eq!(read(&[cuda, vulkan]), sourced(vulkan));
}

/// The machine this runs on: its reports, the topology they make, and one
/// sample of it.
#[test]
fn the_machine_is_discovered_and_sampled() {
    let directory = std::env::temp_dir();
    let gathered = crate::gather::gather(&[directory.clone()]).unwrap();
    eprintln!("{:#?}", gathered.reports);
    let inventory = Inventory::from_gathered(vec![directory.clone()], gathered);
    let topology = inventory.topology();
    show("This machine", &topology);
    assert_consistent(&topology);
    let cpu = topology.resolve(ExecutorSelector::HostCpu).unwrap();
    if *cpu.availability() == Availability::Available {
        assert_eq!(
            placement(&topology, cpu, Contract::DeviceOnly),
            [(StoreKind::HostRam, None)]
        );
    }
    let served = volume(&topology, &directory).unwrap();
    let every = topology
        .executors()
        .iter()
        .map(|executor| executor.id())
        .collect::<Vec<_>>();
    let sample = inventory.sample(&topology, &every);
    let host = sample.host_ram.as_ref().unwrap();
    assert!(host.allocatable.value > 0);
    assert_eq!(sample.volumes.len(), 1);
    assert_eq!(sample.volumes[0].0, served);
    assert!(sample.volumes[0].1.as_ref().unwrap().value > 0);
    // Every grant whose status needs no opened executor is read, from the
    // handles discovery found.
    assert_eq!(sample.grants.len(), topology.grant_sources().count());
    assert!(sample.grants.iter().all(|(_, left)| left.is_ok()));
    eprintln!("{sample:#?}");
    assert_eq!(inventory.refresh().unwrap().revision(), topology.revision());
}

#[test]
fn a_cpu_without_the_instruction_set_floor_is_unavailable_with_its_reason() {
    let mut reports = rtx3090_host();
    // What discovery reads where the CPU reports no native tier.
    reports.cpu.availability = crate::gather::cpu_availability(None);
    let topology = topology(&reports);
    show("CPU-only VM without AVX", &topology);
    let cpu = topology.resolve(ExecutorSelector::HostCpu).unwrap();
    assert!(matches!(
        cpu.availability(),
        Availability::Unavailable { .. }
    ));
    assert!(topology.access_paths().is_empty());
}
