//! The Vulkan backend's description of its devices in the topology's terms
//! (`seismic_topology::VulkanReport`): its memory types and imported host
//! pages, as kinds.

use crate::DeviceHandle;
use seismic_lang::registry::BackendName;
use seismic_topology::{
    Contract, Diagnostic, DiagnosticSource, ExecutorKind, HostReads, KindMemory, KindReport, Obtained, Reach,
    Reservation, ReservationChunks, Reserving, VulkanHeap, VulkanMemory, VulkanReport,
};

/// The kind identifier of imported host pages; every memory type's is its
/// index.
pub const IMPORTED_KIND: u32 = u32::MAX;

/// How `facts` reserves in `kind`.
fn reserving(facts: &crate::Facts, kind: u32) -> Reserving {
    facts.reservation(kind).map(|form| form.map(vulkan_reservation))
}

/// The kinds of `layout`, each reserving as `reserving` says: each memory
/// type a kind on its heap, which the host maps where the type is mappable,
/// through its caches where it is host-cached, and only the device reaches
/// otherwise; and imported host pages, where a reservation can be backed in
/// them or says why not.
fn layout_kinds(layout: &crate::MemoryLayout, reserving: impl Fn(u32) -> Reserving) -> Vec<KindReport> {
    let mut kinds = (0..)
        .zip(&layout.types)
        .map(|(id, memory)| {
            let heap = &layout.heaps[memory.heap as usize];
            KindReport {
                id,
                name: format!("memory type {id}"),
                reach: match (memory.mappable(), memory.host_cached) {
                    (true, true) => Reach::Mapped { reads: HostReads::Cached },
                    (true, false) => Reach::Mapped { reads: HostReads::Uncached },
                    (false, _) => Reach::Device,
                },
                obtained: Obtained::Allocated,
                memory: KindMemory::Heap { index: memory.heap as usize, device_local: heap.device_local, bytes: heap.bytes },
                reservation: reserving(id),
            }
        })
        .collect::<Vec<_>>();
    let imported = reserving(IMPORTED_KIND);
    if imported != Ok(None) {
        kinds.push(KindReport {
            id: IMPORTED_KIND,
            name: "imported host pages".into(),
            reach: Reach::Mapped { reads: HostReads::Cached },
            obtained: Obtained::ImportedHostPages,
            memory: KindMemory::HostRam,
            reservation: imported,
        });
    }
    kinds
}

/// The kinds `facts` allocates.
pub(crate) fn kinds(facts: &crate::Facts) -> Vec<KindReport> {
    layout_kinds(&facts.memory, |kind| reserving(facts, kind))
}

/// `layout`'s heaps as its report states them.
fn heaps(layout: &crate::MemoryLayout) -> Vec<VulkanHeap> {
    layout.heaps.iter().map(|heap| VulkanHeap { bytes: heap.bytes, device_local: heap.device_local }).collect()
}

/// Every Vulkan device's report, with its handle, which status samples
/// read; why none can be enumerated, as `diagnostics`.
pub fn describe(diagnostics: &mut Vec<Diagnostic>) -> Vec<(VulkanReport, DeviceHandle)> {
    match crate::discover() {
        Err(error) => {
            diagnostics.push(Diagnostic {
                source: DiagnosticSource::Backend(BackendName::Vulkan),
                message: error.to_string(),
            });
            Vec::new()
        }
        Ok(descriptions) => {
            if descriptions.is_empty() {
                diagnostics.push(Diagnostic::no_device(BackendName::Vulkan));
            }
            descriptions
                .iter()
                .map(|description| (vulkan_report(description), description.handle.clone()))
                .collect()
        }
    }
}

/// The Vulkan backend's description of one device.
fn vulkan_report(description: &crate::Description) -> VulkanReport {
    let device = &description.facts;
    let kind = match device.device_type {
        crate::DeviceType::Cpu => ExecutorKind::Cpu,
        crate::DeviceType::Integrated => ExecutorKind::IntegratedGpu,
        crate::DeviceType::Discrete => ExecutorKind::DiscreteGpu,
        crate::DeviceType::Other => ExecutorKind::Other,
    };
    VulkanReport {
        uuid: device.uuid,
        kind,
        memory_is_host: (device.device_type != crate::DeviceType::Other)
            .then(|| device.device_type.memory_is_host()),
        name: device.name.clone(),
        floor: description.floor.clone(),
        driver: format!(
            "vulkan driver {} version {}",
            device.driver_id, device.driver_version
        ),
        max_allocation: device.limits.max_allocation_bytes,
        memory: VulkanMemory {
            luid: device.luid,
            heaps: heaps(&device.memory),
        },
        kinds: kinds(device),
    }
}

/// The reservation a Vulkan device's form describes: chunks of its unit,
/// addressed through a record table in upload storage.
fn vulkan_reservation(form: crate::ReservationForm) -> Reservation {
    Reservation {
        granule: form.granule,
        host_addressed: form.host_addressed,
        chunks: Some(ReservationChunks { unit: form.chunk_unit, table: Contract::Upload }),
    }
}

#[cfg(test)]
mod tests {
    use super::layout_kinds;
    use crate::{MemoryHeap, MemoryLayout, MemoryType, facts::chunk_backing};
    use seismic_topology::fixtures::{self, Layout};
    use seismic_topology::{Reports, VulkanReport};

    /// `report`'s memory layout, its types' flags from `recorded`.
    fn layout(report: &VulkanReport, recorded: Layout<'_>) -> MemoryLayout {
        let types = recorded.iter().enumerate().flat_map(|(heap, types)| {
            types.iter().map(move |flags| {
                let has = |flag: &str| flags.split('|').any(|each| each == flag);
                MemoryType {
                    heap: heap as u32,
                    device_local: has("DL"),
                    host_visible: has("HV"),
                    host_coherent: has("HC"),
                    host_cached: has("C"),
                }
            })
        });
        MemoryLayout {
            types: types.collect(),
            heaps: report
                .memory
                .heaps
                .iter()
                .map(|heap| MemoryHeap { bytes: heap.bytes, device_local: heap.device_local })
                .collect(),
        }
    }

    /// Every recorded Vulkan device's kinds are the ones this backend
    /// declares for its memory layout: each kind's reach, memory and
    /// obtaining, and which kinds reserve. How one reserves is the
    /// recording's.
    #[test]
    fn the_recorded_vulkan_kinds_are_the_backends() {
        let only = |reports: Reports| reports.vulkan.into_iter().next().unwrap();
        let [ryzen_igpu, rtx4070]: [VulkanReport; 2] = fixtures::ryzen_igpu_and_rtx4070().vulkan.try_into().unwrap();
        for (report, recorded) in [
            (only(fixtures::rtx3090()), fixtures::RTX3090_LAYOUT),
            (only(fixtures::gb10()), fixtures::GB10_LAYOUT),
            (fixtures::strix_halo_linux_vulkan(), fixtures::RADV_LAYOUT),
            (only(fixtures::strix_halo_windows()), fixtures::STRIX_HALO_WINDOWS_LAYOUT),
            (only(fixtures::lunar_lake()), fixtures::LUNAR_LAKE_LAYOUT),
            (ryzen_igpu, fixtures::RADV_LAYOUT),
            (rtx4070, fixtures::RTX4070_LAYOUT),
            (only(fixtures::a10_8q()), fixtures::A10_8Q_LAYOUT),
        ] {
            let layout = layout(&report, recorded);
            let memory_is_host = report.memory_is_host.unwrap();
            let reservation = |kind| {
                let recorded = report.kinds.iter().find(|each| each.id == kind);
                recorded.map_or(Ok(None), |recorded| recorded.reservation.clone())
            };
            // Whether a device maps at a placed address is recorded only by
            // what reserves, so a kind may reserve under either answer.
            let reserves = |kind| [false, true].iter().any(|&placed| chunk_backing(&layout, memory_is_host, placed, kind).is_some());
            let declared = layout_kinds(&layout, |kind| if reserves(kind) { reservation(kind) } else { Ok(None) });
            assert_eq!(declared, report.kinds, "{}", report.name);
        }
    }
}
