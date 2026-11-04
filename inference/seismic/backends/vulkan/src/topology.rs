//! The Vulkan backend's description of its devices in the topology's terms
//! (`seismic_topology::VulkanReport`).

use crate::DeviceHandle;
use seismic_topology::{
    BackendName, Diagnostic, DiagnosticSource, ExecutorKind, VulkanHeap, VulkanMemory, VulkanReport,
};

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
                .map(|description| (vulkan_report(description), description.handle))
                .collect()
        }
    }
}

/// The Vulkan backend's description of one device.
fn vulkan_report(description: &crate::Description) -> VulkanReport {
    let device = &description.facts;
    let heaps = |types: &[u32]| {
        crate::Allocation::heaps(&device.memory, types)
            .into_iter()
            .map(|heap| heap as usize)
            .collect()
    };
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
            heaps: device
                .memory
                .heaps
                .iter()
                .map(|heap| VulkanHeap {
                    bytes: heap.bytes,
                    device_local: heap.device_local,
                })
                .collect(),
            allocates: [
                heaps(&device.allocation.device),
                heaps(&device.allocation.upload),
            ],
        },
    }
}
