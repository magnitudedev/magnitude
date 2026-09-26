//! The public names of memory domains. Host RAM is the `system` domain; a dedicated memory pool is
//! named by the selector of the first device (in discovery order) allocating from it, the device
//! identity the hardware snapshot and load plans use. Several views of one physical pool (a GPU
//! seen through CUDA and Vulkan) share one name. Assessments, hardware snapshots, merged worker
//! observations and instance allocations share these names.

use magnitude_engine::census::{AllocationCensus, MemoryDomain};
use magnitude_service_contracts::MemoryDomainId;
use magnitude_service_contracts::models::{ModelInstanceAllocation, ModelInstanceMemoryDomain};
use seismic::{DeviceMemory, DeviceTopology, MemoryPoolId};

/// The name of a domain an engine reports: host RAM, or the pool its device allocates from.
pub fn census_domain_id(topology: &DeviceTopology, domain: &MemoryDomain) -> MemoryDomainId {
    match domain {
        MemoryDomain::HostRam => MemoryDomainId::system(),
        MemoryDomain::DeviceLocal { device } => {
            let info = topology
                .devices()
                .iter()
                .find(|info| info.selector == *device)
                .expect("a worker's device is in the topology its load was previewed on");
            match &info.memory {
                DeviceMemory::Established(memory) => pool_domain_id(topology, memory.allocation_pool),
                DeviceMemory::Unsupported { .. } => {
                    unreachable!("a loaded device has established memory")
                }
            }
        }
    }
}

/// The name of a Seismic memory pool: the host pool is `system`; a dedicated pool is named by
/// the first device (in discovery order) allocating from it.
pub fn pool_domain_id(topology: &DeviceTopology, pool: MemoryPoolId) -> MemoryDomainId {
    if pool == topology.host_pool().id {
        return MemoryDomainId::system();
    }
    let device = topology
        .devices()
        .iter()
        .find(|device| {
            matches!(&device.memory, DeviceMemory::Established(memory) if memory.allocation_pool == pool)
        })
        .expect("every dedicated pool belongs to a discovered device");
    MemoryDomainId::new(device.selector.to_string())
}

/// An instance's allocation from its engine's census.
pub fn instance_allocation(
    topology: &DeviceTopology,
    context_window_tokens: u32,
    census: &AllocationCensus,
) -> ModelInstanceAllocation {
    ModelInstanceAllocation {
        context_window_tokens,
        memory_domains: census
            .domains
            .iter()
            .map(|domain| ModelInstanceMemoryDomain {
                memory_domain_id: census_domain_id(topology, &domain.domain),
                model_bytes: domain.model_bytes,
                context_bytes: domain.context_bytes,
                compute_bytes: domain.compute_bytes,
                auxiliary_bytes: domain.auxiliary_bytes,
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_engine::census::DomainAllocation;
    use seismic::DeviceCatalog;

    /// Every census domain is named like the hardware snapshot's domain that holds it.
    #[test]
    fn census_domains_use_the_public_pool_names() {
        let topology = DeviceCatalog::discover().expect("device discovery").topology();
        let host = DomainAllocation {
            domain: MemoryDomain::HostRam,
            model_bytes: 5,
            context_bytes: 1,
            compute_bytes: 2,
            auxiliary_bytes: 3,
        };
        let allocation = instance_allocation(&topology, 4_096, &AllocationCensus { domains: vec![host] });
        assert_eq!(allocation.context_window_tokens, 4_096);
        assert!(allocation.memory_domains[0].memory_domain_id.is_system());
        assert_eq!(allocation.memory_domains[0].auxiliary_bytes, 3);
        for device in topology.devices() {
            let DeviceMemory::Established(memory) = &device.memory else {
                continue;
            };
            if memory.allocates_host_memory() {
                continue;
            }
            assert_eq!(
                census_domain_id(&topology, &MemoryDomain::DeviceLocal { device: device.selector }),
                pool_domain_id(&topology, memory.allocation_pool)
            );
        }
    }
}
