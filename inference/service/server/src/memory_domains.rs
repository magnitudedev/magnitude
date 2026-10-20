//! The public names of memory domains: the topology's stable store
//! identities. Host RAM is the `system` domain; every other store is named
//! by its identity, whichever backend reaches it, so the views of one
//! physical device's memory (a GPU seen through CUDA and Vulkan) share one
//! name. Assessments, hardware snapshots, merged worker observations and
//! instance allocations share these names.

use magnitude_engine::census::AllocationCensus;
use magnitude_service_contracts::MemoryDomainId;
use magnitude_service_contracts::models::{ModelInstanceAllocation, ModelInstanceMemoryDomain};
use seismic::{DeviceTopology, StoreId, StoreIdentity};

/// The name of the store `identity` names, in any process.
pub fn store_identity_id(identity: &StoreIdentity) -> MemoryDomainId {
    match identity {
        StoreIdentity::HostRam => MemoryDomainId::system(),
        identity => MemoryDomainId::new(identity.to_string()),
    }
}

/// The name of `topology`'s store `store`.
pub fn store_domain_id(topology: &DeviceTopology, store: StoreId) -> MemoryDomainId {
    store_identity_id(
        &topology
            .store_identity(store)
            .expect("store belongs to the topology"),
    )
}

/// An instance's allocation from its engine's census.
pub fn instance_allocation(
    context_window_tokens: u32,
    census: &AllocationCensus,
) -> ModelInstanceAllocation {
    ModelInstanceAllocation {
        context_window_tokens,
        memory_domains: census
            .stores
            .iter()
            .map(|store| ModelInstanceMemoryDomain {
                memory_domain_id: store_identity_id(&store.store),
                model_bytes: store.model_bytes,
                context_bytes: store.context_bytes,
                compute_bytes: store.compute_bytes,
                auxiliary_bytes: store.auxiliary_bytes,
            })
            .collect(),
    }
}
