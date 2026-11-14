//! The allocation census a loaded engine reports (integration spec §9.2):
//! each store's charge by class, as the device's store census classified
//! it: every byte the model holds, its host tables' row caches among them,
//! is charged to the ledger once. Because memory is elastic these are the
//! standing at the observation, not a fixed reservation. Beside it, how the
//! owner serves at the observation (`ServingCensus`): its prompt groups'
//! places and where its routed experts are.

use magnitude_executor::{MemoryChargeReconciliation, StoreAllocation};
pub use magnitude_executor::{BankHolding, BankResize, ExpertCover, ExpertResidency, HostResidency, TableResidency};
pub use magnitude_scheduler::serving::ServingCensus;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocationCensus {
    pub stores: Vec<StoreAllocation>,
    pub serving: ServingCensus,
}

impl AllocationCensus {
    /// The census of `charge`'s stores, as their ledgers charged them.
    pub(crate) fn classify(charge: &MemoryChargeReconciliation, serving: ServingCensus) -> Self {
        Self { stores: charge.stores.clone(), serving }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_state::StateHoldingCensus;

    fn store(store: seismic::StoreIdentity, model_bytes: u64) -> StoreAllocation {
        StoreAllocation {
            store,
            model_bytes,
            context_bytes: 0,
            compute_bytes: 0,
            auxiliary_bytes: 0,
        }
    }

    fn serving() -> ServingCensus {
        ServingCensus {
            group_places: std::num::NonZeroUsize::MIN,
            group_capacity: std::num::NonZeroUsize::MIN,
            experts: None,
        }
    }

    fn charge(stores: Vec<StoreAllocation>) -> MemoryChargeReconciliation {
        let allocated = stores.iter().map(|store| store.model_bytes).sum();
        MemoryChargeReconciliation {
            stores,
            allocated,
            target_state: StateHoldingCensus::default(),
            head_state: None,
            graph_pools: 0,
            owned_features: 0,
            target_weights: allocated,
            optional_weights: 0,
            prepared_programs: 0,
            bound_constants: 0,
        }
    }

    /// Every store is reported as its ledger charged it (host RAM for
    /// CUDA's pinned upload storage and the host tables' row caches),
    /// nothing added beside.
    #[test]
    fn stores_are_reported_as_their_ledgers_charged_them() {
        let device = seismic::StoreIdentity::DeviceMemory(seismic::PhysicalIdentity::Uuid([3; 16]));
        let host = seismic::StoreIdentity::HostRam;
        // 700 device bytes, and 300 bytes of tables in host RAM.
        let charged = charge(vec![store(device.clone(), 700), store(host.clone(), 300)]);
        let census = AllocationCensus::classify(&charged, serving());
        assert_eq!(census.stores, [store(device.clone(), 700), store(host, 300)]);
        let local = charge(vec![store(device.clone(), 1_000)]);
        assert_eq!(AllocationCensus::classify(&local, serving()).stores, [store(device, 1_000)]);
    }
}
