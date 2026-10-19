//! The allocation census a loaded engine reports (integration spec §9.2):
//! each store's charge by class, as the device's store census classified
//! it, with the model's host-resident tables, which the runtime does not
//! allocate, added to host RAM. Because memory is elastic these are the
//! standing at the observation, not a fixed reservation.

use magnitude_executor::{MemoryChargeReconciliation, StoreAllocation};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocationCensus {
    pub stores: Vec<StoreAllocation>,
}

impl AllocationCensus {
    /// The census of `charge`'s stores, with `host_table_bytes` of
    /// host-resident tables as model bytes of host RAM: beside what host
    /// RAM's ledger charged, or as its own entry where it charged nothing.
    pub(crate) fn classify(
        charge: &MemoryChargeReconciliation,
        host_table_bytes: u64,
    ) -> Result<Self, String> {
        let mut stores = charge.stores.clone();
        if host_table_bytes != 0 {
            let host_ram = stores
                .iter_mut()
                .find(|store| store.store == seismic::StoreIdentity::HostRam);
            match host_ram {
                Some(host_ram) => {
                    host_ram.model_bytes = host_ram
                        .model_bytes
                        .checked_add(host_table_bytes)
                        .ok_or("host model census overflow")?;
                }
                None => stores.push(StoreAllocation {
                    store: seismic::StoreIdentity::HostRam,
                    model_bytes: host_table_bytes,
                    context_bytes: 0,
                    compute_bytes: 0,
                    auxiliary_bytes: 0,
                }),
            }
        }
        Ok(Self { stores })
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

    fn charge(stores: Vec<StoreAllocation>) -> MemoryChargeReconciliation {
        let charged = stores.iter().map(|store| store.model_bytes).sum();
        MemoryChargeReconciliation {
            stores,
            charged,
            target_state: StateHoldingCensus::default(),
            head_state: None,
            graph_pools: 0,
            owned_features: 0,
            target_weights: charged,
            optional_weights: 0,
            prepared_programs: 0,
            bound_constants: 0,
            external_pins: 0,
            unattributed: 0,
        }
    }

    /// Stored tables are reported in the store whose ledger charged them
    /// (host RAM for CUDA's pinned upload storage), which the census keeps;
    /// the host tables the runtime does not allocate are model bytes of host
    /// RAM, beside what the ledger charged there or as host RAM's own entry.
    #[test]
    fn stored_tables_in_host_ram_are_reported_there() {
        let device = seismic::StoreIdentity::DeviceMemory(seismic::PhysicalIdentity::Uuid([3; 16]));
        let host = seismic::StoreIdentity::HostRam;
        // 700 device bytes, and 300 bytes of stored tables in pinned host RAM.
        let pinned = charge(vec![store(device.clone(), 700), store(host.clone(), 300)]);
        let census = AllocationCensus::classify(&pinned, 10).unwrap();
        assert_eq!(
            census.stores,
            [store(device.clone(), 700), store(host.clone(), 310)]
        );
        // Nothing charged to host RAM: the host tables are its entry.
        let local = charge(vec![store(device.clone(), 1_000)]);
        let census = AllocationCensus::classify(&local, 10).unwrap();
        assert_eq!(census.stores, [store(device.clone(), 1_000), store(host, 10)]);
        // No host tables: the ledger's stores alone.
        let census = AllocationCensus::classify(&local, 0).unwrap();
        assert_eq!(census.stores, [store(device, 1_000)]);
    }
}
