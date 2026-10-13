//! The allocation census a loaded engine reports (integration spec §9.2):
//! the memory heap's observed standing, per memory domain, in the categories
//! the service publishes. Because memory is elastic these are the standing at
//! the observation, not a fixed reservation.

use magnitude_executor::MemoryChargeReconciliation;
use seismic::{DeviceSelector, MemoryPoolKind};
use serde::{Deserialize, Serialize};

/// A Seismic memory pool, named by a cross-process identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryDomain {
    /// Host RAM (also the allocation domain of unified-memory devices).
    HostRam,
    /// A dedicated device's local memory.
    DeviceLocal { device: DeviceSelector },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DomainAllocation {
    pub domain: MemoryDomain,
    /// Model class: target weights, bound constants, the pristine recurrent
    /// seed and host-resident tables.
    pub model_bytes: u64,
    /// Per-conversation state: attention history and recurrent banks that
    /// are live, retained for reuse, in flight or committed headroom, and
    /// request media.
    pub context_bytes: u64,
    /// Prepared programs and graph pools, as committed, and any charge the
    /// reconciliation has not attributed to a holder.
    pub compute_bytes: u64,
    /// Optional components (MTP head, vision), resident or dormant.
    pub auxiliary_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocationCensus {
    pub domains: Vec<DomainAllocation>,
}

impl std::fmt::Display for MemoryDomain {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HostRam => formatter.write_str("host RAM"),
            Self::DeviceLocal { device } => write!(formatter, "{device} device memory"),
        }
    }
}

impl MemoryDomain {
    /// The allocation domain of a device whose allocation pool is `pool`.
    pub(crate) fn of(device: DeviceSelector, pool: MemoryPoolKind) -> Self {
        match pool {
            MemoryPoolKind::HostRam => Self::HostRam,
            MemoryPoolKind::DeviceLocal => Self::DeviceLocal { device },
        }
    }
}

/// The model's bytes in host RAM beside the device's allocation domain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct HostModelBytes {
    /// Host-resident gathered tables, which Seismic does not charge.
    pub tables: u64,
    /// Stored tables' host-shared storage on a dedicated device whose
    /// shared storage is host RAM (`platform::shared_role`): Seismic charges
    /// it with the device's allocations, and the census reports it in host
    /// RAM, where it is.
    pub stored_tables: u64,
}

impl AllocationCensus {
    /// Classify the device allocation domain's current Seismic charge from
    /// its reconciliation against every holder, so every charged byte is
    /// reported exactly once. Storage the reconciliation has not attributed
    /// to a holder is reported with the compute class. The model's
    /// host-resident tables, and its stored tables where they are host RAM
    /// of a dedicated device, are model bytes of host RAM.
    pub(crate) fn classify(
        charge: &MemoryChargeReconciliation,
        host: HostModelBytes,
        domain: MemoryDomain,
    ) -> Result<Self, String> {
        let overflow = || "census byte count overflow".to_owned();
        let state = |census: magnitude_state::StateHoldingCensus| {
            [census.live, census.retained, census.surplus, census.in_flight]
                .into_iter()
                .try_fold(0u64, u64::checked_add)
        };
        let head_state = match charge.head_state {
            Some(census) => state(census).ok_or_else(overflow)?,
            None => 0,
        };
        let context_bytes = [
            state(charge.target_state).ok_or_else(overflow)?,
            head_state,
            charge.owned_features,
            charge.external_pins,
        ]
        .into_iter()
        .try_fold(0u64, u64::checked_add)
        .ok_or_else(overflow)?;
        let compute_bytes = [
            charge.graph_pools,
            charge.prepared_programs,
            charge.unattributed,
        ]
        .into_iter()
        .try_fold(0u64, u64::checked_add)
        .ok_or_else(overflow)?;
        let auxiliary_bytes = charge.optional_weights;
        let model_bytes = [
            charge.target_weights,
            charge.bound_constants,
            charge.target_state.model_seed,
            charge.head_state.map_or(0, |census| census.model_seed),
        ]
        .into_iter()
        .try_fold(0u64, u64::checked_add)
        .ok_or_else(overflow)?;
        let allocation = DomainAllocation {
            domain,
            model_bytes,
            context_bytes,
            compute_bytes,
            auxiliary_bytes,
        };
        let domains = match (domain, host) {
            (_, HostModelBytes { tables: 0, stored_tables: 0 }) => vec![allocation],
            (MemoryDomain::HostRam, HostModelBytes { stored_tables: 0, tables }) => {
                vec![DomainAllocation {
                    model_bytes: model_bytes
                        .checked_add(tables)
                        .ok_or("census byte count overflow")?,
                    ..allocation
                }]
            }
            (MemoryDomain::HostRam, _) => {
                return Err("a host-backed device's stored tables are its own allocations".into());
            }
            (MemoryDomain::DeviceLocal { .. }, host) => vec![
                DomainAllocation {
                    model_bytes: model_bytes
                        .checked_sub(host.stored_tables)
                        .ok_or("stored tables exceed the model's charged bytes")?,
                    ..allocation
                },
                DomainAllocation {
                    domain: MemoryDomain::HostRam,
                    model_bytes: host
                        .tables
                        .checked_add(host.stored_tables)
                        .ok_or("census byte count overflow")?,
                    context_bytes: 0,
                    compute_bytes: 0,
                    auxiliary_bytes: 0,
                },
            ],
        };
        Ok(Self { domains })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_state::StateHoldingCensus;

    fn charge(target_weights: u64) -> MemoryChargeReconciliation {
        MemoryChargeReconciliation {
            charged: target_weights,
            target_state: StateHoldingCensus::default(),
            head_state: None,
            graph_pools: 0,
            owned_features: 0,
            target_weights,
            optional_weights: 0,
            prepared_programs: 0,
            bound_constants: 0,
            external_pins: 0,
            unattributed: 0,
        }
    }

    /// A dedicated device whose host-shared storage is host RAM reports its
    /// stored tables there, beside the host tables, though Seismic charges
    /// them with the device's weights; a host-backed device reports one
    /// domain.
    #[test]
    fn stored_tables_in_host_ram_are_reported_there() {
        let device = MemoryDomain::DeviceLocal {
            device: DeviceSelector::Cuda { uuid: [3; 16] },
        };
        let host = HostModelBytes {
            tables: 10,
            stored_tables: 300,
        };
        let census = AllocationCensus::classify(&charge(1_000), host, device).unwrap();
        assert_eq!(census.domains.len(), 2);
        assert_eq!(census.domains[0].domain, device);
        assert_eq!(census.domains[0].model_bytes, 700);
        assert_eq!(census.domains[1].domain, MemoryDomain::HostRam);
        assert_eq!(census.domains[1].model_bytes, 310);
        assert!(
            AllocationCensus::classify(
                &charge(100),
                HostModelBytes {
                    tables: 0,
                    stored_tables: 300,
                },
                device,
            )
            .is_err()
        );
        let unified = AllocationCensus::classify(
            &charge(1_000),
            HostModelBytes {
                tables: 10,
                stored_tables: 0,
            },
            MemoryDomain::HostRam,
        )
        .unwrap();
        assert_eq!(unified.domains.len(), 1);
        assert_eq!(unified.domains[0].model_bytes, 1_010);
    }
}
