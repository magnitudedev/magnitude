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
    /// Prepared programs, graph pools and activation handoff storage, as
    /// committed, and any charge the
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

impl AllocationCensus {
    /// Classify the device allocation domain's current Seismic charge from
    /// its reconciliation against every holder, so every charged byte is
    /// reported exactly once. Storage the reconciliation has not attributed
    /// to a holder is reported with the compute class. The model's
    /// host-resident tables are model bytes of host RAM.
    pub(crate) fn classify(
        charge: &MemoryChargeReconciliation,
        host_table_bytes: u64,
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
            charge.owned_media,
            charge.external_pins,
        ]
        .into_iter()
        .try_fold(0u64, u64::checked_add)
        .ok_or_else(overflow)?;
        let compute_bytes = [
            charge.graph_pools,
            charge.activation_transfer,
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
        let domains = match (domain, host_table_bytes) {
            (_, 0) => vec![allocation],
            (MemoryDomain::HostRam, tables) => vec![DomainAllocation {
                model_bytes: model_bytes
                    .checked_add(tables)
                    .ok_or("census byte count overflow")?,
                ..allocation
            }],
            (MemoryDomain::DeviceLocal { .. }, tables) => vec![
                allocation,
                DomainAllocation {
                    domain: MemoryDomain::HostRam,
                    model_bytes: tables,
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

    fn charge(transfer: u64) -> MemoryChargeReconciliation {
        MemoryChargeReconciliation {
            charged: 100 + transfer,
            target_state: magnitude_state::StateHoldingCensus::default(),
            head_state: None,
            graph_pools: 20,
            activation_transfer: transfer,
            owned_media: 0,
            target_weights: 60,
            optional_weights: 0,
            prepared_programs: 10,
            bound_constants: 10,
            external_pins: 0,
            unattributed: 0,
        }
    }

    #[test]
    fn activation_transfer_is_classified_once_as_compute() {
        let ordinary = AllocationCensus::classify(&charge(0), 0, MemoryDomain::HostRam).unwrap();
        let paired = AllocationCensus::classify(&charge(32), 0, MemoryDomain::HostRam).unwrap();
        let ordinary = ordinary.domains[0];
        let paired = paired.domains[0];
        assert_eq!(ordinary.model_bytes, 70);
        assert_eq!(ordinary.compute_bytes, 30);
        assert_eq!(paired.model_bytes, ordinary.model_bytes);
        assert_eq!(paired.context_bytes, ordinary.context_bytes);
        assert_eq!(paired.auxiliary_bytes, ordinary.auxiliary_bytes);
        assert_eq!(paired.compute_bytes, ordinary.compute_bytes + 32);
        assert_eq!(
            paired.model_bytes
                + paired.context_bytes
                + paired.compute_bytes
                + paired.auxiliary_bytes,
            charge(32).charged
        );
    }

    #[test]
    fn activation_transfer_classification_rejects_overflow() {
        let mut charge = charge(0);
        charge.activation_transfer = u64::MAX;
        assert!(AllocationCensus::classify(&charge, 0, MemoryDomain::HostRam).is_err());
    }
}
