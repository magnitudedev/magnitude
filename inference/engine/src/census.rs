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
