//! The allocation census a loaded engine reports (integration spec §9.2):
//! the memory heap's observed standing, per memory domain, in the categories
//! the service publishes. Because memory is elastic these are the standing at
//! the observation, not a fixed reservation.

use magnitude_executor::memory::{Holding, HoldingClass, MemoryStanding};
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
    /// Model class: target weights, sealed resources and the state seed.
    pub model_bytes: u64,
    /// Live, retained, surplus and in-flight state.
    pub context_bytes: u64,
    /// Prepared program workspace and graph pools.
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
    /// the heap's standing and holdings. Compute bytes are the plan's graph
    /// pools, which are fixed at startup; the model class is the remainder
    /// of the charge, so every charged byte is reported exactly once.
    pub(crate) fn classify(
        standing: &MemoryStanding,
        holdings: impl IntoIterator<Item = Holding>,
        compute_bytes: u64,
        domain: MemoryDomain,
    ) -> Result<Self, String> {
        let mut context_bytes = 0u64;
        let mut auxiliary_bytes = 0u64;
        for holding in holdings {
            let total = match holding.class {
                HoldingClass::Live
                | HoldingClass::Retained
                | HoldingClass::Surplus
                | HoldingClass::InFlight => &mut context_bytes,
                HoldingClass::Dormant => &mut auxiliary_bytes,
                HoldingClass::Model => continue,
            };
            *total = total
                .checked_add(holding.bytes)
                .ok_or("census byte count overflow")?;
        }
        let model_bytes = standing
            .observation
            .charged_bytes
            .checked_sub(context_bytes)
            .and_then(|bytes| bytes.checked_sub(auxiliary_bytes))
            .and_then(|bytes| bytes.checked_sub(compute_bytes))
            .ok_or("classified memory exceeds the domain's charge")?;
        Ok(Self {
            domains: vec![DomainAllocation {
                domain,
                model_bytes,
                context_bytes,
                compute_bytes,
                auxiliary_bytes,
            }],
        })
    }
}
