//! The engine heap of one opened device: the single claim path from the first
//! startup allocation to unload.

use crate::memory::{
    ClaimId, HoldingClass, MemoryAction, MemoryBand, MemoryError, MemoryHeap, MemoryNeed,
    MemoryObservation,
};
use crate::platform::{
    DomainReading, DomainRole, MemoryBand as ReadingBand, MemoryConstraint, MemoryPolicyError,
    MemoryReserves,
};
use seismic::{Device, DeviceCatalog};
use std::fmt;
use std::rc::Rc;

/// Why the heap refused a claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClaimRefusal {
    /// A required observation failed; missing observations never authorize
    /// an allocation.
    Blind(MemoryPolicyError),
    /// A domain's headroom is at or below its planning reserve: growth waits
    /// while reclaimable holdings are released. `role` is the first such
    /// domain.
    Reclaim { role: DomainRole },
    /// The claim exceeds a domain's ceiling above its planning reserve.
    Deficit {
        role: DomainRole,
        constraint: MemoryConstraint,
        required: u64,
        available: u64,
    },
    /// The classified holdings no longer reconcile with Seismic's charge.
    Accounting(MemoryError),
}

impl fmt::Display for ClaimRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Blind(error) => write!(formatter, "memory observation unavailable: {error}"),
            Self::Reclaim { role } => write!(
                formatter,
                "{role:?} memory headroom is at or below the planning reserve"
            ),
            Self::Deficit {
                constraint,
                required,
                available,
                ..
            } => write!(
                formatter,
                "requires {required} bytes of {constraint}; {available} are available above the \
                 planning reserve"
            ),
            Self::Accounting(error) => write!(formatter, "memory accounting: {error:?}"),
        }
    }
}

impl std::error::Error for ClaimRefusal {}

/// The heap of one opened device together with the host's threshold policy
/// and the catalog that observes the domains the device uses. It exists
/// before the first startup allocation, so loading and serving claim through
/// one authority: every allocation's peak is claimed from a fresh reading,
/// and Seismic's enforced limit is refreshed from the same reading.
pub struct DeviceHeap {
    catalog: DeviceCatalog,
    reserves: MemoryReserves,
    device: Rc<Device>,
    capacity_bytes: u64,
    heap: MemoryHeap,
}

impl DeviceHeap {
    /// The heap of `device`, observed once. The allocation domain's stable
    /// capacity is fixed here.
    pub fn open(
        catalog: DeviceCatalog,
        reserves: MemoryReserves,
        device: Rc<Device>,
    ) -> Result<Self, MemoryPolicyError> {
        let host = catalog
            .host_memory_status()
            .map_err(MemoryPolicyError::Observation)?;
        let capacity_bytes =
            crate::platform::fit_capacities(&catalog.topology(), device.info(), &host, &reserves)?
                .into_iter()
                .find(|(role, _)| *role == DomainRole::Allocation)
                .expect("fit capacities include the allocation domain")
                .1
                .capacity_bytes;
        let mut heap = Self {
            catalog,
            reserves,
            device,
            capacity_bytes,
            heap: MemoryHeap::new(),
        };
        heap.refresh()?;
        Ok(heap)
    }

    pub fn device(&self) -> &Rc<Device> {
        &self.device
    }

    pub fn heap(&self) -> &MemoryHeap {
        &self.heap
    }

    /// The classified holdings. Callers add or resize a holding only after
    /// the physical operation succeeded.
    pub(crate) fn heap_mut(&mut self) -> &mut MemoryHeap {
        &mut self.heap
    }

    /// Observe every domain the device uses, set Seismic's allocation
    /// ceiling from the readings, and refresh the heap's view from them.
    /// A failed observation revokes unspent grants and leaves the heap Blind.
    pub fn refresh(&mut self) -> Result<Vec<DomainReading>, MemoryPolicyError> {
        let readings =
            crate::platform::refresh_device_ceiling(&self.catalog, &self.device, &self.reserves);
        let (available_bytes, band) = match &readings {
            Ok(readings) => (
                readings
                    .iter()
                    .find(|reading| reading.role == DomainRole::Allocation)
                    .expect("readings include the allocation domain")
                    .ceiling_bytes,
                MemoryBand::from(crate::platform::band_of(readings)),
            ),
            Err(_) => (0, MemoryBand::Blind),
        };
        self.heap
            .observe(MemoryObservation {
                capacity_bytes: self.capacity_bytes,
                available_bytes,
                charged_bytes: self.device.memory_usage().charged,
                band,
            })
            .expect("a Seismic reading of the allocation domain is a valid observation");
        readings
    }

    /// Whether a peak of `required` device bytes, of which a dedicated device
    /// stages `staged` through host RAM, fits now: from a fresh reading, in
    /// the Normal band, within every domain's ceiling beyond outstanding
    /// claims. The readings exclude existing charges, so only the added peak
    /// is compared.
    pub fn check(&mut self, required: u64, staged: u64) -> Result<(), ClaimRefusal> {
        let readings = self.refresh().map_err(ClaimRefusal::Blind)?;
        let allocation = readings
            .iter()
            .find(|reading| reading.role == DomainRole::Allocation)
            .expect("readings include the allocation domain");
        let action = self
            .heap
            .decide(MemoryNeed {
                minimum_bytes: required,
                preferred_bytes: required,
                class: HoldingClass::Surplus,
            })
            .map_err(ClaimRefusal::Accounting)?;
        match action {
            // The heap's band is the readings' band, so some reading is in
            // Reclaim.
            MemoryAction::Wait => {
                let reclaiming = readings
                    .iter()
                    .find(|reading| reading.band == ReadingBand::Reclaim)
                    .expect("a Reclaim band has a domain at or below its planning reserve");
                return Err(ClaimRefusal::Reclaim {
                    role: reclaiming.role,
                });
            }
            MemoryAction::Grant { bytes } if bytes >= required => {}
            MemoryAction::Grant { bytes: available } | MemoryAction::Reject { available, .. } => {
                return Err(ClaimRefusal::Deficit {
                    role: DomainRole::Allocation,
                    constraint: allocation.constraint,
                    required,
                    available,
                })
            }
        }
        // A host-backed device has no staging domain: its staged bytes are
        // part of `required` on the same host domain.
        if let Some(staging) = readings
            .iter()
            .find(|reading| reading.role == DomainRole::Staging)
        {
            if staged > staging.ceiling_bytes {
                return Err(ClaimRefusal::Deficit {
                    role: DomainRole::Staging,
                    constraint: staging.constraint,
                    required: staged,
                    available: staging.ceiling_bytes,
                });
            }
        }
        Ok(())
    }

    /// Claim the peak new charge of one physical operation. The claim holds
    /// its bytes against every later decision until the caller releases it,
    /// once Seismic's charge reflects the operation.
    pub fn claim(
        &mut self,
        required: u64,
        staged: u64,
        class: HoldingClass,
    ) -> Result<ClaimId, ClaimRefusal> {
        self.check(required, staged)?;
        self.heap
            .claim(MemoryNeed {
                minimum_bytes: required,
                preferred_bytes: required,
                class,
            })
            .map(|claim| claim.id)
            .map_err(ClaimRefusal::Accounting)
    }

    pub fn release(&mut self, claim: ClaimId) {
        self.heap
            .cancel_claim(claim)
            .expect("a claim is released once");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::BackendName;

    fn cpu_heap() -> DeviceHeap {
        let catalog = DeviceCatalog::discover().unwrap();
        let device = Rc::new(catalog.open_backend(BackendName::Cpu).unwrap());
        DeviceHeap::open(catalog, MemoryReserves::standard(), device).unwrap()
    }

    #[test]
    fn claims_hold_their_peak_against_later_decisions_until_released() {
        let mut heap = cpu_heap();
        let ceiling = heap.heap().standing().unwrap().observation.available_bytes;
        assert!(
            ceiling > 1024,
            "the test host has headroom above its reserve"
        );
        let claim = heap.claim(ceiling / 2, 0, HoldingClass::Live).unwrap();
        assert_eq!(heap.heap().claims().count(), 1);
        // A fresh reading of the same host now leaves only about the
        // unclaimed half of the ceiling.
        assert!(matches!(
            heap.check(ceiling / 4 * 3, 0),
            Err(ClaimRefusal::Deficit {
                role: DomainRole::Allocation,
                constraint: MemoryConstraint::HostRam,
                ..
            })
        ));
        heap.release(claim);
        assert_eq!(heap.heap().claims().count(), 0);
        assert_eq!(heap.check(0, 0), Ok(()));
        assert!(matches!(
            heap.claim(u64::MAX / 2, 0, HoldingClass::Live),
            Err(ClaimRefusal::Deficit { required, .. }) if required == u64::MAX / 2
        ));
        assert_eq!(heap.heap().claims().count(), 0);
    }
}
