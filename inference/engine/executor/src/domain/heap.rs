//! The engine heap of one opened device: the single claim path from the first
//! startup allocation to unload.

use crate::memory::{
    ClaimDemand, ClaimId, HoldingClass, MemoryError, MemoryHeap, MemoryObservation,
};
use crate::platform::{
    DomainReading, DomainRole, Fit, FitMode, HostDistress, MemoryBand, MemoryConstraint,
    MemoryPolicyError, MemoryReserves, StorageDemand,
};
use seismic::{Device, Devices, StorageKind};
use std::fmt;
use std::rc::Rc;

/// Why the heap refused a claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClaimRefusal {
    /// A required observation failed; missing observations never authorize
    /// an allocation.
    Blind(MemoryPolicyError),
    /// A domain is in the Reclaim band: growth waits while reclaimable
    /// holdings are released. `store` is the first such domain; `distress`
    /// is its host's, absent when headroom at or below the planning reserve
    /// is the cause.
    Reclaim {
        store: seismic::StoreId,
        distress: Option<HostDistress>,
    },
    /// The claim exceeds a domain's ceiling above its planning reserve.
    Deficit {
        role: DomainRole,
        store: seismic::StoreId,
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
            Self::Reclaim { distress: None, .. } => {
                formatter.write_str("memory headroom is at or below the planning reserve")
            }
            Self::Reclaim {
                distress: Some(distress),
                ..
            } => distress.fmt(formatter),
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

/// The refusal of the first domain in the Reclaim band, if any is.
fn reclaiming(readings: &[DomainReading]) -> Option<ClaimRefusal> {
    readings
        .iter()
        .find(|reading| reading.band == MemoryBand::Reclaim)
        .map(|reading| ClaimRefusal::Reclaim {
            store: reading.domain,
            distress: reading.distress,
        })
}

/// Whether every domain `device` uses is in the Normal band now, from a fresh
/// reading. A physical operation that runs long under one claim asks this as
/// it proceeds: its own allocations can put the host in distress after the
/// claim was granted, and other programs can take the headroom.
pub fn require_normal(
    catalog: &Devices,
    device: &Device,
    reserves: &MemoryReserves,
) -> Result<(), ClaimRefusal> {
    let readings =
        crate::platform::observe_domains(catalog, device, reserves).map_err(ClaimRefusal::Blind)?;
    reclaiming(&readings).map_or(Ok(()), Err)
}

/// The new storage `total` asks of `executor`'s placements: each kind's
/// bytes beyond `held_free`, the free space of the blocks the device already
/// holds for it ([`DeviceHeap::held_free`]), which a claim takes without the
/// device holding more.
fn new_storage(
    executor: seismic::ExecutorId,
    total: ClaimDemand,
    held_free: impl Fn(StorageKind) -> u64,
) -> [StorageDemand; 2] {
    [
        (StorageKind::Device, total.device),
        (StorageKind::Upload, total.upload),
    ]
    .map(|(kind, bytes)| StorageDemand {
        executor,
        kind,
        bytes: bytes.saturating_sub(held_free(kind)),
    })
}

/// The heap of one opened device together with the host's threshold policy
/// and the catalog that observes the domains the device uses. It exists
/// before the first startup allocation, so loading and serving claim through
/// one authority: every allocation's peak is claimed from a fresh reading,
/// and Seismic's enforced limit is refreshed from the same reading.
pub struct DeviceHeap {
    catalog: Rc<Devices>,
    reserves: MemoryReserves,
    device: Rc<Device>,
    capacity_bytes: u64,
    heap: MemoryHeap,
    /// Host-resident gathered tables' bytes (§3.7 of the model-family plan):
    /// mapped pages the model reads every step, held in the host-RAM domain
    /// for the model's lifetime and never counted as free page cache.
    host_table_bytes: u64,
}

impl DeviceHeap {
    /// The heap of `device`, observed once. Its capacity, the stable fit of
    /// the device's storage ([`MemoryObservation::capacity_bytes`]), is fixed
    /// here.
    pub fn open(
        catalog: Rc<Devices>,
        reserves: MemoryReserves,
        device: Rc<Device>,
    ) -> Result<Self, MemoryPolicyError> {
        let status = catalog.status();
        let capacity_bytes = Fit::new(device.topology(), &status, &reserves)
            .storage_beside(device.info(), &StorageKind::ALL, &[])?
            .stable_bytes;
        let mut heap = Self {
            catalog,
            reserves,
            device,
            capacity_bytes,
            heap: MemoryHeap::new(),
            host_table_bytes: 0,
        };
        heap.observe(&status)?;
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
        let status = self.catalog.status();
        self.observe(&status)
    }

    /// [`Self::refresh`] from the status sample `status`.
    fn observe(
        &mut self,
        status: &seismic::Status,
    ) -> Result<Vec<DomainReading>, MemoryPolicyError> {
        let readings =
            crate::platform::refresh_device_ceiling(&self.device, status, &self.reserves);
        let charged_bytes = StorageKind::ALL
            .into_iter()
            .map(|kind| self.device.storage_usage(kind).charged)
            .sum();
        self.heap
            .observe(MemoryObservation {
                capacity_bytes: self.capacity_bytes,
                charged_bytes,
            })
            .expect("a Seismic reading of the device's storage is a valid observation");
        readings
    }

    /// The free space of the blocks the device holds for `kind` storage
    /// ([`seismic::MemoryUsage::unallocated`]): memory its stores already
    /// lost to the device, so absent from their headroom, that a claim
    /// takes without the device holding more. A claim within it needs no
    /// new room in any store.
    fn held_free(&self, kind: StorageKind) -> u64 {
        self.device.storage_usage(kind).unallocated
    }

    /// Whether a peak of `required` device bytes, `staged` of which the
    /// backend stages through host RAM, fits now: from a fresh reading, in
    /// the Normal band, within every domain's ceiling beyond outstanding
    /// claims. The readings exclude existing charges, so only the added peak
    /// is compared.
    pub fn check(&mut self, required: u64, staged: u64) -> Result<(), ClaimRefusal> {
        self.check_storage(
            ClaimDemand {
                device: required,
                upload: 0,
                staged,
            },
            0,
        )
    }

    /// Whether `claim` and `table` new bytes of host tables fit beside every
    /// outstanding claim and the held host tables, from one fresh status
    /// sample. Fit decides where the claims' staged bytes are charged
    /// ([`Fit::host_staging`]).
    fn check_storage(&mut self, claim: ClaimDemand, table: u64) -> Result<(), ClaimRefusal> {
        let status = self.catalog.status();
        let readings = self.observe(&status).map_err(ClaimRefusal::Blind)?;
        if let Some(refusal) = reclaiming(&readings) {
            return Err(refusal);
        }
        let total = self.heap.claimed().plus(claim);
        let topology = self.device.topology();
        let info = self.device.info();
        let executor = info.id();
        let fit = Fit::new(topology, &status, &self.reserves);
        let host = fit
            .host_staging(info, total.staged)
            .saturating_add(self.host_table_bytes)
            .saturating_add(table);
        let demand = fit
            .demand(
                &new_storage(executor, total, |kind| self.held_free(kind)),
                &[(topology.host_ram().id(), host)],
                FitMode::Live,
            )
            .map_err(ClaimRefusal::Blind)?;
        if let Some(refusal) = demand.refused.first() {
            let role = readings
                .iter()
                .find(|reading| reading.domain == refusal.store)
                .expect("demand stores are observed")
                .role;
            return Err(ClaimRefusal::Deficit {
                role,
                store: refusal.store,
                constraint: refusal.constraint,
                required: refusal.required_bytes,
                available: refusal.required_bytes.saturating_sub(refusal.missing_bytes),
            });
        }
        Ok(())
    }

    /// Hold a host-resident table's `bytes` in host RAM for the model's
    /// lifetime, from a fresh reading within host RAM's ceiling above its
    /// planning reserve, beside every outstanding claim and every storage
    /// placement into host RAM. Its mapped pages then count against every
    /// later decision there.
    pub fn hold_host_table(&mut self, bytes: u64) -> Result<(), ClaimRefusal> {
        self.check_storage(ClaimDemand::default(), bytes)?;
        self.host_table_bytes = self
            .host_table_bytes
            .checked_add(bytes)
            .ok_or(ClaimRefusal::Accounting(MemoryError::HoldingOverflow))?;
        self.refresh().map_err(ClaimRefusal::Blind)?;
        Ok(())
    }

    /// Bytes of the host-resident tables held in the host-RAM domain.
    pub fn host_table_bytes(&self) -> u64 {
        self.host_table_bytes
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
        self.claim_storage(required, 0, staged, class)
    }

    /// [`Self::claim`] with `upload` bytes of upload storage beside the
    /// `required` device bytes.
    pub fn claim_storage(
        &mut self,
        required: u64,
        upload: u64,
        staged: u64,
        class: HoldingClass,
    ) -> Result<ClaimId, ClaimRefusal> {
        let demand = ClaimDemand {
            device: required,
            upload,
            staged,
        };
        self.check_storage(demand, 0)?;
        self.heap
            .claim(demand, class)
            .map(|claim| claim.id)
            .map_err(ClaimRefusal::Accounting)
    }

    /// Release `claim`: its physical operation's charge is now Seismic's.
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

    /// On the RTX 3090, a claim beyond what the device store's headroom
    /// leaves above its reserve fits when the blocks the device holds have
    /// the rest free, and is refused by the store when they do not.
    #[test]
    fn held_free_block_space_admits_a_claim_beyond_the_store_headroom() {
        const GIB: u64 = 1 << 30;
        let topology = seismic::fixtures::recorded::rtx3090();
        let cuda = topology
            .executors()
            .iter()
            .find(|executor| executor.backend() == BackendName::Cuda)
            .unwrap();
        let path = topology
            .access_path(cuda.placement(StorageKind::Device)[0])
            .unwrap();
        let gpu = topology.store(path.store()).unwrap();
        let mut status = seismic::Status {
            sample: seismic::Sample {
                taken: std::time::SystemTime::now(),
                host_ram: Ok(seismic::HostStatus {
                    headroom: seismic::Reported::new(
                        16 * GIB,
                        seismic::ReportedBy::LinuxMemAvailable,
                    ),
                    limits: Vec::new(),
                    limit_visibility: seismic::LimitVisibility::Complete,
                    displacement: None,
                    kernel_pressure: None,
                }),
                volumes: Vec::new(),
                grants: Vec::new(),
            },
            runtime: seismic::RuntimeStatus::default(),
        };
        // The device's blocks leave 4 GiB of its memory as headroom.
        status.runtime.charges = vec![(gpu.id(), gpu.capacity().value - 4 * GIB)];
        let reserves = MemoryReserves::standard();
        let fit = Fit::new(&topology, &status, &reserves);
        let ceiling = fit.store(gpu.id()).unwrap().ceiling_bytes;
        let claim = ClaimDemand {
            device: ceiling + GIB,
            upload: 0,
            staged: 0,
        };
        let demand = |held: u64| {
            let storage = new_storage(cuda.id(), claim, |kind| match kind {
                StorageKind::Device => held,
                StorageKind::Upload => 0,
            });
            fit.demand(&storage, &[], FitMode::Live).unwrap()
        };
        let refused = demand(0);
        assert_eq!(refused.refused.len(), 1);
        assert_eq!(refused.refused[0].store, gpu.id());
        assert!(demand(2 * GIB).fits());
    }

    /// The live ceiling of `heap`'s device storage beside its held host
    /// tables, from a fresh status sample.
    fn live_ceiling(heap: &DeviceHeap) -> u64 {
        let topology = heap.device.topology();
        Fit::new(topology, &heap.catalog.status(), &heap.reserves)
            .storage_beside(
                heap.device.info(),
                &StorageKind::ALL,
                &[(topology.host_ram().id(), heap.host_table_bytes)],
            )
            .unwrap()
            .ceiling_bytes
    }

    fn cpu_heap() -> DeviceHeap {
        let catalog = Rc::new(Devices::discover(&[]).unwrap());
        let device = Rc::new(catalog.open_backend(BackendName::Cpu).unwrap());
        DeviceHeap::open(catalog, MemoryReserves::standard(), device).unwrap()
    }

    #[test]
    fn a_host_in_the_normal_band_lets_a_running_operation_continue() {
        let catalog = Devices::discover(&[]).unwrap();
        let device = catalog.open_backend(BackendName::Cpu).unwrap();
        assert_eq!(
            require_normal(&catalog, &device, &MemoryReserves::standard()),
            Ok(())
        );
    }

    #[test]
    fn claims_hold_their_peak_against_later_decisions_until_released() {
        let mut heap = cpu_heap();
        let ceiling = live_ceiling(&heap);
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

    /// A held host table counts against every later decision in the host
    /// domain (the CPU device's allocation domain), and a table beyond the
    /// ceiling is refused.
    #[test]
    fn host_tables_hold_their_bytes_in_the_host_domain() {
        let mut heap = cpu_heap();
        let ceiling = live_ceiling(&heap);
        assert!(matches!(
            heap.hold_host_table(u64::MAX / 2),
            Err(ClaimRefusal::Deficit {
                role: DomainRole::Allocation,
                ..
            })
        ));
        assert_eq!(heap.host_table_bytes(), 0);
        heap.hold_host_table(ceiling / 2).unwrap();
        assert_eq!(heap.host_table_bytes(), ceiling / 2);
        assert!(live_ceiling(&heap) <= ceiling - ceiling / 2 + ceiling / 8);
        assert!(matches!(
            heap.check(ceiling / 4 * 3, 0),
            Err(ClaimRefusal::Deficit {
                role: DomainRole::Allocation,
                ..
            })
        ));
    }
}
