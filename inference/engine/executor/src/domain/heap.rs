//! The engine heap of one opened device: the single claim path from the first
//! startup allocation to unload.

use crate::memory::{
    ClaimDemand, ClaimId, HoldingClass, MemoryError, MemoryHeap, MemoryObservation,
};
use crate::platform::{
    DomainReading, DomainRole, Fit, FitMode, HostRoom, LoadStage, MemoryBand, MemoryConstraint,
    MemoryPolicyError, MemoryReserves, StorageDemand,
};
use crate::memory::CLAIMED;
use crate::planning::BankTier;
use seismic::{Contract, Device, Devices};
use std::fmt;
use std::rc::Rc;

/// Why the heap refused a claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClaimRefusal {
    /// A required observation failed; missing observations never authorize
    /// an allocation.
    Blind(MemoryPolicyError),
    /// A domain is in the Reclaim band: growth waits while reclaimable
    /// holdings are released. The reading of the first such domain: its
    /// host's distress, absent when headroom at or below the planning
    /// reserve is the cause.
    Reclaim(DomainReading),
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
            Self::Reclaim(reading) => Reclaimed(reading).fmt(formatter),
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

/// Why a domain in the Reclaim band refuses growth, with the reading it was
/// judged on.
pub struct Reclaimed<'a>(pub &'a DomainReading);

impl fmt::Display for Reclaimed<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let reading = self.0;
        match reading.distress {
            Some(distress) => write!(formatter, "{distress} ({reading})"),
            None => write!(
                formatter,
                "memory headroom is at or below the planning reserve: {reading}"
            ),
        }
    }
}

/// The refusal of the first domain in the Reclaim band, if any is.
fn reclaiming(readings: &[DomainReading]) -> Option<ClaimRefusal> {
    readings
        .iter()
        .find(|reading| reading.band == MemoryBand::Reclaim)
        .cloned()
        .map(ClaimRefusal::Reclaim)
}

/// Whether every domain `device` uses is in the Normal band now, from a fresh
/// reading of host RAM's `room` (the load's heap's, [`DeviceHeap::room`]). A
/// physical operation that runs long under one claim asks this as it
/// proceeds: its own allocations can put the host in distress after the
/// claim was granted, and other programs can take the headroom.
pub fn require_normal(
    catalog: &Devices,
    device: &Device,
    reserves: &MemoryReserves,
    room: HostRoom,
) -> Result<(), ClaimRefusal> {
    let readings = crate::platform::observe_domains(catalog, device, reserves, room).map_err(ClaimRefusal::Blind)?;
    reclaiming(&readings).map_or(Ok(()), Err)
}

/// The new storage `total` asks of `executor`'s placements: each kind's
/// allocated bytes beyond `held_free`, the free space of the blocks the
/// device already holds for it ([`DeviceHeap::held_free`]), which an
/// allocation takes without the device holding more, and every reserved
/// byte, which a reservation backs with fresh pages.
fn new_storage(
    executor: seismic::ExecutorId,
    total: ClaimDemand,
    held_free: impl Fn(Contract) -> u64,
) -> [StorageDemand; CLAIMED.len()] {
    [
        (Contract::DeviceOnly, total.device, total.reserved),
        (Contract::Upload, total.upload, 0),
        (Contract::Shared, total.shared, total.host),
    ]
    .map(|(kind, allocated, reserved)| StorageDemand {
        executor,
        kind,
        bytes: allocated
            .saturating_sub(held_free(kind))
            .saturating_add(reserved),
    })
}

/// The demand of `bytes` a reservation of `kind` storage backs: device
/// storage's fresh pages of the device's store, host storage's (a host
/// tier's) host RAM.
fn reserved(kind: BankTier, bytes: u64) -> ClaimDemand {
    match kind {
        BankTier::Bank => ClaimDemand { reserved: bytes, ..ClaimDemand::default() },
        BankTier::Host => ClaimDemand { host: bytes, ..ClaimDemand::default() },
    }
}

/// `demand` tested with `beside` more device bytes, which the claim does
/// not hold.
fn besides(demand: ClaimDemand, beside: u64) -> Result<ClaimDemand, ClaimRefusal> {
    Ok(ClaimDemand {
        device: demand
            .device
            .checked_add(beside)
            .ok_or(ClaimRefusal::Accounting(MemoryError::HoldingOverflow))?,
        ..demand
    })
}

/// What the heap reads of the machine: one status sample a reading, every
/// domain the device uses observed in it, of the device's executor's grants
/// alone: no other executor's grant is read. Seismic's catalog reads the
/// live machine; a test gives the readings it decides on.
pub trait StatusReading {
    /// A sample with host RAM read now: what a claim, and every observation
    /// between claims ([`DeviceHeap::refresh`]), is judged against.
    fn status(&self, device: &Device) -> seismic::Status;

    /// The process's footprint in host RAM read now: what the system counts
    /// it holding there, which witnesses the releases of a device whose
    /// allocations are pages of the process (`ExecutorDomain::host_witness`).
    fn footprint(&self) -> Option<u64>;
}

impl StatusReading for Devices {
    fn status(&self, device: &Device) -> seismic::Status {
        Devices::status(self, device.topology(), device.info().id())
    }

    fn footprint(&self) -> Option<u64> {
        seismic::ProcessMemory::footprint_now()
    }
}

/// The heap of one opened device together with the host's threshold policy
/// and the reading that observes the domains the device uses. It exists
/// before the first startup allocation, so loading and serving claim through
/// one authority: every allocation's peak is claimed from a fresh reading,
/// and Seismic's enforced limit is refreshed from the same reading.
pub struct DeviceHeap {
    reading: Rc<dyn StatusReading>,
    reserves: MemoryReserves,
    device: Rc<Device>,
    capacity_bytes: u64,
    heap: MemoryHeap,
    /// Host-resident gathered tables' bytes (§3.7 of the model-family plan):
    /// mapped pages the model reads every step, held in the host-RAM domain
    /// for the model's lifetime and never counted as free page cache.
    host_table_bytes: u64,
    /// The host RAM room its band, claims and Seismic's limits read, and
    /// its growth's: [`HostRoom::Obtainable`] from the load's opening on,
    /// its growth's displacement counted from when it serves
    /// ([`Self::serve`]).
    room: HostRoom,
}

impl DeviceHeap {
    /// The heap of `device`, observed once, for a load
    /// ([`HostRoom::Obtainable`], accepting its own displacement). Its
    /// capacity, the stable fit of the device's storage
    /// ([`MemoryObservation::capacity_bytes`]), is fixed here.
    pub fn open(
        reading: Rc<dyn StatusReading>,
        reserves: MemoryReserves,
        device: Rc<Device>,
    ) -> Result<Self, MemoryPolicyError> {
        let status = reading.status(&device);
        let room = HostRoom::Obtainable(LoadStage::Loading);
        let capacity_bytes = Fit::new(device.topology(), &status, &reserves, room)
            .storage_beside(device.info(), &CLAIMED, &[])?
            .stable_bytes;
        let mut heap = Self {
            reading,
            reserves,
            device,
            capacity_bytes,
            heap: MemoryHeap::new(),
            host_table_bytes: 0,
            room,
        };
        heap.observe(&status)?;
        Ok(heap)
    }

    /// The model has loaded: the displacement its load caused is accepted,
    /// and from now on its growth reads what can be taken without displacing
    /// any more of another program's data than is displaced now
    /// ([`HostRoom::Obtainable`]). Its band and claims read the same room as
    /// they did while it loaded.
    pub fn serve(&mut self) -> Result<(), MemoryPolicyError> {
        let status = self.status();
        let host = status.sample.host_ram.as_ref().map_err(|error| MemoryPolicyError::Observation(error.clone()))?;
        self.room = HostRoom::serving(host);
        self.observe(&status).map(drop)
    }

    /// The host RAM room its readings read ([`Self::room`]'s field).
    pub fn room(&self) -> HostRoom {
        self.room
    }

    pub fn device(&self) -> &Rc<Device> {
        &self.device
    }

    /// A fresh status of the device's executor.
    fn status(&self) -> seismic::Status {
        self.reading.status(&self.device)
    }

    pub fn heap(&self) -> &MemoryHeap {
        &self.heap
    }

    /// The classified holdings. Callers add or resize a holding only after
    /// the physical operation succeeded.
    pub(crate) fn heap_mut(&mut self) -> &mut MemoryHeap {
        &mut self.heap
    }

    /// The process's host RAM footprint now ([`StatusReading::footprint`]).
    pub fn footprint(&self) -> Option<u64> {
        self.reading.footprint()
    }

    /// Observe every domain the device uses from a sample read now, set
    /// Seismic's allocation ceiling from the readings, and refresh the
    /// heap's view from them: a release the process made shows in the next
    /// one, so no decision stands on a reading older than the release before
    /// it. A failed observation revokes unspent grants and leaves the heap
    /// Blind.
    pub fn refresh(&mut self) -> Result<Vec<DomainReading>, MemoryPolicyError> {
        let status = self.reading.status(&self.device);
        self.observe(&status)
    }

    /// [`Self::refresh`] from the status sample `status`.
    fn observe(
        &mut self,
        status: &seismic::Status,
    ) -> Result<Vec<DomainReading>, MemoryPolicyError> {
        let readings =
            crate::platform::refresh_device_ceiling(&self.device, status, &self.reserves, self.room);
        self.account();
        readings
    }

    /// Refresh the heap's view of Seismic's charge alone, reading no store:
    /// what the process's own claims and releases changed since. The stores
    /// and Seismic's ceiling stand as the last [`Self::refresh`] read them.
    pub fn account(&mut self) {
        let charged_bytes = self.device.ledgers().own.iter().map(|(_, ledger)| ledger.charged).sum();
        self.heap
            .observe(MemoryObservation {
                capacity_bytes: self.capacity_bytes,
                charged_bytes,
            })
            .expect("a Seismic reading of the device's storage is a valid observation");
    }

    /// The free space of the blocks the device holds for `kind` storage
    /// ([`seismic::BackendHeld::unallocated`]): memory its stores already
    /// lost to the device, so absent from their headroom, that a claim
    /// takes without the device holding more. A claim within it needs no
    /// new room in any store.
    fn held_free(&self, kind: Contract) -> u64 {
        self.device.backend_held(kind).unallocated
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
                staged,
                ..ClaimDemand::default()
            },
            0,
            FitMode::Live,
        )
    }

    /// Whether `claim` and `table` new bytes of host tables fit beside every
    /// outstanding claim and the held host tables in `mode`'s room, from one
    /// fresh status sample. Fit decides where the claims' staged bytes are
    /// charged ([`Fit::host_staging`]).
    fn check_storage(&mut self, claim: ClaimDemand, table: u64, mode: FitMode) -> Result<(), ClaimRefusal> {
        let (status, readings) = self.sample()?;
        self.refusal(&status, &readings, claim, table, mode)?.map_or(Ok(()), Err)
    }

    /// One fresh status sample, Seismic's limits set from it (the limit a
    /// resize's recommit runs into, `Fit::placement` of each storage kind
    /// at this sample: no tighter than what [`Self::refusal`] admits of the
    /// kind, which also counts every outstanding claim and the host's
    /// direct demands), and its readings; refused while a domain is in the
    /// Reclaim band.
    fn sample(&mut self) -> Result<(seismic::Status, Vec<DomainReading>), ClaimRefusal> {
        let status = self.status();
        let readings = self.observe(&status).map_err(ClaimRefusal::Blind)?;
        match reclaiming(&readings) {
            Some(refusal) => Err(refusal),
            None => Ok((status, readings)),
        }
    }

    /// Why `claim` and `table` new bytes of host tables do not fit beside
    /// every outstanding claim and the held host tables at `status`, whose
    /// domains `readings` observed; none where they fit. The one test of a
    /// claim: [`Self::check_storage`] applies it, and [`Self::reserved_room`]
    /// finds the most it admits.
    fn refusal(
        &self,
        status: &seismic::Status,
        readings: &[DomainReading],
        claim: ClaimDemand,
        table: u64,
        mode: FitMode,
    ) -> Result<Option<ClaimRefusal>, ClaimRefusal> {
        let total = self.heap.claimed().plus(claim);
        let topology = self.device.topology();
        let info = self.device.info();
        let executor = info.id();
        let fit = Fit::new(topology, status, &self.reserves, self.room);
        let host = fit
            .host_staging(info, total.staged)
            .saturating_add(self.host_table_bytes)
            .saturating_add(table);
        let demand = fit
            .demand(
                &new_storage(executor, total, |kind| self.held_free(kind)),
                &[(topology.host_ram().id(), host)],
                mode,
            )
            .map_err(ClaimRefusal::Blind)?;
        Ok(demand.refused.first().map(|refusal| {
            let role = readings
                .iter()
                .find(|reading| reading.domain == refusal.store)
                .expect("demand stores are observed")
                .role;
            ClaimRefusal::Deficit {
                role,
                store: refusal.store,
                constraint: refusal.constraint,
                required: refusal.required_bytes,
                available: refusal.required_bytes.saturating_sub(refusal.missing_bytes),
            }
        }))
    }

    /// The most bytes [`Self::claim_reserved`] of `kind` storage beside
    /// `beside` device bytes admits now, from one fresh status sample: by
    /// halving over [`Self::refusal`], the one test of a claim, which never
    /// admits more bytes where it refuses fewer. Refused while a domain is
    /// in the Reclaim band, or blind.
    pub fn reserved_room(&mut self, kind: BankTier, beside: u64) -> Result<u64, ClaimRefusal> {
        let (status, readings) = self.sample()?;
        let fits = |bytes: u64| -> Result<bool, ClaimRefusal> {
            let claim = besides(reserved(kind, bytes), beside)?;
            Ok(self.refusal(&status, &readings, claim, 0, FitMode::Growth)?.is_none())
        };
        if !fits(0)? {
            return Ok(0);
        }
        // The most known to fit, and the least known not to.
        let (mut fitting, mut over) = (0u64, 1u64);
        while fits(over)? {
            fitting = over;
            match over.checked_mul(2) {
                Some(next) => over = next,
                None => return Ok(u64::MAX),
            }
        }
        while over - fitting > 1 {
            let middle = fitting + (over - fitting) / 2;
            match fits(middle)? {
                true => fitting = middle,
                false => over = middle,
            }
        }
        Ok(fitting)
    }

    /// Hold a host-resident table's `bytes` in host RAM for the model's
    /// lifetime, from a fresh reading within host RAM's ceiling above its
    /// planning reserve, beside every outstanding claim and every storage
    /// placement into host RAM. Its mapped pages then count against every
    /// later decision there.
    pub fn hold_host_table(&mut self, bytes: u64) -> Result<(), ClaimRefusal> {
        self.check_storage(ClaimDemand::default(), bytes, FitMode::Live)?;
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
        self.claim_storage(ClaimDemand { device: required, staged, ..ClaimDemand::default() }, class)
    }

    /// [`Self::claim`] with `upload` bytes of upload storage beside the
    /// `required` device bytes.
    pub fn claim_storage(&mut self, demand: ClaimDemand, class: HoldingClass) -> Result<ClaimId, ClaimRefusal> {
        self.claim_demand(demand, 0, class)
    }

    /// Claim `bytes` a reservation of `kind` storage backs, the expert
    /// bank's growth, once they fit in the free headroom its growth reads
    /// ([`FitMode::Growth`]) with `beside` more device bytes, as
    /// [`Self::claim_demand`]: no block's free space serves them. Device
    /// storage's are fresh pages of the device's store; host storage's (a
    /// host tier's) host RAM.
    pub fn claim_reserved(
        &mut self,
        kind: BankTier,
        bytes: u64,
        beside: u64,
        class: HoldingClass,
    ) -> Result<ClaimId, ClaimRefusal> {
        self.claim_in(reserved(kind, bytes), beside, class, FitMode::Growth)
    }

    /// Claim `demand` once it fits with `beside` more device bytes, planned
    /// and not yet allocated, which the claim does not hold: an
    /// opportunistic growth's.
    pub(super) fn claim_demand(
        &mut self,
        demand: ClaimDemand,
        beside: u64,
        class: HoldingClass,
    ) -> Result<ClaimId, ClaimRefusal> {
        self.claim_in(demand, beside, class, FitMode::Live)
    }

    /// Claim `demand` once it fits with `beside` more device bytes in
    /// `mode`'s room.
    fn claim_in(
        &mut self,
        demand: ClaimDemand,
        beside: u64,
        class: HoldingClass,
        mode: FitMode,
    ) -> Result<ClaimId, ClaimRefusal> {
        self.check_storage(besides(demand, beside)?, 0, mode)?;
        self.heap
            .claim(demand, class)
            .map(|claim| claim.id)
            .map_err(ClaimRefusal::Accounting)
    }

    /// `claim`, of `kind` storage, now demands `bytes`: the rest of what it claimed is Seismic's
    /// charge already ([`MemoryHeap::reduce_claim`]).
    pub fn reduce_reserved(&mut self, claim: ClaimId, kind: BankTier, bytes: u64) -> Result<(), MemoryError> {
        self.heap.reduce_claim(claim, reserved(kind, bytes))
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
            .access_path(cuda.paths(seismic::Placing::Fixed(Contract::DeviceOnly))[0])
            .unwrap();
        let gpu = topology.store(path.store()).unwrap();
        let mut status = seismic::Status {
            sample: seismic::Sample {
                taken: std::time::SystemTime::now(),
                host_ram: Ok(seismic::HostStatus {
                    allocatable: seismic::Reported::new(
                        16 * GIB,
                        seismic::ReportedBy::LinuxMemAvailable,
                    ),
                    obtainable: seismic::Reported::new(
                        16 * GIB,
                        seismic::ReportedBy::LinuxMemAvailable,
                    ),
                    limits: Vec::new(),
                    limit_visibility: seismic::LimitVisibility::Complete,
                    displacement: None,
                    kernel_pressure: None,
                    held: None,
                    displaced_bytes: None,
                }),
                volumes: Vec::new(),
                executors: vec![cuda.id()],
                grants: Vec::new(),
            },
            runtime: seismic::RuntimeStatus::default(),
        };
        // The device's blocks leave 4 GiB of its memory as headroom.
        status.runtime.charges = vec![(gpu.id(), gpu.capacity().value - 4 * GIB)];
        let reserves = MemoryReserves::standard();
        let fit = Fit::new(&topology, &status, &reserves, HostRoom::Allocatable);
        let ceiling = fit.store(gpu.id()).unwrap().ceiling_bytes;
        let allocated = ClaimDemand {
            device: ceiling + GIB,
            ..ClaimDemand::default()
        };
        let demand = |claim: ClaimDemand, held: u64| {
            let storage = new_storage(cuda.id(), claim, |kind| match kind {
                Contract::DeviceOnly => held,
                _ => 0,
            });
            fit.demand(&storage, &[], FitMode::Live).unwrap()
        };
        let refused = demand(allocated, 0);
        assert_eq!(refused.refused.len(), 1);
        assert_eq!(refused.refused[0].store, gpu.id());
        assert!(demand(allocated, 2 * GIB).fits());
        // A reservation backs fresh pages: the blocks' free space serves
        // none of it.
        let reserved = ClaimDemand {
            reserved: ceiling + GIB,
            ..ClaimDemand::default()
        };
        assert_eq!(demand(reserved, 2 * GIB).refused[0].store, gpu.id());
        // A host tier's step (`claim_reserved` of host storage) is host
        // RAM: beyond all the device's store has room for, it fits in the
        // host's.
        let host_step = ClaimDemand {
            host: ceiling + GIB,
            ..ClaimDemand::default()
        };
        assert!(demand(host_step, 0).fits(), "a host step beyond the device's room");
    }

    /// A host tier's reserved growth is new host storage, which no block's
    /// free space serves, while the upload blocks' free space serves upload
    /// bytes allocated as the backend allocates.
    #[test]
    fn a_host_reservation_is_new_host_storage() {
        const GIB: u64 = 1 << 30;
        let executor = seismic::fixtures::recorded::rtx3090().executors()[0].id();
        let held_free = |kind| match kind {
            Contract::Upload => 3 * GIB,
            _ => 0,
        };
        let bytes = |storage: [StorageDemand; CLAIMED.len()]| storage.map(|demand| demand.bytes);
        let allocated = ClaimDemand { upload: 5 * GIB, ..ClaimDemand::default() };
        let host = ClaimDemand { host: 5 * GIB, ..ClaimDemand::default() };
        assert_eq!(bytes(new_storage(executor, allocated, held_free)), [0, 2 * GIB, 0]);
        assert_eq!(bytes(new_storage(executor, host, held_free)), [0, 0, 5 * GIB]);
    }

    /// The live ceiling of `heap`'s device storage beside its held host
    /// tables, in the host room its claims read, from a fresh status sample.
    fn live_ceiling(heap: &DeviceHeap) -> u64 {
        let topology = heap.device.topology();
        Fit::new(topology, &heap.status(), &heap.reserves, heap.room)
            .storage_beside(
                heap.device.info(),
                &CLAIMED,
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
            require_normal(&catalog, &device, &MemoryReserves::standard(), HostRoom::Obtainable(LoadStage::Loading)),
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
