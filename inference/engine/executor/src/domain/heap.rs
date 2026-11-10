//! The engine heap of one opened device: the single claim path from the first
//! startup allocation to unload.

use crate::memory::{
    ClaimDemand, ClaimId, HoldingClass, MemoryError, MemoryHeap, MemoryObservation,
};
use super::borrowing::BankRoom;
use crate::platform::{
    DomainReading, DomainRole, Fit, FitMode, StoreMotion, HostRoom, LoadStage, MemoryBand, MemoryConstraint,
    MemoryPolicyError, MemoryReserves, Serving, StorageDemand,
};
use crate::memory::CLAIMED;
use crate::planning::BankTier;
use seismic::{Contract, Device, Devices};
use std::fmt;
use std::rc::Rc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

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
pub(crate) fn reserved(kind: BankTier, bytes: u64) -> ClaimDemand {
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
            .ok_or(ClaimRefusal::Accounting(MemoryError::Overflow))?,
        ..demand
    })
}

/// Why [`DeviceHeap::claim_most`] stopped short of a claim: its demand's
/// error, or a refusal.
enum Halted<E> {
    Demand(E),
    Refused(ClaimRefusal),
}

/// The most count that `fits`, between `fitting`, known to fit, and
/// `over`, known not to (`fitting < over`), by halving: `fits` never admits
/// a larger count where it refuses a smaller one.
fn most_fitting<E>(
    mut fitting: u64,
    mut over: u64,
    mut fits: impl FnMut(u64) -> Result<bool, E>,
) -> Result<u64, E> {
    while over - fitting > 1 {
        let middle = fitting + (over - fitting) / 2;
        match fits(middle)? {
            true => fitting = middle,
            false => over = middle,
        }
    }
    Ok(fitting)
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

    /// The time a reading is taken at now: the clock every interval of the
    /// memory path is measured on (the pace's takes, the bank's room), the
    /// one the owner reads memory by. The live machine's is the monotonic
    /// clock the owner's time counts; a test's is its owner's.
    fn now(&self) -> Instant {
        Instant::now()
    }

    /// The periodic reading taken since the last one this returned, without
    /// waiting for one ([`DeviceHeap::refresh_published`]); none where none
    /// was published since. By default it reads now: a test's readings are
    /// taken where the test asks for them. The live machine's are read on a
    /// thread of their own ([`MachineReading`]).
    fn published(&self, device: &Device) -> Option<Published> {
        Some(Published { status: self.status(device), footprint: self.footprint(), at: self.now() })
    }
}

/// One periodic reading of the machine ([`StatusReading::published`]): its
/// status, the process's footprint and when both were read.
pub struct Published {
    pub status: seismic::Status,
    pub footprint: Option<u64>,
    pub at: Instant,
}

impl StatusReading for Devices {
    fn status(&self, device: &Device) -> seismic::Status {
        self.device_status(device)
    }

    fn footprint(&self) -> Option<u64> {
        seismic::ProcessMemory::footprint_now()
    }
}

/// The live machine's readings: a claim's read where it is made, as a claim
/// is judged against the machine now, and the periodic ones on a thread of
/// their own every [`crate::platform::READING_INTERVAL`], each published for
/// the owner to take without waiting ([`StatusReading::published`]). A
/// reading's system calls and driver queries never sit between a step and
/// the next, however long either takes.
pub struct MachineReading {
    devices: Rc<Devices>,
    reader: std::cell::OnceCell<Reader>,
}

impl MachineReading {
    pub fn new(devices: Rc<Devices>) -> Self {
        Self { devices, reader: std::cell::OnceCell::new() }
    }
}

impl StatusReading for MachineReading {
    fn status(&self, device: &Device) -> seismic::Status {
        StatusReading::status(&*self.devices, device)
    }

    fn footprint(&self) -> Option<u64> {
        seismic::ProcessMemory::footprint_now()
    }

    /// The reader starts at the first ask, which therefore finds none yet.
    fn published(&self, device: &Device) -> Option<Published> {
        let reader = self.reader.get_or_init(|| {
            let devices = Devices::new(self.devices.inventory().clone());
            Reader::start(devices, device.clone())
        });
        reader.shared.lock().latest.take()
    }
}

/// The thread [`MachineReading`] reads on, stopped and joined as it drops.
struct Reader {
    shared: Arc<ReaderShared>,
    thread: Option<std::thread::JoinHandle<()>>,
}

struct ReaderShared {
    state: Mutex<ReaderState>,
    stop: Condvar,
}

struct ReaderState {
    /// The newest reading not yet taken: an older one untaken is replaced.
    latest: Option<Published>,
    stopped: bool,
}

impl ReaderShared {
    fn lock(&self) -> std::sync::MutexGuard<'_, ReaderState> {
        self.state.lock().expect("the machine reader's state lock is never poisoned")
    }
}

impl Reader {
    fn start(devices: Devices, device: Device) -> Self {
        let shared = Arc::new(ReaderShared { state: Mutex::new(ReaderState { latest: None, stopped: false }), stop: Condvar::new() });
        let reading = shared.clone();
        let thread = std::thread::Builder::new()
            .name("magnitude-memory".into())
            .spawn(move || loop {
                let status = devices.device_status(&device);
                let footprint = seismic::ProcessMemory::footprint_now();
                let mut state = reading.lock();
                state.latest = Some(Published { status, footprint, at: Instant::now() });
                let (state, _) = reading
                    .stop
                    .wait_timeout_while(state, crate::platform::READING_INTERVAL, |state| !state.stopped)
                    .expect("the machine reader's state lock is never poisoned");
                if state.stopped {
                    return;
                }
            })
            .expect("invariant violated: the memory reading thread could not start");
        Self { shared, thread: Some(thread) }
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        self.shared.lock().stopped = true;
        self.shared.stop.notify_all();
        if let Some(thread) = self.thread.take() {
            thread.join().expect("the memory reading thread does not panic");
        }
    }
}

/// The heap of one opened device together with the host's threshold policy
/// and the reading that observes the domains the device uses. It exists
/// before the first startup allocation, so loading and serving claim through
/// one authority: every allocation's peak is claimed from a fresh reading,
/// and Seismic's enforced limit is refreshed from the same reading.
pub struct DeviceHeap {
    reading: Rc<dyn StatusReading>,
    /// The reserves the heap opened with: the machine's figures and what
    /// was known before this load measured anything.
    known: MemoryReserves,
    /// `known` with what the load measures now ([`Self::measure`]).
    reserves: MemoryReserves,
    device: Rc<Device>,
    /// The right to bound the stores the device reaches, the heap's alone
    /// while it lives.
    bounds: seismic::Bounds,
    capacity_bytes: u64,
    heap: MemoryHeap,
    /// Host-resident gathered tables' bytes (§3.7 of the model-family plan):
    /// mapped pages the model reads every step, held in the host-RAM domain
    /// for the model's lifetime and never counted as free page cache.
    host_table_bytes: u64,
    /// The host RAM room its band, claims and Seismic's limits read, and
    /// its growth's: [`HostRoom::Load`] from the load's opening on, its
    /// growth's displacement counted from when it serves ([`Self::serve`]),
    /// with the hold on the expert bank's borrowed room merged in as it
    /// reads ([`Self::room`]).
    room: HostRoom,
    /// The one store's expert bank's slots, committed and borrowed, from
    /// when it serves; none without one, or where it lies across a link
    /// (a host tier's), whose slots borrow nothing.
    bank: Option<BankRoom>,
    /// The last reading observed, fresh or published: its status, its
    /// domains' readings and the process's own charge it read
    /// ([`Self::charged`]), on which a growth's room is sized while that
    /// charge stands ([`Self::reserved_rooms`]); none where it failed.
    observed: Option<(seismic::Status, Vec<DomainReading>, Vec<(seismic::StoreId, u64)>)>,
}

/// A search on the heap's device reads its room through the heap: the
/// heap holds the one right to bound the device's ledgers while it lives, so
/// the search's readings bound them as the heap's own do, never from a
/// second holder beside it.
impl crate::TuningRoom for DeviceHeap {
    /// A reading that fails reads no room: missing observations never
    /// authorize allocation.
    fn now(&self, device: &Device) -> crate::RoomNow {
        assert_eq!(device.info().selector(), self.device.info().selector(), "a search reads the room of its heap's device");
        let status = self.reading.status(&self.device);
        crate::platform::room_now(&self.device, &self.bounds, &status, &self.reserves, self.room())
            .unwrap_or(crate::RoomNow { device_bytes: 0, host_bytes: 0, host_shared: false })
    }
}

impl DeviceHeap {
    /// Hold every fit from the next reading on at what the load measures
    /// now: host RAM's motion and a device's own memory's, and its release
    /// latency once one finished (`platform::MemoryReserves::measured`),
    /// over what was known when it opened. A motion is the pace's sustained
    /// take, which falls again once a take stops, so each measure starts
    /// from what was known rather than from every earlier measure's peak.
    pub(crate) fn measure(
        &mut self,
        host: Option<StoreMotion>,
        device: Option<StoreMotion>,
        latency: Option<std::time::Duration>,
        release: Option<crate::platform::ReleaseRate>,
        releasable: crate::platform::Releasable,
    ) {
        let measured = self.known.measured(host, device).releasing(release, releasable);
        self.reserves = latency.map_or(measured, |latency| measured.with_latency(latency));
    }

    /// The rate the load releases at, as its reserves hold it
    /// (`platform::MemoryReserves::release_rate`).
    pub(crate) fn release_rate(&self) -> crate::platform::ReleaseRate {
        self.reserves.release_rate()
    }

    /// The heap of `device`, observed once, for a load
    /// ([`HostRoom::Load`], accepting its own displacement). Its
    /// capacity, the stable fit of the device's storage
    /// ([`MemoryObservation::capacity_bytes`]), is fixed here.
    pub fn open(
        reading: Rc<dyn StatusReading>,
        reserves: MemoryReserves,
        device: Rc<Device>,
    ) -> Result<Self, MemoryPolicyError> {
        let status = reading.status(&device);
        let room = HostRoom::Load(LoadStage::Loading);
        let capacity_bytes = Fit::new(device.topology(), &status, &reserves, room)
            .storage_beside(device.info(), &CLAIMED, &[])?
            .stable_bytes;
        let bounds = device.bounds();
        let mut heap = Self {
            reading,
            known: reserves,
            reserves,
            device,
            bounds,
            capacity_bytes,
            heap: MemoryHeap::new(),
            host_table_bytes: 0,
            room,
            bank: None,
            observed: None,
        };
        // Nothing the load plans is held yet, nor admitted.
        heap.observe(&status, 0)?;
        Ok(heap)
    }

    /// The model has loaded: the displacement its load caused is accepted,
    /// and from now on its growth reads what can be taken without displacing
    /// any more of another program's data than is displaced now
    /// ([`HostRoom::Load`]). Its band and claims read the same room as they
    /// did while it loaded. A one store's expert bank of `bank`'s slots of
    /// its bytes each serves from here, every slot committed
    /// ([`BankRoom`]).
    pub fn serve(&mut self, bank: Option<(u64, u64)>, planned: u64) -> Result<(), MemoryPolicyError> {
        let status = self.status();
        let host = status.sample.host_ram.as_ref().map_err(|error| MemoryPolicyError::Observation(error.clone()))?;
        let now = self.reading.now();
        self.room = HostRoom::serving(host, now);
        let counted = matches!(self.room, HostRoom::Load(LoadStage::Serving(_)));
        self.bank = bank.map(|(slots, slot_bytes)| BankRoom::new(counted, slot_bytes, slots, now));
        self.observe(&status, planned).map(drop)
    }

    /// The host RAM room its readings read: its field's, the hold on the
    /// expert bank's borrowed room merged in (`platform::Serving`).
    pub fn room(&self) -> HostRoom {
        match (self.room, &self.bank) {
            (HostRoom::Load(LoadStage::Serving(serving)), Some(bank)) => HostRoom::Load(LoadStage::Serving(Serving {
                baseline: bank.baseline(),
                borrows: true,
                lent: bank.lent(),
                ..serving
            })),
            (room, _) => room,
        }
    }

    /// The one store's expert bank's slots, committed and borrowed; none
    /// before it serves, or without one.
    pub fn bank_room(&self) -> Option<&BankRoom> {
        self.bank.as_ref()
    }

    /// A growth of the one store's bank's slots began, `borrowed` of its
    /// bytes on borrowed room ([`BankRoom::grow`]).
    pub fn grow_bank(&mut self, borrowed: u64) {
        self.bank.as_mut().expect("a growth of the one store's slots is of its bank").grow(borrowed);
    }

    /// The one store's bank's growth of its slots ended, the bank holding
    /// `slots` ([`BankRoom::grown`]). A growth into borrowed room displaced
    /// other programs' data by design: what it displaced is accepted, as a
    /// load's own import's is, so that once the room is given back the hold
    /// line does not count what stays displaced until its programs read it
    /// again (`platform::Serving::displaced_bytes`).
    pub fn bank_grown(&mut self, slots: u64) -> Result<(), MemoryPolicyError> {
        let bank = self.bank.as_mut().expect("a growth of the one store's slots is of its bank");
        if bank.grown(slots, self.reading.now()) {
            if let HostRoom::Load(LoadStage::Serving(serving)) = &mut self.room {
                let status = self.reading.status(&self.device);
                let host = status.sample.host_ram.as_ref().map_err(|error| MemoryPolicyError::Observation(error.clone()))?;
                serving.displaced_bytes = host
                    .displaced_bytes
                    .expect("invariant violated: a host that reported displacement as the load began to serve reports it still");
            }
        }
        Ok(())
    }

    /// The one store's bank's shrink left it `slots` slots
    /// ([`BankRoom::shrunk`]).
    pub fn bank_shrunk(&mut self, slots: u64) {
        if let Some(bank) = self.bank.as_mut() {
            bank.shrunk(slots);
        }
    }

    pub fn device(&self) -> &Rc<Device> {
        &self.device
    }

    /// The heap's right to bound the device's ledgers.
    #[cfg(test)]
    pub(crate) fn bounds(&self) -> &seismic::Bounds {
        &self.bounds
    }

    /// A fresh status of the device's executor.
    fn status(&self) -> seismic::Status {
        self.reading.status(&self.device)
    }

    pub fn heap(&self) -> &MemoryHeap {
        &self.heap
    }

    /// The process's host RAM footprint now ([`StatusReading::footprint`]).
    pub fn footprint(&self) -> Option<u64> {
        self.reading.footprint()
    }

    /// The time a reading is taken at now ([`StatusReading::now`]).
    pub fn now(&self) -> std::time::Instant {
        self.reading.now()
    }

    /// Observe every domain the device uses from a sample read now, set
    /// Seismic's allocation ceiling from the readings, and refresh the
    /// heap's view from them: a release the process made shows in the next
    /// one, so no decision stands on a reading older than the release before
    /// it. A failed observation revokes unspent grants and leaves the heap
    /// Blind.
    pub fn refresh(&mut self, planned: u64) -> Result<Vec<DomainReading>, MemoryPolicyError> {
        let status = self.reading.status(&self.device);
        self.observe(&status, planned)
    }

    /// [`Self::refresh`] from the periodic reading published since the last
    /// ([`StatusReading::published`]), with the footprint and time it was
    /// read; none, refreshing nothing, where none was published since.
    pub fn refresh_published(&mut self, planned: u64) -> Option<(Result<Vec<DomainReading>, MemoryPolicyError>, Option<u64>, Instant)> {
        let Published { status, footprint, at } = self.reading.published(&self.device)?;
        Some((self.observe(&status, planned), footprint, at))
    }

    /// [`Self::refresh`] from the status sample `status`, Seismic's bound
    /// keeping `planned` bytes admissible: the load's planned holdings in
    /// its device storage's store not yet allocated
    /// (`ExecutorDomain::planned_unheld`, the one owner of the figure every
    /// reading is passed), admitted with the load whatever the line does later
    /// (`platform::refresh_device_ceiling`).
    fn observe(
        &mut self,
        status: &seismic::Status,
        planned: u64,
    ) -> Result<Vec<DomainReading>, MemoryPolicyError> {
        let readings =
            crate::platform::refresh_device_ceiling(&self.device, &self.bounds, status, &self.reserves, self.room(), planned);
        self.account();
        let charged = self.charged();
        self.observed = readings.as_ref().ok().map(|readings| (status.clone(), readings.clone(), charged));
        // Each reading's planning line by the terms of its margin, where
        // the memory trace is read.
        let traced = tracing::enabled!(target: super::MEMORY_TRACE, tracing::Level::DEBUG);
        for reading in readings.iter().flatten().filter(|_| traced) {
            let store = self.device.topology().store(reading.domain).expect("a reading is of a store of the device's topology");
            if !matches!(store.kind(), seismic::StoreKind::Volume(_)) {
                let terms = self.reserves.margin_terms(store);
                tracing::debug!(
                    target: super::MEMORY_TRACE,
                    store = ?reading.identity,
                    planning = reading.thresholds.planning_bytes,
                    fluctuation = terms.fluctuation,
                    take = terms.take,
                    latency = terms.latency,
                    releasing = terms.releasing,
                    "memory line"
                );
            }
        }
        if let (Ok(readings), Some(bank)) = (&readings, self.bank.as_mut()) {
            let now = self.reading.now();
            for reading in readings.iter().filter(|reading| reading.constraint == MemoryConstraint::HostRam) {
                bank.observe(reading, now);
            }
        }
        readings
    }

    /// What the process's own allocations charge each of Seismic's stores
    /// now.
    fn charged(&self) -> Vec<(seismic::StoreId, u64)> {
        self.device.ledgers().own.iter().map(|(store, ledger)| (*store, ledger.charged)).collect()
    }

    /// Refresh the heap's view of Seismic's charge alone, reading no store:
    /// what the process's own claims and releases changed since. The stores
    /// and Seismic's ceiling stand as the last [`Self::refresh`] read them.
    pub fn account(&mut self) {
        let charged_bytes = self.charged().into_iter().map(|(_, charged)| charged).sum();
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
    pub fn check(&mut self, required: u64, staged: u64, planned: u64) -> Result<(), ClaimRefusal> {
        self.check_storage(
            ClaimDemand {
                device: required,
                staged,
                ..ClaimDemand::default()
            },
            0,
            FitMode::Live,
            planned,
        )
    }

    /// Whether `claim` and `table` new bytes of host tables fit beside every
    /// outstanding claim and the held host tables in `mode`'s room, from one
    /// fresh status sample. Fit decides where the claims' staged bytes are
    /// charged ([`Fit::host_staging`]).
    fn check_storage(&mut self, claim: ClaimDemand, table: u64, mode: FitMode, planned: u64) -> Result<(), ClaimRefusal> {
        let (status, readings) = self.sample(planned)?;
        self.refusal(&status, &readings, claim, table, mode)?.map_or(Ok(()), Err)
    }

    /// One fresh status sample, Seismic's limits set from it (the limit a
    /// resize's recommit runs into, `Fit::placement` of each storage kind
    /// at this sample: no tighter than what [`Self::refusal`] admits of the
    /// kind, which also counts every outstanding claim and the host's
    /// direct demands), and its readings; refused while a domain is in the
    /// Reclaim band.
    fn sample(&mut self, planned: u64) -> Result<(seismic::Status, Vec<DomainReading>), ClaimRefusal> {
        let status = self.status();
        let readings = self.observe(&status, planned).map_err(ClaimRefusal::Blind)?;
        match reclaiming(&readings) {
            Some(refusal) => Err(refusal),
            None => Ok((status, readings)),
        }
    }

    /// Why `claim` and `table` new bytes of host tables do not fit beside
    /// every outstanding claim and the held host tables at `status`, whose
    /// domains `readings` observed; none where they fit. The one test of a
    /// claim: [`Self::check_storage`] applies it, and [`Self::reserved_rooms`]
    /// finds the most it admits.
    fn refusal(
        &self,
        status: &seismic::Status,
        readings: &[DomainReading],
        claim: ClaimDemand,
        table: u64,
        mode: FitMode,
    ) -> Result<Option<ClaimRefusal>, ClaimRefusal> {
        let fit = Fit::new(self.device.topology(), status, &self.reserves, self.room());
        self.refusal_in(&fit, readings, claim, table, mode)
    }

    /// [`Self::refusal`] at `fit`, the fit of the status `readings` observed.
    fn refusal_in(
        &self,
        fit: &Fit<'_>,
        readings: &[DomainReading],
        claim: ClaimDemand,
        table: u64,
        mode: FitMode,
    ) -> Result<Option<ClaimRefusal>, ClaimRefusal> {
        let total = self.heap.claimed().plus(claim);
        let topology = self.device.topology();
        let info = self.device.info();
        let executor = info.id();
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

    /// The most bytes a reservation of `kind` storage backs ([`reserved`]) beside
    /// `beside` device bytes admits in each of `modes`' rooms on the last
    /// reading ([`Self::observe`]) while the process's own charge stands as
    /// it read it, a fresh one otherwise (a status reads the process's
    /// allocations with every other program's, so one read before an
    /// allocation is no status beside it): by halving
    /// over [`Self::refusal`], the one test of a claim, which never admits
    /// more bytes where it refuses fewer, at the reading's one fit. The
    /// claim of what it sized reads the machine afresh, so a size is never
    /// held on a reading the machine has since left. Refused while a domain
    /// is in the Reclaim band, or blind.
    pub fn reserved_rooms(&mut self, kind: BankTier, beside: u64, modes: &[FitMode], planned: u64) -> Result<Vec<u64>, ClaimRefusal> {
        let charged = self.charged();
        if self.observed.as_ref().is_none_or(|(_, _, read)| *read != charged) {
            self.sample(planned)?;
        }
        let (status, readings, _) = self.observed.as_ref().expect("a sample that succeeds is observed");
        if let Some(refusal) = reclaiming(readings) {
            return Err(refusal);
        }
        let fit = Fit::new(self.device.topology(), status, &self.reserves, self.room());
        modes
            .iter()
            .map(|&mode| {
                let fits = |bytes: u64| -> Result<bool, ClaimRefusal> {
                    let claim = besides(reserved(kind, bytes), beside)?;
                    Ok(self.refusal_in(&fit, readings, claim, 0, mode)?.is_none())
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
                most_fitting(fitting, over, fits)
            })
            .collect()
    }

    /// Claim the most of `least..=most` whose demand (`demand`) fits in
    /// `mode`'s room with `beside` more device bytes, planned and not yet
    /// allocated, which the claim does not hold: found by halving over
    /// [`Self::refusal`], the one test of a claim, on one fresh reading, and
    /// claimed on that same reading, so what it sized is what it holds: the
    /// one way a load's import and its bank's growth size what they claim.
    /// `demand` never demands less of a larger count; an error of it ends
    /// the claim as it is.
    /// Refused as `least` is where not even it fits.
    #[allow(clippy::too_many_arguments)]
    pub fn claim_most<E>(
        &mut self,
        least: u64,
        most: u64,
        demand: impl Fn(u64) -> Result<ClaimDemand, E>,
        beside: u64,
        class: HoldingClass,
        mode: FitMode,
        planned: u64,
    ) -> Result<Result<(u64, ClaimId), ClaimRefusal>, E> {
        assert!(least <= most, "invariant violated: a claim's least count exceeds its most");
        let decided = (|| -> Result<u64, Halted<E>> {
            let (status, readings) = self.sample(planned).map_err(Halted::Refused)?;
            let fit = Fit::new(self.device.topology(), &status, &self.reserves, self.room());
            // Why `count`'s demand does not fit; none where it does.
            let refusal = |count: u64| -> Result<Option<ClaimRefusal>, Halted<E>> {
                let claim = besides(demand(count).map_err(Halted::Demand)?, beside).map_err(Halted::Refused)?;
                self.refusal_in(&fit, &readings, claim, 0, mode).map_err(Halted::Refused)
            };
            let Some(refused) = refusal(most)? else {
                return Ok(most);
            };
            if least == most {
                return Err(Halted::Refused(refused));
            }
            if let Some(refused) = refusal(least)? {
                return Err(Halted::Refused(refused));
            }
            most_fitting(least, most, |count| Ok(refusal(count)?.is_none()))
        })();
        let count = match decided {
            Ok(count) => count,
            Err(Halted::Demand(error)) => return Err(error),
            Err(Halted::Refused(refusal)) => return Ok(Err(refusal)),
        };
        let claim = demand(count)?;
        Ok(self.heap.claim(claim, class).map(|claim| (count, claim.id)).map_err(ClaimRefusal::Accounting))
    }

    /// Hold a host-resident table's `bytes` in host RAM for the model's
    /// lifetime, from a fresh reading within host RAM's ceiling above its
    /// planning reserve, beside every outstanding claim and every storage
    /// placement into host RAM. Its mapped pages then count against every
    /// later decision there.
    pub fn hold_host_table(&mut self, bytes: u64, planned: u64) -> Result<(), ClaimRefusal> {
        self.check_storage(ClaimDemand::default(), bytes, FitMode::Live, planned)?;
        self.host_table_bytes = self
            .host_table_bytes
            .checked_add(bytes)
            .ok_or(ClaimRefusal::Accounting(MemoryError::Overflow))?;
        self.refresh(planned).map_err(ClaimRefusal::Blind)?;
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
        planned: u64,
    ) -> Result<ClaimId, ClaimRefusal> {
        self.claim_storage(ClaimDemand { device: required, staged, ..ClaimDemand::default() }, class, planned)
    }

    /// [`Self::claim`] with `upload` bytes of upload storage beside the
    /// `required` device bytes.
    pub fn claim_storage(&mut self, demand: ClaimDemand, class: HoldingClass, planned: u64) -> Result<ClaimId, ClaimRefusal> {
        self.claim_in(demand, 0, class, FitMode::Live, planned)
    }

    /// Claim `demand` once it fits with `beside` more device bytes, planned
    /// and not yet allocated, which the claim does not hold: an
    /// opportunistic growth's, which a refusal leaves as it is.
    pub(super) fn claim_demand(
        &mut self,
        demand: ClaimDemand,
        beside: u64,
        class: HoldingClass,
        planned: u64,
    ) -> Result<ClaimId, ClaimRefusal> {
        self.claim_in(demand, beside, class, FitMode::Live, planned)
    }

    /// Claim `demand` once it fits with `beside` more device bytes in
    /// `mode`'s room, the reading's bound keeping `planned` bytes
    /// admissible ([`Self::observe`]). A refusal of a claim the load must
    /// have is owed room the owner's release settles, the lowest rate per
    /// byte first.
    fn claim_in(
        &mut self,
        demand: ClaimDemand,
        beside: u64,
        class: HoldingClass,
        mode: FitMode,
        planned: u64,
    ) -> Result<ClaimId, ClaimRefusal> {
        let (status, readings) = self.sample(planned)?;
        if let Some(refusal) = self.refusal(&status, &readings, besides(demand, beside)?, 0, mode)? {
            return Err(refusal);
        }
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
                    free_target: Some(0),
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
        let reserves = MemoryReserves::of_machine(None);
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
        // A host tier's step (`reserved` of host storage) is host
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
        Fit::new(topology, &heap.status(), &heap.reserves, heap.room())
            .storage_beside(
                heap.device.info(),
                &CLAIMED,
                &[(topology.host_ram().id(), heap.host_table_bytes)],
            )
            .unwrap()
            .ceiling_bytes
    }

    /// A count's demand, as a bank's slots move a load's import: `unit`
    /// device bytes a count beside a fixed part.
    fn counted(unit: u64) -> impl Fn(u64) -> Result<ClaimDemand, std::convert::Infallible> {
        move |count| Ok(ClaimDemand { device: unit + count * unit, ..ClaimDemand::default() })
    }

    /// The most a claim decides is what the one test of a claim admits on
    /// the same reading: the count it claims fits, one more does not.
    #[test]
    fn the_most_claimed_is_what_the_claim_test_admits() {
        let (mut heap, _press) = crate::domain::fixtures::pressed_heap(BackendName::Cpu);
        let unit = live_ceiling(&heap) / 64;
        let demand = counted(unit);
        let (count, claim) = heap.claim_most(0, 1000, &demand, 0, HoldingClass::Model, FitMode::Live, 0).unwrap().unwrap();
        assert!((60..64).contains(&count), "{count} of a ceiling of 64 units, one fixed");
        heap.release(claim);
        assert_eq!(heap.check(demand(count).unwrap().device, 0, 0), Ok(()));
        assert!(matches!(heap.check(demand(count + 1).unwrap().device, 0, 0), Err(ClaimRefusal::Deficit { .. })));
        // What is kept beside the claim takes its room first.
        let (beside, claim) = heap.claim_most(0, 1000, &demand, 8 * unit, HoldingClass::Model, FitMode::Live, 0).unwrap().unwrap();
        assert_eq!(beside, count - 8);
        heap.release(claim);
    }

    /// Memory another program takes between a plan and its claim shrinks
    /// the count the claim commits, where a claim of the planned count is
    /// refused; only a least count it cannot hold is refused.
    #[test]
    fn a_press_after_the_plan_shrinks_the_claim_instead_of_refusing_it() {
        let (mut heap, press) = crate::domain::fixtures::pressed_heap(BackendName::Cpu);
        let unit = live_ceiling(&heap) / 64;
        let demand = counted(unit);
        // Planned on the room as first read: 48 units fit.
        let planned = 48;
        assert_eq!(heap.check(demand(planned).unwrap().device, 0, 0), Ok(()));
        press.take(32 * unit);
        assert!(matches!(
            heap.claim(demand(planned).unwrap().device, 0, HoldingClass::Model, 0),
            Err(ClaimRefusal::Deficit { .. })
        ));
        let (count, claim) = heap.claim_most(4, planned, &demand, 0, HoldingClass::Model, FitMode::Live, 0).unwrap().unwrap();
        assert!((4..planned).contains(&count) && count >= 28, "{count} of {planned} planned after 32 units were taken");
        heap.release(claim);
        press.take(63 * unit);
        assert!(matches!(
            heap.claim_most(4, planned, &demand, 0, HoldingClass::Model, FitMode::Live, 0).unwrap(),
            Err(ClaimRefusal::Deficit { .. })
        ));
    }

    /// Two heaps of the process's one shared open of an executor bound its
    /// ledgers from their own readings, each moving the other's room: the
    /// second to take the right to bound them panics at the first it
    /// takes, its path account, naming both. A test opens a device of its
    /// own (`test_support::open`) and takes the right alone.
    #[test]
    #[should_panic(expected = "a path ledger is bounded by two live holders")]
    fn a_second_heap_on_a_shared_device_panics() {
        let catalog = Rc::new(Devices::discover(&[]).unwrap());
        let device = Rc::new(catalog.open_backend(BackendName::Cpu).unwrap());
        let _first = DeviceHeap::open(catalog.clone(), MemoryReserves::of_machine(None), device.clone()).unwrap();
        let _second = DeviceHeap::open(catalog, MemoryReserves::of_machine(None), device);
    }

    /// A search started while the load's heap lives reads the device's
    /// room through the heap, which holds the one right to bound its
    /// ledgers: reading it at every decision takes no second holder, so it
    /// never panics. (A search that bounded the ledgers from a holder of its
    /// own, `Device::bounds`, beside the heap's panicked at its first
    /// reading, naming both.)
    #[test]
    fn a_search_reads_its_room_through_the_live_heap() {
        let heap = cpu_heap();
        let device = heap.device.clone();
        let room: &dyn crate::TuningRoom = &heap;
        room.now(&device);
        room.now(&device);
    }

    fn cpu_heap() -> DeviceHeap {
        let catalog = Rc::new(Devices::discover(&[]).unwrap());
        let device = Rc::new(crate::test_support::open_backend(&catalog, BackendName::Cpu).unwrap());
        DeviceHeap::open(catalog, MemoryReserves::of_machine(None), device).unwrap()
    }

    #[test]
    fn a_host_in_the_normal_band_lets_a_running_operation_continue() {
        let catalog = Devices::discover(&[]).unwrap();
        let device = crate::test_support::open_backend(&catalog, BackendName::Cpu).unwrap();
        assert_eq!(
            require_normal(&catalog, &device, &MemoryReserves::of_machine(None), HostRoom::Load(LoadStage::Loading)),
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
        let claim = heap.claim(ceiling / 2, 0, HoldingClass::Live, 0).unwrap();
        assert_eq!(heap.heap().claims().count(), 1);
        // A fresh reading of the same host now leaves only about the
        // unclaimed half of the ceiling.
        assert!(matches!(
            heap.check(ceiling / 4 * 3, 0, 0),
            Err(ClaimRefusal::Deficit {
                role: DomainRole::Allocation,
                constraint: MemoryConstraint::HostRam,
                ..
            })
        ));
        heap.release(claim);
        assert_eq!(heap.heap().claims().count(), 0);
        assert_eq!(heap.check(0, 0, 0), Ok(()));
        assert!(matches!(
            heap.claim(u64::MAX / 2, 0, HoldingClass::Live, 0),
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
            heap.hold_host_table(u64::MAX / 2, 0),
            Err(ClaimRefusal::Deficit {
                role: DomainRole::Allocation,
                ..
            })
        ));
        assert_eq!(heap.host_table_bytes(), 0);
        heap.hold_host_table(ceiling / 2, 0).unwrap();
        assert_eq!(heap.host_table_bytes(), ceiling / 2);
        assert!(live_ceiling(&heap) <= ceiling - ceiling / 2 + ceiling / 8);
        assert!(matches!(
            heap.check(ceiling / 4 * 3, 0, 0),
            Err(ClaimRefusal::Deficit {
                role: DomainRole::Allocation,
                ..
            })
        ));
    }
}
