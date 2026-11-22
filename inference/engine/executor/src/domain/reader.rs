//! The memory reading of one opened device: the machine's status sampled
//! at the owner's events (a flight boundary, an own change's completion, an
//! OS pressure event, and while something stands that only a later reading
//! answers, the worker's timed reading, `magnitude_scheduler::worker::Reread`),
//! the domains the device uses observed in it, and the device's store
//! ledgers read from it. The OS's pressure events
//! ([`DeviceReader::watch_pressure`]) wake the owner where the OS gives them.

use crate::memory::ledger::{Changes, DeviceLedgers, DeviceReading, Sample, UnreadLedgers};
use crate::platform::{DomainReading, MemoryPolicyError, ReleaseRate};
use seismic::{Device, Devices};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// What the reader reads of the machine: one status sample a reading, every
/// domain the device uses observed in it, of the device's executor's grants
/// alone: no other executor's grant is read. Seismic's catalog reads the
/// live machine; a test gives the readings it decides on.
pub trait StatusReading {
    /// A sample with host RAM read now ([`DeviceReader::refresh`]).
    fn status(&self, device: &Device) -> seismic::Status;

    /// The time a reading is taken at now: what the ledgers order their
    /// readings by (`memory::ledger::Ledger::read`), the clock the owner
    /// reads memory by. The live machine's is the monotonic clock the
    /// owner's time counts; a test's is its owner's.
    fn now(&self) -> Instant {
        Instant::now()
    }

    /// Settle `device`'s driver memory and return the grant read in that
    /// query ([`seismic::Device::reconcile_driver_memory`]).
    fn reconcile(&self, device: &Device) -> Option<seismic::memory::Grant> {
        device.reconcile_driver_memory()
    }

    /// Whether a scripted pressure event arrived since this was last asked.
    fn take_pressure(&self) -> bool {
        false
    }

    /// One sample of `device`'s status, host RAM's among it, at one
    /// instant taken now on the owner's thread, `device`'s driver memory
    /// settled within it, which never waits ([`Sample::try_now`]): none
    /// where a change off its thread is beside it.
    fn sample(&self, device: &Device, changes: &Changes) -> Option<Sample> {
        Sample::try_with(changes, || self.reconcile(device), || self.now(), || self.status(device))
    }
}

impl StatusReading for Devices {
    fn status(&self, device: &Device) -> seismic::Status {
        self.device_status(device)
    }
}

/// The live machine's readings: a sample read where the owner asks for one.
pub struct MachineReading {
    devices: Rc<Devices>,
}

impl MachineReading {
    pub fn new(devices: Rc<Devices>) -> Self {
        Self { devices }
    }
}

impl StatusReading for MachineReading {
    fn status(&self, device: &Device) -> seismic::Status {
        StatusReading::status(&*self.devices, device)
    }
}

/// Whether a reader watches the OS's memory pressure events
/// ([`DeviceReader::watch_pressure`]).
enum Watch {
    /// The owner has not asked.
    Unasked,
    /// Its host RAM store's events wake it while the watch lives.
    Events { _watch: crate::memory::ledger::PressureWatch },
    /// The OS gives no unprivileged events (a Linux kernel or container
    /// without writable PSI triggers): the owner reads at its other events,
    /// and while a shortage, a wait or a deficit stands, at its timed
    /// readings (`magnitude_scheduler::worker::Reread`).
    Unwatchable,
}

/// The memory reading of one opened device together with the machine's
/// release rate: it exists from the device's opening to the load's end.
pub struct DeviceReader {
    reading: Rc<dyn StatusReading>,
    /// The machine's release rate the reader opened with.
    release: ReleaseRate,
    device: Rc<Device>,
    /// An OS pressure event came since the last reading asked
    /// ([`Self::pressured`]).
    pressured: Arc<AtomicBool>,
    /// Whether the reader watches the OS's pressure events
    /// ([`Self::watch_pressure`]).
    watch: Watch,
}

/// The room a load's search is held to: what the device's store ledgers
/// let a load's claims take at the reading the load holds
/// ([`DeviceReader::settled_read`]), every holding since counted
/// (`memory::ledger::DeviceReading::room_after`, `Estimate::room`): the one
/// authority on room the load's claims read too, and the room its plan
/// priced. The search's own heap is held as its grant in host RAM's store
/// (`GrantRole::Tuning`), grown to what this process's footprint grew by
/// since the search began at each decision, so the room it reads has
/// already taken it, and a later reading credits it as the process's own,
/// never as others'.
pub struct LoadRoom<'a> {
    reading: &'a DeviceReading,
    ledgers: &'a DeviceLedgers,
    /// The machine's release rate (`platform::ReleaseRate::of_machine`).
    release: crate::platform::ReleaseRate,
    grant: crate::memory::ledger::Grant,
}

impl<'a> LoadRoom<'a> {
    /// The room at `reading` of `ledgers`, its search's heap held from now.
    pub fn new(reading: &'a DeviceReading, ledgers: &'a DeviceLedgers, release: crate::platform::ReleaseRate) -> Self {
        let host = ledgers.host();
        let grant = host
            .change(|keeping| host.grant(&keeping.claims(), crate::memory::ledger::GrantRole::Tuning, 0))
            .expect("invariant violated: a grant of no bytes is refused");
        grant.attach(seismic::ProcessId::current());
        Self { reading, ledgers, release, grant }
    }
}

impl crate::TuningRoom for LoadRoom<'_> {
    /// The room at the load's reading with every holding since counted,
    /// the search's heap grown since it began among them: reads no OS
    /// figure but this process's footprint, so a search's decisions share
    /// the load's one reading.
    fn now(&self, device: &Device) -> crate::RoomNow {
        self.ledgers.changes().back(|claiming| self.grant.hold_growth(&claiming));
        let at = self.reading.room_after(self.ledgers);
        crate::RoomNow {
            device_bytes: at.device.room(),
            host_bytes: at.host.as_ref().unwrap_or(&at.device).room(),
            host_shared: crate::memory::ledger::shares_host(device),
        }
    }

    fn release(&self) -> crate::platform::ReleaseRate {
        self.release
    }
}

impl DeviceReader {
    /// The reading of `device`, observed once, for a load.
    pub fn open(
        reading: Rc<dyn StatusReading>,
        release: ReleaseRate,
        device: Rc<Device>,
    ) -> Result<Self, MemoryPolicyError> {
        let status = reading.status(&device);
        let reader = Self { reading, release, device, pressured: Arc::new(AtomicBool::new(false)), watch: Watch::Unasked };
        reader.observe(&status)?;
        Ok(reader)
    }

    pub fn device(&self) -> &Rc<Device> {
        &self.device
    }

    /// The machine's release rate, which a load's search prices giving its
    /// inputs up by ([`LoadRoom::release`]).
    pub fn release_rate(&self) -> crate::platform::ReleaseRate {
        self.release
    }

    /// Observe every domain the device uses from a sample read now.
    pub fn refresh(&mut self) -> Result<Vec<DomainReading>, MemoryPolicyError> {
        let status = self.reading.status(&self.device);
        self.observe(&status)
    }

    /// Watch the OS's memory pressure events from now on, where it gives
    /// any, through `host`, host RAM's ledger, whose store has the one
    /// watch every load in the process shares, each event counted there
    /// once (`memory::ledger::Ledger::watch_pressure`): each calls `wake`
    /// (the owner's), and marks the reader pressured until a reading asks
    /// ([`Self::pressured`]).
    ///
    /// # Panics
    /// Where it was asked already: one owner watches once.
    pub fn watch_pressure(&mut self, host: &crate::memory::ledger::Ledger, wake: impl Fn() + Send + Sync + 'static) {
        assert!(matches!(self.watch, Watch::Unasked), "invariant violated: a reader watches memory pressure twice");
        let pressured = self.pressured.clone();
        let watched = host.watch_pressure(move || {
            pressured.store(true, Ordering::Release);
            wake();
        });
        self.watch = match watched {
            Some(watch) => Watch::Events { _watch: watch },
            None => Watch::Unwatchable,
        };
    }

    /// Whether an OS pressure event came since this was last asked.
    pub fn pressured(&self) -> bool {
        self.pressured.swap(false, Ordering::AcqRel) || self.reading.take_pressure()
    }

    /// The pressure event [`Self::pressured`] just took stays owed: the
    /// reading it was taken for did not stand.
    pub fn repress(&self) {
        self.pressured.store(true, Ordering::Release);
    }

    /// The reading of `ledgers` at an owner's event, from a sample taken
    /// now on the owner's thread, which never waits
    /// ([`StatusReading::sample`]). None stands beside a change of ours off
    /// the owner's thread; the ledgers then keep their last reading, and the
    /// change's completion, which wakes the owner, is the event that reads
    /// again.
    #[must_use = "an owner decides only on a reading that stood"]
    pub fn read_at_event(&self, ledgers: &DeviceLedgers) -> Option<DeviceReading> {
        self.read_at_event_after(ledgers, false)
    }

    #[must_use = "an owner decides only on a reading that stood"]
    pub fn read_at_event_after(&self, ledgers: &DeviceLedgers, after_idle: bool) -> Option<DeviceReading> {
        let sample = self.reading.sample(&self.device, ledgers.changes())?;
        Some(ledgers.read_after(&self.device, &sample, after_idle))
    }

    /// The reading a load holds from its start: `ledgers` read from a
    /// sample taken with no change of ours beside it, waiting for one under
    /// way to end (`memory::ledger::Sample::settled`). The
    /// load plans, searches and claims at it (design §5).
    pub fn settled_read(&self, ledgers: &DeviceLedgers) -> DeviceReading {
        self.settled_read_after(ledgers, false)
    }

    /// [`Self::settled_read`] with `after_idle` marking a baseline boundary
    /// (such as post-load/post-warm-up) so one-time initialization overhead is
    /// absorbed into `others` and `budget` without poisoning `guarantee`
    /// (`read.rise`) when no OS memory pressure events occurred.
    pub fn settled_read_after(&self, ledgers: &DeviceLedgers, after_idle: bool) -> DeviceReading {
        ledgers.read_after(&self.device, &self.settled_sample(ledgers.changes()), after_idle)
    }

    /// [`Self::settled_read`] of ledgers never read, opened with the
    /// device: their first reading, which gives the ledgers a load claims
    /// in ([`UnreadLedgers::read`]).
    pub fn read_opened(&self, ledgers: UnreadLedgers) -> (DeviceLedgers, DeviceReading) {
        let sample = self.settled_sample(ledgers.changes());
        ledgers.read(&self.device, &sample)
    }

    /// A sample of the device's stores beside `changes`, its ledgers'
    /// scope's, taken with none of them under way
    /// (`memory::ledger::Sample::settled`).
    fn settled_sample(&self, changes: &Changes) -> Sample {
        Sample::settled(changes, &self.device, || self.reading.now(), || self.reading.status(&self.device))
    }

    /// The readings of every domain the device uses in the status sample
    /// `status`.
    fn observe(&self, status: &seismic::Status) -> Result<Vec<DomainReading>, MemoryPolicyError> {
        crate::platform::domain_readings(&self.device, status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::ledger::{CommittedKind, DeviceLedgers, Ledgers, LoadShape, StoreShape};
    use crate::memory::ClaimDemand;
    use seismic::BackendName;
    use std::cell::Cell;

    const MIB: u64 = 1 << 20;

    /// Host RAM as a test reads it: the machine's status as first read,
    /// still from then on (no figure of the live machine moves a test's
    /// reading), its supply the test's: `free` above the OS's free target
    /// and `clean` file cache, nothing cold, nothing of the process's out
    /// of RAM, at no pressure, every figure taken at the sample's instant
    /// and lagging by nothing, so each change of ours before a sample shows
    /// in it (the live machine's lag, a binding cgroup's 2 s flush, and its
    /// first counters' instant would hide them).
    struct Room {
        first: seismic::Status,
        free: Rc<Cell<u64>>,
        clean: Rc<Cell<u64>>,
    }

    impl Room {
        fn new(catalog: &Devices, device: &Device, free: u64, clean: u64) -> Self {
            Self { first: catalog.device_status(device), free: Rc::new(Cell::new(free)), clean: Rc::new(Cell::new(clean)) }
        }
    }

    impl StatusReading for Room {
        fn status(&self, _: &Device) -> seismic::Status {
            let mut status = self.first.clone();
            let host = &mut status.sample.host;
            let now = self.now();
            host.at = now;
            host.counted = seismic::Counted { at: now, lag: std::time::Duration::ZERO };
            host.lag = crate::domain::fixtures::NO_LAG;
            host.supply = seismic::Supply { free_beyond_target: i64::try_from(self.free.get()).expect("bytes fit i64"), clean_cache: self.clean.get(), cold_reclaimable: 0 };
            host.own_reclaimed = Some(seismic::OwnReclaimed::NONE);
            host.pressure = Some(seismic::PressureLevel::Normal);
            status
        }
    }

    /// The CPU device of a catalog of the machine, as a test opens it.
    fn cpu() -> (Rc<Devices>, Rc<Device>) {
        let catalog = Rc::new(Devices::discover(&[]).unwrap());
        let device = Rc::new(crate::test_support::open_backend(&catalog, BackendName::Cpu).unwrap());
        (catalog, device)
    }

    /// A load holding no state store.
    fn stateless() -> std::sync::Arc<LoadShape> {
        std::sync::Arc::new(LoadShape { stores: Vec::new() })
    }

    /// One store of 1024-wide F32 rows at a context of 1024, and its
    /// page's rows.
    fn paged() -> (std::sync::Arc<LoadShape>, u64) {
        let component = magnitude_state::ComponentDescriptor::new(
            magnitude_state::LayerRef::Target(0),
            magnitude_state::CodecSpec::dense(seismic::DType::F32, 1024, 1024),
            1,
        )
        .expect("a component");
        let domains = vec![magnitude_state::HistoryDomainLayout::Token { components: vec![component] }];
        let page_rows = domains[0].pages(&domains, 1024, 16).expect("the pages").page_rows as u64;
        (std::sync::Arc::new(LoadShape { stores: vec![StoreShape::paged(domains, 1024, 16)] }), page_rows)
    }

    /// Two readers of the process's one open device read its store's one
    /// ledger: neither moves the other's room apart from what the ledger
    /// holds, and a claim made beside one reader's reading is counted once
    /// at the other's. (Before the ledger was the one count, each reader
    /// bound its own copy of the device's room, and the second to take the
    /// right panicked.)
    #[test]
    fn a_second_reader_on_a_shared_device_reads_the_one_ledger() {
        let (catalog, device) = cpu();
        let reading = Rc::new(Room::new(&catalog, &device, 64 * MIB, 0));
        let first = DeviceReader::open(reading.clone(), ReleaseRate::of_machine(None), device.clone()).unwrap();
        let second = DeviceReader::open(reading, ReleaseRate::of_machine(None), device.clone()).unwrap();
        let scope = Ledgers::new();
        let (ours, _) = first.read_opened(DeviceLedgers::open_in(&scope, &device, stateless()));
        let _held = ours.changes().change(|keeping| ours.claim(&keeping.claims(), &device, CommittedKind::Load, ClaimDemand { device: 16 * MIB, ..Default::default() })).expect("the ledger admits it");
        let (theirs, _) = second.read_opened(DeviceLedgers::open_in(&scope, &device, stateless()));
        assert_eq!(theirs.device().estimate().room(), 48 * MIB, "the claim beside the first reader is counted once at the second's");
        assert_eq!(ours.device().estimate().room(), theirs.device().estimate().room(), "one ledger: one room");
    }

    /// What the ledger admits, Seismic allocates, charged once to the claim
    /// that admitted it: the ledger is the one authority on room and the
    /// one count of the process's memory. With 64 MiB of host RAM free, the
    /// ledger admits a 48 MiB allocation, and Seismic makes it. (Seismic's
    /// bound set from the planning line, the room less its reserve, refused
    /// it as a load's expert bank was refused after its admission:
    /// "allocation requires 929955840 bytes; 60887171 bytes available".)
    #[test]
    fn seismic_allocates_what_the_ledger_admits() {
        let (catalog, device) = cpu();
        let reader = DeviceReader::open(Rc::new(Room::new(&catalog, &device, 64 * MIB, 0)), ReleaseRate::of_machine(None), device.clone()).unwrap();
        let (ledgers, _) = reader.read_opened(DeviceLedgers::open_in(&Ledgers::new(), &device, stateless()));
        let bytes = 48 * MIB;
        let mut held = ledgers.changes().change(|keeping| ledgers.claim(&keeping.claims(), &device, CommittedKind::Load, ClaimDemand { device: bytes, ..Default::default() })).expect("the ledger admits it");
        let _tensor = held
            .allocate(&device, |charges| seismic::Tensor::zeros(charges, seismic::Element::bool(), &[bytes]))
            .expect("Seismic allocates what the ledger admitted");
        let estimate = ledgers.device().estimate();
        assert_eq!(estimate.committed(), bytes);
        assert_eq!(estimate.allocated(), held.claims().iter().map(|claim| claim.allocated_bytes()).sum::<u64>(), "the allocation is charged to its claim alone");
        assert!(estimate.allocated() >= bytes);
    }

    /// Room a claim holds stays Seismic's to allocate through any reading
    /// taken before the allocation is made: a load claims what it
    /// allocates after its target import before the import reads the
    /// ledgers, and the import's bank fills the rest. With 64 MiB free, a
    /// 48 MiB claim read before it is allocated still allocates, and the
    /// reading holds its room for it. (Bounded to free room alone, a load's
    /// bound target graphs were refused once the import had read:
    /// "allocation requires 684195840 bytes; 32205612 bytes available".)
    #[test]
    fn seismic_allocates_what_a_claim_holds_across_a_reading() {
        let (catalog, device) = cpu();
        let reader = DeviceReader::open(Rc::new(Room::new(&catalog, &device, 64 * MIB, 0)), ReleaseRate::of_machine(None), device.clone()).unwrap();
        let (ledgers, _) = reader.read_opened(DeviceLedgers::open_in(&Ledgers::new(), &device, stateless()));
        let bytes = 48 * MIB;
        let mut held = ledgers.changes().change(|keeping| ledgers.claim(&keeping.claims(), &device, CommittedKind::Load, ClaimDemand { device: bytes, ..Default::default() })).expect("the ledger admits it");
        assert!(reader.read_at_event(&ledgers).is_some());
        assert_eq!(ledgers.device().estimate().room(), 16 * MIB, "the reading leaves the claim its room");
        let _tensor = held
            .allocate(&device, |charges| seismic::Tensor::zeros(charges, seismic::Element::bool(), &[bytes]))
            .expect("Seismic allocates what the claim holds");
    }

    /// A claim's room is Seismic's to allocate the moment it is taken, with
    /// no reading between the claim and the allocation: a load's claim of
    /// 48 MiB, 16 MiB of it free and the rest clean file cache, which is
    /// supply the OS drops without writing, allocates at once. (A bound
    /// stored at the reading before the claim left 16 MiB less its
    /// reserve, and refused the import: "allocation requires 1479475200
    /// bytes; 1365841544 bytes available", the ledger then holding 11 GB
    /// for it.)
    #[test]
    fn seismic_allocates_what_a_claim_takes_with_no_reading_between() {
        let (catalog, device) = cpu();
        let reader = DeviceReader::open(Rc::new(Room::new(&catalog, &device, 16 * MIB, 48 * MIB)), ReleaseRate::of_machine(None), device.clone()).unwrap();
        let (ledgers, _) = reader.read_opened(DeviceLedgers::open_in(&Ledgers::new(), &device, stateless()));
        let bytes = 48 * MIB;
        let mut held = ledgers.changes().change(|keeping| ledgers.claim(&keeping.claims(), &device, CommittedKind::Load, ClaimDemand { device: bytes, ..Default::default() })).expect("the clean file cache is supply");
        let _tensor = held
            .allocate(&device, |charges| seismic::Tensor::zeros(charges, seismic::Element::bool(), &[bytes]))
            .expect("Seismic allocates what the claim took");
    }

    /// A state growth reaches Seismic through the slab path with no reading
    /// between its reservation and its allocation: the rows a request
    /// reserved and the slab's rounding past them (`StateSlabs::reserve`,
    /// held from free room) are one allocation of the store's pool, whose
    /// bytes the ledger holds before it exists, and once it does the
    /// charge equals the allocations: the rows committed, the rounding
    /// spare cache. (Counted allocated at its reservation, the rounding
    /// left Seismic room for the rows alone, and the slab was refused:
    /// "allocation requires 117793984 bytes; 88004412 bytes available".)
    #[test]
    fn a_state_growth_with_rounding_is_allocated_through_the_slab_path() {
        let (shape, page_rows) = paged();
        let page = shape.state_bytes(page_rows);
        let (weights, rounding) = (3 * page, page / 2);
        let (catalog, device) = cpu();
        // The room holds the weights, the rows and the rounding.
        let room = page + weights + rounding;
        let reader = DeviceReader::open(Rc::new(Room::new(&catalog, &device, room, 0)), ReleaseRate::of_machine(None), device.clone()).unwrap();
        let (ledgers, _) = reader.read_opened(DeviceLedgers::open_in(&Ledgers::new(), &device, shape));

        let mut import = ledgers.changes().change(|keeping| ledgers.claim(&keeping.claims(), &device, CommittedKind::Load, ClaimDemand { device: weights, ..Default::default() })).expect("the weights are admitted");
        let _weights = import
            .allocate(&device, |charges| seismic::Tensor::zeros(charges, seismic::Element::bool(), &[weights]))
            .expect("Seismic allocates the import");

        let ledger = ledgers.device();
        let pool = ledger.pool_block(crate::memory::ledger::PoolClass::State, std::num::NonZeroU64::new(page).unwrap()).blocks;
        let slot = ledgers
            .changes()
            .change(|keeping| ledgers.claim_request_slot(&keeping.claims(), &device, ClaimDemand { device: page, ..Default::default() }))
            .expect("the request slot fits the free room");
        let mut rows = ledger.change(|keeping| ledger.reserve(&keeping.claims(), &slot, crate::RequestId(1), page_rows)).expect("the first request draws on the slot");
        let mut slabs = crate::domain::state::StateSlabs::default();
        ledger.change(|_| slabs.settle(ledger, 0, 0, 0, page));
        let growth = page + rounding;
        ledger.change(|keeping| slabs.reserve(&keeping.claims(), ledger, growth)).expect("the rounding past the rows fits the free room");
        let _slab = seismic::Tensor::zeros(&device.charges(|_| pool.clone()), seismic::Element::bool(), &[growth])
            .expect("Seismic allocates the reserved slab");
        rows.grow(page_rows);
        ledger.change(|_| slabs.settle(ledger, growth, 0, page, rows.bytes() + slot.bytes()));
        let estimate = ledger.estimate();
        assert_eq!(estimate.committed() + estimate.cache(), estimate.allocated(), "the charge equals the allocations once the slab exists");
        assert_eq!(estimate.cache_of(crate::memory::ledger::CacheKind::Spare), rounding, "the rounding is spare");
    }

    /// A load's whole admission → import → first state claim sequence, at
    /// the laptop's figures: the import past the free room takes the clean
    /// file cache, the room is tight once the weights and the floor fill
    /// it, and no reading is taken between a claim and the allocation it
    /// admits. Each step's allocation succeeds in Seismic, and the charge
    /// equals the allocations at the end: the request slot is held once,
    /// the first request's reservation drawn from it. A
    /// group's staging beyond the floor's is claimed as the group opens,
    /// before it is allocated: refused while the room is tight, admitted
    /// and allocated at once when room comes back.
    #[test]
    fn a_load_sequence_allocates_what_each_claim_admits_at_the_laptops_figures() {
        let (shape, page_rows) = paged();
        let page = shape.state_bytes(page_rows);
        let (floor_bytes, weights, staging) = (page, 5 * page, page);

        let (catalog, device) = cpu();
        // Free room holds the floor and the weights less one page; the
        // clean file cache holds that page.
        let reading = Room::new(&catalog, &device, floor_bytes + weights - page, page);
        let (free, clean) = (reading.free.clone(), reading.clean.clone());
        let reader = DeviceReader::open(Rc::new(reading), ReleaseRate::of_machine(None), device.clone()).unwrap();
        let (ledgers, _) = reader.read_opened(DeviceLedgers::open_in(&Ledgers::new(), &device, shape));
        let ledger = ledgers.device();
        let pool = ledger.pool_block(crate::memory::ledger::PoolClass::State, std::num::NonZeroU64::new(page).unwrap()).blocks;

        // The load claims what it needs to serve, and allocates each claim
        // at once.
        let state_floor = ledgers
            .changes()
            .change(|keeping| ledgers.claim_request_slot(&keeping.claims(), &device, ClaimDemand { device: floor_bytes, ..Default::default() }))
            .expect("the load's request slot is admitted");
        let mut import = ledgers
            .changes()
            .change(|keeping| ledgers.claim(&keeping.claims(), &device, CommittedKind::Load, ClaimDemand { device: weights, ..Default::default() }))
            .expect("the weights, past the free room, take the clean file cache");
        let _weights = import
            .allocate(&device, |charges| seismic::Tensor::zeros(charges, seismic::Element::bool(), &[weights]))
            .expect("Seismic allocates the import");

        // The first request draws its rows from the slot: nothing more
        // held, and the page it allocates in the pool is admitted.
        let _state = ledger.change(|keeping| ledger.reserve(&keeping.claims(), &state_floor, crate::RequestId(1), page_rows)).expect("the first request draws on the slot");
        let _page = seismic::Tensor::zeros(&device.charges(|_| pool.clone()), seismic::Element::bool(), &[page])
            .expect("Seismic allocates the first page");
        let estimate = ledger.estimate();
        assert_eq!(estimate.committed(), estimate.allocated(), "the charge equals the allocations: the slot is held once");
        assert_eq!(estimate.committed(), weights + page);
        assert!(ledger.change(|keeping| ledger.reserve(&keeping.claims(), &state_floor, crate::RequestId(2), page_rows)).is_err(), "a second request has no room");

        // A group's staging beyond the floor's: refused while the room is
        // tight, with nothing held; admitted, then allocated with no
        // reading between, once room is back.
        let unit = std::num::NonZeroU64::new(staging).unwrap();
        let (refused, units) = ledger.change(|keeping| ledger.claim_units(&keeping.claims(), CommittedKind::GroupClass, unit, 1));
        assert_eq!(units, 0, "the tight room holds no staging");
        drop(refused);
        // The allocations took the free room and the file cache: what is
        // free now is what others gave back.
        free.set(staging);
        clean.set(0);
        assert!(reader.read_at_event(&ledgers).is_some());
        let (group, units) = ledger.change(|keeping| ledger.claim_units(&keeping.claims(), CommittedKind::GroupClass, unit, 1));
        assert_eq!(units, 1, "the room that came back holds the staging");
        let _staging = seismic::Tensor::zeros(&device.charges(|_| group.entry().clone()), seismic::Element::bool(), &[staging])
            .expect("Seismic allocates the staging its claim admitted");
        let estimate = ledger.estimate();
        assert_eq!(estimate.committed(), estimate.allocated(), "the charge equals the allocations with the group's staging");
        drop(state_floor);
    }

    /// A load's search reads its room from the load's ledgers at the
    /// reading the load holds, the authority its claims read: with 64 MiB
    /// of host RAM free, the search's room is the 64 MiB, in host RAM the
    /// device's storage shares; a claim the load makes is counted at once,
    /// with no reading taken.
    #[test]
    fn a_search_reads_the_ledgers_room() {
        let (catalog, device) = cpu();
        let reader = DeviceReader::open(Rc::new(Room::new(&catalog, &device, 64 * MIB, 0)), ReleaseRate::of_machine(None), device.clone()).unwrap();
        let (ledgers, reading) = reader.read_opened(DeviceLedgers::open_in(&Ledgers::new(), &device, stateless()));
        let load = LoadRoom::new(&reading, &ledgers, reader.release_rate());
        let room: &dyn crate::TuningRoom = &load;
        assert_eq!(room.now(&device), crate::RoomNow { device_bytes: 64 * MIB, host_bytes: 64 * MIB, host_shared: true });
        let _held = ledgers.changes().change(|keeping| ledgers.claim_at(&keeping.claims(), &reading, &device, CommittedKind::Load, ClaimDemand { device: 16 * MIB, ..Default::default() })).expect("the ledger admits it");
        assert_eq!(room.now(&device).device_bytes, 48 * MIB, "the claim is counted at the held reading");
    }

    /// A load prices the reading it holds (the real `read_opened` path) by
    /// the room its claims read
    /// (`DeviceReading::room_after`): with nothing held since, the reading's
    /// own, then less exactly what the claims took. A reading another load or event takes later moves neither, as
    /// the load holds its own.
    #[test]
    fn a_loads_claims_read_the_room_of_the_reading_it_holds() {
        let (catalog, device) = cpu();
        let room = Room::new(&catalog, &device, 64 * MIB, 32 * MIB);
        let free = room.free.clone();
        let reader = DeviceReader::open(Rc::new(room), ReleaseRate::of_machine(None), device.clone()).unwrap();
        let (ledgers, reading) = reader.read_opened(DeviceLedgers::open_in(&Ledgers::new(), &device, stateless()));
        let plan = reading.room_after(&ledgers);
        assert_eq!(plan.device.room(), 96 * MIB, "the supply");
        let _held = ledgers.changes().change(|keeping| ledgers.claim_at(&keeping.claims(), &reading, &device, CommittedKind::Load, ClaimDemand { device: 24 * MIB, ..Default::default() })).expect("the plan's room holds it");
        assert_eq!(reading.room_after(&ledgers).device.room(), plan.device.room() - 24 * MIB);
        free.set(0);
        assert!(reader.read_at_event(&ledgers).is_some(), "a later event reads the store");
        assert_eq!(reading.room_after(&ledgers).device.room(), plan.device.room() - 24 * MIB, "the load's reading stands");
        assert!(ledgers.changes().change(|keeping| ledgers.claim_at(&keeping.claims(), &reading, &device, CommittedKind::Load, ClaimDemand { device: 60 * MIB, ..Default::default() })).is_ok(), "claims at the held reading");
    }

    /// Opening a store's granule pool claims and allocates its absent granule
    /// at the load's reading: when opened before `admit_most_pair` sizes an
    /// expert bank into the remaining room, the bank's admission sees the
    /// absent granules already held and never leaves the pool short; when the
    /// store genuinely lacks room for the absent granule, `open_pool_at`
    /// returns `OpenPoolError::Short` rather than an internal allocation error.
    #[test]
    fn open_pool_at_claims_absent_granule_before_bank_admission() {
        let (catalog, device) = cpu();
        let granule = 16 * MIB;
        let slot = 10 * MIB;
        // Room holds two pools' absent granules (2 × 16 MiB = 32 MiB) plus
        // 3 bank slots (30 MiB) and 2 MiB spare (< 1 slot).
        let room = 2 * granule + 3 * slot + 2 * MIB;
        let reader = DeviceReader::open(Rc::new(Room::new(&catalog, &device, room, 0)), ReleaseRate::of_machine(None), device.clone()).unwrap();
        let (ledgers, reading) = reader.read_opened(DeviceLedgers::open_in(&Ledgers::new(), &device, stateless()));

        let state_pool = ledgers
            .open_pool_at(Some(&reading), &device, seismic::Contract::DeviceOnly, crate::memory::ledger::PoolClass::State, granule)
            .expect("state pool opens and claims its absent granule");
        let records_pool = ledgers
            .open_pool_at(Some(&reading), &device, seismic::Contract::DeviceOnly, crate::memory::ledger::PoolClass::Records, granule)
            .expect("records pool opens and claims its absent granule");
        assert_eq!(reading.room_after(&ledgers).device.room(), 3 * slot + 2 * MIB, "both absent granules are held before bank admission");

        let admitted = ledgers
            .changes()
            .change(|keeping| {
                ledgers.admit_most_pair(
                    &keeping.claims(),
                    &reading,
                    &device,
                    CommittedKind::Load,
                    (1, 10),
                    (0, 0),
                    |slots, _| Ok::<_, std::convert::Infallible>(ClaimDemand { device: slots * slot, ..Default::default() }),
                )
            })
            .unwrap()
            .expect("bank fits into what the pools leave");
        assert_eq!(admitted.0, 3, "bank takes the 3 slots left after both pools' absent granules");
        drop((admitted, records_pool, state_pool));

        // Genuine shortage returns OpenPoolError::Short (not Internal).
        let tight = DeviceReader::open(Rc::new(Room::new(&catalog, &device, granule / 2, 0)), ReleaseRate::of_machine(None), device.clone()).unwrap();
        let (tight_ledgers, tight_reading) = tight.read_opened(DeviceLedgers::open_in(&Ledgers::new(), &device, stateless()));
        let Err(err) = tight_ledgers
            .open_pool_at(Some(&tight_reading), &device, seismic::Contract::DeviceOnly, crate::memory::ledger::PoolClass::State, granule)
        else {
            panic!("tight store must refuse the absent granule");
        };
        assert!(matches!(err, crate::memory::ledger::OpenPoolError::Short { .. }), "shortage reports typed Short, got: {err}");
    }

    /// An import's upload window (`CommittedKind::Upload`) is a reusable peak
    /// reservation across batches: freeing one batch's workspace must keep
    /// the window's committed bytes intact so a later batch allocates inside
    /// it even when free room has vanished, and dropping the window releases
    /// it whole.
    #[test]
    fn upload_window_preserves_reservation_across_batches() {
        let (catalog, device) = cpu();
        let window = 32 * MIB;
        let reader = DeviceReader::open(Rc::new(Room::new(&catalog, &device, window, 0)), ReleaseRate::of_machine(None), device.clone()).unwrap();
        let (ledgers, reading) = reader.read_opened(DeviceLedgers::open_in(&Ledgers::new(), &device, stateless()));
        let mut upload = ledgers
            .changes()
            .change(|keeping| ledgers.claim_at(&keeping.claims(), &reading, &device, CommittedKind::Upload, ClaimDemand { device: window, ..Default::default() }))
            .expect("upload window fits");
        // Batch 1 allocates and drops a full-window workspace.
        let batch1 = upload
            .allocate(&device, |charges| seismic::Tensor::zeros(charges, seismic::Element::bool(), &[window]))
            .expect("batch 1 allocates inside upload window");
        drop(batch1);
        assert_eq!(ledgers.device().estimate().committed(), window, "upload window keeps its committed reservation after batch 1 drops");
        // Batch 2 allocates another full-window workspace inside the same window.
        let batch2 = upload
            .allocate(&device, |charges| seismic::Tensor::zeros(charges, seismic::Element::bool(), &[window]))
            .expect("batch 2 reuses upload window reservation");
        drop(batch2);
        drop(upload);
        assert_eq!(ledgers.device().estimate().committed(), 0, "dropping upload window releases its reservation");
    }
}
