//! The memory reading of one opened device: the machine's status, sampled
//! now or published periodically off the owner's thread, the domains the
//! device uses observed in it, and the device's store ledgers read from it,
//! which alone bound Seismic's ledgers. It holds the one right to bound
//! them while it lives.

use crate::platform::{DomainReading, MemoryPolicyError, MemoryReserves};
use seismic::{Device, Devices};
use std::rc::Rc;
use std::sync::{Arc, Condvar, Mutex};
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

    /// Host RAM's available memory at `at`, the ledgers' one OS input
    /// (`memory::ledger::DeviceLedgers::read`): the live machine's reading
    /// ([`crate::memory::ledger::Available::host`]); a test's, its own.
    fn available(&self, at: Instant) -> Result<crate::memory::ledger::Available, crate::memory::ledger::Blind> {
        crate::memory::ledger::Available::host(at)
    }

    /// The periodic reading taken since the last one this returned, without
    /// waiting for one ([`DeviceReader::refresh_published`]); none where none
    /// was published since. By default it reads now: a test's readings are
    /// taken where the test asks for them. The live machine's are read on a
    /// thread of their own ([`MachineReading`]).
    fn published(&self, device: &Device) -> Option<Published> {
        Some(Published { status: self.status(device), at: self.now() })
    }
}

/// One periodic reading of the machine ([`StatusReading::published`]): its
/// status and when it was read.
pub struct Published {
    pub status: seismic::Status,
    pub at: Instant,
}

impl StatusReading for Devices {
    fn status(&self, device: &Device) -> seismic::Status {
        self.device_status(device)
    }
}

/// The live machine's readings: a sample read where it is asked for, and
/// the periodic ones on a thread of their own every
/// [`crate::platform::READING_INTERVAL`], each published for
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
                let mut state = reading.lock();
                state.latest = Some(Published { status, at: Instant::now() });
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

/// The memory reading of one opened device together with the host's
/// threshold policy: it exists from the device's opening to the load's end,
/// and Seismic's enforced limit is the device's store ledgers' as each
/// reading of them leaves it ([`Self::read_ledgers`]).
pub struct DeviceReader {
    reading: Rc<dyn StatusReading>,
    /// The reserves the reader opened with: the machine's figures.
    reserves: MemoryReserves,
    device: Rc<Device>,
    /// The right to bound the stores the device reaches, the reader's alone
    /// while it lives.
    bounds: seismic::Bounds,
}

/// The room a load's search is held to: what the device's store ledgers
/// leave free at a reading taken for each decision
/// ([`DeviceReader::read_ledgers_now`]), less the speed guarantee's rise
/// of others' memory (`memory::ledger::Snapshot::guarantee`; a search
/// grows no sequence's state). The one authority on room the load's
/// allocations read too. A reading that fails reads no room: missing
/// observations never authorize allocation.
pub struct LoadRoom<'a> {
    pub reader: &'a DeviceReader,
    pub ledgers: &'a crate::memory::ledger::DeviceLedgers,
}

impl crate::TuningRoom for LoadRoom<'_> {
    fn now(&self, device: &Device) -> crate::RoomNow {
        assert_eq!(device.info().selector(), self.reader.device.info().selector(), "a search reads the room of its load's device");
        self.reader.read_ledgers_now(self.ledgers);
        let room = |ledger: &crate::memory::ledger::Ledger| {
            let snapshot = ledger.snapshot();
            snapshot.free().saturating_sub(snapshot.guarantee(0))
        };
        let (device, host) = (self.ledgers.device(), self.ledgers.host());
        crate::RoomNow { device_bytes: room(device), host_bytes: room(host), host_shared: device.store() == host.store() }
    }
}

impl DeviceReader {
    /// The reading of `device`, observed once, for a load.
    pub fn open(
        reading: Rc<dyn StatusReading>,
        reserves: MemoryReserves,
        device: Rc<Device>,
    ) -> Result<Self, MemoryPolicyError> {
        let status = reading.status(&device);
        let bounds = device.bounds();
        let reader = Self { reading, reserves, device, bounds };
        reader.observe(&status)?;
        Ok(reader)
    }

    pub fn device(&self) -> &Rc<Device> {
        &self.device
    }

    /// The reader's right to bound the device's ledgers.
    #[cfg(test)]
    pub(crate) fn bounds(&self) -> &seismic::Bounds {
        &self.bounds
    }

    /// Observe every domain the device uses from a sample read now.
    pub fn refresh(&mut self) -> Result<Vec<DomainReading>, MemoryPolicyError> {
        let status = self.reading.status(&self.device);
        self.observe(&status)
    }

    /// [`Self::refresh`] from the periodic reading published since the last
    /// ([`StatusReading::published`]), with the status and the time it was
    /// read, which the device's ledgers read too
    /// (`memory::ledger::DeviceLedgers::read`); none, refreshing nothing,
    /// where none was published since.
    pub fn refresh_published(&mut self) -> Option<(Result<Vec<DomainReading>, MemoryPolicyError>, seismic::Status, Instant)> {
        let Published { status, at } = self.reading.published(&self.device)?;
        Some((self.observe(&status), status, at))
    }

    /// Read `ledgers` from a sample read now ([`Self::read_ledgers`]).
    pub fn read_ledgers_now(&self, ledgers: &crate::memory::ledger::DeviceLedgers) {
        let at = self.reading.now();
        self.read_ledgers(ledgers, &self.reading.status(&self.device), at);
    }

    /// Read `ledgers` from `status`, sampled at `at`, with host RAM's
    /// available memory then, and bound Seismic's ledger of each store they
    /// keep to what it claims now and what the ledger lets the process
    /// allocate beside it (`memory::ledger::Snapshot::allocatable`): the
    /// room its claims hold for allocations not yet made and the room it
    /// leaves free, less cache that yielded to a claim and is not freed
    /// yet; none where the reading failed. The ledger is the one authority
    /// on room, so Seismic refuses no allocation it admits, nor takes one it
    /// refuses, and a claim that took cache allocates it as it is freed.
    pub fn read_ledgers(&self, ledgers: &crate::memory::ledger::DeviceLedgers, status: &seismic::Status, at: Instant) {
        ledgers.read(&self.device, status, self.reading.available(at), at);
        self.bind_ledgers(ledgers);
    }

    /// Bound Seismic's ledger of each store `ledgers` keep to what it
    /// claims and what the ledger lets the process allocate beside it
    /// (`memory::ledger::Snapshot::allocatable`), read from the ledger at
    /// each admission against the store, never stored: no claim, release or
    /// reading goes unseen by the admission that follows it.
    fn bind_ledgers(&self, ledgers: &crate::memory::ledger::DeviceLedgers) {
        let topology = self.device.topology();
        for view in self.device.ledgers().stores {
            let identity = topology.store_identity(view.store).expect("a device's store ledger is of its topology");
            let Some(ledger) = ledgers.keeping(&identity).cloned() else {
                continue;
            };
            self.bounds
                .bound_by(view.store, move |claim| Some(claim.saturating_add(ledger.snapshot().allocatable())))
                .expect("invariant violated: a device bounds the stores it reaches");
        }
    }

    /// The readings of every domain the device uses in the status sample
    /// `status`.
    fn observe(&self, status: &seismic::Status) -> Result<Vec<DomainReading>, MemoryPolicyError> {
        let readings = crate::platform::domain_readings(&self.device, status, &self.reserves);
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
        readings
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::BackendName;

    /// Two readers of the process's one shared open of an executor bound its
    /// ledgers from their own readings, each moving the other's room: the
    /// second to take the right to bound them panics at the first it
    /// takes, its path account, naming both. A test opens a device of its
    /// own (`test_support::open`) and takes the right alone.
    #[test]
    #[should_panic(expected = "a path ledger is bounded by two live holders")]
    fn a_second_reader_on_a_shared_device_panics() {
        let catalog = Rc::new(Devices::discover(&[]).unwrap());
        let device = Rc::new(catalog.open_backend(BackendName::Cpu).unwrap());
        let _first = DeviceReader::open(catalog.clone(), MemoryReserves::of_machine(None), device.clone()).unwrap();
        let _second = DeviceReader::open(catalog, MemoryReserves::of_machine(None), device);
    }

    /// Host RAM as a test reads it: the machine's status, its allocatable
    /// memory `room`, which the ledger's reading of host RAM gives too.
    struct Room {
        catalog: Rc<Devices>,
        room: u64,
        /// Clean file-backed pages beyond `room`, which only a load counts.
        reclaimable: u64,
    }

    impl StatusReading for Room {
        fn status(&self, device: &Device) -> seismic::Status {
            let mut status = self.catalog.device_status(device);
            let host = status.sample.host_ram.as_mut().expect("the test host reads its memory");
            host.allocatable.value = self.room;
            host.limits.clear();
            status
        }

        fn available(&self, at: Instant) -> Result<crate::memory::ledger::Available, crate::memory::ledger::Blind> {
            Ok(crate::memory::ledger::Available::new(1 << 40, self.room, None, at)
                .expect("the room is within the total")
                .with_reclaimable(self.reclaimable))
        }
    }

    /// What the ledger admits, Seismic allocates: the ledger is the one
    /// authority on room. With 64 MiB of host RAM free and a planning
    /// reserve of 32 MiB, the ledger admits a 48 MiB allocation, and Seismic
    /// makes it. (Seismic's bound set from the planning line, the room less
    /// its reserve, refused it as a load's expert bank was refused after its
    /// admission: "allocation requires 929955840 bytes; 60887171 bytes
    /// available".)
    #[test]
    fn seismic_allocates_what_the_ledger_admits() {
        const MIB: u64 = 1 << 20;
        let catalog = Rc::new(Devices::discover(&[]).unwrap());
        let device = Rc::new(crate::test_support::open_backend(&catalog, BackendName::Cpu).unwrap());
        let reserves = MemoryReserves::new(crate::platform::StoreMotion::new(32 * MIB - 1, 10.0), crate::platform::ReleaseRate::new(1e9));
        let reader = DeviceReader::open(Rc::new(Room { catalog, room: 64 * MIB, reclaimable: 0 }), reserves, device.clone()).unwrap();
        let ledgers = crate::memory::ledger::DeviceLedgers::open_in(
            &crate::memory::ledger::Ledgers::new(),
            &device,
            std::sync::Arc::new(crate::memory::ledger::LoadShape { stores: Vec::new() }),
        );
        reader.read_ledgers_now(&ledgers);
        let bytes = 48 * MIB;
        let _held = ledgers
            .claim(&device, crate::memory::ledger::CommittedKind::Load, crate::memory::ClaimDemand { device: bytes, ..Default::default() })
            .expect("the ledger admits it");
        seismic::Tensor::zeros(&device, seismic::Element::bool(), &[bytes]).expect("Seismic allocates what the ledger admitted");
    }

    /// Room a claim holds stays Seismic's to allocate through any reading
    /// taken before the allocation is made: a load claims what it
    /// allocates after its target import before the import reads the
    /// ledgers, and the import's bank fills the rest. With 64 MiB free, a
    /// 48 MiB claim read before it is allocated still allocates. (Bounded
    /// to free room alone, a load's bound target graphs were refused once
    /// the import had read: "allocation requires 684195840 bytes; 32205612
    /// bytes available".)
    #[test]
    fn seismic_allocates_what_a_claim_holds_across_a_reading() {
        const MIB: u64 = 1 << 20;
        let catalog = Rc::new(Devices::discover(&[]).unwrap());
        let device = Rc::new(crate::test_support::open_backend(&catalog, BackendName::Cpu).unwrap());
        let reserves = MemoryReserves::new(crate::platform::StoreMotion::new(32 * MIB - 1, 10.0), crate::platform::ReleaseRate::new(1e9));
        let reader = DeviceReader::open(Rc::new(Room { catalog, room: 64 * MIB, reclaimable: 0 }), reserves, device.clone()).unwrap();
        let ledgers = crate::memory::ledger::DeviceLedgers::open_in(
            &crate::memory::ledger::Ledgers::new(),
            &device,
            std::sync::Arc::new(crate::memory::ledger::LoadShape { stores: Vec::new() }),
        );
        reader.read_ledgers_now(&ledgers);
        let bytes = 48 * MIB;
        let mut held = ledgers
            .claim(&device, crate::memory::ledger::CommittedKind::Load, crate::memory::ClaimDemand { device: bytes, ..Default::default() })
            .expect("the ledger admits it");
        reader.read_ledgers_now(&ledgers);
        let _tensor = seismic::Tensor::zeros(&device, seismic::Element::bool(), &[bytes]).expect("Seismic allocates what the claim holds");
        held.allocated();
    }

    /// A claim's room is Seismic's to allocate the moment it is charged, with
    /// no reading between the claim and the allocation: a load's claim of
    /// 48 MiB, 16 MiB of it free and the rest clean file cache only a load
    /// counts, allocates at once. (A bound stored at the reading before the
    /// claim left 16 MiB less its reserve, and refused the import: "allocation
    /// requires 1479475200 bytes; 1365841544 bytes available", the ledger
    /// then holding 11 GB for it.)
    #[test]
    fn seismic_allocates_what_a_claim_takes_with_no_reading_between() {
        const MIB: u64 = 1 << 20;
        let catalog = Rc::new(Devices::discover(&[]).unwrap());
        let device = Rc::new(crate::test_support::open_backend(&catalog, BackendName::Cpu).unwrap());
        let reserves = MemoryReserves::new(crate::platform::StoreMotion::new(8 * MIB - 1, 10.0), crate::platform::ReleaseRate::new(1e9));
        let reader =
            DeviceReader::open(Rc::new(Room { catalog, room: 16 * MIB, reclaimable: 48 * MIB }), reserves, device.clone()).unwrap();
        let ledgers = crate::memory::ledger::DeviceLedgers::open_in(
            &crate::memory::ledger::Ledgers::new(),
            &device,
            std::sync::Arc::new(crate::memory::ledger::LoadShape { stores: Vec::new() }),
        );
        reader.read_ledgers_now(&ledgers);
        let bytes = 48 * MIB;
        let mut held = ledgers
            .claim(&device, crate::memory::ledger::CommittedKind::Load, crate::memory::ClaimDemand { device: bytes, ..Default::default() })
            .expect("a load counts the clean file cache");
        let _tensor = seismic::Tensor::zeros(&device, seismic::Element::bool(), &[bytes]).expect("Seismic allocates what the claim took");
        held.allocated();
    }

    /// A state growth reaches Seismic through the slab path with no reading
    /// between its reservation and its allocation: the rows a sequence
    /// claimed and the slab's rounding past them
    /// (`StateSlabs::reserve`, held from free room) are one allocation, whose
    /// bytes the ledger holds and has not allocated until it exists. (Counted
    /// allocated at its reservation, the rounding left Seismic room for the
    /// rows alone, and the slab was refused: "allocation requires 117793984
    /// bytes; 88004412 bytes available".)
    #[test]
    fn a_state_growth_with_rounding_is_allocated_through_the_slab_path() {
        use crate::memory::ledger::{CommittedKind, DeviceLedgers, Ledgers, LoadShape, Need, StoreShape};
        use crate::memory::ClaimDemand;

        let component = magnitude_state::ComponentDescriptor::new(
            magnitude_state::LayerRef::Target(0),
            magnitude_state::CodecSpec::dense(seismic::DType::F32, 1024, 1024),
            1,
        )
        .expect("a component");
        let domains = vec![magnitude_state::HistoryDomainLayout::Token { components: vec![component] }];
        let shape = std::sync::Arc::new(LoadShape { stores: vec![StoreShape { domains: domains.clone(), context: 1024, max_advance: 16 }] });
        let page_rows = domains[0].geometry(&domains, 1024, 16).expect("a geometry").page_rows as u64;
        let page = shape.state_bytes(page_rows);
        let (weights, rounding) = (3 * page, page / 2);

        let catalog = Rc::new(Devices::discover(&[]).unwrap());
        let device = Rc::new(crate::test_support::open_backend(&catalog, BackendName::Cpu).unwrap());
        // The growth view holds the weights, the rows and the rounding.
        let room = page + weights + rounding;
        let reserves = MemoryReserves::new(crate::platform::StoreMotion::new(1, 1.0), crate::platform::ReleaseRate::new(1e9));
        let reader = DeviceReader::open(Rc::new(Room { catalog, room, reclaimable: 0 }), reserves, device.clone()).unwrap();
        let ledgers = DeviceLedgers::open_in(&Ledgers::new(), &device, shape);
        reader.read_ledgers_now(&ledgers);

        let mut import = ledgers
            .claim(&device, CommittedKind::Load, ClaimDemand { device: weights, ..Default::default() })
            .expect("the weights are admitted");
        let _weights = seismic::Tensor::zeros(&device, seismic::Element::bool(), &[weights]).expect("Seismic allocates the import");
        import.allocated();

        let ledger = ledgers.device();
        let mut rows = ledger.claim(Need::StateRows { sequence: crate::RequestId(1), from: 0, to: 0 }).expect("an empty state claim");
        rows.grow(Need::StateRows { sequence: crate::RequestId(1), from: 0, to: page_rows }).expect("the first page fits the free room");
        let mut slabs = crate::domain::state::StateSlabs::new(0);
        let growth = page + rounding;
        slabs.reserve(ledger, growth, std::iter::once(&rows)).expect("the rounding past the rows fits the free room");
        let _slab = seismic::Tensor::zeros(&device, seismic::Element::bool(), &[growth]).expect("Seismic allocates the reserved slab");
        slabs.settle(ledger, growth, std::iter::once(&mut rows));
        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.committed() + snapshot.cache(), snapshot.allocated(), "the charge equals the allocations once the slab exists");
    }

    /// A load's whole admission → import → first state claim sequence, against
    /// a ledger at the laptop's figures: the clean file cache counts for the
    /// load, the growth view is tight (the weights and the floor fill it), and
    /// no reading is taken between a claim and the allocation it admits. Each
    /// step's allocation succeeds in Seismic, and the charge equals the
    /// allocations at the end: the state floor is held once, by the floor,
    /// until the first request's rows are claimed from it. A group's staging
    /// beyond the floor's is claimed as the group opens, before it is
    /// allocated: refused while the growth view is tight, admitted and
    /// allocated at once when room comes back.
    #[test]
    fn a_load_sequence_allocates_what_each_claim_admits_at_the_laptops_figures() {
        use crate::memory::ledger::{CommittedKind, DeviceLedgers, Ledgers, LoadShape, Need, StoreShape};
        use crate::memory::ClaimDemand;
        use std::cell::Cell;

        /// Host RAM as the laptop reads it: the free room moves with the test.
        struct Moving {
            catalog: Rc<Devices>,
            room: Rc<Cell<u64>>,
            reclaimable: u64,
        }
        impl StatusReading for Moving {
            fn status(&self, device: &Device) -> seismic::Status {
                let mut status = self.catalog.device_status(device);
                let host = status.sample.host_ram.as_mut().expect("the test host reads its memory");
                host.allocatable.value = self.room.get();
                host.limits.clear();
                status
            }

            fn available(&self, at: Instant) -> Result<crate::memory::ledger::Available, crate::memory::ledger::Blind> {
                Ok(crate::memory::ledger::Available::new(1 << 40, self.room.get(), None, at)
                    .expect("the room is within the total")
                    .with_reclaimable(self.reclaimable))
            }
        }

        let component = magnitude_state::ComponentDescriptor::new(
            magnitude_state::LayerRef::Target(0),
            magnitude_state::CodecSpec::dense(seismic::DType::F32, 1024, 1024),
            1,
        )
        .expect("a component");
        let domains = vec![magnitude_state::HistoryDomainLayout::Token { components: vec![component] }];
        let shape = std::sync::Arc::new(LoadShape { stores: vec![StoreShape { domains: domains.clone(), context: 1024, max_advance: 16 }] });
        let page_rows = domains[0].geometry(&domains, 1024, 16).expect("a geometry").page_rows as u64;
        let first_page = Need::StateRows { sequence: crate::RequestId(1), from: 0, to: page_rows };
        let second_page = Need::StateRows { sequence: crate::RequestId(1), from: page_rows, to: 2 * page_rows };
        let page = shape.state_bytes(page_rows);
        let (floor_bytes, weights, staging) = (page, 5 * page, page);

        let catalog = Rc::new(Devices::discover(&[]).unwrap());
        let device = Rc::new(crate::test_support::open_backend(&catalog, BackendName::Cpu).unwrap());
        // The growth view holds the floor and the weights' room less one page;
        // the file cache holds the rest.
        let room = Rc::new(Cell::new(floor_bytes + weights - page));
        let reading = Moving { catalog, room: room.clone(), reclaimable: 4 * weights };
        let reserves = MemoryReserves::new(crate::platform::StoreMotion::new(1, 1.0), crate::platform::ReleaseRate::new(1e9));
        let reader = DeviceReader::open(Rc::new(reading), reserves, device.clone()).unwrap();
        let ledgers = DeviceLedgers::open_in(&Ledgers::new(), &device, shape.clone());
        reader.read_ledgers_now(&ledgers);

        // The load claims what it needs to serve, under its policy, and
        // allocates each claim at once.
        let state_floor = ledgers
            .claim(&device, CommittedKind::StateFloor, ClaimDemand { device: floor_bytes, ..Default::default() })
            .expect("the load's minimum request state is admitted");
        let mut import = ledgers
            .claim(&device, CommittedKind::Load, ClaimDemand { device: weights, ..Default::default() })
            .expect("the weights, past the growth view, take the file cache");
        let _weights = seismic::Tensor::zeros(&device, seismic::Element::bool(), &[weights]).expect("Seismic allocates the import");
        import.allocated();

        // The first request draws its rows from the floor: nothing more held,
        // and the page it allocates is admitted.
        let ledger = ledgers.device();
        let mut state = ledger.claim(Need::StateRows { sequence: crate::RequestId(1), from: 0, to: 0 }).expect("an empty state claim");
        state.grow(first_page).expect("the first page draws from the floor");
        let _page = seismic::Tensor::zeros(&device, seismic::Element::bool(), &[page]).expect("Seismic allocates the first page");
        state.allocated(page);
        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.committed(), snapshot.allocated(), "the charge equals the allocations: the floor is held once");
        assert_eq!(snapshot.committed(), weights + page);
        assert!(state.grow(second_page).is_err(), "a page past the floor has no room in the tight growth view");

        // A group's staging beyond the floor's: refused while the view is
        // tight, with nothing held; admitted, then allocated with no reading
        // between, once the room is back.
        let unit = std::num::NonZeroU64::new(staging).unwrap();
        let (refused, units) = ledger.claim_units(CommittedKind::GroupClass, unit, 1);
        assert_eq!(units, 0, "the tight view holds no staging");
        drop(refused);
        // The allocations took the free room and the file cache covered the
        // rest: what is free now is what others gave back.
        room.set(staging);
        reader.read_ledgers_now(&ledgers);
        let (mut group, units) = ledger.claim_units(CommittedKind::GroupClass, unit, 1);
        assert_eq!(units, 1, "the room that came back holds the staging");
        let _staging = seismic::Tensor::zeros(&device, seismic::Element::bool(), &[staging]).expect("Seismic allocates the staging its claim admitted");
        group.allocated(staging);
        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.committed(), snapshot.allocated(), "the charge equals the allocations with the group's staging");
        drop(state_floor);
    }

    /// A load's search reads its room from the load's ledgers, the
    /// authority its allocations read: with 64 MiB of host RAM free and a
    /// 32 MiB planning reserve, the search's room is the 64 MiB, in host
    /// RAM the device's storage shares.
    #[test]
    fn a_search_reads_the_ledgers_room() {
        const MIB: u64 = 1 << 20;
        let catalog = Rc::new(Devices::discover(&[]).unwrap());
        let device = Rc::new(crate::test_support::open_backend(&catalog, BackendName::Cpu).unwrap());
        let reserves = MemoryReserves::new(crate::platform::StoreMotion::new(32 * MIB - 1, 10.0), crate::platform::ReleaseRate::new(1e9));
        let reader = DeviceReader::open(Rc::new(Room { catalog, room: 64 * MIB, reclaimable: 0 }), reserves, device.clone()).unwrap();
        let ledgers = crate::memory::ledger::DeviceLedgers::open_in(
            &crate::memory::ledger::Ledgers::new(),
            &device,
            std::sync::Arc::new(crate::memory::ledger::LoadShape { stores: Vec::new() }),
        );
        let room: &dyn crate::TuningRoom = &LoadRoom { reader: &reader, ledgers: &ledgers };
        assert_eq!(room.now(&device), crate::RoomNow { device_bytes: 64 * MIB, host_bytes: 64 * MIB, host_shared: true });
    }
}
