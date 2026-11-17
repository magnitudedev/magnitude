//! A controlled execution domain over the planning fixture's model: its
//! family (`TestFamily`) submits no device work but completes as a test says,
//! and may hold a fake elastic expert bank (`FakeBank`). Built for the
//! domain's tests and, under the `fixtures` feature, for the tests of the
//! service owner above it.

use super::*;
use crate::{
    AttestedPrograms, ComponentSelection, ExecutionPath, ExecutionPlanner, PlannedMethod,
    ResourceAllocator, ResourceCapacity, ResourceLimits, ResourcePlanner, TargetLaunchCore,
    completion::CompletionWake,
    platform,
    programs::{
        CompletedHeadWork, CompletedVisionWork, CompletedWork, ProgramSubmission, ReadySubmission,
    },
    test_support::available,
};
use magnitude_artifacts::PackageManifest;
use magnitude_family_contracts::FamilyId;
use magnitude_state::KvCodec;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

pub mod loaded;

/// The machine as a test's reader reads it: the device's status and host
/// RAM's sample as first read, moved only by what the fixture's own ledgers
/// allocated since (the process's one count of its memory) and by what
/// another program holds ([`Press`]): the supply host RAM and each grant
/// have left falls by what the fixture takes of their store and rises by
/// what it gives back. No other process's drift enters what a test decides,
/// nor another test's (each fixture opens its own device and ledgers, and
/// tests run side by side in one process, on one GPU), and a release the
/// fixture makes (a bank's shrink) is room the next reading sees.
struct StillHost {
    topology: std::sync::Arc<seismic::DeviceTopology>,
    first: seismic::Status,
    first_grant: Option<seismic::memory::Grant>,
    /// The fixture's ledgers, once opened ([`fixture_ledgers`]): until then
    /// the fixture has allocated nothing.
    ledgers: std::cell::OnceCell<crate::memory::ledger::DeviceLedgers>,
    pressed: Pressed,
    clock: TestClock,
    ticks: std::cell::Cell<u64>,
}

/// No lag: every change of the process's shows in the next reading.
pub(crate) const NO_LAG: seismic::Lag = seismic::Lag {
    release: std::time::Duration::ZERO,
    growth: std::time::Duration::ZERO,
};

/// What other programs hold of each store, as a test presses it, and whether
/// a hold change is owed a pressure reading.
#[derive(Clone, Default)]
pub(crate) struct Pressed {
    holds: Rc<std::cell::RefCell<BTreeMap<seismic::StoreId, u64>>>,
    event: Rc<std::cell::Cell<bool>>,
}

/// Another program's hold on the store of a fixture's device storage
/// ([`press`]), which the fixture's readings read as the machine's: the
/// room left there falls by what it takes, and rises back as it lets go.
pub struct Press {
    store: seismic::StoreId,
    pressed: Pressed,
}

impl Press {
    /// Hold `bytes` of the store from now, in place of what it held.
    pub fn take(&self, bytes: u64) {
        self.pressed.holds.borrow_mut().insert(self.store, bytes);
        self.pressed.event.set(true);
    }

    /// Hold nothing of the store from now.
    pub fn release(&self) {
        self.pressed.holds.borrow_mut().remove(&self.store);
        self.pressed.event.set(true);
    }
}

/// A hold on the store of `domain`'s device storage ([`Press`]).
pub fn press(domain: &ExecutorDomain<TestFamily>) -> Press {
    let store = domain.store_of(seismic::Contract::DeviceOnly).expect("the device's storage lies in a store");
    Press { store, pressed: domain.family.pressed.clone() }
}

/// The time a fixture's reader reads memory at (`StatusReading::now`): the
/// time since the fixture opened that the test's owner reads memory at
/// ([`Self::set`]), as the live machine's readings carry the monotonic
/// time the owner counts. It stands still until a test moves it.
#[derive(Clone)]
pub struct TestClock {
    opened: std::time::Instant,
    state: Arc<Mutex<(std::time::Duration, Option<std::time::Instant>)>>,
}

impl TestClock {
    fn new() -> Self {
        Self {
            opened: std::time::Instant::now(),
            state: Arc::new(Mutex::new((std::time::Duration::ZERO, None))),
        }
    }

    /// The owner's time now, `since_ns` since the fixture opened.
    pub fn set(&self, since_ns: u64) {
        self.state.lock().unwrap().0 = std::time::Duration::from_nanos(since_ns);
    }

    fn advance_sample(&self) -> std::time::Instant {
        let mut state = self.state.lock().unwrap();
        let clock_now = self.opened + state.0;
        let stamp = match state.1 {
            None => clock_now,
            Some(previous) => {
                let step = previous + std::time::Duration::from_millis(1);
                if clock_now > step { clock_now } else { step }
            }
        };
        state.1 = Some(stamp);
        stamp
    }

    fn now(&self) -> std::time::Instant {
        let state = self.state.lock().unwrap();
        let clock_now = self.opened + state.0;
        match state.1 {
            None => clock_now,
            Some(previous) => clock_now.max(previous),
        }
    }
}

/// The clock `domain`'s reader reads memory at ([`TestClock`]).
pub fn clock(domain: &ExecutorDomain<TestFamily>) -> TestClock {
    domain.family.clock.clone()
}

impl StillHost {
    fn of(catalog: seismic::Devices, device: &seismic::Device, pressed: Pressed, clock: TestClock) -> Self {
        let first_grant = device.reconcile_driver_memory();
        let mut first = catalog.device_status(device);
        // Normalize host RAM to a quiet, lag-free machine state so deterministic
        // fixture tests never inherit transient live-machine swap/pressure or
        // unmonitored-Linux withheld clean_cache splitting usable supply.
        let monitored = first.sample.host.harm.is_some() || first.sample.host.pressure.is_some();
        let usable = first.sample.host.supply.free_beyond_target.max(0).unsigned_abs()
            + if monitored { first.sample.host.supply.clean_cache } else { 0 };
        first.sample.host.supply = seismic::Supply {
            free_beyond_target: i64::try_from(usable).expect("host RAM's bytes fit i64"),
            clean_cache: 0,
            cold_reclaimable: 0,
        };
        first.sample.host.total = first.sample.host.total.max(usable);
        first.sample.host.lag = NO_LAG;
        first.sample.host.counted.lag = std::time::Duration::ZERO;
        if first.sample.host.harm.is_some() {
            first.sample.host.harm = Some(seismic::Harm { machine: 0, own: 0 });
        }
        if first.sample.host.own_reclaimed.is_some() {
            first.sample.host.own_reclaimed = Some(seismic::OwnReclaimed::NONE);
        }
        if first.sample.host.pressure.is_some() {
            first.sample.host.pressure = Some(seismic::PressureLevel::Normal);
        }
        Self {
            topology: catalog.topology(),
            first,
            first_grant,
            ledgers: std::cell::OnceCell::new(),
            pressed,
            clock,
            ticks: std::cell::Cell::new(0),
        }
    }

    /// Advance and record the monotonic instant of the next sample so
    /// consecutive samples always advance even when `TestClock` stands still,
    /// while respecting explicit `TestClock::set` jumps.
    fn next_stamp(&self) -> std::time::Instant {
        let tick = self.ticks.get();
        self.ticks.set(tick + 1);
        self.clock.advance_sample()
    }

    /// What the fixture moved of `store` since the first reading: what its
    /// ledger there allocated, and what another program holds of it now.
    fn taken(&self, store: seismic::StoreId) -> u64 {
        let identity = self.topology.store_identity(store).expect("a read store belongs to its topology");
        let own = self.ledgers.get().and_then(|ledgers| ledgers.keeping(&identity)).map_or(0, |ledger| ledger.estimate().allocated());
        own + self.pressed.holds.borrow().get(&store).copied().unwrap_or(0)
    }

    /// Of what the fixture moved of `store`, what the OS's wire count shows:
    /// what its ledger there wires, and what another program holds of it now.
    fn taken_wired(&self, store: seismic::StoreId) -> u64 {
        let identity = self.topology.store_identity(store).expect("a read store belongs to its topology");
        let own = self.ledgers.get().and_then(|ledgers| ledgers.keeping(&identity)).map_or(0, |ledger| ledger.wired());
        own + self.pressed.holds.borrow().get(&store).copied().unwrap_or(0)
    }

}

impl StatusReading for StillHost {
    fn status(&self, _: &seismic::Device) -> seismic::Status {
        let seismic::Status { mut sample, mut runtime } = self.first.clone();
        let stamp = self.clock.now();
        sample.host.at = stamp;
        sample.host.counted.at = stamp;
        let host_ram = self.topology.host_ram().id();
        // Host RAM as first read (free memory alone, [`Self::of`]), its
        // free memory less what the fixture took: below the OS's target
        // where it took more.
        let taken = i64::try_from(self.taken(host_ram)).expect("host RAM's bytes fit i64");
        sample.host.supply.free_beyond_target -= taken;
        if let Some(wired) = &mut sample.host.wired {
            wired.wire_count = wired.wire_count.saturating_add(self.taken_wired(host_ram));
        }
        for (path, grant) in sample.grants.iter_mut().chain(runtime.grants.iter_mut()) {
            if let Ok(grant) = grant {
                let store = self.topology.access_path(*path).expect("a read grant's path belongs to its topology").store();
                grant.value = grant.value.saturating_sub(self.taken(store));
            }
        }
        seismic::Status { sample, runtime }
    }

    fn reconcile(&self, device: &seismic::Device) -> Option<seismic::memory::Grant> {
        let _ = device.reconcile_driver_memory();
        self.first_grant.map(|grant| seismic::memory::Grant {
            left: grant.left.saturating_sub(self.taken(device.device_store())),
        })
    }

    fn now(&self) -> std::time::Instant {
        self.next_stamp()
    }

    fn take_pressure(&self) -> bool {
        self.pressed.event.replace(false)
    }
}

/// The planning fixture's model (the smallest geometry every native kernel
/// admits) with an 8-token context.
pub fn tiny_definition() -> ModelDefinition {
    let mut definition = crate::planning::fixtures::fixture_definition();
    definition.family = FamilyId("controlled-target-domain".into());
    definition.decoder.context_limit = 8;
    definition
}

pub fn tiny_manifest(definition: &ModelDefinition) -> PackageManifest {
    crate::planning::fixtures::fixture_manifest(definition)
}

pub type Fixture = (ExecutorDomain<TestFamily>, StateBindings);

thread_local! {
    /// The stored-table reads the next target submission's output carries:
    /// a test sets them to inject a step's table reads.
    pub static STORED_READS: std::cell::RefCell<Vec<crate::programs::StoredRead>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// The expert bank step the next target submission's output carries: a
    /// test sets it to inject a step's expert reads.
    pub static EXPERT_STEPS: std::cell::RefCell<Option<crate::experts::ExpertStep>> =
        const { std::cell::RefCell::new(None) };
    /// `Some(0)` rejects the next target submission, which then records
    /// how many launches its rejected chain returned.
    pub static REJECTED: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

pub fn fixture(control: Option<PendingControl>) -> Fixture {
    fixture_with(control, false, false)
}

/// A spare head store of 16-wide F32 rows of the head's block
/// (`LayerRef::Head(0)`, the history a head launch appends): 64 rows per
/// 64 KiB slab, 8192 rows, one slab backed, its slabs granules of `pool`. It
/// grows independently of the target model.
fn spare_head_store(charges: &seismic::Charges, pool: seismic::Pool) -> StoreBindings {
    use magnitude_state::{
        BankCapacity, CodecSpec, ComponentDescriptor, HistoryDomainLayout, HistoryDomainPlan,
        LayerRef,
    };
    StateStore::new(
        charges,
        pool,
        8192,
        8192,
        vec![HistoryDomainPlan {
            layout: HistoryDomainLayout::Token {
                components: vec![
                    ComponentDescriptor::new(
                        LayerRef::Head(0),
                        CodecSpec::dense(seismic::DType::F32, 16, 16),
                        1,
                    )
                    .unwrap(),
                ],
            },
            logical_rows: 8192,
        }],
        vec![],
        BankCapacity {
            active: 4,
            in_flight: 4,
            retained: 0,
        },
    )
    .unwrap()
    .1
}

pub fn fixture_with(control: Option<PendingControl>, lookahead: bool, head: bool) -> Fixture {
    fixture_on(control, lookahead, head, platform::DeviceRequest::Automatic)
}

pub fn fixture_on(
    control: Option<PendingControl>,
    lookahead: bool,
    head: bool,
    device_request: platform::DeviceRequest,
) -> Fixture {
    fixture_of(tiny_definition(), limits(2, 1, lookahead), control, head, device_request)
}

/// Launches of up to `rows` rows, in chains of up to `chain_capacity`
/// launches, `lookahead` or not, exporting no logits.
pub fn limits(rows: usize, chain_capacity: usize, lookahead: bool) -> ResourceLimits {
    ResourceLimits {
        max_launch_rows: rows,
        max_launch_slots: 2,
        max_selected_rows: 2,
        max_drafting_slots: 2,
        exported_logits_rows: 0,
        max_images_per_request: magnitude_artifacts::MAX_IMAGES_PER_REQUEST,
        max_image_cells: 0,
        lookahead,
        chain_capacity: std::num::NonZeroUsize::new(chain_capacity).unwrap(),
        conversations: std::num::NonZeroUsize::MIN,
    }
}

/// A domain over `definition` within `limits`, on the device `device_request`
/// selects: one every host has (the CPU, or the device `Automatic`
/// prefers), so discovery, selection and opening each succeed or the test
/// fails.
pub fn fixture_of(
    definition: ModelDefinition,
    limits: ResourceLimits,
    control: Option<PendingControl>,
    head: bool,
    device_request: platform::DeviceRequest,
) -> Fixture {
    let manifest = tiny_manifest(&definition);
    fixture_over(definition, manifest, limits, PlannedMethod::Plain, control, head, device_request)
}

/// [`fixture_of`] over `manifest`'s stored weights, planned for `method`. A
/// method that drafts plans `definition`'s head, and its head store is the
/// plan's (none for a head reading the target's history); `head` adds a
/// spare head store to a load that drafts nothing.
pub fn fixture_over(
    definition: ModelDefinition,
    manifest: PackageManifest,
    limits: ResourceLimits,
    method: PlannedMethod,
    control: Option<PendingControl>,
    head: bool,
    device_request: platform::DeviceRequest,
) -> Fixture {
    assert!(!(head && method.drafts()), "a drafted fixture's head store is its plan's");
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let release = platform::ReleaseRate::of_machine(None);
    let selected = platform::select_device(&catalog, ExecutionPath::Native, device_request, &release)
        .unwrap_or_else(|error| panic!("{device_request} selects a device: {error}"));
    let device = Rc::new(
        crate::test_support::open(&catalog, catalog.resolve(selected.info.selector()).unwrap())
            .unwrap_or_else(|error| panic!("the selected device opens: {error}")),
    );
    let definition = Rc::new(definition);
    let capacity_bytes = ResourceCapacity {
        domain_bytes: crate::StableRoom::of_selected(&selected).capacity().unwrap().min(512 * 1024 * 1024),
        tensor_operations: crate::TensorOperations::of(selected.info.forms_tensor_operations()),
        allocations: seismic::NativeAllocations::EXACT,
    };
    let draft = ExecutionPlanner::prepare(
        &selected,
        &manifest,
        &definition,
        ComponentSelection {
            head: method.drafts(),
            vision: false,
        },
        ExecutionPath::Native,
        method,
        KvCodec::Dense,
        limits,
    )
    .unwrap();
    let state = ResourcePlanner::state_plan(
        &definition,
        draft.load(),
        draft.policy().method(),
        KvCodec::Dense,
        limits,
        capacity_bytes,
    )
    .unwrap();
    let mut programs = AttestedPrograms::prepare_draft(
        &draft,
        &seismic::Charges::tally(&device),
        crate::TuningContext {
            definition: &definition,
            weights: &crate::ZeroTuningWeights,
            observer: &crate::UnreportedTuning,
            cache: None,
            numerical_profile: crate::ResolvedNumericalProfile::standard(),
            settings: crate::TuningSettings::STANDARD,
        },
        crate::UnpreparedUnits::ServeDefaults,
    )
    .unwrap();
    let target_graphs = programs
        .prepare_target_graphs(
            &device,
            draft.load(),
            &definition.decoder,
            &state,
            draft.programs().target(),
            limits,
            draft.load().pass(limits),
            &seismic::Workers::new(1),
        )
        .unwrap();
    let target_readout_graphs = programs
        .prepare_target_readout_graphs(&device, draft.load(), &definition.decoder, limits)
        .unwrap();
    programs
        .prepare_auxiliary_graphs(
            &device,
            draft.load(),
            &definition,
            draft.programs(),
            &state,
            limits,
            method,
        )
        .unwrap();
    let resources_plan = ResourcePlanner::plan_with_state(
        state,
        &target_graphs,
        &target_readout_graphs,
        programs.drafter_graphs(),
        None,
        programs.state_graphs().unwrap(),
    )
    .unwrap();
    let execution = Rc::new(draft.admit(resources_plan).unwrap());
    let id = ResourceDomainId::new("controlled-target-domain").unwrap();
    let (pressed, clock) = (Pressed::default(), TestClock::new());
    let reading = Rc::new(StillHost::of(catalog, &device, pressed.clone(), clock.clone()));
    let ledgers = fixture_ledgers_at(&device, &execution, &*reading, &clock);
    assert!(reading.ledgers.set(ledgers.clone()).is_ok(), "a fixture opens its ledgers once");
    // What the load allocates as it starts, each allocation charged to its
    // claim as the load's are.
    let mut recorded = crate::memory::ledger::Recorded::default();
    // The state pool as a load opens it: at the plan's state granule.
    let pool = ledgers
        .open_pool(&device, seismic::Contract::DeviceOnly, crate::memory::ledger::PoolClass::State, execution.resources().target_state().granule)
        .expect("the fixture's device opens its state pool");
    let scratch = crate::memory::ClaimDemand { device: execution.resources().bytes().scratch, ..Default::default() };
    let arena = crate::memory::ledger::CommittedKind::Arena;
    let resources = fixture_charged(&device, &ledgers, &mut recorded, arena, scratch, |charges| {
        ResourceAllocator::allocate(
            &execution,
            charges,
            id,
            &target_graphs,
            &target_readout_graphs,
            programs.drafter_graphs(),
            None,
            programs.state_graphs().unwrap(),
            &ledgers,
        )
    })
    .unwrap();
    let target_state = execution.resources().target_state();
    // A drafted head's tables beside the target's; a spare head store has
    // no plan, and its initial slab and tables are charged as they are made.
    let head_plan = execution.resources().head_state().filter(|_| method.drafts());
    let spare_slab = if head && head_plan.is_none() { target_state.granule } else { 0 };
    // As a load makes its state: one request's opening as its slot, its
    // first slabs granules within it, and its address tables charged to a
    // claim of their own.
    let request_slot = request_slot(&ledgers, &device, &execution, spare_slab);
    let head_tables = head_plan.map_or(0, |state| state.table_bytes().unwrap());
    let tables = crate::memory::ClaimDemand { device: target_state.table_bytes().unwrap() + head_tables, ..Default::default() };
    let kind = crate::memory::ledger::CommittedKind::StateTables;
    let (store, head) = fixture_charged(&device, &ledgers, &mut recorded, kind, tables, |charges| {
        let store = target_state.allocate(charges, pool.clone());
        let head = match head_plan {
            Some(state) => Some(state.allocate(charges, pool.clone()).unwrap()),
            None => head.then(|| spare_head_store(charges, pool.clone())),
        };
        (store, head)
    });
    let store = store.unwrap();
    let (domain, bindings) = ExecutorDomain::with_family(
        execution,
        definition,
        resources,
        DeviceReader::open(reading.clone(), release, device).unwrap(),
        store,
        head,
        None,
        None,
        TestFamily {
            control,
            programs,
            bank: None,
            waiting: usize::MAX,
            optional: None,
            pressed,
            clock,
            reading,
        },
        recorded,
        request_slot,
    )
    .expect("the fixture holds its arena");
    settle_ledgers(domain.domain.device(), domain.resources.ledgers(), &*domain.family.reading);
    (domain, bindings)
}

/// The request slot a load of `execution` claims on `ledgers` as it starts
/// (`CommittedKind::RequestSlot`): the load's ([`super::request_slot_bytes`]),
/// and `spare` beside it, a spare head store's first slab, which no plan
/// counts and which lies within it.
pub fn request_slot(ledgers: &crate::memory::ledger::DeviceLedgers, device: &seismic::Device, execution: &ExecutionPlan, spare: u64) -> crate::memory::ledger::Claim {
    let slot = crate::memory::ClaimDemand { device: super::request_slot_bytes(execution).unwrap() + spare, ..Default::default() };
    ledgers.claim_request_slot(device, slot).expect("the fixture's ledgers hold its request slot")
}

/// The bytes a request of `domain`'s load holds once reserved through
/// `rows` rows: their slabs in both its stores
/// (`memory::ledger::LoadShape::state_bytes`).
pub fn reserved_bytes<F: ProgramFamily>(domain: &ExecutorDomain<F>, rows: u64) -> u64 {
    domain.resources.ledgers().device().bytes(&crate::memory::ledger::Need::Request { sequence: RequestId(0), rows })
}

/// The rows a request of `domain`'s load reserves as it opens about to
/// write `rows` rows (`memory::ledger::RequestRows::opening`).
pub fn opening_rows<F: ProgramFamily>(domain: &ExecutorDomain<F>, rows: u64) -> u64 {
    crate::memory::ledger::RequestRows::of(&domain.execution).opening(rows)
}

/// The rows `request`'s reservation holds, none where it holds none.
pub fn reservation_rows<F: ProgramFamily>(domain: &ExecutorDomain<F>, request: RequestId) -> Option<u64> {
    domain.state_claims.get(&request).map(crate::memory::ledger::Reservation::rows)
}

/// `make`, an allocation a load makes as it starts, under its claim of
/// `demand` on `ledgers` as `kind` (the load's own, or the arena's): every
/// allocation it makes through the charges it is given charged to the
/// claim within one change of the ledgers' scope, and the claim recorded in
/// `recorded`.
pub(crate) fn fixture_charged<T>(
    device: &seismic::Device,
    ledgers: &crate::memory::ledger::DeviceLedgers,
    recorded: &mut crate::memory::ledger::Recorded,
    kind: crate::memory::ledger::CommittedKind,
    demand: crate::memory::ClaimDemand,
    make: impl FnOnce(&seismic::Charges) -> T,
) -> T {
    let mut held = ledgers
        .claim(device, kind, demand)
        .expect("the fixture's ledgers hold what it allocates");
    let made = held.allocate(device, make);
    recorded.record(held);
    made
}

/// The bytes of a fake bank's slot: a granule of its records pool.
const FAKE_BANK_GRANULE: u64 = 64 << 20;

/// The records pool of the device store a fixture's fake bank lies in, a
/// granule a slot and a block, so each freed slot returns its block. Its
/// blocks are charged to that pool's entry, returned beside it.
fn fixture_pool(
    device: &seismic::Device,
    ledgers: &crate::memory::ledger::DeviceLedgers,
    class: crate::memory::ledger::PoolClass,
) -> (seismic::Pool, Arc<dyn seismic::memory::Entry>) {
    let granule = FAKE_BANK_GRANULE;
    let entries = ledgers.device().pool_block(class, std::num::NonZeroU64::new(granule).expect("a granule"));
    let entry = entries.blocks.clone();
    let pool = seismic::Pool::open(&entries.charges(device, seismic::Contract::DeviceOnly), seismic::Contract::DeviceOnly, granule, granule)
        .unwrap_or_else(|error| panic!("the fixture's device opens its store's {class:?} pool: {error}"));
    (pool, entry)
}


/// The ledgers of the stores `device` reaches, in a scope of the test's own
/// (no other test's claims in them), first read by `reading`, the one every
/// later reading of the domain's is taken by: a reading after it moves
/// others' memory only as the test moves it.
fn settle_ledgers(device: &seismic::Device, ledgers: &crate::memory::ledger::DeviceLedgers, reading: &dyn StatusReading) {
    for _ in 0..2 {
        let sample = reading.sample(device, ledgers.changes()).expect("a test's sample stands beside its own scope's changes");
        ledgers.read(device, &sample);
    }
}

fn fixture_ledgers_in(
    scope: &crate::memory::ledger::Ledgers,
    device: &seismic::Device,
    execution: &crate::ExecutionPlan,
    reading: &dyn StatusReading,
) -> crate::memory::ledger::DeviceLedgers {
    use crate::memory::ledger::{DeviceLedgers, LoadShape};
    let ledgers = DeviceLedgers::open_in(scope, device, Arc::new(LoadShape::of(execution)));
    let first = reading.sample(device, ledgers.changes()).expect("a test's sample stands beside its own scope's changes");
    let ledgers = ledgers.read(device, &first).0;
    let second = reading.sample(device, ledgers.changes()).expect("a test's second sample stands beside its own scope's changes");
    ledgers.read(device, &second);
    ledgers
}

fn fixture_ledgers_at(
    device: &seismic::Device,
    execution: &crate::ExecutionPlan,
    reading: &dyn StatusReading,
    clock: &TestClock,
) -> crate::memory::ledger::DeviceLedgers {
    let clock = clock.clone();
    let scope = crate::memory::ledger::Ledgers::scripted(move || clock.now());
    fixture_ledgers_in(&scope, device, execution, reading)
}

pub(crate) fn fixture_ledgers(device: &seismic::Device, execution: &crate::ExecutionPlan, reading: &dyn StatusReading) -> crate::memory::ledger::DeviceLedgers {
    fixture_ledgers_in(&crate::memory::ledger::Ledgers::new(), device, execution, reading)
}

/// The backends a bank fixture is built on ([`banked`]): each discovery
/// reports available ([`available`]). Every backend's device opens its
/// store's granule pool, which the fake bank's slots are granules of, so no
/// available backend is left out: one whose pool refuses the bank fails the
/// test.
pub fn bank_backends() -> Vec<seismic::BackendName> {
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    seismic::BackendName::ALL.into_iter().filter(|&backend| available(&catalog, backend)).collect()
}

/// A fixture on `backend`'s device whose family holds a fake elastic bank
/// of `rows` rows, each a granule of the device store's pool ([`bank_row`]:
/// a row is then backed, charged and freed whole as a block of its own),
/// `committed` of them backed, which the pool's entry is charged.
pub fn banked(backend: seismic::BackendName, rows: u64, committed: u64) -> Fixture {
    banked_with(backend, rows, committed, false)
}

/// [`banked`], its launches pipelined (`lookahead`) or not.
pub fn banked_with(backend: seismic::BackendName, rows: u64, committed: u64, lookahead: bool) -> Fixture {
    with_bank(fixture_on(None, lookahead, false, platform::DeviceRequest::Backend(backend)), backend, rows, committed)
}

/// [`banked`] on a load whose prompt groups chain up to three launches, so
/// a prompt's group can grow its places.
pub fn banked_chain(backend: seismic::BackendName, rows: u64, committed: u64) -> Fixture {
    let fixture = fixture_of(tiny_definition(), limits(2, 3, false), None, false, platform::DeviceRequest::Backend(backend));
    with_bank(fixture, backend, rows, committed)
}

/// `fixture`, its family holding a fake bank of `rows` rows, `committed`
/// of them backed ([`banked`]).
fn with_bank((mut domain, bindings): Fixture, backend: seismic::BackendName, rows: u64, committed: u64) -> Fixture {
    // The bank's slots lie in its store's records pool, as a load's records do.
    let (pool, entry) = fixture_pool(domain.domain.device(), domain.resources.ledgers(), crate::memory::ledger::PoolClass::Records);
    let region = [seismic::SlabRegion { element: seismic::Element::bool(), row_shape: vec![pool.granule()] }];
    let slots = seismic::SlabTensor::in_pool(&seismic::Charges::tally(domain.domain.device()), &pool, 1, rows, &region)
        .unwrap_or_else(|error| panic!("{backend:?}: the device's pool holds the bank: {error}"));
    let bank = FakeBank::new(slots, domain.resources.ledgers().changes().clone());
    bank.hold_cache(domain.resources.ledgers(), entry, committed);
    domain.family.bank = Some(bank);
    settle_ledgers(domain.domain.device(), domain.resources.ledgers(), &*domain.family.reading);
    (domain, bindings)
}

/// A decode of `steps` steps at the fake bank's one streamed layer of 4
/// experts, each step reading all 4 of them, all `depth` deep in the
/// bank's stack: a bank of `depth` slots misses them, one of `depth + 1`
/// none, beside the generic model counted as one step. Reads take 0.4 ms and 0.5 ms a MB ([`READ`]); an
/// all-hit step 15 ms.
pub fn misses_at(depth: usize, steps: u64) -> crate::assessment::LiveDecode {
    misses_by(steps, &[(depth, 4)])
}

/// [`misses_at`] with each step's reads spread over the stack: `reads` of
/// each `(depth, reads)`, which a bank of `depth` slots misses and one of
/// `depth + 1` hits. The misses stop falling past the deepest.
pub fn misses_by(steps: u64, reads: &[(usize, u64)]) -> crate::assessment::LiveDecode {
    let mut misses = [0; 5];
    let mut requests = [0; 5];
    for &(depth, count) in reads {
        misses[depth] = steps * count;
        requests[depth] = steps;
    }
    crate::assessment::LiveDecode {
        layers: vec![crate::assessment::LayerReads::of_depths(&misses, &requests)],
        steps,
        rows: 1.,
        depth: 100,
    }
}

/// The fake decode's read line: 0.4 ms and 0.5 ms a MB.
pub const READ: crate::assessment::ReadLine = match crate::assessment::ReadLine::new(4e-4, 1. / 2e9) {
    Ok(line) => line,
    Err(_) => panic!("the fixtures' read line"),
};

/// Shrink `domain`'s state surplus ([`ExecutorDomain::shrink_state`]) and
/// wait for the idle graph slots' release it began, if any, to end on the
/// release thread, then finish it as a boundary does
/// (`ExecutorDomain::advance_slot_release`): the bytes the state's pool
/// returned to the OS and the slots' release released.
pub fn shrink_settled<F: ProgramFamily>(domain: &mut ExecutorDomain<F>, bindings: &mut StateBindings) -> u64 {
    let shrunk = domain.shrink_state(bindings, crate::CompletionWake::new(|| {})).unwrap();
    let slots = match shrunk.slots {
        super::state::SlotRelease::Began => loop {
            if let Some(released) = domain.advance_slot_release().unwrap() {
                break released.bytes;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        },
        super::state::SlotRelease::None | super::state::SlotRelease::Running => 0,
    };
    shrunk.state.returned + slots
}

/// A claim of every byte of cache `domain`'s device ledger yields, as a
/// measurement claiming all the room there is takes it, held until it
/// drops.
pub fn take_all_cache<F: ProgramFamily>(domain: &ExecutorDomain<F>) -> crate::memory::ledger::Claim {
    domain
        .resources
        .ledgers()
        .device()
        .clone()
        .claim_units(crate::memory::ledger::CommittedKind::Scratch, std::num::NonZeroU64::MIN, u64::MAX)
        .0
}

/// The growth rule's inputs: the decode the fake bank counted, the
/// fixtures' read line, and the resident step it is priced on: 15 ms at every depth, its all-hit
/// steps' time.
pub fn price_growth(domain: &mut ExecutorDomain<TestFamily>, bindings: &mut StateBindings, decode: crate::assessment::LiveDecode) {
    count_decode(domain, decode);
    measure_load(domain, bindings);
}

/// The load measured its costs: a step of 15 ms in either form at every
/// depth, a banked load's all-hit steps' time; a banked load's experts on
/// the fixtures' read line, its decode priced on whatever the fake bank
/// counted, the generic model alone before any.
pub fn measure_load(domain: &mut ExecutorDomain<TestFamily>, bindings: &mut StateBindings) {
    let experts = domain.family.bank.is_some().then(|| crate::ExpertMeasures {
        lines: crate::assessment::Priced::Measured(std::sync::Arc::from([])),
        read: crate::assessment::Priced::Measured(READ),
        pass: crate::assessment::Priced::Pending("unpriced".into()),
        tier_copy: crate::assessment::Priced::Measured(None),
        convert: crate::assessment::Priced::Measured(None),
    });
    domain.measure_load(crate::LoadMeasures { costs: Ok(load_costs()), experts }, bindings).unwrap();
}

/// A fixture's costs: a step of 15 ms in either form at every depth.
pub fn load_costs() -> crate::LoadCosts {
    crate::LoadCosts { step: Box::new(|_, _| Ok(crate::assessment::FormSeconds { resident: 0.015, streamed: 0.015 })) }
}

/// A domain launching one-row prompt chunks in chains of up to three on
/// the device `device_request` selects: a load whose experts are held
/// whole, its prompt pass a whole chain's three places (`ModelLoadPlan::pass`).
pub fn chained(device_request: platform::DeviceRequest) -> Fixture {
    fixture_of(tiny_definition(), limits(2, 3, false), None, false, device_request)
}

/// The places a prompt group of four one-row chunks opened on `fixture`
/// holds, its first chunk submitted with the rest planned behind it.
pub fn group_places_opened<F: ProgramFamily>((mut domain, mut bindings): (ExecutorDomain<F>, StateBindings)) -> usize {
    let operations = group_opening(&mut domain, &mut bindings);
    let ReservedResources::Target(reservation) = domain.reserve(&mut bindings, &operations).unwrap().into_resources() else {
        panic!("a prompt chunk reserves a target group")
    };
    let flight = domain
        .submit_target(bindings, &operations, reservation)
        .unwrap_or_else(|failure| panic!("target submission failed: {}", failure.error()));
    let places = domain.group_places().get();
    domain.finish_target(flight).unwrap();
    places
}

/// The first chunk of a prompt group of four one-row chunks on `domain`,
/// its request open and the rest planned behind it: what
/// [`group_places_opened`] submits.
pub fn group_opening<F: ProgramFamily>(domain: &mut ExecutorDomain<F>, bindings: &mut StateBindings) -> Vec<Operation> {
    let request = RequestId(80);
    let chunk = |position: usize| Operation::Forward {
        request,
        kind: WorkKind::Prefill,
        tokens: vec![crate::TokenId(1)],
        position,
        conditioning: None,
        demand: crate::batching::Demand::NONE,
        select: Vec::new(),
        committed: 1,
        prime: None,
    };
    domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
    assert!(domain.open_served(bindings, request, None).unwrap().is_empty());
    domain.plan_successors((1..4).map(chunk).collect());
    vec![chunk(0)]
}

/// The bytes of a row of `domain`'s fake bank ([`banked`]).
pub fn bank_row(domain: &ExecutorDomain<TestFamily>) -> u64 {
    domain.family.bank.as_ref().expect("a banked fixture").row()
}

/// The fake bank of `domain` counted `decode`, which its growth is priced
/// on.
pub fn count_decode(domain: &mut ExecutorDomain<TestFamily>, decode: crate::assessment::LiveDecode) {
    domain.family.bank.as_mut().expect("a banked fixture").decode = Some(decode);
}

/// The fake bank of `domain` holds a host tier of at most `ceiling` host
/// slots, as a discrete device's does.
pub fn host_tier(domain: &mut ExecutorDomain<TestFamily>, ceiling: u64) {
    domain.family.bank.as_mut().expect("a banked fixture").host_ceiling = ceiling;
}

/// The planning fixture's streamed model: a domain over it plans an expert
/// bank, which a load's pricing prices.
pub use crate::planning::fixtures::{qwen4exp_fixture_definition, streamed_experts_manifest};

#[derive(Default)]
struct PendingState {
    result: Option<Result<(), crate::DeviceError>>,
    wake: Option<CompletionWake>,
}
#[derive(Clone, Default)]
pub struct PendingControl(Arc<Mutex<PendingState>>);
impl PendingControl {
    pub fn resolve(&self, result: Result<(), crate::DeviceError>) {
        let wake = {
            let mut state = self.0.lock().unwrap();
            assert!(state.result.replace(result).is_none());
            state.wake.take()
        };
        if let Some(wake) = wake {
            wake.complete();
        }
    }
}
pub struct PendingCompletion(pub(crate) PendingControl);
impl Completion for PendingCompletion {
    fn is_complete(&self) -> bool {
        self.0.0.lock().unwrap().result.is_some()
    }
    fn result(&mut self) -> Result<(), crate::DeviceError> {
        self.0
            .0
            .lock()
            .unwrap()
            .result
            .clone()
            .expect("completion is pending")
    }
    fn notify(&mut self, wake: CompletionWake) {
        let wake = {
            let mut state = self.0.0.lock().unwrap();
            if state.result.is_some() {
                Some(wake)
            } else {
                assert!(state.wake.replace(wake).is_none());
                None
            }
        };
        if let Some(wake) = wake {
            wake.complete();
        }
    }
}
pub struct PendingSubmission<L, W, O> {
    pub(crate) completion: PendingCompletion,
    pub(crate) core: L,
    pub(crate) workspace: W,
    pub(crate) output: O,
}

/// A controlled pending submission for lane-level device-failure tests. It has
/// no completed payload because the error resolves before `finish` exposes work.
pub struct PendingFailureSubmission<W> {
    completion: PendingCompletion,
    _completed: PhantomData<fn() -> W>,
}

impl<W> PendingFailureSubmission<W> {
    pub fn new(control: PendingControl) -> Self {
        Self {
            completion: PendingCompletion(control),
            _completed: PhantomData,
        }
    }
}

impl<W> ProgramSubmission for PendingFailureSubmission<W> {
    type CompletedWork = W;

    fn completion(&mut self) -> &mut dyn Completion {
        &mut self.completion
    }

    fn finish(mut self) -> Result<Self::CompletedWork, crate::DeviceError> {
        match self.completion.result() {
            Err(error) => Err(error),
            Ok(()) => panic!("failure-only controlled submission resolved successfully"),
        }
    }
}
impl<L, W, O> ProgramSubmission for PendingSubmission<L, W, O> {
    type CompletedWork = CompletedWork<L, O>;
    fn completion(&mut self) -> &mut dyn Completion {
        &mut self.completion
    }
    fn finish(mut self) -> Result<Self::CompletedWork, crate::DeviceError> {
        self.completion.result()?;
        drop(self.workspace);
        Ok(CompletedWork::new(self.core, self.output))
    }
}

type HeadTestWork = PendingSubmission<
    crate::HeadLaunchCore,
    (NativeGraphWorkspaceLease, Option<NativeGraphOutputLease>),
    crate::programs::HeadOutput,
>;

pub enum ControlledHeadSubmission {
    Failure(PendingFailureSubmission<CompletedHeadWork>),
    Work(HeadTestWork),
}

impl From<PendingFailureSubmission<CompletedHeadWork>> for ControlledHeadSubmission {
    fn from(value: PendingFailureSubmission<CompletedHeadWork>) -> Self {
        Self::Failure(value)
    }
}

impl ProgramSubmission for ControlledHeadSubmission {
    type CompletedWork = CompletedHeadWork;

    fn completion(&mut self) -> &mut dyn Completion {
        match self {
            Self::Failure(work) => work.completion(),
            Self::Work(work) => work.completion(),
        }
    }

    fn finish(self) -> Result<Self::CompletedWork, crate::DeviceError> {
        match self {
            Self::Failure(work) => work.finish(),
            Self::Work(work) => work.finish(),
        }
    }
}

impl crate::programs::SubmittedHead for ControlledHeadSubmission {
    fn launch(&self) -> &crate::HeadLaunchCore {
        match self {
            Self::Failure(_) => panic!("failure-only head has no launch"),
            Self::Work(work) => &work.core,
        }
    }
}

pub enum TestSubmission<L, W, O> {
    Ready(ReadySubmission<L, W, O>),
    Pending(PendingSubmission<L, W, O>),
}
impl<W> crate::programs::SubmittedTarget
    for TestSubmission<TargetLaunchCore, W, crate::TargetOutput>
{
    fn launch(&self) -> &TargetLaunchCore {
        match self {
            Self::Ready(value) => value.launch(),
            Self::Pending(value) => &value.core,
        }
    }
    fn output(&self) -> &crate::TargetOutput {
        match self {
            Self::Ready(value) => value.output(),
            Self::Pending(value) => &value.output,
        }
    }
}
impl<L, W, O> ProgramSubmission for TestSubmission<L, W, O> {
    type CompletedWork = CompletedWork<L, O>;
    fn completion(&mut self) -> &mut dyn Completion {
        match self {
            Self::Ready(value) => value.completion(),
            Self::Pending(value) => value.completion(),
        }
    }
    fn finish(self) -> Result<Self::CompletedWork, crate::DeviceError> {
        match self {
            Self::Ready(value) => value.finish(),
            Self::Pending(value) => value.finish(),
        }
    }
}
pub struct TestFamily {
    pub(crate) control: Option<PendingControl>,
    pub(crate) programs: AttestedPrograms,
    /// A fake elastic expert bank: a granule of the pool a slot.
    pub(crate) bank: Option<FakeBank>,
    /// The size rule's chunks while decode waits, within the capacity.
    pub(crate) waiting: usize,
    /// An optional component's constants, a tensor of its own: what the
    /// optional holding classifies.
    pub(crate) optional: Option<seismic::Tensor>,
    /// What other programs hold of each store its reader reads ([`press`]).
    pressed: Pressed,
    /// The clock its reader reads memory at ([`clock`]).
    clock: TestClock,
    /// The reader its domain samples memory through.
    reading: Rc<dyn StatusReading>,
}

/// A fake elastic expert bank (`ElasticBank`): a slab store of a slot a
/// slab, each a granule of the device store's pool, its floor one slot. A
/// resize prepares at once and finishes over two boundaries, as the bank's
/// does: the first installs (a growth backs its slots, a shrink frees
/// them), the second takes the release (a shrink's blocks returned). Its
/// growth is priced on the decode it counted, `decode`. As the bank's
/// watcher, it is asked to shrink only in a gap: a shrink begins only while
/// none of its `steps` runs.
pub struct FakeBank {
    slots: seismic::SlabTensor,
    resize: std::sync::Mutex<FakeResize>,
    pub(crate) decode: Option<crate::assessment::LiveDecode>,
    steps: std::sync::atomic::AtomicUsize,
    /// A host tier's slots and the most it holds (none without a tier),
    /// each a byte of upload storage, and its growth's outcome until an
    /// advance reports it.
    host: std::sync::atomic::AtomicU64,
    pub(crate) host_ceiling: u64,
    host_grown: std::sync::Mutex<Option<BankResized>>,
    /// Its slots' and its host tier's last finished resizes
    /// ([`ElasticBank::record_resize`]).
    last_resize: std::sync::Mutex<Option<crate::BankResize>>,
    host_last_resize: std::sync::Mutex<Option<crate::BankResize>>,
    /// Its charge's rises: none off the domain's thread, so held by none
    /// but the domain.
    rises: std::sync::Mutex<()>,
    /// Whether its device refuses a tier copy's grid, a test's: it takes
    /// one unless set.
    pub(crate) refuse_tier_copy: bool,
    /// The tier copy costs it last took: its grid's blocks and its copies'
    /// price.
    pub(crate) tier_copy: std::sync::Mutex<Option<crate::assessment::TierCopyCosts>>,
    /// Its cover's starts and ends ([`ElasticBank::cover`]): one, read as
    /// it opens, until a test reads it ([`Self::cover_reads`]).
    covers: std::sync::atomic::AtomicU64,
    /// What its slots above the floor hold on the domain's ledger, as the
    /// served bank's cache does.
    cache: crate::experts::cache::BankCache,
    /// The pool entries of the stores its host tier's slots lie in, which
    /// it charges their bytes as the backend charges a pool's blocks
    /// ([`Self::hold_cache`]).
    pools: std::sync::OnceLock<Vec<std::sync::Arc<dyn seismic::memory::Entry>>>,
}

enum FakeResize {
    Idle,
    /// To `to` slots, with the charge and the slots before it.
    Prepared { to: u64, before: (u64, u64) },
    /// Installed, its release not yet taken.
    Releasing((u64, u64)),
    /// Released, as the bank's resize thread releases, before an advance
    /// reports it.
    Released((u64, u64)),
    /// A growth whose pool's entry refused a block, before an advance
    /// reports it.
    Refused(seismic::TensorError),
    /// A growth stopped before its first slot, at these slots, before an
    /// advance reports it.
    Stopped(u64),
}

impl FakeBank {
    pub fn new(slots: seismic::SlabTensor, changes: std::sync::Arc<crate::memory::ledger::Changes>) -> Self {
        Self {
            slots,
            resize: std::sync::Mutex::new(FakeResize::Idle),
            decode: None,
            steps: std::sync::atomic::AtomicUsize::new(0),
            host: std::sync::atomic::AtomicU64::new(0),
            host_ceiling: 0,
            host_grown: std::sync::Mutex::new(None),
            last_resize: std::sync::Mutex::new(None),
            host_last_resize: std::sync::Mutex::new(None),
            rises: std::sync::Mutex::new(()),
            refuse_tier_copy: false,
            tier_copy: std::sync::Mutex::new(None),
            covers: std::sync::atomic::AtomicU64::new(1),
            cache: crate::experts::cache::BankCache::new(changes),
            pools: std::sync::OnceLock::new(),
        }
    }

    /// Back `committed` slots, their blocks charged to `device_pool`, the
    /// device store's records pool entry, and hold the slots above the floor as
    /// cache on `ledgers`, as the served bank does at build: its bytes are
    /// the process's, never others'.
    pub(crate) fn hold_cache(
        &self,
        ledgers: &crate::memory::ledger::DeviceLedgers,
        device_pool: std::sync::Arc<dyn seismic::memory::Entry>,
        committed: u64,
    ) {
        let ledger = ledgers.of(crate::planning::BankTier::Bank);
        // A host tier's slots are charged to its store's pool entry: the
        // device store's own on unified memory.
        let mut pools = vec![device_pool];
        if ledgers.host().store() != ledgers.device().store() {
            pools.push(ledgers.host().pool_block(crate::memory::ledger::PoolClass::Records, std::num::NonZeroU64::new(self.row()).expect("a slot's block")).blocks);
        }
        assert!(self.pools.set(pools).is_ok(), "invariant violated: a fake bank holds its cache twice");
        ledger
            .change(|_| (0..committed).try_for_each(|_| self.slots.add_slab(seismic::SlabFill::Unwritten).map(drop)))
            .expect("the fixture's ledger holds its bank");
        let above = committed.saturating_sub(1) * self.row();
        let cache = ledger.cache(crate::memory::ledger::CacheKind::Bank, above).expect("the fixture's ledger holds its bank");
        self.cache.hold(crate::BankHolding::Experts, cache);
    }

    /// The pool entry of `ledger`'s store.
    fn pool(&self, ledger: &crate::memory::ledger::Ledger) -> &std::sync::Arc<dyn seismic::memory::Entry> {
        let store = ledger.store();
        let pools = self.pools.get().expect("invariant violated: a fake bank charges before it holds its cache");
        pools.iter().find(|pool| *pool.store() == store).expect("invariant violated: a fake bank's slots in a store without its pool")
    }

    /// The bytes of a slot: a granule of its pool.
    pub(crate) fn row(&self) -> u64 {
        self.slots.pool().granule()
    }

    /// The experts it holds a slot for at most.
    fn experts(&self) -> u64 {
        self.slots.capacity() as u64
    }

    /// Start the cover's reads (`true`) or end them (`false`), as the
    /// bank's residency publishes them.
    pub fn cover_reads(&self, reading: bool) {
        let covers = self.covers.load(std::sync::atomic::Ordering::Acquire);
        assert_ne!(crate::ExpertCover::after(covers).reading, reading, "the cover already reads so");
        self.covers.store(covers + 1, std::sync::atomic::Ordering::Release);
    }

    /// `steps` of the bank's run from now.
    pub fn running_steps(&self, steps: usize) {
        self.steps.store(steps, std::sync::atomic::Ordering::Relaxed);
    }

    /// Take an installed shrink's release now, as the bank's resize thread
    /// does once the device work before it completes: its charge falls
    /// before any advance reports it.
    pub fn release_now(&self) {
        let mut resize = self.resize.lock().unwrap();
        match std::mem::replace(&mut *resize, FakeResize::Idle) {
            FakeResize::Releasing(before) => *resize = self.released(before),
            _ => panic!("no shrink's release is installed"),
        }
    }

    /// Install a resize to `to` slots: a growth backs its slots, each a
    /// granule of the pool, which grows by a block its entry admits; a
    /// shrink frees the slots above `to`. A refused growth backs none.
    fn install(&self, to: u64) -> Result<(), seismic::TensorError> {
        let at = self.slots();
        if to < at {
            let backed = self.slots.backed();
            return backed.iter().rev().take((at - to) as usize).try_for_each(|&slot| self.slots.free_slab(slot).map(drop));
        }
        let mut added = Vec::new();
        for _ in at..to {
            match self.slots.add_slab(seismic::SlabFill::Unwritten) {
                Ok(slot) => added.push(slot),
                Err(error) => {
                    for slot in &added {
                        self.slots.free_slab(*slot)?;
                    }
                    self.slots.pool().return_freed(added.len() as u64 * self.row())?;
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    /// An installed resize released, recorded in the bank's cache as it is
    /// freed: one change, as the served bank's
    /// (`memory::ledger::Changes::change`). A shrink's freed slots return
    /// their blocks, the backend freeing them through the pool's entry, and
    /// its cache holds no more than the slots left above the floor.
    fn released(&self, before: (u64, u64)) -> FakeResize {
        self.cache.changes().clone().change(|_| {
            let after = self.charged();
            if after < before.0 {
                self.slots.pool().return_freed(before.0 - after).expect("a test bank's freed slots return to the OS");
                self.cache.fit(crate::BankHolding::Experts, after.saturating_sub(self.row()));
            }
        });
        FakeResize::Released(before)
    }

    fn in_gap(&self) -> bool {
        self.steps.load(std::sync::atomic::Ordering::Relaxed) == 0
    }

    fn running(&self) -> bool {
        !matches!(*self.resize.lock().unwrap(), FakeResize::Idle) || self.host_grown.lock().unwrap().is_some()
    }

    /// The bytes its slots hold: a granule each.
    fn charged(&self) -> u64 {
        self.slots() * self.row()
    }

    /// Begin a resize to `to` slots, at a boundary.
    fn begin(&self, to: u64, wake: crate::CompletionWake) -> Result<BankStep, DomainError> {
        let mut resize = self.resize.lock().unwrap();
        if !matches!(*resize, FakeResize::Idle) {
            return Ok(BankStep::Running);
        }
        *resize = FakeResize::Prepared { to, before: (self.charged(), self.slots()) };
        wake.complete();
        Ok(BankStep::Begun(crate::planning::BankTier::Bank))
    }
}

impl ElasticBank for FakeBank {
    fn slots(&self) -> u64 {
        self.slots.backed().len() as u64
    }

    fn hold_rises(&self) -> std::sync::MutexGuard<'_, ()> {
        self.rises.lock().expect("the fake bank's charge rises")
    }

    /// The fake streams no layer.
    fn lines(&self) -> std::sync::Arc<[crate::assessment::BankLines]> {
        std::sync::Arc::from([])
    }

    /// The fake streams no layer, so no route reads a policy.
    fn device_convert(&self, _lines: &[crate::assessment::RecordLine]) {}

    #[cfg(test)]
    fn device_landed(&self) -> u64 {
        0
    }

    #[cfg(test)]
    fn resident_decodes(&self) -> u64 {
        0
    }

    /// A growth prepared and not yet installed ends before its first slot,
    /// as the served bank's does; the host tier's grows at once.
    fn stop_growths(&self) {
        let mut resize = self.resize.lock().unwrap();
        *resize = match std::mem::replace(&mut *resize, FakeResize::Idle) {
            FakeResize::Prepared { to, before: (_, at) } if to > at => FakeResize::Stopped(at),
            other => other,
        };
    }

    /// A fake bank moves nothing of its own.
    fn hold_traffic(&self) -> super::TrafficHold {
        super::TrafficHold::new(|| ())
    }

    fn traffic_quiet(&self) -> bool {
        true
    }

    fn wait_traffic_quiet(&self) -> Result<(), DomainError> {
        Ok(())
    }

    /// As the bank's: the copy is priced only once the device holds its
    /// grid.
    fn tier_copy(&self, costs: &crate::assessment::TierCopyCosts) -> Result<(), DomainError> {
        if self.refuse_tier_copy {
            return Err(DomainError::invariant("the tier copy's grid could not be written"));
        }
        *self.tier_copy.lock().unwrap() = Some(costs.clone());
        Ok(())
    }

    fn price(&self, _lines: Option<std::sync::Arc<[crate::assessment::ExpertLines]>>, _pass: Option<crate::assessment::ExpertPass>) {}

    /// Its slots shrink to `to`, at least one; its host tier releases
    /// nothing.
    fn shrink(
        &self,
        holding: crate::BankHolding,
        to: u64,
        between_flights: &mut dyn FnMut() -> Result<(), DomainError>,
        wake: crate::CompletionWake,
    ) -> Result<BankStep, DomainError> {
        if self.running() {
            return Ok(BankStep::Running);
        }
        let to = to.max(1);
        if holding != crate::BankHolding::Experts || to >= self.slots() {
            return Ok(BankStep::AtLimit);
        }
        between_flights()?;
        if !self.in_gap() {
            return Ok(BankStep::Deferred);
        }
        self.begin(to, wake)
    }

    fn carried(&self) -> Result<(), DomainError> {
        Ok(())
    }

    /// Its slots, a granule each, from one to where its decode's
    /// misses stop falling on `price` (every row before the load is
    /// priced), and its host tier's, a byte of upload storage each, to its
    /// ceiling: derived from the domain's ledgers as the served bank's are.
    fn targets(&self, price: Option<&crate::assessment::DecodePrice>) -> Result<Vec<crate::experts::derived::HoldingTarget<'_>>, DomainError> {
        use crate::experts::derived::{BankShape, HoldingTarget};
        let (rows, unit) = (self.experts(), self.row());
        let (curve, at) = (price.and_then(crate::assessment::DecodePrice::saturation), self.slots());
        let mut targets = vec![HoldingTarget {
            holding: crate::BankHolding::Experts,
            tier: crate::planning::BankTier::Bank,
            at,
            owed: self.cache.owed(crate::BankHolding::Experts, at.saturating_sub(1) * unit),
            shape: BankShape::curved(unit, 1, rows, curve, at),
            charge: Box::new(move |slots| slots * unit),
        }];
        if self.host_ceiling > 0 {
            targets.push(HoldingTarget {
                holding: crate::BankHolding::Host,
                tier: crate::planning::BankTier::Host,
                at: self.host.load(std::sync::atomic::Ordering::Relaxed),
                owed: self.cache.owed(crate::BankHolding::Host, self.host.load(std::sync::atomic::Ordering::Relaxed)),
                shape: BankShape::new(1, 0, self.host_ceiling, self.host_ceiling),
                charge: Box::new(|slots| slots),
            });
        }
        Ok(targets)
    }

    /// A row a slot, down to one.
    #[cfg(test)]
    fn released_to(&self, bytes: u64) -> Result<Option<u64>, DomainError> {
        let unit = self.row();
        let at = self.slots();
        Ok((at > 1).then(|| at.saturating_sub(bytes.div_ceil(unit).max(1)).max(1)))
    }

    fn live(&self) -> Option<crate::assessment::LiveDecode> {
        self.decode.clone()
    }

    /// Its one streamed layer of 4 experts, each step routing 4, on the
    /// reference lines of 1 MB records, whatever lines the load priced.
    fn streamed_layers(&self, _lines: std::sync::Arc<[crate::assessment::ExpertLines]>) -> crate::assessment::StreamedLayers {
        crate::assessment::StreamedLayers::new(vec![4], 4, std::sync::Arc::from([crate::assessment::ExpertLines::reference(1_000_000)]))
    }

    /// Its slots' growth takes its cache on `ledgers` first, as the served
    /// bank's does.
    fn grow(&self, ledgers: &crate::memory::ledger::DeviceLedgers, holding: crate::BankHolding, to: u64, wake: crate::CompletionWake) -> Result<BankStep, DomainError> {
        if self.running() {
            return Ok(BankStep::Running);
        }
        if holding == crate::BankHolding::Host {
            let from = self.host.load(std::sync::atomic::Ordering::Relaxed);
            let Ok(claim) = ledgers.of(crate::planning::BankTier::Host).backing(crate::memory::ledger::PoolClass::Records, to.saturating_sub(from)) else {
                return Ok(BankStep::Deferred);
            };
            // Its slots back at once, the host pool's blocks charged.
            let host = ledgers.of(crate::planning::BankTier::Host);
            ledgers.changes().change(|_| { let entry = self.pool(host); entry.take(&entry.change(), to.saturating_sub(from), seismic::memory::Wiring::Pageable) }).expect("the claim admitted the slots");
            self.cache.grow(holding, claim);
            self.host.store(to, std::sync::atomic::Ordering::Relaxed);
            let now = std::time::Instant::now();
            let zero = std::time::Duration::ZERO;
            let times = super::ResizeTimes {
                decided: now,
                queued: zero,
                planned: zero,
                trips: Default::default(),
                                prepared: now,
                installed: now,
                released: now,
            };
            let holding = crate::BankHolding::Host;
            *self.host_grown.lock().unwrap() = Some(BankResized::Grown { holding, from, to, grown: to - from, refused: None, times });
            wake.complete();
            return Ok(BankStep::Begun(crate::planning::BankTier::Host));
        }
        let bytes = to.saturating_sub(self.slots()) * self.row();
        let Ok(claim) = ledgers.of(crate::planning::BankTier::Bank).backing(crate::memory::ledger::PoolClass::Records, bytes) else {
            return Ok(BankStep::Deferred);
        };
        let step = self.begin(to, wake)?;
        if let BankStep::Begun(_) = step {
            // The claim holds the slots' granules as they back; the pool's
            // entry is charged its blocks as the resize installs them.
            self.cache.grow(holding, claim);
        }

        Ok(step)
    }

    fn holding_storage(&self, holding: crate::BankHolding) -> Result<crate::planning::BankTier, DomainError> {
        Ok(match holding {
            crate::BankHolding::Host => crate::planning::BankTier::Host,
            crate::BankHolding::Experts | crate::BankHolding::Table { .. } => crate::planning::BankTier::Bank,
        })
    }

    fn advance(&self, wake: crate::CompletionWake) -> Result<Option<BankResized>, DomainError> {
        if let Some(grown) = self.host_grown.lock().unwrap().take() {
            self.cache.settle(&grown);
            wake.complete();
            return Ok(Some(grown));
        }
        let mut resize = self.resize.lock().unwrap();
        let (next, finished) = match std::mem::replace(&mut *resize, FakeResize::Idle) {
            FakeResize::Prepared { to, before } => match self.install(to) {
                Ok(()) => (FakeResize::Releasing(before), None),
                // A shrink needs no memory: its storage refusing it breaks
                // an invariant, as the bank's does.
                Err(refusal) if to < before.1 => {
                    return Err(DomainError::invariant(format!("the expert bank's storage refused a shrink: {refusal}")));
                }
                Err(refusal) => (FakeResize::Refused(refusal), None),
            },
            // The release taken now, or already by the bank's own thread.
            FakeResize::Releasing(before) => {
                *resize = self.released(before);
                drop(resize);
                return self.advance(wake);
            }
            FakeResize::Released((charged, from)) => {
                let (slots, after) = (self.slots(), self.charged());
                let now = std::time::Instant::now();
                let zero = std::time::Duration::ZERO;
                let times = crate::domain::ResizeTimes {
                    decided: now,
                    queued: zero,
                    planned: zero,
                    trips: Default::default(),
                                        prepared: now,
                    installed: now,
                    released: now,
                };
                let finished = match after < charged {
                    true => BankResized::Shrunk {
                        holding: crate::BankHolding::Experts,
                        from,
                        to: slots,
                        released: charged - after,
                        returned: charged - after,
                        times,
                    },
                    false => BankResized::Grown {
                        holding: crate::BankHolding::Experts,
                        from,
                        to: slots,
                        grown: after - charged,
                        refused: None,
                        times,
                    },
                };
                (FakeResize::Idle, Some(finished))
            }
            FakeResize::Stopped(at) => {
                let stopped = BankResized::Stopped { holding: crate::BankHolding::Experts, at };
                (FakeResize::Idle, Some(stopped))
            }
            FakeResize::Refused(refusal) => {
                let at = self.slots();
                let refused = BankResized::GrowthRefused { holding: crate::BankHolding::Experts, at, refusal };
                (FakeResize::Idle, Some(refused))
            }
            FakeResize::Idle => (FakeResize::Idle, None),
        };
        *resize = next;
        if let Some(finished) = &finished {
            self.cache.settle(finished);
        }
        wake.complete();
        Ok(finished)
    }

    /// One layer of one-byte experts, each a row, held in host RAM.
    fn in_gap(&self) -> bool {
        FakeBank::in_gap(self)
    }

    /// Its chains' encoding never cuts it.
    fn running_begun(&self) -> Option<crate::domain::RunningBegun> {
        None
    }

    fn value(&self, value: crate::memory::ledger::Value) {
        self.cache.value(value);
    }

    fn residency(&self) -> Result<crate::ExpertResidency, DomainError> {
        let experts = self.experts();
        Ok(crate::ExpertResidency {
            experts: std::num::NonZeroU64::new(experts).expect("a fake bank of experts"),
            slots: self.slots(),
            floor: 1,
            ceiling: experts,
            stores: vec![(seismic::StoreIdentity::HostRam, self.charged())],
            // A host tier's host slots, a byte of upload storage each.
            host: std::num::NonZeroU64::new(self.host_ceiling).map(|ceiling| {
                let slots = self.host.load(std::sync::atomic::Ordering::Relaxed);
                let last_resize = *self.host_last_resize.lock().expect("the fake host tier's last resize");
                crate::HostResidency { slots, ceiling: ceiling.get(), bytes: slots, last_resize }
            }),
            cover: self.cover(),
            disk_bytes: experts - self.slots(),
            tables: Vec::new(),
            last_resize: *self.last_resize.lock().expect("the fake bank's last resize"),
        })
    }

    fn record_resize(&self, holding: crate::BankHolding, resize: crate::BankResize) {
        let last = match holding {
            crate::BankHolding::Experts => &self.last_resize,
            crate::BankHolding::Host if self.host_ceiling > 0 => &self.host_last_resize,
            holding => unreachable!("the fake bank holds no {holding:?}"),
        };
        *last.lock().expect("the fake bank's last resize") = Some(resize);
    }

    fn cover(&self) -> crate::ExpertCover {
        crate::ExpertCover::after(self.covers.load(std::sync::atomic::Ordering::Acquire))
    }

    /// The fake's waits are its tests' own: none waits for its cover.
    fn wait_covered(&self) -> Result<(), DomainError> {
        assert!(!self.cover().reading, "a test waits for no cover it reads");
        Ok(())
    }
}

impl TestFamily {
    /// As the bank's watcher takes or gives back a group's places
    /// (`ExpertBank::hold_places`): only between flights, an invariant
    /// broken while a step of the bank runs.
    fn in_gap(&self) -> Result<(), String> {
        match &self.bank {
            Some(bank) if !bank.in_gap() => {
                Err(format!("a bank resize while {} steps run", bank.steps.load(std::sync::atomic::Ordering::Relaxed)))
            }
            _ => Ok(()),
        }
    }

    fn submit<L, W, O>(&self, core: L, workspace: W, output: O) -> TestSubmission<L, W, O> {
        match &self.control {
            Some(control) => TestSubmission::Pending(PendingSubmission {
                completion: PendingCompletion(control.clone()),
                core,
                workspace,
                output,
            }),
            None => TestSubmission::Ready(ReadySubmission::new(core, workspace, output)),
        }
    }
}
impl ProgramFamily for TestFamily {
    type TargetSubmission = TestSubmission<TargetLaunchCore, (), crate::TargetOutput>;
    type HeadSubmission = ControlledHeadSubmission;
    type VisionSubmission = PendingFailureSubmission<CompletedVisionWork>;
    type StateSubmission =
        ReadySubmission<crate::StateLaunchCore, crate::NativeGraphWorkspaceLease, ()>;
    fn bind_head(&mut self, _: &seismic::Charges, _: crate::ResidentHead, _: &ModelDefinition) -> Result<(), String> {
        Ok(())
    }
    fn bind_vision(&mut self, _: &seismic::Charges, _: crate::ResidentVision, _: &ModelDefinition) -> Result<(), String> {
        Ok(())
    }
    fn head_is_bound(&self) -> bool {
        false
    }
    fn family_graph_replays(&self) -> Vec<(&'static str, seismic::GraphReplays)> {
        Vec::new()
    }
    fn vision_is_bound(&self) -> bool {
        false
    }
    fn state_is_bound(&self) -> bool {
        true
    }
    fn unbind_optional(&mut self, _: super::OptionalComponent) {}

    fn target_weight_bytes(&self) -> Result<u64, &'static str> {
        Ok(self.bank.as_ref().map_or(0, FakeBank::charged))
    }

    fn optional_constant_bytes(&self) -> Result<u64, &'static str> {
        Ok(self.optional.as_ref().map_or(0, seismic::Tensor::charged_bytes))
    }

    fn expert_bank(&self) -> Option<&dyn ElasticBank> {
        self.bank.as_ref().map(|bank| bank as &dyn ElasticBank)
    }

    fn hold_group(
        &self,
        _charges: &seismic::Charges,
        _places: std::num::NonZeroUsize,
        _quiet: super::Quiet,
    ) -> Result<(), crate::resources::AllocationError> {
        self.in_gap().map_err(|detail| crate::resources::AllocationError::Plan(crate::error::InvariantError { context: "a group's places", detail }))
    }

    fn release_group(&self, _quiet: super::Quiet) -> Result<(), String> {
        self.in_gap()
    }

    fn prepared_program_bytes(&self) -> u64 {
        self.programs.device_storage_bytes()
    }
    fn chains(&self, _rows: usize) -> bool {
        true
    }
    fn waiting_group(&self, capacity: usize) -> usize {
        self.waiting.min(capacity)
    }
    fn submit_target(
        &mut self,
        chain: Chain<ValidatedTargetLaunch>,
        _spare: crate::TargetGraphOutputLease,
        _pressure: &dyn Fn(&dyn ElasticBank) -> Option<crate::domain::RunningPressure>,
    ) -> Result<Chain<Self::TargetSubmission>, crate::programs::ChainRefusal>
    {
        // The fixture runs no kernels, so it plans no readout: a launch
        // with rows demanding logits, features or a selection is refused
        // here, before it runs, rather than finishing without them.
        if chain.iter().any(|launch| !launch.batch().upload().out_rows.is_empty()) {
            let error = crate::SubmitError::Invariant(crate::InvariantError {
                context: "controlled fixture",
                detail: "the fixture plans no readout for a launch demanding outputs".into(),
            });
            return Err(crate::programs::ChainRefusal { error, chain });
        }
        if REJECTED.with(|rejected| rejected.get()) == Some(0) {
            REJECTED.with(|rejected| rejected.set(Some(chain.len())));
            let error = crate::DeviceError::Execution("the fixture rejects the chain".into());
            return Err(crate::programs::ChainRefusal {
                error: crate::SubmitError::Device(error),
                chain,
            });
        }
        let now = std::time::Instant::now();
        Ok(chain.map(|launch| {
            let (core, _) = launch.into_submission_parts();
            self.submit(
                core,
                (),
                crate::TargetOutput {
                    boundary: None,
                    readout: None,
                    commits: crate::CommitSpan {
                        first: now,
                        last: now,
                    },
                    landed_reads: Vec::new(),
                    stored_reads: STORED_READS
                        .with(|reads| std::mem::take(&mut *reads.borrow_mut())),
                    expert_step: EXPERT_STEPS.with(|steps| steps.borrow_mut().take()),
                },
            )
        }))
    }
    fn submit_head(
        &mut self,
        _: ValidatedHeadLaunch,
    ) -> Result<Self::HeadSubmission, (crate::SubmitError, ValidatedHeadLaunch)> {
        panic!("controlled fixture does not submit head work")
    }
    fn submit_vision(
        &mut self,
        _: ValidatedVisionLaunch,
    ) -> Result<Self::VisionSubmission, (crate::SubmitError, ValidatedVisionLaunch)> {
        panic!("controlled fixture does not submit vision work")
    }
    fn submit_state(
        &mut self,
        launch: ValidatedStateLaunch,
    ) -> Result<Self::StateSubmission, (crate::SubmitError, ValidatedStateLaunch)> {
        // Like target work, this fixture models ownership and completion only.
        // Numerical copy correctness is qualified by native state tests.
        let (core, workspace) = launch.into_submission_parts();
        Ok(ReadySubmission::new(core, workspace, ()))
    }
}
