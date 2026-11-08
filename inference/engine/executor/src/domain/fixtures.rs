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

/// The machine as a test's heap reads it: the host's sample and Seismic's
/// runtime reading (this process's charges, the grants only an opened
/// executor reads: a discrete CUDA device's free memory) as first read,
/// moved only by what the fixture's own `device` charged since (`charged`,
/// each store's then): each store's charge rises and falls with the
/// device's, and the room the host and each grant have left falls by what
/// the device takes of their store and rises by what it gives back. No
/// other process's drift enters what a test decides, nor another test's
/// (each fixture opens its own device, and tests run side by side in one
/// process, on one GPU), and a release the device makes (a bank's shrink)
/// is room the next reading sees, as the machine's would once the host
/// takes it back. Each process limit enforced against charged memory (a
/// cgroup's, a Windows job's commit) moves with the device's charges to
/// host RAM the same way; an address-space or data-segment limit counts the
/// reservation, which a resize keeps. The host's wired pages move with the
/// device's charges too, as a Metal device's allocations are wired, and the
/// process's footprint is what the device charges host RAM, each release
/// leaving it at once. Another program's hold on a store
/// ([`Press`]) takes its bytes of the room left there while it lasts, and,
/// sharing the process's cgroups, of what each cgroup limit has left.
struct StillHost {
    device: Rc<seismic::Device>,
    topology: std::sync::Arc<seismic::DeviceTopology>,
    first: seismic::Status,
    charged: BTreeMap<seismic::StoreId, u64>,
    pressed: Pressed,
}

/// What other programs hold of each store, as a test presses it.
type Pressed = Rc<std::cell::RefCell<BTreeMap<seismic::StoreId, u64>>>;

/// Another program's hold on the store of a fixture's device storage
/// ([`press`]), which the fixture's heap reads as the host's: the room left
/// there falls by what it takes, and rises back as it lets go.
pub struct Press {
    store: seismic::StoreId,
    pressed: Pressed,
}

impl Press {
    /// The store it presses.
    pub fn store(&self) -> seismic::StoreId {
        self.store
    }

    /// Hold `bytes` of the store from now, in place of what it held.
    pub fn take(&self, bytes: u64) {
        self.pressed.borrow_mut().insert(self.store, bytes);
    }

    /// Hold nothing of the store from now.
    pub fn release(&self) {
        self.pressed.borrow_mut().remove(&self.store);
    }
}

/// A hold on the store of `domain`'s device storage ([`Press`]).
pub fn press(domain: &ExecutorDomain<TestFamily>) -> Press {
    let store = domain.store_of(seismic::Contract::DeviceOnly).expect("the device's storage lies in a store");
    Press { store, pressed: domain.family.pressed.clone() }
}

impl StillHost {
    fn of(catalog: seismic::Devices, device: Rc<seismic::Device>, pressed: Pressed) -> Self {
        let first = catalog.status(device.topology(), device.info().id());
        let charged = Self::charges(&device);
        Self { device, topology: catalog.topology(), first, charged, pressed }
    }

    /// What `device`'s allocations charge each store it reaches: its own
    /// ledgers.
    fn charges(device: &seismic::Device) -> BTreeMap<seismic::StoreId, u64> {
        device.ledgers().own.iter().map(|(store, ledger)| (*store, ledger.charged)).collect()
    }
}

impl StatusReading for StillHost {
    fn status(&self, _: &seismic::Device) -> seismic::Status {
        let now = Self::charges(&self.device);
        let charged = |charges: &BTreeMap<seismic::StoreId, u64>, store| charges.get(&store).copied().unwrap_or(0);
        // `left` of `store`, less what the device took of it since the first
        // reading and what another program holds of it now.
        let pressed = self.pressed.borrow();
        let held = |store| pressed.get(&store).copied().unwrap_or(0);
        let moved = |left: u64, store| {
            (left + charged(&self.charged, store)).saturating_sub(charged(&now, store) + held(store))
        };
        // `used` of a limit on `store`, plus what the device took of it since
        // the first reading and what another program sharing it holds.
        let used = |used: u64, store, held: u64| {
            (used + charged(&now, store) + held).saturating_sub(charged(&self.charged, store))
        };
        let topology = &self.topology;
        let seismic::Status { mut sample, mut runtime } = self.first.clone();
        // Each store's charge then, plus what the device charged it since.
        let mut charges = BTreeMap::<seismic::StoreId, u64>::new();
        for &(store, bytes) in &runtime.charges {
            *charges.entry(store).or_default() += bytes;
        }
        for &store in self.charged.keys().chain(now.keys()) {
            charges.entry(store).or_default();
        }
        runtime.charges = charges
            .into_iter()
            .map(|(store, bytes)| (store, (bytes + charged(&now, store)).saturating_sub(charged(&self.charged, store))))
            .collect();
        if let Ok(host) = &mut sample.host_ram {
            let store = topology.host_ram().id();
            host.allocatable.value = moved(host.allocatable.value, store);
            host.obtainable.value = moved(host.obtainable.value, store);
            if let Some(held) = &mut host.held {
                held.wired_bytes = used(held.wired_bytes, store, 0);
            }
            for limit in &mut host.limits {
                limit.used_bytes = match limit.kind {
                    seismic::ProcessLimitKind::CgroupV1 { .. } | seismic::ProcessLimitKind::CgroupV2 { .. } => {
                        used(limit.used_bytes, store, held(store))
                    }
                    seismic::ProcessLimitKind::WindowsJobProcess => used(limit.used_bytes, store, 0),
                    seismic::ProcessLimitKind::AddressSpace | seismic::ProcessLimitKind::DataSegment => limit.used_bytes,
                };
            }
        }
        for (path, grant) in sample.grants.iter_mut().chain(runtime.grants.iter_mut()) {
            if let Ok(grant) = grant {
                let store = topology.access_path(*path).expect("a read grant's path belongs to its topology").store();
                grant.value = moved(grant.value, store);
            }
        }
        seismic::Status { sample, runtime }
    }

    /// What the device charges host RAM now.
    fn footprint(&self) -> Option<u64> {
        Some(Self::charges(&self.device).get(&self.topology.host_ram().id()).copied().unwrap_or(0))
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

fn tiny_manifest(definition: &ModelDefinition) -> PackageManifest {
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

/// A spare head store of 4096-wide F32 rows: 2048 rows per slab, 8192 rows,
/// one slab backed. It grows independently of the target model.
fn spare_head_store(device: Rc<seismic::Device>) -> StoreBindings {
    use magnitude_state::{
        BankCapacity, CodecSpec, ComponentDescriptor, HistoryDomainLayout, HistoryDomainPlan,
        LayerRef,
    };
    StateStore::new(
        device,
        8192,
        8192,
        vec![HistoryDomainPlan {
            layout: HistoryDomainLayout::Token {
                components: vec![
                    ComponentDescriptor::new(
                        LayerRef::Target(0),
                        CodecSpec::dense(seismic::DType::F32, 4096, 4096),
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
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let reserves = platform::MemoryReserves::of_machine(None);
    let selected = platform::select_device(&catalog, ExecutionPath::Native, device_request, &reserves)
        .unwrap_or_else(|error| panic!("{device_request} selects a device: {error}"));
    let device = Rc::new(
        crate::test_support::open(&catalog, catalog.resolve(selected.info.selector()).unwrap())
            .unwrap_or_else(|error| panic!("the selected device opens: {error}")),
    );
    let definition = Rc::new(definition);
    let manifest = tiny_manifest(&definition);
    let capacity_bytes = ResourceCapacity {
        domain_bytes: selected.assessment_capacity_bytes.min(512 * 1024 * 1024),
        tensor_operations: crate::TensorOperations::of(selected.info.forms_tensor_operations()),
    };
    let draft = ExecutionPlanner::prepare(
        &selected,
        &manifest,
        &definition,
        ComponentSelection {
            head: false,
            vision: false,
        },
        ExecutionPath::Native,
        PlannedMethod::Plain,
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
        &device,
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
            PlannedMethod::Plain,
        )
        .unwrap();
    let resources_plan = ResourcePlanner::plan_with_state(
        state,
        &target_graphs,
        &target_readout_graphs,
        None,
        None,
        programs.state_graphs().unwrap(),
    )
    .unwrap();
    let execution = Rc::new(draft.admit(resources_plan).unwrap());
    let id = ResourceDomainId::new("controlled-target-domain").unwrap();
    let resources = ResourceAllocator::allocate(
        &execution,
        &device,
        id,
        &target_graphs,
        &target_readout_graphs,
        None,
        None,
        programs.state_graphs().unwrap(),
    )
    .unwrap();
    let store = execution
        .resources()
        .target_state()
        .allocate(device.clone())
        .unwrap();
    let head = head.then(|| spare_head_store(device.clone()));
    let pressed = Pressed::default();
    ExecutorDomain::with_family(
        execution,
        definition,
        resources,
        DeviceHeap::open(Rc::new(StillHost::of(catalog, device.clone(), pressed.clone())), reserves, device).unwrap(),
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
        },
    )
}

/// The backends a bank fixture is built on ([`banked`]): each discovery
/// reports available ([`available`]). Every backend's devices reserve device
/// storage (each backend's kinds declare a reservation, the CPU's in host
/// memory), so no backend is left out for reserving none: an available one
/// whose device reserves none, or says why it cannot, fails the test.
pub fn bank_backends() -> Vec<seismic::BackendName> {
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    seismic::BackendName::ALL.into_iter().filter(|&backend| available(&catalog, backend)).collect()
}

/// A fixture on `backend`'s device whose family holds a fake elastic bank
/// of `rows` rows, each the device's reservation granule of device storage
/// ([`bank_row`]: a row is then backed, charged and freed whole, whatever
/// the backend's granule, 2 MiB on Metal and CUDA, Vulkan's growing with
/// the device's memory, the host page on the CPU), `committed` of them
/// backed, which the static holding charges.
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
    let device = domain.domain.device();
    let row = match device.info().reserving(seismic::ReservedContract::DeviceOnly) {
        Ok(Some(reservation)) => reservation.granule,
        Ok(None) => panic!("{backend:?}: the device reserves no device storage"),
        Err(why) => panic!("{backend:?}: the device cannot reserve device storage: {why}"),
    };
    let (tensor, resizer) = seismic::Tensor::reserved(device, seismic::Element::bool(), &[rows, row], committed)
        .unwrap_or_else(|error| panic!("{backend:?}: the device reserves the bank: {error}"));
    domain.family.bank = Some(FakeBank::new(tensor, resizer));
    domain.refresh_memory().unwrap();
    (domain, bindings)
}

/// A decode the fake bank counted ([`FakeBank::decode`]), and where its
/// misses stop falling.
#[derive(Clone, Debug)]
pub struct FakeDecode {
    pub live: crate::assessment::LiveDecode,
    pub saturation: u64,
}

/// A decode of `steps` steps at the fake bank's one streamed layer of 4
/// experts, each step reading all 4 of them, all `depth` deep in the
/// bank's stack: a bank of `depth` slots misses them, one of `depth + 1`
/// none, beside the generic model counted as one step. Reads take 0.4 ms and 0.5 ms a MB ([`READ`]); an
/// all-hit step 15 ms.
pub fn misses_at(depth: usize, steps: u64) -> FakeDecode {
    misses_by(steps, &[(depth, 4)])
}

/// [`misses_at`] with each step's reads spread over the stack: `reads` of
/// each `(depth, reads)`, which a bank of `depth` slots misses and one of
/// `depth + 1` hits. The misses stop falling past the deepest.
pub fn misses_by(steps: u64, reads: &[(usize, u64)]) -> FakeDecode {
    let mut misses = [0; 5];
    let mut requests = [0; 5];
    for &(depth, count) in reads {
        misses[depth] = steps * count;
        requests[depth] = steps;
    }
    let depth = reads.iter().map(|&(depth, _)| depth).max().expect("a decode reads");
    FakeDecode {
        live: crate::assessment::LiveDecode {
            layers: vec![crate::assessment::LayerReads::of_depths(&misses, &requests)],
            steps,
            rows: 1.,
            depth: 100,
        },
        saturation: depth as u64 + 1,
    }
}

/// What a host slot of the fake bank saves a decode step, in seconds: its
/// one streamed layer prices no tier read apart from a file read, so the
/// fake gives each slot, a byte of upload storage, an attosecond: a holding
/// that saves something, less per byte than any slot of device storage
/// whose saving the price counts.
pub const HOST_SLOT_SAVES: f64 = 1e-18;

/// The fake decode's read line: 0.4 ms and 0.5 ms a MB.
pub const READ: crate::assessment::ReadLine = match crate::assessment::ReadLine::new(4e-4, 1. / 2e9) {
    Ok(line) => line,
    Err(_) => panic!("the fixtures' read line"),
};

/// Shrink `domain`'s state surplus by `policy` and wait for the idle graph
/// slots' release it began, if any, to end on the release thread, then
/// finish it as a boundary does (`ExecutorDomain::advance_slot_release`):
/// the bytes the shrink released at once and the slots' release released.
pub fn shrink_settled<F: ProgramFamily>(
    domain: &mut ExecutorDomain<F>,
    bindings: &mut StateBindings,
    policy: magnitude_state::ShrinkPolicy,
) -> u64 {
    let shrunk = domain.shrink_state(bindings, policy, crate::CompletionWake::new(|| {})).unwrap();
    let slots = match shrunk.slots {
        super::state::SlotRelease::Began => loop {
            if let Some(released) = domain.advance_slot_release().unwrap() {
                break released.bytes;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        },
        super::state::SlotRelease::None | super::state::SlotRelease::Running => 0,
    };
    shrunk.released + slots
}

/// The growth rule's inputs: the decode the fake bank counted, the
/// fixtures' read line, and the resident step it is priced on: 15 ms at every depth, its all-hit
/// steps' time.
pub fn price_growth(domain: &mut ExecutorDomain<TestFamily>, bindings: &mut StateBindings, decode: FakeDecode) {
    count_decode(domain, decode);
    measure_load(domain, bindings);
}

/// What a fixture's prompt group's submission costs beyond a chained
/// launch: 50 µs, a host round trip's order.
pub const SUBMISSION: f64 = 50e-6;

/// The load measured its costs: a step of 15 ms in either form at every
/// depth, a banked load's all-hit steps' time, and each prompt group's
/// submission [`SUBMISSION`]; a banked load's experts on the fixtures' read
/// line, its decode priced on whatever the fake bank counted, the generic
/// model alone before any.
pub fn measure_load(domain: &mut ExecutorDomain<TestFamily>, bindings: &mut StateBindings) {
    let experts = domain.family.bank.is_some().then(|| crate::ExpertMeasures {
        lines: crate::assessment::Priced::Measured(std::sync::Arc::from([])),
        read: crate::assessment::Priced::Measured(READ),
        pass: crate::assessment::Priced::Pending("unpriced".into()),
        moves: crate::assessment::Priced::Pending("unmeasured".into()),
        tier_copy: crate::assessment::Priced::Measured(None),
        convert: crate::assessment::Priced::Measured(None),
    });
    domain.measure_load(crate::LoadMeasures { costs: Ok(load_costs()), experts }, bindings).unwrap();
}

/// A fixture's costs: a step of 15 ms in either form at every depth, and
/// each prompt group's submission [`SUBMISSION`].
pub fn load_costs() -> crate::LoadCosts {
    crate::LoadCosts {
        step: Box::new(|_, _| Ok(crate::assessment::FormSeconds { resident: 0.015, streamed: 0.015 })),
        submission: SUBMISSION,
    }
}

/// A domain launching one-row prompt chunks in chains of up to three on
/// the device `device_request` selects, its device's store read with a
/// gigabyte of room ([`with_room`]): a load whose prompt groups take their
/// places by their price once it is priced.
pub fn chained(device_request: platform::DeviceRequest) -> Fixture {
    let (mut domain, bindings) = fixture_of(tiny_definition(), limits(2, 3, false), None, false, device_request);
    with_room(&mut domain, 1 << 30);
    (domain, bindings)
}

/// A reading of `domain`'s device store with `room` bytes of headroom, in
/// the Normal band, the host seen taking nothing.
pub fn with_room<F: ProgramFamily>(domain: &mut ExecutorDomain<F>, room: u64) {
    let store = domain.domain.device().device_store();
    domain.pace.borrow_mut().domains.insert(store, DomainPace {
        role: DomainRole::Allocation,
        headroom: room,
        thresholds: crate::platform::DomainThresholds { planning_bytes: 0, emergency_bytes: 0 },
        charged: 0,
        band: MemoryBand::Normal,
        unseen: 0,
        witness: None,
        untaken: Default::default(),
        rate: 0.0,
        fluctuation: Fluctuation::default(),
        host: false,
        grant_left: None,
        borrowable: None,
        takes: None,
        first: std::time::Instant::now(),
    });
}

/// The places a prompt group of four one-row chunks opened on `fixture`
/// holds, its first chunk submitted with the rest planned behind it.
pub fn group_places_opened<F: ProgramFamily>((mut domain, mut bindings): (ExecutorDomain<F>, StateBindings)) -> usize {
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
    assert!(domain.open_state(&mut bindings, request, None).unwrap().is_empty());
    domain.plan_successors((1..4).map(chunk).collect());
    let operations = vec![chunk(0)];
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

/// The bytes of a row of `domain`'s fake bank ([`banked`]).
pub fn bank_row(domain: &ExecutorDomain<TestFamily>) -> u64 {
    domain.family.bank.as_ref().expect("a banked fixture").tensor.extents()[1]
}

/// The fake bank of `domain` counted `decode`, which its growth is priced
/// on.
pub fn count_decode(domain: &mut ExecutorDomain<TestFamily>, decode: FakeDecode) {
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
    /// A fake elastic expert bank: one reserved tensor, one row a slot.
    pub(crate) bank: Option<FakeBank>,
    /// The size rule's chunks while decode waits, within the capacity.
    pub(crate) waiting: usize,
    /// An optional component's constants, a tensor of its own: what the
    /// optional holding classifies.
    pub(crate) optional: Option<seismic::Tensor>,
    /// What other programs hold of each store its heap reads ([`press`]).
    pressed: Pressed,
}

/// A fake elastic expert bank (`ElasticBank`): one reserved tensor of a row
/// a slot, its floor one row. A resize prepares at once and finishes over
/// two boundaries, as the bank's does: the first installs, the second takes
/// the release (a growth, as one slot that joins beside running steps). Its
/// growth is priced on the decode it counted, `decode`. As the bank's watcher, it is asked to
/// shrink only in a gap: a shrink begins only while none of its `steps`
/// runs.
pub struct FakeBank {
    pub(crate) tensor: seismic::Tensor,
    resize: std::sync::Mutex<Option<FakeResize>>,
    pub(crate) decode: Option<FakeDecode>,
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
    /// Its fill's starts and finishes ([`ElasticBank::fill_epoch`]): one,
    /// filled as it opens, until a test fills it ([`Self::fill`]).
    fills: std::sync::atomic::AtomicU64,
}

enum FakeResize {
    Idle(seismic::Resizer),
    /// With the charge and the slots before the resize, and whether it
    /// grows.
    Prepared(seismic::PreparedRecommit, (u64, u64), bool),
    Releasing(seismic::RecommitRelease, (u64, u64)),
    /// Released, as the bank's resize thread releases, before an advance
    /// reports it.
    Released(seismic::Resizer, (u64, u64)),
    /// Refused, a shrink's (`true`) breaking an invariant as the bank's does.
    Refused(seismic::Resizer, seismic::TensorError, bool),
    /// A growth stopped before its first slot, at these slots, before an
    /// advance reports it.
    Stopped(seismic::Resizer, u64),
}

impl FakeBank {
    pub fn new(tensor: seismic::Tensor, resizer: seismic::Resizer) -> Self {
        Self {
            tensor,
            resize: std::sync::Mutex::new(Some(FakeResize::Idle(resizer))),
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
            fills: std::sync::atomic::AtomicU64::new(1),
        }
    }

    /// Start a fill (`true`) or finish it (`false`), as the bank's
    /// residency publishes one.
    pub fn fill(&self, filling: bool) {
        let fills = self.fills.load(std::sync::atomic::Ordering::Acquire);
        assert_eq!(fills % 2 == 0, !filling, "the fill is already {}", if filling { "running" } else { "finished" });
        self.fills.store(fills + 1, std::sync::atomic::Ordering::Release);
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
        match resize.take().unwrap() {
            FakeResize::Releasing(release, before) => *resize = Some(FakeResize::Released(release.release(), before)),
            _ => panic!("no shrink's release is installed"),
        }
    }

    fn in_gap(&self) -> bool {
        self.steps.load(std::sync::atomic::Ordering::Relaxed) == 0
    }

    fn running(&self) -> bool {
        !matches!(self.resize.lock().unwrap().as_ref(), Some(FakeResize::Idle(_)))
            || self.host_grown.lock().unwrap().is_some()
    }

    fn charged(&self) -> u64 {
        self.tensor.observe_storage().charged_bytes().unwrap()
    }

    /// Begin a resize to `to` rows, at a boundary.
    fn begin(&self, to: u64, wake: crate::CompletionWake) -> Result<BankStep, DomainError> {
        let mut resize = self.resize.lock().unwrap();
        let resizer = match resize.take().unwrap() {
            FakeResize::Idle(resizer) => resizer,
            running => {
                *resize = Some(running);
                return Ok(BankStep::Running);
            }
        };
        let before = (self.charged(), self.slots());
        *resize = Some(match resizer.prepare(to) {
            Ok(prepared) => FakeResize::Prepared(prepared, before, to > before.1),
            Err(refusal) => FakeResize::Refused(refusal.resizer, refusal.error, to < before.1),
        });
        wake.complete();
        Ok(BankStep::Begun(crate::planning::BankTier::Bank))
    }
}

impl ElasticBank for FakeBank {
    fn slots(&self) -> u64 {
        self.tensor.committed_rows()
    }

    fn hold_rises(&self) -> std::sync::MutexGuard<'_, ()> {
        self.rises.lock().expect("the fake bank's charge rises")
    }

    /// The fake's tier is filled as it opens.
    fn wait_tier_filled(&self) -> Result<(), DomainError> {
        Ok(())
    }

    fn fill_epoch(&self) -> u64 {
        self.fills.load(std::sync::atomic::Ordering::Acquire)
    }

    /// The fake streams no layer.
    fn lines(&self) -> std::sync::Arc<[crate::assessment::BankLines]> {
        std::sync::Arc::from([])
    }

    /// The fake streams no layer, so no route reads a policy.
    fn device_convert(&self, _lines: &[crate::assessment::RecordLine]) {}

    #[cfg(test)]
    fn slow_host_landing(&self) {}

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
        *resize = Some(match resize.take().unwrap() {
            FakeResize::Prepared(prepared, (_, at), true) => FakeResize::Stopped(prepared.abandon(), at),
            other => other,
        });
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

    /// Device slots, rows of the bank's tensor, a row a step down to one
    /// and up to where its decode's misses stop falling (every row before
    /// it counted one), each saving what the price says a slot more saves;
    /// and host slots, a byte of upload storage each, up to the tier's
    /// ceiling, each saving [`HOST_SLOT_SAVES`], as `ServedBank::curves`
    /// steps them, each as far as `reach` goes.
    fn curves(&self, price: Option<&crate::assessment::DecodePrice>, reach: crate::Reach) -> Result<Vec<crate::BankCurve>, DomainError> {
        use crate::SizeStep;
        let rows = self.tensor.extents()[0];
        let unit = self.tensor.byte_len() / rows;
        let at = self.slots();
        let bound = match price {
            Some(_) => self.decode.as_ref().map_or(rows, |decode| rows.min(decode.saturation)).max(at),
            None => at,
        };
        let step = |slots: u64| price.map_or(0., |price| price.step(crate::assessment::HeldSlots::bank(slots as usize)).total_seconds);
        // Steps of `unit` bytes each, until they reach `bytes`.
        let within = |bytes: u64| bytes.div_ceil(unit.max(1)) as usize;
        let held = (1..at).rev().take(within(reach.held)).map(|to| SizeStep { to, bytes: unit, saves: step(to) - step(to + 1) }).collect();
        let offered = (at + 1..=bound).take(within(reach.offered)).map(|to| SizeStep { to, bytes: unit, saves: step(to - 1) - step(to) }).collect();
        let (bank, host) = (crate::planning::BankTier::Bank, crate::planning::BankTier::Host);
        let mut curves = vec![crate::BankCurve { holding: crate::BankHolding::Experts, kind: bank, at, held, offered }];
        if self.host_ceiling > 0 {
            let at = self.host.load(std::sync::atomic::Ordering::Relaxed);
            let bound = price.map_or(at, |_| self.host_ceiling.max(at));
            let offered = (at + 1..=bound).take(reach.offered as usize).map(|to| SizeStep { to, bytes: 1, saves: HOST_SLOT_SAVES }).collect();
            curves.push(crate::BankCurve { holding: crate::BankHolding::Host, kind: host, at, held: vec![], offered });
        }
        Ok(curves)
    }

    fn group_reads(
        &self,
    ) -> Result<Option<(crate::assessment::GroupReads, crate::assessment::BankSlots)>, DomainError> {
        Ok(None)
    }

    fn live(&self) -> Option<crate::assessment::LiveDecode> {
        self.decode.as_ref().map(|decode| decode.live.clone())
    }

    /// Its one streamed layer of 4 experts, each step routing 4, on the
    /// reference lines of 1 MB records, whatever lines the load priced.
    fn streamed_layers(&self, _lines: std::sync::Arc<[crate::assessment::ExpertLines]>) -> crate::assessment::StreamedLayers {
        crate::assessment::StreamedLayers::new(vec![4], 4, std::sync::Arc::from([crate::assessment::ExpertLines::reference(1_000_000)]))
    }

    fn grow(&self, holding: crate::BankHolding, to: u64, wake: crate::CompletionWake) -> Result<BankStep, DomainError> {
        if self.running() {
            return Ok(BankStep::Running);
        }
        if holding == crate::BankHolding::Host {
            let from = self.host.load(std::sync::atomic::Ordering::Relaxed);
            self.host.store(to, std::sync::atomic::Ordering::Relaxed);
            let now = std::time::Instant::now();
            let zero = std::time::Duration::ZERO;
            let times = super::ResizeTimes {
                decided: now,
                queued: zero,
                planned: zero,
                trips: Default::default(),
                moves: zero,
                prepared: now,
                installed: now,
                released: now,
            };
            let holding = crate::BankHolding::Host;
            *self.host_grown.lock().unwrap() = Some(BankResized::Grown { holding, from, to, grown: to - from, refused: None, times });
            wake.complete();
            return Ok(BankStep::Begun(crate::planning::BankTier::Host));
        }
        self.begin(to, wake)
    }

    /// The fake's growth backs its storage only as its advance installs it.
    fn growth_backed(&self, _: crate::planning::BankTier) -> Result<u64, DomainError> {
        Ok(0)
    }

    fn holding_storage(&self, holding: crate::BankHolding) -> Result<crate::planning::BankTier, DomainError> {
        Ok(match holding {
            crate::BankHolding::Host => crate::planning::BankTier::Host,
            crate::BankHolding::Experts | crate::BankHolding::Table { .. } => crate::planning::BankTier::Bank,
        })
    }

    fn advance(&self, wake: crate::CompletionWake) -> Result<Option<BankResized>, DomainError> {
        if let Some(grown) = self.host_grown.lock().unwrap().take() {
            wake.complete();
            return Ok(Some(grown));
        }
        let mut resize = self.resize.lock().unwrap();
        let (next, finished) = match resize.take().unwrap() {
            FakeResize::Prepared(prepared, before, grows) => match prepared.install() {
                Ok(release) => (FakeResize::Releasing(release, before), None),
                Err(refused) => (FakeResize::Refused(refused.resizer, refused.error, !grows), None),
            },
            // The release taken now, or already by the bank's own thread.
            FakeResize::Releasing(release, before) => {
                *resize = Some(FakeResize::Released(release.release(), before));
                drop(resize);
                return self.advance(wake);
            }
            FakeResize::Released(resizer, (charged, from)) => {
                let (slots, after) = (self.slots(), self.charged());
                let now = std::time::Instant::now();
                let zero = std::time::Duration::ZERO;
                let times = crate::domain::ResizeTimes {
                    decided: now,
                    queued: zero,
                    planned: zero,
                    trips: Default::default(),
                    moves: zero,
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
                (FakeResize::Idle(resizer), Some(finished))
            }
            FakeResize::Stopped(resizer, at) => {
                let stopped = BankResized::Stopped { holding: crate::BankHolding::Experts, at };
                (FakeResize::Idle(resizer), Some(stopped))
            }
            FakeResize::Refused(resizer, refusal, true) => {
                *resize = Some(FakeResize::Idle(resizer));
                return Err(DomainError::invariant(format!("the expert bank's storage refused a shrink: {refusal}")));
            }
            FakeResize::Refused(resizer, refusal, false) => {
                let at = self.slots();
                let refused = BankResized::GrowthRefused { holding: crate::BankHolding::Experts, at, refusal };
                (FakeResize::Idle(resizer), Some(refused))
            }
            idle => (idle, None),
        };
        *resize = Some(next);
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

    fn residency(&self) -> Result<crate::ExpertResidency, DomainError> {
        let experts = self.tensor.extents()[0];
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
                crate::HostResidency { slots, ceiling: ceiling.get(), bytes: slots, fill: crate::TierFill::Filled, last_resize }
            }),
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
}

impl TestFamily {
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
    fn bind_head(&mut self, _: crate::ResidentHead, _: &ModelDefinition) -> Result<(), String> {
        Ok(())
    }
    fn bind_vision(&mut self, _: crate::ResidentVision, _: &ModelDefinition) -> Result<(), String> {
        Ok(())
    }
    fn head_is_bound(&self) -> bool {
        false
    }
    fn graph_replays(&self) -> seismic::GraphReplays {
        seismic::GraphReplays::default()
    }
    fn vision_is_bound(&self) -> bool {
        false
    }
    fn state_is_bound(&self) -> bool {
        true
    }
    fn unbind_optional(&mut self, _: super::OptionalComponent) {}

    fn target_weight_bytes(&self) -> Result<u64, &'static str> {
        self.bank.as_ref().map_or(Ok(0), |bank| {
            bank.tensor
                .observe_storage()
                .charged_bytes()
                .ok_or("the fake bank is charged while it lives")
        })
    }

    fn optional_constant_bytes(&self) -> Result<u64, &'static str> {
        Ok(self.optional.as_ref().map_or(0, seismic::Tensor::charged_bytes))
    }

    fn expert_bank(&self) -> Option<&dyn ElasticBank> {
        self.bank.as_ref().map(|bank| bank as &dyn ElasticBank)
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
