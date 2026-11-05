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
/// takes it back.
struct StillHost {
    device: Rc<seismic::Device>,
    topology: std::sync::Arc<seismic::DeviceTopology>,
    first: seismic::Status,
    charged: BTreeMap<seismic::StoreId, u64>,
}

impl StillHost {
    fn of(catalog: seismic::Devices, device: Rc<seismic::Device>) -> Self {
        let first = catalog.status(device.topology(), device.info().id());
        let charged = Self::charges(&device);
        Self { device, topology: catalog.topology(), first, charged }
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
        // reading.
        let moved = |left: u64, store| (left + charged(&self.charged, store)).saturating_sub(charged(&now, store));
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
        }
        for (path, grant) in sample.grants.iter_mut().chain(runtime.grants.iter_mut()) {
            if let Ok(grant) = grant {
                let store = topology.access_path(*path).expect("a read grant's path belongs to its topology").store();
                grant.value = moved(grant.value, store);
            }
        }
        seismic::Status { sample, runtime }
    }

    /// A still host reads the same however it is read.
    fn observed(&self, device: &seismic::Device) -> seismic::Status {
        self.status(device)
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
    let reserves = platform::MemoryReserves::standard();
    let selected = platform::select_device(&catalog, ExecutionPath::Native, device_request, &reserves)
        .unwrap_or_else(|error| panic!("{device_request} selects a device: {error}"));
    let device = Rc::new(
        catalog
            .open(catalog.resolve(selected.info.selector()).unwrap())
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
    ExecutorDomain::with_family(
        execution,
        definition,
        resources,
        DeviceHeap::open(Rc::new(StillHost::of(catalog, device.clone())), reserves, device).unwrap(),
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
    let static_bytes = domain.static_holding_bytes().unwrap();
    domain.static_holding = Some(
        domain
            .memory
            .borrow_mut()
            .heap_mut()
            .insert(static_bytes, HoldingClass::Model)
            .unwrap(),
    );
    domain.refresh_memory().unwrap();
    (domain, bindings)
}

/// A miss record of `steps` decode steps at one streamed layer of 4
/// experts, each step reading `misses` experts `depth` deep in the bank's
/// LRU stack, along the reference lines (`ExpertLines::reference`): a bank
/// of `depth` slots misses them, one of `depth + 1` none. Reads take 0.4 ms
/// and 0.5 ms a MB; an all-hit step 15 ms.
pub fn misses_at(depth: usize, steps: u64, misses: u64) -> crate::assessment::MissRecord {
    let mut reads = crate::assessment::RequestReads::default();
    for bytes in [1_000_000u64, 2_000_000] {
        reads.add(bytes, 4e-4 + bytes as f64 / 2e9);
    }
    reads.deliver(2_000_000_000, 1.);
    let mut curve = crate::experts::LayerCurve {
        misses: vec![0; 5],
        requests: vec![0; 5],
        choices: 0,
    };
    curve.misses[depth] = steps * misses;
    curve.requests[depth] = steps;
    // A row a read.
    curve.choices = steps * misses;
    crate::assessment::MissRecord {
        steps,
        rows: steps,
        layers: vec![crate::assessment::MeasuredLayer {
            lines: crate::assessment::ExpertLines::reference(1_000_000),
            curve,
            anticipated: crate::assessment::Anticipated::default(),
        }],
        reads,
        // One-row launches of 15 ms that miss nothing.
        residuals: crate::assessment::Residuals {
            rows: std::array::from_fn(|index| crate::assessment::LaunchResiduals {
                launches: if index == 0 { steps } else { 0 },
                seconds: if index == 0 { steps as f64 * 0.015 } else { 0. },
                tokens: if index == 0 { steps * 100 } else { 0 },
            }),
        },
        hints: crate::assessment::HintCounts::default(),
    }
}

/// The bytes of a row of `domain`'s fake bank ([`banked`]).
pub fn bank_row(domain: &ExecutorDomain<TestFamily>) -> u64 {
    domain.family.bank.as_ref().expect("a banked fixture").tensor.extents()[1]
}

/// The fake bank of `domain` takes `record` as its load's latest miss
/// record, which its growth is priced by.
pub fn record_misses(domain: &mut ExecutorDomain<TestFamily>, record: crate::assessment::MissRecord) {
    domain.family.bank.as_mut().expect("a banked fixture").record = Some(record);
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
}

/// A fake elastic expert bank (`ElasticBank`): one reserved tensor of a row
/// a slot, its floor one row. A resize prepares at once and finishes over
/// two boundaries, as the bank's does: the first installs, the second takes
/// the release (a growth, as one slot that joins beside running steps). Its
/// growth is priced by `record`. As the bank's watcher, it is asked to
/// shrink only in a gap: a shrink begins only while none of its `steps`
/// runs.
pub struct FakeBank {
    pub(crate) tensor: seismic::Tensor,
    resize: std::sync::Mutex<Option<FakeResize>>,
    pub(crate) record: Option<crate::assessment::MissRecord>,
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
    /// The device bytes it can still grow by to its ceiling
    /// ([`ElasticBank::growth_shortfall`]), a test's: none unless set.
    pub(crate) shortfall: u64,
    /// Whether it holds every expert ([`ElasticBank::whole`]), a test's:
    /// whole unless set; a fake at a weight cap's ceiling is not.
    pub(crate) whole: bool,
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
    Refused(seismic::Resizer, seismic::TensorError),
    /// A growth stopped before its first slot, at these slots, before an
    /// advance reports it.
    Stopped(seismic::Resizer, u64),
}

impl FakeBank {
    pub fn new(tensor: seismic::Tensor, resizer: seismic::Resizer) -> Self {
        Self {
            tensor,
            resize: std::sync::Mutex::new(Some(FakeResize::Idle(resizer))),
            record: None,
            steps: std::sync::atomic::AtomicUsize::new(0),
            host: std::sync::atomic::AtomicU64::new(0),
            host_ceiling: 0,
            host_grown: std::sync::Mutex::new(None),
            last_resize: std::sync::Mutex::new(None),
            host_last_resize: std::sync::Mutex::new(None),
            rises: std::sync::Mutex::new(()),
            shortfall: 0,
            whole: true,
        }
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
            Err(refusal) => FakeResize::Refused(refusal.resizer, refusal.error),
        });
        wake.complete();
        Ok(BankStep::Begun)
    }
}

impl ElasticBank for FakeBank {
    fn slots(&self) -> u64 {
        self.tensor.committed_rows()
    }

    fn hold_rises(&self) -> std::sync::MutexGuard<'_, ()> {
        self.rises.lock().expect("the fake bank's charge rises")
    }

    /// What the test set ([`FakeBank::shortfall`]): its growth is the
    /// test's.
    fn growth_shortfall(&self) -> Result<u64, DomainError> {
        Ok(self.shortfall)
    }

    fn whole(&self) -> bool {
        self.whole
    }

    /// The fake's tier is filled as it opens.
    fn wait_tier_filled(&self) -> Result<(), DomainError> {
        Ok(())
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

    fn tier_copy(&self, _costs: &crate::assessment::TierCopyCosts) -> Result<(), DomainError> {
        Ok(())
    }

    fn price(&self, _lines: Option<std::sync::Arc<[crate::assessment::ExpertLines]>>, _pass: Option<crate::assessment::ExpertPass>) {}

    fn shrink(
        &self,
        _resident: Option<&crate::ResidentStep>,
        _moves: Option<&crate::assessment::MoveLines>,
        _deficit: &dyn Fn(crate::planning::BankTier) -> u64,
        between_flights: &mut dyn FnMut() -> Result<(), DomainError>,
        wake: crate::CompletionWake,
    ) -> Result<BankStep, DomainError> {
        if self.running() {
            return Ok(BankStep::Running);
        }
        if self.slots() <= 1 {
            return Ok(BankStep::AtLimit);
        }
        between_flights()?;
        if !self.in_gap() {
            return Ok(BankStep::Deferred);
        }
        self.begin(self.slots() - 1, wake)
    }

    fn carried(&self) -> Result<(), DomainError> {
        Ok(())
    }

    /// Device slots, rows of the bank's tensor, to the target within `room`
    /// up to where the record's misses stop falling (`growth_target`), each
    /// saving what as many slots more of the bank save; or host slots, a
    /// byte of upload storage each, toward the tier's ceiling within `room`,
    /// priced by nothing, as `ServedBank::growth` steps them.
    fn growth(
        &self,
        resident: Option<&crate::ResidentStep>,
        _measured: &dyn Fn(crate::BankHolding) -> bool,
        kind: crate::planning::BankTier,
        room: u64,
    ) -> Result<Option<crate::Growth>, DomainError> {
        let slots = self.slots();
        let step = |from, bound, unit: u64| {
            let to = crate::planning::growth_target(from, bound, room, |to| Ok::<_, ()>((to - from) * unit)).unwrap()?;
            Some((from, to, (to - from) * unit))
        };
        if kind != crate::planning::BankTier::Bank {
            let (holding, bound) = (crate::BankHolding::Host, self.host_ceiling);
            let host = self.host.load(std::sync::atomic::Ordering::Relaxed);
            return Ok(step(host, bound, 1)
                .map(|(from, to, bytes)| crate::Growth { holding, kind, from, to, bytes, saving: None, justified: bound }));
        }
        let (Some(record), Some(resident)) = (&self.record, resident) else {
            return Ok(None);
        };
        let rows = self.tensor.extents()[0];
        let bound = rows.min(record.saturation().unwrap() as u64);
        // A step's saving over the record's decode, as `ServedBank::growth`
        // prices it.
        let held = crate::assessment::HeldSlots::bank(slots as usize);
        Ok(step(slots, bound, self.tensor.byte_len() / rows).map(|(from, to, bytes)| {
            let above = crate::assessment::HeldSlots::bank((slots + to - from) as usize);
            crate::Growth {
                holding: crate::BankHolding::Experts,
                kind,
                from,
                to,
                bytes,
                saving: Some(record.saving(held, above, resident).unwrap().unwrap() * record.steps as f64),
                justified: bound,
            }
        }))
    }

    fn group_reads(
        &self,
    ) -> Result<Option<(crate::assessment::GroupReads, crate::assessment::BankSlots)>, DomainError> {
        Ok(None)
    }

    fn latest_record(&self) -> Option<std::sync::Arc<crate::assessment::MissRecord>> {
        self.record.clone().map(std::sync::Arc::new)
    }

    fn grow(
        &self,
        resident: Option<&crate::ResidentStep>,
        measured: &dyn Fn(crate::BankHolding) -> bool,
        kind: crate::planning::BankTier,
        room: u64,
        wake: crate::CompletionWake,
    ) -> Result<BankStep, DomainError> {
        if self.running() {
            return Ok(BankStep::Running);
        }
        let Some(crate::Growth { to, .. }) = self.growth(resident, measured, kind, room)? else {
            return Ok(BankStep::AtLimit);
        };
        if kind == crate::planning::BankTier::Host {
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
            return Ok(BankStep::Begun);
        }
        self.begin(to, wake)
    }

    fn advance(&self, wake: crate::CompletionWake) -> Result<Option<BankResized>, DomainError> {
        if let Some(grown) = self.host_grown.lock().unwrap().take() {
            wake.complete();
            return Ok(Some(grown));
        }
        let mut resize = self.resize.lock().unwrap();
        let (next, finished) = match resize.take().unwrap() {
            FakeResize::Prepared(prepared, before, _) => match prepared.install() {
                Ok(release) => (FakeResize::Releasing(release, before), None),
                Err(refused) => (FakeResize::Refused(refused.resizer, refused.error), None),
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
            FakeResize::Refused(resizer, refusal) => {
                let at = self.slots();
                let refused = BankResized::Refused { holding: crate::BankHolding::Experts, at, refusal };
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
                crate::HostResidency { slots, ceiling, bytes: slots, fill: crate::TierFill::Filled, last_resize }
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
    fn unbind_optional(&mut self) {}

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
    ) -> Result<Chain<Self::TargetSubmission>, crate::programs::ChainRefusal>
    {
        if REJECTED.with(|rejected| rejected.get()) == Some(0) {
            REJECTED.with(|rejected| rejected.set(Some(chain.len())));
            let error = crate::DeviceError::Execution("the fixture rejects the chain".into());
            return Err(crate::programs::ChainRefusal {
                error: crate::SubmitError::Device(error),
                chain,
                misses: None,
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
