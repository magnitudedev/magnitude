//! Controlled completion tests for the production execution domain.

use super::*;
use crate::{
    AttestedPrograms, ComponentSelection, ExecutionPath, ExecutionPlanner, PlannedMethod,
    ResourceAllocator, ResourceCapacity, ResourceLimits, ResourcePlanner, TargetLaunchCore,
    completion::CompletionWake,
    platform,
    programs::{
        CompletedHeadWork, CompletedVisionWork, CompletedWork, ProgramSubmission, ReadySubmission,
    },
};
use magnitude_artifacts::PackageManifest;
use magnitude_family_contracts::FamilyId;
use magnitude_state::{KvCodec, ShrinkPolicy};
use std::marker::PhantomData;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

/// The planning fixture's model (the smallest geometry every native kernel
/// admits) with an 8-token context.
fn tiny_definition() -> ModelDefinition {
    let mut definition = crate::planning::tests::fixture_definition();
    definition.family = FamilyId("controlled-target-domain".into());
    definition.decoder.context_limit = 8;
    definition
}

fn tiny_manifest(definition: &ModelDefinition) -> PackageManifest {
    crate::planning::tests::fixture_manifest(definition)
}

type Fixture = (ExecutorDomain<TestFamily>, StateBindings<TestFamily>);

thread_local! {
    /// The stored-table reads the next target submission's output carries:
    /// a test sets them to inject a step's table reads.
    static STORED_READS: std::cell::RefCell<Vec<crate::programs::StoredRead>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

fn fixture(control: Option<PendingControl>) -> Option<Fixture> {
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

fn fixture_with(control: Option<PendingControl>, lookahead: bool, head: bool) -> Option<Fixture> {
    fixture_on(control, lookahead, head, platform::DeviceRequest::Automatic)
}

fn fixture_on(
    control: Option<PendingControl>,
    lookahead: bool,
    head: bool,
    device_request: platform::DeviceRequest,
) -> Option<Fixture> {
    fixture_of(tiny_definition(), 2, control, lookahead, head, device_request)
}

/// A domain over `definition` launching up to `rows` rows.
fn fixture_of(
    definition: ModelDefinition,
    rows: usize,
    control: Option<PendingControl>,
    lookahead: bool,
    head: bool,
    device_request: platform::DeviceRequest,
) -> Option<Fixture> {
    let catalog = seismic::Devices::discover(&[]).ok()?;
    let reserves = platform::MemoryReserves::standard();
    let selected =
        platform::select_device(&catalog, ExecutionPath::Native, device_request, &reserves).ok()?;
    let device = Rc::new(
        catalog
            .open(catalog.resolve(selected.info.selector()).unwrap())
            .unwrap(),
    );
    let definition = Rc::new(definition);
    let manifest = tiny_manifest(&definition);
    let limits = ResourceLimits {
        max_launch_rows: rows,
        max_launch_slots: 2,
        max_selected_rows: 2,
        max_drafting_slots: 2,
        exported_logits_rows: 0,
        max_images_per_request: magnitude_artifacts::MAX_IMAGES_PER_REQUEST,
        max_image_cells: 0,
        lookahead,
    };
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
        device.backend(),
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
    Some(ExecutorDomain::with_family(
        execution,
        definition,
        resources,
        DeviceHeap::open(Rc::new(catalog), reserves, device).unwrap(),
        store,
        head,
        None,
        None,
        TestFamily { control, programs },
    ))
}

fn forward(request: RequestId, position: usize) -> Operation {
    Operation::Forward {
        request,
        kind: WorkKind::Replay,
        tokens: vec![crate::TokenId(1)],
        position,
        conditioning: None,
        demand: crate::batching::Demand::NONE,
        select: Vec::new(),
        committed: 1,
        prime: None,
    }
}

/// Make `request` resident on fresh state with no prompt rows.
fn open<F: ProgramFamily>(
    domain: &mut ExecutorDomain<F>,
    bindings: &mut StateBindings<F>,
    request: RequestId,
) {
    domain
        .install_input(request, PreparedModelInput::continuation_only())
        .unwrap();
    assert!(
        domain
            .open_state(bindings, request, None)
            .unwrap()
            .is_empty()
    );
}

#[derive(Default)]
struct PendingState {
    result: Option<Result<(), crate::DeviceError>>,
    wake: Option<CompletionWake>,
}
#[derive(Clone, Default)]
struct PendingControl(Arc<Mutex<PendingState>>);
impl PendingControl {
    fn resolve(&self, result: Result<(), crate::DeviceError>) {
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
struct PendingCompletion(PendingControl);
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
struct PendingSubmission<L, W, O> {
    completion: PendingCompletion,
    core: L,
    workspace: W,
    output: O,
}

/// A controlled pending submission for lane-level device-failure tests. It has
/// no completed payload because the error resolves before `finish` exposes work.
struct PendingFailureSubmission<W> {
    completion: PendingCompletion,
    _completed: PhantomData<fn() -> W>,
}

impl<W> PendingFailureSubmission<W> {
    fn new(control: PendingControl) -> Self {
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

enum ControlledHeadSubmission {
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

enum TestSubmission<L, W, O> {
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
struct TestFamily {
    control: Option<PendingControl>,
    programs: AttestedPrograms,
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
    fn recorded_graphs(&self) -> u64 {
        0
    }
    fn vision_is_bound(&self) -> bool {
        false
    }
    fn state_is_bound(&self) -> bool {
        true
    }
    fn unbind_optional(&mut self) {}

    fn prepared_program_bytes(&self) -> Result<u64, &'static str> {
        self.programs.device_storage_bytes()
    }
    fn submit_target(
        &mut self,
        launch: ValidatedTargetLaunch,
    ) -> Result<Self::TargetSubmission, (crate::SubmitError, ValidatedTargetLaunch)> {
        let (core, _) = launch.into_submission_parts();
        let now = std::time::Instant::now();
        Ok(self.submit(
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
                stored_reads: STORED_READS.with(|reads| std::mem::take(&mut *reads.borrow_mut())),
            },
        ))
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

/// Reserve and submit one target group; the flight takes the bindings.
fn submit_reserved_target<F: ProgramFamily>(
    domain: &mut ExecutorDomain<F>,
    mut bindings: StateBindings<F>,
    operations: Vec<Operation>,
) -> TargetFlight<F> {
    let resources = domain
        .reserve(&mut bindings, &operations)
        .unwrap()
        .into_resources();
    let ReservedResources::Target(reservation) = resources else {
        panic!("target operation produced another reservation lane")
    };
    domain
        .submit_target(bindings, &operations, reservation)
        .unwrap_or_else(|failure| panic!("target submission failed: {}", failure.error()))
}

/// Finish a target flight with one outcome.
fn finish_one<F: ProgramFamily>(
    domain: &mut ExecutorDomain<F>,
    flight: TargetFlight<F>,
) -> (PendingOperationOutcome, StateBindings<F>) {
    let (mut pending, bindings) = domain.finish_target(flight).unwrap();
    (pending.pop().unwrap(), bindings)
}

fn prefill(request: RequestId, position: usize) -> Operation {
    Operation::Forward {
        request,
        kind: WorkKind::Prefill,
        tokens: vec![crate::TokenId(1)],
        position,
        conditioning: None,
        demand: crate::batching::Demand::NONE,
        select: Vec::new(),
        committed: 1,
        prime: None,
    }
}

/// Submit and reconcile a prefill chunk at 0 whose planned next chunk is
/// queued behind it: the returned bindings hold that lookahead.
fn queue_lookahead(
    domain: &mut ExecutorDomain<TestFamily>,
    bindings: StateBindings<TestFamily>,
    request: RequestId,
) -> StateBindings<TestFamily> {
    domain.plan_successors(vec![prefill(request, 1)]);
    let flight = submit_reserved_target(domain, bindings, vec![prefill(request, 0)]);
    let (pending, bindings) = finish_one(domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    assert!(bindings.lookahead.is_some(), "the next chunk is queued");
    bindings
}

#[test]
fn ready_target_can_abort_then_reconcile() {
    let Some((mut domain, mut bindings)) = fixture(None) else {
        return;
    };
    let request = RequestId(1);
    open(&mut domain, &mut bindings, request);
    let mut flight = submit_reserved_target(&mut domain, bindings, vec![forward(request, 0)]);
    assert!(flight.completion().is_complete());
    assert!(domain.resume_state(request).is_err());
    let (pending, bindings) = finish_one(&mut domain, flight);
    domain.abort(pending).unwrap();
    assert_eq!(domain.resume_state(request).unwrap().target.position(), 0);
    let flight = submit_reserved_target(&mut domain, bindings, vec![forward(request, 0)]);
    let (pending, _) = finish_one(&mut domain, flight);
    assert!(
        domain
            .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
            .is_ok()
    );
    assert_eq!(domain.resume_state(request).unwrap().target.position(), 1);
}

/// A launch is charged the host work since the flight before it finished
/// when that flight carried its request: a round's host time is part of its
/// cost. Time the owner spent idle, or a flight of other requests, is
/// charged to no launch.
#[test]
fn a_launch_is_charged_its_requests_host_turn_but_not_idle_time() {
    let Some((mut domain, mut bindings)) = fixture(None) else {
        return;
    };
    let gap = std::time::Duration::from_millis(60);
    let (first, second) = (RequestId(70), RequestId(71));
    open(&mut domain, &mut bindings, first);
    open(&mut domain, &mut bindings, second);
    let step = |domain: &mut ExecutorDomain<TestFamily>,
                bindings: StateBindings<TestFamily>,
                request: RequestId,
                position: usize| {
        let flight = submit_reserved_target(domain, bindings, vec![forward(request, position)]);
        let (pending, bindings) = finish_one(domain, flight);
        let duration = pending.physical_duration();
        domain
            .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
            .unwrap();
        (duration, bindings)
    };
    let (_, bindings) = step(&mut domain, bindings, first, 0);
    std::thread::sleep(gap);
    let (charged, bindings) = step(&mut domain, bindings, first, 1);
    assert!(charged >= gap, "the host turn of its own request: {charged:?}");
    std::thread::sleep(gap);
    let (other, bindings) = step(&mut domain, bindings, second, 0);
    assert!(other < gap, "another request's turn: {other:?}");
    std::thread::sleep(gap);
    domain.host_idle();
    let (idle, _) = step(&mut domain, bindings, second, 1);
    assert!(idle < gap, "after the owner idled: {idle:?}");
}

/// A request that stops continuing its queued step discards it, and waiting
/// for that step is part of its next launch: the switch away from plain
/// decoding costs the round that makes it.
#[test]
fn a_launch_that_discards_its_requests_lookahead_is_charged_the_wait() {
    let Some((mut domain, mut bindings)) = fixture_with(None, true, true) else {
        return;
    };
    let gap = std::time::Duration::from_millis(60);
    let request = RequestId(72);
    open(&mut domain, &mut bindings, request);
    let bindings = queue_lookahead(&mut domain, bindings, request);
    std::thread::sleep(gap);
    // A replay row is not the queued prefill step: submitting it discards
    // that step first.
    let flight = submit_reserved_target(&mut domain, bindings, vec![forward(request, 1)]);
    let (pending, bindings) = finish_one(&mut domain, flight);
    assert!(bindings.lookahead.is_none(), "the queued step was discarded");
    assert!(
        pending.physical_duration() >= gap,
        "the wait and the host turn before it: {:?}",
        pending.physical_duration()
    );
}

/// Growth changes bindings: every entry that can grow resolves a queued
/// lookahead first, then commits the slab.
#[test]
fn growth_orphans_a_queued_lookahead_then_grows() {
    let Some((mut domain, mut bindings)) = fixture_with(None, true, true) else {
        return;
    };
    let request = RequestId(60);
    open(&mut domain, &mut bindings, request);
    let mut bindings = queue_lookahead(&mut domain, bindings, request);
    let head = domain.head_store.clone().unwrap();
    let history = head.sole_history_domain().unwrap();
    let committed = head.committed_rows(history);
    let demand = [RowDemand {
        domain: history,
        rows: head.free_rows(history) + 1,
    }];
    domain.provision_open(&mut bindings).unwrap();
    assert!(
        bindings.lookahead.is_none(),
        "the growing entry orphaned the lookahead"
    );
    domain
        .grant_state_growth(&mut bindings, true, &demand, 0)
        .unwrap();
    assert!(head.committed_rows(history) > committed);
    // The orphaned step changed no accepted state.
    assert_eq!(domain.resume_state(request).unwrap().target.position(), 1);
}

/// A queued lookahead reserves the rows after its request's history, so that
/// history's last page reads as blocked by another. Drafter work for the same
/// request, which does not claim the lookahead, must resolve it before any
/// layout question: it provisions without relocating, and the lookahead is
/// gone. (Regression: relocation planned against the lookahead's rows, then
/// orphaned it and failed with "only a blocked last page is relocated".)
#[test]
fn drafter_work_resolves_its_requests_lookahead_before_provisioning() {
    let Some((mut domain, mut bindings)) = fixture_with(None, true, true) else {
        return;
    };
    let request = RequestId(61);
    open(&mut domain, &mut bindings, request);
    let mut bindings = queue_lookahead(&mut domain, bindings, request);
    assert!(
        !domain.target[&request].blocked_tails().is_empty(),
        "the lookahead's rows read as another history's"
    );
    let draft = Operation::Head {
        request,
        phase: crate::HeadPhase::Generation,
        tokens: vec![crate::TokenId(1)],
        conditioning: crate::FeatureRows::new(vec![0u8; 4].into(), 1).unwrap(),
        position: 1,
        proposals: Vec::new(),
        form: crate::DraftForm::Chained,
    };
    // The fixture has no drafter weights, so provisioning ends at loading the
    // head component: after the lookahead and every relocation are resolved.
    match domain.provision(&mut bindings, &[draft], None) {
        Ok(()) => {}
        Err(DomainError::Input(detail)) if detail == "head component is disabled" => {}
        Err(error) => panic!("provisioning drafter work failed: {error}"),
    }
    assert!(
        bindings.lookahead.is_none(),
        "drafter work resolved the lookahead"
    );
    assert!(domain.target[&request].blocked_tails().is_empty());
}

/// A history's next rows in its partly used last page need no free page: a
/// store whose every page is referenced still reserves them.
#[test]
fn in_place_rows_reserve_without_a_free_page() {
    let Some((mut domain, mut bindings)) = fixture(None) else {
        return;
    };
    let request = RequestId(10);
    open(&mut domain, &mut bindings, request);
    let accept =
        |domain: &mut ExecutorDomain<TestFamily>, bindings: StateBindings<TestFamily>, position| {
            let flight = submit_reserved_target(domain, bindings, vec![forward(request, position)]);
            let (pending, bindings) = finish_one(domain, flight);
            domain
                .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
                .unwrap();
            bindings
        };
    let mut bindings = accept(&mut domain, bindings, 0);
    // Hold every other page of the backed slabs with one-row histories.
    let store = domain.target_store.clone();
    let mut held = Vec::new();
    while store
        .history_domains()
        .any(|history| store.free_rows(history) != 0)
    {
        let advance = OwnedStateAdvance::begin(store.create().unwrap(), 1)
            .ok()
            .unwrap();
        let OwnedAdvanceResolution::Committed(filler) = advance.commit_all().ok().unwrap() else {
            panic!("a one-row advance commits");
        };
        held.push(filler);
    }
    for position in 1..4 {
        bindings = accept(&mut domain, bindings, position);
    }
    assert!(
        store
            .history_domains()
            .all(|history| store.free_rows(history) == 0)
    );
    assert_eq!(domain.resume_state(request).unwrap().target.position(), 4);
}

/// Continuing a checkpoint relocates a tail another request extended. The
/// reservation must describe the new layout rather than demand a second page.
#[test]
fn reservation_recomputes_row_demand_after_shared_tail_relocation() {
    let Some((mut domain, mut bindings)) = fixture(None) else {
        return;
    };
    let source = RequestId(71);
    open(&mut domain, &mut bindings, source);
    let flight = submit_reserved_target(&mut domain, bindings, vec![forward(source, 0)]);
    let (pending, mut bindings) = finish_one(&mut domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    let checkpoint = domain.resume_state(source).unwrap();
    let flight = submit_reserved_target(&mut domain, bindings, vec![forward(source, 1)]);
    let (pending, returned) = finish_one(&mut domain, flight);
    bindings = returned;
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    let copy = RequestId(72);
    domain
        .install_input(copy, PreparedModelInput::continuation_only())
        .unwrap();
    domain
        .open_state(&mut bindings, copy, Some(&checkpoint))
        .unwrap();
    assert!(!domain.target[&copy].blocked_tails().is_empty());
    let operations = [forward(copy, 1)];
    let before = domain.requirements(&bindings, &operations).unwrap();
    assert!(before.state_demand.iter().any(|demand| demand.rows > 0));
    let reservation = domain.reserve(&mut bindings, &operations).unwrap();
    assert!(
        reservation
            .requirements
            .state_demand
            .iter()
            .all(|demand| demand.rows == 0)
    );
}

/// A cancelled completed request aborts and the bindings return; a device
/// failure is returned as an error that consumes them.
#[test]
fn pending_target_request_cancellation_aborts_then_device_failure_consumes_the_bindings() {
    let control = PendingControl::default();
    let Some((mut domain, mut bindings)) = fixture(Some(control.clone())) else {
        return;
    };
    let request = RequestId(2);
    open(&mut domain, &mut bindings, request);
    let mut flight = submit_reserved_target(&mut domain, bindings, vec![forward(request, 0)]);
    assert!(!flight.completion().is_complete());
    assert!(domain.resume_state(request).is_err());
    let woke = Arc::new(AtomicBool::new(false));
    let observed = woke.clone();
    flight.completion().notify(CompletionWake::new(move || {
        observed.store(true, Ordering::SeqCst)
    }));
    assert!(!woke.load(Ordering::SeqCst));
    control.resolve(Ok(()));
    assert!(woke.load(Ordering::SeqCst));
    let (pending, _bindings) = finish_one(&mut domain, flight);
    domain.abort(pending).unwrap();
    assert_eq!(domain.resume_state(request).unwrap().target.position(), 0);
    let failed = PendingControl::default();
    let Some((mut failed_domain, mut bindings)) = fixture(Some(failed.clone())) else {
        return;
    };
    let failed_request = RequestId(3);
    open(&mut failed_domain, &mut bindings, failed_request);
    let flight = submit_reserved_target(
        &mut failed_domain,
        bindings,
        vec![forward(failed_request, 0)],
    );
    failed.resolve(Err(crate::DeviceError::Execution(
        "injected target failure".into(),
    )));
    assert!(matches!(
        failed_domain.finish_target(flight),
        Err(DomainError::Device(_))
    ));
    assert!(failed_domain.resume_state(failed_request).is_err());
}

#[test]
fn pending_target_state_is_classified_as_in_flight() {
    let control = PendingControl::default();
    let Some((mut domain, mut bindings)) = fixture(Some(control.clone())) else {
        return;
    };
    let request = RequestId(9);
    open(&mut domain, &mut bindings, request);
    let before = domain.reconcile_memory_charge(&[]).unwrap();
    assert_eq!(before.unattributed, 0, "{before:?}");
    let flight = submit_reserved_target(&mut domain, bindings, vec![forward(request, 0)]);
    let pending = domain.reconcile_memory_charge(&[]).unwrap();
    assert!(pending.target_state.in_flight > before.target_state.in_flight);
    assert_eq!(pending.unattributed, 0);
    control.resolve(Ok(()));
    for outcome in domain.finish_target(flight).unwrap().0 {
        domain.abort(outcome).unwrap();
    }
}

#[test]
fn fresh_memory_observation_reconciles_released_state_backing() {
    let Some((mut domain, mut bindings)) = fixture(None) else {
        return;
    };
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
    let before = domain.memory().standing().unwrap();
    assert_eq!(before.unattributed_bytes, 0);
    let released = domain
        .shrink_state(&mut bindings, ShrinkPolicy::Reclaim)
        .unwrap();
    assert!(released > 0);
    domain.refresh_memory().unwrap();
    let after = domain.memory().standing().unwrap();
    assert!(before.classified_bytes - after.classified_bytes >= released);
    assert_eq!(after.unattributed_bytes, 0);
}

/// A failed head or vision flight returns its device error, and its
/// bindings go with it.
#[test]
fn pending_head_and_vision_device_failures_consume_the_bindings() {
    let failure = |lane: &'static str| {
        crate::DeviceError::Execution(format!("failed pending {lane} submission"))
    };

    let Some((mut head_domain, bindings)) = fixture(None) else {
        return;
    };
    let head_control = PendingControl::default();
    let mut head = HeadFlight {
        requests: Vec::new(),
        steps: 0,
        submission: PendingFailureSubmission::<CompletedHeadWork>::new(head_control.clone()).into(),
        started: Instant::now(),
        recorded_graphs: 0,
        launch_trace: None,
        bindings,
    };
    assert!(!head.completion().is_complete());
    head_control.resolve(Err(failure("head")));
    assert!(matches!(
        head_domain.finish_head(head),
        Err(DomainError::Device(_))
    ));

    let Some((mut vision_domain, bindings)) = fixture(None) else {
        return;
    };
    let vision_control = PendingControl::default();
    let mut vision = VisionFlight {
        request: RequestId(41),
        image: ImageRef::new(vision_domain.resource_identity().clone(), 1).unwrap(),
        submission: PendingFailureSubmission::<CompletedVisionWork>::new(vision_control.clone()),
        started: Instant::now(),
        recorded_graphs: 0,
        bindings,
    };
    assert!(!vision.completion().is_complete());
    vision_control.resolve(Err(failure("vision")));
    assert!(matches!(
        vision_domain.finish_vision(vision),
        Err(DomainError::Device(_))
    ));
}

#[test]
fn completed_head_and_vision_request_cancellation_restores_or_drops() {
    let Some((mut domain, mut bindings)) = fixture(None) else {
        return;
    };
    let request = RequestId(50);
    open(&mut domain, &mut bindings, request);

    // A head advance is independent from the accepted target lane. The target
    // state remaining resident must not be mistaken for a conflicting owner.
    let head_source = domain.target.get(&request).unwrap().checkpoint().fork();
    let head_advance = OwnedStateAdvance::begin(head_source, 1).ok().unwrap();
    domain
        .abort(PendingOperationOutcome {
            request,
            outcome: Outcome::Head {
                proposals: Vec::new(),
            },
            advance: Some(head_advance),
            primed: None,
            rows: 1,
            committed_rows: 0,
            kind: WorkKind::Decode,
            physical_duration: Duration::ZERO,
            replayed: true,
            image: None,
        })
        .unwrap();
    assert_eq!(domain.target.get(&request).unwrap().position(), 0);
    assert_eq!(domain.head.remove(&request).unwrap().position(), 0);

    // Vision output is stateless until reconciliation installs it. Cancelling
    // the request simply drops the retained feature claim.
    let image = ImageRef::new(domain.resource_identity().clone(), 1).unwrap();
    let vision_features = FeatureRef::logical(
        domain.resource_identity().clone(),
        1,
        domain.definition.decoder.hidden as usize,
    )
    .unwrap();
    domain
        .abort(PendingOperationOutcome {
            request,
            outcome: Outcome::Encode {
                features: vision_features,
            },
            advance: None,
            primed: None,
            rows: 0,
            committed_rows: 0,
            kind: WorkKind::Prefill,
            physical_duration: Duration::ZERO,
            replayed: true,
            image: Some(image),
        })
        .unwrap();
    assert_eq!(domain.target.get(&request).unwrap().position(), 0);
}

/// The tiny model with a minimal vision description: 2 x 2 patches of two
/// frames (24-wide patch rows), 2 x 2 cells, a 4 x 4 position table. Only
/// input preparation reads it; the domain encodes nothing here.
fn vision_definition() -> ModelDefinition {
    use magnitude_family_contracts::{
        ActivationDType, CellReduction, PositionSampling, VisionDescription, VisionMerger,
        VisionPositions, VisionPreprocessing, VisionResampling, VisionResize, VisionStem,
        WeightDescriptor,
    };
    let stored = |name: &str, shape: &[u64]| WeightDescriptor::stored(name, shape);
    let mut definition = tiny_definition();
    definition.vision = Some(VisionDescription {
        activation_dtype: ActivationDType::BF16,
        hidden: 128,
        output_hidden: definition.decoder.hidden,
        preprocessing: VisionPreprocessing {
            resize: VisionResize::PixelBounds {
                min_pixels: 16,
                max_pixels: 64,
            },
            resampling: VisionResampling::Bicubic,
            mean: [0.5; 3],
            std: [0.5; 3],
            channels: 3,
            patch: 2,
            merge: 2,
        },
        stem: VisionStem::Patch {
            frames: vec![
                stored("patch.0", &[128, 3, 2, 2]),
                stored("patch.1", &[128, 3, 2, 2]),
            ],
            bias: None,
            positions: VisionPositions {
                table: stored("position", &[16, 128]),
                sampling: PositionSampling::AlignedCorners { side: 4 },
            },
            norm: None,
            registers: None,
        },
        window: None,
        blocks: Vec::new(),
        merger: VisionMerger {
            norm: None,
            reduction: CellReduction::Concatenate,
            standardize: None,
            projection_norm: None,
            stages: Vec::new(),
            output_norm: None,
        },
    });
    definition
}

/// A prompt of `count` `TEXT` rows with images placed at the given rows,
/// each named by identity. Every image is one 2 x 2-patch cell, so one
/// media row.
fn vision_input(placements: &[(usize, &str)], count: usize) -> PreparedModelInput {
    use magnitude_artifacts::{
        BoundaryRule, InputLayout, InputSpan,
        media::{DType, PreparedTensor},
    };
    use magnitude_family_contracts::{PreparedVisionInput, TokenPlan, VisionSpatialControls};
    let definition = vision_definition();
    let image = || {
        let pixels = PreparedTensor::new(
            "pixel_values".into(),
            DType::F32,
            vec![4, 24],
            vec![0; 4 * 24 * 4],
        )
        .unwrap();
        let spatial = VisionSpatialControls::new(
            (0..4).collect(),
            vec![[0, 0]; 4],
            std::array::from_fn(|_| vec![0; 4]),
            std::array::from_fn(|_| vec![0.25; 4]),
            Vec::new(),
        )
        .unwrap();
        PreparedVisionInput::new(&definition, [1, 2, 2], pixels, spatial).unwrap()
    };
    let mut tokens = vec![TEXT; count];
    let spans = placements
        .iter()
        .map(|&(start, identity)| {
            tokens[start] = MEDIA;
            InputSpan {
                start,
                end: start + 1,
                identity: identity.into(),
                boundaries: BoundaryRule::Causal,
                language_history: false,
            }
        })
        .collect();
    PreparedModelInput::new(
        &definition,
        TokenPlan::new(tokens, InputLayout::new(count, spans).unwrap()).unwrap(),
        (0..count as i32).map(|position| [position; 3]).collect(),
        count,
        placements
            .iter()
            .map(|&(_, identity)| (identity.to_owned(), image()))
            .collect(),
    )
    .unwrap()
}

/// An image encodes its declared cells within the engine's bound, across
/// launches; one whose rows attend each other in the decoder fits one launch.
#[test]
fn an_image_encodes_its_declared_cells_and_one_launch_when_bidirectional() {
    use magnitude_family_contracts::{MediaRowAttention, Operator, VisionResize};
    assert_eq!(crate::image_cell_limit(&tiny_definition(), 2).unwrap(), 0);
    let mut definition = vision_definition();
    assert_eq!(crate::image_cell_limit(&definition, 2).unwrap(), 4);
    let vision = definition.vision.as_mut().unwrap();
    vision.preprocessing.resize = VisionResize::PixelBounds {
        min_pixels: 16,
        max_pixels: 1 << 30,
    };
    assert_eq!(
        crate::image_cell_limit(&definition, 512).unwrap(),
        crate::MAX_IMAGE_CELLS
    );
    for block in &mut definition.decoder.blocks {
        for sublayer in &mut block.sublayers {
            if let Operator::Attention(attention) = &mut sublayer.op {
                attention.media_rows = MediaRowAttention::Bidirectional;
            }
        }
    }
    assert_eq!(crate::image_cell_limit(&definition, 512).unwrap(), 512);
}

const TEXT: crate::TokenId = crate::TokenId(5);
const MEDIA: crate::TokenId = crate::TokenId(9);

/// The image each encode names, in order.
fn encoded_images(operations: &[Operation]) -> Vec<ImageRef> {
    operations
        .iter()
        .map(|operation| match operation {
            Operation::Encode { image, .. } => image.clone(),
            _ => panic!("opening a request yields only encodes"),
        })
        .collect()
}

/// Install `image`'s encoder output (one feature row per cell) as a
/// reconciled encode, and return it.
fn reconcile_encode<F: ProgramFamily>(
    domain: &mut ExecutorDomain<F>,
    request: RequestId,
    image: ImageRef,
) -> FeatureRef {
    let tensor = Tensor::zeros(
        domain.domain.device(),
        seismic::Element::f32(),
        &[
            (image.patches() / 4) as u64,
            domain.definition.decoder.hidden,
        ],
    )
    .unwrap();
    let features = domain.domain.publish_features(tensor).unwrap();
    domain
        .reconcile(
            PendingOperationOutcome {
                request,
                outcome: Outcome::Encode {
                    features: features.clone(),
                },
                advance: None,
                primed: None,
                rows: 0,
                committed_rows: 0,
                kind: WorkKind::Prefill,
                physical_duration: Duration::ZERO,
                replayed: true,
                image: Some(image),
            },
            PhysicalDecision { accepted_rows: 0 },
        )
        .unwrap();
    features
}

/// Where each conditioning slice of the prompt rows from `position` reads
/// from: its destination row, source features, and source row range.
fn placements<F: ProgramFamily>(
    domain: &ExecutorDomain<F>,
    request: RequestId,
    position: usize,
) -> Vec<(usize, FeatureRef, usize, usize)> {
    let tokens = domain.input[&request].input.tokens()[position..].to_vec();
    let (_, slices) = domain.input_rows(request, position, &tokens).unwrap();
    slices
        .into_iter()
        .map(|slice| {
            (
                slice.destination,
                slice.source.features,
                slice.source.start,
                slice.source.count,
            )
        })
        .collect()
}

/// One image placed twice in one prompt is one image: it is admitted,
/// encoded once, and both placements read its features. A distinct image
/// between them keeps its own encode.
#[test]
fn a_repeated_image_is_encoded_once_and_placed_at_every_occurrence() {
    let Some((mut domain, mut bindings)) = fixture(None) else {
        return;
    };
    let request = RequestId(60);
    domain
        .install_input(
            request,
            vision_input(&[(1, "repeated"), (3, "other"), (5, "repeated")], 7),
        )
        .unwrap();
    assert_eq!(domain.input[&request].images.len(), 2);

    let images = encoded_images(&domain.open_state(&mut bindings, request, None).unwrap());
    let [first, second] = images.as_slice() else {
        panic!("two distinct images form two encodes, not three")
    };
    assert_ne!(first, second);
    let features =
        [first, second].map(|image| reconcile_encode(&mut domain, request, image.clone()));
    let repeated_features = if domain.input[&request].images["repeated"].image == *first {
        &features[0]
    } else {
        &features[1]
    };
    let placed = placements(&domain, request, 0);
    assert_eq!(placed.len(), 3);
    assert_eq!(
        placed
            .iter()
            .map(|(row, _, start, count)| (*row, *start, *count))
            .collect::<Vec<_>>(),
        vec![(1, 0, 1), (3, 0, 1), (5, 0, 1)]
    );
    assert!(placed[0].1 == *repeated_features && placed[2].1 == *repeated_features);
    assert!(placed[1].1 != *repeated_features);
}

/// A later turn re-sends the conversation, so an image from an earlier turn
/// placed again is a repeat within the new prompt. Resuming past the first
/// placement encodes the image once more for the remaining placement only.
#[test]
fn an_image_repeated_across_turns_resumes_and_encodes_its_remaining_placement() {
    let Some((mut domain, mut bindings)) = fixture(None) else {
        return;
    };

    // Turn one: prefill the whole prompt, then take its resume state.
    let first = RequestId(61);
    domain
        .install_input(first, vision_input(&[(1, "shared")], 3))
        .unwrap();
    let [encode] = encoded_images(&domain.open_state(&mut bindings, first, None).unwrap())
        .try_into()
        .unwrap_or_else(|_| panic!("one image forms one encode"));
    reconcile_encode(&mut domain, first, encode);
    let prefill_rows = |position: usize, tokens: Vec<crate::TokenId>| Operation::Forward {
        request: first,
        kind: WorkKind::Prefill,
        committed: tokens.len(),
        tokens,
        position,
        conditioning: None,
        demand: crate::batching::Demand::NONE,
        select: Vec::new(),
        prime: None,
    };
    for (position, tokens) in [(0, vec![TEXT, MEDIA]), (2, vec![TEXT])] {
        let rows = tokens.len();
        let flight =
            submit_reserved_target(&mut domain, bindings, vec![prefill_rows(position, tokens)]);
        let (pending, returned) = finish_one(&mut domain, flight);
        domain
            .reconcile(
                pending,
                PhysicalDecision {
                    accepted_rows: rows,
                },
            )
            .unwrap();
        bindings = returned;
    }
    let resume = domain.resume_state(first).unwrap();
    assert_eq!(resume.position(), 3);

    // Turn two: the same image again after the resumed prefix.
    let second = RequestId(62);
    domain
        .install_input(second, vision_input(&[(1, "shared"), (4, "shared")], 6))
        .unwrap();
    let [encode] = encoded_images(
        &domain
            .open_state(&mut bindings, second, Some(&resume))
            .unwrap(),
    )
    .try_into()
    .unwrap_or_else(|_| panic!("the unresumed placement forms one encode"));
    let features = reconcile_encode(&mut domain, second, encode);
    let placed = placements(&domain, second, 3);
    let [(row, source, start, count)] = placed.as_slice() else {
        panic!("one placement past the resumed position")
    };
    assert_eq!((*row, *start, *count), (1, 0, 1));
    assert!(*source == features);
}

/// Publishing target output must not synchronously wait for head completion.
/// Its second flight keeps the completion wake and preserves device failures.
#[test]
fn deferred_priming_keeps_pending_completion_and_reports_device_failure() {
    let Some((domain, bindings)) = fixture(None) else {
        return;
    };
    let control = PendingControl::default();
    let mut flight = PrimingCompletionFlight {
        work: PrimingFlight {
            request: RequestId(88),
            submission: PendingFailureSubmission::new(control.clone()).into(),
            continuation: None,
        },
        target_end: 1,
        bindings,
        started: Instant::now(),
    };
    assert!(!flight.completion().is_complete());
    let woke = Arc::new(AtomicBool::new(false));
    let observed = woke.clone();
    flight.completion().notify(CompletionWake::new(move || {
        observed.store(true, Ordering::SeqCst);
    }));
    assert!(!woke.load(Ordering::SeqCst));
    control.resolve(Err(crate::DeviceError::Execution(
        "deferred entry failed".into(),
    )));
    assert!(woke.load(Ordering::SeqCst));
    assert!(flight.completion().is_complete());
    let mut domain = domain;
    assert!(matches!(
        domain.finish_priming(flight),
        Err(DomainError::Device(_))
    ));
}

/// Form real owned state advances and a checked head launch, with controlled
/// completion instead of numerical kernels. This fixture borrows the plain
/// model's readout pool for the head's leases; no graph consumes those buffers.
/// The target has two final prompt rows and the head has their one known
/// successor. The scheduler's final-prompt detection is outside this fixture.
fn target_with_pending_priming(
    request: RequestId,
) -> (
    ExecutorDomain<TestFamily>,
    TargetFlight<TestFamily>,
    PendingControl,
) {
    let (mut domain, mut bindings) = fixture_on(
        None,
        false,
        true,
        platform::DeviceRequest::Backend(seismic::BackendName::Cpu),
    )
    .expect("deferred priming ownership tests require the CPU fixture");
    open(&mut domain, &mut bindings, request);
    let mut operation = prefill(request, 0);
    if let Operation::Forward {
        tokens, committed, ..
    } = &mut operation
    {
        tokens.push(TokenId(2));
        *committed = 2;
    }
    // Target submission precedes its head submission, just as in production.
    // This controlled target is already ready and has returned its workspace.
    let mut flight = submit_reserved_target(&mut domain, bindings, vec![operation]);
    let advance = TentativeAdvance::Accepted(
        OwnedStateAdvance::begin(domain.head.remove(&request).unwrap(), 1)
            .ok()
            .unwrap(),
    );
    let slot = domain
        .head_slot(request, &[TokenId(2)], 0, &[], &advance, 0)
        .unwrap();
    let batch = crate::batching::ValidatedHeadBatch::from_slots(
        &[slot],
        domain.definition.decoder.vocabulary as usize,
        domain.head_class_limits().unwrap(),
    )
    .unwrap();
    let row_bytes = domain.head_conditioning_bytes();
    let conditioning = FeatureRows::new(vec![0u8; row_bytes].into(), 1).unwrap();
    let pool = domain.resources.target_readout_graph();
    let inputs = crate::HeadLaunchInputs::new(
        batch,
        vec![advance],
        crate::HeadConditioning::Rows(vec![conditioning]),
        true,
        pool.acquire_workspace().unwrap(),
        pool.acquire_output().unwrap(),
    );
    let launch = ValidatedHeadLaunch::new(
        inputs,
        domain.head_store.as_ref().unwrap(),
        domain.resource_identity(),
        row_bytes,
    )
    .unwrap_or_else(|(_, error)| panic!("controlled head launch: {error}"));
    let (core, workspace, output) = launch.into_submission_parts();
    let control = PendingControl::default();
    let head = ControlledHeadSubmission::Work(PendingSubmission {
        completion: PendingCompletion(control.clone()),
        core,
        workspace: (workspace, output),
        output: crate::programs::HeadOutput {
            selections: None,
            landed_reads: Vec::new(),
        },
    });
    assert!(flight.work.priming.is_none());
    flight.work.priming = Some(PrimingFlight {
        request,
        submission: head,
        continuation: None,
    });
    (domain, flight, control)
}

fn publish_before_priming(
    domain: &mut ExecutorDomain<TestFamily>,
    flight: TargetFlight<TestFamily>,
) -> (PendingOperationOutcome, PrimingCompletionFlight<TestFamily>) {
    let (outcomes, continuation) = domain.finish_target_for_publication(flight, true).unwrap();
    let [pending] = outcomes
        .try_into()
        .unwrap_or_else(|_| panic!("one target outcome"));
    let TargetContinuation::Priming(mut priming) = continuation else {
        panic!("the pending head must retain the binding right");
    };
    assert!(!priming.completion().is_complete());
    (pending, priming)
}

#[test]
fn deferred_priming_commits_after_target_publication_and_releases_leases() {
    let request = RequestId(93);
    let (mut domain, flight, control) = target_with_pending_priming(request);
    let held_outputs = domain.resources.target_readout_graph().available_output();
    let (pending, mut priming) = publish_before_priming(&mut domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 2 })
        .unwrap();
    assert_eq!(domain.target[&request].position(), 2);
    assert!(!domain.head.contains_key(&request));
    assert!(
        domain.resume_state(request).is_err(),
        "no half-resolved checkpoint"
    );
    assert!(
        domain.release_state(&[request]).is_err(),
        "no eviction before head completion"
    );
    assert_eq!(
        domain.resources.target_readout_graph().available_output(),
        held_outputs
    );
    let woke = Arc::new(AtomicBool::new(false));
    let observed = woke.clone();
    priming.completion().notify(CompletionWake::new(move || {
        observed.store(true, Ordering::SeqCst);
    }));
    control.resolve(Ok(()));
    assert!(woke.load(Ordering::SeqCst));
    let (finished, _, mut bindings) = domain.finish_priming(priming).unwrap();
    assert_eq!(finished, request);
    let checkpoint = domain.resume_state(request).unwrap();
    assert_eq!(checkpoint.target.position(), 2);
    assert_eq!(checkpoint.head.as_ref().unwrap().position(), 1);
    assert_eq!(
        domain.resources.target_readout_graph().available_output(),
        held_outputs + 1
    );
    // The returned binding right is usable only after the second flight ends.
    domain.provision_open(&mut bindings).unwrap();
    domain.close(request).unwrap();
}

#[test]
fn deferred_priming_aborts_when_its_target_chunk_is_rejected() {
    let request = RequestId(94);
    let (mut domain, flight, control) = target_with_pending_priming(request);
    let (pending, priming) = publish_before_priming(&mut domain, flight);
    domain.abort(pending).unwrap();
    assert_eq!(domain.target[&request].position(), 0);
    assert!(domain.resume_state(request).is_err());
    control.resolve(Ok(()));
    let (_, _, bindings) = domain.finish_priming(priming).unwrap();
    let checkpoint = domain.resume_state(request).unwrap();
    assert_eq!(checkpoint.target.position(), 0);
    assert_eq!(checkpoint.head.as_ref().unwrap().position(), 0);
    // Aborting releases the tentative rows and leaves the target reusable.
    let flight = submit_reserved_target(&mut domain, bindings, vec![forward(request, 0)]);
    let (pending, _) = finish_one(&mut domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    domain.close(request).unwrap();
}

/// Cancellation at the executor boundary must wait to close resident state.
/// The scheduler's cancel control itself is not exercised by this domain test.
#[test]
fn deferred_priming_prevents_cancelled_request_retirement_until_completion() {
    let request = RequestId(95);
    let (mut domain, flight, control) = target_with_pending_priming(request);
    let (pending, priming) = publish_before_priming(&mut domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 2 })
        .unwrap();
    assert_eq!(
        domain.close(request).unwrap_err(),
        "request has unresolved work"
    );
    assert!(domain.input.contains_key(&request));
    assert_eq!(domain.target[&request].position(), 2);
    control.resolve(Ok(()));
    let (_, _, mut bindings) = domain.finish_priming(priming).unwrap();
    domain.close(request).unwrap();
    assert!(!domain.input.contains_key(&request));
    assert!(!domain.target.contains_key(&request));
    assert!(!domain.head.contains_key(&request));
    // Retirement leaves both stores usable by a different request.
    let next = RequestId(96);
    open(&mut domain, &mut bindings, next);
    let fresh = domain.resume_state(next).unwrap();
    assert_eq!(fresh.target.position(), 0);
    assert_eq!(fresh.head.as_ref().unwrap().position(), 0);
    domain.close(next).unwrap();
}

#[test]
fn boundary_features_commit_fork_and_abort_with_numerical_state() {
    let Some((mut domain, mut bindings)) = fixture(None) else {
        return;
    };
    let request = RequestId(91);
    open(&mut domain, &mut bindings, request);
    let first = domain.allocate_boundary().unwrap();
    let attach = |pending: &mut PendingOperationOutcome, boundary: &FeatureRef| {
        let aggregate =
            FeatureRef::logical(boundary.domain().clone(), 1, boundary.allocation().width())
                .unwrap()
                .with_boundary(boundary.clone())
                .unwrap();
        pending.outcome = Outcome::Forward {
            rows: vec![RowResult {
                features: Some(aggregate),
                ..RowResult::default()
            }],
        };
    };
    let flight = submit_reserved_target(&mut domain, bindings, vec![forward(request, 0)]);
    let (mut pending, mut bindings) = finish_one(&mut domain, flight);
    attach(&mut pending, &first);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    let checkpoint = domain.resume_state(request).unwrap();
    assert!(
        checkpoint
            .boundary
            .as_ref()
            .unwrap()
            .same_allocation(&first)
    );
    let fork = RequestId(92);
    domain
        .install_input(fork, PreparedModelInput::continuation_only())
        .unwrap();
    domain
        .open_state(&mut bindings, fork, Some(&checkpoint))
        .unwrap();
    domain.release_state(&[request]).unwrap();
    assert!(domain.input[&request].boundary.is_none());
    assert!(
        domain.input[&fork]
            .boundary
            .as_ref()
            .unwrap()
            .same_allocation(&first)
    );
    let second = domain.allocate_boundary().unwrap();
    for invalid in [false, true] {
        let flight = submit_reserved_target(&mut domain, bindings, vec![forward(fork, 1)]);
        let (mut pending, returned) = finish_one(&mut domain, flight);
        bindings = returned;
        attach(&mut pending, &second);
        if invalid {
            assert!(
                domain
                    .reconcile(pending, PhysicalDecision { accepted_rows: 0 })
                    .is_err()
            );
        } else {
            domain.abort(pending).unwrap();
        }
        let restored = domain.resume_state(fork).unwrap();
        assert_eq!(restored.position(), 1);
        assert!(restored.boundary.as_ref().unwrap().same_allocation(&first));
    }
    let flight = submit_reserved_target(&mut domain, bindings, vec![forward(fork, 1)]);
    let (mut pending, _) = finish_one(&mut domain, flight);
    attach(&mut pending, &second);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    assert!(
        domain
            .resume_state(fork)
            .unwrap()
            .boundary
            .as_ref()
            .unwrap()
            .same_allocation(&second)
    );
    assert!(
        checkpoint
            .boundary
            .as_ref()
            .unwrap()
            .same_allocation(&first)
    );
}

/// A forward of `values` at `position`, every row committed.
fn feed(request: RequestId, position: usize, values: &[u32]) -> Operation {
    Operation::Forward {
        request,
        kind: WorkKind::Prefill,
        tokens: values.iter().map(|value| crate::TokenId(*value)).collect(),
        position,
        conditioning: None,
        demand: crate::batching::Demand::NONE,
        select: Vec::new(),
        committed: values.len(),
        prime: None,
    }
}

fn token_values(tokens: &[crate::TokenId]) -> Vec<u32> {
    tokens.iter().map(|token| token.0).collect()
}

/// A request's tokens follow its sequence as its state does: an aborted
/// step's tokens give way to the forward fed again at its position, and a
/// sequence resumed from another's state holds that state's tokens and feeds
/// on from them, leaving its source's unchanged.
#[test]
fn token_history_follows_rollback_and_fork() {
    let Some((mut domain, mut bindings)) = fixture(None) else {
        return;
    };
    let source = RequestId(81);
    open(&mut domain, &mut bindings, source);
    // The fixture launches at most two rows.
    let flight = submit_reserved_target(&mut domain, bindings, vec![feed(source, 0, &[10, 11])]);
    let (pending, returned) = finish_one(&mut domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 2 })
        .unwrap();
    let flight = submit_reserved_target(&mut domain, returned, vec![feed(source, 2, &[12])]);
    let (pending, returned) = finish_one(&mut domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    // A step past the accepted rows is fed, then aborted.
    let flight = submit_reserved_target(&mut domain, returned, vec![feed(source, 3, &[20, 21])]);
    let (pending, returned) = finish_one(&mut domain, flight);
    domain.abort(pending).unwrap();
    assert_eq!(domain.resume_state(source).unwrap().position(), 3);
    assert_eq!(
        token_values(domain.preceding_tokens(source, 3, 2).unwrap()),
        [11, 12]
    );
    let flight = submit_reserved_target(&mut domain, returned, vec![feed(source, 3, &[40])]);
    let (pending, mut bindings) = finish_one(&mut domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    assert_eq!(
        token_values(domain.preceding_tokens(source, 4, 2).unwrap()),
        [12, 40]
    );
    // Nothing is known past the fed tokens.
    assert!(domain.preceding_tokens(source, 5, 2).is_err());

    let checkpoint = domain.resume_state(source).unwrap();
    let copy = RequestId(82);
    domain
        .install_input(copy, PreparedModelInput::continuation_only())
        .unwrap();
    domain
        .open_state(&mut bindings, copy, Some(&checkpoint))
        .unwrap();
    assert_eq!(
        token_values(domain.preceding_tokens(copy, 4, 8).unwrap()),
        [10, 11, 12, 40]
    );
    let flight = submit_reserved_target(&mut domain, bindings, vec![feed(copy, 4, &[50])]);
    let (pending, _) = finish_one(&mut domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    assert_eq!(
        token_values(domain.preceding_tokens(copy, 5, 2).unwrap()),
        [40, 50]
    );
    assert!(domain.preceding_tokens(source, 5, 1).is_err());
}

#[test]
fn a_forward_replaces_the_tokens_at_and_after_its_position() {
    use super::input::TokenHistory;
    let tokens = |values: &[u32]| values.iter().map(|value| crate::TokenId(*value)).collect::<Vec<_>>();
    let mut history = TokenHistory::default();
    history.feed(0, &tokens(&[1, 2, 3, 4])).unwrap();
    // A verification's rejected rows are fed again from the first rejected.
    history.feed(2, &tokens(&[9])).unwrap();
    assert_eq!(token_values(history.before(3).unwrap()), [1, 2, 9]);
    assert!(history.before(4).is_err());
    // No forward follows tokens never fed.
    assert!(history.feed(5, &tokens(&[7])).is_err());
    let resumed = TokenHistory::starting(history.before(2).unwrap());
    assert_eq!(token_values(resumed.before(2).unwrap()), [1, 2]);
}

/// The host's rows for a step (`NGramHash::rows` over the request's token
/// history) equal the rows the device's `ngram_rows` names from the step's
/// tokens and the bank's token window: through a prefill chunk, a
/// verification of which only some rows are accepted, and the decode that
/// continues from the tape version.
#[test]
fn host_rows_from_the_token_history_equal_the_device_rows() {
    use magnitude_family_contracts::NGramHash;
    use magnitude_kernels::ngram_rows;
    use seismic::{BackendName, Element, NativeSpecialization, SlabRegion, SlabTensor, Tensor};
    let hash = NGramHash {
        context: 2,
        heads_per_order: 8,
        multipliers: vec![(i64::MAX as u64 / 248_320) | 1, 1_000_003, (1 << 41) + 17],
        sizes: (0..16).map(|head| 20_000_003 + 2 * head).collect(),
        offsets: (0..16).map(|head| head * 20_000_040).collect(),
        reset: 7,
    };
    let (context, heads) = (2u64, 16usize);
    let tape = 3u64;
    let (banks, slab_banks) = (4u64, 2u64);
    let catalog = seismic::Devices::discover(&[]).unwrap();
    let devices = [BackendName::Metal, BackendName::Cuda, BackendName::Vulkan, BackendName::Cpu]
        .into_iter()
        .filter_map(|backend| catalog.open_backend(backend).ok())
        .collect::<Vec<_>>();
    let ints = |device: &seismic::Device, shape: &[u64], values: &[i32]| {
        Tensor::from_host(
            device,
            Element::i32(),
            shape,
            &values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>(),
        )
        .unwrap()
    };
    let words = |device: &seismic::Device, values: &[u64]| {
        Tensor::from_host(
            device,
            Element::u64(),
            &[values.len() as u64],
            &values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>(),
        )
        .unwrap()
    };
    let tokens = |values: &[u32]| values.iter().map(|value| crate::TokenId(*value)).collect::<Vec<_>>();
    // (position, tokens, the version read (bank, tape rows), stop, the bank
    // published, rows accepted).
    let steps: [(usize, Vec<u32>, (i32, i32), i32, i32, usize); 3] = [
        (0, vec![3, 7, 7, 12, 40, 41, 7, 9, 13], (0, 0), 9, 1, 9),
        (9, vec![22, 7, 31, 5], (1, 0), 1, 2, 2),
        (11, vec![17], (2, 1), 1, 3, 1),
    ];
    for device in devices {
        let backend = device.backend().as_str();
        let mut slabs = SlabTensor::new(
            &device,
            slab_banks,
            banks,
            vec![SlabRegion {
                element: Element::u32(),
                row_shape: vec![context + tape, 1],
            }],
        )
        .unwrap();
        for slab in 0..banks.div_ceil(slab_banks) {
            assert_eq!(slabs.add_slab().unwrap(), slab as usize);
            slabs
                .region_rows(0, slab * slab_banks, slab_banks)
                .unwrap()
                .write_from_host(&vec![0u8; (slab_banks * (context + tape) * 4) as usize])
                .unwrap();
        }
        let mut window = slabs.logical_region(0).unwrap();
        let implementation = seismic::generated::native_implementation::<ngram_rows::Entry>(&device)
            .unwrap()
            .unwrap();
        let statics = match backend {
            "cpu" => NativeSpecialization::new(),
            _ => NativeSpecialization::new()
                .with_static("C", context)
                .with_static("P", 8),
        };
        let kernel = ngram_rows::native_for_device(
            &device,
            &implementation.default_specialization(&statics).unwrap(),
        )
        .unwrap();
        let mut history = super::input::TokenHistory::default();
        for (position, fed, (source, taped), stop, target, accepted) in &steps {
            let fed = tokens(fed);
            let preceding = history
                .before(*position)
                .unwrap()
                .iter()
                .rev()
                .take(context as usize)
                .rev()
                .map(|token| u64::from(token.0))
                .collect::<Vec<_>>();
            let host = hash.rows(&preceding, &fed.iter().map(|token| u64::from(token.0)).collect::<Vec<_>>());
            let rows = fed.len();
            let result = kernel
                .call(ngram_rows::Args {
                    tokens: &ints(
                        &device,
                        &[rows as u64, 2],
                        &fed.iter().flat_map(|token| [token.0 as i32, 0]).collect::<Vec<_>>(),
                    ),
                    multipliers: &words(&device, &hash.multipliers),
                    sizes: &words(&device, &hash.sizes),
                    offsets: &words(&device, &hash.offsets),
                    segments: &ints(&device, &[2, 2], &[0, rows as i32, rows as i32, rows as i32]),
                    stop: &ints(&device, &[1], &[*stop]),
                    previous_bank: &ints(&device, &[1], &[*source]),
                    previous_tape: &ints(&device, &[1], &[*taped]),
                    following_bank: &ints(&device, &[1], &[*target]),
                    window: &mut window,
                    reset: hash.reset as i32,
                    slab_banks: slab_banks as u32,
                })
                .unwrap()
                .value;
            let device_rows = result
                .read_to_host()
                .unwrap()
                .chunks_exact(8)
                .map(|pair| u64::from(u32::from_le_bytes(pair[..4].try_into().unwrap())))
                .collect::<Vec<_>>();
            assert_eq!(device_rows.len(), rows * heads);
            assert_eq!(device_rows, host, "{backend}: step at {position}");
            // The step's tokens are fed; the next forward follows the
            // accepted ones.
            history.feed(*position, &fed).unwrap();
            history.before(position + accepted).unwrap();
        }
    }
}

/// A stored table of 4096 IQ4_NL rows 160 wide whose step 1 named row 1 for
/// batch slot 0 and row 3500 for slot 1, read after the artifact was cut
/// short under the open table: slot 1's row could not be read. The table
/// file's path, to remove, and the step's read.
fn failed_table_read(name: &str) -> (std::path::PathBuf, crate::programs::StoredRead) {
    use crate::stored_table::{StepRows, StoredTable};
    const ROWS: u64 = 4096;
    const ROW_BYTES: u64 = 90;
    let path = std::env::temp_dir().join(format!(
        "magnitude-domain-table-{name}-{}.bin",
        std::process::id()
    ));
    std::fs::write(&path, vec![0x11u8; (ROWS * ROW_BYTES) as usize]).unwrap();
    let device = seismic::Devices::discover(&[])
        .unwrap()
        .open_backend(seismic::BackendName::Cpu)
        .unwrap();
    let table = Arc::new(
        StoredTable::new(
            &device,
            seismic::Element::named("gguf_iq4_nl_block").unwrap(),
            ROWS,
            160,
            64,
            crate::planning::DirectoryForm::Hashed,
            magnitude_artifacts::UncachedSource::open(&path).unwrap(),
            0,
            None,
        )
        .unwrap(),
    );
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(2048 * ROW_BYTES)
        .unwrap();
    let step = table
        .queue(StepRows::Known(vec![vec![1], vec![3500]]))
        .unwrap();
    assert!(step.gate.wait(step.value, std::time::Duration::from_secs(10)));
    // The device read the step's rows.
    step.read.signal(step.value).unwrap();
    (path, crate::programs::StoredRead::new(table, step.value))
}

/// Spec §7.3: a step whose stored-table rows could not be read for one of
/// its two sequences fails that sequence alone, with its typed error. Its
/// outputs are discarded and its accepted state returns, as an orphan's;
/// it is poisoned, so its next step fails too though that step reads
/// nothing that failed, until it closes. The other sequence commits both
/// steps.
#[test]
fn a_table_read_failure_fails_only_the_sequence_that_named_the_rows() {
    let Some((mut domain, mut bindings)) = fixture(None) else {
        return;
    };
    let (kept, failed) = (RequestId(91), RequestId(92));
    open(&mut domain, &mut bindings, kept);
    open(&mut domain, &mut bindings, failed);
    let (path, read) = failed_table_read("poison");
    STORED_READS.with(|reads| reads.borrow_mut().push(read));
    let mut bindings = bindings;
    for (step, position) in [(0, 0usize), (1, 1)] {
        let flight = submit_reserved_target(
            &mut domain,
            bindings,
            vec![feed(kept, position, &[10 + step]), feed(failed, 0, &[20])],
        );
        let (pending, returned) = domain.finish_target(flight).unwrap();
        bindings = returned;
        assert_eq!(pending.len(), 2);
        for pending in pending {
            if pending.request() == kept {
                assert!(matches!(pending.outcome(), Outcome::Forward { .. }));
                domain
                    .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
                    .unwrap();
            } else {
                let Outcome::TableReadFailed(error) = pending.outcome() else {
                    panic!("step {step}: the failed sequence's outcome is not its failure");
                };
                assert!(error.detail.contains("3500"), "{error}");
                assert!(matches!(
                    domain.reconcile(pending, PhysicalDecision { accepted_rows: 1 }),
                    Err(DomainError::TableRead(_))
                ));
            }
        }
        assert_eq!(domain.resume_state(kept).unwrap().position(), position + 1);
        assert_eq!(domain.resume_state(failed).unwrap().position(), 0);
    }
    // Closing the sequence clears its poison.
    domain.close(failed).unwrap();
    open(&mut domain, &mut bindings, failed);
    let flight = submit_reserved_target(&mut domain, bindings, vec![feed(failed, 0, &[20])]);
    let (pending, _) = finish_one(&mut domain, flight);
    assert!(matches!(pending.outcome(), Outcome::Forward { .. }));
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    std::fs::remove_file(path).unwrap();
}

/// The planning fixture with its attention selecting one block of four rows
/// past the dense limit, by an indexer of two 64-wide heads, over a 64-token
/// context.
fn selected_definition() -> ModelDefinition {
    use magnitude_family_contracts::{
        HistoryReads, Indexer, Operator, RmsNorm, WeightDescriptor,
    };
    let mut definition = tiny_definition();
    definition.decoder.context_limit = 64;
    let Operator::Attention(attention) = &mut definition.decoder.blocks[0].sublayers[0].op else {
        panic!("the fixture's mixer is attention");
    };
    let hidden = definition.decoder.hidden;
    let norm = |name: &str| RmsNorm {
        weight: WeightDescriptor::stored(name, vec![64]),
        epsilon: 1e-6,
    };
    attention.reads = HistoryReads::Selected {
        block: 4,
        blocks: 1,
        indexer: Indexer {
            heads: 2,
            width: 64,
            query: WeightDescriptor::stored("indexer_q", vec![128, hidden]),
            key: WeightDescriptor::stored("indexer_k", vec![64, hidden]),
            query_norm: norm("indexer_q_norm"),
            key_norm: norm("indexer_k_norm"),
        },
    };
    definition
}

/// A selected read's launches carry its indexer's histories as the state
/// holds them: every row reads the raw keys of the three positions before
/// it and the blocks complete before it, writes its raw key and the block
/// it completes; a commit of an accepted prefix keeps the blocks that prefix
/// completes, so a rollback across a block boundary continues as a straight
/// run does. The graphs of every class (the indexer, the Segments decode,
/// the WorkList prefill) prepare and seal on the device.
#[test]
fn selected_histories_follow_blocks_across_rollback() {
    use magnitude_state::HistoryDomainId;
    const RAW: HistoryDomainId = HistoryDomainId(1);
    const BLOCKS: HistoryDomainId = HistoryDomainId(2);
    let Some((mut domain, mut bindings)) = fixture_of(
        selected_definition(),
        16,
        None,
        false,
        false,
        platform::DeviceRequest::Automatic,
    ) else {
        return;
    };
    let blocks = |domain: &ExecutorDomain<TestFamily>, request| {
        domain
            .resume_state(request)
            .unwrap()
            .target
            .history_ranges(BLOCKS)
            .iter()
            .map(|(_, rows)| rows)
            .sum::<usize>()
    };
    let source = RequestId(91);
    open(&mut domain, &mut bindings, source);
    // Ten prompt rows complete blocks 0 and 1.
    let tokens = (0..10).collect::<Vec<u32>>();
    let flight = submit_reserved_target(&mut domain, bindings, vec![feed(source, 0, &tokens)]);
    let (pending, bindings) = finish_one(&mut domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 10 })
        .unwrap();
    assert_eq!(blocks(&domain, source), 2);
    // Six rows from 10 complete blocks 2 and 3, then roll back whole: the
    // history keeps blocks 0 and 1.
    let flight = submit_reserved_target(
        &mut domain,
        bindings,
        vec![feed(source, 10, &[20, 21, 22, 23, 24, 25])],
    );
    let (pending, bindings) = finish_one(&mut domain, flight);
    domain.abort(pending).unwrap();
    assert_eq!(domain.resume_state(source).unwrap().position(), 10);
    assert_eq!(blocks(&domain, source), 2);
    // Two other rows from 10 (positions 10, 11) complete block 2 alone.
    let flight = submit_reserved_target(&mut domain, bindings, vec![feed(source, 10, &[40, 41])]);
    let (pending, bindings) = finish_one(&mut domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 2 })
        .unwrap();
    assert_eq!(domain.resume_state(source).unwrap().position(), 12);
    assert_eq!(blocks(&domain, source), 3);
    // The raw keys of positions 9..12 remain for the next block.
    let checkpoint = domain.resume_state(source).unwrap();
    assert_eq!(
        checkpoint
            .target
            .history_ranges(RAW)
            .iter()
            .map(|(_, rows)| rows)
            .sum::<usize>(),
        3
    );
    // Single decode rows continue past the dense limit, completing block 3.
    let mut bindings = bindings;
    for (position, token) in (12..16).zip(30..) {
        let flight = submit_reserved_target(&mut domain, bindings, vec![feed(source, position, &[token])]);
        let (pending, returned) = finish_one(&mut domain, flight);
        domain
            .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
            .unwrap();
        bindings = returned;
    }
    assert_eq!(blocks(&domain, source), 4);
}
