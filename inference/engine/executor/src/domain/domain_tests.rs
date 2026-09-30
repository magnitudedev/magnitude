//! Controlled completion tests for the production execution domain.

use super::*;
use crate::{
    completion::CompletionWake,
    platform,
    programs::{
        CompletedHeadWork, CompletedStateWork, CompletedVisionWork, CompletedWork,
        ProgramSubmission, ReadySubmission,
    },
    AttestedPrograms, ComponentSelection, ExecutionPath, ExecutionPlanner, PlannedMethod,
    ResourceAllocator, ResourceCapacity, ResourceLimits, ResourcePlanner, TargetLaunchCore,
};
use magnitude_artifacts::PackageManifest;
use magnitude_family_contracts::FamilyId;
use magnitude_state::{KvCodec, ShrinkPolicy};
use std::marker::PhantomData;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
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
                components: vec![ComponentDescriptor::new(
                    LayerRef::Target(0),
                    CodecSpec::dense(seismic::DType::F32, 4096, 4096),
                    1,
                )
                .unwrap()],
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
    let catalog = seismic::DeviceCatalog::discover().ok()?;
    let reserves = platform::MemoryReserves::standard();
    let selected = platform::select_device(
        &catalog,
        ExecutionPath::Native,
        platform::DeviceRequest::Automatic,
        &reserves,
    )
    .ok()?;
    let device = Rc::new(
        catalog
            .open(catalog.resolve(selected.info.selector).unwrap())
            .unwrap(),
    );
    let definition = Rc::new(tiny_definition());
    let manifest = tiny_manifest(&definition);
    let limits = ResourceLimits {
        max_launch_rows: 2,
        max_launch_slots: 2,
        max_projected_rows: 2,
        max_images_per_request: magnitude_artifacts::MAX_IMAGES_PER_REQUEST,
        lookahead,
    };
    let capacity_bytes = ResourceCapacity {
        domain_bytes: selected.assessment_capacity_bytes.min(512 * 1024 * 1024),
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
        },
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
            draft.programs().vision(),
            state.target_state(),
            state.head_state(),
            limits,
            0,
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
    Some(ExecutorDomain::with_family(
        execution,
        definition,
        resources,
        DeviceHeap::open(catalog, reserves, device).unwrap(),
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
    assert!(domain.open_state(bindings, request, None).unwrap().is_empty());
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
        self.0 .0.lock().unwrap().result.is_some()
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
            let mut state = self.0 .0.lock().unwrap();
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

impl crate::programs::SubmittedHead for PendingFailureSubmission<CompletedHeadWork> {
    fn launch(&self) -> &crate::HeadLaunchCore {
        unreachable!("controlled fixture submits no head launch")
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
    type HeadSubmission = PendingFailureSubmission<CompletedHeadWork>;
    type VisionSubmission = PendingFailureSubmission<CompletedVisionWork>;
    type StateSubmission = PendingFailureSubmission<CompletedStateWork>;
    fn bind_head(&mut self, _: crate::ResidentHead, _: &ModelDefinition) -> Result<(), String> {
        Ok(())
    }
    fn bind_vision(&mut self, _: crate::ResidentVision, _: &ModelDefinition) -> Result<(), String> {
        Ok(())
    }
    fn head_is_bound(&self) -> bool {
        false
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
                readout: None,
                commits: crate::CommitSpan {
                    first: now,
                    last: now,
                },
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
        _: ValidatedStateLaunch,
    ) -> Result<Self::StateSubmission, (crate::SubmitError, ValidatedStateLaunch)> {
        panic!("controlled fixture does not submit state work")
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
    assert!(domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .is_ok());
    assert_eq!(domain.resume_state(request).unwrap().target.position(), 1);
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
    assert!(bindings.lookahead.is_none(), "the growing entry orphaned the lookahead");
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
    match domain.provision(&mut bindings, &[draft]) {
        Ok(()) => {}
        Err(DomainError::Input(detail)) if detail == "head component is disabled" => {}
        Err(error) => panic!("provisioning drafter work failed: {error}"),
    }
    assert!(bindings.lookahead.is_none(), "drafter work resolved the lookahead");
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
    let accept = |domain: &mut ExecutorDomain<TestFamily>,
                  bindings: StateBindings<TestFamily>,
                  position| {
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
    assert!(store
        .history_domains()
        .all(|history| store.free_rows(history) == 0));
    assert_eq!(domain.resume_state(request).unwrap().target.position(), 4);
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
    let flight =
        submit_reserved_target(&mut failed_domain, bindings, vec![forward(failed_request, 0)]);
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
        submission: PendingFailureSubmission::<CompletedHeadWork>::new(head_control.clone()),
        started: Instant::now(),
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
            image: Some(image),
        })
        .unwrap();
    assert_eq!(domain.target.get(&request).unwrap().position(), 0);
}
