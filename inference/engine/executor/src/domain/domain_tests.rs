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
    fixture_on(control, lookahead, head, platform::DeviceRequest::Automatic)
}

fn fixture_on(
    control: Option<PendingControl>,
    lookahead: bool,
    head: bool,
    request: platform::DeviceRequest,
) -> Option<Fixture> {
    let catalog = seismic::DeviceCatalog::discover().ok()?;
    let reserves = platform::MemoryReserves::standard();
    let selected =
        platform::select_device(&catalog, ExecutionPath::Native, request, &reserves).ok()?;
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
        max_selected_rows: 2,
        max_drafting_slots: 2,
        exported_logits_rows: 0,
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
    assert_eq!(before.activation_transfer, 0);
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
            #[cfg(any(test, feature = "experimental-pipeline-cuda"))]
            pipeline_prefix: None,
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
            #[cfg(any(test, feature = "experimental-pipeline-cuda"))]
            pipeline_prefix: None,
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
        media::{DType, PreparedTensor},
        BoundaryRule, InputLayout, InputSpan,
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
                #[cfg(any(test, feature = "experimental-pipeline-cuda"))]
                pipeline_prefix: None,
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

// Logical paired publication through the real domain, using explicit CPU-only
// preparation and no numerical model execution. These do not qualify CUDA serving.
fn pipeline_pending_fixture(
    prefix_rows: usize,
    suffix_rows: usize,
) -> (Fixture, PendingOperationOutcome) {
    let (domain, bindings) = fixture_on(
        None,
        false,
        false,
        platform::DeviceRequest::Backend(seismic::BackendName::Cpu),
    )
    .expect("CPU domain fixture");
    let cpu = Rc::new(domain.domain.device().clone());
    let prefix_store = domain
        .execution
        .resources()
        .target_state()
        .allocate(cpu)
        .unwrap();
    let prefix = OwnedStateAdvance::begin(prefix_store.create().unwrap(), prefix_rows)
        .ok()
        .unwrap();
    let suffix = OwnedStateAdvance::begin(domain.target_store.create().unwrap(), suffix_rows)
        .ok()
        .unwrap();
    let request = RequestId(901);
    // Physical output is already complete but neither logical source is published.
    assert!(domain.target.is_empty());
    assert!(domain.pipeline_prefix.is_empty());
    let pending = PendingOperationOutcome {
        request,
        outcome: Outcome::Forward { rows: Vec::new() },
        advance: Some(suffix),
        pipeline_prefix: Some(prefix),
        primed: None,
        rows: suffix_rows,
        committed_rows: suffix_rows,
        kind: WorkKind::Prefill,
        physical_duration: Duration::ZERO,
        image: None,
    };
    ((domain, bindings), pending)
}
#[test]
fn pipeline_domain_joint_publication_and_subsequent_state_reuse() {
    let ((mut domain, _bindings), pending) = pipeline_pending_fixture(2, 2);
    let request = pending.request;
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 2 })
        .unwrap();
    assert_eq!(domain.target[&request].position(), 2);
    assert_eq!(domain.pipeline_prefix[&request].position(), 2);
    let prefix = domain.pipeline_prefix.remove(&request).unwrap();
    let suffix = domain.target.remove(&request).unwrap();
    let pending = PendingOperationOutcome {
        request,
        outcome: Outcome::Forward { rows: Vec::new() },
        advance: Some(OwnedStateAdvance::begin(suffix, 1).ok().unwrap()),
        pipeline_prefix: Some(OwnedStateAdvance::begin(prefix, 1).ok().unwrap()),
        primed: None,
        rows: 1,
        committed_rows: 1,
        kind: WorkKind::Decode,
        physical_duration: Duration::ZERO,
        image: None,
    };
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    assert_eq!(domain.target[&request].position(), 3);
    assert_eq!(domain.pipeline_prefix[&request].position(), 3);
}
#[test]
fn pipeline_domain_mismatched_stage_extents_publish_neither() {
    let ((mut domain, _bindings), pending) = pipeline_pending_fixture(1, 2);
    let request = pending.request;
    assert!(domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 2 })
        .is_err());
    assert_eq!(domain.target[&request].position(), 0);
    assert_eq!(domain.pipeline_prefix[&request].position(), 0);
}
#[test]
fn pipeline_domain_completed_cancellation_aborts_both_sources() {
    let ((mut domain, _bindings), pending) = pipeline_pending_fixture(2, 2);
    let request = pending.request;
    domain.abort(pending).unwrap();
    assert_eq!(domain.target[&request].position(), 0);
    assert_eq!(domain.pipeline_prefix[&request].position(), 0);
    assert!(OwnedStateAdvance::begin(domain.target[&request].checkpoint().fork(), 2).is_ok());
}
#[test]
fn pipeline_domain_invalid_decision_restores_both_without_acceptance() {
    let ((mut domain, _bindings), pending) = pipeline_pending_fixture(2, 2);
    let request = pending.request;
    assert!(domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .is_err());
    assert_eq!(domain.target[&request].position(), 0);
    assert_eq!(domain.pipeline_prefix[&request].position(), 0);
}

#[test]
fn pipeline_domain_state_cannot_publish_as_head_or_stateless_output() {
    for head in [true, false] {
        let ((mut domain, _bindings), mut pending) = pipeline_pending_fixture(2, 2);
        pending.outcome = if head {
            Outcome::Head {
                proposals: Vec::new(),
            }
        } else {
            Outcome::Encode {
                features: FeatureRef::logical(
                    domain.resource_identity().clone(),
                    1,
                    domain.definition.decoder.hidden as usize,
                )
                .unwrap(),
            }
        };
        assert!(domain
            .reconcile(pending, PhysicalDecision { accepted_rows: 2 })
            .is_err());
        assert!(domain.target.is_empty());
        assert!(domain.pipeline_prefix.is_empty());
        assert!(domain.head.is_empty());
        // A malformed result was terminally discarded, not published or returned
        // to the wrong lane. New owned advances can still reserve their banks.
        let state = domain.target_store.create().unwrap();
        assert_eq!(state.position(), 0);
        assert!(OwnedStateAdvance::begin(state, 2).is_ok());
    }
}

#[test]
fn pipeline_binding_adoption_requires_exact_store_and_opened_device() {
    let (domain, bindings) = fixture_on(
        None,
        false,
        false,
        platform::DeviceRequest::Backend(seismic::BackendName::Cpu),
    )
    .expect("CPU fixture");
    let device = domain.domain.device();
    assert!(super::pipeline_resources::validate_binding_owner(
        device,
        &domain.target_store,
        &bindings.target
    )
    .is_ok());
    let other_store = domain
        .execution
        .resources()
        .target_state()
        .allocate(Rc::new(device.clone()))
        .unwrap();
    assert!(super::pipeline_resources::validate_binding_owner(
        device,
        &domain.target_store,
        &other_store
    )
    .is_err());
    let foreign = seismic::DeviceCatalog::discover()
        .unwrap()
        .open_backend(seismic::BackendName::Cpu)
        .unwrap();
    assert!(super::pipeline_resources::validate_binding_owner(
        &foreign,
        &domain.target_store,
        &bindings.target
    )
    .is_err());
    assert!(domain.pipeline_owner.is_none());
    assert!(bindings.pipeline_prefix.is_none());
    assert!(domain.pipeline_prefix_memory_charge().unwrap().is_none());
}
