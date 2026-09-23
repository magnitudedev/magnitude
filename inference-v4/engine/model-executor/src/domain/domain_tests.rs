//! Controlled completion tests for the production execution domain.

use super::*;
use crate::{
    AttestedPrograms, ComponentSelection, ExecutionPath, ExecutionPlanner, PlannedMethod,
    ResourceAllocator, ResourceBudget, ResourceLimits, ResourcePlanner, TargetLaunchCore,
    completion::CompletionWake,
    platform,
    programs::{
        CompletedHeadWork, CompletedProjectWork, CompletedStateWork, CompletedVisionWork,
        CompletedWork, ProgramSubmission, ReadySubmission,
    },
};
use magnitude_artifacts::{
    ArtifactIdentity, ComponentManifest, PackageIdentity, PackageManifest,
    gguf::{Encoding, TensorDescriptor},
};
use magnitude_model_contracts::{
    ActivationDType, AttentionGeometry, AttentionWeights, BlockGeometry, BlockWeights,
    DecoderGeometry, DenseFeedForwardWeights, FamilyId, FeedForwardGeometry, FeedForwardWeights,
    InputSemantics, MixerGeometry, MixerWeights, RotarySemantics, TextCoordinateSemantics,
    WeightDescriptor,
};
use magnitude_model_state::KvCodec;
use std::marker::PhantomData;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

fn descriptor(name: &str, shape: &[u64]) -> WeightDescriptor {
    WeightDescriptor {
        name: name.into(),
        shape: shape.to_vec(),
    }
}

fn tiny_definition() -> ModelDefinition {
    ModelDefinition {
        family: FamilyId("controlled-target-domain".into()),
        artifact_identity: PackageIdentity {
            target: ArtifactIdentity([7; 32]),
            projector: None,
        },
        inputs: InputSemantics {
            text_coordinates: TextCoordinateSemantics::ReplicatedPosition,
            coordinate_axes: 1,
        },
        geometry: DecoderGeometry {
            activation_dtype: ActivationDType::BF16,
            hidden: 2,
            vocabulary: 8,
            context_limit: 8,
            epsilon: 1e-6,
            blocks: vec![BlockGeometry {
                mixer: MixerGeometry::Attention(AttentionGeometry {
                    heads: 1,
                    kv_heads: 1,
                    width: 4,
                    rotary: RotarySemantics::Interleaved {
                        width: 2,
                        base: 10_000.0,
                        sections: vec![1],
                        axis_pattern: vec![0],
                    },
                }),
                feedforward: FeedForwardGeometry::Dense { intermediate: 4 },
            }],
        },
        embedding: descriptor("embedding", &[8, 2]),
        blocks: vec![BlockWeights {
            input_norm: descriptor("input_norm", &[2]),
            mixer: MixerWeights::Attention(Box::new(AttentionWeights {
                query_gate: descriptor("qg", &[8, 2]),
                key: descriptor("k", &[4, 2]),
                value: descriptor("v", &[4, 2]),
                query_norm: descriptor("qn", &[4]),
                key_norm: descriptor("kn", &[4]),
                output: descriptor("o", &[2, 4]),
            })),
            feedforward_norm: descriptor("ffn_norm", &[2]),
            feedforward: FeedForwardWeights::Dense(Box::new(DenseFeedForwardWeights {
                gate: descriptor("gate", &[4, 2]),
                up: descriptor("up", &[4, 2]),
                down: descriptor("down", &[2, 4]),
            })),
        }],
        output_norm: descriptor("output_norm", &[2]),
        output: descriptor("output", &[8, 2]),
        head: None,
        vision: None,
    }
}

fn tiny_manifest(definition: &ModelDefinition) -> PackageManifest {
    let block = &definition.blocks[0];
    let MixerWeights::Attention(attention) = &block.mixer else {
        unreachable!()
    };
    let FeedForwardWeights::Dense(dense) = &block.feedforward else {
        unreachable!()
    };
    let descriptors = [
        &definition.embedding,
        &block.input_norm,
        &attention.query_gate,
        &attention.key,
        &attention.value,
        &attention.query_norm,
        &attention.key_norm,
        &attention.output,
        &block.feedforward_norm,
        &dense.gate,
        &dense.up,
        &dense.down,
        &definition.output_norm,
        &definition.output,
    ];
    let mut offset = 0;
    let tensors = descriptors
        .into_iter()
        .map(|descriptor| {
            let nbytes = descriptor.shape.iter().product::<u64>() * 2;
            let tensor = TensorDescriptor {
                name: descriptor.name.clone(),
                shape: descriptor.shape.clone(),
                encoding: Encoding::F16,
                offset,
                nbytes,
            };
            offset += nbytes;
            tensor
        })
        .collect();
    PackageManifest {
        identity: definition.artifact_identity,
        target: ComponentManifest {
            path: "controlled-domain.gguf".into(),
            identity: definition.artifact_identity.target,
            size: offset,
            tensors,
        },
        projector: None,
    }
}

fn fixture(control: Option<PendingControl>) -> Option<ExecutorDomain<TestFamily>> {
    let catalog = seismic::DeviceCatalog::discover().ok()?;
    let selected = platform::select_device(&catalog, ExecutionPath::NativeMetal).ok()?;
    let device = Rc::new(
        catalog
            .open(catalog.resolve(selected.info.selector).unwrap())
            .unwrap(),
    );
    let definition = Rc::new(tiny_definition());
    let manifest = tiny_manifest(&definition);
    let limits = ResourceLimits {
        active_requests: 2,
        in_flight_requests: 2,
        branch_checkpoints: 0,
        max_batch_rows: 2,
        max_projected_rows: 2,
        max_images_per_request: magnitude_artifacts::MAX_IMAGES_PER_REQUEST,
    };
    let budget = ResourceBudget {
        storage_bytes: selected.assessment_capacity_bytes.min(512 * 1024 * 1024),
        retention_bytes: 64 * 1024,
        safety_reserve_bytes: 16 * 1024 * 1024,
    };
    let draft = ExecutionPlanner::prepare(
        &selected,
        &manifest,
        &definition,
        ComponentSelection {
            head: false,
            vision: false,
        },
        ExecutionPath::NativeMetal,
        PlannedMethod::Plain,
        KvCodec::Dense,
        limits,
        budget,
    )
    .unwrap();
    let state =
        ResourcePlanner::state_plan(&definition, draft.load(), KvCodec::Dense, limits, budget)
            .unwrap();
    let mut programs = AttestedPrograms::prepare_draft(&draft, &device).unwrap();
    let target_graphs = programs
        .prepare_target_graphs(&device, draft.load(), &definition.geometry, &state, limits)
        .unwrap();
    let target_readout_graphs = programs
        .prepare_target_readout_graphs(&device, draft.load(), &definition.geometry, limits)
        .unwrap();
    programs
        .prepare_auxiliary_graphs(
            &device,
            draft.load(),
            &definition,
            state.target_state(),
            state.head_state(),
            limits,
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
    Some(ExecutorDomain::with_family(
        execution,
        definition,
        resources,
        device,
        store,
        None,
        None,
        None,
        TestFamily { control },
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
        committed: 0,
    }
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

enum TestSubmission<L, W, O> {
    Ready(ReadySubmission<L, W, O>),
    Pending(PendingSubmission<L, W, O>),
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
    type TargetSubmission =
        TestSubmission<TargetLaunchCore, (), Option<crate::programs::TargetReadoutGraphResult>>;
    type HeadSubmission = PendingFailureSubmission<CompletedHeadWork>;
    type ProjectSubmission = PendingFailureSubmission<CompletedProjectWork>;
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
    fn submit_target(
        &mut self,
        launch: ValidatedTargetLaunch,
    ) -> Result<Self::TargetSubmission, (crate::SubmitError, ValidatedTargetLaunch)> {
        let core = launch.into_submission_parts();
        Ok(self.submit(core, (), None))
    }
    fn submit_head(
        &mut self,
        _: ValidatedHeadLaunch,
    ) -> Result<Self::HeadSubmission, (crate::SubmitError, ValidatedHeadLaunch)> {
        panic!("controlled fixture does not submit head work")
    }
    fn submit_project(
        &mut self,
        _: ValidatedProjectionLaunch,
    ) -> Result<Self::ProjectSubmission, (crate::SubmitError, ValidatedProjectionLaunch)> {
        panic!("controlled fixture does not submit projection work")
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

fn submit_reserved_target<F: ProgramFamily>(
    domain: &mut ExecutorDomain<F>,
    operations: Vec<Operation>,
) -> Result<TargetFlight<F::TargetSubmission>, DomainError> {
    let reservation = domain.reserve(operations)?;
    let (operations, resources) = reservation.into_parts();
    let ReservedResources::Target(reservation) = resources else {
        panic!("target operation produced another reservation lane")
    };
    domain.submit_target(operations, reservation)
}

#[test]
fn ready_target_can_abort_then_reconcile() {
    let Some(mut domain) = fixture(None) else {
        return;
    };
    let request = RequestId(1);
    domain.open(request).unwrap();
    let mut flight = submit_reserved_target(&mut domain, vec![forward(request, 0)]).unwrap();
    assert!(flight.completion().is_complete());
    assert!(domain.checkpoint(request).is_err());
    let pending = domain.finish_target(flight).unwrap().pop().unwrap();
    domain.abort(pending).unwrap();
    assert_eq!(domain.checkpoint(request).unwrap().target.position(), 0);
    let flight = submit_reserved_target(&mut domain, vec![forward(request, 0)]).unwrap();
    let pending = domain.finish_target(flight).unwrap().pop().unwrap();
    assert!(matches!(
        domain.reconcile(
            pending,
            PhysicalDecision {
                accepted_rows: 1,
                head_prefix: None
            }
        ),
        Ok(PhysicalResolution::Committed)
    ));
    assert_eq!(domain.checkpoint(request).unwrap().target.position(), 1);
}

#[test]
fn pending_target_request_cancellation_aborts_without_poisoning_then_device_failure_is_fatal() {
    let control = PendingControl::default();
    let Some(mut domain) = fixture(Some(control.clone())) else {
        return;
    };
    let request = RequestId(2);
    domain.open(request).unwrap();
    let mut flight = submit_reserved_target(&mut domain, vec![forward(request, 0)]).unwrap();
    assert!(!flight.completion().is_complete());
    assert!(domain.checkpoint(request).is_err());
    let woke = Arc::new(AtomicBool::new(false));
    let observed = woke.clone();
    flight.completion().notify(CompletionWake::new(move || {
        observed.store(true, Ordering::SeqCst)
    }));
    assert!(!woke.load(Ordering::SeqCst));
    control.resolve(Ok(()));
    assert!(woke.load(Ordering::SeqCst));
    let pending = domain.finish_target(flight).unwrap().pop().unwrap();
    domain.abort(pending).unwrap();
    assert!(domain.fatal_error().is_none());
    assert_eq!(domain.checkpoint(request).unwrap().target.position(), 0);
    let failed = PendingControl::default();
    let Some(mut failed_domain) = fixture(Some(failed.clone())) else {
        return;
    };
    let failed_request = RequestId(3);
    failed_domain.open(failed_request).unwrap();
    let flight =
        submit_reserved_target(&mut failed_domain, vec![forward(failed_request, 0)]).unwrap();
    failed.resolve(Err(crate::DeviceError::Execution(
        "injected target failure".into(),
    )));
    assert!(failed_domain.finish_target(flight).is_err());
    assert!(failed_domain.fatal_error().is_some());
    assert!(failed_domain.checkpoint(failed_request).is_err());
}

#[test]
fn pending_head_vision_and_state_device_failures_poison_the_domain_owner() {
    let failure = |lane: &'static str| {
        crate::DeviceError::Execution(format!("failed pending {lane} submission"))
    };

    let Some(mut head_domain) = fixture(None) else {
        return;
    };
    let head_control = PendingControl::default();
    let mut head = HeadFlight {
        requests: Vec::new(),
        submission: PendingFailureSubmission::<CompletedHeadWork>::new(head_control.clone()),
        started: Instant::now(),
    };
    assert!(!head.completion().is_complete());
    head_control.resolve(Err(failure("head")));
    assert!(matches!(
        head_domain.finish_head(head),
        Err(DomainError::Device(_))
    ));
    assert!(matches!(
        head_domain.fatal_error(),
        Some(DomainError::Device(_))
    ));

    let Some(mut vision_domain) = fixture(None) else {
        return;
    };
    let vision_control = PendingControl::default();
    let mut vision = VisionFlight {
        request: RequestId(41),
        image: ImageRef::new(vision_domain.resource_identity().clone(), 1).unwrap(),
        submission: PendingFailureSubmission::<CompletedVisionWork>::new(vision_control.clone()),
        started: Instant::now(),
    };
    assert!(!vision.completion().is_complete());
    vision_control.resolve(Err(failure("vision")));
    assert!(matches!(
        vision_domain.finish_vision(vision),
        Err(DomainError::Device(_))
    ));
    assert!(matches!(
        vision_domain.fatal_error(),
        Some(DomainError::Device(_))
    ));

    let Some(mut state_domain) = fixture(None) else {
        return;
    };
    let state_control = PendingControl::default();
    let mut state = StateFlight {
        request: RequestId(42),
        submission: PendingFailureSubmission::<CompletedStateWork>::new(state_control.clone()),
        started: Instant::now(),
        head_prefix: None,
    };
    assert!(!state.completion().is_complete());
    state_control.resolve(Err(failure("state")));
    assert!(matches!(
        state_domain.finish_repair(state),
        Err(DomainError::Device(_))
    ));
    assert!(matches!(
        state_domain.fatal_error(),
        Some(DomainError::Device(_))
    ));
}

#[test]
fn completed_head_and_vision_request_cancellation_restores_or_drops_without_poisoning() {
    let Some(mut domain) = fixture(None) else {
        return;
    };
    let request = RequestId(50);
    domain.open(request).unwrap();

    // A head advance is independent from the accepted target lane. The target
    // state remaining resident must not be mistaken for a conflicting owner.
    let head_source = domain.target.get(&request).unwrap().checkpoint().fork();
    let head_advance = OwnedStateAdvance::begin(head_source, 1).ok().unwrap();
    let head_features = FeatureRef::logical(
        domain.resource_identity().clone(),
        1,
        domain.definition.geometry.hidden as usize,
    )
    .unwrap();
    domain
        .abort(PendingOperationOutcome {
            request,
            outcome: Outcome::Head {
                features: head_features,
            },
            advance: Some(head_advance),
            rows: 1,
            committed_rows: 0,
            kind: WorkKind::Decode,
            physical_duration: Duration::ZERO,
            slot: None,
            conditioning: None,
            conditioning_slices: Vec::new(),
            image: None,
        })
        .unwrap();
    assert!(domain.fatal_error().is_none());
    assert_eq!(domain.target.get(&request).unwrap().position(), 0);
    assert_eq!(domain.head.remove(&request).unwrap().position(), 0);

    // Vision output is stateless until reconciliation installs it. Cancelling
    // the request simply drops the retained feature claim.
    let image = ImageRef::new(domain.resource_identity().clone(), 1).unwrap();
    let vision_features = FeatureRef::logical(
        domain.resource_identity().clone(),
        1,
        domain.definition.geometry.hidden as usize,
    )
    .unwrap();
    domain
        .abort(PendingOperationOutcome {
            request,
            outcome: Outcome::Encode {
                features: vision_features,
            },
            advance: None,
            rows: 0,
            committed_rows: 0,
            kind: WorkKind::Prefill,
            physical_duration: Duration::ZERO,
            slot: None,
            conditioning: None,
            conditioning_slices: Vec::new(),
            image: Some(image),
        })
        .unwrap();
    assert!(domain.fatal_error().is_none());
    assert_eq!(domain.target.get(&request).unwrap().position(), 0);
}
