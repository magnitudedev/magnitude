//! Real-device qualification of a separate draft (DFlash, DSpark, DFlash2):
//! its state transactions (resume-state refusal while a draft transaction is
//! in flight, resuming another request from that state, resuming the evicted
//! source from it) reproduce its proposals exactly, and the header-only
//! assessment charges exactly the weights a load
//! commits and bounds its graph holdings. Run on a host with a target and its
//! draft:
//! `MAGNITUDE_TEST_TARGET_GGUF=T.gguf MAGNITUDE_TEST_DRAFT_GGUF=D.gguf
//! [MAGNITUDE_TEST_DRAFT_METHOD=dflash|dspark|dflash2] cargo test
//! -p magnitude-engine --test separate_draft -- --ignored --nocapture`.

use magnitude_engine::{
    assessment::{
        prepare_model_assessment, AssessmentSetup, ModelPackagePaths, PreparedModelAssessment,
    },
    build_native_domain,
    composition::{EngineConfiguration, ResolvedEngineConfiguration},
    options::{
        standard_service_limits, ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection,
    },
};
use magnitude_executor::{
    platform::{DeviceRequest, MemoryReserves},
    AssessmentGraphResourceBounds, AssessmentMemoryTerms, Demand, DraftForm, ExecutionPath,
    ExecutorDomain, FeatureReader, FeatureRows, FeatureSpan, Operation, Outcome, PhysicalDecision,
    RequestId, ReservedResources, ResourceCapacity, ResourcePlanner, ResumeState, Sampling,
    SelectSpec, Shaping, StateBindings, TokenId, WorkKind,
};
use magnitude_family_contracts::PreparedModelInput;
use std::path::PathBuf;

fn path(variable: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(variable).unwrap_or_else(|| panic!("set {variable}")))
}

fn method() -> ModelMethod {
    match std::env::var("MAGNITUDE_TEST_DRAFT_METHOD").as_deref() {
        Err(_) | Ok("auto") => ModelMethod::Auto,
        Ok("dflash") => ModelMethod::DFlash,
        Ok("dspark") => ModelMethod::DSpark,
        Ok("dflash2") => ModelMethod::DFlash2,
        Ok(other) => panic!("unknown draft method {other}"),
    }
}

fn policy() -> ModelPolicy {
    ModelPolicy {
        method: method(),
        ..ModelPolicy::default()
    }
}

fn resolved() -> ResolvedEngineConfiguration {
    EngineConfiguration {
        package: PackageOptions {
            target: path("MAGNITUDE_TEST_TARGET_GGUF"),
            projector: ProjectorSelection::Disabled,
            draft: Some(path("MAGNITUDE_TEST_DRAFT_GGUF")),
        },
        model: policy(),
        context_tokens: None,
        service: standard_service_limits(),
        path: ExecutionPath::Native,
        device: DeviceRequest::Automatic,
        kernel_cache: std::env::var_os("MAGNITUDE_TEST_KERNEL_CACHE").map(PathBuf::from),
        reserves: MemoryReserves::standard(),
    }
    .resolve()
    .unwrap()
}

fn greedy(position: usize) -> SelectSpec {
    SelectSpec {
        sampling: Sampling::Greedy,
        seed: 0,
        position,
        domain: 0,
        mask: None,
        shaping: Shaping::default(),
        history: None,
    }
}

/// Prefill `tokens` for `request` with the features a draft conditions on,
/// and return the committed rows' features. The flight carries the bindings
/// and returns them.
fn prefill(
    domain: &mut ExecutorDomain,
    mut bindings: StateBindings,
    request: RequestId,
    tokens: &[TokenId],
) -> (FeatureRows, StateBindings) {
    let operation = Operation::Forward {
        request,
        kind: WorkKind::Prefill,
        tokens: tokens.to_vec(),
        position: 0,
        conditioning: None,
        demand: Demand::FEATURES,
        select: Vec::new(),
        committed: tokens.len(),
        prime: None,
    };
    let operations = [operation];
    let ReservedResources::Target(reservation) = domain
        .reserve(&mut bindings, &operations)
        .unwrap()
        .into_resources()
    else {
        panic!("a prefill runs on the target lane")
    };
    let flight = domain
        .submit_target(bindings, &operations, reservation)
        .unwrap_or_else(|failure| panic!("{}", failure.error()));
    let (pending, bindings) = domain.finish_target(flight).unwrap();
    let mut features = None;
    for pending in pending {
        let Outcome::Forward { rows } = pending.outcome().clone() else {
            panic!("a prefill returns forward rows")
        };
        features = rows.iter().find_map(|row| row.features.clone());
        domain
            .reconcile(
                pending,
                PhysicalDecision {
                    accepted_rows: tokens.len(),
                },
            )
            .unwrap();
    }
    let features = domain
        .read(&FeatureSpan::new(features.unwrap(), 0, tokens.len()).unwrap())
        .unwrap();
    (features, bindings)
}

/// Rows `rows` of `features`.
fn rows(features: &FeatureRows, rows: std::ops::Range<usize>) -> FeatureRows {
    let width = features.bytes().len() / features.rows();
    FeatureRows::new(
        features.bytes()[rows.start * width..rows.end * width]
            .to_vec()
            .into(),
        rows.len(),
    )
    .unwrap()
}

/// Install a request with no prompt rows and make it resident, fresh or from
/// `from`.
fn open(
    domain: &mut ExecutorDomain,
    bindings: &mut StateBindings,
    request: RequestId,
    from: Option<&ResumeState>,
) {
    domain
        .install_input(request, PreparedModelInput::continuation_only())
        .unwrap();
    domain.open_state(bindings, request, from).unwrap();
}

/// One draft transaction at draft position `position` entering `tokens`
/// (each paired with its row of `conditioning`), drafting `proposals` after
/// the last. A request refuses a resume state while it is in flight. Returns
/// the proposals and the bindings its flight carried.
fn transact(
    domain: &mut ExecutorDomain,
    mut bindings: StateBindings,
    request: RequestId,
    tokens: &[TokenId],
    conditioning: FeatureRows,
    position: usize,
    proposals: usize,
) -> (Vec<(u32, u8)>, StateBindings) {
    let entered = tokens.len();
    let operation = Operation::Head {
        request,
        phase: magnitude_executor::HeadPhase::Generation,
        tokens: tokens.to_vec(),
        conditioning,
        position,
        proposals: (0..proposals)
            .map(|proposal| greedy(position + entered + 1 + proposal))
            .collect(),
        form: DraftForm::Block,
    };
    let operations = [operation];
    let ReservedResources::Head(workspace, output, advances) = domain
        .reserve(&mut bindings, &operations)
        .unwrap()
        .into_resources()
    else {
        panic!("a draft transaction runs on the head lane")
    };
    let flight = domain
        .submit_head(bindings, &operations, workspace, output, advances)
        .unwrap_or_else(|failure| panic!("{}", failure.error()));
    assert!(
        domain.resume_state(request).is_err(),
        "a resume state waits for the in-flight draft transaction"
    );
    let (pending, bindings) = domain.finish_head(flight).unwrap();
    let mut selected = Vec::new();
    for pending in pending {
        let Outcome::Head { proposals } = pending.outcome().clone() else {
            panic!("a draft transaction returns proposals")
        };
        selected = proposals
            .iter()
            .map(|proposal| (proposal.token.0, proposal.status))
            .collect();
        domain
            .reconcile(
                pending,
                PhysicalDecision {
                    accepted_rows: entered,
                },
            )
            .unwrap();
    }
    (selected, bindings)
}

/// Enter the prompt's pairs `(tokens[p + 1], feature p)` but the anchor's,
/// as a prefill's draft transaction does.
fn inject(
    domain: &mut ExecutorDomain,
    bindings: StateBindings,
    request: RequestId,
    tokens: &[TokenId],
    features: &FeatureRows,
) -> StateBindings {
    let committed = tokens.len() - 1;
    let (proposals, bindings) = transact(
        domain,
        bindings,
        request,
        &tokens[1..committed],
        rows(features, 0..committed - 1),
        0,
        0,
    );
    assert!(proposals.is_empty());
    bindings
}

/// The block draft anchored on `tokens`' last token.
fn draft(
    domain: &mut ExecutorDomain,
    bindings: StateBindings,
    request: RequestId,
    tokens: &[TokenId],
    features: &FeatureRows,
    proposals: usize,
) -> (Vec<(u32, u8)>, StateBindings) {
    let committed = tokens.len() - 1;
    transact(
        domain,
        bindings,
        request,
        &tokens[committed..],
        rows(features, committed - 1..committed),
        committed - 1,
        proposals,
    )
}

#[test]
#[ignore = "requires a device, MAGNITUDE_TEST_TARGET_GGUF and MAGNITUDE_TEST_DRAFT_GGUF"]
fn draft_state_copies_and_restores_reproduce_its_proposals() {
    let resolved = resolved();
    let proposals = resolved.manifest.model.method.proposals();
    assert!(proposals > 0, "the load drafts");
    let (mut domain, mut bindings, _) =
        build_native_domain(&resolved.manifest, resolved.host.shared_package()).unwrap();
    let tokens = resolved
        .host
        .shared_tokenizer()
        .encode(
            "One, two, three, four, five, six, seven, eight, nine, ten, eleven",
            magnitude_chat::SpecialTokens::Recognize,
        )
        .unwrap();
    let committed = tokens.len() - 1;

    let source = RequestId(1);
    open(&mut domain, &mut bindings, source, None);
    let (conditioning, bindings) = prefill(&mut domain, bindings, source, &tokens[..committed]);
    let bindings = inject(&mut domain, bindings, source, &tokens, &conditioning);
    let before_draft = domain.resume_state(source).unwrap();
    let (drafted, mut bindings) =
        draft(&mut domain, bindings, source, &tokens, &conditioning, proposals);
    assert_eq!(drafted.len(), proposals);

    // Another request resumed from the state before the draft drafts the
    // same proposals.
    let copy = RequestId(2);
    open(&mut domain, &mut bindings, copy, Some(&before_draft));
    let (copied, mut bindings) =
        draft(&mut domain, bindings, copy, &tokens, &conditioning, proposals);
    assert_eq!(copied, drafted);
    // Evicting the source and resuming it from that state drafts them too.
    domain.release_state(&[source]).unwrap();
    domain
        .open_state(&mut bindings, source, Some(&before_draft))
        .unwrap();
    let (resumed, _) = draft(&mut domain, bindings, source, &tokens, &conditioning, proposals);
    assert_eq!(resumed, drafted);
    eprintln!("proposals after {committed} committed rows: {drafted:?}");
    for request in [source, copy] {
        domain.close(request).unwrap();
    }
    drop(before_draft);
    let charge = domain.reconcile_memory_charge(&[]).unwrap();
    assert_eq!(charge.unattributed, 0, "{charge:?}");
}

#[test]
#[ignore = "requires a device, MAGNITUDE_TEST_TARGET_GGUF and MAGNITUDE_TEST_DRAFT_GGUF"]
fn header_only_assessment_charges_what_a_load_commits() {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let setup = AssessmentSetup::discover(
        &catalog,
        DeviceRequest::Automatic,
        MemoryReserves::standard(),
        ModelPolicy::default(),
        standard_service_limits(),
    )
    .unwrap();
    let prepared = prepare_model_assessment(
        &ModelPackagePaths {
            target: path("MAGNITUDE_TEST_TARGET_GGUF"),
            projector: None,
            draft: Some(path("MAGNITUDE_TEST_DRAFT_GGUF")),
            method: method(),
        },
        &setup,
    )
    .unwrap();
    let PreparedModelAssessment::Planned { draft: plan, .. } = prepared else {
        panic!("the bundle plans")
    };

    let resolved = resolved();
    let definition = &resolved.manifest.definition;
    let policy = plan.policy();
    let terms = AssessmentMemoryTerms::derive(
        definition,
        plan.load(),
        policy.selection(),
        policy.codec(),
        policy.method(),
        policy.limits(),
    )
    .unwrap();
    let state = ResourcePlanner::state_plan(
        definition,
        plan.load(),
        policy.method(),
        policy.codec(),
        policy.limits(),
        ResourceCapacity {
            domain_bytes: plan.device().assessment_capacity_bytes(),
            tensor_operations: magnitude_executor::TensorOperations::of(
                plan.device().tensor_operations(),
            ),
        },
    )
    .unwrap();
    let graphs = AssessmentGraphResourceBounds::derive(
        definition,
        plan.load(),
        &state,
        policy.method(),
        policy.codec(),
        policy.limits(),
        plan.device().backend(),
    )
    .unwrap();

    // Load the target and, through one draft transaction, the draft.
    let (mut domain, mut bindings, _) =
        build_native_domain(&resolved.manifest, resolved.host.shared_package()).unwrap();
    let request = RequestId(1);
    open(&mut domain, &mut bindings, request, None);
    let tokens = [1, 2, 3, 4, 5].map(TokenId);
    let (conditioning, bindings) = prefill(&mut domain, bindings, request, &tokens[..4]);
    let bindings = inject(&mut domain, bindings, request, &tokens, &conditioning);
    draft(
        &mut domain,
        bindings,
        request,
        &tokens,
        &conditioning,
        resolved.manifest.model.method.proposals(),
    );
    let held = domain.resume_state(request).unwrap();
    domain.close(request).unwrap();
    let charge = domain.reconcile_memory_charge(&[&held]).unwrap();
    eprintln!("assessed {terms:?} graphs {graphs:?}\nloaded {charge:?}");
    assert_eq!(charge.unattributed, 0);
    assert_eq!(
        charge.target_weights, terms.target_weights,
        "target weights"
    );
    assert_eq!(charge.optional_weights, terms.head_weights, "draft weights");
    assert!(
        charge.bound_constants <= graphs.binding_constant_bytes,
        "bound constants {} exceed the assessed {}",
        charge.bound_constants,
        graphs.binding_constant_bytes
    );
    assert!(charge.graph_pools <= graphs.total_bytes);
}
