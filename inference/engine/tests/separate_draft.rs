//! Real-device qualification of a separate draft (DFlash, DSpark, DFlash2):
//! its state transactions (checkpoint refusal while a draft transaction is in
//! flight, state copy into another request, restore) reproduce its proposals
//! exactly, and the header-only assessment charges exactly the weights a load
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
    RequestId, ResourceCapacity, ResourcePlanner, Sampling, SelectSpec, Shaping, TokenId, WorkKind,
};
use magnitude_scheduler::domain::{self as service_domain, DomainFlight};
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
/// and return the committed rows' features.
fn prefill(domain: &mut ExecutorDomain, request: RequestId, tokens: &[TokenId]) -> FeatureRows {
    let operation = Operation::Forward {
        request,
        kind: WorkKind::Prefill,
        tokens: tokens.to_vec(),
        position: 0,
        conditioning: None,
        demand: Demand::FEATURES,
        select: Vec::new(),
        committed: tokens.len(),
    };
    let groups = service_domain::group(domain, vec![operation]);
    let DomainFlight::Target(flight) = service_domain::submit_group(domain, &groups[0]).unwrap()
    else {
        panic!("a prefill runs on the target lane")
    };
    let mut features = None;
    for pending in domain.finish_target(flight).unwrap() {
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
    domain
        .read(&FeatureSpan::new(features.unwrap(), 0, tokens.len()).unwrap())
        .unwrap()
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

/// One draft transaction at draft position `position` entering `tokens`
/// (each paired with its row of `conditioning`), drafting `proposals` after
/// the last. A request refuses a checkpoint while it is in flight. Returns
/// the proposals.
fn transact(
    domain: &mut ExecutorDomain,
    request: RequestId,
    tokens: &[TokenId],
    conditioning: FeatureRows,
    position: usize,
    proposals: usize,
) -> Vec<(u32, u8)> {
    let entered = tokens.len();
    let operation = Operation::Head {
        request,
        tokens: tokens.to_vec(),
        conditioning,
        position,
        proposals: (0..proposals)
            .map(|proposal| greedy(position + entered + 1 + proposal))
            .collect(),
        form: DraftForm::Block,
    };
    let groups = service_domain::group(domain, vec![operation]);
    let DomainFlight::Head(flight) = service_domain::submit_group(domain, &groups[0]).unwrap()
    else {
        panic!("a draft transaction runs on the head lane")
    };
    assert!(
        domain.checkpoint(request).is_err(),
        "a checkpoint waits for the in-flight draft transaction"
    );
    let mut selected = Vec::new();
    for pending in domain.finish_head(flight).unwrap() {
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
    selected
}

/// Enter the prompt's pairs `(tokens[p + 1], feature p)` but the anchor's,
/// as a prefill's draft transaction does.
fn inject(domain: &mut ExecutorDomain, request: RequestId, tokens: &[TokenId], features: &FeatureRows) {
    let committed = tokens.len() - 1;
    let proposals = transact(
        domain,
        request,
        &tokens[1..committed],
        rows(features, 0..committed - 1),
        0,
        0,
    );
    assert!(proposals.is_empty());
}

/// The block draft anchored on `tokens`' last token.
fn draft(
    domain: &mut ExecutorDomain,
    request: RequestId,
    tokens: &[TokenId],
    features: &FeatureRows,
    proposals: usize,
) -> Vec<(u32, u8)> {
    let committed = tokens.len() - 1;
    transact(
        domain,
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
    let (mut domain, _) =
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
    domain.open(source).unwrap();
    let conditioning = prefill(&mut domain, source, &tokens[..committed]);
    inject(&mut domain, source, &tokens, &conditioning);
    let before_draft = domain.checkpoint(source).unwrap();
    let drafted = draft(&mut domain, source, &tokens, &conditioning, proposals);
    assert_eq!(drafted.len(), proposals);

    // A copy of the state before the draft drafts the same proposals.
    let copy = RequestId(2);
    domain.open_checkpoint(copy, &before_draft).unwrap();
    assert_eq!(
        draft(&mut domain, copy, &tokens, &conditioning, proposals),
        drafted
    );
    // Restoring the source to that state and drafting again does too.
    domain.restore(source, &before_draft).unwrap();
    assert_eq!(
        draft(&mut domain, source, &tokens, &conditioning, proposals),
        drafted
    );
    eprintln!("proposals after {committed} committed rows: {drafted:?}");
    for request in [source, copy] {
        domain.close(request).unwrap();
    }
    drop(before_draft);
    let charge = domain.reconcile_memory_charge(&[], &[]).unwrap();
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
    let (mut domain, _) =
        build_native_domain(&resolved.manifest, resolved.host.shared_package()).unwrap();
    let request = RequestId(1);
    domain.open(request).unwrap();
    let tokens = [1, 2, 3, 4, 5].map(TokenId);
    let conditioning = prefill(&mut domain, request, &tokens[..4]);
    inject(&mut domain, request, &tokens, &conditioning);
    draft(
        &mut domain,
        request,
        &tokens,
        &conditioning,
        resolved.manifest.model.method.proposals(),
    );
    let held = domain.checkpoint(request).unwrap();
    domain.close(request).unwrap();
    let charge = domain.reconcile_memory_charge(&[], &[&held]).unwrap();
    eprintln!("assessed {terms:?} graphs {graphs:?}\nloaded {charge:?}");
    assert_eq!(charge.unattributed, 0);
    assert_eq!(charge.target_weights, terms.target_weights, "target weights");
    assert_eq!(charge.optional_weights, terms.head_weights, "draft weights");
    assert!(
        charge.bound_constants <= graphs.binding_constant_bytes,
        "bound constants {} exceed the assessed {}",
        charge.bound_constants,
        graphs.binding_constant_bytes
    );
    assert!(charge.graph_pools <= graphs.total_bytes);
}
