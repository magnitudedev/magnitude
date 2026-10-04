//! A verification commits its accepted prefix as a recurrent tape version:
//! the successor bank holds the state after the committed rows, and the next
//! advance replays the accepted tape rows before its own. Continuing from
//! that version must be bit-identical to continuing from a run that
//! published its state exactly at the accepted prefix.
//!
//! Requires a GPU and an MTP GGUF of a recurrent (Qwen3.5) model:
//! `MAGNITUDE_TEST_MTP_GGUF=<model.gguf> [MAGNITUDE_TEST_KERNEL_CACHE=<dir>]
//! cargo test -p magnitude-engine --release --test tape_versions -- --ignored`.

use magnitude_engine::{
    build_native_domain,
    composition::EngineConfiguration,
    options::{ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection},
};
use magnitude_executor::{
    platform::{DeviceRequest, MemoryReserves},
    Demand, ExecutionPath, ExecutorDomain, Operation, Outcome, PhysicalDecision, RequestId,
    ReservedResources, RowResult, StateBindings, TokenId, WorkKind,
};
use magnitude_family_contracts::PreparedModelInput;
use magnitude_scheduler::ServiceLimits;
use magnitude_state::KvCodec;
use std::path::PathBuf;

/// Draft rows the plan records on the tape; every verification below has
/// this many rows after its anchor.
const PROPOSALS: u8 = 4;
const PROMPT: usize = 40;

fn open_domain() -> Option<(ExecutorDomain, StateBindings, usize)> {
    let model = PathBuf::from(std::env::var_os("MAGNITUDE_TEST_MTP_GGUF")?);
    let resolved = EngineConfiguration {
        package: PackageOptions {
            target: model,
            projector: ProjectorSelection::Disabled,
            draft: None,
        },
        model: ModelPolicy {
            method: ModelMethod::Mtp,
            mtp_proposals: Some(PROPOSALS),
            kv_codec: KvCodec::AffineK8V4,
            lookahead: false,
            exported_logits_rows: 64,
            error_classes: Vec::new(),
        },
        context_tokens: Some(1024),
        service: ServiceLimits {
            prefill_tokens: 64,
            decode_tokens: 16,
            decode_share: 0.5,
            locality_seconds: 1.0,
        },
        path: ExecutionPath::Native,
        device: DeviceRequest::Automatic,
        kernel_cache: std::env::var_os("MAGNITUDE_TEST_KERNEL_CACHE").map(PathBuf::from),
        reserves: MemoryReserves::standard(),
    }
    .resolve()
    .unwrap();
    let vocabulary = resolved.manifest.definition.decoder.vocabulary as usize;
    let package = resolved.host.shared_package();
    let (domain, bindings, _) = build_native_domain(&resolved.manifest, package).unwrap();
    Some((domain, bindings, vocabulary))
}

/// A fixed spread over ordinary vocabulary rows.
fn tokens(vocabulary: usize, start: usize, rows: usize) -> Vec<TokenId> {
    let span = (vocabulary - 1024) as u64;
    (start..start + rows)
        .map(|position| TokenId(((position as u64 * 2_654_435_761 + 12_345) % span + 512) as u32))
        .collect()
}

struct Sequence {
    request: RequestId,
    position: usize,
}

/// Run one forward of `tokens` whose first `committed` rows always commit,
/// accept `accepted` rows, and return its row results. The flight carries
/// the bindings and returns them.
fn advance(
    domain: &mut ExecutorDomain,
    mut bindings: StateBindings,
    sequence: &mut Sequence,
    kind: WorkKind,
    tokens: Vec<TokenId>,
    demand: Demand,
    committed: usize,
    accepted: usize,
) -> (Vec<RowResult>, StateBindings) {
    let operation = Operation::Forward {
        request: sequence.request,
        kind,
        tokens,
        position: sequence.position,
        conditioning: None,
        demand,
        select: Vec::new(),
        committed,
        prime: None,
    };
    let operations = [operation];
    let ReservedResources::Target(reservation) = domain
        .reserve(&mut bindings, &operations)
        .unwrap()
        .into_resources()
    else {
        panic!("a forward runs on the target lane");
    };
    let flight = domain
        .submit_target(bindings, &operations, reservation)
        .unwrap_or_else(|failure| panic!("{}", failure.error()));
    let (mut pending, bindings) = domain.finish_target(flight).unwrap();
    let pending = pending.pop().unwrap();
    let Outcome::Forward { rows } = pending.outcome().clone() else {
        panic!("a forward returns forward rows");
    };
    domain
        .reconcile(
            pending,
            PhysicalDecision {
                accepted_rows: accepted,
            },
        )
        .unwrap();
    sequence.position += accepted;
    (rows, bindings)
}

/// Decode `steps` forced rows and return each step's logits.
fn continuation(
    domain: &mut ExecutorDomain,
    mut bindings: StateBindings,
    sequence: &mut Sequence,
    vocabulary: usize,
    steps: usize,
) -> (Vec<Vec<u32>>, StateBindings) {
    let mut logits = Vec::with_capacity(steps);
    for _ in 0..steps {
        let token = tokens(vocabulary, sequence.position, 1);
        let (rows, next) = advance(
            domain,
            bindings,
            sequence,
            WorkKind::Decode,
            token,
            Demand::LOGITS,
            1,
            1,
        );
        bindings = next;
        logits.push(
            rows[0]
                .logits
                .as_ref()
                .expect("a logits decode exports its row")
                .read_to_host()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect(),
        );
    }
    (logits, bindings)
}

/// Verify a speculative round of the anchor plus `PROPOSALS` rows. `taped`
/// commits the anchor and records the rest on the tape; the other publishes
/// its state exactly at the accepted prefix. Both accept `accepted` rows.
fn round(
    domain: &mut ExecutorDomain,
    bindings: StateBindings,
    sequence: &mut Sequence,
    vocabulary: usize,
    taped: bool,
    accepted: usize,
) -> StateBindings {
    let rows = 1 + usize::from(PROPOSALS);
    let committed = if taped { 1 } else { accepted };
    let tokens = tokens(vocabulary, sequence.position, rows);
    advance(
        domain,
        bindings,
        sequence,
        WorkKind::Replay,
        tokens,
        Demand::NONE,
        committed,
        accepted,
    )
    .1
}

#[test]
#[ignore = "requires a GPU and MAGNITUDE_TEST_MTP_GGUF"]
fn tape_versions_continue_exactly_like_runs_that_stopped_there() {
    let Some((mut domain, mut bindings, vocabulary)) = open_domain() else {
        panic!("set MAGNITUDE_TEST_MTP_GGUF to an MTP GGUF of a recurrent model");
    };
    let mut open = |bindings: &mut StateBindings, id| {
        let request = RequestId(id);
        domain
            .install_input(request, PreparedModelInput::continuation_only())
            .unwrap();
        domain.open_state(bindings, request, None).unwrap();
        Sequence {
            request,
            position: 0,
        }
    };
    let (mut taped, mut stopped) = (open(&mut bindings, 1), open(&mut bindings, 2));
    for sequence in [&mut taped, &mut stopped] {
        let prompt = tokens(vocabulary, 0, PROMPT);
        bindings = advance(
            &mut domain,
            bindings,
            sequence,
            WorkKind::Prefill,
            prompt,
            Demand::NONE,
            PROMPT,
            PROMPT,
        )
        .1;
    }
    // Rounds accepting every count the tape can hold, including a version
    // read by the next round (a round directly after a round).
    for (accepted, steps) in [(3, 2), (2, 0), (5, 1), (1, 2)] {
        bindings = round(&mut domain, bindings, &mut taped, vocabulary, true, accepted);
        bindings = round(&mut domain, bindings, &mut stopped, vocabulary, false, accepted);
        assert_eq!(taped.position, stopped.position);
        let (expected, next) = continuation(&mut domain, bindings, &mut stopped, vocabulary, steps);
        let (actual, next) = continuation(&mut domain, next, &mut taped, vocabulary, steps);
        bindings = next;
        assert!(
            expected == actual,
            "continuation after a {accepted}-row prefix differs from the stopped run"
        );
    }
    // A last continuation straight after a round.
    bindings = round(&mut domain, bindings, &mut taped, vocabulary, true, 4);
    bindings = round(&mut domain, bindings, &mut stopped, vocabulary, false, 4);
    let (expected, bindings) = continuation(&mut domain, bindings, &mut stopped, vocabulary, 3);
    let (actual, _) = continuation(&mut domain, bindings, &mut taped, vocabulary, 3);
    assert!(
        expected == actual,
        "continuation after a 4-row prefix differs"
    );
    domain.close(taped.request).unwrap();
    domain.close(stopped.request).unwrap();
}
