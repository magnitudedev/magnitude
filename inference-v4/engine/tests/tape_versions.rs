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
use magnitude_model_executor::{
    platform::{DeviceRequest, MemoryReserves},
    Demand, ExecutionPath, ExecutorDomain, Operation, Outcome,
    PhysicalDecision, RequestId, RowResult, TokenId, WorkKind,
};
use magnitude_model_state::KvCodec;
use magnitude_service::{
    domain::{self as service_domain, DomainFlight},
    ServiceLimits,
};
use std::path::PathBuf;

/// Draft rows the plan records on the tape; every verification below has
/// this many rows after its anchor.
const PROPOSALS: u8 = 4;
const PROMPT: usize = 40;

fn open_domain() -> Option<(ExecutorDomain, usize)> {
    let model = PathBuf::from(std::env::var_os("MAGNITUDE_TEST_MTP_GGUF")?);
    let resolved = EngineConfiguration {
        package: PackageOptions {
            target: model,
            projector: ProjectorSelection::Disabled,
        },
        model: ModelPolicy {
            method: ModelMethod::Mtp,
            mtp_proposals: Some(PROPOSALS),
            kv_codec: KvCodec::AffineK8V4,
            lookahead: false,
        },
        context_tokens: Some(1024),
        service: ServiceLimits {
            max_requests: 2,
            max_batch: 2,
            prefill_tokens: 64,
            decode_tokens: 16,
            decode_share: 0.5,
            locality_seconds: 1.0,
        },
        path: ExecutionPath::Native,
        device: DeviceRequest::Automatic,
        control_capacity: 64,
        kernel_cache: std::env::var_os("MAGNITUDE_TEST_KERNEL_CACHE").map(PathBuf::from),
        reserves: MemoryReserves::standard(),
    }
    .resolve()
    .unwrap();
    let vocabulary = resolved.manifest.definition.geometry.vocabulary as usize;
    let package = resolved.artifacts.shared_package();
    let (domain, _) = build_native_domain(&resolved.manifest, package).unwrap();
    Some((domain, vocabulary))
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
/// accept `accepted` rows, and return its row results.
fn advance(
    domain: &mut ExecutorDomain,
    sequence: &mut Sequence,
    kind: WorkKind,
    tokens: Vec<TokenId>,
    demand: Demand,
    committed: usize,
    accepted: usize,
) -> Vec<RowResult> {
    let operation = Operation::Forward {
        request: sequence.request,
        kind,
        tokens,
        position: sequence.position,
        conditioning: None,
        demand,
        select: Vec::new(),
        committed,
    };
    let groups = service_domain::group(domain, vec![operation]);
    let [group] = groups.as_slice() else {
        panic!("one forward forms one group");
    };
    let DomainFlight::Target(flight) = service_domain::submit_group(domain, group).unwrap() else {
        panic!("a forward runs on the target lane");
    };
    let pending = domain.finish_target(flight).unwrap().pop().unwrap();
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
    rows
}

/// Decode `steps` forced rows and return each step's logits.
fn continuation(
    domain: &mut ExecutorDomain,
    sequence: &mut Sequence,
    vocabulary: usize,
    steps: usize,
) -> Vec<Vec<u32>> {
    (0..steps)
        .map(|_| {
            let token = tokens(vocabulary, sequence.position, 1);
            let rows = advance(
                domain,
                sequence,
                WorkKind::Decode,
                token,
                Demand::LOGITS,
                1,
                1,
            );
            rows[0]
                .logits
                .as_ref()
                .expect("a logits decode exports its row")
                .read_to_host()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect()
        })
        .collect()
}

/// Verify a speculative round of the anchor plus `PROPOSALS` rows. `taped`
/// commits the anchor and records the rest on the tape; the other publishes
/// its state exactly at the accepted prefix. Both accept `accepted` rows.
fn round(
    domain: &mut ExecutorDomain,
    sequence: &mut Sequence,
    vocabulary: usize,
    taped: bool,
    accepted: usize,
) {
    let rows = 1 + usize::from(PROPOSALS);
    let committed = if taped { 1 } else { accepted };
    let tokens = tokens(vocabulary, sequence.position, rows);
    advance(
        domain,
        sequence,
        WorkKind::Replay,
        tokens,
        Demand::NONE,
        committed,
        accepted,
    );
}

#[test]
#[ignore = "requires a GPU and MAGNITUDE_TEST_MTP_GGUF"]
fn tape_versions_continue_exactly_like_runs_that_stopped_there() {
    let Some((mut domain, vocabulary)) = open_domain() else {
        panic!("set MAGNITUDE_TEST_MTP_GGUF to an MTP GGUF of a recurrent model");
    };
    let mut open = |id| {
        let request = RequestId(id);
        domain.open(request).unwrap();
        Sequence {
            request,
            position: 0,
        }
    };
    let (mut taped, mut stopped) = (open(1), open(2));
    for sequence in [&mut taped, &mut stopped] {
        let prompt = tokens(vocabulary, 0, PROMPT);
        advance(
            &mut domain,
            sequence,
            WorkKind::Prefill,
            prompt,
            Demand::NONE,
            PROMPT,
            PROMPT,
        );
    }
    // Rounds accepting every count the tape can hold, including a version
    // read by the next round (a round directly after a round).
    for (accepted, steps) in [(3, 2), (2, 0), (5, 1), (1, 2)] {
        round(&mut domain, &mut taped, vocabulary, true, accepted);
        round(&mut domain, &mut stopped, vocabulary, false, accepted);
        assert_eq!(taped.position, stopped.position);
        let expected = continuation(&mut domain, &mut stopped, vocabulary, steps);
        let actual = continuation(&mut domain, &mut taped, vocabulary, steps);
        assert!(
            expected == actual,
            "continuation after a {accepted}-row prefix differs from the stopped run"
        );
    }
    // A last continuation straight after a round.
    round(&mut domain, &mut taped, vocabulary, true, 4);
    round(&mut domain, &mut stopped, vocabulary, false, 4);
    let expected = continuation(&mut domain, &mut stopped, vocabulary, 3);
    let actual = continuation(&mut domain, &mut taped, vocabulary, 3);
    assert!(
        expected == actual,
        "continuation after a 4-row prefix differs"
    );
    domain.close(taped.request).unwrap();
    domain.close(stopped.request).unwrap();
}
