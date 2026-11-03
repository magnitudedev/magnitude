//! Real-device qualification of prompt-only readout plans. The executor: a
//! row's logits readout equals that row's full next-token logits at the
//! readout's tokens, for several readout rows of one chunk and for readout
//! rows of several sequences in one launch. The owner: a plan's sequences
//! answer as each does alone while their shared rows are computed once, and
//! retained shared rows serve a later plan without being computed again.
//! Attention blocks: a blocked sequence answers alike in one launch and
//! across launches, and the trained layouts answer as their rows read; a
//! unit wider than any launch answers in chunks as in one launch, and a
//! small unit reading its launch apart reads its launch once written. The
//! readout head: a row's head values are the product of the model's
//! `decision.readout.weight` with that row's final normalized hidden state,
//! in the element the head is stored in. Scores and media plans read what
//! the prefill path computes. A causal model refuses blocked plans.
//!
//! `MAGNITUDE_TEST_GGUF=M.gguf MAGNITUDE_TEST_CAUSAL_GGUF=C.gguf cargo test
//! --release -p magnitude-engine --test readout -- --ignored --nocapture
//! --test-threads=1`. The models' output projections must be stored packed
//! (the attested selected-row projection); a load that reads no logits
//! refuses readout plans. `M`'s attention must be trained non-causal (it
//! carries a decision description with `<arch>.attention.causal` false), the only
//! models that compute a unit wider than one launch; `C`'s is causal.
//! Every path is compared against an independent reference: the same rows
//! of one sequence in one dense launch of the executor, within a bound
//! derived from that computation's own rounding ([`Reference`]); values a
//! deterministic computation repeats are compared bit for bit. Every load
//! names its history codec, overriding the model's default: a test's own
//! history is compressed unless `MAGNITUDE_TEST_KV=dense`, and its dense
//! references are dense. `M` must store a readout head;
//! `MAGNITUDE_TEST_HEAD_GGUFS` lists (`:`-separated) models whose heads the
//! executor test reads, one per stored element.

use magnitude_chat::{ByteBpeTokenizer, SpecialTokens, request::ImageInput};
use magnitude_engine::{
    build_native_domain,
    composition::{EngineConfiguration, ResolvedEngineConfiguration},
    options::{ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection},
};
use magnitude_executor::{
    Demand, ExecutionPath, ExecutorDomain, FeatureReader, FeatureSpan, Operation, Outcome,
    PhysicalDecision, RequestId, ReservedResources, Sampling as RowSampling, SelectSpec,
    Shaping as RowShaping, StateBindings, TokenId, WorkKind, answer_row,
    platform::{DeviceRequest, MemoryReserves},
};
use magnitude_family_contracts::{InputLayout, PreparedModelInput, TokenPlan};
use magnitude_readout::{
    Block, Readout, ReadoutKind, ReadoutPlan, ReadoutSequence, ReadoutValues, Visibility,
};
use magnitude_scheduler::{
    ServiceLimits,
    owner::{AdmissionError, Owner},
    prefix_cache::PrefixCacheCapacity,
    publication::RequestError,
};
use magnitude_state::KvCodec;
use owner_host::{Host, Stream};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

mod owner_host;

/// Prefill rows per launch: small, so the shared rows span several chunks.
const PREFILL: usize = 16;

/// The largest difference between a head readout and the CPU product of the
/// stored head with the row's features, relative to the row's largest
/// value. Both read the same BF16 features in F32, summed in another order;
/// a BF16 head's rounding may then differ by one BF16 step (2^-8).
const HEAD_TOLERANCE: f32 = 4e-3;

fn test_gguf() -> PathBuf {
    PathBuf::from(std::env::var_os("MAGNITUDE_TEST_GGUF").expect("set MAGNITUDE_TEST_GGUF"))
}

fn causal_gguf() -> PathBuf {
    PathBuf::from(
        std::env::var_os("MAGNITUDE_TEST_CAUSAL_GGUF").expect("set MAGNITUDE_TEST_CAUSAL_GGUF"),
    )
}

/// Context tokens of a load whose sequences fit one launch of the largest
/// class.
const CONTEXT: usize = 1024;

fn resolved(exported_logits_rows: usize) -> ResolvedEngineConfiguration {
    resolved_at(test_gguf(), exported_logits_rows)
}

fn resolved_at(target: PathBuf, exported_logits_rows: usize) -> ResolvedEngineConfiguration {
    resolved_with(
        target,
        ProjectorSelection::Disabled,
        exported_logits_rows,
        test_codec(),
        PREFILL,
        CONTEXT,
    )
}

/// A load of `target` with `projector`, whose history uses `kv_codec`,
/// whose prompt chunks take at most `prefill` rows and whose sequences hold
/// `context` tokens.
fn resolved_with(
    target: PathBuf,
    projector: ProjectorSelection,
    exported_logits_rows: usize,
    kv_codec: KvCodec,
    prefill: usize,
    context: usize,
) -> ResolvedEngineConfiguration {
    let mut resolved = EngineConfiguration {
        package: PackageOptions {
            target,
            projector,
            draft: None,
        },
        model: ModelPolicy {
            draft_policy: None,
            method: ModelMethod::Plain,
            mtp_proposals: None,
            kv_codec: Some(kv_codec),
            lookahead: false,
            exported_logits_rows,
            numerical_profile: None,
        },
        context_tokens: Some(context),
        service: limits_with(prefill),
        path: ExecutionPath::Native,
        device: DeviceRequest::Automatic,
        kernel_cache: std::env::var_os("MAGNITUDE_TEST_KERNEL_CACHE").map(PathBuf::from),
        reserves: MemoryReserves::standard(),
    }
    .resolve()
    .unwrap();
    // These tests qualify every readout kind on one load, whatever its
    // decision description plans: each load reads logits of chosen tokens
    // and scores, as a model whose readouts read them does
    // (`DecisionBinding::reads_logits`, `DecisionBinding::reads_scores`).
    let decision = resolved.manifest.definition.decision.as_mut().unwrap();
    decision.reads_logits = true;
    decision.reads_scores = true;
    resolved
}

fn limits_with(prefill: usize) -> ServiceLimits {
    ServiceLimits {
        prefill_tokens: prefill,
        decode_tokens: 32,
        decode_share: 0.5,
        locality_seconds: 1.0,
    }
}

fn text(tokenizer: &ByteBpeTokenizer, text: &str) -> Vec<TokenId> {
    tokenizer.encode(text, SpecialTokens::Literal).unwrap()
}

fn input(tokens: &[TokenId]) -> PreparedModelInput {
    PreparedModelInput::from_text_coordinates(
        TokenPlan::new(
            tokens.to_vec(),
            InputLayout::new(tokens.len(), vec![]).unwrap(),
        )
        .unwrap(),
        (0..tokens.len()).map(|row| [row as i32; 3]).collect(),
    )
    .unwrap()
}

fn logits(row: usize, tokens: &[TokenId]) -> Readout {
    Readout {
        row,
        kind: ReadoutKind::Logits {
            tokens: tokens.into(),
        },
    }
}

/// Tokens of short answer words, distinct and ascending-unordered as a
/// caller might list them.
fn choices(tokenizer: &ByteBpeTokenizer) -> Vec<TokenId> {
    let mut tokens = Vec::new();
    for word in [" yes", " no", " the", " river", " 7", "."] {
        let encoded = text(tokenizer, word);
        if !tokens.contains(&encoded[0]) {
            tokens.push(encoded[0]);
        }
    }
    tokens
}

/// The largest difference between `values` and the full logits `row` at
/// `tokens`, relative to the row's largest magnitude at those tokens.
fn relative_error(values: &[f32], row: &[f32], tokens: &[TokenId]) -> f32 {
    assert_eq!(values.len(), tokens.len());
    let scale = tokens
        .iter()
        .map(|token| row[token.0 as usize].abs())
        .fold(1.0_f32, f32::max);
    values
        .iter()
        .zip(tokens)
        .map(|(value, token)| (value - row[token.0 as usize]).abs() / scale)
        .fold(0.0, f32::max)
}

// --- executor --------------------------------------------------------------

struct Domain {
    domain: ExecutorDomain,
    bindings: Option<StateBindings>,
    next: u64,
}

impl Domain {
    fn of(resolved: &ResolvedEngineConfiguration) -> Self {
        let (domain, bindings, _) =
            build_native_domain(&resolved.manifest, resolved.host.shared_package()).unwrap();
        Self {
            domain,
            bindings: Some(bindings),
            next: 1,
        }
    }

    fn open(&mut self, tokens: &[TokenId]) -> RequestId {
        self.open_visible(tokens, Visibility::causal())
    }

    fn open_visible(&mut self, tokens: &[TokenId], visibility: Visibility) -> RequestId {
        self.open_input(input(tokens), visibility)
    }

    /// Open `input`, whose rows read as `visibility` says, and run the
    /// encodes its residency asks for.
    fn open_input(&mut self, input: PreparedModelInput, visibility: Visibility) -> RequestId {
        let request = RequestId(self.next);
        self.next += 1;
        self.domain
            .install_visible_input(request, input, visibility)
            .unwrap();
        let encodes = self
            .domain
            .open_state(self.bindings.as_mut().unwrap(), request, None)
            .unwrap();
        for encode in encodes {
            let mut bindings = self.bindings.take().unwrap();
            let ReservedResources::Vision(workspace, output) = self
                .domain
                .reserve(&mut bindings, std::slice::from_ref(&encode))
                .unwrap()
                .into_resources()
            else {
                panic!("an encode runs on the vision lane");
            };
            let flight = self
                .domain
                .submit_vision(bindings, &encode, workspace, output)
                .unwrap_or_else(|failure| panic!("{}", failure.error()));
            let (pending, bindings) = self.domain.finish_vision(flight).unwrap();
            self.bindings = Some(bindings);
            self.domain
                .reconcile(pending, PhysicalDecision { accepted_rows: 0 })
                .unwrap();
        }
        request
    }

    /// Submit one launch, commit every row, and return its outcomes.
    fn step(&mut self, operations: Vec<Operation>) -> Vec<Outcome> {
        let mut bindings = self.bindings.take().unwrap();
        let ReservedResources::Target(reservation) = self
            .domain
            .reserve(&mut bindings, &operations)
            .unwrap()
            .into_resources()
        else {
            panic!("a forward runs on the target lane");
        };
        let flight = self
            .domain
            .submit_target(bindings, &operations, reservation)
            .unwrap_or_else(|failure| panic!("{}", failure.error()));
        let (pending, bindings) = self.domain.finish_target(flight).unwrap();
        self.bindings = Some(bindings);
        pending
            .into_iter()
            .map(|pending| {
                let outcome = pending.outcome().clone();
                let rows = pending.rows();
                self.domain
                    .reconcile(
                        pending,
                        PhysicalDecision {
                            accepted_rows: rows,
                        },
                    )
                    .unwrap();
                outcome
            })
            .collect()
    }

    /// The full next-token logits of the last row of `tokens`, from a fresh
    /// sequence prefilled in one chunk.
    fn reference(&mut self, tokens: &[TokenId]) -> Vec<f32> {
        let request = self.open(tokens);
        let outcomes = self.step(vec![Operation::Forward {
            request,
            kind: WorkKind::Prefill,
            tokens: tokens.to_vec(),
            position: 0,
            conditioning: None,
            demand: Demand::SELECT | Demand::LOGITS,
            select: vec![SelectSpec {
                sampling: RowSampling::Greedy,
                seed: 0,
                position: tokens.len() - 1,
                domain: 0,
                mask: None,
                shaping: RowShaping::default(),
                history: None,
            }],
            committed: tokens.len(),
            prime: None,
            readouts: Vec::new(),
        }]);
        self.domain.close(request).unwrap();
        let Outcome::Forward { rows } = &outcomes[0] else {
            panic!("a forward returned {:?}", outcomes[0]);
        };
        rows.last()
            .unwrap()
            .logits
            .as_ref()
            .expect("the last row exports its logits")
            .read_to_host()
            .unwrap()
    }

    /// Open `input`, run the encodes its residency asks for, and prefill it
    /// in chunks of at most `PREFILL` rows that its layout allows, as
    /// generation's prefill path does: every row's features, as F32.
    fn prefill_features(&mut self, input: &PreparedModelInput) -> Vec<Vec<f32>> {
        let request = self.open_input(input.clone(), Visibility::causal());
        let tokens = input.tokens();
        let mut features = Vec::new();
        let mut position = 0;
        while position < tokens.len() {
            let end = input
                .layout()
                .chunk_end(position, tokens.len(), PREFILL)
                .unwrap();
            let outcomes = self.step(vec![Operation::Forward {
                request,
                kind: WorkKind::Prefill,
                tokens: tokens[position..end].to_vec(),
                position,
                conditioning: None,
                demand: Demand::FEATURES,
                select: Vec::new(),
                committed: end - position,
                prime: None,
                readouts: Vec::new(),
            }]);
            let Outcome::Forward { rows } = &outcomes[0] else {
                panic!("a prefill chunk returned {:?}", outcomes[0]);
            };
            let rows = self
                .domain
                .read(&FeatureSpan {
                    features: rows[0]
                        .features
                        .clone()
                        .expect("every row publishes features"),
                    start: 0,
                    count: end - position,
                })
                .unwrap();
            features.extend(rows.bytes().chunks_exact(rows.row_bytes()).map(bf16_values));
            position = end;
        }
        self.domain.close(request).unwrap();
        features
    }

    fn readout(&mut self, tokens: &[TokenId], readouts: Vec<Readout>) -> (RequestId, Operation) {
        let request = self.open(tokens);
        (
            request,
            Operation::Forward {
                request,
                kind: WorkKind::Prefill,
                tokens: tokens.to_vec(),
                position: 0,
                conditioning: None,
                demand: Demand::READOUT,
                select: Vec::new(),
                committed: tokens.len(),
                prime: None,
                readouts,
            },
        )
    }
}

/// Readout values are the rows' logits at their tokens: two readout rows of
/// one chunk, and a readout row of a second sequence in the same launch.
#[test]
#[ignore = "requires a CUDA or Metal device and MAGNITUDE_TEST_GGUF"]
fn readout_values_are_the_rows_logits_at_their_tokens() {
    let resolved = resolved(1);
    let tokenizer = resolved.host.shared_tokenizer();
    let mut domain = Domain::of(&resolved);
    let prompt = text(&tokenizer, "The ferry left at dawn. Did it arrive? Answer:");
    assert!(prompt.len() <= PREFILL, "{} prompt tokens", prompt.len());
    let mid = prompt.len() / 2;
    let choices = choices(&tokenizer);
    let few = &choices[..3];

    let at_mid = domain.reference(&prompt[..=mid]);
    let at_last = domain.reference(&prompt);

    let (whole, reading) = domain.readout(
        &prompt,
        vec![logits(mid, few), logits(prompt.len() - 1, &choices)],
    );
    let (prefix, prefix_reading) = domain.readout(&prompt[..=mid], vec![logits(mid, &choices)]);
    let outcomes = domain.step(vec![reading, prefix_reading]);
    let values = |outcome: &Outcome, row: usize| {
        let Outcome::Forward { rows } = outcome else {
            panic!("a readout returned {outcome:?}");
        };
        let [values] = &rows[row].readouts[..] else {
            panic!("a readout row returns its readout's values");
        };
        values.clone()
    };
    let errors = [
        relative_error(&values(&outcomes[0], mid), &at_mid, few),
        relative_error(&values(&outcomes[0], prompt.len() - 1), &at_last, &choices),
        relative_error(&values(&outcomes[1], mid), &at_mid, &choices),
    ];
    println!("relative readout error against full logits: {errors:?}");
    let Outcome::Forward { rows } = &outcomes[0] else {
        unreachable!()
    };
    assert!(
        rows.iter()
            .enumerate()
            .all(|(row, result)| !result.readouts.is_empty()
                == (row == mid || row + 1 == prompt.len())),
        "only readout rows return values"
    );
    domain.domain.close(whole).unwrap();
    domain.domain.close(prefix).unwrap();
    drop(domain);
    let sequence = |tokens: &[TokenId], readouts| {
        ReadoutSequence::new(input(tokens), Vec::new(), readouts).unwrap()
    };
    // The full logits are computed independently of the readout path; the
    // bound is the rounding of one dense launch of the same rows.
    let reference = Reference::of(
        &ProjectorSelection::Disabled,
        &[
            sequence(
                &prompt,
                vec![logits(mid, few), logits(prompt.len() - 1, &choices)],
            ),
            sequence(&prompt[..=mid], vec![logits(mid, &choices)]),
        ],
    );
    for error in errors {
        assert!(error <= reference.bound(0), "readout error {error}");
    }
}

/// The history codec of every test's loads: compressed unless
/// `MAGNITUDE_TEST_KV=dense`.
fn test_codec() -> KvCodec {
    match std::env::var("MAGNITUDE_TEST_KV").as_deref() {
        Ok("dense") => KvCodec::Dense,
        _ => KvCodec::AffineK8V4,
    }
}

/// A chunk of `request`'s rows `rows` of `tokens` reading the logits of
/// `tokens` at each row of `reads` among them.
fn chunk(
    request: RequestId,
    tokens: &[TokenId],
    rows: std::ops::Range<usize>,
    reads: &[usize],
    columns: &[TokenId],
) -> Operation {
    let readouts = reads
        .iter()
        .filter(|row| rows.contains(row))
        .map(|&row| logits(row, columns))
        .collect::<Vec<_>>();
    Operation::Forward {
        request,
        kind: WorkKind::Prefill,
        tokens: tokens[rows.clone()].to_vec(),
        position: rows.start,
        conditioning: None,
        demand: if readouts.is_empty() {
            Demand::NONE
        } else {
            Demand::READOUT
        },
        select: Vec::new(),
        committed: rows.len(),
        prime: None,
        readouts,
    }
}

/// The values of every readout row of `outcomes`, in row order.
fn read_values(outcomes: &[Outcome]) -> Vec<Vec<f32>> {
    outcomes
        .iter()
        .flat_map(|outcome| {
            let Outcome::Forward { rows } = outcome else {
                panic!("a readout returned {outcome:?}");
            };
            rows.iter().flat_map(|row| row.readouts.iter().cloned())
        })
        .collect()
}

fn visibility(tokens: &[TokenId], blocks: Vec<Block>) -> Visibility {
    ReadoutSequence::new(input(tokens), blocks, Vec::new())
        .unwrap()
        .visibility()
}

/// A trunk and two blocks after it, each reading the trunk and itself
/// (`[T, q1, q2]`, the second reading over the first): computed in one
/// launch, where the second block reads the trunk among its launch's rows
/// as stored history and itself fresh, and in one launch per block, where
/// it reads the trunk as accepted history past the first block, the values
/// agree; the trunk's rows read their whole block, which changes them from
/// causal; and blocks of one row each reading every earlier one are
/// causal. For a dense history (`MAGNITUDE_TEST_KV=dense`) or a compressed
/// one.
#[test]
#[ignore = "requires a CUDA or Metal device and MAGNITUDE_TEST_GGUF"]
fn blocked_rows_read_alike_in_one_launch_and_across_launches() {
    let codec = test_codec();
    let resolved = resolved_with(
        test_gguf(),
        ProjectorSelection::Disabled,
        0,
        codec,
        PREFILL,
        CONTEXT,
    );
    let tokenizer = resolved.host.shared_tokenizer();
    let mut domain = Domain::of(&resolved);
    let trunk = text(
        &tokenizer,
        "The ferry left at dawn with nine crates of linen aboard.",
    );
    let first = text(&tokenizer, " Did it carry linen? Answer:");
    let second = text(&tokenizer, " Was the ferry late? Answer:");
    let tokens = [trunk.clone(), first.clone(), second.clone()].concat();
    let ends = [trunk.len(), trunk.len() + first.len(), tokens.len()];
    assert!(
        tokens.len() <= 2 * PREFILL,
        "{} rows exceed one launch",
        tokens.len()
    );
    let layout = vec![
        Block::new(0..ends[0], vec![0]),
        Block::new(ends[0]..ends[1], vec![0, 1]),
        Block::new(ends[1]..ends[2], vec![0, 2]),
    ];
    let blocks = visibility(&tokens, layout.clone());
    let columns = choices(&tokenizer);
    let reads = [0, ends[0] - 1, ends[1] - 1, ends[2] - 1];

    let whole = domain.open_visible(&tokens, blocks.clone());
    let one_launch = read_values(&domain.step(vec![chunk(
        whole,
        &tokens,
        0..tokens.len(),
        &reads,
        &columns,
    )]));
    let split = domain.open_visible(&tokens, blocks.clone());
    let mut across = Vec::new();
    for rows in [0..ends[0], ends[0]..ends[1], ends[1]..ends[2]] {
        across.extend(read_values(
            &domain.step(vec![chunk(split, &tokens, rows, &reads, &columns)]),
        ));
    }
    let causal = domain.open(&tokens);
    let causal_values = read_values(&domain.step(vec![chunk(
        causal,
        &tokens,
        0..tokens.len(),
        &reads,
        &columns,
    )]));
    let rows = (0..tokens.len())
        .map(|row| Block::new(row..row + 1, (0..=row).collect()))
        .collect::<Vec<_>>();
    let stepwise = domain.open_visible(&tokens, visibility(&tokens, rows));
    let stepwise_values = read_values(&domain.step(vec![chunk(
        stepwise,
        &tokens,
        0..tokens.len(),
        &reads,
        &columns,
    )]));
    let causal_split = domain.open(&tokens);
    let mut causal_across = Vec::new();
    for rows in [0..ends[0], ends[0]..ends[1], ends[1]..ends[2]] {
        causal_across.extend(read_values(&domain.step(vec![chunk(
            causal_split,
            &tokens,
            rows,
            &reads,
            &columns,
        )])));
    }
    for request in [whole, split, causal, stepwise, causal_split] {
        domain.domain.close(request).unwrap();
    }
    let launches = largest_difference(&across, &one_launch);
    let bidirectional = largest_difference(&one_launch[..1], &causal_values[..1]);
    let causal_launches = largest_difference(&causal_across, &causal_values);
    println!(
        "{codec:?}: across launches {launches} (causal rows {causal_launches}), bidirectional \
         trunk row against causal {bidirectional}"
    );
    // One-row blocks each reading every earlier one read exactly the rows a
    // causal row reads, in the same launch: identical bits.
    assert_eq!(
        stepwise_values, causal_values,
        "one-row blocks read otherwise than causal rows"
    );
    drop(domain);
    let readouts = reads.map(|row| logits(row, &columns)).to_vec();
    let reference = Reference::of(
        &ProjectorSelection::Disabled,
        &[layout, Vec::new()]
            .map(|blocks| ReadoutSequence::new(input(&tokens), blocks, readouts.clone()).unwrap()),
    );
    for (what, blocked, causal) in [
        ("one launch", &one_launch, &causal_values),
        ("across launches", &across, &causal_across),
    ] {
        reference.agree(
            &format!("{codec:?}: blocked and causal rows {what}"),
            &[blocked.clone(), causal.clone()],
        );
    }
    // The trunk's rows read their whole block: they differ from causal rows
    // beyond what rounding moves the causal reference.
    let effect = largest_difference(&one_launch[..1], &reference.values[1][..1]);
    assert!(
        effect > reference.bound(0),
        "a bidirectional trunk row reads as causal ({effect})"
    );
}

/// The stored readout head of `path` as F32 rows, `[outputs][hidden]`, and
/// whether it is stored BF16.
fn stored_head(path: &std::path::Path) -> (Vec<Vec<f32>>, bool) {
    use magnitude_artifacts::gguf::{Encoding, GgufArtifact};
    let artifact = GgufArtifact::open(path).unwrap();
    let tensor = artifact.tensor("decision.readout.weight").unwrap();
    let bytes = tensor.read().unwrap();
    let values = match tensor.encoding {
        Encoding::F32 => bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
            .collect::<Vec<_>>(),
        Encoding::BF16 => bytes
            .chunks_exact(2)
            .map(|half| f32::from_bits(u32::from(u16::from_le_bytes([half[0], half[1]])) << 16))
            .collect(),
        encoding => panic!("a readout head stored as {encoding:?}"),
    };
    let [_, hidden] = tensor.shape[..] else {
        panic!("the readout head is a matrix");
    };
    (
        values
            .chunks_exact(hidden as usize)
            .map(<[f32]>::to_vec)
            .collect(),
        tensor.encoding == Encoding::BF16,
    )
}

/// BF16 values stored little-endian, as F32.
fn bf16_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|half| f32::from_bits(u32::from(u16::from_le_bytes([half[0], half[1]])) << 16))
        .collect()
}

/// The CPU product of a stored head with a row's features: F32, rounded to
/// BF16 for a BF16 head.
fn head_product(head: &[Vec<f32>], stored_bf16: bool, hidden: &[f32]) -> Vec<f32> {
    head.iter()
        .map(|weights| {
            let product = weights.iter().zip(hidden).map(|(w, h)| w * h).sum::<f32>();
            if stored_bf16 { bf16(product) } else { product }
        })
        .collect()
}

/// `value` rounded to BF16, to nearest even.
fn bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

/// Head readout values are the stored head's product with the row's final
/// normalized hidden state (the row's features, read independently), in
/// F32 for an F32 head and rounded to BF16 for a BF16 head; a row reading
/// logits and the head returns both, in order.
#[test]
#[ignore = "requires a CUDA or Metal device and MAGNITUDE_TEST_HEAD_GGUFS"]
fn head_values_are_the_stored_heads_product_with_the_rows_features() {
    let paths = std::env::var("MAGNITUDE_TEST_HEAD_GGUFS").expect("set MAGNITUDE_TEST_HEAD_GGUFS");
    for path in paths.split(':').map(PathBuf::from) {
        let (head, stored_bf16) = stored_head(&path);
        let resolved = resolved_at(path.clone(), 0);
        let tokenizer = resolved.host.shared_tokenizer();
        let mut domain = Domain::of(&resolved);
        let prompt = text(&tokenizer, "The ferry left at dawn. Did it arrive? Answer:");
        let rows = [prompt.len() / 2, prompt.len() - 1];
        let choices = choices(&tokenizer);
        let request = domain.open(&prompt);
        let outcomes = domain.step(vec![Operation::Forward {
            request,
            kind: WorkKind::Prefill,
            tokens: prompt.clone(),
            position: 0,
            conditioning: None,
            demand: Demand::READOUT | Demand::FEATURES,
            select: Vec::new(),
            committed: prompt.len(),
            prime: None,
            readouts: vec![
                Readout {
                    row: rows[0],
                    kind: ReadoutKind::Head,
                },
                logits(rows[1], &choices),
                Readout {
                    row: rows[1],
                    kind: ReadoutKind::Head,
                },
            ],
        }]);
        let Outcome::Forward { rows: results } = &outcomes[0] else {
            panic!("a readout returned {:?}", outcomes[0]);
        };
        let features = results[0]
            .features
            .clone()
            .expect("every row publishes features");
        let features = domain
            .domain
            .read(&FeatureSpan {
                features,
                start: 0,
                count: prompt.len(),
            })
            .unwrap();
        let width = head[0].len();
        let mut largest = 0.0_f32;
        for (index, &row) in rows.iter().enumerate() {
            let readouts = &results[row].readouts;
            assert_eq!(
                readouts.len(),
                1 + index,
                "row {row} returns one list per readout"
            );
            let values = readouts.last().unwrap();
            assert_eq!(values.len(), head.len());
            let hidden = bf16_values(
                &features.bytes()[row * features.row_bytes()..][..features.row_bytes()],
            );
            assert_eq!(hidden.len(), width);
            let products = head_product(&head, stored_bf16, &hidden);
            let scale = products
                .iter()
                .fold(1e-6_f32, |m, value| m.max(value.abs()));
            for (value, product) in values.iter().zip(&products) {
                if stored_bf16 {
                    assert_eq!(*value, bf16(*value), "a BF16 head publishes BF16 values");
                }
                largest = largest.max((value - product).abs() / scale);
            }
        }
        println!(
            "{}: {} head, largest difference from the CPU product {largest}",
            path.display(),
            if stored_bf16 { "BF16" } else { "F32" }
        );
        assert!(largest <= HEAD_TOLERANCE, "head difference {largest}");
        domain.domain.close(request).unwrap();
    }
}

// --- owner -----------------------------------------------------------------

fn owner() -> (Host, Arc<ByteBpeTokenizer>) {
    owner_with(test_codec(), PREFILL, CONTEXT)
}

fn owner_with(codec: KvCodec, prefill: usize, context: usize) -> (Host, Arc<ByteBpeTokenizer>) {
    let resolved = resolved_with(
        test_gguf(),
        ProjectorSelection::Disabled,
        0,
        codec,
        prefill,
        context,
    );
    (
        owner_of(&resolved, prefill),
        resolved.host.shared_tokenizer(),
    )
}

/// An owner of a load of `resolved`, whose prompt chunks take at most
/// `prefill` rows.
fn owner_of(resolved: &ResolvedEngineConfiguration, prefill: usize) -> Host {
    let (domain, bindings, _) =
        build_native_domain(&resolved.manifest, resolved.host.shared_package()).unwrap();
    Host::new(|wakes| {
        Owner::with_prefix_cache_capacity(
            domain,
            bindings,
            limits_with(prefill),
            PrefixCacheCapacity {
                max_entries: 16,
                workload_state_bytes: u64::MAX,
            },
            wakes,
        )
    })
}

fn read(host: &mut Host, plan: ReadoutPlan) -> ReadoutValues {
    let mut stream: Stream = host.read(plan).unwrap();
    host.answer(&mut [&mut stream], Duration::from_secs(120));
    stream.values().clone()
}

fn largest_difference(left: &[Vec<f32>], right: &[Vec<f32>]) -> f32 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            assert_eq!(left.len(), right.len());
            let scale = right
                .iter()
                .map(|value| value.abs())
                .fold(1.0_f32, f32::max);
            left.iter()
                .zip(right)
                .map(|(a, b)| (a - b).abs() / scale)
                .fold(0.0_f32, f32::max)
        })
        .fold(0.0, f32::max)
}

// --- reference -------------------------------------------------------------

/// Rows of the largest launch the native execution contract admits: the
/// widest unit one launch computes.
const WIDE: usize = 512;

/// The kinds of readout, each moved by kernel shapes by its own measure: a
/// random head and scores read every direction of the final state alike,
/// where a row's largest logits are dominated by a few.
const KINDS: [&str; 3] = ["logits", "head", "scores"];

fn kind(readout: &Readout) -> usize {
    match readout.kind {
        ReadoutKind::Logits { .. } => 0,
        ReadoutKind::Head => 1,
        ReadoutKind::Scores { .. } => 2,
    }
}

/// Of `values`, read by `readouts` in order, those of kind `wanted`.
fn of_kind(values: &[Vec<f32>], readouts: &[Readout], wanted: usize) -> Vec<Vec<f32>> {
    assert_eq!(values.len(), readouts.len());
    values
        .iter()
        .zip(readouts)
        .filter(|(_, readout)| kind(readout) == wanted)
        .map(|(values, _)| values.clone())
        .collect()
}

/// Each sequence's values computed independently of every path a test
/// checks, and how far that computation's own rounding moves them.
///
/// `values`: each sequence alone, in one launch over dense history, by the
/// executor of a load whose launches take `WIDE` rows: no plan, run, fork,
/// trunk, batch, retention, chunking or layer-major schedule. Every path
/// under test is compared against these values, never against another
/// path under test.
///
/// The bound comes from the reference computation alone, rounded
/// otherwise, per kind of readout. `moved`: the most a sequence's reference
/// values change in the same single launch moved to the end of a launch of
/// each larger row class behind another sequence's rows (another kernel
/// shape, another place in its tiles), and the most its causal twin's (the
/// same tokens and readouts read causally, so no block, fork or plan code
/// takes part) change when its rows are split into two launches, the later
/// one reading the earlier rows as stored history. `stored`: the change of
/// the causal twin's values from reading every earlier row through the
/// test's history codec instead of dense (zero for a dense history); a
/// rounding step in a stored key or value may move a value a codec step.
/// A correct path rounds by at most one such shape or history move and one
/// codec step, and differs from the reference by at most `moved + stored`;
/// [`Self::bound`] allows twice that, and anything beyond is an error.
struct Reference {
    readouts: Vec<Vec<Readout>>,
    values: Vec<Vec<Vec<f32>>>,
    moved: [f32; 3],
    stored: [f32; 3],
}

/// The largest difference between values of kind `wanted` of each pair,
/// read by its readouts in order.
fn largest_of_kind<'a>(
    pairs: impl IntoIterator<Item = (&'a [Vec<f32>], &'a [Vec<f32>], &'a [Readout])>,
    wanted: usize,
) -> f32 {
    pairs
        .into_iter()
        .map(|(left, right, readouts)| {
            largest_difference(
                &of_kind(left, readouts, wanted),
                &of_kind(right, readouts, wanted),
            )
        })
        .fold(0.0, f32::max)
}

/// How a sequence's single launch is computed again besides alone.
#[derive(Clone, Copy)]
enum Again {
    /// At the end of a launch of every larger row class.
    Shapes,
    /// Split into two launches where its rows may be split.
    Split,
    /// Not at all.
    Never,
}

/// A sequence's readout values in plan order, in a launch of its own and
/// computed again ([`Again`]).
struct Launched {
    alone: Vec<Vec<f32>>,
    again: Vec<Vec<Vec<f32>>>,
}

/// `sequence` read causally: its causal twin.
fn causal_twin(sequence: &ReadoutSequence) -> ReadoutSequence {
    ReadoutSequence::new(
        sequence.input().clone(),
        Vec::new(),
        sequence.readouts().to_vec(),
    )
    .unwrap()
}

impl Reference {
    /// The reference of `sequences` on a load of `projector`, for paths
    /// computed over `codec`'s history.
    fn measure(projector: &ProjectorSelection, codec: KvCodec, sequences: &[ReadoutSequence]) -> Self {
        let twins = sequences.iter().map(causal_twin).collect::<Vec<_>>();
        let count = sequences.len();
        let items = sequences
            .iter()
            .map(|sequence| (sequence, Again::Shapes))
            .chain(twins.iter().map(|twin| (twin, Again::Split)))
            .collect::<Vec<_>>();
        let mut dense = launched(projector, KvCodec::Dense, &items);
        let dense_twins = dense.split_off(count);
        let moved = std::array::from_fn(|wanted| {
            largest_of_kind(
                sequences
                    .iter()
                    .zip(&dense)
                    .chain(twins.iter().zip(&dense_twins))
                    .flat_map(|(sequence, launched)| {
                        launched.again.iter().map(|again| {
                            (&again[..], &launched.alone[..], sequence.readouts())
                        })
                    }),
                wanted,
            )
        });
        let stored = match codec {
            KvCodec::Dense => [0.0; 3],
            codec => {
                let items = twins.iter().map(|twin| (twin, Again::Never)).collect::<Vec<_>>();
                let stored = launched(projector, codec, &items);
                std::array::from_fn(|wanted| {
                    largest_of_kind(
                        twins.iter().zip(stored.iter().zip(&dense_twins)).map(
                            |(twin, (stored, dense))| {
                                (&stored.alone[..], &dense.alone[..], twin.readouts())
                            },
                        ),
                        wanted,
                    )
                })
            }
        };
        let reference = Self {
            readouts: sequences
                .iter()
                .map(|sequence| sequence.readouts().to_vec())
                .collect(),
            values: dense.into_iter().map(|launched| launched.alone).collect(),
            moved,
            stored,
        };
        println!(
            "reference rounding: moved {:?}, stored {:?}",
            reference.moved, reference.stored
        );
        reference
    }

    /// The reference of `sequences` for paths over the test's codec.
    fn of(projector: &ProjectorSelection, sequences: &[ReadoutSequence]) -> Self {
        Self::measure(projector, test_codec(), sequences)
    }

    /// The most a correct computation of values of kind `wanted` may differ
    /// from the reference by.
    fn bound(&self, wanted: usize) -> f32 {
        2.0 * (self.moved[wanted] + self.stored[wanted])
    }

    /// The largest difference of kind `wanted` between `values` (one list
    /// per reference sequence, in order) and `expected`.
    fn difference(&self, values: &[Vec<Vec<f32>>], expected: &[Vec<Vec<f32>>], wanted: usize) -> f32 {
        assert_eq!(values.len(), self.readouts.len());
        assert_eq!(expected.len(), self.readouts.len());
        largest_of_kind(
            values
                .iter()
                .zip(expected)
                .zip(&self.readouts)
                .map(|((values, expected), readouts)| (&values[..], &expected[..], &readouts[..])),
            wanted,
        )
    }

    /// Assert that `values` (one list per reference sequence, in order) are
    /// the reference values within the bound of each kind.
    fn agree(&self, what: &str, values: &[Vec<Vec<f32>>]) {
        self.agree_with(what, values, &self.values);
    }

    /// Assert that `values` are `expected`, values computed independently of
    /// the path under test otherwise than the reference, within the bound
    /// of each kind.
    fn agree_with(&self, what: &str, values: &[Vec<Vec<f32>>], expected: &[Vec<Vec<f32>>]) {
        for (wanted, name) in KINDS.iter().enumerate() {
            if !self
                .readouts
                .iter()
                .any(|readouts| readouts.iter().any(|readout| kind(readout) == wanted))
            {
                continue;
            }
            let difference = self.difference(values, expected, wanted);
            let bound = self.bound(wanted);
            println!("{what} {name}: {difference} (bound {bound})");
            assert!(
                difference <= bound,
                "{what} {name} differ by {difference}, beyond {bound}"
            );
        }
    }
}

/// Each sequence's values in one launch of a load of `projector` over
/// `codec`'s history, alone and computed again as its [`Again`] says.
fn launched(
    projector: &ProjectorSelection,
    codec: KvCodec,
    sequences: &[(&ReadoutSequence, Again)],
) -> Vec<Launched> {
    let resolved = resolved_with(test_gguf(), projector.clone(), 0, codec, WIDE, 2 * WIDE);
    let tokenizer = resolved.host.shared_tokenizer();
    let mut domain = Domain::of(&resolved);
    sequences
        .iter()
        .map(|&(sequence, again)| {
            let rows = sequence.input().tokens().len();
            // A forward reads its readouts in the order of the rows answering
            // them.
            let mut order = (0..sequence.readouts().len()).collect::<Vec<_>>();
            order.sort_by_key(|&index| answer_row(&sequence.readouts()[index]));
            let readouts = order
                .iter()
                .map(|&index| sequence.readouts()[index].clone())
                .collect::<Vec<_>>();
            let scores = readouts
                .iter()
                .any(|readout| matches!(readout.kind, ReadoutKind::Scores { .. }));
            let open = |domain: &mut Domain| {
                let request = domain.open_input(sequence.input().clone(), sequence.visibility());
                if scores {
                    domain
                        .domain
                        .retain_features(request, (0..rows).collect(), None)
                        .unwrap();
                }
                request
            };
            // Rows `rows` of the sequence, reading the readouts among them.
            let forward = |request, rows: std::ops::Range<usize>| {
                let readouts = readouts
                    .iter()
                    .filter(|readout| rows.contains(&answer_row(readout)))
                    .cloned()
                    .collect::<Vec<_>>();
                Operation::Forward {
                    request,
                    kind: WorkKind::Prefill,
                    tokens: sequence.input().tokens()[rows.clone()].to_vec(),
                    position: rows.start,
                    conditioning: None,
                    demand: if readouts.is_empty() {
                        Demand::NONE
                    } else {
                        Demand::READOUT
                    },
                    select: Vec::new(),
                    committed: rows.len(),
                    prime: None,
                    readouts,
                }
            };
            // Back in plan order.
            let planned = |values: Vec<Vec<f32>>| {
                assert_eq!(values.len(), order.len());
                let mut planned = vec![Vec::new(); values.len()];
                for (index, values) in order.iter().zip(values) {
                    planned[*index] = values;
                }
                planned
            };
            let request = open(&mut domain);
            let alone = planned(read_values(&domain.step(vec![forward(request, 0..rows)])));
            let mut requests = vec![request];
            let mut computed = Vec::new();
            match again {
                Again::Shapes => {
                    for class in magnitude_batching::row_classes(WIDE) {
                        let Some(others) = class.checked_sub(rows).filter(|others| *others > 0)
                        else {
                            continue;
                        };
                        let other_tokens = repeated(
                            &tokenizer,
                            " The barge left the square with the copper.",
                            others,
                        );
                        let moved = open(&mut domain);
                        let other = domain.open(&other_tokens);
                        let launch = domain.step(vec![
                            chunk(other, &other_tokens, 0..others, &[], &[]),
                            forward(moved, 0..rows),
                        ]);
                        computed.push(planned(read_values(&launch[1..])));
                        requests.extend([moved, other]);
                    }
                }
                // At the first point from its middle where its rows may be
                // split (a boundary outside any media span), when it has one.
                Again::Split => {
                    let visibility = sequence.visibility();
                    let layout = sequence.input().layout();
                    let split = (rows / 2..rows)
                        .find(|&row| row > 0 && visibility.boundary(row) && layout.boundary(row));
                    if let Some(split) = split {
                        let request = open(&mut domain);
                        let mut values =
                            read_values(&domain.step(vec![forward(request, 0..split)]));
                        values.extend(read_values(
                            &domain.step(vec![forward(request, split..rows)]),
                        ));
                        computed.push(planned(values));
                        requests.push(request);
                    }
                }
                Again::Never => {}
            }
            for request in requests {
                domain.domain.close(request).unwrap();
            }
            Launched {
                alone,
                again: computed,
            }
        })
        .collect()
}

/// A plan's sequences answer as the reference (each alone in one dense
/// launch), their shared rows computed once, as do the sequences in plans
/// of their own, one at a time and batched together; the same plan answers
/// bit for bit again; retained shared rows serve a later plan uncomputed,
/// and its answers are bit for bit those of the plan that retained them.
/// Rows read logits, the head, or both.
#[test]
#[ignore = "requires a CUDA or Metal device and MAGNITUDE_TEST_GGUF"]
fn sequences_fork_their_shared_rows_and_retained_rows_are_not_recomputed() {
    let (mut host, tokenizer) = owner();
    let choices = choices(&tokenizer);
    let trunk = text(
        &tokenizer,
        "Report: the courier crossed the ridge at dusk with three crates of linen, two of \
         copper wire and one sealed letter for the harbour master, who signed for all of them. \
         At dawn the barge left the square with the copper, while the linen waited at the \
         glacier depot for the comet ferry, which was late by an hour and a half. Question:",
    );
    let shared = trunk.len();
    assert!(shared > 2 * PREFILL, "the trunk spans several chunks");
    let questions = [
        " Did the courier carry copper? Answer:",
        " Was the letter sealed? Answer:",
        " Did anyone sign for the crates? Answer:",
    ];
    let sequence = |question: &str, trunk_readout: bool| {
        let mut tokens = trunk.clone();
        tokens.extend(text(&tokenizer, question));
        let last = tokens.len() - 1;
        let head = |row| Readout {
            row,
            kind: ReadoutKind::Head,
        };
        let mut readouts = Vec::new();
        if trunk_readout {
            readouts.push(logits(shared - 1, &choices[..2]));
            readouts.push(head(shared - 1));
        }
        readouts.push(logits(shared + 2, &choices[2..]));
        readouts.push(head(last));
        readouts.push(logits(last, &choices));
        ReadoutSequence::new(input(&tokens), Vec::new(), readouts).unwrap()
    };
    let sequences = |trunk_readout: bool| {
        questions
            .iter()
            .map(|question| sequence(question, trunk_readout))
            .collect::<Vec<_>>()
    };
    let continuation_rows = sequences(false)
        .iter()
        .map(|sequence| sequence.input().tokens().len() - shared)
        .sum::<usize>() as u64;

    // Forked: the trunk is computed once, each sequence only past it.
    let forked = read(
        &mut host,
        ReadoutPlan::new(shared, sequences(true), false).unwrap(),
    );
    assert_eq!(forked.usage.cached_tokens, 0);
    assert_eq!(
        forked.usage.computed_tokens,
        shared as u64 + continuation_rows
    );
    // The same plan again: the same launches over the same state.
    let again = read(
        &mut host,
        ReadoutPlan::new(shared, sequences(true), false).unwrap(),
    );
    assert_eq!(again.sequences, forked.sequences, "a plan answers otherwise again");
    // Alone: each sequence from fresh state, one plan at a time, then the
    // three plans together, whose launches batch their rows.
    let alone = sequences(true)
        .into_iter()
        .map(|sequence| {
            let rows = sequence.input().tokens().len() as u64;
            let alone = read(
                &mut host,
                ReadoutPlan::new(shared, vec![sequence], false).unwrap(),
            );
            assert_eq!(alone.usage.computed_tokens, rows);
            alone
        })
        .collect::<Vec<_>>();
    let mut together = sequences(true)
        .into_iter()
        .map(|sequence| {
            host.read(ReadoutPlan::new(shared, vec![sequence], false).unwrap())
                .unwrap()
        })
        .collect::<Vec<_>>();
    host.answer(
        &mut together.iter_mut().collect::<Vec<_>>(),
        Duration::from_secs(120),
    );
    let reference = Reference::of(&ProjectorSelection::Disabled, &sequences(true));
    let alone = alone
        .iter()
        .map(|alone| alone.sequences[0].clone())
        .collect::<Vec<_>>();
    let batched = together
        .iter()
        .map(|together| together.values().sequences[0].clone())
        .collect::<Vec<_>>();
    reference.agree("alone", &alone);
    reference.agree("batched", &batched);
    reference.agree("forked", &forked.sequences);

    // Retained: the first plan computes and keeps its shared rows, the
    // second restores them and computes only its continuations.
    let first = read(
        &mut host,
        ReadoutPlan::new(shared, sequences(false), true).unwrap(),
    );
    assert_eq!(first.usage.cached_tokens, 0);
    let second = read(
        &mut host,
        ReadoutPlan::new(shared, sequences(false), true).unwrap(),
    );
    assert_eq!(second.usage.cached_tokens, shared as u64);
    assert_eq!(second.usage.computed_tokens, continuation_rows);
    // The same state and the same launches: identical bits.
    assert_eq!(
        first.sequences, second.sequences,
        "retained shared rows answer otherwise than computed ones"
    );
    reference.agree(
        "retained",
        &second
            .sequences
            .iter()
            .zip(&alone)
            .map(|(second, alone)| {
                // The plan read no trunk rows: its answers are the last three.
                let mut values = alone.clone();
                let start = values.len() - second.len();
                values[start..].clone_from_slice(second);
                values
            })
            .collect::<Vec<_>>(),
    );

    host.settle();
}

/// Prompt chunks wide enough for a trunk block and its question in one.
const BLOCKED_PREFILL: usize = 128;

/// The trained block layouts through plans, each answer against the
/// reference (its sequence alone in one dense launch). `bidirectional_trunk`:
/// a trunk reading itself whole, forked per question block reading the trunk
/// and itself, answers as the reference forked and alone, and a retained
/// trunk serves a later plan bit-identically. `bidirectional_sequence`:
/// one block over the whole sequence computes every row of it, though its
/// last readout is earlier, and its rows read as no causal row does.
/// `packed_set`: a trunk reading every question packed after it, each
/// question the trunk and itself, answers as the reference alone and batched
/// beside another pack. A block wider than any launch is computed whole. For a dense
/// history (`MAGNITUDE_TEST_KV=dense`) or a compressed one.
#[test]
#[ignore = "requires a CUDA or Metal device and MAGNITUDE_TEST_GGUF"]
fn trained_block_layouts_answer_as_their_rows_read() {
    let (mut host, tokenizer) = owner_with(test_codec(), BLOCKED_PREFILL, CONTEXT);
    let choices = choices(&tokenizer);
    let trunk = text(
        &tokenizer,
        "Report: the courier crossed the ridge at dusk with three crates of linen, two of \
         copper wire and one sealed letter for the harbour master, who signed for all of them. \
         At dawn the barge left the square with the copper, while the linen waited at the \
         glacier depot for the comet ferry, which was late by an hour and a half. Question:",
    );
    let shared = trunk.len();
    assert!(shared < BLOCKED_PREFILL);
    let questions = [
        " Did the courier carry copper? Answer:",
        " Was the letter sealed? Answer:",
        " Did anyone sign for the crates? Answer:",
    ];

    // bidirectional_trunk
    // A trunk readout row must be computed, so only a plan reading none
    // resumes from a retained trunk.
    let branch = |question: &str, trunk_readout: bool| {
        let mut tokens = trunk.clone();
        tokens.extend(text(&tokenizer, question));
        let end = tokens.len();
        assert!(end <= BLOCKED_PREFILL);
        let mut readouts = Vec::new();
        if trunk_readout {
            readouts.push(logits(shared - 1, &choices[..2]));
        }
        readouts.push(logits(end - 1, &choices));
        ReadoutSequence::new(
            input(&tokens),
            vec![
                Block::new(0..shared, vec![0]),
                Block::new(shared..end, vec![0, 1]),
            ],
            readouts,
        )
        .unwrap()
    };
    let branches = |trunk_readout: bool| {
        questions
            .iter()
            .map(|question| branch(question, trunk_readout))
            .collect::<Vec<_>>()
    };
    let forked = read(
        &mut host,
        ReadoutPlan::new(shared, branches(true), false).unwrap(),
    );
    let continuation_rows = branches(true)
        .iter()
        .map(|sequence| sequence.input().tokens().len() - shared)
        .sum::<usize>() as u64;
    assert_eq!(
        forked.usage.computed_tokens,
        shared as u64 + continuation_rows
    );
    let alone = branches(true)
        .into_iter()
        .map(|sequence| {
            read(
                &mut host,
                ReadoutPlan::new(shared, vec![sequence], false).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let first = read(
        &mut host,
        ReadoutPlan::new(shared, branches(false), true).unwrap(),
    );
    let second = read(
        &mut host,
        ReadoutPlan::new(shared, branches(false), true).unwrap(),
    );
    assert_eq!(second.usage.cached_tokens, shared as u64);
    assert_eq!(second.usage.computed_tokens, continuation_rows);
    assert_eq!(
        first.sequences, second.sequences,
        "a retained trunk answers alike"
    );

    // bidirectional_sequence
    let mut whole = trunk.clone();
    whole.extend(text(&tokenizer, questions[0]));
    let end = whole.len();
    let reads = vec![logits(4, &choices), logits(shared - 1, &choices)];
    let sequence = |blocks| ReadoutSequence::new(input(&whole), blocks, reads.clone()).unwrap();
    let bidirectional = read(
        &mut host,
        ReadoutPlan::new(0, vec![sequence(vec![Block::new(0..end, vec![0])])], false).unwrap(),
    );
    assert_eq!(
        bidirectional.usage.computed_tokens, end as u64,
        "the whole block computes"
    );
    let causal = read(
        &mut host,
        ReadoutPlan::new(0, vec![sequence(Vec::new())], false).unwrap(),
    );
    assert_eq!(causal.usage.computed_tokens, shared as u64);

    // packed_set, two packs small enough to share a launch.
    let state = &trunk[..30];
    let pack = |questions: &[&str]| {
        let mut tokens = state.to_vec();
        let mut blocks = vec![Block::new(0..state.len(), (0..=questions.len()).collect())];
        let mut readouts = Vec::new();
        for (index, question) in questions.iter().enumerate() {
            let start = tokens.len();
            tokens.extend(text(&tokenizer, question));
            blocks.push(Block::new(start..tokens.len(), vec![0, index + 1]));
            readouts.push(logits(tokens.len() - 1, &choices));
        }
        assert!(2 * tokens.len() <= BLOCKED_PREFILL);
        ReadoutSequence::new(input(&tokens), blocks, readouts).unwrap()
    };
    let packs = [&questions[..2], &questions[1..]];
    let packed_alone = packs
        .iter()
        .map(|questions| {
            let sequence = pack(questions);
            let rows = sequence.input().tokens().len() as u64;
            let values = read(
                &mut host,
                ReadoutPlan::new(0, vec![sequence], false).unwrap(),
            );
            assert_eq!(values.usage.computed_tokens, rows);
            values
        })
        .collect::<Vec<_>>();
    let mut together = packs
        .iter()
        .map(|questions| {
            host.read(ReadoutPlan::new(0, vec![pack(questions)], false).unwrap())
                .unwrap()
        })
        .collect::<Vec<_>>();
    host.answer(
        &mut together.iter_mut().collect::<Vec<_>>(),
        Duration::from_secs(120),
    );
    // The reference: every branch, every pack, and the whole sequence
    // blocked and causal, in that order.
    let measured = branches(true)
        .into_iter()
        .chain(packs.iter().map(|questions| pack(questions)))
        .chain([vec![Block::new(0..end, vec![0])], Vec::new()].map(sequence))
        .collect::<Vec<_>>();
    let reference = Reference::of(&ProjectorSelection::Disabled, &measured);
    let branch_count = questions.len();
    let pack_count = packs.len();
    let alike = |what: &str, values: &[Vec<Vec<f32>>], at: usize| {
        let mut all = reference.values.clone();
        all[at..at + values.len()].clone_from_slice(values);
        reference.agree(what, &all);
    };
    alike("forked trunk", &forked.sequences, 0);
    alike(
        "trunk alone",
        &alone.iter().map(|alone| alone.sequences[0].clone()).collect::<Vec<_>>(),
        0,
    );
    alike(
        "packed sets alone",
        &packed_alone.iter().map(|alone| alone.sequences[0].clone()).collect::<Vec<_>>(),
        branch_count,
    );
    alike(
        "packed sets batched",
        &together.iter().map(|together| together.values().sequences[0].clone()).collect::<Vec<_>>(),
        branch_count,
    );
    alike(
        "bidirectional and causal sequence",
        &[bidirectional.sequences[0].clone(), causal.sequences[0].clone()],
        branch_count + pack_count,
    );
    // Its rows read the whole block: they differ from causal rows beyond
    // what rounding moves the causal reference.
    let causal_reference = &reference.values[branch_count + pack_count + 1];
    let block_effect = largest_difference(&bidirectional.sequences[0], causal_reference);
    println!("bidirectional sequence against the causal reference {block_effect}");
    assert!(
        block_effect > reference.bound(0),
        "a bidirectional sequence reads as causal"
    );

    // A block wider than any launch is computed whole, in chunks.
    let mut wide = Vec::new();
    while wide.len() <= BLOCKED_PREFILL {
        wide.extend(trunk.iter().copied());
    }
    let rows = wide.len();
    let wide = ReadoutSequence::new(
        input(&wide),
        vec![Block::new(0..rows, vec![0])],
        vec![logits(4, &choices), logits(rows - 1, &choices)],
    )
    .unwrap();
    let wide = read(&mut host, ReadoutPlan::new(0, vec![wide], false).unwrap());
    assert_eq!(wide.usage.computed_tokens, rows as u64);
    host.settle();
}

/// A model whose attention is causal refuses a blocked plan at admission,
/// as an invalid request naming why, whether its blocks fit one launch or
/// not: only a model trained non-causal reads blocks, and only its load
/// prepares the key-and-value projection that publishes each chunk of a
/// wider one. It still reads a causal plan.
#[test]
#[ignore = "requires a CUDA or Metal device and MAGNITUDE_TEST_CAUSAL_GGUF"]
fn causal_models_refuse_blocked_plans() {
    let resolved = resolved_with(
        causal_gguf(),
        ProjectorSelection::Disabled,
        0,
        test_codec(),
        BLOCKED_PREFILL,
        CONTEXT,
    );
    let tokenizer = resolved.host.shared_tokenizer();
    let mut host = owner_of(&resolved, BLOCKED_PREFILL);
    let columns = choices(&tokenizer);
    let tokens = repeated(
        &tokenizer,
        " The courier crossed the ridge at dusk with three crates of linen.",
        BLOCKED_PREFILL + 1,
    );
    let plan = |rows: usize, blocks: Vec<Block>| {
        ReadoutPlan::new(
            0,
            vec![
                ReadoutSequence::new(
                    input(&tokens[..rows]),
                    blocks,
                    vec![logits(rows - 1, &columns)],
                )
                .unwrap(),
            ],
            false,
        )
        .unwrap()
    };
    let causal = read(&mut host, plan(BLOCKED_PREFILL, Vec::new()));
    assert_eq!(causal.usage.computed_tokens, BLOCKED_PREFILL as u64);
    for rows in [BLOCKED_PREFILL, tokens.len()] {
        match host.read(plan(rows, vec![Block::new(0..rows, vec![0])])) {
            Err(AdmissionError::Refused(RequestError::Input(reason))) => assert_eq!(
                reason,
                "a blocked sequence needs a model whose attention was trained non-causal"
            ),
            Err(error) => panic!("a blocked plan is refused otherwise: {error:?}"),
            Ok(_) => panic!("a causal model admits a blocked plan"),
        }
    }
    host.settle();
}

/// A plan over media input reads what generation's prefill path computes
/// for the same input: an image in the shared rows, encoded and computed
/// once, and another in one sequence's own rows, encoded after the fork.
/// Each head readout equals the stored head's product with that row's
/// features from the prefill path.
///
/// `MAGNITUDE_TEST_MMPROJ=P.gguf MAGNITUDE_TEST_IMAGE=I.png` besides
/// `MAGNITUDE_TEST_GGUF`, whose head the plan reads.
#[test]
#[ignore = "requires a CUDA or Metal device, MAGNITUDE_TEST_GGUF, MMPROJ and IMAGE"]
fn media_plans_read_what_the_prefill_path_computes() {
    let variable =
        |name: &str| PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("set {name}")));
    let projector = ProjectorSelection::Explicit(variable("MAGNITUDE_TEST_MMPROJ"));
    let resolved = resolved_with(
        test_gguf(),
        projector.clone(),
        0,
        test_codec(),
        PREFILL,
        CONTEXT,
    );
    let host = &resolved.host;
    let placeholder = host.media_placeholder().expect("a model with a projector");
    let image = ImageInput {
        media_type: "image/png".into(),
        bytes: std::fs::read(variable("MAGNITUDE_TEST_IMAGE"))
            .unwrap()
            .into(),
    };
    let prepare = |text: &str| {
        let tokens = host
            .tokenizer()
            .encode(text, SpecialTokens::Recognize)
            .unwrap();
        let images = vec![image.clone(); text.matches(placeholder).count()];
        host.prepare_input(tokens, &images).unwrap()
    };
    let trunk = format!("Picture: {placeholder} It shows a gradient. Question:");
    let inputs = [
        prepare(&format!(
            "{trunk} Is it bright? Compare {placeholder} again. Answer:"
        )),
        prepare(&format!("{trunk} Is it dark? Answer:")),
    ];
    let shared = prepare(&trunk).tokens().len();
    let [span] = &prepare(&trunk).layout().spans().to_vec()[..] else {
        panic!("the shared rows hold one image");
    };
    println!(
        "{shared} shared rows, the image at {}..{}",
        span.start, span.end
    );
    let rows = |input: &PreparedModelInput| [span.end, shared - 1, input.tokens().len() - 1];
    let (head, stored_bf16) = stored_head(&test_gguf());

    // The prefill path's features, each sequence from fresh state.
    let mut domain = Domain::of(&resolved);
    let expected = inputs
        .iter()
        .map(|input| {
            let features = domain.prefill_features(input);
            rows(input)
                .iter()
                .map(|&row| head_product(&head, stored_bf16, &features[row]))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    drop(domain);

    let mut host = owner_of(&resolved, PREFILL);
    let sequences = inputs
        .iter()
        .map(|input| {
            let readouts = rows(input)
                .into_iter()
                .map(|row| Readout {
                    row,
                    kind: ReadoutKind::Head,
                })
                .collect();
            ReadoutSequence::new(input.clone(), Vec::new(), readouts).unwrap()
        })
        .collect::<Vec<_>>();
    let values = read(
        &mut host,
        ReadoutPlan::new(shared, sequences.clone(), false).unwrap(),
    );
    assert_eq!(
        values.usage.computed_tokens as usize,
        shared
            + inputs
                .iter()
                .map(|input| input.tokens().len() - shared)
                .sum::<usize>(),
        "the shared rows are computed once"
    );
    let reference = Reference::of(&projector, &sequences);
    reference.agree("media plan", &values.sequences);
    reference.agree_with(
        "media plan against the prefill path",
        &values.sequences,
        &expected,
    );
    host.settle();
}

/// Scores readouts are the scaled dot products `h · h_j / √d` of the rows'
/// final normalized hidden states, read independently through the prefill
/// path: a query after its keys, with keys in the shared rows and in its
/// sequence's own; a query in the shared rows before keys in its own; keys
/// in the shared rows alone, answered by them; and keys through the query
/// itself.
#[test]
#[ignore = "requires a CUDA or Metal device and MAGNITUDE_TEST_GGUF"]
fn scores_are_the_dot_products_of_the_rows_features() {
    let resolved = resolved(0);
    let tokenizer = resolved.host.shared_tokenizer();
    let trunk = text(
        &tokenizer,
        "Invoice 4471: forty crates of slate tiles, shipped from the northern quarry on the \
         fourth of May and received at the river warehouse two days later. Question:",
    );
    let shared = trunk.len();
    assert!(shared > 2 * PREFILL, "the shared rows span several chunks");
    let scores = |row, rows| Readout {
        row,
        kind: ReadoutKind::Scores { rows },
    };
    let inputs = [
        " Where was it shipped from? Answer:",
        " How many crates? Answer:",
    ]
    .map(|question| {
        let mut tokens = trunk.clone();
        tokens.extend(text(&tokenizer, question));
        input(&tokens)
    });
    let lasts = inputs.each_ref().map(|input| input.tokens().len() - 1);
    let readouts = [
        vec![
            scores(lasts[0], 2..shared + 3),
            scores(3, shared..shared + 4),
        ],
        vec![
            scores(shared - 1, 0..shared),
            scores(lasts[1], lasts[1] - 2..lasts[1] + 1),
        ],
    ];

    // Each sequence's features through the prefill path, from fresh state.
    let mut domain = Domain::of(&resolved);
    let scale = (resolved.host.definition().decoder.hidden as f32)
        .sqrt()
        .recip();
    let expected = inputs
        .iter()
        .zip(&readouts)
        .map(|(input, readouts)| {
            let features = domain.prefill_features(input);
            readouts
                .iter()
                .map(|readout| {
                    let ReadoutKind::Scores { rows } = &readout.kind else {
                        unreachable!()
                    };
                    rows.clone()
                        .map(|key| {
                            features[readout.row]
                                .iter()
                                .zip(&features[key])
                                .map(|(q, k)| q * k)
                                .sum::<f32>()
                                * scale
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    drop(domain);

    let mut host = owner_of(&resolved, PREFILL);
    let sequences = inputs
        .iter()
        .zip(&readouts)
        .map(|(input, readouts)| {
            ReadoutSequence::new(input.clone(), Vec::new(), readouts.clone()).unwrap()
        })
        .collect::<Vec<_>>();
    let values = read(
        &mut host,
        ReadoutPlan::new(shared, sequences.clone(), false).unwrap(),
    );
    assert_eq!(
        values.usage.computed_tokens as usize,
        shared + lasts.iter().map(|last| last + 1 - shared).sum::<usize>(),
        "the shared rows are computed once"
    );
    let reference = Reference::of(&ProjectorSelection::Disabled, &sequences);
    reference.agree("scores", &values.sequences);
    reference.agree_with(
        "scores against the prefill path's features",
        &values.sequences,
        &expected,
    );
    host.settle();
}

/// `rows` tokens of `text` repeated.
fn repeated(tokenizer: &ByteBpeTokenizer, text_: &str, rows: usize) -> Vec<TokenId> {
    let unit = text(tokenizer, text_);
    unit.iter().copied().cycle().take(rows).collect()
}

/// The rows every `step`-th row of `rows` and its last.
fn spread_rows(rows: usize, step: usize) -> Vec<usize> {
    let mut at = (step - 1..rows).step_by(step).collect::<Vec<_>>();
    if at.last() != Some(&(rows - 1)) {
        at.push(rows - 1);
    }
    at
}

/// A unit wider than one launch computes layer-major in launch-sized
/// chunks and answers as the same unit in one launch of the largest class
/// over dense history, within the reference's rounding: causal rows, a
/// bidirectional sequence (one block over every row) and a packed set whose
/// trunk spans the chunks, read at rows spread across them for logits, the
/// head and scores against every row (its rows retained); and a
/// bidirectional sequence whose last chunk of 36 rows has a class (64) whose
/// launches round otherwise, which it launches in the load's least
/// layer-major class instead, so every row reads every chunk alike. Moving a
/// bidirectional unit changes the key tiles of the rows sharing a query
/// tile with the other sequence's in its first global attention layer (one
/// BF16 step there), which every later layer spreads to every row; causal
/// rows of that tile read keys of one tile either way. The chunked side
/// runs through the owner, whose launches take `BLOCKED_PREFILL` rows; the
/// reference through the executor with launches of `WIDE`, which also reads
/// rows whose tokens together exceed one projection and finds each row's
/// full logits. `MAGNITUDE_TEST_GGUF` stores a readout head.
#[test]
#[ignore = "requires a CUDA or Metal device and MAGNITUDE_TEST_GGUF"]
fn wide_units_compute_layer_major_as_in_one_launch() {
    let codec = test_codec();
    let rows = 500;
    // A bidirectional unit of 3 chunks of `BLOCKED_PREFILL` rows and one of
    // 36.
    let narrow = 3 * BLOCKED_PREFILL + 36;
    let (questions, packed) = (rows - 100, rows - 50);
    let layouts = [
        Vec::new(),
        vec![Block::new(0..rows, vec![0])],
        vec![
            Block::new(0..questions, vec![0, 1, 2]),
            Block::new(questions..packed, vec![0, 1]),
            Block::new(packed..rows, vec![0, 2]),
        ],
    ];
    let resolved = resolved_with(
        test_gguf(),
        ProjectorSelection::Disabled,
        1,
        KvCodec::Dense,
        WIDE,
        2 * WIDE,
    );
    let tokenizer = resolved.host.shared_tokenizer();
    let tokens = repeated(
        &tokenizer,
        " The courier crossed the ridge at dusk with three crates of linen and a sealed \
         letter for the harbour master, who signed for all of them before the comet ferry.",
        rows,
    );
    let columns = choices(&tokenizer);
    // Logits at every read row, the head at every fifth, and the last row's
    // scores against every row, which its sequence retains.
    let readouts_of = |rows: usize| {
        let mut readouts = spread_rows(rows, 20)
            .into_iter()
            .enumerate()
            .flat_map(|(index, row)| {
                let head = (index % 5 == 0).then_some(Readout {
                    row,
                    kind: ReadoutKind::Head,
                });
                std::iter::once(logits(row, &columns)).chain(head)
            })
            .collect::<Vec<_>>();
        readouts.push(Readout {
            row: rows - 1,
            kind: ReadoutKind::Scores { rows: 0..rows },
        });
        readouts
    };
    let reads = spread_rows(rows, 20);
    let sequences = layouts
        .iter()
        .map(|blocks| {
            ReadoutSequence::new(input(&tokens), blocks.clone(), readouts_of(rows)).unwrap()
        })
        .chain(std::iter::once(
            ReadoutSequence::new(
                input(&tokens[..narrow]),
                vec![Block::new(0..narrow, vec![0])],
                readouts_of(narrow),
            )
            .unwrap(),
        ))
        .collect::<Vec<_>>();
    let reference = Reference::measure(&ProjectorSelection::Disabled, codec, &sequences);

    // Rows of 1024 tokens of their own each, whose values together exceed
    // the logits one launch of this load plans (one row over the
    // vocabulary), are read in groups: each row's values are its full
    // logits there.
    let mut domain = Domain::of(&resolved);
    let wide_columns = reads
        .iter()
        .enumerate()
        .map(|(index, &row)| {
            let tokens = (0..1024)
                .map(|offset| TokenId(1000 + (index * 1024 + offset) as u32))
                .collect::<Vec<_>>();
            logits(row, &tokens)
        })
        .collect::<Vec<_>>();
    let request = domain.open(&tokens);
    let grouped = read_values(&domain.step(vec![Operation::Forward {
        request,
        kind: WorkKind::Prefill,
        tokens: tokens.clone(),
        position: 0,
        conditioning: None,
        demand: Demand::READOUT,
        select: Vec::new(),
        committed: rows,
        prime: None,
        readouts: wide_columns.clone(),
    }]));
    domain.domain.close(request).unwrap();
    let grouping = [0, reads.len() / 2, reads.len() - 1]
        .into_iter()
        .map(|index| {
            let ReadoutKind::Logits { tokens: read } = &wide_columns[index].kind else {
                unreachable!()
            };
            let full = domain.reference(&tokens[..=reads[index]]);
            relative_error(&grouped[index], &full, read)
        })
        .fold(0.0_f32, f32::max);
    drop(domain);
    println!(
        "rows reading {} tokens together against full logits {grouping}",
        1024 * reads.len()
    );
    assert!(
        grouping <= reference.bound(0),
        "grouped readout values differ by {grouping}"
    );

    let (mut host, _) = owner_with(codec, BLOCKED_PREFILL, 2 * WIDE);
    // The last 36 rows' own class launches another kernel (the small-row
    // GEMM) than the chunks of `BLOCKED_PREFILL` rows do, so they launch in
    // the chunks' class.
    let layer_major = host
        .owner()
        .inspect_domain(|domain| Ok(domain.layer_major_rows()))
        .unwrap();
    let last = narrow - 3 * BLOCKED_PREFILL;
    assert!(magnitude_batching::row_class(last).unwrap() < layer_major);
    assert_eq!(layer_major, BLOCKED_PREFILL);
    let mut chunked_values = Vec::new();
    for sequence in &sequences {
        let started = std::time::Instant::now();
        let chunked = read(
            &mut host,
            ReadoutPlan::new(0, vec![sequence.clone()], false).unwrap(),
        );
        let rows = sequence.input().tokens().len();
        println!(
            "{rows} rows, {} blocks, in chunks of at most {BLOCKED_PREFILL}: {:?}",
            sequence.blocks().len(),
            started.elapsed()
        );
        assert_eq!(chunked.usage.computed_tokens, rows as u64);
        chunked_values.push(chunked.sequences[0].clone());
    }
    reference.agree(
        &format!("{codec:?} causal, bidirectional and packed: chunks against one dense launch"),
        &chunked_values,
    );
    host.settle();
}

/// A unit of fewer rows than a decode class whose rows read their launch's
/// rows apart (a packed set: questions reading the trunk over another
/// question) reads those rows only once its launch has written them: alone,
/// over history rows another sequence wrote before, it answers bit for bit
/// as over fresh ones. Over a dense history (`MAGNITUDE_TEST_KV=dense`),
/// whose small launches order no such reads, it also answers as in a
/// larger launch beside another sequence's rows.
#[test]
#[ignore = "requires a CUDA or Metal device and MAGNITUDE_TEST_GGUF"]
fn small_units_reading_apart_read_their_launch_once_written() {
    let codec = test_codec();
    let resolved = resolved_with(
        test_gguf(),
        ProjectorSelection::Disabled,
        0,
        codec,
        PREFILL,
        CONTEXT,
    );
    let tokenizer = resolved.host.shared_tokenizer();
    let mut domain = Domain::of(&resolved);
    let tokens = text(&tokenizer, "Linen came. Copper? Ferry? Yes");
    let tokens = &tokens[..8];
    let layout = vec![
        Block::new(0..4, vec![0, 1, 2]),
        Block::new(4..6, vec![0, 1]),
        Block::new(6..8, vec![0, 2]),
    ];
    let packed = visibility(tokens, layout.clone());
    assert!(packed.reads_apart(0..8));
    let columns = choices(&tokenizer);
    let reads = [3, 5, 7];
    let other = text(
        &tokenizer,
        "The barge left the square with the copper while the linen waited at the depot.",
    );
    let other = &other[..PREFILL - 8];
    let alone = |domain: &mut Domain| {
        let request = domain.open_visible(tokens, packed.clone());
        let values =
            read_values(&domain.step(vec![chunk(request, tokens, 0..8, &reads, &columns)]));
        domain.domain.close(request).unwrap();
        values
    };
    let fresh = alone(&mut domain);
    // Another sequence writes history rows and releases them.
    let peer = domain.open(other);
    domain.step(vec![chunk(peer, other, 0..other.len(), &[], &columns)]);
    domain.domain.close(peer).unwrap();
    let over_stale = alone(&mut domain);
    assert_eq!(
        fresh, over_stale,
        "a small unit read rows its launch had not written"
    );
    let beside = domain.open_visible(tokens, packed);
    let peer = domain.open(other);
    let launch = domain.step(vec![
        chunk(beside, tokens, 0..8, &reads, &columns),
        chunk(peer, other, 0..other.len(), &[], &columns),
    ]);
    for request in [beside, peer] {
        domain.domain.close(request).unwrap();
    }
    drop(domain);
    let readouts = reads.map(|row| logits(row, &columns)).to_vec();
    let sequence = ReadoutSequence::new(input(tokens), layout, readouts).unwrap();
    let reference = Reference::of(&ProjectorSelection::Disabled, &[sequence]);
    reference.agree(&format!("{codec:?}: a small unit reading apart alone"), &[fresh]);
    reference.agree(
        &format!("{codec:?}: a small unit reading apart beside another"),
        &[read_values(&launch[..1])],
    );
}

/// The cost of computing a wide unit layer-major: a bidirectional trunk of
/// `LONG` rows in chunks of `WIDE`, and the same rows in blocks of `WIDE`
/// each reading every earlier one (layer-major, attending as causal rows
/// do), against the rows prefilled causally in launches of `WIDE`: the
/// fastest of three interleaved runs of each, after one that seals its
/// graphs.
#[test]
#[ignore = "requires a CUDA or Metal device and MAGNITUDE_TEST_GGUF"]
fn layer_major_cost_against_causal_prefill() {
    const LONG: usize = 32 * WIDE;
    let codec = test_codec();
    let resolved = resolved_with(
        test_gguf(),
        ProjectorSelection::Disabled,
        0,
        codec,
        WIDE,
        LONG + WIDE,
    );
    let tokenizer = resolved.host.shared_tokenizer();
    let mut domain = Domain::of(&resolved);
    let tokens = repeated(
        &tokenizer,
        " The barge left the square with the copper, while the linen waited at the glacier \
         depot for the comet ferry, which was late by an hour and a half.",
        LONG,
    );
    let columns = choices(&tokenizer);
    let trunk = visibility(&tokens, vec![Block::new(0..LONG, vec![0])]);
    // Launch-sized blocks each reading itself and every earlier one: rows
    // attending as causal rows do, computed layer-major.
    let prior = visibility(
        &tokens,
        (0..LONG / WIDE)
            .map(|block| Block::new(block * WIDE..(block + 1) * WIDE, (0..=block).collect()))
            .collect(),
    );
    let mut time = |visibility: Option<&Visibility>| {
        let request = match visibility {
            Some(visibility) => domain.open_visible(&tokens, visibility.clone()),
            None => domain.open(&tokens),
        };
        let started = std::time::Instant::now();
        match visibility {
            Some(_) => {
                domain.step(vec![chunk(
                    request,
                    &tokens,
                    0..LONG,
                    &[LONG - 1],
                    &columns,
                )]);
            }
            None => {
                for start in (0..LONG).step_by(WIDE) {
                    domain.step(vec![chunk(
                        request,
                        &tokens,
                        start..start + WIDE,
                        &[LONG - 1],
                        &columns,
                    )]);
                }
            }
        }
        let elapsed = started.elapsed();
        domain.domain.close(request).unwrap();
        elapsed
    };
    // The fastest of three interleaved runs of each, after one that seals.
    let mut runs = [Duration::MAX; 3];
    for round in 0..4 {
        for (fastest, visibility) in runs.iter_mut().zip([Some(&trunk), Some(&prior), None]) {
            let elapsed = time(visibility);
            if round > 0 {
                *fastest = (*fastest).min(elapsed);
            }
        }
    }
    let [layered, publishing, causal] = runs;
    println!(
        "{codec:?}: {LONG} rows: bidirectional trunk layer-major {layered:?} ({:.2}x), \
         launch-sized blocks reading earlier ones layer-major {publishing:?} ({:.2}x), causal \
         prefill {causal:?}",
        layered.as_secs_f64() / causal.as_secs_f64(),
        publishing.as_secs_f64() / causal.as_secs_f64(),
    );
}
