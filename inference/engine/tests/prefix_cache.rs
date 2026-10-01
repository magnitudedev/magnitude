//! Real-device qualification of the prefix cache: conversation turns resume
//! at the previous reply, rewritten replies at the previous prompt, arbitrary
//! request sequences at their deepest cached prefix, and requests with images
//! keep their image conditioning across eviction and cached resumption.
//!
//! `MAGNITUDE_TEST_GGUF=M.gguf MAGNITUDE_TEST_MMPROJ=P.gguf
//! [VISION_IMAGE=I.jpeg] cargo test --release -p magnitude-engine
//! --test prefix_cache -- --ignored --nocapture`. The model must accept
//! images through its projector; the image defaults to llama.cpp's
//! `test-1.jpeg`.

use magnitude_chat::{request::ImageInput, ByteBpeTokenizer, SpecialTokens};
use magnitude_engine::{
    build_native_domain,
    composition::{EngineConfiguration, ResolvedEngineConfiguration},
    host::HostArtifacts,
    options::{ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection},
};
use magnitude_executor::{
    platform::{DeviceRequest, MemoryReserves},
    Demand, ExecutionPath, ExecutorDomain, Operation, Outcome, PhysicalDecision, RequestId,
    ReservedResources, StateBindings, TokenId, WorkKind,
};
use magnitude_family_contracts::{PreparedModelInput, TokenPlan};
use magnitude_generation::{
    EndOfGeneration, Generation, InputLayout, MethodChoice, Options, Sampling, Shaping,
};
use magnitude_scheduler::{
    owner::Owner,
    prefix_cache::{PrefixCacheCapacity, PrefixRetention, MIN_PREFIX_HIT},
    ServiceLimits,
};
use magnitude_state::{KvCodec, ShrinkPolicy};
use owner_host::Host;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::time::Duration;

mod owner_host;

const CONTEXT: usize = 4096;

fn resolved(projector: bool) -> ResolvedEngineConfiguration {
    let variable = |name: &str| {
        PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("set {name}")))
    };
    EngineConfiguration {
        package: PackageOptions {
            target: variable("MAGNITUDE_TEST_GGUF"),
            projector: if projector {
                ProjectorSelection::Explicit(variable("MAGNITUDE_TEST_MMPROJ"))
            } else {
                ProjectorSelection::Disabled
            },
            draft: None,
        },
        model: ModelPolicy {
            method: ModelMethod::Plain,
            mtp_proposals: None,
            kv_codec: KvCodec::AffineK8V4,
            lookahead: false,
            exported_logits_rows: 512,
        },
        context_tokens: Some(CONTEXT),
        service: limits(),
        path: ExecutionPath::Native,
        device: DeviceRequest::Automatic,
        kernel_cache: std::env::var_os("MAGNITUDE_TEST_KERNEL_CACHE").map(PathBuf::from),
        reserves: MemoryReserves::standard(),
    }
    .resolve()
    .unwrap()
}

fn limits() -> ServiceLimits {
    ServiceLimits {
        prefill_tokens: 512,
        decode_tokens: 32,
        decode_share: 0.5,
        locality_seconds: 1.0,
    }
}

/// A plain owner caching up to 64 prefixes (none are evicted by count in
/// these tests), with its vocabulary and tokenizer.
fn owner() -> (Host, usize, std::sync::Arc<ByteBpeTokenizer>) {
    let resolved = resolved(false);
    let vocabulary = resolved.manifest.definition.decoder.vocabulary as usize;
    let tokenizer = resolved.host.shared_tokenizer();
    let (domain, bindings, _) =
        build_native_domain(&resolved.manifest, resolved.host.shared_package()).unwrap();
    let host = Host::new(|wakes| {
        Owner::with_prefix_cache_capacity(
            domain,
            bindings,
            limits(),
            PrefixCacheCapacity { max_entries: 64 },
            wakes,
        )
    });
    (host, vocabulary, tokenizer)
}

/// Deterministic English text of about `len` tokens; distinct seeds give
/// distinct texts.
fn text(tokenizer: &ByteBpeTokenizer, seed: u32, len: usize) -> Vec<TokenId> {
    let text = (1..=64)
        .map(|line| {
            format!(
                "Note {seed}.{line}: the {} crossed the {} at {} past {}.",
                ["heron", "barge", "courier", "glacier", "comet"][(seed as usize + line) % 5],
                ["river", "square", "ridge", "harbour"][line % 4],
                line * 3 + seed as usize,
                ["noon", "dawn", "dusk"][line % 3],
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    let mut tokens = tokenizer.encode(&text, SpecialTokens::Literal).unwrap();
    assert!(tokens.len() >= len);
    tokens.truncate(len);
    tokens
}

fn greedy(prompt: &[TokenId], vocabulary: usize, max_tokens: usize) -> Generation {
    Generation::new(
        prompt.to_vec(),
        InputLayout::new(prompt.len(), vec![]).unwrap(),
        Options {
            max_tokens,
            output_capacity: max_tokens,
            context_limit: CONTEXT,
            vocabulary,
            stop_tokens: BTreeSet::new(),
            suppressed_tokens: BTreeSet::new(),
            sampling: Sampling::Greedy,
            shaping: Shaping {
                temperature: 0.0,
                ..Shaping::default()
            },
            seed: 0,
            forced_quantum: 0,
            method: MethodChoice::Plain,
            end_of_generation: EndOfGeneration::Suppress,
            reasoning_budget: None,
        },
        None,
    )
    .unwrap()
}

/// The text input the model's family prepares for `prompt`: each row at its
/// own position.
fn text_input(prompt: &[TokenId]) -> PreparedModelInput {
    PreparedModelInput::from_text_coordinates(
        TokenPlan::new(
            prompt.to_vec(),
            InputLayout::new(prompt.len(), vec![]).unwrap(),
        )
        .unwrap(),
        (0..prompt.len()).map(|row| [row as i32; 3]).collect(),
    )
    .unwrap()
}

/// One admitted request of a sequence and what it produced.
struct Turn {
    prompt: Vec<TokenId>,
    output: Vec<TokenId>,
    cached: usize,
}

/// Admit `prompts` together with the prefix cache and run them all to their
/// end. A request completes on its stream only once its terminal prefix is
/// retained (after any reconciliation round the prefix cache needs), and the
/// owner retires it then. Admitting several at once lets later ones wait for
/// an earlier one's shared prefix.
fn run(host: &mut Host, vocabulary: usize, prompts: &[Vec<TokenId>], max_tokens: usize) -> Vec<Turn> {
    let requests = prompts
        .iter()
        .map(|prompt| (prompt.clone(), Vec::new()))
        .collect::<Vec<_>>();
    run_with_cache_points(host, vocabulary, &requests, max_tokens)
}

/// [`run`] with each prompt's declared cache points.
fn run_with_cache_points(
    host: &mut Host,
    vocabulary: usize,
    requests: &[(Vec<TokenId>, Vec<usize>)],
    max_tokens: usize,
) -> Vec<Turn> {
    let prompts = requests
        .iter()
        .map(|(prompt, _)| prompt.clone())
        .collect::<Vec<_>>();
    let mut streams = requests
        .iter()
        .map(|(prompt, cache_points)| {
            host.admit(
                greedy(prompt, vocabulary, max_tokens),
                text_input(prompt),
                PrefixRetention::Retain {
                    cache_points: cache_points.clone(),
                },
                max_tokens,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    host.drive(
        &mut streams.iter_mut().collect::<Vec<_>>(),
        Duration::from_secs(120),
    );
    streams
        .into_iter()
        .zip(prompts)
        .map(|(stream, prompt)| {
            assert_eq!(stream.output.len(), max_tokens);
            assert!(
                host.snapshot(stream.id).is_none(),
                "a completed request is retired"
            );
            Turn {
                prompt,
                cached: stream.usage().cached_tokens,
                output: stream.output,
            }
        })
        .collect()
}

fn path(turn: &Turn) -> Vec<TokenId> {
    turn.prompt.iter().chain(&turn.output).copied().collect()
}

/// Every turn of a conversation extends the previous prompt and reply, as a
/// chat template that re-renders prior replies exactly produces. Each turn
/// resumes at the end of the previous reply.
#[test]
#[ignore = "requires a Metal or CUDA device and MAGNITUDE_TEST_GGUF"]
fn conversation_turns_resume_at_the_previous_reply() {
    let (mut host, vocabulary, tokenizer) = owner();
    let mut prompt = text(&tokenizer, 1, 200);
    let mut previous: Option<usize> = None;
    for turn in 0..4 {
        let [done] = run(&mut host, vocabulary, &[prompt.clone()], 12)
            .try_into()
            .ok()
            .unwrap();
        eprintln!("turn {turn}: prompt={} cached={}", done.prompt.len(), done.cached);
        match previous {
            None => assert_eq!(done.cached, 0),
            Some(end) => assert_eq!(done.cached, end, "turn {turn} resumes at the last reply"),
        }
        let reply_end = done.prompt.len() + done.output.len();
        previous = Some(reply_end);
        prompt = path(&done);
        prompt.extend(text(&tokenizer, 100 + turn, 40));
    }
    assert_eq!(host.owner().reconcile_memory_charge().unwrap().unattributed, 0);
}

/// A template that rewrites the previous reply (dropping reasoning, say)
/// diverges inside it; the next turn resumes one row before the previous
/// prompt's end.
#[test]
#[ignore = "requires a Metal or CUDA device and MAGNITUDE_TEST_GGUF"]
fn a_rewritten_reply_resumes_at_the_previous_prompt() {
    let (mut host, vocabulary, tokenizer) = owner();
    let first = text(&tokenizer, 2, 200);
    let [done] = run(&mut host, vocabulary, &[first.clone()], 12)
        .try_into()
        .ok()
        .unwrap();
    let mut rewritten = first.clone();
    rewritten.extend(text(&tokenizer, 3, 60));
    assert_ne!(rewritten[first.len()], done.output[0]);
    let [next] = run(&mut host, vocabulary, &[rewritten], 12)
        .try_into()
        .ok()
        .unwrap();
    assert_eq!(next.cached, first.len() - 1);
}

/// A long system prompt, then a last message that changes on every request,
/// with its start declared as a cache point (as chat preparation declares
/// the boundary opening the last message). Every request after the first
/// resumes at that start, however its message begins: the second message
/// shares a leading phrase with the first, the third starts differently.
#[test]
#[ignore = "requires a Metal or CUDA device and MAGNITUDE_TEST_GGUF"]
fn a_changed_last_message_resumes_at_its_declared_start() {
    let (mut host, vocabulary, tokenizer) = owner();
    let system = text(&tokenizer, 20, 1000);
    let lead = text(&tokenizer, 21, 24);
    let message = |parts: &[Vec<TokenId>]| {
        let mut prompt = system.clone();
        for part in parts {
            prompt.extend(part);
        }
        (prompt, vec![system.len()])
    };
    let messages = [
        message(&[lead.clone(), text(&tokenizer, 22, 40)]),
        message(&[lead.clone(), text(&tokenizer, 23, 40)]),
        message(&[tokenizer
            .encode(
                "Summarize the harbour log in one line, then list every courier by name.",
                SpecialTokens::Literal,
            )
            .unwrap()]),
    ];
    assert_ne!(messages[2].0[system.len()], lead[0]);
    let mut cached = Vec::new();
    for request in &messages {
        let [done] = run_with_cache_points(&mut host, vocabulary, &[request.clone()], 8)
            .try_into()
            .ok()
            .unwrap();
        eprintln!("prompt={} cached={}", done.prompt.len(), done.cached);
        cached.push(done.cached);
    }
    assert_eq!(cached, [0, system.len(), system.len()]);
    assert_eq!(host.owner().reconcile_memory_charge().unwrap().unattributed, 0);
}

/// Without a declared cache point, a changed last message diverges below
/// every retained state: the request recomputes its prompt and retains a
/// state at the divergence, where the next request diverging there resumes.
#[test]
#[ignore = "requires a Metal or CUDA device and MAGNITUDE_TEST_GGUF"]
fn an_undeclared_divergence_is_retained_by_the_request_that_finds_it() {
    let (mut host, vocabulary, tokenizer) = owner();
    let system = text(&tokenizer, 30, 1000);
    let message = |seed| {
        let mut prompt = system.clone();
        prompt.extend(text(&tokenizer, seed, 40));
        prompt
    };
    let turns = [31, 32, 33].map(|seed| {
        let [done] = run(&mut host, vocabulary, &[message(seed)], 8)
            .try_into()
            .ok()
            .unwrap();
        eprintln!("prompt={} cached={}", done.prompt.len(), done.cached);
        done.cached
    });
    assert_eq!(turns[..2], [0, 0]);
    assert!(turns[2] >= system.len());
}

/// A tiny deterministic generator, so a failing sequence reproduces.
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, bound: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % bound as u64) as usize
    }
}

/// Random sequences of conversations, branches from earlier paths, and
/// concurrent groups sharing prefixes. Every request resumes at least as
/// deeply as the deepest prefix it is guaranteed to find: the whole path of
/// an earlier finished request, or one row before an earlier prompt's end,
/// whichever ends deepest within its prompt.
#[test]
#[ignore = "requires a Metal or CUDA device and MAGNITUDE_TEST_GGUF"]
fn arbitrary_request_sequences_resume_from_their_deepest_cached_prefix() {
    let (mut host, vocabulary, tokenizer) = owner();
    let seed = std::env::var("PREFIX_CACHE_SEED")
        .ok()
        .and_then(|seed| seed.parse().ok())
        .unwrap_or(7);
    let mut random = Lcg(seed);
    let roots = [text(&tokenizer, 10, 160), text(&tokenizer, 11, 220)];
    let mut finished: Vec<Turn> = Vec::new();
    let mut fresh = 1000;
    for round in 0..12 {
        let group = 1 + random.below(3);
        let prompts = (0..group)
            .map(|_| {
                fresh += 1;
                let tail = text(&tokenizer, fresh, 30 + random.below(80));
                let mut prompt = match (finished.len(), random.below(3)) {
                    // A fresh conversation on a shared system prompt.
                    (0, _) | (_, 0) => roots[random.below(roots.len())].clone(),
                    // The next turn of an earlier conversation.
                    (count, 1) => path(&finished[random.below(count)]),
                    // A branch from inside an earlier path.
                    (count, _) => {
                        let source = path(&finished[random.below(count)]);
                        source[..MIN_PREFIX_HIT + random.below(source.len() - MIN_PREFIX_HIT)]
                            .to_vec()
                    }
                };
                prompt.extend(tail);
                prompt
            })
            .collect::<Vec<_>>();
        let turns = run(&mut host, vocabulary, &prompts, 8);
        for turn in &turns {
            let guaranteed = finished
                .iter()
                .flat_map(|earlier| [path(earlier), earlier.prompt[..earlier.prompt.len() - 1].to_vec()])
                .filter(|prefix| {
                    prefix.len() >= MIN_PREFIX_HIT
                        && prefix.len() < turn.prompt.len()
                        && turn.prompt.starts_with(prefix)
                })
                .map(|prefix| prefix.len())
                .max()
                .unwrap_or(0);
            eprintln!(
                "round {round}: prompt={} cached={} guaranteed={guaranteed}",
                turn.prompt.len(),
                turn.cached
            );
            assert!(turn.cached >= guaranteed, "round {round} resumed too shallow");
            assert!(turn.cached < turn.prompt.len());
        }
        finished.extend(turns);
        assert_eq!(host.owner().reconcile_memory_charge().unwrap().unattributed, 0);
    }
}

/// Run the encodes a residency returned and install their features. Each
/// flight carries the bindings and returns them.
fn encode(
    domain: &mut ExecutorDomain,
    mut bindings: StateBindings,
    operations: Vec<Operation>,
) -> StateBindings {
    for operation in operations {
        let ReservedResources::Vision(workspace, output) = domain
            .reserve(&mut bindings, std::slice::from_ref(&operation))
            .unwrap()
            .into_resources()
        else {
            panic!("an encode runs on the vision lane")
        };
        let flight = domain
            .submit_vision(bindings, &operation, workspace, output)
            .unwrap_or_else(|failure| panic!("{}", failure.error()));
        let (pending, next) = domain.finish_vision(flight).unwrap();
        bindings = next;
        domain
            .reconcile(pending, PhysicalDecision { accepted_rows: 0 })
            .unwrap();
    }
    bindings
}

/// Forward the input rows `range` and return the last row's logits.
fn forward(
    domain: &mut ExecutorDomain,
    mut bindings: StateBindings,
    request: RequestId,
    input: &PreparedModelInput,
    range: std::ops::Range<usize>,
) -> (Vec<f32>, StateBindings) {
    let rows = range.len();
    let operation = Operation::Forward {
        request,
        kind: WorkKind::Prefill,
        tokens: input.tokens()[range.clone()].to_vec(),
        position: range.start,
        conditioning: None,
        demand: Demand::LOGITS,
        select: Vec::new(),
        committed: rows,
        prime: None,
    };
    let operations = [operation];
    let ReservedResources::Target(reservation) = domain
        .reserve(&mut bindings, &operations)
        .unwrap()
        .into_resources()
    else {
        panic!("a forward runs on the target lane")
    };
    let flight = domain
        .submit_target(bindings, &operations, reservation)
        .unwrap_or_else(|failure| panic!("{}", failure.error()));
    let (pending, bindings) = domain.finish_target(flight).unwrap();
    let mut logits = Vec::new();
    for pending in pending {
        let Outcome::Forward { rows: results } = pending.outcome().clone() else {
            panic!("a forward returns forward rows")
        };
        for row in results.iter().filter_map(|row| row.logits.as_ref()) {
            logits = row.read_to_host().unwrap();
        }
        domain
            .reconcile(pending, PhysicalDecision { accepted_rows: rows })
            .unwrap();
    }
    assert!(!logits.is_empty());
    (logits, bindings)
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// A prompt with one image (`VISION_IMAGE`, or llama.cpp's `test-1.jpeg`)
/// between two sentences.
fn image_prompt(host: &HostArtifacts) -> PreparedModelInput {
    let placeholder = host.media_placeholder().expect("a model with a projector");
    let image = ImageInput {
        media_type: "image/jpeg".into(),
        bytes: std::fs::read(
            std::env::var_os("VISION_IMAGE")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    PathBuf::from(std::env::var("HOME").unwrap())
                        .join("repos/llama.cpp/tools/mtmd/test-1.jpeg")
                }),
        )
        .unwrap()
        .into(),
    };
    let tokens = host
        .tokenizer()
        .encode(
            &format!("Look at this picture:{placeholder}Now describe what you see in it, carefully."),
            SpecialTokens::Recognize,
        )
        .unwrap();
    host.prepare_input(tokens, &[image]).unwrap()
}

/// Vision holds no activation or output until an image arrives: an image
/// claims them, idle shrink releases them, and every image claims and
/// releases the same bytes.
#[test]
#[ignore = "requires a Metal or CUDA device, MAGNITUDE_TEST_GGUF and MAGNITUDE_TEST_MMPROJ"]
fn vision_claims_its_slots_on_an_image_and_releases_them_at_idle() {
    let resolved = resolved(true);
    let input = image_prompt(&resolved.host);
    let (mut domain, mut bindings, _) =
        build_native_domain(&resolved.manifest, resolved.host.shared_package()).unwrap();
    let charged = |domain: &ExecutorDomain| domain.resources().device().memory_usage().charged;
    let mut cycles = Vec::new();
    for request in [RequestId(1), RequestId(2)] {
        domain.install_input(request, input.clone()).unwrap();
        let encodes = domain.open_state(&mut bindings, request, None).unwrap();
        assert_eq!(encodes.len(), 1);
        let opened = charged(&domain);
        bindings = encode(&mut domain, bindings, encodes);
        let encoded = charged(&domain);
        assert!(encoded > opened, "the image claimed no vision storage");
        domain.close(request).unwrap();
        let released = domain
            .shrink_state(&mut bindings, ShrinkPolicy::Idle)
            .unwrap();
        let idle = charged(&domain);
        assert_eq!(encoded - idle, released);
        assert!(released > 0, "idle released no vision storage");
        let accounted = domain.reconcile_memory_charge(&[]).unwrap();
        assert_eq!(accounted.unattributed, 0, "{accounted:?}");
        eprintln!(
            "vision cycle: opened={opened} encoded={encoded} idle={idle} released={released}"
        );
        cycles.push((encoded - opened, released, idle));
    }
    // The first image also loads the vision weights, which stay resident
    // until optional components are released; the second claims only the
    // slots, and idle releases exactly them both times.
    let [(_, first_released, first_idle), (second_claimed, second_released, second_idle)] =
        cycles[..]
    else {
        unreachable!("two cycles")
    };
    assert_eq!(second_claimed, second_released, "idle kept vision storage");
    assert_eq!(first_released, second_released);
    assert_eq!(first_idle, second_idle);
}

/// A request with an image keeps its input across eviction: replay
/// re-encodes the image and reaches the same logits as an uninterrupted run.
/// A request resuming from a cached prefix past the image needs no encode
/// and reaches them too.
#[test]
#[ignore = "requires a Metal or CUDA device, MAGNITUDE_TEST_GGUF and MAGNITUDE_TEST_MMPROJ"]
fn image_conditioning_survives_eviction_and_cached_resumption() {
    let resolved = resolved(true);
    let host = &resolved.host;
    let input = image_prompt(host);
    let [span] = input.layout().spans() else {
        panic!("one image span")
    };
    // Split after the image, so the first chunk carries the whole image.
    let split = span.end + 2;
    let end = input.tokens().len();
    assert!(split < end && split <= 512, "image span {}..{} of {end}", span.start, span.end);
    let (mut domain, mut bindings, _) =
        build_native_domain(&resolved.manifest, host.shared_package()).unwrap();

    // Uninterrupted: encode, the image chunk, then the rest.
    let reference = RequestId(1);
    domain.install_input(reference, input.clone()).unwrap();
    let encodes = domain.open_state(&mut bindings, reference, None).unwrap();
    assert_eq!(encodes.len(), 1);
    bindings = encode(&mut domain, bindings, encodes);
    let encoded = domain.reconcile_memory_charge(&[]).unwrap();
    assert_eq!(encoded.unattributed, 0, "{encoded:?}");
    bindings = forward(&mut domain, bindings, reference, &input, 0..split).1;
    let after_image = domain.resume_state(reference).unwrap();
    let (expected, next) = forward(&mut domain, bindings, reference, &input, split..end);
    bindings = next;
    domain.close(reference).unwrap();

    // Evicted after the image: its state and features go, its input stays.
    // Becoming resident again re-encodes the image for the replay.
    let evicted = RequestId(2);
    domain.install_input(evicted, input.clone()).unwrap();
    let encodes = domain.open_state(&mut bindings, evicted, None).unwrap();
    bindings = encode(&mut domain, bindings, encodes);
    bindings = forward(&mut domain, bindings, evicted, &input, 0..split).1;
    domain.release_state(&[evicted]).unwrap();
    let encodes = domain.open_state(&mut bindings, evicted, None).unwrap();
    assert_eq!(encodes.len(), 1, "replay re-encodes the released image");
    bindings = encode(&mut domain, bindings, encodes);
    bindings = forward(&mut domain, bindings, evicted, &input, 0..split).1;
    let (replayed, next) = forward(&mut domain, bindings, evicted, &input, split..end);
    bindings = next;
    assert_eq!(bits(&replayed), bits(&expected), "replay keeps the image conditioning");
    domain.close(evicted).unwrap();

    // Resumed from the cached state past the image: nothing to encode.
    let resumed = RequestId(3);
    domain.install_input(resumed, input.clone()).unwrap();
    let encodes = domain
        .open_state(&mut bindings, resumed, Some(&after_image))
        .unwrap();
    assert!(encodes.is_empty(), "an image before the resume position is not re-encoded");
    let (continued, _) = forward(&mut domain, bindings, resumed, &input, split..end);
    assert_eq!(bits(&continued), bits(&expected));
    domain.close(resumed).unwrap();

    let held = domain.reconcile_memory_charge(&[&after_image]).unwrap();
    assert_eq!(held.unattributed, 0, "{held:?}");
    drop(after_image);
    let released = domain.reconcile_memory_charge(&[]).unwrap();
    assert_eq!(released.unattributed, 0, "{released:?}");
}
