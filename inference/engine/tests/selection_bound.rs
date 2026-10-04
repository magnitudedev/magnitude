//! Real-device qualification of the selection bound. The owner's admission:
//! a prefill round that would finish more prompts than the bound finishes the
//! bound's worth and holds the rest one row short of their prompt's end, and
//! every request completes from there. The executor's launch validation:
//! vocabulary work the load did not plan is refused before anything is
//! reserved. Run only on a host with an MTP GGUF:
//! `MAGNITUDE_TEST_MTP_GGUF=/path/model.gguf cargo test --release
//! -p magnitude-engine --test selection_bound -- --ignored --nocapture`.

use magnitude_chat::{ByteBpeTokenizer, SpecialTokens};
use magnitude_engine::{
    build_native_domain,
    composition::EngineConfiguration,
    options::{ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection},
};
use magnitude_executor::{
    self as executor,
    platform::{DeviceRequest, MemoryReserves},
    ExecutionPath, ExecutorDomain, RequestId, StateBindings, TokenId,
};
use magnitude_family_contracts::{PreparedModelInput, TokenPlan};
use magnitude_generation::{
    EndOfGeneration, Generation, InputLayout, MethodChoice, Options, Sampling, Shaping,
};
use magnitude_scheduler::{
    owner::Owner,
    prefix_cache::{PrefixCacheCapacity, PrefixRetention},
    ServiceLimits,
};
use magnitude_state::KvCodec;
use owner_host::{Host, Stream};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::time::Duration;

mod owner_host;

const OUTPUT: usize = 12;

fn plain_owner(limits: ServiceLimits) -> (Host, usize, std::sync::Arc<ByteBpeTokenizer>) {
    let model = PathBuf::from(
        std::env::var_os("MAGNITUDE_TEST_MTP_GGUF").expect("set MAGNITUDE_TEST_MTP_GGUF"),
    );
    let resolved = EngineConfiguration {
        package: PackageOptions {
            target: model,
            projector: ProjectorSelection::Disabled,
            draft: None,
        },
        model: ModelPolicy {
            method: ModelMethod::Plain,
            mtp_proposals: None,
            kv_codec: KvCodec::AffineK8V4,
            lookahead: false,
            exported_logits_rows: 0,
            error_classes: Vec::new(),
        },
        context_tokens: Some(1024),
        service: limits.clone(),
        path: ExecutionPath::Native,
        device: DeviceRequest::Automatic,
        kernel_cache: std::env::var_os("MAGNITUDE_TEST_KERNEL_CACHE").map(PathBuf::from),
        reserves: MemoryReserves::standard(),
    }
    .resolve()
    .unwrap();
    let vocabulary = resolved.manifest.definition.decoder.vocabulary as usize;
    let tokenizer = resolved.host.shared_tokenizer();
    let package = resolved.host.shared_package();
    let (domain, bindings, _) = build_native_domain(&resolved.manifest, package).unwrap();
    let host = Host::new(|wakes| {
        Owner::with_prefix_cache_capacity(
            domain,
            bindings,
            limits,
            PrefixCacheCapacity { max_entries: 0 },
            wakes,
        )
    });
    (host, vocabulary, tokenizer)
}

fn prompt(tokenizer: &ByteBpeTokenizer, seed: u32) -> Vec<TokenId> {
    let text = format!(
        "Ledger {seed}: the warehouse in district {} shipped {} crates of linen.",
        seed * 3 + 1,
        seed * 7 + 2
    );
    tokenizer.encode(&text, SpecialTokens::Literal).unwrap()
}

fn sampled(prompt: &[TokenId], vocabulary: usize, seed: u64) -> Generation {
    Generation::new(
        prompt.to_vec(),
        InputLayout::new(prompt.len(), vec![]).unwrap(),
        Options {
            max_tokens: OUTPUT,
            output_capacity: OUTPUT,
            context_limit: 1024,
            vocabulary,
            stop_tokens: BTreeSet::new(),
            suppressed_tokens: BTreeSet::new(),
            sampling: Sampling::Categorical,
            // Shaped: the selection rows are shaped in place.
            shaping: Shaping {
                temperature: 0.7,
                top_p: 0.9,
                top_k: 40,
                ..Shaping::default()
            },
            seed,
            forced_quantum: 0,
            method: MethodChoice::Plain,
            end_of_generation: EndOfGeneration::Suppress,
            reasoning_budget: None,
        },
        None,
    )
    .unwrap()
}

fn admit(host: &mut Host, generation: Generation) -> Stream {
    let prompt = generation.prompt().to_vec();
    let input = PreparedModelInput::from_text_coordinates(
        TokenPlan::new(
            prompt.clone(),
            InputLayout::new(prompt.len(), vec![]).unwrap(),
        )
        .unwrap(),
        (0..prompt.len()).map(|row| [row as i32; 3]).collect(),
    )
    .unwrap();
    host.admit(generation, input, PrefixRetention::Transient, OUTPUT)
        .unwrap()
}

#[test]
#[ignore = "requires a CUDA or Metal device and MAGNITUDE_TEST_MTP_GGUF"]
fn a_prefill_round_finishes_at_most_the_selection_bound_and_defers_the_rest() {
    // A selection bound of 2, and a prefill allowance that fits every prompt
    // of the cohort in one round: five prompts would finish together.
    const BOUND: usize = 2;
    let limits = ServiceLimits {
        prefill_tokens: 256,
        decode_tokens: BOUND,
        decode_share: 0.5,
        locality_seconds: 1.0,
    };
    let (mut host, vocabulary, tokenizer) = plain_owner(limits);
    let prompts = (0..5)
        .map(|seed| prompt(&tokenizer, seed))
        .collect::<Vec<_>>();
    assert!(prompts.iter().map(Vec::len).sum::<usize>() <= 256);

    let mut streams = prompts
        .iter()
        .enumerate()
        .map(|(seed, tokens)| admit(&mut host, sampled(tokens, vocabulary, seed as u64)))
        .collect::<Vec<_>>();
    // A request has finished its prompt once it has selected a token. Every
    // completed round publishes its selections, so no step sees more than
    // the bound's worth of prompts finish, and a prompt the bound defers
    // waits with exactly its final row left.
    let mut finished = vec![false; streams.len()];
    let mut held_short = false;
    let deadline = std::time::Instant::now() + Duration::from_secs(120);
    while !streams.iter().all(|stream| stream.finished()) {
        assert!(
            std::time::Instant::now() < deadline,
            "requests did not finish"
        );
        host.step(&mut streams.iter_mut().collect::<Vec<_>>());
        let mut newly = 0;
        for (index, stream) in streams.iter().enumerate() {
            if !finished[index] && !stream.output.is_empty() {
                finished[index] = true;
                newly += 1;
            }
            // Without retention no chunk otherwise stops at the final row.
            held_short |= host.snapshot(stream.id).is_some_and(|snapshot| {
                snapshot.output_tokens == 0
                    && snapshot.resident_position == snapshot.prompt_tokens - 1
            });
        }
        assert!(newly <= BOUND, "{newly} prompts finished in one step");
    }
    assert!(held_short, "no prompt was deferred at its final row");
    for stream in &streams {
        assert!(stream.failure().is_none(), "{:?}", stream.failure());
        assert_eq!(stream.output.len(), OUTPUT);
    }
}

/// A plain native domain whose selection bound is `bound`, exporting logits
/// for `exported` rows.
fn plain_domain(bound: usize, exported: usize) -> (ExecutorDomain, StateBindings) {
    let model = PathBuf::from(
        std::env::var_os("MAGNITUDE_TEST_MTP_GGUF").expect("set MAGNITUDE_TEST_MTP_GGUF"),
    );
    let resolved = EngineConfiguration {
        package: PackageOptions {
            target: model,
            projector: ProjectorSelection::Disabled,
            draft: None,
        },
        model: ModelPolicy {
            method: ModelMethod::Plain,
            mtp_proposals: None,
            kv_codec: KvCodec::AffineK8V4,
            lookahead: false,
            exported_logits_rows: exported,
            error_classes: Vec::new(),
        },
        context_tokens: Some(1024),
        service: ServiceLimits {
            prefill_tokens: 64,
            decode_tokens: bound,
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
    let (domain, bindings, _) =
        build_native_domain(&resolved.manifest, resolved.host.shared_package()).unwrap();
    (domain, bindings)
}

/// A finishing prompt chunk of `request`: four rows, selecting at the last
/// with `shaping`, and demanding logits when `logits`.
fn finishing(
    domain: &mut ExecutorDomain,
    bindings: &mut StateBindings,
    request: RequestId,
    shaping: executor::Shaping,
    logits: bool,
) -> executor::Operation {
    let tokens = (1..=4).map(TokenId).collect::<Vec<_>>();
    let input = PreparedModelInput::from_text_coordinates(
        TokenPlan::new(
            tokens.clone(),
            InputLayout::new(tokens.len(), vec![]).unwrap(),
        )
        .unwrap(),
        (0..tokens.len()).map(|row| [row as i32; 3]).collect(),
    )
    .unwrap();
    domain.install_input(request, input).unwrap();
    domain.open_state(bindings, request, None).unwrap();
    executor::Operation::Forward {
        request,
        kind: executor::WorkKind::Prefill,
        tokens: tokens.clone(),
        position: 0,
        conditioning: None,
        demand: if logits {
            executor::Demand::SELECT | executor::Demand::LOGITS
        } else {
            executor::Demand::SELECT
        },
        select: vec![executor::SelectSpec {
            sampling: executor::Sampling::Greedy,
            seed: 0,
            position: tokens.len() - 1,
            domain: 0,
            mask: None,
            shaping,
            history: None,
        }],
        committed: tokens.len(),
        prime: None,
    }
}

fn refusal(result: Result<executor::DomainReservation, executor::DomainError>) -> String {
    match result {
        Err(executor::DomainError::Input(detail)) => detail,
        Err(other) => panic!("refused as {other:?}, not as invalid input"),
        Ok(_) => panic!("the launch was admitted"),
    }
}

/// The executor refuses every launch whose vocabulary work the load did not
/// plan, before reserving anything, and admits the same work within bounds.
#[test]
#[ignore = "requires a CUDA or Metal device and MAGNITUDE_TEST_MTP_GGUF"]
fn launch_validation_refuses_vocabulary_work_the_load_did_not_plan() {
    use executor::Shaping as Shape;
    let shaped = Shape {
        temperature: 0.7,
        top_k: 40,
        ..Shape::default()
    };

    // A served load: bound 2, no logits export.
    let (mut domain, mut bindings) = plain_domain(2, 0);
    let three = (1..=3)
        .map(|id| finishing(&mut domain, &mut bindings, RequestId(id), shaped, false))
        .collect::<Vec<_>>();
    let detail = refusal(domain.reserve(&mut bindings, &three));
    assert!(detail.contains("selection bound is 2"), "{detail}");
    let logits = finishing(
        &mut domain,
        &mut bindings,
        RequestId(4),
        Shape::default(),
        true,
    );
    let detail = refusal(domain.reserve(&mut bindings, std::slice::from_ref(&logits)));
    assert!(detail.contains("exports none"), "{detail}");
    assert!(
        domain.reserve(&mut bindings, &three[..2]).is_ok(),
        "two selections fit the bound"
    );

    // A diagnostic load exporting one row's logits.
    let (mut domain, mut bindings) = plain_domain(2, 1);
    let shaped_logits = finishing(&mut domain, &mut bindings, RequestId(1), shaped, true);
    let detail = refusal(domain.reserve(&mut bindings, std::slice::from_ref(&shaped_logits)));
    assert!(detail.contains("selects unshaped"), "{detail}");
    let two = (2..=3)
        .map(|id| {
            finishing(
                &mut domain,
                &mut bindings,
                RequestId(id),
                Shape::default(),
                true,
            )
        })
        .collect::<Vec<_>>();
    let detail = refusal(domain.reserve(&mut bindings, &two));
    assert!(detail.contains("the load exports 1"), "{detail}");
    assert!(
        domain.reserve(&mut bindings, &two[..1]).is_ok(),
        "one exported row fits"
    );
}
