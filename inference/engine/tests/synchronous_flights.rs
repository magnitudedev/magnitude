//! The CPU device completes a flight during its submission. The owner must
//! still return to its worker after each flight, so controls and events that
//! arrive while a request generates (status, stop, cancellation, admission,
//! memory observation) are applied between flights rather than after the
//! request ends, and output reaches the host as it is produced. Run with
//! `MAGNITUDE_TEST_GGUF=<plain dense model.gguf> cargo test --release
//! -p magnitude-engine --test synchronous_flights -- --ignored --nocapture`.

use magnitude_engine::{
    build_native_domain,
    composition::EngineConfiguration,
    options::{ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection},
};
use magnitude_executor::{
    platform::{DeviceRequest, MemoryReserves},
    ExecutionPath, TokenId,
};
use magnitude_family_contracts::{PreparedModelInput, TokenPlan};
use magnitude_generation::{
    EndOfGeneration, Generation, InputLayout, MethodChoice, Options, Sampling, Shaping,
};
use magnitude_scheduler::{owner::Owner, prefix_cache::PrefixRetention, ServiceLimits};
use magnitude_state::KvCodec;
use owner_host::Host;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::time::{Duration, Instant};

mod owner_host;

#[test]
#[ignore = "requires MAGNITUDE_TEST_GGUF"]
fn each_run_reconciles_at_most_one_cpu_flight() {
    let model =
        PathBuf::from(std::env::var_os("MAGNITUDE_TEST_GGUF").expect("set MAGNITUDE_TEST_GGUF"));
    let limits = ServiceLimits {
        prefill_tokens: 64,
        decode_tokens: 16,
        decode_share: 0.5,
        locality_seconds: 1.0,
    };
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
        device: "cpu".parse::<DeviceRequest>().unwrap(),
        kernel_cache: std::env::var_os("MAGNITUDE_TEST_KERNEL_CACHE").map(PathBuf::from),
        reserves: MemoryReserves::standard(),
    }
    .resolve()
    .unwrap();
    let vocabulary = resolved.manifest.definition.decoder.vocabulary as usize;
    let package = resolved.host.shared_package();
    let (domain, bindings, _) = build_native_domain(&resolved.manifest, package).unwrap();
    let mut host = Host::new(|wakes| Owner::new(domain, bindings, limits, wakes));

    const MAX_TOKENS: usize = 12;
    let prompt = (0..40u64)
        .map(|index| {
            TokenId(((index * 2_654_435_761 + 12_345) % (vocabulary as u64 - 1024) + 512) as u32)
        })
        .collect::<Vec<_>>();
    let generation = Generation::new(
        prompt.clone(),
        InputLayout::new(prompt.len(), vec![]).unwrap(),
        Options {
            max_tokens: MAX_TOKENS,
            output_capacity: MAX_TOKENS,
            context_limit: 1024,
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
    .unwrap();
    let input = PreparedModelInput::from_text_coordinates(
        TokenPlan::new(
            prompt.clone(),
            InputLayout::new(prompt.len(), vec![]).unwrap(),
        )
        .unwrap(),
        (0..prompt.len()).map(|row| [row as i32; 3]).collect(),
    )
    .unwrap();
    let mut stream = host
        .admit(generation, input, PrefixRetention::Transient, MAX_TOKENS)
        .unwrap();

    let deadline = Instant::now() + Duration::from_secs(120);
    let mut runs = 0;
    while !stream.finished() {
        assert!(Instant::now() < deadline, "the request did not finish");
        let before = stream.output.len();
        host.step(&mut [&mut stream]);
        runs += 1;
        // A plain decode flight accepts one token; a run that drove several
        // synchronous flights would publish more than one.
        assert!(
            stream.output.len() - before <= 1,
            "one run published {} tokens",
            stream.output.len() - before
        );
    }
    assert_eq!(stream.usage().completion_tokens as usize, MAX_TOKENS);
    assert!(runs > MAX_TOKENS, "{runs} runs for {MAX_TOKENS} tokens");
}
