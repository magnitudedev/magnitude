//! Real-device memory qualification: optional-component residency, demand
//! and Reclaim release under a process limit, idle slab release and retained
//! release with replay. Run only on a host with an MTP GGUF:
//! `MAGNITUDE_TEST_MTP_GGUF=/path/model.gguf cargo test --release
//! -p magnitude-engine --test optional_memory -- --ignored --nocapture`.

use magnitude_chat::{ByteBpeTokenizer, SpecialTokens};
use magnitude_engine::{
    build_native_domain,
    composition::EngineConfiguration,
    options::{ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection},
};
use magnitude_executor::{
    platform::{DeviceRequest, MemoryReserves},
    ExecutionPath, FeatureRows, Operation, PhysicalDecision, RequestId, TokenId,
};
use magnitude_generation::{
    EndOfGeneration, Generation, InputLayout, MethodChoice, Mtp, Options, Sampling, Shaping,
};
use magnitude_scheduler::{
    domain::{self as service_domain, DomainFlight},
    owner::{Owner, Status, Step},
    retention::{RetentionCapacity, RetentionKey, RetentionRequest, TokenizerIdentity},
    ServiceLimits,
};
use magnitude_state::{CodecIdentity, KvCodec};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

/// A plain-method owner over the test model with retention of up to
/// `retained` prefixes, its vocabulary, the retention key of its model and
/// its tokenizer.
fn plain_owner(
    limits: ServiceLimits,
    retained: usize,
) -> (Owner, usize, RetentionKey, std::sync::Arc<ByteBpeTokenizer>) {
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
    let key = RetentionKey::new(
        resolved.manifest.package.identity.clone(),
        TokenizerIdentity::new("optional-memory").unwrap(),
        CodecIdentity::new("affine-k8v4").unwrap(),
    );
    let tokenizer = resolved.host.shared_tokenizer();
    let package = resolved.host.shared_package();
    let (domain, _) = build_native_domain(&resolved.manifest, package).unwrap();
    let owner = Owner::with_retention_capacity(
        domain,
        limits,
        RetentionCapacity {
            max_entries: retained,
        },
    )
    .unwrap();
    (owner, vocabulary, key, tokenizer)
}

/// The first `len` tokens of a deterministic English text; distinct seeds
/// give distinct texts.
fn prompt(tokenizer: &ByteBpeTokenizer, seed: u32, len: usize) -> Vec<TokenId> {
    let text = (1..=64)
        .map(|line| {
            format!(
                "Ledger {seed}, entry {line}: the warehouse in district {} shipped {} crates \
                 of {} to the harbour before noon.",
                seed * 3 + line,
                line * 7 + seed,
                ["apples", "copper wire", "linen", "glass", "timber"][(line as usize) % 5],
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
    .unwrap()
}

fn retention(key: &RetentionKey, prompt: &[TokenId]) -> RetentionRequest {
    let plan = magnitude_family_contracts::TokenPlan::new(
        prompt.to_vec(),
        InputLayout::new(prompt.len(), vec![]).unwrap(),
    )
    .unwrap();
    RetentionRequest::from_token_plan(key.clone(), &plan).unwrap()
}

/// The owner's clock: every call advances it, as a worker's periodic
/// observations and steps would.
struct Clock(u64);

impl Clock {
    fn tick(&mut self) -> u64 {
        self.0 += 10_000_000;
        self.0
    }
}

/// Step until every request is terminal, draining output as it arrives.
fn drive(owner: &mut Owner, clock: &mut Clock, requests: &[RequestId]) -> Vec<Vec<TokenId>> {
    let mut outputs = vec![Vec::new(); requests.len()];
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while std::time::Instant::now() < deadline {
        if matches!(
            owner.step(clock.tick()).unwrap(),
            Step::Waiting | Step::Idle
        ) {
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
        let mut done = true;
        for (request, output) in requests.iter().zip(&mut outputs) {
            output.extend(
                owner
                    .take(*request, usize::MAX)
                    .unwrap()
                    .into_iter()
                    .map(|token| token.token),
            );
            done &= matches!(owner.status(*request).unwrap(), Status::Terminal(_));
        }
        if done {
            assert!(requests
                .iter()
                .all(|request| owner.error(*request).is_none()));
            return outputs;
        }
    }
    panic!(
        "requests did not finish: {:?}",
        requests
            .iter()
            .map(|request| (owner.status(*request), owner.output_len(*request)))
            .collect::<Vec<_>>()
    );
}

/// Step until a finished request has consumed its last token numerically,
/// so retiring it retains its terminal prefix.
fn reconcile_terminal(owner: &mut Owner, clock: &mut Clock, request: RequestId, tokens: usize) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while std::time::Instant::now() < deadline {
        if owner.resident_position(request).unwrap() == tokens {
            return;
        }
        if matches!(
            owner.step(clock.tick()).unwrap(),
            Step::Waiting | Step::Idle
        ) {
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
    }
    panic!(
        "terminal state did not reconcile: resident at {} of {tokens}",
        owner.resident_position(request).unwrap()
    );
}

/// Admit one prompt with retention, run it to completion and retire it,
/// returning its output and the prompt tokens retention served.
#[cfg(target_os = "linux")]
fn retained_turn(
    owner: &mut Owner,
    clock: &mut Clock,
    key: &RetentionKey,
    vocabulary: usize,
    tokens: &[TokenId],
    max_tokens: usize,
) -> (Vec<TokenId>, usize) {
    let request = owner
        .admit_retained_with(
            greedy(tokens, vocabulary, max_tokens),
            retention(key, tokens),
            clock.tick(),
            |_, _, _, _| Ok(Vec::new()),
        )
        .unwrap();
    let output = drive(owner, clock, &[request]).remove(0);
    reconcile_terminal(owner, clock, request, tokens.len() + output.len());
    let cached = owner.usage(request).unwrap().cached_tokens;
    owner.retire(request).unwrap();
    (output, cached)
}

/// Admit one prompt with retention, take its first `count` tokens and
/// release it before it finishes, so it retains no terminal prefix of its
/// own. Returns those tokens and the prompt tokens retention served.
fn stopped_turn(
    owner: &mut Owner,
    clock: &mut Clock,
    key: &RetentionKey,
    vocabulary: usize,
    tokens: &[TokenId],
    count: usize,
) -> (Vec<TokenId>, usize) {
    let request = owner
        .admit_retained_with(
            greedy(tokens, vocabulary, 4 * count),
            retention(key, tokens),
            clock.tick(),
            |_, _, _, _| Ok(Vec::new()),
        )
        .unwrap();
    let mut output = Vec::new();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while output.len() < count {
        assert!(std::time::Instant::now() < deadline, "the turn stalled");
        if matches!(
            owner.step(clock.tick()).unwrap(),
            Step::Waiting | Step::Idle
        ) {
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
        output.extend(
            owner
                .take(request, usize::MAX)
                .unwrap()
                .into_iter()
                .map(|token| token.token),
        );
    }
    let cached = owner.usage(request).unwrap().cached_tokens;
    owner.release(request).unwrap();
    while owner.status(request).is_ok() {
        assert!(std::time::Instant::now() < deadline, "the release stalled");
        owner.step(clock.tick()).unwrap();
    }
    output.truncate(count);
    (output, cached)
}

/// Step until nothing is live, so the owner shrinks idle state backing.
fn settle(owner: &mut Owner, clock: &mut Clock) {
    for _ in 0..8 {
        owner.step(clock.tick()).unwrap();
    }
}

/// Measures the device charge held by increasingly wide, fully drained
/// cohorts and verifies that idle shrink releases their state backing.
#[test]
#[ignore = "requires a CUDA or Metal device and MAGNITUDE_TEST_MTP_GGUF"]
fn concurrent_cohort_memory_growth_and_idle_release() {
    let limits = ServiceLimits {
        prefill_tokens: 512,
        decode_tokens: 32,
        decode_share: 0.5,
        locality_seconds: 1.0,
    };
    let (mut owner, vocabulary, _, tokenizer) = plain_owner(limits, 0);
    let mut clock = Clock(0);
    let baseline = owner.reconcile_memory_charge().unwrap();
    assert_eq!(baseline.unattributed, 0);
    eprintln!("cohort baseline: {baseline:?}");

    for count in [1, 4, 16] {
        let requests = (0..count)
            .map(|seed| {
                owner
                    .admit(
                        greedy(&prompt(&tokenizer, 1000 + seed, 658), vocabulary, 24),
                        clock.tick(),
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let mut peak = owner.reconcile_memory_charge().unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(90);
        loop {
            assert!(
                std::time::Instant::now() < deadline,
                "cohort {count} stalled"
            );
            if matches!(
                owner.step(clock.tick()).unwrap(),
                Step::Waiting | Step::Idle
            ) {
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
            for &request in &requests {
                owner.take(request, usize::MAX).unwrap();
            }
            let charge = owner.reconcile_memory_charge().unwrap();
            assert_eq!(charge.unattributed, 0);
            if charge.charged > peak.charged {
                peak = charge;
            }
            if requests
                .iter()
                .all(|&request| matches!(owner.status(request).unwrap(), Status::Terminal(_)))
            {
                break;
            }
        }
        let completed = owner.reconcile_memory_charge().unwrap();
        eprintln!("cohort N={count} pre-retirement: peak={peak:?} completed={completed:?}");
        for request in requests {
            assert!(
                owner.error(request).is_none(),
                "cohort {count} request {request:?}: {:?}",
                owner.error(request)
            );
            owner.retire(request).unwrap();
        }
        settle(&mut owner, &mut clock);
        let idle = owner.reconcile_memory_charge().unwrap();
        eprintln!("cohort N={count}: peak={peak:?} completed={completed:?} idle={idle:?}");
        assert_eq!(idle.unattributed, 0);
        assert_eq!(idle.target_state.live, 0);
        assert_eq!(idle.target_state.retained, 0);
        assert!(peak.charged > baseline.charged);
        assert_eq!(idle.charged, baseline.charged);
    }
}

#[test]
#[ignore = "requires a CUDA or Metal device and MAGNITUDE_TEST_MTP_GGUF"]
fn mtp_head_charge_releases_and_reloads_after_idle() {
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
            method: ModelMethod::Mtp,
            mtp_proposals: Some(4),
            kv_codec: KvCodec::AffineK8V4,
            lookahead: false,
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
    let row_bytes = usize::try_from(resolved.manifest.definition.decoder.hidden).unwrap() * 2;
    let package = resolved.host.shared_package();
    let (mut domain, _) = build_native_domain(&resolved.manifest, package).unwrap();
    let charge = |domain: &magnitude_executor::ExecutorDomain| {
        domain.resources().device().memory_usage().charged
    };
    let baseline = charge(&domain);
    let mut first_cycle = None;

    for request in [RequestId(1), RequestId(2)] {
        domain.open(request).unwrap();
        let head = Operation::Head {
            request,
            tokens: vec![TokenId(1)],
            conditioning: FeatureRows::new(vec![0; row_bytes].into(), 1).unwrap(),
            position: 0,
            proposals: Vec::new(),
            form: magnitude_executor::DraftForm::Chained,
        };
        let groups = service_domain::group(&domain, vec![head]);
        let [group] = groups.as_slice() else {
            panic!("one head operation forms one group");
        };
        let DomainFlight::Head(flight) = service_domain::submit_group(&mut domain, group).unwrap()
        else {
            panic!("head operation submitted to another lane");
        };
        for pending in domain.finish_head(flight).unwrap() {
            domain
                .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
                .unwrap();
        }
        let retained = domain.checkpoint(request).unwrap();
        domain.close(request).unwrap();
        let resident = domain.reconcile_memory_charge(&[], &[&retained]).unwrap();
        assert!(resident.optional_weights > 0);
        assert!(charge(&domain) > baseline);
        let before = charge(&domain);
        let released = domain.release_idle_optional_components().unwrap();
        let after = charge(&domain);
        assert!(released > 0, "optional component held no physical charge");
        assert_eq!(before - after, released);
        let idle = domain.reconcile_memory_charge(&[], &[&retained]).unwrap();
        assert_eq!(idle.optional_weights, 0);
        assert_eq!(resident.unattributed, 0);
        assert_eq!(idle.unattributed, 0);
        assert_eq!(
            resident.head_state.unwrap().retained,
            idle.head_state.unwrap().retained
        );
        assert!(idle.head_state.unwrap().retained > 0);
        assert_eq!(
            released,
            resident.optional_weights + resident.bound_constants - idle.bound_constants
        );
        if let Some(charges) = first_cycle {
            assert_eq!(
                (before, after),
                charges,
                "reload changed steady-state charge"
            );
        } else {
            first_cycle = Some((before, after));
        }
        eprintln!(
            "MTP optional charge: loaded={before} released={released} idle={after} optional_weights={} bound_constants_before={} bound_constants_after={} unattributed_before={} unattributed_after={}",
            resident.optional_weights,
            resident.bound_constants,
            idle.bound_constants,
            resident.unattributed,
            idle.unattributed,
        );
        drop(retained);
    }
}

/// Exercise lazy head binding after several target prefills have entered the
/// owner's ordinary scheduling path. Direct head submission does not cover
/// the state and graph holdings already committed by those target rounds.
#[test]
#[ignore = "requires a CUDA or Metal device and MAGNITUDE_TEST_MTP_GGUF"]
fn concurrent_mtp_first_head_bind_reconciles_memory() {
    let model = PathBuf::from(
        std::env::var_os("MAGNITUDE_TEST_MTP_GGUF").expect("set MAGNITUDE_TEST_MTP_GGUF"),
    );
    let limits = magnitude_engine::options::standard_service_limits();
    let resolved = EngineConfiguration {
        package: PackageOptions {
            target: model,
            projector: ProjectorSelection::Disabled,
            draft: None,
        },
        model: ModelPolicy {
            method: ModelMethod::Mtp,
            mtp_proposals: Some(4),
            kv_codec: KvCodec::AffineK8V4,
            lookahead: false,
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
    let (domain, _) = build_native_domain(&resolved.manifest, package).unwrap();
    let mut owner =
        Owner::with_retention_capacity(domain, limits, RetentionCapacity { max_entries: 0 })
            .unwrap();
    let mut clock = Clock(0);
    let method = Arc::new(Mtp::new("optional-memory-first-bind", 4).unwrap());
    let requests = (0..4)
        .map(|seed| {
            let tokens = prompt(&tokenizer, seed, 22);
            let generation = Generation::new_with_method(
                tokens.clone(),
                InputLayout::new(tokens.len(), vec![]).unwrap(),
                Options {
                    max_tokens: 16,
                    output_capacity: 16,
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
                    method: MethodChoice::Mtp { proposals: 4 },
                    end_of_generation: EndOfGeneration::Suppress,
                    reasoning_budget: None,
                },
                None,
                method.clone(),
            )
            .unwrap();
            owner.admit(generation, clock.tick()).unwrap()
        })
        .collect::<Vec<_>>();
    let outputs = drive(&mut owner, &mut clock, &requests);
    assert!(outputs.iter().all(|tokens| !tokens.is_empty()));
    let charge = owner.reconcile_memory_charge().unwrap();
    assert!(charge.optional_weights > 0);
    assert_eq!(charge.unattributed, 0);
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires Sparky's unified CUDA device and MAGNITUDE_TEST_MTP_GGUF"]
fn admission_reclaims_under_a_real_process_limit() {
    use magnitude_executor::platform::{self, DomainReading, MemoryBand, MemoryConstraint};
    use magnitude_scheduler::worker::Driven;
    use seismic::DeviceCatalog;

    struct AddressSpaceLimit(libc::rlimit);
    impl Drop for AddressSpaceLimit {
        fn drop(&mut self) {
            assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_AS, &self.0) }, 0);
        }
    }

    let model = PathBuf::from(
        std::env::var_os("MAGNITUDE_TEST_MTP_GGUF").expect("set MAGNITUDE_TEST_MTP_GGUF"),
    );
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
    let package = resolved.host.shared_package();
    let (domain, _) = build_native_domain(&resolved.manifest, package).unwrap();
    let catalog = DeviceCatalog::discover().unwrap();
    let reserves = MemoryReserves::standard();
    let allocation = |device: &seismic::Device| -> Result<DomainReading, String> {
        let readings = platform::observe_domains(&catalog, device, &reserves)
            .map_err(|error| error.to_string())?;
        Ok(readings[0])
    };
    let available = allocation(domain.resources().device()).unwrap();
    assert_eq!(available.constraint, MemoryConstraint::HostRam);
    let planning = available.thresholds.planning_bytes;
    let seed_bytes = domain.state_holding_census(&[], &[]).unwrap().0.model_seed;
    assert!(seed_bytes > 0, "fixture needs recurrent banks");
    let mut owner = Owner::new(domain, limits).unwrap();
    let generation = |prompt_tokens: usize| {
        Generation::new(
            vec![TokenId(1); prompt_tokens],
            InputLayout::new(prompt_tokens, vec![]).unwrap(),
            Options {
                max_tokens: 64,
                output_capacity: 16,
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
                end_of_generation: EndOfGeneration::Stop,
                reasoning_budget: None,
            },
            None,
        )
        .unwrap()
    };
    // Keep several requests live with decoded output so a later growth
    // deficit has eligible replayable victims.
    let existing = (0..5)
        .map(|_| owner.admit(generation(1), 0).unwrap())
        .collect::<Vec<_>>();
    for tick in 0..128 {
        if existing
            .iter()
            .all(|request| owner.output_len(*request).unwrap() > 0)
        {
            break;
        }
        owner.step(tick * 1_000_000_000).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(existing
        .iter()
        .all(|request| owner.output_len(*request).unwrap() > 0));
    let before = owner
        .inspect_domain(|domain| domain.reconcile_memory_charge(&[], &[]))
        .unwrap();

    let vm_kib = std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find_map(|line| line.strip_prefix("VmSize:"))
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .parse::<u64>()
        .unwrap();
    let vm_bytes = vm_kib.checked_mul(1024).unwrap();
    let mut previous = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    assert_eq!(
        unsafe { libc::getrlimit(libc::RLIMIT_AS, &mut previous) },
        0
    );
    let headroom = std::env::var("MAGNITUDE_TEST_PROCESS_HEADROOM_MIB")
        .ok()
        .map(|value| value.parse::<u64>().expect("process headroom must be MiB"))
        .unwrap_or(2)
        * 1024
        * 1024;
    // The limit leaves the requested headroom above the planning reserve,
    // so the incoming request meets a demand deficit, not the Reclaim band.
    let limit = vm_bytes
        .checked_add(planning)
        .and_then(|bytes| bytes.checked_add(headroom))
        .unwrap();
    assert!(limit < previous.rlim_cur && limit <= previous.rlim_max);
    let restricted = libc::rlimit {
        rlim_cur: limit,
        rlim_max: previous.rlim_max,
    };
    assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_AS, &restricted) }, 0);
    let guard = AddressSpaceLimit(previous);
    let observed = owner
        .inspect_domain(|domain| allocation(domain.resources().device()))
        .unwrap();
    eprintln!(
        "admission deficit setup: seed={seed_bytes} vm={vm_bytes} limit={limit} available={} charged={} state={:?}",
        observed.ceiling_bytes, before.charged, before.target_state
    );
    assert!(observed.ceiling_bytes <= headroom);
    assert_eq!(observed.band, MemoryBand::Normal);
    let prompt_tokens = std::env::var("MAGNITUDE_TEST_INCOMING_PROMPT_TOKENS")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("incoming prompt tokens must be an integer")
        })
        .unwrap_or(256);
    let incoming = owner
        .admit(generation(prompt_tokens), 128 * 1_000_000_000)
        .unwrap();
    let mut saw_preemption = false;
    let mut lowest_charge = before.charged;
    let mut in_limit_output = 0;
    // Drive the owner as its worker does: a periodic memory observation,
    // then a step.
    for tick in 128..192 {
        Driven::periodic(&mut owner, tick * 1_000_000_000).unwrap();
        owner.step(tick * 1_000_000_000).unwrap();
        saw_preemption |= existing
            .iter()
            .any(|request| owner.status(*request).unwrap() == Status::Preempted);
        lowest_charge = lowest_charge.min(
            owner
                .inspect_domain(|domain| Ok(domain.resources().device().memory_usage().charged))
                .unwrap(),
        );
        in_limit_output = owner.output_len(incoming).unwrap();
        if in_limit_output > 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let in_limit_status = owner.status(incoming).unwrap();
    let in_limit_charge = owner
        .inspect_domain(|domain| Ok(domain.resources().device().memory_usage().charged))
        .unwrap();
    eprintln!(
        "admission in-limit: headroom={headroom} prompt_tokens={prompt_tokens} incoming={in_limit_status:?} output={in_limit_output} saw_preemption={saw_preemption} charged={in_limit_charge} lowest_charge={lowest_charge}",
    );
    drop(guard);
    for tick in 192..256 {
        if owner.output_len(incoming).unwrap() > 0 {
            break;
        }
        Driven::periodic(&mut owner, tick * 1_000_000_000).unwrap();
        owner.step(tick * 1_000_000_000).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let after = owner
        .inspect_domain(|domain| domain.reconcile_memory_charge(&[], &[]))
        .unwrap();
    eprintln!(
        "admission deficit result: saw_preemption={saw_preemption} existing={:?} errors={:?} incoming={:?} output={} charged_before={} lowest_charge={} charged_after={} state_after={:?} in_flight_before={} in_flight_after={} unattributed_before={} unattributed_after={}",
        existing.iter().map(|request| owner.status(*request).unwrap()).collect::<Vec<_>>(),
        existing.iter().map(|request| owner.error(*request).map(|error| format!("{error:?}"))).collect::<Vec<_>>(),
        owner.status(incoming).unwrap(),
        owner.output_len(incoming).unwrap(),
        before.charged,
        lowest_charge,
        after.charged,
        after.target_state,
        before.target_state.in_flight,
        after.target_state.in_flight,
        before.unattributed,
        after.unattributed,
    );
    assert!(saw_preemption);
    assert!(lowest_charge < before.charged);
    assert!(owner.output_len(incoming).unwrap() > 0);
    assert!(existing
        .iter()
        .all(|request| owner.error(*request).is_none()));
    assert_eq!(after.unattributed, before.unattributed);
}

/// A process limit that leaves headroom below the planning reserve puts the
/// engine in the Reclaim band: admission is refused, releases run, and when
/// they cannot restore headroom the model unloads with the memory-pressure
/// cause within the escalation interval, finishing open requests with it.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires Sparky's unified CUDA device and MAGNITUDE_TEST_MTP_GGUF"]
fn reclaim_band_releases_then_unloads_under_a_real_process_limit() {
    use magnitude_executor::platform::{self, MemoryBand};
    use magnitude_scheduler::owner::AdmissionError;
    use magnitude_scheduler::publication::{ModelUnloadCause, RequestError};
    use magnitude_scheduler::worker::Driven;
    use seismic::DeviceCatalog;

    struct AddressSpaceLimit(libc::rlimit);
    impl Drop for AddressSpaceLimit {
        fn drop(&mut self) {
            assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_AS, &self.0) }, 0);
        }
    }

    let model = PathBuf::from(
        std::env::var_os("MAGNITUDE_TEST_MTP_GGUF").expect("set MAGNITUDE_TEST_MTP_GGUF"),
    );
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
    let package = resolved.host.shared_package();
    let (domain, _) = build_native_domain(&resolved.manifest, package).unwrap();
    let catalog = DeviceCatalog::discover().unwrap();
    let reserves = MemoryReserves::standard();
    let planning = platform::observe_domains(&catalog, domain.resources().device(), &reserves)
        .unwrap()[0]
        .thresholds
        .planning_bytes;
    let mut owner = Owner::new(domain, limits).unwrap();
    let generation = || {
        Generation::new(
            vec![TokenId(1); 8],
            InputLayout::new(8, vec![]).unwrap(),
            Options {
                max_tokens: 4096,
                output_capacity: 16,
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
                end_of_generation: EndOfGeneration::Stop,
                reasoning_budget: None,
            },
            None,
        )
        .unwrap()
    };
    let open = (0..2)
        .map(|_| owner.admit(generation(), 0).unwrap())
        .collect::<Vec<_>>();
    for tick in 0..64 {
        if open
            .iter()
            .all(|request| owner.output_len(*request).unwrap() > 0)
        {
            break;
        }
        owner.step(tick * 1_000_000_000).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    // Leave half the planning reserve of address space: headroom is now at
    // or below the planning reserve, which only an external limit can cause.
    let vm_bytes = std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find_map(|line| line.strip_prefix("VmSize:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|kib| kib.parse::<u64>().ok())
        .unwrap()
        * 1024;
    let mut previous = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    assert_eq!(
        unsafe { libc::getrlimit(libc::RLIMIT_AS, &mut previous) },
        0
    );
    let limit = vm_bytes + planning / 2;
    assert!(limit < previous.rlim_cur && limit <= previous.rlim_max);
    let restricted = libc::rlimit {
        rlim_cur: limit,
        rlim_max: previous.rlim_max,
    };
    assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_AS, &restricted) }, 0);
    let guard = AddressSpaceLimit(previous);
    let band = owner
        .inspect_domain(|domain| {
            platform::observe_domains(&catalog, domain.resources().device(), &reserves)
                .map_err(|error| error.to_string())
        })
        .unwrap()[0]
        .band;
    assert_eq!(band, MemoryBand::Reclaim);

    // Periodic observation enters Reclaim; admission is refused while it lasts.
    let start = 64 * 1_000_000_000;
    owner.periodic(start).unwrap();
    assert!(matches!(
        owner.admit(generation(), start),
        Err(AdmissionError::MemoryReclaim)
    ));
    // Releases cannot restore half a planning reserve, so the model unloads
    // once Reclaim outlasts the escalation interval after releases.
    let mut unloaded_at = None;
    for step in 1..=40u64 {
        let now = start + step * 100_000_000;
        owner.periodic(now).unwrap();
        owner.step(now).unwrap();
        if owner.memory_unloading() {
            unloaded_at = Some(now - start);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    drop(guard);
    eprintln!(
        "reclaim band: planning={planning} vm={vm_bytes} limit={limit} unloaded_after_ns={unloaded_at:?} errors={:?}",
        open.iter()
            .map(|request| owner.error(*request).map(|error| format!("{error:?}")))
            .collect::<Vec<_>>()
    );
    let unloaded_after = unloaded_at.expect("persistent Reclaim unloads the model");
    assert!(unloaded_after >= 1_000_000_000);
    assert!(open.iter().all(|request| matches!(
        owner.error(*request),
        Some(RequestError::ModelUnloaded {
            cause: ModelUnloadCause::MemoryPressure
        })
    )));
    assert!(matches!(
        owner.admit(generation(), start + 5_000_000_000),
        Err(AdmissionError::ModelUnloaded {
            cause: ModelUnloadCause::MemoryPressure
        })
    ));
}

/// Idle slab release on the real device. Seven peers finish and stay open,
/// so the stores hold many rows and banks while a session runs above them.
/// After the peers retire, idle shrink frees their empty slabs. The retained
/// session still serves the same next turn and every charged byte is attributed.
#[test]
#[ignore = "requires a CUDA or Metal device and MAGNITUDE_TEST_MTP_GGUF"]
fn idle_shrink_preserves_a_retained_session() {
    let limits = ServiceLimits {
        prefill_tokens: 256,
        decode_tokens: 16,
        decode_share: 0.5,
        locality_seconds: 1.0,
    };
    let (mut owner, vocabulary, key, tokenizer) = plain_owner(limits, 4);
    let mut clock = Clock(0);
    let session = prompt(&tokenizer, 0, 96);
    // The peers finish but remain open, holding the low rows and banks; every
    // bank the session and its next turn acquire lies above theirs.
    let peers = (1..8)
        .map(|seed| {
            owner
                .admit(
                    greedy(&prompt(&tokenizer, seed, 96), vocabulary, 96),
                    clock.tick(),
                )
                .unwrap()
        })
        .collect::<Vec<_>>();
    drive(&mut owner, &mut clock, &peers);
    let first = owner
        .admit_retained_with(
            greedy(&session, vocabulary, 24),
            retention(&key, &session),
            clock.tick(),
            |_, _, _, _| Ok(Vec::new()),
        )
        .unwrap();
    let output = drive(&mut owner, &mut clock, &[first]).remove(0);
    // Retain the session's terminal prefix while its peers stay open.
    let resumed = session.len() + output.len();
    reconcile_terminal(&mut owner, &mut clock, first, resumed);
    owner.retire(first).unwrap();
    let next_turn = session
        .iter()
        .chain(&output)
        .copied()
        .chain([TokenId(4242)])
        .collect::<Vec<_>>();
    // The peers stay open. Both turns below run alone, in the same launch
    // shapes, so their numerics are comparable bit for bit.
    // Stopped turns retain nothing of their own: the session's prompt and
    // terminal prefixes stay the only retained state.
    let (reference, cached) =
        stopped_turn(&mut owner, &mut clock, &key, vocabulary, &next_turn, 16);
    assert_eq!(cached, resumed, "the next turn resumes from the session");
    assert_eq!(owner.retained_entries(), 2);
    let before = owner.reconcile_memory_charge().unwrap();
    for &request in &peers {
        owner.retire(request).unwrap();
    }
    settle(&mut owner, &mut clock);
    let after = owner.reconcile_memory_charge().unwrap();
    eprintln!(
        "idle shrink: retained={} charged {} -> {} state {:?} -> {:?} unattributed {} -> {}",
        owner.retained_entries(),
        before.charged,
        after.charged,
        before.target_state,
        after.target_state,
        before.unattributed,
        after.unattributed,
    );
    // Idle shrink releases empty peer slabs while preserving the session's
    // retained history for the turn below.
    assert!(after.charged < before.charged);
    assert_eq!((before.unattributed, after.unattributed), (0, 0));
    let (replayed, cached) = stopped_turn(&mut owner, &mut clock, &key, vocabulary, &next_turn, 16);
    eprintln!("idle shrink replay: reference={reference:?} replayed={replayed:?}");
    assert_eq!(
        cached, resumed,
        "the retained session still serves the turn"
    );
    assert_eq!(replayed, reference);
    assert_eq!(owner.reconcile_memory_charge().unwrap().unattributed, 0);
}

/// Retained-state release in the Reclaim band, with replay. A session's
/// next turn is computed cold and then served from its retained prefix.
/// A process limit then leaves headroom below the planning reserve: the
/// Reclaim releases evict every retained prefix, and Seismic's charge falls
/// by the released state with every byte still attributed. Once headroom
/// returns within the unload interval, the same turn has nothing retained
/// to resume from, is replayed in full, and produces the cold output.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires Sparky's unified CUDA device and MAGNITUDE_TEST_MTP_GGUF"]
fn reclaim_releases_retained_sessions_and_replays_them() {
    use magnitude_executor::platform::{self, MemoryBand};
    use magnitude_scheduler::worker::Driven;
    use seismic::DeviceCatalog;

    struct AddressSpaceLimit(libc::rlimit);
    impl Drop for AddressSpaceLimit {
        fn drop(&mut self) {
            assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_AS, &self.0) }, 0);
        }
    }

    let limits = ServiceLimits {
        prefill_tokens: 64,
        decode_tokens: 16,
        decode_share: 0.5,
        locality_seconds: 1.0,
    };
    let (mut owner, vocabulary, key, tokenizer) = plain_owner(limits, 8);
    let mut clock = Clock(0);
    let session = prompt(&tokenizer, 0, 96);
    let (first, _) = retained_turn(&mut owner, &mut clock, &key, vocabulary, &session, 16);
    let resumed = session.len() + first.len();
    let next_turn = session
        .iter()
        .chain(&first)
        .copied()
        .chain([TokenId(4242)])
        .collect::<Vec<_>>();
    // Without a retention request nothing is looked up or retained.
    let cold_request = owner
        .admit(greedy(&next_turn, vocabulary, 16), clock.tick())
        .unwrap();
    let cold = drive(&mut owner, &mut clock, &[cold_request]).remove(0);
    assert_eq!(owner.usage(cold_request).unwrap().cached_tokens, 0);
    owner.retire(cold_request).unwrap();
    let (served, cached) = retained_turn(&mut owner, &mut clock, &key, vocabulary, &next_turn, 16);
    assert_eq!(cached, resumed);
    settle(&mut owner, &mut clock);
    let retained = owner.retained_entries();
    let before = owner.reconcile_memory_charge().unwrap();
    assert!(retained >= 2 && before.target_state.retained > 0);
    assert_eq!(before.unattributed, 0);

    let catalog = DeviceCatalog::discover().unwrap();
    let reserves = MemoryReserves::standard();
    let band = |owner: &Owner| {
        owner
            .inspect_domain(|domain| {
                platform::observe_domains(&catalog, domain.resources().device(), &reserves)
                    .map_err(|error| error.to_string())
            })
            .unwrap()[0]
    };
    let planning = band(&owner).thresholds.planning_bytes;
    let vm_bytes = std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find_map(|line| line.strip_prefix("VmSize:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|kib| kib.parse::<u64>().ok())
        .unwrap()
        * 1024;
    let mut previous = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    assert_eq!(
        unsafe { libc::getrlimit(libc::RLIMIT_AS, &mut previous) },
        0
    );
    let restricted = libc::rlimit {
        rlim_cur: vm_bytes + planning / 2,
        rlim_max: previous.rlim_max,
    };
    assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_AS, &restricted) }, 0);
    let guard = AddressSpaceLimit(previous);
    assert_eq!(band(&owner).band, MemoryBand::Reclaim);
    owner.periodic(clock.tick()).unwrap();
    let released = owner.reconcile_memory_charge().unwrap();
    let unloading = owner.memory_unloading();
    drop(guard);
    owner.periodic(clock.tick()).unwrap();
    assert_eq!(band(&owner).band, MemoryBand::Normal);
    eprintln!(
        "reclaim retained release: entries {retained} -> {} charged {} -> {} state {:?} -> {:?} \
         unattributed {} -> {} unloading={unloading}",
        owner.retained_entries(),
        before.charged,
        released.charged,
        before.target_state,
        released.target_state,
        before.unattributed,
        released.unattributed,
    );
    assert!(!unloading, "headroom returned within the unload interval");
    assert_eq!(owner.retained_entries(), 0);
    assert_eq!(released.target_state.retained, 0);
    assert!(released.charged < before.charged);
    assert_eq!(released.unattributed, 0);

    let (replayed, cached) =
        retained_turn(&mut owner, &mut clock, &key, vocabulary, &next_turn, 16);
    eprintln!(
        "reclaim replay: cold={cold:?} served={served:?} replayed={replayed:?} \
         served_matches_cold={}",
        served == cold
    );
    assert_eq!(cached, 0, "nothing retained survived the Reclaim release");
    assert_eq!(replayed, cold);
    assert_eq!(owner.reconcile_memory_charge().unwrap().unattributed, 0);
}
