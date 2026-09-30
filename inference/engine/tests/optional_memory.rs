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
    ExecutionPath, FeatureRows, Operation, PhysicalDecision, RequestId, ReservedResources,
    TokenId,
};
use magnitude_family_contracts::{PreparedModelInput, TokenPlan};
use magnitude_generation::{
    EndOfGeneration, Generation, InputLayout, MethodChoice, Mtp, Options, Sampling, Shaping,
};
use magnitude_scheduler::{
    owner::{AdmissionError, Owner, Status},
    prefix_cache::PrefixCacheCapacity,
    ServiceLimits,
};
use magnitude_state::KvCodec;
use owner_host::{Host, Stream};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

mod owner_host;

/// A plain-method owner over the test model caching up to `cached`
/// prefixes, its vocabulary and its tokenizer.
fn plain_owner(
    limits: ServiceLimits,
    cached: usize,
) -> (Host, usize, std::sync::Arc<ByteBpeTokenizer>) {
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
    let tokenizer = resolved.host.shared_tokenizer();
    let package = resolved.host.shared_package();
    let (domain, bindings, _) = build_native_domain(&resolved.manifest, package).unwrap();
    let host = Host::new(|wakes| {
        Owner::with_prefix_cache_capacity(
            domain,
            bindings,
            limits,
            PrefixCacheCapacity {
                max_entries: cached,
            },
            wakes,
        )
    });
    (host, vocabulary, tokenizer)
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

/// Admit `generation` with the text input of its prompt, as the test
/// model's family prepares one: each row at its own position. The host's
/// stream holds `output_capacity` publications.
fn admit(
    host: &mut Host,
    generation: Generation,
    prefix_cache: bool,
    output_capacity: usize,
) -> Result<Stream, AdmissionError> {
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
    host.admit(generation, input, prefix_cache, output_capacity)
}

/// Whether the owner holds `stream`'s request resident and blocked on its
/// host's output credit: live, with decoded output, and not finished.
fn output_blocked(host: &Host, stream: &Stream) -> bool {
    host.snapshot(stream.id)
        .is_some_and(|snapshot| snapshot.status == Status::OutputBlocked)
}

/// Admit one prompt with retention and run it to completion. A request
/// completes only once its terminal prefix is retained, and is retired
/// then. Returns its output and the prompt tokens retention served.
#[cfg(target_os = "linux")]
fn retained_turn(
    host: &mut Host,
    vocabulary: usize,
    tokens: &[TokenId],
    max_tokens: usize,
) -> (Vec<TokenId>, usize) {
    let mut stream = admit(
        host,
        greedy(tokens, vocabulary, max_tokens),
        true,
        max_tokens,
    )
    .unwrap();
    host.drive(&mut [&mut stream], Duration::from_secs(60));
    let cached = stream.usage().cached_tokens;
    (stream.output, cached)
}

/// Admit one prompt with retention, take its first `count` tokens and
/// release it before it finishes, so it retains no terminal prefix of its
/// own. Returns those tokens and the prompt tokens retention served.
fn stopped_turn(
    host: &mut Host,
    vocabulary: usize,
    tokens: &[TokenId],
    count: usize,
) -> (Vec<TokenId>, usize) {
    let mut stream = admit(
        host,
        greedy(tokens, vocabulary, 4 * count),
        true,
        4 * count,
    )
    .unwrap();
    host.step_until(&mut [&mut stream], Duration::from_secs(60), |_, streams| {
        streams[0].output.len() >= count
    });
    assert!(!stream.finished(), "the turn finished before its release");
    let cached = host
        .snapshot(stream.id)
        .expect("a live request")
        .cached_tokens;
    let mut output = stream.output.clone();
    host.release(stream, Duration::from_secs(60));
    output.truncate(count);
    (output, cached)
}

/// Measures the device charge held by increasingly wide, fully drained
/// cohorts and verifies that idle shrink releases their state backing once
/// they have completed and retired.
#[test]
#[ignore = "requires a CUDA or Metal device and MAGNITUDE_TEST_MTP_GGUF"]
fn concurrent_cohort_memory_growth_and_idle_release() {
    let limits = ServiceLimits {
        prefill_tokens: 512,
        decode_tokens: 32,
        decode_share: 0.5,
        locality_seconds: 1.0,
    };
    let (mut host, vocabulary, tokenizer) = plain_owner(limits, 0);
    let baseline = host.owner().reconcile_memory_charge().unwrap();
    assert_eq!(baseline.unattributed, 0);
    eprintln!("cohort baseline: {baseline:?}");

    for count in [1, 4, 16] {
        let mut streams = (0..count)
            .map(|seed| {
                admit(
                    &mut host,
                    greedy(&prompt(&tokenizer, 1000 + seed, 658), vocabulary, 24),
                    false,
                    24,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let mut peak = host.owner().reconcile_memory_charge().unwrap();
        host.step_until(
            &mut streams.iter_mut().collect::<Vec<_>>(),
            Duration::from_secs(90),
            |host, streams| {
                let charge = host.owner().reconcile_memory_charge().unwrap();
                assert_eq!(charge.unattributed, 0);
                if charge.charged > peak.charged {
                    peak = charge;
                }
                streams.iter().all(|stream| stream.finished())
            },
        );
        let completed = host.owner().reconcile_memory_charge().unwrap();
        for stream in &streams {
            assert!(
                stream.failure().is_none(),
                "cohort {count} request {:?}: {:?}",
                stream.id,
                stream.failure()
            );
            assert_eq!(stream.usage().completion_tokens, 24);
            assert!(host.snapshot(stream.id).is_none(), "a completed request is retired");
        }
        host.settle();
        let idle = host.owner().reconcile_memory_charge().unwrap();
        eprintln!("cohort N={count}: peak={peak:?} completed={completed:?} idle={idle:?}");
        assert_eq!(idle.unattributed, 0);
        assert_eq!(idle.target_state.live, 0);
        assert_eq!(idle.target_state.retained, 0);
        // Small cohorts fit the spare state held at load; the largest grows.
        if count == 16 {
            assert!(peak.charged > baseline.charged);
        }
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
    let (mut domain, mut bindings, _) = build_native_domain(&resolved.manifest, package).unwrap();
    let charge = |domain: &magnitude_executor::ExecutorDomain| {
        domain.resources().device().memory_usage().charged
    };
    let baseline = charge(&domain);
    let mut first_cycle = None;

    for request in [RequestId(1), RequestId(2)] {
        domain
            .install_input(request, PreparedModelInput::continuation_only())
            .unwrap();
        domain.open_state(&mut bindings, request, None).unwrap();
        let head = Operation::Head {
            request,
            phase: magnitude_executor::HeadPhase::Priming { draft_from: 1 },
            tokens: vec![TokenId(1)],
            conditioning: FeatureRows::new(vec![0; row_bytes].into(), 1).unwrap(),
            position: 0,
            proposals: Vec::new(),
            form: magnitude_executor::DraftForm::Chained,
        };
        let operations = [head];
        let ReservedResources::Head(workspace, output, advances) = domain
            .reserve(&mut bindings, &operations)
            .unwrap()
            .into_resources()
        else {
            panic!("head operation reserved another lane");
        };
        let flight = domain
            .submit_head(bindings, &operations, workspace, output, advances)
            .unwrap_or_else(|failure| panic!("{}", failure.error()));
        let (pending, next) = domain.finish_head(flight).unwrap();
        bindings = next;
        for pending in pending {
            domain
                .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
                .unwrap();
        }
        let retained = domain.resume_state(request).unwrap();
        domain.close(request).unwrap();
        let resident = domain.reconcile_memory_charge(&[&retained]).unwrap();
        assert!(resident.optional_weights > 0);
        assert!(charge(&domain) > baseline);
        let before = charge(&domain);
        let released = domain
            .release_idle_optional_components(&mut bindings)
            .unwrap();
        let after = charge(&domain);
        assert!(released > 0, "optional component held no physical charge");
        assert_eq!(before - after, released);
        let idle = domain.reconcile_memory_charge(&[&retained]).unwrap();
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
    let (domain, bindings, _) = build_native_domain(&resolved.manifest, package).unwrap();
    let mut host = Host::new(|wakes| Owner::new(domain, bindings, limits, wakes));
    let method = Arc::new(Mtp::new("optional-memory-first-bind", 4).unwrap());
    let mut streams = (0..4)
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
            admit(&mut host, generation, false, 16).unwrap()
        })
        .collect::<Vec<_>>();
    host.drive(
        &mut streams.iter_mut().collect::<Vec<_>>(),
        Duration::from_secs(60),
    );
    assert!(streams.iter().all(|stream| !stream.output.is_empty()));
    let charge = host.owner().reconcile_memory_charge().unwrap();
    assert!(charge.optional_weights > 0);
    assert_eq!(charge.unattributed, 0);
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires Sparky's unified CUDA device and MAGNITUDE_TEST_MTP_GGUF"]
fn admission_reclaims_under_a_real_process_limit() {
    use magnitude_executor::platform::{self, DomainReading, MemoryBand, MemoryConstraint};
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
    let (domain, bindings, _) = build_native_domain(&resolved.manifest, package).unwrap();
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
    let seed_bytes = domain.state_holding_census(&[]).unwrap().0.model_seed;
    assert!(seed_bytes > 0, "fixture needs recurrent banks");
    let mut host = Host::new(|wakes| Owner::new(domain, bindings, limits, wakes));
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
    // deficit has eligible replayable victims. Their hosts never drain, so
    // each stays resident, blocked on output credit once its stream fills.
    let mut existing = (0..5)
        .map(|_| admit(&mut host, generation(1), false, 16).unwrap())
        .collect::<Vec<_>>();
    for tick in 0..128 {
        if existing.iter().all(|stream| output_blocked(&host, stream)) {
            break;
        }
        host.run_at(tick * 1_000_000_000);
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        existing.iter().all(|stream| output_blocked(&host, stream)),
        "existing requests: {:?}",
        existing
            .iter()
            .map(|stream| host.snapshot(stream.id))
            .collect::<Vec<_>>()
    );
    let before = host
        .owner()
        .inspect_domain(|domain| domain.reconcile_memory_charge(&[]))
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
    let observed = host
        .owner()
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
    // Admission is logical: it succeeds at once, and the deficit is met when
    // a round makes the request resident.
    let mut incoming = admit(&mut host, generation(prompt_tokens), false, 16).unwrap();
    let charged = |host: &Host| {
        host.owner()
            .inspect_domain(|domain| Ok(domain.resources().device().memory_usage().charged))
            .unwrap()
    };
    let mut saw_preemption = false;
    let mut lowest_charge = before.charged;
    // Drive the owner as its worker does: each run observes memory when due,
    // then runs every possible transition.
    for tick in 128..192 {
        host.run_at(tick * 1_000_000_000);
        saw_preemption |= existing.iter().any(|stream| {
            host.snapshot(stream.id)
                .is_some_and(|snapshot| snapshot.status == Status::Preempted)
        });
        lowest_charge = lowest_charge.min(charged(&host));
        incoming.drain();
        if !incoming.output.is_empty() || incoming.finished() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let in_limit_output = incoming.output.len();
    let in_limit_snapshot = host.snapshot(incoming.id);
    let in_limit_charge = charged(&host);
    eprintln!(
        "admission in-limit: headroom={headroom} prompt_tokens={prompt_tokens} incoming={in_limit_snapshot:?} terminal={:?} output={in_limit_output} saw_preemption={saw_preemption} charged={in_limit_charge} lowest_charge={lowest_charge}",
        incoming.terminal,
    );
    drop(guard);
    for tick in 192..256 {
        if !incoming.output.is_empty() || incoming.finished() {
            break;
        }
        host.run_at(tick * 1_000_000_000);
        incoming.drain();
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let after = host
        .owner()
        .inspect_domain(|domain| domain.reconcile_memory_charge(&[]))
        .unwrap();
    let existing_snapshots = existing
        .iter()
        .map(|stream| host.snapshot(stream.id))
        .collect::<Vec<_>>();
    existing.iter_mut().for_each(Stream::drain);
    eprintln!(
        "admission deficit result: saw_preemption={saw_preemption} existing={existing_snapshots:?} errors={:?} incoming={:?} incoming_error={:?} output={} charged_before={} lowest_charge={} charged_after={} state_after={:?} in_flight_before={} in_flight_after={} unattributed_before={} unattributed_after={}",
        existing.iter().map(|stream| stream.failure().map(|error| format!("{error:?}"))).collect::<Vec<_>>(),
        host.snapshot(incoming.id),
        incoming.failure(),
        incoming.output.len(),
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
    assert!(!incoming.output.is_empty());
    assert!(incoming.failure().is_none());
    assert!(existing.iter().all(|stream| stream.failure().is_none()));
    assert_eq!(after.unattributed, before.unattributed);
}

/// A process limit that leaves headroom below the planning reserve puts the
/// engine in the Reclaim band: admission is refused, releases run, and when
/// they cannot restore headroom the owner decides to unload with the
/// memory-pressure cause within the escalation interval, finishing open
/// requests with it.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires Sparky's unified CUDA device and MAGNITUDE_TEST_MTP_GGUF"]
fn reclaim_band_releases_then_unloads_under_a_real_process_limit() {
    use magnitude_executor::platform::{self, MemoryBand};
    use magnitude_scheduler::publication::{ModelUnloadCause, RequestError};
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
    let (domain, bindings, _) = build_native_domain(&resolved.manifest, package).unwrap();
    let catalog = DeviceCatalog::discover().unwrap();
    let reserves = MemoryReserves::standard();
    let planning = platform::observe_domains(&catalog, domain.resources().device(), &reserves)
        .unwrap()[0]
        .thresholds
        .planning_bytes;
    let mut host = Host::new(|wakes| Owner::new(domain, bindings, limits, wakes));
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
    let mut open = (0..2)
        .map(|_| admit(&mut host, generation(), false, 16).unwrap())
        .collect::<Vec<_>>();
    for tick in 0..64 {
        open.iter_mut().for_each(Stream::drain);
        if open.iter().all(|stream| !stream.output.is_empty()) {
            break;
        }
        host.run_at(tick * 1_000_000_000);
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(open
        .iter()
        .all(|stream| !stream.output.is_empty() && !stream.finished()));

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
    let band = host
        .owner()
        .inspect_domain(|domain| {
            platform::observe_domains(&catalog, domain.resources().device(), &reserves)
                .map_err(|error| error.to_string())
        })
        .unwrap()[0]
        .band;
    assert_eq!(band, MemoryBand::Reclaim);

    // The observation at `start` enters Reclaim; admission is refused while
    // it lasts.
    let start = 64 * 1_000_000_000;
    assert!(!host.run_at(start).unload);
    assert!(matches!(
        admit(&mut host, generation(), false, 16),
        Err(AdmissionError::MemoryReclaim)
    ));
    // Releases cannot restore half a planning reserve, so the owner decides
    // to unload once Reclaim outlasts the escalation interval after releases.
    let mut unloaded_at = None;
    for step in 1..=40u64 {
        let now = start + step * 100_000_000;
        if host.run_at(now).unload {
            unloaded_at = Some(now - start);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    drop(guard);
    open.iter_mut().for_each(Stream::drain);
    eprintln!(
        "reclaim band: planning={planning} vm={vm_bytes} limit={limit} unloaded_after_ns={unloaded_at:?} terminals={:?}",
        open.iter().map(|stream| &stream.terminal).collect::<Vec<_>>()
    );
    let unloaded_after = unloaded_at.expect("persistent Reclaim unloads the model");
    assert!(unloaded_after >= 1_000_000_000);
    assert!(open.iter().all(|stream| matches!(
        stream.failure(),
        Some(RequestError::ModelUnloaded {
            cause: ModelUnloadCause::MemoryPressure
        })
    )));
}

/// Idle slab release on the real device. Seven peers stay open, resident and
/// blocked on output credit their hosts never grant, so the stores hold many
/// rows and banks while a session runs above them. After the peers are
/// released, idle shrink frees their empty slabs. The retained session still
/// serves the same next turn and every charged byte is attributed.
#[test]
#[ignore = "requires a CUDA or Metal device and MAGNITUDE_TEST_MTP_GGUF"]
fn idle_shrink_preserves_a_retained_session() {
    let limits = ServiceLimits {
        prefill_tokens: 256,
        decode_tokens: 16,
        decode_share: 0.5,
        locality_seconds: 1.0,
    };
    let (mut host, vocabulary, tokenizer) = plain_owner(limits, 4);
    let session = prompt(&tokenizer, 0, 96);
    // The peers decode until their streams (64 publications, fewer than
    // their 96 tokens) fill, then wait on credit, holding the low rows and
    // banks; every bank the session and its next turn acquire lies above
    // theirs.
    let peers = (1..8)
        .map(|seed| {
            admit(
                &mut host,
                greedy(&prompt(&tokenizer, seed, 96), vocabulary, 96),
                false,
                64,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    host.step_until(&mut [], Duration::from_secs(60), |host, _| {
        peers.iter().all(|peer| output_blocked(host, peer))
    });
    let mut first = admit(&mut host, greedy(&session, vocabulary, 24), true, 24).unwrap();
    host.drive(&mut [&mut first], Duration::from_secs(60));
    // Completion retained the session's terminal prefix while its peers
    // stay open.
    let output = first.output;
    let resumed = session.len() + output.len();
    let next_turn = session
        .iter()
        .chain(&output)
        .copied()
        .chain([TokenId(4242)])
        .collect::<Vec<_>>();
    // The peers stay open but are never selected. Both turns below run
    // alone, in the same launch shapes, so their numerics are comparable bit
    // for bit. Stopped turns retain nothing of their own: the session's
    // prompt and terminal prefixes stay the only retained state.
    let (reference, cached) = stopped_turn(&mut host, vocabulary, &next_turn, 16);
    assert_eq!(cached, resumed, "the next turn resumes from the session");
    assert_eq!(host.owner().cached_prefixes(), 2);
    assert!(
        peers.iter().all(|peer| output_blocked(&host, peer)),
        "the peers stayed resident: {:?}",
        peers
            .iter()
            .map(|peer| host.snapshot(peer.id))
            .collect::<Vec<_>>()
    );
    let before = host.owner().reconcile_memory_charge().unwrap();
    for peer in peers {
        host.release(peer, Duration::from_secs(20));
    }
    host.settle();
    let after = host.owner().reconcile_memory_charge().unwrap();
    eprintln!(
        "idle shrink: retained={} charged {} -> {} state {:?} -> {:?} unattributed {} -> {}",
        host.owner().cached_prefixes(),
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
    let (replayed, cached) = stopped_turn(&mut host, vocabulary, &next_turn, 16);
    eprintln!("idle shrink replay: reference={reference:?} replayed={replayed:?}");
    assert_eq!(
        cached, resumed,
        "the retained session still serves the turn"
    );
    assert_eq!(replayed, reference);
    assert_eq!(host.owner().reconcile_memory_charge().unwrap().unattributed, 0);
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
    let (mut host, vocabulary, tokenizer) = plain_owner(limits, 8);
    let session = prompt(&tokenizer, 0, 96);
    let (first, _) = retained_turn(&mut host, vocabulary, &session, 16);
    let resumed = session.len() + first.len();
    let next_turn = session
        .iter()
        .chain(&first)
        .copied()
        .chain([TokenId(4242)])
        .collect::<Vec<_>>();
    // Without the prefix cache nothing is looked up or retained.
    let mut cold_request = admit(&mut host, greedy(&next_turn, vocabulary, 16), false, 16).unwrap();
    host.drive(&mut [&mut cold_request], Duration::from_secs(60));
    assert_eq!(cold_request.usage().cached_tokens, 0);
    let cold = cold_request.output;
    let (served, cached) = retained_turn(&mut host, vocabulary, &next_turn, 16);
    assert_eq!(cached, resumed);
    host.settle();
    let retained = host.owner().cached_prefixes();
    let before = host.owner().reconcile_memory_charge().unwrap();
    assert!(retained >= 2 && before.target_state.retained > 0);
    assert_eq!(before.unattributed, 0);

    let catalog = DeviceCatalog::discover().unwrap();
    let reserves = MemoryReserves::standard();
    let band = |host: &Host| {
        host.owner()
            .inspect_domain(|domain| {
                platform::observe_domains(&catalog, domain.resources().device(), &reserves)
                    .map_err(|error| error.to_string())
            })
            .unwrap()[0]
    };
    let planning = band(&host).thresholds.planning_bytes;
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
    assert_eq!(band(&host).band, MemoryBand::Reclaim);
    // The next memory observation enters Reclaim and, with the bindings
    // between flights, runs its releases.
    let unloading = host.observe().unload;
    let released = host.owner().reconcile_memory_charge().unwrap();
    drop(guard);
    host.observe();
    assert_eq!(band(&host).band, MemoryBand::Normal);
    eprintln!(
        "reclaim retained release: entries {retained} -> {} charged {} -> {} state {:?} -> {:?} \
         unattributed {} -> {} unloading={unloading}",
        host.owner().cached_prefixes(),
        before.charged,
        released.charged,
        before.target_state,
        released.target_state,
        before.unattributed,
        released.unattributed,
    );
    assert!(!unloading, "headroom returned within the unload interval");
    assert_eq!(host.owner().cached_prefixes(), 0);
    assert_eq!(released.target_state.retained, 0);
    assert!(released.charged < before.charged);
    assert_eq!(released.unattributed, 0);

    let (replayed, cached) = retained_turn(&mut host, vocabulary, &next_turn, 16);
    eprintln!(
        "reclaim replay: cold={cold:?} served={served:?} replayed={replayed:?} \
         served_matches_cold={}",
        served == cold
    );
    assert_eq!(cached, 0, "nothing retained survived the Reclaim release");
    assert_eq!(replayed, cold);
    assert_eq!(host.owner().reconcile_memory_charge().unwrap().unattributed, 0);
}
