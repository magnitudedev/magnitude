//! Real-device optional-component residency check. Run only on a host with an
//! MTP GGUF: `MAGNITUDE_TEST_MTP_GGUF=/path/model.gguf cargo test --release
//! -p magnitude-engine --test optional_memory -- --ignored --nocapture`.

use magnitude_engine::{
    build_native_domain,
    composition::EngineConfiguration,
    options::{ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection},
};
use magnitude_model_executor::{
    platform::DeviceRequest, ExecutionPath, FeatureRows, Operation, PhysicalDecision, RequestId,
    TokenId,
};
use magnitude_model_state::KvCodec;
use magnitude_service::{
    domain::{self as service_domain, DomainFlight},
    ServiceLimits,
};
use std::path::PathBuf;

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
        },
        model: ModelPolicy {
            method: ModelMethod::Mtp,
            mtp_proposals: Some(4),
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
    }
    .resolve()
    .unwrap();
    let row_bytes = usize::try_from(resolved.manifest.definition.geometry.hidden).unwrap() * 2;
    let package = resolved.artifacts.shared_package();
    let (mut domain, _) = build_native_domain(&resolved.manifest, package).unwrap();
    let charge = |domain: &magnitude_model_executor::ExecutorDomain| {
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

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires Sparky's unified CUDA device and MAGNITUDE_TEST_MTP_GGUF"]
fn admission_reclaims_under_a_real_process_limit() {
    use magnitude_generation::{Generation, InputLayout, MethodChoice, Options, Sampling, Shaping};
    use magnitude_model_executor::platform::{self, MemoryConstraint};
    use magnitude_service::owner::{Owner, Status};
    use seismic::DeviceCatalog;
    use std::collections::BTreeSet;

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
        max_requests: 8,
        max_batch: 2,
        prefill_tokens: 64,
        decode_tokens: 16,
        decode_share: 0.5,
        locality_seconds: 1.0,
    };
    let resolved = EngineConfiguration {
        package: PackageOptions {
            target: model,
            projector: ProjectorSelection::Disabled,
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
        control_capacity: 64,
        kernel_cache: std::env::var_os("MAGNITUDE_TEST_KERNEL_CACHE").map(PathBuf::from),
    }
    .resolve()
    .unwrap();
    let vocabulary = resolved.manifest.definition.geometry.vocabulary as usize;
    let package = resolved.artifacts.shared_package();
    let (domain, _) = build_native_domain(&resolved.manifest, package).unwrap();
    let catalog = DeviceCatalog::discover().unwrap();
    let available = platform::growth_availability(&catalog, domain.resources().device()).unwrap();
    assert_eq!(available.constraint, MemoryConstraint::HostRam);
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
                sampling: Sampling::Greedy,
                shaping: Shaping {
                    temperature: 0.0,
                    ..Shaping::default()
                },
                seed: 0,
                forced_quantum: 0,
                method: MethodChoice::Plain,
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
    let limit = vm_bytes.checked_add(headroom).unwrap();
    assert!(limit < previous.rlim_cur && limit <= previous.rlim_max);
    let restricted = libc::rlimit {
        rlim_cur: limit,
        rlim_max: previous.rlim_max,
    };
    assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_AS, &restricted) }, 0);
    let guard = AddressSpaceLimit(previous);
    let observed = owner
        .inspect_domain(|domain| {
            platform::growth_availability(&catalog, domain.resources().device())
                .map_err(|error| error.to_string())
        })
        .unwrap();
    eprintln!(
        "admission deficit setup: seed={seed_bytes} vm={vm_bytes} limit={limit} available={} charged={} state={:?}",
        observed.bytes, before.charged, before.target_state
    );
    assert!(observed.bytes <= headroom);
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
    for tick in 128..192 {
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
    let (target_shape, head_shape) = owner
        .inspect_domain(|domain| domain.pressure_state_shape())
        .unwrap();
    eprintln!(
        "admission in-limit: headroom={headroom} prompt_tokens={prompt_tokens} incoming={in_limit_status:?} output={in_limit_output} saw_preemption={saw_preemption} charged={in_limit_charge} lowest_charge={lowest_charge} target_shape={target_shape:?} head_shape={head_shape:?}",
    );
    drop(guard);
    for tick in 192..256 {
        if owner.output_len(incoming).unwrap() > 0 {
            break;
        }
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
