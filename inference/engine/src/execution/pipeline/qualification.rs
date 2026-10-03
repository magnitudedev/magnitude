//! Manual ordinary worker qualification, including its Owner; not public API evidence.
use super::*;
use crate::composition::EngineConfiguration;
use crate::options::{ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection};
use magnitude_generation::{EndOfGeneration, FinishReason, MethodChoice, Options, Sampling, Shaping};
use magnitude_scheduler::prefix_cache::PrefixRetention;
use serde::Deserialize;
use std::collections::BTreeSet;
#[derive(Deserialize)]
struct Fixture {
    vocabulary: usize,
    stop_tokens: Vec<i32>,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    tokens: Vec<i32>,
}
fn configuration(devices: [DeviceSelector; 2]) -> crate::composition::ResolvedEngineConfiguration {
    EngineConfiguration {
        package: PackageOptions {
            target: std::env::var_os("MAGNITUDE_PIPELINE_MODEL")
                .expect("complete model")
                .into(),
            projector: ProjectorSelection::Disabled,
            draft: None,
        },
        model: ModelPolicy {
            method: ModelMethod::Plain,
            mtp_proposals: None,
            kv_codec: KvCodec::Dense,
            lookahead: false,
            exported_logits_rows: 0,
        },
        context_tokens: Some(256),
        service: ServiceLimits {
            prefill_tokens: 2,
            decode_tokens: 1,
            decode_share: 0.5,
            locality_seconds: 1.0,
        },
        path: ExecutionPath::Native,
        device: platform::DeviceRequest::Selector(devices[0]),
        kernel_cache: std::env::var_os("MAGNITUDE_PIPELINE_CACHE").map(Into::into),
        reserves: platform::MemoryReserves::standard(),
    }
    .resolve()
    .unwrap()
}

fn request(fixture: &Fixture, case: &Case) -> crate::worker::RequestOptions {
    let tokens = case
        .tokens
        .iter()
        .map(|&t| TokenId(t as u32))
        .collect::<Vec<_>>();
    let layout = InputLayout::new(tokens.len(), vec![]).unwrap();
    crate::worker::RequestOptions {
        input: PreparedModelInput::from_text_coordinates(
            TokenPlan::new(tokens, layout).unwrap(),
            (0..case.tokens.len()).map(|p| [p as i32; 3]).collect(),
        )
        .unwrap(),
        options: Options {
            max_tokens: 8,
            output_capacity: 64,
            context_limit: 256,
            vocabulary: fixture.vocabulary,
            stop_tokens: fixture
                .stop_tokens
                .iter()
                .map(|&t| TokenId(t as u32))
                .collect(),
            suppressed_tokens: BTreeSet::new(),
            sampling: Sampling::Greedy,
            shaping: Shaping {
                temperature: 0.0,
                ..Default::default()
            },
            seed: 42,
            forced_quantum: 1,
            method: MethodChoice::Plain,
            end_of_generation: EndOfGeneration::Stop,
            reasoning_budget: None,
        },
        constraint: None,
        retention: PrefixRetention::Transient,
        output_capacity: 64,
    }
}
async fn worker_tokens(
    client: &crate::worker::EngineClient,
    request: crate::worker::RequestOptions,
) -> Vec<u32> {
    let mut stream = client.admit(request).await.unwrap();
    let mut tokens = Vec::new();
    loop {
        match tokio::time::timeout(std::time::Duration::from_secs(60), stream.receive())
            .await
            .unwrap()
        {
            Some(crate::worker::RequestEvent::Output(output)) => {
                tokens.extend(output.into_iter().map(|t| t.token.0))
            }
            Some(crate::worker::RequestEvent::Completed { finish, .. }) => {
                assert_eq!(finish, FinishReason::Length);
                assert_eq!(tokens.len(), 8);
                return tokens;
            }
            Some(crate::worker::RequestEvent::Failed(error)) => {
                panic!("worker request failed: {error}")
            }
            None => panic!("worker stream closed without terminal outcome"),
        }
    }
}
async fn shutdown(client: &crate::worker::EngineClient) {
    client.shutdown().unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        while client.check().is_ok() {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
}
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires complete model, token fixture and two CUDA devices"]
async fn paired_worker_generates_and_observes_both_owned_domains() {
    let fixture: Fixture = serde_json::from_reader(
        std::fs::File::open(std::env::var_os("MAGNITUDE_PIPELINE_TOKEN_FIXTURE").unwrap()).unwrap(),
    )
    .unwrap();
    let ordinals = std::env::var("MAGNITUDE_PIPELINE_CUDA_ORDINALS")
        .unwrap()
        .split(',')
        .map(|x| x.parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(ordinals.len(), 2);
    let catalog = DeviceCatalog::discover().unwrap();
    let devices = catalog
        .topology()
        .devices()
        .iter()
        .filter(|d| d.backend == BackendName::Cuda)
        .map(|d| d.selector)
        .collect::<Vec<_>>();
    let devices = [devices[ordinals[0]], devices[ordinals[1]]];
    let control = crate::composition::start_in_process(configuration(devices), |_| {}).unwrap();
    assert!(control.ready_info().pipeline.is_none());
    let mut expected = Vec::new();
    for case in &fixture.cases {
        expected.push(worker_tokens(control.client(), request(&fixture, case)).await);
    }
    shutdown(control.client()).await;
    drop(control);
    let mut resolved = configuration(devices);
    resolved.manifest.pipeline = Some(ExplicitPipeline {
        devices,
        split: std::env::var("MAGNITUDE_PIPELINE_CUT")
            .unwrap()
            .parse()
            .unwrap(),
    });
    let placement = resolved.manifest.pipeline;
    let paired = crate::composition::start_in_process(resolved, |_| {}).unwrap();
    let loaded_census = paired.ready_info().census.clone();
    // The worker builds the same ordinary Owner and preserves the former
    // direct-composition qualifier's placement and local readiness checks.
    let ready = paired.ready_info().pipeline.as_ref().unwrap();
    assert_eq!(Some(ready.placement), placement);
    for plan in &ready.stages {
        assert!(plan.immutable_bytes > 0);
        assert!(plan.planned_bytes <= plan.domain_capacity_bytes);
    }
    assert!(paired
        .ready_info()
        .census
        .domains
        .iter()
        .all(|domain| matches!(domain.domain, MemoryDomain::DeviceLocal { .. })));
    assert_eq!(paired.ready_info().census.domains.len(), 2);
    let recovery_control = expected[0].clone();
    for (case, expected) in fixture.cases.iter().zip(expected) {
        let tokens = worker_tokens(paired.client(), request(&fixture, case)).await;
        assert_eq!(tokens, expected);
        eprintln!("paired_worker generated={tokens:?}");
    }
    let observed = paired.client().observe().await.unwrap();
    assert_eq!(observed.census.domains.len(), 2);
    for device in devices {
        let domain = MemoryDomain::DeviceLocal { device };
        assert!(observed.census.domains.iter().any(|d| d.domain == domain));
        assert!(observed.domains.iter().any(|d| d.domain == domain));
    }
    let mut retained = request(&fixture, &fixture.cases[0]);
    retained.retention = PrefixRetention::Retain {
        cache_points: Vec::new(),
    };
    assert!(paired.client().admit(retained).await.is_err());
    // Refusal must leave the ordinary worker serving and allow fresh recovery.
    assert_eq!(
        worker_tokens(paired.client(), request(&fixture, &fixture.cases[0])).await,
        recovery_control
    );
    let mut long = request(&fixture, &fixture.cases[0]);
    long.options.max_tokens = 200;
    // Lifecycle controls must stay responsive without forced output backpressure.
    long.output_capacity = 256;
    let mut active = paired.client().admit(long).await.unwrap();
    assert!(matches!(
        active.receive().await,
        Some(crate::worker::RequestEvent::Output(_))
    ));
    assert!(paired
        .client()
        .admit(request(&fixture, &fixture.cases[1]))
        .await
        .is_err());
    let terminal = tokio::time::timeout(std::time::Duration::from_secs(10), active.stop())
        .await
        .unwrap();
    assert!(matches!(
        terminal,
        Some(crate::worker::RequestEvent::Completed {
            finish: FinishReason::Cancelled,
            ..
        })
    ));
    assert_eq!(
        worker_tokens(paired.client(), request(&fixture, &fixture.cases[0])).await,
        recovery_control
    );
    shutdown(paired.client()).await;
    drop(paired);
    // Rebuild in the same process after the drained worker has released both
    // allocation authorities. A new process exiting cannot mask a leaked owner.
    let mut resolved = configuration(devices);
    resolved.manifest.pipeline = placement;
    let reloaded = crate::composition::start_in_process(resolved, |_| {}).unwrap();
    assert_eq!(reloaded.ready_info().census, loaded_census);
    assert_eq!(
        worker_tokens(reloaded.client(), request(&fixture, &fixture.cases[0])).await,
        recovery_control
    );
    shutdown(reloaded.client()).await;
    eprintln!("PASS normal ExecutionOwner/Worker/Session paired generation, two-domain observation, retention/concurrency refusal without output backpressure, ordered stop/recovery, same-process unload/reload census and generation, shutdown; HTTP_API=false");
}
