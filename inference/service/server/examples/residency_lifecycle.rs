//! Model residency on real engine workers (gate G-D): load plan, host-only properties without a
//! load, ensure (readiness verified against the host), the hardware snapshot merged with the
//! resident worker's readings, a completion through the mounted protocol router, join, replacement, stop,
//! stop during load, worker crash, memory-pressure kill with the five-second admission gate, and
//! idle release. Development evidence only.
//!
//! ```text
//! residency_lifecycle --bundle model-planner-inputs.bundle --cache-root DIR --model-store DIR \
//!     --hf-cache DIR --first REPO_SUBSTRING --second REPO_SUBSTRING
//! ```
//!
//! The executable is also its own `inference-worker`, exactly as the service binary is.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt as _;
use magnitude_executor::platform::MemoryReserves;
use magnitude_service_api::{AppState, app};
use magnitude_service_contracts::HardwareProvider;
use magnitude_service_contracts::models::{
    CatalogModelState, CatalogModels as _, DiscoveredModelState, DiscoveredModels as _,
    EffectiveModel, ModelInstance, ModelInstanceFailure, ModelInstanceId, ModelInstanceLifecycle,
    ModelReleaseReason,
};
use magnitude_service_models::{
    InventoryConfig, ManagedModelDownloads, ManagedModelStore, ModelDomainResolver,
    load_release_catalog, managed_model_services,
};
use magnitude_service_server::assessment::measurement::DeviceExclusion;
use magnitude_service_server::configurations::ResolvedConfigurations;
use magnitude_service_server::hardware::HardwareInventory;
use magnitude_service_server::residency::controller::{ModelInstances, ResidencyEnvironment};
use magnitude_service_server::residency::supervisor::{HostMemoryObserver, SeismicHostMemory};
use magnitude_service_server::residency::worker::run_inference_worker;
use magnitude_service_server::serving::ServiceModels;
use magnitude_service_server::worker_process::{WorkerLauncher, install_parent_watchdog};
use tower::ServiceExt as _;

const POLL: Duration = Duration::from_millis(20);
const NO_OVERRIDE: u64 = u64::MAX;

/// Seismic's host headroom, or a scripted value while one is set.
struct ScriptedHostMemory {
    seismic: SeismicHostMemory,
    headroom: AtomicU64,
}

impl HostMemoryObserver for ScriptedHostMemory {
    fn headroom_bytes(&self) -> Result<u64, String> {
        match self.headroom.load(Ordering::Acquire) {
            NO_OVERRIDE => self.seismic.headroom_bytes(),
            bytes => Ok(bytes),
        }
    }
}

fn flag(arguments: &[String], name: &str) -> String {
    arguments
        .iter()
        .position(|argument| argument == name)
        .and_then(|index| arguments.get(index + 1))
        .cloned()
        .unwrap_or_else(|| panic!("{name} is required"))
}

fn main() -> anyhow::Result<()> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if arguments.first().map(String::as_str) == Some("inference-worker") {
        install_parent_watchdog()?;
        std::process::exit(run_inference_worker());
    }
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(lifecycle(arguments))
}

struct Harness {
    instances: ModelInstances,
    router: axum::Router,
}

async fn lifecycle(arguments: Vec<String>) -> anyhow::Result<()> {
    let release = Arc::new(load_release_catalog(&PathBuf::from(flag(
        &arguments, "--bundle",
    )))?);
    let mut config = InventoryConfig::with_roots(
        PathBuf::from(flag(&arguments, "--model-store")),
        PathBuf::from(flag(&arguments, "--cache-root")),
    )?;
    config
        .hf_cache_dirs
        .push(PathBuf::from(flag(&arguments, "--hf-cache")));
    config.catalog_models = release.catalog().models.clone();
    let inventory = Arc::new(ManagedModelStore::open(config).await?);
    let resolver = ModelDomainResolver::new(inventory.clone(), release.catalog().clone());
    let downloads = Arc::new(ManagedModelDownloads::open(inventory.clone()).await?);
    let catalog = Arc::new(seismic::DeviceCatalog::discover()?);
    let reserves = MemoryReserves::standard();
    let host_memory = Arc::new(ScriptedHostMemory {
        seismic: SeismicHostMemory::new(catalog.clone()),
        headroom: AtomicU64::new(NO_OVERRIDE),
    });
    let configurations = Arc::new(ResolvedConfigurations::new(
        inventory.clone(),
        inventory.derived_cache().kernel_directory(),
        reserves,
    ));
    let device_exclusion = DeviceExclusion::default();
    let harness = |idle_timeout: Duration, namespace: &str| {
        let instances = ModelInstances::start(ResidencyEnvironment {
            models: inventory.clone(),
            model_variants: resolver.clone(),
            configurations: configurations.clone(),
            catalog: catalog.clone(),
            reserves,
            host_memory: host_memory.clone(),
            idle_timeout,
            launcher: WorkerLauncher::current().expect("harness executable"),
            device_exclusion: device_exclusion.clone(),
            instance_id_namespace: namespace.to_owned(),
        });
        let router = app(
            AppState::new().with_model_controller(Arc::new(instances.clone())),
            magnitude_serving::Serving::new(Arc::new(ServiceModels::new(instances.clone()))),
        );
        Harness { instances, router }
    };
    let main = harness(Duration::from_secs(3600), "main");
    let services = managed_model_services(
        resolver.clone(),
        downloads,
        Arc::new(main.instances.clone()),
        Arc::new(main.instances.clone()),
    )?;
    // Local packages are catalog models when the catalog names them, discovered models otherwise;
    // either becomes ready once its validation completes.
    let started = Instant::now();
    services.discovered.refresh_discovery().await?;
    let ready_models = loop {
        let discovered = services.discovered.list_discovered().await?;
        let catalog = services.catalog.list_catalog().await?;
        let ready = discovered
            .models
            .iter()
            .filter(|model| matches!(model.state, DiscoveredModelState::Ready { .. }))
            .map(|model| model.id.to_string())
            .chain(catalog.models.iter().filter_map(|model| {
                matches!(
                    &model.local_state,
                    CatalogModelState::Installed {
                        effective: EffectiveModel::Ready { .. },
                        ..
                    }
                )
                .then(|| model.id.to_string())
            }))
            .collect::<Vec<_>>();
        if discovered.reconciliation_complete && ready.len() >= 2 {
            break ready;
        }
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "local models did not settle: ready {ready:?}; discovered {:?}; catalog {:?}",
            discovered
                .models
                .iter()
                .map(|model| (model.id.to_string(), format!("{:?}", model.state)))
                .collect::<Vec<_>>(),
            catalog
                .models
                .iter()
                .filter(|model| !matches!(model.local_state, CatalogModelState::NotInstalled))
                .map(|model| (model.id.to_string(), format!("{:?}", model.local_state)))
                .collect::<Vec<_>>()
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    };
    println!(
        "local models {:>7.3} s  {ready_models:?}",
        started.elapsed().as_secs_f64()
    );
    let pick = |needle: &str| {
        ready_models
            .iter()
            .find(|id| id.contains(needle))
            .cloned()
            .unwrap_or_else(|| panic!("no ready local model matches {needle}"))
    };
    let first = pick(&flag(&arguments, "--first"));
    let second = pick(&flag(&arguments, "--second"));
    println!("first model  {first}");
    println!("second model {second}");

    // Load plan: the engine preview on the service catalog.
    let started = Instant::now();
    let plan = main.instances.preview_load(&first).await?;
    println!(
        "preview      {:>7.3} s  context {} required {} device {} ({:?})",
        started.elapsed().as_secs_f64(),
        plan.context_window_tokens,
        plan.required_memory_bytes,
        plan.device.id.as_str(),
        plan.device.backend
    );

    // Host-only properties: answered from the resolved configuration, never loading.
    let started = Instant::now();
    let (status, body) = post(
        &main.router,
        &format!(
            "/api/v1/models/{}/properties",
            first.replace('%', "%25").replace('/', "%2F")
        ),
        serde_json::json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(
        main.instances.instances().await?.instances.is_empty(),
        "properties loaded a model"
    );
    println!(
        "properties   {:>7.3} s  no instance created",
        started.elapsed().as_secs_f64()
    );

    // Ensure: a worker loads exactly the previewed device.
    let started = Instant::now();
    let ready = main.instances.ensure_resident(&first).await?;
    let ModelInstanceLifecycle::Ready { allocation } = &ready.lifecycle else {
        panic!("ensure returned {:?}", ready.lifecycle)
    };
    println!(
        "ensure       {:>7.3} s  {} context {} domains {:?}",
        started.elapsed().as_secs_f64(),
        ready.id.0,
        allocation.context_window_tokens,
        allocation
            .memory_domains
            .iter()
            .map(|domain| (
                domain.memory_domain_id.as_str().to_owned(),
                domain.model_bytes,
                domain.context_bytes,
                domain.compute_bytes,
                domain.auxiliary_bytes
            ))
            .collect::<Vec<_>>()
    );
    let worker = worker_pid().expect("a ready instance has a worker");

    // Hardware while resident: the service topology merged with the worker's fresh readings.
    let started = Instant::now();
    let resident = main
        .instances
        .resident_memory()
        .await
        .expect("a resident worker answers an observation");
    let hardware =
        HardwareInventory::new(catalog.clone(), reserves, Arc::new(main.instances.clone()));
    let snapshot = HardwareProvider::snapshot(&hardware).await?;
    println!(
        "hardware     {:>7.3} s  worker domains {:?}; snapshot domains {:?}",
        started.elapsed().as_secs_f64(),
        resident
            .domains
            .iter()
            .map(|domain| (domain.domain.to_string(), domain.headroom_bytes))
            .collect::<Vec<_>>(),
        snapshot
            .memory_domains
            .iter()
            .map(|domain| (domain.id.as_str().to_owned(), domain.current_free_bytes))
            .collect::<Vec<_>>()
    );

    // A completion through the mounted protocol router, on the lease.
    let started = Instant::now();
    let (status, body) = post(
        &main.router,
        "/v1/chat/completions",
        serde_json::json!({
            "model": first,
            "messages": [{"role": "user", "content": "Reply with one word: hello."}],
            "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": false},
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    println!(
        "completion   {:>7.3} s  {:?}",
        started.elapsed().as_secs_f64(),
        body["choices"][0]["message"]["content"]
    );

    // Concurrent completions exercise the mounted protocol, instance lease,
    // worker transport, scheduler and elastic state store together.
    for count in [4, 16] {
        let started = Instant::now();
        let mut requests = Vec::with_capacity(count);
        for index in 0..count {
            let router = main.router.clone();
            let model = first.clone();
            requests.push(tokio::spawn(async move {
                post(
                    &router,
                    "/v1/chat/completions",
                    serde_json::json!({
                        "model": model,
                        "messages": [{"role": "user", "content": format!("Reply with one word for item {index}: hello.")}],
                        "max_tokens": 8,
                        "chat_template_kwargs": {"enable_thinking": false},
                    }),
                )
                .await
            }));
        }
        for request in requests {
            let (status, body) = request.await?;
            assert_eq!(status, StatusCode::OK, "{body}");
            assert!(
                body["choices"][0]["message"]["content"].is_string(),
                "{body}"
            );
        }
        println!(
            "completions {:>7.3} s  {count} concurrent requests completed",
            started.elapsed().as_secs_f64()
        );
    }

    // Join: the same configuration is the same instance.
    let started = Instant::now();
    let joined = main.instances.ensure_resident(&first).await?;
    assert_eq!(joined.id, ready.id);
    println!(
        "join         {:>7.3} s  same instance",
        started.elapsed().as_secs_f64()
    );

    // Replacement: another model releases the first.
    let started = Instant::now();
    let replacement = main.instances.ensure_resident(&second).await?;
    assert_eq!(
        state_of(&main.instances, &ready.id).await,
        ModelInstanceLifecycle::Stopped {
            reason: ModelReleaseReason::Replacement
        }
    );
    assert!(!alive(worker), "the replaced worker still runs");
    println!(
        "replace      {:>7.3} s  {} replaced by {}",
        started.elapsed().as_secs_f64(),
        ready.id.0,
        replacement.id.0
    );

    // Stop with a request in flight: the request ends as `model_instance_stopped` and the worker
    // is retired before the stop returns.
    let worker = worker_pid().expect("a ready instance has a worker");
    let active = tokio::spawn({
        let router = main.router.clone();
        let second = second.clone();
        async move {
            post(
                &router,
                "/v1/chat/completions",
                serde_json::json!({
                    "model": second,
                    "messages": [{"role": "user", "content": "Count from one to five hundred."}],
                    "max_tokens": 4000,
                    "chat_template_kwargs": {"enable_thinking": false},
                }),
            )
            .await
        }
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        !active.is_finished(),
        "the completion finished before stop could interrupt it"
    );
    let started = Instant::now();
    main.instances.stop_instance(replacement.id.clone()).await?;
    let stopped_at = started.elapsed();
    let (status, body) = active.await?;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["code"], "model_instance_stopped", "{body}");
    assert!(!alive(worker), "the stopped worker still runs");
    assert_eq!(
        state_of(&main.instances, &replacement.id).await,
        ModelInstanceLifecycle::Stopped {
            reason: ModelReleaseReason::UserStop
        }
    );
    println!(
        "stop         {:>7.3} s  active request ended model_instance_stopped, worker {worker} retired",
        stopped_at.as_secs_f64()
    );

    // Stop during load: the loading worker is killed and the waiter is told.
    let loading = tokio::spawn({
        let instances = main.instances.clone();
        let first = first.clone();
        async move { instances.ensure_resident(&first).await }
    });
    let loading_id = wait_for(&main.instances, |instance| {
        matches!(instance.lifecycle, ModelInstanceLifecycle::Loading { .. })
    })
    .await;
    while worker_pid().is_none() {
        tokio::time::sleep(POLL).await;
    }
    let worker = worker_pid().expect("loading worker");
    let started = Instant::now();
    main.instances.stop_instance(loading_id.clone()).await?;
    let waiter = loading.await?;
    assert!(
        matches!(&waiter, Err(magnitude_service_contracts::InventoryError::ModelOperation { code, .. }) if code == "model_instance_stopped"),
        "{waiter:?}"
    );
    assert!(!alive(worker), "the loading worker still runs");
    println!(
        "stop-loading {:>7.3} s  worker {worker} retired, waiter stopped",
        started.elapsed().as_secs_f64()
    );

    // Crash: a killed worker fails its instance.
    let crashed = main.instances.ensure_resident(&first).await?;
    let worker = worker_pid().expect("ready worker");
    let started = Instant::now();
    unsafe { libc::kill(worker as libc::pid_t, libc::SIGKILL) };
    wait_for(&main.instances, |instance| {
        instance.id == crashed.id
            && matches!(instance.lifecycle, ModelInstanceLifecycle::Failed { .. })
    })
    .await;
    let ModelInstanceLifecycle::Failed {
        failure: ModelInstanceFailure::Operation { code, .. },
    } = state_of(&main.instances, &crashed.id).await
    else {
        panic!("crash did not fail the instance with an operation failure")
    };
    println!(
        "crash        {:>7.3} s  instance failed: {code}",
        started.elapsed().as_secs_f64()
    );

    // Memory pressure: the first sample at the emergency reserve kills the worker; the next load
    // waits until headroom has stayed above the planning reserve for five seconds.
    let pressured = main.instances.ensure_resident(&first).await?;
    let worker = worker_pid().expect("ready worker");
    let started = Instant::now();
    host_memory.headroom.store(0, Ordering::Release);
    wait_for(&main.instances, |instance| {
        instance.id == pressured.id
            && instance.lifecycle
                == ModelInstanceLifecycle::Stopped {
                    reason: ModelReleaseReason::MemoryPressure,
                }
    })
    .await;
    assert!(!alive(worker), "the pressured worker still runs");
    println!(
        "pressure     {:>7.3} s  killed, released as memory_pressure",
        started.elapsed().as_secs_f64()
    );
    let reload = tokio::spawn({
        let instances = main.instances.clone();
        let first = first.clone();
        async move { instances.ensure_resident(&first).await }
    });
    tokio::time::sleep(Duration::from_millis(500)).await;
    let released = Instant::now();
    host_memory.headroom.store(NO_OVERRIDE, Ordering::Release);
    let reloaded = reload.await??;
    let gated = released.elapsed();
    assert!(
        gated >= Duration::from_secs(5),
        "admission reopened after {gated:?}"
    );
    println!(
        "admission    {:>7.3} s  after headroom recovered (load included)",
        gated.as_secs_f64()
    );
    main.instances.stop_instance(reloaded.id).await?;

    // Idle: an unleased ready instance is released after its idle timeout.
    let idle = harness(Duration::from_secs(2), "idle");
    let resident = idle.instances.ensure_resident(&first).await?;
    let worker = worker_pid().expect("ready worker");
    let started = Instant::now();
    wait_for(&idle.instances, |instance| {
        instance.id == resident.id
            && instance.lifecycle
                == ModelInstanceLifecycle::Stopped {
                    reason: ModelReleaseReason::IdleTimeout,
                }
    })
    .await;
    assert!(!alive(worker), "the idle worker still runs");
    println!(
        "idle         {:>7.3} s  released after the 2 s idle timeout",
        started.elapsed().as_secs_f64()
    );
    println!("G-D residency lifecycle: all scenarios passed");
    Ok(())
}

async fn post(
    router: &axum::Router,
    path: &str,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let response = router
        .clone()
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("router is infallible");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("response body")
        .to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
    )
}

async fn state_of(instances: &ModelInstances, id: &ModelInstanceId) -> ModelInstanceLifecycle {
    instances
        .instances()
        .await
        .expect("snapshot")
        .instances
        .into_iter()
        .find(|instance| &instance.id == id)
        .unwrap_or_else(|| panic!("instance {} is not in the snapshot", id.0))
        .lifecycle
}

/// Wait until an instance satisfies `condition`, for at most one minute.
async fn wait_for(
    instances: &ModelInstances,
    condition: impl Fn(&ModelInstance) -> bool,
) -> ModelInstanceId {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Some(instance) = instances
            .instances()
            .await
            .expect("snapshot")
            .instances
            .into_iter()
            .find(|instance| condition(instance))
        {
            return instance.id;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for an instance state"
        );
        tokio::time::sleep(POLL).await;
    }
}

/// The harness's live `inference-worker` child, if one runs.
fn worker_pid() -> Option<u32> {
    let output = std::process::Command::new("/usr/bin/pgrep")
        .args([
            "-P",
            &std::process::id().to_string(),
            "-f",
            "inference-worker",
        ])
        .output()
        .expect("pgrep");
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .map(|pid| pid.trim().parse().expect("pid"))
}

fn alive(pid: u32) -> bool {
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}
