//! `magnitude-engine`: serve one explicit local model. The engine loads
//! in-process (the same worker and protocol a worker process runs) and the
//! engine's protocol library serves it, with the standalone routes the
//! benchmark adapters use: `/health`, `/v1/models` and `/v1/count`.
use axum::Json;
use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures_util::future::BoxFuture;
use magnitude_engine::chat::SessionLimits;
use magnitude_engine::composition::{EngineConfiguration, start_in_process};
use magnitude_engine::invocation::{InvocationSource, SingleEngine};
use magnitude_engine::options::{
    ExplicitPipeline, ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection,
    standard_service_limits,
};
use magnitude_engine::telemetry::{DEFAULT_TRACES_ENDPOINT, Telemetry};
use magnitude_engine::worker::EngineClient;
use magnitude_executor::ExecutionPath;
use magnitude_executor::platform::{DeviceRequest, MemoryReserves};
use magnitude_serving::engine::{EngineHost, EngineInvocation};
use magnitude_serving::{
    HostChat, LoadProgress, ModelInvocation, ServedModels, Serving, ServingError,
};
use magnitude_state::KvCodec;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

struct Options {
    target: PathBuf,
    projector: ProjectorSelection,
    draft: Option<PathBuf>,
    host: String,
    port: u16,
    served_model: String,
    context_tokens: Option<usize>,
    output_capacity: usize,
    prefill_tokens: usize,
    method: ModelMethod,
    mtp_proposals: Option<u8>,
    kv_codec: KvCodec,
    lookahead: bool,
    telemetry_endpoint: String,
    device: DeviceRequest,
    kernel_cache: Option<PathBuf>,
    pipeline: Option<ExplicitPipeline>,
}

const USAGE: &str = "magnitude-engine --model TARGET.gguf [--projector PROJECTOR.gguf | --no-projector] \
[--draft DRAFT.gguf] [--host ADDR] [--port N] [--served-model NAME] [--context-tokens N] \
[--output-capacity N] [--prefill-tokens N] [--method auto|plain|mtp|dflash|dspark|dflash2] [--mtp-proposals N] \
[--kv-codec dense|affine-k8v4] [--lookahead on|off] [--telemetry URL] \
[--device auto|metal|cuda|vulkan|cpu|SELECTOR] [--cache-dir DIR] \
[--experimental-pipeline CUDA_SELECTOR,CUDA_SELECTOR,SPLIT]";

/// `on` or `off`.
fn switch(flag: &str, value: &str) -> Result<bool, String> {
    match value {
        "on" => Ok(true),
        "off" => Ok(false),
        other => Err(format!("{flag} takes on or off, not {other}")),
    }
}

fn value(flag: &str, args: &mut impl Iterator<Item = String>) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn number<T: std::str::FromStr>(
    flag: &str,
    args: &mut impl Iterator<Item = String>,
) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    value(flag, args)?
        .parse()
        .map_err(|error| format!("{flag}: {error}"))
}

fn parse() -> Result<Options, String> {
    parse_args(std::env::args().skip(1))
}

/// Only explicit physical identities are accepted; no ordinal or cut search.
fn pipeline(value: &str) -> Result<ExplicitPipeline, String> {
    let fields = value.split(',').collect::<Vec<_>>();
    if fields.len() != 3 {
        return Err("--experimental-pipeline requires CUDA_SELECTOR,CUDA_SELECTOR,SPLIT".into());
    }
    let a = fields[0]
        .parse::<seismic::DeviceSelector>()
        .map_err(|e| e.to_string())?;
    let b = fields[1]
        .parse::<seismic::DeviceSelector>()
        .map_err(|e| e.to_string())?;
    let split = fields[2].parse::<usize>().map_err(|e| e.to_string())?;
    if !matches!(
        (a, b),
        (
            seismic::DeviceSelector::Cuda { .. },
            seismic::DeviceSelector::Cuda { .. }
        )
    ) || a == b
        || split == 0
    {
        return Err(
            "experimental pipeline requires two distinct exact CUDA selectors and a positive split"
                .into(),
        );
    }
    Ok(ExplicitPipeline {
        devices: [a, b],
        split,
    })
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Options, String> {
    let defaults = ModelPolicy::default();
    let mut target = None;
    let mut projector = ProjectorSelection::Discover;
    let mut draft = None;
    let mut host = "127.0.0.1".to_owned();
    let mut port = 8080;
    let mut served_model = None;
    let mut context_tokens = None;
    let mut output_capacity = 256;
    let mut prefill_tokens = standard_service_limits().prefill_tokens;
    let mut method = ModelMethod::Auto;
    let mut mtp_proposals = None;
    let mut kv_codec = defaults.kv_codec;
    let mut lookahead = defaults.lookahead;
    let mut telemetry_endpoint = DEFAULT_TRACES_ENDPOINT.to_owned();
    let mut device = DeviceRequest::Automatic;
    let mut kernel_cache = None;
    let mut selected_pipeline = None;
    let mut device_explicit = false;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => target = Some(PathBuf::from(value(&flag, &mut args)?)),
            "--projector" => {
                projector = ProjectorSelection::Explicit(PathBuf::from(value(&flag, &mut args)?))
            }
            "--no-projector" => projector = ProjectorSelection::Disabled,
            "--draft" => draft = Some(PathBuf::from(value(&flag, &mut args)?)),
            "--host" => host = value(&flag, &mut args)?,
            "--port" => port = number(&flag, &mut args)?,
            "--served-model" => served_model = Some(value(&flag, &mut args)?),
            "--context-tokens" => context_tokens = Some(number(&flag, &mut args)?),
            "--output-capacity" => output_capacity = number(&flag, &mut args)?,
            "--prefill-tokens" => prefill_tokens = number(&flag, &mut args)?,
            "--method" => {
                method = match value(&flag, &mut args)?.as_str() {
                    "auto" => ModelMethod::Auto,
                    "plain" => ModelMethod::Plain,
                    "mtp" => ModelMethod::Mtp,
                    "dflash" => ModelMethod::DFlash,
                    "dspark" => ModelMethod::DSpark,
                    "dflash2" => ModelMethod::DFlash2,
                    other => return Err(format!("unknown generation method: {other}")),
                }
            }
            "--mtp-proposals" => mtp_proposals = Some(number(&flag, &mut args)?),
            "--kv-codec" => kv_codec = value(&flag, &mut args)?.parse()?,
            "--lookahead" => lookahead = switch(&flag, &value(&flag, &mut args)?)?,
            "--telemetry" => telemetry_endpoint = value(&flag, &mut args)?,
            "--experimental-pipeline" => {
                if !cfg!(feature = "experimental-pipeline-cuda") {
                    return Err("this build does not enable experimental-pipeline-cuda".into());
                }
                if selected_pipeline.is_some() {
                    return Err("--experimental-pipeline may only be supplied once".into());
                }
                selected_pipeline = Some(pipeline(&value(&flag, &mut args)?)?);
            }
            "--device" => {
                device_explicit = true;
                device = value(&flag, &mut args)?
                    .parse()
                    .map_err(|error| format!("--device: {error}"))?
            }
            "--cache-dir" => kernel_cache = Some(PathBuf::from(value(&flag, &mut args)?)),
            "--help" | "-h" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag: {other} (try --help)")),
        }
    }
    let target = target.ok_or("--model is required")?;
    let served_model = match served_model {
        Some(name) => name,
        None => target
            .file_stem()
            .map(|name| name.to_string_lossy().into_owned())
            .ok_or("--served-model is required when the model path has no file name")?,
    };
    if context_tokens == Some(0) || output_capacity == 0 || prefill_tokens == 0 {
        return Err("context, output capacity, and prefill tokens must be positive".into());
    }
    if selected_pipeline.is_some() && device_explicit {
        return Err("--device and --experimental-pipeline are mutually exclusive".into());
    }
    Ok(Options {
        target,
        projector,
        draft,
        host,
        port,
        served_model,
        context_tokens,
        output_capacity,
        prefill_tokens,
        method,
        mtp_proposals,
        kv_codec,
        lookahead,
        telemetry_endpoint,
        device,
        kernel_cache,
        pipeline: selected_pipeline,
    })
}

/// The standalone engine's one served model.
struct Standalone {
    engine: SingleEngine,
    limits: SessionLimits,
}

impl ServedModels for Standalone {
    fn invoke(
        &self,
        model: &str,
        _progress: Option<LoadProgress>,
    ) -> BoxFuture<'_, Result<Box<dyn ModelInvocation>, ServingError>> {
        let model = model.to_owned();
        Box::pin(async move {
            let invocation = self.engine.resolve(&model).await?;
            Ok(Box::new(EngineInvocation::new(invocation, self.limits))
                as Box<dyn ModelInvocation>)
        })
    }

    fn host(&self, model: &str) -> BoxFuture<'_, Result<Arc<dyn HostChat>, ServingError>> {
        let model = model.to_owned();
        Box::pin(async move {
            let invocation = self.engine.resolve(&model).await?;
            Ok(Arc::new(EngineHost::new(invocation.host)) as Arc<dyn HostChat>)
        })
    }
}

/// What `/health` and `/v1/models` report.
#[derive(Clone)]
struct Identity {
    model: String,
    context_tokens: u64,
    vocabulary: u64,
    engine: EngineClient,
}

async fn health(State(identity): State<Identity>) -> Response {
    match identity.engine.check() {
        Ok(()) => Json(json!({
            "model": identity.model,
            "context_tokens": identity.context_tokens,
            "vocabulary": identity.vocabulary,
            "ready": true,
        }))
        .into_response(),
        Err(error) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": {"message": error.to_string(), "type": "server_error"}})),
        )
            .into_response(),
    }
}

async fn models(State(identity): State<Identity>) -> Json<serde_json::Value> {
    Json(json!({
        "object": "list",
        "data": [{"id": identity.model, "object": "model", "created": 0, "owned_by": "magnitude"}],
    }))
}

async fn memory(State(identity): State<Identity>) -> Response {
    match identity.engine.observe().await {
        Ok(observation) => Json(json!(observation)).into_response(),
        Err(error) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": {"message": error.to_string(), "type": "server_error"}})),
        )
            .into_response(),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("magnitude-engine: {error}");
        std::process::exit(1);
    }
}

async fn shutdown_signal() {
    #[cfg(unix)]
    match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
        Ok(mut terminate) => {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {},
                _ = terminate.recv() => {},
            }
            return;
        }
        Err(error) => eprintln!("magnitude-engine: termination signal: {error}"),
    }
    let _ = tokio::signal::ctrl_c().await;
}

fn run() -> Result<(), String> {
    let load_started = Instant::now();
    let options = parse()?;
    let _telemetry = Telemetry::open(&options.telemetry_endpoint);
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let mut service = standard_service_limits();
    service.prefill_tokens = options.prefill_tokens;
    // The explicitly selected experimental executor has one active decode row.
    // This does not change the ordinary service's batching policy.
    if options.pipeline.is_some() {
        service.decode_tokens = 1;
    }
    let mut resolved = EngineConfiguration {
        package: PackageOptions {
            target: options.target,
            projector: options.projector,
            draft: options.draft,
        },
        model: ModelPolicy {
            method: options.method,
            mtp_proposals: options.mtp_proposals,
            kv_codec: options.kv_codec,
            lookahead: options.lookahead,
            exported_logits_rows: 0,
        },
        context_tokens: options.context_tokens,
        service,
        path: ExecutionPath::Native,
        device: options.device,
        kernel_cache: options.kernel_cache,
        reserves: MemoryReserves::standard(),
    }
    .resolve()
    .map_err(|error| error.to_string())?;
    resolved.manifest.pipeline = options.pipeline;
    eprintln!(
        "magnitude-engine: host resolution in {:.2} s",
        load_started.elapsed().as_secs_f64()
    );
    let geometry = &resolved.host.definition().decoder;
    let (context_tokens, vocabulary) = (geometry.context_limit, geometry.vocabulary);
    let worker_started = Instant::now();
    let mut reported = None;
    let ready = start_in_process(resolved, |progress| {
        // Each step once; its measured progress repeats per unit and weight.
        let step = std::mem::discriminant(&progress);
        if reported != Some(step) {
            reported = Some(step);
            eprintln!("magnitude-engine: load {progress:?}");
        }
    })
    .map_err(|error| error.to_string())?;
    eprintln!(
        "magnitude-engine: worker readiness in {:.2} s",
        worker_started.elapsed().as_secs_f64()
    );
    let backend = ready.ready_info().backend;
    if let Some(pipeline) = &ready.ready_info().pipeline {
        eprintln!("magnitude-engine: experimental pipeline {:?}", pipeline);
    }
    let (host, engine, _) = ready.into_parts();
    let identity = Identity {
        model: options.served_model.clone(),
        context_tokens,
        vocabulary,
        engine: engine.clone(),
    };
    let shutdown_engine = engine.clone();
    let serving = Serving::new(Arc::new(Standalone {
        engine: SingleEngine::new(options.served_model.clone(), host, engine),
        limits: SessionLimits {
            output_capacity: options.output_capacity,
            max_output_bytes: 8 << 20,
        },
    }));
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/memory", get(memory))
        .with_state(identity)
        .merge(
            Router::new()
                .route(
                    "/v1/count",
                    post(magnitude_serving::chat::count_chat_tokens),
                )
                .with_state(serving.clone()),
        )
        .merge(magnitude_serving::router(serving));
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| error.to_string())?;
    runtime.block_on(async move {
        let address = format!("{}:{}", options.host, options.port);
        let listener = tokio::net::TcpListener::bind(&address)
            .await
            .map_err(|error| format!("binding {address}: {error}"))?;
        eprintln!(
            "magnitude-engine: model={} path=native backend={} context={} vocabulary={}",
            options.served_model,
            backend.as_str(),
            context_tokens,
            vocabulary,
        );
        eprintln!("magnitude-engine: serving http://{address}/v1/chat/completions");
        eprintln!(
            "magnitude-engine: load to serving in {:.2} s",
            load_started.elapsed().as_secs_f64()
        );
        let result = axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|error| error.to_string());
        shutdown_engine
            .shutdown()
            .map_err(|error| error.to_string())?;
        while shutdown_engine.check().is_ok() {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    const PAIR: &str =
        "cuda:01010101-0101-0101-0101-010101010101,cuda:02020202-0202-0202-0202-020202020202,7";
    fn args(flags: &[&str]) -> impl Iterator<Item = String> {
        flags
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .into_iter()
    }
    #[test]
    fn ordinary_cli_does_not_select_pipeline() {
        let options = parse_args(args(&["--model", "model.gguf"])).unwrap();
        assert!(options.pipeline.is_none());
        assert_eq!(options.device, DeviceRequest::Automatic);
    }
    #[test]
    fn explicit_pipeline_geometry_is_validated_without_devices() {
        let selected = pipeline(PAIR).unwrap();
        assert_eq!(selected.split, 7);
        for bad in [
            "cuda,cuda,7",
            "host-cpu,host-cpu,7",
            "cuda:01,cuda:02,7",
            "",
            "cuda:01010101010101010101010101010101,cuda:01010101010101010101010101010101,7",
            "cuda:01010101010101010101010101010101,cuda:02020202020202020202020202020202,0",
        ] {
            assert!(pipeline(bad).is_err(), "{bad}");
        }
    }
    #[test]
    fn opt_in_is_feature_gated_and_never_overrides_device() {
        let selected = parse_args(args(&[
            "--model",
            "model.gguf",
            "--experimental-pipeline",
            PAIR,
        ]));
        assert_eq!(
            selected.is_ok(),
            cfg!(feature = "experimental-pipeline-cuda")
        );
        assert!(parse_args(args(&[
            "--model",
            "model.gguf",
            "--device",
            "cuda",
            "--experimental-pipeline",
            PAIR
        ]))
        .is_err());
        assert!(parse_args(args(&[
            "--model",
            "model.gguf",
            "--experimental-pipeline",
            PAIR,
            "--experimental-pipeline",
            PAIR
        ]))
        .is_err());
    }
}
