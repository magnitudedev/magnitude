//! Standalone HTTP composition of the Inference V4 engine.

use magnitude_engine::{
    chat::CacheLimits,
    composition::{EngineConfiguration, MediaSourcePolicy},
    options::{
        standard_service_limits, ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection,
    },
    serving::Config as ServerConfig,
    telemetry::{Telemetry, DEFAULT_TRACES_ENDPOINT},
};
use magnitude_model_executor::{
    platform::{DeviceRequest, MemoryReserves},
    ExecutionPath,
};
use magnitude_model_state::KvCodec;
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

struct Options {
    target: PathBuf,
    projector: ProjectorSelection,
    host: String,
    port: u16,
    served_model: String,
    context_tokens: Option<usize>,
    max_batch: usize,
    output_capacity: usize,
    method: ModelMethod,
    mtp_proposals: Option<u8>,
    kv_codec: KvCodec,
    lookahead: bool,
    telemetry_endpoint: String,
    device: DeviceRequest,
    kernel_cache: Option<PathBuf>,
}

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

fn parse() -> Result<Options, String> {
    let mut target = None;
    let mut projector = ProjectorSelection::Discover;
    let mut host = "127.0.0.1".to_owned();
    let mut port = 8080;
    let mut served_model = None;
    let mut context_tokens = None;
    let mut max_batch = 1_usize;
    let mut output_capacity = 256_usize;
    let mut method = ModelMethod::Auto;
    let mut mtp_proposals = None;
    let mut kv_codec = ModelPolicy::default().kv_codec;
    let mut lookahead = ModelPolicy::default().lookahead;
    let mut telemetry_endpoint = DEFAULT_TRACES_ENDPOINT.to_owned();
    let mut device = DeviceRequest::Automatic;
    let mut kernel_cache = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => target = Some(PathBuf::from(value(&flag, &mut args)?)),
            "--projector" => {
                projector = ProjectorSelection::Explicit(PathBuf::from(value(&flag, &mut args)?))
            }
            "--no-projector" => projector = ProjectorSelection::Disabled,
            "--host" => host = value(&flag, &mut args)?,
            "--port" => {
                port = value(&flag, &mut args)?
                    .parse()
                    .map_err(|e| format!("{e}"))?
            }
            "--served-model" => served_model = Some(value(&flag, &mut args)?),
            "--context-tokens" => {
                context_tokens = Some(
                    value(&flag, &mut args)?
                        .parse()
                        .map_err(|e| format!("{e}"))?,
                )
            }
            "--max-batch" => {
                max_batch = value(&flag, &mut args)?
                    .parse()
                    .map_err(|e| format!("{e}"))?
            }
            "--output-capacity" => {
                output_capacity = value(&flag, &mut args)?
                    .parse()
                    .map_err(|e| format!("{e}"))?
            }
            "--method" => {
                method = match value(&flag, &mut args)?.as_str() {
                    "auto" => ModelMethod::Auto,
                    "plain" => ModelMethod::Plain,
                    "mtp" => ModelMethod::Mtp,
                    value => return Err(format!("unknown generation method: {value}")),
                }
            }
            "--mtp-proposals" => {
                mtp_proposals = Some(
                    value(&flag, &mut args)?
                        .parse()
                        .map_err(|e| format!("{e}"))?,
                )
            }
            "--kv-codec" => kv_codec = value(&flag, &mut args)?.parse()?,
            "--lookahead" => lookahead = switch(&flag, &value(&flag, &mut args)?)?,
            "--telemetry" => telemetry_endpoint = value(&flag, &mut args)?,
            "--device" => {
                device = value(&flag, &mut args)?
                    .parse()
                    .map_err(|error| format!("{error}"))?
            }
            "--cache-dir" => kernel_cache = Some(PathBuf::from(value(&flag, &mut args)?)),
            "--help" | "-h" => {
                println!(
                    "magnitude-engine --model TARGET.gguf [--projector PROJECTOR.gguf | --no-projector] \
                     [--host ADDR] [--port N] [--served-model NAME] [--context-tokens N] \
                     [--max-batch N] [--output-capacity N] [--method auto|plain|mtp] \
                     [--mtp-proposals N] [--kv-codec dense|affine-k8v4] [--lookahead on|off] \
                     [--telemetry URL] \
                     [--device auto|metal|cuda|vulkan|cpu|SELECTOR] [--cache-dir DIR]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag: {other} (try --help)")),
        }
    }
    let target = target.ok_or("--model is required")?;
    let served_model = served_model.unwrap_or_else(|| {
        target
            .file_stem()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "qwen".to_owned())
    });
    if context_tokens == Some(0) || max_batch == 0 || output_capacity == 0 {
        return Err("context, max batch, and output capacity must be positive".into());
    }
    Ok(Options {
        target,
        projector,
        host,
        port,
        served_model,
        context_tokens,
        max_batch,
        output_capacity,
        method,
        mtp_proposals,
        kv_codec,
        lookahead,
        telemetry_endpoint,
        device,
        kernel_cache,
    })
}

fn main() {
    if let Err(error) = run() {
        eprintln!("magnitude-engine: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let load_started = Instant::now();
    let options = parse()?;
    let _telemetry = Telemetry::open(&options.telemetry_endpoint);
    let service = standard_service_limits(options.max_batch);
    let resolved = EngineConfiguration {
        package: PackageOptions {
            target: options.target,
            projector: options.projector,
        },
        model: ModelPolicy {
            method: options.method,
            mtp_proposals: options.mtp_proposals,
            kv_codec: options.kv_codec,
            lookahead: options.lookahead,
        },
        context_tokens: options.context_tokens,
        service,
        path: ExecutionPath::Native,
        device: options.device,
        control_capacity: 256,
        kernel_cache: options.kernel_cache,
        reserves: MemoryReserves::standard(),
    }
    .resolve()?;
    eprintln!(
        "magnitude-engine: host admission in {:.2} s",
        load_started.elapsed().as_secs_f64()
    );
    let context_tokens = usize::try_from(resolved.artifacts.definition().geometry.context_limit)
        .map_err(|_| "model context limit exceeds host domain")?;
    let vocabulary = usize::try_from(resolved.artifacts.definition().geometry.vocabulary)
        .map_err(|_| "model vocabulary exceeds host domain")?;
    let method = resolved.manifest.model.method.policy();
    let worker_started = Instant::now();
    let ready = resolved.start()?;
    eprintln!(
        "magnitude-engine: worker readiness in {:.2} s",
        worker_started.elapsed().as_secs_f64()
    );
    let backend = ready.ready_info().backend;
    let host_started = Instant::now();
    let server = ready.into_server(
        MediaSourcePolicy::data_urls_only(),
        CacheLimits {
            entries: 16,
            bytes: 64 << 20,
        },
        ServerConfig {
            model: options.served_model.clone(),
            context_tokens,
            vocabulary,
            output_capacity: options.output_capacity,
            forced_quantum: 0,
            method,
            template_variant: None,
            template_override: None,
            max_body_bytes: 16 << 20,
            max_response_bytes: 8 << 20,
            max_connections: 64,
            request_timeout: Duration::from_secs(900),
        },
    )?;
    eprintln!(
        "magnitude-engine: host serving setup in {:.2} s",
        host_started.elapsed().as_secs_f64()
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| error.to_string())?;
    let local = tokio::task::LocalSet::new();
    runtime.block_on(local.run_until(async move {
        let address = format!("{}:{}", options.host, options.port);
        let listener = tokio::net::TcpListener::bind(&address)
            .await
            .map_err(|error| format!("binding {address}: {error}"))?;
        eprintln!(
            "magnitude-engine: model={} path=native backend={} context={} vocabulary={} max_batch={}",
            options.served_model,
            backend.as_str(),
            context_tokens,
            vocabulary,
            options.max_batch
        );
        eprintln!("magnitude-engine: serving http://{address}/v1/chat/completions");
        eprintln!("magnitude-engine: load to serving in {:.2} s", load_started.elapsed().as_secs_f64());
        server.serve(listener, std::future::pending::<()>()).await
    }))
}
