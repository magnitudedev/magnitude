//! Standalone HTTP composition of the Inference V4 engine.

use magnitude_engine::{
    chat::CacheLimits,
    composition::{EngineConfiguration, MediaSourcePolicy},
    options::{ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection, StoragePolicy},
    service::ServiceLimits,
    serving::Config as ServerConfig,
    telemetry::{Telemetry, DEFAULT_TRACES_ENDPOINT},
};
use magnitude_model_executor::ExecutionPath;
use std::{path::PathBuf, time::Duration};

struct Options {
    target: PathBuf,
    projector: ProjectorSelection,
    host: String,
    port: u16,
    served_model: String,
    context_tokens: Option<usize>,
    storage_bytes: u64,
    max_batch: usize,
    output_capacity: usize,
    method: ModelMethod,
    mtp_proposals: Option<u8>,
    telemetry_endpoint: String,
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
    let mut storage_gib = 28_u64;
    let mut max_batch = 1_usize;
    let mut output_capacity = 256_usize;
    let mut method = ModelMethod::Auto;
    let mut mtp_proposals = None;
    let mut telemetry_endpoint = DEFAULT_TRACES_ENDPOINT.to_owned();
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
            "--storage-gib" => {
                storage_gib = value(&flag, &mut args)?
                    .parse()
                    .map_err(|e| format!("{e}"))?
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
            "--telemetry" => telemetry_endpoint = value(&flag, &mut args)?,
            "--help" | "-h" => {
                println!(
                    "magnitude-engine --model TARGET.gguf [--projector PROJECTOR.gguf | --no-projector] \
                     [--host ADDR] [--port N] [--served-model NAME] [--context-tokens N] [--storage-gib N] \
                     [--max-batch N] [--output-capacity N] [--method auto|plain|mtp] \
                     [--mtp-proposals N] [--telemetry URL]"
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
    let storage_bytes = storage_gib
        .checked_mul(1_u64 << 30)
        .ok_or("storage budget exceeds u64")?;
    if storage_bytes == 0 || context_tokens == Some(0) || max_batch == 0 || output_capacity == 0 {
        return Err("storage, context, max batch, and output capacity must be positive".into());
    }
    Ok(Options {
        target,
        projector,
        host,
        port,
        served_model,
        context_tokens,
        storage_bytes,
        max_batch,
        output_capacity,
        method,
        mtp_proposals,
        telemetry_endpoint,
    })
}

fn main() {
    if let Err(error) = run() {
        eprintln!("magnitude-engine: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let options = parse()?;
    let _telemetry = Telemetry::open(&options.telemetry_endpoint);
    let service = ServiceLimits {
        max_requests: 128,
        max_batch: options.max_batch,
        prefill_tokens: 512,
        decode_tokens: 32,
        decode_share: 0.5,
        locality_seconds: 1.0,
    };
    let safety_reserve_bytes = (options.storage_bytes / 10).min(1_u64 << 30);
    let resolved = EngineConfiguration {
        package: PackageOptions {
            target: options.target,
            projector: options.projector,
        },
        model: ModelPolicy {
            method: options.method,
            mtp_proposals: options.mtp_proposals,
            ..ModelPolicy::default()
        },
        context_tokens: options.context_tokens,
        service,
        storage: StoragePolicy {
            storage_bytes: options.storage_bytes,
            retention_bytes: None,
            safety_reserve_bytes,
        },
        path: ExecutionPath::NativeMetal,
        control_capacity: 256,
    }
    .resolve()?;
    let context_tokens = usize::try_from(resolved.artifacts.definition().geometry.context_limit)
        .map_err(|_| "model context limit exceeds host domain")?;
    let vocabulary = usize::try_from(resolved.artifacts.definition().geometry.vocabulary)
        .map_err(|_| "model vocabulary exceeds host domain")?;
    let method = resolved.manifest.model.method.policy();
    let server = resolved.start()?.into_server(
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
            "magnitude-engine: model={} path=native-metal context={} vocabulary={} max_batch={}",
            options.served_model, context_tokens, vocabulary, options.max_batch
        );
        eprintln!("magnitude-engine: serving http://{address}/v1/chat/completions");
        server.serve(listener, std::future::pending::<()>()).await
    }))
}
