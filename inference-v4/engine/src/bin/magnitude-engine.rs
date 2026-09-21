//! Engine CLI: serve one model artifact directly.
//!
//! `magnitude-engine --model <GGUF file or MLX/safetensors directory> [options]`
//! opens the artifact, composes the execution owner on the selected backend
//! (compiling every entry eagerly), and serves HTTP chat completions until
//! interrupted. Startup reports artifact-open and composition timings.

use magnitude_engine::{
    generation::constraints::CacheLimits,
    models::qwen35::loading::Model,
    telemetry::{Telemetry, DEFAULT_TRACES_ENDPOINT},
    service::policy::Limits,
    serving::{startup::ExecutionLimits, Config},
};
use seismic_runtime::{plan::Settings, Device};
use std::{path::PathBuf, time::Duration};

fn value_of(flag: &str, args: &mut impl Iterator<Item = String>) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn backend_device(name: &str) -> Result<Device, String> {
    let device = match name {
        "metal" => Device::metal(),
        "cuda" => Device::cuda(),
        "cpu" => Device::cpu(),
        other => return Err(format!("unknown backend: {other}")),
    };
    device.map_err(|e| format!("opening {name} device: {e}"))
}

fn default_backend() -> &'static str {
    if cfg!(target_os = "macos") {
        "metal"
    } else {
        "cuda"
    }
}

struct Options {
    model: PathBuf,
    host: String,
    port: u16,
    backend: String,
    context_tokens: usize,
    memory_gib: u64,
    max_active: usize,
    output_capacity: usize,
    served_model: String,
    telemetry_endpoint: String,
}

fn parse_options() -> Result<Options, String> {
    let mut model = None;
    let mut host = "127.0.0.1".into();
    let mut port: u16 = 8080;
    let mut backend = default_backend().to_string();
    let mut context_tokens = 0usize;
    let mut memory_gib = 0u64;
    let mut max_active = 1usize;
    let mut output_capacity = 256usize;
    let mut served_model = None;
    let mut telemetry_endpoint = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value_of(&flag, &mut args)?)),
            "--host" => host = value_of(&flag, &mut args)?,
            "--port" => port = value_of(&flag, &mut args)?.parse().map_err(|e| format!("{e}"))?,
            "--backend" => backend = value_of(&flag, &mut args)?,
            "--context-tokens" => {
                context_tokens =
                    value_of(&flag, &mut args)?.parse().map_err(|e| format!("{e}"))?
            }
            "--memory-gib" => {
                memory_gib = value_of(&flag, &mut args)?.parse().map_err(|e| format!("{e}"))?
            }
            "--max-active" => {
                max_active = value_of(&flag, &mut args)?.parse().map_err(|e| format!("{e}"))?
            }
            "--output-capacity" => {
                output_capacity = value_of(&flag, &mut args)?.parse().map_err(|e| format!("{e}"))?
            }
            "--served-model" => served_model = Some(value_of(&flag, &mut args)?),
            "--telemetry" => telemetry_endpoint = Some(value_of(&flag, &mut args)?),
            "--help" | "-h" => {
                println!(
                    "magnitude-engine --model <artifact> [--host ADDR] [--port N] \
                     [--backend metal|cuda|cpu] [--context-tokens N] [--memory-gib N] \
                     [--max-active N] [--output-capacity N] [--served-model NAME]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag: {other} (try --help)")),
        }
    }
    let model = model.ok_or("--model is required (GGUF file or MLX/safetensors directory)")?;
    let served_model = served_model.unwrap_or_else(|| {
        model
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "qwen".into())
    });
    Ok(Options {
        model,
        host,
        port,
        backend,
        context_tokens,
        memory_gib,
        max_active,
        output_capacity,
        served_model,
        telemetry_endpoint: telemetry_endpoint
            .unwrap_or_else(|| DEFAULT_TRACES_ENDPOINT.to_string()),
    })
}

fn main() {
    if let Err(error) = run() {
        eprintln!("magnitude-engine: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let options = parse_options()?;
    let telemetry = Telemetry::open(&options.telemetry_endpoint);
    eprintln!(
        "magnitude-engine: telemetry -> {}",
        options.telemetry_endpoint
    );
    let opened = std::time::Instant::now();
    let model = Model::open(&options.model)?;
    let open_seconds = opened.elapsed().as_secs_f64();
    let geometry = &model.description().geometry;
    let context_limit =
        usize::try_from(geometry.context_limit).map_err(|_| "context limit exceeds host domain")?;
    let context_tokens = if options.context_tokens == 0 {
        context_limit
    } else {
        options.context_tokens
    };
    let vocabulary =
        usize::try_from(geometry.vocabulary).map_err(|_| "vocabulary exceeds host domain")?;
    let config = Config {
        model: options.served_model.clone(),
        context_tokens,
        vocabulary,
        output_capacity: options.output_capacity,
        forced_quantum: 0,
        template_variant: None,
        template_override: None,
        max_body_bytes: 16 << 20,
        max_response_bytes: 8 << 20,
        max_connections: 64,
        request_timeout: Duration::from_secs(900),
    };
    let memory_bytes = if options.memory_gib == 0 {
        28 << 30
    } else {
        usize::try_from(options.memory_gib << 30)
            .map_err(|_| "memory budget exceeds host domain")?
    };
    let limits = ExecutionLimits {
        storage_bytes: memory_bytes,
        control_capacity: 1 << 30,
        scheduler: Limits {
            max_requests: 64,
            max_batch: options.max_active.max(1),
            prefill_tokens: 8192,
            decode_tokens: 256,
            decode_share: 0.5,
            locality_seconds: 1.0,
        },
        grammar_cache: CacheLimits { entries: 16, bytes: 64 << 20 },
    };
    let backend = options.backend.clone();
    let execution =
        move || -> Result<(Device, Settings), String> { Ok((backend_device(&backend)?, Settings::default())) };
    let composed = std::time::Instant::now();
    let server = magnitude_engine::serving::startup::qwen(
        &options.model,
        config,
        limits,
        execution,
    )?;
    let compile_seconds = composed.elapsed().as_secs_f64();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("{e}"))?;
    runtime.block_on(async move {
        let address = format!("{}:{}", options.host, options.port);
        let listener = tokio::net::TcpListener::bind(&address)
            .await
            .map_err(|e| format!("binding {address}: {e}"))?;
        eprintln!(
            "magnitude-engine: model={} backend={} context_tokens={} vocabulary={} max_active={}",
            options.served_model, options.backend, context_tokens, vocabulary, options.max_active
        );
        eprintln!(
            "magnitude-engine: artifact open {open_seconds:.3}s; composition+compile {compile_seconds:.3}s"
        );
        eprintln!("magnitude-engine: serving http://{address}/v1/chat/completions");
        let shutdown = std::future::pending::<()>();
        server.serve(listener, shutdown).await
    })
}
