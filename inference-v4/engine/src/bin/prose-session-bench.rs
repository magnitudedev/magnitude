//! Reproducible single-session Qwen prose benchmark used by W12.

use magnitude_engine::models::qwen35::session::{Report, Session};
use seismic::{BackendName, DeviceCatalog, PrecisionPolicy};
use serde::Serialize;
use std::{fs, path::PathBuf, rc::Rc, time::Instant};

const USAGE: &str = "usage:
  prose-session-bench --model <GGUF|MLX path> --fixture <Gutenberg text> \
    --backend <metal|cuda|cpu> --context <tokens> [--prefill-chunk <tokens>] \
    [--decode <tokens>] [--precision <exact|unconstrained>] [--output <json>]";

struct Options {
    model: PathBuf,
    fixture: PathBuf,
    backend: BackendName,
    context: usize,
    prefill_chunk: usize,
    decode: usize,
    precision: PrecisionPolicy,
    output: Option<PathBuf>,
}

#[derive(Serialize)]
struct BenchmarkReport {
    catalog_discovery_seconds: f64,
    device_open_and_profile_seconds: f64,
    benchmark: Report,
}

fn value(flag: &str, args: &mut impl Iterator<Item = String>) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn positive(flag: &str, value: String) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|error| format!("{flag}: {error}"))
        .and_then(|value| {
            if value == 0 {
                Err(format!("{flag} must be positive"))
            } else {
                Ok(value)
            }
        })
}

fn options() -> Result<Options, String> {
    let mut model = None;
    let mut fixture = None;
    let mut backend = None;
    let mut context = None;
    let mut prefill_chunk = 512;
    let mut decode = 128;
    let mut precision = PrecisionPolicy::Exact;
    let mut output = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value(&flag, &mut args)?)),
            "--fixture" => fixture = Some(PathBuf::from(value(&flag, &mut args)?)),
            "--backend" => {
                let name = value(&flag, &mut args)?;
                backend = Some(
                    BackendName::parse(&name).ok_or_else(|| format!("unknown backend `{name}`"))?,
                );
            }
            "--context" => context = Some(positive(&flag, value(&flag, &mut args)?)?),
            "--prefill-chunk" => prefill_chunk = positive(&flag, value(&flag, &mut args)?)?,
            "--decode" => decode = positive(&flag, value(&flag, &mut args)?)?,
            "--precision" => {
                precision = match value(&flag, &mut args)?.as_str() {
                    "exact" => PrecisionPolicy::Exact,
                    "unconstrained" => PrecisionPolicy::Unconstrained,
                    other => return Err(format!("unknown precision policy `{other}`")),
                }
            }
            "--output" => output = Some(PathBuf::from(value(&flag, &mut args)?)),
            "--help" | "-h" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}`\n{USAGE}")),
        }
    }
    Ok(Options {
        model: model.ok_or_else(|| format!("--model is required\n{USAGE}"))?,
        fixture: fixture.ok_or_else(|| format!("--fixture is required\n{USAGE}"))?,
        backend: backend.ok_or_else(|| format!("--backend is required\n{USAGE}"))?,
        context: context.ok_or_else(|| format!("--context is required\n{USAGE}"))?,
        prefill_chunk,
        decode,
        precision,
        output,
    })
}

fn run() -> Result<(), String> {
    let options = options()?;
    let discovery = Instant::now();
    let catalog =
        DeviceCatalog::discover().map_err(|error| format!("device discovery: {error}"))?;
    let catalog_discovery_seconds = discovery.elapsed().as_secs_f64();
    let opening = Instant::now();
    let device = catalog
        .open_backend(options.backend)
        .map_err(|error| format!("device open/profile: {error}"))?;
    let device_open_and_profile_seconds = opening.elapsed().as_secs_f64();
    eprintln!(
        "target={} backend={} memory={} discovery={:.3}s open+profile={:.3}s",
        device.info().name,
        device.backend().as_str(),
        device.info().memory_bytes,
        catalog_discovery_seconds,
        device_open_and_profile_seconds
    );
    let fixture = fs::read(&options.fixture)
        .map_err(|error| format!("{}: {error}", options.fixture.display()))?;
    let mut session = Session::load(
        &options.model,
        Rc::new(device),
        options.precision,
        options.context,
    )
    .map_err(|error| error.to_string())?;
    let benchmark = session
        .measure(&fixture, options.prefill_chunk, options.decode)
        .map_err(|error| error.to_string())?;
    let report = BenchmarkReport {
        catalog_discovery_seconds,
        device_open_and_profile_seconds,
        benchmark,
    };
    let json = serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?;
    if let Some(path) = options.output {
        fs::write(&path, format!("{json}\n"))
            .map_err(|error| format!("{}: {error}", path.display()))?;
    } else {
        println!("{json}");
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("prose-session-bench: {error}");
        std::process::exit(1);
    }
}
