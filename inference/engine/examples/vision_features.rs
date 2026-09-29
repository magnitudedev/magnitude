//! Vision features: the image features one projector computes for one image,
//! with a SHA-256 of their bits and, optionally, the raw F32 rows.
//!
//! Two runs on the same device print identical lines when the vision tower is
//! bit-identical (the vision entries declare no arithmetic tuning parameter).
//! The raw rows (`--out`, little-endian F32 `[rows, width]`) are what the
//! projector references in `inference/validation/` compare against; the
//! prepared patch rows (`--pixels`, F32 `[patches, patch width]` in the
//! processor's order) let a reference run its tower on the same input.
//!
//! ```text
//! vision_features --model M.gguf --projector P.gguf --image I.jpg
//!     [--backend metal|cuda|vulkan|cpu] [--cache-dir DIR] [--out features.f32]
//!     [--pixels pixels.f32]
//! ```

use magnitude_chat::request::ImageInput;
use magnitude_engine::{
    build_native_domain,
    composition::EngineConfiguration,
    inputs::SpecialTokens,
    options::{ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection},
};
use magnitude_executor::{
    platform::{DeviceRequest, MemoryReserves},
    ExecutionPath, Outcome, RequestId,
};
use magnitude_scheduler::{
    domain::{self as service_domain, DomainFlight},
    ServiceLimits,
};
use magnitude_state::KvCodec;
use seismic::BackendName;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::Instant;

struct Options {
    model: PathBuf,
    projector: PathBuf,
    image: PathBuf,
    backend: Option<BackendName>,
    cache: Option<PathBuf>,
    out: Option<PathBuf>,
    pixels: Option<PathBuf>,
}

fn options() -> Result<Options, String> {
    let mut args = std::env::args().skip(1);
    let mut options = Options {
        model: PathBuf::new(),
        projector: PathBuf::new(),
        image: PathBuf::new(),
        backend: None,
        cache: None,
        out: None,
        pixels: None,
    };
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or(format!("{flag} requires a value"));
        match flag.as_str() {
            "--model" => options.model = PathBuf::from(value()?),
            "--projector" => options.projector = PathBuf::from(value()?),
            "--image" => options.image = PathBuf::from(value()?),
            "--backend" => {
                options.backend = Some(match value()?.as_str() {
                    "metal" => BackendName::Metal,
                    "cuda" => BackendName::Cuda,
                    "vulkan" => BackendName::Vulkan,
                    "cpu" => BackendName::Cpu,
                    other => return Err(format!("unknown backend {other}")),
                })
            }
            "--cache-dir" => options.cache = Some(PathBuf::from(value()?)),
            "--out" => options.out = Some(PathBuf::from(value()?)),
            "--pixels" => options.pixels = Some(PathBuf::from(value()?)),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if [&options.model, &options.projector, &options.image]
        .iter()
        .any(|path| path.as_os_str().is_empty())
    {
        return Err("--model, --projector and --image are required".into());
    }
    Ok(options)
}

fn main() -> Result<(), String> {
    let options = options()?;
    let resolved = EngineConfiguration {
        package: PackageOptions {
            target: options.model.clone(),
            projector: ProjectorSelection::Explicit(options.projector.clone()),
            draft: None,
        },
        model: ModelPolicy {
            method: ModelMethod::Plain,
            mtp_proposals: None,
            kv_codec: KvCodec::AffineK8V4,
            lookahead: false,
        },
        context_tokens: Some(8192),
        // The service's default launch bound: images up to 512 merged rows.
        service: ServiceLimits {
            prefill_tokens: 512,
            decode_tokens: 16,
            decode_share: 0.5,
            locality_seconds: 1.0,
        },
        path: ExecutionPath::Native,
        device: options
            .backend
            .map_or(DeviceRequest::Automatic, DeviceRequest::Backend),
        kernel_cache: options.cache.clone(),
        reserves: MemoryReserves::standard(),
    }
    .resolve()
    .map_err(|error| error.to_string())?;

    let host = &resolved.host;
    let placeholder = host
        .media_placeholder()
        .ok_or("the model has no vision component")?;
    let tokens = host
        .tokenizer()
        .encode(&format!("Describe:{placeholder}"), SpecialTokens::Recognize)
        .map_err(|error| error.to_string())?;
    let image = ImageInput {
        media_type: "image/jpeg".into(),
        bytes: std::fs::read(&options.image)
            .map_err(|error| format!("{}: {error}", options.image.display()))?
            .into(),
    };
    let input = host
        .prepare_input(tokens, &[image])
        .map_err(|error| error.to_string())?;
    let [prepared] = input.vision() else {
        return Err("the prompt must prepare exactly one image".into());
    };
    let grid = prepared.grid();
    if let Some(path) = &options.pixels {
        std::fs::write(path, prepared.pixels().data())
            .map_err(|error| format!("{}: {error}", path.display()))?;
    }

    let started = Instant::now();
    let (mut domain, _) = build_native_domain(&resolved.manifest, host.shared_package())
        .map_err(|error| error.to_string())?;
    println!("backend={:?} load={:.1}s", domain.execution_backend(), started.elapsed().as_secs_f64());

    let request = RequestId(1);
    domain.install_input(request, input)?;
    let operations = domain
        .open_state(request, None)
        .map_err(|error| error.to_string())?;
    let groups = service_domain::group(&domain, operations);
    let [group] = groups.as_slice() else {
        return Err("one image forms one encode group".into());
    };
    let started = Instant::now();
    let DomainFlight::Vision(flight) =
        service_domain::submit_group(&mut domain, group).map_err(|error| error.to_string())?
    else {
        return Err("an encode runs on the vision lane".into());
    };
    let pending = domain
        .finish_vision(flight)
        .map_err(|error| error.to_string())?;
    let encode = started.elapsed().as_secs_f64();
    let Outcome::Encode { features } = pending.outcome() else {
        return Err("an encode returns features".into());
    };
    let values = features
        .read_to_host()
        .map_err(|error| error.to_string())?;
    let rows = features.rows();
    let mut hasher = Sha256::new();
    for value in &values {
        hasher.update(value.to_bits().to_le_bytes());
    }
    let digest = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let rms = (values.iter().map(|value| f64::from(*value).powi(2)).sum::<f64>()
        / values.len() as f64)
        .sqrt();
    println!(
        "grid={grid:?} rows={rows} width={} rms={rms:.6} encode={encode:.3}s sha256={digest}",
        values.len() / rows
    );
    if let Some(out) = options.out {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>();
        std::fs::write(&out, bytes).map_err(|error| format!("{}: {error}", out.display()))?;
    }
    Ok(())
}
