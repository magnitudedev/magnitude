//! Metadata-only model assessment on this machine's selected device.
//!
//! Loads (or measures once and stores) the device's measurement basis, then
//! assesses every model from its headers alone and prints one JSON object per
//! model, shaped like the service's per-profile assessment result, with the
//! model's capabilities and template fingerprint alongside.

use magnitude_engine::{
    assessment::{
        assess_model, AssessmentEnvironment, ModelAssessment, ModelPackagePaths, UnsupportedModel,
    },
    options::{standard_service_limits, ModelPolicy},
};
use magnitude_model_executor::{
    assessment::{
        load_basis, measure_basis, store_basis, BasisIdentity, DomainFit, ExecutionAssessment,
        IncompatibleReason, MeasurementBasis, PerformanceConfidence,
    },
    platform::{self, DeviceRequest, MemoryReserves, PlatformConfig},
    ExecutionPath, KernelCache, DEFAULT_KERNEL_CACHE_BYTES,
};
use seismic::DeviceCatalog;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const ENGINE_BUILD: &str = concat!(env!("CARGO_PKG_NAME"), "@", env!("CARGO_PKG_VERSION"));
const STANDARD_DEPTHS: [u32; 3] = [25_000, 50_000, 75_000];

struct Options {
    device: DeviceRequest,
    cache_dir: PathBuf,
    depths: Vec<u32>,
    models: Vec<ModelPackagePaths>,
}

const USAGE: &str = "magnitude-assess --device auto|metal|cuda|vulkan|cpu --cache-dir DIR \
     [--depths 25000,50000,75000] TARGET.gguf[,PROJECTOR.gguf]...";

fn value(flag: &str, args: &mut impl Iterator<Item = String>) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse() -> Result<Options, String> {
    let mut device = None;
    let mut cache_dir = None;
    let mut depths = STANDARD_DEPTHS.to_vec();
    let mut models = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--device" => {
                device = Some(
                    value(&argument, &mut args)?
                        .parse()
                        .map_err(|error| format!("{error}"))?,
                )
            }
            "--cache-dir" => cache_dir = Some(PathBuf::from(value(&argument, &mut args)?)),
            "--depths" => {
                depths = value(&argument, &mut args)?
                    .split(',')
                    .map(|depth| {
                        depth
                            .parse::<u32>()
                            .map_err(|error| format!("--depths {depth}: {error}"))
                    })
                    .collect::<Result<_, _>>()?
            }
            "--help" | "-h" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            flag if flag.starts_with("--") => {
                return Err(format!("unknown flag: {flag} (try --help)"))
            }
            model => {
                let mut components = model.splitn(2, ',');
                models.push(ModelPackagePaths {
                    target: PathBuf::from(components.next().expect("split yields one part")),
                    projector: components.next().map(PathBuf::from),
                });
            }
        }
    }
    if models.is_empty() {
        return Err(format!("no model given\n{USAGE}"));
    }
    Ok(Options {
        device: device.ok_or("--device is required")?,
        cache_dir: cache_dir.ok_or("--cache-dir is required")?,
        depths,
        models,
    })
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(error) => {
            eprintln!("magnitude-assess: {error}");
            std::process::exit(1);
        }
    }
}

/// The basis cached for this device and build, or a fresh measurement that is
/// stored for the next run.
fn basis(
    catalog: &DeviceCatalog,
    device: DeviceRequest,
    reserves: MemoryReserves,
    cache_dir: &Path,
) -> Result<MeasurementBasis, String> {
    let selected = platform::select_device(catalog, ExecutionPath::Native, device, &reserves)
        .map_err(|error| error.to_string())?;
    let kernels = KernelCache::open(cache_dir.join("kernels"), DEFAULT_KERNEL_CACHE_BYTES)
        .map_err(|error| error.to_string())?;
    let opened = platform::open_selected(
        catalog,
        selected.info.selector,
        PlatformConfig {
            path: ExecutionPath::Native,
            artifacts: Some(Arc::new(kernels) as Arc<dyn seismic::ArtifactStore>),
            reserves,
        },
    )
    .map_err(|error| error.to_string())?;
    let identity = BasisIdentity::for_device(opened.device(), ENGINE_BUILD);
    let directory = cache_dir.join("assessment-basis");
    if let Some(basis) = load_basis(&directory, &identity) {
        eprintln!("magnitude-assess: using the stored basis for {}", identity.device);
        return Ok(basis);
    }
    let started = Instant::now();
    let basis = measure_basis(catalog, opened.device(), reserves, identity)
        .map_err(|error| error.to_string())?;
    eprintln!(
        "magnitude-assess: measured the basis ({} classes) in {:.2} s",
        basis.classes.len(),
        started.elapsed().as_secs_f64()
    );
    store_basis(&directory, &basis).map_err(|error| error.to_string())?;
    Ok(basis)
}

/// Returns whether every model produced a result.
fn run() -> Result<bool, String> {
    let options = parse()?;
    let catalog = DeviceCatalog::discover().map_err(|error| error.to_string())?;
    let reserves = MemoryReserves::standard();
    let basis = basis(&catalog, options.device, reserves, &options.cache_dir)?;
    // The standalone engine's defaults: one conversation per batch.
    let service = standard_service_limits(1);
    let environment = AssessmentEnvironment::discover(
        &catalog,
        options.device,
        reserves,
        basis,
        ModelPolicy::default(),
        service,
    )
    .map_err(|error| error.to_string())?;
    let mut complete = true;
    for model in &options.models {
        let started = Instant::now();
        match assess_model(model, &environment, &options.depths) {
            Ok(assessment) => {
                let mut object = render(&environment, &assessment);
                object["model"] = json!(model.target.display().to_string());
                object["assessmentSeconds"] = json!(started.elapsed().as_secs_f64());
                println!("{object}");
            }
            Err(error) => {
                complete = false;
                eprintln!("magnitude-assess: {}: {error}", model.target.display());
            }
        }
    }
    Ok(complete)
}

fn render(environment: &AssessmentEnvironment, assessment: &ModelAssessment) -> Value {
    let (facts, execution) = match assessment {
        ModelAssessment::Unsupported(unsupported) => {
            let code = match unsupported {
                UnsupportedModel::Family(_) => "unsupported_family",
                UnsupportedModel::Tokenizer(_) => "unsupported_tokenizer",
                UnsupportedModel::Template(_) => "unsupported_template",
            };
            return incompatible(code, unsupported.to_string());
        }
        ModelAssessment::Assessed { facts, execution } => (facts, execution),
    };
    let memory = |domains: &[DomainFit]| {
        domains
            .iter()
            .map(|domain| {
                json!({
                    "memoryDomainId": domain_id(environment, domain.domain),
                    "capacityBytes": domain.capacity_bytes,
                    "requiredBytes": domain.required_bytes,
                    "compatibilityReserveBytes": domain.reserve_bytes,
                    "remainingBytes": domain.remaining_bytes,
                })
            })
            .collect::<Vec<_>>()
    };
    let mut object = match execution {
        ExecutionAssessment::Fits {
            domains,
            performance,
            ..
        } => json!({
            "_tag": "Fits",
            "memory": memory(domains),
            "performance": performance.iter().map(|estimate| json!({
                "contextTokens": estimate.context_tokens,
                "lowerTokensPerSecond": estimate.lower_tokens_per_second,
                "estimatedTokensPerSecond": estimate.estimated_tokens_per_second,
                "upperTokensPerSecond": estimate.upper_tokens_per_second,
                "confidence": match estimate.confidence {
                    PerformanceConfidence::High => "high",
                    PerformanceConfidence::Moderate => "moderate",
                    PerformanceConfidence::Low => "low",
                },
            })).collect::<Vec<_>>(),
        }),
        ExecutionAssessment::DoesNotFit {
            domains,
            limiting,
            deficit_bytes,
            ..
        } => json!({
            "_tag": "DoesNotFit",
            "memory": memory(domains),
            "limitingResource": domain_id(environment, *limiting),
            "deficitBytes": deficit_bytes,
        }),
        ExecutionAssessment::Incompatible {
            reason: IncompatibleReason::OutsideBasis { classes },
        } => incompatible(
            "unsupported_operation",
            format!(
                "the device's measurement basis does not cover {}",
                classes
                    .iter()
                    .map(|(key, reason)| {
                        let bindings = key
                            .bindings
                            .iter()
                            .map(|element| element.name())
                            .collect::<Vec<_>>()
                            .join(",");
                        let geometry = key
                            .geometry
                            .iter()
                            .map(|(name, value)| format!("{name}={value}"))
                            .collect::<Vec<_>>()
                            .join(",");
                        match reason {
                            Some(reason) => {
                                format!("{:?}[{bindings}]({geometry}): {reason}", key.class)
                            }
                            None => format!("{:?}[{bindings}]({geometry})", key.class),
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
        ),
        ExecutionAssessment::Incompatible {
            reason: IncompatibleReason::Unsupported { reason },
        } => incompatible("unsupported_representation", reason.clone()),
    };
    let reasoning = &facts.capabilities.reasoning;
    object["capabilities"] = json!({
        "vision": facts.capabilities.vision,
        "tools": facts.capabilities.tools,
        "structuredOutput": facts.capabilities.structured_output,
        "reasoning": {
            "supported": reasoning.supported(),
            "efforts": reasoning.efforts,
            "defaultEffort": reasoning.default_effort,
        },
    });
    object["templateFingerprint"] = json!(facts.template_fingerprint);
    object["contextLimit"] = json!(facts.context_limit);
    object
}

fn incompatible(code: &str, message: String) -> Value {
    json!({
        "_tag": "Incompatible",
        "failure": { "code": code, "message": message, "retryable": false },
    })
}

/// Host RAM is the `system` domain; a dedicated device's memory is named by
/// the device's selector.
fn domain_id(environment: &AssessmentEnvironment, domain: seismic::MemoryPoolId) -> String {
    if domain == environment.topology.host_pool().id {
        "system".to_owned()
    } else {
        environment.selected.info.selector.to_string()
    }
}
