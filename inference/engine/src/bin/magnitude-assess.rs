//! Metadata-only model assessment on this machine's selected device.
//!
//! Loads the device's stored measurement basis and completes it (measuring
//! and storing what it lacks, printing each entry as it completes) while
//! every model is prepared from its headers; then assesses every model and
//! prints one JSON object per model, shaped like the service's per-profile
//! assessment result, with the model's capabilities and template fingerprint
//! alongside.

use magnitude_engine::{
    assessment::{
        AssessmentEnvironment, AssessmentSetup, ModelAssessment, ModelPackagePaths,
        PreparedModelAssessment, finish_model_assessment, prepare_model_assessment,
    },
    error::UnsupportedModel,
    options::{ModelMethod, ModelPolicy, standard_service_limits},
    worker::protocol::EngineBuild,
};
use magnitude_executor::{
    DEFAULT_KERNEL_CACHE_BYTES, ExecutionPath, KernelCache,
    assessment::{
        BasisIdentity, ClassMeasurement, DomainFit, ExecutionAssessment, IncompatibleReason,
        MeasurementBasis, PerformanceConfidence, complete_basis, load_basis, store_basis,
        term_seconds,
    },
    platform::{self, DeviceRequest, MemoryReserves, PlatformConfig},
};
use seismic::DeviceCatalog;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const STANDARD_DEPTHS: [u32; 3] = [25_000, 50_000, 75_000];

struct Options {
    device: DeviceRequest,
    cache_dir: PathBuf,
    depths: Vec<u32>,
    /// Print every demand term's share of the step at each depth.
    breakdown: bool,
    models: Vec<ModelPackagePaths>,
}

const USAGE: &str = "magnitude-assess --device auto|metal|cuda|vulkan|cpu --cache-dir DIR \
     [--depths 25000,50000,75000] [--breakdown] TARGET.gguf[,[PROJECTOR.gguf][,DRAFT.gguf]]...";

fn value(flag: &str, args: &mut impl Iterator<Item = String>) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse() -> Result<Options, String> {
    let mut device = None;
    let mut cache_dir = None;
    let mut depths = STANDARD_DEPTHS.to_vec();
    let mut breakdown = false;
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
            "--breakdown" => breakdown = true,
            "--help" | "-h" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            flag if flag.starts_with("--") => {
                return Err(format!("unknown flag: {flag} (try --help)"));
            }
            model => {
                // TARGET[,PROJECTOR[,DRAFT]]; an empty projector names none.
                let mut components = model.splitn(3, ',');
                let component = |path: Option<&str>| {
                    path.filter(|path| !path.is_empty()).map(PathBuf::from)
                };
                models.push(ModelPackagePaths {
                    target: PathBuf::from(components.next().expect("split yields one part")),
                    projector: component(components.next()),
                    draft: component(components.next()),
                    method: ModelMethod::Auto,
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
        breakdown,
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

/// The basis stored for this device and build, completed with what it lacks
/// (measured now and stored for the next run).
fn basis(
    catalog: &DeviceCatalog,
    setup: &AssessmentSetup,
    cache_dir: &Path,
) -> Result<MeasurementBasis, String> {
    let kernels = KernelCache::open(cache_dir.join("kernels"), DEFAULT_KERNEL_CACHE_BYTES)
        .map_err(|error| error.to_string())?;
    let opened = platform::open_selected(
        catalog,
        setup.selected.info.selector,
        PlatformConfig {
            path: ExecutionPath::Native,
            artifacts: Some(Arc::new(kernels) as Arc<dyn seismic::ArtifactStore>),
            reserves: setup.reserves,
        },
    )
    .map_err(|error| error.to_string())?;
    let identity = BasisIdentity::for_device(opened.device(), &EngineBuild::current().0);
    let directory = cache_dir.join("assessment-basis");
    let stored = load_basis(&directory, &identity).unwrap_or(MeasurementBasis {
        identity,
        classes: Vec::new(),
    });
    let known = stored.classes.len();
    let started = Instant::now();
    let basis = complete_basis(
        catalog,
        opened.device(),
        setup.reserves,
        stored,
        |key, measurement, profile| {
            let outcome = match measurement {
                ClassMeasurement::Measured { .. } => "measured".to_owned(),
                ClassMeasurement::Formed => "formed".to_owned(),
                ClassMeasurement::Unsupported { reason } => format!("unsupported: {reason}"),
            };
            eprintln!(
                "magnitude-assess: {:>7.3} s (formation {:.3}, allocation {:.3}, device {:.3}) \
                 {key} {outcome}",
                profile.total.as_secs_f64(),
                profile.formation.as_secs_f64(),
                profile.allocation.as_secs_f64(),
                profile.device.as_secs_f64(),
            );
        },
    );
    let basis = match basis {
        Ok(basis) => basis,
        // The classes measured before the failure are kept for the next run.
        Err(failure) => {
            if failure.basis.classes.len() > known {
                store_basis(&directory, &failure.basis).map_err(|error| error.to_string())?;
            }
            return Err(failure.error.to_string());
        }
    };
    if basis.classes.len() == known {
        eprintln!(
            "magnitude-assess: the stored basis for {} is complete",
            basis.identity.device
        );
    } else {
        eprintln!(
            "magnitude-assess: measured {} entries in {:.2} s",
            basis.classes.len() - known,
            started.elapsed().as_secs_f64()
        );
        store_basis(&directory, &basis).map_err(|error| error.to_string())?;
    }
    Ok(basis)
}

/// Returns whether every model produced a result.
fn run() -> Result<bool, String> {
    let options = parse()?;
    let catalog = DeviceCatalog::discover().map_err(|error| error.to_string())?;
    // The standalone engine's defaults: one conversation per batch.
    let setup = Arc::new(
        AssessmentSetup::discover(
            &catalog,
            options.device,
            MemoryReserves::standard(),
            ModelPolicy::default(),
            standard_service_limits(),
        )
        .map_err(|error| error.to_string())?,
    );
    let mut complete = true;
    let mut report = |model: &ModelPackagePaths, error: &dyn std::fmt::Display| {
        complete = false;
        eprintln!("magnitude-assess: {}: {error}", model.target.display());
    };
    // The basis is model-free: it is completed while the models are
    // prepared.
    let (basis, prepared) = std::thread::scope(|scope| {
        let basis = scope.spawn(|| basis(&catalog, &setup, &options.cache_dir));
        let prepared = options
            .models
            .iter()
            .map(|model| {
                let started = Instant::now();
                (model, prepare_model_assessment(model, &setup), started.elapsed())
            })
            .collect::<Vec<_>>();
        (basis.join().expect("measurement thread panicked"), prepared)
    });
    let basis = basis?;
    let prepared = prepared
        .into_iter()
        .filter_map(|(model, prepared, preparation)| match prepared {
            Ok(prepared) => Some((model, prepared, preparation)),
            Err(error) => {
                report(model, &error);
                None
            }
        })
        .collect::<Vec<_>>();
    let environment = Arc::clone(&setup)
        .with_basis(basis)
        .map_err(|error| error.to_string())?;
    // Assessment time is the model's own preparation and finish, without the
    // shared measurement.
    for (model, prepared, preparation) in &prepared {
        if options.breakdown {
            print_breakdown(model, prepared, &environment, &options.depths);
        }
        let started = Instant::now();
        match finish_model_assessment(prepared, &environment, &options.depths) {
            Ok(assessment) => {
                let mut object = render(&environment, &assessment);
                object["model"] = json!(model.target.display().to_string());
                object["assessmentSeconds"] =
                    json!((*preparation + started.elapsed()).as_secs_f64());
                println!("{object}");
            }
            Err(error) => report(model, &error),
        }
    }
    Ok(complete)
}

/// Every demand term's median microseconds per step at each depth, largest
/// first, on stderr: the estimate's composition, to compare with a real
/// decode's per-entry attribution.
fn print_breakdown(
    model: &ModelPackagePaths,
    prepared: &PreparedModelAssessment,
    environment: &AssessmentEnvironment,
    depths: &[u32],
) {
    let PreparedModelAssessment::Planned { execution, .. } = prepared else {
        return;
    };
    for &depth in depths {
        match term_seconds(execution.demand(), &environment.basis, depth) {
            Ok(mut terms) => {
                terms.sort_by(|left, right| right.1.median.total_cmp(&left.1.median));
                let step = terms.iter().map(|(_, seconds)| seconds.median).sum::<f64>();
                eprintln!(
                    "magnitude-assess: {} at {depth}: {:.1} µs per step",
                    model.target.display(),
                    step * 1e6
                );
                for (term, seconds) in terms {
                    eprintln!(
                        "magnitude-assess:   {:>9.1} µs  {:>4} launches  {}",
                        seconds.median * 1e6,
                        term.launches,
                        term.key
                    );
                }
            }
            Err(error) => eprintln!("magnitude-assess: breakdown at {depth}: {error}"),
        }
    }
}

fn render(environment: &AssessmentEnvironment, assessment: &ModelAssessment) -> Value {
    let (facts, execution) = match assessment {
        ModelAssessment::Unsupported(unsupported) => {
            let code = match unsupported {
                UnsupportedModel::Family { .. } => "unsupported_family",
                UnsupportedModel::Representation { .. } => "unsupported_representation",
                UnsupportedModel::Backend { .. } => "unsupported_backend",
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
                    .map(|(key, reason)| match reason {
                        Some(reason) => format!("{key}: {reason}"),
                        None => key.to_string(),
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
    if domain == environment.setup.topology.host_pool().id {
        "system".to_owned()
    } else {
        environment.setup.selected.info.selector.to_string()
    }
}
