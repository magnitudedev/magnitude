//! Metadata-only model assessment on this machine's selected device.
//!
//! Prepares every model from its headers and assesses it against stable
//! memory capacity and the device's memory bandwidth; prints one JSON object
//! per model, shaped like the service's per-profile assessment result, with
//! the model's capabilities and template fingerprint alongside. With
//! `--topology`, prints the device topology and its discovery diagnostics
//! instead. Opens no device.

use magnitude_engine::{
    assessment::{
        AssessmentSetup, ModelAssessment, ModelPackagePaths, PreparedModelAssessment,
        finish_model_assessment, prepare_model_assessment,
    },
    error::UnsupportedModel,
    options::{ModelMethod, ModelPolicy, standard_service_limits},
};
use magnitude_executor::{
    assessment::{BandwidthSource, DeviceClass, DomainFit, ExecutionAssessment, step_seconds},
    platform::{DeviceRequest, MemoryReserves},
};
use seismic::{Devices, DiagnosticSource, StoreId, StoreIdentity};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::time::Instant;

const STANDARD_DEPTHS: [u32; 3] = [25_000, 50_000, 75_000];

struct Options {
    device: DeviceRequest,
    depths: Vec<u32>,
    /// Print the parts of the step at each depth.
    breakdown: bool,
    /// Print the topology and its discovery diagnostics rather than assess
    /// models.
    topology: bool,
    models: Vec<ModelPackagePaths>,
}

const USAGE: &str = "magnitude-assess --device auto|metal|cuda|vulkan|cpu \
     [--depths 25000,50000,75000] [--breakdown] \
     (--topology | TARGET.gguf[,[PROJECTOR.gguf][,DRAFT.gguf]]...)";

fn value(flag: &str, args: &mut impl Iterator<Item = String>) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse() -> Result<Options, String> {
    let mut device = None;
    let mut depths = STANDARD_DEPTHS.to_vec();
    let mut breakdown = false;
    let mut topology = false;
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
            "--topology" => topology = true,
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
    if topology == !models.is_empty() {
        return Err(format!(
            "give --topology or models, not both or neither\n{USAGE}"
        ));
    }
    Ok(Options {
        device: device.ok_or("--device is required")?,
        depths,
        breakdown,
        topology,
        models,
    })
}

fn main() {
    magnitude_executor::platform::relax_gpu_watchdog();
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(error) => {
            eprintln!("magnitude-assess: {error}");
            std::process::exit(1);
        }
    }
}

/// Returns whether every model produced a result.
fn run() -> Result<bool, String> {
    let options = parse()?;
    // The directory of each model is a served path: its assessed load reads
    // from that volume.
    let mut served = Vec::new();
    for model in &options.models {
        let directory = std::path::absolute(&model.target)
            .map_err(|error| error.to_string())?
            .parent()
            .ok_or_else(|| format!("{} has no directory", model.target.display()))?
            .to_path_buf();
        if !served.contains(&directory) {
            served.push(directory);
        }
    }
    let catalog = Devices::discover(&served).map_err(|error| error.to_string())?;
    // The standalone engine's defaults: one conversation per batch.
    let setup = AssessmentSetup::discover(
        &catalog,
        options.device,
        MemoryReserves::standard(),
        ModelPolicy::default(),
        standard_service_limits(),
    )
    .map_err(|error| error.to_string())?;
    let source = match setup.bandwidth.source {
        BandwidthSource::Reported => "reported by the driver".to_owned(),
        BandwidthSource::Published => "published".to_owned(),
        BandwidthSource::Assumed(class) => format!(
            "assumed for {}",
            match class {
                DeviceClass::DedicatedGpu => "a dedicated GPU",
                DeviceClass::IntegratedGpu => "an integrated GPU",
                DeviceClass::Cpu => "a CPU",
            }
        ),
    };
    let executor = &setup.selected.info;
    eprintln!(
        "magnitude-assess: {} ({}): {:.0} GB/s, {source}",
        executor.name(),
        executor.backend().as_str(),
        setup.bandwidth.bytes_per_second as f64 / 1e9
    );
    if options.topology {
        // Why a backend or adapter is absent from the graph (an absent
        // driver, an enumeration failure), as discovery retained it.
        let diagnostics = setup
            .topology
            .diagnostics()
            .iter()
            .map(|diagnostic| {
                let source = match diagnostic.source {
                    DiagnosticSource::Backend(backend) => json!({ "backend": backend.as_str() }),
                    DiagnosticSource::Adapters => json!("adapters"),
                };
                json!({ "source": source, "message": diagnostic.message })
            })
            .collect::<Vec<_>>();
        let topology: Value = serde_json::from_str(&setup.topology.normalized())
            .expect("the normalized topology is JSON");
        println!(
            "{}",
            json!({
                "topology": topology,
                "diagnostics": diagnostics,
                "executor": executor.selector().to_string(),
            })
        );
        return Ok(true);
    }
    let mut complete = true;
    for model in &options.models {
        let started = Instant::now();
        let assessed = prepare_model_assessment(model, &setup).and_then(|prepared| {
            if options.breakdown {
                print_breakdown(model, &prepared, &setup, &options.depths);
            }
            finish_model_assessment(&prepared, &setup, &options.depths)
        });
        match assessed {
            Ok(assessment) => {
                let mut object = render(&setup, &assessment);
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

/// The parts of the step at each depth, on stderr: the estimate's
/// composition, to compare with a real decode.
fn print_breakdown(
    model: &ModelPackagePaths,
    prepared: &PreparedModelAssessment,
    setup: &AssessmentSetup,
    depths: &[u32],
) {
    let PreparedModelAssessment::Planned { execution, .. } = prepared else {
        return;
    };
    let demand = execution.demand();
    eprintln!(
        "magnitude-assess: {}: {} launches, {:.2} GB streamed per step",
        model.target.display(),
        demand.launches,
        demand.streamed_bytes as f64 / 1e9
    );
    for &depth in depths {
        let parts = step_seconds(demand, setup.bandwidth, depth);
        eprintln!(
            "magnitude-assess:   at {depth}: {:.1} µs per step = step {:.1} + launches {:.1} \
             + weights {:.1} + history {:.1}",
            parts.total() * 1e6,
            parts.step * 1e6,
            parts.launches * 1e6,
            parts.weights * 1e6,
            parts.history * 1e6,
        );
    }
}

fn render(setup: &AssessmentSetup, assessment: &ModelAssessment) -> Value {
    let (facts, execution) = match assessment {
        ModelAssessment::Unsupported(unsupported) => {
            let code = match unsupported {
                UnsupportedModel::Family { .. } => "unsupported_family",
                UnsupportedModel::Representation { .. } => "unsupported_representation",
                UnsupportedModel::Backend { .. } => "unsupported_backend",
                UnsupportedModel::KernelDomain { .. } => "unsupported_kernel_domain",
            };
            return json!({
                "_tag": "Unsupported",
                "failure": { "code": code, "message": unsupported.to_string(), "retryable": false },
            });
        }
        ModelAssessment::Assessed { facts, execution } => (facts, execution),
    };
    let memory = |domains: &[DomainFit]| {
        domains
            .iter()
            .map(|domain| {
                json!({
                    "memoryDomainId": domain_id(setup, domain.domain),
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
                "estimatedTokensPerSecond": estimate.tokens_per_second,
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
            "limitingResource": domain_id(setup, *limiting),
            "deficitBytes": deficit_bytes,
        }),
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

/// Host RAM is the `system` domain; another store is named by its identity.
fn domain_id(setup: &AssessmentSetup, domain: StoreId) -> String {
    match setup
        .topology
        .store_identity(domain)
        .expect("assessment domain belongs to topology")
    {
        StoreIdentity::HostRam => "system".to_owned(),
        identity => identity.to_string(),
    }
}
