//! Metadata-only model assessment on this machine's selected device.
//!
//! Measures the machine first, in process, through the function the
//! measurement job runs (`worker::measure::measure_machine`): the records of
//! the selected executor's cost points the kernel cache holds are kept, and
//! only the points without one are timed (which opens the executor). Then
//! derives the executor's costs from the records as a start does, prepares
//! every model from its headers and assesses it against stable memory
//! capacity and, when it fits, estimates its decode speed from those costs;
//! prints one JSON object per model, shaped like the service's per-profile
//! assessment result, with the model's capabilities and template
//! fingerprint alongside. `--topology` prints the topology, its discovery
//! diagnostics and the executor's costs (each with its samples and interval)
//! instead; `--breakdown` prints each model's step call by call.

use magnitude_engine::{
    assessment::{
        finish_model_assessment, prepare_model_assessment, AssessmentSetup, AssessmentStructure,
        DrafterVerdict, Measurements, ModelAssessment, ModelExecution, ModelPackagePaths,
        Performance, PreparedModelAssessment,
    },
    error::UnsupportedModel,
    options::{standard_service_limits, ModelMethod, ModelPolicy, PROFILE_SEQUENCES},
    worker::measure::{measure_machine, MeasureEvent, MeasureJob, MeasureRequest},
};
use magnitude_executor::{
    assessment::{
        decode_pricing, step_seconds, DomainFit, ExecutionFit, ExecutionShortfall, ExecutorCosts,
        MeasuredCost,
    },
    platform::{DeviceRequest, MemoryReserves},
};
use seismic::{Devices, DiagnosticSource, StoreId, StoreIdentity};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

const STANDARD_DEPTHS: [u32; 3] = [25_000, 50_000, 75_000];

struct Options {
    device: DeviceRequest,
    /// The kernel cache, which holds the machine's measurements.
    kernels: PathBuf,
    depths: Vec<u32>,
    /// The conversations decoding together.
    sequences: u32,
    /// Print each model's step call by call at each depth.
    breakdown: bool,
    /// Print the topology, its discovery diagnostics, its measurements and
    /// the executor costs rather than assess models.
    topology: bool,
    models: Vec<ModelPackagePaths>,
}

const USAGE: &str = "magnitude-assess --device auto|metal|cuda|vulkan|cpu --kernels DIR \
     [--depths 25000,50000,75000] [--sequences 1] [--breakdown] [--method auto|plain] \
     (--topology | TARGET.gguf[,[PROJECTOR.gguf][,DRAFT.gguf]]...)";

fn value(flag: &str, args: &mut impl Iterator<Item = String>) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse() -> Result<Options, String> {
    let mut device = None;
    let mut kernels = None;
    let mut depths = STANDARD_DEPTHS.to_vec();
    // A serving profile's, as the service estimates.
    let mut sequences = PROFILE_SEQUENCES;
    let mut breakdown = false;
    let mut method = ModelMethod::Auto;
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
            "--sequences" => {
                sequences = value(&argument, &mut args)?
                    .parse()
                    .map_err(|error| format!("--sequences: {error}"))?
            }
            "--kernels" => kernels = Some(PathBuf::from(value(&argument, &mut args)?)),
            "--breakdown" => breakdown = true,
            "--method" => {
                method = match value(&argument, &mut args)?.as_str() {
                    "auto" => ModelMethod::Auto,
                    "plain" => ModelMethod::Plain,
                    other => return Err(format!("--method takes auto or plain, not {other}")),
                }
            }
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
    for model in &mut models {
        model.method = method;
    }
    Ok(Options {
        device: device.ok_or("--device is required")?,
        kernels: kernels.ok_or("--kernels is required")?,
        depths,
        sequences,
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
    // The kernel cache's volume is a served path, and so is the directory
    // of each model, whose volume its assessed load reads from.
    std::fs::create_dir_all(&options.kernels).map_err(|error| error.to_string())?;
    let mut served = vec![options.kernels.clone()];
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
    let structure = AssessmentStructure::discover(
        &catalog,
        options.device,
        MemoryReserves::standard(),
        ModelPolicy::default(),
        standard_service_limits(),
    )
    .map_err(|error| error.to_string())?;
    let executor = &structure.selected.info;
    eprintln!(
        "magnitude-assess: {} ({})",
        executor.name(),
        executor.backend().as_str()
    );

    // The models are prepared from their headers; the measurements are
    // the executor's own, whatever the models read.
    let prepared = options
        .models
        .iter()
        .map(|model| {
            let started = Instant::now();
            (
                prepare_model_assessment(model, &structure),
                started.elapsed(),
            )
        })
        .collect::<Vec<_>>();
    // Measured as the job measures: what is stored is kept, and each record
    // stored now is printed.
    let started = Instant::now();
    let failures = measure_machine(
        &MeasureRequest {
            paths: served,
            kernels: options.kernels.clone(),
            executor: executor.selector(),
            // Nothing preempts the tool's job; its process names it.
            job: MeasureJob(u64::from(std::process::id())),
        },
        &mut |event| {
            if let MeasureEvent::Recorded(key, record) = event {
                eprintln!(
                    "magnitude-assess: {:.3} s: {key:?}: {record:?}",
                    started.elapsed().as_secs_f64()
                )
            }
        },
    );
    let measure_seconds = started.elapsed().as_secs_f64();
    eprintln!("magnitude-assess: measured in {measure_seconds:.3} s");
    for failure in &failures {
        eprintln!("magnitude-assess: not measured: {failure}");
    }
    let measurements = structure.stored_measurements(&options.kernels);

    if options.topology {
        let costs = match &measurements {
            Measurements::Measured {
                costs: measured, ..
            } => costs(measured),
            Measurements::Pending => json!("pending"),
            Measurements::Unavailable { reason } => json!({ "unavailable": reason }),
        };
        let topology = catalog.topology();
        // Why a backend or adapter is absent from the graph (an absent
        // driver, an enumeration failure), as discovery retained it.
        let diagnostics = topology
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
        let normalized: Value =
            serde_json::from_str(&topology.normalized()).expect("the normalized topology is JSON");
        println!(
            "{}",
            json!({
                "topology": normalized,
                "diagnostics": diagnostics,
                "executor": executor.selector().to_string(),
                "executorCosts": costs,
                "failures": failures.iter().map(ToString::to_string).collect::<Vec<_>>(),
                "measureSeconds": measure_seconds,
            })
        );
        return Ok(failures.is_empty());
    }

    let setup = AssessmentSetup {
        structure: Arc::new(structure),
        measurements,
    };
    let mut complete = true;
    for (model, (prepared, preparation)) in options.models.iter().zip(prepared) {
        let started = Instant::now();
        let assessed = prepared.and_then(|prepared| {
            if options.breakdown {
                print_breakdown(model, &prepared, &setup, &options)?;
            }
            finish_model_assessment(&prepared, &setup, &options.depths, options.sequences)
        });
        match assessed {
            Ok(assessment) => {
                let mut object = render(&setup, &assessment);
                object["model"] = json!(model.target.display().to_string());
                object["assessmentSeconds"] =
                    json!((preparation + started.elapsed()).as_secs_f64());
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

fn cost(cost: &MeasuredCost) -> Value {
    json!({ "value": cost.value, "interval": cost.interval, "samples": cost.samples })
}

/// The executor's costs, each with its interval and samples.
fn costs(costs: &ExecutorCosts) -> Value {
    json!({
        "weightRates": costs.weight_rates.iter().map(|rate| {
            // The launch cost's share of one call: its two weights of F × 4096
            // at the rate measured.
            let call = rate
                .format
                .canonical_byte_len(&[rate.features, 4096])
                .map(|bytes| 2.0 * bytes as f64 / rate.bytes_per_second.value)
                .ok();
            json!({
                "route": rate.route,
                "format": rate.format.name(),
                "bytesPerSecond": cost(&rate.bytes_per_second),
                "features": rate.features,
                "launchFraction": call.map(|call| costs.launch_seconds.value / call),
            })
        }).collect::<Vec<_>>(),
        "matrixFlopsPerSecond": costs.matrices.iter().map(|matrix| json!({
            "format": matrix.format.name(),
            "flops": cost(&matrix.flops),
            "features": matrix.features,
        })).collect::<Vec<_>>(),
        "history": costs.histories.iter().map(|history| json!({
            "layout": history.layout.identity(),
            "fixedSeconds": cost(&history.fixed_seconds),
            "bytesPerSecond": cost(&history.bytes_per_second),
            "depths": history.depths,
        })).collect::<Vec<_>>(),
        "composition": cost(&costs.composition),
        "launchSeconds": cost(&costs.launch_seconds),
        "stepPipelinedSeconds": cost(&costs.step_pipelined_seconds),
        "stepRoundTripSeconds": cost(&costs.step_round_trip_seconds),
        "unmeasured": costs.unmeasured.iter().map(|unmeasured| json!({
            "figure": unmeasured.figure.to_string(),
            "reason": unmeasured.reason,
        })).collect::<Vec<_>>(),
    })
}

/// Each call of the step at each depth, on stderr: whether its bytes or its
/// arithmetic bound it, its launches, and the step cost, to compare with a
/// real decode.
fn print_breakdown(
    model: &ModelPackagePaths,
    prepared: &PreparedModelAssessment,
    setup: &AssessmentSetup,
    options: &Options,
) -> Result<(), magnitude_engine::assessment::ModelAssessmentError> {
    let PreparedModelAssessment::Planned {
        draft, execution, ..
    } = prepared
    else {
        return Ok(());
    };
    let Measurements::Measured { costs, .. } = &setup.measurements else {
        eprintln!(
            "magnitude-assess: {}: no breakdown without measurements",
            model.target.display()
        );
        return Ok(());
    };
    let charge = execution.memory_charge();
    let bounds = &charge.bounds;
    let host = &bounds.host_process;
    eprintln!(
        "magnitude-assess: {}: memory {}",
        model.target.display(),
        json!({
            "exact_resident_bytes": charge.exact_resident_bytes,
            "state_bytes": charge.state_bytes,
            "startup_state_bytes": charge.startup_state_bytes,
            "prepared_resource_bytes": bounds.prepared_resource_bytes,
            "startup_additional_bytes": bounds.startup_additional_bytes,
            "component_upload_bytes": bounds.component_upload_bytes,
            "serving_bytes": charge.serving_bytes,
            "allocation_bytes": charge.allocation_bytes,
            "host_table_bytes": charge.host_table_bytes,
            "sealed_graph_host_bytes": bounds.sealed_graph_host_bytes.at_most,
            "bound_graph_host_bytes": bounds.bound_graph_host_bytes,
            "drafter_graph_host_bytes": bounds.drafter_graph_host_bytes,
            "host_runtime_bytes": host.runtime,
            "host_vocabulary_bytes": host.vocabulary,
            "host_program_bytes": host.programs,
            "host_vision_bytes": host.vision,
            "device_context_bytes": bounds.device_process.context,
            "device_program_bytes": bounds.device_process.programs,
            "device_slack_bytes": bounds.device_process.slack,
            "kernels": host.kernels,
            "host_bytes": charge.host_bytes().ok(),
        })
    );
    let failed = magnitude_engine::assessment::ModelAssessmentError::Assessment;
    let sequences = u64::from(options.sequences);
    let pricing = decode_pricing(draft, costs);
    let demand = execution.demand();
    eprintln!(
        "magnitude-assess: {}: {} calls, {sequences} sequences, lookahead {}, activations at \
         {:.1} GB/s, uploads at {:.1} GB/s",
        model.target.display(),
        demand.calls.len(),
        pricing.lookahead,
        pricing
            .activations(seismic::StorageKind::Device)
            .map_err(failed)?
            / 1e9,
        pricing
            .activations(seismic::StorageKind::Upload)
            .map_err(failed)?
            / 1e9,
    );
    for &depth in &options.depths {
        let step = step_seconds(demand, &pricing, sequences, depth).map_err(failed)?;
        let bound = |compute: bool| {
            step.calls
                .iter()
                .filter(|call| call.compute_bound() == compute)
                .map(|call| call.bytes.max(call.compute))
                .sum::<f64>()
        };
        eprintln!(
            "magnitude-assess:   at {depth}: {:.1} µs per step = {:.3} × (bytes-bound {:.1} + \
             compute-bound {:.1} + attention fixed {:.1} + {} launches {:.1}) + step {:.1}",
            step.total() * 1e6,
            step.composition,
            bound(false) * 1e6,
            bound(true) * 1e6,
            step.calls.iter().map(|call| call.fixed).sum::<f64>() * 1e6,
            step.calls.iter().map(|call| call.launches).sum::<u64>(),
            step.calls.iter().map(|call| call.launch).sum::<f64>() * 1e6,
            step.step * 1e6,
        );
        for (call, seconds) in demand.calls.iter().zip(&step.calls) {
            eprintln!(
                "magnitude-assess:     {:<32} {:>9.2} µs bytes {:>9.2} µs compute \
                 {:>7.2} µs fixed {} launches",
                call.launches.kernel().name(),
                seconds.bytes * 1e6,
                seconds.compute * 1e6,
                seconds.fixed * 1e6,
                seconds.launches,
            );
        }
    }
    Ok(())
}

fn render(setup: &AssessmentSetup, assessment: &ModelAssessment) -> Value {
    let (facts, execution) = match assessment {
        ModelAssessment::Unsupported {
            unsupported,
            fingerprint,
        } => {
            let mut object = unsupported_json(unsupported);
            object["_tag"] = json!("Unsupported");
            object["fingerprint"] = json!(fingerprint);
            return object;
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
        ModelExecution::Fits {
            fit: ExecutionFit { domains, .. },
            performance,
        } => json!({
            "_tag": "Fits",
            "memory": memory(domains),
            "performance": match performance {
                Performance::Pending => json!({ "_tag": "Pending" }),
                Performance::Unavailable { reason } => {
                    json!({ "_tag": "Unavailable", "reason": reason })
                }
                Performance::Estimated(estimates) => json!({
                    "_tag": "Estimated",
                    "decode": estimates.iter().map(|estimate| json!({
                        "contextTokens": estimate.context_tokens,
                        "estimatedTokensPerSecond": estimate.tokens_per_second,
                    })).collect::<Vec<_>>(),
                }),
            },
        }),
        ModelExecution::DoesNotFit(ExecutionShortfall {
            domains,
            limiting,
            deficit_bytes,
            ..
        }) => json!({
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
    object["fingerprint"] = json!(facts.fingerprint);
    object["drafter"] = match &facts.drafter {
        DrafterVerdict::None => json!({ "_tag": "None" }),
        DrafterVerdict::Bound {
            method,
            memory_bytes,
        } => json!({
            "_tag": "Bound",
            "method": format!("{method:?}"),
            "memoryBytes": memory_bytes,
        }),
        DrafterVerdict::Rejected(unsupported) => {
            let mut object = unsupported_json(unsupported);
            object["_tag"] = json!("Rejected");
            object
        }
    };
    object
}

/// The reasons (by their debug form, developer output) and detail of a
/// refusal.
fn unsupported_json(unsupported: &UnsupportedModel) -> Value {
    json!({
        "reasons": unsupported
            .reasons
            .iter()
            .map(|reason| format!("{reason:?}"))
            .collect::<Vec<_>>(),
        "detail": unsupported.detail,
    })
}

/// Host RAM is the `system` domain; every other store is named by its store
/// identity.
fn domain_id(setup: &AssessmentSetup, domain: StoreId) -> String {
    match setup
        .structure
        .topology
        .store_identity(domain)
        .expect("assessment domain belongs to topology")
    {
        StoreIdentity::HostRam => "system".to_owned(),
        identity => identity.to_string(),
    }
}
