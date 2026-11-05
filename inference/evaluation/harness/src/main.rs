//! `magnitude-harness`: the only in-process driver of the engine for
//! evaluation. Every engine subcommand takes a catalog model (its acquired
//! paths, `--artifacts`) and a fixture set, renders the fixtures through the
//! engine's own serving chat path, and writes an evaluation result record
//! (`result.json`) with its traces
//! into a run directory. `mrig` runs it; each subcommand serves instruments:
//!
//! - `render`: the exact tokens of each fixture (`rendered.jsonl`).
//! - `time`: prefill and decode throughput at context lengths and
//!   concurrency (`step`).
//! - `logits`: the distributions at every scored position (`logits.f32`,
//!   `logits.json`), judged by `magnitude-references kl` (`kl`).
//! - `draft`: the catalog drafter's proposals at anchors in the scored spans
//!   (`draft.jsonl`, `draft`).
//! - `prepare`: the model's Prepare alone (its tuning search, publishing
//!   records), imported nowhere: with `--tuning-survey`, the time of every
//!   configuration of the surveyed entries at the model's points, from the
//!   model's headers alone (a catalog header export opens as the model with
//!   zero weights) (`kernel`).
//! - `memory`, `recommit`: one load's host memory against its charge, and
//!   what a reserved tensor's recommit costs, loading nothing (`memory`).
//!
//! Every engine subcommand takes the development tuning tools
//! ([`development`]): pins and the survey.
//!
//! Every request is rendered with `--request-options`, the request fields the
//! serving benchmark sends (reasoning disabled by default), recorded in each
//! rendered fixture. Every load is the engine's sequence: the load, in this
//! process, which never searches; and, when it finds tuning units without a
//! stored record, first the model's Prepare in a worker process of this
//! executable, which searches them and exits. Each appends its tuning trace
//! to `tuning.jsonl`.

mod artifacts;
mod bar;
mod development;
mod digest;
mod draft;
mod engine;
mod fixture;
mod logits;
mod machine;
mod memory;
mod record;
mod render;
mod rest;
mod selection;
mod steps;
mod time;
mod tuning;

use artifacts::ArtifactsFile;
use clap::{Args, Parser, Subcommand};
use engine::EngineOptions;
use fixture::FixtureSet;
use magnitude_engine::host::HostArtifacts;
use magnitude_engine::options::{PackageOptions, ProjectorSelection};
use magnitude_executor::STANDARD_TUNING_BUDGET;
use record::{Run, RunRecord};
use render::{Rendered, Renderer, RequestOptions};
use serde_json::{Map, Value, json};
use std::cell::Cell;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Parser)]
#[command(
    name = "magnitude-harness",
    about = "Run fixture sets through the engine in process"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// The engine's worker over standard streams: what the harness starts
    /// itself as for a model's Prepare, when a load finds tuning records
    /// missing.
    #[command(hide = true, name = engine::WORKER_COMMAND)]
    Worker {
        #[command(flatten)]
        survey: development::SurveyArgs,
    },
    /// Render fixtures to the exact tokens the engine serves.
    Render {
        #[command(flatten)]
        run: RunArgs,
        /// Size each fixture to this many tokens.
        #[arg(long)]
        context: Option<usize>,
    },
    /// Prefill and decode throughput.
    Time {
        #[command(flatten)]
        run: RunArgs,
        #[command(flatten)]
        engine: EngineArgs,
        /// Context lengths the fixtures are sized to.
        #[arg(long, value_delimiter = ',', default_value = "4096,16384")]
        contexts: Vec<usize>,
        /// Sequences prefilled or decoded together.
        #[arg(long, value_delimiter = ',', default_value = "1,4")]
        concurrency: Vec<usize>,
        /// Decode steps before measuring.
        #[arg(long, default_value_t = 16)]
        warm: usize,
        /// Measured decode steps.
        #[arg(long, default_value_t = 32)]
        steps: usize,
        /// Decode steps whose device time is attributed by entry, each launch
        /// timed alone (a diagnostic).
        #[arg(long, default_value_t = 0)]
        attribution_steps: usize,
        /// One further prefill sample whose device time is attributed by
        /// entry, each launch timed alone (a diagnostic).
        #[arg(long)]
        prefill_attribution: bool,
        /// Prefill samples, the first of them cold.
        #[arg(long, default_value_t = 4)]
        prefill_samples: usize,
        /// Queue each decode step's successor before it completes, as served.
        #[arg(long, default_value = "on", value_parser = on_off, action = clap::ArgAction::Set)]
        lookahead: bool,
        /// A resident load's `time` result on this host, whose decode at each
        /// context and concurrency is the resident step a streamed load's
        /// decode is priced on (`assessment::cost`).
        #[arg(long)]
        resident_decode: Option<PathBuf>,
        /// The run counts rather than times (a routing trace): no overlap
        /// efficiency is measured, the one timing a resident load adds.
        #[arg(long)]
        untimed: bool,
    },
    /// One load's host memory against its charge, in a worker process
    /// (`memory`).
    Memory {
        /// A unified-memory backend: metal or cpu.
        #[arg(long)]
        device: String,
        /// Load at this context; the model's declared one by default.
        #[arg(long)]
        context: Option<usize>,
        /// A ceiling on every weight byte the load holds.
        #[arg(long, value_name = "BYTES")]
        weight_cap: Option<u64>,
        /// Decode the first fixture of this rendered set once loaded.
        #[arg(long, value_name = "RENDERED", requires = "decode_tokens")]
        decode: Option<PathBuf>,
        /// Tokens the decode generates; under --pressure, the tokens before
        /// pressing (a quarter), under the held pressure (half) and after
        /// it (a quarter), the decode sized to fit them.
        #[arg(long, requires = "decode")]
        decode_tokens: Option<usize>,
        /// Decode from the fixture's first N tokens: a pressured decode's
        /// prompt, as it printed it (`decode_prompt_tokens`), for the same
        /// decode resident.
        #[arg(long, value_name = "N", requires = "decode", conflicts_with = "pressure")]
        decode_prompt_tokens: Option<usize>,
        /// Another process presses the load BYTES below its planning line
        /// over a window of the decode.
        #[arg(long, value_name = "BYTES", requires_all = ["decode", "pressure_store"])]
        pressure: Option<u64>,
        /// The pressure starts once the request is admitted, so its prompt's
        /// groups run under it.
        #[arg(long, requires = "pressure")]
        pressure_prompt: bool,
        /// Physical store to press; device resolves every allocation placement.
        #[arg(long, requires = "pressure")]
        pressure_store: Option<memory::PressureStore>,
        /// Keep the loaded worker until standard input closes.
        #[arg(long)]
        hold: bool,
        /// The kernel cache the load prepares from and keeps its
        /// measurements in, as a served load's: its executor's costs price
        /// the expert bank's growth.
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// The target model file.
        target: PathBuf,
    },
    /// What an in-place recommit of a reserved tensor (an elastic expert
    /// bank's resize) costs at each commitment, loading nothing.
    Recommit {
        /// A unified-memory backend: metal or cpu.
        #[arg(long)]
        device: String,
        /// Commitments, in gigabytes.
        #[arg(long, value_delimiter = ',', required = true)]
        gigabytes: Vec<u64>,
    },
    /// Another process's memory: what `memory --pressure` starts beside a
    /// load to press it below its planning line.
    #[command(hide = true)]
    Occupy {
        #[arg(long)]
        device: String,
        /// Bytes below the planning line.
        #[arg(long)]
        deficit: u64,
        #[arg(long)]
        pressure_store: memory::PressureStore,
    },
    /// How long occupying takes here: everything down to the planning line
    /// taken, with no load beside it, timed and given back.
    #[command(hide = true)]
    OccupyTime {
        #[arg(long)]
        device: String,
        #[arg(long)]
        pressure_store: memory::PressureStore,
    },
    /// Next-token distributions at every scored position.
    Logits {
        #[command(flatten)]
        run: RunArgs,
        #[command(flatten)]
        engine: EngineArgs,
        /// Size each fixture to this many tokens.
        #[arg(long)]
        context: Option<usize>,
        /// Measure these exact tokens (rendered-fixture JSON Lines, from
        /// another build's run) instead of rendering: they must be rendered
        /// from `--fixtures` for this model and target.
        #[arg(long, value_name = "FILE", conflicts_with_all = ["context", "request_options"])]
        rendered: Option<PathBuf>,
        /// Fixtures decoded together.
        #[arg(long, default_value_t = 8)]
        batch: usize,
        /// Scored rows per teacher-forced forward; above one they run as one
        /// verification forward.
        #[arg(long, default_value_t = 1)]
        verify_width: usize,
        /// Passes over the fixtures on the one load, each past the first
        /// written beside the first's logits as `logits-<pass>.f32`.
        #[arg(long, default_value_t = 1)]
        passes: usize,
    },
    /// The model's Prepare alone: its tuning search, nothing imported.
    Prepare {
        #[command(flatten)]
        run: RunArgs,
        #[command(flatten)]
        engine: EngineArgs,
        /// Context length the load would serve: the tuning points' histories.
        #[arg(long)]
        context: usize,
        /// Fail when a unit's default configuration spills on this device,
        /// listing each spilling launch.
        #[arg(long)]
        check_default_spills: bool,
    },
    /// The catalog drafter's proposals at anchors in the scored spans.
    Draft {
        #[command(flatten)]
        run: RunArgs,
        #[command(flatten)]
        engine: EngineArgs,
        /// Size each fixture to this many tokens.
        #[arg(long)]
        context: Option<usize>,
        /// Anchors per fixture, spread over its scored positions.
        #[arg(long, default_value_t = 8)]
        anchors: usize,
        /// Proposals per draft; the drafter's own width when absent.
        #[arg(long)]
        proposals: Option<u8>,
        /// Time each anchor's draft, verification and a plain decode step
        /// on the device (a diagnostic).
        #[arg(long)]
        attribute: bool,
    },
}

#[derive(Args)]
struct Common {
    /// The catalog acquisition object of the model (JSON).
    #[arg(long, value_name = "FILE")]
    artifacts: PathBuf,
    /// The fixture set (JSON Lines).
    #[arg(long, value_name = "FILE")]
    fixtures: PathBuf,
    /// Chat-completions request fields every rendered request carries, as
    /// the serving benchmark sends them (a JSON object); by default
    /// `{"chat_template_kwargs": {"enable_thinking": false}}`.
    #[arg(long, value_name = "JSON")]
    request_options: Option<String>,
}

impl Common {
    fn request_options(&self) -> Result<RequestOptions, String> {
        self.request_options.as_deref().map_or_else(
            || Ok(render::standard_request_options()),
            render::request_options,
        )
    }
}

/// What every subcommand that writes a run takes.
#[derive(Args)]
struct RunArgs {
    #[command(flatten)]
    common: Common,
    /// The run directory.
    #[arg(long, value_name = "DIR")]
    output: PathBuf,
    /// The build identity (`{"commit", "dirty_digest"}`) from the evaluation
    /// remote package; the record's build is `null` without it.
    #[arg(long, value_name = "FILE")]
    build_identity: Option<PathBuf>,
}

/// The model's host artifacts, opened for rendering as the target serves.
fn host(artifacts: &ArtifactsFile) -> Result<HostArtifacts, String> {
    HostArtifacts::open(&PackageOptions {
        target: artifacts.artifacts.target.clone(),
        projector: ProjectorSelection::Disabled,
        draft: None,
    })
    .map_err(|error| error.to_string())
}

#[derive(Args)]
struct EngineArgs {
    /// The engine's kernel cache (formed kernels and tuning results).
    #[arg(long, value_name = "DIR")]
    cache_dir: Option<PathBuf>,
    /// auto, metal, cuda, vulkan, cpu or an exact device selector.
    #[arg(long, default_value = "auto")]
    device: String,
    /// The target history codec.
    #[arg(long, default_value = "affine-k8v4")]
    kv_codec: String,
    /// Explicitly approved complete numerical computations (JSON).
    #[arg(long, value_name = "FILE")]
    numerical_profile: Option<PathBuf>,
    /// Seconds the first load of a model spends tuning.
    #[arg(long, value_name = "SECONDS", default_value_t = STANDARD_TUNING_BUDGET.as_secs_f64())]
    tuning_budget: f64,
    /// Tune the run's first load afresh, ignoring stored tuning results.
    #[arg(long)]
    fresh_tuning: bool,
    /// A ceiling on every weight byte a load holds: past it, the routed
    /// experts stream through an expert bank.
    #[arg(long, value_name = "BYTES")]
    weight_cap: Option<u64>,
    #[command(flatten)]
    pin: development::PinArgs,
    #[command(flatten)]
    survey: development::SurveyArgs,
}

fn on_off(value: &str) -> Result<bool, String> {
    match value {
        "on" => Ok(true),
        "off" => Ok(false),
        other => Err(format!("takes on or off, not {other}")),
    }
}

fn main() {
    // The measurement job's device probes run in processes of this executable.
    seismic::serve_probe_if_asked();
    magnitude_executor::diagnostics::install();
    magnitude_executor::platform::relax_gpu_watchdog();
    if let Err(error) = run(Cli::parse().command) {
        eprintln!("magnitude-harness: {error}");
        std::process::exit(1);
    }
}

/// What every subcommand reads before it runs.
struct Inputs {
    run: Run,
    build: Value,
    artifacts: ArtifactsFile,
    set: FixtureSet,
    request_options: RequestOptions,
}

impl Inputs {
    fn read(args: &RunArgs, tool: &'static str) -> Result<Self, String> {
        let build = record::build(args.build_identity.as_deref())?;
        let request_options = args.common.request_options()?;
        let artifacts = ArtifactsFile::read(&args.common.artifacts)?;
        let set = FixtureSet::read(&args.common.fixtures)?;
        let run = Run::create(args.output.clone(), tool)?;
        Ok(Self {
            run,
            build,
            artifacts,
            set,
            request_options,
        })
    }

    /// Render the set for the model's target, as it serves.
    fn render(&self, context: Option<usize>) -> Result<Vec<Rendered>, String> {
        let host = host(&self.artifacts)?;
        let renderer = Renderer::new(
            &host,
            &self.artifacts.artifacts.model,
            &self.request_options,
        );
        self.set
            .fixtures
            .iter()
            .map(|fixture| renderer.render(fixture, context))
            .collect()
    }

    /// Write `rendered.jsonl` and return its digest.
    fn write_rendered(&self, rendered: &[Rendered]) -> Result<String, String> {
        render::write_rendered(
            &self.run.path("rendered.jsonl"),
            rendered,
            &self.set.digest,
            &self.artifacts.artifacts.model,
            &self.artifacts.artifacts.target_sha256,
        )
    }

    fn finish(
        self,
        rendered_digest: String,
        devices: Option<Vec<String>>,
        measurements: Map<String, Value>,
        traces: Map<String, Value>,
    ) -> Result<(), String> {
        self.run.finish(RunRecord {
            build: self.build,
            model: self.artifacts.artifacts.model.clone(),
            artifacts: self.artifacts.raw,
            fixture_set: (self.set.path, self.set.digest),
            rendered_digest,
            host: record::host(devices),
            measurements,
            traces,
        })
    }
}

/// The engine options of a run whose tuning trace is in `inputs`' run
/// directory, with the development tools installed.
fn engine_options(
    engine: &EngineArgs,
    inputs: &Inputs,
) -> Result<(EngineOptions, Development), String> {
    let budget = Duration::try_from_secs_f64(engine.tuning_budget)
        .ok()
        .filter(|budget| !budget.is_zero())
        .ok_or("--tuning-budget takes a positive number of seconds")?;
    let numerical_profile = engine
        .numerical_profile
        .as_ref()
        .map(|path| {
            let bytes = std::fs::read(path)
                .map_err(|error| format!("numerical profile {}: {error}", path.display()))?;
            serde_json::from_slice(&bytes)
                .map_err(|error| format!("numerical profile {}: {error}", path.display()))
        })
        .transpose()?;
    // A survey measures: it is installed in the worker process that runs a
    // model's Prepare, never here, where the model is loaded.
    let worker_arguments = engine.survey.worker_arguments();
    let development = Development {
        pin: development::Pin::install(&engine.pin, &inputs.run.path("selection.json"))?,
        survey: engine.survey.clone(),
    };
    Ok((
        EngineOptions {
            cache_dir: engine.cache_dir.clone(),
            device: engine
                .device
                .parse()
                .map_err(|error| format!("--device: {error}"))?,
            kv_codec: engine
                .kv_codec
                .parse()
                .map_err(|error| format!("--kv-codec: {error}"))?,
            numerical_profile,
            weight_cap: engine.weight_cap,
            budget,
            fresh: Cell::new(engine.fresh_tuning),
            tuning_trace: inputs.run.path(tuning::TRACE),
            worker_arguments,
        },
        development,
    ))
}

/// The installed development tools, finished after the run.
struct Development {
    pin: development::Pin,
    survey: development::SurveyArgs,
}

impl Development {
    fn finish(self) -> Result<(), String> {
        self.pin.finish()?;
        self.survey.report()
    }
}

/// The measurements every engine run records: its tuning.
fn tuned(
    engine: &EngineArgs,
    options: &EngineOptions,
) -> Result<(Map<String, Value>, Map<String, Value>), String> {
    let mut measurements = Map::new();
    measurements.insert(
        "tuning".into(),
        tuning::measurement(
            &options.tuning_trace,
            options.budget.as_secs_f64(),
            engine.fresh_tuning,
        )?,
    );
    let mut traces = Map::new();
    traces.insert("tuning".into(), json!(tuning::TRACE));
    Ok((measurements, traces))
}

fn run(command: Command) -> Result<(), String> {
    // The memory step's decisions (each growth step offered, each resize
    // finished) print as a run takes them. The memory instrument reads each streamed
    // step's reads from the worker; elsewhere a step's reads are its result's.
    memory::print_events(match command {
        Command::Worker { .. } => memory::Scope::Steps,
        _ => memory::Scope::Decisions,
    })?;
    match command {
        Command::Worker { survey } => {
            use magnitude_engine::worker::{
                WorkerExit, serve_worker, transport::stdio_worker_transport,
            };
            survey.install()?;
            match serve_worker(stdio_worker_transport()) {
                WorkerExit::Prepared | WorkerExit::Shutdown => Ok(()),
                exit => Err(format!("worker ended: {exit:?}")),
            }
        }
        Command::Render { run, context } => {
            let inputs = Inputs::read(&run, "harness.render")?;
            let rendered = inputs.render(context)?;
            let digest = inputs.write_rendered(&rendered)?;
            inputs.finish(digest, None, Map::new(), Map::new())
        }
        Command::Time {
            run,
            engine,
            contexts,
            concurrency,
            warm,
            steps,
            attribution_steps,
            prefill_attribution,
            prefill_samples,
            lookahead,
            resident_decode,
            untimed,
        } => {
            if contexts.is_empty()
                || concurrency.is_empty()
                || concurrency.contains(&0)
                || steps < 2
            {
                return Err(
                    "time needs contexts, positive concurrency and at least two measured steps"
                        .into(),
                );
            }
            if prefill_samples < 2 {
                return Err("--prefill-samples needs a cold and a warm sample".into());
            }
            let inputs = Inputs::read(&run, "harness.time")?;
            let (options, development) = engine_options(&engine, &inputs)?;
            let host = host(&inputs.artifacts)?;
            let renderer = Renderer::new(
                &host,
                &inputs.artifacts.artifacts.model,
                &inputs.request_options,
            );
            let timed = time::run(
                &inputs.artifacts.artifacts,
                &inputs.set,
                &renderer,
                &options,
                &time::TimeOptions {
                    contexts,
                    concurrency,
                    warm,
                    steps,
                    attribution_steps,
                    prefill_attribution,
                    prefill_samples,
                    lookahead,
                    resident_decode,
                            untimed,
                },
            )?;
            development.finish()?;
            let digest = inputs.write_rendered(&timed.rendered)?;
            let (mut measurements, traces) = tuned(&engine, &options)?;
            measurements.insert("speed".into(), Value::from(timed.speed));
            measurements.insert("load".into(), timed.load);
            inputs.finish(digest, Some(timed.devices), measurements, traces)
        }
        Command::Prepare {
            run,
            engine,
            context,
            check_default_spills,
        } => {
            let inputs = Inputs::read(&run, "harness.prepare")?;
            let rendered = inputs.render(Some(context))?;
            let digest = inputs.write_rendered(&rendered)?;
            let (options, development) = engine_options(&engine, &inputs)?;
            engine::prepare(
                &inputs.artifacts.artifacts,
                &options,
                &engine::Load {
                    drafter: false,
                    proposals: None,
                    context_tokens: context,
                    sequences: std::num::NonZeroUsize::MIN,
                    decode_rows: 1,
                    exported_logits_rows: 0,
                    lookahead: true,
                },
            )?;
            development.finish()?;
            let (measurements, traces) = tuned(&engine, &options)?;
            inputs.finish(digest, None, measurements, traces)?;
            if check_default_spills {
                let spills = tuning::default_spills(&options.tuning_trace)?;
                if !spills.is_empty() {
                    return Err(format!("default configurations spill:\n{}", spills.join("\n")));
                }
            }
            Ok(())
        }
        Command::Logits {
            run,
            engine,
            context,
            rendered,
            batch,
            verify_width,
            passes,
        } => {
            if batch == 0 || verify_width == 0 || passes == 0 {
                return Err("--batch, --verify-width and --passes must be positive".into());
            }
            let inputs = Inputs::read(&run, "harness.logits")?;
            let (rendered, digest) = match rendered {
                Some(path) => {
                    let file = render::RenderedFile::read(
                        &path,
                        &inputs.set.digest,
                        &inputs.artifacts.artifacts.model,
                        &inputs.artifacts.artifacts.target_sha256,
                    )?;
                    // The run keeps the exact bytes its rows refer to.
                    let copy = inputs.run.path("rendered.jsonl");
                    std::fs::write(&copy, &file.bytes)
                        .map_err(|error| format!("writing {}: {error}", copy.display()))?;
                    (file.rendered, file.digest)
                }
                None => {
                    let rendered = inputs.render(context)?;
                    let digest = inputs.write_rendered(&rendered)?;
                    (rendered, digest)
                }
            };
            let (options, development) = engine_options(&engine, &inputs)?;
            let device = logits::run(
                &inputs.artifacts.artifacts,
                &options,
                &logits::LogitsOptions {
                    batch,
                    verify_width,
                    passes,
                },
                &rendered,
                &digest,
                &inputs.run.path("logits"),
            )?;
            development.finish()?;
            let (measurements, traces) = tuned(&engine, &options)?;
            inputs.finish(digest, Some(vec![device]), measurements, traces)
        }
        Command::Memory {
            device,
            context,
            weight_cap,
            decode,
            decode_tokens,
            decode_prompt_tokens,
            pressure,
            pressure_prompt,
            pressure_store,
            hold,
            cache_dir,
            target,
        } => memory::load(memory::Load {
            backend: memory::backend(&device)?,
            target,
            kernel_cache: cache_dir,
            context_tokens: context,
            weight_cap,
            decode: decode.zip(decode_tokens).map(|(rendered, tokens)| memory::Decode {
                rendered,
                tokens,
                prompt_tokens: decode_prompt_tokens,
            }),
            // Clap requires each of `--pressure` and `--pressure-store` with the other.
            pressure: pressure.zip(pressure_store).map(|(deficit, store)| memory::Pressing {
                store,
                deficit,
                from_prompt: pressure_prompt,
            }),
            hold,
        }),
        Command::Recommit { device, gigabytes } => {
            let commitments = gigabytes
                .iter()
                .map(|gigabytes| {
                    gigabytes
                        .checked_mul(1_000_000_000)
                        .ok_or_else(|| format!("--gigabytes {gigabytes} overflows"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            memory::recommit(memory::backend(&device)?, &commitments)
        }
        Command::Occupy { device, deficit, pressure_store } => memory::occupy(memory::backend(&device)?, pressure_store, deficit),
        Command::OccupyTime { device, pressure_store } => memory::time_occupying(memory::backend(&device)?, pressure_store),
        Command::Draft {
            run,
            engine,
            context,
            anchors,
            proposals,
            attribute,
        } => {
            let inputs = Inputs::read(&run, "harness.draft")?;
            let rendered = inputs.render(context)?;
            let digest = inputs.write_rendered(&rendered)?;
            let (options, development) = engine_options(&engine, &inputs)?;
            let (drafting, device) = draft::run(
                &inputs.artifacts.artifacts,
                &options,
                &draft::DraftOptions {
                    anchors,
                    proposals,
                    attribute,
                },
                &rendered,
                &inputs.run.path("draft.jsonl"),
            )?;
            development.finish()?;
            let (mut measurements, traces) = tuned(&engine, &options)?;
            measurements.insert("drafting".into(), drafting);
            inputs.finish(digest, Some(vec![device]), measurements, traces)
        }
    }
}
