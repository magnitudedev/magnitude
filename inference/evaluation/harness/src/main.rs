//! `magnitude-harness`: the only in-process driver of the engine for
//! evaluation. Every engine subcommand takes a catalog model (its acquired
//! paths, `--artifacts`) and a fixture set, renders the fixtures through the
//! engine's own serving chat path, and writes an evaluation result record
//! (`result.json`, `specs/26-10-08/evaluation-contracts.md`) with its traces
//! into a run directory.
//!
//! - `render`: the exact tokens of each fixture (`rendered.jsonl`), or with
//!   `--stdin` one rendered fixture per request line, for sizing searches.
//! - `time`: prefill and decode throughput at context lengths and
//!   concurrency, and drafting acceptance with the catalog drafter.
//! - `logits`: the distributions at every scored position (`logits.f32`,
//!   `logits.json`).
//! - `draft`: the catalog drafter's proposals at anchors in the scored spans
//!   (`draft.jsonl`).
//! - `precision-probe` (development feature): the added KL of a pinned tuning
//!   selection alongside dimensionless precision measurements from its trace
//!   (`precision_probe.json`).
//!
//! Every request is rendered with `--request-options`, the request fields the
//! serving benchmark sends (reasoning disabled by default), recorded in each
//! rendered fixture. Every load is the engine's sequence: the load, in this
//! process, which never searches; and, when it finds tuning units without a
//! stored record, first the model's Prepare in a worker process of this
//! executable, which searches them and exits. Each appends its tuning trace
//! to `tuning.jsonl`.

mod artifacts;
mod development;
mod digest;
mod draft;
mod engine;
mod fixture;
mod logits;
#[cfg(feature = "precision-probe")]
mod probe;
mod record;
mod render;
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
        #[cfg(feature = "tuning-survey")]
        #[command(flatten)]
        survey: development::SurveyArgs,
    },
    /// Render fixtures to the exact tokens the engine serves.
    Render {
        #[command(flatten)]
        common: Common,
        /// Size each fixture to this many tokens.
        #[arg(long)]
        context: Option<usize>,
        /// Render on request instead of writing a run: each stdin line is
        /// `{"fixture_id": ID, "context": N | null}`, answered by one stdout
        /// line, the rendered fixture or `{"error": MESSAGE}`.
        #[arg(long, conflicts_with_all = ["context", "output", "build_identity"])]
        stdin: bool,
    },
    /// Prefill and decode throughput, and drafting acceptance.
    Time {
        #[command(flatten)]
        common: Common,
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
        /// timed alone (a diagnostic; a step's launches must fit the device's
        /// timestamp buffer).
        #[arg(long, default_value_t = 0)]
        attribution_steps: usize,
        /// Prefill samples, the first of them cold.
        #[arg(long, default_value_t = 4)]
        prefill_samples: usize,
        /// Queue each decode step's successor before it completes, as served.
        #[arg(long, default_value = "on", value_parser = on_off, action = clap::ArgAction::Set)]
        lookahead: bool,
        /// Skip drafting acceptance even when the catalog declares a drafter.
        #[arg(long)]
        no_drafting: bool,
    },
    /// Next-token distributions at every scored position.
    Logits {
        #[command(flatten)]
        common: Common,
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
    },
    /// The catalog drafter's proposals at anchors in the scored spans.
    Draft {
        #[command(flatten)]
        common: Common,
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
    /// The added KL of a pinned tuning selection against the defaults (and
    /// references), alongside the trace's dimensionless precision measurements.
    #[cfg(feature = "precision-probe")]
    PrecisionProbe {
        #[command(flatten)]
        common: Common,
        #[command(flatten)]
        engine: EngineArgs,
        #[command(flatten)]
        probe: probe::ProbeArgs,
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
    /// The run directory; a new one under evaluation/results by default.
    #[arg(long, value_name = "DIR")]
    output: Option<PathBuf>,
    /// The build identity (`{"commit", "dirty_digest"}`) from the evaluation
    /// remote package; the record's build is `null` without it.
    #[arg(long, value_name = "FILE")]
    build_identity: Option<PathBuf>,
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

/// The model's host artifacts, opened for rendering as the target serves.
fn host(artifacts: &ArtifactsFile) -> Result<HostArtifacts, String> {
    HostArtifacts::open(&PackageOptions {
        target: artifacts.artifacts.target.clone(),
        projector: ProjectorSelection::Disabled,
        draft: None,
    })
    .map_err(|error| error.to_string())
}

/// `render --stdin`: answer render requests over stdin until it closes.
fn serve_render(common: &Common) -> Result<(), String> {
    use std::io::{BufRead, Write};
    let options = common.request_options()?;
    let artifacts = ArtifactsFile::read(&common.artifacts)?;
    let set = FixtureSet::read(&common.fixtures)?;
    let host = host(&artifacts)?;
    let renderer = Renderer::new(&host, &artifacts.artifacts.model, &options);
    let render = |line: &str| -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Request {
            fixture_id: String,
            context: Option<usize>,
        }
        let request: Request =
            serde_json::from_str(line).map_err(|error| format!("render request: {error}"))?;
        let fixture = set
            .fixtures
            .iter()
            .find(|fixture| fixture.id == request.fixture_id)
            .ok_or_else(|| format!("no fixture {} in the set", request.fixture_id))?;
        let rendered = renderer.render(fixture, request.context)?;
        serde_json::to_value(rendered.line(
            &set.digest,
            &artifacts.artifacts.model,
            &artifacts.artifacts.target_sha256,
        ))
        .map_err(|error| error.to_string())
    };
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for line in std::io::stdin().lock().lines() {
        let line = line.map_err(|error| format!("reading stdin: {error}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let answer = render(&line).unwrap_or_else(|error| json!({ "error": error }));
        serde_json::to_writer(&mut out, &answer)
            .map_err(|error| error.to_string())
            .and_then(|()| out.write_all(b"\n").map_err(|error| error.to_string()))
            .and_then(|()| out.flush().map_err(|error| error.to_string()))?;
    }
    Ok(())
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
    #[cfg(feature = "pinned-tuning")]
    #[command(flatten)]
    pin: development::PinArgs,
    #[cfg(feature = "tuning-survey")]
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
    fn read(common: &Common, tool: &'static str) -> Result<Self, String> {
        let build = record::build(common.build_identity.as_deref())?;
        let request_options = common.request_options()?;
        let artifacts = ArtifactsFile::read(&common.artifacts)?;
        let set = FixtureSet::read(&common.fixtures)?;
        let run = Run::create(common.output.clone(), tool)?;
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
    #[cfg(feature = "tuning-survey")]
    let worker_arguments = engine.survey.worker_arguments();
    #[cfg(not(feature = "tuning-survey"))]
    let worker_arguments = Vec::new();
    let development = Development {
        #[cfg(feature = "pinned-tuning")]
        pin: development::Pin::install(&engine.pin)?,
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
    #[cfg(feature = "pinned-tuning")]
    pin: development::Pin,
}

impl Development {
    fn finish(self) -> Result<(), String> {
        #[cfg(feature = "pinned-tuning")]
        self.pin.finish()?;
        Ok(())
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
    match command {
        Command::Worker {
            #[cfg(feature = "tuning-survey")]
            survey,
        } => {
            use magnitude_engine::worker::{
                WorkerExit, serve_worker, transport::stdio_worker_transport,
            };
            #[cfg(feature = "tuning-survey")]
            survey.install()?;
            match serve_worker(stdio_worker_transport()) {
                WorkerExit::Prepared | WorkerExit::Shutdown => Ok(()),
                exit => Err(format!("worker ended: {exit:?}")),
            }
        }
        #[cfg(feature = "precision-probe")]
        Command::PrecisionProbe {
            common,
            engine,
            probe,
        } => {
            if engine.pin.tuning_record.is_some()
                || engine.pin.tuning_replay.is_some()
                || engine.pin.tuning_defaults
            {
                return Err("the precision probe pins its own selections".into());
            }
            let inputs = Inputs::read(&common, "harness.precision_probe")?;
            let (options, development) = engine_options(&engine, &inputs)?;
            let measurement = probe::run(&inputs, &options, &probe)?;
            development.finish()?;
            let (mut measurements, traces) = tuned(&engine, &options)?;
            measurements.insert("precision_probe".into(), measurement.result);
            inputs.finish(
                measurement.rendered_digest,
                Some(vec![measurement.device]),
                measurements,
                traces,
            )
        }
        Command::Render {
            common,
            stdin: true,
            ..
        } => serve_render(&common),
        Command::Render {
            common, context, ..
        } => {
            let inputs = Inputs::read(&common, "harness.render")?;
            let rendered = inputs.render(context)?;
            let digest = inputs.write_rendered(&rendered)?;
            inputs.finish(digest, None, Map::new(), Map::new())
        }
        Command::Time {
            common,
            engine,
            contexts,
            concurrency,
            warm,
            steps,
            attribution_steps,
            prefill_samples,
            lookahead,
            no_drafting,
        } => {
            if contexts.is_empty()
                || concurrency.is_empty()
                || concurrency.contains(&0)
                || steps == 0
            {
                return Err("time needs contexts, positive concurrency and measured steps".into());
            }
            if prefill_samples < 2 {
                return Err("--prefill-samples needs a cold and a warm sample".into());
            }
            let inputs = Inputs::read(&common, "harness.time")?;
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
                    prefill_samples,
                    lookahead,
                    drafting: !no_drafting,
                },
            )?;
            development.finish()?;
            let digest = inputs.write_rendered(&timed.rendered)?;
            let (mut measurements, traces) = tuned(&engine, &options)?;
            measurements.insert("speed".into(), Value::from(timed.speed));
            if let Some(drafting) = timed.drafting {
                measurements.insert("drafting".into(), drafting);
            }
            inputs.finish(digest, Some(timed.devices), measurements, traces)
        }
        Command::Logits {
            common,
            engine,
            context,
            rendered,
            batch,
            verify_width,
        } => {
            if batch == 0 || verify_width == 0 {
                return Err("--batch and --verify-width must be positive".into());
            }
            let inputs = Inputs::read(&common, "harness.logits")?;
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
                },
                &rendered,
                &digest,
                &inputs.run.path("logits"),
            )?;
            development.finish()?;
            let (measurements, traces) = tuned(&engine, &options)?;
            inputs.finish(digest, Some(vec![device]), measurements, traces)
        }
        Command::Draft {
            common,
            engine,
            context,
            anchors,
            proposals,
            attribute,
        } => {
            let inputs = Inputs::read(&common, "harness.draft")?;
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
