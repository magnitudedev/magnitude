//! `time`: prefill and decode throughput at context lengths and concurrency,
//! on fixtures sized to each context, plus drafting acceptance from the
//! engine's own counters when the catalog declares a drafter.
//!
//! Cells, at each context N and concurrency C:
//! - `prefill`: C fresh sequences take their fixtures' tokens in prefill
//!   steps of at most `PREFILL_ROWS` rows packed across sequences, as the
//!   service bounds them; one cold sample, then warm samples (median). A
//!   full prefill step launches more kernels than a device's timestamp
//!   buffer can time one by one, so prefill is not attributed by entry;
//! - `decode`: C sequences hold their fixtures' tokens as history, then
//!   decode together, each step feeding back its greedy selections: warm
//!   steps, then measured steps (median).
//!
//! Every measured step is traced per device submission: host time before
//! the first encode, host encode time, device busy and idle within the step,
//! device idle between steps, and host time after the device finished.
//! With `--attribution-steps`, further decode steps run each launch in its
//! own timed unit and decompose device time by entry; they are slower by
//! construction and are not the cell's timing.

use crate::artifacts::Artifacts;
use crate::engine::{self, Engine, EngineOptions, Load, PREFILL_ROWS, Sequence, Step};
use crate::fixture::FixtureSet;
use crate::render::{Rendered, Renderer};
use crate::steps::{attribution, device_end, median, medians, summary, window};
use magnitude_engine::composition::ReadyEngine;
use magnitude_engine::options::ResolvedMethod;
use magnitude_engine::worker::{RequestEvent, RequestOptions};
use magnitude_executor::{Demand, Operation, WorkKind};
use magnitude_generation::{EndOfGeneration, MethodChoice, Options, Sampling, Shaping};
use magnitude_scheduler::prefix_cache::PrefixRetention;
use seismic::{SubmissionTrace, TraceDetail, host_seconds};
use serde_json::{Value, json};
use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

const MS: f64 = 1e3;

pub struct TimeOptions {
    pub contexts: Vec<usize>,
    pub concurrency: Vec<usize>,
    pub warm: usize,
    pub steps: usize,
    pub attribution_steps: usize,
    /// Prefill samples, the first of them cold.
    pub prefill_samples: usize,
    pub lookahead: bool,
    pub drafting: bool,
}

pub struct Timed {
    pub rendered: Vec<Rendered>,
    pub speed: Vec<Value>,
    pub drafting: Option<Value>,
    pub devices: Vec<String>,
}

pub fn run(
    artifacts: &Artifacts,
    set: &FixtureSet,
    renderer: &Renderer<'_>,
    options: &EngineOptions,
    time: &TimeOptions,
) -> Result<Timed, String> {
    let mut by_context = Vec::with_capacity(time.contexts.len());
    for &context in &time.contexts {
        let rendered = set
            .fixtures
            .iter()
            .map(|fixture| renderer.render(fixture, Some(context)))
            .collect::<Result<Vec<_>, _>>()?;
        by_context.push((context, rendered));
    }
    let drafting = time.drafting && artifacts.drafter.is_some();

    let longest = by_context
        .iter()
        .flat_map(|(_, rendered)| rendered.iter().map(|fixture| fixture.tokens.len()))
        .max()
        .expect("a fixture set is nonempty");
    let widest = time.concurrency.iter().copied().max().unwrap_or(1);
    let mut engine = Engine::open(
        artifacts,
        options,
        &Load {
            drafter: false,
            proposals: None,
            context_tokens: longest + time.warm + time.steps + time.attribution_steps + 1,
            decode_rows: widest,
            exported_logits_rows: 0,
            lookahead: time.lookahead,
        },
    )?;
    let devices = vec![engine.device().info().name.clone()];
    let mut speed = Vec::new();
    for (context, rendered) in &by_context {
        for &concurrency in &time.concurrency {
            let histories = (0..concurrency)
                .map(|index| &rendered[index % rendered.len()])
                .collect::<Vec<_>>();
            eprintln!("magnitude-harness: prefill context={context} concurrency={concurrency}");
            let cell = prefill_cell(&mut engine, time, &histories)?;
            eprintln!(
                "  {:.1} tok/s",
                cell["tokens_per_second"].as_f64().unwrap_or(f64::NAN)
            );
            speed.push(with_cell(
                cell,
                "prefill",
                *context,
                concurrency,
                &histories,
            ));
            eprintln!("magnitude-harness: decode context={context} concurrency={concurrency}");
            let cell = decode_cell(&mut engine, time, &histories)?;
            eprintln!(
                "  {:.1} tok/s",
                cell["tokens_per_second"].as_f64().unwrap_or(f64::NAN)
            );
            speed.push(with_cell(cell, "decode", *context, concurrency, &histories));
        }
    }
    drop(engine);
    // Drafting serves each fixture's generation request as sized to the
    // first context.
    let drafting = if drafting {
        let (context, rendered) = &by_context[0];
        let prompts = set
            .fixtures
            .iter()
            .zip(rendered)
            .filter_map(|(fixture, rendered)| {
                fixture
                    .generation
                    .as_ref()
                    .map(|generation| (rendered, generation.max_tokens))
            })
            .collect::<Vec<_>>();
        let mut drafting = drafting_acceptance(artifacts, options, &prompts)?;
        drafting["context"] = json!(context);
        Some(drafting)
    } else {
        None
    };
    let rendered = by_context
        .into_iter()
        .flat_map(|(_, rendered)| rendered)
        .collect::<Vec<_>>();
    Ok(Timed {
        rendered,
        speed,
        drafting,
        devices,
    })
}

fn with_cell(
    mut cell: Value,
    phase: &str,
    context: usize,
    concurrency: usize,
    histories: &[&Rendered],
) -> Value {
    cell["phase"] = json!(phase);
    cell["context"] = json!(context);
    cell["concurrency"] = json!(concurrency);
    cell["fixtures"] = json!(
        histories
            .iter()
            .map(|history| &history.fixture_id)
            .collect::<Vec<_>>()
    );
    cell["history_tokens"] = json!(
        histories
            .iter()
            .map(|history| history.tokens.len())
            .collect::<Vec<_>>()
    );
    cell
}

fn traced(engine: &Engine, detail: TraceDetail) -> Result<SubmissionTrace, String> {
    engine
        .device()
        .trace_submissions(detail)
        .map_err(|error| error.to_string())
}

/// One prefill sample: fresh sequences take their histories in steps packed
/// up to `PREFILL_ROWS` rows; each sequence's last chunk selects.
fn prefill_sample(
    engine: &mut Engine,
    histories: &[&Rendered],
    trace: &SubmissionTrace,
) -> Result<(f64, Vec<Step>), String> {
    let mut sequences = histories
        .iter()
        .map(|_| engine.open_sequence())
        .collect::<Result<Vec<_>, _>>()?;
    let tokens = histories
        .iter()
        .map(|history| engine::tokens(&history.tokens))
        .collect::<Vec<_>>();
    let start = host_seconds();
    let mut steps = Vec::new();
    while sequences
        .iter()
        .zip(&tokens)
        .any(|(sequence, tokens)| sequence.position < tokens.len())
    {
        let mut budget = PREFILL_ROWS;
        let mut operations = Vec::new();
        let mut advancing = Vec::new();
        for (sequence, tokens) in sequences.iter_mut().zip(&tokens) {
            let rows = (tokens.len() - sequence.position).min(PREFILL_ROWS);
            if rows == 0 || rows > budget {
                continue;
            }
            budget -= rows;
            let finishing = sequence.position + rows == tokens.len();
            let piece = tokens[sequence.position..sequence.position + rows].to_vec();
            let demand = if finishing {
                Demand::SELECT
            } else {
                Demand::NONE
            };
            operations.push(Engine::forward(sequence, WorkKind::Prefill, piece, demand));
            advancing.push(sequence);
        }
        let (step, _) = engine.step(operations, &mut advancing, Some(trace))?;
        steps.push(step);
    }
    let seconds = host_seconds() - start;
    for sequence in sequences {
        engine.close(sequence)?;
    }
    Ok((seconds, steps))
}

fn prefill_cell(
    engine: &mut Engine,
    time: &TimeOptions,
    histories: &[&Rendered],
) -> Result<Value, String> {
    let rows = histories
        .iter()
        .map(|history| history.tokens.len())
        .sum::<usize>() as f64;
    let trace = traced(engine, TraceDetail::Submissions)?;
    let mut samples_ms = Vec::with_capacity(time.prefill_samples);
    let mut summaries = Vec::new();
    for sample in 0..time.prefill_samples {
        let (seconds, steps) = prefill_sample(engine, histories, &trace)?;
        samples_ms.push(seconds * MS);
        if sample > 0 {
            for step in &steps {
                summaries.push(summary(step, None)?);
            }
        }
    }
    drop(trace);
    let mut warm = samples_ms[1..].to_vec();
    let median_ms = median(&mut warm);
    Ok(json!({
        "tokens_per_second": rows / (median_ms / MS),
        "samples": samples_ms[1..].iter().map(|ms| rows / (ms / MS)).collect::<Vec<_>>(),
        "detail": {
            "rows": rows,
            "median_ms": median_ms,
            "cold_ms": samples_ms[0],
            "samples_ms": &samples_ms[1..],
            "median_step": medians(&summaries),
        },
    }))
}

fn decode_cell(
    engine: &mut Engine,
    time: &TimeOptions,
    histories: &[&Rendered],
) -> Result<Value, String> {
    let mut sequences = Vec::with_capacity(histories.len());
    for history in histories {
        let mut sequence = engine.open_sequence()?;
        engine.prefill(&mut sequence, &engine::tokens(&history.tokens), true)?;
        sequences.push(sequence);
    }
    let decode = |sequences: &[Sequence]| -> Result<Vec<Operation>, String> {
        sequences
            .iter()
            .map(|sequence| {
                let token = sequence.selected.ok_or("a decode follows a selection")?;
                Ok(Engine::forward(
                    sequence,
                    WorkKind::Decode,
                    vec![token],
                    Demand::SELECT,
                ))
            })
            .collect()
    };
    let detail = measure(engine, &mut sequences, time, &decode)?;
    let step_ms = detail["median_ms"].as_f64().expect("a median");
    let concurrency = sequences.len() as f64;
    let samples = detail["steps"]
        .as_array()
        .expect("steps")
        .iter()
        .map(|step| concurrency / (step["wall_ms"].as_f64().expect("wall") / MS))
        .collect::<Vec<_>>();
    for sequence in sequences {
        engine.close(sequence)?;
    }
    Ok(json!({
        "tokens_per_second": concurrency / (step_ms / MS),
        "samples": samples,
        "detail": detail,
    }))
}

/// Measure `time.steps` steps made by `make` after `time.warm`, traced per
/// submission, then attribute `time.attribution_steps` more by entry.
fn measure(
    engine: &mut Engine,
    sequences: &mut [Sequence],
    time: &TimeOptions,
    make: &dyn Fn(&[Sequence]) -> Result<Vec<Operation>, String>,
) -> Result<Value, String> {
    let run = |engine: &mut Engine,
               sequences: &mut [Sequence],
               trace: &SubmissionTrace|
     -> Result<Step, String> {
        let operations = make(sequences)?;
        let mut refs = sequences.iter_mut().collect::<Vec<_>>();
        // Under lookahead the trace is collected once per window: a per-step
        // collect would wait for the step queued behind this one.
        let per_step = (!engine.lookahead).then_some(trace);
        engine
            .step(operations, &mut refs, per_step)
            .map(|(step, _)| step)
    };
    let collect = |trace: &SubmissionTrace| trace.collect().map_err(|error| error.to_string());
    let trace = traced(engine, TraceDetail::Submissions)?;
    for _ in 0..time.warm {
        run(engine, sequences, &trace)?;
    }
    collect(&trace)?;
    let mut summaries = Vec::with_capacity(time.steps);
    let mut previous = None;
    let mut submissions = Vec::new();
    let mut span = (f64::INFINITY, f64::NEG_INFINITY);
    for _ in 0..time.steps {
        let step = run(engine, sequences, &trace)?;
        span = (span.0.min(step.start), span.1.max(step.end));
        if engine.lookahead {
            summaries.push(json!({ "wall_ms": (step.end - step.start) * MS }));
        } else {
            summaries.push(summary(&step, previous)?);
            previous = Some(device_end(&step));
        }
        submissions.extend(step.submissions);
    }
    submissions.extend(collect(&trace)?);
    drop(trace);
    let attributed = if time.attribution_steps == 0 {
        Value::Null
    } else {
        let trace = traced(engine, TraceDetail::Launches)?;
        let mut attributed = Vec::new();
        let mut attributed_end = f64::NEG_INFINITY;
        for _ in 0..time.attribution_steps {
            // Attributed steps are untimed, so each collects its own launches
            // even under lookahead: a step's launch timestamps can fit the
            // trace's buffer, a window's do not.
            let operations = make(sequences)?;
            let mut refs = sequences.iter_mut().collect::<Vec<_>>();
            let (step, _) = engine.step(operations, &mut refs, Some(&trace))?;
            attributed_end = step.end;
            attributed.extend(step.submissions);
        }
        attributed.extend(collect(&trace)?);
        attribution(&attributed, time.attribution_steps, attributed_end)
    };
    let mut wall = summaries
        .iter()
        .map(|summary| summary["wall_ms"].as_f64().expect("wall"))
        .collect::<Vec<_>>();
    Ok(json!({
        "warm_steps": time.warm,
        "lookahead": engine.lookahead,
        "median_ms": median(&mut wall),
        "median_step": medians(&summaries),
        "window": window(time.steps, span, &submissions),
        "steps": summaries,
        "attribution": attributed,
    }))
}

/// Drive a future on this thread; the engine client is executor-agnostic.
fn block_on<T>(future: impl Future<Output = T>) -> T {
    struct Unpark(std::thread::Thread);
    impl Wake for Unpark {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

/// Greedy generation of every fixture's serving request with the catalog
/// drafter, through the engine's worker: proposed and accepted drafts are
/// the generation's own counters. A verification commits its accepted
/// proposals plus one token, after the first token the prompt yields, so
/// verifications are `completion - 1 - accepted`.
fn drafting_acceptance(
    artifacts: &Artifacts,
    options: &EngineOptions,
    prompts: &[(&Rendered, usize)],
) -> Result<Value, String> {
    if prompts.is_empty() {
        return Err("drafting needs fixtures with a generation request".into());
    }
    let longest = prompts
        .iter()
        .map(|(rendered, max_tokens)| {
            rendered
                .generation_prompt
                .as_ref()
                .expect("a generation request")
                .len()
                + max_tokens
        })
        .max()
        .expect("nonempty");
    let resolved = engine::configuration(
        artifacts,
        options,
        &Load {
            drafter: true,
            proposals: None,
            context_tokens: longest,
            decode_rows: magnitude_engine::options::standard_service_limits().decode_tokens,
            exported_logits_rows: 0,
            lookahead: true,
        },
    )?;
    // Each load step once; its measured progress repeats per unit and weight.
    let mut reported = None;
    let engine = engine::start(resolved, options, |progress| {
        let step = std::mem::discriminant(&progress);
        if reported != Some(step) {
            reported = Some(step);
            eprintln!("magnitude-harness: load {progress:?}");
        }
    })?;
    let method = match engine.ready_info().model.method {
        ResolvedMethod::DFlash { proposals } => MethodChoice::DFlash { proposals },
        ResolvedMethod::Mtp {
            greedy_proposals, ..
        } => MethodChoice::Mtp {
            proposals: greedy_proposals,
        },
        ResolvedMethod::Plain => {
            return Err("the catalog drafter did not load a drafting method".into());
        }
    };
    let (mut proposed, mut accepted, mut verifications) = (0usize, 0usize, 0usize);
    let mut requests = Vec::with_capacity(prompts.len());
    for (rendered, max_tokens) in prompts {
        let prompt = rendered
            .generation_prompt
            .as_ref()
            .expect("a generation request");
        let usage = generate(&engine, prompt, *max_tokens, method)?;
        let verified = usage
            .completion_tokens
            .saturating_sub(1 + usage.draft_n_accepted);
        proposed += usage.draft_n;
        accepted += usage.draft_n_accepted;
        verifications += verified;
        requests.push(json!({
            "fixture_id": rendered.fixture_id,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "proposed": usage.draft_n,
            "accepted": usage.draft_n_accepted,
            "verifications": verified,
        }));
    }
    drop(engine);
    Ok(json!({
        "proposed": proposed,
        "accepted": accepted,
        "tokens_per_verify": (accepted + verifications) as f64 / verifications.max(1) as f64,
        "requests": requests,
    }))
}

fn generate(
    engine: &ReadyEngine,
    prompt: &[u32],
    max_tokens: usize,
    method: MethodChoice,
) -> Result<magnitude_generation::DetailedUsage, String> {
    let host = engine.host();
    let input = host
        .prepare_input(
            prompt
                .iter()
                .copied()
                .map(magnitude_generation::TokenId)
                .collect(),
            &[],
        )
        .map_err(|error| error.to_string())?;
    let definition = host.definition();
    let options = Options {
        max_tokens,
        output_capacity: 16,
        context_limit: usize::try_from(definition.decoder.context_limit)
            .map_err(|_| "context exceeds the host domain")?,
        vocabulary: usize::try_from(definition.decoder.vocabulary)
            .map_err(|_| "vocabulary exceeds the host domain")?,
        stop_tokens: host.tokenizer().stop_tokens().clone(),
        suppressed_tokens: host.tokenizer().suppressed_tokens().clone(),
        sampling: Sampling::Greedy,
        shaping: Shaping {
            temperature: 0.0,
            ..Shaping::default()
        },
        seed: 0,
        forced_quantum: 0,
        method,
        end_of_generation: EndOfGeneration::Stop,
        reasoning_budget: None,
    };
    block_on(async {
        let mut request = engine
            .client()
            .admit(RequestOptions {
                input,
                options,
                constraint: None,
                retention: PrefixRetention::Transient,
                output_capacity: 16,
            })
            .await
            .map_err(|error| error.to_string())?;
        loop {
            match request.receive().await {
                Some(RequestEvent::Output(_)) => {}
                Some(RequestEvent::Completed { usage, .. }) => return Ok(usage),
                Some(RequestEvent::Failed(error)) => return Err(error.to_string()),
                None => return Err("the request ended without a terminal event".into()),
            }
        }
    })
}
