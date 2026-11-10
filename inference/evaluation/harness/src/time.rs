//! `time`: prefill and decode throughput at context lengths and concurrency,
//! on fixtures sized to each context, without drafting (the drafter's
//! acceptance is `draft`'s; served drafting is the serving benchmark's).
//!
//! Cells, at each context N and concurrency C:
//! - `prefill`: C fresh sequences take their fixtures' tokens in prefill
//!   steps of at most `PREFILL_ROWS` rows packed across sequences, as the
//!   service bounds them; one cold sample, then warm samples (median). With
//!   `--prefill-attribution`, one further sample runs each launch in its own
//!   timed unit and decomposes the prefill's device time by entry;
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
//!
//! A streamed load's decode cell also records what its experts cost the
//! window, per streamed layer and per step, and its `model`: the step the
//! cost function prices from those costs, the resident load's decode
//! (`--resident-decode`) and the load's own figures, beside the step
//! measured ([`streamed_model`]); its `bar`, the best step any
//! implementation could reach at its cap on this machine, and the gaps
//! between the three ([`streamed_bar`]), from the machine's own lines
//! measured as the run loads ([`crate::machine`]); and its `estimate`, what
//! assessment estimates it at from the plan's structure and the machine's
//! stored figures ([`estimated`]).
//! Its prefill cell runs a lone prompt as prompt groups, as the service
//! does, and records the last timed sample's groups against their bar
//! ([`prompt_groups`]).

use crate::rest::{BankRest, Progress, decode_context_rows, steady_window};
use crate::artifacts::Artifacts;
use crate::engine::{self, Checkpoint, Engine, EngineOptions, Load, PREFILL_ROWS, Sequence, Step};
use crate::fixture::FixtureSet;
use crate::render::{Rendered, Renderer};
use crate::steps::{self, attribution, device_end, median, medians, summary, window};
use magnitude_executor::assessment::RequestReads;
use magnitude_executor::{Demand, Operation, WorkKind};
use seismic::{SubmissionTrace, TraceDetail, host_seconds};
use serde_json::{Value, json};

const MS: f64 = 1e3;

pub struct TimeOptions {
    pub contexts: Vec<usize>,
    pub concurrency: Vec<usize>,
    pub warm: usize,
    pub steps: usize,
    pub attribution_steps: usize,
    /// Attribute one further prefill sample's device time by entry.
    pub prefill_attribution: bool,
    /// Prefill samples, the first of them cold.
    pub prefill_samples: usize,
    pub lookahead: bool,
    /// A resident load's `time` result, whose decode is a streamed load's
    /// resident step (`model`).
    pub resident_decode: Option<std::path::PathBuf>,
    /// The run counts rather than times: no overlap efficiency is measured.
    pub untimed: bool,
}

pub struct Timed {
    pub rendered: Vec<Rendered>,
    pub speed: Vec<Value>,
    /// The target load's wall and processor time.
    pub load: Value,
    pub devices: Vec<String>,
}

pub fn run(
    artifacts: &Artifacts,
    set: &FixtureSet,
    renderer: &Renderer<'_>,
    options: &EngineOptions,
    time: &TimeOptions,
) -> Result<Timed, String> {
    // A named resident result is read after every cell's decode: its absence
    // ends the run before them.
    if let Some(resident) = &time.resident_decode
        && !resident.is_file()
    {
        return Err(format!("{}: the resident result does not exist", resident.display()));
    }
    let mut by_context = Vec::with_capacity(time.contexts.len());
    for &context in &time.contexts {
        let rendered = set
            .fixtures
            .iter()
            .map(|fixture| renderer.render(fixture, Some(context)))
            .collect::<Result<Vec<_>, _>>()?;
        by_context.push((context, rendered));
    }

    let longest = by_context
        .iter()
        .flat_map(|(_, rendered)| rendered.iter().map(|fixture| fixture.tokens.len()))
        .max()
        .expect("a fixture set is nonempty");
    let widest = time.concurrency.iter().copied().max().expect("a run has a concurrency");
    let assessment = assessment(artifacts, options);
    // The CPU's own lines, as the measurement job stores them (it runs here
    // where they are missing): a streamed decode's bar prices a CPU-computed
    // miss by them.
    let machine = crate::machine::CpuMachine::measured(
        options.cache_dir.as_deref(),
        &magnitude_artifacts::model_store(&artifacts.target)
            .map_err(|error| format!("{}: {error}", artifacts.target.display()))?,
    )?;
    let opening = (host_seconds(), process_cpu_seconds());
    let mut engine = Engine::open(
        artifacts,
        options,
        &Load {
            drafter: false,
            proposals: None,
            context_tokens: longest + decode_context_rows(time.warm, time.steps, time.attribution_steps),
            // A cell's sequences are held together.
            sequences: std::num::NonZeroUsize::new(widest).expect("a concurrency is positive"),
            decode_rows: widest,
            exported_logits_rows: 0,
            lookahead: time.lookahead,
        },
    )?;
    // The whole load as this process ran it: a load that finds tuning units
    // unprepared also waits for the worker process that prepares them, whose
    // processor time is not this process's.
    let load = json!({
        "wall_ms": (host_seconds() - opening.0) * MS,
        "process_cpu_ms": (process_cpu_seconds() - opening.1) * MS,
    });
    // A load that streams (a bank of fewer than every expert) reads its
    // misses and its prompt groups' cold experts from the storage device:
    // its own lines at a record's size, before any step; a resident load
    // (no bank, or one of every expert) searches its peak stream once,
    // which its decode cells measure their overlap efficiency under
    // ([`overlap`]).
    let streams = engine.domain.expert_bank_plan().is_some_and(|bank| !bank.whole());
    let storage = engine
        .domain
        .expert_bank_lines()
        .filter(|_| streams)
        .and_then(|lines| lines.first().map(|lines| lines.record_bytes))
        .map(|bytes| crate::machine::StorageMachine::measure(&artifacts.target, bytes));
    let peak = (!streams && !time.untimed).then(|| crate::machine::PeakStream::search(&artifacts.target));
    // A streamed bank's device ceilings, which its bars price moving its
    // experts by, before any step.
    let ceilings = match engine.domain.expert_bank_plan().filter(|_| streams) {
        Some(plan) => Some(Ceilings::beside(&engine.domain, plan, engine.device(), options.cache_dir.as_deref())?),
        None => None,
    };
    let devices = vec![engine.device().info().name().to_owned()];
    let mut speed = Vec::new();
    for (context, rendered) in &by_context {
        for &concurrency in &time.concurrency {
            let histories = (0..concurrency)
                .map(|index| &rendered[index % rendered.len()])
                .collect::<Vec<_>>();
            eprintln!("magnitude-harness: prefill context={context} concurrency={concurrency}");
            // The last timed sample's prompt groups, judged once the decode
            // cell has given the run its read line.
            let (cell, group) = prefill_cell(&mut engine, time, &histories)?;
            eprintln!(
                "  {:.1} tok/s",
                cell["tokens_per_second"].as_f64().unwrap_or(f64::NAN)
            );
            let prefill = speed.len();
            speed.push(with_cell(
                cell,
                "prefill",
                *context,
                concurrency,
                &histories,
            ));
            eprintln!("magnitude-harness: decode context={context} concurrency={concurrency}");
            let mut cell = decode_cell(&mut engine, time, &histories, peak.as_ref().map(|peak| (artifacts.target.as_path(), peak)))?;
            eprintln!(
                "  {:.1} tok/s",
                cell["tokens_per_second"].as_f64().unwrap_or(f64::NAN)
            );
            if !cell["detail"]["experts"].is_null() {
                // A streamed load reads its experts past the file cache: a
                // page of its expert stacks the cache holds before its decode
                // or after served reads from it, and voids them.
                let pages = &cell["detail"]["expert_pages"];
                let cached = ["cached_before", "cached_after"]
                    .map(|moment| pages[moment].as_u64().ok_or("the expert stacks' cached pages"));
                let [before, after] = cached;
                cell["cache_served"] = json!(before? + after? > 0);
                let model = streamed_model(&engine, time, *context, concurrency, &cell);
                let priced = model.as_ref().ok().map(|&(_, step)| step);
                cell["model"] = judged(model.map(|(model, _)| model))?;
                let bar = resident_decode(time, *context, concurrency).and_then(|(_, resident, resident_ms)| {
                    let link = ceilings.as_ref().and_then(|ceilings| ceilings.link.clone());
                    let inputs = BarInputs::of(&engine, &machine, &storage, link, options.weight_bound, &resident)?;
                    streamed_bar(&cell, &resident, resident_ms, concurrency, &inputs, priced)
                });
                cell["bar"] = judged(bar)?;
                let estimate = assessment
                    .as_ref()
                    .map_err(Clone::clone)
                    .and_then(|assessment| estimated(assessment, &engine, *context, concurrency, &cell));
                cell["estimate"] = judged(estimate)?;
            }
            if let Some(sample) = group {
                // What the storage device holds over a read that long.
                let read_rate = match &storage {
                    Some(Ok(storage)) => Ok(crate::bar::ReadRate { peak: storage.peak_bytes_per_second, sustained: storage.sustained_bytes_per_second, burst: storage.burst_bytes }),
                    Some(Err(reason)) => Err(format!("the storage device's lines: {reason}")),
                    None => Err("the load streams nothing from storage".to_owned()),
                };
                let resident_seconds = resident_cell(time, "prefill", *context, concurrency)
                    .map(|(_, cell)| cell["detail"]["median_ms"].as_f64())
                    .map_err(|unjudged| match unjudged {
                        Unjudged::Unavailable(reason) | Unjudged::Failed(reason) => reason,
                    })
                    .and_then(|ms| ms.map(|ms| ms / MS).ok_or_else(|| "the resident prefill records no median".to_owned()));
                let ceilings = ceilings.as_ref().expect("a load that runs prompt groups streams, and its ceilings are measured");
                speed[prefill]["detail"]["group"] = prompt_groups(&sample, read_rate, resident_seconds, ceilings);
            }
            speed.push(with_cell(cell, "decode", *context, concurrency, &histories));
        }
    }
    drop(engine);
    let rendered = by_context
        .into_iter()
        .flat_map(|(_, rendered)| rendered)
        .collect::<Vec<_>>();
    Ok(Timed {
        rendered,
        speed,
        load,
        devices,
    })
}

/// Why a streamed cell's model or estimate is not judged.
#[derive(Clone, Debug)]
enum Unjudged {
    /// An input it is judged from is absent: the cell records it
    /// unavailable, and why.
    Unavailable(String),
    /// Anything else: the run fails.
    Failed(String),
}

impl From<String> for Unjudged {
    fn from(error: String) -> Self {
        Self::Failed(error)
    }
}

impl From<&str> for Unjudged {
    fn from(error: &str) -> Self {
        Self::Failed(error.to_owned())
    }
}

/// `unjudged`'s cell value: what was judged, or why it is unavailable; a
/// failure fails the run.
fn judged(unjudged: Result<Value, Unjudged>) -> Result<Value, String> {
    match unjudged {
        Ok(value) => Ok(value),
        Err(Unjudged::Unavailable(reason)) => Ok(json!({ "unavailable": reason })),
        Err(Unjudged::Failed(error)) => Err(error),
    }
}

/// Why an absent input leaves a cell unjudged.
fn unavailable(reason: impl Into<String>) -> Unjudged {
    Unjudged::Unavailable(reason.into())
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
/// up to `PREFILL_ROWS` rows; each sequence's last chunk selects. With the
/// sample's time and steps, the prompt groups a streamed load ran them in,
/// in order, each as its last step left it.
fn prefill_sample(
    engine: &mut Engine,
    histories: &[&Rendered],
    trace: &SubmissionTrace,
) -> Result<(f64, Vec<Step>, Vec<magnitude_executor::PromptGroupCost>), String> {
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
    let mut groups = std::collections::BTreeMap::new();
    while sequences
        .iter()
        .zip(&tokens)
        .any(|(sequence, tokens)| sequence.position < tokens.len())
    {
        let mut budget = PREFILL_ROWS;
        let mut operations = Vec::new();
        let mut advancing = Vec::new();
        let mut rests = Vec::new();
        for (sequence, tokens) in sequences.iter_mut().zip(&tokens) {
            let rows = (tokens.len() - sequence.position).min(PREFILL_ROWS);
            if rows == 0 || rows > budget || !engine.domain.claims(sequence.request) {
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
            rests.push(&tokens[sequence.position..]);
            advancing.push(sequence);
        }
        // A lone prompt chunk runs with the chunks after it as one prompt
        // group, as the service runs a prompt.
        if let ([sequence], [rest]) = (&advancing[..], &rests[..]) {
            engine.plan_prompt(sequence, rest, true)?;
        }
        let (step, _) = engine.step(operations, &mut advancing, Some(trace))?;
        steps.push(step);
        // A lock and a copy of one group's per-layer counts: microseconds
        // of the sample's time, against a chunk's hundreds of milliseconds.
        if let Some(group) = engine.domain.expert_costs().and_then(|costs| costs.group) {
            groups.insert(group.first, group);
        }
    }
    let seconds = host_seconds() - start;
    for sequence in sequences {
        engine.close(sequence)?;
    }
    Ok((seconds, steps, groups.into_values().collect()))
}

/// A streamed bank's device ceilings ([`crate::bar`]): each streamed
/// layer's convert ceiling, a record's seconds (`ConvertCosts`, kept in the
/// kernel cache's measurement records where there is one, so an executor
/// measures it once), and where the bank has a host tier, the link's, a
/// byte's seconds (`TierCopyCosts`, as the measurement job records it); or
/// why each is unmeasured.
struct Ceilings {
    pub(crate) convert: Result<Vec<f64>, String>,
    pub(crate) link: Option<Result<f64, String>>,
}

impl Ceilings {
    /// [`Self::measure`] beside the load in `domain`, its own traffic held
    /// and drained first (`ExecutorDomain::hold_expert_traffic`), as the
    /// measurement job's room beside a model holds it: no point reads the
    /// load's copies, reads or landings. Why the drain ended short: serving
    /// ended, or the load stops (a read or copy that never ends trips the
    /// load's watchdog, which reports it and ends the process).
    fn beside(
        domain: &magnitude_executor::ExecutorDomain,
        plan: &magnitude_executor::ExpertBankPlan,
        device: &seismic::Device,
        cache: Option<&std::path::Path>,
    ) -> Result<Self, String> {
        // Held until this returns, by whichever exit.
        let _traffic = domain.hold_expert_traffic();
        domain
            .wait_expert_traffic_quiet()
            .map(|()| Self::measure(device, plan, cache))
            .map_err(engine::text)
    }

    fn measure(
        device: &seismic::Device,
        plan: &magnitude_executor::ExpertBankPlan,
        cache: Option<&std::path::Path>,
    ) -> Self {
        let link = plan.tier_reservation().map(|_| with_records(cache, |store| Self::link(device, store)));
        Self { convert: with_records(cache, |store| Self::convert(device, plan, store)), link }
    }

    /// The link's ceiling, a byte's seconds: a tier copy on the whole grid
    /// (`TierCopyCosts::link_byte_seconds`), the job's point of copying from
    /// pinned host memory into device memory, measured first where `store`
    /// holds none ([`measure_requested`]). A route's rate is a kernel's
    /// reading weights, which the link's peak is not.
    fn link(
        device: &seismic::Device,
        store: &dyn seismic::measure::RecordStore<seismic::PointKey>,
    ) -> Result<f64, String> {
        use magnitude_executor::assessment::{PointRequest, TierCopyCosts};
        measure_requested(device, store, &[PointRequest::TierCopy])?;
        let costs = TierCopyCosts::stored(device.topology(), device.info(), store).map_err(|error| error.to_string())?;
        Ok(costs.link_byte_seconds())
    }

    /// Each streamed layer's convert ceiling (`ConvertCosts::of_bank`), its
    /// points measured first where `store` holds none, as the measurement
    /// job measures requested points ([`measure_requested`]).
    fn convert(
        device: &seismic::Device,
        plan: &magnitude_executor::ExpertBankPlan,
        store: &dyn seismic::measure::RecordStore<seismic::PointKey>,
    ) -> Result<Vec<f64>, String> {
        use magnitude_executor::assessment::{ConvertCosts, PointRequest};
        let requests = ConvertCosts::forms(plan).map(PointRequest::Convert).collect::<Vec<_>>();
        measure_requested(device, store, &requests)?;
        ConvertCosts::of_bank(device.topology(), device.info(), plan, store).map_err(|error| error.to_string())
    }
}

/// Measure `requests` (a load's) on `device` in this process, as the
/// measurement job measures them (`assessment::requests::measure`), each
/// walk given its job time (`worker::measure::JobTime`), into `store`,
/// until no point is left expired: the harness is a tool, which may
/// measure in its own process.
/// Why the walks could not run, or stopped before measuring the rest.
fn measure_requested(
    device: &seismic::Device,
    store: &dyn seismic::measure::RecordStore<seismic::PointKey>,
    requests: &[magnitude_executor::assessment::PointRequest],
) -> Result<(), String> {
    use magnitude_engine::worker::measure::{JobProgress, JobTime};
    use magnitude_executor::assessment::requests;
    use seismic::measure::Outcome;
    let mut time = JobTime::default();
    loop {
        let mut measurer = seismic::Measurer::new(magnitude_executor::SEARCH_SETTINGS.error_rate, magnitude_executor::MIN_SAMPLE_SECONDS);
        let deadline = requests::JobDeadlines::new(std::time::Instant::now(), time.given());
        let (mut measured, mut failed, mut expired) = (false, false, false);
        let walked = requests::measure(
            &mut measurer,
            device,
            store,
            &magnitude_executor::assessment::Outstanding { standing: Vec::new(), loads: requests.to_vec() },
            deadline,
            &mut |_| seismic::measure::Attempt::Proceed,
            &mut |_, outcome| match outcome {
                Outcome::Loaded(_) => {}
                Outcome::Measured(_) | Outcome::Skipped(_) | Outcome::Unformed(_) => measured = true,
                Outcome::Unmade(_) | Outcome::Failed(_) | Outcome::Store(_) => failed = true,
                Outcome::Expired => expired = true,
            },
        )
        .map_err(|error| error.to_string())?;
        if let Some(error) = walked.ended.into_iter().chain(walked.failures).next() {
            return Err(error.to_string());
        }
        // A point left unrecorded otherwise is the caller's to report, from
        // what is stored.
        if !expired {
            return Ok(());
        }
        let progress = match measured {
            true => JobProgress::Measured,
            false => JobProgress::Empty { failed },
        };
        time.after(progress).map_err(|stall| format!("the measurement walk stopped: {stall}"))?;
    }
}

/// `then` over the kernel cache's measurement records at `cache`, or over
/// records held for the run alone without one.
fn with_records<T>(cache: Option<&std::path::Path>, then: impl FnOnce(&dyn seismic::measure::RecordStore<seismic::PointKey>) -> T) -> T {
    match cache {
        Some(root) => then(&magnitude_executor::KernelCache::measurement_records(root)),
        None => then(&seismic::measure::HeldStore::default()),
    }
}

/// A prefill attributed by entry (`--prefill-attribution`): its tier
/// copy's seconds and the records its prompt groups pulled. What the run
/// achieved, reported beside the bar, which it does not price. Its device
/// convert has no seconds of its own: each round's landing launch
/// (`routed_round`) parks on the reads it converts.
#[derive(Clone, Copy, Debug, PartialEq)]
struct AttributedPrefill {
    copy_seconds: f64,
    copied: u64,
}

/// A streamed load's last timed prefill sample, run whole as prompt groups:
/// its groups, its time, and the prefill attributed by entry, or why there
/// is none.
struct GroupSample {
    groups: Vec<magnitude_executor::PromptGroupCost>,
    measured_ms: f64,
    attributed: Result<AttributedPrefill, String>,
}

/// The prefill cell, and of a streamed load whose last timed sample ran its
/// whole prompt as prompt groups, that sample ([`GroupSample`]).
fn prefill_cell(
    engine: &mut Engine,
    time: &TimeOptions,
    histories: &[&Rendered],
) -> Result<(Value, Option<GroupSample>), String> {
    let rows = histories
        .iter()
        .map(|history| history.tokens.len())
        .sum::<usize>() as f64;
    let trace = traced(engine, TraceDetail::Submissions)?;
    // The cold sample, then warm samples until `prefill_samples - 1` of them
    // ran with the bank at rest (`BankRest`): only those are the cell's
    // prefill. The others are reported apart, with what moved; a bank that
    // moves between them is waited for while its moves make progress
    // ([`Progress`]).
    let wanted = time.prefill_samples.saturating_sub(1);
    let mut samples_ms = Vec::with_capacity(time.prefill_samples);
    let mut unsteady = Vec::new();
    let mut summaries = Vec::new();
    let mut groups = Vec::new();
    // The graphs each steady sample recorded and updated: a warm prefill
    // runs only shapes the first ran.
    let mut warm_graphs = seismic::GraphReplays::default();
    let mut attempts = 0;
    let mut progress = Progress::default();
    while samples_ms.len() < 1 + wanted {
        let cold = samples_ms.is_empty();
        let (graphs, before) = (engine.graph_replays(), BankRest::covered(engine)?);
        let (seconds, steps, sampled) = prefill_sample(engine, histories, &trace)?;
        if !cold {
            attempts += 1;
            if let Some(moved) = BankRest::moved(before, BankRest::of(engine)?) {
                let verdict = progress.note(&moved, BankRest::targets(engine)?);
                unsteady.push(json!({ "ms": seconds * MS, "moved": moved.record() }));
                if let Err(why) = verdict {
                    return Err(format!(
                        "{} of {attempts} warm prefill samples ran with the expert bank at rest, \
                         of {wanted} wanted: {why}: {}",
                        samples_ms.len() - 1,
                        Value::from(unsteady)
                    ));
                }
                continue;
            }
            warm_graphs = warm_graphs.plus(engine.graph_replays().since(graphs));
            // A chunk claimed from a prompt group's chain submits nothing of
            // its own: the group's first step submitted the chain.
            for step in steps.iter().filter(|step| !step.submissions.is_empty()) {
                summaries.push(summary(step, None)?);
            }
        }
        samples_ms.push(seconds * MS);
        groups = sampled;
    }
    drop(trace);
    let chunks = histories
        .iter()
        .map(|history| history.tokens.len().div_ceil(PREFILL_ROWS))
        .sum::<usize>();
    // Whether a sample ran its whole prompt as prompt groups.
    let whole = |groups: &[magnitude_executor::PromptGroupCost]| groups.iter().map(|group| group.steps as usize).sum::<usize>() == chunks;
    let measured_ms = *samples_ms.last().expect("a prefill sample");
    let mut warm = samples_ms[1..].to_vec();
    let median_ms = median(&mut warm);
    // Untimed: each launch runs in its own timed unit, and each step of the
    // sample collects its own launches.
    let (attributed, prefill) = if time.prefill_attribution {
        let trace = traced(engine, TraceDetail::Launches)?;
        let (_, steps, attributed_groups) = prefill_sample(engine, histories, &trace)?;
        let mut submissions = steps
            .into_iter()
            .flat_map(|step| step.submissions)
            .collect::<Vec<_>>();
        submissions.extend(trace.collect().map_err(|error| error.to_string())?);
        let attributed = attribution(&submissions, 1, (f64::NEG_INFINITY, f64::INFINITY));
        let prefill = match whole(&attributed_groups) {
            true => attributed_prefill(&attributed, &attributed_groups),
            false => Err("the attributed prefill did not run whole as prompt groups".to_owned()),
        };
        (attributed, prefill)
    } else {
        (Value::Null, Err("no prefill was attributed by entry (--prefill-attribution)".to_owned()))
    };
    let group = whole(&groups).then_some(GroupSample { groups, measured_ms, attributed: prefill });
    let cell = json!({
        "tokens_per_second": rows / (median_ms / MS),
        "samples": samples_ms[1..].iter().map(|ms| rows / (ms / MS)).collect::<Vec<_>>(),
        "detail": {
            "rows": rows,
            "median_ms": median_ms,
            "cold_ms": samples_ms[0],
            "samples_ms": &samples_ms[1..],
            "median_step": medians(&summaries),
            "attribution": attributed,
            "unsteady": unsteady,
            "graphs": graph_counts(warm_graphs, wanted),
            "graph_budget": graph_budget(engine),
        },
    });
    Ok((cell, group))
}

/// Graphs recorded and updated over `units` steps or prefills, in all and
/// per unit.
fn graph_counts(counts: seismic::GraphReplays, units: usize) -> Value {
    let per = |count: u64| (units > 0).then(|| count as f64 / units as f64);
    json!({
        "units": units,
        "recorded": counts.recorded,
        "updated": counts.updated,
        "recorded_per_unit": per(counts.recorded),
        "updated_per_unit": per(counts.updated),
    })
}

/// The device's graph budget: what its families' live slots hold recorded,
/// one graph per shape and launch kept in flight, and the memory that
/// census costs.
fn graph_budget(engine: &Engine) -> Value {
    let (census, bytes) = engine.domain.graph_budget();
    json!({
        "graphs": census.graphs,
        "launches": census.launches,
        "arguments": census.arguments,
        "device_bytes": bytes.device,
        "upload_bytes": bytes.upload,
    })
}

/// The tier copy of a prefill attributed by entry
/// (`attribution`, its one sample), and the records its prompt groups
/// (`groups`, the whole prompt) pulled from the host tier.
fn attributed_prefill(attribution: &Value, groups: &[magnitude_executor::PromptGroupCost]) -> Result<AttributedPrefill, String> {
    let seconds = |entry: &str| entry_launches(attribution, &[entry]).map(|launches| launches.ms / MS);
    Ok(AttributedPrefill {
        copy_seconds: seconds(<magnitude_kernels::routed_copy::Entry as seismic::Entry>::NAME)?,
        copied: groups.iter().flat_map(|group| &group.layers).map(|layer| layer.pulled).sum(),
    })
}

/// A streamed load's prompt groups (a prefill cell's last timed sample, the
/// whole prompt in them, [`GroupSample`]) against their bar
/// ([`crate::bar::PrefillBar`]): the groups' cold bytes at the
/// storage device's `read_rate` over a read that long ([`crate::bar::ReadRate`]),
/// the resident prefill (`resident_seconds`, the resident load's at the same
/// context and concurrency) plus their conveyed records' convert at its
/// ceiling, and the bytes they move over a host tier's link at its ceiling
/// (`ceilings`), the slowest; and beside it, what the attributed prefill's
/// copy achieved. The bar is unavailable, with why, without an
/// input one of its terms needs. The layers' counts are summed over the
/// groups.
fn prompt_groups(sample: &GroupSample, read_rate: Result<crate::bar::ReadRate, String>, resident_seconds: Result<f64, String>, ceilings: &Ceilings) -> Value {
    let groups = &sample.groups;
    let layers = groups.first().map_or(0, |group| group.layers.len());
    let count = |field: fn(&magnitude_executor::PromptGroupLayer) -> u64, layer: usize| {
        groups.iter().map(|group| field(&group.layers[layer])).sum::<u64>()
    };
    let inputs = crate::bar::PrefillInputs {
        layers: (0..layers)
            .map(|layer| crate::bar::PrefillLayer {
                conveyed: count(|layer| layer.conveyed, layer),
                cold_bytes: count(|layer| layer.cold_bytes, layer),
                pulled: count(|layer| layer.pulled, layer),
                tier_bytes: count(|layer| layer.tier_bytes, layer),
            })
            .collect(),
        read_rate,
        resident_seconds,
        convert: ceilings.convert.clone(),
        link: ceilings.link.clone(),
    };
    let bar = crate::bar::PrefillBar::of(&inputs);
    let measured_ms = sample.measured_ms;
    let ms = |field: fn(&magnitude_executor::PromptGroupLayer) -> std::time::Duration, layer: usize| {
        groups.iter().map(|group| field(&group.layers[layer]).as_secs_f64()).sum::<f64>() * MS
    };
    json!({
        "groups": groups.len(),
        "sizes": groups.iter().map(|group| group.steps).collect::<Vec<_>>(),
        "steps": groups.iter().map(|group| group.steps).sum::<u64>(),
        "measured_ms": measured_ms,
        "bar_ms": match &bar {
            Ok(bar) => json!(bar.seconds() * MS),
            Err(reason) => json!({ "unavailable": reason }),
        },
        "ratio": bar.as_ref().ok().map(|bar| bar.seconds() * MS / measured_ms),
        "bar_terms": bar.as_ref().ok().map(|bar| json!({
            "read_ms": bar.read * MS,
            "resident_ms": bar.resident * MS,
            "convert_ms": bar.convert * MS,
            "link_ms": bar.link * MS,
        })),
        "ceilings": {
            "convert_us_per_record": match &inputs.convert {
                Ok(records) => json!(records.iter().map(|seconds| seconds * 1e6).collect::<Vec<_>>()),
                Err(reason) => json!({ "unavailable": reason }),
            },
            "link_bytes_per_second": match &inputs.link {
                None => Value::Null,
                Some(Ok(byte)) => json!(1. / byte),
                Some(Err(reason)) => json!({ "unavailable": reason }),
            },
        },
        "achieved": match &sample.attributed {
            Ok(attributed) => json!({
                "copy_ms": attributed.copy_seconds * MS,
                "copy_us_per_record": (attributed.copied > 0).then(|| attributed.copy_seconds / attributed.copied as f64 * 1e6),
            }),
            Err(reason) => json!({ "unavailable": reason }),
        },
        "resident_ms": inputs.resident_seconds.as_ref().ok().map(|seconds| seconds * MS),
        "read_rate": match &inputs.read_rate {
            Ok(rate) => json!({ "peak_bytes_per_second": rate.peak, "sustained_bytes_per_second": rate.sustained, "burst_bytes": rate.burst }),
            Err(reason) => json!({ "unavailable": reason }),
        },
        "layers": (0..layers).map(|layer| json!({
            "union": count(|layer| layer.union, layer),
            "hits": count(|layer| layer.hits, layer),
            "conveyed": count(|layer| layer.conveyed, layer),
            "cold_bytes": count(|layer| layer.cold_bytes, layer),
            "pulled": count(|layer| layer.pulled, layer),
            "seeded": count(|layer| layer.seeded, layer),
            "parked_ms": ms(|layer| layer.parked, layer),
            "compute_ms": ms(|layer| layer.compute, layer),
            "computed": count(|layer| u64::from(layer.computed), layer),
            "timeline": groups.first().map(|group| timeline(&group.layers[layer].timeline)),
        })).collect::<Vec<_>>(),
    })
}

/// A layer's timeline of a prompt group (`PromptGroupLayer::timeline`), in
/// ms from the group's first request: its reads submitted and ended, first
/// and last, and per step its request taken, its round 0 arrived and its
/// request served whole.
fn timeline(timeline: &magnitude_executor::PromptLayerTimeline) -> Value {
    let at = |at: std::time::Duration| at.as_secs_f64() * MS;
    let span = |span: Option<magnitude_executor::PromptSpan>| span.map(|span| [at(span.first), at(span.last)]);
    json!({
        "submitted": span(timeline.submitted),
        "landed": span(timeline.landed),
        "steps": timeline.steps.iter().map(|step| json!({
            "requested": step.requested.map(at),
            "arrived": step.arrived.map(at),
            "served": step.served.map(at),
        })).collect::<Vec<_>>(),
    })
}

fn decode_cell(
    engine: &mut Engine,
    time: &TimeOptions,
    histories: &[&Rendered],
    peak: Option<(&std::path::Path, &Result<crate::machine::PeakStream, String>)>,
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
    // The file cache's pages of a streamed load's expert stacks before its
    // first decode step and once its last has ended: read where no step is
    // in flight, so the reading's time enters no step's (nor `σ`'s
    // residuals, which take a queued step's time from its submission).
    let streamed = engine.domain.expert_costs().is_some();
    let pages_before = streamed.then(|| engine.cached_expert_pages()).transpose()?;
    let mut detail = measure(engine, &mut sequences, time, &decode)?;
    let step_ms = detail["median_ms"].as_f64().expect("a median");
    let concurrency = sequences.len() as f64;
    let samples = window_steps(&detail)?
        .iter()
        .map(|step| Ok(concurrency / (step["interval_ms"].as_f64().ok_or("a step's interval")? / MS)))
        .collect::<Result<Vec<_>, String>>()?;
    for sequence in sequences {
        engine.close(sequence)?;
    }
    // A resident load's overlap efficiency `η`, which a streamed load's
    // unified ceiling takes (`crate::bar`), once the cell's sequences are
    // closed: its runs prefill their own.
    if let Some((target, peak)) = peak {
        detail["overlap"] = match peak.clone().and_then(|peak| overlap(engine, histories, time.steps, &decode, target, peak)) {
            Ok(overlap) => overlap,
            Err(reason) => json!({ "unavailable": reason }),
        };
    }
    let pages_after = streamed.then(|| engine.cached_expert_pages()).transpose()?;
    detail["expert_pages"] = json!(pages_before.zip(pages_after).map(|((before, pages), (after, _))| {
        json!({ "pages": pages, "cached_before": before, "cached_after": after })
    }));
    Ok(json!({
        "tokens_per_second": concurrency / (step_ms / MS),
        "samples": samples,
        "detail": detail,
    }))
}

/// `count` steps made by `make`: the mean seconds a step, each from the
/// completion of the step before it ([`step_intervals`]).
fn decode_steps(
    engine: &mut Engine,
    sequences: &mut [Sequence],
    count: usize,
    make: &dyn Fn(&[Sequence]) -> Result<Vec<Operation>, String>,
) -> Result<f64, String> {
    let mut ends = Vec::with_capacity(count);
    for _ in 0..count {
        let operations = make(sequences)?;
        let mut refs = sequences.iter_mut().collect::<Vec<_>>();
        let (step, _) = engine.step(operations, &mut refs, None)?;
        ends.push(step.end);
    }
    let intervals = step_intervals(&ends);
    match intervals.len() {
        0 => Err("fewer than two steps make no interval".into()),
        count => Ok(intervals.iter().sum::<f64>() / count as f64),
    }
}

/// A run of steps' times: each from the completion before it to its own,
/// the steps' `ends` in order. Under lookahead the host queues a step
/// before the one before has completed, so a step's own host span (its
/// call's start to its end) is not its time; the device paces the
/// completions, so their intervals are.
fn step_intervals(ends: &[f64]) -> Vec<f64> {
    ends.windows(2).map(|pair| pair[1] - pair[0]).collect()
}

/// The overlap efficiency `η` of a resident decode on this host: `count`
/// steps alone, then `count` while a read stream runs at the storage
/// device's `peak` on the model's `target` file ([`crate::machine::PeakStream`]),
/// then `count` alone again, each run on sequences prefilled afresh from
/// `histories`, so each decodes from the cell's depth within the context the
/// load holds; `η` the steps alone, the mean of both runs, over the steps
/// under reads.
fn overlap(
    engine: &mut Engine,
    histories: &[&Rendered],
    count: usize,
    make: &dyn Fn(&[Sequence]) -> Result<Vec<Operation>, String>,
    target: &std::path::Path,
    peak: crate::machine::PeakStream,
) -> Result<Value, String> {
    let run = |engine: &mut Engine| -> Result<f64, String> {
        let mut sequences = Vec::with_capacity(histories.len());
        for history in histories {
            let mut sequence = engine.open_sequence()?;
            engine.prefill(&mut sequence, &engine::tokens(&history.tokens), true)?;
            sequences.push(sequence);
        }
        let seconds = decode_steps(engine, &mut sequences, count, make)?;
        for sequence in sequences {
            engine.close(sequence)?;
        }
        Ok(seconds)
    };
    let before = run(engine)?;
    let stream = crate::machine::ReadStream::start(target, peak.bytes, peak.depth)?;
    let under = run(engine);
    let streamed = stream.stop()?;
    let under = under?;
    let after = run(engine)?;
    let alone = (before + after) / 2.;
    Ok(json!({
        "eta": alone / under,
        "alone_ms": alone * MS,
        "under_reads_ms": under * MS,
        "stream": peak,
        "streamed_bytes_per_second": streamed,
    }))
}

/// A cell's decode as [`measure`] runs it: its sequences, each's state at
/// its history, and the routing of every step before its window.
struct Decoding<'a> {
    engine: &'a mut Engine,
    sequences: &'a mut Vec<Sequence>,
    checkpoints: Vec<Checkpoint>,
    warm_expert_steps: Vec<Value>,
}

/// Measure `time.steps` steps made by `make` after `time.warm`, traced per
/// submission, then attribute `time.attribution_steps` more by entry.
fn measure(
    engine: &mut Engine,
    sequences: &mut Vec<Sequence>,
    time: &TimeOptions,
    make: &dyn Fn(&[Sequence]) -> Result<Vec<Operation>, String>,
) -> Result<Value, String> {
    // Every step's selections, per sequence in order, its digest of the
    // first ones ([`selections_digest`]).
    let selected = std::cell::RefCell::new(Vec::<u8>::new());
    let run = |engine: &mut Engine,
               sequences: &mut [Sequence],
               trace: &SubmissionTrace|
     -> Result<Step, String> {
        let operations = make(sequences)?;
        let mut refs = sequences.iter_mut().collect::<Vec<_>>();
        // Under lookahead the trace is collected once per window: a per-step
        // collect would wait for the step queued behind this one.
        let per_step = (!engine.lookahead).then_some(trace);
        let (step, _) = engine.step(operations, &mut refs, per_step)?;
        selected.borrow_mut().extend(
            sequences
                .iter()
                .flat_map(|sequence| sequence.selected.map_or(u32::MAX, |token| token.0).to_le_bytes()),
        );
        Ok(step)
    };
    let collect = |trace: &SubmissionTrace| trace.collect().map_err(|error| error.to_string());
    let trace = traced(engine, TraceDetail::Submissions)?;
    // What the bank holds as decode begins: where a replay of its routing
    // starts.
    let held = engine.domain.experts_held();
    // The decode as it begins, each sequence at its history: where a
    // discarded attempt's decode begins again.
    let checkpoints = sequences.iter().map(|sequence| engine.checkpoint(sequence)).collect::<Result<Vec<_>, _>>()?;
    let mut decode = Decoding { engine, sequences, checkpoints, warm_expert_steps: Vec::new() };
    // The window is one span the bank rests across (`BankRest`): warm
    // steps, then more until one runs at rest, then the window; an attempt
    // whose window any step the bank moved across is discarded, with what
    // moved, and the decode begins again from its histories
    // ([`steady_window`]). So no resize or cover read is timed as decode,
    // every figure of the window is of one steady span, and no attempt runs
    // deeper than the load's context.
    let ((summaries, mut submissions, phases, span, cpu, threads, throttling, experts, graphs), discarded) = steady_window(
        &mut decode,
        time.warm,
        |decode| {
            let before = BankRest::covered(decode.engine)?;
            run(decode.engine, decode.sequences, &trace)?;
            decode.warm_expert_steps.extend(expert_step_of(decode.engine)?);
            Ok(BankRest::moved(before, BankRest::of(decode.engine)?))
        },
        |decode| BankRest::targets(decode.engine),
        |decode| {
            collect(&trace)?;
            let mut summaries = Vec::with_capacity(time.steps);
            let mut previous = None;
            let mut submissions = Vec::new();
            // Each measured step's instants on this thread, the next's start
            // filled in as it begins.
            let mut phases: Vec<steps::HostPhases> = Vec::with_capacity(time.steps);
            // The window opens at the end of the first measured step. Collecting
            // the warm steps' trace waited for everything queued, which under
            // lookahead includes the first measured step itself: it finished
            // before any clock here started, so a window opened at its start
            // would cover one step more than the device ran.
            let mut span = (f64::NAN, f64::NAN);
            let mut cpu = (0.0, 0.0);
            // Each thread kind's processor time, and the quota's throttling, at
            // the window's start and end.
            let (mut threads, mut throttling) = (None, None);
            let (mut threads_end, mut throttling_end) = (None, None);
            // The load's graphs recorded and updated at the window's start.
            let mut graphs = decode.engine.graph_replays();
            // A streamed load's per-layer expert costs at the window's ends.
            let mut experts = None;
            let mut moved = None;
            // The bank after each step is the bank before the next: one reading
            // a step, and between a step's end and the next's launch nothing
            // more than the step loop itself does, as served, but for a copy of
            // the expert costs, which are rendered once the window has closed.
            let mut rest = BankRest::of(decode.engine)?;
            let mut expert_costs = Vec::with_capacity(time.steps);
            // An untimed streamed cell records its window's read timeline
            // (`seismic_io::timeline`): the warm steps' events dropped.
            if time.untimed && decode.engine.domain.expert_costs().is_some() {
                seismic_io::timeline::record();
                drop(seismic_io::timeline::take());
            }
            for index in 0..time.steps {
                let step = run(decode.engine, decode.sequences, &trace)?;
                if let Some(last) = phases.last_mut() {
                    last.next = step.start;
                }
                phases.push(steps::HostPhases {
                    start: step.start,
                    submitted: step.submitted,
                    finished: step.finished,
                    reconciled: step.reconciled,
                    end: step.end,
                    next: step.end,
                });
                let after = BankRest::of(decode.engine)?;
                if let Some(moves) = BankRest::moved(rest, after) {
                    expert_costs.push((decode.engine.domain.expert_costs(), after.and_then(|after| after.host_slots())));
                    moved = Some((index, moves.record()));
                    break;
                }
                rest = after;
                if index == 0 {
                    span.0 = step.end;
                    cpu.0 = process_cpu_seconds();
                    threads = thread_cpu_seconds();
                    throttling = throttled();
                    graphs = decode.engine.graph_replays();
                    experts = decode.engine.domain.expert_costs();
                }
                let mut summary = if decode.engine.lookahead {
                    json!({})
                } else {
                    let summary = summary(&step, previous)?;
                    previous = Some(device_end(&step));
                    summary
                };
                // The step's time, the one every figure of the cell takes
                // ([`step_intervals`]): from the end of the step before it, as
                // steps complete one after another.
                if index > 0 {
                    summary["interval_ms"] = json!((step.end - span.1) * MS);
                }
                // The domain's host timing of the step (`TargetHostTiming`): its
                // launch to its first commit, the previous selection read to
                // that commit (none or zero for a step queued before it), and
                // its finish.
                if let Some(timing) = decode.engine.domain.target_timing() {
                    summary["target_launch_ms"] = json!(timing.launch.as_secs_f64() * MS);
                    summary["target_finish_ms"] = json!(timing.finish.as_secs_f64() * MS);
                    summary["target_selection_to_commit_ms"] =
                        json!(timing.selection_to_commit.map(|gap| gap.as_secs_f64() * MS));
                }
                span.1 = step.end;
                // The window's end, read together where the process time always
                // was: nothing run after the last step (an attribution's steps)
                // counts in its figures.
                if index + 1 == time.steps {
                    cpu.1 = process_cpu_seconds();
                    threads_end = thread_cpu_seconds();
                    throttling_end = throttled();
                }
                expert_costs.push((decode.engine.domain.expert_costs(), rest.and_then(|rest| rest.host_slots())));
                summaries.push(summary);
                submissions.extend(step.submissions);
            }
            let threads = threads.zip(threads_end);
            let throttling = throttling.zip(throttling_end);
            let expert_steps = expert_costs
                .into_iter()
                .map(|(costs, host_slots)| expert_step(costs, host_slots))
                .collect::<Result<Vec<_>, _>>()?;
            let Some((step, moves)) = moved else {
                for (summary, expert_step) in summaries.iter_mut().zip(expert_steps) {
                    if let Some(expert_step) = expert_step {
                        summary["expert_step"] = expert_step;
                    }
                }
                return Ok(Ok((summaries, submissions, phases, span, cpu, threads, throttling, experts, graphs)));
            };
            // The discarded window's work is collected and dropped, but for
            // its steps' routing, which a replay of the decode runs through.
            decode.warm_expert_steps.extend(expert_steps.into_iter().flatten());
            collect(&trace)?;
            Ok(Err(json!({ "step": step, "moved": moves })))
        },
        |decode| {
            for sequence in std::mem::take(decode.sequences) {
                decode.engine.close(sequence)?;
            }
            *decode.sequences = decode.checkpoints.iter().map(|checkpoint| decode.engine.open_from(checkpoint)).collect::<Result<_, _>>()?;
            selected.borrow_mut().clear();
            Ok(())
        },
    )?;
    let Decoding { engine, sequences, warm_expert_steps, .. } = decode;
    submissions.extend(collect(&trace)?);
    drop(trace);
    let read_timeline = seismic_io::timeline::recording().then(|| {
        use std::fmt::Write;
        seismic_io::timeline::take().into_iter().fold(String::new(), |mut lines, event| {
            let _ = writeln!(lines, "{} {} {} {} {} {}", event.at, event.kind.name(), event.job, event.a, event.b, event.c);
            lines
        })
    });
    // Per streamed layer, what the window's steps (after the first, as the
    // window's span) read on demand, how long the device parked for it and
    // how long its reads took; and the window's effective read bandwidth,
    // its missed bytes over the summed time of their reads (none when it
    // read nothing).
    // Also the window's demand requests, and every request of the served
    // file the load has timed (its probe at load, then each decode step's):
    // the sums of the served file's read line (`RequestReads::line`).
    let (experts, read_bytes_per_second, demand_reads, site) = match (experts, engine.domain.expert_costs()) {
        (Some(start), Some(end)) => {
            let reads = end.reads.since(&start.reads);
            let sums = |reads| serde_json::to_value(reads).expect("read sums serialize");
            let (demand_reads, served_reads) = (sums(reads), sums(end.reads));
            // The bandwidth is the demand requests' that read every miss.
            let bandwidth = (reads.seconds > 0.).then(|| reads.bytes / reads.seconds);
            let layers = start
                .layers
                .iter()
                .zip(&end.layers)
                .map(|(start, end)| {
                    json!({
                        "misses": end.misses - start.misses,
                        "missed_bytes": end.missed_bytes - start.missed_bytes,
                        "hosted": end.hosted - start.hosted,
                        "reading": end.reading - start.reading,
                        "unanticipated": end.unanticipated - start.unanticipated,
                        "lead": end.lead - start.lead,
                        "led": end.led - start.led,
                        "parked_ms": (end.parked - start.parked).as_secs_f64() * MS,
                        "read_ms": (end.read - start.read).as_secs_f64() * MS,
                    })
                })
                .collect::<Vec<_>>();
            // The window's hints, judged by their one recall and precision
            // (`HintCost::recall`, `HintCost::precision`).
            let window = end.hints.since(&start.hints);
            let hints = json!({
                "reads": window.reads,
                "bytes": window.bytes,
                "candidates": window.candidates,
                "named": window.named,
                "early": window.early,
                "hintable": window.hintable,
                "unhintable": window.unhintable,
                "recall": window.recall(),
                "precision": window.precision(),
                "layer_steps": window.layer_steps,
            });
            // The CPU site's serving over the window: the misses it served,
            // the rest of the tier's the device read over the link, and each
            // term of its requests' path, per step and per request.
            let served = end.site.since(&start.site);
            let steps = (time.steps - 1) as f64;
            let missed = start.layers.iter().zip(&end.layers).map(|(start, end)| end.misses - start.misses).sum::<u64>();
            let sited = start.layers.iter().zip(&end.layers).map(|(start, end)| end.sited - start.sited).sum::<u64>();
            let per_request = |duration: std::time::Duration| {
                (served.requests > 0).then(|| duration.as_secs_f64() * 1e6 / served.requests as f64)
            };
            let terms = magnitude_executor::SITE_TERM_NAMES
                .iter()
                .zip(served.terms)
                .map(|(name, duration)| {
                    (name.to_string(), json!({ "ms_per_step": duration.as_secs_f64() * MS / steps, "us_per_request": per_request(duration) }))
                })
                .collect::<serde_json::Map<_, _>>();
            let site = json!({
                "promotions_per_step": (end.promoted - start.promoted) as f64 / steps,
                "requests_per_step": served.requests as f64 / steps,
                "cpu_misses_per_step": sited as f64 / steps,
                "link_misses_per_step": missed.checked_sub(sited).ok_or("the site served more misses than the window had")? as f64 / steps,
                "terms": terms,
            });
            (
                Value::from(layers),
                json!(bandwidth),
                json!({
                    "demand": demand_reads,
                    "served": served_reads,
                    "hints": hints,
                }),
                site,
            )
        }
        _ => (Value::Null, Value::Null, Value::Null, Value::Null),
    };
    // Where the experts are as the timed window ends: what a growth begun
    // during it has joined by then.
    let residency = engine.domain.expert_residency().map_err(|error| format!("the expert residency: {error}"))?;
    let attributed = if time.attribution_steps == 0 {
        Value::Null
    } else {
        let trace = traced(engine, TraceDetail::Launches)?;
        let mut attributed = Vec::new();
        // The window, from a leading step's end (`from`) to the last
        // attributed step's (`until`). Under lookahead the leading step was
        // queued before the trace opened, so its launches are not traced;
        // each step after it is queued behind the one before and starts
        // after that one's end, so the window holds exactly the attributed
        // steps' launches. Attributed steps are untimed, so each collects
        // its own launches even under lookahead.
        let mut attributed_step = || {
            let operations = make(sequences)?;
            let mut refs = sequences.iter_mut().collect::<Vec<_>>();
            let (step, _) = engine.step(operations, &mut refs, Some(&trace))?;
            attributed.extend(step.submissions);
            Ok::<_, String>(step.end)
        };
        let from = attributed_step()?;
        let until = (0..time.attribution_steps).try_fold(from, |_, _| attributed_step())?;
        attributed.extend(collect(&trace)?);
        attribution(&attributed, time.attribution_steps, (from, until))
    };
    // The window's steps' times, each from the completion before it.
    let mut intervals = summaries[1..]
        .iter()
        .map(|summary| summary["interval_ms"].as_f64().expect("an interval of each step after the first"))
        .collect::<Vec<_>>();
    let mut window = window(time.steps - 1, span, &submissions, &phases);
    // The windows discarded before this one, each with the step the bank
    // moved across and what moved.
    window["unsteady"] = json!(discarded);
    // What the window's steps cost in graphs, which serving pays none of in
    // steady state: every shape it runs ran before the window.
    window["graphs"] = graph_counts(engine.graph_replays().since(graphs), time.steps - 1);
    window["process_cpu_ms_per_step"] = json!((cpu.1 - cpu.0) * MS / (time.steps - 1) as f64);
    window["thread_cpu"] = match threads {
        Some((before, after)) => thread_cpu_per_step(&before, &after, time.steps - 1),
        None => Value::Null,
    };
    // Whether the quota throttled the process in the window: its periods
    // throttled and the time, per step.
    window["throttled"] = match throttling {
        Some((before, after)) => json!({
            "periods_per_step": (after.0 - before.0) as f64 / (time.steps - 1) as f64,
            "ms_per_step": (after.1 - before.1) as f64 / 1000. / (time.steps - 1) as f64,
        }),
        None => Value::Null,
    };
    Ok(json!({
        "warm_steps": time.warm,
        "lookahead": engine.lookahead,
        "median_ms": median(&mut intervals),
        "median_step": medians(&summaries),
        "window": window,
        "selected_sha256": selections_digest(&selected.borrow(), time.warm + time.steps, sequences.len()),
        "experts": experts,
        "read_bytes_per_second": read_bytes_per_second,
        "reads": demand_reads,
        "site": site,
        "read_timeline": read_timeline,
        "expert_bank": engine.domain.expert_costs().map(|costs| json!({
            // What the weight cap bounds, by the owner of each byte: the
            // bytes beside the experts' records that decide the slots.
            "cap_owners": match engine.domain.weight_owners() {
                Ok(owners) => json!(owners.into_iter().map(|(owner, bytes)| json!({ "owner": owner, "bytes": bytes })).collect::<Vec<_>>()),
                Err(reason) => json!({ "unavailable": reason }),
            },
            "slots": costs.slots,
            "records": costs.records,
            "layer_groups": costs.layer_groups,
            "experts": costs.experts,
            "held": held,
            "residency": residency,
        })),
        "warm_expert_steps": warm_expert_steps,
        "steps": summaries,
        "attribution": attributed,
    }))
}

/// The digest of the first `steps` steps' selections of `sequences`
/// sequences: the steps every run of a cell takes, so two runs that decode
/// the same tokens have the same digest, however many more steps a bank
/// took to come to rest in one of them. Decode runs on through those, so
/// the prefix is the same tokens at the same positions.
fn selections_digest(selected: &[u8], steps: usize, sequences: usize) -> String {
    crate::digest::sha256_hex(&selected[..steps * sequences * size_of::<u32>()])
}

/// A streamed decode cell's bank as its record names it: each group's
/// records, each streamed layer's group, a layer's experts, and what each
/// layer held as decode began.
struct BankShape {
    records: Vec<usize>,
    layers: Vec<usize>,
    experts: usize,
    held: Vec<Vec<usize>>,
}

impl BankShape {
    fn of(bank: &Value) -> Result<Self, String> {
        fn field<T: serde::de::DeserializeOwned>(bank: &Value, name: &str) -> Result<T, String> {
            serde_json::from_value(bank[name].clone()).map_err(|error| format!("the bank's {name}: {error}"))
        }
        Ok(Self {
            records: field(bank, "records")?,
            layers: field(bank, "layer_groups")?,
            experts: field(bank, "experts")?,
            held: field(bank, "held")?,
        })
    }
}

/// A decode cell's expert steps: its warm steps', then its measured steps',
/// in order.
fn expert_steps(detail: &Value) -> Result<(Vec<&Value>, Vec<&Value>), String> {
    let warm = detail["warm_expert_steps"].as_array().ok_or("the cell has no warm expert steps")?;
    let measured = detail["steps"].as_array().ok_or("the cell has no steps")?;
    Ok((warm.iter().collect(), measured.iter().map(|step| &step["expert_step"]).collect()))
}

/// Every decode step's routing at each streamed layer, the steps in order,
/// each step's layers in decoder order, from its routing records.
fn decode_reads(warm: &[&Value], measured: &[&Value]) -> Result<Vec<magnitude_executor::RoutedRead>, String> {
    let mut reads = Vec::new();
    for step in warm.iter().chain(measured) {
        let sequence = step["sequence"].as_u64().ok_or("an expert step without its sequence")?;
        let layers = step["experts"].as_array().ok_or("an expert step without its experts")?;
        for (layer, read) in layers.iter().enumerate() {
            let experts: Vec<usize> = match &read["routed"] {
                Value::Null => return Err("a decode step whose routing record was rewritten before it was read".into()),
                routed => serde_json::from_value(routed.clone()).map_err(|error| format!("{routed}: {error}"))?,
            };
            let choices = read["choices"].as_u64().ok_or("a decode step's routing without its choices")?;
            reads.push(magnitude_executor::RoutedRead { sequence, layer, experts, choices });
        }
    }
    Ok(reads)
}

/// A streamed load's expert step that completed with the step just run
/// (none for a load that streams nothing): each streamed layer's timeline
/// counts (`LayerStep`), the experts it missed, evicted and routed to, and
/// the host slots its host tier held as it completed (none without a tier):
/// what the estimate prices the window's misses at.
fn expert_step_of(engine: &Engine) -> Result<Option<Value>, String> {
    let host_slots = engine
        .domain
        .expert_residency()
        .map_err(|error| format!("the expert residency: {error}"))?
        .and_then(|residency| residency.host)
        .map(|host| host.slots);
    expert_step(engine.domain.expert_costs(), host_slots)
}

/// [`expert_step_of`] from a copy of the costs and the host tier's slots
/// taken as the step completed, rendered whenever.
fn expert_step(costs: Option<magnitude_executor::ExpertCosts>, host_slots: Option<u64>) -> Result<Option<Value>, String> {
    let Some(costs) = costs else {
        return Ok(None);
    };
    let (sequence, cost) = costs
        .completed
        .ok_or("a streamed load's step completes its expert step")?;
    let layers = serde_json::to_value(&cost.layers).map_err(|error| error.to_string())?;
    let experts = cost
        .experts
        .iter()
        .map(|layer| {
            json!({
                "missed": layer.missed,
                "sited": layer.sited,
                "site_us": layer.site_path.as_secs_f64() * 1e6,
                "routed": layer.routed.as_ref().map(|read| &read.experts),
                "choices": layer.routed.as_ref().map(|read| read.choices),
            })
        })
        .collect::<Vec<_>>();
    Ok(Some(json!({
        "sequence": sequence,
        "misses": cost.misses,
        "missed_bytes": cost.missed_bytes,
        "layers": layers,
        "experts": experts,
        "host_slots": host_slots,
    })))
}

/// A streamed decode cell's step as the cost function prices it
/// (`assessment::cost`), beside the step the window measured. Its
/// inputs, each listed with where it came from:
/// - `C`: the resident load's decode at the same context and concurrency
///   (`time.resident_decode`), its window's mean step;
/// - `D`, `λ`: the read line of the load's own requests of the file
///   actually served (`RequestReads::line`): its demand reads and what its
///   volume delivered;
/// - each layer's lines (`ExpertLines`): the load's host lines and the
///   GPU's its kernels' costs price, of this executor;
/// - each step's streamed layers (`LayerStep`), as the watcher counted
///   them, each exposing what the cost function prices of its misses
///   (`LayerStep::exposed`, by what held the layer's end); the window is the
///   mean of its steps';
/// - `σ`: the window's all-hit steps' mean less `C`; without an all-hit
///   step it is unmeasured (`overhead` none) and `T` is priced on `C`
///   alone, which the judgment says.
///
/// Why it is unavailable, when an input is absent: no resident decode named
/// or none of the cell's in it, no read line, the load's own measurement
/// failed.
fn streamed_model(
    engine: &Engine,
    time: &TimeOptions,
    context: usize,
    concurrency: usize,
    cell: &Value,
) -> Result<(Value, ModelStep), Unjudged> {
    let detail = &cell["detail"];
    let (path, _, resident_ms) = resident_decode(time, context, concurrency)?;
    let served = served_line(detail)?;
    let lines = expert_lines(engine)?;
    // The window: every measured step after the first, each with its
    // layers' timeline counts.
    let layer_steps = window_layer_steps(detail)?;
    let window = layer_steps.len() as f64;
    let exposed = layer_steps
        .iter()
        .flat_map(|layers| layers.iter().zip(lines.iter()).map(|(layer, lines)| layer.exposed(lines, &served)))
        .sum::<f64>()
        / window;
    // What prediction had done by each miss's request, as shares of the
    // window's misses.
    let total = |name: &str| {
        detail["experts"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|layer| layer[name].as_f64())
            .sum::<f64>()
    };
    let misses = total("misses");
    let share = |name: &str| (misses > 0.).then(|| total(name) / misses);
    // The model is judged against the run, so its σ is measured apart from
    // the model's exposure: the all-hit steps' mean less `C`.
    let all_hit = window_steps(detail)?
        .iter()
        .filter(|step| step["expert_step"]["misses"] == json!(0))
        .map(|step| step["interval_ms"].as_f64().ok_or("a step's interval"))
        .collect::<Result<Vec<_>, _>>()?;
    let overhead_ms = (!all_hit.is_empty())
        .then(|| all_hit.iter().sum::<f64>() / all_hit.len() as f64 - resident_ms);
    // The step is `C + σ` and what its misses expose, on `C` alone where
    // `σ` is unmeasured (`overhead` none says so).
    let step_seconds = resident_ms / MS + overhead_ms.map_or(0., |ms| ms / MS) + exposed;
    let measured_ms = detail["window"]["wall_ms_per_step"]
        .as_f64()
        .ok_or("the window's mean step")?;
    let measured = concurrency as f64 / (measured_ms / MS);
    let predicted = concurrency as f64 / step_seconds;
    let model = json!({
        "measured_tok_s": measured,
        "predicted_tok_s": predicted,
        "ratio": measured / predicted,
        "window_steps": layer_steps.len(),
        "all_hit_steps": all_hit.len(),
        "misses_per_step": misses / window,
        "miss_shares": {
            "hosted": share("hosted"),
            "reading": share("reading"),
            "unanticipated": share("unanticipated"),
        },
        "lead_layers_per_request": (total("led") > 0.).then(|| total("lead") / total("led")),
        "step_ms": {
            "compute": resident_ms,
            "exposed": exposed * MS,
            "total": step_seconds * MS,
        },
        "figures": {
            "latency_ms": served.latency_seconds() * MS,
            "seconds_per_byte": served.seconds_per_byte(),
            // Each streamed layer's lines the cost function priced it along.
            "lines": lines.iter().collect::<Vec<_>>(),
            "overhead_ms": overhead_ms,
        },
        "provenance": {
            "resident": { "result": path, "wall_ms_per_step": resident_ms },
        },
    });
    let step = ModelStep { overhead_seconds: overhead_ms.map(|ms| ms / MS), exposed, total_seconds: step_seconds };
    Ok((model, step))
}

/// The model's step by term, in seconds: its `σ` (none unmeasured), what
/// its layers' misses expose, by part, and its whole step.
#[derive(Clone, Copy, Debug)]
struct ModelStep {
    overhead_seconds: Option<f64>,
    exposed: f64,
    total_seconds: f64,
}

/// The resident load's `phase` cell at `context` and `concurrency`
/// (`time.resident_decode`, a resident `time` result), with the result it
/// is of.
fn resident_cell<'a>(time: &'a TimeOptions, phase: &str, context: usize, concurrency: usize) -> Result<(&'a std::path::Path, Value), Unjudged> {
    let path = time
        .resident_decode
        .as_deref()
        .ok_or_else(|| unavailable("no resident result was named (--resident-decode)"))?;
    let resident: Value = std::fs::read_to_string(path)
        .map_err(|error| error.to_string())
        .and_then(|text| serde_json::from_str(&text).map_err(|error| error.to_string()))
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let cell = resident["measurements"]["speed"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|speed| speed["phase"] == phase && speed["context"] == json!(context) && speed["concurrency"] == json!(concurrency))
        .ok_or_else(|| unavailable(format!("{} has no {phase} at context {context}", path.display())))?;
    Ok((path, cell.clone()))
}

/// The resident load's decode cell at `context` and `concurrency`, with the
/// result it is of and its window's mean step in ms: the streamed cell's
/// `C`.
fn resident_decode(time: &TimeOptions, context: usize, concurrency: usize) -> Result<(&std::path::Path, Value, f64), Unjudged> {
    let (path, cell) = resident_cell(time, "decode", context, concurrency)?;
    let step_ms = cell["detail"]["window"]["wall_ms_per_step"]
        .as_f64()
        .ok_or_else(|| unavailable(format!("{}'s decode records no window", path.display())))?;
    Ok((path, cell, step_ms))
}

/// The read line of a decode cell's own requests of the file served
/// (`RequestReads::line`); why it has none.
fn served_line(detail: &Value) -> Result<magnitude_executor::assessment::ReadLine, Unjudged> {
    serde_json::from_value::<RequestReads>(detail["reads"]["served"].clone())
        .map_err(|error| format!("the served file's reads: {error}"))?
        .line()
        .map_err(|error| unavailable(format!("the served file's read line: {error}")))
}

/// Each streamed layer's lines the load measured, in decoder order.
fn expert_lines(engine: &Engine) -> Result<std::sync::Arc<[magnitude_executor::assessment::ExpertLines]>, Unjudged> {
    match engine.domain.expert_lines() {
        Some(magnitude_executor::assessment::Priced::Measured(lines)) => {
            let costs = engine.domain.expert_costs().ok_or("a streamed cell's load streams no experts")?;
            Ok(lines.iter().zip(costs.landings).map(|(lines, landing)| lines.landed_by(landing)).collect())
        }
        Some(magnitude_executor::assessment::Priced::Pending(why) | magnitude_executor::assessment::Priced::Unavailable(why)) => {
            Err(unavailable(format!("no expert lines: {why}")))
        }
        None => Err("a streamed cell's load streams no experts".into()),
    }
}

/// A decode cell's window: every measured step after the first, the steps
/// its mean step is of.
fn window_steps(detail: &Value) -> Result<&[Value], String> {
    let steps = detail["steps"].as_array().ok_or("measured steps")?;
    steps.get(1..).ok_or_else(|| "a window of no step".into())
}

/// The window's steps' streamed layers, each step's as the watcher counted
/// them ([`magnitude_executor::assessment::LayerStep`]).
fn window_layer_steps(detail: &Value) -> Result<Vec<Vec<magnitude_executor::assessment::LayerStep>>, String> {
    window_steps(detail)?
        .iter()
        .map(|step| serde_json::from_value(step["expert_step"]["layers"].clone()))
        .collect::<Result<_, _>>()
        .map_err(|error| format!("each step's per-layer misses: {error}"))
}

/// The launches of `entries` (every launch of each) in an attribution
/// ([`crate::steps::attribution`]): the entries named, their device time
/// per step and their timed launches per step, none where none launched.
struct Launches<'a> {
    named: Vec<&'a str>,
    ms: f64,
    timed: f64,
}

fn entry_launches<'a>(attribution: &'a Value, entries: &[&str]) -> Result<Launches<'a>, String> {
    let rows = attribution["entries"].as_array().ok_or("the attribution has no entries")?;
    let mut launches = Launches { named: Vec::new(), ms: 0., timed: 0. };
    for row in rows {
        let entry = row["entry"].as_str().ok_or("an attributed entry without its name")?;
        if !entries.contains(&entry.split('#').next().expect("a split has a first part")) {
            continue;
        }
        launches.ms += row["device_ms_per_step"].as_f64().ok_or("an entry's device time")?;
        launches.timed += row["timed_per_step"].as_f64().ok_or("an entry's timed launches")?;
        launches.named.push(entry);
    }
    Ok(launches)
}

/// The router's launches of a decode cell's attribution among `entries`:
/// its device time per step and per launch.
fn router(attribution: &Value, entries: &[&str]) -> Result<Value, String> {
    let Launches { named, ms, timed } = entry_launches(attribution, entries)?;
    match timed > 0. {
        true => Ok(json!({ "entries": named, "us_per_step": ms * 1e3, "us_per_launch": ms * 1e3 / timed })),
        false => Err(format!("no timed launch of {entries:?}")),
    }
}

/// The streamed route's measured device time against the resident
/// router's, each from its cell's attribution (`--attribution-steps`): the
/// streamed layers' routing launches (`routed_route_streamed`,
/// `routed_select_streamed`: the resident router's routing, bit for bit,
/// then the step's resolve and the next layer's hints), and the resident
/// load's (`routed_route`, `routed_route_shared`, `routed_select`), over
/// every layer of a step. Absent, with why, where either cell's trace has
/// no per-entry times.
fn routers(cell: &Value, resident: &Value) -> Value {
    use magnitude_kernels::{routed_route, routed_route_shared, routed_route_streamed, routed_select, routed_select_streamed};
    use seismic::Entry;
    const STREAMED: [&str; 2] = [routed_route_streamed::Entry::NAME, routed_select_streamed::Entry::NAME];
    const RESIDENT: [&str; 3] = [routed_route::Entry::NAME, routed_route_shared::Entry::NAME, routed_select::Entry::NAME];
    let of = |cell: &Value, entries: &[&str], whose: &str| match &cell["detail"]["attribution"] {
        Value::Null => Err(format!("the {whose} cell's trace has no per-entry times (--attribution-steps 0)")),
        attribution => router(attribution, entries).map_err(|reason| format!("the {whose} cell's: {reason}")),
    };
    match (of(cell, &STREAMED, "streamed"), of(resident, &RESIDENT, "resident")) {
        (Ok(streamed), Ok(resident)) => json!({ "streamed": streamed, "resident": resident }),
        (Err(reason), _) | (_, Err(reason)) => json!({ "absent": reason }),
    }
}

/// What a streamed decode cell's bar is computed from beyond the cell
/// itself, each as the load measured it: each streamed layer's lines
/// (`ExpertLines`), each layer's expert's resident bytes and FLOPs a row
/// (`ExpertPass::expert`, none without the executor's costs), the host
/// CPU's own lines and, of a bank with no host tier (unified memory), its
/// storage device's ([`crate::machine`]). The bar's record keeps them.
#[derive(Debug, serde::Serialize)]
pub struct BarInputs {
    pub lines: Vec<magnitude_executor::assessment::ExpertLines>,
    pub experts: Option<Vec<(u64, u64)>>,
    pub machine: crate::machine::CpuMachine,
    pub storage: Option<Result<crate::machine::StorageMachine, String>>,
    /// The link's ceiling, a byte's seconds (`Ceilings::link`), where the bank
    /// has a host tier.
    pub link: Option<Result<f64, String>>,
    /// The overlap efficiency `η` ([`crate::bar`]) the resident decode
    /// measured on the host ([`overlap`]), where it did.
    pub overlap: Option<f64>,
    /// The bound the load holds its weights to, where one was set.
    pub weight_bound: Option<magnitude_executor::WeightBound>,
}

impl BarInputs {
    /// The load's inputs, the machine's lines measured as the run loaded.
    fn of(
        engine: &Engine,
        machine: &crate::machine::CpuMachine,
        storage: &Option<Result<crate::machine::StorageMachine, String>>,
        link: Option<Result<f64, String>>,
        weight_bound: Option<magnitude_executor::WeightBound>,
        resident: &Value,
    ) -> Result<Self, Unjudged> {
        let lines = expert_lines(engine)?.to_vec();
        let experts = match engine.domain.expert_pass() {
            Some(magnitude_executor::assessment::Priced::Measured(pass)) => Some((0..lines.len()).map(|layer| pass.expert(layer)).collect()),
            _ => None,
        };
        Ok(Self { lines, experts, machine: machine.clone(), storage: storage.clone(), link, overlap: overlap_of(resident), weight_bound })
    }

    /// The link's ceiling, a byte's seconds, or why it is unmeasured.
    fn link_ceiling(&self) -> Result<f64, String> {
        match &self.link {
            Some(Ok(byte)) => Ok(*byte),
            Some(Err(reason)) => Err(format!("the link's ceiling: {reason}")),
            None => Err("the link's ceiling was not measured".into()),
        }
    }

    /// The storage device's lines, or why they are unmeasured.
    fn storage(&self) -> Result<&crate::machine::StorageMachine, String> {
        match &self.storage {
            Some(Ok(storage)) => Ok(storage),
            Some(Err(reason)) => Err(format!("the storage device's lines: {reason}")),
            None => Err("the storage device's lines were not measured".into()),
        }
    }

    /// Each streamed layer's lines at machine peak ([`crate::bar::peak_lines`]);
    /// unified memory's, without a host tier, as measured but for its
    /// landing, our own code's. Why a line the bar needs is unmeasured.
    fn bar_lines(&self) -> Result<Vec<magnitude_executor::assessment::ExpertLines>, String> {
        let mut peak = Vec::with_capacity(self.lines.len());
        for (layer, lines) in self.lines.iter().enumerate() {
            if lines.host.memory.is_none() {
                peak.push(magnitude_executor::assessment::ExpertLines { host: magnitude_executor::assessment::HostLines { landing: None, ..lines.host }, ..*lines });
                continue;
            }
            let of = |reason: &str| format!("streamed layer {layer}: {reason}");
            let handshake = lines.host.cpu.ok_or_else(|| of("no handshake: the layer has no CPU site, or its pool's lines are unmeasured"))?.handshake;
            let (bytes, flops) = self.experts.as_ref().ok_or_else(|| of("no expert's bytes and FLOPs: the executor's costs are unmeasured"))?[layer];
            let byte = self.link_ceiling().map_err(|reason| of(&reason))?;
            let link = magnitude_executor::assessment::Nanoseconds::of(bytes as f64 * byte).map_err(|error| of(&error.to_string()))?;
            let unmeasured = |what: &str, reason: &String| of(&format!("the CPU's {what}: {reason}"));
            let rate = self.machine.int8_ops_per_second.as_ref().map_err(|reason| unmeasured("int8 rate", reason))?;
            peak.push(crate::bar::peak_lines(lines, flops as f64 / rate, handshake, link).map_err(|error| of(&error.to_string()))?);
        }
        Ok(peak)
    }
}

/// The overlap efficiency `η` a resident decode cell measured
/// ([`overlap`]), none where it did not: at most 1, since a read beside
/// compute never speeds it and a measure above 1 is the measure's noise.
fn overlap_of(resident: &Value) -> Option<f64> {
    resident["detail"]["overlap"]["eta"].as_f64().map(|eta| eta.min(1.))
}

/// The storage device's raw read line at its peak `P`: one read of a record
/// alone (`StorageMachine::first_seconds`, nothing of our read path) less its
/// bytes at `P` (`λ = t₁ − b/P`).
fn raw_read(storage: &crate::machine::StorageMachine) -> Result<magnitude_executor::assessment::ReadLine, String> {
    let byte_seconds = 1. / storage.peak_bytes_per_second;
    let latency = (storage.first_seconds - storage.bytes as f64 * byte_seconds).max(0.);
    magnitude_executor::assessment::ReadLine::new(latency, byte_seconds).map_err(|error| error.to_string())
}

/// The storage device's sustained seconds a byte, `1/S`: what it holds once
/// any burst has drained, at most its peak.
fn sustained(storage: &crate::machine::StorageMachine) -> f64 {
    1. / storage.sustained_bytes_per_second.min(storage.peak_bytes_per_second)
}

/// The readout's part of a resident decode cell's step, `h`: its attributed
/// `readout_*` and `sample_*` entries' device time a step, which run after
/// the last streamed layer. Why it is unknown: the cell attributed no entry.
fn readout_head(resident: &Value) -> Result<f64, String> {
    let entries = resident["detail"]["attribution"]["entries"].as_array().ok_or("the resident cell attributed no entry (--attribution-steps 0)")?;
    let mut ms = 0.;
    for row in entries {
        let entry = row["entry"].as_str().ok_or("an attributed entry without its name")?;
        if entry.starts_with("readout_") || entry.starts_with("sample_") {
            ms += row["device_ms_per_step"].as_f64().ok_or("an entry's device time")?;
        }
    }
    Ok(ms / MS)
}

/// Each read of `reads`' compulsory misses: the experts it names that
/// neither `held` held as the trace began nor any read before it named,
/// which any bank misses however it is run.
fn compulsory_reads(held: &[Vec<usize>], reads: &[magnitude_executor::RoutedRead]) -> Vec<u64> {
    let mut seen = held.iter().map(|experts| experts.iter().copied().collect::<std::collections::HashSet<_>>()).collect::<Vec<_>>();
    reads.iter().map(|read| read.experts.iter().filter(|&&expert| seen[read.layer].insert(expert)).count() as u64).collect()
}

/// A streamed decode cell's performance bar ([`crate::bar::Ceiling`]): the
/// best step any implementation could reach at the load's cap on this machine,
/// beside the step measured and the model's (`model`, none where it is
/// unavailable). Its inputs, each with where it came from:
/// - `C`: the resident load's decode at the same context and concurrency
///   (`resident_ms`, the `--resident-decode` result's window mean step);
/// - each step's misses, three counts: the oracle (Bélády) bank of the
///   load's records and the bank's own ranking (the online frontier's),
///   each group's records shared by its layers as the bank
///   shares them, each replayed (`replay_routing`, one replay) over the
///   cell's own routing records (warm and measured steps) from what the
///   bank held as decode began; and the run's live misses; the window's
///   steps priced, the gap in misses split into the design's (ranking over
///   Bélády) and the implementation's (live over ranking);
/// - each layer's predictable share: on unified memory the file misses a
///   hint had read or begun to read by their request (a discrete GPU's
///   tier misses are priced from host memory, predicted or not);
/// - `inputs` ([`BarInputs`]) and the run's read line.
///
/// It records the decode terms: the bar, the model and the measured
/// step, in ms a step and tok/s; the bar's terms; per layer the oracle's
/// misses against the run's and the predictable share; the CPU's line and
/// the handshake; the tier's wait; the cores the CPU's line was measured
/// over; the gaps, measured less the model (the implementation's) and the
/// model less the bar (the design's, term by term); and the routers' device
/// time ([`routers`]). Why it is unavailable, when an input is absent.
fn streamed_bar(
    cell: &Value,
    resident: &Value,
    resident_ms: f64,
    concurrency: usize,
    inputs: &BarInputs,
    model: Option<ModelStep>,
) -> Result<Value, Unjudged> {
    use crate::bar::OracleLayer;
    use magnitude_executor::{EvictionRecency, replay_routing};
    let detail = &cell["detail"];
    let bank = &detail["expert_bank"];
    let BankShape { records, layers, experts, held } = BankShape::of(bank)?;
    let read = served_line(detail)?;
    let unified = inputs.lines.iter().all(|lines| lines.host.memory.is_none());
    // A unified bank's records are read raw at the storage device's peak; a
    // discrete bar reads no file.
    let raw = match unified {
        true => inputs.storage().and_then(raw_read).map_err(unavailable)?,
        false => read,
    };
    let bar_lines = inputs.bar_lines().map_err(unavailable)?;
    let overlap = if unified { inputs.overlap.unwrap_or(1.) } else { 1. };
    let sustained = match unified {
        true => sustained(inputs.storage().map_err(unavailable)?),
        false => read.seconds_per_byte(),
    };
    let head = readout_head(resident).map_err(unavailable)?;
    // The frontier's prediction: every route of a token known once it exists,
    // none wasted.
    let full = magnitude_executor::assessment::ReadAhead { precision: 1., lead: magnitude_executor::assessment::Lead::Token };
    let priced_ahead = |ahead, read, steps: &[Vec<OracleLayer>]| {
        let inputs = magnitude_executor::assessment::StepInputs { compute: resident_ms / MS, head, overlap, read, sustained, ahead, lines: &bar_lines };
        crate::bar::step(&inputs, steps)
    };
    let priced = |read, steps: &[Vec<OracleLayer>]| priced_ahead(full, read, steps);
    let layer_steps = window_layer_steps(detail)?;
    // Each layer's predictable share, as the run's prediction measured it.
    let predictable = (0..layers.len())
        .map(|layer| match unified {
            true => {
                let at = &detail["experts"][layer];
                let count = |name: &str| at[name].as_f64().ok_or(format!("streamed layer {layer}'s {name}"));
                let misses = count("misses")?;
                Ok(if misses > 0. { count("reading")? / misses } else { 0. })
            }
            false => Ok(0.),
        })
        .collect::<Result<Vec<_>, String>>()?;
    let (warm, measured) = expert_steps(detail)?;
    let reads = decode_reads(&warm, &measured).map_err(unavailable)?;
    // The window's reads (the measured steps' after the first) as two
    // replays of the run's own routing over the bank's records leave them:
    // Bélády's at the bank's records and the bank's own ranking's (the
    // online frontier's).
    let first = (warm.len() + 1) * layers.len();
    // Every miss the window's steps take counts, the compulsory ones priced
    // like the rest, since they are what these steps could reach. Bélády
    // within the window is exact without the reads past
    // it, which no miss the window counts depends on.
    let replayed = |recency| -> Result<Vec<Vec<OracleLayer>>, Unjudged> {
        let replayed = replay_routing(&records, &layers, experts, &held, &reads, recency)?;
        Ok(reads[first..]
            .chunks(layers.len())
            .zip(replayed.misses[first..].chunks(layers.len()))
            .map(|(reads, misses)| {
                reads
                    .iter()
                    .zip(misses)
                    .zip(&predictable)
                    .map(|((read, &misses), &predictable)| OracleLayer {
                        routed: read.experts.len() as u64,
                        choices: read.choices,
                        misses,
                        predictable,
                    })
                    .collect::<Vec<_>>()
            })
            .collect())
    };
    let (steps, ranked) = (replayed(EvictionRecency::NextRead)?, replayed(EvictionRecency::Reuse)?);
    let window = steps.len() as f64;
    let per_layer = |of: &dyn Fn(usize) -> f64| (0..layers.len()).map(of).collect::<Vec<_>>();
    let misses_of = |steps: &[Vec<OracleLayer>]| per_layer(&|layer| steps.iter().map(|step| step[layer].misses).sum::<u64>() as f64 / window);
    let (oracle_misses, ranked_misses) = (misses_of(&steps), misses_of(&ranked));
    // The run's live misses, tier and file, each window step's layers'.
    let live = layer_steps
        .iter()
        .map(|run| run.iter().map(|run| run.hosted + run.misses).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let actual_misses = per_layer(&|layer| live.iter().map(|step| step[layer]).sum::<u64>() as f64 / window);
    // Of Bélády's misses a step, the compulsory ones ([`compulsory_reads`]):
    // what no bank of these records avoids in this window, the rest capacity.
    let compulsory = compulsory_reads(&held, &reads)[first..].iter().sum::<u64>() as f64 / window;
    let belady = oracle_misses.iter().sum::<f64>();
    let window_state = json!({ "compulsory_per_step": compulsory, "capacity_per_step": belady - compulsory });
    // On unified memory the frontier hints every layer but the token's
    // first; a discrete cell's tier misses are read from host memory.
    let oracle = |steps: &[Vec<OracleLayer>]| -> Vec<Vec<OracleLayer>> {
        steps
            .iter()
            .map(|step| {
                step.iter()
                    .enumerate()
                    .map(|(layer, oracle)| OracleLayer { predictable: if !unified { oracle.predictable } else if layer == 0 { 0. } else { 1. }, ..*oracle })
                    .collect()
            })
            .collect()
    };
    // The ceiling (`crate::bar`): Bélády at the most records the bound holds,
    // every read ahead, through what the misses cross.
    let group_bytes = (0..records.len())
        .map(|group| layers.iter().position(|&of| of == group).map(|layer| inputs.lines[layer].host.record_bytes).ok_or("a bank group of no layer"))
        .collect::<Result<Vec<_>, _>>()?;
    let owned = serde_json::from_value::<Vec<(String, u64)>>(json!(bank["cap_owners"].as_array().ok_or("the bank's cap owners")?.iter().map(|owner| (&owner["owner"], &owner["bytes"])).collect::<Vec<_>>()))
        .map_err(|error| format!("the bank's cap owners: {error}"))?;
    let every_step = owned.iter().filter(|(owner, _)| owner.contains("weights")).map(|(_, bytes)| bytes).sum::<u64>();
    if every_step == 0 {
        return Err(unavailable("no owner of the weights beside the bank"));
    }
    // The bank's records lie on the device alone, so what a host tier
    // holds is never theirs: a cap counts the tier's bytes among its own,
    // a device room never does, and an unbounded load's owners hold it
    // beside the device's.
    let tier = owned.iter().filter(|(owner, _)| owner.starts_with("host tier")).map(|(_, bytes)| bytes).sum::<u64>();
    let device = match inputs.weight_bound {
        Some(magnitude_executor::WeightBound::Cap(cap)) => cap.saturating_sub(tier),
        Some(magnitude_executor::WeightBound::Room(room)) => room,
        None => owned.iter().map(|(_, bytes)| bytes).sum::<u64>() - tier,
    };
    let capacity = crate::bar::records_at_cap(&records, &group_bytes, device.saturating_sub(every_step));
    let at_cap = replay_routing(&capacity, &layers, experts, &held, &reads, EvictionRecency::NextRead)?;
    let record_bytes = inputs.lines.iter().map(|lines| lines.host.record_bytes).collect::<Vec<_>>();
    let moved_byte = match unified {
        true => sustained,
        false => inputs.link_ceiling().map_err(unavailable)?,
    };
    let ceiling = crate::bar::Ceiling::of(resident_ms / MS, overlap, moved_byte, &record_bytes, &at_cap.misses[first..]);
    // The design frontier: the step model at the ranking's misses, every
    // other input ideal. A diagnostic, not a bar.
    let frontier = priced(raw, &oracle(&ranked)).total;
    let frontier_ms = frontier * MS;
    // What the predictor reaches: the frontier's misses, as many of them
    // hinted as the run's hints saved, the rest on demand, at the run's lead
    // and precision (`HintCounts::ahead`). The frontier assumes every route
    // known at the token's start and none wasted, which no predictor from
    // the routes' inputs is; this is the bar for implementation, and the gap
    // between the two is prediction's.
    let reachable_ms = unified.then(|| {
        use magnitude_executor::assessment::{Anticipated, HintCounts};
        let counts = layer_steps.iter().fold(HintCounts::default(), |sum, step| sum.plus(HintCounts::of(step)));
        let recall = per_layer(&|layer| layer_steps.iter().fold(Anticipated::default(), |sum, step| sum.plus(step[layer].anticipated())).recall());
        let steps = ranked
            .iter()
            .map(|step| step.iter().enumerate().map(|(layer, oracle)| OracleLayer { predictable: recall[layer], ..*oracle }).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        priced_ahead(counts.ahead(), raw, &steps).total * MS
    });
    let measured_ms = detail["window"]["wall_ms_per_step"].as_f64().ok_or("the window's mean step")?;
    let (bar_ms, moved_ms) = (ceiling.seconds() * MS, ceiling.moved * MS);
    let exposed = bar_ms / MS - resident_ms / MS;
    // The gap's split: the same step at the ranking's misses, the run's,
    // and our served line; each request's latency is within them.
    let ceiling = match unified {
        false => Value::Null,
        true => {
            let actual = oracle(
                &steps
                    .iter()
                    .zip(&live)
                    .map(|(step, live)| step.iter().zip(live).map(|(oracle, &misses)| OracleLayer { misses, ..*oracle }).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
            );
            let at_actual = priced(raw, &actual).total;
            // Each line by one record's time alone, so no latency fitted
            // under load is charged beside its bytes.
            let record_bytes = inputs.lines[0].host.record_bytes as f64;
            let per_record = |line: magnitude_executor::assessment::ReadLine| {
                magnitude_executor::assessment::ReadLine::new(0., (line.latency_seconds() + line.transfer_seconds(record_bytes)) / record_bytes).map_err(|error| unavailable(error.to_string()))
            };
            let at_actual_record = priced(per_record(raw)?, &actual).total;
            let at_served = priced(per_record(read)?, &actual).total;
            json!({
                "overlap": match inputs.overlap {
                    Some(eta) => json!(eta),
                    None => json!({ "unmeasured": "the ceiling is taken at full overlap (η = 1), the strict bound" }),
                },
                "ceiling_ms": bar_ms,
                "schedule_ms": ceiling.compute * MS,
                "sustained_ms": ceiling.moved * MS,
                "gap_split_ms": {
                    "ranking_and_schedule": frontier * MS - bar_ms,
                    "misses_implementation": (at_actual - frontier) * MS,
                    "read_path_below_raw": (at_served - at_actual_record) * MS,
                    "exposure": measured_ms - (at_actual + at_served - at_actual_record) * MS,
                },
            })
        }
    };
    let tok_s = |ms: f64| concurrency as f64 / (ms / MS);
    // Each distinct value of a line over the streamed layers, in µs.
    let distinct_of = |lines: &[magnitude_executor::assessment::ExpertLines], of: fn(&magnitude_executor::assessment::ExpertLines) -> Option<f64>| {
        let mut values = lines.iter().filter_map(of).map(|ns| ns * 1e-3).collect::<Vec<_>>();
        values.sort_by(|a, b| a.partial_cmp(b).expect("a line is a number"));
        values.dedup();
        values
    };
    let distinct = |of: fn(&magnitude_executor::assessment::ExpertLines) -> Option<f64>| distinct_of(&inputs.lines, of);
    // The tier's wait: the host's notice to acknowledgement of the
    // window's requests, a step.
    let tier_wait_ms = detail["experts"].as_array().map(|layers| {
        layers.iter().filter_map(|layer| layer["parked_ms"].as_f64()).sum::<f64>() / window
    });
    let model_ms = model.map(|model| model.total_seconds * MS);
    let gaps = match model {
        None => json!({ "unavailable": "the model is unavailable" }),
        Some(model) => json!({
            "implementation_ms": measured_ms - model.total_seconds * MS,
            "design_ms": {
                "route": model.overhead_seconds.map(|seconds| seconds * MS),
                // What the model's misses expose past the bar's.
                "exposed": (model.exposed - exposed) * MS,
                "total": model.total_seconds * MS - bar_ms,
            },
        }),
    };
    Ok(json!({
        "form": if unified { "unified" } else { "discrete" },
        "step_ms": {
            "measured": measured_ms,
            "model": model_ms,
            "bar": bar_ms,
        },
        "tok_s": { "measured": tok_s(measured_ms), "model": model_ms.map(tok_s), "bar": tok_s(bar_ms) },
        "window": window_state,
        "bar_ms": {
            "resident": resident_ms,
            "exposed": exposed * MS,
            "moved": moved_ms,
            "records_at_cap": capacity,
        },
        "unified": ceiling,
        "frontier_ms": frontier_ms,
        "frontier_tok_s": tok_s(frontier_ms),
        "reachable_ms": reachable_ms,
        "misses_per_step": {
            "belady": belady,
            "ranking": ranked_misses.iter().sum::<f64>(),
            "live": actual_misses.iter().sum::<f64>(),
            "design": (ranked_misses.iter().sum::<f64>() - oracle_misses.iter().sum::<f64>()),
            "implementation": (actual_misses.iter().sum::<f64>() - ranked_misses.iter().sum::<f64>()),
            "by_layer": { "belady": oracle_misses, "ranking": ranked_misses, "live": actual_misses },
        },
        "predictable_by_layer": predictable,
        "tier_wait_ms_per_step": tier_wait_ms,
        "window_steps": steps.len(),
        "gaps": gaps,
        "router": routers(cell, resident),
        "inputs": {
            "resident_ms": resident_ms,
            "records": records,
            "served_line": { "latency_ms": read.latency_seconds() * MS, "bytes_per_second": 1. / read.seconds_per_byte() },
            "storage_raw": unified.then(|| json!({ "latency_ms": raw.latency_seconds() * MS, "bytes_per_second": 1. / raw.seconds_per_byte() })),
            "memory_us_per_record": distinct(|lines| lines.host.memory.map(|line| line.get())),
            "link_us_per_record": distinct(|lines| Some(lines.gpu.link.weights.get())),
            "link_ceiling_us_per_record": distinct_of(&bar_lines, |lines| lines.host.memory.map(|_| lines.gpu.link.weights.get())),
            "cpu_expert_us": distinct(|lines| lines.host.cpu.map(|cpu| cpu.expert.one.get())),
            "handshake_us": distinct(|lines| lines.host.cpu.map(|cpu| cpu.handshake.get())),
            "cores_allowed": inputs.machine.threads,
            "load": inputs,
        },
    }))
}

/// What assessment estimates from, read before the load: the structure the
/// service assesses on, this machine's measurements, and the model prepared
/// on them.
struct Assessment {
    structure: magnitude_engine::assessment::AssessmentStructure,
    measurements: magnitude_engine::assessment::Measurements,
    prepared: magnitude_engine::assessment::PreparedModelAssessment,
}

/// The assessment the run's loads are judged by, as the service
/// assesses the model on this machine; unavailable without a kernel cache.
fn assessment(artifacts: &Artifacts, options: &EngineOptions) -> Result<Assessment, Unjudged> {
    use magnitude_engine::assessment::{
        AssessmentStructure, ModelPackagePaths, prepare_model_assessment,
    };
    use magnitude_engine::options::{ModelMethod, ModelPolicy, standard_service_limits};
    let cache = options
        .cache_dir
        .as_ref()
        .ok_or_else(|| unavailable("no kernel cache holds the records (--cache-dir)"))?;
    let catalog = seismic::Devices::discover(&[]).map_err(|error| error.to_string())?;
    // The resident plan's: assessment's `C`.
    let policy = ModelPolicy {
        kv_codec: options.kv_codec,
        numerical_profile: options.numerical_profile.clone(),
        ..ModelPolicy::default()
    };
    let topology = catalog.topology();
    let reserves = magnitude_executor::platform::MemoryReserves::of_machine(Some((cache.as_path(), &topology)));
    let structure = AssessmentStructure::discover(&catalog, options.device, Some(cache.as_path()), reserves, policy, standard_service_limits())
        .map_err(|error| error.to_string())?;
    let storage = magnitude_artifacts::model_store(&artifacts.target).map_err(|error| format!("{}: {error}", artifacts.target.display()))?;
    let measurements = structure.stored_measurements(cache, &storage);
    let package = ModelPackagePaths {
        target: artifacts.target.clone(),
        projector: None,
        draft: None,
        method: ModelMethod::Plain,
    };
    let prepared = prepare_model_assessment(&package, &structure).map_err(|error| error.to_string())?;
    Ok(Assessment {
        structure,
        measurements,
        prepared,
    })
}

/// The estimate of a streamed decode cell at `context` and `concurrency`: what
/// assessment estimates its decode at, at the load's own slots and at every
/// other, beside the host tier the window held ([`window_host_slots`]), from
/// the plan's structure and the machine's stored figures alone, against the
/// window's measured mean step, each per sequence. Why it is unavailable,
/// when an input is absent: the model unsupported here, this machine's
/// costs or streamed figures unmeasured, a plan that holds every expert
/// (nothing streams to estimate), no estimate at a count of slots.
fn estimated(
    assessment: &Assessment,
    engine: &Engine,
    context: usize,
    concurrency: usize,
    cell: &Value,
) -> Result<Value, Unjudged> {
    use magnitude_engine::assessment::{Measurements, Performance, PreparedModelAssessment, streamed_estimate};
    use magnitude_executor::assessment::{decode_pricing, step_seconds, Priced};
    let Assessment {
        structure,
        measurements,
        prepared,
    } = assessment;
    let PreparedModelAssessment::Planned { facts, execution, .. } = prepared else {
        return Err(unavailable("the model is unsupported here"));
    };
    let Measurements::Measured { costs: machine, streamed, .. } = measurements else {
        return Err(unavailable("this machine's costs are not measured"));
    };
    let figures = match streamed {
        Priced::Measured(figures) => figures,
        Priced::Pending(why) | Priced::Unavailable(why) => return Err(unavailable(format!("the streamed figures: {why}"))),
    };
    let costs = engine.domain.expert_costs().ok_or("the load streams no experts")?;
    let own = costs.slots;
    let context = u32::try_from(context).map_err(|error| error.to_string())?;
    let request = magnitude_executor::assessment::AssessmentRequest {
        context_limit: facts.context_limit,
        performance_depths: vec![context],
        sequences: u32::try_from(concurrency).map_err(|error| error.to_string())?,
        reserves: structure.reserves,
    };
    // The window's host tier as the load held it, step by step: its misses
    // are priced where that tier served them, not where a plan's tier
    // would (the mean of its steps' host slots; none without a tier).
    let host = window_host_slots(cell)?;
    // The bank the plan streams its experts through, whose slots are priced
    // at each count: none where the plan holds them whole.
    let bank = execution.draft().load().expert_bank().filter(|bank| !bank.whole()).ok_or_else(|| unavailable("the plan holds its experts whole"))?;
    let mut by_slots = Vec::new();
    for slots in 1..=costs.experts {
        let held = magnitude_executor::assessment::HeldSlots { slots, host };
        let performance = streamed_estimate(machine, figures, execution, bank, &request, held)
            .map_err(|error| error.to_string())?;
        let Performance::Estimated(estimates) = performance else {
            return Err(unavailable(format!("at {slots} slots: {performance:?}")));
        };
        let estimate = estimates.iter().find(|estimate| estimate.context_tokens == context);
        by_slots.push((slots, estimate.ok_or("no estimate at the context")?.tokens_per_second));
    }
    let estimated = by_slots
        .iter()
        .find(|&&(slots, _)| slots == own)
        .map(|&(_, tok_s)| tok_s)
        .ok_or("no estimate at the load's slots")?;
    let measured = MS
        / cell["detail"]["window"]["wall_ms_per_step"]
            .as_f64()
            .ok_or("the window's mean step")?;
    // The resident step as the estimate prices it: at the request's rows and
    // the context.
    let pricing = decode_pricing(execution.draft(), machine);
    let resident = step_seconds(execution.demand(), &pricing, u64::from(request.sequences), context)
        .map_err(|error| error.to_string())?
        .total();
    Ok(json!({
        "slots": own,
        "tok_s": estimated,
        "measured_tok_s": measured,
        "ratio": measured / estimated,
        "read_line": { "latency_ms": figures.read.latency_seconds() * MS, "seconds_per_byte": figures.read.seconds_per_byte() },
        "resident_ms": resident * MS,
        "by_slots": by_slots
            .iter()
            .map(|&(slots, tok_s)| json!({ "slots": slots, "tok_s": tok_s }))
            .collect::<Vec<_>>(),
        "host_slots": host,
    }))
}

/// The host slots a streamed cell's window held: the mean, to the nearest
/// slot, of its steps' (each as it completed, `expert_step`), the steps
/// the window's mean is of; none without a host tier.
fn window_host_slots(cell: &Value) -> Result<usize, String> {
    let held = window_steps(&cell["detail"])?
        .iter()
        .map(|step| match &step["expert_step"]["host_slots"] {
            Value::Null => Ok(0),
            slots => slots.as_u64().ok_or("a step's host slots"),
        })
        .collect::<Result<Vec<_>, _>>()?;
    match held.len() {
        0 => Err("a window of no step".into()),
        count => Ok((held.iter().sum::<u64>() as f64 / count as f64).round() as usize),
    }
}

/// Processor time this process has used, on every thread, in seconds.
pub fn process_cpu_seconds() -> f64 {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `clock_gettime` writes a whole `timespec` to the pointer it is
    // given; the process clock exists on every supported host.
    let read = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut time) };
    assert_eq!(read, 0, "the process has a processor-time clock");
    time.tv_sec as f64 + time.tv_nsec as f64 * 1e-9
}

/// Processor time each kind of this process's threads has used, in seconds:
/// the threads by name, a name's trailing index dropped (`expert-compute-3`
/// is `expert-compute`), so a pool counts as one. Read from `/proc`, where
/// a thread's ticks are kept; none elsewhere.
fn thread_cpu_seconds() -> Option<std::collections::BTreeMap<String, f64>> {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: `sysconf` reads a constant.
        let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as f64;
        let mut threads = std::collections::BTreeMap::<String, f64>::new();
        for task in std::fs::read_dir("/proc/self/task").ok()?.flatten() {
            let (Ok(name), Ok(stat)) =
                (std::fs::read_to_string(task.path().join("comm")), std::fs::read_to_string(task.path().join("stat")))
            else {
                continue;
            };
            // Past the parenthesized name: state is field 3, utime 14, stime 15.
            let fields = stat[stat.rfind(')')? + 2..].split(' ').collect::<Vec<_>>();
            let used = fields.get(11)?.parse::<f64>().ok()? + fields.get(12)?.parse::<f64>().ok()?;
            let kind = name.trim().trim_end_matches(|c: char| c.is_ascii_digit() || c == '-' || c == '_');
            *threads.entry(kind.to_owned()).or_default() += used / ticks;
        }
        Some(threads)
    }
    #[cfg(not(target_os = "linux"))]
    None
}

/// The process's cgroup's CPU quota throttling so far, (periods throttled,
/// µs throttled), from cgroup v2's `cpu.stat`; none without one.
fn throttled() -> Option<(u64, u64)> {
    let stat = std::fs::read_to_string("/sys/fs/cgroup/cpu.stat").ok()?;
    let field = |name: &str| {
        stat.lines()
            .find_map(|line| line.strip_prefix(name)?.strip_prefix(' ')?.trim().parse::<u64>().ok())
    };
    Some((field("nr_throttled")?, field("throttled_usec")?))
}

/// Each thread kind's processor time between `before` and `after`
/// ([`thread_cpu_seconds`]), in ms per step of `steps`, the busiest first:
/// what holds the process's cores.
fn thread_cpu_per_step(before: &std::collections::BTreeMap<String, f64>, after: &std::collections::BTreeMap<String, f64>, steps: usize) -> Value {
    let mut used = after
        .iter()
        .map(|(kind, &seconds)| (kind.clone(), (seconds - before.get(kind).copied().unwrap_or(0.)) * MS / steps as f64))
        .filter(|(_, ms)| *ms > 0.)
        .collect::<Vec<_>>();
    used.sort_by(|a, b| b.1.total_cmp(&a.1));
    Value::Array(used.into_iter().map(|(kind, ms)| json!({ "thread": kind, "cpu_ms_per_step": ms })).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A host running ahead of the device returns from each step's call
    /// 4.426 ms after it began while the device completes a step every
    /// 7.57 ms: the steps' times are the completions'
    /// intervals, 7.57 ms, never the calls' spans.
    #[test]
    fn a_step_is_timed_by_its_completions_not_its_hosts_span() {
        let (device, host) = (7.57e-3, 4.426e-3);
        // Each call begins once the one before returned, and returns
        // `host` after it began, but no sooner than the step before it
        // completed: the host is a step ahead.
        let ends = (0..8).map(|step| (step + 1) as f64 * device).collect::<Vec<_>>();
        let starts = ends.iter().map(|end| end - host).collect::<Vec<_>>();
        assert!(starts.iter().zip(&ends).all(|(start, end)| ((end - start) - host).abs() < 1e-12));
        let intervals = step_intervals(&ends);
        assert_eq!(intervals.len(), 7);
        assert!(intervals.iter().all(|interval| (interval - device).abs() < 1e-12), "{intervals:?}");
        assert!(step_intervals(&ends[..1]).is_empty());
    }

    /// A run that warmed on, or ran a window again, until its bank came to
    /// rest decoded the same tokens as one that did not; a selection within
    /// the steps every run takes differs.
    #[test]
    fn a_longer_warm_keeps_the_digest_of_the_same_tokens() {
        let selections = |tokens: &[u32]| tokens.iter().flat_map(|token| token.to_le_bytes()).collect::<Vec<_>>();
        let digest = selections_digest(&selections(&[1, 2, 3, 4]), 2, 2);
        assert_eq!(digest, selections_digest(&selections(&[1, 2, 3, 4, 5, 6]), 2, 2));
        assert_ne!(digest, selections_digest(&selections(&[1, 2, 3, 9, 5, 6]), 2, 2));
    }

    use magnitude_executor::{PromptGroupCost, PromptGroupLayer};

    fn layer(conveyed: u64, cold_bytes: u64, pulled: u64) -> PromptGroupLayer {
        PromptGroupLayer { union: 4, conveyed, cold_bytes, pulled, tier_bytes: pulled * 1_000, ..PromptGroupLayer::default() }
    }

    fn close(value: &Value, expected: f64) {
        let value = value.as_f64().expect("a number");
        assert!((value - expected).abs() < 1e-9, "{value} is not {expected}");
    }

    /// The groups' bar ([`crate::bar::PrefillBar`]) over each layer's
    /// totals across the groups at the ceilings, its terms, and the
    /// attributed prefill's achieved copy beside it; without an
    /// input a term needs, there is none.
    #[test]
    fn prompt_groups_are_barred_at_the_ceilings() {
        let group = |first, steps, layers| PromptGroupCost { first, steps, layers };
        let attributed = AttributedPrefill { copy_seconds: 4e-3, copied: 4 };
        let sample = GroupSample {
            groups: vec![group(1, 2, vec![layer(1, 1_000, 3), layer(0, 0, 4)]), group(3, 1, vec![layer(5, 5_000, 0), layer(0, 0, 1)])],
            measured_ms: 20.0,
            attributed: Ok(attributed),
        };
        let ceilings = Ceilings { convert: Ok(vec![1e-4, 2e-4]), link: Some(Ok(1e-6)) };
        // Reads 6 kB at 1 MB/s, 6 ms; resident 6.5 + 6 records' convert at
        // 0.1 ms; 8 kB pulled and 6 kB conveyed over the link at 1 µs a
        // byte, 14 ms.
        let rate = crate::bar::ReadRate { peak: 1e6, sustained: 1e6, burst: 0. };
        let bar = prompt_groups(&sample, Ok(rate), Ok(6.5e-3), &ceilings);
        assert_eq!(bar["groups"], 2);
        assert_eq!(bar["sizes"], json!([2, 1]));
        assert_eq!(bar["steps"], 3);
        close(&bar["bar_ms"], 14.0);
        close(&bar["ratio"], 14.0 / 20.0);
        close(&bar["bar_terms"]["read_ms"], 6.0);
        close(&bar["bar_terms"]["convert_ms"], 0.6);
        close(&bar["bar_terms"]["link_ms"], 14.0);
        close(&bar["ceilings"]["link_bytes_per_second"], 1e6);
        close(&bar["achieved"]["copy_us_per_record"], 1e3);
        close(&bar["resident_ms"], 6.5);
        close(&bar["read_rate"]["sustained_bytes_per_second"], 1e6);
        assert_eq!(bar["layers"][0]["cold_bytes"], 6_000);
        assert_eq!(bar["layers"][1]["union"], 8);

        let unrated = prompt_groups(&sample, Err("unmeasured".into()), Ok(6.5e-3), &ceilings);
        assert_eq!(unrated["bar_ms"]["unavailable"], "unmeasured");
        assert!(unrated["ratio"].is_null() && unrated["bar_terms"].is_null());
        let unresident = prompt_groups(&sample, Ok(rate), Err("no resident".into()), &ceilings);
        assert_eq!(unresident["bar_ms"]["unavailable"], "no resident");
    }
}
