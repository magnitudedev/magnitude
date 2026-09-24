//! V4 forward bench (native kernel program R4/R7).
//!
//! Drives the native target program through the executor domain, below the
//! service: no tokenizer, scheduler or HTTP. Tokens are forced (a fixed
//! pseudo-random sequence), so every cell is a pure forward measurement.
//!
//! Cells (the R3 V3 forward bench runs the same ones):
//! - `decode`: one sequence with `--context` tokens of history, then 16 warm
//!   and 32 measured greedy decode steps (median reported);
//! - `prefill`: fresh sequences of 32/128/512 tokens, one cold and three warm
//!   (median of warm); with `--prefill-history H`, the chunks instead follow
//!   H tokens of history on one sequence, back to back;
//! - `concurrent`: `--sequences N` sequences with `--context` history each,
//!   decoding together; aggregate tokens per second.
//!
//! Every measured step is traced per submission (command buffer / CUDA
//! batch): host time before the first encode, host encode time, device busy
//! and idle within the step, device idle between steps, and host time after
//! the device finished. A separate attribution pass per cell runs each launch
//! in its own timed unit and decomposes device time by entry (R7); those
//! steps are slower by construction and are not the cell's timing.
//!
//! `qualify` compares V4 logits against a llama.cpp KL-divergence base file
//! (D4 precision gate, Q1), with the loaded model's own D4 limits.
//!
//! ```text
//! forward_bench bench --model M.gguf --output out.json [--path native]
//!     [--storage-gib 16] [--cells decode,prefill,concurrent]
//!     [--context 256,4096,16384] [--prefill 32,128,512] [--prefill-history 0]
//!     [--sequences 1,2,4,8]
//!     [--warm 16] [--steps 32] [--attribution-steps 4]
//! forward_bench qualify --model M.gguf --reference REF_DIR --output out.json
//!     [--chunks N] [--label NAME] [--verify-width W]
//! ```
//!
//! Either mode takes `--cache-dir DIR`, the engine's kernel cache (compiled
//! kernels and tuning results); without it every load forms and tunes every
//! entry. Either mode takes `--kv-codec dense|affine-k8v4` (default
//! affine-k8v4, the engine default), the target history codec.
//!
//! Built with `--features pinned-tuning` (development only), either mode also
//! takes `--tuning-record FILE` or `--tuning-replay FILE` to pin tuned
//! configurations across runs for bit-exact comparisons (`tuning_pin`).
//! Built with `--features tuning-survey` (development only), either mode also
//! takes `--tuning-survey DIR` to record every admissible tuning
//! configuration's measurements for replaying the search (`tuning_survey`).

use magnitude_engine::{
    build_native_domain,
    composition::EngineConfiguration,
    options::{ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection, StoragePolicy},
};
use magnitude_model_contracts::DecoderGeometry;
use magnitude_model_state::KvCodec;
use magnitude_model_executor::{
    Demand, DomainError, ExecutionPath, ExecutorDomain, Operation, Outcome, PhysicalDecision,
    RequestId, Sampling, SelectSpec, Shaping, TokenId, WorkKind,
};
use magnitude_service::{
    domain::{self as service_domain, DomainFlight},
    ServiceLimits,
};
use seismic::{host_seconds, SubmissionTrace, TraceDetail, TracedSubmission};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Rows per prefill chunk when building history; the largest prefill class.
const PREFILL_ROWS: usize = 512;

fn main() {
    if let Err(error) = run() {
        eprintln!("forward_bench: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    #[cfg_attr(
        not(any(feature = "pinned-tuning", feature = "tuning-survey")),
        allow(unused_mut)
    )]
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    #[cfg(feature = "pinned-tuning")]
    let pin = tuning_pin::Pin::extract(&mut args)?;
    #[cfg(feature = "tuning-survey")]
    tuning_survey::extract(&mut args)?;
    let mut args = args.into_iter();
    let mode = args.next().ok_or("usage: forward_bench bench|qualify ...")?;
    let options = Options::parse(args)?;
    let outcome = match mode.as_str() {
        "bench" => bench(&options),
        "qualify" => qualify::run(&options),
        other => Err(format!("unknown mode {other:?}; expected bench or qualify")),
    };
    #[cfg(feature = "pinned-tuning")]
    let outcome = outcome.and(pin.finish());
    outcome
}

/// `--tuning-record FILE` / `--tuning-replay FILE`, only with the
/// development feature `pinned-tuning`
/// (`cargo run --release --example forward_bench --features pinned-tuning`).
///
/// A measurement tool for bit-exact comparisons between runs: tuning at load
/// may choose different configurations each run, so the baseline run records
/// the configuration chosen per entry and the comparison run replays exactly
/// those (it tunes nothing). Within the recording run every load reuses the
/// first choice per entry. See `magnitude_model_executor::pinned_tuning`.
#[cfg(feature = "pinned-tuning")]
mod tuning_pin {
    use magnitude_model_executor::pinned_tuning::{self, PinnedConfiguration, TuningPin};
    use serde_json::{json, Value};
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    pub(crate) struct Pin {
        record: Option<PathBuf>,
    }

    impl Pin {
        /// Remove the pin flags from `args` and install the pin they ask for.
        pub(crate) fn extract(args: &mut Vec<String>) -> Result<Self, String> {
            let (mut record, mut replay) = (None, None);
            let mut index = 0;
            while index < args.len() {
                let slot = match args[index].as_str() {
                    "--tuning-record" => &mut record,
                    "--tuning-replay" => &mut replay,
                    _ => {
                        index += 1;
                        continue;
                    }
                };
                if index + 1 == args.len() {
                    return Err(format!("{} requires a file", args[index]));
                }
                *slot = Some(PathBuf::from(args.remove(index + 1)));
                args.remove(index);
            }
            match (record, replay) {
                (Some(_), Some(_)) => {
                    Err("--tuning-record and --tuning-replay exclude each other".into())
                }
                (None, Some(path)) => {
                    let configurations = read(&path)?;
                    eprintln!(
                        "forward_bench: replaying {} pinned tuning configurations from {}",
                        configurations.len(),
                        path.display()
                    );
                    pinned_tuning::install(TuningPin::Replay(configurations));
                    Ok(Self { record: None })
                }
                (record, None) => {
                    if record.is_some() {
                        pinned_tuning::install(TuningPin::Record);
                    }
                    Ok(Self { record })
                }
            }
        }

        /// Write the recorded configurations when recording.
        pub(crate) fn finish(self) -> Result<(), String> {
            let Some(path) = self.record else {
                return Ok(());
            };
            let configurations = pinned_tuning::recorded();
            let entries = configurations
                .iter()
                .map(|pinned| {
                    json!({
                        "entry": pinned.entry,
                        "bindings": pinned.bindings,
                        "statics": pinned.statics,
                        "params": pinned.params,
                    })
                })
                .collect::<Vec<_>>();
            let text = serde_json::to_string_pretty(&entries).map_err(|error| error.to_string())?;
            std::fs::write(&path, text)
                .map_err(|error| format!("writing {}: {error}", path.display()))?;
            eprintln!(
                "forward_bench: recorded {} tuning configurations to {}",
                configurations.len(),
                path.display()
            );
            Ok(())
        }
    }

    fn read(path: &std::path::Path) -> Result<Vec<PinnedConfiguration>, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("reading {}: {error}", path.display()))?;
        let entries: Vec<Value> = serde_json::from_str(&text)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        let malformed = || format!("{}: malformed tuning pin", path.display());
        let map = |value: &Value| -> Result<BTreeMap<String, u64>, String> {
            value
                .as_object()
                .ok_or_else(malformed)?
                .iter()
                .map(|(name, value)| Ok((name.clone(), value.as_u64().ok_or_else(malformed)?)))
                .collect()
        };
        entries
            .iter()
            .map(|entry| {
                let text = |field: &str| {
                    entry[field].as_str().map(str::to_owned).ok_or_else(malformed)
                };
                Ok(PinnedConfiguration {
                    entry: text("entry")?,
                    bindings: text("bindings")?,
                    statics: map(&entry["statics"])?,
                    params: map(&entry["params"])?,
                })
            })
            .collect()
    }
}

/// `--tuning-survey DIR [--tuning-survey-samples N] [--tuning-survey-entries
/// E,..] [--tuning-survey-widen ENTRY:P=v,v,..;Q=v,..]`, only with the
/// development feature `tuning-survey`
/// (`cargo run --release --example forward_bench --features tuning-survey`).
///
/// Every tuned entry instance (or the named entries) is surveyed instead of
/// searched: every admissible configuration measured with N samples per
/// point (15 by default) and validated, one JSON record per instance in DIR:
/// the true costs a search's choice is judged against (tuning spec §E).
/// `--tuning-survey-widen` replaces an entry's parameter domains (keeping
/// each default first). On a shared host, run the whole process under the
/// GPU lock. See `magnitude_model_executor::tuning_survey`.
#[cfg(feature = "tuning-survey")]
mod tuning_survey {
    use magnitude_model_executor::tuning_survey::{self, TuningSurvey};
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    /// Remove the survey flags from `args` and install the survey they ask
    /// for.
    pub(crate) fn extract(args: &mut Vec<String>) -> Result<(), String> {
        let mut flags = BTreeMap::new();
        let mut index = 0;
        while index < args.len() {
            if !args[index].starts_with("--tuning-survey") {
                index += 1;
                continue;
            }
            if index + 1 == args.len() {
                return Err(format!("{} requires a value", args[index]));
            }
            let value = args.remove(index + 1);
            flags.insert(args.remove(index), value);
        }
        let Some(directory) = flags.remove("--tuning-survey") else {
            return match flags.keys().next() {
                Some(flag) => Err(format!("{flag} requires --tuning-survey")),
                None => Ok(()),
            };
        };
        let samples = flags
            .remove("--tuning-survey-samples")
            .map(|value| value.parse().map_err(|_| "--tuning-survey-samples requires a count"))
            .transpose()?
            .unwrap_or(15);
        let entries = flags
            .remove("--tuning-survey-entries")
            .map(|value| value.split(',').map(str::to_owned).collect())
            .unwrap_or_default();
        let mut domains: BTreeMap<String, BTreeMap<String, Vec<u64>>> = BTreeMap::new();
        if let Some(widen) = flags.remove("--tuning-survey-widen") {
            let (entry, parameters) = widen
                .split_once(':')
                .ok_or("--tuning-survey-widen takes ENTRY:P=v,v,..;Q=v,..")?;
            for parameter in parameters.split(';') {
                let (name, values) = parameter
                    .split_once('=')
                    .ok_or("--tuning-survey-widen takes ENTRY:P=v,v,..;Q=v,..")?;
                let values = values
                    .split(',')
                    .map(|value| value.parse::<u64>().map_err(|_| format!("{value:?} is not a count")))
                    .collect::<Result<Vec<_>, _>>()?;
                domains
                    .entry(entry.to_owned())
                    .or_default()
                    .insert(name.to_owned(), values);
            }
        }
        if let Some(flag) = flags.keys().next() {
            return Err(format!("unknown flag {flag}"));
        }
        eprintln!("forward_bench: surveying tuning into {directory} ({samples} samples per point)");
        tuning_survey::install(TuningSurvey {
            directory: PathBuf::from(directory),
            samples,
            entries,
            domains,
        });
        Ok(())
    }
}

pub(crate) struct Options {
    model: PathBuf,
    output: PathBuf,
    path: ExecutionPath,
    storage_gib: u64,
    cells: Vec<String>,
    contexts: Vec<usize>,
    prefills: Vec<usize>,
    /// History tokens before each prefill cell's chunks (0 = fresh sequences).
    prefill_history: usize,
    sequences: Vec<usize>,
    /// Rows of one sequence's verification step (`verify` cell).
    widths: Vec<usize>,
    warm: usize,
    steps: usize,
    attribution_steps: usize,
    reference: Option<PathBuf>,
    chunks: Option<usize>,
    /// `qualify`: scored rows per teacher-forced step; above 1 they run as one
    /// verification forward (the MTP verify shapes) instead of single-row decodes.
    verify_width: usize,
    label: Option<String>,
    /// The kernel cache directory (`--cache-dir`); none caches nothing.
    kernel_cache: Option<PathBuf>,
    /// The target history codec (`--kv-codec`).
    kv_codec: KvCodec,
    /// The device to run on (`--device auto|metal|cuda|vulkan|cpu|SELECTOR`).
    device: magnitude_model_executor::platform::DeviceRequest,
    /// Cross-step pipelining (`--lookahead on|off`, default off). It needs
    /// `--decode-tokens selected`: the queued successor step embeds the
    /// previous selection, so only that decode stream claims it.
    lookahead: bool,
    /// Decode input tokens (`--decode-tokens forced|selected`, default
    /// forced): the forced stream, or each step's greedy selection fed back.
    feedback: bool,
}

fn list(value: &str) -> Result<Vec<usize>, String> {
    value
        .split(',')
        .map(|item| {
            item.parse::<usize>()
                .map_err(|_| format!("{item:?} is not a count"))
        })
        .collect()
}

impl Options {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut model = None;
        let mut output = None;
        let mut options = Self {
            model: PathBuf::new(),
            output: PathBuf::new(),
            path: ExecutionPath::Native,
            storage_gib: 16,
            cells: vec!["decode".into(), "prefill".into(), "concurrent".into()],
            contexts: vec![256, 4096, 16384],
            prefills: vec![32, 128, 512],
            prefill_history: 0,
            sequences: vec![1, 2, 4, 8],
            widths: vec![1, 2, 3, 4, 5, 6, 8],
            warm: 16,
            steps: 32,
            attribution_steps: 4,
            reference: None,
            chunks: None,
            verify_width: 1,
            label: None,
            kernel_cache: None,
            kv_codec: KvCodec::AffineK8V4,
            device: magnitude_model_executor::platform::DeviceRequest::Automatic,
            lookahead: false,
            feedback: false,
        };
        while let Some(flag) = args.next() {
            let mut value = || args.next().ok_or(format!("{flag} requires a value"));
            match flag.as_str() {
                "--model" => model = Some(PathBuf::from(value()?)),
                "--output" => output = Some(PathBuf::from(value()?)),
                "--path" => {
                    options.path = match value()?.as_str() {
                        "native" => ExecutionPath::Native,
                        other => return Err(format!("unsupported execution path {other:?}")),
                    }
                }
                "--storage-gib" => {
                    options.storage_gib = value()?
                        .parse()
                        .map_err(|_| "--storage-gib requires a count")?
                }
                "--cells" => options.cells = value()?.split(',').map(str::to_owned).collect(),
                "--context" => options.contexts = list(&value()?)?,
                "--prefill" => options.prefills = list(&value()?)?,
                "--sequences" => options.sequences = list(&value()?)?,
                "--widths" => options.widths = list(&value()?)?,
                "--prefill-history" => {
                    options.prefill_history = value()?
                        .parse()
                        .map_err(|_| "--prefill-history requires a count")?
                }
                "--warm" => options.warm = value()?.parse().map_err(|_| "--warm requires a count")?,
                "--steps" => {
                    options.steps = value()?.parse().map_err(|_| "--steps requires a count")?
                }
                "--attribution-steps" => {
                    options.attribution_steps = value()?
                        .parse()
                        .map_err(|_| "--attribution-steps requires a count")?
                }
                "--reference" => options.reference = Some(PathBuf::from(value()?)),
                "--chunks" => {
                    options.chunks =
                        Some(value()?.parse().map_err(|_| "--chunks requires a count")?)
                }
                "--verify-width" => {
                    options.verify_width = value()?
                        .parse()
                        .ok()
                        .filter(|width| *width > 0)
                        .ok_or("--verify-width requires a positive count")?
                }
                "--label" => options.label = Some(value()?),
                "--cache-dir" => options.kernel_cache = Some(PathBuf::from(value()?)),
                "--kv-codec" => options.kv_codec = value()?.parse()?,
                "--device" => options.device = value()?.parse().map_err(|error| format!("{error}"))?,
                "--lookahead" => {
                    options.lookahead = match value()?.as_str() {
                        "on" => true,
                        "off" => false,
                        other => return Err(format!("--lookahead takes on or off, not {other}")),
                    }
                }
                "--decode-tokens" => {
                    options.feedback = match value()?.as_str() {
                        "forced" => false,
                        "selected" => true,
                        other => {
                            return Err(format!(
                                "--decode-tokens takes forced or selected, not {other}"
                            ))
                        }
                    }
                }
                other => return Err(format!("unknown flag {other}")),
            }
        }
        options.model = model.ok_or("--model is required")?;
        options.output = output.ok_or("--output is required")?;
        if options.steps == 0 || options.cells.is_empty() {
            return Err("--steps and --cells must be non-empty".into());
        }
        if options.lookahead && !options.feedback {
            return Err("--lookahead on needs --decode-tokens selected".into());
        }
        for cell in &options.cells {
            if !matches!(cell.as_str(), "decode" | "prefill" | "concurrent" | "verify") {
                return Err(format!("unknown cell {cell:?}"));
            }
        }
        Ok(options)
    }
}

/// The executor domain opened for one engine configuration, plus the forced
/// token stream and request identities.
pub(crate) struct Bench {
    domain: ExecutorDomain,
    device: seismic::Device,
    vocabulary: usize,
    /// The loaded model's decoder geometry (`qualify` selects the D4 limits by it).
    geometry: DecoderGeometry,
    next_request: u64,
    /// Cross-step pipelining: traces are collected once per measurement
    /// window (a per-step collect would wait for the step queued behind it).
    lookahead: bool,
    /// Decode feeds back each step's selection instead of forced tokens.
    feedback: bool,
}

/// One open request and its accepted position.
pub(crate) struct Sequence {
    request: RequestId,
    position: usize,
    /// The token the last step selected, which decode feeds back under
    /// lookahead.
    selected: Option<TokenId>,
}

/// Host and device times of one step. All times are `seismic::host_seconds`.
pub(crate) struct Step {
    start: f64,
    submitted: f64,
    finished: f64,
    end: f64,
    submissions: Vec<TracedSubmission>,
}

impl Bench {
    pub(crate) fn open(
        model: &std::path::Path,
        path: ExecutionPath,
        storage_gib: u64,
        context_tokens: usize,
        max_batch: usize,
        decode_rows: usize,
        kernel_cache: Option<&std::path::Path>,
        kv_codec: KvCodec,
        device: magnitude_model_executor::platform::DeviceRequest,
        lookahead: bool,
    ) -> Result<Self, String> {
        let storage_bytes = storage_gib
            .checked_mul(1 << 30)
            .ok_or("storage budget exceeds u64")?;
        let resolved = EngineConfiguration {
            package: PackageOptions {
                target: model.to_path_buf(),
                projector: ProjectorSelection::Disabled,
            },
            model: ModelPolicy {
                method: ModelMethod::Plain,
                mtp_proposals: None,
                kv_codec,
                lookahead,
            },
            context_tokens: Some(context_tokens),
            service: ServiceLimits {
                max_requests: max_batch,
                max_batch,
                prefill_tokens: PREFILL_ROWS,
                decode_tokens: decode_rows,
                decode_share: 0.5,
                locality_seconds: 1.0,
            },
            storage: StoragePolicy {
                storage_bytes,
                retention_bytes: Some(0),
                safety_reserve_bytes: (storage_bytes / 10).min(1 << 30),
            },
            path,
            device,
            control_capacity: 256,
            kernel_cache: kernel_cache.map(std::path::Path::to_path_buf),
        }
        .resolve()?;
        let vocabulary = usize::try_from(resolved.manifest.definition.geometry.vocabulary)
            .map_err(|_| "vocabulary exceeds host domain")?;
        let geometry = resolved.manifest.definition.geometry.clone();
        let (domain, _) = build_native_domain(&resolved.manifest)?;
        let device = domain.resources().device().clone();
        Ok(Self {
            domain,
            device,
            vocabulary,
            geometry,
            next_request: 1,
            lookahead,
            feedback: false,
        })
    }

    pub(crate) fn device(&self) -> &seismic::Device {
        &self.device
    }

    pub(crate) fn open_sequence(&mut self) -> Result<Sequence, String> {
        let request = RequestId(self.next_request);
        self.next_request += 1;
        self.domain.open(request).map_err(|error| error.to_string())?;
        Ok(Sequence {
            request,
            position: 0,
            selected: None,
        })
    }

    /// A decode step's input token: the forced token, or with feedback the
    /// sequence's last selection (under lookahead, the step queued ahead).
    fn decode_token(&self, sequence: &Sequence) -> TokenId {
        match sequence.selected {
            Some(token) if self.feedback => token,
            _ => self.token(sequence.position),
        }
    }

    pub(crate) fn close(&mut self, sequence: Sequence) -> Result<(), String> {
        self.domain.close(sequence.request)
    }

    /// The forced token at a position: a fixed spread over ordinary
    /// vocabulary rows, away from the special tokens at the top.
    fn token(&self, position: usize) -> TokenId {
        let span = (self.vocabulary - 1024) as u64;
        TokenId(((position as u64 * 2_654_435_761 + 12_345) % span + 512) as u32)
    }

    pub(crate) fn forced(&self, start: usize, rows: usize) -> Vec<TokenId> {
        (start..start + rows).map(|position| self.token(position)).collect()
    }

    fn greedy(position: usize) -> SelectSpec {
        SelectSpec {
            sampling: Sampling::Greedy,
            seed: 0,
            position,
            domain: 0,
            mask: None,
            shaping: Shaping::default(),
            history: None,
        }
    }

    /// A forward of `tokens` at the sequence's position. `demand` decides the
    /// readout: `SELECT` samples greedily from the last row (a decode, or a
    /// finishing prefill), `LOGITS` exports the last row's logits.
    pub(crate) fn forward(
        sequence: &Sequence,
        kind: WorkKind,
        tokens: Vec<TokenId>,
        demand: Demand,
    ) -> Operation {
        let last = sequence.position + tokens.len() - 1;
        let select = if demand.contains(Demand::SELECT) {
            vec![Self::greedy(last)]
        } else {
            Vec::new()
        };
        let committed = tokens.len();
        Operation::Forward {
            request: sequence.request,
            kind,
            tokens,
            position: sequence.position,
            conditioning: None,
            demand,
            select,
            committed,
        }
    }

    /// Submit one group of forwards, wait, commit every row and advance the
    /// sequences. Returns the step's timings (from `trace`, when present) and
    /// the outcomes in operation order.
    pub(crate) fn step(
        &mut self,
        operations: Vec<Operation>,
        sequences: &mut [&mut Sequence],
        trace: Option<&SubmissionTrace>,
    ) -> Result<(Step, Vec<Outcome>), String> {
        let text = |error: DomainError| error.to_string();
        let start = host_seconds();
        let groups = service_domain::group(&self.domain, operations);
        let [group] = groups.as_slice() else {
            return Err("a bench step must form exactly one target group".into());
        };
        let DomainFlight::Target(flight) =
            service_domain::submit_group(&mut self.domain, group).map_err(text)?
        else {
            return Err("a bench step must run on the target lane".into());
        };
        let submitted = host_seconds();
        let pending = self.domain.finish_target(flight).map_err(text)?;
        let finished = host_seconds();
        if pending.len() != sequences.len() {
            return Err("step outcomes differ from its sequences".into());
        }
        let mut outcomes = Vec::with_capacity(pending.len());
        for (pending, sequence) in pending.into_iter().zip(sequences.iter_mut()) {
            if pending.request() != sequence.request {
                return Err("step outcome order differs from its sequences".into());
            }
            if let Outcome::Forward { rows } = pending.outcome() {
                sequence.selected = rows.last().and_then(|row| row.selected).map(|row| row.token);
            }
            outcomes.push(pending.outcome().clone());
            let rows = pending.rows();
            self.domain
                .reconcile(
                    pending,
                    PhysicalDecision {
                        accepted_rows: rows,
                    },
                )
                .map_err(text)?;
            sequence.position += rows;
        }
        let end = host_seconds();
        let submissions = trace
            .map(|trace| trace.collect().map_err(|error| error.to_string()))
            .transpose()?
            .unwrap_or_default();
        Ok((
            Step {
                start,
                submitted,
                finished,
                end,
                submissions,
            },
            outcomes,
        ))
    }

    /// Accept `rows` tokens of history in prefill chunks without readout.
    pub(crate) fn history(
        &mut self,
        sequence: &mut Sequence,
        rows: usize,
        trace: Option<&SubmissionTrace>,
    ) -> Result<(), String> {
        let target = sequence.position + rows;
        while sequence.position < target {
            let chunk = (target - sequence.position).min(PREFILL_ROWS);
            let tokens = self.forced(sequence.position, chunk);
            let operation = Self::forward(sequence, WorkKind::Prefill, tokens, Demand::NONE);
            self.step(vec![operation], &mut [sequence], trace)?;
        }
        Ok(())
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values[middle]
    } else {
        (values[middle - 1] + values[middle]) / 2.0
    }
}

/// Total length of the union of intervals.
fn union(mut intervals: Vec<(f64, f64)>) -> f64 {
    intervals.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut total = 0.0;
    let mut current: Option<(f64, f64)> = None;
    for (start, end) in intervals {
        current = match current {
            Some((open, close)) if start <= close => Some((open, close.max(end))),
            Some((open, close)) => {
                total += close - open;
                Some((start, end))
            }
            None => Some((start, end)),
        };
    }
    total + current.map_or(0.0, |(open, close)| close - open)
}

const MS: f64 = 1e3;

/// Per-step host/device decomposition, in milliseconds.
fn step_summary(step: &Step, previous_device_end: Option<f64>) -> Result<Value, String> {
    let first = step
        .submissions
        .first()
        .ok_or("traced step recorded no submissions")?;
    let last = step.submissions.last().expect("non-empty");
    let device_start = step
        .submissions
        .iter()
        .map(|submission| submission.device.0)
        .fold(f64::INFINITY, f64::min);
    let device_end = step
        .submissions
        .iter()
        .map(|submission| submission.device.1)
        .fold(f64::NEG_INFINITY, f64::max);
    let busy = union(
        step.submissions
            .iter()
            .map(|submission| submission.device)
            .collect(),
    );
    let encode: f64 = step
        .submissions
        .iter()
        .map(|submission| submission.committed - submission.encode_start)
        .sum();
    Ok(json!({
        "wall_ms": (step.end - step.start) * MS,
        "submissions": step.submissions.len(),
        "launches": step.submissions.iter().map(|submission| submission.launches.len()).sum::<usize>(),
        "host_before_first_encode_ms": (first.encode_start - step.start) * MS,
        "host_encode_ms": encode * MS,
        "host_submit_span_ms": (last.committed - first.encode_start) * MS,
        "host_submit_return_ms": (step.submitted - step.start) * MS,
        "host_finish_ms": (step.finished - step.submitted) * MS,
        "host_reconcile_ms": (step.end - step.finished) * MS,
        "device_first_start_after_step_start_ms": (device_start - step.start) * MS,
        "device_busy_ms": busy * MS,
        "device_span_ms": (device_end - device_start) * MS,
        "device_idle_within_step_ms": (device_end - device_start - busy) * MS,
        "host_after_device_ms": (step.end - device_end) * MS,
        "device_gap_from_previous_step_ms": previous_device_end.map(|end| (device_start - end) * MS),
    }))
}

fn device_end(step: &Step) -> f64 {
    step.submissions
        .iter()
        .map(|submission| submission.device.1)
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Median of every numeric field across step summaries.
fn medians(summaries: &[Value]) -> Value {
    let mut fields: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for summary in summaries {
        for (key, value) in summary.as_object().expect("summary object") {
            if let Some(value) = value.as_f64() {
                fields.entry(key.clone()).or_default().push(value);
            }
        }
    }
    Value::Object(
        fields
            .into_iter()
            .map(|(key, mut values)| (key, json!(median(&mut values))))
            .collect(),
    )
}

/// Device time per entry over attribution steps, per step.
/// Device time per entry of `steps` attributed steps, from their launches
/// that started before `until` (the last step's end: a step queued behind it
/// under lookahead is not one of them).
fn attribution(submissions: &[TracedSubmission], steps: usize, until: f64) -> Value {
    let mut entries: BTreeMap<String, (usize, f64)> = BTreeMap::new();
    let mut timed = 0.0;
    for submission in submissions {
        for launch in &submission.launches {
            let Some((start, end)) = launch.device.filter(|(start, _)| *start < until) else {
                continue;
            };
            let label = if launch.launch == 0 {
                launch.entry.clone()
            } else {
                format!("{}#{}", launch.entry, launch.launch)
            };
            let entry = entries.entry(label).or_default();
            entry.0 += 1;
            entry.1 += end - start;
            timed += end - start;
        }
    }
    let count = steps.max(1) as f64;
    let mut rows = entries
        .into_iter()
        .map(|(entry, (launches, seconds))| {
            json!({
                "entry": entry,
                "launches_per_step": launches as f64 / count,
                "device_ms_per_step": seconds / count * MS,
                "mean_launch_us": seconds / launches as f64 * 1e6,
                "share": seconds / timed,
            })
        })
        .collect::<Vec<_>>();
    let device_ms = |row: &Value| row["device_ms_per_step"].as_f64().expect("device time");
    rows.sort_by(|a, b| device_ms(b).total_cmp(&device_ms(a)));
    json!({
        "steps": steps,
        "timed_device_ms_per_step": timed / count * MS,
        "entries": rows,
    })
}

/// Measure `measured` steps (after `warm`) produced by `make`, with a
/// production trace, then attribute `attribution_steps` more.
fn measure(
    bench: &mut Bench,
    sequences: &mut [Sequence],
    warm: usize,
    measured: usize,
    attribution_steps: usize,
    make: &dyn Fn(&Bench, &[Sequence]) -> Vec<Operation>,
) -> Result<Value, String> {
    let traced = |bench: &Bench, detail| {
        bench
            .device()
            .trace_submissions(detail)
            .map_err(|error| error.to_string())
    };
    let run = |bench: &mut Bench,
               sequences: &mut [Sequence],
               trace: &SubmissionTrace|
     -> Result<Step, String> {
        let operations = make(bench, sequences);
        let mut refs = sequences.iter_mut().collect::<Vec<_>>();
        // Under lookahead the trace is collected once per window: a per-step
        // collect would wait for the step queued behind this one.
        let per_step = (!bench.lookahead).then_some(trace);
        bench.step(operations, &mut refs, per_step).map(|(step, _)| step)
    };
    let collect = |trace: &SubmissionTrace| trace.collect().map_err(|error| error.to_string());
    let trace = traced(bench, TraceDetail::Submissions)?;
    for _ in 0..warm {
        run(bench, sequences, &trace)?;
    }
    // Under lookahead a step's submissions are traced during the previous
    // step, so steps are summarized by host time only and device time over
    // the whole window.
    collect(&trace)?;
    let mut summaries = Vec::with_capacity(measured);
    let mut previous = None;
    let mut window = Vec::new();
    let mut span = (f64::INFINITY, f64::NEG_INFINITY);
    // Every sequence's selections, step by step: equal with and without
    // lookahead when decode feeds them back.
    let mut selected = Vec::with_capacity(measured);
    for _ in 0..measured {
        let step = run(bench, sequences, &trace)?;
        selected.push(
            sequences
                .iter()
                .map(|sequence| sequence.selected.map(|token| token.0))
                .collect::<Vec<_>>(),
        );
        span = (span.0.min(step.start), span.1.max(step.end));
        if bench.lookahead {
            summaries.push(json!({ "wall_ms": (step.end - step.start) * MS }));
        } else {
            summaries.push(step_summary(&step, previous)?);
            previous = Some(device_end(&step));
        }
        window.extend(step.submissions);
    }
    window.extend(collect(&trace)?);
    drop(trace);
    let trace = traced(bench, TraceDetail::Launches)?;
    let mut attributed = Vec::new();
    let mut attributed_end = f64::NEG_INFINITY;
    for _ in 0..attribution_steps {
        let step = run(bench, sequences, &trace)?;
        attributed_end = step.end;
        attributed.extend(step.submissions);
    }
    attributed.extend(collect(&trace)?);
    drop(trace);
    let mut wall = summaries
        .iter()
        .map(|summary| summary["wall_ms"].as_f64().expect("wall"))
        .collect::<Vec<_>>();
    Ok(json!({
        "warm_steps": warm,
        "lookahead": bench.lookahead,
        "median_ms": median(&mut wall),
        "median": medians(&summaries),
        "window": window_summary(measured, span, &window),
        "selected": selected,
        "steps": summaries,
        "attribution": attribution(&attributed, attribution_steps, attributed_end),
    }))
}

/// Device time over a window of consecutive steps: busy is the union of the
/// submissions' device intervals clipped to the window, idle the rest.
fn window_summary(steps: usize, (start, end): (f64, f64), submissions: &[TracedSubmission]) -> Value {
    let busy = union(
        submissions
            .iter()
            .map(|submission| (submission.device.0.max(start), submission.device.1.min(end)))
            .filter(|(from, to)| to > from)
            .collect(),
    );
    let steps = steps as f64;
    json!({
        "wall_ms_per_step": (end - start) * MS / steps,
        "device_busy_ms_per_step": busy * MS / steps,
        "device_idle_ms_per_step": (end - start - busy) * MS / steps,
    })
}

fn decode_operations(bench: &Bench, sequences: &[Sequence]) -> Vec<Operation> {
    sequences
        .iter()
        .map(|sequence| {
            Bench::forward(
                sequence,
                WorkKind::Decode,
                vec![bench.decode_token(sequence)],
                Demand::SELECT,
            )
        })
        .collect()
}

fn decode_cell(
    bench: &mut Bench,
    options: &Options,
    context: usize,
    count: usize,
) -> Result<Value, String> {
    let mut sequences = Vec::with_capacity(count);
    for _ in 0..count {
        let mut sequence = bench.open_sequence()?;
        bench.history(&mut sequence, context, None)?;
        sequences.push(sequence);
    }
    let mut result = measure(
        bench,
        &mut sequences,
        options.warm,
        options.steps,
        options.attribution_steps,
        &decode_operations,
    )?;
    let step_ms = result["median_ms"].as_f64().expect("median");
    result["context"] = json!(context);
    result["sequences"] = json!(count);
    result["aggregate_tokens_per_second"] = json!(count as f64 / (step_ms / MS));
    for sequence in sequences {
        bench.close(sequence)?;
    }
    Ok(result)
}

/// Prefill chunks per cell: one cold, three warm (timed), one attributed.
const PREFILL_CHUNKS: usize = 5;

/// Without history, every chunk is a fresh sequence at position 0. With
/// `--prefill-history H`, one sequence first accepts H tokens of history and
/// the chunks follow each other on it, so chunk `i` starts at H + i * rows.
fn prefill_cell(bench: &mut Bench, options: &Options, rows: usize) -> Result<Value, String> {
    let history = options.prefill_history;
    let mut shared = if history == 0 {
        None
    } else {
        let mut sequence = bench.open_sequence()?;
        bench.history(&mut sequence, history, None)?;
        Some(sequence)
    };
    let chunk = |bench: &mut Bench,
                 shared: &mut Option<Sequence>,
                 trace: &SubmissionTrace|
     -> Result<Step, String> {
        let mut fresh = match shared {
            Some(_) => None,
            None => Some(bench.open_sequence()?),
        };
        let sequence = shared.as_mut().or(fresh.as_mut()).expect("a sequence");
        let operation = Bench::forward(
            sequence,
            WorkKind::Prefill,
            bench.forced(sequence.position, rows),
            Demand::SELECT,
        );
        let (step, _) = bench.step(vec![operation], &mut [sequence], Some(trace))?;
        if let Some(sequence) = fresh {
            bench.close(sequence)?;
        }
        Ok(step)
    };
    let trace = bench
        .device()
        .trace_submissions(TraceDetail::Submissions)
        .map_err(|error| error.to_string())?;
    let cold = step_summary(&chunk(bench, &mut shared, &trace)?, None)?;
    let mut summaries = Vec::new();
    for _ in 1..PREFILL_CHUNKS - 1 {
        summaries.push(step_summary(&chunk(bench, &mut shared, &trace)?, None)?);
    }
    drop(trace);
    let trace = bench
        .device()
        .trace_submissions(TraceDetail::Launches)
        .map_err(|error| error.to_string())?;
    let attributed = chunk(bench, &mut shared, &trace)?;
    drop(trace);
    if let Some(sequence) = shared {
        bench.close(sequence)?;
    }
    let mut wall = summaries
        .iter()
        .map(|summary| summary["wall_ms"].as_f64().expect("wall"))
        .collect::<Vec<_>>();
    Ok(json!({
        "rows": rows,
        "history": history,
        "median_ms": median(&mut wall),
        "cold": cold,
        "median": medians(&summaries),
        "steps": summaries,
        "attribution": attribution(&attributed.submissions, 1, attributed.end),
    }))
}

fn host() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_default()
}

fn bench(options: &Options) -> Result<(), String> {
    let decode = options.cells.iter().any(|cell| cell == "decode");
    let concurrent = options.cells.iter().any(|cell| cell == "concurrent");
    let prefill = options.cells.iter().any(|cell| cell == "prefill");
    let verify = options.cells.iter().any(|cell| cell == "verify");
    let longest = if decode || concurrent || verify {
        options.contexts.iter().copied().max().unwrap_or(0)
    } else {
        0
    };
    // A verification step accepts every row it verifies.
    let widest = if verify {
        options.widths.iter().copied().max().unwrap_or(1)
    } else {
        1
    };
    let steps = (options.warm + options.steps + options.attribution_steps + 1) * widest;
    let widest_prefill = options.prefills.iter().copied().max().unwrap_or(0);
    let longest_prefill = match (prefill, options.prefill_history) {
        (false, _) => 0,
        (true, 0) => widest_prefill,
        (true, history) => history + PREFILL_CHUNKS * widest_prefill,
    };
    let context_tokens = (longest + steps).max(longest_prefill);
    let max_batch = if concurrent {
        options.sequences.iter().copied().max().unwrap_or(1)
    } else {
        1
    };
    let loaded = host_seconds();
    let mut bench = Bench::open(
        &options.model,
        options.path,
        options.storage_gib,
        context_tokens,
        max_batch,
        max_batch * widest,
        options.kernel_cache.as_deref(),
        options.kv_codec,
        options.device,
        options.lookahead,
    )?;
    bench.feedback = options.feedback;
    let load_seconds = host_seconds() - loaded;
    let mut report = json!({
        "tool": "engine/examples/forward_bench.rs",
        "lookahead": options.lookahead,
        "decode_tokens": if options.feedback { "selected" } else { "forced" },
        "protocol": "forced-token forward through the executor domain (below the service); greedy selection on decode and finishing prefill; per-submission device intervals on the host clock; attribution steps time each launch separately",
        "host": host(),
        "model": options.model,
        "path": options.path.to_string(),
        "kv_codec": options.kv_codec.identity(),
        "device": bench.device().info().name.clone(),
        "context_tokens": context_tokens,
        "max_batch": max_batch,
        "load_seconds": load_seconds,
        "decode": [],
        "prefill": [],
        "concurrent": [],
        "verify": [],
    });
    let write = |report: &Value| {
        std::fs::write(
            &options.output,
            serde_json::to_string_pretty(report).expect("report serializes"),
        )
        .map_err(|error| format!("writing {}: {error}", options.output.display()))
    };
    if decode {
        for &context in &options.contexts {
            eprintln!("decode context={context}");
            let cell = decode_cell(&mut bench, options, context, 1)?;
            eprintln!("  median {:.3} ms", cell["median_ms"].as_f64().unwrap_or(f64::NAN));
            report["decode"].as_array_mut().expect("array").push(cell);
            write(&report)?;
        }
    }
    if prefill {
        for &rows in &options.prefills {
            eprintln!("prefill rows={rows}");
            let cell = prefill_cell(&mut bench, options, rows)?;
            eprintln!("  median {:.3} ms", cell["median_ms"].as_f64().unwrap_or(f64::NAN));
            report["prefill"].as_array_mut().expect("array").push(cell);
            write(&report)?;
        }
    }
    if concurrent {
        for &context in &options.contexts {
            for &count in &options.sequences {
                eprintln!("concurrent sequences={count} context={context}");
                let cell = decode_cell(&mut bench, options, context, count)?;
                eprintln!(
                    "  {:.1} tok/s",
                    cell["aggregate_tokens_per_second"].as_f64().unwrap_or(f64::NAN)
                );
                report["concurrent"].as_array_mut().expect("array").push(cell);
                write(&report)?;
            }
        }
    }
    if verify {
        for &context in &options.contexts {
            for &width in &options.widths {
                eprintln!("verify width={width} context={context}");
                let cell = verify_cell(&mut bench, options, context, width)?;
                eprintln!("  median {:.3} ms", cell["median_ms"].as_f64().unwrap_or(f64::NAN));
                report["verify"].as_array_mut().expect("array").push(cell);
                write(&report)?;
            }
        }
    }
    write(&report)
}

/// One sequence verifying `width` rows per step, every row selecting (the
/// MTP verification shape); every row is accepted.
fn verify_cell(
    bench: &mut Bench,
    options: &Options,
    context: usize,
    width: usize,
) -> Result<Value, String> {
    let mut sequence = bench.open_sequence()?;
    bench.history(&mut sequence, context, None)?;
    let make = |bench: &Bench, sequences: &[Sequence]| {
        sequences
            .iter()
            .map(|sequence| {
                let tokens = bench.forced(sequence.position, width);
                let select = (0..width)
                    .map(|row| Bench::greedy(sequence.position + row))
                    .collect();
                Operation::Forward {
                    request: sequence.request,
                    kind: if width == 1 {
                        WorkKind::Decode
                    } else {
                        WorkKind::Verify
                    },
                    tokens,
                    position: sequence.position,
                    conditioning: None,
                    demand: Demand::SELECT,
                    select,
                    committed: width,
                }
            })
            .collect()
    };
    let mut sequences = [sequence];
    let mut result = measure(
        bench,
        &mut sequences,
        options.warm,
        options.steps,
        options.attribution_steps,
        &make,
    )?;
    result["context"] = json!(context);
    result["width"] = json!(width);
    let [sequence] = sequences;
    bench.close(sequence)?;
    Ok(result)
}

/// D4 precision runner (Q1): V4 logits against a llama.cpp
/// `--kl-divergence-base` file, with llama.cpp's own KL and same-top
/// definitions (`validation/precision/README.md` documents the file).
mod qualify {
    use super::{host, Bench, DecoderGeometry, Demand, Operation, Options, Outcome, WorkKind};
    use magnitude_model_contracts::FeedForwardGeometry;
    use magnitude_model_executor::TokenId;
    use serde_json::json;
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};

    /// Base log-probabilities at or below this are floored and skipped.
    const FLOOR: f32 = -16.0;

    struct BaseFile {
        file: File,
        n_ctx: usize,
        n_vocab: usize,
        n_chunk: usize,
        tokens: Vec<i32>,
        rows_offset: u64,
        row_bytes: usize,
    }

    fn read_i32s(file: &mut File, count: usize) -> Result<Vec<i32>, String> {
        let mut bytes = vec![0u8; count * 4];
        file.read_exact(&mut bytes).map_err(|error| error.to_string())?;
        Ok(bytes
            .chunks_exact(4)
            .map(|word| i32::from_le_bytes(word.try_into().expect("four bytes")))
            .collect())
    }

    impl BaseFile {
        fn open(path: &std::path::Path) -> Result<Self, String> {
            let mut file =
                File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
            let mut magic = [0u8; 8];
            file.read_exact(&mut magic).map_err(|error| error.to_string())?;
            if &magic != b"_logits_" {
                return Err(format!("{} is not a llama.cpp logits base file", path.display()));
            }
            let header = read_i32s(&mut file, 3)?;
            let count = |value: i32| usize::try_from(value).map_err(|_| "negative header field");
            let (n_ctx, n_vocab, n_chunk) =
                (count(header[0])?, count(header[1])?, count(header[2])?);
            let tokens = read_i32s(&mut file, n_chunk * n_ctx)?;
            let nv = 2 * n_vocab.div_ceil(2) + 4;
            Ok(Self {
                file,
                n_ctx,
                n_vocab,
                n_chunk,
                tokens,
                rows_offset: 20 + 4 * (n_chunk * n_ctx) as u64,
                row_bytes: 2 * nv,
            })
        }

        fn first(&self) -> usize {
            self.n_ctx / 2
        }

        fn evaluated(&self) -> usize {
            self.n_ctx - 1 - self.first()
        }

        fn chunk_tokens(&self, chunk: usize) -> Result<Vec<TokenId>, String> {
            self.tokens[chunk * self.n_ctx..(chunk + 1) * self.n_ctx]
                .iter()
                .map(|&token| {
                    u32::try_from(token)
                        .map(TokenId)
                        .map_err(|_| "negative token id".to_owned())
                })
                .collect()
        }

        /// Decoded base log-probabilities of one stored row.
        fn row(&mut self, chunk: usize, row: usize) -> Result<Vec<f32>, String> {
            let offset = self.rows_offset
                + (self.row_bytes * (chunk * self.evaluated() + row)) as u64;
            self.file
                .seek(SeekFrom::Start(offset))
                .map_err(|error| error.to_string())?;
            let mut bytes = vec![0u8; 8 + 2 * self.n_vocab];
            self.file
                .read_exact(&mut bytes)
                .map_err(|error| error.to_string())?;
            let scale = f32::from_le_bytes(bytes[0..4].try_into().expect("four bytes"));
            let min_log_prob = f32::from_le_bytes(bytes[4..8].try_into().expect("four bytes"));
            Ok(bytes[8..]
                .chunks_exact(2)
                .map(|half| {
                    scale * f32::from(u16::from_le_bytes(half.try_into().expect("two bytes")))
                        + min_log_prob
                })
                .collect())
        }
    }

    /// First index attaining the maximum.
    fn argmax(values: &[f32]) -> usize {
        let mut best = 0;
        for (index, &value) in values.iter().enumerate() {
            if value > values[best] {
                best = index;
            }
        }
        best
    }

    /// Reference top-two log-probability gaps (nats) at which same-top is also
    /// reported over only the positions whose gap exceeds the margin
    /// (`kl_base.py` reports the same set).
    const MARGINS: [f32; 5] = [0.01, 0.05, 0.1, 0.2, 0.5];

    /// Base log-probability of the top token minus that of the runner-up.
    fn top_gap(base: &[f32]) -> f32 {
        let (mut first, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &value in base {
            if value > first {
                second = first;
                first = value;
            } else if value > second {
                second = value;
            }
        }
        first - second
    }

    /// KL(base ‖ candidate) over base terms above the floor, and same-top.
    fn compare(base: &[f32], logits: &[f32]) -> (f64, bool) {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f64 = logits.iter().map(|&l| f64::from((l - max).exp())).sum();
        let log_sum = sum.ln() as f32;
        let kl = base
            .iter()
            .zip(logits)
            .filter(|(&b, _)| b > FLOOR)
            .map(|(&b, &l)| {
                let candidate = l - max - log_sum;
                f64::from(b.exp()) * f64::from(b - candidate)
            })
            .sum();
        (kl, argmax(base) == argmax(logits))
    }

    fn percentile(sorted: &[f64], fraction: f64) -> f64 {
        sorted[((sorted.len() - 1) as f64 * fraction).round() as usize]
    }

    /// The fixed prompt set's categories: `<reference dir>/<category>.bin`.
    const CATEGORIES: [&str; 3] = ["prose", "code", "tool_json"];

    /// The D4 prompt set's chunk length (64 scored positions per chunk). A reference
    /// with longer chunks is a long-context check: reported, but D4's limits don't apply.
    const D4_N_CTX: usize = 130;

    /// One model's D4 limits against its F32 reference (spec
    /// `seismic-native-kernel-program.md` §1 D4: limits are per model).
    struct Limits {
        model: &'static str,
        mean_kld_overall: f64,
        mean_kld_category: f64,
        same_top_overall: f64,
        same_top_category: f64,
        kld_p99_category: f64,
    }

    const QWEN35_4B: Limits = Limits {
        model: "Qwen3.5 4B",
        mean_kld_overall: 0.010,
        mean_kld_category: 0.015,
        same_top_overall: 0.960,
        same_top_category: 0.954,
        kld_p99_category: 0.10,
    };

    const QWEN35_35B_A3B: Limits = Limits {
        model: "Qwen3.5 35B-A3B",
        mean_kld_overall: 0.013,
        mean_kld_category: 0.025,
        same_top_overall: 0.958,
        same_top_category: 0.950,
        kld_p99_category: 0.23,
    };

    /// The D4 limits of the loaded model, identified by its geometry: hidden
    /// width, block count and feed-forward kind (expert count when routed).
    fn limits(geometry: &DecoderGeometry) -> Result<&'static Limits, String> {
        let experts = geometry
            .blocks
            .iter()
            .map(|block| match &block.feedforward {
                FeedForwardGeometry::Dense { .. } => None,
                FeedForwardGeometry::Routed(experts) => Some(experts.count),
            })
            .max()
            .flatten();
        match (geometry.hidden, geometry.blocks.len(), experts) {
            (2560, 32, None) => Ok(&QWEN35_4B),
            (2048, 40, Some(256)) => Ok(&QWEN35_35B_A3B),
            (hidden, blocks, experts) => Err(format!(
                "no D4 limits for this model (hidden {hidden}, {blocks} blocks, experts {experts:?}); \
                 derive them per spec D4 from its own F32 reference and llama.cpp spreads"
            )),
        }
    }

    /// Per-position results: KL(base ‖ V4), same top, reference top-two gap.
    #[derive(Default)]
    struct Positions {
        divergences: Vec<f64>,
        same_top: Vec<bool>,
        gaps: Vec<f32>,
    }

    impl Positions {
        fn push(&mut self, kl: f64, top: bool, gap: f32) {
            self.divergences.push(kl);
            self.same_top.push(top);
            self.gaps.push(gap);
        }

        fn extend(&mut self, other: &Self) {
            self.divergences.extend(&other.divergences);
            self.same_top.extend(&other.same_top);
            self.gaps.extend(&other.gaps);
        }

        fn mean_kld(&self) -> f64 {
            self.divergences.iter().sum::<f64>() / self.divergences.len() as f64
        }

        fn same_top(&self) -> f64 {
            self.same_top.iter().filter(|&&top| top).count() as f64 / self.same_top.len() as f64
        }

        fn kld_percentile(&self, fraction: f64) -> f64 {
            let mut sorted = self.divergences.clone();
            sorted.sort_by(f64::total_cmp);
            percentile(&sorted, fraction)
        }

        /// llama.cpp's summary statistics plus the top-two margin diagnostic.
        fn summary(&self) -> serde_json::Value {
            let count = self.divergences.len() as f64;
            let mean = self.mean_kld();
            let variance = self.divergences.iter().map(|kl| kl * kl).sum::<f64>() / count - mean * mean;
            let same_top = self.same_top();
            let mut sorted = self.divergences.clone();
            sorted.sort_by(f64::total_cmp);
            let above_margin: serde_json::Map<String, serde_json::Value> = MARGINS
                .iter()
                .map(|&margin| {
                    let (positions, agreeing) = self
                        .gaps
                        .iter()
                        .zip(&self.same_top)
                        .filter(|(gap, _)| **gap > margin)
                        .fold((0usize, 0usize), |(positions, agreeing), (_, top)| {
                            (positions + 1, agreeing + usize::from(*top))
                        });
                    (
                        margin.to_string(),
                        json!({
                            "positions": positions,
                            "same_top": agreeing as f64 / positions as f64,
                            "flips": positions - agreeing,
                        }),
                    )
                })
                .collect();
            json!({
                "positions": self.divergences.len(),
                "mean_kld": mean,
                "mean_kld_uncertainty": (variance.max(0.0) / (count - 1.0)).sqrt(),
                "same_top": same_top,
                "same_top_uncertainty": (same_top * (1.0 - same_top) / (count - 1.0)).sqrt(),
                "kld_percentiles": {
                    "p50": percentile(&sorted, 0.5),
                    "p90": percentile(&sorted, 0.9),
                    "p99": percentile(&sorted, 0.99),
                    "p99.9": percentile(&sorted, 0.999),
                    "max": sorted[sorted.len() - 1],
                },
                "same_top_above_margin": above_margin,
            })
        }
    }

    /// The D4 verdict over complete categories.
    fn verdict(
        limits: &Limits,
        categories: &[(&str, Positions)],
        overall: &Positions,
    ) -> serde_json::Value {
        let mut checks = vec![
            json!({"metric": "mean_kld overall", "value": overall.mean_kld(),
                   "limit": limits.mean_kld_overall, "pass": overall.mean_kld() <= limits.mean_kld_overall}),
            json!({"metric": "same_top overall", "value": overall.same_top(),
                   "limit": limits.same_top_overall, "pass": overall.same_top() >= limits.same_top_overall}),
        ];
        for (name, positions) in categories {
            let p99 = positions.kld_percentile(0.99);
            checks.push(json!({"metric": format!("mean_kld {name}"), "value": positions.mean_kld(),
                               "limit": limits.mean_kld_category, "pass": positions.mean_kld() <= limits.mean_kld_category}));
            checks.push(json!({"metric": format!("same_top {name}"), "value": positions.same_top(),
                               "limit": limits.same_top_category, "pass": positions.same_top() >= limits.same_top_category}));
            checks.push(json!({"metric": format!("kld_p99 {name}"), "value": p99,
                               "limit": limits.kld_p99_category, "pass": p99 <= limits.kld_p99_category}));
        }
        let pass = checks.iter().all(|check| check["pass"] == json!(true));
        json!({"pass": pass, "limits": limits.model, "checks": checks})
    }

    /// Qualifies every category of a reference directory in one engine load
    /// (tuning at load dominates start-up), rewriting the report after each
    /// category.
    pub(super) fn run(options: &Options) -> Result<(), String> {
        let directory = options
            .reference
            .as_ref()
            .ok_or("qualify requires --reference <directory of prose/code/tool_json .bin>")?;
        let mut bases = CATEGORIES
            .iter()
            .map(|name| BaseFile::open(&directory.join(format!("{name}.bin"))).map(|base| (*name, base)))
            .collect::<Result<Vec<_>, _>>()?;
        let n_ctx = bases[0].1.n_ctx;
        for (name, base) in &bases {
            if base.n_ctx != n_ctx || base.evaluated() == 0 {
                return Err(format!("{name}: reference chunk geometry differs"));
            }
        }
        // The bench seals prefill classes up to PREFILL_ROWS and load warm-up runs each of them
        // as one request, so the context must hold the largest class, not just n_ctx.
        let context = n_ctx.max(super::PREFILL_ROWS);
        let mut bench = Bench::open(
            &options.model,
            options.path,
            options.storage_gib,
            context,
            1,
            options.verify_width,
            options.kernel_cache.as_deref(),
            options.kv_codec,
            options.device,
            false,
        )?;
        let limits = limits(&bench.geometry)?;
        let mut report = json!({
            "tool": "engine/examples/forward_bench.rs qualify",
            "protocol": "V4 native target: prefill of the chunk's first n_ctx/2 tokens (in PREFILL_ROWS chunks), then teacher-forced decode with LOGITS demand for each evaluated position; KL(base || V4) over base log-probs above -16 nats and first-index argmax agreement, as llama-perplexity --kl-divergence",
            "label": options.label,
            "kv_codec": options.kv_codec.identity(),
            "verify_width": options.verify_width,
            "host": host(),
            "model": options.model,
            "reference": directory,
            "n_ctx": n_ctx,
            "categories": {},
        });
        let mut done: Vec<(&str, Positions)> = Vec::new();
        for (name, base) in &mut bases {
            if bench.vocabulary != base.n_vocab {
                return Err(format!(
                    "{name}: reference vocabulary {} differs from the model's {}",
                    base.n_vocab, bench.vocabulary
                ));
            }
            let chunks = options.chunks.unwrap_or(base.n_chunk).min(base.n_chunk);
            let (positions, per_chunk) =
                category(&mut bench, base, chunks, options.verify_width, name)?;
            let mut summary = positions.summary();
            summary["chunks"] = json!(chunks);
            summary["per_chunk"] = json!(per_chunk);
            report["categories"][*name] = summary;
            done.push((*name, positions));
            if done.len() == CATEGORIES.len() {
                let mut overall = Positions::default();
                for (_, positions) in &done {
                    overall.extend(positions);
                }
                report["overall"] = overall.summary();
                if n_ctx == D4_N_CTX {
                    report["d4"] = verdict(limits, &done, &overall);
                }
            }
            std::fs::write(
                &options.output,
                serde_json::to_string_pretty(&report).expect("report serializes"),
            )
            .map_err(|error| format!("writing {}: {error}", options.output.display()))?;
        }
        Ok(())
    }

    /// Scored rows per `span_mean_kld` entry of a chunk.
    const SPAN: usize = 512;

    fn category(
        bench: &mut Bench,
        base: &mut BaseFile,
        chunks: usize,
        width: usize,
        name: &str,
    ) -> Result<(Positions, Vec<serde_json::Value>), String> {
        let mut positions = Positions::default();
        let mut per_chunk = Vec::with_capacity(chunks);
        for chunk in 0..chunks {
            let tokens = base.chunk_tokens(chunk)?;
            let first = base.first();
            let mut sequence = bench.open_sequence()?;
            for rows in tokens[..first].chunks(super::PREFILL_ROWS) {
                let prefill =
                    Bench::forward(&sequence, WorkKind::Prefill, rows.to_vec(), Demand::NONE);
                bench.step(vec![prefill], &mut [&mut sequence], None)?;
            }
            let (mut chunk_kl, mut chunk_same) = (0.0, 0usize);
            // Mean KL per SPAN scored rows, to locate where along the history a
            // long-context divergence starts.
            let mut span_kl = vec![0.0; base.evaluated().div_ceil(SPAN)];
            let mut row = 0;
            while row < base.evaluated() {
                let group = width.min(base.evaluated() - row);
                let forced = tokens[first + row..first + row + group].to_vec();
                let step = if group == 1 {
                    Bench::forward(&sequence, WorkKind::Decode, forced, Demand::LOGITS)
                } else {
                    Operation::Forward {
                        request: sequence.request,
                        kind: WorkKind::Verify,
                        tokens: forced,
                        position: sequence.position,
                        conditioning: None,
                        demand: Demand::LOGITS | Demand::SELECT,
                        select: (0..group)
                            .map(|offset| Bench::greedy(sequence.position + offset))
                            .collect(),
                        committed: group,
                    }
                };
                let (_, outcomes) = bench.step(vec![step], &mut [&mut sequence], None)?;
                let [Outcome::Forward { rows }] = outcomes.as_slice() else {
                    return Err("scored step returned a non-forward outcome".into());
                };
                if rows.len() != group {
                    return Err("scored step returned a row count unlike its tokens".into());
                }
                for (offset, result) in rows.iter().enumerate() {
                    let logits = result
                        .logits
                        .as_ref()
                        .ok_or("scored step returned no logits")?
                        .read_to_host()
                        .map_err(|error| error.to_string())?;
                    if logits.len() != base.n_vocab {
                        return Err("logits row width differs from the vocabulary".into());
                    }
                    let reference = base.row(chunk, row + offset)?;
                    let (kl, top) = compare(&reference, &logits);
                    positions.push(kl, top, top_gap(&reference));
                    chunk_kl += kl;
                    chunk_same += usize::from(top);
                    span_kl[(row + offset) / SPAN] += kl;
                }
                row += group;
            }
            bench.close(sequence)?;
            let spans = span_kl
                .iter()
                .enumerate()
                .map(|(span, total)| {
                    total / (base.evaluated() - span * SPAN).min(SPAN) as f64
                })
                .collect::<Vec<_>>();
            per_chunk.push(json!({
                "chunk": chunk,
                "mean_kld": chunk_kl / base.evaluated() as f64,
                "same_top": chunk_same as f64 / base.evaluated() as f64,
                "span_rows": SPAN,
                "span_mean_kld": spans,
            }));
            eprintln!(
                "{name} chunk {chunk}: running mean KLD {:.6}, same top {:.4}",
                positions.mean_kld(),
                positions.same_top()
            );
        }
        Ok((positions, per_chunk))
    }
}
