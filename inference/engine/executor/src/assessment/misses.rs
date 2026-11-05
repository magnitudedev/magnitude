//! The miss record: what streamed loads' decodes measured of how this
//! model's routing misses on this machine, and the streamed decode step it
//! prices through the one cost function ([`MissRecord::step_seconds`],
//! [`super::cost`]). A load's own record prices its bank's growth; the store
//! keeps every load's, merged as counts ([`MissRecord::merge`]), in the
//! kernel cache's measurement store keyed by executor and model
//! ([`StoredMisses`]), so a short session adds evidence and never replaces
//! it. Before a model's first streamed load it has none, and its streamed
//! estimate is unmeasured.

use super::cost::ExpertLines;
use super::step::{LayerLoad, Lead, ReadAhead, StepInputs};
use super::streaming_costs::{ReadLine, ReadLineError, RequestReads};
use crate::experts::LayerCurve;
use serde::{Deserialize, Serialize};
use std::fmt;

/// The relative standard error within which a record's miss rate at its
/// load's own capacity is published: 5%, a third of the estimate's ±15%
/// ([`super::estimate`]), so sampling leaves the rest to the model. `M`
/// misses have a relative standard error of `1/√M` (Poisson).
pub const MISS_RATE_ERROR: f64 = 0.05;

/// The most rows a decode launch runs: the decode form's widest
/// (`routed.seismic`'s `DECODE_ROWS`); a wider launch is a prompt step's.
pub const LAUNCH_ROWS: usize = magnitude_kernels::constants::DECODE_ROWS as usize;

/// A load's decode launches alone in their groups, by their rows (index
/// rows − 1), each its seconds less what the cost function exposes of its
/// own misses ([`LayerStep::exposed`]): the launch that misses nothing, the
/// resident launch of its rows and `σ` as the load ran them, measured from
/// every launch whether or not one hit everywhere ([`MissRecord::overhead`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Residuals {
    pub rows: [LaunchResiduals; LAUNCH_ROWS],
}

/// The decode launches of one row count: how many, the sums of their
/// residuals and of their context tokens.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LaunchResiduals {
    pub launches: u64,
    pub seconds: f64,
    pub tokens: u64,
}

impl LaunchResiduals {
    fn plus(self, other: Self) -> Self {
        Self {
            launches: self.launches + other.launches,
            seconds: self.seconds + other.seconds,
            tokens: self.tokens + other.tokens,
        }
    }

    fn less(self, part: Self) -> Option<Self> {
        Some(Self {
            launches: self.launches.checked_sub(part.launches)?,
            seconds: self.seconds - part.seconds,
            tokens: self.tokens.checked_sub(part.tokens)?,
        })
    }
}

/// Why a miss record prices no step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MissRecordError {
    /// Its curve counted no decode step.
    NoSteps,
    /// Its decode requests make no read line.
    ReadLine(ReadLineError),
    /// No decode launch alone in its group measured `σ`.
    NoResiduals,
    /// Its steps' mean depth exceeds what a resident step is priced at.
    Depth { tokens: u64 },
}

/// Why two records do not merge: they are of different shapes, so not of
/// one model's streamed layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissMergeError {
    /// Their streamed layers' counts differ.
    Layers { earlier: usize, later: usize },
    /// Streamed layer `layer`'s expert bytes or depth buckets differ.
    Layer { layer: usize },
    /// The stored record does not hold what the load stored into it last:
    /// another writer replaced it.
    Uncontained,
}

impl fmt::Display for MissMergeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Layers { earlier, later } => {
                write!(formatter, "a record of {later} streamed layers merged onto one of {earlier}")
            }
            Self::Layer { layer } => write!(formatter, "streamed layer {layer}'s shape differs between the records"),
            Self::Uncontained => formatter.write_str("the stored record does not hold what the load stored last"),
        }
    }
}

impl std::error::Error for MissMergeError {}

impl fmt::Display for MissRecordError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSteps => formatter.write_str("a miss record of no decode steps"),
            Self::ReadLine(error) => write!(formatter, "its read line: {error}"),
            Self::NoResiduals => formatter.write_str("no decode launch alone in its group measured σ"),
            Self::Depth { tokens } => write!(formatter, "a mean depth of {tokens} tokens"),
        }
    }
}

impl std::error::Error for MissRecordError {}

impl Residuals {
    /// Count a decode launch of `rows` rows (1 to [`LAUNCH_ROWS`]) of
    /// `seconds` at `tokens` context, of which the cost function exposed
    /// `exposed` of its misses: its residual, `C + σ` as it ran, on which the
    /// record prices a step again ([`MissRecord::step_seconds`]).
    pub fn add(&mut self, rows: usize, seconds: f64, exposed: f64, tokens: u64) {
        let at = &mut self.rows[rows - 1];
        *at = at.plus(LaunchResiduals { launches: 1, seconds: seconds - exposed, tokens });
    }

    /// The launches counted, of every row count.
    pub fn launches(&self) -> u64 {
        self.rows.iter().map(|at| at.launches).sum()
    }

    /// The launches' mean depth, in context tokens; none before any.
    fn depth(&self) -> Option<u64> {
        let tokens = self.rows.iter().map(|at| at.tokens).sum::<u64>();
        tokens.checked_div(self.launches())
    }

    fn plus(&self, other: &Self) -> Self {
        Self { rows: std::array::from_fn(|index| self.rows[index].plus(other.rows[index])) }
    }

    fn less(&self, part: &Self) -> Option<Self> {
        let mut rows = [LaunchResiduals::default(); LAUNCH_ROWS];
        for (index, at) in rows.iter_mut().enumerate() {
            *at = self.rows[index].less(part.rows[index])?;
        }
        Some(Self { rows })
    }
}

/// What one streamed layer of one decode step routed and missed: the
/// experts its route chose and their choices, the misses its host tier
/// held (`hosted`), and those read from the file and landed into the tier
/// (`misses`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LayerStep {
    pub routed: u64,
    pub choices: u64,
    pub hosted: u64,
    pub misses: u64,
    /// Of `misses`, those a hint had staged or begun reading by their
    /// request.
    pub hinted: u64,
    /// The reads this step's hints made for the layer; of them those of an
    /// expert its route chose, which saved a read whether they landed before
    /// the route (a hit) or after (of `hinted`); and their leads, in layers
    /// from the route that hinted to this layer's, summed over the reads.
    pub hint_reads: u64,
    pub used: u64,
    pub lead: u64,
}

impl LayerStep {
    /// What the layer's reads asked of the file and its hints answered: its
    /// misses on demand and every read a hint saved, hit or not, which is
    /// what a bank without read-ahead would have missed.
    pub fn anticipated(&self) -> Anticipated {
        Anticipated { file: self.misses - self.hinted + self.used, hinted: self.used }
    }

    /// What the layer's misses added to its step along `lines`, its file
    /// misses read along `read` as their one request, none hinted
    /// ([`super::step`]): its routed experts each of the step's mean rows
    /// an expert.
    pub fn exposed(&self, lines: &ExpertLines, read: &ReadLine) -> f64 {
        let all = [*lines];
        let inputs = StepInputs {
            compute: 0.,
            head: 0.,
            overlap: 1.,
            read: *read,
            sustained: read.seconds_per_byte(),
            ahead: ReadAhead::NONE,
            lines: &all,
        };
        inputs
            .one(
                &[LayerLoad {
                    routed: self.routed as f64,
                    rows: self.choices as f64 / self.routed as f64,
                    tier: self.hosted,
                    file: self.misses,
                    recall: 0.,
                }],
                &mut [0.],
            )
            .0
    }
}

/// What one streamed layer of a load measured, and what it is priced by.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeasuredLayer {
    /// Its lines on the load's machine ([`ExpertLines`]).
    pub lines: ExpertLines,
    /// Every decode read's depth in the ranking's stack: its misses and
    /// requests at every number of slots as the loads ran them.
    pub curve: LayerCurve,
    /// The reads it asked of the file and those its hints saved
    /// ([`LayerStep::anticipated`]).
    pub anticipated: Anticipated,
}

/// A streamed layer's file reads a bank without read-ahead would miss (`file`)
/// and those a hint read (`hinted`): its recall `R`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Anticipated {
    pub file: u64,
    pub hinted: u64,
}

impl Anticipated {
    pub fn plus(self, other: Self) -> Self {
        Self { file: self.file + other.file, hinted: self.hinted + other.hinted }
    }

    fn less(self, part: Self) -> Option<Self> {
        Some(Self { file: self.file.checked_sub(part.file)?, hinted: self.hinted.checked_sub(part.hinted)? })
    }

    pub fn recall(self) -> f64 {
        match self.file {
            0 => 0.,
            file => self.hinted as f64 / file as f64,
        }
    }
}

/// A load's hint telemetry: the reads hints made, those of an expert the
/// layer's route chose (their precision `P`), and their leads in layers,
/// summed over the reads (`ℓ`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HintCounts {
    pub reads: u64,
    pub used: u64,
    pub lead: u64,
}

impl HintCounts {
    /// The reads of `step`'s layers ([`LayerStep`]).
    pub fn of(step: &[LayerStep]) -> Self {
        step.iter().fold(Self::default(), |sum, layer| sum.plus(Self { reads: layer.hint_reads, used: layer.used, lead: layer.lead }))
    }

    pub fn plus(self, other: Self) -> Self {
        Self { reads: self.reads + other.reads, used: self.used + other.used, lead: self.lead + other.lead }
    }

    fn less(self, part: Self) -> Option<Self> {
        Some(Self {
            reads: self.reads.checked_sub(part.reads)?,
            used: self.used.checked_sub(part.used)?,
            lead: self.lead.checked_sub(part.lead)?,
        })
    }

    /// `P` and `ℓ`: no hint read wastes nothing and leads nothing.
    pub fn ahead(self) -> ReadAhead {
        match self.reads {
            0 => ReadAhead::NONE,
            reads => ReadAhead { precision: (self.used as f64 / reads as f64).max(f64::MIN_POSITIVE), lead: Lead::Layers(self.lead as f64 / reads as f64) },
        }
    }
}

/// What a streamed load's decode measured.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MissRecord {
    /// The decode steps its curves counted: the evidence it is of.
    pub steps: u64,
    /// Those steps' launches' rows, summed (`r₀` over `steps`).
    pub rows: u64,
    /// Each streamed layer's, in decoder order.
    pub layers: Vec<MeasuredLayer>,
    /// Its decode requests' demand reads: the served file's read line.
    pub reads: RequestReads,
    /// Its decode launches less their misses' exposure, by rows.
    pub residuals: Residuals,
    /// Its loads' hints.
    pub hints: HintCounts,
}

/// A decode launch alone in its group, as its step measured it for `σ`
/// ([`Residuals::add`]): its rows, its context tokens, its seconds from its
/// group's beginning or the step before's end, whichever is later, to its
/// end, and what its misses exposed along the run's read and landing lines
/// then ([`LayerStep::exposed`]), which those lines do not stay.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LaunchSample {
    pub rows: usize,
    pub tokens: u64,
    pub seconds: f64,
    pub exposed: f64,
}

/// A streamed decode step a record prices ([`MissRecord::step_seconds`]):
/// `C`, what its layers' misses add (the cost function's, a mean over its
/// steps), and `T`, their sum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StreamedStep {
    pub compute_seconds: f64,
    pub exposed_seconds: f64,
    pub total_seconds: f64,
}

/// A record's counts as a load's decode steps add them, the one assembly
/// of a [`MissRecord`]: the load's watcher feeds it each counted step, and
/// a replay of a run feeds it the same steps as the run recorded them.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct RecordFeed {
    residuals: Residuals,
    /// Each streamed layer's lines the residuals were taken along.
    lines: Option<std::sync::Arc<[ExpertLines]>>,
    /// Each streamed layer's file misses and those hinted, and every layer's
    /// hint reads and leads, of every counted decode step.
    anticipated: Vec<Anticipated>,
    hints: HintCounts,
}

impl RecordFeed {
    /// A counted decode step: its layers' file misses and hints are
    /// counted, and a launch alone it measured, `sample`, enters `σ` less
    /// what it exposed, along the recording's `lines`.
    pub fn decode(&mut self, step: &[LayerStep], sample: Option<LaunchSample>, lines: Option<std::sync::Arc<[ExpertLines]>>) {
        self.anticipated.resize(step.len(), Anticipated::default());
        for (anticipated, layer) in self.anticipated.iter_mut().zip(step) {
            *anticipated = anticipated.plus(layer.anticipated());
        }
        self.hints = self.hints.plus(HintCounts::of(step));
        if let (Some(LaunchSample { rows, tokens, seconds, exposed }), Some(lines)) = (sample, lines) {
            self.residuals.add(rows, seconds, exposed, tokens);
            self.lines = Some(lines);
        }
    }

    /// The decode launches that measured `σ`.
    pub fn launches(&self) -> u64 {
        self.residuals.launches()
    }

    /// Each streamed layer's lines the feed was last taken along; none
    /// before a launch measured `σ`.
    pub fn lines(&self) -> Option<&std::sync::Arc<[ExpertLines]>> {
        self.lines.as_ref()
    }

    /// The record of what was fed, its reads counted in `curve` (the
    /// ranking's stack's) over steps of `rows` rows in all, and `reads` the
    /// served file's: none before a launch measured `σ` along the
    /// recording's lines.
    pub fn record(&self, curve: &crate::experts::MissCurve, rows: u64, reads: RequestReads) -> Option<MissRecord> {
        let lines = self.lines.as_deref()?;
        let layers = curve
            .layers
            .iter()
            .zip(lines)
            .enumerate()
            .map(|(layer, (curve, &lines))| MeasuredLayer {
                lines,
                curve: curve.clone(),
                anticipated: self.anticipated[layer],
            })
            .collect();
        Some(MissRecord { steps: curve.steps, rows, layers, reads, residuals: self.residuals, hints: self.hints })
    }
}

/// A record as the store keeps it: with the executor it ran on and the
/// model it is of, which find it; `M` the model's fingerprint, as its host
/// names it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredMisses<M> {
    pub executor: seismic::Identity,
    pub model: M,
    pub record: MissRecord,
}

impl<M: Serialize + serde::de::DeserializeOwned> seismic::measure::Stored for StoredMisses<M> {
    const SCHEMA: &'static str = "misses.1";
}

impl<M: Serialize + Clone> StoredMisses<M> {
    /// The store key of `executor` and `model`, in the record's form
    /// ([`MissRecord::form`]): a record of another form lies under another
    /// key and is never read.
    pub fn key(executor: &seismic::Identity, model: &M) -> serde_json::Value {
        serde_json::json!({ "misses": { "form": MissRecord::form(), "executor": executor, "model": model } })
    }

    /// The most bytes a record of `executor` and `model` with `layers`
    /// streamed layers of `experts` experts is written as: its widest
    /// ([`MissRecord::widest`]).
    pub fn largest(
        executor: &seismic::Identity,
        model: &M,
        layers: usize,
        experts: usize,
    ) -> serde_json::Result<u64> {
        let widest = Self { executor: executor.clone(), model: model.clone(), record: MissRecord::widest(layers, experts) };
        serde_json::to_vec(&widest).map(|bytes| bytes.len() as u64)
    }
}

/// `value`'s fields and their kinds, whatever its values and lengths.
fn shape(value: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match value {
        Value::Object(fields) => Value::Object(fields.iter().map(|(name, field)| (name.clone(), shape(field))).collect()),
        Value::Array(items) => Value::Array(items.first().map(shape).into_iter().collect()),
        Value::Null => "null".into(),
        Value::Bool(_) => "bool".into(),
        Value::Number(_) => "number".into(),
        Value::String(_) => "string".into(),
    }
}

impl MissRecord {
    /// The form records are stored in: the digest of the fields a record
    /// serializes, each present in [`Self::widest`]. It is derived, so a
    /// change to them cannot leave a record of another form under its key.
    pub fn form() -> &'static str {
        use sha2::{Digest, Sha256};
        static FORM: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        FORM.get_or_init(|| {
            let widest = serde_json::to_value(Self::widest(1, 1)).expect("a record serializes");
            Sha256::digest(shape(&widest).to_string()).iter().map(|byte| format!("{byte:02x}")).collect()
        })
    }

    /// The record of `layers` streamed layers of `experts` experts whose
    /// every integer is `u64::MAX` and every float written at its widest
    /// (`-f64::MIN_POSITIVE`, 24 characters).
    fn widest(layers: usize, experts: usize) -> Self {
        use super::cost::{CpuLines, GpuLines, HostLines, Nanoseconds, PassLine, RowCost, RowLine};
        use super::streaming_costs::{FileLine, RecordLine};
        let (integer, float) = (u64::MAX, -f64::MIN_POSITIVE);
        let line = Nanoseconds::try_from(i32::MAX as u32).expect("a policy word's most");
        let row = RowLine { one: line, row: RowCost::Measured(line) };
        let pass = PassLine { weights: line, row: line };
        let layer = MeasuredLayer {
            lines: ExpertLines {
                host: HostLines {
                    cpu: Some(CpuLines { expert: row, handshake: line }),
                    memory: Some(line),
                    // A line's terms are never negative: at their widest
                    // they are `f64::MIN_POSITIVE` (23 characters).
                    convert: Some(RecordLine::new(f64::MIN_POSITIVE, f64::MIN_POSITIVE).expect("positive terms are a line's")),
                    file: Some(FileLine {
                        landing: RecordLine::new(f64::MIN_POSITIVE, f64::MIN_POSITIVE).expect("positive terms are a line's"),
                    }),
                    record_bytes: integer,
                },
                gpu: GpuLines { link: pass, resident: pass },
            },
            curve: LayerCurve {
                misses: vec![integer; experts + 1],
                requests: vec![integer; experts + 1],
                choices: integer,
            },
            anticipated: Anticipated { file: integer, hinted: integer },
        };
        Self {
            steps: integer,
            rows: integer,
            layers: vec![layer; layers],
            reads: RequestReads {
                requests: integer,
                bytes: float,
                seconds: float,
                delivered: float,
                busy_seconds: float,
            },
            residuals: Residuals {
                rows: [LaunchResiduals { launches: integer, seconds: float, tokens: integer }; LAUNCH_ROWS],
            },
            hints: HintCounts { reads: integer, used: integer, lead: integer },
        }
    }

    /// The fewest slots, a bank's and its host tier's together, at which the
    /// record's misses stop falling: one past the deepest read any layer
    /// counted. None before any read was counted.
    pub fn saturation(&self) -> Option<usize> {
        self.layers
            .iter()
            .filter_map(|layer| layer.curve.misses.iter().rposition(|&misses| misses > 0))
            .max()
            .map(|deepest| deepest + 1)
    }

    /// `σ` as the load ran it: each row count's launches' residuals less
    /// the resident launch of those rows at their mean depth
    /// (`resident(rows, depth)`), over every launch. Why the record has
    /// none (`Ok(Err)`), or why `resident` failed.
    pub fn overhead<E>(
        &self,
        resident: &dyn Fn(u32, u32) -> Result<f64, E>,
    ) -> Result<Result<f64, MissRecordError>, E> {
        let launches = self.residuals.launches();
        if launches == 0 {
            return Ok(Err(MissRecordError::NoResiduals));
        }
        let mut excess = 0.;
        for (rows, at) in (1u32..).zip(&self.residuals.rows) {
            if at.launches == 0 {
                continue;
            }
            let tokens = at.tokens / at.launches;
            let Ok(depth) = u32::try_from(tokens) else {
                return Ok(Err(MissRecordError::Depth { tokens }));
            };
            excess += at.seconds - at.launches as f64 * resident(rows, depth)?;
        }
        Ok(Ok(excess / launches as f64))
    }

    /// What a bank of `slots` slots misses at each streamed layer a step,
    /// `(requests, misses)`, in expectation: what its counted reads missed
    /// at `slots` over the steps counted, as the loads ran them. A decode's
    /// first reads of its experts count among them, at the depths its prompt
    /// left them, and at a small bank they are most of what misses, so the
    /// record prices the decodes this machine runs, however long they are.
    ///
    /// At `rows` rows a step, not the record's ([`Self::rows`]), a step
    /// reads the layer's distinct experts `D(rows)` rather than `D(r₀)`,
    /// `D(r) = E·(1 − (1 − k/E)^r)` the distinct experts `r` rows of `k`
    /// choices each name of `E` ([`Self::spread`]): its reads scale by
    /// `D(rows)/D(r₀)`, and so does every depth, the experts read between
    /// a read and its next, its steps the same, so a bank of
    /// `slots` misses what one of `slots · D(r₀)/D(rows)` did in the record.
    pub fn at(&self, slots: usize, rows: f64) -> Result<Vec<(f64, f64)>, MissRecordError> {
        (0..self.layers.len()).map(|layer| self.layer_at(layer, slots, rows)).collect()
    }

    /// [`Self::at`] of streamed layer `layer` alone: the one reading of a
    /// layer's misses at a size.
    fn layer_at(&self, layer: usize, slots: usize, rows: f64) -> Result<(f64, f64), MissRecordError> {
        if self.steps == 0 {
            return Err(MissRecordError::NoSteps);
        }
        let streamed = &self.layers[layer];
        let spread = self.spread(layer, rows)?;
        let depth = (slots as f64 / spread).round() as usize;
        let (requests, misses) = streamed.curve.at(depth);
        let per_step = |count: u64| count as f64 / self.steps as f64 * spread;
        Ok((per_step(requests), per_step(misses)))
    }

    /// Streamed layer `layer`'s expert reads a step.
    fn reads(&self, layer: usize) -> f64 {
        self.layers[layer].curve.reads() as f64 / self.steps as f64
    }

    /// The decode rows of the record's steps, `r₀`: the mean rows of the
    /// steps its curves counted, the population its choices are of. Why
    /// the record has none: it counted no step.
    pub fn rows(&self) -> Result<f64, MissRecordError> {
        match self.steps {
            0 => Err(MissRecordError::NoSteps),
            steps => Ok(self.rows as f64 / steps as f64),
        }
    }

    /// Streamed layer `layer`'s choices a row, `k`: its choices over its
    /// steps' rows.
    fn choices(&self, layer: usize) -> Result<f64, MissRecordError> {
        Ok(self.layers[layer].curve.choices as f64 / (self.steps as f64 * self.rows()?))
    }

    /// Streamed layer `layer`'s distinct experts a step at `rows` rows over
    /// those at the record's, `D(rows)/D(r₀)` ([`Self::at`]).
    fn spread(&self, layer: usize, rows: f64) -> Result<f64, MissRecordError> {
        let experts = (self.layers[layer].curve.misses.len() - 1) as f64;
        let choices = self.choices(layer)?;
        let distinct = |rows: f64| experts * (1. - (1. - choices / experts).powf(rows));
        Ok(distinct(rows) / distinct(self.rows()?))
    }

    /// What streamed layer `layer` reads over a period of `period` steps
    /// ([`Self::period`]) at `rows` rows holding `held`. Its misses at
    /// `held.slots` ([`Self::at`]), to the nearest whole miss over the
    /// period, come in its requests (rounded up, at least one, at most a
    /// miss each and one a step), as evenly as integers allow; of them,
    /// those a bank of `max(held.slots, held.host)` would miss lie beyond the
    /// host tier and are read from the file (an inclusive tier holds the
    /// stack's top `held.host`, the bank's experts among them), spread
    /// evenly. A step routes the
    /// layer's reads at `rows` rows ([`Self::at`]), its `k · rows` choices
    /// spread over them. A step past the requests misses nothing.
    fn layer_period(&self, layer: usize, period: u64, held: HeldSlots, rows: f64) -> Result<Vec<LayerLoad>, MissRecordError> {
        let streamed = &self.layers[layer];
        let (requests, misses) = self.layer_at(layer, held.slots, rows)?;
        let total = (misses * period as f64).round() as u64;
        if total == 0 {
            return Ok(Vec::new());
        }
        let filed = ((self.layer_at(layer, held.slots.max(held.host), rows)?.1 * period as f64).round() as u64).min(total);
        let requests = ((requests * period as f64).ceil() as u64).clamp(1, total.min(period));
        let reads = self.reads(layer) * self.spread(layer, rows)?;
        let choices = self.choices(layer)? * rows / reads;
        let recall = streamed.anticipated.recall();
        Ok((0..requests)
            .map(|request| {
                let (first, last) = (request * total / requests, (request + 1) * total / requests);
                let file = last * filed / total - first * filed / total;
                LayerLoad {
                    routed: reads,
                    rows: choices,
                    tier: last - first - file,
                    file,
                    recall,
                }
            })
            .collect())
    }

    /// The steps a record's priced period holds: `E`, the widest streamed
    /// layer's experts, at which every layer's rates resolve, as its depth
    /// buckets do in slots.
    fn period(&self) -> usize {
        self.layers.iter().map(|layer| layer.curve.misses.len() - 1).max().unwrap_or(0)
    }

    /// The record's steps over a period of `E` steps ([`Self::period`]) holding `held` at
    /// `rows` rows a step: each step every layer's load there, a layer's
    /// requests spread as evenly over the period as integers allow, the
    /// steps between them missing nothing.
    fn period_steps(&self, held: HeldSlots, rows: f64) -> Result<Vec<Vec<LayerLoad>>, MissRecordError> {
        let period = self.period();
        let mut steps = vec![Vec::with_capacity(self.layers.len()); period];
        for layer in 0..self.layers.len() {
            let loads = self.layer_period(layer, period as u64, held, rows)?;
            let quiet = LayerLoad {
                routed: self.reads(layer) * self.spread(layer, rows)?,
                rows: self.choices(layer)? * rows / (self.reads(layer) * self.spread(layer, rows)?),
                ..LayerLoad::default()
            };
            let mut placed = loads.iter();
            for (at, step) in steps.iter_mut().enumerate() {
                // Request `r` of `n` lies at step `r·E/n`, `n` at most `E`:
                // each is due once.
                let due = (at * loads.len()).div_ceil(period) < ((at + 1) * loads.len()).div_ceil(period);
                step.push(match due {
                    true => *placed.next().expect("a request due is a request placed"),
                    false => quiet,
                });
            }
        }
        Ok(steps)
    }

    /// The record's read-ahead: `P` and `ℓ` of its hints.
    pub fn read_ahead(&self) -> ReadAhead {
        self.hints.ahead()
    }

    /// The record a load stores when the store holds `stored` and the load
    /// last stored `previous` into it: `stored` less `previous`, then
    /// merged with the load's `current` ([`Self::merge`]), so a load that
    /// stores as its record grows counts each of its steps once, however
    /// many other loads stored between. Why they do not merge, or why
    /// `stored` does not hold `previous`.
    pub fn restored(stored: Option<&Self>, previous: Option<&Self>, current: &Self) -> Result<Self, MissMergeError> {
        match (stored, previous) {
            (None, _) => Ok(current.clone()),
            (Some(stored), None) => stored.merge(current),
            (Some(stored), Some(previous)) => stored.less(previous)?.merge(current),
        }
    }

    /// This record less `part`, a record merged into it: every count's
    /// difference. Why `part` is not of its shape or not within it.
    fn less(&self, part: &Self) -> Result<Self, MissMergeError> {
        let shaped = self.layers.len() == part.layers.len()
            && self.layers.iter().zip(&part.layers).all(|(whole, part)| whole.curve.misses.len() == part.curve.misses.len());
        if !shaped {
            return Err(MissMergeError::Uncontained);
        }
        let less = |whole: &[u64], part: &[u64]| -> Option<Vec<u64>> {
            whole.iter().zip(part).map(|(whole, part)| whole.checked_sub(*part)).collect()
        };
        let layers = self
            .layers
            .iter()
            .zip(&part.layers)
            .map(|(whole, part)| {
                Some(MeasuredLayer {
                    lines: whole.lines,
                    curve: LayerCurve {
                        misses: less(&whole.curve.misses, &part.curve.misses)?,
                        requests: less(&whole.curve.requests, &part.curve.requests)?,
                        choices: whole.curve.choices.checked_sub(part.curve.choices)?,
                    },
                    anticipated: whole.anticipated.less(part.anticipated)?,
                })
            })
            .collect::<Option<_>>()
            .ok_or(MissMergeError::Uncontained)?;
        Ok(Self {
            steps: self.steps.checked_sub(part.steps).ok_or(MissMergeError::Uncontained)?,
            rows: self.rows.checked_sub(part.rows).ok_or(MissMergeError::Uncontained)?,
            layers,
            reads: self.reads.since(&part.reads),
            residuals: self.residuals.less(&part.residuals).ok_or(MissMergeError::Uncontained)?,
            hints: self.hints.less(part.hints).ok_or(MissMergeError::Uncontained)?,
        })
    }

    /// The record of this one's loads and then `later`'s: every count
    /// summed (steps, reads by depth and by return and their choices, demand
    /// reads, residuals), each layer's lines `later`'s, the machine's latest
    /// measure of them. Why they are not of one shape.
    pub fn merge(&self, later: &Self) -> Result<Self, MissMergeError> {
        if self.layers.len() != later.layers.len() {
            return Err(MissMergeError::Layers {
                earlier: self.layers.len(),
                later: later.layers.len(),
            });
        }
        let layers = self
            .layers
            .iter()
            .zip(&later.layers)
            .enumerate()
            .map(|(layer, (earlier, later))| {
                let buckets = earlier.curve.misses.len();
                let shaped = [&later.curve.misses, &later.curve.requests, &earlier.curve.requests]
                    .into_iter()
                    .all(|counts| counts.len() == buckets);
                if earlier.lines.host.record_bytes != later.lines.host.record_bytes || !shaped {
                    return Err(MissMergeError::Layer { layer });
                }
                let mut curve = earlier.curve.clone();
                curve.add(&later.curve);
                Ok(MeasuredLayer { lines: later.lines, curve, anticipated: earlier.anticipated.plus(later.anticipated) })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            steps: self.steps + later.steps,
            rows: self.rows + later.rows,
            layers,
            reads: self.reads.merge(&later.reads),
            residuals: self.residuals.plus(&later.residuals),
            hints: self.hints.plus(later.hints),
        })
    }

    /// The one way a record prices a streamed step: `T` of a decode of
    /// `rows` rows holding `held` on a resident step of `compute_seconds`
    /// (`C`), and what each layer's
    /// misses add (the one step model, [`super::step`], its file misses read
    /// along the run's own read line): structure and machine lines alone. The record's
    /// `σ` ([`Self::overhead`]) is what a run measured past them, reported
    /// beside the estimate as its gap, never priced into it. Why the record
    /// prices none.
    pub fn step_seconds(&self, held: HeldSlots, rows: f64, compute_seconds: f64) -> Result<StreamedStep, MissRecordError> {
        let read = self.reads.line().map_err(MissRecordError::ReadLine)?;
        let lines = self.layers.iter().map(|layer| layer.lines).collect::<Vec<_>>();
        let inputs = StepInputs {
            compute: compute_seconds,
            head: 0.,
            overlap: 1.,
            read,
            sustained: read.seconds_per_byte(),
            ahead: self.read_ahead(),
            lines: &lines,
        };
        let step = inputs.run(self.period_steps(held, rows)?.iter().map(Vec::as_slice));
        Ok(StreamedStep { compute_seconds, exposed_seconds: step.total - compute_seconds, total_seconds: step.total })
    }

    /// `C` of a decode like the record's: the one-row resident step
    /// `resident(1, depth)` at its launches' mean depth. Why the record has
    /// none (`Ok(Err)`), or why `resident` failed.
    pub fn compute<E>(&self, resident: &dyn Fn(u32, u32) -> Result<f64, E>) -> Result<Result<f64, MissRecordError>, E> {
        let Some(tokens) = self.residuals.depth() else {
            return Ok(Err(MissRecordError::NoResiduals));
        };
        let Ok(depth) = u32::try_from(tokens) else {
            return Ok(Err(MissRecordError::Depth { tokens }));
        };
        resident(1, depth).map(Ok)
    }

    /// What holding `to` rather than `from` saves a step of a decode like
    /// the record's, in seconds: its step ([`Self::step_seconds`]) at `from`
    /// less at `to`, each on its `C` ([`Self::compute`]).
    /// The one pricer of a bank slot more and of a host slot more. What prices the
    /// saving is the record's miss rates, at its depth. Why the record
    /// prices none (`Ok(Err)`), or why `resident` failed.
    pub fn saving<E>(
        &self,
        from: HeldSlots,
        to: HeldSlots,
        resident: &dyn Fn(u32, u32) -> Result<f64, E>,
    ) -> Result<Result<f64, MissRecordError>, E> {
        let compute = match self.compute(resident)? {
            Ok(compute) => compute,
            Err(error) => return Ok(Err(error)),
        };
        let rows = match self.rows() {
            Ok(rows) => rows,
            Err(error) => return Ok(Err(error)),
        };
        let step = |held: HeldSlots| self.step_seconds(held, rows, compute).map(|step| step.total_seconds);
        Ok(match (step(from), step(to)) {
            (Ok(at), Ok(above)) => Ok(at - above),
            (Err(error), _) | (_, Err(error)) => Err(error),
        })
    }
}

/// What a load holds of each streamed layer's experts: `slots` in the bank,
/// and `host` in a host tier, inclusive of the bank's (none on unified
/// memory): the tier holds the bank's experts too, so it holds more only
/// past `slots`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeldSlots {
    pub slots: usize,
    pub host: usize,
}

impl HeldSlots {
    /// `slots` in the bank and nothing in a host tier.
    pub fn bank(slots: usize) -> Self {
        Self { slots, host: 0 }
    }
}

/// What a stored table's decode steps read (`stored_table::RowStack`): its
/// sampled rows by their depth in its LRU stack, in records of
/// `record_rows` rows (the last bucket deeper than every record), over
/// `steps` steps naming `rows` distinct rows in all, each `row_bytes` as
/// stored. A table of `r` records misses exactly the rows `r` or more deep.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableCurve {
    pub steps: u64,
    pub rows: u64,
    pub sampled: Vec<u64>,
    pub record_rows: u64,
    pub row_bytes: u64,
}

impl TableCurve {
    /// The fewest records at which the table's misses stop falling: one past
    /// the deepest sampled row a record holds. A table of `r` records misses
    /// the sampled rows `r` or more deep ([`Self::step_seconds`]), and the
    /// last bucket, deeper than every record, misses at any size.
    pub fn saturation(&self) -> u64 {
        let held = &self.sampled[..self.sampled.len().saturating_sub(1)];
        held.iter().rposition(|&rows| rows > 0).map_or(0, |deepest| deepest as u64 + 1)
    }

    /// What a decode step's table read costs at `records` records, in
    /// seconds: its misses' one read along `line` (`ReadLine::seconds`), its
    /// latency taken when any of its rows misses and its transfer that of
    /// the rows expected to, a row's miss share the sampled rows' at that
    /// depth, independent of the step's other rows (they are hashed rows of
    /// the table). None before a step counted a sampled row.
    pub fn step_seconds(&self, records: u64, line: &ReadLine) -> Option<f64> {
        let total = self.sampled.iter().sum::<u64>();
        if total == 0 || self.steps == 0 {
            return None;
        }
        let depth = (records as usize).min(self.sampled.len() - 1);
        let share = self.sampled[depth..].iter().sum::<u64>() as f64 / total as f64;
        let rows = self.rows as f64 / self.steps as f64;
        let misses = share * rows;
        let missing = 1. - (1. - share).powf(rows);
        Some(missing * line.latency_seconds() + line.transfer_seconds(misses * self.row_bytes as f64))
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    const READ: ReadLine = match ReadLine::new(4e-4, 1. / 2e9) {
        Ok(line) => line,
        Err(_) => panic!("the tests' read line"),
    };

    /// The reference lines' own records, where they hold the figures the
    /// tests state.
    const RECORD: u64 = ExpertLines::REFERENCE_RECORD;

    fn lines() -> ExpertLines {
        ExpertLines::reference(RECORD)
    }

    /// A streamed layer of `curve`.
    fn layer(curve: LayerCurve) -> MeasuredLayer {
        MeasuredLayer { lines: lines(), curve, anticipated: Anticipated::default() }
    }

    /// A step routing 8 experts at each of four layers, one choice each.
    fn routed() -> LayerStep {
        LayerStep { routed: 8, choices: 8, ..LayerStep::default() }
    }

    /// A launch's residual is its seconds less what the cost function
    /// exposes of its misses: launches that cost exactly that beyond the
    /// launch of their rows that misses nothing leave exactly that launch.
    #[test]
    fn residuals_leave_the_launch_of_their_rows_that_misses_nothing() {
        let free = |rows: usize| 0.015 + 0.002 * rows as f64;
        let lines: std::sync::Arc<[ExpertLines]> = vec![lines(); 4].into();
        let mut feed = RecordFeed::default();
        for step in 0..12u64 {
            let rows = [1, 4][(step % 2) as usize];
            let mut layers = vec![routed(); 4];
            layers[(step % 4) as usize] = LayerStep { hosted: 1 + step % 3, misses: step % 2, ..routed() };
            let exposed = layers.iter().zip(lines.iter()).map(|(layer, lines)| layer.exposed(lines, &READ)).sum::<f64>();
            assert!(exposed > 0.);
            let sample = LaunchSample { rows, tokens: 1000 + step, seconds: free(rows) + exposed, exposed };
            feed.decode(&layers, Some(sample), Some(lines.clone()));
        }
        let residuals = feed.residuals;
        assert_eq!(residuals.launches(), 12);
        assert_eq!(residuals.depth(), Some(1005));
        for rows in [1, 4] {
            let at = residuals.rows[rows - 1];
            assert_eq!(at.launches, 6);
            assert!((at.seconds / 6. - free(rows)).abs() < 1e-12, "{at:?}");
        }
        assert_eq!(Residuals::default().depth(), None);
    }

    /// A step's layer exposes its tier misses beyond its hits on the CPU
    /// and its file misses after their read.
    #[test]
    fn a_layer_step_exposes_its_misses_by_the_cost_function() {
        let lines = lines();
        assert_eq!(routed().exposed(&lines, &READ), 0.);
        // 6 hits (30 µs) beside one tier miss on the CPU (its handshake and
        // 40 µs), then the file miss once read, 40 µs more.
        let one = LayerStep { hosted: 1, misses: 1, ..routed() };
        let filed = READ.latency_seconds() + RECORD as f64 / 2e9;
        assert!((one.exposed(&lines, &READ) - (filed + 40e-6 - 30e-6)).abs() < 1e-12);
        // Two tier misses alone: 4 + 40 + 40 µs.
        let tiered = LayerStep { hosted: 2, ..routed() };
        assert!((tiered.exposed(&lines, &READ) - (84e-6 - 30e-6)).abs() < 1e-12);
    }

    /// `σ` is every launch's residual less the resident launch of its rows
    /// at its row count's depth: verify launches of four rows measure it as
    /// one-row launches do.
    #[test]
    fn sigma_is_each_launch_beyond_the_resident_launch_of_its_rows() {
        let resident = |rows: u32, depth: u32| Ok::<_, ()>(0.010 + 0.002 * rows as f64 + depth as f64 * 1e-6);
        let mut record = unpriced(vec![spread(); 4]);
        record.residuals.rows[0] = LaunchResiduals { launches: 2, seconds: 2. * (0.012 + 1e-3 + 0.003), tokens: 2000 };
        record.residuals.rows[3] = LaunchResiduals { launches: 6, seconds: 6. * (0.018 + 3e-3 + 0.003), tokens: 18000 };
        assert!((record.overhead(&resident).unwrap().unwrap() - 0.003).abs() < 1e-12);
        let only_verify = MissRecord {
            residuals: Residuals { rows: std::array::from_fn(|index| match index {
                3 => record.residuals.rows[3],
                _ => LaunchResiduals::default(),
            }) },
            ..record.clone()
        };
        assert!((only_verify.overhead(&resident).unwrap().unwrap() - 0.003).abs() < 1e-12);
        // `C` is the one-row resident step at the launches' mean depth.
        assert!((only_verify.compute(&resident).unwrap().unwrap() - (0.012 + 3e-3)).abs() < 1e-12);
    }

    /// The read line of the tests' records, and residuals of two one-row
    /// launches of 17 ms at depth 1000.
    fn measured(layers: Vec<MeasuredLayer>) -> MissRecord {
        let mut reads = RequestReads::default();
        for bytes in [1_000_000u64, 2_000_000] {
            reads.add(bytes, 4e-4 + bytes as f64 / 2e9);
        }
        reads.deliver(2_000_000_000, 1.);
        let mut residuals = Residuals::default();
        for _ in 0..2 {
            residuals.add(1, 0.017, 0., 1000);
        }
        MissRecord { reads, residuals, ..unpriced(layers) }
    }

    /// A record prices `T` on the resident step it is given and its misses'
    /// exposure alone, whatever `σ` its launches measured; without a read
    /// line it prices none.
    #[test]
    fn a_record_prices_its_step_on_the_resident_step() {
        // Each layer's one read lies at the top of the stack: a bank of
        // a slot misses nothing.
        let quiet = LayerCurve { misses: vec![1, 0, 0], requests: vec![1, 0, 0], choices: 1 };
        let record = MissRecord { steps: 2, rows: 2, ..measured(vec![layer(quiet); 4]) };
        let step = record.step_seconds(HeldSlots::bank(1), 1., 0.02).unwrap();
        assert_eq!(step.exposed_seconds, 0.);
        assert_eq!(step.total_seconds, 0.02, "{step:?}");
        // `σ` prices nothing: a record no launch measured it on prices the same step.
        let unmeasured = MissRecord { residuals: Residuals::default(), ..record.clone() };
        assert_eq!(unmeasured.step_seconds(HeldSlots::bank(1), 1., 0.02), Ok(step));
        let unread = MissRecord { reads: RequestReads::default(), ..record };
        assert_eq!(
            unread.step_seconds(HeldSlots::bank(1), 1., 0.02),
            Err(MissRecordError::ReadLine(ReadLineError::Unread { requests: 0 }))
        );
    }

    /// A layer of four experts over 8 steps whose 8 reads lie 4 at depth 1,
    /// 3 at depth 2 and 1 at depth 3: a bank of one slot misses a
    /// read a step, of two half one, of three an eighth; at one row a use.
    fn spread() -> MeasuredLayer {
        let curve = LayerCurve { misses: vec![0, 4, 3, 1, 0], requests: vec![0, 2, 2, 1, 0], choices: 8 };
        layer(curve)
    }

    /// A record of one-row steps priced at more rows reads the distinct
    /// experts more rows name, `D(r) = E·(1 − (1 − k/E)^r)`, and finds each
    /// return as much deeper: of four experts, one chosen a row, two rows
    /// name 1.75 where one names one, so a bank of 2 slots misses what one
    /// of 1 did (every read) on 1.75 reads, and one of 4 what one of 2 did
    /// (half). At the record's own rows nothing changes.
    #[test]
    fn a_record_prices_misses_at_the_requests_rows() {
        let record = measured(vec![spread(); 4]);
        assert_eq!(record.rows(), Ok(1.));
        let misses = |slots: usize, rows: f64| record.at(slots, rows).unwrap()[0].1;
        assert_eq!(misses(2, 1.), 0.5);
        assert!((misses(2, 2.) - 1.75).abs() < 1e-12, "{}", misses(2, 2.));
        assert!((misses(4, 2.) - 0.875).abs() < 1e-12, "{}", misses(4, 2.));
    }

    /// A record of `layers` over 8 steps of one row, with no read line.
    fn unpriced(layers: Vec<MeasuredLayer>) -> MissRecord {
        MissRecord {
            steps: 8,
            rows: 8,
            layers,
            reads: RequestReads::default(),
            residuals: one_row(),
            hints: HintCounts::default(),
        }
    }

    /// Residuals of one launch of one row: a record's steps of one row.
    fn one_row() -> Residuals {
        let mut residuals = Residuals::default();
        residuals.add(1, 0.017, 0., 1000);
        residuals
    }

    /// A layer's exposure over its period: at one slot its 4 misses of a
    /// period of 4 steps come in 3 requests of 1, 1 and 2, a step routing one
    /// expert, so no hit hides them. Beside a tier of the bank's size each is
    /// read from the file first; beside a tier of two host slots the 2 that
    /// miss two slots are, the other 2 served from the tier at once.
    #[test]
    fn a_layers_misses_are_priced_over_its_period() {
        let record = unpriced(vec![spread()]);
        let lines = lines();
        let filed = |count: u64| READ.latency_seconds() + READ.transfer_seconds((count * RECORD) as f64);
        let cpu = |count: f64| count * 40e-6 + 4e-6;
        let all = [lines];
        let inputs = StepInputs { compute: 0., head: 0., overlap: 1., read: READ, sustained: READ.seconds_per_byte(), ahead: ReadAhead::NONE, lines: &all };
        let exposed = |held| inputs.run(record.period_steps(held, 1.).unwrap().iter().map(Vec::as_slice)).total;
        let tier = exposed(HeldSlots { slots: 1, host: 4 });
        assert!((tier - (cpu(1.) + cpu(1.) + cpu(2.)) / 4.).abs() < 1e-15, "{tier}");
        let bank = exposed(HeldSlots { slots: 1, host: 1 });
        assert_eq!(bank, exposed(HeldSlots::bank(1)), "a tier of the bank's size holds nothing more");
        assert!((bank - (filed(1) + cpu(1.) + filed(1) + cpu(1.) + filed(2) + cpu(2.)) / 4.).abs() < 1e-15, "{bank}");
        let host = exposed(HeldSlots { slots: 1, host: 2 });
        assert!(tier < host && host < bank, "{tier} {host} {bank}");
        assert_eq!(exposed(HeldSlots::bank(4)), 0.);
    }

    /// One pricer of both holdings: a host slot saves a decode the reads it
    /// spares, and a bank slot the misses it removes, more.
    #[test]
    fn a_host_slot_saves_the_reads_of_the_misses_it_holds() {
        let record = measured(vec![spread(); 4]);
        let resident = |_, _| Ok::<_, ()>(0.016);
        let saved = |from: HeldSlots, to: HeldSlots| record.saving(from, to, &resident).unwrap().unwrap();
        let held = HeldSlots { slots: 1, host: 1 };
        let bank_slot = |held: HeldSlots| HeldSlots { slots: held.slots + 1, ..held };
        let host_slot = |held: HeldSlots| HeldSlots { host: held.host + 1, ..held };
        assert_eq!(saved(HeldSlots::bank(1), held), 0., "an inclusive tier of the bank's size saves nothing");
        let (host, slot) = (saved(held, host_slot(held)), saved(held, bank_slot(held)));
        assert!(host > 0. && slot > host, "host {host} slot {slot}");
        let compute = record.compute(&resident).unwrap().unwrap();
        let step = |held| record.step_seconds(held, record.rows().unwrap(), compute).unwrap().total_seconds;
        assert!((slot - (step(held) - step(bank_slot(held)))).abs() < 1e-15);
        let whole = HeldSlots { slots: 1, host: 4 };
        assert_eq!(saved(whole, host_slot(whole)), 0.);
    }

    /// Loads storing their growing records in turn each replace only what
    /// they stored last, so the store holds every load's latest; a store
    /// that does not hold what a load stored last refuses it.
    #[test]
    fn loads_storing_in_turn_each_keep_their_latest() {
        let (a1, a2) = (counted(2, vec![1, 2, 0], 0.01), counted(5, vec![3, 3, 1], 0.01));
        let (b1, b2) = (counted(3, vec![3, 0, 0], 0.02), counted(4, vec![3, 1, 1], 0.02));
        let stored = MissRecord::restored(None, None, &a1).unwrap();
        let stored = MissRecord::restored(Some(&stored), None, &b1).unwrap();
        let stored = MissRecord::restored(Some(&stored), Some(&a1), &a2).unwrap();
        let stored = MissRecord::restored(Some(&stored), Some(&b1), &b2).unwrap();
        let both = a2.merge(&b2).unwrap();
        assert_eq!(stored.steps, both.steps);
        assert_eq!(stored.layers, both.layers);
        assert_eq!(stored.residuals.rows[0].tokens, both.residuals.rows[0].tokens);
        assert!((stored.residuals.rows[0].seconds - both.residuals.rows[0].seconds).abs() < 1e-12);
        assert_eq!(MissRecord::restored(Some(&a1), Some(&a2), &a2), Err(MissMergeError::Uncontained));
    }

    /// A record of reads of one layer at the depths `misses` counts, `steps`
    /// steps, priced on a read line and residuals of `seconds` a one-row
    /// launch, one a step.
    fn counted(steps: u64, misses: Vec<u64>, seconds: f64) -> MissRecord {
        let mut reads = RequestReads::default();
        reads.add(1_000_000, 1e-3);
        reads.deliver(2_000_000_000, 1.);
        MissRecord {
            steps,
            rows: steps,
            layers: vec![layer(LayerCurve { requests: misses.clone(), choices: misses.iter().sum(), misses })],
            reads,
            residuals: Residuals {
                rows: std::array::from_fn(|index| match index {
                    0 => LaunchResiduals { launches: steps, seconds: seconds * steps as f64, tokens: 100 * steps },
                    _ => LaunchResiduals::default(),
                }),
            },
            hints: HintCounts::default(),
        }
    }

    /// Layers of different widths each keep their own miss rate over the
    /// one period the step is priced on.
    #[test]
    fn layers_of_different_widths_keep_their_miss_rates() {
        let widened = |counts: Vec<u64>| [counts, vec![0; 12]].concat();
        let narrow = spread();
        let curve = LayerCurve {
            misses: widened(narrow.curve.misses.clone()),
            requests: widened(narrow.curve.requests.clone()),
            choices: narrow.curve.choices,
        };
        let wide = layer(curve);
        let record = measured(vec![narrow, wide]);
        let rate = |layer: usize, steps: &[Vec<LayerLoad>]| {
            steps.iter().map(|step| (step[layer].tier + step[layer].file) as f64).sum::<f64>() / steps.len() as f64
        };
        let held = HeldSlots::bank(1);
        let steps = record.period_steps(held, 1.).unwrap();
        assert_eq!(steps.len(), 16);
        for (layer, &(_, misses)) in record.at(1, 1.).unwrap().iter().enumerate() {
            assert!((rate(layer, &steps) - misses).abs() < 1. / 16., "layer {layer}: {} against {misses}", rate(layer, &steps));
        }
    }

    /// Merging sums every count, takes the later load's lines, and refuses
    /// records of other shapes.
    #[test]
    fn records_merge_as_counts() {
        let earlier = counted(4, vec![1, 3, 0], 0.01);
        let mut later = counted(6, vec![4, 0, 1], 0.02);
        later.layers[0].lines.gpu.link.weights = crate::assessment::Nanoseconds::try_from(10_000).unwrap();
        let merged = earlier.merge(&later).unwrap();
        assert_eq!(merged.steps, 10);
        let layer = &merged.layers[0];
        assert_eq!(layer.curve.misses, [5, 3, 1]);
        assert_eq!(layer.curve.requests, [5, 3, 1]);
        assert_eq!(layer.lines, later.layers[0].lines);
        assert_eq!(merged.reads.requests, 2);
        assert_eq!(merged.residuals.launches(), 10);
        assert_eq!(merged.residuals.rows[0].tokens, 1000);
        assert!((merged.residuals.rows[0].seconds - 0.16).abs() < 1e-12);
        let two = MissRecord { layers: vec![earlier.layers[0].clone(); 2], ..earlier.clone() };
        assert_eq!(earlier.merge(&two), Err(MissMergeError::Layers { earlier: 1, later: 2 }));
        let deeper = counted(1, vec![1, 0, 0, 0], 0.01);
        assert_eq!(earlier.merge(&deeper), Err(MissMergeError::Layer { layer: 0 }));
    }

    /// One layer of `experts` experts, each step routing `chosen` of them
    /// by a skewed draw that favours the experts of the last step (a
    /// stationary routing with locality), after a prompt that read every
    /// expert: `steps` decode steps through the stack, as a record of those
    /// steps.
    fn session(steps: u64, state: &mut u64) -> MissRecord {
        let (experts, chosen) = (32, 4);
        let mut stack = crate::experts::Stack::alone(1, experts, experts);
        let prompt = crate::experts::RoutedRead {
            sequence: 0,
            layer: 0,
            experts: (0..experts).collect(),
            choices: experts as u64,
        };
        drop(stack.read_group(&prompt, &vec![1; experts]));
        let mut uniform = || {
            *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (*state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut last = Vec::new();
        for sequence in 1..=steps as u64 {
            let mut routed = Vec::new();
            while routed.len() < chosen {
                let expert = match uniform() < 0.5 && !last.is_empty() {
                    true => last[(uniform() * last.len() as f64) as usize],
                    false => (uniform().powi(3) * experts as f64) as usize,
                };
                if !routed.contains(&expert) {
                    routed.push(expert);
                }
            }
            let read = crate::experts::RoutedRead {
                sequence,
                layer: 0,
                experts: routed.clone(),
                choices: chosen as u64,
            };
            stack.read(&read, Some(1));
            last = routed;
        }
        let curve = stack.curve();
        let mut record = counted(steps, vec![0; experts + 1], 0.01);
        record.layers[0].curve = curve.layers[0].clone();
        record
    }

    /// Misses a step at `slots` a record estimates.
    fn rate(record: &MissRecord, slots: usize) -> f64 {
        record.at(slots, record.rows().unwrap()).unwrap()[0].1
    }

    /// A decode's first reads of its experts miss where its prompt left them,
    /// and a record counts them: a short session misses more a step than a
    /// long one of the same routing, a bank holding every expert misses
    /// nothing, and a session merged onto another prices their steps as run.
    #[test]
    fn a_record_counts_a_decodes_first_reads() {
        let mut state = 0x2545_f491_u64;
        let long = session(20_000, &mut state);
        let short = session(200, &mut state);
        for slots in [4, 8, 12, 16] {
            assert!(rate(&short, slots) > rate(&long, slots), "{slots} slots");
            let merged = rate(&long.merge(&short).unwrap(), slots);
            let run = (rate(&long, slots) * 20_000. + rate(&short, slots) * 200.) / 20_200.;
            assert!((merged - run).abs() <= 1e-9, "{slots} slots: merged {merged} against {run}");
        }
        assert_eq!(rate(&short, 32), 0.);
    }

    /// A record's form is its fields': another field is another key, and a
    /// record's values change neither.
    #[test]
    fn a_records_form_follows_its_fields() {
        let widest = serde_json::to_value(MissRecord::widest(1, 1)).unwrap();
        assert_eq!(shape(&widest), shape(&serde_json::to_value(MissRecord::widest(3, 9)).unwrap()));
        let mut changed = widest.clone();
        changed["layers"][0]["returns"] = serde_json::json!({ "returned": [0] });
        assert_ne!(shape(&widest), shape(&changed));
    }
}
