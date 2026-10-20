//! The precision probe (development feature `precision-probe`): does a tuned
//! selection cost what the tuning trace's precision model says?
//!
//! The rendered fixtures run through `logits` twice: every unit at its
//! declared default (`TuningPin::Defaults`), then with the trace's chosen
//! configurations pinned (all of one load's, or the units matching
//! `--unit`; every other unit runs its default). Per scored position it
//! reports the KL of the pinned distributions from the defaults' and, with
//! `--reference`, the added KL against a reference (KL(reference ‖ pinned) −
//! KL(reference ‖ defaults)). Squared error ratios and α² are dimensionless
//! contributions relative to resolution noise, not predicted KL. Missing
//! precision measurements remain null. Quality figures follow the metric rule
//! of `specs/26-10-09/single-run-tuning.md` §5. Everything goes to
//! `precision_probe.json` in the run directory (per-position values
//! included) and to the result record's `precision_probe` measurement.

use crate::Inputs;
use crate::engine::EngineOptions;
use crate::logits::{self, LogitsOptions};
use magnitude_executor::pinned_tuning::{self, PinnedConfiguration, TuningPin};
use seismic::Configuration;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(clap::Args)]
pub struct ProbeArgs {
    /// Size each fixture to this many tokens.
    #[arg(long)]
    context: Option<usize>,
    /// The tuning trace (or run directory) whose chosen configurations are
    /// pinned.
    #[arg(long, value_name = "PATH")]
    selection: PathBuf,
    /// Which load of the trace (1-based).
    #[arg(long, default_value_t = 1)]
    load: usize,
    /// Pin only the units matching this (any part of `entry [bindings]
    /// {statics}`); every unit by default.
    #[arg(long)]
    unit: Option<String>,
    /// A reference logits output (`STEM.f32`, `STEM.json`, and a passing
    /// `reference.json` beside them) of the same model and rendered fixtures.
    #[arg(long, value_name = "STEM")]
    reference: Option<PathBuf>,
    /// Fixtures decoded together.
    #[arg(long, default_value_t = 8)]
    batch: usize,
}

pub struct Probe {
    pub result: Value,
    pub rendered_digest: String,
    pub device: String,
}

pub fn run(inputs: &Inputs, options: &EngineOptions, args: &ProbeArgs) -> Result<Probe, String> {
    let load = Selection::read(&args.selection, args.load)?;
    let selection = load
        .units
        .iter()
        .filter(|unit| {
            args.unit
                .as_ref()
                .is_none_or(|pattern| unit.unit.name().contains(pattern.as_str()))
        })
        .collect::<Vec<_>>();
    if selection.is_empty() {
        return Err("the selection pins no unit".into());
    }
    let rendered = inputs.render(args.context)?;
    let rendered_digest = inputs.write_rendered(&rendered)?;
    let mut reference = args
        .reference
        .as_deref()
        .map(|stem| {
            ReferenceOutput::open(stem, &inputs.artifacts.artifacts.model, &rendered_digest)
        })
        .transpose()?;
    let logits = |stem: &str| {
        logits::run(
            &inputs.artifacts.artifacts,
            options,
            &LogitsOptions {
                batch: args.batch,
                verify_width: 1,
            },
            &rendered,
            &rendered_digest,
            &inputs.run.path(stem),
        )
    };
    pinned_tuning::install(TuningPin::Defaults);
    let device = logits("defaults")?;
    pinned_tuning::install(TuningPin::Replay(
        selection
            .iter()
            .map(|selected| PinnedConfiguration {
                entry: selected.unit.entry.clone(),
                bindings: selected.unit.bindings.clone(),
                statics: selected.unit.statics.clone(),
                params: selected.configuration.params.clone(),
                launch_params: selected
                    .configuration
                    .launches
                    .iter()
                    .enumerate()
                    .flat_map(|(launch, params)| {
                        params
                            .iter()
                            .map(move |(name, value)| ((launch, name.clone()), *value))
                    })
                    .collect(),
            })
            .collect(),
    ));
    logits("pinned")?;
    let mut defaults = LogitsOutput::open(&inputs.run.path("defaults"))?;
    let mut pinned = LogitsOutput::open(&inputs.run.path("pinned"))?;
    for other in std::iter::once(&pinned).chain(reference.iter().map(|reference| &reference.logits))
    {
        defaults.require_matching(other)?;
    }
    let mut from_defaults = Vec::with_capacity(defaults.rows.len());
    let mut added = Vec::new();
    for row in 0..defaults.rows.len() {
        let (base, candidate) = (
            log_softmax(&defaults.row(row)?),
            log_softmax(&pinned.row(row)?),
        );
        from_defaults.push(kl(&base, &candidate));
        if let Some(reference) = reference.as_mut() {
            let truth = log_softmax(&reference.logits.row(row)?);
            added.push(kl(&truth, &candidate) - kl(&truth, &base));
        }
    }
    let units = selection
        .iter()
        .map(|selected| {
            json!({
                "unit": selected.unit,
                "configuration": selected.configuration,
                "error_ratio": selected.error_ratio,
                "contribution": selected.error_ratio.map(|ratio| ratio * ratio),
            })
        })
        .collect::<Vec<_>>();
    let predicted = selection
        .iter()
        .map(|selected| selected.error_ratio.map(|ratio| ratio * ratio))
        .sum::<Option<f64>>();
    let positions = json!({
        "rows": defaults.rows,
        "kl_from_defaults": from_defaults,
        "added_kl_against_reference": reference.as_ref().map(|_| &added),
    });
    let result = json!({
        "selection": args.selection,
        "load": args.load,
        "alpha": load.alpha,
        "admission": load.admission,
        "contribution_units": "squared_error_ratio",
        "bound_per_unit": load.alpha.map(|alpha| alpha * alpha),
        "units": units,
        "predicted_contribution": predicted,
        "reference": reference.as_ref().map(|reference| &reference.record),
        "kl_from_defaults": statistics(&mut from_defaults),
        "added_kl_against_reference": reference.as_ref().map(|_| statistics(&mut added)),
    });
    let path = inputs.run.path("precision_probe.json");
    std::fs::write(
        &path,
        serde_json::to_vec(&json!({ "summary": result, "positions": positions }))
            .expect("the probe serializes"),
    )
    .map_err(|error| format!("writing {}: {error}", path.display()))?;
    Ok(Probe {
        result,
        rendered_digest,
        device,
    })
}

/// A unit as the tuning trace names it.
#[derive(Clone, Debug, Deserialize, serde::Serialize)]
struct UnitId {
    entry: String,
    bindings: String,
    statics: BTreeMap<String, u64>,
}

impl UnitId {
    fn name(&self) -> String {
        let statics = self
            .statics
            .iter()
            .map(|(name, value)| format!("{name}={value}"))
            .collect::<Vec<_>>()
            .join(",");
        format!("{} [{}] {{{statics}}}", self.entry, self.bindings)
    }
}

/// One unit of the selection: its chosen configuration and error ratio.
struct Selected {
    unit: UnitId,
    configuration: Configuration,
    error_ratio: Option<f64>,
}

/// One load's chosen configurations, from the tuning trace (contract
/// "Tuning trace").
struct Selection {
    alpha: Option<f64>,
    admission: Option<String>,
    units: Vec<Selected>,
}

impl Selection {
    /// Load `number` (1-based) of the trace at `path`, a run directory or
    /// a `tuning.jsonl`.
    fn read(path: &Path, number: usize) -> Result<Self, String> {
        let path = if path.is_dir() {
            path.join(crate::tuning::TRACE)
        } else {
            path.to_path_buf()
        };
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("tuning trace {}: {error}", path.display()))?;
        Self::parse(&text, number)
            .map_err(|error| format!("tuning trace {}: {error}", path.display()))
    }

    fn parse(text: &str, number: usize) -> Result<Self, String> {
        #[derive(Deserialize)]
        struct Chosen {
            configuration: Configuration,
        }
        #[derive(Deserialize)]
        #[serde(tag = "event", rename_all = "snake_case")]
        enum Event {
            Began {
                schema: Option<String>,
                alpha: Option<f64>,
                admission: Option<String>,
            },
            Finished {
                unit: UnitId,
                chosen: Chosen,
                chosen_error_ratio: Option<f64>,
            },
            #[serde(other)]
            Other,
        }
        let mut loads: Vec<Self> = Vec::new();
        for (index, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let event: Event = serde_json::from_str(line)
                .map_err(|error| format!("line {}: {error}", index + 1))?;
            match event {
                Event::Began {
                    schema,
                    alpha,
                    admission,
                } => {
                    if schema.as_deref() != Some(TRACE_SCHEMA) {
                        return Err(format!(
                            "line {}: schema {schema:?}, not {TRACE_SCHEMA}",
                            index + 1
                        ));
                    }
                    if alpha.is_some_and(|alpha| !alpha.is_finite() || alpha <= 0.0) {
                        return Err(format!(
                            "line {}: alpha must be finite and positive",
                            index + 1
                        ));
                    }
                    loads.push(Self {
                        alpha,
                        admission,
                        units: Vec::new(),
                    });
                }
                Event::Finished {
                    unit,
                    chosen,
                    chosen_error_ratio,
                } => {
                    if chosen_error_ratio.is_some_and(|ratio| !ratio.is_finite() || ratio < 0.0) {
                        return Err(format!(
                            "line {}: error ratio must be finite and nonnegative",
                            index + 1
                        ));
                    }
                    loads
                        .last_mut()
                        .ok_or_else(|| format!("line {}: a unit precedes its load", index + 1))?
                        .units
                        .push(Selected {
                            unit,
                            configuration: chosen.configuration,
                            error_ratio: chosen_error_ratio,
                        });
                }
                Event::Other => {}
            }
        }
        let count = loads.len();
        number
            .checked_sub(1)
            .filter(|index| *index < count)
            .map(|index| loads.swap_remove(index))
            .ok_or_else(|| format!("no load {number} among {count}"))
    }
}

/// The schema of the traces this reader reads: the producer's own.
const TRACE_SCHEMA: &str = magnitude_executor::TRACE_SCHEMA;

#[derive(Deserialize)]
struct Metadata {
    producer: String,
    model: String,
    rendered_digest: String,
    vocabulary: usize,
    rows: Vec<Row>,
}

#[derive(Debug, PartialEq, Eq, Deserialize, serde::Serialize)]
struct Row {
    fixture_id: String,
    position: usize,
}

/// A contract logits output, read row by row.
struct LogitsOutput {
    file: File,
    path: PathBuf,
    producer: String,
    model: String,
    vocabulary: usize,
    rendered_digest: String,
    rows: Vec<Row>,
}

impl LogitsOutput {
    fn open(stem: &Path) -> Result<Self, String> {
        let json = stem.with_extension("json");
        let metadata: Metadata = serde_json::from_slice(
            &std::fs::read(&json).map_err(|error| format!("{}: {error}", json.display()))?,
        )
        .map_err(|error| format!("{}: {error}", json.display()))?;
        if metadata.vocabulary == 0 || metadata.rows.is_empty() {
            return Err(format!(
                "{}: logits require a nonempty vocabulary and scored rows",
                json.display()
            ));
        }
        let data = stem.with_extension("f32");
        let file = File::open(&data).map_err(|error| format!("{}: {error}", data.display()))?;
        let expected = metadata
            .rows
            .len()
            .checked_mul(metadata.vocabulary)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| format!("{}: logits dimensions overflow", json.display()))?;
        let actual = file
            .metadata()
            .map_err(|error| format!("{}: {error}", data.display()))?
            .len();
        if actual != expected as u64 {
            return Err(format!(
                "{}: {actual} bytes, expected {expected} for its rows and vocabulary",
                data.display()
            ));
        }
        Ok(Self {
            file,
            path: data,
            producer: metadata.producer,
            model: metadata.model,
            vocabulary: metadata.vocabulary,
            rendered_digest: metadata.rendered_digest,
            rows: metadata.rows,
        })
    }

    fn require_matching(&self, other: &Self) -> Result<(), String> {
        if self.model != other.model
            || self.rendered_digest != other.rendered_digest
            || self.vocabulary != other.vocabulary
            || self.rows != other.rows
        {
            return Err(format!(
                "{} and {} differ in model, rendered fixtures, vocabulary or scored rows",
                self.path.display(),
                other.path.display()
            ));
        }
        Ok(())
    }

    fn row(&mut self, row: usize) -> Result<Vec<f32>, String> {
        if row >= self.rows.len() {
            return Err(format!("{}: no logits row {row}", self.path.display()));
        }
        let mut bytes = vec![0u8; self.vocabulary * 4];
        self.file
            .seek(SeekFrom::Start((row * bytes.len()) as u64))
            .and_then(|_| self.file.read_exact(&mut bytes))
            .map_err(|error| error.to_string())?;
        let values = bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().expect("four bytes")))
            .collect::<Vec<_>>();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "{}: nonfinite logits at row {row} ({:?})",
                self.path.display(),
                self.rows[row]
            ));
        }
        Ok(values)
    }
}

#[derive(Deserialize, serde::Serialize)]
struct ReferenceRecord {
    model: String,
    rendered_digest: String,
    family: String,
    implementation_digest: String,
    dtype: String,
    sanity: Sanity,
}

#[derive(Deserialize, serde::Serialize)]
struct Sanity {
    #[serde(flatten)]
    aggregate: SanityMeasurement,
    bounds: SanityBounds,
    fixtures: Vec<FixtureSanity>,
}

#[derive(Deserialize, serde::Serialize)]
struct SanityBounds {
    perplexity_max: f64,
    top1_min: f64,
}

#[derive(Deserialize, serde::Serialize)]
struct SanityMeasurement {
    perplexity: f64,
    top1_accuracy: f64,
    passed: bool,
}

impl SanityMeasurement {
    fn passes(&self, bounds: &SanityBounds) -> bool {
        self.passed
            && self.perplexity.is_finite()
            && self.perplexity > 0.0
            && self.perplexity <= bounds.perplexity_max
            && self.top1_accuracy.is_finite()
            && (0.0..=1.0).contains(&self.top1_accuracy)
            && self.top1_accuracy >= bounds.top1_min
    }
}

#[derive(Deserialize, serde::Serialize)]
struct FixtureSanity {
    fixture_id: String,
    #[serde(flatten)]
    measurement: SanityMeasurement,
}

/// An identified reference whose recorded gate passed for every scored fixture.
struct ReferenceOutput {
    logits: LogitsOutput,
    record: ReferenceRecord,
}

impl ReferenceOutput {
    fn open(stem: &Path, model: &str, rendered_digest: &str) -> Result<Self, String> {
        let logits = LogitsOutput::open(stem)?;
        let path = stem.with_file_name("reference.json");
        let record: ReferenceRecord = serde_json::from_slice(
            &std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?,
        )
        .map_err(|error| format!("{}: {error}", path.display()))?;
        if record.model != model
            || logits.model != model
            || record.rendered_digest != rendered_digest
            || logits.rendered_digest != rendered_digest
            || record.family.is_empty()
            || record.implementation_digest.is_empty()
            || record.dtype != "float32"
            || logits.producer != format!("reference:{}", record.family)
        {
            return Err(format!(
                "{}: reference identity differs from the model, rendered input or logits producer",
                path.display()
            ));
        }
        let bounds = &record.sanity.bounds;
        if !bounds.perplexity_max.is_finite()
            || bounds.perplexity_max <= 0.0
            || !bounds.top1_min.is_finite()
            || !(0.0..=1.0).contains(&bounds.top1_min)
            || !record.sanity.aggregate.passes(bounds)
        {
            return Err(format!(
                "{}: reference failed its sanity gate",
                path.display()
            ));
        }
        let expected = logits
            .rows
            .iter()
            .map(|row| row.fixture_id.as_str())
            .collect::<BTreeSet<_>>();
        let mut measured = BTreeSet::new();
        for fixture in &record.sanity.fixtures {
            if !measured.insert(fixture.fixture_id.as_str()) || !fixture.measurement.passes(bounds)
            {
                return Err(format!(
                    "{}: reference sanity failed or duplicated fixture {}",
                    path.display(),
                    fixture.fixture_id
                ));
            }
        }
        if measured != expected {
            return Err(format!(
                "{}: reference sanity does not cover exactly the scored fixtures",
                path.display()
            ));
        }
        Ok(Self { logits, record })
    }
}

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum = logits
        .iter()
        .map(|&logit| (logit as f64 - max).exp())
        .sum::<f64>();
    let log_sum = max + sum.ln();
    logits.iter().map(|&logit| logit as f64 - log_sum).collect()
}

/// KL(p ‖ q) of two log distributions.
fn kl(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).map(|(&p, &q)| p.exp() * (p - q)).sum()
}

/// The metric rule: mean, median, p90, mean without the ten worst
/// positions, and the positions above 0.5 nats.
fn statistics(values: &mut [f64]) -> Value {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    let at = |fraction: f64| {
        let index = (n - 1) as f64 * fraction;
        let low = index.floor() as usize;
        let high = index.ceil() as usize;
        values[low] + (values[high] - values[low]) * (index - low as f64)
    };
    let trimmed = &values[..n.saturating_sub(10)];
    json!({
        "positions": n,
        "mean": values.iter().sum::<f64>() / n as f64,
        "median": at(0.5),
        "p90": at(0.9),
        "mean_without_10_worst": (!trimmed.is_empty()).then(|| trimmed.iter().sum::<f64>() / trimmed.len() as f64),
        "above_0_5_nats": values.iter().filter(|&&value| value > 0.5).count(),
        "max": values[n - 1],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "magnitude-probe-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn json(&self, name: &str, value: &Value) {
            std::fs::write(self.0.join(name), serde_json::to_vec(value).unwrap()).unwrap();
        }

        fn reference(&self) -> (PathBuf, Value, Value) {
            let metadata = json!({
                "producer": "reference:fixture", "model": "fixture:gguf:q4",
                "rendered_digest": "rendered-fixture-digest", "vocabulary": 3,
                "rows": [{"fixture_id": "chat", "position": 1}],
            });
            let record = json!({
                "model": "fixture:gguf:q4", "rendered_digest": "rendered-fixture-digest",
                "family": "fixture", "implementation_digest": "reference-implementation-digest", "dtype": "float32",
                "sanity": {"perplexity": 2.0, "top1_accuracy": 0.5, "passed": true,
                    "bounds": {"perplexity_max": 300.0, "top1_min": 0.1},
                    "fixtures": [{"fixture_id": "chat", "perplexity": 2.0, "top1_accuracy": 0.5, "passed": true}]},
            });
            self.json("logits.json", &metadata);
            self.json("reference.json", &record);
            std::fs::write(
                self.0.join("logits.f32"),
                [1.0f32, 2.0, 3.0].map(f32::to_le_bytes).concat(),
            )
            .unwrap();
            (self.0.join("logits"), metadata, record)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn actual_tuning_trace_producer_can_be_pinned_without_inventing_precision() {
        use magnitude_executor::{
            ResolvedNumericalProfile, TunedEntry, TuningEvent, TuningOrigin, TuningSettings,
            TuningTrace,
        };
        let scratch = Scratch::new();
        let path = scratch.0.join("tuning.jsonl");
        let trace = TuningTrace::begin(
            &path,
            magnitude_executor::TracedWork::Search,
            TuningSettings::STANDARD,
            ResolvedNumericalProfile::standard(),
        )
        .unwrap();
        let configuration = Configuration {
            statics: BTreeMap::from([("G".into(), 8)]),
            params: BTreeMap::from([("SIMDS".into(), 8)]),
            launches: vec![BTreeMap::from([("ROWS".into(), 4)])],
        };
        let receipt = std::sync::Arc::new(seismic::TuningResult {
            disposition: seismic::TuningDisposition::QualifiedCandidate,
            tuning_identity: "device".into(),
            entry: "attention".into(),
            backend: "metal".into(),
            points: vec![seismic::PointRecord {
                label: "m1".into(), weight: 1.0, class: None,
            }],
            requirement: seismic::NumericalRequirement::RoundingSteps {
                alpha: 4.0,
                activation: Some(seismic::DType::BF16),
            },
            measurement_identity: "witness".into(),
            // The chosen configuration's one measurement: its bf16 result
            // two rounding steps from the default's.
            observations: vec![seismic::NumericalObservation {
                point: "m1".into(),
                key: seismic::LaunchKey {
                    launches: vec![0],
                    values: BTreeMap::from([("SIMDS".into(), 8)]),
                },
                rotations: vec![vec![seismic::SubjectMeasurement {
                    subject: "r0".into(),
                    comparison: seismic::SubjectComparison::Differs(seismic::Difference {
                        stored: Some(seismic::DType::BF16),
                        elements: 16,
                        differing: 16,
                        exact_mismatches: 0,
                        non_finite_mismatches: 0,
                        error_power: 4.0,
                        reference_power: 16.0,
                        maximum_absolute_error: 0.5,
                        step_power: vec![seismic::StepPower {
                            dtype: seismic::DType::BF16,
                            power: 1.0,
                        }],
                    }),
                }]],
            }],
            served: vec![configuration.clone()],
            reselection: None,
            implementation_identity: "native-source".into(),
            parameters: Vec::new(),
            configurations: vec![seismic::ConfigurationRecord {
                formation: None,
                configuration: configuration.clone(),
                artifact: Some("retained-artifact".into()),
                observations: vec![0],
                outcome: seismic::Outcome::Measured {
                    points: Vec::new(),
                    confirmed: Vec::new(),
                },
            }],
            overall: configuration.clone(),
            method: seismic::TuningMethod::Census,
            time: seismic::TuningTime::default(),
            coverage: None,
        });
        let mut tuned = TunedEntry {
            receipt: magnitude_executor::Receipt::Held(receipt),
            entry: "attention",
            bindings: "Q=bf16".into(),
            overall: Configuration {
                statics: BTreeMap::from([("G".into(), 8)]),
                params: BTreeMap::from([("SIMDS".into(), 8)]),
                launches: vec![BTreeMap::from([("ROWS".into(), 4)])],
            },
            statics: BTreeMap::from([("G".into(), 8)]),
            origin: TuningOrigin::Searched,
            qualified: true,
            stop: Some(seismic::SearchStop::Expired),
            coverage: Some(seismic::CoverageCheck {
                outcome: seismic::CoverageOutcome::Passed,
                failed: 1,
            }),
            progress: None,
            measured: 2,
            excluded: 1,
            rejections: 1,
            first_rejection: None,
            seconds: 0.5,
            time: seismic::TuningTime::default(),
        };
        trace.event(&TuningEvent::Finished(tuned.clone()));
        tuned.entry = "fallback";
        tuned.qualified = false;
        tuned.coverage.as_mut().unwrap().outcome = seismic::CoverageOutcome::Failed;
        trace.event(&TuningEvent::Finished(tuned.clone()));
        tuned.entry = "cached";
        tuned.origin = TuningOrigin::Stored;
        trace.event(&TuningEvent::Finished(tuned.clone()));
        trace.conclude(&[tuned]).unwrap();

        let selection = Selection::read(&path, 1).unwrap();
        assert_eq!(selection.units.len(), 3);
        assert_eq!(selection.admission.as_deref(), Some("rounding_steps"));
        assert_eq!(selection.units[0].configuration.params["SIMDS"], 8);
        assert_eq!(selection.units[0].configuration.launches[0]["ROWS"], 4);
        // The trace states what was measured of each unit's choice: its
        // largest error in the rounding steps that govern its admission.
        assert!(
            selection
                .units
                .iter()
                .all(|unit| unit.error_ratio == Some(2.0))
        );
        let records = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(records[1]["chosen"]["reason"], "fastest_admissible");
        assert_eq!(records[2]["chosen"]["reason"], "default");
        assert_eq!(records[3]["chosen"]["reason"], "stored");
        assert_eq!(records[1]["coverage"]["outcome"], "passed");
        assert_eq!(records[2]["coverage"]["outcome"], "failed");
        assert_eq!(records[1]["coverage"]["failed_count"], 1);
        assert!(records[1]["coverage"]["failed"].is_null());
        assert_eq!(records[1]["points"][0]["label"], "m1");
        // A candidate states its artifact once and names its observations
        // by position among the unit's, each stated once with its key.
        let candidate = &records[1]["candidates"][0];
        assert_eq!(candidate["artifact"], "retained-artifact");
        assert_eq!(candidate["observations"], serde_json::json!([0]));
        assert_eq!(candidate["error_ratio"], 2.0);
        assert_eq!(candidate["admitted"], true);
        assert!(candidate["violation"].is_null());
        assert_eq!(records[1]["measurement_identity"], "witness");
        assert_eq!(records[1]["observations"][0]["point"], "m1");
        assert_eq!(records[1]["observations"][0]["key"]["values"]["SIMDS"], 8);
        assert_eq!(
            records[1]["observations"][0]["rotations"][0][0]["comparison"]["Differs"]
                ["error_power"],
            4.0
        );
        assert_eq!(records[1]["requirement"]["RoundingSteps"]["alpha"], 4.0);
        assert!(records[1]["reselection"].is_null());
        assert_eq!(records[1]["candidates"], records[3]["candidates"]);
        assert_eq!(records[1]["observations"], records[3]["observations"]);
        assert_eq!(records[1]["chosen_error_ratio"], 2.0);
        assert!(records[1].get("speedup").is_some_and(Value::is_null));
        assert_eq!(records[0]["governing_step"], "activation");
        assert_eq!(
            records[0]["comparison_version"],
            seismic::COMPARISON_VERSION
        );
        assert!(
            !records[3]["stop"].is_null(),
            "stored results retain their search outcome"
        );
    }

    #[test]
    fn a_load_selection_is_read_from_its_finished_units() {
        let trace = [
            r#"{"event":"began","schema":"tuning-trace/3","alpha":0.14,"budget_seconds":60,"fresh":true}"#,
            r#"{"event":"finished","seconds":1,"unit":{"entry":"attention","bindings":"Q=bf16","statics":{"G":8}},"chosen":{"configuration":{"statics":{},"params":{"SIMDS":8},"launches":[{"ROWS":4}]},"reason":"fastest_admissible"},"chosen_error_ratio":0.08}"#,
            r#"{"event":"pinned","seconds":1,"unit":{"entry":"rope","bindings":"","statics":{}},"choice":"not_replayed","configuration":{"statics":{},"params":{},"launches":[]}}"#,
            r#"{"event":"began","schema":"tuning-trace/3","alpha":0.07,"budget_seconds":60,"fresh":false}"#,
        ]
        .join("\n");
        // The schema written here is the producer's.
        assert!(trace.contains(&format!("\"schema\":\"{TRACE_SCHEMA}\"")));
        let first = Selection::parse(&trace, 1).unwrap();
        assert_eq!(first.alpha, Some(0.14));
        assert_eq!(first.units.len(), 1);
        assert_eq!(first.units[0].unit.name(), "attention [Q=bf16] {G=8}");
        assert_eq!(first.units[0].configuration.launches[0]["ROWS"], 4);
        assert_eq!(first.units[0].error_ratio, Some(0.08));
        assert_eq!(Selection::parse(&trace, 2).unwrap().alpha, Some(0.07));
        assert!(Selection::parse(&trace, 3).is_err_and(|error| error.contains("no load 3")));
        let old = r#"{"event":"began","budget_seconds":60,"fresh":true}"#;
        assert!(Selection::parse(old, 1).is_err_and(|error| error.contains("schema")));
        let previous = trace.replace(TRACE_SCHEMA, "tuning-trace/2");
        assert!(Selection::parse(&previous, 1).is_err_and(|error| error.contains("schema")));
        let missing_choice = trace.replace("\"chosen\":", "\"wrong_field\":");
        assert!(Selection::parse(&missing_choice, 1).is_err_and(|error| error.contains("chosen")));
        let negative_ratio = trace.replace(
            "\"chosen_error_ratio\":0.08",
            "\"chosen_error_ratio\":-0.08",
        );
        assert!(
            Selection::parse(&negative_ratio, 1).is_err_and(|error| error.contains("nonnegative"))
        );
    }

    #[test]
    fn references_require_matching_identity_and_every_fixture_to_pass_sanity() {
        let scratch = Scratch::new();
        let (stem, metadata, record) = scratch.reference();
        let open = || ReferenceOutput::open(&stem, "fixture:gguf:q4", "rendered-fixture-digest");
        assert!(open().is_ok());
        for (pointer, value) in [
            ("/model", json!("other:gguf:q4")),
            ("/rendered_digest", json!("other-render")),
            ("/family", json!("other-family")),
            ("/implementation_digest", json!("")),
            ("/sanity/passed", json!(false)),
            ("/sanity/perplexity", json!(301.0)),
            ("/sanity/fixtures/0/passed", json!(false)),
            ("/sanity/fixtures/0/top1_accuracy", json!(0.01)),
            ("/sanity/fixtures/0/fixture_id", json!("other-fixture")),
            ("/sanity/fixtures", json!([])),
        ] {
            let mut bad = record.clone();
            *bad.pointer_mut(pointer).unwrap() = value;
            scratch.json("reference.json", &bad);
            assert!(open().is_err(), "accepted invalid reference at {pointer}");
        }
        scratch.json("reference.json", &record);
        for (field, value) in [
            ("model", json!("other:gguf:q4")),
            ("rendered_digest", json!("other")),
            ("producer", json!("harness")),
        ] {
            let mut bad = metadata.clone();
            bad[field] = value;
            scratch.json("logits.json", &bad);
            assert!(open().is_err(), "accepted invalid logits {field}");
        }
        scratch.json("logits.json", &metadata);
        std::fs::remove_file(scratch.0.join("reference.json")).unwrap();
        assert!(open().is_err_and(|error| error.contains("reference.json")));
    }

    #[test]
    fn comparison_refuses_different_vocabularies_rows_and_nonfinite_or_truncated_data() {
        let scratch = Scratch::new();
        let (stem, metadata, _) = scratch.reference();
        let base = LogitsOutput::open(&stem).unwrap();
        let mut other = LogitsOutput::open(&stem).unwrap();
        assert!(base.require_matching(&other).is_ok());
        other.vocabulary = 2;
        assert!(base.require_matching(&other).is_err());
        other.vocabulary = 3;
        other.rows[0].position = 2;
        assert!(base.require_matching(&other).is_err());
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            std::fs::write(
                stem.with_extension("f32"),
                [1.0f32, value, 3.0].map(f32::to_le_bytes).concat(),
            )
            .unwrap();
            assert!(
                LogitsOutput::open(&stem)
                    .unwrap()
                    .row(0)
                    .is_err_and(|error| error.contains("nonfinite"))
            );
        }
        std::fs::write(stem.with_extension("f32"), [0u8; 8]).unwrap();
        assert!(LogitsOutput::open(&stem).is_err_and(|error| error.contains("bytes")));
        let mut empty = metadata;
        empty["rows"] = json!([]);
        scratch.json("logits.json", &empty);
        assert!(LogitsOutput::open(&stem).is_err_and(|error| error.contains("scored rows")));
    }

    #[test]
    fn kl_of_identical_distributions_is_zero_and_positive_otherwise() {
        let p = log_softmax(&[1.0, 2.0, 3.0]);
        let q = log_softmax(&[1.0, 2.5, 2.0]);
        assert!(kl(&p, &p).abs() < 1e-12);
        assert!(kl(&p, &q) > 0.0);
    }

    #[test]
    fn statistics_follow_the_metric_rule() {
        let mut values = (0..20).map(|value| value as f64 / 10.0).collect::<Vec<_>>();
        let stats = statistics(&mut values);
        assert_eq!(stats["positions"], 20);
        assert_eq!(stats["above_0_5_nats"], 14);
        assert!((stats["mean_without_10_worst"].as_f64().unwrap() - 0.45).abs() < 1e-12);
        assert_eq!(stats["max"], 1.9);
        assert_eq!(stats["median"], 0.95);
        assert!(statistics(&mut [0.1, 0.2])["mean_without_10_worst"].is_null());
    }
}
