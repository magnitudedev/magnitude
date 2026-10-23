//! A load's chosen configurations read from its tuning trace (contract
//! "Tuning trace"): what `--tuning-select` pins, and what each choice's
//! trace says of its precision.

use magnitude_executor::pinned_tuning::PinnedConfiguration;
use seismic::Configuration;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::Path;

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
pub struct Selection {
    alpha: Option<f64>,
    admission: Option<String>,
    units: Vec<Selected>,
}

impl Selection {
    /// The units of load `number` (1-based) of the trace at `path` matching
    /// `unit` (any part of `entry [bindings] {statics}`), or all of them.
    pub fn select(path: &Path, number: usize, unit: Option<&str>) -> Result<Self, String> {
        let mut selection = Self::read(path, number)?;
        selection
            .units
            .retain(|selected| unit.is_none_or(|pattern| selected.unit.name().contains(pattern)));
        if selection.units.is_empty() {
            return Err("the selection pins no unit".into());
        }
        Ok(selection)
    }

    /// Each unit's chosen configuration, as a pin replays it.
    pub fn pins(&self) -> Vec<PinnedConfiguration> {
        self.units
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
            .collect()
    }

    /// What the trace states of the selection's precision: each unit's
    /// largest error in the rounding steps governing its admission, and its
    /// square, a dimensionless contribution relative to resolution noise
    /// (not a predicted KL), bounded per unit by α².
    pub fn summary(&self) -> Value {
        let units = self
            .units
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
        json!({
            "alpha": self.alpha,
            "admission": self.admission,
            "contribution_units": "squared_error_ratio",
            "bound_per_unit": self.alpha.map(|alpha| alpha * alpha),
            "units": units,
            "predicted_contribution": self
                .units
                .iter()
                .map(|selected| selected.error_ratio.map(|ratio| ratio * ratio))
                .sum::<Option<f64>>(),
        })
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "magnitude-selection-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
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
}
