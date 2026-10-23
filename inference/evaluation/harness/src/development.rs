//! The harness's development tuning tools (the engine's `pinned-tuning` and
//! `tuning-survey` features, which only the harness enables; a served engine
//! never carries them).
//!
//! - Pins: `--tuning-record FILE` records the configuration tuned per entry;
//!   `--tuning-replay FILE` runs exactly those and tunes nothing;
//!   `--tuning-defaults` runs every entry's default configuration;
//!   `--tuning-select TRACE` runs the configurations one load of a tuning
//!   trace chose (`--tuning-select-load`, `--tuning-select-unit`) and writes
//!   what the trace states of their precision to `selection.json`. Two
//!   builds, or two selections, can then be compared bit for bit
//!   (`magnitude_executor::pinned_tuning`). A replay never tunes: a unit the
//!   recording has no configuration for (a shape first prepared in the
//!   replaying run) runs its declared default and is listed under
//!   `pinned.not_replayed` in the run's tuning outcome (`pinned` events with
//!   choice `not_replayed` in `tuning.jsonl`).
//! - Survey: `--tuning-survey DIR` surveys every admissible configuration of
//!   every tuned entry (or `--tuning-survey-entries`) with
//!   `--tuning-survey-samples` samples per point, one JSON record per entry
//!   instance (`magnitude_executor::tuning_survey`), and writes the
//!   production search's replay against them to `DIR/replay.md`.
//!   `--tuning-survey-widen ENTRY:P=v,v;Q=v` replaces an entry's parameter
//!   domains, keeping each default first.

pub use pin::{Pin, PinArgs};
pub use survey::SurveyArgs;

mod pin {
    use crate::selection::Selection;
    use magnitude_executor::pinned_tuning::{self, PinnedConfiguration, TuningPin};
    use serde_json::{Value, json};
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    #[derive(clap::Args, Clone, Debug)]
    pub struct PinArgs {
        /// Record the configuration tuned per entry into FILE.
        #[arg(long, value_name = "FILE",
              conflicts_with_all = ["tuning_replay", "tuning_defaults", "tuning_select"])]
        pub tuning_record: Option<PathBuf>,
        /// Run exactly the configurations recorded in FILE; tune nothing.
        #[arg(long, value_name = "FILE", conflicts_with_all = ["tuning_defaults", "tuning_select"])]
        pub tuning_replay: Option<PathBuf>,
        /// Run every entry's default configuration.
        #[arg(long, conflicts_with = "tuning_select")]
        pub tuning_defaults: bool,
        /// Run the configurations a load of this tuning trace (or run
        /// directory) chose; every other unit runs its default.
        #[arg(long, value_name = "TRACE")]
        pub tuning_select: Option<PathBuf>,
        /// Which load of the selected trace (1-based).
        #[arg(long, default_value_t = 1, requires = "tuning_select")]
        pub tuning_select_load: usize,
        /// Select only the units matching this (any part of `entry
        /// [bindings] {statics}`).
        #[arg(long, requires = "tuning_select")]
        pub tuning_select_unit: Option<String>,
    }

    /// An installed pin; a recording one writes its file when finished.
    pub struct Pin {
        record: Option<PathBuf>,
    }

    impl Pin {
        /// Install the pin `args` name; a selection's precision goes to
        /// `selection_record`.
        pub fn install(args: &PinArgs, selection_record: &Path) -> Result<Self, String> {
            if let Some(trace) = &args.tuning_select {
                let selection = Selection::select(
                    trace,
                    args.tuning_select_load,
                    args.tuning_select_unit.as_deref(),
                )?;
                let pins = selection.pins();
                eprintln!(
                    "magnitude-harness: running {} selected configurations from {}",
                    pins.len(),
                    trace.display()
                );
                pinned_tuning::install(TuningPin::Replay(pins));
                let text = serde_json::to_vec_pretty(&selection.summary())
                    .map_err(|error| error.to_string())?;
                std::fs::write(selection_record, text)
                    .map_err(|error| format!("writing {}: {error}", selection_record.display()))?;
            } else if args.tuning_defaults {
                pinned_tuning::install(TuningPin::Defaults);
                eprintln!("magnitude-harness: running every entry's default configuration");
            } else if let Some(path) = &args.tuning_replay {
                let configurations = read(path)?;
                eprintln!(
                    "magnitude-harness: replaying {} pinned tuning configurations from {}",
                    configurations.len(),
                    path.display()
                );
                pinned_tuning::install(TuningPin::Replay(configurations));
            } else if args.tuning_record.is_some() {
                pinned_tuning::install(TuningPin::Record);
            }
            Ok(Self {
                record: args.tuning_record.clone(),
            })
        }

        /// Write the recorded configurations when recording.
        pub fn finish(self) -> Result<(), String> {
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
                        "launch_params": pinned
                            .launch_params
                            .iter()
                            .map(|((launch, name), value)| (format!("{launch}:{name}"), *value))
                            .collect::<BTreeMap<_, _>>(),
                    })
                })
                .collect::<Vec<_>>();
            let text = serde_json::to_string_pretty(&entries).map_err(|error| error.to_string())?;
            std::fs::write(&path, text)
                .map_err(|error| format!("writing {}: {error}", path.display()))?;
            eprintln!(
                "magnitude-harness: recorded {} tuning configurations to {}",
                configurations.len(),
                path.display()
            );
            Ok(())
        }
    }

    fn read(path: &Path) -> Result<Vec<PinnedConfiguration>, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("reading {}: {error}", path.display()))?;
        let entries: Vec<Value> =
            serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
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
                    entry[field]
                        .as_str()
                        .map(str::to_owned)
                        .ok_or_else(malformed)
                };
                Ok(PinnedConfiguration {
                    entry: text("entry")?,
                    bindings: text("bindings")?,
                    statics: map(&entry["statics"])?,
                    params: map(&entry["params"])?,
                    launch_params: map(&entry["launch_params"])?
                        .into_iter()
                        .map(|(key, value)| {
                            let (launch, name) = key.split_once(':').ok_or_else(malformed)?;
                            Ok((
                                (launch.parse().map_err(|_| malformed())?, name.to_owned()),
                                value,
                            ))
                        })
                        .collect::<Result<_, String>>()?,
                })
            })
            .collect()
    }
}

mod survey {
    use magnitude_executor::tuning_survey::{self, TuningSurvey};
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    #[derive(clap::Args, Clone, Debug)]
    pub struct SurveyArgs {
        /// Survey tuning into DIR instead of searching.
        #[arg(long, value_name = "DIR")]
        pub tuning_survey: Option<PathBuf>,
        /// Samples per point of every surveyed configuration.
        #[arg(long, default_value_t = 15, requires = "tuning_survey")]
        pub tuning_survey_samples: usize,
        /// Entries to survey; every tuned entry when absent.
        #[arg(long, value_delimiter = ',', requires = "tuning_survey")]
        pub tuning_survey_entries: Vec<String>,
        /// `ENTRY:P=v,v,..;Q=v,..`: replace an entry's parameter domains.
        #[arg(long, requires = "tuning_survey")]
        pub tuning_survey_widen: Option<String>,
    }

    impl SurveyArgs {
        /// These arguments again, for the worker process this executable
        /// starts: a survey measures, so it runs where the search runs, in
        /// a model's Prepare, never in a load.
        pub fn worker_arguments(&self) -> Vec<std::ffi::OsString> {
            let Some(directory) = &self.tuning_survey else {
                return Vec::new();
            };
            let mut arguments = vec![
                "--tuning-survey".into(),
                directory.clone().into_os_string(),
                "--tuning-survey-samples".into(),
                self.tuning_survey_samples.to_string().into(),
            ];
            if !self.tuning_survey_entries.is_empty() {
                arguments.push("--tuning-survey-entries".into());
                arguments.push(self.tuning_survey_entries.join(",").into());
            }
            if let Some(widen) = &self.tuning_survey_widen {
                arguments.push("--tuning-survey-widen".into());
                arguments.push(widen.into());
            }
            arguments
        }

        /// After the run, the production search replayed against the
        /// survey's records (`DIR/replay.md`), when it surveyed a unit (an
        /// entry the load does not tune leaves no record and no directory).
        pub fn report(&self) -> Result<(), String> {
            let Some(directory) = self.tuning_survey.as_ref().filter(|path| path.is_dir()) else {
                return Ok(());
            };
            let report = tuning_survey::replay_report(directory)?;
            let path = directory.join("replay.md");
            std::fs::write(&path, report)
                .map_err(|error| format!("writing {}: {error}", path.display()))
        }

        pub fn install(&self) -> Result<(), String> {
            let Some(directory) = &self.tuning_survey else {
                return Ok(());
            };
            let mut domains: BTreeMap<String, BTreeMap<String, Vec<u64>>> = BTreeMap::new();
            if let Some(widen) = &self.tuning_survey_widen {
                let malformed = || "--tuning-survey-widen takes ENTRY:P=v,v,..;Q=v,..".to_owned();
                let (entry, parameters) = widen.split_once(':').ok_or_else(malformed)?;
                for parameter in parameters.split(';') {
                    let (name, values) = parameter.split_once('=').ok_or_else(malformed)?;
                    let values = values
                        .split(',')
                        .map(|value| {
                            value
                                .parse::<u64>()
                                .map_err(|_| format!("{value:?} is not a count"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    domains
                        .entry(entry.to_owned())
                        .or_default()
                        .insert(name.to_owned(), values);
                }
            }
            eprintln!(
                "magnitude-harness: surveying tuning into {} ({} samples per point)",
                directory.display(),
                self.tuning_survey_samples
            );
            tuning_survey::install(TuningSurvey {
                directory: directory.clone(),
                samples: self.tuning_survey_samples,
                entries: self.tuning_survey_entries.clone(),
                domains,
            });
            Ok(())
        }
    }
}
