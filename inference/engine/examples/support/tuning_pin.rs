//! `--tuning-record FILE` / `--tuning-replay FILE`, only with the
//! development feature `pinned-tuning`
//! (`cargo run --release --example forward_bench --features pinned-tuning`).
//!
//! A measurement tool for bit-exact comparisons between runs: tuning at load
//! may choose different configurations each run, so the baseline run records
//! the configuration chosen per entry and the comparison run replays exactly
//! those (it tunes nothing). Within the recording run every load reuses the
//! first choice per entry. See `magnitude_executor::pinned_tuning`.

use magnitude_executor::pinned_tuning::{self, PinnedConfiguration, TuningPin};
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
        if let Some(index) = args.iter().position(|arg| arg == "--tuning-defaults") {
            args.remove(index);
            pinned_tuning::install(TuningPin::Defaults);
            eprintln!("forward_bench: running every entry's default configuration");
            return Ok(Self { record: None });
        }
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
