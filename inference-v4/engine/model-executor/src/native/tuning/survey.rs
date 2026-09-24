//! Tuning survey: a development-only measurement tool (cargo feature
//! `tuning-survey`), never part of a served engine.
//!
//! A process that installs a [`TuningSurvey`] replaces the search of every
//! tuned entry instance (or of the named entries) by a survey: every
//! admissible configuration is formed, measured with
//! [`TuningSurvey::samples`] samples per point, and validated, and the whole
//! Seismic `TuningResult` (every raw sample of every configuration) is
//! written to one JSON file per entry instance: the true cost of every
//! configuration in the declared space, against which a search's choice is
//! judged. The replay harness of the tuning spec (§E2), which runs the
//! production search (`seismic::search`) against these files, is not built
//! yet.
//!
//! On a shared host, [`TuningSurvey::lock`] names a lock directory taken
//! (`mkdir`, waiting while it exists) around each entry's survey and
//! removed after it, so each hold lasts one entry.
//!
//! Only `forward_bench` enables the feature (`--tuning-survey DIR`). The
//! survey is process-global so that no production API carries it.

use super::TuningKey;
use seismic::{SurveyPlan, TuningResult};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

/// What to survey and where the records go.
#[derive(Clone, Debug)]
pub struct TuningSurvey {
    pub directory: PathBuf,
    /// Samples per point of every configuration.
    pub samples: usize,
    /// Entries to survey; every tuned entry when empty. The others are
    /// searched as usual.
    pub entries: Vec<String>,
    /// Widened parameter domains by entry (each list keeps the declared
    /// default first), to test the search on larger spaces.
    pub domains: BTreeMap<String, BTreeMap<String, Vec<u64>>>,
    /// A lock directory held around each entry's survey.
    pub lock: Option<PathBuf>,
}

static SURVEY: Mutex<Option<TuningSurvey>> = Mutex::new(None);

/// Install the survey for every later program preparation in this process.
pub fn install(survey: TuningSurvey) {
    *SURVEY.lock().expect("tuning survey lock poisoned") = Some(survey);
}

/// The installed survey, when it names `entry`.
fn surveying(entry: &str) -> Option<TuningSurvey> {
    let guard = SURVEY.lock().expect("tuning survey lock poisoned");
    let survey = guard.as_ref()?;
    (survey.entries.is_empty() || survey.entries.iter().any(|name| name == entry))
        .then(|| survey.clone())
}

/// The survey plan for `entry`, when a survey is installed and names it.
pub(super) fn plan(entry: &str) -> Option<SurveyPlan> {
    surveying(entry).map(|survey| SurveyPlan {
        samples: survey.samples,
        min_sample_seconds: super::MIN_SAMPLE_SECONDS,
        domains: survey.domains.get(entry).cloned().unwrap_or_default(),
    })
}

/// Holds the survey's lock directory while it lives.
pub(super) struct Held(Option<PathBuf>);

/// Take the survey's lock for `entry`'s survey (waiting while another holder
/// has it); nothing when `entry` is not surveyed or no lock is named.
pub(super) fn hold(entry: &str) -> Result<Held, String> {
    let Some(path) = surveying(entry).and_then(|survey| survey.lock) else {
        return Ok(Held(None));
    };
    loop {
        match std::fs::create_dir(&path) {
            Ok(()) => return Ok(Held(Some(path))),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                std::thread::sleep(Duration::from_secs(2));
            }
            Err(error) => return Err(format!("taking survey lock {}: {error}", path.display())),
        }
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        if let Some(path) = &self.0 {
            if let Err(error) = std::fs::remove_dir(path) {
                eprintln!("tuning survey: releasing lock {}: {error}", path.display());
            }
        }
    }
}

/// Write one entry instance's survey.
pub(super) fn record(key: &TuningKey, result: &TuningResult) -> Result<(), String> {
    let guard = SURVEY.lock().expect("tuning survey lock poisoned");
    let survey = guard.as_ref().expect("a survey result implies an installed survey");
    let (entry, bindings, statics) = key;
    let instance = crate::kernel_cache::TuningCacheKey::of(&format!("{bindings}{statics:?}"));
    let path = survey
        .directory
        .join(format!("{entry}-{}.json", &instance.as_str()[..12]));
    let record = serde_json::json!({
        "entry": entry,
        "bindings": bindings,
        "statics": statics,
        "result": result,
    });
    std::fs::create_dir_all(&survey.directory)
        .and_then(|()| {
            std::fs::write(
                &path,
                serde_json::to_vec(&record).expect("a survey record serializes"),
            )
        })
        .map_err(|error| format!("writing survey {}: {error}", path.display()))
}
