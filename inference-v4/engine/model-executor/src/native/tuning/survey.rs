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
//! judged. Points are measured once per key, as the search measures them.
//! [`replay_report`] runs the production search (`seismic::replay`) against
//! these files (tuning spec §E2) and reports, per entry instance, the choice
//! quality at the budget the survey's load allocated, the configurations
//! needed, and the per-point gap of one configuration for all points
//! (example `tuning_replay`).
//!
//! On a shared host the whole surveying process must run under the host's
//! GPU lock: its model load, the tuning of entries it does not survey and
//! its bench cells use the device as much as the survey does.
//!
//! Only `forward_bench` enables the feature (`--tuning-survey DIR`). The
//! survey is process-global so that no production API carries it.

use super::TuningKey;
use seismic::{SurveyPlan, TuningResult};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;

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

/// Write one entry instance's survey, with the search budget this load
/// allocated to it.
pub(super) fn record(key: &TuningKey, budget: usize, result: &TuningResult) -> Result<(), String> {
    let guard = SURVEY.lock().expect("tuning survey lock poisoned");
    let survey = guard
        .as_ref()
        .expect("a survey result implies an installed survey");
    let (entry, bindings, statics) = key;
    let instance = crate::kernel_cache::TuningCacheKey::of(&format!("{bindings}{statics:?}"));
    let path = survey
        .directory
        .join(format!("{entry}-{}.json", &instance.as_str()[..12]));
    let record = serde_json::json!({
        "entry": entry,
        "bindings": bindings,
        "statics": statics,
        "budget": budget,
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

/// Replays per entry instance.
pub const REPLAY_RUNS: usize = 1000;

/// Replay the production search against every survey file in `directory`
/// ([`REPLAY_RUNS`] runs each, §E2) and report, as Markdown: per entry
/// instance, the chosen configuration's true cost relative to the true best
/// at the budget the survey's load allocated and with the whole space as
/// budget, for the production objective and the previous one; the
/// configurations needed for 95% of runs to reach 2% of the best (`n95`);
/// and per point, the time of the overall best configuration against the
/// best at that point alone (what a per-size launch could recover).
pub fn replay_report(directory: &std::path::Path) -> Result<String, String> {
    use seismic::replay::{replay, Objective, Recording};
    use std::fmt::Write;
    let mut files = std::fs::read_dir(directory)
        .map_err(|error| format!("reading {}: {error}", directory.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("reading {}: {error}", directory.display()))?;
    files.retain(|path| {
        path.extension()
            .is_some_and(|extension| extension == "json")
    });
    files.sort();
    let settings = super::SEARCH_SETTINGS;
    let mut report = String::new();
    let percent = |value: f64| format!("{:.1}%", 100.0 * value);
    for path in files {
        let text =
            std::fs::read(&path).map_err(|error| format!("reading {}: {error}", path.display()))?;
        let record: serde_json::Value = serde_json::from_slice(&text)
            .map_err(|error| format!("parsing {}: {error}", path.display()))?;
        let field = |name: &str| {
            record
                .get(name)
                .cloned()
                .ok_or_else(|| format!("{} has no `{name}`", path.display()))
        };
        let result: TuningResult = serde_json::from_value(field("result")?)
            .map_err(|error| format!("parsing {}: {error}", path.display()))?;
        let budget = field("budget")?
            .as_u64()
            .ok_or_else(|| format!("{}: `budget` is not a count", path.display()))?
            as usize;
        let recording =
            Recording::new(&result).map_err(|error| format!("{}: {error}", path.display()))?;
        let space = recording.space().len();
        writeln!(
            report,
            "### {} [{}] {}\n\n{} admissible, {} measured, budget {budget}; true best {:?}\n",
            result.entry,
            field("bindings")?.as_str().unwrap_or_default(),
            field("statics")?,
            space,
            recording.measured(),
            recording.space().values(recording.best()),
        )
        .expect("writing to a string");
        writeln!(
            report,
            "| objective | budget | within 1% | within 2% | within 5% | median excess | p95 excess | evaluated | n95 (2%) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        .expect("writing to a string");
        for (name, objective) in [
            ("keyed (production)", Objective::Keyed),
            ("separate (previous)", Objective::Separate),
        ] {
            for budget in [budget, space] {
                let replayed = replay(&recording, budget, &settings, objective, REPLAY_RUNS);
                let evaluated =
                    replayed.evaluated.iter().sum::<usize>() as f64 / REPLAY_RUNS as f64;
                writeln!(
                    report,
                    "| {name} | {budget} | {} | {} | {} | {} | {} | {evaluated:.1} | {} |",
                    percent(replayed.within(0.01)),
                    percent(replayed.within(0.02)),
                    percent(replayed.within(0.05)),
                    percent(replayed.quantile(0.5)),
                    percent(replayed.quantile(0.95)),
                    replayed
                        .needed(0.02, 0.95)
                        .map_or("never".to_owned(), |needed| needed.to_string()),
                )
                .expect("writing to a string");
            }
        }
        writeln!(
            report,
            "\n| point | weight | best overall µs | best here µs | gap | best here |\n|---|---:|---:|---:|---:|---|"
        )
        .expect("writing to a string");
        let gaps = recording.gaps();
        for gap in &gaps {
            writeln!(
                report,
                "| {} | {:.3} | {:.1} | {:.1} | {} | {:?} |",
                gap.label,
                gap.weight,
                gap.shared_seconds * 1e6,
                gap.local_seconds * 1e6,
                percent(gap.gap()),
                gap.local,
            )
            .expect("writing to a string");
        }
        let weighted = gaps.iter().map(|gap| gap.weight * gap.gap()).sum::<f64>()
            / gaps.iter().map(|gap| gap.weight).sum::<f64>();
        writeln!(report, "\nWeighted per-point gap: {}\n", percent(weighted))
            .expect("writing to a string");
    }
    Ok(report)
}
