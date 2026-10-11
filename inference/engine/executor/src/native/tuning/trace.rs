//! The tuning trace: every event of a load's tuning as one JSON object per
//! line, for hosts that record how tuning spent its budget.
//!
//! A search (the Prepare worker's) and a load each write `began` with their
//! `work` (`search` or `load`) and [`TuningSettings`], the tuner's own events
//! as they happen and `concluded` with the outcome. A load's events are
//! `finished` per unit it read a stored record for and `unprepared` per unit
//! it prepared at its canonical default for want of one. A search's are:
//! `progress`;
//! `mandatory` per unit; `started`; one `stage` per piece of optional work
//! with its class, its forecast and what it took, written as the piece
//! ends, so the record of the work is in the trace whether the load
//! succeeds or fails; one `published` per record published when the
//! optional work stops, with its forecast and what it took; one `finished`
//! per unit, its complete record; `pinned` for a unit a development pin
//! fixed, with the configuration it runs; and `budget`, what became of the
//! load's budget: used, unused and overrun. Every event after `began`
//! carries `seconds`, the time since it. Several loads may append to one
//! file; each starts with its own `began`.
//!
//! A `finished` record states the requirement its choice was made under
//! and lists `observations`: every measurement of a candidate against the
//! unit's default, per point, rotation and subject, each made once and
//! stated once with the key it was made under. Each of its `candidates`
//! states its artifact identity once and names its own observations by
//! their positions in that list, whether it was admitted or rejected, with
//! its largest error ratio and the first constraint it violates. `began`
//! states α, which storage step governs admission and the comparison's
//! version.

use super::{PinnedChoice, TunedEntry, TuningEvent, TuningOrigin, TuningSettings};
use seismic::SearchStop;
use serde::ser::{Serialize, SerializeMap, Serializer};
use serde_json::{Value, json};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The trace's schema, stated by every `began`. 3: a `stage` carries
/// `formed_ahead`, its `spent` by operation class and `publishing_seconds`
/// only for a publication the piece made; `published` records each record
/// published when the optional work stops; `budget` carries
/// `settling_seconds` and the load's `memory`; a refused offer its
/// `reason`; `pinned` the `configuration` its unit runs; a unit's
/// `standing` its planning `requests`. A `stage` also states its unit's
/// `statics` and lists its `raises`: what each raise of the unit's clock
/// did (`passed` or `expired`, its `readings`, its `first` reading, the
/// `level` it read, the `bound` it was held to, its host `seconds`, and as
/// `sample` the first sample of the session behind it, which names that
/// session's samples in the unit's `finished` record) and the clock's
/// establishment (`established`, with no bound), which no reader of
/// version 3 is asked to know.
pub const TRACE_SCHEMA: &str = "tuning-trace/3";

pub struct TuningTrace {
    path: PathBuf,
    /// Who writes the trace's lines; always present between calls.
    writer: RefCell<Option<Writer>>,
    began: Instant,
    /// Units the traced load prepared at their canonical defaults for want
    /// of a stored record.
    unprepared: std::cell::Cell<usize>,
    /// The file's length when this trace began.
    start: u64,
}

/// What a trace's `began` opens: a search, which forms, measures and
/// publishes records, or a load, which reads them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TracedWork {
    Search,
    Load,
}

/// One line of the trace, as its event hands it over.
enum Line {
    Rendered(Vec<u8>),
    /// A unit's record, `seconds` after the trace began: stated from the
    /// unit's evidence by whoever writes the line.
    Finished { tuned: TunedEntry, seconds: f64 },
}

/// The trace's file, written a line at a time in the order given.
struct Sink {
    file: File,
    /// The first line that could not be written. The observer cannot fail
    /// a load, so the failure is returned when the load concludes.
    failure: Option<String>,
}

impl Sink {
    fn take(&mut self, line: Line) {
        let bytes = match line {
            Line::Rendered(bytes) => bytes,
            Line::Finished { tuned, seconds } => match tuned.receipt.read() {
                Ok(receipt) => render(&Finished {
                    tuned: &tuned,
                    receipt: &receipt,
                    seconds,
                }),
                Err(error) => {
                    self.failure.get_or_insert(format!(
                        "the record of {} [{}] cannot be stated: {error}",
                        tuned.entry, tuned.bindings
                    ));
                    return;
                }
            },
        };
        if let Err(error) = self.file.write_all(&bytes) {
            self.failure.get_or_insert(error.to_string());
        }
    }
}

/// Who writes a trace's lines.
enum Writer {
    /// A search writes each event inside its observer call: a buffered tail
    /// must not defer candidate-receipt I/O until the search has left the
    /// tuning clock.
    Calling(Sink),
    /// A load's lines are written by the trace's own thread, which is
    /// joined when the load concludes. A load serves each stored unit's
    /// choice and reads nothing of its evidence; the trace states the
    /// evidence, megabytes of it for one unit, so it is this thread that
    /// reads it from the store and renders it, beside the load and not in
    /// its way. (Read and rendered on the load's own thread, the 22 records
    /// of a Qwen3.5-4B Q8 cached load took 0.12 s to read and 0.24 s to
    /// state: M4 Pro, 2026-10-10.)
    Own(std::sync::mpsc::Sender<Line>, std::thread::JoinHandle<Sink>),
}

/// `record` as one line of the trace.
fn render(record: &impl Serialize) -> Vec<u8> {
    let mut line = serde_json::to_vec(record).expect("a trace record serializes");
    line.push(b'\n');
    line
}

impl TuningTrace {
    /// Open `path` for appending and record that `work` begins.
    pub fn begin(
        path: &Path,
        work: TracedWork,
        settings: TuningSettings,
        profile: &super::ResolvedNumericalProfile,
    ) -> Result<Self, String> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|error| format!("tuning trace {}: {error}", path.display()))?;
        let start = file
            .metadata()
            .map_err(|error| format!("tuning trace {}: {error}", path.display()))?
            .len();
        let sink = Sink {
            file,
            failure: None,
        };
        let writer = match work {
            TracedWork::Search => Writer::Calling(sink),
            TracedWork::Load => {
                let (lines, taken) = std::sync::mpsc::channel();
                let thread = std::thread::Builder::new()
                    .name("tuning-trace".into())
                    .spawn(move || {
                        let mut sink = sink;
                        for line in taken {
                            sink.take(line);
                        }
                        sink
                    })
                    .map_err(|error| format!("tuning trace {}: {error}", path.display()))?;
                Writer::Own(lines, thread)
            }
        };
        let trace = Self {
            path: path.to_path_buf(),
            writer: RefCell::new(Some(writer)),
            began: Instant::now(),
            unprepared: std::cell::Cell::new(0),
            start,
        };
        trace.write(json!({
            "event": "began",
            "schema": TRACE_SCHEMA,
            "work": match work {
                TracedWork::Search => "search",
                TracedWork::Load => "load",
            },
            "budget_seconds": settings.budget.as_secs_f64(),
            "fresh": settings.fresh,
            "alpha": profile.alpha(),
            "admission": "rounding_steps",
            "governing_step": seismic::GOVERNING_STEP,
            "comparison_version": seismic::COMPARISON_VERSION,
        }));
        Ok(trace)
    }

    /// Record one event of the tuner.
    pub fn event(&self, event: &TuningEvent) {
        let record = match event {
            TuningEvent::Progress { completed, total } => json!({
                "event": "progress",
                "completed_ms": completed,
                "total_ms": total,
            }),
            TuningEvent::Started {
                entry,
                bindings,
                points,
            } => json!({
                "event": "started",
                "entry": entry,
                "bindings": bindings,
                "points": points,
            }),
            TuningEvent::Mandatory {
                entry,
                bindings,
                began_seconds,
                forming_seconds,
                publishing_seconds,
            } => json!({
                "event": "mandatory",
                "entry": entry,
                "bindings": bindings,
                "began_seconds": began_seconds,
                "forming_seconds": forming_seconds,
                "publishing_seconds": publishing_seconds,
            }),
            TuningEvent::Stage(stage) => json!({
                "event": "stage",
                "entry": stage.entry,
                "bindings": stage.bindings,
                "statics": stage.statics,
                "share": stage.share,
                "class": stage_name(stage.stage),
                "priced_seconds": stage.priced_seconds,
                "unpriced_operations": stage.unpriced,
                "forecast_seconds": stage.forecast_seconds,
                "began_seconds": stage.began_seconds,
                "actual_seconds": stage.actual_seconds,
                "formed_ahead": stage.formed,
                "spent": stage.spent,
                "raises": stage.raises,
                "publishing_seconds": stage.publishing_seconds,
                "promoted": stage.promoted,
            }),
            TuningEvent::Published {
                entry,
                bindings,
                forecast_seconds,
                began_seconds,
                actual_seconds,
            } => json!({
                "event": "published",
                "entry": entry,
                "bindings": bindings,
                "forecast_seconds": forecast_seconds,
                "began_seconds": began_seconds,
                "actual_seconds": actual_seconds,
            }),
            TuningEvent::Budget(budget) => json!({
                "event": "budget",
                "budget_seconds": budget.budget_seconds,
                "used_seconds": budget.used_seconds,
                "settling_seconds": budget.settling_seconds,
                "unused_seconds": budget.unused_seconds,
                "overrun_seconds": budget.overrun_seconds,
                "stages": budget.stages,
                "refused": budget.refused.iter().map(|refused| json!({
                    "unit": refused.unit,
                    "class": stage_name(refused.stage),
                    "priced_seconds": refused.priced_seconds,
                    "unpriced_operations": refused.unpriced_operations,
                    "footprint_bytes": refused.footprint_bytes,
                    "unpriced_footprints": refused.unpriced_footprints,
                    "reason": refused.reason,
                })).collect::<Vec<_>>(),
                "ending": {
                    "began_seconds": budget.ending.began_seconds,
                    "held_seconds": budget.ending.held_seconds,
                    "passes": budget.ending.passes,
                    "deciding_seconds": budget.ending.deciding_seconds,
                    "narrowed": budget.ending.narrowed,
                    "narrowing_seconds": budget.ending.narrowing_seconds,
                    "released": budget.ending.released,
                    "releasing_seconds": budget.ending.releasing_seconds,
                },
                "memory": {
                    "ceiling_bytes": budget.memory.ceiling_bytes,
                    "held_peak_bytes": budget.memory.held_peak_bytes,
                    "margin_bytes": budget.memory.margin_bytes,
                    "process_began_bytes": budget.memory.process_began_bytes,
                    "process_peak_footprint_bytes": budget.memory.process_peak_footprint_bytes,
                    "process_ended_bytes": budget.memory.process_ended_bytes,
                    "device_separate": budget.memory.device_separate,
                    "footprint_ceiling_bytes": budget.memory.footprint_ceiling_bytes,
                    "footprint_peak_bytes": budget.memory.footprint_peak_bytes,
                    "footprint_margin_bytes": budget.memory.footprint_margin_bytes,
                    "host_ceiling_bytes": budget.memory.host_ceiling_bytes,
                    "host_peak_bytes": budget.memory.host_peak_bytes,
                    "host_margin_bytes": budget.memory.host_margin_bytes,
                    "host_admission": host_admission(&budget.memory),
                    "compile_host_bytes": budget.memory.compile_host_bytes,
                    "host_added_by_unit": budget.memory.host_added_by_unit.iter().map(|(unit, bytes)| json!({
                        "unit": unit,
                        "bytes": bytes,
                    })).collect::<Vec<_>>(),
                    "began_bytes": budget.memory.began_bytes,
                    "mandatory_bytes": budget.memory.mandatory_bytes,
                    "peak_bytes": budget.memory.peak_bytes,
                    "ended_bytes": budget.memory.ended_bytes,
                    "process_peak_bytes": budget.memory.process_peak_bytes,
                    "breakdown": {
                        "began": breakdown(&budget.memory.breakdown_began),
                        "peak": breakdown(&budget.memory.breakdown_peak),
                        "ended": breakdown(&budget.memory.breakdown_ended),
                        "returned": breakdown(&budget.memory.breakdown_returned),
                    },
                    "returned_bytes": budget.memory.returned_bytes,
                    "returning_seconds": budget.memory.returning_seconds,
                    "held_by_class": budget.memory.held_by_class.iter().map(|(stage, bytes)| json!({
                        "class": stage_name(*stage),
                        "bytes": bytes,
                    })).collect::<Vec<_>>(),
                    "held_by_unit": budget.memory.held_by_unit.iter().map(|(unit, bytes)| json!({
                        "unit": unit,
                        "bytes": bytes,
                    })).collect::<Vec<_>>(),
                },
            }),
            TuningEvent::Finished(tuned) => {
                return self.line(Line::Finished {
                    tuned: tuned.clone(),
                    seconds: self.began.elapsed().as_secs_f64(),
                });
            }
            TuningEvent::Unprepared {
                entry,
                bindings,
                statics,
            } => {
                self.unprepared.set(self.unprepared.get() + 1);
                json!({
                    "event": "unprepared",
                    "unit": unit(entry, bindings, statics),
                    "entry": entry,
                    "bindings": bindings,
                    "statics": statics,
                })
            }
            TuningEvent::Pinned {
                entry,
                bindings,
                statics,
                choice,
                configuration,
            } => json!({
                "event": "pinned",
                "unit": unit(entry, bindings, statics),
                "entry": entry,
                "bindings": bindings,
                "statics": statics,
                "choice": match choice {
                    PinnedChoice::Replayed => "replayed",
                    PinnedChoice::Defaults => "defaults",
                    PinnedChoice::NotReplayed => "not_replayed",
                },
                // What the unit runs, in the shape of a searched unit's
                // `chosen.configuration`.
                "configuration": configuration,
            }),
        };
        self.write(record);
    }

    /// Remove everything this trace wrote: a load that stopped unprepared
    /// is not the load, and the one that follows the search writes its
    /// trace. Nothing else writes the file meanwhile: the search starts
    /// after this load has ended.
    pub fn abandon(self) -> Result<(), String> {
        let truncated = self.written().file.set_len(self.start);
        truncated.map_err(|error| format!("tuning trace {}: {error}", self.path.display()))
    }

    /// The trace's file once every line handed over is written: a load's
    /// own thread is joined, and the calling thread writes from here on.
    fn written(&self) -> std::cell::RefMut<'_, Sink> {
        let mut writer = self.writer.borrow_mut();
        let sink = match writer.take().expect("a trace has its writer between calls") {
            Writer::Calling(sink) => sink,
            Writer::Own(lines, thread) => {
                drop(lines);
                thread
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            }
        };
        *writer = Some(Writer::Calling(sink));
        std::cell::RefMut::map(writer, |writer| match writer {
            Some(Writer::Calling(sink)) => sink,
            _ => unreachable!("the calling thread writes a trace whose lines are written"),
        })
    }

    /// Record the load's outcome and flush the trace. Fails when any record
    /// of this load could not be written.
    pub fn conclude(&self, tuned: &[TunedEntry]) -> Result<(), String> {
        let searched = tuned
            .iter()
            .filter(|tuned| tuned.origin == TuningOrigin::Searched)
            .count();
        self.write(json!({
            "event": "concluded",
            "units": tuned.len(),
            "searched": searched,
            "stored": tuned.len() - searched,
            "unprepared": self.unprepared.get(),
            "canonical_fallbacks": tuned.iter().filter(|tuned| !tuned.qualified).count(),
            "tuning_seconds": tuned.iter().map(|tuned| tuned.seconds).sum::<f64>(),
        }));
        let mut sink = self.written();
        if let Err(error) = sink.file.flush() {
            sink.failure.get_or_insert(error.to_string());
        }
        match sink.failure.take() {
            Some(failure) => Err(format!("tuning trace {}: {failure}", self.path.display())),
            None => Ok(()),
        }
    }

    fn write(&self, mut record: Value) {
        if record["event"] != "began" {
            record["seconds"] = json!(self.began.elapsed().as_secs_f64());
        }
        self.line(Line::Rendered(render(&record)));
    }

    /// Hand the trace's next line to its writer.
    fn line(&self, line: Line) {
        match self
            .writer
            .borrow_mut()
            .as_mut()
            .expect("a trace has its writer between calls")
        {
            Writer::Calling(sink) => sink.take(line),
            // A thread that is gone panicked: joining it, when the load
            // concludes, resumes the panic.
            Writer::Own(lines, _) => drop(lines.send(line)),
        }
    }
}

fn stage_name(stage: seismic::Stage) -> &'static str {
    match stage {
        seismic::Stage::Discovery => "discovery",
        seismic::Stage::Form => "form",
        seismic::Stage::Measure => "measure",
        seismic::Stage::Promote => "promote",
        seismic::Stage::Confirm => "confirm",
    }
}

/// Why a unit runs what it runs: a qualified candidate, or its canonical
/// default because its search did not reach its end within the budget,
/// because its remaining pieces did not fit the memory room, because every
/// candidate was rejected, or because nothing beat it.
pub(super) fn qualification(tuned: &TunedEntry) -> &'static str {
    if tuned.qualified {
        return "qualified";
    }
    match tuned.stop {
        Some(SearchStop::Expired) => "unqualified_budget",
        Some(SearchStop::NoRoom) => "unqualified_no_room",
        _ if tuned.rejections > 0 => "unqualified_rejected",
        _ => "unqualified",
    }
}

/// What admission by bytes did beyond its ceilings and peaks: whether the
/// host's peak is of readings or the system's own mark, what the mandatory
/// phase left held on the host, every piece and burst of compiles that
/// added more than its price, and the stop for bytes when there was one,
/// with each bounded domain as it stood and the holders of a load the host
/// ceiling has no term for.
fn host_admission(memory: &super::TuningMemory) -> Value {
    json!({
        "peak_sampled": memory.host_peak_sampled,
        "mandatory_bytes": memory.host_mandatory_bytes,
        "mispriced": memory.mispriced.iter().map(|mispricing| json!({
            "kind": mispricing.kind,
            "priced_bytes": mispricing.priced_bytes,
            "added_bytes": mispricing.added_bytes,
        })).collect::<Vec<_>>(),
        "no_room": memory.no_room.as_ref().map(|stop| json!({
            "domains": stop.domains.iter().map(|room| json!({
                "domain": room.domain,
                "ceiling_bytes": room.ceiling_bytes,
                "held_bytes": room.held_bytes,
                "smallest_footprint_bytes": room.smallest_footprint_bytes,
                "lacking_bytes": room.lacking_bytes,
            })).collect::<Vec<_>>(),
            "unplanned_holders": stop.unplanned_holders,
        })),
    })
}

/// One reading of the process's memory by what holds it, with the footprint
/// neither live heap nor graphics accounts for (on Metal that holds the
/// device's wired allocations too).
fn breakdown(memory: &seismic::ProcessMemory) -> Value {
    json!({
        "footprint_bytes": memory.footprint_bytes,
        "malloc_live_bytes": memory.malloc_live_bytes,
        "graphics_bytes": memory.graphics_bytes,
        "mapped_bytes": memory.mapped_bytes,
        "rest_bytes": memory.rest_bytes(),
    })
}

/// What every trace event identifies a unit by.
fn unit(entry: &str, bindings: &str, statics: &BTreeMap<String, u64>) -> Value {
    json!({ "entry": entry, "bindings": bindings, "statics": statics })
}

/// The largest error among `record`'s measurements against the default, in
/// the rounding steps that govern admission under the receipt's
/// requirement: zero for a record that moved no floating subject, the
/// default's among them.
fn error_ratio(receipt: &seismic::TuningResult, record: &seismic::ConfigurationRecord) -> f64 {
    record
        .observations
        .iter()
        .flat_map(|observation| receipt.observations[*observation].rotations.iter().flatten())
        .filter_map(|subject| match &subject.comparison {
            seismic::SubjectComparison::Differs(difference) => {
                seismic::governing_ratio(difference, &receipt.requirement)
            }
            _ => None,
        })
        .fold(0., f64::max)
}

/// The points `record` was measured at; none for one that was excluded.
fn measured(record: &seismic::ConfigurationRecord) -> &[seismic::PointMeasurement] {
    match &record.outcome {
        seismic::Outcome::Measured { points, .. } => points,
        seismic::Outcome::Excluded(_) => &[],
    }
}

/// A unit's `finished` record, written `seconds` after the trace began.
///
/// It is serialized from the unit's evidence (`receipt`) where that is:
/// the unit's observations and candidates are never copied into a value
/// first. A record's size follows what its search measured (one unit of a
/// Qwen3.5-4B Q8 search measured 1,932 candidates: 7 MB as written).
/// Copied into values first, the 22 records of that model's cached load
/// took 0.24 s to state, 0.16 s of it the copying (M4 Pro, 2026-10-10).
struct Finished<'a> {
    tuned: &'a TunedEntry,
    receipt: &'a seismic::TuningResult,
    seconds: f64,
}

impl Serialize for Finished<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let tuned = self.tuned;
        let receipt = self.receipt;
        let chosen_error_ratio = receipt
            .configurations
            .iter()
            .find(|record| record.configuration == receipt.overall)
            // A unit with no record of its choice serves its unmeasured default.
            .map_or(0., |record| error_ratio(receipt, record));
        let mut record = serializer.serialize_map(None)?;
        record.serialize_entry("event", "finished")?;
        record.serialize_entry(
            "unit",
            &unit(tuned.entry, &tuned.bindings, &tuned.statics),
        )?;
        record.serialize_entry(
            "coverage",
            &tuned.coverage.as_ref().map(|coverage| {
                json!({
                    "outcome": coverage.outcome,
                    "failed_count": coverage.failed,
                })
            }),
        )?;
        record.serialize_entry("entry", tuned.entry)?;
        record.serialize_entry("bindings", &tuned.bindings)?;
        record.serialize_entry(
            "origin",
            match tuned.origin {
                TuningOrigin::Searched => "searched",
                TuningOrigin::Stored => "stored",
            },
        )?;
        record.serialize_entry("qualification", qualification(tuned))?;
        record.serialize_entry("unit_seconds", &tuned.seconds)?;
        record.serialize_entry("stop", &tuned.stop)?;
        record.serialize_entry(
            "chosen",
            &json!({
                "configuration": tuned.overall,
                "reason": match tuned.origin {
                    TuningOrigin::Stored => "stored",
                    TuningOrigin::Searched if tuned.qualified => "fastest_admissible",
                    TuningOrigin::Searched => "default",
                },
            }),
        )?;
        record.serialize_entry("points", &receipt.points)?;
        record.serialize_entry("requirement", &receipt.requirement)?;
        record.serialize_entry("reselection", &receipt.reselection)?;
        record.serialize_entry("measurement_identity", &receipt.measurement_identity)?;
        record.serialize_entry("observations", &receipt.observations)?;
        record.serialize_entry("served", &receipt.served)?;
        record.serialize_entry("candidates", &Candidates(receipt))?;
        record.serialize_entry("disposition", &receipt.disposition)?;
        record.serialize_entry("method", &receipt.method)?;
        record.serialize_entry("parameters", &receipt.parameters)?;
        record.serialize_entry(
            "identity",
            &json!({
                "tuning": receipt.tuning_identity,
                "implementation": receipt.implementation_identity,
                "backend": receipt.backend,
            }),
        )?;
        record.serialize_entry("speedup", &Value::Null)?;
        // The served choice's largest error against its default, in the
        // rounding steps that govern its admission.
        record.serialize_entry("chosen_error_ratio", &chosen_error_ratio)?;
        record.serialize_entry("measured", &tuned.measured)?;
        record.serialize_entry("excluded", &tuned.excluded)?;
        record.serialize_entry("rejections", &tuned.rejections)?;
        record.serialize_entry("first_rejection", &tuned.first_rejection)?;
        record.serialize_entry("share", &tuned.progress.map(|progress| progress.share))?;
        record.serialize_entry("stages", &tuned.progress.map(|progress| progress.stages))?;
        record.serialize_entry(
            "standing",
            &tuned
                .progress
                .map(|progress| progress.standing)
                .map(|standing| {
                    json!({
                        "cost": standing.cost,
                        "measured": standing.measured,
                        "generated": standing.generated,
                        "requests": standing.requests,
                        // Measured of the declared product; `exhausted` when the
                        // search asked for everything its order reaches, and then
                        // `generated` is how many of the declared are admissible.
                        "declared": standing.declared.map(|declared| declared.to_string()),
                        "exhausted": standing.exhausted,
                        "programs": standing.programs,
                        "forming_seconds": standing.forming_seconds,
                    })
                }),
        )?;
        record.serialize_entry("time", &tuned.time)?;
        record.serialize_entry("seconds", &self.seconds)?;
        record.end()
    }
}

/// A unit's candidates, each stating its artifact identity once and naming
/// its observations by their positions among the unit's.
struct Candidates<'a>(&'a seismic::TuningResult);

impl Serialize for Candidates<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let receipt = self.0;
        // The defaults' time at each point: what every candidate's time
        // there is relative to (all are in the unit's canonical units).
        let defaults = receipt
            .configurations
            .iter()
            .find(|record| receipt.served.first() == Some(&record.configuration))
            .map_or(&[][..], measured);
        serializer.collect_seq(receipt.configurations.iter().map(|record| Candidate {
            receipt,
            record,
            defaults,
        }))
    }
}

/// One candidate of a unit, beside the defaults' measurements.
struct Candidate<'a> {
    receipt: &'a seismic::TuningResult,
    record: &'a seismic::ConfigurationRecord,
    defaults: &'a [seismic::PointMeasurement],
}

/// A candidate's time at one point over the defaults' there, with the
/// spread of that ratio.
#[derive(serde::Serialize)]
struct RelativeTime<'a> {
    point: &'a str,
    ratio: f64,
    spread: f64,
}

impl Serialize for Candidate<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let Self {
            receipt,
            record,
            defaults,
        } = *self;
        let points = measured(record);
        let relative = points
            .iter()
            .filter_map(|point| {
                let reference = defaults
                    .iter()
                    .find(|reference| reference.point == point.point)?
                    .median_seconds;
                Some(RelativeTime {
                    point: &point.point,
                    ratio: point.median_seconds / reference,
                    spread: point.deviation_seconds / reference,
                })
            })
            .collect::<Vec<_>>();
        // The device seconds its own samples took in all.
        let sampling_seconds = points
            .iter()
            .map(|point| point.samples.iter().sum::<f64>() * point.repetitions as f64)
            .sum::<f64>();
        let violation = receipt
            .admitted(record)
            .err()
            .map(|violation| violation.to_string());
        // An unbounded ratio (an error where the step power is zero) has no
        // number: the violation states it.
        let ratio = error_ratio(receipt, record);
        let mut candidate = serializer.serialize_map(None)?;
        candidate.serialize_entry("configuration", &record.configuration)?;
        candidate.serialize_entry("artifact", &record.artifact)?;
        candidate.serialize_entry("outcome", &record.outcome)?;
        candidate.serialize_entry("relative", &relative)?;
        candidate.serialize_entry("sampling_seconds", &sampling_seconds)?;
        // The formation calls made for it: their seconds, the programs they
        // formed together, and whether none compiled.
        candidate.serialize_entry("formation", &record.formation)?;
        candidate.serialize_entry("observations", &record.observations)?;
        candidate.serialize_entry("error_ratio", &ratio.is_finite().then_some(ratio))?;
        candidate.serialize_entry("admitted", &violation.is_none())?;
        candidate.serialize_entry("violation", &violation)?;
        candidate.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::tuning::{BudgetReport, RefusedOffer, TuningMemory};

    /// What the trace's readers read is what the producer writes under the
    /// schema it states: the harness's measurement reads `budget`'s
    /// `used_seconds`, `overrun_seconds` and `refused`, and the fidelity
    /// tool reads a `pinned` unit's `configuration` in the shape of a
    /// searched unit's choice.
    #[test]
    fn the_trace_states_its_schema_and_what_its_readers_read() {
        let path = std::env::temp_dir().join(format!(
            "magnitude-tuning-trace-{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let trace = TuningTrace::begin(
            &path,
            TracedWork::Search,
            TuningSettings::STANDARD,
            crate::ResolvedNumericalProfile::standard(),
        )
        .unwrap();
        let configuration = seismic::Configuration {
            statics: BTreeMap::from([("G".into(), 8)]),
            params: BTreeMap::from([("SIMDS".into(), 8)]),
            launches: vec![BTreeMap::from([("ROWS".into(), 4)])],
        };
        trace.event(&TuningEvent::Pinned {
            entry: "attention",
            bindings: "Q=bf16".into(),
            statics: configuration.statics.clone(),
            choice: PinnedChoice::Defaults,
            configuration: configuration.clone(),
        });
        trace.event(&TuningEvent::Published {
            entry: "attention",
            bindings: "Q=bf16".into(),
            forecast_seconds: Some(0.002),
            began_seconds: 59.,
            actual_seconds: 0.003,
        });
        trace.event(&TuningEvent::Budget(BudgetReport {
            budget_seconds: 60.,
            used_seconds: 59.4,
            settling_seconds: 0.1,
            unused_seconds: 0.6,
            overrun_seconds: 0.,
            stages: 3,
            refused: vec![RefusedOffer {
                unit: "attention [Q=bf16]".into(),
                stage: seismic::Stage::Measure,
                priced_seconds: 1.5,
                unpriced_operations: 0,
                footprint_bytes: 0,
                unpriced_footprints: 0,
                reason: "seconds",
            }],
            ending: crate::native::tuning::EndingReport::default(),
            memory: TuningMemory {
                ceiling_bytes: Some(1 << 30),
                ..TuningMemory::default()
            },
        }));
        trace.conclude(&[]).unwrap();
        let events = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        std::fs::remove_file(&path).unwrap();
        let [began, pinned, published, budget, concluded] = events.as_slice() else {
            panic!("{events:?}");
        };
        assert_eq!(began["schema"], TRACE_SCHEMA);
        assert_eq!(TRACE_SCHEMA, "tuning-trace/3");
        assert_eq!(pinned["event"], "pinned");
        assert_eq!(pinned["choice"], "defaults");
        assert_eq!(
            pinned["configuration"],
            json!({ "statics": { "G": 8 }, "params": { "SIMDS": 8 }, "launches": [{ "ROWS": 4 }] })
        );
        assert_eq!(
            serde_json::from_value::<seismic::Configuration>(pinned["configuration"].clone())
                .unwrap(),
            configuration
        );
        assert_eq!(published["event"], "published");
        assert_eq!(published["forecast_seconds"], 0.002);
        assert_eq!(published["actual_seconds"], 0.003);
        assert_eq!(budget["used_seconds"], 59.4);
        assert_eq!(budget["settling_seconds"], 0.1);
        assert_eq!(budget["overrun_seconds"], 0.);
        assert_eq!(budget["refused"][0]["reason"], "seconds");
        assert_eq!(budget["memory"]["ceiling_bytes"], 1u64 << 30);
        assert_eq!(concluded["event"], "concluded");
    }
}
