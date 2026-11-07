//! Host/device decomposition of traced steps: what each measured step spent
//! on the host before, during and after the device, and where device time
//! went by entry.

use crate::engine::Step;
use seismic::TracedSubmission;
use serde_json::{Value, json};
use std::collections::BTreeMap;

const MS: f64 = 1e3;

pub fn median(values: &mut [f64]) -> f64 {
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

/// One traced step's host/device decomposition, in milliseconds.
pub fn summary(step: &Step, previous_device_end: Option<f64>) -> Result<Value, String> {
    let first = step
        .submissions
        .first()
        .ok_or("a traced step recorded no submissions")?;
    let last = step.submissions.last().expect("non-empty");
    let device_start = step
        .submissions
        .iter()
        .map(|submission| submission.device.0)
        .fold(f64::INFINITY, f64::min);
    let end = device_end(step);
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
        "device_span_ms": (end - device_start) * MS,
        "device_idle_within_step_ms": (end - device_start - busy) * MS,
        "host_after_device_ms": (step.end - end) * MS,
        "device_gap_from_previous_step_ms": previous_device_end.map(|previous| (device_start - previous) * MS),
    }))
}

pub fn device_end(step: &Step) -> f64 {
    step.submissions
        .iter()
        .map(|submission| submission.device.1)
        .fold(f64::NEG_INFINITY, f64::max)
}

/// The median of every numeric field across step summaries.
pub fn medians(summaries: &[Value]) -> Value {
    let mut fields: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for summary in summaries {
        for (key, value) in summary.as_object().expect("a summary is an object") {
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

/// One entry's launches over the attributed steps: every launch its calls
/// list, those whose condition held (`active`, an evaluated geometry), those
/// the device timed (an active launch with work), and the device time those
/// held it alone.
#[derive(Default)]
struct EntryLaunches {
    listed: usize,
    active: usize,
    timed: usize,
    seconds: f64,
}

/// Launches and device time per entry of `steps` attributed steps, from
/// their submissions that started from `from` (the end of the step before
/// the first: its own launches, untraced under lookahead, are not one of
/// them) to before `until` (the last step's end: a step queued behind it
/// under lookahead is not one of them). Every launch
/// a submission lists is counted: inactive ones (their condition did not
/// hold) and empty ones (active, no work, so untimed) as well as timed ones.
/// The device runs launches one after another in submission order, and a
/// launch's samples bracket its pass, which may open while the launch
/// before is still running; so a timed launch is attributed from the later
/// of its start and the previous launch's end to its own end: what it held
/// the device alone (an empty dispatch, its encode).
pub fn attribution(submissions: &[TracedSubmission], steps: usize, (from, until): (f64, f64)) -> Value {
    let mut entries: BTreeMap<String, EntryLaunches> = BTreeMap::new();
    let mut timed = 0.0;
    let mut previous = f64::NEG_INFINITY;
    for submission in submissions.iter().filter(|submission| (from..until).contains(&submission.device.0)) {
        for launch in &submission.launches {
            let label = if launch.launch == 0 {
                launch.entry.clone()
            } else {
                format!("{}#{}", launch.entry, launch.launch)
            };
            let entry = entries.entry(label).or_default();
            entry.listed += 1;
            entry.active += usize::from(launch.geometry.is_some());
            let Some((sampled, end)) = launch.device else {
                continue;
            };
            let start = sampled.max(previous).min(end);
            previous = previous.max(end);
            entry.timed += 1;
            entry.seconds += end - start;
            timed += end - start;
        }
    }
    let count = steps.max(1) as f64;
    let mut rows = entries
        .into_iter()
        .map(|(entry, launches)| {
            json!({
                "entry": entry,
                "launches_per_step": launches.listed as f64 / count,
                "active_per_step": launches.active as f64 / count,
                "timed_per_step": launches.timed as f64 / count,
                "device_ms_per_step": launches.seconds / count * MS,
                "mean_launch_us": (launches.timed > 0).then(|| launches.seconds / launches.timed as f64 * 1e6),
                "share": launches.seconds / timed,
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

/// Device time over a window of consecutive steps: busy is the union of
/// the device's execution intervals clipped to the window, idle the rest,
/// whatever its cause. A submission divided into queue submissions at its
/// timeline boundaries (`TracedSubmission::queue`) executes in their
/// intervals alone: its gaps between them are idle.
///
/// The idle is one account, by cause (`idle`):
///
/// - **between runs**, located at the submission it precedes, per kind
///   (first entry and launch count): from when the device finished all
///   earlier work until the submission starts, `host` while it was not yet
///   committed and `queued` once it was;
/// - **at boundaries**, the gap from one queue submission's end to the
///   next's start: `held` while the next was not yet handed to the queue
///   (held on the host behind a park); once handed, while a gate it was
///   handed over waiting on was still shut, `awaiting_data` where a park's
///   answer opened it as the answer landed (by the park's kind, and by its
///   kind and layer), `answer_latency` where the
///   keeper opened it on the device's arrival (the answer was ready), and
///   `awaiting_signal` where a plain host signal did; the rest `starting`
///   (the driver and device starting it), of which `submitting` while the
///   call handing it over had not returned.
///
/// The host part of the idle between runs is split by what the harness's
/// thread was doing (`host_by_phase`, from each step's [`HostPhases`]):
/// launching a step (`launch`: reserving and submitting, to its submission's
/// return), finishing it (`finish`: waiting for its completion and taking
/// its outcomes), reconciling them (`reconcile`), the domain's work between
/// steps (`between_steps`), and the harness's own work before it begins the
/// next (`harness`: its per-step reads and records); `other` where no step's
/// phase covers it.
///
/// The causes sum to the idle, each counted once, clipped to the window,
/// which ends as its last step's work does: an idle tail after the
/// device's last work, before work that starts past the window, is none
/// of them.
pub fn window(steps: usize, (start, end): (f64, f64), submissions: &[TracedSubmission], phases: &[HostPhases]) -> Value {
    let clip = |(from, to): (f64, f64)| (from.max(start), to.min(end));
    let busy = union(
        submissions
            .iter()
            .flat_map(|submission| match submission.queue.as_slice() {
                [] => vec![submission.device],
                queue => queue.iter().map(|queued| queued.device).collect(),
            })
            .map(clip)
            .filter(|(from, to)| to > from)
            .collect(),
    );
    let mut ordered = submissions.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| a.device.0.total_cmp(&b.device.0));
    // Per kind: submissions, host idle, queued idle, encode, commit to
    // device start, device time; all seconds.
    let mut kinds: BTreeMap<(String, usize), [f64; 6]> = BTreeMap::new();
    let mut finished = f64::NEG_INFINITY;
    let mut at = Boundaries::default();
    // The host idle between runs, by the harness thread's phase.
    let mut by_phase = BTreeMap::<&'static str, f64>::new();
    for submission in ordered {
        let (begin, finish) = submission.device;
        for pair in submission.queue.windows(2) {
            at.add(pair[0].device.1, &pair[1], (start, end));
        }
        if begin > start && begin <= end {
            let free = finished.max(start);
            let gap = (begin - free).max(0.0);
            let host = (submission.committed - free).clamp(0.0, gap);
            let mut covered = 0.0;
            for (phase, seconds) in phases.iter().flat_map(|phases| phases.overlaps((free, free + host))) {
                *by_phase.entry(phase).or_default() += seconds;
                covered += seconds;
            }
            *by_phase.entry("other").or_default() += host - covered;
            let entry = submission.launches.first().map_or("", |launch| launch.entry.as_str());
            let kind = kinds
                .entry((entry.to_owned(), submission.launches.len()))
                .or_default();
            for (total, value) in kind.iter_mut().zip([
                1.0,
                host,
                gap - host,
                submission.committed - submission.encode_start,
                begin - submission.committed,
                finish - begin,
            ]) {
                *total += value;
            }
        }
        finished = finished.max(finish);
    }
    let steps = steps as f64;
    let per_step = |seconds: f64| seconds * MS / steps;
    let awaiting_data = at.awaiting_data.values().sum::<f64>();
    let mut awaiting_by_park = BTreeMap::<&'static str, f64>::new();
    for ((park, _), seconds) in &at.awaiting_data {
        *awaiting_by_park.entry(park).or_default() += seconds;
    }
    let (between_host, between_queued) = kinds
        .values()
        .fold((0.0, 0.0), |(host, queued), kind| (host + kind[1], queued + kind[2]));
    let by_submission = kinds
        .into_iter()
        .map(|((entry, launches), [count, host, queued, encode, wait, device])| {
            json!({
                "first_entry": entry,
                "launches": launches,
                "submissions_per_step": count / steps,
                "host_idle_ms_per_step": per_step(host),
                "queued_idle_ms_per_step": per_step(queued),
                "mean_encode_ms": encode * MS / count,
                "mean_commit_to_start_ms": wait * MS / count,
                "mean_device_ms": device * MS / count,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "wall_ms_per_step": per_step(end - start),
        "device_busy_ms_per_step": per_step(busy),
        "device_idle_ms_per_step": per_step(end - start - busy),
        "idle": {
            "between_runs": {
                "ms_per_step": per_step(between_host + between_queued),
                "host_ms_per_step": per_step(between_host),
                "queued_ms_per_step": per_step(between_queued),
                "host_by_phase_ms_per_step": by_phase
                    .iter()
                    .map(|(phase, seconds)| (phase.to_string(), json!(per_step(*seconds))))
                    .collect::<serde_json::Map<_, _>>(),
                "by_submission": by_submission,
            },
            "at_boundaries": {
                "per_step": at.count / steps,
                "ms_per_step": per_step(at.held + awaiting_data + at.answer_latency + at.awaiting_signal + at.starting),
                "held_ms_per_step": per_step(at.held),
                "awaiting_data_ms_per_step": per_step(awaiting_data),
                "awaiting_data_by_park_ms_per_step": awaiting_by_park
                    .iter()
                    .map(|(park, seconds)| (park.to_string(), json!(per_step(*seconds))))
                    .collect::<serde_json::Map<_, _>>(),
                "awaiting_data_by_layer_ms_per_step": at
                    .awaiting_data
                    .iter()
                    .map(|((park, layer), seconds)| json!({ "park": park, "layer": layer, "ms_per_step": per_step(*seconds) }))
                    .collect::<Vec<_>>(),
                "answer_latency_ms_per_step": per_step(at.answer_latency),
                "awaiting_signal_ms_per_step": per_step(at.awaiting_signal),
                "starting_ms_per_step": per_step(at.starting),
                "submitting_ms_per_step": per_step(at.submitting),
            },
        },
    })
}

/// One step's instants on the harness thread (`Engine::step`): its start,
/// its submission's return, its finish's return, its reconciliation, its
/// end, and when the harness began the step after it.
#[derive(Clone, Copy, Debug)]
pub struct HostPhases {
    pub start: f64,
    pub submitted: f64,
    pub finished: f64,
    pub reconciled: f64,
    pub end: f64,
    pub next: f64,
}

impl HostPhases {
    /// How much of `(from, to)` each phase covers.
    fn overlaps(&self, (from, to): (f64, f64)) -> impl Iterator<Item = (&'static str, f64)> + '_ {
        [
            ("launch", self.start, self.submitted),
            ("finish", self.submitted, self.finished),
            ("reconcile", self.finished, self.reconciled),
            ("between_steps", self.reconciled, self.end),
            ("harness", self.end, self.next),
        ]
        .into_iter()
        .map(move |(phase, begin, end)| (phase, (end.min(to) - begin.max(from)).max(0.0)))
        .filter(|(_, seconds)| *seconds > 0.0)
    }
}

/// The device's gaps at timeline boundaries, by cause ([`window`]); seconds.
#[derive(Default)]
struct Boundaries {
    count: f64,
    held: f64,
    /// By the kind and layer of the park whose answer opened the gate.
    awaiting_data: BTreeMap<(&'static str, usize), f64>,
    answer_latency: f64,
    awaiting_signal: f64,
    starting: f64,
    submitting: f64,
}

impl Boundaries {
    /// The gap from `free`, the device's end of the queue submission
    /// before, to `next`'s start, clipped to `window`: counted where any of
    /// it lies in the window.
    fn add(&mut self, free: f64, next: &seismic::TracedQueueSubmission, (start, end): (f64, f64)) {
        let (from, to) = (free.max(start), next.device.0.min(end));
        if to <= from {
            return;
        }
        // Each cause holds the gap from `from` to the instant it ended,
        // past the causes before it.
        let mut reached = from;
        let mut until = |instant: f64| {
            let span = (instant.min(to) - reached).max(0.0);
            reached += span;
            span
        };
        self.count += 1.0;
        self.held += until(next.handed);
        let submitting = until(next.taken);
        if let Some((opened, opening)) = next.opened {
            let awaiting = until(opened);
            *match opening {
                seismic::GateOpening::Answered(seismic::ParkAnswer { by: seismic::AnsweredBy::Ready, park, layer }) => {
                    self.awaiting_data.entry((park, layer)).or_default()
                }
                seismic::GateOpening::Answered(seismic::ParkAnswer { by: seismic::AnsweredBy::Arrival, .. }) => {
                    &mut self.answer_latency
                }
                seismic::GateOpening::Signalled => &mut self.awaiting_signal,
            } += awaiting;
        }
        self.submitting += submitting;
        self.starting += submitting + until(to);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::TracedLaunch;

    /// A launch whose condition held (an evaluated geometry), timed where
    /// `device` says.
    fn launch(entry: &str, index: usize, device: Option<(f64, f64)>) -> TracedLaunch {
        TracedLaunch {
            entry: entry.into(),
            launch: index,
            occurrence: 0,
            geometry: Some(seismic::TracedLaunchGeometry {
                workgroups: [1, 1, 1],
                threads_per_workgroup: [32, 1, 1],
                shared_bytes: 0,
            }),
            device,
        }
    }

    /// A launch whose condition did not hold.
    fn inactive(entry: &str) -> TracedLaunch {
        TracedLaunch {
            geometry: None,
            ..launch(entry, 0, None)
        }
    }

    fn submission(encode_start: f64, committed: f64, device: (f64, f64), launches: Vec<TracedLaunch>) -> TracedSubmission {
        TracedSubmission {
            encode_start,
            committed,
            device,
            launches,
            queue: Vec::new(),
        }
    }

    fn close(value: &Value, expected: f64) {
        let value = value.as_f64().expect("a number");
        assert!((value - expected).abs() < 1e-9, "{value} is not {expected}");
    }

    /// A launch is attributed from the later of its start and the previous
    /// launch's end to its own end: one opened while the launch before ran
    /// takes only what it held alone, one wholly inside it takes nothing.
    /// A submission starting from `until` on is not attributed.
    #[test]
    fn a_launch_is_attributed_what_it_held_the_device_alone() {
        let submissions = [
            submission(
                0.0,
                0.0,
                (0.0, 3.0),
                vec![
                    launch("a", 0, Some((0.0, 2.0))),
                    launch("a", 1, Some((1.0, 3.0))),
                    launch("b", 0, Some((1.5, 2.5))),
                ],
            ),
            submission(4.0, 4.0, (5.0, 8.0), vec![launch("c", 0, Some((5.0, 6.0))), launch("c", 0, Some((7.0, 8.0)))]),
            submission(8.0, 8.0, (9.0, 10.0), vec![launch("d", 0, Some((9.0, 10.0)))]),
        ];
        let attributed = attribution(&submissions, 2, (f64::NEG_INFINITY, 8.5));
        close(&attributed["timed_device_ms_per_step"], 2500.0);
        let entries = attributed["entries"].as_array().unwrap();
        let names = entries.iter().map(|entry| entry["entry"].as_str().unwrap()).collect::<Vec<_>>();
        assert_eq!(names[0], "a");
        assert!(!names.contains(&"d"));
        let entry = |name: &str| entries.iter().find(|entry| entry["entry"] == name).unwrap();
        close(&entry("a")["device_ms_per_step"], 1000.0);
        close(&entry("a#1")["device_ms_per_step"], 500.0);
        close(&entry("b")["device_ms_per_step"], 0.0);
        close(&entry("c")["device_ms_per_step"], 1000.0);
        close(&entry("c")["launches_per_step"], 1.0);
        close(&entry("a")["share"], 0.4);
    }

    /// The window holds the attributed steps alone, at their full launch
    /// counts: the shape of a recorded decode attribution at eight steps
    /// (30 gated-delta launches and one readout a step), whose leading
    /// step's submission and the step queued behind the last fall outside
    /// the window, so every count divides by the steps whose launches it
    /// holds.
    #[test]
    fn the_window_holds_exactly_the_attributed_steps() {
        let step = |start: f64| {
            let mut launches = (0..30)
                .map(|layer| {
                    let at = start + layer as f64 * 0.01;
                    launch("gated_delta", 0, Some((at, at + 0.01)))
                })
                .collect::<Vec<_>>();
            launches.push(launch("readout", 0, Some((start + 0.3, start + 0.4))));
            submission(start, start, (start, start + 0.4), launches)
        };
        // Steps run from 0.0 a second apart; each ends 0.5 after it starts.
        // The leading step at 0.0 ends the window's start at 0.5, the
        // eighth attributed step ends it at 8.5, and the step queued behind
        // it starts at 9.0.
        let submissions = (0..10).map(|index| step(index as f64)).collect::<Vec<_>>();
        let attributed = attribution(&submissions, 8, (0.5, 8.5));
        let entries = attributed["entries"].as_array().unwrap();
        let entry = |name: &str| entries.iter().find(|entry| entry["entry"] == name).unwrap();
        close(&entry("gated_delta")["launches_per_step"], 30.0);
        close(&entry("readout")["launches_per_step"], 1.0);
        close(&entry("gated_delta")["device_ms_per_step"], 300.0);
        close(&attributed["timed_device_ms_per_step"], 400.0);
    }

    /// Every launch an attributed submission lists is counted under its
    /// entry, as listed, active (its condition held) and timed (active with
    /// work), whether or not the device timed it; an entry with no timed
    /// launch has no mean launch time.
    #[test]
    fn every_listed_launch_is_counted_timed_or_not() {
        let submissions = [
            submission(
                0.0,
                0.0,
                (0.0, 2.0),
                vec![
                    launch("fill", 0, Some((0.0, 1.0))),
                    launch("fill", 0, None),
                    inactive("fill"),
                    inactive("port"),
                    inactive("port"),
                ],
            ),
            submission(2.0, 2.0, (2.0, 3.0), vec![launch("fill", 0, Some((2.0, 3.0))), inactive("port")]),
        ];
        let attributed = attribution(&submissions, 2, (f64::NEG_INFINITY, f64::INFINITY));
        let entries = attributed["entries"].as_array().unwrap();
        let entry = |name: &str| entries.iter().find(|entry| entry["entry"] == name).unwrap();
        close(&entry("fill")["launches_per_step"], 2.0);
        close(&entry("fill")["active_per_step"], 1.5);
        close(&entry("fill")["timed_per_step"], 1.0);
        close(&entry("fill")["mean_launch_us"], 1e6);
        close(&entry("port")["launches_per_step"], 1.5);
        close(&entry("port")["active_per_step"], 0.0);
        close(&entry("port")["timed_per_step"], 0.0);
        assert!(entry("port")["mean_launch_us"].is_null());
        let listed = entries.iter().map(|entry| entry["launches_per_step"].as_f64().unwrap()).sum::<f64>();
        close(&json!(listed), 3.5);
    }

    /// The window's idle before a submission is split at its commit: host
    /// idle while it was not committed, queued idle once it was, from the
    /// end of all earlier work (or the window's start); busy is the union of
    /// the submissions clipped to the window.
    #[test]
    fn the_window_splits_idle_at_each_submission_commit() {
        let submissions = [
            // Before the window: it only sets when the device is free.
            submission(6.0, 7.0, (8.0, 12.0), vec![launch("early", 0, None)]),
            // Idle 12 to 14: uncommitted to 13, queued after.
            submission(12.5, 13.0, (14.0, 16.0), vec![launch("a", 0, None)]),
            // Idle 16 to 16.5, committed before it began: all queued.
            submission(14.9, 15.0, (16.5, 22.0), vec![launch("b", 0, None)]),
        ];
        let window = window(2, (10.0, 20.0), &submissions, &[]);
        close(&window["wall_ms_per_step"], 5000.0);
        close(&window["device_busy_ms_per_step"], 3750.0);
        close(&window["device_idle_ms_per_step"], 1250.0);
        close(&window["idle"]["between_runs"]["ms_per_step"], 1250.0);
        // With no step's phases, the host idle (12 to 13) is other.
        close(&window["idle"]["between_runs"]["host_by_phase_ms_per_step"]["other"], 500.0);
        let phases = HostPhases { start: 11.0, submitted: 11.5, finished: 12.25, reconciled: 12.5, end: 12.75, next: 14.0 };
        let split = super::window(2, (10.0, 20.0), &submissions, &[phases]);
        let by_phase = &split["idle"]["between_runs"]["host_by_phase_ms_per_step"];
        close(&by_phase["finish"], 125.0);
        close(&by_phase["reconcile"], 125.0);
        close(&by_phase["between_steps"], 125.0);
        close(&by_phase["harness"], 125.0);
        assert!(by_phase.get("other").is_none_or(|other| other.as_f64().unwrap().abs() < 1e-9));
        let idle = window["idle"]["between_runs"]["by_submission"].as_array().unwrap();
        assert_eq!(idle.len(), 2);
        assert_eq!(idle[0]["first_entry"], "a");
        close(&idle[0]["host_idle_ms_per_step"], 500.0);
        close(&idle[0]["queued_idle_ms_per_step"], 500.0);
        close(&idle[0]["mean_encode_ms"], 500.0);
        close(&idle[0]["mean_commit_to_start_ms"], 1000.0);
        assert_eq!(idle[1]["first_entry"], "b");
        close(&idle[1]["host_idle_ms_per_step"], 0.0);
        close(&idle[1]["queued_idle_ms_per_step"], 250.0);
        close(&idle[1]["mean_device_ms"], 5500.0);
    }

    /// A divided submission's gaps between its queue submissions are idle,
    /// each counted once by cause: held until the next was handed to the
    /// queue; then, while a gate it was handed over waiting on was shut,
    /// awaiting its data or the keeper's answer as the gate's opening says;
    /// the rest starting. A gap outside the window is not counted, and one
    /// across its end is clipped.
    #[test]
    fn the_window_splits_each_boundary_by_cause() {
        let queued = |handed: f64, opened: Option<(f64, seismic::GateOpening)>, device: (f64, f64)| {
            seismic::TracedQueueSubmission { handed, taken: handed + 0.25, opened, device }
        };
        let answered = |at: f64, by, park| Some((at, seismic::GateOpening::Answered(seismic::ParkAnswer { by, park, layer: 3 })));
        let mut divided = submission(11.0, 11.0, (12.0, 23.0), vec![launch("a", 0, None)]);
        divided.queue = vec![
            queued(11.0, None, (12.0, 13.0)),
            // Handed at 14, a second after the device freed; its call
            // returns at 14.25; started at 14.5: 1 held, 0.5 starting
            // (0.25 submitting).
            queued(14.0, None, (14.5, 15.0)),
            // Handed before the device freed, its gate opened at 16.5 as
            // its data landed, started at 17: 1.5 awaiting data, 0.5
            // starting.
            queued(14.5, answered(16.5, seismic::AnsweredBy::Ready, "tier wait"), (17.0, 19.0)),
            // Opened by the keeper at 20, started at 20.25: 1 answer
            // latency, 0.25 starting.
            queued(18.0, answered(20.0, seismic::AnsweredBy::Arrival, "tier wait"), (20.25, 23.0)),
        ];
        let mut outside = submission(1.0, 1.0, (2.0, 4.0), vec![launch("b", 0, None)]);
        outside.queue = vec![queued(1.0, None, (2.0, 2.5)), queued(3.0, None, (3.5, 4.0))];
        let submissions = [outside, divided];
        let whole = window(1, (10.0, 23.0), &submissions, &[]);
        let at = &whole["idle"]["at_boundaries"];
        close(&at["per_step"], 3.0);
        close(&at["held_ms_per_step"], 1000.0);
        close(&at["awaiting_data_ms_per_step"], 1500.0);
        close(&at["awaiting_data_by_park_ms_per_step"]["tier wait"], 1500.0);
        assert_eq!(at["awaiting_data_by_layer_ms_per_step"][0]["layer"], 3);
        close(&at["answer_latency_ms_per_step"], 1000.0);
        close(&at["awaiting_signal_ms_per_step"], 0.0);
        close(&at["starting_ms_per_step"], 1250.0);
        close(&at["submitting_ms_per_step"], 250.0);
        close(&at["ms_per_step"], 4750.0);
        // The device idles at the boundaries and before the run: the idle
        // is every cause, once.
        close(&whole["device_idle_ms_per_step"], 4750.0 + 2000.0);
        // A window ending at 16 holds the first gap and a second of the next.
        let clipped = window(1, (10.0, 16.0), &submissions, &[]);
        close(&clipped["idle"]["at_boundaries"]["held_ms_per_step"], 1000.0);
        close(&clipped["idle"]["at_boundaries"]["awaiting_data_ms_per_step"], 1000.0);
    }
}
