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

/// Device time per entry of `steps` attributed steps, from their launches
/// that started before `until` (the last step's end: a step queued behind it
/// under lookahead is not one of them).
pub fn attribution(submissions: &[TracedSubmission], steps: usize, until: f64) -> Value {
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

/// Device time over a window of consecutive steps: busy is the union of the
/// submissions' device intervals clipped to the window, idle the rest.
pub fn window(steps: usize, (start, end): (f64, f64), submissions: &[TracedSubmission]) -> Value {
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
