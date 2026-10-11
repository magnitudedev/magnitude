//! The run's tuning measurement, read back from the tuning trace its loads
//! appended (`magnitude_executor::TuningTrace`).

use serde_json::{Value, json};
use std::path::Path;

/// File name of the tuning trace in a run directory.
pub const TRACE: &str = "tuning.jsonl";

/// `{ budget_seconds, fresh, outcome }`: the outcome's `loads` list, in
/// order, every search and every load the trace records, each tagged with
/// its `work`. A `search` entry is a model's Prepare, in a worker process
/// of its own: tuning is read from it (its units searched and stored, its
/// tuning seconds, and what became of its budget: when its optional work
/// ended, the time it ran past the budget and the offered pieces that did
/// not fit). A `load` entry is the load in this process, which searched
/// nothing: the units it read a stored record for, the units it served at
/// their canonical defaults for want of one (`unprepared`) and, under a
/// development tuning pin, its pinned units: counts of replayed and
/// default units, and every unit not replayed (no recorded configuration,
/// so its declared default ran untuned). A load whose records were all
/// stored has no search entry before it.
pub fn measurement(trace: &Path, budget_seconds: f64, fresh: bool) -> Result<Value, String> {
    let text = std::fs::read_to_string(trace)
        .map_err(|error| format!("tuning trace {}: {error}", trace.display()))?;
    measurement_of(&text, budget_seconds, fresh)
        .map_err(|error| format!("tuning trace {}: {error}", trace.display()))
}

fn measurement_of(text: &str, budget_seconds: f64, fresh: bool) -> Result<Value, String> {
    let mut loads: Vec<Value> = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let event: Value = serde_json::from_str(line)
            .map_err(|error| format!("line {}: {error}", index + 1))?;
        match event["event"].as_str() {
            Some("began") => {
                if event["schema"] != magnitude_executor::TRACE_SCHEMA {
                    return Err(format!(
                        "line {}: schema {}, not {}",
                        index + 1,
                        event["schema"],
                        magnitude_executor::TRACE_SCHEMA
                    ));
                }
                loads.push(json!({
                    "work": event["work"],
                    "budget_seconds": event["budget_seconds"],
                    "fresh": event["fresh"],
                    "pinned": { "replayed": 0, "defaults": 0, "not_replayed": [] },
                }));
            }
            Some("pinned") => {
                let load = loads.last_mut().ok_or("a pinned unit precedes its load")?;
                let choice = event["choice"]
                    .as_str()
                    .ok_or("a pinned unit names no choice")?;
                match choice {
                    // Units that ran their declared default because the
                    // recording has none are listed, never folded into a count.
                    "not_replayed" => load["pinned"]["not_replayed"]
                        .as_array_mut()
                        .expect("an array")
                        .push(json!({
                            "entry": event["entry"],
                            "bindings": event["bindings"],
                            "statics": event["statics"],
                        })),
                    "replayed" | "defaults" => {
                        let count = load["pinned"][choice].as_u64().expect("a count");
                        load["pinned"][choice] = json!(count + 1);
                    }
                    other => return Err(format!("unknown pinned choice {other}")),
                }
            }
            Some("budget") => {
                let load = loads.last_mut().ok_or("a tuning budget precedes its load")?;
                load["concluded_at_seconds"] = event["used_seconds"].clone();
                load["overrun_seconds"] = event["overrun_seconds"].clone();
                load["refused"] = json!(
                    event["refused"]
                        .as_array()
                        .ok_or("a tuning budget lists no refused offers")?
                        .len()
                );
            }
            Some("concluded") => {
                let load = loads
                    .last_mut()
                    .ok_or("a tuning conclusion precedes its load")?;
                for key in ["units", "searched", "stored", "unprepared", "tuning_seconds"] {
                    load[key] = event[key].clone();
                }
                load["seconds"] = event["seconds"].clone();
            }
            _ => {}
        }
    }
    if let Some(open) = loads.iter().position(|load| load.get("units").is_none()) {
        return Err(format!("load {open} never concluded"));
    }
    Ok(json!({
        "budget_seconds": budget_seconds,
        "fresh": fresh,
        "outcome": { "loads": loads },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The measurement reads what the engine's trace of this schema
    /// writes: a first load is its Prepare's search (the budget from its
    /// `budget` event, its conclusion) and then the load, which searched
    /// nothing (its pinned units, the units it served unprepared). A trace
    /// of another schema is refused.
    #[test]
    fn a_loads_measurement_is_read_from_the_producers_schema() {
        let schema = magnitude_executor::TRACE_SCHEMA;
        let trace = [
            format!(r#"{{"event":"began","schema":"{schema}","work":"search","budget_seconds":60.0,"fresh":true}}"#),
            r#"{"event":"stage","seconds":1.0,"class":"measure"}"#.to_owned(),
            r#"{"event":"published","seconds":59.0,"forecast_seconds":0.002,"actual_seconds":0.003}"#.to_owned(),
            r#"{"event":"budget","seconds":59.5,"budget_seconds":60.0,"used_seconds":59.4,"settling_seconds":0.1,"unused_seconds":0.6,"overrun_seconds":0.0,"stages":3,"refused":[{"unit":"a","class":"measure","reason":"seconds"},{"unit":"b","class":"confirm","reason":"bytes"}]}"#.to_owned(),
            r#"{"event":"concluded","seconds":60.2,"units":2,"searched":2,"stored":0,"unprepared":0,"tuning_seconds":58.0}"#.to_owned(),
            format!(r#"{{"event":"began","schema":"{schema}","work":"load","budget_seconds":60.0,"fresh":false}}"#),
            r#"{"event":"pinned","seconds":0.1,"entry":"rope","bindings":"","statics":{},"choice":"defaults","configuration":{"statics":{},"params":{"W":4},"launches":[]}}"#.to_owned(),
            r#"{"event":"unprepared","seconds":0.2,"entry":"dense","bindings":"","statics":{}}"#.to_owned(),
            r#"{"event":"concluded","seconds":0.3,"units":2,"searched":0,"stored":2,"unprepared":1,"tuning_seconds":0.0}"#.to_owned(),
        ]
        .join("\n");
        let measured = measurement_of(&trace, 60.0, true).unwrap();
        let loads = measured["outcome"]["loads"].as_array().unwrap();
        assert_eq!(loads.len(), 2);
        // Tuning is read from the search, loading from the load.
        assert_eq!((&loads[0]["work"], &loads[1]["work"]), (&json!("search"), &json!("load")));
        assert_eq!((&loads[1]["stored"], &loads[1]["unprepared"]), (&json!(2), &json!(1)));
        assert_eq!(loads[0]["concluded_at_seconds"], 59.4);
        assert_eq!(loads[0]["overrun_seconds"], 0.0);
        assert_eq!(loads[0]["refused"], 2);
        assert_eq!(loads[0]["searched"], 2);
        assert_eq!(loads[0]["seconds"], 60.2);
        // A load that searched nothing reports no budget.
        assert!(loads[1].get("concluded_at_seconds").is_none());
        assert_eq!(loads[1]["pinned"]["defaults"], 1);
        let previous = trace.replace(schema, "tuning-trace/2");
        assert!(measurement_of(&previous, 60.0, true).is_err_and(|error| error.contains("schema")));
    }
}
