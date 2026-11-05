//! The run's tuning measurement, read back from the tuning trace its loads
//! appended (`magnitude_executor::TuningTrace`).

use serde_json::{Value, json};
use std::path::Path;

/// File name of the tuning trace in a run directory.
pub const TRACE: &str = "tuning.jsonl";

/// A run's tuning trace (`magnitude_executor::TuningTrace`), its records
/// parsed once for every reading of it.
pub struct Trace {
    /// What errors name it by: its path.
    name: String,
    events: Vec<Value>,
}

/// How a backend's compiler shows a formed program's spill, as the trace's
/// `began` record names a device's backend. Metal and the CPU route report
/// no statistic, so none of theirs is checked.
#[derive(Clone, Copy)]
enum Spill {
    /// CUDA's function attributes: a thread's local memory, which holds its
    /// spills and its stack arrays alike, at registers at their limit.
    LocalBytes,
    /// NVIDIA's Vulkan pipeline executable statistics: a thread's registers
    /// at their limit.
    Registers,
}

impl Spill {
    fn of(backend: &str) -> Option<Self> {
        match backend {
            "Cuda" => Some(Self::LocalBytes),
            "Vulkan" => Some(Self::Registers),
            _ => None,
        }
    }

    /// The statistics it is read from: the thread's registers first.
    fn statistics(self) -> &'static [&'static str] {
        match self {
            Self::LocalBytes => &["registers", "local bytes", "max threads per block"],
            Self::Registers => &["Register Count"],
        }
    }
}

impl Trace {
    /// The trace at `path`, each line a record.
    pub fn read(path: &Path) -> Result<Self, String> {
        let name = path.display().to_string();
        let text = std::fs::read_to_string(path).map_err(|error| format!("tuning trace {name}: {error}"))?;
        Self::parse(name, &text)
    }

    fn parse(name: String, text: &str) -> Result<Self, String> {
        let events = text
            .lines()
            .enumerate()
            .map(|(index, line)| serde_json::from_str(line).map_err(|error| format!("tuning trace {name} line {}: {error}", index + 1)))
            .collect::<Result<_, _>>()?;
        Ok(Self { name, events })
    }

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
    pub fn measurement(&self, budget_seconds: f64, fresh: bool) -> Result<Value, String> {
        self.measurement_of(budget_seconds, fresh).map_err(|error| format!("tuning trace {}: {error}", self.name))
    }

    fn measurement_of(&self, budget_seconds: f64, fresh: bool) -> Result<Value, String> {
        let mut loads: Vec<Value> = Vec::new();
        for (index, event) in self.events.iter().enumerate() {
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

    /// The units of the trace whose default configuration spills, one line
    /// each: a launch whose formed program's registers are at their limit,
    /// the device's per-thread ceiling (`registers_per_thread`) or a
    /// workgroup's threads times its registers reaching the register file
    /// (`registers_per_group`, one workgroup per unit, the compiler's
    /// registers bounded by the file), and whose backend shows a spill
    /// there. CUDA reports a thread's registers and local memory ("local
    /// bytes"), which holds its spills and its stack arrays alike: the
    /// compiler spills only at the limit, so local memory below it is a
    /// stack array, and at it is taken to spill. The threads the compiler
    /// bounds a CUDA function's registers by are its "max threads per
    /// block" (its `__launch_bounds__`, as its registers admit). NVIDIA's
    /// Vulkan driver reports no usable spill size ("Local Memory Size" is a
    /// 2^36 sentinel, "Stack Size" always 0), so a program at the limit
    /// ("Register Count"; its workgroup's threads) is taken to spill. A
    /// compiler may also spill below the ceiling (NVIDIA 615.71.09 gives a
    /// Vulkan kernel far past its registers 168 of 255), which no statistic
    /// shows. Both limits are the device's properties, from the trace's
    /// `began` record: CUDA's from the driver, Vulkan's probed. A device
    /// that has not had them measured (its probes not run by the
    /// measurement job) is an error, as is a Vulkan launch without its
    /// workgroup's threads.
    ///
    /// A unit's defaults are recorded whether or not they were measured
    /// (`default_launches`). On a device whose backend shows spills
    /// ([`Spill`]), a unit with a formed launch that does not carry
    /// its backend's statistics (a driver that names its statistics
    /// otherwise), or with no formed launch, was not checked: the check
    /// fails listing every such unit. A trace in which no unit is on such a device
    /// is an error, not a pass: nothing was checked.
    pub fn default_spills(&self) -> Result<Vec<String>, String> {
        self.default_spills_of().map_err(|error| format!("tuning trace {}: {error}", self.name))
    }

    fn default_spills_of(&self) -> Result<Vec<String>, String> {
        let mut spills = Vec::new();
        let mut unchecked = Vec::new();
        let mut checked = 0;
        // The device of the load the records that follow belong to.
        let unknown = Value::Null;
        let mut device = &unknown;
        for event in &self.events {
            if event["event"] == "began" {
                device = &event["device"];
                continue;
            }
            if event["event"] != "finished" {
                continue;
            }
            // A unit is named by its entry, its bindings and its defaults'
            // settings (the first choice it served); a record without them
            // names nothing a spill could be reported against.
            let named = |key: &str| event[key].as_str().ok_or_else(|| format!("a finished unit record has no `{key}`"));
            let defaults = event["served"][0]
                .get("params")
                .ok_or("a finished unit record serves no default configuration")?;
            let unit = format!("{} [{}] {defaults}", named("entry")?, named("bindings")?);
            let Some((backend, properties)) = device.as_object().and_then(|properties| properties.iter().next()) else {
                continue;
            };
            let Some(spill) = Spill::of(backend) else {
                continue;
            };
            let value = |launch: &Value, name: &str| {
                launch["statistics"].as_array().into_iter().flatten().find(|statistic| statistic["name"] == name).and_then(
                    |statistic| statistic["value"].as_object()?.values().next()?.as_u64(),
                )
            };
            let launches = event["default_launches"].as_array().into_iter().flatten().enumerate().filter(|(_, launch)| !launch.is_null()).collect::<Vec<_>>();
            let bare = launches
                .iter()
                .filter(|(_, launch)| spill.statistics().iter().any(|name| value(launch, name).is_none()))
                .map(|(ordinal, _)| ordinal.to_string())
                .collect::<Vec<_>>();
            if launches.is_empty() {
                unchecked.push(format!("{unit}: no formed launch"));
                continue;
            }
            if !bare.is_empty() {
                let names = spill.statistics().iter().map(|name| format!("\"{name}\"")).collect::<Vec<_>>();
                unchecked.push(format!("{unit}: launches {} do not carry each of {}", bare.join(", "), names.join(", ")));
                continue;
            }
            let limit = |name: &str| {
                properties[name]
                    .as_u64()
                    .ok_or_else(|| format!("the device's `{name}` is not measured: run its probes (the measurement job) first"))
            };
            let (ceiling, file) = (limit("registers_per_thread")?, limit("registers_per_group")?);
            checked += 1;
            for (ordinal, launch) in launches {
                let read = |name: &str| value(launch, name).expect("every launch checked carries its backend's statistics");
                let registers = read(spill.statistics()[0]);
                let threads = match spill {
                    Spill::LocalBytes => read("max threads per block"),
                    Spill::Registers => launch["threads"]
                        .as_u64()
                        .ok_or_else(|| format!("{unit} launch {ordinal} reports registers without its workgroup's threads"))?,
                };
                let limited = if registers >= ceiling {
                    Some(format!("{registers} registers, the per-thread ceiling"))
                } else if threads * registers >= file {
                    Some(format!("{threads} threads of {registers} registers take the whole {file}-register file"))
                } else {
                    None
                };
                let reason = match spill {
                    Spill::LocalBytes => {
                        let local = read("local bytes");
                        limited.filter(|_| local > 0).map(|limited| format!("{local} local bytes per thread at {limited}"))
                    }
                    Spill::Registers => limited,
                };
                if let Some(reason) = reason {
                    spills.push(format!("{unit} launch {ordinal}: {reason}"));
                }
            }
        }
        if !unchecked.is_empty() {
            return Err(format!("units whose defaults were not checked:\n{}", unchecked.join("\n")));
        }
        if checked == 0 {
            return Err("no unit is on a device whose backend shows spills, so nothing was checked".into());
        }
        Ok(spills)
    }
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
        let measured = Trace::parse("t".into(), &trace).unwrap().measurement(60.0, true).unwrap();
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
        assert!(Trace::parse("t".into(), &previous).unwrap().measurement(60.0, true).is_err_and(|error| error.contains("schema")));
    }

    /// A default spills at the device's probed per-thread ceiling, or takes
    /// the whole register file at threads times registers; a device whose
    /// limits are not measured, a register count without its threads, and a
    /// unit on a backend that shows spills without the statistic its
    /// backend shows them by are errors, not passes; a unit on a backend
    /// that reports none is not checked.
    #[test]
    fn default_spills_read_the_devices_probed_register_limits() {
        let registers = |count: u64| format!(r#"[{{"name":"Register Count","description":"","value":{{"Uint":{count}}}}}]"#);
        let finished = |entry: &str, threads: &str, statistics: &str| {
            format!(
                r#"{{"event":"finished","entry":"{entry}","bindings":"b","served":[{{"params":{{}}}}],"default_launches":[null,{{"threads":{threads},"statistics":{statistics}}}]}}"#
            )
        };
        let began = |device: &str| format!(r#"{{"event":"began","device":{device}}}"#);
        let vulkan = |file: &str| began(&format!(r#"{{"Vulkan":{{"registers_per_thread":255,"registers_per_group":{file}}}}}"#));
        let spills = |lines: &[String]| Trace::parse("t".into(), &lines.join("\n")).unwrap().default_spills();
        let found = spills(&[
            vulkan("65536"),
            finished("ceiling", "128", &registers(255)),
            finished("file", "1024", &registers(64)),
            finished("fits", "512", &registers(64)),
            began(r#"{"Metal":{"shared_bytes_per_group":32768,"tensor_ops":false}}"#),
            finished("metal", "null", "[]"),
        ])
        .unwrap();
        assert_eq!(found.len(), 2, "{found:?}");
        assert!(found[0].starts_with("ceiling [b] {} launch 1: 255 registers"));
        assert!(found[1].contains("1024 threads of 64 registers take the whole 65536-register file"));
        assert!(spills(&[vulkan("null"), finished("fits", "512", &registers(64))]).is_err_and(|error| error.contains("registers_per_group")));
        assert!(spills(&[vulkan("65536"), finished("fits", "null", &registers(64))]).is_err_and(|error| error.contains("without its workgroup's threads")));
        let bare = spills(&[vulkan("65536"), finished("fits", "512", &registers(64)), finished("bare", "512", "[]")]);
        assert!(bare.is_err_and(|error| error.contains("bare [b] {}: launches 1 do not carry each of \"Register Count\"")));
        assert!(spills(&[began(r#"{"Metal":{}}"#), finished("metal", "null", "[]")]).is_err_and(|error| error.contains("nothing was checked")));
    }

    /// On CUDA, local memory spills only where the registers are at their
    /// limit for the function's block bound: below it, it is a stack array.
    #[test]
    fn cuda_local_memory_spills_only_at_the_register_limit() {
        let function = |registers: u64, local: u64, threads: u64| {
            let statistic = |name: &str, value: u64| format!(r#"{{"name":"{name}","description":"","value":{{"Uint":{value}}}}}"#);
            format!(
                "[{},{},{}]",
                statistic("registers", registers),
                statistic("local bytes", local),
                statistic("max threads per block", threads)
            )
        };
        let finished = |entry: &str, statistics: &str| {
            format!(
                r#"{{"event":"finished","entry":"{entry}","bindings":"b","served":[{{"params":{{}}}}],"default_launches":[{{"threads":null,"statistics":{statistics}}}]}}"#
            )
        };
        let cuda = r#"{"event":"began","device":{"Cuda":{"registers_per_thread":255,"registers_per_group":65536}}}"#.to_owned();
        let spills = |lines: &[String]| Trace::parse("t".into(), &lines.join("\n")).unwrap().default_spills();
        let found = spills(&[
            cuda.clone(),
            finished("ceiling", &function(255, 104, 128)),
            finished("file", &function(64, 16, 1024)),
            finished("stack", &function(136, 32, 384)),
            finished("registers", &function(255, 0, 256)),
        ])
        .unwrap();
        assert_eq!(found.len(), 2, "{found:?}");
        assert!(found[0].contains("ceiling [b] {} launch 0: 104 local bytes per thread at 255 registers, the per-thread ceiling"));
        assert!(found[1].contains("file [b] {} launch 0: 16 local bytes per thread at 1024 threads of 64 registers"));
        let bare = spills(&[cuda, finished("bare", r#"[{"name":"registers","description":"","value":{"Uint":40}}]"#)]);
        assert!(bare.is_err_and(|error| error.contains("bare [b] {}: launches 0 do not carry each of")));
    }
}
