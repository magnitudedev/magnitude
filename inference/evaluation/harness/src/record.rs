//! The evaluation result record (contract "Result record") and the run
//! directory it is written into.

use crate::digest::sha256_hex;
use serde_json::{Map, Value, json};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

pub const SCHEMA: &str = "evaluation-result/1";

/// One harness run: its directory, from creation to its `result.json`.
pub struct Run {
    pub directory: PathBuf,
    tool: &'static str,
    started_at: String,
    command: Vec<String>,
}

/// What a finished run records besides its timing and reproduction command.
pub struct RunRecord {
    pub build: Value,
    pub model: String,
    pub artifacts: Value,
    pub fixture_set: (PathBuf, String),
    pub rendered_digest: String,
    pub host: Value,
    pub measurements: Map<String, Value>,
    /// Trace kinds and their files, relative to the run directory.
    pub traces: Map<String, Value>,
}

impl Run {
    /// Create the run directory: `output`, or a new directory under the
    /// evaluation results.
    pub fn create(output: Option<PathBuf>, tool: &'static str) -> Result<Self, String> {
        let now = SystemTime::now();
        let directory = match output {
            Some(directory) => directory,
            None => {
                let nanos = now
                    .duration_since(UNIX_EPOCH)
                    .expect("the clock is after the epoch")
                    .as_nanos();
                let short = &sha256_hex(format!("{nanos}/{}", std::process::id()).as_bytes())[..8];
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../results")
                    .join(format!("{}-{tool}-{short}", utc(now).1))
            }
        };
        std::fs::create_dir_all(&directory)
            .map_err(|error| format!("run directory {}: {error}", directory.display()))?;
        eprintln!("magnitude-harness: run directory {}", directory.display());
        Ok(Self {
            directory,
            tool,
            started_at: utc(now).0,
            command: std::env::args().collect(),
        })
    }

    pub fn path(&self, name: &str) -> PathBuf {
        self.directory.join(name)
    }

    pub fn finish(self, record: RunRecord) -> Result<(), String> {
        let (fixture_path, fixture_digest) = record.fixture_set;
        let result = json!({
            "schema": SCHEMA,
            "tool": self.tool,
            "started_at": self.started_at,
            "finished_at": utc(SystemTime::now()).0,
            "build": record.build,
            "model": record.model,
            "artifacts": record.artifacts,
            "fixture_set": { "path": fixture_path, "digest": fixture_digest },
            "rendered_digest": record.rendered_digest,
            "host": record.host,
            "command": self.command,
            "measurements": record.measurements,
            "traces": record.traces,
        });
        let path = self.path("result.json");
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&result).expect("a result record serializes"),
        )
        .map_err(|error| format!("writing {}: {error}", path.display()))?;
        eprintln!("magnitude-harness: wrote {}", path.display());
        // What the process holds once its measurement is over, beside the
        // engine's lines for the moments of each load, and the most its
        // footprint has been.
        let memory = seismic::ProcessMemory::read();
        eprintln!(
            "magnitude-harness: memory {}",
            json!({
                "moment": "run finished",
                "footprint_bytes": memory.footprint_bytes,
                "malloc_live_bytes": memory.malloc_live_bytes,
                "graphics_bytes": memory.graphics_bytes,
                "mapped_bytes": memory.mapped_bytes,
                "rest_bytes": memory.rest_bytes(),
                "peak_footprint_bytes": seismic::ProcessMemory::peak_footprint(),
            })
        );
        Ok(())
    }
}

/// `(RFC 3339, compact)` UTC renderings of `time`, to the second.
fn utc(time: SystemTime) -> (String, String) {
    let seconds = time
        .duration_since(UNIX_EPOCH)
        .expect("the clock is after the epoch")
        .as_secs();
    let (days, of_day) = ((seconds / 86_400) as i64, seconds % 86_400);
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    let (hour, minute, second) = (of_day / 3600, of_day / 60 % 60, of_day % 60);
    (
        format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z"),
        format!("{year:04}{month:02}{day:02}T{hour:02}{minute:02}{second:02}Z"),
    )
}

/// The build this harness runs, as the caller identified it: `identity` is a
/// JSON file `{"commit", "dirty_digest"}` written by the evaluation remote
/// package, which alone defines build identity. Without it the record's
/// build is `null`.
pub fn build(identity: Option<&Path>) -> Result<Value, String> {
    let Some(path) = identity else {
        return Ok(Value::Null);
    };
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("build identity {}: {error}", path.display()))?;
    let value: Value = serde_json::from_str(&text)
        .map_err(|error| format!("build identity {}: {error}", path.display()))?;
    if !value["commit"].is_string()
        || !(value["dirty_digest"].is_string() || value["dirty_digest"].is_null())
    {
        return Err(format!(
            "build identity {} must hold commit and dirty_digest",
            path.display()
        ));
    }
    Ok(json!({ "commit": value["commit"], "dirty_digest": value["dirty_digest"] }))
}

/// Hardware facts of this host, with unavailable values `null`. `gpus` are
/// the execution devices the run opened, when it opened any.
pub fn host(gpus: Option<Vec<String>>) -> Value {
    let run = |program: &str, arguments: &[&str]| {
        Command::new(program)
            .args(arguments)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
            .filter(|text| !text.is_empty())
    };
    let macos = cfg!(target_os = "macos");
    let meminfo = || {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|text| {
                text.lines()
                    .find(|line| line.starts_with("MemTotal:"))
                    .and_then(|line| line.split_whitespace().nth(1))
                    .and_then(|kib| kib.parse::<u64>().ok())
                    .map(|kib| kib * 1024)
            })
    };
    let cpuinfo = || {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|text| {
                text.lines()
                    .find(|line| line.starts_with("model name"))
                    .and_then(|line| line.split_once(':'))
                    .map(|(_, name)| name.trim().to_owned())
            })
    };
    json!({
        "hostname": run("hostname", &[]),
        "os": std::env::consts::OS,
        "os_version": if macos { run("sw_vers", &["-productVersion"]) } else { run("uname", &["-r"]) },
        "chip": if macos { run("sysctl", &["-n", "machdep.cpu.brand_string"]) } else { cpuinfo() },
        "cpu_cores": if macos {
            run("sysctl", &["-n", "hw.physicalcpu"]).and_then(|cores| cores.parse::<u64>().ok())
        } else {
            std::thread::available_parallelism().ok().map(|cores| cores.get() as u64)
        },
        "memory_bytes": if macos {
            run("sysctl", &["-n", "hw.memsize"]).and_then(|bytes| bytes.parse::<u64>().ok())
        } else {
            meminfo()
        },
        "gpus": gpus,
    })
}
