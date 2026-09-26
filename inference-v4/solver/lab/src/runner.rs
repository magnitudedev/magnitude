use crate::{generators::Instance, LabResult};
use magnitude_solver::{
    result::{Outcome, StopReason},
    search::{Limits, Options, Policy, Search},
};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::Path,
    process::{Command, Stdio},
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunSettings {
    pub policy: String,
    pub seconds: f64,
    pub memory_mib: usize,
    pub sample_work: u64,
    pub repeat: usize,
}
impl Default for RunSettings {
    fn default() -> Self {
        Self {
            policy: "dfs".into(),
            seconds: 60.0,
            memory_mib: 2048,
            sample_work: 1000,
            repeat: 0,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Sample {
    pub elapsed_ms: f64,
    pub work: u64,
    pub lower_bound: u64,
    pub upper_bound: Option<u64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RunRecord {
    pub schema: u32,
    pub instance: String,
    pub model_fingerprint: String,
    pub model_retained_bytes: usize,
    pub model_variables: usize,
    pub model_factors: usize,
    pub policy: String,
    pub repeat: usize,
    pub status: String,
    pub reason: Option<String>,
    pub cost: Option<u64>,
    pub lower_bound: Option<u64>,
    pub validation_ms: f64,
    pub generation_ms: Option<f64>,
    pub solve_ms: f64,
    pub process_wall_ms: f64,
    pub process_overhead_ms: f64,
    pub observed_rss_bytes: Option<u64>,
    pub observed_cpu_seconds: Option<f64>,
    pub first_feasible_ms: Option<f64>,
    pub winner_found_ms: Option<f64>,
    pub proof_ms: Option<f64>,
    pub settings: RunSettings,
    pub machine: String,
    pub build: String,
    pub revision: Option<String>,
    pub source_worktree_dirty: bool,
    pub stats: serde_json::Value,
    pub samples: Vec<Sample>,
}

pub fn options(policy: &str) -> LabResult<Options> {
    let mut options = Options::default();
    match policy {
        "dfs" => {}
        "best-first" => options.policy = Policy::BestFirst,
        "no-cache" => options.memoize = false,
        "no-decompose" => options.decompose = false,
        "weak-bounds" => options.bounds = false,
        _ => {
            return Err(format!(
                "unknown policy {policy}; use dfs,best-first,no-cache,no-decompose,weak-bounds"
            )
            .into())
        }
    }
    Ok(options)
}

pub fn solve(instance: &Instance, settings: &RunSettings) -> LabResult<RunRecord> {
    if !settings.seconds.is_finite()
        || settings.seconds <= 0.0
        || settings.sample_work == 0
        || settings.memory_mib == 0
    {
        return Err("seconds, memory-mib and sample-work must be positive".into());
    }
    let mut record = RunRecord {
        schema: 1,
        instance: instance.name.clone(),
        model_fingerprint: instance.fingerprint()?,
        model_retained_bytes: instance.model.retained_bytes(),
        model_variables: instance.model.variables().len(),
        model_factors: instance.model.factors().len(),
        generation_ms: instance.generation_ms,
        policy: settings.policy.clone(),
        repeat: settings.repeat,
        settings: settings.clone(),
        machine: command_text("uname", &["-a"]).unwrap_or_else(|| std::env::consts::OS.into()),
        build: format!(
            "{}; arch={}; threads=1; RUSTFLAGS={}",
            if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            },
            std::env::consts::ARCH,
            std::env::var("RUSTFLAGS").unwrap_or_default()
        ),
        revision: command_text("git", &["rev-parse", "HEAD"]),
        source_worktree_dirty: command_text("git", &["status", "--porcelain"])
            .is_some_and(|text| !text.is_empty()),
        ..RunRecord::default()
    };
    let validation_start = Instant::now();
    let mut search = Search::new(Arc::new(instance.model.clone()), options(&settings.policy)?)?;
    record.validation_ms = validation_start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    let total = Duration::from_secs_f64(settings.seconds);
    loop {
        let remaining = total.saturating_sub(start.elapsed());
        let outcome = search.advance(Limits {
            work: settings.sample_work,
            time: Some(remaining),
            memory_bytes: Some(
                settings
                    .memory_mib
                    .checked_mul(1024 * 1024)
                    .ok_or("memory limit overflow")?,
            ),
        })?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        match outcome {
            Outcome::Optimal(solution) => {
                // Deliberately revalidate the returned assignment against the original model.
                let checked = instance.model.validate_assignment(solution.values())?;
                if checked.infeasible
                    || checked.unresolved.is_some()
                    || checked.exact_cost != Some(solution.cost())
                {
                    return Err("solver returned an invalid optimal witness".into());
                }
                record.status = "optimal".into();
                record.cost = Some(solution.cost());
                record.lower_bound = record.cost;
                record.proof_ms = Some(elapsed_ms);
                record.samples.push(Sample {
                    elapsed_ms,
                    work: search.stats().work,
                    lower_bound: solution.cost(),
                    upper_bound: record.cost,
                });
                break;
            }
            Outcome::Infeasible => {
                record.status = "infeasible".into();
                record.proof_ms = Some(elapsed_ms);
                break;
            }
            Outcome::Incomplete(progress) => {
                let incumbent = progress.incumbent.as_ref().map(|solution| solution.cost());
                record.samples.push(Sample {
                    elapsed_ms,
                    work: search.stats().work,
                    lower_bound: progress.lower_bound,
                    upper_bound: incumbent,
                });
                record.lower_bound = Some(progress.lower_bound);
                record.cost = incumbent;
                let work_boundary = matches!(progress.reason, StopReason::Work);
                let reason = format!("{:?}", progress.reason);
                // Work is a cooperative sample boundary. All other stops terminate this run.
                if !work_boundary || start.elapsed() >= total {
                    record.status = "incomplete".into();
                    record.reason = Some(if work_boundary { "Time".into() } else { reason });
                    break;
                }
            }
        }
    }
    record.solve_ms = start.elapsed().as_secs_f64() * 1000.0;
    record.first_feasible_ms = search.stats().first_feasible_seconds.map(|s| s * 1000.0);
    record.winner_found_ms = if record.status == "optimal" {
        search.stats().winner_found_seconds.map(|s| s * 1000.0)
    } else {
        None
    };
    record.stats = serde_json::to_value(search.stats())?;
    Ok(record)
}

/// Run each measured case in its own process. The supervisor bounds time even
/// if preprocessing or one solver operation fails to cooperate with a budget.
pub fn supervised(
    executable: &Path,
    input: &Path,
    output: &Path,
    settings: &RunSettings,
) -> LabResult<RunRecord> {
    if !settings.seconds.is_finite()
        || settings.seconds <= 0.0
        || settings.memory_mib == 0
        || settings.sample_work == 0
    {
        return Err("positive finite time, memory and sample-work budgets required".into());
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = output.with_extension("pending.json");
    if temporary.exists() {
        fs::remove_file(&temporary)?;
    }
    let stderr_path = output.with_extension("stderr.txt");
    let stderr = fs::File::create(&stderr_path)?;
    let start = Instant::now();
    let mut child = Command::new(executable)
        .arg("__solve")
        .arg("--input")
        .arg(input)
        .arg("--out")
        .arg(&temporary)
        .arg("--policy")
        .arg(&settings.policy)
        .arg("--seconds-per-case")
        .arg(settings.seconds.to_string())
        .arg("--memory-mib")
        .arg(settings.memory_mib.to_string())
        .arg("--sample-work")
        .arg(settings.sample_work.to_string())
        .arg("--repeat")
        .arg(settings.repeat.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr))
        .spawn()?;
    let mut peak_rss = None::<u64>;
    let mut cpu_seconds = None::<f64>;
    let mut last_sample = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .unwrap_or_else(Instant::now);
    let timeout = Duration::from_secs_f64(settings.seconds + 1.0);
    let mut watchdog = None;
    let exit = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if last_sample.elapsed() >= Duration::from_millis(100) {
            if let Some((rss, cpu)) = process_usage(child.id()) {
                peak_rss = Some(peak_rss.unwrap_or(0).max(rss));
                cpu_seconds = Some(cpu_seconds.unwrap_or(0.0).max(cpu));
                if rss > (settings.memory_mib as u64).saturating_mul(1024 * 1024) {
                    watchdog = Some("watchdog memory limit".to_string());
                }
            }
            last_sample = Instant::now();
        }
        if start.elapsed() >= timeout {
            watchdog = Some("watchdog wall-time limit (includes preprocessing)".into());
        }
        if watchdog.is_some() {
            child.kill()?;
            break child.wait()?;
        }
        thread::sleep(Duration::from_millis(2));
    };
    let process_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut record = if exit.success() && temporary.exists() {
        serde_json::from_slice::<RunRecord>(&fs::read(&temporary)?)?
    } else {
        let instance = Instance::read(input)?;
        RunRecord {
            schema: 1,
            instance: instance.name.clone(),
            model_fingerprint: instance.fingerprint()?,
            model_retained_bytes: instance.model.retained_bytes(),
            model_variables: instance.model.variables().len(),
            model_factors: instance.model.factors().len(),
            generation_ms: instance.generation_ms,
            source_worktree_dirty: command_text("git", &["status", "--porcelain"])
                .is_some_and(|text| !text.is_empty()),
            policy: settings.policy.clone(),
            repeat: settings.repeat,
            status: if watchdog.is_some() {
                "incomplete".into()
            } else {
                "error".into()
            },
            reason: Some(watchdog.unwrap_or_else(|| {
                format!(
                    "child {exit}: {}",
                    fs::read_to_string(&stderr_path).unwrap_or_default().trim()
                )
            })),
            solve_ms: process_ms,
            settings: settings.clone(),
            ..RunRecord::default()
        }
    };
    record.process_wall_ms = process_ms;
    record.process_overhead_ms = (process_ms - record.solve_ms - record.validation_ms).max(0.0);
    record.observed_rss_bytes = peak_rss;
    record.observed_cpu_seconds = cpu_seconds;
    fs::write(output, serde_json::to_vec_pretty(&record)?)?;
    if temporary.exists() {
        fs::remove_file(temporary)?;
    }
    if fs::metadata(&stderr_path)?.len() == 0 {
        fs::remove_file(stderr_path)?;
    }
    Ok(record)
}

fn command_text(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}
pub(crate) fn process_usage(pid: u32) -> Option<(u64, f64)> {
    let raw = command_text("ps", &["-o", "rss=", "-o", "time=", "-p", &pid.to_string()])?;
    let mut fields = raw.split_whitespace();
    let bytes = fields.next()?.parse::<u64>().ok()?.checked_mul(1024)?;
    let time = fields.next()?;
    let (days, clock) = match time.split_once('-') {
        Some((days, clock)) => (days.parse::<f64>().ok()? * 86400.0, clock),
        None => (0.0, time),
    };
    let seconds = clock.split(':').try_fold(0.0, |acc, part| {
        Some(acc * 60.0 + part.parse::<f64>().ok()?)
    })?;
    Some((bytes, days + seconds))
}
