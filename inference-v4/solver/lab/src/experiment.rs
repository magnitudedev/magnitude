//! Fixed-cohort approximate-search experiments. All policies consume saved Model
//! values; reference metadata is used only outside the timed search.
use crate::{
    generators::{self, Instance, Parameters, Random, Reference},
    reference::{self, ReferenceOutcome, ReferenceResult},
    LabResult,
};
use magnitude_solver::{
    model::{Arithmetic, Constraint, Factor, FactorKind, Model},
    result::{Outcome, StopReason},
    scheduling::SchedulingConstraint,
    search::{Algorithm, Limits, NeighborhoodOptions, Options, Search},
};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::Path,
    process::{Command, Stdio},
    sync::Arc,
    time::{Duration, Instant},
};

pub const METHODS: &[&str] = &["exact", "random", "greedy", "anneal", "joint", "lns"];
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    pub milliseconds: u64,
    pub sample_work: u64,
    pub memory_mib: usize,
    pub neighborhood: NeighborhoodOptions,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            milliseconds: 1000,
            sample_work: 100,
            memory_mib: 512,
            neighborhood: NeighborhoodOptions::default(),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Point {
    pub elapsed_ms: f64,
    pub search_ms: f64,
    pub work: u64,
    pub cost: Option<u64>,
    pub lower_bound: u64,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Record {
    pub instance: String,
    pub family: String,
    pub fingerprint: String,
    pub method: String,
    pub seed: u64,
    pub settings: Settings,
    pub options: Option<Options>,
    pub parameters: Parameters,
    pub variables: usize,
    pub factors: usize,
    pub model_bytes: usize,
    pub construction_ms: Option<f64>,
    pub initialization_ms: Option<f64>,
    pub validation_ms: Option<f64>,
    pub elapsed_ms: f64,
    pub process_ms: f64,
    pub process_startup_ms: Option<f64>,
    pub observed_rss_bytes: Option<u64>,
    pub status: String,
    pub reason: Option<String>,
    pub reference: Option<ReferenceResult>,
    pub points: Vec<Point>,
    pub values: Option<Vec<i64>>,
    pub stats: serde_json::Value,
    pub selected_implementation_optimum: Option<ReferenceResult>,
    pub final_schedule_quality: Option<ScheduleQuality>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScheduleQuality {
    pub witness_cost: u64,
    pub selected_implementation_optimum: u64,
    pub global_optimum: u64,
    pub schedule_ratio: Option<f64>,
    pub implementation_ratio: Option<f64>,
    pub note: String,
}

pub fn options(method: &str, seed: u64, settings: &Settings) -> LabResult<Option<Options>> {
    let mut n = settings.neighborhood.clone();
    n.seed = seed;
    match method {
        "random" => return Ok(None),
        "exact" => return Ok(Some(Options::default())),
        "greedy" => {
            n.max_neighborhood_variables = 1;
            n.population_size = 1;
            n.exploration = false;
        }
        "anneal" => {
            n.max_neighborhood_variables = 1;
            n.population_size = 1;
        }
        "joint" => {
            n.population_size = 1;
            n.exploration = false;
            n.restart_after = u64::MAX;
        }
        "lns" => {}
        _ => return Err(format!("unknown experiment method {method}").into()),
    }
    Ok(Some(Options {
        algorithm: Algorithm::Neighborhood(n),
        ..Options::default()
    }))
}

pub fn solve(saved: &Instance, method: &str, seed: u64, settings: &Settings) -> LabResult<Record> {
    if settings.milliseconds == 0 || settings.sample_work == 0 || settings.memory_mib == 0 {
        return Err("positive experiment budgets required".into());
    }
    let fingerprint = saved.fingerprint()?;
    let options = options(method, seed, settings)?;
    // Reconstruct this particular saved fixture in every child. This charges
    // per-instance generation and validation rather than a warmed model cache.
    let begin = Instant::now();
    let generated = generators::generate(&saved.family, saved.parameters.clone(), saved.seed)?;
    if generated.model != saved.model {
        return Err(
            "saved fixture does not reproduce from its parameters; refuse incomparable timing"
                .into(),
        );
    }
    let model = Arc::new(generated.model);
    let construction_ms = ms(begin.elapsed());
    let init = Instant::now();
    let mut search = options
        .as_ref()
        .map(|o| Search::new(model.clone(), o.clone()))
        .transpose()?;
    let initialization_ms = ms(init.elapsed());
    let search_begin = Instant::now();
    let total = Duration::from_millis(settings.milliseconds);
    let memory = settings
        .memory_mib
        .checked_mul(1024 * 1024)
        .ok_or("memory overflow")?;
    let mut rng = Random::new(seed);
    let mut work = 0_u64;
    let mut cost = None;
    let mut values = None;
    let mut points = vec![Point {
        elapsed_ms: ms(begin.elapsed()),
        search_ms: 0.0,
        work,
        cost,
        lower_bound: 0,
    }];
    let mut validation_ms = 0.0;
    let mut status = "incomplete".to_string();
    let mut reason = Some("time budget".into());
    let mut milestones: Vec<_> = [10, 30, 100, 300, 1000]
        .into_iter()
        .filter(|&x| x < settings.milliseconds)
        .chain([settings.milliseconds])
        .collect();
    milestones.sort_unstable();
    let mut checkpoint = 0;
    while begin.elapsed() < total {
        while checkpoint + 1 < milestones.len()
            && ms(begin.elapsed()) >= milestones[checkpoint] as f64
        {
            checkpoint += 1;
        }
        let remaining =
            Duration::from_millis(milestones[checkpoint]).saturating_sub(begin.elapsed());
        let mut terminal = false;
        let mut lower = 0;
        let proposal = if let Some(search) = &mut search {
            let outcome = search.advance(Limits {
                work: settings.sample_work,
                time: Some(remaining),
                memory_bytes: Some(memory),
            })?;
            work = search.stats().work;
            match outcome {
                Outcome::Optimal(solution) => {
                    terminal = true;
                    status = "optimal".into();
                    reason = None;
                    lower = solution.cost();
                    Some((solution.values().to_vec(), solution.cost()))
                }
                Outcome::Infeasible => {
                    terminal = true;
                    status = "infeasible".into();
                    reason = None;
                    None
                }
                Outcome::Incomplete(progress) => {
                    lower = progress.lower_bound;
                    if !matches!(progress.reason, StopReason::Work | StopReason::Time) {
                        terminal = true;
                        reason = Some(format!("{:?}", progress.reason));
                    }
                    progress.incumbent.map(|s| (s.values().to_vec(), s.cost()))
                }
            }
        } else {
            // Complete candidate sampling, using only typed model relations to
            // construct arithmetic consequences. No kernel-family annotation.
            let candidate = random_candidate(&model, &mut rng)?;
            work += 1;
            let result = reference::evaluate(&model, &candidate)?;
            if result.feasible && !result.unresolved {
                Some((candidate, result.cost))
            } else {
                None
            }
        };
        if let Some((candidate, candidate_cost)) = proposal {
            if cost.is_none_or(|old| candidate_cost < old) {
                let validation = Instant::now();
                let independent = reference::evaluate(&model, &candidate)?;
                let original = model.validate_assignment(&candidate)?;
                if !independent.feasible
                    || independent.unresolved
                    || independent.cost != candidate_cost
                    || original.infeasible
                    || original.unresolved.is_some()
                    || original.exact_cost != Some(candidate_cost)
                {
                    return Err("experiment received invalid witness".into());
                }
                validation_ms += ms(validation.elapsed());
                // A result validated after the budget is recorded as late, not
                // credited as a one-second result by the report.
                cost = Some(candidate_cost);
                values = Some(candidate);
            }
        }
        let elapsed_ms = ms(begin.elapsed());
        // Avoid millions of duplicate random-sampler points, while preserving
        // every observed improvement and work/time checkpoint.
        if points
            .last()
            .is_none_or(|p| p.cost != cost || elapsed_ms - p.elapsed_ms >= 10.0)
            || terminal
        {
            points.push(Point {
                elapsed_ms,
                search_ms: ms(search_begin.elapsed()),
                work,
                cost,
                lower_bound: lower,
            });
        }
        if terminal {
            break;
        }
    }
    let elapsed_ms = ms(begin.elapsed());
    if points.last().is_none_or(|p| p.work != work) {
        points.push(Point {
            elapsed_ms,
            search_ms: ms(search_begin.elapsed()),
            work,
            cost,
            lower_bound: 0,
        });
    }
    let stats = match &search {
        Some(s) => serde_json::to_value(s.stats())?,
        None => {
            serde_json::json!({"candidate_attempts":work,"work_unit":"complete samples; not solver propagation units"})
        }
    };
    let selected_implementation_optimum =
        if matches!(saved.reference, Some(Reference::KernelSchedule { .. })) {
            values
                .as_ref()
                .map(|v| {
                    reference::implementation_schedule(
                        saved,
                        Some(v),
                        2_000_000,
                        Duration::from_secs(5),
                    )
                })
                .transpose()?
        } else {
            None
        };
    Ok(Record {
        instance: saved.name.clone(),
        family: saved.family.clone(),
        fingerprint,
        method: method.into(),
        seed,
        settings: settings.clone(),
        options,
        parameters: saved.parameters.clone(),
        variables: model.variables().len(),
        factors: model.factors().len(),
        model_bytes: model.retained_bytes(),
        construction_ms: Some(construction_ms),
        initialization_ms: Some(initialization_ms),
        validation_ms: Some(validation_ms),
        elapsed_ms,
        process_ms: 0.0,
        process_startup_ms: None,
        observed_rss_bytes: None,
        status,
        reason,
        reference: None,
        points,
        values,
        stats,
        selected_implementation_optimum,
        final_schedule_quality: None,
    })
}
fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn random_candidate(model: &Model, rng: &mut Random) -> LabResult<Vec<i64>> {
    let mut v: Vec<_> = model
        .variables()
        .iter()
        .map(|var| {
            var.domain
                .nth(u128::from(rng.next_u64()) % var.domain.cardinality())
                .ok_or("empty random domain")
        })
        .collect::<Result<_, _>>()?;
    let mut factors = Vec::new();
    fn flatten<'a>(factor: &'a Factor, output: &mut Vec<&'a Factor>) {
        output.push(factor);
        if let FactorKind::Fragment { factors } = &factor.kind {
            for f in factors {
                flatten(f, output);
            }
        }
    }
    for factor in model.factors() {
        flatten(factor, &mut factors);
    }
    // Uniform table-row proposals improve feasibility; they need not agree with
    // other factors and therefore always pass through whole-model validation.
    for f in &factors {
        if f.guards.iter().any(|g| v[g.variable.0] != g.value) {
            continue;
        }
        if let FactorKind::Constraint(Constraint::Table { variables, tuples }) = &f.kind {
            if !tuples.is_empty() {
                let row = &tuples[(rng.next_u64() as usize) % tuples.len()];
                for (var, value) in variables.iter().zip(row) {
                    v[var.0] = *value;
                }
            }
        }
    }
    for _ in 0..model.variables().len().max(1) {
        let previous = v.clone();
        for f in &factors {
            if f.guards.iter().any(|g| v[g.variable.0] != g.value) {
                continue;
            }
            if let FactorKind::Constraint(c) = &f.kind {
                match c {
                    Constraint::Arithmetic(a) => match a {
                        Arithmetic::Product {
                            left,
                            right,
                            product,
                        } => {
                            if let Some(x) = v[left.0].checked_mul(v[right.0]) {
                                v[product.0] = x;
                            }
                        }
                        Arithmetic::CeilDiv {
                            numerator,
                            denominator,
                            quotient,
                        } => {
                            if v[denominator.0] > 0 {
                                v[quotient.0] = i64::try_from(
                                    (v[numerator.0] as i128 + v[denominator.0] as i128 - 1)
                                        / (v[denominator.0] as i128),
                                )?;
                            }
                        }
                        Arithmetic::DivRem {
                            numerator,
                            denominator,
                            quotient,
                            remainder,
                        } => {
                            if v[denominator.0] > 0 {
                                v[quotient.0] = v[numerator.0] / v[denominator.0];
                                v[remainder.0] = v[numerator.0] % v[denominator.0];
                            }
                        }
                        Arithmetic::Minimum {
                            left,
                            right,
                            result,
                        } => v[result.0] = v[left.0].min(v[right.0]),
                        Arithmetic::Maximum {
                            left,
                            right,
                            result,
                        } => v[result.0] = v[left.0].max(v[right.0]),
                    },
                    Constraint::InDomain { variable, domain } if domain.is_singleton() => {
                        v[variable.0] = domain.min().unwrap()
                    }
                    Constraint::InactiveValue {
                        active,
                        variable,
                        inactive,
                    } if v[active.variable.0] != active.value => v[variable.0] = *inactive,
                    Constraint::Implies {
                        premise,
                        consequence,
                    } if v[premise.variable.0] == premise.value => {
                        v[consequence.variable.0] = consequence.value
                    }
                    Constraint::BoolAnd { output, inputs } => {
                        v[output.0] = i64::from(inputs.iter().all(|x| v[x.0] == 1))
                    }
                    Constraint::Schedule(SchedulingConstraint::Activity(a)) => {
                        if a.presence.is_none_or(|p| v[p.0] == 1) {
                            if let Some(end) = v[a.start.0].checked_add(v[a.duration.0]) {
                                v[a.end.0] = end;
                            }
                        }
                    }
                    Constraint::Schedule(SchedulingConstraint::ActivityCumulative {
                        completion,
                        activities,
                        ..
                    }) => {
                        let mut end = 0;
                        for r in activities {
                            let a = &r.activity;
                            if a.presence.is_none_or(|p| v[p.0] == 1) {
                                if let Some(e) = v[a.start.0].checked_add(v[a.duration.0]) {
                                    v[a.end.0] = e;
                                    end = end.max(e);
                                }
                            }
                        }
                        v[completion.0] = end;
                    }
                    _ => {}
                }
            }
        }
        if v == previous {
            break;
        }
    }
    Ok(v)
}

/// Child-process isolation includes construction, solver work and validation in
/// its watchdog. Offline schedule-reference work has a separately bounded grace.
pub fn supervised(
    executable: &Path,
    input: &Path,
    output: &Path,
    method: &str,
    seed: u64,
    settings: &Settings,
    reference: Option<ReferenceResult>,
) -> LabResult<Record> {
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let settings_path = output.with_extension("settings.json");
    fs::write(&settings_path, serde_json::to_vec(settings)?)?;
    let pending = output.with_extension("pending.json");
    let ready = output.with_extension("ready");
    if pending.exists() {
        fs::remove_file(&pending)?;
    }
    if ready.exists() {
        fs::remove_file(&ready)?;
    }
    let stderr_path = output.with_extension("stderr.txt");
    let stderr = fs::File::create(&stderr_path)?;
    let mut child = Command::new(executable)
        .args(["__study", "--input"])
        .arg(input)
        .arg("--out")
        .arg(&pending)
        .arg("--settings")
        .arg(&settings_path)
        .arg("--ready")
        .arg(&ready)
        .arg("--method")
        .arg(method)
        .arg("--seed")
        .arg(seed.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr))
        .spawn()?;
    let start = Instant::now();
    let mut sampled = Instant::now();
    let mut peak = None::<u64>;
    let mut watchdog = None;
    let mut began = None;
    let exit = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if sampled.elapsed() >= Duration::from_millis(100) {
            if let Some((rss, _)) = crate::runner::process_usage(child.id()) {
                peak = Some(peak.unwrap_or(0).max(rss));
                if rss > (settings.memory_mib as u64) * 1024 * 1024 {
                    watchdog = Some("process memory watchdog".to_string());
                }
            }
            sampled = Instant::now();
        }
        if began.is_none() && ready.exists() {
            began = Some(Instant::now());
        }
        if let Some(began) = began {
            if began.elapsed() > Duration::from_millis(settings.milliseconds + 7000) {
                watchdog =
                    Some("process time watchdog, including offline schedule reference".into());
            }
        } else if start.elapsed() > Duration::from_secs(120) {
            watchdog = Some("process startup watchdog before study readiness".into());
        }
        if watchdog.is_some() {
            child.kill()?;
            break child.wait()?;
        }
        std::thread::sleep(Duration::from_millis(2));
    };
    let mut record = if exit.success() && pending.exists() {
        serde_json::from_slice::<Record>(&fs::read(&pending)?)?
    } else {
        let saved = Instance::read(input)?;
        Record {
            instance: saved.name.clone(),
            family: saved.family.clone(),
            fingerprint: saved.fingerprint()?,
            method: method.into(),
            seed,
            settings: settings.clone(),
            options: options(method, seed, settings)?,
            parameters: saved.parameters.clone(),
            variables: saved.model.variables().len(),
            factors: saved.model.factors().len(),
            model_bytes: saved.model.retained_bytes(),
            construction_ms: None,
            initialization_ms: None,
            validation_ms: None,
            elapsed_ms: ms(start.elapsed()),
            process_ms: 0.0,
            process_startup_ms: None,
            observed_rss_bytes: None,
            status: if !ready.exists() {
                "startup-error".into()
            } else if watchdog.is_some() {
                "incomplete".into()
            } else {
                "error".into()
            },
            reason: Some(watchdog.unwrap_or_else(|| {
                format!(
                    "child {exit}: {}",
                    fs::read_to_string(&stderr_path).unwrap_or_default()
                )
            })),
            reference: None,
            points: vec![],
            values: None,
            stats: serde_json::Value::Null,
            selected_implementation_optimum: None,
            final_schedule_quality: None,
        }
    };
    record.reference = reference;
    if let Some(expected) = &record.reference {
        let contradiction = match expected.outcome {
            ReferenceOutcome::Optimal(opt) => {
                record.status == "infeasible"
                    || record
                        .points
                        .iter()
                        .any(|p| p.cost.is_some_and(|cost| cost < opt))
                    || (record.status == "optimal"
                        && record.points.last().and_then(|p| p.cost) != Some(opt))
            }
            ReferenceOutcome::Infeasible => {
                record.points.iter().any(|p| p.cost.is_some()) || record.status == "optimal"
            }
            ReferenceOutcome::Incomplete => false,
        };
        if contradiction {
            record.status = "error".into();
            record.reason = Some("independent reference contradicts a solver witness or terminal proof; retained for investigation".into());
        }
    }
    if let (
        Some(ReferenceResult {
            outcome: ReferenceOutcome::Optimal(opt),
            ..
        }),
        Some(ReferenceResult {
            outcome: ReferenceOutcome::Optimal(selected),
            ..
        }),
        Some(cost),
    ) = (
        &record.reference,
        &record.selected_implementation_optimum,
        record.points.last().and_then(|p| p.cost),
    ) {
        record.final_schedule_quality = Some(ScheduleQuality { witness_cost: cost,
            selected_implementation_optimum: *selected, global_optimum: *opt,
            schedule_ratio: (*selected != 0).then(|| cost as f64 / *selected as f64),
            implementation_ratio: (*opt != 0).then(|| *selected as f64 / *opt as f64),
            note: "Final returned witness; checkpoint timing remains in points. Zero optima use absolute costs instead of ratios.".into() });
    }
    record.process_ms = ms(start.elapsed());
    record.process_startup_ms = began.map(|at| ms(at.duration_since(start)));
    record.observed_rss_bytes = peak;
    fs::write(output, serde_json::to_vec_pretty(&record)?)?;
    fs::remove_file(settings_path)?;
    if ready.exists() {
        fs::remove_file(ready)?;
    }
    if pending.exists() {
        fs::remove_file(pending)?;
    }
    if fs::metadata(&stderr_path)?.len() == 0 {
        fs::remove_file(stderr_path)?;
    }
    Ok(record)
}

#[derive(Serialize)]
struct Summary {
    family: String,
    regime: String,
    variables: usize,
    method: String,
    runs: usize,
    errors: usize,
    unresolved_references: usize,
    checkpoints: Vec<Checkpoint>,
    median_target_ms: Option<f64>,
    median_target_work: Option<f64>,
    target_reached: usize,
    median_construction_ms: Option<f64>,
    median_initialization_ms: Option<f64>,
}
#[derive(Serialize)]
struct Checkpoint {
    budget_ms: u64,
    runs: usize,
    exact_references: usize,
    no_feasible: usize,
    feasible: usize,
    within_five_percent: usize,
    regret_p50: Option<f64>,
    regret_p95: Option<f64>,
    regret_max: Option<f64>,
    over_twenty_percent: usize,
}
fn percentile(values: &mut [f64], p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[((values.len() - 1) as f64 * p).ceil() as usize])
}
fn optimum(r: &Record) -> Option<u64> {
    match r.reference.as_ref()?.outcome {
        ReferenceOutcome::Optimal(c) => Some(c),
        _ => None,
    }
}
fn regret(cost: u64, opt: u64) -> f64 {
    if opt == 0 {
        (cost as f64) * 100.0
    } else {
        (cost as f64 / opt as f64 - 1.0) * 100.0
    }
}
pub fn report(records: &[Record], output: &Path) -> LabResult<()> {
    let mut groups = std::collections::BTreeMap::<(String, String, String), Vec<&Record>>::new();
    for r in records {
        groups
            .entry((
                r.family.clone(),
                serde_json::to_string(&r.parameters)?,
                r.method.clone(),
            ))
            .or_default()
            .push(r);
    }
    let mut rows = Vec::new();
    for ((family, regime, method), group) in groups {
        let variables = group[0].variables;
        let mut checkpoints = Vec::new();
        for budget_ms in [10, 30, 100, 300, 1000] {
            let eligible: Vec<_> = group
                .iter()
                .filter(|r| r.settings.milliseconds >= budget_ms)
                .collect();
            if eligible.is_empty() {
                continue;
            }
            let runs = eligible.len();
            let exact_references = eligible.iter().filter(|r| optimum(r).is_some()).count();
            let mut regrets = Vec::new();
            let mut feasible = 0;
            let mut within = 0;
            let mut severe = 0;
            for r in eligible {
                let cost = r
                    .points
                    .iter()
                    .rev()
                    .find(|p| p.elapsed_ms <= budget_ms as f64)
                    .and_then(|p| p.cost);
                if let Some(cost) = cost {
                    feasible += 1;
                    if let Some(opt) = optimum(r) {
                        let value = regret(cost, opt);
                        regrets.push(value);
                        within += usize::from(value <= 5.0 + 1e-9);
                        severe += usize::from(value > 20.0 + 1e-9);
                    }
                }
            }
            checkpoints.push(Checkpoint {
                budget_ms,
                runs,
                exact_references,
                no_feasible: runs - feasible,
                feasible,
                within_five_percent: within,
                regret_p50: percentile(&mut regrets, 0.5),
                regret_p95: percentile(&mut regrets, 0.95),
                regret_max: percentile(&mut regrets, 1.0),
                over_twenty_percent: severe,
            });
        }
        let mut times = Vec::new();
        let mut works = Vec::new();
        let mut construction: Vec<_> = group.iter().filter_map(|r| r.construction_ms).collect();
        let mut init: Vec<_> = group.iter().filter_map(|r| r.initialization_ms).collect();
        for r in &group {
            if let Some(opt) = optimum(r) {
                if let Some(p) = r.points.iter().find(|p| {
                    p.elapsed_ms <= r.settings.milliseconds as f64
                        && p.cost.is_some_and(|c| regret(c, opt) <= 5.0 + 1e-9)
                }) {
                    times.push(p.elapsed_ms);
                    works.push(p.work as f64);
                }
            }
        }
        rows.push(Summary {
            family,
            regime,
            variables,
            method,
            runs: group.len(),
            errors: group
                .iter()
                .filter(|r| matches!(r.status.as_str(), "error" | "startup-error"))
                .count(),
            unresolved_references: group.iter().filter(|r| optimum(r).is_none()).count(),
            checkpoints,
            target_reached: times.len(),
            median_target_ms: percentile(&mut times, 0.5),
            median_target_work: percentile(&mut works, 0.5),
            median_construction_ms: percentile(&mut construction, 0.5),
            median_initialization_ms: percentile(&mut init, 0.5),
        });
    }
    fs::create_dir_all(output)?;
    fs::write(
        output.join("summary.json"),
        serde_json::to_vec_pretty(&rows)?,
    )?;
    let mut text=String::from("# Neighborhood solver experiment\n\nTimes include reconstruction, initialization and independent validation. Reference work and process startup are excluded. Regret percentiles cover feasible runs with exact references; the feasible and target counts retain failures. Time-to-target medians are conditional on success and cannot establish speedups when success rates differ. Random work units are complete samples and are not comparable with solver work.\n\n| Family | Regime | Variables | Method | Runs | Reached 5% | Median target ms | Median target work | Errors | Unresolved refs |\n|---|---|---:|---|---:|---:|---:|---:|---:|---:|\n");
    for r in rows {
        text.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            r.family,
            r.regime,
            r.variables,
            r.method,
            r.runs,
            r.target_reached,
            r.median_target_ms.map_or("—".into(), |x| format!("{x:.3}")),
            r.median_target_work
                .map_or("—".into(), |x| format!("{x:.0}")),
            r.errors,
            r.unresolved_references
        ));
    }
    fs::write(output.join("report.md"), text)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reports_do_not_credit_late_witnesses_or_censor_no_feasible_runs() {
        let directory =
            std::env::temp_dir().join(format!("solver-lns-report-test-{}", std::process::id()));
        let mut record = Record {
            family: "test".into(),
            method: "lns".into(),
            reference: Some(ReferenceResult {
                outcome: ReferenceOutcome::Optimal(10),
                assignments: 1,
                elapsed_ms: 0.0,
                method: "test".into(),
                reason: None,
            }),
            points: vec![
                Point {
                    elapsed_ms: 0.1,
                    search_ms: 0.0,
                    work: 0,
                    cost: None,
                    lower_bound: 0,
                },
                Point {
                    elapsed_ms: 1000.1,
                    search_ms: 1000.0,
                    work: 10,
                    cost: Some(10),
                    lower_bound: 0,
                },
            ],
            ..Record::default()
        };
        record.settings = Settings::default();
        report(&[record], &directory).unwrap();
        let rows: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.join("summary.json")).unwrap()).unwrap();
        assert_eq!(rows[0]["target_reached"], 0);
        assert_eq!(rows[0]["checkpoints"][4]["feasible"], 0);
        assert_eq!(rows[0]["checkpoints"][4]["within_five_percent"], 0);
        fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn baseline_options_change_search_not_the_model_contract() {
        let settings = Settings::default();
        for name in ["greedy", "anneal", "joint", "lns"] {
            let o = options(name, 42, &settings).unwrap().unwrap();
            let Algorithm::Neighborhood(n) = o.algorithm else {
                panic!()
            };
            assert_eq!(n.seed, 42);
            assert_eq!(n.repair_work, settings.neighborhood.repair_work);
            assert_eq!(
                n.max_neighborhood_variables == 1,
                matches!(name, "greedy" | "anneal")
            );
            assert_eq!(n.exploration, matches!(name, "anneal" | "lns"));
        }
    }
}
