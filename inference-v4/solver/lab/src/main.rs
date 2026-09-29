use magnitude_solver_lab::{
    experiment,
    generators::{self, Instance, Parameters},
    reference::{self, ReferenceOutcome},
    reporting,
    runner::{self, RunSettings},
    LabResult,
};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

const HELP: &str = "magnitude-solver-lab <generate|verify|bench|compare|report> [--key value]\n\n\
generate --suite oracle-small --seeds 0..9 --out DIR\n\
generate --family chain --n 128 --d 4 --seed 0 --out DIR\n\
verify --input DIR --reference exhaustive|analytic|cp-sat --assignments 1000000 --seconds-per-case 60\n\
bench --suite structure-v1 --seeds 0..9 --policy dfs --seconds-per-case 60 --memory-mib 2048 --repeats 5 --out DIR\n\
bench --input DIR --policy dfs --out DIR\n\
compare --input DIR --policies dfs,best-first,no-cache,no-decompose,weak-bounds\n\
report --input DIR --out DIR\n\n\
search-study --input DIR --out DIR --methods exact,random,greedy,anneal,joint,lns --search-seeds 0,1,2,3,4 --milliseconds 1000\n\
study-report --input DIR/runs --out DIR\n\
Generator options: n,d,width,depth,repeat,capacity,horizon,outputs,input-length,overhead,element-bytes.\n\
Bench options: sample-work,warmups,repeats,seconds-per-case,memory-mib,stop-after-incomplete.\n\
All inputs are saved before solving. Timed-out runs remain incomplete. CP-SAT requires reference/requirements.txt.\n";

struct Args {
    command: String,
    values: BTreeMap<String, String>,
}
impl Args {
    fn read() -> LabResult<Self> {
        let mut iter = std::env::args().skip(1);
        let command = iter.next().unwrap_or_else(|| "help".into());
        let mut values = BTreeMap::new();
        while let Some(key) = iter.next() {
            let key = key
                .strip_prefix("--")
                .ok_or("options must use --key value")?
                .to_owned();
            let value = iter
                .next()
                .ok_or_else(|| format!("missing value for --{key}"))?;
            if values.insert(key.clone(), value).is_some() {
                return Err(format!("duplicate --{key}").into());
            }
        }
        Ok(Self { command, values })
    }
    fn text(&self, key: &str, default: &str) -> String {
        self.values
            .get(key)
            .cloned()
            .unwrap_or_else(|| default.into())
    }
    fn number<T: std::str::FromStr>(&self, key: &str, default: T) -> LabResult<T>
    where
        T::Err: std::error::Error + 'static,
    {
        match self.values.get(key) {
            Some(value) => Ok(value.parse()?),
            None => Ok(default),
        }
    }
    fn path(&self, key: &str) -> LabResult<PathBuf> {
        Ok(PathBuf::from(
            self.values
                .get(key)
                .ok_or_else(|| format!("--{key} is required"))?,
        ))
    }
    fn settings(&self) -> LabResult<RunSettings> {
        Ok(RunSettings {
            policy: self.text("policy", "dfs"),
            seconds: self.number("seconds-per-case", 60.0)?,
            memory_mib: self.number("memory-mib", 2048)?,
            sample_work: self.number("sample-work", 1000)?,
            repeat: self.number("repeat", 0)?,
        })
    }
    fn parameters(&self) -> LabResult<Parameters> {
        let p = Parameters::default();
        Ok(Parameters {
            n: self.number("n", p.n)?,
            d: self.number("d", p.d)?,
            width: self.number("width", p.width)?,
            depth: self.number("depth", p.depth)?,
            repeat: self.number("repeat", p.repeat)?,
            capacity: self.number("capacity", p.capacity)?,
            horizon: self.number("horizon", p.horizon)?,
            outputs: self.number("outputs", p.outputs)?,
            input_length: self.number("input-length", p.input_length)?,
            overhead: self.number("overhead", p.overhead)?,
            element_bytes: self.number("element-bytes", p.element_bytes)?,
        })
    }
    fn seeds(&self) -> LabResult<Vec<u64>> {
        let default = if self.values.contains_key("family") {
            "0"
        } else {
            "0..9"
        };
        let text = self.text("seeds", &self.text("seed", default));
        if let Some((start, end)) = text.split_once("..") {
            let start: u64 = start.parse()?;
            let end: u64 = end.parse()?;
            if start > end || end - start > 1000 {
                return Err("seed range must be ascending and at most 1001 seeds".into());
            }
            Ok((start..=end).collect())
        } else {
            text.split(',').map(|seed| Ok(seed.parse()?)).collect()
        }
    }
}

fn main() {
    if let Err(error) = execute() {
        eprintln!("solver-lab: {error}");
        std::process::exit(1);
    }
}
fn execute() -> LabResult<()> {
    let args = Args::read()?;
    match args.command.as_str() {
        "help" | "--help" | "-h" => {
            print!("{HELP}");
            Ok(())
        }
        "generate" => {
            let paths = generate(&args, &args.path("out")?)?;
            println!("saved {} instances", paths.len());
            Ok(())
        }
        "verify" => verify(&args),
        "bench" => bench(&args),
        "compare" => compare(&args),
        "report" => reporting::report(&args.path("input")?, &args.path("out")?),
        "search-study" => study(&args),
        "study-report" => {
            let mut records = Vec::new();
            for entry in fs::read_dir(args.path("input")?)? {
                let path = entry?.path();
                if path
                    .file_name()
                    .and_then(|x| x.to_str())
                    .is_some_and(|x| x.ends_with(".study.json"))
                {
                    records.push(serde_json::from_slice::<experiment::Record>(&fs::read(
                        path,
                    )?)?);
                }
            }
            experiment::report(&records, &args.path("out")?)
        }
        "__study" => {
            if let Some(ready) = args.values.get("ready") {
                fs::write(ready, "started")?;
            }
            let instance = Instance::read(&args.path("input")?)?;
            let settings: experiment::Settings =
                serde_json::from_slice(&fs::read(args.path("settings")?)?)?;
            let record = experiment::solve(
                &instance,
                &args.text("method", "lns"),
                args.number("seed", 0_u64)?,
                &settings,
            )?;
            fs::write(args.path("out")?, serde_json::to_vec_pretty(&record)?)?;
            Ok(())
        }
        "__solve" => {
            let instance = Instance::read(&args.path("input")?)?;
            let result = runner::solve(&instance, &args.settings()?)?;
            fs::write(args.path("out")?, serde_json::to_vec_pretty(&result)?)?;
            Ok(())
        }
        _ => Err(format!("unknown command {}\n{HELP}", args.command).into()),
    }
}

fn study(args: &Args) -> LabResult<()> {
    let output = args.path("out")?;
    fs::create_dir_all(&output)?;
    let paths = inputs(&args.path("input")?)?;
    if paths.is_empty() {
        return Err("no saved inputs".into());
    }
    let search_seeds: Vec<u64> = args
        .text("search-seeds", "0,1,2,3,4")
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let methods: Vec<_> = args
        .text("methods", &experiment::METHODS.join(","))
        .split(',')
        .map(str::to_owned)
        .collect();
    let mut settings = experiment::Settings {
        milliseconds: args.number("milliseconds", 1000)?,
        sample_work: args.number("sample-work", 100)?,
        memory_mib: args.number("memory-mib", 512)?,
        ..experiment::Settings::default()
    };
    settings.neighborhood.repair_work =
        args.number("repair-work", settings.neighborhood.repair_work)?;
    settings.neighborhood.max_neighborhood_variables = args.number(
        "neighborhood-variables",
        settings.neighborhood.max_neighborhood_variables,
    )?;
    settings.neighborhood.population_size =
        args.number("population", settings.neighborhood.population_size)?;
    settings.neighborhood.restart_after =
        args.number("restart-after", settings.neighborhood.restart_after)?;
    for method in &methods {
        experiment::options(method, 0, &settings)?;
    }
    let command = |program: &str, flags: &[&str]| {
        std::process::Command::new(program)
            .args(flags)
            .output()
            .ok()
            .filter(|x| x.status.success())
            .map(|x| String::from_utf8_lossy(&x.stdout).trim().to_owned())
    };
    let executable = std::env::current_exe()?;
    let executable_text = executable.to_string_lossy();
    fs::write(
        output.join("cohort.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "command": std::env::args().collect::<Vec<_>>(), "methods":methods, "search_seeds":search_seeds,
            "settings":settings,"machine":command("uname", &["-a"]),"rustc":command("rustc", &["--version"]),
            "revision":command("git", &["rev-parse","HEAD"]),"source_state":command("git", &["status","--porcelain"]),
            "profile":if cfg!(debug_assertions){"debug"}else{"release"},
            "binary_sha256":command("shasum", &["-a","256",executable_text.as_ref()]), "timing":"reconstructed instance, initialization, search and checked witnesses; offline references and process startup excluded"
        }))?,
    )?;
    let mut records = Vec::new();
    for path in paths {
        let instance = Instance::read(&path)?;
        let saved = instance.save(&output.join("instances"))?;
        let reference = match reference::analytic(&instance)? {
            Some(r) => r,
            None => reference::exhaustive(
                &instance,
                args.number("reference-assignments", 2_000_000)?,
                Duration::from_secs_f64(args.number("reference-seconds", 5.0)?),
            )?,
        };
        fs::create_dir_all(output.join("references"))?;
        fs::write(
            output
                .join("references")
                .join(format!("{}.reference.json", instance.name)),
            serde_json::to_vec_pretty(&reference)?,
        )?;
        for &seed in &search_seeds {
            // Rotate method order by seed so a fixed method does not always
            // get the same thermal/order position. Children remain sequential.
            for offset in 0..methods.len() {
                let method = &methods[(offset + seed as usize) % methods.len()];
                let file = output
                    .join("runs")
                    .join(format!("{}-{method}-s{seed}.study.json", instance.name));
                let record = experiment::supervised(
                    &std::env::current_exe()?,
                    &saved,
                    &file,
                    method,
                    seed,
                    &settings,
                    Some(reference.clone()),
                )?;
                println!(
                    "{} {method} seed {seed}: {} cost {:?} {:.2}ms",
                    instance.name,
                    record.status,
                    record.points.last().and_then(|p| p.cost),
                    record.elapsed_ms
                );
                records.push(record);
                experiment::report(&records, &output)?;
            }
        }
    }
    Ok(())
}

fn generate(args: &Args, directory: &Path) -> LabResult<Vec<PathBuf>> {
    let seeds = args.seeds()?;
    let instances = if let Some(family) = args.values.get("family") {
        seeds
            .into_iter()
            .map(|seed| generators::generate(family, args.parameters()?, seed))
            .collect::<LabResult<Vec<_>>>()?
    } else {
        generators::suite(&args.text("suite", "oracle-small"), &seeds)?
    };
    instances
        .into_iter()
        .map(|instance| instance.save(directory))
        .collect()
}

fn inputs(path: &Path) -> LabResult<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_owned()]);
    }
    let mut files = Vec::new();
    for entry in fs::read_dir(path)? {
        let path = entry?.path();
        if path.is_dir() {
            files.extend(inputs(&path)?);
        } else if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".instance.json"))
        {
            files.push(path);
        }
    }
    files.sort();
    files.dedup();
    Ok(files)
}

#[derive(Serialize)]
struct Verification {
    instance: String,
    reference: reference::ReferenceResult,
    solver_status: String,
    solver_cost: Option<u64>,
    agreement: Option<bool>,
}

fn verification_agreement(
    reference_kind: &str,
    expected: &reference::ReferenceResult,
    actual: &runner::RunRecord,
    known_coverage_gap: bool,
) -> Option<bool> {
    match expected.outcome {
        ReferenceOutcome::Optimal(cost) if actual.status == "optimal" => {
            Some(actual.cost == Some(cost))
        }
        ReferenceOutcome::Infeasible if actual.status == "infeasible" => Some(true),
        ReferenceOutcome::Optimal(_) if actual.status == "infeasible" => Some(false),
        ReferenceOutcome::Infeasible if actual.status == "optimal" => Some(false),
        ReferenceOutcome::Incomplete
            if known_coverage_gap
                || (reference_kind == "exhaustive"
                    && expected.reason.as_deref() == Some("unresolved domain coverage")) =>
        {
            Some(actual.status == "incomplete")
        }
        // CP-SAT conservatively stops for any declared missing coverage, even
        // behind an inactive guard. That is not a proof that the solver must
        // remain incomplete: it can exclude the affected region soundly.
        _ => None,
    }
}

fn verify(args: &Args) -> LabResult<()> {
    let input = args.path("input")?;
    let paths = inputs(&input)?;
    if paths.is_empty() {
        return Err("no input instances".into());
    }
    let reference_kind = args.text("reference", "exhaustive");
    let seconds: f64 = args.number("seconds-per-case", 60.0)?;
    let limit = args.number("assignments", 1_000_000)?;
    let mut records = Vec::new();
    let mut disagreements = 0;
    for path in paths {
        let instance = Instance::read(&path)?;
        let expected = match reference_kind.as_str() {
            "exhaustive" => {
                reference::exhaustive(&instance, limit, Duration::from_secs_f64(seconds))?
            }
            "analytic" => reference::analytic(&instance)?
                .ok_or_else(|| format!("{} has no analytic reference", instance.name))?,
            "cp-sat" => cp_sat(&path, seconds)?,
            _ => return Err("reference must be exhaustive,analytic,cp-sat".into()),
        };
        let actual = runner::solve(&instance, &args.settings()?)?;
        let agreement = verification_agreement(
            &reference_kind,
            &expected,
            &actual,
            matches!(instance.reference, Some(generators::Reference::Incomplete)),
        );
        if agreement == Some(false) {
            disagreements += 1;
        }
        println!(
            "{}: {:?}; solver {}; agreement {:?}",
            instance.name, expected.outcome, actual.status, agreement
        );
        records.push(Verification {
            instance: instance.name,
            reference: expected,
            solver_status: actual.status,
            solver_cost: actual.cost,
            agreement,
        });
    }
    let output = args
        .values
        .get("out")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            if input.is_dir() {
                input.join("verification.json")
            } else {
                input.with_extension("verification.json")
            }
        });
    fs::write(output, serde_json::to_vec_pretty(&records)?)?;
    if disagreements > 0 {
        return Err(format!("{disagreements} exact-reference disagreements").into());
    }
    Ok(())
}

fn cp_sat(input: &Path, seconds: f64) -> LabResult<reference::ReferenceResult> {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/cp_sat.py");
    let python = std::env::var("SOLVER_LAB_PYTHON").unwrap_or_else(|_| "python3".into());
    let output = std::process::Command::new(python)
        .arg(script)
        .arg(input)
        .arg("--seconds")
        .arg(seconds.to_string())
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "CP-SAT reference failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    let raw: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let result: reference::ReferenceResult = serde_json::from_value(raw.clone())?;
    if let ReferenceOutcome::Optimal(cost) = result.outcome {
        let witness: Vec<i64> = serde_json::from_value(
            raw.get("witness")
                .ok_or("CP-SAT optimum missing witness")?
                .clone(),
        )?;
        let instance = Instance::read(input)?;
        let independent = reference::evaluate(&instance.model, &witness)?;
        if !independent.feasible || independent.unresolved || independent.cost != cost {
            return Err("CP-SAT witness failed independent interpretation".into());
        }
        let checked = instance.model.validate_assignment(&witness)?;
        if checked.infeasible || checked.unresolved.is_some() || checked.exact_cost != Some(cost) {
            return Err("CP-SAT witness failed original-model validation".into());
        }
    }
    Ok(result)
}

fn bench(args: &Args) -> LabResult<()> {
    let output = args.path("out")?;
    fs::create_dir_all(&output)?;
    let paths = match args.values.get("input") {
        Some(input) => {
            let paths = inputs(Path::new(input))?;
            let mut copied = Vec::new();
            for path in paths {
                copied.push(Instance::read(&path)?.save(&output.join("instances"))?);
            }
            copied
        }
        None => generate(args, &output.join("instances"))?,
    };
    run_cases(args, &paths, &output, &[args.text("policy", "dfs")])
}

fn compare(args: &Args) -> LabResult<()> {
    let input = args.path("input")?;
    let paths = inputs(&input)?;
    let output = args
        .values
        .get("out")
        .map(PathBuf::from)
        .unwrap_or_else(|| input.join("comparison"));
    let policies: Vec<_> = args
        .text(
            "policies",
            "dfs,best-first,no-cache,no-decompose,weak-bounds",
        )
        .split(',')
        .map(str::to_owned)
        .collect();
    run_cases(args, &paths, &output, &policies)
}

fn run_cases(args: &Args, paths: &[PathBuf], output: &Path, policies: &[String]) -> LabResult<()> {
    if paths.is_empty() {
        return Err("no benchmark instances".into());
    }
    let executable = std::env::current_exe()?;
    let repeats = args.number("repeats", 5_usize)?;
    let warmups = args.number("warmups", 1_usize)?;
    if repeats == 0 {
        return Err("repeats must be positive".into());
    }
    // Mixed saved inputs are not a monotone sweep. Default to running every case.
    let stop_after = args.number("stop-after-incomplete", 0_usize)?;
    let mut skipped = Vec::new();
    for policy in policies {
        runner::options(policy)?;
        let mut exhausted: BTreeMap<String, usize> = BTreeMap::new();
        for path in paths {
            let instance = Instance::read(path)?;
            // A whole family is censored after repeated envelope exhaustion; saved instances remain available.
            if stop_after > 0 && exhausted.get(&instance.family).copied().unwrap_or(0) >= stop_after
            {
                skipped.push(serde_json::json!({"instance":instance.name,"policy":policy,"reason":"family resource envelope repeatedly exhausted"}));
                continue;
            }
            let mut settings = args.settings()?;
            settings.policy = policy.clone();
            for warmup in 0..warmups {
                settings.repeat = warmup;
                let file = output
                    .join("warmup")
                    .join(format!("{}-{policy}-{warmup}.warmup.json", instance.name));
                runner::supervised(&executable, path, &file, &settings)?;
            }
            let mut incomplete = false;
            for repeat in 0..repeats {
                settings.repeat = repeat;
                let file = output
                    .join("runs")
                    .join(format!("{}-{policy}-{repeat}.run.json", instance.name));
                let record = runner::supervised(&executable, path, &file, &settings)?;
                println!(
                    "{} {policy} {repeat}: {} {:.3} ms",
                    instance.name, record.status, record.solve_ms
                );
                incomplete |= record.status == "incomplete";
            }
            let count = exhausted.entry(instance.family.clone()).or_default();
            *count = if incomplete { *count + 1 } else { 0 };
        }
    }
    fs::create_dir_all(output)?;
    fs::write(
        output.join("skipped.json"),
        serde_json::to_vec_pretty(&skipped)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conservative_reference_incompleteness_cannot_disprove_an_optimum() {
        let reference = reference::ReferenceResult {
            outcome: ReferenceOutcome::Incomplete,
            assignments: 0,
            elapsed_ms: 0.0,
            method: "CP-SAT".into(),
            reason: Some("unresolved domain coverage".into()),
        };
        let completed = runner::RunRecord {
            status: "optimal".into(),
            cost: Some(0),
            ..runner::RunRecord::default()
        };
        assert_eq!(
            verification_agreement("cp-sat", &reference, &completed, false),
            None
        );
        // Exhaustive interpretation establishes whether missing coverage is
        // still competitive; a fixture may also require it explicitly.
        assert_eq!(
            verification_agreement("exhaustive", &reference, &completed, false),
            Some(false)
        );
        assert_eq!(
            verification_agreement("cp-sat", &reference, &completed, true),
            Some(false)
        );
    }
}
