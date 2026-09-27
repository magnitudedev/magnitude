use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use serde_json::Value;

use crate::assets::{Profile, ReferenceBuildManifest, UpstreamTarget, UpstreamTargetKind};
use crate::model::{CaseDefinition, CommandSpec, DecoderKind, StdinKind};

#[derive(Clone, Debug)]
pub struct ResolvedEngine {
    pub backend: String,
    pub backend_device: Option<String>,
    pub threads: u32,
    pub reference_gpu_layers: i32,
    pub candidate_gpu_layers: u32,
    pub flash_attention: String,
    pub cpu_strict: bool,
    pub threadpool_poll: u32,
}

impl ResolvedEngine {
    pub fn resolve(profile: &Profile, settings: &BTreeMap<String, String>) -> anyhow::Result<Self> {
        let threads = match profile.engine.threads.as_str() {
            "physical-cores" => settings
                .get("physical_cores")
                .context("profile requests physical-cores; provide --set physical_cores=N from controlled host inventory")?
                .parse()?,
            value => value.parse().with_context(|| format!("invalid profile thread policy {value}"))?,
        };
        if threads == 0 {
            bail!("resolved engine thread count must be positive");
        }
        let (reference_gpu_layers, candidate_gpu_layers) = match profile.engine.gpu_layers.as_str()
        {
            // Pinned llama.cpp common arguments use -1 for automatic placement
            // and -2 for every supported layer. Keep those policies distinct in
            // both the invocation and the evidence contract.
            "all" => (-2, u32::MAX),
            value => {
                let parsed = value
                    .parse::<u32>()
                    .with_context(|| format!("invalid gpu_layers {value}"))?;
                (
                    i32::try_from(parsed).context("reference gpu_layers exceeds i32")?,
                    parsed,
                )
            }
        };
        if !matches!(
            profile.engine.flash_attention.as_str(),
            "auto" | "on" | "off"
        ) {
            bail!(
                "invalid flash attention policy {}",
                profile.engine.flash_attention
            );
        }
        let backend = if profile.engine.backend == "auto" {
            settings
                .get("backend")
                .cloned()
                .unwrap_or_else(|| "auto".to_owned())
        } else {
            profile.engine.backend.clone()
        };
        Ok(Self {
            backend,
            backend_device: settings.get("backend_device").cloned(),
            threads,
            reference_gpu_layers,
            candidate_gpu_layers,
            flash_attention: profile.engine.flash_attention.clone(),
            cpu_strict: settings
                .get("cpu_strict")
                .is_some_and(|value| value == "true" || value == "1"),
            threadpool_poll: settings
                .get("threadpool_poll")
                .map(|value| value.parse())
                .transpose()?
                .unwrap_or(50),
        })
    }
}

pub fn upstream_command(
    target: &UpstreamTarget,
    manifest: &ReferenceBuildManifest,
    case: &CaseDefinition,
    engine: &ResolvedEngine,
    model_paths: &BTreeMap<String, PathBuf>,
    fixture_paths: &BTreeMap<String, PathBuf>,
    repetition_override: Option<u64>,
) -> anyhow::Result<CommandSpec> {
    if target.kind == UpstreamTargetKind::CtestSuite && case.operation == "baseline.ctest" {
        return ctest_command(target, manifest);
    }
    let artifact_id = target
        .artifacts
        .first()
        .context("upstream target has no executable artifact")?;
    let program = manifest.artifact(artifact_id)?.path.clone();
    let timeout_seconds = if case.category == crate::model::Category::Performance {
        3600
    } else {
        600
    };
    let base = |args: Vec<String>, decoder: DecoderKind, stdin: StdinKind| CommandSpec {
        program: program.clone(),
        args,
        cwd: Some(manifest.build_directory.clone()),
        env: BTreeMap::new(),
        clear_env: false,
        stdin,
        decoder,
        timeout_seconds: Some(timeout_seconds),
        max_stdout_bytes: Some(64 * 1024 * 1024),
        max_stderr_bytes: Some(8 * 1024 * 1024),
    };
    match target.id.as_str() {
        "llama-bench" => Ok(base(
            llama_bench_args(case, engine, model_paths, repetition_override)?,
            DecoderKind::LlamaBenchJson,
            StdinKind::None,
        )),
        "llama-batched-bench" => Ok(base(
            batched_bench_args(case, engine, model_paths)?,
            DecoderKind::BatchedBenchJsonl,
            StdinKind::None,
        )),
        "llama-perplexity" => Ok(base(
            perplexity_args(case, engine, model_paths, fixture_paths)?,
            DecoderKind::PerplexityText,
            StdinKind::None,
        )),
        "backend-ops-perf" if case.operation == "backend-ops.perf" => Ok(base(
            backend_ops_args(case, engine)?,
            DecoderKind::BackendOpsSql,
            StdinKind::None,
        )),
        "oracle" => Ok(base(
            Vec::new(),
            DecoderKind::ProbeJsonl,
            StdinKind::ProbeJsonl,
        )),
        other => bail!("no pinned command adapter exists for upstream target {other}"),
    }
}

fn ctest_command(
    target: &UpstreamTarget,
    manifest: &ReferenceBuildManifest,
) -> anyhow::Result<CommandSpec> {
    let args = ctest_selection_args(target, manifest, false)?;
    Ok(CommandSpec {
        program: "ctest".into(),
        args,
        cwd: Some(manifest.build_directory.clone()),
        env: BTreeMap::new(),
        clear_env: false,
        stdin: StdinKind::None,
        decoder: DecoderKind::ExitStatus,
        timeout_seconds: Some(1800),
        max_stdout_bytes: Some(64 * 1024 * 1024),
        max_stderr_bytes: Some(8 * 1024 * 1024),
    })
}

pub fn ctest_inventory_command(
    target: &UpstreamTarget,
    manifest: &ReferenceBuildManifest,
) -> anyhow::Result<CommandSpec> {
    Ok(CommandSpec {
        program: "ctest".into(),
        args: ctest_selection_args(target, manifest, true)?,
        cwd: Some(manifest.build_directory.clone()),
        env: BTreeMap::new(),
        clear_env: false,
        stdin: StdinKind::None,
        decoder: DecoderKind::Json,
        timeout_seconds: Some(120),
        max_stdout_bytes: Some(16 * 1024 * 1024),
        max_stderr_bytes: Some(2 * 1024 * 1024),
    })
}

fn ctest_selection_args(
    target: &UpstreamTarget,
    manifest: &ReferenceBuildManifest,
    show_only: bool,
) -> anyhow::Result<Vec<String>> {
    if target.ctest_names.is_empty() {
        bail!("CTest target {} declares no test names", target.id);
    }
    let names = target
        .ctest_names
        .iter()
        .map(|name| regex::escape(name))
        .collect::<Vec<_>>()
        .join("|");
    let mut args = vec![
        "--test-dir".to_owned(),
        manifest.build_directory.display().to_string(),
        "--build-config".to_owned(),
        manifest.build_type.clone(),
        "--output-on-failure".to_owned(),
        "--no-tests=error".to_owned(),
        "-R".to_owned(),
        format!("^({names})$"),
    ];
    if show_only {
        args.push("--show-only=json-v1".to_owned());
    }
    if !target.ctest_setup_names.is_empty() {
        let setup = target
            .ctest_setup_names
            .iter()
            .map(|name| regex::escape(name))
            .collect::<Vec<_>>()
            .join("|");
        args.extend(["-FS".to_owned(), format!("^({setup})$")]);
    }
    Ok(args)
}

fn llama_bench_args(
    case: &CaseDefinition,
    engine: &ResolvedEngine,
    models: &BTreeMap<String, PathBuf>,
    repetition_override: Option<u64>,
) -> anyhow::Result<Vec<String>> {
    let model = required_model(case, models)?;
    let prompt = input_u64(case, "prompt_tokens").unwrap_or(0);
    let generation = input_u64(case, "generation_tokens").unwrap_or(0);
    let depth = input_u64(case, "context_depth").unwrap_or(0);
    if prompt + generation == 0 {
        bail!("llama-bench case must declare non-zero prompt or generation tokens");
    }
    let batch = required_input_u64(case, "batch_tokens")?;
    let ubatch = required_input_u64(case, "micro_batch_tokens")?;
    let repetitions = repetition_override.unwrap_or(required_input_u64(case, "repetitions")?);
    let mut args = vec![
        "-m".to_owned(),
        model.display().to_string(),
        "-d".to_owned(),
        depth.to_string(),
        "-t".to_owned(),
        engine.threads.to_string(),
        "-b".to_owned(),
        batch.to_string(),
        "-ub".to_owned(),
        ubatch.to_string(),
        "-r".to_owned(),
        repetitions.to_string(),
        "-ngl".to_owned(),
        engine.reference_gpu_layers.to_string(),
        "-fa".to_owned(),
        engine.flash_attention.clone(),
        "--cpu-strict".to_owned(),
        if engine.cpu_strict { "1" } else { "0" }.to_owned(),
        "--poll".to_owned(),
        engine.threadpool_poll.to_string(),
        "-ctk".to_owned(),
        "f16".to_owned(),
        "-ctv".to_owned(),
        "f16".to_owned(),
        "-o".to_owned(),
        "json".to_owned(),
    ];
    if prompt > 0 && generation > 0 {
        args.extend(["-pg".to_owned(), format!("{prompt},{generation}")]);
    } else {
        args.extend([
            "-p".to_owned(),
            prompt.to_string(),
            "-n".to_owned(),
            generation.to_string(),
        ]);
    }
    if !input_bool(case, "warmup").unwrap_or(true) {
        args.push("--no-warmup".to_owned());
    }
    Ok(args)
}

fn batched_bench_args(
    case: &CaseDefinition,
    engine: &ResolvedEngine,
    models: &BTreeMap<String, PathBuf>,
) -> anyhow::Result<Vec<String>> {
    let model = required_model(case, models)?;
    let context = required_input_u64(case, "context_tokens")?;
    let batch = required_input_u64(case, "batch_tokens")?;
    let ubatch = required_input_u64(case, "micro_batch_tokens")?;
    let prompt = required_input_u64(case, "prompt_tokens")?;
    let generation = required_input_u64(case, "generation_tokens_per_sequence")?;
    let parallel = required_input_u64(case, "parallel_sequences")?;
    let mut args = vec![
        "-m".to_owned(),
        model.display().to_string(),
        "-c".to_owned(),
        context.to_string(),
        "-b".to_owned(),
        batch.to_string(),
        "-ub".to_owned(),
        ubatch.to_string(),
        "-t".to_owned(),
        engine.threads.to_string(),
        "-tb".to_owned(),
        engine.threads.to_string(),
        "-ngl".to_owned(),
        engine.reference_gpu_layers.to_string(),
        "-fa".to_owned(),
        engine.flash_attention.clone(),
        "-npp".to_owned(),
        prompt.to_string(),
        "-ntg".to_owned(),
        generation.to_string(),
        "-npl".to_owned(),
        parallel.to_string(),
        "--output-format".to_owned(),
        "jsonl".to_owned(),
    ];
    if input_bool(case, "shared_prompt") == Some(true) {
        args.push("-pps".to_owned());
    }
    if input_bool(case, "separate_generation") == Some(true) {
        args.push("-tgs".to_owned());
    }
    args.push(if input_bool(case, "kv_unified").unwrap_or(false) {
        "-kvu".to_owned()
    } else {
        "-no-kvu".to_owned()
    });
    Ok(args)
}

fn perplexity_args(
    case: &CaseDefinition,
    engine: &ResolvedEngine,
    models: &BTreeMap<String, PathBuf>,
    fixtures: &BTreeMap<String, PathBuf>,
) -> anyhow::Result<Vec<String>> {
    let model = required_model(case, models)?;
    let corpus_key = case
        .inputs
        .get("corpus")
        .and_then(Value::as_str)
        .context("perplexity case is missing corpus")?;
    let corpus = fixtures
        .get(corpus_key)
        .with_context(|| format!("unresolved corpus fixture {corpus_key}"))?;
    Ok(vec![
        "-m".to_owned(),
        model.display().to_string(),
        "-f".to_owned(),
        corpus.display().to_string(),
        "-c".to_owned(),
        required_input_u64(case, "context_tokens")?.to_string(),
        "-b".to_owned(),
        required_input_u64(case, "batch_tokens")?.to_string(),
        "-ub".to_owned(),
        required_input_u64(case, "micro_batch_tokens")?.to_string(),
        "--ppl-stride".to_owned(),
        required_input_u64(case, "stride_tokens")?.to_string(),
        "-t".to_owned(),
        engine.threads.to_string(),
        "-ngl".to_owned(),
        engine.reference_gpu_layers.to_string(),
        "-fa".to_owned(),
        engine.flash_attention.clone(),
    ])
}

fn backend_ops_args(
    case: &CaseDefinition,
    _engine: &ResolvedEngine,
) -> anyhow::Result<Vec<String>> {
    let mode = case
        .inputs
        .get("mode")
        .and_then(Value::as_str)
        .context("backend-ops case is missing mode")?;
    if mode != "perf" {
        bail!("backend-ops performance adapter only accepts mode=perf");
    }
    let filter = case
        .inputs
        .get("operations")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .map(|value| value.as_str().context("backend operation must be a string"))
                .collect::<anyhow::Result<Vec<_>>>()
        })
        .transpose()?
        .unwrap_or_default();
    if filter.is_empty() {
        bail!("backend-ops perf requires an explicit non-empty operations list");
    }
    let device = case.inputs.get("backend_device").and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .context("backend-ops qualification requires an exact --set backend_device=<upstream device name>")?;
    Ok(vec![
        "perf".to_owned(),
        "-o".to_owned(),
        filter.join(","),
        "-b".to_owned(),
        device.to_owned(),
        "--output".to_owned(),
        "sql".to_owned(),
    ])
}

fn required_model<'a>(
    case: &CaseDefinition,
    models: &'a BTreeMap<String, PathBuf>,
) -> anyhow::Result<&'a Path> {
    let id = case
        .inputs
        .get("model_id")
        .and_then(Value::as_str)
        .or_else(|| {
            case.requirements
                .model
                .as_ref()?
                .ids
                .first()
                .map(String::as_str)
        })
        .context("case does not select a concrete model id")?;
    models
        .get(id)
        .map(PathBuf::as_path)
        .with_context(|| format!("model {id} has no resolved primary artifact"))
}

fn required_input_u64(case: &CaseDefinition, name: &str) -> anyhow::Result<u64> {
    input_u64(case, name)
        .with_context(|| format!("case {} is missing unsigned integer input {name}", case.id))
}

fn input_u64(case: &CaseDefinition, name: &str) -> Option<u64> {
    case.inputs.get(name).and_then(Value::as_u64)
}

fn input_bool(case: &CaseDefinition, name: &str) -> Option<bool> {
    case.inputs.get(name).and_then(Value::as_bool)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CaseDefinition;

    fn load_case(relative: &str) -> CaseDefinition {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../parity/cases")
            .join(relative);
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    }

    #[test]
    fn llama_bench_combined_uses_pg_not_cartesian_sweep() {
        let mut case = load_case("performance/llama-bench/prompt-generation-128-32.json");
        case.inputs
            .insert("model_id".to_owned(), serde_json::json!("stories15m-q4-0"));
        let engine = ResolvedEngine {
            backend: "metal".to_owned(),
            backend_device: None,
            threads: 4,
            reference_gpu_layers: -2,
            candidate_gpu_layers: u32::MAX,
            flash_attention: "off".to_owned(),
            cpu_strict: false,
            threadpool_poll: 50,
        };
        let models = BTreeMap::from([("stories15m-q4-0".to_owned(), PathBuf::from("model.gguf"))]);
        let args = llama_bench_args(&case, &engine, &models, Some(1)).unwrap();
        assert!(args.windows(2).any(|pair| pair == ["-pg", "128,32"]));
        assert!(args.windows(2).any(|pair| pair == ["-ngl", "-2"]));
        assert!(!args.iter().any(|value| value == "--prompt-tokens"));
    }

    #[test]
    fn batched_bench_uses_pinned_cli_names() {
        let mut case = load_case("performance/batched-bench/two-sequence-independent-prompts.json");
        case.inputs
            .insert("model_id".to_owned(), serde_json::json!("stories15m-q4-0"));
        let engine = ResolvedEngine {
            backend: "cpu".to_owned(),
            backend_device: None,
            threads: 4,
            reference_gpu_layers: 0,
            candidate_gpu_layers: 0,
            flash_attention: "off".to_owned(),
            cpu_strict: false,
            threadpool_poll: 50,
        };
        let models = BTreeMap::from([("stories15m-q4-0".to_owned(), PathBuf::from("model.gguf"))]);
        let args = batched_bench_args(&case, &engine, &models).unwrap();
        assert!(args.windows(2).any(|pair| pair == ["-npp", "64"]));
        assert!(args.windows(2).any(|pair| pair == ["-npl", "2"]));
        assert!(!args.iter().any(|value| value == "-pps"));
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--output-format", "jsonl"])
        );
    }
}
