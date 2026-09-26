use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use benchmark_runner::{
    BenchmarkRun, BenchmarkRunConfig, BenchmarkRunner, ProgressEvent, TargetConfig, TargetKind,
    compare_evidence, comparison_markdown, validate_assets,
};
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Debug, Parser)]
#[command(
    name = "benchmark-runner",
    about = "Run controlled inference endpoint benchmarks"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Validate {
        #[arg(long, default_value = "benchmark")]
        root: PathBuf,
    },
    Run {
        #[arg(long, default_value = "benchmark")]
        root: PathBuf,
        #[arg(long)]
        profile: String,
        #[arg(long)]
        endpoint: String,
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "target")]
        name: String,
        #[arg(long, value_enum, default_value_t = KindArg::Generic)]
        kind: KindArg,
        #[arg(long)]
        api_key_env: Option<String>,
        #[arg(long)]
        model_sha256: Option<String>,
        #[arg(long)]
        process_id: Option<u32>,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        run_id: Option<String>,
        #[arg(long)]
        controlled_host: bool,
        #[arg(long)]
        exclusive_device: bool,
        #[arg(long = "set", value_name = "KEY=JSON")]
        settings: Vec<String>,
    },
    Compare {
        #[arg(long, default_value = "benchmark")]
        root: PathBuf,
        #[arg(long)]
        profile: String,
        #[arg(long)]
        candidate_endpoint: String,
        #[arg(long)]
        candidate_model: String,
        #[arg(long, default_value = "candidate")]
        candidate_name: String,
        #[arg(long, value_enum, default_value_t = KindArg::Icn)]
        candidate_kind: KindArg,
        #[arg(long)]
        reference_endpoint: String,
        #[arg(long)]
        reference_model: String,
        #[arg(long, default_value = "reference")]
        reference_name: String,
        #[arg(long, value_enum, default_value_t = KindArg::LlamaCpp)]
        reference_kind: KindArg,
        #[arg(long)]
        api_key_env: Option<String>,
        #[arg(long)]
        model_sha256: Option<String>,
        #[arg(long)]
        candidate_process_id: Option<u32>,
        #[arg(long)]
        reference_process_id: Option<u32>,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        run_id: Option<String>,
        #[arg(long)]
        controlled_host: bool,
        #[arg(long)]
        exclusive_device: bool,
        #[arg(long = "set", value_name = "KEY=JSON")]
        settings: Vec<String>,
        #[arg(long = "candidate-set", value_name = "KEY=JSON")]
        candidate_settings: Vec<String>,
        #[arg(long = "reference-set", value_name = "KEY=JSON")]
        reference_settings: Vec<String>,
    },
    CompareEvidence {
        #[arg(long)]
        candidate: PathBuf,
        #[arg(long)]
        reference: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum KindArg {
    Generic,
    Icn,
    LlamaCpp,
}

impl From<KindArg> for TargetKind {
    fn from(value: KindArg) -> Self {
        match value {
            KindArg::Generic => Self::Generic,
            KindArg::Icn => Self::Icn,
            KindArg::LlamaCpp => Self::LlamaCpp,
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Validate { root } => {
            validate_assets(&root).await?;
            println!("benchmark assets are valid: {}", root.display());
        }
        Command::Run {
            root,
            profile,
            endpoint,
            model,
            name,
            kind,
            api_key_env,
            model_sha256,
            process_id,
            output,
            run_id,
            controlled_host,
            exclusive_device,
            settings,
        } => {
            let api_key = resolve_api_key(api_key_env.as_deref())?;
            let configuration = parse_settings(&settings)?;
            run(BenchmarkRunConfig {
                root,
                profile,
                targets: vec![TargetConfig {
                    name,
                    endpoint,
                    model,
                    kind: kind.into(),
                    api_key,
                    model_sha256,
                    process_id,
                    configuration,
                }],
                output_dir: output,
                run_id,
                controlled_host,
                exclusive_device,
            })
            .await?;
        }
        Command::Compare {
            root,
            profile,
            candidate_endpoint,
            candidate_model,
            candidate_name,
            candidate_kind,
            reference_endpoint,
            reference_model,
            reference_name,
            reference_kind,
            api_key_env,
            model_sha256,
            candidate_process_id,
            reference_process_id,
            output,
            run_id,
            controlled_host,
            exclusive_device,
            settings,
            candidate_settings,
            reference_settings,
        } => {
            let api_key = resolve_api_key(api_key_env.as_deref())?;
            let configuration = parse_settings(&settings)?;
            let mut candidate_configuration = configuration.clone();
            candidate_configuration.extend(parse_settings(&candidate_settings)?);
            let mut reference_configuration = configuration;
            reference_configuration.extend(parse_settings(&reference_settings)?);
            run(BenchmarkRunConfig {
                root,
                profile,
                targets: vec![
                    TargetConfig {
                        name: candidate_name,
                        endpoint: candidate_endpoint,
                        model: candidate_model,
                        kind: candidate_kind.into(),
                        api_key: api_key.clone(),
                        model_sha256: model_sha256.clone(),
                        process_id: candidate_process_id,
                        configuration: candidate_configuration,
                    },
                    TargetConfig {
                        name: reference_name,
                        endpoint: reference_endpoint,
                        model: reference_model,
                        kind: reference_kind.into(),
                        api_key,
                        model_sha256,
                        process_id: reference_process_id,
                        configuration: reference_configuration,
                    },
                ],
                output_dir: output,
                run_id,
                controlled_host,
                exclusive_device,
            })
            .await?;
        }
        Command::CompareEvidence {
            candidate,
            reference,
            output,
        } => {
            let candidate: BenchmarkRun =
                serde_json::from_slice(&tokio::fs::read(&candidate).await?)?;
            let reference: BenchmarkRun =
                serde_json::from_slice(&tokio::fs::read(&reference).await?)?;
            let comparison = compare_evidence(&candidate, &reference)?;
            tokio::fs::create_dir_all(&output).await?;
            tokio::fs::write(
                output.join("comparison.json"),
                serde_json::to_vec_pretty(&comparison)?,
            )
            .await?;
            tokio::fs::write(
                output.join("comparison.md"),
                comparison_markdown(&comparison),
            )
            .await?;
            println!("{}", output.join("comparison.md").display());
        }
    }
    Ok(())
}

async fn run(config: BenchmarkRunConfig) -> anyhow::Result<()> {
    let output = config.output_dir.clone();
    let runner = BenchmarkRunner::new().with_progress(Arc::new(|event| match event {
        ProgressEvent::RunStarted { run_id } => eprintln!("starting {run_id}"),
        ProgressEvent::TargetStarted { target } => eprintln!("target {target}"),
        ProgressEvent::ExperimentStarted { target, experiment } => {
            eprintln!("{target}: {experiment}")
        }
        ProgressEvent::SampleCompleted {
            target,
            experiment,
            arm,
            repetition,
        } => {
            eprintln!("{target}: {experiment}/{arm} repetition {}", repetition + 1)
        }
        ProgressEvent::TargetCompleted { target } => eprintln!("completed target {target}"),
        ProgressEvent::RunCompleted { run_id } => eprintln!("completed {run_id}"),
    }));
    let (_, comparison) = runner.run(config).await?;
    println!("{}", output.join("summary.md").display());
    if comparison.is_some() {
        println!("{}", output.join("comparison.md").display());
    }
    Ok(())
}

fn resolve_api_key(name: Option<&str>) -> anyhow::Result<Option<String>> {
    name.map(|name| {
        std::env::var(name).with_context(|| format!("environment variable {name} is not set"))
    })
    .transpose()
}

fn parse_settings(
    values: &[String],
) -> anyhow::Result<std::collections::BTreeMap<String, serde_json::Value>> {
    values
        .iter()
        .map(|value| {
            let (key, raw) = value
                .split_once('=')
                .with_context(|| format!("invalid --set {value:?}; expected KEY=JSON"))?;
            if key.is_empty() {
                anyhow::bail!("invalid --set {value:?}; key is empty");
            }
            let parsed = serde_json::from_str(raw)
                .unwrap_or_else(|_| serde_json::Value::String(raw.to_owned()));
            Ok((key.to_owned(), parsed))
        })
        .collect()
}
