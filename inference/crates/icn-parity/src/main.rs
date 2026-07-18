use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use icn_parity::assets::{AssetRepository, SchemaKind};
use icn_parity::compare::{ComparisonContext, compare_evidence};
use icn_parity::decode::read_evidence;
use icn_parity::digest::sha256_file;
use icn_parity::model::{EngineOrder, ProducerRole};
use icn_parity::models::all_verified;
use icn_parity::runner::{RunOptions, RunStatus, run_profile};
use tokio::io::AsyncWriteExt;

#[derive(Debug, Parser)]
#[command(
    name = "icn-parity",
    version,
    about = "Primitive parity validation for ICN and pinned llama.cpp"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Validate every declarative parity asset and cross-reference.
    Validate(ValidateArgs),
    /// List cases, optionally after applying a profile's selection.
    List(ListArgs),
    /// Execute an immutable parity run.
    Run(RunArgs),
    /// Compare two already-recorded evidence documents.
    Compare(CompareArgs),
    /// Inspect, verify, or explicitly fetch registry-pinned models.
    Models(ModelsArgs),
}

#[derive(Debug, Args)]
struct RootArgs {
    /// Parity asset root (normally inference/parity).
    #[arg(long)]
    root: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct ValidateArgs {
    #[command(flatten)]
    root: RootArgs,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct ListArgs {
    #[command(flatten)]
    root: RootArgs,
    #[arg(long)]
    profile: Option<String>,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct RunArgs {
    #[command(flatten)]
    root: RootArgs,
    #[arg(long)]
    profile: String,
    #[arg(long)]
    output_root: Option<PathBuf>,
    #[arg(long)]
    run_id: Option<String>,
    #[arg(long)]
    reference_manifest: Option<PathBuf>,
    #[arg(long)]
    candidate_manifest: Option<PathBuf>,
    #[arg(long)]
    model_root: Option<PathBuf>,
    #[arg(long = "model")]
    models: Vec<String>,
    /// Explicit runner setting in NAME=VALUE form. May be repeated.
    #[arg(long = "set", value_name = "NAME=VALUE")]
    settings: Vec<String>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliEngineOrder {
    ReferenceFirst,
    CandidateFirst,
    AlternatingPairs,
    NotApplicable,
}

impl From<CliEngineOrder> for EngineOrder {
    fn from(value: CliEngineOrder) -> Self {
        match value {
            CliEngineOrder::ReferenceFirst => Self::ReferenceFirst,
            CliEngineOrder::CandidateFirst => Self::CandidateFirst,
            CliEngineOrder::AlternatingPairs => Self::AlternatingPairs,
            CliEngineOrder::NotApplicable => Self::NotApplicable,
        }
    }
}

#[derive(Debug, Args)]
struct CompareArgs {
    #[command(flatten)]
    root: RootArgs,
    #[arg(long)]
    case: String,
    #[arg(long)]
    reference: PathBuf,
    #[arg(long)]
    candidate: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_enum, default_value = "not-applicable")]
    engine_order: CliEngineOrder,
}

#[derive(Debug, Args)]
struct ModelsArgs {
    #[command(flatten)]
    root: RootArgs,
    #[command(subcommand)]
    command: ModelsCommand,
}

#[derive(Debug, Subcommand)]
enum ModelsCommand {
    List {
        #[arg(long)]
        json: bool,
    },
    Verify(ModelSelection),
    Fetch(ModelSelection),
}

#[derive(Debug, Args)]
struct ModelSelection {
    #[arg(long = "id")]
    ids: Vec<String>,
    /// Select every accepted registry model. Fetch otherwise requires --id.
    #[arg(long, conflicts_with = "ids")]
    all: bool,
    #[arg(long)]
    model_root: Option<PathBuf>,
    #[arg(long)]
    json: bool,
}

#[tokio::main]
async fn main() -> ExitCode {
    match execute(Cli::parse()).await {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::from(2)
        }
    }
}

async fn execute(cli: Cli) -> anyhow::Result<ExitCode> {
    match cli.command {
        Command::Validate(args) => validate(args).await,
        Command::List(args) => list(args),
        Command::Run(args) => run(args).await,
        Command::Compare(args) => compare(args).await,
        Command::Models(args) => models(args).await,
    }
}

fn repository(root: &RootArgs) -> anyhow::Result<AssetRepository> {
    let root = AssetRepository::discover_root(root.root.as_deref())?;
    AssetRepository::load(root)
}

async fn validate(args: ValidateArgs) -> anyhow::Result<ExitCode> {
    let repository = repository(&args.root)?;
    let report = repository.validate().await;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        for diagnostic in &report.errors {
            println!(
                "error [{}] {}: {}",
                diagnostic.code,
                diagnostic.path.display(),
                diagnostic.message
            );
        }
        for diagnostic in &report.warnings {
            println!(
                "warning [{}] {}: {}",
                diagnostic.code,
                diagnostic.path.display(),
                diagnostic.message
            );
        }
        println!(
            "{} error(s), {} warning(s)",
            report.errors.len(),
            report.warnings.len()
        );
    }
    Ok(if report.is_valid() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
}

fn list(args: ListArgs) -> anyhow::Result<ExitCode> {
    let repository = repository(&args.root)?;
    let cases = if let Some(profile) = &args.profile {
        repository.selected_cases(repository.profile(profile)?)
    } else {
        repository.cases.values().collect()
    };
    if args.json {
        let definitions = cases
            .iter()
            .map(|case| &case.definition)
            .collect::<Vec<_>>();
        println!("{}", serde_json::to_string_pretty(&definitions)?);
    } else {
        for case in cases {
            println!(
                "{}\t{:?}\t{}\t{:?}",
                case.definition.id,
                case.definition.status,
                case.definition.primitive,
                case.definition.category,
            );
        }
    }
    Ok(ExitCode::SUCCESS)
}

async fn run(args: RunArgs) -> anyhow::Result<ExitCode> {
    let repository = repository(&args.root)?;
    let settings = parse_settings(&args.settings)?;
    let summary = run_profile(
        &repository,
        RunOptions {
            profile_id: args.profile,
            output_root: args.output_root,
            run_id: args.run_id,
            reference_manifest: args.reference_manifest,
            candidate_manifest: args.candidate_manifest,
            model_root: args.model_root,
            model_ids: args.models,
            settings,
        },
    )
    .await?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(if summary.status == RunStatus::Pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
}

async fn compare(args: CompareArgs) -> anyhow::Result<ExitCode> {
    let repository = repository(&args.root)?;
    let case = &repository
        .cases
        .get(&args.case)
        .with_context(|| format!("unknown parity case {}", args.case))?
        .definition;
    let reference = read_evidence(&args.reference).await?;
    let candidate = read_evidence(&args.candidate).await?;
    repository.validate_schema(SchemaKind::Evidence, &reference)?;
    repository.validate_schema(SchemaKind::Evidence, &candidate)?;
    reference.validate_for(case, ProducerRole::Reference)?;
    candidate.validate_for(case, ProducerRole::Candidate)?;
    if reference.run_id != candidate.run_id {
        bail!("reference and candidate evidence have different run IDs");
    }
    let (reference_sha256, _) = sha256_file(&args.reference).await?;
    let (candidate_sha256, _) = sha256_file(&args.candidate).await?;
    let reference_path = args.reference.to_string_lossy();
    let candidate_path = args.candidate.to_string_lossy();
    let comparison = compare_evidence(
        case,
        &reference,
        &candidate,
        ComparisonContext {
            run_id: &reference.run_id,
            reference_path: &reference_path,
            reference_sha256: &reference_sha256,
            candidate_path: &candidate_path,
            candidate_sha256: &candidate_sha256,
            engine_order: args.engine_order.into(),
        },
    )?;
    repository.validate_schema(SchemaKind::Comparison, &comparison)?;
    write_create_new_json(&args.output, &comparison).await?;
    println!("{}", serde_json::to_string_pretty(&comparison)?);
    Ok(
        if comparison.status == icn_parity::model::ComparisonStatus::Pass {
            ExitCode::SUCCESS
        } else {
            ExitCode::from(1)
        },
    )
}

async fn models(args: ModelsArgs) -> anyhow::Result<ExitCode> {
    let repository = repository(&args.root)?;
    match args.command {
        ModelsCommand::List { json } => {
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&repository.models.models)?
                );
            } else {
                for model in &repository.models.models {
                    println!("{}\t{}\t{}", model.id, model.status, model.display_name);
                }
            }
            Ok(ExitCode::SUCCESS)
        }
        ModelsCommand::Verify(selection) => {
            let root = repository
                .models
                .artifact_root(&repository.root, selection.model_root.as_deref());
            let statuses = repository.models.verify(&selection.ids, &root).await?;
            print_model_statuses(&statuses, selection.json)?;
            Ok(if all_verified(&statuses) {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            })
        }
        ModelsCommand::Fetch(selection) => {
            let root = repository
                .models
                .artifact_root(&repository.root, selection.model_root.as_deref());
            let ids = if selection.all {
                repository
                    .models
                    .models
                    .iter()
                    .map(|model| model.id.clone())
                    .collect::<Vec<_>>()
            } else {
                selection.ids.clone()
            };
            if ids.is_empty() {
                bail!("model fetch requires at least one --id or explicit --all");
            }
            let statuses = repository.models.fetch(&ids, &root).await?;
            print_model_statuses(&statuses, selection.json)?;
            Ok(if all_verified(&statuses) {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            })
        }
    }
}

fn parse_settings(values: &[String]) -> anyhow::Result<BTreeMap<String, String>> {
    let mut settings = BTreeMap::new();
    for value in values {
        let (name, value) = value.split_once('=').context("--set requires NAME=VALUE")?;
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-'))
            || value.is_empty()
        {
            bail!("invalid --set value {name:?}=...");
        }
        if settings.insert(name.to_owned(), value.to_owned()).is_some() {
            bail!("duplicate --set key {name}");
        }
    }
    Ok(settings)
}

fn print_model_statuses(
    statuses: &[icn_parity::models::ModelFileStatus],
    json: bool,
) -> anyhow::Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(statuses)?);
    } else {
        for status in statuses {
            println!(
                "{}\t{:?}\t{}",
                status.model_id,
                status.state,
                status.path.display()
            );
        }
    }
    Ok(())
}

async fn write_create_new_json<T: serde::Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    let parent = path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    tokio::fs::create_dir_all(parent)
        .await
        .with_context(|| format!("failed to create output directory {}", parent.display()))?;
    let mut options = tokio::fs::OpenOptions::new();
    options.write(true).create_new(true);
    let mut file = options
        .open(path)
        .await
        .with_context(|| format!("refusing to overwrite comparison output {}", path.display()))?;
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    file.write_all(&bytes).await?;
    file.flush().await?;
    file.sync_all().await?;
    Ok(())
}
