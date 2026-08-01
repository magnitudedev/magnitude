use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use clap::{Parser, Subcommand};
use icn_contracts::{EffectiveTemplateInputs, TemplateAssessment, TemplateAssessor};
use icn_engine::NativeBackend;
use icn_models::{
    InventoryConfig, ModelManager, ResolvingRecommendableCatalog, advance_model_catalog_lock,
    load_release_catalog,
};

#[derive(Debug, Parser)]
#[command(
    name = "icn-catalog",
    about = "Build Magnitude's release model catalog"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Advance the curated model source lock to current upstream commits.
    UpdateLock {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        model_store: PathBuf,
        #[arg(long)]
        cache_root: PathBuf,
        #[arg(long = "hf-cache")]
        hf_caches: Vec<PathBuf>,
    },
    /// Build planner inputs from the pinned catalog revisions.
    BuildBundle {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        model_store: PathBuf,
        #[arg(long)]
        cache_root: PathBuf,
        #[arg(long = "hf-cache")]
        hf_caches: Vec<PathBuf>,
    },
}

struct NativeTemplateAssessor {
    backend: NativeBackend,
}

impl NativeTemplateAssessor {
    fn new(backend: NativeBackend) -> Self {
        Self { backend }
    }
}

impl TemplateAssessor for NativeTemplateAssessor {
    fn cache_identity(&self) -> &str {
        icn_reasoning::TEMPLATE_INSPECTION_CACHE_IDENTITY
    }

    fn assess(&self, inputs: &EffectiveTemplateInputs) -> Result<TemplateAssessment, String> {
        let inspection = icn_reasoning::inspect_template_inputs_with_backend(
            self.backend.as_llama_backend(),
            inputs,
        )
        .map_err(|error| error.to_string())?;
        Ok(TemplateAssessment {
            capabilities: inspection.capabilities,
            reasoning: inspection.reasoning,
            fingerprint: inspection.template_fingerprint,
        })
    }
}

async fn update_lock(
    output: PathBuf,
    model_store: PathBuf,
    cache_root: PathBuf,
    hf_caches: Vec<PathBuf>,
) -> anyhow::Result<()> {
    let mut config = InventoryConfig::with_roots(model_store, cache_root)
        .context("invalid model catalog lock inventory configuration")?;
    config.hf_cache_dirs.extend(hf_caches);
    let models = Arc::new(ModelManager::open(config).await?);
    let encoded = serde_json::to_vec_pretty(&advance_model_catalog_lock(models).await?)?;
    publish(&output, &[encoded.as_slice(), b"\n"].concat(), "json.tmp").await?;
    println!("updated {}", output.display());
    Ok(())
}

async fn build_bundle(
    output: PathBuf,
    model_store: PathBuf,
    cache_root: PathBuf,
    hf_caches: Vec<PathBuf>,
) -> anyhow::Result<()> {
    if output.is_file() && load_release_catalog(&output).is_ok() {
        eprintln!("Model catalog bundle is already current.");
        return Ok(());
    }
    icn_engine::disable_native_diagnostics();
    let backend = NativeBackend::initialize().context("failed to initialize native backend")?;
    let mut config = InventoryConfig::with_roots(model_store, cache_root)
        .context("invalid catalog-build inventory configuration")?;
    config.hf_cache_dirs.extend(hf_caches);
    let models = Arc::new(
        ModelManager::open_with_template_assessor(
            config,
            Some(Arc::new(NativeTemplateAssessor::new(backend.clone()))),
        )
        .await?,
    );
    eprintln!("Resolving pinned catalog models...");
    let generated = ResolvingRecommendableCatalog::new(models)
        .resolve_release_catalog(report_progress)
        .await
        .context("failed to resolve curated model catalog")?;
    eprintln!("Encoding planner input bundle...");
    let bytes = generated.encode_planner_bundle(|completed, total| {
        report_progress("Encoded planner inputs", completed, total);
    })?;
    publish(&output, &bytes, "bundle.tmp").await?;
    eprintln!("Validating published planner input bundle...");
    let published = load_release_catalog(&output)
        .context("built planner inputs do not satisfy the runtime contract")?;
    anyhow::ensure!(
        serde_json::to_vec(published.catalog())? == serde_json::to_vec(&generated.catalog)?,
        "built planner inputs changed the resolved catalog"
    );
    println!("generated {}", output.display());
    Ok(())
}

fn report_progress(stage: &str, completed: usize, total: usize) {
    if completed == 1 || completed.is_multiple_of(5) || completed == total {
        eprintln!("{stage}: {completed}/{total}");
    }
}

async fn publish(output: &Path, bytes: &[u8], temporary_extension: &str) -> anyhow::Result<()> {
    let parent = output
        .parent()
        .context("catalog output must have a parent directory")?;
    tokio::fs::create_dir_all(parent).await?;
    let temporary = output.with_extension(temporary_extension);
    tokio::fs::write(&temporary, bytes).await?;
    tokio::fs::rename(&temporary, output).await?;
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::UpdateLock {
            output,
            model_store,
            cache_root,
            hf_caches,
        } => update_lock(output, model_store, cache_root, hf_caches).await,
        Command::BuildBundle {
            output,
            model_store,
            cache_root,
            hf_caches,
        } => build_bundle(output, model_store, cache_root, hf_caches).await,
    }
}
