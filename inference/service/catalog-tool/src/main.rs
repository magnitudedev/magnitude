mod headers;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::Context;
use clap::{Parser, Subcommand};
use magnitude_service_contracts::ModelDownloadEvent;
use magnitude_service_contracts::models::{ModelId, SpeculativeMethod};
use magnitude_service_models::{
    AcquiredCatalogModel, AcquiredFile, CatalogLockAdvance, InventoryConfig, ManagedModelStore,
    ResolvingRecommendableCatalog, advance_model_catalog_lock, load_release_catalog,
};
use serde_json::{Value, json};

#[derive(Debug, Parser)]
#[command(
    name = env!("CARGO_BIN_NAME"),
    about = "Build Magnitude's release model catalog"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Advance the curated model source lock to current upstream commits.
    ///
    /// Without `--model`, every model that is not deprecated advances. With it, only the named
    /// models advance and every other entry keeps its locked revisions.
    UpdateLock {
        #[arg(long)]
        output: PathBuf,
        #[arg(long = "model")]
        models: Vec<String>,
        #[arg(long)]
        model_store: PathBuf,
        #[arg(long)]
        cache_root: PathBuf,
        #[arg(long = "hf-cache")]
        hf_caches: Vec<PathBuf>,
    },
    /// Build the assessment header bundle from the pinned catalog revisions.
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
    /// Download (only what is missing) and verify one catalog model's target, drafter and
    /// projector at their locked revisions, then print their paths as JSON.
    Acquire {
        /// Catalog model ID, `<model id>:<variant id>` (for example `qwen3.5-4b:gguf:q4`).
        #[arg(long)]
        model: ModelId,
        #[arg(long)]
        model_store: PathBuf,
        /// Derived-cache root for repository metadata and GGUF headers.
        #[arg(long)]
        cache_root: Option<PathBuf>,
    },
    /// Dump every catalog component's GGUF header from a planner bundle as the JSON the engine
    /// family tests read, replacing the contents of `--output`.
    DumpHeaders {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}

async fn update_lock(
    output: PathBuf,
    advance: CatalogLockAdvance,
    model_store: PathBuf,
    cache_root: PathBuf,
    hf_caches: Vec<PathBuf>,
) -> anyhow::Result<()> {
    let models = open_catalog_models(model_store, cache_root, hf_caches).await?;
    eprintln!("Resolving current catalog revisions...");
    let lock = advance_model_catalog_lock(Arc::clone(&models), advance).await?;
    eprintln!("Validating candidate catalog lock...");
    let generated = ResolvingRecommendableCatalog::new(models)
        .resolve_release_catalog_with_lock(lock.clone(), report_progress)
        .await
        .context("failed to validate candidate model catalog lock")?;
    ensure_catalog_resolved(&generated)?;
    let encoded = serde_json::to_vec_pretty(&lock)?;
    publish(&output, &[encoded.as_slice(), b"\n"].concat(), "json.tmp").await?;
    println!("updated {}", output.display());
    Ok(())
}

async fn open_catalog_models(
    model_store: PathBuf,
    cache_root: PathBuf,
    hf_caches: Vec<PathBuf>,
) -> anyhow::Result<Arc<ManagedModelStore>> {
    let mut config = InventoryConfig::with_roots(model_store, cache_root)
        .context("invalid catalog-build inventory configuration")?;
    config.add_hugging_face_caches(hf_caches);
    let space = Arc::new(magnitude_executor::platform::volume_space(
        config.root.clone(),
        magnitude_executor::platform::MemoryReserves::standard(),
    ));
    Ok(Arc::new(ManagedModelStore::open(config, space).await?))
}

fn ensure_catalog_resolved(
    generated: &magnitude_service_models::GeneratedReleaseCatalog,
) -> anyhow::Result<()> {
    for diagnostic in &generated.catalog.diagnostics {
        eprintln!(
            "Catalog resolution failed for {} variant {}: {}",
            diagnostic.model_id, diagnostic.variant_id, diagnostic.failure.message
        );
    }
    anyhow::ensure!(
        generated.catalog.diagnostics.is_empty(),
        "catalog resolution failed for {} entries",
        generated.catalog.diagnostics.len()
    );
    Ok(())
}

async fn build_bundle(
    output: PathBuf,
    model_store: PathBuf,
    cache_root: PathBuf,
    hf_caches: Vec<PathBuf>,
) -> anyhow::Result<()> {
    if output.is_file()
        && load_release_catalog(&output)
            .and_then(|catalog| catalog.verify_all_headers())
            .is_ok()
    {
        eprintln!("Model catalog bundle is already current.");
        return Ok(());
    }
    let models = open_catalog_models(model_store, cache_root, hf_caches).await?;
    eprintln!("Resolving pinned catalog models...");
    let generated = ResolvingRecommendableCatalog::new(models)
        .resolve_release_catalog(report_progress)
        .await
        .context("failed to resolve curated model catalog")?;
    ensure_catalog_resolved(&generated)?;
    eprintln!("Encoding planner input bundle...");
    let bytes = generated.encode_planner_bundle(|completed, total| {
        report_progress("Encoded planner inputs", completed, total);
    })?;
    publish(&output, &bytes, "bundle.tmp").await?;
    eprintln!("Validating published planner input bundle...");
    let published = load_release_catalog(&output)
        .context("built planner inputs do not satisfy the runtime contract")?;
    published
        .verify_all_headers()
        .context("built planner inputs contain a corrupt header chunk")?;
    anyhow::ensure!(
        serde_json::to_vec(published.catalog())? == serde_json::to_vec(&generated.catalog)?,
        "built planner inputs changed the resolved catalog"
    );
    println!("generated {}", output.display());
    Ok(())
}

async fn acquire(
    model: ModelId,
    model_store: PathBuf,
    cache_root: Option<PathBuf>,
) -> anyhow::Result<()> {
    let cache_root = match cache_root {
        Some(cache_root) => cache_root,
        None => InventoryConfig::default_cache_root()?,
    };
    let models = open_catalog_models(
        std::path::absolute(&model_store)?,
        std::path::absolute(&cache_root)?,
        Vec::new(),
    )
    .await?;
    let resolved = ResolvingRecommendableCatalog::new(Arc::clone(&models))
        .resolve_catalog_model(&model)
        .await
        .with_context(|| format!("failed to resolve catalog model {model}"))?;
    let last_report = Mutex::new(None::<Instant>);
    let acquired = models
        .acquire_catalog_model(&resolved, |event| report_download(event, &last_report))
        .await
        .with_context(|| format!("failed to acquire catalog model {model}"))?;
    println!("{}", serde_json::to_string_pretty(&acquisition_json(&model, &acquired))?);
    Ok(())
}

fn report_download(event: &ModelDownloadEvent, last_report: &Mutex<Option<Instant>>) {
    match event {
        ModelDownloadEvent::Resolving {
            repository,
            revision,
            ..
        } => eprintln!("Acquiring {repository}@{revision}"),
        ModelDownloadEvent::Progress {
            stage,
            completed_bytes,
            total_bytes,
            file,
            ..
        } => {
            let mut last = last_report.lock().expect("download report lock");
            if last.is_some_and(|last| last.elapsed() < Duration::from_secs(5)) {
                return;
            }
            *last = Some(Instant::now());
            eprintln!(
                "{stage:?} {}: {completed_bytes}/{total_bytes} bytes",
                file.path.display()
            );
        }
        ModelDownloadEvent::Ready { .. } => eprintln!("Verified and published"),
        ModelDownloadEvent::CheckingSpace { .. }
        | ModelDownloadEvent::Cancelled { .. }
        | ModelDownloadEvent::Failed { .. } => {}
    }
}

/// The evaluation contract's catalog-acquisition object.
fn acquisition_json(model: &ModelId, acquired: &AcquiredCatalogModel) -> Value {
    let file = |file: &AcquiredFile| json!({ "path": file.path, "sha256": file.sha256 });
    json!({
        "model": model.as_str(),
        "target": acquired.target.path,
        "target_sha256": acquired.target.sha256,
        "drafter": acquired.drafter.as_ref().map(|drafter| json!({
            "method": speculative_method_name(&drafter.method),
            "path": drafter.file.path,
            "sha256": drafter.file.sha256,
        })),
        "projector": acquired.projector.as_ref().map(file),
    })
}

/// The catalog's own spelling of a speculative-decoding method.
fn speculative_method_name(method: &SpeculativeMethod) -> &'static str {
    match method {
        SpeculativeMethod::Mtp => "mtp",
        SpeculativeMethod::DFlash => "dflash",
        SpeculativeMethod::DFlash2 => "dflash2",
        SpeculativeMethod::DSpark => "dspark",
    }
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
            models,
            model_store,
            cache_root,
            hf_caches,
        } => {
            let advance = if models.is_empty() {
                CatalogLockAdvance::All
            } else {
                CatalogLockAdvance::Models(models.into_iter().collect())
            };
            update_lock(output, advance, model_store, cache_root, hf_caches).await
        }
        Command::BuildBundle {
            output,
            model_store,
            cache_root,
            hf_caches,
        } => build_bundle(output, model_store, cache_root, hf_caches).await,
        Command::Acquire {
            model,
            model_store,
            cache_root,
        } => acquire(model, model_store, cache_root).await,
        Command::DumpHeaders { bundle, output } => {
            let count = headers::dump(&bundle, &output)?;
            println!("dumped {count} component headers into {}", output.display());
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(path: &str, sha256: &str) -> AcquiredFile {
        AcquiredFile {
            path: PathBuf::from(path),
            sha256: sha256.to_owned(),
        }
    }

    #[test]
    fn acquisition_json_matches_the_contract() {
        let model: ModelId = "qwen3.5-4b:gguf:q4".parse().unwrap();
        let acquired = AcquiredCatalogModel {
            target: file("/store/target.gguf", "aa"),
            drafter: Some(magnitude_service_models::AcquiredDrafter {
                method: SpeculativeMethod::DFlash,
                file: file("/store/draft.gguf", "bb"),
            }),
            projector: Some(file("/store/mmproj.gguf", "cc")),
        };
        assert_eq!(
            acquisition_json(&model, &acquired),
            json!({
                "model": "qwen3.5-4b:gguf:q4",
                "target": "/store/target.gguf",
                "target_sha256": "aa",
                "drafter": { "method": "dflash", "path": "/store/draft.gguf", "sha256": "bb" },
                "projector": { "path": "/store/mmproj.gguf", "sha256": "cc" },
            })
        );
    }

    #[test]
    fn absent_drafter_and_projector_are_null() {
        let model: ModelId = "bonsai-8b-q1:gguf:q1".parse().unwrap();
        let acquired = AcquiredCatalogModel {
            target: file("/store/target.gguf", "aa"),
            drafter: None,
            projector: None,
        };
        let value = acquisition_json(&model, &acquired);
        assert_eq!(value["drafter"], Value::Null);
        assert_eq!(value["projector"], Value::Null);
    }

    #[test]
    fn model_argument_rejects_a_malformed_id() {
        assert!(Cli::try_parse_from(["catalog", "acquire", "--model", "qwen3.5-4b", "--model-store", "/m"]).is_err());
        assert!(Cli::try_parse_from(["catalog", "acquire", "--model", "qwen3.5-4b:gguf:q4", "--model-store", "/m"]).is_ok());
    }
}
