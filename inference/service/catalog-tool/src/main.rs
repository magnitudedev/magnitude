mod audit;
mod fixture_header;
mod gguf_source;
mod slice;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::Context;
use clap::{Parser, Subcommand};
use gguf_source::{Gguf, GgufSource};
use magnitude_artifacts::catalog_headers::{self, HeaderIndex};
use magnitude_artifacts::gguf::{Array, Scalar, Value as GgufValue, inspect_header};
use serde_json::Map;
use magnitude_service_contracts::ModelDownloadEvent;
use magnitude_service_contracts::models::{ModelId, SpeculativeMethod};
use magnitude_service_models::{
    AcquiredCatalogModel, AcquiredFile, CatalogLockAdvance, InventoryConfig, ManagedModelStore,
    ResolvingRecommendableCatalog, WeightSample, advance_model_catalog_lock, catalog_lock_digest,
    load_release_catalog,
};
use serde_json::{Value, json};

#[derive(Debug, Parser)]
#[command(
    name = env!("CARGO_BIN_NAME"),
    about = "Build Magnitude's release model catalog and produce and inspect its model files"
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
    /// Export every catalog component's exact header at its locked revision, and each pinned
    /// artifact's outside the catalog, for header-only tests: builds the header bundle under
    /// `output` (only when it is not current for the lock), then writes the headers to
    /// `output/headers/` with their index (`magnitude_artifacts::catalog_headers`) unless that
    /// export is of the current lock and artifacts.
    Headers {
        #[arg(long)]
        output: PathBuf,
        #[command(flatten)]
        store: StoreArgs,
    },
    /// Download (only what is missing) and verify one catalog model's target, drafter and
    /// projector at their locked revisions, then print their paths as JSON.
    Acquire {
        /// Catalog model ID, `<model id>:<variant id>` (for example `qwen3.5-4b:gguf:q4`).
        #[arg(long)]
        model: ModelId,
        #[command(flatten)]
        store: StoreArgs,
    },
    /// Fetch one GGUF header for the assessment fixtures by range requests only, trim it (keep
    /// every metadata key, the vocabulary's length and token kinds and the tensor directory;
    /// empty `tokenizer.ggml.merges` and blank the text and score of every normal token no
    /// special-token key names) and gzip it to `<output-dir>/<file name>.gz`.
    FixtureHeader {
        #[arg(long)]
        repository: String,
        /// An immutable commit.
        #[arg(long)]
        revision: String,
        /// The file's path in the repository.
        #[arg(long)]
        path: PathBuf,
        #[arg(long, default_value = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../engine/fixtures/assessment/headers"
        ))]
        output_dir: PathBuf,
        #[command(flatten)]
        store: StoreArgs,
    },
    /// Write one GGUF's weight sample for the weight-lineage tests to `<output-dir>/<name>.json`,
    /// as `WeightSample` serializes it, reading the header and then only the sampled values.
    WeightSample {
        #[arg(long)]
        name: String,
        #[arg(long, default_value = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../engine/fixtures/assessment/weight-samples"
        ))]
        output_dir: PathBuf,
        #[command(subcommand)]
        source: SampleSource,
    },
    /// Print a GGUF's metadata and tensor directory as JSON: arrays longer than 64 as their
    /// length and first 8 values, each tensor as its shape (innermost first) and type. A split
    /// GGUF is named by its first shard and printed whole.
    Header {
        #[command(subcommand)]
        source: GgufSource,
    },
    /// Write a truncated real-weight GGUF: the global tensors, a selection of main layers
    /// renumbered from 0 and the MTP layers after them, reading only the kept tensors' bytes.
    ///
    /// The default selection is the shortest prefix of the main layers holding every layer
    /// variant (a layer's tensor names and shapes and its per-layer metadata entries) and at
    /// least `--minimum` layers. A provenance record `<output>.json` lists the source files,
    /// kept layers, renaming and output sha256.
    Slice {
        #[arg(long, required_unless_present = "list")]
        output: Option<PathBuf>,
        /// Comma-separated ascending main-layer indices.
        #[arg(long, value_delimiter = ',')]
        layers: Option<Vec<usize>>,
        /// Minimum main layers in the default prefix.
        #[arg(long, default_value_t = 4)]
        minimum: usize,
        /// Drop the trailing MTP layers.
        #[arg(long)]
        no_mtp: bool,
        /// Print the variants and the default selection only.
        #[arg(long)]
        list: bool,
        /// Concurrent tensor range reads.
        #[arg(long, default_value_t = 8)]
        parallel: usize,
        #[command(subcommand)]
        source: GgufSource,
    },
    /// Audit the tensor types of the locked catalog: tally each catalog component file's tensor
    /// encodings from the header export (see `headers`) under `--headers`, written first unless
    /// it is current, into `--output`. Fails when a file declares a tensor type the parser
    /// doesn't know.
    Audit {
        #[arg(long)]
        headers: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[command(flatten)]
        store: StoreArgs,
    },
}

/// The managed model store a command reads the Hub through and downloads into.
#[derive(Debug, clap::Args)]
struct StoreArgs {
    #[arg(long)]
    model_store: PathBuf,
    /// Derived-cache root for repository metadata and GGUF headers.
    #[arg(long)]
    cache_root: Option<PathBuf>,
}

impl StoreArgs {
    /// The absolute model store and cache roots; the cache root defaults to the derived cache's.
    fn roots(&self) -> anyhow::Result<(PathBuf, PathBuf)> {
        let cache_root = match &self.cache_root {
            Some(cache_root) => cache_root.clone(),
            None => InventoryConfig::default_cache_root()?,
        };
        Ok((
            std::path::absolute(&self.model_store)?,
            std::path::absolute(cache_root)?,
        ))
    }

    /// The store, presenting `HF_TOKEN` on Hub requests when it is set.
    async fn open(&self) -> anyhow::Result<Arc<ManagedModelStore>> {
        let (model_store, cache_root) = self.roots()?;
        let models = open_catalog_models(model_store, cache_root, Vec::new()).await?;
        if let Ok(token) = std::env::var("HF_TOKEN") {
            models.configure_hub_credential(Some(token))?;
        }
        Ok(models)
    }
}

/// Where a weight sample is read from.
#[derive(Debug, Subcommand)]
enum SampleSource {
    #[command(flatten)]
    Gguf(GgufSource),
    /// A model layer blob of the Ollama registry's library model, read by range requests.
    Ollama {
        /// The library model (`registry.ollama.ai/library/<model>`).
        #[arg(long)]
        model: String,
        /// The blob's digest, `sha256:<hex>`.
        #[arg(long)]
        digest: String,
        #[command(flatten)]
        store: StoreArgs,
    },
}

async fn fixture_header(
    repository: String,
    revision: String,
    path: PathBuf,
    output_dir: PathBuf,
    store: StoreArgs,
) -> anyhow::Result<()> {
    let store = store.open().await?;
    let snapshot = gguf_source::pinned_snapshot(&store, &repository, &revision).await?;
    let header = fixture_header::trim(&gguf_source::hub_header(&store, &snapshot, &path).await?)?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("the path names no file")?;
    let output = output_dir.join(format!("{name}.gz"));
    fixture_header::write_gzip(&output, name, &header)?;
    println!("{}: {} header bytes", output.display(), header.len());
    Ok(())
}

async fn weight_sample(
    source: SampleSource,
    name: String,
    output_dir: PathBuf,
) -> anyhow::Result<()> {
    let sample = match &source {
        SampleSource::Gguf(GgufSource::Local { path }) => {
            let directory = magnitude_service_models::gguf::known_tensors(inspect_header(path))
                .with_context(|| format!("failed to read {}", path.display()))?;
            WeightSample::read_file(&directory, path)?
        }
        SampleSource::Gguf(GgufSource::Hub {
            repository,
            revision,
            path,
            store,
        }) => {
            let store = store.open().await?;
            let snapshot = gguf_source::pinned_snapshot(&store, repository, revision).await?;
            store.hub_weight_sample(&snapshot, path).await?
        }
        SampleSource::Ollama {
            model,
            digest,
            store,
        } => {
            store
                .open()
                .await?
                .ollama_registry_weight_sample(model, digest)
                .await?
        }
    }
    .context("too few sampled tensors carry values")?;
    std::fs::create_dir_all(&output_dir)?;
    let output = output_dir.join(format!("{name}.json"));
    std::fs::write(&output, [sample.encode(), b"\n".to_vec()].concat())?;
    println!("{}: {} tensors", output.display(), sample.tensors().count());
    Ok(())
}

/// A GGUF's metadata, long arrays as their length and first values, and its tensor directory.
fn describe(gguf: &Gguf) -> Value {
    let scalar = |scalar: Scalar| match scalar {
        Scalar::String(value) => json!(value),
        Scalar::Bool(value) => json!(value),
        Scalar::Unsigned(value) => json!(value),
        Scalar::Signed(value) => json!(value),
        Scalar::Float(value) => json!(value),
    };
    let elements = |array: &Array, count: usize| {
        Value::Array(
            array
                .iter()
                .take(count)
                .map(|element| scalar(element.to_scalar()))
                .collect(),
        )
    };
    let metadata = gguf
        .first()
        .metadata
        .iter()
        .map(|entry| {
            let value = match &entry.value {
                GgufValue::Scalar(value) => scalar(value.clone()),
                GgufValue::Array(array) if array.len() > 64 => {
                    json!({ "length": array.len(), "head": elements(array, 8) })
                }
                GgufValue::Array(array) => elements(array, array.len()),
            };
            (entry.name.clone(), value)
        })
        .collect::<Map<_, _>>();
    let tensors = gguf
        .tensors()
        .map(|tensor| {
            let shape = tensor.shape.iter().rev().collect::<Vec<_>>();
            (
                tensor.name.clone(),
                json!([shape, tensor.encoding.ggml_name()]),
            )
        })
        .collect::<Map<_, _>>();
    json!({ "metadata": metadata, "tensors": tensors })
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

/// Writes the header export under `output` unless it is current, and returns its directory.
async fn headers(output: PathBuf, store: &StoreArgs) -> anyhow::Result<PathBuf> {
    let output = std::path::absolute(&output)?;
    let (model_store, cache_root) = store.roots()?;
    let bundle = output.join("model-planner-inputs.bundle");
    build_bundle(bundle.clone(), model_store, cache_root, Vec::new()).await?;
    let exported = output.join("headers");
    let artifacts = catalog_headers::artifacts();
    // An export of another lock or other artifacts, or none, is replaced.
    let current = std::fs::read(exported.join(catalog_headers::INDEX))
        .ok()
        .and_then(|index| serde_json::from_slice::<HeaderIndex>(&index).ok())
        .is_some_and(|index| index.lock == catalog_lock_digest() && index.artifacts == artifacts);
    if current {
        eprintln!("Catalog headers are already current.");
    } else {
        let staging = output.join("headers.tmp");
        if staging.exists() {
            tokio::fs::remove_dir_all(&staging).await?;
        }
        let components = load_release_catalog(&bundle)?.export_headers(&staging)?;
        let models = store.open().await?;
        for artifact in &artifacts {
            models.export_artifact_header(artifact, &staging).await?;
        }
        let index = HeaderIndex {
            lock: catalog_lock_digest(),
            components,
            artifacts,
        };
        let encoded = serde_json::to_vec_pretty(&index)?;
        tokio::fs::write(staging.join(catalog_headers::INDEX), encoded).await?;
        if exported.exists() {
            tokio::fs::remove_dir_all(&exported).await?;
        }
        tokio::fs::rename(&staging, &exported).await?;
    }
    Ok(exported)
}

async fn acquire(model: ModelId, store: &StoreArgs) -> anyhow::Result<()> {
    let models = store.open().await?;
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
        SpeculativeMethod::LiLiCorr => "lilicorr",
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
        Command::Headers { output, store } => {
            let exported = headers(output, &store).await?;
            println!("{}", exported.display());
            Ok(())
        }
        Command::Acquire { model, store } => acquire(model, &store).await,
        Command::FixtureHeader {
            repository,
            revision,
            path,
            output_dir,
            store,
        } => fixture_header(repository, revision, path, output_dir, store).await,
        Command::WeightSample {
            source,
            name,
            output_dir,
        } => weight_sample(source, name, output_dir).await,
        Command::Header { source } => {
            let gguf = Gguf::open(&source).await?;
            println!("{}", serde_json::to_string_pretty(&describe(&gguf))?);
            Ok(())
        }
        Command::Slice {
            output,
            layers,
            minimum,
            no_mtp,
            list,
            parallel,
            source,
        } => {
            let options = slice::Options {
                source,
                output,
                layers,
                minimum,
                no_mtp,
                list,
                parallel,
            };
            slice::run(options).await
        }
        Command::Audit {
            headers: output,
            output: report,
            store,
        } => {
            let exported = headers(output, &store).await?;
            audit::run(&exported, &report)
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

    /// A local source needs no model store; a remote one requires it.
    #[test]
    fn only_remote_sources_take_a_model_store() {
        let parses = |arguments: &[&str]| {
            Cli::try_parse_from([&["catalog"], arguments].concat()).is_ok()
        };
        let hub = ["hub", "--repository", "o/r", "--revision", "c", "--path", "m.gguf"];
        assert!(parses(&["header", "local", "m.gguf"]));
        assert!(!parses(&["header", "local", "m.gguf", "--model-store", "/m"]));
        assert!(!parses(&[&["header"], &hub[..]].concat()));
        assert!(parses(&[&["header"], &hub[..], &["--model-store", "/m"]].concat()));
        assert!(parses(&["slice", "--output", "s.gguf", "local", "m.gguf"]));
        assert!(parses(&["weight-sample", "--name", "n", "local", "m.gguf"]));
        assert!(!parses(&["weight-sample", "--name", "n", "ollama", "--model", "q", "--digest", "sha256:0"]));
    }
}
