//! Assess the complete release catalog through the service's production assessment path (the
//! environment and its identity, the automatic pool and the per-target assessor) and report the
//! snapshot with its timings. Development evidence only.
//!
//! ```text
//! catalog_assessment --bundle model-planner-inputs.bundle --cache-root DIR --model-store DIR \
//!     [--snapshot OUT.json]
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::future::BoxFuture;
use magnitude_service_contracts::InventoryError;
use magnitude_service_contracts::models::{
    CatalogModelOptimizer, CatalogOptimizationProgress, CatalogPackageRemover, ModelAssessment,
    ModelAssessmentDomainSnapshot, ModelAssessmentEntryState, ModelAssessments, ModelId,
    ModelPackageId,
};
use magnitude_service_models::{
    InventoryConfig, ManagedModelDownloads, ManagedModelStore, ModelDomainResolver,
    load_release_catalog, managed_model_services,
};
use magnitude_service_server::assessment::ManagedModelAssessments;
use magnitude_service_server::assessment::assessor::ModelAssessor;
use magnitude_service_server::assessment::environment::AssessmentEnvironment;
use serde_json::json;

const POLL: Duration = Duration::from_millis(20);

struct NoCatalogEffects;

impl CatalogModelOptimizer for NoCatalogEffects {
    fn optimize_catalog_model(
        &self,
        _model_id: ModelId,
        _progress: Box<dyn Fn(CatalogOptimizationProgress) + Send + Sync>,
    ) -> BoxFuture<'static, ()> {
        Box::pin(async {})
    }
}

impl CatalogPackageRemover for NoCatalogEffects {
    fn remove_catalog_packages(
        &self,
        _package_ids: Vec<ModelPackageId>,
    ) -> BoxFuture<'_, Result<u64, InventoryError>> {
        Box::pin(async {
            Err(InventoryError::Unsupported(
                "no removal in this harness".into(),
            ))
        })
    }
}

fn flag(arguments: &[String], name: &str) -> Option<String> {
    arguments
        .iter()
        .position(|argument| argument == name)
        .and_then(|index| arguments.get(index + 1))
        .cloned()
}

fn required(arguments: &[String], name: &str) -> PathBuf {
    PathBuf::from(flag(arguments, name).unwrap_or_else(|| panic!("{name} is required")))
}

fn main() -> anyhow::Result<()> {
    let process_started = Instant::now();
    if let Ok(filter) = tracing_subscriber::EnvFilter::try_from_default_env() {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .try_init()
            .map_err(|error| anyhow::anyhow!("{error}"))?;
    }
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(assess_catalog(arguments, process_started))
}

async fn assess_catalog(arguments: Vec<String>, process_started: Instant) -> anyhow::Result<()> {
    let bundle = required(&arguments, "--bundle");
    let started = Instant::now();
    let release = Arc::new(load_release_catalog(&bundle)?);
    let catalog_loaded = started.elapsed();
    let mut config = InventoryConfig::with_roots(
        required(&arguments, "--model-store"),
        required(&arguments, "--cache-root"),
    )?;
    config.catalog_models = release.catalog().models.clone();
    let inventory = Arc::new(ManagedModelStore::open(config).await?);
    let resolver = ModelDomainResolver::new(inventory.clone(), release.catalog().clone());
    let downloads = Arc::new(ManagedModelDownloads::open(inventory.clone()).await?);
    let services = managed_model_services(resolver.clone(), downloads, Arc::new(NoCatalogEffects), Arc::new(NoCatalogEffects))?;
    let environment = Arc::new(AssessmentEnvironment::establish(
        &seismic::DeviceCatalog::discover()?,
    )?);
    let environment_established = started.elapsed();
    let assessor = Arc::new(ModelAssessor::new(inventory, resolver, release, environment));
    let pool = ManagedModelAssessments::start(
        assessor,
        services.catalog.clone(),
        services.discovered.clone(),
    );
    let pool_started = Instant::now();
    let snapshot = loop {
        let snapshot = pool.snapshot().await?;
        if let ModelAssessmentDomainSnapshot::Available { entries, .. } = &snapshot.catalog
            && entries
                .iter()
                .all(|entry| !matches!(entry.state, ModelAssessmentEntryState::Assessing))
        {
            break snapshot;
        }
        tokio::time::sleep(POLL).await;
    };
    let settled = pool_started.elapsed();
    let ModelAssessmentDomainSnapshot::Available { entries, .. } = &snapshot.catalog else {
        unreachable!("the loop returns an available catalog");
    };
    for entry in entries {
        let summary = match &entry.state {
            ModelAssessmentEntryState::Assessed { profiles, .. } => match profiles.as_slice() {
                [
                    ModelAssessment::Fits {
                        profile,
                        memory,
                        performance,
                        ..
                    },
                ] => json!({
                    "result": "Fits",
                    "context": profile.context_length,
                    "requiredBytes": memory.iter().map(|domain| domain.required_bytes).collect::<Vec<_>>(),
                    "tokensPerSecond": performance
                        .iter()
                        .map(|sample| (sample.context_tokens, (sample.estimated_tokens_per_second * 10.0).round() / 10.0))
                        .collect::<Vec<_>>(),
                }),
                [
                    ModelAssessment::DoesNotFit {
                        profile,
                        limiting_resource,
                        deficit_bytes,
                        ..
                    },
                ] => json!({
                    "result": "DoesNotFit",
                    "context": profile.context_length,
                    "limitingResource": limiting_resource,
                    "deficitBytes": deficit_bytes,
                }),
                [ModelAssessment::Unsupported { failure, .. }] => json!({
                    "result": "Unsupported",
                    "code": failure.code,
                    "message": failure.message.chars().take(160).collect::<String>(),
                }),
                other => json!({ "result": "unexpected", "profiles": other.len() }),
            },
            ModelAssessmentEntryState::Dropped => json!({ "result": "Dropped" }),
            ModelAssessmentEntryState::Assessing => unreachable!("settled"),
        };
        println!("{} {}", entry.subject.model_id().as_str(), summary);
    }
    println!(
        "{}",
        json!({
            "environmentId": snapshot.environment_id.0,
            "entries": entries.len(),
            "catalogLoadSeconds": catalog_loaded.as_secs_f64(),
            "environmentEstablishedSeconds": environment_established.as_secs_f64(),
            "catalogSettledSeconds": settled.as_secs_f64(),
            "processTotalSeconds": process_started.elapsed().as_secs_f64(),
        })
    );
    if let Some(path) = flag(&arguments, "--snapshot") {
        std::fs::write(path, serde_json::to_vec_pretty(&snapshot)?)?;
    }
    Ok(())
}
