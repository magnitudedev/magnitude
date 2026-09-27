//! Assess the complete release catalog through the service's production assessment path (the
//! measurement job, the environment identity, the automatic pool and the per-target assessor) and
//! report the snapshot with its timings. Development evidence only.
//!
//! ```text
//! catalog_assessment --bundle model-planner-inputs.bundle --cache-root DIR --model-store DIR \
//!     [--snapshot OUT.json]
//! ```
//!
//! The executable is also its own measurement worker (`measurement-worker ...`), exactly as the
//! service binary is.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::future::BoxFuture;
use magnitude_service_contracts::InventoryError;
use magnitude_service_contracts::models::{
    CatalogPackageRemover, ModelAssessment, ModelAssessmentDomainSnapshot,
    ModelAssessmentEntryState, ModelAssessmentPoolState, ModelAssessments, ModelPackageId,
};
use magnitude_service_models::{
    InventoryConfig, ManagedModelDownloads, ManagedModelStore, ModelDomainResolver,
    load_release_catalog, managed_model_services,
};
use magnitude_service_server::assessment::ManagedModelAssessments;
use magnitude_service_server::assessment::assessor::ModelAssessor;
use magnitude_service_server::assessment::measurement::{
    DeviceExclusion, MeasurementJob, MeasurementWorkerArgs, run_measurement_worker,
};
use magnitude_service_server::worker_process::{WorkerLauncher, install_parent_watchdog};
use serde_json::json;

const POLL: Duration = Duration::from_millis(20);

struct NoRemoval;

impl CatalogPackageRemover for NoRemoval {
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
    if arguments.first().map(String::as_str) == Some("measurement-worker") {
        install_parent_watchdog()?;
        run_measurement_worker(MeasurementWorkerArgs {
            device: flag(&arguments, "--device")
                .expect("--device")
                .parse()
                .map_err(|error| anyhow::anyhow!("--device: {error:?}"))?,
            basis_directory: required(&arguments, "--basis-dir"),
            kernel_directory: required(&arguments, "--kernel-dir"),
        })?;
        return Ok(());
    }
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
    let services = managed_model_services(resolver.clone(), downloads, Arc::new(NoRemoval))?;
    let cache = inventory.derived_cache().clone();
    let assessor = Arc::new(ModelAssessor::new(
        inventory,
        resolver,
        release,
        Arc::new(seismic::DeviceCatalog::discover()?),
        MeasurementJob::new(
            WorkerLauncher::current()?,
            cache.measurement_basis_directory(),
            cache.kernel_directory(),
            DeviceExclusion::default(),
        ),
    ));
    let opened = started.elapsed();
    let pool = ManagedModelAssessments::start(
        assessor,
        services.catalog.clone(),
        services.discovered.clone(),
    );
    let pool_started = Instant::now();
    let mut ready_after = None;
    let snapshot = loop {
        let snapshot = pool.snapshot().await?;
        match &snapshot.state {
            ModelAssessmentPoolState::Preparing => {}
            ModelAssessmentPoolState::Failed { failure } => {
                anyhow::bail!(
                    "assessment pool failed: {} {}",
                    failure.code,
                    failure.message
                )
            }
            ModelAssessmentPoolState::Ready { catalog, .. } => {
                ready_after.get_or_insert_with(|| pool_started.elapsed());
                if let ModelAssessmentDomainSnapshot::Available { entries, .. } = catalog
                    && entries
                        .iter()
                        .all(|entry| !matches!(entry.state, ModelAssessmentEntryState::Assessing))
                {
                    break snapshot;
                }
            }
        }
        tokio::time::sleep(POLL).await;
    };
    let settled = pool_started.elapsed();
    let ModelAssessmentPoolState::Ready {
        environment_id,
        catalog: ModelAssessmentDomainSnapshot::Available { entries, .. },
        ..
    } = &snapshot.state
    else {
        unreachable!("the loop returns a ready snapshot with an available catalog");
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
                [ModelAssessment::Incompatible { failure, .. }] => json!({
                    "result": "Incompatible",
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
            "environmentId": environment_id.0,
            "entries": entries.len(),
            "catalogLoadSeconds": catalog_loaded.as_secs_f64(),
            "openSeconds": opened.as_secs_f64(),
            "environmentReadySeconds": ready_after.expect("ready").as_secs_f64(),
            "catalogSettledSeconds": settled.as_secs_f64(),
            "processTotalSeconds": process_started.elapsed().as_secs_f64(),
        })
    );
    if let Some(path) = flag(&arguments, "--snapshot") {
        std::fs::write(path, serde_json::to_vec_pretty(&snapshot)?)?;
    }
    Ok(())
}
