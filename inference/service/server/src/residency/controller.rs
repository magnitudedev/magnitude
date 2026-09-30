//! Model residency on the engine worker (integration spec §10): the driver that loads models in
//! an `inference-worker` process, and the instance operations the API serves.
//!
//! A load resolves the model through the shared resolved-configuration cache (§8.5), waits for
//! memory admission, previews the load on the service's device catalog (the admission check and
//! the load plan), and loads exactly the previewed device in a worker. The worker's measured load
//! progress drives the instance's stage and fraction; readiness is verified against the host's
//! resolution before the instance is Ready.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use futures_util::{FutureExt as _, StreamExt as _};
use magnitude_engine::census::MemoryDomain;
use magnitude_engine::composition::{ReadinessMismatch, ReadyEngine};
use magnitude_engine::error::{LoadError, PreviewError};
use magnitude_engine::host::HostArtifacts;
use magnitude_engine::preview::LoadPreview;
use magnitude_engine::worker::EngineClient;
use magnitude_engine::worker::protocol::{LoadProgress, MemoryObservation};
use magnitude_executor::platform::{DeviceRequest, MemoryReserves};
use magnitude_service_contracts::models::{
    CatalogModelOptimizer, CatalogOptimizationProgress, CatalogPackageRemover,
    InstalledModelPackages as _, ModelId, ModelInstance, ModelInstanceId,
    ModelInstancesInvalidation, ModelInstancesSnapshot, ModelLoadDevice, ModelLoadPlan,
    ModelLoadStage, ModelPackageId, ModelServingConfiguration,
};
use magnitude_service_contracts::{HardwareDeviceId, InventoryError};
use magnitude_service_models::{
    ManagedModelStore, ModelDomainResolver, serving_configuration_fingerprint,
    servable_model_bundle_key_for_bundle,
};
use seismic::{DeviceCatalog, DeviceMemory};

use super::optimization::{self, PreparationJobs};
use super::supervisor::{HostMemoryObserver, MemorySupervisor};
use super::worker::EngineWorker;
use super::{
    LoadingObserver, LowMemory, ModelOperationFailure, ModelResidency, ModelResidencyDriver,
    PreparedResidency, ReleaseControl, ResidencyAcquisition, ResidencyClient, ResidencyGrant,
    ResidencyLease, ResidencyNotification, ResidencyWorker, ResolvedResidencyTarget,
};
use crate::configurations::{ResolvedConfiguration, ResolvedConfigurations};
use crate::hardware::execution_backend;
use crate::memory_domains::instance_allocation;
use crate::worker_process::WorkerLauncher;

/// How long a ready instance with no lease stays resident.
pub const MODEL_IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60 * 60);
/// How long a released worker has to unload and exit before it is killed.
const GRACEFUL_EXIT: std::time::Duration = std::time::Duration::from_secs(2);
/// The same bound for an explicit stop, which must end active execution promptly.
const INTERRUPTED_EXIT: std::time::Duration = std::time::Duration::from_millis(500);

/// A ready instance's engine: its host chat semantics and its worker client. A lease hands it
/// out; the protocol layer binds requests to it.
#[derive(Clone)]
pub struct ResidentEngine {
    pub host: Arc<HostArtifacts>,
    pub client: EngineClient,
    pub context_window_tokens: u32,
}

/// Everything a model load needs besides the model.
pub struct ResidencyEnvironment {
    pub models: Arc<ManagedModelStore>,
    pub model_variants: Arc<ModelDomainResolver>,
    pub configurations: Arc<ResolvedConfigurations>,
    /// The service process's device catalog: previews read it.
    pub catalog: Arc<DeviceCatalog>,
    pub reserves: MemoryReserves,
    /// System-RAM headroom for the memory guard ([`SeismicHostMemory`] on the service catalog).
    pub host_memory: Arc<dyn HostMemoryObserver>,
    /// [`MODEL_IDLE_TIMEOUT`] in the service.
    pub idle_timeout: std::time::Duration,
    pub launcher: WorkerLauncher,
    /// Distinguishes this service run's instance identifiers.
    pub instance_id_namespace: String,
}

#[derive(Clone)]
struct EngineResidency {
    environment: Arc<ResidencyEnvironment>,
    client: ResidencyClient<ResidentEngine>,
    supervisor: MemorySupervisor,
    preparations: PreparationJobs,
    next_instance_id: Arc<AtomicU64>,
}

/// The model instances this service holds: residency, load plans and host-only resolution.
#[derive(Clone)]
pub struct ModelInstances {
    environment: Arc<ResidencyEnvironment>,
    residency: ResidencyClient<ResidentEngine>,
    supervisor: MemorySupervisor,
    preparations: PreparationJobs,
}

impl ModelInstances {
    /// Start the residency actor and the memory supervisor.
    pub fn start(environment: ResidencyEnvironment) -> Self {
        let environment = Arc::new(environment);
        let (client, inbox) = ResidencyClient::channel();
        let host_capacity = environment.catalog.topology().host_pool().capacity_bytes;
        let supervisor = MemorySupervisor::start(
            Arc::clone(&environment.host_memory),
            environment.reserves.for_domain(host_capacity),
            client.notifier(),
        );
        let preparations = PreparationJobs::default();
        let driver = EngineResidency {
            environment: Arc::clone(&environment),
            client: client.clone(),
            supervisor: supervisor.clone(),
            preparations: preparations.clone(),
            next_instance_id: Arc::new(AtomicU64::new(1)),
        };
        tokio::spawn(ModelResidency::new(driver).run(inbox));
        Self {
            environment,
            residency: client,
            supervisor,
            preparations,
        }
    }

    /// A fresh memory observation from the resident worker, when one is resident.
    pub async fn resident_memory(&self) -> Option<MemoryObservation> {
        self.supervisor.observe_resident().await
    }

    /// What loading `model_id` now would use: the engine's preview on this service's catalog.
    pub async fn preview_load(&self, model_id: &str) -> Result<ModelLoadPlan, InventoryError> {
        let target = self.target(model_id)?;
        let resolved = self
            .environment
            .configurations
            .resolve(&target.configuration)
            .await?;
        let preview = preview(&self.environment.catalog, &resolved)
            .await
            .map_err(InventoryError::from)?;
        load_plan(&preview).map_err(InventoryError::from)
    }

    /// Make `model_id` resident without leasing it.
    pub async fn ensure_resident(&self, model_id: &str) -> Result<ModelInstance, InventoryError> {
        match self
            .residency
            .acquire(self.target(model_id)?, ResidencyAcquisition::Warm)
            .await?
        {
            ResidencyGrant::Ready(instance) => Ok(instance),
            ResidencyGrant::Lease(_) => unreachable!("a warm acquisition is granted without a lease"),
        }
    }

    /// Lease `model_id` for inference, loading it if needed.
    pub async fn acquire_for_inference(
        &self,
        model_id: &str,
        progress: Option<LoadingObserver>,
    ) -> Result<ResidencyLease<ResidentEngine>, InventoryError> {
        match self
            .residency
            .acquire(
                self.target(model_id)?,
                ResidencyAcquisition::Inference { progress },
            )
            .await?
        {
            ResidencyGrant::Lease(lease) => Ok(lease),
            ResidencyGrant::Ready(_) => {
                unreachable!("an inference acquisition is granted with a lease")
            }
        }
    }

    /// Lease exactly the ready instance `instance_id`.
    pub async fn lease(
        &self,
        instance_id: ModelInstanceId,
    ) -> Result<ResidencyLease<ResidentEngine>, InventoryError> {
        self.residency.lease(instance_id).await
    }

    pub async fn stop_instance(&self, instance_id: ModelInstanceId) -> Result<(), InventoryError> {
        self.residency.stop(instance_id).await
    }

    pub async fn instances(&self) -> Result<ModelInstancesSnapshot, InventoryError> {
        self.residency.snapshot().await
    }

    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<ModelInstancesInvalidation> {
        self.residency.subscribe()
    }

    /// The model's host chat semantics, for operations that never lease or load (§8.5).
    pub async fn host(&self, model_id: &str) -> Result<Arc<HostArtifacts>, InventoryError> {
        let target = self.target(model_id)?;
        Ok(Arc::clone(
            &self
                .environment
                .configurations
                .resolve(&target.configuration)
                .await?
                .host,
        ))
    }

    fn target(&self, model_id: &str) -> Result<ResolvedResidencyTarget, InventoryError> {
        let model_id = model_id
            .parse::<ModelId>()
            .map_err(|error| InventoryError::InvalidId(error.to_string()))?;
        let configuration = self.environment.model_variants.serving_configuration(&model_id)?;
        Ok(residency_target(model_id, configuration))
    }
}

impl magnitude_service_api::ModelInstanceController for ModelInstances {
    fn preview_load(&self, model_id: String) -> BoxFuture<'_, Result<ModelLoadPlan, InventoryError>> {
        Box::pin(async move { ModelInstances::preview_load(self, &model_id).await })
    }

    fn ensure_resident(
        &self,
        model_id: String,
    ) -> BoxFuture<'_, Result<ModelInstance, InventoryError>> {
        Box::pin(async move { ModelInstances::ensure_resident(self, &model_id).await })
    }

    fn stop_instance(&self, instance_id: ModelInstanceId) -> BoxFuture<'_, Result<(), InventoryError>> {
        Box::pin(ModelInstances::stop_instance(self, instance_id))
    }

    fn instances(&self) -> BoxFuture<'_, Result<ModelInstancesSnapshot, InventoryError>> {
        Box::pin(ModelInstances::instances(self))
    }

    /// The current revision, then every change.
    fn watch_instances(&self) -> BoxStream<'static, ModelInstancesInvalidation> {
        let changes = futures_util::stream::unfold(self.subscribe(), |mut receiver| async move {
            loop {
                match receiver.recv().await {
                    Ok(event) => return Some((event, receiver)),
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => return None,
                }
            }
        });
        let residency = self.residency.clone();
        let current = futures_util::stream::once(async move { residency.snapshot().await })
            .filter_map(|snapshot| async move {
                snapshot.ok().map(|snapshot| ModelInstancesInvalidation {
                    revision: snapshot.revision,
                })
            });
        Box::pin(current.chain(changes))
    }
}

impl CatalogModelOptimizer for ModelInstances {
    fn optimize_catalog_model(
        &self,
        model_id: ModelId,
        progress: Box<dyn Fn(CatalogOptimizationProgress) + Send + Sync>,
    ) -> BoxFuture<'static, ()> {
        let environment = Arc::clone(&self.environment);
        let preparations = self.preparations.clone();
        Box::pin(async move {
            optimization::optimize(&environment, &preparations, &model_id, Arc::from(progress)).await;
        })
    }
}

impl CatalogPackageRemover for ModelInstances {
    fn remove_catalog_packages(
        &self,
        package_ids: Vec<ModelPackageId>,
    ) -> BoxFuture<'_, Result<u64, InventoryError>> {
        Box::pin(async move {
            let mut permits = Vec::with_capacity(package_ids.len());
            for package_id in &package_ids {
                permits.push(self.residency.acquire_package_removal(package_id.clone()).await?);
            }
            self.environment.configurations.evict_packages(&package_ids);
            let mut reclaimed_bytes = 0_u64;
            for package_id in package_ids {
                reclaimed_bytes = reclaimed_bytes.saturating_add(
                    self.environment
                        .models
                        .remove_installed(&package_id)
                        .await?
                        .freed_bytes,
                );
            }
            drop(permits);
            Ok(reclaimed_bytes)
        })
    }
}

/// Instance equality: the serving configuration fingerprint (bundle identity and serving
/// profile). The engine policy is the service's single serving policy.
fn residency_target(
    model_id: ModelId,
    configuration: ModelServingConfiguration,
) -> ResolvedResidencyTarget {
    let configuration_key = serving_configuration_fingerprint(
        &servable_model_bundle_key_for_bundle(&configuration.bundle),
        &configuration.profile,
    );
    ResolvedResidencyTarget {
        model_id,
        configuration,
        configuration_key,
    }
}

pub(super) async fn preview(
    catalog: &Arc<DeviceCatalog>,
    resolved: &Arc<ResolvedConfiguration>,
) -> Result<LoadPreview, ModelOperationFailure> {
    let catalog = Arc::clone(catalog);
    let resolved = Arc::clone(resolved);
    crate::spawn_blocking_traced(move || resolved.manifest.preview(&catalog))
        .await
        .map_err(|error| {
            ModelOperationFailure::new(
                "load_preview_failed",
                format!("load preview task failed: {error}"),
                true,
            )
        })?
        .map_err(preview_failure)
}

pub(super) fn load_plan(preview: &LoadPreview) -> Result<ModelLoadPlan, ModelOperationFailure> {
    Ok(ModelLoadPlan {
        context_window_tokens: context_window_tokens(preview.context_tokens)?,
        required_memory_bytes: preview.required_bytes,
        device: ModelLoadDevice {
            id: HardwareDeviceId::new(preview.device.to_string()),
            backend: execution_backend(preview.backend),
        },
    })
}

fn context_window_tokens(tokens: u64) -> Result<u32, ModelOperationFailure> {
    u32::try_from(tokens).map_err(|_| {
        ModelOperationFailure::new(
            "unsupported_model",
            format!("the served context of {tokens} tokens exceeds the public context range"),
            false,
        )
    })
}

fn preview_failure(error: PreviewError) -> ModelOperationFailure {
    let (code, retryable) = match &error {
        PreviewError::Unsupported(_) => ("unsupported_model", false),
        PreviewError::Device(_) => ("device_unavailable", true),
        PreviewError::MemoryObservationUnavailable { .. } => ("memory_observation_unavailable", true),
        PreviewError::Internal { .. } => ("load_preview_failed", false),
    };
    ModelOperationFailure::new(code, error.to_string(), retryable)
}

/// A worker's load failure. Insufficient memory names the planning reserve of the domain whose
/// claim failed, the line every engine claim keeps above.
fn load_failure(
    error: LoadError,
    catalog: &DeviceCatalog,
    reserves: &MemoryReserves,
    worker: &EngineWorker,
) -> ModelOperationFailure {
    let retryable = error.retryable();
    let code = match &error {
        LoadError::InsufficientMemory { domain, memory, .. } => {
            let planning_reserve_bytes = planning_reserve(catalog, domain, reserves);
            return ModelOperationFailure::LowMemory(LowMemory {
                required_memory_bytes: memory.required,
                allocation_headroom_bytes: memory.available.saturating_add(planning_reserve_bytes),
                system_reserve_bytes: planning_reserve_bytes,
            });
        }
        LoadError::Artifact(_) => "invalid_model_package",
        LoadError::Unsupported(_) => "unsupported_model",
        LoadError::Device(_) => "device_unavailable",
        LoadError::MemoryObservationUnavailable { .. } => "memory_observation_unavailable",
        LoadError::DeviceLost { .. } => "device_lost",
        LoadError::WorkerLost { .. } => "worker_exited",
        LoadError::Internal { .. } => "model_load_failed",
    };
    let message = match &error {
        LoadError::WorkerLost { .. } => match worker.diagnostics() {
            diagnostics if diagnostics.is_empty() => error.to_string(),
            diagnostics => format!("{error}; worker diagnostics: {diagnostics}"),
        },
        _ => error.to_string(),
    };
    ModelOperationFailure::new(code, message, retryable)
}

/// The planning reserve of a memory domain the worker claimed from.
fn planning_reserve(catalog: &DeviceCatalog, domain: &MemoryDomain, reserves: &MemoryReserves) -> u64 {
    let topology = catalog.topology();
    let device = match domain {
        MemoryDomain::HostRam => {
            return reserves.for_domain(topology.host_pool().capacity_bytes).planning_bytes;
        }
        MemoryDomain::DeviceLocal { device } => *device,
    };
    let info = topology
        .devices()
        .iter()
        .find(|info| info.selector == device)
        .expect("a previewed device is in the catalog it was previewed on");
    let capacity = match &info.memory {
        DeviceMemory::Established(memory) => {
            topology
                .pool(memory.allocation_pool)
                .expect("a device's pools belong to its topology")
                .capacity_bytes
        }
        DeviceMemory::Unsupported { .. } => {
            unreachable!("a previewed device has established memory")
        }
    };
    reserves.for_domain(capacity).planning_bytes
}

/// The share of a tuning load's fraction that tuning fills; weight import fills the rest, or all
/// of a load that does not tune.
const TUNING_SHARE: f32 = 0.5;
/// The least fraction a stage reports again after, so a load publishes a bounded number of
/// snapshots however many weights it imports.
const REPORT_STEP: f32 = 0.005;

/// A load's stage and fraction from its worker's progress, by measured work. Monotonic.
#[derive(Default)]
struct LoadProgression {
    tuned: bool,
    reported: Option<(ModelLoadStage, f32)>,
}

impl LoadProgression {
    /// The stage and fraction `progress` advances the load to, or `None` when it is too little
    /// to report.
    fn advance(&mut self, progress: LoadProgress) -> Option<(ModelLoadStage, f32)> {
        let previous = self.reported.map_or(0.0, |(_, fraction)| fraction);
        let (stage, fraction) = match progress {
            LoadProgress::Preparing => (ModelLoadStage::Preparing, 0.0),
            LoadProgress::Tuning { completed, total } => {
                self.tuned = true;
                (ModelLoadStage::Optimizing, TUNING_SHARE * ratio(completed, total))
            }
            LoadProgress::ImportingWeights {
                completed_bytes,
                total_bytes,
            } => {
                let start = if self.tuned { TUNING_SHARE } else { 0.0 };
                (
                    ModelLoadStage::LoadingWeights,
                    start + (1.0 - start) * ratio(completed_bytes, total_bytes),
                )
            }
            LoadProgress::Finalizing => (ModelLoadStage::Finalizing, previous),
        };
        let fraction = fraction.max(previous);
        let reported = self.reported.is_some_and(|(reported, _)| reported == stage)
            && fraction < 1.0
            && fraction - previous < REPORT_STEP;
        if reported {
            return None;
        }
        self.reported = Some((stage, fraction));
        self.reported
    }
}

/// `completed` of `total`, in `[0, 1]`; nothing to do is complete.
fn ratio(completed: u64, total: u64) -> f32 {
    if total == 0 {
        return 1.0;
    }
    (completed as f64 / total as f64).min(1.0) as f32
}

impl EngineResidency {
    fn progress(
        &self,
        instance_id: &ModelInstanceId,
        stage: ModelLoadStage,
        plan: Option<&ModelLoadPlan>,
    ) {
        self.client
            .notifier()
            .send(ResidencyNotification::LoadProgress {
                instance_id: instance_id.clone(),
                stage,
                fraction: 0.0,
                planned_allocation: plan.cloned(),
            });
    }

    async fn load(
        &self,
        instance_id: &ModelInstanceId,
        configuration: ModelServingConfiguration,
    ) -> Result<PreparedResidency<ResidentEngine>, ModelOperationFailure> {
        let environment = &self.environment;
        let resolved = environment.configurations.resolve(&configuration).await?;
        if !self.supervisor.admission_open() {
            self.progress(instance_id, ModelLoadStage::Queued, None);
            tracing::info!("load waits for system memory to recover from pressure");
            self.supervisor.admission().await;
            self.progress(instance_id, ModelLoadStage::Preparing, None);
        }
        let preview = preview(&environment.catalog, &resolved).await?;
        let plan = load_plan(&preview)?;
        self.progress(instance_id, ModelLoadStage::Preparing, Some(&plan));

        // A load tunes whatever its post-install preparation has not stored yet.
        self.preparations
            .stop(&servable_model_bundle_key_for_bundle(&configuration.bundle));
        let spawned = EngineWorker::spawn(&environment.launcher).map_err(|error| {
            ModelOperationFailure::new("worker_spawn_failed", format!("{error:#}"), true)
        })?;
        let worker: Arc<dyn ResidencyWorker> = spawned.worker.clone();
        self.client
            .notifier()
            .send(ResidencyNotification::LoadWorkerStarted {
                instance_id: instance_id.clone(),
                worker,
            });
        let mut manifest = resolved.manifest.clone();
        manifest.device = DeviceRequest::Selector(preview.device);
        let notifier = self.client.notifier();
        let progress_instance = instance_id.clone();
        let progress_plan = plan.clone();
        let connected = crate::spawn_blocking_traced(move || {
            let mut progression = LoadProgression::default();
            spawned.connect(manifest, move |progress| {
                if let Some((stage, fraction)) = progression.advance(progress) {
                    notifier.send(ResidencyNotification::LoadProgress {
                        instance_id: progress_instance.clone(),
                        stage,
                        fraction,
                        planned_allocation: Some(progress_plan.clone()),
                    });
                }
            })
        })
        .await
        .map_err(|error| {
            ModelOperationFailure::new(
                "worker_exited",
                format!("worker connection task failed: {error}"),
                true,
            )
        })?;
        let (worker, connection) = connected.map_err(|(worker, error)| {
            load_failure(error, &environment.catalog, &environment.reserves, &worker)
        })?;

        // The worker read its own package: it must agree with the host's resolution on the
        // package identity, the chat templates and the input modalities. This is the end of the
        // worker's finalizing stage.
        let ready = ReadyEngine::new(Arc::clone(&resolved.host), connection).map_err(|mismatch| {
            let code = match &mismatch {
                ReadinessMismatch::Package { .. } => "worker_package_mismatch",
                ReadinessMismatch::TemplateFingerprint { .. } => "worker_template_mismatch",
                ReadinessMismatch::Modalities { .. } => "worker_modalities_mismatch",
            };
            ModelOperationFailure::new(code, mismatch.to_string(), false)
        })?;
        if ready.ready_info().device != preview.device {
            return Err(ModelOperationFailure::new(
                "worker_device_mismatch",
                format!(
                    "the worker opened {} but the load was previewed on {}",
                    ready.ready_info().device,
                    preview.device
                ),
                false,
            ));
        }
        let context_window_tokens = plan.context_window_tokens;
        let allocation = instance_allocation(
            &environment.catalog.topology(),
            context_window_tokens,
            &ready.ready_info().census,
        );
        let (host, client, _) = ready.into_parts();
        tracing::info!(device = %preview.device, "model ready");
        Ok(PreparedResidency {
            package_ids: resolved.package_ids.clone(),
            allocation,
            worker,
            resident: ResidentEngine {
                host,
                client,
                context_window_tokens,
            },
        })
    }
}

impl ModelResidencyDriver for EngineResidency {
    type Resident = ResidentEngine;

    fn client(&self) -> &ResidencyClient<ResidentEngine> {
        &self.client
    }

    fn next_instance_id(&self) -> ModelInstanceId {
        ModelInstanceId(format!(
            "instance-{}-{}",
            self.environment.instance_id_namespace,
            self.next_instance_id.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn start_model_load(
        &self,
        instance_id: ModelInstanceId,
        target: ResolvedResidencyTarget,
    ) -> Result<tokio::task::JoinHandle<()>, InventoryError> {
        let driver = self.clone();
        Ok(tokio::spawn(async move {
            let result = std::panic::AssertUnwindSafe(driver.load(&instance_id, target.configuration))
                .catch_unwind()
                .await
                .unwrap_or_else(|_| {
                    Err(ModelOperationFailure::new(
                        "model_instance_operation_panicked",
                        "model instance operation panicked",
                        true,
                    ))
                });
            driver.client.load_completion().finish(instance_id, result);
        }))
    }

    fn start_worker_release(
        &self,
        instance_id: ModelInstanceId,
        worker: Arc<dyn ResidencyWorker>,
        interrupt_immediately: bool,
    ) -> ReleaseControl {
        let (interrupt, mut interruption) = tokio::sync::watch::channel(interrupt_immediately);
        let control = ReleaseControl::new(interrupt, Arc::clone(&worker));
        let notifier = self.client.notifier();
        tokio::spawn(async move {
            // Shutdown first in every case: the worker ends its open requests with
            // `ModelUnloaded { Shutdown }` (`model_instance_stopped`) before it exits. An
            // interrupted release then kills it after a short bound instead of the graceful one.
            worker.shutdown();
            let mut deadline = tokio::time::Instant::now()
                + if interrupt_immediately { INTERRUPTED_EXIT } else { GRACEFUL_EXIT };
            let result = loop {
                match worker.try_wait() {
                    Ok(Some(_)) => break Ok(()),
                    Ok(None) if tokio::time::Instant::now() < deadline => {
                        tokio::select! {
                            biased;
                            changed = interruption.changed() => {
                                if changed.is_ok() && *interruption.borrow() {
                                    deadline = deadline.min(tokio::time::Instant::now() + INTERRUPTED_EXIT);
                                }
                            }
                            () = tokio::time::sleep(std::time::Duration::from_millis(10)) => {}
                        }
                    }
                    Ok(None) => {
                        worker.terminate(
                            "worker_unresponsive",
                            "inference worker did not exit after shutdown",
                        );
                        break Ok(());
                    }
                    Err(error) => {
                        worker.terminate(
                            "worker_monitor_failed",
                            "failed to observe inference worker during shutdown",
                        );
                        break Err(ModelOperationFailure::new(
                            "worker_monitor_failed",
                            format!("failed to observe inference worker during shutdown: {error}"),
                            true,
                        ));
                    }
                }
            };
            notifier.send(ResidencyNotification::ReleaseFinished {
                instance_id,
                result,
            });
        });
        control
    }

    fn supervise_worker(
        &self,
        instance_id: ModelInstanceId,
        worker: Arc<dyn ResidencyWorker>,
        resident: ResidentEngine,
    ) {
        self.supervisor.supervise(
            instance_id,
            worker,
            Arc::new(resident.client),
            resident.context_window_tokens,
            self.environment.catalog.topology(),
        );
    }

    fn idle_timeout(&self) -> std::time::Duration {
        self.environment.idle_timeout
    }
}

impl From<ModelOperationFailure> for InventoryError {
    fn from(failure: ModelOperationFailure) -> Self {
        InventoryError::ModelOperation {
            code: failure.code().to_owned(),
            message: failure.message(),
            retryable: failure.retryable(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights(completed_bytes: u64, total_bytes: u64) -> LoadProgress {
        LoadProgress::ImportingWeights {
            completed_bytes,
            total_bytes,
        }
    }

    #[test]
    fn a_tuning_load_fills_half_tuning_then_half_weights() {
        let mut progression = LoadProgression::default();
        let steps = [
            LoadProgress::Preparing,
            LoadProgress::Tuning {
                completed: 0,
                total: 100,
            },
            LoadProgress::Tuning {
                completed: 50,
                total: 100,
            },
            weights(0, 1000),
            weights(500, 1000),
            weights(1000, 1000),
            LoadProgress::Finalizing,
        ];
        let reported = steps
            .into_iter()
            .map(|progress| progression.advance(progress).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            reported,
            [
                (ModelLoadStage::Preparing, 0.0),
                (ModelLoadStage::Optimizing, 0.0),
                (ModelLoadStage::Optimizing, 0.25),
                (ModelLoadStage::LoadingWeights, 0.5),
                (ModelLoadStage::LoadingWeights, 0.75),
                (ModelLoadStage::LoadingWeights, 1.0),
                (ModelLoadStage::Finalizing, 1.0),
            ]
        );
    }

    #[test]
    fn a_load_without_tuning_fills_with_weights() {
        let mut progression = LoadProgression::default();
        progression.advance(LoadProgress::Preparing);
        assert_eq!(
            progression.advance(weights(250, 1000)),
            Some((ModelLoadStage::LoadingWeights, 0.25))
        );
    }

    #[test]
    fn progress_reports_in_bounded_steps_and_never_regresses() {
        let mut progression = LoadProgression::default();
        progression.advance(weights(0, 100_000));
        // Less than a report step within the same stage is not reported.
        assert_eq!(progression.advance(weights(100, 100_000)), None);
        assert_eq!(
            progression.advance(weights(1000, 100_000)),
            Some((ModelLoadStage::LoadingWeights, 0.01))
        );
        // A unit found unexpectedly to search raises the tuning total; the fraction holds.
        let mut progression = LoadProgression::default();
        progression.advance(LoadProgress::Tuning {
            completed: 50,
            total: 100,
        });
        assert_eq!(
            progression.advance(LoadProgress::Tuning {
                completed: 60,
                total: 200,
            }),
            None
        );
    }
}
