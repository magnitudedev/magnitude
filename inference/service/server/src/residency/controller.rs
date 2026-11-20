//! Model residency on the engine worker (integration spec §10): the driver that loads models in
//! an `inference-worker` process, and the instance operations the API serves.
//!
//! A load resolves the model through the shared resolved-configuration cache (§8.5), previews the
//! load on the serving device in the service's device catalog (the admission check and the load
//! plan), waiting first, without the device, for the machine's standing figures its plan is priced
//! on, takes its share of the device, previews again if it waited for either, and loads exactly
//! that device in a worker. The worker's
//! measured load progress drives the instance's stage and fraction; readiness is verified against
//! the host's resolution before the instance is Ready, and the Ready instance records the device
//! the worker opened.

use magnitude_serving::source::{ModelLoadDetails, KernelTuningWork, KernelTuningOperation};
use magnitude_engine::worker::protocol::TuningOperation;
use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use futures_util::{FutureExt as _, StreamExt as _};
use magnitude_engine::composition::{ReadinessMismatch, ReadyEngine};
use magnitude_engine::error::{LoadError, PreviewError, RequestError};
use magnitude_engine::host::HostArtifacts;
use magnitude_engine::preview::LoadPreview;
use magnitude_engine::worker::{
    AcquiredDevice, EngineClient, Preparation, PricedError, PricedFailure, load_prepared,
    run_priced,
};
use magnitude_engine::worker::protocol::{LoadProgress, MemoryObservation, RoomAnswer};
use magnitude_service_contracts::models::{
    CatalogModelOptimizer, CatalogOptimizationProgress, CatalogPackageRemover, LoadedPackages,
    InstalledModelPackages as _, MemoryShortage, ModelAssessment, ModelAssessmentDomainSnapshot,
    CatalogModelSelection, ModelAssessmentEntryState, ModelAssessmentSubject, ModelAssessments,
    ModelAssessmentsSnapshot, ModelId, ModelInstance,
    ModelInstanceId,
    ModelInstancesInvalidation, ModelInstancesSnapshot, ModelLoadDevice, ModelLoadPlan,
    ModelLoadStage, ModelPackageId, ModelServingConfiguration, ServingProfile,
};
use magnitude_service_contracts::{HardwareDeviceId, InventoryError};
use magnitude_service_models::{
    ManagedModelStore, ModelDomainResolver, serving_configuration_fingerprint,
    servable_model_bundle_key_for_bundle,
};
use seismic::{BackendName, Devices, ExecutorSelector};

use super::optimization::{self, PreparationJobs};
use super::supervisor::MemorySupervisor;
use super::worker::{EngineWorker, SpawnedWorker};
use super::{
    LoadSequence, LoadingObserver, ModelOperationFailure, ModelResidency, ModelResidencyDriver,
    PreparedResidency, ReleaseControl, ResidencyAcquisition, ResidencyClient, ResidencyGrant,
    ResidencyLease, ResidencyNotification, ResidencyWorker, ResolvedResidencyTarget,
};
use crate::configurations::{HeldConfiguration, ResolvedConfiguration, ResolvedConfigurations};
use crate::device_lock::{DeviceHold, DeviceLock, DeviceShare, IdleResident, ResidentRoom};
use crate::hardware::execution_backend;
use crate::memory_domains::instance_allocation;
use crate::worker_process::WorkerLauncher;

/// How long a ready instance with no lease stays resident.
pub const MODEL_IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60 * 60);
/// How long a released worker has to answer its shutdown before it is killed: to end every
/// request it holds and make every durable effect of its load. Its process then ends at once,
/// and the system's release of what it held is waited for however long it takes, as a kill's
/// is: a kill cannot shorten it.
pub(crate) const GRACEFUL_EXIT: std::time::Duration = std::time::Duration::from_secs(2);
/// The same bound for an explicit stop, which must end active execution promptly.
const INTERRUPTED_ANSWER: std::time::Duration = std::time::Duration::from_millis(500);

/// A ready instance's engine: its host chat semantics and its worker client. A lease hands it
/// out; the protocol layer binds requests to it. It holds its resolved configuration, so
/// host-only operations on the resident model find it in the shared cache however many other
/// models were resolved since.
#[derive(Clone)]
pub struct ResidentEngine {
    pub host: Arc<HostArtifacts>,
    pub client: EngineClient,
    pub context_window_tokens: u32,
    _configuration: Arc<HeldConfiguration>,
}

/// Everything a model load needs besides the model.
pub struct ResidencyEnvironment {
    pub models: Arc<ManagedModelStore>,
    pub model_variants: Arc<ModelDomainResolver>,
    pub configurations: Arc<ResolvedConfigurations>,
    /// The service process's device catalog: previews read it.
    pub catalog: Arc<Devices>,
    /// [`MODEL_IDLE_TIMEOUT`] in the service.
    pub idle_timeout: std::time::Duration,
    pub launcher: WorkerLauncher,
    /// Shared by every load and optimization for its lifetime, against the measurement job.
    pub device: DeviceLock,
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
    /// Residency's share of the device while it has demand
    /// ([`ModelResidencyDriver::demand_ended`]): taken by the first load, held across each release
    /// and the next queued load, so the device is never free between them.
    demand: Arc<std::sync::Mutex<Option<Arc<DeviceShare>>>>,
}

/// The model instances this service holds: residency, load plans and host-only resolution.
#[derive(Clone)]
pub struct ModelInstances {
    environment: Arc<ResidencyEnvironment>,
    residency: ResidencyClient<ResidentEngine>,
    supervisor: MemorySupervisor,
    preparations: PreparationJobs,
    /// The assessment pool whose verdicts admit loads. It is built from these instances, so it
    /// is attached once both exist.
    assessments: Arc<std::sync::OnceLock<Arc<dyn ModelAssessments>>>,
}

impl ModelInstances {
    /// Start the residency actor and the resident-worker supervisor.
    pub fn start(environment: ResidencyEnvironment) -> Self {
        let environment = Arc::new(environment);
        let (client, inbox) = ResidencyClient::channel();
        let supervisor = MemorySupervisor::new(client.notifier());
        let preparations = PreparationJobs::default();
        let driver = EngineResidency {
            environment: Arc::clone(&environment),
            client: client.clone(),
            supervisor: supervisor.clone(),
            preparations: preparations.clone(),
            next_instance_id: Arc::new(AtomicU64::new(1)),
            demand: Arc::default(),
        };
        magnitude_service_utils::owner::spawn("model residency", ModelResidency::new(driver).run(inbox));
        Self {
            environment,
            residency: client,
            supervisor,
            preparations,
            assessments: Arc::default(),
        }
    }

    /// Admit loads by `assessments`' verdicts from now on.
    pub fn attach_assessments(&self, assessments: Arc<dyn ModelAssessments>) {
        let _ = self.assessments.set(assessments);
    }

    /// The residency target serving `model_id`: its configuration at the context its assessment
    /// serves it at, so a change of that context is a different instance. A load the assessment
    /// says cannot fit or the engine cannot run is refused before the residency replaces anything.
    /// A model still being assessed waits for its verdict, since nothing else bounds the context
    /// it would serve. A model whose assessment failed, or a service without an assessment pool,
    /// serves the full context, guarded only by the engine's own resource plan.
    async fn serving_target(&self, model_id: &str) -> Result<ResolvedResidencyTarget, InventoryError> {
        let target = self.target(model_id)?;
        let Some(assessments) = self.assessments.get() else {
            return Ok(target);
        };
        let mut changes = assessments.watch();
        let deadline = tokio::time::Instant::now() + ASSESSMENT_WAIT;
        loop {
            let snapshot = assessments.snapshot().await?;
            match served_profile(&snapshot, &target.model_id, &target.configuration.profile) {
                ServedProfile::Served(profile) => {
                    return Ok(residency_target(
                        target.model_id,
                        ModelServingConfiguration {
                            profile,
                            ..target.configuration
                        },
                    ));
                }
                ServedProfile::Refused(refusal) => return Err(refusal),
                ServedProfile::Unassessed => return Ok(target),
                ServedProfile::Assessing => {}
            }
            match tokio::time::timeout_at(deadline, changes.next()).await {
                Ok(Some(_)) => {}
                Ok(None) | Err(_) => {
                    return Err(InventoryError::NotReady(format!(
                        "{} is still being assessed for this computer",
                        target.model_id
                    )));
                }
            }
        }
    }

    /// The residency loads `target` unless it is resident, and the load's Prepare searches what
    /// its bundle's catalog optimization would: the optimization leaves the device to it while
    /// the request lasts. Taken before the residency releases another model, whose release frees
    /// the device.
    fn requested(&self, target: &ResolvedResidencyTarget) -> optimization::LoadRequest {
        self.preparations
            .request(servable_model_bundle_key_for_bundle(&target.configuration.bundle))
    }

    /// A fresh memory observation from the resident worker, when one is resident.
    pub async fn resident_memory(&self) -> Result<Option<MemoryObservation>, RequestError> {
        self.supervisor.observe_resident().await
    }

    /// What loading `model_id` now would use: the engine's preview on this service's catalog.
    pub fn preview_load(&self, model_id: String) -> BoxFuture<'_, Result<ModelLoadPlan, InventoryError>> {
        Box::pin(async move {
            let target = self.serving_target(&model_id).await?;
            let resolved = self
                .environment
                .configurations
                .resolve(&target.configuration)
                .await?;
            let preview = preview(&self.environment.catalog, &resolved.configuration)
                .await
                .map_err(|error| InventoryError::from(preview_failure(error)))?;
            load_plan(&preview).map_err(InventoryError::from)
        })
    }

    /// Make `model_id` resident without leasing it.
    pub fn ensure_resident(&self, model_id: String) -> BoxFuture<'_, Result<ModelInstance, InventoryError>> {
        Box::pin(async move {
            let target = self.serving_target(&model_id).await?;
            let _request = self.requested(&target);
            match self
                .residency
                .acquire(target, ResidencyAcquisition::Warm)
                .await?
            {
                ResidencyGrant::Ready(instance) => Ok(instance),
                ResidencyGrant::Lease(_) => unreachable!("a warm acquisition is granted without a lease"),
            }
        })
    }

    /// Lease `model_id` for inference, loading it if needed.
    pub async fn acquire_for_inference(
        &self,
        model_id: &str,
        progress: Option<LoadingObserver>,
    ) -> Result<ResidencyLease<ResidentEngine>, InventoryError> {
        let target = self.serving_target(model_id).await?;
        let _request = self.requested(&target);
        match self
            .residency
            .acquire(
                target,
                ResidencyAcquisition::Inference { progress },
            )
            .await?
        {
            ResidencyGrant::Lease(lease) => Ok(self.admitted(lease).await),
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
        let lease = self.residency.lease(instance_id).await?;
        Ok(self.admitted(lease).await)
    }

    /// `lease` once the measurement it preempted beside its idle model is gone
    /// ([`DeviceLock::admit_request`]): no request is served while one is timed.
    async fn admitted(&self, lease: ResidencyLease<ResidentEngine>) -> ResidencyLease<ResidentEngine> {
        self.environment.device.admit_request().await;
        lease
    }

    pub fn stop_instance(
        &self,
        instance_id: ModelInstanceId,
    ) -> BoxFuture<'_, Result<(), InventoryError>> {
        Box::pin(self.residency.stop(instance_id))
    }

    pub fn instances(&self) -> BoxFuture<'_, Result<ModelInstancesSnapshot, InventoryError>> {
        Box::pin(self.residency.snapshot())
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
        let model_id = self.environment.model_variants.canonical_model_id(model_id)?;
        let configuration = self.environment.model_variants.serving_configuration(&model_id)?;
        Ok(residency_target(model_id, configuration))
    }
}

impl magnitude_service_api::ModelInstanceController for ModelInstances {
    fn preview_load(&self, model_id: String) -> BoxFuture<'_, Result<ModelLoadPlan, InventoryError>> {
        ModelInstances::preview_load(self, model_id)
    }

    fn ensure_resident(
        &self,
        model_id: String,
    ) -> BoxFuture<'_, Result<ModelInstance, InventoryError>> {
        ModelInstances::ensure_resident(self, model_id)
    }

    fn stop_instance(&self, instance_id: ModelInstanceId) -> BoxFuture<'_, Result<(), InventoryError>> {
        ModelInstances::stop_instance(self, instance_id)
    }

    fn instances(&self) -> BoxFuture<'_, Result<ModelInstancesSnapshot, InventoryError>> {
        ModelInstances::instances(self)
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
        loaded: LoadedPackages,
    ) -> BoxFuture<'_, Result<u64, InventoryError>> {
        let (models, residency) = (Arc::clone(&self.environment.models), self.residency.clone());
        // On a task of its own: a removal runs to its end whatever its caller does.
        let removal = tokio::spawn(async move {
            let release = residency.release_packages(package_ids.clone(), loaded);
            models.remove_installed_packages(&package_ids, Box::pin(release)).await
        });
        Box::pin(async move {
            match removal.await {
                Ok(removed) => removed,
                Err(ended) => match ended.try_into_panic() {
                    Ok(panic) => std::panic::resume_unwind(panic),
                    Err(_) => Err(InventoryError::Internal("the removal's task was cancelled".to_owned())),
                },
            }
        })
    }
}

/// How long a load waits for its model's assessment, which reads only headers.
const ASSESSMENT_WAIT: std::time::Duration = std::time::Duration::from_secs(600);

/// What the assessment of `model_id` at `profile`, its ceiling, says it serves.
enum ServedProfile {
    /// It fits at this context.
    Served(ServingProfile),
    /// It fits at no context, or the engine cannot run it, with the engine's reasons.
    Refused(InventoryError),
    /// Its assessment is still to come.
    Assessing,
    /// Its assessments ended without a verdict for the profile: one failed.
    Unassessed,
}

fn served_profile(
    snapshot: &ModelAssessmentsSnapshot,
    model_id: &ModelId,
    profile: &ServingProfile,
) -> ServedProfile {
    let entries = [&snapshot.catalog, &snapshot.custom]
        .into_iter()
        .flat_map(|domain| match domain {
            ModelAssessmentDomainSnapshot::Available { entries, .. } => entries.as_slice(),
            ModelAssessmentDomainSnapshot::Pending { .. } => &[],
        })
        .filter(|entry| entry.subject.model_id() == model_id)
        // A catalog model is served as its effective selection; its desired one may be another
        // bundle at the same profile.
        .filter(|entry| {
            !matches!(
                entry.subject,
                ModelAssessmentSubject::Catalog {
                    selection: CatalogModelSelection::Desired,
                    ..
                }
            )
        })
        .collect::<Vec<_>>();
    let assessed = entries
        .iter()
        .filter_map(|entry| match &entry.state {
            ModelAssessmentEntryState::Assessed { profiles, .. } => Some(profiles),
            _ => None,
        })
        .flatten()
        .find_map(|assessment| match assessment {
            ModelAssessment::Fits {
                profile: assessed,
                served,
                ..
            } if assessed == profile => Some(ServedProfile::Served(served.clone())),
            ModelAssessment::DoesNotFit {
                profile: assessed,
                shortfall,
                deficit_bytes,
                ..
            } if assessed == profile => Some(ServedProfile::Refused(InventoryError::DoesNotFit(
                format!(
                    "{model_id} needs {} more memory than this computer can provide, even for a {} context",
                    memory_size(*deficit_bytes),
                    context_size(shortfall.context_length)
                ),
            ))),
            ModelAssessment::Unsupported {
                profile: assessed, reasons, detail,
            } if assessed.as_ref().is_none_or(|assessed| assessed == profile) => Some(ServedProfile::Refused(InventoryError::ModelUnsupported {
                message: format!("Magnitude can't run {model_id}: {detail}"),
                reasons: reasons.clone(),
            })),
            _ => None,
        });
    // Without a verdict for the profile, the load waits while any of the model's assessments is
    // still to come (none listed yet, or one assessing); once all have ended without one, it
    // has none to wait for.
    match assessed {
        Some(served) => served,
        None if entries.is_empty()
            || entries
                .iter()
                .any(|entry| matches!(entry.state, ModelAssessmentEntryState::Assessing)) =>
        {
            ServedProfile::Assessing
        }
        None => ServedProfile::Unassessed,
    }
}

/// A context as people name it: 32K for 32,768 tokens.
fn context_size(tokens: u32) -> String {
    if tokens >= 1024 && tokens % 1024 == 0 {
        format!("{}K", tokens / 1024)
    } else {
        format!("{tokens}-token")
    }
}

/// `bytes` as the clients show a shortfall: binary units from MB, one decimal, rounded up.
fn memory_size(bytes: u64) -> String {
    const MEBIBYTE: f64 = 1024.0 * 1024.0;
    let tenths = |amount: f64| (amount * 10.0).ceil() / 10.0;
    let megabytes = tenths(bytes as f64 / MEBIBYTE);
    if megabytes < 1024.0 {
        format!("{megabytes:.1} MB")
    } else {
        format!("{:.1} GB", tenths(bytes as f64 / (MEBIBYTE * 1024.0)))
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
    catalog: &Arc<Devices>,
    resolved: &Arc<ResolvedConfiguration>,
) -> Result<LoadPreview, PreviewError> {
    let catalog = Arc::clone(catalog);
    let resolved = Arc::clone(resolved);
    magnitude_service_utils::blocking::run(move || resolved.manifest.preview(&catalog)).await
}

/// Ask the measurement job for pending figures via `defer` and wait for the next device release,
/// holding nothing of the device. Subscribes to `device.releases()` before `defer` so a release
/// that finishes during `defer` is never missed.
pub(super) async fn settle_prices(
    device: &DeviceLock,
    defer: impl std::future::Future<Output = Result<(), ModelOperationFailure>>,
) -> Result<(), ModelOperationFailure> {
    let mut releases = device.releases();
    defer.await?;
    device.requested();
    releases.released().await;
    Ok(())
}

/// Ask the measurement job for the standing figures `resolved`'s plan is priced on.
pub(super) async fn request_prices(
    catalog: &Arc<Devices>,
    resolved: &Arc<ResolvedConfiguration>,
) -> Result<(), ModelOperationFailure> {
    let (catalog, resolved) = (Arc::clone(catalog), Arc::clone(resolved));
    magnitude_service_utils::blocking::run(move || resolved.manifest.request_prices(&catalog))
        .await
        .map_err(|reason| ModelOperationFailure::new("measurement_request_failed", reason, true))
}

pub(super) fn load_plan(preview: &LoadPreview) -> Result<ModelLoadPlan, ModelOperationFailure> {
    Ok(ModelLoadPlan {
        context_window_tokens: context_window_tokens(preview.context_tokens)?,
        required_memory_bytes: preview.required_bytes,
        device: load_device(preview.device, preview.backend),
    })
}

/// An engine executor as the instance state names it.
fn load_device(device: ExecutorSelector, backend: BackendName) -> ModelLoadDevice {
    ModelLoadDevice {
        id: HardwareDeviceId::new(device.to_string()),
        backend: execution_backend(backend),
    }
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
        PreviewError::WeightBound(bound) => return ModelOperationFailure::WeightBound(*bound),
        PreviewError::PricesPending { .. } => ("prices_pending", true),
        PreviewError::Internal { .. } => ("load_preview_failed", false),
    };
    ModelOperationFailure::new(code, error.to_string(), retryable)
}

/// A worker's load failure. A memory refusal is a shortage: blocked by the bytes the engine's
/// claim found available above the domain's reserve.
fn load_failure(error: LoadError, worker: &EngineWorker) -> ModelOperationFailure {
    let retryable = error.retryable();
    let code = match &error {
        LoadError::InsufficientMemory { memory, .. } => {
            return ModelOperationFailure::MemoryShortage(MemoryShortage::Blocked {
                required_bytes: memory.required,
                available_bytes: memory.available,
            });
        }
        LoadError::WeightBound(bound) => return ModelOperationFailure::WeightBound(*bound),
        LoadError::Artifact(_) => "invalid_model_package",
        LoadError::Unsupported(_) => "unsupported_model",
        LoadError::Device(_) => "device_unavailable",
        LoadError::MemoryObservationUnavailable { .. } => "memory_observation_unavailable",
        LoadError::DeviceLost { .. } => "device_lost",
        LoadError::WorkerLost { .. } => "worker_exited",
        LoadError::Configuration { .. } => "invalid_model_configuration",
        LoadError::PricesPending { .. } => "prices_pending",
        LoadError::Internal { .. } => "model_load_failed",
    };
    let message = match &error {
        LoadError::WorkerLost { .. } => super::ResidencyWorker::loss(worker, &error),
        _ => error.to_string(),
    };
    ModelOperationFailure::new(code, message, retryable)
}

/// The share of a load's fraction that a Prepare worker's tuning fills when the load needed
/// one; the Load worker's weight import fills the rest, or all of a load that needed none.
const TUNING_SHARE: f32 = 0.5;
/// The least fraction a stage reports again after, so a load publishes a bounded number of
/// snapshots however many weights it imports.
const REPORT_STEP: f32 = 0.005;

/// A load's stage and fraction from the progress of its workers in turn (the Load; or, with
/// tuning records missing, a Load that ends unprepared, the Prepare and the Load), by measured
/// work. Monotonic.
#[derive(Default)]
struct LoadProgression {
    tuned: bool,
    reported: Option<(ModelLoadStage, f32)>,
    details: Option<ModelLoadDetails>,
}

impl LoadProgression {
    /// The stage and fraction `progress` advances the load to, or `None` when it is too little
    /// to report.
    fn advance(&mut self, progress: LoadProgress) -> Option<(ModelLoadStage, f32, Option<ModelLoadDetails>)> {
        let previous = self.reported.map_or(0.0, |(_, fraction)| fraction);
        let mut details = None;
        let (stage, fraction) = match progress {
            // The second Load worker prepares again after the Prepare worker tuned: the load
            // does not return to `preparing`.
            LoadProgress::Preparing if self.tuned => return None,
            LoadProgress::Preparing => (ModelLoadStage::Preparing, 0.0),
            LoadProgress::Tuning { completed, total, work, measured_configurations, formed_programs } => {
                self.tuned = true;
                details = Some(ModelLoadDetails::Tuning {
                    elapsed_ms: completed,
                    budget_ms: total,
                    work: work.map(|work| KernelTuningWork {
                        entry: work.entry,
                        operation: match work.operation {
                            TuningOperation::Discovery => KernelTuningOperation::Discovery,
                            TuningOperation::Form => KernelTuningOperation::Form,
                            TuningOperation::Measure => KernelTuningOperation::Measure,
                            TuningOperation::Promote => KernelTuningOperation::Promote,
                            TuningOperation::Confirm => KernelTuningOperation::Confirm,
                        },
                    }),
                    measured_configurations,
                    formed_programs,
                });
                (ModelLoadStage::Optimizing, TUNING_SHARE * ratio(completed, total))
            }
            LoadProgress::ImportingWeights {
                completed_bytes,
                total_bytes,
            } => {
                details = Some(ModelLoadDetails::Weights { completed_bytes, total_bytes });
                let start = if self.tuned { TUNING_SHARE } else { 0.0 };
                (
                    ModelLoadStage::LoadingWeights,
                    start + (1.0 - start) * ratio(completed_bytes, total_bytes),
                )
            }
            LoadProgress::Finalizing => (ModelLoadStage::Finalizing, previous),
        };
        let fraction = fraction.max(previous);
        // Do not throttle away a selected operation or a fresh count snapshot.
        let changed_work = match (&self.details, &details) {
            (Some(ModelLoadDetails::Tuning { work: a, measured_configurations: am, formed_programs: ap, .. }),
             Some(ModelLoadDetails::Tuning { work: b, measured_configurations: bm, formed_programs: bp, .. })) =>
                a != b || am != bm || ap != bp,
            _ => false,
        };
        let reported = !changed_work && self.reported.is_some_and(|(reported, _)| reported == stage)
            && fraction < 1.0
            && fraction - previous < REPORT_STEP;
        if reported {
            return None;
        }
        self.reported = Some((stage, fraction));
        self.details = details.clone();
        Some((stage, fraction, details))
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
                details: None,
                planned_allocation: plan.cloned(),
            });
    }

    /// Spawn an inference worker for `instance_id`'s load, sharing the load's `device`, and
    /// hand it to the residency, which terminates it when the load is stopped.
    fn spawn_load_worker(
        &self,
        instance_id: &ModelInstanceId,
        device: &Arc<DeviceShare>,
    ) -> Result<SpawnedWorker, ModelOperationFailure> {
        let spawned = EngineWorker::spawn(
            &self.environment.launcher,
            DeviceHold::Share(Arc::clone(device)),
        )
        .map_err(|error| {
            ModelOperationFailure::new("worker_spawn_failed", format!("{error:#}"), true)
        })?;
        let worker: Arc<dyn ResidencyWorker> = spawned.worker.clone();
        self.client
            .notifier()
            .send(ResidencyNotification::LoadWorkerStarted {
                instance_id: instance_id.clone(),
                worker,
            });
        Ok(spawned)
    }

    /// Defer `instance_id`'s load until the machine's standing figures its plan is priced on are
    /// stored: it shows `Queued`, gives up residency's share of the device, which a load after a
    /// release kept, so the measurement job can take the device, and asks the job for the
    /// figures of its model's store.
    async fn defer_unpriced(
        &self,
        instance_id: &ModelInstanceId,
        resolved: &Arc<ResolvedConfiguration>,
    ) -> Result<(), ModelOperationFailure> {
        self.progress(instance_id, ModelLoadStage::Queued, None);
        self.demand.lock().expect("device demand").take();
        request_prices(&self.environment.catalog, resolved).await
    }

    fn load<'a>(
        &'a self,
        instance_id: &'a ModelInstanceId,
        configuration: ModelServingConfiguration,
        sequence: LoadSequence,
    ) -> BoxFuture<'a, Result<PreparedResidency<ResidentEngine>, ModelOperationFailure>> {
        Box::pin(async move {
            let environment = &self.environment;
            let resolved = Arc::new(environment.configurations.resolve(&configuration).await?);
            let manifest = resolved.manifest.clone();
            let bundle_key = servable_model_bundle_key_for_bundle(&configuration.bundle);

            let ((worker, connection), preview, _device) = run_priced(
                || preview(&environment.catalog, &resolved.configuration),
                || settle_prices(&environment.device, self.defer_unpriced(instance_id, &resolved.configuration)),
                |_| async {
                    let held = self.demand.lock().expect("device demand").clone();
                    match held {
                        Some(device) => {
                            environment.device.admit_request().await;
                            Ok(AcquiredDevice::immediate(device))
                        }
                        None => {
                            let (device, waited) = match environment.device.try_share() {
                                Some(device) => (Arc::new(device), false),
                                None => {
                                    self.progress(instance_id, ModelLoadStage::Queued, None);
                                    tracing::info!(
                                        "load preempts the job holding the device and waits for its kill"
                                    );
                                    let device = Arc::new(environment.device.share().await);
                                    self.progress(instance_id, ModelLoadStage::Preparing, None);
                                    (device, true)
                                }
                            };
                            *self.demand.lock().expect("device demand") = Some(Arc::clone(&device));
                            Ok(AcquiredDevice { device, waited })
                        }
                    }
                },
                |device| {
                    self.demand.lock().expect("device demand").take();
                    drop(device);
                },
                sequence,
                |preview, device, sequence| {
                    let plan = load_plan(preview);
                    let driver = self.clone();
                    let load_instance = instance_id.clone();
                    let manifest = manifest.clone();
                    let worker_device = Arc::clone(device);
                    let bundle_key = bundle_key.clone();
                    async move {
                        let plan = plan.map_err(PricedFailure::Host)?;
                        driver.progress(&load_instance, ModelLoadStage::Preparing, Some(&plan));
                        let preparation = driver.preparations.load(bundle_key, plan.device.clone());
                        magnitude_service_utils::blocking::run(move || {
                            let progression = RefCell::new(LoadProgression::default());
                            let notifier = driver.client.notifier();
                            let report = |progress: LoadProgress| {
                                if let Some((stage, fraction, details)) =
                                    progression.borrow_mut().advance(progress)
                                {
                                    notifier.send(ResidencyNotification::LoadProgress {
                                        instance_id: load_instance.clone(),
                                        stage,
                                        fraction,
                                        details,
                                        planned_allocation: Some(plan.clone()),
                                    });
                                }
                            };
                            let last_worker = RefCell::new(None);
                            let spawn_failure = RefCell::new(None);
                            let spawn = || {
                                let stopped = |reason: String| LoadError::WorkerLost { reason };
                                let Some(spawned) = sequence.start(|| {
                                    driver.spawn_load_worker(&load_instance, &worker_device)
                                }) else {
                                    return Err(stopped("the load was stopped".into()));
                                };
                                match spawned {
                                    Ok(spawned) => {
                                        *last_worker.borrow_mut() = Some(Arc::clone(&spawned.worker));
                                        Ok(spawned)
                                    }
                                    Err(failure) => {
                                        let reason = failure.message();
                                        *spawn_failure.borrow_mut() = Some(failure);
                                        Err(stopped(reason))
                                    }
                                }
                            };
                            let preparation = RefCell::new(preparation);
                            let loaded = load_prepared(
                                &manifest,
                                |unprepared| {
                                    spawn()?
                                        .connect(manifest.clone(), unprepared, &report)
                                        .map_err(|(_, error)| error)
                                },
                                || spawn().map(SpawnedWorker::into_process),
                                |progress: LoadProgress| {
                                    preparation.borrow_mut().report(progress.clone());
                                    report(progress);
                                },
                            );
                            match loaded {
                                Ok((loaded, prepared)) => {
                                    match prepared {
                                        Preparation::Incomplete(error) => tracing::warn!(
                                            %error,
                                            "kernel preparation did not finish; the load serves \
                                             canonical defaults for the units without a tuning record"
                                        ),
                                        Preparation::Unneeded
                                        | Preparation::Prepared
                                        | Preparation::Unshared => {
                                            preparation.into_inner().prepared();
                                        }
                                    }
                                    Ok(loaded)
                                }
                                Err(error) => Err(match spawn_failure.take() {
                                    Some(failure) => PricedFailure::Host(failure),
                                    None => PricedFailure::Load(error, last_worker.take(), sequence),
                                }),
                            }
                        })
                        .await
                    }
                },
            )
            .await
            .map_err(|error| match error {
                PricedError::Preview(error) => preview_failure(error),
                PricedError::Host(failure) => failure,
                PricedError::Load(error, Some(worker)) => load_failure(error, &worker),
                PricedError::Load(error, None) => {
                    ModelOperationFailure::new("worker_exited", error.to_string(), true)
                }
            })?;
            let plan = load_plan(&preview)?;

            // The worker read its own package: it must agree with the host's resolution on the
            // package identity, the chat templates and the input modalities. This is the end of the
            // worker's finalizing stage.
            let ready =
                ReadyEngine::new(Arc::clone(&resolved.host), connection).map_err(|mismatch| {
                    let code = match &mismatch {
                        ReadinessMismatch::Package { .. } => "worker_package_mismatch",
                        ReadinessMismatch::TemplateFingerprint { .. } => "worker_template_mismatch",
                        ReadinessMismatch::Modalities { .. } => "worker_modalities_mismatch",
                        ReadinessMismatch::ImageCells { .. } => "worker_image_cells_mismatch",
                    };
                    ModelOperationFailure::new(code, mismatch.to_string(), false)
                })?;
            let context_window_tokens = plan.context_window_tokens;
            let ready_info = ready.ready_info();
            let allocation = instance_allocation(context_window_tokens, &ready_info.census)
                .map_err(|error| {
                    ModelOperationFailure::new("worker_census_invalid", error.to_string(), false)
                })?;
            let device = load_device(ready_info.device, ready_info.backend);
            let (host, client, _) = ready.into_parts();
            tracing::info!(device = %device.id, "model ready");
            Ok(PreparedResidency {
                package_ids: resolved.package_ids.clone(),
                allocation,
                device,
                worker,
                resident: ResidentEngine {
                    host,
                    client,
                    context_window_tokens,
                    _configuration: Arc::clone(&resolved),
                },
            })
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
        sequence: LoadSequence,
    ) -> Result<tokio::task::JoinHandle<()>, InventoryError> {
        let driver = self.clone();
        Ok(tokio::spawn(async move {
            let result = std::panic::AssertUnwindSafe(driver.load(&instance_id, target.configuration, sequence))
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
        magnitude_service_utils::owner::spawn("a worker's release", async move {
            // Shutdown first in every case: the worker ends its open requests with
            // `ModelUnloaded { Shutdown }` (`model_instance_stopped`), then its process ends. An
            // interrupted release bounds its answer by the short bound instead of the graceful
            // one.
            worker.shutdown();
            let mut deadline = tokio::time::Instant::now() + if interrupt_immediately { INTERRUPTED_ANSWER } else { GRACEFUL_EXIT };
            // Once it answered, or once killed, the worker is polled until the kernel reaps it,
            // however long that takes: the release ends only with its retirement shown, and
            // nothing waits on it.
            let mut killed = false;
            let result = loop {
                match worker.try_wait() {
                    Ok(Some(_)) => break Ok(()),
                    Ok(None) if killed || worker.answered_shutdown() || tokio::time::Instant::now() < deadline => {
                        tokio::select! {
                            biased;
                            changed = interruption.changed() => {
                                if changed.is_ok() && *interruption.borrow() {
                                    deadline = deadline.min(tokio::time::Instant::now() + INTERRUPTED_ANSWER);
                                }
                            }
                            () = tokio::time::sleep(std::time::Duration::from_millis(10)) => {}
                        }
                    }
                    Ok(None) => match worker.terminate(
                        "worker_unresponsive",
                        "inference worker did not answer its shutdown",
                    ) {
                        Ok(()) => killed = true,
                        Err(error) => {
                            break Err(ModelOperationFailure::new(
                                "worker_retirement_failed",
                                format!("the inference worker could not be killed: {error}"),
                                true,
                            ));
                        }
                    },
                    Err(error) => {
                        if let Err(kill) = worker.terminate(
                            "worker_monitor_failed",
                            "failed to observe inference worker during shutdown",
                        ) {
                            tracing::error!(error = %kill, "the inference worker could not be killed");
                        }
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
        );
    }

    fn idle_timeout(&self) -> std::time::Duration {
        self.environment.idle_timeout
    }

    fn demand_ended(&self) {
        self.demand.lock().expect("device demand").take();
    }

    fn resident_idle(&self, resident: Option<(&ModelInstanceId, &ResidentEngine)>) {
        self.environment.device.resident_idle(resident.map(|(instance, resident)| IdleResident {
            instance: instance.clone(),
            room: Arc::new(resident.client.clone()),
        }));
    }

    fn resident_leased(&self) {
        self.environment.device.resident_leased();
    }
}

/// A measurement beside the resident model reaches its worker through its client.
impl ResidentRoom for EngineClient {
    fn claim(&self, bytes: u64) -> futures_util::future::BoxFuture<'static, Result<RoomAnswer, RequestError>> {
        let client = self.clone();
        Box::pin(async move { client.claim_measurement_room(bytes).await })
    }

    fn attach(&self, process: seismic::ProcessId) {
        // A worker that is gone holds no room to credit.
        if let Err(error) = self.attach_measurement_process(process) {
            tracing::debug!(%error, "the measurement process reached no worker");
        }
    }

    fn release(&self) {
        // A worker that is gone holds no room.
        if let Err(error) = self.release_measurement_room() {
            tracing::debug!(%error, "the measurement room's release reached no worker");
        }
    }

    fn reprice(&self) {
        // A worker that is gone prices nothing.
        if let Err(error) = EngineClient::reprice(self) {
            tracing::debug!(%error, "the reprice reached no worker");
        }
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

    const MODEL: &str = "qwen3.5-4b:gguf:q4";

    /// A weight cap below the load's least weights is the service's own failure, with its own
    /// code and both numbers, never retried: not a memory shortage, which clears with memory.
    #[test]
    fn a_weight_cap_below_the_least_weights_is_never_retried() {
        let cap = magnitude_engine::error::WeightBoundBelowWeights { bound: magnitude_executor::WeightBound::Cap(2_100_000_000), needed: 2_130_136_892 };
        let failure = preview_failure(PreviewError::WeightBound(cap));
        assert!(matches!(&failure, ModelOperationFailure::WeightBound(found) if *found == cap));
        assert_eq!(failure.code(), magnitude_serving::error::WEIGHT_BOUND_BELOW_WEIGHTS);
        assert!(!failure.retryable());
        assert!(failure.message().contains("2100000000") && failure.message().contains("2130136892"));
    }

    fn sample_pending_prices() -> magnitude_engine::error::PendingPrices {
        magnitude_engine::error::PendingPrices::new(magnitude_executor::assessment::Outstanding {
            standing: vec![magnitude_executor::assessment::PointRequest::GraphCost],
            loads: Vec::new(),
        })
        .expect("non-empty pending set")
    }

    /// A load on a machine with none of the standing figures its plan is priced on stored is
    /// deferred before it takes the device: it shows `Queued`, asks the measurement job for the
    /// figures once and tells it so, and holds nothing of the device, which the job takes. Once
    /// the job has stored them and released the device, the load proceeds with its preview.
    #[tokio::test]
    async fn an_unpriced_load_waits_without_the_device_until_its_figures_are_stored() {
        use std::sync::atomic::AtomicBool;
        let device = DeviceLock::default();
        let stored = Arc::new(AtomicBool::new(false));
        let (requests, previews) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
        let (shown, mut stages) = tokio::sync::mpsc::unbounded_channel();
        let mut changes = device.changes();
        let load = tokio::spawn({
            let (device, stored, requests, previews) =
                (device.clone(), Arc::clone(&stored), Arc::clone(&requests), Arc::clone(&previews));
            async move {
                run_priced(
                    || {
                        previews.fetch_add(1, Ordering::Relaxed);
                        let priced = stored.load(Ordering::Relaxed);
                        async move {
                            match priced {
                                true => Ok("the plan"),
                                false => Err(PreviewError::PricesPending {
                                    reason: "nothing is measured".into(),
                                    pending: sample_pending_prices(),
                                }),
                            }
                        }
                    },
                    || {
                        settle_prices(&device, async {
                            shown.send(ModelLoadStage::Queued).expect("stages");
                            requests.fetch_add(1, Ordering::Relaxed);
                            Ok(())
                        })
                    },
                    |_| std::future::ready(Ok(AcquiredDevice::immediate(()))),
                    |()| {},
                    (),
                    |&plan, &(), ()| std::future::ready(Ok::<_, PricedFailure<ModelOperationFailure, ()>>(plan)),
                )
                .await
                .map(|(plan, _, _)| plan)
            }
        });
        // The job is told of the load's requests, and takes the device the load left to it.
        changes.changed().await;
        assert_eq!(stages.recv().await, Some(ModelLoadStage::Queued));
        assert_eq!(requests.load(Ordering::Relaxed), 1);
        let job = device.try_job().expect("a deferred load holds nothing of the device");
        tokio::task::yield_now().await;
        assert!(!load.is_finished(), "the load waits while the job measures");
        stored.store(true, Ordering::Relaxed);
        drop(job);
        assert_eq!(load.await.expect("load").expect("priced"), "the plan");
        assert_eq!(requests.load(Ordering::Relaxed), 1, "the figures are asked for once");
        assert_eq!(previews.load(Ordering::Relaxed), 2, "the load previews again only once the job ended");
        assert!(device.try_share().is_some(), "the load then takes its share");
    }

    /// If a device release ends without changing this load's pending figures (for example, the
    /// measurement job failed or measured another load's points), `run_priced` sees the same
    /// pending set after one settle and fails with `PricesPending` instead of looping forever.
    #[tokio::test]
    async fn an_unpriced_load_fails_when_a_measurement_release_stores_nothing() {
        let device = DeviceLock::default();
        let previews = Arc::new(AtomicU64::new(0));
        let mut changes = device.changes();
        let load = tokio::spawn({
            let (device, previews) = (device.clone(), Arc::clone(&previews));
            async move {
                run_priced(
                    || {
                        previews.fetch_add(1, Ordering::Relaxed);
                        async {
                            Err::<&str, _>(PreviewError::PricesPending {
                                reason: "nothing is measured".into(),
                                pending: sample_pending_prices(),
                            })
                        }
                    },
                    || settle_prices(&device, async { Ok(()) }),
                    |_| std::future::ready(Ok(AcquiredDevice::immediate(()))),
                    |()| {},
                    (),
                    |&plan, &(), ()| {
                        std::future::ready(Ok::<_, PricedFailure<ModelOperationFailure, ()>>(plan))
                    },
                )
                .await
            }
        });
        changes.changed().await;
        let job = device.try_job().expect("a deferred load holds nothing of the device");
        drop(job);
        let outcome = load.await.expect("load");
        assert!(matches!(
            outcome,
            Err(PricedError::Preview(PreviewError::PricesPending { .. }))
        ));
        assert_eq!(previews.load(Ordering::Relaxed), 2, "re-previews once after settle, then stops");
    }

    fn snapshot(model: &str, assessment: serde_json::Value) -> ModelAssessmentsSnapshot {
        snapshot_in(
            model,
            serde_json::json!({
                "_tag": "Assessed",
                "capabilities": {
                    "vision": false, "tools": true, "structuredOutput": true,
                    "reasoning": { "supported": false, "efforts": [] },
                },
                "templateFingerprint": "template",
                "profiles": [assessment],
            }),
        )
    }

    fn snapshot_in(model: &str, state: serde_json::Value) -> ModelAssessmentsSnapshot {
        serde_json::from_value(serde_json::json!({
            "revision": 1,
            "environmentId": "environment",
            "catalog": {
                "_tag": "Available",
                "sourceRevision": 1,
                "entries": [{
                    "subject": { "_tag": "Catalog", "modelId": model, "selection": "Effective" },
                    "state": state,
                }],
            },
            "custom": { "_tag": "Pending", "sourceRevision": 0 },
            "measurement": { "_tag": "Unmeasured" },
        }))
        .expect("assessment snapshot")
    }

    fn does_not_fit(context_length: u32, deficit_bytes: u64) -> serde_json::Value {
        serde_json::json!({
            "_tag": "DoesNotFit",
            "profile": { "contextLength": context_length },
            "shortfall": { "contextLength": 32_768 },
            "assessmentId": "assessment",
            "memory": [],
            "limitingResource": "gpu",
            "deficitBytes": deficit_bytes,
            "projector": { "_tag": "None" },
            "drafter": { "_tag": "None" },
        })
    }

    fn fits(context_length: u32, served: u32) -> serde_json::Value {
        serde_json::json!({
            "_tag": "Fits",
            "profile": { "contextLength": context_length },
            "served": { "contextLength": served },
            "assessmentId": "assessment",
            "memory": [],
            "memoryBreakdown": { "weightsBytes": 0, "contextBytes": 0, "overheadBytes": 0 },
            "placement": { "deviceBytes": 0, "hostBytes": 0, "diskBytes": 0, "lookupDiskBytes": 0 },
            "residency": { "_tag": "Resident", "performance": { "_tag": "Pending" } },
            "projector": { "_tag": "None" },
            "drafter": { "_tag": "None" },
        })
    }

    fn model_id(model: &str) -> ModelId {
        model.parse().expect("model id")
    }

    const CEILING: ServingProfile = ServingProfile { context_length: 262_144 };

    #[test]
    fn a_load_assessed_as_not_fitting_is_refused_with_the_shortfall() {
        let snapshot = snapshot(MODEL, does_not_fit(262_144, 1_825_361_000));
        let ServedProfile::Refused(InventoryError::DoesNotFit(message)) =
            served_profile(&snapshot, &model_id(MODEL), &CEILING)
        else {
            panic!("expected a does-not-fit refusal");
        };
        assert_eq!(
            message,
            format!(
                "{MODEL} needs 1.7 GB more memory than this computer can provide, even for a 32K context"
            )
        );
    }

    #[test]
    fn a_fitting_load_serves_its_assessed_context() {
        let ServedProfile::Served(served) =
            served_profile(&snapshot(MODEL, fits(262_144, 65_536)), &model_id(MODEL), &CEILING)
        else {
            panic!("expected the served profile");
        };
        assert_eq!(served, ServingProfile { context_length: 65_536 });
    }

    #[test]
    fn a_catalog_load_serves_its_effective_selection() {
        let mut snapshot = snapshot(MODEL, fits(262_144, 131_072));
        let ModelAssessmentDomainSnapshot::Available { entries, .. } = &mut snapshot.catalog else {
            unreachable!()
        };
        let mut desired = entries[0].clone();
        desired.subject = ModelAssessmentSubject::Catalog {
            model_id: model_id(MODEL),
            selection: CatalogModelSelection::Desired,
        };
        if let ModelAssessmentEntryState::Assessed { profiles, .. } = &mut desired.state {
            profiles[0] = serde_json::from_value(fits(262_144, 32_768)).unwrap();
        }
        entries.insert(0, desired);
        let ServedProfile::Served(served) = served_profile(&snapshot, &model_id(MODEL), &CEILING)
        else {
            panic!("expected the served profile");
        };
        assert_eq!(served.context_length, 131_072);
    }

    #[test]
    fn a_load_waits_for_its_assessment() {
        let pending = |snapshot: &ModelAssessmentsSnapshot| {
            matches!(
                served_profile(snapshot, &model_id(MODEL), &CEILING),
                ServedProfile::Assessing
            )
        };
        assert!(pending(&snapshot_in(MODEL, serde_json::json!({ "_tag": "Assessing" }))));
        assert!(pending(&snapshot("other:gguf:q4", does_not_fit(262_144, 1))));
        // A verdict for another profile only, with no assessment to come, leaves nothing to wait
        // for; nor does a failed one.
        assert!(matches!(
            served_profile(&snapshot(MODEL, does_not_fit(131_072, 1)), &model_id(MODEL), &CEILING),
            ServedProfile::Unassessed
        ));
        assert!(matches!(
            served_profile(
                &snapshot_in(MODEL, serde_json::json!({ "_tag": "Dropped" })),
                &model_id(MODEL),
                &CEILING
            ),
            ServedProfile::Unassessed
        ));
    }

    #[test]
    fn an_unsupported_model_without_a_profile_is_refused_at_every_ceiling() {
        let snapshot = snapshot(MODEL, serde_json::json!({
            "_tag": "Unsupported",
            "reasons": [],
            "detail": "cannot read model header",
        }));
        assert!(matches!(served_profile(&snapshot, &model_id(MODEL), &CEILING), ServedProfile::Refused(_)));
    }

    #[test]
    fn a_load_the_engine_cannot_run_is_refused_with_its_reasons() {
        let snapshot = snapshot(
            MODEL,
            serde_json::json!({
                "_tag": "Unsupported",
                "profile": { "contextLength": 131_072 },
                "reasons": [{ "_tag": "Architecture", "architecture": "deepseek4" }],
                "detail": "no family recognizes deepseek4",
            }),
        );
        assert!(matches!(served_profile(&snapshot, &model_id(MODEL), &CEILING), ServedProfile::Unassessed));
        let ServedProfile::Refused(InventoryError::ModelUnsupported { message, reasons }) =
            served_profile(&snapshot, &model_id(MODEL), &ServingProfile { context_length: 131_072 })
        else {
            panic!("expected an unsupported refusal");
        };
        assert_eq!(
            message,
            format!("Magnitude can't run {MODEL}: no family recognizes deepseek4")
        );
        assert_eq!(
            serde_json::to_value(&reasons).unwrap(),
            serde_json::json!([{ "_tag": "Architecture", "architecture": "deepseek4" }])
        );
    }

    #[test]
    fn contexts_read_as_people_name_them() {
        assert_eq!(context_size(32_768), "32K");
        assert_eq!(context_size(262_144), "256K");
        assert_eq!(context_size(40_960), "40K");
        assert_eq!(context_size(8_000), "8000-token");
    }

    #[test]
    fn shortfalls_read_as_the_clients_show_them() {
        assert_eq!(memory_size(1_000), "0.1 MB");
        assert_eq!(memory_size(359_000_000), "342.4 MB");
        assert_eq!(memory_size(1_825_361_000), "1.7 GB");
        assert_eq!(memory_size(3_758_096_384), "3.5 GB");
    }

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
            work: None,
            measured_configurations: 0,
            formed_programs: 0,
            },
            LoadProgress::Tuning {
                completed: 50,
                total: 100,
            work: None,
            measured_configurations: 0,
            formed_programs: 0,
            },
            weights(0, 1000),
            weights(500, 1000),
            weights(1000, 1000),
            LoadProgress::Finalizing,
        ];
        let reported = steps
            .into_iter()
            .map(|progress| {
                let (stage, fraction, _) = progression.advance(progress).unwrap();
                (stage, fraction)
            })
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

    /// A first load is three workers in turn. The Prepare worker's tuning fills the first
    /// half; the second Load worker's own preparation does not take the load back to
    /// `preparing`.
    #[test]
    fn the_load_worker_continues_where_the_prepare_worker_ended() {
        let mut progression = LoadProgression::default();
        // The Load worker that ends unprepared, then the Prepare worker.
        progression.advance(LoadProgress::Preparing);
        progression.advance(LoadProgress::Preparing);
        progression.advance(LoadProgress::Tuning {
            completed: 100,
            total: 100,
            work: None,
            measured_configurations: 0,
            formed_programs: 0,
        });
        // The Load worker.
        assert_eq!(progression.advance(LoadProgress::Preparing), None);
        assert_eq!(
            progression.advance(weights(500, 1000)).map(|(stage, fraction, _)| (stage, fraction)),
            Some((ModelLoadStage::LoadingWeights, 0.75))
        );
        // No tuning was reported: the load fills with weights.
        let mut progression = LoadProgression::default();
        progression.advance(LoadProgress::Preparing);
        assert_eq!(progression.advance(LoadProgress::Preparing), None);
        assert_eq!(
            progression.advance(weights(500, 1000)).map(|(stage, fraction, _)| (stage, fraction)),
            Some((ModelLoadStage::LoadingWeights, 0.5))
        );
    }

    #[test]
    fn a_load_without_tuning_fills_with_weights() {
        let mut progression = LoadProgression::default();
        progression.advance(LoadProgress::Preparing);
        assert_eq!(
            progression.advance(weights(250, 1000)).map(|(stage, fraction, _)| (stage, fraction)),
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
            progression.advance(weights(1000, 100_000)).map(|(stage, fraction, _)| (stage, fraction)),
            Some((ModelLoadStage::LoadingWeights, 0.01))
        );
        // A unit found unexpectedly to search raises the tuning total; the fraction holds.
        let mut progression = LoadProgression::default();
        progression.advance(LoadProgress::Tuning {
            completed: 50,
            total: 100,
            work: None,
            measured_configurations: 0,
            formed_programs: 0,
        });
        assert_eq!(
            progression.advance(LoadProgress::Tuning {
                completed: 60,
                total: 200,
            work: None,
            measured_configurations: 0,
            formed_programs: 0,
            }),
            None
        );
    }
}
