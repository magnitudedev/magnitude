use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};

use anyhow::Context;
use clap::{Parser, Subcommand};
use futures_util::{FutureExt, StreamExt, future::BoxFuture, stream::BoxStream};
use icn_api::{
    AppState, FakeBackend, ModelInstanceController, ModelInstanceLease, ServerIdentity, app,
};
use icn_contracts::bootstrap_protocol::{
    IcnInstallationBackend, IcnStartupBackend, IcnStartupProgressRecord,
    IcnStartupProgressRecordType, IcnStartupRecord, IcnStartupRecordType,
};
use icn_contracts::models::{
    AssessModelRequest, AssessModelResult, AssessModelsRequest, AssessModelsResponse,
    AssessmentEnvironmentId, InstalledModelPackages as _, LoadModelReady, LoadModelRequest,
    MemoryAssessment, ModelAssessment, ModelAssessmentId,
    ModelAssessmentProfile as DomainModelAssessmentProfile, ModelAssessor, ModelBundleInput,
    ModelFailure as DomainModelFailure, ModelInstance, ModelInstanceFailure, ModelInstanceId,
    ModelInstanceLifecycle, ModelInstancesInvalidation, ModelInstancesSnapshot, ModelLoadEvent,
    ModelLoadPlan, ModelLoadStage, ModelPackageId, ModelPackageOperand, ModelReleaseReason,
    ModelServingConfiguration, ModelServingConfigurationId, ModelStoppingAllocation,
    PerformanceConfidence, PerformanceEvidence, PreviewModelLoadRequest,
    RemoveInstalledModelPackageResponse, ServableModelBundle as DomainServableModelBundle,
    SpeculativeDraftSource as ModelSpeculativeDraftSource, SpeculativeDraftSourceInput,
    SpeculativeMethod,
};
use icn_contracts::{
    CacheType, CompletionBackend, ComponentRole, ExecutionIntent, GenerationPerformanceAssessment,
    HardwareAssessment, HardwareProvider, HardwareSnapshot, InventoryError,
    ModelExecutionAssessment, ModelPreviewProfile, ResolvedModel, ResolvedModelAssessor,
    SpeculativeDecodingConfig, SpeculativeDraftSource, SpeculativeMethodConfig, TemplateAssessment,
    TemplateAssessor,
};
use icn_engine::{
    ModelLoadObserver, ModelPlanDefaults, NativeBackend, execution_intent, model_plan_defaults,
};
use icn_hardware::CapacityPolicy;
use icn_models::{
    InventoryConfig, ManagedCatalogModels, ManagedModelDownloads, ManagedModelStore, ModelCache,
    ReleaseCatalog, ReleaseRecommendableCatalog, canonical_package_id, load_release_catalog,
    servable_model_bundle_key, serving_configuration_id, serving_configuration_identity_is_valid,
    speculative_servable_model_bundle_key,
};
use llama_cpp_2::model::params::fit::{
    FitCalibration as NativeHardwareCalibration,
    FitCalibrationMetric as NativeHardwareCalibrationMetric,
};
use sha2::{Digest, Sha256};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tower_http::trace::{DefaultOnResponse, TraceLayer};

mod backend_eligibility;
mod build_identity;
mod cuda_driver;
mod inference_worker;
mod installation;
mod load_progress;
mod memory_supervisor;
mod telemetry;
mod worker_process;

use inference_worker::{InferenceWorker, LoadEvent, RemoteBackend};
use load_progress::{LoadProgressEstimator, LoadProgressTracker};
use memory_supervisor::{IDLE_POLL_INTERVAL, RECOVERY_STABLE_TIME};
use memory_supervisor::{MONITOR_LOSS_DEADLINE, POLL_INTERVAL, SystemMemoryObserver};
use worker_process::NativeWorkerRole;
use worker_process::{NativeRuntimeAuthority, NativeWorkerArgs, NativeWorkerLauncher};

#[derive(Debug, Parser)]
#[command(
    name = "magnitude-icn",
    version,
    about = "Magnitude inference control node"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
// Clap's flat `serve` command intentionally keeps its complete execution profile visible in
// `--help`; boxing individual flags would only optimize the one-time CLI parse allocation.
#[allow(clippy::large_enum_variant)]
enum Command {
    Serve {
        #[arg(long, default_value = "127.0.0.1:8080")]
        bind: SocketAddr,
        /// Opaque owner-provided identity echoed by the startup and health protocols.
        #[arg(long, default_value = "standalone")]
        instance_id: String,
        /// Exit when the private owning process closes stdin.
        #[arg(long)]
        exit_on_stdin_eof: bool,
        /// Private owner capability. Prefer the environment-backed form used by managed launch.
        #[arg(long, env = "MAGNITUDE_ICN_AUTH_TOKEN", hide_env_values = true)]
        auth_token: Option<String>,
        /// Deterministic in-memory backend used only by protocol tests.
        #[arg(long)]
        fake: bool,
        /// Magnitude-owned model inventory and Hugging Face cache root.
        #[arg(long, visible_alias = "models-dir")]
        model_store: Option<PathBuf>,
        /// Magnitude-owned root for all disposable derived cache data.
        #[arg(long)]
        cache_root: Option<PathBuf>,
        /// Additional read-only Hugging Face hub cache roots.
        #[arg(long = "hf-cache", visible_alias = "hf-cache-dir")]
        hf_caches: Vec<PathBuf>,
        /// Verified release or prepared development installation.
        #[arg(long)]
        installation: Option<PathBuf>,
    },
    Doctor,
    /// Probe supported accelerator APIs without loading an accelerator module.
    BackendEligibility {
        #[arg(long)]
        json: bool,
    },
    Version {
        #[arg(long)]
        json: bool,
    },
    #[command(hide = true)]
    PlanningWorker {
        #[command(flatten)]
        runtime: NativeWorkerArgs,
    },
    #[command(hide = true)]
    TemplateWorker {
        #[command(flatten)]
        runtime: NativeWorkerArgs,
    },
    #[command(hide = true)]
    InferenceWorker {
        #[command(flatten)]
        runtime: NativeWorkerArgs,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelExecutionProfile {
    context_length: u32,
}

#[derive(Clone)]
struct InstanceRuntime {
    inner: Arc<RwLock<InstanceRuntimeState>>,
    active_leases: Arc<AtomicU64>,
    mutating: Arc<AtomicBool>,
    mutation_available: Arc<tokio::sync::Notify>,
    lease_released: Arc<tokio::sync::Notify>,
    activity: Arc<Mutex<InstanceActivity>>,
    activity_changed: Arc<tokio::sync::Notify>,
}

struct InstanceRuntimeState {
    backend: Option<Arc<dyn CompletionBackend>>,
    instance_id: Option<ModelInstanceId>,
    configuration_id: Option<ModelServingConfigurationId>,
    generation: u64,
}

#[derive(Debug, Clone, Copy)]
struct InstanceActivity {
    generation: u64,
    active_leases: u64,
    idle_since: Option<std::time::Instant>,
}

struct InstanceMutationGuard {
    mutating: Arc<AtomicBool>,
    mutation_available: Arc<tokio::sync::Notify>,
}

impl Drop for InstanceMutationGuard {
    fn drop(&mut self) {
        self.mutating.store(false, Ordering::Release);
        self.mutation_available.notify_one();
    }
}

impl InstanceRuntime {
    fn empty() -> Self {
        Self {
            inner: Arc::new(RwLock::new(InstanceRuntimeState {
                backend: None,
                instance_id: None,
                configuration_id: None,
                generation: 0,
            })),
            active_leases: Arc::new(AtomicU64::new(0)),
            mutating: Arc::new(AtomicBool::new(false)),
            mutation_available: Arc::new(tokio::sync::Notify::new()),
            lease_released: Arc::new(tokio::sync::Notify::new()),
            activity: Arc::new(Mutex::new(InstanceActivity {
                generation: 0,
                active_leases: 0,
                idle_since: None,
            })),
            activity_changed: Arc::new(tokio::sync::Notify::new()),
        }
    }

    fn acquire(
        &self,
        instance_id: &ModelInstanceId,
        configuration_id: &ModelServingConfigurationId,
    ) -> Option<ModelInstanceLease> {
        if self.mutating.load(Ordering::Acquire) {
            return None;
        }
        let state = self.inner.read().ok()?;
        if state.instance_id.as_ref() != Some(instance_id)
            || state.configuration_id.as_ref() != Some(configuration_id)
        {
            return None;
        }
        let backend = state.backend.clone()?;
        self.active_leases.fetch_add(1, Ordering::AcqRel);
        if self.mutating.load(Ordering::Acquire) {
            self.active_leases.fetch_sub(1, Ordering::AcqRel);
            self.lease_released.notify_waiters();
            return None;
        }
        if let Ok(mut activity) = self.activity.lock() {
            activity.active_leases = activity.active_leases.saturating_add(1);
            activity.idle_since = None;
        }
        self.activity_changed.notify_waiters();
        let active_leases = Arc::clone(&self.active_leases);
        let lease_released = Arc::clone(&self.lease_released);
        let activity = Arc::clone(&self.activity);
        let activity_changed = Arc::clone(&self.activity_changed);
        Some(ModelInstanceLease::new(
            backend,
            instance_id.clone(),
            configuration_id.clone(),
            Arc::new(std::collections::BTreeSet::new()),
            move || {
                active_leases.fetch_sub(1, Ordering::AcqRel);
                if let Ok(mut activity) = activity.lock() {
                    activity.active_leases = activity.active_leases.saturating_sub(1);
                    if activity.active_leases == 0 {
                        activity.idle_since = Some(std::time::Instant::now());
                    }
                }
                activity_changed.notify_waiters();
                lease_released.notify_waiters();
            },
        ))
    }

    fn try_begin_mutation(&self) -> Option<InstanceMutationGuard> {
        self.mutating
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()?;
        if self.active_leases.load(Ordering::Acquire) > 0 {
            self.mutating.store(false, Ordering::Release);
            self.mutation_available.notify_one();
            return None;
        }
        Some(InstanceMutationGuard {
            mutating: Arc::clone(&self.mutating),
            mutation_available: Arc::clone(&self.mutation_available),
        })
    }

    async fn begin_mutation(&self) -> InstanceMutationGuard {
        loop {
            let available = self.mutation_available.notified();
            if self
                .mutating
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                while self.active_leases.load(Ordering::Acquire) > 0 {
                    let released = self.lease_released.notified();
                    if self.active_leases.load(Ordering::Acquire) > 0 {
                        released.await;
                    }
                }
                return InstanceMutationGuard {
                    mutating: Arc::clone(&self.mutating),
                    mutation_available: Arc::clone(&self.mutation_available),
                };
            }
            available.await;
        }
    }

    fn install(
        &self,
        instance_id: ModelInstanceId,
        configuration_id: ModelServingConfigurationId,
        backend: Arc<dyn CompletionBackend>,
    ) -> u64 {
        let mut state = self.inner.write().expect("instance runtime lock poisoned");
        state.generation = state.generation.saturating_add(1);
        state.backend = Some(backend);
        state.instance_id = Some(instance_id);
        state.configuration_id = Some(configuration_id);
        let generation = state.generation;
        drop(state);
        if let Ok(mut activity) = self.activity.lock() {
            *activity = InstanceActivity {
                generation,
                active_leases: 0,
                idle_since: Some(std::time::Instant::now()),
            };
        }
        self.activity_changed.notify_waiters();
        generation
    }

    fn clear(&self) {
        let mut state = self.inner.write().expect("instance runtime lock poisoned");
        state.generation = state.generation.saturating_add(1);
        state.backend = None;
        state.instance_id = None;
        state.configuration_id = None;
        let generation = state.generation;
        drop(state);
        if let Ok(mut activity) = self.activity.lock() {
            *activity = InstanceActivity {
                generation,
                active_leases: 0,
                idle_since: None,
            };
        }
        self.activity_changed.notify_waiters();
    }

    fn activity(&self) -> InstanceActivity {
        self.activity
            .lock()
            .map(|activity| *activity)
            .unwrap_or(InstanceActivity {
                generation: 0,
                active_leases: 0,
                idle_since: None,
            })
    }

    fn activity_changed(&self) -> impl std::future::Future<Output = ()> + '_ {
        self.activity_changed.notified()
    }
}

#[derive(Clone)]
struct ReadyInstanceRecord {
    configuration_id: ModelServingConfigurationId,
    instance_id: ModelInstanceId,
    generation: u64,
    package_ids: Vec<ModelPackageId>,
    allocation: icn_contracts::models::ModelInstanceAllocation,
    runtime: InstanceRuntime,
}

#[derive(Clone)]
struct ModelInstanceEntry {
    instance: ModelInstance,
    stop_requested: Arc<AtomicBool>,
    worker: Option<OwnedInferenceWorker>,
    ready: Option<ReadyInstanceRecord>,
}

#[derive(Clone)]
struct ModelInstanceRegistry {
    revision: u64,
    entries: std::collections::BTreeMap<ModelInstanceId, ModelInstanceEntry>,
    resident_instance_id: Option<ModelInstanceId>,
}

impl ModelInstanceRegistry {
    fn admit(
        &mut self,
        instance_id: ModelInstanceId,
        configuration_id: ModelServingConfigurationId,
    ) -> Result<(Arc<AtomicBool>, bool, u64), DomainModelFailure> {
        if let Some(existing) = self.entries.get(&instance_id) {
            return if existing.instance.configuration_id == configuration_id {
                Ok((Arc::clone(&existing.stop_requested), false, self.revision))
            } else {
                Err(DomainModelFailure {
                    code: "model_instance_identity_conflict".to_owned(),
                    message: "model instance ID was already admitted for another configuration"
                        .to_owned(),
                    retryable: false,
                })
            };
        }
        self.revision = self.revision.saturating_add(1);
        let stop_requested = Arc::new(AtomicBool::new(false));
        self.entries.insert(
            instance_id.clone(),
            ModelInstanceEntry {
                instance: ModelInstance {
                    id: instance_id,
                    configuration_id,
                    lifecycle: ModelInstanceLifecycle::Loading {
                        stage: ModelLoadStage::Queued,
                        progress: None,
                        planned_allocation: None,
                    },
                },
                stop_requested: Arc::clone(&stop_requested),
                worker: None,
                ready: None,
            },
        );
        Ok((stop_requested, true, self.revision))
    }

    fn publish(&mut self, instance: ModelInstance) -> Option<u64> {
        let current = self
            .entries
            .get(&instance.id)
            .expect("model instance must be admitted before publication");
        assert_eq!(
            current.instance.configuration_id, instance.configuration_id,
            "an admitted model instance cannot change configuration"
        );
        if current.instance == instance {
            return None;
        }
        let transition_allowed = matches!(
            (&current.instance.lifecycle, &instance.lifecycle),
            (
                ModelInstanceLifecycle::Loading { .. },
                ModelInstanceLifecycle::Loading { .. }
            ) | (
                ModelInstanceLifecycle::Loading { .. },
                ModelInstanceLifecycle::Stopping { .. }
            ) | (
                ModelInstanceLifecycle::Loading { .. },
                ModelInstanceLifecycle::Stopped { .. }
            ) | (
                ModelInstanceLifecycle::Loading { .. },
                ModelInstanceLifecycle::Failed { .. }
            ) | (
                ModelInstanceLifecycle::Ready { .. },
                ModelInstanceLifecycle::Stopping { .. }
            ) | (
                ModelInstanceLifecycle::Ready { .. },
                ModelInstanceLifecycle::Failed { .. }
            ) | (
                ModelInstanceLifecycle::Stopping { .. },
                ModelInstanceLifecycle::Stopped { .. }
            ) | (
                ModelInstanceLifecycle::Stopping { .. },
                ModelInstanceLifecycle::Failed { .. }
            )
        );
        let loading_after_stop =
            matches!(&instance.lifecycle, ModelInstanceLifecycle::Loading { .. })
                && current.stop_requested.load(Ordering::Acquire);
        if !transition_allowed || loading_after_stop {
            return None;
        }
        self.revision = self.revision.saturating_add(1);
        let current = self
            .entries
            .get_mut(&instance.id)
            .expect("model instance entry remains present while borrowed");
        current.instance = instance;
        Some(self.revision)
    }

    fn snapshot(&self) -> ModelInstancesSnapshot {
        ModelInstancesSnapshot {
            revision: self.revision,
            instances: self
                .entries
                .values()
                .map(|entry| entry.instance.clone())
                .collect(),
        }
    }

    fn ready_instance(&self) -> Option<ReadyInstanceRecord> {
        let instance_id = self.resident_instance_id.as_ref()?;
        self.entries.get(instance_id)?.ready.clone()
    }

    fn publish_ready(&mut self, ready: ReadyInstanceRecord) -> Option<u64> {
        let entry = self
            .entries
            .get_mut(&ready.instance_id)
            .expect("ready resources belong to an admitted model instance");
        assert_eq!(
            entry.instance.configuration_id, ready.configuration_id,
            "ready resources must match the admitted configuration"
        );
        assert!(
            entry.worker.is_some(),
            "an instance cannot become ready without its entry-owned worker"
        );
        if !matches!(
            entry.instance.lifecycle,
            ModelInstanceLifecycle::Loading { .. }
        ) || entry.stop_requested.load(Ordering::Acquire)
        {
            return None;
        }
        self.revision = self.revision.saturating_add(1);
        entry.instance.lifecycle = ModelInstanceLifecycle::Ready {
            allocation: ready.allocation.clone(),
        };
        entry.ready = Some(ready.clone());
        self.resident_instance_id = Some(ready.instance_id);
        Some(self.revision)
    }

    fn clear_ready(&mut self, instance_id: &ModelInstanceId) {
        if let Some(entry) = self.entries.get_mut(instance_id) {
            entry.ready = None;
        }
        if self.resident_instance_id.as_ref() == Some(instance_id) {
            self.resident_instance_id = None;
        }
    }

    fn install_worker(&mut self, instance_id: &ModelInstanceId, worker: InferenceWorker) {
        let entry = self
            .entries
            .get_mut(instance_id)
            .expect("worker belongs to an admitted model instance");
        entry.worker = Some(OwnedInferenceWorker { worker });
    }

    fn owns_worker(&self, instance_id: &ModelInstanceId, pid: Option<u32>) -> bool {
        self.entries
            .get(instance_id)
            .and_then(|entry| entry.worker.as_ref())
            .is_some_and(|owned| owned.worker.pid() == pid)
    }

    fn take_worker(&mut self, instance_id: &ModelInstanceId) -> Option<OwnedInferenceWorker> {
        self.entries.get_mut(instance_id)?.worker.take()
    }
}

#[derive(Clone)]
struct InstanceEntries {
    state: Arc<tokio::sync::RwLock<ModelInstanceRegistry>>,
    changes: tokio::sync::broadcast::Sender<ModelInstancesInvalidation>,
}

impl InstanceEntries {
    fn new() -> Self {
        let (changes, _) = tokio::sync::broadcast::channel(16);
        Self {
            state: Arc::new(tokio::sync::RwLock::new(ModelInstanceRegistry {
                revision: 0,
                entries: std::collections::BTreeMap::new(),
                resident_instance_id: None,
            })),
            changes,
        }
    }

    async fn admit(
        &self,
        instance_id: ModelInstanceId,
        configuration_id: ModelServingConfigurationId,
    ) -> Result<(Arc<AtomicBool>, bool), DomainModelFailure> {
        let (stop_requested, is_new, revision) = self
            .state
            .write()
            .await
            .admit(instance_id, configuration_id)?;
        if is_new {
            let _ = self.changes.send(ModelInstancesInvalidation { revision });
        }
        Ok((stop_requested, is_new))
    }

    async fn publish(&self, instance: ModelInstance) -> u64 {
        let mut state = self.state.write().await;
        let Some(revision) = state.publish(instance) else {
            return state.revision;
        };
        drop(state);
        let _ = self.changes.send(ModelInstancesInvalidation { revision });
        revision
    }

    async fn instance(&self, instance_id: &ModelInstanceId) -> Option<ModelInstance> {
        self.state
            .read()
            .await
            .entries
            .get(instance_id)
            .map(|entry| entry.instance.clone())
    }

    async fn entry(&self, instance_id: &ModelInstanceId) -> Option<ModelInstanceEntry> {
        self.state.read().await.entries.get(instance_id).cloned()
    }

    async fn snapshot(&self) -> ModelInstancesSnapshot {
        self.state.read().await.snapshot()
    }

    async fn revision(&self) -> u64 {
        self.state.read().await.revision
    }

    fn subscribe(&self) -> tokio::sync::broadcast::Receiver<ModelInstancesInvalidation> {
        self.changes.subscribe()
    }

    async fn ready_instance(&self) -> Option<ReadyInstanceRecord> {
        self.state.read().await.ready_instance()
    }

    async fn publish_ready(&self, ready: ReadyInstanceRecord) -> bool {
        let Some(revision) = self.state.write().await.publish_ready(ready) else {
            return false;
        };
        let _ = self.changes.send(ModelInstancesInvalidation { revision });
        true
    }

    async fn clear_ready(&self, instance_id: &ModelInstanceId) {
        self.state.write().await.clear_ready(instance_id);
    }

    async fn install_worker(&self, instance_id: &ModelInstanceId, worker: InferenceWorker) {
        self.state.write().await.install_worker(instance_id, worker);
    }

    async fn owns_worker(&self, instance_id: &ModelInstanceId, pid: Option<u32>) -> bool {
        self.state.read().await.owns_worker(instance_id, pid)
    }

    async fn take_worker(&self, instance_id: &ModelInstanceId) -> Option<OwnedInferenceWorker> {
        self.state.write().await.take_worker(instance_id)
    }
}

fn select_model_allocation(
    candidates: &[(u32, u64)],
    sample: memory_supervisor::MemorySample,
) -> Option<(u32, u64)> {
    candidates
        .iter()
        .rev()
        .copied()
        .find(|(_, required)| sample.permits_load(*required))
}

fn credit_replaced_instance_memory(
    mut sample: memory_supervisor::MemorySample,
    releasable_system_memory_bytes: u64,
) -> memory_supervisor::MemorySample {
    sample.physical_available_bytes = sample
        .physical_available_bytes
        .saturating_add(releasable_system_memory_bytes)
        .min(sample.physical_capacity_bytes);
    sample.allocation_headroom_bytes = sample
        .allocation_headroom_bytes
        .saturating_add(releasable_system_memory_bytes)
        .min(sample.allocation_capacity_bytes);
    sample
}

#[derive(Clone)]
struct NativeResolvedModelAssessor {
    defaults: ModelPlanDefaults,
    cache: Option<ModelCache>,
    planning_executor: PlanningExecutor,
    native_backend: NativeBackend,
    native_executor: Arc<RwLock<Option<Weak<RemoteBackend>>>>,
    gate: Arc<tokio::sync::Mutex<()>>,
    assessment_work_gates:
        Arc<tokio::sync::Mutex<std::collections::BTreeMap<String, Weak<tokio::sync::Mutex<()>>>>>,
    assessment_concurrency: AssessmentConcurrency,
    hardware_calibration: Arc<NativeHardwareCalibration>,
    enabled_backends: Arc<Vec<String>>,
}

#[derive(Clone)]
struct AssessmentConcurrency {
    concurrency: usize,
}

impl AssessmentConcurrency {
    fn new(concurrency: usize) -> Self {
        Self { concurrency }
    }

    const fn concurrency(&self) -> usize {
        self.concurrency
    }
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct HardwareCalibrationRecord {
    input_identity: String,
    measured_at_seconds: u64,
    hardware_calibration: NativeHardwareCalibration,
    hardware_calibration_identity: String,
}

type NativeAssessorServices = (
    Arc<NativeResolvedModelAssessor>,
    Arc<RwLock<Option<Weak<RemoteBackend>>>>,
);

fn native_assessor_services(
    inventory: &Arc<ManagedModelStore>,
    planning_executor: PlanningExecutor,
    native_backend: NativeBackend,
    defaults: ModelPlanDefaults,
    hardware_calibration: NativeHardwareCalibration,
    enabled_backends: Vec<String>,
) -> NativeAssessorServices {
    let native_executor = Arc::new(RwLock::new(None));
    let assessor = Arc::new(NativeResolvedModelAssessor {
        defaults,
        cache: Some(inventory.derived_cache().clone()),
        planning_executor,
        native_backend,
        native_executor: Arc::clone(&native_executor),
        gate: Arc::new(tokio::sync::Mutex::new(())),
        assessment_work_gates: Arc::new(tokio::sync::Mutex::new(std::collections::BTreeMap::new())),
        assessment_concurrency: AssessmentConcurrency::new(assessment_orchestration_concurrency()),
        hardware_calibration: Arc::new(hardware_calibration),
        enabled_backends: Arc::new(enabled_backends),
    });
    (assessor, native_executor)
}

const MAX_ASSESSMENT_ORCHESTRATION_CONCURRENCY: usize = 12;
const ASSESSMENT_RESERVE_BYTES_PER_DEDICATED_DOMAIN: u64 = 1536 * 1024 * 1024;

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct PlanningWorkerRequest {
    deadline_at_ms: Option<u64>,
    hardware: HardwareSnapshot,
    primary: PathBuf,
    projector: Option<PathBuf>,
    speculative: SpeculativeDecodingConfig,
    defaults: Vec<ModelPlanDefaults>,
    performance_context_tokens: Vec<Vec<u32>>,
    operation: PlanningOperation,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum PlanningOperation {
    Capacity,
    Execution {
        hardware_calibration: NativeHardwareCalibration,
    },
}

impl PlanningOperation {
    const fn as_str(&self) -> &'static str {
        match self {
            Self::Capacity => "capacity",
            Self::Execution { .. } => "execution",
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[cfg(not(test))]
enum IsolatedWorkerOutcome {
    Deadline,
    OutputBound,
}

#[cfg(not(test))]
impl IsolatedWorkerOutcome {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Deadline => "deadline",
            Self::OutputBound => "output_bound",
        }
    }
}

#[derive(Debug)]
#[cfg(not(test))]
struct IsolatedWorkerFailure {
    message: String,
}

#[cfg(not(test))]
impl IsolatedWorkerFailure {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[cfg(not(test))]
impl std::fmt::Display for IsolatedWorkerFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

#[cfg(not(test))]
impl std::error::Error for IsolatedWorkerFailure {}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum PlanningWorkerResponse {
    Capacity {
        assessments: Vec<HardwareAssessment>,
    },
    Execution {
        assessments: Vec<ModelExecutionAssessment>,
    },
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum PlanningWorkerCommand {
    Initialize {
        hardware_calibration: Option<NativeHardwareCalibration>,
    },
    Assess {
        request: PlanningWorkerRequest,
    },
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum PlanningWorkerReply {
    Initialized {
        hardware_calibration: NativeHardwareCalibration,
    },
    Assessed {
        response: PlanningWorkerResponse,
    },
    Defect {
        message: String,
    },
}

const MAX_PLANNING_FRAME_BYTES: usize = 1024 * 1024;
const MAX_PLANNING_DIAGNOSTIC_BYTES: usize = 64 * 1024;
const MODEL_ASSESSMENT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);
const MAX_PLANNING_WORKERS: usize = 8;
const PLANNING_WORKER_REAP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

#[derive(Clone)]
enum PlanningExecutor {
    Worker(PersistentPlanningWorkerPool),
    #[cfg_attr(not(test), allow(dead_code))]
    InProcess(NativeBackend),
}

impl PlanningExecutor {
    async fn assess(
        &self,
        request: PlanningWorkerRequest,
    ) -> Result<PlanningWorkerResponse, InventoryError> {
        match self {
            Self::Worker(worker) => worker.assess(request).await,
            Self::InProcess(native_backend) => {
                let native_backend = native_backend.clone();
                spawn_blocking_traced(move || {
                    assess_planning_request_with_backend(request, &native_backend)
                })
                .await
                .map_err(|error| {
                    InventoryError::Internal(format!("model assessment task defect: {error}"))
                })?
                .map_err(|error| {
                    InventoryError::Internal(format!("model assessment defect: {error:#}"))
                })
            }
        }
    }
}

struct PlanningJob {
    command: PlanningWorkerCommand,
    enqueued_at: std::time::Instant,
    deadline: tokio::time::Instant,
    response: tokio::sync::oneshot::Sender<Result<PlanningWorkerReply, InventoryError>>,
}

struct PlanningWorkerProcess {
    child: tokio::process::Child,
    stderr_tail: Arc<std::sync::Mutex<Vec<u8>>>,
    stderr_reader: tokio::task::JoinHandle<()>,
}

impl PlanningWorkerProcess {
    fn diagnostics(&self) -> String {
        self.stderr_tail
            .lock()
            .map(|tail| String::from_utf8_lossy(&tail).trim().to_owned())
            .unwrap_or_else(|_| "planning worker diagnostic buffer was poisoned".to_owned())
    }

    fn terminate(mut self) -> String {
        let diagnostics = self.diagnostics();
        let _ = self.child.start_kill();
        tokio::spawn(async move {
            let _ = tokio::time::timeout(PLANNING_WORKER_REAP_TIMEOUT, self.child.wait()).await;
            self.stderr_reader.abort();
            let _ = self.stderr_reader.await;
        });
        diagnostics
    }
}

#[derive(Clone)]
struct PersistentPlanningWorker {
    jobs: tokio::sync::mpsc::Sender<PlanningJob>,
}

impl PersistentPlanningWorker {
    fn start(
        launcher: NativeWorkerLauncher,
        hardware_calibration: Arc<std::sync::OnceLock<NativeHardwareCalibration>>,
        initialization_gate: Arc<tokio::sync::Mutex<()>>,
    ) -> Self {
        let (jobs, receiver) = tokio::sync::mpsc::channel(32);
        tokio::spawn(run_planning_worker_supervisor(
            launcher,
            receiver,
            hardware_calibration,
            initialization_gate,
        ));
        Self { jobs }
    }

    async fn execute(
        &self,
        command: PlanningWorkerCommand,
    ) -> Result<PlanningWorkerReply, InventoryError> {
        let now = tokio::time::Instant::now();
        let hard_deadline = now + MODEL_ASSESSMENT_TIMEOUT;
        let operation_deadline = match &command {
            PlanningWorkerCommand::Assess { request } => request.deadline_at_ms.map(|deadline| {
                now + std::time::Duration::from_millis(deadline.saturating_sub(unix_time_millis()))
            }),
            PlanningWorkerCommand::Initialize { .. } => None,
        };
        let deadline =
            operation_deadline.map_or(hard_deadline, |deadline| deadline.min(hard_deadline));
        let (response, result) = tokio::sync::oneshot::channel();
        tokio::time::timeout_at(
            deadline,
            self.jobs.send(PlanningJob {
                command,
                enqueued_at: std::time::Instant::now(),
                deadline,
                response,
            }),
        )
        .await
        .map_err(|_| {
            planning_worker_failure("planning_deadline", "model assessment deadline expired")
        })?
        .map_err(|_| {
            planning_worker_failure("planner_unavailable", "planning worker is unavailable")
        })?;
        tokio::time::timeout_at(deadline, result)
            .await
            .map_err(|_| {
                planning_worker_failure("planning_deadline", "model assessment deadline expired")
            })?
            .map_err(|_| {
                planning_worker_failure("planner_unavailable", "planning worker stopped")
            })?
    }

    async fn initialize(
        &self,
        hardware_calibration: Option<NativeHardwareCalibration>,
    ) -> Result<NativeHardwareCalibration, InventoryError> {
        match self
            .execute(PlanningWorkerCommand::Initialize {
                hardware_calibration,
            })
            .await?
        {
            PlanningWorkerReply::Initialized {
                hardware_calibration,
            } => Ok(hardware_calibration),
            PlanningWorkerReply::Defect { message } => Err(InventoryError::Internal(format!(
                "planning worker defect: {message}"
            ))),
            PlanningWorkerReply::Assessed { .. } => Err(InventoryError::Internal(
                "planning worker returned assessment during initialization".to_owned(),
            )),
        }
    }

    async fn assess(
        &self,
        request: PlanningWorkerRequest,
    ) -> Result<PlanningWorkerResponse, InventoryError> {
        match self
            .execute(PlanningWorkerCommand::Assess { request })
            .await?
        {
            PlanningWorkerReply::Assessed { response } => Ok(response),
            PlanningWorkerReply::Defect { message } => Err(InventoryError::Internal(format!(
                "model assessment defect: {message}"
            ))),
            PlanningWorkerReply::Initialized { .. } => Err(InventoryError::Internal(
                "planning worker returned initialization during assessment".to_owned(),
            )),
        }
    }
}

#[derive(Clone)]
struct PersistentPlanningWorkerPool {
    workers: Arc<Vec<PersistentPlanningWorker>>,
    hardware_calibration: Arc<std::sync::OnceLock<NativeHardwareCalibration>>,
    available: tokio::sync::mpsc::UnboundedSender<usize>,
    available_receiver: Arc<tokio::sync::Mutex<tokio::sync::mpsc::UnboundedReceiver<usize>>>,
}

struct PlanningWorkerLease {
    index: usize,
    available: tokio::sync::mpsc::UnboundedSender<usize>,
}

impl Drop for PlanningWorkerLease {
    fn drop(&mut self) {
        let _ = self.available.send(self.index);
    }
}

impl PersistentPlanningWorkerPool {
    fn start(launcher: NativeWorkerLauncher, size: usize) -> Self {
        let size = size.max(1);
        let hardware_calibration = Arc::new(std::sync::OnceLock::new());
        let initialization_gate = Arc::new(tokio::sync::Mutex::new(()));
        let workers = (0..size)
            .map(|_| {
                PersistentPlanningWorker::start(
                    launcher.clone(),
                    hardware_calibration.clone(),
                    initialization_gate.clone(),
                )
            })
            .collect::<Vec<_>>();
        let (available, available_receiver) = tokio::sync::mpsc::unbounded_channel();
        for index in 0..workers.len() {
            available
                .send(index)
                .expect("planning-worker availability receiver exists");
        }
        Self {
            workers: Arc::new(workers),
            hardware_calibration,
            available,
            available_receiver: Arc::new(tokio::sync::Mutex::new(available_receiver)),
        }
    }

    async fn initialize(
        &self,
        hardware_calibration: Option<NativeHardwareCalibration>,
    ) -> Result<NativeHardwareCalibration, InventoryError> {
        let established = self.workers[0].initialize(hardware_calibration).await?;
        self.hardware_calibration
            .set(established.clone())
            .map_err(|_| {
                InventoryError::Internal(
                    "planning worker pool was initialized more than once".to_owned(),
                )
            })?;
        Ok(established)
    }

    async fn assess(
        &self,
        request: PlanningWorkerRequest,
    ) -> Result<PlanningWorkerResponse, InventoryError> {
        let now = tokio::time::Instant::now();
        let hard_deadline = now + MODEL_ASSESSMENT_TIMEOUT;
        let deadline = request.deadline_at_ms.map_or(hard_deadline, |deadline| {
            (now + std::time::Duration::from_millis(deadline.saturating_sub(unix_time_millis())))
                .min(hard_deadline)
        });
        let index = tokio::time::timeout_at(deadline, async {
            self.available_receiver.lock().await.recv().await
        })
        .await
        .map_err(|_| {
            planning_worker_failure("planning_deadline", "model assessment deadline expired")
        })?
        .ok_or_else(|| {
            planning_worker_failure("planner_unavailable", "planning worker pool stopped")
        })?;
        let _lease = PlanningWorkerLease {
            index,
            available: self.available.clone(),
        };
        self.workers[index].assess(request).await
    }
}

fn planning_worker_pool_size() -> usize {
    std::thread::available_parallelism().map_or(1, |cores| cores.get().min(MAX_PLANNING_WORKERS))
}

fn planning_worker_failure(code: &str, message: &str) -> InventoryError {
    InventoryError::ModelOperation {
        code: code.to_owned(),
        message: message.to_owned(),
        retryable: true,
    }
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct TemplateWorkerRequest {
    model_path: PathBuf,
}

#[derive(Debug)]
struct NativeTemplateAssessor {
    worker_launcher: NativeWorkerLauncher,
}

impl TemplateAssessor for NativeTemplateAssessor {
    fn cache_identity(&self) -> &str {
        icn_reasoning::TEMPLATE_INSPECTION_CACHE_IDENTITY
    }

    fn assess(
        &self,
        inputs: &icn_contracts::EffectiveTemplateInputs,
    ) -> Result<TemplateAssessment, String> {
        run_isolated_template_inspection(
            TemplateWorkerRequest {
                model_path: inputs.model_path.clone(),
            },
            &self.worker_launcher,
        )
        .map_err(|error| format!("{error:#}"))
    }
}

#[cfg(not(test))]
const MAX_PLANNING_WORKER_OUTPUT_BYTES: usize = 1024 * 1024;
const HARDWARE_CALIBRATION_MAX_AGE_SECONDS: u64 = 7 * 24 * 60 * 60;
const HARDWARE_CALIBRATION_CACHE_METHOD: &str = "icn-hardware-calibration-cache-v1";
const LOW_MEMORY_FAILURE_CODE: &str = "low_memory";
#[cfg(not(test))]
const TEMPLATE_WORKER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

fn unix_time_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn unix_time_millis() -> u64 {
    u64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
    )
    .unwrap_or(u64::MAX)
}

fn normalized_backend_name(backend: &str) -> String {
    match backend.to_ascii_lowercase().as_str() {
        "mtl" => "metal".to_owned(),
        other => other.to_owned(),
    }
}

fn hardware_calibration_input_identity(
    snapshot: &HardwareSnapshot,
) -> Result<String, InventoryError> {
    let enabled_backends = snapshot
        .enabled_backends
        .iter()
        .map(|backend| normalized_backend_name(backend))
        .collect::<std::collections::BTreeSet<_>>();
    let backend_runtime = (
        enabled_backends
            .contains("cuda")
            .then(backend_eligibility::probe_cuda),
        enabled_backends
            .contains("vulkan")
            .then(backend_eligibility::probe_vulkan),
        enabled_backends
            .contains("metal")
            .then(backend_eligibility::probe_metal),
    );
    let bytes = serde_json::to_vec(&(
        HARDWARE_CALIBRATION_CACHE_METHOD,
        llama_cpp_2::model::params::fit::FIT_CALIBRATION_METHOD,
        build_identity::backend_module_abi(),
        &snapshot.native_build,
        &snapshot.enabled_backends,
        &snapshot.platform,
        &snapshot.architecture,
        &snapshot.system_product_name,
        &snapshot.cpu_model,
        snapshot.logical_cores,
        &snapshot.topology_fingerprint,
        backend_runtime,
    ))
    .map_err(|error| InventoryError::Internal(error.to_string()))?;
    Ok(format!(
        "hardware_calibration_input_{:x}",
        Sha256::digest(bytes)
    ))
}

fn hardware_calibration_covers_snapshot(
    hardware_calibration: &NativeHardwareCalibration,
    snapshot: &HardwareSnapshot,
) -> bool {
    let covers = |backend: &str, device_id: &Option<String>| {
        let backend = normalized_backend_name(backend);
        [false, true].into_iter().all(|routed| {
            hardware_calibration.metrics.iter().any(|metric| {
                normalized_backend_name(&metric.backend) == backend
                    && metric.device_id == *device_id
                    && metric.routed == routed
            })
        })
    };
    snapshot.enabled_backends.iter().all(|backend| {
        let devices = snapshot
            .memory_domains
            .iter()
            .flat_map(|domain| &domain.devices)
            .filter(|device| normalized_backend_name(&device.backend) == *backend)
            .collect::<Vec<_>>();
        if devices.is_empty() {
            covers(backend, &None)
        } else {
            devices
                .into_iter()
                .all(|device| covers(backend, &device.physical_id))
        }
    })
}

fn hardware_calibration_is_valid(hardware_calibration: &NativeHardwareCalibration) -> bool {
    if hardware_calibration.method != llama_cpp_2::model::params::fit::FIT_CALIBRATION_METHOD
        || hardware_calibration.metrics.is_empty()
    {
        return false;
    }
    let mut keys = std::collections::BTreeSet::new();
    hardware_calibration.metrics.iter().all(|metric| {
        !metric.backend.trim().is_empty()
            && metric.bytes_per_second.is_finite()
            && metric.bytes_per_second > 0.0
            && metric.launch_microseconds.is_finite()
            && metric.launch_microseconds >= 0.0
            && metric.relative_spread.is_finite()
            && metric.relative_spread >= 0.0
            && metric.sample_count > 0
            && metric.measured_microseconds > 0
            && keys.insert((
                metric.backend_type,
                metric.backend.clone(),
                metric.device_id.clone(),
                metric.tensor_type,
                metric.routed,
            ))
    })
}

fn cached_hardware_calibration(
    cache: &ModelCache,
    snapshot: &HardwareSnapshot,
    now: u64,
) -> Result<Option<NativeHardwareCalibration>, InventoryError> {
    let input_identity = hardware_calibration_input_identity(snapshot)?;
    let Some(record) = cache.read_index::<HardwareCalibrationRecord>(
        icn_models::ModelIndexKind::HardwareCalibration,
        &input_identity,
    ) else {
        return Ok(None);
    };
    Ok((record.input_identity == input_identity
        && record.measured_at_seconds <= now
        && now.saturating_sub(record.measured_at_seconds) <= HARDWARE_CALIBRATION_MAX_AGE_SECONDS
        && hardware_calibration_is_valid(&record.hardware_calibration)
        && NativeResolvedModelAssessor::hardware_calibration_identity(
            &record.hardware_calibration,
        )? == record.hardware_calibration_identity
        && hardware_calibration_covers_snapshot(&record.hardware_calibration, snapshot))
    .then_some(record.hardware_calibration))
}

fn fixture_hardware_calibration() -> NativeHardwareCalibration {
    NativeHardwareCalibration {
        method: llama_cpp_2::model::params::fit::FIT_CALIBRATION_METHOD.to_owned(),
        metrics: [false, true]
            .into_iter()
            .map(|routed| NativeHardwareCalibrationMetric {
                backend_type: 0,
                backend: "CPU".to_owned(),
                device_id: None,
                tensor_type: 0,
                routed,
                bytes_per_second: 1_000_000_000.0,
                launch_microseconds: 1.0,
                relative_spread: 0.0,
                sample_count: 1,
                measured_microseconds: 1,
                stable: true,
            })
            .collect(),
        elapsed_microseconds: 1,
    }
}

async fn discover_startup_hardware(
    native_backend: NativeBackend,
    enabled_backends: Vec<String>,
) -> Result<HardwareSnapshot, InventoryError> {
    let native_build = build_identity::native_build();
    spawn_blocking_traced(move || {
        let snapshot = native_backend.discover_hardware(
            CapacityPolicy::default(),
            native_build,
            enabled_backends,
        );
        let observation =
            icn_hardware::observe_system_memory().map_err(InventoryError::Internal)?;
        Ok(icn_hardware::with_system_memory_observation(
            snapshot,
            observation,
        ))
    })
    .await
    .map_err(|error| InventoryError::Internal(format!("hardware discovery task failed: {error}")))?
}

async fn establish_hardware_calibration(
    cache: &ModelCache,
    snapshot: &HardwareSnapshot,
    planning_workers: &PersistentPlanningWorkerPool,
) -> Result<NativeHardwareCalibration, InventoryError> {
    let input_identity = hardware_calibration_input_identity(snapshot)?;
    let now = unix_time_seconds();
    if let Some(hardware_calibration) = cached_hardware_calibration(cache, snapshot, now)? {
        let hardware_calibration = planning_workers
            .initialize(Some(hardware_calibration))
            .await?;
        tracing::info!(
            cache = "hit",
            metrics = hardware_calibration.metrics.len(),
            "hardware calibration established"
        );
        return Ok(hardware_calibration);
    }

    let hardware_calibration = planning_workers.initialize(None).await?;
    if !hardware_calibration_is_valid(&hardware_calibration)
        || !hardware_calibration_covers_snapshot(&hardware_calibration, snapshot)
    {
        return Err(InventoryError::Internal(
            "hardware calibration did not cover every enabled backend".to_owned(),
        ));
    }
    let hardware_calibration_identity =
        NativeResolvedModelAssessor::hardware_calibration_identity(&hardware_calibration)?;
    let stable_metrics = hardware_calibration
        .metrics
        .iter()
        .filter(|metric| metric.stable)
        .count();
    let total_samples = hardware_calibration
        .metrics
        .iter()
        .map(|metric| u64::from(metric.sample_count))
        .sum::<u64>();
    tracing::info!(
        cache = "miss",
        method = hardware_calibration.method,
        elapsed_microseconds = hardware_calibration.elapsed_microseconds,
        metrics = hardware_calibration.metrics.len(),
        stable_metrics,
        total_samples,
        "hardware calibration established"
    );
    cache.write_index(
        icn_models::ModelIndexKind::HardwareCalibration,
        &input_identity,
        &HardwareCalibrationRecord {
            input_identity: input_identity.clone(),
            measured_at_seconds: now,
            hardware_calibration: hardware_calibration.clone(),
            hardware_calibration_identity,
        },
    );
    Ok(hardware_calibration)
}

impl NativeResolvedModelAssessor {
    fn hardware_calibration_identity(
        hardware_calibration: &NativeHardwareCalibration,
    ) -> Result<String, InventoryError> {
        let bytes =
            serde_json::to_vec(&(&hardware_calibration.method, &hardware_calibration.metrics))
                .map_err(|error| InventoryError::Internal(error.to_string()))?;
        Ok(format!("hardware_calibration_{:x}", Sha256::digest(bytes)))
    }

    async fn assessment_work_gate(&self, key: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut gates = self.assessment_work_gates.lock().await;
        gates.retain(|_, gate| gate.strong_count() > 0);
        if let Some(gate) = gates.get(key).and_then(Weak::upgrade) {
            return gate;
        }
        let gate = Arc::new(tokio::sync::Mutex::new(()));
        gates.insert(key.to_owned(), Arc::downgrade(&gate));
        gate
    }

    fn effective_defaults(&self, profile: Option<&ModelPreviewProfile>) -> ModelPlanDefaults {
        let mut defaults = self.defaults.clone();
        if let Some(profile) = profile {
            defaults.context_size = profile.context_length;
            defaults.max_sequences = profile.parallel_sequences;
            defaults.physical_context_size = profile
                .context_length
                .saturating_mul(profile.parallel_sequences);
            defaults.execution.kv_unified = false;
        }
        defaults
    }

    async fn assess_resolved(
        &self,
        resolved: ResolvedModel,
        profile: Option<&icn_contracts::ModelPreviewProfile>,
    ) -> Result<HardwareAssessment, InventoryError> {
        let profiles = profile.cloned().into_iter().collect();
        let mut assessments = self.assess_resolved_profiles(resolved, profiles).await?;
        assessments.pop().ok_or_else(|| {
            InventoryError::Internal("native planner returned no assessment".to_owned())
        })
    }

    async fn assess_resolved_profiles(
        &self,
        resolved: ResolvedModel,
        profiles: Vec<ModelPreviewProfile>,
    ) -> Result<Vec<HardwareAssessment>, InventoryError> {
        let hardware = HardwareProvider::snapshot(self).await?;
        self.assess_resolved_capacity_plans_with_hardware(resolved, profiles, hardware)
            .await
    }

    async fn assess_resolved_execution_profiles(
        &self,
        resolved: ResolvedModel,
        profiles: Vec<ModelPreviewProfile>,
    ) -> Result<Vec<ModelExecutionAssessment>, InventoryError> {
        let hardware = HardwareProvider::snapshot(self).await?;
        self.assess_resolved_execution_plans_with_hardware(resolved, profiles, hardware)
            .await
    }

    async fn assess_resolved_plans_cached(
        &self,
        resolved: ResolvedModel,
        profiles: Vec<ModelPreviewProfile>,
        snapshot: &HardwareSnapshot,
        configuration_id: &ModelServingConfigurationId,
    ) -> Result<Vec<HardwareAssessment>, InventoryError> {
        let Some(cache) = self.cache.clone() else {
            return self
                .assess_resolved_capacity_plans_with_hardware(resolved, profiles, snapshot.clone())
                .await;
        };
        let topology = icn_contracts::MemoryTopology::from_snapshot(snapshot).ok_or_else(|| {
            InventoryError::Internal("hardware snapshot has an invalid memory topology".to_owned())
        })?;
        let content_id = resolved.model.content_id.clone();
        let mut entries = profiles
            .into_iter()
            .map(|profile| {
                let planner_evidence =
                    self.capacity_assessment_cache_key(Some(&profile), snapshot)?;
                let evidence = serde_json::to_string(&(&configuration_id.0, planner_evidence))
                    .map_err(|error| InventoryError::Internal(error.to_string()))?;
                let assessment = cache.read_hardware_assessment(&content_id, &evidence, &topology);
                Ok((profile, evidence, assessment))
            })
            .collect::<Result<Vec<_>, InventoryError>>()?;
        if entries
            .iter()
            .any(|(_, _, assessment)| assessment.is_none())
        {
            let gate_key = serde_json::to_string(&(
                &content_id.0,
                &configuration_id.0,
                entries
                    .iter()
                    .map(|(_, evidence, _)| evidence)
                    .collect::<Vec<_>>(),
            ))
            .map_err(|error| InventoryError::Internal(error.to_string()))?;
            let cache_guard = self
                .assessment_work_gate(&gate_key)
                .await
                .lock_owned()
                .await;
            for (_, evidence, assessment) in &mut entries {
                if assessment.is_none() {
                    *assessment = cache.read_hardware_assessment(&content_id, evidence, &topology);
                }
            }
            let missing = entries
                .iter()
                .enumerate()
                .filter_map(|(index, (profile, _, assessment))| {
                    assessment.is_none().then_some((
                        index,
                        profile.clone(),
                        entries[index].1.clone(),
                    ))
                })
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                let assessor = self.clone();
                let task_cache = cache.clone();
                let task_content_id = content_id.clone();
                let task_hardware = snapshot.clone();
                let planned = tokio::spawn(async move {
                    let _cache_guard = cache_guard;
                    let measured = assessor
                        .assess_resolved_capacity_plans_with_hardware(
                            resolved,
                            missing
                                .iter()
                                .map(|(_, profile, _)| profile.clone())
                                .collect(),
                            task_hardware,
                        )
                        .await?;
                    if measured.len() != missing.len() {
                        return Err(InventoryError::Internal(
                            "native planner returned the wrong number of cached assessments"
                                .to_owned(),
                        ));
                    }
                    Ok::<_, InventoryError>(
                        missing
                            .into_iter()
                            .zip(measured)
                            .map(|((index, _, evidence), assessment)| {
                                task_cache.write_hardware_assessment(
                                    &task_content_id,
                                    &evidence,
                                    &assessment,
                                );
                                (index, assessment)
                            })
                            .collect::<Vec<_>>(),
                    )
                })
                .await
                .map_err(|error| {
                    InventoryError::Internal(format!("cached native planning task failed: {error}"))
                })??;
                for (index, assessment) in planned {
                    entries[index].2 = Some(assessment);
                }
            }
        }
        entries
            .into_iter()
            .map(|(_, _, assessment)| {
                assessment.ok_or_else(|| {
                    InventoryError::Internal(
                        "assessment was neither cached nor measured".to_owned(),
                    )
                })
            })
            .collect()
    }

    async fn run_resolved_plans_with_hardware(
        &self,
        resolved: ResolvedModel,
        speculative: SpeculativeDecodingConfig,
        profiles: Vec<ModelPreviewProfile>,
        estimate_performance: bool,
        hardware: HardwareSnapshot,
        deadline_at_ms: Option<u64>,
    ) -> Result<PlanningWorkerResponse, InventoryError> {
        let id = resolved.model.id.clone();
        let primary = resolved
            .components
            .iter()
            .filter(|component| {
                matches!(
                    component.role,
                    ComponentRole::Weights | ComponentRole::Shard
                )
            })
            .min_by_key(|component| component.shard_index.unwrap_or(0))
            .map(|component| component.path.clone())
            .ok_or_else(|| InventoryError::NotReady("model has no runnable weights".into()))?;
        let projector = resolved
            .components
            .iter()
            .find(|component| component.role == ComponentRole::Projector)
            .map(|component| component.path.clone());
        let defaults = if profiles.is_empty() {
            vec![self.effective_defaults(None)]
        } else {
            profiles
                .iter()
                .map(|profile| self.effective_defaults(Some(profile)))
                .collect()
        };
        let performance_context_tokens = if profiles.is_empty() {
            defaults
                .iter()
                .map(|defaults| vec![defaults.context_size])
                .collect()
        } else {
            profiles
                .iter()
                .map(|profile| profile.performance_context_tokens.clone())
                .collect()
        };
        let hardware_calibration = if estimate_performance {
            Some(self.hardware_calibration.as_ref().clone())
        } else {
            None
        };
        let request = PlanningWorkerRequest {
            deadline_at_ms,
            hardware,
            primary,
            projector,
            speculative,
            defaults,
            performance_context_tokens,
            operation: if estimate_performance {
                PlanningOperation::Execution {
                    hardware_calibration: hardware_calibration
                        .expect("execution planning established hardware calibration"),
                }
            } else {
                PlanningOperation::Capacity
            },
        };
        let operation = request.operation.as_str();
        let profile_count = request.defaults.len();
        let worker_started = std::time::Instant::now();
        let worker_result = self.planning_executor.assess(request).await;
        let worker_microseconds =
            u64::try_from(worker_started.elapsed().as_micros()).unwrap_or(u64::MAX);
        let (outcome, error) = match &worker_result {
            Ok(_) => ("success", None),
            Err(error) => ("assessment_error", Some(error.to_string())),
        };
        tracing::info!(
            model.id = %id.0,
            operation,
            profile_count,
            worker_microseconds,
            outcome,
            error,
            "native planner completed"
        );
        match worker_result {
            Ok(response) => Ok(response),
            Err(error) => Err(error),
        }
    }

    async fn assess_resolved_capacity_plans_with_hardware(
        &self,
        resolved: ResolvedModel,
        profiles: Vec<ModelPreviewProfile>,
        hardware: HardwareSnapshot,
    ) -> Result<Vec<HardwareAssessment>, InventoryError> {
        match self
            .run_resolved_plans_with_hardware(
                resolved,
                SpeculativeDecodingConfig::default(),
                profiles,
                false,
                hardware,
                None,
            )
            .await?
        {
            PlanningWorkerResponse::Capacity { assessments } => Ok(assessments),
            PlanningWorkerResponse::Execution { .. } => Err(InventoryError::Internal(
                "capacity planner returned execution assessments".to_owned(),
            )),
        }
    }

    async fn assess_resolved_execution_plans_with_hardware(
        &self,
        resolved: ResolvedModel,
        profiles: Vec<ModelPreviewProfile>,
        hardware: HardwareSnapshot,
    ) -> Result<Vec<ModelExecutionAssessment>, InventoryError> {
        match self
            .run_resolved_plans_with_hardware(
                resolved,
                SpeculativeDecodingConfig::default(),
                profiles,
                true,
                hardware,
                None,
            )
            .await?
        {
            PlanningWorkerResponse::Execution { assessments, .. } => Ok(assessments),
            PlanningWorkerResponse::Capacity { .. } => Err(InventoryError::Internal(
                "execution planner returned capacity assessments".to_owned(),
            )),
        }
    }

    async fn assess_resolved_execution_plans_with_hardware_deadline(
        &self,
        resolved: ResolvedModel,
        speculative: SpeculativeDecodingConfig,
        profiles: Vec<ModelPreviewProfile>,
        hardware: HardwareSnapshot,
        deadline_at_ms: u64,
    ) -> Result<Vec<ModelExecutionAssessment>, InventoryError> {
        match self
            .run_resolved_plans_with_hardware(
                resolved,
                speculative,
                profiles,
                true,
                hardware,
                Some(deadline_at_ms),
            )
            .await?
        {
            PlanningWorkerResponse::Execution { assessments, .. } => Ok(assessments),
            PlanningWorkerResponse::Capacity { .. } => Err(InventoryError::Internal(
                "execution planner returned capacity assessments".to_owned(),
            )),
        }
    }

    fn capacity_assessment_cache_key(
        &self,
        profile: Option<&ModelPreviewProfile>,
        snapshot: &HardwareSnapshot,
    ) -> Result<String, InventoryError> {
        serde_json::to_string(&(
            &snapshot.native_build,
            &snapshot.enabled_backends,
            &snapshot.topology_fingerprint,
            self.effective_defaults(profile),
        ))
        .map_err(|error| InventoryError::Internal(error.to_string()))
    }

    fn execution_assessment_cache_key(
        &self,
        profile: Option<&ModelPreviewProfile>,
        snapshot: &HardwareSnapshot,
    ) -> Result<String, InventoryError> {
        serde_json::to_string(&(
            llama_cpp_2::model::params::fit::FIT_DECODE_WORKLOAD_METHOD,
            Self::hardware_calibration_identity(&self.hardware_calibration)?,
            &snapshot.native_build,
            &snapshot.enabled_backends,
            &snapshot.topology_fingerprint,
            self.effective_defaults(profile),
            profile.map(|profile| &profile.performance_context_tokens),
        ))
        .map_err(|error| InventoryError::Internal(error.to_string()))
    }

    #[cfg(test)]
    fn capacity_assessment_cache_key_with_policy(
        &self,
        profile: Option<&ModelPreviewProfile>,
        snapshot: &HardwareSnapshot,
        capacity_policy: CapacityPolicy,
    ) -> Result<String, InventoryError> {
        let snapshot = icn_hardware::with_capacity_policy(snapshot.clone(), capacity_policy);
        self.capacity_assessment_cache_key(profile, &snapshot)
    }
}

struct NativeModelAssessor {
    models: Arc<ManagedModelStore>,
    assessor: Arc<NativeResolvedModelAssessor>,
    release_catalog: Arc<ReleaseCatalog>,
}

#[derive(Clone)]
struct AssessmentEnvironment {
    id: AssessmentEnvironmentId,
    snapshot: HardwareSnapshot,
    topology: icn_contracts::MemoryTopology,
}

impl NativeModelAssessor {
    fn new(
        models: Arc<ManagedModelStore>,
        assessor: Arc<NativeResolvedModelAssessor>,
        release_catalog: Arc<ReleaseCatalog>,
    ) -> Self {
        Self {
            models,
            assessor,
            release_catalog,
        }
    }

    async fn environment(&self) -> Result<AssessmentEnvironment, InventoryError> {
        let snapshot = HardwareProvider::snapshot(self.assessor.as_ref()).await?;
        let thresholds =
            icn_hardware::system_memory_thresholds(snapshot.system_memory.physical_capacity_bytes);
        let snapshot = icn_hardware::with_capacity_policy(
            snapshot,
            CapacityPolicy {
                reserve_bytes_per_domain: ASSESSMENT_RESERVE_BYTES_PER_DEDICATED_DOMAIN,
                system_reserve_bytes: Some(thresholds.assess_reserve_bytes),
            },
        );
        let topology =
            icn_contracts::MemoryTopology::from_snapshot(&snapshot).ok_or_else(|| {
                InventoryError::Internal(
                    "hardware snapshot has an invalid memory topology".to_owned(),
                )
            })?;
        let mut digest = Sha256::new();
        let hardware_calibration_identity =
            NativeResolvedModelAssessor::hardware_calibration_identity(
                &self.assessor.hardware_calibration,
            )?;
        let identity = serde_json::to_vec(&(
            &snapshot.native_build,
            &snapshot.enabled_backends,
            &snapshot.topology_fingerprint,
            ASSESSMENT_RESERVE_BYTES_PER_DEDICATED_DOMAIN,
            thresholds.assess_reserve_bytes,
            snapshot.system_memory.physical_capacity_bytes,
            snapshot.system_memory.assess_reserve_bytes,
            hardware_calibration_identity,
        ))
        .map_err(|error| InventoryError::Internal(error.to_string()))?;
        digest.update(identity);
        Ok(AssessmentEnvironment {
            id: AssessmentEnvironmentId(format!("environment_{:x}", digest.finalize())),
            snapshot,
            topology,
        })
    }

    async fn assess_request(
        &self,
        item: AssessModelRequest,
        environment: &AssessmentEnvironment,
        release_catalog: Arc<ReleaseCatalog>,
        deadline_at_ms: u64,
    ) -> Result<AssessModelResult, InventoryError> {
        let request_id = item.request_id;
        let bundle_key = match bundle_input_key(&item.bundle) {
            Ok(bundle_key) => bundle_key,
            Err(message) => {
                return Ok(AssessModelResult::InvalidBundle {
                    request_id,
                    failure: DomainModelFailure {
                        code: "invalid_target".to_owned(),
                        message,
                        retryable: false,
                    },
                });
            }
        };
        if let Err(message) = validate_model_assessment_profiles(&item.profiles) {
            return Ok(AssessModelResult::InvalidBundle {
                request_id,
                failure: DomainModelFailure {
                    code: "invalid_profiles".to_owned(),
                    message,
                    retryable: false,
                },
            });
        }
        let cached = self
            .cached_profiles(&bundle_key, &item.profiles, environment)
            .await?;
        if let Some(profiles) = cached {
            return Ok(AssessModelResult::Assessed {
                request_id,
                profiles,
            });
        }

        let release_bundle_key = bundle_key.clone();
        let release_bundle =
            spawn_blocking_traced(move || release_catalog.resolve_bundle(&release_bundle_key))
                .await
                .map_err(|error| {
                    InventoryError::Internal(format!(
                        "release model preparation task failed for {}: {error}",
                        bundle_key.0
                    ))
                })??;
        let resolved = match release_bundle {
            Some(resolved) => Ok(resolved),
            None if bundle_uses_only_installed_packages(&item.bundle) => {
                self.models.resolve_bundle(item.bundle).await
            }
            None => Err(InventoryError::InvalidRequest(format!(
                "bundle {} is not installed or part of the release catalog",
                bundle_key.0
            ))),
        };
        match resolved {
            Ok(resolved) => Ok(AssessModelResult::Assessed {
                request_id,
                profiles: self
                    .assess_profiles(&resolved, &item.profiles, environment, deadline_at_ms)
                    .await?,
            }),
            Err(error) => Ok(AssessModelResult::InvalidBundle {
                request_id,
                failure: assessment_bundle_failure(error)?,
            }),
        }
    }

    fn resolved_for_planning(
        resolved: &icn_contracts::models::ResolvedServableModelBundle,
    ) -> ResolvedModel {
        let mut target = resolved.target_model.clone();
        if let Some(draft) = &resolved.draft_model {
            target
                .components
                .extend(draft.components.iter().cloned().map(|mut component| {
                    component.role = ComponentRole::Draft;
                    component
                }));
        }
        target
    }

    fn speculative_config_for_resolved(
        resolved: &icn_contracts::models::ResolvedServableModelBundle,
    ) -> Result<SpeculativeDecodingConfig, InventoryError> {
        let DomainServableModelBundle::SpeculativeDecoding {
            draft_source,
            method,
            ..
        } = &resolved.bundle
        else {
            return Ok(SpeculativeDecodingConfig::Disabled {
                reason: "standalone_bundle".to_owned(),
            });
        };
        let method = match method {
            SpeculativeMethod::Mtp => SpeculativeMethodConfig::Mtp {
                min_draft_probability: 0.0,
            },
            SpeculativeMethod::DFlash => SpeculativeMethodConfig::DFlash {
                min_sample_probability: 0.0,
            },
            SpeculativeMethod::DSpark => SpeculativeMethodConfig::DSpark {
                acceptance_threshold: 0.0,
            },
        };
        let source = match draft_source {
            ModelSpeculativeDraftSource::Embedded => {
                if resolved.draft_model.is_some() {
                    return Err(InventoryError::InvalidRequest(
                        "embedded speculative bundle unexpectedly resolved a separate draft"
                            .to_owned(),
                    ));
                }
                SpeculativeDraftSource::Embedded
            }
            ModelSpeculativeDraftSource::Separate { .. } => {
                let draft = resolved.draft_model.as_ref().ok_or_else(|| {
                    InventoryError::NotReady(
                        "separate speculative bundle did not resolve its draft package".to_owned(),
                    )
                })?;
                let model_path = draft
                    .components
                    .iter()
                    .filter(|component| {
                        matches!(
                            component.role,
                            ComponentRole::Weights | ComponentRole::Shard
                        )
                    })
                    .min_by_key(|component| component.shard_index.unwrap_or(0))
                    .map(|component| component.path.clone())
                    .ok_or_else(|| {
                        InventoryError::NotReady(
                            "separate speculative package has no runnable weights".to_owned(),
                        )
                    })?;
                SpeculativeDraftSource::Separate { model_path }
            }
        };
        Ok(SpeculativeDecodingConfig::Enabled {
            source,
            method,
            n_max: 3,
            n_min: 0,
            cache_type_k: CacheType::F16,
            cache_type_v: CacheType::F16,
        })
    }

    async fn assessment_evidence(
        &self,
        bundle_key: &icn_contracts::models::ServableModelBundleKey,
        profiles: &[DomainModelAssessmentProfile],
        environment: &AssessmentEnvironment,
    ) -> Result<Vec<String>, InventoryError> {
        let calibration_identity = NativeResolvedModelAssessor::hardware_calibration_identity(
            &self.assessor.hardware_calibration,
        )?;
        profiles
            .iter()
            .map(|profile| {
                let defaults = self.assessor.effective_defaults(Some(&ModelPreviewProfile {
                    id: "assessment-cache-key".to_owned(),
                    context_length: profile.profile.context_length,
                    parallel_sequences: 1,
                    performance_context_tokens: profile.performance_context_tokens.clone(),
                }));
                serde_json::to_string(&(
                    llama_cpp_2::model::params::fit::FIT_DECODE_WORKLOAD_METHOD,
                    &calibration_identity,
                    &environment.id.0,
                    &bundle_key.0,
                    defaults,
                    &profile.performance_context_tokens,
                ))
                .map_err(|error| InventoryError::Internal(error.to_string()))
            })
            .collect()
    }

    async fn cached_profiles(
        &self,
        bundle_key: &icn_contracts::models::ServableModelBundleKey,
        profiles: &[DomainModelAssessmentProfile],
        environment: &AssessmentEnvironment,
    ) -> Result<Option<Vec<ModelAssessment>>, InventoryError> {
        let evidence = self
            .assessment_evidence(bundle_key, profiles, environment)
            .await?;
        let results = evidence
            .iter()
            .map(|key| {
                self.models
                    .read_model_assessment(key, &environment.topology)
            })
            .collect::<Option<Vec<_>>>();
        tracing::info!(
            target.id = %bundle_key.0,
            profile_count = profiles.len(),
            cache = if results.is_some() { "hit" } else { "partial_or_miss" },
            "model assessment cache checked"
        );
        Ok(results)
    }

    async fn assess_profiles(
        &self,
        resolved: &icn_contracts::models::ResolvedServableModelBundle,
        profiles: &[DomainModelAssessmentProfile],
        environment: &AssessmentEnvironment,
        deadline_at_ms: u64,
    ) -> Result<Vec<ModelAssessment>, InventoryError> {
        let hardware = &environment.snapshot;
        let thresholds =
            icn_hardware::system_memory_thresholds(hardware.system_memory.physical_capacity_bytes);
        let evidence = self
            .assessment_evidence(&resolved.bundle_key, profiles, environment)
            .await?;
        // Serialize misses for one immutable target in one assessment environment. The waiter
        // rechecks every exact profile key after admission, so overlapping requests reuse results
        // produced by the current owner instead of opening the same model concurrently.
        let gate_key = serde_json::to_string(&(&resolved.bundle_key.0, &environment.id.0))
            .map_err(|error| InventoryError::Internal(error.to_string()))?;
        let gate = self
            .assessor
            .assessment_work_gate(&gate_key)
            .await
            .lock_owned()
            .await;
        let models = Arc::clone(&self.models);
        let assessor = Arc::clone(&self.assessor);
        let resolved = resolved.clone();
        let profiles = profiles.to_vec();
        let environment = environment.clone();
        tokio::spawn(async move {
            let _gate = gate;
            let mut results = evidence
                .iter()
                .map(|key| models.read_model_assessment(key, &environment.topology))
                .collect::<Vec<_>>();
            let missing = results
                .iter()
                .enumerate()
                .filter_map(|(index, assessment)| assessment.is_none().then_some(index))
                .collect::<Vec<_>>();
            tracing::info!(
                target.id = %resolved.bundle_key.0,
                profile_count = profiles.len(),
                missing_profile_count = missing.len(),
                cache_hit_count = profiles.len().saturating_sub(missing.len()),
                "model assessment batch resolved"
            );
            if missing.is_empty() {
                return Ok(results
                    .into_iter()
                    .map(|assessment| assessment.expect("cache hit was checked"))
                    .collect());
            }
            let native_profiles = missing
                .iter()
                .map(|index| ModelPreviewProfile {
                    id: format!("assessment-{index}"),
                    context_length: profiles[*index].profile.context_length,
                    parallel_sequences: 1,
                    performance_context_tokens: profiles[*index].performance_context_tokens.clone(),
                })
                .collect::<Vec<_>>();
            let assessed = assessor
                .assess_resolved_execution_plans_with_hardware_deadline(
                    Self::resolved_for_planning(&resolved),
                    Self::speculative_config_for_resolved(&resolved)?,
                    native_profiles,
                    environment.snapshot.clone(),
                    deadline_at_ms,
                )
                .await?;
            if assessed.len() != missing.len() {
                return Err(InventoryError::Internal(
                    "native assessment returned the wrong profile count".to_owned(),
                ));
            }
            for (index, assessment) in missing.into_iter().zip(assessed) {
                let assessment = model_assessment(
                    &resolved.bundle_key,
                    &resolved.bundle,
                    profiles[index].clone(),
                    &environment.id,
                    ASSESSMENT_RESERVE_BYTES_PER_DEDICATED_DOMAIN,
                    thresholds.assess_reserve_bytes,
                    assessment,
                )?;
                models.write_model_assessment(&evidence[index], &assessment);
                results[index] = Some(assessment);
            }
            results
                .into_iter()
                .map(|assessment| {
                    assessment.ok_or_else(|| {
                        InventoryError::Internal(
                            "assessment was neither cached nor measured".to_owned(),
                        )
                    })
                })
                .collect()
        })
        .await
        .map_err(|error| {
            InventoryError::Internal(format!("model assessment owner task failed: {error}"))
        })?
    }
}

fn model_assessment(
    bundle_key: &icn_contracts::models::ServableModelBundleKey,
    bundle: &DomainServableModelBundle,
    assessment_profile: DomainModelAssessmentProfile,
    environment_id: &AssessmentEnvironmentId,
    reserve_bytes: u64,
    system_reserve_bytes: u64,
    assessment: ModelExecutionAssessment,
) -> Result<ModelAssessment, InventoryError> {
    let DomainModelAssessmentProfile {
        profile,
        performance_context_tokens,
    } = assessment_profile;
    let configuration_id = serving_configuration_id(bundle_key, &profile);
    let configuration = ModelServingConfiguration {
        id: configuration_id,
        bundle: bundle.clone(),
        profile: profile.clone(),
    };
    let mut digest = Sha256::new();
    digest.update(bundle_key.0.as_bytes());
    digest.update(profile.context_length.to_le_bytes());
    digest.update(environment_id.0.as_bytes());
    digest.update(reserve_bytes.to_le_bytes());
    digest.update(system_reserve_bytes.to_le_bytes());
    for context_tokens in performance_context_tokens {
        digest.update(context_tokens.to_le_bytes());
    }
    let assessment_id = ModelAssessmentId(format!("assessment_{:x}", digest.finalize()));
    let (hardware, performance) = match assessment {
        ModelExecutionAssessment::Executable {
            hardware,
            performance,
        } => (hardware, Some(performance)),
        ModelExecutionAssessment::NotExecutable { hardware } => (hardware, None),
    };
    match hardware {
        HardwareAssessment::Fits { memory, .. } => {
            let performance = performance.ok_or_else(|| {
                InventoryError::Internal(
                    "executable hardware assessment omitted measured performance".to_owned(),
                )
            })?;
            if performance.is_empty() {
                return Err(InventoryError::Internal(
                    "executable hardware assessment returned no performance samples".to_owned(),
                ));
            }
            Ok(ModelAssessment::Fits {
                configuration,
                assessment_id,
                memory: memory
                    .domains
                    .into_iter()
                    .map(|domain| {
                        let domain_reserve = if domain.memory_domain.is_system() {
                            system_reserve_bytes
                        } else {
                            reserve_bytes
                        };
                        MemoryAssessment {
                            compatibility_reserve_bytes: domain_reserve,
                            memory_domain_id: domain.memory_domain,
                            capacity_bytes: domain
                                .usable_capacity_bytes
                                .saturating_add(domain_reserve),
                            required_bytes: domain.required_bytes,
                            remaining_bytes: domain.margin_bytes,
                        }
                    })
                    .collect(),
                performance: performance.into_iter().map(performance_result).collect(),
            })
        }
        HardwareAssessment::DoesNotFit {
            memory,
            limiting_resource,
            ..
        } => Ok(ModelAssessment::DoesNotFit {
            configuration,
            assessment_id,
            memory: memory
                .domains
                .into_iter()
                .map(|domain| {
                    let domain_reserve = if domain.memory_domain.is_system() {
                        system_reserve_bytes
                    } else {
                        reserve_bytes
                    };
                    MemoryAssessment {
                        compatibility_reserve_bytes: domain_reserve,
                        memory_domain_id: domain.memory_domain,
                        capacity_bytes: domain.usable_capacity_bytes.saturating_add(domain_reserve),
                        required_bytes: domain.required_bytes,
                        remaining_bytes: domain.margin_bytes,
                    }
                })
                .collect(),
            limiting_resource,
            deficit_bytes: memory.deficit_bytes.max(1),
        }),
        HardwareAssessment::InvalidArtifact { code, message } => {
            Ok(ModelAssessment::Incompatible {
                configuration,
                failure: DomainModelFailure {
                    code,
                    message,
                    retryable: false,
                },
            })
        }
        HardwareAssessment::IncompatibleArtifact { code, message } => {
            Ok(ModelAssessment::Incompatible {
                configuration,
                failure: DomainModelFailure {
                    code,
                    message,
                    retryable: false,
                },
            })
        }
        HardwareAssessment::NotAssessed { reason } => Err(InventoryError::Internal(format!(
            "native assessment produced no result: {reason}"
        ))),
    }
}

fn performance_result(assessment: GenerationPerformanceAssessment) -> PerformanceEvidence {
    PerformanceEvidence {
        context_tokens: assessment.context_tokens,
        lower_tokens_per_second: assessment.lower_tokens_per_second,
        estimated_tokens_per_second: assessment.expected_tokens_per_second,
        upper_tokens_per_second: assessment.upper_tokens_per_second,
        confidence: match assessment.confidence {
            icn_contracts::GenerationPerformanceConfidence::High => PerformanceConfidence::High,
            icn_contracts::GenerationPerformanceConfidence::Moderate => {
                PerformanceConfidence::Moderate
            }
            icn_contracts::GenerationPerformanceConfidence::Low => PerformanceConfidence::Low,
        },
    }
}

fn package_operand_id(operand: &ModelPackageOperand) -> Result<&ModelPackageId, String> {
    match operand {
        ModelPackageOperand::Installed { package_id } => Ok(package_id),
        ModelPackageOperand::SourceBacked { package } => {
            let canonical = canonical_package_id(&package.files, &package.relationships);
            if canonical != package.id {
                return Err("source-backed package identity does not match its files".to_owned());
            }
            Ok(&package.id)
        }
    }
}

fn bundle_input_key(
    bundle: &ModelBundleInput,
) -> Result<icn_contracts::models::ServableModelBundleKey, String> {
    match bundle {
        ModelBundleInput::Standalone { package } => {
            Ok(servable_model_bundle_key(&[package_operand_id(package)?]))
        }
        ModelBundleInput::SpeculativeDecoding {
            target,
            draft_source,
            method,
        } => Ok(speculative_servable_model_bundle_key(
            package_operand_id(target)?,
            match draft_source {
                SpeculativeDraftSourceInput::Embedded => None,
                SpeculativeDraftSourceInput::Separate { draft } => Some(package_operand_id(draft)?),
            },
            method,
        )),
    }
}

fn validate_model_assessment_profiles(
    profiles: &[DomainModelAssessmentProfile],
) -> Result<(), String> {
    if profiles.is_empty() || profiles.len() > 16 {
        return Err("model assessment requires between one and sixteen profiles".to_owned());
    }
    for requested in profiles {
        let contexts = &requested.performance_context_tokens;
        if requested.profile.context_length == 0
            || contexts.is_empty()
            || contexts.last() != Some(&requested.profile.context_length)
            || contexts.windows(2).any(|pair| pair[0] >= pair[1])
            || contexts
                .iter()
                .any(|context| *context == 0 || *context > requested.profile.context_length)
        {
            return Err(
                "performance sample contexts must be unique, ascending, and end at the profile context"
                    .to_owned(),
            );
        }
    }
    Ok(())
}

fn bundle_uses_only_installed_packages(bundle: &ModelBundleInput) -> bool {
    match bundle {
        ModelBundleInput::Standalone { package } => {
            matches!(package, ModelPackageOperand::Installed { .. })
        }
        ModelBundleInput::SpeculativeDecoding {
            target,
            draft_source,
            ..
        } => {
            matches!(target, ModelPackageOperand::Installed { .. })
                && match draft_source {
                    SpeculativeDraftSourceInput::Embedded => true,
                    SpeculativeDraftSourceInput::Separate { draft } => {
                        matches!(draft, ModelPackageOperand::Installed { .. })
                    }
                }
        }
    }
}

fn assessment_bundle_failure(error: InventoryError) -> Result<DomainModelFailure, InventoryError> {
    match error {
        error @ (InventoryError::InvalidId(_)
        | InventoryError::InvalidRequest(_)
        | InventoryError::NotFound(_)) => Ok(DomainModelFailure {
            code: "invalid_target".to_owned(),
            message: error.to_string(),
            retryable: false,
        }),
        InventoryError::ModelOperation {
            code,
            message,
            retryable: false,
        } => Ok(DomainModelFailure {
            code,
            message,
            retryable: false,
        }),
        error => Err(error),
    }
}

fn inventory_model_failure(error: InventoryError) -> DomainModelFailure {
    let (code, retryable) = match &error {
        InventoryError::InvalidId(_) => ("invalid_id".to_owned(), false),
        InventoryError::InvalidRequest(_) => ("invalid_request".to_owned(), false),
        InventoryError::NotFound(_) => ("not_found".to_owned(), false),
        InventoryError::NotReady(_) => ("not_ready".to_owned(), true),
        InventoryError::Busy(_) => ("busy".to_owned(), true),
        InventoryError::Loaded(_) => ("already_loaded".to_owned(), false),
        InventoryError::DeletionUnsafe(_) => ("deletion_unsafe".to_owned(), false),
        InventoryError::Unsupported(_) => ("unsupported".to_owned(), false),
        InventoryError::Io(_) => ("io_failed".to_owned(), true),
        InventoryError::Upstream(_) => ("upstream_failed".to_owned(), true),
        InventoryError::Integrity(_) => ("integrity_failed".to_owned(), false),
        InventoryError::ConcurrentMutation(_) => ("concurrent_mutation".to_owned(), true),
        InventoryError::ModelOperation {
            code, retryable, ..
        } => (code.clone(), *retryable),
        InventoryError::Internal(_) => ("internal".to_owned(), true),
    };
    DomainModelFailure {
        code,
        message: error.to_string(),
        retryable,
    }
}

fn failed_assessment_result(
    request_id: icn_contracts::models::ModelAssessmentRequestId,
    error: InventoryError,
) -> AssessModelResult {
    AssessModelResult::Failed {
        request_id,
        failure: inventory_model_failure(error),
    }
}

impl ModelAssessor for NativeModelAssessor {
    fn assess(
        &self,
        request: AssessModelsRequest,
    ) -> BoxFuture<'_, Result<AssessModelsResponse, InventoryError>> {
        Box::pin(async move {
            let remaining = MODEL_ASSESSMENT_TIMEOUT;
            let deadline_at_ms = unix_time_millis()
                .saturating_add(u64::try_from(remaining.as_millis()).unwrap_or(u64::MAX));
            tokio::time::timeout(remaining, async move {
                let environment = self.environment().await?;
                let release_catalog = Arc::clone(&self.release_catalog);
                let evaluated =
                    futures_util::stream::iter(request.requests.into_iter().enumerate())
                        .map(|(index, item)| {
                            let environment = environment.clone();
                            let release_catalog = Arc::clone(&release_catalog);
                            let failed_request_id = item.request_id.clone();
                            async move {
                                self.assess_request(
                                    item,
                                    &environment,
                                    release_catalog,
                                    deadline_at_ms,
                                )
                                .await
                            }
                            .map(move |evaluated| {
                                let result = evaluated.unwrap_or_else(|error| {
                                    failed_assessment_result(failed_request_id, error)
                                });
                                (index, result)
                            })
                        })
                        .buffer_unordered(self.assessor.assessment_concurrency.concurrency())
                        .collect::<Vec<_>>()
                        .await;
                let mut results = evaluated;
                results.sort_unstable_by_key(|(index, _)| *index);
                Ok(AssessModelsResponse {
                    environment_id: environment.id,
                    results: results.into_iter().map(|(_, result)| result).collect(),
                })
            })
            .await
            .map_err(|_| {
                planning_worker_failure(
                    "planning_deadline",
                    "model assessment operation deadline expired",
                )
            })?
        })
    }
}

fn assessment_orchestration_concurrency() -> usize {
    std::thread::available_parallelism().map_or(1, |cores| {
        cores.get().min(MAX_ASSESSMENT_ORCHESTRATION_CONCURRENCY)
    })
}

fn assess_planning_request_with_backend(
    request: PlanningWorkerRequest,
    native_backend: &NativeBackend,
) -> anyhow::Result<PlanningWorkerResponse> {
    let PlanningWorkerRequest {
        hardware,
        primary,
        projector,
        speculative,
        defaults,
        performance_context_tokens,
        operation,
        ..
    } = request;
    let backend = native_backend.as_llama_backend();
    let topology = icn_contracts::MemoryTopology::from_snapshot(&hardware)
        .context("planning request contains an invalid memory topology")?;
    let mut plans = defaults
        .into_iter()
        .map(|defaults| execution_intent(primary.clone(), projector.clone(), &defaults))
        .collect::<Vec<_>>();
    let hardware_calibration = match operation {
        PlanningOperation::Capacity => {
            let base = icn_hardware::assess_profiles_with_backend(backend, &topology, &plans)?;
            let assessments = plans
                .iter_mut()
                .zip(base)
                .map(|(plan, base)| {
                    if !matches!(base, HardwareAssessment::Fits { .. }) {
                        return Ok(base);
                    }
                    plan.speculative = speculative.clone();
                    plan.speculative = icn_speculative::preflight_with_backend(backend, plan)
                        .context("failed to preflight the speculative bundle")?;
                    if matches!(
                        plan.speculative,
                        icn_contracts::SpeculativeDecodingConfig::Disabled { .. }
                    ) {
                        return Ok(base);
                    }
                    Ok(icn_hardware::assess_with_backend(backend, &topology, plan)?.assessment)
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            return Ok(PlanningWorkerResponse::Capacity { assessments });
        }
        PlanningOperation::Execution {
            hardware_calibration,
        } => hardware_calibration,
    };
    let base = icn_hardware::assess_execution_profiles_with_backend(
        backend,
        &topology,
        &plans,
        &hardware_calibration,
        &performance_context_tokens,
    )?;
    let assessments = plans
        .iter_mut()
        .zip(base)
        .map(|(plan, base)| {
            let ModelExecutionAssessment::Executable {
                hardware: base_hardware,
                performance,
            } = base
            else {
                return Ok(base);
            };
            debug_assert!(matches!(base_hardware, HardwareAssessment::Fits { .. }));
            plan.speculative = speculative.clone();
            plan.speculative = icn_speculative::preflight_with_backend(backend, plan)
                .context("failed to preflight the speculative bundle")?;
            if matches!(
                plan.speculative,
                icn_contracts::SpeculativeDecodingConfig::Disabled { .. }
            ) {
                return Ok(ModelExecutionAssessment::Executable {
                    hardware: base_hardware,
                    performance,
                });
            }
            let hardware = icn_hardware::assess_with_backend(backend, &topology, plan)?.assessment;
            if matches!(hardware, HardwareAssessment::Fits { .. }) {
                // Phase 1 intentionally estimates baseline target-model decode. Speculation
                // changes memory but is not credited with an unmeasured speedup.
                Ok(ModelExecutionAssessment::Executable {
                    hardware,
                    performance,
                })
            } else {
                Ok(ModelExecutionAssessment::NotExecutable { hardware })
            }
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(PlanningWorkerResponse::Execution { assessments })
}

#[cfg(test)]
fn test_native_backend() -> NativeBackend {
    static BACKEND: std::sync::OnceLock<NativeBackend> = std::sync::OnceLock::new();
    BACKEND
        .get_or_init(|| NativeBackend::initialize().expect("initialize test native backend"))
        .clone()
}

async fn run_planning_worker_supervisor(
    launcher: NativeWorkerLauncher,
    mut jobs: tokio::sync::mpsc::Receiver<PlanningJob>,
    shared_hardware_calibration: Arc<std::sync::OnceLock<NativeHardwareCalibration>>,
    initialization_gate: Arc<tokio::sync::Mutex<()>>,
) {
    let mut child: Option<PlanningWorkerProcess> = None;
    let mut initialized = false;
    let mut hardware_calibration: Option<NativeHardwareCalibration> = None;
    while let Some(job) = jobs.recv().await {
        let queue_microseconds =
            u64::try_from(job.enqueued_at.elapsed().as_micros()).unwrap_or(u64::MAX);
        let worker_started = std::time::Instant::now();
        let mut result = async {
            if tokio::time::Instant::now() >= job.deadline {
                return Err(planning_worker_failure(
                    "planning_deadline",
                    "model assessment deadline expired in the planning queue",
                ));
            }
            if child.is_none() {
                child = Some(spawn_planning_worker(&launcher)?);
                initialized = false;
            }
            let initialization_guard = if initialized {
                None
            } else {
                Some(initialization_gate.lock().await)
            };
            if !initialized && !matches!(job.command, PlanningWorkerCommand::Initialize { .. }) {
                let calibration = hardware_calibration
                    .clone()
                    .or_else(|| shared_hardware_calibration.get().cloned())
                    .ok_or_else(|| {
                        planning_worker_failure(
                            "planner_unavailable",
                            "planning worker has no hardware calibration",
                        )
                    })?;
                let reply = exchange_planning_frame(
                    &mut child
                        .as_mut()
                        .expect("planning child was established")
                        .child,
                    &PlanningWorkerCommand::Initialize {
                        hardware_calibration: Some(calibration),
                    },
                    job.deadline,
                )
                .await?;
                match reply {
                    PlanningWorkerReply::Initialized {
                        hardware_calibration: restored,
                    } => {
                        hardware_calibration = Some(restored);
                        initialized = true;
                    }
                    PlanningWorkerReply::Defect { message } => {
                        return Err(InventoryError::Internal(format!(
                            "planning worker initialization defect: {message}"
                        )));
                    }
                    PlanningWorkerReply::Assessed { .. } => {
                        return Err(InventoryError::Internal(
                            "planning worker assessed before initialization".to_owned(),
                        ));
                    }
                }
                drop(initialization_guard);
            }
            let reply = exchange_planning_frame(
                &mut child
                    .as_mut()
                    .expect("planning child was established")
                    .child,
                &job.command,
                job.deadline,
            )
            .await?;
            if let PlanningWorkerReply::Defect { message } = &reply {
                return Err(InventoryError::Internal(format!(
                    "planning worker defect: {message}"
                )));
            }
            if let PlanningWorkerReply::Initialized {
                hardware_calibration: established,
            } = &reply
            {
                hardware_calibration = Some(established.clone());
                initialized = true;
            }
            Ok(reply)
        }
        .await;
        let worker_microseconds =
            u64::try_from(worker_started.elapsed().as_micros()).unwrap_or(u64::MAX);
        tracing::info!(
            queue_microseconds,
            worker_microseconds,
            outcome = if result.is_ok() { "success" } else { "failure" },
            "planning worker command completed"
        );
        if result.is_err() {
            if let Some(failed) = child.take() {
                let diagnostics = failed.terminate();
                if !diagnostics.is_empty() {
                    tracing::warn!(
                        diagnostics,
                        "planning worker failed with native diagnostics"
                    );
                    result = result
                        .map_err(|error| planning_failure_with_diagnostics(error, &diagnostics));
                }
            }
            initialized = false;
        }
        let _ = job.response.send(result);
    }
    if let Some(child) = child {
        child.terminate();
    }
}

fn spawn_planning_worker(
    launcher: &NativeWorkerLauncher,
) -> Result<PlanningWorkerProcess, InventoryError> {
    let command = launcher
        .command(NativeWorkerRole::Planning)
        .map_err(|error| InventoryError::Internal(error.to_string()))?;
    let mut command = tokio::process::Command::from(command);
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| {
            planning_worker_failure(
                "planner_spawn_failed",
                &format!("failed to start planning worker: {error}"),
            )
        })?;
    let stderr = child.stderr.take().ok_or_else(|| {
        planning_worker_failure("planner_protocol", "planning worker stderr is unavailable")
    })?;
    let stderr_tail = Arc::new(std::sync::Mutex::new(Vec::new()));
    let stderr_reader = tokio::spawn(drain_planning_worker_stderr(
        stderr,
        Arc::clone(&stderr_tail),
    ));
    Ok(PlanningWorkerProcess {
        child,
        stderr_tail,
        stderr_reader,
    })
}

async fn drain_planning_worker_stderr(
    mut stderr: tokio::process::ChildStderr,
    tail: Arc<std::sync::Mutex<Vec<u8>>>,
) {
    use tokio::io::AsyncReadExt as _;

    let mut chunk = [0_u8; 8 * 1024];
    loop {
        let Ok(read) = stderr.read(&mut chunk).await else {
            return;
        };
        if read == 0 {
            return;
        }
        if let Ok(mut retained) = tail.lock() {
            retain_planning_diagnostics(&mut retained, &chunk[..read]);
        }
    }
}

fn retain_planning_diagnostics(retained: &mut Vec<u8>, chunk: &[u8]) {
    retained.extend_from_slice(chunk);
    let excess = retained.len().saturating_sub(MAX_PLANNING_DIAGNOSTIC_BYTES);
    if excess > 0 {
        retained.drain(..excess);
    }
}

fn planning_failure_with_diagnostics(error: InventoryError, diagnostics: &str) -> InventoryError {
    let diagnostics = diagnostics
        .chars()
        .rev()
        .take(4_096)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    match error {
        InventoryError::ModelOperation {
            code,
            message,
            retryable,
        } => InventoryError::ModelOperation {
            code,
            message: format!("{message}; planner diagnostics: {diagnostics}"),
            retryable,
        },
        InventoryError::Internal(message) => {
            InventoryError::Internal(format!("{message}; planner diagnostics: {diagnostics}"))
        }
        error => error,
    }
}

async fn exchange_planning_frame(
    child: &mut tokio::process::Child,
    command: &PlanningWorkerCommand,
    deadline: tokio::time::Instant,
) -> Result<PlanningWorkerReply, InventoryError> {
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

    let payload =
        serde_json::to_vec(command).map_err(|error| InventoryError::Internal(error.to_string()))?;
    if payload.len() > MAX_PLANNING_FRAME_BYTES {
        return Err(InventoryError::InvalidRequest(
            "planning request exceeds its frame bound".to_owned(),
        ));
    }
    let length = u32::try_from(payload.len())
        .map_err(|_| InventoryError::InvalidRequest("planning request is too large".to_owned()))?;
    let stdin = child.stdin.as_mut().ok_or_else(|| {
        planning_worker_failure("planner_protocol", "planning worker stdin is unavailable")
    })?;
    tokio::time::timeout_at(deadline, async {
        stdin.write_all(&length.to_le_bytes()).await?;
        stdin.write_all(&payload).await?;
        stdin.flush().await
    })
    .await
    .map_err(|_| planning_worker_failure("planning_deadline", "planning worker write timed out"))?
    .map_err(|error| planning_worker_failure("planner_protocol", &error.to_string()))?;

    let stdout = child.stdout.as_mut().ok_or_else(|| {
        planning_worker_failure("planner_protocol", "planning worker stdout is unavailable")
    })?;
    let payload = tokio::time::timeout_at(deadline, async {
        let mut length = [0_u8; 4];
        stdout.read_exact(&mut length).await?;
        let length = usize::try_from(u32::from_le_bytes(length)).unwrap_or(usize::MAX);
        if length > MAX_PLANNING_FRAME_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "planning response exceeds its frame bound",
            ));
        }
        let mut payload = vec![0_u8; length];
        stdout.read_exact(&mut payload).await?;
        Ok::<_, std::io::Error>(payload)
    })
    .await
    .map_err(|_| {
        planning_worker_failure("planning_deadline", "planning worker execution timed out")
    })?
    .map_err(|error| planning_worker_failure("planner_protocol", &error.to_string()))?;
    serde_json::from_slice(&payload).map_err(|error| {
        InventoryError::Internal(format!(
            "planning worker returned malformed output: {error}"
        ))
    })
}

fn read_planning_frame(
    reader: &mut impl std::io::Read,
) -> anyhow::Result<Option<PlanningWorkerCommand>> {
    let mut length = [0_u8; 4];
    match reader.read_exact(&mut length) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(error) => return Err(error.into()),
    }
    let length = usize::try_from(u32::from_le_bytes(length)).unwrap_or(usize::MAX);
    anyhow::ensure!(
        length <= MAX_PLANNING_FRAME_BYTES,
        "planning request exceeds its frame bound"
    );
    let mut payload = vec![0_u8; length];
    reader.read_exact(&mut payload)?;
    Ok(Some(serde_json::from_slice(&payload)?))
}

fn write_planning_frame(
    writer: &mut impl std::io::Write,
    reply: &PlanningWorkerReply,
) -> anyhow::Result<()> {
    let payload = serde_json::to_vec(reply)?;
    anyhow::ensure!(
        payload.len() <= MAX_PLANNING_FRAME_BYTES,
        "planning response exceeds its frame bound"
    );
    writer.write_all(&u32::try_from(payload.len())?.to_le_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()?;
    Ok(())
}

fn run_planning_worker(authority: NativeRuntimeAuthority) -> anyhow::Result<()> {
    let native_backend = initialize_native_runtime(&authority)?;
    let mut input = std::io::stdin().lock();
    let mut output = std::io::stdout().lock();
    let mut hardware_calibration: Option<NativeHardwareCalibration> = None;
    while let Some(command) = read_planning_frame(&mut input)? {
        let reply = match command {
            PlanningWorkerCommand::Initialize {
                hardware_calibration: supplied,
            } => {
                let established = match supplied.or_else(|| hardware_calibration.clone()) {
                    Some(calibration) => calibration,
                    None => NativeHardwareCalibration::measure(native_backend.as_llama_backend())?,
                };
                hardware_calibration = Some(established.clone());
                PlanningWorkerReply::Initialized {
                    hardware_calibration: established,
                }
            }
            PlanningWorkerCommand::Assess { request } => {
                if hardware_calibration.is_none() {
                    PlanningWorkerReply::Defect {
                        message: "planning worker was not initialized".to_owned(),
                    }
                } else {
                    match assess_planning_request_with_backend(request, &native_backend) {
                        Ok(response) => PlanningWorkerReply::Assessed { response },
                        Err(error) => PlanningWorkerReply::Defect {
                            message: format!("{error:#}"),
                        },
                    }
                }
            }
        };
        write_planning_frame(&mut output, &reply)?;
    }
    Ok(())
}

fn inspect_template_request_with_backend(
    request: TemplateWorkerRequest,
    native_backend: &NativeBackend,
) -> anyhow::Result<TemplateAssessment> {
    let inspection = icn_reasoning::inspect_template_inputs_with_backend(
        native_backend.as_llama_backend(),
        &icn_contracts::EffectiveTemplateInputs {
            model_path: request.model_path,
        },
    )?;
    Ok(TemplateAssessment {
        capabilities: inspection.capabilities,
        reasoning: inspection.reasoning,
        fingerprint: inspection.template_fingerprint,
    })
}

#[cfg(test)]
fn run_isolated_template_inspection(
    request: TemplateWorkerRequest,
    _worker_launcher: &NativeWorkerLauncher,
) -> anyhow::Result<TemplateAssessment> {
    let native_backend = test_native_backend();
    inspect_template_request_with_backend(request, &native_backend)
}

#[cfg(not(test))]
fn run_isolated_template_inspection(
    request: TemplateWorkerRequest,
    worker_launcher: &NativeWorkerLauncher,
) -> anyhow::Result<TemplateAssessment> {
    run_isolated_json_worker(
        worker_launcher,
        NativeWorkerRole::Template,
        Some(&request),
        TEMPLATE_WORKER_TIMEOUT,
        "native template inspection",
    )
}

#[cfg(not(test))]
fn run_isolated_json_worker<Request, Response>(
    worker_launcher: &NativeWorkerLauncher,
    role: NativeWorkerRole,
    request: Option<&Request>,
    timeout: std::time::Duration,
    operation: &str,
) -> anyhow::Result<Response>
where
    Request: serde::Serialize,
    Response: serde::de::DeserializeOwned,
{
    let encoded_request = request
        .map(serde_json::to_vec)
        .transpose()
        .with_context(|| format!("failed to encode isolated {operation} request"))?;
    let mut child = worker_launcher.command(role)?;
    if encoded_request.is_some() {
        child.stdin(Stdio::piped());
    }
    child.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = child
        .spawn()
        .with_context(|| format!("failed to start isolated {operation}"))?;
    let output_bound_exceeded = Arc::new(AtomicBool::new(false));
    let stdout_reader = spawn_bounded_worker_output_reader(
        child
            .stdout
            .take()
            .with_context(|| format!("isolated {operation} stdout was unavailable"))?,
        Arc::clone(&output_bound_exceeded),
    );
    let stderr_reader = spawn_bounded_worker_output_reader(
        child
            .stderr
            .take()
            .with_context(|| format!("isolated {operation} stderr was unavailable"))?,
        Arc::clone(&output_bound_exceeded),
    );
    let stdin_writer = encoded_request
        .map(|request| {
            let stdin = child
                .stdin
                .take()
                .with_context(|| format!("isolated {operation} stdin was unavailable"))?;
            Ok::<_, anyhow::Error>(spawn_worker_input_writer(stdin, request))
        })
        .transpose()?;
    let deadline = std::time::Instant::now() + timeout;
    let (status, forced_outcome) = loop {
        if output_bound_exceeded.load(Ordering::Relaxed) {
            let _ = child.kill();
            let status = child
                .wait()
                .with_context(|| format!("failed to reap isolated {operation}"))?;
            break (status, Some(IsolatedWorkerOutcome::OutputBound));
        }
        if let Some(status) = child
            .try_wait()
            .with_context(|| format!("failed to observe isolated {operation}"))?
        {
            break (status, None);
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let status = child
                .wait()
                .with_context(|| format!("failed to reap isolated {operation}"))?;
            break (status, Some(IsolatedWorkerOutcome::Deadline));
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    let stdin_result = stdin_writer
        .map(|writer| join_worker_input_writer(writer, operation))
        .transpose();
    let stdout = join_worker_output_reader(stdout_reader, operation, "stdout")?;
    let stderr = join_worker_output_reader(stderr_reader, operation, "stderr")?;
    let forced_outcome = forced_outcome.or_else(|| {
        output_bound_exceeded
            .load(Ordering::Relaxed)
            .then_some(IsolatedWorkerOutcome::OutputBound)
    });
    if let Some(outcome) = forced_outcome {
        return Err(IsolatedWorkerFailure::new(format!(
            "isolated {operation} ended with {}",
            outcome.as_str()
        ))
        .into());
    }
    if !status.success() {
        return Err(IsolatedWorkerFailure::new(format!(
            "isolated {operation} exited with {}: {}",
            status,
            String::from_utf8_lossy(&stderr)
                .trim()
                .chars()
                .take(4_096)
                .collect::<String>()
        ))
        .into());
    }
    stdin_result?;
    serde_json::from_slice(&stdout).map_err(|error| {
        IsolatedWorkerFailure::new(format!(
            "isolated {operation} returned invalid JSON: {error}"
        ))
        .into()
    })
}

#[cfg(not(test))]
fn spawn_worker_input_writer(
    mut stdin: std::process::ChildStdin,
    request: Vec<u8>,
) -> std::thread::JoinHandle<std::io::Result<()>> {
    use std::io::Write as _;

    std::thread::spawn(move || {
        stdin.write_all(&request)?;
        stdin.flush()
    })
}

#[cfg(not(test))]
fn join_worker_input_writer(
    writer: std::thread::JoinHandle<std::io::Result<()>>,
    operation: &str,
) -> anyhow::Result<()> {
    writer
        .join()
        .map_err(|_| anyhow::anyhow!("isolated {operation} stdin writer panicked"))?
        .with_context(|| format!("failed to write isolated {operation} request"))
}

#[cfg(not(test))]
fn spawn_bounded_worker_output_reader<Reader>(
    reader: Reader,
    output_bound_exceeded: Arc<AtomicBool>,
) -> std::thread::JoinHandle<std::io::Result<Vec<u8>>>
where
    Reader: std::io::Read + Send + 'static,
{
    std::thread::spawn(move || {
        read_bounded_worker_output(
            reader,
            MAX_PLANNING_WORKER_OUTPUT_BYTES,
            &output_bound_exceeded,
        )
    })
}

fn read_bounded_worker_output<Reader>(
    mut reader: Reader,
    limit: usize,
    output_bound_exceeded: &AtomicBool,
) -> std::io::Result<Vec<u8>>
where
    Reader: std::io::Read,
{
    let mut retained = Vec::new();
    let mut chunk = [0_u8; 8 * 1024];
    loop {
        let read = reader.read(&mut chunk)?;
        if read == 0 {
            return Ok(retained);
        }
        let remaining = limit.saturating_sub(retained.len());
        retained.extend_from_slice(&chunk[..read.min(remaining)]);
        if read > remaining {
            output_bound_exceeded.store(true, Ordering::Relaxed);
        }
    }
}

#[cfg(not(test))]
fn join_worker_output_reader(
    reader: std::thread::JoinHandle<std::io::Result<Vec<u8>>>,
    operation: &str,
    stream: &str,
) -> anyhow::Result<Vec<u8>> {
    reader
        .join()
        .map_err(|_| anyhow::anyhow!("isolated {operation} {stream} reader panicked"))?
        .with_context(|| format!("failed to read isolated {operation} {stream}"))
}

fn run_template_worker(authority: NativeRuntimeAuthority) -> anyhow::Result<()> {
    let native_backend = initialize_native_runtime(&authority)?;
    let request = serde_json::from_reader(std::io::stdin().lock())
        .context("failed to decode native template request")?;
    let assessment = inspect_template_request_with_backend(request, &native_backend)?;
    serde_json::to_writer(std::io::stdout().lock(), &assessment)
        .context("failed to encode native template assessment")?;
    Ok(())
}

impl ResolvedModelAssessor for NativeResolvedModelAssessor {
    fn execution_cache_key(
        &self,
        profile: Option<&ModelPreviewProfile>,
        snapshot: &HardwareSnapshot,
    ) -> Result<String, InventoryError> {
        self.execution_assessment_cache_key(profile, snapshot)
    }

    fn assess_profile(
        &self,
        model: ResolvedModel,
        profile: Option<ModelPreviewProfile>,
    ) -> BoxFuture<'_, Result<HardwareAssessment, InventoryError>> {
        Box::pin(async move { self.assess_resolved(model, profile.as_ref()).await })
    }

    fn assess_profiles(
        &self,
        model: ResolvedModel,
        profiles: Vec<ModelPreviewProfile>,
    ) -> BoxFuture<'_, Result<Vec<HardwareAssessment>, InventoryError>> {
        Box::pin(async move { self.assess_resolved_profiles(model, profiles).await })
    }

    fn assess_execution_profiles(
        &self,
        model: ResolvedModel,
        profiles: Vec<ModelPreviewProfile>,
    ) -> BoxFuture<'_, Result<Vec<ModelExecutionAssessment>, InventoryError>> {
        Box::pin(async move {
            self.assess_resolved_execution_profiles(model, profiles)
                .await
        })
    }
}

impl HardwareProvider for NativeResolvedModelAssessor {
    fn snapshot(&self) -> BoxFuture<'_, Result<HardwareSnapshot, InventoryError>> {
        Box::pin(async move {
            let _guard = self.gate.lock().await;
            let native_executor = self
                .native_executor
                .read()
                .map_err(|_| InventoryError::Internal("native executor lock poisoned".to_owned()))?
                .as_ref()
                .and_then(Weak::upgrade);
            let native_build = build_identity::native_build();
            let enabled_backends = self.enabled_backends.as_ref().clone();
            let native_backend = self.native_backend.clone();
            let snapshot = spawn_blocking_traced(move || match native_executor {
                Some(resident) => {
                    let observation = resident
                        .observe_model_instance(
                            CapacityPolicy::default(),
                            native_build,
                            enabled_backends,
                        )
                        .map_err(|error| InventoryError::Internal(error.to_string()))?;
                    Ok(observation.hardware)
                }
                None => Ok(native_backend.discover_hardware(
                    CapacityPolicy::default(),
                    native_build,
                    enabled_backends,
                )),
            })
            .await
            .map_err(|error| InventoryError::Internal(error.to_string()))??;
            let observation =
                icn_hardware::observe_system_memory().map_err(InventoryError::Internal)?;
            Ok(icn_hardware::with_system_memory_observation(
                snapshot,
                observation,
            ))
        })
    }
}

#[derive(Clone)]
struct NativeModelInstanceController {
    inventory: Arc<ManagedModelStore>,
    assessor: Arc<NativeResolvedModelAssessor>,
    native_executor: Arc<RwLock<Option<Weak<RemoteBackend>>>>,
    worker_launcher: NativeWorkerLauncher,
    memory_observer: Arc<SystemMemoryObserver>,
    next_worker_generation: Arc<AtomicU64>,
    admission_blocked_until: Arc<Mutex<Option<std::time::Instant>>>,
    defaults: ModelPlanDefaults,
    load_progress: Arc<LoadProgressEstimator>,
    loaded_configurations: Arc<Mutex<std::collections::BTreeSet<String>>>,
    instances: InstanceEntries,
    mutation: Arc<tokio::sync::Mutex<()>>,
    residency_policy: Arc<RwLock<ModelResidencyPolicyState>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModelResidencyPolicyState {
    generation: u64,
    idle_timeout: std::time::Duration,
}

fn next_model_residency_policy(
    current: ModelResidencyPolicyState,
    generation: u64,
    idle_timeout: std::time::Duration,
) -> Result<ModelResidencyPolicyState, InventoryError> {
    if idle_timeout.is_zero() {
        return Err(InventoryError::InvalidRequest(
            "model residency idle timeout must be greater than zero".to_owned(),
        ));
    }
    if generation < current.generation {
        return Err(InventoryError::InvalidRequest(format!(
            "model residency policy generation {generation} is older than {}",
            current.generation,
        )));
    }
    if generation == current.generation {
        return if idle_timeout == current.idle_timeout {
            Ok(current)
        } else {
            Err(InventoryError::InvalidRequest(format!(
                "model residency policy generation {generation} conflicts with the established timeout",
            )))
        };
    }
    Ok(ModelResidencyPolicyState {
        generation,
        idle_timeout,
    })
}

#[derive(Clone)]
struct OwnedInferenceWorker {
    worker: InferenceWorker,
}

#[derive(Clone)]
enum ModelOperationFailure {
    Operation {
        code: String,
        message: String,
        retryable: bool,
    },
    LowMemory {
        message: String,
        required_system_memory_bytes: u64,
        allocation_headroom_bytes: u64,
        system_reserve_bytes: u64,
        load_boundary_bytes: u64,
        minimum_additional_available_bytes: u64,
        parallel_sequences: u32,
    },
}

impl ModelOperationFailure {
    fn new(code: impl Into<String>, message: impl Into<String>, retryable: bool) -> Self {
        Self::Operation {
            code: code.into(),
            message: message.into(),
            retryable,
        }
    }

    fn low_memory(
        sample: memory_supervisor::MemorySample,
        required_system_memory_bytes: u64,
        parallel_sequences: u32,
    ) -> Self {
        let system_reserve_bytes = sample.abort_reserve_bytes();
        let load_boundary_bytes = required_system_memory_bytes.saturating_add(system_reserve_bytes);
        let allocation_headroom_bytes = sample.allocation_headroom_bytes;
        let minimum_additional_available_bytes = load_boundary_bytes
            .saturating_sub(allocation_headroom_bytes)
            .saturating_add(1);
        let shortfall_unit = if minimum_additional_available_bytes == 1 {
            "byte"
        } else {
            "bytes"
        };
        let message = format!(
            "not enough memory available: model requires {required_system_memory_bytes} bytes plus {system_reserve_bytes} bytes reserved for the system; {allocation_headroom_bytes} bytes are available ({minimum_additional_available_bytes} {shortfall_unit} short)"
        );
        Self::LowMemory {
            message,
            required_system_memory_bytes,
            allocation_headroom_bytes,
            system_reserve_bytes,
            load_boundary_bytes,
            minimum_additional_available_bytes,
            parallel_sequences,
        }
    }

    fn code(&self) -> &str {
        match self {
            Self::Operation { code, .. } => code,
            Self::LowMemory { .. } => LOW_MEMORY_FAILURE_CODE,
        }
    }

    fn message(&self) -> &str {
        match self {
            Self::Operation { message, .. } | Self::LowMemory { message, .. } => message,
        }
    }

    fn retryable(&self) -> bool {
        match self {
            Self::Operation { retryable, .. } => *retryable,
            Self::LowMemory { .. } => true,
        }
    }

    fn into_instance_failure(self) -> ModelInstanceFailure {
        match self {
            Self::Operation {
                code,
                message,
                retryable,
            } => ModelInstanceFailure::Operation {
                code,
                message,
                retryable,
            },
            Self::LowMemory {
                message,
                required_system_memory_bytes,
                allocation_headroom_bytes,
                system_reserve_bytes,
                load_boundary_bytes,
                minimum_additional_available_bytes,
                parallel_sequences,
            } => ModelInstanceFailure::LowMemory {
                code: LOW_MEMORY_FAILURE_CODE.to_owned(),
                message,
                retryable: true,
                required_system_memory_bytes,
                allocation_headroom_bytes,
                system_reserve_bytes,
                load_boundary_bytes,
                minimum_additional_available_bytes,
                parallel_sequences,
            },
        }
    }
}

struct ModelTransitionFailure {
    event: ModelOperationFailure,
}

fn idle_release_elapsed(
    expected: &ReadyInstanceRecord,
    current: Option<&ReadyInstanceRecord>,
    activity: InstanceActivity,
    expected_policy_generation: u64,
    policy: ModelResidencyPolicyState,
    now: std::time::Instant,
) -> Option<std::time::Duration> {
    let current = current?;
    if current.generation != expected.generation
        || current.instance_id != expected.instance_id
        || activity.generation != expected.generation
        || activity.active_leases != 0
        || policy.generation != expected_policy_generation
    {
        return None;
    }
    let elapsed = now.checked_duration_since(activity.idle_since?)?;
    (elapsed >= policy.idle_timeout).then_some(elapsed)
}

fn restart_idle_interval(
    activity: &mut InstanceActivity,
    expected_generation: u64,
    now: std::time::Instant,
) {
    if activity.generation == expected_generation && activity.active_leases == 0 {
        activity.idle_since = Some(now);
    }
}

impl ModelTransitionFailure {
    fn new(event: ModelOperationFailure) -> Self {
        Self { event }
    }

    fn stopped() -> Self {
        Self::new(ModelOperationFailure::new(
            "model_instance_stopped",
            "model instance was stopped",
            false,
        ))
    }
}

impl From<InventoryError> for ModelTransitionFailure {
    fn from(error: InventoryError) -> Self {
        match error {
            InventoryError::ModelOperation {
                code,
                message,
                retryable,
            } => Self::new(ModelOperationFailure::new(code, message, retryable)),
            error => Self::new(ModelOperationFailure::new(
                "model_transition_failed",
                error.to_string(),
                true,
            )),
        }
    }
}

impl NativeModelInstanceController {
    const DISCONNECTED_IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10 * 60);

    fn new(
        inventory: Arc<ManagedModelStore>,
        assessor: Arc<NativeResolvedModelAssessor>,
        native_executor: Arc<RwLock<Option<Weak<RemoteBackend>>>>,
        worker_launcher: NativeWorkerLauncher,
        defaults: ModelPlanDefaults,
        cache: ModelCache,
        native_build: String,
    ) -> Self {
        Self {
            inventory,
            assessor,
            native_executor,
            worker_launcher,
            memory_observer: Arc::new(SystemMemoryObserver::new()),
            next_worker_generation: Arc::new(AtomicU64::new(1)),
            admission_blocked_until: Arc::new(Mutex::new(None)),
            defaults,
            load_progress: Arc::new(LoadProgressEstimator::new(cache, native_build)),
            loaded_configurations: Arc::new(Mutex::new(std::collections::BTreeSet::new())),
            instances: InstanceEntries::new(),
            mutation: Arc::new(tokio::sync::Mutex::new(())),
            residency_policy: Arc::new(RwLock::new(ModelResidencyPolicyState {
                generation: 0,
                idle_timeout: Self::DISCONNECTED_IDLE_TIMEOUT,
            })),
        }
    }

    async fn publish_loading(
        &self,
        instance_id: &ModelInstanceId,
        configuration_id: &ModelServingConfigurationId,
        stage: ModelLoadStage,
        progress: Option<f32>,
        planned_allocation: Option<ModelLoadPlan>,
    ) {
        self.instances
            .publish(ModelInstance {
                id: instance_id.clone(),
                configuration_id: configuration_id.clone(),
                lifecycle: ModelInstanceLifecycle::Loading {
                    stage,
                    progress,
                    planned_allocation,
                },
            })
            .await;
    }

    async fn publish_failed(
        &self,
        instance_id: &ModelInstanceId,
        configuration_id: &ModelServingConfigurationId,
        failure: ModelInstanceFailure,
    ) {
        self.instances
            .publish(ModelInstance {
                id: instance_id.clone(),
                configuration_id: configuration_id.clone(),
                lifecycle: ModelInstanceLifecycle::Failed { failure },
            })
            .await;
    }

    async fn publish_stopped_loading(
        &self,
        instance_id: &ModelInstanceId,
        configuration_id: &ModelServingConfigurationId,
    ) {
        self.instances
            .publish(ModelInstance {
                id: instance_id.clone(),
                configuration_id: configuration_id.clone(),
                lifecycle: ModelInstanceLifecycle::Stopped {
                    reason: ModelReleaseReason::UserStop,
                },
            })
            .await;
    }

    async fn replay_load_events(
        &self,
        instance_id: &ModelInstanceId,
        events: &tokio::sync::mpsc::UnboundedSender<ModelLoadEvent>,
    ) {
        let mut changes = self.instances.subscribe();
        loop {
            let Some(instance) = self.instances.instance(instance_id).await else {
                return;
            };
            match instance.lifecycle {
                ModelInstanceLifecycle::Loading {
                    stage,
                    progress,
                    planned_allocation,
                } => {
                    let _ = events.send(ModelLoadEvent::Progress {
                        stage,
                        fraction: progress,
                        plan: planned_allocation,
                    });
                }
                ModelInstanceLifecycle::Ready { allocation } => {
                    let _ = events.send(ModelLoadEvent::Ready {
                        ready: LoadModelReady {
                            instance_id: instance.id,
                            configuration_id: instance.configuration_id,
                            allocation,
                        },
                    });
                    return;
                }
                ModelInstanceLifecycle::Stopping { .. } => {}
                ModelInstanceLifecycle::Stopped { .. } => {
                    let _ = events.send(ModelLoadEvent::Stopped {
                        instance_id: instance.id,
                    });
                    return;
                }
                ModelInstanceLifecycle::Failed { failure } => {
                    let _ = events.send(ModelLoadEvent::Failed { failure });
                    return;
                }
            }
            match changes.recv().await {
                Ok(_) | Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                Err(tokio::sync::broadcast::error::RecvError::Closed) => return,
            }
        }
    }

    fn start_idle_supervisor(&self, ready_instance: ReadyInstanceRecord) {
        let controller = self.clone();
        tokio::spawn(async move {
            loop {
                let activity_changed = ready_instance.runtime.activity_changed();
                tokio::pin!(activity_changed);
                let activity = ready_instance.runtime.activity();
                if activity.generation != ready_instance.generation {
                    return;
                }
                let Some(idle_since) = activity.idle_since.filter(|_| activity.active_leases == 0)
                else {
                    activity_changed.await;
                    continue;
                };
                let policy = *controller
                    .residency_policy
                    .read()
                    .expect("model residency policy lock poisoned");
                let deadline = tokio::time::Instant::from_std(idle_since + policy.idle_timeout);
                tokio::select! {
                    _ = tokio::time::sleep_until(deadline) => {}
                    _ = &mut activity_changed => continue,
                }

                let _operation = controller.mutation.lock().await;
                let Some(backend_mutation) = ready_instance.runtime.try_begin_mutation() else {
                    continue;
                };
                let current = controller.instances.ready_instance().await;
                let activity = ready_instance.runtime.activity();
                let current_policy = *controller
                    .residency_policy
                    .read()
                    .expect("model residency policy lock poisoned");
                let Some(elapsed) = idle_release_elapsed(
                    &ready_instance,
                    current.as_ref(),
                    activity,
                    policy.generation,
                    current_policy,
                    std::time::Instant::now(),
                ) else {
                    drop(backend_mutation);
                    continue;
                };
                tracing::info!(
                    model.configuration.id = %ready_instance.configuration_id.0,
                    model.instance.id = %ready_instance.instance_id.0,
                    generation = ready_instance.generation,
                    idle_seconds = elapsed.as_secs_f64(),
                    "model idle deadline won admission race"
                );
                let _ = controller
                    .stop_ready_instance_under_mutation(
                        &ready_instance,
                        ModelReleaseReason::IdleTimeout,
                        backend_mutation,
                    )
                    .await;
                return;
            }
        });
    }

    fn profile_defaults(
        &self,
        profile: &ModelExecutionProfile,
    ) -> Result<ModelPlanDefaults, InventoryError> {
        let mut defaults = self.defaults.clone();
        defaults.context_size = profile.context_length;
        defaults.physical_context_size = profile.context_length;
        defaults.max_sequences = 1;
        defaults.execution.kv_unified = false;
        Ok(defaults)
    }

    async fn resolved_configuration_load(
        &self,
        configuration: &ModelServingConfiguration,
    ) -> Result<
        (
            ResolvedModel,
            ExecutionIntent,
            SpeculativeDecodingConfig,
            Vec<ModelPackageId>,
        ),
        InventoryError,
    > {
        if !serving_configuration_identity_is_valid(configuration) {
            return Err(InventoryError::InvalidRequest(format!(
                "configuration {} does not identify its exact bundle and profile",
                configuration.id.0
            )));
        }
        let (target, package_ids) = match &configuration.bundle {
            DomainServableModelBundle::Standalone { package } => (
                ModelBundleInput::Standalone {
                    package: ModelPackageOperand::Installed {
                        package_id: package.id.clone(),
                    },
                },
                vec![package.id.clone()],
            ),
            DomainServableModelBundle::SpeculativeDecoding {
                target,
                draft_source,
                method,
            } => (
                ModelBundleInput::SpeculativeDecoding {
                    target: ModelPackageOperand::Installed {
                        package_id: target.id.clone(),
                    },
                    draft_source: match draft_source {
                        ModelSpeculativeDraftSource::Embedded => {
                            SpeculativeDraftSourceInput::Embedded
                        }
                        ModelSpeculativeDraftSource::Separate { draft } => {
                            SpeculativeDraftSourceInput::Separate {
                                draft: ModelPackageOperand::Installed {
                                    package_id: draft.id.clone(),
                                },
                            }
                        }
                    },
                    method: method.clone(),
                },
                match draft_source {
                    ModelSpeculativeDraftSource::Embedded => vec![target.id.clone()],
                    ModelSpeculativeDraftSource::Separate { draft } => {
                        vec![target.id.clone(), draft.id.clone()]
                    }
                },
            ),
        };
        let resolved = self.inventory.resolve_bundle(target).await?;
        let speculative = NativeModelAssessor::speculative_config_for_resolved(&resolved)?;
        let mut model = resolved.target_model;
        if let Some(draft) = resolved.draft_model {
            model
                .components
                .extend(draft.components.into_iter().map(|mut component| {
                    component.role = ComponentRole::Draft;
                    component
                }));
        }
        let primary = model
            .components
            .iter()
            .filter(|component| {
                matches!(
                    component.role,
                    ComponentRole::Weights | ComponentRole::Shard
                )
            })
            .min_by_key(|component| component.shard_index.unwrap_or(0))
            .map(|component| component.path.clone())
            .ok_or_else(|| InventoryError::NotReady("model has no runnable weights".into()))?;
        let projector = model
            .components
            .iter()
            .find(|component| component.role == ComponentRole::Projector)
            .map(|component| component.path.clone());
        let defaults = self.profile_defaults(&ModelExecutionProfile {
            context_length: configuration.profile.context_length,
        })?;
        let plan = execution_intent(primary, projector, &defaults);
        Ok((model, plan, speculative, package_ids))
    }

    async fn assess_load_candidates(
        &self,
        resolved: ResolvedModel,
        profile: &ModelExecutionProfile,
        configuration_id: &ModelServingConfigurationId,
    ) -> Result<(Vec<(u32, u64)>, u64, HardwareSnapshot), ModelTransitionFailure> {
        const MAX_DYNAMIC_PARALLEL_SEQUENCES: u32 = 4;
        let hardware = HardwareProvider::snapshot(self.assessor.as_ref())
            .await
            .map_err(ModelTransitionFailure::from)?;
        let assess_reserve =
            icn_hardware::system_memory_thresholds(hardware.system_memory.physical_capacity_bytes)
                .assess_reserve_bytes;
        let resident = self.instances.ready_instance().await;
        let releasable_system_memory_bytes = resident
            .as_ref()
            .map(|resident| {
                resident
                    .allocation
                    .memory_domains
                    .iter()
                    .filter(|domain| domain.memory_domain_id.is_system())
                    .map(|domain| {
                        domain
                            .model_bytes
                            .saturating_add(domain.context_bytes)
                            .saturating_add(domain.compute_bytes)
                            .saturating_add(domain.auxiliary_bytes)
                    })
                    .fold(0_u64, u64::saturating_add)
            })
            .unwrap_or_default();
        let maximum = if resolved
            .components
            .iter()
            .any(|component| component.role == ComponentRole::Projector)
        {
            1
        } else {
            MAX_DYNAMIC_PARALLEL_SEQUENCES
        };
        let profiles = (1..=maximum)
            .map(|parallel_sequences| ModelPreviewProfile {
                id: format!("load-allocation-{parallel_sequences}"),
                context_length: profile.context_length,
                parallel_sequences,
                performance_context_tokens: Vec::new(),
            })
            .collect::<Vec<_>>();
        let capacity_policy = CapacityPolicy {
            reserve_bytes_per_domain: CapacityPolicy::default().reserve_bytes_per_domain,
            system_reserve_bytes: Some(assess_reserve),
        };
        let hardware = icn_hardware::with_capacity_policy(hardware, capacity_policy);
        let assessments = self
            .assessor
            .assess_resolved_plans_cached(resolved, profiles, &hardware, configuration_id)
            .await
            .map_err(ModelTransitionFailure::from)?;
        let mut candidates = Vec::new();
        for (index, assessment) in assessments.into_iter().enumerate() {
            let parallel_sequences = u32::try_from(index + 1).expect("four candidates fit u32");
            match assessment {
                HardwareAssessment::Fits { memory, .. } => {
                    let required = memory
                        .domains
                        .iter()
                        .find(|domain| domain.memory_domain.is_system())
                        .map(|domain| domain.required_bytes)
                        .ok_or_else(|| {
                            ModelTransitionFailure::new(ModelOperationFailure::new(
                                "memory_assessment_incomplete",
                                "native planner omitted the system-memory domain",
                                false,
                            ))
                        })?;
                    candidates.push((parallel_sequences, required));
                }
                HardwareAssessment::DoesNotFit { .. } if parallel_sequences > 1 => break,
                HardwareAssessment::DoesNotFit {
                    limiting_resource,
                    memory,
                    ..
                } => {
                    return Err(ModelTransitionFailure::new(ModelOperationFailure::new(
                        "insufficient_resources",
                        format!(
                            "native baseline does not fit {limiting_resource}: {} byte deficit",
                            memory.deficit_bytes
                        ),
                        false,
                    )));
                }
                HardwareAssessment::InvalidArtifact { code, message }
                | HardwareAssessment::IncompatibleArtifact { code, message } => {
                    tracing::error!(
                        model.configuration.id = %configuration_id.0,
                        error.code = %code,
                        error.retryable = false,
                        "model load planning rejected the selected artifact"
                    );
                    return Err(ModelTransitionFailure::new(ModelOperationFailure::new(
                        code, message, false,
                    )));
                }
                HardwareAssessment::NotAssessed { reason } if parallel_sequences == 1 => {
                    return Err(ModelTransitionFailure::new(ModelOperationFailure::new(
                        "memory_estimate_failed",
                        reason,
                        true,
                    )));
                }
                HardwareAssessment::NotAssessed { .. } => break,
            }
        }
        Ok((candidates, releasable_system_memory_bytes, hardware))
    }

    async fn cleanup_owned_worker_under_mutation(
        &self,
        _model_mutation: &tokio::sync::MutexGuard<'_, ()>,
        instance_id: &ModelInstanceId,
        worker: &InferenceWorker,
        code: &str,
        reason: &str,
    ) {
        let pid = worker.pid();
        let is_current = self.instances.owns_worker(instance_id, pid).await;
        let resident = self
            .instances
            .ready_instance()
            .await
            .filter(|resident| resident.instance_id == *instance_id);
        if !is_current {
            return;
        }
        worker.terminate(code, reason);
        if let Some(resident) = resident.as_ref() {
            resident.runtime.clear();
        }
        if let Ok(mut slot) = self.native_executor.write() {
            *slot = None;
        }
        {
            self.instances.clear_ready(instance_id).await;
            self.instances.take_worker(instance_id).await;
        }
        if let Some(resident) = resident {
            self.instances
                .publish(ModelInstance {
                    id: resident.instance_id,
                    configuration_id: resident.configuration_id,
                    lifecycle: ModelInstanceLifecycle::Failed {
                        failure: ModelInstanceFailure::Operation {
                            code: code.to_owned(),
                            message: reason.to_owned(),
                            retryable: true,
                        },
                    },
                })
                .await;
        }
    }

    async fn cleanup_owned_worker(
        &self,
        instance_id: &ModelInstanceId,
        worker: &InferenceWorker,
        code: &str,
        reason: &str,
    ) {
        let model_mutation = self.mutation.lock().await;
        self.cleanup_owned_worker_under_mutation(
            &model_mutation,
            instance_id,
            worker,
            code,
            reason,
        )
        .await;
    }

    async fn release_owned_ready_instance(
        &self,
        instance_id: &ModelInstanceId,
        worker: &InferenceWorker,
        reason: ModelReleaseReason,
    ) -> Result<bool, InventoryError> {
        let _model_mutation = self.mutation.lock().await;
        if !self.instances.owns_worker(instance_id, worker.pid()).await {
            return Ok(false);
        }
        let Some(resident) = self
            .instances
            .ready_instance()
            .await
            .filter(|resident| resident.instance_id == *instance_id)
        else {
            return Ok(false);
        };
        let backend_mutation = resident.runtime.begin_mutation().await;
        self.stop_ready_instance_under_mutation(&resident, reason, backend_mutation)
            .await
    }

    fn block_memory_admission(&self) {
        if let Ok(mut blocked_until) = self.admission_blocked_until.lock() {
            *blocked_until = Some(std::time::Instant::now() + RECOVERY_STABLE_TIME);
        }
    }

    fn start_idle_memory_observer(&self) {
        let controller = self.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(IDLE_POLL_INTERVAL);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tick.tick().await;
                let blocked = controller
                    .admission_blocked_until
                    .lock()
                    .ok()
                    .and_then(|blocked_until| *blocked_until);
                let Some(blocked_until) = blocked else {
                    // Still sample while idle so observer failures are exercised before admission.
                    let _ = controller.memory_observer.sample();
                    continue;
                };
                match controller.memory_observer.sample() {
                    Ok(sample) if sample.recovered() => {
                        if std::time::Instant::now() >= blocked_until
                            && let Ok(mut state) = controller.admission_blocked_until.lock()
                        {
                            *state = None;
                        }
                    }
                    Ok(_) | Err(_) => controller.block_memory_admission(),
                }
            }
        });
    }

    fn supervise_worker(&self, instance_id: ModelInstanceId, worker: InferenceWorker) {
        let controller = self.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(POLL_INTERVAL);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            let mut observation_failed_at = None;
            loop {
                tick.tick().await;
                match worker.try_wait() {
                    Ok(Some(status)) => {
                        controller
                            .cleanup_owned_worker(
                                &instance_id,
                                &worker,
                                "worker_exited",
                                &format!("inference worker exited unexpectedly: {status}"),
                            )
                            .await;
                        break;
                    }
                    Err(error) => {
                        controller
                            .cleanup_owned_worker(
                                &instance_id,
                                &worker,
                                "worker_monitor_failed",
                                &format!("failed to observe inference worker: {error}"),
                            )
                            .await;
                        break;
                    }
                    Ok(None) => {}
                }
                match controller.memory_observer.sample() {
                    Ok(sample) => {
                        observation_failed_at = None;
                        if sample.requires_eviction() {
                            let worker_resident_bytes = worker.pid().and_then(|pid| {
                                controller.memory_observer.worker_resident_bytes(pid)
                            });
                            controller.block_memory_admission();
                            tracing::warn!(
                                memory.physical_available_bytes = sample.physical_available_bytes,
                                memory.allocation_headroom_bytes = sample.allocation_headroom_bytes,
                                memory.reserve_bytes = sample.abort_reserve_bytes(),
                                memory.sample_age_ms = sample.captured_at.elapsed().as_millis(),
                                worker.resident_bytes = ?worker_resident_bytes,
                                worker.pid = worker.pid(),
                                "evicting inference worker under system memory pressure"
                            );
                            match controller
                                .release_owned_ready_instance(
                                    &instance_id,
                                    &worker,
                                    ModelReleaseReason::MemoryPressure,
                                )
                                .await
                            {
                                Ok(true) => {}
                                Ok(false) => {
                                    controller
                                        .cleanup_owned_worker(
                                            &instance_id,
                                            &worker,
                                            LOW_MEMORY_FAILURE_CODE,
                                            "inference worker evicted under system memory pressure",
                                        )
                                        .await;
                                }
                                Err(error) => {
                                    tracing::warn!(
                                        model.instance.id = %instance_id.0,
                                        error = %error,
                                        "memory-pressure model release failed"
                                    );
                                    controller
                                        .cleanup_owned_worker(
                                            &instance_id,
                                            &worker,
                                            LOW_MEMORY_FAILURE_CODE,
                                            "inference worker evicted under system memory pressure",
                                        )
                                        .await;
                                }
                            }
                            break;
                        }
                    }
                    Err(error) => {
                        let failed_at =
                            observation_failed_at.get_or_insert_with(std::time::Instant::now);
                        if failed_at.elapsed() >= MONITOR_LOSS_DEADLINE {
                            controller.block_memory_admission();
                            controller
                                .cleanup_owned_worker(
                                    &instance_id,
                                    &worker,
                                    "memory_monitor_unavailable",
                                    &format!("system memory supervision unavailable: {error}"),
                                )
                                .await;
                            break;
                        }
                    }
                }
            }
        });
    }

    async fn select_load_allocation(
        &self,
        resolved: ResolvedModel,
        profile: &ModelExecutionProfile,
        configuration_id: &ModelServingConfigurationId,
    ) -> Result<(u32, u64, HardwareSnapshot), ModelTransitionFailure> {
        let recovery_blocked = self
            .admission_blocked_until
            .lock()
            .ok()
            .and_then(|blocked_until| *blocked_until)
            .is_some_and(|blocked_until| blocked_until > std::time::Instant::now());
        if recovery_blocked {
            return Err(ModelTransitionFailure::new(ModelOperationFailure::new(
                LOW_MEMORY_FAILURE_CODE,
                "system memory is still in the post-eviction recovery period",
                true,
            )));
        }
        let (candidates, releasable_system_memory_bytes, hardware) = self
            .assess_load_candidates(resolved, profile, configuration_id)
            .await?;
        // Selection uses a fresh observation immediately after planning. Both preview and load
        // call this exact function; neither reconstructs parallelism from persisted assessments.
        let sample = self.memory_observer.sample().map_err(|error| {
            ModelTransitionFailure::new(ModelOperationFailure::new(
                "memory_monitor_unavailable",
                error,
                true,
            ))
        })?;
        let sample = credit_replaced_instance_memory(sample, releasable_system_memory_bytes);
        if self
            .admission_blocked_until
            .lock()
            .ok()
            .and_then(|blocked_until| *blocked_until)
            .is_some()
            && !sample.recovered()
        {
            return Err(ModelTransitionFailure::new(ModelOperationFailure::new(
                LOW_MEMORY_FAILURE_CODE,
                format!(
                    "system memory has not recovered above the {} byte reserve and hysteresis margin",
                    sample.abort_reserve_bytes()
                ),
                true,
            )));
        }
        if let Ok(mut blocked_until) = self.admission_blocked_until.lock() {
            *blocked_until = None;
        }
        let Some(selected) = select_model_allocation(&candidates, sample) else {
            let minimum_required = candidates
                .first()
                .map(|(_, required)| *required)
                .unwrap_or_default();
            return Err(ModelTransitionFailure::new(
                ModelOperationFailure::low_memory(sample, minimum_required, 1),
            ));
        };
        Ok((selected.0, selected.1, hardware))
    }

    #[tracing::instrument(
        name = "icn.model.load.operation",
        skip_all,
        fields(model.configuration.id = %configuration.id.0)
    )]
    async fn perform_prepared_transition(
        self,
        configuration: ModelServingConfiguration,
        resolved: ResolvedModel,
        mut plan: ExecutionIntent,
        speculative: SpeculativeDecodingConfig,
        package_ids: Vec<ModelPackageId>,
        events: tokio::sync::mpsc::UnboundedSender<ModelLoadEvent>,
        instance_id: ModelInstanceId,
        stop_requested: Arc<AtomicBool>,
        model_mutation: &tokio::sync::MutexGuard<'_, ()>,
    ) -> Result<icn_contracts::models::ModelInstanceAllocation, ModelTransitionFailure> {
        if stop_requested.load(Ordering::Acquire) {
            return Err(ModelTransitionFailure::stopped());
        }
        let configuration_id = configuration.id.clone();
        let model_id = configuration_id.0.clone();
        let profile = ModelExecutionProfile {
            context_length: configuration.profile.context_length,
        };
        let existing = self.instances.ready_instance().await;
        let _backend_mutation = match existing.as_ref() {
            Some(resident) => Some(resident.runtime.begin_mutation().await),
            None => None,
        };
        if stop_requested.load(Ordering::Acquire) {
            return Err(ModelTransitionFailure::stopped());
        }

        if existing.is_some() {
            let _ = events.send(ModelLoadEvent::Progress {
                stage: ModelLoadStage::Unloading,
                fraction: None,
                plan: None,
            });
        }
        if let Some(resident) = existing.as_ref() {
            self.instances
                .publish(ModelInstance {
                    id: resident.instance_id.clone(),
                    configuration_id: resident.configuration_id.clone(),
                    lifecycle: ModelInstanceLifecycle::Stopping {
                        reason: ModelReleaseReason::Replacement,
                        allocation: ModelStoppingAllocation::Resident {
                            allocation: resident.allocation.clone(),
                        },
                    },
                })
                .await;
        }
        if let Some(resident) = existing.as_ref() {
            self.stop_worker_gracefully(&resident.instance_id).await;
        }
        if let Some(resident) = existing.as_ref() {
            self.instances.clear_ready(&resident.instance_id).await;
            resident.runtime.clear();
        }
        if let Ok(mut slot) = self.native_executor.write() {
            *slot = None;
        }
        if let Some(resident) = existing {
            self.instances
                .publish(ModelInstance {
                    id: resident.instance_id,
                    configuration_id: resident.configuration_id,
                    lifecycle: ModelInstanceLifecycle::Stopped {
                        reason: ModelReleaseReason::Replacement,
                    },
                })
                .await;
        }

        let (parallel_sequences, required_system_memory_bytes, hardware) = self
            .select_load_allocation(resolved.clone(), &profile, &configuration_id)
            .await?;
        if stop_requested.load(Ordering::Acquire) {
            return Err(ModelTransitionFailure::stopped());
        }
        let physical_context_tokens = plan
            .context_size
            .checked_mul(parallel_sequences)
            .ok_or_else(|| {
                ModelTransitionFailure::new(ModelOperationFailure::new(
                    "invalid_model_allocation",
                    "context length multiplied by selected parallelism exceeds u32",
                    false,
                ))
            })?;
        plan.max_sequences = parallel_sequences;
        plan.physical_context_size = physical_context_tokens;
        plan.execution.kv_unified = false;
        tracing::info!(
            parallel_sequences,
            physical_context_tokens,
            required_system_memory_bytes,
            "selected model allocation"
        );

        let worker_generation = self.next_worker_generation.fetch_add(1, Ordering::Relaxed);
        let expected_build = build_identity::native_build();
        let worker_launcher = self.worker_launcher.clone();
        let (worker, mut load_events) = tokio::task::spawn_blocking(move || {
            InferenceWorker::spawn(worker_generation, expected_build, &worker_launcher)
        })
        .await
        .map_err(|error| {
            ModelTransitionFailure::new(ModelOperationFailure::new(
                "worker_spawn_failed",
                error.to_string(),
                true,
            ))
        })?
        .map_err(|error| {
            ModelTransitionFailure::new(ModelOperationFailure::new(
                "worker_spawn_failed",
                error.to_string(),
                true,
            ))
        })?;
        if stop_requested.load(Ordering::Acquire) {
            worker.shutdown();
            return Err(ModelTransitionFailure::stopped());
        }
        self.instances
            .install_worker(&instance_id, worker.clone())
            .await;
        self.supervise_worker(instance_id.clone(), worker.clone());
        if let Err(error) = worker.start_load(model_id.clone(), plan, speculative, hardware) {
            self.cleanup_owned_worker_under_mutation(
                model_mutation,
                &instance_id,
                &worker,
                "worker_protocol_error",
                "failed to send worker load command",
            )
            .await;
            return Err(ModelTransitionFailure::new(ModelOperationFailure::new(
                "worker_protocol_error",
                error.to_string(),
                true,
            )));
        }

        let previously_loaded_in_process = self
            .loaded_configurations
            .lock()
            .is_ok_and(|loaded| loaded.contains(&model_id));
        let prepared = loop {
            tokio::select! {
                event = load_events.recv() => break event,
                _ = tokio::time::sleep(std::time::Duration::from_millis(50)) => {
                    if stop_requested.load(Ordering::Acquire) {
                        self.cleanup_owned_worker_under_mutation(
                            model_mutation,
                            &instance_id,
                            &worker,
                            "model_instance_stopped",
                            "model instance was stopped",
                        ).await;
                        return Err(ModelTransitionFailure::stopped());
                    }
                }
            }
        };
        let (acceleration, signature, tracker) = match prepared {
            Some(LoadEvent::Prepared {
                acceleration,
                timing_plan_identity,
                phases,
            }) => {
                let signature = self.load_progress.signature(
                    &configuration,
                    &acceleration,
                    &timing_plan_identity,
                    &phases,
                    previously_loaded_in_process,
                );
                let estimates =
                    self.load_progress
                        .estimate(&signature, &configuration, &acceleration, &phases);
                (acceleration, signature, LoadProgressTracker::new(estimates))
            }
            Some(LoadEvent::Failed(message)) => {
                self.cleanup_owned_worker_under_mutation(
                    model_mutation,
                    &instance_id,
                    &worker,
                    "backend_load_failed",
                    &message,
                )
                .await;
                return Err(ModelTransitionFailure::new(ModelOperationFailure::new(
                    "backend_load_failed",
                    message,
                    true,
                )));
            }
            Some(LoadEvent::Lost { code, message }) => {
                self.cleanup_owned_worker_under_mutation(
                    model_mutation,
                    &instance_id,
                    &worker,
                    &code,
                    &message,
                )
                .await;
                return Err(ModelTransitionFailure::new(ModelOperationFailure::new(
                    code, message, true,
                )));
            }
            Some(LoadEvent::Phase { .. }) | Some(LoadEvent::Loaded(_)) => {
                self.cleanup_owned_worker_under_mutation(
                    model_mutation,
                    &instance_id,
                    &worker,
                    "worker_protocol_error",
                    "worker load protocol order violation",
                )
                .await;
                return Err(ModelTransitionFailure::new(ModelOperationFailure::new(
                    "worker_protocol_error",
                    "worker sent load activity before its prepared plan",
                    false,
                )));
            }
            None => {
                self.cleanup_owned_worker_under_mutation(
                    model_mutation,
                    &instance_id,
                    &worker,
                    "worker_exited",
                    "worker load channel closed",
                )
                .await;
                return Err(ModelTransitionFailure::new(ModelOperationFailure::new(
                    "worker_exited",
                    "inference worker stopped during load",
                    true,
                )));
            }
        };
        let _ = events.send(ModelLoadEvent::Progress {
            stage: ModelLoadStage::Loading,
            fraction: Some(0.0),
            plan: Some(ModelLoadPlan {
                context_window_tokens: profile.context_length,
                parallel_sequences,
                physical_context_tokens,
                required_system_memory_bytes,
            }),
        });
        self.publish_loading(
            &instance_id,
            &configuration_id,
            ModelLoadStage::Loading,
            Some(0.0),
            Some(ModelLoadPlan {
                context_window_tokens: profile.context_length,
                parallel_sequences,
                physical_context_tokens,
                required_system_memory_bytes,
            }),
        )
        .await;
        let mut progress_tick = tokio::time::interval(std::time::Duration::from_millis(100));
        progress_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let properties = loop {
            tokio::select! {
                event = load_events.recv() => match event {
                    Some(LoadEvent::Phase { phase, started }) => {
                        if started {
                            tracker.phase_started(phase);
                        } else {
                            tracker.phase_completed(phase);
                        }
                    }
                    Some(LoadEvent::Loaded(properties)) => break *properties,
                    Some(LoadEvent::Failed(message)) => {
                        self.cleanup_owned_worker_under_mutation(
                            model_mutation,
                            &instance_id,
                            &worker,
                            "backend_load_failed",
                            &message,
                        ).await;
                        return Err(ModelTransitionFailure::new(ModelOperationFailure::new(
                            "backend_load_failed",
                            message,
                            true,
                        )));
                    }
                    Some(LoadEvent::Lost { code, message }) => {
                        self.cleanup_owned_worker_under_mutation(
                            model_mutation,
                            &instance_id,
                            &worker,
                            &code,
                            &message,
                        ).await;
                        return Err(ModelTransitionFailure::new(ModelOperationFailure::new(
                            code,
                            message,
                            true,
                        )));
                    }
                    Some(LoadEvent::Prepared { .. }) => {
                        self.cleanup_owned_worker_under_mutation(
                            model_mutation,
                            &instance_id,
                            &worker,
                            "worker_protocol_error",
                            "worker sent duplicate prepared event",
                        ).await;
                        return Err(ModelTransitionFailure::new(ModelOperationFailure::new(
                            "worker_protocol_error",
                            "worker sent duplicate prepared event",
                            false,
                        )));
                    }
                    None => {
                        self.cleanup_owned_worker_under_mutation(
                            model_mutation,
                            &instance_id,
                            &worker,
                            "worker_exited",
                            "worker load channel closed",
                        ).await;
                        return Err(ModelTransitionFailure::new(ModelOperationFailure::new(
                            "worker_exited",
                            "inference worker stopped during load",
                            true,
                        )));
                    }
                },
                _ = progress_tick.tick() => {
                    if stop_requested.load(Ordering::Acquire) {
                        self.cleanup_owned_worker_under_mutation(
                            model_mutation,
                            &instance_id,
                            &worker,
                            "model_instance_stopped",
                            "model instance was stopped",
                        ).await;
                        return Err(ModelTransitionFailure::stopped());
                    }
                    let fraction = tracker.fraction();
                    let _ = events.send(ModelLoadEvent::Progress {
                        stage: ModelLoadStage::Loading,
                        fraction: Some(fraction),
                        plan: None,
                    });
                    self.publish_loading(
                        &instance_id,
                        &configuration_id,
                        ModelLoadStage::Loading,
                        Some(fraction),
                        Some(ModelLoadPlan {
                            context_window_tokens: profile.context_length,
                            parallel_sequences,
                            physical_context_tokens,
                            required_system_memory_bytes,
                        }),
                    )
                    .await;
                }
            }
        };
        let backend = Arc::new(worker.backend(model_id.clone(), properties));
        if stop_requested.load(Ordering::Acquire) {
            self.cleanup_owned_worker_under_mutation(
                model_mutation,
                &instance_id,
                &worker,
                "model_instance_stopped",
                "model instance was stopped",
            )
            .await;
            return Err(ModelTransitionFailure::stopped());
        }
        let _ = events.send(ModelLoadEvent::Progress {
            stage: ModelLoadStage::Verifying,
            fraction: Some(tracker.fraction()),
            plan: None,
        });
        self.publish_loading(
            &instance_id,
            &configuration_id,
            ModelLoadStage::Verifying,
            Some(tracker.fraction()),
            Some(ModelLoadPlan {
                context_window_tokens: profile.context_length,
                parallel_sequences,
                physical_context_tokens,
                required_system_memory_bytes,
            }),
        )
        .await;
        let observation_backend = Arc::clone(&backend);
        let enabled_backends = self.assessor.enabled_backends.as_ref().clone();
        let observation_result = spawn_blocking_traced(move || {
            observation_backend.observe_model_instance(
                CapacityPolicy::default(),
                build_identity::native_build(),
                enabled_backends,
            )
        })
        .await;
        let observation = match observation_result {
            Ok(Ok(observation)) => observation,
            Ok(Err(error)) => {
                self.cleanup_owned_worker_under_mutation(
                    model_mutation,
                    &instance_id,
                    &worker,
                    "model_instance_observation_failed",
                    &error,
                )
                .await;
                return Err(ModelTransitionFailure::new(ModelOperationFailure::new(
                    "model_instance_observation_failed",
                    error,
                    true,
                )));
            }
            Err(error) => {
                let message = error.to_string();
                self.cleanup_owned_worker_under_mutation(
                    model_mutation,
                    &instance_id,
                    &worker,
                    "model_instance_observation_failed",
                    &message,
                )
                .await;
                return Err(ModelTransitionFailure::new(ModelOperationFailure::new(
                    "model_instance_observation_failed",
                    message,
                    true,
                )));
            }
        };
        let allocation = observation.allocation;
        let runtime = InstanceRuntime::empty();
        let generation = runtime.install(
            instance_id.clone(),
            configuration_id.clone(),
            Arc::clone(&backend) as Arc<dyn CompletionBackend>,
        );
        if let Ok(mut slot) = self.native_executor.write() {
            *slot = Some(Arc::downgrade(&backend));
        }
        let resident = ReadyInstanceRecord {
            configuration_id: configuration_id.clone(),
            instance_id: instance_id.clone(),
            generation,
            package_ids,
            allocation: allocation.clone(),
            runtime,
        };
        if !self.instances.publish_ready(resident.clone()).await {
            self.cleanup_owned_worker_under_mutation(
                model_mutation,
                &instance_id,
                &worker,
                "model_instance_stopped",
                "model instance was stopped",
            )
            .await;
            return Err(ModelTransitionFailure::stopped());
        }
        self.start_idle_supervisor(resident);
        tracker.phase_completed(icn_engine::ModelLoadPhase::Finalize);
        let _ = events.send(ModelLoadEvent::Progress {
            stage: ModelLoadStage::Verifying,
            fraction: Some(tracker.fraction()),
            plan: None,
        });
        if let Ok(mut loaded) = self.loaded_configurations.lock() {
            loaded.insert(model_id.clone());
        }
        self.load_progress
            .record_success(&signature, &acceleration, &tracker);
        tracing::info!("model ready");
        Ok(allocation)
    }

    async fn stop_worker_gracefully(&self, instance_id: &ModelInstanceId) {
        let owned = self.instances.take_worker(instance_id).await;
        let Some(OwnedInferenceWorker { worker, .. }) = owned else {
            return;
        };
        worker.shutdown();
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            match worker.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if tokio::time::Instant::now() < deadline => {
                    tokio::time::sleep(std::time::Duration::from_millis(25)).await;
                }
                Ok(None) | Err(_) => {
                    worker.terminate(
                        "worker_unresponsive",
                        "inference worker graceful shutdown timed out",
                    );
                    break;
                }
            }
        }
    }

    async fn stop_ready_instance_under_mutation(
        &self,
        expected: &ReadyInstanceRecord,
        reason: ModelReleaseReason,
        _runtime_mutation: InstanceMutationGuard,
    ) -> Result<bool, InventoryError> {
        let current = self.instances.ready_instance().await;
        let Some(resident) = current.filter(|resident| {
            resident.generation == expected.generation
                && resident.instance_id == expected.instance_id
        }) else {
            return Ok(false);
        };

        self.instances
            .publish(ModelInstance {
                id: resident.instance_id.clone(),
                configuration_id: resident.configuration_id.clone(),
                lifecycle: ModelInstanceLifecycle::Stopping {
                    reason,
                    allocation: ModelStoppingAllocation::Resident {
                        allocation: resident.allocation.clone(),
                    },
                },
            })
            .await;
        resident.runtime.clear();
        if let Ok(mut slot) = self.native_executor.write() {
            *slot = None;
        }
        self.instances.clear_ready(&resident.instance_id).await;
        self.stop_worker_gracefully(&resident.instance_id).await;
        self.instances
            .publish(ModelInstance {
                id: resident.instance_id,
                configuration_id: resident.configuration_id,
                lifecycle: ModelInstanceLifecycle::Stopped { reason },
            })
            .await;
        Ok(true)
    }

    async fn stop_ready_instance(
        &self,
        ready_instance: &ReadyInstanceRecord,
        reason: ModelReleaseReason,
    ) -> Result<bool, InventoryError> {
        let backend_mutation = ready_instance.runtime.begin_mutation().await;
        self.stop_ready_instance_under_mutation(ready_instance, reason, backend_mutation)
            .await
    }
}

impl ModelInstanceController for NativeModelInstanceController {
    fn set_residency_policy(
        &self,
        generation: u64,
        idle_timeout: std::time::Duration,
    ) -> BoxFuture<'_, Result<(), InventoryError>> {
        Box::pin(async move {
            let _guard = self.mutation.lock().await;
            {
                let current = *self.residency_policy.read().map_err(|_| {
                    InventoryError::Internal("model residency policy lock poisoned".to_owned())
                })?;
                let next = next_model_residency_policy(current, generation, idle_timeout)?;
                if next == current {
                    return Ok(());
                }
                *self.residency_policy.write().map_err(|_| {
                    InventoryError::Internal("model residency policy lock poisoned".to_owned())
                })? = next;
            }

            if let Some(resident) = self.instances.ready_instance().await {
                if let Ok(mut activity) = resident.runtime.activity.lock() {
                    restart_idle_interval(
                        &mut activity,
                        resident.generation,
                        std::time::Instant::now(),
                    );
                }
                resident.runtime.activity_changed.notify_waiters();
            }
            Ok(())
        })
    }

    fn preview_load(
        &self,
        request: PreviewModelLoadRequest,
    ) -> BoxFuture<'_, Result<ModelLoadPlan, InventoryError>> {
        Box::pin(async move {
            let profile = ModelExecutionProfile {
                context_length: request.configuration.profile.context_length,
            };
            let (resolved, plan, _, _) = self
                .resolved_configuration_load(&request.configuration)
                .await?;
            let (parallel_sequences, required_system_memory_bytes, _) = self
                .select_load_allocation(resolved, &profile, &request.configuration.id)
                .await
                .map_err(|failure| InventoryError::ModelOperation {
                    code: failure.event.code().to_owned(),
                    message: failure.event.message().to_owned(),
                    retryable: failure.event.retryable(),
                })?;
            let physical_context_tokens = plan
                .context_size
                .checked_mul(parallel_sequences)
                .ok_or_else(|| {
                    InventoryError::InvalidRequest(
                        "context length multiplied by selected parallelism exceeds u32".to_owned(),
                    )
                })?;
            Ok(ModelLoadPlan {
                context_window_tokens: plan.context_size,
                parallel_sequences,
                physical_context_tokens,
                required_system_memory_bytes,
            })
        })
    }

    fn load_instance(&self, request: LoadModelRequest) -> BoxStream<'static, ModelLoadEvent> {
        let controller = self.clone();
        let (events, receiver) = tokio::sync::mpsc::unbounded_channel();
        let instance_id = request.instance_id.clone();
        let configuration_id = request.configuration.id.clone();
        tokio::spawn(async move {
            let (stop_requested, is_new) = match controller
                .instances
                .admit(instance_id.clone(), configuration_id.clone())
                .await
            {
                Ok(admission) => admission,
                Err(failure) => {
                    let _ = events.send(ModelLoadEvent::Failed {
                        failure: failure.into(),
                    });
                    return;
                }
            };
            if !is_new {
                controller.replay_load_events(&instance_id, &events).await;
                return;
            }
            let run = async {
                let send_stopped = || {
                    let _ = events.send(ModelLoadEvent::Stopped {
                        instance_id: instance_id.clone(),
                    });
                };
                let _ = events.send(ModelLoadEvent::Progress {
                    stage: ModelLoadStage::Queued,
                    fraction: None,
                    plan: None,
                });
                let model_mutation = controller.mutation.lock().await;
                if stop_requested.load(Ordering::Acquire) {
                    controller
                        .publish_stopped_loading(&instance_id, &configuration_id)
                        .await;
                    send_stopped();
                    return;
                }
                let configuration = request.configuration;
                let _ = events.send(ModelLoadEvent::Progress {
                    stage: ModelLoadStage::Resolving,
                    fraction: None,
                    plan: None,
                });
                controller
                    .publish_loading(
                        &instance_id,
                        &configuration_id,
                        ModelLoadStage::Resolving,
                        None,
                        None,
                    )
                    .await;
                let (resolved, plan, speculative, package_ids) = match controller
                    .resolved_configuration_load(&configuration)
                    .await
                {
                    Ok(resolved) => resolved,
                    Err(error) => {
                        if stop_requested.load(Ordering::Acquire) {
                            controller
                                .publish_stopped_loading(&instance_id, &configuration_id)
                                .await;
                            send_stopped();
                        } else {
                            let failure =
                                ModelInstanceFailure::from(inventory_model_failure(error));
                            controller
                                .publish_failed(&instance_id, &configuration_id, failure.clone())
                                .await;
                            let _ = events.send(ModelLoadEvent::Failed { failure });
                        }
                        return;
                    }
                };
                if stop_requested.load(Ordering::Acquire) {
                    controller
                        .publish_stopped_loading(&instance_id, &configuration_id)
                        .await;
                    send_stopped();
                    return;
                }
                let allocation = match controller
                    .clone()
                    .perform_prepared_transition(
                        configuration,
                        resolved,
                        plan,
                        speculative,
                        package_ids,
                        events.clone(),
                        instance_id.clone(),
                        Arc::clone(&stop_requested),
                        &model_mutation,
                    )
                    .await
                {
                    Ok(allocation) => allocation,
                    Err(failure) if failure.event.code() == "model_instance_stopped" => {
                        controller
                            .publish_stopped_loading(&instance_id, &configuration_id)
                            .await;
                        send_stopped();
                        return;
                    }
                    Err(failure) => {
                        let failure = failure.event.into_instance_failure();
                        controller
                            .publish_failed(&instance_id, &configuration_id, failure.clone())
                            .await;
                        let _ = events.send(ModelLoadEvent::Failed { failure });
                        return;
                    }
                };
                if stop_requested.load(Ordering::Acquire) {
                    if let Some(resident) = controller
                        .instances
                        .ready_instance()
                        .await
                        .filter(|resident| resident.instance_id == instance_id)
                    {
                        let _ = controller
                            .stop_ready_instance(&resident, ModelReleaseReason::UserStop)
                            .await;
                    }
                    send_stopped();
                    return;
                }
                let _ = events.send(ModelLoadEvent::Ready {
                    ready: LoadModelReady {
                        instance_id: instance_id.clone(),
                        configuration_id: configuration_id.clone(),
                        allocation,
                    },
                });
            };
            if std::panic::AssertUnwindSafe(run)
                .catch_unwind()
                .await
                .is_err()
            {
                if let Some(worker) = controller
                    .instances
                    .entry(&instance_id)
                    .await
                    .and_then(|entry| entry.worker)
                {
                    controller
                        .cleanup_owned_worker(
                            &instance_id,
                            &worker.worker,
                            "model_instance_operation_panicked",
                            "model instance operation panicked",
                        )
                        .await;
                }
                if !matches!(
                    controller.instances.instance(&instance_id).await,
                    Some(ModelInstance {
                        lifecycle: ModelInstanceLifecycle::Failed { .. },
                        ..
                    })
                ) {
                    controller
                        .publish_failed(
                            &instance_id,
                            &configuration_id,
                            ModelInstanceFailure::Operation {
                                code: "model_instance_operation_panicked".to_owned(),
                                message: "model instance operation panicked".to_owned(),
                                retryable: true,
                            },
                        )
                        .await;
                }
            }
        });
        UnboundedReceiverStream::new(receiver).boxed()
    }

    fn stop_instance(
        &self,
        instance_id: ModelInstanceId,
    ) -> BoxFuture<'_, Result<(), InventoryError>> {
        Box::pin(async move {
            let entry = self.instances.entry(&instance_id).await;
            if let Some(entry) = entry {
                entry.stop_requested.store(true, Ordering::Release);
                let loading = Some(entry.instance);
                if let Some(ModelInstance {
                    configuration_id,
                    lifecycle:
                        ModelInstanceLifecycle::Loading {
                            planned_allocation, ..
                        },
                    ..
                }) = loading
                {
                    self.instances
                        .publish(ModelInstance {
                            id: instance_id.clone(),
                            configuration_id,
                            lifecycle: ModelInstanceLifecycle::Stopping {
                                reason: ModelReleaseReason::UserStop,
                                allocation: ModelStoppingAllocation::Planned {
                                    allocation: planned_allocation,
                                },
                            },
                        })
                        .await;
                }
            }
            let _guard = self.mutation.lock().await;
            let resident = self
                .instances
                .ready_instance()
                .await
                .filter(|resident| resident.instance_id == instance_id);
            if let Some(resident) = resident {
                self.stop_ready_instance(&resident, ModelReleaseReason::UserStop)
                    .await?;
            }
            Ok(())
        })
    }

    fn instances(&self) -> BoxFuture<'_, ModelInstancesSnapshot> {
        Box::pin(async move { self.instances.snapshot().await })
    }

    fn watch_instances(&self) -> BoxStream<'static, ModelInstancesInvalidation> {
        let receiver = self.instances.subscribe();
        let instances = self.instances.clone();
        let changes = futures_util::stream::unfold(receiver, |mut receiver| async move {
            loop {
                match receiver.recv().await {
                    Ok(event) => return Some((event, receiver)),
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => return None,
                }
            }
        });
        Box::pin(
            futures_util::stream::once(async move {
                ModelInstancesInvalidation {
                    revision: instances.revision().await,
                }
            })
            .chain(changes),
        )
    }

    fn remove_installed(
        &self,
        package_id: ModelPackageId,
    ) -> BoxFuture<'_, Result<RemoveInstalledModelPackageResponse, InventoryError>> {
        Box::pin(async move {
            let _guard = self.mutation.lock().await;
            if self
                .instances
                .ready_instance()
                .await
                .is_some_and(|resident| resident.package_ids.contains(&package_id))
            {
                return Err(InventoryError::Loaded(package_id.0));
            }
            self.inventory.remove_installed(&package_id).await
        })
    }

    fn lease(
        &self,
        instance_id: ModelInstanceId,
        configuration_id: ModelServingConfigurationId,
    ) -> BoxFuture<'_, Result<ModelInstanceLease, InventoryError>> {
        Box::pin(async move {
            let _guard = self.mutation.lock().await;
            let resident = self.instances.ready_instance().await;
            let Some(resident) = resident.filter(|resident| {
                resident.instance_id == instance_id && resident.configuration_id == configuration_id
            }) else {
                return Err(InventoryError::NotReady(format!(
                    "configuration {} is not loaded",
                    configuration_id.0
                )));
            };
            resident
                .runtime
                .acquire(&instance_id, &configuration_id)
                .ok_or_else(|| {
                    InventoryError::NotReady(format!(
                        "configuration {} is not available for inference",
                        configuration_id.0
                    ))
                })
        })
    }
}

impl icn_contracts::models::CatalogPackageRemover for NativeModelInstanceController {
    fn remove_catalog_package(
        &self,
        package_id: ModelPackageId,
    ) -> BoxFuture<'_, Result<(), InventoryError>> {
        Box::pin(async move {
            <Self as ModelInstanceController>::remove_installed(self, package_id)
                .await
                .map(|_| ())
        })
    }
}

fn open_installation_catalog(
    installation: &installation::Installation,
) -> anyhow::Result<ReleaseCatalog> {
    if installation.native_build() != build_identity::native_build() {
        anyhow::bail!("ICN installation native build does not match its executable");
    }
    if installation.backend_module_abi() != build_identity::backend_module_abi() {
        anyhow::bail!("ICN installation backend module ABI does not match its executable");
    }
    load_release_catalog(&installation.planner_bundle())
        .context("failed to load the release planner bundle")
}

fn load_installation_backends(installation: &installation::Installation) -> anyhow::Result<()> {
    anyhow::ensure!(
        installation.native_build() == build_identity::native_build(),
        "ICN installation native build does not match its executable"
    );
    anyhow::ensure!(
        installation.backend_module_abi() == build_identity::backend_module_abi(),
        "ICN installation backend module ABI does not match its executable"
    );
    let declared = installation
        .executable()
        .canonicalize()
        .context("failed to resolve the declared ICN executable")?;
    let running = std::env::current_exe()?
        .canonicalize()
        .context("failed to resolve the running ICN executable")?;
    if declared != running {
        anyhow::bail!("running executable is not part of the declared ICN installation");
    }
    #[cfg(feature = "dynamic-backends")]
    {
        llama_cpp_2::llama_backend::load_backends_from_path(&installation.backend_directory());
        Ok(())
    }
    #[cfg(not(feature = "dynamic-backends"))]
    {
        let _ = installation;
        anyhow::bail!("ICN executable does not support dynamic backend modules")
    }
}

fn initialize_native_runtime(authority: &NativeRuntimeAuthority) -> anyhow::Result<NativeBackend> {
    if let Some(installation) = authority.installation() {
        if installation.backend() == IcnInstallationBackend::Cuda {
            let driver = cuda_driver::require()
                .context("failed to resolve the host CUDA driver provider")?;
            tracing::info!(
                cuda.driver.path = %driver.path.display(),
                cuda.driver.api = driver.driver_api,
                "resolved host CUDA driver provider"
            );
        }
        load_installation_backends(installation).with_context(|| {
            format!(
                "failed to load native runtime from {}",
                installation.declaration_path().display()
            )
        })?;
        // Prove that the declared modules registered before llama.cpp gets an
        // opportunity to search its executable, cwd, or compiled build path.
        validate_registered_backend(installation).with_context(|| {
            format!(
                "native runtime {} did not register the declared {} backend",
                installation.declaration_path().display(),
                installation.backend().name()
            )
        })?;
    }
    NativeBackend::initialize().context("failed to initialize the process native backend")
}

fn validate_registered_backend(installation: &installation::Installation) -> anyhow::Result<()> {
    use llama_cpp_2::{LlamaBackendDeviceType, list_llama_ggml_backend_devices};

    let devices = list_llama_ggml_backend_devices();
    if !devices
        .iter()
        .any(|device| device.device_type == LlamaBackendDeviceType::Cpu)
    {
        anyhow::bail!("ICN installation did not register a CPU backend");
    }
    if installation.backend() == IcnInstallationBackend::Cpu {
        if devices.iter().any(|device| {
            device.device_type != LlamaBackendDeviceType::Cpu
                && device.device_type != LlamaBackendDeviceType::Unknown
        }) {
            anyhow::bail!("CPU installation registered an accelerator backend");
        }
        return Ok(());
    }
    let required = installation.backend().name();
    if !devices.iter().any(|device| {
        (device.backend.eq_ignore_ascii_case(required)
            || (required == "metal" && device.backend.eq_ignore_ascii_case("mtl")))
            && device.device_type != LlamaBackendDeviceType::Cpu
            && device.device_type != LlamaBackendDeviceType::Unknown
    }) {
        anyhow::bail!("ICN installation did not register a usable {required} device");
    }
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    if matches!(
        &cli.command,
        Command::Serve {
            exit_on_stdin_eof: true,
            ..
        }
    ) {
        install_parent_stdin_guard();
    }
    let _telemetry = telemetry::init(matches!(&cli.command, Command::Serve { .. }))?;
    // Native planner diagnostics are extremely verbose and can dominate metadata-only assessment.
    // ICN emits bounded, structured operation telemetry at the service boundary instead.
    icn_engine::disable_native_diagnostics();
    match cli.command {
        Command::Serve {
            bind,
            instance_id,
            exit_on_stdin_eof: _,
            auth_token,
            fake,
            model_store,
            cache_root,
            hf_caches,
            installation,
        } => {
            let installation = installation
                .as_deref()
                .map(installation::Installation::load)
                .transpose()
                .context("invalid ICN installation")?;
            if installation.is_none() && !fake {
                anyhow::bail!(
                    "ICN installation is not prepared; run `bun icn:build` before development"
                );
            }
            let runtime_authority = installation
                .clone()
                .map(NativeRuntimeAuthority::installed)
                .unwrap_or_else(NativeRuntimeAuthority::development);
            let worker_launcher = NativeWorkerLauncher::new(runtime_authority.clone());
            let inventory_root = match model_store {
                Some(root) => root,
                None => InventoryConfig::default_root()
                    .context("failed to determine default model store")?,
            };
            let cache_root = match cache_root {
                Some(root) => root,
                None => InventoryConfig::default_cache_root()
                    .context("failed to determine default cache root")?,
            };
            let mut inventory_config = InventoryConfig::with_roots(inventory_root, cache_root)
                .context("invalid model inventory configuration")?;
            inventory_config.hf_cache_dirs.extend(hf_caches);
            let release_catalog = installation
                .as_ref()
                .map(open_installation_catalog)
                .transpose()?
                .map(Arc::new);
            if let Some(catalog) = &release_catalog {
                inventory_config.catalog_models = catalog.catalog().models.clone();
            }
            let plan_defaults = model_plan_defaults();
            let native_backend = initialize_native_runtime(&runtime_authority)?;
            let inventory = Arc::new(
                ManagedModelStore::open_with_template_assessor(
                    inventory_config,
                    Some(Arc::new(NativeTemplateAssessor {
                        worker_launcher: worker_launcher.clone(),
                    })),
                )
                .await
                .context("failed to initialize model inventory")?,
            );
            let enabled_backends = installation.as_ref().map_or_else(
                || {
                    build_identity::enabled_backends()
                        .into_iter()
                        .map(str::to_owned)
                        .collect::<Vec<_>>()
                },
                |installation| {
                    let selected = installation.backend().name();
                    let mut backends = vec!["cpu".to_owned()];
                    if selected != "cpu" {
                        backends.push(selected.to_owned());
                    }
                    backends
                },
            );
            let startup_hardware =
                discover_startup_hardware(native_backend.clone(), enabled_backends.clone())
                    .await
                    .context("failed to discover hardware during ICN startup")?;
            if let Some(installation) = &installation {
                let hardware_label = startup_hardware
                    .memory_domains
                    .iter()
                    .flat_map(|domain| &domain.devices)
                    .find(|device| {
                        normalized_backend_name(&device.backend) == installation.backend().name()
                    })
                    .map(|device| device.description.clone())
                    .or_else(|| startup_hardware.cpu_model.clone())
                    .unwrap_or_else(|| installation.backend().name().to_owned());
                let backend = match installation.backend() {
                    IcnInstallationBackend::Cpu => IcnStartupBackend::Cpu { hardware_label },
                    IcnInstallationBackend::Metal => IcnStartupBackend::Metal { hardware_label },
                    IcnInstallationBackend::Cuda => IcnStartupBackend::Cuda { hardware_label },
                    IcnInstallationBackend::Vulkan => IcnStartupBackend::Vulkan { hardware_label },
                };
                let progress = IcnStartupProgressRecord {
                    record_type: IcnStartupProgressRecordType::PreparingBackend,
                    backend,
                };
                println!(
                    "MAGNITUDE_ICN_PROGRESS {}",
                    serde_json::to_string(&progress)?
                );
                use std::io::Write as _;
                std::io::stdout()
                    .flush()
                    .context("failed to publish backend preparation progress")?;
            }
            let planning_workers = PersistentPlanningWorkerPool::start(
                worker_launcher.clone(),
                planning_worker_pool_size(),
            );
            let hardware_calibration = if fake {
                planning_workers
                    .initialize(Some(fixture_hardware_calibration()))
                    .await
                    .context("failed to initialize planning workers")?
            } else {
                establish_hardware_calibration(
                    inventory.derived_cache(),
                    &startup_hardware,
                    &planning_workers,
                )
                .await
                .context("hardware calibration failed during ICN startup")?
            };
            let (model_assessor, native_executor_slot) = native_assessor_services(
                &inventory,
                PlanningExecutor::Worker(planning_workers),
                native_backend.clone(),
                plan_defaults.clone(),
                hardware_calibration,
                enabled_backends,
            );
            let model_downloads = Arc::new(
                ManagedModelDownloads::open(inventory.clone())
                    .await
                    .context("failed to initialize model downloads")?,
            );
            let native_build = build_identity::native_build();
            let identity = ServerIdentity {
                instance_id: instance_id.clone(),
                api_version: 1,
                native_build: native_build.clone(),
            };
            let model_controller = (!fake).then(|| {
                let controller = Arc::new(NativeModelInstanceController::new(
                    inventory.clone(),
                    model_assessor.clone(),
                    native_executor_slot,
                    worker_launcher,
                    plan_defaults,
                    inventory.derived_cache().clone(),
                    native_build.clone(),
                ));
                controller.start_idle_memory_observer();
                controller
            });
            let mut state = if fake {
                AppState::new(FakeBackend::new("icn-fake", "Hello from ICN."))
            } else {
                AppState::model_free()
            }
            .with_installed_packages(inventory.clone())
            .with_hardware(model_assessor.clone())
            .with_model_downloads(model_downloads.clone())
            .with_identity(identity);
            if let Some(release_catalog) = release_catalog {
                if let Some(model_controller) = &model_controller {
                    state = state.with_catalog_models(ManagedCatalogModels::new(
                        inventory.clone(),
                        release_catalog.catalog().clone(),
                        model_downloads,
                        model_controller.clone(),
                    )?);
                }
                state = state
                    .with_model_assessor(Arc::new(NativeModelAssessor::new(
                        inventory,
                        model_assessor,
                        release_catalog.clone(),
                    )))
                    .with_recommendable_catalog(Arc::new(ReleaseRecommendableCatalog::new(
                        release_catalog.catalog().clone(),
                    )));
            }
            if let Some(model_controller) = model_controller {
                state = state.with_model_controller(model_controller);
            }
            if let Some(auth_token) = auth_token {
                state = state.with_authorization(auth_token);
            }
            let listener = tokio::net::TcpListener::bind(bind)
                .await
                .with_context(|| format!("failed to bind {bind}"))?;
            let address = listener
                .local_addr()
                .context("failed to read bound address")?;
            let origin = format!("http://{address}");
            let startup = IcnStartupRecord {
                record_type: IcnStartupRecordType::IcnReady,
                protocol_version: 1,
                origin,
                instance_id: instance_id.clone(),
                pid: std::process::id(),
                api_version: 1,
                native_build: native_build.clone(),
            };
            println!("MAGNITUDE_ICN_READY {}", serde_json::to_string(&startup)?);
            tracing::info!(
                service.name = telemetry::SERVICE_NAME,
                server.address = %address,
                "ICN server ready"
            );
            let app = app(state).layer(
                TraceLayer::new_for_http()
                    .make_span_with(telemetry::http_request_span)
                    .on_response(DefaultOnResponse::new().level(tracing::Level::INFO)),
            );
            let serve_result = axum::serve(listener, app)
                .with_graceful_shutdown(interrupt_signal())
                .await;
            serve_result?;
            tracing::info!("ICN server stopped");
        }
        Command::Doctor => println!("ICN inference engine and native backend loaded successfully"),
        Command::BackendEligibility { json } => {
            let report = backend_eligibility::probe();
            if json {
                println!("{}", serde_json::to_string(&report)?);
            } else {
                println!("{}", serde_json::to_string_pretty(&report)?);
            }
        }
        Command::Version { json } => {
            if json {
                println!("{}", serde_json::to_string(&build_identity::identity())?);
            } else {
                println!("{}", env!("CARGO_PKG_VERSION"));
            }
        }
        Command::PlanningWorker { runtime } => run_planning_worker(runtime.authority()?)?,
        Command::TemplateWorker { runtime } => run_template_worker(runtime.authority()?)?,
        Command::InferenceWorker { runtime } => {
            let authority = runtime.authority()?;
            let native_backend = initialize_native_runtime(&authority)?;
            inference_worker::run_worker(build_identity::native_build(), native_backend)?
        }
    }
    Ok(())
}

fn install_parent_stdin_guard() {
    // This thread starts before telemetry or native initialization. An ordinary
    // ACN shutdown signals ICN first; abrupt owner loss closes the private pipe
    // and must terminate ICN even if synchronous native initialization is busy.
    std::thread::spawn(move || {
        use std::io::Read as _;

        let mut stdin = std::io::stdin().lock();
        let mut buffer = [0_u8; 1];
        loop {
            match stdin.read(&mut buffer) {
                Ok(0) | Err(_) => std::process::exit(0),
                Ok(_) => {}
            }
        }
    });
}

#[cfg(unix)]
async fn interrupt_signal() {
    use tokio::signal::unix::{SignalKind, signal};
    let mut terminate = signal(SignalKind::terminate()).expect("SIGTERM handler must install");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {},
        _ = terminate.recv() => {},
    }
}

#[cfg(not(unix))]
async fn interrupt_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

fn spawn_blocking_traced<F, R>(operation: F) -> tokio::task::JoinHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let span = tracing::Span::current();
    tokio::task::spawn_blocking(move || span.in_scope(operation))
}

#[cfg(test)]
mod tests {
    use super::*;
    use icn_contracts::ModelInventory as _;

    #[test]
    fn model_transition_preserves_typed_model_operation_failure() {
        let failure = ModelTransitionFailure::from(InventoryError::ModelOperation {
            code: "invalid_split_layout".to_owned(),
            message: "the shard layout is invalid".to_owned(),
            retryable: false,
        });

        assert_eq!(failure.event.code(), "invalid_split_layout");
        assert_eq!(failure.event.message(), "the shard layout is invalid");
        assert!(!failure.event.retryable());
    }

    #[test]
    fn stable_artifact_rejection_is_scoped_to_one_assessment_target() {
        let failure = assessment_bundle_failure(InventoryError::ModelOperation {
            code: "invalid_split_layout".to_owned(),
            message: "the shard layout is invalid".to_owned(),
            retryable: false,
        })
        .expect("stable target rejection");
        assert_eq!(failure.code, "invalid_split_layout");
        assert!(!failure.retryable);

        let operational = assessment_bundle_failure(InventoryError::ModelOperation {
            code: "planning_deadline".to_owned(),
            message: "planning timed out".to_owned(),
            retryable: true,
        })
        .expect_err("operational failure remains distinct from invalid input");
        assert!(matches!(
            &operational,
            InventoryError::ModelOperation {
                retryable: true,
                ..
            }
        ));

        let result = failed_assessment_result(
            icn_contracts::models::ModelAssessmentRequestId("request-failed".to_owned()),
            operational,
        );
        assert!(matches!(
            result,
            AssessModelResult::Failed {
                request_id,
                failure: DomainModelFailure {
                    code,
                    retryable: true,
                    ..
                },
            } if request_id.0 == "request-failed" && code == "planning_deadline"
        ));
    }

    fn calibration_test_snapshot() -> HardwareSnapshot {
        HardwareSnapshot {
            captured_at: 1,
            platform: "test".to_owned(),
            architecture: "test".to_owned(),
            system_product_name: Some("test-system".to_owned()),
            cpu_model: Some("test-cpu".to_owned()),
            logical_cores: 1,
            system_memory: icn_contracts::HardwareSystemMemory {
                physical_capacity_bytes: 10,
                physical_available_bytes: 10,
                allocation_capacity_bytes: 10,
                allocation_headroom_bytes: 10,
                assess_reserve_bytes: 0,
                abort_reserve_bytes: 0,
            },
            native_build: build_identity::native_build(),
            enabled_backends: vec!["cpu".to_owned()],
            topology_fingerprint: "test-topology".to_owned(),
            memory_domains: vec![icn_contracts::HardwareMemoryDomain {
                id: icn_contracts::MemoryDomainId::system(),
                kind: icn_contracts::HardwareMemoryDomainKind::System,
                total_capacity_bytes: 10,
                stable_capacity_bytes: 10,
                current_free_bytes: Some(10),
                shares_system_memory: true,
                devices: Vec::new(),
            }],
        }
    }

    #[test]
    fn hardware_calibration_cache_requires_current_valid_evidence() {
        let temporary = tempfile::tempdir().unwrap();
        let cache = ModelCache::new(temporary.path());
        let snapshot = calibration_test_snapshot();
        let input_identity = hardware_calibration_input_identity(&snapshot).unwrap();
        let hardware_calibration = fixture_hardware_calibration();
        let measured_at_seconds = 1_000_000;
        cache.write_index(
            icn_models::ModelIndexKind::HardwareCalibration,
            &input_identity,
            &HardwareCalibrationRecord {
                input_identity: input_identity.clone(),
                measured_at_seconds,
                hardware_calibration: hardware_calibration.clone(),
                hardware_calibration_identity:
                    NativeResolvedModelAssessor::hardware_calibration_identity(
                        &hardware_calibration,
                    )
                    .unwrap(),
            },
        );

        assert_eq!(
            cached_hardware_calibration(&cache, &snapshot, measured_at_seconds).unwrap(),
            Some(hardware_calibration)
        );
        assert!(
            cached_hardware_calibration(
                &cache,
                &snapshot,
                measured_at_seconds + HARDWARE_CALIBRATION_MAX_AGE_SECONDS + 1,
            )
            .unwrap()
            .is_none()
        );
        assert!(
            cached_hardware_calibration(&cache, &snapshot, measured_at_seconds - 1)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn worker_output_is_drained_but_retained_only_to_its_bound() {
        let output_bound_exceeded = AtomicBool::new(false);
        let output = read_bounded_worker_output(
            std::io::Cursor::new(b"0123456789"),
            4,
            &output_bound_exceeded,
        )
        .expect("read bounded output");

        assert_eq!(output, b"0123");
        assert!(output_bound_exceeded.load(Ordering::Relaxed));
    }

    #[test]
    fn planning_worker_rejects_an_oversized_frame_before_allocating_it() {
        let oversized = u32::try_from(MAX_PLANNING_FRAME_BYTES + 1)
            .expect("planning frame bound fits in u32")
            .to_le_bytes();
        let error = read_planning_frame(&mut oversized.as_slice())
            .expect_err("oversized planning frame must be rejected");

        assert!(error.to_string().contains("exceeds its frame bound"));
    }

    #[test]
    fn planning_worker_diagnostics_keep_the_most_recent_bounded_tail() {
        let mut retained = vec![b'a'; MAX_PLANNING_DIAGNOSTIC_BYTES];
        retain_planning_diagnostics(&mut retained, b"terminal failure");

        assert_eq!(retained.len(), MAX_PLANNING_DIAGNOSTIC_BYTES);
        assert!(retained.ends_with(b"terminal failure"));
    }

    fn test_model_instance_allocation() -> icn_contracts::models::ModelInstanceAllocation {
        icn_contracts::models::ModelInstanceAllocation {
            context_window_tokens: 1,
            parallel_sequences: 1,
            physical_context_tokens: 1,
            memory_domains: Vec::new(),
        }
    }

    #[tokio::test]
    async fn concurrent_model_instance_admission_is_single_and_identity_preserving() {
        let instances = InstanceEntries::new();
        let mut changes = instances.subscribe();
        let instance_id = ModelInstanceId("instance".to_owned());
        let configuration_id = ModelServingConfigurationId("configuration".to_owned());

        let (first, second) = tokio::join!(
            instances.admit(instance_id.clone(), configuration_id.clone()),
            instances.admit(instance_id.clone(), configuration_id.clone()),
        );
        let (first_stop, first_is_new) = first.expect("first admission should succeed");
        let (second_stop, second_is_new) = second.expect("second admission should succeed");
        assert_ne!(first_is_new, second_is_new);
        assert!(Arc::ptr_eq(&first_stop, &second_stop));
        assert_eq!(
            changes
                .recv()
                .await
                .expect("one admission invalidation")
                .revision,
            1
        );
        assert!(changes.try_recv().is_err());

        first_stop.store(true, Ordering::Release);
        let late_loading = ModelInstance {
            id: instance_id.clone(),
            configuration_id: configuration_id.clone(),
            lifecycle: ModelInstanceLifecycle::Loading {
                stage: ModelLoadStage::Loading,
                progress: Some(0.5),
                planned_allocation: None,
            },
        };
        instances.publish(late_loading.clone()).await;
        assert_ne!(
            instances.instance(&instance_id).await,
            Some(late_loading.clone()),
            "Loading cannot advance after Stop has been requested",
        );
        instances
            .publish(ModelInstance {
                id: instance_id.clone(),
                configuration_id: configuration_id.clone(),
                lifecycle: ModelInstanceLifecycle::Stopping {
                    reason: ModelReleaseReason::UserStop,
                    allocation: ModelStoppingAllocation::Planned { allocation: None },
                },
            })
            .await;
        instances.publish(late_loading).await;
        assert!(matches!(
            instances
                .instance(&instance_id)
                .await
                .expect("instance remains observable")
                .lifecycle,
            ModelInstanceLifecycle::Stopping { .. }
        ));

        let terminal = ModelInstance {
            id: instance_id.clone(),
            configuration_id: configuration_id.clone(),
            lifecycle: ModelInstanceLifecycle::Stopped {
                reason: ModelReleaseReason::UserStop,
            },
        };
        instances.publish(terminal.clone()).await;
        let (repeated_stop, repeated_is_new) = instances
            .admit(instance_id.clone(), configuration_id)
            .await
            .expect("equivalent admission should resolve to the existing instance");

        assert!(!repeated_is_new);
        assert!(Arc::ptr_eq(&first_stop, &repeated_stop));

        let second_id = ModelInstanceId("second-instance".to_owned());
        let second_configuration = ModelServingConfigurationId("second-configuration".to_owned());
        instances
            .admit(second_id.clone(), second_configuration.clone())
            .await
            .expect("second identity should be admitted");
        let second_terminal = ModelInstance {
            id: second_id,
            configuration_id: second_configuration,
            lifecycle: ModelInstanceLifecycle::Failed {
                failure: ModelInstanceFailure::Operation {
                    code: "test_failure".to_owned(),
                    message: "test failure".to_owned(),
                    retryable: false,
                },
            },
        };
        instances.publish(second_terminal.clone()).await;
        assert_eq!(
            instances.snapshot().await.instances,
            vec![terminal, second_terminal],
            "terminal identity history must remain exactly observable",
        );

        let conflict = instances
            .admit(
                instance_id,
                ModelServingConfigurationId("different-configuration".to_owned()),
            )
            .await
            .expect_err("an instance ID cannot be rebound");
        assert_eq!(conflict.code, "model_instance_identity_conflict");
        assert!(!conflict.retryable);
    }

    #[test]
    fn performance_evidence_preserves_the_exact_requested_context_and_bounds() {
        let evidence = performance_result(GenerationPerformanceAssessment {
            confidence: icn_contracts::GenerationPerformanceConfidence::Moderate,
            workload: "baseline_single_sequence_decode".to_owned(),
            always_active_weight_bytes: 10,
            routed_expert_weight_bytes: 80,
            expert_count: 8,
            expert_used_count: 2,
            cross_memory_domain_placement: true,
            context_tokens: 262_144,
            kv_bytes_read_per_token: 8_192,
            lower_tokens_per_second: 15.0,
            expected_tokens_per_second: 18.0,
            upper_tokens_per_second: 21.0,
        });

        assert_eq!(evidence.context_tokens, 262_144);
        assert_eq!(evidence.lower_tokens_per_second, 15.0);
        assert_eq!(evidence.estimated_tokens_per_second, 18.0);
        assert_eq!(evidence.upper_tokens_per_second, 21.0);
        assert_eq!(evidence.confidence, PerformanceConfidence::Moderate);
    }

    fn parity_test_defaults() -> ModelPlanDefaults {
        ModelPlanDefaults {
            context_size: 128,
            physical_context_size: 128,
            batch_size: 128,
            ubatch_size: 64,
            max_sequences: 1,
            prefill_quantum: 128,
            execution: icn_contracts::ExecutionConfig {
                kv_unified: false,
                ..icn_contracts::ExecutionConfig::default()
            },
            projector_use_gpu: true,
            projector_warmup: true,
            image_min_tokens: None,
            image_max_tokens: None,
        }
    }

    #[test]
    fn preview_parallelism_reserves_one_full_context_partition_per_sequence() {
        let assessor = NativeResolvedModelAssessor {
            defaults: parity_test_defaults(),
            cache: None,
            planning_executor: PlanningExecutor::InProcess(test_native_backend()),
            native_backend: test_native_backend(),
            native_executor: Arc::new(RwLock::new(None)),
            gate: Arc::new(tokio::sync::Mutex::new(())),
            assessment_work_gates: Arc::new(tokio::sync::Mutex::new(
                std::collections::BTreeMap::new(),
            )),
            assessment_concurrency: AssessmentConcurrency::new(1),
            hardware_calibration: Arc::new(fixture_hardware_calibration()),
            enabled_backends: Arc::new(vec!["cpu".to_owned()]),
        };

        let defaults = assessor.effective_defaults(Some(&ModelPreviewProfile {
            id: "p4".to_owned(),
            context_length: 32_768,
            parallel_sequences: 4,
            performance_context_tokens: vec![32_768],
        }));

        assert_eq!(defaults.context_size, 32_768);
        assert_eq!(defaults.physical_context_size, 131_072);
        assert_eq!(defaults.max_sequences, 4);
        assert!(!defaults.execution.kv_unified);
    }

    #[test]
    fn load_allocation_descends_to_the_highest_freshly_permitted_parallelism() {
        let gib = 1024 * 1024 * 1024;
        let sample = memory_supervisor::MemorySample {
            captured_at: std::time::Instant::now(),
            physical_capacity_bytes: 16 * gib,
            physical_available_bytes: 7 * gib,
            allocation_capacity_bytes: 16 * gib,
            allocation_headroom_bytes: 7 * gib,
        };

        assert_eq!(
            select_model_allocation(&[(1, gib), (2, 4 * gib), (3, 6 * gib)], sample),
            Some((2, 4 * gib))
        );
    }

    #[test]
    fn low_memory_failure_preserves_attempt_arithmetic() {
        let gib = 1024 * 1024 * 1024;
        let sample = memory_supervisor::MemorySample {
            captured_at: std::time::Instant::now(),
            physical_capacity_bytes: 16 * gib,
            physical_available_bytes: 7 * gib,
            allocation_capacity_bytes: 16 * gib,
            allocation_headroom_bytes: 7 * gib,
        };

        assert_eq!(
            ModelOperationFailure::low_memory(sample, 6 * gib, 1).into_instance_failure(),
            ModelInstanceFailure::LowMemory {
                code: LOW_MEMORY_FAILURE_CODE.to_owned(),
                message: format!(
                    "not enough memory available: model requires {} bytes plus {} bytes reserved for the system; {} bytes are available (1 byte short)",
                    6 * gib,
                    gib,
                    7 * gib,
                ),
                retryable: true,
                required_system_memory_bytes: 6 * gib,
                allocation_headroom_bytes: 7 * gib,
                system_reserve_bytes: gib,
                load_boundary_bytes: 7 * gib,
                minimum_additional_available_bytes: 1,
                parallel_sequences: 1,
            }
        );
    }

    #[test]
    fn replacement_preview_credits_only_memory_the_current_residency_will_release() {
        let gib = 1024 * 1024 * 1024;
        let sample = memory_supervisor::MemorySample {
            captured_at: std::time::Instant::now(),
            physical_capacity_bytes: 16 * gib,
            physical_available_bytes: 3 * gib,
            allocation_capacity_bytes: 20 * gib,
            allocation_headroom_bytes: 4 * gib,
        };

        assert_eq!(select_model_allocation(&[(1, 4 * gib)], sample), None);
        let credited = credit_replaced_instance_memory(sample, 3 * gib);
        assert_eq!(credited.physical_available_bytes, 6 * gib);
        assert_eq!(credited.allocation_headroom_bytes, 7 * gib);
        assert_eq!(
            select_model_allocation(&[(1, 4 * gib)], credited),
            Some((1, 4 * gib))
        );
    }

    #[test]
    fn idle_release_requires_the_exact_residency_and_a_full_idle_interval() {
        let timeout = std::time::Duration::from_secs(600);
        let policy = ModelResidencyPolicyState {
            generation: 3,
            idle_timeout: timeout,
        };
        let now = std::time::Instant::now();
        let resident = ReadyInstanceRecord {
            configuration_id: ModelServingConfigurationId("model-a".to_owned()),
            instance_id: ModelInstanceId("instance-a".to_owned()),
            generation: 7,
            package_ids: Vec::new(),
            allocation: test_model_instance_allocation(),
            runtime: InstanceRuntime::empty(),
        };
        let activity = InstanceActivity {
            generation: 7,
            active_leases: 0,
            idle_since: Some(now - timeout),
        };

        assert_eq!(
            idle_release_elapsed(&resident, Some(&resident), activity, 3, policy, now),
            Some(timeout)
        );
        assert_eq!(
            idle_release_elapsed(
                &resident,
                Some(&resident),
                InstanceActivity {
                    idle_since: Some(now - timeout + std::time::Duration::from_nanos(1)),
                    ..activity
                },
                3,
                policy,
                now,
            ),
            None
        );
    }

    #[test]
    fn residency_policy_generations_are_idempotent_and_monotonic() {
        let current = ModelResidencyPolicyState {
            generation: 4,
            idle_timeout: std::time::Duration::from_secs(600),
        };
        assert_eq!(
            next_model_residency_policy(current, 4, current.idle_timeout).unwrap(),
            current,
        );
        assert!(
            next_model_residency_policy(current, 4, std::time::Duration::from_secs(3600),).is_err()
        );
        assert!(next_model_residency_policy(current, 3, current.idle_timeout).is_err());
        assert_eq!(
            next_model_residency_policy(current, 5, std::time::Duration::from_secs(3600),).unwrap(),
            ModelResidencyPolicyState {
                generation: 5,
                idle_timeout: std::time::Duration::from_secs(3600),
            },
        );
    }

    #[test]
    fn idle_release_rejects_active_or_stale_observations() {
        let timeout = std::time::Duration::from_secs(600);
        let policy = ModelResidencyPolicyState {
            generation: 3,
            idle_timeout: timeout,
        };
        let now = std::time::Instant::now();
        let resident = ReadyInstanceRecord {
            configuration_id: ModelServingConfigurationId("model-a".to_owned()),
            instance_id: ModelInstanceId("instance-a".to_owned()),
            generation: 7,
            package_ids: Vec::new(),
            allocation: test_model_instance_allocation(),
            runtime: InstanceRuntime::empty(),
        };
        let stale = ReadyInstanceRecord {
            configuration_id: ModelServingConfigurationId("model-b".to_owned()),
            instance_id: ModelInstanceId("instance-b".to_owned()),
            generation: 8,
            package_ids: Vec::new(),
            allocation: test_model_instance_allocation(),
            runtime: InstanceRuntime::empty(),
        };
        let idle = InstanceActivity {
            generation: 7,
            active_leases: 0,
            idle_since: Some(now - timeout),
        };

        assert_eq!(
            idle_release_elapsed(
                &resident,
                Some(&resident),
                InstanceActivity {
                    active_leases: 1,
                    ..idle
                },
                3,
                policy,
                now,
            ),
            None
        );
        assert_eq!(
            idle_release_elapsed(&resident, Some(&stale), idle, 3, policy, now),
            None
        );
        assert_eq!(
            idle_release_elapsed(
                &resident,
                Some(&resident),
                InstanceActivity {
                    generation: 8,
                    ..idle
                },
                3,
                policy,
                now,
            ),
            None
        );
        assert_eq!(
            idle_release_elapsed(&resident, Some(&resident), idle, 2, policy, now,),
            None
        );
    }

    #[tokio::test(start_paused = true)]
    async fn residency_policy_changes_restart_connected_and_disconnected_idle_intervals() {
        let resident = ReadyInstanceRecord {
            configuration_id: ModelServingConfigurationId("model-a".to_owned()),
            instance_id: ModelInstanceId("instance-a".to_owned()),
            generation: 7,
            package_ids: Vec::new(),
            allocation: test_model_instance_allocation(),
            runtime: InstanceRuntime::empty(),
        };
        let mut activity = InstanceActivity {
            generation: resident.generation,
            active_leases: 0,
            idle_since: Some(tokio::time::Instant::now().into_std()),
        };

        tokio::time::advance(std::time::Duration::from_secs(5 * 60)).await;
        restart_idle_interval(
            &mut activity,
            resident.generation,
            tokio::time::Instant::now().into_std(),
        );
        let connected = ModelResidencyPolicyState {
            generation: 1,
            idle_timeout: std::time::Duration::from_secs(60 * 60),
        };
        tokio::time::advance(connected.idle_timeout - std::time::Duration::from_nanos(1)).await;
        assert_eq!(
            idle_release_elapsed(
                &resident,
                Some(&resident),
                activity,
                connected.generation,
                connected,
                tokio::time::Instant::now().into_std(),
            ),
            None,
        );
        tokio::time::advance(std::time::Duration::from_nanos(1)).await;
        assert!(
            idle_release_elapsed(
                &resident,
                Some(&resident),
                activity,
                connected.generation,
                connected,
                tokio::time::Instant::now().into_std(),
            )
            .is_some()
        );

        restart_idle_interval(
            &mut activity,
            resident.generation,
            tokio::time::Instant::now().into_std(),
        );
        let disconnected = ModelResidencyPolicyState {
            generation: 2,
            idle_timeout: std::time::Duration::from_secs(10 * 60),
        };
        tokio::time::advance(disconnected.idle_timeout).await;
        assert!(
            idle_release_elapsed(
                &resident,
                Some(&resident),
                activity,
                disconnected.generation,
                disconnected,
                tokio::time::Instant::now().into_std(),
            )
            .is_some()
        );
    }

    #[tokio::test(start_paused = true)]
    async fn active_inference_wins_the_deadline_and_receives_a_fresh_idle_interval() {
        let timeout = std::time::Duration::from_secs(10 * 60);
        let policy = ModelResidencyPolicyState {
            generation: 3,
            idle_timeout: timeout,
        };
        let resident = ReadyInstanceRecord {
            configuration_id: ModelServingConfigurationId("model-a".to_owned()),
            instance_id: ModelInstanceId("instance-a".to_owned()),
            generation: 7,
            package_ids: Vec::new(),
            allocation: test_model_instance_allocation(),
            runtime: InstanceRuntime::empty(),
        };
        let mut activity = InstanceActivity {
            generation: resident.generation,
            active_leases: 1,
            idle_since: None,
        };

        tokio::time::advance(timeout).await;
        assert_eq!(
            idle_release_elapsed(
                &resident,
                Some(&resident),
                activity,
                policy.generation,
                policy,
                tokio::time::Instant::now().into_std(),
            ),
            None,
        );

        activity.active_leases = 0;
        activity.idle_since = Some(tokio::time::Instant::now().into_std());
        tokio::time::advance(timeout - std::time::Duration::from_nanos(1)).await;
        assert_eq!(
            idle_release_elapsed(
                &resident,
                Some(&resident),
                activity,
                policy.generation,
                policy,
                tokio::time::Instant::now().into_std(),
            ),
            None,
        );
        tokio::time::advance(std::time::Duration::from_nanos(1)).await;
        assert!(
            idle_release_elapsed(
                &resident,
                Some(&resident),
                activity,
                policy.generation,
                policy,
                tokio::time::Instant::now().into_std(),
            )
            .is_some()
        );
    }

    #[test]
    fn capacity_cache_keys_share_resolved_profile_identity() {
        let assessor = NativeResolvedModelAssessor {
            defaults: parity_test_defaults(),
            cache: None,
            planning_executor: PlanningExecutor::InProcess(test_native_backend()),
            native_backend: test_native_backend(),
            native_executor: Arc::new(RwLock::new(None)),
            gate: Arc::new(tokio::sync::Mutex::new(())),
            assessment_work_gates: Arc::new(tokio::sync::Mutex::new(
                std::collections::BTreeMap::new(),
            )),
            assessment_concurrency: AssessmentConcurrency::new(1),
            hardware_calibration: Arc::new(fixture_hardware_calibration()),
            enabled_backends: Arc::new(vec!["cpu".to_owned()]),
        };
        let snapshot = HardwareSnapshot {
            captured_at: 1,
            platform: "test".to_owned(),
            architecture: "test".to_owned(),
            system_product_name: None,
            cpu_model: None,
            logical_cores: 1,
            system_memory: icn_contracts::HardwareSystemMemory {
                physical_capacity_bytes: 10,
                physical_available_bytes: 10,
                allocation_capacity_bytes: 10,
                allocation_headroom_bytes: 10,
                assess_reserve_bytes: 0,
                abort_reserve_bytes: 0,
            },
            native_build: "native".to_owned(),
            enabled_backends: vec!["cpu".to_owned()],
            topology_fingerprint: "topology".to_owned(),
            memory_domains: vec![icn_contracts::HardwareMemoryDomain {
                id: icn_contracts::MemoryDomainId::system(),
                kind: icn_contracts::HardwareMemoryDomainKind::System,
                total_capacity_bytes: 10,
                stable_capacity_bytes: 10,
                current_free_bytes: Some(10),
                shares_system_memory: true,
                devices: Vec::new(),
            }],
        };
        let equivalent_preview = ModelPreviewProfile {
            id: "caller-correlation-does-not-affect-assessment".to_owned(),
            context_length: 128,
            parallel_sequences: 1,
            performance_context_tokens: vec![128],
        };
        assert_eq!(
            assessor
                .capacity_assessment_cache_key(None, &snapshot)
                .unwrap(),
            assessor
                .capacity_assessment_cache_key(Some(&equivalent_preview), &snapshot)
                .unwrap()
        );
        assert_ne!(
            assessor
                .capacity_assessment_cache_key(None, &snapshot)
                .unwrap(),
            assessor
                .capacity_assessment_cache_key(
                    Some(&ModelPreviewProfile {
                        context_length: 4096,
                        ..equivalent_preview.clone()
                    }),
                    &snapshot,
                )
                .unwrap()
        );
        let mut availability_only_change = snapshot.clone();
        availability_only_change.captured_at = 2;
        availability_only_change
            .system_memory
            .physical_available_bytes = 0;
        availability_only_change
            .system_memory
            .allocation_headroom_bytes = 0;
        assert_eq!(
            assessor
                .capacity_assessment_cache_key(Some(&equivalent_preview), &snapshot)
                .unwrap(),
            assessor
                .capacity_assessment_cache_key(Some(&equivalent_preview), &availability_only_change)
                .unwrap()
        );
        assert_ne!(
            assessor
                .capacity_assessment_cache_key_with_policy(
                    Some(&equivalent_preview),
                    &snapshot,
                    CapacityPolicy::default(),
                )
                .unwrap(),
            assessor
                .capacity_assessment_cache_key_with_policy(
                    Some(&equivalent_preview),
                    &snapshot,
                    CapacityPolicy {
                        reserve_bytes_per_domain: 1,
                        system_reserve_bytes: Some(2),
                    },
                )
                .unwrap()
        );
    }

    #[test]
    fn execution_cache_identity_tracks_concrete_calibration_metrics() {
        let calibration = |bytes_per_second, elapsed_microseconds| NativeHardwareCalibration {
            method: llama_cpp_2::model::params::fit::FIT_CALIBRATION_METHOD.to_owned(),
            metrics: vec![NativeHardwareCalibrationMetric {
                backend_type: 2,
                backend: "CUDA".to_owned(),
                device_id: Some("GPU0".to_owned()),
                tensor_type: 1,
                routed: false,
                bytes_per_second,
                launch_microseconds: 10.0,
                relative_spread: 0.05,
                sample_count: 5,
                measured_microseconds: 50_000,
                stable: true,
            }],
            elapsed_microseconds,
        };
        let original = calibration(1_000.0, 60_000);
        let changed_measurement = calibration(2_000.0, 60_000);
        let changed_wall_time = calibration(1_000.0, 70_000);

        assert_ne!(
            NativeResolvedModelAssessor::hardware_calibration_identity(&original).unwrap(),
            NativeResolvedModelAssessor::hardware_calibration_identity(&changed_measurement)
                .unwrap()
        );
        assert_eq!(
            NativeResolvedModelAssessor::hardware_calibration_identity(&original).unwrap(),
            NativeResolvedModelAssessor::hardware_calibration_identity(&changed_wall_time).unwrap()
        );
    }

    fn sparse_header_copy(source: &std::path::Path, destination: &std::path::Path) {
        use std::io::{Read, Write};

        let inspection = icn_models::gguf::inspect(source).expect("inspect complete fixture");
        let header_bytes = usize::try_from(inspection.header_bytes).expect("header fits usize");
        let mut input = std::fs::File::open(source).expect("open complete fixture");
        let mut header = vec![0_u8; header_bytes];
        input.read_exact(&mut header).expect("read complete header");
        let mut output = std::fs::File::create(destination).expect("create sparse preview");
        output.write_all(&header).expect("write preview header");
        output
            .set_len(input.metadata().expect("fixture metadata").len())
            .expect("preserve preview logical length");
    }

    /// Verified parity fixtures are optional in ordinary source checkouts, but CI/dev environments
    /// that stage them exercise both a tiny dense model and a production-scale MoE model.
    #[tokio::test]
    async fn available_and_sparse_preview_artifacts_have_identical_model_assessments() {
        let inference_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let fixtures = [
            inference_root.join("target/parity-models/tinyllamas/stories15M-q4_0.gguf"),
            inference_root
                .join("target/parity-models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
        ];
        let fixtures = fixtures
            .into_iter()
            .filter(|path| path.is_file())
            .collect::<Vec<_>>();
        if fixtures.is_empty() {
            return;
        }

        let assessor = Arc::new(NativeResolvedModelAssessor {
            defaults: parity_test_defaults(),
            cache: None,
            planning_executor: PlanningExecutor::InProcess(test_native_backend()),
            native_backend: test_native_backend(),
            native_executor: Arc::new(RwLock::new(None)),
            gate: Arc::new(tokio::sync::Mutex::new(())),
            assessment_work_gates: Arc::new(tokio::sync::Mutex::new(
                std::collections::BTreeMap::new(),
            )),
            assessment_concurrency: AssessmentConcurrency::new(1),
            hardware_calibration: Arc::new(fixture_hardware_calibration()),
            enabled_backends: Arc::new(vec!["cpu".to_owned()]),
        });
        let profile = ModelPreviewProfile {
            id: "parity".to_owned(),
            context_length: 128,
            parallel_sequences: 1,
            performance_context_tokens: vec![128],
        };

        for fixture in fixtures {
            let store =
                tempfile::tempdir_in(inference_root.join("target")).expect("temporary model store");
            let mut config = InventoryConfig::with_roots(
                store.path().join("inventory"),
                store.path().join("cache"),
            )
            .expect("inventory config");
            let hf_cache = store.path().join("hf-cache");
            let snapshot = hf_cache
                .join("models--test--parity")
                .join("snapshots")
                .join("0123456789abcdef");
            std::fs::create_dir_all(&snapshot).expect("create Hugging Face cache snapshot");
            std::fs::hard_link(
                &fixture,
                snapshot.join(fixture.file_name().expect("fixture filename")),
            )
            .expect("link parity fixture into Hugging Face cache");
            config.hf_cache_dirs.push(hf_cache);
            let manager = ManagedModelStore::open_with_template_assessor(
                config,
                Some(Arc::new(NativeTemplateAssessor {
                    worker_launcher: NativeWorkerLauncher::development(),
                })),
            )
            .await
            .expect("open inventory");
            manager
                .ensure_model_inventory()
                .await
                .expect("inspect available fixture");
            let model = manager
                .list()
                .await
                .expect("list inventory")
                .into_iter()
                .find(|model| {
                    model
                        .location
                        .components()
                        .iter()
                        .any(|component| component.path.file_name() == fixture.file_name())
                })
                .expect("fixture inventory model");
            let available = manager
                .resolve_ready(&model.id)
                .await
                .expect("resolve available fixture");

            let sparse_root = store.path().join("sparse-preview");
            std::fs::create_dir_all(&sparse_root).expect("create sparse preview directory");
            let mut preview = available.clone();
            for component in &mut preview.components {
                let destination =
                    sparse_root.join(component.path.file_name().expect("component file name"));
                sparse_header_copy(&component.path, &destination);
                component.path = destination;
            }

            let default_available_assessment = assessor
                .assess_resolved(available.clone(), None)
                .await
                .expect("assess available fixture with defaults");
            let default_preview_assessment = assessor
                .assess_resolved(preview.clone(), None)
                .await
                .expect("assess sparse preview with defaults");
            assert_eq!(
                default_preview_assessment,
                default_available_assessment,
                "the available and preview paths diverged for {}",
                fixture.display()
            );

            let available_assessment = assessor
                .assess_resolved(available, Some(&profile))
                .await
                .expect("assess available fixture");
            let preview_assessment = assessor
                .assess_resolved(preview, Some(&profile))
                .await
                .expect("assess sparse preview fixture");
            assert_eq!(
                preview_assessment,
                available_assessment,
                "preview and available assessment diverged for {}",
                fixture.display()
            );
        }
    }

    #[test]
    fn inventory_flag_aliases_parse() {
        let aliases = Cli::try_parse_from([
            "magnitude-icn",
            "serve",
            "--fake",
            "--models-dir",
            "/tmp/models",
            "--hf-cache-dir",
            "/tmp/hf",
        ])
        .expect("documented inventory flag aliases should parse");
        let Command::Serve {
            model_store,
            hf_caches,
            ..
        } = aliases.command
        else {
            panic!("expected serve command")
        };
        assert_eq!(model_store, Some(PathBuf::from("/tmp/models")));
        assert_eq!(hf_caches, vec![PathBuf::from("/tmp/hf")]);
    }

    #[test]
    fn managed_parent_pipe_flag_parses() {
        let managed =
            Cli::try_parse_from(["magnitude-icn", "serve", "--fake", "--exit-on-stdin-eof"])
                .expect("managed parent-pipe flag should parse");
        let Command::Serve {
            exit_on_stdin_eof, ..
        } = managed.command
        else {
            panic!("expected serve command")
        };
        assert!(exit_on_stdin_eof);
    }

    #[test]
    fn version_json_reports_native_and_build_provenance() {
        let value = build_identity::identity();
        assert_eq!(value.native_build, build_identity::native_build());
        assert_eq!(value.target, build_identity::TARGET);
        assert_eq!(value.profile, build_identity::PROFILE);
        assert!(!value.backends.is_empty());
    }
}
