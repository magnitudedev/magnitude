use std::net::SocketAddr;
use std::num::NonZeroU32;
use std::path::PathBuf;
#[cfg(not(test))]
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
    IcnInstallationBackend, IcnStartupRecord, IcnStartupRecordType,
};
use icn_contracts::models::{
    AssessModelResult, AssessModelsRequest, AssessModelsResponse, AssessmentEnvironmentId,
    FitModelResult, FitModelsRequest, FitModelsResponse, InstalledModelPackages as _,
    LoadModelReady, LoadModelRequest, MemoryAssessment, ModelEvaluator,
    ModelFailure as DomainModelFailure, ModelInstance, ModelInstanceId, ModelInstanceLifecycle,
    ModelInstancesInvalidation, ModelInstancesSnapshot, ModelLoadEvent, ModelLoadPlan,
    ModelLoadStage, ModelOfferingTarget as DomainModelOfferingTarget, ModelPackageId,
    ModelPackageOperand, ModelReleaseReason, ModelServingConfiguration,
    ModelServingConfigurationId, ModelStoppingAllocation, ModelTargetInput, OfferingAssessment,
    OfferingAssessmentId, PerformanceConfidence, PerformanceEvidence, PerformanceUnavailable,
    PreviewModelLoadRequest, RemoveInstalledModelPackageResponse,
    ServingProfile as DomainServingProfile,
};
use icn_contracts::{
    CacheType, CompletionBackend, ComponentRole, ExecutionConfig, ExecutionIntent, FlashAttention,
    GenerationPerformanceAssessment, GpuLayers, HardwareAssessment, HardwareProvider,
    HardwareSnapshot, InventoryError, InventoryHardwareAssessor, ModelExecutionAssessment,
    ModelHardwareAssessor, ModelPreviewProfile, ProjectorConfig, ResolvedModel, SplitMode,
    TemplateAssessment, TemplateAssessor,
};
use icn_engine::{ModelLoadObserver, MtpCandidateSelection, NativeBackend};
use icn_hardware::CapacityPolicy;
use icn_models::{
    InventoryConfig, ManagedModelDownloads, ModelCache, ModelIndexKind, ModelManager,
    ModelPreviewService, ReleaseCatalog, ReleaseRecommendableCatalog,
    ResolvingRecommendableCatalog, canonical_package_id, load_release_catalog, offering_target_id,
    release_catalog_manifest,
};
use sha2::{Digest, Sha256};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tower_http::trace::{DefaultOnResponse, TraceLayer};

mod backend_eligibility;
mod build_identity;
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
#[cfg(not(test))]
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
        /// Owning process. ICN exits if this process disappears.
        #[arg(long)]
        parent_pid: Option<u32>,
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
        /// Additional read-only directories containing GGUF models.
        #[arg(long = "model-source")]
        model_sources: Vec<PathBuf>,
        /// Additional read-only Hugging Face hub cache roots.
        #[arg(long = "hf-cache", visible_alias = "hf-cache-dir")]
        hf_caches: Vec<PathBuf>,
        /// Verified release or prepared development installation.
        #[arg(long)]
        installation: Option<PathBuf>,
    },
    /// Maintains the release-bound curated model catalog.
    #[command(hide = true)]
    Catalog {
        #[command(subcommand)]
        command: CatalogCommand,
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
    PlanWorker {
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

#[derive(Debug, Subcommand)]
enum CatalogCommand {
    /// Resolve the curated source catalog with the production model parser.
    Generate {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        model_store: PathBuf,
        #[arg(long)]
        cache_root: PathBuf,
        #[arg(long = "hf-cache")]
        hf_caches: Vec<PathBuf>,
    },
    /// Validate prepared release catalog sidecars.
    Check {
        #[arg(long)]
        installation: PathBuf,
    },
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct ModelPlanDefaults {
    context_size: u32,
    physical_context_size: u32,
    batch_size: u32,
    ubatch_size: u32,
    max_sequences: u32,
    prefill_quantum: u32,
    execution: ExecutionConfig,
    projector_use_gpu: bool,
    projector_warmup: bool,
    image_min_tokens: Option<NonZeroU32>,
    image_max_tokens: Option<NonZeroU32>,
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
    sample.available_bytes = sample
        .available_bytes
        .saturating_add(releasable_system_memory_bytes)
        .min(sample.total_bytes);
    sample.available_commit_bytes = sample.available_commit_bytes.map(|available| {
        available
            .saturating_add(releasable_system_memory_bytes)
            .min(sample.commit_limit_bytes.unwrap_or(u64::MAX))
    });
    sample
}

fn model_plan_defaults() -> ModelPlanDefaults {
    ModelPlanDefaults {
        // Managed product models always overwrite context from their persisted serving
        // configuration. This conservative value is only the discovery/migration fallback for
        // unmanaged local artifacts that predate serving configurations.
        context_size: 4096,
        physical_context_size: 4096,
        batch_size: 512,
        ubatch_size: 512,
        max_sequences: 1,
        prefill_quantum: 512,
        execution: ExecutionConfig {
            gpu_layers: GpuLayers::Auto,
            use_mmap: true,
            use_mlock: false,
            split_mode: SplitMode::Layer,
            tensor_split: None,
            cache_type_k: CacheType::F16,
            cache_type_v: CacheType::F16,
            offload_kqv: true,
            operation_offload: true,
            swa_full: false,
            kv_unified: false,
            threads: None,
            threads_batch: None,
            flash_attention: FlashAttention::Auto,
        },
        projector_use_gpu: true,
        projector_warmup: true,
        image_min_tokens: None,
        image_max_tokens: None,
    }
}

fn execution_intent(
    model_path: PathBuf,
    projector_path: Option<PathBuf>,
    defaults: &ModelPlanDefaults,
) -> anyhow::Result<ExecutionIntent> {
    Ok(ExecutionIntent {
        model_path,
        context_size: defaults.context_size,
        physical_context_size: defaults.physical_context_size,
        batch_size: defaults.batch_size,
        ubatch_size: defaults.ubatch_size,
        max_sequences: defaults.max_sequences,
        prefill_quantum: defaults.prefill_quantum,
        execution: defaults.execution.clone(),
        projector: projector_path.map(|path| {
            let mut projector = ProjectorConfig::new(path);
            projector.use_gpu = defaults.projector_use_gpu;
            projector.warmup = defaults.projector_warmup;
            projector.image_min_tokens = defaults.image_min_tokens;
            projector.image_max_tokens = defaults.image_max_tokens;
            projector
        }),
        mtp: icn_contracts::MtpConfig::default(),
    })
}

#[derive(Clone)]
struct NativeHardwareAssessor {
    defaults: ModelPlanDefaults,
    cache: Option<ModelCache>,
    native_backend: NativeBackend,
    worker_launcher: NativeWorkerLauncher,
    native_executor: Arc<RwLock<Option<Weak<RemoteBackend>>>>,
    gate: Arc<tokio::sync::Mutex<()>>,
    assessment_work_gates:
        Arc<tokio::sync::Mutex<std::collections::BTreeMap<String, Weak<tokio::sync::Mutex<()>>>>>,
    planning_slots: Arc<tokio::sync::Semaphore>,
    calibration: Arc<tokio::sync::Mutex<CalibrationCache>>,
}

#[derive(Default)]
struct CalibrationCache {
    topology_fingerprint: Option<String>,
    evidence: Option<String>,
    result: Option<Result<llama_cpp_2::model::params::fit::FitCalibration, String>>,
}

type NativeAssessorServices = (
    Arc<NativeHardwareAssessor>,
    Arc<RwLock<Option<Weak<RemoteBackend>>>>,
);

fn native_assessor_services(
    inventory: &Arc<ModelManager>,
    native_backend: NativeBackend,
    defaults: ModelPlanDefaults,
    worker_launcher: NativeWorkerLauncher,
) -> NativeAssessorServices {
    let native_executor = Arc::new(RwLock::new(None));
    let assessor = Arc::new(NativeHardwareAssessor {
        defaults,
        cache: Some(inventory.derived_cache().clone()),
        native_backend,
        worker_launcher,
        native_executor: Arc::clone(&native_executor),
        gate: Arc::new(tokio::sync::Mutex::new(())),
        assessment_work_gates: Arc::new(tokio::sync::Mutex::new(std::collections::BTreeMap::new())),
        planning_slots: Arc::new(tokio::sync::Semaphore::new(planner_concurrency())),
        calibration: Arc::new(tokio::sync::Mutex::new(CalibrationCache::default())),
    });
    (assessor, native_executor)
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
struct PersistedCalibration {
    captured_at: u64,
    calibration: llama_cpp_2::model::params::fit::FitCalibration,
}

const MODEL_ASSESSMENT_CONCURRENCY: usize = 12;
const CALIBRATION_MAX_AGE_SECONDS: u64 = 30 * 24 * 60 * 60;

fn unix_time_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn calibration_evidence(snapshot: &HardwareSnapshot) -> Result<String, InventoryError> {
    serde_json::to_string(&(
        llama_cpp_2::model::params::fit::FIT_CALIBRATION_METHOD,
        &snapshot.native_build,
        &snapshot.enabled_backends,
        &snapshot.topology_fingerprint,
        &snapshot.platform,
        &snapshot.architecture,
        sysinfo::System::long_os_version(),
        sysinfo::System::kernel_long_version(),
    ))
    .map_err(|error| InventoryError::Internal(error.to_string()))
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct PlanningWorkerRequest {
    hardware: HardwareSnapshot,
    primary: PathBuf,
    projector: Option<PathBuf>,
    mtp: Vec<PathBuf>,
    defaults: Vec<ModelPlanDefaults>,
    estimate_performance: bool,
    calibration: Option<llama_cpp_2::model::params::fit::FitCalibration>,
    calibration_unavailable: Option<String>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct PlanningWorkerResponse {
    assessments: Vec<ModelExecutionAssessment>,
    calibration: Option<Result<llama_cpp_2::model::params::fit::FitCalibration, String>>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct TemplateWorkerRequest {
    model_path: PathBuf,
}

#[derive(Debug)]
struct NativeTemplateAssessor {
    worker_launcher: NativeWorkerLauncher,
}

fn native_template_identity() -> &'static str {
    concat!(
        "icn-native-model-template:",
        env!("CARGO_PKG_VERSION"),
        ":",
        env!("ICN_BINDINGS_REVISION"),
        ":",
        env!("ICN_NATIVE_BACKEND_REVISION")
    )
}

fn native_planner_identity() -> String {
    format!(
        "{}:{}:{}:{}",
        native_template_identity(),
        icn_models::PLANNER_STUB_FORMAT_IDENTITY,
        llama_cpp_2::model::params::fit::FIT_DECODE_WORKLOAD_METHOD,
        llama_cpp_2::model::params::fit::FIT_CALIBRATION_METHOD,
    )
}

impl TemplateAssessor for NativeTemplateAssessor {
    fn cache_identity(&self) -> &str {
        native_template_identity()
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
const PLANNING_WORKER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);
#[cfg(not(test))]
const MAX_PLANNING_WORKER_OUTPUT_BYTES: usize = 1024 * 1024;
const LOW_MEMORY_FAILURE_CODE: &str = "low_memory";
#[cfg(not(test))]
const TEMPLATE_WORKER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

impl NativeHardwareAssessor {
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
        Ok(self
            .assess_resolved_plans(resolved, profiles, false)
            .await?
            .into_iter()
            .map(|assessment| assessment.hardware)
            .collect())
    }

    async fn assess_resolved_execution_profiles(
        &self,
        resolved: ResolvedModel,
        profiles: Vec<ModelPreviewProfile>,
    ) -> Result<Vec<ModelExecutionAssessment>, InventoryError> {
        self.assess_resolved_plans(resolved, profiles, true).await
    }

    async fn assess_resolved_plans(
        &self,
        resolved: ResolvedModel,
        profiles: Vec<ModelPreviewProfile>,
        estimate_performance: bool,
    ) -> Result<Vec<ModelExecutionAssessment>, InventoryError> {
        let hardware = HardwareProvider::snapshot(self).await?;
        self.assess_resolved_plans_with_hardware(resolved, profiles, estimate_performance, hardware)
            .await
    }

    async fn assess_resolved_plans_cached(
        &self,
        resolved: ResolvedModel,
        profiles: Vec<ModelPreviewProfile>,
        snapshot: &HardwareSnapshot,
        configuration_id: &ModelServingConfigurationId,
    ) -> Result<Vec<ModelExecutionAssessment>, InventoryError> {
        let Some(cache) = self.cache.clone() else {
            return self
                .assess_resolved_plans_with_hardware(resolved, profiles, false, snapshot.clone())
                .await;
        };
        let topology = icn_contracts::MemoryTopology::from_snapshot(snapshot).ok_or_else(|| {
            InventoryError::Internal("hardware snapshot has an invalid memory topology".to_owned())
        })?;
        let content_id = resolved.model.content_id.clone();
        let mut entries = profiles
            .into_iter()
            .map(|profile| {
                let planner_evidence = self.assessment_cache_key(Some(&profile), snapshot)?;
                let evidence = serde_json::to_string(&(&configuration_id.0, planner_evidence))
                    .map_err(|error| InventoryError::Internal(error.to_string()))?;
                let assessment = cache.read_execution_assessment(&content_id, &evidence, &topology);
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
                    *assessment = cache.read_execution_assessment(&content_id, evidence, &topology);
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
                        .assess_resolved_plans_with_hardware(
                            resolved,
                            missing
                                .iter()
                                .map(|(_, profile, _)| profile.clone())
                                .collect(),
                            false,
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
                                task_cache.write_execution_assessment(
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

    async fn assess_resolved_plans_with_hardware(
        &self,
        resolved: ResolvedModel,
        profiles: Vec<ModelPreviewProfile>,
        estimate_performance: bool,
        hardware: HardwareSnapshot,
    ) -> Result<Vec<ModelExecutionAssessment>, InventoryError> {
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
        let mtp: Vec<PathBuf> = resolved
            .components
            .iter()
            .filter(|component| matches!(component.role, ComponentRole::Mtp | ComponentRole::Draft))
            .map(|component| component.path.clone())
            .collect();
        let defaults = if profiles.is_empty() {
            vec![self.effective_defaults(None)]
        } else {
            profiles
                .iter()
                .map(|profile| self.effective_defaults(Some(profile)))
                .collect()
        };
        // Hardware-only planning never calibrates. For execution assessment, only the first
        // request holds the calibration lock across native planning. Once a model-free result is
        // cached, concurrent model inspections proceed independently through the bounded pool.
        let mut calibration_guard = if estimate_performance {
            Some(self.calibration.lock().await)
        } else {
            None
        };
        let calibration_result = calibration_guard
            .as_ref()
            .and_then(|guard| guard.result.as_ref().cloned());
        if calibration_result.is_some() {
            calibration_guard.take();
        }
        let (calibration, calibration_unavailable) = match calibration_result {
            Some(Ok(calibration)) => (Some(calibration), None),
            Some(Err(error)) => (None, Some(error)),
            None => (None, None),
        };
        let request = PlanningWorkerRequest {
            hardware,
            primary,
            projector,
            mtp,
            defaults,
            estimate_performance,
            calibration,
            calibration_unavailable,
        };
        let permit = Arc::clone(&self.planning_slots)
            .acquire_owned()
            .await
            .map_err(|_| InventoryError::Internal("native planner pool closed".to_owned()))?;
        let worker_launcher = self.worker_launcher.clone();
        let response = match spawn_blocking_traced(move || {
            let _permit = permit;
            run_isolated_planning(request, &worker_launcher)
        })
        .await
        {
            Ok(Ok(response)) => response,
            Ok(Err(error)) => {
                return Err(InventoryError::Internal(format!(
                    "hardware assessment failed for {}: {error:#}",
                    id.0
                )));
            }
            Err(error) => {
                return Err(InventoryError::Internal(format!(
                    "hardware assessment task failed for {}: {error}",
                    id.0
                )));
            }
        };
        if let Some(mut guard) = calibration_guard
            && guard.result.is_none()
            && let Some(calibration) = response.calibration.clone()
        {
            if let Ok(value) = &calibration
                && let Some(evidence) = guard.evidence.as_deref()
                && let Some(cache) = &self.cache
            {
                let stable_metrics = value.metrics.iter().filter(|metric| metric.stable).count();
                let total_samples = value
                    .metrics
                    .iter()
                    .map(|metric| u64::from(metric.sample_count))
                    .sum::<u64>();
                tracing::info!(
                    method = value.method,
                    elapsed_microseconds = value.elapsed_microseconds,
                    metrics = value.metrics.len(),
                    stable_metrics,
                    total_samples,
                    "hardware calibration completed"
                );
                cache.write_index(
                    ModelIndexKind::Calibration,
                    evidence,
                    &PersistedCalibration {
                        captured_at: unix_time_seconds(),
                        calibration: value.clone(),
                    },
                );
            }
            guard.result = Some(calibration);
        }
        Ok(response.assessments)
    }

    fn assessment_cache_key(
        &self,
        profile: Option<&ModelPreviewProfile>,
        snapshot: &HardwareSnapshot,
    ) -> Result<String, InventoryError> {
        serde_json::to_string(&(
            icn_hardware::GENERATION_PERFORMANCE_METHOD,
            llama_cpp_2::model::params::fit::FIT_DECODE_WORKLOAD_METHOD,
            llama_cpp_2::model::params::fit::FIT_CALIBRATION_METHOD,
            &snapshot.native_build,
            &snapshot.enabled_backends,
            &snapshot.topology_fingerprint,
            self.effective_defaults(profile),
        ))
        .map_err(|error| InventoryError::Internal(error.to_string()))
    }

    #[cfg(test)]
    fn assessment_cache_key_with_policy(
        &self,
        profile: Option<&ModelPreviewProfile>,
        snapshot: &HardwareSnapshot,
        capacity_policy: CapacityPolicy,
    ) -> Result<String, InventoryError> {
        let snapshot = icn_hardware::with_capacity_policy(snapshot.clone(), capacity_policy);
        self.assessment_cache_key(profile, &snapshot)
    }
}

struct NativeModelEvaluator {
    models: Arc<ModelManager>,
    assessor: Arc<NativeHardwareAssessor>,
    release_catalog: Arc<ReleaseCatalog>,
}

#[derive(Clone)]
struct AssessmentEnvironment {
    id: AssessmentEnvironmentId,
    snapshot: HardwareSnapshot,
    topology: icn_contracts::MemoryTopology,
}

impl NativeModelEvaluator {
    fn new(
        models: Arc<ModelManager>,
        assessor: Arc<NativeHardwareAssessor>,
        release_catalog: Arc<ReleaseCatalog>,
    ) -> Self {
        Self {
            models,
            assessor,
            release_catalog,
        }
    }

    async fn environment(
        &self,
        reserve_bytes: u64,
    ) -> Result<AssessmentEnvironment, InventoryError> {
        let snapshot = HardwareProvider::snapshot(self.assessor.as_ref()).await?;
        let thresholds = icn_hardware::system_memory_thresholds(snapshot.system_memory.total_bytes);
        let snapshot = icn_hardware::with_capacity_policy(
            snapshot,
            CapacityPolicy {
                reserve_bytes_per_domain: reserve_bytes,
                system_reserve_bytes: Some(reserve_bytes.max(thresholds.assess_reserve_bytes)),
            },
        );
        let topology =
            icn_contracts::MemoryTopology::from_snapshot(&snapshot).ok_or_else(|| {
                InventoryError::Internal(
                    "hardware snapshot has an invalid memory topology".to_owned(),
                )
            })?;
        let mut digest = Sha256::new();
        let identity =
            serde_json::to_vec(&(&snapshot.native_build, &snapshot.topology_fingerprint))
                .map_err(|error| InventoryError::Internal(error.to_string()))?;
        digest.update(identity);
        Ok(AssessmentEnvironment {
            id: AssessmentEnvironmentId(format!("environment_{:x}", digest.finalize())),
            snapshot,
            topology,
        })
    }

    fn resolved_for_planning(
        resolved: &icn_contracts::models::ResolvedModelTarget,
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

    fn assessment_evidence(
        &self,
        target_id: &icn_contracts::models::ModelOfferingTargetId,
        profiles: &[DomainServingProfile],
        reserve_bytes: u64,
        include_performance: bool,
        environment: &AssessmentEnvironment,
    ) -> Result<Vec<String>, InventoryError> {
        profiles
            .iter()
            .map(|profile| {
                serde_json::to_string(&(
                    icn_hardware::GENERATION_PERFORMANCE_METHOD,
                    llama_cpp_2::model::params::fit::FIT_DECODE_WORKLOAD_METHOD,
                    llama_cpp_2::model::params::fit::FIT_CALIBRATION_METHOD,
                    &environment.id.0,
                    &target_id.0,
                    profile.context_length,
                    reserve_bytes,
                    include_performance,
                ))
                .map_err(|error| InventoryError::Internal(error.to_string()))
            })
            .collect()
    }

    fn cached_profiles(
        &self,
        target_id: &icn_contracts::models::ModelOfferingTargetId,
        profiles: &[DomainServingProfile],
        reserve_bytes: u64,
        include_performance: bool,
        environment: &AssessmentEnvironment,
    ) -> Result<Option<Vec<OfferingAssessment>>, InventoryError> {
        let evidence = self.assessment_evidence(
            target_id,
            profiles,
            reserve_bytes,
            include_performance,
            environment,
        )?;
        let results = evidence
            .iter()
            .map(|key| {
                self.models
                    .read_offering_assessment(key, &environment.topology)
            })
            .collect::<Option<Vec<_>>>();
        Ok(results)
    }

    async fn assess_profiles(
        &self,
        resolved: &icn_contracts::models::ResolvedModelTarget,
        profiles: &[DomainServingProfile],
        reserve_bytes: u64,
        include_performance: bool,
        environment: &AssessmentEnvironment,
    ) -> Result<Vec<OfferingAssessment>, InventoryError> {
        let hardware = &environment.snapshot;
        let thresholds = icn_hardware::system_memory_thresholds(hardware.system_memory.total_bytes);
        let system_reserve_bytes = reserve_bytes.max(thresholds.assess_reserve_bytes);
        let evidence = self.assessment_evidence(
            &resolved.target_id,
            profiles,
            reserve_bytes,
            include_performance,
            environment,
        )?;
        let mut results = evidence
            .iter()
            .map(|key| {
                self.models
                    .read_offering_assessment(key, &environment.topology)
            })
            .collect::<Vec<_>>();
        let missing = results
            .iter()
            .enumerate()
            .filter_map(|(index, assessment)| assessment.is_none().then_some(index))
            .collect::<Vec<_>>();
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
                context_length: profiles[*index].context_length,
                parallel_sequences: 1,
            })
            .collect::<Vec<_>>();
        let assessed = self
            .assessor
            .assess_resolved_plans_with_hardware(
                Self::resolved_for_planning(resolved),
                native_profiles,
                include_performance,
                environment.snapshot.clone(),
            )
            .await?;
        for (index, assessment) in missing.into_iter().zip(assessed) {
            let assessment = offering_assessment(
                &resolved.target_id,
                profiles[index].clone(),
                reserve_bytes,
                system_reserve_bytes,
                thresholds.warning_reserve_bytes,
                assessment,
            );
            self.models
                .write_offering_assessment(&evidence[index], &assessment);
            results[index] = Some(assessment);
        }
        Ok(results
            .into_iter()
            .map(|assessment| assessment.expect("missing assessment was populated"))
            .collect())
    }

    async fn fit_one(
        &self,
        resolved: &icn_contracts::models::ResolvedModelTarget,
        request: &FitModelsRequest,
        environment: &AssessmentEnvironment,
    ) -> Result<FitModelResult, InventoryError> {
        let target_limit = match &resolved.target {
            icn_contracts::models::ModelOfferingTarget::Package { package } => {
                package.properties.maximum_context_length
            }
            icn_contracts::models::ModelOfferingTarget::SpeculativeDecodingPair {
                target, ..
            } => target.properties.maximum_context_length,
        };
        let maximum_context = request
            .maximum_context_length
            .min(target_limit)
            .min(200_000);
        if maximum_context < request.minimum_context_length {
            return Ok(FitModelResult::InvalidTarget {
                request_id: icn_contracts::models::ModelAssessmentRequestId(String::new()),
                failure: DomainModelFailure {
                    code: "model_context_limit".to_owned(),
                    message: format!(
                        "model context limit is {maximum_context} tokens, below the requested minimum of {}",
                        request.minimum_context_length
                    ),
                    retryable: false,
                },
            });
        }

        let reserve = request
            .capacity_policy
            .required_reserve_bytes_per_memory_domain;
        let mut lower = request.minimum_context_length;
        let mut upper = maximum_context;
        let upper_assessment = self
            .assess_profiles(
                resolved,
                &[DomainServingProfile {
                    context_length: upper,
                }],
                reserve,
                false,
                environment,
            )
            .await?
            .pop()
            .expect("one requested profile produces one assessment");
        let mut best_context =
            matches!(upper_assessment, OfferingAssessment::Fits { .. }).then_some(upper);
        if best_context.is_none() {
            let minimum = self
                .assess_profiles(
                    resolved,
                    &[DomainServingProfile {
                        context_length: lower,
                    }],
                    reserve,
                    false,
                    environment,
                )
                .await?
                .pop()
                .expect("one requested profile produces one assessment");
            if !matches!(minimum, OfferingAssessment::Fits { .. }) {
                return Ok(match minimum {
                    OfferingAssessment::DoesNotFit {
                        limiting_resource,
                        deficit_bytes,
                        ..
                    } => FitModelResult::DoesNotFit {
                        request_id: icn_contracts::models::ModelAssessmentRequestId(String::new()),
                        target_id: resolved.target_id.clone(),
                        limiting_resource,
                        deficit_bytes,
                    },
                    OfferingAssessment::Incompatible { failure, .. } => {
                        FitModelResult::InvalidTarget {
                            request_id: icn_contracts::models::ModelAssessmentRequestId(
                                String::new(),
                            ),
                            failure,
                        }
                    }
                    OfferingAssessment::Fits { .. } => unreachable!(),
                });
            }
            best_context = Some(lower);
            while lower + 1 < upper {
                let middle = lower + (upper - lower) / 2;
                let assessment = self
                    .assess_profiles(
                        resolved,
                        &[DomainServingProfile {
                            context_length: middle,
                        }],
                        reserve,
                        false,
                        environment,
                    )
                    .await?
                    .pop()
                    .expect("one requested profile produces one assessment");
                if matches!(assessment, OfferingAssessment::Fits { .. }) {
                    lower = middle;
                    best_context = Some(middle);
                } else {
                    upper = middle;
                }
            }
        }
        let context_length = best_context.expect("minimum fitting context was recorded");
        let profile = DomainServingProfile { context_length };
        let assessment = self
            .assess_profiles(
                resolved,
                std::slice::from_ref(&profile),
                reserve,
                true,
                environment,
            )
            .await?
            .pop()
            .ok_or_else(|| InventoryError::Internal("fit assessment was omitted".to_owned()))?;
        let configuration_id = serving_configuration_id(&resolved.target_id, &profile);
        Ok(FitModelResult::Fitted {
            request_id: icn_contracts::models::ModelAssessmentRequestId(String::new()),
            target_id: resolved.target_id.clone(),
            configuration: ModelServingConfiguration {
                id: configuration_id,
                target: resolved.target.clone(),
                profile,
            },
            assessment,
        })
    }
}

fn serving_configuration_id(
    target_id: &icn_contracts::models::ModelOfferingTargetId,
    profile: &DomainServingProfile,
) -> ModelServingConfigurationId {
    let mut digest = Sha256::new();
    digest.update(target_id.0.as_bytes());
    digest.update(profile.context_length.to_le_bytes());
    ModelServingConfigurationId(format!("configuration_{:x}", digest.finalize()))
}

fn offering_assessment(
    target_id: &icn_contracts::models::ModelOfferingTargetId,
    profile: DomainServingProfile,
    reserve_bytes: u64,
    system_reserve_bytes: u64,
    warning_reserve_bytes: u64,
    assessment: ModelExecutionAssessment,
) -> OfferingAssessment {
    let context_tokens = profile.context_length;
    let configuration_id = serving_configuration_id(target_id, &profile);
    let mut digest = Sha256::new();
    digest.update(target_id.0.as_bytes());
    digest.update(profile.context_length.to_le_bytes());
    digest.update(reserve_bytes.to_le_bytes());
    digest.update(system_reserve_bytes.to_le_bytes());
    let assessment_id = OfferingAssessmentId(format!("assessment_{:x}", digest.finalize()));
    match assessment.hardware {
        HardwareAssessment::Fits { memory, .. } => {
            let (performance, performance_unavailable) =
                performance_result(assessment.performance, context_tokens);
            OfferingAssessment::Fits {
                profile,
                configuration_id,
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
                            warning_reserve_bytes: if domain.memory_domain.is_system() {
                                warning_reserve_bytes
                            } else {
                                domain_reserve
                            },
                            memory_domain_id: domain.memory_domain,
                            capacity_bytes: domain
                                .usable_capacity_bytes
                                .saturating_add(domain_reserve),
                            required_bytes: domain.required_bytes,
                            remaining_bytes: domain.margin_bytes,
                        }
                    })
                    .collect(),
                performance,
                performance_unavailable,
            }
        }
        HardwareAssessment::DoesNotFit {
            memory,
            limiting_resource,
            ..
        } => OfferingAssessment::DoesNotFit {
            profile,
            configuration_id,
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
                        warning_reserve_bytes: if domain.memory_domain.is_system() {
                            warning_reserve_bytes
                        } else {
                            domain_reserve
                        },
                        memory_domain_id: domain.memory_domain,
                        capacity_bytes: domain.usable_capacity_bytes.saturating_add(domain_reserve),
                        required_bytes: domain.required_bytes,
                        remaining_bytes: domain.margin_bytes,
                    }
                })
                .collect(),
            limiting_resource,
            deficit_bytes: memory.deficit_bytes.max(1),
        },
        HardwareAssessment::InvalidArtifact { code, message } => OfferingAssessment::Incompatible {
            profile,
            configuration_id,
            failure: DomainModelFailure {
                code,
                message,
                retryable: false,
            },
        },
        HardwareAssessment::IncompatibleArtifact { code, message } => {
            OfferingAssessment::Incompatible {
                profile,
                configuration_id,
                failure: DomainModelFailure {
                    code,
                    message,
                    retryable: false,
                },
            }
        }
        HardwareAssessment::NotAssessed { reason } => OfferingAssessment::Incompatible {
            profile,
            configuration_id,
            failure: DomainModelFailure {
                code: "not_assessed".to_owned(),
                message: reason,
                retryable: true,
            },
        },
    }
}

fn performance_result(
    assessment: GenerationPerformanceAssessment,
    context_tokens: u32,
) -> (Option<PerformanceEvidence>, Option<PerformanceUnavailable>) {
    match assessment {
        GenerationPerformanceAssessment::Estimated {
            method,
            confidence,
            points,
            ..
        } => {
            let evidence = points
                .into_iter()
                .find(|point| point.context_tokens == context_tokens)
                .map(|point| PerformanceEvidence {
                    context_tokens: point.context_tokens,
                    lower_tokens_per_second: point.lower_tokens_per_second,
                    estimated_tokens_per_second: point.expected_tokens_per_second,
                    upper_tokens_per_second: point.upper_tokens_per_second,
                    confidence: match confidence {
                        icn_contracts::GenerationPerformanceConfidence::High => {
                            PerformanceConfidence::High
                        }
                        icn_contracts::GenerationPerformanceConfidence::Moderate => {
                            PerformanceConfidence::Moderate
                        }
                        icn_contracts::GenerationPerformanceConfidence::Low => {
                            PerformanceConfidence::Low
                        }
                    },
                    method: method.clone(),
                });
            match evidence {
                Some(evidence) => (Some(evidence), None),
                None => (
                    None,
                    Some(PerformanceUnavailable {
                        method,
                        code: "requested_context_unavailable".to_owned(),
                        message: format!(
                            "generation performance has no point for requested context {context_tokens}"
                        ),
                    }),
                ),
            }
        }
        GenerationPerformanceAssessment::Unavailable {
            method,
            code,
            message,
        } => (
            None,
            Some(PerformanceUnavailable {
                method,
                code,
                message,
            }),
        ),
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

fn target_input_id(
    target: &ModelTargetInput,
) -> Result<icn_contracts::models::ModelOfferingTargetId, String> {
    match target {
        ModelTargetInput::Package { package } => {
            Ok(offering_target_id(&[package_operand_id(package)?]))
        }
        ModelTargetInput::SpeculativeDecodingPair { target, draft } => Ok(offering_target_id(&[
            package_operand_id(target)?,
            package_operand_id(draft)?,
        ])),
    }
}

fn target_uses_only_installed_packages(target: &ModelTargetInput) -> bool {
    match target {
        ModelTargetInput::Package { package } => {
            matches!(package, ModelPackageOperand::Installed { .. })
        }
        ModelTargetInput::SpeculativeDecodingPair { target, draft } => {
            matches!(target, ModelPackageOperand::Installed { .. })
                && matches!(draft, ModelPackageOperand::Installed { .. })
        }
    }
}

impl ModelEvaluator for NativeModelEvaluator {
    fn assess(
        &self,
        request: AssessModelsRequest,
    ) -> BoxFuture<'_, Result<AssessModelsResponse, InventoryError>> {
        Box::pin(async move {
            let reserve_bytes = request
                .capacity_policy
                .required_reserve_bytes_per_memory_domain;
            let environment = self.environment(reserve_bytes).await?;
            let include_performance = request.include_performance;
            let release_catalog = Arc::clone(&self.release_catalog);
            let evaluated = futures_util::stream::iter(request.requests.into_iter().enumerate())
                .map(|(index, item)| {
                    let environment = environment.clone();
                    let release_catalog = Arc::clone(&release_catalog);
                    async move {
                        let request_id = item.request_id;
                        let target_id = match target_input_id(&item.target) {
                            Ok(target_id) => target_id,
                            Err(message) => {
                                return Ok::<_, InventoryError>((
                                    index,
                                    AssessModelResult::InvalidTarget {
                                        request_id,
                                        failure: DomainModelFailure {
                                            code: "invalid_target".to_owned(),
                                            message,
                                            retryable: false,
                                        },
                                    },
                                ));
                            }
                        };
                        let cached = self.cached_profiles(
                            &target_id,
                            &item.profiles,
                            reserve_bytes,
                            include_performance,
                            &environment,
                        )?;
                        let result = if let Some(profiles) = cached {
                            AssessModelResult::Assessed {
                                request_id,
                                target_id,
                                profiles,
                            }
                        } else {
                            let release_target_id = target_id.clone();
                            let release_target = spawn_blocking_traced(move || {
                                release_catalog.resolve_target(&release_target_id)
                            })
                            .await
                            .map_err(|error| {
                                InventoryError::Internal(format!(
                                    "release model preparation task failed for {}: {error}",
                                    target_id.0
                                ))
                            })??;
                            let resolved = match release_target {
                                Some(resolved) => Ok(resolved),
                                None if target_uses_only_installed_packages(&item.target) => {
                                    self.models.resolve_target(item.target).await
                                }
                                None => Err(InventoryError::InvalidRequest(format!(
                                    "target {} is not installed or part of the release catalog",
                                    target_id.0
                                ))),
                            };
                            match resolved {
                                Ok(resolved) => AssessModelResult::Assessed {
                                    request_id,
                                    target_id: resolved.target_id.clone(),
                                    profiles: self
                                        .assess_profiles(
                                            &resolved,
                                            &item.profiles,
                                            reserve_bytes,
                                            include_performance,
                                            &environment,
                                        )
                                        .await?,
                                },
                                Err(error) => AssessModelResult::InvalidTarget {
                                    request_id,
                                    failure: DomainModelFailure {
                                        code: "invalid_target".to_owned(),
                                        message: error.to_string(),
                                        retryable: false,
                                    },
                                },
                            }
                        };
                        Ok::<_, InventoryError>((index, result))
                    }
                })
                .buffer_unordered(MODEL_ASSESSMENT_CONCURRENCY)
                .collect::<Vec<_>>()
                .await;
            let mut results = evaluated.into_iter().collect::<Result<Vec<_>, _>>()?;
            results.sort_unstable_by_key(|(index, _)| *index);
            Ok(AssessModelsResponse {
                environment_id: environment.id,
                results: results.into_iter().map(|(_, result)| result).collect(),
            })
        })
    }

    fn fit(
        &self,
        request: FitModelsRequest,
    ) -> BoxFuture<'_, Result<FitModelsResponse, InventoryError>> {
        Box::pin(async move {
            if request.minimum_context_length == 0
                || request.minimum_context_length > request.maximum_context_length
            {
                return Err(InventoryError::InvalidRequest(
                    "fit bounds must be positive and ordered".to_owned(),
                ));
            }
            let reserve_bytes = request
                .capacity_policy
                .required_reserve_bytes_per_memory_domain;
            let environment = self.environment(reserve_bytes).await?;
            let mut results = Vec::with_capacity(request.targets.len());
            for item in &request.targets {
                let request_id = item.request_id.clone();
                match self.models.resolve_target(item.target.clone()).await {
                    Ok(resolved) => {
                        let mut result = self.fit_one(&resolved, &request, &environment).await?;
                        match &mut result {
                            FitModelResult::Fitted {
                                request_id: result_request_id,
                                ..
                            }
                            | FitModelResult::DoesNotFit {
                                request_id: result_request_id,
                                ..
                            }
                            | FitModelResult::InvalidTarget {
                                request_id: result_request_id,
                                ..
                            } => *result_request_id = request_id,
                        }
                        results.push(result);
                    }
                    Err(error) => results.push(FitModelResult::InvalidTarget {
                        request_id,
                        failure: DomainModelFailure {
                            code: "invalid_target".to_owned(),
                            message: error.to_string(),
                            retryable: false,
                        },
                    }),
                }
            }
            Ok(FitModelsResponse {
                environment_id: environment.id,
                results,
            })
        })
    }
}

fn planner_concurrency() -> usize {
    std::thread::available_parallelism().map_or(1, |cores| cores.get().clamp(1, 16))
}

fn unavailable_performance(
    code: &str,
    message: impl Into<String>,
) -> GenerationPerformanceAssessment {
    GenerationPerformanceAssessment::Unavailable {
        method: icn_hardware::GENERATION_PERFORMANCE_METHOD.to_owned(),
        code: code.to_owned(),
        message: message.into(),
    }
}

fn assess_planning_request_with_backend(
    request: PlanningWorkerRequest,
    native_backend: &NativeBackend,
) -> anyhow::Result<PlanningWorkerResponse> {
    let backend = native_backend.as_llama_backend();
    let topology = icn_contracts::MemoryTopology::from_snapshot(&request.hardware)
        .context("planning request contains an invalid memory topology")?;
    let mut plans = request
        .defaults
        .into_iter()
        .map(|defaults| {
            execution_intent(
                request.primary.clone(),
                request.projector.clone(),
                &defaults,
            )
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let calibration = if request.estimate_performance {
        Some(
            match (request.calibration, request.calibration_unavailable) {
                (Some(calibration), _) => Ok(calibration),
                (None, Some(error)) => Err(error),
                (None, None) => llama_cpp_2::model::params::fit::FitCalibration::measure(backend)
                    .map_err(|error| error.to_string()),
            },
        )
    } else {
        None
    };
    let assess_without_performance = |code: &str, message: String| {
        icn_hardware::assess_profiles_with_backend(backend, &topology, &plans).map(|assessments| {
            assessments
                .into_iter()
                .map(|hardware| ModelExecutionAssessment {
                    hardware,
                    performance: unavailable_performance(code, message.clone()),
                })
                .collect()
        })
    };
    let base = match calibration.as_ref() {
        Some(Ok(calibration)) => icn_hardware::assess_execution_profiles_with_backend(
            backend,
            &topology,
            &plans,
            calibration,
        )
        .or_else(|error| {
            assess_without_performance("performance_estimation_failed", error.to_string())
        }),
        Some(Err(calibration_error)) => {
            assess_without_performance("calibration_failed", calibration_error.clone())
        }
        None => icn_hardware::assess_profiles_with_backend(backend, &topology, &plans).map(
            |assessments| {
                assessments
                    .into_iter()
                    .map(|hardware| ModelExecutionAssessment {
                        hardware,
                        performance: GenerationPerformanceAssessment::not_requested(),
                    })
                    .collect()
            },
        ),
    }?;
    let assessments = plans
        .iter_mut()
        .zip(base)
        .map(|(plan, base)| {
            if !matches!(base.hardware, HardwareAssessment::Fits { .. }) {
                return Ok(base);
            }
            plan.mtp = icn_mtp::select_mtp_with_backend(
                backend,
                plan,
                icn_mtp::CandidatePolicy::Automatic(&request.mtp),
            )
            .context("failed to select a native MTP configuration")?;
            if matches!(plan.mtp, icn_contracts::MtpConfig::Disabled { .. }) {
                return Ok(base);
            }
            let hardware = icn_hardware::assess_with_backend(backend, &topology, plan)?.assessment;
            let performance = if matches!(hardware, HardwareAssessment::Fits { .. }) {
                // Phase 1 intentionally estimates baseline target-model decode. MTP changes fit
                // memory but is not credited with an unmeasured speculative-decoding speedup.
                base.performance
            } else {
                unavailable_performance(
                    "configuration_does_not_fit",
                    "generation performance is unavailable for a configuration that does not fit",
                )
            };
            Ok(ModelExecutionAssessment {
                hardware,
                performance,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(PlanningWorkerResponse {
        assessments,
        calibration,
    })
}

#[cfg(test)]
fn test_native_backend() -> NativeBackend {
    static BACKEND: std::sync::OnceLock<NativeBackend> = std::sync::OnceLock::new();
    BACKEND
        .get_or_init(|| NativeBackend::initialize().expect("initialize test native backend"))
        .clone()
}

#[cfg(test)]
fn run_isolated_planning(
    request: PlanningWorkerRequest,
    _worker_launcher: &NativeWorkerLauncher,
) -> anyhow::Result<PlanningWorkerResponse> {
    icn_engine::disable_native_diagnostics();
    let native_backend = test_native_backend();
    assess_planning_request_with_backend(request, &native_backend)
}

#[cfg(not(test))]
fn run_isolated_planning(
    request: PlanningWorkerRequest,
    worker_launcher: &NativeWorkerLauncher,
) -> anyhow::Result<PlanningWorkerResponse> {
    use std::io::Write as _;

    let mut child = worker_launcher.command(NativeWorkerRole::Planner)?;
    child
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = child
        .spawn()
        .context("failed to start isolated native planner")?;
    serde_json::to_writer(
        child
            .stdin
            .as_mut()
            .context("isolated native planner stdin was unavailable")?,
        &request,
    )
    .context("failed to encode isolated native planner request")?;
    child
        .stdin
        .take()
        .context("isolated native planner stdin was unavailable")?
        .flush()
        .context("failed to flush isolated native planner request")?;
    let deadline = std::time::Instant::now() + PLANNING_WORKER_TIMEOUT;
    loop {
        if child
            .try_wait()
            .context("failed to observe isolated native planner")?
            .is_some()
        {
            break;
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!("isolated native planner exceeded its time bound");
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let output = child
        .wait_with_output()
        .context("failed to await isolated native planner")?;
    if output.stdout.len() > MAX_PLANNING_WORKER_OUTPUT_BYTES
        || output.stderr.len() > MAX_PLANNING_WORKER_OUTPUT_BYTES
    {
        anyhow::bail!("isolated native planner exceeded its output bound");
    }
    if !output.status.success() {
        let diagnostic = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "isolated native planner exited with {}: {}",
            output.status,
            diagnostic.trim().chars().take(4_096).collect::<String>()
        );
    }
    serde_json::from_slice(&output.stdout)
        .context("isolated native planner returned an invalid assessment")
}

fn run_planning_worker(authority: NativeRuntimeAuthority) -> anyhow::Result<()> {
    let native_backend = initialize_native_runtime(&authority)?;
    let request = serde_json::from_reader(std::io::stdin().lock())
        .context("failed to decode native planner request")?;
    let assessment = assess_planning_request_with_backend(request, &native_backend)?;
    serde_json::to_writer(std::io::stdout().lock(), &assessment)
        .context("failed to encode native planner result")?;
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
    use std::io::Write as _;

    let mut child = worker_launcher.command(NativeWorkerRole::Template)?;
    child
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = child
        .spawn()
        .context("failed to start isolated native template worker")?;
    serde_json::to_writer(
        child
            .stdin
            .as_mut()
            .context("template worker stdin was unavailable")?,
        &request,
    )
    .context("failed to encode template worker request")?;
    child
        .stdin
        .take()
        .context("template worker stdin was unavailable")?
        .flush()
        .context("failed to flush template worker request")?;
    let deadline = std::time::Instant::now() + TEMPLATE_WORKER_TIMEOUT;
    loop {
        if child
            .try_wait()
            .context("failed to observe isolated native template worker")?
            .is_some()
        {
            break;
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!("isolated native template worker exceeded its time bound");
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let output = child
        .wait_with_output()
        .context("failed to await isolated native template worker")?;
    if output.stdout.len() > MAX_PLANNING_WORKER_OUTPUT_BYTES
        || output.stderr.len() > MAX_PLANNING_WORKER_OUTPUT_BYTES
    {
        anyhow::bail!("isolated native template worker exceeded its output bound");
    }
    if !output.status.success() {
        anyhow::bail!(
            "template worker exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
                .trim()
                .chars()
                .take(4_096)
                .collect::<String>()
        );
    }
    serde_json::from_slice(&output.stdout).context("template worker returned an invalid assessment")
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

impl InventoryHardwareAssessor for NativeHardwareAssessor {
    fn cache_key(&self, snapshot: &HardwareSnapshot) -> Result<String, InventoryError> {
        ModelHardwareAssessor::cache_key(self, None, snapshot)
    }

    fn assess(
        &self,
        resolved: ResolvedModel,
    ) -> BoxFuture<'_, Result<HardwareAssessment, InventoryError>> {
        Box::pin(self.assess_resolved(resolved, None))
    }

    fn assess_serving(
        &self,
        resolved: ResolvedModel,
        profile: icn_contracts::ServingProfile,
    ) -> BoxFuture<'_, Result<HardwareAssessment, InventoryError>> {
        Box::pin(async move {
            self.assess_resolved(
                resolved,
                Some(&ModelPreviewProfile {
                    id: "serving".to_owned(),
                    context_length: profile.context_length,
                    parallel_sequences: 1,
                }),
            )
            .await
        })
    }
}

impl ModelHardwareAssessor for NativeHardwareAssessor {
    fn cache_key(
        &self,
        profile: Option<&ModelPreviewProfile>,
        snapshot: &HardwareSnapshot,
    ) -> Result<String, InventoryError> {
        self.assessment_cache_key(profile, snapshot)
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

impl HardwareProvider for NativeHardwareAssessor {
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
            let enabled_backends = build_identity::enabled_backends()
                .into_iter()
                .map(str::to_owned)
                .collect();
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
            let mut calibration = self.calibration.lock().await;
            let evidence = calibration_evidence(&snapshot)?;
            if calibration.evidence.as_deref() != Some(evidence.as_str()) {
                calibration.topology_fingerprint = Some(snapshot.topology_fingerprint.clone());
                calibration.evidence = Some(evidence.clone());
                calibration.result = self
                    .cache
                    .as_ref()
                    .and_then(|cache| {
                        cache.read_index::<PersistedCalibration>(
                            ModelIndexKind::Calibration,
                            &evidence,
                        )
                    })
                    .filter(|cached| {
                        unix_time_seconds().saturating_sub(cached.captured_at)
                            <= CALIBRATION_MAX_AGE_SECONDS
                    })
                    .map(|cached| Ok(cached.calibration));
            }
            Ok(snapshot)
        })
    }
}

#[derive(Clone)]
struct NativeModelInstanceController {
    inventory: Arc<ModelManager>,
    assessor: Arc<NativeHardwareAssessor>,
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
    idle_timeout: std::time::Duration,
}

#[derive(Clone)]
struct OwnedInferenceWorker {
    worker: InferenceWorker,
}

#[derive(Clone)]
struct ModelOperationFailure {
    code: String,
    message: String,
    retryable: bool,
}

impl ModelOperationFailure {
    fn new(code: impl Into<String>, message: impl Into<String>, retryable: bool) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            retryable,
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
    timeout: std::time::Duration,
    now: std::time::Instant,
) -> Option<std::time::Duration> {
    let current = current?;
    if current.generation != expected.generation
        || current.instance_id != expected.instance_id
        || activity.generation != expected.generation
        || activity.active_leases != 0
    {
        return None;
    }
    let elapsed = now.checked_duration_since(activity.idle_since?)?;
    (elapsed >= timeout).then_some(elapsed)
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
        let message = error.to_string();
        Self::new(ModelOperationFailure::new(
            "model_transition_failed",
            message,
            true,
        ))
    }
}

impl NativeModelInstanceController {
    const IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10 * 60);

    fn load_failure(error: InventoryError) -> DomainModelFailure {
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

    fn new(
        inventory: Arc<ModelManager>,
        assessor: Arc<NativeHardwareAssessor>,
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
            idle_timeout: Self::IDLE_TIMEOUT,
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
        failure: DomainModelFailure,
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
                let deadline = tokio::time::Instant::from_std(idle_since + controller.idle_timeout);
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
                let Some(elapsed) = idle_release_elapsed(
                    &ready_instance,
                    current.as_ref(),
                    activity,
                    controller.idle_timeout,
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
            MtpCandidateSelection,
            Vec<ModelPackageId>,
        ),
        InventoryError,
    > {
        let (target, package_ids) = match &configuration.target {
            DomainModelOfferingTarget::Package { package } => (
                ModelTargetInput::Package {
                    package: ModelPackageOperand::Installed {
                        package_id: package.id.clone(),
                    },
                },
                vec![package.id.clone()],
            ),
            DomainModelOfferingTarget::SpeculativeDecodingPair { target, draft, .. } => (
                ModelTargetInput::SpeculativeDecodingPair {
                    target: ModelPackageOperand::Installed {
                        package_id: target.id.clone(),
                    },
                    draft: ModelPackageOperand::Installed {
                        package_id: draft.id.clone(),
                    },
                },
                vec![target.id.clone(), draft.id.clone()],
            ),
        };
        let resolved = self.inventory.resolve_target(target).await?;
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
        let mtp = model
            .components
            .iter()
            .filter(|component| matches!(component.role, ComponentRole::Mtp | ComponentRole::Draft))
            .map(|component| component.path.clone())
            .collect();
        let defaults = self.profile_defaults(&ModelExecutionProfile {
            context_length: configuration.profile.context_length,
        })?;
        let plan = execution_intent(primary, projector, &defaults).map_err(|error| {
            InventoryError::Internal(format!(
                "failed to resolve model execution intent: {error:#}"
            ))
        })?;
        Ok((
            model,
            plan,
            MtpCandidateSelection::Automatic(mtp),
            package_ids,
        ))
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
            icn_hardware::system_memory_thresholds(hardware.system_memory.total_bytes)
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
            match assessment.hardware {
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
                        failure: DomainModelFailure {
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
                                memory.available_bytes = sample.available_bytes,
                                memory.reserve_bytes = sample.abort_reserve_bytes(),
                                memory.available_commit_bytes = ?sample.available_commit_bytes,
                                memory.commit_limit_bytes = ?sample.commit_limit_bytes,
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
            return Err(ModelTransitionFailure::new(ModelOperationFailure::new(
                LOW_MEMORY_FAILURE_CODE,
                format!(
                    "model requires at least {minimum_required} bytes of system memory at parallelism 1, but only {} bytes are available with a {} byte system reserve",
                    sample.available_bytes,
                    sample.abort_reserve_bytes()
                ),
                true,
            )));
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
        mtp_selection: MtpCandidateSelection,
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
        if let Err(error) = worker.start_load(model_id.clone(), plan, mtp_selection, hardware) {
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
        let observation_result = spawn_blocking_traced(move || {
            observation_backend.observe_model_instance(
                CapacityPolicy::default(),
                build_identity::native_build(),
                build_identity::enabled_backends()
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
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
                    code: failure.event.code,
                    message: failure.event.message,
                    retryable: failure.event.retryable,
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
                    let _ = events.send(ModelLoadEvent::Failed { failure });
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
                let (resolved, plan, mtp_selection, package_ids) = match controller
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
                            let failure = Self::load_failure(error);
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
                        mtp_selection,
                        package_ids,
                        events.clone(),
                        instance_id.clone(),
                        Arc::clone(&stop_requested),
                        &model_mutation,
                    )
                    .await
                {
                    Ok(allocation) => allocation,
                    Err(failure) if failure.event.code == "model_instance_stopped" => {
                        controller
                            .publish_stopped_loading(&instance_id, &configuration_id)
                            .await;
                        send_stopped();
                        return;
                    }
                    Err(failure) => {
                        let failure = DomainModelFailure {
                            code: failure.event.code.to_owned(),
                            message: failure.event.message,
                            retryable: failure.event.retryable,
                        };
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
                            DomainModelFailure {
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

async fn generate_release_catalog(
    output: PathBuf,
    model_store: PathBuf,
    cache_root: PathBuf,
    hf_caches: Vec<PathBuf>,
) -> anyhow::Result<()> {
    let mut config = InventoryConfig::with_roots(model_store, cache_root)
        .context("invalid catalog generation inventory configuration")?;
    config.hf_cache_dirs.extend(hf_caches);
    let runtime_authority = NativeRuntimeAuthority::development();
    let native_backend = initialize_native_runtime(&runtime_authority)
        .context("failed to initialize the native backend for catalog generation")?;
    let worker_launcher = NativeWorkerLauncher::new(runtime_authority);
    let inventory = Arc::new(
        ModelManager::open_with_template_assessor(
            config,
            Some(Arc::new(NativeTemplateAssessor {
                worker_launcher: worker_launcher.clone(),
            })),
        )
        .await
        .context("failed to initialize catalog generation inventory")?,
    );
    let (assessor, _) = native_assessor_services(
        &inventory,
        native_backend,
        model_plan_defaults(),
        worker_launcher,
    );
    inventory
        .set_hardware_assessor(assessor.clone())
        .context("failed to configure catalog generation hardware assessment")?;
    let repositories = Arc::new(ModelPreviewService::new(
        inventory.clone(),
        assessor.clone(),
    ));
    let resolver = ResolvingRecommendableCatalog::new(inventory, repositories);
    let generated = resolver
        .resolve_release_catalog()
        .await
        .context("failed to resolve the curated model catalog")?;
    verify_compact_planner_parity(&generated, assessor.as_ref())
        .await
        .context("compact planner inputs changed native planning results")?;
    let planner_bundle = generated
        .encode_planner_bundle()
        .context("failed to encode release planner inputs")?;
    let manifest = release_catalog_manifest(
        &generated,
        native_template_identity(),
        native_planner_identity(),
        &planner_bundle,
    )
    .context("refusing to publish an incomplete release catalog")?;
    let encoded =
        serde_json::to_vec_pretty(&manifest).context("failed to encode the release catalog")?;
    let parent = output
        .parent()
        .context("catalog output must have a parent directory")?;
    tokio::fs::create_dir_all(parent)
        .await
        .context("failed to create the catalog output directory")?;
    let temporary = output.with_extension("json.tmp");
    tokio::fs::write(&temporary, [&encoded[..], b"\n"].concat())
        .await
        .context("failed to write the generated catalog")?;
    tokio::fs::rename(&temporary, &output)
        .await
        .context("failed to publish the generated catalog")?;
    println!("generated {}", output.display());
    Ok(())
}

async fn verify_compact_planner_parity(
    generated: &icn_models::GeneratedReleaseCatalog,
    assessor: &NativeHardwareAssessor,
) -> anyhow::Result<()> {
    for model in &generated.catalog.models {
        let profiles = model
            .eligible_serving_profiles
            .iter()
            .enumerate()
            .map(|(index, profile)| ModelPreviewProfile {
                id: format!("release-{index}"),
                context_length: profile.context_length,
                parallel_sequences: 1,
            })
            .collect::<Vec<_>>();
        let source = generated
            .resolve_source_planner_target(&model.target_id)
            .with_context(|| format!("failed to materialize source input for {}", model.id.0))?;
        let compact = generated
            .resolve_compact_planner_target(&model.target_id)
            .with_context(|| format!("failed to materialize compact input for {}", model.id.0))?;
        let source_assessments = assessor
            .assess_resolved_profiles(source.target_model.clone(), profiles.clone())
            .await
            .with_context(|| format!("failed to plan source input for {}", model.id.0))?;
        let compact_assessments = assessor
            .assess_resolved_profiles(compact.target_model.clone(), profiles)
            .await
            .with_context(|| format!("failed to plan compact input for {}", model.id.0))?;
        if compact_assessments != source_assessments {
            anyhow::bail!(
                "native planning parity failed for catalog target {}",
                model.id.0
            );
        }
    }
    Ok(())
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
    load_release_catalog(
        &installation.catalog_lock(),
        &installation.planner_bundle(),
        native_template_identity(),
        &native_planner_identity(),
    )
    .context("failed to load release catalog sidecars")
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
    let _telemetry = telemetry::init(matches!(&cli.command, Command::Serve { .. }))?;
    // Native planner diagnostics are extremely verbose and can dominate metadata-only fitting.
    // ICN emits bounded, structured operation telemetry at the service boundary instead.
    icn_engine::disable_native_diagnostics();
    match cli.command {
        Command::Serve {
            bind,
            instance_id,
            parent_pid,
            auth_token,
            fake,
            model_store,
            cache_root,
            model_sources,
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
            inventory_config.model_sources.extend(model_sources);
            inventory_config.hf_cache_dirs.extend(hf_caches);
            let plan_defaults = model_plan_defaults();
            let native_backend = initialize_native_runtime(&runtime_authority)?;
            let inventory = Arc::new(
                ModelManager::open_with_template_assessor(
                    inventory_config,
                    Some(Arc::new(NativeTemplateAssessor {
                        worker_launcher: worker_launcher.clone(),
                    })),
                )
                .await
                .context("failed to initialize model inventory")?,
            );
            let (inventory_hardware_assessor, native_executor_slot) = native_assessor_services(
                &inventory,
                native_backend.clone(),
                plan_defaults.clone(),
                worker_launcher.clone(),
            );
            inventory
                .set_hardware_assessor(inventory_hardware_assessor.clone())
                .context("failed to configure inventory hardware assessment")?;
            let release_catalog = installation
                .as_ref()
                .map(open_installation_catalog)
                .transpose()?
                .map(Arc::new);
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
                    inventory_hardware_assessor.clone(),
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
            .with_hardware(inventory_hardware_assessor.clone())
            .with_model_downloads(model_downloads)
            .with_identity(identity);
            if let Some(release_catalog) = release_catalog {
                state = state
                    .with_model_evaluator(Arc::new(NativeModelEvaluator::new(
                        inventory,
                        inventory_hardware_assessor,
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
                .with_graceful_shutdown(shutdown_signal(parent_pid))
                .await;
            serve_result?;
            tracing::info!("ICN server stopped");
        }
        Command::Catalog { command } => match command {
            CatalogCommand::Generate {
                output,
                model_store,
                cache_root,
                hf_caches,
            } => generate_release_catalog(output, model_store, cache_root, hf_caches).await?,
            CatalogCommand::Check { installation } => {
                let installation = installation::Installation::load(&installation)
                    .context("invalid ICN installation")?;
                let catalog = open_installation_catalog(&installation)?;
                println!(
                    "validated {} release catalog models",
                    catalog.catalog().models.len()
                );
            }
        },
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
        Command::PlanWorker { runtime } => run_planning_worker(runtime.authority()?)?,
        Command::TemplateWorker { runtime } => run_template_worker(runtime.authority()?)?,
        Command::InferenceWorker { runtime } => {
            let authority = runtime.authority()?;
            let native_backend = initialize_native_runtime(&authority)?;
            inference_worker::run_worker(build_identity::native_build(), native_backend)?
        }
    }
    Ok(())
}

async fn shutdown_signal(parent_pid: Option<u32>) {
    tokio::select! {
        _ = interrupt_signal() => {},
        _ = parent_watchdog(parent_pid), if parent_pid.is_some() => {},
        _ = parent_stdin_eof(), if parent_pid.is_some() => {},
    }
}

async fn parent_stdin_eof() {
    // Tokio implements stdin reads on its blocking pool. A pending read then
    // prevents Runtime::drop from completing during an ordinary SIGTERM while
    // the parent still owns the pipe, creating a parent/child shutdown cycle.
    // A detached OS thread has the desired semantics: EOF wakes the async
    // watchdog after abrupt parent death, while orderly process exit does not
    // wait for the read to finish.
    let (eof, observed) = tokio::sync::oneshot::channel();
    std::thread::spawn(move || {
        use std::io::Read as _;

        let mut stdin = std::io::stdin().lock();
        let mut buffer = [0_u8; 1];
        loop {
            match stdin.read(&mut buffer) {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
        }
        let _ = eof.send(());
    });
    let _ = observed.await;
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

async fn parent_watchdog(parent_pid: Option<u32>) {
    let Some(parent_pid) = parent_pid else {
        std::future::pending::<()>().await;
        return;
    };
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
    loop {
        interval.tick().await;
        if !process_exists(parent_pid) {
            return;
        }
    }
}

#[cfg(unix)]
fn process_exists(pid: u32) -> bool {
    // Signal zero performs an existence/permission check without delivering a signal.
    let result = unsafe { libc::kill(pid as libc::pid_t, 0) };
    result == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(not(unix))]
fn process_exists(_pid: u32) -> bool {
    true
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
                failure: DomainModelFailure {
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
        let (evidence, unavailable) = performance_result(
            GenerationPerformanceAssessment::Estimated {
                method: "native".to_owned(),
                confidence: icn_contracts::GenerationPerformanceConfidence::Moderate,
                workload: "baseline_single_sequence_decode".to_owned(),
                always_active_weight_bytes: 10,
                routed_expert_weight_bytes: 80,
                expert_count: 8,
                expert_used_count: 2,
                cross_memory_domain_placement: true,
                points: vec![
                    icn_contracts::GenerationSpeedPoint {
                        context_tokens: 100_000,
                        kv_bytes_read_per_token: 4_096,
                        lower_tokens_per_second: 20.0,
                        expected_tokens_per_second: 24.0,
                        upper_tokens_per_second: 28.0,
                    },
                    icn_contracts::GenerationSpeedPoint {
                        context_tokens: 200_000,
                        kv_bytes_read_per_token: 8_192,
                        lower_tokens_per_second: 15.0,
                        expected_tokens_per_second: 18.0,
                        upper_tokens_per_second: 21.0,
                    },
                ],
            },
            100_000,
        );
        let evidence = evidence.expect("matching performance evidence");

        assert!(unavailable.is_none());
        assert_eq!(evidence.context_tokens, 100_000);
        assert_eq!(evidence.lower_tokens_per_second, 20.0);
        assert_eq!(evidence.estimated_tokens_per_second, 24.0);
        assert_eq!(evidence.upper_tokens_per_second, 28.0);
        assert_eq!(evidence.confidence, PerformanceConfidence::Moderate);
        assert_eq!(evidence.method, "native");
    }

    #[test]
    fn performance_result_preserves_typed_unavailable_evidence() {
        let (evidence, unavailable) = performance_result(
            GenerationPerformanceAssessment::Unavailable {
                method: "native_decode_v3".to_owned(),
                code: "calibration_coverage_missing".to_owned(),
                message: "no routed calibration covers backend CUDA device GPU0".to_owned(),
            },
            100_000,
        );

        assert!(evidence.is_none());
        assert_eq!(
            unavailable,
            Some(PerformanceUnavailable {
                method: "native_decode_v3".to_owned(),
                code: "calibration_coverage_missing".to_owned(),
                message: "no routed calibration covers backend CUDA device GPU0".to_owned(),
            })
        );
    }

    #[test]
    fn native_template_cache_identity_tracks_both_native_pins() {
        let assessor = NativeTemplateAssessor {
            worker_launcher: NativeWorkerLauncher::development(),
        };
        let identity = assessor.cache_identity();

        assert!(identity.contains(build_identity::BINDINGS_REVISION));
        assert!(identity.contains(build_identity::NATIVE_BACKEND_REVISION));
    }

    fn parity_test_defaults() -> ModelPlanDefaults {
        ModelPlanDefaults {
            context_size: 128,
            physical_context_size: 128,
            batch_size: 128,
            ubatch_size: 64,
            max_sequences: 1,
            prefill_quantum: 128,
            execution: ExecutionConfig {
                kv_unified: false,
                ..ExecutionConfig::default()
            },
            projector_use_gpu: true,
            projector_warmup: true,
            image_min_tokens: None,
            image_max_tokens: None,
        }
    }

    #[test]
    fn preview_parallelism_reserves_one_full_context_partition_per_sequence() {
        let assessor = NativeHardwareAssessor {
            defaults: parity_test_defaults(),
            cache: None,
            native_backend: test_native_backend(),
            worker_launcher: NativeWorkerLauncher::development(),
            native_executor: Arc::new(RwLock::new(None)),
            gate: Arc::new(tokio::sync::Mutex::new(())),
            assessment_work_gates: Arc::new(tokio::sync::Mutex::new(
                std::collections::BTreeMap::new(),
            )),
            planning_slots: Arc::new(tokio::sync::Semaphore::new(1)),
            calibration: Arc::new(tokio::sync::Mutex::new(CalibrationCache::default())),
        };

        let defaults = assessor.effective_defaults(Some(&ModelPreviewProfile {
            id: "p4".to_owned(),
            context_length: 32_768,
            parallel_sequences: 4,
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
            total_bytes: 16 * gib,
            available_bytes: 7 * gib,
            commit_limit_bytes: None,
            available_commit_bytes: None,
        };

        assert_eq!(
            select_model_allocation(&[(1, gib), (2, 4 * gib), (3, 6 * gib)], sample),
            Some((2, 4 * gib))
        );
    }

    #[test]
    fn replacement_preview_credits_only_memory_the_current_residency_will_release() {
        let gib = 1024 * 1024 * 1024;
        let sample = memory_supervisor::MemorySample {
            captured_at: std::time::Instant::now(),
            total_bytes: 16 * gib,
            available_bytes: 3 * gib,
            commit_limit_bytes: Some(20 * gib),
            available_commit_bytes: Some(4 * gib),
        };

        assert_eq!(select_model_allocation(&[(1, 4 * gib)], sample), None);
        let credited = credit_replaced_instance_memory(sample, 3 * gib);
        assert_eq!(credited.available_bytes, 6 * gib);
        assert_eq!(credited.available_commit_bytes, Some(7 * gib));
        assert_eq!(
            select_model_allocation(&[(1, 4 * gib)], credited),
            Some((1, 4 * gib))
        );
    }

    #[test]
    fn idle_release_requires_the_exact_residency_and_a_full_idle_interval() {
        let timeout = std::time::Duration::from_secs(600);
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
            idle_release_elapsed(&resident, Some(&resident), activity, timeout, now),
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
                timeout,
                now,
            ),
            None
        );
    }

    #[test]
    fn idle_release_rejects_active_or_stale_observations() {
        let timeout = std::time::Duration::from_secs(600);
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
                timeout,
                now,
            ),
            None
        );
        assert_eq!(
            idle_release_elapsed(&resident, Some(&stale), idle, timeout, now),
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
                timeout,
                now,
            ),
            None
        );
    }

    #[test]
    fn available_and_preview_cache_keys_share_resolved_profile_identity() {
        let assessor = NativeHardwareAssessor {
            defaults: parity_test_defaults(),
            cache: None,
            native_backend: test_native_backend(),
            worker_launcher: NativeWorkerLauncher::development(),
            native_executor: Arc::new(RwLock::new(None)),
            gate: Arc::new(tokio::sync::Mutex::new(())),
            assessment_work_gates: Arc::new(tokio::sync::Mutex::new(
                std::collections::BTreeMap::new(),
            )),
            planning_slots: Arc::new(tokio::sync::Semaphore::new(1)),
            calibration: Arc::new(tokio::sync::Mutex::new(CalibrationCache::default())),
        };
        let snapshot = HardwareSnapshot {
            captured_at: 1,
            platform: "test".to_owned(),
            architecture: "test".to_owned(),
            system_product_name: None,
            cpu_model: None,
            logical_cores: 1,
            system_memory: icn_contracts::HardwareSystemMemory {
                total_bytes: 10,
                current_available_bytes: 10,
                warning_reserve_bytes: 0,
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
            id: "caller-correlation-does-not-affect-fit".to_owned(),
            context_length: 128,
            parallel_sequences: 1,
        };
        assert_eq!(
            assessor.assessment_cache_key(None, &snapshot).unwrap(),
            assessor
                .assessment_cache_key(Some(&equivalent_preview), &snapshot)
                .unwrap()
        );
        assert_ne!(
            assessor.assessment_cache_key(None, &snapshot).unwrap(),
            assessor
                .assessment_cache_key(
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
            .current_available_bytes = 0;
        assert_eq!(
            assessor
                .assessment_cache_key(Some(&equivalent_preview), &snapshot)
                .unwrap(),
            assessor
                .assessment_cache_key(Some(&equivalent_preview), &availability_only_change)
                .unwrap()
        );
        assert_ne!(
            assessor
                .assessment_cache_key_with_policy(
                    Some(&equivalent_preview),
                    &snapshot,
                    CapacityPolicy::default(),
                )
                .unwrap(),
            assessor
                .assessment_cache_key_with_policy(
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

    /// This exercises the exact native assessor used by both inventory and preview models. The
    /// verified parity fixtures are optional in ordinary source checkouts, but CI/dev environments
    /// that stage them exercise both a tiny dense model and a production-scale MoE model.
    #[tokio::test]
    async fn available_and_sparse_preview_artifacts_have_identical_fit_assessments() {
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

        let assessor = Arc::new(NativeHardwareAssessor {
            defaults: parity_test_defaults(),
            cache: None,
            native_backend: test_native_backend(),
            worker_launcher: NativeWorkerLauncher::development(),
            native_executor: Arc::new(RwLock::new(None)),
            gate: Arc::new(tokio::sync::Mutex::new(())),
            assessment_work_gates: Arc::new(tokio::sync::Mutex::new(
                std::collections::BTreeMap::new(),
            )),
            planning_slots: Arc::new(tokio::sync::Semaphore::new(1)),
            calibration: Arc::new(tokio::sync::Mutex::new(CalibrationCache::default())),
        });
        let profile = ModelPreviewProfile {
            id: "parity".to_owned(),
            context_length: 128,
            parallel_sequences: 1,
        };

        for fixture in fixtures {
            let store = tempfile::tempdir().expect("temporary model store");
            let mut config = InventoryConfig::with_roots(
                store.path().join("inventory"),
                store.path().join("cache"),
            )
            .expect("inventory config");
            config.model_sources = vec![fixture.parent().expect("fixture parent").to_path_buf()];
            config.hf_cache_dirs.clear();
            let manager = ModelManager::open_with_template_assessor(
                config,
                Some(Arc::new(NativeTemplateAssessor {
                    worker_launcher: NativeWorkerLauncher::development(),
                })),
            )
            .await
            .expect("open inventory");
            manager
                .set_hardware_assessor(assessor.clone())
                .expect("configure inventory assessor");
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
            let inventory_assessment = model.hardware.clone();
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

            let default_preview_assessment = assessor
                .assess_resolved(preview.clone(), None)
                .await
                .expect("assess sparse preview with inventory defaults");
            assert_eq!(
                default_preview_assessment,
                inventory_assessment,
                "the inventory and preview paths diverged for {}",
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
                "preview and available fitting diverged for {}",
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
    fn version_json_reports_native_and_build_provenance() {
        let value = build_identity::identity();
        assert_eq!(value.native_build, build_identity::native_build());
        assert_eq!(value.target, build_identity::TARGET);
        assert_eq!(value.profile, build_identity::PROFILE);
        assert!(!value.backends.is_empty());
    }
}
