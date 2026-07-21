use std::net::SocketAddr;
use std::num::NonZeroU32;
use std::path::PathBuf;
#[cfg(not(test))]
use std::process::{Command as ProcessCommand, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use anyhow::Context;
use clap::{Parser, Subcommand, ValueEnum};
use futures_util::{
    future::BoxFuture,
    stream::{BoxStream, StreamExt},
};
use icn_api::{
    AppState, BackendRegistry, FakeBackend, LoadRuntimeModelRequest, RuntimeController,
    RuntimeExecutionProfile, RuntimeLoadStage, RuntimeModelEvent, RuntimeStateResponse,
    RuntimeStatus, ServerIdentity, app,
};
use icn_contracts::{
    CacheType, CompletionBackend, ComponentRole, ExecutionConfig, ExecutionIntent, FlashAttention,
    GenerationPerformanceAssessment, GpuLayers, HardwareAssessment, HardwareProvider,
    HardwareSnapshot, InventoryError, InventoryHardwareAssessor, LoadStage,
    ModelExecutionAssessment, ModelHardwareAssessor, ModelId, ModelInventory, ModelPreviewProfile,
    ModelStatus, ProjectorConfig, ResolvedModel, SplitMode, TemplateAssessment, TemplateAssessor,
};
use icn_engine::LlamaCompletionBackend;
use icn_hardware::{CapacityPolicy, assess as assess_hardware, discover, discover_hardware};
use icn_models::{InventoryConfig, ModelManager, ModelPreviewService};
use tokio_stream::wrappers::ReceiverStream;
use tower_http::trace::{DefaultOnResponse, TraceLayer};
use tracing::Instrument as _;

mod build_identity;
mod telemetry;

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

#[derive(Debug, Clone, Copy, ValueEnum)]
enum FlashAttentionArg {
    Auto,
    Off,
    On,
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
        #[arg(long, conflicts_with_all = ["model", "model_id"])]
        fake: bool,
        #[arg(long, conflicts_with = "model_id")]
        model: Option<PathBuf>,
        /// Load a ready inventory model by its stable ID.
        #[arg(long, conflicts_with = "model")]
        model_id: Option<String>,
        /// Multimodal projector GGUF paired with the loaded text model.
        #[arg(long, requires = "model")]
        mmproj: Option<PathBuf>,
        /// Explicit separate MTP GGUF. Bundled MTP is selected natively before this override.
        #[arg(long, requires = "model")]
        mtp_model: Option<PathBuf>,
        #[arg(long, requires = "mmproj")]
        no_mmproj_offload: bool,
        #[arg(long, requires = "mmproj")]
        no_mmproj_warmup: bool,
        #[arg(long, requires = "mmproj")]
        image_min_tokens: Option<NonZeroU32>,
        #[arg(long, requires = "mmproj")]
        image_max_tokens: Option<NonZeroU32>,
        #[arg(long)]
        model_alias: Option<String>,
        /// Magnitude-owned model inventory and Hugging Face cache root.
        #[arg(long, visible_alias = "models-dir")]
        model_store: Option<PathBuf>,
        /// Additional read-only directories containing GGUF models.
        #[arg(long = "model-source")]
        model_sources: Vec<PathBuf>,
        /// Additional read-only Hugging Face hub cache roots.
        #[arg(long = "hf-cache", visible_alias = "hf-cache-dir")]
        hf_caches: Vec<PathBuf>,
        #[arg(long, default_value_t = 4096)]
        context_size: u32,
        #[arg(long, default_value_t = 512)]
        batch_size: u32,
        #[arg(long, default_value_t = 512)]
        ubatch_size: u32,
        #[arg(long, default_value_t = 1)]
        max_sequences: u32,
        #[arg(long)]
        prefill_quantum: Option<u32>,
        /// GPU layers: `auto` runs pinned common/fit, `all` fully offloads, or use a count.
        #[arg(long, default_value = "auto")]
        gpu_layers: GpuLayers,
        /// Disable model memory mapping (enabled by the native service profile).
        #[arg(long)]
        no_mmap: bool,
        /// Keep mapped model pages resident in RAM.
        #[arg(long)]
        mlock: bool,
        #[arg(long, default_value = "layer")]
        split_mode: SplitMode,
        /// Comma-separated per-device model placement proportions.
        #[arg(long)]
        tensor_split: Option<TensorSplitArg>,
        #[arg(long, default_value = "f16")]
        cache_type_k: CacheType,
        #[arg(long, default_value = "f16")]
        cache_type_v: CacheType,
        #[arg(long)]
        no_kv_offload: bool,
        #[arg(long)]
        no_op_offload: bool,
        #[arg(long)]
        swa_full: bool,
        #[arg(long)]
        kv_unified: bool,
        #[arg(long)]
        threads: Option<NonZeroU32>,
        #[arg(long)]
        threads_batch: Option<NonZeroU32>,
        #[arg(long, value_enum, default_value_t = FlashAttentionArg::Auto)]
        flash_attention: FlashAttentionArg,
    },
    Doctor,
    Version {
        #[arg(long)]
        json: bool,
    },
    #[command(hide = true)]
    PlanWorker,
    #[command(hide = true)]
    TemplateWorker,
}

#[derive(Debug, Clone)]
struct TensorSplitArg(Vec<f32>);

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct RuntimePlanDefaults {
    context_size: u32,
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

#[derive(Debug)]
enum MtpSelection {
    Automatic(Vec<PathBuf>),
    Explicit(PathBuf),
}

fn select_mtp(plan: &mut ExecutionIntent, mtp_selection: MtpSelection) -> anyhow::Result<()> {
    let candidates = match &mtp_selection {
        MtpSelection::Automatic(paths) => icn_mtp::CandidatePolicy::Automatic(paths),
        MtpSelection::Explicit(path) => icn_mtp::CandidatePolicy::Explicit(path),
    };
    plan.mtp = icn_mtp::select_mtp(plan, candidates)
        .context("failed to select a native MTP configuration")?;
    Ok(())
}

fn execution_intent(
    model_path: PathBuf,
    projector_path: Option<PathBuf>,
    defaults: &RuntimePlanDefaults,
) -> anyhow::Result<ExecutionIntent> {
    Ok(ExecutionIntent {
        model_path,
        context_size: defaults.context_size,
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

fn load_execution_intent(
    model_path: PathBuf,
    projector_path: Option<PathBuf>,
    mtp_selection: MtpSelection,
    defaults: &RuntimePlanDefaults,
) -> anyhow::Result<ExecutionIntent> {
    let mut intent = execution_intent(model_path, projector_path, defaults)?;
    match assess_hardware(&intent, CapacityPolicy::default())?.assessment {
        HardwareAssessment::InvalidArtifact { code, message } => {
            anyhow::bail!("invalid artifact ({code}): {message}")
        }
        HardwareAssessment::IncompatibleArtifact { code, message } => {
            anyhow::bail!("incompatible artifact ({code}): {message}")
        }
        HardwareAssessment::DoesNotFit { .. } | HardwareAssessment::NotAssessed { .. } => {
            return Ok(intent);
        }
        HardwareAssessment::Fits { .. } => {}
    }
    select_mtp(&mut intent, mtp_selection)?;
    Ok(intent)
}

struct NativeHardwareAssessor {
    defaults: RuntimePlanDefaults,
    native_executor: Arc<RwLock<Option<Arc<LlamaCompletionBackend>>>>,
    gate: tokio::sync::Mutex<()>,
    planning_slots: Arc<tokio::sync::Semaphore>,
    calibration: tokio::sync::Mutex<CalibrationCache>,
}

#[derive(Default)]
struct CalibrationCache {
    topology_fingerprint: Option<String>,
    result: Option<Result<llama_cpp_2::model::params::fit::FitCalibration, String>>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct PlanningWorkerRequest {
    primary: PathBuf,
    projector: Option<PathBuf>,
    mtp: Vec<PathBuf>,
    defaults: Vec<RuntimePlanDefaults>,
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

#[derive(Debug, Default)]
struct NativeTemplateAssessor;

impl TemplateAssessor for NativeTemplateAssessor {
    fn cache_identity(&self) -> &str {
        concat!("icn-native-model-template:", env!("CARGO_PKG_VERSION"))
    }

    fn assess(
        &self,
        inputs: &icn_contracts::EffectiveTemplateInputs,
    ) -> Result<TemplateAssessment, String> {
        run_isolated_template_inspection(TemplateWorkerRequest {
            model_path: inputs.model_path.clone(),
        })
        .map_err(|error| format!("{error:#}"))
    }
}

const ASSESSMENT_PROFILE_RESOLVER: &str = "icn-backend-plan-v1";
#[cfg(not(test))]
const PLANNING_WORKER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);
#[cfg(not(test))]
const MAX_PLANNING_WORKER_OUTPUT_BYTES: usize = 1024 * 1024;
#[cfg(not(test))]
const TEMPLATE_WORKER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

impl NativeHardwareAssessor {
    fn effective_defaults(&self, profile: Option<&ModelPreviewProfile>) -> RuntimePlanDefaults {
        let mut defaults = self.defaults.clone();
        if let Some(profile) = profile {
            defaults.context_size = profile.context_length;
            defaults.max_sequences = profile.parallel_sequences;
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
        let response = match spawn_blocking_traced(move || {
            let _permit = permit;
            run_isolated_planning(request)
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
            guard.result = Some(calibration);
        }
        Ok(response.assessments)
    }

    fn assessment_cache_key(
        &self,
        profile: Option<&ModelPreviewProfile>,
        snapshot: &HardwareSnapshot,
    ) -> Result<String, InventoryError> {
        if profile.is_some_and(|profile| profile.policy != ASSESSMENT_PROFILE_RESOLVER) {
            return Err(InventoryError::InvalidRequest(format!(
                "unsupported model assessment policy; expected {ASSESSMENT_PROFILE_RESOLVER}"
            )));
        }
        serde_json::to_string(&(
            ASSESSMENT_PROFILE_RESOLVER,
            icn_hardware::GENERATION_PERFORMANCE_METHOD,
            llama_cpp_2::model::params::fit::FIT_DECODE_WORKLOAD_METHOD,
            llama_cpp_2::model::params::fit::FIT_CALIBRATION_METHOD,
            &snapshot.native_build,
            &snapshot.enabled_backends,
            &snapshot.topology_fingerprint,
            &snapshot.capacity_policy,
            self.effective_defaults(profile),
        ))
        .map_err(|error| InventoryError::Internal(error.to_string()))
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

fn assess_planning_request(
    request: PlanningWorkerRequest,
) -> anyhow::Result<PlanningWorkerResponse> {
    let backend = llama_cpp_2::llama_backend::LlamaBackend::init()?;
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
                (None, None) => llama_cpp_2::model::params::fit::FitCalibration::measure(&backend)
                    .map_err(|error| error.to_string()),
            },
        )
    } else {
        None
    };
    let assess_without_performance = |code: &str, message: String| {
        icn_hardware::assess_profiles_with_backend(&backend, &plans, CapacityPolicy::default()).map(
            |assessments| {
                assessments
                    .into_iter()
                    .map(|hardware| ModelExecutionAssessment {
                        hardware,
                        performance: unavailable_performance(code, message.clone()),
                    })
                    .collect()
            },
        )
    };
    let base = match calibration.as_ref() {
        Some(Ok(calibration)) => icn_hardware::assess_execution_profiles_with_backend(
            &backend,
            &plans,
            CapacityPolicy::default(),
            calibration,
        )
        .or_else(|error| {
            assess_without_performance("performance_estimation_failed", error.to_string())
        }),
        Some(Err(calibration_error)) => {
            assess_without_performance("calibration_failed", calibration_error.clone())
        }
        None => {
            icn_hardware::assess_profiles_with_backend(&backend, &plans, CapacityPolicy::default())
                .map(|assessments| {
                    assessments
                        .into_iter()
                        .map(|hardware| ModelExecutionAssessment {
                            hardware,
                            performance: GenerationPerformanceAssessment::not_requested(),
                        })
                        .collect()
                })
        }
    }?;
    let assessments = plans
        .iter_mut()
        .zip(base)
        .map(|(plan, base)| {
            if !matches!(base.hardware, HardwareAssessment::Fits { .. }) {
                return Ok(base);
            }
            plan.mtp = icn_mtp::select_mtp_with_backend(
                &backend,
                plan,
                icn_mtp::CandidatePolicy::Automatic(&request.mtp),
            )
            .context("failed to select a native MTP configuration")?;
            if matches!(plan.mtp, icn_contracts::MtpConfig::Disabled { .. }) {
                return Ok(base);
            }
            let hardware =
                icn_hardware::assess_with_backend(&backend, plan, CapacityPolicy::default())?
                    .assessment;
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
fn run_isolated_planning(request: PlanningWorkerRequest) -> anyhow::Result<PlanningWorkerResponse> {
    icn_engine::disable_native_diagnostics();
    assess_planning_request(request)
}

#[cfg(not(test))]
fn run_isolated_planning(request: PlanningWorkerRequest) -> anyhow::Result<PlanningWorkerResponse> {
    use std::io::Write as _;

    let executable = std::env::current_exe().context("failed to locate ICN planner executable")?;
    let mut child = ProcessCommand::new(executable)
        .arg("plan-worker")
        .env("MAGNITUDE_OTEL", "0")
        .env("RUST_LOG", "error")
        .env_remove("MAGNITUDE_OTEL_ENDPOINT")
        .env_remove("OTEL_EXPORTER_OTLP_ENDPOINT")
        .env_remove("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        .env_remove("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
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

fn run_planning_worker() -> anyhow::Result<()> {
    let request = serde_json::from_reader(std::io::stdin().lock())
        .context("failed to decode native planner request")?;
    let assessment = assess_planning_request(request)?;
    serde_json::to_writer(std::io::stdout().lock(), &assessment)
        .context("failed to encode native planner result")?;
    Ok(())
}

fn inspect_template_request(request: TemplateWorkerRequest) -> anyhow::Result<TemplateAssessment> {
    icn_engine::disable_native_diagnostics();
    let inspection =
        icn_reasoning::inspect_template_inputs(&icn_contracts::EffectiveTemplateInputs {
            model_path: request.model_path,
        })?;
    Ok(TemplateAssessment {
        capabilities: inspection.capabilities,
        reasoning: inspection.reasoning,
        fingerprint: inspection.template_fingerprint,
    })
}

#[cfg(test)]
fn run_isolated_template_inspection(
    request: TemplateWorkerRequest,
) -> anyhow::Result<TemplateAssessment> {
    inspect_template_request(request)
}

#[cfg(not(test))]
fn run_isolated_template_inspection(
    request: TemplateWorkerRequest,
) -> anyhow::Result<TemplateAssessment> {
    use std::io::Write as _;

    let executable = std::env::current_exe().context("failed to locate ICN template worker")?;
    let mut child = ProcessCommand::new(executable)
        .arg("template-worker")
        .env("MAGNITUDE_OTEL", "0")
        .env("RUST_LOG", "error")
        .env_remove("MAGNITUDE_OTEL_ENDPOINT")
        .env_remove("OTEL_EXPORTER_OTLP_ENDPOINT")
        .env_remove("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        .env_remove("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
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

fn run_template_worker() -> anyhow::Result<()> {
    let request = serde_json::from_reader(std::io::stdin().lock())
        .context("failed to decode native template request")?;
    let assessment = inspect_template_request(request)?;
    serde_json::to_writer(std::io::stdout().lock(), &assessment)
        .context("failed to encode native template assessment")?;
    Ok(())
}

impl InventoryHardwareAssessor for NativeHardwareAssessor {
    fn cache_key(&self) -> BoxFuture<'_, Result<String, InventoryError>> {
        Box::pin(async move {
            let snapshot = HardwareProvider::snapshot(self).await?;
            ModelHardwareAssessor::cache_key(self, None, &snapshot)
        })
    }

    fn assess(
        &self,
        resolved: ResolvedModel,
    ) -> BoxFuture<'_, Result<HardwareAssessment, InventoryError>> {
        Box::pin(self.assess_resolved(resolved, None))
    }
}

impl ModelHardwareAssessor for NativeHardwareAssessor {
    fn policy_identity(&self) -> &str {
        ASSESSMENT_PROFILE_RESOLVER
    }

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
                .clone();
            let native_build = build_identity::native_build();
            let enabled_backends = build_identity::enabled_backends()
                .into_iter()
                .map(str::to_owned)
                .collect();
            let snapshot = spawn_blocking_traced(move || match native_executor {
                Some(executor) => executor
                    .run_exclusive_native(move |backend| {
                        Ok(discover_hardware(
                            backend,
                            CapacityPolicy::default(),
                            native_build,
                            enabled_backends,
                            ASSESSMENT_PROFILE_RESOLVER,
                        ))
                    })
                    .map_err(|error| InventoryError::Internal(error.to_string()))?,
                None => discover(
                    CapacityPolicy::default(),
                    native_build,
                    enabled_backends,
                    ASSESSMENT_PROFILE_RESOLVER,
                )
                .map_err(|error| InventoryError::Internal(error.to_string())),
            })
            .await
            .map_err(|error| InventoryError::Internal(error.to_string()))??;
            let mut calibration = self.calibration.lock().await;
            if calibration.topology_fingerprint.as_deref()
                != Some(snapshot.topology_fingerprint.as_str())
            {
                calibration.topology_fingerprint = Some(snapshot.topology_fingerprint.clone());
                calibration.result = None;
            }
            Ok(snapshot)
        })
    }
}

#[derive(Clone)]
struct NativeRuntimeController {
    backends: BackendRegistry,
    inventory: Arc<ModelManager>,
    assessor: Arc<NativeHardwareAssessor>,
    native_executor: Arc<RwLock<Option<Arc<LlamaCompletionBackend>>>>,
    defaults: RuntimePlanDefaults,
    state: Arc<tokio::sync::RwLock<RuntimeStateResponse>>,
    mutation: Arc<tokio::sync::Mutex<()>>,
    next_operation: Arc<AtomicU64>,
}

impl NativeRuntimeController {
    fn new(
        backends: BackendRegistry,
        inventory: Arc<ModelManager>,
        assessor: Arc<NativeHardwareAssessor>,
        native_executor: Arc<RwLock<Option<Arc<LlamaCompletionBackend>>>>,
        defaults: RuntimePlanDefaults,
        initial: RuntimeStateResponse,
    ) -> Self {
        Self {
            backends,
            inventory,
            assessor,
            native_executor,
            defaults,
            state: Arc::new(tokio::sync::RwLock::new(initial)),
            mutation: Arc::new(tokio::sync::Mutex::new(())),
            next_operation: Arc::new(AtomicU64::new(1)),
        }
    }

    async fn emit(
        sender: &tokio::sync::mpsc::Sender<RuntimeModelEvent>,
        event: RuntimeModelEvent,
    ) -> bool {
        sender.send(event).await.is_ok()
    }

    async fn fail(
        &self,
        sender: &tokio::sync::mpsc::Sender<RuntimeModelEvent>,
        operation_id: String,
        model_id: String,
        code: &str,
        message: impl Into<String>,
        retryable: bool,
        runtime_lost: bool,
    ) {
        let message = message.into();
        tracing::error!(
            operation.id = %operation_id,
            model.id = %model_id,
            error.code = code,
            error.message = %message,
            error.retryable = retryable,
            runtime.lost = runtime_lost,
            "runtime model operation failed"
        );
        if runtime_lost {
            let generation = self.backends.generation();
            *self.state.write().await = RuntimeStateResponse {
                generation,
                status: RuntimeStatus::Failed {
                    generation,
                    code: code.to_owned(),
                    message: message.clone(),
                },
                operation_id: None,
            };
        } else {
            self.state.write().await.operation_id = None;
        }
        let _ = Self::emit(
            sender,
            RuntimeModelEvent::Failed {
                operation_id,
                model_id,
                code: code.to_owned(),
                message,
                retryable,
            },
        )
        .await;
    }

    fn profile_defaults(
        &self,
        profile: &RuntimeExecutionProfile,
    ) -> Result<RuntimePlanDefaults, InventoryError> {
        if profile.policy != ASSESSMENT_PROFILE_RESOLVER {
            return Err(InventoryError::InvalidRequest(format!(
                "unsupported runtime profile policy; expected {ASSESSMENT_PROFILE_RESOLVER}"
            )));
        }
        let mut defaults = self.defaults.clone();
        defaults.context_size = profile.context_length;
        defaults.max_sequences = profile.parallel_sequences;
        Ok(defaults)
    }

    async fn resolved_load(
        &self,
        model_id: &str,
        profile: &RuntimeExecutionProfile,
    ) -> Result<(ResolvedModel, ExecutionIntent), InventoryError> {
        let id = ModelId::parse(model_id.to_owned())?;
        self.inventory.ensure_model_inventory().await?;
        let resolved = self.inventory.resolve_ready(&id).await?;
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
        let mtp = resolved
            .components
            .iter()
            .filter(|component| matches!(component.role, ComponentRole::Mtp | ComponentRole::Draft))
            .map(|component| component.path.clone())
            .collect();
        let defaults = self.profile_defaults(profile)?;
        let plan = spawn_blocking_traced(move || {
            load_execution_intent(primary, projector, MtpSelection::Automatic(mtp), &defaults)
        })
        .await
        .map_err(|error| InventoryError::Internal(error.to_string()))?
        .map_err(|error| {
            InventoryError::Internal(format!("failed to resolve runtime plan: {error:#}"))
        })?;
        Ok((resolved, plan))
    }

    #[tracing::instrument(
        name = "icn.runtime.load.operation",
        skip_all,
        fields(operation.id = %operation_id, model.id = %request.model_id)
    )]
    async fn run_load(
        self,
        request: LoadRuntimeModelRequest,
        operation_id: String,
        sender: tokio::sync::mpsc::Sender<RuntimeModelEvent>,
    ) {
        let Ok(_guard) = self.mutation.try_lock() else {
            self.fail(
                &sender,
                operation_id,
                request.model_id,
                "runtime_busy",
                "another runtime mutation is already active",
                true,
                false,
            )
            .await;
            return;
        };

        let existing = self.state.read().await.clone();
        if matches!(
            &existing.status,
            RuntimeStatus::Ready { model_id, profile, .. }
                if model_id == &request.model_id && profile == &request.profile
        ) {
            let _ = Self::emit(
                &sender,
                RuntimeModelEvent::Ready {
                    operation_id,
                    state: existing,
                },
            )
            .await;
            return;
        };
        let Some(_backend_mutation) = self.backends.try_begin_mutation() else {
            self.fail(
                &sender,
                operation_id,
                request.model_id,
                "model_in_use",
                "the active runtime has in-flight inference or template requests",
                true,
                false,
            )
            .await;
            return;
        };

        self.state.write().await.operation_id = Some(operation_id.clone());
        for stage in [RuntimeLoadStage::Queued, RuntimeLoadStage::Resolving] {
            if !Self::emit(
                &sender,
                RuntimeModelEvent::Progress {
                    operation_id: operation_id.clone(),
                    model_id: request.model_id.clone(),
                    stage,
                },
            )
            .await
            {
                self.state.write().await.operation_id = None;
                return;
            }
        }

        let (resolved, plan) = match self
            .resolved_load(&request.model_id, &request.profile)
            .await
        {
            Ok(value) => value,
            Err(error) => {
                self.fail(
                    &sender,
                    operation_id,
                    request.model_id,
                    "model_unavailable",
                    error.to_string(),
                    false,
                    false,
                )
                .await;
                return;
            }
        };
        let _ = Self::emit(
            &sender,
            RuntimeModelEvent::Progress {
                operation_id: operation_id.clone(),
                model_id: request.model_id.clone(),
                stage: RuntimeLoadStage::Assessing,
            },
        )
        .await;

        let assessment_result = {
            let _assessment_guard = self.assessor.gate.lock().await;
            spawn_blocking_traced(move || assess_hardware(&plan, CapacityPolicy::default())).await
        };
        let assessed = match assessment_result {
            Ok(Ok(value)) => value,
            Ok(Err(error)) => {
                self.fail(
                    &sender,
                    operation_id,
                    request.model_id,
                    "assessment_failed",
                    error.to_string(),
                    true,
                    false,
                )
                .await;
                return;
            }
            Err(error) => {
                self.fail(
                    &sender,
                    operation_id,
                    request.model_id,
                    "assessment_task_failed",
                    error.to_string(),
                    true,
                    false,
                )
                .await;
                return;
            }
        };
        if let HardwareAssessment::DoesNotFit { memory, .. } = &assessed.assessment {
            self.fail(
                &sender,
                operation_id,
                request.model_id,
                "does_not_fit",
                format!(
                    "runtime requires {} bytes but stable capacity is {} bytes",
                    memory.required_bytes, memory.available_bytes
                ),
                false,
                false,
            )
            .await;
            return;
        }
        if matches!(assessed.assessment, HardwareAssessment::NotAssessed { .. }) {
            self.fail(
                &sender,
                operation_id,
                request.model_id,
                "assessment_incomplete",
                "runtime assessment did not produce a complete result",
                false,
                false,
            )
            .await;
            return;
        }
        let execution_backend = match &assessed.assessment {
            HardwareAssessment::Fits { profile, .. } => profile.acceleration.clone(),
            _ => "native".to_owned(),
        };

        let _ = Self::emit(
            &sender,
            RuntimeModelEvent::Progress {
                operation_id: operation_id.clone(),
                model_id: request.model_id.clone(),
                stage: RuntimeLoadStage::Unloading,
            },
        )
        .await;
        if let RuntimeStatus::Ready { model_id, .. } = &existing.status
            && let Ok(id) = ModelId::parse(model_id.clone())
        {
            let _ = self
                .inventory
                .update_status(
                    &id,
                    ModelStatus::Available {
                        ready_at: unix_timestamp(),
                    },
                )
                .await;
        }
        if let Ok(mut slot) = self.native_executor.write() {
            *slot = None;
        }
        self.backends.clear();

        let _ = self
            .inventory
            .update_status(
                &resolved.model.id,
                ModelStatus::Loading {
                    load_id: operation_id.clone(),
                    stage: LoadStage::Opening,
                    started_at: unix_timestamp(),
                },
            )
            .await;
        let _ = Self::emit(
            &sender,
            RuntimeModelEvent::Progress {
                operation_id: operation_id.clone(),
                model_id: request.model_id.clone(),
                stage: RuntimeLoadStage::Loading,
            },
        )
        .await;
        let model_id = request.model_id.clone();
        let backend = match spawn_blocking_traced(move || {
            LlamaCompletionBackend::load(model_id, assessed.plan)
        })
        .await
        {
            Ok(Ok(backend)) => Arc::new(backend),
            Ok(Err(error)) => {
                let _ = self
                    .inventory
                    .update_status(
                        &resolved.model.id,
                        ModelStatus::LoadFailed {
                            attempted_at: unix_timestamp(),
                            stage: LoadStage::Opening,
                            code: "backend_load_failed".to_owned(),
                            retryable: true,
                        },
                    )
                    .await;
                self.fail(
                    &sender,
                    operation_id,
                    request.model_id,
                    "backend_load_failed",
                    error.to_string(),
                    true,
                    true,
                )
                .await;
                return;
            }
            Err(error) => {
                self.fail(
                    &sender,
                    operation_id,
                    request.model_id,
                    "load_task_failed",
                    error.to_string(),
                    true,
                    true,
                )
                .await;
                return;
            }
        };
        let _ = Self::emit(
            &sender,
            RuntimeModelEvent::Progress {
                operation_id: operation_id.clone(),
                model_id: request.model_id.clone(),
                stage: RuntimeLoadStage::Verifying,
            },
        )
        .await;
        let properties = match backend.properties() {
            Ok(properties) => properties,
            Err(error) => {
                self.fail(
                    &sender,
                    operation_id,
                    request.model_id,
                    "verification_failed",
                    error.to_string(),
                    true,
                    true,
                )
                .await;
                return;
            }
        };
        let mut aliases = std::collections::BTreeSet::new();
        aliases.insert(resolved.model.name.clone());
        let generation = self
            .backends
            .replace(Arc::clone(&backend) as Arc<dyn CompletionBackend>, aliases);
        if let Ok(mut slot) = self.native_executor.write() {
            *slot = Some(Arc::clone(&backend));
        }
        let execution: std::collections::BTreeMap<_, _> =
            serde_json::to_value(&properties.execution.resolved)
                .ok()
                .and_then(|value| value.as_object().cloned())
                .unwrap_or_default()
                .into_iter()
                .collect();
        let _ = self
            .inventory
            .update_status(
                &resolved.model.id,
                ModelStatus::Loaded {
                    loaded_at: unix_timestamp(),
                    backend: execution_backend,
                    context_length: properties.context_tokens,
                    execution,
                },
            )
            .await;
        let state = RuntimeStateResponse {
            generation,
            status: RuntimeStatus::Ready {
                model_id: request.model_id,
                generation,
                profile: request.profile,
            },
            operation_id: None,
        };
        *self.state.write().await = state.clone();
        let _ = Self::emit(
            &sender,
            RuntimeModelEvent::Ready {
                operation_id,
                state,
            },
        )
        .await;
        tracing::info!("runtime model ready");
    }
}

impl RuntimeController for NativeRuntimeController {
    fn state(&self) -> BoxFuture<'_, Result<RuntimeStateResponse, InventoryError>> {
        Box::pin(async move { Ok(self.state.read().await.clone()) })
    }

    fn load(&self, request: LoadRuntimeModelRequest) -> BoxStream<'static, RuntimeModelEvent> {
        let operation_id = format!(
            "runtime-{}",
            self.next_operation.fetch_add(1, Ordering::Relaxed)
        );
        let runtime = self.clone();
        let (sender, receiver) = tokio::sync::mpsc::channel(16);
        tokio::spawn(
            runtime
                .run_load(request, operation_id, sender)
                .in_current_span(),
        );
        ReceiverStream::new(receiver).boxed()
    }

    fn unload(&self) -> BoxFuture<'_, Result<RuntimeStateResponse, InventoryError>> {
        Box::pin(async move {
            let _guard = self.mutation.lock().await;
            let Some(_backend_mutation) = self.backends.try_begin_mutation() else {
                return Err(InventoryError::Busy(
                    "the active runtime has in-flight inference or template requests".into(),
                ));
            };
            let previous = self.state.read().await.clone();
            if let RuntimeStatus::Ready { model_id, .. } = previous.status
                && let Ok(id) = ModelId::parse(model_id)
            {
                self.inventory
                    .update_status(
                        &id,
                        ModelStatus::Available {
                            ready_at: unix_timestamp(),
                        },
                    )
                    .await?;
            }
            if let Ok(mut slot) = self.native_executor.write() {
                *slot = None;
            }
            let generation = self.backends.clear();
            let state = RuntimeStateResponse {
                generation,
                status: RuntimeStatus::Empty,
                operation_id: None,
            };
            *self.state.write().await = state.clone();
            Ok(state)
        })
    }
}

impl std::str::FromStr for TensorSplitArg {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.is_empty() {
            return Err("tensor split must contain at least one weight".to_owned());
        }
        let weights = value
            .split([',', '/'])
            .enumerate()
            .map(|(index, weight)| {
                weight.parse::<f32>().map_err(|_| {
                    format!("tensor split weight {index} is not a valid number: {weight:?}")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if let Some((index, weight)) = weights
            .iter()
            .copied()
            .enumerate()
            .find(|(_, weight)| !weight.is_finite() || *weight < 0.0)
        {
            return Err(format!(
                "tensor split weight {index} must be finite and non-negative, received {weight}"
            ));
        }
        if !weights.iter().any(|weight| *weight > 0.0) {
            return Err("tensor split must assign a positive weight to at least one device".into());
        }
        Ok(Self(weights))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _telemetry = telemetry::init()?;
    // Native planner diagnostics are extremely verbose and can dominate metadata-only fitting.
    // ICN emits bounded, structured operation telemetry at the service boundary instead.
    icn_engine::disable_native_diagnostics();
    match Cli::parse().command {
        Command::Serve {
            bind,
            instance_id,
            parent_pid,
            auth_token,
            fake,
            model,
            model_id,
            mmproj,
            mtp_model,
            no_mmproj_offload,
            no_mmproj_warmup,
            image_min_tokens,
            image_max_tokens,
            model_alias,
            model_store,
            model_sources,
            hf_caches,
            context_size,
            batch_size,
            ubatch_size,
            max_sequences,
            prefill_quantum,
            gpu_layers,
            no_mmap,
            mlock,
            split_mode,
            tensor_split,
            cache_type_k,
            cache_type_v,
            no_kv_offload,
            no_op_offload,
            swa_full,
            kv_unified,
            threads,
            threads_batch,
            flash_attention,
        } => {
            let inventory_root = match model_store {
                Some(root) => root,
                None => InventoryConfig::default_root()
                    .context("failed to determine default model store")?,
            };
            let mut inventory_config = InventoryConfig::with_root(inventory_root)
                .context("invalid model inventory configuration")?;
            inventory_config.model_sources.extend(model_sources);
            inventory_config.hf_cache_dirs.extend(hf_caches);
            let plan_defaults = RuntimePlanDefaults {
                context_size,
                batch_size,
                ubatch_size,
                max_sequences,
                prefill_quantum: prefill_quantum.unwrap_or(batch_size),
                execution: ExecutionConfig {
                    gpu_layers,
                    use_mmap: !no_mmap,
                    use_mlock: mlock,
                    split_mode,
                    tensor_split: tensor_split.map(|value| value.0),
                    cache_type_k,
                    cache_type_v,
                    offload_kqv: !no_kv_offload,
                    operation_offload: !no_op_offload,
                    swa_full,
                    kv_unified,
                    threads,
                    threads_batch,
                    flash_attention: match flash_attention {
                        FlashAttentionArg::Auto => FlashAttention::Auto,
                        FlashAttentionArg::Off => FlashAttention::Disabled,
                        FlashAttentionArg::On => FlashAttention::Enabled,
                    },
                },
                projector_use_gpu: !no_mmproj_offload,
                projector_warmup: !no_mmproj_warmup,
                image_min_tokens,
                image_max_tokens,
            };
            let inventory = Arc::new(
                ModelManager::open_with_template_assessor(
                    inventory_config,
                    Some(Arc::new(NativeTemplateAssessor)),
                )
                .await
                .context("failed to initialize model inventory")?,
            );
            let native_executor_slot = Arc::new(RwLock::new(None));
            let inventory_hardware_assessor = Arc::new(NativeHardwareAssessor {
                defaults: plan_defaults.clone(),
                native_executor: Arc::clone(&native_executor_slot),
                gate: tokio::sync::Mutex::new(()),
                planning_slots: Arc::new(tokio::sync::Semaphore::new(planner_concurrency())),
                calibration: tokio::sync::Mutex::new(CalibrationCache::default()),
            });
            inventory
                .set_hardware_assessor(inventory_hardware_assessor.clone())
                .context("failed to configure inventory hardware assessment")?;
            let previewer = Arc::new(ModelPreviewService::new(
                inventory.clone(),
                inventory_hardware_assessor.clone(),
            ));
            let (model, mmproj, mtp_model, model_alias, selected_inventory_id) = if let Some(
                raw_id,
            ) = model_id
            {
                if mmproj.is_some() || mtp_model.is_some() {
                    anyhow::bail!(
                        "--model-id resolves projector and MTP components from inventory; explicit --mmproj/--mtp-model overrides are not allowed"
                    );
                }
                let id = ModelId::parse(raw_id).context("invalid inventory model ID")?;
                inventory
                    .ensure_model_inventory()
                    .await
                    .context("failed to reconcile inventory for model selection")?;
                let resolved = inventory
                    .resolve_ready(&id)
                    .await
                    .context("failed to resolve inventory model")?;
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
                    .context("inventory model has no runnable weight component")?;
                let projector = resolved
                    .components
                    .iter()
                    .find(|component| component.role == ComponentRole::Projector)
                    .map(|component| component.path.clone());
                let mtp = resolved
                    .components
                    .iter()
                    .filter(|component| {
                        matches!(component.role, ComponentRole::Mtp | ComponentRole::Draft)
                    })
                    .map(|component| component.path.clone())
                    .collect();
                (
                    Some(primary),
                    projector,
                    MtpSelection::Automatic(mtp),
                    model_alias.or(Some(resolved.model.name)),
                    Some(id),
                )
            } else {
                (
                    model,
                    mmproj,
                    mtp_model.map_or_else(
                        || MtpSelection::Automatic(Vec::new()),
                        MtpSelection::Explicit,
                    ),
                    model_alias,
                    None,
                )
            };
            let backends = BackendRegistry::empty();
            let (native_executor, initial_runtime) = if fake {
                let model_id = model_alias.unwrap_or_else(|| "icn-fake".into());
                let mut aliases = std::collections::BTreeSet::new();
                aliases.insert(model_id.clone());
                let generation = backends.replace(
                    Arc::new(FakeBackend::new(model_id.clone(), "Hello from ICN.")),
                    aliases,
                );
                (
                    None,
                    RuntimeStateResponse {
                        generation,
                        status: RuntimeStatus::Ready {
                            model_id,
                            generation,
                            profile: RuntimeExecutionProfile {
                                policy: ASSESSMENT_PROFILE_RESOLVER.to_owned(),
                                context_length: plan_defaults.context_size,
                                parallel_sequences: plan_defaults.max_sequences,
                            },
                        },
                        operation_id: None,
                    },
                )
            } else if model.is_none() {
                (
                    None,
                    RuntimeStateResponse {
                        generation: backends.generation(),
                        status: RuntimeStatus::Empty,
                        operation_id: None,
                    },
                )
            } else {
                let path = model.expect("model is present");
                let alias = model_alias.unwrap_or_else(|| {
                    path.file_stem()
                        .and_then(|value| value.to_str())
                        .unwrap_or("local-model")
                        .to_owned()
                });
                let inventory_id = match selected_inventory_id {
                    Some(id) => id,
                    None => inventory
                        .register_active_model(&path, Some(&alias))
                        .await
                        .context("failed to register the active model")?,
                };
                let requested_plan =
                    load_execution_intent(path, mmproj, mtp_model, &plan_defaults)?;
                let assessed = assess_hardware(&requested_plan, CapacityPolicy::default())
                    .context("failed to assess execution intent")?;
                if let icn_contracts::HardwareAssessment::DoesNotFit { memory, .. } =
                    &assessed.assessment
                {
                    anyhow::bail!(
                        "execution intent requires {} bytes but stable capacity is {} bytes",
                        memory.required_bytes,
                        memory.available_bytes
                    );
                }
                let execution_backend = match &assessed.assessment {
                    HardwareAssessment::Fits { profile, .. } => profile.acceleration.clone(),
                    _ => "native".to_owned(),
                };
                let load_id = format!("load-{}", unix_timestamp());
                inventory
                    .update_status(
                        &inventory_id,
                        ModelStatus::Loading {
                            load_id,
                            stage: LoadStage::Opening,
                            started_at: unix_timestamp(),
                        },
                    )
                    .await
                    .context("failed to project model loading state")?;
                let backend = match LlamaCompletionBackend::load(
                    inventory_id.0.clone(),
                    assessed.plan.clone(),
                ) {
                    Ok(backend) => backend,
                    Err(error) => {
                        let _ = inventory
                            .update_status(
                                &inventory_id,
                                ModelStatus::LoadFailed {
                                    attempted_at: unix_timestamp(),
                                    stage: LoadStage::Opening,
                                    code: "backend_load_failed".to_owned(),
                                    retryable: true,
                                },
                            )
                            .await;
                        return Err(error).context("failed to load native backend");
                    }
                };
                let properties = backend
                    .properties()
                    .context("failed to read loaded model properties")?;
                let execution: std::collections::BTreeMap<_, _> =
                    serde_json::to_value(&properties.execution.resolved)
                        .ok()
                        .and_then(|value| value.as_object().cloned())
                        .unwrap_or_default()
                        .into_iter()
                        .collect();
                inventory
                    .update_status(
                        &inventory_id,
                        ModelStatus::Loaded {
                            loaded_at: unix_timestamp(),
                            backend: execution_backend,
                            context_length: properties.context_tokens,
                            execution,
                        },
                    )
                    .await
                    .context("failed to project loaded model state")?;
                let backend = Arc::new(backend);
                let mut aliases = std::collections::BTreeSet::new();
                aliases.insert(alias);
                let generation =
                    backends.replace(Arc::clone(&backend) as Arc<dyn CompletionBackend>, aliases);
                (
                    Some(backend),
                    RuntimeStateResponse {
                        generation,
                        status: RuntimeStatus::Ready {
                            model_id: inventory_id.0,
                            generation,
                            profile: RuntimeExecutionProfile {
                                policy: ASSESSMENT_PROFILE_RESOLVER.to_owned(),
                                context_length: plan_defaults.context_size,
                                parallel_sequences: plan_defaults.max_sequences,
                            },
                        },
                        operation_id: None,
                    },
                )
            };
            *native_executor_slot
                .write()
                .map_err(|_| anyhow::anyhow!("native executor lock poisoned"))? = native_executor;
            let runtime = Arc::new(NativeRuntimeController::new(
                backends.clone(),
                inventory.clone(),
                inventory_hardware_assessor.clone(),
                native_executor_slot.clone(),
                plan_defaults,
                initial_runtime,
            ));
            let native_build = build_identity::native_build();
            let identity = ServerIdentity {
                instance_id: instance_id.clone(),
                api_version: 1,
                native_build: native_build.clone(),
            };
            let mut state = AppState::model_free(backends)
                .with_inventory(inventory)
                .with_hardware(inventory_hardware_assessor)
                .with_previewer(previewer.clone())
                .with_hugging_face_catalog(previewer)
                .with_runtime(runtime)
                .with_identity(identity);
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
            println!(
                "MAGNITUDE_ICN_READY {}",
                serde_json::json!({
                    "type": "icn_ready",
                    "protocolVersion": 1,
                    "origin": origin,
                    "instanceId": instance_id,
                    "pid": std::process::id(),
                    "apiVersion": 1,
                    "nativeBuild": native_build,
                })
            );
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
            axum::serve(listener, app)
                .with_graceful_shutdown(shutdown_signal(parent_pid))
                .await?;
            tracing::info!("ICN server stopped");
        }
        Command::Doctor => println!("ICN runtime and native backend loaded successfully"),
        Command::Version { json } => {
            if json {
                println!("{}", build_identity::json());
            } else {
                println!("{}", env!("CARGO_PKG_VERSION"));
            }
        }
        Command::PlanWorker => run_planning_worker()?,
        Command::TemplateWorker => run_template_worker()?,
    }
    Ok(())
}

async fn shutdown_signal(parent_pid: Option<u32>) {
    tokio::select! {
        _ = interrupt_signal() => {},
        _ = parent_watchdog(parent_pid), if parent_pid.is_some() => {},
    }
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

fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
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

    fn parity_test_defaults() -> RuntimePlanDefaults {
        RuntimePlanDefaults {
            context_size: 128,
            batch_size: 128,
            ubatch_size: 64,
            max_sequences: 1,
            prefill_quantum: 128,
            execution: ExecutionConfig::default(),
            projector_use_gpu: true,
            projector_warmup: true,
            image_min_tokens: None,
            image_max_tokens: None,
        }
    }

    #[test]
    fn available_and_preview_cache_keys_share_resolved_profile_identity() {
        let assessor = NativeHardwareAssessor {
            defaults: parity_test_defaults(),
            native_executor: Arc::new(RwLock::new(None)),
            gate: tokio::sync::Mutex::new(()),
            planning_slots: Arc::new(tokio::sync::Semaphore::new(1)),
            calibration: tokio::sync::Mutex::new(CalibrationCache::default()),
        };
        let snapshot = HardwareSnapshot {
            captured_at: 1,
            platform: "test".to_owned(),
            architecture: "test".to_owned(),
            cpu_model: None,
            logical_cores: 1,
            system_memory: icn_contracts::HardwareSystemMemory {
                total_bytes: 1,
                current_available_bytes: Some(1),
            },
            native_build: "native".to_owned(),
            enabled_backends: vec!["cpu".to_owned()],
            assessment_policy: ASSESSMENT_PROFILE_RESOLVER.to_owned(),
            capacity_policy: "capacity".to_owned(),
            topology_fingerprint: "topology".to_owned(),
            memory_domains: Vec::new(),
        };
        let equivalent_preview = ModelPreviewProfile {
            id: "caller-correlation-does-not-affect-fit".to_owned(),
            policy: ASSESSMENT_PROFILE_RESOLVER.to_owned(),
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
        assert!(
            assessor
                .assessment_cache_key(
                    Some(&ModelPreviewProfile {
                        policy: "unknown-policy".to_owned(),
                        ..equivalent_preview
                    }),
                    &snapshot,
                )
                .is_err()
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
            native_executor: Arc::new(RwLock::new(None)),
            gate: tokio::sync::Mutex::new(()),
            planning_slots: Arc::new(tokio::sync::Semaphore::new(1)),
            calibration: tokio::sync::Mutex::new(CalibrationCache::default()),
        });
        let profile = ModelPreviewProfile {
            id: "parity".to_owned(),
            policy: ASSESSMENT_PROFILE_RESOLVER.to_owned(),
            context_length: 128,
            parallel_sequences: 1,
        };

        for fixture in fixtures {
            let store = tempfile::tempdir().expect("temporary model store");
            let mut config = InventoryConfig::with_root(store.path().join("inventory"))
                .expect("inventory config");
            config.model_sources = vec![fixture.parent().expect("fixture parent").to_path_buf()];
            config.hf_cache_dirs.clear();
            let manager = ModelManager::open_with_template_assessor(
                config,
                Some(Arc::new(NativeTemplateAssessor)),
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
    fn execution_cli_defaults_and_explicit_values_are_typed() {
        let defaults = Cli::try_parse_from(["magnitude-icn", "serve", "--fake"]).unwrap();
        let Command::Serve {
            gpu_layers,
            no_mmap,
            mlock,
            split_mode,
            tensor_split,
            cache_type_k,
            cache_type_v,
            no_kv_offload,
            no_op_offload,
            swa_full,
            kv_unified,
            threads,
            threads_batch,
            ..
        } = defaults.command
        else {
            panic!("expected serve command")
        };
        assert_eq!(gpu_layers, GpuLayers::Auto);
        assert!(!no_mmap && !mlock);
        assert_eq!(split_mode, SplitMode::Layer);
        assert!(tensor_split.is_none());
        assert_eq!(cache_type_k, CacheType::F16);
        assert_eq!(cache_type_v, CacheType::F16);
        assert!(!no_kv_offload && !no_op_offload && !swa_full && !kv_unified);
        assert!(threads.is_none() && threads_batch.is_none());

        let explicit = Cli::try_parse_from([
            "magnitude-icn",
            "serve",
            "--fake",
            "--gpu-layers",
            "all",
            "--no-mmap",
            "--mlock",
            "--split-mode",
            "row",
            "--tensor-split",
            "3,1",
            "--cache-type-k",
            "q8_0",
            "--threads",
            "6",
            "--threads-batch",
            "8",
        ])
        .unwrap();
        let Command::Serve {
            gpu_layers,
            no_mmap,
            mlock,
            split_mode,
            tensor_split,
            cache_type_k,
            threads,
            threads_batch,
            ..
        } = explicit.command
        else {
            panic!("expected serve command")
        };
        assert_eq!(gpu_layers, GpuLayers::All);
        assert!(no_mmap && mlock);
        assert_eq!(split_mode, SplitMode::Row);
        assert_eq!(tensor_split.unwrap().0, vec![3.0, 1.0]);
        assert_eq!(cache_type_k, CacheType::Q8_0);
        assert_eq!(threads, NonZeroU32::new(6));
        assert_eq!(threads_batch, NonZeroU32::new(8));
    }

    #[test]
    fn tensor_split_cli_rejects_unsafe_weights() {
        assert!("0,0".parse::<TensorSplitArg>().is_err());
        assert!("1,-1".parse::<TensorSplitArg>().is_err());
        assert!("NaN,1".parse::<TensorSplitArg>().is_err());
    }

    #[test]
    fn inventory_model_id_is_mutually_exclusive_with_paths_and_fake_mode() {
        assert!(
            Cli::try_parse_from([
                "magnitude-icn",
                "serve",
                "--model-id",
                "mdl_0123456789abcdef"
            ])
            .is_ok()
        );
        assert!(
            Cli::try_parse_from([
                "magnitude-icn",
                "serve",
                "--model-id",
                "mdl_0123456789abcdef",
                "--model",
                "/tmp/model.gguf"
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "magnitude-icn",
                "serve",
                "--model-id",
                "mdl_0123456789abcdef",
                "--fake"
            ])
            .is_err()
        );

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
        let value = build_identity::json();
        assert_eq!(value["native_build"], build_identity::native_build());
        assert!(value.get("bindings_revision").is_none());
        assert!(value.get("native_backend_revision").is_none());
        assert_eq!(value["target"], build_identity::TARGET);
        assert_eq!(value["profile"], build_identity::PROFILE);
        assert!(
            value["backends"]
                .as_array()
                .is_some_and(|values| !values.is_empty())
        );
    }
}
