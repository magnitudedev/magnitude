//! Persistent llama.cpp executor for ICN.

use std::collections::{BTreeMap, VecDeque};
use std::num::{NonZeroI32, NonZeroU32};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{
    Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError, sync_channel,
};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use icn_contracts::output::{StopBuffer, Utf8Buffer};
use icn_contracts::{
    AllowedToolsMode, CacheType, ChatContent, ChatContentPart, ChatRequest, ChatTemplateRequest,
    CompletionBackend, ExecutionConfig, ExecutionConfigReport, ExecutionIntent, FinishReason,
    FlashAttention, Generation, GenerationMetrics, GenerationSnapshot, GpuLayers, GrammarTrigger,
    HardwareAssessment, HardwareSnapshot, ImageInput, InferenceError, InferenceEvent,
    InferenceStreamEvent, ModelModalities, ModelProperties, PreparedChatInfo, ProjectorConfig,
    ReasoningControl, ResidentMemory, ResidentMemoryDomain, ResponseFormat, SplitMode,
    TemplateCapabilities, ToolCall, ToolChoice,
};
use llama_cpp_2::LlamaStateSeqFlags;
use llama_cpp_2::TokenToStringError;
use llama_cpp_2::common_chat::{
    ChatContent as NativeChatContent, ChatContentPart as NativeChatContentPart,
    ChatContentPartKind, ChatMessage as NativeChatMessage, ChatParserOptions, ChatPrepareOptions,
    ChatReasoningFormat, ChatSemanticDelta, ChatStreamParser, ChatTemplateKwarg, ChatTool,
    ChatToolCall, ChatToolChoice, CommonChatTemplates, ParsedChatMessage, PreparedChat,
};
use llama_cpp_2::common_sampling::{
    CommonGrammar, CommonGrammarKind, CommonGrammarTrigger, CommonReasoningBudget, CommonSampler,
    CommonSamplerConfig, ReasoningBudgetLimit,
};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::{LlamaMemoryBreakdown, LlamaMemoryBreakdownError, LlamaMemoryLocation};
use llama_cpp_2::llama_backend::{LlamaBackend, LlamaThreadPool, LlamaThreadPoolParams};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::{LlamaGpuLayers, LlamaModelParams};
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::mtp::{MtpOperations, MtpParams, MtpSession};
use llama_cpp_2::token::LlamaToken;
use sha2::{Digest, Sha256};

mod scheduler;

/// Suppress the native backend's process-global diagnostic callback.
///
/// ICN emits bounded structured diagnostics around native operations; the backend's verbose model
/// planning dump is neither bounded nor suitable for service telemetry.
pub fn disable_native_diagnostics() {
    llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default().with_logs_enabled(false));
}

/// Process-lifetime ownership of llama.cpp's global backend registration.
///
/// ICN constructs this capability once while entering its ready lifetime. Model executors and
/// model-free hardware observations borrow the same registration through clones of this handle;
/// neither operation can initialize or tear down the process-global backend independently.
#[derive(Clone)]
pub struct NativeBackend {
    backend: Arc<LlamaBackend>,
}

impl NativeBackend {
    /// Initialize the process-global native backend.
    ///
    /// This is a composition-root operation. Runtime operations accept an existing
    /// [`NativeBackend`] and therefore cannot surface `BackendAlreadyInitialized` as an
    /// operational model or hardware failure.
    pub fn initialize() -> Result<Self, llama_cpp_2::LlamaCppError> {
        LlamaBackend::init().map(|backend| Self {
            backend: Arc::new(backend),
        })
    }

    /// Observe model-free hardware through this process's initialized backend.
    #[must_use]
    pub fn discover_hardware(
        &self,
        policy: icn_hardware::CapacityPolicy,
        native_build: impl Into<String>,
        enabled_backends: Vec<String>,
    ) -> HardwareSnapshot {
        icn_hardware::discover_hardware(
            self.backend.as_ref(),
            policy,
            native_build,
            enabled_backends,
        )
    }

    /// Borrow the initialized backend for isolated native planning within this process.
    #[must_use]
    pub fn as_llama_backend(&self) -> &LlamaBackend {
        self.backend.as_ref()
    }

    /// Resolve the exact native load plan without making a model resident.
    pub fn prepare_load(
        &self,
        model_id: impl Into<String>,
        config: ExecutionIntent,
        mtp_selection: MtpCandidateSelection,
    ) -> Result<PreparedModelLoad, ModelLoadError> {
        PreparedModelLoad::prepare(
            Arc::clone(&self.backend),
            model_id.into(),
            config,
            mtp_selection,
        )
    }
}

#[cfg(feature = "parity-probe")]
#[doc(hidden)]
pub mod parity_probe;

#[cfg(feature = "mtmd")]
mod multimodal;

#[cfg(not(feature = "mtmd"))]
mod multimodal {
    use std::marker::PhantomData;

    pub(crate) struct MultimodalPrompt;
    pub(crate) struct MultimodalRuntime<'model>(PhantomData<&'model ()>);
}

use multimodal::{MultimodalPrompt, MultimodalRuntime};
use scheduler::{
    BatchPlanner, BatchWork, PromptCheckpoint, SequenceCache, SequencePool, WorkCandidate, WorkKind,
};

const COMMAND_QUEUE_CAPACITY: usize = 32;
// Keep transport serialization off the native decode critical path. This remains bounded, but is
// large enough for a full batch of per-token semantic events while the async HTTP task catches up.
const EVENT_QUEUE_CAPACITY: usize = 512;
const OUTBOUND_QUEUE_CAPACITY: usize = 64;
const IDLE_POLL_INTERVAL: Duration = Duration::from_millis(1);
// A native prefill can occupy the scheduler thread for seconds on large models. When an idle
// endpoint receives an explicitly concurrent burst, give sibling requests one millisecond to
// reach the command queue before beginning that first blocking decode. This mirrors the natural
// task coalescing in llama-server's update loop without delaying an already-active sequence.
const IDLE_ADMISSION_COALESCE_INTERVAL: Duration = Duration::from_millis(1);

type ExclusiveNativeTask = Box<dyn FnOnce(&LlamaBackend) + Send + 'static>;

struct HardwareObservationRequest {
    model_id: String,
    runtime_generation: u64,
    policy: icn_hardware::CapacityPolicy,
    native_build: String,
    enabled_backends: Vec<String>,
    response: SyncSender<Result<HardwareSnapshot, HardwareObservationError>>,
}

#[derive(Debug)]
pub enum HardwareObservationError {
    ExecutorStopped,
    MemoryDomainUnresolved { location: String },
}

impl std::fmt::Display for HardwareObservationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExecutorStopped => formatter.write_str("inference executor stopped"),
            Self::MemoryDomainUnresolved { location } => write!(
                formatter,
                "resident allocation location does not map to a hardware memory domain: {location}"
            ),
        }
    }
}

impl std::error::Error for HardwareObservationError {}

#[derive(Clone)]
struct ResidentAllocation {
    location: LlamaMemoryLocation,
    model_bytes: u64,
    context_bytes: u64,
    compute_bytes: u64,
    auxiliary_bytes: u64,
}

impl From<LlamaMemoryBreakdown> for ResidentAllocation {
    fn from(value: LlamaMemoryBreakdown) -> Self {
        Self {
            location: value.location,
            model_bytes: value.model_bytes,
            context_bytes: value.context_bytes,
            compute_bytes: value.compute_bytes,
            auxiliary_bytes: 0,
        }
    }
}

enum ExecutorCommand {
    Complete {
        request: ChatRequest,
        events: SyncSender<ExecutorItem>,
        cancelled: Arc<AtomicBool>,
        queued_at: Instant,
        span: tracing::Span,
    },
    ApplyTemplate {
        request: ChatTemplateRequest,
        response: SyncSender<Result<PreparedChatInfo, InferenceError>>,
        span: tracing::Span,
    },
    RunExclusiveNative {
        task: ExclusiveNativeTask,
    },
    ObserveHardware(HardwareObservationRequest),
    Shutdown,
}

enum ExecutorItem {
    Event(InferenceStreamEvent),
    Completed(Generation),
    Failed(InferenceError),
}

struct QueuedCompletion {
    request: ChatRequest,
    events: SyncSender<ExecutorItem>,
    cancelled: Arc<AtomicBool>,
    queued_at: Instant,
    span: tracing::Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestPhase {
    Prefill,
    ReadyToSample { batch_index: i32 },
    Decode { token: LlamaToken, position: i32 },
    Terminal,
}

struct ActiveRequest<'model> {
    sequence_id: Option<i32>,
    events: SyncSender<ExecutorItem>,
    span: tracing::Span,
    cancelled: Arc<AtomicBool>,
    outbound: VecDeque<ExecutorItem>,
    phase: RequestPhase,
    prompt: Vec<LlamaToken>,
    /// Tokens whose target KV state is known to be committed. The currently
    /// sampled decode token is deliberately excluded until verification.
    cache_history: Vec<LlamaToken>,
    prompt_offset: usize,
    prompt_tokens: usize,
    cached_prompt_tokens: usize,
    prompt_checkpoints: Vec<PromptCheckpoint>,
    pending_checkpoint_prefixes: VecDeque<usize>,
    next_position: i32,
    multimodal_prompt: Option<MultimodalPrompt>,
    generation_limit: usize,
    generated_tokens: usize,
    mtp_started: bool,
    mtp_draft: Vec<LlamaToken>,
    mtp_indices: Vec<i32>,
    draft_tokens: usize,
    accepted_draft_tokens: usize,
    draft_ms: f64,
    verification_ms: f64,
    cache_prompt: bool,
    cacheable: bool,
    ignore_eos: bool,
    timings_per_token: bool,
    sampler: CommonSampler<'model>,
    utf8: Utf8Buffer,
    stops: StopBuffer,
    semantic: SemanticStream,
    queue_ms: f64,
    prompt_started_at: Option<Instant>,
    prompt_ms: f64,
    generation_started_at: Option<Instant>,
    last_sample_at: Option<Instant>,
    first_event_at: Option<Instant>,
    queued_at: Instant,
}

struct TokenizedPrompt {
    text_tokens: Vec<LlamaToken>,
    total_tokens: usize,
    next_position: i32,
    multimodal: Option<MultimodalPrompt>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FlushOutcome {
    Empty,
    Backpressured,
    Disconnected,
}

/// A handle to a dedicated model executor thread.
pub struct LlamaCompletionBackend {
    model_id: String,
    properties: ModelProperties,
    acceleration: String,
    commands: SyncSender<ExecutorCommand>,
    executor: Mutex<Option<JoinHandle<()>>>,
}

#[derive(Debug, Clone)]
pub enum MtpCandidateSelection {
    Automatic(Vec<PathBuf>),
    Explicit(PathBuf),
}

/// Stable semantic phases of prepared native model loading.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum ModelLoadPhase {
    TargetModel,
    TargetContext,
    DraftModel,
    DraftContext,
    Projector,
    Runtime,
    Warmup,
    Finalize,
}

/// Receives synchronous phase boundaries from the executor.
///
/// `Finalize` begins after warm-up; the prepared-load caller completes it after verification and
/// resident publication so the last measured phase covers the complete ready boundary.
pub trait ModelLoadObserver: Send + Sync + 'static {
    fn phase_started(&self, phase: ModelLoadPhase);
    fn phase_completed(&self, phase: ModelLoadPhase);
}

/// A fully resolved native plan which can be executed without replanning.
pub struct PreparedModelLoad {
    model_id: String,
    acceleration: String,
    timing_plan_identity: String,
    phases: Vec<ModelLoadPhase>,
    commands: SyncSender<ExecutorCommand>,
    start: SyncSender<Arc<dyn ModelLoadObserver>>,
    ready: Receiver<Result<(ModelProperties, String), ModelLoadError>>,
    executor: JoinHandle<()>,
}

impl PreparedModelLoad {
    fn prepare(
        backend: Arc<LlamaBackend>,
        model_id: String,
        config: ExecutionIntent,
        mtp_selection: MtpCandidateSelection,
    ) -> Result<Self, ModelLoadError> {
        validate_model_config(&config).map_err(ModelLoadError::from)?;
        tracing::Span::current().record("model.id", model_id.as_str());
        let (commands, command_receiver) = sync_channel(COMMAND_QUEUE_CAPACITY);
        let (ready_sender, ready) = sync_channel(1);
        let (prepared_sender, prepared_receiver) = sync_channel(1);
        let (start, start_receiver) = sync_channel(1);
        let executor_model_id = model_id.clone();
        let executor = thread::Builder::new()
            .name(format!("icn-llama-{model_id}"))
            .spawn(move || {
                let result = prepare_native_plan(backend.as_ref(), config, mtp_selection);
                match result {
                    Ok((planned, acceleration, phases)) => {
                        let timing_plan_identity = timing_plan_identity(&planned.assessed.plan);
                        if prepared_sender
                            .send(Ok((acceleration.clone(), timing_plan_identity, phases)))
                            .is_err()
                        {
                            return;
                        }
                        let Ok(observer) = start_receiver.recv() else {
                            return;
                        };
                        executor_main(
                            backend,
                            planned,
                            acceleration,
                            command_receiver,
                            ready_sender,
                            observer,
                        );
                    }
                    Err(error) => {
                        let _ = prepared_sender.send(Err(error));
                    }
                }
            })
            .map_err(|error| ModelLoadError::Backend(error.to_string()))?;
        match prepared_receiver.recv() {
            Ok(Ok((acceleration, timing_plan_identity, phases))) => Ok(Self {
                model_id: executor_model_id,
                acceleration,
                timing_plan_identity,
                phases,
                commands,
                start,
                ready,
                executor,
            }),
            Ok(Err(error)) => {
                let _ = executor.join();
                Err(error)
            }
            Err(_) => {
                let _ = executor.join();
                Err(InferenceError::ExecutorStopped.into())
            }
        }
    }

    #[must_use]
    pub fn phases(&self) -> &[ModelLoadPhase] {
        &self.phases
    }

    #[must_use]
    pub fn acceleration(&self) -> &str {
        &self.acceleration
    }

    /// Path-independent identity of load- and allocation-relevant resolved plan values.
    #[must_use]
    pub fn timing_plan_identity(&self) -> &str {
        &self.timing_plan_identity
    }

    pub fn execute(
        self,
        observer: Arc<dyn ModelLoadObserver>,
    ) -> Result<LlamaCompletionBackend, ModelLoadError> {
        let Self {
            model_id,
            commands,
            start,
            ready,
            executor,
            ..
        } = self;
        start
            .send(observer)
            .map_err(|_| ModelLoadError::from(InferenceError::ExecutorStopped))?;
        match ready.recv() {
            Ok(Ok((properties, acceleration))) => Ok(LlamaCompletionBackend {
                model_id,
                properties,
                acceleration,
                commands,
                executor: Mutex::new(Some(executor)),
            }),
            Ok(Err(error)) => {
                let _ = executor.join();
                Err(error)
            }
            Err(_) => {
                let _ = executor.join();
                Err(InferenceError::ExecutorStopped.into())
            }
        }
    }
}

#[derive(Debug)]
pub enum ModelLoadError {
    InvalidConfiguration(String),
    MtpSelection(String),
    Planning(String),
    AssessmentRejected(Box<HardwareAssessment>),
    MemoryAttribution(LlamaMemoryBreakdownError),
    Backend(String),
}

impl std::fmt::Display for ModelLoadError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => {
                write!(formatter, "invalid model configuration: {message}")
            }
            Self::MtpSelection(message) => write!(formatter, "MTP selection failed: {message}"),
            Self::Planning(message) => write!(formatter, "native load planning failed: {message}"),
            Self::AssessmentRejected(assessment) => write!(
                formatter,
                "native load assessment rejected the model: {assessment:?}"
            ),
            Self::MemoryAttribution(error) => {
                write!(formatter, "resident-memory attribution failed: {error}")
            }
            Self::Backend(message) => {
                write!(formatter, "native backend initialization failed: {message}")
            }
        }
    }
}

impl std::error::Error for ModelLoadError {}

impl From<InferenceError> for ModelLoadError {
    fn from(error: InferenceError) -> Self {
        match error {
            InferenceError::InvalidConfig(message) => Self::InvalidConfiguration(message),
            other => Self::Backend(other.to_string()),
        }
    }
}

impl LlamaCompletionBackend {
    /// The normalized acceleration selected by the native load plan.
    #[must_use]
    pub fn acceleration(&self) -> &str {
        &self.acceleration
    }

    /// Run model-free native planning against the executor's initialized llama.cpp backend.
    ///
    /// The operation waits until resident inference work is idle and then runs on the executor
    /// thread. This is intended for native helpers such as `common/fit` that require initialized
    /// devices and process-global serialization, not for model inference or arbitrary callbacks.
    pub fn run_exclusive_native<T, F>(&self, operation: F) -> Result<T, InferenceError>
    where
        T: Send + 'static,
        F: FnOnce(&LlamaBackend) -> T + Send + 'static,
    {
        let (response, receiver) = sync_channel(1);
        let span = tracing::Span::current();
        let task = Box::new(move |backend: &LlamaBackend| {
            span.in_scope(|| {
                let _ = response.send(operation(backend));
            });
        });
        self.commands
            .try_send(ExecutorCommand::RunExclusiveNative { task })
            .map_err(|error| match error {
                TrySendError::Full(_) => InferenceError::Overloaded,
                TrySendError::Disconnected(_) => InferenceError::ExecutorStopped,
            })?;
        receiver.recv().map_err(|_| InferenceError::ExecutorStopped)
    }

    /// Capture current hardware state between scheduler batches without waiting for inference to
    /// become idle.
    pub fn observe_hardware(
        &self,
        runtime_generation: u64,
        policy: icn_hardware::CapacityPolicy,
        native_build: String,
        enabled_backends: Vec<String>,
    ) -> Result<HardwareSnapshot, HardwareObservationError> {
        let (response, receiver) = sync_channel(1);
        self.commands
            .send(ExecutorCommand::ObserveHardware(
                HardwareObservationRequest {
                    model_id: self.model_id.clone(),
                    runtime_generation,
                    policy,
                    native_build,
                    enabled_backends,
                    response,
                },
            ))
            .map_err(|_| HardwareObservationError::ExecutorStopped)?;
        receiver
            .recv()
            .map_err(|_| HardwareObservationError::ExecutorStopped)?
    }
}

impl CompletionBackend for LlamaCompletionBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn properties(&self) -> Result<ModelProperties, InferenceError> {
        Ok(self.properties.clone())
    }

    fn apply_template(
        &self,
        request: ChatTemplateRequest,
    ) -> Result<PreparedChatInfo, InferenceError> {
        let (response, receiver) = sync_channel(1);
        self.commands
            .try_send(ExecutorCommand::ApplyTemplate {
                request,
                response,
                span: tracing::Span::current(),
            })
            .map_err(|error| match error {
                TrySendError::Full(_) => InferenceError::Overloaded,
                TrySendError::Disconnected(_) => InferenceError::ExecutorStopped,
            })?;
        receiver
            .recv()
            .map_err(|_| InferenceError::ExecutorStopped)?
    }

    #[tracing::instrument(
        name = "icn.inference.complete",
        skip_all,
        fields(model.id = %self.model_id),
        err
    )]
    fn complete(
        &self,
        request: ChatRequest,
        on_event: &mut dyn FnMut(InferenceStreamEvent) -> Result<(), InferenceError>,
    ) -> Result<Generation, InferenceError> {
        let (events, event_receiver) = sync_channel(EVENT_QUEUE_CAPACITY);
        let cancelled = Arc::new(AtomicBool::new(false));
        match self.commands.try_send(ExecutorCommand::Complete {
            request,
            events,
            cancelled: Arc::clone(&cancelled),
            queued_at: Instant::now(),
            span: tracing::Span::current(),
        }) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => return Err(InferenceError::Overloaded),
            Err(TrySendError::Disconnected(_)) => return Err(InferenceError::ExecutorStopped),
        }

        loop {
            match event_receiver.recv() {
                Ok(ExecutorItem::Event(event)) => {
                    if let Err(error) = on_event(event) {
                        cancelled.store(true, Ordering::Release);
                        return Err(error);
                    }
                }
                Ok(ExecutorItem::Completed(generation)) => return Ok(generation),
                Ok(ExecutorItem::Failed(error)) => return Err(error),
                Err(_) => return Err(InferenceError::ExecutorStopped),
            }
        }
    }
}

impl Drop for LlamaCompletionBackend {
    fn drop(&mut self) {
        let _ = self.commands.send(ExecutorCommand::Shutdown);
        if let Ok(mut executor) = self.executor.lock()
            && let Some(executor) = executor.take()
        {
            let _ = executor.join();
        }
    }
}

fn prepare_native_plan(
    backend: &LlamaBackend,
    mut requested: ExecutionIntent,
    mtp_selection: MtpCandidateSelection,
) -> Result<(icn_hardware::BackendLoadPlan, String, Vec<ModelLoadPhase>), ModelLoadError> {
    let candidates = match &mtp_selection {
        MtpCandidateSelection::Automatic(paths) => icn_mtp::CandidatePolicy::Automatic(paths),
        MtpCandidateSelection::Explicit(path) => icn_mtp::CandidatePolicy::Explicit(path),
    };
    requested.mtp = icn_mtp::select_mtp_with_backend(backend, &requested, candidates)
        .map_err(|error| ModelLoadError::MtpSelection(error.to_string()))?;
    let planned = icn_hardware::plan_load_with_backend(
        backend,
        &requested,
        icn_hardware::CapacityPolicy::default(),
    )
    .map_err(|error| ModelLoadError::Planning(error.to_string()))?;
    let acceleration = match &planned.assessed.assessment {
        HardwareAssessment::Fits { profile, .. } => profile.acceleration.clone(),
        assessment => {
            return Err(ModelLoadError::AssessmentRejected(Box::new(
                assessment.clone(),
            )));
        }
    };
    let mut phases = vec![ModelLoadPhase::TargetModel, ModelLoadPhase::TargetContext];
    if matches!(
        planned.assessed.plan.mtp,
        icn_contracts::MtpConfig::Enabled {
            source: icn_contracts::MtpSource::Separate { .. },
            ..
        }
    ) {
        phases.push(ModelLoadPhase::DraftModel);
    }
    if matches!(
        planned.assessed.plan.mtp,
        icn_contracts::MtpConfig::Enabled { .. }
    ) {
        phases.push(ModelLoadPhase::DraftContext);
    }
    if planned.assessed.plan.projector.is_some() {
        phases.push(ModelLoadPhase::Projector);
    }
    phases.extend([
        ModelLoadPhase::Runtime,
        ModelLoadPhase::Warmup,
        ModelLoadPhase::Finalize,
    ]);
    Ok((planned, acceleration, phases))
}

fn timing_plan_identity(config: &ExecutionIntent) -> String {
    let mtp = match &config.mtp {
        icn_contracts::MtpConfig::Disabled { .. } => serde_json::json!({ "enabled": false }),
        icn_contracts::MtpConfig::Enabled {
            source,
            n_max,
            n_min,
            p_min,
            cache_type_k,
            cache_type_v,
        } => serde_json::json!({
            "enabled": true,
            "source": match source {
                icn_contracts::MtpSource::Bundled => "bundled",
                icn_contracts::MtpSource::Separate { .. } => "separate",
            },
            "nMax": n_max,
            "nMin": n_min,
            "pMin": p_min,
            "cacheTypeK": cache_type_k,
            "cacheTypeV": cache_type_v,
        }),
    };
    let projector = config.projector.as_ref().map(|projector| {
        serde_json::json!({
            "useGpu": projector.use_gpu,
            "warmup": projector.warmup,
            "imageMinTokens": projector.image_min_tokens,
            "imageMaxTokens": projector.image_max_tokens,
            "inputLimits": projector.input_limits,
        })
    });
    let evidence = serde_json::json!({
        "contextSize": config.context_size,
        "batchSize": config.batch_size,
        "ubatchSize": config.ubatch_size,
        "maxSequences": config.max_sequences,
        "prefillQuantum": config.prefill_quantum,
        "execution": config.execution,
        "projector": projector,
        "mtp": mtp,
    });
    format!("{:x}", Sha256::digest(evidence.to_string().as_bytes()))
}

fn executor_main(
    backend: Arc<LlamaBackend>,
    planned: icn_hardware::BackendLoadPlan,
    acceleration: String,
    commands: Receiver<ExecutorCommand>,
    ready: SyncSender<Result<(ModelProperties, String), ModelLoadError>>,
    observer: Arc<dyn ModelLoadObserver>,
) {
    #[cfg(feature = "mtmd")]
    let auxiliary_allocations = planned
        .assessed
        .projector_memory
        .iter()
        .map(|estimate| ResidentAllocation {
            location: estimate
                .device_index
                .map_or(LlamaMemoryLocation::Host, |native_index| {
                    LlamaMemoryLocation::Device {
                        backend: String::new(),
                        physical_id: None,
                        native_index,
                    }
                }),
            model_bytes: 0,
            context_bytes: 0,
            compute_bytes: 0,
            auxiliary_bytes: estimate.bytes,
        })
        .collect::<Vec<_>>();
    #[cfg(not(feature = "mtmd"))]
    let auxiliary_allocations = Vec::<ResidentAllocation>::new();
    let config = planned.assessed.plan;
    let native_mtp = planned.native_mtp.map(|plan| plan.into_parts());
    let (model_path, model_params, context_params, threads, threads_batch) =
        planned.native.into_parts();
    let threads = match nonzero_i32(threads, "threads") {
        Ok(value) => value,
        Err(error) => {
            let _ = ready.send(Err(error.into()));
            return;
        }
    };
    let threads_batch = match nonzero_i32(threads_batch, "threads_batch") {
        Ok(value) => value,
        Err(error) => {
            let _ = ready.send(Err(error.into()));
            return;
        }
    };
    let resolved_execution = resolved_execution_config(
        &config.execution,
        model_params.as_ref().get_ref(),
        threads,
        threads_batch,
    );
    observer.phase_started(ModelLoadPhase::TargetModel);
    let model =
        match LlamaModel::load_from_file(&backend, &model_path, model_params.as_ref().get_ref()) {
            Ok(model) => model,
            Err(error) => {
                let _ = ready.send(Err(backend_error(error).into()));
                return;
            }
        };
    observer.phase_completed(ModelLoadPhase::TargetModel);
    // Native model parameters are needed only for weight loading.
    drop(model_params);
    observer.phase_started(ModelLoadPhase::TargetContext);
    let chat_templates = match CommonChatTemplates::from_model(&model) {
        Ok(templates) => templates,
        Err(error) => {
            let _ = ready.send(Err(backend_error(error).into()));
            return;
        }
    };
    let context = match model.new_context(&backend, context_params) {
        Ok(context) => context,
        Err(error) => {
            let _ = ready.send(Err(backend_error(error).into()));
            return;
        }
    };
    observer.phase_completed(ModelLoadPhase::TargetContext);
    let mut context = Some(context);
    let draft_model = match (&config.mtp, native_mtp.as_ref()) {
        (
            icn_contracts::MtpConfig::Enabled {
                source: icn_contracts::MtpSource::Separate { model_path },
                ..
            },
            Some((_, draft_model_params, _, _, _)),
        ) => {
            observer.phase_started(ModelLoadPhase::DraftModel);
            match LlamaModel::load_from_file(
                &backend,
                model_path,
                draft_model_params.as_ref().get_ref(),
            ) {
                Ok(model) => {
                    observer.phase_completed(ModelLoadPhase::DraftModel);
                    Some(model)
                }
                Err(error) => {
                    let _ = ready.send(Err(backend_error(error).into()));
                    return;
                }
            }
        }
        (icn_contracts::MtpConfig::Enabled { .. }, None) => {
            let _ = ready.send(Err(InferenceError::InvalidConfig(
                "native planner omitted the enabled MTP plan".to_owned(),
            )
            .into()));
            return;
        }
        _ => None,
    };
    let draft_has_separate_model = draft_model.is_some();
    let mut mtp = match &config.mtp {
        icn_contracts::MtpConfig::Disabled { .. } => None,
        icn_contracts::MtpConfig::Enabled {
            n_max,
            n_min,
            p_min,
            ..
        } => {
            observer.phase_started(ModelLoadPhase::DraftContext);
            let Some((_, _, draft_context_params, _, _)) = native_mtp.as_ref() else {
                let _ = ready.send(Err(InferenceError::InvalidConfig(
                    "native planner omitted the enabled MTP context".to_owned(),
                )
                .into()));
                return;
            };
            let draft_context_params = draft_context_params.clone();
            let draft_model = draft_model.as_ref().unwrap_or(&model);
            match MtpSession::new_linked(
                context.take().expect("target context is constructed once"),
                draft_model,
                &backend,
                draft_context_params,
                MtpParams {
                    n_max: i32::try_from(*n_max).unwrap_or(i32::MAX),
                    n_min: i32::try_from(*n_min).unwrap_or(i32::MAX),
                    p_min: *p_min,
                },
                config.max_sequences,
            ) {
                Ok(mtp) => {
                    observer.phase_completed(ModelLoadPhase::DraftContext);
                    Some(mtp)
                }
                Err(error) => {
                    let _ = ready.send(Err(backend_error(error).into()));
                    return;
                }
            }
        }
    };
    let mut multimodal = {
        #[cfg(feature = "mtmd")]
        {
            match config.projector.as_ref() {
                Some(projector) => {
                    observer.phase_started(ModelLoadPhase::Projector);
                    match MultimodalRuntime::load(
                        projector,
                        &model,
                        config.execution.flash_attention,
                        Some(threads.get()),
                    ) {
                        Ok(runtime) => {
                            observer.phase_completed(ModelLoadPhase::Projector);
                            Some(runtime)
                        }
                        Err(error) => {
                            let _ = ready.send(Err(error.into()));
                            return;
                        }
                    }
                }
                None => None,
            }
        }
        #[cfg(not(feature = "mtmd"))]
        {
            None::<MultimodalRuntime<'_>>
        }
    };
    observer.phase_started(ModelLoadPhase::Runtime);
    let mut main_pool = match LlamaThreadPool::new(&backend, &LlamaThreadPoolParams::new(threads)) {
        Ok(pool) => pool,
        Err(error) => {
            let _ = ready.send(Err(backend_error(error).into()));
            return;
        }
    };
    if let Some(mtp) = mtp.as_mut() {
        let mut draft_main_pool =
            match LlamaThreadPool::new(&backend, &LlamaThreadPoolParams::new(threads)) {
                Ok(pool) => pool,
                Err(error) => {
                    let _ = ready.send(Err(backend_error(error).into()));
                    return;
                }
            };
        let (context, draft_context, mut operations) = mtp.split_all_mut();
        if threads == threads_batch {
            let mut draft_attached = draft_context.attach_threadpool(&mut draft_main_pool);
            let mut attached = context.attach_threadpool(&mut main_pool);
            observer.phase_completed(ModelLoadPhase::Runtime);
            run_initialized_executor(
                &backend,
                &config,
                resolved_execution,
                &model,
                &chat_templates,
                &mut attached,
                Some(&mut draft_attached),
                draft_has_separate_model,
                &auxiliary_allocations,
                Some(&mut operations),
                &mut multimodal,
                &commands,
                &ready,
                acceleration.clone(),
                observer.as_ref(),
            );
        } else {
            let mut batch_pool =
                match LlamaThreadPool::new(&backend, &LlamaThreadPoolParams::new(threads_batch)) {
                    Ok(pool) => pool,
                    Err(error) => {
                        let _ = ready.send(Err(backend_error(error).into()));
                        return;
                    }
                };
            let mut draft_batch_pool =
                match LlamaThreadPool::new(&backend, &LlamaThreadPoolParams::new(threads_batch)) {
                    Ok(pool) => pool,
                    Err(error) => {
                        let _ = ready.send(Err(backend_error(error).into()));
                        return;
                    }
                };
            let mut draft_attached =
                draft_context.attach_threadpools(&mut draft_main_pool, &mut draft_batch_pool);
            let mut attached = context.attach_threadpools(&mut main_pool, &mut batch_pool);
            observer.phase_completed(ModelLoadPhase::Runtime);
            run_initialized_executor(
                &backend,
                &config,
                resolved_execution,
                &model,
                &chat_templates,
                &mut attached,
                Some(&mut draft_attached),
                draft_has_separate_model,
                &auxiliary_allocations,
                Some(&mut operations),
                &mut multimodal,
                &commands,
                &ready,
                acceleration.clone(),
                observer.as_ref(),
            );
        }
    } else if threads == threads_batch {
        let mut context = context
            .take()
            .expect("non-MTP target context remains owned");
        let mut attached = context.attach_threadpool(&mut main_pool);
        observer.phase_completed(ModelLoadPhase::Runtime);
        run_initialized_executor(
            &backend,
            &config,
            resolved_execution,
            &model,
            &chat_templates,
            &mut attached,
            None,
            false,
            &auxiliary_allocations,
            None,
            &mut multimodal,
            &commands,
            &ready,
            acceleration.clone(),
            observer.as_ref(),
        );
    } else {
        let mut context = context
            .take()
            .expect("non-MTP target context remains owned");
        let mut batch_pool =
            match LlamaThreadPool::new(&backend, &LlamaThreadPoolParams::new(threads_batch)) {
                Ok(pool) => pool,
                Err(error) => {
                    let _ = ready.send(Err(backend_error(error).into()));
                    return;
                }
            };
        let mut attached = context.attach_threadpools(&mut main_pool, &mut batch_pool);
        observer.phase_completed(ModelLoadPhase::Runtime);
        run_initialized_executor(
            &backend,
            &config,
            resolved_execution,
            &model,
            &chat_templates,
            &mut attached,
            None,
            false,
            &auxiliary_allocations,
            None,
            &mut multimodal,
            &commands,
            &ready,
            acceleration,
            observer.as_ref(),
        );
    }
}

fn nonzero_i32(value: NonZeroU32, field: &str) -> Result<NonZeroI32, InferenceError> {
    let value = i32::try_from(value.get())
        .map_err(|_| InferenceError::InvalidConfig(format!("{field} must not exceed i32::MAX")))?;
    Ok(NonZeroI32::new(value).expect("a converted NonZeroU32 remains non-zero"))
}

fn resolved_execution_config(
    requested: &ExecutionConfig,
    model_params: &LlamaModelParams,
    threads: NonZeroI32,
    threads_batch: NonZeroI32,
) -> ExecutionConfig {
    let mut resolved = requested.clone();
    resolved.gpu_layers = match model_params.gpu_layers() {
        LlamaGpuLayers::Auto => GpuLayers::Auto,
        LlamaGpuLayers::All => GpuLayers::All,
        LlamaGpuLayers::Count(value) => GpuLayers::Count(value),
    };
    resolved.tensor_split = trimmed_tensor_split(model_params.tensor_split());
    resolved.threads = NonZeroU32::new(threads.get().cast_unsigned());
    resolved.threads_batch = NonZeroU32::new(threads_batch.get().cast_unsigned());
    resolved
}

fn trimmed_tensor_split(weights: &[f32]) -> Option<Vec<f32>> {
    let last = weights.iter().rposition(|weight| *weight != 0.0)?;
    Some(weights[..=last].to_vec())
}

#[allow(clippy::too_many_arguments)]
fn run_initialized_executor<'model>(
    backend: &LlamaBackend,
    config: &ExecutionIntent,
    resolved_execution: ExecutionConfig,
    model: &'model LlamaModel,
    chat_templates: &CommonChatTemplates,
    context: &mut LlamaContext<'model>,
    draft_context: Option<&mut LlamaContext<'model>>,
    draft_has_separate_model: bool,
    auxiliary_allocations: &[ResidentAllocation],
    mut mtp: Option<&mut MtpOperations<'_>>,
    multimodal: &mut Option<MultimodalRuntime<'model>>,
    commands: &Receiver<ExecutorCommand>,
    ready: &SyncSender<Result<(ModelProperties, String), ModelLoadError>>,
    acceleration: String,
    observer: &dyn ModelLoadObserver,
) {
    observer.phase_started(ModelLoadPhase::Warmup);
    if let Err(error) = warm_up(model, context, mtp.as_deref_mut()) {
        let _ = ready.send(Err(error.into()));
        return;
    }
    observer.phase_completed(ModelLoadPhase::Warmup);
    observer.phase_started(ModelLoadPhase::Finalize);
    let resident_allocations = match capture_resident_allocations(
        context,
        draft_context.as_deref(),
        draft_has_separate_model,
        auxiliary_allocations,
    ) {
        Ok(allocations) => allocations,
        Err(error) => {
            let _ = ready.send(Err(ModelLoadError::MemoryAttribution(error)));
            return;
        }
    };
    let modalities = multimodal
        .as_ref()
        .map_or_else(ModelModalities::default, multimodal_modalities);
    let properties = match model_properties(
        config,
        resolved_execution,
        model,
        context,
        chat_templates,
        modalities,
    ) {
        Ok(properties) => properties,
        Err(error) => {
            let _ = ready.send(Err(error.into()));
            return;
        }
    };
    if ready.send(Ok((properties, acceleration))).is_err() {
        return;
    }
    run_scheduler(
        backend,
        config,
        model,
        chat_templates,
        context,
        draft_context,
        mtp,
        multimodal,
        commands,
        resident_allocations,
    );
}

fn capture_resident_allocations(
    context: &LlamaContext<'_>,
    draft_context: Option<&LlamaContext<'_>>,
    draft_has_separate_model: bool,
    auxiliary_allocations: &[ResidentAllocation],
) -> Result<Vec<ResidentAllocation>, LlamaMemoryBreakdownError> {
    let target = context.memory_breakdown()?;
    let mut allocations = target
        .into_iter()
        .map(ResidentAllocation::from)
        .collect::<Vec<_>>();
    if let Some(draft_context) = draft_context {
        let draft = draft_context.memory_breakdown()?;
        let mut draft = draft
            .into_iter()
            .map(ResidentAllocation::from)
            .collect::<Vec<_>>();
        if !draft_has_separate_model {
            for allocation in &mut draft {
                allocation.model_bytes = 0;
            }
        }
        allocations.extend(draft);
    }
    allocations.extend_from_slice(auxiliary_allocations);
    Ok(allocations)
}

fn resident_memory_state(
    snapshot: &HardwareSnapshot,
    allocations: &[ResidentAllocation],
    model_id: String,
    runtime_generation: u64,
) -> Result<ResidentMemory, HardwareObservationError> {
    let mut domains = BTreeMap::<String, ResidentMemoryDomain>::new();
    for allocation in allocations {
        let location = match &allocation.location {
            LlamaMemoryLocation::Host => icn_hardware::NativeMemoryLocation::Host,
            LlamaMemoryLocation::Device {
                backend,
                physical_id,
                native_index,
                ..
            } => icn_hardware::NativeMemoryLocation::Device {
                backend: backend.clone(),
                physical_id: physical_id.clone(),
                native_index: *native_index,
            },
        };
        let domain_id =
            icn_hardware::resolve_memory_domain(snapshot, &location).ok_or_else(|| {
                HardwareObservationError::MemoryDomainUnresolved {
                    location: format!("{location:?}"),
                }
            })?;
        let domain = domains
            .entry(domain_id.to_owned())
            .or_insert_with(|| ResidentMemoryDomain {
                memory_domain_id: domain_id.to_owned(),
                model_bytes: 0,
                context_bytes: 0,
                compute_bytes: 0,
                auxiliary_bytes: 0,
            });
        domain.model_bytes = domain.model_bytes.saturating_add(allocation.model_bytes);
        domain.context_bytes = domain
            .context_bytes
            .saturating_add(allocation.context_bytes);
        domain.compute_bytes = domain
            .compute_bytes
            .saturating_add(allocation.compute_bytes);
        domain.auxiliary_bytes = domain
            .auxiliary_bytes
            .saturating_add(allocation.auxiliary_bytes);
    }
    Ok(ResidentMemory {
        model_id,
        runtime_generation,
        domains: domains.into_values().collect(),
    })
}

// The scheduler composition root receives each owned runtime subsystem explicitly. Keeping these
// borrows visible is clearer than hiding them behind a second mutable service-locator struct.
#[allow(clippy::too_many_arguments)]
fn run_scheduler<'model>(
    backend: &LlamaBackend,
    config: &ExecutionIntent,
    model: &'model LlamaModel,
    chat_templates: &CommonChatTemplates,
    context: &mut LlamaContext<'model>,
    mut draft_context: Option<&mut LlamaContext<'model>>,
    mut mtp: Option<&mut MtpOperations<'_>>,
    multimodal: &mut Option<MultimodalRuntime<'model>>,
    commands: &Receiver<ExecutorCommand>,
    resident_allocations: Vec<ResidentAllocation>,
) {
    let mut sequence_pool = SequencePool::new(config.max_sequences);
    let mut planner = BatchPlanner::new(config.prefill_quantum as usize);
    let mut decode_buffer = LlamaBatch::new(context.n_batch() as usize, 1);
    let mut queued = VecDeque::<QueuedCompletion>::new();
    let mut exclusive_native = VecDeque::<ExclusiveNativeTask>::new();
    let mut hardware_observations = VecDeque::<HardwareObservationRequest>::new();
    let mut active = Vec::<ActiveRequest<'_>>::new();
    let mut shutting_down = false;
    let max_tracked = COMMAND_QUEUE_CAPACITY + config.max_sequences as usize;

    loop {
        drain_commands(
            commands,
            chat_templates,
            multimodal.as_ref().map(multimodal_marker),
            &mut queued,
            &mut exclusive_native,
            &mut hardware_observations,
            &active,
            max_tracked,
            &mut shutting_down,
        );

        if let Some(observation) = hardware_observations.pop_front() {
            let mut snapshot = icn_hardware::discover_hardware(
                backend,
                observation.policy,
                observation.native_build,
                observation.enabled_backends,
            );
            let observed = resident_memory_state(
                &snapshot,
                &resident_allocations,
                observation.model_id,
                observation.runtime_generation,
            )
            .map(|resident_memory| {
                snapshot.resident_memory = Some(resident_memory);
                snapshot
            });
            let _ = observation.response.try_send(observed);
        }

        if active.is_empty()
            && queued.is_empty()
            && !shutting_down
            && let Some(task) = exclusive_native.pop_front()
        {
            task(backend);
            continue;
        }

        if active.is_empty() && !queued.is_empty() && !shutting_down {
            let deadline = Instant::now() + IDLE_ADMISSION_COALESCE_INTERVAL;
            while queued.len() < config.max_sequences as usize {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    break;
                }
                match commands.recv_timeout(remaining) {
                    Ok(command) => handle_command(
                        command,
                        chat_templates,
                        multimodal.as_ref().map(multimodal_marker),
                        &mut queued,
                        &mut exclusive_native,
                        &mut hardware_observations,
                        0,
                        max_tracked,
                        &mut shutting_down,
                    ),
                    Err(RecvTimeoutError::Timeout) => break,
                    Err(RecvTimeoutError::Disconnected) => {
                        shutting_down = true;
                        break;
                    }
                }
            }
        }

        cleanup_requests(context, mtp.as_deref_mut(), &mut sequence_pool, &mut active);

        if shutting_down {
            fail_queued(&mut queued, InferenceError::ExecutorStopped);
            fail_active(
                context,
                mtp.as_deref_mut(),
                &mut sequence_pool,
                &mut active,
                InferenceError::ExecutorStopped,
            );
            cleanup_requests(context, mtp.as_deref_mut(), &mut sequence_pool, &mut active);
            if active.is_empty() {
                break;
            }
        } else {
            admit_requests(
                model,
                chat_templates,
                multimodal.as_ref(),
                context,
                draft_context.as_deref_mut(),
                mtp.as_deref_mut(),
                &mut sequence_pool,
                &mut queued,
                &mut active,
                config.context_size as usize,
            );
        }

        sample_ready_requests(
            model,
            context,
            mtp.as_deref_mut(),
            &mut sequence_pool,
            &mut active,
        );
        cleanup_requests(context, mtp.as_deref_mut(), &mut sequence_pool, &mut active);

        let decoded = if shutting_down {
            false
        } else {
            match decode_batch(
                model,
                context,
                draft_context.as_deref_mut(),
                mtp.as_deref_mut(),
                multimodal,
                &mut planner,
                &mut decode_buffer,
                &mut active,
            ) {
                Ok(decoded) => decoded,
                Err(error) => {
                    // A failed decode can leave shared native memory in an uncertain state. Fail
                    // every resident request and reset the whole context before admitting more
                    // work rather than guessing which sequence committed.
                    context.synchronize();
                    context.clear_memory(false);
                    if let Some(draft_context) = draft_context.as_deref_mut() {
                        draft_context.synchronize();
                        draft_context.clear_memory(false);
                    }
                    let failure = if matches!(error, InferenceError::Cancelled) {
                        InferenceError::Cancelled
                    } else {
                        InferenceError::Backend(error.to_string())
                    };
                    fail_active(
                        context,
                        mtp.as_deref_mut(),
                        &mut sequence_pool,
                        &mut active,
                        failure,
                    );
                    false
                }
            }
        };

        cleanup_requests(context, mtp.as_deref_mut(), &mut sequence_pool, &mut active);

        if !decoded {
            match commands.recv_timeout(IDLE_POLL_INTERVAL) {
                Ok(command) => handle_command(
                    command,
                    chat_templates,
                    multimodal.as_ref().map(multimodal_marker),
                    &mut queued,
                    &mut exclusive_native,
                    &mut hardware_observations,
                    active.len(),
                    max_tracked,
                    &mut shutting_down,
                ),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => shutting_down = true,
            }
        }
    }
}

// Command dispatch keeps completion admission and exclusive native work as distinct bounded
// queues; listing both mutable destinations makes their ordering and ownership explicit.
#[allow(clippy::too_many_arguments)]
fn drain_commands(
    commands: &Receiver<ExecutorCommand>,
    chat_templates: &CommonChatTemplates,
    media_marker: Option<&str>,
    queued: &mut VecDeque<QueuedCompletion>,
    exclusive_native: &mut VecDeque<ExclusiveNativeTask>,
    hardware_observations: &mut VecDeque<HardwareObservationRequest>,
    active: &[ActiveRequest<'_>],
    max_tracked: usize,
    shutting_down: &mut bool,
) {
    for _ in 0..COMMAND_QUEUE_CAPACITY {
        match commands.try_recv() {
            Ok(command) => handle_command(
                command,
                chat_templates,
                media_marker,
                queued,
                exclusive_native,
                hardware_observations,
                active.len(),
                max_tracked,
                shutting_down,
            ),
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                *shutting_down = true;
                break;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_command(
    command: ExecutorCommand,
    chat_templates: &CommonChatTemplates,
    media_marker: Option<&str>,
    queued: &mut VecDeque<QueuedCompletion>,
    exclusive_native: &mut VecDeque<ExclusiveNativeTask>,
    hardware_observations: &mut VecDeque<HardwareObservationRequest>,
    active_count: usize,
    max_tracked: usize,
    shutting_down: &mut bool,
) {
    match command {
        ExecutorCommand::Complete {
            request,
            events,
            cancelled,
            queued_at,
            span,
        } => {
            let entered_span = span.clone();
            let _entered = entered_span.enter();
            if *shutting_down {
                let _ = events.try_send(ExecutorItem::Failed(InferenceError::ExecutorStopped));
            } else if queued.len() + active_count >= max_tracked {
                let _ = events.try_send(ExecutorItem::Failed(InferenceError::Overloaded));
            } else {
                queued.push_back(QueuedCompletion {
                    request,
                    events,
                    cancelled,
                    queued_at,
                    span,
                });
            }
        }
        ExecutorCommand::ApplyTemplate {
            request,
            response,
            span,
        } => {
            let _entered = span.enter();
            let result = if *shutting_down {
                Err(InferenceError::ExecutorStopped)
            } else {
                prepare_chat(chat_templates, &request, media_marker)
                    .and_then(|prepared| prepared_chat_info(chat_templates, &prepared))
            };
            let _ = response.send(result);
        }
        ExecutorCommand::RunExclusiveNative { task } => {
            if !*shutting_down {
                exclusive_native.push_back(task);
            }
        }
        ExecutorCommand::ObserveHardware(observation) => {
            if *shutting_down {
                drop(observation.response);
            } else {
                hardware_observations.push_back(observation);
            }
        }
        ExecutorCommand::Shutdown => *shutting_down = true,
    }
}

#[allow(clippy::too_many_arguments)]
fn admit_requests<'model>(
    model: &'model LlamaModel,
    chat_templates: &CommonChatTemplates,
    multimodal: Option<&MultimodalRuntime<'model>>,
    context: &mut LlamaContext<'model>,
    mut draft_context: Option<&mut LlamaContext<'model>>,
    mut mtp: Option<&mut MtpOperations<'_>>,
    sequence_pool: &mut SequencePool,
    queued: &mut VecDeque<QueuedCompletion>,
    active: &mut Vec<ActiveRequest<'model>>,
    shared_context_capacity: usize,
) {
    while !queued.is_empty() {
        let matching_prompt = queued.front().and_then(|queued| {
            (queued.request.cache_prompt && request_images(&queued.request.template).is_empty())
                .then(|| {
                    prepare_chat(
                        chat_templates,
                        &queued.request.template,
                        multimodal.map(multimodal_marker),
                    )
                    .and_then(|prepared| plain_prompt(model, &prepared))
                    .map(|prompt| prompt.text_tokens)
                    .ok()
                })
                .flatten()
        });
        let sequence_id = match matching_prompt.as_deref() {
            Some(prompt) => sequence_pool.acquire_matching(prompt),
            None => sequence_pool.acquire(),
        };
        let Some(sequence_id) = sequence_id else {
            break;
        };
        let queued_request = queued
            .pop_front()
            .expect("queue was checked before acquiring a sequence");
        if queued_request.cancelled.load(Ordering::Acquire) {
            let _ = queued_request
                .events
                .try_send(ExecutorItem::Failed(InferenceError::Cancelled));
            sequence_pool.release(sequence_id);
            continue;
        }
        let cached = sequence_pool.take_cache(sequence_id);
        match ActiveRequest::admit(
            model,
            chat_templates,
            multimodal,
            shared_context_capacity,
            context.n_batch() as usize,
            context.n_ubatch() as usize,
            sequence_id,
            queued_request,
            cached.as_ref(),
        ) {
            Ok(mut request) => {
                let requested_start = request.prompt_offset;
                let partial = clear_sequence_range(
                    context,
                    mtp.as_deref_mut(),
                    sequence_id,
                    i32::try_from(requested_start).unwrap_or(i32::MAX),
                    -1,
                );
                if partial.is_err() && requested_start > 0 {
                    let checkpoint = cached.as_ref().and_then(|cache| {
                        cache
                            .checkpoints
                            .iter()
                            .rev()
                            .find(|checkpoint| checkpoint.prefix <= requested_start)
                    });
                    let restored = checkpoint.is_some_and(|checkpoint| {
                        restore_prompt_checkpoint(
                            context,
                            draft_context.as_deref_mut(),
                            sequence_id,
                            checkpoint,
                        )
                    });
                    request.prompt_offset = checkpoint
                        .filter(|_| restored)
                        .map_or(0, |value| value.prefix);
                    request.cached_prompt_tokens = request.prompt_offset;
                    if clear_sequence_range(
                        context,
                        mtp.as_deref_mut(),
                        sequence_id,
                        i32::try_from(request.prompt_offset).unwrap_or(i32::MAX),
                        -1,
                    )
                    .is_err()
                    {
                        let _ =
                            request
                                .events
                                .try_send(ExecutorItem::Failed(InferenceError::Backend(format!(
                                    "llama.cpp refused to reset cached sequence {sequence_id}"
                                ))));
                        sequence_pool.quarantine(sequence_id);
                        continue;
                    }
                }
                active.push(request);
            }
            Err((events, error)) => {
                let _ = events.try_send(ExecutorItem::Failed(error));
                if clear_sequence(context, mtp.as_deref_mut(), sequence_id).is_ok() {
                    sequence_pool.release(sequence_id);
                } else {
                    sequence_pool.quarantine(sequence_id);
                }
            }
        }
    }
}

fn sample_ready_requests<'model>(
    model: &'model LlamaModel,
    context: &mut LlamaContext<'model>,
    mut mtp: Option<&mut MtpOperations<'_>>,
    sequence_pool: &mut SequencePool,
    active: &mut [ActiveRequest<'model>],
) {
    for request in active {
        let RequestPhase::ReadyToSample { batch_index } = request.phase else {
            continue;
        };
        let current_span = request.span.clone();
        let _entered = current_span.enter();
        if request.cancelled.load(Ordering::Acquire) {
            cancel_request(request);
            release_sequence(context, mtp.as_deref_mut(), sequence_pool, request);
            continue;
        }
        if !request.mtp_started
            && request.multimodal_prompt.is_none()
            && let Some(operations) = mtp.as_deref_mut()
        {
            let sequence_id = request.sequence_id.expect("ready request owns a sequence");
            if let Err(error) = operations.begin(sequence_id, &request.cache_history) {
                fail_request(request, backend_error(error));
                release_sequence(context, mtp.as_deref_mut(), sequence_pool, request);
                continue;
            }
            request.mtp_started = true;
        }
        match request.sample_next(model, context, batch_index) {
            Ok(Some(reason)) => {
                if let Err(error) = request.complete(reason) {
                    fail_request(request, error);
                }
                release_sequence(context, mtp.as_deref_mut(), sequence_pool, request);
            }
            Ok(None) => {}
            Err(error) => {
                fail_request(request, error);
                release_sequence(context, mtp.as_deref_mut(), sequence_pool, request);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_batch<'model>(
    model: &'model LlamaModel,
    context: &mut LlamaContext<'model>,
    mut draft_context: Option<&mut LlamaContext<'model>>,
    mut mtp: Option<&mut MtpOperations<'_>>,
    multimodal: &mut Option<MultimodalRuntime<'model>>,
    planner: &mut BatchPlanner,
    batch: &mut LlamaBatch<'_>,
    active: &mut [ActiveRequest<'model>],
) -> Result<bool, InferenceError> {
    if decode_multimodal_prefill(context, multimodal, active)? {
        return Ok(true);
    }

    let can_checkpoint_prompt = mtp.is_none() || draft_context.is_some();
    if can_checkpoint_prompt {
        for request in active.iter_mut().filter(|request| {
            request.cache_prompt
                && request.cacheable
                && matches!(request.phase, RequestPhase::Prefill)
                && request.pending_checkpoint_prefixes.front().copied()
                    == Some(request.prompt_offset)
        }) {
            let prefix = request
                .pending_checkpoint_prefixes
                .pop_front()
                .expect("checkpoint position was matched");
            let sequence_id = request
                .sequence_id
                .expect("prefill request owns a sequence");
            let target_checkpoint = context
                .capture_sequence_state(sequence_id, LlamaStateSeqFlags::PARTIAL_ONLY)
                .ok()
                .filter(|checkpoint| !checkpoint.is_empty());
            let draft_checkpoint = draft_context.as_deref_mut().and_then(|draft_context| {
                draft_context
                    .capture_sequence_state(sequence_id, LlamaStateSeqFlags::PARTIAL_ONLY)
                    .ok()
                    .filter(|checkpoint| !checkpoint.is_empty())
            });
            if let Some(target) = target_checkpoint
                && (mtp.is_none() || draft_checkpoint.is_some())
            {
                request.prompt_checkpoints.push(PromptCheckpoint {
                    target,
                    draft: draft_checkpoint,
                    prefix,
                });
                request
                    .prompt_checkpoints
                    .sort_by_key(|checkpoint| checkpoint.prefix);
                if request.prompt_checkpoints.len() > 32 {
                    request.prompt_checkpoints.remove(0);
                }
            }
        }
    }

    let mut draft_extra_tokens = 0_usize;
    if let Some(operations) = mtp.as_mut() {
        let mut drafted_sequences = Vec::new();
        let started = Instant::now();
        let decode_count = active
            .iter()
            .filter(|request| {
                request.sequence_id.is_some()
                    && request.outbound.is_empty()
                    && matches!(request.phase, RequestPhase::Decode { .. })
            })
            .count();
        let mut extra_budget = (context.n_batch() as usize).saturating_sub(decode_count);
        for request in active.iter_mut().filter(|request| {
            request.mtp_started
                && request.mtp_draft.is_empty()
                && request.sequence_id.is_some()
                && request.outbound.is_empty()
                && matches!(request.phase, RequestPhase::Decode { .. })
        }) {
            let RequestPhase::Decode { token, position } = request.phase else {
                unreachable!()
            };
            let remaining = request
                .generation_limit
                .saturating_sub(request.generated_tokens);
            let n_max = remaining
                .min(operations.max_draft_tokens())
                .min(extra_budget);
            if n_max == 0 {
                continue;
            }
            let sequence_id = request.sequence_id.expect("selected request owns sequence");
            operations
                .prepare_draft(sequence_id, position, token, &request.cache_history, n_max)
                .map_err(backend_error)?;
            extra_budget -= n_max;
            drafted_sequences.push(sequence_id);
        }
        if !drafted_sequences.is_empty() {
            operations.draft_all().map_err(backend_error)?;
            let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
            for sequence_id in drafted_sequences {
                let request = request_by_sequence(active, sequence_id)?;
                request.mtp_draft = operations.take_draft(sequence_id).map_err(backend_error)?;
                let RequestPhase::Decode { position, .. } = request.phase else {
                    return Err(InferenceError::Backend(
                        "MTP request left decode state while drafting".into(),
                    ));
                };
                // MTP autoregressively advances its draft context while producing the
                // proposal. The target verification batch will mirror the sampled token
                // and accepted proposal back into that context, so discard the temporary
                // speculative suffix first. This is the same transition performed by
                // llama.cpp's server immediately after common_speculative_draft().
                operations
                    .remove_sequence_range(sequence_id, position, -1)
                    .map_err(backend_error)?;
                draft_extra_tokens = draft_extra_tokens.saturating_add(request.mtp_draft.len());
                request.draft_tokens = request.draft_tokens.saturating_add(request.mtp_draft.len());
                request.draft_ms += elapsed;
            }
        }
    }

    let candidates = active
        .iter()
        .filter(|request| {
            request.sequence_id.is_some()
                && request.outbound.is_empty()
                && !request.cancelled.load(Ordering::Acquire)
        })
        .filter_map(|request| {
            let sequence_id = request.sequence_id?;
            let kind = match request.phase {
                RequestPhase::Prefill => WorkKind::Prefill {
                    remaining: request
                        .pending_checkpoint_prefixes
                        .front()
                        .filter(|_| can_checkpoint_prompt && request.cache_prompt)
                        .map_or_else(
                            || request.prompt.len().saturating_sub(request.prompt_offset),
                            |prefix| prefix.saturating_sub(request.prompt_offset),
                        ),
                },
                RequestPhase::Decode { .. } => WorkKind::Decode,
                RequestPhase::ReadyToSample { .. } | RequestPhase::Terminal => return None,
            };
            Some(WorkCandidate { sequence_id, kind })
        })
        .collect::<Vec<_>>();
    let plan = planner.plan(
        &candidates,
        (context.n_batch() as usize).saturating_sub(draft_extra_tokens),
    );
    if plan.is_empty() {
        return Ok(false);
    }
    batch.clear();
    let mut logits = Vec::<(i32, i32)>::new();
    let batch_started = Instant::now();

    for work in plan {
        match work {
            BatchWork::Decode { sequence_id } => {
                let request = request_by_sequence(active, sequence_id)?;
                let RequestPhase::Decode { token, position } = request.phase else {
                    return Err(InferenceError::Backend(format!(
                        "scheduler selected sequence {sequence_id} for decode in the wrong state"
                    )));
                };
                request.mtp_indices.clear();
                batch
                    .add(token, position, &[sequence_id], true)
                    .map_err(backend_error)?;
                request.mtp_indices.push(batch.n_tokens() - 1);
                if request.mtp_draft.is_empty() {
                    logits.push((sequence_id, batch.n_tokens() - 1));
                } else {
                    for (offset, draft) in request.mtp_draft.iter().copied().enumerate() {
                        let draft_position = position
                            .checked_add(i32::try_from(offset + 1).map_err(backend_error)?)
                            .ok_or_else(|| {
                                InferenceError::Backend(
                                    "speculative position exceeded i32::MAX".into(),
                                )
                            })?;
                        batch
                            .add(draft, draft_position, &[sequence_id], true)
                            .map_err(backend_error)?;
                        request.mtp_indices.push(batch.n_tokens() - 1);
                    }
                }
            }
            BatchWork::Prefill {
                sequence_id,
                tokens,
            } => {
                let request = request_by_sequence(active, sequence_id)?;
                request.prompt_started_at.get_or_insert(batch_started);
                let start = request.prompt_offset;
                let end = start + tokens;
                for (relative, token) in request.prompt[start..end].iter().enumerate() {
                    let absolute = start + relative;
                    let final_prompt_token = absolute + 1 == request.prompt.len();
                    batch
                        .add(
                            *token,
                            i32::try_from(absolute).map_err(backend_error)?,
                            &[sequence_id],
                            final_prompt_token,
                        )
                        .map_err(backend_error)?;
                    if final_prompt_token {
                        logits.push((sequence_id, batch.n_tokens() - 1));
                    }
                }
                request.prompt_offset = end;
            }
        }
    }

    let verification_started = Instant::now();
    context.decode(batch).map_err(backend_error)?;
    if let Some(operations) = mtp.as_mut() {
        operations.process(batch).map_err(backend_error)?;
    }
    let verification_ms = verification_started.elapsed().as_secs_f64() * 1_000.0;
    for (sequence_id, batch_index) in logits {
        let request = request_by_sequence(active, sequence_id)?;
        if let RequestPhase::Decode { token, .. } = request.phase {
            request.cache_history.push(token);
        }
        request.phase = RequestPhase::ReadyToSample { batch_index };
    }
    if let Some(operations) = mtp.as_mut() {
        verify_mtp_batch(model, context, operations, active, verification_ms)?;
    }
    Ok(true)
}

fn verify_mtp_batch<'model>(
    model: &'model LlamaModel,
    context: &mut LlamaContext<'model>,
    operations: &mut MtpOperations<'_>,
    active: &mut [ActiveRequest<'model>],
    verification_ms: f64,
) -> Result<(), InferenceError> {
    for request in active
        .iter_mut()
        .filter(|request| !request.mtp_draft.is_empty())
    {
        let sequence_id = request
            .sequence_id
            .ok_or_else(|| InferenceError::Backend("MTP request lost its sequence".into()))?;
        let RequestPhase::Decode {
            token: pending,
            position: _,
        } = request.phase
        else {
            return Err(InferenceError::Backend(
                "MTP verification request was not in decode state".into(),
            ));
        };
        let accepted = request
            .sampler
            .sample_and_accept_n(context, &request.mtp_indices, &request.mtp_draft, false)
            .map_err(backend_error)?;
        if accepted.is_empty() {
            return Err(InferenceError::Backend(
                "native speculative verification returned no continuation token".into(),
            ));
        }
        let accepted_drafts = accepted.len() - 1;
        operations
            .accept(
                sequence_id,
                u16::try_from(accepted_drafts).map_err(backend_error)?,
            )
            .map_err(backend_error)?;

        request.cache_history.push(pending);
        request
            .cache_history
            .extend(accepted.iter().take(accepted_drafts).copied());
        let next_position = i32::try_from(request.cache_history.len()).map_err(backend_error)?;
        operations
            .remove_sequence_range(sequence_id, next_position, -1)
            .map_err(backend_error)?;
        request.accepted_draft_tokens = request
            .accepted_draft_tokens
            .saturating_add(accepted_drafts);
        request.verification_ms += verification_ms;
        request.mtp_draft.clear();
        request.mtp_indices.clear();

        let mut terminal = None;
        for token in accepted.iter().copied() {
            if let Some(reason) = request.emit_accepted_token(model, token)? {
                terminal = Some(reason);
                break;
            }
        }
        if let Some(reason) = terminal {
            request.complete(reason)?;
            request.phase = RequestPhase::Terminal;
        } else {
            let continuation = *accepted.last().expect("checked non-empty");
            request.phase = RequestPhase::Decode {
                token: continuation,
                position: next_position,
            };
            request.next_position = next_position.checked_add(1).ok_or_else(|| {
                InferenceError::Backend("generation position exceeded i32::MAX".into())
            })?;
        }
    }
    Ok(())
}

#[cfg(feature = "mtmd")]
fn decode_multimodal_prefill<'model>(
    context: &mut LlamaContext<'model>,
    multimodal: &mut Option<MultimodalRuntime<'model>>,
    active: &mut [ActiveRequest<'model>],
) -> Result<bool, InferenceError> {
    let Some(index) = active.iter().position(|request| {
        request.multimodal_prompt.is_some()
            && matches!(request.phase, RequestPhase::Prefill)
            && request.sequence_id.is_some()
            && request.outbound.is_empty()
            && !request.cancelled.load(Ordering::Acquire)
    }) else {
        return Ok(false);
    };
    if active
        .iter()
        .filter(|request| request.sequence_id.is_some())
        .count()
        != 1
    {
        return Err(InferenceError::Backend(
            "multimodal evaluation cannot share a context with another resident sequence".into(),
        ));
    }
    let runtime = multimodal.as_mut().ok_or_else(|| {
        InferenceError::Backend("multimodal request was admitted without a projector".into())
    })?;
    let request = &mut active[index];
    let sequence_id = request
        .sequence_id
        .ok_or_else(|| InferenceError::Backend("multimodal sequence lost ownership".into()))?;
    let batch_size = i32::try_from(context.n_batch()).map_err(backend_error)?;
    let started = Instant::now();
    request.prompt_started_at.get_or_insert(started);
    context.install_abort_callback_with_flag(Arc::clone(&request.cancelled));
    let result = runtime.evaluate_prompt(
        request
            .multimodal_prompt
            .as_ref()
            .expect("multimodal request was selected"),
        context,
        sequence_id,
        batch_size,
    );
    context.clear_abort_callback();
    if request.cancelled.load(Ordering::Acquire) {
        return Err(InferenceError::Cancelled);
    }
    let next_position = result?;
    request.next_position = next_position;
    request.prompt_offset = request.prompt.len();
    request.phase = RequestPhase::ReadyToSample { batch_index: -1 };
    Ok(true)
}

#[cfg(not(feature = "mtmd"))]
fn decode_multimodal_prefill<'model>(
    _context: &mut LlamaContext<'model>,
    _multimodal: &mut Option<MultimodalRuntime<'model>>,
    active: &mut [ActiveRequest<'model>],
) -> Result<bool, InferenceError> {
    if active
        .iter()
        .any(|request| request.multimodal_prompt.is_some())
    {
        return Err(InferenceError::Backend(
            "multimodal support was not compiled into this ICN binary".into(),
        ));
    }
    Ok(false)
}

fn request_by_sequence<'a, 'model>(
    active: &'a mut [ActiveRequest<'model>],
    sequence_id: i32,
) -> Result<&'a mut ActiveRequest<'model>, InferenceError> {
    active
        .iter_mut()
        .find(|request| request.sequence_id == Some(sequence_id))
        .ok_or_else(|| {
            InferenceError::Backend(format!(
                "scheduler referenced unowned sequence {sequence_id}"
            ))
        })
}

fn common_token_prefix(left: &[LlamaToken], right: &[LlamaToken]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

fn restore_prompt_checkpoint(
    context: &mut LlamaContext<'_>,
    draft_context: Option<&mut LlamaContext<'_>>,
    sequence_id: i32,
    checkpoint: &PromptCheckpoint,
) -> bool {
    if let Some(draft_context) = draft_context {
        let Some(draft_checkpoint) = checkpoint.draft.as_ref() else {
            return false;
        };
        if !draft_context.restore_sequence_state(draft_checkpoint, sequence_id) {
            return false;
        }
    }
    context.restore_sequence_state(&checkpoint.target, sequence_id)
}

fn cleanup_requests(
    context: &mut LlamaContext<'_>,
    mut mtp: Option<&mut MtpOperations<'_>>,
    sequence_pool: &mut SequencePool,
    active: &mut Vec<ActiveRequest<'_>>,
) {
    let mut index = 0;
    while index < active.len() {
        if active[index].cancelled.load(Ordering::Acquire)
            && !matches!(active[index].phase, RequestPhase::Terminal)
        {
            cancel_request(&mut active[index]);
            release_sequence(
                context,
                mtp.as_deref_mut(),
                sequence_pool,
                &mut active[index],
            );
        }
        match flush_outbound(&mut active[index]) {
            FlushOutcome::Empty if matches!(active[index].phase, RequestPhase::Terminal) => {
                release_sequence(
                    context,
                    mtp.as_deref_mut(),
                    sequence_pool,
                    &mut active[index],
                );
                if active[index].outbound.is_empty() {
                    active.remove(index);
                } else {
                    index += 1;
                }
            }
            FlushOutcome::Disconnected => {
                release_sequence(
                    context,
                    mtp.as_deref_mut(),
                    sequence_pool,
                    &mut active[index],
                );
                active.remove(index);
            }
            FlushOutcome::Empty | FlushOutcome::Backpressured => index += 1,
        }
    }
}

fn flush_outbound(request: &mut ActiveRequest<'_>) -> FlushOutcome {
    while let Some(item) = request.outbound.pop_front() {
        match request.events.try_send(item) {
            Ok(()) => {}
            Err(TrySendError::Full(item)) => {
                request.outbound.push_front(item);
                return FlushOutcome::Backpressured;
            }
            Err(TrySendError::Disconnected(_)) => {
                request.outbound.clear();
                return FlushOutcome::Disconnected;
            }
        }
    }
    FlushOutcome::Empty
}

fn release_sequence(
    context: &mut LlamaContext<'_>,
    mtp: Option<&mut MtpOperations<'_>>,
    sequence_pool: &mut SequencePool,
    request: &mut ActiveRequest<'_>,
) {
    let Some(sequence_id) = request.sequence_id.take() else {
        return;
    };
    if request.cache_prompt && request.cacheable {
        sequence_pool.release_cached(
            sequence_id,
            SequenceCache {
                prompt: request.prompt.clone(),
                checkpoints: std::mem::take(&mut request.prompt_checkpoints),
            },
        );
        return;
    }
    // Full sequence removal is supported for every llama.cpp memory implementation. This is the
    // sole cache policy required by this milestone: a sequence is never reassigned while resident
    // state still belongs to the previous request.
    match clear_sequence(context, mtp, sequence_id) {
        Ok(()) => sequence_pool.release(sequence_id),
        Err(error) => {
            // Never hand a sequence to another request unless native state removal succeeded.
            sequence_pool.quarantine(sequence_id);
            request.phase = RequestPhase::Terminal;
            request.outbound.clear();
            request.outbound.push_back(ExecutorItem::Failed(error));
        }
    }
}

fn clear_sequence(
    context: &mut LlamaContext<'_>,
    mtp: Option<&mut MtpOperations<'_>>,
    sequence_id: i32,
) -> Result<(), InferenceError> {
    clear_sequence_range(context, mtp, sequence_id, 0, -1)
}

fn clear_sequence_range(
    context: &mut LlamaContext<'_>,
    mtp: Option<&mut MtpOperations<'_>>,
    sequence_id: i32,
    start: i32,
    end: i32,
) -> Result<(), InferenceError> {
    if let Some(mtp) = mtp {
        return mtp
            .remove_sequence_range(sequence_id, start, end)
            .map_err(backend_error);
    }
    let sequence = u32::try_from(sequence_id).map_err(backend_error)?;
    let removed = context
        .clear_kv_cache_seq(
            Some(sequence),
            (start > 0).then_some(start as u32),
            (end >= 0).then_some(end as u32),
        )
        .map_err(backend_error)?;
    if removed {
        Ok(())
    } else {
        Err(InferenceError::Backend(format!(
            "llama.cpp refused to fully remove sequence {sequence_id}"
        )))
    }
}

fn fail_request(request: &mut ActiveRequest<'_>, error: InferenceError) {
    request.phase = RequestPhase::Terminal;
    request.cacheable = false;
    if request.outbound.len() >= OUTBOUND_QUEUE_CAPACITY {
        request.outbound.clear();
    }
    request.outbound.push_back(ExecutorItem::Failed(error));
}

fn cancel_request(request: &mut ActiveRequest<'_>) {
    request.outbound.clear();
    request.cacheable = false;
    fail_request(request, InferenceError::Cancelled);
}

fn fail_queued(queued: &mut VecDeque<QueuedCompletion>, reason: InferenceError) {
    while let Some(request) = queued.pop_front() {
        let _ = request
            .events
            .try_send(ExecutorItem::Failed(clone_inference_error(&reason)));
    }
}

fn fail_active(
    context: &mut LlamaContext<'_>,
    mut mtp: Option<&mut MtpOperations<'_>>,
    sequence_pool: &mut SequencePool,
    active: &mut [ActiveRequest<'_>],
    reason: InferenceError,
) {
    for request in active {
        if !matches!(request.phase, RequestPhase::Terminal) {
            fail_request(request, clone_inference_error(&reason));
        }
        release_sequence(context, mtp.as_deref_mut(), sequence_pool, request);
    }
}

fn clone_inference_error(error: &InferenceError) -> InferenceError {
    match error {
        InferenceError::InvalidConfig(message) => InferenceError::InvalidConfig(message.clone()),
        InferenceError::Backend(message) => InferenceError::Backend(message.clone()),
        InferenceError::Cancelled => InferenceError::Cancelled,
        InferenceError::Overloaded => InferenceError::Overloaded,
        InferenceError::ExecutorStopped => InferenceError::ExecutorStopped,
        InferenceError::Callback(message) => InferenceError::Callback(message.clone()),
    }
}

fn validate_model_config(config: &ExecutionIntent) -> Result<(), InferenceError> {
    if !config.model_path.is_file() {
        return Err(InferenceError::InvalidConfig(format!(
            "GGUF model does not exist: {}",
            config.model_path.display()
        )));
    }
    if config.context_size == 0 {
        return Err(InferenceError::InvalidConfig(
            "context_size must be greater than zero".into(),
        ));
    }
    if config.batch_size == 0 {
        return Err(InferenceError::InvalidConfig(
            "batch_size must be greater than zero".into(),
        ));
    }
    if config.ubatch_size == 0 || config.ubatch_size > config.batch_size {
        return Err(InferenceError::InvalidConfig(
            "ubatch_size must be greater than zero and no larger than batch_size".into(),
        ));
    }
    if config.max_sequences == 0 || config.max_sequences > i32::MAX as u32 {
        return Err(InferenceError::InvalidConfig(
            "max_sequences must be between 1 and i32::MAX".into(),
        ));
    }
    if config.context_size < config.max_sequences {
        return Err(InferenceError::InvalidConfig(
            "context_size must provide at least one token per sequence".into(),
        ));
    }
    if config.prefill_quantum == 0 || config.prefill_quantum > config.batch_size {
        return Err(InferenceError::InvalidConfig(
            "prefill_quantum must be greater than zero and no larger than batch_size".into(),
        ));
    }
    if matches!(config.execution.gpu_layers, GpuLayers::Count(value) if value > i32::MAX as u32) {
        return Err(InferenceError::InvalidConfig(
            "an explicit GPU-layer count must not exceed i32::MAX; use 'all' for full offload"
                .into(),
        ));
    }
    if config.execution.split_mode == SplitMode::None && config.execution.tensor_split.is_some() {
        return Err(InferenceError::InvalidConfig(
            "tensor_split requires split_mode layer, row, or tensor".into(),
        ));
    }
    if config
        .execution
        .threads
        .is_some_and(|threads| threads.get() > i32::MAX as u32)
        || config
            .execution
            .threads_batch
            .is_some_and(|threads| threads.get() > i32::MAX as u32)
    {
        return Err(InferenceError::InvalidConfig(
            "thread counts must not exceed i32::MAX".into(),
        ));
    }
    if config.execution.flash_attention == FlashAttention::Disabled
        && matches!(
            config.execution.cache_type_v,
            CacheType::Q8_0
                | CacheType::Q4_0
                | CacheType::Q4_1
                | CacheType::Iq4Nl
                | CacheType::Q5_0
                | CacheType::Q5_1
        )
    {
        return Err(InferenceError::InvalidConfig(
            "a quantized V cache requires Flash Attention".into(),
        ));
    }
    if let Some(projector) = &config.projector {
        validate_projector_config(config, projector)?;
    }
    Ok(())
}

#[cfg(not(feature = "mtmd"))]
fn validate_projector_config(
    _config: &ExecutionIntent,
    _projector: &ProjectorConfig,
) -> Result<(), InferenceError> {
    Err(InferenceError::InvalidConfig(
        "a multimodal projector was configured, but this ICN binary was built without the mtmd feature"
            .into(),
    ))
}

#[cfg(feature = "mtmd")]
fn validate_projector_config(
    config: &ExecutionIntent,
    projector: &ProjectorConfig,
) -> Result<(), InferenceError> {
    if !projector.path.is_file() {
        return Err(InferenceError::InvalidConfig(format!(
            "multimodal projector does not exist: {}",
            projector.path.display()
        )));
    }
    if config.max_sequences != 1 {
        return Err(InferenceError::InvalidConfig(
            "multimodal projector mode currently requires max_sequences=1 because llama.cpp's mtmd helper performs direct decode calls outside ICN's shared batch"
                .into(),
        ));
    }
    if config.batch_size > i32::MAX as u32 {
        return Err(InferenceError::InvalidConfig(
            "multimodal projector mode requires batch_size <= i32::MAX".into(),
        ));
    }
    if projector
        .image_min_tokens
        .zip(projector.image_max_tokens)
        .is_some_and(|(minimum, maximum)| minimum > maximum)
    {
        return Err(InferenceError::InvalidConfig(
            "image_min_tokens must not exceed image_max_tokens".into(),
        ));
    }
    if projector
        .image_min_tokens
        .is_some_and(|tokens| tokens.get() > i32::MAX as u32)
        || projector
            .image_max_tokens
            .is_some_and(|tokens| tokens.get() > i32::MAX as u32)
    {
        return Err(InferenceError::InvalidConfig(
            "image token budgets must not exceed i32::MAX".into(),
        ));
    }
    if projector.input_limits.max_total_decoded_bytes
        < projector.input_limits.max_decoded_bytes_per_image
    {
        return Err(InferenceError::InvalidConfig(
            "max_total_decoded_bytes must be at least max_decoded_bytes_per_image".into(),
        ));
    }
    Ok(())
}

fn warm_up(
    model: &LlamaModel,
    context: &mut LlamaContext<'_>,
    mtp: Option<&mut MtpOperations<'_>>,
) -> Result<(), InferenceError> {
    let tokens = model
        .str_to_token(" ", AddBos::Always)
        .map_err(backend_error)?;
    if let Some(token) = tokens.first().copied() {
        let mut batch = LlamaBatch::new(1, 1);
        batch.add(token, 0, &[0], false).map_err(backend_error)?;
        context.decode(&mut batch).map_err(backend_error)?;
        if let Some(mtp) = mtp {
            mtp.process(&batch).map_err(backend_error)?;
        }
        context.synchronize();
        context.clear_kv_cache();
    }
    context.reset_timings();
    Ok(())
}

fn model_properties(
    config: &ExecutionIntent,
    resolved_execution: ExecutionConfig,
    model: &LlamaModel,
    _context: &LlamaContext<'_>,
    templates: &CommonChatTemplates,
    modalities: ModelModalities,
) -> Result<ModelProperties, InferenceError> {
    let chat_template = templates.source(None).map_err(backend_error)?;
    let capabilities = templates.capabilities().map_err(backend_error)?;
    let reasoning = icn_reasoning::inspect_templates(templates).map_err(backend_error)?;
    Ok(ModelProperties {
        model_path: config.model_path.clone(),
        model_size_bytes: model.size(),
        architecture: model.meta_val_str("general.architecture").ok(),
        name: model.meta_val_str("general.name").ok(),
        context_tokens: config.context_size,
        training_context_tokens: model.n_ctx_train(),
        sliding_window_tokens: model.n_swa(),
        template_fingerprint: fingerprint(&chat_template),
        chat_template,
        capabilities: TemplateCapabilities {
            string_content: capabilities.supports_string_content,
            typed_content: capabilities.supports_typed_content,
            tools: capabilities.supports_tools,
            tool_calls: capabilities.supports_tool_calls,
            parallel_tool_calls: capabilities.supports_parallel_tool_calls,
            system_role: capabilities.supports_system_role,
            preserve_reasoning: capabilities.supports_preserve_reasoning,
            object_arguments: capabilities.supports_object_arguments,
            enable_thinking: capabilities.supports_enable_thinking,
        },
        reasoning: reasoning.profile,
        modalities,
        mtp: match &config.mtp {
            icn_contracts::MtpConfig::Disabled { reason } => {
                icn_contracts::MtpRuntimeProperties::Disabled {
                    reason: reason.clone(),
                }
            }
            icn_contracts::MtpConfig::Enabled {
                source,
                n_max,
                n_min,
                p_min,
                ..
            } => icn_contracts::MtpRuntimeProperties::Enabled {
                source: source.clone(),
                n_max: *n_max,
                n_min: *n_min,
                p_min: *p_min,
            },
        },
        execution: ExecutionConfigReport {
            requested: config.execution.clone(),
            resolved: resolved_execution,
        },
    })
}

fn prepared_chat_info(
    templates: &CommonChatTemplates,
    prepared: &PreparedChat,
) -> Result<PreparedChatInfo, InferenceError> {
    let template = templates.source(None).map_err(backend_error)?;
    Ok(PreparedChatInfo {
        prompt: prepared.prompt().to_owned(),
        generation_prompt: prepared.generation_prompt().to_owned(),
        grammar: prepared.grammar().to_owned(),
        grammar_lazy: prepared.grammar_lazy(),
        grammar_triggers: prepared
            .grammar_triggers()
            .iter()
            .map(|trigger| match trigger {
                llama_cpp_2::common_chat::ChatGrammarTrigger::Token { value, token } => {
                    GrammarTrigger::Token {
                        value: value.clone(),
                        token: *token,
                    }
                }
                llama_cpp_2::common_chat::ChatGrammarTrigger::Word(value) => {
                    GrammarTrigger::Word(value.clone())
                }
                llama_cpp_2::common_chat::ChatGrammarTrigger::Pattern(value) => {
                    GrammarTrigger::Pattern(value.clone())
                }
                llama_cpp_2::common_chat::ChatGrammarTrigger::PatternFull(value) => {
                    GrammarTrigger::PatternFull(value.clone())
                }
            })
            .collect(),
        preserved_tokens: prepared.preserved_tokens().to_vec(),
        additional_stops: prepared.additional_stops().to_vec(),
        supports_thinking: prepared.supports_thinking(),
        thinking_start_tag: prepared.thinking_start_tag().map(str::to_owned),
        thinking_end_tag: prepared.thinking_end_tag().map(str::to_owned),
        template_fingerprint: fingerprint(&template),
    })
}

fn fingerprint(value: &str) -> String {
    format!("sha256:{:x}", Sha256::digest(value.as_bytes()))
}

fn request_images(request: &ChatTemplateRequest) -> Vec<ImageInput> {
    request
        .messages
        .iter()
        .filter_map(|message| match &message.content {
            Some(ChatContent::Parts(parts)) => Some(parts),
            None | Some(ChatContent::Text(_)) => None,
        })
        .flat_map(|parts| parts.iter())
        .filter_map(|part| match part {
            ChatContentPart::Image(image) => Some(image.clone()),
            ChatContentPart::Text { .. } => None,
        })
        .collect()
}

fn plain_prompt(
    model: &LlamaModel,
    prepared: &PreparedChat,
) -> Result<TokenizedPrompt, InferenceError> {
    let text_tokens = model
        .str_to_token(prepared.prompt(), AddBos::Always)
        .map_err(backend_error)?;
    let total_tokens = text_tokens.len();
    Ok(TokenizedPrompt {
        text_tokens,
        total_tokens,
        next_position: i32::try_from(total_tokens).map_err(backend_error)?,
        multimodal: None,
    })
}

#[cfg(feature = "mtmd")]
fn tokenize_prepared_prompt(
    model: &LlamaModel,
    prepared: &PreparedChat,
    multimodal: Option<&MultimodalRuntime<'_>>,
    images: &[ImageInput],
) -> Result<TokenizedPrompt, InferenceError> {
    if images.is_empty() {
        return plain_prompt(model, prepared);
    }
    let runtime = multimodal.ok_or_else(|| {
        InferenceError::InvalidConfig(
            "image content requires a multimodal projector configured with --mmproj".into(),
        )
    })?;
    let prompt = runtime.prepare_prompt(prepared.prompt().to_owned(), images)?;
    Ok(TokenizedPrompt {
        text_tokens: prompt.text_tokens().to_vec(),
        total_tokens: prompt.total_tokens(),
        next_position: 0,
        multimodal: Some(prompt),
    })
}

#[cfg(not(feature = "mtmd"))]
fn tokenize_prepared_prompt(
    model: &LlamaModel,
    prepared: &PreparedChat,
    _multimodal: Option<&MultimodalRuntime<'_>>,
    images: &[ImageInput],
) -> Result<TokenizedPrompt, InferenceError> {
    if !images.is_empty() {
        return Err(InferenceError::InvalidConfig(
            "image content requires an ICN binary compiled with multimodal support".into(),
        ));
    }
    plain_prompt(model, prepared)
}

#[cfg(feature = "mtmd")]
fn multimodal_marker<'runtime>(runtime: &'runtime MultimodalRuntime<'_>) -> &'runtime str {
    runtime.marker()
}

#[cfg(not(feature = "mtmd"))]
fn multimodal_marker<'runtime>(_runtime: &'runtime MultimodalRuntime<'_>) -> &'runtime str {
    unreachable!("the feature-disabled build never creates a multimodal runtime")
}

#[cfg(feature = "mtmd")]
fn multimodal_modalities(runtime: &MultimodalRuntime<'_>) -> ModelModalities {
    runtime.modalities()
}

#[cfg(not(feature = "mtmd"))]
fn multimodal_modalities(_runtime: &MultimodalRuntime<'_>) -> ModelModalities {
    ModelModalities::default()
}

fn validate_prompt_capacity(
    prompt_tokens: usize,
    context_capacity: usize,
) -> Result<(), InferenceError> {
    if prompt_tokens >= context_capacity {
        return Err(InferenceError::InvalidConfig(format!(
            "prompt ({prompt_tokens} tokens) leaves no generation capacity in the effective per-sequence context ({context_capacity})"
        )));
    }
    Ok(())
}

impl<'model> ActiveRequest<'model> {
    #[allow(clippy::too_many_arguments)]
    fn admit(
        model: &'model LlamaModel,
        chat_templates: &CommonChatTemplates,
        multimodal: Option<&MultimodalRuntime<'model>>,
        context_capacity: usize,
        batch_size: usize,
        ubatch_size: usize,
        sequence_id: i32,
        queued: QueuedCompletion,
        cached: Option<&SequenceCache>,
    ) -> Result<Self, (SyncSender<ExecutorItem>, InferenceError)> {
        let QueuedCompletion {
            request,
            events,
            cancelled,
            queued_at,
            span,
        } = queued;
        let entered_span = span.clone();
        let _entered = entered_span.enter();
        let result = (|| {
            validate_request(&request)?;
            let admitted_at = Instant::now();
            let images = request_images(&request.template);
            let prepared = prepare_chat(
                chat_templates,
                &request.template,
                multimodal.map(multimodal_marker),
            )?;
            let parser = prepared
                .stream_parser(ChatParserOptions {
                    parse_tool_calls: !request.template.tools.is_empty()
                        && !matches!(request.template.tool_choice, ToolChoice::None),
                    ..ChatParserOptions::default()
                })
                .map_err(backend_error)?;
            let tokenized = tokenize_prepared_prompt(model, &prepared, multimodal, &images)?;
            if tokenized.text_tokens.is_empty() {
                return Err(InferenceError::InvalidConfig(
                    "the prepared prompt tokenized to an empty sequence".into(),
                ));
            }
            let prompt_tokens = tokenized.total_tokens;
            validate_prompt_capacity(prompt_tokens, context_capacity)?;

            let mut sampler = make_sampler(model, &request, &prepared)?;
            sampler
                .accept_prompt(tokenized.text_tokens.iter())
                .map_err(backend_error)?;
            let mut stops = request.stop.clone();
            stops.extend(prepared.additional_stops().iter().cloned());
            let mut cached_prompt_tokens = if request.cache_prompt && tokenized.multimodal.is_none()
            {
                cached.map_or(0, |cached| {
                    common_token_prefix(&cached.prompt, &tokenized.text_tokens)
                })
            } else {
                0
            };
            // The last prompt token must be evaluated to obtain logits for the first sample.
            if cached_prompt_tokens == tokenized.text_tokens.len() {
                cached_prompt_tokens = cached_prompt_tokens.saturating_sub(1);
            }
            let cacheable = tokenized.multimodal.is_none();
            let prompt_checkpoints = cached.map_or_else(Vec::new, |cache| {
                cache
                    .checkpoints
                    .iter()
                    .filter(|checkpoint| checkpoint.prefix <= cached_prompt_tokens)
                    .cloned()
                    .collect()
            });
            // Match llama-server's two bounded-memory prompt checkpoints: one micro-batch
            // before the end and one four tokens before the end. The former permits an exact
            // rollback across a changed prompt tail; the latter makes identical prompts cheap.
            let mut pending_checkpoint_prefixes = [4_usize.saturating_add(ubatch_size), 4]
                .into_iter()
                .map(|offset| {
                    tokenized
                        .text_tokens
                        .len()
                        .saturating_sub(offset.min(batch_size))
                })
                .filter(|prefix| *prefix > cached_prompt_tokens && *prefix > 0)
                .collect::<Vec<_>>();
            pending_checkpoint_prefixes.sort_unstable();
            pending_checkpoint_prefixes.dedup();

            Ok(Self {
                sequence_id: Some(sequence_id),
                events: events.clone(),
                span,
                cancelled,
                outbound: VecDeque::new(),
                phase: RequestPhase::Prefill,
                cache_history: tokenized.text_tokens.clone(),
                prompt: tokenized.text_tokens,
                prompt_offset: cached_prompt_tokens,
                prompt_tokens,
                cached_prompt_tokens,
                prompt_checkpoints,
                pending_checkpoint_prefixes: pending_checkpoint_prefixes.into(),
                next_position: tokenized.next_position,
                multimodal_prompt: tokenized.multimodal,
                generation_limit: (request.max_tokens as usize)
                    .min(context_capacity.saturating_sub(prompt_tokens)),
                generated_tokens: 0,
                mtp_started: false,
                mtp_draft: Vec::new(),
                mtp_indices: Vec::new(),
                draft_tokens: 0,
                accepted_draft_tokens: 0,
                draft_ms: 0.0,
                verification_ms: 0.0,
                cache_prompt: request.cache_prompt,
                cacheable,
                ignore_eos: request.ignore_eos,
                timings_per_token: request.timings_per_token,
                sampler,
                utf8: Utf8Buffer::default(),
                stops: StopBuffer::new(stops),
                semantic: SemanticStream::new(parser),
                queue_ms: admitted_at.duration_since(queued_at).as_secs_f64() * 1_000.0,
                prompt_started_at: None,
                prompt_ms: 0.0,
                generation_started_at: None,
                last_sample_at: None,
                first_event_at: None,
                queued_at,
            })
        })();
        result.map_err(|error| (events, error))
    }

    fn sample_next(
        &mut self,
        model: &LlamaModel,
        context: &LlamaContext<'model>,
        batch_index: i32,
    ) -> Result<Option<FinishReason>, InferenceError> {
        let token = self
            .sampler
            .sample(context, batch_index, false)
            .map_err(backend_error)?;
        self.sampler
            .accept_generated(token)
            .map_err(backend_error)?;
        if let Some(reason) = self.emit_accepted_token(model, token)? {
            return Ok(Some(reason));
        }

        let position = self.next_position;
        self.next_position = self.next_position.checked_add(1).ok_or_else(|| {
            InferenceError::Backend("generation position exceeded i32::MAX".into())
        })?;
        self.phase = RequestPhase::Decode { token, position };
        Ok(None)
    }

    fn emit_accepted_token(
        &mut self,
        model: &LlamaModel,
        token: LlamaToken,
    ) -> Result<Option<FinishReason>, InferenceError> {
        let sampled_at = Instant::now();
        let is_eog = model.is_eog_token(token);
        account_sample(&mut self.generated_tokens);
        self.record_sample(sampled_at);
        let starts_stream = self.generated_tokens == 1;
        if is_eog && !self.ignore_eos {
            let events = sampled_result_events(Vec::new(), starts_stream);
            let timings = (partial_timing_eligible(self.timings_per_token, false)
                && !events.is_empty())
            .then(|| self.generation_snapshot());
            self.enqueue_events(events, timings)?;
            return Ok(Some(FinishReason::Stop));
        }

        let decoded = self.utf8.push(&token_piece_bytes(model, token)?);
        let has_complete_utf8 = !decoded.is_empty() || !self.utf8.has_pending();
        if has_complete_utf8 && self.emit_decoded(decoded, self.timings_per_token, starts_stream)? {
            return Ok(Some(FinishReason::Stop));
        }
        if self.generated_tokens >= self.generation_limit {
            return Ok(Some(FinishReason::Length));
        }

        Ok(None)
    }

    fn record_sample(&mut self, sampled_at: Instant) {
        if self.generation_started_at.is_none() {
            self.prompt_ms = self.prompt_started_at.map_or(0.0, |prompt_started| {
                sampled_at.duration_since(prompt_started).as_secs_f64() * 1_000.0
            });
            self.generation_started_at = Some(sampled_at);
        }
        self.last_sample_at = Some(sampled_at);
    }

    fn emit_decoded(
        &mut self,
        decoded: String,
        with_timings: bool,
        starts_stream: bool,
    ) -> Result<bool, InferenceError> {
        let output = self.stops.push(&decoded);
        let matched = output.matched.is_some();
        self.emit_parsed(
            output.text,
            partial_timing_eligible(with_timings, matched),
            starts_stream,
        )?;
        Ok(matched)
    }

    fn emit_parsed(
        &mut self,
        text: String,
        with_timings: bool,
        starts_stream: bool,
    ) -> Result<(), InferenceError> {
        let events = if text.is_empty() {
            Vec::new()
        } else {
            self.semantic.push(text)?
        };
        if !events.is_empty() {
            self.first_event_at.get_or_insert_with(Instant::now);
        }
        let events = sampled_result_events(events, starts_stream);
        let timings = (with_timings && !events.is_empty()).then(|| self.generation_snapshot());
        self.enqueue_events(events, timings)
    }

    fn enqueue_events(
        &mut self,
        events: Vec<InferenceEvent>,
        timings: Option<GenerationSnapshot>,
    ) -> Result<(), InferenceError> {
        if self.outbound.len() + events.len() > OUTBOUND_QUEUE_CAPACITY {
            return Err(InferenceError::Backend(format!(
                "semantic event burst exceeded the bounded outbound capacity ({OUTBOUND_QUEUE_CAPACITY})"
            )));
        }
        for event in stream_events_with_timings(events, timings) {
            if !matches!(&event.delta, InferenceEvent::StreamStart) {
                self.first_event_at.get_or_insert_with(Instant::now);
            }
            self.outbound.push_back(ExecutorItem::Event(event));
        }
        Ok(())
    }

    fn generation_snapshot(&self) -> GenerationSnapshot {
        let decode_ms = generation_elapsed_ms(self.generation_started_at, self.last_sample_at);
        let time_to_first_token_ms = self.first_event_at.map_or(0.0, |instant| {
            instant.duration_since(self.queued_at).as_secs_f64() * 1_000.0
        });
        GenerationSnapshot {
            cached_prompt_tokens: self.cached_prompt_tokens,
            prompt_tokens: self.prompt_tokens,
            generated_tokens: self.generated_tokens,
            metrics: GenerationMetrics {
                queue_ms: self.queue_ms,
                prompt_ms: self.prompt_ms,
                decode_ms,
                time_to_first_token_ms,
                prompt_tokens_per_second: rate(self.prompt_tokens, self.prompt_ms),
                decode_tokens_per_second: rate(self.generated_tokens, decode_ms),
                sampler_ms: self.sampler.performance().sample_milliseconds,
                parser_ms: self.semantic.parser_ms(),
                draft_tokens: self.draft_tokens,
                accepted_draft_tokens: self.accepted_draft_tokens,
                draft_ms: self.draft_ms,
                verification_ms: self.verification_ms,
            },
        }
    }

    fn complete(&mut self, reason: FinishReason) -> Result<(), InferenceError> {
        if !self.stops.is_stopped() {
            let final_utf8 = self.utf8.finish();
            let _ = self.emit_decoded(final_utf8, false, false)?;
            let tail = self.stops.finish();
            self.emit_parsed(tail, false, false)?;
        }
        let (parsed, final_events) = self.semantic.finish()?;
        self.enqueue_events(final_events, None)?;
        let snapshot = self.generation_snapshot();
        let ParsedChatMessage {
            content,
            reasoning_content,
            tool_calls,
            ..
        } = parsed;
        let has_tool_calls = !tool_calls.is_empty();
        let tool_calls = tool_calls
            .into_iter()
            .enumerate()
            .map(|(index, call)| ToolCall {
                id: tool_call_id(index, call.id.as_deref()),
                name: call.name,
                arguments: call.arguments,
            })
            .collect();
        let generation = Generation {
            text: content,
            reasoning: reasoning_content.unwrap_or_default(),
            tool_calls,
            cached_prompt_tokens: snapshot.cached_prompt_tokens,
            prompt_tokens: snapshot.prompt_tokens,
            generated_tokens: snapshot.generated_tokens,
            finish_reason: if has_tool_calls {
                FinishReason::ToolCalls
            } else {
                reason
            },
            metrics: snapshot.metrics,
        };
        self.phase = RequestPhase::Terminal;
        if self.outbound.len() == OUTBOUND_QUEUE_CAPACITY {
            return Err(InferenceError::Backend(
                "completion could not fit in the bounded outbound queue".into(),
            ));
        }
        self.outbound.push_back(ExecutorItem::Completed(generation));
        Ok(())
    }
}

fn validate_request(request: &ChatRequest) -> Result<(), InferenceError> {
    if request.template.messages.is_empty() {
        return Err(InferenceError::InvalidConfig(
            "messages must not be empty".into(),
        ));
    }
    if request.max_tokens == 0 {
        return Err(InferenceError::InvalidConfig(
            "max_tokens must be greater than zero".into(),
        ));
    }
    if !request.temperature.is_finite() || request.temperature < 0.0 {
        return Err(InferenceError::InvalidConfig(
            "temperature must be finite and non-negative".into(),
        ));
    }
    if !request.top_p.is_finite() || !(0.0..=1.0).contains(&request.top_p) {
        return Err(InferenceError::InvalidConfig(
            "top_p must be finite and between zero and one".into(),
        ));
    }
    if request.stop.iter().any(String::is_empty) {
        return Err(InferenceError::InvalidConfig(
            "stop strings must not be empty".into(),
        ));
    }
    Ok(())
}

fn prepare_chat(
    templates: &CommonChatTemplates,
    request: &ChatTemplateRequest,
    media_marker: Option<&str>,
) -> Result<PreparedChat, InferenceError> {
    let messages = request
        .messages
        .iter()
        .map(|message| {
            let content = match &message.content {
                None => None,
                Some(ChatContent::Text(text)) => Some(NativeChatContent::Text(text.clone())),
                Some(ChatContent::Parts(parts)) => {
                    let parts = parts
                        .iter()
                        .map(|part| match part {
                            ChatContentPart::Text { text } => Ok(NativeChatContentPart {
                                kind: ChatContentPartKind::Text,
                                text: text.clone(),
                            }),
                            ChatContentPart::Image(_) => {
                                let marker = media_marker.ok_or_else(|| {
                                    InferenceError::InvalidConfig(
                                        "image content requires a multimodal projector configured with --mmproj"
                                            .into(),
                                    )
                                })?;
                                Ok(NativeChatContentPart {
                                    kind: ChatContentPartKind::MediaMarker,
                                    text: marker.to_owned(),
                                })
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Some(NativeChatContent::Parts(parts))
                }
            };
            Ok(NativeChatMessage {
                role: message.role.as_str().into(),
                content,
                tool_calls: message
                    .tool_calls
                    .iter()
                    .map(|call| ChatToolCall {
                        name: call.name.clone(),
                        arguments: call.arguments.clone(),
                        id: Some(call.id.clone()),
                    })
                    .collect(),
                reasoning_content: message.reasoning.clone(),
                tool_name: None,
                tool_call_id: message.tool_call_id.clone(),
            })
        })
        .collect::<Result<Vec<_>, InferenceError>>()?;

    let (selected_tools, tool_choice) = select_tools(request)?;
    let tools = selected_tools
        .into_iter()
        .map(|tool| {
            Ok(ChatTool {
                name: tool.name.clone(),
                description: tool.description.clone().unwrap_or_default(),
                parameters_json: serde_json::to_string(&tool.parameters).map_err(backend_error)?,
            })
        })
        .collect::<Result<Vec<_>, InferenceError>>()?;

    let (grammar, json_schema) = match &request.response_format {
        ResponseFormat::Text => (None, None),
        // This deliberately matches llama-server's default response_format=json_object schema.
        ResponseFormat::JsonObject => (None, Some("{}".to_owned())),
        ResponseFormat::Grammar { grammar } => (Some(grammar.clone()), None),
        ResponseFormat::JsonSchema { schema, .. } => (
            None,
            Some(serde_json::to_string(schema).map_err(backend_error)?),
        ),
    };
    if grammar.is_some() && !tools.is_empty() && tool_choice != ChatToolChoice::None {
        return Err(InferenceError::InvalidConfig(
            "a custom grammar cannot be combined with enabled tools".into(),
        ));
    }

    let mut effective_template_args = request.template_args.clone();
    let enable_thinking = match &request.reasoning {
        ReasoningControl::ModelDefault => None,
        ReasoningControl::Disabled => Some(false),
        ReasoningControl::Enabled { .. } => Some(true),
        ReasoningControl::Resolved {
            controls,
            template_fingerprint,
            ..
        } => {
            let source = templates.source(None).map_err(backend_error)?;
            let actual_fingerprint = fingerprint(&source);
            if &actual_fingerprint != template_fingerprint {
                return Err(InferenceError::InvalidConfig(format!(
                    "reasoning recipe template fingerprint mismatch: expected {template_fingerprint}, got {actual_fingerprint}"
                )));
            }
            for (key, value) in &controls.template_args {
                if let Some(existing) = effective_template_args.get(key)
                    && existing != value
                {
                    return Err(InferenceError::InvalidConfig(format!(
                        "reasoning recipe conflicts with chat_template_kwargs.{key}"
                    )));
                }
                effective_template_args.insert(key.clone(), value.clone());
            }
            controls.enable_thinking
        }
    };
    let template_kwargs = effective_template_args
        .iter()
        .map(|(key, value)| {
            Ok(ChatTemplateKwarg {
                key: key.clone(),
                value_json: serde_json::to_string(value).map_err(backend_error)?,
            })
        })
        .collect::<Result<Vec<_>, InferenceError>>()?;
    templates
        .prepare(&ChatPrepareOptions {
            messages,
            grammar,
            json_schema,
            tools,
            tool_choice,
            parallel_tool_calls: Some(request.parallel_tool_calls),
            reasoning_format: ChatReasoningFormat::DeepSeek,
            enable_thinking,
            template_kwargs,
            ..ChatPrepareOptions::default()
        })
        .map_err(backend_error)
}

fn select_tools(
    request: &ChatTemplateRequest,
) -> Result<(Vec<&icn_contracts::ToolDefinition>, ChatToolChoice), InferenceError> {
    let selected = match &request.tool_choice {
        ToolChoice::None => return Ok((Vec::new(), ChatToolChoice::None)),
        ToolChoice::Auto => return Ok((request.tools.iter().collect(), ChatToolChoice::Auto)),
        ToolChoice::Required => {
            if request.tools.is_empty() {
                return Err(InferenceError::InvalidConfig(
                    "required tool choice needs at least one tool".into(),
                ));
            }
            return Ok((request.tools.iter().collect(), ChatToolChoice::Required));
        }
        ToolChoice::Function { name } => vec![name.as_str()],
        ToolChoice::AllowedTools { names, .. } => names.iter().map(String::as_str).collect(),
    };
    let tools = selected
        .iter()
        .map(|name| {
            request
                .tools
                .iter()
                .find(|tool| tool.name == *name)
                .ok_or_else(|| {
                    InferenceError::InvalidConfig(format!(
                        "tool choice references undefined tool: {name}"
                    ))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let choice = match &request.tool_choice {
        ToolChoice::Function { .. } => ChatToolChoice::Required,
        ToolChoice::AllowedTools {
            mode: AllowedToolsMode::Auto,
            ..
        } => ChatToolChoice::Auto,
        ToolChoice::AllowedTools {
            mode: AllowedToolsMode::Required,
            ..
        } => ChatToolChoice::Required,
        _ => unreachable!("early-returned tool choice"),
    };
    Ok((tools, choice))
}

fn make_sampler<'model>(
    model: &'model LlamaModel,
    request: &ChatRequest,
    prepared: &PreparedChat,
) -> Result<CommonSampler<'model>, InferenceError> {
    let grammar = if prepared.grammar().is_empty() {
        None
    } else {
        let tools_enabled = !request.template.tools.is_empty()
            && !matches!(request.template.tool_choice, ToolChoice::None);
        let kind = if tools_enabled {
            CommonGrammarKind::ToolCalls
        } else {
            match request.template.response_format {
                ResponseFormat::JsonObject | ResponseFormat::JsonSchema { .. } => {
                    CommonGrammarKind::OutputFormat
                }
                ResponseFormat::Grammar { .. } | ResponseFormat::Text => CommonGrammarKind::User,
            }
        };
        Some(CommonGrammar {
            kind,
            source: prepared.grammar().to_owned(),
        })
    };
    let grammar_triggers = grammar.as_ref().map(|_| {
        prepared
            .grammar_triggers()
            .iter()
            .map(|trigger| match trigger {
                llama_cpp_2::common_chat::ChatGrammarTrigger::Token { value, token } => {
                    CommonGrammarTrigger::Token {
                        token: LlamaToken::new(*token),
                        value: Some(value.clone()),
                    }
                }
                llama_cpp_2::common_chat::ChatGrammarTrigger::Word(value) => {
                    CommonGrammarTrigger::Word(value.clone())
                }
                llama_cpp_2::common_chat::ChatGrammarTrigger::Pattern(value) => {
                    CommonGrammarTrigger::Pattern(value.clone())
                }
                llama_cpp_2::common_chat::ChatGrammarTrigger::PatternFull(value) => {
                    CommonGrammarTrigger::PatternFull(value.clone())
                }
            })
            .collect()
    });
    let reasoning_budget_tokens = match &request.template.reasoning {
        ReasoningControl::Enabled {
            budget_tokens: Some(tokens),
        } => Some(*tokens),
        ReasoningControl::Resolved {
            automatic_budget,
            explicit_budget_tokens,
            ..
        } => explicit_budget_tokens.or(match automatic_budget {
            icn_contracts::AutomaticReasoningBudget::Disabled => None,
            icn_contracts::AutomaticReasoningBudget::FixedTokens { tokens } => Some(*tokens),
        }),
        _ => None,
    };
    let reasoning_budget = match reasoning_budget_tokens {
        Some(tokens) => {
            let start_tag = prepared.thinking_start_tag().ok_or_else(|| {
                InferenceError::InvalidConfig(
                    "the active template does not expose a reasoning start tag for budgeting"
                        .into(),
                )
            })?;
            let end_tag = prepared.thinking_end_tag().ok_or_else(|| {
                InferenceError::InvalidConfig(
                    "the active template does not expose a reasoning end tag for budgeting".into(),
                )
            })?;
            Some(CommonReasoningBudget {
                limit: ReasoningBudgetLimit::Tokens(tokens),
                start_tag: start_tag.to_owned(),
                end_tag: end_tag.to_owned(),
                forced_message: String::new(),
                controllable: true,
            })
        }
        None => None,
    };

    CommonSampler::new(
        model,
        &CommonSamplerConfig {
            seed: Some(request.seed),
            // Match llama.cpp server semantics: `ignore_eos` suppresses every
            // end-of-generation token in the sampler, rather than allowing a
            // special token to be selected and emitted as ordinary text.
            ignore_eos: Some(request.ignore_eos),
            top_p: Some(request.top_p),
            temperature: Some(request.temperature),
            grammar,
            grammar_lazy: (!prepared.grammar().is_empty()).then_some(prepared.grammar_lazy()),
            grammar_triggers,
            preserved_tokens: (!prepared.grammar().is_empty())
                .then(|| prepared.preserved_tokens().to_vec()),
            generation_prompt: (!prepared.generation_prompt().is_empty())
                .then(|| prepared.generation_prompt().to_owned()),
            reasoning_budget,
            ..CommonSamplerConfig::default()
        },
    )
    .map_err(backend_error)
}

fn token_piece_bytes(model: &LlamaModel, token: LlamaToken) -> Result<Vec<u8>, InferenceError> {
    match model.token_to_piece_bytes(token, 32, true, None) {
        Ok(bytes) => Ok(bytes),
        Err(TokenToStringError::InsufficientBufferSpace(required)) => model
            .token_to_piece_bytes(
                token,
                usize::try_from(-required).map_err(backend_error)?,
                true,
                None,
            )
            .map_err(backend_error),
        Err(error) => Err(backend_error(error)),
    }
}

struct SemanticStream {
    parser: ChatStreamParser,
    tools: Vec<StreamingToolCall>,
    parser_time: Duration,
}

#[derive(Debug)]
struct StreamingToolCall {
    id: String,
    name: String,
    pending_arguments: String,
    header_sent: bool,
}

impl SemanticStream {
    fn new(parser: ChatStreamParser) -> Self {
        Self {
            parser,
            tools: Vec::new(),
            parser_time: Duration::ZERO,
        }
    }

    fn push(&mut self, text: String) -> Result<Vec<InferenceEvent>, InferenceError> {
        let parse_started = Instant::now();
        let deltas = self.parser.push(&text).map_err(backend_error)?;
        self.parser_time += parse_started.elapsed();
        Ok(self.translate_deltas(deltas, false))
    }

    fn finish(&mut self) -> Result<(ParsedChatMessage, Vec<InferenceEvent>), InferenceError> {
        let parse_started = Instant::now();
        let (mut final_message, deltas) = self.parser.finish().map_err(backend_error)?;
        self.parser_time += parse_started.elapsed();
        let mut events = self.translate_deltas(deltas, true);

        self.reconcile_final_tools(&mut final_message, &mut events);
        Ok((final_message, events))
    }

    fn reconcile_final_tools(
        &mut self,
        final_message: &mut ParsedChatMessage,
        events: &mut Vec<InferenceEvent>,
    ) {
        // The native parser owns semantic diffing, while ICN owns transport policy. In
        // particular, ICN waits for a useful tool name before emitting a tool header and supplies
        // stable synthetic IDs when the model omits one. Reconcile the terminal snapshot so a tool
        // discovered only during final parsing still produces a header.
        for (index, call) in final_message.tool_calls.iter_mut().enumerate() {
            self.ensure_tool(index, call.id.as_deref());
            let tool = &mut self.tools[index];
            if tool.name.is_empty() {
                tool.name.clone_from(&call.name);
            }
            if !tool.header_sent && !call.name.is_empty() {
                events.push(InferenceEvent::ToolCallDelta {
                    index,
                    id: Some(tool.id.clone()),
                    name: Some(call.name.clone()),
                    arguments: call.arguments.clone(),
                });
                tool.header_sent = true;
                tool.pending_arguments.clear();
            }
            call.id = Some(tool.id.clone());
        }
    }

    fn parser_ms(&self) -> f64 {
        self.parser_time.as_secs_f64() * 1_000.0
    }

    fn translate_deltas(
        &mut self,
        deltas: Vec<ChatSemanticDelta>,
        is_final: bool,
    ) -> Vec<InferenceEvent> {
        let mut events = Vec::new();
        for delta in deltas {
            match delta {
                ChatSemanticDelta::Reasoning(text) if !text.is_empty() => {
                    events.push(InferenceEvent::ReasoningDelta { text });
                }
                ChatSemanticDelta::Content(text) if !text.is_empty() => {
                    events.push(InferenceEvent::ContentDelta { text });
                }
                ChatSemanticDelta::Reasoning(_) | ChatSemanticDelta::Content(_) => {}
                ChatSemanticDelta::ToolCall {
                    index,
                    id,
                    name,
                    arguments,
                } => {
                    self.ensure_tool(index, id.as_deref());
                    let tool = &mut self.tools[index];
                    if let Some(name) = name {
                        tool.name = name;
                    }
                    if tool.header_sent {
                        if !arguments.is_empty() {
                            events.push(InferenceEvent::ToolCallDelta {
                                index,
                                id: None,
                                name: None,
                                arguments,
                            });
                        }
                    } else {
                        tool.pending_arguments.push_str(&arguments);
                        if !tool.name.is_empty() && (is_final || !tool.pending_arguments.is_empty())
                        {
                            events.push(InferenceEvent::ToolCallDelta {
                                index,
                                id: Some(tool.id.clone()),
                                name: Some(tool.name.clone()),
                                arguments: std::mem::take(&mut tool.pending_arguments),
                            });
                            tool.header_sent = true;
                        }
                    }
                }
            }
        }
        events
    }

    fn ensure_tool(&mut self, index: usize, native_id: Option<&str>) {
        while self.tools.len() <= index {
            let next = self.tools.len();
            let native_id = if next == index { native_id } else { None };
            self.tools.push(StreamingToolCall {
                id: tool_call_id(next, native_id),
                name: String::new(),
                pending_arguments: String::new(),
                header_sent: false,
            });
        }
        if let Some(native_id) = native_id.filter(|id| !id.is_empty()) {
            let tool = &mut self.tools[index];
            if !tool.header_sent {
                tool.id = native_id.to_owned();
            }
        }
    }
}

fn tool_call_id(index: usize, native: Option<&str>) -> String {
    native
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| format!("call_icn_{index}"))
}

fn stream_events_with_timings(
    events: Vec<InferenceEvent>,
    mut timings: Option<GenerationSnapshot>,
) -> impl Iterator<Item = InferenceStreamEvent> {
    let last = events.len().checked_sub(1);
    events
        .into_iter()
        .enumerate()
        .map(move |(index, delta)| InferenceStreamEvent {
            delta,
            timings: if Some(index) == last {
                timings.take()
            } else {
                None
            },
        })
}

fn sampled_result_events(
    mut semantic_events: Vec<InferenceEvent>,
    starts_stream: bool,
) -> Vec<InferenceEvent> {
    if starts_stream {
        semantic_events.insert(0, InferenceEvent::StreamStart);
    }
    semantic_events
}

fn partial_timing_eligible(timings_per_token: bool, stopped_before_send: bool) -> bool {
    timings_per_token || stopped_before_send
}

fn generation_elapsed_ms(started: Option<Instant>, last_sampled: Option<Instant>) -> f64 {
    match (started, last_sampled) {
        (Some(started), Some(last_sampled)) => {
            (last_sampled.duration_since(started).as_secs_f64() * 1_000.0).max(0.001)
        }
        _ => 0.0,
    }
}

fn rate(tokens: usize, milliseconds: f64) -> f64 {
    if tokens == 0 || milliseconds <= 0.0 {
        0.0
    } else {
        tokens as f64 * 1_000.0 / milliseconds
    }
}

fn account_sample(generated_tokens: &mut usize) {
    // llama-server increments n_decoded immediately after sampling/accepting, before checking EOG.
    // Matching that ordering keeps OpenAI usage parity when the stop token itself is sampled.
    *generated_tokens += 1;
}

fn backend_error(error: impl std::fmt::Display) -> InferenceError {
    InferenceError::Backend(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use icn_contracts::{ChatMessage, ChatRole, ReasoningControl, ResponseFormat, ToolDefinition};

    use super::*;

    const CHATML: &str = r#"{%- for message in messages -%}
{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|im_start|>assistant\n' -}}
{%- endif -%}"#;

    const TYPED_CHAT: &str = r#"{%- for message in messages -%}
{{- '<|im_start|>' + message.role + '\n' -}}
{%- if message.content is string -%}
{{- message.content -}}
{%- else -%}
{%- for part in message.content -%}{{- part.text -}}{%- endfor -%}
{%- endif -%}
{{- '<|im_end|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}"#;

    const AUTHORED_DISABLED: &str = r#"{% set enable_thinking = enable_thinking | default(false) %}{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}{% if enable_thinking %}<think>{% endif %}assistant:"#;

    fn request() -> ChatRequest {
        ChatRequest {
            template: ChatTemplateRequest {
                messages: vec![ChatMessage::text(ChatRole::User, "Hello")],
                tools: Vec::new(),
                tool_choice: ToolChoice::Auto,
                parallel_tool_calls: true,
                reasoning: ReasoningControl::ModelDefault,
                response_format: ResponseFormat::Text,
                template_args: BTreeMap::new(),
            },
            stop: Vec::new(),
            max_tokens: 32,
            temperature: 0.0,
            top_p: 0.95,
            seed: 42,
            cache_prompt: true,
            ignore_eos: false,
            timings_per_token: false,
        }
    }

    #[test]
    fn common_chat_preparation_is_used_for_plain_messages() {
        let templates = CommonChatTemplates::from_template(CHATML, None, None).unwrap();
        let prepared = prepare_chat(&templates, &request().template, None).unwrap();
        assert_eq!(
            prepared.prompt(),
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
        assert!(!prepared.parser_definition().is_empty());
    }

    #[test]
    fn model_default_uses_llama_cpp_thinking_default() {
        let templates = CommonChatTemplates::from_template(AUTHORED_DISABLED, None, None).unwrap();
        let prepared = prepare_chat(&templates, &request().template, None).unwrap();
        assert!(prepared.prompt().contains("<think>"));
    }

    #[test]
    fn resolved_recipe_applies_controls_and_checks_fingerprint() {
        let templates = CommonChatTemplates::from_template(AUTHORED_DISABLED, None, None).unwrap();
        let mut request = request();
        request.template.reasoning = ReasoningControl::Resolved {
            effort: icn_contracts::NormalizedReasoningEffort("high".into()),
            controls: icn_contracts::NativeReasoningControls {
                enable_thinking: Some(true),
                template_args: BTreeMap::new(),
            },
            automatic_budget: icn_contracts::AutomaticReasoningBudget::Disabled,
            explicit_budget_tokens: None,
            template_fingerprint: fingerprint(AUTHORED_DISABLED),
        };
        let prepared = prepare_chat(&templates, &request.template, None).unwrap();
        assert!(prepared.prompt().contains("<think>"));

        if let ReasoningControl::Resolved {
            template_fingerprint,
            ..
        } = &mut request.template.reasoning
        {
            *template_fingerprint = "sha256:stale".into();
        }
        let error = prepare_chat(&templates, &request.template, None).unwrap_err();
        assert!(error.to_string().contains("fingerprint mismatch"));
    }

    #[test]
    fn prompt_capacity_reserves_at_least_one_generation_token() {
        assert!(validate_prompt_capacity(127, 128).is_ok());
        assert_eq!(
            validate_prompt_capacity(128, 128).unwrap_err().to_string(),
            "invalid configuration: prompt (128 tokens) leaves no generation capacity in the effective per-sequence context (128)"
        );
    }

    #[test]
    fn image_parts_become_explicit_native_media_markers_in_order() {
        let templates = CommonChatTemplates::from_template(TYPED_CHAT, None, None).unwrap();
        let mut request = request();
        request.template.messages[0].content = Some(ChatContent::Parts(vec![
            ChatContentPart::Text {
                text: "before".into(),
            },
            ChatContentPart::Image(ImageInput::new("image/png", vec![1])),
            ChatContentPart::Text {
                text: "after".into(),
            },
        ]));
        let images = request_images(&request.template);
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].bytes(), [1]);
        let prepared =
            prepare_chat(&templates, &request.template, Some("<__media_test__>")).unwrap();
        assert!(prepared.prompt().contains("before<__media_test__>after"));
    }

    #[test]
    fn image_parts_require_a_loaded_projector_marker() {
        let templates = CommonChatTemplates::from_template(TYPED_CHAT, None, None).unwrap();
        let mut request = request();
        request.template.messages[0].content =
            Some(ChatContent::Parts(vec![ChatContentPart::Image(
                ImageInput::new("image/png", vec![1]),
            )]));
        let error = prepare_chat(&templates, &request.template, None).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("requires a multimodal projector")
        );
    }

    fn model_config_with_projector(max_sequences: u32) -> ExecutionIntent {
        let executable = std::env::current_exe().unwrap();
        ExecutionIntent {
            model_path: executable.clone(),
            context_size: 128,
            batch_size: 32,
            ubatch_size: 32,
            max_sequences,
            prefill_quantum: 16,
            execution: ExecutionConfig {
                gpu_layers: GpuLayers::Count(0),
                threads: NonZeroU32::new(1),
                threads_batch: NonZeroU32::new(1),
                ..ExecutionConfig::default()
            },
            projector: Some(ProjectorConfig::new(executable)),
            mtp: icn_contracts::MtpConfig::default(),
        }
    }

    fn model_config() -> ExecutionIntent {
        let mut config = model_config_with_projector(1);
        config.projector = None;
        config
    }

    #[test]
    fn execution_validation_rejects_ambiguous_or_native_invalid_combinations() {
        let mut config = model_config();
        config.execution.gpu_layers = GpuLayers::All;
        config.execution.tensor_split = None;
        config.execution.cache_type_v = CacheType::Q4_0;
        config.execution.flash_attention = FlashAttention::Disabled;
        assert!(
            validate_model_config(&config)
                .unwrap_err()
                .to_string()
                .contains("quantized V cache")
        );
    }

    #[test]
    fn tensor_split_reporting_removes_only_native_padding() {
        assert_eq!(
            trimmed_tensor_split(&[3.0, 0.0, 1.0, 0.0, 0.0]),
            Some(vec![3.0, 0.0, 1.0])
        );
        assert_eq!(trimmed_tensor_split(&[0.0, 0.0]), None);
    }

    #[cfg(feature = "mtmd")]
    #[test]
    fn projector_mode_truthfully_rejects_continuous_batching() {
        let error = validate_model_config(&model_config_with_projector(2)).unwrap_err();
        assert!(error.to_string().contains("requires max_sequences=1"));
        validate_model_config(&model_config_with_projector(1)).unwrap();
    }

    #[cfg(not(feature = "mtmd"))]
    #[test]
    fn feature_disabled_binary_rejects_a_projector() {
        let error = validate_model_config(&model_config_with_projector(1)).unwrap_err();
        assert!(error.to_string().contains("without the mtmd feature"));
    }

    #[test]
    fn tool_selection_filters_named_and_allowed_tools() {
        let mut request = request();
        request.template.tools = ["one", "two"]
            .into_iter()
            .map(|name| ToolDefinition {
                name: name.into(),
                description: None,
                parameters: serde_json::json!({"type": "object"}),
            })
            .collect();
        request.template.tool_choice = ToolChoice::Function { name: "two".into() };
        let (selected, choice) = select_tools(&request.template).unwrap();
        assert_eq!(choice, ChatToolChoice::Required);
        assert_eq!(
            selected
                .iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>(),
            ["two"]
        );

        request.template.tool_choice = ToolChoice::AllowedTools {
            mode: AllowedToolsMode::Auto,
            names: vec!["one".into()],
        };
        let (selected, choice) = select_tools(&request.template).unwrap();
        assert_eq!(choice, ChatToolChoice::Auto);
        assert_eq!(
            selected
                .iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>(),
            ["one"]
        );
    }

    #[test]
    fn semantic_stream_is_chunk_invariant_for_content() {
        let templates = CommonChatTemplates::from_template(CHATML, None, None).unwrap();
        let prepared = prepare_chat(&templates, &request().template, None).unwrap();
        let parser = prepared
            .stream_parser(ChatParserOptions::default())
            .unwrap();
        let mut stream = SemanticStream::new(parser);
        let mut events = stream.push("Hel".into()).unwrap();
        events.extend(stream.push("lo".into()).unwrap());
        let (final_message, final_events) = stream.finish().unwrap();
        events.extend(final_events);
        assert_eq!(final_message.content, "Hello");
        let deltas = events
            .into_iter()
            .map(|event| match event {
                InferenceEvent::ContentDelta { text } => text,
                _ => panic!("unexpected semantic event"),
            })
            .collect::<String>();
        assert_eq!(deltas, "Hello");
    }

    #[test]
    fn semantic_stream_keeps_tool_transport_policy_outside_native_parser() {
        let templates = CommonChatTemplates::from_template(CHATML, None, None).unwrap();
        let prepared = prepare_chat(&templates, &request().template, None).unwrap();
        let parser = prepared
            .stream_parser(ChatParserOptions::default())
            .unwrap();
        let mut stream = SemanticStream::new(parser);

        assert!(
            stream
                .translate_deltas(
                    vec![ChatSemanticDelta::ToolCall {
                        index: 0,
                        id: None,
                        name: Some("get_weather".into()),
                        arguments: String::new(),
                    }],
                    false,
                )
                .is_empty()
        );

        assert_eq!(
            stream.translate_deltas(
                vec![ChatSemanticDelta::ToolCall {
                    index: 0,
                    id: None,
                    name: None,
                    arguments: "{\"city\":".into(),
                }],
                false,
            ),
            vec![InferenceEvent::ToolCallDelta {
                index: 0,
                id: Some("call_icn_0".into()),
                name: Some("get_weather".into()),
                arguments: "{\"city\":".into(),
            }]
        );

        assert_eq!(
            stream.translate_deltas(
                vec![ChatSemanticDelta::ToolCall {
                    index: 0,
                    id: None,
                    name: None,
                    arguments: "\"Paris\"}".into(),
                }],
                false,
            ),
            vec![InferenceEvent::ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments: "\"Paris\"}".into(),
            }]
        );
    }

    #[test]
    fn semantic_stream_adopts_a_late_native_id_before_emitting_the_header() {
        let templates = CommonChatTemplates::from_template(CHATML, None, None).unwrap();
        let prepared = prepare_chat(&templates, &request().template, None).unwrap();
        let parser = prepared
            .stream_parser(ChatParserOptions::default())
            .unwrap();
        let mut stream = SemanticStream::new(parser);

        assert!(
            stream
                .translate_deltas(
                    vec![ChatSemanticDelta::ToolCall {
                        index: 0,
                        id: None,
                        name: Some("get_weather".into()),
                        arguments: String::new(),
                    }],
                    false,
                )
                .is_empty()
        );
        assert_eq!(
            stream.translate_deltas(
                vec![ChatSemanticDelta::ToolCall {
                    index: 0,
                    id: Some("native-call-id".into()),
                    name: None,
                    arguments: "{}".into(),
                }],
                false,
            ),
            vec![InferenceEvent::ToolCallDelta {
                index: 0,
                id: Some("native-call-id".into()),
                name: Some("get_weather".into()),
                arguments: "{}".into(),
            }]
        );

        // Once a header is visible its ID is immutable, even if a later native delta disagrees.
        assert_eq!(
            stream.translate_deltas(
                vec![ChatSemanticDelta::ToolCall {
                    index: 0,
                    id: Some("different-id".into()),
                    name: None,
                    arguments: " ".into(),
                }],
                false,
            ),
            vec![InferenceEvent::ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments: " ".into(),
            }]
        );
        assert_eq!(stream.tools[0].id, "native-call-id");
    }

    #[test]
    fn semantic_stream_emits_a_tool_found_only_in_the_final_snapshot() {
        let templates = CommonChatTemplates::from_template(CHATML, None, None).unwrap();
        let prepared = prepare_chat(&templates, &request().template, None).unwrap();
        let parser = prepared
            .stream_parser(ChatParserOptions::default())
            .unwrap();
        let mut stream = SemanticStream::new(parser);
        let mut final_message = ParsedChatMessage {
            role: "assistant".into(),
            content: String::new(),
            reasoning_content: None,
            tool_calls: vec![ChatToolCall {
                name: "get_weather".into(),
                arguments: r#"{"city":"Paris"}"#.into(),
                id: None,
            }],
            tool_name: None,
            tool_call_id: None,
        };
        let mut events = Vec::new();

        stream.reconcile_final_tools(&mut final_message, &mut events);

        assert_eq!(
            events,
            vec![InferenceEvent::ToolCallDelta {
                index: 0,
                id: Some("call_icn_0".into()),
                name: Some("get_weather".into()),
                arguments: r#"{"city":"Paris"}"#.into(),
            }]
        );
        assert_eq!(
            final_message.tool_calls[0].id.as_deref(),
            Some("call_icn_0")
        );
    }

    #[test]
    fn semantic_stream_preserves_interleaved_multi_tool_event_order() {
        let templates = CommonChatTemplates::from_template(CHATML, None, None).unwrap();
        let prepared = prepare_chat(&templates, &request().template, None).unwrap();
        let parser = prepared
            .stream_parser(ChatParserOptions::default())
            .unwrap();
        let mut stream = SemanticStream::new(parser);

        let events = stream.translate_deltas(
            vec![
                ChatSemanticDelta::Content("Checking both cities. ".into()),
                ChatSemanticDelta::ToolCall {
                    index: 0,
                    id: Some("weather-id".into()),
                    name: Some("get_weather".into()),
                    arguments: "{".into(),
                },
                ChatSemanticDelta::Reasoning("Need local time too. ".into()),
                ChatSemanticDelta::ToolCall {
                    index: 1,
                    id: None,
                    name: Some("get_time".into()),
                    arguments: "{}".into(),
                },
                ChatSemanticDelta::ToolCall {
                    index: 0,
                    id: None,
                    name: None,
                    arguments: "}".into(),
                },
            ],
            false,
        );

        assert_eq!(
            events,
            vec![
                InferenceEvent::ContentDelta {
                    text: "Checking both cities. ".into(),
                },
                InferenceEvent::ToolCallDelta {
                    index: 0,
                    id: Some("weather-id".into()),
                    name: Some("get_weather".into()),
                    arguments: "{".into(),
                },
                InferenceEvent::ReasoningDelta {
                    text: "Need local time too. ".into(),
                },
                InferenceEvent::ToolCallDelta {
                    index: 1,
                    id: Some("call_icn_1".into()),
                    name: Some("get_time".into()),
                    arguments: "{}".into(),
                },
                InferenceEvent::ToolCallDelta {
                    index: 0,
                    id: None,
                    name: None,
                    arguments: "}".into(),
                },
            ]
        );
    }

    #[test]
    fn sampled_token_accounting_includes_the_eventual_eog_token() {
        let mut generated_tokens = 0;
        account_sample(&mut generated_tokens);
        account_sample(&mut generated_tokens); // the second accepted sample may be EOG
        assert_eq!(generated_tokens, 2);
    }

    #[test]
    fn sampled_token_timings_are_attached_only_to_the_last_semantic_delta() {
        let snapshot = GenerationSnapshot {
            cached_prompt_tokens: 0,
            prompt_tokens: 11,
            generated_tokens: 3,
            metrics: GenerationMetrics::default(),
        };
        let events = vec![
            InferenceEvent::StreamStart,
            InferenceEvent::ReasoningDelta {
                text: "thinking".into(),
            },
            InferenceEvent::ContentDelta {
                text: "answer".into(),
            },
        ];

        let events = stream_events_with_timings(events, Some(snapshot)).collect::<Vec<_>>();

        assert_eq!(events.len(), 3);
        assert!(events[0].timings.is_none());
        assert!(events[1].timings.is_none());
        assert_eq!(events[2].timings.as_ref().unwrap().prompt_tokens, 11);
        assert_eq!(events[2].timings.as_ref().unwrap().generated_tokens, 3);
    }

    #[test]
    fn first_sample_without_semantic_delta_attaches_timing_to_stream_start() {
        let snapshot = GenerationSnapshot {
            cached_prompt_tokens: 0,
            prompt_tokens: 11,
            generated_tokens: 1,
            metrics: GenerationMetrics {
                decode_ms: 0.001,
                ..GenerationMetrics::default()
            },
        };

        let events =
            stream_events_with_timings(sampled_result_events(Vec::new(), true), Some(snapshot))
                .collect::<Vec<_>>();

        assert_eq!(events.len(), 1);
        assert!(matches!(events[0].delta, InferenceEvent::StreamStart));
        assert_eq!(events[0].timings.as_ref().unwrap().generated_tokens, 1);
        assert_eq!(events[0].timings.as_ref().unwrap().metrics.decode_ms, 0.001);
    }

    #[test]
    fn later_parser_empty_sampled_result_has_no_transport_event() {
        assert!(sampled_result_events(Vec::new(), false).is_empty());
    }

    #[test]
    fn partial_timing_eligibility_matches_llama_stop_detection_order() {
        assert!(!partial_timing_eligible(false, false));
        assert!(partial_timing_eligible(true, false));
        assert!(partial_timing_eligible(false, true));
        assert!(partial_timing_eligible(true, true));
    }

    #[test]
    fn empty_or_final_parser_groups_do_not_emit_per_token_timings() {
        let snapshot = GenerationSnapshot {
            cached_prompt_tokens: 0,
            prompt_tokens: 11,
            generated_tokens: 3,
            metrics: GenerationMetrics::default(),
        };
        assert_eq!(
            stream_events_with_timings(Vec::new(), Some(snapshot)).count(),
            0
        );

        let final_events = stream_events_with_timings(
            vec![InferenceEvent::ContentDelta {
                text: "tail".into(),
            }],
            None,
        )
        .collect::<Vec<_>>();
        assert_eq!(final_events.len(), 1);
        assert!(final_events[0].timings.is_none());
    }

    #[test]
    fn generation_clock_matches_llama_first_token_floor() {
        let started = Instant::now();
        assert_eq!(generation_elapsed_ms(Some(started), Some(started)), 0.001);
        assert_eq!(
            generation_elapsed_ms(Some(started), Some(started + Duration::from_millis(12))),
            12.0
        );
        assert_eq!(generation_elapsed_ms(None, None), 0.0);
    }

    #[test]
    fn resident_allocations_collapse_host_and_metal_into_unified_memory() {
        use icn_contracts::{
            HardwareDevice, HardwareDeviceKind, HardwareMemoryDomain, HardwareMemoryDomainKind,
            HardwareSystemMemory,
        };

        let snapshot = HardwareSnapshot {
            captured_at: 1,
            platform: "macos".to_owned(),
            architecture: "aarch64".to_owned(),
            system_product_name: Some("MacBook Pro".to_owned()),
            cpu_model: Some("Apple".to_owned()),
            logical_cores: 8,
            system_memory: HardwareSystemMemory {
                total_bytes: 64,
                current_available_bytes: Some(20),
            },
            native_build: "test".to_owned(),
            enabled_backends: vec!["MTL".to_owned()],
            topology_fingerprint: "test".to_owned(),
            memory_domains: vec![HardwareMemoryDomain {
                id: "system".to_owned(),
                kind: HardwareMemoryDomainKind::UnifiedMemory,
                total_capacity_bytes: 64,
                stable_capacity_bytes: 60,
                current_free_bytes: Some(20),
                shares_system_memory: true,
                devices: vec![HardwareDevice {
                    id: "metal".to_owned(),
                    native_index: 1,
                    backend: "MTL".to_owned(),
                    physical_id: Some("metal-0".to_owned()),
                    name: "MTL0".to_owned(),
                    description: "Apple".to_owned(),
                    kind: HardwareDeviceKind::Gpu,
                    memory_limit: None,
                }],
            }],
            resident_memory: None,
        };
        let evidence = vec![
            ResidentAllocation {
                location: LlamaMemoryLocation::Host,
                model_bytes: 3,
                context_bytes: 2,
                compute_bytes: 1,
                auxiliary_bytes: 0,
            },
            ResidentAllocation {
                location: LlamaMemoryLocation::Device {
                    backend: "MTL".to_owned(),
                    physical_id: Some("metal-0".to_owned()),
                    native_index: 1,
                },
                model_bytes: 5,
                context_bytes: 4,
                compute_bytes: 3,
                auxiliary_bytes: 2,
            },
        ];

        let resident = resident_memory_state(&snapshot, &evidence, "model".to_owned(), 7)
            .expect("exact device identities resolve");
        assert_eq!(resident.domains.len(), 1);
        assert_eq!(resident.domains[0].memory_domain_id, "system");
        assert_eq!(resident.domains[0].model_bytes, 8);
        assert_eq!(resident.domains[0].context_bytes, 6);
        assert_eq!(resident.domains[0].compute_bytes, 4);
        assert_eq!(resident.domains[0].auxiliary_bytes, 2);
    }
}
