//! Persistent llama.cpp executor for ICN.

use std::collections::VecDeque;
use std::num::{NonZeroI32, NonZeroU32};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{
    Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError, sync_channel,
};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use icn_contracts::models::{ModelInstanceAllocation, ModelInstanceMemoryDomain};
use icn_contracts::output::{StopBuffer, Utf8Buffer};
use icn_contracts::{
    AllowedToolsMode, CacheType, ChatContent, ChatContentPart, ChatRequest, ChatTemplateRequest,
    CompletionBackend, ExecutionConfig, ExecutionConfigReport, ExecutionIntent, FinishReason,
    FlashAttention, Generation, GenerationMetrics, GenerationSnapshot, GpuLayers, GrammarTrigger,
    HardwareAssessment, HardwareSnapshot, ImageInput, InferenceError, InferenceEvent,
    InferenceProgress, InferenceStreamEvent, MemoryAccountant, MemoryBreakdown, MemoryCharge,
    MemoryChargeOwner, MemoryLocation, MemoryTopology, ModelModalities, ModelProperties,
    NativeDeviceLocator, PreparedChatInfo, ProjectorConfig, ReasoningControl, ResponseFormat,
    SplitMode, TemplateCapabilities, ToolCall, ToolChoice,
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
use llama_cpp_2::speculative::{
    SpeculativeDraft, SpeculativeMethod as NativeSpeculativeMethod, SpeculativeOperations,
    SpeculativeParams, SpeculativeSession, SpeculativeVerificationResolution,
};
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

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ModelPlanDefaults {
    pub context_size: u32,
    pub physical_context_size: u32,
    pub batch_size: u32,
    pub ubatch_size: u32,
    pub max_sequences: u32,
    pub prefill_quantum: u32,
    pub execution: ExecutionConfig,
    pub projector_use_gpu: bool,
    pub projector_warmup: bool,
    pub image_min_tokens: Option<NonZeroU32>,
    pub image_max_tokens: Option<NonZeroU32>,
}

#[must_use]
pub fn model_plan_defaults() -> ModelPlanDefaults {
    ModelPlanDefaults {
        // Managed product models overwrite context from their serving configuration. This
        // conservative value remains the discovery fallback for unmanaged local artifacts.
        context_size: 4096,
        physical_context_size: 4096,
        batch_size: 512,
        ubatch_size: 512,
        max_sequences: 1,
        prefill_quantum: 512,
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

#[must_use]
pub fn execution_intent(
    model_path: PathBuf,
    projector_path: Option<PathBuf>,
    defaults: &ModelPlanDefaults,
) -> ExecutionIntent {
    ExecutionIntent {
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
        speculative: icn_contracts::SpeculativeDecodingConfig::default(),
    }
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
        speculative: icn_contracts::SpeculativeDecodingConfig,
        hardware: HardwareSnapshot,
    ) -> Result<PreparedModelLoad, ModelLoadError> {
        let topology = MemoryTopology::from_snapshot(&hardware).ok_or_else(|| {
            ModelLoadError::Planning("load request contains an invalid memory topology".to_owned())
        })?;
        PreparedModelLoad::prepare(
            Arc::clone(&self.backend),
            model_id.into(),
            config,
            speculative,
            topology,
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
    ActiveSequence, BatchPlanner, BatchWork, PromptCheckpoint, PromptCheckpointState,
    ReusablePrefix, SequencePool, WorkCandidate, WorkKind,
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

struct ModelInstanceObservationRequest {
    policy: icn_hardware::CapacityPolicy,
    native_build: String,
    enabled_backends: Vec<String>,
    response: SyncSender<Result<ModelInstanceObservation, ModelInstanceObservationError>>,
}

#[derive(Debug)]
pub enum ModelInstanceObservationError {
    ExecutorStopped,
    MemoryDomainUnresolved { location: String },
}

impl std::fmt::Display for ModelInstanceObservationError {
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

impl std::error::Error for ModelInstanceObservationError {}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelInstanceObservation {
    pub hardware: HardwareSnapshot,
    pub allocation: ModelInstanceAllocation,
}

#[derive(Clone)]
struct ResidentAllocation {
    location: LlamaMemoryLocation,
    memory: MemoryBreakdown,
}

impl From<LlamaMemoryBreakdown> for ResidentAllocation {
    fn from(value: LlamaMemoryBreakdown) -> Self {
        Self {
            location: value.location,
            memory: MemoryBreakdown::new(
                value.model_bytes,
                value.context_bytes,
                value.compute_bytes,
                0,
            ),
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
    ObserveModelInstance(ModelInstanceObservationRequest),
    Shutdown,
}

enum ExecutorItem {
    Event(InferenceStreamEvent),
    Completed(Generation),
    Failed(InferenceError),
}

struct QueuedCompletion {
    request: ChatRequest,
    prepared: Option<PreparedInput>,
    events: SyncSender<ExecutorItem>,
    cancelled: Arc<AtomicBool>,
    queued_at: Instant,
    span: tracing::Span,
}

struct PreparedInput {
    chat: PreparedChat,
    prompt: TokenizedPrompt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestPhase {
    Prefill,
    ReadyToSample {
        batch_index: i32,
    },
    Decode {
        token: LlamaToken,
        position: scheduler::PromptBoundary,
    },
    Terminal,
}

/// Request-state changes earned by one successful target and linked-draft native batch.
/// Assembly records effects here so active requests never expose staged prompt progress.
struct BatchCommit {
    started_at: Instant,
    prompt_ends: Vec<(i32, scheduler::PromptBoundary)>,
    speculative_indices: Vec<(i32, Vec<i32>)>,
    logits: Vec<(i32, i32)>,
}

impl BatchCommit {
    fn new(started_at: Instant) -> Self {
        Self {
            started_at,
            prompt_ends: Vec::new(),
            speculative_indices: Vec::new(),
            logits: Vec::new(),
        }
    }

    fn prompt_start(
        &self,
        sequence_id: i32,
        committed: scheduler::PromptBoundary,
    ) -> scheduler::PromptBoundary {
        self.prompt_ends
            .iter()
            .rev()
            .find_map(|(id, end)| (*id == sequence_id).then_some(*end))
            .unwrap_or(committed)
    }

    fn advance_prompt(&mut self, sequence_id: i32, end: scheduler::PromptBoundary) {
        if let Some((_, current)) = self
            .prompt_ends
            .iter_mut()
            .find(|(id, _)| *id == sequence_id)
        {
            *current = end;
        } else {
            self.prompt_ends.push((sequence_id, end));
        }
    }

    fn record_speculative_indices(&mut self, sequence_id: i32, indices: Vec<i32>) {
        self.speculative_indices.push((sequence_id, indices));
    }

    fn record_logits(&mut self, sequence_id: i32, batch_index: i32) {
        self.logits.push((sequence_id, batch_index));
    }

    fn apply(self, active: &mut [ActiveRequest<'_>]) -> Result<(), InferenceError> {
        for (sequence_id, boundary) in self.prompt_ends {
            let request = request_by_sequence(active, sequence_id)?;
            request.prompt_started_at.get_or_insert(self.started_at);
            request.processed_prompt_tokens = boundary.logical_tokens;
            request.next_boundary = boundary;
            request.pending_progress = Some(InferenceProgress::Prefill {
                completed_tokens: boundary.logical_tokens,
                total_tokens: request.prompt_tokens,
                cached_tokens: request.cached_prompt_tokens,
            });
        }
        for (sequence_id, indices) in self.speculative_indices {
            request_by_sequence(active, sequence_id)?.speculative_indices = indices;
        }
        for (sequence_id, batch_index) in self.logits {
            let request = request_by_sequence(active, sequence_id)?;
            if let RequestPhase::Decode { token, .. } = request.phase {
                request.token_history.push(token);
            }
            request.phase = RequestPhase::ReadyToSample { batch_index };
        }
        Ok(())
    }
}

struct ActiveRequest<'model> {
    sequence: Option<ActiveSequence>,
    events: SyncSender<ExecutorItem>,
    span: tracing::Span,
    cancelled: Arc<AtomicBool>,
    outbound: VecDeque<ExecutorItem>,
    pending_progress: Option<InferenceProgress>,
    last_progress_emitted_at: Option<Instant>,
    phase: RequestPhase,
    prompt_layout: scheduler::PromptLayout,
    /// Full logical prompt followed by generated tokens committed to target KV. During prefill,
    /// `processed_prompt_tokens`, rather than this history, is the resident prompt boundary.
    token_history: Vec<LlamaToken>,
    processed_prompt_tokens: usize,
    prompt_tokens: usize,
    cached_prompt_tokens: usize,
    prompt_checkpoints: Vec<PromptCheckpoint>,
    pending_checkpoint_prefixes: VecDeque<scheduler::PromptBoundary>,
    next_boundary: scheduler::PromptBoundary,
    multimodal_prompt: Option<MultimodalPrompt>,
    generation_limit: usize,
    generated_tokens: usize,
    speculative_started: bool,
    speculative_draft: SpeculativeDraft,
    speculative_indices: Vec<i32>,
    speculative_replaying: bool,
    sampling_temperature: f32,
    draft_tokens: usize,
    proposal_distribution_draft_tokens: usize,
    accepted_draft_tokens: usize,
    draft_ms: f64,
    verification_ms: f64,
    cache_prompt: bool,
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
    layout: scheduler::PromptLayout,
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
        speculative: icn_contracts::SpeculativeDecodingConfig,
        topology: MemoryTopology,
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
                let result = prepare_native_plan(backend.as_ref(), &topology, config, speculative);
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
    SpeculativePreflight(String),
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
            Self::SpeculativePreflight(message) => {
                write!(formatter, "speculative selection failed: {message}")
            }
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

    /// Capture the resident model instance and current hardware topology between scheduler batches
    /// without waiting for inference to become idle.
    pub fn observe_model_instance(
        &self,
        policy: icn_hardware::CapacityPolicy,
        native_build: String,
        enabled_backends: Vec<String>,
    ) -> Result<ModelInstanceObservation, ModelInstanceObservationError> {
        let (response, receiver) = sync_channel(1);
        self.commands
            .send(ExecutorCommand::ObserveModelInstance(
                ModelInstanceObservationRequest {
                    policy,
                    native_build,
                    enabled_backends,
                    response,
                },
            ))
            .map_err(|_| ModelInstanceObservationError::ExecutorStopped)?;
        receiver
            .recv()
            .map_err(|_| ModelInstanceObservationError::ExecutorStopped)?
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
    topology: &MemoryTopology,
    mut requested: ExecutionIntent,
    speculative: icn_contracts::SpeculativeDecodingConfig,
) -> Result<(icn_hardware::BackendLoadPlan, String, Vec<ModelLoadPhase>), ModelLoadError> {
    requested.speculative = speculative;
    requested.speculative = icn_speculative::preflight_with_backend(backend, &requested)
        .map_err(|error| ModelLoadError::SpeculativePreflight(error.to_string()))?;
    let planned = match icn_hardware::plan_load_with_backend(backend, topology, &requested)
        .map_err(|error| ModelLoadError::Planning(error.to_string()))?
    {
        icn_hardware::BackendLoadPlanningOutcome::Planned(planned) => planned,
        icn_hardware::BackendLoadPlanningOutcome::Rejected(assessed) => {
            return Err(ModelLoadError::AssessmentRejected(Box::new(
                assessed.assessment,
            )));
        }
    };
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
        planned.assessed.plan.speculative,
        icn_contracts::SpeculativeDecodingConfig::Enabled {
            source: icn_contracts::SpeculativeDraftSource::Separate { .. },
            ..
        }
    ) {
        phases.push(ModelLoadPhase::DraftModel);
    }
    if matches!(
        planned.assessed.plan.speculative,
        icn_contracts::SpeculativeDecodingConfig::Enabled { .. }
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
    let speculative = match &config.speculative {
        icn_contracts::SpeculativeDecodingConfig::Disabled { .. } => {
            serde_json::json!({ "enabled": false })
        }
        icn_contracts::SpeculativeDecodingConfig::Enabled {
            source,
            method,
            n_max,
            n_min,
            cache_type_k,
            cache_type_v,
        } => serde_json::json!({
            "enabled": true,
            "source": match source {
                icn_contracts::SpeculativeDraftSource::Embedded => "bundled",
                icn_contracts::SpeculativeDraftSource::Separate { .. } => "separate",
            },
            "nMax": n_max,
            "nMin": n_min,
            "method": method,
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
        "physicalContextSize": config.physical_context_size,
        "batchSize": config.batch_size,
        "ubatchSize": config.ubatch_size,
        "maxSequences": config.max_sequences,
        "prefillQuantum": config.prefill_quantum,
        "execution": config.execution,
        "projector": projector,
        "speculative": speculative,
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
            memory: MemoryBreakdown::new(0, 0, 0, estimate.bytes),
        })
        .collect::<Vec<_>>();
    #[cfg(not(feature = "mtmd"))]
    let auxiliary_allocations = Vec::<ResidentAllocation>::new();
    let config = planned.assessed.plan;
    let native_speculative = planned.native_speculative.map(|plan| plan.into_parts());
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
    let draft_model = match (&config.speculative, native_speculative.as_ref()) {
        (
            icn_contracts::SpeculativeDecodingConfig::Enabled {
                source: icn_contracts::SpeculativeDraftSource::Separate { model_path },
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
        (icn_contracts::SpeculativeDecodingConfig::Enabled { .. }, None) => {
            let _ = ready.send(Err(InferenceError::InvalidConfig(
                "native planner omitted the enabled speculative plan".to_owned(),
            )
            .into()));
            return;
        }
        _ => None,
    };
    let draft_has_separate_model = draft_model.is_some();
    let mut speculative = match &config.speculative {
        icn_contracts::SpeculativeDecodingConfig::Disabled { .. } => None,
        icn_contracts::SpeculativeDecodingConfig::Enabled {
            n_max,
            n_min,
            method,
            ..
        } => {
            observer.phase_started(ModelLoadPhase::DraftContext);
            let Some((_, _, draft_context_params, _, _)) = native_speculative.as_ref() else {
                let _ = ready.send(Err(InferenceError::InvalidConfig(
                    "native planner omitted the enabled speculative context".to_owned(),
                )
                .into()));
                return;
            };
            let draft_context_params = draft_context_params.clone();
            let draft_model = draft_model.as_ref().unwrap_or(&model);
            match SpeculativeSession::new_linked(
                context.take().expect("target context is constructed once"),
                draft_model,
                &backend,
                draft_context_params,
                SpeculativeParams {
                    method: match method {
                        icn_contracts::SpeculativeMethodConfig::Mtp {
                            min_draft_probability,
                        } => NativeSpeculativeMethod::Mtp {
                            min_draft_probability: *min_draft_probability,
                        },
                        icn_contracts::SpeculativeMethodConfig::DFlash {
                            min_sample_probability,
                        } => NativeSpeculativeMethod::DFlash {
                            min_sample_probability: *min_sample_probability,
                        },
                        icn_contracts::SpeculativeMethodConfig::DSpark {
                            acceptance_threshold,
                        } => NativeSpeculativeMethod::DSpark {
                            acceptance_threshold: *acceptance_threshold,
                        },
                    },
                    n_max: i32::try_from(*n_max).unwrap_or(i32::MAX),
                    n_min: i32::try_from(*n_min).unwrap_or(i32::MAX),
                },
                config.max_sequences,
            ) {
                Ok(speculative) => {
                    observer.phase_completed(ModelLoadPhase::DraftContext);
                    Some(speculative)
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
    if let Some(speculative) = speculative.as_mut() {
        let mut draft_main_pool =
            match LlamaThreadPool::new(&backend, &LlamaThreadPoolParams::new(threads)) {
                Ok(pool) => pool,
                Err(error) => {
                    let _ = ready.send(Err(backend_error(error).into()));
                    return;
                }
            };
        let (context, draft_context, mut operations) = speculative.split_all_mut();
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
            .expect("non-speculative target context remains owned");
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
            .expect("non-speculative target context remains owned");
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
    mut speculative: Option<&mut SpeculativeOperations<'_>>,
    multimodal: &mut Option<MultimodalRuntime<'model>>,
    commands: &Receiver<ExecutorCommand>,
    ready: &SyncSender<Result<(ModelProperties, String), ModelLoadError>>,
    acceleration: String,
    observer: &dyn ModelLoadObserver,
) {
    observer.phase_started(ModelLoadPhase::Warmup);
    if let Err(error) = warm_up(model, context, speculative.as_deref_mut()) {
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
        speculative,
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
                allocation.memory = allocation.memory.without_model();
            }
        }
        allocations.extend(draft);
    }
    allocations.extend_from_slice(auxiliary_allocations);
    Ok(allocations)
}

fn model_instance_allocation(
    snapshot: &HardwareSnapshot,
    allocations: &[ResidentAllocation],
    config: &ExecutionIntent,
) -> Result<ModelInstanceAllocation, ModelInstanceObservationError> {
    let topology = MemoryTopology::from_snapshot(snapshot).ok_or_else(|| {
        ModelInstanceObservationError::MemoryDomainUnresolved {
            location: "hardware snapshot contains an invalid memory topology".to_owned(),
        }
    })?;
    let mut accountant = MemoryAccountant::new(&topology);
    for allocation in allocations {
        let location = match &allocation.location {
            LlamaMemoryLocation::Host => MemoryLocation::Host,
            LlamaMemoryLocation::Device {
                backend,
                physical_id,
                native_index,
                ..
            } => MemoryLocation::NativeDevice(NativeDeviceLocator::observed(
                backend,
                physical_id.clone(),
                *native_index,
            )),
        };
        let charge = MemoryCharge::new(
            MemoryChargeOwner::ResidentRuntime,
            location,
            allocation.memory,
        );
        accountant.record(charge).map_err(|error| {
            ModelInstanceObservationError::MemoryDomainUnresolved {
                location: format!("{:?}", error.location),
            }
        })?;
    }
    Ok(ModelInstanceAllocation {
        context_window_tokens: config.context_size,
        parallel_sequences: config.max_sequences,
        physical_context_tokens: config.physical_context_size,
        memory_domains: accountant
            .finish()
            .domains
            .into_iter()
            .map(|domain| ModelInstanceMemoryDomain {
                memory_domain_id: domain.id,
                model_bytes: domain.memory.model_bytes,
                context_bytes: domain.memory.context_bytes,
                compute_bytes: domain.memory.compute_bytes,
                auxiliary_bytes: domain.memory.auxiliary_bytes,
            })
            .collect(),
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
    mut speculative: Option<&mut SpeculativeOperations<'_>>,
    multimodal: &mut Option<MultimodalRuntime<'model>>,
    commands: &Receiver<ExecutorCommand>,
    resident_allocations: Vec<ResidentAllocation>,
) {
    let mut sequence_pool = SequencePool::new(config.max_sequences);
    let mut planner = BatchPlanner::new(config.prefill_quantum as usize);
    let mut decode_buffer = LlamaBatch::new(context.n_batch() as usize, 1);
    let mut queued = VecDeque::<QueuedCompletion>::new();
    let mut exclusive_native = VecDeque::<ExclusiveNativeTask>::new();
    let mut model_instance_observations = VecDeque::<ModelInstanceObservationRequest>::new();
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
            &mut model_instance_observations,
            &active,
            max_tracked,
            &mut shutting_down,
        );

        if let Some(observation) = model_instance_observations.pop_front() {
            let hardware = icn_hardware::discover_hardware(
                backend,
                observation.policy,
                observation.native_build,
                observation.enabled_backends,
            );
            let observed = model_instance_allocation(&hardware, &resident_allocations, config).map(
                |allocation| ModelInstanceObservation {
                    hardware,
                    allocation,
                },
            );
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
                        &mut model_instance_observations,
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

        cleanup_requests(
            context,
            speculative.as_deref_mut(),
            &mut sequence_pool,
            &mut active,
        );

        if shutting_down {
            fail_queued(&mut queued, InferenceError::ExecutorStopped);
            fail_active(
                context,
                speculative.as_deref_mut(),
                &mut sequence_pool,
                &mut active,
                InferenceError::ExecutorStopped,
            );
            cleanup_requests(
                context,
                speculative.as_deref_mut(),
                &mut sequence_pool,
                &mut active,
            );
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
                speculative.as_deref_mut(),
                &mut sequence_pool,
                &mut queued,
                &mut active,
                config.context_size as usize,
            );
        }

        sample_ready_requests(
            model,
            context,
            speculative.as_deref_mut(),
            &mut sequence_pool,
            &mut active,
        );
        cleanup_requests(
            context,
            speculative.as_deref_mut(),
            &mut sequence_pool,
            &mut active,
        );

        let decoded = if shutting_down {
            false
        } else {
            match decode_batch(
                model,
                context,
                draft_context.as_deref_mut(),
                speculative.as_deref_mut(),
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
                    // The whole native context includes available sequences too. Remove their
                    // reusable prefixes before later admission can observe a false cache hit.
                    sequence_pool.invalidate_reuse();
                    let failure = if matches!(error, InferenceError::Cancelled) {
                        InferenceError::Cancelled
                    } else {
                        InferenceError::Backend(error.to_string())
                    };
                    fail_active_after_context_reset(&mut sequence_pool, &mut active, failure);
                    false
                }
            }
        };

        cleanup_requests(
            context,
            speculative.as_deref_mut(),
            &mut sequence_pool,
            &mut active,
        );

        if !decoded {
            match commands.recv_timeout(IDLE_POLL_INTERVAL) {
                Ok(command) => handle_command(
                    command,
                    chat_templates,
                    multimodal.as_ref().map(multimodal_marker),
                    &mut queued,
                    &mut exclusive_native,
                    &mut model_instance_observations,
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
    model_instance_observations: &mut VecDeque<ModelInstanceObservationRequest>,
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
                model_instance_observations,
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
    model_instance_observations: &mut VecDeque<ModelInstanceObservationRequest>,
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
                let _ = events.try_send(ExecutorItem::Event(InferenceStreamEvent {
                    delta: InferenceEvent::Progress(InferenceProgress::Queued),
                    timings: None,
                }));
                queued.push_back(QueuedCompletion {
                    request,
                    prepared: None,
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
        ExecutorCommand::ObserveModelInstance(observation) => {
            if *shutting_down {
                drop(observation.response);
            } else {
                model_instance_observations.push_back(observation);
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
    mut speculative: Option<&mut SpeculativeOperations<'_>>,
    sequence_pool: &mut SequencePool,
    queued: &mut VecDeque<QueuedCompletion>,
    active: &mut Vec<ActiveRequest<'model>>,
    shared_context_capacity: usize,
) {
    while !queued.is_empty() {
        if queued
            .front()
            .is_some_and(|queued| queued.cancelled.load(Ordering::Acquire))
        {
            let cancelled = queued.pop_front().expect("queue front exists");
            let _ = cancelled
                .events
                .try_send(ExecutorItem::Failed(InferenceError::Cancelled));
            continue;
        }
        if sequence_pool.is_empty() {
            break;
        }
        if queued
            .front()
            .is_some_and(|queued| queued.prepared.is_none())
        {
            let pending = queued.front_mut().expect("queue front exists");
            let _ = pending
                .events
                .try_send(ExecutorItem::Event(InferenceStreamEvent {
                    delta: InferenceEvent::Progress(InferenceProgress::Preparing),
                    timings: None,
                }));
            match prepare_input(model, chat_templates, multimodal, &pending.request) {
                Ok(prepared) => pending.prepared = Some(prepared),
                Err(error) => {
                    let failed = queued.pop_front().expect("queue front exists");
                    let _ = failed.events.try_send(ExecutorItem::Failed(error));
                    continue;
                }
            }
        }
        let queued_front = queued.front().expect("queue front exists");
        let acquired = if queued_front.request.cache_prompt {
            sequence_pool.acquire_matching(
                &queued_front
                    .prepared
                    .as_ref()
                    .expect("request was prepared")
                    .prompt
                    .layout,
            )
        } else {
            sequence_pool.acquire()
        };
        let Some(mut acquired) = acquired else {
            break;
        };
        let sequence_id = acquired.id();
        let queued_request = queued
            .pop_front()
            .expect("queue was checked before acquiring a sequence");
        if queued_request.cancelled.load(Ordering::Acquire) {
            let _ = queued_request
                .events
                .try_send(ExecutorItem::Failed(InferenceError::Cancelled));
            sequence_pool.release(acquired);
            continue;
        }
        let available_prefix = acquired.reusable_prefix.as_ref();
        match ActiveRequest::admit(
            model,
            shared_context_capacity,
            context.n_batch() as usize,
            context.n_ubatch() as usize,
            queued_request,
            available_prefix,
        ) {
            Ok(mut request) => {
                let reusable_prefix = acquired.reusable_prefix.take();
                let sequence = acquired.activate();
                let requested_start = request.next_boundary;
                let partial = clear_sequence_range(
                    context,
                    speculative.as_deref_mut(),
                    sequence_id,
                    requested_start,
                    None,
                );
                if partial.is_err() && requested_start.logical_tokens == 0 {
                    let _ = request
                        .events
                        .try_send(ExecutorItem::Failed(InferenceError::Backend(format!(
                            "llama.cpp refused to reset sequence {sequence_id}"
                        ))));
                    sequence.quarantine();
                    continue;
                }
                if partial.is_err() {
                    let checkpoint = reusable_prefix.as_ref().and_then(|prefix| {
                        prefix.checkpoints.iter().rev().find(|checkpoint| {
                            checkpoint.boundary.logical_tokens <= requested_start.logical_tokens
                        })
                    });
                    let restored = checkpoint.is_some_and(|checkpoint| {
                        restore_prompt_checkpoint(
                            context,
                            draft_context.as_deref_mut(),
                            speculative.as_deref_mut(),
                            sequence_id,
                            checkpoint,
                        )
                    });
                    let restored_boundary = checkpoint
                        .filter(|_| restored)
                        .map_or_else(scheduler::PromptBoundary::default, |value| value.boundary);
                    request.processed_prompt_tokens = restored_boundary.logical_tokens;
                    request.next_boundary = restored_boundary;
                    request.cached_prompt_tokens = request.processed_prompt_tokens;
                    request.pending_progress = Some(InferenceProgress::Prefill {
                        completed_tokens: request.processed_prompt_tokens,
                        total_tokens: request.prompt_tokens,
                        cached_tokens: request.cached_prompt_tokens,
                    });
                    if clear_sequence_range(
                        context,
                        speculative.as_deref_mut(),
                        sequence_id,
                        request.next_boundary,
                        None,
                    )
                    .is_err()
                    {
                        let _ =
                            request
                                .events
                                .try_send(ExecutorItem::Failed(InferenceError::Backend(format!(
                                    "llama.cpp refused to reset cached sequence {sequence_id}"
                                ))));
                        sequence.quarantine();
                        continue;
                    }
                }
                request.sequence = Some(sequence);
                active.push(request);
            }
            Err((events, error)) => {
                let _ = events.try_send(ExecutorItem::Failed(error));
                // Admission performs no native mutation before returning an error.
                sequence_pool.release(acquired);
            }
        }
    }
}

fn sample_ready_requests<'model>(
    model: &'model LlamaModel,
    context: &mut LlamaContext<'model>,
    mut speculative: Option<&mut SpeculativeOperations<'_>>,
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
            release_sequence(context, speculative.as_deref_mut(), sequence_pool, request);
            continue;
        }
        if !request.speculative_started
            && let Some(operations) = speculative.as_deref_mut()
        {
            let sequence_id = request
                .sequence_id()
                .expect("ready request owns a sequence");
            if let Err(error) = operations.begin(sequence_id, &request.token_history) {
                fail_request(request, backend_error(error));
                discard_sequence(context, speculative.as_deref_mut(), sequence_pool, request);
                continue;
            }
            request.speculative_started = true;
        }
        match request.sample_next(model, context, batch_index) {
            Ok(Some(reason)) => {
                if let Err(error) = request.complete(reason) {
                    fail_request(request, error);
                }
                release_sequence(context, speculative.as_deref_mut(), sequence_pool, request);
            }
            Ok(None) => {}
            Err(error) => {
                fail_request(request, error);
                release_sequence(context, speculative.as_deref_mut(), sequence_pool, request);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_batch<'model>(
    model: &'model LlamaModel,
    context: &mut LlamaContext<'model>,
    draft_context: Option<&mut LlamaContext<'model>>,
    mut speculative: Option<&mut SpeculativeOperations<'_>>,
    multimodal: &mut Option<MultimodalRuntime<'model>>,
    planner: &mut BatchPlanner,
    batch: &mut LlamaBatch<'_>,
    active: &mut [ActiveRequest<'model>],
) -> Result<bool, InferenceError> {
    if decode_multimodal_prefill(context, speculative.as_deref_mut(), multimodal, active)? {
        return Ok(true);
    }

    let can_checkpoint_prompt = speculative.is_none() || draft_context.is_some();
    if can_checkpoint_prompt {
        for request in active.iter_mut().filter(|request| {
            request.cache_prompt
                && matches!(request.phase, RequestPhase::Prefill)
                && request.pending_checkpoint_prefixes.front().copied()
                    == Some(scheduler::PromptBoundary {
                        logical_tokens: request.processed_prompt_tokens,
                        native_position: request.next_boundary.native_position,
                    })
        }) {
            let boundary = request
                .pending_checkpoint_prefixes
                .pop_front()
                .expect("checkpoint position was matched");
            let sequence_id = request
                .sequence_id()
                .expect("prefill request owns a sequence");
            let state = match speculative.as_deref_mut() {
                Some(operations) => draft_context.as_deref().and_then(|draft_context| {
                    match operations.capture_prompt_state(context, draft_context, sequence_id) {
                        Ok(state) => Some(PromptCheckpointState::Speculative(state)),
                        Err(error) => {
                            tracing::warn!(
                                sequence_id,
                                error = %error,
                                "failed to capture speculative prompt checkpoint"
                            );
                            None
                        }
                    }
                }),
                None => context
                    .capture_sequence_state(sequence_id, LlamaStateSeqFlags::PARTIAL_ONLY)
                    .ok()
                    .filter(|checkpoint| !checkpoint.is_empty())
                    .map(PromptCheckpointState::Target),
            };
            if let Some(state) = state {
                request
                    .prompt_checkpoints
                    .push(PromptCheckpoint { state, boundary });
                request
                    .prompt_checkpoints
                    .sort_by_key(|checkpoint| checkpoint.boundary.logical_tokens);
                if request.prompt_checkpoints.len() > 32 {
                    request.prompt_checkpoints.remove(0);
                }
            }
        }
    }

    let mut draft_extra_tokens = active
        .iter()
        .filter(|request| {
            request.sequence_id().is_some()
                && request.outbound.is_empty()
                && matches!(request.phase, RequestPhase::Decode { .. })
        })
        .map(|request| request.speculative_draft.len())
        .sum::<usize>();
    if let Some(operations) = speculative.as_mut() {
        let mut drafted_sequences = Vec::new();
        let started = Instant::now();
        let decode_count = active
            .iter()
            .filter(|request| {
                request.sequence_id().is_some()
                    && request.outbound.is_empty()
                    && matches!(request.phase, RequestPhase::Decode { .. })
            })
            .count();
        let mut extra_budget = (context.n_batch() as usize)
            .saturating_sub(decode_count)
            .saturating_sub(draft_extra_tokens);
        for request in active.iter_mut().filter(|request| {
            request.speculative_started
                && request.speculative_draft.is_empty()
                && request.sequence_id().is_some()
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
            let sequence_id = request
                .sequence_id()
                .expect("selected request owns sequence");
            operations
                .prepare_draft(
                    sequence_id,
                    position.speculative_position().ok_or_else(|| {
                        InferenceError::Backend("draft position exceeded i32::MAX".into())
                    })?,
                    token,
                    &request.token_history,
                    n_max,
                    request.sampling_temperature,
                    request.sampler.seed(),
                )
                .map_err(backend_error)?;
            extra_budget -= n_max;
            drafted_sequences.push(sequence_id);
        }
        if !drafted_sequences.is_empty() {
            operations.draft_all().map_err(backend_error)?;
            let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
            for sequence_id in drafted_sequences {
                let request = request_by_sequence(active, sequence_id)?;
                request.speculative_draft =
                    operations.take_draft(sequence_id).map_err(backend_error)?;
                draft_extra_tokens =
                    draft_extra_tokens.saturating_add(request.speculative_draft.len());
                request.draft_tokens = request
                    .draft_tokens
                    .saturating_add(request.speculative_draft.len());
                if request.speculative_draft.has_proposal_distributions() {
                    request.proposal_distribution_draft_tokens = request
                        .proposal_distribution_draft_tokens
                        .saturating_add(request.speculative_draft.len());
                }
                request.draft_ms += elapsed;
            }
        }
    }

    let candidates = active
        .iter()
        .filter(|request| {
            request.sequence_id().is_some()
                && request.outbound.is_empty()
                && !request.cancelled.load(Ordering::Acquire)
        })
        .filter_map(|request| {
            let sequence_id = request.sequence_id()?;
            let kind = match request.phase {
                RequestPhase::Prefill => WorkKind::Prefill {
                    remaining: request
                        .pending_checkpoint_prefixes
                        .front()
                        .filter(|_| can_checkpoint_prompt && request.cache_prompt)
                        .map_or_else(
                            || {
                                request
                                    .prompt_layout
                                    .text_tokens_at(request.processed_prompt_tokens)
                                    .map_or(0, <[LlamaToken]>::len)
                            },
                            |boundary| {
                                boundary
                                    .logical_tokens
                                    .saturating_sub(request.processed_prompt_tokens)
                                    .min(
                                        request
                                            .prompt_layout
                                            .text_tokens_at(request.processed_prompt_tokens)
                                            .map_or(0, <[LlamaToken]>::len),
                                    )
                            },
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
    let mut draft_positions = Vec::new();
    let batch_started = Instant::now();
    let mut commit = BatchCommit::new(batch_started);

    for work in plan {
        match work {
            BatchWork::Decode { sequence_id } => {
                let request = request_by_sequence(active, sequence_id)?;
                let RequestPhase::Decode { token, position } = request.phase else {
                    return Err(InferenceError::Backend(format!(
                        "scheduler selected sequence {sequence_id} for decode in the wrong state"
                    )));
                };
                batch
                    .add(token, position.native_position, &[sequence_id], true)
                    .map_err(backend_error)?;
                draft_positions.push(i32::try_from(position.logical_tokens).map_err(|_| {
                    InferenceError::Backend("draft position exceeded i32::MAX".into())
                })?);
                if request.speculative_draft.is_empty() {
                    commit.record_logits(sequence_id, batch.n_tokens() - 1);
                } else {
                    let mut indices = vec![batch.n_tokens() - 1];
                    for (offset, draft) in request
                        .speculative_draft
                        .tokens()
                        .iter()
                        .copied()
                        .enumerate()
                    {
                        let draft_position = position.advance(offset + 1).ok_or_else(|| {
                            InferenceError::Backend(
                                "speculative position exceeded its numeric range".into(),
                            )
                        })?;
                        batch
                            .add(draft, draft_position.native_position, &[sequence_id], true)
                            .map_err(backend_error)?;
                        draft_positions.push(
                            i32::try_from(draft_position.logical_tokens).map_err(|_| {
                                InferenceError::Backend("draft position exceeded i32::MAX".into())
                            })?,
                        );
                        indices.push(batch.n_tokens() - 1);
                    }
                    commit.record_speculative_indices(sequence_id, indices);
                }
            }
            BatchWork::Prefill {
                sequence_id,
                tokens,
            } => {
                let request = request_by_sequence(active, sequence_id)?;
                let start = commit.prompt_start(
                    sequence_id,
                    scheduler::PromptBoundary {
                        logical_tokens: request.processed_prompt_tokens,
                        native_position: request.next_boundary.native_position,
                    },
                );
                let prompt_tokens = request
                    .prompt_layout
                    .text_tokens_at(start.logical_tokens)
                    .ok_or_else(|| {
                        InferenceError::Backend(format!(
                            "scheduler selected text prefill at non-text token {}",
                            start.logical_tokens
                        ))
                    })?;
                if tokens > prompt_tokens.len() {
                    return Err(InferenceError::Backend(
                        "scheduler selected text prefill across a media boundary".into(),
                    ));
                }
                for (relative, token) in prompt_tokens[..tokens].iter().enumerate() {
                    let absolute = start.logical_tokens + relative;
                    let final_prompt_token = absolute + 1 == request.prompt_tokens;
                    batch
                        .add(
                            *token,
                            start
                                .native_position
                                .checked_add(i32::try_from(relative).map_err(backend_error)?)
                                .ok_or_else(|| {
                                    InferenceError::Backend(
                                        "prompt position exceeded i32::MAX".into(),
                                    )
                                })?,
                            &[sequence_id],
                            final_prompt_token,
                        )
                        .map_err(backend_error)?;
                    draft_positions.push(i32::try_from(absolute).map_err(|_| {
                        InferenceError::Backend("draft position exceeded i32::MAX".into())
                    })?);
                    if final_prompt_token {
                        commit.record_logits(sequence_id, batch.n_tokens() - 1);
                    }
                }
                commit.advance_prompt(
                    sequence_id,
                    scheduler::PromptBoundary {
                        logical_tokens: start.logical_tokens + tokens,
                        native_position: start
                            .native_position
                            .checked_add(i32::try_from(tokens).map_err(backend_error)?)
                            .ok_or_else(|| {
                                InferenceError::Backend("prompt position exceeded i32::MAX".into())
                            })?,
                    },
                );
            }
        }
    }

    let verification_started = Instant::now();
    context.decode(batch).map_err(backend_error)?;
    if let Some(operations) = speculative.as_mut() {
        operations
            .process(batch, &draft_positions)
            .map_err(backend_error)?;
    }
    commit.apply(active)?;
    let verification_ms = verification_started.elapsed().as_secs_f64() * 1_000.0;
    if let Some(operations) = speculative.as_mut() {
        verify_speculative_batch(model, context, operations, active, verification_ms)?;
    }
    Ok(true)
}

fn verify_speculative_batch<'model>(
    model: &'model LlamaModel,
    context: &mut LlamaContext<'model>,
    operations: &mut SpeculativeOperations<'_>,
    active: &mut [ActiveRequest<'model>],
    verification_ms: f64,
) -> Result<(), InferenceError> {
    for request in active
        .iter_mut()
        .filter(|request| !request.speculative_draft.is_empty())
    {
        let sequence_id = request.sequence_id().ok_or_else(|| {
            InferenceError::Backend("speculative request lost its sequence".into())
        })?;
        let RequestPhase::Decode {
            token: pending,
            position: verification_start_position,
        } = request.phase
        else {
            return Err(InferenceError::Backend(
                "speculative verification request was not in decode state".into(),
            ));
        };
        let sampler_checkpoint = request.sampler.snapshot().map_err(backend_error)?;
        let proposed_drafts = request.speculative_draft.len();
        let has_proposal_distributions = request.speculative_draft.has_proposal_distributions();
        let accepted = request
            .sampler
            .sample_and_accept_draft(
                context,
                &request.speculative_indices,
                &request.speculative_draft,
                false,
            )
            .map_err(backend_error)?;
        if accepted.is_empty() {
            return Err(InferenceError::Backend(
                "native speculative verification returned no continuation token".into(),
            ));
        }
        let accepted_drafts = accepted.len() - 1;
        tracing::debug!(
            sequence_id,
            proposed_drafts,
            accepted_drafts,
            has_proposal_distributions,
            "verified speculative draft"
        );
        let next_position = verification_start_position
            .advance(1_usize.saturating_add(accepted_drafts))
            .ok_or_else(|| {
                InferenceError::Backend("speculative position exceeded its numeric range".into())
            })?;
        let resolution = operations
            .resolve_verification(
                sequence_id,
                proposed_drafts,
                accepted_drafts,
                next_position.speculative_position().ok_or_else(|| {
                    InferenceError::Backend("draft position exceeded i32::MAX".into())
                })?,
            )
            .map_err(backend_error)?;
        if resolution == SpeculativeVerificationResolution::Replay {
            request
                .sampler
                .restore(&sampler_checkpoint)
                .map_err(backend_error)?;
            request.speculative_draft = SpeculativeDraft::from_tokens(accepted);
            request.speculative_indices.clear();
            request.speculative_replaying = true;
            continue;
        }

        request.token_history.push(pending);
        request
            .token_history
            .extend(accepted.iter().take(accepted_drafts).copied());
        let accepted_original_drafts =
            accepted_drafts.saturating_sub(usize::from(request.speculative_replaying));
        request.accepted_draft_tokens = request
            .accepted_draft_tokens
            .saturating_add(accepted_original_drafts);
        request.verification_ms += verification_ms;
        request.speculative_draft.clear();
        request.speculative_indices.clear();
        request.speculative_replaying = false;

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
            request.next_boundary = next_position.advance(1).ok_or_else(|| {
                InferenceError::Backend("generation position exceeded its numeric range".into())
            })?;
        }
    }
    Ok(())
}

#[cfg(feature = "mtmd")]
fn decode_multimodal_prefill<'model>(
    context: &mut LlamaContext<'model>,
    speculative: Option<&mut SpeculativeOperations<'_>>,
    multimodal: &mut Option<MultimodalRuntime<'model>>,
    active: &mut [ActiveRequest<'model>],
) -> Result<bool, InferenceError> {
    let Some(index) = active.iter().position(|request| {
        request.multimodal_prompt.is_some()
            && matches!(request.phase, RequestPhase::Prefill)
            && request
                .prompt_layout
                .media_at(request.processed_prompt_tokens)
                .is_some()
            && request.sequence_id().is_some()
            && request.outbound.is_empty()
            && !request.cancelled.load(Ordering::Acquire)
    }) else {
        return Ok(false);
    };
    let runtime = multimodal.as_mut().ok_or_else(|| {
        InferenceError::Backend("multimodal request was admitted without a projector".into())
    })?;
    let request = &mut active[index];
    let sequence_id = request
        .sequence_id()
        .ok_or_else(|| InferenceError::Backend("multimodal sequence lost ownership".into()))?;
    let batch_size = i32::try_from(context.n_batch()).map_err(backend_error)?;
    let started = Instant::now();
    request.prompt_started_at.get_or_insert(started);
    context.install_abort_callback_with_flag(Arc::clone(&request.cancelled));
    let result = runtime.evaluate_media(
        request
            .multimodal_prompt
            .as_ref()
            .expect("multimodal request was selected"),
        context,
        scheduler::PromptBoundary {
            logical_tokens: request.processed_prompt_tokens,
            native_position: request.next_boundary.native_position,
        },
        sequence_id,
        batch_size,
        speculative,
    );
    context.clear_abort_callback();
    if request.cancelled.load(Ordering::Acquire) {
        return Err(InferenceError::Cancelled);
    }
    let boundary = result?;
    request.next_boundary = boundary;
    request.processed_prompt_tokens = boundary.logical_tokens;
    request.pending_progress = Some(InferenceProgress::Prefill {
        completed_tokens: request.processed_prompt_tokens,
        total_tokens: request.prompt_tokens,
        cached_tokens: request.cached_prompt_tokens,
    });
    Ok(true)
}

#[cfg(not(feature = "mtmd"))]
fn decode_multimodal_prefill<'model>(
    _context: &mut LlamaContext<'model>,
    _speculative: Option<&mut SpeculativeOperations<'_>>,
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
        .find(|request| request.sequence_id() == Some(sequence_id))
        .ok_or_else(|| {
            InferenceError::Backend(format!(
                "scheduler referenced unowned sequence {sequence_id}"
            ))
        })
}

fn restore_prompt_checkpoint(
    context: &mut LlamaContext<'_>,
    draft_context: Option<&mut LlamaContext<'_>>,
    speculative: Option<&mut SpeculativeOperations<'_>>,
    sequence_id: i32,
    checkpoint: &PromptCheckpoint,
) -> bool {
    match (&checkpoint.state, draft_context, speculative) {
        (PromptCheckpointState::Target(state), None, None) => {
            context.restore_sequence_state(state, sequence_id)
        }
        (PromptCheckpointState::Speculative(state), Some(draft), Some(operations)) => {
            match operations.restore_prompt_state(context, draft, sequence_id, state) {
                Ok(()) => true,
                Err(error) => {
                    tracing::warn!(
                        sequence_id,
                        error = %error,
                        "failed to restore speculative prompt checkpoint"
                    );
                    false
                }
            }
        }
        _ => false,
    }
}

fn cleanup_requests(
    context: &mut LlamaContext<'_>,
    mut speculative: Option<&mut SpeculativeOperations<'_>>,
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
                speculative.as_deref_mut(),
                sequence_pool,
                &mut active[index],
            );
        }
        match flush_outbound(&mut active[index]) {
            FlushOutcome::Empty if matches!(active[index].phase, RequestPhase::Terminal) => {
                release_sequence(
                    context,
                    speculative.as_deref_mut(),
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
                    speculative.as_deref_mut(),
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
    if let Some(progress) = request.pending_progress {
        const PROGRESS_INTERVAL: Duration = Duration::from_millis(100);
        let final_prefill = matches!(
            progress,
            InferenceProgress::Prefill {
                completed_tokens,
                total_tokens,
                ..
            } if completed_tokens >= total_tokens
        );
        let due = final_prefill
            || request
                .last_progress_emitted_at
                .is_none_or(|last| last.elapsed() >= PROGRESS_INTERVAL);
        if due {
            match request
                .events
                .try_send(ExecutorItem::Event(InferenceStreamEvent {
                    delta: InferenceEvent::Progress(progress),
                    timings: None,
                })) {
                Ok(()) => {
                    request.pending_progress = None;
                    request.last_progress_emitted_at = Some(Instant::now());
                }
                Err(TrySendError::Full(_)) => {}
                Err(TrySendError::Disconnected(_)) => {
                    request.pending_progress = None;
                    return FlushOutcome::Disconnected;
                }
            }
        }
    }
    FlushOutcome::Empty
}

fn release_sequence(
    context: &mut LlamaContext<'_>,
    speculative: Option<&mut SpeculativeOperations<'_>>,
    sequence_pool: &mut SequencePool,
    request: &mut ActiveRequest<'_>,
) {
    let Some(sequence) = request.sequence.take() else {
        return;
    };
    if let Some(reusable_prefix) = request.take_reusable_prefix() {
        sequence_pool.release(sequence.into_available(Some(reusable_prefix)));
        return;
    }
    let sequence_id = sequence.id();
    // Full sequence removal is supported for every llama.cpp memory implementation. This is the
    // sole cache policy required by this milestone: a sequence is never reassigned while resident
    // state still belongs to the previous request.
    match clear_sequence(context, speculative, sequence_id) {
        Ok(()) => sequence_pool.release(sequence.into_available(None)),
        Err(error) => {
            // Never hand a sequence to another request unless native state removal succeeded.
            sequence.quarantine();
            request.phase = RequestPhase::Terminal;
            request.outbound.clear();
            request.outbound.push_back(ExecutorItem::Failed(error));
        }
    }
}

fn discard_sequence(
    context: &mut LlamaContext<'_>,
    speculative: Option<&mut SpeculativeOperations<'_>>,
    sequence_pool: &mut SequencePool,
    request: &mut ActiveRequest<'_>,
) {
    let Some(sequence) = request.sequence.take() else {
        return;
    };
    let sequence_id = sequence.id();
    match clear_sequence(context, speculative, sequence_id) {
        Ok(()) => sequence_pool.release(sequence.into_available(None)),
        Err(error) => {
            sequence.quarantine();
            request.phase = RequestPhase::Terminal;
            request.outbound.clear();
            request.outbound.push_back(ExecutorItem::Failed(error));
        }
    }
}

fn clear_sequence(
    context: &mut LlamaContext<'_>,
    speculative: Option<&mut SpeculativeOperations<'_>>,
    sequence_id: i32,
) -> Result<(), InferenceError> {
    clear_sequence_range(
        context,
        speculative,
        sequence_id,
        scheduler::PromptBoundary::default(),
        None,
    )
}

fn clear_sequence_range(
    context: &mut LlamaContext<'_>,
    speculative: Option<&mut SpeculativeOperations<'_>>,
    sequence_id: i32,
    start: scheduler::PromptBoundary,
    end: Option<scheduler::PromptBoundary>,
) -> Result<(), InferenceError> {
    if let Some(speculative) = speculative {
        let draft_end = end
            .map(|boundary| {
                boundary.speculative_position().ok_or_else(|| {
                    InferenceError::Backend("draft position exceeded i32::MAX".into())
                })
            })
            .transpose()?;
        return speculative
            .remove_sequence_range(
                sequence_id,
                start.speculative_position().ok_or_else(|| {
                    InferenceError::Backend("draft position exceeded i32::MAX".into())
                })?,
                draft_end,
            )
            .map_err(backend_error);
    }
    let sequence = u32::try_from(sequence_id).map_err(backend_error)?;
    let removed = context
        .clear_kv_cache_seq(
            Some(sequence),
            (start.native_position > 0).then_some(start.native_position as u32),
            end.map(|boundary| boundary.native_position as u32),
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
    if request.outbound.len() >= OUTBOUND_QUEUE_CAPACITY {
        request.outbound.clear();
    }
    request.outbound.push_back(ExecutorItem::Failed(error));
}

fn cancel_request(request: &mut ActiveRequest<'_>) {
    request.outbound.clear();
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
    mut speculative: Option<&mut SpeculativeOperations<'_>>,
    sequence_pool: &mut SequencePool,
    active: &mut [ActiveRequest<'_>],
    reason: InferenceError,
) {
    for request in active {
        if !matches!(request.phase, RequestPhase::Terminal) {
            fail_request(request, clone_inference_error(&reason));
        }
        release_sequence(context, speculative.as_deref_mut(), sequence_pool, request);
    }
}

fn fail_active_after_context_reset(
    sequence_pool: &mut SequencePool,
    active: &mut [ActiveRequest<'_>],
    reason: InferenceError,
) {
    for request in active {
        if !matches!(request.phase, RequestPhase::Terminal) {
            fail_request(request, clone_inference_error(&reason));
        }
        if let Some(sequence) = request.sequence.take() {
            sequence_pool.release(sequence.into_available(None));
        }
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
    if config.physical_context_size < config.context_size {
        return Err(InferenceError::InvalidConfig(
            "physical_context_size must be at least context_size".into(),
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
    if matches!(
        config.speculative,
        icn_contracts::SpeculativeDecodingConfig::Enabled {
            method: icn_contracts::SpeculativeMethodConfig::Mtp { .. },
            ..
        }
    ) {
        return Err(InferenceError::InvalidConfig(
            "multimodal projector mode does not support MTP because the native MTP drafter cannot consume media embedding batches"
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
    speculative: Option<&mut SpeculativeOperations<'_>>,
) -> Result<(), InferenceError> {
    let tokens = model
        .str_to_token(" ", AddBos::Always)
        .map_err(backend_error)?;
    if let Some(token) = tokens.first().copied() {
        let mut batch = LlamaBatch::new(1, 1);
        batch.add(token, 0, &[0], false).map_err(backend_error)?;
        context.decode(&mut batch).map_err(backend_error)?;
        if let Some(speculative) = speculative {
            speculative.process(&batch, &[0]).map_err(backend_error)?;
            speculative
                .remove_sequence_range(
                    0,
                    llama_cpp_2::speculative::SpeculativePosition {
                        target: 0,
                        draft: 0,
                    },
                    None,
                )
                .map_err(backend_error)?;
        } else {
            context.clear_kv_cache();
        }
        context.synchronize();
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
        speculative: match &config.speculative {
            icn_contracts::SpeculativeDecodingConfig::Disabled { reason } => {
                icn_contracts::SpeculativeDecodingRuntimeProperties::Disabled {
                    reason: reason.clone(),
                }
            }
            icn_contracts::SpeculativeDecodingConfig::Enabled {
                source,
                method,
                n_max,
                n_min,
                ..
            } => icn_contracts::SpeculativeDecodingRuntimeProperties::Enabled {
                source: source.clone(),
                method: method.clone(),
                n_max: *n_max,
                n_min: *n_min,
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

fn prepare_input(
    model: &LlamaModel,
    chat_templates: &CommonChatTemplates,
    multimodal: Option<&MultimodalRuntime<'_>>,
    request: &ChatRequest,
) -> Result<PreparedInput, InferenceError> {
    validate_request(request)?;
    let images = request_images(&request.template);
    let chat = prepare_chat(
        chat_templates,
        &request.template,
        multimodal.map(multimodal_marker),
    )?;
    let prompt = tokenize_prepared_prompt(model, &chat, multimodal, &images)?;
    Ok(PreparedInput { chat, prompt })
}

fn plain_prompt(
    model: &LlamaModel,
    prepared: &PreparedChat,
) -> Result<TokenizedPrompt, InferenceError> {
    let text_tokens = model
        .str_to_token(prepared.prompt(), AddBos::Always)
        .map_err(backend_error)?;
    let layout = scheduler::PromptLayout::text(text_tokens.clone());
    Ok(TokenizedPrompt {
        text_tokens,
        layout,
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
    let layout = prompt.layout().clone();
    Ok(TokenizedPrompt {
        text_tokens: layout.text_tokens(),
        layout,
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
    fn sequence_id(&self) -> Option<i32> {
        self.sequence.as_ref().map(ActiveSequence::id)
    }

    #[allow(clippy::too_many_arguments)]
    fn admit(
        model: &'model LlamaModel,
        context_capacity: usize,
        batch_size: usize,
        ubatch_size: usize,
        queued: QueuedCompletion,
        reusable_prefix: Option<&ReusablePrefix>,
    ) -> Result<Self, (SyncSender<ExecutorItem>, InferenceError)> {
        let QueuedCompletion {
            request,
            prepared,
            events,
            cancelled,
            queued_at,
            span,
        } = queued;
        let entered_span = span.clone();
        let _entered = entered_span.enter();
        let result = (|| {
            let PreparedInput {
                chat: prepared,
                prompt: tokenized,
            } = prepared.ok_or_else(|| {
                InferenceError::Backend("request reached admission without preparation".into())
            })?;
            let admitted_at = Instant::now();
            let parser = prepared
                .stream_parser(ChatParserOptions {
                    parse_tool_calls: !request.template.tools.is_empty()
                        && !matches!(request.template.tool_choice, ToolChoice::None),
                    ..ChatParserOptions::default()
                })
                .map_err(backend_error)?;
            if tokenized.text_tokens.is_empty() {
                return Err(InferenceError::InvalidConfig(
                    "the prepared prompt tokenized to an empty sequence".into(),
                ));
            }
            let prompt_tokens = tokenized.layout.logical_tokens();
            validate_prompt_capacity(prompt_tokens, context_capacity)?;

            let mut sampler = make_sampler(model, &request, &prepared)?;
            sampler
                .accept_prompt(tokenized.text_tokens.iter())
                .map_err(backend_error)?;
            let mut stops = request.stop.clone();
            stops.extend(prepared.additional_stops().iter().cloned());
            let mut cached_boundary = if request.cache_prompt {
                reusable_prefix.map_or_else(scheduler::PromptBoundary::default, |prefix| {
                    prefix.layout.common_prefix(&tokenized.layout)
                })
            } else {
                scheduler::PromptBoundary::default()
            };
            // The last prompt token must be evaluated to obtain logits for the first sample.
            if cached_boundary.logical_tokens == prompt_tokens {
                cached_boundary = tokenized
                    .layout
                    .boundary_before_final_text_token()
                    .unwrap_or_default();
            }
            let cached_prompt_tokens = cached_boundary.logical_tokens;
            let prompt_checkpoints = reusable_prefix.map_or_else(Vec::new, |prefix| {
                prefix
                    .checkpoints
                    .iter()
                    .filter(|checkpoint| checkpoint.boundary.logical_tokens <= cached_prompt_tokens)
                    .cloned()
                    .collect()
            });
            // Keep two bounded-memory prompt checkpoints: one micro-batch before the logits
            // token for changed prompt tails, and one immediately before it for exact reuse.
            // Hybrid recurrent models cannot partially erase native state, so the latter is
            // what lets an identical request replay only the final token needed to refresh
            // logits instead of falling back to an arbitrary multi-token tail.
            let mut pending_checkpoint_prefixes = [1_usize.saturating_add(ubatch_size), 1]
                .into_iter()
                .map(|offset| prompt_tokens.saturating_sub(offset.min(batch_size)))
                .filter(|prefix| *prefix > cached_prompt_tokens && *prefix > 0)
                .filter_map(|prefix| tokenized.layout.boundary_at_or_after(prefix))
                .collect::<Vec<_>>();
            pending_checkpoint_prefixes.sort_unstable_by_key(|boundary| boundary.logical_tokens);
            pending_checkpoint_prefixes.dedup();

            Ok(Self {
                sequence: None,
                events: events.clone(),
                span,
                cancelled,
                outbound: VecDeque::new(),
                pending_progress: Some(InferenceProgress::Prefill {
                    completed_tokens: cached_prompt_tokens,
                    total_tokens: prompt_tokens,
                    cached_tokens: cached_prompt_tokens,
                }),
                last_progress_emitted_at: None,
                phase: RequestPhase::Prefill,
                token_history: tokenized.text_tokens.clone(),
                prompt_layout: tokenized.layout,
                processed_prompt_tokens: cached_prompt_tokens,
                prompt_tokens,
                cached_prompt_tokens,
                prompt_checkpoints,
                pending_checkpoint_prefixes: pending_checkpoint_prefixes.into(),
                next_boundary: cached_boundary,
                multimodal_prompt: tokenized.multimodal,
                generation_limit: (request.max_tokens as usize)
                    .min(context_capacity.saturating_sub(prompt_tokens)),
                generated_tokens: 0,
                speculative_started: false,
                speculative_draft: SpeculativeDraft::default(),
                speculative_indices: Vec::new(),
                speculative_replaying: false,
                sampling_temperature: request.temperature,
                draft_tokens: 0,
                proposal_distribution_draft_tokens: 0,
                accepted_draft_tokens: 0,
                draft_ms: 0.0,
                verification_ms: 0.0,
                cache_prompt: request.cache_prompt,
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

    fn take_reusable_prefix(&mut self) -> Option<ReusablePrefix> {
        if !self.cache_prompt {
            return None;
        }
        let boundary = self
            .prompt_layout
            .boundary_at(self.processed_prompt_tokens)?;
        let layout = self.prompt_layout.prefix(boundary)?;
        if boundary.logical_tokens == 0 {
            return None;
        }
        debug_assert!(
            self.prompt_checkpoints.iter().all(
                |checkpoint| checkpoint.boundary.logical_tokens <= self.processed_prompt_tokens
            )
        );
        Some(ReusablePrefix {
            layout,
            checkpoints: std::mem::take(&mut self.prompt_checkpoints),
        })
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

        let position = self.next_boundary;
        self.next_boundary = self.next_boundary.advance(1).ok_or_else(|| {
            InferenceError::Backend("generation position exceeded its numeric range".into())
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
        if starts_stream {
            self.pending_progress = Some(InferenceProgress::Generating);
        }
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
                proposal_distribution_draft_tokens: self.proposal_distribution_draft_tokens,
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
    fn batch_commit_keeps_repeated_prompt_quanta_local_until_apply() {
        let mut commit = BatchCommit::new(Instant::now());
        let boundary = |value| scheduler::PromptBoundary {
            logical_tokens: value,
            native_position: i32::try_from(value).unwrap(),
        };
        assert_eq!(commit.prompt_start(2, boundary(7)), boundary(7));
        commit.advance_prompt(2, boundary(9));
        assert_eq!(commit.prompt_start(2, boundary(7)), boundary(9));
        commit.advance_prompt(2, boundary(11));
        assert_eq!(commit.prompt_ends, vec![(2, boundary(11))]);
    }

    struct NoopLoadObserver;

    impl ModelLoadObserver for NoopLoadObserver {
        fn phase_started(&self, _phase: ModelLoadPhase) {}

        fn phase_completed(&self, _phase: ModelLoadPhase) {}
    }

    #[test]
    #[ignore = "loads target and DFlash2 GGUFs selected through ICN_DFLASH2_TEST_* variables"]
    fn dflash2_uses_proposal_distributions_with_real_model() {
        disable_native_diagnostics();
        let required_path = |name: &str| {
            let path = PathBuf::from(
                std::env::var_os(name).unwrap_or_else(|| panic!("{name} must name a local file")),
            );
            assert!(path.is_file(), "missing {}", path.display());
            path
        };
        let model_path = required_path("ICN_DFLASH2_TEST_MODEL");
        let draft_path = required_path("ICN_DFLASH2_TEST_DRAFT");

        let speculative = icn_contracts::SpeculativeDecodingConfig::Enabled {
            source: icn_contracts::SpeculativeDraftSource::Separate {
                model_path: draft_path,
            },
            method: icn_contracts::SpeculativeMethodConfig::DFlash {
                min_sample_probability: 0.1,
            },
            n_max: 3,
            n_min: 0,
            cache_type_k: CacheType::F16,
            cache_type_v: CacheType::F16,
        };
        let native = NativeBackend::initialize().expect("initialize native backend");
        let hardware = native.discover_hardware(
            icn_hardware::CapacityPolicy::default(),
            "real-dflash2-distribution-test",
            Vec::new(),
        );
        let mut defaults = model_plan_defaults();
        defaults.context_size = 4_096;
        defaults.physical_context_size = 4_096;
        defaults.batch_size = 512;
        defaults.ubatch_size = 512;
        defaults.max_sequences = 1;
        defaults.prefill_quantum = 128;
        let mut intent = execution_intent(model_path, None, &defaults);
        intent.speculative = speculative.clone();
        let prepared = native
            .prepare_load(
                "installed-dflash2-distribution-test",
                intent,
                speculative,
                hardware,
            )
            .expect("prepare DFlash2 model load");
        let backend = prepared
            .execute(Arc::new(NoopLoadObserver))
            .expect("load DFlash2 model");

        let mut completion = request();
        completion.template.messages = vec![ChatMessage::text(
            ChatRole::User,
            "Explain in several sentences why speculative decoding can preserve the target model's sampling distribution.",
        )];
        completion.max_tokens = 48;
        completion.temperature = 0.8;
        completion.seed = 42;
        completion.ignore_eos = true;
        let generation = backend
            .complete(completion, &mut |_| Ok(()))
            .expect("complete DFlash2 request");

        assert!(
            generation.metrics.draft_tokens > 0,
            "DFlash2 produced no draft tokens"
        );
        assert_eq!(
            generation.metrics.proposal_distribution_draft_tokens, generation.metrics.draft_tokens,
            "not every DFlash2 draft carried a proposal distribution"
        );
        assert!(
            generation.metrics.accepted_draft_tokens > 0,
            "DFlash2 accepted no draft tokens"
        );
        eprintln!(
            "dflash2 drafted={} distribution_drafted={} accepted={} generated={}",
            generation.metrics.draft_tokens,
            generation.metrics.proposal_distribution_draft_tokens,
            generation.metrics.accepted_draft_tokens,
            generation.generated_tokens,
        );
    }

    #[test]
    #[ignore = "loads the repository's real 18 MB GGUF acceptance fixture"]
    fn interrupted_generation_reuses_prompt_prefix_with_real_model() {
        disable_native_diagnostics();
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../.parity-models/tinyllamas/stories15M-q4_0.gguf");
        assert!(model_path.is_file(), "missing {}", model_path.display());

        let native = NativeBackend::initialize().expect("initialize native backend");
        let hardware = native.discover_hardware(
            icn_hardware::CapacityPolicy::default(),
            "real-model-prompt-retention-test",
            Vec::new(),
        );
        let mut defaults = model_plan_defaults();
        defaults.context_size = 512;
        defaults.physical_context_size = 512;
        defaults.batch_size = 128;
        defaults.ubatch_size = 128;
        defaults.max_sequences = 1;
        defaults.prefill_quantum = 32;
        let intent = execution_intent(model_path, None, &defaults);
        let prepared = native
            .prepare_load(
                "stories15m-retention-test",
                intent,
                icn_contracts::SpeculativeDecodingConfig::default(),
                hardware,
            )
            .expect("prepare real model load");
        let backend = prepared
            .execute(Arc::new(NoopLoadObserver))
            .expect("load real model");

        let mut completion = request();
        completion.template.messages = vec![ChatMessage::text(
            ChatRole::User,
            "Write one short continuation for this story. The small red fox walked through the quiet forest every morning. It knew every mossy stone, every narrow path, and every bird song. Today it found a bright blue box beneath the oldest oak tree. The box was warm, and something inside made a gentle ticking sound.",
        )];
        completion.max_tokens = 8;
        completion.ignore_eos = true;

        let mut reached_generation = false;
        let interrupted = backend.complete(completion.clone(), &mut |event| {
            if matches!(
                event.delta,
                InferenceEvent::StreamStart
                    | InferenceEvent::ContentDelta { .. }
                    | InferenceEvent::ReasoningDelta { .. }
                    | InferenceEvent::ToolCallDelta { .. }
            ) {
                reached_generation = true;
                return Err(InferenceError::Callback(
                    "intentional real-model interruption".into(),
                ));
            }
            Ok(())
        });
        assert!(
            reached_generation,
            "request produced no stream event before returning {interrupted:?}"
        );
        assert!(matches!(interrupted, Err(InferenceError::Callback(_))));

        let mut observed_cached_tokens = 0;
        let completed = backend
            .complete(completion, &mut |event| {
                if let InferenceEvent::Progress(InferenceProgress::Prefill {
                    cached_tokens, ..
                }) = event.delta
                {
                    observed_cached_tokens = observed_cached_tokens.max(cached_tokens);
                }
                Ok(())
            })
            .expect("complete request after interruption");

        assert!(
            observed_cached_tokens > 0,
            "follow-up prefill did not report cached prompt tokens"
        );
        assert_eq!(completed.cached_prompt_tokens, observed_cached_tokens);
        assert!(completed.generated_tokens > 0);
    }

    #[test]
    #[ignore = "loads an installed vision model selected through ICN_VISION_TEST_* variables"]
    fn multimodal_prompt_reuse_with_real_model() {
        disable_native_diagnostics();
        let model_path = PathBuf::from(
            std::env::var_os("ICN_VISION_TEST_MODEL")
                .expect("ICN_VISION_TEST_MODEL must name an installed GGUF"),
        );
        let projector_path = PathBuf::from(
            std::env::var_os("ICN_VISION_TEST_PROJECTOR")
                .expect("ICN_VISION_TEST_PROJECTOR must name its projector GGUF"),
        );
        let image_path = PathBuf::from(
            std::env::var_os("ICN_VISION_TEST_IMAGE")
                .expect("ICN_VISION_TEST_IMAGE must name the cache probe image"),
        );
        let expected_text =
            std::env::var("ICN_VISION_TEST_EXPECTED").unwrap_or_else(|_| "7429".into());
        assert!(model_path.is_file(), "missing {}", model_path.display());
        assert!(
            projector_path.is_file(),
            "missing {}",
            projector_path.display()
        );
        let image = std::fs::read(&image_path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", image_path.display()));

        let native = NativeBackend::initialize().expect("initialize native backend");
        let hardware = native.discover_hardware(
            icn_hardware::CapacityPolicy::default(),
            "real-multimodal-cache-test",
            Vec::new(),
        );
        let mut defaults = model_plan_defaults();
        defaults.context_size = 8192;
        defaults.physical_context_size = 8192;
        defaults.batch_size = 512;
        defaults.ubatch_size = 512;
        defaults.max_sequences = 1;
        defaults.prefill_quantum = 128;
        let intent = execution_intent(model_path, Some(projector_path), &defaults);
        let prepared = native
            .prepare_load(
                "installed-vision-cache-test",
                intent,
                icn_contracts::SpeculativeDecodingConfig::default(),
                hardware,
            )
            .expect("prepare vision model load");
        let backend = prepared
            .execute(Arc::new(NoopLoadObserver))
            .expect("load vision model");

        let mut completion = request();
        completion.template.messages = vec![ChatMessage {
            role: ChatRole::User,
            content: Some(ChatContent::Parts(vec![
                ChatContentPart::Image(ImageInput::new("image/png", image)),
                ChatContentPart::Text {
                    text: "Transcribe the large verification code in this image. Reply with only the code."
                        .into(),
                },
            ])),
            reasoning: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        }];
        completion.max_tokens = 128;

        let first = backend
            .complete(completion.clone(), &mut |_| Ok(()))
            .expect("complete initial vision request");
        let perceived = format!("{} {}", first.reasoning, first.text);
        eprintln!(
            "cold vision response={perceived:?} prompt_tokens={} cached_tokens={} prompt_ms={:.2}",
            first.prompt_tokens, first.cached_prompt_tokens, first.metrics.prompt_ms
        );
        assert!(
            perceived.contains(&expected_text),
            "model did not perceive expected image text {expected_text:?}: {perceived:?}"
        );
        assert_eq!(first.cached_prompt_tokens, 0);

        let mut observed_cached_tokens = 0usize;
        let second = backend
            .complete(completion, &mut |event| {
                if let InferenceEvent::Progress(InferenceProgress::Prefill {
                    cached_tokens, ..
                }) = event.delta
                {
                    observed_cached_tokens = observed_cached_tokens.max(cached_tokens);
                }
                Ok(())
            })
            .expect("complete cached vision request");

        eprintln!(
            "cached vision response={:?} prompt_tokens={} cached_tokens={} prompt_ms={:.2}",
            format!("{} {}", second.reasoning, second.text),
            second.prompt_tokens,
            second.cached_prompt_tokens,
            second.metrics.prompt_ms
        );

        assert_eq!(
            second.cached_prompt_tokens,
            second.prompt_tokens.saturating_sub(1),
            "identical multimodal prompt did not reuse every token except the logits token"
        );
        assert_eq!(second.cached_prompt_tokens, observed_cached_tokens);
        assert!(
            second.metrics.prompt_ms < first.metrics.prompt_ms,
            "cached prefill ({:.2} ms) was not faster than cold prefill ({:.2} ms)",
            second.metrics.prompt_ms,
            first.metrics.prompt_ms
        );
    }

    #[test]
    #[ignore = "loads an installed vision model selected through ICN_VISION_TEST_* variables"]
    fn multimodal_multiturn_cache_and_speculation_with_real_model() {
        disable_native_diagnostics();
        let required_path = |name: &str| {
            let path = PathBuf::from(
                std::env::var_os(name).unwrap_or_else(|| panic!("{name} must name a local file")),
            );
            assert!(path.is_file(), "missing {}", path.display());
            path
        };
        let model_path = required_path("ICN_VISION_TEST_MODEL");
        let projector_path = required_path("ICN_VISION_TEST_PROJECTOR");
        let first_image = std::fs::read(required_path("ICN_VISION_TEST_IMAGE"))
            .expect("read first vision test image");
        let second_image = std::fs::read(required_path("ICN_VISION_TEST_IMAGE_2"))
            .expect("read second vision test image");
        let first_expected =
            std::env::var("ICN_VISION_TEST_EXPECTED").unwrap_or_else(|_| "7429".into());
        let second_expected =
            std::env::var("ICN_VISION_TEST_EXPECTED_2").unwrap_or_else(|_| "3816".into());

        let speculative = match std::env::var("ICN_VISION_TEST_SPECULATIVE_METHOD")
            .unwrap_or_else(|_| "none".into())
            .as_str()
        {
            "none" => icn_contracts::SpeculativeDecodingConfig::default(),
            method @ ("dflash" | "dspark") => {
                let draft = required_path("ICN_VISION_TEST_DRAFT");
                let threshold = std::env::var("ICN_VISION_TEST_SPECULATIVE_THRESHOLD")
                    .ok()
                    .and_then(|value| value.parse::<f32>().ok())
                    .unwrap_or(0.1);
                let method = if method == "dflash" {
                    icn_contracts::SpeculativeMethodConfig::DFlash {
                        min_sample_probability: threshold,
                    }
                } else {
                    icn_contracts::SpeculativeMethodConfig::DSpark {
                        acceptance_threshold: threshold,
                    }
                };
                icn_contracts::SpeculativeDecodingConfig::Enabled {
                    source: icn_contracts::SpeculativeDraftSource::Separate { model_path: draft },
                    method,
                    n_max: 3,
                    n_min: 0,
                    cache_type_k: CacheType::F16,
                    cache_type_v: CacheType::F16,
                }
            }
            method => panic!("unsupported ICN_VISION_TEST_SPECULATIVE_METHOD {method:?}"),
        };
        let speculative_enabled = matches!(
            speculative,
            icn_contracts::SpeculativeDecodingConfig::Enabled { .. }
        );

        let native = NativeBackend::initialize().expect("initialize native backend");
        let hardware = native.discover_hardware(
            icn_hardware::CapacityPolicy::default(),
            "real-multimodal-speculative-test",
            Vec::new(),
        );
        let mut defaults = model_plan_defaults();
        defaults.context_size = 32_768;
        defaults.physical_context_size = 32_768;
        defaults.batch_size = 512;
        defaults.ubatch_size = 512;
        defaults.max_sequences = 1;
        defaults.prefill_quantum = 128;
        let mut intent = execution_intent(model_path, Some(projector_path), &defaults);
        intent.speculative = speculative.clone();
        let prepared = native
            .prepare_load(
                "installed-vision-multiturn-test",
                intent,
                speculative,
                hardware,
            )
            .expect("prepare vision model load");
        let backend = prepared
            .execute(Arc::new(NoopLoadObserver))
            .expect("load vision model");

        fn image_message(bytes: Vec<u8>) -> ChatMessage {
            ChatMessage {
                role: ChatRole::User,
                content: Some(ChatContent::Parts(vec![
                    ChatContentPart::Image(ImageInput::new("image/png", bytes)),
                    ChatContentPart::Text {
                        text: "Transcribe the large verification code in this image. Reply with only the four-digit code."
                            .into(),
                    },
                ])),
                reasoning: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
            }
        }
        let run = |messages: Vec<ChatMessage>| {
            let mut completion = request();
            completion.template.messages = messages;
            completion.template.reasoning = ReasoningControl::Disabled;
            completion.max_tokens = 64;
            backend
                .complete(completion, &mut |_| Ok(()))
                .expect("complete multimodal turn")
        };

        let first_message = image_message(first_image);
        let first = run(vec![first_message.clone()]);
        let first_text = format!("{} {}", first.reasoning, first.text);
        assert!(
            first_text.contains(&first_expected),
            "first image was not perceived: {first_text:?}"
        );

        let second_message = image_message(second_image);
        let conversation = vec![
            first_message,
            ChatMessage::text(ChatRole::Assistant, first.text.clone()),
            second_message,
        ];
        let second = run(conversation.clone());
        let second_text = format!("{} {}", second.reasoning, second.text);
        assert!(
            second_text.contains(&second_expected),
            "second image was not perceived: {second_text:?}"
        );
        assert!(
            second.cached_prompt_tokens > 0,
            "the second turn did not reuse the first image prefix"
        );

        let repeated = run(conversation);
        let repeated_text = format!("{} {}", repeated.reasoning, repeated.text);
        assert!(
            repeated_text.contains(&second_expected),
            "cached second image was not perceived: {repeated_text:?}"
        );
        assert_eq!(
            repeated.cached_prompt_tokens,
            repeated.prompt_tokens.saturating_sub(1),
            "the repeated multimodal turn did not reuse both images"
        );
        assert!(
            repeated.metrics.prompt_ms < second.metrics.prompt_ms,
            "exact cached prefill ({:.2} ms) was not faster than the extended turn ({:.2} ms)",
            repeated.metrics.prompt_ms,
            second.metrics.prompt_ms
        );
        if speculative_enabled {
            let drafted = first.metrics.draft_tokens
                + second.metrics.draft_tokens
                + repeated.metrics.draft_tokens;
            let accepted = first.metrics.accepted_draft_tokens
                + second.metrics.accepted_draft_tokens
                + repeated.metrics.accepted_draft_tokens;
            assert!(
                drafted > 0,
                "the configured speculative method produced no drafts"
            );
            assert!(
                accepted > 0,
                "the configured speculative method accepted no draft tokens"
            );
        }
        eprintln!(
            "vision multiturn first_cached={} second_cached={} repeated_cached={} first_ms={:.2} second_ms={:.2} repeated_ms={:.2} drafted={} accepted={}",
            first.cached_prompt_tokens,
            second.cached_prompt_tokens,
            repeated.cached_prompt_tokens,
            first.metrics.prompt_ms,
            second.metrics.prompt_ms,
            repeated.metrics.prompt_ms,
            first.metrics.draft_tokens
                + second.metrics.draft_tokens
                + repeated.metrics.draft_tokens,
            first.metrics.accepted_draft_tokens
                + second.metrics.accepted_draft_tokens
                + repeated.metrics.accepted_draft_tokens,
        );
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
            physical_context_size: 128 * max_sequences,
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
            speculative: icn_contracts::SpeculativeDecodingConfig::default(),
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
    fn projector_mode_accepts_continuous_batching() {
        validate_model_config(&model_config_with_projector(4)).unwrap();
    }

    #[cfg(feature = "mtmd")]
    #[test]
    fn projector_mode_rejects_embedded_and_separate_mtp_before_loading() {
        for source in [
            icn_contracts::SpeculativeDraftSource::Embedded,
            icn_contracts::SpeculativeDraftSource::Separate {
                model_path: "draft.gguf".into(),
            },
        ] {
            let mut config = model_config_with_projector(1);
            config.speculative = icn_contracts::SpeculativeDecodingConfig::Enabled {
                source,
                method: icn_contracts::SpeculativeMethodConfig::Mtp {
                    min_draft_probability: 0.1,
                },
                n_max: 3,
                n_min: 0,
                cache_type_k: CacheType::F16,
                cache_type_v: CacheType::F16,
            };

            let error = validate_model_config(&config).unwrap_err();
            assert!(error.to_string().contains("does not support MTP"));
        }
    }

    #[cfg(feature = "mtmd")]
    #[test]
    fn projector_mode_accepts_embedding_capable_speculative_methods() {
        for method in [
            icn_contracts::SpeculativeMethodConfig::DFlash {
                min_sample_probability: 0.1,
            },
            icn_contracts::SpeculativeMethodConfig::DSpark {
                acceptance_threshold: 0.1,
            },
        ] {
            let mut config = model_config_with_projector(1);
            config.speculative = icn_contracts::SpeculativeDecodingConfig::Enabled {
                source: icn_contracts::SpeculativeDraftSource::Separate {
                    model_path: "draft.gguf".into(),
                },
                method,
                n_max: 3,
                n_min: 0,
                cache_type_k: CacheType::F16,
                cache_type_v: CacheType::F16,
            };
            validate_model_config(&config).unwrap();
        }
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
                physical_capacity_bytes: 64,
                physical_available_bytes: 20,
                allocation_capacity_bytes: 64,
                allocation_headroom_bytes: 20,
                assess_reserve_bytes: 0,
                abort_reserve_bytes: 0,
            },
            native_build: "test".to_owned(),
            enabled_backends: vec!["MTL".to_owned()],
            topology_fingerprint: "test".to_owned(),
            memory_domains: vec![HardwareMemoryDomain {
                id: icn_contracts::MemoryDomainId::system(),
                kind: HardwareMemoryDomainKind::UnifiedMemory,
                total_capacity_bytes: 64,
                stable_capacity_bytes: 60,
                current_free_bytes: Some(20),
                shares_system_memory: true,
                devices: vec![HardwareDevice {
                    id: icn_contracts::HardwareDeviceId::new("metal"),
                    native_index: 1,
                    backend: "MTL".to_owned(),
                    physical_id: Some("metal-0".to_owned()),
                    name: "MTL0".to_owned(),
                    description: "Apple".to_owned(),
                    kind: HardwareDeviceKind::Gpu,
                    memory_limit: None,
                }],
            }],
        };
        let evidence = vec![
            ResidentAllocation {
                location: LlamaMemoryLocation::Host,
                memory: MemoryBreakdown::new(3, 2, 1, 0),
            },
            ResidentAllocation {
                location: LlamaMemoryLocation::Device {
                    backend: "MTL".to_owned(),
                    physical_id: Some("metal-0".to_owned()),
                    native_index: 1,
                },
                memory: MemoryBreakdown::new(5, 4, 3, 2),
            },
        ];

        let config = model_config_with_projector(4);
        let resident = model_instance_allocation(&snapshot, &evidence, &config)
            .expect("exact device identities resolve");
        assert_eq!(resident.context_window_tokens, 128);
        assert_eq!(resident.parallel_sequences, 4);
        assert_eq!(resident.physical_context_tokens, 512);
        assert_eq!(resident.memory_domains.len(), 1);
        assert_eq!(
            resident.memory_domains[0].memory_domain_id,
            icn_contracts::MemoryDomainId::system()
        );
        assert_eq!(resident.memory_domains[0].model_bytes, 8);
        assert_eq!(resident.memory_domains[0].context_bytes, 6);
        assert_eq!(resident.memory_domains[0].compute_bytes, 4);
        assert_eq!(resident.memory_domains[0].auxiliary_bytes, 2);
    }
}
