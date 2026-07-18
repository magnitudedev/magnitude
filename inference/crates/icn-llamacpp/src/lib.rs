//! Persistent llama.cpp executor for ICN.

use std::collections::VecDeque;
use std::ffi::CString;
use std::num::{NonZeroI32, NonZeroU32};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{
    Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError, sync_channel,
};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use icn_core::output::{StopBuffer, Utf8Buffer};
use icn_core::{
    AllowedToolsMode, CacheType, ChatContent, ChatContentPart, ChatRequest, ChatTemplateRequest,
    CompletionBackend, ExecutionConfig, ExecutionConfigReport, FinishReason, FlashAttention,
    Generation, GenerationMetrics, GenerationSnapshot, GpuLayers, GrammarTrigger, ImageInput,
    InferenceError, InferenceEvent, InferenceStreamEvent, ModelConfig, ModelModalities,
    ModelProperties, PreparedChatInfo, ProjectorConfig, ReasoningControl, ResponseFormat,
    SplitMode, TemplateCapabilities, ToolCall, ToolChoice,
};
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
use llama_cpp_2::context::params::{FlashAttentionPolicy, KvCacheType, LlamaContextParams};
use llama_cpp_2::llama_backend::{LlamaBackend, LlamaThreadPool, LlamaThreadPoolParams};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::fit::FitStatus;
use llama_cpp_2::model::params::{LlamaGpuLayers, LlamaModelParams, LlamaSplitMode};
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;
use sha2::{Digest, Sha256};

mod scheduler;

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
use scheduler::{BatchPlanner, BatchWork, SequencePool, WorkCandidate, WorkKind};

const COMMAND_QUEUE_CAPACITY: usize = 32;
const EVENT_QUEUE_CAPACITY: usize = 16;
const OUTBOUND_QUEUE_CAPACITY: usize = 64;
const IDLE_POLL_INTERVAL: Duration = Duration::from_millis(1);

enum ExecutorCommand {
    Complete {
        request: ChatRequest,
        events: SyncSender<ExecutorItem>,
        cancelled: Arc<AtomicBool>,
        queued_at: Instant,
    },
    ApplyTemplate {
        request: ChatTemplateRequest,
        response: SyncSender<Result<PreparedChatInfo, InferenceError>>,
    },
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
    cancelled: Arc<AtomicBool>,
    outbound: VecDeque<ExecutorItem>,
    phase: RequestPhase,
    prompt: Vec<LlamaToken>,
    prompt_offset: usize,
    prompt_tokens: usize,
    next_position: i32,
    multimodal_prompt: Option<MultimodalPrompt>,
    generation_limit: usize,
    generated_tokens: usize,
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
    commands: SyncSender<ExecutorCommand>,
    executor: Mutex<Option<JoinHandle<()>>>,
}

impl LlamaCompletionBackend {
    /// Load a model and initialize its persistent context before returning.
    pub fn load(model_id: impl Into<String>, config: ModelConfig) -> Result<Self, InferenceError> {
        validate_model_config(&config)?;
        let model_id = model_id.into();
        let (commands, command_receiver) = sync_channel(COMMAND_QUEUE_CAPACITY);
        let (ready_sender, ready_receiver) = sync_channel(1);
        let executor = thread::Builder::new()
            .name(format!("icn-llama-{}", model_id))
            .spawn(move || executor_main(config, command_receiver, ready_sender))
            .map_err(backend_error)?;

        match ready_receiver.recv() {
            Ok(Ok(properties)) => Ok(Self {
                model_id,
                properties,
                commands,
                executor: Mutex::new(Some(executor)),
            }),
            Ok(Err(error)) => {
                let _ = executor.join();
                Err(error)
            }
            Err(_) => {
                let _ = executor.join();
                Err(InferenceError::ExecutorStopped)
            }
        }
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
            .try_send(ExecutorCommand::ApplyTemplate { request, response })
            .map_err(|error| match error {
                TrySendError::Full(_) => InferenceError::Overloaded,
                TrySendError::Disconnected(_) => InferenceError::ExecutorStopped,
            })?;
        receiver
            .recv()
            .map_err(|_| InferenceError::ExecutorStopped)?
    }

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

fn executor_main(
    config: ModelConfig,
    commands: Receiver<ExecutorCommand>,
    ready: SyncSender<Result<ModelProperties, InferenceError>>,
) {
    let backend = match LlamaBackend::init() {
        Ok(backend) => backend,
        Err(error) => {
            let _ = ready.send(Err(backend_error(error)));
            return;
        }
    };
    let (threads, threads_batch) = match resolved_thread_counts(&config.execution) {
        Ok(counts) => counts,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    let mut context_params = native_context_params(&config, threads, threads_batch);
    let model_params = match native_model_params(&config.execution) {
        Ok(params) => params,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    let mut model_params = std::pin::pin!(model_params);
    if config.execution.gpu_layers == GpuLayers::Auto {
        let model_path = match CString::new(config.model_path.to_string_lossy().as_bytes()) {
            Ok(path) => path,
            Err(error) => {
                let _ = ready.send(Err(backend_error(error)));
                return;
            }
        };
        let mut margins = vec![1024 * 1024 * 1024; llama_cpp_2::max_devices()];
        let report = match model_params.as_mut().fit_params_report(
            &model_path,
            &mut context_params,
            &mut margins,
            4_096,
        ) {
            Ok(report) => report,
            Err(error) => {
                let _ = ready.send(Err(backend_error(error)));
                return;
            }
        };
        if report.status != FitStatus::Success {
            let _ = ready.send(Err(InferenceError::Backend(format!(
                "llama.cpp automatic placement failed with {:?}: {:?}",
                report.status, report.warnings
            ))));
            return;
        }
    }
    let resolved_execution = resolved_execution_config(
        &config.execution,
        model_params.as_ref().get_ref(),
        threads,
        threads_batch,
    );
    let model = match LlamaModel::load_from_file(
        &backend,
        &config.model_path,
        model_params.as_ref().get_ref(),
    ) {
        Ok(model) => model,
        Err(error) => {
            let _ = ready.send(Err(backend_error(error)));
            return;
        }
    };
    let chat_templates = match CommonChatTemplates::from_model(&model) {
        Ok(templates) => templates,
        Err(error) => {
            let _ = ready.send(Err(backend_error(error)));
            return;
        }
    };
    let mut context = match model.new_context(&backend, context_params) {
        Ok(context) => context,
        Err(error) => {
            let _ = ready.send(Err(backend_error(error)));
            return;
        }
    };
    let mut multimodal = {
        #[cfg(feature = "mtmd")]
        {
            match config.projector.as_ref() {
                Some(projector) => match MultimodalRuntime::load(
                    projector,
                    &model,
                    config.execution.flash_attention,
                    Some(threads.get()),
                ) {
                    Ok(runtime) => Some(runtime),
                    Err(error) => {
                        let _ = ready.send(Err(error));
                        return;
                    }
                },
                None => None,
            }
        }
        #[cfg(not(feature = "mtmd"))]
        {
            None::<MultimodalRuntime<'_>>
        }
    };
    let mut main_pool = match LlamaThreadPool::new(&backend, &LlamaThreadPoolParams::new(threads)) {
        Ok(pool) => pool,
        Err(error) => {
            let _ = ready.send(Err(backend_error(error)));
            return;
        }
    };
    if threads == threads_batch {
        let mut attached = context.attach_threadpool(&mut main_pool);
        run_initialized_executor(
            &config,
            resolved_execution,
            &model,
            &chat_templates,
            &mut attached,
            &mut multimodal,
            &commands,
            &ready,
        );
    } else {
        let mut batch_pool =
            match LlamaThreadPool::new(&backend, &LlamaThreadPoolParams::new(threads_batch)) {
                Ok(pool) => pool,
                Err(error) => {
                    let _ = ready.send(Err(backend_error(error)));
                    return;
                }
            };
        let mut attached = context.attach_threadpools(&mut main_pool, &mut batch_pool);
        run_initialized_executor(
            &config,
            resolved_execution,
            &model,
            &chat_templates,
            &mut attached,
            &mut multimodal,
            &commands,
            &ready,
        );
    }
}

fn resolved_thread_counts(
    execution: &ExecutionConfig,
) -> Result<(NonZeroI32, NonZeroI32), InferenceError> {
    let main = match execution.threads {
        Some(threads) => nonzero_i32(threads, "threads")?,
        None => available_math_threads(),
    };
    let batch = match execution.threads_batch {
        Some(threads) => nonzero_i32(threads, "threads_batch")?,
        None => main,
    };
    Ok((main, batch))
}

fn available_math_threads() -> NonZeroI32 {
    let logical = std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get);
    let selected =
        platform_physical_core_count().unwrap_or_else(|| fallback_math_thread_count(logical));
    let selected = selected.clamp(1, i32::MAX as usize) as i32;
    NonZeroI32::new(selected).expect("the resolved math-thread count is always positive")
}

/// Mirrors llama.cpp common's portable fallback while respecting the logical CPUs Rust reports as
/// available to this process. A failed logical-CPU query uses llama.cpp's four-thread fallback.
fn fallback_math_thread_count(logical: usize) -> usize {
    match logical {
        0 => 4,
        1..=4 => logical,
        _ => logical / 2,
    }
}

#[cfg(target_os = "macos")]
fn platform_physical_core_count() -> Option<usize> {
    macos_physical_core_count_with(macos_positive_i32_sysctl)
}

#[cfg(any(target_os = "macos", test))]
fn macos_physical_core_count_with(
    mut read_sysctl: impl FnMut(&std::ffi::CStr) -> Option<usize>,
) -> Option<usize> {
    read_sysctl(c"hw.perflevel0.physicalcpu").or_else(|| read_sysctl(c"hw.physicalcpu"))
}

#[cfg(target_os = "macos")]
fn macos_positive_i32_sysctl(name: &std::ffi::CStr) -> Option<usize> {
    let mut value: i32 = 0;
    let mut length = std::mem::size_of_val(&value);
    // SAFETY: `name` is NUL terminated; `value` and `length` point to writable objects of the
    // advertised size; and both the replacement pointer and replacement length are null/zero for
    // this read-only query.
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&raw mut value).cast(),
            &raw mut length,
            std::ptr::null_mut(),
            0,
        )
    };
    (status == 0 && value > 0).then_some(value as usize)
}

#[cfg(target_os = "linux")]
fn platform_physical_core_count() -> Option<usize> {
    linux_physical_core_count_with(|cpu| {
        std::fs::read_to_string(format!(
            "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings"
        ))
        .ok()
    })
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn platform_physical_core_count() -> Option<usize> {
    None
}

/// Count the unique Linux thread-sibling masks, stopping at the first unavailable CPU exactly as
/// pinned llama.cpp common does. On hybrid x86 Linux, upstream may instead temporarily change the
/// caller's CPU affinity and probe each core with CPUID. ICN deliberately does not reproduce that
/// potentially disruptive native behavior, so this safe path returns the ordinary physical-core
/// count.
#[cfg(any(target_os = "linux", test))]
fn linux_physical_core_count_with(
    mut read_siblings: impl FnMut(u32) -> Option<String>,
) -> Option<usize> {
    let mut sibling_sets = std::collections::HashSet::new();
    for cpu in 0..u32::MAX {
        let Some(contents) = read_siblings(cpu) else {
            break;
        };
        if let Some(first_line) = contents.split_terminator('\n').next() {
            sibling_sets.insert(first_line.to_owned());
        }
    }
    (!sibling_sets.is_empty()).then_some(sibling_sets.len())
}

fn nonzero_i32(value: NonZeroU32, field: &str) -> Result<NonZeroI32, InferenceError> {
    let value = i32::try_from(value.get())
        .map_err(|_| InferenceError::InvalidConfig(format!("{field} must not exceed i32::MAX")))?;
    Ok(NonZeroI32::new(value).expect("a converted NonZeroU32 remains non-zero"))
}

fn native_model_params(execution: &ExecutionConfig) -> Result<LlamaModelParams, InferenceError> {
    let params = LlamaModelParams::default()
        .with_gpu_layers(match execution.gpu_layers {
            GpuLayers::Auto => LlamaGpuLayers::Auto,
            GpuLayers::All => LlamaGpuLayers::All,
            GpuLayers::Count(value) => LlamaGpuLayers::Count(value),
        })
        .with_use_mmap(execution.use_mmap)
        .with_use_mlock(execution.use_mlock)
        .with_split_mode(match execution.split_mode {
            SplitMode::None => LlamaSplitMode::None,
            SplitMode::Layer => LlamaSplitMode::Layer,
            SplitMode::Row => LlamaSplitMode::Row,
            SplitMode::Tensor => LlamaSplitMode::Tensor,
        });
    match &execution.tensor_split {
        Some(weights) => params.with_tensor_split(weights).map_err(backend_error),
        None => Ok(params),
    }
}

fn native_context_params(
    config: &ModelConfig,
    threads: NonZeroI32,
    threads_batch: NonZeroI32,
) -> LlamaContextParams {
    let execution = &config.execution;
    LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(config.context_size))
        .with_n_batch(config.batch_size)
        .with_n_ubatch(config.ubatch_size)
        .with_n_seq_max(config.max_sequences)
        .with_n_threads(threads.get())
        .with_n_threads_batch(threads_batch.get())
        .with_type_k(native_cache_type(execution.cache_type_k))
        .with_type_v(native_cache_type(execution.cache_type_v))
        .with_offload_kqv(execution.offload_kqv)
        .with_op_offload(execution.operation_offload)
        .with_swa_full(execution.swa_full)
        .with_kv_unified(execution.kv_unified)
        .with_flash_attention(match execution.flash_attention {
            FlashAttention::Auto => FlashAttentionPolicy::Auto,
            FlashAttention::Disabled => FlashAttentionPolicy::Disabled,
            FlashAttention::Enabled => FlashAttentionPolicy::Enabled,
        })
}

fn native_cache_type(cache_type: CacheType) -> KvCacheType {
    match cache_type {
        CacheType::F32 => KvCacheType::F32,
        CacheType::F16 => KvCacheType::F16,
        CacheType::Bf16 => KvCacheType::BF16,
        CacheType::Q8_0 => KvCacheType::Q8_0,
        CacheType::Q4_0 => KvCacheType::Q4_0,
        CacheType::Q4_1 => KvCacheType::Q4_1,
        CacheType::Iq4Nl => KvCacheType::IQ4_NL,
        CacheType::Q5_0 => KvCacheType::Q5_0,
        CacheType::Q5_1 => KvCacheType::Q5_1,
    }
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
    config: &ModelConfig,
    resolved_execution: ExecutionConfig,
    model: &'model LlamaModel,
    chat_templates: &CommonChatTemplates,
    context: &mut LlamaContext<'model>,
    multimodal: &mut Option<MultimodalRuntime<'model>>,
    commands: &Receiver<ExecutorCommand>,
    ready: &SyncSender<Result<ModelProperties, InferenceError>>,
) {
    if let Err(error) = warm_up(model, context) {
        let _ = ready.send(Err(error));
        return;
    }
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
            let _ = ready.send(Err(error));
            return;
        }
    };
    if ready.send(Ok(properties)).is_err() {
        return;
    }
    run_scheduler(config, model, chat_templates, context, multimodal, commands);
}

fn run_scheduler<'model>(
    config: &ModelConfig,
    model: &'model LlamaModel,
    chat_templates: &CommonChatTemplates,
    context: &mut LlamaContext<'model>,
    multimodal: &mut Option<MultimodalRuntime<'model>>,
    commands: &Receiver<ExecutorCommand>,
) {
    let mut sequence_pool = SequencePool::new(config.max_sequences);
    let mut planner = BatchPlanner::new(config.prefill_quantum as usize);
    let mut queued = VecDeque::<QueuedCompletion>::new();
    let mut active = Vec::<ActiveRequest<'_>>::new();
    let mut shutting_down = false;
    let max_tracked = COMMAND_QUEUE_CAPACITY + config.max_sequences as usize;

    loop {
        drain_commands(
            commands,
            chat_templates,
            multimodal.as_ref().map(multimodal_marker),
            &mut queued,
            &active,
            max_tracked,
            &mut shutting_down,
        );

        cleanup_requests(context, &mut sequence_pool, &mut active);

        if shutting_down {
            fail_queued(&mut queued, InferenceError::ExecutorStopped);
            fail_active(
                context,
                &mut sequence_pool,
                &mut active,
                InferenceError::ExecutorStopped,
            );
            cleanup_requests(context, &mut sequence_pool, &mut active);
            if active.is_empty() {
                break;
            }
        } else {
            admit_requests(
                model,
                chat_templates,
                multimodal.as_ref(),
                context,
                &mut sequence_pool,
                &mut queued,
                &mut active,
            );
        }

        sample_ready_requests(model, context, &mut sequence_pool, &mut active);
        cleanup_requests(context, &mut sequence_pool, &mut active);

        let decoded = if shutting_down {
            false
        } else {
            match decode_batch(context, multimodal, &mut planner, &mut active) {
                Ok(decoded) => decoded,
                Err(error) => {
                    // A failed decode can leave shared native memory in an uncertain state. Fail
                    // every resident request and reset the whole context before admitting more
                    // work rather than guessing which sequence committed.
                    context.synchronize();
                    context.clear_memory(false);
                    let failure = if matches!(error, InferenceError::Cancelled) {
                        InferenceError::Cancelled
                    } else {
                        InferenceError::Backend(error.to_string())
                    };
                    fail_active(context, &mut sequence_pool, &mut active, failure);
                    false
                }
            }
        };

        cleanup_requests(context, &mut sequence_pool, &mut active);

        if !decoded {
            match commands.recv_timeout(IDLE_POLL_INTERVAL) {
                Ok(command) => handle_command(
                    command,
                    chat_templates,
                    multimodal.as_ref().map(multimodal_marker),
                    &mut queued,
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

fn drain_commands(
    commands: &Receiver<ExecutorCommand>,
    chat_templates: &CommonChatTemplates,
    media_marker: Option<&str>,
    queued: &mut VecDeque<QueuedCompletion>,
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

fn handle_command(
    command: ExecutorCommand,
    chat_templates: &CommonChatTemplates,
    media_marker: Option<&str>,
    queued: &mut VecDeque<QueuedCompletion>,
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
        } => {
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
                });
            }
        }
        ExecutorCommand::ApplyTemplate { request, response } => {
            let result = if *shutting_down {
                Err(InferenceError::ExecutorStopped)
            } else {
                prepare_chat(chat_templates, &request, media_marker)
                    .and_then(|prepared| prepared_chat_info(chat_templates, &prepared))
            };
            let _ = response.send(result);
        }
        ExecutorCommand::Shutdown => *shutting_down = true,
    }
}

fn admit_requests<'model>(
    model: &'model LlamaModel,
    chat_templates: &CommonChatTemplates,
    multimodal: Option<&MultimodalRuntime<'model>>,
    context: &mut LlamaContext<'model>,
    sequence_pool: &mut SequencePool,
    queued: &mut VecDeque<QueuedCompletion>,
    active: &mut Vec<ActiveRequest<'model>>,
) {
    while let Some(sequence_id) = sequence_pool.acquire() {
        let Some(queued_request) = queued.pop_front() else {
            sequence_pool.release(sequence_id);
            break;
        };
        if queued_request.cancelled.load(Ordering::Acquire) {
            let _ = queued_request
                .events
                .try_send(ExecutorItem::Failed(InferenceError::Cancelled));
            sequence_pool.release(sequence_id);
            continue;
        }
        if let Err(error) = clear_sequence(context, sequence_id) {
            let _ = queued_request.events.try_send(ExecutorItem::Failed(error));
            sequence_pool.quarantine(sequence_id);
            continue;
        }
        match ActiveRequest::admit(
            model,
            chat_templates,
            multimodal,
            context.n_ctx_seq() as usize,
            sequence_id,
            queued_request,
        ) {
            Ok(request) => active.push(request),
            Err((events, error)) => {
                let _ = events.try_send(ExecutorItem::Failed(error));
                if clear_sequence(context, sequence_id).is_ok() {
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
    sequence_pool: &mut SequencePool,
    active: &mut [ActiveRequest<'model>],
) {
    for request in active {
        let RequestPhase::ReadyToSample { batch_index } = request.phase else {
            continue;
        };
        if request.cancelled.load(Ordering::Acquire) {
            cancel_request(request);
            release_sequence(context, sequence_pool, request);
            continue;
        }
        match request.sample_next(model, context, batch_index) {
            Ok(Some(reason)) => {
                if let Err(error) = request.complete(reason) {
                    fail_request(request, error);
                }
                release_sequence(context, sequence_pool, request);
            }
            Ok(None) => {}
            Err(error) => {
                fail_request(request, error);
                release_sequence(context, sequence_pool, request);
            }
        }
    }
}

fn decode_batch<'model>(
    context: &mut LlamaContext<'model>,
    multimodal: &mut Option<MultimodalRuntime<'model>>,
    planner: &mut BatchPlanner,
    active: &mut [ActiveRequest<'model>],
) -> Result<bool, InferenceError> {
    if decode_multimodal_prefill(context, multimodal, active)? {
        return Ok(true);
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
                    remaining: request.prompt.len().saturating_sub(request.prompt_offset),
                },
                RequestPhase::Decode { .. } => WorkKind::Decode,
                RequestPhase::ReadyToSample { .. } | RequestPhase::Terminal => return None,
            };
            Some(WorkCandidate { sequence_id, kind })
        })
        .collect::<Vec<_>>();
    let plan = planner.plan(&candidates, context.n_batch() as usize);
    if plan.is_empty() {
        return Ok(false);
    }

    let mut batch = LlamaBatch::new(context.n_batch() as usize, 1);
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
                batch
                    .add(token, position, &[sequence_id], true)
                    .map_err(backend_error)?;
                logits.push((sequence_id, batch.n_tokens() - 1));
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

    context.decode(&mut batch).map_err(backend_error)?;
    for (sequence_id, batch_index) in logits {
        let request = request_by_sequence(active, sequence_id)?;
        request.phase = RequestPhase::ReadyToSample { batch_index };
    }
    Ok(true)
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

fn cleanup_requests(
    context: &mut LlamaContext<'_>,
    sequence_pool: &mut SequencePool,
    active: &mut Vec<ActiveRequest<'_>>,
) {
    let mut index = 0;
    while index < active.len() {
        if active[index].cancelled.load(Ordering::Acquire)
            && !matches!(active[index].phase, RequestPhase::Terminal)
        {
            cancel_request(&mut active[index]);
            release_sequence(context, sequence_pool, &mut active[index]);
        }
        match flush_outbound(&mut active[index]) {
            FlushOutcome::Empty if matches!(active[index].phase, RequestPhase::Terminal) => {
                active.remove(index);
            }
            FlushOutcome::Disconnected => {
                release_sequence(context, sequence_pool, &mut active[index]);
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
    sequence_pool: &mut SequencePool,
    request: &mut ActiveRequest<'_>,
) {
    let Some(sequence_id) = request.sequence_id.take() else {
        return;
    };
    // Full sequence removal is supported for every llama.cpp memory implementation. This is the
    // sole cache policy required by this milestone: a sequence is never reassigned while resident
    // state still belongs to the previous request.
    match clear_sequence(context, sequence_id) {
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

fn clear_sequence(context: &mut LlamaContext<'_>, sequence_id: i32) -> Result<(), InferenceError> {
    let sequence = u32::try_from(sequence_id).map_err(backend_error)?;
    let removed = context
        .clear_kv_cache_seq(Some(sequence), None, None)
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
    sequence_pool: &mut SequencePool,
    active: &mut [ActiveRequest<'_>],
    reason: InferenceError,
) {
    for request in active {
        if !matches!(request.phase, RequestPhase::Terminal) {
            fail_request(request, clone_inference_error(&reason));
        }
        release_sequence(context, sequence_pool, request);
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

fn validate_model_config(config: &ModelConfig) -> Result<(), InferenceError> {
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
    if config.execution.gpu_layers == GpuLayers::Auto && config.execution.tensor_split.is_some() {
        return Err(InferenceError::InvalidConfig(
            "gpu_layers=auto cannot be combined with tensor_split because common/fit owns placement"
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
    _config: &ModelConfig,
    _projector: &ProjectorConfig,
) -> Result<(), InferenceError> {
    Err(InferenceError::InvalidConfig(
        "a multimodal projector was configured, but this ICN binary was built without the mtmd feature"
            .into(),
    ))
}

#[cfg(feature = "mtmd")]
fn validate_projector_config(
    config: &ModelConfig,
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

fn warm_up(model: &LlamaModel, context: &mut LlamaContext<'_>) -> Result<(), InferenceError> {
    let tokens = model
        .str_to_token(" ", AddBos::Always)
        .map_err(backend_error)?;
    if let Some(token) = tokens.first().copied() {
        let mut batch = LlamaBatch::new(1, 1);
        batch.add(token, 0, &[0], false).map_err(backend_error)?;
        context.decode(&mut batch).map_err(backend_error)?;
        context.synchronize();
        context.clear_kv_cache();
    }
    context.reset_timings();
    Ok(())
}

fn model_properties(
    config: &ModelConfig,
    resolved_execution: ExecutionConfig,
    model: &LlamaModel,
    context: &LlamaContext<'_>,
    templates: &CommonChatTemplates,
    modalities: ModelModalities,
) -> Result<ModelProperties, InferenceError> {
    let chat_template = templates.source(None).map_err(backend_error)?;
    let capabilities = templates.capabilities().map_err(backend_error)?;
    Ok(ModelProperties {
        model_path: config.model_path.clone(),
        model_size_bytes: model.size(),
        architecture: model.meta_val_str("general.architecture").ok(),
        name: model.meta_val_str("general.name").ok(),
        context_tokens: context.n_ctx_seq(),
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
        modalities,
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
    format!("{:x}", Sha256::digest(value.as_bytes()))
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

impl<'model> ActiveRequest<'model> {
    fn admit(
        model: &'model LlamaModel,
        chat_templates: &CommonChatTemplates,
        multimodal: Option<&MultimodalRuntime<'model>>,
        context_capacity: usize,
        sequence_id: i32,
        queued: QueuedCompletion,
    ) -> Result<Self, (SyncSender<ExecutorItem>, InferenceError)> {
        let QueuedCompletion {
            request,
            events,
            cancelled,
            queued_at,
        } = queued;
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
            if prompt_tokens >= context_capacity {
                return Err(InferenceError::InvalidConfig(format!(
                    "prompt ({prompt_tokens} tokens) leaves no generation capacity in the effective per-sequence context ({context_capacity})"
                )));
            }

            let mut sampler = make_sampler(model, &request, &prepared)?;
            sampler
                .accept_prompt(tokenized.text_tokens.iter())
                .map_err(backend_error)?;
            let mut stops = request.stop.clone();
            stops.extend(prepared.additional_stops().iter().cloned());

            Ok(Self {
                sequence_id: Some(sequence_id),
                events: events.clone(),
                cancelled,
                outbound: VecDeque::new(),
                phase: RequestPhase::Prefill,
                prompt: tokenized.text_tokens,
                prompt_offset: 0,
                prompt_tokens,
                next_position: tokenized.next_position,
                multimodal_prompt: tokenized.multimodal,
                generation_limit: (request.max_tokens as usize)
                    .min(context_capacity.saturating_sub(prompt_tokens)),
                generated_tokens: 0,
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
        let sampled_at = Instant::now();
        let is_eog = model.is_eog_token(token);
        account_sample(&mut self.generated_tokens);
        self.record_sample(sampled_at);
        let starts_stream = self.generated_tokens == 1;
        if is_eog {
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

        let position = self.next_position;
        self.next_position = self.next_position.checked_add(1).ok_or_else(|| {
            InferenceError::Backend("generation position exceeded i32::MAX".into())
        })?;
        self.phase = RequestPhase::Decode { token, position };
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
            cached_prompt_tokens: 0,
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

    let template_kwargs = request
        .template_args
        .iter()
        .map(|(key, value)| {
            Ok(ChatTemplateKwarg {
                key: key.clone(),
                value_json: serde_json::to_string(value).map_err(backend_error)?,
            })
        })
        .collect::<Result<Vec<_>, InferenceError>>()?;
    let enable_thinking = !matches!(request.reasoning, ReasoningControl::Disabled);

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
) -> Result<(Vec<&icn_core::ToolDefinition>, ChatToolChoice), InferenceError> {
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
    let reasoning_budget = match request.template.reasoning {
        ReasoningControl::Enabled {
            budget_tokens: Some(tokens),
        } => {
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
        _ => None,
    };

    CommonSampler::new(
        model,
        &CommonSamplerConfig {
            seed: Some(request.seed),
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

    use icn_core::{ChatMessage, ChatRole, ReasoningControl, ResponseFormat, ToolDefinition};

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

    fn model_config_with_projector(max_sequences: u32) -> ModelConfig {
        let executable = std::env::current_exe().unwrap();
        ModelConfig {
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
        }
    }

    fn model_config() -> ModelConfig {
        let mut config = model_config_with_projector(1);
        config.projector = None;
        config
    }

    #[test]
    fn portable_math_thread_fallback_matches_llama_common_policy() {
        assert_eq!(fallback_math_thread_count(0), 4);
        assert_eq!(fallback_math_thread_count(1), 1);
        assert_eq!(fallback_math_thread_count(2), 2);
        assert_eq!(fallback_math_thread_count(4), 4);
        assert_eq!(fallback_math_thread_count(5), 2);
        assert_eq!(fallback_math_thread_count(8), 4);
        assert_eq!(fallback_math_thread_count(17), 8);
    }

    #[test]
    fn macos_physical_core_count_prefers_performance_cores() {
        let mut requested = Vec::new();
        let count = macos_physical_core_count_with(|name| {
            requested.push(name.to_owned());
            (name == c"hw.perflevel0.physicalcpu").then_some(12)
        });
        assert_eq!(count, Some(12));
        assert_eq!(requested, [c"hw.perflevel0.physicalcpu".to_owned()]);
    }

    #[test]
    fn macos_physical_core_count_falls_back_to_all_physical_cores() {
        let mut requested = Vec::new();
        let count = macos_physical_core_count_with(|name| {
            requested.push(name.to_owned());
            (name == c"hw.physicalcpu").then_some(8)
        });
        assert_eq!(count, Some(8));
        assert_eq!(
            requested,
            [
                c"hw.perflevel0.physicalcpu".to_owned(),
                c"hw.physicalcpu".to_owned()
            ]
        );
    }

    #[test]
    fn linux_physical_core_count_deduplicates_thread_sibling_masks() {
        let files = [
            "00000000,00000003\n",
            "00000000,00000003\n",
            "00000000,0000000c\n",
            "00000000,0000000c\n",
        ];
        let count = linux_physical_core_count_with(|cpu| {
            files.get(cpu as usize).map(|contents| (*contents).into())
        });
        assert_eq!(count, Some(2));
    }

    #[test]
    fn linux_physical_core_count_stops_at_the_first_missing_cpu() {
        let count = linux_physical_core_count_with(|cpu| match cpu {
            0 => Some("00000003\n".into()),
            2 => Some("0000000c\n".into()),
            _ => None,
        });
        assert_eq!(count, Some(1));
    }

    #[test]
    fn linux_physical_core_count_rejects_an_empty_topology() {
        assert_eq!(linux_physical_core_count_with(|_| None), None);
        assert_eq!(
            linux_physical_core_count_with(|cpu| (cpu == 0).then(String::new)),
            None
        );
    }

    #[test]
    fn native_execution_params_preserve_server_defaults_and_sentinels() {
        let mut config = model_config();
        config.execution = ExecutionConfig::default();
        let threads = NonZeroI32::new(3).unwrap();
        let threads_batch = NonZeroI32::new(5).unwrap();
        let context = native_context_params(&config, threads, threads_batch);
        assert_eq!(context.type_k(), KvCacheType::F16);
        assert_eq!(context.type_v(), KvCacheType::F16);
        assert!(context.offload_kqv());
        assert!(context.op_offload());
        assert!(!context.swa_full());
        assert!(!context.kv_unified());
        assert_eq!(context.n_threads(), 3);
        assert_eq!(context.n_threads_batch(), 5);

        config.execution.gpu_layers = GpuLayers::All;
        let all = native_model_params(&config.execution).unwrap();
        assert_eq!(all.gpu_layers(), LlamaGpuLayers::All);
        config.execution.gpu_layers = GpuLayers::Count(7);
        let exact = native_model_params(&config.execution).unwrap();
        assert_eq!(exact.gpu_layers(), LlamaGpuLayers::Count(7));
    }

    #[test]
    fn execution_validation_rejects_ambiguous_or_native_invalid_combinations() {
        let mut config = model_config();
        config.execution.gpu_layers = GpuLayers::Auto;
        config.execution.tensor_split = Some(vec![1.0]);
        assert!(
            validate_model_config(&config)
                .unwrap_err()
                .to_string()
                .contains("common/fit owns placement")
        );

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
}
