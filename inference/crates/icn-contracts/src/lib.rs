use std::collections::BTreeMap;
use std::fmt;
use std::num::{NonZeroU32, NonZeroUsize};
use std::path::PathBuf;
use std::sync::Arc;

pub mod inventory;
pub mod output;

pub use inventory::*;

/// Fully resolved model, component, context, and execution configuration.
///
/// Hardware assessment and the engine must consume this same value. Neither
/// layer is permitted to apply independent execution defaults.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ResolvedExecutionPlan {
    pub model_path: PathBuf,
    /// Total context capacity shared by all concurrently resident sequences.
    pub context_size: u32,
    pub batch_size: u32,
    pub ubatch_size: u32,
    /// Maximum number of independently owned llama.cpp sequence IDs.
    pub max_sequences: u32,
    /// Maximum prompt tokens allocated to one sequence in a round-robin pass.
    pub prefill_quantum: u32,
    /// Model loading, KV allocation, offload, and native worker-pool policy.
    pub execution: ExecutionConfig,
    /// Optional multimodal projector loaded into the same model executor.
    pub projector: Option<ProjectorConfig>,
    /// Fully resolved speculative-decoding configuration.
    pub mtp: MtpConfig,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MtpConfig {
    Disabled {
        reason: String,
    },
    Enabled {
        source: MtpSource,
        n_max: u32,
        n_min: u32,
        p_min: f32,
        cache_type_k: CacheType,
        cache_type_v: CacheType,
    },
}

impl Default for MtpConfig {
    fn default() -> Self {
        Self::Disabled {
            reason: "not_supported".to_owned(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MtpSource {
    /// Prediction layers are executable from the target GGUF itself.
    Bundled,
    /// Prediction layers are executable from a distinct GGUF linked to the target context.
    Separate { model_path: PathBuf },
}

#[derive(Debug, Clone, PartialEq)]
pub enum MtpRuntimeProperties {
    Disabled {
        reason: String,
    },
    Enabled {
        source: MtpSource,
        n_max: u32,
        n_min: u32,
        p_min: f32,
    },
}

/// llama.cpp execution settings whose values affect loading or context allocation.
///
/// Defaults intentionally match the pinned `llama-server` configuration rather than bare
/// `llama_context_default_params` where those differ (notably `swa_full`).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ExecutionConfig {
    pub gpu_layers: GpuLayers,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub split_mode: SplitMode,
    /// Explicit per-device proportions. `None` lets llama.cpp select placement.
    pub tensor_split: Option<Vec<f32>>,
    pub cache_type_k: CacheType,
    pub cache_type_v: CacheType,
    pub offload_kqv: bool,
    pub operation_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    /// `None` selects ICN's safe mirror of pinned llama.cpp common's math-thread default.
    ///
    /// On macOS this prefers the performance-level physical-core count and then the total physical
    /// core count. On ordinary Linux it counts unique `thread_siblings` masks. Other platforms (or
    /// unavailable topology data) use all available logical CPUs up to four, otherwise half. The
    /// pinned x86 Linux hybrid-CPU special case temporarily changes the calling thread's affinity
    /// and uses CPUID to exclude efficiency cores. ICN cannot reproduce that exactly without this
    /// potentially disruptive native behavior, so set this field explicitly when exact
    /// hybrid-host parity is required.
    pub threads: Option<NonZeroU32>,
    /// `None` reuses the generation pool for prompt processing.
    pub threads_batch: Option<NonZeroU32>,
    pub flash_attention: FlashAttention,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
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
        }
    }
}

/// A requested execution configuration and the concrete native parameters selected at startup.
///
/// `resolved` describes parameters passed to libllama, not measured physical tensor placement.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionConfigReport {
    pub requested: ExecutionConfig,
    pub resolved: ExecutionConfig,
}

/// llama.cpp GPU-layer selection semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuLayers {
    /// Run pinned `common/fit` before loading and use its selected layer count and placement.
    Auto,
    /// Ask libllama to offload every supported layer.
    All,
    /// Offload at most the explicit number of layers.
    Count(u32),
}

impl std::str::FromStr for GpuLayers {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        if value.eq_ignore_ascii_case("all") {
            return Ok(Self::All);
        }
        value
            .parse::<u32>()
            .map(Self::Count)
            .map_err(|_| "GPU layers must be 'auto', 'all', or a non-negative integer".to_owned())
    }
}

/// Model distribution strategy accepted by the pinned llama.cpp server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitMode {
    None,
    Layer,
    Row,
    Tensor,
}

impl std::str::FromStr for SplitMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "none" => Ok(Self::None),
            "layer" => Ok(Self::Layer),
            "row" => Ok(Self::Row),
            "tensor" => Ok(Self::Tensor),
            _ => Err("split mode must be one of: none, layer, row, tensor".to_owned()),
        }
    }
}

/// KV-cache data types accepted by the pinned llama.cpp server CLI.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheType {
    F32,
    F16,
    Bf16,
    Q8_0,
    Q4_0,
    Q4_1,
    Iq4Nl,
    Q5_0,
    Q5_1,
}

impl std::str::FromStr for CacheType {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "bf16" => Ok(Self::Bf16),
            "q8_0" => Ok(Self::Q8_0),
            "q4_0" => Ok(Self::Q4_0),
            "q4_1" => Ok(Self::Q4_1),
            "iq4_nl" => Ok(Self::Iq4Nl),
            "q5_0" => Ok(Self::Q5_0),
            "q5_1" => Ok(Self::Q5_1),
            _ => Err(format!(
                "unsupported cache type {value:?}; expected f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, or q5_1"
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ProjectorConfig {
    pub path: PathBuf,
    pub use_gpu: bool,
    pub warmup: bool,
    pub image_min_tokens: Option<NonZeroU32>,
    pub image_max_tokens: Option<NonZeroU32>,
    pub input_limits: ImageInputLimits,
}

impl ProjectorConfig {
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            use_gpu: true,
            warmup: true,
            image_min_tokens: None,
            image_max_tokens: None,
            input_limits: ImageInputLimits::default(),
        }
    }
}

/// Resource limits applied before native media preprocessing allocates decoded pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct ImageInputLimits {
    pub max_images: NonZeroU32,
    /// Maximum compressed bytes accepted for one image after data-URL decoding.
    pub max_input_bytes_per_image: NonZeroUsize,
    /// Maximum RGB bytes a single decoded image may occupy.
    pub max_decoded_bytes_per_image: NonZeroUsize,
    /// Maximum aggregate RGB bytes across all images in one request.
    pub max_total_decoded_bytes: NonZeroUsize,
}

impl Default for ImageInputLimits {
    fn default() -> Self {
        Self {
            max_images: NonZeroU32::new(4).expect("non-zero constant"),
            max_input_bytes_per_image: NonZeroUsize::new(8 * 1024 * 1024)
                .expect("non-zero constant"),
            max_decoded_bytes_per_image: NonZeroUsize::new(64 * 1024 * 1024)
                .expect("non-zero constant"),
            max_total_decoded_bytes: NonZeroUsize::new(128 * 1024 * 1024)
                .expect("non-zero constant"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FlashAttention {
    Auto,
    Disabled,
    Enabled,
}

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_new_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
}

#[derive(Debug, Clone, Default)]
pub struct GenerationMetrics {
    pub queue_ms: f64,
    pub prompt_ms: f64,
    pub decode_ms: f64,
    pub time_to_first_token_ms: f64,
    pub prompt_tokens_per_second: f64,
    pub decode_tokens_per_second: f64,
    pub sampler_ms: f64,
    pub parser_ms: f64,
    pub draft_tokens: usize,
    pub accepted_draft_tokens: usize,
    pub draft_ms: f64,
    pub verification_ms: f64,
}

#[derive(Debug, Clone, Default)]
pub struct GenerationSnapshot {
    pub cached_prompt_tokens: usize,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub metrics: GenerationMetrics,
}

#[derive(Debug, Clone)]
pub struct Generation {
    pub text: String,
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
    pub cached_prompt_tokens: usize,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub finish_reason: FinishReason,
    pub metrics: GenerationMetrics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

impl ChatRole {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatContentPart {
    Text { text: String },
    Image(ImageInput),
}

/// Validated, local image bytes. HTTP/network fetching is intentionally outside the executor.
#[derive(Clone, PartialEq, Eq)]
pub struct ImageInput {
    media_type: String,
    bytes: Arc<[u8]>,
}

impl ImageInput {
    #[must_use]
    pub fn new(media_type: impl Into<String>, bytes: impl Into<Arc<[u8]>>) -> Self {
        Self {
            media_type: media_type.into(),
            bytes: bytes.into(),
        }
    }

    #[must_use]
    pub fn media_type(&self) -> &str {
        &self.media_type
    }

    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl fmt::Debug for ImageInput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ImageInput")
            .field("media_type", &self.media_type)
            .field("byte_length", &self.bytes.len())
            .finish()
    }
}

/// The encoded shape of a chat message's content.
///
/// Keeping string content distinct from typed parts matters because Jinja templates can advertise
/// support for one representation but not the other, and llama.cpp adapts them differently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatContent {
    Text(String),
    Parts(Vec<ChatContentPart>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: Option<ChatContent>,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    #[must_use]
    pub fn text(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(ChatContent::Text(content.into())),
            reasoning: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    #[must_use]
    pub fn text_content(&self) -> String {
        match &self.content {
            None => String::new(),
            Some(ChatContent::Text(text)) => text.clone(),
            Some(ChatContent::Parts(parts)) => parts
                .iter()
                .filter_map(|part| match part {
                    ChatContentPart::Text { text } => Some(text.as_str()),
                    ChatContentPart::Image(_) => None,
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Function {
        name: String,
    },
    AllowedTools {
        mode: AllowedToolsMode,
        names: Vec<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllowedToolsMode {
    Auto,
    Required,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReasoningControl {
    ModelDefault,
    Disabled,
    Enabled { budget_tokens: Option<u32> },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResponseFormat {
    Text,
    JsonObject,
    Grammar {
        grammar: String,
    },
    JsonSchema {
        name: String,
        schema: serde_json::Value,
        strict: bool,
    },
}

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub template: ChatTemplateRequest,
    pub stop: Vec<String>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u32,
    /// Whether a capable endpoint may reuse resident prompt state. ICN currently accepts this
    /// compatibility control but always evaluates the full prompt.
    pub cache_prompt: bool,
    /// Continue sampling through model EOG tokens until the explicit generation limit.
    pub ignore_eos: bool,
    pub timings_per_token: bool,
}

#[derive(Debug, Clone)]
pub struct ChatTemplateRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: bool,
    pub reasoning: ReasoningControl,
    pub response_format: ResponseFormat,
    pub template_args: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrammarTrigger {
    Token { value: String, token: i32 },
    Word(String),
    Pattern(String),
    PatternFull(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateCapabilities {
    pub string_content: bool,
    pub typed_content: bool,
    pub tools: bool,
    pub tool_calls: bool,
    pub parallel_tool_calls: bool,
    pub system_role: bool,
    pub preserve_reasoning: bool,
    pub object_arguments: bool,
    pub enable_thinking: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedChatInfo {
    pub prompt: String,
    pub generation_prompt: String,
    pub grammar: String,
    pub grammar_lazy: bool,
    pub grammar_triggers: Vec<GrammarTrigger>,
    pub preserved_tokens: Vec<String>,
    pub additional_stops: Vec<String>,
    pub supports_thinking: bool,
    pub thinking_start_tag: Option<String>,
    pub thinking_end_tag: Option<String>,
    pub template_fingerprint: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelProperties {
    pub model_path: PathBuf,
    pub model_size_bytes: u64,
    pub architecture: Option<String>,
    pub name: Option<String>,
    pub context_tokens: u32,
    pub training_context_tokens: u32,
    pub sliding_window_tokens: i32,
    pub chat_template: String,
    pub capabilities: TemplateCapabilities,
    pub modalities: ModelModalities,
    pub mtp: MtpRuntimeProperties,
    pub execution: ExecutionConfigReport,
    pub template_fingerprint: String,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModelModalities {
    pub vision: bool,
    pub audio: bool,
    pub video: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferenceEvent {
    /// Begins the assistant stream for the first sampled-token result.
    ///
    /// Keeping this in the sampled result group lets transports reproduce llama.cpp's timing
    /// placement when the first token produces no semantic parser delta.
    StreamStart,
    ContentDelta {
        text: String,
    },
    ReasoningDelta {
        text: String,
    },
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments: String,
    },
}

#[derive(Debug, Clone)]
pub struct InferenceStreamEvent {
    pub delta: InferenceEvent,
    pub timings: Option<GenerationSnapshot>,
}

#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("model backend failed: {0}")]
    Backend(String),
    #[error("inference request was cancelled")]
    Cancelled,
    #[error("inference executor is overloaded")]
    Overloaded,
    #[error("inference executor stopped")]
    ExecutorStopped,
    #[error("token callback failed: {0}")]
    Callback(String),
}

pub trait InferenceEngine {
    fn generate(
        &mut self,
        request: &GenerateRequest,
        on_token: &mut dyn FnMut(&str) -> Result<(), InferenceError>,
    ) -> Result<Generation, InferenceError>;
}

pub trait ChatInferenceEngine {
    fn generate_chat(
        &mut self,
        messages: &[ChatMessage],
        request: &GenerateRequest,
        on_token: &mut dyn FnMut(&str) -> Result<(), InferenceError>,
    ) -> Result<Generation, InferenceError>;
}

pub trait CompletionBackend: Send + Sync + 'static {
    fn model_id(&self) -> &str;
    fn properties(&self) -> Result<ModelProperties, InferenceError> {
        Err(InferenceError::Backend(
            "this inference backend does not expose model properties".into(),
        ))
    }
    fn apply_template(
        &self,
        request: ChatTemplateRequest,
    ) -> Result<PreparedChatInfo, InferenceError> {
        let _ = request;
        Err(InferenceError::Backend(
            "this inference backend does not expose chat-template preparation".into(),
        ))
    }
    fn complete(
        &self,
        request: ChatRequest,
        on_event: &mut dyn FnMut(InferenceStreamEvent) -> Result<(), InferenceError>,
    ) -> Result<Generation, InferenceError>;
}

#[cfg(test)]
mod execution_config_tests {
    use super::*;

    #[test]
    fn defaults_match_pinned_llama_server_execution_defaults() {
        let config = ExecutionConfig::default();
        assert_eq!(config.gpu_layers, GpuLayers::Auto);
        assert!(config.use_mmap);
        assert!(!config.use_mlock);
        assert_eq!(config.split_mode, SplitMode::Layer);
        assert!(config.tensor_split.is_none());
        assert_eq!(config.cache_type_k, CacheType::F16);
        assert_eq!(config.cache_type_v, CacheType::F16);
        assert!(config.offload_kqv);
        assert!(config.operation_offload);
        assert!(!config.swa_full);
        assert!(!config.kv_unified);
        assert!(config.threads.is_none());
        assert!(config.threads_batch.is_none());
        assert_eq!(config.flash_attention, FlashAttention::Auto);
    }

    #[test]
    fn engine_vocabulary_uses_upstream_cli_spellings() {
        assert_eq!("auto".parse::<GpuLayers>(), Ok(GpuLayers::Auto));
        assert_eq!("all".parse::<GpuLayers>(), Ok(GpuLayers::All));
        assert_eq!("12".parse::<GpuLayers>(), Ok(GpuLayers::Count(12)));
        assert_eq!("tensor".parse::<SplitMode>(), Ok(SplitMode::Tensor));
        assert_eq!("iq4_nl".parse::<CacheType>(), Ok(CacheType::Iq4Nl));
        assert!("q6_k".parse::<CacheType>().is_err());
    }
}
