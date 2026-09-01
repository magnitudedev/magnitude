use std::fmt;
use std::num::{NonZeroU32, NonZeroUsize};
use std::path::PathBuf;
use std::sync::Arc;

pub mod bootstrap_protocol;
pub mod inference;
pub mod inventory;
pub mod models;
pub mod output;

pub use inventory::*;

/// Serializable execution intent supplied to the native backend planner.
///
/// This value deliberately contains policy and requested limits only. It is not an executable
/// plan: native defaults, compatibility decisions, tensor placement, and final allocation
/// parameters belong to a process-local backend plan and must never be reconstructed from this
/// representation.
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct ExecutionIntent {
    pub model_path: PathBuf,
    /// Maximum context capacity exposed to one request.
    pub context_size: u32,
    /// Physical native context allocation across all sequence partitions.
    pub physical_context_size: u32,
    pub batch_size: u32,
    pub ubatch_size: u32,
    /// Maximum number of independently owned native sequence IDs.
    pub max_sequences: u32,
    /// Maximum prompt tokens allocated to one sequence in a round-robin pass.
    pub prefill_quantum: u32,
    /// Model loading, KV allocation, offload, and native worker-pool policy.
    pub execution: ExecutionConfig,
    /// Optional multimodal projector loaded into the same model executor.
    pub projector: Option<ProjectorConfig>,
    /// Requested speculative-decoding configuration selected through native preflight.
    pub speculative: SpeculativeDecodingConfig,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SpeculativeDecodingConfig {
    Disabled {
        reason: String,
    },
    Enabled {
        source: SpeculativeDraftSource,
        method: SpeculativeMethodConfig,
        n_max: u32,
        n_min: u32,
        cache_type_k: CacheType,
        cache_type_v: CacheType,
    },
}

impl Default for SpeculativeDecodingConfig {
    fn default() -> Self {
        Self::Disabled {
            reason: "not_supported".to_owned(),
        }
    }
}

/// Durable speculative-decoding selection resolved by assessment. It names the selected method
/// and tuning parameters but never a draft file location; execution binds the draft path from
/// the currently resolved bundle.
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SpeculativeDecodingSelection {
    Disabled {
        reason: String,
    },
    Enabled {
        method: SpeculativeMethodConfig,
        n_max: u32,
        n_min: u32,
        cache_type_k: CacheType,
        cache_type_v: CacheType,
    },
}

impl From<&SpeculativeDecodingConfig> for SpeculativeDecodingSelection {
    fn from(config: &SpeculativeDecodingConfig) -> Self {
        match config {
            SpeculativeDecodingConfig::Disabled { reason } => Self::Disabled {
                reason: reason.clone(),
            },
            SpeculativeDecodingConfig::Enabled {
                source: _,
                method,
                n_max,
                n_min,
                cache_type_k,
                cache_type_v,
            } => Self::Enabled {
                method: method.clone(),
                n_max: *n_max,
                n_min: *n_min,
                cache_type_k: *cache_type_k,
                cache_type_v: *cache_type_v,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SpeculativeDraftSource {
    /// The selected method's prediction capability is executable from the target GGUF itself.
    Embedded,
    /// The selected method's prediction capability is executable from a distinct GGUF linked to
    /// the target context.
    Separate { model_path: PathBuf },
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum SpeculativeMethodConfig {
    Mtp { min_draft_probability: f32 },
    DFlash { min_sample_probability: f32 },
    DSpark { acceptance_threshold: f32 },
}

impl SpeculativeMethodConfig {
    #[must_use]
    pub fn threshold(&self) -> f32 {
        match self {
            Self::Mtp {
                min_draft_probability,
            } => *min_draft_probability,
            Self::DFlash {
                min_sample_probability,
            } => *min_sample_probability,
            Self::DSpark {
                acceptance_threshold,
            } => *acceptance_threshold,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum SpeculativeDecodingRuntimeProperties {
    Disabled {
        reason: String,
    },
    Enabled {
        source: SpeculativeDraftSource,
        method: SpeculativeMethodConfig,
        n_max: u32,
        n_min: u32,
    },
}

/// Native execution settings whose values affect loading or context allocation.
///
/// Defaults intentionally match the pinned native service profile rather than bare context
/// defaults where those differ (notably `swa_full`).
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct ExecutionConfig {
    pub gpu_layers: GpuLayers,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub split_mode: SplitMode,
    /// Explicit per-device proportions. `None` lets the native backend select placement.
    pub tensor_split: Option<Vec<f32>>,
    pub cache_type_k: CacheType,
    pub cache_type_v: CacheType,
    pub offload_kqv: bool,
    pub operation_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    /// `None` delegates selection to the pinned native backend.
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
            kv_unified: true,
            threads: None,
            threads_batch: None,
            flash_attention: FlashAttention::Auto,
        }
    }
}

/// A requested execution configuration and the concrete native parameters selected at startup.
///
/// `resolved` describes parameters passed to the native backend, not measured physical tensor placement.
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct ExecutionConfigReport {
    pub requested: ExecutionConfig,
    pub resolved: ExecutionConfig,
}

/// Native GPU-layer selection semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuLayers {
    /// Run pinned `common/fit` before loading and use its selected layer count and placement.
    Auto,
    /// Ask the native backend to offload every supported layer.
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

/// Model distribution strategy accepted by the pinned native backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
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

/// KV-cache data types accepted by the pinned native backend.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
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

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FlashAttention {
    Auto,
    Disabled,
    Enabled,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize, serde::Serialize)]
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

#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct GenerationSnapshot {
    pub cached_prompt_tokens: usize,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub metrics: GenerationMetrics,
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

impl serde::Serialize for ImageInput {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use base64::Engine as _;
        use serde::ser::SerializeStruct as _;

        let mut image = serializer.serialize_struct("ImageInput", 2)?;
        image.serialize_field("media_type", &self.media_type)?;
        image.serialize_field(
            "data_base64",
            &base64::engine::general_purpose::STANDARD.encode(&self.bytes),
        )?;
        image.end()
    }
}

impl<'de> serde::Deserialize<'de> for ImageInput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use base64::Engine as _;

        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct WireImage {
            media_type: String,
            data_base64: String,
        }

        let wire = WireImage::deserialize(deserializer)?;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(wire.data_base64)
            .map_err(serde::de::Error::custom)?;
        Ok(Self::new(wire.media_type, bytes))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub enum GrammarTrigger {
    Token { value: String, token: i32 },
    Word(String),
    Pattern(String),
    PatternFull(String),
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
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

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
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

#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
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
    pub reasoning: ReasoningProfile,
    pub modalities: ModelModalities,
    pub speculative: SpeculativeDecodingRuntimeProperties,
    pub execution: ExecutionConfigReport,
    pub template_fingerprint: String,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct ModelModalities {
    pub vision: bool,
    pub audio: bool,
    pub video: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub enum InferenceProgress {
    Queued,
    Preparing,
    Prefill {
        completed_tokens: usize,
        total_tokens: usize,
        cached_tokens: usize,
    },
    Generating,
}

#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("model backend failed: {0}")]
    Backend(String),
    #[error("inference request was cancelled")]
    Cancelled,
    #[error("model instance was stopped")]
    ModelInstanceStopped,
    #[error("inference executor is overloaded")]
    Overloaded,
    #[error("inference executor stopped")]
    ExecutorStopped,
    #[error("token callback failed: {0}")]
    Callback(String),
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
        request: inference::ResolvedInferenceRequest,
    ) -> Result<PreparedChatInfo, InferenceError> {
        let _ = request;
        Err(InferenceError::Backend(
            "this inference backend does not expose chat-template preparation".into(),
        ))
    }
    fn count_tokens(
        &self,
        request: inference::ResolvedInferenceRequest,
    ) -> Result<u64, InferenceError> {
        let _ = request;
        Err(InferenceError::Backend(
            "this inference backend does not expose token counting".into(),
        ))
    }
    fn complete(
        &self,
        request: inference::ResolvedInferenceRequest,
        on_event: &mut dyn FnMut(inference::InferenceObservation) -> Result<(), InferenceError>,
    ) -> Result<inference::InferenceCompletion, InferenceError>;
}

#[cfg(test)]
mod execution_config_tests {
    use super::*;

    #[test]
    fn defaults_match_pinned_native_service_execution_defaults() {
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
        assert!(config.kv_unified);
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

    #[test]
    fn image_input_json_uses_base64_instead_of_integer_arrays() {
        let image = ImageInput::new("image/png", vec![0, 1, 2, 255]);
        let encoded = serde_json::to_value(&image).unwrap();
        assert_eq!(encoded["media_type"], "image/png");
        assert_eq!(encoded["data_base64"], "AAEC/w==");
        let decoded: ImageInput = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, image);
    }

    #[test]
    fn speculative_methods_round_trip_with_method_specific_thresholds() {
        let methods = [
            SpeculativeMethodConfig::Mtp {
                min_draft_probability: 0.1,
            },
            SpeculativeMethodConfig::DFlash {
                min_sample_probability: 0.2,
            },
            SpeculativeMethodConfig::DSpark {
                acceptance_threshold: 0.3,
            },
        ];
        for method in methods {
            let encoded = serde_json::to_value(&method).unwrap();
            let decoded: SpeculativeMethodConfig = serde_json::from_value(encoded).unwrap();
            assert_eq!(decoded, method);
        }
    }
}
