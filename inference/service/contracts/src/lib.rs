use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

pub mod bootstrap_protocol;
pub mod inference;
pub mod inventory;
pub mod models;
pub mod output;

pub use inventory::*;

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
    },
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

/// Chat-template preparation result. Every field is derived from the model's chat template by the
/// engine's template stack; none describes an execution backend.
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

#[derive(Debug, Clone, thiserror::Error)]
pub enum InferenceError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    #[error(
        "prompt is too long: {prompt_tokens} tokens leave no generation capacity in a {context_capacity}-token context"
    )]
    ContextLengthExceeded {
        prompt_tokens: u64,
        context_capacity: u64,
    },
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

/// Validates that a prepared prompt leaves at least one context position for generation.
pub fn validate_inference_capacity(
    prompt_tokens: u64,
    context_capacity: u64,
) -> Result<(), InferenceError> {
    if prompt_tokens >= context_capacity {
        return Err(InferenceError::ContextLengthExceeded {
            prompt_tokens,
            context_capacity,
        });
    }
    Ok(())
}

#[cfg(test)]
mod inference_capacity_tests {
    use super::*;

    #[test]
    fn prompt_must_leave_at_least_one_generation_position() {
        assert!(validate_inference_capacity(31, 32).is_ok());
        assert!(matches!(
            validate_inference_capacity(32, 32),
            Err(InferenceError::ContextLengthExceeded {
                prompt_tokens: 32,
                context_capacity: 32,
            })
        ));
        assert!(matches!(
            validate_inference_capacity(33, 32),
            Err(InferenceError::ContextLengthExceeded {
                prompt_tokens: 33,
                context_capacity: 32,
            })
        ));
    }
}

#[cfg(test)]
mod wire_encoding_tests {
    use super::*;

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
