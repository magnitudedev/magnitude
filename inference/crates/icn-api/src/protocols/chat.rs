use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::convert::Infallible;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use icn_contracts::inference as domain;
use icn_contracts::{
    GenerationMetrics, GenerationSnapshot, GrammarTrigger, ImageInput, InferenceError,
    InferenceProgress, PreparedChatInfo,
};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value as JsonValue;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use utoipa::openapi::Ref;
use utoipa::openapi::schema::AnyOfBuilder;
use utoipa::{PartialSchema, ToSchema};

use super::super::{
    ApiError, ApiErrorBody, AppState, ErrorResponse, InferenceAdmission, ModelLoadingObserver,
    ResidentInvocation, acquire_invocation, await_inference_admission, domain_error,
    execute_with_journal, inference_error_body, media, non_empty_text, non_empty_vec,
    unix_timestamp, with_openai_request_id,
};

const DEFAULT_TEMPERATURE: f32 = 0.8;
const DEFAULT_TOP_P: f32 = 0.95;
const DEFAULT_SEED: u32 = 42;

fn deserialize_bool_or_false<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(JsonValue::deserialize(deserializer)?
        .as_bool()
        .unwrap_or(false))
}

const fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ApplyTemplateRequest {
    #[schema(nullable = false)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessageRequest>,
    #[schema(nullable = false)]
    pub tools: Option<Vec<ChatToolRequest>>,
    #[schema(nullable = false)]
    pub tool_choice: Option<ToolChoiceRequest>,
    #[schema(nullable = false)]
    pub parallel_tool_calls: Option<bool>,
    #[schema(nullable = false)]
    pub response_format: Option<ResponseFormatRequest>,
    #[schema(nullable = false)]
    pub chat_template_kwargs: Option<BTreeMap<String, JsonValue>>,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ApplyTemplateResponse {
    pub prompt: String,
    pub generation_prompt: String,
    pub grammar: String,
    pub grammar_lazy: bool,
    pub grammar_triggers: Vec<GrammarTriggerResponse>,
    pub preserved_tokens: Vec<String>,
    pub additional_stops: Vec<String>,
    pub supports_thinking: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub thinking_start_tag: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub thinking_end_tag: Option<String>,
    pub template_fingerprint: String,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum GrammarTriggerResponse {
    Token { value: String, token: i32 },
    Word { value: String },
    Pattern { value: String },
    PatternFull { value: String },
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ChatCompletionRequest {
    #[schema(nullable = false)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessageRequest>,
    #[schema(nullable = false)]
    pub max_tokens: Option<u32>,
    #[schema(nullable = false)]
    pub max_completion_tokens: Option<u32>,
    #[schema(nullable = false)]
    pub temperature: Option<f32>,
    #[schema(nullable = false)]
    pub top_p: Option<f32>,
    #[schema(nullable = false)]
    pub seed: Option<u32>,
    #[schema(nullable = false)]
    pub tools: Option<Vec<ChatToolRequest>>,
    #[schema(nullable = false)]
    pub tool_choice: Option<ToolChoiceRequest>,
    #[schema(nullable = false)]
    pub parallel_tool_calls: Option<bool>,
    #[schema(nullable = false)]
    pub store: Option<bool>,
    #[schema(nullable = false)]
    pub reasoning_effort: Option<ReasoningEffortRequest>,
    #[schema(nullable = false)]
    pub thinking_budget_tokens: Option<u32>,
    #[schema(nullable = false)]
    pub response_format: Option<ResponseFormatRequest>,
    #[schema(nullable = false)]
    pub chat_template_kwargs: Option<BTreeMap<String, JsonValue>>,
    #[schema(nullable = false)]
    pub stop: Option<StopRequest>,
    #[serde(default)]
    pub stream: bool,
    #[schema(nullable = false)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default = "default_true")]
    #[schema(default = true)]
    pub cache_prompt: bool,
    #[serde(default, deserialize_with = "deserialize_bool_or_false")]
    #[schema(default = false)]
    pub ignore_eos: bool,
    #[serde(default, deserialize_with = "deserialize_bool_or_false")]
    #[schema(default = false)]
    pub timings_per_token: bool,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatMessageRequest {
    System {
        content: String,
    },
    Developer {
        content: String,
    },
    User {
        content: ChatContentRequest,
    },
    Assistant {
        #[schema(nullable = true)]
        content: Option<String>,
        #[serde(default)]
        #[schema(nullable = false)]
        reasoning_content: Option<String>,
        #[serde(default)]
        tool_calls: Vec<ChatToolCallRequest>,
    },
    Tool {
        tool_call_id: String,
        content: ChatContentRequest,
        #[schema(nullable = false)]
        #[allow(dead_code)]
        name: Option<String>,
    },
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum ChatContentRequest {
    Text(String),
    Parts(Vec<ChatContentPartRequest>),
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatContentPartRequest {
    Text { text: String },
    ImageUrl { image_url: ImageUrlRequest },
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ImageUrlRequest {
    pub url: String,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ChatToolCallRequest {
    pub id: String,
    #[allow(dead_code)]
    pub r#type: FunctionType,
    pub function: NamedFunctionCallRequest,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct NamedFunctionCallRequest {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ChatToolRequest {
    #[allow(dead_code)]
    pub r#type: FunctionType,
    pub function: FunctionDefinitionRequest,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct FunctionDefinitionRequest {
    pub name: String,
    #[schema(nullable = false)]
    pub description: Option<String>,
    pub parameters: JsonValue,
    #[schema(nullable = false)]
    #[allow(dead_code)]
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Copy, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum FunctionType {
    Function,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum ToolChoiceRequest {
    Mode(ToolChoiceModeRequest),
    Function(FunctionToolChoiceRequest),
    AllowedTools(AllowedToolsChoiceRequest),
}

#[derive(Debug, Clone, Copy, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceModeRequest {
    None,
    Auto,
    Required,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct FunctionToolChoiceRequest {
    #[allow(dead_code)]
    pub r#type: FunctionType,
    pub function: FunctionNameRequest,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct FunctionNameRequest {
    pub name: String,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct AllowedToolsChoiceRequest {
    #[allow(dead_code)]
    pub r#type: AllowedToolsType,
    pub allowed_tools: AllowedToolsRequest,
}

#[derive(Debug, Clone, Copy, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum AllowedToolsType {
    AllowedTools,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct AllowedToolsRequest {
    pub mode: AllowedToolsModeRequest,
    pub tools: Vec<AllowedToolRequest>,
}

#[derive(Debug, Clone, Copy, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum AllowedToolsModeRequest {
    Auto,
    Required,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct AllowedToolRequest {
    #[allow(dead_code)]
    pub r#type: FunctionType,
    pub function: FunctionNameRequest,
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(transparent)]
pub struct ReasoningEffortRequest(pub String);

impl ReasoningEffortRequest {
    pub(crate) fn normalize(&self) -> Result<icn_contracts::NormalizedReasoningEffort, ApiError> {
        icn_contracts::NormalizedReasoningEffort::parse(&self.0).ok_or_else(|| {
            ApiError::invalid(format!("unsupported reasoning_effort spelling: {}", self.0))
        })
    }
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormatRequest {
    Text,
    JsonObject,
    Grammar { grammar: String },
    JsonSchema { json_schema: JsonSchemaRequest },
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct JsonSchemaRequest {
    pub name: String,
    pub schema: JsonValue,
    #[serde(default)]
    pub strict: bool,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum StopRequest {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Default, Deserialize, ToSchema)]
pub struct StreamOptions {
    #[schema(nullable = false)]
    pub include_usage: Option<bool>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub progress: Option<ChatCompletionProgress>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub timings: Option<Timings>,
}

/// The data payload of a Chat Completions SSE frame. Successful frames are
/// chunks; a failure after HTTP commitment is the standard OpenAI error
/// envelope carried by an `error` SSE event.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ChatCompletionStreamEvent {
    Chunk(ChatCompletionChunk),
    Error(ErrorResponse),
}

impl PartialSchema for ChatCompletionStreamEvent {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        AnyOfBuilder::new()
            .item(Ref::from_schema_name(ChatCompletionChunk::name()))
            .item(Ref::from_schema_name(ErrorResponse::name()))
            .description(Some(
                "A successful Chat Completions chunk or a post-commit OpenAI error envelope.",
            ))
            .into()
    }
}

impl ToSchema for ChatCompletionStreamEvent {
    fn schemas(
        schemas: &mut Vec<(
            String,
            utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>,
        )>,
    ) {
        schemas.push((
            ChatCompletionChunk::name().into_owned(),
            ChatCompletionChunk::schema(),
        ));
        ChatCompletionChunk::schemas(schemas);
        schemas.push((ErrorResponse::name().into_owned(), ErrorResponse::schema()));
        ErrorResponse::schemas(schemas);
    }
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum ChatCompletionProgress {
    ModelLoading {
        fraction: f32,
    },
    Queued,
    Preparing,
    Prefill {
        completed_tokens: u64,
        total_tokens: u64,
        cached_tokens: u64,
    },
    Generating,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: ChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Option<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub tool_calls: Option<Vec<ChunkToolCall>>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ChunkToolCall {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub r#type: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub function: Option<ChunkFunctionDelta>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ChunkFunctionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Usage,
    pub timings: Timings,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatCompletionMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionMessage {
    pub role: &'static str,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub tool_calls: Option<Vec<CompletionToolCall>>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct CompletionToolCall {
    pub id: String,
    pub r#type: &'static str,
    pub function: CompletionFunctionCall,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct CompletionFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub prompt_tokens_details: PromptTokensDetails,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct PromptTokensDetails {
    pub cached_tokens: u64,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct Timings {
    pub cache_n: u64,
    pub prompt_n: u64,
    pub prompt_ms: f64,
    pub time_to_first_token_ms: f64,
    pub prompt_per_token_ms: f64,
    pub prompt_per_second: f64,
    pub predicted_n: u64,
    pub predicted_ms: f64,
    pub predicted_per_token_ms: f64,
    pub predicted_per_second: f64,
    /// Time spent inside the native sampler for this request.
    pub sampler_ms: f64,
    /// Time spent incrementally parsing generated chat output for this request.
    pub parser_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub draft_n: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub draft_n_accepted: Option<u64>,
}

pub(crate) fn chat_completion_response(
    id: String,
    created: u64,
    model: String,
    result: &domain::InferenceResult,
) -> ChatCompletionResponse {
    let output = result.output();
    let tool_calls = (!output.tool_calls().is_empty()).then(|| {
        let calls = output.tool_calls();
        calls
            .iter()
            .map(|call| CompletionToolCall {
                id: call.id().as_str().to_owned(),
                r#type: "function",
                function: CompletionFunctionCall {
                    name: call.name().as_str().to_owned(),
                    arguments: serde_json::to_string(call.input().as_map())
                        .expect("validated JSON object is serializable"),
                },
            })
            .collect()
    });
    ChatCompletionResponse {
        id,
        object: "chat.completion",
        created,
        model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant",
                content: output.text().map(|text| text.as_str().to_owned()),
                reasoning_content: output
                    .reasoning()
                    .map(|reasoning| reasoning.as_str().to_owned()),
                tool_calls,
            },
            finish_reason: chat_finish_reason(result.termination()).to_owned(),
        }],
        usage: usage_values(result),
        timings: generation_timings(result),
    }
}

pub(crate) fn chat_finish_reason(termination: &domain::Termination) -> &'static str {
    match termination {
        domain::Termination::Natural | domain::Termination::StopSequence { .. } => "stop",
        domain::Termination::OutputLimit => "length",
        domain::Termination::ToolCalls => "tool_calls",
    }
}

fn usage_values(result: &domain::InferenceResult) -> Usage {
    let prompt_tokens = result.usage().input_tokens();
    let completion_tokens = result.usage().output_tokens();
    Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens.saturating_add(completion_tokens),
        prompt_tokens_details: PromptTokensDetails {
            cached_tokens: result.usage().cached_input_tokens(),
        },
    }
}

pub(crate) fn choice_chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: ChunkDelta,
    finish_reason: Option<String>,
    timings: Option<Timings>,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.into(),
        object: "chat.completion.chunk",
        created,
        model: model.into(),
        choices: vec![ChunkChoice {
            index: 0,
            delta,
            finish_reason,
        }],
        progress: None,
        usage: None,
        timings,
    }
}

pub(crate) fn usage_chunk(
    id: &str,
    created: u64,
    model: &str,
    generation: &domain::InferenceResult,
) -> ChatCompletionChunk {
    let prompt_tokens = generation.usage().input_tokens();
    let completion_tokens = generation.usage().output_tokens();
    ChatCompletionChunk {
        id: id.into(),
        object: "chat.completion.chunk",
        created,
        model: model.into(),
        choices: Vec::new(),
        progress: None,
        usage: Some(Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
            prompt_tokens_details: PromptTokensDetails {
                cached_tokens: generation.usage().cached_input_tokens(),
            },
        }),
        timings: Some(generation_timings(generation)),
    }
}

pub(crate) fn progress_chunk(
    id: &str,
    created: u64,
    model: &str,
    progress: InferenceProgress,
) -> ChatCompletionChunk {
    let progress = match progress {
        InferenceProgress::Queued => ChatCompletionProgress::Queued,
        InferenceProgress::Preparing => ChatCompletionProgress::Preparing,
        InferenceProgress::Prefill {
            completed_tokens,
            total_tokens,
            cached_tokens,
        } => ChatCompletionProgress::Prefill {
            completed_tokens: completed_tokens as u64,
            total_tokens: total_tokens as u64,
            cached_tokens: cached_tokens as u64,
        },
        InferenceProgress::Generating => ChatCompletionProgress::Generating,
    };
    ChatCompletionChunk {
        id: id.into(),
        object: "chat.completion.chunk",
        created,
        model: model.into(),
        choices: Vec::new(),
        progress: Some(progress),
        usage: None,
        timings: None,
    }
}

pub(crate) fn loading_progress_chunk(
    id: &str,
    created: u64,
    model: &str,
    fraction: f32,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.into(),
        object: "chat.completion.chunk",
        created,
        model: model.into(),
        choices: Vec::new(),
        progress: Some(ChatCompletionProgress::ModelLoading {
            fraction: fraction.clamp(0.0, 1.0),
        }),
        usage: None,
        timings: None,
    }
}

pub(crate) fn generation_timings(generation: &domain::InferenceResult) -> Timings {
    timing_values(
        generation.usage().cached_input_tokens() as usize,
        generation.usage().input_tokens() as usize,
        generation.usage().output_tokens() as usize,
        generation.metrics(),
    )
}

pub(crate) fn snapshot_timings(snapshot: &GenerationSnapshot) -> Timings {
    timing_values(
        snapshot.cached_prompt_tokens,
        snapshot.prompt_tokens,
        snapshot.generated_tokens,
        &snapshot.metrics,
    )
}

pub(crate) fn timing_values(
    cached_prompt_tokens: usize,
    prompt_tokens: usize,
    generated_tokens: usize,
    metrics: &GenerationMetrics,
) -> Timings {
    let prompt_n = prompt_tokens.saturating_sub(cached_prompt_tokens);
    Timings {
        cache_n: cached_prompt_tokens as u64,
        prompt_n: prompt_n as u64,
        prompt_ms: metrics.prompt_ms,
        time_to_first_token_ms: metrics.time_to_first_token_ms,
        prompt_per_token_ms: per_token_ms(prompt_n, metrics.prompt_ms),
        prompt_per_second: rate(prompt_n, metrics.prompt_ms),
        predicted_n: generated_tokens as u64,
        predicted_ms: metrics.decode_ms,
        predicted_per_token_ms: per_token_ms(generated_tokens, metrics.decode_ms),
        predicted_per_second: rate(generated_tokens, metrics.decode_ms),
        sampler_ms: metrics.sampler_ms,
        parser_ms: metrics.parser_ms,
        draft_n: (metrics.draft_tokens > 0).then_some(metrics.draft_tokens as u64),
        draft_n_accepted: (metrics.draft_tokens > 0)
            .then_some(metrics.accepted_draft_tokens as u64),
    }
}

fn per_token_ms(tokens: usize, elapsed_ms: f64) -> f64 {
    if tokens == 0 {
        0.0
    } else {
        elapsed_ms / tokens as f64
    }
}

fn rate(tokens: usize, elapsed_ms: f64) -> f64 {
    if tokens == 0 || elapsed_ms <= 0.0 {
        0.0
    } else {
        1_000.0 * tokens as f64 / elapsed_ms
    }
}

pub(crate) fn inference_output_delta(
    event: domain::InferenceOutputEvent,
) -> Result<Option<ChunkDelta>, InferenceError> {
    Ok(Some(match event {
        domain::InferenceOutputEvent::Started => ChunkDelta {
            role: Some("assistant".into()),
            content: Some(None),
            ..ChunkDelta::default()
        },
        domain::InferenceOutputEvent::TextDelta { text } => ChunkDelta {
            content: Some(Some(text.as_str().to_owned())),
            ..ChunkDelta::default()
        },
        domain::InferenceOutputEvent::ReasoningDelta { text } => ChunkDelta {
            reasoning_content: Some(text.as_str().to_owned()),
            ..ChunkDelta::default()
        },
        domain::InferenceOutputEvent::ToolCallStarted { index, id, name } => {
            let index = u32::try_from(index).map_err(|_| {
                InferenceError::Callback("tool-call index exceeds the HTTP protocol range".into())
            })?;
            ChunkDelta {
                tool_calls: Some(vec![ChunkToolCall {
                    index,
                    r#type: Some("function"),
                    id: Some(id.as_str().to_owned()),
                    function: Some(ChunkFunctionDelta {
                        name: Some(name.as_str().to_owned()),
                        arguments: None,
                    }),
                }]),
                ..ChunkDelta::default()
            }
        }
        domain::InferenceOutputEvent::ToolInputDelta {
            index,
            json_fragment,
        } => {
            let index = u32::try_from(index).map_err(|_| {
                InferenceError::Callback("tool-call index exceeds the HTTP protocol range".into())
            })?;
            ChunkDelta {
                tool_calls: Some(vec![ChunkToolCall {
                    index,
                    r#type: None,
                    id: None,
                    function: Some(ChunkFunctionDelta {
                        name: None,
                        arguments: Some(json_fragment.as_str().to_owned()),
                    }),
                }]),
                ..ChunkDelta::default()
            }
        }
        domain::InferenceOutputEvent::ToolCallFinished { .. } => return Ok(None),
    }))
}

pub(crate) fn emit_chunk(
    sender: &mpsc::Sender<Result<Event, Infallible>>,
    chunk: &ChatCompletionChunk,
) -> bool {
    serde_json::to_string(&ChatCompletionStreamEvent::Chunk(chunk.clone()))
        .ok()
        .and_then(|data| sender.blocking_send(Ok(Event::default().data(data))).ok())
        .is_some()
}

pub(crate) fn emit_done(sender: &mpsc::Sender<Result<Event, Infallible>>) {
    let _ = sender.blocking_send(Ok(Event::default().data("[DONE]")));
}

pub(crate) fn emit_stream_error(
    sender: &mpsc::Sender<Result<Event, Infallible>>,
    error: ApiErrorBody,
) {
    if let Ok(data) =
        serde_json::to_string(&ChatCompletionStreamEvent::Error(ErrorResponse { error }))
    {
        let _ = sender.blocking_send(Ok(Event::default().event("error").data(data)));
    }
}

pub(crate) fn validate_apply_template_request(
    request: ApplyTemplateRequest,
) -> Result<domain::InferenceRequest<domain::ReasoningIntent>, ApiError> {
    let validated = validate_request(ChatCompletionRequest {
        model: request.model,
        messages: request.messages,
        max_tokens: None,
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        seed: None,
        tools: request.tools,
        tool_choice: request.tool_choice,
        parallel_tool_calls: request.parallel_tool_calls,
        store: None,
        reasoning_effort: None,
        thinking_budget_tokens: None,
        response_format: request.response_format,
        chat_template_kwargs: request.chat_template_kwargs,
        stop: None,
        stream: true,
        stream_options: None,
        cache_prompt: true,
        ignore_eos: false,
        timings_per_token: false,
    })?;
    let (request, _) = finalize_request(validated)?;
    Ok(request)
}

pub(crate) fn apply_template_response(prepared: PreparedChatInfo) -> ApplyTemplateResponse {
    ApplyTemplateResponse {
        prompt: prepared.prompt,
        generation_prompt: prepared.generation_prompt,
        grammar: prepared.grammar,
        grammar_lazy: prepared.grammar_lazy,
        grammar_triggers: prepared
            .grammar_triggers
            .into_iter()
            .map(|trigger| match trigger {
                GrammarTrigger::Token { value, token } => {
                    GrammarTriggerResponse::Token { value, token }
                }
                GrammarTrigger::Word(value) => GrammarTriggerResponse::Word { value },
                GrammarTrigger::Pattern(value) => GrammarTriggerResponse::Pattern { value },
                GrammarTrigger::PatternFull(value) => GrammarTriggerResponse::PatternFull { value },
            })
            .collect(),
        preserved_tokens: prepared.preserved_tokens,
        additional_stops: prepared.additional_stops,
        supports_thinking: prepared.supports_thinking,
        thinking_start_tag: prepared.thinking_start_tag,
        thinking_end_tag: prepared.thinking_end_tag,
        template_fingerprint: prepared.template_fingerprint,
    }
}

pub(crate) struct ValidatedChatRequest {
    pub(crate) model: Option<String>,
    context: domain::InferenceContext,
    tools: Vec<domain::ToolDefinition>,
    tool_choice: domain::ToolChoice,
    parallel_tool_calls: bool,
    reasoning_effort: Option<ReasoningEffortRequest>,
    thinking_budget_tokens: Option<u32>,
    response_format: domain::OutputConstraint,
    template_args: BTreeMap<String, JsonValue>,
    stop: Vec<String>,
    max_tokens: Option<NonZeroU32>,
    temperature: f32,
    top_p: f32,
    seed: u32,
    cache_prompt: bool,
    ignore_eos: bool,
    pub(crate) timings_per_token: bool,
    include_usage: bool,
    pub(crate) stream: bool,
}

pub(crate) struct AdaptedChatRequest {
    pub(crate) invocation: domain::InferenceInvocation,
    pub(crate) stream: bool,
    pub(crate) timings_per_token: bool,
    pub(crate) include_usage: bool,
}

pub(crate) fn adapt_request(
    request: ChatCompletionRequest,
) -> Result<AdaptedChatRequest, ApiError> {
    let validated = validate_request(request)?;
    let model = validated
        .model
        .clone()
        .filter(|model| !model.is_empty())
        .ok_or_else(|| ApiError::invalid("model is required"))?;
    let stream = validated.stream;
    let timings_per_token = validated.timings_per_token;
    let (request, include_usage) = finalize_request(validated)?;
    Ok(AdaptedChatRequest {
        invocation: domain::InferenceInvocation::new(
            domain::InferenceModelSelector::try_new(model).map_err(domain_error)?,
            request,
        ),
        stream,
        timings_per_token,
        include_usage,
    })
}

pub(crate) fn validate_request(
    request: ChatCompletionRequest,
) -> Result<ValidatedChatRequest, ApiError> {
    if request.store == Some(true) {
        return Err(ApiError::invalid(
            "store is not supported by this local runtime",
        ));
    }
    if request.messages.is_empty() {
        return Err(ApiError::invalid("messages must not be empty"));
    }
    if request.model.as_deref().is_some_and(str::is_empty) {
        return Err(ApiError::invalid("model must not be empty"));
    }
    if request.max_tokens.is_some() && request.max_completion_tokens.is_some() {
        return Err(ApiError::invalid(
            "max_tokens and max_completion_tokens cannot both be set",
        ));
    }
    let max_tokens = request
        .max_completion_tokens
        .or(request.max_tokens)
        .map(|value| {
            NonZeroU32::new(value)
                .ok_or_else(|| ApiError::invalid("max tokens must be greater than zero"))
        })
        .transpose()?;
    let temperature = request.temperature.unwrap_or(DEFAULT_TEMPERATURE);
    if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
        return Err(ApiError::invalid(
            "temperature must be finite and between 0 and 2",
        ));
    }
    let top_p = request.top_p.unwrap_or(DEFAULT_TOP_P);
    if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
        return Err(ApiError::invalid(
            "top_p must be finite and between 0 and 1",
        ));
    }
    let context = chat_context(request.messages)?;
    let (tools, tool_names) = tools(request.tools.unwrap_or_default())?;
    let tool_choice = tool_choice(request.tool_choice, &tool_names)?;
    let template_args = request.chat_template_kwargs.unwrap_or_default();
    if template_args.keys().any(String::is_empty) {
        return Err(ApiError::invalid(
            "chat_template_kwargs keys must not be empty",
        ));
    }
    let response_format = response_format(request.response_format)?;
    let stop = stops(request.stop)?;
    Ok(ValidatedChatRequest {
        model: request.model,
        context,
        tools,
        tool_choice,
        parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
        reasoning_effort: request.reasoning_effort,
        thinking_budget_tokens: request.thinking_budget_tokens,
        response_format,
        template_args,
        stop,
        max_tokens,
        temperature,
        top_p,
        seed: request.seed.unwrap_or(DEFAULT_SEED),
        cache_prompt: request.cache_prompt,
        ignore_eos: request.ignore_eos,
        timings_per_token: request.timings_per_token,
        include_usage: request
            .stream_options
            .and_then(|options| options.include_usage)
            .unwrap_or(false),
        stream: request.stream,
    })
}

pub(crate) fn finalize_request(
    mut validated: ValidatedChatRequest,
) -> Result<(domain::InferenceRequest<domain::ReasoningIntent>, bool), ApiError> {
    let reasoning = reasoning_intent(
        validated.reasoning_effort,
        validated.thinking_budget_tokens,
        &mut validated.template_args,
    )?;
    let tools = domain::ToolConfiguration::try_new(
        validated.tools,
        validated.tool_choice,
        if validated.parallel_tool_calls {
            domain::ToolParallelism::Parallel
        } else {
            domain::ToolParallelism::Sequential
        },
    )
    .map_err(domain_error)?;
    let stops = validated
        .stop
        .into_iter()
        .map(|stop| domain::StopSequence::try_new(stop).map_err(domain_error))
        .collect::<Result<Vec<_>, _>>()?;
    let generation = domain::GenerationParameters::new(
        validated.max_tokens,
        domain::SamplingParameters::new(
            domain::Temperature::try_new(validated.temperature).map_err(domain_error)?,
            domain::TopP::try_new(validated.top_p).map_err(domain_error)?,
            validated.seed,
        ),
        stops,
        if validated.ignore_eos {
            domain::EndOfGenerationPolicy::IgnoreModelEnd
        } else {
            domain::EndOfGenerationPolicy::StopAtModelEnd
        },
    );
    Ok((
        domain::InferenceRequest::new(
            validated.context,
            tools,
            reasoning,
            validated.response_format,
            generation,
            if validated.cache_prompt {
                domain::PromptReusePolicy::Allowed
            } else {
                domain::PromptReusePolicy::Disabled
            },
        ),
        validated.include_usage,
    ))
}

fn chat_context(messages: Vec<ChatMessageRequest>) -> Result<domain::InferenceContext, ApiError> {
    let mut messages = VecDeque::from(messages);
    let mut instructions = Vec::new();
    while matches!(
        messages.front(),
        Some(ChatMessageRequest::System { .. } | ChatMessageRequest::Developer { .. })
    ) {
        let content = match messages.pop_front().expect("front was present") {
            ChatMessageRequest::System { content } | ChatMessageRequest::Developer { content } => {
                content
            }
            _ => unreachable!("front variant was checked"),
        };
        if !content.is_empty() {
            instructions.push(content);
        }
    }
    let system = optional_non_empty_text(
        (!instructions.is_empty()).then(|| instructions.join("\n")),
        "system and developer message content",
    )?;
    let mut entries = Vec::new();
    while let Some(message) = messages.pop_front() {
        match message {
            ChatMessageRequest::System { .. } | ChatMessageRequest::Developer { .. } => {
                return Err(ApiError::invalid(
                    "system and developer messages must precede conversation entries",
                ));
            }
            ChatMessageRequest::User { content } => {
                entries.push(domain::ContextEntry::User {
                    entry: domain::UserEntry::new(user_content(content)?),
                });
            }
            ChatMessageRequest::Tool { .. } => {
                return Err(ApiError::invalid(
                    "tool results must immediately follow the assistant tool calls they complete",
                ));
            }
            ChatMessageRequest::Assistant {
                content,
                reasoning_content,
                tool_calls,
            } => {
                let reasoning =
                    optional_non_empty_text(reasoning_content, "assistant reasoning_content")?;
                let text = optional_non_empty_text(content, "assistant content")?;
                if text.is_none() && reasoning.is_none() && tool_calls.is_empty() {
                    return Err(ApiError::invalid(
                        "assistant content is required unless tool_calls are present",
                    ));
                }
                let exchanges = if tool_calls.is_empty() {
                    Vec::new()
                } else {
                    let calls = canonical_tool_calls(tool_calls)?;
                    let mut results = BTreeMap::new();
                    while matches!(messages.front(), Some(ChatMessageRequest::Tool { .. })) {
                        let Some(ChatMessageRequest::Tool {
                            tool_call_id,
                            content,
                            name: _,
                        }) = messages.pop_front()
                        else {
                            unreachable!("front variant was checked")
                        };
                        require_non_empty(&tool_call_id, "tool_call_id")?;
                        if results
                            .insert(tool_call_id.clone(), tool_result(content)?)
                            .is_some()
                        {
                            return Err(ApiError::invalid(format!(
                                "duplicate tool result for call: {tool_call_id}"
                            )));
                        }
                    }
                    let mut exchanges = Vec::with_capacity(calls.len());
                    for call in calls {
                        let id = call.id().as_str();
                        let result = results.remove(id).ok_or_else(|| {
                            ApiError::invalid(format!(
                                "assistant tool call {id} has no immediately following result"
                            ))
                        })?;
                        exchanges.push(domain::ToolExchange::new(call, result));
                    }
                    if let Some(unmatched) = results.keys().next() {
                        return Err(ApiError::invalid(format!(
                            "tool result {unmatched} does not match an assistant tool call"
                        )));
                    }
                    exchanges
                };
                let entry = domain::AssistantEntry::new(reasoning, text, exchanges);
                entries.push(domain::ContextEntry::Assistant { entry });
            }
        }
    }
    Ok(domain::InferenceContext::new(
        system,
        non_empty_vec(entries, "conversation entries")?,
    ))
}

fn canonical_tool_calls(
    calls: Vec<ChatToolCallRequest>,
) -> Result<Vec<domain::ToolCall>, ApiError> {
    let mut ids = BTreeSet::new();
    calls
        .into_iter()
        .map(|call| {
            let id = domain::ToolCallId::try_new(call.id).map_err(domain_error)?;
            if !ids.insert(id.as_str().to_owned()) {
                return Err(ApiError::invalid(format!(
                    "duplicate assistant tool-call id: {}",
                    id.as_str()
                )));
            }
            let name = domain::ToolName::try_new(call.function.name).map_err(domain_error)?;
            let input = serde_json::from_str::<domain::JsonObject>(&call.function.arguments)
                .map_err(|error| {
                    ApiError::invalid(format!(
                        "assistant tool-call arguments must be a JSON object: {error}"
                    ))
                })?;
            Ok(domain::ToolCall::new(id, name, input))
        })
        .collect()
}

fn user_content(content: ChatContentRequest) -> Result<Vec<domain::UserContent>, ApiError> {
    let mut values = Vec::new();
    match content {
        ChatContentRequest::Text(text) => {
            if let Some(text) = optional_non_empty_text(Some(text), "user content")? {
                values.push(domain::UserContent::Text { text });
            }
        }
        ChatContentRequest::Parts(parts) => {
            for part in parts {
                match part {
                    ChatContentPartRequest::Text { text } => {
                        if let Some(text) =
                            optional_non_empty_text(Some(text), "user text content")?
                        {
                            values.push(domain::UserContent::Text { text });
                        }
                    }
                    ChatContentPartRequest::ImageUrl { image_url } => {
                        values.push(domain::UserContent::Image {
                            image: decoded_image(image_url)?,
                        });
                    }
                }
            }
        }
    }
    Ok(values)
}

fn tool_result(content: ChatContentRequest) -> Result<domain::ToolResult, ApiError> {
    let mut values = Vec::new();
    match content {
        ChatContentRequest::Text(text) => {
            if let Some(text) = optional_non_empty_text(Some(text), "tool result content")? {
                values.push(domain::ToolResultContent::Text { text });
            }
        }
        ChatContentRequest::Parts(parts) => {
            for part in parts {
                match part {
                    ChatContentPartRequest::Text { text } => {
                        if let Some(text) = optional_non_empty_text(Some(text), "tool result text")?
                        {
                            values.push(domain::ToolResultContent::Text { text });
                        }
                    }
                    ChatContentPartRequest::ImageUrl { image_url } => {
                        values.push(domain::ToolResultContent::Image {
                            image: decoded_image(image_url)?,
                        });
                    }
                }
            }
        }
    }
    Ok(domain::ToolResult::new(
        domain::ToolOutcome::Success,
        values,
    ))
}

fn decoded_image(image_url: ImageUrlRequest) -> Result<ImageInput, ApiError> {
    require_non_empty(&image_url.url, "image_url.url")?;
    let image = media::decode_image_data_url(&image_url.url, media::MAX_HTTP_IMAGE_BYTES)
        .map_err(|error| ApiError::invalid(error.to_string()))?;
    Ok(ImageInput::new(image.media_type, image.bytes))
}

fn optional_non_empty_text(
    value: Option<String>,
    field: &'static str,
) -> Result<Option<domain::NonEmptyText>, ApiError> {
    value
        .filter(|value| !value.is_empty())
        .map(|value| non_empty_text(value, field))
        .transpose()
}

fn tools(
    requests: Vec<ChatToolRequest>,
) -> Result<(Vec<domain::ToolDefinition>, BTreeSet<String>), ApiError> {
    let mut names = BTreeSet::new();
    let tools = requests
        .into_iter()
        .map(|tool| {
            let ChatToolRequest {
                r#type: _,
                function,
            } = tool;
            require_non_empty(&function.name, "tool function name")?;
            if !names.insert(function.name.clone()) {
                return Err(ApiError::invalid(format!(
                    "duplicate tool function name: {}",
                    function.name
                )));
            }
            let JsonValue::Object(parameters) = function.parameters else {
                return Err(ApiError::invalid(
                    "tool function parameters must be a JSON Schema object",
                ));
            };
            Ok(domain::ToolDefinition::new(
                domain::ToolName::try_new(function.name).map_err(domain_error)?,
                function.description,
                domain::JsonObject::new(parameters),
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((tools, names))
}

fn tool_choice(
    request: Option<ToolChoiceRequest>,
    tool_names: &BTreeSet<String>,
) -> Result<domain::ToolChoice, ApiError> {
    let choice = match request {
        None | Some(ToolChoiceRequest::Mode(ToolChoiceModeRequest::Auto)) => {
            domain::ToolChoice::Auto
        }
        Some(ToolChoiceRequest::Mode(ToolChoiceModeRequest::None)) => domain::ToolChoice::Disabled,
        Some(ToolChoiceRequest::Mode(ToolChoiceModeRequest::Required)) => {
            if tool_names.is_empty() {
                return Err(ApiError::invalid("tool_choice required requires tools"));
            }
            domain::ToolChoice::Required
        }
        Some(ToolChoiceRequest::Function(request)) => {
            require_non_empty(&request.function.name, "tool_choice function name")?;
            require_known_tool(&request.function.name, tool_names)?;
            domain::ToolChoice::Specific {
                name: domain::ToolName::try_new(request.function.name).map_err(domain_error)?,
            }
        }
        Some(ToolChoiceRequest::AllowedTools(request)) => {
            if request.allowed_tools.tools.is_empty() {
                return Err(ApiError::invalid(
                    "tool_choice allowed_tools requires at least one tool",
                ));
            }
            let mut selected = BTreeSet::new();
            let names = request
                .allowed_tools
                .tools
                .into_iter()
                .map(|tool| {
                    require_non_empty(&tool.function.name, "allowed tool name")?;
                    require_known_tool(&tool.function.name, tool_names)?;
                    if !selected.insert(tool.function.name.clone()) {
                        return Err(ApiError::invalid(format!(
                            "duplicate allowed tool name: {}",
                            tool.function.name
                        )));
                    }
                    domain::ToolName::try_new(tool.function.name).map_err(domain_error)
                })
                .collect::<Result<Vec<_>, _>>()?;
            domain::ToolChoice::Allowed {
                names,
                required: matches!(
                    request.allowed_tools.mode,
                    AllowedToolsModeRequest::Required
                ),
            }
        }
    };
    Ok(choice)
}

fn reasoning_intent(
    effort: Option<ReasoningEffortRequest>,
    budget_tokens: Option<u32>,
    template_args: &mut BTreeMap<String, JsonValue>,
) -> Result<domain::ReasoningIntent, ApiError> {
    const OWNED_KEYS: &[&str] = &[
        "enable_thinking",
        "thinking",
        "thinking_mode",
        "reasoning_effort",
        "thinking_budget",
    ];
    let raw_reasoning_controls = template_args
        .keys()
        .any(|key| OWNED_KEYS.contains(&key.as_str()));

    let budget = budget_tokens
        .map(|value| {
            NonZeroU32::new(value)
                .ok_or_else(|| ApiError::invalid("thinking_budget_tokens must be positive"))
        })
        .transpose()?;
    let template_args = std::mem::take(template_args);
    match effort {
        Some(effort) => {
            if raw_reasoning_controls {
                return Err(ApiError::invalid(
                    "reasoning_effort conflicts with reasoning controls in chat_template_kwargs",
                ));
            }
            let effort = effort.normalize()?;
            if budget.is_some() && effort.as_str() == "none" {
                return Err(ApiError::invalid(
                    "thinking_budget_tokens cannot be used when reasoning is disabled (reasoning_effort none)",
                ));
            }
            Ok(domain::ReasoningIntent::Effort {
                effort,
                template_args,
                budget,
            })
        }
        None if raw_reasoning_controls => {
            let explicitly_disabled = matches!(
                template_args
                    .get("enable_thinking")
                    .or_else(|| template_args.get("thinking")),
                Some(JsonValue::Bool(false))
            ) || matches!(
                template_args
                    .get("thinking_mode")
                    .and_then(JsonValue::as_str),
                Some("chat" | "disabled")
            ) || template_args
                .get("reasoning_effort")
                .and_then(JsonValue::as_str)
                .and_then(icn_contracts::NormalizedReasoningEffort::parse)
                .is_some_and(|effort| effort.as_str() == "none");
            if budget.is_some() && explicitly_disabled {
                return Err(ApiError::invalid(
                    "thinking_budget_tokens cannot be used when raw template controls disable reasoning",
                ));
            }
            Ok(domain::ReasoningIntent::ModelDefault {
                template_args,
                budget,
            })
        }
        None => Ok(domain::ReasoningIntent::ModelDefault {
            template_args,
            budget,
        }),
    }
}

fn response_format(
    request: Option<ResponseFormatRequest>,
) -> Result<domain::OutputConstraint, ApiError> {
    match request.unwrap_or(ResponseFormatRequest::Text) {
        ResponseFormatRequest::Text => Ok(domain::OutputConstraint::Text),
        ResponseFormatRequest::JsonObject => Ok(domain::OutputConstraint::JsonObject),
        ResponseFormatRequest::Grammar { grammar } => {
            require_non_empty(&grammar, "response_format grammar")?;
            Ok(domain::OutputConstraint::Grammar {
                constraint: domain::GrammarConstraint::try_new(grammar).map_err(domain_error)?,
            })
        }
        ResponseFormatRequest::JsonSchema { json_schema } => {
            require_non_empty(&json_schema.name, "response_format json_schema name")?;
            let JsonValue::Object(schema) = json_schema.schema else {
                return Err(ApiError::invalid(
                    "response_format JSON Schema must be a JSON object",
                ));
            };
            Ok(domain::OutputConstraint::JsonSchema {
                constraint: domain::JsonSchemaConstraint::new(
                    json_schema.name,
                    domain::JsonObject::new(schema),
                    json_schema.strict,
                ),
            })
        }
    }
}

fn stops(request: Option<StopRequest>) -> Result<Vec<String>, ApiError> {
    let values = match request {
        None => Vec::new(),
        Some(StopRequest::One(stop)) => vec![stop],
        Some(StopRequest::Many(stops)) => stops,
    };
    let mut seen = BTreeSet::new();
    values
        .into_iter()
        .map(|stop| {
            require_non_empty(&stop, "stop sequence")?;
            if !seen.insert(stop.clone()) {
                return Err(ApiError::invalid(format!(
                    "duplicate stop sequence: {stop}"
                )));
            }
            Ok(stop)
        })
        .collect()
}

fn require_known_tool(name: &str, tool_names: &BTreeSet<String>) -> Result<(), ApiError> {
    if tool_names.contains(name) {
        Ok(())
    } else {
        Err(ApiError::invalid(format!(
            "tool_choice references undefined tool: {name}"
        )))
    }
}

fn require_non_empty(value: &str, field: &str) -> Result<(), ApiError> {
    if value.is_empty() {
        Err(ApiError::invalid(format!("{field} must not be empty")))
    } else {
        Ok(())
    }
}
#[utoipa::path(post, path = "/v1/chat/completions", operation_id = "createChatCompletion", tag = "chat",
    request_body = ChatCompletionRequest,
    params(
        ("Magnitude-Include-Progress" = Option<bool>, Header, nullable = false, description = "Include Magnitude loading and inference progress events")
    ),
    responses(
        (status = 200, description = "OpenAI-compatible completion or event stream", content(
            (ChatCompletionResponse = "application/json"),
            (String = "text/event-stream")
        )),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 404, description = "Requested model is unavailable", body = ErrorResponse),
        (status = 409, description = "Runtime model cannot be admitted", body = ErrorResponse),
        (status = 422, description = "Runtime target failed validation", body = ErrorResponse),
        (status = 500, description = "Runtime load or inference failed", body = ErrorResponse)
    )
)]
#[tracing::instrument(
    name = "icn.chat_completions",
    skip_all,
    fields(completion.id = tracing::field::Empty, model.id = tracing::field::Empty),
    err(Debug)
)]
pub(crate) async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) = payload.map_err(|error| ApiError::invalid(error.body_text()))?;
    let request = adapt_request(request)?;
    let model_id = request.invocation.model().as_str().to_owned();
    let controller = state
        .model_controller
        .as_ref()
        .ok_or_else(|| ApiError::server("model control is not configured"))?;
    let include_progress = headers
        .get("Magnitude-Include-Progress")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.eq_ignore_ascii_case("true"));
    if include_progress {
        let id = format!(
            "chatcmpl-icn-{}",
            state.next_id.fetch_add(1, Ordering::Relaxed)
        );
        let created = unix_timestamp();
        let model = model_id.clone();
        let (sender, receiver) = mpsc::channel::<Result<Event, Infallible>>(16);
        let progress_sender = sender.clone();
        let progress_id = id.clone();
        let progress_model = model.clone();
        let progress: ModelLoadingObserver = Arc::new(move |fraction| {
            let chunk = loading_progress_chunk(&progress_id, created, &progress_model, fraction);
            if let Ok(data) = serde_json::to_string(&chunk) {
                let _ = progress_sender.try_send(Ok(Event::default().data(data)));
            }
        });
        let controller = Arc::clone(controller);
        let span = tracing::Span::current();
        let request_id = id.clone();
        tokio::spawn(async move {
            let resident = tokio::select! {
                result = acquire_invocation(&controller, request.invocation, Some(progress)) => result,
                _ = sender.closed() => return,
            };
            match resident {
                Ok(resident) => start_chat_completion(
                    resident,
                    InferenceAdmission::detached(),
                    id,
                    created,
                    sender,
                    span,
                    ChatStreamOptions {
                        include_progress: true,
                        timings_per_token: request.timings_per_token,
                        include_usage: request.include_usage,
                    },
                ),
                Err(error) => {
                    if let Ok(data) = serde_json::to_string(&ErrorResponse {
                        error: error.body.error,
                    }) {
                        let _ = sender
                            .send(Ok(Event::default().event("error").data(data)))
                            .await;
                    }
                }
            }
        });
        let response = Sse::new(ReceiverStream::new(receiver))
            .keep_alive(KeepAlive::default())
            .into_response();
        return Ok(with_openai_request_id(response, &request_id));
    }
    let stream = request.stream;
    let timings_per_token = request.timings_per_token;
    let include_usage = request.include_usage;
    let resident = acquire_invocation(controller, request.invocation, None)
        .await
        .map_err(|error| error.with_param("messages"))?;
    chat_completion_with_resident(state, stream, timings_per_token, include_usage, resident).await
}

async fn chat_completion_with_resident(
    state: AppState,
    stream: bool,
    timings_per_token: bool,
    include_usage: bool,
    resident: ResidentInvocation,
) -> Result<Response, ApiError> {
    let id = format!(
        "chatcmpl-icn-{}",
        state.next_id.fetch_add(1, Ordering::Relaxed)
    );
    let created = unix_timestamp();
    if !stream {
        let ResidentInvocation {
            lease,
            model: response_model,
            request,
        } = resident;
        let response_id = id.clone();
        let span = tracing::Span::current();
        let result = tokio::task::spawn_blocking(move || {
            span.in_scope(|| {
                execute_with_journal(lease.backend().as_ref(), request, |_| Ok(()), |_| Ok(()))
            })
        })
        .await
        .map_err(|error| ApiError::server(format!("inference task failed: {error}")))?
        .map_err(|error| ApiError::from_inference(error).with_param("messages"))?;
        let response = Json(chat_completion_response(
            response_id,
            created,
            response_model,
            &result,
        ))
        .into_response();
        return Ok(with_openai_request_id(response, &id));
    }
    let (sender, receiver) = mpsc::channel::<Result<Event, Infallible>>(16);
    let (admission, admitted) = InferenceAdmission::channel();
    let request_id = id.clone();
    start_chat_completion(
        resident,
        admission,
        id,
        created,
        sender,
        tracing::Span::current(),
        ChatStreamOptions {
            include_progress: false,
            timings_per_token,
            include_usage,
        },
    );
    await_inference_admission(admitted)
        .await
        .map_err(|error| error.with_param("messages"))?;
    let response = Sse::new(ReceiverStream::new(receiver))
        .keep_alive(KeepAlive::default())
        .into_response();
    Ok(with_openai_request_id(response, &request_id))
}

struct ChatStreamOptions {
    include_progress: bool,
    timings_per_token: bool,
    include_usage: bool,
}

fn start_chat_completion(
    resident: ResidentInvocation,
    mut admission: InferenceAdmission,
    id: String,
    created: u64,
    sender: mpsc::Sender<Result<Event, Infallible>>,
    current_span: tracing::Span,
    options: ChatStreamOptions,
) {
    let ResidentInvocation {
        lease,
        model,
        request,
    } = resident;
    current_span.record("completion.id", id.as_str());
    current_span.record("model.id", model.as_str());

    let span = current_span;
    tokio::task::spawn_blocking(move || {
        span.in_scope(|| {
            let mut callback = |observation: &domain::InferenceObservation| {
                let (event, timings) = observation.clone().into_parts();
                let keep_timings = options.timings_per_token
                    || matches!(
                        &event,
                        domain::InferenceObservationEvent::Output {
                            event: domain::InferenceOutputEvent::Started
                        }
                    );
                let timings = keep_timings
                    .then_some(timings)
                    .flatten()
                    .map(|snapshot| snapshot_timings(&snapshot));
                let chunk = match event {
                    domain::InferenceObservationEvent::Progress { .. }
                        if !options.include_progress =>
                    {
                        return Ok(());
                    }
                    domain::InferenceObservationEvent::Progress { progress } => {
                        Some(progress_chunk(&id, created, &model, progress))
                    }
                    domain::InferenceObservationEvent::Output { event } => {
                        inference_output_delta(event)?
                            .map(|delta| choice_chunk(&id, created, &model, delta, None, timings))
                    }
                };
                if chunk
                    .as_ref()
                    .is_none_or(|chunk| emit_chunk(&sender, chunk))
                {
                    Ok(())
                } else {
                    Err(InferenceError::Callback(
                        "stream consumer disconnected".into(),
                    ))
                }
            };
            let generation = execute_with_journal(
                lease.backend().as_ref(),
                request,
                |prompt_tokens| admission.admitted(prompt_tokens),
                &mut callback,
            );
            admission.finish(&generation);
            let generation = match generation {
                Ok(generation) => generation,
                Err(error) => {
                    tracing::error!(error = %error, "chat completion failed");
                    emit_stream_error(&sender, inference_error_body(&error));
                    return;
                }
            };
            let reason = chat_finish_reason(generation.termination());
            let usage = generation.usage();
            let metrics = generation.metrics();
            tracing::info!(
                completion.id = %id,
                model.id = %model,
                finish.reason = reason,
                input.tokens = usage.input_tokens(),
                output.tokens = usage.output_tokens(),
                queue.ms = metrics.queue_ms,
                prompt.ms = metrics.prompt_ms,
                decode.ms = metrics.decode_ms,
                "chat completion finished"
            );
            let terminal_timings =
                (!options.include_usage).then(|| generation_timings(&generation));
            if !emit_chunk(
                &sender,
                &choice_chunk(
                    &id,
                    created,
                    &model,
                    ChunkDelta::default(),
                    Some(reason.into()),
                    terminal_timings,
                ),
            ) {
                return;
            }
            if options.include_usage
                && !emit_chunk(&sender, &usage_chunk(&id, created, &model, &generation))
            {
                return;
            }
            emit_done(&sender);
        });
    });
}
