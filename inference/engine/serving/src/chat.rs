//! OpenAI Chat Completions (stream and non-stream) and chat template
//! application.
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::convert::Infallible;
use std::num::NonZeroU32;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use magnitude_chat::output::{
    Completion, OutputEvent, OutputJournal, Progress, Termination, TimingSnapshot,
};
use magnitude_chat::request::{
    AssistantTurn, Conversation, Entry, GenerationControls, OutputFormat, PromptCache,
    SamplingControls, ToolCall, ToolDefinition, ToolExchange, ToolResultPart, Tools, UserPart,
};
use magnitude_chat::{
    ChatInput, EndOfGeneration, GenerationRequest, ReasoningIntent, ToolChoice,
    reasoning::normalize_effort,
};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value as JsonValue};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use utoipa::openapi::Ref;
use utoipa::openapi::schema::AnyOfBuilder;
use utoipa::{PartialSchema, ToSchema};

use crate::error::{ApiError, ApiErrorBody, ErrorResponse, ServingError};
use crate::source::{GenerationEvent, GenerationStream, LoadProgress};
use magnitude_engine::chat::AppliedTemplate;
use crate::{Serving, include_progress, media, unix_timestamp, with_request_id};

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

const fn one() -> u32 {
    1
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

/// Lazy grammar triggers. The engine's grammars are never lazy, so responses
/// carry none; the shape remains part of the published contract.
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
    /// Keep the `top_k` most likely tokens; 0 disables the filter.
    #[schema(nullable = false)]
    pub top_k: Option<u32>,
    #[schema(nullable = false)]
    pub min_p: Option<f32>,
    #[schema(nullable = false)]
    pub repetition_penalty: Option<f32>,
    #[schema(nullable = false)]
    pub presence_penalty: Option<f32>,
    #[schema(nullable = false)]
    pub frequency_penalty: Option<f32>,
    #[schema(nullable = false)]
    pub seed: Option<u32>,
    /// Only one choice is generated.
    #[serde(default = "one")]
    #[schema(default = 1)]
    pub n: u32,
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
    /// The normalized public effort this spelling names.
    pub(crate) fn normalize(&self) -> Result<&'static str, ApiError> {
        normalize_effort(&self.0).ok_or_else(|| {
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
    /// Time spent inside the sampler for this request.
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

pub(crate) fn chat_finish_reason(termination: &Termination) -> &'static str {
    match termination {
        Termination::Natural | Termination::StopSequence(_) => "stop",
        Termination::OutputLimit => "length",
        Termination::ToolCalls => "tool_calls",
    }
}

pub(crate) fn usage_values(completion: &Completion) -> Usage {
    let usage = completion.usage;
    Usage {
        prompt_tokens: usage.input_tokens,
        completion_tokens: usage.output_tokens,
        total_tokens: usage.input_tokens.saturating_add(usage.output_tokens),
        prompt_tokens_details: PromptTokensDetails {
            cached_tokens: usage.cached_input_tokens,
        },
    }
}

pub(crate) fn timing_values(snapshot: &TimingSnapshot) -> Timings {
    let prompt_n = snapshot
        .prompt_tokens
        .saturating_sub(snapshot.cached_prompt_tokens);
    let timings = &snapshot.timings;
    Timings {
        cache_n: snapshot.cached_prompt_tokens,
        prompt_n,
        prompt_ms: timings.prompt_ms,
        time_to_first_token_ms: timings.time_to_first_token_ms,
        prompt_per_token_ms: per_token_ms(prompt_n, timings.prompt_ms),
        prompt_per_second: rate(prompt_n, timings.prompt_ms),
        predicted_n: snapshot.generated_tokens,
        predicted_ms: timings.decode_ms,
        predicted_per_token_ms: per_token_ms(snapshot.generated_tokens, timings.decode_ms),
        predicted_per_second: rate(snapshot.generated_tokens, timings.decode_ms),
        sampler_ms: timings.sampler_ms,
        parser_ms: timings.parser_ms,
        draft_n: (timings.draft_tokens > 0).then_some(timings.draft_tokens),
        draft_n_accepted: (timings.draft_tokens > 0).then_some(timings.accepted_draft_tokens),
    }
}

fn per_token_ms(tokens: u64, elapsed_ms: f64) -> f64 {
    if tokens == 0 {
        0.0
    } else {
        elapsed_ms / tokens as f64
    }
}

fn rate(tokens: u64, elapsed_ms: f64) -> f64 {
    if tokens == 0 || elapsed_ms <= 0.0 {
        0.0
    } else {
        1_000.0 * tokens as f64 / elapsed_ms
    }
}

/// Collect a generation's terminal result and aggregate output. Output events
/// before admission and failures of the stream contract are errors.
pub(crate) async fn collect(
    mut stream: GenerationStream,
) -> Result<(magnitude_chat::output::Output, Completion), ServingError> {
    let mut journal = OutputJournal::default();
    loop {
        match stream.next().await {
            Some(GenerationEvent::Progress(_) | GenerationEvent::Admitted { .. }) => {}
            Some(GenerationEvent::Output { event, .. }) => {
                journal.push(&event).map_err(ServingError::Output)?
            }
            Some(GenerationEvent::Completed(completion)) => {
                return Ok((journal.finish().map_err(ServingError::Output)?, completion));
            }
            Some(GenerationEvent::Failed(error)) => return Err(error),
            None => return Err(ended_without_outcome()),
        }
    }
}

pub(crate) fn ended_without_outcome() -> ServingError {
    ServingError::Internal("generation ended without a terminal outcome".into())
}

/// Wait for admission, keeping every event observed on the way. A failure
/// before admission is the request's HTTP error.
pub(crate) async fn admitted(
    stream: &mut GenerationStream,
) -> Result<(u64, Vec<GenerationEvent>), ServingError> {
    let mut before = Vec::new();
    loop {
        match stream.next().await {
            Some(GenerationEvent::Admitted { prompt_tokens }) => return Ok((prompt_tokens, before)),
            Some(GenerationEvent::Failed(error)) => return Err(error),
            Some(event @ GenerationEvent::Progress(_)) => before.push(event),
            Some(_) => {
                return Err(ServingError::Internal(
                    "generation produced output before admission".into(),
                ));
            }
            None => return Err(ended_without_outcome()),
        }
    }
}

pub(crate) fn chat_completion_response(
    id: String,
    created: u64,
    model: String,
    output: magnitude_chat::output::Output,
    completion: &Completion,
) -> ChatCompletionResponse {
    let tool_calls = (!output.tool_calls.is_empty()).then(|| {
        output
            .tool_calls
            .iter()
            .map(|call| CompletionToolCall {
                id: call.id.clone(),
                r#type: "function",
                function: CompletionFunctionCall {
                    name: call.name.clone(),
                    arguments: serde_json::to_string(&call.arguments)
                        .expect("a JSON object serializes"),
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
                content: output.text,
                reasoning_content: output.reasoning,
                tool_calls,
            },
            finish_reason: chat_finish_reason(&completion.termination).to_owned(),
        }],
        usage: usage_values(completion),
        timings: timing_values(&completion.snapshot()),
    }
}

fn chunk(id: &str, created: u64, model: &str) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.into(),
        object: "chat.completion.chunk",
        created,
        model: model.into(),
        choices: Vec::new(),
        progress: None,
        usage: None,
        timings: None,
    }
}

fn choice_chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: ChunkDelta,
    finish_reason: Option<String>,
    timings: Option<Timings>,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        choices: vec![ChunkChoice {
            index: 0,
            delta,
            finish_reason,
        }],
        timings,
        ..chunk(id, created, model)
    }
}

fn progress_value(progress: Progress) -> ChatCompletionProgress {
    match progress {
        Progress::Preparing => ChatCompletionProgress::Preparing,
        Progress::Queued => ChatCompletionProgress::Queued,
        Progress::Prefill {
            completed_tokens,
            total_tokens,
            cached_tokens,
        } => ChatCompletionProgress::Prefill {
            completed_tokens,
            total_tokens,
            cached_tokens,
        },
        Progress::Generating => ChatCompletionProgress::Generating,
    }
}

pub(crate) fn output_delta(event: OutputEvent) -> Result<Option<ChunkDelta>, ServingError> {
    let index = |index: usize| {
        u32::try_from(index).map_err(|_| {
            ServingError::Internal("tool-call index exceeds the HTTP protocol range".into())
        })
    };
    Ok(Some(match event {
        OutputEvent::Started => ChunkDelta {
            role: Some("assistant".into()),
            content: Some(None),
            ..ChunkDelta::default()
        },
        OutputEvent::TextDelta(text) => ChunkDelta {
            content: Some(Some(text)),
            ..ChunkDelta::default()
        },
        OutputEvent::ReasoningDelta(text) => ChunkDelta {
            reasoning_content: Some(text),
            ..ChunkDelta::default()
        },
        OutputEvent::ToolCallStarted { index: call, id, name } => ChunkDelta {
            tool_calls: Some(vec![ChunkToolCall {
                index: index(call)?,
                r#type: Some("function"),
                id: Some(id),
                function: Some(ChunkFunctionDelta {
                    name: Some(name),
                    arguments: None,
                }),
            }]),
            ..ChunkDelta::default()
        },
        OutputEvent::ToolInputDelta { index: call, fragment } => ChunkDelta {
            tool_calls: Some(vec![ChunkToolCall {
                index: index(call)?,
                r#type: None,
                id: None,
                function: Some(ChunkFunctionDelta {
                    name: None,
                    arguments: Some(fragment),
                }),
            }]),
            ..ChunkDelta::default()
        },
        OutputEvent::ToolCallFinished { .. } => return Ok(None),
    }))
}

type SseSender = mpsc::Sender<Result<Event, Infallible>>;

async fn emit(sender: &SseSender, event: &ChatCompletionStreamEvent) -> bool {
    let kind_error = matches!(event, ChatCompletionStreamEvent::Error(_));
    let Ok(data) = serde_json::to_string(event) else {
        return false;
    };
    let mut frame = Event::default().data(data);
    if kind_error {
        frame = frame.event("error");
    }
    sender.send(Ok(frame)).await.is_ok()
}

async fn emit_chunk(sender: &SseSender, chunk: ChatCompletionChunk) -> bool {
    emit(sender, &ChatCompletionStreamEvent::Chunk(chunk)).await
}

async fn emit_error(sender: &SseSender, error: ApiErrorBody) {
    emit(sender, &ChatCompletionStreamEvent::Error(ErrorResponse { error })).await;
}

pub(crate) fn validate_apply_template_request(
    request: ApplyTemplateRequest,
) -> Result<ChatInput, ApiError> {
    let validated = validate_request(ChatCompletionRequest {
        model: request.model,
        messages: request.messages,
        max_tokens: None,
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        top_k: None,
        min_p: None,
        repetition_penalty: None,
        presence_penalty: None,
        frequency_penalty: None,
        seed: None,
        n: 1,
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
    Ok(finalize_request(validated)?.0.input)
}

pub(crate) fn apply_template_response(prepared: AppliedTemplate) -> ApplyTemplateResponse {
    ApplyTemplateResponse {
        prompt: prepared.prompt,
        generation_prompt: prepared.generation_prompt,
        grammar: prepared.grammar,
        grammar_lazy: false,
        grammar_triggers: Vec::new(),
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
    conversation: Conversation,
    tools: Vec<ToolDefinition>,
    tool_choice: ToolChoice,
    parallel_tool_calls: bool,
    reasoning_effort: Option<ReasoningEffortRequest>,
    thinking_budget_tokens: Option<u32>,
    response_format: OutputFormat,
    template_args: BTreeMap<String, JsonValue>,
    stop: Vec<String>,
    max_tokens: Option<NonZeroU32>,
    sampling: SamplingControls,
    cache_prompt: bool,
    ignore_eos: bool,
    pub(crate) timings_per_token: bool,
    include_usage: bool,
    pub(crate) stream: bool,
}

pub(crate) struct AdaptedChatRequest {
    pub(crate) model: String,
    pub(crate) request: GenerationRequest,
    pub(crate) stream: bool,
    pub(crate) timings_per_token: bool,
    pub(crate) include_usage: bool,
}

pub(crate) fn adapt_request(request: ChatCompletionRequest) -> Result<AdaptedChatRequest, ApiError> {
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
        model,
        request,
        stream,
        timings_per_token,
        include_usage,
    })
}

fn finite_in(value: f32, range: std::ops::RangeInclusive<f32>) -> bool {
    value.is_finite() && range.contains(&value)
}

pub(crate) fn validate_request(
    request: ChatCompletionRequest,
) -> Result<ValidatedChatRequest, ApiError> {
    if request.store == Some(true) {
        return Err(ApiError::invalid(
            "store is not supported by this local runtime",
        ));
    }
    if request.n != 1 {
        return Err(ApiError::invalid("only one choice (n = 1) is supported"));
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
    if !finite_in(temperature, 0.0..=2.0) {
        return Err(ApiError::invalid(
            "temperature must be finite and between 0 and 2",
        ));
    }
    let top_p = request.top_p.unwrap_or(DEFAULT_TOP_P);
    if !finite_in(top_p, 0.0..=1.0) || top_p == 0.0 {
        return Err(ApiError::invalid(
            "top_p must be finite, greater than 0 and at most 1",
        ));
    }
    let min_p = request.min_p.unwrap_or(0.0);
    if !finite_in(min_p, 0.0..=1.0) {
        return Err(ApiError::invalid("min_p must be finite and between 0 and 1"));
    }
    let repetition_penalty = request.repetition_penalty.unwrap_or(1.0);
    if !repetition_penalty.is_finite() || repetition_penalty <= 0.0 {
        return Err(ApiError::invalid(
            "repetition_penalty must be finite and positive",
        ));
    }
    let presence_penalty = request.presence_penalty.unwrap_or(0.0);
    let frequency_penalty = request.frequency_penalty.unwrap_or(0.0);
    if !finite_in(presence_penalty, -2.0..=2.0) || !finite_in(frequency_penalty, -2.0..=2.0) {
        return Err(ApiError::invalid(
            "presence_penalty and frequency_penalty must be finite and between -2 and 2",
        ));
    }
    let top_k = request.top_k.unwrap_or(0);
    if top_k > 1 << 24 {
        return Err(ApiError::invalid("top_k exceeds the supported range"));
    }
    let conversation = chat_context(request.messages)?;
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
        conversation,
        tools,
        tool_choice,
        parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
        reasoning_effort: request.reasoning_effort,
        thinking_budget_tokens: request.thinking_budget_tokens,
        response_format,
        template_args,
        stop,
        max_tokens,
        sampling: SamplingControls {
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            seed: u64::from(request.seed.unwrap_or(DEFAULT_SEED)),
        },
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
) -> Result<(GenerationRequest, bool), ApiError> {
    let (reasoning, budget) = reasoning_intent(
        validated.reasoning_effort,
        validated.thinking_budget_tokens,
        &validated.template_args,
    )?;
    let tools = Tools::new(
        validated.tools,
        validated.tool_choice,
        validated.parallel_tool_calls,
    )
    .map_err(|error| ApiError::invalid(error.to_string()))?;
    let template_arguments = std::mem::take(&mut validated.template_args)
        .into_iter()
        .collect::<Map<_, _>>();
    Ok((
        GenerationRequest {
            input: ChatInput {
                conversation: validated.conversation,
                tools,
                reasoning,
                output: validated.response_format,
                template_arguments,
            },
            controls: GenerationControls {
                max_output_tokens: validated.max_tokens,
                sampling: validated.sampling,
                stops: validated.stop,
                end_of_generation: if validated.ignore_eos {
                    EndOfGeneration::Suppress
                } else {
                    EndOfGeneration::Stop
                },
                reasoning_budget: budget,
                prompt_cache: if validated.cache_prompt {
                    PromptCache::Allowed
                } else {
                    PromptCache::Disabled
                },
            },
        },
        validated.include_usage,
    ))
}

fn chat_context(messages: Vec<ChatMessageRequest>) -> Result<Conversation, ApiError> {
    let mut messages = VecDeque::from(messages);
    let mut instructions = Vec::new();
    while matches!(
        messages.front(),
        Some(ChatMessageRequest::System { .. } | ChatMessageRequest::Developer { .. })
    ) {
        let (ChatMessageRequest::System { content } | ChatMessageRequest::Developer { content }) =
            messages.pop_front().expect("front was present")
        else {
            unreachable!("front variant was checked")
        };
        if !content.is_empty() {
            instructions.push(content);
        }
    }
    let system = (!instructions.is_empty()).then(|| instructions.join("\n"));
    let mut entries = Vec::new();
    while let Some(message) = messages.pop_front() {
        match message {
            ChatMessageRequest::System { .. } | ChatMessageRequest::Developer { .. } => {
                return Err(ApiError::invalid(
                    "system and developer messages must precede conversation entries",
                ));
            }
            ChatMessageRequest::User { content } => {
                entries.push(Entry::User(user_content(content)?));
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
                let reasoning = reasoning_content.filter(|value| !value.is_empty());
                let text = content.filter(|value| !value.is_empty());
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
                        let result = results.remove(&call.id).ok_or_else(|| {
                            ApiError::invalid(format!(
                                "assistant tool call {} has no immediately following result",
                                call.id
                            ))
                        })?;
                        exchanges.push(ToolExchange { call, result });
                    }
                    if let Some(unmatched) = results.keys().next() {
                        return Err(ApiError::invalid(format!(
                            "tool result {unmatched} does not match an assistant tool call"
                        )));
                    }
                    exchanges
                };
                entries.push(Entry::Assistant(AssistantTurn {
                    reasoning,
                    text,
                    tool_calls: exchanges,
                }));
            }
        }
    }
    Conversation::new(system, entries).map_err(|error| ApiError::invalid(error.to_string()))
}

fn canonical_tool_calls(calls: Vec<ChatToolCallRequest>) -> Result<Vec<ToolCall>, ApiError> {
    let mut ids = BTreeSet::new();
    calls
        .into_iter()
        .map(|call| {
            require_non_empty(&call.id, "tool call id")?;
            require_non_empty(&call.function.name, "tool call function name")?;
            if !ids.insert(call.id.clone()) {
                return Err(ApiError::invalid(format!(
                    "duplicate assistant tool-call id: {}",
                    call.id
                )));
            }
            let arguments = serde_json::from_str::<Map<String, JsonValue>>(&call.function.arguments)
                .map_err(|error| {
                    ApiError::invalid(format!(
                        "assistant tool-call arguments must be a JSON object: {error}"
                    ))
                })?;
            Ok(ToolCall {
                id: call.id,
                name: call.function.name,
                arguments,
            })
        })
        .collect()
}

fn user_content(content: ChatContentRequest) -> Result<Vec<UserPart>, ApiError> {
    let mut values = Vec::new();
    match content {
        ChatContentRequest::Text(text) => {
            if !text.is_empty() {
                values.push(UserPart::Text(text));
            }
        }
        ChatContentRequest::Parts(parts) => {
            for part in parts {
                match part {
                    ChatContentPartRequest::Text { text } => {
                        if !text.is_empty() {
                            values.push(UserPart::Text(text));
                        }
                    }
                    ChatContentPartRequest::ImageUrl { image_url } => {
                        values.push(UserPart::Image(decoded_image(image_url)?));
                    }
                }
            }
        }
    }
    Ok(values)
}

fn tool_result(content: ChatContentRequest) -> Result<Vec<ToolResultPart>, ApiError> {
    let mut values = Vec::new();
    match content {
        ChatContentRequest::Text(text) => {
            if !text.is_empty() {
                values.push(ToolResultPart::Text(text));
            }
        }
        ChatContentRequest::Parts(parts) => {
            for part in parts {
                match part {
                    ChatContentPartRequest::Text { text } => {
                        if !text.is_empty() {
                            values.push(ToolResultPart::Text(text));
                        }
                    }
                    ChatContentPartRequest::ImageUrl { image_url } => {
                        values.push(ToolResultPart::Image(decoded_image(image_url)?));
                    }
                }
            }
        }
    }
    Ok(values)
}

fn decoded_image(
    image_url: ImageUrlRequest,
) -> Result<magnitude_chat::request::ImageInput, ApiError> {
    require_non_empty(&image_url.url, "image_url.url")?;
    media::image(&image_url.url).map_err(|error| ApiError::invalid(error.to_string()))
}

fn tools(requests: Vec<ChatToolRequest>) -> Result<(Vec<ToolDefinition>, BTreeSet<String>), ApiError> {
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
            Ok(ToolDefinition {
                name: function.name,
                description: function.description,
                parameters,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((tools, names))
}

fn tool_choice(
    request: Option<ToolChoiceRequest>,
    tool_names: &BTreeSet<String>,
) -> Result<ToolChoice, ApiError> {
    Ok(match request {
        None | Some(ToolChoiceRequest::Mode(ToolChoiceModeRequest::Auto)) => ToolChoice::Auto,
        Some(ToolChoiceRequest::Mode(ToolChoiceModeRequest::None)) => ToolChoice::None,
        Some(ToolChoiceRequest::Mode(ToolChoiceModeRequest::Required)) => {
            if tool_names.is_empty() {
                return Err(ApiError::invalid("tool_choice required requires tools"));
            }
            ToolChoice::Required
        }
        Some(ToolChoiceRequest::Function(request)) => {
            require_non_empty(&request.function.name, "tool_choice function name")?;
            require_known_tool(&request.function.name, tool_names)?;
            ToolChoice::Named(request.function.name)
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
                    Ok(tool.function.name)
                })
                .collect::<Result<Vec<_>, _>>()?;
            ToolChoice::Allowed {
                names,
                required: matches!(
                    request.allowed_tools.mode,
                    AllowedToolsModeRequest::Required
                ),
            }
        }
    })
}

/// Raw template controls that select reasoning. A normalized effort and these
/// controls are mutually exclusive.
const RAW_REASONING_CONTROLS: &[&str] = &[
    "enable_thinking",
    "thinking",
    "thinking_mode",
    "reasoning_effort",
    "thinking_budget",
];

fn reasoning_intent(
    effort: Option<ReasoningEffortRequest>,
    budget_tokens: Option<u32>,
    template_args: &BTreeMap<String, JsonValue>,
) -> Result<(ReasoningIntent, Option<NonZeroU32>), ApiError> {
    let raw_reasoning_controls = template_args
        .keys()
        .any(|key| RAW_REASONING_CONTROLS.contains(&key.as_str()));
    let budget = budget_tokens
        .map(|value| {
            NonZeroU32::new(value)
                .ok_or_else(|| ApiError::invalid("thinking_budget_tokens must be positive"))
        })
        .transpose()?;
    match effort {
        Some(effort) => {
            if raw_reasoning_controls {
                return Err(ApiError::invalid(
                    "reasoning_effort conflicts with reasoning controls in chat_template_kwargs",
                ));
            }
            let effort = effort.normalize()?;
            if budget.is_some() && effort == "none" {
                return Err(ApiError::invalid(
                    "thinking_budget_tokens cannot be used when reasoning is disabled (reasoning_effort none)",
                ));
            }
            Ok((
                ReasoningIntent::Effort {
                    effort: effort.into(),
                },
                budget,
            ))
        }
        None => {
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
                .and_then(normalize_effort)
                == Some("none");
            if budget.is_some() && explicitly_disabled {
                return Err(ApiError::invalid(
                    "thinking_budget_tokens cannot be used when raw template controls disable reasoning",
                ));
            }
            Ok((ReasoningIntent::ModelDefault, budget))
        }
    }
}

fn response_format(request: Option<ResponseFormatRequest>) -> Result<OutputFormat, ApiError> {
    match request.unwrap_or(ResponseFormatRequest::Text) {
        ResponseFormatRequest::Text => Ok(OutputFormat::Text),
        ResponseFormatRequest::JsonObject => Ok(OutputFormat::JsonObject),
        ResponseFormatRequest::Grammar { grammar } => {
            require_non_empty(&grammar, "response_format grammar")?;
            Ok(OutputFormat::Grammar(grammar))
        }
        ResponseFormatRequest::JsonSchema { json_schema } => {
            require_non_empty(&json_schema.name, "response_format json_schema name")?;
            let JsonValue::Object(schema) = json_schema.schema else {
                return Err(ApiError::invalid(
                    "response_format JSON Schema must be a JSON object",
                ));
            };
            Ok(OutputFormat::JsonSchema {
                name: json_schema.name,
                schema,
                strict: json_schema.strict,
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

pub(crate) fn require_non_empty(value: &str, field: &str) -> Result<(), ApiError> {
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
        (status = 500, description = "Runtime load or inference failed", body = ErrorResponse),
        (status = 503, description = "Memory or request capacity is temporarily unavailable", body = ErrorResponse)
    )
)]
#[tracing::instrument(
    name = "serving.chat_completions",
    skip_all,
    fields(completion.id = tracing::field::Empty, model.id = tracing::field::Empty),
    err(Debug)
)]
pub async fn chat_completions(
    State(state): State<Serving>,
    headers: HeaderMap,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) = payload.map_err(|error| ApiError::invalid(error.body_text()))?;
    let adapted = adapt_request(request)?;
    let id = state.next_id("chatcmpl-icn-");
    let created = unix_timestamp();
    let span = tracing::Span::current();
    span.record("completion.id", id.as_str());
    span.record("model.id", adapted.model.as_str());
    let options = ChatStreamOptions {
        include_progress: include_progress(&headers),
        timings_per_token: adapted.timings_per_token,
        include_usage: adapted.include_usage,
    };
    if options.include_progress {
        // The stream commits before the model is bound so loading progress
        // reaches the client; every later failure is an in-stream error.
        let (sender, receiver) = mpsc::channel::<Result<Event, Infallible>>(16);
        let progress_sender = sender.clone();
        let progress_id = id.clone();
        let progress_model = adapted.model.clone();
        let progress: LoadProgress = std::sync::Arc::new(move |fraction| {
            let chunk = ChatCompletionChunk {
                progress: Some(ChatCompletionProgress::ModelLoading {
                    fraction: fraction.clamp(0.0, 1.0),
                }),
                ..chunk(&progress_id, created, &progress_model)
            };
            if let Ok(data) = serde_json::to_string(&chunk) {
                let _ = progress_sender.try_send(Ok(Event::default().data(data)));
            }
        });
        let frame = Frame {
            id: id.clone(),
            created,
            model: adapted.model.clone(),
        };
        tokio::spawn(async move {
            let invocation = tokio::select! {
                result = state.models.invoke(&adapted.model, Some(progress)) => result,
                _ = sender.closed() => return,
            };
            match invocation {
                Ok(invocation) => {
                    let stream = invocation.generate(adapted.request);
                    stream_chat(stream, Vec::new(), frame, sender, options).await;
                }
                Err(error) => emit_error(&sender, ApiError::from(error).body).await,
            }
        });
        let response = Sse::new(ReceiverStream::new(receiver))
            .keep_alive(KeepAlive::default())
            .into_response();
        return Ok(with_request_id(response, &id));
    }
    let invocation = state
        .models
        .invoke(&adapted.model, None)
        .await
        .map_err(|error| ApiError::from(error).with_param("model"))?;
    let mut stream = invocation.generate(adapted.request);
    if !adapted.stream {
        let (output, completion) = collect(stream)
            .await
            .map_err(|error| ApiError::from(error).with_param("messages"))?;
        let response = Json(chat_completion_response(
            id.clone(),
            created,
            adapted.model,
            output,
            &completion,
        ))
        .into_response();
        return Ok(with_request_id(response, &id));
    }
    // An ordinary stream commits only once the engine admitted the request,
    // so preparation and admission failures keep their HTTP status.
    let (_, before) = admitted(&mut stream)
        .await
        .map_err(|error| ApiError::from(error).with_param("messages"))?;
    let (sender, receiver) = mpsc::channel::<Result<Event, Infallible>>(16);
    let frame = Frame {
        id: id.clone(),
        created,
        model: adapted.model,
    };
    tokio::spawn(stream_chat(stream, before, frame, sender, options));
    let response = Sse::new(ReceiverStream::new(receiver))
        .keep_alive(KeepAlive::default())
        .into_response();
    Ok(with_request_id(response, &id))
}

#[derive(Clone, Copy)]
struct ChatStreamOptions {
    include_progress: bool,
    timings_per_token: bool,
    include_usage: bool,
}

struct Frame {
    id: String,
    created: u64,
    model: String,
}

/// Frame one generation as Chat Completions chunks. A disconnected client
/// drops the stream, which cancels the request.
async fn stream_chat(
    mut stream: GenerationStream,
    before: Vec<GenerationEvent>,
    frame: Frame,
    sender: SseSender,
    options: ChatStreamOptions,
) {
    let Frame { id, created, model } = frame;
    let mut pending = before.into_iter();
    loop {
        let event = match pending.next() {
            Some(event) => Some(event),
            None => tokio::select! {
                event = stream.next() => event,
                () = sender.closed() => return,
            },
        };
        let chunk = match event {
            Some(GenerationEvent::Progress(progress)) => {
                if !options.include_progress {
                    continue;
                }
                ChatCompletionChunk {
                    progress: Some(progress_value(progress)),
                    ..chunk(&id, created, &model)
                }
            }
            Some(GenerationEvent::Admitted { .. }) => continue,
            Some(GenerationEvent::Output { event, snapshot }) => {
                let keep_timings = options.timings_per_token || event == OutputEvent::Started;
                let timings = keep_timings
                    .then_some(snapshot)
                    .flatten()
                    .map(|snapshot| timing_values(&snapshot));
                match output_delta(event) {
                    Ok(Some(delta)) => choice_chunk(&id, created, &model, delta, None, timings),
                    Ok(None) => continue,
                    Err(error) => {
                        emit_error(&sender, ApiError::from(error).body).await;
                        return;
                    }
                }
            }
            Some(GenerationEvent::Completed(completion)) => {
                let reason = chat_finish_reason(&completion.termination);
                tracing::info!(
                    completion.id = %id,
                    model.id = %model,
                    finish.reason = reason,
                    input.tokens = completion.usage.input_tokens,
                    output.tokens = completion.usage.output_tokens,
                    prompt.ms = completion.timings.prompt_ms,
                    decode.ms = completion.timings.decode_ms,
                    "chat completion finished"
                );
                let timings = timing_values(&completion.snapshot());
                let terminal_timings = (!options.include_usage).then(|| timings.clone());
                if !emit_chunk(
                    &sender,
                    choice_chunk(
                        &id,
                        created,
                        &model,
                        ChunkDelta::default(),
                        Some(reason.into()),
                        terminal_timings,
                    ),
                )
                .await
                {
                    return;
                }
                if options.include_usage
                    && !emit_chunk(
                        &sender,
                        ChatCompletionChunk {
                            usage: Some(usage_values(&completion)),
                            timings: Some(timings),
                            ..chunk(&id, created, &model)
                        },
                    )
                    .await
                {
                    return;
                }
                let _ = sender.send(Ok(Event::default().data("[DONE]"))).await;
                return;
            }
            Some(GenerationEvent::Failed(error)) => {
                tracing::error!(error = %error, "chat completion failed");
                emit_error(&sender, ApiError::from(error).body).await;
                return;
            }
            None => {
                emit_error(&sender, ApiError::from(ended_without_outcome()).body).await;
                return;
            }
        };
        if !emit_chunk(&sender, chunk).await {
            return;
        }
    }
}

/// Input tokens a Chat Completions request occupies.
#[derive(Debug, Serialize, ToSchema)]
pub struct CountResponse {
    pub prompt_tokens: u64,
}

/// Count a Chat Completions request's input tokens with the same rendering
/// and tokenization as generation. Host-only: the model is never bound.
pub async fn count_chat_tokens(
    State(state): State<Serving>,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Json<CountResponse>, ApiError> {
    let Json(request) = payload.map_err(|error| ApiError::invalid(error.body_text()))?;
    let adapted = adapt_request(request)?;
    let host = state
        .models
        .host(&adapted.model)
        .await
        .map_err(ApiError::from)?;
    let input = adapted.request.input;
    let prompt_tokens = tokio::task::spawn_blocking(move || host.count(&input))
        .await
        .map_err(|error| ApiError::server(format!("token-count task failed: {error}")))?
        .map_err(ApiError::from)?;
    Ok(Json(CountResponse { prompt_tokens }))
}

#[utoipa::path(post, path = "/api/v1/chat/templates/apply", operation_id = "applyChatTemplate", tag = "chat",
    request_body = ApplyTemplateRequest,
    responses(
        (status = 200, description = "Prepared native chat prompt and constraints", body = ApplyTemplateResponse),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 404, description = "Model is not installed", body = ErrorResponse),
        (status = 409, description = "Installed model cannot be resolved for serving", body = ErrorResponse),
        (status = 422, description = "Installed model failed integrity validation", body = ErrorResponse),
        (status = 500, description = "Template preparation failed", body = ErrorResponse)
    )
)]
#[tracing::instrument(
    name = "serving.apply_template",
    skip_all,
    fields(model.id = tracing::field::Empty),
    err(Debug)
)]
pub async fn apply_template(
    State(state): State<Serving>,
    payload: Result<Json<ApplyTemplateRequest>, JsonRejection>,
) -> Result<Json<ApplyTemplateResponse>, ApiError> {
    let Json(request) = payload.map_err(|error| ApiError::invalid(error.body_text()))?;
    let model = request
        .model
        .clone()
        .filter(|model| !model.is_empty())
        .ok_or_else(|| ApiError::invalid("model is required"))?;
    tracing::Span::current().record("model.id", model.as_str());
    let input = validate_apply_template_request(request)?;
    let host = state.models.host(&model).await.map_err(ApiError::from)?;
    let applied = tokio::task::spawn_blocking(move || host.apply_template(&input))
        .await
        .map_err(|error| ApiError::server(format!("template task failed: {error}")))?
        .map_err(ApiError::from)?;
    Ok(Json(apply_template_response(applied)))
}
