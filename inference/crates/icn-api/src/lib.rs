use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use icn_core::{
    AllowedToolsMode, CacheType, ChatContent, ChatContentPart, ChatMessage, ChatRequest, ChatRole,
    ChatTemplateRequest, CompletionBackend, ExecutionConfig, ExecutionConfigReport, FinishReason,
    FlashAttention, Generation, GenerationMetrics, GpuLayers, GrammarTrigger, ImageInput,
    InferenceError, InferenceEvent, ModelModalities, ModelProperties, PreparedChatInfo,
    ReasoningControl, ResponseFormat, SplitMode, TemplateCapabilities, ToolCall, ToolChoice,
    ToolDefinition,
};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use utoipa::openapi::extensions::Extensions;
use utoipa::openapi::path::Operation;
use utoipa::openapi::{Components, OpenApi as OpenApiDocument, RefOr};
use utoipa::{OpenApi, PartialSchema, ToSchema};

mod media;

const DEFAULT_MAX_TOKENS: u32 = 256;
const DEFAULT_TEMPERATURE: f32 = 0.8;
const DEFAULT_TOP_P: f32 = 0.95;
const DEFAULT_SEED: u32 = 42;
const STREAM_EXTENSION: &str = "x-magnitude-stream";

#[derive(Clone)]
pub struct AppState {
    backend: Arc<dyn CompletionBackend>,
    next_id: Arc<AtomicU64>,
}

impl AppState {
    pub fn new(backend: impl CompletionBackend) -> Self {
        Self {
            backend: Arc::new(backend),
            next_id: Arc::new(AtomicU64::new(1)),
        }
    }
}

pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/props", get(props))
        .route("/v1/props", get(props))
        .route("/apply-template", post(apply_template))
        .route("/v1/apply-template", post(apply_template))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HealthResponse {
    status: &'static str,
    version: &'static str,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ModelList {
    object: &'static str,
    data: Vec<Model>,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct Model {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct PropsResponse {
    pub build_info: String,
    pub model_path: String,
    pub model_size_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub general_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub general_architecture: Option<String>,
    pub default_generation_settings: DefaultGenerationSettings,
    pub modalities: Modalities,
    pub execution: ExecutionConfigResponse,
    pub chat_template: String,
    pub template_fingerprint: String,
    pub template_capabilities: TemplateCapabilitiesResponse,
    pub training_context_tokens: u32,
    pub sliding_window_tokens: i32,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ExecutionConfigResponse {
    pub requested: ExecutionSettingsResponse,
    /// Concrete parameters passed to libllama after common/fit and thread selection.
    pub resolved: ExecutionSettingsResponse,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ExecutionSettingsResponse {
    pub gpu_layers: GpuLayersResponse,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub split_mode: SplitModeResponse,
    pub tensor_split: Option<Vec<f32>>,
    pub cache_type_k: CacheTypeResponse,
    pub cache_type_v: CacheTypeResponse,
    pub offload_kqv: bool,
    pub operation_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    pub threads: Option<u32>,
    pub threads_batch: Option<u32>,
    pub flash_attention: FlashAttentionResponse,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum GpuLayersResponse {
    Auto,
    All,
    Count { value: u32 },
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum SplitModeResponse {
    None,
    Layer,
    Row,
    Tensor,
}

#[allow(non_camel_case_types)]
#[derive(Debug, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum CacheTypeResponse {
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

#[derive(Debug, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum FlashAttentionResponse {
    Auto,
    Disabled,
    Enabled,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct DefaultGenerationSettings {
    pub n_ctx: u32,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct Modalities {
    pub vision: bool,
    pub audio: bool,
    pub video: bool,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct TemplateCapabilitiesResponse {
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
#[serde(deny_unknown_fields)]
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
    pub reasoning_effort: Option<ReasoningEffortRequest>,
    #[schema(nullable = false)]
    pub thinking_budget_tokens: Option<u32>,
    #[schema(nullable = false)]
    pub response_format: Option<ResponseFormatRequest>,
    #[schema(nullable = false)]
    pub chat_template_kwargs: Option<BTreeMap<String, JsonValue>>,
    #[schema(nullable = false)]
    pub stop: Option<StopRequest>,
    pub stream: bool,
    #[schema(nullable = false)]
    pub stream_options: Option<StreamOptions>,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(tag = "role", rename_all = "lowercase", deny_unknown_fields)]
pub enum ChatMessageRequest {
    System {
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
    },
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum ChatContentRequest {
    Text(String),
    Parts(Vec<ChatContentPartRequest>),
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ChatContentPartRequest {
    Text { text: String },
    ImageUrl { image_url: ImageUrlRequest },
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ImageUrlRequest {
    pub url: String,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatToolCallRequest {
    pub id: String,
    pub r#type: FunctionType,
    pub function: NamedFunctionCallRequest,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct NamedFunctionCallRequest {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatToolRequest {
    pub r#type: FunctionType,
    pub function: FunctionDefinitionRequest,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct FunctionDefinitionRequest {
    pub name: String,
    #[schema(nullable = false)]
    pub description: Option<String>,
    pub parameters: JsonValue,
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
#[serde(deny_unknown_fields)]
pub struct FunctionToolChoiceRequest {
    pub r#type: FunctionType,
    pub function: FunctionNameRequest,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct FunctionNameRequest {
    pub name: String,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct AllowedToolsChoiceRequest {
    pub r#type: AllowedToolsType,
    pub allowed_tools: AllowedToolsRequest,
}

#[derive(Debug, Clone, Copy, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum AllowedToolsType {
    AllowedTools,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub struct AllowedToolRequest {
    pub r#type: FunctionType,
    pub function: FunctionNameRequest,
}

#[derive(Debug, Clone, Copy, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffortRequest {
    None,
    Low,
    Medium,
    High,
}

impl ReasoningEffortRequest {
    const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ResponseFormatRequest {
    Text,
    JsonObject,
    Grammar { grammar: String },
    JsonSchema { json_schema: JsonSchemaRequest },
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub timings: Option<Timings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub error: Option<ApiErrorBody>,
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
    #[schema(nullable = false)]
    pub content: Option<String>,
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
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct Timings {
    pub prompt_n: u64,
    pub prompt_ms: f64,
    pub prompt_per_second: f64,
    pub predicted_n: u64,
    pub predicted_ms: f64,
    pub predicted_per_second: f64,
    pub queue_ms: f64,
    pub time_to_first_token_ms: f64,
    pub sampler_ms: f64,
    pub parser_ms: f64,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ErrorResponse {
    pub error: ApiErrorBody,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ApiErrorBody {
    pub message: String,
    pub r#type: &'static str,
    pub code: &'static str,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    body: ErrorResponse,
}

impl ApiError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            body: ErrorResponse {
                error: ApiErrorBody {
                    message: message.into(),
                    r#type: "invalid_request_error",
                    code: "invalid_request",
                },
            },
        }
    }

    fn server(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: ErrorResponse {
                error: ApiErrorBody {
                    message: message.into(),
                    r#type: "server_error",
                    code: "backend_error",
                },
            },
        }
    }

    fn from_inference(error: InferenceError) -> Self {
        match error {
            InferenceError::InvalidConfig(message) => Self::invalid(message),
            error => Self::server(error.to_string()),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body)).into_response()
    }
}

#[utoipa::path(get, path = "/health", operation_id = "health", tag = "system", responses(
    (status = 200, description = "ICN is running", body = HealthResponse)
))]
async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
    })
}

#[utoipa::path(get, path = "/v1/models", operation_id = "listModels", tag = "models", responses(
    (status = 200, description = "Loaded models", body = ModelList)
))]
async fn models(State(state): State<AppState>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![Model {
            id: state.backend.model_id().to_owned(),
            object: "model",
            owned_by: "magnitude",
        }],
    })
}

#[utoipa::path(get, path = "/v1/props", operation_id = "getModelProperties", tag = "models", responses(
    (status = 200, description = "Loaded model and active template properties", body = PropsResponse),
    (status = 500, description = "Properties unavailable", body = ErrorResponse)
))]
async fn props(State(state): State<AppState>) -> Result<Json<PropsResponse>, ApiError> {
    let properties = state
        .backend
        .properties()
        .map_err(ApiError::from_inference)?;
    Ok(Json(props_response(properties)))
}

#[utoipa::path(post, path = "/v1/apply-template", operation_id = "applyChatTemplate", tag = "chat",
    request_body = ApplyTemplateRequest,
    responses(
        (status = 200, description = "Prepared llama.cpp chat prompt and constraints", body = ApplyTemplateResponse),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 500, description = "Template preparation failed", body = ErrorResponse)
    )
)]
async fn apply_template(
    State(state): State<AppState>,
    Json(request): Json<ApplyTemplateRequest>,
) -> Result<Json<ApplyTemplateResponse>, ApiError> {
    validate_model_selection(request.model.as_deref(), state.backend.model_id())?;
    let request = validate_apply_template_request(request)?;
    let backend = Arc::clone(&state.backend);
    let prepared = tokio::task::spawn_blocking(move || backend.apply_template(request))
        .await
        .map_err(|error| ApiError::server(format!("template task failed: {error}")))?
        .map_err(ApiError::from_inference)?;
    Ok(Json(apply_template_response(prepared)))
}

#[utoipa::path(post, path = "/v1/chat/completions", operation_id = "createChatCompletion", tag = "chat",
    request_body = ChatCompletionRequest,
    responses(
        (status = 200, description = "OpenAI-compatible server-sent events", body = String, content_type = "text/event-stream"),
        (status = 400, description = "Invalid request", body = ErrorResponse)
    )
)]
async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    validate_model_selection(request.model.as_deref(), state.backend.model_id())?;
    let (request, include_usage) = validate_request(request)?;
    let id = format!(
        "chatcmpl-icn-{}",
        state.next_id.fetch_add(1, Ordering::Relaxed)
    );
    let created = unix_timestamp();
    let model = state.backend.model_id().to_owned();
    let backend = Arc::clone(&state.backend);
    let (sender, receiver) = mpsc::channel::<Result<Event, Infallible>>(16);

    tokio::task::spawn_blocking(move || {
        if !emit_chunk(
            &sender,
            &choice_chunk(
                &id,
                created,
                &model,
                ChunkDelta {
                    role: Some("assistant".into()),
                    ..ChunkDelta::default()
                },
                None,
                None,
            ),
        ) {
            return;
        }
        let mut callback = |event: InferenceEvent| {
            let delta = inference_event_delta(event)?;
            if emit_chunk(
                &sender,
                &choice_chunk(&id, created, &model, delta, None, None),
            ) {
                Ok(())
            } else {
                Err(InferenceError::Callback(
                    "stream consumer disconnected".into(),
                ))
            }
        };
        let generation = match backend.complete(request, &mut callback) {
            Ok(generation) => generation,
            Err(error) => {
                if emit_chunk(
                    &sender,
                    &error_chunk(&id, created, &model, inference_error_body(&error)),
                ) {
                    emit_done(&sender);
                }
                return;
            }
        };
        let reason = match generation.finish_reason {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
        };
        let terminal_timings = (!include_usage).then(|| timings(&generation));
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
        if include_usage && !emit_chunk(&sender, &usage_chunk(&id, created, &model, &generation)) {
            return;
        }
        emit_done(&sender);
    });
    Ok(Sse::new(ReceiverStream::new(receiver))
        .keep_alive(KeepAlive::default())
        .into_response())
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
        id: id.into(),
        object: "chat.completion.chunk",
        created,
        model: model.into(),
        choices: vec![ChunkChoice {
            index: 0,
            delta,
            finish_reason,
        }],
        usage: None,
        timings,
        error: None,
    }
}

fn usage_chunk(
    id: &str,
    created: u64,
    model: &str,
    generation: &Generation,
) -> ChatCompletionChunk {
    let prompt_tokens = generation.prompt_tokens as u64;
    let completion_tokens = generation.generated_tokens as u64;
    ChatCompletionChunk {
        id: id.into(),
        object: "chat.completion.chunk",
        created,
        model: model.into(),
        choices: Vec::new(),
        usage: Some(Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
        }),
        timings: Some(timings(generation)),
        error: None,
    }
}

fn error_chunk(id: &str, created: u64, model: &str, error: ApiErrorBody) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.into(),
        object: "chat.completion.chunk",
        created,
        model: model.into(),
        choices: Vec::new(),
        usage: None,
        timings: None,
        error: Some(error),
    }
}

fn timings(generation: &Generation) -> Timings {
    Timings {
        prompt_n: generation.prompt_tokens as u64,
        prompt_ms: generation.metrics.prompt_ms,
        prompt_per_second: generation.metrics.prompt_tokens_per_second,
        predicted_n: generation.generated_tokens as u64,
        predicted_ms: generation.metrics.decode_ms,
        predicted_per_second: generation.metrics.decode_tokens_per_second,
        queue_ms: generation.metrics.queue_ms,
        time_to_first_token_ms: generation.metrics.time_to_first_token_ms,
        sampler_ms: generation.metrics.sampler_ms,
        parser_ms: generation.metrics.parser_ms,
    }
}

fn inference_event_delta(event: InferenceEvent) -> Result<ChunkDelta, InferenceError> {
    Ok(match event {
        InferenceEvent::ContentDelta { text } => ChunkDelta {
            content: Some(text),
            ..ChunkDelta::default()
        },
        InferenceEvent::ReasoningDelta { text } => ChunkDelta {
            reasoning_content: Some(text),
            ..ChunkDelta::default()
        },
        InferenceEvent::ToolCallDelta {
            index,
            id,
            name,
            arguments,
        } => {
            let index = u32::try_from(index).map_err(|_| {
                InferenceError::Callback("tool-call index exceeds the HTTP protocol range".into())
            })?;
            let has_function = name.is_some() || !arguments.is_empty();
            ChunkDelta {
                tool_calls: Some(vec![ChunkToolCall {
                    index,
                    r#type: id.as_ref().map(|_| "function"),
                    id,
                    function: has_function.then_some(ChunkFunctionDelta {
                        name,
                        arguments: (!arguments.is_empty()).then_some(arguments),
                    }),
                }]),
                ..ChunkDelta::default()
            }
        }
    })
}

fn emit_chunk(
    sender: &mpsc::Sender<Result<Event, Infallible>>,
    chunk: &ChatCompletionChunk,
) -> bool {
    serde_json::to_string(chunk)
        .ok()
        .and_then(|data| sender.blocking_send(Ok(Event::default().data(data))).ok())
        .is_some()
}

fn emit_done(sender: &mpsc::Sender<Result<Event, Infallible>>) {
    let _ = sender.blocking_send(Ok(Event::default().data("[DONE]")));
}

fn inference_error_body(error: &InferenceError) -> ApiErrorBody {
    let (error_type, code) = match error {
        InferenceError::InvalidConfig(_) => ("invalid_request_error", "invalid_request"),
        InferenceError::Backend(_) => ("server_error", "backend_error"),
        InferenceError::Cancelled => ("cancelled", "request_cancelled"),
        InferenceError::Overloaded => ("server_error", "overloaded"),
        InferenceError::ExecutorStopped => ("server_error", "executor_stopped"),
        InferenceError::Callback(_) => ("server_error", "stream_callback_error"),
    };
    ApiErrorBody {
        message: error.to_string(),
        r#type: error_type,
        code,
    }
}

fn validate_model_selection(requested: Option<&str>, loaded: &str) -> Result<(), ApiError> {
    match requested {
        Some("") => Err(ApiError::invalid("model must not be empty")),
        Some(requested) if requested != loaded => Err(ApiError::invalid(format!(
            "model {requested} is not loaded by this inference node"
        ))),
        _ => Ok(()),
    }
}

fn validate_apply_template_request(
    request: ApplyTemplateRequest,
) -> Result<ChatTemplateRequest, ApiError> {
    let (request, _) = validate_request(ChatCompletionRequest {
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
        reasoning_effort: None,
        thinking_budget_tokens: None,
        response_format: request.response_format,
        chat_template_kwargs: request.chat_template_kwargs,
        stop: None,
        stream: true,
        stream_options: None,
    })?;
    Ok(request.template)
}

fn props_response(properties: ModelProperties) -> PropsResponse {
    PropsResponse {
        build_info: format!("magnitude-icn {}", env!("CARGO_PKG_VERSION")),
        model_path: properties.model_path.display().to_string(),
        model_size_bytes: properties.model_size_bytes,
        general_name: properties.name,
        general_architecture: properties.architecture,
        default_generation_settings: DefaultGenerationSettings {
            n_ctx: properties.context_tokens,
        },
        modalities: Modalities {
            vision: properties.modalities.vision,
            audio: properties.modalities.audio,
            video: properties.modalities.video,
        },
        execution: execution_config_response(properties.execution),
        chat_template: properties.chat_template,
        template_fingerprint: properties.template_fingerprint,
        template_capabilities: TemplateCapabilitiesResponse {
            string_content: properties.capabilities.string_content,
            typed_content: properties.capabilities.typed_content,
            tools: properties.capabilities.tools,
            tool_calls: properties.capabilities.tool_calls,
            parallel_tool_calls: properties.capabilities.parallel_tool_calls,
            system_role: properties.capabilities.system_role,
            preserve_reasoning: properties.capabilities.preserve_reasoning,
            object_arguments: properties.capabilities.object_arguments,
            enable_thinking: properties.capabilities.enable_thinking,
        },
        training_context_tokens: properties.training_context_tokens,
        sliding_window_tokens: properties.sliding_window_tokens,
    }
}

fn execution_config_response(report: ExecutionConfigReport) -> ExecutionConfigResponse {
    ExecutionConfigResponse {
        requested: execution_settings_response(report.requested),
        resolved: execution_settings_response(report.resolved),
    }
}

fn execution_settings_response(config: ExecutionConfig) -> ExecutionSettingsResponse {
    ExecutionSettingsResponse {
        gpu_layers: match config.gpu_layers {
            GpuLayers::Auto => GpuLayersResponse::Auto,
            GpuLayers::All => GpuLayersResponse::All,
            GpuLayers::Count(value) => GpuLayersResponse::Count { value },
        },
        use_mmap: config.use_mmap,
        use_mlock: config.use_mlock,
        split_mode: match config.split_mode {
            SplitMode::None => SplitModeResponse::None,
            SplitMode::Layer => SplitModeResponse::Layer,
            SplitMode::Row => SplitModeResponse::Row,
            SplitMode::Tensor => SplitModeResponse::Tensor,
        },
        tensor_split: config.tensor_split,
        cache_type_k: cache_type_response(config.cache_type_k),
        cache_type_v: cache_type_response(config.cache_type_v),
        offload_kqv: config.offload_kqv,
        operation_offload: config.operation_offload,
        swa_full: config.swa_full,
        kv_unified: config.kv_unified,
        threads: config.threads.map(NonZeroU32::get),
        threads_batch: config.threads_batch.map(NonZeroU32::get),
        flash_attention: match config.flash_attention {
            FlashAttention::Auto => FlashAttentionResponse::Auto,
            FlashAttention::Disabled => FlashAttentionResponse::Disabled,
            FlashAttention::Enabled => FlashAttentionResponse::Enabled,
        },
    }
}

fn cache_type_response(cache_type: CacheType) -> CacheTypeResponse {
    match cache_type {
        CacheType::F32 => CacheTypeResponse::F32,
        CacheType::F16 => CacheTypeResponse::F16,
        CacheType::Bf16 => CacheTypeResponse::Bf16,
        CacheType::Q8_0 => CacheTypeResponse::Q8_0,
        CacheType::Q4_0 => CacheTypeResponse::Q4_0,
        CacheType::Q4_1 => CacheTypeResponse::Q4_1,
        CacheType::Iq4Nl => CacheTypeResponse::Iq4Nl,
        CacheType::Q5_0 => CacheTypeResponse::Q5_0,
        CacheType::Q5_1 => CacheTypeResponse::Q5_1,
    }
}

fn apply_template_response(prepared: PreparedChatInfo) -> ApplyTemplateResponse {
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

fn validate_request(request: ChatCompletionRequest) -> Result<(ChatRequest, bool), ApiError> {
    if !request.stream {
        return Err(ApiError::invalid(
            "ICN's MVP chat endpoint requires stream: true",
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
        .unwrap_or(DEFAULT_MAX_TOKENS);
    if max_tokens == 0 {
        return Err(ApiError::invalid("max tokens must be greater than zero"));
    }
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
    let messages = request
        .messages
        .into_iter()
        .map(chat_message)
        .collect::<Result<Vec<_>, _>>()?;
    let (tools, tool_names) = tools(request.tools.unwrap_or_default())?;
    let tool_choice = tool_choice(request.tool_choice, &tool_names)?;
    let mut template_args = request.chat_template_kwargs.unwrap_or_default();
    if template_args.keys().any(String::is_empty) {
        return Err(ApiError::invalid(
            "chat_template_kwargs keys must not be empty",
        ));
    }
    let reasoning = reasoning_control(
        request.reasoning_effort,
        request.thinking_budget_tokens,
        &mut template_args,
    )?;
    let response_format = response_format(request.response_format)?;
    let stop = stops(request.stop)?;
    Ok((
        ChatRequest {
            template: ChatTemplateRequest {
                messages,
                tools,
                tool_choice,
                parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
                reasoning,
                response_format,
                template_args,
            },
            stop,
            max_tokens,
            temperature,
            top_p,
            seed: request.seed.unwrap_or(DEFAULT_SEED),
        },
        request
            .stream_options
            .and_then(|options| options.include_usage)
            .unwrap_or(false),
    ))
}

fn chat_message(message: ChatMessageRequest) -> Result<ChatMessage, ApiError> {
    match message {
        ChatMessageRequest::System { content } => Ok(ChatMessage::text(ChatRole::System, content)),
        ChatMessageRequest::User { content } => Ok(ChatMessage {
            role: ChatRole::User,
            content: Some(chat_content(content)?),
            reasoning: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        }),
        ChatMessageRequest::Assistant {
            content,
            reasoning_content,
            tool_calls,
        } => {
            if content.is_none() && reasoning_content.is_none() && tool_calls.is_empty() {
                return Err(ApiError::invalid(
                    "assistant messages require content, reasoning_content, or tool_calls",
                ));
            }
            let mut ids = BTreeSet::new();
            let tool_calls = tool_calls
                .into_iter()
                .map(|tool_call| {
                    let ChatToolCallRequest {
                        id,
                        r#type: _,
                        function,
                    } = tool_call;
                    require_non_empty(&id, "assistant tool-call id")?;
                    require_non_empty(&function.name, "assistant tool-call function name")?;
                    if !ids.insert(id.clone()) {
                        return Err(ApiError::invalid(format!(
                            "duplicate assistant tool-call id: {id}"
                        )));
                    }
                    Ok(ToolCall {
                        id,
                        name: function.name,
                        arguments: function.arguments,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ChatMessage {
                role: ChatRole::Assistant,
                content: content.map(ChatContent::Text),
                reasoning: reasoning_content,
                tool_calls,
                tool_call_id: None,
            })
        }
        ChatMessageRequest::Tool {
            tool_call_id,
            content,
        } => {
            require_non_empty(&tool_call_id, "tool_call_id")?;
            Ok(ChatMessage {
                role: ChatRole::Tool,
                content: Some(chat_content(content)?),
                reasoning: None,
                tool_calls: Vec::new(),
                tool_call_id: Some(tool_call_id),
            })
        }
    }
}

fn chat_content(content: ChatContentRequest) -> Result<ChatContent, ApiError> {
    match content {
        ChatContentRequest::Text(text) => Ok(ChatContent::Text(text)),
        ChatContentRequest::Parts(parts) => {
            if parts.is_empty() {
                return Err(ApiError::invalid("message content parts must not be empty"));
            }
            Ok(ChatContent::Parts(
                parts
                    .into_iter()
                    .map(|part| match part {
                        ChatContentPartRequest::Text { text } => Ok(ChatContentPart::Text { text }),
                        ChatContentPartRequest::ImageUrl { image_url } => {
                            require_non_empty(&image_url.url, "image_url.url")?;
                            let image = media::decode_image_data_url(
                                &image_url.url,
                                media::MAX_HTTP_IMAGE_BYTES,
                            )
                            .map_err(|error| ApiError::invalid(error.to_string()))?;
                            Ok(ChatContentPart::Image(ImageInput::new(
                                image.media_type,
                                image.bytes,
                            )))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ))
        }
    }
}

fn tools(
    requests: Vec<ChatToolRequest>,
) -> Result<(Vec<ToolDefinition>, BTreeSet<String>), ApiError> {
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
            require_json_schema(&function.parameters, "tool function parameters")?;
            Ok(ToolDefinition {
                name: function.name,
                description: function.description,
                parameters: function.parameters,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((tools, names))
}

fn tool_choice(
    request: Option<ToolChoiceRequest>,
    tool_names: &BTreeSet<String>,
) -> Result<ToolChoice, ApiError> {
    let choice = match request {
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
            ToolChoice::Function {
                name: request.function.name,
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
                    Ok(tool.function.name)
                })
                .collect::<Result<Vec<_>, _>>()?;
            ToolChoice::AllowedTools {
                mode: match request.allowed_tools.mode {
                    AllowedToolsModeRequest::Auto => AllowedToolsMode::Auto,
                    AllowedToolsModeRequest::Required => AllowedToolsMode::Required,
                },
                names,
            }
        }
    };
    Ok(choice)
}

fn reasoning_control(
    effort: Option<ReasoningEffortRequest>,
    budget_tokens: Option<u32>,
    template_args: &mut BTreeMap<String, JsonValue>,
) -> Result<ReasoningControl, ApiError> {
    if let Some(effort) = effort {
        let value = JsonValue::String(effort.as_str().into());
        if let Some(existing) = template_args.get("reasoning_effort") {
            if existing != &value {
                return Err(ApiError::invalid(
                    "reasoning_effort conflicts with chat_template_kwargs.reasoning_effort",
                ));
            }
        } else {
            template_args.insert("reasoning_effort".into(), value);
        }
    }

    let enable_thinking = match template_args.get("enable_thinking") {
        Some(JsonValue::Bool(enabled)) => Some(*enabled),
        Some(_) => {
            return Err(ApiError::invalid(
                "chat_template_kwargs.enable_thinking must be a boolean",
            ));
        }
        None => None,
    };
    let effort_enabled = effort.map(|effort| !matches!(effort, ReasoningEffortRequest::None));
    if let (Some(from_template), Some(from_effort)) = (enable_thinking, effort_enabled)
        && from_template != from_effort
    {
        return Err(ApiError::invalid(
            "reasoning_effort conflicts with chat_template_kwargs.enable_thinking",
        ));
    }
    let enabled = enable_thinking.or(effort_enabled);
    if budget_tokens.is_some() && enabled == Some(false) {
        return Err(ApiError::invalid(
            "thinking_budget_tokens cannot be used when reasoning is disabled",
        ));
    }
    Ok(match (enabled, budget_tokens) {
        (Some(false), None) => ReasoningControl::Disabled,
        (_, Some(budget_tokens)) => ReasoningControl::Enabled {
            budget_tokens: Some(budget_tokens),
        },
        (Some(true), None) => ReasoningControl::Enabled {
            budget_tokens: None,
        },
        (None, None) => ReasoningControl::ModelDefault,
    })
}

fn response_format(request: Option<ResponseFormatRequest>) -> Result<ResponseFormat, ApiError> {
    match request.unwrap_or(ResponseFormatRequest::Text) {
        ResponseFormatRequest::Text => Ok(ResponseFormat::Text),
        ResponseFormatRequest::JsonObject => Ok(ResponseFormat::JsonObject),
        ResponseFormatRequest::Grammar { grammar } => {
            require_non_empty(&grammar, "response_format grammar")?;
            Ok(ResponseFormat::Grammar { grammar })
        }
        ResponseFormatRequest::JsonSchema { json_schema } => {
            require_non_empty(&json_schema.name, "response_format json_schema name")?;
            require_json_schema(&json_schema.schema, "response_format json_schema schema")?;
            Ok(ResponseFormat::JsonSchema {
                name: json_schema.name,
                schema: json_schema.schema,
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

fn require_json_schema(value: &JsonValue, field: &str) -> Result<(), ApiError> {
    if value.is_object() || value.is_boolean() {
        Ok(())
    } else {
        Err(ApiError::invalid(format!(
            "{field} must be a JSON Schema object or boolean"
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

#[derive(OpenApi)]
#[openapi(
    info(title = "Magnitude Inference Control Node", version = "0.1.0"),
    paths(health, models, props, apply_template, chat_completions),
    components(schemas(
        HealthResponse,
        ModelList,
        Model,
        PropsResponse,
        ExecutionConfigResponse,
        ExecutionSettingsResponse,
        GpuLayersResponse,
        SplitModeResponse,
        CacheTypeResponse,
        FlashAttentionResponse,
        DefaultGenerationSettings,
        Modalities,
        TemplateCapabilitiesResponse,
        ApplyTemplateRequest,
        ApplyTemplateResponse,
        GrammarTriggerResponse,
        ChatCompletionRequest,
        ChatMessageRequest,
        ChatContentRequest,
        ChatContentPartRequest,
        ImageUrlRequest,
        ChatToolCallRequest,
        NamedFunctionCallRequest,
        ChatToolRequest,
        FunctionDefinitionRequest,
        FunctionType,
        ToolChoiceRequest,
        ToolChoiceModeRequest,
        FunctionToolChoiceRequest,
        FunctionNameRequest,
        AllowedToolsChoiceRequest,
        AllowedToolsType,
        AllowedToolsRequest,
        AllowedToolsModeRequest,
        AllowedToolRequest,
        ReasoningEffortRequest,
        ResponseFormatRequest,
        JsonSchemaRequest,
        StopRequest,
        StreamOptions,
        ChatCompletionChunk,
        ChunkChoice,
        ChunkDelta,
        ChunkToolCall,
        ChunkFunctionDelta,
        Usage,
        Timings,
        ErrorResponse,
        ApiErrorBody
    ))
)]
struct IcnOpenApi;

#[derive(Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
#[allow(dead_code)]
enum StreamFraming {
    Sse,
    Ndjson,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
#[allow(dead_code)]
enum StreamTermination {
    Sentinel { value: &'static str },
    Eof,
    LongLived,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
#[allow(dead_code)]
enum StreamReconnect {
    None,
    LastEventId,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct StreamData {
    encoding: &'static str,
    schema: StreamSchemaRef,
}

#[derive(Debug, Serialize)]
struct StreamSchemaRef {
    #[serde(rename = "$ref")]
    reference: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct StreamMetadata {
    version: u8,
    response_status: u16,
    framing: StreamFraming,
    data: StreamData,
    termination: StreamTermination,
    reconnect: StreamReconnect,
}

trait StreamContract {
    type Event: ToSchema;
    const RESPONSE_STATUS: u16;
    fn metadata() -> StreamMetadata;
}

struct ChatCompletionStream;

impl StreamContract for ChatCompletionStream {
    type Event = ChatCompletionChunk;
    const RESPONSE_STATUS: u16 = 200;
    fn metadata() -> StreamMetadata {
        StreamMetadata {
            version: 1,
            response_status: Self::RESPONSE_STATUS,
            framing: StreamFraming::Sse,
            data: StreamData {
                encoding: "json",
                schema: StreamSchemaRef {
                    reference: format!("#/components/schemas/{}", Self::Event::name()),
                },
            },
            termination: StreamTermination::Sentinel { value: "[DONE]" },
            reconnect: StreamReconnect::None,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum OpenApiExportError {
    #[error("OpenAPI operation {0} was not generated")]
    MissingOperation(&'static str),
    #[error("OpenAPI response {status} for {operation} was not generated")]
    MissingResponse {
        operation: &'static str,
        status: u16,
    },
    #[error("OpenAPI response {status} for {operation} does not declare {media_type}")]
    MissingMediaType {
        operation: &'static str,
        status: u16,
        media_type: &'static str,
    },
    #[error("failed to encode stream metadata: {0}")]
    Metadata(#[from] serde_json::Error),
}

pub fn openapi() -> Result<OpenApiDocument, OpenApiExportError> {
    let mut document = IcnOpenApi::openapi();
    attach_stream_contract::<ChatCompletionStream>(
        &mut document,
        "createChatCompletion",
        "text/event-stream",
    )?;
    Ok(document)
}

fn attach_stream_contract<C: StreamContract>(
    document: &mut OpenApiDocument,
    operation_id: &'static str,
    media_type: &'static str,
) -> Result<(), OpenApiExportError> {
    let mut schemas = vec![(C::Event::name().into_owned(), C::Event::schema())];
    C::Event::schemas(&mut schemas);
    document
        .components
        .get_or_insert_with(Components::new)
        .schemas
        .extend(schemas);
    let operation = find_operation(document, operation_id)
        .ok_or(OpenApiExportError::MissingOperation(operation_id))?;
    let status = C::RESPONSE_STATUS.to_string();
    let response =
        operation
            .responses
            .responses
            .get(&status)
            .ok_or(OpenApiExportError::MissingResponse {
                operation: operation_id,
                status: C::RESPONSE_STATUS,
            })?;
    let RefOr::T(response) = response else {
        return Err(OpenApiExportError::MissingResponse {
            operation: operation_id,
            status: C::RESPONSE_STATUS,
        });
    };
    if !response.content.contains_key(media_type) {
        return Err(OpenApiExportError::MissingMediaType {
            operation: operation_id,
            status: C::RESPONSE_STATUS,
            media_type,
        });
    }
    let metadata = serde_json::to_value(C::metadata())?;
    operation
        .extensions
        .get_or_insert_with(Extensions::default)
        .insert(STREAM_EXTENSION.into(), metadata);
    Ok(())
}

fn find_operation<'a>(
    document: &'a mut OpenApiDocument,
    operation_id: &str,
) -> Option<&'a mut Operation> {
    for item in document.paths.paths.values_mut() {
        for operation in [
            &mut item.get,
            &mut item.put,
            &mut item.post,
            &mut item.delete,
            &mut item.options,
            &mut item.head,
            &mut item.patch,
            &mut item.trace,
        ] {
            if operation
                .as_ref()
                .and_then(|operation| operation.operation_id.as_deref())
                == Some(operation_id)
            {
                return operation.as_mut();
            }
        }
    }
    None
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

pub struct FakeBackend {
    model_id: String,
    response: String,
}
impl FakeBackend {
    pub fn new(model_id: impl Into<String>, response: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            response: response.into(),
        }
    }
}

impl CompletionBackend for FakeBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn properties(&self) -> Result<ModelProperties, InferenceError> {
        Ok(ModelProperties {
            model_path: "/tmp/fake.gguf".into(),
            model_size_bytes: 1,
            architecture: Some("fake".into()),
            name: Some(self.model_id.clone()),
            context_tokens: 4096,
            training_context_tokens: 4096,
            sliding_window_tokens: 0,
            chat_template: "fake-template".into(),
            capabilities: TemplateCapabilities {
                string_content: true,
                typed_content: true,
                tools: true,
                tool_calls: true,
                parallel_tool_calls: true,
                system_role: true,
                preserve_reasoning: true,
                object_arguments: true,
                enable_thinking: true,
            },
            modalities: ModelModalities::default(),
            execution: ExecutionConfigReport {
                requested: ExecutionConfig::default(),
                resolved: ExecutionConfig::default(),
            },
            template_fingerprint: "fake-fingerprint".into(),
        })
    }

    fn apply_template(
        &self,
        request: ChatTemplateRequest,
    ) -> Result<PreparedChatInfo, InferenceError> {
        Ok(PreparedChatInfo {
            prompt: request
                .messages
                .iter()
                .map(ChatMessage::text_content)
                .collect::<Vec<_>>()
                .join("\n"),
            generation_prompt: String::new(),
            grammar: String::new(),
            grammar_lazy: false,
            grammar_triggers: Vec::new(),
            preserved_tokens: Vec::new(),
            additional_stops: Vec::new(),
            supports_thinking: true,
            thinking_start_tag: Some("<think>".into()),
            thinking_end_tag: Some("</think>".into()),
            template_fingerprint: "fake-fingerprint".into(),
        })
    }
    fn complete(
        &self,
        request: ChatRequest,
        on_event: &mut dyn FnMut(InferenceEvent) -> Result<(), InferenceError>,
    ) -> Result<Generation, InferenceError> {
        for token in self.response.split_inclusive(' ') {
            on_event(InferenceEvent::ContentDelta {
                text: token.to_owned(),
            })?;
        }
        Ok(Generation {
            text: self.response.clone(),
            reasoning: String::new(),
            tool_calls: Vec::new(),
            prompt_tokens: request.template.messages.len(),
            generated_tokens: self.response.split_whitespace().count(),
            finish_reason: FinishReason::Stop,
            metrics: GenerationMetrics::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use serde_json::{Value, json};
    use tower::ServiceExt;

    fn request_from_json(value: Value) -> ChatCompletionRequest {
        serde_json::from_value(value).expect("request must decode")
    }

    fn minimal_request() -> Value {
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        })
    }

    fn stream_json(body: &str) -> Vec<Value> {
        body.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|data| *data != "[DONE]")
            .map(|data| serde_json::from_str(data).expect("SSE data must be JSON"))
            .collect()
    }

    async fn post_chat(backend: impl CompletionBackend, request: Value) -> (StatusCode, String) {
        let response = app(AppState::new(backend))
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }

    struct ScriptedBackend {
        events: Vec<InferenceEvent>,
        fail: bool,
    }

    impl CompletionBackend for ScriptedBackend {
        fn model_id(&self) -> &str {
            "test-model"
        }

        fn complete(
            &self,
            _request: ChatRequest,
            on_event: &mut dyn FnMut(InferenceEvent) -> Result<(), InferenceError>,
        ) -> Result<Generation, InferenceError> {
            for event in &self.events {
                on_event(event.clone())?;
            }
            if self.fail {
                return Err(InferenceError::Backend("scripted failure".into()));
            }
            Ok(Generation {
                text: "answer".into(),
                reasoning: "thought".into(),
                tool_calls: vec![ToolCall {
                    id: "call-1".into(),
                    name: "lookup".into(),
                    arguments: "{}".into(),
                }],
                prompt_tokens: 11,
                generated_tokens: 7,
                finish_reason: FinishReason::ToolCalls,
                metrics: GenerationMetrics {
                    queue_ms: 1.0,
                    prompt_ms: 2.0,
                    decode_ms: 3.0,
                    time_to_first_token_ms: 4.0,
                    prompt_tokens_per_second: 5.0,
                    decode_tokens_per_second: 6.0,
                    sampler_ms: 0.5,
                    parser_ms: 0.25,
                },
            })
        }
    }

    #[test]
    fn exported_chat_operation_has_explicit_stream_contract() {
        let value = serde_json::to_value(openapi().unwrap()).unwrap();
        let contract = &value["paths"]["/v1/chat/completions"]["post"][STREAM_EXTENSION];
        assert_eq!(contract["framing"], "sse");
        assert_eq!(
            contract["data"]["schema"]["$ref"],
            "#/components/schemas/ChatCompletionChunk"
        );
        assert_eq!(contract["termination"]["type"], "sentinel");
        assert_eq!(
            value["paths"]["/v1/chat/completions"]["post"]["responses"]["200"]["content"]["text/event-stream"]
                ["schema"]["type"],
            "string"
        );
        let schemas = &value["components"]["schemas"];
        assert!(schemas["ChatCompletionRequest"]["properties"]["tools"].is_object());
        assert!(schemas["ChunkDelta"]["properties"]["reasoning_content"].is_object());
        assert!(schemas["ChunkDelta"]["properties"]["tool_calls"].is_object());
        assert!(schemas["ChatCompletionChunk"]["properties"]["error"].is_object());
        assert!(schemas["ChatCompletionChunk"]["properties"]["timings"].is_object());
    }

    #[tokio::test]
    async fn exposes_typed_properties_and_template_preparation() {
        let service = app(AppState::new(FakeBackend::new("test-model", "ok")));
        let properties = service
            .clone()
            .oneshot(Request::get("/v1/props").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(properties.status(), StatusCode::OK);
        let properties: Value =
            serde_json::from_slice(&properties.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(properties["model_path"], "/tmp/fake.gguf");
        assert_eq!(properties["template_capabilities"]["enable_thinking"], true);
        assert_eq!(
            properties["execution"]["requested"]["gpu_layers"]["mode"],
            "auto"
        );
        assert_eq!(properties["execution"]["requested"]["swa_full"], false);

        let response = service
            .oneshot(
                Request::post("/v1/apply-template")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "test-model",
                            "messages": [
                                {"role": "system", "content": "system"},
                                {"role": "user", "content": "hello"}
                            ],
                            "chat_template_kwargs": {"enable_thinking": true}
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let response: Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(response["prompt"], "system\nhello");
        assert_eq!(response["template_fingerprint"], "fake-fingerprint");
    }

    #[tokio::test]
    async fn rejects_a_model_not_loaded_by_the_node() {
        let (status, body) = post_chat(
            FakeBackend::new("test-model", "ok"),
            json!({
                "model": "different-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.contains("not loaded"));
    }

    #[test]
    fn maps_the_complete_chat_request_contract() {
        let request = request_from_json(json!({
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
                ]},
                {
                    "role": "assistant",
                    "content": null,
                    "reasoning_content": "because",
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{\"q\":\"x\"}"}
                    }]
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "result"}
            ],
            "tools": [
                {"type": "function", "function": {
                    "name": "lookup",
                    "description": "Look something up",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
                }},
                {"type": "function", "function": {
                    "name": "other",
                    "parameters": true
                }}
            ],
            "tool_choice": {
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": "required",
                    "tools": [{"type": "function", "function": {"name": "lookup"}}]
                }
            },
            "parallel_tool_calls": false,
            "reasoning_effort": "high",
            "thinking_budget_tokens": 64,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": true,
                    "schema": {"type": "object", "required": ["ok"]}
                }
            },
            "chat_template_kwargs": {"enable_thinking": true, "custom": 7},
            "stop": ["END", "STOP"],
            "max_completion_tokens": 99,
            "temperature": 0.25,
            "top_p": 0.75,
            "seed": 9,
            "stream": true,
            "stream_options": {"include_usage": true}
        }));

        let (request, include_usage) = validate_request(request).unwrap();
        assert!(include_usage);
        assert_eq!(request.template.messages.len(), 4);
        assert_eq!(request.template.messages[0].role, ChatRole::System);
        assert_eq!(request.template.messages[1].role, ChatRole::User);
        assert_eq!(
            request.template.messages[1].content,
            Some(ChatContent::Parts(vec![
                ChatContentPart::Text {
                    text: "look".into()
                },
                ChatContentPart::Image(ImageInput::new("image/png", vec![0]))
            ]))
        );
        assert_eq!(
            request.template.messages[2].reasoning.as_deref(),
            Some("because")
        );
        assert_eq!(request.template.messages[2].tool_calls[0].name, "lookup");
        assert_eq!(
            request.template.messages[3].tool_call_id.as_deref(),
            Some("call-1")
        );
        assert_eq!(request.template.tools.len(), 2);
        assert_eq!(request.template.tools[0].name, "lookup");
        assert_eq!(
            request.template.tool_choice,
            ToolChoice::AllowedTools {
                mode: AllowedToolsMode::Required,
                names: vec!["lookup".into()]
            }
        );
        assert!(!request.template.parallel_tool_calls);
        assert_eq!(
            request.template.reasoning,
            ReasoningControl::Enabled {
                budget_tokens: Some(64)
            }
        );
        assert_eq!(
            request.template.template_args.get("reasoning_effort"),
            Some(&json!("high"))
        );
        assert_eq!(
            request.template.template_args.get("custom"),
            Some(&json!(7))
        );
        match request.template.response_format {
            ResponseFormat::JsonSchema {
                name,
                schema,
                strict,
            } => {
                assert_eq!(name, "answer");
                assert_eq!(schema["type"], "object");
                assert!(strict);
            }
            response => panic!("unexpected response format: {response:?}"),
        }
        assert_eq!(request.stop, ["END", "STOP"]);
        assert_eq!(request.max_tokens, 99);
        assert_eq!(request.temperature, 0.25);
        assert_eq!(request.top_p, 0.75);
        assert_eq!(request.seed, 9);
    }

    #[test]
    fn rejects_network_image_urls_before_the_executor() {
        let mut request = minimal_request();
        request["messages"] = json!([{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": "https://example.invalid/image.png"}
            }]
        }]);

        let error = validate_request(request_from_json(request)).unwrap_err();
        assert!(
            error
                .body
                .error
                .message
                .contains("network URLs are not supported")
        );
    }

    #[test]
    fn preserves_model_defaults_when_optional_controls_are_omitted() {
        let (request, include_usage) =
            validate_request(request_from_json(minimal_request())).unwrap();
        assert!(!include_usage);
        assert_eq!(request.template.tool_choice, ToolChoice::Auto);
        assert!(request.template.parallel_tool_calls);
        assert_eq!(request.template.reasoning, ReasoningControl::ModelDefault);
        assert_eq!(request.template.response_format, ResponseFormat::Text);
        assert!(request.template.template_args.is_empty());
        assert!(request.stop.is_empty());
    }

    #[test]
    fn maps_grammar_response_format() {
        let mut request = minimal_request();
        request["response_format"] = json!({
            "type": "grammar",
            "grammar": "root ::= \"yes\" | \"no\""
        });

        let (request, _) = validate_request(request_from_json(request)).unwrap();
        assert_eq!(
            request.template.response_format,
            ResponseFormat::Grammar {
                grammar: "root ::= \"yes\" | \"no\"".into()
            }
        );
    }

    #[test]
    fn rejects_conflicting_or_lossy_request_controls() {
        let mut request = minimal_request();
        request["reasoning_effort"] = json!("none");
        request["thinking_budget_tokens"] = json!(10);
        let error = validate_request(request_from_json(request)).unwrap_err();
        assert!(error.body.error.message.contains("reasoning is disabled"));

        let mut request = minimal_request();
        request["tools"] = json!([{"type": "function", "function": {
            "name": "known", "parameters": {"type": "object"}
        }}]);
        request["tool_choice"] = json!({
            "type": "function", "function": {"name": "missing"}
        });
        let error = validate_request(request_from_json(request)).unwrap_err();
        assert!(error.body.error.message.contains("undefined tool"));

        let mut request = minimal_request();
        request["response_format"] = json!({
            "type": "json_schema",
            "json_schema": {"name": "bad", "schema": 42}
        });
        let error = validate_request(request_from_json(request)).unwrap_err();
        assert!(error.body.error.message.contains("JSON Schema"));

        let mut request = minimal_request();
        request["response_format"] = json!({"type": "grammar", "grammar": ""});
        let error = validate_request(request_from_json(request)).unwrap_err();
        assert!(
            error
                .body
                .error
                .message
                .contains("grammar must not be empty")
        );

        let mut request = minimal_request();
        request["stop"] = json!(["END", "END"]);
        let error = validate_request(request_from_json(request)).unwrap_err();
        assert!(error.body.error.message.contains("duplicate stop"));
    }

    #[tokio::test]
    async fn fake_backend_serves_openai_compatible_sse() {
        let response = app(AppState::new(FakeBackend::new("test-model", "hello world")))
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get("content-type").unwrap(),
            "text/event-stream"
        );
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("chat.completion.chunk"));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn streams_reasoning_content_tool_calls_finish_usage_and_timings() {
        let backend = ScriptedBackend {
            events: vec![
                InferenceEvent::ReasoningDelta {
                    text: "thought".into(),
                },
                InferenceEvent::ContentDelta {
                    text: "answer".into(),
                },
                InferenceEvent::ToolCallDelta {
                    index: 0,
                    id: Some("call-1".into()),
                    name: Some("lookup".into()),
                    arguments: "{".into(),
                },
                InferenceEvent::ToolCallDelta {
                    index: 0,
                    id: None,
                    name: None,
                    arguments: "}".into(),
                },
            ],
            fail: false,
        };
        let mut request = minimal_request();
        request["stream_options"] = json!({"include_usage": true});
        let (status, body) = post_chat(backend, request).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("data: [DONE]"));
        let chunks = stream_json(&body);
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(
            chunks[1]["choices"][0]["delta"]["reasoning_content"],
            "thought"
        );
        assert_eq!(chunks[2]["choices"][0]["delta"]["content"], "answer");
        assert_eq!(
            chunks[3]["choices"][0]["delta"]["tool_calls"][0]["id"],
            "call-1"
        );
        assert_eq!(
            chunks[3]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "lookup"
        );
        assert_eq!(chunks[5]["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(chunks[6]["choices"], json!([]));
        assert_eq!(chunks[6]["usage"]["prompt_tokens"], 11);
        assert_eq!(chunks[6]["usage"]["completion_tokens"], 7);
        assert_eq!(chunks[6]["usage"]["total_tokens"], 18);
        assert_eq!(chunks[6]["timings"]["prompt_ms"], 2.0);
        assert_eq!(chunks[6]["timings"]["predicted_per_second"], 6.0);
    }

    #[tokio::test]
    async fn backend_failure_is_an_explicit_stream_error_followed_by_done() {
        let backend = ScriptedBackend {
            events: vec![InferenceEvent::ContentDelta {
                text: "partial".into(),
            }],
            fail: true,
        };
        let (status, body) = post_chat(backend, minimal_request()).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("data: [DONE]"));
        let chunks = stream_json(&body);
        let error = chunks.last().unwrap();
        assert_eq!(error["choices"], json!([]));
        assert_eq!(error["error"]["type"], "server_error");
        assert_eq!(error["error"]["code"], "backend_error");
        assert!(
            error["error"]["message"]
                .as_str()
                .unwrap()
                .contains("scripted failure")
        );
    }
}
