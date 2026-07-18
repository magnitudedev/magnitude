use std::convert::Infallible;
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
    ChatMessage, ChatRequest, CompletionBackend, FinishReason, Generation, InferenceError,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use utoipa::openapi::extensions::Extensions;
use utoipa::openapi::path::Operation;
use utoipa::openapi::{Components, OpenApi as OpenApiDocument, RefOr};
use utoipa::{OpenApi, PartialSchema, ToSchema};

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
    pub stream: bool,
    #[schema(nullable = false)]
    pub stream_options: Option<StreamOptions>,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatMessageRequest {
    pub role: String,
    pub content: String,
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
        let emit = |sender: &mpsc::Sender<Result<Event, Infallible>>,
                    chunk: &ChatCompletionChunk| {
            let Ok(data) = serde_json::to_string(chunk) else {
                return false;
            };
            sender
                .blocking_send(Ok(Event::default().data(data)))
                .is_ok()
        };
        if !emit(
            &sender,
            &chunk(
                &id,
                created,
                &model,
                ChunkDelta {
                    role: Some("assistant".into()),
                    content: None,
                },
                None,
                None,
            ),
        ) {
            return;
        }
        let mut callback = |piece: &str| {
            if emit(
                &sender,
                &chunk(
                    &id,
                    created,
                    &model,
                    ChunkDelta {
                        role: None,
                        content: Some(piece.to_owned()),
                    },
                    None,
                    None,
                ),
            ) {
                Ok(())
            } else {
                Err(InferenceError::Callback(
                    "stream consumer disconnected".into(),
                ))
            }
        };
        let Ok(generation) = backend.complete(request, &mut callback) else {
            return;
        };
        let reason = match generation.finish_reason {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        };
        if !emit(
            &sender,
            &chunk(
                &id,
                created,
                &model,
                ChunkDelta::default(),
                Some(reason.into()),
                None,
            ),
        ) {
            return;
        }
        if include_usage {
            let usage = Usage {
                prompt_tokens: generation.prompt_tokens as u64,
                completion_tokens: generation.generated_tokens as u64,
                total_tokens: (generation.prompt_tokens + generation.generated_tokens) as u64,
            };
            if !emit(
                &sender,
                &chunk(
                    &id,
                    created,
                    &model,
                    ChunkDelta::default(),
                    None,
                    Some(usage),
                ),
            ) {
                return;
            }
        }
        let _ = sender.blocking_send(Ok(Event::default().data("[DONE]")));
    });
    Ok(Sse::new(ReceiverStream::new(receiver))
        .keep_alive(KeepAlive::default())
        .into_response())
}

fn chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: ChunkDelta,
    finish_reason: Option<String>,
    usage: Option<Usage>,
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
        usage,
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
        .map(|message| {
            if !matches!(message.role.as_str(), "system" | "user" | "assistant") {
                return Err(ApiError::invalid(format!(
                    "unsupported message role: {}",
                    message.role
                )));
            }
            Ok(ChatMessage {
                role: message.role,
                content: message.content,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((
        ChatRequest {
            messages,
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

#[derive(OpenApi)]
#[openapi(
    info(title = "Magnitude Inference Control Node", version = "0.1.0"),
    paths(health, models, chat_completions),
    components(schemas(
        HealthResponse,
        ModelList,
        Model,
        ChatCompletionRequest,
        ChatMessageRequest,
        StreamOptions,
        ChatCompletionChunk,
        ChunkChoice,
        ChunkDelta,
        Usage,
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
    fn complete(
        &self,
        request: ChatRequest,
        on_token: &mut dyn FnMut(&str) -> Result<(), InferenceError>,
    ) -> Result<Generation, InferenceError> {
        for token in self.response.split_inclusive(' ') {
            on_token(token)?;
        }
        Ok(Generation {
            text: self.response.clone(),
            prompt_tokens: request.messages.len(),
            generated_tokens: self.response.split_whitespace().count(),
            finish_reason: FinishReason::Stop,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;
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
}
