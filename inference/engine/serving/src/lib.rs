//! The one implementation of the inference protocols: OpenAI Chat
//! Completions, OpenAI Responses (HTTP, SSE and WebSocket) and Anthropic
//! Messages with `count_tokens` (integration spec §8.2). Handlers are
//! router-independent and reach models only through [`ServedModels`]; the
//! service and the standalone engine each mount [`router`] with their own
//! source.
pub mod anthropic;
pub mod chat;
pub mod engine;
pub mod error;
mod media;
pub mod properties;
pub mod responses;
pub mod source;
#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::http::HeaderMap;
use axum::response::Response;
use axum::routing::{get, post};

pub use error::{ApiError, ApiErrorBody, ErrorResponse, ModelUnavailable, ServingError};
pub use media::MAX_HTTP_BODY_BYTES;
pub use source::{
    GenerationEvent, GenerationStream, HostChat, LoadProgress, ModelInvocation, ModelLoadProgress,
    ModelLoadStage, ServedModels,
};

/// Protocol handler state: the model source and response identities.
#[derive(Clone)]
pub struct Serving {
    models: Arc<dyn ServedModels>,
    next_id: Arc<AtomicU64>,
}

impl Serving {
    pub fn new(models: Arc<dyn ServedModels>) -> Self {
        Self {
            models,
            next_id: Arc::new(AtomicU64::new(1)),
        }
    }

    pub(crate) fn next_id(&self, prefix: &str) -> String {
        format!("{prefix}{}", self.next_id.fetch_add(1, Ordering::Relaxed))
    }
}

/// Every protocol route at its published path.
pub fn router(serving: Serving) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route(
            "/v1/responses",
            get(responses::responses_websocket).post(responses::responses),
        )
        .route("/anthropic/v1/messages", post(anthropic::anthropic_messages))
        .route(
            "/anthropic/v1/messages/count_tokens",
            post(anthropic::anthropic_count_tokens),
        )
        .route("/api/v1/chat/templates/apply", post(chat::apply_template))
        .route("/api/v1/models/{model_id}/properties", post(properties::props))
        .layer(DefaultBodyLimit::max(MAX_HTTP_BODY_BYTES))
        .with_state(serving)
}

pub(crate) fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

/// `Magnitude-Include-Progress: true` asks for loading and inference progress.
pub(crate) fn include_progress(headers: &HeaderMap) -> bool {
    headers
        .get("Magnitude-Include-Progress")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.eq_ignore_ascii_case("true"))
}

pub(crate) fn with_request_id(mut response: Response, request_id: &str) -> Response {
    if let Ok(value) = request_id.parse() {
        response.headers_mut().insert("x-request-id", value);
    }
    response
}
