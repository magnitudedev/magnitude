//! The one implementation of the inference protocols: OpenAI Chat
//! Completions, OpenAI Responses (HTTP, SSE and WebSocket), Anthropic
//! Messages with `count_tokens` (integration spec §8.2), and the decision
//! formats System One and OpenAI Decisions. Chat routes serve chat models
//! and decision routes decision models; each refuses the other kind.
//! Handlers are router-independent and reach models only through
//! [`ServedModels`]; the service and the standalone engine each mount
//! [`router`] with their own source.
pub mod anthropic;
pub mod chat;
pub mod decisions;
pub mod engine;
pub mod error;
mod media;
pub mod properties;
pub mod responses;
mod sampling;
pub mod source;
pub mod systemone;
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
pub use magnitude_artifacts::Serves;
use magnitude_decision::{DecisionAnswers, DecisionRequest};

pub use error::{ApiError, ApiErrorBody, ErrorResponse, ModelUnavailable, ServingError};
pub use media::MAX_HTTP_BODY_BYTES;
pub use source::{
    ChatInvocation, ChatTarget, DecisionInvocation, DecisionTarget, GenerationEvent,
    GenerationStream, HostChat, LoadProgress, ModelLoadProgress, ModelLoadStage, ResolvedModel,
    ServedModels, ServedTarget,
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

    /// Bind the chat model `model` names: a model that serves decisions is
    /// refused before it is bound.
    pub(crate) async fn invoke(
        &self,
        model: &str,
        progress: Option<LoadProgress>,
    ) -> Result<Box<dyn ChatInvocation>, ServingError> {
        match self.models.resolve(model).await?.target {
            ServedTarget::Chat(target) => target.invoke(progress).await,
            ServedTarget::Decisions(_) => Err(unsupported(model, Serves::Decisions, Serves::Chat)),
        }
    }

    /// The host chat semantics of the chat model `model` names.
    pub(crate) async fn host(&self, model: &str) -> Result<Arc<dyn HostChat>, ServingError> {
        let resolved = self.models.resolve(model).await?;
        match resolved.target {
            ServedTarget::Chat(_) => Ok(resolved.host),
            ServedTarget::Decisions(_) => Err(unsupported(model, Serves::Decisions, Serves::Chat)),
        }
    }

    /// Answer `request` with the decision model `model` names: a chat model
    /// is refused before it is bound.
    pub(crate) async fn decide(
        &self,
        model: &str,
        request: DecisionRequest,
    ) -> Result<DecisionAnswers, ServingError> {
        match self.models.resolve(model).await?.target {
            ServedTarget::Decisions(target) => target.invoke().await?.decide(request).await,
            ServedTarget::Chat(_) => Err(unsupported(model, Serves::Chat, Serves::Decisions)),
        }
    }
}

/// The unsupported-model outcome for a model that serves `serves` on a
/// route for `route`.
fn unsupported(model: &str, serves: Serves, route: Serves) -> ServingError {
    ServingError::Model(ModelUnavailable::Unsupported(format!(
        "model `{model}` serves {}, not {}",
        serves_name(serves),
        serves_name(route)
    )))
}

fn serves_name(serves: Serves) -> &'static str {
    match serves {
        Serves::Chat => "chat",
        Serves::Decisions => "decisions",
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
        .route("/v1/systemone", post(systemone::systemone))
        .route("/v1/decisions", post(decisions::decisions))
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
