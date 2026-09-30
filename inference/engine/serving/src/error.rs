//! Failure classification for every protocol. Each failure maps to one HTTP
//! status and one OpenAI-style error body (`{error: {message, type, param,
//! code}}`); Anthropic and WebSocket framings are derived from that body.
use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use magnitude_chat::{ChatError, output::JournalError};
use magnitude_engine::error::{LoadError, RequestError, UnloadCause, UnsupportedModel};
use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use utoipa::ToSchema;

static NEXT_HTTP_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

/// Why a served model could not be bound, as the model source reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelUnavailable {
    /// The model identity itself is malformed.
    InvalidModel(String),
    /// No served model has this name, or its material is not installed.
    NotFound(String),
    /// The model exists but cannot take requests yet. Retryable.
    NotReady(String),
    /// The model is occupied by an exclusive operation. Retryable.
    Busy(String),
    /// The source does not support this operation for the model.
    Unsupported(String),
    /// Installed material failed integrity validation.
    Integrity(String),
    /// The instance serving this request was stopped or replaced.
    InstanceStopped,
    /// A managed-model operation failed with a source-defined code.
    Operation {
        code: String,
        message: String,
        retryable: bool,
    },
    /// The source failed internally. Retryable.
    Internal(String),
}

/// Every failure a protocol handler can report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServingError {
    /// The protocol request is malformed or uses unsupported controls.
    InvalidRequest(String),
    /// The request cannot be prepared for the model.
    Chat(ChatError),
    /// The engine refused or ended the request.
    Request(RequestError),
    /// Loading the model for this request failed.
    Load(LoadError),
    /// The model could not be bound.
    Model(ModelUnavailable),
    /// The generated output violated the semantic stream contract.
    Output(JournalError),
    Internal(String),
}

impl std::fmt::Display for ServingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequest(message) | Self::Internal(message) => formatter.write_str(message),
            Self::Chat(error) => error.fmt(formatter),
            Self::Request(error) => error.fmt(formatter),
            Self::Load(error) => error.fmt(formatter),
            Self::Model(error) => match error {
                ModelUnavailable::InvalidModel(message)
                | ModelUnavailable::NotFound(message)
                | ModelUnavailable::NotReady(message)
                | ModelUnavailable::Busy(message)
                | ModelUnavailable::Unsupported(message)
                | ModelUnavailable::Integrity(message)
                | ModelUnavailable::Operation { message, .. }
                | ModelUnavailable::Internal(message) => formatter.write_str(message),
                ModelUnavailable::InstanceStopped => {
                    formatter.write_str("model instance was stopped")
                }
            },
            Self::Output(error) => write!(formatter, "model output was malformed: {error}"),
        }
    }
}

impl std::error::Error for ServingError {}

impl From<ChatError> for ServingError {
    fn from(error: ChatError) -> Self {
        Self::Chat(error)
    }
}

impl From<RequestError> for ServingError {
    fn from(error: RequestError) -> Self {
        Self::Request(error)
    }
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
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = true)]
    pub param: Option<String>,
    pub code: String,
    #[serde(skip)]
    #[schema(ignore)]
    pub retryable: bool,
}

/// A classified failure ready to frame.
#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: StatusCode,
    pub body: ApiErrorBody,
}

const INVALID: &str = "invalid_request_error";
const SERVER: &str = "server_error";
const MODEL: &str = "model_error";

impl ApiError {
    fn new(
        status: StatusCode,
        kind: &'static str,
        code: &str,
        retryable: bool,
        message: String,
    ) -> Self {
        Self {
            status,
            body: ApiErrorBody {
                message,
                r#type: kind,
                param: None,
                code: code.to_owned(),
                retryable,
            },
        }
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, INVALID, "invalid_request", false, message.into())
    }

    pub fn server(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            SERVER,
            "backend_error",
            true,
            message.into(),
        )
    }

    #[must_use]
    pub fn with_param(mut self, param: &'static str) -> Self {
        self.body.param = Some(param.to_owned());
        self
    }

    pub fn response(self) -> Response {
        let mut response = (self.status, Json(ErrorResponse { error: self.body })).into_response();
        let request_id = format!(
            "req_icn_{}",
            NEXT_HTTP_REQUEST_ID.fetch_add(1, Ordering::Relaxed)
        );
        if let Ok(value) = request_id.parse() {
            response.headers_mut().insert("x-request-id", value);
        }
        response
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        self.response()
    }
}

impl From<ServingError> for ApiError {
    fn from(error: ServingError) -> Self {
        let message = error.to_string();
        let (status, kind, code, retryable) = match &error {
            ServingError::InvalidRequest(_) => (StatusCode::BAD_REQUEST, INVALID, "invalid_request", false),
            ServingError::Chat(ChatError::ContextLengthExceeded { .. }) => (
                StatusCode::BAD_REQUEST,
                INVALID,
                "context_length_exceeded",
                false,
            ),
            ServingError::Chat(ChatError::Internal(_)) => {
                (StatusCode::INTERNAL_SERVER_ERROR, SERVER, "backend_error", true)
            }
            ServingError::Chat(ChatError::InvalidRequest(_) | ChatError::Reasoning(_)) => {
                (StatusCode::BAD_REQUEST, INVALID, "invalid_request", false)
            }
            ServingError::Request(error) => request_class(error),
            ServingError::Load(error) => load_class(error),
            ServingError::Model(error) => model_class(error),
            ServingError::Output(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, SERVER, "backend_error", true)
            }
            ServingError::Internal(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, SERVER, "backend_error", true)
            }
        };
        let owned;
        let code = match (&error, code) {
            (ServingError::Model(ModelUnavailable::Operation { code, .. }), _) => {
                owned = code.clone();
                owned.as_str()
            }
            (_, code) => code,
        };
        Self::new(status, kind, code, retryable, message)
    }
}

type Class = (StatusCode, &'static str, &'static str, bool);

/// Engine request outcomes (integration spec §9.6).
fn request_class(error: &RequestError) -> Class {
    let unavailable = StatusCode::SERVICE_UNAVAILABLE;
    match error {
        RequestError::InvalidRequest { .. } => (StatusCode::BAD_REQUEST, INVALID, "invalid_request", false),
        RequestError::ContextLengthExceeded { .. } => {
            (StatusCode::BAD_REQUEST, INVALID, "context_length_exceeded", false)
        }
        RequestError::InsufficientMemory(_) => (unavailable, SERVER, "insufficient_memory", true),
        RequestError::MemoryObservationUnavailable { .. } => {
            (unavailable, SERVER, "memory_observation_unavailable", true)
        }
        RequestError::MemoryReclaim => (unavailable, SERVER, "memory_pressure", true),
        RequestError::ModelUnloaded {
            cause: UnloadCause::MemoryPressure,
        } => (unavailable, SERVER, "model_unloaded", true),
        RequestError::ModelUnloaded {
            cause: UnloadCause::DeviceLost { .. },
        }
        | RequestError::DeviceLost { .. } => {
            (StatusCode::INTERNAL_SERVER_ERROR, SERVER, "device_lost", true)
        }
        RequestError::ModelUnloaded {
            cause: UnloadCause::Shutdown,
        } => (StatusCode::CONFLICT, MODEL, "model_instance_stopped", false),
        RequestError::ModelUnloaded {
            cause: UnloadCause::Internal { .. },
        } => (StatusCode::INTERNAL_SERVER_ERROR, SERVER, "worker_exited", true),
        RequestError::Overloaded => (unavailable, SERVER, "overloaded", true),
        RequestError::Cancelled => (StatusCode::INTERNAL_SERVER_ERROR, "cancelled", "request_cancelled", true),
        RequestError::WorkerLost { .. } => {
            (StatusCode::INTERNAL_SERVER_ERROR, SERVER, "worker_exited", true)
        }
        RequestError::Internal { .. } => {
            (StatusCode::INTERNAL_SERVER_ERROR, SERVER, "backend_error", true)
        }
    }
}

/// Load failures surfaced to a request that triggered the load.
fn load_class(error: &LoadError) -> Class {
    match error {
        LoadError::Unsupported(UnsupportedModel::Family { .. })
        | LoadError::Unsupported(UnsupportedModel::Representation { .. })
        | LoadError::Unsupported(UnsupportedModel::Backend { .. })
        | LoadError::Unsupported(UnsupportedModel::KernelDomain { .. }) => {
            (StatusCode::CONFLICT, MODEL, "unsupported_model", false)
        }
        LoadError::Artifact(_) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            INVALID,
            "integrity_failed",
            false,
        ),
        LoadError::InsufficientMemory { .. } => {
            (StatusCode::SERVICE_UNAVAILABLE, SERVER, "insufficient_memory", true)
        }
        LoadError::MemoryObservationUnavailable { .. } => (
            StatusCode::SERVICE_UNAVAILABLE,
            SERVER,
            "memory_observation_unavailable",
            true,
        ),
        LoadError::DeviceLost { .. } => {
            (StatusCode::INTERNAL_SERVER_ERROR, SERVER, "device_lost", true)
        }
        LoadError::Device(_) => (StatusCode::CONFLICT, MODEL, "device_unavailable", true),
        LoadError::WorkerLost { .. } => {
            (StatusCode::INTERNAL_SERVER_ERROR, SERVER, "worker_exited", true)
        }
        LoadError::Internal { .. } => {
            (StatusCode::INTERNAL_SERVER_ERROR, SERVER, "backend_error", true)
        }
    }
}

fn model_class(error: &ModelUnavailable) -> Class {
    match error {
        ModelUnavailable::InvalidModel(_) => (StatusCode::BAD_REQUEST, INVALID, "invalid_request", false),
        ModelUnavailable::NotFound(_) => (StatusCode::NOT_FOUND, INVALID, "model_not_found", false),
        ModelUnavailable::NotReady(_) => (StatusCode::CONFLICT, INVALID, "model_not_ready", true),
        ModelUnavailable::Busy(_) => (StatusCode::CONFLICT, INVALID, "model_busy", true),
        ModelUnavailable::Unsupported(_) => {
            (StatusCode::CONFLICT, INVALID, "operation_unsupported", false)
        }
        ModelUnavailable::Integrity(_) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            INVALID,
            "integrity_failed",
            false,
        ),
        ModelUnavailable::InstanceStopped => {
            (StatusCode::CONFLICT, MODEL, "model_instance_stopped", false)
        }
        ModelUnavailable::Operation { retryable, .. } => (
            StatusCode::CONFLICT,
            MODEL,
            // Replaced by the operation's own code.
            "model_error",
            *retryable,
        ),
        ModelUnavailable::Internal(_) => {
            (StatusCode::INTERNAL_SERVER_ERROR, SERVER, "inventory_error", true)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stopped_model_instance_has_a_non_retryable_error_contract() {
        let error = ApiError::from(ServingError::Model(ModelUnavailable::InstanceStopped));
        assert_eq!(error.status, StatusCode::CONFLICT);
        assert_eq!(error.body.code, "model_instance_stopped");
        assert!(!error.body.retryable);
    }

    #[test]
    fn engine_outcomes_use_the_integration_error_codes() {
        for (error, status, code) in [
            (
                RequestError::MemoryReclaim,
                StatusCode::SERVICE_UNAVAILABLE,
                "memory_pressure",
            ),
            (
                RequestError::Overloaded,
                StatusCode::SERVICE_UNAVAILABLE,
                "overloaded",
            ),
            (
                RequestError::ModelUnloaded {
                    cause: UnloadCause::MemoryPressure,
                },
                StatusCode::SERVICE_UNAVAILABLE,
                "model_unloaded",
            ),
            (
                RequestError::DeviceLost {
                    reason: "lost".into(),
                },
                StatusCode::INTERNAL_SERVER_ERROR,
                "device_lost",
            ),
        ] {
            let error = ApiError::from(ServingError::Request(error));
            assert_eq!((error.status, error.body.code.as_str()), (status, code));
            assert!(error.body.retryable);
        }
        let unsupported = ApiError::from(ServingError::Load(LoadError::Unsupported(
            UnsupportedModel::Family {
                reason: "unknown".into(),
            },
        )));
        assert_eq!(unsupported.status, StatusCode::CONFLICT);
        assert_eq!(unsupported.body.code, "unsupported_model");
        assert!(!unsupported.body.retryable);
    }
}
