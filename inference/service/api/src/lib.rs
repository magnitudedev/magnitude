use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::extract::{DefaultBodyLimit, Path, Query, Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::{future::BoxFuture, stream::BoxStream};
use magnitude_service_contracts::bootstrap_protocol::{
    IcnBinaryIdentity, IcnInstallationDeclaration, IcnParentCommand, IcnParentCommandType,
    IcnStartupBackend, IcnStartupProgressRecord, IcnStartupProgressRecordType, IcnStartupRecord,
    IcnStartupRecordType,
};
use magnitude_service_contracts::models::{
    CatalogInstallationAdmission, CatalogInstallationOperation, CatalogInstallationOperationId,
    CatalogInstallationRemoval, CatalogInstallations, CatalogInstallationsResponse, CatalogModel,
    CatalogModelState, CatalogModels, CatalogModelsResponse, DiscoveredModel, DiscoveredModelState,
    DiscoveredModels, DiscoveredModelsResponse, EffectiveModel, ModelAssessmentDomainSnapshot,
    ModelAssessmentEntryState, ModelAssessmentPoolState, ModelAssessmentSubject, ModelAssessments,
    ModelAssessmentsSnapshot, ModelCapabilities, ModelDownloads, ModelId, ModelInstance,
    ModelInstanceId, ModelInstancesInvalidation, ModelInstancesSnapshot, ModelLoadDevice,
    ModelLoadPlan, ParsedModelId,
};
use magnitude_service_contracts::{
    ExecutionBackend, HardwareProvider, HardwareSnapshot, HuggingFaceModelCatalog,
    HuggingFaceModelSearchRequest, HuggingFaceModelSearchResults, HuggingFaceRepositoryRequest,
    HuggingFaceRepositorySnapshot, InventoryError,
};
use serde::{Deserialize, Serialize};
use utoipa::openapi::extensions::Extensions;
use utoipa::openapi::path::Operation;
use utoipa::openapi::schema::{AdditionalProperties, Schema};
use utoipa::openapi::{Components, OpenApi as OpenApiDocument, RefOr};
use utoipa::{OpenApi, PartialSchema, ToSchema};

const CONNECTOR_MAX_OUTPUT_TOKENS: u32 = 32_768;

use magnitude_serving::chat::{
    AllowedToolRequest, AllowedToolsChoiceRequest, AllowedToolsModeRequest, AllowedToolsRequest,
    AllowedToolsType, ApplyTemplateRequest, ApplyTemplateResponse, ChatCompletionChoice,
    ChatCompletionChunk, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionStreamEvent, ChatContentPartRequest, ChatContentRequest, ChatMessageRequest,
    ChatToolCallRequest, ChatToolRequest, ChunkChoice, ChunkDelta, ChunkFunctionDelta,
    ChunkToolCall, CompletionFunctionCall, CompletionToolCall, FunctionDefinitionRequest,
    FunctionNameRequest, FunctionToolChoiceRequest, FunctionType, GrammarTriggerResponse,
    ImageUrlRequest, JsonSchemaRequest, NamedFunctionCallRequest, ReasoningEffortRequest,
    ResponseFormatRequest, StopRequest, StreamOptions, Timings, ToolChoiceModeRequest,
    ToolChoiceRequest, Usage,
};
use magnitude_serving::properties::{
    DefaultGenerationSettings, Modalities, PropsResponse, TemplateCapabilitiesResponse,
};
use magnitude_serving::responses::{ResponseCreateRequest, ResponseStreamEvent};
use magnitude_serving::{ApiErrorBody, ErrorResponse};

const STREAM_EXTENSION: &str = "x-magnitude-stream";
static NEXT_HTTP_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
pub struct AppState {
    catalog_models: Option<Arc<dyn CatalogModels>>,
    discovered_models: Option<Arc<dyn DiscoveredModels>>,
    catalog_installations: Option<Arc<dyn CatalogInstallations>>,
    model_assessments: Option<Arc<dyn ModelAssessments>>,
    model_downloads: Option<Arc<dyn ModelDownloads>>,
    hardware: Option<Arc<dyn HardwareProvider>>,
    hugging_face_catalog: Option<Arc<dyn HuggingFaceModelCatalog>>,
    model_controller: Option<Arc<dyn ModelInstanceController>>,
    identity: ServerIdentity,
    authorization: Option<Arc<str>>,
    resource_revision: Arc<AtomicU64>,
    resource_changes: tokio::sync::broadcast::Sender<InferenceResourceInvalidation>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct OpenAiModel {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
    pub name: String,
    pub description: String,
    pub context_length: u32,
    pub architecture: OpenAiModelArchitecture,
    pub supported_parameters: Vec<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<OpenAiModelReasoning>,
    pub top_provider: OpenAiTopProvider,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct OpenAiModelArchitecture {
    pub input_modalities: Vec<&'static str>,
    pub output_modalities: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct OpenAiModelReasoning {
    pub supported_efforts: Vec<String>,
    pub default_effort: String,
    pub default_enabled: bool,
    pub mandatory: bool,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct OpenAiTopProvider {
    pub context_length: u32,
    pub max_completion_tokens: u32,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct OpenAiModelsResponse {
    pub object: &'static str,
    pub data: Vec<OpenAiModel>,
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EnsureModelInstanceRequest {
    pub model_id: ModelId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ToSchema)]
#[serde(rename_all = "kebab-case")]
pub enum InferenceResourceTopic {
    Hardware,
    Catalog,
    Discovery,
    ModelAssessments,
    CatalogInstallations,
    Instances,
}

impl InferenceResourceTopic {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Hardware => "hardware",
            Self::Catalog => "catalog",
            Self::Discovery => "discovery",
            Self::ModelAssessments => "model-assessments",
            Self::CatalogInstallations => "catalog-installations",
            Self::Instances => "instances",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "hardware" => Some(Self::Hardware),
            "catalog" => Some(Self::Catalog),
            "discovery" => Some(Self::Discovery),
            "model-assessments" => Some(Self::ModelAssessments),
            "catalog-installations" => Some(Self::CatalogInstallations),
            "instances" => Some(Self::Instances),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct InferenceResourceInvalidation {
    pub topic: InferenceResourceTopic,
    pub revision: u64,
}

#[derive(Debug, Default, Deserialize)]
struct InferenceEventQuery {
    topics: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ServerIdentity {
    pub instance_id: String,
    pub api_version: u32,
    pub native_build: String,
}

impl Default for ServerIdentity {
    fn default() -> Self {
        Self {
            instance_id: "embedded".to_owned(),
            api_version: 1,
            native_build: "unknown".to_owned(),
        }
    }
}

impl AppState {
    /// Management state with no domain configured; the `with_*` builders configure each.
    pub fn new() -> Self {
        let (resource_changes, _) = tokio::sync::broadcast::channel(64);
        Self {
            catalog_models: None,
            discovered_models: None,
            catalog_installations: None,
            model_assessments: None,
            model_downloads: None,
            hardware: None,
            hugging_face_catalog: None,
            model_controller: None,
            identity: ServerIdentity::default(),
            authorization: None,
            resource_revision: Arc::new(AtomicU64::new(0)),
            resource_changes,
        }
    }

    pub fn with_model_domains(
        mut self,
        catalog_models: Arc<dyn CatalogModels>,
        discovered_models: Arc<dyn DiscoveredModels>,
        catalog_installations: Arc<dyn CatalogInstallations>,
    ) -> Self {
        self.catalog_models = Some(catalog_models);
        self.discovered_models = Some(discovered_models);
        self.catalog_installations = Some(catalog_installations);
        self
    }

    pub fn with_model_assessments(mut self, model_assessments: Arc<dyn ModelAssessments>) -> Self {
        self.model_assessments = Some(model_assessments);
        self
    }

    pub fn with_model_downloads(mut self, model_downloads: Arc<dyn ModelDownloads>) -> Self {
        self.model_downloads = Some(model_downloads);
        self
    }

    pub fn with_hardware(mut self, hardware: Arc<dyn HardwareProvider>) -> Self {
        self.hardware = Some(hardware);
        self
    }

    pub fn with_hugging_face_catalog(mut self, catalog: Arc<dyn HuggingFaceModelCatalog>) -> Self {
        self.hugging_face_catalog = Some(catalog);
        self
    }

    pub fn with_model_controller(mut self, controller: Arc<dyn ModelInstanceController>) -> Self {
        self.model_controller = Some(controller);
        self
    }

    pub fn with_identity(mut self, identity: ServerIdentity) -> Self {
        self.identity = identity;
        self
    }

    pub fn with_authorization(mut self, capability: impl Into<Arc<str>>) -> Self {
        self.authorization = Some(capability.into());
        self
    }

    fn invalidate_resources(&self, topics: impl IntoIterator<Item = InferenceResourceTopic>) {
        for topic in topics {
            let revision = self
                .resource_revision
                .fetch_add(1, Ordering::AcqRel)
                .saturating_add(1);
            let _ = self
                .resource_changes
                .send(InferenceResourceInvalidation { topic, revision });
        }
    }
}

/// The service router: management routes plus the engine's protocol routes
/// (`magnitude_serving`), both behind the owner capability.
pub fn app(state: AppState, serving: magnitude_serving::Serving) -> Router {
    let mut protected = Router::new()
        .route("/api/v1/hardware", get(hardware))
        .route("/api/v1/catalog/models", get(catalog_models))
        .route("/api/v1/catalog/models/{model_id}", get(catalog_model))
        .route(
            "/api/v1/catalog/models/{model_id}/install",
            post(install_catalog_model),
        )
        .route(
            "/api/v1/catalog/models/{model_id}/installation",
            axum::routing::delete(remove_catalog_model_installation),
        )
        .route("/api/v1/catalog/installations", get(catalog_installations))
        .route(
            "/api/v1/catalog/installations/{operation_id}",
            get(catalog_installation),
        )
        .route(
            "/api/v1/catalog/installations/{operation_id}/cancel",
            post(cancel_catalog_installation),
        )
        .route(
            "/api/v1/catalog/installations/{operation_id}/acknowledge-failure",
            post(acknowledge_catalog_installation_failure),
        )
        .route("/api/v1/discovery/models", get(discovered_models))
        .route("/api/v1/discovery/refresh", post(refresh_discovery))
        .route("/api/v1/model-assessments", get(model_assessments))
        .route(
            "/api/v1/models/{model_id}/load-plan",
            post(preview_model_load),
        )
        .route(
            "/api/v1/instances",
            get(model_instances).post(ensure_model_instance),
        )
        .route("/api/v1/instances/{instance_id}", get(model_instance))
        .route("/api/v1/events", get(watch_inference_events))
        .route(
            "/api/v1/instances/{instance_id}/stop",
            post(stop_model_instance),
        )
        .route(
            "/api/v1/sources/hugging-face/search",
            post(search_hugging_face_models),
        )
        .route(
            "/api/v1/sources/hugging-face/resolve",
            post(resolve_hugging_face_repository),
        )
        .route("/v1/models", get(standard_models))
        .route("/openapi.json", get(serve_openapi))
        .with_state(state.clone())
        .merge(magnitude_serving::router(serving));
    if let Some(capability) = state.authorization.clone() {
        protected = protected.route_layer(middleware::from_fn_with_state(capability, authorize));
    }
    Router::new()
        .route("/health", get(health))
        .with_state(state)
        .merge(protected)
        .layer(DefaultBodyLimit::max(magnitude_serving::MAX_HTTP_BODY_BYTES))
}

async fn authorize(State(capability): State<Arc<str>>, request: Request, next: Next) -> Response {
    let expected = format!("Bearer {capability}");
    let supplied = request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    let matches = supplied.len() == expected.len()
        && supplied
            .bytes()
            .zip(expected.bytes())
            .fold(0_u8, |difference, (left, right)| {
                difference | (left ^ right)
            })
            == 0;
    if matches {
        next.run(request).await
    } else {
        StatusCode::UNAUTHORIZED.into_response()
    }
}

async fn serve_openapi() -> Result<Json<OpenApiDocument>, ApiError> {
    openapi()
        .map(Json)
        .map_err(|error| ApiError::server(format!("OpenAPI export failed: {error}")))
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HealthResponse {
    status: &'static str,
    ready: bool,
    version: &'static str,
    api_version: u32,
    instance_id: String,
    native_build: String,
}

pub trait ModelInstanceController: Send + Sync + 'static {
    fn preview_load(
        &self,
        model_id: String,
    ) -> BoxFuture<'_, Result<ModelLoadPlan, InventoryError>>;
    fn ensure_resident(
        &self,
        model_id: String,
    ) -> BoxFuture<'_, Result<ModelInstance, InventoryError>>;
    fn stop_instance(
        &self,
        instance_id: ModelInstanceId,
    ) -> BoxFuture<'_, Result<(), InventoryError>>;
    fn instances(&self) -> BoxFuture<'_, Result<ModelInstancesSnapshot, InventoryError>>;
    fn watch_instances(&self) -> BoxStream<'static, ModelInstancesInvalidation>;
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
                    param: None,
                    code: "invalid_request".to_owned(),
                    retryable: false,
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
                    param: None,
                    code: "backend_error".to_owned(),
                    retryable: true,
                },
            },
        }
    }

    fn from_inventory(error: InventoryError) -> Self {
        let (status, error_type, code, retryable) = match &error {
            InventoryError::InvalidId(_) | InventoryError::InvalidRequest(_) => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "invalid_request",
                false,
            ),
            InventoryError::NotFound(_) => (
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                "model_not_found",
                false,
            ),
            InventoryError::NotReady(_) => (
                StatusCode::CONFLICT,
                "invalid_request_error",
                "model_not_ready",
                true,
            ),
            InventoryError::Busy(_) => (
                StatusCode::CONFLICT,
                "invalid_request_error",
                "model_busy",
                true,
            ),
            InventoryError::Loaded(_) => (
                StatusCode::CONFLICT,
                "invalid_request_error",
                "model_loaded",
                false,
            ),
            InventoryError::DeletionUnsafe(_) => (
                StatusCode::CONFLICT,
                "invalid_request_error",
                "deletion_unsafe",
                false,
            ),
            InventoryError::Unsupported(_) => (
                StatusCode::CONFLICT,
                "invalid_request_error",
                "operation_unsupported",
                false,
            ),
            InventoryError::Integrity(_) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_request_error",
                "integrity_failed",
                false,
            ),
            InventoryError::ModelOperation {
                code, retryable, ..
            } => (
                StatusCode::CONFLICT,
                "model_error",
                code.as_str(),
                *retryable,
            ),
            InventoryError::Io(_)
            | InventoryError::Upstream(_)
            | InventoryError::ConcurrentMutation(_)
            | InventoryError::Internal(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "inventory_error",
                true,
            ),
        };
        Self {
            status,
            body: ErrorResponse {
                error: ApiErrorBody {
                    message: error.to_string(),
                    r#type: error_type,
                    param: None,
                    code: code.to_owned(),
                    retryable,
                },
            },
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut response = (self.status, Json(self.body)).into_response();
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

#[utoipa::path(get, path = "/health", operation_id = "health", tag = "system", responses(
    (status = 200, description = "ICN is running", body = HealthResponse)
))]
async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        ready: true,
        version: env!("CARGO_PKG_VERSION"),
        api_version: state.identity.api_version,
        instance_id: state.identity.instance_id,
        native_build: state.identity.native_build,
    })
}

#[utoipa::path(post, path = "/api/v1/instances", operation_id = "ensureModelInstance", tag = "models",
    request_body(content = EnsureModelInstanceRequest, content_type = "application/json"),
    responses(
        (status = 200, description = "Current ready instance for the model", body = ModelInstance),
        (status = 400, description = "Invalid model identity or request", body = ErrorResponse),
        (status = 404, description = "Model is not installed", body = ErrorResponse),
        (status = 409, description = "Model cannot currently be admitted", body = ErrorResponse),
        (status = 422, description = "Installed model failed integrity validation", body = ErrorResponse),
        (status = 500, description = "Runtime control unavailable", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.model_instance.ensure", skip_all, err(Debug))]
async fn ensure_model_instance(
    State(state): State<AppState>,
    Json(request): Json<EnsureModelInstanceRequest>,
) -> Result<Json<ModelInstance>, ApiError> {
    let controller = state
        .model_controller
        .as_ref()
        .ok_or_else(|| ApiError::server("model control is not configured"))?;
    controller
        .ensure_resident(request.model_id.to_string())
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(post, path = "/api/v1/models/{model_id}/load-plan", operation_id = "previewModelLoad", tag = "models",
    params(("model_id" = String, Path, description = "Canonical model ID")),
    responses(
        (status = 200, description = "Plan ICN would select from current admission evidence", body = ModelLoadPlan),
        (status = 400, description = "Configuration cannot be resolved", body = ErrorResponse),
        (status = 404, description = "Model is not installed", body = ErrorResponse),
        (status = 409, description = "Model cannot currently be admitted", body = ErrorResponse),
        (status = 422, description = "Installed model failed integrity validation", body = ErrorResponse),
        (status = 500, description = "Load preview failed", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.model_load.preview", skip_all, err(Debug))]
async fn preview_model_load(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<ModelLoadPlan>, ApiError> {
    let model_id = model_id
        .parse::<ModelId>()
        .map_err(|error| ApiError::invalid(error.to_string()))?;
    let controller = state
        .model_controller
        .as_ref()
        .ok_or_else(|| ApiError::server("model control is not configured"))?;
    controller
        .preview_load(model_id.to_string())
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(post, path = "/api/v1/instances/{instance_id}/stop", operation_id = "stopModelInstance", tag = "models",
    params(("instance_id" = String, Path, description = "Model instance ID")),
    responses(
        (status = 204, description = "The exact model instance is stopped"),
        (status = 400, description = "Invalid instance identity or request", body = ErrorResponse),
        (status = 404, description = "Model instance not found", body = ErrorResponse),
        (status = 409, description = "Model instance cannot currently be stopped", body = ErrorResponse),
        (status = 500, description = "Model instance stop failed", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.model_instance.stop", skip_all, err(Debug))]
async fn stop_model_instance(
    State(state): State<AppState>,
    Path(instance_id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let controller = state
        .model_controller
        .as_ref()
        .ok_or_else(|| ApiError::server("model control is not configured"))?;
    controller
        .stop_instance(ModelInstanceId(instance_id))
        .await
        .map(|()| StatusCode::NO_CONTENT)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(get, path = "/api/v1/instances", operation_id = "getModelInstances", tag = "models",
    responses(
        (status = 200, description = "Authoritative native model instances", body = ModelInstancesSnapshot),
        (status = 500, description = "Runtime control unavailable", body = ErrorResponse)
    )
)]
async fn model_instances(
    State(state): State<AppState>,
) -> Result<Json<ModelInstancesSnapshot>, ApiError> {
    let controller = state
        .model_controller
        .as_ref()
        .ok_or_else(|| ApiError::server("model control is not configured"))?;
    controller
        .instances()
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(get, path = "/api/v1/instances/{instance_id}", operation_id = "getModelInstance", tag = "models",
    params(("instance_id" = String, Path, description = "Model instance ID")),
    responses(
        (status = 200, description = "Exact model instance", body = ModelInstance),
        (status = 404, description = "Model instance not found", body = ErrorResponse),
        (status = 500, description = "Runtime control unavailable", body = ErrorResponse)
    )
)]
async fn model_instance(
    State(state): State<AppState>,
    Path(instance_id): Path<String>,
) -> Result<Json<ModelInstance>, ApiError> {
    let controller = state
        .model_controller
        .as_ref()
        .ok_or_else(|| ApiError::server("model control is not configured"))?;
    controller
        .instances()
        .await
        .map_err(ApiError::from_inventory)?
        .instances
        .into_iter()
        .find(|instance| instance.id.0 == instance_id)
        .map(Json)
        .ok_or_else(|| ApiError::from_inventory(InventoryError::NotFound(instance_id)))
}

#[utoipa::path(get, path = "/api/v1/events", operation_id = "watchInferenceEvents", tag = "system",
    params(("topics" = Option<String>, Query, description = "Comma-separated inference resource topics")),
    responses(
        (status = 200, description = "Multiplexed native inference-resource invalidations", body = String, content_type = "text/event-stream"),
        (status = 400, description = "Invalid resource topic filter", body = ErrorResponse),
        (status = 500, description = "Runtime control unavailable", body = ErrorResponse)
    )
)]
async fn watch_inference_events(
    State(state): State<AppState>,
    Query(query): Query<InferenceEventQuery>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let selected = match query.topics {
        None => None,
        Some(topics) => {
            let values = topics
                .split(',')
                .map(str::trim)
                .filter(|topic| !topic.is_empty())
                .map(str::to_owned)
                .collect::<BTreeSet<_>>();
            if values.is_empty() {
                return Err(ApiError::invalid("topics must name at least one resource"));
            }
            for topic in &values {
                if InferenceResourceTopic::parse(topic).is_none() {
                    return Err(ApiError::invalid(format!(
                        "unknown inference resource topic: {topic}"
                    )));
                }
            }
            Some(values)
        }
    };
    let controller = state
        .model_controller
        .as_ref()
        .ok_or_else(|| ApiError::server("model control is not configured"))?;
    let installations = state
        .catalog_installations
        .as_ref()
        .ok_or_else(|| ApiError::server("catalog installations are not configured"))?;
    let catalog = state
        .catalog_models
        .as_ref()
        .ok_or_else(|| ApiError::server("catalog models are not configured"))?;
    let discovery = state
        .discovered_models
        .as_ref()
        .ok_or_else(|| ApiError::server("discovered models are not configured"))?;
    let assessments = state
        .model_assessments
        .as_ref()
        .ok_or_else(|| ApiError::server("model assessments are not configured"))?;
    let instance_events =
        futures_util::StreamExt::flat_map(controller.watch_instances(), |event| {
            futures_util::stream::iter(
                [
                    InferenceResourceTopic::Instances,
                    InferenceResourceTopic::Hardware,
                ]
                .map(|topic| InferenceResourceInvalidation {
                    topic,
                    revision: event.revision,
                }),
            )
        });
    let installation_events = futures_util::StreamExt::map(
        installations.watch_catalog_installations(),
        |event| InferenceResourceInvalidation {
            topic: InferenceResourceTopic::CatalogInstallations,
            revision: event.revision,
        },
    );
    let catalog_events = futures_util::StreamExt::map(catalog.watch_catalog(), |event| {
        InferenceResourceInvalidation {
            topic: InferenceResourceTopic::Catalog,
            revision: event.revision,
        }
    });
    let discovery_events = futures_util::StreamExt::map(discovery.watch_discovery(), |event| {
        InferenceResourceInvalidation {
            topic: InferenceResourceTopic::Discovery,
            revision: event.revision,
        }
    });
    let assessment_events =
        futures_util::StreamExt::map(assessments.watch(), |event| InferenceResourceInvalidation {
            topic: InferenceResourceTopic::ModelAssessments,
            revision: event.revision,
        });
    let direct_events = futures_util::stream::unfold(
        state.resource_changes.subscribe(),
        |mut receiver| async move {
            loop {
                match receiver.recv().await {
                    Ok(event) => return Some((event, receiver)),
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => return None,
                }
            }
        },
    );
    let events = futures_util::stream::select(
        futures_util::stream::select(
            futures_util::stream::select(
                futures_util::stream::select(instance_events, installation_events),
                catalog_events,
            ),
            discovery_events,
        ),
        futures_util::stream::select(assessment_events, direct_events),
    );
    let events = futures_util::StreamExt::filter(events, move |event| {
        std::future::ready(
            selected
                .as_ref()
                .is_none_or(|topics| topics.contains(event.topic.as_str())),
        )
    });
    let framed = tokio_stream::StreamExt::map(events, |invalidation| {
        Ok(Event::default().event("invalidation").data(
            serde_json::to_string(&invalidation).expect("inference invalidation is serializable"),
        ))
    });
    Ok(Sse::new(framed).keep_alive(KeepAlive::default()))
}

#[utoipa::path(get, path = "/api/v1/hardware", operation_id = "getHardware", tag = "system", responses(
    (status = 200, description = "Hardware visible to the pinned ICN process", body = HardwareSnapshot),
    (status = 500, description = "Hardware discovery failed", body = ErrorResponse)
))]
#[tracing::instrument(name = "icn.hardware.snapshot", skip_all, err(Debug))]
async fn hardware(State(state): State<AppState>) -> Result<Json<HardwareSnapshot>, ApiError> {
    let provider = state
        .hardware
        .as_ref()
        .ok_or_else(|| ApiError::server("hardware discovery is not configured"))?;
    provider
        .snapshot()
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(post, path = "/api/v1/sources/hugging-face/search", operation_id = "searchHuggingFaceModels", tag = "hugging-face",
    request_body(content = HuggingFaceModelSearchRequest, content_type = "application/json"),
    responses(
        (status = 200, description = "Live Hugging Face GGUF model search", body = HuggingFaceModelSearchResults),
        (status = 400, description = "Invalid search request", body = ErrorResponse),
        (status = 500, description = "Hugging Face search failed", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.hugging_face.search", skip_all, err(Debug))]
async fn search_hugging_face_models(
    State(state): State<AppState>,
    Json(request): Json<HuggingFaceModelSearchRequest>,
) -> Result<Json<HuggingFaceModelSearchResults>, ApiError> {
    let catalog = state
        .hugging_face_catalog
        .as_ref()
        .ok_or_else(|| ApiError::server("Hugging Face discovery is not configured"))?;
    catalog
        .search(request)
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(post, path = "/api/v1/sources/hugging-face/resolve", operation_id = "resolveHuggingFaceRepository", tag = "hugging-face",
    request_body(content = HuggingFaceRepositoryRequest, content_type = "application/json"),
    responses(
        (status = 200, description = "Immutable snapshot of the requested live Hugging Face repository", body = HuggingFaceRepositorySnapshot),
        (status = 400, description = "Invalid repository request", body = ErrorResponse),
        (status = 500, description = "Hugging Face resolution failed", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.hugging_face.resolve", skip_all, err(Debug))]
async fn resolve_hugging_face_repository(
    State(state): State<AppState>,
    Json(request): Json<HuggingFaceRepositoryRequest>,
) -> Result<Json<HuggingFaceRepositorySnapshot>, ApiError> {
    let catalog = state
        .hugging_face_catalog
        .as_ref()
        .ok_or_else(|| ApiError::server("Hugging Face discovery is not configured"))?;
    catalog
        .resolve(request)
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(get, path = "/api/v1/catalog/models", operation_id = "listCatalogModels", tag = "catalog",
    responses(
        (status = 200, description = "Catalog declarations and current managed local state", body = CatalogModelsResponse),
        (status = 500, description = "Catalog state unavailable", body = ErrorResponse)
    )
)]
async fn catalog_models(
    State(state): State<AppState>,
) -> Result<Json<CatalogModelsResponse>, ApiError> {
    state
        .catalog_models
        .as_ref()
        .ok_or_else(|| ApiError::server("catalog models are not configured"))?
        .list_catalog()
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(get, path = "/api/v1/catalog/models/{model_id}", operation_id = "getCatalogModel", tag = "catalog",
    params(("model_id" = String, Path, description = "Canonical catalog model ID")),
    responses(
        (status = 200, description = "Exact catalog model", body = CatalogModel),
        (status = 404, description = "Catalog model not found", body = ErrorResponse),
        (status = 500, description = "Catalog state unavailable", body = ErrorResponse)
    )
)]
async fn catalog_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<CatalogModel>, ApiError> {
    let parsed_id = model_id
        .parse::<ModelId>()
        .map_err(|error| ApiError::invalid(error.to_string()))?;
    state
        .catalog_models
        .as_ref()
        .ok_or_else(|| ApiError::server("catalog models are not configured"))?
        .list_catalog()
        .await
        .map_err(ApiError::from_inventory)?
        .models
        .into_iter()
        .find(|model| model.id == parsed_id)
        .map(Json)
        .ok_or_else(|| ApiError::from_inventory(InventoryError::NotFound(model_id)))
}

#[utoipa::path(post, path = "/api/v1/catalog/models/{model_id}/install", operation_id = "installCatalogModel", tag = "catalog",
    params(("model_id" = String, Path, description = "Canonical catalog model ID")),
    responses(
        (status = 200, description = "Catalog installation admission", body = CatalogInstallationAdmission),
        (status = 404, description = "Catalog model not found", body = ErrorResponse),
        (status = 409, description = "Installation cannot be admitted", body = ErrorResponse),
        (status = 500, description = "Installation failed", body = ErrorResponse)
    )
)]
async fn install_catalog_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<CatalogInstallationAdmission>, ApiError> {
    let parsed_id = model_id
        .parse::<ModelId>()
        .map_err(|error| ApiError::invalid(error.to_string()))?;
    if !matches!(parsed_id.parsed(), ParsedModelId::Catalog { .. }) {
        return Err(ApiError::invalid(
            "catalog installation requires a catalog model ID",
        ));
    }
    let result = state
        .catalog_models
        .as_ref()
        .ok_or_else(|| ApiError::server("catalog models are not configured"))?
        .install_catalog_model(&parsed_id)
        .await
        .map_err(ApiError::from_inventory)?;
    state.invalidate_resources([
        InferenceResourceTopic::Catalog,
        InferenceResourceTopic::CatalogInstallations,
    ]);
    Ok(Json(result))
}

#[utoipa::path(delete, path = "/api/v1/catalog/models/{model_id}/installation", operation_id = "removeCatalogModelInstallation", tag = "catalog",
    params(("model_id" = String, Path, description = "Canonical catalog model ID")),
    responses(
        (status = 200, description = "Managed catalog installation removal result", body = CatalogInstallationRemoval),
        (status = 404, description = "Catalog model not found", body = ErrorResponse),
        (status = 409, description = "Model is live", body = ErrorResponse),
        (status = 500, description = "Removal failed", body = ErrorResponse)
    )
)]
async fn remove_catalog_model_installation(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<CatalogInstallationRemoval>, ApiError> {
    let parsed_id = model_id
        .parse::<ModelId>()
        .map_err(|error| ApiError::invalid(error.to_string()))?;
    if !matches!(parsed_id.parsed(), ParsedModelId::Catalog { .. }) {
        return Err(ApiError::invalid(
            "catalog removal requires a catalog model ID",
        ));
    }
    let result = state
        .catalog_models
        .as_ref()
        .ok_or_else(|| ApiError::server("catalog models are not configured"))?
        .remove_catalog_model_installation(&parsed_id)
        .await
        .map_err(ApiError::from_inventory)?;
    state.invalidate_resources([InferenceResourceTopic::Catalog]);
    Ok(Json(result))
}

#[utoipa::path(get, path = "/api/v1/catalog/installations", operation_id = "listCatalogInstallations", tag = "catalog",
    responses((status = 200, description = "Managed catalog installation occurrences", body = CatalogInstallationsResponse))
)]
async fn catalog_installations(
    State(state): State<AppState>,
) -> Result<Json<CatalogInstallationsResponse>, ApiError> {
    state
        .catalog_installations
        .as_ref()
        .ok_or_else(|| ApiError::server("catalog installations are not configured"))?
        .list_catalog_installations()
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(get, path = "/api/v1/catalog/installations/{operation_id}", operation_id = "getCatalogInstallation", tag = "catalog",
    params(("operation_id" = String, Path)), responses((status = 200, body = CatalogInstallationOperation), (status = 404, body = ErrorResponse))
)]
async fn catalog_installation(
    State(state): State<AppState>,
    Path(operation_id): Path<String>,
) -> Result<Json<CatalogInstallationOperation>, ApiError> {
    state
        .catalog_installations
        .as_ref()
        .ok_or_else(|| ApiError::server("catalog installations are not configured"))?
        .list_catalog_installations()
        .await
        .map_err(ApiError::from_inventory)?
        .operations
        .into_iter()
        .find(|operation| operation.operation_id.0 == operation_id)
        .map(Json)
        .ok_or_else(|| ApiError::from_inventory(InventoryError::NotFound(operation_id)))
}

#[utoipa::path(post, path = "/api/v1/catalog/installations/{operation_id}/cancel", operation_id = "cancelCatalogInstallation", tag = "catalog",
    params(("operation_id" = String, Path)), responses((status = 200, body = CatalogInstallationOperation), (status = 404, body = ErrorResponse))
)]
async fn cancel_catalog_installation(
    State(state): State<AppState>,
    Path(operation_id): Path<String>,
) -> Result<Json<CatalogInstallationOperation>, ApiError> {
    let result = state
        .catalog_installations
        .as_ref()
        .ok_or_else(|| ApiError::server("catalog installations are not configured"))?
        .cancel_catalog_installation(&CatalogInstallationOperationId(operation_id))
        .await
        .map_err(ApiError::from_inventory)?;
    state.invalidate_resources([InferenceResourceTopic::CatalogInstallations]);
    Ok(Json(result))
}

#[utoipa::path(post, path = "/api/v1/catalog/installations/{operation_id}/acknowledge-failure", operation_id = "acknowledgeCatalogInstallationFailure", tag = "catalog",
    params(("operation_id" = String, Path)), responses((status = 200, body = CatalogInstallationOperation), (status = 404, body = ErrorResponse))
)]
async fn acknowledge_catalog_installation_failure(
    State(state): State<AppState>,
    Path(operation_id): Path<String>,
) -> Result<Json<CatalogInstallationOperation>, ApiError> {
    let result = state
        .catalog_installations
        .as_ref()
        .ok_or_else(|| ApiError::server("catalog installations are not configured"))?
        .acknowledge_catalog_installation_failure(&CatalogInstallationOperationId(operation_id))
        .await
        .map_err(ApiError::from_inventory)?;
    state.invalidate_resources([InferenceResourceTopic::CatalogInstallations]);
    Ok(Json(result))
}

#[utoipa::path(get, path = "/api/v1/discovery/models", operation_id = "listDiscoveredModels", tag = "discovery",
    responses((status = 200, description = "Current non-catalog discoveries", body = DiscoveredModelsResponse))
)]
async fn discovered_models(
    State(state): State<AppState>,
) -> Result<Json<DiscoveredModelsResponse>, ApiError> {
    state
        .discovered_models
        .as_ref()
        .ok_or_else(|| ApiError::server("model discovery is not configured"))?
        .list_discovered()
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(post, path = "/api/v1/discovery/refresh", operation_id = "refreshDiscoveredModels", tag = "discovery",
    responses((status = 200, description = "Refreshed discovery snapshot", body = DiscoveredModelsResponse))
)]
async fn refresh_discovery(
    State(state): State<AppState>,
) -> Result<Json<DiscoveredModelsResponse>, ApiError> {
    let result = state
        .discovered_models
        .as_ref()
        .ok_or_else(|| ApiError::server("model discovery is not configured"))?
        .refresh_discovery()
        .await
        .map_err(ApiError::from_inventory)?;
    state.invalidate_resources([InferenceResourceTopic::Discovery]);
    Ok(Json(result))
}

#[utoipa::path(get, path = "/api/v1/model-assessments", operation_id = "getModelAssessments", tag = "models",
    responses((status = 200, description = "Current automatic model-assessment pool", body = ModelAssessmentsSnapshot))
)]
async fn model_assessments(
    State(state): State<AppState>,
) -> Result<Json<ModelAssessmentsSnapshot>, ApiError> {
    state
        .model_assessments
        .as_ref()
        .ok_or_else(|| ApiError::server("model assessments are not configured"))?
        .snapshot()
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(get, path = "/v1/models", operation_id = "listServableModels", tag = "inference",
    responses(
        (status = 200, description = "Installed models available for inference", body = OpenAiModelsResponse),
        (status = 500, description = "Model discovery failed", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.inference.models.list", skip_all, err(Debug))]
async fn standard_models(
    State(state): State<AppState>,
) -> Result<Json<OpenAiModelsResponse>, ApiError> {
    let catalog = state
        .catalog_models
        .as_ref()
        .ok_or_else(|| ApiError::server("catalog models are not configured"))?
        .list_catalog()
        .await
        .map_err(ApiError::from_inventory)?;
    let discovered = state
        .discovered_models
        .as_ref()
        .ok_or_else(|| ApiError::server("model discovery is not configured"))?
        .list_discovered()
        .await
        .map_err(ApiError::from_inventory)?;
    let assessment_snapshot = state
        .model_assessments
        .as_ref()
        .ok_or_else(|| ApiError::server("model assessments are not configured"))?
        .snapshot()
        .await
        .map_err(ApiError::from_inventory)?;
    let assessed_capabilities = assessed_capabilities(&assessment_snapshot);
    let catalog_models = catalog.models.into_iter().filter_map(|model| {
        let CatalogModelState::Installed { effective, .. } = model.local_state else {
            return None;
        };
        let EffectiveModel::Ready { model: target } = effective else {
            return None;
        };
        let capabilities = assessed_capabilities.get(&ModelAssessmentSubject::Catalog {
            model_id: model.id.clone(),
            selection: magnitude_service_contracts::models::CatalogModelSelection::Effective,
        })?;
        let name = format!("{} ({})", model.display_name, model.variant_label);
        let id = model.id.to_string();
        Some(open_ai_model(
            id,
            "magnitude",
            name,
            model.description,
            target.profile.context_length,
            capabilities.clone(),
        ))
    });
    let discovered_models = discovered.models.into_iter().filter_map(|model| {
        let DiscoveredModel { id, state } = model;
        let DiscoveredModelState::Ready { model: target, .. } = state else {
            return None;
        };
        let capabilities = assessed_capabilities.get(&ModelAssessmentSubject::Discovery {
            model_id: id.clone(),
        })?;
        let ParsedModelId::HuggingFace {
            repository_id,
            artifact_selector,
        } = id.parsed()
        else {
            return None;
        };
        let display_name = std::path::Path::new(artifact_selector.as_str())
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or(artifact_selector.as_str())
            .to_owned();
        let repository = repository_id.as_str().to_owned();
        let id = id.to_string();
        Some(open_ai_model(
            id,
            "huggingface-cache",
            display_name,
            format!("Discovered in Hugging Face cache from {repository}"),
            target.profile.context_length,
            capabilities.clone(),
        ))
    });
    let data = catalog_models.chain(discovered_models).collect();
    Ok(Json(OpenAiModelsResponse {
        object: "list",
        data,
    }))
}

fn assessed_capabilities(
    snapshot: &ModelAssessmentsSnapshot,
) -> BTreeMap<ModelAssessmentSubject, ModelCapabilities> {
    let ModelAssessmentPoolState::Ready {
        catalog,
        discovered,
        ..
    } = &snapshot.state
    else {
        return BTreeMap::new();
    };
    [catalog, discovered]
        .into_iter()
        .filter_map(|domain| match domain {
            ModelAssessmentDomainSnapshot::Available { entries, .. } => Some(entries),
            ModelAssessmentDomainSnapshot::Pending { .. }
            | ModelAssessmentDomainSnapshot::Failed { .. } => None,
        })
        .flatten()
        .filter_map(|entry| match &entry.state {
            ModelAssessmentEntryState::Assessed { capabilities, .. } => {
                Some((entry.subject.clone(), capabilities.clone()))
            }
            ModelAssessmentEntryState::Assessing | ModelAssessmentEntryState::Dropped => None,
        })
        .collect()
}

fn open_ai_model(
    id: String,
    owned_by: &'static str,
    name: String,
    description: String,
    context_length: u32,
    capabilities: magnitude_service_contracts::models::ModelCapabilities,
) -> OpenAiModel {
    let mut supported_parameters = vec!["max_tokens"];
    if capabilities.tools {
        supported_parameters.extend(["tools", "tool_choice"]);
    }
    if capabilities.structured_output {
        supported_parameters.extend(["structured_outputs", "response_format"]);
    }
    let reasoning = capabilities.reasoning.supported.then(|| {
        supported_parameters.push("reasoning");
        let default_effort = capabilities
            .reasoning
            .default_effort
            .clone()
            .expect("assessed reasoning capabilities must include a default effort");
        OpenAiModelReasoning {
            default_enabled: default_effort != "none",
            mandatory: !capabilities
                .reasoning
                .efforts
                .iter()
                .any(|effort| effort == "none"),
            supported_efforts: capabilities.reasoning.efforts,
            default_effort,
        }
    });
    let mut input_modalities = vec!["text"];
    if capabilities.vision {
        input_modalities.push("image");
    }
    OpenAiModel {
        id,
        object: "model",
        created: 0,
        owned_by,
        name,
        description,
        context_length,
        architecture: OpenAiModelArchitecture {
            input_modalities,
            output_modalities: vec!["text"],
        },
        supported_parameters,
        reasoning,
        top_provider: OpenAiTopProvider {
            context_length,
            max_completion_tokens: context_length.min(CONNECTOR_MAX_OUTPUT_TOKENS),
        },
    }
}

#[derive(OpenApi)]
#[openapi(
    info(title = "Magnitude Inference Control Node", version = "0.1.0"),
    paths(
        health,
        hardware,
        catalog_models,
        catalog_model,
        install_catalog_model,
        remove_catalog_model_installation,
        catalog_installations,
        catalog_installation,
        cancel_catalog_installation,
        acknowledge_catalog_installation_failure,
        discovered_models,
        refresh_discovery,
        model_assessments,
        standard_models,
        search_hugging_face_models,
        resolve_hugging_face_repository,
        preview_model_load,
        ensure_model_instance,
        model_instances,
        model_instance,
        watch_inference_events,
        stop_model_instance,
        magnitude_serving::properties::props,
        magnitude_serving::chat::apply_template,
        magnitude_serving::chat::chat_completions,
        magnitude_serving::responses::responses,
        magnitude_serving::anthropic::anthropic_messages,
        magnitude_serving::anthropic::anthropic_count_tokens
    ),
    components(schemas(
        HealthResponse,
        HardwareSnapshot,
        CatalogModelsResponse,
        DiscoveredModelsResponse,
        ModelAssessmentsSnapshot,
        CatalogInstallationsResponse,
        CatalogInstallationAdmission,
        CatalogInstallationRemoval,
        EnsureModelInstanceRequest,
        OpenAiModel,
        OpenAiModelArchitecture,
        OpenAiModelReasoning,
        OpenAiTopProvider,
        OpenAiModelsResponse,
        HuggingFaceModelSearchRequest,
        HuggingFaceModelSearchResults,
        HuggingFaceRepositoryRequest,
        HuggingFaceRepositorySnapshot,
        ModelLoadPlan,
        ModelInstancesSnapshot,
        ModelInstancesInvalidation,
        InferenceResourceInvalidation,
        PropsResponse,
        DefaultGenerationSettings,
        Modalities,
        TemplateCapabilitiesResponse,
        ApplyTemplateRequest,
        ApplyTemplateResponse,
        GrammarTriggerResponse,
        ChatCompletionRequest,
        ResponseCreateRequest,
        magnitude_serving::responses::ResponseObject,
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
        ChatCompletionResponse,
        ChatCompletionChoice,
        ChatCompletionMessage,
        CompletionToolCall,
        CompletionFunctionCall,
        ChunkChoice,
        ChunkDelta,
        ChunkToolCall,
        ChunkFunctionDelta,
        Usage,
        Timings,
        ErrorResponse,
        ApiErrorBody,
        IcnBinaryIdentity,
        IcnStartupRecord,
        IcnStartupRecordType,
        IcnStartupProgressRecord,
        IcnParentCommand,
        IcnParentCommandType,
        IcnStartupProgressRecordType,
        IcnStartupBackend,
        IcnInstallationDeclaration,
        ExecutionBackend,
        ModelLoadDevice
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
    type Event = ChatCompletionStreamEvent;
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

struct ResponsesStream;

impl StreamContract for ResponsesStream {
    type Event = ResponseStreamEvent;
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
            termination: StreamTermination::Eof,
            reconnect: StreamReconnect::None,
        }
    }
}

struct InferenceEventsStream;

impl StreamContract for InferenceEventsStream {
    type Event = InferenceResourceInvalidation;
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
            termination: StreamTermination::LongLived,
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
    preserve_typed_request_client_contract(&mut document);
    attach_stream_contract::<ChatCompletionStream>(
        &mut document,
        "createChatCompletion",
        "text/event-stream",
    )?;
    attach_stream_contract::<ResponsesStream>(
        &mut document,
        "createResponse",
        "text/event-stream",
    )?;
    attach_stream_contract::<InferenceEventsStream>(
        &mut document,
        "watchInferenceEvents",
        "text/event-stream",
    )?;
    Ok(document)
}

fn preserve_typed_request_client_contract(document: &mut OpenApiDocument) {
    const TYPED_REQUEST_SCHEMAS: [&str; 25] = [
        "AllowedToolRequest",
        "AllowedToolsChoiceRequest",
        "AllowedToolsRequest",
        "ChatCompletionRequest",
        "ChatToolCallRequest",
        "ChatToolRequest",
        "CountTokensRequest",
        "FunctionDefinitionRequest",
        "FunctionNameRequest",
        "FunctionToolChoiceRequest",
        "ImageUrlRequest",
        "JsonSchemaRequest",
        "Message",
        "MessagesRequest",
        "NamedFunctionCallRequest",
        "ResponseCreateRequest",
        "ResponseFunctionCall",
        "ResponseFunctionCallOutput",
        "ResponseFunctionTool",
        "ResponseInputMessage",
        "ResponseReasoning",
        "ResponseReasoningInput",
        "ResponseText",
        "StreamOptions",
        "Tool",
    ];
    let Some(components) = document.components.as_mut() else {
        return;
    };
    for name in TYPED_REQUEST_SCHEMAS {
        if let Some(RefOr::T(Schema::Object(schema))) = components.schemas.get_mut(name) {
            schema.additional_properties = Some(Box::new(AdditionalProperties::FreeForm(false)));
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use magnitude_service_contracts::models::{
        ModelAssessmentsInvalidation, ModelInstanceAllocation, ModelInstanceLifecycle,
    };
    use magnitude_serving::{
        HostChat, LoadProgress, ModelInvocation, ModelUnavailable, ServedModels, Serving,
        ServingError,
    };
    use serde_json::{Value, json};
    use tower::ServiceExt;

    /// A model source with no models: protocol routes answer `model_not_found`.
    struct NoModels;

    impl ServedModels for NoModels {
        fn invoke(
            &self,
            model: &str,
            _progress: Option<LoadProgress>,
        ) -> BoxFuture<'_, Result<Box<dyn ModelInvocation>, ServingError>> {
            let model = model.to_owned();
            Box::pin(async move { Err(ServingError::Model(ModelUnavailable::NotFound(model))) })
        }

        fn host(&self, model: &str) -> BoxFuture<'_, Result<Arc<dyn HostChat>, ServingError>> {
            let model = model.to_owned();
            Box::pin(async move { Err(ServingError::Model(ModelUnavailable::NotFound(model))) })
        }
    }

    fn test_app(state: AppState) -> Router {
        app(state, Serving::new(Arc::new(NoModels)))
    }

    #[test]
    fn openai_model_discovery_projects_harness_metadata_into_data() {
        let model = open_ai_model(
            "local/model".to_owned(),
            "magnitude",
            "Local Model".to_owned(),
            "Local fixture.".to_owned(),
            65_536,
            magnitude_service_contracts::models::ModelCapabilities {
                vision: true,
                tools: true,
                structured_output: true,
                reasoning: magnitude_service_contracts::models::ModelReasoningCapabilities {
                    supported: true,
                    efforts: vec!["none".to_owned(), "high".to_owned()],
                    default_effort: Some("high".to_owned()),
                },
            },
        );

        let value = serde_json::to_value(model).expect("serializable model");
        assert_eq!(value["context_length"], 65_536);
        assert_eq!(value["top_provider"]["max_completion_tokens"], 32_768);
        assert_eq!(
            value["architecture"]["input_modalities"],
            json!(["text", "image"])
        );
        assert_eq!(
            value["reasoning"]["supported_efforts"],
            json!(["none", "high"])
        );
        assert_eq!(value["reasoning"]["default_effort"], "high");
        assert_eq!(value["reasoning"]["mandatory"], false);
        assert!(
            value["supported_parameters"]
                .as_array()
                .expect("parameters")
                .contains(&json!("reasoning"))
        );
    }

    #[tokio::test]
    async fn direct_resource_invalidations_are_published_with_monotonic_revisions() {
        let state = AppState::new();
        let mut changes = state.resource_changes.subscribe();

        state.invalidate_resources([
            InferenceResourceTopic::Catalog,
            InferenceResourceTopic::Catalog,
        ]);

        let models = changes.recv().await.expect("models invalidation");
        let packages = changes.recv().await.expect("packages invalidation");
        assert_eq!(models.topic, InferenceResourceTopic::Catalog);
        assert_eq!(packages.topic, InferenceResourceTopic::Catalog);
        assert!(packages.revision > models.revision);
    }

    #[tokio::test]
    async fn invalid_inference_event_topics_are_rejected_before_opening_a_stream() {
        for target in ["/api/v1/events?topics=unknown", "/api/v1/events?topics="] {
            let response = test_app(AppState::new())
                .oneshot(Request::get(target).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }
    }

    struct StubHardware;

    struct StubModelAssessments(ModelAssessmentsSnapshot);

    impl ModelAssessments for StubModelAssessments {
        fn snapshot(&self) -> BoxFuture<'_, Result<ModelAssessmentsSnapshot, InventoryError>> {
            Box::pin(async { Ok(self.0.clone()) })
        }

        fn watch(&self) -> BoxStream<'static, ModelAssessmentsInvalidation> {
            Box::pin(futures_util::stream::empty())
        }
    }

    struct StubHuggingFaceCatalog;

    impl HuggingFaceModelCatalog for StubHuggingFaceCatalog {
        fn search(
            &self,
            request: HuggingFaceModelSearchRequest,
        ) -> BoxFuture<'_, Result<HuggingFaceModelSearchResults, InventoryError>> {
            Box::pin(async move {
                Ok(HuggingFaceModelSearchResults {
                    models: vec![magnitude_service_contracts::HuggingFaceModelSearchResult {
                        repository: format!("owner/{}", request.query),
                        commit: "a".repeat(40),
                        last_modified: None,
                        downloads: Some(10),
                        likes: Some(2),
                        gated: false,
                        private: false,
                        tags: vec!["gguf".to_owned()],
                    }],
                })
            })
        }

        fn resolve(
            &self,
            request: HuggingFaceRepositoryRequest,
        ) -> BoxFuture<'_, Result<HuggingFaceRepositorySnapshot, InventoryError>> {
            Box::pin(async move {
                Ok(HuggingFaceRepositorySnapshot {
                    repository: request.repository,
                    commit: "b".repeat(40),
                    last_modified: None,
                    downloads: None,
                    likes: None,
                    gated: false,
                    private: false,
                    license: Some("apache-2.0".to_owned()),
                    license_url: None,
                    base_models: Vec::new(),
                    tags: vec!["gguf".to_owned()],
                    gguf_files: vec![magnitude_service_contracts::HuggingFaceRepositoryFile {
                        path: "model.gguf".into(),
                        size_bytes: 123,
                        content: magnitude_service_contracts::ContentIdentity::Sha256 {
                            value: "c".repeat(64),
                        },
                    }],
                })
            })
        }
    }

    impl HardwareProvider for StubHardware {
        fn snapshot(
            &self,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<HardwareSnapshot, InventoryError>>
                    + Send
                    + '_,
            >,
        > {
            Box::pin(async {
                serde_json::from_value(json!({
                    "captured_at": 10,
                    "platform": "test",
                    "architecture": "test64",
                    "cpu_model": "Test CPU",
                    "logical_cores": 8,
                    "system_memory": {
                        "physical_capacity_bytes": 1024,
                        "physical_available_bytes": 512,
                        "allocation_capacity_bytes": 1024,
                        "allocation_headroom_bytes": 512,
                        "assess_reserve_bytes": 128,
                        "abort_reserve_bytes": 64
                    },
                    "native_build": "test-build",
                    "enabled_backends": ["cpu"],
                    "topology_fingerprint": "topology",
                    "memory_domains": [{
                        "id": "system",
                        "kind": "system",
                        "total_capacity_bytes": 1024,
                        "stable_capacity_bytes": 768,
                        "current_free_bytes": 512,
                        "shares_system_memory": true,
                        "devices": [{
                            "id": "cpu",
                            "backend": "cpu",
                            "name": "CPU",
                            "description": "Test CPU",
                            "kind": "cpu",
                            "memory_limit": null
                        }]
                    }]
                }))
                .map_err(|error| InventoryError::Internal(error.to_string()))
            })
        }
    }

    #[tokio::test]
    async fn hardware_endpoint_returns_the_provider_snapshot() {
        let response = test_app(AppState::new().with_hardware(Arc::new(StubHardware)))
            .oneshot(
                Request::get("/api/v1/hardware")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(body["topology_fingerprint"], "topology");
        assert_eq!(body["memory_domains"][0]["stable_capacity_bytes"], 768);
    }

    #[tokio::test]
    async fn hugging_face_endpoints_expose_live_search_and_immutable_resolution() {
        let state = AppState::new().with_hugging_face_catalog(Arc::new(StubHuggingFaceCatalog));
        let search = test_app(state.clone())
            .oneshot(
                Request::post("/api/v1/sources/hugging-face/search")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({ "query": "model", "limit": 5 }).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(search.status(), StatusCode::OK);
        let search_body: Value =
            serde_json::from_slice(&search.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(search_body["models"][0]["repository"], "owner/model");

        let resolve = test_app(state)
            .oneshot(
                Request::post("/api/v1/sources/hugging-face/resolve")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({ "repository": "owner/model", "revision": "main" }).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resolve.status(), StatusCode::OK);
        let resolve_body: Value =
            serde_json::from_slice(&resolve.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(resolve_body["commit"], "b".repeat(40));
        assert_eq!(resolve_body["gguf_files"][0]["size_bytes"], 123);
    }

    struct StubModelInstanceController;

    impl ModelInstanceController for StubModelInstanceController {
        fn preview_load(
            &self,
            _model_id: String,
        ) -> BoxFuture<'_, Result<ModelLoadPlan, InventoryError>> {
            Box::pin(async {
                Err(InventoryError::Unsupported(
                    "model load preview is unavailable in the stub model controller".to_owned(),
                ))
            })
        }

        fn ensure_resident(
            &self,
            _model_id: String,
        ) -> BoxFuture<'_, Result<ModelInstance, InventoryError>> {
            Box::pin(async move {
                Ok(self
                    .instances()
                    .await?
                    .instances
                    .remove(0))
            })
        }

        fn stop_instance(
            &self,
            _instance_id: ModelInstanceId,
        ) -> BoxFuture<'_, Result<(), InventoryError>> {
            Box::pin(async { Ok(()) })
        }

        fn instances(&self) -> BoxFuture<'_, Result<ModelInstancesSnapshot, InventoryError>> {
            Box::pin(async {
                Ok(ModelInstancesSnapshot {
                    revision: 0,
                    instances: vec![ModelInstance {
                        id: ModelInstanceId("test-instance".to_owned()),
                        model_id: "test-model:gguf:f16".parse().unwrap(),
                        lifecycle: ModelInstanceLifecycle::Ready {
                            allocation: ModelInstanceAllocation {
                                context_window_tokens: 1,
                                memory_domains: Vec::new(),
                            },
                        },
                    }],
                })
            })
        }

        fn watch_instances(&self) -> BoxStream<'static, ModelInstancesInvalidation> {
            Box::pin(futures_util::stream::empty())
        }
    }

    #[tokio::test]
    async fn exposes_the_authoritative_model_instances_snapshot() {
        let response = test_app(
            AppState::new().with_model_controller(Arc::new(StubModelInstanceController)),
        )
            .oneshot(
                Request::get("/api/v1/instances")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body: Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(body["revision"], 0);
        assert_eq!(body["instances"][0]["id"], "test-instance");
        assert_eq!(body["instances"][0]["lifecycle"]["_tag"], "Ready");
    }

    #[tokio::test]
    async fn exposes_the_automatic_model_assessment_snapshot() {
        let assessments = Arc::new(StubModelAssessments(ModelAssessmentsSnapshot {
            revision: 7,
            state: magnitude_service_contracts::models::ModelAssessmentPoolState::Preparing,
        }));
        let response = test_app(AppState::new().with_model_assessments(assessments))
            .oneshot(
                Request::get("/api/v1/model-assessments")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body: Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(body["revision"], 7);
        assert_eq!(body["state"]["_tag"], "Preparing");
    }

    #[test]
    fn exported_chat_operation_has_explicit_stream_contract() {
        let value = serde_json::to_value(openapi().unwrap()).unwrap();
        let contract = &value["paths"]["/v1/chat/completions"]["post"][STREAM_EXTENSION];
        assert_eq!(contract["framing"], "sse");
        assert_eq!(
            contract["data"]["schema"]["$ref"],
            "#/components/schemas/ChatCompletionStreamEvent"
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
        assert!(
            schemas["ChatCompletionChunk"]["properties"]
                .get("error")
                .is_none()
        );
        assert!(schemas["ChatCompletionChunk"]["properties"]["timings"].is_object());
        assert_eq!(
            schemas["ChatCompletionRequest"]["properties"]["timings_per_token"]["type"],
            "boolean"
        );
        assert_eq!(
            schemas["ChatCompletionRequest"]["properties"]["timings_per_token"]["default"],
            false
        );
        for field in [
            "cache_n",
            "prompt_n",
            "prompt_ms",
            "prompt_per_token_ms",
            "prompt_per_second",
            "predicted_n",
            "predicted_ms",
            "predicted_per_token_ms",
            "predicted_per_second",
            "sampler_ms",
            "parser_ms",
        ] {
            assert!(schemas["Timings"]["properties"][field].is_object());
        }
    }

    #[test]
    fn exported_invalidation_stream_requires_snapshot_refresh_after_reconnect() {
        let value = serde_json::to_value(openapi().unwrap()).unwrap();
        let contract = &value["paths"]["/api/v1/events"]["get"][STREAM_EXTENSION];
        assert_eq!(contract["termination"]["type"], "long-lived");
        assert_eq!(contract["reconnect"]["type"], "none");
    }

    #[test]
    fn exported_model_admission_operations_declare_every_inventory_error_status() {
        let value = serde_json::to_value(openapi().unwrap()).unwrap();
        let statuses = ["400", "404", "409", "422", "500"];
        for (path, host_only) in [
            ("/api/v1/instances", false),
            ("/v1/chat/completions", false),
            ("/v1/responses", false),
            // Host-only operations resolve installed material and never admit an instance; a
            // resolution failure (unsupported model, invalid package) is their conflict.
            ("/api/v1/models/{model_id}/properties", true),
            ("/api/v1/chat/templates/apply", true),
            ("/anthropic/v1/messages/count_tokens", true),
        ] {
            let responses = &value["paths"][path]["post"]["responses"];
            for status in statuses {
                assert!(
                    responses[status].is_object(),
                    "post {path} must declare inventory error status {status}",
                );
            }
            let conflict = responses["409"]["description"].as_str().unwrap();
            assert!(
                !(host_only && conflict.contains("admitted")),
                "host-only post {path} declares an admission conflict {conflict:?}"
            );
        }
    }

    #[tokio::test]
    async fn private_routes_require_the_owner_capability_but_health_does_not() {
        let service = test_app(
            AppState::new()
                .with_model_controller(Arc::new(StubModelInstanceController))
                .with_authorization("private-capability"),
        );
        let health = service
            .clone()
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::OK);

        for (method, path) in [
            ("GET", "/api/v1/instances"),
            ("POST", "/api/v1/models/test-model/properties"),
            ("POST", "/v1/chat/completions"),
        ] {
            let denied = service
                .clone()
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(path)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(denied.status(), StatusCode::UNAUTHORIZED, "{method} {path}");
        }

        let allowed = service
            .clone()
            .oneshot(
                Request::get("/api/v1/instances")
                    .header("authorization", "Bearer private-capability")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(allowed.status(), StatusCode::OK);

        // Protocol routes reach the model source once authorized.
        let protocol = service
            .oneshot(
                Request::post("/api/v1/models/test-model/properties")
                    .header("authorization", "Bearer private-capability")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"test-model"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(protocol.status(), StatusCode::NOT_FOUND);
    }

}
